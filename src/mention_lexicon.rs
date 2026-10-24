//! 可生长的提及词表（Mention Lexicon）。
//!
//! 图谱实体、确认别名与显式准入的词面是「已经学会怎么读」的证据。本模块把
//! 这些证据收拢成一个轻量覆盖层，供分词阶段把长实体保留为单一 token：
//!
//! - `Confirmed`：参与切分覆盖层（图谱实体、确认别名、显式准入）；
//! - `Pending`：只提名，不参与切分，也不写图谱；
//! - `Tombstoned`：用户拒绝后永久挡板，既不再提名，也不再覆盖切分。
//!
//! 覆盖层不做贪心最长匹配，而是在字符格上做一次「覆盖字数最大、指称数最少」
//! 的动态规划选路，避免短词把长实体切碎。所有证据都带来源与观察次数，报告
//! 按词面排序，便于复核。

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

/// 词表条目状态。墓碑优先于确认：拒绝过的词面不会被后续观察悄悄复活。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexiconStatus {
    Confirmed,
    Pending,
    Tombstoned,
}

impl LexiconStatus {
    pub fn code(self) -> &'static str {
        match self {
            LexiconStatus::Confirmed => "confirmed",
            LexiconStatus::Pending => "pending",
            LexiconStatus::Tombstoned => "tombstoned",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LexiconStatus::Confirmed => "已确认",
            LexiconStatus::Pending => "待确认",
            LexiconStatus::Tombstoned => "已拒绝",
        }
    }
}

/// 一个词面条目：词面 → 规范名/类型/词性，附带来源与观察次数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexiconEntry {
    pub surface: String,
    pub canonical: String,
    pub pos: String,
    pub etype: Option<String>,
    pub status: LexiconStatus,
    pub origin: String,
    pub observations: u32,
    /// 同一词面被确认到其他规范名时保留的候选；主规范名不静默切换。
    pub alternates: Vec<String>,
}

/// 覆盖层选中的一个指称跨度（字符区间 `[start, end)`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexiconSpan {
    pub surface: String,
    pub canonical: String,
    pub pos: String,
    pub etype: Option<String>,
    pub origin: String,
    pub start: usize,
    pub end: usize,
}

impl LexiconSpan {
    pub fn char_len(&self) -> usize {
        self.end - self.start
    }
}

/// 词表本体。按词面排序存储，另维护首字符索引供线性扫描。
#[derive(Debug, Clone, Default)]
pub struct MentionLexicon {
    entries: BTreeMap<String, LexiconEntry>,
    // 首字符 → 词面列表（长度降序、字典序升序），索引只是派生数据
    by_first: HashMap<char, Vec<String>>,
}

impl MentionLexicon {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entry(&self, surface: &str) -> Option<&LexiconEntry> {
        self.entries.get(surface.trim())
    }

    pub fn entries(&self) -> impl Iterator<Item = &LexiconEntry> {
        self.entries.values()
    }

    pub fn confirmed_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.status == LexiconStatus::Confirmed)
            .count()
    }

    pub fn pending_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.status == LexiconStatus::Pending)
            .count()
    }

    pub fn tombstoned_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.status == LexiconStatus::Tombstoned)
            .count()
    }

    /// 按实体类型给出分词词性。词性只影响模式路由的槽位，不写回图谱类型。
    pub fn pos_for_etype(etype: Option<&str>) -> &'static str {
        let Some(etype) = etype.map(str::trim).filter(|etype| !etype.is_empty()) else {
            return "n";
        };
        let lower = etype.to_lowercase();
        if lower.contains("person")
            || etype.contains("人物")
            || etype.contains("人名")
            || etype.contains("角色")
            || etype == "人"
        {
            return "nr";
        }
        if lower.contains("place")
            || lower.contains("location")
            || lower.contains("city")
            || lower.contains("country")
            || etype.contains("地点")
            || etype.contains("城市")
            || etype.contains("国家")
            || etype.contains("地区")
        {
            return "ns";
        }
        if lower.contains("organization")
            || lower.contains("organisation")
            || lower.contains("company")
            || lower.contains("institution")
            || lower.contains("org")
            || etype.contains("组织")
            || etype.contains("机构")
            || etype.contains("公司")
            || etype.contains("大学")
            || etype.contains("医院")
            || etype.contains("学院")
        {
            return "nt";
        }
        "n"
    }

    /// 确认一个词面（显式准入 / 已确认别名 / 图谱实体）。
    ///
    /// 幂等：相同规范名与类型只增加观察次数并返回 `false`。墓碑优先：
    /// 已拒绝的词面不能被任何观察重新确认，返回 `false`。
    pub fn confirm(
        &mut self,
        surface: &str,
        canonical: &str,
        etype: Option<&str>,
        origin: &str,
    ) -> bool {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() {
            return false;
        }
        let canonical = if canonical.is_empty() {
            surface
        } else {
            canonical
        };
        let etype = etype.map(str::trim).filter(|etype| !etype.is_empty());
        let pos = Self::pos_for_etype(etype).to_string();

        if let Some(existing) = self.entries.get_mut(surface) {
            if existing.status == LexiconStatus::Tombstoned {
                return false;
            }
            let was_confirmed = existing.status == LexiconStatus::Confirmed;
            let mut changed = !was_confirmed;
            if !was_confirmed {
                existing.status = LexiconStatus::Confirmed;
                existing.canonical = canonical.to_string();
                existing.pos = pos;
                existing.etype = etype.map(str::to_string);
                existing.alternates.clear();
            } else if existing.canonical == canonical {
                // 同义项只允许补全类型，不允许用缺失类型把已知类型降级。
                if let Some(etype) = etype {
                    if existing.etype.as_deref() != Some(etype) {
                        existing.etype = Some(etype.to_string());
                        existing.pos = pos;
                        changed = true;
                    }
                }
            } else if !existing.alternates.iter().any(|alt| alt == canonical) {
                // 同一词面指向多个规范名：保留主规范名，另存候选，避免静默择一。
                existing.alternates.push(canonical.to_string());
                existing.alternates.sort();
                changed = true;
            }
            existing.origin = origin.to_string();
            existing.observations = existing.observations.saturating_add(1);
            return changed;
        }

        self.entries.insert(
            surface.to_string(),
            LexiconEntry {
                surface: surface.to_string(),
                canonical: canonical.to_string(),
                pos,
                etype: etype.map(str::to_string),
                status: LexiconStatus::Confirmed,
                origin: origin.to_string(),
                observations: 1,
                alternates: Vec::new(),
            },
        );
        self.insert_index(surface);
        true
    }

    /// 提名一个 OOV 词面：只记录待确认，不进入切分覆盖层。
    pub fn nominate(&mut self, surface: &str, etype: Option<&str>, origin: &str) -> bool {
        let surface = surface.trim();
        if surface.is_empty() {
            return false;
        }
        if let Some(existing) = self.entries.get_mut(surface) {
            match existing.status {
                LexiconStatus::Tombstoned | LexiconStatus::Confirmed => return false,
                LexiconStatus::Pending => {
                    existing.observations = existing.observations.saturating_add(1);
                    return false;
                }
            }
        }
        let etype = etype.map(str::trim).filter(|etype| !etype.is_empty());
        self.entries.insert(
            surface.to_string(),
            LexiconEntry {
                surface: surface.to_string(),
                canonical: surface.to_string(),
                pos: Self::pos_for_etype(etype).to_string(),
                etype: etype.map(str::to_string),
                status: LexiconStatus::Pending,
                origin: origin.to_string(),
                observations: 1,
                alternates: Vec::new(),
            },
        );
        true
    }

    /// 拒绝一个词面：墓碑永久挡板；返回是否改变。
    pub fn reject(&mut self, surface: &str, origin: &str) -> bool {
        let surface = surface.trim();
        if surface.is_empty() {
            return false;
        }
        if let Some(existing) = self.entries.get_mut(surface) {
            if existing.status == LexiconStatus::Tombstoned {
                return false;
            }
            existing.status = LexiconStatus::Tombstoned;
            existing.origin = origin.to_string();
            existing.observations = existing.observations.saturating_add(1);
            self.remove_index(surface);
            return true;
        }
        self.entries.insert(
            surface.to_string(),
            LexiconEntry {
                surface: surface.to_string(),
                canonical: surface.to_string(),
                pos: "n".to_string(),
                etype: None,
                status: LexiconStatus::Tombstoned,
                origin: origin.to_string(),
                observations: 1,
                alternates: Vec::new(),
            },
        );
        true
    }

    /// 彻底忘记一个词面（撤销墓碑或确认；主要用于显式维护）。
    pub fn forget(&mut self, surface: &str) -> bool {
        let surface = surface.trim();
        let removed = self.entries.remove(surface).is_some();
        if removed {
            self.remove_index(surface);
        }
        removed
    }

    /// 当前文本上选中的覆盖跨度：覆盖字数最大，平局取指称数更少者。
    pub fn confirmed_spans(&self, text: &str) -> Vec<LexiconSpan> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let mut candidates: Vec<LexiconSpan> = Vec::new();
        for start in 0..chars.len() {
            let Some(bucket) = self.by_first.get(&chars[start]) else {
                continue;
            };
            for surface in bucket {
                let Some(entry) = self.entries.get(surface) else {
                    continue;
                };
                if entry.status != LexiconStatus::Confirmed {
                    continue;
                }
                let surface_chars: Vec<char> = surface.chars().collect();
                if surface_chars.len() < 2 {
                    continue;
                }
                let end = start + surface_chars.len();
                if end > chars.len() || chars[start..end] != surface_chars[..] {
                    continue;
                }
                candidates.push(LexiconSpan {
                    surface: entry.surface.clone(),
                    canonical: entry.canonical.clone(),
                    pos: entry.pos.clone(),
                    etype: entry.etype.clone(),
                    origin: entry.origin.clone(),
                    start,
                    end,
                });
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }
        // 加权区间调度：best[pos] = 前缀 [0,pos) 的最优选路
        candidates.sort_by(|a, b| {
            a.end
                .cmp(&b.end)
                .then_with(|| a.start.cmp(&b.start))
                .then_with(|| b.char_len().cmp(&a.char_len()))
                .then_with(|| a.surface.cmp(&b.surface))
        });
        let n = chars.len();
        let mut by_end: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (index, candidate) in candidates.iter().enumerate() {
            by_end[candidate.end].push(index);
        }
        #[derive(Debug, Clone, Copy)]
        struct Best {
            covered: usize,
            spans: usize,
            candidate: Option<usize>,
        }
        let mut best: Vec<Best> = vec![
            Best {
                covered: 0,
                spans: 0,
                candidate: None,
            };
            n + 1
        ];
        for pos in 1..=n {
            best[pos] = best[pos - 1];
            for &index in &by_end[pos] {
                let candidate = &candidates[index];
                let previous = &best[candidate.start];
                let covered = previous.covered + candidate.char_len();
                let spans = previous.spans + 1;
                let replace = covered > best[pos].covered
                    || (covered == best[pos].covered && spans < best[pos].spans)
                    || (covered == best[pos].covered
                        && spans == best[pos].spans
                        && best[pos]
                            .candidate
                            .map(|current| candidate.char_len() > candidates[current].char_len())
                            .unwrap_or(true));
                if replace {
                    best[pos] = Best {
                        covered,
                        spans,
                        candidate: Some(index),
                    };
                }
            }
        }
        let mut selected: Vec<LexiconSpan> = Vec::new();
        let mut pos = n;
        while let Some(index) = best[pos].candidate {
            let candidate = &candidates[index];
            selected.push(candidate.clone());
            pos = candidate.start;
        }
        selected.reverse();
        selected
    }

    /// 覆盖层匹配报告：选中跨度、规范名与来源（只读审计）。
    pub fn render_match_report(&self, text: &str) -> String {
        let spans = self.confirmed_spans(text);
        if spans.is_empty() {
            return format!(
                "提及词表：「{}」没有已确认词面命中（确认 {} / 待定 {} / 墓碑 {}）。",
                text,
                self.confirmed_count(),
                self.pending_count(),
                self.tombstoned_count()
            );
        }
        let mut output = format!(
            "提及词表：「{}」选中 {} 个已确认跨度（确认 {} / 待定 {} / 墓碑 {}）。\n",
            text,
            spans.len(),
            self.confirmed_count(),
            self.pending_count(),
            self.tombstoned_count()
        );
        for span in spans {
            let alt_note = self
                .entry(&span.surface)
                .filter(|entry| !entry.alternates.is_empty())
                .map(|entry| format!("｜多候选 {}", entry.alternates.join(" / ")))
                .unwrap_or_default();
            let _ = writeln!(
                output,
                "  • 「{}」（第 {} 字起，{} 字）→ {}｜词性 {}｜来源 {}{}",
                span.surface,
                span.start + 1,
                span.char_len(),
                span.canonical,
                span.pos,
                span.origin,
                alt_note
            );
        }
        output.trim_end().to_string()
    }

    /// 完整词表报告：按词面排序，状态与来源可复核。
    pub fn render_report(&self) -> String {
        if self.entries.is_empty() {
            return "提及词表：空（准入实体、确认别名或观察图谱后自动增长）。".to_string();
        }
        let mut output = format!(
            "提及词表：{} 条（确认 {} / 待定 {} / 墓碑 {}）。\n",
            self.len(),
            self.confirmed_count(),
            self.pending_count(),
            self.tombstoned_count()
        );
        for entry in self.entries.values() {
            let type_note = entry
                .etype
                .as_deref()
                .map(|etype| format!("，类型 {}", etype))
                .unwrap_or_default();
            let alt_note = if entry.alternates.is_empty() {
                String::new()
            } else {
                format!("，多候选 {}", entry.alternates.join(" / "))
            };
            let _ = writeln!(
                output,
                "  • 「{}」→「{}」（{}，词性 {}{}{}；来源 {}；观察 {}）",
                entry.surface,
                entry.canonical,
                entry.status.label(),
                entry.pos,
                type_note,
                alt_note,
                entry.origin,
                entry.observations
            );
        }
        output.trim_end().to_string()
    }

    fn insert_index(&mut self, surface: &str) {
        let Some(&first) = surface.chars().next().as_ref() else {
            return;
        };
        let len = surface.chars().count();
        let bucket = self.by_first.entry(first).or_default();
        if bucket.iter().any(|existing| existing == surface) {
            return;
        }
        let position = bucket
            .binary_search_by(|probe| {
                probe
                    .chars()
                    .count()
                    .cmp(&len)
                    .reverse()
                    .then_with(|| probe.as_str().cmp(surface))
            })
            .unwrap_or_else(|position| position);
        bucket.insert(position, surface.to_string());
    }

    fn remove_index(&mut self, surface: &str) {
        let Some(first) = surface.chars().next() else {
            return;
        };
        let Some(bucket) = self.by_first.get_mut(&first) else {
            return;
        };
        bucket.retain(|existing| existing != surface);
        if bucket.is_empty() {
            self.by_first.remove(&first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon_with(entries: &[(&str, &str, Option<&str>)]) -> MentionLexicon {
        let mut lexicon = MentionLexicon::new();
        for (surface, canonical, etype) in entries {
            lexicon.confirm(surface, canonical, *etype, "test");
        }
        lexicon
    }

    #[test]
    fn confirmed_long_surface_wins_over_short_overlap() {
        let lexicon = lexicon_with(&[
            ("南京市", "南京市", Some("City")),
            ("南京市长江大桥", "南京市长江大桥", Some("Bridge")),
        ]);
        let spans = lexicon.confirmed_spans("南京市长江大桥的限速");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].surface, "南京市长江大桥");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 7);
        assert_eq!(spans[0].pos, "n");
    }

    #[test]
    fn non_overlapping_spans_cover_more_characters() {
        let lexicon = lexicon_with(&[
            ("张三", "张三", Some("Person")),
            ("李四", "李四", Some("Person")),
        ]);
        let spans = lexicon.confirmed_spans("张三和李四");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].surface, "张三");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 2);
        assert_eq!(spans[1].surface, "李四");
        assert_eq!(spans[1].start, 3);
        assert_eq!(spans[1].end, 5);
    }

    #[test]
    fn pending_entries_never_enter_the_overlay() {
        let mut lexicon = MentionLexicon::new();
        assert!(lexicon.nominate("未名科技公司", Some("Organization"), "oov"));
        assert_eq!(lexicon.pending_count(), 1);
        assert!(lexicon.confirmed_spans("未名科技公司").is_empty());
    }

    #[test]
    fn tombstone_blocks_confirmation_and_overlay() {
        let mut lexicon = lexicon_with(&[("未名科技公司", "未名科技公司", Some("Organization"))]);
        assert!(lexicon.reject("未名科技公司", "user"));
        assert!(!lexicon.confirm("未名科技公司", "未名科技公司", Some("Organization"), "kb"));
        assert!(lexicon.confirmed_spans("未名科技公司").is_empty());
        assert_eq!(lexicon.tombstoned_count(), 1);
    }

    #[test]
    fn etype_maps_to_pattern_pos() {
        assert_eq!(MentionLexicon::pos_for_etype(Some("Person")), "nr");
        assert_eq!(MentionLexicon::pos_for_etype(Some("城市")), "ns");
        assert_eq!(MentionLexicon::pos_for_etype(Some("Organization")), "nt");
        assert_eq!(MentionLexicon::pos_for_etype(Some("Fruit")), "n");
        assert_eq!(MentionLexicon::pos_for_etype(None), "n");
    }

    #[test]
    fn repeated_confirmation_is_idempotent_but_counts_evidence() {
        let mut lexicon = lexicon_with(&[("姚明", "姚明", Some("Person"))]);
        assert!(!lexicon.confirm("姚明", "姚明", Some("Person"), "kb"));
        assert_eq!(lexicon.entry("姚明").unwrap().observations, 2);
        assert!(lexicon.render_report().contains("观察 2"));
    }

    #[test]
    fn missing_type_never_downgrades_a_confirmed_entry() {
        let mut lexicon = lexicon_with(&[("未名科技", "未名科技", Some("Organization"))]);
        assert!(!lexicon.confirm("未名科技", "未名科技", None, "kb_relation"));
        let entry = lexicon.entry("未名科技").unwrap();
        assert_eq!(entry.pos, "nt");
        assert_eq!(entry.etype.as_deref(), Some("Organization"));
        assert_eq!(entry.origin, "kb_relation");
    }

    #[test]
    fn conflicting_canonicals_are_kept_as_alternates() {
        let mut lexicon = lexicon_with(&[("未名", "未名科技", Some("Organization"))]);
        assert!(lexicon.confirm("未名", "未名大学", Some("University"), "confirmed_alias"));
        let entry = lexicon.entry("未名").unwrap();
        assert_eq!(entry.canonical, "未名科技");
        assert_eq!(entry.alternates, vec!["未名大学".to_string()]);
        assert!(lexicon.render_report().contains("多候选 未名大学"));
        assert!(lexicon
            .render_match_report("未名科技")
            .contains("多候选 未名大学"));
    }

    #[test]
    fn report_is_sorted_and_reports_all_states() {
        let mut lexicon = lexicon_with(&[
            ("张三", "张三", Some("Person")),
            ("李四", "李四", Some("Person")),
        ]);
        lexicon.nominate("未名词", None, "oov");
        lexicon.reject("坏词", "user");
        let report = lexicon.render_report();
        assert!(report.contains("确认 2 / 待定 1 / 墓碑 1"), "{}", report);
        let zhang = report.find("张三").unwrap();
        let li = report.find("李四").unwrap();
        assert!(zhang < li, "报告应按词面排序：{}", report);
    }
}

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::DialogueKnowledge;
use crate::quantity::{self, Comparison, Dimension, Quantity, RankSpec};

const PAGE_SIZE: usize = 5;
const MAX_ROWS: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryRow {
    pub subject: String,
    pub predicate: String,
    pub value: String,
    pub evidence: String,
}

impl QueryRow {
    fn new(subject: &str, predicate: &str, value: &str) -> Self {
        QueryRow {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            value: value.to_string(),
            evidence: format!("{} {} {}", subject, predicate, value),
        }
    }

    fn with_evidence(subject: &str, predicate: &str, value: &str, evidence: String) -> Self {
        QueryRow {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            value: value.to_string(),
            evidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuerySource {
    Scan {
        predicate: Option<String>,
        entity_type: Option<String>,
    },
    Compare {
        left: String,
        right: String,
        attribute: String,
    },
    /// 极值/排序结果：属性、方向与名次范围。
    Rank {
        predicate: String,
        descending: bool,
    },
    /// 等待用户补全「按哪个属性排名」。
    RankPending { descending: bool },
    Manual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardFrame {
    pub op: String,
    pub program: String,
    pub source: QuerySource,
    pub rows: Vec<QueryRow>,
    #[serde(default)]
    pub selected: Option<usize>,
    #[serde(default)]
    pub page: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default)]
    pub revision: u64,
}

fn default_page_size() -> usize {
    PAGE_SIZE
}

impl BoardFrame {
    fn new(
        op: &str,
        program: String,
        source: QuerySource,
        rows: Vec<QueryRow>,
        revision: u64,
    ) -> Self {
        BoardFrame {
            op: op.to_string(),
            program,
            source,
            rows,
            selected: None,
            page: 0,
            page_size: PAGE_SIZE,
            revision,
        }
    }

    fn page_count(&self) -> usize {
        self.rows.len().div_ceil(self.page_size.max(1)).max(1)
    }

    fn selected_row(&self) -> Option<&QueryRow> {
        self.selected.and_then(|index| self.rows.get(index))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryBoard {
    #[serde(default)]
    frames: Vec<BoardFrame>,
    #[serde(default)]
    cursor: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardView {
    pub frame: usize,
    pub frames: usize,
    pub program: String,
    pub rows: usize,
    pub page: usize,
    pub pages: usize,
    pub selected_row: Option<usize>,
    /// 选中行的主语（客户端无需再取回整帧即可显示「当前选中谁」）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_subject: Option<String>,
    pub revision: u64,
    pub stale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardOutcome {
    pub text: String,
    pub plan: Vec<String>,
    pub provenance: Vec<String>,
    pub focus: Option<String>,
    pub revision: u64,
}

impl QueryBoard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_active(&self) -> bool {
        self.cursor
            .and_then(|index| self.frames.get(index))
            .is_some()
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn cursor(&self) -> Option<usize> {
        self.cursor
    }

    pub fn current(&self) -> Option<&BoardFrame> {
        self.cursor.and_then(|index| self.frames.get(index))
    }

    pub fn summary(&self) -> String {
        match self.current() {
            Some(frame) => format!(
                "查询板：第 {}/{} 帧 · {} · {} 行 · 知识版本 v{}",
                self.cursor.map_or(0, |index| index + 1),
                self.frames.len(),
                frame.program,
                frame.rows.len(),
                frame.revision
            ),
            None => "查询板：（空）".to_string(),
        }
    }

    pub fn view(&self, revision: u64) -> Option<BoardView> {
        let index = self.cursor?;
        let frame = self.frames.get(index)?;
        Some(BoardView {
            frame: index + 1,
            frames: self.frames.len(),
            program: frame.program.clone(),
            rows: frame.rows.len(),
            page: frame.page + 1,
            pages: frame.page_count(),
            selected_row: frame.selected.map(|selected| selected + 1),
            selected_subject: frame.selected_row().map(|row| row.subject.clone()),
            revision: frame.revision,
            stale: frame.revision != revision,
        })
    }

    fn push_frame(&mut self, frame: BoardFrame) {
        let keep = self.cursor.map_or(0, |index| index + 1);
        self.frames.truncate(keep);
        self.frames.push(frame);
        self.cursor = Some(self.frames.len() - 1);
    }

    pub fn apply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        self.apply_with_hint(input, knowledge, revision, None)
    }

    /// 与 [`Self::apply`] 相同，但允许调用方提供上一轮属性作为省略补全。
    ///
    /// 「姚明的身高是多少？」之后直接问「谁最高」时，属性由对话历史给出，
    /// 查询板不必再猜。
    pub fn apply_with_hint(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
        attribute_hint: Option<&str>,
    ) -> Option<BoardOutcome> {
        let text = clean(input);
        if text.is_empty() {
            return None;
        }
        if let Some(outcome) = self.resolve_pending_rank(&text, knowledge, revision) {
            return Some(outcome);
        }
        if let Some(spec) = parse_extreme(&text, self.is_active()) {
            if let Some(outcome) =
                self.apply_extreme(spec, knowledge, revision, attribute_hint)
            {
                return Some(outcome);
            }
        }
        if !self.is_active() {
            let spec = parse_scan(&text, knowledge)?;
            return self.start_scan(spec, knowledge, revision);
        }

        if is_undo(&text) {
            return self.apply_undo(knowledge, revision);
        }
        if is_reset(&text) {
            return self.apply_reset();
        }
        if is_explain(&text) {
            return self.apply_explain(knowledge, revision);
        }
        if is_count(&text) {
            return self.apply_count(knowledge, revision);
        }
        if let Some(other) = parse_compare(&text) {
            return self.apply_compare(&other, knowledge, revision);
        }
        if let Some(sort) = parse_sort(&text) {
            return self.apply_sort(sort, knowledge, revision);
        }
        if let Some(filter) = parse_filter(&text) {
            return self.apply_filter(filter, knowledge, revision);
        }
        if let Some(page) = parse_page_number(&text) {
            return self.apply_page(page, knowledge, revision);
        }
        if is_next_page(&text) {
            return self.apply_next_page(&text, knowledge, revision);
        }
        if is_prev_page(&text) {
            return self.apply_prev_page(knowledge, revision);
        }
        if let Some(ordinal) = parse_ordinal(&text) {
            return self.apply_select(ordinal, knowledge, revision);
        }
        if let Some(target) = parse_select_entity(&text) {
            return self.apply_select_entity(&target, knowledge, revision);
        }
        if is_identity_request(&text) {
            return self.apply_identity(knowledge, revision);
        }
        if let Some(spec) = parse_scan(&text, knowledge) {
            return self.start_scan(spec, knowledge, revision);
        }
        None
    }

    /// 待澄清的排名请求：用户下一句只回答属性词时，直接开始排名。
    fn resolve_pending_rank(
        &mut self,
        text: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let descending = match self.current().map(|frame| &frame.source) {
            Some(QuerySource::RankPending { descending }) => *descending,
            _ => return None,
        };
        let answer = text.trim_matches(|ch: char| {
            matches!(ch, '。' | '，' | ',' | '！' | '!' | '？' | '?' | ' ')
        });
        if answer.is_empty() || answer.chars().count() > 12 {
            return None;
        }
        let attribute = resolve_attribute_hint(answer, knowledge)?;
        self.start_rank(attribute, descending, 1, knowledge, revision)
    }

    fn apply_extreme(
        &mut self,
        spec: ExtremeSpec,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
        attribute_hint: Option<&str>,
    ) -> Option<BoardOutcome> {
        let attribute = spec
            .attribute
            .clone()
            .or_else(|| attribute_hint.map(str::to_string))
            .or_else(|| self.current_rank_attribute());
        let Some(attribute) = attribute else {
            let frame = BoardFrame::new(
                "rank_pending",
                "等待补全排名属性".to_string(),
                QuerySource::RankPending {
                    descending: spec.descending,
                },
                Vec::new(),
                revision,
            );
            self.push_frame(frame);
            return Some(BoardOutcome {
                text: "你想按哪个属性排名？例如「谁的身高最高」或直接回答「身高」。".to_string(),
                plan: vec!["board:rank_pending".to_string()],
                provenance: vec!["查询板：排名请求缺少属性，未做任何猜测".to_string()],
                focus: None,
                revision,
            });
        };

        if self.is_active() {
            let frame = self.current()?.clone();
            let matching: Vec<QueryRow> = frame
                .rows
                .iter()
                .filter(|row| same_predicate(&row.predicate, &attribute, knowledge))
                .cloned()
                .collect();
            if !matching.is_empty() {
                let mut next = frame;
                next.rows = matching;
                next.selected = None;
                next.page = 0;
                next.op = "rank".to_string();
                next.source = QuerySource::Rank {
                    predicate: attribute.clone(),
                    descending: spec.descending,
                };
                let skipped =
                    sort_rows_by_quantity(&mut next.rows, &SortSpec {
                        column: Some(attribute.clone()),
                        descending: spec.descending,
                    }, knowledge);
                next.program = describe_rank(&attribute, spec.descending, skipped, knowledge);
                let summary =
                    rank_summary(&next.rows, spec.limit, &attribute, spec.descending, knowledge);
                let text = format!("{}\n{}", summary, render_frame(&next, revision));
                let focus = next.rows.first().map(|row| row.subject.clone());
                let provenance = provenance_for(&next, next.rows.first());
                self.push_frame(next);
                return Some(BoardOutcome {
                    text,
                    plan: vec!["board:rank".to_string()],
                    provenance,
                    focus,
                    revision,
                });
            }
        }
        self.start_rank(attribute, spec.descending, spec.limit, knowledge, revision)
    }

    fn start_rank(
        &mut self,
        attribute: String,
        descending: bool,
        limit: usize,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let spec = ScanSpec {
            predicate: Some(attribute.clone()),
            entity_type: None,
        };
        let rows = scan_rows(&spec, knowledge);
        if rows.is_empty() {
            return None;
        }
        let mut frame = BoardFrame::new(
            "rank",
            describe_rank(&attribute, descending, 0, knowledge),
            QuerySource::Rank {
                predicate: attribute.clone(),
                descending,
            },
            rows,
            revision,
        );
        let skipped = sort_rows_by_quantity(
            &mut frame.rows,
            &SortSpec {
                column: Some(attribute.clone()),
                descending,
            },
            knowledge,
        );
        frame.program = describe_rank(&attribute, descending, skipped, knowledge);
        let summary = rank_summary(&frame.rows, limit, &attribute, descending, knowledge);
        let text = format!("{}\n{}", summary, render_frame(&frame, revision));
        let focus = frame.rows.first().map(|row| row.subject.clone());
        let provenance = provenance_for(&frame, frame.rows.first());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:rank".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn current_rank_attribute(&self) -> Option<String> {
        let frame = self.current()?;
        match &frame.source {
            QuerySource::Scan {
                predicate: Some(predicate),
                ..
            }
            | QuerySource::Rank { predicate, .. } => Some(predicate.clone()),
            _ => {
                let first = frame.rows.first()?;
                if frame
                    .rows
                    .iter()
                    .all(|row| row.predicate == first.predicate)
                {
                    Some(first.predicate.clone())
                } else {
                    frame
                        .selected_row()
                        .map(|row| row.predicate.clone())
                }
            }
        }
    }

    fn start_scan(
        &mut self,
        spec: ScanSpec,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let rows = scan_rows(&spec, knowledge);
        let program = describe_scan(&spec);
        let source = QuerySource::Scan {
            predicate: spec.predicate.clone(),
            entity_type: spec.entity_type.clone(),
        };
        let frame = BoardFrame::new("scan", program, source, rows, revision);
        let focus = frame.rows.first().map(|row| row.subject.clone());
        let text = render_frame(&frame, revision);
        let plan = vec!["board:scan".to_string()];
        let provenance = provenance_for(&frame, frame.rows.first());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan,
            provenance,
            focus,
            revision,
        })
    }

    fn apply_undo(
        &mut self,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let index = self.cursor?;
        if index == 0 {
            let frame = self.frames.first()?;
            let mut text = render_frame(frame, revision);
            text.push_str("\n（已经是最早一步。）");
            return Some(BoardOutcome {
                text,
                plan: vec!["board:undo".to_string()],
                provenance: provenance_for(frame, frame.selected_row()),
                focus: frame.selected_row().map(|row| row.subject.clone()),
                revision,
            });
        }
        let target = index - 1;
        self.cursor = Some(target);
        let frame = self.frames.get(target)?;
        let mut text = render_frame(frame, revision);
        text.push_str(&format!("\n（已回到第 {} 帧。）", target + 1));
        let _ = knowledge;
        Some(BoardOutcome {
            text,
            plan: vec!["board:undo".to_string()],
            provenance: provenance_for(frame, frame.selected_row()),
            focus: frame.selected_row().map(|row| row.subject.clone()),
            revision,
        })
    }

    fn apply_reset(&mut self) -> Option<BoardOutcome> {
        if !self.is_active() {
            return None;
        }
        self.frames.clear();
        self.cursor = None;
        Some(BoardOutcome {
            text: "查询板已清空。可以直接提出新的集合问题，例如「列出所有人的身高」。".to_string(),
            plan: vec!["board:reset".to_string()],
            provenance: vec!["查询板：清空全部帧".to_string()],
            focus: None,
            revision: 0,
        })
    }

    fn apply_explain(
        &self,
        _knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let frame = self.current()?;
        let row = frame.selected_row().or_else(|| frame.rows.first());
        let mut text = match row {
            Some(row) => format!(
                "查询板证据（第 {} 帧 · 知识版本 v{}）：\n{}\n程序：{}\n该行完整记录：{}",
                self.cursor.map_or(0, |index| index + 1),
                frame.revision,
                row.evidence,
                frame.program,
                row.evidence
            ),
            None => "查询板当前没有可解释的结果行。".to_string(),
        };
        if row.is_some() && frame.revision != revision {
            text.push_str(&format!(
                "\n[知识版本 v{} → v{}：该证据来自旧版本，当前查询应重新计算。]",
                frame.revision, revision
            ));
        }
        Some(BoardOutcome {
            text,
            plan: vec!["board:explain".to_string()],
            provenance: vec![
                format!(
                    "查询板第 {} 帧：{}",
                    self.cursor.map_or(0, |i| i + 1),
                    frame.program
                ),
                row.map(|row| row.evidence.clone())
                    .unwrap_or_else(|| "无结果行".to_string()),
            ],
            focus: row.map(|row| row.subject.clone()),
            revision,
        })
    }

    fn apply_count(
        &self,
        _knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let frame = self.current()?;
        let selected = frame
            .selected_row()
            .map(|row| format!("；当前选中「{}」", row.subject))
            .unwrap_or_default();
        Some(BoardOutcome {
            text: format!(
                "当前查询板共 {} 行，分 {} 页（第 {} 页）{}。",
                frame.rows.len(),
                frame.page_count(),
                frame.page + 1,
                selected
            ),
            plan: vec!["board:count".to_string()],
            provenance: provenance_for(frame, frame.selected_row()),
            focus: frame.selected_row().map(|row| row.subject.clone()),
            revision,
        })
    }

    fn apply_next_page(
        &mut self,
        raw: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let _ = knowledge;
        let mut frame = self.current()?.clone();
        if frame.page + 1 >= frame.page_count() {
            if matches!(raw, "还有吗" | "还有呢" | "还有别的吗" | "还有其它吗") {
                return None;
            }
            let mut text = render_frame(&frame, revision);
            text.push_str("\n（已经是最后一页。）");
            return Some(BoardOutcome {
                text,
                plan: vec!["board:page".to_string()],
                provenance: provenance_for(&frame, frame.selected_row()),
                focus: frame.selected_row().map(|row| row.subject.clone()),
                revision,
            });
        }
        frame.page += 1;
        let text = render_frame(&frame, revision);
        let focus = frame.selected_row().map(|row| row.subject.clone());
        let provenance = provenance_for(&frame, frame.selected_row());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:page".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_prev_page(
        &mut self,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let _ = knowledge;
        let mut frame = self.current()?.clone();
        if frame.page == 0 {
            let mut text = render_frame(&frame, revision);
            text.push_str("\n（已经是第一页。）");
            return Some(BoardOutcome {
                text,
                plan: vec!["board:page".to_string()],
                provenance: provenance_for(&frame, frame.selected_row()),
                focus: frame.selected_row().map(|row| row.subject.clone()),
                revision,
            });
        }
        frame.page -= 1;
        let text = render_frame(&frame, revision);
        let focus = frame.selected_row().map(|row| row.subject.clone());
        let provenance = provenance_for(&frame, frame.selected_row());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:page".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_page(
        &mut self,
        page: usize,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let _ = knowledge;
        let mut frame = self.current()?.clone();
        let pages = frame.page_count();
        if page >= pages {
            let mut text = render_frame(&frame, revision);
            text.push_str(&format!("\n（只有 {} 页，未切换。）", pages));
            return Some(BoardOutcome {
                text,
                plan: vec!["board:page".to_string()],
                provenance: provenance_for(&frame, frame.selected_row()),
                focus: frame.selected_row().map(|row| row.subject.clone()),
                revision,
            });
        }
        frame.page = page;
        let text = render_frame(&frame, revision);
        let focus = frame.selected_row().map(|row| row.subject.clone());
        let provenance = provenance_for(&frame, frame.selected_row());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:page".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_select(
        &mut self,
        ordinal: usize,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let _ = knowledge;
        let mut frame = self.current()?.clone();
        if ordinal >= frame.rows.len() {
            let mut text = render_frame(&frame, revision);
            text.push_str(&format!(
                "\n（没有第 {} 行；当前共 {} 行。）",
                ordinal + 1,
                frame.rows.len()
            ));
            return Some(BoardOutcome {
                text,
                plan: vec!["board:select".to_string()],
                provenance: provenance_for(&frame, frame.selected_row()),
                focus: frame.selected_row().map(|row| row.subject.clone()),
                revision,
            });
        }
        frame.selected = Some(ordinal);
        frame.page = ordinal / frame.page_size.max(1);
        let row = frame.selected_row().cloned();
        let text = render_frame(&frame, revision);
        let focus = row.as_ref().map(|row| row.subject.clone());
        let provenance = provenance_for(&frame, row.as_ref());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:select".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_select_entity(
        &mut self,
        target: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let _ = knowledge;
        let frame = self.current()?;
        let index = frame.rows.iter().position(|row| {
            row.subject.contains(target)
                || row.value.contains(target)
                || row.predicate.contains(target)
        })?;
        self.apply_select(index, knowledge, revision)
    }

    fn apply_filter(
        &mut self,
        filter: FilterSpec,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let mut frame = self.current()?.clone();
        let old_selection = frame.selected_row().cloned();
        frame.rows.retain(|row| match &filter {
            FilterSpec::Text(needle) => subject_matches(knowledge, &row.subject, needle),
            FilterSpec::Numeric {
                column,
                op,
                threshold,
            } => {
                let predicate_matches = column
                    .as_deref()
                    .map(|column| same_predicate(&row.predicate, column, knowledge))
                    .unwrap_or(true);
                predicate_matches
                    && quantity::parse_quantity(&row.value)
                        .map(|value| op.matches(value.compare(threshold)))
                        .unwrap_or(false)
            }
        });
        frame.selected =
            old_selection.and_then(|selected| frame.rows.iter().position(|row| *row == selected));
        frame.page = 0;
        frame.op = "filter".to_string();
        frame.program = format!("{} · 筛选：{}", frame.program, filter.describe());
        let text = render_frame(&frame, revision);
        let provenance = provenance_for(&frame, frame.selected_row());
        let focus = frame.selected_row().map(|row| row.subject.clone());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:filter".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_sort(
        &mut self,
        sort: SortSpec,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let mut frame = self.current()?.clone();
        let old_selection = frame.selected_row().cloned();
        let skipped = sort_rows_by_quantity(&mut frame.rows, &sort, knowledge);
        frame.selected =
            old_selection.and_then(|selected| frame.rows.iter().position(|row| *row == selected));
        if let Some(selected) = frame.selected {
            frame.page = selected / frame.page_size.max(1);
        } else {
            frame.page = 0;
        }
        frame.op = "sort".to_string();
        let note = if skipped > 0 {
            format!(" · {} 行量纲不同或非数值，未参与数值排序", skipped)
        } else {
            String::new()
        };
        frame.program = format!("{} · {}{}", frame.program, sort.describe(), note);
        let text = render_frame(&frame, revision);
        let provenance = provenance_for(&frame, frame.selected_row());
        let focus = frame.selected_row().map(|row| row.subject.clone());
        self.push_frame(frame);
        Some(BoardOutcome {
            text,
            plan: vec!["board:sort".to_string()],
            provenance,
            focus,
            revision,
        })
    }

    fn apply_compare(
        &mut self,
        other: &str,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let frame = self.current()?;
        let base = frame.selected_row().or_else(|| frame.rows.first())?.clone();
        let left = if base.subject == other {
            frame
                .rows
                .iter()
                .find(|row| row.subject != other)
                .cloned()
                .unwrap_or(base)
        } else {
            base
        };

        if other.contains("这两个") || other.contains("前两个") || other.contains("两行") {
            let first = frame.rows.first()?.clone();
            let second = frame.rows.get(1)?.clone();
            let text = render_comparison(&first, &second, knowledge);
            let rows = vec![first.clone(), second.clone()];
            let mut next = frame.clone();
            next.op = "compare".to_string();
            next.program = format!(
                "{} · 比较：{} 与 {}",
                frame.program, first.subject, second.subject
            );
            next.source = QuerySource::Compare {
                left: first.subject.clone(),
                right: second.subject.clone(),
                attribute: first.predicate.clone(),
            };
            next.rows = rows;
            next.selected = None;
            next.page = 0;
            let provenance = provenance_for(&next, None);
            self.push_frame(next);
            return Some(BoardOutcome {
                text,
                plan: vec!["board:compare".to_string()],
                provenance,
                focus: Some(first.subject.clone()),
                revision,
            });
        }

        let right_values = predicate_values(knowledge, other, &left.predicate);
        let text = if right_values.is_empty() {
            format!(
                "查询板：没有记录「{} 的 {}」，无法与「{}」比较。",
                other, left.predicate, left.subject
            )
        } else {
            let right_value = right_values.join("、");
            let right_row = QueryRow::new(other, &left.predicate, &right_value);
            let text = render_comparison(&left, &right_row, knowledge);
            let rows = vec![left.clone(), right_row.clone()];
            let mut next = frame.clone();
            next.op = "compare".to_string();
            next.program = format!(
                "{} · 比较：{} 与 {} 的 {}",
                frame.program, left.subject, other, left.predicate
            );
            next.source = QuerySource::Compare {
                left: left.subject.clone(),
                right: other.to_string(),
                attribute: left.predicate.clone(),
            };
            next.rows = rows;
            next.selected = None;
            next.page = 0;
            let provenance = provenance_for(&next, None);
            self.push_frame(next);
            return Some(BoardOutcome {
                text,
                plan: vec!["board:compare".to_string()],
                provenance,
                focus: Some(left.subject.clone()),
                revision,
            });
        };
        Some(BoardOutcome {
            text,
            plan: vec!["board:compare".to_string()],
            provenance: provenance_for(frame, Some(&left)),
            focus: Some(left.subject.clone()),
            revision,
        })
    }

    fn apply_identity(
        &self,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let frame = self.current()?;
        let row = frame.selected_row().or_else(|| frame.rows.first())?;
        let description = knowledge.describe(&row.subject);
        let facts = entity_facts(knowledge, &row.subject);
        let mut text = match description {
            Some(description) => format!("「{}」：{}", row.subject, description),
            None if !facts.is_empty() => {
                let mut text = format!("「{}」的记录：", row.subject);
                for (predicate, value) in facts.iter().take(8) {
                    text.push_str(&format!("\n  · {}：{}", predicate, value));
                }
                text
            }
            None => format!("「{}」：知识库中没有更多记录。", row.subject),
        };
        if frame.revision != revision {
            text.push_str(&format!(
                "\n（该行来自知识版本 v{}，当前为 v{}。）",
                frame.revision, revision
            ));
        }
        Some(BoardOutcome {
            text,
            plan: vec!["board:identity".to_string()],
            provenance: provenance_for(frame, Some(row)),
            focus: Some(row.subject.clone()),
            revision,
        })
    }
}

#[derive(Debug, Clone)]
struct ScanSpec {
    predicate: Option<String>,
    entity_type: Option<String>,
}

fn describe_scan(spec: &ScanSpec) -> String {
    let mut parts = vec!["扫描".to_string()];
    if let Some(entity_type) = &spec.entity_type {
        parts.push(format!("类型「{}」", entity_type));
    }
    match &spec.predicate {
        Some(predicate) => parts.push(format!("谓词「{}」", predicate)),
        None => parts.push("全部实体".to_string()),
    }
    parts.join(" · ")
}

fn scan_rows(spec: &ScanSpec, knowledge: &dyn DialogueKnowledge) -> Vec<QueryRow> {
    let entities: Vec<String> = if let Some(entity_type) = &spec.entity_type {
        let mut members = knowledge.entities_of_type(entity_type);
        if members.is_empty() {
            members = knowledge
                .all_entities()
                .into_iter()
                .filter(|entity| {
                    knowledge.entity_type(entity).as_deref() == Some(entity_type.as_str())
                })
                .collect();
        }
        members
    } else {
        knowledge.all_entities()
    };

    let mut rows = Vec::new();
    for entity in entities {
        if let Some(predicate) = &spec.predicate {
            let mut facts = knowledge.entity_attributes(&entity);
            facts.extend(knowledge.entity_relations(&entity));
            facts.sort();
            facts.dedup();
            for (key, value) in facts {
                if same_predicate(&key, predicate, knowledge) {
                    let label = knowledge.predicate_label(&key);
                    rows.push(QueryRow::with_evidence(
                        &entity,
                        &label,
                        &value,
                        format!("{} {} {}", entity, key, value),
                    ));
                    if rows.len() >= MAX_ROWS {
                        break;
                    }
                }
            }
        } else {
            let entity_type = knowledge
                .entity_type(&entity)
                .unwrap_or_else(|| "实体".to_string());
            rows.push(QueryRow::with_evidence(
                &entity,
                "类型",
                &entity_type,
                format!("{} 的类型是 {}", entity, entity_type),
            ));
        }
        if rows.len() >= MAX_ROWS {
            break;
        }
    }
    rows.sort_by(|left, right| {
        left.subject
            .cmp(&right.subject)
            .then_with(|| left.predicate.cmp(&right.predicate))
            .then_with(|| left.value.cmp(&right.value))
    });
    rows.truncate(MAX_ROWS);
    rows
}

fn entity_facts(knowledge: &dyn DialogueKnowledge, entity: &str) -> Vec<(String, String)> {
    let mut facts = knowledge.entity_attributes(entity);
    facts.extend(knowledge.entity_relations(entity));
    facts.sort();
    facts.dedup();
    facts
}

fn predicate_values(
    knowledge: &dyn DialogueKnowledge,
    entity: &str,
    predicate: &str,
) -> Vec<String> {
    let mut values = knowledge.attribute_values(entity, predicate);
    if values.is_empty() {
        values = knowledge
            .entity_relations(entity)
            .into_iter()
            .filter(|(key, _)| same_predicate(key, predicate, knowledge))
            .map(|(_, value)| value)
            .collect();
    }
    values.sort();
    values.dedup();
    values
}

fn same_predicate(key: &str, predicate: &str, knowledge: &dyn DialogueKnowledge) -> bool {
    if key == predicate {
        return true;
    }
    let key_canonical = knowledge
        .canonical_predicate(key)
        .unwrap_or_else(|| key.to_string());
    let predicate_canonical = knowledge
        .canonical_predicate(predicate)
        .unwrap_or_else(|| predicate.to_string());
    key_canonical == predicate_canonical
}

fn subject_matches(knowledge: &dyn DialogueKnowledge, subject: &str, needle: &str) -> bool {
    if subject.contains(needle) {
        return true;
    }
    if knowledge
        .entity_type(subject)
        .map(|entity_type| entity_type.contains(needle))
        .unwrap_or(false)
    {
        return true;
    }
    entity_facts(knowledge, subject).iter().any(|(key, value)| {
        key.contains(needle)
            || value.contains(needle)
            || knowledge.predicate_label(key).contains(needle)
    })
}

fn render_frame(frame: &BoardFrame, revision: u64) -> String {
    let total = frame.rows.len();
    let pages = frame.page_count();
    let mut output = format!(
        "查询板 · 第 {} 页/共 {} 页 · {} · 共 {} 行",
        frame.page + 1,
        pages,
        frame.program,
        total
    );
    if frame.revision != revision {
        output.push_str(&format!(
            "\n[知识版本 v{} → v{}：以下结果基于旧版本，仅供对照；重新查询可重算。]",
            frame.revision, revision
        ));
    }
    if total == 0 {
        output.push_str("\n没有找到符合条件的结果。");
    } else {
        let page_size = frame.page_size.max(1);
        let start = frame.page.saturating_mul(page_size).min(total);
        let end = start.saturating_add(page_size).min(total);
        for index in start..end {
            let row = &frame.rows[index];
            let marker = if frame.selected == Some(index) {
                '→'
            } else {
                ' '
            };
            if row.predicate == "类型" {
                output.push_str(&format!(
                    "\n{} {}. {}（{}）",
                    marker,
                    index + 1,
                    row.subject,
                    row.value
                ));
            } else {
                output.push_str(&format!(
                    "\n{} {}. {} 的 {} = {}",
                    marker,
                    index + 1,
                    row.subject,
                    row.predicate,
                    row.value
                ));
            }
        }
    }
    if let Some(row) = frame.selected_row() {
        output.push_str(&format!("\n当前选中：{}（{}）", row.subject, row.predicate));
    }
    output.push_str("\n可说：第二个 / 下一页 / 只看… / 按…排序 / 跟…比 / 依据 / 回到上一步");
    output
}

fn render_comparison(
    left: &QueryRow,
    right: &QueryRow,
    _knowledge: &dyn DialogueKnowledge,
) -> String {
    let mut output = format!(
        "对比「{}」与「{}」的 {}：\n  {}：{}\n  {}：{}",
        left.subject,
        right.subject,
        left.predicate,
        left.subject,
        left.value,
        right.subject,
        right.value
    );
    let (Some(left_quantity), Some(right_quantity)) = (
        quantity::parse_quantity(&left.value),
        quantity::parse_quantity(&right.value),
    ) else {
        output.push_str("\n  数值不可比较（至少一侧不是数值）。");
        return output;
    };
    match left_quantity.compare(&right_quantity) {
        Comparison::Equal => output.push_str("\n  两者数值相同。"),
        Comparison::Less | Comparison::Greater => {
            let word = left_quantity
                .dimension()
                .comparative_word(&left.predicate);
            let difference = left_quantity
                .difference(&right_quantity)
                .map(|value| value.abs())
                .unwrap_or_default();
            let unit = if left_quantity.dimension() == Dimension::Count
                && !left_quantity.has_unit()
            {
                ""
            } else {
                left_quantity.dimension().canonical_unit()
            };
            let direction_word = match left_quantity.compare(&right_quantity) {
                Comparison::Greater => word,
                _ => comparison_lower_word(word),
            };
            output.push_str(&format!(
                "\n  「{}」比「{}」{} {}{}。",
                left.subject,
                right.subject,
                direction_word,
                quantity::format_number(difference),
                unit
            ));
        }
        Comparison::Uncertain => output.push_str(
            "\n  约数区间重叠，无法确定谁更大；上面分别保留了原始取值。",
        ),
        Comparison::Incomparable(reason) => {
            output.push_str(&format!(
                "\n  两者单位不同（{}），无法直接相减；上面分别保留了原始取值。",
                reason.describe()
            ));
        }
    }
    output
}

fn provenance_for(frame: &BoardFrame, row: Option<&QueryRow>) -> Vec<String> {
    let mut provenance = vec![format!(
        "查询板 · {} · 知识版本 v{}",
        frame.program, frame.revision
    )];
    if let Some(row) = row {
        provenance.push(row.evidence.clone());
    }
    provenance
}

#[derive(Debug, Clone)]
enum FilterSpec {
    Text(String),
    Numeric {
        column: Option<String>,
        op: CompareOp,
        threshold: Quantity,
    },
}

impl FilterSpec {
    fn describe(&self) -> String {
        match self {
            FilterSpec::Text(text) => text.clone(),
            FilterSpec::Numeric {
                column,
                op,
                threshold,
            } => format!(
                "{} {} {}",
                column.as_deref().unwrap_or("数值"),
                op.label(),
                threshold.raw()
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl CompareOp {
    /// 只接受确定的同量纲比较；`Uncertain`/`Incomparable` 一律不命中，
    /// 避免「约值区间重叠」被当成严格大于。
    fn matches(self, comparison: Comparison) -> bool {
        match (self, comparison) {
            (CompareOp::Lt, Comparison::Less) => true,
            (CompareOp::Le, Comparison::Less | Comparison::Equal) => true,
            (CompareOp::Gt, Comparison::Greater) => true,
            (CompareOp::Ge, Comparison::Greater | Comparison::Equal) => true,
            (CompareOp::Eq, Comparison::Equal) => true,
            (_, _) => false,
        }
    }

    fn label(self) -> &'static str {
        match self {
            CompareOp::Lt => "<",
            CompareOp::Le => "≤",
            CompareOp::Gt => ">",
            CompareOp::Ge => "≥",
            CompareOp::Eq => "=",
        }
    }
}

#[derive(Debug, Clone)]
struct SortSpec {
    column: Option<String>,
    descending: bool,
}

impl SortSpec {
    fn describe(&self) -> String {
        format!(
            "排序：{}{}",
            self.column.as_deref().unwrap_or("值"),
            if self.descending {
                "（降序）"
            } else {
                "（升序）"
            }
        )
    }
}

/// 极值/排名请求：属性可省略，方向与条数显式。
#[derive(Debug, Clone)]
struct ExtremeSpec {
    attribute: Option<String>,
    descending: bool,
    limit: usize,
}

fn describe_rank(
    attribute: &str,
    descending: bool,
    skipped: usize,
    knowledge: &dyn DialogueKnowledge,
) -> String {
    let label = knowledge.predicate_label(attribute);
    let direction = if descending {
        "从高到低"
    } else {
        "从低到高"
    };
    let note = if skipped > 0 {
        format!(" · {} 行量纲不同或非数值，未参与排名", skipped)
    } else {
        String::new()
    };
    format!("排名：{}（{}）{}", label, direction, note)
}

/// 比较级里「更低」的用词。`矮` 只用于「谁最矮」，差值句习惯说「低」。
fn comparison_lower_word(positive: &str) -> &'static str {
    match positive {
        "高" => "低",
        "重" => "轻",
        "大" => "小",
        "多" => "少",
        "长" => "短",
        "远" => "近",
        "快" => "慢",
        "年长" => "年轻",
        "久" => "短",
        _ => "小",
    }
}

fn superlative_word(dimension: Dimension, attribute: &str, descending: bool) -> &'static str {
    let positive = dimension.comparative_word(attribute);
    if descending {
        return positive;
    }
    match positive {
        "高" => "矮",
        "重" => "轻",
        "大" => "小",
        "多" => "少",
        "长" => "短",
        "远" => "近",
        "快" => "慢",
        "年长" => "年轻",
        "久" => "短",
        _ => "小",
    }
}

/// 生成排名结论。每一行都绑定原始取值；约数区间重叠时明确说「不确定」。
fn rank_summary(
    rows: &[QueryRow],
    limit: usize,
    attribute: &str,
    descending: bool,
    knowledge: &dyn DialogueKnowledge,
) -> String {
    let label = knowledge.predicate_label(attribute);
    let Some(first) = rows.first() else {
        return format!("没有可用于排名的「{}」记录。", label);
    };
    let Some(first_quantity) = quantity::parse_quantity(&first.value) else {
        return format!("「{}」的取值不是数值，无法排名。", label);
    };
    let word = superlative_word(first_quantity.dimension(), &label, descending);
    let limit = limit.max(1);

    if limit == 1 {
        let mut cohort = vec![first];
        let mut uncertain = false;
        for row in rows.iter().skip(1) {
            let Some(quantity) = quantity::parse_quantity(&row.value) else {
                break;
            };
            match first_quantity.compare(&quantity) {
                Comparison::Equal => cohort.push(row),
                Comparison::Uncertain => {
                    cohort.push(row);
                    uncertain = true;
                }
                _ => break,
            }
        }
        if uncertain {
            let names: Vec<&str> = cohort.iter().map(|row| row.subject.as_str()).collect();
            return format!(
                "「{}」的「{}」约数区间重叠，无法确定谁最{}。",
                names.join("」「"),
                label,
                word
            );
        }
        if cohort.len() > 1 {
            let names: Vec<&str> = cohort.iter().map(|row| row.subject.as_str()).collect();
            let values: Vec<&str> = cohort.iter().map(|row| row.value.as_str()).collect();
            return format!(
                "「{}」并列最{}（{}）。",
                names.join("」「"),
                word,
                values.join("、")
            );
        }
        return format!(
            "「{}」的「{}」最{}（{}）。",
            first.subject, label, word, first.value
        );
    }

    let mut output = format!(
        "按「{}」{}的前{}名：",
        label,
        if descending {
            "从高到低"
        } else {
            "从低到高"
        },
        limit
    );
    let mut rank = 1;
    let mut previous: Option<Quantity> = None;
    for (index, row) in rows.iter().take(limit).enumerate() {
        let quantity = quantity::parse_quantity(&row.value);
        let (shown_rank, tied, uncertain) = match (previous.as_ref(), quantity.as_ref()) {
            (Some(previous), Some(quantity)) => match previous.compare(quantity) {
                Comparison::Equal => (rank, true, false),
                Comparison::Uncertain => (rank, true, true),
                _ => {
                    rank = index + 1;
                    (rank, false, false)
                }
            },
            _ => (index + 1, false, false),
        };
        let marker = if uncertain {
            "（约数，不确定）"
        } else if tied {
            "（并列）"
        } else {
            ""
        };
        output.push_str(&format!(
            "\n  {}. {}{}（{}）",
            shown_rank, row.subject, marker, row.value
        ));
        previous = quantity;
    }
    output
}

/// 把「身高」这类短答解析成图谱中的谓词键；找不到就返回 None。
fn resolve_attribute_hint(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    let cleaned = text.trim_matches(|ch: char| {
        matches!(ch, '。' | '，' | ',' | '！' | '!' | '？' | '?' | ' ' | '的')
    });
    if cleaned.is_empty() {
        return None;
    }
    let mut attributes: BTreeMap<String, String> = BTreeMap::new();
    for entity in knowledge.all_entities() {
        for (key, _) in knowledge.entity_attributes(&entity) {
            attributes
                .entry(key.clone())
                .or_insert_with(|| knowledge.predicate_label(&key));
        }
        for (key, _) in knowledge.entity_relations(&entity) {
            attributes
                .entry(key.clone())
                .or_insert_with(|| knowledge.predicate_label(&key));
        }
    }
    if let Some(canonical) = knowledge.canonical_predicate(cleaned) {
        if attributes.contains_key(&canonical) {
            return Some(canonical);
        }
    }
    for (key, label) in &attributes {
        if key == cleaned || label == cleaned || cleaned.contains(label.as_str()) {
            return Some(key.clone());
        }
    }
    None
}

fn has_rank_interrogative(text: &str) -> bool {
    ["谁", "哪个", "哪个人", "哪位", "哪些", "什么"]
        .iter()
        .any(|marker| text.contains(marker))
}

fn parse_extreme(text: &str, board_active: bool) -> Option<ExtremeSpec> {
    if let Some((limit, attribute)) = parse_top_n(text) {
        return Some(ExtremeSpec {
            attribute,
            descending: true,
            limit,
        });
    }
    let position = text.find('最')?;
    let after = &text[position + '最'.len_utf8()..];
    let (descending, consumed) = if after.starts_with("年长") {
        (true, "年长".len())
    } else if after.starts_with("年轻") {
        (false, "年轻".len())
    } else {
        let character = after.chars().next()?;
        let descending = match character {
            '高' | '大' | '重' | '多' | '长' | '远' | '快' | '强' | '贵' | '早' | '深'
            | '厚' | '宽' => true,
            '矮' | '小' | '轻' | '少' | '短' | '近' | '慢' | '弱' | '晚' | '浅' | '薄'
            | '窄' | '便' => false,
            _ => return None,
        };
        (descending, character.len_utf8())
    };
    if !board_active && !has_rank_interrogative(text) {
        return None;
    }

    let mut candidate = text[..position].trim();
    while let Some(stripped) = candidate.strip_suffix('的') {
        candidate = stripped.trim_end();
    }
    for marker in ["请问", "那么", "到底", "究竟", "帮我看看", "看看"] {
        if let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped.trim_start();
        }
    }
    for marker in [
        "哪几个人",
        "哪个人",
        "哪一位",
        "哪几个",
        "哪位",
        "哪个",
        "谁",
        "什么",
    ] {
        while let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped.trim_start_matches(['的', ' ']).trim();
        }
    }
    let attribute = if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    };
    let limit = parse_rank_limit(&after[consumed..]).unwrap_or(1);
    Some(ExtremeSpec {
        attribute,
        descending,
        limit,
    })
}

fn parse_top_n(text: &str) -> Option<(usize, Option<String>)> {
    let position = text.find('前')?;
    let after = &text[position + '前'.len_utf8()..];
    let (limit, consumed) = leading_count(after)?;
    let suffix = &after[consumed..];
    if !(suffix.is_empty()
        || suffix.starts_with(['名', '个', '位', '条', '项', '种', '人', '是']))
    {
        return None;
    }
    let mut candidate = text[..position].trim();
    while let Some(stripped) = candidate.strip_suffix('的') {
        candidate = stripped.trim_end();
    }
    for marker in ["哪几个人", "哪个人", "哪一位", "哪几个", "哪位", "哪个", "谁"] {
        while let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped.trim_start_matches(['的', ' ']).trim();
        }
    }
    let attribute = if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    };
    Some((limit, attribute))
}

fn parse_rank_limit(after: &str) -> Option<usize> {
    let rest = after.trim_start_matches(['的', ' ']);
    let (limit, consumed) = leading_count(rest)?;
    let suffix = &rest[consumed..];
    if suffix.starts_with(['个', '名', '位', '条', '项', '种', '人']) {
        Some(limit)
    } else {
        None
    }
}

fn leading_count(text: &str) -> Option<(usize, usize)> {
    let mut end = 0;
    for (index, character) in text.char_indices() {
        let is_number = character.is_ascii_digit()
            || matches!(
                character,
                '零' | '〇'
                    | '一'
                    | '二'
                    | '两'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '十'
                    | '百'
                    | '千'
            );
        if is_number {
            end = index + character.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let quantity = quantity::parse_quantity(&text[..end])?;
    let value = quantity.base_value();
    if value <= 0.0 || value.fract().abs() > 1e-9 || value > 100.0 {
        return None;
    }
    Some((value as usize, end))
}

fn sort_value(row: &QueryRow, sort: &SortSpec, _knowledge: &dyn DialogueKnowledge) -> String {
    if let Some(column) = &sort.column {
        if row.predicate.contains(column) || column.contains(row.predicate.as_str()) {
            return row.value.clone();
        }
    }
    row.value.clone()
}

/// 按数量语义排序：同量纲内换算后比较，跨量纲/非数值行显式排到末尾。
///
/// 返回未参与数值排序的行数，调用方把它写进程序描述，避免「看起来排过序」
/// 却把不可比值混进名次。
fn sort_rows_by_quantity(
    rows: &mut Vec<QueryRow>,
    sort: &SortSpec,
    knowledge: &dyn DialogueKnowledge,
) -> usize {
    let mut parsed: Vec<(QueryRow, Quantity)> = Vec::new();
    let mut unparsed: Vec<QueryRow> = Vec::new();
    for row in rows.drain(..) {
        let raw = sort_value(&row, sort, knowledge);
        match quantity::parse_quantity(&raw) {
            Some(value) => parsed.push((row, value)),
            None => unparsed.push(row),
        }
    }
    let outcome = quantity::rank(
        parsed,
        &RankSpec {
            descending: sort.descending,
            offset: 0,
            limit: usize::MAX,
        },
    );
    let skipped = outcome.skipped.len() + unparsed.len();
    let mut sorted: Vec<QueryRow> = outcome.entries.into_iter().map(|entry| entry.item).collect();
    sorted.extend(outcome.skipped);
    sorted.extend(unparsed);
    *rows = sorted;
    skipped
}

fn clean(input: &str) -> String {
    input
        .trim()
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | '，' | ',' | '；' | ';' | '~' | ' '
            )
        })
        .trim()
        .to_string()
}

fn is_undo(text: &str) -> bool {
    matches!(
        text,
        "回到上一步" | "上一步" | "撤销" | "回退" | "撤销上一步" | "退回一步" | "往回一步"
    )
}

fn is_reset(text: &str) -> bool {
    matches!(
        text,
        "重新开始" | "清空查询" | "清空查询板" | "新查询" | "清空" | "重置查询"
    )
}

fn is_explain(text: &str) -> bool {
    [
        "依据是什么",
        "有什么依据",
        "依据",
        "证据呢",
        "证据",
        "出处",
        "来源是什么",
        "怎么得出",
        "根据什么",
        "为什么",
        "凭什么",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn is_count(text: &str) -> bool {
    [
        "一共多少",
        "总共多少",
        "有多少条",
        "多少条",
        "有几条",
        "几条",
        "总数",
        "共几",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn is_next_page(text: &str) -> bool {
    matches!(
        text,
        "下一页"
            | "下页"
            | "翻页"
            | "还有吗"
            | "还有呢"
            | "还有别的吗"
            | "还有其它吗"
            | "还有其他的吗"
            | "继续看"
            | "再多看几个"
            | "更多"
            | "往后翻"
            | "再下一页"
    )
}

fn is_prev_page(text: &str) -> bool {
    matches!(text, "上一页" | "上页" | "往前翻" | "前一页")
}

fn is_identity_request(text: &str) -> bool {
    matches!(
        text,
        "他是谁"
            | "她是谁"
            | "它是什么"
            | "这个人是谁"
            | "这个实体是什么"
            | "介绍一下"
            | "介绍这个"
            | "这个呢"
            | "他呢"
            | "她呢"
            | "它呢"
    )
}

fn parse_scan(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<ScanSpec> {
    if !is_scan_request(text) {
        return None;
    }
    let body = strip_scan_prefix(text);
    if leading_entity(&body, knowledge).is_some() {
        return None;
    }
    let entity_type = detect_type(text, knowledge);
    let predicate = extract_predicate(text);
    Some(ScanSpec {
        predicate,
        entity_type,
    })
}

fn is_scan_request(text: &str) -> bool {
    const STRONG: [&str; 6] = ["列出", "列举", "罗列", "帮我列出", "所有", "全部"];
    if STRONG.iter().any(|prefix| text.starts_with(prefix)) {
        return true;
    }
    const SOFT: [&str; 6] = ["显示", "查看", "看看", "看一下", "给我看看", "请列出"];
    if SOFT.iter().any(|prefix| text.starts_with(prefix)) {
        return text.contains("所有")
            || text.contains("全部")
            || text.contains("有哪些")
            || text.contains("哪些有")
            || text.contains("哪些实体")
            || text.ends_with("列表")
            || text.ends_with("清单");
    }
    if text.contains("有哪些")
        || text.contains("哪些有")
        || text.contains("哪些实体")
        || text.contains("哪几个实体")
        || text.contains("哪些人")
        || text.contains("哪几个人")
    {
        return true;
    }
    text.ends_with("列表") || text.ends_with("清单")
}

fn strip_scan_prefix(text: &str) -> String {
    let mut body = text.to_string();
    for prefix in [
        "帮我列出",
        "给我看看",
        "请列出",
        "列出",
        "列举",
        "罗列",
        "显示",
        "查看",
        "看看",
        "看一下",
    ] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim().to_string();
            break;
        }
    }
    body
}

/// 请求以具体实体开头时（如「列出姚明的身高」），交给实体问答而不是全表扫描。
fn leading_entity(body: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    let mut best: Option<String> = None;
    for entity in knowledge.all_entities() {
        if entity.is_empty() {
            continue;
        }
        let Some(rest) = body.strip_prefix(&entity) else {
            continue;
        };
        let rest = rest.trim_start();
        if !(rest.is_empty()
            || rest.starts_with(['的', '有', '是', '在', '和', '与', '跟', '，', ',']))
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|current| current.chars().count() < entity.chars().count())
        {
            best = Some(entity);
        }
    }
    best
}

fn detect_type(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    use crate::symbolic_generator::localized_type_label;
    let types = knowledge.all_types();
    for entity_type in &types {
        if text.contains(entity_type.as_str()) {
            return Some(entity_type.clone());
        }
    }
    for entity_type in &types {
        let label = localized_type_label(entity_type);
        if label != entity_type.as_str() && text.contains(label) {
            return Some(entity_type.clone());
        }
    }
    const ALIASES: [(&str, &str); 12] = [
        ("人物", "Person"),
        ("人", "Person"),
        ("公司", "Company"),
        ("企业", "Company"),
        ("城市", "City"),
        ("国家", "Country"),
        ("电影", "Film"),
        ("书籍", "Book"),
        ("学校", "School"),
        ("组织", "Organization"),
        ("球员", "Person"),
        ("运动员", "Person"),
    ];
    for (alias, canonical) in ALIASES {
        if !text.contains(alias) {
            continue;
        }
        if types.iter().any(|entity_type| entity_type == canonical) {
            return Some(canonical.to_string());
        }
        let wanted = localized_type_label(canonical);
        if let Some(found) = types
            .iter()
            .find(|entity_type| localized_type_label(entity_type) == wanted)
        {
            return Some(found.clone());
        }
    }
    None
}

fn extract_predicate(text: &str) -> Option<String> {
    let mut body = strip_scan_prefix(text);
    for trim in ["一下", "所有", "全部", "每个", "各个", "各"] {
        while let Some(rest) = body.strip_prefix(trim) {
            body = rest.trim().to_string();
        }
    }

    let mut candidate = None;
    if let Some((_, rest)) = body.split_once("有哪些") {
        candidate = Some(rest.to_string());
    } else if let Some((_, rest)) = body.split_once("哪些有") {
        candidate = Some(rest.to_string());
    } else if let Some(index) = body.find("哪些") {
        let rest = &body[index + "哪些".len()..];
        if let Some((_, tail)) = rest.split_once('有') {
            candidate = Some(tail.to_string());
        }
    }

    let mut candidate = candidate.unwrap_or_else(|| {
        if let Some((_, tail)) = body.split_once('有') {
            if !tail.is_empty() && !tail.starts_with("哪些") {
                tail.to_string()
            } else {
                body.clone()
            }
        } else {
            body.clone()
        }
    });

    for suffix in ["的实体", "的人", "的记录", "的条目"] {
        if let Some(stripped) = candidate.strip_suffix(suffix) {
            candidate = stripped.trim().to_string();
        }
    }
    if let Some((_, tail)) = candidate.rsplit_once('的') {
        candidate = tail.to_string();
    }
    let candidate = candidate
        .trim()
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | '吗' | '呢' | '吧' | '啊' | '的' | ' '
            )
        })
        .trim();
    for suffix in ["是什么", "是多少", "有哪些", "有谁", "列表", "清单"] {
        if let Some(stripped) = candidate.strip_suffix(suffix) {
            return normalize_predicate(stripped);
        }
    }
    normalize_predicate(candidate)
}

fn normalize_predicate(candidate: &str) -> Option<String> {
    let mut candidate = candidate
        .trim()
        .trim_matches(|c: char| matches!(c, '「' | '」' | '"' | '\'' | '“' | '”'))
        .trim();
    // 「哪些实体」「哪几个人」这类整表问法没有谓词，归一到具体谓词或 None；
    // 否则「哪些实体」会被当成谓词去查值，结果永远为空。
    for prefix in [
        "哪些实体",
        "哪几个实体",
        "哪些",
        "哪几个",
        "哪几",
        "所有的",
        "全体",
    ] {
        if let Some(rest) = candidate.strip_prefix(prefix) {
            candidate = rest.trim();
        }
    }
    if candidate.is_empty()
        || matches!(
            candidate,
            "实体" | "东西" | "记录" | "数据" | "内容" | "人" | "人物" | "条目" | "成员"
        )
    {
        return None;
    }
    Some(candidate.to_string())
}

fn parse_filter(text: &str) -> Option<FilterSpec> {
    let mut rest = None;
    for prefix in ["只看", "只要", "筛选", "过滤", "限定", "其中"] {
        if let Some(stripped) = text.strip_prefix(prefix) {
            rest = Some(stripped.trim());
            break;
        }
    }
    let rest = rest?;
    let rest = rest
        .trim_start_matches(['是', '有', '的', ' '])
        .trim_end_matches(['的', '呢', '吧', ' '])
        .trim();
    if rest.is_empty() {
        return None;
    }
    if let Some(numeric) = parse_numeric_filter(rest) {
        return Some(numeric);
    }
    Some(FilterSpec::Text(rest.to_string()))
}

fn parse_numeric_filter(text: &str) -> Option<FilterSpec> {
    const OPERATORS: [(&str, CompareOp); 18] = [
        ("不低于", CompareOp::Ge),
        ("不少于", CompareOp::Ge),
        ("不小于", CompareOp::Ge),
        ("不超过", CompareOp::Le),
        ("不大于", CompareOp::Le),
        ("大于等于", CompareOp::Ge),
        ("小于等于", CompareOp::Le),
        ("高于", CompareOp::Gt),
        ("低于", CompareOp::Lt),
        ("超过", CompareOp::Gt),
        ("大于", CompareOp::Gt),
        ("小于", CompareOp::Lt),
        ("至少", CompareOp::Ge),
        ("至多", CompareOp::Le),
        ("等于", CompareOp::Eq),
        ("不到", CompareOp::Lt),
        (">=", CompareOp::Ge),
        ("<=", CompareOp::Le),
    ];
    for (marker, op) in OPERATORS {
        if let Some((column, value)) = text.split_once(marker) {
            if let Some(threshold) = quantity::parse_quantity(value) {
                let column = column
                    .trim()
                    .trim_end_matches('的')
                    .trim_end_matches("数值")
                    .trim();
                return Some(FilterSpec::Numeric {
                    column: if column.is_empty() {
                        None
                    } else {
                        Some(column.to_string())
                    },
                    op,
                    threshold,
                });
            }
        }
    }
    for (marker, op) in [
        (">", CompareOp::Gt),
        ("<", CompareOp::Lt),
        ("=", CompareOp::Eq),
    ] {
        if let Some((column, value)) = text.split_once(marker) {
            if let Some(threshold) = quantity::parse_quantity(value) {
                let column = column.trim().trim_end_matches('的').trim();
                return Some(FilterSpec::Numeric {
                    column: if column.is_empty() {
                        None
                    } else {
                        Some(column.to_string())
                    },
                    op,
                    threshold,
                });
            }
        }
    }
    None
}

fn parse_sort(text: &str) -> Option<SortSpec> {
    let descending = text.contains("从高到低")
        || text.contains("从大到小")
        || text.contains("降序")
        || text.contains("倒序");
    let ascending = text.contains("从低到高")
        || text.contains("从小到大")
        || text.contains("升序")
        || text.contains("正序");
    let mut column = None;
    if let Some(index) = text.find('按') {
        let rest = &text[index + '按'.len_utf8()..];
        let mut end = rest.len();
        for marker in [
            "从低到高",
            "从高到低",
            "从小到大",
            "从大到小",
            "升序",
            "降序",
            "倒序",
            "正序",
            "排序",
            "排一下",
            "排",
            "呢",
            "吧",
            "。",
        ] {
            if let Some(position) = rest.find(marker) {
                end = end.min(position);
            }
        }
        let candidate = rest[..end].trim().trim_end_matches('的').trim();
        if !candidate.is_empty() && !matches!(candidate, "照" | "顺序") {
            column = Some(candidate.to_string());
        }
    } else if !(text.contains("排序") || text.contains("排一下") || text.ends_with('排')) {
        return None;
    }
    if !(text.contains('按')
        || ascending
        || descending
        || text.contains("排序")
        || text.ends_with('排'))
    {
        return None;
    }
    Some(SortSpec { column, descending })
}

fn parse_compare(text: &str) -> Option<String> {
    for prefix in ["跟", "和", "与", "同"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            let mut other = rest.trim();
            for suffix in ["相比呢", "相比", "比较呢", "比较", "比呢", "比", "呢"] {
                if let Some(stripped) = other.strip_suffix(suffix) {
                    other = stripped.trim();
                    break;
                }
            }
            if !other.is_empty() {
                return Some(other.to_string());
            }
        }
    }
    None
}

fn parse_ordinal(text: &str) -> Option<usize> {
    let index = text.find('第')?;
    let rest = &text[index + '第'.len_utf8()..];
    let digits: String = rest
        .chars()
        .take_while(|c| {
            !matches!(
                c,
                '个' | '条'
                    | '行'
                    | '位'
                    | '页'
                    | '呢'
                    | '的'
                    | ' '
                    | '高'
                    | '矮'
                    | '大'
                    | '小'
                    | '重'
                    | '轻'
                    | '多'
                    | '少'
                    | '长'
                    | '短'
                    | '远'
                    | '近'
                    | '快'
                    | '慢'
            )
        })
        .collect();
    chinese_number(digits.trim()).map(|ordinal| ordinal.saturating_sub(1))
}

fn parse_page_number(text: &str) -> Option<usize> {
    let index = text.find('第')?;
    let rest = &text[index + '第'.len_utf8()..];
    let digits: String = rest
        .chars()
        .take_while(|c| !matches!(c, '页' | '呢' | '的' | ' '))
        .collect();
    if !rest.chars().any(|c| c == '页') {
        return None;
    }
    let page = chinese_number(digits.trim())?;
    Some(page.saturating_sub(1))
}

fn parse_select_entity(text: &str) -> Option<String> {
    for prefix in ["选择", "切换到", "切到", "选", "看"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            let candidate = rest.trim().trim_end_matches(['呢', '吧', '的', ' ']).trim();
            if !candidate.is_empty()
                && !matches!(
                    candidate,
                    "一下" | "看" | "这个" | "那个" | "第一个" | "第二个"
                )
            {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

fn chinese_number(raw: &str) -> Option<usize> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    if text.chars().all(|c| c.is_ascii_digit()) {
        return text.parse().ok();
    }
    let mut total = 0usize;
    let mut current = 0usize;
    let mut seen = false;
    for character in text.chars() {
        match character {
            '零' => {
                seen = true;
            }
            '一' => {
                current = 1;
                seen = true;
            }
            '二' | '两' => {
                current = 2;
                seen = true;
            }
            '三' => {
                current = 3;
                seen = true;
            }
            '四' => {
                current = 4;
                seen = true;
            }
            '五' => {
                current = 5;
                seen = true;
            }
            '六' => {
                current = 6;
                seen = true;
            }
            '七' => {
                current = 7;
                seen = true;
            }
            '八' => {
                current = 8;
                seen = true;
            }
            '九' => {
                current = 9;
                seen = true;
            }
            '十' => {
                if current == 0 {
                    current = 1;
                }
                total += current * 10;
                current = 0;
                seen = true;
            }
            '百' => {
                if current == 0 {
                    current = 1;
                }
                total += current * 100;
                current = 0;
                seen = true;
            }
            _ => return None,
        }
    }
    if !seen {
        return None;
    }
    Some(total + current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct TestKnowledge {
        entities: Vec<String>,
        types: Vec<(String, String)>,
        attributes: HashMap<String, Vec<(String, String)>>,
        relations: HashMap<String, Vec<(String, String)>>,
    }

    impl TestKnowledge {
        fn new() -> Self {
            let mut knowledge = TestKnowledge::default();
            for (entity, entity_type, height, country) in [
                ("姚明", "Person", "226", "中国"),
                ("王治郅", "Person", "214", "中国"),
                ("易建联", "Person", "213", "中国"),
                ("诺维茨基", "Person", "213", "德国"),
                ("加索尔", "Person", "216", "西班牙"),
                ("奥尼尔", "Person", "216", "美国"),
            ] {
                knowledge.entities.push(entity.to_string());
                knowledge
                    .types
                    .push((entity.to_string(), entity_type.to_string()));
                knowledge.attributes.insert(
                    entity.to_string(),
                    vec![
                        ("身高".to_string(), height.to_string()),
                        ("国籍".to_string(), country.to_string()),
                    ],
                );
                knowledge.relations.insert(
                    entity.to_string(),
                    vec![("球队".to_string(), "火箭队".to_string())],
                );
            }
            knowledge
        }
    }

    impl DialogueKnowledge for TestKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.entities.iter().any(|candidate| candidate == entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            let mut keys: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
                .unwrap_or_default();
            keys.sort();
            keys.dedup();
            keys
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            let mut values: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| {
                    pairs
                        .iter()
                        .filter(|(key, _)| key == attribute)
                        .map(|(_, value)| value.clone())
                        .collect()
                })
                .unwrap_or_default();
            values.sort();
            values.dedup();
            values
        }

        fn all_entities(&self) -> Vec<String> {
            let mut entities = self.entities.clone();
            entities.sort();
            entities
        }

        fn all_types(&self) -> Vec<String> {
            let mut types: Vec<String> = self.types.iter().map(|(_, t)| t.clone()).collect();
            types.sort();
            types.dedup();
            types
        }

        fn entities_of_type(&self, entity_type: &str) -> Vec<String> {
            let mut entities: Vec<String> = self
                .types
                .iter()
                .filter(|(_, t)| t == entity_type)
                .map(|(entity, _)| entity.clone())
                .collect();
            entities.sort();
            entities
        }

        fn entity_type(&self, entity: &str) -> Option<String> {
            self.types
                .iter()
                .find(|(candidate, _)| candidate == entity)
                .map(|(_, entity_type)| entity_type.clone())
        }

        fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
            self.attributes.get(entity).cloned().unwrap_or_default()
        }

        fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
            self.relations.get(entity).cloned().unwrap_or_default()
        }

        fn describe(&self, entity: &str) -> Option<String> {
            self.known_entity(entity)
                .then(|| format!("{}是测试实体。", entity))
        }

        fn revision(&self) -> u64 {
            7
        }
    }

    fn board() -> QueryBoard {
        QueryBoard::new()
    }

    #[test]
    fn scan_and_select_by_ordinal() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        let first = board
            .apply("列出所有人的身高", &knowledge, 7)
            .expect("应接管集合查询");
        assert!(first.text.contains("姚明"));
        let frame = board.current().expect("应留下查询帧");
        assert_eq!(frame.rows.len(), 6);

        let expected = frame.rows[1].subject.clone();
        let second = board
            .apply("第二个呢", &knowledge, 7)
            .expect("应选中第二行");
        assert_eq!(
            board
                .current()
                .and_then(|frame| frame.selected_row())
                .map(|row| row.subject.as_str()),
            Some(expected.as_str())
        );
        assert!(second.text.contains("当前选中"));
    }

    #[test]
    fn filter_sort_and_compare_keep_evidence() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("只看中国的", &knowledge, 7).unwrap();
        assert_eq!(board.current().unwrap().rows.len(), 3);

        board.apply("按身高从高到低排", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.first().unwrap().subject, "姚明");
        assert_eq!(frame.rows.last().unwrap().subject, "易建联");

        let compare = board.apply("跟易建联比呢", &knowledge, 7).unwrap();
        assert!(compare.text.contains("姚明"));
        assert!(compare.text.contains("易建联"));
        assert!(compare.text.contains("高 13"));
        assert!(!compare.provenance.is_empty());
    }

    #[test]
    fn pagination_undo_and_stale_revision() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let next = board.apply("下一页", &knowledge, 7).unwrap();
        assert!(next.text.contains("第 2 页"));
        let undone = board.apply("回到上一步", &knowledge, 7).unwrap();
        assert!(undone.text.contains("第 1 页"));
        assert_eq!(board.cursor(), Some(0));
        assert_eq!(board.frame_count(), 2);

        let stale = board.apply("依据", &knowledge, 9).unwrap();
        assert!(stale.text.contains("v7 → v9"));
    }

    #[test]
    fn numeric_filter_and_missing_records() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("只看身高超过215", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.len(), 3);
        assert!(frame
            .rows
            .iter()
            .all(|row| row.value.parse::<i64>().unwrap() > 215));

        let missing = board.apply("跟不存在比呢", &knowledge, 7).unwrap();
        assert!(missing.text.contains("没有记录"));
    }

    #[test]
    fn board_snapshot_round_trip_and_reset() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有实体", &knowledge, 7).unwrap();
        board.apply("只看中国", &knowledge, 7).unwrap();
        let snapshot = board.clone();
        let mut restored = snapshot;
        assert_eq!(restored.current().unwrap().rows.len(), 3);
        let reset = restored.apply("重新开始", &knowledge, 7).unwrap();
        assert!(reset.text.contains("已清空"));
        assert!(!restored.is_active());
    }

    #[test]
    fn localized_type_names_still_scan() {
        let mut knowledge = TestKnowledge::new();
        for (_, entity_type) in &mut knowledge.types {
            *entity_type = "人物".to_string();
        }
        let mut board = board();
        let scan = board
            .apply("列出所有人", &knowledge, 7)
            .expect("中文本体类型也应能扫描");
        assert!(scan.text.contains("姚明"));
        assert_eq!(board.current().unwrap().rows.len(), 6);
    }

    #[test]
    fn whole_table_phrases_scan_every_entity() {
        let knowledge = TestKnowledge::new();
        for phrase in ["哪些实体", "有哪些实体", "哪几个实体", "列出所有实体"] {
            let mut board = board();
            let scan = board
                .apply(phrase, &knowledge, 7)
                .unwrap_or_else(|| panic!("「{}」应触发整表扫描", phrase));
            assert!(
                scan.text.contains("姚明"),
                "「{}」的结果缺少实体：{}",
                phrase,
                scan.text
            );
            let frame = board.current().expect("应留下查询帧");
            assert_eq!(frame.rows.len(), 6, "「{}」应扫描全部实体", phrase);
            assert!(frame.rows.iter().all(|row| row.predicate == "类型"));
        }
    }

    #[test]
    fn predicate_phrase_still_scans_after_whole_table_trim() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board
            .apply("哪些实体有身高", &knowledge, 7)
            .expect("带谓词的整表问法应扫描身高");
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.len(), 6);
        assert!(frame.rows.iter().all(|row| row.predicate == "身高"));
    }

    #[test]
    fn empty_board_defers_normal_questions() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        assert!(board.apply("姚明有多高", &knowledge, 7).is_none());
        assert!(board.apply("姚明的所有孩子是谁", &knowledge, 7).is_none());
        assert!(board.apply("显示姚明的身高", &knowledge, 7).is_none());
        assert!(board.apply("列出姚明的身高", &knowledge, 7).is_none());
        assert!(board.apply("你好", &knowledge, 7).is_none());
    }

    #[test]
    fn chinese_ordinals_cover_common_forms() {
        assert_eq!(chinese_number("一"), Some(1));
        assert_eq!(chinese_number("十"), Some(10));
        assert_eq!(chinese_number("十二"), Some(12));
        assert_eq!(chinese_number("二十"), Some(20));
        assert_eq!(chinese_number("二十三"), Some(23));
        assert_eq!(chinese_number("3"), Some(3));
        assert_eq!(parse_page_number("第2页"), Some(1));
    }

    #[test]
    fn compare_op_matches_thresholds() {
        assert!(CompareOp::Gt.matches(Comparison::Greater));
        assert!(CompareOp::Le.matches(Comparison::Equal));
        assert!(CompareOp::Eq.matches(Comparison::Equal));
        assert!(!CompareOp::Gt.matches(Comparison::Uncertain));
        assert!(!CompareOp::Ge.matches(Comparison::Incomparable(
            crate::quantity::Incomparable::NotAQuantity
        )));
    }

    #[test]
    fn quantity_normalizes_common_units_and_refuses_mixed_units() {
        let left = quantity::parse_quantity("2.26米").unwrap();
        let right = quantity::parse_quantity("226厘米").unwrap();
        assert_eq!(left.dimension(), right.dimension());
        assert!((left.base_value() - right.base_value()).abs() < 1e-9);

        let weight = quantity::parse_quantity("100斤").unwrap();
        assert_eq!(weight.dimension(), Dimension::Mass);
        assert!((weight.base_value() - 50.0).abs() < 1e-9);

        let left = QueryRow::new("甲", "身高", "226");
        let right = QueryRow::new("乙", "身高", "2.26米");
        let text = render_comparison(&left, &right, &TestKnowledge::new());
        assert!(text.contains("单位不同"));
    }

    #[test]
    fn sort_uses_unit_conversion_not_raw_numbers() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "2.26米"),
            ("王治郅", "214厘米"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "2.16米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("按身高从高到低排", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.first().unwrap().subject, "姚明");
        assert_eq!(frame.rows.get(1).unwrap().subject, "加索尔");
        let values: Vec<f64> = frame
            .rows
            .iter()
            .map(|row| quantity::parse_quantity(&row.value).unwrap().base_value())
            .collect();
        assert!(
            values.windows(2).all(|pair| pair[0] >= pair[1]),
            "换算后应按米降序：{:?}",
            values
        );
        assert!((values.last().copied().unwrap() - 2.13).abs() < 1e-9);
    }

    #[test]
    fn rank_answers_who_is_highest_and_second_highest() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let highest = board.apply("谁最高", &knowledge, 7).expect("应回答最高者");
        assert!(highest.text.contains("姚明"), "{}", highest.text);
        assert!(highest.text.contains("最"), "{}", highest.text);
        assert!(highest
            .provenance
            .iter()
            .any(|note| note.contains("226")));
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::Rank {
                descending: true,
                ..
            }
        ));

        let second = board.apply("第二高呢", &knowledge, 7).unwrap();
        assert_eq!(
            board
                .current()
                .and_then(|frame| frame.selected_row())
                .map(|row| row.subject.as_str()),
            Some("加索尔")
        );
        assert!(second.text.contains("当前选中"));

        let top3 = board.apply("前三名", &knowledge, 7).unwrap();
        assert!(top3.text.contains("姚明"), "{}", top3.text);
        assert!(top3.text.contains("加索尔"), "{}", top3.text);
        assert!(top3.text.contains("王治郅"), "{}", top3.text);

        let why = board.apply("为什么姚明最高", &knowledge, 7).unwrap();
        assert!(why.text.contains("证据"), "{}", why.text);
    }

    #[test]
    fn rank_without_attribute_asks_once_then_resolves_short_answer() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        let pending = board.apply("谁最高", &knowledge, 7).expect("应发起属性澄清");
        assert!(pending.text.contains("哪个属性"), "{}", pending.text);
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::RankPending { descending: true }
        ));

        let resolved = board
            .apply("身高", &knowledge, 7)
            .expect("短答属性应完成排名");
        assert!(resolved.text.contains("姚明"), "{}", resolved.text);
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::Rank { .. }
        ));
    }

    #[test]
    fn rank_skips_cross_dimension_values() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "2.26米"),
            ("王治郅", "67万平方公里"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "2.16米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let outcome = board.apply("谁最高", &knowledge, 7).unwrap();
        assert!(outcome.text.contains("姚明"), "{}", outcome.text);
        assert!(
            outcome.text.contains("未参与排名"),
            "跨量纲行必须显式说明：{}",
            outcome.text
        );
        assert_eq!(board.current().unwrap().rows.first().unwrap().subject, "姚明");
    }
}

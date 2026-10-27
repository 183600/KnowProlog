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
            let mut text =
                "你想按哪个属性排名？例如「谁的身高最高」或直接回答「身高」。".to_string();
            let candidates = rankable_attributes(knowledge);
            if !candidates.is_empty() {
                text.push_str(&format!(
                    "\n当前可排名的属性：{}。",
                    candidates
                        .iter()
                        .take(6)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                ));
            }
            return Some(BoardOutcome {
                text,
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
                let ranking = rank_rows(
                    matching,
                    &SortSpec {
                        column: Some(attribute.clone()),
                        descending: spec.descending,
                    },
                    knowledge,
                );
                let skipped = ranking.skipped;
                let summary = rank_summary(
                    &ranking.ranked,
                    spec.limit,
                    &attribute,
                    spec.descending,
                    knowledge,
                );
                let mut next = frame;
                next.rows = ranking.rows;
                next.selected = None;
                next.page = 0;
                next.op = "rank".to_string();
                next.source = QuerySource::Rank {
                    predicate: attribute.clone(),
                    descending: spec.descending,
                };
                next.program = describe_rank(&attribute, spec.descending, skipped, knowledge);
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
        mut attribute: String,
        descending: bool,
        limit: usize,
        knowledge: &dyn DialogueKnowledge,
        revision: u64,
    ) -> Option<BoardOutcome> {
        let spec = ScanSpec {
            predicate: Some(attribute.clone()),
            entity_type: None,
        };
        let mut rows = scan_rows(&spec, knowledge);
        if rows.is_empty() {
            // 「哪个城市面积最大」里的属性词面是「城市面积」，图谱谓词却是
            // 「面积」；只在原词面查不到时做一次受控归一，不猜新谓词。
            if let Some(resolved) = resolve_attribute_hint(&attribute, knowledge) {
                if resolved != attribute {
                    attribute = resolved;
                    rows = scan_rows(
                        &ScanSpec {
                            predicate: Some(attribute.clone()),
                            entity_type: None,
                        },
                        knowledge,
                    );
                }
            }
        }
        if rows.is_empty() {
            return None;
        }
        let ranking = rank_rows(
            rows,
            &SortSpec {
                column: Some(attribute.clone()),
                descending,
            },
            knowledge,
        );
        let skipped = ranking.skipped;
        let summary = rank_summary(
            &ranking.ranked,
            limit,
            &attribute,
            descending,
            knowledge,
        );
        let frame = BoardFrame::new(
            "rank",
            describe_rank(&attribute, descending, skipped, knowledge),
            QuerySource::Rank {
                predicate: attribute.clone(),
                descending,
            },
            ranking.rows,
            revision,
        );
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
        knowledge: &dyn DialogueKnowledge,
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
        let mut provenance = vec![
            format!(
                "查询板第 {} 帧：{}",
                self.cursor.map_or(0, |i| i + 1),
                frame.program
            ),
            row.map(|row| row.evidence.clone())
                .unwrap_or_else(|| "无结果行".to_string()),
        ];
        // 排名帧不只报「出处」，还要给出与相邻名次的取值和差值。
        if let Some(reason) = rank_reason(frame, knowledge) {
            text.push_str("\n");
            text.push_str(&reason.text);
            provenance.extend(reason.evidence);
        }
        if row.is_some() && frame.revision != revision {
            text.push_str(&format!(
                "\n[知识版本 v{} → v{}：该证据来自旧版本，当前查询应重新计算。]",
                frame.revision, revision
            ));
        }
        Some(BoardOutcome {
            text,
            plan: vec!["board:explain".to_string()],
            provenance,
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
        // 排名帧被重新排序后，「为什么」必须按新的方向解释，不能沿用旧方向。
        if let QuerySource::Rank { descending, .. } = &mut frame.source {
            *descending = sort.descending;
        }
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
    knowledge: &dyn DialogueKnowledge,
) -> String {
    let label = knowledge.predicate_label(&left.predicate);
    let mut output = format!(
        "对比「{}」与「{}」的 {}：\n  {}：{}\n  {}：{}",
        left.subject,
        right.subject,
        label,
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
            let word = left_quantity.dimension().comparative_word(&label);
            // 差值写回左侧的原始单位：226 厘米 vs 216 厘米说「10 厘米」，
            // 而不是让用户自己把 0.1 米换算回去。
            let (difference, unit) = left_quantity
                .difference_in_own_unit(&right_quantity)
                .map(|(value, unit)| (value.abs(), unit.unwrap_or("")))
                .unwrap_or((0.0, ""));
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
struct RankReason {
    text: String,
    evidence: Vec<String>,
}
/// 为排名帧生成「为什么排在这里」：相邻名次的取值、换算后差值与原始出处。
///
/// 只比较同量纲行；约数区间重叠时明确说无法确定差距，而不是硬算一个数。
fn rank_reason(frame: &BoardFrame, knowledge: &dyn DialogueKnowledge) -> Option<RankReason> {
    let QuerySource::Rank {
        predicate,
        descending,
    } = &frame.source
    else {
        return None;
    };
    let selected = frame.selected_row().or_else(|| frame.rows.first())?;
    let index = frame.rows.iter().position(|row| row == selected)?;
    let quantity = quantity::parse_quantity(&selected.value)?;
    let matches_dimension = |row: &QueryRow| {
        quantity::parse_quantity(&row.value)
            .map(|candidate| candidate.dimension() == quantity.dimension())
            .unwrap_or(false)
    };
    let previous = frame.rows[..index]
        .iter()
        .rev()
        .find(|row| matches_dimension(row))
        .and_then(|row| quantity::parse_quantity(&row.value).map(|value| (row, value)));
    let next = frame.rows[index + 1..]
        .iter()
        .find(|row| matches_dimension(row))
        .and_then(|row| quantity::parse_quantity(&row.value).map(|value| (row, value)));
    let label = knowledge.predicate_label(predicate);
    let positive = quantity.dimension().comparative_word(&label);
    let word = superlative_word(quantity.dimension(), &label, *descending);

    let mut text = format!(
        "排名依据：{} 的 {} 是 {}。",
        selected.subject, label, selected.value
    );
    // 选中行的出处已由 apply_explain 放入 provenance，这里只补相邻名次。
    let mut evidence = Vec::new();
    match (previous, next) {
        (None, None) => {
            text.push_str("没有其他同量纲记录可以比较，名次由唯一可比取值决定。");
        }
        (None, Some((next_row, next_quantity))) => {
            text.push_str(&relation_words(
                selected,
                &quantity,
                next_row,
                &next_quantity,
                positive,
                &word,
            ));
            evidence.push(next_row.evidence.clone());
        }
        (Some((previous_row, previous_quantity)), None) => {
            text.push_str(&relation_words(
                previous_row,
                &previous_quantity,
                selected,
                &quantity,
                positive,
                &word,
            ));
            evidence.push(previous_row.evidence.clone());
        }
        (Some((previous_row, previous_quantity)), Some((next_row, next_quantity))) => {
            text.push_str(&relation_words(
                previous_row,
                &previous_quantity,
                selected,
                &quantity,
                positive,
                &word,
            ));
            text.push(' ');
            text.push_str(&relation_words(
                selected,
                &quantity,
                next_row,
                &next_quantity,
                positive,
                &word,
            ));
            evidence.push(previous_row.evidence.clone());
            evidence.push(next_row.evidence.clone());
        }
    }
    Some(RankReason { text, evidence })
}

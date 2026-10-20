use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::DialogueKnowledge;

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
        let text = clean(input);
        if text.is_empty() {
            return None;
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
                    && numeric_value(&row.value)
                        .map(|value| op.apply(value, *threshold))
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
        frame.rows.sort_by(|left, right| {
            let left_value = sort_value(left, &sort, knowledge);
            let right_value = sort_value(right, &sort, knowledge);
            let ordering = compare_values(&left_value, &right_value);
            let ordering = if sort.descending {
                ordering.reverse()
            } else {
                ordering
            };
            ordering
                .then_with(|| left.subject.cmp(&right.subject))
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.value.cmp(&right.value))
        });
        frame.selected =
            old_selection.and_then(|selected| frame.rows.iter().position(|row| *row == selected));
        if let Some(selected) = frame.selected {
            frame.page = selected / frame.page_size.max(1);
        } else {
            frame.page = 0;
        }
        frame.op = "sort".to_string();
        frame.program = format!("{} · {}", frame.program, sort.describe());
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
    match (quantity(&left.value), quantity(&right.value)) {
        (Some((left_value, left_unit)), Some((right_value, right_unit)))
            if left_unit == right_unit =>
        {
            let difference = left_value - right_value;
            if difference.abs() < f64::EPSILON {
                output.push_str("\n  两者数值相同。");
            } else {
                let direction = if difference > 0.0 { "高" } else { "低" };
                output.push_str(&format!(
                    "\n  「{}」比「{}」{} {}{}。",
                    left.subject,
                    right.subject,
                    direction,
                    format_number(difference.abs()),
                    left_unit
                ));
            }
        }
        (Some(_), Some(_)) => {
            output.push_str("\n  两者单位不同，无法直接相减；上面分别保留了原始取值。");
        }
        _ => {
            output.push_str("\n  数值不可比较（至少一侧不是数值）。");
        }
    }
    let _ = knowledge;
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
        threshold: f64,
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
                format_number(*threshold)
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
    fn apply(self, left: f64, right: f64) -> bool {
        match self {
            CompareOp::Lt => left < right,
            CompareOp::Le => left <= right,
            CompareOp::Gt => left > right,
            CompareOp::Ge => left >= right,
            CompareOp::Eq => (left - right).abs() < f64::EPSILON,
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

fn sort_value(row: &QueryRow, sort: &SortSpec, _knowledge: &dyn DialogueKnowledge) -> String {
    if let Some(column) = &sort.column {
        if row.predicate.contains(column) || column.contains(row.predicate.as_str()) {
            return row.value.clone();
        }
    }
    row.value.clone()
}

fn compare_values(left: &str, right: &str) -> Ordering {
    match (numeric_value(left), numeric_value(right)) {
        (Some(left), Some(right)) => left
            .partial_cmp(&right)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.to_string().cmp(&right.to_string())),
        _ => left.cmp(right),
    }
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
    if text.contains("有哪些") || text.contains("哪些有") || text.contains("哪些实体") {
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
    let candidate = candidate
        .trim()
        .trim_matches(|c: char| matches!(c, '「' | '」' | '"' | '\'' | '“' | '”'))
        .trim();
    if candidate.is_empty()
        || matches!(
            candidate,
            "实体" | "东西" | "记录" | "数据" | "内容" | "人" | "人物"
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
            if let Some(threshold) = numeric_value(value) {
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
            if let Some(threshold) = numeric_value(value) {
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
        .take_while(|c| !matches!(c, '个' | '条' | '行' | '位' | '页' | '呢' | '的' | ' '))
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

fn numeric_value(raw: &str) -> Option<f64> {
    let text = raw.trim();
    let mut number = String::new();
    let mut started = false;
    for (index, character) in text.char_indices() {
        let is_sign = matches!(character, '-' | '+') && !started && index == 0;
        if character.is_ascii_digit() || character == '.' || is_sign {
            number.push(character);
            if character.is_ascii_digit() {
                started = true;
            }
        } else if started {
            break;
        }
    }
    let value: f64 = number.parse().ok()?;
    if text.contains('万') {
        Some(value * 10_000.0)
    } else if text.contains('亿') {
        Some(value * 100_000_000.0)
    } else {
        Some(value)
    }
}

fn quantity(raw: &str) -> Option<(f64, String)> {
    let value = numeric_value(raw)?;
    let unit = numeric_unit(raw).unwrap_or_default();
    let (value, unit) = match unit.as_str() {
        "" => (value, String::new()),
        "cm" | "厘米" => (value * 0.01, "米".to_string()),
        "mm" | "毫米" => (value * 0.001, "米".to_string()),
        "m" | "米" | "公尺" => (value, "米".to_string()),
        "km" | "千米" | "公里" => (value * 1000.0, "米".to_string()),
        "kg" | "公斤" | "千克" => (value, "千克".to_string()),
        "g" | "克" => (value * 0.001, "千克".to_string()),
        "斤" => (value * 0.5, "千克".to_string()),
        _ => (value, unit.clone()),
    };
    Some((value, unit))
}

fn numeric_unit(raw: &str) -> Option<String> {
    let text = raw.trim();
    let mut number = String::new();
    let mut started = false;
    let mut unit = String::new();
    for (index, character) in text.char_indices() {
        let is_sign = matches!(character, '-' | '+') && !started && index == 0;
        if character.is_ascii_digit() || character == '.' || is_sign {
            number.push(character);
            if character.is_ascii_digit() {
                started = true;
            }
        } else if started {
            if character == '万' || character == '亿' {
                continue;
            }
            if !character.is_whitespace() {
                unit.push(character);
            }
        }
    }
    if unit.is_empty() {
        Some(String::new())
    } else {
        Some(unit)
    }
}

fn format_number(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        let mut text = format!("{:.2}", value);
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
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
        assert!(CompareOp::Gt.apply(2.0, 1.0));
        assert!(CompareOp::Le.apply(1.0, 1.0));
        assert!(CompareOp::Eq.apply(1.0, 1.0));
    }

    #[test]
    fn quantity_normalizes_common_units_and_refuses_mixed_units() {
        let (left, left_unit) = quantity("2.26米").unwrap();
        let (right, right_unit) = quantity("226厘米").unwrap();
        assert_eq!(left_unit, right_unit);
        assert!((left - right).abs() < 1e-9);

        let (weight, weight_unit) = quantity("100斤").unwrap();
        assert_eq!(weight_unit, "千克");
        assert!((weight - 50.0).abs() < 1e-9);

        let left = QueryRow::new("甲", "身高", "226");
        let right = QueryRow::new("乙", "身高", "2.26米");
        let text = render_comparison(&left, &right, &TestKnowledge::new());
        assert!(text.contains("单位不同"));
    }
}

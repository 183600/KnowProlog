use super::anchor::{Anchor, AnchorDraft, Locator};
use super::asset::{AssetId, MediaError, Modality};

/// 单次解析允许的最大表格单元格数（含表头），防止异常输入撑爆锚点。
pub const MAX_TABLE_CELLS: usize = 4096;

/// 最小表格 schema：第 1 行为表头，第 1 列为键列，其余单元格为值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableGrid {
    rows: Vec<Vec<String>>,
    columns: usize,
}

impl TableGrid {
    /// 从二维行构造并校验：至少 2×2、等宽、表头与键列非空。
    pub fn from_rows(rows: Vec<Vec<String>>) -> Result<Self, MediaError> {
        if rows.len() < 2 {
            return Err(MediaError::InvalidTable("表格至少需要表头行和一行数据"));
        }
        let columns = rows[0].len();
        if columns < 2 {
            return Err(MediaError::InvalidTable("表格至少需要键列和一个值列"));
        }
        let cells = rows.len().saturating_mul(columns);
        if cells > MAX_TABLE_CELLS {
            return Err(MediaError::TableTooLarge {
                cells,
                limit: MAX_TABLE_CELLS,
            });
        }
        if rows.iter().any(|row| row.len() != columns) {
            return Err(MediaError::InvalidTable("表格各行必须等宽"));
        }
        if rows[0].iter().any(|cell| cell.trim().is_empty()) {
            return Err(MediaError::InvalidTable("表头不能有空单元格"));
        }
        if rows[1..]
            .iter()
            .any(|row| row.first().map_or(true, |cell| cell.trim().is_empty()))
        {
            return Err(MediaError::InvalidTable("键列不能有空单元格"));
        }
        Ok(Self {
            rows: rows
                .into_iter()
                .map(|row| row.into_iter().map(|cell| cell.trim().to_string()).collect())
                .collect(),
            columns,
        })
    }

    /// 解析逗号分隔的表格文本；空行被忽略，支持中文逗号。
    pub fn parse_csv(text: &str) -> Result<Self, MediaError> {
        let rows: Vec<Vec<String>> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| {
                line.split(['，', ','])
                    .map(|cell| cell.trim().to_string())
                    .collect()
            })
            .collect();
        Self::from_rows(rows)
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn column_count(&self) -> usize {
        self.columns
    }

    pub fn headers(&self) -> &[String] {
        &self.rows[0]
    }

    /// 1-based 单元格；越界返回 `None`。
    pub fn cell(&self, row: usize, column: usize) -> Option<&str> {
        if row == 0 || column == 0 || column > self.columns {
            return None;
        }
        self.rows
            .get(row - 1)
            .and_then(|items| items.get(column - 1))
            .map(String::as_str)
    }

    /// 数据单元格（表头行与键列之外的非空值）。
    pub fn data_cells(&self) -> Vec<(u32, u32, String)> {
        let mut cells = Vec::new();
        for (row_index, row) in self.rows.iter().enumerate().skip(1) {
            for (column_index, value) in row.iter().enumerate().skip(1) {
                if !value.is_empty() {
                    cells.push((
                        (row_index + 1) as u32,
                        (column_index + 1) as u32,
                        value.clone(),
                    ));
                }
            }
        }
        cells
    }

    /// 键列 + 表头 + 值 → 可核对的事实候选（不自动写入图谱）。
    pub fn candidate(
        &self,
        row: usize,
        column: usize,
    ) -> Option<TableFactCandidate> {
        let subject = self.cell(row, 1)?;
        let predicate = self.cell(1, column)?;
        let object = self.cell(row, column)?;
        if row < 2 || column < 2 || subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return None;
        }
        Some(TableFactCandidate {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            locator: Locator::TableCell {
                row: row as u32,
                column: column as u32,
            },
        })
    }

    /// 全部数据单元格的事实候选，按行列排序。
    pub fn candidates(&self) -> Vec<TableFactCandidate> {
        self.data_cells()
            .into_iter()
            .filter_map(|(row, column, _)| self.candidate(row as usize, column as usize))
            .collect()
    }

    fn anchor_drafts(&self) -> Vec<AnchorDraft> {
        self.data_cells()
            .into_iter()
            .map(|(row, column, value)| {
                let entities = self
                    .cell(row as usize, 1)
                    .filter(|key| !key.is_empty())
                    .map(|key| vec![key.to_string()])
                    .unwrap_or_default();
                AnchorDraft::new(
                    Locator::TableCell { row, column },
                    value,
                    "table_parser",
                    1.0,
                )
                .with_entities(entities)
            })
            .collect()
    }
}

/// 一条表格事实候选；升格为图谱事实仍走人工/教学路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFactCandidate {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub locator: Locator,
}

impl TableFactCandidate {
    pub fn render_zh(&self) -> String {
        format!(
            "{} 的 {} 是 {}（{}）",
            self.subject,
            self.predicate,
            self.object,
            self.locator.label_zh()
        )
    }
}

impl super::store::MediaStore {
    /// 把表格数据单元格批量写成锚点；先验证资产模态与容量，避免半途写入。
    pub fn ingest_table(
        &mut self,
        asset_id: &AssetId,
        grid: &TableGrid,
    ) -> Result<Vec<Anchor>, MediaError> {
        let (modality, anchor_count, text_limit) = {
            let stored = self
                .assets
                .get(asset_id)
                .ok_or_else(|| MediaError::UnknownAsset(asset_id.as_str().to_string()))?;
            (
                stored.record.modality,
                stored.anchors.len(),
                self.limits.max_anchor_text_chars,
            )
        };
        if modality != Modality::Table {
            return Err(MediaError::LocatorModalityMismatch {
                modality,
                locator: "table_cell",
            });
        }
        let drafts = grid.anchor_drafts();
        if anchor_count.saturating_add(drafts.len()) > self.limits.max_anchors_per_asset {
            return Err(MediaError::AnchorCapacityExceeded {
                limit: self.limits.max_anchors_per_asset,
            });
        }
        if drafts
            .iter()
            .any(|draft| draft.text.chars().count() > text_limit)
        {
            return Err(MediaError::AnchorTextTooLong {
                actual: drafts
                    .iter()
                    .map(|draft| draft.text.chars().count())
                    .max()
                    .unwrap_or(0),
                limit: text_limit,
            });
        }
        let mut anchors = Vec::with_capacity(drafts.len());
        for draft in drafts {
            anchors.push(self.add_anchor(asset_id, draft)?);
        }
        Ok(anchors)
    }
}

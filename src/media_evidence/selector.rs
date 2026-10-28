use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::anchor::{LocatorHint, Side};
use super::asset::Modality;

/// 相对序号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SelectorOrdinal {
    Any,
    Latest,
    Previous,
    Index(usize),
}

/// 从自然语言指称解析出的媒体选择器。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaSelector {
    pub raw: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modalities: Vec<Modality>,
    pub ordinal: SelectorOrdinal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<LocatorHint>,
}

impl MediaSelector {
    /// 解析媒体指称；没有任何媒体信号时返回 `None`（不劫持普通问题）。
    pub fn parse(input: &str) -> Option<Self> {
        let compact: Vec<char> = compact(input);
        if compact.is_empty() {
            return None;
        }
        let mut modalities = BTreeSet::new();
        if has_image_reference(&compact) {
            modalities.insert(Modality::Image);
        }
        if has_table_reference(&compact) {
            modalities.insert(Modality::Table);
        }
        if has_audio_reference(&compact) {
            modalities.insert(Modality::Audio);
        }
        if has_document_reference(&compact) {
            modalities.insert(Modality::Document);
        }
        let locator = parse_locator(&compact);
        match locator {
            Some(LocatorHint::TableCell { .. })
            | Some(LocatorHint::TableRow { .. })
            | Some(LocatorHint::TableColumn { .. }) => {
                modalities.insert(Modality::Table);
            }
            Some(LocatorHint::ImageSide(_)) => {
                modalities.insert(Modality::Image);
            }
            _ => {}
        }
        let ordinal = parse_ordinal(&compact);
        if modalities.is_empty() {
            if let SelectorOrdinal::Index(_) = ordinal {
                if contains_sequence(&compact, "张") || contains_sequence(&compact, "幅") {
                    modalities.insert(Modality::Image);
                } else if contains_sequence(&compact, "页") {
                    modalities.insert(Modality::Document);
                }
            }
        }
        if modalities.is_empty() && locator.is_none() {
            return None;
        }
        Some(Self {
            raw: input.to_string(),
            modalities: modalities.into_iter().collect(),
            ordinal,
            locator,
        })
    }
}

fn compact(input: &str) -> Vec<char> {
    input
        .chars()
        .filter(|character| {
            !character.is_whitespace()
                && !matches!(
                    character,
                    '，' | '。'
                        | '？'
                        | '！'
                        | '、'
                        | '；'
                        | '：'
                        | ','
                        | '.'
                        | '?'
                        | '!'
                        | ';'
                        | ':'
                        | '“'
                        | '”'
                        | '‘'
                        | '’'
                        | '"'
                        | '\''
                )
        })
        .collect()
}

fn find_sequence(haystack: &[char], needle: &str) -> Option<usize> {
    let needle: Vec<char> = needle.chars().collect();
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn contains_sequence(haystack: &[char], needle: &str) -> bool {
    find_sequence(haystack, needle).is_some()
}

fn has_image_reference(chars: &[char]) -> bool {
    for token in [
        "图片",
        "图像",
        "照片",
        "截图",
        "截屏",
        "插图",
        "地图",
        "蓝图",
        "草图",
        "流程图",
        "架构图",
        "示意图",
        "柱状图",
        "折线图",
        "饼图",
        "统计图",
        "分布图",
        "走势图",
        "施工图",
        "电路图",
        "思维导图",
        "平面图",
        "立面图",
        "剖面图",
        "这张图",
        "这幅图",
        "此图",
        "上图",
        "下图",
    ] {
        if contains_sequence(chars, token) {
            return true;
        }
    }
    chars.iter().enumerate().any(|(index, character)| {
        if *character != '图' {
            return false;
        }
        let previous = index.checked_sub(1).and_then(|at| chars.get(at));
        let next = chars.get(index + 1);
        if previous == Some(&'谱') || next == Some(&'谱') {
            return false;
        }
        let previous_ok = previous.is_some_and(|previous| {
            matches!(
                previous,
                '这' | '那' | '上' | '下' | '此' | '本' | '该' | '第' | '张' | '幅' | '看' | '如'
                    | '见' | '附' | '示'
            ) || previous.is_ascii_digit()
        });
        let next_ok = next.is_some_and(|next| {
            matches!(
                next,
                '中' | '里' | '上' | '下' | '内' | '的' | '所' | '示' | '片' | '像'
            ) || next.is_ascii_digit()
                || is_numeral_char(*next)
        });
        previous_ok || next_ok
    })
}

fn has_table_reference(chars: &[char]) -> bool {
    for token in [
        "表格",
        "报表",
        "表单",
        "列表",
        "统计表",
        "时间表",
        "课程表",
        "日程表",
        "对照表",
        "价目表",
        "数据表",
        "图表",
    ] {
        if contains_sequence(chars, token) {
            return true;
        }
    }
    for (index, character) in chars.iter().enumerate() {
        if *character != '表' {
            continue;
        }
        let previous_ok = index > 0
            && matches!(
                chars[index - 1],
                '这' | '上' | '下' | '该' | '此' | '那' | '第' | '图' | '张' | '幅' | '段'
                    | '份'
            );
        let next_ok = chars.get(index + 1).is_some_and(|next| {
            next.is_ascii_digit()
                || is_numeral_char(*next)
                || matches!(next, '中' | '里' | '内' | '上' | '下')
        });
        if previous_ok || next_ok {
            return true;
        }
    }
    false
}

fn has_audio_reference(chars: &[char]) -> bool {
    ["录音", "音频", "语音", "声音"]
        .iter()
        .any(|token| contains_sequence(chars, token))
}

fn has_document_reference(chars: &[char]) -> bool {
    ["文档", "文件", "附件", "PDF"]
        .iter()
        .any(|token| contains_sequence(chars, token))
}

fn parse_ordinal(chars: &[char]) -> SelectorOrdinal {
    for index in 0..chars.len() {
        if chars[index] != '第' {
            continue;
        }
        let Some((value, end)) = parse_number_at(chars, index + 1) else {
            continue;
        };
        if value == 0 {
            continue;
        }
        match chars.get(end) {
            Some('行' | '列' | '句') => continue,
            Some('张' | '幅' | '个' | '段' | '条' | '份' | '组' | '块' | '页') => {
                return SelectorOrdinal::Index(value);
            }
            Some(_) | None => return SelectorOrdinal::Index(value),
        }
    }
    for (index, character) in chars.iter().enumerate() {
        if !matches!(*character, '图' | '表') {
            continue;
        }
        if index > 0 && chars[index - 1] == '谱' {
            continue;
        }
        if let Some((value, _)) = parse_number_at(chars, index + 1) {
            if value > 0 {
                return SelectorOrdinal::Index(value);
            }
        }
    }
    if let Some(start) = find_sequence(chars, "录音") {
        if let Some((value, _)) = parse_number_at(chars, start + 2) {
            if value > 0 {
                return SelectorOrdinal::Index(value);
            }
        }
    }
    for token in ["最后", "刚才", "刚刚", "最近", "当前"] {
        if contains_sequence(chars, token) {
            return SelectorOrdinal::Latest;
        }
    }
    for token in ["这张", "这幅", "这段", "这个", "此图", "此表", "本图", "本表"] {
        if contains_sequence(chars, token) {
            return SelectorOrdinal::Latest;
        }
    }
    for token in ["上一", "前一张", "前一幅", "前一段", "前一个"] {
        if contains_sequence(chars, token) {
            return SelectorOrdinal::Previous;
        }
    }
    SelectorOrdinal::Any
}

fn parse_locator(chars: &[char]) -> Option<LocatorHint> {
    let mut row = None;
    let mut column = None;
    let mut sentence = None;
    for index in 0..chars.len() {
        if chars[index] != '第' {
            continue;
        }
        let Some((value, end)) = parse_number_at(chars, index + 1) else {
            continue;
        };
        if value == 0 {
            continue;
        }
        match chars.get(end) {
            Some('行') => row = Some(value as u32),
            Some('列') => column = Some(value as u32),
            Some('句') => sentence = Some(value),
            _ => {}
        }
    }
    match (row, column) {
        (Some(row), Some(column)) => return Some(LocatorHint::TableCell { row, column }),
        (Some(row), None) => return Some(LocatorHint::TableRow { row }),
        (None, Some(column)) => return Some(LocatorHint::TableColumn { column }),
        (None, None) => {}
    }
    if let Some(sentence) = sentence {
        return Some(LocatorHint::Sentence(sentence));
    }
    if contains_sequence(chars, "最后一句")
        || contains_sequence(chars, "最后那句话")
        || contains_sequence(chars, "末句")
    {
        return Some(LocatorHint::LastSentence);
    }
    if ["左边", "左侧", "左面", "左上"]
        .iter()
        .any(|token| contains_sequence(chars, token))
    {
        return Some(LocatorHint::ImageSide(Side::Left));
    }
    if ["右边", "右侧", "右面", "右下"]
        .iter()
        .any(|token| contains_sequence(chars, token))
    {
        return Some(LocatorHint::ImageSide(Side::Right));
    }
    if ["中间", "中央", "正中"]
        .iter()
        .any(|token| contains_sequence(chars, token))
    {
        return Some(LocatorHint::ImageSide(Side::Center));
    }
    if ["上面", "上方", "顶部", "上边"]
        .iter()
        .any(|token| contains_sequence(chars, token))
    {
        return Some(LocatorHint::ImageSide(Side::Top));
    }
    if ["下面", "下方", "底部", "下边"]
        .iter()
        .any(|token| contains_sequence(chars, token))
    {
        return Some(LocatorHint::ImageSide(Side::Bottom));
    }
    None
}

fn is_numeral_char(character: char) -> bool {
    matches!(
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
    )
}

fn numeral_value(character: char) -> Option<usize> {
    match character {
        '零' | '〇' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn parse_number_at(chars: &[char], start: usize) -> Option<(usize, usize)> {
    let mut end = start;
    while end < chars.len() && (chars[end].is_ascii_digit() || is_numeral_char(chars[end])) {
        end += 1;
    }
    if end == start {
        return None;
    }
    let token: String = chars[start..end].iter().collect();
    parse_number(&token).map(|value| (value, end))
}

/// 解析阿拉伯数字或 0..=999 的中文数字；失败返回 `None`。
pub fn parse_number(token: &str) -> Option<usize> {
    if token.is_empty() {
        return None;
    }
    if token.chars().all(|character| character.is_ascii_digit()) {
        return token.parse().ok();
    }
    if !token.chars().all(is_numeral_char) {
        return None;
    }
    if token.contains('百') || token.contains('十') {
        let mut total = 0usize;
        let mut current = 0usize;
        for character in token.chars() {
            match character {
                '百' => {
                    total = total.checked_add(current.max(1).checked_mul(100)?)?;
                    current = 0;
                }
                '十' => {
                    total = total.checked_add(current.max(1).checked_mul(10)?)?;
                    current = 0;
                }
                other => {
                    current = numeral_value(other)?;
                }
            }
        }
        return total.checked_add(current);
    }
    let mut value = 0usize;
    for character in token.chars() {
        value = value.checked_mul(10)?;
        value = value.checked_add(numeral_value(character)?)?;
    }
    Some(value)
}

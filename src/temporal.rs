//! TPWM：时序路径工作记忆的只读时间内核。
//!
//! 时间在「句子 → 关系帧」层由 `relation_frame` 的 `TemporalScope` 表达；
//! 本模块把时间落实为事实层、路径层与 Prolog 层可检查的数据：
//!   - [`TimePoint`] / [`TimeSpan`]：规范化到年月日的闭区间，含开区间；
//!   - [`Validity`]：一条事实的有效期，显式区分「未知」「无界（长期）」
//!     「区间」「空」——没写时间不等于永远为真；
//!   - [`TemporalIndex`]：只读时序索引（按时刻/区间检索、单值谓词时序冲突）；
//!   - [`TemporalSession`]：会话参考时间与焦点区间，相对时间在没有参考时间
//!     时返回 `UndefinedReferenceTime`，绝不猜系统时钟；
//!   - [`TemporalConstraint`]：路径/查询上的时间约束，供路径内核统一执行。
//!
//! 纯符号、零训练、同输入同字节：本模块不做时区/农历/日历换算，只到
//! 年月日，年粒度优先。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::prolog_engine::PrologFact;

/// 时序方言标识（sidecar / CLI 输出用）。
pub const TEMPORAL_DIALECT: &str = "knowprolog-temporal/1";

// ---------------------------------------------------------------------------
// 时间点与区间
// ---------------------------------------------------------------------------

/// 时间粒度：年、月、日。比较时越细的粒度越精确。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeGranularity {
    Year,
    Month,
    Day,
}

impl TimeGranularity {
    pub fn label(self) -> &'static str {
        match self {
            TimeGranularity::Year => "年",
            TimeGranularity::Month => "月",
            TimeGranularity::Day => "日",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            TimeGranularity::Year => "year",
            TimeGranularity::Month => "month",
            TimeGranularity::Day => "day",
        }
    }

    fn rank(self) -> u8 {
        match self {
            TimeGranularity::Year => 0,
            TimeGranularity::Month => 1,
            TimeGranularity::Day => 2,
        }
    }

    /// 取更细的粒度（一个年区间与一个日区间的交按日算）。
    pub fn finer(self, other: TimeGranularity) -> TimeGranularity {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }
}

/// 规范化日历时间点（公历，允许公元前为负数年份）。
///
/// 字段始终被规范化：月 1..=12、日 1..=当月天数。序关系即年月日字典序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimePoint {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

impl TimePoint {
    /// 构造时间点；月/日会被夹到合法范围，不做日历校验失败。
    pub fn new(year: i32, month: u32, day: u32) -> Self {
        let month = month.clamp(1, 12) as u8;
        let max_day = days_in_month(year, month);
        let day = day.clamp(1, max_day as u32) as u8;
        Self { year, month, day }
    }

    pub fn year(year: i32) -> Self {
        Self::new(year, 1, 1)
    }

    pub fn month(year: i32, month: u32) -> Self {
        Self::new(year, month, 1)
    }

    pub fn day(year: i32, month: u32, day: u32) -> Self {
        Self::new(year, month, day)
    }

    /// 公历日序（1970-01-01 为 0），可正可负；用于区间长度与前后移动。
    pub fn to_ordinal(self) -> i64 {
        let y = self.year as i64 - if self.month <= 2 { 1 } else { 0 };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let mp = ((self.month as i64) + 9) % 12;
        let doy = (153 * mp + 2) / 5 + self.day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    pub fn from_ordinal(ordinal: i64) -> Self {
        let z = ordinal + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = if month <= 2 { y + 1 } else { y };
        Self {
            year: year as i32,
            month: month as u8,
            day: day as u8,
        }
    }

    pub fn next_day(self) -> Self {
        Self::from_ordinal(self.to_ordinal() + 1)
    }

    pub fn prev_day(self) -> Self {
        Self::from_ordinal(self.to_ordinal() - 1)
    }

    /// 该时间点所在粒度单位的最后一天。
    pub fn period_end(self, granularity: TimeGranularity) -> Self {
        match granularity {
            TimeGranularity::Year => Self::new(self.year, 12, 31),
            TimeGranularity::Month => {
                Self::new(self.year, self.month as u32, days_in_month(self.year, self.month) as u32)
            }
            TimeGranularity::Day => self,
        }
    }

    /// 该时间点所在粒度单位的第一天。
    pub fn period_start(self, granularity: TimeGranularity) -> Self {
        match granularity {
            TimeGranularity::Year => Self::new(self.year, 1, 1),
            TimeGranularity::Month => Self::new(self.year, self.month as u32, 1),
            TimeGranularity::Day => self,
        }
    }

    pub fn render(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    pub fn render_at(&self, granularity: TimeGranularity) -> String {
        match granularity {
            TimeGranularity::Year => format!("{}年", self.year),
            TimeGranularity::Month => format!("{:04}-{:02}", self.year, self.month),
            TimeGranularity::Day => self.render(),
        }
    }

    /// Prolog `temporal_edge/8` 的整数编码：YYYYMMDD（公元前为负）。
    pub fn to_prolog_int(self) -> i64 {
        (self.year as i64) * 10_000 + (self.month as i64) * 100 + self.day as i64
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 31,
    }
}

/// 闭区间 `[start, end]`；`None` 表示该侧无界（开区间）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeSpan {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<TimePoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<TimePoint>,
    pub granularity: TimeGranularity,
}

impl TimeSpan {
    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
            granularity: TimeGranularity::Year,
        }
    }

    pub fn year(year: i32) -> Self {
        Self {
            start: Some(TimePoint::year(year)),
            end: Some(TimePoint::new(year, 12, 31)),
            granularity: TimeGranularity::Year,
        }
    }

    pub fn month(year: i32, month: u32) -> Self {
        let start = TimePoint::month(year, month);
        Self {
            start: Some(start),
            end: Some(start.period_end(TimeGranularity::Month)),
            granularity: TimeGranularity::Month,
        }
    }

    pub fn day(year: i32, month: u32, day: u32) -> Self {
        let point = TimePoint::day(year, month, day);
        Self {
            start: Some(point),
            end: Some(point),
            granularity: TimeGranularity::Day,
        }
    }

    /// 由两个边界构造；起点晚于终点时返回 `None`。
    pub fn from_bounds(
        start: Option<TimePoint>,
        end: Option<TimePoint>,
        granularity: TimeGranularity,
    ) -> Option<Self> {
        if let (Some(start), Some(end)) = (start, end) {
            if start > end {
                return None;
            }
        }
        Some(Self {
            start,
            end,
            granularity,
        })
    }

    /// `point` 所在粒度单位之前（不含该单位）。
    pub fn before(point: TimePoint, granularity: TimeGranularity) -> Self {
        let start = point.period_start(granularity);
        Self {
            start: None,
            end: Some(start.prev_day()),
            granularity,
        }
    }

    /// `point` 所在粒度单位之后（不含该单位）。
    pub fn after(point: TimePoint, granularity: TimeGranularity) -> Self {
        let end = point.period_end(granularity);
        Self {
            start: Some(end.next_day()),
            end: None,
            granularity,
        }
    }

    /// 从 `point` 所在粒度单位的起点开始（“以来/起”）。
    pub fn since(point: TimePoint, granularity: TimeGranularity) -> Self {
        Self {
            start: Some(point.period_start(granularity)),
            end: None,
            granularity,
        }
    }

    pub fn contains(&self, point: TimePoint) -> bool {
        if let Some(start) = self.start {
            if point < start {
                return false;
            }
        }
        if let Some(end) = self.end {
            if point > end {
                return false;
            }
        }
        true
    }

    pub fn overlaps(&self, other: &TimeSpan) -> bool {
        self.intersect(other).is_some()
    }

    pub fn intersect(&self, other: &TimeSpan) -> Option<TimeSpan> {
        let start = match (self.start, other.start) {
            (None, value) => value,
            (value, None) => value,
            (Some(left), Some(right)) => Some(left.max(right)),
        };
        let end = match (self.end, other.end) {
            (None, value) => value,
            (value, None) => value,
            (Some(left), Some(right)) => Some(left.min(right)),
        };
        TimeSpan::from_bounds(start, end, self.granularity.finer(other.granularity))
    }

    pub fn is_unbounded(&self) -> bool {
        self.start.is_none() && self.end.is_none()
    }

    /// 区间覆盖的日数；无界时为 `None`。
    pub fn day_count(&self) -> Option<i64> {
        match (self.start, self.end) {
            (Some(start), Some(end)) => Some(end.to_ordinal() - start.to_ordinal() + 1),
            _ => None,
        }
    }

    pub fn render(&self) -> String {
        match (self.start, self.end) {
            (None, None) => "时间不限".to_string(),
            (None, Some(end)) => format!("{}之前", end.next_day().render_at(self.granularity)),
            (Some(start), None) => format!("{}以来", start.render_at(self.granularity)),
            (Some(start), Some(end)) => {
                if self.granularity == TimeGranularity::Year && start.year == end.year {
                    format!("{}年", start.year)
                } else if start == end {
                    start.render_at(self.granularity)
                } else {
                    format!(
                        "{}至{}",
                        start.render_at(self.granularity),
                        end.render_at(self.granularity)
                    )
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 有效期与约束
// ---------------------------------------------------------------------------

/// 一条事实的有效期（valid time）。
///
/// 刻意区分 [`Validity::Unknown`] 与 [`Validity::Unbounded`]：没写时间的边是
/// 「未知」，不是「永远为真」；只有显式「一直/长期」才是无界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Validity {
    /// 无时间信息（默认）。
    #[default]
    Unknown,
    /// 显式无界：长期有效。
    Unbounded,
    /// 有界有效期。
    Span(TimeSpan),
    /// 有效期为空（自相矛盾记录），任何时刻都不成立。
    Never,
}

impl Validity {
    pub fn from_span(span: TimeSpan) -> Self {
        Validity::Span(span)
    }

    /// 闭区间年份（含首尾）。
    pub fn from_years(start_year: i32, end_year: i32) -> Self {
        match TimeSpan::from_bounds(
            Some(TimePoint::year(start_year.min(end_year))),
            Some(TimePoint::new(start_year.max(end_year), 12, 31)),
            TimeGranularity::Year,
        ) {
            Some(span) => Validity::Span(span),
            None => Validity::Never,
        }
    }

    pub fn span(&self) -> Option<TimeSpan> {
        match self {
            Validity::Span(span) => Some(*span),
            _ => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Validity::Unknown)
    }

    pub fn is_known(&self) -> bool {
        !matches!(self, Validity::Unknown)
    }

    pub fn is_unbounded(&self) -> bool {
        matches!(self, Validity::Unbounded)
    }

    pub fn is_never(&self) -> bool {
        matches!(self, Validity::Never)
    }

    /// 该有效期内是否存在任何时刻。
    pub fn is_satisfiable(self) -> bool {
        match self {
            Validity::Never => false,
            Validity::Span(span) => span.start.is_none()
                || span.end.is_none()
                || matches!((span.start, span.end), (Some(s), Some(e)) if s <= e),
            _ => true,
        }
    }

    pub fn contains(self, point: TimePoint) -> bool {
        match self {
            Validity::Unbounded => true,
            Validity::Span(span) => span.contains(point),
            _ => false,
        }
    }

    pub fn intersects(self, span: &TimeSpan) -> bool {
        match self {
            Validity::Unbounded => true,
            Validity::Span(own) => own.overlaps(span),
            _ => false,
        }
    }

    /// 两个有效期的交集；与未知相交仍是未知，与空相交是空。
    pub fn intersection(self, other: Validity) -> Validity {
        match (self, other) {
            (Validity::Never, _) | (_, Validity::Never) => Validity::Never,
            (Validity::Unknown, _) | (_, Validity::Unknown) => Validity::Unknown,
            (Validity::Unbounded, other) => other,
            (this, Validity::Unbounded) => this,
            (Validity::Span(left), Validity::Span(right)) => match left.intersect(&right) {
                Some(span) => Validity::Span(span),
                None => Validity::Never,
            },
        }
    }

    pub fn render(self) -> String {
        match self {
            Validity::Unknown => "时间未知".to_string(),
            Validity::Unbounded => "长期有效".to_string(),
            Validity::Span(span) => span.render(),
            Validity::Never => "无有效时间".to_string(),
        }
    }
}

/// 未写时间的边在带时间约束的查询里如何处置。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownTimePolicy {
    /// 严格：未知时间不参与 AsOf/区间查询。
    #[default]
    Strict,
    /// 宽松：未知时间只作背景，允许通过但标注 caveat。
    Background,
}

impl UnknownTimePolicy {
    pub fn label(self) -> &'static str {
        match self {
            UnknownTimePolicy::Strict => "严格",
            UnknownTimePolicy::Background => "宽松",
        }
    }
}

/// 查询/路径上的时间约束，由路径内核在每个步骤上执行。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalConstraint {
    /// 不约束。
    #[default]
    Any,
    /// 状态查询：事实必须在 `point` 时刻为真。
    AsOf(TimePoint),
    /// 区间查询：事实有效期与 `span` 必须有交集。
    During(TimeSpan),
}

impl TemporalConstraint {
    pub fn is_any(&self) -> bool {
        matches!(self, TemporalConstraint::Any)
    }

    pub fn as_of(year: i32) -> Self {
        TemporalConstraint::AsOf(TimePoint::year(year))
    }

    pub fn during(span: TimeSpan) -> Self {
        TemporalConstraint::During(span)
    }

    /// 从区间表达式得到约束：点状（年/月/日）优先按 AsOf 处理。
    pub fn from_span(span: TimeSpan) -> Self {
        if let (Some(start), Some(end)) = (span.start, span.end) {
            if start == end {
                return TemporalConstraint::AsOf(start);
            }
        }
        TemporalConstraint::During(span)
    }

    pub fn allows(&self, validity: Validity, policy: UnknownTimePolicy) -> bool {
        match self {
            TemporalConstraint::Any => true,
            TemporalConstraint::AsOf(point) => match validity {
                Validity::Never => false,
                Validity::Unknown => policy == UnknownTimePolicy::Background,
                Validity::Unbounded => true,
                Validity::Span(span) => span.contains(*point),
            },
            TemporalConstraint::During(span) => match validity {
                Validity::Never => false,
                Validity::Unknown => policy == UnknownTimePolicy::Background,
                Validity::Unbounded => true,
                Validity::Span(own) => own.overlaps(span),
            },
        }
    }

    pub fn describe(&self) -> String {
        match self {
            TemporalConstraint::Any => "时间不限".to_string(),
            TemporalConstraint::AsOf(point) => format!("{}时", point.render()),
            TemporalConstraint::During(span) => format!("{}内", span.render()),
        }
    }
}

/// 状态（有区间）vs 事件（发生点/区间）。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAspect {
    #[default]
    State,
    Event,
}

impl TemporalAspect {
    pub fn label(self) -> &'static str {
        match self {
            TemporalAspect::State => "状态",
            TemporalAspect::Event => "事件",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            TemporalAspect::State => "state",
            TemporalAspect::Event => "event",
        }
    }
}

// ---------------------------------------------------------------------------
// 时序事实与只读索引
// ---------------------------------------------------------------------------

/// 带有效期的一条事实（时序索引/ProLog 事实源的统一形状）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub validity: Validity,
    #[serde(default)]
    pub aspect: TemporalAspect,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    #[serde(default = "default_confidence_bps")]
    pub confidence_bps: u16,
}

fn default_confidence_bps() -> u16 {
    10_000
}

impl TemporalFact {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            object: object.trim().to_string(),
            validity: Validity::Unknown,
            aspect: TemporalAspect::State,
            source: String::new(),
            confidence_bps: 10_000,
        }
    }

    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }

    pub fn with_years(mut self, start_year: i32, end_year: i32) -> Self {
        self.validity = Validity::from_years(start_year, end_year);
        self
    }

    pub fn with_aspect(mut self, aspect: TemporalAspect) -> Self {
        self.aspect = aspect;
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.trim().to_string();
        self
    }

    pub fn with_confidence_bps(mut self, confidence_bps: u16) -> Self {
        self.confidence_bps = confidence_bps.min(10_000);
        self
    }

    pub fn render(&self) -> String {
        format!(
            "{}({}, {}) [{}，{}]",
            self.predicate,
            self.subject,
            self.object,
            self.aspect.label(),
            self.validity.render()
        )
    }

    pub fn sort_key(&self) -> (i64, String, String, String) {
        let ordinal = match self.validity {
            Validity::Span(span) => span
                .start
                .map(|point| point.to_ordinal())
                .unwrap_or(i64::MIN),
            _ => i64::MIN,
        };
        (
            ordinal,
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
        )
    }
}

/// 单值谓词在不同时间点上的重叠冲突。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalConflict {
    pub subject: String,
    pub predicate: String,
    pub left: String,
    pub right: String,
    pub left_validity: Validity,
    pub right_validity: Validity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap: Option<TimeSpan>,
}

impl TemporalConflict {
    pub fn render(&self) -> String {
        let overlap = match self.overlap {
            Some(span) => format!("在{}重叠", span.render()),
            None => "在已知时间段内重叠".to_string(),
        };
        format!(
            "「{}」的「{}」同时为「{}」（{}）与「{}」（{}），{}",
            self.subject,
            self.predicate,
            self.left,
            self.left_validity.render(),
            self.right,
            self.right_validity.render(),
            overlap
        )
    }
}

/// 只读时序索引：按时刻/区间检索事实，做时序单值冲突判定；不物化进 RDF。
#[derive(Debug, Clone, Default)]
pub struct TemporalIndex {
    facts: Vec<TemporalFact>,
    nodes: BTreeSet<String>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, fact: TemporalFact) {
        self.nodes.insert(fact.subject.clone());
        self.nodes.insert(fact.object.clone());
        self.facts.push(fact);
    }

    pub fn extend(&mut self, facts: impl IntoIterator<Item = TemporalFact>) {
        for fact in facts {
            self.insert(fact);
        }
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn facts(&self) -> &[TemporalFact] {
        &self.facts
    }

    pub fn has_node(&self, node: &str) -> bool {
        self.nodes.contains(node)
    }

    pub fn nodes(&self) -> &BTreeSet<String> {
        &self.nodes
    }

    /// 在 `point` 时刻成立的事实（严格：时间未知的边不算）。
    pub fn valid_at(&self, point: TimePoint) -> Vec<&TemporalFact> {
        let mut facts: Vec<&TemporalFact> = self
            .facts
            .iter()
            .filter(|fact| fact.validity.contains(point))
            .collect();
        facts.sort_by_key(|fact| fact.sort_key());
        facts
    }

    /// 与 `span` 有交集的事实（严格）。
    pub fn overlapping(&self, span: &TimeSpan) -> Vec<&TemporalFact> {
        let mut facts: Vec<&TemporalFact> = self
            .facts
            .iter()
            .filter(|fact| fact.validity.intersects(span))
            .collect();
        facts.sort_by_key(|fact| fact.sort_key());
        facts
    }

    /// 时间未知的事实（宽松模式下的背景）。
    pub fn unknown_time_facts(&self) -> Vec<&TemporalFact> {
        self.facts
            .iter()
            .filter(|fact| fact.validity.is_unknown())
            .collect()
    }

    pub fn for_entity(&self, entity: &str) -> Vec<&TemporalFact> {
        let mut facts: Vec<&TemporalFact> = self
            .facts
            .iter()
            .filter(|fact| fact.subject == entity || fact.object == entity)
            .collect();
        facts.sort_by_key(|fact| fact.sort_key());
        facts
    }

    /// 时间线：按有效期起点排序，未知时间的排在最后。
    pub fn timeline(&self) -> Vec<&TemporalFact> {
        let mut facts: Vec<&TemporalFact> = self.facts.iter().collect();
        facts.sort_by_key(|fact| fact.sort_key());
        facts
    }

    /// 时序单值冲突：同一 (subject, predicate) 下不同对象且有效期重叠。
    ///
    /// 与全局单值判定不同，历史更替（先后不重叠）不会被误报。
    pub fn functional_conflicts(
        &self,
        functional_predicates: &BTreeSet<String>,
    ) -> Vec<TemporalConflict> {
        let mut groups: BTreeMap<(String, String), Vec<&TemporalFact>> = BTreeMap::new();
        for fact in &self.facts {
            if !functional_predicates.contains(&fact.predicate) {
                continue;
            }
            groups
                .entry((fact.subject.clone(), fact.predicate.clone()))
                .or_default()
                .push(fact);
        }
        let mut conflicts = Vec::new();
        for ((subject, predicate), facts) in groups {
            for (index, left) in facts.iter().enumerate() {
                for right in facts.iter().skip(index + 1) {
                    if left.object == right.object {
                        continue;
                    }
                    let (overlap, conflict) = Self::temporal_overlap(left.validity, right.validity);
                    if conflict {
                        conflicts.push(TemporalConflict {
                            subject: subject.clone(),
                            predicate: predicate.clone(),
                            left: left.object.clone(),
                            right: right.object.clone(),
                            left_validity: left.validity,
                            right_validity: right.validity,
                            overlap,
                        });
                    }
                }
            }
        }
        conflicts
    }

    fn temporal_overlap(left: Validity, right: Validity) -> (Option<TimeSpan>, bool) {
        match (left, right) {
            (Validity::Never, _) | (_, Validity::Never) => (None, false),
            // 未知时间无法判定重叠，宁缺毋滥。
            (Validity::Unknown, _) | (_, Validity::Unknown) => (None, false),
            (Validity::Unbounded, Validity::Unbounded) => (None, true),
            (Validity::Unbounded, Validity::Span(span))
            | (Validity::Span(span), Validity::Unbounded) => (Some(span), true),
            (Validity::Span(left_span), Validity::Span(right_span)) => {
                match left_span.intersect(&right_span) {
                    Some(span) => (Some(span), true),
                    None => (None, false),
                }
            }
        }
    }

    /// 从图谱事实构建索引。
    ///
    /// RDF 事实本身是三元组；有效期来自主语实体上的约定字段
    /// （`valid_from`/`start_year`/`since` 与 `valid_to`/`end_year`/`until`）。
    /// 没有这些字段的边保持 [`Validity::Unknown`]，绝不当成永远为真。
    pub fn from_prolog_facts(facts: &[PrologFact]) -> Self {
        let mut validity_by_entity: BTreeMap<&str, Validity> = BTreeMap::new();
        for fact in facts {
            if let PrologFact::Entity {
                name, properties, ..
            } = fact
            {
                if let Some(validity) = validity_from_properties(properties) {
                    validity_by_entity.insert(name.as_str(), validity);
                    // 快照键可能是 IRI 尾部（如 "kyoto"），关系端点却用 kg:name
                    // 的字面量（如 "京都"）：两个别名都登记，避免时序元数据丢失。
                    if let Some(alias) = properties.get("name") {
                        validity_by_entity.insert(alias.as_str(), validity);
                    }
                }
            }
        }
        let mut index = TemporalIndex::new();
        for fact in facts {
            if let PrologFact::Relation { from, rel_type, to } = fact {
                let validity = validity_by_entity
                    .get(from.as_str())
                    .copied()
                    .unwrap_or(Validity::Unknown);
                let aspect = aspect_for_predicate(rel_type);
                index.insert(
                    TemporalFact::new(from, rel_type, to)
                        .with_validity(validity)
                        .with_aspect(aspect)
                        .with_source("知识图谱"),
                );
            }
        }
        index
    }
}

const VALID_FROM_FIELDS: &[&str] = &["valid_from", "start_year", "since", "开始年份", "起始年份"];
const VALID_TO_FIELDS: &[&str] = &["valid_to", "end_year", "until", "结束年份", "终止年份"];
const EVENT_PREDICATES: &[&str] = &[
    "成为", "就任", "即位", "迁都", "出生", "去世", "逝世", "成立", "解散", "当选", "发布",
];

/// 该属性名是否是有效时间元数据字段（不应作为普通关系参与路径）。
pub fn is_temporal_field(name: &str) -> bool {
    VALID_FROM_FIELDS.contains(&name) || VALID_TO_FIELDS.contains(&name)
}

fn aspect_for_predicate(predicate: &str) -> TemporalAspect {
    if EVENT_PREDICATES
        .iter()
        .any(|marker| predicate.contains(marker))
    {
        TemporalAspect::Event
    } else {
        TemporalAspect::State
    }
}

/// 由起止字段文本构造有效期；任一侧缺失即开区间，两侧都没有则未知。
pub fn validity_from_bounds(start_text: Option<&str>, end_text: Option<&str>) -> Validity {
    let start_span = start_text.and_then(parse_time_span);
    let end_span = end_text.and_then(parse_time_span);
    if start_span.is_none() && end_span.is_none() {
        return Validity::Unknown;
    }
    let start = start_span.and_then(|span| span.start.or(span.end));
    let end = end_span.and_then(|span| span.end.or(span.start));
    let granularity = start_span
        .map(|span| span.granularity)
        .unwrap_or(TimeGranularity::Year)
        .finer(
            end_span
                .map(|span| span.granularity)
                .unwrap_or(TimeGranularity::Year),
        );
    match (start, end) {
        (Some(start), Some(end)) => {
            TimeSpan::from_bounds(Some(start), Some(end), granularity)
                .map(Validity::Span)
                .unwrap_or(Validity::Never)
        }
        (Some(start), None) => Validity::Span(TimeSpan {
            start: Some(start),
            end: None,
            granularity,
        }),
        (None, Some(end)) => Validity::Span(TimeSpan {
            start: None,
            end: Some(end),
            granularity,
        }),
        (None, None) => Validity::Unknown,
    }
}

/// 从实体属性表读取有效期约定字段。
pub fn validity_from_properties(
    properties: &std::collections::HashMap<String, String>,
) -> Option<Validity> {
    let start_text = VALID_FROM_FIELDS
        .iter()
        .find_map(|field| properties.get(*field))
        .map(String::as_str);
    let end_text = VALID_TO_FIELDS
        .iter()
        .find_map(|field| properties.get(*field))
        .map(String::as_str);
    match validity_from_bounds(start_text, end_text) {
        Validity::Unknown => None,
        validity => Some(validity),
    }
}

// ---------------------------------------------------------------------------
// 会话参考时间
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalError {
    /// 相对时间需要参考时间，但会话里没有显式设置。
    UndefinedReferenceTime { expression: String },
    /// “当时/那时”需要焦点区间，但会话里没有。
    UndefinedFocus { expression: String },
    /// 区间边界非法。
    InvalidRange { detail: String },
}

impl fmt::Display for TemporalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalError::UndefinedReferenceTime { expression } => write!(
                formatter,
                "「{}」是相对时间，但当前会话没有参考时间（不会用系统时钟猜测）",
                expression
            ),
            TemporalError::UndefinedFocus { expression } => write!(
                formatter,
                "「{}」需要上下文焦点时间，但当前会话没有",
                expression
            ),
            TemporalError::InvalidRange { detail } => write!(formatter, "时间区间非法：{}", detail),
        }
    }
}

impl std::error::Error for TemporalError {}

/// 会话级时间工作记忆：参考日、焦点区间、最近一次 AsOf 区间。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemporalSession {
    reference_day: Option<TimePoint>,
    focus: Option<TimeSpan>,
    last_as_of: Option<TimeSpan>,
}

impl TemporalSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_reference_day(day: TimePoint) -> Self {
        Self {
            reference_day: Some(day),
            ..Self::default()
        }
    }

    pub fn with_reference_year(year: i32) -> Self {
        Self::with_reference_day(TimePoint::year(year))
    }

    pub fn reference_day(&self) -> Option<TimePoint> {
        self.reference_day
    }

    pub fn set_reference_day(&mut self, day: TimePoint) {
        self.reference_day = Some(day);
    }

    pub fn set_reference_year(&mut self, year: i32) {
        self.reference_day = Some(TimePoint::year(year));
    }

    pub fn clear_reference_day(&mut self) {
        self.reference_day = None;
    }

    pub fn focus(&self) -> Option<TimeSpan> {
        self.focus
    }

    pub fn set_focus(&mut self, span: TimeSpan) {
        self.focus = Some(span);
    }

    pub fn clear_focus(&mut self) {
        self.focus = None;
    }

    pub fn last_as_of(&self) -> Option<TimeSpan> {
        self.last_as_of
    }

    pub fn record_as_of(&mut self, span: TimeSpan) {
        self.focus = Some(span);
        self.last_as_of = Some(span);
    }

    /// 解析时间表达；返回 `Ok(None)` 表示这句话里没有时间表达。
    ///
    /// 绝对时间直接解析；相对时间必须命中会话参考时间，否则返回
    /// [`TemporalError::UndefinedReferenceTime`]。
    pub fn resolve(&mut self, text: &str) -> Result<Option<TimeSpan>, TemporalError> {
        if let Some(span) = parse_time_span(text) {
            self.record_as_of(span);
            return Ok(Some(span));
        }
        if let Some(span) = self.resolve_relative(text)? {
            self.record_as_of(span);
            return Ok(Some(span));
        }
        Ok(None)
    }

    fn resolve_relative(&self, text: &str) -> Result<Option<TimeSpan>, TemporalError> {
        let Some(marker) = first_relative_marker(text) else {
            return Ok(None);
        };
        if marker == "当时" || marker == "那时" {
            return self
                .focus
                .ok_or(TemporalError::UndefinedFocus {
                    expression: marker,
                })
                .map(Some);
        }
        let reference = self
            .reference_day
            .ok_or_else(|| TemporalError::UndefinedReferenceTime {
                expression: marker.clone(),
            })?;
        let span = match marker.as_str() {
            "前年" => TimeSpan::year(reference.year - 2),
            "去年" => TimeSpan::year(reference.year - 1),
            "今年" => TimeSpan::year(reference.year),
            "明年" => TimeSpan::year(reference.year + 1),
            "后年" => TimeSpan::year(reference.year + 2),
            "前天" => {
                let day = reference.prev_day().prev_day();
                TimeSpan::day(day.year, day.month as u32, day.day as u32)
            }
            "昨天" => {
                let day = reference.prev_day();
                TimeSpan::day(day.year, day.month as u32, day.day as u32)
            }
            "今天" => TimeSpan::day(reference.year, reference.month as u32, reference.day as u32),
            "明天" => {
                let day = reference.next_day();
                TimeSpan::day(day.year, day.month as u32, day.day as u32)
            }
            "大前天" => {
                let day = reference.prev_day().prev_day().prev_day();
                TimeSpan::day(day.year, day.month as u32, day.day as u32)
            }
            "大后天" => {
                let day = reference.next_day().next_day().next_day();
                TimeSpan::day(day.year, day.month as u32, day.day as u32)
            }
            _ => return Ok(None),
        };
        Ok(Some(span))
    }
}

fn first_relative_marker(text: &str) -> Option<String> {
    const MARKERS: &[&str] = &[
        "大前天", "大后天", "前年", "去年", "今年", "明年", "后年", "前天", "昨天", "今天",
        "明天", "后天", "当时", "那时",
    ];
    MARKERS
        .iter()
        .find(|marker| text.contains(**marker))
        .map(|marker| (*marker).to_string())
}

// ---------------------------------------------------------------------------
// 中文时间表达解析
// ---------------------------------------------------------------------------

fn chinese_date_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?P<bc>公元前\s*)?(?P<year>\d{1,4})\s*年(?:\s*(?P<month>\d{1,2})\s*月)?(?:\s*(?P<day>\d{1,2})\s*[日号])?",
        )
        .expect("合法中文日期正则")
    })
}

fn iso_date_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?P<year>\d{4})-(?P<month>\d{1,2})(?:-(?P<day>\d{1,2}))?")
            .expect("合法 ISO 日期正则")
    })
}

#[derive(Debug, Clone, Copy)]
struct DateCapture {
    point: TimePoint,
    granularity: TimeGranularity,
}

fn capture_from_match(capture: &regex::Captures<'_>) -> Option<DateCapture> {
    let year: i32 = capture.name("year")?.as_str().parse().ok()?;
    let year = if capture.name("bc").is_some() { -year } else { year };
    let month: Option<u32> = capture
        .name("month")
        .and_then(|value| value.as_str().parse().ok());
    let day: Option<u32> = capture
        .name("day")
        .and_then(|value| value.as_str().parse().ok());
    let granularity = if day.is_some() {
        TimeGranularity::Day
    } else if month.is_some() {
        TimeGranularity::Month
    } else {
        TimeGranularity::Year
    };
    let point = TimePoint::new(year, month.unwrap_or(1), day.unwrap_or(1));
    Some(DateCapture { point, granularity })
}

/// 解析显式时间表达为区间；句内没有时间表达时返回 `None`。
///
/// 支持：`1995年`、`1995年6月`、`1995年6月1日`、`1991年到1999年`、
/// `1995年之前/以后/以来`、`公元前221年`、`1995-06-01`。
pub fn parse_time_span(text: &str) -> Option<TimeSpan> {
    let captures: Vec<regex::Captures<'_>> = chinese_date_regex().captures_iter(text).collect();
    if captures.len() >= 2 {
        let first = capture_from_match(&captures[0])?;
        let last = capture_from_match(captures.last()?)?;
        let first_end = captures[0].get(0).map(|m| m.end()).unwrap_or(0);
        let last_start = captures
            .last()
            .and_then(|capture| capture.get(0))
            .map(|m| m.start())
            .unwrap_or(first_end);
        let connector = text
            .get(first_end..last_start)
            .unwrap_or_default()
            .chars()
            .any(|ch| matches!(ch, '到' | '至' | '~' | '～' | '-' | '—' | '–'));
        if connector {
            let start = first.point.period_start(first.granularity);
            let end = last.point.period_end(last.granularity);
            let granularity = first.granularity.finer(last.granularity);
            return TimeSpan::from_bounds(Some(start), Some(end), granularity);
        }
    }

    if let Some(span) = parse_iso_range(text) {
        return Some(span);
    }
    if let Some(span) = parse_iso_span(text) {
        return Some(span);
    }

    if captures.is_empty() {
        if let Some(point) = bare_year(text) {
            return Some(TimeSpan {
                start: Some(point.period_start(TimeGranularity::Year)),
                end: Some(point.period_end(TimeGranularity::Year)),
                granularity: TimeGranularity::Year,
            });
        }
        return None;
    }
    let first = capture_from_match(&captures[0])?;
    let last = capture_from_match(captures.last()?)?;

    if text.contains("之前") || text.contains("以前") {
        return Some(TimeSpan::before(
            first.point.period_start(first.granularity),
            first.granularity,
        ));
    }
    if text.contains("之后") || text.contains("以后") {
        return Some(TimeSpan::after(
            last.point.period_end(last.granularity),
            last.granularity,
        ));
    }
    if text.contains("以来") {
        return Some(TimeSpan::since(
            first.point.period_start(first.granularity),
            first.granularity,
        ));
    }
    if text.contains("至今") || text.contains("直到现在") || text.contains("到现在") {
        return Some(TimeSpan {
            start: Some(first.point.period_start(first.granularity)),
            end: None,
            granularity: first.granularity,
        });
    }
    Some(TimeSpan {
        start: Some(first.point.period_start(first.granularity)),
        end: Some(first.point.period_end(first.granularity)),
        granularity: first.granularity,
    })
}

fn parse_iso_range(text: &str) -> Option<TimeSpan> {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    let regex = REGEX.get_or_init(|| {
        Regex::new(r"(?P<a>\d{4})\s*[-~～—–]\s*(?P<b>\d{4})").expect("合法 ISO 年区间正则")
    });
    let capture = regex.captures(text)?;
    let start: i32 = capture.name("a")?.as_str().parse().ok()?;
    let end: i32 = capture.name("b")?.as_str().parse().ok()?;
    if start > end {
        return None;
    }
    TimeSpan::from_bounds(
        Some(TimePoint::year(start)),
        Some(TimePoint::new(end, 12, 31)),
        TimeGranularity::Year,
    )
}

fn parse_iso_span(text: &str) -> Option<TimeSpan> {
    let capture = iso_date_regex().captures(text)?;
    let month: u32 = capture.name("month")?.as_str().parse().ok()?;
    let day: u32 = capture
        .name("day")
        .and_then(|value| value.as_str().parse().ok())
        .unwrap_or(1);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let date = capture_from_match(&capture)?;
    Some(TimeSpan {
        start: Some(date.point.period_start(date.granularity)),
        end: Some(date.point.period_end(date.granularity)),
        granularity: date.granularity,
    })
}

fn bare_year(text: &str) -> Option<TimePoint> {
    let trimmed = text.trim();
    let (negative, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed),
    };
    if (1..=4).contains(&digits.len()) && digits.chars().all(|ch| ch.is_ascii_digit()) {
        let year: i32 = digits.parse().ok()?;
        return Some(TimePoint::year(if negative { -year } else { year }));
    }
    None
}

/// 解析单个时间点。除中文/ISO 表达外，也接受裸年份（如属性值 `"1868"`）。
pub fn parse_time_point(text: &str) -> Option<TimePoint> {
    if let Some(capture) = chinese_date_regex().captures(text) {
        return capture_from_match(&capture).map(|date| date.point);
    }
    if let Some(capture) = iso_date_regex().captures(text) {
        return capture_from_match(&capture).map(|date| date.point);
    }
    bare_year(text)
}

/// 解析有效期：显式“一直/长期”为无界，显式时间为区间，否则未知。
pub fn parse_validity(text: &str) -> Validity {
    if text.contains("一直")
        || text.contains("长期")
        || text.contains("始终")
        || text.contains("自古以来")
        || text.contains("永久")
    {
        return Validity::Unbounded;
    }
    parse_time_span(text)
        .map(Validity::Span)
        .unwrap_or(Validity::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn time_point_ordering_and_arithmetic() {
        let day = TimePoint::day(1995, 12, 31);
        assert_eq!(day.next_day(), TimePoint::day(1996, 1, 1));
        assert_eq!(day.next_day().prev_day(), day);
        assert!(TimePoint::year(1995) < TimePoint::day(1995, 6, 1));
        assert_eq!(TimePoint::new(1995, 13, 40), TimePoint::day(1995, 12, 31));
    }

    #[test]
    fn time_span_intersection_and_overlap() {
        let left = TimeSpan::year(1991);
        let right = TimeSpan::from_bounds(
            Some(TimePoint::year(1990)),
            Some(TimePoint::year(1995)),
            TimeGranularity::Year,
        )
        .unwrap();
        let overlap = left.intersect(&right).unwrap();
        assert_eq!(overlap.render(), "1991年");
        assert!(left.overlaps(&right));
        let disjoint = TimeSpan::year(2000);
        assert!(!left.overlaps(&disjoint));
        assert!(!left.contains(TimePoint::year(1995)));
    }

    #[test]
    fn unbounded_spans_intersect_with_bounded() {
        let all = TimeSpan::unbounded();
        let span = TimeSpan::year(1995);
        assert_eq!(all.intersect(&span), Some(span));
        assert!(all.contains(TimePoint::day(1900, 1, 1)));
    }

    #[test]
    fn validity_distinguishes_unknown_from_unbounded() {
        let mut left = Validity::Span(TimeSpan::year(1995));
        let right = Validity::Span(TimeSpan::year(1996));
        assert_eq!(left.intersection(right), Validity::Never);
        left = Validity::Unknown;
        assert_eq!(left.intersection(Validity::Unbounded), Validity::Unknown);
        assert_eq!(
            Validity::Unbounded.intersection(Validity::Span(TimeSpan::year(1995))),
            Validity::Span(TimeSpan::year(1995))
        );
    }

    #[test]
    fn constraint_enforces_intersection() {
        let constraint = TemporalConstraint::as_of(1995);
        let valid_1990s = Validity::from_years(1990, 1999);
        let future = Validity::from_years(2000, 2010);
        assert!(constraint.allows(valid_1990s, UnknownTimePolicy::Strict));
        assert!(!constraint.allows(future, UnknownTimePolicy::Strict));
        assert!(!constraint.allows(Validity::Unknown, UnknownTimePolicy::Strict));
        assert!(constraint.allows(Validity::Unknown, UnknownTimePolicy::Background));
    }

    #[test]
    fn parse_chinese_dates_and_ranges() {
        assert_eq!(
            parse_time_span("出生于1995年6月1日"),
            TimeSpan::day(1995, 6, 1).into()
        );
        assert_eq!(parse_time_span("1995年6月"), Some(TimeSpan::month(1995, 6)));
        assert_eq!(parse_time_span("1995年"), Some(TimeSpan::year(1995)));
        assert_eq!(
            parse_time_span("从1991年到1999年"),
            TimeSpan::from_bounds(
                Some(TimePoint::year(1991)),
                Some(TimePoint::new(1999, 12, 31)),
                TimeGranularity::Year
            )
        );
        assert_eq!(
            parse_time_span("1995年之前"),
            Some(TimeSpan::before(TimePoint::year(1995), TimeGranularity::Year))
        );
        assert_eq!(parse_time_point("公元前221年"), Some(TimePoint::year(-221)));
        assert_eq!(
            parse_time_point("1995-06-01"),
            Some(TimePoint::day(1995, 6, 1))
        );
        assert_eq!(
            parse_time_span("1991-1999"),
            TimeSpan::from_bounds(
                Some(TimePoint::year(1991)),
                Some(TimePoint::new(1999, 12, 31)),
                TimeGranularity::Year
            )
        );
        // 两个互不连接的时间点不能被误读成一个区间。
        assert_eq!(
            parse_time_span("1991年出生，1999年上学"),
            Some(TimeSpan::year(1991))
        );
        assert!(parse_time_span("没有时间").is_none());
    }

    #[test]
    fn unknown_vs_unbounded_parsing() {
        assert_eq!(parse_validity("没有时间信息"), Validity::Unknown);
        assert_eq!(parse_validity("一直有效"), Validity::Unbounded);
        assert_eq!(
            parse_validity("1995年"),
            Validity::Span(TimeSpan::year(1995))
        );
    }

    #[test]
    fn session_relative_time_requires_reference() {
        let mut session = TemporalSession::new();
        let error = session.resolve("去年").unwrap_err();
        assert!(matches!(
            error,
            TemporalError::UndefinedReferenceTime { .. }
        ));
        session.set_reference_year(2026);
        assert_eq!(session.resolve("去年").unwrap(), Some(TimeSpan::year(2025)));
        assert_eq!(
            session.resolve("那1991年呢").unwrap(),
            Some(TimeSpan::year(1991))
        );
        assert_eq!(session.last_as_of(), Some(TimeSpan::year(1991)));
    }

    #[test]
    fn session_focus_resolves_anaphora() {
        let mut session = TemporalSession::with_reference_year(2026);
        session.record_as_of(TimeSpan::year(1991));
        assert_eq!(session.focus(), Some(TimeSpan::year(1991)));
        assert_eq!(session.resolve("当时").unwrap(), Some(TimeSpan::year(1991)));
        session.clear_focus();
        assert!(matches!(
            session.resolve("当时").unwrap_err(),
            TemporalError::UndefinedFocus { .. }
        ));
    }

    #[test]
    fn temporal_index_valid_at_and_timeline() {
        let mut index = TemporalIndex::new();
        index.insert(
            TemporalFact::new("东京", "首都", "日本")
                .with_years(1868, 9999)
                .with_source("编年史"),
        );
        index.insert(TemporalFact::new("京都", "首都", "日本").with_years(-794, 1868));
        let valid_now = index.valid_at(TimePoint::year(1868));
        assert_eq!(valid_now.len(), 2);
        let timeline = index.timeline();
        assert_eq!(timeline[0].subject, "京都");
        assert!(index.for_entity("日本").len() == 2);
    }

    #[test]
    fn functional_conflicts_respect_time() {
        let mut index = TemporalIndex::new();
        index.insert(TemporalFact::new("日本", "首都", "京都").with_years(794, 1867));
        index.insert(TemporalFact::new("日本", "首都", "东京").with_years(1868, 9999));
        let functional: BTreeSet<String> = ["首都".to_string()].into_iter().collect();
        assert!(index.functional_conflicts(&functional).is_empty());

        index.insert(TemporalFact::new("日本", "首都", "大阪").with_years(1800, 1900));
        let conflicts = index.functional_conflicts(&functional);
        assert_eq!(conflicts.len(), 2);
        assert!(conflicts[0].render().contains("重叠"));
    }

    #[test]
    fn unknown_time_is_not_a_conflict() {
        let mut index = TemporalIndex::new();
        index.insert(TemporalFact::new("日本", "首都", "京都"));
        index.insert(TemporalFact::new("日本", "首都", "东京").with_years(1868, 1868));
        let functional: BTreeSet<String> = ["首都".to_string()].into_iter().collect();
        assert!(index.functional_conflicts(&functional).is_empty());
    }

    #[test]
    fn prolog_facts_build_temporal_index() {
        let mut properties = HashMap::new();
        properties.insert("valid_from".to_string(), "1868".to_string());
        properties.insert("valid_to".to_string(), "9999".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "东京".to_string(),
                entity_type: "City".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "东京".to_string(),
                rel_type: "首都".to_string(),
                to: "日本".to_string(),
            },
        ];
        let index = TemporalIndex::from_prolog_facts(&facts);
        assert_eq!(index.len(), 1);
        assert_eq!(index.facts()[0].validity, Validity::from_years(1868, 9999));
    }

    #[test]
    fn prolog_name_alias_keeps_validity() {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), "京都".to_string());
        properties.insert("valid_to".to_string(), "1867".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "kyoto".to_string(),
                entity_type: "City".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "京都".to_string(),
                rel_type: "capital".to_string(),
                to: "日本".to_string(),
            },
        ];
        let index = TemporalIndex::from_prolog_facts(&facts);
        let at = |year| {
            index
                .valid_at(TimePoint::year(year))
                .iter()
                .any(|fact| fact.subject == "京都" && fact.predicate == "capital")
        };
        assert!(at(1800));
        assert!(!at(1868));
    }

    #[test]
    fn constraint_from_span_prefers_point() {
        let point = TimeSpan::day(1995, 6, 1);
        assert_eq!(
            TemporalConstraint::from_span(point),
            TemporalConstraint::AsOf(TimePoint::day(1995, 6, 1))
        );
        let range = TimeSpan::from_bounds(
            Some(TimePoint::year(1991)),
            Some(TimePoint::year(1999)),
            TimeGranularity::Year,
        )
        .unwrap();
        assert_eq!(
            TemporalConstraint::from_span(range),
            TemporalConstraint::During(range)
        );
    }
}

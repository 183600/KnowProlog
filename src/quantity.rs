//! 数量语义（Quantity semantics）：一处解析、一处比较、一处排序。
//!
//! 真实知识库里的数值很少是裸数字：`226 厘米`、`67万平方公里`、`约46亿年`
//! 都同时携带数值、量纲与精确度。把这三件事分开建模后：
//!
//! - **量纲先行**：长度、质量、时间、面积、体积、温度、计数、比例、货币
//!   各自独立；跨量纲比较返回 [`Comparison::Incomparable`]，不再偷偷按
//!   字典序或裸数字排序。
//! - **单位不是字符串**：单位在 [`UNITS`] 里声明为 `&'static Unit`，解析期
//!   查表，比较期只做量纲内的基准值换算。
//! - **约数不伪装成精确值**：`约`/`多`/`余` 记为
//!   [`Exactness::Approximate`]，`超过`/`不足` 记为单侧区间；区间重叠时
//!   返回 [`Comparison::Uncertain`]，而不是硬判冠军。
//! - **显式偏序**：[`Quantity`] 不实现 `Ord`，调用方必须处理
//!   `Uncertain` 与 `Incomparable`；排序由 [`rank`] 提供展示用稳定名次，
//!   并把不可比项单独返回。
//!
//! 本模块不依赖知识库、不写对话状态，可独立测试；`bindings`、`query_board`
//! 与 `binding_frames` 共用同一入口，避免三套数值解析各说各话。

use crate::chinese_reasoning_nlu::parse_chinese_number_u64;

/// 可比较的量纲。枚举顺序也用作确定性排序的次级键。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dimension {
    /// 长度（基准：米）。
    Length,
    /// 质量（基准：千克）。
    Mass,
    /// 时间/年龄（基准：年）。
    Duration,
    /// 面积（基准：平方米）。
    Area,
    /// 体积（基准：立方米）。
    Volume,
    /// 温度（基准：摄氏度）。
    Temperature,
    /// 计数（基准：个；无单位数字也归入此类）。
    Count,
    /// 比例/百分比（基准：百分点）。
    Ratio,
    /// 货币（基准：元；不做汇率换算）。
    Currency,
}

impl Dimension {
    /// 人话量纲名，用于「不可比」的解释。
    pub fn label(self) -> &'static str {
        match self {
            Dimension::Length => "长度",
            Dimension::Mass => "质量",
            Dimension::Duration => "时间",
            Dimension::Area => "面积",
            Dimension::Volume => "体积",
            Dimension::Temperature => "温度",
            Dimension::Count => "计数",
            Dimension::Ratio => "比例",
            Dimension::Currency => "货币",
        }
    }

    /// 该量纲的基准单位；`format_canonical` 用它输出换算后的值。
    pub fn canonical_unit(self) -> &'static str {
        match self {
            Dimension::Length => "米",
            Dimension::Mass => "千克",
            Dimension::Duration => "年",
            Dimension::Area => "平方米",
            Dimension::Volume => "立方米",
            Dimension::Temperature => "摄氏度",
            Dimension::Count => "个",
            Dimension::Ratio => "%",
            Dimension::Currency => "元",
        }
    }

    /// 默认比较级用词（`higher`/`larger` 的中文表达）。
    ///
    /// 属性词优先：`年龄` 说「年长」、`体重` 说「重」，其余按量纲回退。
    pub fn comparative_word(self, attribute: &str) -> &'static str {
        if attribute.contains("年龄") || attribute.contains('岁') {
            return "年长";
        }
        if attribute.contains("体重") || attribute.contains('重') {
            return "重";
        }
        if attribute.contains("身高") || attribute.contains('高') {
            return "高";
        }
        if attribute.contains("面积") || attribute.contains("体积") {
            return "大";
        }
        match self {
            Dimension::Length => "高",
            Dimension::Mass => "重",
            Dimension::Duration => "久",
            Dimension::Area | Dimension::Volume => "大",
            Dimension::Temperature => "高",
            Dimension::Count | Dimension::Ratio | Dimension::Currency => "多",
        }
    }
}

/// 单位声明：基准值 = `value * factor + offset`。
///
/// `offset` 只用于温标（摄氏/华氏/开尔文）；其余单位为 0。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unit {
    pub dimension: Dimension,
    /// 该单位换算到量纲基准单位时的乘数。
    pub factor: f64,
    /// 该单位换算到量纲基准单位时的加数。
    pub offset: f64,
}

const fn unit(dimension: Dimension, factor: f64, offset: f64) -> Unit {
    Unit {
        dimension,
        factor,
        offset,
    }
}

/// 手写单位表：只覆盖知识库里实际出现的量纲，不做单位本体学习。
///
/// 查表按整词匹配；同一单位允许多个常用写法（中文、符号、全角）。
pub static UNITS: &[(&str, Unit)] = &[
    // 长度
    ("平方米", unit(Dimension::Area, 1.0, 0.0)),
    ("平方公尺", unit(Dimension::Area, 1.0, 0.0)),
    ("平方公里", unit(Dimension::Area, 1_000_000.0, 0.0)),
    ("平方千米", unit(Dimension::Area, 1_000_000.0, 0.0)),
    ("平方厘米", unit(Dimension::Area, 0.0001, 0.0)),
    ("平米", unit(Dimension::Area, 1.0, 0.0)),
    ("公顷", unit(Dimension::Area, 10_000.0, 0.0)),
    ("亩", unit(Dimension::Area, 666.666_666_666_666_7, 0.0)),
    ("千米", unit(Dimension::Length, 1000.0, 0.0)),
    ("公里", unit(Dimension::Length, 1000.0, 0.0)),
    ("厘米", unit(Dimension::Length, 0.01, 0.0)),
    ("毫米", unit(Dimension::Length, 0.001, 0.0)),
    ("分米", unit(Dimension::Length, 0.1, 0.0)),
    ("公尺", unit(Dimension::Length, 1.0, 0.0)),
    ("英里", unit(Dimension::Length, 1609.344, 0.0)),
    ("英尺", unit(Dimension::Length, 0.3048, 0.0)),
    ("英寸", unit(Dimension::Length, 0.0254, 0.0)),
    ("cm", unit(Dimension::Length, 0.01, 0.0)),
    ("mm", unit(Dimension::Length, 0.001, 0.0)),
    ("dm", unit(Dimension::Length, 0.1, 0.0)),
    ("km", unit(Dimension::Length, 1000.0, 0.0)),
    ("ft", unit(Dimension::Length, 0.3048, 0.0)),
    ("in", unit(Dimension::Length, 0.0254, 0.0)),
    ("米", unit(Dimension::Length, 1.0, 0.0)),
    ("m", unit(Dimension::Length, 1.0, 0.0)),
    ("里", unit(Dimension::Length, 500.0, 0.0)),
    ("丈", unit(Dimension::Length, 10.0 / 3.0, 0.0)),
    ("尺", unit(Dimension::Length, 1.0 / 3.0, 0.0)),
    ("寸", unit(Dimension::Length, 1.0 / 30.0, 0.0)),
    // 面积/体积
    ("立方米", unit(Dimension::Volume, 1.0, 0.0)),
    ("立方公尺", unit(Dimension::Volume, 1.0, 0.0)),
    ("立方厘米", unit(Dimension::Volume, 0.000001, 0.0)),
    ("毫升", unit(Dimension::Volume, 0.000001, 0.0)),
    ("升", unit(Dimension::Volume, 0.001, 0.0)),
    ("ml", unit(Dimension::Volume, 0.000001, 0.0)),
    ("mL", unit(Dimension::Volume, 0.000001, 0.0)),
    ("L", unit(Dimension::Volume, 0.001, 0.0)),
    // 质量
    ("千克", unit(Dimension::Mass, 1.0, 0.0)),
    ("公斤", unit(Dimension::Mass, 1.0, 0.0)),
    ("毫克", unit(Dimension::Mass, 0.000001, 0.0)),
    ("克", unit(Dimension::Mass, 0.001, 0.0)),
    ("吨", unit(Dimension::Mass, 1000.0, 0.0)),
    ("斤", unit(Dimension::Mass, 0.5, 0.0)),
    ("两", unit(Dimension::Mass, 0.05, 0.0)),
    ("磅", unit(Dimension::Mass, 0.453_592_37, 0.0)),
    ("kg", unit(Dimension::Mass, 1.0, 0.0)),
    ("mg", unit(Dimension::Mass, 0.000001, 0.0)),
    ("g", unit(Dimension::Mass, 0.001, 0.0)),
    ("t", unit(Dimension::Mass, 1000.0, 0.0)),
    ("lb", unit(Dimension::Mass, 0.453_592_37, 0.0)),
    // 时间/年龄（基准：年）
    ("个月", unit(Dimension::Duration, 1.0 / 12.0, 0.0)),
    ("星期", unit(Dimension::Duration, 7.0 / 365.0, 0.0)),
    ("小时", unit(Dimension::Duration, 1.0 / 8760.0, 0.0)),
    ("分钟", unit(Dimension::Duration, 1.0 / 525_600.0, 0.0)),
    ("年", unit(Dimension::Duration, 1.0, 0.0)),
    ("岁", unit(Dimension::Duration, 1.0, 0.0)),
    ("月", unit(Dimension::Duration, 1.0 / 12.0, 0.0)),
    ("天", unit(Dimension::Duration, 1.0 / 365.0, 0.0)),
    ("日", unit(Dimension::Duration, 1.0 / 365.0, 0.0)),
    ("周", unit(Dimension::Duration, 7.0 / 365.0, 0.0)),
    ("秒", unit(Dimension::Duration, 1.0 / 31_536_000.0, 0.0)),
    // 温度
    ("摄氏度", unit(Dimension::Temperature, 1.0, 0.0)),
    ("华氏度", unit(Dimension::Temperature, 5.0 / 9.0, -32.0 * 5.0 / 9.0)),
    ("开尔文", unit(Dimension::Temperature, 1.0, -273.15)),
    ("℃", unit(Dimension::Temperature, 1.0, 0.0)),
    ("°C", unit(Dimension::Temperature, 1.0, 0.0)),
    ("°F", unit(Dimension::Temperature, 5.0 / 9.0, -32.0 * 5.0 / 9.0)),
    ("K", unit(Dimension::Temperature, 1.0, -273.15)),
    // 计数
    ("个", unit(Dimension::Count, 1.0, 0.0)),
    ("只", unit(Dimension::Count, 1.0, 0.0)),
    ("人", unit(Dimension::Count, 1.0, 0.0)),
    ("位", unit(Dimension::Count, 1.0, 0.0)),
    ("名", unit(Dimension::Count, 1.0, 0.0)),
    ("条", unit(Dimension::Count, 1.0, 0.0)),
    ("张", unit(Dimension::Count, 1.0, 0.0)),
    ("本", unit(Dimension::Count, 1.0, 0.0)),
    ("台", unit(Dimension::Count, 1.0, 0.0)),
    ("辆", unit(Dimension::Count, 1.0, 0.0)),
    ("头", unit(Dimension::Count, 1.0, 0.0)),
    ("匹", unit(Dimension::Count, 1.0, 0.0)),
    ("件", unit(Dimension::Count, 1.0, 0.0)),
    ("次", unit(Dimension::Count, 1.0, 0.0)),
    ("份", unit(Dimension::Count, 1.0, 0.0)),
    ("家", unit(Dimension::Count, 1.0, 0.0)),
    ("所", unit(Dimension::Count, 1.0, 0.0)),
    ("座", unit(Dimension::Count, 1.0, 0.0)),
    ("枚", unit(Dimension::Count, 1.0, 0.0)),
    ("颗", unit(Dimension::Count, 1.0, 0.0)),
    ("支", unit(Dimension::Count, 1.0, 0.0)),
    ("首", unit(Dimension::Count, 1.0, 0.0)),
    ("篇", unit(Dimension::Count, 1.0, 0.0)),
    ("项", unit(Dimension::Count, 1.0, 0.0)),
    ("种", unit(Dimension::Count, 1.0, 0.0)),
    // 比例
    ("个百分点", unit(Dimension::Ratio, 1.0, 0.0)),
    ("%", unit(Dimension::Ratio, 1.0, 0.0)),
    ("％", unit(Dimension::Ratio, 1.0, 0.0)),
    ("倍", unit(Dimension::Ratio, 1.0, 0.0)),
    // 货币（不做跨币种换算）
    ("人民币", unit(Dimension::Currency, 1.0, 0.0)),
    ("元", unit(Dimension::Currency, 1.0, 0.0)),
    ("块", unit(Dimension::Currency, 1.0, 0.0)),
];

/// 查单位。返回静态引用，比较时无需再分配字符串。
pub fn lookup_unit(text: &str) -> Option<&'static Unit> {
    lookup_unit_entry(text).map(|(_, unit)| unit)
}

/// 同时取出单位写法与定义；写法用于把差值写回用户熟悉的单位。
fn lookup_unit_entry(text: &str) -> Option<(&'static str, &'static Unit)> {
    UNITS
        .iter()
        .find(|(alias, _)| *alias == text)
        .map(|(alias, unit)| (*alias, unit))
}

/// 数值的精确度。区间语义让「约」不再被当成精确值。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exactness {
    /// 精确值。
    Exact,
    /// 约数（`约`/`大约`/`多`/`余`/`左右`）：中心值 ±5%。
    Approximate,
    /// 下界（`超过`/`至少`/`多于`）。
    AtLeast,
    /// 上界（`不足`/`不到`/`至多`）。
    AtMost,
}

impl Exactness {
    pub fn label(self) -> &'static str {
        match self {
            Exactness::Exact => "精确",
            Exactness::Approximate => "约",
            Exactness::AtLeast => "至少",
            Exactness::AtMost => "至多",
        }
    }

    /// 区间上下界；无穷表示单侧无界。
    fn bounds(self, value: f64) -> (f64, f64) {
        match self {
            Exactness::Exact => (value, value),
            Exactness::Approximate => {
                let margin = (value.abs() * 0.05).max(1e-9);
                (value - margin, value + margin)
            }
            Exactness::AtLeast => (value, f64::INFINITY),
            Exactness::AtMost => (f64::NEG_INFINITY, value),
        }
    }
}

/// 解析后的数量：原始写法 + 量纲内的基准值 + 精确度。
///
/// 不实现 `Ord`：数量比较是偏序，跨量纲与约数重叠都必须显式返回。
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    raw: String,
    /// 统一到 `dimension` 基准单位后的值（温标已处理 offset）。
    base: f64,
    dimension: Dimension,
    /// 原始单位；无单位数字为 `None`。
    unit: Option<&'static Unit>,
    /// 单位在 [`UNITS`] 中的原始写法，用于把差值写回用户熟悉的单位。
    unit_alias: Option<&'static str>,
    exactness: Exactness,
}

impl Quantity {
    /// 原始写法，用于生成时保留证据原文。
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// 基准单位下的数值（如厘米已换算成米）。
    pub fn base_value(&self) -> f64 {
        self.base
    }

    pub fn dimension(&self) -> Dimension {
        self.dimension
    }

    pub fn unit(&self) -> Option<&'static Unit> {
        self.unit
    }

    /// 单位原始写法（如 `厘米`、`cm`）；无单位数字为 `None`。
    pub fn unit_alias(&self) -> Option<&'static str> {
        self.unit_alias
    }

    pub fn exactness(&self) -> Exactness {
        self.exactness
    }

    /// 是否有显式单位；无单位数字归入计数。
    pub fn has_unit(&self) -> bool {
        self.unit.is_some()
    }

    /// 同量纲才可比较；单位有无不再是障碍（无单位数字按计数处理）。
    pub fn compare(&self, other: &Quantity) -> Comparison {
        if self.dimension != other.dimension {
            return Comparison::Incomparable(Incomparable::DimensionMismatch {
                left: self.dimension,
                right: other.dimension,
            });
        }
        let (left_low, left_high) = self.exactness.bounds(self.base);
        let (right_low, right_high) = other.exactness.bounds(other.base);
        // 两个精确值先做近似相等判定，避免 226 厘米换算后的浮点尾差
        // 被区间比较误判成严格小于。
        if self.exactness == Exactness::Exact && other.exactness == Exactness::Exact {
            return if nearly_equal(self.base, other.base) {
                Comparison::Equal
            } else if self.base > other.base {
                Comparison::Greater
            } else {
                Comparison::Less
            };
        }
        if left_low > right_high {
            return Comparison::Greater;
        }
        if left_high < right_low {
            return Comparison::Less;
        }
        Comparison::Uncertain
    }

    /// 同量纲的基准值差（`self - other`）。跨量纲显式拒绝。
    pub fn difference(&self, other: &Quantity) -> Result<f64, Incomparable> {
        if self.dimension != other.dimension {
            return Err(Incomparable::DimensionMismatch {
                left: self.dimension,
                right: other.dimension,
            });
        }
        Ok(self.base - other.base)
    }

    /// 同量纲差值，换算回 `self` 的原始单位，供「相差 10 厘米」这类解释使用。
    ///
    /// 返回 `(差值, 单位写法)`；无单位计数返回 `None` 作单位，调用方不必
    /// 强行补一个「个」。跨量纲仍显式拒绝。
    pub fn difference_in_own_unit(
        &self,
        other: &Quantity,
    ) -> Result<(f64, Option<&'static str>), Incomparable> {
        let difference = self.difference(other)?;
        match (self.unit, self.unit_alias) {
            (Some(unit), Some(alias)) if unit.factor != 0.0 => {
                Ok((difference / unit.factor, Some(alias)))
            }
            _ if self.dimension == Dimension::Count => Ok((difference, None)),
            _ => Ok((difference, Some(self.dimension.canonical_unit()))),
        }
    }

    /// 换算后的规范写法，如 `226厘米` → `2.26米`。
    pub fn format_canonical(&self) -> String {
        let prefix = match self.exactness {
            Exactness::Exact => "",
            Exactness::Approximate => "约",
            Exactness::AtLeast => "至少",
            Exactness::AtMost => "至多",
        };
        if self.dimension == Dimension::Count && self.unit.is_none() {
            return format!("{}{}", prefix, format_number(self.base));
        }
        format!(
            "{}{}{}",
            prefix,
            format_number(self.base),
            self.dimension.canonical_unit()
        )
    }
}

fn nearly_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1e-12
}

/// 数量的三值比较 + 两个显式「不能比」出口。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    Less,
    Equal,
    Greater,
    /// 约数区间重叠：只能确定「可能相等，也可能一大一小」。
    Uncertain,
    /// 量纲不同或一侧不是数量。
    Incomparable(Incomparable),
}

impl Comparison {
    /// 只在确定有序时给出 `Ordering`；`Uncertain`/`Incomparable` 返回 `None`。
    pub fn ordering(self) -> Option<std::cmp::Ordering> {
        match self {
            Comparison::Less => Some(std::cmp::Ordering::Less),
            Comparison::Equal => Some(std::cmp::Ordering::Equal),
            Comparison::Greater => Some(std::cmp::Ordering::Greater),
            Comparison::Uncertain | Comparison::Incomparable(_) => None,
        }
    }

    /// 确定更大/更小/相等时为 `true`；不确定或不可比时为 `false`。
    pub fn is_definite(self) -> bool {
        self.ordering().is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Incomparable {
    /// 两侧量纲不同（例如长度与质量）。
    DimensionMismatch {
        left: Dimension,
        right: Dimension,
    },
    /// 至少一侧无法解析成数量。
    NotAQuantity,
}

impl Incomparable {
    pub fn describe(self) -> String {
        match self {
            Incomparable::DimensionMismatch { left, right } => {
                format!("量纲不同（{} vs {}）", left.label(), right.label())
            }
            Incomparable::NotAQuantity => "至少一侧不是数值".to_string(),
        }
    }
}

/// 解析「数值 + 可选单位 + 精确度」。
///
/// 支持：
/// - 阿拉伯数字、小数、正负号；
/// - 中文数字（`四十六亿`）、阿拉伯数字后的 `万/亿/千/百/十`；
/// - 常用中文单位与符号（见 [`UNITS`]）；
/// - `约/大约/多/余/左右` 与 `超过/至少/不足/至多` 前缀。
///
/// 解析失败返回 `None`，调用方不得把未知文本猜成 0。
pub fn parse_quantity(raw: &str) -> Option<Quantity> {
    let trimmed = raw.trim().trim_matches(|ch: char| {
        matches!(
            ch,
            '。' | '，' | ',' | '；' | ';' | '！' | '!' | '？' | '?' | '"' | '\'' | '“' | '”'
        )
    });
    if trimmed.is_empty() {
        return None;
    }
    let compact: String = trimmed.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.is_empty() {
        return None;
    }
    let (mut exactness, body) = strip_exactness(&compact);
    let (number_text, rest) = split_number(&body)?;
    let value = parse_number_prefix(number_text)?;
    if !value.is_finite() {
        return None;
    }

    let mut rest = rest
        .trim_start_matches('的')
        .trim_matches(|ch: char| matches!(ch, '。' | '，' | ',' | '；' | ';' | '！' | '!'));
    // 「100多米」「3万余人」：多/余写在数字和单位之间，语义是约数。
    if let Some(stripped) = rest.strip_prefix('多').or_else(|| rest.strip_prefix('余')) {
        if exactness == Exactness::Exact {
            exactness = Exactness::Approximate;
        }
        rest = stripped;
    }

    if rest.is_empty() {
        return Some(Quantity {
            raw: raw.to_string(),
            base: value,
            dimension: Dimension::Count,
            unit: None,
            unit_alias: None,
            exactness,
        });
    }

    let (unit_alias, unit) = lookup_unit_entry(rest)?;
    Some(Quantity {
        raw: raw.to_string(),
        base: value * unit.factor + unit.offset,
        dimension: unit.dimension,
        unit: Some(unit),
        unit_alias: Some(unit_alias),
        exactness,
    })
}

fn strip_exactness(text: &str) -> (Exactness, String) {
    let mut exactness = Exactness::Exact;
    let mut body = text.to_string();
    // 前缀按长度降序，避免「不超过」被「超过」抢先匹配。
    const PREFIXES: &[(&str, Exactness)] = &[
        ("不低于", Exactness::AtLeast),
        ("不少于", Exactness::AtLeast),
        ("不小于", Exactness::AtLeast),
        ("不超过", Exactness::AtMost),
        ("不大于", Exactness::AtMost),
        ("大于", Exactness::AtLeast),
        ("多于", Exactness::AtLeast),
        ("高于", Exactness::AtLeast),
        ("超过", Exactness::AtLeast),
        ("至少", Exactness::AtLeast),
        ("小于", Exactness::AtMost),
        ("少于", Exactness::AtMost),
        ("低于", Exactness::AtMost),
        ("不足", Exactness::AtMost),
        ("不到", Exactness::AtMost),
        ("至多", Exactness::AtMost),
        ("最多", Exactness::AtMost),
        ("将近", Exactness::Approximate),
        ("接近", Exactness::Approximate),
        ("大约", Exactness::Approximate),
        ("大概", Exactness::Approximate),
        ("约", Exactness::Approximate),
        ("近", Exactness::Approximate),
    ];
    for (prefix, kind) in PREFIXES {
        if let Some(rest) = body.strip_prefix(prefix) {
            exactness = *kind;
            body = rest.trim_start_matches([' ', '　']).to_string();
            break;
        }
    }
    for suffix in ["左右", "上下"] {
        if let Some(stripped) = body.strip_suffix(suffix) {
            if exactness == Exactness::Exact {
                exactness = Exactness::Approximate;
            }
            body = stripped.to_string();
            break;
        }
    }
    (exactness, body)
}

/// 切出数字前缀，返回（数字部分，余下的单位部分）。
fn split_number(text: &str) -> Option<(&str, &str)> {
    let mut end = 0;
    let mut seen_number = false;
    let mut seen_dot = false;
    for (index, ch) in text.char_indices() {
        let starts_number = (ch == '-' || ch == '+') && index == 0;
        if ch.is_ascii_digit() {
            seen_number = true;
            end = index + ch.len_utf8();
        } else if ch == '.' {
            if !seen_number || seen_dot {
                break;
            }
            seen_dot = true;
            end = index + ch.len_utf8();
        } else if starts_number {
            end = index + ch.len_utf8();
        } else if is_chinese_number_char(ch) {
            // 中文数字与万/亿/十/百/千都属于数字前缀；「千米」「千克」这类
            // 以倍数开头的单位会在乘法后回退成正确基准值。
            seen_number = true;
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    if !seen_number {
        return None;
    }
    Some((&text[..end], &text[end..]))
}

fn is_chinese_number_char(ch: char) -> bool {
    matches!(
        ch,
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
            | '万'
            | '亿'
    )
}

fn parse_number_prefix(text: &str) -> Option<f64> {
    if text.is_empty() {
        return None;
    }
    let has_ascii = text.chars().any(|ch| ch.is_ascii_digit());
    if !has_ascii {
        return parse_chinese_number_u64(text).map(|value| value as f64);
    }
    let mut sign = 1.0;
    let mut body = text;
    if let Some(rest) = body.strip_prefix('-') {
        sign = -1.0;
        body = rest;
    } else if let Some(rest) = body.strip_prefix('+') {
        body = rest;
    }
    let mut end = 0;
    for (index, ch) in body.char_indices() {
        if ch.is_ascii_digit() || ch == '.' {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let mut value: f64 = body[..end].parse().ok()?;
    for ch in body[end..].chars() {
        value *= match ch {
            '万' => 10_000.0,
            '亿' => 100_000_000.0,
            '千' => 1_000.0,
            '百' => 100.0,
            '十' => 10.0,
            _ => return None,
        };
    }
    Some(sign * value)
}

/// 格式化数值：整数不带小数点，小数最多两位且去掉尾零。
pub fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
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

/// 排序请求：方向 + 页码式偏移 + 条数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankSpec {
    pub descending: bool,
    pub offset: usize,
    pub limit: usize,
}

impl Default for RankSpec {
    fn default() -> Self {
        RankSpec {
            descending: true,
            offset: 0,
            limit: 1,
        }
    }
}

/// 一条带名次的记录。
#[derive(Debug, Clone, PartialEq)]
pub struct Ranked<T> {
    pub item: T,
    pub quantity: Quantity,
    /// 竞赛式名次（并列后跳号：1, 1, 3）。
    pub rank: usize,
    /// 与相邻项并列。
    pub tied: bool,
    /// 并列来自约数区间重叠，名次不是确定结论。
    pub uncertain: bool,
}

/// 一次排序的完整结果：只含同量纲可比项；其余显式放进 `skipped`。
#[derive(Debug, Clone, PartialEq)]
pub struct RankOutcome<T> {
    pub entries: Vec<Ranked<T>>,
    pub skipped: Vec<T>,
    pub dimension: Option<Dimension>,
    pub comparable: usize,
}

/// 稳定排序并分配名次。
///
/// 先选出现次数最多的量纲（并列时按枚举顺序），跨量纲项进入 `skipped`；
/// 同量纲内按基准值排序，约数区间重叠的相邻项共享名次并标记 `uncertain`。
/// 名次分配在 `offset/limit` 之前完成，因此「第二高」仍能拿到第 2 名。
pub fn rank<T: Clone>(items: Vec<(T, Quantity)>, spec: &RankSpec) -> RankOutcome<T> {
    let mut counts: std::collections::BTreeMap<Dimension, usize> = std::collections::BTreeMap::new();
    for (_, quantity) in &items {
        *counts.entry(quantity.dimension()).or_insert(0) += 1;
    }
    // 有显式单位时优先按单位量纲排名：缺单位数字（Count）通常来自抽取
    // 丢单位，不能让它们凭数量优势把真正可比的带单位值全部挤进 skipped。
    let preferred: Vec<(&Dimension, &usize)> = if counts.len() > 1 {
        let with_units: Vec<(&Dimension, &usize)> = counts
            .iter()
            .filter(|(dimension, _)| **dimension != Dimension::Count)
            .collect();
        if with_units.is_empty() {
            counts.iter().collect()
        } else {
            with_units
        }
    } else {
        counts.iter().collect()
    };
    let dimension = preferred
        .into_iter()
        .max_by(|(left_dimension, left), (right_dimension, right)| {
            left.cmp(right)
                .then_with(|| right_dimension.cmp(left_dimension))
        })
        .map(|(dimension, _)| *dimension);

    let mut comparable: Vec<(T, Quantity)> = Vec::new();
    let mut skipped: Vec<T> = Vec::new();
    for (item, quantity) in items {
        if Some(quantity.dimension()) == dimension {
            comparable.push((item, quantity));
        } else {
            skipped.push(item);
        }
    }
    comparable.sort_by(|(_, left), (_, right)| {
        let ordering = left
            .base_value()
            .partial_cmp(&right.base_value())
            .unwrap_or(std::cmp::Ordering::Equal);
        if spec.descending {
            ordering.reverse()
        } else {
            ordering
        }
    });

    let total = comparable.len();
    let mut ranked: Vec<Ranked<T>> = Vec::with_capacity(total);
    for (index, (item, quantity)) in comparable.into_iter().enumerate() {
        let (rank, tied, uncertain) = match ranked.last() {
            None => (1, false, false),
            Some(previous) => match previous.quantity.compare(&quantity) {
                Comparison::Equal => (previous.rank, true, previous.uncertain),
                Comparison::Uncertain => (previous.rank, true, true),
                _ => (index + 1, false, false),
            },
        };
        if tied {
            // 并列要同时标记前一项，否则「第一名」不知道自己共享名次。
            if let Some(previous) = ranked.last_mut() {
                previous.tied = true;
                previous.uncertain |= uncertain;
            }
        }
        ranked.push(Ranked {
            item,
            quantity,
            rank,
            tied,
            uncertain,
        });
    }

    let start = spec.offset.min(ranked.len());
    let end = start.saturating_add(spec.limit).min(ranked.len());
    RankOutcome {
        entries: ranked[start..end].to_vec(),
        skipped,
        dimension,
        comparable: total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_arabic_and_chinese_numbers_with_multipliers() {
        assert_eq!(parse_quantity("226").unwrap().base_value(), 226.0);
        assert_eq!(parse_quantity("67万").unwrap().base_value(), 670_000.0);
        assert_eq!(parse_quantity("三亿").unwrap().base_value(), 300_000_000.0);
        assert_eq!(parse_quantity("四十六亿").unwrap().base_value(), 4_600_000_000.0);
        assert_eq!(parse_quantity("1.5万").unwrap().base_value(), 15_000.0);
        assert!(parse_quantity("张三").is_none());
        assert!(parse_quantity("").is_none());
    }

    #[test]
    fn normalizes_units_inside_one_dimension() {
        let meters = parse_quantity("2.26米").unwrap();
        let centimeters = parse_quantity("226厘米").unwrap();
        assert_eq!(meters.dimension(), Dimension::Length);
        assert_eq!(centimeters.dimension(), Dimension::Length);
        assert!((meters.base_value() - centimeters.base_value()).abs() < 1e-9);
        assert_eq!(meters.compare(&centimeters), Comparison::Equal);
        assert_eq!(
            parse_quantity("1.5米")
                .unwrap()
                .compare(&parse_quantity("140厘米").unwrap()),
            Comparison::Greater
        );
        assert_eq!(
            parse_quantity("100斤").unwrap().base_value(),
            50.0,
            "斤应换算为千克"
        );
        assert_eq!(
            parse_quantity("5千米").unwrap().base_value(),
            5000.0,
            "千米的千是倍数而不是单位的一部分"
        );
        assert_eq!(parse_quantity("3千克").unwrap().base_value(), 3.0);
    }

    #[test]
    fn refuses_cross_dimension_comparison() {
        let area = parse_quantity("67万平方公里").unwrap();
        let height = parse_quantity("8848米").unwrap();
        assert_eq!(area.dimension(), Dimension::Area);
        assert_eq!(height.dimension(), Dimension::Length);
        assert!(matches!(
            area.compare(&height),
            Comparison::Incomparable(Incomparable::DimensionMismatch { .. })
        ));
        assert!(area.difference(&height).is_err());
    }

    #[test]
    fn approximate_values_expose_uncertainty() {
        let exact = parse_quantity("46亿年").unwrap();
        let approximate = parse_quantity("约46亿年").unwrap();
        assert_eq!(approximate.exactness(), Exactness::Approximate);
        assert_eq!(exact.compare(&approximate), Comparison::Uncertain);
        assert_eq!(
            parse_quantity("超过10米")
                .unwrap()
                .compare(&parse_quantity("11米").unwrap()),
            Comparison::Uncertain
        );
        assert_eq!(
            parse_quantity("超过10米")
                .unwrap()
                .compare(&parse_quantity("9米").unwrap()),
            Comparison::Greater
        );
        assert_eq!(
            parse_quantity("100多米").unwrap().exactness(),
            Exactness::Approximate
        );
        assert_eq!(
            parse_quantity("20岁左右").unwrap().exactness(),
            Exactness::Approximate
        );
    }

    #[test]
    fn difference_is_reported_in_the_original_unit() {
        let tall = parse_quantity("226厘米").unwrap();
        let short = parse_quantity("2.16米").unwrap();
        let (value, unit) = tall.difference_in_own_unit(&short).unwrap();
        assert!((value - 10.0).abs() < 1e-9);
        assert_eq!(unit, Some("厘米"));
        // 无单位计数不编造「个」。
        let (value, unit) = parse_quantity("20")
            .unwrap()
            .difference_in_own_unit(&parse_quantity("18").unwrap())
            .unwrap();
        assert_eq!(value, 2.0);
        assert_eq!(unit, None);
        assert!(tall
            .difference_in_own_unit(&parse_quantity("8848平方公里").unwrap())
            .is_err());
    }

    #[test]
    fn canonical_format_is_dimension_stable() {
        assert_eq!(
            parse_quantity("226厘米").unwrap().format_canonical(),
            "2.26米"
        );
        assert_eq!(parse_quantity("3.5吨").unwrap().format_canonical(), "3500千克");
        assert_eq!(parse_quantity("约20岁").unwrap().format_canonical(), "约20年");
        assert_eq!(parse_quantity("226").unwrap().format_canonical(), "226");
    }

    #[test]
    fn rank_keeps_ties_and_skips_incomparable_rows() {
        let items = vec![
            ("甲".to_string(), parse_quantity("226厘米").unwrap()),
            ("乙".to_string(), parse_quantity("2.26米").unwrap()),
            ("丙".to_string(), parse_quantity("1.5米").unwrap()),
            ("丁".to_string(), parse_quantity("67万平方公里").unwrap()),
        ];
        let outcome = rank(items, &RankSpec::default());
        assert_eq!(outcome.dimension, Some(Dimension::Length));
        assert_eq!(outcome.comparable, 3);
        assert_eq!(outcome.skipped, vec!["丁".to_string()]);
        assert_eq!(outcome.entries.len(), 1);
        assert_eq!(outcome.entries[0].rank, 1);
        assert_eq!(outcome.entries[0].item, "甲");
    }

    #[test]
    fn rank_competition_ranking_and_offset() {
        let items = vec![
            ("甲".to_string(), parse_quantity("226厘米").unwrap()),
            ("乙".to_string(), parse_quantity("226厘米").unwrap()),
            ("丙".to_string(), parse_quantity("1.5米").unwrap()),
        ];
        let outcome = rank(
            items,
            &RankSpec {
                descending: true,
                offset: 2,
                limit: 1,
            },
        );
        assert_eq!(outcome.entries[0].rank, 3, "并列第一后下一名是第三");
        assert_eq!(outcome.entries[0].item, "丙");
    }

    #[test]
    fn rank_prefers_unit_bearing_dimension_over_bare_counts() {
        let items = vec![
            ("甲".to_string(), parse_quantity("2.26米").unwrap()),
            ("乙".to_string(), parse_quantity("1.90米").unwrap()),
            ("丙".to_string(), parse_quantity("226").unwrap()),
            ("丁".to_string(), parse_quantity("190").unwrap()),
            ("戊".to_string(), parse_quantity("180").unwrap()),
        ];
        let outcome = rank(
            items,
            &RankSpec {
                descending: true,
                offset: 0,
                limit: 2,
            },
        );
        assert_eq!(outcome.dimension, Some(Dimension::Length));
        assert_eq!(outcome.entries.len(), 2);
        assert_eq!(outcome.entries[0].item, "甲");
        assert_eq!(outcome.entries[1].item, "乙");
        assert_eq!(outcome.skipped.len(), 3);
    }

    #[test]
    fn rank_marks_approximate_overlap_as_uncertain() {
        let items = vec![
            ("甲".to_string(), parse_quantity("46亿年").unwrap()),
            ("乙".to_string(), parse_quantity("约46亿年").unwrap()),
        ];
        assert_eq!(
            parse_quantity("约46亿年").unwrap().exactness(),
            Exactness::Approximate
        );
        assert_eq!(
            parse_quantity("46亿年")
                .unwrap()
                .compare(&parse_quantity("约46亿年").unwrap()),
            Comparison::Uncertain
        );
        let outcome = rank(items, &RankSpec::default());
        assert!(outcome.entries[0].tied);
        assert!(outcome.entries[0].uncertain);
    }
}

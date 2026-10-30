//! 属性本体桥（Attribute Ontology Bridge）。
//!
//! 文本里的属性事实长期是「二等公民」：抽取规则只认识固定词性模式，
//! 类型/量纲/单位检查散落在查询与形状校验层，属性值往往是未经归一的
//! 原始字符串。本模块把属性提升为带约束的一等谓词：
//!
//! - [`AttributeProfileConfig`] 声明规范名、词面别名、定义域、值类型、
//!   量纲、规范单位、单值性与父属性；
//! - [`AttributeRegistry`] 负责词面归一、父属性闭包、配置体检与候选校验；
//! - [`AttributeRegistry::assess`] 在落库前检查定义域/值类型/量纲，并把
//!   数量换算成声明单位；未知类型按开放世界放行但显式标注；
//! - [`AttributeRegistry::extract_mentions`] 从「主体 + 属性词面 + 值」
//!   句式中抽取结构化候选，供 `KnowledgeExtractor` 接入既有导入管线。
//!
//! 设计上不引入任何统计模型：每个接受/拒绝都能回溯到一条配置声明或
//! [`crate::quantity`] 中的单位表。未知属性不会静默落库，而是显式拒绝，
//! 交给对话层追问或后续登记。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::quantity::{lookup_unit, parse_quantity, Dimension, Quantity};
use crate::type_system::{TypeSystem, TypeVerdict};
use crate::TaggedWord;

/// config.toml 中的 `[[attribute_profiles]]` 声明。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeProfileConfig {
    /// 规范属性名（落库键）。
    pub name: String,
    /// 文本中可命中的词面别名。
    #[serde(default)]
    pub aliases: Vec<String>,
    /// 定义域：主体允许的类型（空 = 不限制）。
    #[serde(default)]
    pub domain: Vec<String>,
    /// 值类型，默认 `string`；也接受别名 `data_type`。
    #[serde(default, alias = "data_type")]
    pub value_type: AttributeValueType,
    /// 数量量纲（`length`/`mass`/`duration`… 或中文名）。
    #[serde(default)]
    pub dimension: Option<String>,
    /// 规范/展示单位；数量先换算到此单位再落库。
    #[serde(default, alias = "canonical_unit", alias = "display_unit")]
    pub unit: Option<String>,
    /// 单值属性（同一主体只应有一个取值）。
    #[serde(default, alias = "single_valued")]
    pub functional: bool,
    /// 父属性：本属性 ⊑ 父属性，查询父属性时召回本属性事实。
    #[serde(default, alias = "subproperty_of")]
    pub parent: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// config.toml 中的 `[attributes]` 段。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeBridgeConfig {
    /// 是否在文本导入时启用属性候选抽取（默认开启；无档案时自然不生效）。
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub profiles: Vec<AttributeProfileConfig>,
}

fn default_enabled() -> bool {
    true
}

impl Default for AttributeBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            profiles: Vec::new(),
        }
    }
}

/// 属性值的类型约束。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeValueType {
    #[default]
    #[serde(alias = "text")]
    String,
    #[serde(alias = "int")]
    Integer,
    #[serde(alias = "float", alias = "double")]
    Number,
    Quantity,
    #[serde(alias = "bool")]
    Boolean,
    Date,
    Datetime,
    Any,
}

impl AttributeValueType {
    pub fn label(self) -> &'static str {
        match self {
            AttributeValueType::String => "字符串",
            AttributeValueType::Integer => "整数",
            AttributeValueType::Number => "数值",
            AttributeValueType::Quantity => "数量",
            AttributeValueType::Boolean => "布尔",
            AttributeValueType::Date => "日期",
            AttributeValueType::Datetime => "日期时间",
            AttributeValueType::Any => "任意",
        }
    }
}

/// 声明解析过程中发现的问题；不阻止其余画像继续工作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeIssue {
    pub attribute: String,
    pub detail: String,
}

/// 运行时属性画像。
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeProfile {
    pub canonical: String,
    pub aliases: BTreeSet<String>,
    pub domain: BTreeSet<String>,
    pub value_type: AttributeValueType,
    pub dimension: Option<Dimension>,
    pub unit: Option<String>,
    pub functional: bool,
    pub parent: Option<String>,
    pub description: Option<String>,
}

impl AttributeProfile {
    /// 词面（别名 ∪ 规范名的书写形式），用于抽取与诊断。
    pub fn surfaces(&self) -> Vec<String> {
        let mut surfaces: BTreeSet<String> = self.aliases.clone();
        surfaces.insert(self.canonical.clone());
        surfaces.into_iter().collect()
    }

    /// 稳定属性 IRI：`{base}attribute/{编码后的规范名}`。
    pub fn iri(&self) -> String {
        crate::identity::attribute_uri(crate::identity::DEFAULT_BASE_NS, &self.canonical)
    }

    /// 旧式平面谓词 IRI：`{base}{规范名}`（只读兼容，不切换默认写路径）。
    pub fn legacy_iri(&self) -> String {
        format!(
            "{}{}",
            crate::identity::DEFAULT_BASE_NS,
            self.canonical
        )
    }
}

/// 属性注册表：规范名 + 别名索引 + 父子闭包 + 声明体检结果。
#[derive(Debug, Clone, Default)]
pub struct AttributeRegistry {
    profiles: BTreeMap<String, AttributeProfile>,
    aliases: BTreeMap<String, String>,
    issues: Vec<AttributeIssue>,
}

impl AttributeRegistry {
    /// 从配置编译注册表。别名冲突、坏单位、量纲不一致都记入
    /// [`Self::issues`]，不静默丢弃整张表。
    pub fn from_config(config: &AttributeBridgeConfig) -> Self {
        let mut registry = Self::default();
        let canonicals: BTreeSet<String> = config
            .profiles
            .iter()
            .map(|profile| normalize_key(&profile.name))
            .filter(|name| !name.is_empty())
            .collect();

        for config_profile in &config.profiles {
            let key = normalize_key(&config_profile.name);
            if key.is_empty() {
                registry.issues.push(AttributeIssue {
                    attribute: config_profile.name.clone(),
                    detail: "属性规范名为空，已跳过".to_string(),
                });
                continue;
            }
            if registry.profiles.contains_key(&key) {
                registry.issues.push(AttributeIssue {
                    attribute: config_profile.name.clone(),
                    detail: "规范名重复，后声明被忽略".to_string(),
                });
                continue;
            }

            let mut dimension = config_profile
                .dimension
                .as_deref()
                .and_then(parse_dimension);
            if config_profile.dimension.is_some() && dimension.is_none() {
                registry.issues.push(AttributeIssue {
                    attribute: config_profile.name.clone(),
                    detail: format!(
                        "未知量纲「{}」；已按未声明处理",
                        config_profile.dimension.as_deref().unwrap_or("")
                    ),
                });
            }

            let mut unit = config_profile
                .unit
                .as_deref()
                .map(str::trim)
                .filter(|unit| !unit.is_empty())
                .map(ToString::to_string);
            if let Some(unit_alias) = unit.as_deref() {
                match lookup_unit(unit_alias) {
                    Some(unit_def) => {
                        if let Some(declared) = dimension {
                            if declared != unit_def.dimension {
                                registry.issues.push(AttributeIssue {
                                    attribute: config_profile.name.clone(),
                                    detail: format!(
                                        "单位「{}」属于{}量纲，与声明的{}量纲不一致；已忽略单位",
                                        unit_alias,
                                        unit_def.dimension.label(),
                                        declared.label()
                                    ),
                                });
                                unit = None;
                            }
                        } else {
                            dimension = Some(unit_def.dimension);
                        }
                    }
                    None => {
                        registry.issues.push(AttributeIssue {
                            attribute: config_profile.name.clone(),
                            detail: format!("未知单位「{}」；已忽略单位", unit_alias),
                        });
                        unit = None;
                    }
                }
            }

            let mut aliases = BTreeSet::new();
            for alias in &config_profile.aliases {
                let alias_key = normalize_key(alias);
                if alias_key.is_empty() {
                    continue;
                }
                if canonicals.contains(&alias_key) && alias_key != key {
                    registry.issues.push(AttributeIssue {
                        attribute: config_profile.name.clone(),
                        detail: format!("别名「{}」与其它属性规范名冲突，已忽略", alias.trim()),
                    });
                    continue;
                }
                aliases.insert(alias.trim().to_string());
            }

            registry.profiles.insert(
                key,
                AttributeProfile {
                    canonical: config_profile.name.trim().to_string(),
                    aliases,
                    domain: config_profile
                        .domain
                        .iter()
                        .map(|name| name.trim().to_string())
                        .filter(|name| !name.is_empty())
                        .collect(),
                    value_type: config_profile.value_type,
                    dimension,
                    unit,
                    functional: config_profile.functional,
                    parent: config_profile
                        .parent
                        .as_deref()
                        .map(str::trim)
                        .filter(|parent| !parent.is_empty())
                        .map(ToString::to_string),
                    description: config_profile.description.clone(),
                },
            );
        }

        // 第二遍登记别名，别名冲突按字典序后的首次声明优先，保证确定性。
        let keys: Vec<String> = registry.profiles.keys().cloned().collect();
        for key in keys {
            let profile = registry.profiles.get(&key).cloned().expect("画像刚插入");
            for alias in &profile.aliases {
                let alias_key = normalize_key(alias);
                if alias_key == key {
                    continue;
                }
                if let Some(existing) = registry.aliases.get(&alias_key) {
                    if existing != &key {
                        registry.issues.push(AttributeIssue {
                            attribute: profile.canonical.clone(),
                            detail: format!(
                                "别名「{}」已被「{}」占用，未覆盖既有索引",
                                alias.trim(),
                                existing
                            ),
                        });
                    }
                    continue;
                }
                registry.aliases.insert(alias_key, key.clone());
            }
        }

        // 父属性环与悬空引用体检。
        for key in registry.profiles.keys() {
            let mut seen = BTreeSet::new();
            let mut current = key.clone();
            loop {
                if !seen.insert(current.clone()) {
                    registry.issues.push(AttributeIssue {
                        attribute: registry
                            .profiles
                            .get(key)
                            .map(|profile| profile.canonical.clone())
                            .unwrap_or_else(|| key.clone()),
                        detail: "父属性链成环".to_string(),
                    });
                    break;
                }
                let Some(profile) = registry.profiles.get(&current) else {
                    break;
                };
                let Some(parent) = profile.parent.as_deref() else {
                    break;
                };
                match registry.resolve_key(parent) {
                    Some(parent_key) => current = parent_key,
                    None => {
                        registry.issues.push(AttributeIssue {
                            attribute: profile.canonical.clone(),
                            detail: format!("父属性「{}」未登记", parent),
                        });
                        break;
                    }
                }
            }
        }

        registry
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn issues(&self) -> &[AttributeIssue] {
        &self.issues
    }

    pub fn canonical_names(&self) -> Vec<&str> {
        self.profiles
            .values()
            .map(|profile| profile.canonical.as_str())
            .collect()
    }

    fn resolve_key(&self, surface: &str) -> Option<String> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        // 读兼容：规范属性 IRI 与旧式平面谓词 IRI 都归一到本地名。
        let local = crate::identity::attribute_local_name(
            crate::identity::DEFAULT_BASE_NS,
            surface,
        )
        .or_else(|| legacy_predicate_local_name(surface))
        .unwrap_or_else(|| surface.to_string());
        let key = normalize_key(&local);
        if key.is_empty() {
            return None;
        }
        if self.profiles.contains_key(&key) {
            return Some(key);
        }
        self.aliases.get(&key).cloned()
    }

    /// 词面 → 规范画像。
    pub fn resolve(&self, surface: &str) -> Option<&AttributeProfile> {
        self.resolve_key(surface)
            .and_then(|key| self.profiles.get(&key))
    }

    /// 属性 IRI → 规范画像：新式 `kg:attribute/{名}` 与旧式 `kg:{名}` 都接受。
    ///
    /// 实体路径与外部命名空间一律返回 `None`，避免把实体名误读成属性。
    pub fn resolve_iri(&self, iri: &str) -> Option<&AttributeProfile> {
        let local = crate::identity::attribute_local_name(
            crate::identity::DEFAULT_BASE_NS,
            iri,
        )
        .or_else(|| legacy_predicate_local_name(iri))?;
        self.resolve(&local)
    }

    /// 词面/规范名 → 稳定属性 IRI（未登记返回 `None`）。
    pub fn iri_for(&self, surface: &str) -> Option<String> {
        self.resolve(surface).map(AttributeProfile::iri)
    }

    pub fn profile(&self, canonical: &str) -> Option<&AttributeProfile> {
        self.resolve(canonical)
    }

    /// 父属性链（不含自身，去重，确定性排序）。
    pub fn ancestors(&self, surface: &str) -> Vec<String> {
        let Some(start) = self.resolve_key(surface) else {
            return Vec::new();
        };
        let mut result = BTreeSet::new();
        let start_key = start.clone();
        let mut current = start;
        let mut guard = 0;
        while guard <= self.profiles.len() {
            guard += 1;
            let Some(profile) = self.profiles.get(&current) else {
                break;
            };
            let Some(parent) = profile.parent.as_deref() else {
                break;
            };
            let Some(parent_key) = self.resolve_key(parent) else {
                break;
            };
            if !result.insert(parent_key.clone()) || parent_key == start_key {
                break;
            }
            if parent_key == current {
                break;
            }
            current = parent_key;
        }
        result
            .into_iter()
            .filter_map(|key| self.profiles.get(&key))
            .map(|profile| profile.canonical.clone())
            .collect()
    }

    /// 某属性的全部子属性（含自身），用于父属性查询召回。
    pub fn subattributes_of(&self, surface: &str) -> Vec<String> {
        let Some(target) = self.resolve_key(surface) else {
            return Vec::new();
        };
        let target_canonical = self
            .profiles
            .get(&target)
            .map(|profile| profile.canonical.clone())
            .unwrap_or_else(|| target.clone());
        let mut result = BTreeSet::new();
        for profile in self.profiles.values() {
            let key = normalize_key(&profile.canonical);
            if key == target
                || self
                    .ancestors(&profile.canonical)
                    .iter()
                    .any(|ancestor| normalize_key(ancestor) == target)
            {
                result.insert(profile.canonical.clone());
            }
        }
        debug_assert!(result.contains(&target_canonical));
        result.into_iter().collect()
    }

    /// 全部可命中的词面（按长度降序，供最长匹配抽取）。
    pub fn extraction_surfaces(&self) -> Vec<(String, String)> {
        let mut surfaces: Vec<(String, String)> = Vec::new();
        for profile in self.profiles.values() {
            for surface in profile.surfaces() {
                surfaces.push((surface, profile.canonical.clone()));
            }
        }
        surfaces.sort_by(|left, right| {
            right
                .0
                .chars()
                .count()
                .cmp(&left.0.chars().count())
                .then_with(|| left.0.cmp(&right.0))
        });
        surfaces
    }

    /// 校验一条属性断言并给出规范化后的取值。
    ///
    /// 开放世界：主体类型未知或与声明定义域无法证明相交时接受，但返回
    /// [`AttributeVerdict::AcceptedTypeUnconfirmed`]；只有可证明互斥才拒绝。
    pub fn assess(
        &self,
        types: &TypeSystem,
        subject_types: &BTreeSet<String>,
        attribute: &str,
        raw_value: &str,
    ) -> AttributeAssessment {
        let original_value = raw_value.trim().to_string();
        let Some(profile) = self.resolve(attribute) else {
            return AttributeAssessment {
                verdict: AttributeVerdict::UnknownAttribute,
                attribute: attribute.trim().to_string(),
                original_value,
                normalized_value: None,
                dimension: None,
                notes: vec!["属性未在 [[attribute_profiles]] 中登记".to_string()],
            };
        };

        let mut notes = Vec::new();
        let mut type_unconfirmed = false;
        match types.sets_admit(subject_types, &profile.domain) {
            TypeVerdict::Disjoint => {
                return AttributeAssessment {
                    verdict: AttributeVerdict::DomainMismatch,
                    attribute: profile.canonical.clone(),
                    original_value,
                    normalized_value: None,
                    dimension: profile.dimension,
                    notes: vec![format!(
                        "主体类型 {} 与「{}」的定义域 {} 可证明互斥",
                        render_types(subject_types),
                        profile.canonical,
                        render_types(&profile.domain)
                    )],
                };
            }
            TypeVerdict::Unknown => {
                if !profile.domain.is_empty() && subject_types.is_empty() {
                    type_unconfirmed = true;
                    notes.push(format!(
                        "主体类型未知；开放世界下接受，定义域为 {}",
                        render_types(&profile.domain)
                    ));
                } else if !profile.domain.is_empty() {
                    type_unconfirmed = true;
                    notes.push("主体类型与定义域无法证明相交，按开放世界接受".to_string());
                }
            }
            TypeVerdict::Compatible => {}
        }

        let parsed = match parse_attribute_value(profile, &original_value) {
            Ok(parsed) => parsed,
            Err(reason) => {
                return AttributeAssessment {
                    verdict: AttributeVerdict::ValueMismatch,
                    attribute: profile.canonical.clone(),
                    original_value,
                    normalized_value: None,
                    dimension: profile.dimension,
                    notes: vec![reason],
                };
            }
        };

        AttributeAssessment {
            verdict: if type_unconfirmed {
                AttributeVerdict::AcceptedTypeUnconfirmed
            } else {
                AttributeVerdict::Accepted
            },
            attribute: profile.canonical.clone(),
            original_value,
            normalized_value: Some(parsed.normalized),
            dimension: parsed.dimension,
            notes,
        }
    }

    /// 从「主体 + 属性词面 + 值」句式中抽取全部属性候选。
    ///
    /// 只负责结构切分，不判断值是否合法；调用方用 [`Self::assess`] 决定
    /// 接受、标记或拒绝。问句式后缀（多少/什么/吗…）不会被当作断言。
    ///
    /// 同一句里的后续属性可以省略主体（「姚明身高226，体重120」）：
    /// 优先使用属性词面左侧最近的姓名 token，找不到时沿用前一条候选的
    /// 主体。值域以下一个属性词面为右边界，避免把「226体重120」粘成一个值。
    pub fn extract_mentions(&self, text: &str, tagged_words: &[TaggedWord]) -> Vec<AttributeMention> {
        if text.trim().is_empty() || self.profiles.is_empty() {
            return Vec::new();
        }
        let surfaces = self.extraction_surfaces();
        let mut mentions = Vec::new();
        let mut cursor = 0usize;
        let mut last_subject: Option<String> = None;
        while cursor < text.len() {
            let Some((surface, attribute, start, end)) =
                find_surface_at_or_after(text, cursor, &surfaces)
            else {
                break;
            };
            let next_start = find_surface_at_or_after(text, end, &surfaces)
                .map(|(_, _, next, _)| next)
                .unwrap_or(text.len());
            let prefix = &text[..start];
            let suffix = &text[end..next_start];
            if let Some(raw_value) = clean_value(suffix) {
                if let Some(subject) = extract_subject(
                    prefix,
                    text,
                    start,
                    tagged_words,
                    last_subject.as_deref(),
                    &surfaces,
                ) {
                    last_subject = Some(subject.clone());
                    mentions.push(AttributeMention {
                        subject,
                        attribute,
                        surface,
                        raw_value,
                        start,
                        end,
                    });
                }
            }
            cursor = end;
        }
        mentions
    }

    fn find_first_surface(&self, text: &str) -> Option<(String, String, usize, usize)> {
        find_surface_at_or_after(text, 0, &self.extraction_surfaces())
    }
}

/// 在 `from` 之后寻找最早的属性词面；同一位置按已排序词面取最长者。
fn find_surface_at_or_after(
    text: &str,
    from: usize,
    surfaces: &[(String, String)],
) -> Option<(String, String, usize, usize)> {
    if from >= text.len() || !text.is_char_boundary(from) {
        return None;
    }
    for (offset, _) in text[from..].char_indices() {
        let index = from + offset;
        for (surface, canonical) in surfaces {
            let length = surface.len();
            if index + length > text.len() || !text.is_char_boundary(index + length) {
                continue;
            }
            let candidate = &text[index..index + length];
            if candidate == surface || candidate.to_lowercase() == surface.to_lowercase() {
                return Some((surface.clone(), canonical.clone(), index, index + length));
            }
        }
    }
    None
}

/// 单位化属性问句的解析结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeRequest {
    pub subject: String,
    pub attribute: String,
    pub surface: String,
    /// 用户显式要求的单位（如「多少厘米」中的「厘米」）。
    pub unit: Option<String>,
}

impl AttributeRegistry {
    /// 解析「主体 + 已登记属性词面 + 单位问句」。
    ///
    /// 只有问句里出现显式单位时才返回 `Some(unit)`；无单位问句交给既有
    /// 属性回答路径，避免本模块抢占普通「X 的身高是多少」。
    pub fn parse_request(&self, question: &str) -> Option<AttributeRequest> {
        if !crate::function_words::contains_question_marker(question) {
            return None;
        }
        let (surface, attribute, start, end) = self.find_first_surface(question)?;
        let profile = self.resolve(&attribute)?;
        let subject = clean_subject(&question[..start]);
        if subject.is_empty() {
            return None;
        }
        let suffix = &question[end..];
        let unit = find_requested_unit(suffix, profile)?;
        Some(AttributeRequest {
            subject,
            attribute: profile.canonical.clone(),
            surface,
            unit: Some(unit),
        })
    }
}

fn find_requested_unit(suffix: &str, profile: &AttributeProfile) -> Option<String> {
    let mut best: Option<(&'static str, usize)> = None;
    for (alias, unit) in crate::quantity::UNITS {
        if profile
            .dimension
            .is_some_and(|dimension| dimension != unit.dimension)
        {
            continue;
        }
        if !suffix.contains(alias) {
            continue;
        }
        let length = alias.chars().count();
        if best.is_none_or(|(_, best_length)| length > best_length) {
            best = Some((alias, length));
        }
    }
    best.map(|(alias, _)| alias.to_string())
}

fn normalize_key(text: &str) -> String {
    text.trim().to_lowercase()
}

/// 旧式平面谓词 IRI → 本地名：仅接受 `{base}` 下的非实体、非属性路径。
fn legacy_predicate_local_name(iri: &str) -> Option<String> {
    let rest = iri.strip_prefix(crate::identity::DEFAULT_BASE_NS)?;
    if rest.starts_with(crate::identity::ENTITY_PATH)
        || rest.starts_with(crate::identity::ATTRIBUTE_PATH)
    {
        return None;
    }
    crate::identity::decode_segment(rest)
}

pub(crate) fn parse_dimension(text: &str) -> Option<Dimension> {
    match normalize_key(text).as_str() {
        "length" | "长度" | "距离" => Some(Dimension::Length),
        "mass" | "weight" | "质量" | "重量" => Some(Dimension::Mass),
        "duration" | "time" | "时间" | "时长" | "年龄" => Some(Dimension::Duration),
        "area" | "面积" => Some(Dimension::Area),
        "volume" | "体积" => Some(Dimension::Volume),
        "temperature" | "温度" => Some(Dimension::Temperature),
        "count" | "计数" | "数量" => Some(Dimension::Count),
        "ratio" | "比例" | "百分比" => Some(Dimension::Ratio),
        "currency" | "money" | "货币" | "金额" => Some(Dimension::Currency),
        _ => None,
    }
}

fn render_types(types: &BTreeSet<String>) -> String {
    if types.is_empty() {
        "（未知）".to_string()
    } else {
        types.iter().cloned().collect::<Vec<_>>().join("、")
    }
}

/// 校验结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeVerdict {
    /// 校验通过。
    Accepted,
    /// 值合法，但主体类型无法确认（开放世界放行）。
    AcceptedTypeUnconfirmed,
    /// 属性未登记。
    UnknownAttribute,
    /// 主体类型与定义域可证明互斥。
    DomainMismatch,
    /// 取值类型/量纲不符合声明。
    ValueMismatch,
}

impl AttributeVerdict {
    pub fn code(self) -> &'static str {
        match self {
            AttributeVerdict::Accepted => "accepted",
            AttributeVerdict::AcceptedTypeUnconfirmed => "accepted_type_unconfirmed",
            AttributeVerdict::UnknownAttribute => "unknown_attribute",
            AttributeVerdict::DomainMismatch => "domain_mismatch",
            AttributeVerdict::ValueMismatch => "value_mismatch",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AttributeVerdict::Accepted => "通过",
            AttributeVerdict::AcceptedTypeUnconfirmed => "通过（类型未确认）",
            AttributeVerdict::UnknownAttribute => "属性未登记",
            AttributeVerdict::DomainMismatch => "定义域不符",
            AttributeVerdict::ValueMismatch => "取值不符",
        }
    }

    pub fn is_accepted(self) -> bool {
        matches!(
            self,
            AttributeVerdict::Accepted | AttributeVerdict::AcceptedTypeUnconfirmed
        )
    }
}

/// 单条属性断言的校验结果。
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeAssessment {
    pub verdict: AttributeVerdict,
    pub attribute: String,
    pub original_value: String,
    pub normalized_value: Option<String>,
    pub dimension: Option<Dimension>,
    pub notes: Vec<String>,
}

impl AttributeAssessment {
    pub fn is_accepted(&self) -> bool {
        self.verdict.is_accepted()
    }

    /// 可落库取值：只有通过校验时存在。
    pub fn value(&self) -> Option<&str> {
        self.normalized_value.as_deref()
    }
}

/// 结构抽取结果（尚未做类型/量纲校验）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeMention {
    pub subject: String,
    pub attribute: String,
    pub surface: String,
    pub raw_value: String,
    /// 词面在原文中的字节起止位置，供证据回溯。
    pub start: usize,
    pub end: usize,
}

struct ParsedValue {
    normalized: String,
    dimension: Option<Dimension>,
}

fn parse_attribute_value(
    profile: &AttributeProfile,
    raw: &str,
) -> Result<ParsedValue, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("属性取值为空".to_string());
    }
    match profile.value_type {
        AttributeValueType::String | AttributeValueType::Any => {
            if looks_like_question(trimmed) {
                return Err(format!("「{}」是问句片段，不是属性取值", trimmed));
            }
            Ok(ParsedValue {
                normalized: trimmed.to_string(),
                dimension: profile.dimension,
            })
        }
        AttributeValueType::Integer => trimmed
            .parse::<i64>()
            .map(|value| ParsedValue {
                normalized: value.to_string(),
                dimension: profile.dimension,
            })
            .map_err(|_| format!("「{}」不是整数", trimmed)),
        AttributeValueType::Number => trimmed
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .map(|value| ParsedValue {
                normalized: crate::quantity::format_number(value),
                dimension: profile.dimension,
            })
            .ok_or_else(|| format!("「{}」不是数值", trimmed)),
        AttributeValueType::Boolean => {
            let lowered = trimmed.to_lowercase();
            let normalized = match lowered.as_str() {
                "true" | "1" | "yes" | "是" => "true",
                "false" | "0" | "no" | "否" => "false",
                _ => return Err(format!("「{}」不是布尔值", trimmed)),
            };
            Ok(ParsedValue {
                normalized: normalized.to_string(),
                dimension: profile.dimension,
            })
        }
        AttributeValueType::Date => crate::temporal::parse_time_point(trimmed)
            .map(|point| ParsedValue {
                normalized: point.render(),
                dimension: profile.dimension,
            })
            .ok_or_else(|| format!("「{}」不是可解析的日期", trimmed)),
        AttributeValueType::Datetime => crate::shape_gate::datatype_conforms("datetime", trimmed)
            .then(|| ParsedValue {
                normalized: trimmed.to_string(),
                dimension: profile.dimension,
            })
            .ok_or_else(|| format!("「{}」不是可解析的日期时间", trimmed)),
        AttributeValueType::Quantity => parse_quantity_value(profile, trimmed),
    }
}

fn parse_quantity_value(
    profile: &AttributeProfile,
    raw: &str,
) -> Result<ParsedValue, String> {
    let normalized_raw = raw
        .strip_prefix("约为")
        .or_else(|| raw.strip_prefix("约是"))
        .map(|rest| format!("约{}", rest.trim_start()))
        .unwrap_or_else(|| raw.to_string());
    let raw = normalized_raw.as_str();
    let parsed = parse_quantity(raw).or_else(|| {
        let unit = profile.unit.as_deref()?;
        if raw.chars().any(|ch| ch.is_ascii_digit()) {
            parse_quantity(&format!("{}{}", raw, unit))
        } else {
            None
        }
    });
    let Some(quantity) = parsed else {
        return Err(format!("「{}」不是可解析的数量", raw));
    };

    let quantity = interpret_unitless(profile, quantity);
    if let Some(expected) = profile.dimension {
        if quantity.dimension() != expected {
            return Err(format!(
                "量纲不符：期望{}，实际{}（{}）",
                expected.label(),
                quantity.dimension().label(),
                raw
            ));
        }
    }

    let normalized = profile
        .unit
        .as_deref()
        .and_then(|unit| quantity.format_in_unit(unit))
        .unwrap_or_else(|| quantity.format_canonical());
    Ok(ParsedValue {
        normalized,
        dimension: Some(quantity.dimension()),
    })
}

fn interpret_unitless(profile: &AttributeProfile, quantity: Quantity) -> Quantity {
    if quantity.has_unit() {
        return quantity;
    }
    let Some(expected) = profile.dimension else {
        return quantity;
    };
    if quantity.dimension() != Dimension::Count || expected == Dimension::Count {
        return quantity;
    }
    let Some(unit) = profile.unit.as_deref() else {
        return quantity;
    };
    parse_quantity(&format!("{}{}", quantity.raw(), unit)).unwrap_or(quantity)
}

fn extract_subject(
    prefix: &str,
    text: &str,
    alias_start: usize,
    tagged_words: &[TaggedWord],
    last_subject: Option<&str>,
    surfaces: &[(String, String)],
) -> Option<String> {
    if let Some(name) = last_name_token(text, alias_start, tagged_words) {
        return Some(name);
    }
    let cleaned = clean_subject(prefix);
    if cleaned.is_empty() {
        return last_subject.map(str::to_string);
    }
    // 省略主语的后续属性：前缀里已经出现过属性词面（「姚明身高226」），
    // 说明 cleaned 把上一条属性也吞进来了，沿用上一条主体而不是造出新名字。
    if last_subject.is_some()
        && surfaces
            .iter()
            .any(|(surface, _)| prefix.contains(surface.as_str()))
    {
        return last_subject.map(str::to_string);
    }
    Some(cleaned)
}

fn last_name_token(
    text: &str,
    alias_start: usize,
    tagged_words: &[TaggedWord],
) -> Option<String> {
    let mut cursor = 0usize;
    let mut last = None;
    for word in tagged_words {
        let needle = word.text.trim();
        if needle.is_empty() {
            continue;
        }
        let Some(relative) = text[cursor..alias_start].find(needle) else {
            continue;
        };
        let start = cursor + relative;
        let end = start + needle.len();
        if end > alias_start {
            continue;
        }
        cursor = end;
        if matches!(word.pos.as_str(), "nr" | "nz" | "nrt") {
            last = Some(needle.to_string());
        }
    }
    last
}

fn clean_subject(prefix: &str) -> String {
    let mut subject = prefix.trim().trim_end_matches(['。', '，', ',', '；', ';', '！', '!', '？', '?']);
    for directive in [
        "录入：", "录入:", "添加：", "添加:", "新增：", "新增:", "记录：", "记录:", "导入：",
        "导入:", "补充：", "补充:", "记住：", "记住:", "告诉", "我说",
    ] {
        if let Some(rest) = subject.strip_prefix(directive) {
            subject = rest.trim();
        }
    }
    if let Some((_, tail)) = subject.rsplit_once(['，', ',', '；', ';', '：', ':']) {
        subject = tail.trim();
    }
    loop {
        let trimmed = subject.trim();
        let next = trimmed
            .strip_suffix('的')
            .or_else(|| trimmed.strip_suffix('是'))
            .or_else(|| trimmed.strip_suffix('为'))
            .or_else(|| trimmed.strip_suffix('有'))
            .or_else(|| trimmed.strip_suffix('在'))
            .or_else(|| trimmed.strip_suffix('也'))
            .or_else(|| trimmed.strip_suffix('则'))
            .unwrap_or(trimmed);
        if next == trimmed {
            break;
        }
        subject = next;
    }
    subject.trim().to_string()
}

fn clean_value(suffix: &str) -> Option<String> {
    let mut value = suffix
        .trim()
        .trim_matches(['：', ':', '，', ',', '。', '；', ';', '！', '!', '？', '?', ' ', '\t']);
    loop {
        let next = value
            .strip_prefix("是")
            .or_else(|| value.strip_prefix("为"))
            .or_else(|| value.strip_prefix("有"))
            .or_else(|| value.strip_prefix("达到"))
            .or_else(|| value.strip_prefix("高达"))
            .or_else(|| value.strip_prefix("等于"))
            .or_else(|| value.strip_prefix("约为"))
            .unwrap_or(value)
            .trim_start();
        if next == value {
            break;
        }
        value = next;
    }
    value = value
        .split(['，', ',', '。', '；', ';', '！', '!', '？', '?'])
        .next()
        .unwrap_or(value)
        .trim();
    if value.is_empty() || looks_like_question(value) {
        return None;
    }
    Some(value.to_string())
}

fn looks_like_question(value: &str) -> bool {
    [
        "多少", "几", "什么", "啥", "吗", "呢", "谁", "哪", "如何", "怎么", "多大", "多高", "多重",
    ]
    .iter()
    .any(|marker| value.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_system::{TypeDisjointConfig, TypeSubclassConfig, TypeSystemConfig};

    fn sample_config() -> AttributeBridgeConfig {
        AttributeBridgeConfig {
            enabled: true,
            profiles: vec![
                AttributeProfileConfig {
                    name: "height".to_string(),
                    aliases: vec!["身高".to_string(), "身长".to_string()],
                    domain: vec!["Person".to_string()],
                    value_type: AttributeValueType::Quantity,
                    dimension: Some("length".to_string()),
                    unit: Some("厘米".to_string()),
                    functional: true,
                    ..AttributeProfileConfig::default()
                },
                AttributeProfileConfig {
                    name: "net_height".to_string(),
                    aliases: vec!["净身高".to_string()],
                    domain: vec!["Person".to_string()],
                    value_type: AttributeValueType::Quantity,
                    dimension: Some("length".to_string()),
                    unit: Some("厘米".to_string()),
                    functional: true,
                    parent: Some("height".to_string()),
                    ..AttributeProfileConfig::default()
                },
                AttributeProfileConfig {
                    name: "weight".to_string(),
                    aliases: vec!["体重".to_string()],
                    domain: vec!["Person".to_string()],
                    value_type: AttributeValueType::Quantity,
                    dimension: Some("mass".to_string()),
                    unit: Some("千克".to_string()),
                    functional: true,
                    ..AttributeProfileConfig::default()
                },
                AttributeProfileConfig {
                    name: "age".to_string(),
                    aliases: vec!["年龄".to_string()],
                    domain: vec!["Person".to_string()],
                    value_type: AttributeValueType::Quantity,
                    dimension: Some("duration".to_string()),
                    unit: Some("岁".to_string()),
                    functional: true,
                    ..AttributeProfileConfig::default()
                },
                AttributeProfileConfig {
                    name: "birthplace".to_string(),
                    aliases: vec!["出生地".to_string(), "籍贯".to_string()],
                    domain: vec!["Person".to_string()],
                    value_type: AttributeValueType::String,
                    ..AttributeProfileConfig::default()
                },
            ],
        }
    }

    fn person_types() -> TypeSystem {
        TypeSystem::from_config(&TypeSystemConfig {
            subclasses: vec![TypeSubclassConfig {
                sub: "Athlete".to_string(),
                sup: "Person".to_string(),
            }],
            disjoint: vec![TypeDisjointConfig {
                left: "Person".to_string(),
                right: "Company".to_string(),
            }],
            ..TypeSystemConfig::default()
        })
    }

    fn type_set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn registry_resolves_aliases_and_parent_closure() {
        let registry = AttributeRegistry::from_config(&sample_config());
        assert!(registry.issues().is_empty(), "{:?}", registry.issues());
        assert_eq!(registry.resolve("身高").unwrap().canonical, "height");
        assert_eq!(registry.resolve("身长").unwrap().canonical, "height");
        assert_eq!(registry.ancestors("净身高"), vec!["height".to_string()]);
        assert_eq!(
            registry.subattributes_of("身高"),
            vec!["height".to_string(), "net_height".to_string()]
        );
    }

    #[test]
    fn attribute_iris_are_stable_and_read_both_new_and_legacy_forms() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let height = registry.profile("height").expect("height 已登记");
        assert_eq!(
            height.iri(),
            "http://example.org/kg/attribute/height"
        );
        assert_eq!(height.legacy_iri(), "http://example.org/kg/height");
        assert_eq!(
            registry.iri_for("身高").as_deref(),
            Some("http://example.org/kg/attribute/height")
        );
        assert_eq!(
            registry
                .resolve_iri("http://example.org/kg/attribute/height")
                .map(|profile| profile.canonical.as_str()),
            Some("height")
        );
        assert_eq!(
            registry
                .resolve_iri("http://example.org/kg/height")
                .map(|profile| profile.canonical.as_str()),
            Some("height"),
            "旧式平面谓词必须仍可读"
        );
        assert_eq!(
            registry
                .resolve_iri("http://example.org/kg/身高")
                .map(|profile| profile.canonical.as_str()),
            Some("height"),
            "旧式别名谓词也应可读"
        );
        assert!(
            registry
                .resolve_iri("http://example.org/kg/entity/height")
                .is_none(),
            "实体路径不得被误读成属性"
        );
        assert!(registry
            .resolve_iri("http://other.org/kg/height")
            .is_none());
        assert!(registry.iri_for("未登记").is_none());
    }

    #[test]
    fn assess_accepts_new_and_legacy_attribute_iris() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let types = person_types();
        for attribute in [
            "http://example.org/kg/attribute/height",
            "http://example.org/kg/height",
            "http://example.org/kg/身高",
        ] {
            let assessment =
                registry.assess(&types, &type_set(&["Person"]), attribute, "2.26米");
            assert_eq!(
                assessment.verdict,
                AttributeVerdict::Accepted,
                "{attribute}"
            );
            assert_eq!(assessment.attribute, "height");
            assert_eq!(assessment.value(), Some("226厘米"));
        }
    }

    #[test]
    fn known_subject_type_anchors_attribute_domain() {
        let extra = r#"
[[extraction_rules]]
name = "公司"
entity_type = "Company"
priority = 10
patterns = [{ pos_pattern = ["nr"], keyword_patterns = ["公司"], property_mappings = { name = "pos:nr" } }]

[type_system]
[[type_system.disjoint]]
left = "Person"
right = "Company"
"#;
        let config = std::sync::Arc::new(extraction_config_with(extra));
        let extractor = crate::KnowledgeExtractor::new(config);
        let tagged = vec![
            TaggedWord {
                text: "甲公司".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let items = extractor.extract("甲公司身高2米", &tagged);
        assert_eq!(items.len(), 1, "{:?}", items);
        match &items[0] {
            crate::KnowledgeItem::Entity { label, properties } => {
                assert_eq!(label, "Company");
                assert_eq!(properties.get("name").map(String::as_str), Some("甲公司"));
                assert!(
                    properties.get("height").is_none(),
                    "可证明互斥的主体类型必须拒绝属性断言：{:?}",
                    properties
                );
            }
            other => panic!("应只保留公司实体，实际 {:?}", other),
        }
    }

    #[test]
    fn unknown_subject_type_keeps_open_world_attribute() {
        let extra = r#"
[[extraction_rules]]
name = "公司"
entity_type = "Company"
priority = 10
patterns = [{ pos_pattern = ["nr"], keyword_patterns = ["公司"], property_mappings = { name = "pos:nr" } }]
"#;
        let config = std::sync::Arc::new(extraction_config_with(extra));
        let extractor = crate::KnowledgeExtractor::new(config);
        let tagged = vec![
            TaggedWord {
                text: "甲公司".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let items = extractor.extract("甲公司身高2米", &tagged);
        assert_eq!(items.len(), 1, "{:?}", items);
        match &items[0] {
            crate::KnowledgeItem::Entity { label, properties } => {
                assert_eq!(label, "Company");
                assert_eq!(
                    properties.get("height").map(String::as_str),
                    Some("200厘米"),
                    "未声明互斥时按开放世界接受：{:?}",
                    properties
                );
            }
            other => panic!("应合并为公司实体，实际 {:?}", other),
        }
    }

    #[test]
    fn quantity_values_normalize_to_declared_unit() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let types = person_types();

        let meters = registry.assess(&types, &type_set(&["Person"]), "身高", "2.26米");
        assert_eq!(meters.verdict, AttributeVerdict::Accepted);
        assert_eq!(meters.value(), Some("226厘米"));

        let centimeters = registry.assess(&types, &type_set(&["Person"]), "身长", "226厘米");
        assert_eq!(centimeters.value(), Some("226厘米"));

        let unitless = registry.assess(&types, &type_set(&["Person"]), "身高", "226");
        assert_eq!(unitless.value(), Some("226厘米"));

        let approximate = registry.assess(&types, &type_set(&["Person"]), "身高", "约2.26米");
        assert_eq!(approximate.value(), Some("约226厘米"));

        let age = registry.assess(&types, &type_set(&["Athlete"]), "年龄", "18");
        assert_eq!(age.verdict, AttributeVerdict::Accepted);
        assert_eq!(age.value(), Some("18岁"));
    }

    #[test]
    fn dimension_and_type_violations_are_explicit() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let types = person_types();

        let mismatch = registry.assess(&types, &type_set(&["Person"]), "身高", "120千克");
        assert_eq!(mismatch.verdict, AttributeVerdict::ValueMismatch);
        assert!(mismatch.notes[0].contains("量纲不符"));

        let domain = registry.assess(&types, &type_set(&["Company"]), "身高", "226厘米");
        assert_eq!(domain.verdict, AttributeVerdict::DomainMismatch);

        let unknown_subject = registry.assess(&types, &BTreeSet::new(), "身高", "226厘米");
        assert_eq!(
            unknown_subject.verdict,
            AttributeVerdict::AcceptedTypeUnconfirmed
        );
        assert_eq!(unknown_subject.value(), Some("226厘米"));

        let unknown_attribute = registry.assess(&types, &type_set(&["Person"]), "臂展", "220厘米");
        assert_eq!(
            unknown_attribute.verdict,
            AttributeVerdict::UnknownAttribute
        );
    }

    #[test]
    fn mentions_split_subject_attribute_and_value() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let text = "录入：姚明的身高是2.26米。";
        let tagged = vec![
            TaggedWord {
                text: "姚明".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let mentions = registry.extract_mentions(text, &tagged);
        assert_eq!(mentions.len(), 1);
        let mention = &mentions[0];
        assert_eq!(mention.subject, "姚明");
        assert_eq!(mention.attribute, "height");
        assert_eq!(mention.raw_value, "2.26米");

        let question = registry.extract_mentions(
            "姚明的身高是多少？",
            &[TaggedWord {
                text: "姚明".to_string(),
                pos: "nr".to_string(),
            }],
        );
        assert!(question.is_empty(), "问句不应落成属性断言");
    }

    #[test]
    fn mentions_extract_all_attributes_with_shared_subject() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let tagged = vec![
            TaggedWord {
                text: "姚明".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
            TaggedWord {
                text: "体重".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let mentions = registry.extract_mentions("姚明身高2.26米，体重120千克。", &tagged);
        assert_eq!(mentions.len(), 2, "{mentions:?}");
        assert_eq!(mentions[0].subject, "姚明");
        assert_eq!(mentions[0].attribute, "height");
        assert_eq!(mentions[0].raw_value, "2.26米");
        assert_eq!(mentions[1].subject, "姚明", "省略主体应沿用前一条候选");
        assert_eq!(mentions[1].attribute, "weight");
        assert_eq!(mentions[1].raw_value, "120千克");
    }

    #[test]
    fn mentions_prefer_longest_surface_at_same_position() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let mentions = registry.extract_mentions("姚明净身高225厘米", &[]);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].attribute, "net_height");
        assert_eq!(mentions[0].raw_value, "225厘米");
    }

    #[test]
    fn mentions_bound_value_by_next_attribute_without_punctuation() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let mentions = registry.extract_mentions("姚明身高226体重120", &[]);
        assert_eq!(mentions.len(), 2, "{mentions:?}");
        assert_eq!(mentions[0].raw_value, "226");
        assert_eq!(mentions[1].attribute, "weight");
        assert_eq!(mentions[1].raw_value, "120");
    }

    #[test]
    fn string_profile_accepts_and_question_is_rejected() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let types = person_types();
        let birthplace = registry.assess(&types, &type_set(&["Person"]), "籍贯", "上海");
        assert_eq!(birthplace.value(), Some("上海"));
        let question = registry.assess(&types, &type_set(&["Person"]), "籍贯", "哪里");
        assert_eq!(question.verdict, AttributeVerdict::ValueMismatch);
    }

    #[test]
    fn malformed_declarations_are_reported_without_dropping_table() {
        let registry = AttributeRegistry::from_config(&AttributeBridgeConfig {
            enabled: true,
            profiles: vec![
                AttributeProfileConfig {
                    name: "height".to_string(),
                    aliases: vec!["高度".to_string()],
                    unit: Some("不存在单位".to_string()),
                    ..AttributeProfileConfig::default()
                },
                AttributeProfileConfig {
                    name: "width".to_string(),
                    aliases: vec!["高度".to_string()],
                    parent: Some("missing_parent".to_string()),
                    ..AttributeProfileConfig::default()
                },
            ],
        });
        assert_eq!(registry.len(), 2);
        assert!(registry.issues().iter().any(|issue| issue.detail.contains("未知单位")));
        assert!(registry
            .issues()
            .iter()
            .any(|issue| issue.detail.contains("已被")));
        assert!(registry
            .issues()
            .iter()
            .any(|issue| issue.detail.contains("父属性")));
    }
    fn extraction_config() -> crate::SystemConfig {
        extraction_config_with("")
    }

    fn extraction_config_with(extra: &str) -> crate::SystemConfig {
        toml::from_str(&format!(
            r#"
[system]
oxigraph_endpoint = "http://localhost:7878"
oxigraph_store_path = "oxigraph_data"
default_language = "zh"

[nlp]
import_keywords = ["录入"]
pos_dict = {{}}
[nlp.name_patterns]
prefixes = []
suffixes = []
full_names = []
profession_suffix = []

[[entity_types]]
label = "Person"
examples = []
[[entity_types.properties]]
name = "name"
data_type = "string"
required = true

[[extraction_rules]]
name = "无"
entity_type = "Person"
patterns = []
priority = 0

[[prolog_rules]]
name = "属性查询"
pattern = ["nr"]
query_type = "property_query"
priority = 1

[messages]
startup = ""
ready = ""
query_prompt = ""
import_prompt = ""
quit_prompt = ""
importing = ""
no_knowledge_extracted = ""
cannot_understand = ""
process_error = ""

[attributes]
enabled = true
[[attributes.profiles]]
name = "height"
aliases = ["身高"]
domain = ["Person"]
value_type = "quantity"
dimension = "length"
unit = "厘米"
functional = true

[[attributes.profiles]]
name = "net_height"
aliases = ["净身高"]
domain = ["Person"]
value_type = "quantity"
dimension = "length"
unit = "厘米"
parent = "height"
{extra}"#
        ))
        .expect("最小配置必须可解析")
    }

    #[test]
    fn knowledge_extractor_emits_normalized_attribute_fact() {
        let config = std::sync::Arc::new(extraction_config());
        let extractor = crate::KnowledgeExtractor::new(config);
        let tagged = vec![
            TaggedWord {
                text: "姚明".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let items = extractor.extract("姚明的身高是2.26米", &tagged);
        assert_eq!(items.len(), 1, "{:?}", items);
        match &items[0] {
            crate::KnowledgeItem::Entity { label, properties } => {
                assert_eq!(label, "Person");
                assert_eq!(properties.get("name").map(String::as_str), Some("姚明"));
                assert_eq!(
                    properties.get("height").map(String::as_str),
                    Some("226厘米")
                );
            }
            other => panic!("期望实体项，实际 {:?}", other),
        }
    }

    #[test]
    fn knowledge_extractor_merges_multiple_attribute_facts_for_one_subject() {
        let config = std::sync::Arc::new(extraction_config());
        let extractor = crate::KnowledgeExtractor::new(config);
        let tagged = vec![
            TaggedWord {
                text: "姚明".to_string(),
                pos: "nr".to_string(),
            },
            TaggedWord {
                text: "身高".to_string(),
                pos: "unknown".to_string(),
            },
            TaggedWord {
                text: "净身高".to_string(),
                pos: "unknown".to_string(),
            },
        ];
        let items = extractor.extract("姚明身高2.26米，净身高2.25米", &tagged);
        assert_eq!(items.len(), 1, "{:?}", items);
        match &items[0] {
            crate::KnowledgeItem::Entity { label, properties } => {
                assert_eq!(label, "Person");
                assert_eq!(properties.get("name").map(String::as_str), Some("姚明"));
                assert_eq!(
                    properties.get("height").map(String::as_str),
                    Some("226厘米")
                );
                assert_eq!(
                    properties.get("net_height").map(String::as_str),
                    Some("225厘米")
                );
            }
            other => panic!("期望实体项，实际 {:?}", other),
        }
    }

    #[test]
    fn attribute_value_overrides_rule_value_for_same_subject() {
        let mut items = vec![crate::KnowledgeItem::Entity {
            label: "Person".to_string(),
            properties: [
                ("name".to_string(), "姚明".to_string()),
                ("height".to_string(), "226".to_string()),
            ]
            .into_iter()
            .collect(),
        }];
        crate::merge_attribute_item(
            &mut items,
            crate::KnowledgeItem::Entity {
                label: "Person".to_string(),
                properties: [
                    ("name".to_string(), "姚明".to_string()),
                    ("height".to_string(), "226厘米".to_string()),
                ]
                .into_iter()
                .collect(),
            },
        );
        assert_eq!(items.len(), 1);
        match &items[0] {
            crate::KnowledgeItem::Entity { properties, .. } => {
                assert_eq!(
                    properties.get("height").map(String::as_str),
                    Some("226厘米")
                );
            }
            other => panic!("期望实体项，实际 {:?}", other),
        }
    }

    #[test]
    fn request_parser_requires_question_and_explicit_unit() {
        let registry = AttributeRegistry::from_config(&sample_config());
        let request = registry
            .parse_request("姚明的身高是多少厘米？")
            .expect("应识别单位化属性问句");
        assert_eq!(request.subject, "姚明");
        assert_eq!(request.attribute, "height");
        assert_eq!(request.unit.as_deref(), Some("厘米"));

        let meters = registry.parse_request("姚明身高多少米").unwrap();
        assert_eq!(meters.unit.as_deref(), Some("米"));

        assert!(
            registry.parse_request("姚明的身高是多少").is_none(),
            "无显式单位不抢答"
        );
        assert!(
            registry.parse_request("姚明身高2.26米").is_none(),
            "陈述句不是问句"
        );
    }

    #[test]
    fn shipped_config_compiles_attribute_profiles_without_issues() {
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let registry = AttributeRegistry::from_config(&config.attributes);
        assert!(registry.issues().is_empty(), "{:?}", registry.issues());
        assert!(registry.resolve("身高").is_some());
        assert!(registry.resolve("净身高").is_some());
        assert_eq!(registry.ancestors("净身高"), vec!["height".to_string()]);
    }

}

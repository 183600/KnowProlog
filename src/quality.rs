//! QPR：质量护照（Quality Passport）。
//!
//! 为每条知识给出**确定性可信向量**：
//! 图谱快照 + 谓词画像 + 类型体系 + 冲突笔记 → 可解释、可复算、可棘轮的分数。
//!
//! 设计要点（纯符号，零训练）：
//! - **固定公式、常量权重**：五个维度（完整性 / 一致性 / 可溯源性 / 连通性 /
//!   类型覆盖）全部由本模块顶部的常量决定，同输入必同字节；
//! - **类型让质量可判定**：完整性以「类型的必备属性」为坐标——没有类型就
//!   不知道缺什么，于是未落型实体得到中性先验分与明确的缺口条目
//!   （可被对话层转成澄清提问）；
//! - **质量让类型可说出口**：冲突（导入时被拒的「不同说法」）记入护照的
//!   缺口与一致性扣分——低分用于排序、措辞与追问，
//!   **绝不写回、绝不删事实**（低分 ≠ 假：事实仍可查询、可被引用）；
//! - **digest 锁定**：`QualityReport.digest` 由规范 JSON + FNV-1a 得出，
//!   语料/画像/类型体系任一变化都会改变摘要，供 `eval-quality` 棘轮判定
//!   「评的是不是同一份东西」。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::predicate_registry::{ImportConflict, PredicateRegistry};
use crate::prolog_engine::PrologFact;
use crate::type_system::TypeSystem;
use crate::EntityTypeConfig;

/// 综合分权重：完整性 / 一致性 / 可溯源性 / 连通性（常量，合计 1.0）。
pub const WEIGHT_COMPLETENESS: f64 = 0.35;
pub const WEIGHT_CONSISTENCY: f64 = 0.35;
pub const WEIGHT_PROVENANCE: f64 = 0.15;
pub const WEIGHT_CONNECTIVITY: f64 = 0.15;

/// 未落型实体的完整性中性先验（不知道缺什么，不奖励也不重罚）。
pub const NEUTRAL_COMPLETENESS: f64 = 0.5;
/// 每条冲突笔记的一致性扣分。
pub const CONFLICT_PENALTY: f64 = 0.5;
/// 每对互斥类型共存的一致性扣分。
pub const DISJOINT_TYPE_PENALTY: f64 = 0.5;
/// 可溯源性分档：无来源记录 / 单一来源标签 / 多来源标签。
pub const PROV_NO_SOURCE: f64 = 0.3;
pub const PROV_SINGLE_SOURCE: f64 = 0.6;
pub const PROV_MULTI_SOURCE: f64 = 0.85;
/// 连通性饱和度：关系度达到 4 即记满分。
pub const CONNECTIVITY_FULL_DEGREE: f64 = 4.0;
/// 低分阈值（仅用于报告排序与追问建议，不改变任何存储）。
pub const LOW_SCORE_THRESHOLD: f64 = 0.5;
/// 报告正文中清单类字段的最大展示条数。
pub const REPORT_LIST_CAP: usize = 12;

/// 单个实体的质量护照。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityPassport {
    pub name: String,
    /// 规范类型名（含闭包展开，排序）。
    pub types: Vec<String>,
    /// 类型 → 来源标签（排序）。
    pub type_sources: BTreeMap<String, Vec<String>>,
    pub completeness: f64,
    pub consistency: f64,
    pub provenance: f64,
    pub connectivity: f64,
    /// 加权综合分（四维常量权重）。
    pub score: f64,
    /// 关系度（涉及该实体的关系条数）。
    pub degree: usize,
    /// 可行动缺口（必备属性缺失 / 未落型 / 冲突笔记）。
    pub gaps: Vec<String>,
    /// 与该实体相关的冲突笔记（导入时被拒的入值）。
    pub conflicts: Vec<ImportConflict>,
    /// 该实体涉及谓词的来源标签并集（确定性排序）。
    ///
    /// 这是护照里唯一直接面向「怎么引用」的字段：无来源时为空，
    /// 但**空来源不等于事实为假**，只表示引用方式要更保守。
    #[serde(default)]
    pub source_tags: Vec<String>,
}

/// 一条断言（实体 + 可选属性）的「说话许可证」。
///
/// 质量层不输出单一置信度，而是把护照翻译成可执行的话语许可：
/// - `conflicts` 非空 → 必须披露两种说法，不能静默择一；
/// - `typed == false` → 身份问句应先澄清类型；
/// - `gaps` 非空 → 可以追问一个缺口，但不得把缺属性说成事实不确定；
/// - `source_tags` 决定引用方式，而不是真值。
///
/// 该结构只含字符串/布尔/整数，便于写入对话快照并逐字节复现。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimWarrant {
    pub entity: String,
    /// 规范化后的属性（若配置/画像能解析）；身份问句为 None。
    pub attribute: Option<String>,
    /// 已落类型（规范名，排序）。
    pub types: Vec<String>,
    /// 该实体在该属性上的已知取值（排序、去重）。
    pub known_values: Vec<String>,
    /// 该断言可引用的来源标签。
    pub source_tags: Vec<String>,
    /// 与该断言相关的冲突笔记。
    pub conflicts: Vec<ImportConflict>,
    /// 护照缺口（未落型 / 缺必备属性 / 冲突 / 互斥类型）。
    pub gaps: Vec<String>,
    /// 是否有明确类型。
    pub typed: bool,
    /// 护照是否无缺口（不等于真值）。
    pub complete: bool,
    /// 生成该担保时的知识版本。
    pub revision: u64,
}

impl ClaimWarrant {
    pub fn has_source(&self) -> bool {
        !self.source_tags.is_empty()
    }

    /// 是否需要先澄清类型（未落型且问的是身份）。
    pub fn needs_type_clarification(&self) -> bool {
        !self.typed
    }

    /// 第一个可行动、且尚未被冲突/类型话语覆盖的缺口。
    pub fn actionable_gap(&self) -> Option<&str> {
        self.gaps
            .iter()
            .find(|gap| !gap.starts_with("未落型") && !gap.starts_with("存在未裁决冲突"))
            .map(String::as_str)
    }

    /// 冲突披露句（无冲突时为 None；按现有值优先，绝不覆盖）。
    ///
    /// 多条冲突全部列出，避免只披露第一条而静默隐藏其余说法。
    pub fn conflict_note(&self) -> Option<String> {
        if self.conflicts.is_empty() {
            return None;
        }
        let notes: Vec<String> = self
            .conflicts
            .iter()
            .map(|conflict| {
                let existing = if conflict.existing.is_empty() {
                    "（无）".to_string()
                } else {
                    conflict.existing.join("、")
                };
                format!(
                    "关于「{}」，还有另一种说法「{}」，与已有「{}」不一致",
                    conflict.predicate, conflict.incoming, existing
                )
            })
            .collect();
        Some(format!(
            "{}；我暂按已有记录回答，你可以确认哪个为准。",
            notes.join("；")
        ))
    }

    /// 引用说明：有来源就列出，没有来源就如实说明（不改变真值）。
    pub fn citation_note(&self) -> String {
        if self.source_tags.is_empty() {
            "（据现有记录，该信息尚未登记来源）".to_string()
        } else {
            format!("（出处：{}）", self.source_tags.join("、"))
        }
    }
}

/// 质量索引：把一次护照报告投影成对话可消费的「担保查询表」。
///
/// 只读派生数据：不写图谱、不改事实。`revision` 由宿主注入，
/// 用于多轮记忆里判断「该回答基于哪一版知识」。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityIndex {
    pub revision: u64,
    pub digest: String,
    entities: BTreeMap<String, EntityPassport>,
    /// 实体 → 属性 → 取值（仅主语侧事实；逆谓词不混入）。
    values: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    /// 实体 → 谓词 → 来源标签。
    predicate_sources: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    /// 非逆向别名 → 规范谓词（让「师父/老师」问句也享受同一份担保）。
    attribute_aliases: BTreeMap<String, String>,
}

impl QualityIndex {
    pub fn empty() -> Self {
        QualityIndex {
            revision: 0,
            digest: String::new(),
            entities: BTreeMap::new(),
            values: BTreeMap::new(),
            predicate_sources: BTreeMap::new(),
            attribute_aliases: BTreeMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn with_revision(mut self, revision: u64) -> Self {
        self.revision = revision;
        self
    }

    pub fn passport(&self, entity: &str) -> Option<&EntityPassport> {
        self.entities.get(entity)
    }

    /// 从护照报告 + 事实快照 + 谓词画像构建索引（确定性）。
    pub fn from_report(
        report: &QualityReport,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
    ) -> Self {
        let entities: BTreeMap<String, EntityPassport> = report
            .entities
            .iter()
            .map(|passport| (passport.name.clone(), passport.clone()))
            .collect();
        let mut values: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name, properties, ..
                } => {
                    let slots = values.entry(name.clone()).or_default();
                    for (key, value) in properties {
                        let entry = slots.entry(key.clone()).or_default();
                        entry.push(value.clone());
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    let entry = values
                        .entry(from.clone())
                        .or_default()
                        .entry(rel_type.clone())
                        .or_default();
                    entry.push(to.clone());
                }
            }
        }
        for slots in values.values_mut() {
            for entry in slots.values_mut() {
                entry.sort();
                entry.dedup();
            }
        }
        let mut predicate_sources: BTreeMap<String, BTreeMap<String, Vec<String>>> =
            BTreeMap::new();
        for (entity, slots) in &values {
            let sources = predicate_sources.entry(entity.clone()).or_default();
            for predicate in slots.keys() {
                if let Some(profile) = registry.profile(predicate) {
                    if !profile.sources.is_empty() {
                        sources
                            .insert(predicate.clone(), profile.sources.iter().cloned().collect());
                    }
                }
            }
        }
        let mut attribute_aliases: BTreeMap<String, String> = BTreeMap::new();
        for canonical in registry.canonical_names() {
            attribute_aliases.insert(canonical.to_string(), canonical.to_string());
        }
        for (alias, canonical, inverted) in registry.alias_pairs() {
            if !inverted {
                attribute_aliases.insert(alias, canonical);
            }
        }
        QualityIndex {
            revision: 0,
            digest: report.digest.clone(),
            entities,
            values,
            predicate_sources,
            attribute_aliases,
        }
    }

    /// 查询一条断言的担保；实体不存在时返回 None（交给常规管线）。
    pub fn warrant_for(&self, entity: &str, attribute: Option<&str>) -> Option<ClaimWarrant> {
        let passport = self.entities.get(entity)?;
        let canonical_attribute = attribute.map(|raw| {
            let trimmed = raw.trim();
            self.attribute_aliases
                .get(trimmed)
                .cloned()
                .unwrap_or_else(|| trimmed.to_string())
        });
        let known_values = canonical_attribute
            .as_ref()
            .and_then(|attribute| self.values.get(entity)?.get(attribute))
            .cloned()
            .unwrap_or_default();
        let conflicts: Vec<ImportConflict> = match &canonical_attribute {
            Some(attribute) => passport
                .conflicts
                .iter()
                .filter(|conflict| conflict.predicate == *attribute)
                .cloned()
                .collect(),
            None => passport.conflicts.clone(),
        };
        // 来源必须按断言（谓词）取，不能回退到实体级并集：
        // 「另一个属性有出处」不能给本属性背书。身份问句（无属性）才用实体并集。
        let source_tags = match canonical_attribute.as_ref() {
            Some(attribute) => self
                .predicate_sources
                .get(entity)
                .and_then(|sources| sources.get(attribute))
                .cloned()
                .unwrap_or_default(),
            None => passport.source_tags.clone(),
        };
        Some(ClaimWarrant {
            entity: entity.to_string(),
            attribute: canonical_attribute,
            types: passport.types.clone(),
            known_values,
            source_tags,
            conflicts,
            gaps: passport.gaps.clone(),
            typed: !passport.types.is_empty(),
            complete: passport.gaps.is_empty(),
            revision: self.revision,
        })
    }
}

/// 语料级汇总。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualitySummary {
    pub entities: usize,
    pub typed_entities: usize,
    pub typed_rate: f64,
    pub mean_score: f64,
    pub low_score: usize,
    pub conflicts: usize,
    /// 未落型且在图中有连接的实体（最值得问「这是人物还是地点？」的清单）。
    pub clarify_candidates: Vec<String>,
}

/// 质量护照总报告（实体护照 + 汇总 + 摘要）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityReport {
    pub entities: Vec<EntityPassport>,
    pub summary: QualitySummary,
    /// 规范内容摘要（entities + summary，FNV-1a 64）。
    pub digest: String,
}

impl QualityReport {
    pub fn passport_for(&self, entity: &str) -> Option<&EntityPassport> {
        self.entities
            .iter()
            .find(|passport| passport.name == entity)
    }

    /// 报告正文（确定性；低分实体与缺口清单优先展示）。
    pub fn render_text(&self) -> String {
        let summary = &self.summary;
        let mut lines = vec![format!(
            "质量护照：实体 {} 个（已落型 {}，落型率 {:.3}），平均分 {:.3}，低分 {} 个，冲突笔记 {} 条。",
            summary.entities,
            summary.typed_entities,
            summary.typed_rate,
            summary.mean_score,
            summary.low_score,
            summary.conflicts
        )];
        if !summary.clarify_candidates.is_empty() {
            lines.push(format!(
                "未落型（可追问「是人物还是地点」）：{}",
                summary
                    .clarify_candidates
                    .iter()
                    .take(REPORT_LIST_CAP)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("、")
            ));
        }
        let mut low: Vec<&EntityPassport> = self
            .entities
            .iter()
            .filter(|passport| passport.score < LOW_SCORE_THRESHOLD)
            .collect();
        low.sort_by(|left, right| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.name.cmp(&right.name))
        });
        if low.is_empty() {
            lines.push("没有低于阈值的实体：全部护照均达到低分线。".to_string());
        } else {
            lines.push(format!(
                "低分实体（按分数升序，仅用于排序与追问，低分≠假）：共 {} 个",
                low.len()
            ));
            for passport in low.iter().take(REPORT_LIST_CAP) {
                lines.push(format!(
                    "  • {}：{:.3}（完整 {:.2} / 一致 {:.2} / 溯源 {:.2} / 连通 {:.2}）",
                    passport.name,
                    passport.score,
                    passport.completeness,
                    passport.consistency,
                    passport.provenance,
                    passport.connectivity
                ));
                for gap in &passport.gaps {
                    lines.push(format!("      - {}", gap));
                }
            }
            if low.len() > REPORT_LIST_CAP {
                lines.push(format!("  ……其余 {} 个省略", low.len() - REPORT_LIST_CAP));
            }
        }
        lines.push(format!("摘要：{}", self.digest));
        lines.join("\n")
    }
}

/// 实体护照正文（交互查询「质量护照：X」用）。
pub fn render_entity_passport(passport: &EntityPassport) -> String {
    let mut lines = vec![format!(
        "实体「{}」的护照：综合 {:.3}（完整 {:.2} / 一致 {:.2} / 溯源 {:.2} / 连通 {:.2}）。",
        passport.name,
        passport.score,
        passport.completeness,
        passport.consistency,
        passport.provenance,
        passport.connectivity
    )];
    if passport.types.is_empty() {
        lines.push("  类型：未落型（完整性按中性先验计分）".to_string());
    } else {
        let types: Vec<String> = passport
            .types
            .iter()
            .map(|type_name| {
                let sources = passport
                    .type_sources
                    .get(type_name)
                    .map(|sources| sources.join("+"))
                    .unwrap_or_else(|| "未知".to_string());
                format!("{}（{}）", type_name, sources)
            })
            .collect();
        lines.push(format!("  类型：{}", types.join("；")));
    }
    if passport.gaps.is_empty() {
        lines.push("  缺口：无".to_string());
    } else {
        for gap in &passport.gaps {
            lines.push(format!("  缺口：{}", gap));
        }
    }
    lines.join("\n")
}

/// 构建质量护照。
///
/// 输入全部只读：图谱事实快照、谓词画像（画像来源统计与冲突笔记）、
/// 类型体系（别名/子类/互斥 + 推断）、实体类型配置（必备属性坐标）。
pub fn build_report(
    facts: &[PrologFact],
    registry: &PredicateRegistry,
    types: &TypeSystem,
    entity_types: &[EntityTypeConfig],
) -> QualityReport {
    let mut explicit: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut properties: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut relations: Vec<(String, String, String)> = Vec::new();
    for fact in facts {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties: props,
            } => {
                explicit
                    .entry(name.clone())
                    .or_default()
                    .insert(entity_type.clone());
                let slot = properties.entry(name.clone()).or_default();
                for (key, value) in props {
                    slot.insert(key.clone(), value.clone());
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                relations.push((from.clone(), rel_type.clone(), to.clone()));
            }
        }
    }

    // 谓词画像 → 域/值域（类型推断的输入）
    let mut domains: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut ranges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for canonical in registry.canonical_names() {
        if let Some(profile) = registry.profile(canonical) {
            if !profile.domain.is_empty() {
                domains.insert(
                    profile.canonical.clone(),
                    profile.domain.iter().cloned().collect(),
                );
            }
            if !profile.range.is_empty() {
                ranges.insert(
                    profile.canonical.clone(),
                    profile.range.iter().cloned().collect(),
                );
            }
        }
    }
    let inferred = types.infer_entity_types(&relations, &domains, &ranges, &explicit);

    // 冲突笔记按主语归组
    let mut conflicts_by_subject: BTreeMap<String, Vec<ImportConflict>> = BTreeMap::new();
    for conflict in registry.conflicts() {
        conflicts_by_subject
            .entry(conflict.subject.clone())
            .or_default()
            .push(conflict);
    }

    // 节点全集：实体事实 + 关系两端
    let mut names: BTreeSet<String> = BTreeSet::new();
    names.extend(explicit.keys().cloned());
    names.extend(properties.keys().cloned());
    for (from, _, to) in &relations {
        names.insert(from.clone());
        names.insert(to.clone());
    }

    let mut degree_map: BTreeMap<String, usize> = BTreeMap::new();
    for (from, _, to) in &relations {
        *degree_map.entry(from.clone()).or_insert(0) += 1;
        if from != to {
            *degree_map.entry(to.clone()).or_insert(0) += 1;
        }
    }

    let required_by_type: BTreeMap<String, BTreeSet<String>> = entity_types
        .iter()
        .map(|config| {
            let required: BTreeSet<String> = config
                .properties
                .iter()
                .filter(|property| property.required)
                .map(|property| property.name.clone())
                .collect();
            (types.canonical(&config.label), required)
        })
        .collect();

    let mut entities: Vec<EntityPassport> = Vec::with_capacity(names.len());
    for name in &names {
        let degree = degree_map.get(name).copied().unwrap_or(0);
        let empty = BTreeMap::new();
        let type_map = inferred.get(name).unwrap_or(&empty);
        let resolved_types: Vec<String> = type_map.keys().cloned().collect();
        let type_sources: BTreeMap<String, Vec<String>> = type_map
            .iter()
            .map(|(type_name, sources)| {
                (
                    type_name.clone(),
                    sources
                        .iter()
                        .map(|source| source.label().to_string())
                        .collect(),
                )
            })
            .collect();
        let props = properties.get(name);

        // 完整性：所有已落类型的必备属性并集覆盖比例；
        // 未落型 → 中性先验；落型但无必备属性声明 → 满分。
        let mut required: BTreeSet<String> = BTreeSet::new();
        for type_name in &resolved_types {
            if let Some(needed) = required_by_type.get(type_name) {
                required.extend(needed.iter().cloned());
            }
        }
        let mut gaps: Vec<String> = Vec::new();
        let completeness = if resolved_types.is_empty() || required.is_empty() {
            if resolved_types.is_empty() {
                NEUTRAL_COMPLETENESS
            } else {
                1.0
            }
        } else {
            let present = required
                .iter()
                .filter(|key| props.map_or(false, |props| props.contains_key(*key)))
                .count();
            present as f64 / required.len() as f64
        };
        if resolved_types.is_empty() {
            gaps.push("未落型：无法判定必备属性，可回答「这是人物还是地点？」后回填".to_string());
        }
        for key in &required {
            if !props.map_or(false, |props| props.contains_key(key)) {
                gaps.push(format!("缺少必备属性「{}」", key));
            }
        }

        // 一致性：冲突笔记 + 互斥类型共存，逐项固定扣分，下限 0。
        let entity_conflicts = conflicts_by_subject.get(name).cloned().unwrap_or_default();
        let mut disjoint_pairs = 0usize;
        for (index, left) in resolved_types.iter().enumerate() {
            for right in resolved_types.iter().skip(index + 1) {
                if types.types_disjoint(left, right) {
                    disjoint_pairs += 1;
                }
            }
        }
        let consistency = (1.0
            - CONFLICT_PENALTY * entity_conflicts.len() as f64
            - DISJOINT_TYPE_PENALTY * disjoint_pairs as f64)
            .max(0.0);
        for conflict in &entity_conflicts {
            gaps.push(format!(
                "存在未裁决冲突：{}已有「{}」，另有说法「{}」（不覆盖，暂按已有值）",
                conflict.predicate,
                if conflict.existing.is_empty() {
                    "（无）".to_string()
                } else {
                    conflict.existing.join("、")
                },
                conflict.incoming
            ));
        }
        for (index, left) in resolved_types.iter().enumerate() {
            for right in resolved_types.iter().skip(index + 1) {
                if types.types_disjoint(left, right) {
                    gaps.push(format!("互斥类型共存：「{}」与「{}」", left, right));
                }
            }
        }

        // 可溯源性：实体涉及谓词的来源标签并集（画像口径）。
        let mut predicates: BTreeSet<String> = BTreeSet::new();
        if let Some(props) = props {
            predicates.extend(props.keys().cloned());
        }
        for (from, predicate, to) in &relations {
            if from == name || to == name {
                predicates.insert(predicate.clone());
            }
        }
        let mut source_tags: BTreeSet<String> = BTreeSet::new();
        for predicate in &predicates {
            if let Some(profile) = registry.profile(predicate) {
                source_tags.extend(profile.sources.iter().cloned());
            }
        }
        let provenance = match source_tags.len() {
            0 => PROV_NO_SOURCE,
            1 => PROV_SINGLE_SOURCE,
            _ => PROV_MULTI_SOURCE,
        };

        let connectivity = (degree as f64 / CONNECTIVITY_FULL_DEGREE).min(1.0);
        let score = WEIGHT_COMPLETENESS * completeness
            + WEIGHT_CONSISTENCY * consistency
            + WEIGHT_PROVENANCE * provenance
            + WEIGHT_CONNECTIVITY * connectivity;
        entities.push(EntityPassport {
            name: name.clone(),
            types: resolved_types,
            type_sources,
            completeness,
            consistency,
            provenance,
            connectivity,
            score,
            degree,
            gaps,
            conflicts: entity_conflicts,
            source_tags: source_tags.into_iter().collect(),
        });
    }

    let typed_entities = entities
        .iter()
        .filter(|passport| !passport.types.is_empty())
        .count();
    let entity_count = entities.len();
    let mean_score = if entity_count == 0 {
        0.0
    } else {
        entities.iter().map(|passport| passport.score).sum::<f64>() / entity_count as f64
    };
    let low_score = entities
        .iter()
        .filter(|passport| passport.score < LOW_SCORE_THRESHOLD)
        .count();
    let conflicts = entities
        .iter()
        .map(|passport| passport.conflicts.len())
        .sum();

    let mut clarify_candidates: Vec<(usize, String)> = entities
        .iter()
        .filter(|passport| passport.types.is_empty() && passport.degree > 0)
        .map(|passport| (passport.degree, passport.name.clone()))
        .collect();
    clarify_candidates
        .sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let clarify_candidates: Vec<String> = clarify_candidates
        .into_iter()
        .map(|(_, name)| name)
        .collect();

    let summary = QualitySummary {
        entities: entity_count,
        typed_entities,
        typed_rate: if entity_count == 0 {
            0.0
        } else {
            typed_entities as f64 / entity_count as f64
        },
        mean_score,
        low_score,
        conflicts,
        clarify_candidates,
    };
    let digest = {
        let value = serde_json::json!({
            "entities": &entities,
            "summary": &summary,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    };
    QualityReport {
        entities,
        summary,
        digest,
    }
}

// ---------------------------------------------------------------------------
// eval-quality：基线棘轮
// ---------------------------------------------------------------------------

/// 棘轮判定使用的数值容差。
pub const RATCHET_EPSILON: f64 = 1e-6;

/// 质量基线（`eval-quality --update-baseline` 落盘）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBaseline {
    pub digest: String,
    pub entities: usize,
    pub typed_rate: f64,
    pub mean_score: f64,
    pub low_score: usize,
    pub conflicts: usize,
}

impl QualityBaseline {
    pub fn from_report(report: &QualityReport) -> Self {
        Self {
            digest: report.digest.clone(),
            entities: report.summary.entities,
            typed_rate: report.summary.typed_rate,
            mean_score: report.summary.mean_score,
            low_score: report.summary.low_score,
            conflicts: report.summary.conflicts,
        }
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap_or_default())
    }

    pub fn load(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// 单项指标对比。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityDelta {
    pub metric: String,
    pub before: f64,
    pub after: f64,
    /// "higher_better" / "lower_better"
    pub direction: String,
    pub passed: bool,
}

/// 棘轮裁决。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub deltas: Vec<QualityDelta>,
}

impl QualityVerdict {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "eval-quality 棘轮：{}（语料摘要{}）。",
            if self.passed { "通过" } else { "未通过" },
            if self.digest_matches {
                "一致"
            } else {
                "已变化：指标按新语料比较"
            }
        )];
        for delta in &self.deltas {
            lines.push(format!(
                "  {} {}：{:.6} → {:.6}{}",
                if delta.passed { "✓" } else { "✗" },
                delta.metric,
                delta.before,
                delta.after,
                if delta.direction == "higher_better" {
                    "（越高越好）"
                } else {
                    "（越低越好）"
                }
            ));
        }
        lines.join("\n")
    }
}

/// 对比基线：均值/落型率不得下降，低分/冲突数不得上升（容差 `epsilon`）。
pub fn compare_baseline(
    baseline: &QualityBaseline,
    current: &QualityReport,
    epsilon: f64,
) -> QualityVerdict {
    let mut deltas = Vec::new();
    let mut push = |metric: &str, before: f64, after: f64, higher_better: bool| {
        let passed = if higher_better {
            after + epsilon >= before
        } else {
            after <= before + epsilon
        };
        deltas.push(QualityDelta {
            metric: metric.to_string(),
            before,
            after,
            direction: if higher_better {
                "higher_better".to_string()
            } else {
                "lower_better".to_string()
            },
            passed,
        });
    };
    push(
        "mean_score",
        baseline.mean_score,
        current.summary.mean_score,
        true,
    );
    push(
        "typed_rate",
        baseline.typed_rate,
        current.summary.typed_rate,
        true,
    );
    push(
        "low_score",
        baseline.low_score as f64,
        current.summary.low_score as f64,
        false,
    );
    push(
        "conflicts",
        baseline.conflicts as f64,
        current.summary.conflicts as f64,
        false,
    );
    let passed = deltas.iter().all(|delta| delta.passed);
    QualityVerdict {
        passed,
        digest_matches: baseline.digest == current.digest,
        deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use crate::PropertyConfig;
    use std::collections::HashMap;

    fn type_system() -> TypeSystem {
        TypeSystem::from_config(&TypeSystemConfig {
            closure_gate: true,
            aliases: vec![TypeAliasConfig {
                alias: "Person".to_string(),
                canonical: "人物".to_string(),
            }],
            subclasses: Vec::new(),
            disjoint: vec![TypeDisjointConfig {
                left: "人物".to_string(),
                right: "地点".to_string(),
            }],
        })
    }

    fn entity_config() -> Vec<EntityTypeConfig> {
        vec![EntityTypeConfig {
            label: "Person".to_string(),
            properties: vec![
                PropertyConfig {
                    name: "name".to_string(),
                    data_type: "string".to_string(),
                    required: true,
                    extraction_patterns: Vec::new(),
                },
                PropertyConfig {
                    name: "age".to_string(),
                    data_type: "integer".to_string(),
                    required: false,
                    extraction_patterns: Vec::new(),
                },
            ],
            examples: vec!["人物信息".to_string()],
        }]
    }

    fn entity(name: &str, entity_type: &str, properties: &[(&str, &str)]) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            properties: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
        }
    }

    fn relation(from: &str, rel_type: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: rel_type.to_string(),
            to: to.to_string(),
        }
    }

    fn registry_with_master() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        let mut master = PredicateProfileConfig {
            name: "师傅".to_string(),
            domain: vec!["人物".to_string()],
            range: vec!["人物".to_string()],
            ..Default::default()
        };
        master.functional = true;
        registry.declare(&master);
        registry
    }

    #[test]
    fn completeness_uses_declared_required_properties() {
        let registry = registry_with_master();
        let facts = vec![
            entity("张三", "Person", &[("name", "张三"), ("age", "30")]),
            entity("李四", "Person", &[("name", "李四")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let zhangsan = report.passport_for("张三").unwrap();
        assert_eq!(zhangsan.types, vec!["人物".to_string()]);
        assert!((zhangsan.completeness - 1.0).abs() < 1e-9);
        assert!(zhangsan
            .gaps
            .iter()
            .all(|gap| !gap.contains("缺少必备属性")));

        let lisi = report.passport_for("李四").unwrap();
        assert!(
            (lisi.completeness - 1.0).abs() < 1e-9,
            "name 必备属性已提供"
        );
        assert!(lisi.types.contains(&"人物".to_string()));
        assert!(
            lisi.type_sources
                .get("人物")
                .map(|sources| sources.iter().any(|source| source.contains("值域推断")))
                .unwrap_or(false),
            "李四的类型应来自值域推断：{:?}",
            lisi.type_sources
        );
    }

    #[test]
    fn untyped_entity_gets_neutral_score_and_clarify_gap() {
        let registry = registry_with_master();
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            entity("王五", "Entity", &[]), // 已落 type（未声明必备属性）→ 完整性满分
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "认识".to_string(),
                to: "张三".to_string(),
            },
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let wangwu = report.passport_for("王五").unwrap();
        assert_eq!(wangwu.types, vec!["entity".to_string()]);
        assert!((wangwu.completeness - 1.0).abs() < 1e-9);
        assert!(
            !report
                .summary
                .clarify_candidates
                .contains(&"王五".to_string()),
            "已落型实体不追问"
        );

        // 域推断补型：赵六 是 师傅 的主语，由定义域「人物」补型
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("赵六", "师傅", "张三"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let zhaoliu = report.passport_for("赵六").unwrap();
        assert!(zhaoliu.types.contains(&"人物".to_string()), "域推断已补型");
        assert!(
            (zhaoliu.completeness - 0.0).abs() < 1e-9,
            "补型后走必备属性口径：缺少 name 记 0"
        );
        assert!(
            zhaoliu
                .gaps
                .iter()
                .any(|gap| gap.contains("缺少必备属性「name」")),
            "{:?}",
            zhaoliu.gaps
        );

        // 无域/值域声明的谓词不补型：真正未落型 → 中性先验 + 澄清候选
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("神秘人", "认识", "张三"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let mystery = report.passport_for("神秘人").unwrap();
        assert!(mystery.types.is_empty());
        assert!((mystery.completeness - NEUTRAL_COMPLETENESS).abs() < 1e-9);
        assert!(mystery.gaps.iter().any(|gap| gap.contains("未落型")));
        assert!(report
            .summary
            .clarify_candidates
            .contains(&"神秘人".to_string()));
    }

    #[test]
    fn conflict_notes_lower_consistency_and_surface_gaps() {
        let mut registry = registry_with_master();
        registry.observe("张三", "师傅", "李四", Some("知识图谱"));
        registry.note_conflict("张三", "师傅", "王五");
        assert_eq!(registry.conflict_count(), 1);

        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            entity("李四", "Person", &[("name", "李四")]),
            entity("王五", "Person", &[("name", "王五")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let zhangsan = report.passport_for("张三").unwrap();
        assert!((zhangsan.consistency - 0.5).abs() < 1e-9);
        assert!(
            zhangsan
                .gaps
                .iter()
                .any(|gap| gap.contains("未裁决冲突") && gap.contains("王五")),
            "{:?}",
            zhangsan.gaps
        );
        assert_eq!(report.summary.conflicts, 1);
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let registry = registry_with_master();
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            entity("李四", "Person", &[("name", "李四")]),
            relation("张三", "师傅", "李四"),
        ];
        let first = build_report(&facts, &registry, &type_system(), &entity_config());
        let second = build_report(&facts, &registry, &type_system(), &entity_config());
        assert_eq!(first.digest, second.digest);
        assert_eq!(first, second);

        let mut changed = facts.clone();
        changed.push(relation("李四", "师傅", "张三"));
        let third = build_report(&changed, &registry, &type_system(), &entity_config());
        assert_ne!(first.digest, third.digest);
    }

    #[test]
    fn low_score_entities_remain_in_report_and_render() {
        let mut registry = registry_with_master();
        registry.note_conflict("神秘人", "阵营", "混沌");
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("神秘人", "认识", "张三"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        assert!(report.entities.len() >= 2, "低分实体不得被删除/隐藏");
        let mystery = report.passport_for("神秘人").unwrap();
        assert!(
            mystery.score < LOW_SCORE_THRESHOLD,
            "未落型 + 冲突应低于低分线：{}",
            mystery.score
        );
        assert!(report.summary.low_score >= 1);
        let text = report.render_text();
        assert!(text.contains("神秘人"), "{}", text);
        assert!(text.contains("低分≠假"), "{}", text);
        assert!(report.passport_for("神秘人").is_some());
    }

    #[test]
    fn ratchet_detects_regression_and_accepts_improvement() {
        let registry = registry_with_master();
        let facts = vec![entity("张三", "Person", &[("name", "张三")])];
        let base_report = build_report(&facts, &registry, &type_system(), &entity_config());
        let baseline = QualityBaseline::from_report(&base_report);
        let verdict = compare_baseline(&baseline, &base_report, RATCHET_EPSILON);
        assert!(verdict.passed);
        assert!(verdict.digest_matches);

        let mut worse = base_report.clone();
        worse.summary.mean_score -= 0.1;
        let verdict = compare_baseline(&baseline, &worse, RATCHET_EPSILON);
        assert!(!verdict.passed);
        assert!(verdict
            .deltas
            .iter()
            .any(|delta| delta.metric == "mean_score" && !delta.passed));
    }

    #[test]
    fn warrant_separates_truth_from_citation_and_conflict() {
        let mut registry = registry_with_master();
        registry.observe("张三", "师傅", "李四", Some("知识图谱"));
        registry.note_conflict("张三", "师傅", "王五");
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            entity("李四", "Person", &[("name", "李四")]),
            entity("王五", "Person", &[("name", "王五")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let index = QualityIndex::from_report(&report, &facts, &registry).with_revision(7);
        let warrant = index.warrant_for("张三", Some("师傅")).unwrap();
        assert_eq!(warrant.revision, 7);
        assert_eq!(warrant.known_values, vec!["李四".to_string()]);
        assert!(warrant.typed);
        assert!(warrant.has_source());
        assert!(
            warrant.conflict_note().unwrap().contains("王五"),
            "冲突必须同时披露 incoming：{:?}",
            warrant.conflicts
        );
        assert!(warrant.citation_note().contains("出处"));
        assert_eq!(warrant.actionable_gap(), None, "冲突不应被当成缺属性追问");

        assert_eq!(index.warrant_for("不存在", Some("师傅")), None);
        assert_eq!(
            index
                .warrant_for("张三", Some("未知属性"))
                .unwrap()
                .known_values,
            Vec::<String>::new()
        );
    }

    #[test]
    fn warrant_source_is_per_predicate_not_entity_union() {
        let mut registry = registry_with_master();
        registry.observe("张三", "师傅", "李四", Some("知识图谱"));
        let facts = vec![
            entity("张三", "Person", &[("name", "张三"), ("age", "30")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let index = QualityIndex::from_report(&report, &facts, &registry);

        let sourced = index.warrant_for("张三", Some("师傅")).unwrap();
        assert!(sourced.has_source());
        assert!(
            sourced.source_tags.iter().any(|tag| tag == "知识图谱"),
            "{:?}",
            sourced.source_tags
        );

        let unsourced = index.warrant_for("张三", Some("age")).unwrap();
        assert!(
            !unsourced.has_source(),
            "age 没有来源，不能借用师傅的来源：{:?}",
            unsourced.source_tags
        );
        assert!(unsourced.citation_note().contains("未登记来源"));

        // 身份问句问的是整个实体，可以用实体级来源并集。
        let identity = index.warrant_for("张三", None).unwrap();
        assert!(identity.has_source());
    }

    #[test]
    fn warrant_alias_resolves_to_same_predicate() {
        let mut registry = registry_with_master();
        registry.register_alias("师父", "师傅", "人工确认");
        registry.observe("张三", "师傅", "李四", Some("知识图谱"));
        registry.note_conflict("张三", "师傅", "王五");
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let index = QualityIndex::from_report(&report, &facts, &registry);
        let warrant = index.warrant_for("张三", Some("师父")).unwrap();
        assert_eq!(warrant.attribute.as_deref(), Some("师傅"));
        assert_eq!(warrant.known_values, vec!["李四".to_string()]);
        assert_eq!(warrant.conflicts.len(), 1, "别名问句也必须披露同一冲突");
    }

    #[test]
    fn index_digest_and_order_are_deterministic() {
        let registry = registry_with_master();
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("张三", "师傅", "李四"),
        ];
        let report = build_report(&facts, &registry, &type_system(), &entity_config());
        let first = QualityIndex::from_report(&report, &facts, &registry);
        let second = QualityIndex::from_report(&report, &facts, &registry);
        assert_eq!(first, second);
        assert_eq!(first.digest, report.digest);
        assert_eq!(first.len(), report.entities.len());
    }
}

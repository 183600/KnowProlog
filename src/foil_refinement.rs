//! 反例采样驱动的规则精化（Foil-Guided Rule Refinement）。
//!
//! 链接预测、律典与推测台账都从正面计数出发：支持度、置信度、覆盖率。
//! 它们回答「规则推对过多少」，却不回答「规则差点误推了谁」。本模块补上
//! 这一层，并坚持开放世界约定：
//!
//! - **反例探针（foil）不是负例**：同一主语/谓词下的另一个取值只是「近邻
//!   替代」，未登录不等于假。只有显式否认、功能谓词冲突或其它已声明约束
//!   才能把 foil 升级为反例；
//! - **判别充分性**：不仅问「正例有没有支持」，还问「在近邻替代中，正例
//!   是否是唯一有支持的读法」。结论固定为五档：
//!   `Assert` / `Additive` / `Conflict` / `Refuted` / `Unknown`；
//! - **硬反例驱动精化**：用户否认或功能冲突可以编译成规则守卫
//!   （`ExclusiveValue` / `ExceptSubject` / `ExceptObject` / `ExceptInstance`），
//!   在保留全部已知正例的前提下阻断反例；没有可分离守卫时明确报告
//!   `Unrefinable`，绝不悄悄把规则删掉；
//! - **确定性**：候选枚举、排序、守卫选择与报告输出均按稳定键排序，
//!   同样输入得到同样结果；
//! - **只读**：本模块不写知识图谱。守卫与反例只存在于 `FoilLedger`，
//!   显式保存时才落盘。

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::link_prediction::{HornRule, LinkPredictor, Prediction, RuleTemplate, Triple};

pub const FOIL_DIALECT: &str = "knowprolog-foil/1";
pub const DEFAULT_MAX_FOILS_PER_POSITIVE: usize = 6;
pub const DEFAULT_MAX_RULE_APPLICATIONS: usize = 512;
pub const DEFAULT_MAX_GUARDS_PER_RULE: usize = 8;
pub const DEFAULT_MAX_BACKTRACK_OBJECTS: usize = 64;

// ---------------------------------------------------------------------------
// 五值对比结论
// ---------------------------------------------------------------------------

/// 一条命题在近邻替代中的判别结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoilVerdict {
    /// 有支持，且近邻替代中没有已知取值或已驳替代。
    Assert,
    /// 有支持，同一主语/谓词下还有其它已知取值（多值关系）。
    Additive,
    /// 有支持，但与功能谓词约束或显式否认相冲突。
    Conflict,
    /// 没有支持，且被约束或显式否认排除。
    Refuted,
    /// 没有支持，也没有足够证据排除；开放世界下保持未知。
    Unknown,
}

impl FoilVerdict {
    pub fn label(self) -> &'static str {
        match self {
            FoilVerdict::Assert => "可断言",
            FoilVerdict::Additive => "可增补",
            FoilVerdict::Conflict => "冲突",
            FoilVerdict::Refuted => "已反驳",
            FoilVerdict::Unknown => "未知",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            FoilVerdict::Assert => "assert",
            FoilVerdict::Additive => "additive",
            FoilVerdict::Conflict => "conflict",
            FoilVerdict::Refuted => "refuted",
            FoilVerdict::Unknown => "unknown",
        }
    }

    /// 该结论是否允许把命题当作确定事实输出。
    pub fn asserts(self) -> bool {
        matches!(self, FoilVerdict::Assert | FoilVerdict::Additive)
    }
}

/// 近邻替代的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoilKind {
    /// 同一主语/谓词下已有的另一个取值。
    KnownSibling,
    /// 规则枚举出的同主语/谓词候选（尚未登录）。
    RulePrediction,
    /// 通过共享邻居找到的近邻候选（尚未登录）。
    SharedNeighbor,
}

impl FoilKind {
    pub fn label(self) -> &'static str {
        match self {
            FoilKind::KnownSibling => "已知同槽取值",
            FoilKind::RulePrediction => "规则近邻候选",
            FoilKind::SharedNeighbor => "共享邻居候选",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            FoilKind::KnownSibling => "known_sibling",
            FoilKind::RulePrediction => "rule_prediction",
            FoilKind::SharedNeighbor => "shared_neighbor",
        }
    }
}

/// 一个近邻替代在当前证据下的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoilStatus {
    /// 图谱中已有该事实：它是真正的替代读法，不是噪声。
    Supported,
    /// 被用户显式否认：硬反例。
    Refuted,
    /// 同一事实既被记录又被否认：证据冲突。
    Conflict,
    /// 功能谓词下与已知取值互斥：约束探针排除。
    Excluded,
    /// 开放世界中的未知值。
    Unknown,
}

impl FoilStatus {
    pub fn label(self) -> &'static str {
        match self {
            FoilStatus::Supported => "有支持",
            FoilStatus::Refuted => "已反驳",
            FoilStatus::Conflict => "证据冲突",
            FoilStatus::Excluded => "约束排除",
            FoilStatus::Unknown => "未知",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            FoilStatus::Supported => "supported",
            FoilStatus::Refuted => "refuted",
            FoilStatus::Conflict => "conflict",
            FoilStatus::Excluded => "excluded",
            FoilStatus::Unknown => "unknown",
        }
    }

    fn priority(self) -> u8 {
        match self {
            FoilStatus::Conflict => 0,
            FoilStatus::Refuted => 1,
            FoilStatus::Supported => 2,
            FoilStatus::Excluded => 3,
            FoilStatus::Unknown => 4,
        }
    }
}

/// 一个近邻替代及其证据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Foil {
    pub positive: Triple,
    pub foil: Triple,
    pub kind: FoilKind,
    pub status: FoilStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_index: Option<usize>,
    pub detail: String,
}

impl Foil {
    pub fn render_text(&self) -> String {
        let rule = self
            .rule_index
            .map(|index| format!("（规则 #{}）", index))
            .unwrap_or_default();
        format!(
            "{} {} -[{}]-> {}{}",
            self.status.label(),
            self.foil.subject,
            self.foil.predicate,
            self.foil.object,
            rule
        )
    }
}

/// 一条正例的完整对比报告。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoilReport {
    pub positive: Triple,
    pub verdict: FoilVerdict,
    pub supported: bool,
    pub refuted: bool,
    pub foils: Vec<Foil>,
    pub note: String,
}

impl FoilReport {
    pub fn known_alternatives(&self) -> usize {
        self.foils
            .iter()
            .filter(|foil| foil.status == FoilStatus::Supported)
            .count()
    }

    pub fn hard_negatives(&self) -> usize {
        self.foils
            .iter()
            .filter(|foil| {
                matches!(
                    foil.status,
                    FoilStatus::Refuted | FoilStatus::Conflict | FoilStatus::Excluded
                )
            })
            .count()
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "「{}」的「{}」是「{}」：{}（{}）\n",
            self.positive.subject,
            self.positive.predicate,
            self.positive.object,
            self.verdict.label(),
            self.verdict.tag()
        );
        output.push_str(&format!(
            "  支持：{}；近邻替代 {} 个（硬反例 {} 个）\n",
            if self.supported { "有" } else { "无" },
            self.foils.len(),
            self.hard_negatives()
        ));
        for foil in &self.foils {
            output.push_str(&format!("    - {}\n", foil.render_text()));
            if !foil.detail.trim().is_empty() {
                output.push_str(&format!("      {}\n", foil.detail.trim()));
            }
        }
        output.push_str(&format!("  {}\n", self.note));
        output
    }
}

// ---------------------------------------------------------------------------
// 采样
// ---------------------------------------------------------------------------

/// 采样预算与开关。所有上限都是条目数，不依赖挂钟。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoilOptions {
    #[serde(default = "default_max_foils_per_positive")]
    pub max_foils_per_positive: usize,
    #[serde(default = "default_max_predictions")]
    pub max_predictions: usize,
    #[serde(default = "default_max_backtrack_objects")]
    pub max_backtrack_objects: usize,
}

fn default_max_foils_per_positive() -> usize {
    DEFAULT_MAX_FOILS_PER_POSITIVE
}

fn default_max_predictions() -> usize {
    DEFAULT_MAX_RULE_APPLICATIONS
}

fn default_max_backtrack_objects() -> usize {
    DEFAULT_MAX_BACKTRACK_OBJECTS
}

impl Default for FoilOptions {
    fn default() -> Self {
        Self {
            max_foils_per_positive: DEFAULT_MAX_FOILS_PER_POSITIVE,
            max_predictions: DEFAULT_MAX_RULE_APPLICATIONS,
            max_backtrack_objects: DEFAULT_MAX_BACKTRACK_OBJECTS,
        }
    }
}

/// 基于当前图谱与会话账本的正例/近邻采样器。
pub struct FoilSampler<'a> {
    predictor: &'a LinkPredictor,
    ledger: &'a FoilLedger,
    options: FoilOptions,
}

impl<'a> FoilSampler<'a> {
    pub fn new(predictor: &'a LinkPredictor, ledger: &'a FoilLedger) -> Self {
        Self {
            predictor,
            ledger,
            options: FoilOptions::default(),
        }
    }

    pub fn with_options(mut self, options: FoilOptions) -> Self {
        self.options.max_foils_per_positive = options.max_foils_per_positive.max(1);
        self.options.max_predictions = options.max_predictions.max(1);
        self.options.max_backtrack_objects = options.max_backtrack_objects.max(1);
        self
    }

    pub fn options(&self) -> &FoilOptions {
        &self.options
    }

    /// 为每条正例采样近邻替代，并按固定顺序返回报告。
    pub fn sample(&self, positives: &[Triple]) -> Vec<FoilReport> {
        let mut reports: Vec<FoilReport> = positives
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|positive| self.sample_one(&positive))
            .collect();
        reports.sort_by(|left, right| left.positive.cmp(&right.positive));
        reports
    }

    /// 为一条正例采样。算法分三路：
    ///
    /// 1. 同槽已知取值（硬事实，用于判别 Additive / Conflict）；
    /// 2. 链接预测规则枚举出的同槽候选（困难近邻，未登录）；
    /// 3. 与已知取值共享邻居、但尚未登录的对象（兜底困难近邻）。
    ///
    /// 所有候选按 `(status, kind, triple)` 稳定排序后截断。
    pub fn sample_one(&self, positive: &Triple) -> FoilReport {
        let supported =
            self.predictor
                .has_triple(&positive.subject, &positive.predicate, &positive.object);
        let refuted = self.ledger.is_refuted(positive);
        let functional = self.ledger.is_functional(&positive.predicate);
        let mut candidates: BTreeMap<Triple, (FoilKind, Option<usize>, String)> = BTreeMap::new();

        for object in self
            .predictor
            .objects_of(&positive.subject, &positive.predicate)
        {
            if object == positive.object {
                continue;
            }
            let foil = Triple::new(&positive.subject, &positive.predicate, &object);
            let detail = if self
                .predictor
                .has_triple(&foil.subject, &foil.predicate, &foil.object)
            {
                "图谱中已有的同槽取值。".to_string()
            } else {
                String::new()
            };
            candidates
                .entry(foil)
                .or_insert((FoilKind::KnownSibling, None, detail));
        }

        for triple in self.ledger.refutations() {
            if triple.subject == positive.subject
                && triple.predicate == positive.predicate
                && triple.object != positive.object
            {
                candidates.entry(triple.clone()).or_insert((
                    FoilKind::KnownSibling,
                    None,
                    "该取值已被显式否认。".to_string(),
                ));
            }
        }

        let predictions = self.predictor.predict_for_entity(
            &positive.subject,
            Some(&positive.predicate),
            self.options.max_predictions,
        );
        for prediction in predictions {
            let triple = &prediction.triple;
            if triple.subject != positive.subject
                || triple.predicate != positive.predicate
                || triple.object == positive.object
            {
                continue;
            }
            let detail = self
                .predictor
                .rules()
                .get(prediction.rule_index)
                .map(|rule| format!("规则 {} 枚举。", rule.rule_text()))
                .unwrap_or_else(|| "链接预测枚举。".to_string());
            candidates.entry(triple.clone()).or_insert((
                FoilKind::RulePrediction,
                Some(prediction.rule_index),
                detail,
            ));
        }

        for object in self.shared_neighbor_objects(positive) {
            if object == positive.object {
                continue;
            }
            let foil = Triple::new(&positive.subject, &positive.predicate, &object);
            if self
                .predictor
                .has_triple(&foil.subject, &foil.predicate, &foil.object)
            {
                continue;
            }
            let detail = self
                .shared_neighbor_detail(positive, &object)
                .unwrap_or_else(|| "与已知取值共享邻居。".to_string());
            candidates
                .entry(foil)
                .or_insert((FoilKind::SharedNeighbor, None, detail));
        }

        let mut foils: Vec<Foil> = candidates
            .into_iter()
            .map(|(foil, (kind, rule_index, detail))| {
                self.build_foil(positive, foil, kind, rule_index, detail)
            })
            .collect();
        foils.sort_by(|left, right| {
            left.status
                .priority()
                .cmp(&right.status.priority())
                .then_with(|| left.kind.cmp(&right.kind))
                .then_with(|| left.foil.cmp(&right.foil))
        });
        foils.truncate(self.options.max_foils_per_positive);

        let known_alternative = foils
            .iter()
            .any(|foil| foil.status == FoilStatus::Supported);
        let verdict = Self::verdict(supported, refuted, functional, known_alternative);
        let note = Self::note(verdict, functional, &foils);
        FoilReport {
            positive: positive.clone(),
            verdict,
            supported,
            refuted,
            foils,
            note,
        }
    }

    fn build_foil(
        &self,
        positive: &Triple,
        foil: Triple,
        kind: FoilKind,
        rule_index: Option<usize>,
        detail: String,
    ) -> Foil {
        let known = self
            .predictor
            .has_triple(&foil.subject, &foil.predicate, &foil.object);
        let refuted = self.ledger.is_refuted(&foil);
        let functional = self.ledger.is_functional(&foil.predicate);
        let has_other_value = self
            .predictor
            .objects_of(&foil.subject, &foil.predicate)
            .into_iter()
            .any(|object| object != foil.object)
            || self
                .ledger
                .objects_of(&foil.subject, &foil.predicate)
                .into_iter()
                .any(|object| object != foil.object);
        let status = if known && refuted {
            FoilStatus::Conflict
        } else if refuted {
            FoilStatus::Refuted
        } else if known {
            FoilStatus::Supported
        } else if functional && has_other_value {
            FoilStatus::Excluded
        } else {
            FoilStatus::Unknown
        };
        let mut detail = detail;
        if status == FoilStatus::Excluded {
            if !detail.is_empty() {
                detail.push(' ');
            }
            detail.push_str(&format!(
                "「{}」是功能谓词，且「{}」已有其它取值。",
                foil.predicate, foil.subject
            ));
        }
        Foil {
            positive: positive.clone(),
            foil,
            kind,
            status,
            rule_index,
            detail,
        }
    }

    fn verdict(
        supported: bool,
        refuted: bool,
        functional: bool,
        known_alternative: bool,
    ) -> FoilVerdict {
        match (supported, refuted) {
            (true, true) => FoilVerdict::Conflict,
            (false, true) => FoilVerdict::Refuted,
            (true, false) if functional && known_alternative => FoilVerdict::Conflict,
            (false, false) if functional && known_alternative => FoilVerdict::Refuted,
            (true, false) if known_alternative => FoilVerdict::Additive,
            (true, false) => FoilVerdict::Assert,
            (false, false) => FoilVerdict::Unknown,
        }
    }

    fn note(verdict: FoilVerdict, functional: bool, foils: &[Foil]) -> String {
        let refuted = foils
            .iter()
            .filter(|foil| foil.status == FoilStatus::Refuted)
            .count();
        let excluded = foils
            .iter()
            .filter(|foil| foil.status == FoilStatus::Excluded)
            .count();
        let unknown = foils
            .iter()
            .filter(|foil| foil.status == FoilStatus::Unknown)
            .count();
        let mut parts = vec![match verdict {
            FoilVerdict::Assert => "正例有支持，近邻替代中没有已登录取值。".to_string(),
            FoilVerdict::Additive => "正例有支持，同槽还有其它已登录取值（多值关系）。".to_string(),
            FoilVerdict::Conflict => {
                if functional {
                    "功能谓词上正例与其它取值冲突，需先裁决再输出。".to_string()
                } else {
                    "同一事实同时存在支持与否认，证据冲突。".to_string()
                }
            }
            FoilVerdict::Refuted => {
                if functional {
                    "功能谓词约束排除了该取值。".to_string()
                } else {
                    "该命题已被显式否认。".to_string()
                }
            }
            FoilVerdict::Unknown => {
                "没有支持证据；未登录的近邻替代不构成否定（开放世界）。".to_string()
            }
        }];
        if refuted > 0 {
            parts.push(format!("其中 {} 个近邻被显式否认。", refuted));
        }
        if excluded > 0 {
            parts.push(format!("其中 {} 个近邻被功能约束排除。", excluded));
        }
        if unknown > 0 {
            parts.push(format!(
                "另有 {} 个未登录近邻仅作困难候选，不自动升级为反例。",
                unknown
            ));
        }
        parts.join(" ")
    }

    fn shared_neighbor_objects(&self, positive: &Triple) -> Vec<String> {
        let mut objects: BTreeSet<String> = BTreeSet::new();
        for triple in self.predictor.triples() {
            if triple.predicate == positive.predicate && triple.object != positive.object {
                objects.insert(triple.object.clone());
            }
        }
        objects
            .into_iter()
            .take(self.options.max_backtrack_objects)
            .filter(|object| self.shares_neighbor(&positive.object, object, &positive.predicate))
            .collect()
    }

    fn shares_neighbor(&self, left: &str, right: &str, predicate: &str) -> bool {
        let left_neighbors = self.neighbors(left, predicate);
        if left_neighbors.is_empty() {
            return false;
        }
        !left_neighbors.is_disjoint(&self.neighbors(right, predicate))
    }

    fn neighbors(&self, node: &str, predicate: &str) -> BTreeSet<String> {
        self.predictor
            .triples()
            .iter()
            .filter(|triple| triple.predicate == predicate)
            .filter_map(|triple| {
                if triple.subject == node {
                    Some(triple.object.clone())
                } else if triple.object == node {
                    Some(triple.subject.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn shared_neighbor_detail(&self, positive: &Triple, object: &str) -> Option<String> {
        let left = self.neighbors(&positive.object, &positive.predicate);
        let right = self.neighbors(object, &positive.predicate);
        let shared: Vec<&String> = left.intersection(&right).collect();
        if shared.is_empty() {
            return None;
        }
        Some(format!(
            "与「{}」共享 {} 个邻居：{}。",
            positive.object,
            shared.len(),
            shared
                .into_iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("、")
        ))
    }
}

// ---------------------------------------------------------------------------
// 规则守卫与精化
// ---------------------------------------------------------------------------

/// 一条规则守卫：命中即阻断该规则的这次应用。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleGuard {
    /// 功能谓词上主语已有其它取值时阻断。
    ExclusiveValue { predicate: String },
    /// 只阻断特定主语上的该谓词应用。
    ExceptSubject { predicate: String, subject: String },
    /// 只阻断特定客体上的该谓词应用。
    ExceptObject { predicate: String, object: String },
    /// 只阻断一个确切的三元组（案例例外）。
    ExceptInstance {
        subject: String,
        predicate: String,
        object: String,
    },
}

impl RuleGuard {
    pub fn label(&self) -> &'static str {
        match self {
            RuleGuard::ExclusiveValue { .. } => "功能互斥",
            RuleGuard::ExceptSubject { .. } => "主语例外",
            RuleGuard::ExceptObject { .. } => "客体例外",
            RuleGuard::ExceptInstance { .. } => "实例例外",
        }
    }

    pub fn render(&self) -> String {
        match self {
            RuleGuard::ExclusiveValue { predicate } => {
                format!("若主语已有其它「{}」取值则阻断", predicate)
            }
            RuleGuard::ExceptSubject { predicate, subject } => {
                format!("阻断「{}」作主语时的「{}」应用", subject, predicate)
            }
            RuleGuard::ExceptObject { predicate, object } => {
                format!("阻断「{}」作客体时的「{}」应用", object, predicate)
            }
            RuleGuard::ExceptInstance {
                subject,
                predicate,
                object,
            } => format!("阻断 {}|{}|{}", subject, predicate, object),
        }
    }

    fn priority(&self) -> u8 {
        match self {
            RuleGuard::ExclusiveValue { .. } => 0,
            RuleGuard::ExceptInstance { .. } => 1,
            RuleGuard::ExceptObject { .. } => 2,
            RuleGuard::ExceptSubject { .. } => 3,
        }
    }

    /// 守卫是否阻断该三元组。功能守卫需要借助图谱/会话中的已知取值。
    pub fn blocks(
        &self,
        triple: &Triple,
        predictor: &LinkPredictor,
        extra_values: &BTreeMap<(String, String), BTreeSet<String>>,
    ) -> bool {
        match self {
            RuleGuard::ExclusiveValue { predicate } => {
                if triple.predicate != *predicate {
                    return false;
                }
                let key = (triple.subject.clone(), triple.predicate.clone());
                let mut values: BTreeSet<String> = predictor
                    .objects_of(&triple.subject, &triple.predicate)
                    .into_iter()
                    .collect();
                if let Some(extra) = extra_values.get(&key) {
                    values.extend(extra.iter().cloned());
                }
                values.iter().any(|value| value != &triple.object)
            }
            RuleGuard::ExceptSubject { predicate, subject } => {
                triple.predicate == *predicate && triple.subject == *subject
            }
            RuleGuard::ExceptObject { predicate, object } => {
                triple.predicate == *predicate && triple.object == *object
            }
            RuleGuard::ExceptInstance {
                subject,
                predicate,
                object,
            } => {
                triple.subject == *subject
                    && triple.predicate == *predicate
                    && triple.object == *object
            }
        }
    }
}

/// 一次规则应用：头三元组与触发它的证据边。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleApplication {
    pub triple: Triple,
    pub evidence: Vec<Triple>,
    #[serde(default)]
    pub rule_index: usize,
}

/// 枚举一条规则的全部应用（含已登录正例），按三元组去重并稳定排序。
pub fn rule_applications(
    predictor: &LinkPredictor,
    rule_index: usize,
    cap: usize,
) -> Vec<RuleApplication> {
    let Some(rule) = predictor.rules().get(rule_index) else {
        return Vec::new();
    };
    let cap = cap.max(1);
    let mut triples: Vec<&Triple> = predictor.triples().iter().collect();
    triples.sort();
    let mut by_predicate: BTreeMap<&str, Vec<&Triple>> = BTreeMap::new();
    let mut by_subject: BTreeMap<(&str, &str), Vec<&Triple>> = BTreeMap::new();
    let mut by_object: BTreeMap<(&str, &str), Vec<&Triple>> = BTreeMap::new();
    for triple in &triples {
        by_predicate
            .entry(triple.predicate.as_str())
            .or_default()
            .push(triple);
        by_subject
            .entry((triple.predicate.as_str(), triple.subject.as_str()))
            .or_default()
            .push(triple);
        by_object
            .entry((triple.predicate.as_str(), triple.object.as_str()))
            .or_default()
            .push(triple);
    }

    let mut applications: BTreeMap<Triple, Vec<Triple>> = BTreeMap::new();
    match rule.template {
        RuleTemplate::Inverse => {
            let Some(edges) = by_predicate.get(rule.body1()) else {
                return Vec::new();
            };
            for edge in edges {
                let triple = Triple::new(&edge.object, &rule.head, &edge.subject);
                applications
                    .entry(triple)
                    .or_insert_with(|| vec![(*edge).clone()]);
            }
        }
        RuleTemplate::Compose => {
            let Some(firsts) = by_predicate.get(rule.body1()) else {
                return Vec::new();
            };
            let Some(seconds) = by_predicate.get(rule.body2().unwrap_or("")) else {
                return Vec::new();
            };
            let mut second_index: BTreeMap<&str, Vec<&Triple>> = BTreeMap::new();
            for edge in seconds {
                second_index
                    .entry(edge.subject.as_str())
                    .or_default()
                    .push(edge);
            }
            for first in firsts {
                let Some(matches) = second_index.get(first.object.as_str()) else {
                    continue;
                };
                for second in matches {
                    if first.subject == second.object {
                        continue;
                    }
                    let triple = Triple::new(&first.subject, &rule.head, &second.object);
                    applications
                        .entry(triple)
                        .or_insert_with(|| vec![(*first).clone(), (*second).clone()]);
                }
            }
        }
        RuleTemplate::CommonNeighbor => {
            let Some(firsts) = by_predicate.get(rule.body1()) else {
                return Vec::new();
            };
            if !by_predicate.contains_key(rule.body2().unwrap_or("")) {
                return Vec::new();
            }
            for first in firsts {
                let Some(matches) =
                    by_object.get(&(rule.body2().unwrap_or(""), first.object.as_str()))
                else {
                    continue;
                };
                for second in matches {
                    if second.subject == first.subject {
                        continue;
                    }
                    let triple = Triple::new(&first.subject, &rule.head, &second.subject);
                    applications
                        .entry(triple)
                        .or_insert_with(|| vec![(*first).clone(), (*second).clone()]);
                }
            }
        }
        RuleTemplate::CommonObject => {
            let Some(firsts) = by_predicate.get(rule.body1()) else {
                return Vec::new();
            };
            if !by_predicate.contains_key(rule.body2().unwrap_or("")) {
                return Vec::new();
            }
            for first in firsts {
                let Some(matches) =
                    by_subject.get(&(rule.body2().unwrap_or(""), first.subject.as_str()))
                else {
                    continue;
                };
                for second in matches {
                    if second.object == first.object {
                        continue;
                    }
                    let triple = Triple::new(&first.object, &rule.head, &second.object);
                    applications
                        .entry(triple)
                        .or_insert_with(|| vec![(*first).clone(), (*second).clone()]);
                }
            }
        }
    }

    applications
        .into_iter()
        .take(cap)
        .map(|(triple, evidence)| RuleApplication {
            triple,
            evidence,
            rule_index,
        })
        .collect()
}

/// 精化状态：不把「暂时分不开」伪装成成功。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementStatus {
    /// 所有可推导硬反例都被守卫阻断，且已知正例全部保留。
    Refined,
    /// 有守卫，但仍有硬反例未被阻断。
    Partial,
    /// 没有可推导的硬反例。
    NoHardNegatives,
    /// 规则在图中没有任何应用。
    NoApplications,
    /// 找不到不牺牲正例的守卫。
    Unrefinable,
}

impl RefinementStatus {
    pub fn label(self) -> &'static str {
        match self {
            RefinementStatus::Refined => "已精化",
            RefinementStatus::Partial => "部分精化",
            RefinementStatus::NoHardNegatives => "无硬反例",
            RefinementStatus::NoApplications => "无应用",
            RefinementStatus::Unrefinable => "不可精化",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            RefinementStatus::Refined => "refined",
            RefinementStatus::Partial => "partial",
            RefinementStatus::NoHardNegatives => "no_hard_negatives",
            RefinementStatus::NoApplications => "no_applications",
            RefinementStatus::Unrefinable => "unrefinable",
        }
    }
}

/// 守卫候选的评估：阻断多少硬反例、牺牲多少已知正例。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardEvaluation {
    pub guard: RuleGuard,
    pub blocked_negatives: usize,
    pub dropped_positives: usize,
    pub retained_positives: usize,
}

/// 一条规则的硬反例精化结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleRefinement {
    pub rule_index: usize,
    pub rule_text: String,
    pub hard_negatives: Vec<Triple>,
    pub unmatched_negatives: Vec<Triple>,
    pub positives: Vec<Triple>,
    pub evaluations: Vec<GuardEvaluation>,
    pub guards: Vec<RuleGuard>,
    pub blocked: usize,
    pub status: RefinementStatus,
    pub note: String,
}

impl RuleRefinement {
    pub fn blocked_negatives(&self) -> usize {
        self.blocked
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "规则 #{} {}：{}（正例 {}，硬反例 {}，已阻断 {}）\n",
            self.rule_index,
            self.rule_text,
            self.status.label(),
            self.positives.len(),
            self.hard_negatives.len(),
            self.blocked_negatives()
        );
        for guard in &self.guards {
            output.push_str(&format!("  守卫：{}\n", guard.render()));
        }
        for evaluation in &self.evaluations {
            output.push_str(&format!(
                "  候选：{}（阻断 {}，保留 {}，牺牲 {}）\n",
                evaluation.guard.render(),
                evaluation.blocked_negatives,
                evaluation.retained_positives,
                evaluation.dropped_positives
            ));
        }
        if !self.unmatched_negatives.is_empty() {
            output.push_str(&format!(
                "  不可由该规则导出的硬反例：{}\n",
                self.unmatched_negatives
                    .iter()
                    .map(|triple| format!(
                        "{}|{}|{}",
                        triple.subject, triple.predicate, triple.object
                    ))
                    .collect::<Vec<_>>()
                    .join("；")
            ));
        }
        output.push_str(&format!("  {}\n", self.note));
        output
    }
}

/// 精化选项。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefineOptions {
    #[serde(default = "default_max_rule_applications")]
    pub max_applications: usize,
    #[serde(default = "default_max_guards_per_rule")]
    pub max_guards: usize,
}

fn default_max_rule_applications() -> usize {
    DEFAULT_MAX_RULE_APPLICATIONS
}

fn default_max_guards_per_rule() -> usize {
    DEFAULT_MAX_GUARDS_PER_RULE
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            max_applications: DEFAULT_MAX_RULE_APPLICATIONS,
            max_guards: DEFAULT_MAX_GUARDS_PER_RULE,
        }
    }
}

/// 用硬反例精化一条规则，但不修改规则本身：返回可采纳的守卫计划。
pub fn refine_rule(
    predictor: &LinkPredictor,
    rule_index: usize,
    hard_negatives: &[Triple],
    functional: &BTreeSet<String>,
    options: &RefineOptions,
) -> RuleRefinement {
    let Some(rule) = predictor.rules().get(rule_index) else {
        return RuleRefinement {
            rule_index,
            rule_text: String::new(),
            hard_negatives: Vec::new(),
            unmatched_negatives: Vec::new(),
            positives: Vec::new(),
            evaluations: Vec::new(),
            guards: Vec::new(),
            blocked: 0,
            status: RefinementStatus::NoApplications,
            note: "规则下标不存在。".to_string(),
        };
    };
    let rule_text = rule.rule_text();
    let applications = rule_applications(predictor, rule_index, options.max_applications);
    if applications.is_empty() {
        return RuleRefinement {
            rule_index,
            rule_text,
            hard_negatives: Vec::new(),
            unmatched_negatives: hard_negatives.to_vec(),
            positives: Vec::new(),
            evaluations: Vec::new(),
            guards: Vec::new(),
            blocked: 0,
            status: RefinementStatus::NoApplications,
            note: "规则在图中没有可枚举的应用。".to_string(),
        };
    }

    let requested: BTreeSet<Triple> = hard_negatives.iter().cloned().collect();
    let derivable: BTreeSet<Triple> = applications
        .iter()
        .map(|application| application.triple.clone())
        .collect();
    let matched: Vec<Triple> = requested
        .iter()
        .filter(|triple| derivable.contains(*triple))
        .cloned()
        .collect();
    let unmatched: Vec<Triple> = requested
        .iter()
        .filter(|triple| !derivable.contains(*triple))
        .cloned()
        .collect();
    let positives: Vec<Triple> = applications
        .iter()
        .filter(|application| {
            predictor.has_triple(
                &application.triple.subject,
                &application.triple.predicate,
                &application.triple.object,
            )
        })
        .map(|application| application.triple.clone())
        .collect();

    if matched.is_empty() {
        return RuleRefinement {
            rule_index,
            rule_text,
            hard_negatives: Vec::new(),
            unmatched_negatives: unmatched,
            positives,
            evaluations: Vec::new(),
            guards: Vec::new(),
            blocked: 0,
            status: RefinementStatus::NoHardNegatives,
            note: "没有可由该规则导出的硬反例；规则保持不变。".to_string(),
        };
    }

    let mut extras: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for application in &applications {
        if predictor.has_triple(
            &application.triple.subject,
            &application.triple.predicate,
            &application.triple.object,
        ) {
            extras
                .entry((
                    application.triple.subject.clone(),
                    application.triple.predicate.clone(),
                ))
                .or_default()
                .insert(application.triple.object.clone());
        }
    }
    let mut candidates: BTreeSet<RuleGuard> = BTreeSet::new();
    if functional.contains(&rule.head) {
        candidates.insert(RuleGuard::ExclusiveValue {
            predicate: rule.head.clone(),
        });
    }
    for negative in &matched {
        candidates.insert(RuleGuard::ExceptInstance {
            subject: negative.subject.clone(),
            predicate: negative.predicate.clone(),
            object: negative.object.clone(),
        });
        candidates.insert(RuleGuard::ExceptSubject {
            predicate: negative.predicate.clone(),
            subject: negative.subject.clone(),
        });
        candidates.insert(RuleGuard::ExceptObject {
            predicate: negative.predicate.clone(),
            object: negative.object.clone(),
        });
    }

    let mut evaluations: Vec<GuardEvaluation> = candidates
        .into_iter()
        .map(|guard| {
            let blocked_negatives = matched
                .iter()
                .filter(|triple| guard.blocks(triple, predictor, &extras))
                .count();
            let dropped_positives = positives
                .iter()
                .filter(|triple| guard.blocks(triple, predictor, &extras))
                .count();
            GuardEvaluation {
                guard,
                blocked_negatives,
                dropped_positives,
                retained_positives: positives.len().saturating_sub(dropped_positives),
            }
        })
        .collect();
    evaluations.sort_by(|left, right| {
        right
            .blocked_negatives
            .cmp(&left.blocked_negatives)
            .then_with(|| left.dropped_positives.cmp(&right.dropped_positives))
            .then_with(|| left.guard.priority().cmp(&right.guard.priority()))
            .then_with(|| left.guard.render().cmp(&right.guard.render()))
    });
    evaluations.truncate(options.max_guards.max(1));

    let mut selected: Vec<RuleGuard> = Vec::new();
    let mut blocked: BTreeSet<Triple> = BTreeSet::new();
    for evaluation in &evaluations {
        if evaluation.blocked_negatives == 0 || evaluation.dropped_positives > 0 {
            continue;
        }
        let new_blocks = matched
            .iter()
            .filter(|triple| !blocked.contains(*triple))
            .filter(|triple| evaluation.guard.blocks(triple, predictor, &extras))
            .count();
        if new_blocks == 0 {
            continue;
        }
        selected.push(evaluation.guard.clone());
        for triple in &matched {
            if evaluation.guard.blocks(triple, predictor, &extras) {
                blocked.insert(triple.clone());
            }
        }
        if blocked.len() == matched.len() {
            break;
        }
    }

    let status = if selected.is_empty() {
        RefinementStatus::Unrefinable
    } else if blocked.len() == matched.len() {
        RefinementStatus::Refined
    } else {
        RefinementStatus::Partial
    };
    let note = match status {
        RefinementStatus::Refined => format!(
            "已用 {} 个守卫阻断全部 {} 个硬反例，已知正例保留。",
            selected.len(),
            matched.len()
        ),
        RefinementStatus::Partial => format!(
            "已用 {} 个守卫阻断 {}/{} 个硬反例；其余反例仍需更细的谓词/类型条件。",
            selected.len(),
            blocked.len(),
            matched.len()
        ),
        RefinementStatus::Unrefinable => {
            "任何候选守卫都会牺牲已知正例或无法阻断反例；规则保持不变，需人工裁决。".to_string()
        }
        _ => String::new(),
    };

    RuleRefinement {
        rule_index,
        rule_text,
        hard_negatives: matched,
        unmatched_negatives: unmatched,
        positives,
        evaluations,
        guards: selected,
        blocked: blocked.len(),
        status,
        note,
    }
}

// ---------------------------------------------------------------------------
// 会话账本
// ---------------------------------------------------------------------------

/// 可持久化的反例/功能谓词/守卫账本。只记录会话决定，不写图谱。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoilLedger {
    #[serde(default = "default_dialect")]
    pub dialect: String,
    #[serde(default)]
    refuted: BTreeSet<Triple>,
    #[serde(default)]
    functional: BTreeSet<String>,
    #[serde(default)]
    asserted: BTreeSet<Triple>,
    #[serde(default)]
    guards: BTreeMap<String, Vec<RuleGuard>>,
}

fn default_dialect() -> String {
    FOIL_DIALECT.to_string()
}

impl Default for FoilLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl FoilLedger {
    pub fn new() -> Self {
        Self {
            dialect: default_dialect(),
            refuted: BTreeSet::new(),
            functional: BTreeSet::new(),
            asserted: BTreeSet::new(),
            guards: BTreeMap::new(),
        }
    }

    pub fn refute(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.refuted.insert(Triple::new(subject, predicate, object))
    }

    pub fn retract_refutation(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.refuted
            .remove(&Triple::new(subject, predicate, object))
    }

    pub fn assert_observation(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.asserted
            .insert(Triple::new(subject, predicate, object))
    }

    pub fn retract_observation(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.asserted
            .remove(&Triple::new(subject, predicate, object))
    }

    pub fn mark_functional(&mut self, predicate: &str) -> bool {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return false;
        }
        self.functional.insert(predicate.to_string())
    }

    pub fn unmark_functional(&mut self, predicate: &str) -> bool {
        self.functional.remove(predicate.trim())
    }

    pub fn is_refuted(&self, triple: &Triple) -> bool {
        self.refuted.contains(triple)
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.functional.contains(predicate.trim())
    }

    pub fn refutations(&self) -> impl Iterator<Item = &Triple> {
        self.refuted.iter()
    }

    pub fn functional_predicates(&self) -> impl Iterator<Item = &String> {
        self.functional.iter()
    }

    pub fn observations(&self) -> impl Iterator<Item = &Triple> {
        self.asserted.iter()
    }

    pub fn objects_of(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.asserted
            .iter()
            .filter(|triple| triple.subject == subject && triple.predicate == predicate)
            .map(|triple| triple.object.clone())
            .collect()
    }

    pub fn guards_for(&self, rule_key: &str) -> Vec<RuleGuard> {
        self.guards.get(rule_key).cloned().unwrap_or_default()
    }

    pub fn guarded_rule_count(&self) -> usize {
        self.guards
            .values()
            .filter(|guards| !guards.is_empty())
            .count()
    }

    pub fn adopt_refinement(&mut self, refinement: &RuleRefinement) -> bool {
        if refinement.guards.is_empty() {
            return false;
        }
        let key = rule_key_from_text(&refinement.rule_text, refinement.rule_index);
        let entry = self.guards.entry(key).or_default();
        let before = entry.len();
        for guard in &refinement.guards {
            if !entry.contains(guard) {
                entry.push(guard.clone());
            }
        }
        entry.sort();
        entry.dedup();
        entry.len() != before
    }

    pub fn clear_guards(&mut self, rule_key: &str) -> bool {
        self.guards.remove(rule_key).is_some()
    }

    pub fn clear_all_guards(&mut self) -> usize {
        let removed = self.guards.len();
        self.guards.clear();
        removed
    }

    /// 该账本是否会阻断一次预测（按预测携带的规则下标定位守卫）。
    pub fn blocks_prediction(&self, predictor: &LinkPredictor, prediction: &Prediction) -> bool {
        let Some(rule) = predictor.rules().get(prediction.rule_index) else {
            return false;
        };
        let key = rule_key_from_text(&rule.rule_text(), prediction.rule_index);
        let Some(guards) = self.guards.get(&key) else {
            return false;
        };
        let extras = self.value_index();
        guards
            .iter()
            .any(|guard| guard.blocks(&prediction.triple, predictor, &extras))
    }

    /// 过滤预测：只保留未被任何已采纳守卫阻断的候选。
    pub fn filter_predictions<'a>(
        &self,
        predictor: &LinkPredictor,
        predictions: &'a [Prediction],
    ) -> Vec<&'a Prediction> {
        predictions
            .iter()
            .filter(|prediction| !self.blocks_prediction(predictor, prediction))
            .collect()
    }

    fn value_index(&self) -> BTreeMap<(String, String), BTreeSet<String>> {
        let mut index: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
        for triple in &self.asserted {
            index
                .entry((triple.subject.clone(), triple.predicate.clone()))
                .or_default()
                .insert(triple.object.clone());
        }
        index
    }

    pub fn save_to_path(&self, path: &str) -> Result<(), String> {
        let encoded = serde_json::to_string_pretty(self)
            .map_err(|error| format!("序列化反例账本失败：{}", error))?;
        std::fs::write(path, encoded).map_err(|error| format!("写入反例账本失败：{}", error))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let ledger: Self = serde_json::from_str(&content).ok()?;
        if ledger.dialect != FOIL_DIALECT {
            return None;
        }
        Some(ledger)
    }

    pub fn render_report(&self, predictor: &LinkPredictor) -> String {
        let mut output = format!(
            "反例账本：显式否认 {} 条，功能谓词 {} 个，已守卫规则 {} 条。",
            self.refuted.len(),
            self.functional.len(),
            self.guarded_rule_count()
        );
        if !self.refuted.is_empty() {
            output.push_str("\n  显式否认：");
            for triple in &self.refuted {
                output.push_str(&format!(
                    "\n  - {}|{}|{}",
                    triple.subject, triple.predicate, triple.object
                ));
            }
        }
        if !self.functional.is_empty() {
            output.push_str(&format!(
                "\n  功能谓词：{}",
                self.functional
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("、")
            ));
        }
        for (rule_key, guards) in &self.guards {
            output.push_str(&format!("\n  规则 {}：", rule_key));
            for guard in guards {
                output.push_str(&format!("\n    - {}", guard.render()));
            }
        }
        let predictions = predictor.predict_missing(16);
        if !predictions.is_empty() {
            let blocked = predictions
                .iter()
                .filter(|prediction| self.blocks_prediction(predictor, prediction))
                .count();
            output.push_str(&format!(
                "\n  预测复核：抽样 {} 条，守卫阻断 {} 条。",
                predictions.len(),
                blocked
            ));
        }
        output
    }
}

fn rule_key_from_text(rule_text: &str, rule_index: usize) -> String {
    if rule_text.trim().is_empty() {
        format!("rule#{}", rule_index)
    } else {
        rule_text.trim().to_string()
    }
}

/// 便于外部（CLI/会话）复用的规则键。
pub fn rule_key(rule: &HornRule, rule_index: usize) -> String {
    rule_key_from_text(&rule.rule_text(), rule_index)
}

/// 把功能谓词上的规则候选编译成硬反例：若主语已有其它取值，规则推出的
/// 未知候选在功能约束下就是冲突，而不是开放世界中的普通未知。
///
/// 仅枚举已挖出的规则应用，不写图谱；结果去重并稳定排序。
pub fn functional_conflict_negatives(
    predictor: &LinkPredictor,
    ledger: &FoilLedger,
) -> Vec<Triple> {
    let mut out: BTreeSet<Triple> = BTreeSet::new();
    for (index, rule) in predictor.rules().iter().enumerate() {
        if !ledger.is_functional(&rule.head) {
            continue;
        }
        for application in rule_applications(predictor, index, DEFAULT_MAX_RULE_APPLICATIONS) {
            let triple = &application.triple;
            if predictor.has_triple(&triple.subject, &triple.predicate, &triple.object) {
                continue;
            }
            let mut values: BTreeSet<String> = predictor
                .objects_of(&triple.subject, &triple.predicate)
                .into_iter()
                .collect();
            values.extend(ledger.objects_of(&triple.subject, &triple.predicate));
            if values.iter().any(|value| value != &triple.object) {
                out.insert(triple.clone());
            }
        }
    }
    out.into_iter().collect()
}

/// 把一组硬反例合并成去重、稳定排序的集合。
pub fn normalize_negatives(negatives: &[Triple]) -> Vec<Triple> {
    let mut set: HashSet<Triple> = HashSet::new();
    for triple in negatives {
        set.insert(triple.clone());
    }
    let mut out: Vec<Triple> = set.into_iter().collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predictor() -> LinkPredictor {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.add_triple("李四", "师祖", "王五");
        predictor.mine_rules(1, 0.5);
        predictor
    }

    fn ledger() -> FoilLedger {
        FoilLedger::new()
    }

    #[test]
    fn known_sibling_makes_the_reading_additive() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "朋友", "李四");
        predictor.add_triple("张三", "朋友", "王五");
        let ledger = ledger();
        let reports =
            FoilSampler::new(&predictor, &ledger).sample(&[Triple::new("张三", "朋友", "李四")]);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].verdict, FoilVerdict::Additive);
        assert!(reports[0].foils.iter().any(|foil| {
            foil.foil == Triple::new("张三", "朋友", "王五") && foil.status == FoilStatus::Supported
        }));
        assert!(reports[0].render_text().contains("可增补"));
    }

    #[test]
    fn functional_alternative_refutes_an_unknown_candidate() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "身高", "180");
        let mut ledger = ledger();
        ledger.mark_functional("身高");
        let reports =
            FoilSampler::new(&predictor, &ledger).sample(&[Triple::new("张三", "身高", "175")]);
        assert_eq!(reports[0].verdict, FoilVerdict::Refuted);
        assert!(reports[0]
            .foils
            .iter()
            .any(|foil| foil.foil == Triple::new("张三", "身高", "180")
                && foil.status == FoilStatus::Supported));
        assert!(
            reports[0].note.contains("功能谓词约束"),
            "{}",
            reports[0].note
        );
    }

    #[test]
    fn explicit_refutation_is_a_hard_negative_not_an_open_world_gap() {
        let predictor = LinkPredictor::new();
        let mut ledger = ledger();
        assert!(ledger.refute("张三", "师傅", "王五"));
        let reports =
            FoilSampler::new(&predictor, &ledger).sample(&[Triple::new("张三", "师傅", "王五")]);
        assert_eq!(reports[0].verdict, FoilVerdict::Refuted);
        assert!(reports[0].refuted);
        assert!(!reports[0].supported);
        assert!(reports[0].hard_negatives() == 0);
    }

    #[test]
    fn rule_predictions_are_sampled_as_hard_neighbours() {
        let predictor = predictor();
        let ledger = ledger();
        let reports =
            FoilSampler::new(&predictor, &ledger).sample(&[Triple::new("李四", "师祖", "王五")]);
        assert_eq!(reports[0].verdict, FoilVerdict::Assert);
        assert!(reports[0].foils.iter().any(|foil| {
            foil.foil == Triple::new("李四", "师祖", "赵六")
                && foil.kind == FoilKind::RulePrediction
                && foil.status == FoilStatus::Unknown
        }));
    }

    #[test]
    fn refinement_prefers_exclusive_value_guard() {
        let predictor = predictor();
        let functional = BTreeSet::from(["师祖".to_string()]);
        let negatives = [Triple::new("李四", "师祖", "赵六")];
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let refinement = refine_rule(
            &predictor,
            index,
            &negatives,
            &functional,
            &RefineOptions::default(),
        );
        assert_eq!(refinement.status, RefinementStatus::Refined);
        assert!(refinement.guards.iter().any(
            |guard| matches!(guard, RuleGuard::ExclusiveValue { predicate } if predicate == "师祖")
        ));
        assert_eq!(refinement.blocked_negatives(), 1);
        assert!(refinement
            .positives
            .iter()
            .any(|triple| { triple == &Triple::new("张三", "师祖", "王五") }));
    }

    #[test]
    fn instance_exception_when_subject_guard_would_drop_a_positive() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("张三", "师傅", "赵六");
        predictor.add_triple("赵六", "师傅", "孙七");
        predictor.add_triple("张三", "师傅", "吴九");
        predictor.add_triple("吴九", "师傅", "周八");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.add_triple("张三", "师祖", "孙七");
        predictor.mine_rules(1, 0.5);
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let applications = rule_applications(&predictor, index, 64);
        assert!(applications
            .iter()
            .any(|application| application.triple == Triple::new("张三", "师祖", "周八")));
        let negatives = [Triple::new("张三", "师祖", "周八")];
        let refinement = refine_rule(
            &predictor,
            index,
            &negatives,
            &BTreeSet::new(),
            &RefineOptions::default(),
        );
        assert!(refinement
            .guards
            .iter()
            .any(|guard| matches!(guard, RuleGuard::ExceptInstance { .. })));
        assert!(!refinement
            .guards
            .iter()
            .any(|guard| matches!(guard, RuleGuard::ExceptSubject { .. })));
        assert!(refinement.evaluations.iter().any(|evaluation| {
            matches!(evaluation.guard, RuleGuard::ExceptSubject { .. })
                && evaluation.dropped_positives > 0
        }));
    }

    #[test]
    fn functional_conflicts_are_derived_as_hard_negatives() {
        let predictor = predictor();
        let mut ledger = FoilLedger::new();
        ledger.mark_functional("师祖");
        let negatives = functional_conflict_negatives(&predictor, &ledger);
        assert!(negatives.contains(&Triple::new("李四", "师祖", "赵六")));
        assert!(!negatives.contains(&Triple::new("李四", "师祖", "王五")));
        assert!(functional_conflict_negatives(&predictor, &FoilLedger::new()).is_empty());
    }

    #[test]
    fn adopted_guards_filter_predictions() {
        let predictor = predictor();
        let functional = BTreeSet::from(["师祖".to_string()]);
        let negatives = [Triple::new("李四", "师祖", "赵六")];
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let refinement = refine_rule(
            &predictor,
            index,
            &negatives,
            &functional,
            &RefineOptions::default(),
        );
        let mut ledger = ledger();
        assert!(ledger.adopt_refinement(&refinement));
        let predictions = predictor.predict_for_entity("李四", Some("师祖"), 8);
        assert!(!predictions.is_empty());
        let filtered = ledger.filter_predictions(&predictor, &predictions);
        assert!(filtered
            .iter()
            .all(|prediction| prediction.triple.object != "赵六"));
    }

    #[test]
    fn clear_all_guards_keeps_refutations() {
        let predictor = predictor();
        let negatives = [Triple::new("李四", "师祖", "赵六")];
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let refinement = refine_rule(
            &predictor,
            index,
            &negatives,
            &BTreeSet::new(),
            &RefineOptions::default(),
        );
        let mut ledger = FoilLedger::new();
        ledger.refute("李四", "师祖", "赵六");
        assert!(ledger.adopt_refinement(&refinement));
        assert_eq!(ledger.guarded_rule_count(), 1);
        assert_eq!(ledger.clear_all_guards(), 1);
        assert_eq!(ledger.guarded_rule_count(), 0);
        assert!(ledger.is_refuted(&Triple::new("李四", "师祖", "赵六")));
    }

    #[test]
    fn ledger_roundtrip_preserves_refutations_and_guards() {
        let predictor = predictor();
        let functional = BTreeSet::from(["师祖".to_string()]);
        let negatives = [Triple::new("李四", "师祖", "赵六")];
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let refinement = refine_rule(
            &predictor,
            index,
            &negatives,
            &functional,
            &RefineOptions::default(),
        );
        let mut ledger = ledger();
        ledger.refute("李四", "师祖", "赵六");
        ledger.mark_functional("师祖");
        assert!(ledger.adopt_refinement(&refinement));
        let encoded = serde_json::to_string(&ledger).expect("账本应可序列化");
        let restored: FoilLedger = serde_json::from_str(&encoded).expect("账本应可反序列化");
        assert_eq!(restored, ledger);
        assert!(restored.is_refuted(&Triple::new("李四", "师祖", "赵六")));
        assert!(restored.is_functional("师祖"));
        assert_eq!(restored.guarded_rule_count(), 1);
    }

    #[test]
    fn rule_application_enumeration_is_stable_and_deduped() {
        let predictor = predictor();
        let index = predictor
            .rules()
            .iter()
            .position(|rule| {
                rule.head == "师祖"
                    && rule.template == RuleTemplate::Compose
                    && rule.body == vec!["师傅".to_string(), "师傅".to_string()]
            })
            .expect("应挖出组合规则");
        let first = rule_applications(&predictor, index, 64);
        let second = rule_applications(&predictor, index, 64);
        assert_eq!(first, second);
        let mut triples: Vec<Triple> = first
            .iter()
            .map(|application| application.triple.clone())
            .collect();
        triples.sort();
        triples.dedup();
        assert_eq!(triples.len(), first.len());
    }
}

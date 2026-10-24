use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::link_prediction::{LinkPredictor, Prediction, Triple};
use crate::ontology_reasoning::OntologyClosure;

pub const SCORE_SCALE: f64 = 1_000_000.0;
pub const DEFAULT_MAX_HYPOTHESES: usize = 3;
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.5;
pub const DEFAULT_MIN_MARGIN: f64 = 0.15;
pub const DEFAULT_TTL_TURNS: u64 = 3;
pub const DEFAULT_MAX_LEDGER: usize = 64;
pub const MAX_CLARIFICATION_OPTIONS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HypothesisKey {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl HypothesisKey {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }

    pub fn display(&self) -> String {
        format!("{}|{}|{}", self.subject, self.predicate, self.object)
    }

    pub fn sentence(&self) -> String {
        format!(
            "「{}」的「{}」是「{}」",
            self.subject, self.predicate, self.object
        )
    }
}

impl From<&Triple> for HypothesisKey {
    fn from(triple: &Triple) -> Self {
        Self::new(&triple.subject, &triple.predicate, &triple.object)
    }
}

impl From<&HypothesisKey> for Triple {
    fn from(key: &HypothesisKey) -> Self {
        Triple::new(&key.subject, &key.predicate, &key.object)
    }
}

pub fn hypothesis_id(key: &HypothesisKey) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in key
        .subject
        .as_bytes()
        .iter()
        .chain([0u8].iter())
        .chain(key.predicate.as_bytes())
        .chain([0u8].iter())
        .chain(key.object.as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("p{:016x}", hash)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    Pending,
    Confirmed,
    Refuted,
    Expired,
}

impl HypothesisStatus {
    pub fn label(self) -> &'static str {
        match self {
            HypothesisStatus::Pending => "待定",
            HypothesisStatus::Confirmed => "已确认",
            HypothesisStatus::Refuted => "已否定",
            HypothesisStatus::Expired => "已过期",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    AlreadyKnown,
    SessionConfirmed,
    SessionRefuted,
    AlreadyAdjudicated,
    Expired,
    FunctionalConflict,
    TypeConflict,
    LowConfidence,
    Ambiguous,
    Duplicate,
    Budget,
}

impl RejectReason {
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::AlreadyKnown => "已知事实",
            RejectReason::SessionConfirmed => "会话已确认",
            RejectReason::SessionRefuted => "会话已否认",
            RejectReason::AlreadyAdjudicated => "已裁决",
            RejectReason::Expired => "已过期",
            RejectReason::FunctionalConflict => "功能谓词冲突",
            RejectReason::TypeConflict => "类型域/值域冲突",
            RejectReason::LowConfidence => "置信度不足",
            RejectReason::Ambiguous => "候选歧义",
            RejectReason::Duplicate => "重复候选",
            RejectReason::Budget => "预算截断",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElaborationHypothesis {
    pub id: String,
    pub key: HypothesisKey,
    pub confidence_milli: u32,
    pub support: usize,
    pub rule: String,
    pub evidence: Vec<String>,
    pub explanation: String,
    pub status: HypothesisStatus,
    pub created_tick: u64,
    pub resolved_tick: Option<u64>,
}

impl ElaborationHypothesis {
    pub fn confidence(&self) -> f64 {
        f64::from(self.confidence_milli) / SCORE_SCALE
    }

    pub fn sentence(&self) -> String {
        self.key.sentence()
    }

    pub fn is_pending(&self) -> bool {
        self.status == HypothesisStatus::Pending
    }

    pub fn status_line(&self) -> String {
        match self.status {
            HypothesisStatus::Pending => format!(
                "[{}] {}（置信度 {:.2}，支持 {} 例）",
                self.id,
                self.sentence(),
                self.confidence(),
                self.support
            ),
            HypothesisStatus::Confirmed => format!(
                "[{}] {}（已确认，第 {} 轮）",
                self.id,
                self.sentence(),
                self.resolved_tick.unwrap_or(self.created_tick)
            ),
            HypothesisStatus::Refuted => format!(
                "[{}] {}（已否定，第 {} 轮）",
                self.id,
                self.sentence(),
                self.resolved_tick.unwrap_or(self.created_tick)
            ),
            HypothesisStatus::Expired => format!(
                "[{}] {}（已过期，创建于第 {} 轮）",
                self.id,
                self.sentence(),
                self.created_tick
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rejection {
    pub key: HypothesisKey,
    pub reason: RejectReason,
    pub detail: String,
    pub confidence_milli: u32,
}

#[derive(Debug, Clone)]
pub struct Clarification {
    pub subject: String,
    pub predicate: String,
    pub options: Vec<String>,
    pub question: String,
}

#[derive(Debug, Clone, Default)]
pub struct ElaborationPlan {
    pub entity: String,
    pub hypotheses: Vec<ElaborationHypothesis>,
    pub clarifications: Vec<Clarification>,
    pub rejections: Vec<Rejection>,
    pub truncated: bool,
}

impl ElaborationPlan {
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty() && self.clarifications.is_empty()
    }

    pub fn audit_line(&self) -> String {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for rejection in &self.rejections {
            *counts.entry(rejection.reason.label()).or_insert(0) += 1;
        }
        let reasons = if counts.is_empty() {
            "无".to_string()
        } else {
            counts
                .iter()
                .map(|(label, count)| format!("{}×{}", label, count))
                .collect::<Vec<_>>()
                .join("、")
        };
        format!(
            "（推测审计：采纳 {} 条，澄清 {} 组，拒绝 {} 条：{}）",
            self.hypotheses.len(),
            self.clarifications.len(),
            self.rejections.len(),
            reasons
        )
    }

    pub fn render(&self) -> String {
        if self.is_empty() {
            let mut output = format!("关于「{}」，本轮没有可呈现的推测。", self.entity);
            if !self.rejections.is_empty() {
                output.push('\n');
                output.push_str(&self.audit_line());
            }
            return output;
        }
        let mut output = format!(
            "（以下为关于「{}」的推测，不是已确认事实；依据图谱规则与证据，可回复「确认预测：<序号或编号>」或「否定预测：<序号或编号>」）",
            self.entity
        );
        for (index, hypothesis) in self.hypotheses.iter().enumerate() {
            output.push_str(&format!(
                "\n  {}. 可能：{}。（置信度 {:.2}，支持 {} 例）",
                index + 1,
                hypothesis.sentence(),
                hypothesis.confidence(),
                hypothesis.support
            ));
            if !hypothesis.explanation.trim().is_empty() {
                output.push_str(&format!("\n     依据：{}", hypothesis.explanation.trim()));
            }
        }
        for clarification in &self.clarifications {
            output.push_str(&format!("\n  需要澄清：{}", clarification.question));
        }
        output.push('\n');
        output.push_str(&self.audit_line());
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    NotFound(String),
    NotPending(String, HypothesisStatus),
    AmbiguousSelector(String),
}

impl fmt::Display for PredictionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::NotFound(selector) => {
                write!(formatter, "没有找到待定推测「{}」", selector)
            }
            PredictionError::NotPending(id, status) => {
                write!(
                    formatter,
                    "推测「{}」当前状态为「{}」，不能再次裁决",
                    id,
                    status.label()
                )
            }
            PredictionError::AmbiguousSelector(selector) => {
                write!(
                    formatter,
                    "推测选择「{}」命中了多个候选，请给出更完整的编号",
                    selector
                )
            }
        }
    }
}

impl std::error::Error for PredictionError {}

#[derive(Debug, Clone, Default)]
pub struct PredictionGuard {
    known: HashSet<HypothesisKey>,
    session: HashSet<HypothesisKey>,
    refuted: HashSet<HypothesisKey>,
    functional: BTreeSet<String>,
    values: HashMap<(String, String), BTreeSet<String>>,
    types: HashMap<String, BTreeSet<String>>,
    closure: Option<OntologyClosure>,
}

impl PredictionGuard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_known(&mut self, subject: &str, predicate: &str, object: &str) {
        let key = HypothesisKey::new(subject, predicate, object);
        self.values
            .entry((key.subject.clone(), key.predicate.clone()))
            .or_default()
            .insert(key.object.clone());
        self.known.insert(key);
    }

    pub fn insert_session(&mut self, subject: &str, predicate: &str, object: &str) {
        let key = HypothesisKey::new(subject, predicate, object);
        self.values
            .entry((key.subject.clone(), key.predicate.clone()))
            .or_default()
            .insert(key.object.clone());
        self.session.insert(key);
    }

    pub fn insert_refuted(&mut self, subject: &str, predicate: &str, object: &str) {
        self.refuted
            .insert(HypothesisKey::new(subject, predicate, object));
    }

    pub fn mark_functional(&mut self, predicate: &str) {
        let predicate = predicate.trim();
        if !predicate.is_empty() {
            self.functional.insert(predicate.to_string());
        }
    }

    pub fn insert_type(&mut self, entity: &str, etype: &str) {
        self.types
            .entry(entity.to_string())
            .or_default()
            .insert(etype.to_string());
    }

    pub fn set_closure(&mut self, closure: OntologyClosure) {
        self.closure = Some(closure);
    }

    pub fn check(&self, key: &HypothesisKey) -> Result<(), (RejectReason, String)> {
        if self.known.contains(key) {
            return Err((
                RejectReason::AlreadyKnown,
                "该三元组已存在于基础图谱".to_string(),
            ));
        }
        if self.session.contains(key) {
            return Err((
                RejectReason::SessionConfirmed,
                "该三元组已在会话事实层被确认".to_string(),
            ));
        }
        if self.refuted.contains(key) {
            return Err((
                RejectReason::SessionRefuted,
                "该候选已被用户否认，本会话不再提名".to_string(),
            ));
        }
        if self.functional.contains(&key.predicate) {
            if let Some(values) = self
                .values
                .get(&(key.subject.clone(), key.predicate.clone()))
            {
                if let Some(existing) = values.iter().find(|value| *value != &key.object) {
                    return Err((
                        RejectReason::FunctionalConflict,
                        format!(
                            "功能谓词「{}」在「{}」上已有取值「{}」",
                            key.predicate, key.subject, existing
                        ),
                    ));
                }
            }
        }
        if let Some(detail) = self.type_conflict(key) {
            return Err((RejectReason::TypeConflict, detail));
        }
        Ok(())
    }

    fn type_conflict(&self, key: &HypothesisKey) -> Option<String> {
        let closure = self.closure.as_ref()?;
        let subject_types = self.known_types(&key.subject, closure);
        let domain = closure.domain_classes(&key.predicate);
        if let Some((actual, expected)) = Self::first_disjoint(&subject_types, &domain, closure) {
            return Some(format!(
                "主体「{}」的类型「{}」与「{}」的定义域「{}」互斥",
                key.subject, actual, key.predicate, expected
            ));
        }
        let object_types = self.known_types(&key.object, closure);
        let range = closure.range_classes(&key.predicate);
        if let Some((actual, expected)) = Self::first_disjoint(&object_types, &range, closure) {
            return Some(format!(
                "客体「{}」的类型「{}」与「{}」的值域「{}」互斥",
                key.object, actual, key.predicate, expected
            ));
        }
        None
    }

    fn known_types(&self, entity: &str, closure: &OntologyClosure) -> BTreeSet<String> {
        let mut types: BTreeSet<String> = self.types.get(entity).cloned().unwrap_or_default();
        types.extend(closure.types_of(entity));
        types
    }

    fn first_disjoint(
        actual_types: &BTreeSet<String>,
        expected_types: &[String],
        closure: &OntologyClosure,
    ) -> Option<(String, String)> {
        if actual_types.is_empty() || expected_types.is_empty() {
            return None;
        }
        let compatible = actual_types.iter().any(|actual| {
            expected_types
                .iter()
                .any(|expected| actual == expected || closure.is_subclass(actual, expected))
        });
        if compatible {
            return None;
        }
        for actual in actual_types {
            for expected in expected_types {
                if closure.are_disjoint(actual, expected) {
                    return Some((actual.clone(), expected.clone()));
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    key: HypothesisKey,
    confidence_milli: u32,
    support: usize,
    rule: String,
    evidence: Vec<String>,
    explanation: String,
}

impl Candidate {
    fn from_prediction(prediction: &Prediction, predictor: &LinkPredictor) -> Self {
        let rule = predictor
            .rules()
            .get(prediction.rule_index)
            .map(|rule| rule.rule_text())
            .unwrap_or_else(|| "未知规则".to_string());
        let support = predictor
            .rules()
            .get(prediction.rule_index)
            .map(|rule| rule.support)
            .unwrap_or(0);
        Self {
            key: HypothesisKey::from(&prediction.triple),
            confidence_milli: to_milli(prediction.confidence),
            support,
            rule,
            evidence: prediction
                .evidence
                .iter()
                .map(|triple| format!("{}|{}|{}", triple.subject, triple.predicate, triple.object))
                .collect(),
            explanation: predictor.explain(prediction),
        }
    }

    fn better_than(&self, other: &Self) -> bool {
        self.confidence_milli > other.confidence_milli
            || (self.confidence_milli == other.confidence_milli
                && (self.support > other.support
                    || (self.support == other.support && self.rule < other.rule)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveElaboration {
    #[serde(default)]
    tick: u64,
    #[serde(default)]
    hypotheses: BTreeMap<String, ElaborationHypothesis>,
    #[serde(default = "default_max_hypotheses")]
    max_hypotheses: usize,
    #[serde(default = "default_min_confidence_milli")]
    min_confidence_milli: u32,
    #[serde(default = "default_min_margin_milli")]
    min_margin_milli: u32,
    #[serde(default = "default_ttl_turns")]
    ttl_turns: u64,
    #[serde(default = "default_max_ledger")]
    max_ledger: usize,
}

fn default_max_hypotheses() -> usize {
    DEFAULT_MAX_HYPOTHESES
}

fn default_min_confidence_milli() -> u32 {
    to_milli(DEFAULT_MIN_CONFIDENCE)
}

fn default_min_margin_milli() -> u32 {
    to_milli(DEFAULT_MIN_MARGIN)
}

fn default_ttl_turns() -> u64 {
    DEFAULT_TTL_TURNS
}

fn default_max_ledger() -> usize {
    DEFAULT_MAX_LEDGER
}

fn to_milli(score: f64) -> u32 {
    (score.clamp(0.0, 1.0) * SCORE_SCALE).round() as u32
}

impl Default for PredictiveElaboration {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictiveElaboration {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_MAX_HYPOTHESES,
            DEFAULT_MIN_CONFIDENCE,
            DEFAULT_MIN_MARGIN,
            DEFAULT_TTL_TURNS,
        )
    }

    pub fn with_limits(
        max_hypotheses: usize,
        min_confidence: f64,
        min_margin: f64,
        ttl_turns: u64,
    ) -> Self {
        Self {
            tick: 0,
            hypotheses: BTreeMap::new(),
            max_hypotheses: max_hypotheses.max(1),
            min_confidence_milli: to_milli(min_confidence),
            min_margin_milli: to_milli(min_margin),
            ttl_turns: ttl_turns.max(1),
            max_ledger: DEFAULT_MAX_LEDGER.max(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn begin_turn(&mut self) {
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        let ttl = self.ttl_turns;
        for hypothesis in self.hypotheses.values_mut() {
            if hypothesis.status == HypothesisStatus::Pending
                && tick.saturating_sub(hypothesis.created_tick) > ttl
            {
                hypothesis.status = HypothesisStatus::Expired;
                hypothesis.resolved_tick = Some(tick);
            }
        }
    }

    pub fn pending(&self) -> Vec<&ElaborationHypothesis> {
        let mut pending: Vec<&ElaborationHypothesis> = self
            .hypotheses
            .values()
            .filter(|hypothesis| hypothesis.is_pending())
            .collect();
        pending.sort_by(|left, right| {
            right
                .confidence_milli
                .cmp(&left.confidence_milli)
                .then_with(|| left.key.cmp(&right.key))
        });
        pending
    }

    pub fn hypothesis(&self, id: &str) -> Option<&ElaborationHypothesis> {
        self.hypotheses.get(id)
    }

    pub fn propose(
        &mut self,
        entity: &str,
        predictions: &[Prediction],
        predictor: &LinkPredictor,
        guard: &PredictionGuard,
    ) -> ElaborationPlan {
        self.begin_turn();
        let mut rejections: Vec<Rejection> = Vec::new();
        let mut candidates: BTreeMap<HypothesisKey, Candidate> = BTreeMap::new();

        for prediction in predictions {
            let key = HypothesisKey::from(&prediction.triple);
            if let Some(existing) = self.hypotheses.get(&hypothesis_id(&key)) {
                if existing.status != HypothesisStatus::Pending {
                    rejections.push(Rejection {
                        key: key.clone(),
                        reason: if existing.status == HypothesisStatus::Expired {
                            RejectReason::Expired
                        } else {
                            RejectReason::AlreadyAdjudicated
                        },
                        detail: format!("该候选当前状态为「{}」", existing.status.label()),
                        confidence_milli: to_milli(prediction.confidence),
                    });
                    continue;
                }
            }
            if let Err((reason, detail)) = guard.check(&key) {
                rejections.push(Rejection {
                    key,
                    reason,
                    detail,
                    confidence_milli: to_milli(prediction.confidence),
                });
                continue;
            }
            if to_milli(prediction.confidence) < self.min_confidence_milli {
                rejections.push(Rejection {
                    key,
                    reason: RejectReason::LowConfidence,
                    detail: format!(
                        "置信度 {:.2} 低于阈值 {:.2}",
                        prediction.confidence,
                        f64::from(self.min_confidence_milli) / SCORE_SCALE
                    ),
                    confidence_milli: to_milli(prediction.confidence),
                });
                continue;
            }
            let candidate = Candidate::from_prediction(prediction, predictor);
            match candidates.get(&candidate.key) {
                Some(existing) if !candidate.better_than(existing) => {
                    rejections.push(Rejection {
                        key: candidate.key.clone(),
                        reason: RejectReason::Duplicate,
                        detail: format!("同一候选已有更强证据「{}」", existing.rule),
                        confidence_milli: candidate.confidence_milli,
                    });
                }
                Some(existing) => {
                    rejections.push(Rejection {
                        key: existing.key.clone(),
                        reason: RejectReason::Duplicate,
                        detail: format!("同一候选已有更强证据「{}」", candidate.rule),
                        confidence_milli: existing.confidence_milli,
                    });
                    candidates.insert(candidate.key.clone(), candidate);
                }
                None => {
                    candidates.insert(candidate.key.clone(), candidate);
                }
            }
        }

        let clarifications = self.remove_ambiguous(&mut candidates, &mut rejections);

        let mut accepted: Vec<Candidate> = candidates.into_values().collect();
        accepted.sort_by(|left, right| {
            right
                .confidence_milli
                .cmp(&left.confidence_milli)
                .then_with(|| left.key.cmp(&right.key))
        });

        let mut truncated = false;
        if accepted.len() > self.max_hypotheses {
            truncated = true;
            for candidate in accepted.drain(self.max_hypotheses..) {
                rejections.push(Rejection {
                    key: candidate.key,
                    reason: RejectReason::Budget,
                    detail: format!("本轮最多呈现 {} 条推测", self.max_hypotheses),
                    confidence_milli: candidate.confidence_milli,
                });
            }
        }

        let mut registered: Vec<ElaborationHypothesis> = Vec::new();
        for candidate in accepted {
            if !self.make_room() {
                rejections.push(Rejection {
                    key: candidate.key,
                    reason: RejectReason::Budget,
                    detail: format!("推测台账最多保留 {} 条", self.max_ledger),
                    confidence_milli: candidate.confidence_milli,
                });
                truncated = true;
                continue;
            }
            let id = hypothesis_id(&candidate.key);
            let tick = self.tick;
            let hypothesis = self
                .hypotheses
                .entry(id.clone())
                .and_modify(|existing| {
                    existing.confidence_milli = candidate.confidence_milli;
                    existing.support = candidate.support;
                    existing.rule = candidate.rule.clone();
                    existing.evidence = candidate.evidence.clone();
                    existing.explanation = candidate.explanation.clone();
                })
                .or_insert_with(|| ElaborationHypothesis {
                    id: id.clone(),
                    key: candidate.key.clone(),
                    confidence_milli: candidate.confidence_milli,
                    support: candidate.support,
                    rule: candidate.rule.clone(),
                    evidence: candidate.evidence.clone(),
                    explanation: candidate.explanation.clone(),
                    status: HypothesisStatus::Pending,
                    created_tick: tick,
                    resolved_tick: None,
                })
                .clone();
            registered.push(hypothesis);
        }

        rejections.sort_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then_with(|| left.reason.cmp(&right.reason))
        });

        ElaborationPlan {
            entity: entity.to_string(),
            hypotheses: registered,
            clarifications,
            rejections,
            truncated,
        }
    }

    fn remove_ambiguous(
        &self,
        candidates: &mut BTreeMap<HypothesisKey, Candidate>,
        rejections: &mut Vec<Rejection>,
    ) -> Vec<Clarification> {
        let mut groups: BTreeMap<(String, String), Vec<HypothesisKey>> = BTreeMap::new();
        for key in candidates.keys() {
            groups
                .entry((key.subject.clone(), key.predicate.clone()))
                .or_default()
                .push(key.clone());
        }
        let mut clarifications = Vec::new();
        for ((subject, predicate), mut keys) in groups {
            if keys.len() < 2 {
                continue;
            }
            keys.sort_by(|left, right| {
                let left_candidate = candidates.get(left).expect("候选应存在");
                let right_candidate = candidates.get(right).expect("候选应存在");
                right_candidate
                    .confidence_milli
                    .cmp(&left_candidate.confidence_milli)
                    .then_with(|| left.object.cmp(&right.object))
            });
            let ambiguous = keys.windows(2).any(|pair| {
                let left = candidates.get(&pair[0]).expect("候选应存在");
                let right = candidates.get(&pair[1]).expect("候选应存在");
                left.confidence_milli.saturating_sub(right.confidence_milli) < self.min_margin_milli
            });
            if !ambiguous {
                continue;
            }
            let options: Vec<String> = keys
                .iter()
                .take(MAX_CLARIFICATION_OPTIONS)
                .filter_map(|key| candidates.get(key))
                .map(|candidate| format!("「{}」", candidate.key.object))
                .collect();
            let question = format!(
                "「{}」的「{}」有多个相近候选：{}。你指的是哪一个？",
                subject,
                predicate,
                options.join("、")
            );
            for key in &keys {
                let candidate = candidates.remove(key).expect("候选应存在");
                rejections.push(Rejection {
                    key: key.clone(),
                    reason: RejectReason::Ambiguous,
                    detail: "同一主谓的候选分差不足，改为澄清而不是并列输出".to_string(),
                    confidence_milli: candidate.confidence_milli,
                });
            }
            clarifications.push(Clarification {
                subject,
                predicate,
                options,
                question,
            });
        }
        clarifications.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
        });
        clarifications
    }

    fn make_room(&mut self) -> bool {
        if self.hypotheses.len() < self.max_ledger {
            return true;
        }
        let mut removable: Vec<(u64, String)> = self
            .hypotheses
            .iter()
            .filter(|(_, hypothesis)| !hypothesis.is_pending())
            .map(|(id, hypothesis)| (hypothesis.created_tick, id.clone()))
            .collect();
        removable.sort();
        for (_, id) in removable {
            if self.hypotheses.len() < self.max_ledger {
                break;
            }
            self.hypotheses.remove(&id);
        }
        self.hypotheses.len() < self.max_ledger
    }

    pub fn resolve(
        &mut self,
        selector: &str,
        confirmed: bool,
    ) -> Result<ElaborationHypothesis, PredictionError> {
        let id = self.find_pending(selector)?;
        let tick = self.tick;
        let hypothesis = self
            .hypotheses
            .get_mut(&id)
            .expect("已定位的待定推测应存在");
        hypothesis.status = if confirmed {
            HypothesisStatus::Confirmed
        } else {
            HypothesisStatus::Refuted
        };
        hypothesis.resolved_tick = Some(tick);
        Ok(hypothesis.clone())
    }

    pub fn reopen(&mut self, id: &str) -> bool {
        let Some(hypothesis) = self.hypotheses.get_mut(id) else {
            return false;
        };
        if hypothesis.status == HypothesisStatus::Pending {
            return false;
        }
        hypothesis.status = HypothesisStatus::Pending;
        hypothesis.resolved_tick = None;
        true
    }

    fn find_pending(&self, selector: &str) -> Result<String, PredictionError> {
        let selector = selector.trim();
        if selector.is_empty() {
            return Err(PredictionError::NotFound(selector.to_string()));
        }
        let pending = self.pending();
        if let Ok(index) = selector.parse::<usize>() {
            if index == 0 || index > pending.len() {
                return Err(PredictionError::NotFound(selector.to_string()));
            }
            return Ok(pending[index - 1].id.clone());
        }
        if let Some(hypothesis) = self.hypotheses.get(selector) {
            return if hypothesis.is_pending() {
                Ok(hypothesis.id.clone())
            } else {
                Err(PredictionError::NotPending(
                    hypothesis.id.clone(),
                    hypothesis.status,
                ))
            };
        }
        let matches: Vec<&ElaborationHypothesis> = pending
            .iter()
            .copied()
            .filter(|hypothesis| hypothesis.id.starts_with(selector))
            .collect();
        if matches.len() == 1 {
            return Ok(matches[0].id.clone());
        }
        if matches.len() > 1 {
            return Err(PredictionError::AmbiguousSelector(selector.to_string()));
        }
        let parts: Vec<&str> = selector.split(['|', '｜']).map(str::trim).collect();
        if parts.len() == 3 && parts.iter().all(|part| !part.is_empty()) {
            let key = HypothesisKey::new(parts[0], parts[1], parts[2]);
            let id = hypothesis_id(&key);
            if let Some(hypothesis) = self.hypotheses.get(&id) {
                return if hypothesis.is_pending() {
                    Ok(id)
                } else {
                    Err(PredictionError::NotPending(id, hypothesis.status))
                };
            }
        }
        Err(PredictionError::NotFound(selector.to_string()))
    }

    pub fn render_report(&self) -> String {
        let mut counts: BTreeMap<HypothesisStatus, usize> = BTreeMap::new();
        for hypothesis in self.hypotheses.values() {
            *counts.entry(hypothesis.status).or_insert(0) += 1;
        }
        let pending = self.pending();
        let mut output = format!(
            "推测台账：待定 {}，已确认 {}，已否定 {}，已过期 {}（轮次 {}）。",
            counts.get(&HypothesisStatus::Pending).copied().unwrap_or(0),
            counts
                .get(&HypothesisStatus::Confirmed)
                .copied()
                .unwrap_or(0),
            counts.get(&HypothesisStatus::Refuted).copied().unwrap_or(0),
            counts.get(&HypothesisStatus::Expired).copied().unwrap_or(0),
            self.tick
        );
        if pending.is_empty() {
            output.push_str("\n  当前没有待定推测。");
        } else {
            output.push_str("\n  待定推测：");
            for hypothesis in pending {
                output.push_str(&format!("\n  {}", hypothesis.status_line()));
            }
        }
        let mut resolved: Vec<&ElaborationHypothesis> = self
            .hypotheses
            .values()
            .filter(|hypothesis| !hypothesis.is_pending())
            .collect();
        resolved.sort_by(|left, right| {
            right
                .resolved_tick
                .cmp(&left.resolved_tick)
                .then_with(|| left.id.cmp(&right.id))
        });
        if !resolved.is_empty() {
            output.push_str("\n  最近裁决：");
            for hypothesis in resolved.into_iter().take(8) {
                output.push_str(&format!("\n  {}", hypothesis.status_line()));
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology_reasoning::OntologyReasoner;

    fn chain_predictor() -> LinkPredictor {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.mine_rules(1, 0.5);
        predictor
    }

    fn guard_from(predictor: &LinkPredictor) -> PredictionGuard {
        let mut guard = PredictionGuard::new();
        for triple in predictor.triples() {
            guard.insert_known(&triple.subject, &triple.predicate, &triple.object);
        }
        guard
    }

    #[test]
    fn propose_renders_hedged_hypothesis_with_evidence() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", None, 8);
        let guard = guard_from(&predictor);
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);

        assert!(plan
            .hypotheses
            .iter()
            .any(|hypothesis| hypothesis.key.object == "赵六"));
        let text = plan.render();
        assert!(text.contains("推测"));
        assert!(text.contains("不是已确认事实"));
        assert!(text.contains("依据"));
        assert!(text.contains("审计"));
        assert!(!plan.hypotheses.is_empty());
    }

    #[test]
    fn known_and_refuted_triples_are_rejected() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", None, 8);
        let target = predictions
            .iter()
            .find(|prediction| prediction.triple.object == "赵六")
            .expect("应有预测")
            .triple
            .clone();
        let mut guard = guard_from(&predictor);
        guard.insert_refuted(&target.subject, &target.predicate, &target.object);
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);
        assert!(!plan
            .hypotheses
            .iter()
            .any(|hypothesis| hypothesis.key.object == "赵六"));
        assert!(plan.rejections.iter().any(|rejection| {
            rejection.reason == RejectReason::SessionRefuted && rejection.key.object == "赵六"
        }));

        let mut guard = guard_from(&predictor);
        guard.insert_session("李四", "师祖", "赵六");
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);
        assert!(!plan
            .hypotheses
            .iter()
            .any(|hypothesis| hypothesis.key.object == "赵六"));
        assert!(plan.rejections.iter().any(|rejection| {
            rejection.reason == RejectReason::SessionConfirmed && rejection.key.object == "赵六"
        }));
    }

    #[test]
    fn functional_predicate_conflict_is_rejected() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("甲", "r1", "乙");
        predictor.add_triple("乙", "r2", "丙");
        predictor.add_triple("甲", "r3", "丙");
        predictor.add_triple("甲", "r1", "辛");
        predictor.add_triple("辛", "r2", "壬");
        predictor.add_triple("戊", "r1", "己");
        predictor.add_triple("己", "r2", "庚");
        predictor.add_triple("戊", "r3", "庚");
        predictor.mine_rules(1, 0.5);
        let predictions = predictor.predict_for_entity("甲", Some("r3"), 8);
        assert!(!predictions.is_empty());
        let mut guard = guard_from(&predictor);
        guard.mark_functional("r3");
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("甲", &predictions, &predictor, &guard);
        assert!(plan.hypotheses.is_empty());
        assert!(plan
            .rejections
            .iter()
            .any(|rejection| rejection.reason == RejectReason::FunctionalConflict));
    }

    #[test]
    fn ambiguous_group_becomes_one_clarification() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("甲", "r1", "乙");
        predictor.add_triple("乙", "r2", "丙");
        predictor.add_triple("甲", "r1", "丁");
        predictor.add_triple("丁", "r2", "戊");
        predictor.add_triple("己", "r1", "庚");
        predictor.add_triple("庚", "r2", "辛");
        predictor.add_triple("壬", "r1", "癸");
        predictor.add_triple("癸", "r2", "子");
        predictor.add_triple("己", "r3", "辛");
        predictor.add_triple("壬", "r3", "子");
        predictor.mine_rules(1, 0.5);
        let predictions = predictor.predict_for_entity("甲", Some("r3"), 8);
        assert!(predictions.len() >= 2);
        let guard = guard_from(&predictor);
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("甲", &predictions, &predictor, &guard);
        assert!(plan.hypotheses.is_empty());
        assert_eq!(plan.clarifications.len(), 1);
        assert!(plan.clarifications[0].question.contains("哪一个"));
        assert!(plan
            .rejections
            .iter()
            .all(|rejection| rejection.reason == RejectReason::Ambiguous));
    }

    #[test]
    fn type_conflict_is_rejected_against_ontology_closure() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", Some("师祖"), 8);
        assert!(!predictions.is_empty());
        let mut ontology = OntologyReasoner::new(4);
        ontology.add_instance("李四", "植物");
        ontology.add_domain("师祖", "人物");
        ontology.add_disjoint("植物", "人物");
        let mut guard = guard_from(&predictor);
        guard.set_closure(ontology.materialize());
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);
        assert!(plan.hypotheses.is_empty());
        assert!(plan
            .rejections
            .iter()
            .any(|rejection| rejection.reason == RejectReason::TypeConflict));
    }

    #[test]
    fn ttl_expires_pending_but_keeps_receipt() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", None, 8);
        let guard = guard_from(&predictor);
        let mut ledger = PredictiveElaboration::with_limits(3, 0.5, 0.15, 1);
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);
        let id = plan.hypotheses[0].id.clone();
        ledger.begin_turn();
        assert!(ledger.hypothesis(&id).unwrap().is_pending());
        ledger.begin_turn();
        assert_eq!(
            ledger.hypothesis(&id).unwrap().status,
            HypothesisStatus::Expired
        );
        assert!(ledger.render_report().contains("已过期"));
        assert!(ledger.resolve(&id, true).is_err());
    }

    #[test]
    fn confirm_and_refute_change_lifecycle_state() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", None, 8);
        let guard = guard_from(&predictor);

        let mut confirmed_ledger = PredictiveElaboration::new();
        let plan = confirmed_ledger.propose("李四", &predictions, &predictor, &guard);
        let first = plan.hypotheses[0].id.clone();
        assert_eq!(
            confirmed_ledger.resolve(&first, true).unwrap().status,
            HypothesisStatus::Confirmed
        );
        assert!(confirmed_ledger.resolve(&first, true).is_err());
        assert!(confirmed_ledger.render_report().contains("已确认 1"));

        let mut refuted_ledger = PredictiveElaboration::new();
        let plan = refuted_ledger.propose("李四", &predictions, &predictor, &guard);
        let first = plan.hypotheses[0].id.clone();
        assert_eq!(
            refuted_ledger.resolve(&first, false).unwrap().status,
            HypothesisStatus::Refuted
        );
        assert!(refuted_ledger.render_report().contains("已否定 1"));
    }

    #[test]
    fn ledger_snapshot_roundtrip_preserves_pending_state() {
        let predictor = chain_predictor();
        let predictions = predictor.predict_for_entity("李四", None, 8);
        let guard = guard_from(&predictor);
        let mut ledger = PredictiveElaboration::new();
        let plan = ledger.propose("李四", &predictions, &predictor, &guard);
        let id = plan.hypotheses[0].id.clone();
        let encoded = serde_json::to_string(&ledger).expect("台账应可序列化");
        let restored: PredictiveElaboration =
            serde_json::from_str(&encoded).expect("台账应可反序列化");
        assert_eq!(restored.tick(), ledger.tick());
        assert!(restored.hypothesis(&id).unwrap().is_pending());
        let mut restored = restored;
        assert!(restored.resolve(&id, true).is_ok());
    }

    #[test]
    fn budget_truncation_is_audited() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("甲", "r1", "乙");
        predictor.add_triple("乙", "r2", "丙");
        predictor.add_triple("甲", "r1", "丁");
        predictor.add_triple("丁", "r2", "戊");
        predictor.add_triple("己", "r1", "庚");
        predictor.add_triple("庚", "r2", "辛");
        predictor.add_triple("壬", "r1", "癸");
        predictor.add_triple("癸", "r2", "子");
        predictor.add_triple("己", "r3", "辛");
        predictor.add_triple("壬", "r3", "子");
        predictor.mine_rules(1, 0.5);
        let predictions = predictor.predict_for_entity("甲", Some("r3"), 8);
        assert!(predictions.len() >= 2);
        let guard = guard_from(&predictor);
        let mut ledger = PredictiveElaboration::with_limits(1, 0.5, 0.0, 3);
        let plan = ledger.propose("甲", &predictions, &predictor, &guard);
        assert_eq!(plan.hypotheses.len(), 1);
        assert!(plan.truncated);
        assert!(plan
            .rejections
            .iter()
            .any(|rejection| rejection.reason == RejectReason::Budget));
    }
}

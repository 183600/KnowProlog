//! 对话级证据装配器（Contextual RAG，纯符号、零训练）。
//!
//! 本模块不是新的检索器，而是把已有器官闭合起来：
//!
//! 1. [`QueryFrame`]：把当前问句、对话焦点栈、上一谓词、期望答案类型与
//!    已引用证据编译成一次检索请求。追问「还有呢」「那杜甫呢」「他多高」
//!    即使没有谓词词面，也能沿用焦点与上一谓词。
//! 2. 多通道证据：词面/结构/谓词信号来自 [`crate::dense_retrieval`]，
//!    全息 VSA 软召回来自 [`crate::holographic_memory`]，上下文信号在这里
//!    做 late interaction。只有通过硬核查（索引事实或全息登记的同一三元组）
//!    的证据才能进入答句。
//! 3. [`EvidenceBundle`]：按问句形态装配单条、画像、槽位列表、路径或对比
//!    证据束，用 MMR 去冗余，并报告多值槽位与覆盖度。
//! 4. 受约束生成：每一句都以 `[E1]`、`[E2]` 绑定证据 id；证据不足则弃权，
//!    不补写、不把候选列表冒充确定结论。
//!
//! 证书语义：一旦引入上下文/全息/MMR 融合，排序就是启发式的，回执固定
//! 标记 `certified = false`；精确 top-k 仍由 `DenseRetriever::search` 单独证明。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::dense_retrieval::RetrievalHit;

/// 默认证据束上限。
pub const DEFAULT_MAX_ITEMS: usize = 6;
/// 低于该融合分的证据不进入答句（弱候选只留在回执里）。
const MIN_SCORE: f64 = 0.02;
/// MMR 的相关性权重（其余给冗余惩罚）。
const MMR_LAMBDA: f64 = 0.72;
/// 已被上文引用的证据惩罚/过滤阈值。
const REFERENCED_PENALTY: f64 = 0.45;

/// 问句期望的答案类型（只做软排序，不做硬过滤）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerExpectation {
    Person,
    Place,
    Number,
    Time,
    Entity,
    Predicate,
    Unknown,
}

impl AnswerExpectation {
    /// 从问句表面线索推断期望类型；线索冲突时按更具体的类型优先。
    pub fn from_question(question: &str) -> Self {
        let q = question.trim();
        if q.is_empty() {
            return Self::Unknown;
        }
        if ["什么时候", "何时", "哪一年", "哪年", "几月", "几日", "什么时间"]
            .iter()
            .any(|cue| q.contains(cue))
        {
            return Self::Time;
        }
        if [
            "多少", "多大", "多高", "多重", "多长", "多久", "多远", "几年", "几岁", "几个", "几次",
            "百分比", "数量",
        ]
        .iter()
        .any(|cue| q.contains(cue))
        {
            return Self::Number;
        }
        if ["哪里", "哪儿", "何处", "何地", "什么地方", "哪个城市", "哪个国家"]
            .iter()
            .any(|cue| q.contains(cue))
        {
            return Self::Place;
        }
        if ["谁", "哪位", "哪个人", "什么人", "人物"].iter().any(|cue| q.contains(cue)) {
            return Self::Person;
        }
        if ["什么关系", "什么联系", "怎样的关系"].iter().any(|cue| q.contains(cue)) {
            return Self::Predicate;
        }
        if ["什么", "哪个", "哪项", "哪一个"].iter().any(|cue| q.contains(cue)) {
            return Self::Entity;
        }
        Self::Unknown
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Person => "人物",
            Self::Place => "地点",
            Self::Number => "数量",
            Self::Time => "时间",
            Self::Entity => "实体",
            Self::Predicate => "关系",
            Self::Unknown => "未指定",
        }
    }

    /// 取值是否在表面形态上符合期望（只用于软加权）。
    pub fn value_matches(self, value: &str) -> bool {
        let value = value.trim();
        if value.is_empty() {
            return false;
        }
        match self {
            Self::Number => {
                value.chars().any(|c| c.is_ascii_digit())
                    || value.chars().any(|c| {
                        matches!(
                            c,
                            '零' | '一'
                                | '二'
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
                    })
            }
            Self::Time => {
                value.chars().any(|c| c.is_ascii_digit())
                    || ["年", "月", "日", "时", "世纪", "年代"]
                        .iter()
                        .any(|cue| value.contains(cue))
                    || Self::Number.value_matches(value)
            }
            Self::Person | Self::Place | Self::Entity | Self::Predicate | Self::Unknown => true,
        }
    }
}

/// 证据三元组的稳定键。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EvidenceKey {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl EvidenceKey {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, object: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// 全息通道送入装配器的一条已核实候选。
#[derive(Debug, Clone, PartialEq)]
pub struct HoloEvidence {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
    pub verified: bool,
}

impl HoloEvidence {
    pub fn verified(subject: &str, predicate: &str, object: &str, score: f64) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            score,
            verified: true,
        }
    }

    pub fn unverified(subject: &str, predicate: &str, object: &str, score: f64) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            score,
            verified: false,
        }
    }
}

/// 编译查询帧所需的对话态输入。
#[derive(Debug, Clone)]
pub struct FrameInput<'a> {
    pub question: &'a str,
    /// 当前问句里显式出现的实体。
    pub explicit_anchors: Vec<String>,
    /// 当前问句里显式提及的谓词。
    pub explicit_predicate: Option<String>,
    pub focus: &'a [String],
    pub last_predicate: Option<&'a str>,
    pub referenced: &'a [EvidenceKey],
}

/// 一次检索的对话态查询帧。
#[derive(Debug, Clone, Serialize)]
pub struct QueryFrame {
    pub text: String,
    pub anchors: Vec<String>,
    pub explicit_anchors: Vec<String>,
    pub focus: Vec<String>,
    pub predicate: Option<String>,
    pub expectation: AnswerExpectation,
    pub continuation: bool,
    pub referenced: Vec<EvidenceKey>,
    pub notes: Vec<String>,
}

/// 是否是「接着上文说」的追问（仅用于软解释；不改变知识真假）。
pub fn is_continuation(question: &str) -> bool {
    let q = question.trim();
    if q.is_empty() {
        return false;
    }
    [
        "还有", "其他", "别的", "另外", "接着", "继续", "上面", "刚才", "之前", "前面",
        "同样", "补充", "再说", "再多", "那",
    ]
    .iter()
    .any(|cue| q.contains(cue))
        || (q.ends_with('呢') && q.chars().count() <= 20)
}

/// 问句是否值得走开放式证据装配（避免把陈述句当问题回答）。
pub fn looks_like_question(question: &str) -> bool {
    let q = question.trim();
    if q.is_empty() {
        return false;
    }
    q.contains('?')
        || q.contains('？')
        || is_continuation(q)
        || [
            "谁", "什么", "哪", "多少", "多高", "多大", "多重", "多长", "几", "怎么", "如何",
            "何时", "介绍", "说说", "吗",
        ]
        .iter()
        .any(|cue| q.contains(cue))
}

impl QueryFrame {
    /// 编译查询帧：显式锚点优先；追问且无新实体时沿用焦点，无新谓词时沿用上一谓词。
    pub fn compile(input: FrameInput<'_>) -> Self {
        let question = input.question.trim();
        let continuation = is_continuation(question);
        let mut notes = Vec::new();
        let explicit_anchors = normalized_entities(input.explicit_anchors);
        let mut anchors = explicit_anchors.clone();
        if anchors.is_empty() {
            anchors = normalized_entities(input.focus.iter().cloned()).into_iter().take(3).collect();
            if !anchors.is_empty() {
                notes.push("问句未出现新实体：以对话焦点作为检索锚点。".to_string());
            }
        }
        let explicit_predicate = input
            .explicit_predicate
            .as_deref()
            .map(str::trim)
            .filter(|predicate| !predicate.is_empty())
            .map(str::to_string);
        let mut predicate = explicit_predicate.clone();
        if predicate.is_none() && continuation {
            predicate = input
                .last_predicate
                .map(str::trim)
                .filter(|predicate| !predicate.is_empty())
                .map(str::to_string);
            if predicate.is_some() {
                notes.push("问句未出现新谓词：沿用上一轮谓词作为槽位约束。".to_string());
            }
        }
        let expectation = AnswerExpectation::from_question(question);
        let focus = normalized_entities(input.focus.iter().cloned());
        let referenced: Vec<EvidenceKey> = input
            .referenced
            .iter()
            .filter(|key| !key.subject.is_empty() && !key.predicate.is_empty())
            .cloned()
            .collect();
        Self {
            text: question.to_string(),
            anchors,
            explicit_anchors,
            focus,
            predicate,
            expectation,
            continuation,
            referenced,
            notes,
        }
    }

    pub fn has_anchor(&self) -> bool {
        !self.anchors.is_empty()
    }

    pub fn is_empty_request(&self) -> bool {
        self.anchors.is_empty() && self.predicate.is_none()
    }
}

fn normalized_entities<I>(entities: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut out: Vec<String> = Vec::new();
    for entity in entities {
        let entity = entity.trim();
        if entity.is_empty() || out.iter().any(|seen| seen == entity) {
            continue;
        }
        out.push(entity.to_string());
    }
    out
}

#[derive(Debug, Clone)]
struct MergedEvidence {
    base: f64,
    lexical: f64,
    structural: f64,
    predicate_match: f64,
    holographic: f64,
    reasons: Vec<String>,
    channels: Vec<&'static str>,
}

impl MergedEvidence {
    fn absorb_hit(&mut self, hit: &RetrievalHit) {
        self.base = self.base.max(hit.score);
        self.lexical = self.lexical.max(hit.lexical);
        self.structural = self.structural.max(hit.structural);
        self.predicate_match = self.predicate_match.max(hit.predicate_match);
        self.push_channel("词面", hit.lexical > 0.001);
        self.push_channel("结构", hit.structural > 0.001);
        self.push_channel("谓词", hit.predicate_match > 0.0);
        for reason in &hit.reasons {
            if !self.reasons.contains(reason) {
                self.reasons.push(reason.clone());
            }
        }
    }

    fn absorb_holo(&mut self, hit: &HoloEvidence) {
        self.base = self.base.max(hit.score * 0.6);
        self.holographic = self.holographic.max(hit.score);
        self.push_channel("全息", hit.score > 0.0);
        let reason = format!("全息软召回（余弦 {:.3}，已硬核查）", hit.score);
        if !self.reasons.contains(&reason) {
            self.reasons.push(reason);
        }
    }

    fn push_channel(&mut self, channel: &'static str, present: bool) {
        if present && !self.channels.contains(&channel) {
            self.channels.push(channel);
        }
    }
}

/// 一条进入证据束的已核实证据。
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceItem {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
    pub lexical: f64,
    pub structural: f64,
    pub predicate_match: f64,
    pub holographic: f64,
    pub contextual: f64,
    pub verified: bool,
    pub novelty: bool,
    pub channels: Vec<String>,
    pub reasons: Vec<String>,
}

impl EvidenceItem {
    pub fn key(&self) -> EvidenceKey {
        EvidenceKey::new(&self.subject, &self.predicate, &self.object)
    }

    pub fn text(&self) -> String {
        format!("{} 的 {} 是 {}", self.subject, self.predicate, self.object)
    }
}

/// 证据束形态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleKind {
    Single,
    Profile,
    SlotList,
    Path,
    Comparison,
    NoMore,
}

impl BundleKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "单条证据",
            Self::Profile => "实体画像",
            Self::SlotList => "槽位列表",
            Self::Path => "显式路径",
            Self::Comparison => "对比",
            Self::NoMore => "无新证据",
        }
    }
}

/// 结构化证据束：答句中的所有事实都来自 `items`。
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBundle {
    pub kind: BundleKind,
    pub anchor: Option<String>,
    pub secondary_anchor: Option<String>,
    pub predicate: Option<String>,
    pub expectation: AnswerExpectation,
    pub items: Vec<EvidenceItem>,
    /// 同一 `(主体, 谓词)` 的多个登记取值（只提示，不擅自归一）。
    pub multiple_values: Vec<String>,
    pub coverage: f64,
    pub answer: String,
}

impl EvidenceBundle {
    pub fn keys(&self) -> Vec<EvidenceKey> {
        self.items.iter().map(EvidenceItem::key).collect()
    }
}

/// 装配回执：候选来源、丢弃原因、通道贡献与证书状态。
#[derive(Debug, Clone, Default, Serialize)]
pub struct AssemblyReceipt {
    pub dense_candidates: usize,
    pub holo_candidates: usize,
    pub merged: usize,
    pub anchor_pruned: usize,
    pub unverified_dropped: usize,
    pub referenced_dropped: usize,
    pub weak_dropped: usize,
    pub selected: usize,
    pub lexical_support: usize,
    pub structural_support: usize,
    pub predicate_support: usize,
    pub holographic_support: usize,
    pub contextual_support: usize,
    /// 上下文/全息/MMR 融合后的排序不宣称全局最优。
    pub certified: bool,
    pub notes: Vec<String>,
}

impl AssemblyReceipt {
    pub fn render_text(&self) -> String {
        format!(
            "证据装配回执：稠密候选 {}，全息候选 {}，融合 {}，选中 {}；\
             丢弃：锚点外 {}、未核查 {}、已引用 {}、弱证据 {}；\
             通道支持：词面 {}、结构 {}、谓词 {}、全息 {}、上下文 {}；{}。",
            self.dense_candidates,
            self.holo_candidates,
            self.merged,
            self.selected,
            self.anchor_pruned,
            self.unverified_dropped,
            self.referenced_dropped,
            self.weak_dropped,
            self.lexical_support,
            self.structural_support,
            self.predicate_support,
            self.holographic_support,
            self.contextual_support,
            if self.certified {
                "top-k 已证明"
            } else {
                "融合排序，不证明全局 top-k（certified=false）"
            }
        )
    }
}

/// 装配结果：帧 + 证据束（可能弃权）+ 回执。
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceAssembly {
    pub frame: QueryFrame,
    pub bundle: Option<EvidenceBundle>,
    pub receipt: AssemblyReceipt,
}

impl EvidenceAssembly {
    pub fn answer(&self) -> Option<&str> {
        self.bundle.as_ref().map(|bundle| bundle.answer.as_str())
    }

    pub fn selected_keys(&self) -> Vec<EvidenceKey> {
        self.bundle
            .as_ref()
            .map(EvidenceBundle::keys)
            .unwrap_or_default()
    }
}

/// 装配证据束：融合稠密检索与全息召回，按对话帧做 late interaction，
/// 过滤未核查/已引用证据，MMR 选择后生成逐句引用的答句。
pub fn assemble(
    frame: &QueryFrame,
    dense_hits: &[RetrievalHit],
    holo_hits: &[HoloEvidence],
    type_hints: &HashMap<String, String>,
    max_items: usize,
) -> EvidenceAssembly {
    let mut receipt = AssemblyReceipt {
        dense_candidates: dense_hits.len(),
        holo_candidates: holo_hits.len(),
        certified: false,
        ..AssemblyReceipt::default()
    };
    let mut merged: BTreeMap<EvidenceKey, MergedEvidence> = BTreeMap::new();
    for hit in dense_hits {
        let key = EvidenceKey::new(&hit.subject, &hit.predicate, &hit.object);
        merged
            .entry(key.clone())
            .or_insert_with(|| MergedEvidence {
                base: 0.0,
                lexical: 0.0,
                structural: 0.0,
                predicate_match: 0.0,
                holographic: 0.0,
                reasons: Vec::new(),
                channels: vec!["稠密"],
            })
            .absorb_hit(hit);
    }
    for hit in holo_hits {
        if !hit.verified {
            receipt.unverified_dropped += 1;
            continue;
        }
        let key = EvidenceKey::new(&hit.subject, &hit.predicate, &hit.object);
        merged
            .entry(key.clone())
            .or_insert_with(|| MergedEvidence {
                base: 0.0,
                lexical: 0.0,
                structural: 0.0,
                predicate_match: 0.0,
                holographic: 0.0,
                reasons: Vec::new(),
                channels: Vec::new(),
            })
            .absorb_holo(hit);
    }
    receipt.merged = merged.len();

    let referenced: BTreeSet<EvidenceKey> = frame.referenced.iter().cloned().collect();
    let mut candidates: Vec<EvidenceItem> = Vec::new();
    for (key, candidate) in merged {
        if !frame.anchors.is_empty()
            && !frame
                .anchors
                .iter()
                .any(|anchor| &key.subject == anchor || &key.object == anchor)
        {
            receipt.anchor_pruned += 1;
            continue;
        }
        let (contextual, contextual_notes) = contextual_signal(frame, &key, type_hints);
        let mut score =
            (candidate.base * 0.8 + candidate.holographic * 0.4 + contextual).clamp(0.0, 1.0);
        let novelty = !referenced.contains(&key);
        if !novelty {
            score = (score - REFERENCED_PENALTY).max(0.0);
        }
        if score < MIN_SCORE && (candidate.base > 0.0 || candidate.holographic > 0.0) {
            receipt.weak_dropped += 1;
            continue;
        }
        receipt.lexical_support += usize::from(candidate.lexical > 0.001);
        receipt.structural_support += usize::from(candidate.structural > 0.001);
        receipt.predicate_support += usize::from(candidate.predicate_match > 0.0);
        receipt.holographic_support += usize::from(candidate.holographic > 0.0);
        receipt.contextual_support += usize::from(contextual > 0.001);
        let mut reasons = candidate.reasons;
        reasons.extend(contextual_notes);
        let mut channels: Vec<String> =
            candidate.channels.iter().map(|channel| channel.to_string()).collect();
        if contextual > 0.001 && !channels.iter().any(|channel| channel == "上下文") {
            channels.push("上下文".to_string());
        }
        candidates.push(EvidenceItem {
            id: String::new(),
            subject: key.subject,
            predicate: key.predicate,
            object: key.object,
            score,
            lexical: candidate.lexical,
            structural: candidate.structural,
            predicate_match: candidate.predicate_match,
            holographic: candidate.holographic,
            contextual,
            verified: true,
            novelty,
            channels,
            reasons,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.key().cmp(&right.key()))
    });

    let referenced_dropped = candidates.iter().filter(|item| !item.novelty).count();
    receipt.referenced_dropped = referenced_dropped;
    let novel: Vec<EvidenceItem> = candidates.into_iter().filter(|item| item.novelty).collect();
    if novel.is_empty() {
        if referenced_dropped > 0 {
            let answer = format!(
                "除上文已引用的 {} 条证据外，没有新的已核实证据。",
                referenced_dropped
            );
            receipt.notes.push("候选全部已被上文引用。".to_string());
            return EvidenceAssembly {
                frame: frame.clone(),
                bundle: Some(EvidenceBundle {
                    kind: BundleKind::NoMore,
                    anchor: frame.anchors.first().cloned(),
                    secondary_anchor: frame.anchors.get(1).cloned(),
                    predicate: frame.predicate.clone(),
                    expectation: frame.expectation,
                    items: Vec::new(),
                    multiple_values: Vec::new(),
                    coverage: 0.0,
                    answer,
                }),
                receipt,
            };
        }
        receipt.notes.push("没有通过锚定与硬核查的证据。".to_string());
        return EvidenceAssembly {
            frame: frame.clone(),
            bundle: None,
            receipt,
        };
    }

    let available = novel.len();
    let mut selected = mmr_select(&novel, max_items.clamp(1, 12), MMR_LAMBDA);
    let coverage = if available == 0 {
        0.0
    } else {
        selected.len() as f64 / available as f64
    };
    for (index, item) in selected.iter_mut().enumerate() {
        item.id = format!("E{}", index + 1);
    }
    let multiple_values = multiple_values(&selected);
    let (kind, ordered) = classify(frame, selected);
    let answer = render_bundle_answer(frame, kind, &ordered, &multiple_values, referenced_dropped, coverage);
    receipt.selected = ordered.len();
    if frame.notes.is_empty() {
        receipt.notes.push("已按对话焦点与槽位约束装配证据。".to_string());
    } else {
        receipt.notes.extend(frame.notes.clone());
    }
    receipt
        .notes
        .push("答句逐条绑定证据 id；未核查候选不进入引用。".to_string());
    EvidenceAssembly {
        frame: frame.clone(),
        bundle: Some(EvidenceBundle {
            kind,
            anchor: frame.anchors.first().cloned(),
            secondary_anchor: frame.anchors.get(1).cloned(),
            predicate: frame.predicate.clone(),
            expectation: frame.expectation,
            items: ordered,
            multiple_values,
            coverage,
            answer,
        }),
        receipt,
    }
}

fn contextual_signal(
    frame: &QueryFrame,
    key: &EvidenceKey,
    type_hints: &HashMap<String, String>,
) -> (f64, Vec<String>) {
    let mut score = 0.0;
    let mut notes = Vec::new();
    if !frame.anchors.is_empty() {
        let boost = if frame.anchors.len() == 1 { 0.25 } else { 0.12 };
        if frame.anchors.iter().any(|anchor| &key.subject == anchor) {
            score += boost;
            notes.push("上下文：命中焦点主体".to_string());
        }
        if frame.anchors.iter().any(|anchor| &key.object == anchor) {
            score += boost;
            notes.push("上下文：命中焦点客体".to_string());
        }
    }
    if let Some(predicate) = &frame.predicate {
        if &key.predicate == predicate {
            score += 0.2;
            notes.push("上下文：命中沿用谓词".to_string());
        }
    }
    match frame.expectation {
        AnswerExpectation::Number | AnswerExpectation::Time => {
            if frame.expectation.value_matches(&key.object) {
                score += 0.05;
                notes.push(format!("答案类型：符合「{}」形态", frame.expectation.label()));
            } else {
                score -= 0.03;
            }
        }
        AnswerExpectation::Person | AnswerExpectation::Place => {
            if let Some(hint) = type_hints.get(&key.object) {
                let expected = match frame.expectation {
                    AnswerExpectation::Person => {
                        hint.contains('人') || hint.to_lowercase().contains("person")
                    }
                    AnswerExpectation::Place => {
                        hint.contains('地')
                            || hint.contains("城市")
                            || hint.contains("国家")
                            || hint.to_lowercase().contains("place")
                    }
                    _ => false,
                };
                if expected {
                    score += 0.05;
                    notes.push(format!("答案类型：与「{}」类型相容", frame.expectation.label()));
                } else {
                    score -= 0.05;
                }
            }
        }
        AnswerExpectation::Entity | AnswerExpectation::Predicate | AnswerExpectation::Unknown => {}
    }
    (score, notes)
}

fn evidence_similarity(left: &EvidenceItem, right: &EvidenceItem) -> f64 {
    if left.key() == right.key() {
        return 1.0;
    }
    if left.subject == right.subject && left.predicate == right.predicate {
        // 同一槽位的多个取值是列表内容，不是重复。
        return 0.05;
    }
    if left.predicate == right.predicate && left.object == right.object {
        return 0.15;
    }
    if left.predicate == right.predicate {
        return 0.1;
    }
    if left.subject == right.subject {
        return 0.2;
    }
    if left.object == right.object {
        return 0.15;
    }
    0.0
}

fn mmr_select(candidates: &[EvidenceItem], max_items: usize, lambda: f64) -> Vec<EvidenceItem> {
    if candidates.is_empty() || max_items == 0 {
        return Vec::new();
    }
    let max_score = candidates
        .iter()
        .map(|item| item.score)
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let mut pool: Vec<usize> = (0..candidates.len()).collect();
    let mut selected: Vec<EvidenceItem> = Vec::new();
    while selected.len() < max_items && !pool.is_empty() {
        let mut best_index = pool[0];
        let mut best_value = mmr_value(
            &candidates[pool[0]],
            &selected,
            max_score,
            lambda,
        );
        for &index in pool.iter().skip(1) {
            let value = mmr_value(&candidates[index], &selected, max_score, lambda);
            let prefer = value > best_value + 1e-12
                || ((value - best_value).abs() <= 1e-12
                    && candidates[index].key() < candidates[best_index].key());
            if prefer {
                best_index = index;
                best_value = value;
            }
        }
        selected.push(candidates[best_index].clone());
        pool.retain(|index| *index != best_index);
    }
    selected
}

fn mmr_value(
    candidate: &EvidenceItem,
    selected: &[EvidenceItem],
    max_score: f64,
    lambda: f64,
) -> f64 {
    let relevance = candidate.score / max_score;
    let redundancy = selected
        .iter()
        .map(|chosen| evidence_similarity(candidate, chosen))
        .fold(0.0_f64, f64::max);
    lambda * relevance - (1.0 - lambda) * redundancy
}

fn multiple_values(items: &[EvidenceItem]) -> Vec<String> {
    let mut groups: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for item in items {
        groups
            .entry((item.subject.clone(), item.predicate.clone()))
            .or_default()
            .insert(item.object.clone());
    }
    groups
        .into_iter()
        .filter(|(_, objects)| objects.len() > 1)
        .map(|((subject, predicate), objects)| {
            format!(
                "「{} 的 {}」有多个登记取值：{}（未擅自归一）",
                subject,
                predicate,
                objects.into_iter().collect::<Vec<_>>().join("、")
            )
        })
        .collect()
}

fn touches(item: &EvidenceItem, entity: &str) -> bool {
    item.subject == entity || item.object == entity
}

fn find_path_indices(items: &[EvidenceItem], start: &str, goal: &str) -> Option<Vec<usize>> {
    for (first_index, first) in items.iter().enumerate() {
        if first.subject != start {
            continue;
        }
        for (second_index, second) in items.iter().enumerate() {
            if first_index == second_index || second.object != goal {
                continue;
            }
            if first.object == second.subject
                && !first.object.is_empty()
                && first.object != start
                && first.object != goal
            {
                return Some(vec![first_index, second_index]);
            }
        }
    }
    None
}

fn classify(
    frame: &QueryFrame,
    selected: Vec<EvidenceItem>,
) -> (BundleKind, Vec<EvidenceItem>) {
    let anchors = &frame.anchors;
    if let (Some(start), Some(goal)) = (anchors.first(), anchors.get(1)) {
        if let Some(indices) = find_path_indices(&selected, start, goal) {
            let mut ordered: Vec<EvidenceItem> = indices
                .iter()
                .filter_map(|index| selected.get(*index).cloned())
                .collect();
            for (index, item) in selected.into_iter().enumerate() {
                if !indices.contains(&index) {
                    ordered.push(item);
                }
            }
            return (BundleKind::Path, ordered);
        }
        let start_seen = selected.iter().any(|item| touches(item, start));
        let goal_seen = selected.iter().any(|item| touches(item, goal));
        if start_seen && goal_seen {
            return (BundleKind::Comparison, selected);
        }
    }
    if selected.len() == 1 {
        return (BundleKind::Single, selected);
    }
    if let Some(predicate) = &frame.predicate {
        let matching = selected
            .iter()
            .filter(|item| &item.predicate == predicate)
            .count();
        if matching > 0 && matching * 2 >= selected.len() {
            return (BundleKind::SlotList, selected);
        }
    }
    (BundleKind::Profile, selected)
}

fn render_bundle_answer(
    frame: &QueryFrame,
    kind: BundleKind,
    items: &[EvidenceItem],
    multiple_values: &[String],
    referenced_dropped: usize,
    coverage: f64,
) -> String {
    let anchor = frame.anchors.first().map(String::as_str);
    let secondary = frame.anchors.get(1).map(String::as_str);
    let predicate = frame.predicate.as_deref();
    let mut output = String::new();
    match kind {
        BundleKind::Single => {
            output.push_str("根据已核实证据：\n");
        }
        BundleKind::SlotList => {
            if frame.continuation && anchor.is_some() && predicate.is_some() {
                output.push_str(&format!(
                    "接着上文关于「{}」的「{}」，还能核对：\n",
                    anchor.unwrap_or(""),
                    predicate.unwrap_or("")
                ));
            } else if anchor.is_some() && predicate.is_some() {
                output.push_str(&format!(
                    "与「{}」的「{}」相关的已核实证据：\n",
                    anchor.unwrap_or(""),
                    predicate.unwrap_or("")
                ));
            } else if let Some(predicate) = predicate {
                output.push_str(&format!("与「{}」相关的已核实证据：\n", predicate));
            } else {
                output.push_str("已核实的槽位列表：\n");
            }
        }
        BundleKind::Profile => {
            if let Some(anchor) = anchor {
                output.push_str(&format!("关于「{}」，证据索引里有这些可核对的事实：\n", anchor));
            } else {
                output.push_str("证据索引里有这些可核对的事实：\n");
            }
        }
        BundleKind::Path => {
            output.push_str(&format!(
                "从「{}」到「{}」存在显式路径：\n",
                anchor.unwrap_or(""),
                secondary.unwrap_or("")
            ));
        }
        BundleKind::Comparison => {
            output.push_str(&format!(
                "把「{}」和「{}」放在一起核对：\n",
                anchor.unwrap_or(""),
                secondary.unwrap_or("")
            ));
        }
        BundleKind::NoMore => {
            return format!(
                "除上文已引用的 {} 条证据外，没有新的已核实证据。",
                referenced_dropped
            );
        }
    }
    for item in items {
        if kind == BundleKind::Path {
            output.push_str(&format!(
                "  [{}] {} —{}→ {}\n",
                item.id, item.subject, item.predicate, item.object
            ));
        } else {
            output.push_str(&format!(
                "  [{}] {} 的 {} 是 {}\n",
                item.id, item.subject, item.predicate, item.object
            ));
        }
    }
    for warning in multiple_values {
        output.push_str(&format!("注意：{}。\n", warning));
    }
    if referenced_dropped > 0 {
        output.push_str(&format!(
            "（已跳过上文引用过的 {} 条证据。）\n",
            referenced_dropped
        ));
    }
    output.push_str(&format!(
        "证据束：{}；已核实 {} 条；覆盖本轮已核实候选的 {:.0}%；上下文/全息/MMR 融合排序（不宣称全局 top-k）。",
        kind.label(),
        items.len(),
        coverage * 100.0
    ));
    output.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(
        subject: &str,
        predicate: &str,
        object: &str,
        score: f64,
        lexical: f64,
        structural: f64,
        predicate_match: f64,
    ) -> RetrievalHit {
        RetrievalHit {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            score,
            lexical,
            structural,
            predicate_match,
            reasons: Vec::new(),
        }
    }

    fn frame(question: &str, anchors: &[&str], predicate: Option<&str>) -> QueryFrame {
        QueryFrame::compile(FrameInput {
            question,
            explicit_anchors: anchors.iter().map(|anchor| anchor.to_string()).collect(),
            explicit_predicate: predicate.map(str::to_string),
            focus: &[],
            last_predicate: None,
            referenced: &[],
        })
    }

    #[test]
    fn frame_followup_uses_focus_and_last_predicate() {
        let focus = vec!["李白".to_string()];
        let frame = QueryFrame::compile(FrameInput {
            question: "还有呢",
            explicit_anchors: Vec::new(),
            explicit_predicate: None,
            focus: &focus,
            last_predicate: Some("朝代"),
            referenced: &[],
        });
        assert_eq!(frame.anchors, vec!["李白"]);
        assert_eq!(frame.predicate.as_deref(), Some("朝代"));
        assert!(frame.continuation);
        assert!(!frame.notes.is_empty());
    }

    #[test]
    fn frame_explicit_entity_overrides_focus_but_keeps_predicate() {
        let focus = vec!["李白".to_string()];
        let frame = QueryFrame::compile(FrameInput {
            question: "那杜甫呢",
            explicit_anchors: vec!["杜甫".to_string()],
            explicit_predicate: None,
            focus: &focus,
            last_predicate: Some("朝代"),
            referenced: &[],
        });
        assert_eq!(frame.anchors, vec!["杜甫"]);
        assert_eq!(frame.predicate.as_deref(), Some("朝代"));
        assert_eq!(frame.expectation, AnswerExpectation::Unknown);
    }

    #[test]
    fn expectation_is_inferred_from_surface_cues() {
        assert_eq!(
            AnswerExpectation::from_question("李白的身高是多少"),
            AnswerExpectation::Number
        );
        assert_eq!(
            AnswerExpectation::from_question("李白出生在什么时候"),
            AnswerExpectation::Time
        );
        assert_eq!(
            AnswerExpectation::from_question("李白出生在哪里"),
            AnswerExpectation::Place
        );
        assert_eq!(
            AnswerExpectation::from_question("李白是谁"),
            AnswerExpectation::Person
        );
    }

    #[test]
    fn single_evidence_answer_is_cited_and_verified() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[hit("李白", "朝代", "唐", 0.8, 0.6, 0.4, 1.0)],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应有证据束");
        assert_eq!(bundle.kind, BundleKind::Single);
        assert_eq!(bundle.items.len(), 1);
        assert_eq!(bundle.items[0].id, "E1");
        assert!(bundle.answer.contains("[E1]"));
        assert!(bundle.answer.contains("唐"));
        assert!(!assembly.receipt.certified);
    }

    #[test]
    fn unverified_holographic_candidate_never_enters_answer() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[],
            &[HoloEvidence::unverified("李白", "朝代", "宋", 0.9)],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        assert!(assembly.bundle.is_none());
        assert_eq!(assembly.receipt.unverified_dropped, 1);
    }

    #[test]
    fn verified_holographic_candidate_supplements_dense_channel() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[],
            &[HoloEvidence::verified("李白", "朝代", "唐", 0.3)],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("已核查全息候选应可作答");
        assert!(bundle.answer.contains("唐"));
        assert!(bundle.items[0].channels.iter().any(|channel| channel == "全息"));
    }

    #[test]
    fn referenced_evidence_is_skipped_for_continuation() {
        let referenced = vec![EvidenceKey::new("李白", "朝代", "唐")];
        let frame = QueryFrame::compile(FrameInput {
            question: "还有呢",
            explicit_anchors: Vec::new(),
            explicit_predicate: Some("朝代".to_string()),
            focus: &["李白".to_string()],
            last_predicate: Some("朝代"),
            referenced: &referenced,
        });
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "朝代", "唐", 0.9, 0.6, 0.4, 1.0),
                hit("李白", "朝代", "宋", 0.4, 0.3, 0.2, 1.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应只回答新证据");
        assert_eq!(bundle.items.len(), 1);
        assert_eq!(bundle.items[0].object, "宋");
        assert_eq!(assembly.receipt.referenced_dropped, 1);
        assert!(bundle.answer.contains("已跳过"));
    }

    #[test]
    fn all_referenced_evidence_yields_explicit_no_more_answer() {
        let referenced = vec![EvidenceKey::new("李白", "朝代", "唐")];
        let frame = QueryFrame::compile(FrameInput {
            question: "还有呢",
            explicit_anchors: Vec::new(),
            explicit_predicate: Some("朝代".to_string()),
            focus: &["李白".to_string()],
            last_predicate: Some("朝代"),
            referenced: &referenced,
        });
        let assembly = assemble(
            &frame,
            &[hit("李白", "朝代", "唐", 0.9, 0.6, 0.4, 1.0)],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应诚实说明没有新证据");
        assert_eq!(bundle.kind, BundleKind::NoMore);
        assert!(bundle.answer.contains("没有新的已核实证据"));
    }

    #[test]
    fn profile_bundle_lists_multiple_predicates() {
        let frame = frame("介绍一下李白", &["李白"], None);
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "朝代", "唐", 0.8, 0.5, 0.3, 0.0),
                hit("李白", "职业", "诗人", 0.6, 0.4, 0.3, 0.0),
                hit("李白", "作品", "静夜思", 0.5, 0.3, 0.2, 0.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("画像应可生成");
        assert_eq!(bundle.kind, BundleKind::Profile);
        assert_eq!(bundle.items.len(), 3);
        assert!(bundle.answer.contains("静夜思"));
    }

    #[test]
    fn duplicate_triples_are_merged_once() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "朝代", "唐", 0.8, 0.5, 0.3, 1.0),
                hit("李白", "朝代", "唐", 0.7, 0.4, 0.2, 1.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应去重后作答");
        assert_eq!(bundle.items.len(), 1);
        assert_eq!(assembly.receipt.merged, 1);
    }

    #[test]
    fn path_bundle_connects_two_anchors() {
        let frame = frame("甲乙之间有什么关系", &["甲", "乙"], None);
        let assembly = assemble(
            &frame,
            &[
                hit("甲", "师傅", "丙", 0.7, 0.4, 0.3, 0.0),
                hit("丙", "师傅", "乙", 0.6, 0.3, 0.3, 0.0),
                hit("甲", "籍贯", "长安", 0.4, 0.2, 0.1, 0.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应能装配路径");
        assert_eq!(bundle.kind, BundleKind::Path);
        assert!(bundle.answer.contains("甲 —师傅→ 丙"));
        assert!(bundle.answer.contains("丙 —师傅→ 乙"));
    }

    #[test]
    fn comparison_bundle_groups_two_anchors() {
        let frame = frame("李白和杜甫的朝代对比", &["李白", "杜甫"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "朝代", "唐", 0.8, 0.5, 0.3, 1.0),
                hit("杜甫", "朝代", "唐", 0.7, 0.4, 0.3, 1.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应能装配对比");
        assert_eq!(bundle.kind, BundleKind::Comparison);
        assert!(bundle.answer.contains("李白"));
        assert!(bundle.answer.contains("杜甫"));
    }

    #[test]
    fn numeric_expectation_orders_numeric_value_first() {
        let frame = frame("李白活了多久", &["李白"], Some("寿命"));
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "寿命", "很长", 0.55, 0.4, 0.2, 1.0),
                hit("李白", "寿命", "61年", 0.5, 0.4, 0.2, 1.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("应能按期望类型重排");
        assert_eq!(bundle.items[0].object, "61年");
    }

    #[test]
    fn multiple_values_are_reported_not_collapsed() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(
            &frame,
            &[
                hit("李白", "朝代", "唐", 0.8, 0.5, 0.3, 1.0),
                hit("李白", "朝代", "盛唐", 0.6, 0.4, 0.2, 1.0),
            ],
            &[],
            &HashMap::new(),
            DEFAULT_MAX_ITEMS,
        );
        let bundle = assembly.bundle.expect("多值不应被折叠");
        assert_eq!(bundle.multiple_values.len(), 1);
        assert!(bundle.answer.contains("多个登记取值"));
    }

    #[test]
    fn empty_evidence_abstains() {
        let frame = frame("李白的朝代是什么", &["李白"], Some("朝代"));
        let assembly = assemble(&frame, &[], &[], &HashMap::new(), DEFAULT_MAX_ITEMS);
        assert!(assembly.bundle.is_none());
        assert!(assembly.answer().is_none());
    }

    #[test]
    fn assembly_is_deterministic() {
        let frame = frame("介绍一下李白", &["李白"], None);
        let hits = vec![
            hit("李白", "朝代", "唐", 0.8, 0.5, 0.3, 0.0),
            hit("李白", "职业", "诗人", 0.8, 0.5, 0.3, 0.0),
            hit("李白", "作品", "静夜思", 0.8, 0.5, 0.3, 0.0),
        ];
        let first = assemble(&frame, &hits, &[], &HashMap::new(), DEFAULT_MAX_ITEMS);
        let second = assemble(&frame, &hits, &[], &HashMap::new(), DEFAULT_MAX_ITEMS);
        assert_eq!(
            first.bundle.as_ref().map(|bundle| bundle.answer.clone()),
            second.bundle.as_ref().map(|bundle| bundle.answer.clone())
        );
    }
}

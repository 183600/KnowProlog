use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::contextual_rag::{AnswerExpectation, EvidenceAssembly, EvidenceBundle, EvidenceItem};
use crate::dense_retrieval::{DenseRetriever, RetrievalHit, SearchBudget, SearchConstraints};
use crate::holographic_memory::HolographicMemory;

const DEFAULT_PAGE_SIZE: usize = 3;
const DEFAULT_MAX_ENTRIES: usize = 24;
const DEFAULT_MAX_RESULTS: usize = 6;

fn default_page_size() -> usize {
    DEFAULT_PAGE_SIZE
}

fn default_max_entries() -> usize {
    DEFAULT_MAX_ENTRIES
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SearchSessionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
}

impl Default for SearchSessionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            page_size: default_page_size(),
            max_entries: default_max_entries(),
        }
    }
}

impl SearchSessionConfig {
    pub fn normalized(&self) -> Self {
        Self {
            enabled: self.enabled,
            page_size: self.page_size.clamp(1, 20),
            max_entries: self.max_entries.clamp(1, 128),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Lookup,
    Enumerate,
    Describe,
    Verify,
}

impl SearchMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Lookup => "lookup",
            Self::Enumerate => "enumerate",
            Self::Describe => "describe",
            Self::Verify => "verify",
        }
    }

    pub fn from_question(question: &str) -> Self {
        let question = question.trim();
        if ["是不是", "是否", "对不对", "对吗", "真的吗"]
            .iter()
            .any(|cue| question.contains(cue))
            || ((question.ends_with('吗') || question.ends_with('么')) && question.contains('是'))
        {
            return Self::Verify;
        }
        if [
            "哪些",
            "都有谁",
            "有谁",
            "列出",
            "所有",
            "全部",
            "几个",
            "多少位",
            "多少人",
        ]
        .iter()
        .any(|cue| question.contains(cue))
        {
            return Self::Enumerate;
        }
        if ["介绍", "描述", "画像", "什么样", "什么样子"]
            .iter()
            .any(|cue| question.contains(cue))
        {
            return Self::Describe;
        }
        Self::Lookup
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostClass {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendCapability {
    pub channel: String,
    pub modes: Vec<SearchMode>,
    pub type_expansion: bool,
    pub certified_top_k: bool,
    pub cost: CostClass,
}

impl BackendCapability {
    pub fn supports(&self, mode: SearchMode) -> bool {
        self.modes.contains(&mode)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub constraints: SearchConstraints,
    pub soft_terms: Vec<String>,
    pub expectation: AnswerExpectation,
    pub max_results: usize,
    pub budget: SearchBudget,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            mode: SearchMode::Describe,
            constraints: SearchConstraints::default(),
            soft_terms: Vec::new(),
            expectation: AnswerExpectation::Unknown,
            max_results: DEFAULT_MAX_RESULTS,
            budget: SearchBudget::default(),
        }
    }
}

impl SearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Self::default()
        }
    }

    pub fn from_question(
        question: impl Into<String>,
        anchor: Option<String>,
        predicate: Option<String>,
        max_results: usize,
    ) -> Self {
        let query = question.into();
        Self {
            mode: SearchMode::from_question(&query),
            constraints: SearchConstraints {
                subject: anchor,
                predicate,
                ..SearchConstraints::default()
            },
            expectation: AnswerExpectation::from_question(&query),
            max_results,
            query,
            ..Self::default()
        }
    }

    pub fn effective_query(&self) -> String {
        let query = self.query.trim();
        if query.is_empty() {
            self.soft_terms.join(" ")
        } else {
            query.to_string()
        }
    }

    fn normalized_max_results(&self) -> usize {
        self.max_results.clamp(1, 50)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
    pub lexical: f64,
    pub structural: f64,
    pub predicate_match: f64,
    pub verified: bool,
    pub channels: Vec<String>,
    pub reasons: Vec<String>,
}

impl SearchHit {
    pub fn key(&self) -> (String, String, String) {
        (
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
        )
    }

    pub fn text(&self) -> String {
        format!("{} 的 {} 是 {}", self.subject, self.predicate, self.object)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PortReceipt {
    pub docs: usize,
    pub candidates: usize,
    pub constraint_pruned: usize,
    pub scanned: usize,
    pub returned: usize,
    pub top_k: usize,
    pub bound_pruned: usize,
    pub budget_exhausted: bool,
    pub certified_top_k: bool,
    pub channels: Vec<String>,
    pub notes: Vec<String>,
}

impl PortReceipt {
    pub fn render_text(&self) -> String {
        let certificate = if self.certified_top_k {
            "已证明 top-k"
        } else {
            "未证明 top-k"
        };
        format!(
            "语义检索回执：通道 {}，候选 {}，约束剪枝 {}，打分 {}，返回 {}/{}；{}。",
            if self.channels.is_empty() {
                "无".to_string()
            } else {
                self.channels.join("+")
            },
            self.candidates,
            self.constraint_pruned,
            self.scanned,
            self.returned,
            self.top_k,
            certificate
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchBatch {
    pub channel: String,
    pub mode: SearchMode,
    pub hits: Vec<SearchHit>,
    pub receipt: PortReceipt,
}

impl SearchBatch {
    fn empty(channel: &str, mode: SearchMode, note: impl Into<String>) -> Self {
        Self {
            channel: channel.to_string(),
            mode,
            hits: Vec::new(),
            receipt: PortReceipt {
                channels: vec![channel.to_string()],
                notes: vec![note.into()],
                ..PortReceipt::default()
            },
        }
    }
}

pub trait SearchBackend {
    fn capability(&self) -> BackendCapability;

    fn search(&mut self, request: &SearchRequest) -> SearchBatch;
}

pub struct DenseSearchBackend<'a> {
    retriever: &'a mut DenseRetriever,
}

impl<'a> DenseSearchBackend<'a> {
    pub fn new(retriever: &'a mut DenseRetriever) -> Self {
        Self { retriever }
    }
}

impl SearchBackend for DenseSearchBackend<'_> {
    fn capability(&self) -> BackendCapability {
        BackendCapability {
            channel: "dense".to_string(),
            modes: vec![
                SearchMode::Lookup,
                SearchMode::Enumerate,
                SearchMode::Describe,
                SearchMode::Verify,
            ],
            type_expansion: true,
            certified_top_k: true,
            cost: CostClass::High,
        }
    }

    fn search(&mut self, request: &SearchRequest) -> SearchBatch {
        let query = request.effective_query();
        let search = self.retriever.search(
            &query,
            request.normalized_max_results(),
            &request.constraints,
            request.budget,
        );
        let hit_count = search.hits.len();
        let hits = search
            .hits
            .into_iter()
            .map(|hit| dense_hit(hit, request.expectation))
            .collect();
        let receipt = PortReceipt {
            docs: search.receipt.docs,
            candidates: search.receipt.candidates,
            constraint_pruned: search.receipt.constraint_pruned,
            scanned: search.receipt.scanned,
            returned: hit_count,
            top_k: search.receipt.top_k,
            bound_pruned: search.receipt.bound_pruned,
            budget_exhausted: search.receipt.budget_exhausted,
            certified_top_k: search.receipt.certified_top_k,
            channels: vec!["dense".to_string()],
            notes: search.receipt.notes,
        };
        SearchBatch {
            channel: "dense".to_string(),
            mode: request.mode,
            hits,
            receipt,
        }
    }
}

fn dense_hit(hit: RetrievalHit, expectation: AnswerExpectation) -> SearchHit {
    let mut reasons = hit.reasons;
    if expectation != AnswerExpectation::Unknown && expectation.value_matches(&hit.object) {
        reasons.push(format!("答案类型符合「{}」", expectation.label()));
    }
    SearchHit {
        subject: hit.subject,
        predicate: hit.predicate,
        object: hit.object,
        score: hit.score,
        lexical: hit.lexical,
        structural: hit.structural,
        predicate_match: hit.predicate_match,
        verified: true,
        channels: vec!["dense".to_string()],
        reasons,
    }
}

pub struct HoloSearchBackend<'a> {
    memory: &'a HolographicMemory,
}

impl<'a> HoloSearchBackend<'a> {
    pub fn new(memory: &'a HolographicMemory) -> Self {
        Self { memory }
    }
}

impl SearchBackend for HoloSearchBackend<'_> {
    fn capability(&self) -> BackendCapability {
        BackendCapability {
            channel: "holo".to_string(),
            modes: vec![SearchMode::Lookup, SearchMode::Verify],
            type_expansion: false,
            certified_top_k: false,
            cost: CostClass::Low,
        }
    }

    fn search(&mut self, request: &SearchRequest) -> SearchBatch {
        let capability = self.capability();
        if !capability.supports(request.mode) {
            return SearchBatch::empty(
                "holo",
                request.mode,
                format!("全息通道不支持 {} 模式。", request.mode.label()),
            );
        }
        let constraints = &request.constraints;
        let mut raw = 0usize;
        let mut dropped = 0usize;
        let mut hits = Vec::new();
        if let (Some(subject), Some(predicate), Some(object)) = (
            constraints.subject.as_deref(),
            constraints.predicate.as_deref(),
            constraints.object.as_deref(),
        ) {
            raw = 1;
            if self.memory.contains_triple(subject, predicate, object) {
                hits.push(SearchHit {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: object.to_string(),
                    score: 1.0,
                    lexical: 0.0,
                    structural: 0.0,
                    predicate_match: 1.0,
                    verified: true,
                    channels: vec!["holo".to_string()],
                    reasons: vec!["全息硬核查命中登记三元组".to_string()],
                });
            }
        } else {
            let recalled: Vec<(String, String, String, f64, bool)> = match (
                constraints.subject.as_deref(),
                constraints.predicate.as_deref(),
                constraints.object.as_deref(),
            ) {
                (Some(subject), Some(predicate), None) => self
                    .memory
                    .recall_object(subject, predicate)
                    .into_iter()
                    .map(|hit| {
                        (
                            subject.to_string(),
                            predicate.to_string(),
                            hit.symbol,
                            hit.score,
                            hit.verified,
                        )
                    })
                    .collect(),
                (Some(subject), None, Some(object)) => self
                    .memory
                    .recall_predicate(subject, object)
                    .into_iter()
                    .map(|hit| {
                        (
                            subject.to_string(),
                            hit.symbol,
                            object.to_string(),
                            hit.score,
                            hit.verified,
                        )
                    })
                    .collect(),
                (None, Some(predicate), Some(object)) => self
                    .memory
                    .recall_subject(predicate, object)
                    .into_iter()
                    .map(|hit| {
                        (
                            hit.symbol,
                            predicate.to_string(),
                            object.to_string(),
                            hit.score,
                            hit.verified,
                        )
                    })
                    .collect(),
                _ => Vec::new(),
            };
            for (subject, predicate, object, score, verified) in recalled {
                raw += 1;
                if !verified {
                    dropped += 1;
                    continue;
                }
                if hits.len() >= request.normalized_max_results() {
                    continue;
                }
                hits.push(SearchHit {
                    subject,
                    predicate,
                    object,
                    score,
                    lexical: 0.0,
                    structural: 0.0,
                    predicate_match: 0.0,
                    verified: true,
                    channels: vec!["holo".to_string()],
                    reasons: vec!["全息软召回通过三元组硬核查".to_string()],
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.key().cmp(&right.key()))
        });
        hits.truncate(request.normalized_max_results());
        let mut notes = Vec::new();
        if raw == 0 {
            notes.push(
                "全息通道需要主体+谓词、主体+客体或谓词+客体三个约束中的至少两个。".to_string(),
            );
        }
        if dropped > 0 {
            notes.push(format!("丢弃 {} 条未通过硬核查的全息候选。", dropped));
        }
        notes.push("全息通道是软召回，不证明全局 top-k。".to_string());
        SearchBatch {
            channel: "holo".to_string(),
            mode: request.mode,
            hits: hits.clone(),
            receipt: PortReceipt {
                docs: self.memory.triple_count(),
                candidates: raw,
                scanned: raw,
                returned: hits.len(),
                top_k: request.normalized_max_results(),
                certified_top_k: false,
                channels: vec!["holo".to_string()],
                notes,
                ..PortReceipt::default()
            },
        }
    }
}

pub struct SearchPort<'a> {
    backends: Vec<Box<dyn SearchBackend + 'a>>,
}

impl Default for SearchPort<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SearchPort<'a> {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn register<B: SearchBackend + 'a>(&mut self, backend: B) {
        self.backends.push(Box::new(backend));
    }

    pub fn capabilities(&self) -> Vec<BackendCapability> {
        self.backends
            .iter()
            .map(|backend| backend.capability())
            .collect()
    }

    pub fn search(&mut self, request: &SearchRequest) -> PortOutcome {
        let batches: Vec<SearchBatch> = self
            .backends
            .iter_mut()
            .map(|backend| backend.search(request))
            .collect();
        let (hits, receipt) = merge_batches(&batches, request);
        PortOutcome {
            hits,
            receipt,
            batches,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PortOutcome {
    pub hits: Vec<SearchHit>,
    pub receipt: PortReceipt,
    pub batches: Vec<SearchBatch>,
}

fn merge_batches(
    batches: &[SearchBatch],
    request: &SearchRequest,
) -> (Vec<SearchHit>, PortReceipt) {
    let mut merged: BTreeMap<(String, String, String), SearchHit> = BTreeMap::new();
    for batch in batches {
        for hit in &batch.hits {
            merged
                .entry(hit.key())
                .and_modify(|existing| {
                    existing.score = existing.score.max(hit.score);
                    existing.lexical = existing.lexical.max(hit.lexical);
                    existing.structural = existing.structural.max(hit.structural);
                    existing.predicate_match = existing.predicate_match.max(hit.predicate_match);
                    existing.verified |= hit.verified;
                    for channel in &hit.channels {
                        if !existing.channels.contains(channel) {
                            existing.channels.push(channel.clone());
                        }
                    }
                    for reason in &hit.reasons {
                        if !existing.reasons.contains(reason) {
                            existing.reasons.push(reason.clone());
                        }
                    }
                })
                .or_insert_with(|| hit.clone());
        }
    }
    let mut hits: Vec<SearchHit> = merged.into_values().collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.key().cmp(&right.key()))
    });
    let merged_count = hits.len();
    let max_results = request.normalized_max_results();
    let truncated = merged_count > max_results;
    hits.truncate(max_results);

    let mut receipt = PortReceipt {
        docs: batches
            .iter()
            .map(|batch| batch.receipt.docs)
            .max()
            .unwrap_or(0),
        candidates: batches.iter().map(|batch| batch.receipt.candidates).sum(),
        constraint_pruned: batches
            .iter()
            .map(|batch| batch.receipt.constraint_pruned)
            .sum(),
        scanned: batches.iter().map(|batch| batch.receipt.scanned).sum(),
        returned: hits.len(),
        top_k: max_results,
        bound_pruned: batches.iter().map(|batch| batch.receipt.bound_pruned).sum(),
        budget_exhausted: batches.iter().any(|batch| batch.receipt.budget_exhausted),
        certified_top_k: batches.len() == 1 && batches[0].receipt.certified_top_k && !truncated,
        channels: batches
            .iter()
            .flat_map(|batch| batch.receipt.channels.iter().cloned())
            .collect(),
        notes: Vec::new(),
    };
    receipt.channels.sort();
    receipt.channels.dedup();
    for batch in batches {
        for note in &batch.receipt.notes {
            receipt.notes.push(format!("[{}] {}", batch.channel, note));
        }
    }
    if batches.len() > 1 {
        receipt
            .notes
            .push("多通道融合排序不宣称全局 top-k。".to_string());
    }
    if truncated {
        receipt.notes.push(format!(
            "合并后的 {} 条证据按上限截断为 {} 条。",
            merged_count, max_results
        ));
    }
    if hits.is_empty() {
        receipt
            .notes
            .push("没有找到满足约束的已核实证据。".to_string());
    }
    (hits, receipt)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FollowUpAction {
    NextPage,
    Select(usize),
    SelectLast,
    Explain,
    Exclude(String),
    Refine(String),
}

impl FollowUpAction {
    pub fn label(&self) -> &'static str {
        match self {
            Self::NextPage => "next_page",
            Self::Select(_) => "select",
            Self::SelectLast => "select_last",
            Self::Explain => "explain",
            Self::Exclude(_) => "exclude",
            Self::Refine(_) => "refine",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEntry {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
    pub channels: Vec<String>,
    pub reasons: Vec<String>,
}

impl SessionEntry {
    fn from_item(item: &EvidenceItem) -> Self {
        Self {
            id: item.id.clone(),
            subject: item.subject.clone(),
            predicate: item.predicate.clone(),
            object: item.object.clone(),
            score: item.score,
            channels: item.channels.clone(),
            reasons: item.reasons.clone(),
        }
    }

    pub fn text(&self) -> String {
        format!("{} 的 {} 是 {}", self.subject, self.predicate, self.object)
    }

    fn render_line(&self) -> String {
        let id = if self.id.is_empty() {
            "-".to_string()
        } else {
            self.id.clone()
        };
        format!("  [{}] {}", id, self.text())
    }

    fn matches(&self, needle: &str) -> bool {
        self.subject.contains(needle)
            || self.predicate.contains(needle)
            || self.object.contains(needle)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResultSet {
    pub question: String,
    pub kind: String,
    pub anchor: Option<String>,
    pub predicate: Option<String>,
    pub entries: Vec<SessionEntry>,
    pub page_size: usize,
    pub cursor: usize,
    pub receipts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchSession {
    last: Option<ResultSet>,
    page_size: usize,
    max_entries: usize,
}

impl Default for SearchSession {
    fn default() -> Self {
        Self::new(SearchSessionConfig::default())
    }
}

impl SearchSession {
    pub fn new(config: SearchSessionConfig) -> Self {
        let config = config.normalized();
        Self {
            last: None,
            page_size: config.page_size,
            max_entries: config.max_entries,
        }
    }

    pub fn clear(&mut self) {
        self.last = None;
    }

    pub fn has_result_set(&self) -> bool {
        self.last.is_some()
    }

    pub fn last(&self) -> Option<&ResultSet> {
        self.last.as_ref()
    }

    pub fn record_assembly(&mut self, question: &str, assembly: &EvidenceAssembly) -> bool {
        let Some(bundle) = &assembly.bundle else {
            return false;
        };
        if bundle.items.is_empty() {
            return false;
        }
        self.record(question, bundle, assembly.receipt.render_text())
    }

    pub fn record_search_hits(
        &mut self,
        question: &str,
        kind: &str,
        anchor: Option<String>,
        predicate: Option<String>,
        hits: &[RetrievalHit],
        receipt: String,
    ) -> bool {
        let entries: Vec<SessionEntry> = hits
            .iter()
            .take(self.max_entries)
            .enumerate()
            .map(|(index, hit)| SessionEntry {
                id: format!("E{}", index + 1),
                subject: hit.subject.clone(),
                predicate: hit.predicate.clone(),
                object: hit.object.clone(),
                score: hit.score,
                channels: vec!["dense".to_string()],
                reasons: hit.reasons.clone(),
            })
            .collect();
        if entries.is_empty() {
            return false;
        }
        self.last = Some(ResultSet {
            question: question.trim().to_string(),
            kind: kind.to_string(),
            anchor,
            predicate,
            entries,
            page_size: self.page_size,
            cursor: 0,
            receipts: vec![receipt],
        });
        true
    }

    fn record(&mut self, question: &str, bundle: &EvidenceBundle, receipt: String) -> bool {
        let entries: Vec<SessionEntry> = bundle
            .items
            .iter()
            .take(self.max_entries)
            .map(SessionEntry::from_item)
            .collect();
        if entries.is_empty() {
            return false;
        }
        self.last = Some(ResultSet {
            question: question.trim().to_string(),
            kind: bundle.kind.label().to_string(),
            anchor: bundle.anchor.clone(),
            predicate: bundle.predicate.clone(),
            entries,
            page_size: self.page_size,
            cursor: 0,
            receipts: vec![receipt],
        });
        true
    }

    pub fn respond(&mut self, action: &FollowUpAction) -> Option<String> {
        let set = self.last.as_mut()?;
        Some(match action {
            FollowUpAction::NextPage => next_page(set),
            FollowUpAction::Select(index) => select(set, *index),
            FollowUpAction::SelectLast => select(set, set.entries.len()),
            FollowUpAction::Explain => explain(set),
            FollowUpAction::Exclude(needle) => retain_matching(set, needle, false),
            FollowUpAction::Refine(needle) => retain_matching(set, needle, true),
        })
    }
}

fn next_page(set: &mut ResultSet) -> String {
    let total = set.entries.len();
    if total == 0 {
        return "上一轮结果集为空，没有可继续查看的证据。".to_string();
    }
    if set.cursor >= total {
        return format!("上一轮结果集共 {} 条，已经看完，没有更多了。", total);
    }
    let start = set.cursor;
    let end = (start + set.page_size).min(total);
    set.cursor = end;
    let mut output = format!(
        "继续查看上一轮结果集的第 {}-{} 条（共 {} 条）：\n",
        start + 1,
        end,
        total
    );
    for entry in &set.entries[start..end] {
        output.push_str(&entry.render_line());
        output.push('\n');
    }
    if end < total {
        output.push_str(&format!("还有 {} 条，可以继续说「还有呢」。", total - end));
    } else {
        output.push_str("这已是全部结果。");
    }
    output.trim_end().to_string()
}

fn select(set: &mut ResultSet, index: usize) -> String {
    let total = set.entries.len();
    if index == 0 || index > total {
        return format!(
            "上一轮结果集共 {} 条，不存在第 {} 条；可以直接说「第几条」或「还有呢」。",
            total, index
        );
    }
    set.cursor = index;
    let entry = &set.entries[index - 1];
    format!(
        "上一轮结果集的第 {} 条是：\n{}\n依据：{}。",
        index,
        entry.render_line().trim_start(),
        if entry.reasons.is_empty() {
            "证据索引中的已核实三元组".to_string()
        } else {
            entry.reasons.join("；")
        }
    )
}

fn explain(set: &ResultSet) -> String {
    let mut output = format!(
        "上一轮问句是「{}」，结果集形态为「{}」，共 {} 条已核实证据。\n",
        set.question,
        set.kind,
        set.entries.len()
    );
    for entry in &set.entries {
        output.push_str(&entry.render_line());
        if !entry.channels.is_empty() {
            output.push_str(&format!("；通道：{}", entry.channels.join("+")));
        }
        if !entry.reasons.is_empty() {
            output.push_str(&format!("；理由：{}", entry.reasons.join("；")));
        }
        output.push('\n');
    }
    for receipt in &set.receipts {
        output.push_str(receipt);
        output.push('\n');
    }
    output.trim_end().to_string()
}

fn retain_matching(set: &mut ResultSet, needle: &str, keep_matching: bool) -> String {
    let needle = needle.trim();
    if needle.is_empty() {
        return if keep_matching {
            "没有指定要细化的条件。".to_string()
        } else {
            "没有指定要排除的内容。".to_string()
        };
    }
    let before = set.entries.len();
    set.entries.retain(|entry| {
        let matched = entry.matches(needle);
        if keep_matching {
            matched
        } else {
            !matched
        }
    });
    set.cursor = 0;
    if set.entries.len() == before {
        return format!("上一轮结果集里没有包含「{}」的条目。", needle);
    }
    if set.entries.is_empty() {
        return format!("按「{}」处理后结果集为空：没有找到满足条件的证据。", needle);
    }
    let verb = if keep_matching { "细化" } else { "排除" };
    let mut output = format!(
        "按「{}」{}后还剩 {} 条（原 {} 条）：\n",
        needle,
        verb,
        set.entries.len(),
        before
    );
    for entry in &set.entries {
        output.push_str(&entry.render_line());
        output.push('\n');
    }
    output.trim_end().to_string()
}

pub fn parse_followup(question: &str) -> Option<FollowUpAction> {
    let question = question
        .trim()
        .trim_end_matches(['?', '？', '。', '!', '！', ' ']);
    if question.is_empty() {
        return None;
    }
    if let Some(action) = parse_select(question) {
        return Some(action);
    }
    if let Some(value) = parse_exclude(question) {
        return Some(FollowUpAction::Exclude(value));
    }
    if let Some(value) = parse_refine(question) {
        return Some(FollowUpAction::Refine(value));
    }
    if is_explain(question) {
        return Some(FollowUpAction::Explain);
    }
    if is_next_page(question) {
        return Some(FollowUpAction::NextPage);
    }
    None
}

fn parse_select(question: &str) -> Option<FollowUpAction> {
    if question.contains("前者") {
        return Some(FollowUpAction::Select(1));
    }
    if question.contains("后者") {
        return Some(FollowUpAction::Select(2));
    }
    if question.contains("最后")
        && ["条", "个", "项", "位", "点", "句", "页"]
            .iter()
            .any(|suffix| question.contains(suffix))
    {
        return Some(FollowUpAction::SelectLast);
    }
    let chars: Vec<char> = question.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if *ch != '第' {
            continue;
        }
        let mut cursor = index + 1;
        while cursor < chars.len() && chars[cursor].is_whitespace() {
            cursor += 1;
        }
        let start = cursor;
        while cursor < chars.len()
            && (chars[cursor].is_ascii_digit()
                || chars[cursor] == '十'
                || chinese_digit(chars[cursor]).is_some())
        {
            cursor += 1;
        }
        if cursor == start {
            continue;
        }
        let number: String = chars[start..cursor].iter().collect();
        let Some(value) = parse_ordinal(&number) else {
            continue;
        };
        let Some(suffix) = chars.get(cursor) else {
            continue;
        };
        if matches!(
            suffix,
            '个' | '条' | '项' | '位' | '点' | '名' | '则' | '句' | '页' | '张'
        ) {
            return Some(FollowUpAction::Select(value));
        }
    }
    None
}

fn parse_ordinal(text: &str) -> Option<usize> {
    if text.chars().all(|ch| ch.is_ascii_digit()) && !text.is_empty() {
        return text.parse().ok();
    }
    if text == "十" {
        return Some(10);
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() == 1 {
        return chinese_digit(chars[0]);
    }
    if let Some(index) = chars.iter().position(|ch| *ch == '十') {
        if index > 1 || chars.len() > index + 2 {
            return None;
        }
        let tens = if index == 0 {
            1
        } else {
            chinese_digit(chars[0])?
        };
        let ones = if index + 1 == chars.len() {
            0
        } else {
            chinese_digit(chars[index + 1])?
        };
        return Some(tens * 10 + ones);
    }
    None
}

fn chinese_digit(ch: char) -> Option<usize> {
    Some(match ch {
        '零' => 0,
        '一' => 1,
        '二' | '两' => 2,
        '三' => 3,
        '四' => 4,
        '五' => 5,
        '六' => 6,
        '七' => 7,
        '八' => 8,
        '九' => 9,
        _ => return None,
    })
}

fn parse_exclude(question: &str) -> Option<String> {
    let prefixes = ["除了", "除去", "排除", "去掉", "不要"];
    let mut found: Option<(usize, usize)> = None;
    for prefix in prefixes {
        if let Some(index) = question.find(prefix) {
            if found.is_none_or(|(start, _)| index < start) {
                found = Some((index, prefix.len()));
            }
        }
    }
    if found.is_none() {
        if let Some(rest) = question.strip_prefix("除") {
            if !rest.starts_with("此") {
                found = Some((0, "除".len()));
            }
        }
    }
    let (start, prefix_len) = found?;
    let rest = question[start + prefix_len..].trim_start_matches(['，', ',', '、', ' ']);
    let cut = [
        "还有", "之外", "以外", "呢", "吗", "？", "?", "，", ",", "。", ".",
    ]
    .iter()
    .filter_map(|marker| rest.find(marker))
    .min()
    .unwrap_or(rest.len());
    let value = rest[..cut]
        .trim()
        .trim_end_matches('外')
        .trim_end_matches('的')
        .trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn parse_refine(question: &str) -> Option<String> {
    let prefixes = ["其中", "只看", "只要", "仅看", "光看", "仅限"];
    for prefix in prefixes {
        let Some(index) = question.find(prefix) else {
            continue;
        };
        let rest = &question[index + prefix.len()..];
        let cut = [
            "的呢", "呢", "吗", "？", "?", "。", ".", "，", ",", "是", "有",
        ]
        .iter()
        .filter_map(|marker| rest.find(marker))
        .min()
        .unwrap_or(rest.len());
        let value = rest[..cut].trim().trim_end_matches('的').trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn is_explain(question: &str) -> bool {
    [
        "为什么",
        "依据",
        "证据",
        "出处",
        "来源",
        "解释",
        "怎么知道",
        "怎么得出",
        "凭什么",
    ]
    .iter()
    .any(|cue| question.contains(cue))
}

fn is_next_page(question: &str) -> bool {
    [
        "还有呢",
        "还有吗",
        "还有别的",
        "还有其他",
        "还有哪些",
        "还有谁",
        "还有多少",
        "继续",
        "接着",
        "下一页",
        "下一个",
        "接下来",
        "再多",
        "别的呢",
        "其他呢",
        "换一批",
        "再换",
    ]
    .iter()
    .any(|cue| question.contains(cue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual_rag::{
        AnswerExpectation, AssemblyReceipt, BundleKind, EvidenceAssembly, EvidenceBundle,
        EvidenceItem, QueryFrame,
    };

    fn dense_request() -> SearchRequest {
        SearchRequest {
            query: "师傅".to_string(),
            mode: SearchMode::Lookup,
            constraints: SearchConstraints {
                subject: Some("张三".to_string()),
                predicate: Some("师傅".to_string()),
                ..SearchConstraints::default()
            },
            max_results: 5,
            ..SearchRequest::default()
        }
    }

    fn dense() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(8, 0.85, 12);
        retriever.observe_triple("张三", "师傅", "李四");
        retriever.observe_triple("张三", "师傅", "王五");
        retriever.observe_triple("李四", "师傅", "王五");
        retriever
    }

    fn holo() -> HolographicMemory {
        let mut memory = HolographicMemory::new(512, 0.0, 6);
        memory.observe_triple("张三", "师傅", "李四");
        memory.observe_triple("张三", "师傅", "王五");
        memory
    }

    #[test]
    fn dense_backend_keeps_constraints_and_certificate() {
        let mut retriever = dense();
        let request = dense_request();
        let mut backend = DenseSearchBackend::new(&mut retriever);
        assert!(backend.capability().certified_top_k);
        let batch = backend.search(&request);
        assert_eq!(batch.channel, "dense");
        assert!(!batch.hits.is_empty());
        assert!(batch
            .hits
            .iter()
            .all(|hit| hit.subject == "张三" && hit.predicate == "师傅"));
        assert!(batch.receipt.certified_top_k);
        assert!(batch.hits.iter().all(|hit| hit.verified));
    }

    #[test]
    fn holo_backend_only_supports_lookup_and_hard_checks() {
        let memory = holo();
        let mut backend = HoloSearchBackend::new(&memory);
        let capability = backend.capability();
        assert!(capability.supports(SearchMode::Lookup));
        assert!(!capability.supports(SearchMode::Enumerate));
        assert!(!capability.certified_top_k);

        let batch = backend.search(&dense_request());
        assert!(!batch.hits.is_empty());
        assert!(batch.hits.iter().all(|hit| hit.verified));
        assert!(batch
            .hits
            .iter()
            .all(|hit| hit.subject == "张三" && hit.predicate == "师傅"));

        let unsupported = SearchRequest {
            mode: SearchMode::Enumerate,
            ..dense_request()
        };
        assert!(backend.search(&unsupported).hits.is_empty());
    }

    #[test]
    fn port_merges_channels_without_overclaiming_certificate() {
        let mut retriever = dense();
        let memory = holo();
        let mut port = SearchPort::new();
        port.register(DenseSearchBackend::new(&mut retriever));
        port.register(HoloSearchBackend::new(&memory));
        assert_eq!(port.capabilities().len(), 2);
        let outcome = port.search(&dense_request());
        assert_eq!(outcome.batches.len(), 2);
        assert!(!outcome.receipt.certified_top_k);
        let hit = outcome.hits.first().expect("应有融合证据");
        assert!(hit.channels.contains(&"dense".to_string()));
        assert!(hit.channels.contains(&"holo".to_string()));
        assert!(outcome
            .receipt
            .notes
            .iter()
            .any(|note| note.contains("多通道融合")));
    }

    fn item(id: &str, subject: &str, predicate: &str, object: &str) -> EvidenceItem {
        EvidenceItem {
            id: id.to_string(),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            score: 1.0,
            lexical: 0.5,
            structural: 0.2,
            predicate_match: 0.3,
            holographic: 0.0,
            contextual: 0.1,
            verified: true,
            novelty: true,
            channels: vec!["dense".to_string()],
            reasons: vec!["测试证据".to_string()],
            arbitration: None,
        }
    }

    fn session() -> SearchSession {
        let mut session = SearchSession::new(SearchSessionConfig {
            enabled: true,
            page_size: 2,
            max_entries: 10,
        });
        let assembly = EvidenceAssembly {
            frame: QueryFrame {
                text: "张三的师傅是谁".to_string(),
                anchors: vec!["张三".to_string()],
                explicit_anchors: vec!["张三".to_string()],
                focus: Vec::new(),
                predicate: Some("师傅".to_string()),
                expectation: AnswerExpectation::Person,
                continuation: false,
                referenced: Vec::new(),
                notes: Vec::new(),
            },
            bundle: Some(EvidenceBundle {
                kind: BundleKind::SlotList,
                anchor: Some("张三".to_string()),
                secondary_anchor: None,
                predicate: Some("师傅".to_string()),
                expectation: AnswerExpectation::Person,
                items: vec![
                    item("E1", "张三", "师傅", "李四"),
                    item("E2", "张三", "师傅", "王五"),
                    item("E3", "张三", "师傅", "赵六"),
                ],
                multiple_values: Vec::new(),
                coverage: 1.0,
                answer: "测试回答".to_string(),
                arbitration: None,
            }),
            receipt: AssemblyReceipt::default(),
        };
        assert!(session.record_assembly("张三的师傅是谁", &assembly));
        session
    }

    #[test]
    fn question_compiles_to_mode_constraints_and_expectation() {
        assert_eq!(
            SearchMode::from_question("张三的师傅是谁"),
            SearchMode::Lookup
        );
        assert_eq!(
            SearchMode::from_question("张三的师傅有哪些"),
            SearchMode::Enumerate
        );
        assert_eq!(
            SearchMode::from_question("介绍一下张三"),
            SearchMode::Describe
        );
        assert_eq!(
            SearchMode::from_question("张三的师傅是李四吗"),
            SearchMode::Verify
        );
        assert_eq!(
            SearchMode::from_question("什么是知识图谱"),
            SearchMode::Lookup
        );

        let request = SearchRequest::from_question(
            "张三有几个师傅",
            Some("张三".to_string()),
            Some("师傅".to_string()),
            4,
        );
        assert_eq!(request.mode, SearchMode::Enumerate);
        assert_eq!(request.constraints.subject.as_deref(), Some("张三"));
        assert_eq!(request.constraints.predicate.as_deref(), Some("师傅"));
        assert_eq!(request.expectation, AnswerExpectation::Number);
        assert_eq!(request.max_results, 4);
    }

    #[test]
    fn followup_parser_covers_result_set_actions() {
        assert_eq!(parse_followup("还有呢"), Some(FollowUpAction::NextPage));
        assert_eq!(parse_followup("下一个"), Some(FollowUpAction::NextPage));
        assert_eq!(
            parse_followup("第二个是谁"),
            Some(FollowUpAction::Select(2))
        );
        assert_eq!(parse_followup("第十二条"), Some(FollowUpAction::Select(12)));
        assert_eq!(parse_followup("最后一条"), Some(FollowUpAction::SelectLast));
        assert_eq!(
            parse_followup("除了李四还有谁"),
            Some(FollowUpAction::Exclude("李四".to_string()))
        );
        assert_eq!(
            parse_followup("除李四外还有谁"),
            Some(FollowUpAction::Exclude("李四".to_string()))
        );
        assert_eq!(
            parse_followup("其中姓李的呢"),
            Some(FollowUpAction::Refine("姓李".to_string()))
        );
        assert_eq!(parse_followup("为什么"), Some(FollowUpAction::Explain));
        assert_eq!(parse_followup("张三的师傅是谁"), None);
    }

    #[test]
    fn session_selects_pages_excludes_and_explains() {
        let mut session = session();
        let selected = session
            .respond(&FollowUpAction::Select(2))
            .expect("应有上一轮结果集");
        assert!(selected.contains("王五"), "{}", selected);
        assert!(selected.contains("第 2 条"), "{}", selected);

        let page = session
            .respond(&FollowUpAction::NextPage)
            .expect("应有下一页");
        assert!(page.contains("赵六"), "{}", page);
        assert!(page.contains("这已是全部结果"), "{}", page);
        let exhausted = session
            .respond(&FollowUpAction::NextPage)
            .expect("翻完也应回执");
        assert!(exhausted.contains("没有更多"), "{}", exhausted);

        let excluded = session
            .respond(&FollowUpAction::Exclude("王五".to_string()))
            .expect("应可排除");
        assert!(!excluded.contains("[E2]"), "{}", excluded);
        assert!(excluded.contains("[E1]"), "{}", excluded);
        assert!(excluded.contains("[E3]"), "{}", excluded);

        let refined = session
            .respond(&FollowUpAction::Refine("赵六".to_string()))
            .expect("应可细化");
        assert!(refined.contains("赵六"), "{}", refined);
        assert!(!refined.contains("李四"), "{}", refined);

        let explained = session.respond(&FollowUpAction::Explain).expect("应可解释");
        assert!(explained.contains("张三的师傅是谁"), "{}", explained);
        assert!(explained.contains("证据装配回执"), "{}", explained);
    }

    #[test]
    fn session_reports_out_of_range_and_no_result_set() {
        let mut empty = SearchSession::default();
        assert!(empty.respond(&FollowUpAction::NextPage).is_none());

        let mut session = session();
        let out = session
            .respond(&FollowUpAction::Select(9))
            .expect("越界也应回执");
        assert!(out.contains("不存在第 9 条"), "{}", out);
        let missing = session
            .respond(&FollowUpAction::Exclude("不存在".to_string()))
            .expect("未命中应诚实说明");
        assert!(missing.contains("没有包含"), "{}", missing);
    }
}

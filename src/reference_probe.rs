use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::contextual_reference::{
    rewrite_at, ClarifyOption, ClarifyQuestion, ClarifyRequest, DiscriminatingFeature,
    MentionResolution,
};

pub const GROUNDING_CERTIFICATE_VERSION: u32 = 2;
pub const MAX_PROBE_CANDIDATES: usize = 3;
pub const MAX_GROUNDING_CERTIFICATES: usize = 32;
pub const MAX_ANSWER_PREVIEW_CHARS: usize = 48;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    Answered(Vec<String>),
    NoAnswer,
    Truncated,
    Unsupported,
}

impl ProbeStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ProbeStatus::Answered(_) => "answered",
            ProbeStatus::NoAnswer => "no_answer",
            ProbeStatus::Truncated => "truncated",
            ProbeStatus::Unsupported => "unsupported",
        }
    }

    fn unresolved(&self) -> bool {
        matches!(self, ProbeStatus::Truncated | ProbeStatus::Unsupported)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeSensitivity {
    Stable,
    LinkSensitive,
    Repairable,
    OutcomeAmbiguous,
    Uninformative,
}

impl ProbeSensitivity {
    pub fn label(self) -> &'static str {
        match self {
            ProbeSensitivity::Stable => "stable",
            ProbeSensitivity::LinkSensitive => "link_sensitive",
            ProbeSensitivity::Repairable => "repairable",
            ProbeSensitivity::OutcomeAmbiguous => "outcome_ambiguous",
            ProbeSensitivity::Uninformative => "uninformative",
        }
    }

    pub fn render(self) -> &'static str {
        match self {
            ProbeSensitivity::Stable => "稳定（答案不随指称选择改变）",
            ProbeSensitivity::LinkSensitive => "链接敏感（答案取决于指称选择）",
            ProbeSensitivity::Repairable => "可修复（当前义项无答案，替代义项可答）",
            ProbeSensitivity::OutcomeAmbiguous => "结果歧义（无法仅凭探测裁决）",
            ProbeSensitivity::Uninformative => "无信息（所有候选均无答案）",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeDecision {
    Bind {
        entity: String,
        reason: String,
    },
    Repair {
        entity: String,
        reason: String,
    },
    Clarify {
        reason: String,
    },
}

impl ProbeDecision {
    pub fn action(&self) -> &'static str {
        match self {
            ProbeDecision::Bind { .. } => "bind",
            ProbeDecision::Repair { .. } => "repair",
            ProbeDecision::Clarify { .. } => "clarify",
        }
    }

    pub fn entity(&self) -> Option<&str> {
        match self {
            ProbeDecision::Bind { entity, .. } | ProbeDecision::Repair { entity, .. } => {
                Some(entity)
            }
            ProbeDecision::Clarify { .. } => None,
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            ProbeDecision::Bind { reason, .. }
            | ProbeDecision::Repair { reason, .. }
            | ProbeDecision::Clarify { reason } => reason,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedCandidate {
    pub rank: usize,
    pub entity: String,
    pub votes: i32,
    pub rewritten: String,
    pub status: ProbeStatus,
    pub answers: Vec<String>,
    pub answer_fingerprint: Option<String>,
}

impl ProbedCandidate {
    fn answerable(&self) -> bool {
        !self.answers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateCandidate {
    pub rank: usize,
    pub entity: String,
    pub votes: i32,
    pub status: String,
    pub answer_count: usize,
    pub answer_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rewritten: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundingCertificate {
    pub version: u32,
    pub revision: u64,
    pub question_digest: String,
    pub mention: String,
    pub start: usize,
    pub key: String,
    pub sensitivity: String,
    pub action: String,
    pub entity: Option<String>,
    pub candidates: Vec<CertificateCandidate>,
    pub certificate_digest: String,
}

impl GroundingCertificate {
    pub fn verify(&self) -> bool {
        self.certificate_digest == self.compute_digest()
    }

    pub fn canonical_text(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        canonical_json(&value)
    }

    fn compute_digest(&self) -> String {
        let mut body = self.clone();
        body.certificate_digest.clear();
        let value = serde_json::to_value(&body).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "指称接地证书 v{}（知识版本 {}，摘要 {}）",
            self.version, self.revision, self.certificate_digest
        );
        let _ = writeln!(
            out,
            "  指称「{}」@{}，键「{}」",
            self.mention, self.start, self.key
        );
        let _ = writeln!(out, "  敏感度：{}", self.sensitivity);
        let _ = writeln!(
            out,
            "  动作：{}{}",
            self.action,
            self.entity
                .as_deref()
                .map(|entity| format!("（{}）", entity))
                .unwrap_or_default()
        );
        for candidate in &self.candidates {
            let fingerprint = candidate
                .answer_fingerprint
                .as_deref()
                .map(|fingerprint| format!("，指纹 {}", fingerprint))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "    {}. {} [{} 票] {}（{} 项答案{}）",
                candidate.rank,
                candidate.entity,
                candidate.votes,
                candidate.status,
                candidate.answer_count,
                fingerprint
            );
        }
        out.trim_end().to_string()
    }

    /// 在当前知识版本上重放证书中的候选探测，核对答案指纹、敏感度与动作。
    /// 旧版证书若没有保存改写问题，则标记为不可完整重放，而不是误报一致。
    pub fn replay<F>(&self, current_revision: u64, mut probe: F) -> GroundingReplayReport
    where
        F: FnMut(&str, &str) -> ProbeStatus,
    {
        let mut candidates = Vec::with_capacity(self.candidates.len());
        let mut observations = Vec::new();
        let mut replayable = true;
        for candidate in &self.candidates {
            if candidate.rewritten.trim().is_empty() {
                replayable = false;
                candidates.push(ReplayCandidate {
                    rank: candidate.rank,
                    entity: candidate.entity.clone(),
                    expected_status: candidate.status.clone(),
                    actual_status: "unavailable".to_string(),
                    expected_fingerprint: candidate.answer_fingerprint.clone(),
                    actual_fingerprint: None,
                    matches: false,
                });
                continue;
            }
            let status = normalize_status(probe(&candidate.entity, &candidate.rewritten));
            let answers = answer_values(&status);
            let answer_fingerprint = fingerprint_answers(&answers);
            let matches = status.label() == candidate.status
                && answer_fingerprint == candidate.answer_fingerprint;
            candidates.push(ReplayCandidate {
                rank: candidate.rank,
                entity: candidate.entity.clone(),
                expected_status: candidate.status.clone(),
                actual_status: status.label().to_string(),
                expected_fingerprint: candidate.answer_fingerprint.clone(),
                actual_fingerprint: answer_fingerprint.clone(),
                matches,
            });
            observations.push(ProbedCandidate {
                rank: candidate.rank,
                entity: candidate.entity.clone(),
                votes: candidate.votes,
                rewritten: candidate.rewritten.clone(),
                status,
                answers,
                answer_fingerprint,
            });
        }

        let (actual_sensitivity, actual_decision) = if observations.is_empty() {
            (
                ProbeSensitivity::Uninformative,
                ProbeDecision::Clarify {
                    reason: "没有可重放的候选".to_string(),
                },
            )
        } else {
            let (sensitivity, decision, _) = classify(&observations);
            (sensitivity, decision)
        };
        let actual_action = actual_decision.action().to_string();
        let action_matches =
            actual_action == self.action && actual_decision.entity() == self.entity.as_deref();
        let sensitivity_matches = actual_sensitivity.label() == self.sensitivity;
        let candidate_matches = candidates.iter().all(|candidate| candidate.matches);
        let matches = replayable && candidate_matches && action_matches && sensitivity_matches;
        let revision_changed = current_revision != self.revision;

        let mut notes = Vec::new();
        if revision_changed {
            notes.push(format!(
                "知识版本已从 {} 变为 {}",
                self.revision, current_revision
            ));
        }
        if !replayable {
            notes.push("证书缺少改写问题（旧版证书），无法完整重放".to_string());
        } else if matches {
            notes.push("候选答案指纹、敏感度与动作均与证书一致".to_string());
        } else {
            notes.push("重放结果与证书不一致：知识库或探测行为已变化".to_string());
        }
        if !action_matches {
            notes.push(format!(
                "动作不一致：证书 {} → 重放 {}",
                self.action, actual_action
            ));
        }
        if !sensitivity_matches {
            notes.push(format!(
                "敏感度不一致：证书 {} → 重放 {}",
                self.sensitivity,
                actual_sensitivity.label()
            ));
        }

        GroundingReplayReport {
            certificate_digest: self.certificate_digest.clone(),
            certificate_revision: self.revision,
            current_revision,
            revision_changed,
            replayable,
            matches,
            expected_sensitivity: self.sensitivity.clone(),
            actual_sensitivity: actual_sensitivity.label().to_string(),
            expected_action: self.action.clone(),
            actual_action,
            candidates,
            notes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCandidate {
    pub rank: usize,
    pub entity: String,
    pub expected_status: String,
    pub actual_status: String,
    pub expected_fingerprint: Option<String>,
    pub actual_fingerprint: Option<String>,
    pub matches: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundingReplayReport {
    pub certificate_digest: String,
    pub certificate_revision: u64,
    pub current_revision: u64,
    pub revision_changed: bool,
    pub replayable: bool,
    pub matches: bool,
    pub expected_sensitivity: String,
    pub actual_sensitivity: String,
    pub expected_action: String,
    pub actual_action: String,
    pub candidates: Vec<ReplayCandidate>,
    pub notes: Vec<String>,
}

impl GroundingReplayReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "指称接地证书重放：摘要 {}（知识版本 {} → {}）",
            self.certificate_digest, self.certificate_revision, self.current_revision
        );
        let _ = writeln!(
            out,
            "  结果：{}",
            if !self.replayable {
                "不可完整重放"
            } else if self.matches {
                "与证书一致"
            } else {
                "与证书不一致"
            }
        );
        let _ = writeln!(
            out,
            "  敏感度：{} → {}",
            self.expected_sensitivity, self.actual_sensitivity
        );
        let _ = writeln!(
            out,
            "  动作：{} → {}",
            self.expected_action, self.actual_action
        );
        for candidate in &self.candidates {
            let _ = writeln!(
                out,
                "    {}. {} {} → {}，指纹 {} → {}（{}）",
                candidate.rank,
                candidate.entity,
                candidate.expected_status,
                candidate.actual_status,
                candidate.expected_fingerprint.as_deref().unwrap_or("-"),
                candidate.actual_fingerprint.as_deref().unwrap_or("-"),
                if candidate.matches {
                    "一致"
                } else if candidate.actual_status == "unavailable" {
                    "不可重放"
                } else {
                    "变化"
                }
            );
        }
        for note in &self.notes {
            let _ = writeln!(out, "  - {}", note);
        }
        out.trim_end().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceProbeReport {
    pub mention: String,
    pub start: usize,
    pub sensitivity: ProbeSensitivity,
    pub decision: ProbeDecision,
    pub observations: Vec<ProbedCandidate>,
    pub question: ClarifyQuestion,
    pub notes: Vec<String>,
    pub certificate: GroundingCertificate,
}

impl ReferenceProbeReport {
    pub fn render_text(&self) -> String {
        let mut out = self.certificate.render_text();
        out.push('\n');
        for note in &self.notes {
            let _ = writeln!(out, "  - {}", note);
        }
        out.trim_end().to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingLedger {
    #[serde(default = "ledger_version")]
    version: u32,
    #[serde(default)]
    entries: Vec<GroundingCertificate>,
}

fn ledger_version() -> u32 {
    GROUNDING_CERTIFICATE_VERSION
}

impl Default for GroundingLedger {
    fn default() -> Self {
        Self {
            version: ledger_version(),
            entries: Vec::new(),
        }
    }
}

impl GroundingLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[GroundingCertificate] {
        &self.entries
    }

    pub fn latest(&self) -> Option<&GroundingCertificate> {
        self.entries.last()
    }

    pub fn push(&mut self, certificate: GroundingCertificate) {
        if self
            .entries
            .iter()
            .any(|entry| entry.certificate_digest == certificate.certificate_digest)
        {
            return;
        }
        self.entries.push(certificate);
        if self.entries.len() > MAX_GROUNDING_CERTIFICATES {
            let excess = self.entries.len() - MAX_GROUNDING_CERTIFICATES;
            self.entries.drain(0..excess);
        }
    }

    pub fn report(&self) -> String {
        let mut out = format!(
            "指称接地证书台账：{} 条（保留最近 {} 条）\n",
            self.entries.len(),
            MAX_GROUNDING_CERTIFICATES
        );
        match self.latest() {
            Some(certificate) => {
                out.push_str(&certificate.render_text());
            }
            None => {
                out.push_str("  （空：发生一次答案探测式消歧后，这里会留下证书）");
            }
        }
        out.trim_end().to_string()
    }

    pub fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn from_snapshot_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|error| error.to_string())
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        let text = self.snapshot_json().map_err(std::io::Error::other)?;
        std::fs::write(path, text)
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_snapshot_json(&text).ok()
    }
}

pub fn probe_clarification<F>(
    request: &ClarifyRequest,
    resolution: Option<&MentionResolution>,
    revision: u64,
    max_candidates: usize,
    mut probe: F,
) -> ReferenceProbeReport
where
    F: FnMut(&str, &str) -> ProbeStatus,
{
    let limit = max_candidates.clamp(1, MAX_PROBE_CANDIDATES);
    let mut observations = Vec::new();
    for (index, option) in request.question.options.iter().take(limit).enumerate() {
        let rewritten = rewrite_at(
            &request.original_input,
            request.start,
            request.mention.chars().count(),
            &option.entity,
        );
        let status = normalize_status(probe(&option.entity, &rewritten));
        let answers = answer_values(&status);
        let answer_fingerprint = fingerprint_answers(&answers);
        let votes = resolution
            .and_then(|resolution| {
                resolution
                    .candidates
                    .iter()
                    .find(|candidate| candidate.entity == option.entity)
            })
            .map(|candidate| candidate.votes)
            .unwrap_or(0);
        observations.push(ProbedCandidate {
            rank: index + 1,
            entity: option.entity.clone(),
            votes,
            rewritten,
            status,
            answers,
            answer_fingerprint,
        });
    }

    let (sensitivity, decision, notes) = classify(&observations);
    let question = enrich_question(request, &observations);
    let certificate = certificate_for(
        request,
        resolution,
        revision,
        sensitivity,
        &decision,
        &observations,
    );

    ReferenceProbeReport {
        mention: request.mention.clone(),
        start: request.start,
        sensitivity,
        decision,
        observations,
        question,
        notes,
        certificate,
    }
}

fn normalize_status(status: ProbeStatus) -> ProbeStatus {
    match status {
        ProbeStatus::Answered(values) => {
            let normalized = normalize_answers(&values);
            if normalized.is_empty() {
                ProbeStatus::NoAnswer
            } else {
                ProbeStatus::Answered(normalized)
            }
        }
        other => other,
    }
}

fn normalize_answers(values: &[String]) -> Vec<String> {
    let mut normalized: Vec<String> = values
        .iter()
        .filter_map(|value| {
            let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
            let trimmed = collapsed.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn answer_values(status: &ProbeStatus) -> Vec<String> {
    match status {
        ProbeStatus::Answered(values) => normalize_answers(values),
        _ => Vec::new(),
    }
}

fn fingerprint_answers(answers: &[String]) -> Option<String> {
    if answers.is_empty() {
        return None;
    }
    let mut input = String::new();
    for answer in answers {
        input.push_str(answer);
        input.push('\u{1}');
    }
    Some(fnv1a_hex(input.as_bytes()))
}

fn classify(
    observations: &[ProbedCandidate],
) -> (ProbeSensitivity, ProbeDecision, Vec<String>) {
    if observations.is_empty() {
        return (
            ProbeSensitivity::Uninformative,
            ProbeDecision::Clarify {
                reason: "没有可探测的候选".to_string(),
            },
            vec!["指称探测：没有可探测的候选，保留澄清".to_string()],
        );
    }

    let any_unresolved = observations.iter().any(|obs| obs.status.unresolved());
    let answerable: Vec<&ProbedCandidate> = observations
        .iter()
        .filter(|obs| obs.answerable())
        .collect();
    let top_answerable = observations[0].answerable();
    let distinct: BTreeSet<&str> = answerable
        .iter()
        .filter_map(|obs| obs.answer_fingerprint.as_deref())
        .collect();

    let sensitivity = if any_unresolved {
        ProbeSensitivity::OutcomeAmbiguous
    } else if answerable.is_empty() {
        ProbeSensitivity::Uninformative
    } else if distinct.len() > 1 {
        if top_answerable {
            ProbeSensitivity::LinkSensitive
        } else {
            ProbeSensitivity::OutcomeAmbiguous
        }
    } else if top_answerable && answerable.len() == observations.len() {
        ProbeSensitivity::Stable
    } else if top_answerable {
        ProbeSensitivity::LinkSensitive
    } else {
        ProbeSensitivity::Repairable
    };

    let decision = if any_unresolved {
        ProbeDecision::Clarify {
            reason: "存在截断或不支持的探测，不能据此宣称不可答".to_string(),
        }
    } else if answerable.is_empty() {
        ProbeDecision::Clarify {
            reason: "所有候选均无答案，保留澄清且不宣称事实不存在".to_string(),
        }
    } else if distinct.len() > 1 {
        ProbeDecision::Clarify {
            reason: "候选义项给出不同答案，保留澄清".to_string(),
        }
    } else if top_answerable {
        ProbeDecision::Bind {
            entity: observations[0].entity.clone(),
            reason: if answerable.len() == observations.len() {
                "所有可答义项给出同一答案".to_string()
            } else {
                "只有最高分义项可答".to_string()
            },
        }
    } else {
        ProbeDecision::Repair {
            entity: answerable[0].entity.clone(),
            reason: "最高分义项无答案，替代义项可答".to_string(),
        }
    };

    let notes = match &decision {
        ProbeDecision::Bind { entity, reason } => vec![format!(
            "指称探测：{}，静默采用「{}」",
            reason, entity
        )],
        ProbeDecision::Repair { entity, reason } => {
            let preview = answerable
                .iter()
                .find(|obs| obs.entity == *entity)
                .map(|obs| preview_answers(&obs.answers))
                .unwrap_or_default();
            vec![format!(
                "指称探测：{}；按当前知识库「{}」可答：{}。要按「{}」理解吗？",
                reason, entity, preview, entity
            )]
        }
        ProbeDecision::Clarify { reason } => {
            vec![format!("指称探测：{}", reason)]
        }
    };

    (sensitivity, decision, notes)
}

fn preview_answers(answers: &[String]) -> String {
    let joined = answers.join("、");
    let mut chars = joined.chars();
    let preview: String = chars.by_ref().take(MAX_ANSWER_PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        format!("{}…", preview)
    } else {
        preview
    }
}

fn enrich_question(
    request: &ClarifyRequest,
    observations: &[ProbedCandidate],
) -> ClarifyQuestion {
    let mut question = request.question.clone();
    for option in &mut question.options {
        let Some(observation) = observations
            .iter()
            .find(|observation| observation.entity == option.entity)
        else {
            continue;
        };
        let suffix = match &observation.status {
            ProbeStatus::Answered(_) if observation.answerable() => {
                format!(" → {}", preview_answers(&observation.answers))
            }
            ProbeStatus::Answered(_) | ProbeStatus::NoAnswer => {
                "（当前知识库无答案）".to_string()
            }
            ProbeStatus::Truncated => "（探测被截断）".to_string(),
            ProbeStatus::Unsupported => "（不支持探测）".to_string(),
        };
        option.label = format!("{}{}", option.label, suffix);
    }
    question
}

fn certificate_for(
    request: &ClarifyRequest,
    resolution: Option<&MentionResolution>,
    revision: u64,
    sensitivity: ProbeSensitivity,
    decision: &ProbeDecision,
    observations: &[ProbedCandidate],
) -> GroundingCertificate {
    let candidates = observations
        .iter()
        .map(|observation| CertificateCandidate {
            rank: observation.rank,
            entity: observation.entity.clone(),
            votes: observation.votes,
            status: observation.status.label().to_string(),
            answer_count: observation.answers.len(),
            answer_fingerprint: observation.answer_fingerprint.clone(),
            rewritten: observation.rewritten.clone(),
        })
        .collect();
    let key = resolution
        .map(|resolution| resolution.key.clone())
        .unwrap_or_else(|| request.key.clone());
    let mut certificate = GroundingCertificate {
        version: GROUNDING_CERTIFICATE_VERSION,
        revision,
        question_digest: fnv1a_hex(request.original_input.as_bytes()),
        mention: request.mention.clone(),
        start: request.start,
        key,
        sensitivity: sensitivity.label().to_string(),
        action: decision.action().to_string(),
        entity: decision.entity().map(str::to_string),
        candidates,
        certificate_digest: String::new(),
    };
    certificate.certificate_digest = certificate.compute_digest();
    certificate
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptedProbe {
    Answers(&'static [&'static str]),
    NoAnswer,
    Truncated,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldProbeAction {
    Bind(&'static str),
    Repair(&'static str),
    Clarify,
}

#[derive(Debug, Clone, Copy)]
pub struct ProbeGoldCase {
    pub name: &'static str,
    pub question: &'static str,
    pub mention: &'static str,
    pub start: usize,
    pub options: &'static [&'static str],
    pub script: &'static [(&'static str, ScriptedProbe)],
    pub expected_action: GoldProbeAction,
    pub expected_sensitivity: ProbeSensitivity,
}

pub fn seed_cases() -> &'static [ProbeGoldCase] {
    static SEED: &[ProbeGoldCase] = &[
        ProbeGoldCase {
            name: "同一答案静默绑定",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::Answers(&["红色"])),
            ],
            expected_action: GoldProbeAction::Bind("苹果"),
            expected_sensitivity: ProbeSensitivity::Stable,
        },
        ProbeGoldCase {
            name: "只有最高分义项可答",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::NoAnswer),
            ],
            expected_action: GoldProbeAction::Bind("苹果"),
            expected_sensitivity: ProbeSensitivity::LinkSensitive,
        },
        ProbeGoldCase {
            name: "不同答案保留澄清",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::Answers(&["库克"])),
            ],
            expected_action: GoldProbeAction::Clarify,
            expected_sensitivity: ProbeSensitivity::LinkSensitive,
        },
        ProbeGoldCase {
            name: "替代义项可修复",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::NoAnswer),
                ("苹果公司", ScriptedProbe::Answers(&["库克"])),
            ],
            expected_action: GoldProbeAction::Repair("苹果公司"),
            expected_sensitivity: ProbeSensitivity::Repairable,
        },
        ProbeGoldCase {
            name: "多个替代义项同答",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司", "苹果唱片"],
            script: &[
                ("苹果", ScriptedProbe::NoAnswer),
                ("苹果公司", ScriptedProbe::Answers(&["库克"])),
                ("苹果唱片", ScriptedProbe::Answers(&["库克"])),
            ],
            expected_action: GoldProbeAction::Repair("苹果公司"),
            expected_sensitivity: ProbeSensitivity::Repairable,
        },
        ProbeGoldCase {
            name: "全部无答案不宣称不存在",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::NoAnswer),
                ("苹果公司", ScriptedProbe::NoAnswer),
            ],
            expected_action: GoldProbeAction::Clarify,
            expected_sensitivity: ProbeSensitivity::Uninformative,
        },
        ProbeGoldCase {
            name: "截断不得宣称稳定",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::Truncated),
            ],
            expected_action: GoldProbeAction::Clarify,
            expected_sensitivity: ProbeSensitivity::OutcomeAmbiguous,
        },
        ProbeGoldCase {
            name: "替代义项答案冲突",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司", "苹果唱片"],
            script: &[
                ("苹果", ScriptedProbe::NoAnswer),
                ("苹果公司", ScriptedProbe::Answers(&["库克"])),
                ("苹果唱片", ScriptedProbe::Answers(&["唱片"])),
            ],
            expected_action: GoldProbeAction::Clarify,
            expected_sensitivity: ProbeSensitivity::OutcomeAmbiguous,
        },
        ProbeGoldCase {
            name: "空白差异仍是同一答案",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色 多汁"])),
                ("苹果公司", ScriptedProbe::Answers(&["红色\n多汁"])),
            ],
            expected_action: GoldProbeAction::Bind("苹果"),
            expected_sensitivity: ProbeSensitivity::Stable,
        },
        ProbeGoldCase {
            name: "不支持探测不得静默绑定",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::Unsupported),
            ],
            expected_action: GoldProbeAction::Clarify,
            expected_sensitivity: ProbeSensitivity::OutcomeAmbiguous,
        },
        ProbeGoldCase {
            name: "三个义项同答",
            question: "苹果是什么",
            mention: "苹果",
            start: 0,
            options: &["苹果", "苹果公司", "苹果唱片"],
            script: &[
                ("苹果", ScriptedProbe::Answers(&["红色"])),
                ("苹果公司", ScriptedProbe::Answers(&["红色"])),
                ("苹果唱片", ScriptedProbe::Answers(&["红色"])),
            ],
            expected_action: GoldProbeAction::Bind("苹果"),
            expected_sensitivity: ProbeSensitivity::Stable,
        },
    ];
    SEED
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundingEvalReport {
    pub corpus_digest: String,
    pub cases: u32,
    pub action_accuracy: f64,
    pub sensitivity_accuracy: f64,
    pub repair_recall: f64,
    pub over_ask_rate: f64,
    pub silent_bind_rate: f64,
    pub truncation_safety: f64,
    pub preview_coverage: f64,
    pub certificate_replay_rate: f64,
}

impl GroundingEvalReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "指称接地评测：{} 个金标场景，语料摘要 {}。",
            self.cases, self.corpus_digest
        );
        let _ = writeln!(
            out,
            "action_accuracy {:.4}；sensitivity_accuracy {:.4}；repair_recall {:.4}",
            self.action_accuracy, self.sensitivity_accuracy, self.repair_recall
        );
        let _ = writeln!(
            out,
            "over_ask_rate {:.4}；silent_bind_rate {:.4}；truncation_safety {:.4}",
            self.over_ask_rate, self.silent_bind_rate, self.truncation_safety
        );
        let _ = writeln!(
            out,
            "preview_coverage {:.4}；certificate_replay_rate {:.4}",
            self.preview_coverage, self.certificate_replay_rate
        );
        out.trim_end().to_string()
    }

    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "grounding.action_accuracy".to_string(),
            self.action_accuracy,
        );
        metrics.insert(
            "grounding.sensitivity_accuracy".to_string(),
            self.sensitivity_accuracy,
        );
        metrics.insert("grounding.repair_recall".to_string(), self.repair_recall);
        metrics.insert("grounding.over_ask_rate".to_string(), self.over_ask_rate);
        metrics.insert(
            "grounding.silent_bind_rate".to_string(),
            self.silent_bind_rate,
        );
        metrics.insert(
            "grounding.truncation_safety".to_string(),
            self.truncation_safety,
        );
        metrics.insert(
            "grounding.preview_coverage".to_string(),
            self.preview_coverage,
        );
        metrics.insert(
            "grounding.certificate_replay_rate".to_string(),
            self.certificate_replay_rate,
        );
        metrics
    }
}

pub const GROUNDING_LOWER_IS_BETTER: [&str; 2] = [
    "grounding.over_ask_rate",
    "grounding.silent_bind_rate",
];

fn eval_ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn case_request(case: &ProbeGoldCase) -> ClarifyRequest {
    ClarifyRequest {
        original_input: case.question.to_string(),
        mention: case.mention.to_string(),
        start: case.start,
        key: case.mention.to_string(),
        question: ClarifyQuestion {
            mention: case.mention.to_string(),
            feature: DiscriminatingFeature::Names,
            options: case
                .options
                .iter()
                .map(|entity| ClarifyOption {
                    entity: (*entity).to_string(),
                    label: (*entity).to_string(),
                    etype: None,
                })
                .collect(),
        },
    }
}

fn scripted_status(case: &ProbeGoldCase, entity: &str) -> ProbeStatus {
    match case
        .script
        .iter()
        .find(|(name, _)| *name == entity)
        .map(|(_, script)| *script)
    {
        Some(ScriptedProbe::Answers(values)) => {
            ProbeStatus::Answered(values.iter().map(|value| (*value).to_string()).collect())
        }
        Some(ScriptedProbe::Truncated) => ProbeStatus::Truncated,
        Some(ScriptedProbe::Unsupported) => ProbeStatus::Unsupported,
        Some(ScriptedProbe::NoAnswer) | None => ProbeStatus::NoAnswer,
    }
}

fn action_matches(expected: GoldProbeAction, decision: &ProbeDecision) -> bool {
    match (expected, decision) {
        (GoldProbeAction::Bind(expected), ProbeDecision::Bind { entity, .. }) => expected == entity,
        (GoldProbeAction::Repair(expected), ProbeDecision::Repair { entity, .. }) => {
            expected == entity
        }
        (GoldProbeAction::Clarify, ProbeDecision::Clarify { .. }) => true,
        _ => false,
    }
}

pub fn evaluate_seed() -> GroundingEvalReport {
    evaluate(seed_cases())
}

pub fn evaluate(cases: &[ProbeGoldCase]) -> GroundingEvalReport {
    let mut digest_input = String::new();
    let mut action_hits = 0u32;
    let mut sensitivity_hits = 0u32;
    let mut repair_expected = 0u32;
    let mut repair_hits = 0u32;
    let mut bind_expected = 0u32;
    let mut over_asks = 0u32;
    let mut clarify_expected = 0u32;
    let mut silent_binds = 0u32;
    let mut unresolved_cases = 0u32;
    let mut truncation_safe = 0u32;
    let mut preview_expected = 0u32;
    let mut preview_hits = 0u32;
    let mut replay_hits = 0u32;

    for case in cases {
        digest_input.push_str(case.name);
        digest_input.push('\u{1}');
        digest_input.push_str(case.question);
        digest_input.push('\u{2}');
        let request = case_request(case);
        let report = probe_clarification(&request, None, 1, MAX_PROBE_CANDIDATES, |entity, _| {
            scripted_status(case, entity)
        });
        let replay = report
            .certificate
            .replay(1, |entity, _| scripted_status(case, entity));
        if action_matches(case.expected_action, &report.decision) {
            action_hits += 1;
        }
        if report.sensitivity == case.expected_sensitivity {
            sensitivity_hits += 1;
        }
        match case.expected_action {
            GoldProbeAction::Bind(_) => {
                bind_expected += 1;
                if matches!(report.decision, ProbeDecision::Clarify { .. }) {
                    over_asks += 1;
                }
            }
            GoldProbeAction::Repair(_) => {
                repair_expected += 1;
                if matches!(report.decision, ProbeDecision::Repair { .. }) {
                    repair_hits += 1;
                }
            }
            GoldProbeAction::Clarify => {
                clarify_expected += 1;
                if matches!(report.decision, ProbeDecision::Bind { .. }) {
                    silent_binds += 1;
                }
            }
        }
        let unresolved = case
            .script
            .iter()
            .any(|(_, script)| matches!(script, ScriptedProbe::Truncated | ScriptedProbe::Unsupported));
        if unresolved {
            unresolved_cases += 1;
            if !matches!(report.decision, ProbeDecision::Bind { .. }) {
                truncation_safe += 1;
            }
        }
        let has_answer = report
            .observations
            .iter()
            .any(|observation| observation.answerable());
        if has_answer && !matches!(case.expected_action, GoldProbeAction::Bind(_)) {
            preview_expected += 1;
            if report
                .question
                .options
                .iter()
                .any(|option| option.label.contains("→"))
            {
                preview_hits += 1;
            }
        }
        if report.certificate.verify() && replay.replayable && replay.matches {
            replay_hits += 1;
        }
    }

    let total = cases.len() as u32;
    GroundingEvalReport {
        corpus_digest: fnv1a_hex(digest_input.as_bytes()),
        cases: total,
        action_accuracy: eval_ratio(action_hits, total),
        sensitivity_accuracy: eval_ratio(sensitivity_hits, total),
        repair_recall: eval_ratio(repair_hits, repair_expected),
        over_ask_rate: eval_ratio(over_asks, bind_expected),
        silent_bind_rate: eval_ratio(silent_binds, clarify_expected),
        truncation_safety: eval_ratio(truncation_safe, unresolved_cases),
        preview_coverage: eval_ratio(preview_hits, preview_expected),
        certificate_replay_rate: eval_ratio(replay_hits, total),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundingEvalBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

pub const GROUNDING_BASELINE_VERSION: u32 = 1;

impl GroundingEvalBaseline {
    pub fn from_report(report: &GroundingEvalReport) -> Self {
        Self {
            version: GROUNDING_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            metrics: report.flatten_metrics(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroundingMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroundingRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<GroundingMetricDelta>,
    pub gains: Vec<GroundingMetricDelta>,
}

impl GroundingRatchetVerdict {
    pub fn render_text(&self) -> String {
        if self.passed {
            return "指称接地棘轮检查通过：无指标低于基线。\n".to_string();
        }
        let mut out = String::from("指称接地棘轮检查未通过：\n");
        if !self.digest_matches {
            out.push_str("  - 语料摘要不匹配：金标语料已漂移，基线不可直接比较\n");
        }
        for regression in &self.regressions {
            let direction = if GROUNDING_LOWER_IS_BETTER.contains(&regression.metric.as_str()) {
                "（越低越好）"
            } else {
                ""
            };
            out.push_str(&format!(
                "  - {}: {:.6} → {:.6}{}\n",
                regression.metric, regression.baseline, regression.current, direction
            ));
        }
        out
    }
}

pub fn compare_baseline(
    baseline: &GroundingEvalBaseline,
    current: &GroundingEvalReport,
    epsilon: f64,
) -> GroundingRatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        let lower_is_better = GROUNDING_LOWER_IS_BETTER.contains(&metric.as_str());
        let (is_regression, is_gain) = if lower_is_better {
            (
                current_value > baseline_value + epsilon,
                current_value < baseline_value - epsilon,
            )
        } else {
            (
                current_value < baseline_value - epsilon,
                current_value > baseline_value + epsilon,
            )
        };
        if is_regression {
            regressions.push(GroundingMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(GroundingMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    GroundingRatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual_reference::{
        CandidateVote, ClarifyOption, DiscriminatingFeature, MentionSketch, Verdict,
    };
    use crate::entity_linker::LinkMethod;

    fn clarify_option(entity: &str) -> ClarifyOption {
        ClarifyOption {
            entity: entity.to_string(),
            label: entity.to_string(),
            etype: None,
        }
    }

    fn request(options: &[&str]) -> ClarifyRequest {
        ClarifyRequest {
            original_input: "苹果是什么".to_string(),
            mention: "苹果".to_string(),
            start: 0,
            key: "苹果".to_string(),
            question: ClarifyQuestion {
                mention: "苹果".to_string(),
                feature: DiscriminatingFeature::Names,
                options: options.iter().map(|entity| clarify_option(entity)).collect(),
            },
        }
    }

    fn resolution(options: &[(&str, i32)]) -> MentionResolution {
        MentionResolution {
            sketch: MentionSketch {
                mention: "苹果".to_string(),
                start: 0,
                quantifier: None,
                appositive_role: None,
                expectation_types: Vec::new(),
                relational_relation: None,
                key_surface: "苹果".to_string(),
                key_offset: 0,
                is_description: false,
            },
            key: "苹果".to_string(),
            verdict: Verdict::Ambiguous,
            chosen: None,
            candidates: options
                .iter()
                .map(|(entity, votes)| CandidateVote {
                    entity: (*entity).to_string(),
                    method: LinkMethod::ExactName,
                    votes: *votes,
                    hard_rejected: None,
                    etype: None,
                    evidence: Vec::new(),
                })
                .collect(),
            question: None,
            notes: Vec::new(),
        }
    }

    fn answered(value: &str) -> ProbeStatus {
        ProbeStatus::Answered(vec![value.to_string()])
    }

    #[test]
    fn same_answer_binds_top_candidate() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let report = probe_clarification(&request, Some(&resolution), 7, 3, |_, _| {
            answered("红色")
        });
        assert_eq!(report.sensitivity, ProbeSensitivity::Stable);
        assert!(matches!(
            report.decision,
            ProbeDecision::Bind { ref entity, .. } if entity == "苹果"
        ));
        assert_eq!(report.certificate.action, "bind");
        assert_eq!(report.certificate.entity.as_deref(), Some("苹果"));
    }

    #[test]
    fn different_answers_keep_clarification_with_previews() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let report = probe_clarification(&request, Some(&resolution), 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色")
            } else {
                answered("库克")
            }
        });
        assert_eq!(report.sensitivity, ProbeSensitivity::LinkSensitive);
        assert!(matches!(report.decision, ProbeDecision::Clarify { .. }));
        assert!(report.question.options[0].label.contains("红色"));
        assert!(report.question.options[1].label.contains("库克"));
        assert_ne!(
            report.observations[0].answer_fingerprint,
            report.observations[1].answer_fingerprint
        );
    }

    #[test]
    fn no_answer_top_is_repairable_by_alternative() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let report = probe_clarification(&request, Some(&resolution), 7, 3, |entity, _| {
            if entity == "苹果" {
                ProbeStatus::NoAnswer
            } else {
                answered("库克")
            }
        });
        assert_eq!(report.sensitivity, ProbeSensitivity::Repairable);
        assert!(matches!(
            report.decision,
            ProbeDecision::Repair { ref entity, .. } if entity == "苹果公司"
        ));
        assert!(report.notes.iter().any(|note| note.contains("库克")));
    }

    #[test]
    fn truncated_probe_never_claims_stability() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let report = probe_clarification(&request, Some(&resolution), 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色")
            } else {
                ProbeStatus::Truncated
            }
        });
        assert_eq!(report.sensitivity, ProbeSensitivity::OutcomeAmbiguous);
        assert!(matches!(report.decision, ProbeDecision::Clarify { .. }));
        assert!(report.question.options[1].label.contains("截断"));
    }

    #[test]
    fn all_no_answer_stays_uninformative_and_clarifies() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let report = probe_clarification(&request, Some(&resolution), 7, 3, |_, _| {
            ProbeStatus::NoAnswer
        });
        assert_eq!(report.sensitivity, ProbeSensitivity::Uninformative);
        assert!(matches!(report.decision, ProbeDecision::Clarify { .. }));
        assert!(report
            .certificate
            .candidates
            .iter()
            .all(|candidate| candidate.answer_count == 0));
    }

    #[test]
    fn certificate_is_deterministic_and_verifiable() {
        let request = request(&["苹果", "苹果公司"]);
        let resolution = resolution(&[("苹果", 4), ("苹果公司", 3)]);
        let first = probe_clarification(&request, Some(&resolution), 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色")
            } else {
                ProbeStatus::NoAnswer
            }
        });
        let second = probe_clarification(&request, Some(&resolution), 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色")
            } else {
                ProbeStatus::NoAnswer
            }
        });
        assert_eq!(first.certificate, second.certificate);
        assert_eq!(
            first.certificate.canonical_text(),
            second.certificate.canonical_text()
        );
        assert!(first.certificate.verify());
        assert!(second.certificate.verify());
    }

    #[test]
    fn replay_checks_fingerprints_actions_and_legacy_certificates() {
        let request = request(&["苹果", "苹果公司"]);
        let report = probe_clarification(&request, None, 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色")
            } else {
                ProbeStatus::NoAnswer
            }
        });
        let same = report
            .certificate
            .replay(7, |entity, _| {
                if entity == "苹果" {
                    answered("红色")
                } else {
                    ProbeStatus::NoAnswer
                }
            });
        assert!(same.replayable);
        assert!(same.matches, "{}", same.render_text());
        assert!(!same.revision_changed);

        let changed = report.certificate.replay(8, |entity, _| {
            if entity == "苹果" {
                answered("绿色")
            } else {
                answered("库克")
            }
        });
        assert!(!changed.matches);
        assert!(changed.revision_changed);
        assert_eq!(changed.actual_action, "clarify");

        let mut legacy = report.certificate.clone();
        for candidate in &mut legacy.candidates {
            candidate.rewritten.clear();
        }
        legacy.certificate_digest = legacy.compute_digest();
        assert!(legacy.verify());
        let snapshot = serde_json::to_string(&legacy).expect("serialize legacy");
        assert!(!snapshot.contains("\"rewritten\""));
        let restored: GroundingCertificate =
            serde_json::from_str(&snapshot).expect("deserialize legacy");
        assert!(restored.verify());
        let old = restored.replay(7, |_, _| answered("红色"));
        assert!(!old.replayable);
        assert!(!old.matches);
        assert!(old.notes.iter().any(|note| note.contains("旧版证书")));
    }

    #[test]
    fn certificate_keeps_fingerprints_not_full_answers() {
        let request = request(&["苹果", "苹果公司"]);
        let report = probe_clarification(&request, None, 7, 3, |entity, _| {
            if entity == "苹果" {
                answered("红色多汁")
            } else {
                answered("库克是首席执行官")
            }
        });
        let certificate = report.certificate.canonical_text();
        assert!(!certificate.contains("红色多汁"));
        assert!(!certificate.contains("库克是首席执行官"));
        assert!(report.question.render().contains("红色多汁"));
    }

    #[test]
    fn probe_respects_candidate_budget() {
        let request = request(&["苹果", "苹果公司", "苹果唱片", "苹果电影"]);
        let report = probe_clarification(&request, None, 7, 2, |_, _| answered("同一答案"));
        assert_eq!(report.observations.len(), 2);
        assert_eq!(report.certificate.candidates.len(), 2);
    }

    #[test]
    fn ledger_deduplicates_bounds_and_round_trips() {
        let request = request(&["苹果", "苹果公司"]);
        let mut ledger = GroundingLedger::new();
        for revision in 0..(MAX_GROUNDING_CERTIFICATES as u64 + 8) {
            let report = probe_clarification(&request, None, revision, 3, |_, _| {
                answered("红色")
            });
            ledger.push(report.certificate);
        }
        assert_eq!(ledger.len(), MAX_GROUNDING_CERTIFICATES);
        let before = ledger.len();
        let duplicate = ledger.latest().cloned().expect("ledger should not be empty");
        ledger.push(duplicate);
        assert_eq!(ledger.len(), before);
        let snapshot = ledger.snapshot_json().expect("snapshot");
        let restored = GroundingLedger::from_snapshot_json(&snapshot).expect("restore");
        assert_eq!(restored.len(), ledger.len());
        let latest = restored.latest().expect("restored certificate");
        assert!(latest.verify());
        let replay = latest.replay(latest.revision, |_, _| answered("红色"));
        assert!(replay.replayable);
        assert!(replay.matches, "{}", replay.render_text());
        assert!(restored.report().contains("指称接地证书"));
    }

    #[test]
    fn seed_eval_is_perfect_and_deterministic() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second);
        assert_eq!(first.action_accuracy, 1.0, "{}", first.render_text());
        assert_eq!(
            first.sensitivity_accuracy, 1.0,
            "{}",
            first.render_text()
        );
        assert_eq!(first.repair_recall, 1.0, "{}", first.render_text());
        assert_eq!(first.over_ask_rate, 0.0, "{}", first.render_text());
        assert_eq!(first.silent_bind_rate, 0.0, "{}", first.render_text());
        assert_eq!(first.truncation_safety, 1.0, "{}", first.render_text());
        assert_eq!(first.preview_coverage, 1.0, "{}", first.render_text());
        assert_eq!(
            first.certificate_replay_rate, 1.0,
            "{}",
            first.render_text()
        );
    }

    #[test]
    fn ratchet_rejects_over_ask_regression() {
        let report = evaluate_seed();
        let mut baseline = GroundingEvalBaseline::from_report(&report);
        baseline
            .metrics
            .insert("grounding.over_ask_rate".to_string(), 0.0);
        let mut broken = report.clone();
        broken.over_ask_rate = 0.5;
        let verdict = compare_baseline(&baseline, &broken, 1e-6);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "grounding.over_ask_rate"));
    }
}

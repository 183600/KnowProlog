use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::contextual_reference::{
    rewrite_at, ClarifyQuestion, ClarifyRequest, MentionResolution,
};

pub const GROUNDING_CERTIFICATE_VERSION: u32 = 1;
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
        assert!(restored.latest().map(GroundingCertificate::verify).unwrap_or(false));
        assert!(restored.report().contains("指称接地证书"));
    }
}

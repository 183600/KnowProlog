//! 历史可信度：把「回答为什么可信 / 为什么失效」变成可复核的路径。
//!
//! CST 把编年史编译成情景、信念与候选原因；本模块在同一个事件流上
//! 建一张支持图，把回答与语义事件直接关联：
//!
//!   语义事件（导入/断言/更正） --support--> 回答
//!   更正/撤回/冲突/快照漂移      --invalidate--> 回答
//!
//! 设计边界（三态，不假装全知）：
//!   - `Exact`：回答带归结证明，且证明的事实叶子覆盖了该支持；
//!   - `Inferred`：结构化匹配到支持（实体/问句 + 回答中的取值），但没有形式证明；
//!   - `Unknown`：没有任何可核对的支持，一律判 `Unassessed`。
//!
//! 只有 `Exact` 能判 `Clean`；`Inferred` 最多判 `Suspect`；更正、撤回与
//! 冲突是三种不同的事：更正 = 旧值错了（`Invalid`），撤回 = 当时可能对但
//! 不再相信（`Superseded`），冲突 = 两个来源不一致且未裁决（`Contested`）。

use std::collections::BTreeMap;

use crate::chronicle::{question_key, Chronicle};
use crate::conversation_theory::{
    AnswerRecord, ConversationTheory, EventKind, TheoryConflict, TheoryCorrection, TheoryEffect,
    TheoryEvent,
};
use crate::proof_journal::{
    JournalProjection, ProofCertificate, ProofNodeRecord, ProofOrigin, ProofSourceRecord,
};

/// 默认最多评估最近多少条回答（更早的回答仍留在编年史里）。
pub const DEFAULT_MAX_ANSWERS: usize = 128;

/// 支持的证据等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustBasis {
    /// 有形式证明的事实叶子覆盖该支持。
    Exact,
    /// 结构化匹配（实体/问句 + 取值），但无形式证明。
    Inferred,
    /// 没有可核对的支持。
    Unknown,
}

impl TrustBasis {
    pub fn label(self) -> &'static str {
        match self {
            TrustBasis::Exact => "精确",
            TrustBasis::Inferred => "推断",
            TrustBasis::Unknown => "未知",
        }
    }
}

/// 回答的可信状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustVerdict {
    /// 有精确支持，且没有更强的失效事件。
    Clean,
    /// 仅有推断支持，或存在无法定级的风险（如读取了旧快照）。
    Suspect,
    /// 旧值被明确更正：这条回答当时就错了。
    Invalid,
    /// 支持被撤回：这条回答当时可能对，但已不再被相信。
    Superseded,
    /// 来源冲突且未裁决。
    Contested,
    /// 没有可核对的支持，拒绝评价。
    Unassessed,
}

impl TrustVerdict {
    pub fn label(self) -> &'static str {
        match self {
            TrustVerdict::Clean => "可信",
            TrustVerdict::Suspect => "可疑",
            TrustVerdict::Invalid => "已失效（旧值错了）",
            TrustVerdict::Superseded => "已被撤回",
            TrustVerdict::Contested => "有争议",
            TrustVerdict::Unassessed => "未评估",
        }
    }

    fn rank(self) -> u8 {
        match self {
            TrustVerdict::Invalid => 5,
            TrustVerdict::Contested => 4,
            TrustVerdict::Superseded => 3,
            TrustVerdict::Suspect => 2,
            TrustVerdict::Clean => 1,
            TrustVerdict::Unassessed => 0,
        }
    }
}

/// 一条支持边：哪个事件、用哪组 (entity, predicate, value) 支撑了回答。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportClaim {
    pub event_seq: u64,
    pub event_kind: EventKind,
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub basis: TrustBasis,
}

/// 失效类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InvalidationKind {
    Corrected,
    Retracted,
    Forgotten,
    Conflict,
    StaleRead,
}

impl InvalidationKind {
    pub fn label(self) -> &'static str {
        match self {
            InvalidationKind::Corrected => "更正",
            InvalidationKind::Retracted => "撤回导入",
            InvalidationKind::Forgotten => "撤回断言",
            InvalidationKind::Conflict => "来源冲突",
            InvalidationKind::StaleRead => "读取旧快照",
        }
    }
}

/// 一条失效边：哪个后续事件让哪条支持不再成立。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invalidation {
    pub event_seq: u64,
    pub event_kind: EventKind,
    pub kind: InvalidationKind,
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub basis: TrustBasis,
    pub detail: String,
}

/// 单条回答的可信度报告。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerTrust {
    pub answer: AnswerRecord,
    pub verdict: TrustVerdict,
    pub basis: TrustBasis,
    pub support: Vec<SupportClaim>,
    pub invalidations: Vec<Invalidation>,
    pub stale_read: bool,
    pub reasons: Vec<String>,
}

/// 一次评估的结果。
#[derive(Debug, Clone, Default)]
pub struct TrustAssessment {
    pub answers: Vec<AnswerTrust>,
    pub events: usize,
    pub window_start_seq: u64,
    pub window_end_seq: u64,
    pub truncated: bool,
    pub notes: Vec<String>,
}

impl TrustAssessment {
    pub fn from_chronicle(chronicle: &Chronicle) -> Self {
        let theory = ConversationTheory::build(chronicle);
        Self::from_theory(chronicle, &theory)
    }

    pub fn from_theory(chronicle: &Chronicle, theory: &ConversationTheory) -> Self {
        let projection = JournalProjection::from_chronicle(chronicle);
        let proofs: BTreeMap<&str, &ProofCertificate> = projection
            .proofs
            .iter()
            .map(|proof| (proof.id.as_str(), proof))
            .collect();
        let events = theory.events();
        let start = theory.answers().len().saturating_sub(DEFAULT_MAX_ANSWERS);
        let truncated = start > 0;
        let answers = theory.answers()[start..]
            .iter()
            .map(|answer| {
                assess_answer(
                    answer,
                    events,
                    answer.proof_id.as_deref().and_then(|id| proofs.get(id).copied()),
                )
            })
            .collect::<Vec<_>>();
        let (window_start_seq, window_end_seq) = theory.window();
        let mut notes = Vec::new();
        if truncated {
            notes.push(format!(
                "只评估最近 {} 条回答（更早回答仍在编年史里）。",
                DEFAULT_MAX_ANSWERS
            ));
        }
        if theory.truncated_belief_facts() {
            notes.push("CST 信念事实超出预算；本报告只使用事件效果与证明叶子。".to_string());
        }
        Self {
            answers,
            events: events.len(),
            window_start_seq,
            window_end_seq,
            truncated,
            notes,
        }
    }

    pub fn matching(&self, query: &str) -> Vec<&AnswerTrust> {
        let query = query.trim();
        if query.is_empty() {
            return self.answers.iter().collect();
        }
        let key = question_key(query);
        let exact: Vec<&AnswerTrust> = self
            .answers
            .iter()
            .filter(|entry| entry.answer.question_key == key)
            .collect();
        if !exact.is_empty() {
            return exact;
        }
        let needle = query.to_lowercase();
        self.answers
            .iter()
            .filter(|entry| {
                entry.answer.question.to_lowercase().contains(&needle)
                    || entry.answer.answer.to_lowercase().contains(&needle)
            })
            .collect()
    }

    pub fn render(&self, query: Option<&str>) -> String {
        let query = query.map(str::trim).filter(|query| !query.is_empty());
        let matched: Vec<&AnswerTrust> = match query {
            Some(query) => self.matching(query),
            None => self.answers.iter().rev().take(12).collect(),
        };
        let mut lines = Vec::new();
        match query {
            Some(query) => lines.push(format!(
                "历史可信度「{}」：{} 条回答（窗口 e{}..e{}，共评估 {} 条）",
                query,
                matched.len(),
                self.window_start_seq,
                self.window_end_seq,
                self.answers.len()
            )),
            None => lines.push(format!(
                "历史可信度：最近 {} 条回答（窗口 e{}..e{}，共评估 {} 条）",
                matched.len(),
                self.window_start_seq,
                self.window_end_seq,
                self.answers.len()
            )),
        }
        if matched.is_empty() {
            lines.push("编年史里没有匹配的回答；无法评估。".to_string());
            lines.extend(self.notes.iter().map(|note| format!("· {}", note)));
            return lines.join("\n");
        }
        for entry in matched {
            lines.push(format!(
                "- 第 {} 条「{}」：{} / {}",
                entry.answer.reply_seq,
                truncate(&entry.answer.question, 32),
                entry.verdict.label(),
                entry.basis.label()
            ));
            lines.push(format!(
                "    回答：{}",
                truncate(&entry.answer.answer, 56)
            ));
            if entry.support.is_empty() {
                lines.push("    支持：无（没有任何事件效果或证明叶子覆盖这条回答）".to_string());
            } else {
                for claim in entry.support.iter().take(4) {
                    lines.push(format!(
                        "    支持：e{} {} {}.{} = {}（{}）",
                        claim.event_seq,
                        claim.event_kind.as_str(),
                        claim.entity,
                        claim.predicate,
                        claim.value,
                        claim.basis.label()
                    ));
                }
                if entry.support.len() > 4 {
                    lines.push(format!(
                        "    …另有 {} 条支持边省略",
                        entry.support.len() - 4
                    ));
                }
            }
            for invalidation in &entry.invalidations {
                lines.push(format!(
                    "    失效：e{} {}（{}；{}）",
                    invalidation.event_seq,
                    invalidation.event_kind.as_str(),
                    invalidation.kind.label(),
                    invalidation.detail
                ));
            }
            for reason in &entry.reasons {
                lines.push(format!("    · {}", reason));
            }
        }
        lines.extend(self.notes.iter().map(|note| format!("· {}", note)));
        lines.join("\n")
    }
}

fn assess_answer(
    answer: &AnswerRecord,
    events: &[TheoryEvent],
    proof: Option<&ProofCertificate>,
) -> AnswerTrust {
    let support = collect_support(answer, events, proof);
    let basis = overall_basis(&support);
    let stale_read = detect_stale_read(answer, events);
    let mut invalidations = collect_invalidations(answer, events, &support);
    if stale_read {
        invalidations.push(Invalidation {
            event_seq: answer.observed_seq,
            event_kind: EventKind::Other,
            kind: InvalidationKind::StaleRead,
            entity: String::new(),
            predicate: String::new(),
            value: String::new(),
            basis: TrustBasis::Inferred,
            detail: format!(
                "回答读取于版本 {}，提交于版本 {}（观察水位 e{}）",
                answer.read_revision, answer.commit_revision, answer.observed_seq
            ),
        });
    }
    invalidations.sort_by(|left, right| {
        left.event_seq
            .cmp(&right.event_seq)
            .then(left.kind.cmp(&right.kind))
    });
    invalidations.dedup();

    let verdict = verdict_for(basis, &invalidations);
    let mut reasons = Vec::new();
    match basis {
        TrustBasis::Exact => {
            reasons.push("支持链含归结证明的事实叶子：可以定级。".to_string());
        }
        TrustBasis::Inferred => {
            reasons.push("只有结构化匹配、没有形式证明：最多判「可疑」，不冒充可信。".to_string());
        }
        TrustBasis::Unknown => {
            reasons.push("没有可核对的支持：拒绝评价（不把未知说成无影响）。".to_string());
        }
    }
    if stale_read {
        reasons.push("回答基于旧快照：期间发生了知识变更。".to_string());
    }
    if verdict == TrustVerdict::Clean {
        reasons.push("未发现更正、撤回或未裁决冲突。".to_string());
    }
    if verdict == TrustVerdict::Suspect && !invalidations.is_empty() {
        reasons.push("存在失效信号，但证据等级不足以下「已失效」的结论。".to_string());
    }
    AnswerTrust {
        answer: answer.clone(),
        verdict,
        basis,
        support,
        invalidations,
        stale_read,
        reasons,
    }
}

fn overall_basis(support: &[SupportClaim]) -> TrustBasis {
    if support.is_empty() {
        return TrustBasis::Unknown;
    }
    if support
        .iter()
        .all(|claim| claim.basis == TrustBasis::Exact)
    {
        TrustBasis::Exact
    } else {
        TrustBasis::Inferred
    }
}

fn verdict_for(basis: TrustBasis, invalidations: &[Invalidation]) -> TrustVerdict {
    if basis == TrustBasis::Unknown {
        return TrustVerdict::Unassessed;
    }
    let mut verdict = if basis == TrustBasis::Exact {
        TrustVerdict::Clean
    } else {
        TrustVerdict::Suspect
    };
    for invalidation in invalidations {
        let candidate = match invalidation.kind {
            InvalidationKind::Corrected => {
                if invalidation.basis == TrustBasis::Exact {
                    TrustVerdict::Invalid
                } else {
                    TrustVerdict::Suspect
                }
            }
            InvalidationKind::Retracted | InvalidationKind::Forgotten => {
                if invalidation.basis == TrustBasis::Exact {
                    TrustVerdict::Superseded
                } else {
                    TrustVerdict::Suspect
                }
            }
            InvalidationKind::Conflict => {
                if invalidation.basis == TrustBasis::Exact {
                    TrustVerdict::Contested
                } else {
                    TrustVerdict::Suspect
                }
            }
            InvalidationKind::StaleRead => TrustVerdict::Suspect,
        };
        if candidate.rank() > verdict.rank() {
            verdict = candidate;
        }
    }
    verdict
}

fn collect_support(
    answer: &AnswerRecord,
    events: &[TheoryEvent],
    proof: Option<&ProofCertificate>,
) -> Vec<SupportClaim> {
    let answer_text = answer.answer.to_lowercase();
    if answer_text.is_empty() {
        return Vec::new();
    }
    let question_text = answer.question.to_lowercase();
    let focus = answer.focus.as_deref().unwrap_or_default().to_lowercase();
    let mut claims: BTreeMap<(u64, String, String, String), SupportClaim> = BTreeMap::new();
    for event in events {
        // 只使用回答读取水位之前的事件：之后的写入是并发/不可见的。
        if event.seq > answer.observed_seq || !event.kind.is_semantic() {
            continue;
        }
        for effect in &event.effects {
            if effect.op != crate::conversation_theory::EffectOp::Add || effect.value.is_empty() {
                continue;
            }
            let value = effect.value.to_lowercase();
            if value.is_empty() || !answer_text.contains(&value) {
                continue;
            }
            let entity = effect.entity.to_lowercase();
            let predicate = effect.predicate.to_lowercase();
            // 结构化自指（`姚明.name = 姚明` / `姚明.type = 人`）不构成属性答案的支持；
            // 除非问句确实在问名字/类型，否则只算噪声。
            let asks_name_or_type = question_text.contains("名字")
                || question_text.contains("叫什么")
                || question_text.contains("类型")
                || question_text.contains("name")
                || question_text.contains("type");
            let structural_self = (matches!(predicate.as_str(), "name" | "label")
                && value == entity)
                || predicate == "type";
            if structural_self && !asks_name_or_type {
                continue;
            }
            let entity_hit = !entity.is_empty()
                && (focus.contains(&entity)
                    || question_text.contains(&entity)
                    || answer_text.contains(&entity));
            let predicate_hit = !predicate.is_empty()
                && (question_text.contains(&predicate) || answer_text.contains(&predicate));
            if !entity_hit && !predicate_hit {
                continue;
            }
            if removed_before(event.seq, answer.observed_seq, effect, events) {
                continue;
            }
            let basis = if proof_fact_covers(proof, effect) {
                TrustBasis::Exact
            } else {
                TrustBasis::Inferred
            };
            claims.insert(
                (
                    event.seq,
                    effect.entity.clone(),
                    effect.predicate.clone(),
                    effect.value.clone(),
                ),
                SupportClaim {
                    event_seq: event.seq,
                    event_kind: event.kind,
                    entity: effect.entity.clone(),
                    predicate: effect.predicate.clone(),
                    value: effect.value.clone(),
                    basis,
                },
            );
        }
    }
    claims.into_values().collect()
}

fn removed_before(
    source_seq: u64,
    visible_until: u64,
    effect: &TheoryEffect,
    events: &[TheoryEvent],
) -> bool {
    events.iter().any(|event| {
        event.seq > source_seq
            && event.seq <= visible_until
            && event.effects.iter().any(|candidate| {
                candidate.op == crate::conversation_theory::EffectOp::Remove
                    && same_key(&candidate.entity, &candidate.predicate, &effect.entity, &effect.predicate)
                    && candidate.value.eq_ignore_ascii_case(&effect.value)
            })
    })
}

fn proof_fact_covers(proof: Option<&ProofCertificate>, effect: &TheoryEffect) -> bool {
    let Some(proof) = proof else {
        return false;
    };
    if proof.origin != ProofOrigin::Resolution {
        return false;
    }
    let value = effect.value.to_lowercase();
    let entity = effect.entity.to_lowercase();
    let predicate = effect.predicate.to_lowercase();
    let mut stack: Vec<&ProofNodeRecord> = proof.nodes.iter().collect();
    while let Some(node) = stack.pop() {
        stack.extend(node.children.iter());
        if node.source != ProofSourceRecord::Fact {
            continue;
        }
        let atom = node.atom.to_lowercase();
        if !atom.contains(&value) {
            continue;
        }
        let entity_hit = entity.is_empty() || atom.contains(&entity);
        let predicate_hit = predicate.is_empty() || atom.contains(&predicate);
        if entity_hit || predicate_hit {
            return true;
        }
    }
    false
}

fn collect_invalidations(
    answer: &AnswerRecord,
    events: &[TheoryEvent],
    support: &[SupportClaim],
) -> Vec<Invalidation> {
    let mut invalidations = Vec::new();
    let later: Vec<&TheoryEvent> = events
        .iter()
        .filter(|event| event.seq > answer.reply_seq)
        .collect();
    // 回答读取时已经可见的冲突也应计入：在冲突未裁决时给出的回答不是干净的。
    let visible_to_read: Vec<&TheoryEvent> = events
        .iter()
        .filter(|event| event.seq <= answer.observed_seq)
        .collect();

    for claim in support {
        let conflict_candidates: Vec<&TheoryEvent> = visible_to_read
            .iter()
            .copied()
            .chain(later.iter().copied())
            .filter(|event| event.seq > claim.event_seq)
            .collect();
        for event in &later {
            for correction in &event.corrections {
                if correction_affects(claim, correction) {
                    if correction.from.eq_ignore_ascii_case(&claim.value) {
                        invalidations.push(Invalidation {
                            event_seq: event.seq,
                            event_kind: event.kind,
                            kind: InvalidationKind::Corrected,
                            entity: claim.entity.clone(),
                            predicate: claim.predicate.clone(),
                            value: claim.value.clone(),
                            basis: claim.basis,
                            detail: format!(
                                "{} 的 {} 由「{}」更正为「{}」",
                                correction.entity, correction.predicate, correction.from, correction.to
                            ),
                        });
                    }
                }
            }
            if matches!(event.kind, EventKind::Retract | EventKind::Forget) {
                for effect in &event.effects {
                    if effect.op == crate::conversation_theory::EffectOp::Remove
                        && same_key(
                            &effect.entity,
                            &effect.predicate,
                            &claim.entity,
                            &claim.predicate,
                        )
                        && effect.value.eq_ignore_ascii_case(&claim.value)
                    {
                        invalidations.push(Invalidation {
                            event_seq: event.seq,
                            event_kind: event.kind,
                            kind: if event.kind == EventKind::Retract {
                                InvalidationKind::Retracted
                            } else {
                                InvalidationKind::Forgotten
                            },
                            entity: claim.entity.clone(),
                            predicate: claim.predicate.clone(),
                            value: claim.value.clone(),
                            basis: claim.basis,
                            detail: format!(
                                "{} 的 {} 不再包含「{}」",
                                claim.entity, claim.predicate, claim.value
                            ),
                        });
                    }
                }
            }
        }
        for event in conflict_candidates {
            for conflict in &event.conflicts {
                if conflict_affects(claim, conflict) && !resolved_after(event.seq, conflict, events) {
                    invalidations.push(Invalidation {
                        event_seq: event.seq,
                        event_kind: event.kind,
                        kind: InvalidationKind::Conflict,
                        entity: claim.entity.clone(),
                        predicate: claim.predicate.clone(),
                        value: claim.value.clone(),
                        basis: claim.basis,
                        detail: format!(
                            "{} 的 {} 同时存在「{}」与「{}」，未裁决",
                            conflict.entity,
                            conflict.predicate,
                            conflict.existing.join("、"),
                            conflict.incoming
                        ),
                    });
                }
            }
        }
    }

    // 撤回只在「没有任何替代来源」时构成 Superseded。
    let claims_by_key: BTreeMap<(String, String, String), Vec<u64>> = {
        let mut grouped: BTreeMap<(String, String, String), Vec<u64>> = BTreeMap::new();
        for claim in support {
            grouped
                .entry((
                    claim.entity.to_lowercase(),
                    claim.predicate.to_lowercase(),
                    claim.value.to_lowercase(),
                ))
                .or_default()
                .push(claim.event_seq);
        }
        grouped
    };
    invalidations.retain(|invalidation| match invalidation.kind {
        InvalidationKind::Retracted | InvalidationKind::Forgotten => {
            let key = (
                invalidation.entity.to_lowercase(),
                invalidation.predicate.to_lowercase(),
                invalidation.value.to_lowercase(),
            );
            let sources = claims_by_key.get(&key);
            match sources {
                Some(sources) if sources.len() > 1 => false,
                _ => true,
            }
        }
        _ => true,
    });

    invalidations
}

fn resolved_after(
    conflict_seq: u64,
    conflict: &TheoryConflict,
    events: &[TheoryEvent],
) -> bool {
    events.iter().any(|event| {
        event.seq > conflict_seq
            && event.corrections.iter().any(|correction| {
                same_key(
                    &correction.entity,
                    &correction.predicate,
                    &conflict.entity,
                    &conflict.predicate,
                ) && (correction.from.eq_ignore_ascii_case(&conflict.incoming)
                    || conflict.existing.iter().any(|existing| {
                        correction.from.eq_ignore_ascii_case(existing)
                            || correction.to.eq_ignore_ascii_case(existing)
                    }))
            })
    })
}

fn correction_affects(claim: &SupportClaim, correction: &TheoryCorrection) -> bool {
    if !same_key(
        &claim.entity,
        &claim.predicate,
        &correction.entity,
        &correction.predicate,
    ) {
        return false;
    }
    correction.from.eq_ignore_ascii_case(&claim.value)
        || correction.to.eq_ignore_ascii_case(&claim.value)
        || claim.value.is_empty()
}

fn conflict_affects(claim: &SupportClaim, conflict: &TheoryConflict) -> bool {
    same_key(
        &claim.entity,
        &claim.predicate,
        &conflict.entity,
        &conflict.predicate,
    ) && (conflict.incoming.eq_ignore_ascii_case(&claim.value)
        || conflict
            .existing
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&claim.value)))
}

fn same_key(
    left_entity: &str,
    left_predicate: &str,
    right_entity: &str,
    right_predicate: &str,
) -> bool {
    let entity_equal = left_entity.eq_ignore_ascii_case(right_entity)
        || left_entity.is_empty()
        || right_entity.is_empty();
    if !entity_equal {
        return false;
    }
    let left = left_predicate.to_lowercase();
    let right = right_predicate.to_lowercase();
    left.is_empty() || right.is_empty() || left == right || left.contains(&right) || right.contains(&left)
}

fn detect_stale_read(answer: &AnswerRecord, events: &[TheoryEvent]) -> bool {
    if answer.commit_revision != answer.read_revision {
        return true;
    }
    events.iter().any(|event| {
        event.seq > answer.observed_seq
            && event.seq < answer.reply_seq
            && event.kind.is_semantic()
            && event.revision != answer.read_revision
    })
}

fn truncate(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}

/// 一条支持是否被后续事件覆盖（供 UI 快速判断，不重算整份报告）。
pub fn support_is_invalidated(assessment: &TrustAssessment, reply_seq: u64) -> bool {
    assessment
        .answers
        .iter()
        .find(|entry| entry.answer.reply_seq == reply_seq)
        .map(|entry| {
            matches!(
                entry.verdict,
                TrustVerdict::Invalid | TrustVerdict::Superseded | TrustVerdict::Contested
            )
        })
        .unwrap_or(false)
}

/// 汇总事件效果，便于外部只读消费者复用（不参与折叠）。
pub fn add_effects(event: &TheoryEvent) -> Vec<&TheoryEffect> {
    event
        .effects
        .iter()
        .filter(|effect| effect.op == crate::conversation_theory::EffectOp::Add)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::{Chronicle, FsyncPolicy};
    use crate::conversation_theory::{
        append_assert, append_conflict, append_correction, append_forget, append_import,
        append_retract, EffectOp, TheoryCorrection,
    };
    use crate::predicate_registry::{
        ImportAction, ImportConflict, ImportReceipt,
    };
    use crate::proof_journal::{ProofCertificate, ProofLimits};
    use crate::resolution_engine::{ProofNode, ProofSource};
    use tempfile::TempDir;

    fn chronicle(dir: &TempDir) -> (Chronicle, String) {
        let path = dir.path().join("trust.chronicle.jsonl");
        let text = path.to_string_lossy().to_string();
        let (chronicle, audit) = Chronicle::open(&text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        (chronicle, text)
    }

    fn receipt(subject: &str, predicate: &str, value: &str) -> ImportReceipt {
        ImportReceipt {
            added: 1,
            actions: vec![ImportAction::Literal {
                subject: format!("http://example.org/kg/entity/{}", subject),
                predicate: format!("http://example.org/kg/{}", predicate),
                value: value.to_string(),
            }],
            ..Default::default()
        }
    }

    fn ask(chronicle: &mut Chronicle, question: &str) {
        chronicle
            .append_turn(question, "query", Some("姚明"), 0)
            .unwrap();
    }

    fn answer(chronicle: &mut Chronicle, text: &str) -> u64 {
        chronicle.append_reply(text, &[], &[]).unwrap()
    }

    #[test]
    fn inferred_support_is_never_clean() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "229"),
            "导入",
            1,
        )
        .unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = &assessment.answers[0];
        assert_eq!(entry.basis, TrustBasis::Inferred, "{:?}", entry);
        assert_eq!(entry.verdict, TrustVerdict::Suspect);
        assert!(!entry.support.is_empty());
    }

    #[test]
    fn import_support_and_correction_invalidates_with_proof() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        // 形式证明：事实叶子 height(姚明, 226)
        let certificate = ProofCertificate::from_resolution(
            "height(姚明, 226)",
            &std::collections::HashMap::new(),
            &[ProofNode {
                atom: crate::resolution_engine::Atom::rel("height", "姚明", "226"),
                source: ProofSource::Fact,
                confidence: 1.0,
                children: Vec::new(),
            }],
            1.0,
            ProofLimits::default(),
        );
        let _ = certificate.record(&mut chronicle);
        let reply = chronicle
            .append_reply_with_context(
                Some("姚明的身高是多少？"),
                "姚明身高 226 厘米。",
                &[],
                &[],
                Some(&certificate.id),
            )
            .unwrap();
        let correction = TheoryCorrection {
            entity: "姚明".to_string(),
            predicate: "height".to_string(),
            from: "226".to_string(),
            to: "229".to_string(),
        };
        append_correction(&mut chronicle, "更正：姚明身高 229", &[correction], 2).unwrap();

        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = assessment
            .answers
            .iter()
            .find(|entry| entry.answer.reply_seq == reply)
            .unwrap();
        assert_eq!(entry.basis, TrustBasis::Exact, "{:?}", entry);
        assert_eq!(entry.verdict, TrustVerdict::Invalid, "{:?}", entry);
        assert!(entry
            .invalidations
            .iter()
            .any(|invalidation| invalidation.kind == InvalidationKind::Corrected));
    }

    #[test]
    fn inferred_correction_degrades_to_suspect() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        append_correction(
            &mut chronicle,
            "更正：姚明身高 229",
            &[TheoryCorrection {
                entity: "姚明".to_string(),
                predicate: "height".to_string(),
                from: "226".to_string(),
                to: "229".to_string(),
            }],
            2,
        )
        .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = &assessment.answers[0];
        assert_eq!(entry.basis, TrustBasis::Inferred);
        assert_eq!(entry.verdict, TrustVerdict::Suspect);
        assert!(!entry.invalidations.is_empty());
    }

    #[test]
    fn retract_without_alternative_is_superseded() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let bill = receipt("姚明", "height", "226");
        append_import(&mut chronicle, &bill, "导入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        let reply = chronicle
            .append_reply_with_context(Some("姚明的身高是多少？"), "姚明身高 226 厘米。", &[], &[], None)
            .unwrap();
        // 无证明：即使撤回，也只能降级为 Suspect
        append_retract(&mut chronicle, &bill, "撤销导入", 2).unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = assessment
            .answers
            .iter()
            .find(|entry| entry.answer.reply_seq == reply)
            .unwrap();
        assert_eq!(entry.verdict, TrustVerdict::Suspect);
        assert!(entry
            .invalidations
            .iter()
            .any(|invalidation| invalidation.kind == InvalidationKind::Retracted));
    }

    #[test]
    fn conflict_marks_supported_answer_contested_only_with_proof() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        let certificate = ProofCertificate::from_resolution(
            "height(姚明, 226)",
            &std::collections::HashMap::new(),
            &[ProofNode {
                atom: crate::resolution_engine::Atom::rel("height", "姚明", "226"),
                source: ProofSource::Fact,
                confidence: 1.0,
                children: Vec::new(),
            }],
            1.0,
            ProofLimits::default(),
        );
        let _ = certificate.record(&mut chronicle);
        let reply = chronicle
            .append_reply_with_context(
                Some("姚明的身高是多少？"),
                "姚明身高 226 厘米。",
                &[],
                &[],
                Some(&certificate.id),
            )
            .unwrap();
        append_conflict(
            &mut chronicle,
            &[ImportConflict {
                subject: "姚明".to_string(),
                predicate: "height".to_string(),
                existing: vec!["226".to_string()],
                incoming: "229".to_string(),
            }],
            "导入",
            2,
        )
        .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = assessment
            .answers
            .iter()
            .find(|entry| entry.answer.reply_seq == reply)
            .unwrap();
        assert_eq!(entry.verdict, TrustVerdict::Contested, "{:?}", entry);
    }

    #[test]
    fn conflict_visible_before_answer_is_recorded() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        append_conflict(
            &mut chronicle,
            &[ImportConflict {
                subject: "姚明".to_string(),
                predicate: "height".to_string(),
                existing: vec!["226".to_string()],
                incoming: "229".to_string(),
            }],
            "导入",
            2,
        )
        .unwrap();
        chronicle.set_kb_revision(2);
        ask(&mut chronicle, "姚明的身高是多少？");
        let certificate = ProofCertificate::from_resolution(
            "height(姚明, 226)",
            &std::collections::HashMap::new(),
            &[ProofNode {
                atom: crate::resolution_engine::Atom::rel("height", "姚明", "226"),
                source: ProofSource::Fact,
                confidence: 1.0,
                children: Vec::new(),
            }],
            1.0,
            ProofLimits::default(),
        );
        let _ = certificate.record(&mut chronicle);
        let reply = chronicle
            .append_reply_with_context(
                Some("姚明的身高是多少？"),
                "姚明身高 226 厘米。",
                &[],
                &[],
                Some(&certificate.id),
            )
            .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = assessment
            .answers
            .iter()
            .find(|entry| entry.answer.reply_seq == reply)
            .unwrap();
        assert_eq!(entry.verdict, TrustVerdict::Contested, "{:?}", entry);
        assert!(entry
            .invalidations
            .iter()
            .any(|invalidation| invalidation.kind == InvalidationKind::Conflict));
    }

    #[test]
    fn correction_resolves_earlier_conflict() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let bill = receipt("姚明", "height", "226");
        append_import(&mut chronicle, &bill, "导入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        let certificate = ProofCertificate::from_resolution(
            "height(姚明, 226)",
            &std::collections::HashMap::new(),
            &[ProofNode {
                atom: crate::resolution_engine::Atom::rel("height", "姚明", "226"),
                source: ProofSource::Fact,
                confidence: 1.0,
                children: Vec::new(),
            }],
            1.0,
            ProofLimits::default(),
        );
        let _ = certificate.record(&mut chronicle);
        chronicle
            .append_reply_with_context(
                Some("姚明的身高是多少？"),
                "姚明身高 226 厘米。",
                &[],
                &[],
                Some(&certificate.id),
            )
            .unwrap();
        append_conflict(
            &mut chronicle,
            &[ImportConflict {
                subject: "姚明".to_string(),
                predicate: "height".to_string(),
                existing: vec!["226".to_string()],
                incoming: "229".to_string(),
            }],
            "导入",
            2,
        )
        .unwrap();
        append_correction(
            &mut chronicle,
            "更正：姚明身高 229",
            &[TheoryCorrection {
                entity: "姚明".to_string(),
                predicate: "height".to_string(),
                from: "226".to_string(),
                to: "229".to_string(),
            }],
            3,
        )
        .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = &assessment.answers[0];
        assert_eq!(entry.verdict, TrustVerdict::Invalid, "{:?}", entry);
        assert!(entry
            .invalidations
            .iter()
            .all(|invalidation| invalidation.kind != InvalidationKind::Conflict));
    }

    #[test]
    fn answer_without_support_is_unassessed_even_with_correction() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "不知道。");
        append_correction(
            &mut chronicle,
            "更正：姚明身高 229",
            &[TheoryCorrection {
                entity: "姚明".to_string(),
                predicate: "height".to_string(),
                from: "226".to_string(),
                to: "229".to_string(),
            }],
            1,
        )
        .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        assert_eq!(assessment.answers[0].basis, TrustBasis::Unknown);
        assert_eq!(
            assessment.answers[0].verdict,
            TrustVerdict::Unassessed
        );
    }

    #[test]
    fn later_answer_does_not_use_removed_source() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let bill = receipt("姚明", "height", "226");
        append_import(&mut chronicle, &bill, "导入", 1).unwrap();
        append_retract(&mut chronicle, &bill, "撤销导入", 2).unwrap();
        chronicle.set_kb_revision(2);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        assert!(assessment.answers[0].support.is_empty());
        assert_eq!(assessment.answers[0].verdict, TrustVerdict::Unassessed);
    }

    #[test]
    fn stale_read_is_detected_from_reply_watermark() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        let reply = chronicle
            .append_reply_with_context(
                Some("姚明的身高是多少？"),
                "姚明身高 226 厘米。",
                &[],
                &[],
                None,
            )
            .unwrap();
        let entry = chronicle.entries().last().unwrap();
        assert_eq!(reply, entry.seq);
        assert_eq!(entry.payload["read_revision"], 1);
        assert_eq!(entry.payload["commit_revision"], 1);
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        assert!(!assessment.answers[0].stale_read);
        assert!(!support_is_invalidated(&assessment, reply));
    }

    #[test]
    fn conflict_event_does_not_change_folded_beliefs() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_import(
            &mut chronicle,
            &receipt("姚明", "height", "226"),
            "导入",
            1,
        )
        .unwrap();
        append_conflict(
            &mut chronicle,
            &[ImportConflict {
                subject: "姚明".to_string(),
                predicate: "height".to_string(),
                existing: vec!["226".to_string()],
                incoming: "229".to_string(),
            }],
            "导入",
            2,
        )
        .unwrap();
        let theory = ConversationTheory::build(&chronicle);
        let beliefs = theory.beliefs_at(chronicle.last_seq());
        assert!(beliefs.iter().any(|fact| fact.value == "226"));
        assert!(!beliefs.iter().any(|fact| fact.value == "229"));
    }

    #[test]
    fn forget_removes_assertion_support() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let effects = crate::conversation_theory::effects_from_text("姚明的身高是229", EffectOp::Add);
        append_assert(&mut chronicle, "姚明的身高是229", &effects, 0).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");
        append_forget(
            &mut chronicle,
            "姚明的身高是229",
            &crate::conversation_theory::effects_from_text(
                "姚明的身高是229",
                EffectOp::Remove,
            ),
            2,
        )
        .unwrap();
        let assessment = TrustAssessment::from_chronicle(&chronicle);
        let entry = &assessment.answers[0];
        assert!(entry
            .invalidations
            .iter()
            .any(|invalidation| invalidation.kind == InvalidationKind::Forgotten));
    }
}

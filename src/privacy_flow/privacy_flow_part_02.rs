impl PrivacyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// 开启新的一轮；台账中的记录使用该轮号。
    pub fn begin_turn(&mut self) -> u64 {
        self.turn = self.turn.saturating_add(1);
        self.turn
    }

    pub fn grant(
        &mut self,
        subject: &str,
        category: PrivacyCategory,
        purpose: &str,
        scope: ConsentScope,
    ) {
        self.consents.push(ConsentRecord {
            subject: subject.to_string(),
            category,
            purpose: purpose.to_string(),
            scope,
            granted: true,
            consumed: false,
            turn: self.turn,
        });
    }

    pub fn revoke(&mut self, subject: &str, category: PrivacyCategory, purpose: &str) {
        self.consents.push(ConsentRecord {
            subject: subject.to_string(),
            category,
            purpose: purpose.to_string(),
            scope: ConsentScope::Session,
            granted: false,
            consumed: false,
            turn: self.turn,
        });
    }

    /// 把一条隐私言语行为落到台账上。`AskPolicy` 与 `Other` 不改变状态。
    pub fn apply_act(&mut self, subject: &str, purpose: &str, act: &PrivacyAct) {
        match act {
            PrivacyAct::Grant { category, scope } => {
                self.grant(subject, *category, purpose, *scope);
            }
            PrivacyAct::Revoke { category } => {
                self.revoke(subject, *category, purpose);
            }
            PrivacyAct::Forget {
                category: Some(category),
            } => self.forget_category(subject, *category, "对话中的遗忘请求"),
            PrivacyAct::Forget { category: None } => {
                self.forget_subject(subject, "对话中的遗忘请求")
            }
            PrivacyAct::AskPolicy | PrivacyAct::Other => {}
        }
    }

    pub fn forget_subject(&mut self, subject: &str, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: None,
            category: None,
            fact_digest: None,
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn forget_category(&mut self, subject: &str, category: PrivacyCategory, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: None,
            category: Some(category),
            fact_digest: None,
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn forget_fact(&mut self, subject: &str, predicate: &str, object: &str, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: Some(predicate.to_string()),
            category: None,
            fact_digest: Some(fact_digest(subject, predicate, object)),
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn is_tombstoned(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        category: PrivacyCategory,
    ) -> bool {
        let digest = fact_digest(subject, predicate, object);
        self.tombstones
            .iter()
            .any(|tombstone| tombstone.matches(subject, predicate, category, &digest))
    }

    /// 返回可用的显式授权（一次性授权未消费才算可用）。
    pub fn consent_permits(&self, subject: &str, category: PrivacyCategory, purpose: &str) -> bool {
        for record in self.consents.iter().rev() {
            if record.subject != subject || record.category != category {
                continue;
            }
            if record.purpose != purpose && record.purpose != "*" {
                continue;
            }
            if !record.granted {
                return false;
            }
            if record.scope == ConsentScope::Once && record.consumed {
                continue;
            }
            return true;
        }
        false
    }

    fn consume_consent(&mut self, subject: &str, category: PrivacyCategory, purpose: &str) {
        for record in self.consents.iter_mut().rev() {
            if record.subject != subject || record.category != category || !record.granted {
                continue;
            }
            if record.purpose != purpose && record.purpose != "*" {
                continue;
            }
            if record.scope == ConsentScope::Once && !record.consumed {
                record.consumed = true;
                return;
            }
        }
    }

    /// 同一槽位此前披露过的最粗精度；后续轮次不得更细。
    pub fn disclosure_cap(&self, subject: &str, predicate: &str) -> Option<DisclosurePrecision> {
        self.disclosures
            .iter()
            .filter(|record| record.subject == subject && record.predicate == predicate)
            .map(|record| record.precision)
            .min()
    }

    fn note_disclosure(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        category: PrivacyCategory,
        purpose: &str,
        precision: DisclosurePrecision,
    ) {
        self.disclosures.push(DisclosureRecord {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            category,
            purpose: purpose.to_string(),
            precision,
            turn: self.turn,
            fact_digest: fact_digest(subject, predicate, object),
        });
    }

    pub fn canonical_json(&self) -> Result<String, String> {
        crate::canonical::canonical_text(self).map_err(|error| error.to_string())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let ledger: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if ledger.version > PRIVACY_LEDGER_VERSION {
            return Err(format!(
                "隐私台账版本 {} 高于支持版本 {}",
                ledger.version, PRIVACY_LEDGER_VERSION
            ));
        }
        Ok(ledger)
    }

    pub fn digest(&self) -> String {
        crate::canonical::canonical_digest(self).unwrap_or_default()
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "隐私台账：第 {} 轮，授权 {} 条，披露 {} 条，墓碑 {} 条\n",
            self.turn,
            self.consents.len(),
            self.disclosures.len(),
            self.tombstones.len()
        ));
        for record in &self.consents {
            output.push_str(&format!(
                "  授权：{} 的{}，用途「{}」，范围 {}，{}\n",
                record.subject,
                record.category.label(),
                record.purpose,
                record.scope.code(),
                if record.granted {
                    if record.consumed {
                        "已消费"
                    } else {
                        "有效"
                    }
                } else {
                    "已撤回"
                }
            ));
        }
        for record in &self.disclosures {
            output.push_str(&format!(
                "  披露：{} {}（{}，{}）\n",
                record.subject,
                record.predicate,
                record.category.label(),
                record.precision.label()
            ));
        }
        for tombstone in &self.tombstones {
            output.push_str(&format!(
                "  墓碑：{}{}{}（{}）\n",
                tombstone.subject.as_deref().unwrap_or("*"),
                tombstone
                    .category
                    .map(|category| format!("/{}", category.code()))
                    .unwrap_or_default(),
                tombstone
                    .predicate
                    .as_deref()
                    .map(|predicate| format!("/{}", predicate))
                    .unwrap_or_default(),
                tombstone.reason
            ));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// 裁决
// ---------------------------------------------------------------------------
/// 一次披露请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub purpose: String,
    pub requested_precision: DisclosurePrecision,
    pub wh: Option<WhKind>,
    pub k_required: Option<usize>,
}
impl DisclosureRequest {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            purpose: "问答".to_string(),
            requested_precision: DisclosurePrecision::Exact,
            wh: None,
            k_required: None,
        }
    }

    pub fn with_purpose(mut self, purpose: &str) -> Self {
        self.purpose = purpose.to_string();
        self
    }

    pub fn with_precision(mut self, precision: DisclosurePrecision) -> Self {
        self.requested_precision = precision;
        self
    }

    pub fn with_question(mut self, question: &str) -> Self {
        self.wh = wh_kind(question);
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k_required = Some(k);
        self
    }
}
/// 裁决结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureStatus {
    Allowed,
    Downgraded,
    NeedsConsent,
    Blocked,
}
impl DisclosureStatus {
    pub const fn code(self) -> &'static str {
        match self {
            DisclosureStatus::Allowed => "allowed",
            DisclosureStatus::Downgraded => "downgraded",
            DisclosureStatus::NeedsConsent => "needs_consent",
            DisclosureStatus::Blocked => "blocked",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            DisclosureStatus::Allowed => "允许",
            DisclosureStatus::Downgraded => "降级披露",
            DisclosureStatus::NeedsConsent => "需要授权",
            DisclosureStatus::Blocked => "阻断",
        }
    }
}
/// 稳定决策码，供回执与指标使用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionCode {
    AuditShadow,
    Tombstoned,
    ConsentRequired,
    ConsentGranted,
    KAnonymity,
    InferenceLeak,
    PriorDisclosure,
    PolicyCeiling,
    WeakEvidence,
    ClosureTruncated,
    WhMismatch,
}
impl DecisionCode {
    pub const fn code(self) -> &'static str {
        match self {
            DecisionCode::AuditShadow => "audit_shadow",
            DecisionCode::Tombstoned => "tombstoned",
            DecisionCode::ConsentRequired => "consent_required",
            DecisionCode::ConsentGranted => "consent_granted",
            DecisionCode::KAnonymity => "k_anonymity",
            DecisionCode::InferenceLeak => "inference_leak",
            DecisionCode::PriorDisclosure => "prior_disclosure",
            DecisionCode::PolicyCeiling => "policy_ceiling",
            DecisionCode::WeakEvidence => "weak_evidence",
            DecisionCode::ClosureTruncated => "closure_truncated",
            DecisionCode::WhMismatch => "wh_mismatch",
        }
    }
}
/// 审计模式下「如果执行会怎样」的影子结论。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditFinding {
    pub would_be_status: DisclosureStatus,
    pub would_be_precision: DisclosurePrecision,
    pub codes: Vec<DecisionCode>,
}
/// 只含决策码与摘要的隐私回执，不落明文。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyReceipt {
    pub mode: PolicyMode,
    pub category: PrivacyCategory,
    pub status: DisclosureStatus,
    pub requested_precision: DisclosurePrecision,
    pub effective_precision: DisclosurePrecision,
    pub simulated_status: DisclosureStatus,
    pub simulated_precision: DisclosurePrecision,
    pub codes: Vec<String>,
    pub fact_digest: String,
    pub ledger_digest: String,
    pub turn: u64,
}
/// 批量审计摘要。`Audit` 模式使用影子结论统计，`Enforce` 模式使用实际结论，
/// 因而同一份策略可以先在影子模式度量公开事实保留率与误拦率。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total: usize,
    pub allowed: usize,
    pub downgraded: usize,
    pub needs_consent: usize,
    pub blocked: usize,
    /// 非敏感（`Other`）事实数：它们本应原样回答。
    pub public_total: usize,
    pub public_retained: usize,
    /// 非敏感事实中被降级或阻断的数量。
    pub public_changed: usize,
    pub sensitive_total: usize,
    pub sensitive_retained: usize,
    pub closure_truncated: usize,
    pub code_counts: BTreeMap<String, usize>,
}
impl AuditSummary {
    pub fn from_decisions<'a>(
        decisions: impl IntoIterator<Item = &'a DisclosureDecision>,
    ) -> Self {
        let mut summary = Self::default();
        for decision in decisions {
            summary.total += 1;
            let (status, precision) = match &decision.audit {
                Some(audit) => (audit.would_be_status, audit.would_be_precision),
                None => (decision.status, decision.effective_precision),
            };
            match status {
                DisclosureStatus::Allowed => summary.allowed += 1,
                DisclosureStatus::Downgraded => summary.downgraded += 1,
                DisclosureStatus::NeedsConsent => summary.needs_consent += 1,
                DisclosureStatus::Blocked => summary.blocked += 1,
            }
            let retained = precision == decision.requested_precision;
            if decision.category == PrivacyCategory::Other {
                summary.public_total += 1;
                if retained {
                    summary.public_retained += 1;
                } else {
                    summary.public_changed += 1;
                }
            } else {
                summary.sensitive_total += 1;
                if retained {
                    summary.sensitive_retained += 1;
                }
            }
            let codes = match &decision.audit {
                Some(audit) => audit.codes.as_slice(),
                None => decision.codes.as_slice(),
            };
            for code in codes {
                *summary
                    .code_counts
                    .entry(code.code().to_string())
                    .or_insert(0) += 1;
                if *code == DecisionCode::ClosureTruncated {
                    summary.closure_truncated += 1;
                }
            }
        }
        summary
    }

    /// 公开事实按请求精度保留的比例；没有公开事实时视为 1。
    pub fn public_retention(&self) -> f64 {
        ratio(self.public_retained, self.public_total)
    }

    /// 公开事实被误拦的比例；没有公开事实时视为 0。
    pub fn false_block_rate(&self) -> f64 {
        if self.public_total == 0 {
            0.0
        } else {
            (self.public_changed as f64) / (self.public_total as f64)
        }
    }

    pub fn render(&self) -> String {
        let mut text = format!(
            "审计摘要：{} 条；允许 {}，降级 {}，待授权 {}，阻断 {}；公开事实保留 {}/{}（{:.1}%），误降级/误拦 {:.1}%；敏感事实保持请求精度 {}/{}",
            self.total,
            self.allowed,
            self.downgraded,
            self.needs_consent,
            self.blocked,
            self.public_retained,
            self.public_total,
            self.public_retention() * 100.0,
            self.false_block_rate() * 100.0,
            self.sensitive_retained,
            self.sensitive_total
        );
        if self.closure_truncated > 0 {
            text.push_str(&format!(
                "；闭包截断 {} 条（按保守结论计）",
                self.closure_truncated
            ));
        }
        if !self.code_counts.is_empty() {
            let codes = self
                .code_counts
                .iter()
                .map(|(code, count)| format!("{}={}", code, count))
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&format!("；决策码：{}", codes));
        }
        text
    }
}
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        (numerator as f64) / (denominator as f64)
    }
}
/// 一次披露的完整裁决。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureDecision {
    pub status: DisclosureStatus,
    pub requested_precision: DisclosurePrecision,
    pub effective_precision: DisclosurePrecision,
    pub category: PrivacyCategory,
    pub replacement: Option<String>,
    pub codes: Vec<DecisionCode>,
    pub inference_leaks: Vec<InferenceLeak>,
    pub audit: Option<AuditFinding>,
    pub receipt: PrivacyReceipt,
}
impl DisclosureDecision {
    pub fn replacement_or(&self, original: &str) -> String {
        self.replacement
            .clone()
            .unwrap_or_else(|| original.to_string())
    }

    /// 把裁决接到生成叠加层：隐藏整个槽位，或用类型化表达替换原值。
    ///
    /// 注意：`Audit` 模式的 `effective_precision` 等于请求精度，因此不会改动
    /// 生成结果；只有 `Enforce` 模式才会真正写入 overlay。
    pub fn apply_to_overlay(
        &self,
        overlay: &mut crate::symbolic_generator::GenerationOverlay,
        subject: &str,
        predicate: &str,
        original: &str,
    ) {
        if self.status == DisclosureStatus::Blocked
            || self.effective_precision == DisclosurePrecision::Hidden
        {
            overlay
                .hide
                .insert((subject.to_string(), predicate.to_string()));
            return;
        }
        if self.effective_precision == DisclosurePrecision::Exact {
            return;
        }
        let replacement = self
            .replacement
            .clone()
            .unwrap_or_else(|| original.to_string());
        if replacement != original {
            overlay.replace.insert(
                (subject.to_string(), predicate.to_string()),
                vec![replacement],
            );
        }
    }
}
/// 隐私裁决器：策略 + 图谱 + 可变的会话台账。
pub struct PrivacyAdjudicator<'a> {
    policy: &'a PrivacyPolicy,
    graph: &'a PrivacyGraph,
    ledger: &'a mut PrivacyLedger,
}
impl<'a> PrivacyAdjudicator<'a> {
    pub fn new(
        policy: &'a PrivacyPolicy,
        graph: &'a PrivacyGraph,
        ledger: &'a mut PrivacyLedger,
    ) -> Self {
        Self {
            policy,
            graph,
            ledger,
        }
    }

    pub fn adjudicate(&mut self, request: &DisclosureRequest) -> DisclosureDecision {
        let fact = self
            .graph
            .find(&request.subject, &request.predicate, &request.object)
            .cloned()
            .unwrap_or_else(|| {
                PrivacyFact::label(
                    self.policy,
                    &request.subject,
                    &request.predicate,
                    &request.object,
                )
            });
        let category = fact.category;
        let requested = fact.precision.min(request.requested_precision);
        let explicit_consent =
            self.ledger
                .consent_permits(&fact.subject, category, &request.purpose);
        let mut effective = requested;
        let mut codes: Vec<DecisionCode> = Vec::new();
        let mut inference_leaks: Vec<InferenceLeak> = Vec::new();

        if !category.compatible_with(request.wh) {
            add_code(&mut codes, DecisionCode::WhMismatch);
        }

        if self
            .ledger
            .is_tombstoned(&fact.subject, &fact.predicate, &fact.object, category)
        {
            effective = DisclosurePrecision::Hidden;
            add_code(&mut codes, DecisionCode::Tombstoned);
        } else {
            if !explicit_consent {
                let ceiling = self
                    .policy
                    .max_precision_for_predicate(&fact.predicate)
                    .unwrap_or_else(|| self.policy.max_precision(category));
                if effective > ceiling {
                    effective = ceiling;
                    add_code(&mut codes, DecisionCode::PolicyCeiling);
                }
                if should_k_anonymize(category, self.policy) {
                    let k = request.k_required.unwrap_or(self.policy.k_anonymity).max(1);
                    while effective != DisclosurePrecision::Hidden
                        && self.graph.class_size(&fact, effective) < k
                    {
                        effective = effective.coarsen();
                        add_code(&mut codes, DecisionCode::KAnonymity);
                    }
                }
            }

            let (leaks, truncated) = self.graph.inference_leaks(&fact, self.policy, effective);
            if truncated {
                add_code(&mut codes, DecisionCode::ClosureTruncated);
            }
            if !leaks.is_empty() {
                let cap = leaks
                    .iter()
                    .map(|leak| leak.max_precision)
                    .min()
                    .unwrap_or(effective);
                if cap < effective {
                    effective = cap;
                    add_code(&mut codes, DecisionCode::InferenceLeak);
                }
                inference_leaks = leaks;
            }

            if self.policy.requires_consent(category) && requested > DisclosurePrecision::TypeOnly {
                if explicit_consent {
                    add_code(&mut codes, DecisionCode::ConsentGranted);
                } else {
                    effective = DisclosurePrecision::TypeOnly.min(effective);
                    add_code(&mut codes, DecisionCode::ConsentRequired);
                }
            }

            if !explicit_consent {
                if let Some(previous) = self.ledger.disclosure_cap(&fact.subject, &fact.predicate) {
                    if previous < effective {
                        effective = previous;
                        add_code(&mut codes, DecisionCode::PriorDisclosure);
                    }
                }
            }
        }

        let would_be_status = if effective == DisclosurePrecision::Hidden {
            if requested == DisclosurePrecision::Hidden {
                DisclosureStatus::Allowed
            } else {
                DisclosureStatus::Blocked
            }
        } else if codes.contains(&DecisionCode::ConsentRequired) {
            DisclosureStatus::NeedsConsent
        } else if effective < requested {
            DisclosureStatus::Downgraded
        } else {
            DisclosureStatus::Allowed
        };

        let audit_mode = self.policy.mode == PolicyMode::Audit;
        let (status, effective_precision, replacement) = if audit_mode {
            add_code(&mut codes, DecisionCode::AuditShadow);
            (
                DisclosureStatus::Allowed,
                requested,
                Some(fact.object.clone()),
            )
        } else {
            let replacement_category = inference_leaks
                .iter()
                .map(|leak| leak.category)
                .find(|candidate| request.wh.is_some() && candidate.compatible_with(request.wh))
                .unwrap_or(category);
            let replacement = match would_be_status {
                DisclosureStatus::Blocked => None,
                _ if effective == DisclosurePrecision::Exact => Some(fact.object.clone()),
                _ => coarsen_value(replacement_category, effective, &fact.object, request.wh),
            };
            (would_be_status, effective, replacement)
        };

        if !audit_mode && would_be_status != DisclosureStatus::Blocked {
            self.ledger.note_disclosure(
                &fact.subject,
                &fact.predicate,
                &fact.object,
                category,
                &request.purpose,
                effective,
            );
            if explicit_consent {
                self.ledger
                    .consume_consent(&fact.subject, category, &request.purpose);
            }
        }

        let receipt = PrivacyReceipt {
            mode: self.policy.mode,
            category,
            status,
            requested_precision: requested,
            effective_precision,
            simulated_status: would_be_status,
            simulated_precision: effective,
            codes: codes.iter().map(|code| code.code().to_string()).collect(),
            fact_digest: fact.fact_digest(),
            ledger_digest: self.ledger.digest(),
            turn: self.ledger.turn,
        };
        let audit = if audit_mode {
            Some(AuditFinding {
                would_be_status,
                would_be_precision: effective,
                codes: codes
                    .iter()
                    .copied()
                    .filter(|code| *code != DecisionCode::AuditShadow)
                    .collect(),
            })
        } else {
            None
        };

        DisclosureDecision {
            status,
            requested_precision: requested,
            effective_precision,
            category,
            replacement,
            codes,
            inference_leaks,
            audit,
            receipt,
        }
    }
}
fn add_code(codes: &mut Vec<DecisionCode>, code: DecisionCode) {
    if !codes.contains(&code) {
        codes.push(code);
    }
}
fn should_k_anonymize(category: PrivacyCategory, policy: &PrivacyPolicy) -> bool {
    category.is_personal()
        && category != PrivacyCategory::PersonalIdentity
        && policy.max_precision(category) < DisclosurePrecision::Exact
}

// ---------------------------------------------------------------------------
// 类型保持改写
// ---------------------------------------------------------------------------
/// 按披露类别与疑问类型给出占位表达。类别优先保证槽位语义，疑问类型
/// 只在该类别本身没有更具体表达时兜底。
pub fn typed_placeholder(category: PrivacyCategory, wh: Option<WhKind>) -> &'static str {
    match category {
        PrivacyCategory::PersonalIdentity => match wh {
            Some(WhKind::Person) | None => "某位人士",
            _ => "某人的身份",
        },
        PrivacyCategory::Contact => "某联系方式",
        PrivacyCategory::Credential => "某证件信息",
        PrivacyCategory::Location => "某地",
        PrivacyCategory::Health => "某健康状况",
        PrivacyCategory::Finance => "某财务信息",
        PrivacyCategory::Demographic => match wh {
            Some(WhKind::Time) => "某个时间段",
            _ => "某个年龄范围",
        },
        PrivacyCategory::Relationship => "某位亲友",
        PrivacyCategory::Other => match wh {
            Some(WhKind::Person) => "某位人士",
            Some(WhKind::Place) => "某地",
            Some(WhKind::Time) => "某个时间",
            Some(WhKind::Quantity) => "某个范围",
            Some(WhKind::Reason) => "相关原因（已省略）",
            Some(WhKind::Thing) | None => "相关信息",
        },
    }
}
/// 类型保持改写：同一类别在同一精度下输出稳定表达，便于多轮指代。
pub fn coarsen_value(
    category: PrivacyCategory,
    precision: DisclosurePrecision,
    value: &str,
    wh: Option<WhKind>,
) -> Option<String> {
    if value.trim().is_empty() {
        return None;
    }
    match precision {
        DisclosurePrecision::Exact => Some(value.to_string()),
        DisclosurePrecision::Hidden => None,
        DisclosurePrecision::TypeOnly => Some(typed_placeholder(category, wh).to_string()),
        DisclosurePrecision::Granular => match category {
            PrivacyCategory::Location => cut_through(value, &['区', '县'])
                .or_else(|| cut_through(value, &['市', '省']))
                .or_else(|| Some(typed_placeholder(category, wh).to_string())),
            PrivacyCategory::Demographic => coarsen_demographic(value, 5),
            PrivacyCategory::PersonalIdentity => {
                let surname: String = value.chars().take(1).collect();
                if surname.is_empty() {
                    Some(typed_placeholder(category, wh).to_string())
                } else {
                    Some(format!("{}某", surname))
                }
            }
            PrivacyCategory::Contact => Some(mask_contact(value)),
            PrivacyCategory::Credential => Some(mask_credential(value)),
            PrivacyCategory::Finance => Some(mask_finance(value)),
            PrivacyCategory::Health => Some("某类健康状况".to_string()),
            PrivacyCategory::Relationship => Some("某位亲友".to_string()),
            PrivacyCategory::Other => Some(value.to_string()),
        },
        DisclosurePrecision::Coarse => match category {
            PrivacyCategory::Location => cut_through(value, &['省'])
                .or_else(|| cut_through(value, &['市']))
                .or_else(|| Some(typed_placeholder(category, wh).to_string())),
            PrivacyCategory::Demographic => coarsen_demographic(value, 10),
            PrivacyCategory::PersonalIdentity => Some(typed_placeholder(category, wh).to_string()),
            PrivacyCategory::Contact => Some("某联系方式".to_string()),
            PrivacyCategory::Credential => Some("某证件信息".to_string()),
            PrivacyCategory::Finance => Some("某财务信息".to_string()),
            PrivacyCategory::Health => Some("某健康状况".to_string()),
            PrivacyCategory::Relationship => Some("某位亲友".to_string()),
            PrivacyCategory::Other => Some(value.to_string()),
        },
    }
}
fn cut_through(value: &str, suffixes: &[char]) -> Option<String> {
    for (index, character) in value.char_indices() {
        if suffixes.contains(&character) {
            return Some(value[..index + character.len_utf8()].to_string());
        }
    }
    None
}
fn coarsen_demographic(value: &str, width: u32) -> Option<String> {
    if value.contains('岁') {
        if let Some(age) = extract_digits(value) {
            if age > 130 {
                return Some(
                    typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Quantity))
                        .to_string(),
                );
            }
            let start = (age / width) * width;
            return Some(format!("{}-{}岁", start, start + width.saturating_sub(1)));
        }
    }
    if let Some(year) = extract_year(value) {
        if (1000..=2999).contains(&year) {
            if width <= 5 {
                return Some(format!("{}年", year));
            }
            return Some(format!("{}年代", year / 10 * 10));
        }
    }
    Some(typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Quantity)).to_string())
}
fn extract_year(value: &str) -> Option<u32> {
    let digits: String = value
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    if digits.len() < 4 {
        return None;
    }
    digits[..4].parse().ok()
}
fn extract_digits(value: &str) -> Option<u32> {
    let digits: String = value
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}
fn mask_contact(value: &str) -> String {
    if let Some((local, domain)) = value.split_once('@') {
        let first: String = local.chars().take(1).collect();
        return format!("{}***@{}", first, domain);
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 7 {
        let head: String = chars.iter().take(3).collect();
        let tail: String = chars.iter().skip(chars.len() - 2).collect();
        format!("{}****{}", head, tail)
    } else {
        "某联系方式".to_string()
    }
}
fn mask_credential(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 6 {
        let head: String = chars.iter().take(3).collect();
        let tail: String = chars.iter().skip(chars.len() - 1).collect();
        format!("{}****{}", head, tail)
    } else {
        "某证件信息".to_string()
    }
}
fn mask_finance(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 8 {
        let tail: String = chars.iter().skip(chars.len() - 4).collect();
        format!("****{}", tail)
    } else {
        "某财务信息".to_string()
    }
}
fn fact_digest(subject: &str, predicate: &str, object: &str) -> String {
    let mut material = String::with_capacity(subject.len() + predicate.len() + object.len() + 2);
    material.push_str(subject);
    material.push('\u{1}');
    material.push_str(predicate);
    material.push('\u{1}');
    material.push_str(object);
    crate::context_image::fnv1a_hex(material.as_bytes())
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

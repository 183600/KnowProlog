//! 回答计划（Answer Plan）与风格实现（Realization）。
//!
//! 生成式回答被拆成两层：
//!
//! - **计划层**：一组已经过生成自检、带稳定槽位与来源的命题。它只回答
//!   「说什么」，随会话快照持久化，重启后仍可复述；
//! - **表达层**：`StyleDirective` 只决定取哪些点、按什么顺序、是否披露
//!   来源与容量，绝不引入计划之外的 claim。
//!
//! 因此「简短点」「完整讲讲」「换种表达」「第二点是什么」都是对同一份
//! 计划的符号操作，而不是重新检索或继续枚举。计划带内容指纹、知识版本
//! 与容量上限：知识版本一变就拒绝旧计划，避免两代事实混在一段回答里。

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::confidence::ClaimEnvelope;
use crate::context_image::fnv1a_hex;
use crate::media_evidence::{Locator, MediaCitation};
use crate::resource_ledger::{ResourceKind, ResourceLedger, ResourceVector};

/// 回答计划的结构版本；不兼容变更时递增。
pub const ANSWER_PLAN_VERSION: u32 = 1;

/// 单份计划最多保留的命题数：限制快照体积与重述成本。
pub const MAX_PLAN_CLAIMS: usize = 24;

/// 简短模式默认展示的命题数。
pub const BRIEF_CLAIM_LIMIT: usize = 3;

/// 一条命题所依据的登记事实（不含推断过程，只含可核查的三元组）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClaimEvidence {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// 是否来自会话叠加层（未写入图谱）。
    #[serde(default, skip_serializing_if = "is_false")]
    pub session: bool,
}

impl ClaimEvidence {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            session,
        }
    }

    pub fn render(&self) -> String {
        format!(
            "{} {} {} {}",
            if self.session { "会话" } else { "图谱" },
            self.subject,
            self.predicate,
            self.object
        )
    }
}

/// 回答计划中的媒体引用：结构化定位 + 整数置信度，保持计划可比较。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanMediaCitation {
    pub asset_id: String,
    pub asset_label: String,
    pub anchor_id: String,
    pub locator: Locator,
    pub extractor: String,
    /// 置信度的万分之一整数表示（0..=10000），避免浮点进入计划身份。
    pub confidence_bps: u16,
}

impl PlanMediaCitation {
    pub fn render_zh(&self) -> String {
        format!(
            "{} · {} · {}（置信度 {:.2}）",
            self.asset_label,
            self.locator.label_zh(),
            self.extractor,
            self.confidence_bps as f64 / 10_000.0
        )
    }
}

impl From<&MediaCitation> for PlanMediaCitation {
    fn from(citation: &MediaCitation) -> Self {
        let confidence_bps = (citation.confidence * 10_000.0)
            .round()
            .clamp(0.0, 10_000.0) as u16;
        Self {
            asset_id: citation.asset_id.as_str().to_string(),
            asset_label: citation.asset_label.clone(),
            anchor_id: citation.anchor_id.as_str().to_string(),
            locator: citation.locator.clone(),
            extractor: citation.extractor.clone(),
            confidence_bps,
        }
    }
}

/// 一条命题的推导依据：规则来源 + 前提事实 + 有界深度与整数置信度。
///
/// 与 [`ClaimEvidence`] 的分工：证据回答「依据哪些登记事实」，推导回答
/// 「这些事实如何由哪条规则推出结论」。直接事实的命题没有推导；推出命题
/// 同时携带前提证据与推导，表达层才能在详细模式里说清断言与推断的边界。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimJustification {
    /// 规则来源（教学规则、链接预测规则、本体公理等）。
    pub rule_source: String,
    /// 规则体实际使用的前提三元组（已去重、排序）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub premises: Vec<ClaimEvidence>,
    /// 前向饱和中的推导轮次（直接事实为 0）。
    pub depth: usize,
    /// 推导置信度的万分之一整数表示（0..=10000）。
    pub confidence_bps: u16,
}

impl ClaimJustification {
    pub fn new(
        rule_source: impl Into<String>,
        mut premises: Vec<ClaimEvidence>,
        depth: usize,
        confidence_bps: u16,
    ) -> Self {
        premises.sort();
        premises.dedup();
        Self {
            rule_source: rule_source.into(),
            premises,
            depth,
            confidence_bps: confidence_bps.min(10_000),
        }
    }

    /// 简短的中文依据说明；前提过多时只展开前三条并给出总数。
    pub fn render_zh(&self) -> String {
        const SHOWN: usize = 3;
        let mut premises = self
            .premises
            .iter()
            .take(SHOWN)
            .map(ClaimEvidence::render)
            .collect::<Vec<_>>()
            .join("；");
        if self.premises.is_empty() {
            premises = "无登记前提".to_string();
        } else if self.premises.len() > SHOWN {
            let _ = write!(premises, "；等 {} 条", self.premises.len());
        }
        format!(
            "由「{}」推出（第 {} 轮，置信度 {:.2}；前提：{}）",
            self.rule_source,
            self.depth,
            self.confidence_bps as f64 / 10_000.0,
            premises
        )
    }
}

/// 一条可被表达层选择的命题。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanClaim {
    /// 稳定槽位（由生成器的 `ClauseKey` 编码而来）。
    pub slot: String,
    /// 规范化谓词；用于详细模式的出处标签与换序表达。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub predicate: String,
    /// 已经过生成自检的完整句子。
    pub text: String,
    /// 是否来自本会话叠加层（未写入图谱）。
    #[serde(default, skip_serializing_if = "is_false")]
    pub session: bool,
    /// 逐命题置信区间；旧计划缺省为 None，序列化字节保持不变。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<ClaimEnvelope>,
    /// 该命题依据的事实三元组；空证据跳过序列化，旧计划字节不变。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<ClaimEvidence>,
    /// 推出命题的规则依据；直接事实缺省，旧计划字节不变。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub justification: Option<ClaimJustification>,
    /// 该命题可引用的媒体锚点；空引用跳过序列化，旧计划字节不变。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<PlanMediaCitation>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl PlanClaim {
    pub fn new(
        slot: impl Into<String>,
        predicate: impl Into<String>,
        text: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            slot: slot.into(),
            predicate: predicate.into(),
            text: text.into(),
            session,
            envelope: None,
            evidence: Vec::new(),
            justification: None,
            media: Vec::new(),
        }
    }

    /// 附加/覆盖逐命题置信区间；不改变命题内容。
    pub fn with_envelope(mut self, envelope: ClaimEnvelope) -> Self {
        self.envelope = Some(envelope);
        self
    }

    /// 附加/覆盖命题所依据的登记事实；空证据不会写入序列化。
    pub fn with_evidence(mut self, evidence: Vec<ClaimEvidence>) -> Self {
        self.evidence = evidence;
        self
    }

    /// 附加/覆盖命题的推导依据；直接事实不调用，序列化字节保持不变。
    pub fn with_justification(mut self, justification: ClaimJustification) -> Self {
        self.justification = Some(justification);
        self
    }

    /// 附加/覆盖命题可引用的媒体锚点；空引用不会写入序列化。
    pub fn with_media_citations(mut self, media: Vec<PlanMediaCitation>) -> Self {
        self.media = media;
        self
    }

    fn evidence_note(&self) -> String {
        if self.evidence.is_empty() {
            return String::new();
        }
        format!("；依据：{}", self.evidence_digest())
    }

    /// 详细模式才展开推导；简短模式只保留命题与媒体摘要。
    fn justification_note(&self, detailed: bool) -> String {
        match (&self.justification, detailed) {
            (Some(justification), true) => format!("；推导：{}", justification.render_zh()),
            _ => String::new(),
        }
    }

    fn media_note(&self, detailed: bool) -> String {
        if self.media.is_empty() {
            return String::new();
        }
        let mut shown: Vec<&PlanMediaCitation> = if detailed {
            self.media.iter().collect()
        } else {
            self.media
                .iter()
                .max_by(|left, right| {
                    left.confidence_bps
                        .cmp(&right.confidence_bps)
                        .then_with(|| right.asset_id.cmp(&left.asset_id))
                        .then_with(|| right.anchor_id.cmp(&left.anchor_id))
                })
                .into_iter()
                .collect()
        };
        shown.sort_by(|left, right| {
            left.asset_id
                .cmp(&right.asset_id)
                .then_with(|| left.anchor_id.cmp(&right.anchor_id))
        });
        let rendered = shown
            .iter()
            .map(|citation| citation.render_zh())
            .collect::<Vec<_>>()
            .join("；");
        if detailed && self.media.len() > 3 {
            format!("；媒体：{}；等 {} 条", rendered, self.media.len())
        } else {
            format!("；媒体：{}", rendered)
        }
    }

    fn media_digest(&self) -> String {
        if self.media.is_empty() {
            return "无".to_string();
        }
        self.media
            .iter()
            .map(PlanMediaCitation::render_zh)
            .collect::<Vec<_>>()
            .join("；")
    }

    fn evidence_digest(&self) -> String {
        if self.evidence.is_empty() {
            return "无".to_string();
        }
        const SHOWN: usize = 3;
        let mut digest = self
            .evidence
            .iter()
            .take(SHOWN)
            .map(ClaimEvidence::render)
            .collect::<Vec<_>>()
            .join("；");
        if self.evidence.len() > SHOWN {
            let _ = write!(digest, "；等 {} 条", self.evidence.len());
        }
        digest
    }

    fn envelope_note(&self) -> String {
        match self.envelope {
            Some(envelope) => format!("；置信区间：{}", envelope.render_interval()),
            None => String::new(),
        }
    }

    fn source_label(&self) -> &'static str {
        if self.session {
            "会话事实"
        } else {
            "图谱事实"
        }
    }
}

/// 一份可序列化、可追溯的回答计划。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerPlan {
    #[serde(default = "current_version")]
    pub version: u32,
    /// 内容指纹（与命题顺序无关，同内容同 id）。
    pub id: String,
    /// 触发本次计划的问句或请求。
    #[serde(default)]
    pub question: String,
    pub subject: String,
    /// 计划生成时的知识版本。
    pub revision: u64,
    pub claims: Vec<PlanClaim>,
    /// 因容量上限未纳入计划的命题数。
    #[serde(default)]
    pub omitted: usize,
    /// 因生成自检未通过而剔除的命题数。
    #[serde(default)]
    pub suppressed: usize,
}

fn current_version() -> u32 {
    ANSWER_PLAN_VERSION
}

impl AnswerPlan {
    /// 从已过滤的命题建立计划；容量上限在这里统一执行。
    pub fn new(
        question: impl Into<String>,
        subject: impl Into<String>,
        revision: u64,
        claims: Vec<PlanClaim>,
        omitted: usize,
        suppressed: usize,
    ) -> Self {
        let mut claims = claims;
        let overflow = claims.len().saturating_sub(MAX_PLAN_CLAIMS);
        claims.truncate(MAX_PLAN_CLAIMS);
        let mut plan = Self {
            version: ANSWER_PLAN_VERSION,
            id: String::new(),
            question: question.into(),
            subject: subject.into(),
            revision,
            claims,
            omitted: omitted.saturating_add(overflow),
            suppressed,
        };
        plan.id = plan.compute_id();
        plan
    }

    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    pub fn is_current_schema(&self) -> bool {
        self.version == ANSWER_PLAN_VERSION
    }

    /// 计划是否仍然对应当前知识版本。
    pub fn is_fresh_for(&self, revision: u64) -> bool {
        self.revision == revision
    }

    /// 内容指纹：命题按槽位与文本排序后参与哈希，展示顺序不影响 id。
    fn compute_id(&self) -> String {
        let mut claims: Vec<&PlanClaim> = self.claims.iter().collect();
        claims.sort_by(|left, right| {
            left.slot
                .cmp(&right.slot)
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| left.predicate.cmp(&right.predicate))
        });
        let mut material = String::new();
        let _ = write!(
            material,
            "v{}|{}|{}|{}|{}",
            self.version, self.subject, self.revision, self.omitted, self.suppressed
        );
        for claim in claims {
            let _ = write!(
                material,
                "\n{}|{}|{}|{}",
                claim.slot, claim.predicate, claim.text, claim.session
            );
            let mut media: Vec<&PlanMediaCitation> = claim.media.iter().collect();
            media.sort_by(|left, right| {
                left.asset_id
                    .cmp(&right.asset_id)
                    .then_with(|| left.anchor_id.cmp(&right.anchor_id))
                    .then_with(|| left.confidence_bps.cmp(&right.confidence_bps))
            });
            for citation in media {
                let _ = write!(
                    material,
                    "|m:{}|{}|{}|{}|{}",
                    citation.asset_id,
                    citation.anchor_id,
                    citation.locator.label_zh(),
                    citation.extractor,
                    citation.confidence_bps
                );
            }
            if let Some(envelope) = claim.envelope {
                let _ = write!(
                    material,
                    "|{}:{}",
                    envelope.lower_bps(),
                    envelope.upper_bps()
                );
            }
            let mut evidence: Vec<&ClaimEvidence> = claim.evidence.iter().collect();
            evidence.sort();
            for item in evidence {
                let _ = write!(
                    material,
                    "|e:{}|{}|{}|{}",
                    item.subject, item.predicate, item.object, item.session
                );
            }
            if let Some(justification) = &claim.justification {
                let _ = write!(
                    material,
                    "|j:{}|{}|{}",
                    justification.rule_source, justification.depth, justification.confidence_bps
                );
                let mut premises: Vec<&ClaimEvidence> = justification.premises.iter().collect();
                premises.sort();
                for item in premises {
                    let _ = write!(
                        material,
                        "|jp:{}|{}|{}|{}",
                        item.subject, item.predicate, item.object, item.session
                    );
                }
            }
        }
        fnv1a_hex(material.as_bytes())
    }

    /// 按风格指令表达同一份计划。
    pub fn realize(&self, directive: StyleDirective) -> Result<String, PlanRenderError> {
        self.surface_units(directive).map(|stream| stream.concat())
    }

    /// 把同一份表达拆成可逐单位提交的表层事件。
    ///
    /// 不变量：`surface_units(d).concat()` 与 `realize(d)` 逐字节相等。流式
    /// 输出因此不是第二套生成器，而只是同一份计划的增量呈现；单位索引、
    /// 计划 id 与表达标签共同构成可恢复的事件 id。
    pub fn surface_units(
        &self,
        directive: StyleDirective,
    ) -> Result<SurfaceUnits, PlanRenderError> {
        if self.claims.is_empty() {
            return Err(PlanRenderError::Empty {
                subject: self.subject.clone(),
            });
        }
        if let StyleDirective::Budget(limit) = directive {
            return self.budgeted_surface(limit).map(|budgeted| budgeted.units);
        }
        let mut builder = SurfaceBuilder::new(self, directive);
        match directive {
            StyleDirective::Brief => self.build_window(&mut builder, BRIEF_CLAIM_LIMIT, false),
            StyleDirective::Detailed => self.build_window(&mut builder, usize::MAX, true),
            StyleDirective::Restate => self.build_restate(&mut builder),
            StyleDirective::Focus(index) => self.build_focus(&mut builder, index)?,
            StyleDirective::Budget(_) => unreachable!("预算指令已在上方分流"),
        }
        Ok(builder.finish())
    }

    /// 在字符预算内选择完整命题并表达同一份计划。
    ///
    /// 选择只按计划原顺序取前缀：不切断句子、不改写命题，也不引入计划外
    /// 内容。抬头、截断提示与计划容量提示都计入预算；预算连第一条完整命题
    /// 都放不下时显式返回 [`PlanRenderError::BudgetTooSmall`]，并给出最小
    /// 可用字符数，而不是输出空白或半句话。
    pub fn budgeted_surface(
        &self,
        budget_chars: usize,
    ) -> Result<BudgetedSurface, PlanRenderError> {
        if self.claims.is_empty() {
            return Err(PlanRenderError::Empty {
                subject: self.subject.clone(),
            });
        }
        let total = self.claims.len();
        let mut best: Option<(usize, SurfaceUnits)> = None;
        for kept in 1..=total {
            let units = self.build_budget_surface(budget_chars, kept);
            if units.concat().chars().count() <= budget_chars {
                best = Some((kept, units));
            }
        }
        let Some((kept_claims, units)) = best else {
            let minimum = self
                .build_budget_surface(budget_chars, 1)
                .concat()
                .chars()
                .count();
            return Err(PlanRenderError::BudgetTooSmall {
                subject: self.subject.clone(),
                budget: budget_chars,
                minimum,
            });
        };
        let mut ledger = ResourceLedger::new(ResourceVector::output_only(budget_chars));
        for unit in &units.units {
            ledger
                .try_charge(ResourceKind::OutputChars, unit.text.chars().count())
                .map_err(|denial| PlanRenderError::BudgetTooSmall {
                    subject: self.subject.clone(),
                    budget: budget_chars,
                    minimum: denial.used.saturating_add(denial.requested),
                })?;
        }
        Ok(BudgetedSurface {
            units,
            ledger,
            kept_claims,
            omitted_claims: total.saturating_sub(kept_claims),
        })
    }

    fn build_budget_surface(&self, budget_chars: usize, kept: usize) -> SurfaceUnits {
        let total = self.claims.len();
        let kept = kept.clamp(1, total.max(1));
        let mut builder = SurfaceBuilder::new(self, StyleDirective::Budget(budget_chars));
        builder.push(
            SurfaceUnitKind::Header,
            format!(
                "预算版（保留 {}/{} 点，预算 {} 字，未新增事实）：\n",
                kept, total, budget_chars
            ),
            None,
            None,
        );
        for (index, claim) in self.claims.iter().take(kept).enumerate() {
            builder.push(
                SurfaceUnitKind::Claim,
                format!(
                    "  {}. {}{}\n",
                    index + 1,
                    claim.text.trim(),
                    claim.media_note(false)
                ),
                Some(&claim.slot),
                Some(index + 1),
            );
        }
        if kept < total {
            builder.push(
                SurfaceUnitKind::Omission,
                format!("  ……其余 {} 点因预算未展开。\n", total - kept),
                None,
                None,
            );
        }
        let mut caveats = Vec::new();
        if self.suppressed > 0 {
            caveats.push(format!("{} 句未通过自检", self.suppressed));
        }
        if self.omitted > 0 {
            caveats.push(format!("{} 点超出计划容量", self.omitted));
        }
        if !caveats.is_empty() {
            builder.push(
                SurfaceUnitKind::Capacity,
                format!("  （计划另有{}。）\n", caveats.join("、")),
                None,
                None,
            );
        }
        builder.finish()
    }

    fn build_window(&self, builder: &mut SurfaceBuilder, limit: usize, detailed: bool) {
        let total = self.claims.len();
        let shown = limit.min(total);
        builder.push(
            SurfaceUnitKind::Header,
            if detailed {
                format!(
                    "完整版（{}/{} 点，计划 {}，知识版本 v{}）：\n",
                    shown, total, self.id, self.revision
                )
            } else {
                format!("简短版（取前 {}/{} 点，未新增事实）：\n", shown, total)
            },
            None,
            None,
        );
        for (index, claim) in self.claims.iter().take(shown).enumerate() {
            builder.push(
                SurfaceUnitKind::Claim,
                format!(
                    "  {}. {}{}\n",
                    index + 1,
                    claim.text.trim(),
                    if detailed {
                        format!(
                            "（谓词：{}；来源：{}）{}",
                            claim.predicate,
                            claim.source_label(),
                            format!(
                                "{}{}{}{}",
                                claim.envelope_note(),
                                claim.evidence_note(),
                                claim.justification_note(true),
                                claim.media_note(true)
                            )
                        )
                    } else {
                        claim.media_note(false)
                    }
                ),
                Some(&claim.slot),
                Some(index + 1),
            );
        }
        if shown < total {
            builder.push(
                SurfaceUnitKind::Omission,
                format!(
                    "  ……其余 {} 点未展开；输入「完整讲讲」查看全部。\n",
                    total - shown
                ),
                None,
                None,
            );
        }
        self.build_plan_notes(builder, detailed);
    }

    fn build_focus(
        &self,
        builder: &mut SurfaceBuilder,
        index: usize,
    ) -> Result<(), PlanRenderError> {
        let total = self.claims.len();
        let Some(claim) = self.claims.get(index.saturating_sub(1)) else {
            return Err(PlanRenderError::FocusOutOfRange {
                subject: self.subject.clone(),
                requested: index,
                total,
            });
        };
        builder.push(
            SurfaceUnitKind::Header,
            format!(
                "第 {} 点（共 {} 点，计划 {}，知识版本 v{}）：\n",
                index, total, self.id, self.revision
            ),
            None,
            None,
        );
        builder.push(
            SurfaceUnitKind::Claim,
            format!(
                "  {}（谓词：{}；来源：{}）{}\n",
                claim.text.trim(),
                claim.predicate,
                claim.source_label(),
                format!(
                    "{}{}{}{}",
                    claim.envelope_note(),
                    claim.evidence_note(),
                    claim.justification_note(true),
                    claim.media_note(true)
                )
            ),
            Some(&claim.slot),
            Some(index),
        );
        if total > 1 {
            builder.push(
                SurfaceUnitKind::Followup,
                format!("  输入「完整讲讲」查看全部 {} 点。\n", total),
                None,
                None,
            );
        }
        self.build_plan_notes(builder, true);
        Ok(())
    }

    fn build_restate(&self, builder: &mut SurfaceBuilder) {
        let mut indices: Vec<usize> = (0..self.claims.len()).collect();
        indices.sort_by(|left, right| {
            let left = &self.claims[*left];
            let right = &self.claims[*right];
            left.predicate
                .cmp(&right.predicate)
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        builder.push(
            SurfaceUnitKind::Header,
            format!(
                "换种表达（同一份计划按主题重排，共 {} 点，未新增事实）：\n",
                self.claims.len()
            ),
            None,
            None,
        );
        for index in indices {
            let claim = &self.claims[index];
            builder.push(
                SurfaceUnitKind::Claim,
                format!("  • {}\n", claim.text.trim()),
                Some(&claim.slot),
                Some(index + 1),
            );
        }
        self.build_plan_notes(builder, false);
    }

    /// 比较两份计划，按稳定槽位给出可审计的差异。
    ///
    /// 差异只比较「计划里说了什么」，不重新解释问题，也不改写旧回答；
    /// 因此旧计划即使已过期，仍能作为对照物保留。槽位相同但文本或来源
    /// 变化记为更新，知识变化导致的消失/新增分别记为移除/新增。
    pub fn diff(&self, current: &AnswerPlan) -> PlanDiff {
        let mut matched: Vec<bool> = vec![false; current.claims.len()];
        let mut changes: Vec<PlanClaimChange> = Vec::new();
        for before in &self.claims {
            let exact = current
                .claims
                .iter()
                .enumerate()
                .find(|(index, after)| {
                    !matched[*index]
                        && after.slot == before.slot
                        && after.predicate == before.predicate
                        && after.text == before.text
                        && after.session == before.session
                })
                .map(|(index, _)| index);
            let same_slot = exact.or_else(|| {
                current
                    .claims
                    .iter()
                    .enumerate()
                    .find(|(index, after)| !matched[*index] && after.slot == before.slot)
                    .map(|(index, _)| index)
            });
            match same_slot {
                Some(index) => {
                    matched[index] = true;
                    let after = &current.claims[index];
                    let kind = if after == before {
                        PlanChangeKind::Unchanged
                    } else {
                        PlanChangeKind::Updated
                    };
                    changes.push(PlanClaimChange {
                        slot: before.slot.clone(),
                        kind,
                        before: Some(before.clone()),
                        after: Some(after.clone()),
                    });
                }
                None => changes.push(PlanClaimChange {
                    slot: before.slot.clone(),
                    kind: PlanChangeKind::Removed,
                    before: Some(before.clone()),
                    after: None,
                }),
            }
        }
        for (index, after) in current.claims.iter().enumerate() {
            if !matched[index] {
                changes.push(PlanClaimChange {
                    slot: after.slot.clone(),
                    kind: PlanChangeKind::Added,
                    before: None,
                    after: Some(after.clone()),
                });
            }
        }
        changes.sort_by(|left, right| {
            left.slot
                .cmp(&right.slot)
                .then_with(|| left.kind.rank().cmp(&right.kind.rank()))
                .then_with(|| {
                    left.before
                        .as_ref()
                        .map(|claim| claim.text.as_str())
                        .unwrap_or("")
                        .cmp(
                            right
                                .before
                                .as_ref()
                                .map(|claim| claim.text.as_str())
                                .unwrap_or(""),
                        )
                })
                .then_with(|| {
                    left.after
                        .as_ref()
                        .map(|claim| claim.text.as_str())
                        .unwrap_or("")
                        .cmp(
                            right
                                .after
                                .as_ref()
                                .map(|claim| claim.text.as_str())
                                .unwrap_or(""),
                        )
                })
        });
        PlanDiff {
            base_id: self.id.clone(),
            current_id: current.id.clone(),
            base_revision: self.revision,
            current_revision: current.revision,
            changes,
        }
    }

    fn build_plan_notes(&self, builder: &mut SurfaceBuilder, detailed: bool) {
        if self.suppressed > 0 {
            builder.push(
                SurfaceUnitKind::Suppression,
                format!(
                    "  （另有 {} 句未通过生成自检，未纳入计划。）\n",
                    self.suppressed
                ),
                None,
                None,
            );
        }
        if self.omitted > 0 {
            builder.push(
                SurfaceUnitKind::Capacity,
                format!(
                    "  （计划容量上限 {} 点，另有 {} 点未纳入。）\n",
                    MAX_PLAN_CLAIMS, self.omitted
                ),
                None,
                None,
            );
        }
        if detailed && self.claims.len() > BRIEF_CLAIM_LIMIT {
            builder.push(
                SurfaceUnitKind::Followup,
                format!(
                    "  （说「简短点」可回到前 {} 点摘要。）\n",
                    BRIEF_CLAIM_LIMIT
                ),
                None,
                None,
            );
        }
    }
}

/// 表层单位类别：只决定客户端如何呈现或续传，不改变回答内容。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceUnitKind {
    /// 版本/风格抬头。
    Header,
    /// 一条计划命题（可对应到稳定槽位）。
    Claim,
    /// 简短版截断提示。
    Omission,
    /// 生成自检剔除计数。
    Suppression,
    /// 计划容量上限计数。
    Capacity,
    /// 后续可执行动作提示。
    Followup,
}

impl SurfaceUnitKind {
    pub fn code(self) -> &'static str {
        match self {
            SurfaceUnitKind::Header => "header",
            SurfaceUnitKind::Claim => "claim",
            SurfaceUnitKind::Omission => "omission",
            SurfaceUnitKind::Suppression => "suppression",
            SurfaceUnitKind::Capacity => "capacity",
            SurfaceUnitKind::Followup => "followup",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            SurfaceUnitKind::Header => "抬头",
            SurfaceUnitKind::Claim => "命题",
            SurfaceUnitKind::Omission => "截断",
            SurfaceUnitKind::Suppression => "自检",
            SurfaceUnitKind::Capacity => "容量",
            SurfaceUnitKind::Followup => "后续",
        }
    }
}

/// 一个可按序提交的表层单位；`text` 是该单位要写出的精确字节。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceUnit {
    /// 在表层流中的位置（从 0 开始）。
    pub index: usize,
    pub kind: SurfaceUnitKind,
    /// 命题单位对应的计划槽位。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    /// 命题在计划中的展示序号（从 1 开始；换序表达仍指向原计划位次）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_index: Option<usize>,
    pub text: String,
}

/// 一份表达的全部表层单位，保证拼接后与 `AnswerPlan::realize` 完全一致。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceUnits {
    pub plan_id: String,
    pub revision: u64,
    /// 表达指令的稳定标签（`brief` / `detailed` / `restate` / `focus:N`）。
    pub directive: String,
    pub units: Vec<SurfaceUnit>,
}

impl SurfaceUnits {
    pub fn concat(&self) -> String {
        self.units.iter().map(|unit| unit.text.as_str()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SurfaceUnit> {
        self.units.iter()
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// 第 `position` 个单位的事件 id；同一计划、同一表达下稳定可复现。
    pub fn event_id(&self, position: usize) -> Option<String> {
        self.units
            .get(position)
            .map(|unit| format!("{}:{}:{}", self.plan_id, self.directive, unit.index))
    }
}

/// 预算约束下的一次表达：表层单位、资源账本与保留/省略计数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedSurface {
    pub units: SurfaceUnits,
    pub ledger: ResourceLedger,
    pub kept_claims: usize,
    pub omitted_claims: usize,
}

impl BudgetedSurface {
    pub fn concat(&self) -> String {
        self.units.concat()
    }

    pub fn used_chars(&self) -> usize {
        self.ledger.used_of(ResourceKind::OutputChars)
    }

    pub fn budget_chars(&self) -> usize {
        self.ledger.limits().output_chars
    }

    pub fn render_receipt(&self) -> String {
        format!(
            "预算表达：保留 {}/{} 点，{}；{}",
            self.kept_claims,
            self.kept_claims + self.omitted_claims,
            self.ledger.render_receipt(),
            if self.omitted_claims > 0 {
                "未展开命题只被省略，未被改写"
            } else {
                "全部命题均在预算内完整表达"
            }
        )
    }
}

struct SurfaceBuilder {
    plan_id: String,
    revision: u64,
    directive: String,
    units: Vec<SurfaceUnit>,
}

impl SurfaceBuilder {
    fn new(plan: &AnswerPlan, directive: StyleDirective) -> Self {
        Self {
            plan_id: plan.id.clone(),
            revision: plan.revision,
            directive: directive.tag(),
            units: Vec::new(),
        }
    }

    fn push(
        &mut self,
        kind: SurfaceUnitKind,
        text: impl Into<String>,
        slot: Option<&str>,
        claim_index: Option<usize>,
    ) {
        self.units.push(SurfaceUnit {
            index: self.units.len(),
            kind,
            slot: slot.map(str::to_string),
            claim_index,
            text: text.into(),
        });
    }

    fn finish(mut self) -> SurfaceUnits {
        if let Some(last) = self.units.last_mut() {
            let trimmed = last.text.trim_end();
            if trimmed.len() != last.text.len() {
                last.text.truncate(trimmed.len());
            }
        }
        SurfaceUnits {
            plan_id: self.plan_id,
            revision: self.revision,
            directive: self.directive,
            units: self.units,
        }
    }
}

/// 一条计划差异的类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanChangeKind {
    /// 同一槽位、同一内容（知识版本仍可能变化）。
    Unchanged,
    /// 槽位相同，但文本、谓词或来源发生变化。
    Updated,
    /// 旧计划有、当前计划不再支持。
    Removed,
    /// 当前计划新增、旧计划没有。
    Added,
}

impl PlanChangeKind {
    fn rank(self) -> u8 {
        match self {
            PlanChangeKind::Updated => 0,
            PlanChangeKind::Removed => 1,
            PlanChangeKind::Added => 2,
            PlanChangeKind::Unchanged => 3,
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            PlanChangeKind::Unchanged => "未变",
            PlanChangeKind::Updated => "更新",
            PlanChangeKind::Removed => "移除",
            PlanChangeKind::Added => "新增",
        }
    }
}

/// 一条可逐项核对的计划差异。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanClaimChange {
    pub slot: String,
    pub kind: PlanChangeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<PlanClaim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<PlanClaim>,
}

/// 两份计划之间的差异回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDiff {
    pub base_id: String,
    pub current_id: String,
    pub base_revision: u64,
    pub current_revision: u64,
    pub changes: Vec<PlanClaimChange>,
}

impl PlanDiff {
    /// 是否只有「未变」项。
    pub fn is_unchanged(&self) -> bool {
        self.changes
            .iter()
            .all(|change| change.kind == PlanChangeKind::Unchanged)
    }

    /// 更新 / 移除 / 新增 / 未变的计数。
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let count = |kind| {
            self.changes
                .iter()
                .filter(|change| change.kind == kind)
                .count()
        };
        (
            count(PlanChangeKind::Updated),
            count(PlanChangeKind::Removed),
            count(PlanChangeKind::Added),
            count(PlanChangeKind::Unchanged),
        )
    }

    /// 渲染复核回执；`include_unchanged` 为 false 时只列变化项。
    pub fn render_zh(&self, include_unchanged: bool) -> String {
        let (updated, removed, added, unchanged) = self.counts();
        let mut output = format!(
            "回答计划复核：知识版本 v{} → v{}（旧计划 {}，当前计划 {}）\n",
            self.base_revision, self.current_revision, self.base_id, self.current_id
        );
        if updated + removed + added == 0 {
            output.push_str("  结论：计划内容未变");
            if self.base_revision != self.current_revision {
                output.push_str("（知识版本已更新，但逐点命题仍一致）");
            }
            output.push_str(&format!("；未变 {} 点。\n", unchanged));
        } else {
            output.push_str(&format!(
                "  变化：更新 {}，移除 {}，新增 {}，未变 {}。\n",
                updated, removed, added, unchanged
            ));
        }
        for change in &self.changes {
            if change.kind == PlanChangeKind::Unchanged && !include_unchanged {
                continue;
            }
            match change.kind {
                PlanChangeKind::Unchanged => {
                    let text = change
                        .after
                        .as_ref()
                        .map(|claim| claim.text.trim())
                        .unwrap_or("");
                    output.push_str(&format!("  [未变] {}：{}\n", change.slot, text));
                }
                PlanChangeKind::Updated => {
                    let before = change
                        .before
                        .as_ref()
                        .map(|claim| claim.text.trim())
                        .unwrap_or("");
                    let after = change
                        .after
                        .as_ref()
                        .map(|claim| claim.text.trim())
                        .unwrap_or("");
                    let evidence_only = match (&change.before, &change.after) {
                        (Some(before), Some(after)) => {
                            before.text == after.text
                                && before.predicate == after.predicate
                                && before.session == after.session
                                && before.evidence != after.evidence
                                && before.media == after.media
                        }
                        _ => false,
                    };
                    let media_only = match (&change.before, &change.after) {
                        (Some(before), Some(after)) => {
                            before.text == after.text
                                && before.predicate == after.predicate
                                && before.session == after.session
                                && before.evidence == after.evidence
                                && before.media != after.media
                        }
                        _ => false,
                    };
                    if media_only {
                        let before_media = change
                            .before
                            .as_ref()
                            .map(PlanClaim::media_digest)
                            .unwrap_or_else(|| "无".to_string());
                        let after_media = change
                            .after
                            .as_ref()
                            .map(PlanClaim::media_digest)
                            .unwrap_or_else(|| "无".to_string());
                        output.push_str(&format!(
                            "  [更新] {}：{}（媒体：{} → {}）\n",
                            change.slot, after, before_media, after_media
                        ));
                    } else if evidence_only {
                        let before_evidence = change
                            .before
                            .as_ref()
                            .map(PlanClaim::evidence_digest)
                            .unwrap_or_else(|| "无".to_string());
                        let after_evidence = change
                            .after
                            .as_ref()
                            .map(PlanClaim::evidence_digest)
                            .unwrap_or_else(|| "无".to_string());
                        output.push_str(&format!(
                            "  [更新] {}：{}（依据：{} → {}）\n",
                            change.slot, after, before_evidence, after_evidence
                        ));
                    } else {
                        output.push_str(&format!(
                            "  [更新] {}：{} → {}\n",
                            change.slot, before, after
                        ));
                    }
                }
                PlanChangeKind::Removed => {
                    let before = change
                        .before
                        .as_ref()
                        .map(|claim| claim.text.trim())
                        .unwrap_or("");
                    output.push_str(&format!(
                        "  [移除] {}：{}（当前知识不再支持）\n",
                        change.slot, before
                    ));
                }
                PlanChangeKind::Added => {
                    let after = change
                        .after
                        .as_ref()
                        .map(|claim| claim.text.trim())
                        .unwrap_or("");
                    output.push_str(&format!(
                        "  [新增] {}：{}（旧计划之后新增）\n",
                        change.slot, after
                    ));
                }
            }
        }
        output.push_str("  说明：复核只比较新旧计划，不重新解释问题，也不改写已经说出口的回答。\n");
        output.trim_end().to_string()
    }
}

/// 一次表达请求；只影响计划的选择与呈现。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleDirective {
    /// 只取计划前几点的简短版。
    Brief,
    /// 展开计划中的全部命题，并披露谓词与来源。
    Detailed,
    /// 同一组命题按主题重排，换一种说法。
    Restate,
    /// 只取第 N 点（从 1 开始）。
    Focus(usize),
    /// 在 N 个字符内选择完整命题；省略项显式披露。
    Budget(usize),
}

impl StyleDirective {
    /// 解析自然语言或显式前缀（`风格：…` / `回答计划：…`）。
    ///
    /// 刻意不接管已有的「详细一点」「展开说说」「换个说法」：这些短语
    /// 在对话管理器中已有明确的续说/换角度语义，不能悄悄改成重述。
    pub fn parse(input: &str) -> Option<Self> {
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', '；', ';', '，', ',', ' '])
            .trim();
        if stripped.is_empty() {
            return None;
        }
        for prefix in BUDGET_PREFIXES {
            if let Some(rest) = stripped.strip_prefix(prefix) {
                return parse_budget(rest, true).map(StyleDirective::Budget);
            }
        }
        for prefix in ["风格：", "风格:", "回答计划：", "回答计划:"] {
            if let Some(rest) = stripped.strip_prefix(prefix) {
                let rest = rest.trim();
                if rest.is_empty() {
                    return None;
                }
                if let Some(index) = parse_focus(rest, true) {
                    return Some(StyleDirective::Focus(index));
                }
                return match rest {
                    "简短" | "短" | "精简" | "要点" | "重点" => {
                        Some(StyleDirective::Brief)
                    }
                    "详细" | "完整" | "全部" | "展开" => Some(StyleDirective::Detailed),
                    "换种表达" | "换一种表达" | "换种说法" | "重排" | "重新组织" => {
                        Some(StyleDirective::Restate)
                    }
                    _ => None,
                };
            }
        }
        if BRIEF_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Brief);
        }
        if DETAILED_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Detailed);
        }
        if RESTATE_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Restate);
        }
        if let Some(budget) = parse_budget(stripped, false) {
            return Some(StyleDirective::Budget(budget));
        }
        parse_focus(stripped, false).map(StyleDirective::Focus)
    }

    pub fn label(self) -> &'static str {
        match self {
            StyleDirective::Brief => "简短版",
            StyleDirective::Detailed => "完整版",
            StyleDirective::Restate => "换种表达",
            StyleDirective::Focus(_) => "定位单点",
            StyleDirective::Budget(_) => "预算版",
        }
    }

    /// 稳定标签：进入表层事件 id，区分同一计划的不同表达。
    pub fn tag(self) -> String {
        match self {
            StyleDirective::Brief => "brief".to_string(),
            StyleDirective::Detailed => "detailed".to_string(),
            StyleDirective::Restate => "restate".to_string(),
            StyleDirective::Focus(index) => format!("focus:{}", index),
            StyleDirective::Budget(limit) => format!("budget:{}", limit),
        }
    }
}

const BUDGET_PREFIXES: &[&str] = &[
    "预算：",
    "预算:",
    "回答预算：",
    "回答预算:",
    "字数：",
    "字数:",
];

/// 解析「预算：120字」「控制在 80 字以内」这类长度指令。
///
/// 裸数字（如「120」）不接管：它可能是澄清短答或普通数值。只有显式前缀
/// 或「字 / 字以内 / 不超过 / 最多」等量词与限制词出现时才视为预算。
fn parse_budget(text: &str, explicit: bool) -> Option<usize> {
    let mut body = text.trim();
    let mut explicit = explicit;
    for prefix in ["回答字数", "回答字符数", "字数", "字符数"] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim();
            explicit = true;
            break;
        }
    }
    for prefix in ["控制在", "不超过", "最多", "限制在", "限制为", "限制", "限"] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim();
            explicit = true;
            break;
        }
    }
    for suffix in [
        "字以内",
        "字符以内",
        "个字以内",
        "字内",
        "字符",
        "个字",
        "字",
    ] {
        if let Some(rest) = body.strip_suffix(suffix) {
            body = rest.trim();
            explicit = true;
            break;
        }
    }
    if !explicit || body.is_empty() || !body.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    body.parse::<usize>().ok().filter(|value| *value > 0)
}

const BRIEF_PHRASES: &[&str] = &[
    "简短点",
    "简短一点",
    "简短些",
    "短一点",
    "再短点",
    "说简短点",
    "讲简短点",
    "简洁点",
    "简单点",
    "简单说说",
    "说简单点",
    "讲简单点",
    "精简点",
    "精简一下",
    "只讲重点",
    "只讲要点",
    "说重点",
    "讲重点",
    "一句话说说",
    "概括一下",
    "概括说说",
];

const DETAILED_PHRASES: &[&str] = &[
    "说详细点",
    "讲详细点",
    "完整讲讲",
    "完整说说",
    "完整说一遍",
    "全部展开",
    "展开全部",
    "详细展开",
    "详细版",
    "完整版",
    "回到详细版",
    "回到完整版",
    "全都说说",
];

const RESTATE_PHRASES: &[&str] = &[
    "换种表达",
    "换一种表达",
    "重新表达",
    "换种说法",
    "再表达一次",
    "重排一下",
    "换个叙述顺序",
];

fn parse_focus(text: &str, explicit: bool) -> Option<usize> {
    let mut body = text.trim();
    let mut prefixed = false;
    for prefix in ["展开", "回到", "看看", "讲讲", "说说", "看"] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim();
            prefixed = true;
            break;
        }
    }
    let had_di = if let Some(rest) = body.strip_prefix('第') {
        body = rest.trim();
        true
    } else {
        false
    };
    let (index, rest) = split_ordinal(body)?;
    let rest = rest.trim();
    let named_point = matches!(
        rest,
        "点" | "点呢"
            | "点是什么"
            | "点什么"
            | "点的内容"
            | "点内容"
            | "点讲讲"
            | "点说说"
            | "条"
            | "条呢"
            | "条是什么"
            | "条的内容"
            | "条内容"
            | "条讲讲"
            | "条说说"
    );
    // 裸数字「1」是澄清短答或普通数字，不是风格指令；只有显式前缀
    // 或「展开/第N点」这类带点标记的说法才进入计划定位。
    let suffix_ok = named_point || (rest.is_empty() && (explicit || prefixed));
    if !suffix_ok {
        return None;
    }
    let bare = had_di && matches!(rest, "点" | "条");
    if bare && !explicit && !prefixed {
        return None;
    }
    Some(index)
}

/// 读取开头的中文/阿拉伯序数，返回数值与剩余文本。
fn split_ordinal(text: &str) -> Option<(usize, &str)> {
    let mut chars = text.char_indices().peekable();
    let (_, first) = chars.next()?;
    if first.is_ascii_digit() {
        let end = text
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(text.len());
        let value: usize = text[..end].parse().ok()?;
        return (value > 0).then_some((value, &text[end..]));
    }
    let mut value = 0usize;
    let mut current = 0usize;
    let mut consumed = 0usize;
    for (offset, ch) in text.char_indices() {
        if let Some(digit) = chinese_digit(ch) {
            current = digit;
            consumed = offset + ch.len_utf8();
        } else if ch == '十' {
            let base = if current == 0 { 1 } else { current };
            value = value.saturating_add(base.saturating_mul(10));
            current = 0;
            consumed = offset + ch.len_utf8();
        } else if ch == '两' {
            current = 2;
            consumed = offset + ch.len_utf8();
        } else {
            break;
        }
    }
    if consumed == 0 {
        return None;
    }
    let value = value.saturating_add(current);
    (value > 0).then_some((value, &text[consumed..]))
}

fn chinese_digit(ch: char) -> Option<usize> {
    match ch {
        '一' => Some(1),
        '二' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

/// 表达失败：计划为空或定位越界。调用方据此给出可行动的提示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRenderError {
    Empty {
        subject: String,
    },
    FocusOutOfRange {
        subject: String,
        requested: usize,
        total: usize,
    },
    BudgetTooSmall {
        subject: String,
        budget: usize,
        minimum: usize,
    },
}

impl std::fmt::Display for PlanRenderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanRenderError::Empty { subject } => {
                write!(
                    formatter,
                    "关于「{}」的回答计划为空，没有可复述的命题。",
                    subject
                )
            }
            PlanRenderError::FocusOutOfRange {
                subject,
                requested,
                total,
            } => write!(
                formatter,
                "关于「{}」的回答计划只有 {} 点，无法定位第 {} 点。",
                subject, total, requested
            ),
            PlanRenderError::BudgetTooSmall {
                subject,
                budget,
                minimum,
            } => write!(
                formatter,
                "预算 {} 字不足以完整表达「{}」的第一条命题；至少需要 {} 字才能保留抬头、完整命题与省略说明。",
                budget, subject, minimum
            ),
        }
    }
}

impl std::error::Error for PlanRenderError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_claims() -> Vec<PlanClaim> {
        vec![
            PlanClaim::new("a#age", "age", "张三的年龄是18岁。", false),
            PlanClaim::new("r#师傅", "师傅", "张三的师傅是李四。", false),
            PlanClaim::new("a#出生地", "出生地", "张三的出生地是北京。", true),
            PlanClaim::new("r#徒弟", "徒弟", "张三的徒弟是王五。", false),
            PlanClaim::new("a#身高", "身高", "张三的身高是175厘米。", false),
        ]
    }

    fn sample_plan() -> AnswerPlan {
        AnswerPlan::new("介绍：张三", "张三", 7, sample_claims(), 0, 0)
    }

    #[test]
    fn plan_id_is_order_independent_and_revision_sensitive() {
        let plan = sample_plan();
        let mut reversed = sample_claims();
        reversed.reverse();
        let other = AnswerPlan::new("换个问法", "张三", 7, reversed, 0, 0);
        assert_eq!(plan.id, other.id, "同内容不同展示顺序应共享计划 id");
        let newer = AnswerPlan::new("介绍：张三", "张三", 8, sample_claims(), 0, 0);
        assert_ne!(plan.id, newer.id, "知识版本必须进入内容指纹");
        assert!(plan.is_fresh_for(7));
        assert!(!plan.is_fresh_for(8));
    }

    #[test]
    fn brief_uses_only_prefix_claims_and_never_invents() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Brief).unwrap();
        assert!(text.contains("简短版"), "{}", text);
        for claim in plan.claims.iter().take(BRIEF_CLAIM_LIMIT) {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        for claim in plan.claims.iter().skip(BRIEF_CLAIM_LIMIT) {
            assert!(!text.contains(claim.text.trim()), "越界 claim：{}", text);
        }
        assert!(text.contains("完整讲讲"), "{}", text);
    }

    #[test]
    fn budget_keeps_whole_claims_within_a_hard_character_limit() {
        let plan = sample_plan();
        let budgeted = plan.budgeted_surface(90).unwrap();
        let text = budgeted.concat();
        assert!(text.chars().count() <= 90, "{}", text);
        assert!(text.contains("预算版"), "{}", text);
        assert!(budgeted.kept_claims >= 1);
        assert!(budgeted.kept_claims < plan.claim_count(), "{}", text);
        assert!(text.contains("因预算未展开"), "{}", text);
        for claim in plan.claims.iter().take(budgeted.kept_claims) {
            assert!(
                text.contains(claim.text.trim()),
                "保留命题必须完整：{}",
                text
            );
        }
        for claim in plan.claims.iter().skip(budgeted.kept_claims) {
            assert!(
                !text.contains(claim.text.trim()),
                "省略命题不得出现：{}",
                text
            );
        }
        assert_eq!(budgeted.used_chars(), text.chars().count());
        assert_eq!(budgeted.budget_chars(), 90);
        assert!(budgeted.ledger.fits_within_limits());
        assert!(budgeted.render_receipt().contains("预算表达"));
        assert_eq!(budgeted, plan.budgeted_surface(90).unwrap());
        assert_eq!(
            plan.surface_units(StyleDirective::Budget(90)).unwrap(),
            budgeted.units
        );
        assert_eq!(budgeted.units.directive, "budget:90");
    }

    #[test]
    fn tiny_budget_is_an_explicit_error_with_a_minimum() {
        let plan = sample_plan();
        let error = plan.budgeted_surface(8).unwrap_err();
        match &error {
            PlanRenderError::BudgetTooSmall {
                subject,
                budget,
                minimum,
            } => {
                assert_eq!(subject, "张三");
                assert_eq!(*budget, 8);
                assert!(*minimum > 8, "minimum={}", minimum);
            }
            other => panic!("应返回预算不足错误，实际：{:?}", other),
        }
        let text = error.to_string();
        assert!(text.contains("预算 8 字"), "{}", text);
        assert!(text.contains("至少需要"), "{}", text);
        assert!(text.contains("完整命题"), "{}", text);
    }

    #[test]
    fn generous_budget_keeps_every_claim_without_an_omission_note() {
        let plan = sample_plan();
        let budgeted = plan.budgeted_surface(10_000).unwrap();
        assert_eq!(budgeted.kept_claims, plan.claim_count());
        assert_eq!(budgeted.omitted_claims, 0);
        assert!(!budgeted.concat().contains("因预算未展开"));
        for claim in &plan.claims {
            assert!(budgeted.concat().contains(claim.text.trim()));
        }
    }

    #[test]
    fn budget_surface_counts_plan_caveats_inside_the_limit() {
        let plan = AnswerPlan::new("介绍：张三", "张三", 7, sample_claims(), 2, 1);
        let budgeted = plan.budgeted_surface(180).unwrap();
        let text = budgeted.concat();
        assert!(text.chars().count() <= 180, "{}", text);
        assert!(text.contains("1 句未通过自检"), "{}", text);
        assert!(text.contains("2 点超出计划容量"), "{}", text);
    }

    #[test]
    fn detailed_discloses_every_claim_and_its_source() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Detailed).unwrap();
        assert!(text.contains("完整版"), "{}", text);
        assert!(text.contains(&plan.id), "{}", text);
        assert!(text.contains("知识版本 v7"), "{}", text);
        for claim in &plan.claims {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        assert!(text.contains("来源：会话事实"), "{}", text);
        assert!(text.contains("来源：图谱事实"), "{}", text);
    }

    #[test]
    fn focus_returns_exactly_one_claim_and_reports_range_errors() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Focus(2)).unwrap();
        assert!(text.contains("第 2 点"), "{}", text);
        assert!(text.contains(plan.claims[1].text.trim()), "{}", text);
        assert!(!text.contains(plan.claims[0].text.trim()), "{}", text);
        assert!(!text.contains(plan.claims[2].text.trim()), "{}", text);

        let error = plan.realize(StyleDirective::Focus(9)).unwrap_err();
        assert!(error.to_string().contains("只有 5 点"), "{}", error);
        assert!(error.to_string().contains("第 9 点"), "{}", error);
    }

    #[test]
    fn restate_preserves_the_claim_set_but_reorders_it() {
        let plan = AnswerPlan::new(
            "介绍：张三",
            "张三",
            7,
            vec![
                PlanClaim::new("z", "zeta", "先讲的命题。", false),
                PlanClaim::new("a", "alpha", "后讲的命题。", false),
            ],
            0,
            0,
        );
        let text = plan.realize(StyleDirective::Restate).unwrap();
        assert!(text.contains("换种表达"), "{}", text);
        for claim in &plan.claims {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        let first_restated = text
            .lines()
            .find(|line| line.trim_start().starts_with('•'))
            .expect("换种表达应输出条目");
        assert!(
            first_restated.contains("后讲的命题"),
            "按谓词重排后应把 alpha 放到最前：{}",
            text
        );
    }

    #[test]
    fn empty_plan_is_an_explicit_error_not_a_blank_answer() {
        let plan = AnswerPlan::new("介绍：无人", "无人", 1, Vec::new(), 0, 0);
        let error = plan.realize(StyleDirective::Brief).unwrap_err();
        assert!(error.to_string().contains("无人"), "{}", error);
    }

    #[test]
    fn capacity_and_suppression_are_disclosed() {
        let mut claims = Vec::new();
        for index in 0..(MAX_PLAN_CLAIMS + 2) {
            claims.push(PlanClaim::new(
                format!("s{}", index),
                "p",
                format!("第{}条事实。", index),
                false,
            ));
        }
        let plan = AnswerPlan::new("介绍：很多", "很多", 1, claims, 0, 3);
        assert_eq!(plan.claim_count(), MAX_PLAN_CLAIMS);
        assert_eq!(plan.omitted, 2);
        let text = plan.realize(StyleDirective::Detailed).unwrap();
        assert!(text.contains("未通过生成自检"), "{}", text);
        assert!(text.contains("容量上限"), "{}", text);
    }

    #[test]
    fn parser_accepts_new_style_words_and_leaves_existing_followups_alone() {
        for (input, expected) in [
            ("简短点", StyleDirective::Brief),
            ("短一点", StyleDirective::Brief),
            ("简单说说", StyleDirective::Brief),
            ("只讲重点", StyleDirective::Brief),
            ("完整讲讲", StyleDirective::Detailed),
            ("回到详细版", StyleDirective::Detailed),
            ("换种表达", StyleDirective::Restate),
            ("第二点是什么", StyleDirective::Focus(2)),
            ("第二条是什么", StyleDirective::Focus(2)),
            ("展开第三点", StyleDirective::Focus(3)),
            ("风格：第2点", StyleDirective::Focus(2)),
            ("回答计划：简短", StyleDirective::Brief),
            ("风格：完整", StyleDirective::Detailed),
            ("风格：换种表达", StyleDirective::Restate),
            ("预算：120字", StyleDirective::Budget(120)),
            ("预算:80", StyleDirective::Budget(80)),
            ("回答预算：60字以内", StyleDirective::Budget(60)),
            ("控制在 45 字", StyleDirective::Budget(45)),
            ("最多30字", StyleDirective::Budget(30)),
            ("120字以内", StyleDirective::Budget(120)),
            ("字数不超过100字", StyleDirective::Budget(100)),
            ("回答字数不超过90字", StyleDirective::Budget(90)),
        ] {
            assert_eq!(
                StyleDirective::parse(input),
                Some(expected),
                "应识别「{}」",
                input
            );
        }
        for input in [
            "详细一点",
            "展开说说",
            "展开讲",
            "换个说法",
            "换个角度说",
            "接着说",
            "第二点",
            "第二条",
            "第2条",
            "1",
            "2",
            "二十",
            "120",
            "预算：0字",
            "预算：很多字",
            "介绍一下北京",
        ] {
            assert_eq!(StyleDirective::parse(input), None, "不应接管「{}」", input);
        }
    }

    #[test]
    fn diff_reports_updated_removed_and_added_claims_by_slot() {
        let before = AnswerPlan::new(
            "介绍：张三",
            "张三",
            3,
            vec![
                PlanClaim::new("a#age", "age", "张三的年龄是18岁。", false),
                PlanClaim::new("r#师傅", "师傅", "张三的师傅是李四。", false),
                PlanClaim::new("r#徒弟", "徒弟", "张三的徒弟是王五。", false),
            ],
            0,
            0,
        );
        let after = AnswerPlan::new(
            "介绍：张三",
            "张三",
            4,
            vec![
                PlanClaim::new("a#age", "age", "张三的年龄是19岁。", false),
                PlanClaim::new("r#徒弟", "徒弟", "张三的徒弟是王五。", false),
                PlanClaim::new("a#出生地", "出生地", "张三的出生地是北京。", true),
            ],
            0,
            0,
        );
        let diff = before.diff(&after);
        assert_eq!(diff.counts(), (1, 1, 1, 1), "{:?}", diff);
        assert!(!diff.is_unchanged());
        assert_eq!(diff.base_revision, 3);
        assert_eq!(diff.current_revision, 4);
        let text = diff.render_zh(false);
        assert!(text.contains("[更新] a#age"), "{}", text);
        assert!(text.contains("[移除] r#师傅"), "{}", text);
        assert!(text.contains("[新增] a#出生地"), "{}", text);
        assert!(!text.contains("[未变]"), "默认只列变化项：{}", text);
        assert!(text.contains("不改写已经说出口的回答"), "{}", text);
    }

    #[test]
    fn diff_prefers_exact_matches_before_same_slot_updates() {
        let before = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "旧说法。", false)],
            0,
            0,
        );
        let after = AnswerPlan::new(
            "q",
            "张三",
            2,
            vec![
                PlanClaim::new("s", "p", "新说法。", false),
                PlanClaim::new("s", "p", "旧说法。", false),
            ],
            0,
            0,
        );
        let diff = before.diff(&after);
        assert_eq!(diff.counts(), (0, 0, 1, 1), "{:?}", diff);
    }

    #[test]
    fn diff_of_identical_content_across_revisions_is_still_explicit() {
        let first = sample_plan();
        let second = AnswerPlan::new(
            "介绍：张三",
            "张三",
            first.revision + 1,
            first.claims.clone(),
            0,
            0,
        );
        let diff = first.diff(&second);
        assert!(diff.is_unchanged());
        assert_eq!(diff.counts(), (0, 0, 0, first.claim_count()));
        let text = diff.render_zh(true);
        assert!(text.contains("计划内容未变"), "{}", text);
        assert!(text.contains("知识版本已更新"), "{}", text);
        assert!(text.contains("[未变]"), "{}", text);
    }

    #[test]
    fn diff_serializes_without_optional_claim_placeholders() {
        let before = sample_plan();
        let mut claims = before.claims.clone();
        claims.pop();
        let after = AnswerPlan::new("介绍：张三", "张三", 8, claims, 0, 0);
        let json = serde_json::to_string(&before.diff(&after)).unwrap();
        let restored: PlanDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.counts().1, 1);
    }

    #[test]
    fn evidence_is_optional_skipped_in_json_and_changes_identity() {
        let plain = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "命题。", false)],
            0,
            0,
        );
        let json = serde_json::to_string(&plain).unwrap();
        assert!(
            !json.contains("evidence"),
            "空证据不得增加序列化字节：{}",
            json
        );

        let annotated = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "命题。", false)
                .with_evidence(vec![ClaimEvidence::new("张三", "师傅", "李四", false)])],
            0,
            0,
        );
        assert_ne!(plain.id, annotated.id, "证据属于计划内容，必须进入指纹");
        let detailed = annotated.realize(StyleDirective::Detailed).unwrap();
        assert!(
            detailed.contains("依据：图谱 张三 师傅 李四"),
            "{}",
            detailed
        );
        let brief = annotated.realize(StyleDirective::Brief).unwrap();
        assert!(!brief.contains("依据："), "简短版不展开证据：{}", brief);

        let roundtrip: AnswerPlan =
            serde_json::from_str(&serde_json::to_string(&annotated).unwrap()).unwrap();
        assert_eq!(roundtrip, annotated);

        let old: AnswerPlan = serde_json::from_str(
            r#"{"version":1,"id":"x","question":"q","subject":"张三","revision":1,"claims":[{"slot":"s","predicate":"p","text":"命题。","session":false}],"omitted":0,"suppressed":0}"#,
        )
        .unwrap();
        assert!(old.claims[0].evidence.is_empty());
    }

    #[test]
    fn diff_surfaces_evidence_only_updates() {
        let before = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "命题。", false)
                .with_evidence(vec![ClaimEvidence::new("张三", "师傅", "李四", false)])],
            0,
            0,
        );
        let after = AnswerPlan::new(
            "q",
            "张三",
            2,
            vec![PlanClaim::new("s", "p", "命题。", false)
                .with_evidence(vec![ClaimEvidence::new("张三", "师傅", "王五", false)])],
            0,
            0,
        );
        let diff = before.diff(&after);
        assert_eq!(diff.counts().0, 1, "证据变化应算更新");
        let text = diff.render_zh(false);
        assert!(text.contains("依据："), "{}", text);
        assert!(text.contains("李四") && text.contains("王五"), "{}", text);
    }

    #[test]
    fn envelopes_are_optional_but_change_identity_and_detailed_rendering() {
        let plain = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "命题。", false)],
            0,
            0,
        );
        let annotated = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "p", "命题。", false)
                .with_envelope(ClaimEnvelope::new(6_300, 9_000))],
            0,
            0,
        );
        assert_ne!(plain.id, annotated.id, "区间属于计划内容，必须进入指纹");
        let text = annotated.realize(StyleDirective::Detailed).unwrap();
        assert!(text.contains("置信区间：63–90%"), "{}", text);
        let brief = annotated.realize(StyleDirective::Brief).unwrap();
        assert!(
            !brief.contains("置信区间"),
            "简短版不展开置信元数据：{}",
            brief
        );

        let json = serde_json::to_string(&annotated).unwrap();
        let restored: AnswerPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, annotated);
        let old: AnswerPlan = serde_json::from_str(
            r#"{"version":1,"id":"x","question":"q","subject":"张三","revision":1,"claims":[{"slot":"s","predicate":"p","text":"命题。","session":false}],"omitted":0,"suppressed":0}"#,
        )
        .unwrap();
        assert!(old.claims[0].envelope.is_none());
    }

    #[test]
    fn plan_survives_json_roundtrip_and_old_payloads_default_to_none() {
        let plan = sample_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let restored: AnswerPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, plan);
        assert!(restored.is_current_schema());

        #[derive(Deserialize)]
        struct Holder {
            #[serde(default)]
            plan: Option<AnswerPlan>,
        }
        let old: Holder = serde_json::from_str("{}").unwrap();
        assert!(old.plan.is_none());
    }

    #[test]
    fn surface_units_concatenate_to_exactly_the_rendered_text() {
        let plan = sample_plan();
        for directive in [
            StyleDirective::Brief,
            StyleDirective::Detailed,
            StyleDirective::Restate,
            StyleDirective::Focus(2),
            StyleDirective::Budget(90),
        ] {
            let stream = plan.surface_units(directive).unwrap();
            assert_eq!(
                stream.concat(),
                plan.realize(directive).unwrap(),
                "{} 必须由同一份表层单位拼接",
                directive.label()
            );
            assert_eq!(stream.len(), stream.units.len());
            for (position, unit) in stream.iter().enumerate() {
                assert_eq!(unit.index, position);
                assert_eq!(
                    stream.event_id(position),
                    Some(format!("{}:{}:{}", plan.id, directive.tag(), position))
                );
            }
        }
    }

    #[test]
    fn claim_units_map_to_slots_without_leaking_brief_metadata() {
        let plan = sample_plan();
        let brief = plan.surface_units(StyleDirective::Brief).unwrap();
        let claims: Vec<&SurfaceUnit> = brief
            .iter()
            .filter(|unit| unit.kind == SurfaceUnitKind::Claim)
            .collect();
        assert_eq!(claims.len(), BRIEF_CLAIM_LIMIT);
        for (position, unit) in claims.iter().enumerate() {
            assert_eq!(
                unit.slot.as_deref(),
                Some(plan.claims[position].slot.as_str())
            );
            assert_eq!(unit.claim_index, Some(position + 1));
            assert!(
                !unit.text.contains("谓词："),
                "简短版不展开元数据：{}",
                unit.text
            );
        }
        assert_eq!(
            brief.iter().next().map(|unit| unit.kind),
            Some(SurfaceUnitKind::Header)
        );
        assert_eq!(
            brief.iter().last().map(|unit| unit.kind),
            Some(SurfaceUnitKind::Omission)
        );
        assert!(brief.concat().contains("完整讲讲"), "{}", brief.concat());

        let detailed = plan.surface_units(StyleDirective::Detailed).unwrap();
        assert_eq!(
            detailed
                .iter()
                .filter(|unit| unit.kind == SurfaceUnitKind::Claim)
                .count(),
            plan.claim_count()
        );
        assert!(detailed.concat().contains("来源：图谱事实"));
        assert_eq!(
            detailed.iter().last().map(|unit| unit.kind),
            Some(SurfaceUnitKind::Followup)
        );
    }

    #[test]
    fn surface_units_preserve_focus_errors_and_serialize_with_event_ids() {
        let plan = sample_plan();
        assert_eq!(
            plan.surface_units(StyleDirective::Focus(9)).unwrap_err(),
            PlanRenderError::FocusOutOfRange {
                subject: plan.subject.clone(),
                requested: 9,
                total: plan.claim_count(),
            }
        );
        let empty = AnswerPlan::new("介绍：无人", "无人", 1, Vec::new(), 0, 0);
        assert!(matches!(
            empty.surface_units(StyleDirective::Brief),
            Err(PlanRenderError::Empty { .. })
        ));

        let brief = plan.surface_units(StyleDirective::Brief).unwrap();
        let detailed = plan.surface_units(StyleDirective::Detailed).unwrap();
        let json = serde_json::to_string(&brief).unwrap();
        let restored: SurfaceUnits = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, brief);
        assert_ne!(brief.event_id(0), detailed.event_id(0));
        assert_eq!(brief.event_id(brief.len()), None);
    }

    fn media_plan_citation(label: &str, confidence: f64) -> PlanMediaCitation {
        use crate::media_evidence::{AnchorDraft, Locator, MediaLimits, MediaStore, Modality};
        let mut store = MediaStore::new(MediaLimits::default());
        let bytes = format!("{}|{}", label, confidence);
        let asset = store
            .publish(Modality::Table, "text/csv", label, bytes.as_bytes())
            .unwrap();
        let anchor = store
            .add_anchor(
                &asset.id,
                AnchorDraft::new(
                    Locator::TableCell { row: 3, column: 2 },
                    "1981",
                    "table_parser",
                    confidence,
                ),
            )
            .unwrap();
        PlanMediaCitation::from(&anchor.citation(&asset))
    }

    #[test]
    fn media_citations_are_optional_serialized_and_change_identity() {
        let plain = AnswerPlan::new(
            "q",
            "公司",
            1,
            vec![PlanClaim::new("s", "p", "公司成立于1981年。", false)],
            0,
            0,
        );
        let json = serde_json::to_string(&plain).unwrap();
        assert!(
            !json.contains("media"),
            "空媒体引用不得增加序列化字节：{}",
            json
        );

        let citation = media_plan_citation("表 1", 0.92);
        let annotated = AnswerPlan::new(
            "q",
            "公司",
            1,
            vec![PlanClaim::new("s", "p", "公司成立于1981年。", false)
                .with_media_citations(vec![citation])],
            0,
            0,
        );
        assert_ne!(plain.id, annotated.id, "媒体引用属于计划内容，必须进入指纹");
        let json = serde_json::to_string(&annotated).unwrap();
        assert!(json.contains("media"));
        let detailed = annotated.realize(StyleDirective::Detailed).unwrap();
        assert!(detailed.contains("媒体：表 1"), "{}", detailed);
        assert!(detailed.contains("表格第 3 行第 2 列"), "{}", detailed);
        assert!(detailed.contains("table_parser"), "{}", detailed);
    }

    #[test]
    fn brief_shows_strongest_media_citation_and_diff_tracks_media_only_updates() {
        let weak = media_plan_citation("表 1", 0.40);
        let strong = media_plan_citation("表 2", 0.95);
        let claim = PlanClaim::new("s", "p", "公司成立于1981年。", false)
            .with_media_citations(vec![weak.clone(), strong.clone()]);
        let plan = AnswerPlan::new("q", "公司", 1, vec![claim], 0, 0);
        let brief = plan.realize(StyleDirective::Brief).unwrap();
        assert!(brief.contains("表 2"), "{}", brief);
        assert!(!brief.contains("表 1"), "{}", brief);
        let detailed = plan.realize(StyleDirective::Detailed).unwrap();
        assert!(
            detailed.contains("表 1") && detailed.contains("表 2"),
            "{}",
            detailed
        );

        let before = AnswerPlan::new(
            "q",
            "公司",
            1,
            vec![PlanClaim::new("s", "p", "公司成立于1981年。", false)
                .with_media_citations(vec![weak])],
            0,
            0,
        );
        let after = AnswerPlan::new(
            "q",
            "公司",
            1,
            vec![PlanClaim::new("s", "p", "公司成立于1981年。", false)
                .with_media_citations(vec![strong])],
            0,
            0,
        );
        let diff = before.diff(&after);
        assert_eq!(diff.counts().0, 1);
        let rendered = diff.render_zh(false);
        assert!(rendered.contains("媒体：表 1"), "{}", rendered);
        assert!(rendered.contains("表 2"), "{}", rendered);
    }

    #[test]
    fn justification_is_optional_serialized_and_visible_only_in_detailed_mode() {
        let plain = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "师傅", "张三的师傅是李四。", false)],
            0,
            0,
        );
        let plain_json = serde_json::to_string(&plain).unwrap();
        assert!(
            !plain_json.contains("justification"),
            "直接事实不得增加推导字段：{}",
            plain_json
        );

        let justification = ClaimJustification::new(
            "传递规则",
            vec![
                ClaimEvidence::new("张三", "师傅", "李四", false),
                ClaimEvidence::new("李四", "师傅", "王五", false),
            ],
            2,
            8_100,
        );
        let derived = AnswerPlan::new(
            "q",
            "张三",
            1,
            vec![PlanClaim::new("s", "师祖", "张三的师祖是王五。", false)
                .with_evidence(vec![
                    ClaimEvidence::new("张三", "师傅", "李四", false),
                    ClaimEvidence::new("李四", "师傅", "王五", false),
                ])
                .with_justification(justification)],
            0,
            0,
        );
        assert_ne!(plain.id, derived.id, "推导依据必须进入计划指纹");
        let json = serde_json::to_string(&derived).unwrap();
        assert!(json.contains("justification"), "{}", json);
        let detailed = derived.realize(StyleDirective::Detailed).unwrap();
        assert!(
            detailed.contains("推导：由「传递规则」推出"),
            "{}",
            detailed
        );
        assert!(detailed.contains("置信度 0.81"), "{}", detailed);
        assert!(detailed.contains("第 2 轮"), "{}", detailed);
        let brief = derived.realize(StyleDirective::Brief).unwrap();
        assert!(!brief.contains("推导："), "简短模式不应展开推导：{}", brief);
    }
}

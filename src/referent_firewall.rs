//! 指称防火墙（Referent Firewall）：链接之后、回答之前的多世界合并。
//!
//! 一次指称链接把 mention 绑定到候选实体集合 C。绑定不是单纯的准确率问题，
//! 而是一次读取授权：允许系统读取该实体的子图并以其身份作答。本模块把 C 中
//! 的每个实体看作一个候选世界，对每个世界生成回答，再按离散证据格合并：
//!
//! - 所有世界都成立的命题 → 不变回答，可直接回答；
//! - 只在部分世界成立的命题 → 条件式回答（分支数有上限）；
//! - 同一槽位在不同世界取值分裂、分支过多或候选集不完整 → 澄清，不直答；
//! - 任一候选世界的披露策略更严 → 全体按 meet（最严）执行；
//! - 候选集被截断或探测未完成 → 不得宣称不变，降级为澄清；
//! - 票差只用于排序与风险标注，不冒充排除证据。
//!
//! 输入是候选世界（实体、票数、授权依据、回答、披露等级），输出是确定性
//! 的防火墙报告与证书。排序、分组、渲染均不依赖挂钟、随机数或哈希遍历序；
//! 证书只保存计数、标签与指纹，不保存回答原文。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};

pub const REFERENT_FIREWALL_VERSION: u32 = 1;
pub const MAX_CONDITIONAL_BRANCHES: usize = 3;
pub const MAX_OPTION_PREVIEW_CHARS: usize = 40;

/// 候选仍然可能的原因（显式授权 / 硬约束筛选 / 仅票差排序）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Warrant {
    /// 用户显式选择或语境绑定已经签字。
    Authorized,
    /// 类型、谓词或实体后缀等硬约束筛掉了其它候选。
    Constrained,
    /// 只按票差排序，候选未被任何证据排除。
    Ranked,
    /// 候选集本身被截断，无法确认是否还有存活候选。
    Truncated,
}

impl Warrant {
    pub fn label(self) -> &'static str {
        match self {
            Warrant::Authorized => "显式授权",
            Warrant::Constrained => "硬约束筛选",
            Warrant::Ranked => "仅票差排序",
            Warrant::Truncated => "候选集截断",
        }
    }

    pub fn is_exclusion(self) -> bool {
        matches!(self, Warrant::Authorized | Warrant::Constrained)
    }
}

/// 一个候选世界里的命题。空槽位用于无法结构化回答的自由文本。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Claim {
    pub slot: String,
    pub value: String,
}

impl Claim {
    pub fn new(slot: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            value: value.into(),
        }
    }

    pub fn answer(value: impl Into<String>) -> Self {
        Self::new(String::new(), value)
    }
}

/// 候选世界对当前问题的回答状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldAnswer {
    Answered(Vec<Claim>),
    NoAnswer,
    Unresolved(String),
}

impl WorldAnswer {
    pub fn label(&self) -> &'static str {
        match self {
            WorldAnswer::Answered(_) => "answered",
            WorldAnswer::NoAnswer => "no_answer",
            WorldAnswer::Unresolved(_) => "unresolved",
        }
    }

    pub fn claims(&self) -> Vec<Claim> {
        match self {
            WorldAnswer::Answered(claims) => normalize_claims(claims),
            _ => Vec::new(),
        }
    }
}

/// 披露等级按 Public < Coarsened < Blocked 排序，meet 取最大值。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureLevel {
    Public,
    Coarsened,
    Blocked,
}

impl DisclosureLevel {
    pub fn label(self) -> &'static str {
        match self {
            DisclosureLevel::Public => "public",
            DisclosureLevel::Coarsened => "coarsened",
            DisclosureLevel::Blocked => "blocked",
        }
    }

    pub fn render_zh(self) -> &'static str {
        match self {
            DisclosureLevel::Public => "公开",
            DisclosureLevel::Coarsened => "降级",
            DisclosureLevel::Blocked => "阻断",
        }
    }
}

/// 单个候选世界的披露裁定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureGrade {
    pub level: DisclosureLevel,
    pub reason: String,
}

impl Default for DisclosureGrade {
    fn default() -> Self {
        Self::public()
    }
}

impl DisclosureGrade {
    pub fn public() -> Self {
        Self {
            level: DisclosureLevel::Public,
            reason: String::new(),
        }
    }

    pub fn coarsened(reason: impl Into<String>) -> Self {
        Self {
            level: DisclosureLevel::Coarsened,
            reason: reason.into(),
        }
    }

    pub fn blocked(reason: impl Into<String>) -> Self {
        Self {
            level: DisclosureLevel::Blocked,
            reason: reason.into(),
        }
    }

    pub fn meet(&self, other: &Self) -> Self {
        match self.level.cmp(&other.level) {
            std::cmp::Ordering::Greater => self.clone(),
            std::cmp::Ordering::Less => other.clone(),
            std::cmp::Ordering::Equal => {
                let mut reasons = [self.reason.clone(), other.reason.clone()];
                reasons.sort();
                Self {
                    level: self.level,
                    reason: reasons
                        .into_iter()
                        .find(|reason| !reason.is_empty())
                        .unwrap_or_default(),
                }
            }
        }
    }
}

/// 一个候选世界：实体视角下的回答与披露裁定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateWorld {
    pub entity: String,
    pub label: Option<String>,
    pub rank: usize,
    pub votes: i32,
    pub warrant: Warrant,
    pub answer: WorldAnswer,
    pub disclosure: DisclosureGrade,
}

impl CandidateWorld {
    pub fn new(
        entity: impl Into<String>,
        rank: usize,
        votes: i32,
        warrant: Warrant,
        answer: WorldAnswer,
    ) -> Self {
        Self {
            entity: entity.into(),
            label: None,
            rank,
            votes,
            warrant,
            answer,
            disclosure: DisclosureGrade::public(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_disclosure(mut self, disclosure: DisclosureGrade) -> Self {
        self.disclosure = disclosure;
        self
    }

    fn display_label(&self) -> String {
        self.label.clone().unwrap_or_else(|| self.entity.clone())
    }
}

/// 防火墙输入：一个指称的候选世界集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirewallRequest {
    pub mention: String,
    /// 候选集是否完整。截断的候选集不能支撑「不变回答」。
    pub candidate_set_complete: bool,
    pub worlds: Vec<CandidateWorld>,
}

impl FirewallRequest {
    pub fn new(mention: impl Into<String>, worlds: Vec<CandidateWorld>) -> Self {
        Self {
            mention: mention.into(),
            candidate_set_complete: true,
            worlds,
        }
    }

    pub fn with_completeness(mut self, complete: bool) -> Self {
        self.candidate_set_complete = complete;
        self
    }
}

/// 条件式回答的一个分支。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalBranch {
    pub entities: Vec<String>,
    pub claims: Vec<Claim>,
    pub no_answer: bool,
}

impl ConditionalBranch {
    pub fn render_zh(&self, mention: &str) -> String {
        let entities = self.entities.join(" / ");
        if self.no_answer {
            format!("若「{}」指 {}：当前知识库无答案", mention, entities)
        } else {
            format!(
                "若「{}」指 {}：{}",
                mention,
                entities,
                render_claims(&self.claims)
            )
        }
    }
}

/// 防火墙裁决。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FirewallVerdict {
    Invariant {
        claims: Vec<Claim>,
    },
    Conditional {
        shared: Vec<Claim>,
        branches: Vec<ConditionalBranch>,
    },
    Conflict {
        slot: String,
        values: Vec<String>,
    },
    NeedClarification {
        reason: String,
    },
    Uninformative,
    Blocked {
        reason: String,
    },
}

impl FirewallVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            FirewallVerdict::Invariant { .. } => "invariant",
            FirewallVerdict::Conditional { .. } => "conditional",
            FirewallVerdict::Conflict { .. } => "conflict",
            FirewallVerdict::NeedClarification { .. } => "need_clarification",
            FirewallVerdict::Uninformative => "uninformative",
            FirewallVerdict::Blocked { .. } => "blocked",
        }
    }

    pub fn allows_direct_answer(&self) -> bool {
        matches!(self, FirewallVerdict::Invariant { .. })
    }
}

/// 过策略后的澄清选项：阻断时为空，降级时不带答案预览。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirewallOption {
    pub entity: String,
    pub label: String,
    pub preview: Option<String>,
}

/// 授权依据计数（用于证书区分「排除到唯一」与「票高者胜」）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarrantSummary {
    pub authorized: usize,
    pub constrained: usize,
    pub ranked: usize,
    pub truncated: usize,
}

impl WarrantSummary {
    fn from_worlds(worlds: &[CandidateWorld]) -> Self {
        let mut summary = Self::default();
        for world in worlds {
            match world.warrant {
                Warrant::Authorized => summary.authorized += 1,
                Warrant::Constrained => summary.constrained += 1,
                Warrant::Ranked => summary.ranked += 1,
                Warrant::Truncated => summary.truncated += 1,
            }
        }
        summary
    }
}

/// 歧义证书：只保存计数与标签，不保存候选回答原文。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirewallCertificate {
    pub version: u32,
    pub mention: String,
    pub world_count: usize,
    pub distinct_answers: usize,
    pub complete: bool,
    pub basis: String,
    pub verdict: String,
    pub policy: String,
    pub warrants: WarrantSummary,
    pub certificate_digest: String,
}

impl FirewallCertificate {
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
            "指称歧义证书 v{}（摘要 {}）",
            self.version, self.certificate_digest
        );
        let _ = writeln!(
            out,
            "  指称「{}」：{} 个候选世界，{} 种回答，候选集{}",
            self.mention,
            self.world_count,
            self.distinct_answers,
            if self.complete { "完整" } else { "不完整" }
        );
        let _ = writeln!(
            out,
            "  裁决：{}；依据：{}；披露：{}",
            self.verdict, self.basis, self.policy
        );
        let _ = writeln!(
            out,
            "  授权：显式 {} / 硬约束 {} / 仅票差 {} / 截断 {}",
            self.warrants.authorized,
            self.warrants.constrained,
            self.warrants.ranked,
            self.warrants.truncated
        );
        out.trim_end().to_string()
    }
}

/// 防火墙报告。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirewallReport {
    pub mention: String,
    pub verdict: FirewallVerdict,
    pub world_count: usize,
    pub distinct_answers: usize,
    pub complete: bool,
    pub basis: String,
    pub disclosure: DisclosureGrade,
    pub warrants: WarrantSummary,
    pub options: Vec<FirewallOption>,
    pub notes: Vec<String>,
    pub certificate: FirewallCertificate,
}

impl FirewallReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "指称防火墙「{}」：{} 个候选世界，裁决 {}（依据 {}，披露 {}）",
            self.mention,
            self.world_count,
            self.verdict.label(),
            self.basis,
            self.disclosure.level.label()
        );
        let public = self.disclosure.level == DisclosureLevel::Public;
        match &self.verdict {
            FirewallVerdict::Invariant { claims } => {
                if public {
                    let _ = writeln!(out, "  不变回答：{}", render_claims(claims));
                } else {
                    let _ = writeln!(out, "  不变回答：已按披露策略隐藏");
                }
            }
            FirewallVerdict::Conditional { shared, branches } => {
                if !public {
                    let _ = writeln!(out, "  条件式回答已按披露策略隐藏");
                } else {
                    if !shared.is_empty() {
                        let _ = writeln!(out, "  所有世界共有：{}", render_claims(shared));
                    }
                    for branch in branches {
                        let _ = writeln!(out, "  {}", branch.render_zh(&self.mention));
                    }
                }
            }
            FirewallVerdict::Conflict { slot, values } => {
                let _ = writeln!(
                    out,
                    "  槽位「{}」取值冲突：{}",
                    if slot.is_empty() { "回答" } else { slot },
                    if public {
                        values.join(" / ")
                    } else {
                        "已按披露策略隐藏".to_string()
                    }
                );
            }
            FirewallVerdict::NeedClarification { reason } | FirewallVerdict::Blocked { reason } => {
                let _ = writeln!(out, "  {}", reason);
            }
            FirewallVerdict::Uninformative => {
                let _ = writeln!(out, "  所有候选均无答案，不宣称事实不存在");
            }
        }
        for note in &self.notes {
            let _ = writeln!(out, "  - {}", note);
        }
        out.trim_end().to_string()
    }

    /// 条件式回答文本；仅在条件式裁决下返回。
    pub fn render_conditional_answer(&self) -> Option<String> {
        let FirewallVerdict::Conditional { shared, branches } = &self.verdict else {
            return None;
        };
        if self.disclosure.level != DisclosureLevel::Public {
            return None;
        }
        let mut out = String::new();
        if !shared.is_empty() {
            let _ = writeln!(out, "所有候选解释都成立：{}", render_claims(shared));
        }
        for branch in branches {
            let _ = writeln!(out, "{}", branch.render_zh(&self.mention));
        }
        out.push_str("（指称尚未唯一确定，以上为条件式回答。）");
        Some(out)
    }

    pub fn certificate(&self) -> &FirewallCertificate {
        &self.certificate
    }
}

/// 对候选世界执行防火墙合并。
pub fn evaluate(request: &FirewallRequest) -> FirewallReport {
    let mut worlds = request.worlds.clone();
    worlds.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then_with(|| a.entity.cmp(&b.entity))
            .then_with(|| a.votes.cmp(&b.votes))
    });

    let disclosure = worlds
        .iter()
        .map(|world| world.disclosure.clone())
        .fold(DisclosureGrade::public(), |meet, grade| meet.meet(&grade));
    let warrants = WarrantSummary::from_worlds(&worlds);
    let world_count = worlds.len();

    let unresolved_reason = worlds.iter().find_map(|world| match &world.answer {
        WorldAnswer::Unresolved(reason) => {
            Some(format!("候选「{}」探测未完成：{}", world.entity, reason))
        }
        _ => None,
    });

    let mut groups: BTreeMap<Vec<Claim>, Vec<String>> = BTreeMap::new();
    let mut no_answer_entities: Vec<String> = Vec::new();
    let mut claimed_worlds: Vec<&CandidateWorld> = Vec::new();
    for world in &worlds {
        match &world.answer {
            WorldAnswer::Answered(claims) => {
                claimed_worlds.push(world);
                groups
                    .entry(normalize_claims(claims))
                    .or_default()
                    .push(world.entity.clone());
            }
            WorldAnswer::NoAnswer => no_answer_entities.push(world.entity.clone()),
            WorldAnswer::Unresolved(_) => {}
        }
    }
    for entities in groups.values_mut() {
        entities.sort();
        entities.dedup();
    }
    no_answer_entities.sort();
    no_answer_entities.dedup();

    let has_no_answer = !no_answer_entities.is_empty();
    let distinct_answers = groups.len() + usize::from(has_no_answer);
    let complete =
        request.candidate_set_complete && unresolved_reason.is_none() && warrants.truncated == 0;
    let basis = basis_label(&worlds, complete, unresolved_reason.is_some());

    let conflict = detect_conflict(&claimed_worlds);
    let verdict = if disclosure.level == DisclosureLevel::Blocked {
        FirewallVerdict::Blocked {
            reason: format!(
                "候选世界披露策略按最严者执行：{}",
                if disclosure.reason.is_empty() {
                    "存在阻断候选"
                } else {
                    disclosure.reason.as_str()
                }
            ),
        }
    } else if let Some(reason) = unresolved_reason {
        FirewallVerdict::NeedClarification { reason }
    } else if !request.candidate_set_complete || warrants.truncated > 0 {
        FirewallVerdict::NeedClarification {
            reason: "候选集被截断，不能宣称不变回答".to_string(),
        }
    } else if claimed_worlds.is_empty() && no_answer_entities.is_empty() {
        FirewallVerdict::NeedClarification {
            reason: "没有可合并的候选世界".to_string(),
        }
    } else if claimed_worlds.is_empty() {
        FirewallVerdict::Uninformative
    } else if let Some((slot, values)) = conflict {
        FirewallVerdict::Conflict { slot, values }
    } else if distinct_answers > MAX_CONDITIONAL_BRANCHES {
        FirewallVerdict::NeedClarification {
            reason: format!(
                "候选回答分裂为 {} 个分支，超过条件式回答上限 {}",
                distinct_answers, MAX_CONDITIONAL_BRANCHES
            ),
        }
    } else if groups.len() == 1 && !has_no_answer {
        FirewallVerdict::Invariant {
            claims: groups.keys().next().cloned().unwrap_or_default(),
        }
    } else {
        FirewallVerdict::Conditional {
            shared: shared_claims(&groups, has_no_answer),
            branches: branches(&groups, &no_answer_entities),
        }
    };

    let options = build_options(&worlds, &verdict, &disclosure);
    let mut notes = firewall_notes(&verdict, &disclosure, &warrants, complete);
    notes.sort();
    notes.dedup();

    let certificate = certificate_for(
        request,
        &verdict,
        &disclosure,
        &warrants,
        distinct_answers,
        complete,
        &basis,
    );
    FirewallReport {
        mention: request.mention.clone(),
        verdict,
        world_count,
        distinct_answers,
        complete,
        basis,
        disclosure,
        warrants,
        options,
        notes,
        certificate,
    }
}

fn normalize_claims(claims: &[Claim]) -> Vec<Claim> {
    let mut normalized = claims.to_vec();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn detect_conflict(worlds: &[&CandidateWorld]) -> Option<(String, Vec<String>)> {
    let mut slots: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for world in worlds {
        if let WorldAnswer::Answered(claims) = &world.answer {
            for claim in claims {
                slots
                    .entry(claim.slot.clone())
                    .or_default()
                    .insert(claim.value.clone());
            }
        }
    }
    for (slot, values) in slots {
        if values.len() > 1 {
            return Some((slot, values.into_iter().collect()));
        }
    }
    None
}

fn shared_claims(groups: &BTreeMap<Vec<Claim>, Vec<String>>, has_no_answer: bool) -> Vec<Claim> {
    if has_no_answer || groups.len() < 2 {
        return Vec::new();
    }
    let mut keys = groups.keys();
    let Some(first) = keys.next() else {
        return Vec::new();
    };
    let mut shared: BTreeSet<Claim> = first.iter().cloned().collect();
    for key in keys {
        let claims: BTreeSet<Claim> = key.iter().cloned().collect();
        shared = shared.intersection(&claims).cloned().collect();
    }
    shared.into_iter().collect()
}

fn branches(
    groups: &BTreeMap<Vec<Claim>, Vec<String>>,
    no_answer_entities: &[String],
) -> Vec<ConditionalBranch> {
    let mut branches = Vec::new();
    for (claims, entities) in groups {
        branches.push(ConditionalBranch {
            entities: entities.clone(),
            claims: claims.clone(),
            no_answer: false,
        });
    }
    if !no_answer_entities.is_empty() {
        branches.push(ConditionalBranch {
            entities: no_answer_entities.to_vec(),
            claims: Vec::new(),
            no_answer: true,
        });
    }
    branches.sort_by(|a, b| {
        a.no_answer
            .cmp(&b.no_answer)
            .then_with(|| a.entities.cmp(&b.entities))
            .then_with(|| a.claims.cmp(&b.claims))
    });
    branches
}

fn basis_label(worlds: &[CandidateWorld], complete: bool, unresolved: bool) -> String {
    if !complete || unresolved || worlds.iter().any(|w| w.warrant == Warrant::Truncated) {
        return "incomplete".to_string();
    }
    if worlds.len() == 1 && worlds[0].warrant.is_exclusion() {
        return if worlds[0].warrant == Warrant::Authorized {
            "authorized".to_string()
        } else {
            "excluded".to_string()
        };
    }
    if !worlds.is_empty() && worlds.iter().all(|w| w.warrant == Warrant::Authorized) {
        return "authorized".to_string();
    }
    if worlds.iter().any(|w| w.warrant == Warrant::Ranked) {
        return "ranked_only".to_string();
    }
    "constrained".to_string()
}

fn build_options(
    worlds: &[CandidateWorld],
    verdict: &FirewallVerdict,
    disclosure: &DisclosureGrade,
) -> Vec<FirewallOption> {
    if disclosure.level == DisclosureLevel::Blocked {
        return Vec::new();
    }
    let previews_allowed =
        disclosure.level == DisclosureLevel::Public && !verdict.allows_direct_answer();
    let mut options: Vec<FirewallOption> = worlds
        .iter()
        .map(|world| FirewallOption {
            entity: world.entity.clone(),
            label: world.display_label(),
            preview: if previews_allowed {
                match &world.answer {
                    WorldAnswer::Answered(claims) => {
                        let claims = normalize_claims(claims);
                        (!claims.is_empty()).then(|| preview_claims(&claims))
                    }
                    WorldAnswer::NoAnswer => Some("（当前知识库无答案）".to_string()),
                    WorldAnswer::Unresolved(_) => Some("（探测未完成）".to_string()),
                }
            } else {
                None
            },
        })
        .collect();
    options.sort_by(|a, b| a.entity.cmp(&b.entity));
    options
}

fn firewall_notes(
    verdict: &FirewallVerdict,
    disclosure: &DisclosureGrade,
    warrants: &WarrantSummary,
    complete: bool,
) -> Vec<String> {
    let mut notes = Vec::new();
    if disclosure.level == DisclosureLevel::Blocked {
        notes.push("披露策略按最严候选执行：本轮拒绝枚举候选与回答".to_string());
    } else if disclosure.level == DisclosureLevel::Coarsened {
        notes.push("披露策略按最严候选降级：澄清选项不展示答案预览".to_string());
    }
    if !complete {
        notes.push("候选集不完整：无法排除未探测候选，不宣称不变回答".to_string());
    }
    match verdict {
        FirewallVerdict::Invariant { .. } => {
            notes.push("所有候选世界给出同一答案，可直接回答".to_string());
        }
        FirewallVerdict::Conditional { .. } => {
            notes.push("答案只在部分候选世界成立，改用条件式回答".to_string());
        }
        FirewallVerdict::Conflict { slot, .. } => {
            notes.push(format!(
                "槽位「{}」在候选世界间取值不同，不能直接回答",
                if slot.is_empty() { "回答" } else { slot }
            ));
        }
        FirewallVerdict::NeedClarification { .. } => {
            notes.push("探测不足以收窄指称，保留澄清".to_string());
        }
        FirewallVerdict::Uninformative => {
            notes.push("所有候选均无答案，不把「无答案」说成「不存在」".to_string());
        }
        FirewallVerdict::Blocked { .. } => {}
    }
    if warrants.ranked > 0 && !verdict.allows_direct_answer() {
        notes.push("候选中仍有仅靠票差排序者：票差不是排除证据".to_string());
    }
    notes
}

fn certificate_for(
    request: &FirewallRequest,
    verdict: &FirewallVerdict,
    disclosure: &DisclosureGrade,
    warrants: &WarrantSummary,
    distinct_answers: usize,
    complete: bool,
    basis: &str,
) -> FirewallCertificate {
    let mut certificate = FirewallCertificate {
        version: REFERENT_FIREWALL_VERSION,
        mention: request.mention.clone(),
        world_count: request.worlds.len(),
        distinct_answers,
        complete,
        basis: basis.to_string(),
        verdict: verdict.label().to_string(),
        policy: disclosure.level.label().to_string(),
        warrants: warrants.clone(),
        certificate_digest: String::new(),
    };
    certificate.certificate_digest = certificate.compute_digest();
    certificate
}

fn render_claims(claims: &[Claim]) -> String {
    let mut parts: Vec<String> = claims
        .iter()
        .map(|claim| {
            if claim.slot.is_empty() {
                claim.value.clone()
            } else {
                format!("{}={}", claim.slot, claim.value)
            }
        })
        .collect();
    parts.sort();
    parts.dedup();
    parts.join("；")
}

fn preview_claims(claims: &[Claim]) -> String {
    let joined = render_claims(claims);
    let mut chars = joined.chars();
    let preview: String = chars.by_ref().take(MAX_OPTION_PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        format!("{}…", preview)
    } else {
        preview
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(entity: &str, rank: usize, answer: WorldAnswer) -> CandidateWorld {
        CandidateWorld::new(entity, rank, 10 - rank as i32, Warrant::Ranked, answer)
    }

    fn claims(slot: &str, value: &str) -> WorldAnswer {
        WorldAnswer::Answered(vec![Claim::new(slot, value)])
    }

    #[test]
    fn invariant_when_all_worlds_agree() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("颜色", "红色")),
                world("苹果公司", 2, claims("颜色", "红色")),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::Invariant { ref claims }
                if claims == &vec![Claim::new("颜色", "红色")]
        ));
        assert!(report.verdict.allows_direct_answer());
        assert_eq!(report.basis, "ranked_only");
        assert!(report.certificate.verify());
        assert!(report.render_text().contains("不变回答"));
    }

    #[test]
    fn narrowing_becomes_conditional_not_silent_bind() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("颜色", "红色")),
                world("苹果公司", 2, WorldAnswer::NoAnswer),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::Conditional { ref branches, .. } if branches.len() == 2
        ));
        assert!(!report.verdict.allows_direct_answer());
        let text = report
            .render_conditional_answer()
            .expect("conditional answer");
        assert!(text.contains("若「苹果」指 苹果"));
        assert!(text.contains("当前知识库无答案"));
        assert_eq!(report.distinct_answers, 2);
    }

    #[test]
    fn divergent_slot_is_conflict_and_asks() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("创始人", "乔布斯")),
                world("苹果公司", 2, claims("创始人", "库克")),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::Conflict { ref slot, ref values }
                if slot == "创始人" && values == &vec!["乔布斯".to_string(), "库克".to_string()]
        ));
        assert!(!report.verdict.allows_direct_answer());
        assert!(report.render_conditional_answer().is_none());
        assert!(report.options.iter().all(|option| option.preview.is_some()));
    }

    #[test]
    fn truncated_candidate_set_never_claims_invariant() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("颜色", "红色")),
                world("苹果公司", 2, claims("颜色", "红色")),
            ],
        )
        .with_completeness(false);
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::NeedClarification { .. }
        ));
        assert!(!report.complete);
        assert_eq!(report.basis, "incomplete");
        assert!(report.render_conditional_answer().is_none());
    }

    #[test]
    fn unresolved_world_never_claims_invariant() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("颜色", "红色")),
                world(
                    "苹果公司",
                    2,
                    WorldAnswer::Unresolved("查询超时".to_string()),
                ),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::NeedClarification { ref reason } if reason.contains("超时")
        ));
    }

    #[test]
    fn policy_meet_blocks_every_option_and_answer() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("手机号", "13800000000")),
                world("苹果公司", 2, claims("手机号", "13900000000"))
                    .with_disclosure(DisclosureGrade::blocked("个人联系方式")),
            ],
        );
        let report = evaluate(&request);
        assert_eq!(report.disclosure.level, DisclosureLevel::Blocked);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::Blocked { ref reason } if reason.contains("个人联系方式")
        ));
        assert!(report.options.is_empty());
        assert!(report.render_conditional_answer().is_none());
        assert!(!report.render_text().contains("13800000000"));
        assert!(!report.render_text().contains("13900000000"));
    }

    #[test]
    fn coarsened_policy_hides_previews_but_keeps_options() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, claims("颜色", "红色")),
                world("苹果公司", 2, claims("颜色", "库克"))
                    .with_disclosure(DisclosureGrade::coarsened("敏感属性")),
            ],
        );
        let report = evaluate(&request);
        assert_eq!(report.disclosure.level, DisclosureLevel::Coarsened);
        assert!(!report.options.is_empty());
        assert!(report.options.iter().all(|option| option.preview.is_none()));
        assert!(!report.render_text().contains("红色"));
        assert!(!report.render_text().contains("库克"));
    }

    #[test]
    fn explicit_binding_is_recorded_as_authorized_basis() {
        let request = FirewallRequest::new(
            "苹果",
            vec![CandidateWorld::new(
                "苹果公司",
                1,
                3,
                Warrant::Authorized,
                claims("创始人", "乔布斯"),
            )],
        );
        let report = evaluate(&request);
        assert_eq!(report.basis, "authorized");
        assert!(matches!(report.verdict, FirewallVerdict::Invariant { .. }));
        assert!(report.certificate.verify());
        assert_eq!(report.warrants.authorized, 1);
    }

    #[test]
    fn branch_budget_forces_clarification() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("甲", 1, claims("产地", "山东")),
                world("乙", 2, claims("创始地", "陕西")),
                world("丙", 3, claims("总部", "新疆")),
                world("丁", 4, claims("注册地", "甘肃")),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(
            report.verdict,
            FirewallVerdict::NeedClarification { ref reason } if reason.contains("上限")
        ));
    }

    #[test]
    fn certificate_is_deterministic_and_contains_no_answer_text() {
        let build = || {
            FirewallRequest::new(
                "苹果",
                vec![
                    world("苹果", 1, claims("颜色", "红色多汁")),
                    world("苹果公司", 2, WorldAnswer::NoAnswer),
                ],
            )
        };
        let first = evaluate(&build());
        let second = evaluate(&build());
        assert_eq!(first.certificate, second.certificate);
        assert!(first.certificate.verify());
        let text = first.certificate.canonical_text();
        assert!(!text.contains("红色多汁"));
        assert_eq!(first.certificate.verdict, "conditional");
        assert_eq!(first.certificate.policy, "public");
    }

    #[test]
    fn no_answer_everywhere_is_uninformative_not_negative_fact() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果", 1, WorldAnswer::NoAnswer),
                world("苹果公司", 2, WorldAnswer::NoAnswer),
            ],
        );
        let report = evaluate(&request);
        assert!(matches!(report.verdict, FirewallVerdict::Uninformative));
        assert!(!report.verdict.allows_direct_answer());
        assert!(report.render_text().contains("不宣称事实不存在"));
    }

    #[test]
    fn branch_rendering_is_stable() {
        let request = FirewallRequest::new(
            "苹果",
            vec![
                world("苹果公司", 2, claims("总部", "美国")),
                world("苹果", 1, claims("产地", "山东")),
            ],
        );
        let report = evaluate(&request);
        let first = report.render_text();
        let second = evaluate(&request).render_text();
        assert_eq!(first, second);
        assert!(!first.contains("苹果公司：苹果公司"));
    }
}

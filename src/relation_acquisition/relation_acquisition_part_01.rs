use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::predicate_registry::{PredicateProfile, PredicateRegistry};
use crate::prolog_engine::PrologFact;
use crate::shape_gate::{
    CandidateStatus, Claim, ClaimObject, InterpretationCandidate, RepairAction, ShapeSet,
};
use crate::type_system::TypeSystem;
pub const ACQUISITION_DIALECT: &str = "knowprolog-qra/1";
/// 同一取值在至少两个独立轮次出现，或收到一次显式确认，才提升为已确认。
pub const CONFIRMATION_SUPPORT: usize = 2;
pub const MAX_CLAIMS_PER_SLOT: usize = 12;
pub const MAX_MATCHED_PATTERNS: usize = 8;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionSource {
    UserAnswer,
    UserConfirmation,
    UserCorrection,
    UserAssertion,
    Graph,
    QuestionPresupposition,
    InducedPattern,
}
impl AcquisitionSource {
    pub fn label(self) -> &'static str {
        match self {
            AcquisitionSource::UserAnswer => "用户回答",
            AcquisitionSource::UserConfirmation => "用户确认",
            AcquisitionSource::UserCorrection => "用户纠错",
            AcquisitionSource::UserAssertion => "用户陈述",
            AcquisitionSource::Graph => "知识图谱",
            AcquisitionSource::QuestionPresupposition => "问句预设",
            AcquisitionSource::InducedPattern => "对话诱导模式",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionStatus {
    Proposed,
    Confirmed,
    Conflicted,
    Refuted,
    Admitted,
    Rejected,
}
impl AcquisitionStatus {
    pub fn label(self) -> &'static str {
        match self {
            AcquisitionStatus::Proposed => "候选",
            AcquisitionStatus::Confirmed => "已确认",
            AcquisitionStatus::Conflicted => "冲突",
            AcquisitionStatus::Refuted => "已否证",
            AcquisitionStatus::Admitted => "已采纳",
            AcquisitionStatus::Rejected => "已拒绝",
        }
    }

    pub fn is_live(self) -> bool {
        matches!(
            self,
            AcquisitionStatus::Proposed
                | AcquisitionStatus::Confirmed
                | AcquisitionStatus::Conflicted
                | AcquisitionStatus::Admitted
        )
    }

    pub fn is_committed(self) -> bool {
        matches!(
            self,
            AcquisitionStatus::Confirmed | AcquisitionStatus::Admitted
        )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerAct {
    Provide,
    Confirm,
    Deny,
    Correct,
    Unknown,
    Refuse,
    Clarify,
}
impl AnswerAct {
    pub fn label(self) -> &'static str {
        match self {
            AnswerAct::Provide => "提供取值",
            AnswerAct::Confirm => "确认",
            AnswerAct::Deny => "否认",
            AnswerAct::Correct => "纠错",
            AnswerAct::Unknown => "不知道",
            AnswerAct::Refuse => "拒绝回答",
            AnswerAct::Clarify => "反问/澄清",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalOrigin {
    MissingAnswer,
    UserRequest,
    ShapeRepair,
    Conflict,
}
impl GoalOrigin {
    pub fn label(self) -> &'static str {
        match self {
            GoalOrigin::MissingAnswer => "回答缺口",
            GoalOrigin::UserRequest => "用户请求",
            GoalOrigin::ShapeRepair => "形状修复",
            GoalOrigin::Conflict => "冲突裁定",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionForm {
    Who,
    What,
    Where,
    Value,
    Plain,
}
impl QuestionForm {
    pub fn label(self) -> &'static str {
        match self {
            QuestionForm::Who => "谁",
            QuestionForm::What => "什么",
            QuestionForm::Where => "哪里",
            QuestionForm::Value => "多少",
            QuestionForm::Plain => "陈述",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueShape {
    Entity,
    Literal,
    Infer,
}
impl ValueShape {
    pub fn label(self) -> &'static str {
        match self {
            ValueShape::Entity => "实体",
            ValueShape::Literal => "字面量",
            ValueShape::Infer => "待定",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expectation {
    /// 回答槽允许的实体类型（已按类型体系归一）。
    #[serde(default)]
    pub entity_types: Vec<String>,
    /// 字面量属性的数据类型（integer / date / ...）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    #[serde(default)]
    pub functional: bool,
    #[serde(default)]
    pub inverse: bool,
    #[serde(default)]
    pub required: bool,
    pub value_shape: ValueShape,
}
impl Expectation {
    pub fn render_text(&self) -> String {
        let mut parts = Vec::new();
        if !self.entity_types.is_empty() {
            parts.push(format!("值域 {}", self.entity_types.join(" / ")));
        }
        if let Some(datatype) = &self.datatype {
            parts.push(format!("类型 {}", datatype));
        }
        if self.functional {
            parts.push("单值槽".to_string());
        }
        if self.inverse {
            parts.push("互逆方向".to_string());
        }
        if parts.is_empty() {
            parts.push(format!("取值{}", self.value_shape.label()));
        }
        parts.join("；")
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationGoal {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    /// 用户实际使用的谓词词面（可能是别名或互逆方向）。
    pub surface: String,
    pub expectation: Expectation,
    pub origin: GoalOrigin,
    pub form: QuestionForm,
    pub question: String,
    pub turn: usize,
    pub asked: usize,
}
impl RelationGoal {
    pub fn render_text(&self) -> String {
        format!(
            "{} -[{}]-> ?  〔目标 {}；{}；来源 {}；已问 {} 次〕",
            self.subject,
            self.surface,
            self.id,
            self.expectation.render_text(),
            self.origin.label(),
            self.asked
        )
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquiredClaim {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_id: Option<String>,
    pub claim: Claim,
    pub status: AcquisitionStatus,
    pub source: AcquisitionSource,
    pub turn: usize,
    pub raw_span: String,
    pub rule: String,
    pub support: usize,
    pub confirmations: usize,
    #[serde(default)]
    pub conflicts: BTreeSet<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}
impl AcquiredClaim {
    pub fn render_text(&self) -> String {
        let mut text = format!(
            "- {} [{}] {}-[{}]->{} · 轮次 {} · 来源 {} · 规则 {} · 支持 {} / 确认 {}",
            self.id,
            self.status.label(),
            self.claim.subject,
            self.claim.predicate,
            self.claim.object.text(),
            self.turn,
            self.source.label(),
            self.rule,
            self.support,
            self.confirmations
        );
        if let Some(goal) = &self.goal_id {
            text.push_str(&format!(" · 问题 {}", goal));
        }
        if !self.conflicts.is_empty() {
            text.push_str(&format!(
                " · 冲突值 {}",
                self.conflicts
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("、")
            ));
        }
        if !self.notes.is_empty() {
            text.push_str(&format!(" · {}", self.notes.join("；")));
        }
        text
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueInducedPattern {
    pub id: String,
    pub predicate: String,
    pub surface: String,
    pub form: QuestionForm,
    pub question_template: String,
    pub statement_template: String,
    pub support: usize,
    pub active: bool,
    pub retired: bool,
    #[serde(default)]
    pub source_claims: BTreeSet<String>,
    #[serde(default)]
    pub examples: BTreeSet<String>,
}
impl DialogueInducedPattern {
    pub fn render_text(&self) -> String {
        let state = if self.retired {
            "已退役"
        } else if self.active {
            "已激活"
        } else {
            "待激活"
        };
        format!(
            "- {} [{}] 谓词「{}」（{}）· 支持 {} · 问句 {} · 陈述 {}",
            self.id,
            state,
            self.predicate,
            self.form.label(),
            self.support,
            self.question_template,
            self.statement_template
        )
    }

    /// 提升为统一模式库条目（来源 `induced#<id>`）：陈述模板 → `{{主体}}/{{客体}}` 洞。
    ///
    /// `ir` 是教学语法串（见 [`crate::pattern_ir::parse_teach_pattern`]），
    /// 由调用方按谓词词面生成，保持模式语言统一。
    pub fn to_pattern_entry(&self, ir: &str) -> crate::pattern_ir::PatternEntry {
        crate::pattern_ir::PatternEntry::new(
            format!("诱导-{}", self.surface),
            format!("induced#{}", self.id),
            self.predicate.clone(),
            8,
            true,
            crate::pattern_ir::parse_teach_pattern(ir)
                .unwrap_or_else(|_| vec![crate::pattern_ir::Elem::Any]),
        )
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionOverlayRow {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub claim_id: String,
    pub status: AcquisitionStatus,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionOverlay {
    #[serde(default)]
    pub rows: Vec<SessionOverlayRow>,
}
impl SessionOverlay {
    pub fn values(&self, subject: &str, predicate: &str) -> Vec<&SessionOverlayRow> {
        self.rows
            .iter()
            .filter(|row| row.subject == subject && row.predicate == predicate)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "会话覆盖层（仅本会话，未写入知识图谱）：{} 条",
            self.rows.len()
        )];
        if self.rows.is_empty() {
            lines.push("  （空；先输入「关系获取：S 的 P」打开一个槽位）".to_string());
            return lines.join("\n");
        }
        for row in &self.rows {
            lines.push(format!(
                "  {} 的 {} = {}  〔{}，来源 {}〕",
                row.subject,
                row.predicate,
                row.object,
                row.status.label(),
                row.claim_id
            ));
        }
        lines.join("\n")
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQuestion {
    pub subject: String,
    pub predicate: String,
    pub form: QuestionForm,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalOpen {
    Opened {
        goal: RelationGoal,
    },
    AlreadyKnown {
        subject: String,
        predicate: String,
        values: Vec<String>,
        source: String,
    },
    UnknownPredicate {
        predicate: String,
    },
    NoRelation {
        reason: String,
    },
}
impl GoalOpen {
    pub fn render_text(&self) -> String {
        match self {
            GoalOpen::Opened { goal } => format!(
                "{}\n（{}）",
                goal.question,
                goal.render_text()
            ),
            GoalOpen::AlreadyKnown {
                subject,
                predicate,
                values,
                source,
            } => format!(
                "「{}」的「{}」已有取值：{}（来源：{}；无需再追问）。",
                subject,
                predicate,
                values.join("、"),
                source
            ),
            GoalOpen::UnknownPredicate { predicate } => format!(
                "谓词「{}」尚未登记；请先用「谓词报告」查看，或用「登记别名：{} = 规范谓词」建立画像。",
                predicate, predicate
            ),
            GoalOpen::NoRelation { reason } => format!("无法建立关系获取目标：{}", reason),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitProposal {
    pub claim_id: String,
    pub sentence: String,
    pub status: AcquisitionStatus,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquisitionOutcome {
    pub act: AnswerAct,
    pub status: Option<AcquisitionStatus>,
    pub claim_id: Option<String>,
    pub message: String,
    pub commit: Option<CommitProposal>,
    pub follow_up: Option<String>,
}
impl AcquisitionOutcome {
    fn message(act: AnswerAct, message: impl Into<String>) -> Self {
        Self {
            act,
            status: None,
            claim_id: None,
            message: message.into(),
            commit: None,
            follow_up: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayAnswer {
    pub subject: String,
    pub predicate: String,
    pub values: Vec<String>,
    pub claim_ids: Vec<String>,
}
impl OverlayAnswer {
    pub fn render_text(&self) -> String {
        format!(
            "【会话覆盖层】根据你在本会话中告诉我的（尚未写入知识图谱）：\n  {} 的 {} = {}\n  出处：{}",
            self.subject,
            self.predicate,
            self.values.join("、"),
            self.claim_ids.join("、")
        )
    }
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatternScan {
    pub claims: Vec<Claim>,
    pub patterns: BTreeSet<String>,
}
impl PatternScan {
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn render_text(&self) -> String {
        if self.claims.is_empty() {
            return "模式扫描：没有命中的候选关系（模式需先被两次确认激活）。".to_string();
        }
        let mut lines = vec![format!(
            "模式扫描：命中 {} 条候选关系，使用 {} 个诱导模式。",
            self.claims.len(),
            self.patterns.len()
        )];
        for claim in &self.claims {
            lines.push(format!(
                "  {}-[{}]->{}",
                claim.subject,
                claim.predicate,
                claim.object.text()
            ));
        }
        lines.join("\n")
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionSnapshot {
    pub dialect: String,
    pub turn: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_claim: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_goal: Option<RelationGoal>,
    #[serde(default)]
    pub claims: BTreeMap<String, AcquiredClaim>,
    #[serde(default)]
    pub patterns: BTreeMap<String, DialogueInducedPattern>,
}
/// 会话级关系获取账本与覆盖层。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelationAcquisition {
    turn: usize,
    last_claim: Option<String>,
    active_goal: Option<RelationGoal>,
    claims: BTreeMap<String, AcquiredClaim>,
    patterns: BTreeMap<String, DialogueInducedPattern>,
}

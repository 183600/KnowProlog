//! QRA：Question-driven Relation Acquisition（问答式关系获取 / 以问代抽）。
//!
//! 目标：把「系统答不出」或「形状校验发现缺口」变成一个显式的槽位问题；
//! 用户回答先经过对话行为分类与形状闸门，再进入可审计的会话账本；
//! 只有被确认的候选才进入会话覆盖层，并可显式采纳到会话事实集。
//!
//! 与既有模块的分工：
//! - `predicate_registry` / `type_system` 提供谓词期望（定义域/值域/单值/互逆）；
//! - `shape_gate` 负责硬判定（实体未知/类型冲突/功能谓词冲突）与修复动作；
//! - 本模块负责问题生成、回答分类、账本状态机、覆盖层与模式诱导。
//!
//! 全部纯符号、零权重：账本、覆盖层与模式都按 BTree 排序，摘要由规范 JSON 得出。

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

impl RelationAcquisition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn_count(&self) -> usize {
        self.turn
    }

    pub fn active_goal(&self) -> Option<&RelationGoal> {
        self.active_goal.as_ref()
    }

    pub fn has_pending_goal(&self) -> bool {
        self.active_goal.is_some()
    }

    pub fn clear_goal(&mut self) {
        self.active_goal = None;
    }

    pub fn claims(&self) -> impl Iterator<Item = &AcquiredClaim> {
        self.claims.values()
    }

    pub fn claim(&self, id: &str) -> Option<&AcquiredClaim> {
        self.claims.get(id)
    }

    pub fn patterns(&self) -> impl Iterator<Item = &DialogueInducedPattern> {
        self.patterns.values()
    }

    pub fn status_count(&self, status: AcquisitionStatus) -> usize {
        self.claims
            .values()
            .filter(|claim| claim.status == status)
            .count()
    }

    pub fn overlay(&self) -> SessionOverlay {
        let mut rows = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() {
                continue;
            }
            rows.push(SessionOverlayRow {
                subject: claim.claim.subject.clone(),
                predicate: claim.claim.predicate.clone(),
                object: claim.claim.object.text(),
                claim_id: claim.id.clone(),
                status: claim.status,
            });
        }
        rows.sort_by(|left, right| {
            (&left.subject, &left.predicate, &left.object, &left.claim_id).cmp(&(
                &right.subject,
                &right.predicate,
                &right.object,
                &right.claim_id,
            ))
        });
        SessionOverlay { rows }
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self.snapshot()).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn snapshot(&self) -> AcquisitionSnapshot {
        AcquisitionSnapshot {
            dialect: ACQUISITION_DIALECT.to_string(),
            turn: self.turn,
            last_claim: self.last_claim.clone(),
            active_goal: self.active_goal.clone(),
            claims: self.claims.clone(),
            patterns: self.patterns.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &AcquisitionSnapshot) -> bool {
        if snapshot.dialect != ACQUISITION_DIALECT {
            return false;
        }
        self.turn = snapshot.turn;
        self.last_claim = snapshot.last_claim.clone();
        self.active_goal = snapshot.active_goal.clone();
        self.claims = snapshot.claims.clone();
        self.patterns = snapshot.patterns.clone();
        true
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&self.snapshot())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        fs::write(path, format!("{}\n", text))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let snapshot: AcquisitionSnapshot = serde_json::from_str(&text).ok()?;
        let mut acquisition = Self::new();
        acquisition.restore(&snapshot).then_some(acquisition)
    }

    // ---------------------------------------------------------------------
    // 问题理解与目标生成
    // ---------------------------------------------------------------------

    pub fn looks_like_question(input: &str) -> bool {
        let text = input.trim();
        text.ends_with('？')
            || text.ends_with('?')
            || text.ends_with('吗')
            || text.contains("是谁")
            || text.contains("是什么")
            || text.contains("是多少")
            || text.contains("哪一位")
            || text.contains("哪一个")
            || text.contains("在哪")
            || text.contains("什么关系")
    }

    pub fn looks_like_answer(input: &str) -> bool {
        let text = input.trim();
        !text.is_empty() && !Self::looks_like_question(text)
    }

    pub fn parse_question(input: &str) -> Option<ParsedQuestion> {
        let mut text = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim()
            .to_string();
        for prefix in [
            "请问",
            "问一下",
            "追问：",
            "追问:",
            "关系获取：",
            "关系获取:",
            "问答获取：",
            "问答获取:",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim().to_string();
                break;
            }
        }
        if text.is_empty() {
            return None;
        }

        if let Some((subject, predicate)) = parse_where_work_question(&text) {
            return Some(ParsedQuestion {
                subject,
                predicate,
                form: QuestionForm::Where,
            });
        }

        let reverse = text
            .strip_prefix("谁")
            .and_then(|rest| rest.strip_prefix("是").or_else(|| rest.strip_prefix("为")));
        if let Some(rest) = reverse {
            let (subject, predicate) = split_possession(rest)?;
            return Some(ParsedQuestion {
                subject,
                predicate,
                form: QuestionForm::Who,
            });
        }

        let (body, form) = split_question_marker(&text);
        if form == QuestionForm::Plain {
            return None;
        }
        let (subject, predicate) = split_possession(body)?;
        Some(ParsedQuestion {
            subject,
            predicate,
            form,
        })
    }

    pub fn start_question(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        let Some(parsed) = Self::parse_question(input) else {
            return GoalOpen::NoRelation {
                reason: "请使用「S 的 P 是谁/是什么」句式，或显式输入「关系获取：S 的 P」。"
                    .to_string(),
            };
        };
        self.start_relation_with_form(
            &parsed.subject,
            &parsed.predicate,
            parsed.form,
            GoalOrigin::UserRequest,
            facts,
            registry,
            types,
            shapes,
        )
    }

    /// 显式命令入口：既接受问句，也接受「S 的 P」这种不带疑问词的槽位声明。
    /// 由形状修复动作打开一个缺口追问（`RepairAction::AskProperty` 的落点）。
    pub fn start_shape_gap(
        &mut self,
        subject: &str,
        predicate: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        self.start_relation_with_form(
            subject,
            predicate,
            QuestionForm::Plain,
            GoalOrigin::ShapeRepair,
            facts,
            registry,
            types,
            shapes,
        )
    }

    pub fn start_command(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        if let Some(parsed) = Self::parse_question(input) {
            return self.start_relation_with_form(
                &parsed.subject,
                &parsed.predicate,
                parsed.form,
                GoalOrigin::UserRequest,
                facts,
                registry,
                types,
                shapes,
            );
        }
        if let Some((subject, predicate)) = split_possession(input) {
            return self.start_relation_with_form(
                &subject,
                &predicate,
                QuestionForm::Plain,
                GoalOrigin::UserRequest,
                facts,
                registry,
                types,
                shapes,
            );
        }
        self.start_question(input, facts, registry, types, shapes)
    }

    pub fn start_relation(
        &mut self,
        subject: &str,
        predicate: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        self.start_relation_with_form(
            subject,
            predicate,
            QuestionForm::Plain,
            GoalOrigin::UserRequest,
            facts,
            registry,
            types,
            shapes,
        )
    }

    fn start_relation_with_form(
        &mut self,
        subject: &str,
        predicate: &str,
        form: QuestionForm,
        origin: GoalOrigin,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        let subject = subject.trim();
        let surface = predicate.trim();
        if subject.is_empty() || surface.is_empty() {
            return GoalOpen::NoRelation {
                reason: "主语与谓词都不能为空。".to_string(),
            };
        }
        let resolution = registry.resolve(surface);
        let canonical = resolution.canonical.trim().to_string();
        if canonical.is_empty() {
            return GoalOpen::NoRelation {
                reason: "谓词为空。".to_string(),
            };
        }
        let known_path = registry.is_known(&canonical)
            || shapes.relations.iter().any(|shape| shape.path == canonical)
            || graph_path_exists(facts, registry, &canonical);
        if !known_path {
            return GoalOpen::UnknownPredicate {
                predicate: surface.to_string(),
            };
        }

        let graph_values =
            graph_values_for(facts, registry, subject, &canonical, resolution.inverted);
        if !graph_values.is_empty() {
            return GoalOpen::AlreadyKnown {
                subject: subject.to_string(),
                predicate: canonical,
                values: graph_values,
                source: "知识图谱".to_string(),
            };
        }
        let overlay_values = self.overlay_values_for(subject, &canonical, resolution.inverted);
        if !overlay_values.is_empty() {
            return GoalOpen::AlreadyKnown {
                subject: subject.to_string(),
                predicate: canonical,
                values: overlay_values,
                source: "会话覆盖层".to_string(),
            };
        }

        let slot_claims = self
            .claims
            .values()
            .filter(|claim| {
                claim.claim.subject == subject
                    && claim.claim.predicate == canonical
                    && claim.status.is_live()
            })
            .count();
        if slot_claims >= MAX_CLAIMS_PER_SLOT {
            return GoalOpen::NoRelation {
                reason: format!(
                    "槽位「{} 的 {}」已有 {} 条候选，达到上限；请先确认、否认或撤回旧候选。",
                    subject, canonical, slot_claims
                ),
            };
        }

        self.turn += 1;
        let profile = registry.profile(&canonical);
        let expectation = expectation_for(&canonical, resolution.inverted, profile, shapes, types);
        let question = question_text(subject, surface, &expectation);
        let asked = self
            .claims
            .values()
            .filter(|claim| claim.claim.subject == subject && claim.claim.predicate == canonical)
            .count()
            + 1;
        let id = format!(
            "qg_{}",
            fnv1a_hex(
                canonical_json(&serde_json::json!({
                    "subject": subject,
                    "predicate": canonical,
                    "surface": surface,
                    "form": form,
                    "turn": self.turn,
                }))
                .as_bytes(),
            )
        );
        let goal = RelationGoal {
            id,
            subject: subject.to_string(),
            predicate: canonical,
            surface: surface.to_string(),
            expectation,
            origin,
            form,
            question,
            turn: self.turn,
            asked,
        };
        self.active_goal = Some(goal.clone());
        GoalOpen::Opened { goal }
    }

    // ---------------------------------------------------------------------
    // 回答分类与账本写入
    // ---------------------------------------------------------------------

    /// 自由陈述入口：把已抽取、已归一方向的断言写入账本。
    ///
    /// 与 `interpret_answer` 的区别是：这里没有槽位目标，用户陈述本身
    /// 就是候选；是否确认仍由形状闸门与用户下一轮回答决定。
    #[allow(clippy::too_many_arguments)]
    pub fn record_assertion(
        &mut self,
        claim: Claim,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        self.turn += 1;
        self.record_claim(
            None, claim, source, raw, rule, facts, registry, types, shapes,
        )
    }

    pub fn interpret_answer(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let Some(goal) = self.active_goal.clone() else {
            return AcquisitionOutcome::message(
                AnswerAct::Clarify,
                "当前没有待填的关系槽位；请先用「关系获取：S 的 P」打开一个目标。",
            );
        };
        self.turn += 1;
        let classified = classify_answer(input);
        match classified.act {
            AnswerAct::Unknown => {
                self.active_goal = None;
                AcquisitionOutcome::message(
                    AnswerAct::Unknown,
                    format!(
                        "已记下：你暂时不知道「{}」的「{}」。该槽位保持未确认，不会写入覆盖层。",
                        goal.subject, goal.surface
                    ),
                )
            }
            AnswerAct::Refuse => {
                self.active_goal = None;
                AcquisitionOutcome::message(
                    AnswerAct::Refuse,
                    format!(
                        "已尊重你的选择：不再追问「{}」的「{}」。",
                        goal.subject, goal.surface
                    ),
                )
            }
            AnswerAct::Clarify => AcquisitionOutcome {
                follow_up: Some(goal.question.clone()),
                ..AcquisitionOutcome::message(
                    AnswerAct::Clarify,
                    format!(
                        "我需要一个具体取值来填「{}」的「{}」。\n{}",
                        goal.subject, goal.surface, goal.question
                    ),
                )
            },
            AnswerAct::Confirm => {
                if !classified.values.is_empty() {
                    let mut outcomes = Vec::new();
                    for value in &classified.values {
                        outcomes.push(self.record_value(
                            &goal,
                            value,
                            AcquisitionSource::UserConfirmation,
                            input,
                            "确认回答",
                            facts,
                            registry,
                            types,
                            shapes,
                        ));
                    }
                    return merge_outcomes(AnswerAct::Confirm, outcomes);
                }
                self.confirm(None)
            }
            AnswerAct::Deny => {
                let mut outcome = self.deny(None);
                outcome.act = AnswerAct::Deny;
                outcome
            }
            AnswerAct::Correct => {
                let mut messages = Vec::new();
                if let Some(target) = self.last_proposed_for_goal(&goal.id) {
                    if let Some(claim) = self.claims.get_mut(&target) {
                        claim.status = AcquisitionStatus::Refuted;
                        claim.notes.push(format!("用户纠错否证：{}", input.trim()));
                        messages.push(format!("已否证候选 {}", target));
                    }
                }
                if classified.values.is_empty() {
                    return AcquisitionOutcome::message(
                        AnswerAct::Correct,
                        format!(
                            "{}。请给出修正后的取值，例如「不是叶莉，是邓紫棋」。",
                            messages.join("；")
                        ),
                    );
                }
                let mut outcomes = Vec::new();
                for value in &classified.values {
                    outcomes.push(self.record_value(
                        &goal,
                        value,
                        AcquisitionSource::UserCorrection,
                        input,
                        "纠错回答",
                        facts,
                        registry,
                        types,
                        shapes,
                    ));
                }
                let mut merged = merge_outcomes(AnswerAct::Correct, outcomes);
                if !messages.is_empty() {
                    merged.message = format!("{}；{}", messages.join("；"), merged.message);
                }
                merged
            }
            AnswerAct::Provide => {
                if classified.values.is_empty() {
                    return AcquisitionOutcome {
                        follow_up: Some(goal.question.clone()),
                        ..AcquisitionOutcome::message(
                            AnswerAct::Clarify,
                            format!("没有从「{}」中解析出可用的取值。", input.trim()),
                        )
                    };
                }
                let mut outcomes = Vec::new();
                for value in &classified.values {
                    outcomes.push(self.record_value(
                        &goal,
                        value,
                        AcquisitionSource::UserAnswer,
                        input,
                        "直接回答",
                        facts,
                        registry,
                        types,
                        shapes,
                    ));
                }
                merge_outcomes(AnswerAct::Provide, outcomes)
            }
        }
    }

    fn record_value(
        &mut self,
        goal: &RelationGoal,
        value: &str,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let value = clean_value(value);
        if value.is_empty() {
            return AcquisitionOutcome::message(
                AnswerAct::Clarify,
                "取值为空，请重新给出一个具体值。",
            );
        }
        let claim = build_claim_for_goal(goal, &value);
        self.record_claim(
            Some(goal),
            claim,
            source,
            raw,
            rule,
            facts,
            registry,
            types,
            shapes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_claim(
        &mut self,
        goal: Option<&RelationGoal>,
        claim: Claim,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let claim = normalize_claim(claim, registry);
        let id = claim_id(&claim);
        let candidate = InterpretationCandidate {
            claim: claim.clone(),
            score: 100,
            segmentation: Vec::new(),
        };
        let decision = shapes.decide_candidates(&[candidate], facts, registry, types);
        let Some(verdict) = decision.verdicts.first() else {
            return AcquisitionOutcome::message(
                AnswerAct::Provide,
                "形状闸门没有返回裁决；本轮不写入账本。",
            );
        };

        let mut notes = Vec::new();
        let mut conflicts: BTreeSet<String> = BTreeSet::new();
        let mut status = match verdict.status {
            CandidateStatus::Accepted => {
                if matches!(
                    source,
                    AcquisitionSource::UserConfirmation | AcquisitionSource::UserCorrection
                ) {
                    AcquisitionStatus::Confirmed
                } else {
                    AcquisitionStatus::Proposed
                }
            }
            CandidateStatus::UnknownEntity => {
                notes.push(verdict.detail.clone());
                notes.push("新词面先作为候选，需确认后才进入覆盖层。".to_string());
                AcquisitionStatus::Proposed
            }
            CandidateStatus::UnknownPredicate => {
                notes.push(verdict.detail.clone());
                AcquisitionStatus::Rejected
            }
            CandidateStatus::TypeMismatch => {
                notes.push(verdict.detail.clone());
                AcquisitionStatus::Conflicted
            }
            CandidateStatus::ValueConflict => {
                notes.push(verdict.detail.clone());
                if let Some(RepairAction::DiscloseConflict { values, .. }) = &verdict.repair {
                    conflicts.extend(values.iter().cloned());
                }
                AcquisitionStatus::Conflicted
            }
        };
        if let Some(RepairAction::DiscloseConflict { values, .. }) = &verdict.repair {
            conflicts.extend(values.iter().cloned());
        }

        let object = claim.object.text();
        let functional = goal
            .map(|goal| goal.expectation.functional)
            .unwrap_or_else(|| registry.is_functional(&claim.predicate));
        if functional && status != AcquisitionStatus::Rejected {
            let mut others: BTreeSet<String> = self
                .claims
                .values()
                .filter(|other| {
                    other.id != id
                        && other.status.is_live()
                        && other.claim.subject == claim.subject
                        && other.claim.predicate == claim.predicate
                        && other.claim.object.text() != object
                })
                .map(|other| other.claim.object.text())
                .collect();
            if !others.is_empty() {
                status = AcquisitionStatus::Conflicted;
                notes.push("单值槽出现不同取值，需用户裁定。".to_string());
                conflicts.append(&mut others);
            }
        }

        let outcome = if let Some(existing) = self.claims.get_mut(&id) {
            let independent = existing.turn != self.turn || existing.raw_span != raw;
            if independent {
                existing.support += 1;
            }
            if source == AcquisitionSource::UserConfirmation {
                existing.confirmations += 1;
            }
            if existing.status == AcquisitionStatus::Refuted
                || existing.status == AcquisitionStatus::Rejected
            {
                existing.status = AcquisitionStatus::Proposed;
                existing.conflicts.clear();
                existing.notes.clear();
            }
            if status == AcquisitionStatus::Conflicted {
                existing.status = AcquisitionStatus::Conflicted;
                existing.conflicts.extend(conflicts.iter().cloned());
            } else if matches!(
                source,
                AcquisitionSource::UserConfirmation | AcquisitionSource::UserCorrection
            ) && existing.conflicts.is_empty()
            {
                existing.status = AcquisitionStatus::Confirmed;
            } else if existing.support >= CONFIRMATION_SUPPORT
                && existing.status == AcquisitionStatus::Proposed
                && existing.conflicts.is_empty()
            {
                existing.status = AcquisitionStatus::Confirmed;
            }
            for note in &notes {
                if !existing.notes.contains(note) {
                    existing.notes.push(note.clone());
                }
            }
            existing.turn = self.turn;
            existing.raw_span = raw.trim().to_string();
            existing.source = source;
            existing.rule = rule.to_string();
            existing.status
        } else {
            let entry = AcquiredClaim {
                id: id.clone(),
                goal_id: goal.map(|goal| goal.id.clone()),
                claim: claim.clone(),
                status,
                source,
                turn: self.turn,
                raw_span: raw.trim().to_string(),
                rule: rule.to_string(),
                support: 1,
                confirmations: usize::from(source == AcquisitionSource::UserConfirmation),
                conflicts,
                notes,
            };
            let status = entry.status;
            self.claims.insert(id.clone(), entry);
            status
        };

        self.last_claim = Some(id.clone());
        if outcome.is_committed() {
            if let Some(goal) = goal {
                self.induce_pattern(goal, &claim);
            }
        }

        let claim_ref = self.claims.get(&id);
        let mut message =
            match outcome {
                AcquisitionStatus::Proposed => format!(
                    "已记为候选：{} 的 {} = {}。若正确请回复「确认」，否则请纠错。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Confirmed => format!(
                    "已确认：{} 的 {} = {}（仅会话覆盖层，未写入知识图谱）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Conflicted => format!(
                "发现冲突：{} 的 {} = {}；已有取值 {}。请用「不是…，是…」纠错或输入「确认」裁定。",
                claim.subject,
                claim.predicate,
                claim.object.text(),
                claim_ref
                    .map(|claim| claim.conflicts.iter().cloned().collect::<Vec<_>>().join("、"))
                    .unwrap_or_default()
            ),
                AcquisitionStatus::Refuted => format!(
                    "已否证：{} 的 {} = {}。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Admitted => format!(
                    "已采纳：{} 的 {} = {}（已加入会话事实集）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Rejected => format!(
                    "未写入账本：{} 的 {} = {}（形状闸门拒绝）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
            };
        if let Some(claim) = claim_ref {
            if !claim.notes.is_empty() {
                message.push_str(&format!("\n  说明：{}", claim.notes.join("；")));
            }
        }
        let commit = if outcome.is_committed() {
            Some(CommitProposal {
                claim_id: id.clone(),
                sentence: commit_sentence(&claim),
                status: outcome,
            })
        } else {
            None
        };
        let follow_up = match outcome {
            AcquisitionStatus::Proposed => Some(
                "如果这条关系正确，请回复「确认」；如果不对，请回复「不是X，是Y」。".to_string(),
            ),
            AcquisitionStatus::Conflicted => Some("请先裁定冲突值，再写入会话覆盖层。".to_string()),
            AcquisitionStatus::Confirmed => Some(format!(
                "可输入「采纳获取 {}」把它加入会话事实集（仍不写知识图谱）。",
                id
            )),
            _ => None,
        };
        AcquisitionOutcome {
            act: AnswerAct::Provide,
            status: Some(outcome),
            claim_id: Some(id),
            message,
            commit,
            follow_up,
        }
    }

    // ---------------------------------------------------------------------
    // 确认 / 否认 / 撤回 / 采纳
    // ---------------------------------------------------------------------

    pub fn confirm(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_proposed_for_goal_current())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                "账本中没有可确认的候选；请先回答一个关系槽位。",
            );
        };
        let Some(claim) = self.claims.get(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                format!("账本中没有条目 {}。", id),
            );
        };
        if matches!(
            claim.status,
            AcquisitionStatus::Refuted | AcquisitionStatus::Rejected
        ) {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                format!(
                    "条目 {} 已被否证或拒绝，不能直接确认；请重新回答该槽位。",
                    id
                ),
            );
        }
        let subject = claim.claim.subject.clone();
        let predicate = claim.claim.predicate.clone();
        let object = claim.claim.object.text();
        let goal = self
            .active_goal
            .clone()
            .filter(|goal| goal.subject == subject || goal.predicate == predicate);
        let mut refuted = Vec::new();
        if claim.status == AcquisitionStatus::Conflicted {
            let conflicting: Vec<String> = self
                .claims
                .values()
                .filter(|other| {
                    other.id != id
                        && other.claim.subject == subject
                        && other.claim.predicate == predicate
                        && other.claim.object.text() != object
                        && other.status.is_live()
                })
                .map(|other| other.id.clone())
                .collect();
            for other in conflicting {
                if let Some(other) = self.claims.get_mut(&other) {
                    other.status = AcquisitionStatus::Refuted;
                    other.notes.push(format!("用户确认 {} 后退役冲突值。", id));
                }
                refuted.push(other);
            }
        }
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(AnswerAct::Confirm, "账本条目已消失。");
        };
        claim.status = AcquisitionStatus::Confirmed;
        claim.confirmations += 1;
        claim.support += 1;
        claim.conflicts.clear();
        claim.notes.push("用户显式确认。".to_string());
        claim.turn = self.turn;
        let sentence = commit_sentence(&claim.claim);
        self.last_claim = Some(id.clone());
        let goal_for_pattern = goal.clone();
        let claim_for_pattern = claim.claim.clone();
        if let Some(goal) = goal_for_pattern.as_ref() {
            self.induce_pattern(goal, &claim_for_pattern);
        }
        let mut message = format!(
            "已确认：{} 的 {} = {}（仅会话覆盖层，未写入知识图谱）。",
            subject, predicate, object
        );
        if !refuted.is_empty() {
            message.push_str(&format!(
                "\n  已按你的裁定退役 {} 条冲突候选。",
                refuted.len()
            ));
        }
        AcquisitionOutcome {
            act: AnswerAct::Confirm,
            status: Some(AcquisitionStatus::Confirmed),
            claim_id: Some(id.clone()),
            message,
            commit: Some(CommitProposal {
                claim_id: id.clone(),
                sentence,
                status: AcquisitionStatus::Confirmed,
            }),
            follow_up: Some(format!(
                "可输入「采纳获取 {}」把它加入会话事实集（仍不写知识图谱）。",
                id
            )),
        }
    }

    pub fn deny(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_proposed_for_goal_current())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(AnswerAct::Deny, "账本中没有可否认的候选。");
        };
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Deny,
                format!("账本中没有条目 {}。", id),
            );
        };
        claim.status = AcquisitionStatus::Refuted;
        claim.notes.push("用户否认。".to_string());
        let rendered = format!(
            "{} 的 {} = {}",
            claim.claim.subject,
            claim.claim.predicate,
            claim.claim.object.text()
        );
        if self
            .active_goal
            .as_ref()
            .map(|goal| self.last_proposed_for_goal(&goal.id).is_none())
            .unwrap_or(false)
        {
            self.active_goal = None;
        }
        AcquisitionOutcome {
            act: AnswerAct::Deny,
            status: Some(AcquisitionStatus::Refuted),
            claim_id: Some(id),
            message: format!("已否证候选：{}。", rendered),
            commit: None,
            follow_up: None,
        }
    }

    pub fn retract(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(AnswerAct::Deny, "账本中没有可撤回的条目。");
        };
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Deny,
                format!("账本中没有条目 {}。", id),
            );
        };
        claim.status = AcquisitionStatus::Refuted;
        claim.notes.push("用户撤回。".to_string());
        let rendered = format!(
            "{} 的 {} = {}",
            claim.claim.subject,
            claim.claim.predicate,
            claim.claim.object.text()
        );
        AcquisitionOutcome {
            act: AnswerAct::Deny,
            status: Some(AcquisitionStatus::Refuted),
            claim_id: Some(id),
            message: format!("已撤回会话覆盖层条目：{}。", rendered),
            commit: None,
            follow_up: None,
        }
    }

    pub fn commit_proposal(&self, id: Option<&str>) -> Option<CommitProposal> {
        let id = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_claim.clone())?;
        let claim = self.claims.get(&id)?;
        if !claim.status.is_committed() {
            return None;
        }
        Some(CommitProposal {
            claim_id: id,
            sentence: commit_sentence(&claim.claim),
            status: claim.status,
        })
    }

    pub fn mark_admitted(&mut self, id: &str) -> bool {
        let Some(claim) = self.claims.get_mut(id) else {
            return false;
        };
        if !claim.status.is_committed() {
            return false;
        }
        claim.status = AcquisitionStatus::Admitted;
        claim.notes.push("已采纳到会话事实集。".to_string());
        true
    }

    pub fn retire_pattern(&mut self, id: &str) -> bool {
        let Some(pattern) = self.patterns.get_mut(id) else {
            return false;
        };
        pattern.retired = true;
        pattern.active = false;
        true
    }

    // ---------------------------------------------------------------------
    // 覆盖层查询与模式诱导
    // ---------------------------------------------------------------------

    fn overlay_values_for(&self, subject: &str, predicate: &str, inverted: bool) -> Vec<String> {
        let mut values = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() || claim.claim.predicate != predicate {
                continue;
            }
            if inverted {
                if claim.claim.object.text() == subject {
                    values.push(claim.claim.subject.clone());
                }
            } else if claim.claim.subject == subject {
                values.push(claim.claim.object.text());
            }
        }
        values.sort();
        values.dedup();
        values
    }

    pub fn answer_from_overlay(
        &self,
        input: &str,
        registry: &PredicateRegistry,
    ) -> Option<OverlayAnswer> {
        let parsed = Self::parse_question(input)?;
        let resolution = registry.resolve(&parsed.predicate);
        let canonical = resolution.canonical.trim();
        if canonical.is_empty() {
            return None;
        }
        let mut values = Vec::new();
        let mut claim_ids = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() || claim.claim.predicate != canonical {
                continue;
            }
            let matches = if resolution.inverted {
                claim.claim.object.text() == parsed.subject
            } else {
                claim.claim.subject == parsed.subject
            };
            if matches {
                values.push(if resolution.inverted {
                    claim.claim.subject.clone()
                } else {
                    claim.claim.object.text()
                });
                claim_ids.push(claim.id.clone());
            }
        }
        if values.is_empty() {
            return None;
        }
        values.sort();
        values.dedup();
        claim_ids.sort();
        claim_ids.dedup();
        Some(OverlayAnswer {
            subject: parsed.subject,
            predicate: parsed.predicate,
            values,
            claim_ids,
        })
    }

    fn last_proposed_for_goal(&self, goal_id: &str) -> Option<String> {
        self.claims
            .values()
            .filter(|claim| claim.goal_id.as_deref() == Some(goal_id))
            .filter(|claim| {
                matches!(
                    claim.status,
                    AcquisitionStatus::Proposed | AcquisitionStatus::Conflicted
                )
            })
            .max_by(|left, right| {
                left.turn
                    .cmp(&right.turn)
                    .then_with(|| left.id.cmp(&right.id))
            })
            .map(|claim| claim.id.clone())
    }

    fn last_proposed_for_goal_current(&self) -> Option<String> {
        let goal = self.active_goal.as_ref()?;
        self.last_proposed_for_goal(&goal.id)
    }

    fn induce_pattern(&mut self, goal: &RelationGoal, claim: &Claim) {
        let key = format!("{}|{}", claim.predicate, goal.form.label());
        let id = format!("dip_{}", fnv1a_hex(key.as_bytes()));
        let source_claim = claim_id(claim);
        let entry = self
            .patterns
            .entry(id.clone())
            .or_insert_with(|| DialogueInducedPattern {
                id: id.clone(),
                predicate: claim.predicate.clone(),
                surface: goal.surface.clone(),
                form: goal.form,
                question_template: if goal.form == QuestionForm::Plain {
                    format!("{{subject}}的{}是什么", goal.surface)
                } else {
                    format!("{{subject}}的{}是{}", goal.surface, goal.form.label())
                },
                statement_template: format!("{{subject}}的{}是{{object}}", goal.surface),
                support: 0,
                active: false,
                retired: false,
                source_claims: BTreeSet::new(),
                examples: BTreeSet::new(),
            });
        if !entry.source_claims.insert(source_claim) {
            return;
        }
        entry.support += 1;
        entry.examples.insert(format!(
            "{}的{}是{}",
            claim.subject,
            goal.surface,
            claim.object.text()
        ));
        if entry.support >= CONFIRMATION_SUPPORT && !entry.retired {
            entry.active = true;
        }
    }

    pub fn scan_text(&self, text: &str, registry: &PredicateRegistry) -> PatternScan {
        let active: Vec<&DialogueInducedPattern> = self
            .patterns
            .values()
            .filter(|pattern| pattern.active)
            .take(MAX_MATCHED_PATTERNS)
            .collect();
        if active.is_empty() {
            return PatternScan::default();
        }
        let mut claims = Vec::new();
        let mut used = BTreeSet::new();
        for segment in text.split(|ch| matches!(ch, '。' | '；' | ';' | '！' | '!' | '\n')) {
            let segment = segment.trim();
            if segment.is_empty() || segment.contains('？') || segment.contains('?') {
                continue;
            }
            let Some((subject, rest)) = segment.split_once('的') else {
                continue;
            };
            let Some((predicate_surface, object)) = rest.split_once('是') else {
                continue;
            };
            let object = clean_value(object);
            if subject.trim().is_empty() || object.is_empty() {
                continue;
            }
            let resolution = registry.resolve(predicate_surface.trim());
            let canonical = resolution.canonical.trim();
            for pattern in &active {
                if pattern.predicate != canonical {
                    continue;
                }
                let claim = if resolution.inverted {
                    Claim::entity(object.clone(), canonical, subject.trim())
                } else {
                    Claim::entity(subject.trim(), canonical, object.clone())
                };
                used.insert(pattern.id.clone());
                claims.push(claim);
            }
        }
        claims.sort_by(|left, right| claim_key(left).cmp(&claim_key(right)));
        claims.dedup_by(|left, right| claim_key(left) == claim_key(right));
        PatternScan {
            claims,
            patterns: used,
        }
    }

    pub fn absorb_pattern_scan(
        &mut self,
        text: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        self.turn += 1;
        let scan = self.scan_text(text, registry);
        if scan.is_empty() {
            return AcquisitionOutcome {
                act: AnswerAct::Provide,
                status: None,
                claim_id: None,
                message: scan.render_text(),
                commit: None,
                follow_up: None,
            };
        }
        let claims = scan.claims.clone();
        let mut outcomes = Vec::new();
        for claim in claims {
            outcomes.push(self.record_claim(
                None,
                claim,
                AcquisitionSource::InducedPattern,
                text,
                "对话诱导模式扫描",
                facts,
                registry,
                types,
                shapes,
            ));
        }
        let mut merged = merge_outcomes(AnswerAct::Provide, outcomes);
        merged.message = format!("{}\n{}", scan.render_text(), merged.message);
        merged
    }

    // ---------------------------------------------------------------------
    // 报告
    // ---------------------------------------------------------------------

    pub fn report(&self) -> String {
        let mut lines = vec![format!(
            "问答式关系获取（QRA v{}）：轮次 {}，账本 {} 条（候选 {} / 已确认 {} / 冲突 {} / 已否证 {} / 已拒绝 {} / 已采纳 {}），模式 {} 个（激活 {}）",
            ACQUISITION_DIALECT.trim_start_matches("knowprolog-qra/"),
            self.turn,
            self.claims.len(),
            self.status_count(AcquisitionStatus::Proposed),
            self.status_count(AcquisitionStatus::Confirmed),
            self.status_count(AcquisitionStatus::Conflicted),
            self.status_count(AcquisitionStatus::Refuted),
            self.status_count(AcquisitionStatus::Rejected),
            self.status_count(AcquisitionStatus::Admitted),
            self.patterns.len(),
            self.patterns.values().filter(|pattern| pattern.active).count()
        )];
        match &self.active_goal {
            Some(goal) => lines.push(format!("待填槽位：{}", goal.render_text())),
            None => lines.push("待填槽位：（无）".to_string()),
        }
        if self.claims.is_empty() {
            lines.push("账本：（空；输入「关系获取：姚明的女儿是谁」开始）".to_string());
        } else {
            lines.push("账本：".to_string());
            for claim in self.claims.values() {
                lines.push(claim.render_text());
            }
        }
        lines.push(self.overlay().render_text());
        lines.push(self.patterns_report());
        lines.push(format!("账本摘要：{}", self.digest()));
        lines.join("\n")
    }

    pub fn patterns_report(&self) -> String {
        let mut lines = vec![format!(
            "对话诱导模式（DIP）：{} 个（激活 {}）",
            self.patterns.len(),
            self.patterns
                .values()
                .filter(|pattern| pattern.active)
                .count()
        )];
        if self.patterns.is_empty() {
            lines.push("  （空；同一谓词被两次确认后自动诱导）".to_string());
        } else {
            for pattern in self.patterns.values() {
                lines.push(pattern.render_text());
            }
        }
        lines.join("\n")
    }
}

/// 「S 在哪(里/儿/个单位)工作」：中文里最常见的任职问法，不是所有格。
/// 归一为规范谓词「任职于」，让覆盖层与图谱走同一方向。
fn parse_where_work_question(text: &str) -> Option<(String, String)> {
    let suffix = ["工作", "上班", "任职", "就职", "供职"]
        .iter()
        .find(|suffix| text.ends_with(*suffix))?;
    let body = text[..text.len() - suffix.len()]
        .trim_end_matches(['了', '过'])
        .trim();
    let position = body.rfind('在')?;
    let tail = &body[position + '在'.len_utf8()..];
    if !(tail.is_empty() || tail.starts_with('哪') || tail.starts_with("什么")) {
        return None;
    }
    let subject = body[..position].trim();
    if subject.is_empty() {
        return None;
    }
    Some((subject.to_string(), "任职于".to_string()))
}

fn split_possession(text: &str) -> Option<(String, String)> {
    let (subject, predicate) = text.rsplit_once('的')?;
    let subject = subject.trim();
    let predicate = predicate.trim();
    if subject.is_empty() || predicate.is_empty() {
        return None;
    }
    Some((subject.to_string(), predicate.to_string()))
}

fn split_question_marker(text: &str) -> (&str, QuestionForm) {
    let markers: [(&str, QuestionForm); 18] = [
        ("是谁", QuestionForm::Who),
        ("是哪一位", QuestionForm::Who),
        ("是哪一个", QuestionForm::Who),
        ("是什么人", QuestionForm::Who),
        ("叫什么", QuestionForm::Who),
        ("是什么", QuestionForm::What),
        ("是啥", QuestionForm::What),
        ("是多少", QuestionForm::Value),
        ("是几", QuestionForm::Value),
        ("是哪里", QuestionForm::Where),
        ("在哪", QuestionForm::Where),
        ("呢", QuestionForm::What),
        ("吗", QuestionForm::What),
        ("谁", QuestionForm::Who),
        ("什么", QuestionForm::What),
        ("多少", QuestionForm::Value),
        ("哪里", QuestionForm::Where),
        ("哪儿", QuestionForm::Where),
    ];
    for (marker, form) in markers {
        if let Some(body) = text.strip_suffix(marker) {
            let body = body.trim();
            if !body.is_empty() {
                return (body, form);
            }
        }
    }
    (text, QuestionForm::Plain)
}

fn graph_path_exists(facts: &[PrologFact], registry: &PredicateRegistry, canonical: &str) -> bool {
    facts.iter().any(|fact| match fact {
        PrologFact::Entity { properties, .. } => properties
            .keys()
            .any(|key| registry.resolve(key).canonical == canonical),
        PrologFact::Relation { rel_type, .. } => registry.resolve(rel_type).canonical == canonical,
    })
}

fn graph_values_for(
    facts: &[PrologFact],
    registry: &PredicateRegistry,
    subject: &str,
    canonical: &str,
    inverted: bool,
) -> Vec<String> {
    let mut values = Vec::new();
    for fact in facts {
        match fact {
            PrologFact::Relation { from, rel_type, to } => {
                let resolution = registry.resolve(rel_type);
                if resolution.canonical != canonical {
                    continue;
                }
                let (canonical_subject, canonical_object) = if resolution.inverted {
                    (to.as_str(), from.as_str())
                } else {
                    (from.as_str(), to.as_str())
                };
                if inverted {
                    if canonical_object == subject {
                        values.push(canonical_subject.to_string());
                    }
                } else if canonical_subject == subject {
                    values.push(canonical_object.to_string());
                }
            }
            PrologFact::Entity {
                name, properties, ..
            } => {
                if inverted || name != subject {
                    continue;
                }
                for (key, value) in properties {
                    if registry.resolve(key).canonical == canonical {
                        values.push(value.clone());
                    }
                }
            }
        }
    }
    values.sort();
    values.dedup();
    values
}

fn expectation_for(
    canonical: &str,
    inverted: bool,
    profile: Option<&PredicateProfile>,
    shapes: &ShapeSet,
    types: &TypeSystem,
) -> Expectation {
    let relation_shape = shapes
        .relations
        .iter()
        .find(|shape| shape.path == canonical);
    let property_shape = shapes
        .nodes
        .iter()
        .flat_map(|node| node.properties.iter())
        .find(|shape| shape.path == canonical);

    let mut raw_types = BTreeSet::new();
    if let Some(profile) = profile {
        if inverted {
            raw_types.extend(profile.domain.iter().cloned());
        } else {
            raw_types.extend(profile.range.iter().cloned());
        }
    }
    if let Some(shape) = relation_shape {
        if inverted {
            raw_types.extend(shape.subject_class.iter().cloned());
        } else {
            raw_types.extend(shape.class.iter().cloned());
        }
    }
    if let Some(shape) = property_shape {
        raw_types.extend(shape.class.iter().cloned());
    }
    let entity_types: Vec<String> = raw_types
        .into_iter()
        .map(|name| types.canonical(&name))
        .filter(|name| !name.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let datatype = property_shape.and_then(|shape| shape.datatype.clone());
    let functional = profile.map(|profile| profile.functional).unwrap_or(false)
        || relation_shape
            .and_then(|shape| shape.max_count)
            .map(|count| count == 1)
            .unwrap_or(false);
    let required = property_shape
        .and_then(|shape| shape.min_count)
        .map(|count| count > 0)
        .unwrap_or(false);
    let value_shape = if datatype.is_some() {
        ValueShape::Literal
    } else if entity_types.is_empty() {
        ValueShape::Infer
    } else {
        ValueShape::Entity
    };
    Expectation {
        entity_types,
        datatype,
        functional,
        inverse: inverted,
        required,
        value_shape,
    }
}

fn question_text(subject: &str, surface: &str, expectation: &Expectation) -> String {
    let slot = match expectation.value_shape {
        ValueShape::Entity => {
            if expectation
                .entity_types
                .iter()
                .any(|name| name == "人物" || name == "person")
            {
                "谁"
            } else {
                "哪个实体"
            }
        }
        ValueShape::Literal => "什么值",
        ValueShape::Infer => "什么",
    };
    let hint = expectation.render_text();
    if expectation.functional {
        format!(
            "我暂时没有「{}」的「{}」记录。你能告诉我是{}吗？（{}，只接受一个取值）",
            subject, surface, slot, hint
        )
    } else {
        format!(
            "我暂时没有「{}」的「{}」记录。你能告诉我是{}吗？（期望：{}）",
            subject, surface, slot, hint
        )
    }
}

fn clean_value(raw: &str) -> String {
    let mut value = raw
        .trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '。' | '，' | ',' | '；' | ';' | '！' | '!' | ' ' | '“' | '”' | '"' | '\''
            )
        })
        .trim()
        .to_string();
    for prefix in [
        "应该是",
        "其实是",
        "就是",
        "她叫",
        "他叫",
        "叫作",
        "叫做",
        "叫",
        "是",
    ] {
        if let Some(rest) = value.strip_prefix(prefix) {
            value = rest.trim().to_string();
            break;
        }
    }
    value
}

fn classify_answer(input: &str) -> ClassifiedAnswer {
    let text = input.trim().trim_end_matches(['。', '！', '!', ' ']).trim();
    let lower = text.to_ascii_lowercase();
    if text.is_empty() {
        return ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new());
    }
    if ClassifiedAnswer::contains_any(
        text,
        &[
            "不知道",
            "不清楚",
            "不确定",
            "不记得",
            "忘了",
            "没印象",
            "无可奉告",
        ],
    ) {
        return ClassifiedAnswer::new(AnswerAct::Unknown, Vec::new());
    }
    if ClassifiedAnswer::contains_any(
        text,
        &[
            "不想说",
            "不能说",
            "不方便说",
            "保密",
            "拒绝回答",
            "不想回答",
        ],
    ) {
        return ClassifiedAnswer::new(AnswerAct::Refuse, Vec::new());
    }
    if text.ends_with('？') || text.ends_with('?') || text.ends_with('吗') {
        return ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new());
    }

    if is_denial(text) {
        if let Some(replacement) = correction_tail(text) {
            let values = extract_values(&replacement);
            if !values.is_empty() {
                return ClassifiedAnswer::new(AnswerAct::Correct, values);
            }
        }
        return ClassifiedAnswer::new(AnswerAct::Deny, Vec::new());
    }

    if matches!(
        lower.as_str(),
        "对" | "对呀"
            | "对的"
            | "是"
            | "是的"
            | "是的呀"
            | "没错"
            | "正确"
            | "确定"
            | "嗯"
            | "嗯嗯"
            | "好"
            | "可以"
            | "yes"
            | "y"
            | "ok"
            | "okay"
    ) {
        return ClassifiedAnswer::new(AnswerAct::Confirm, Vec::new());
    }
    if let Some(rest) = text
        .strip_prefix("确认")
        .or_else(|| text.strip_prefix("对，"))
        .or_else(|| text.strip_prefix("对,"))
    {
        let values = extract_values(rest);
        return ClassifiedAnswer::new(AnswerAct::Confirm, values);
    }

    let values = extract_values(text);
    if values.is_empty() {
        ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new())
    } else {
        ClassifiedAnswer::new(AnswerAct::Provide, values)
    }
}

fn is_denial(text: &str) -> bool {
    let text = text.trim();
    matches!(
        text,
        "不" | "不是" | "不对" | "错" | "错了" | "否" | "没有" | "没"
    ) || text.starts_with("不是")
        || text.starts_with("不对")
        || text.starts_with("错了")
        || text.starts_with("错，")
        || text.starts_with("错,")
        || text.starts_with("否")
        || text.starts_with("不，")
        || text.starts_with("不,")
}

fn correction_tail(text: &str) -> Option<String> {
    for marker in ["而是", "改是", "应该是", "其实是"] {
        if let Some(index) = text.rfind(marker) {
            let tail = text[index + marker.len()..].trim();
            if !tail.is_empty() {
                return Some(tail.to_string());
            }
        }
    }
    let denial_start = ["不是", "不对", "错了", "错", "不"]
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()?;
    let after_denial = &text[denial_start..];
    let denial = ["不是", "不对", "错了", "错", "不"]
        .iter()
        .find(|marker| after_denial.starts_with(*marker))?;
    let tail_zone = &after_denial[denial.len()..];
    let index = tail_zone.rfind('是')?;
    let tail = tail_zone[index + '是'.len_utf8()..].trim();
    if tail.is_empty() {
        None
    } else {
        Some(tail.to_string())
    }
}

fn extract_values(text: &str) -> Vec<String> {
    let text = clean_value(text);
    if text.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::new();
    // 一个槽位只取主句：逗号后的补充说明（如「叶莉，1980 年出生」）不当作第二取值。
    let head = text.split(['，', ',']).next().unwrap_or(&text);
    for part in head.split(['、', '；', ';']) {
        let value = clean_value(part);
        if value.is_empty() || value == "谁" || value == "什么" {
            continue;
        }
        values.push(value);
    }
    values.sort();
    values.dedup();
    values
}

fn build_claim_for_goal(goal: &RelationGoal, value: &str) -> Claim {
    let literal = match goal.expectation.value_shape {
        ValueShape::Literal => true,
        ValueShape::Entity => false,
        ValueShape::Infer => looks_like_literal(value),
    };
    let normalize_object = |value: &str| {
        if literal {
            ClaimObject::Literal(value.to_string())
        } else {
            ClaimObject::Entity(value.to_string())
        }
    };
    if goal.expectation.inverse {
        Claim {
            subject: value.to_string(),
            predicate: goal.predicate.clone(),
            object: normalize_object(&goal.subject),
        }
    } else {
        Claim {
            subject: goal.subject.clone(),
            predicate: goal.predicate.clone(),
            object: normalize_object(value),
        }
    }
}

fn looks_like_literal(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.parse::<f64>().is_ok()
        || matches!(trimmed, "是" | "否" | "true" | "false")
        || crate::shape_gate::datatype_conforms("date", trimmed)
        || crate::shape_gate::datatype_conforms("time", trimmed)
        || crate::shape_gate::datatype_conforms("datetime", trimmed)
}

fn normalize_claim(claim: Claim, registry: &PredicateRegistry) -> Claim {
    let resolution = registry.resolve(&claim.predicate);
    if !resolution.inverted {
        Claim {
            subject: claim.subject,
            predicate: resolution.canonical,
            object: claim.object,
        }
    } else {
        match claim.object {
            ClaimObject::Entity(object) => Claim {
                subject: object,
                predicate: resolution.canonical,
                object: ClaimObject::Entity(claim.subject),
            },
            ClaimObject::Literal(value) => Claim {
                subject: claim.subject,
                predicate: resolution.canonical,
                object: ClaimObject::Literal(value),
            },
        }
    }
}

fn claim_key(claim: &Claim) -> String {
    canonical_json(&serde_json::json!({
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
    }))
}

fn claim_id(claim: &Claim) -> String {
    format!("qc_{}", fnv1a_hex(claim_key(claim).as_bytes()))
}

fn commit_sentence(claim: &Claim) -> String {
    format!(
        "{}的{}是{}",
        claim.subject,
        claim.predicate,
        claim.object.text()
    )
}

fn merge_outcomes(act: AnswerAct, outcomes: Vec<AcquisitionOutcome>) -> AcquisitionOutcome {
    let mut message = Vec::new();
    let mut claim_id = None;
    let mut status = None;
    let mut commit = None;
    let mut follow_up = None;
    for outcome in outcomes {
        if !outcome.message.is_empty() {
            message.push(outcome.message);
        }
        if outcome.claim_id.is_some() {
            claim_id = outcome.claim_id;
        }
        if outcome.status.is_some() {
            status = outcome.status;
        }
        if outcome.commit.is_some() {
            commit = outcome.commit;
        }
        if outcome.follow_up.is_some() {
            follow_up = outcome.follow_up;
        }
    }
    AcquisitionOutcome {
        act,
        status,
        claim_id,
        message: message.join("\n"),
        commit,
        follow_up,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClassifiedAnswer {
    act: AnswerAct,
    values: Vec<String>,
}

impl ClassifiedAnswer {
    fn new(act: AnswerAct, values: Vec<String>) -> Self {
        Self { act, values }
    }

    fn contains_any(text: &str, needles: &[&str]) -> bool {
        needles.iter().any(|needle| text.contains(needle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use std::collections::HashMap;

    fn entity(name: &str, etype: &str, properties: &[(&str, &str)]) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
        }
    }

    fn relation(from: &str, predicate: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: predicate.to_string(),
            to: to.to_string(),
        }
    }

    fn profile(
        name: &str,
        functional: bool,
        domain: &[&str],
        range: &[&str],
    ) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            functional,
            domain: domain.iter().map(|value| value.to_string()).collect(),
            range: range.iter().map(|value| value.to_string()).collect(),
            ..Default::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("女儿", false, &["人物"], &["人物"]));
        registry.declare(&profile("年龄", true, &[], &[]));
        registry
    }

    fn types() -> TypeSystem {
        TypeSystem::from_config(&TypeSystemConfig {
            closure_gate: true,
            aliases: vec![TypeAliasConfig {
                alias: "Person".to_string(),
                canonical: "人物".to_string(),
            }],
            subclasses: Vec::new(),
            disjoint: vec![TypeDisjointConfig {
                left: "人物".to_string(),
                right: "地点".to_string(),
            }],
        })
    }

    fn shapes(registry: &PredicateRegistry, types: &TypeSystem) -> ShapeSet {
        ShapeSet::from_schema(&[], registry, types)
    }

    fn facts() -> Vec<PrologFact> {
        vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            entity("邓紫棋", "Person", &[("name", "邓紫棋")]),
            entity("李明", "Person", &[("name", "李明")]),
        ]
    }

    #[test]
    fn parses_possession_questions_and_reverse_who_questions() {
        let parsed = RelationAcquisition::parse_question("姚明的女儿是谁？").unwrap();
        assert_eq!(parsed.subject, "姚明");
        assert_eq!(parsed.predicate, "女儿");
        assert_eq!(parsed.form, QuestionForm::Who);

        let reverse = RelationAcquisition::parse_question("谁是姚明的女儿").unwrap();
        assert_eq!(reverse.subject, "姚明");
        assert_eq!(reverse.predicate, "女儿");

        assert!(RelationAcquisition::parse_question("姚明的女儿").is_none());
    }

    #[test]
    fn question_opens_goal_with_range_expectation() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("姚明的女儿是谁？", &facts(), &registry, &types, &shapes);
        let GoalOpen::Opened { goal } = opening else {
            panic!("应打开关系目标：{:?}", opening);
        };
        assert_eq!(goal.subject, "姚明");
        assert_eq!(goal.predicate, "女儿");
        assert_eq!(goal.expectation.entity_types, vec!["人物".to_string()]);
        assert!(!goal.expectation.functional);
        assert!(goal.question.contains("谁"));
        assert!(acquisition.has_pending_goal());
    }

    #[test]
    fn short_answer_is_proposed_and_confirmation_promotes_to_overlay() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);

        let outcome = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Provide);
        assert_eq!(outcome.status, Some(AcquisitionStatus::Proposed));
        assert!(acquisition.overlay().is_empty(), "候选不应直接进入覆盖层");

        let confirmed = acquisition.confirm(None);
        assert_eq!(confirmed.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        assert_eq!(overlay.values("姚明", "女儿").len(), 1);
        assert_eq!(overlay.values("姚明", "女儿")[0].object, "叶莉");

        let answered = acquisition
            .answer_from_overlay("姚明的女儿是谁？", &registry)
            .expect("覆盖层应能回答已确认的关系");
        assert_eq!(answered.values, vec!["叶莉".to_string()]);
        assert!(answered.render_text().contains("未写入知识图谱"));
    }

    #[test]
    fn correction_refutes_previous_candidate_and_replaces_it() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let first = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let first_id = first.claim_id.unwrap();

        let corrected =
            acquisition.interpret_answer("不是叶莉，是邓紫棋", &facts, &registry, &types, &shapes);
        assert_eq!(corrected.act, AnswerAct::Correct);
        assert_eq!(
            acquisition.claim(&first_id).unwrap().status,
            AcquisitionStatus::Refuted
        );
        assert_eq!(corrected.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        let values: Vec<&str> = overlay
            .values("姚明", "女儿")
            .iter()
            .map(|row| row.object.as_str())
            .collect();
        assert_eq!(values, vec!["邓紫棋"]);
    }

    #[test]
    fn unknown_answer_does_not_write_overlay() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("我不知道", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Unknown);
        assert!(acquisition.overlay().is_empty());
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn two_independent_answers_promote_same_value() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let first = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(first.status, Some(AcquisitionStatus::Proposed));
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let second = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(second.status, Some(AcquisitionStatus::Confirmed));
        assert_eq!(acquisition.overlay().rows.len(), 1);
    }

    #[test]
    fn functional_slot_conflict_is_disclosed_and_user_can_adjudicate() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let goal = RelationGoal {
            id: "qg_test".to_string(),
            subject: "姚明".to_string(),
            predicate: "年龄".to_string(),
            surface: "年龄".to_string(),
            expectation: Expectation {
                entity_types: Vec::new(),
                datatype: None,
                functional: true,
                inverse: false,
                required: false,
                value_shape: ValueShape::Infer,
            },
            origin: GoalOrigin::UserRequest,
            form: QuestionForm::Value,
            question: "姚明的年龄是多少？".to_string(),
            turn: 1,
            asked: 1,
        };
        acquisition.active_goal = Some(goal.clone());
        acquisition.turn = 1;
        let first = acquisition.interpret_answer("18", &facts, &registry, &types, &shapes);
        assert_eq!(first.status, Some(AcquisitionStatus::Proposed));
        acquisition.active_goal = Some(goal);
        acquisition.turn += 1;
        let second = acquisition.interpret_answer("19", &facts, &registry, &types, &shapes);
        assert_eq!(second.status, Some(AcquisitionStatus::Conflicted));
        assert!(second.message.contains("冲突"));

        let adjudicated = acquisition.confirm(second.claim_id.as_deref());
        assert_eq!(adjudicated.status, Some(AcquisitionStatus::Confirmed));
        assert_eq!(
            acquisition
                .claims()
                .filter(|claim| claim.claim.subject == "姚明" && claim.claim.predicate == "年龄")
                .filter(|claim| claim.status == AcquisitionStatus::Refuted)
                .count(),
            1
        );
    }

    #[test]
    fn induced_pattern_activates_after_two_confirmations_and_scans_text() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        for (subject, value) in [("姚明", "叶莉"), ("李明", "邓紫棋")] {
            let question = format!("{}的女儿是谁？", subject);
            let _ = acquisition.start_question(&question, &facts, &registry, &types, &shapes);
            let _ = acquisition.interpret_answer(value, &facts, &registry, &types, &shapes);
            let _ = acquisition.confirm(None);
        }
        assert_eq!(acquisition.patterns().count(), 1);
        assert!(acquisition.patterns().next().unwrap().active);

        let scan = acquisition.scan_text("王强的女儿是赵敏。", &registry);
        assert_eq!(scan.claims.len(), 1);
        assert_eq!(scan.claims[0].subject, "王强");
        assert_eq!(scan.claims[0].object.text(), "赵敏");
    }

    #[test]
    fn adopted_claims_remain_visible_in_overlay_and_snapshot_roundtrips() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let id = outcome.claim_id.unwrap();
        let _ = acquisition.confirm(Some(&id));
        assert!(acquisition.mark_admitted(&id));
        assert_eq!(
            acquisition.claim(&id).unwrap().status,
            AcquisitionStatus::Admitted
        );
        assert!(!acquisition.overlay().is_empty());

        let snapshot = acquisition.snapshot();
        let mut restored = RelationAcquisition::new();
        assert!(restored.restore(&snapshot));
        assert_eq!(restored.digest(), acquisition.digest());
        assert_eq!(restored.overlay().rows, acquisition.overlay().rows);
    }

    #[test]
    fn already_known_graph_value_does_not_open_goal() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            relation("姚明", "女儿", "叶莉"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        assert!(matches!(opening, GoalOpen::AlreadyKnown { .. }));
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn denial_without_replacement_is_not_mistaken_for_correction() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let _ = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("不是叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Deny);
        assert!(acquisition.overlay().is_empty());
        assert!(acquisition
            .claims()
            .all(|claim| claim.status == AcquisitionStatus::Refuted));
    }

    #[test]
    fn valid_value_starting_with_bu_is_not_treated_as_denial() {
        let classified = classify_answer("不丹");
        assert_eq!(classified.act, AnswerAct::Provide);
        assert_eq!(classified.values, vec!["不丹".to_string()]);
    }

    #[test]
    fn shape_gap_goal_records_its_origin() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("姚明", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_shape_gap("姚明", "女儿", &facts, &registry, &types, &shapes);
        let GoalOpen::Opened { goal } = opening else {
            panic!("形状缺口应打开目标：{:?}", opening);
        };
        assert_eq!(goal.origin, GoalOrigin::ShapeRepair);
    }

    #[test]
    fn free_assertion_is_proposed_and_confirmation_reaches_overlay() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("任职于", false, &["人物"], &["机构"]));
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("张三", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let outcome = acquisition.record_assertion(
            Claim::entity("张三", "任职于", "清华大学"),
            AcquisitionSource::UserAssertion,
            "张三在清华大学工作。",
            "断言阅读",
            &facts,
            &registry,
            &types,
            &shapes,
        );
        assert_eq!(outcome.status, Some(AcquisitionStatus::Proposed));
        assert!(acquisition.overlay().is_empty());

        let id = outcome.claim_id.unwrap();
        let confirmed = acquisition.confirm(Some(&id));
        assert_eq!(confirmed.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        assert_eq!(overlay.values("张三", "任职于").len(), 1);
        assert_eq!(overlay.values("张三", "任职于")[0].object, "清华大学");
    }

    #[test]
    fn where_work_question_parses_and_answers_from_overlay() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("任职于", false, &["人物"], &["机构"]));
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("张三", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let outcome = acquisition.record_assertion(
            Claim::entity("张三", "任职于", "清华大学"),
            AcquisitionSource::UserAssertion,
            "张三在清华大学工作。",
            "断言阅读",
            &facts,
            &registry,
            &types,
            &shapes,
        );
        let id = outcome.claim_id.unwrap();
        let _ = acquisition.confirm(Some(&id));

        let parsed = RelationAcquisition::parse_question("张三在哪工作？").unwrap();
        assert_eq!(parsed.subject, "张三");
        assert_eq!(parsed.predicate, "任职于");
        assert_eq!(parsed.form, QuestionForm::Where);
        let answer = acquisition
            .answer_from_overlay("张三在哪工作？", &registry)
            .expect("覆盖层应能回答任职问句");
        assert_eq!(answer.values, vec!["清华大学".to_string()]);
    }

    #[test]
    fn inverse_question_is_normalized_to_canonical_direction() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &["人物"], &["人物"]));
        registry.register_inverse("师傅", "徒弟", "测试");
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            relation("姚明", "师傅", "叶莉"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("叶莉的徒弟是谁？", &facts, &registry, &types, &shapes);
        assert!(matches!(opening, GoalOpen::AlreadyKnown { .. }));
    }
}

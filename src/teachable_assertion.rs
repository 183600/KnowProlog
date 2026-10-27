//! TAR：Teachable Assertion Reader（可教断言阅读器）。
//!
//! 自由陈述默认不是知识库写入，而是一次「可确认的语义假设」：
//!   1. 守卫：问句、否定、条件、模态、传闻句不进入自动学习；
//!   2. 提及：主语/宾语按图谱类型、后缀、人名与谓词值域形成类型证据；
//!   3. 关系：复用 `parse_lessons` 的命题解析与 `GrammarExtractor` 的三元组；
//!   4. 三值决策：提议确认 / 追问实体类型 / 忽略（交给既有管线）；
//!   5. 记账：确认后由 QRA 会话覆盖层保存，仍不写知识图谱。
//!
//! 本模块只维护「待确认断言 + 提及卡」这一层跨轮状态；关系账本与形状闸门
//! 仍由 `relation_acquisition` / `shape_gate` 负责，避免出现第二套事实源。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chinese_reasoning_nlu::{parse_lessons, LessonKind};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::grammar_extractor::GrammarTriple;
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::shape_gate::{Claim, ClaimObject};
use crate::type_system::TypeSystem;

pub const TAR_DIALECT: &str = "knowprolog-tar/1";
pub const MAX_MENTIONS: usize = 64;
pub const MAX_EVIDENCE: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionStatus {
    Proposed,
    Confirmed,
}

impl MentionStatus {
    pub fn label(self) -> &'static str {
        match self {
            MentionStatus::Proposed => "待确认",
            MentionStatus::Confirmed => "已确认",
        }
    }
}

/// 会话级提及卡：一个词面「可能是哪类实体」以及它来自哪些证据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MentionCard {
    pub surface: String,
    #[serde(default)]
    pub types: BTreeSet<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub status: MentionStatus,
    pub first_turn: usize,
    pub last_turn: usize,
}

impl MentionCard {
    pub fn render_text(&self) -> String {
        let types = if self.types.is_empty() {
            "（类型未知）".to_string()
        } else {
            self.types.iter().cloned().collect::<Vec<_>>().join(" / ")
        };
        format!(
            "- 「{}」：{}（{}）· 证据：{}",
            self.surface,
            types,
            self.status.label(),
            if self.evidence.is_empty() {
                "（无）".to_string()
            } else {
                self.evidence.join("；")
            }
        )
    }
}

/// 一条待确认的断言帧（已归一谓词方向）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssertionFrame {
    pub subject: String,
    pub predicate: String,
    pub predicate_surface: String,
    pub object: ClaimObject,
    #[serde(default)]
    pub subject_types: Vec<String>,
    #[serde(default)]
    pub object_types: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub rule: String,
    pub source_text: String,
}

impl AssertionFrame {
    pub fn to_claim(&self) -> Claim {
        Claim {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            object: self.object.clone(),
        }
    }

    pub fn render_text(&self) -> String {
        let subject = render_typed(&self.subject, &self.subject_types);
        let object = match &self.object {
            ClaimObject::Entity(value) => render_typed(value, &self.object_types),
            ClaimObject::Literal(value) => value.clone(),
        };
        format!("{} —{}→ {}", subject, self.predicate, object)
    }
}

fn render_typed(value: &str, types: &[String]) -> String {
    if types.is_empty() {
        value.to_string()
    } else {
        format!("{}（{}）", value, types.join(" / "))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum PendingStage {
    ClarifyType { mention: String },
    ConfirmClaim { claim_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingAssertion {
    pub frame: AssertionFrame,
    pub stage: PendingStage,
    pub turn: usize,
}

/// 阅读结论。`Ignore` 表示让位给既有管线，不改变任何状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TarDecision {
    Ignore,
    Propose {
        frame: AssertionFrame,
    },
    ClarifyType {
        frame: AssertionFrame,
        mention: String,
        question: String,
    },
    TypeLearned {
        mention: String,
        type_name: String,
    },
    AlreadyKnown {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionReading {
    pub decision: TarDecision,
    pub message: String,
    pub notes: Vec<String>,
}

impl AssertionReading {
    fn ignore() -> Self {
        Self {
            decision: TarDecision::Ignore,
            message: String::new(),
            notes: Vec::new(),
        }
    }
}

/// 待确认断言在下一轮的落点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TarResponse {
    None,
    Confirm { claim_id: String },
    Deny { claim_id: String },
    TypeResolved {
        frame: AssertionFrame,
        mention: String,
        type_name: String,
    },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TarSnapshot {
    pub dialect: String,
    pub turn: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<PendingAssertion>,
    #[serde(default)]
    pub mentions: BTreeMap<String, MentionCard>,
}

/// 可教断言阅读器：只保存跨轮状态，不持有任何图谱句柄。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeachableAssertionReader {
    turn: usize,
    pending: Option<PendingAssertion>,
    mentions: BTreeMap<String, MentionCard>,
}

impl TeachableAssertionReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn_count(&self) -> usize {
        self.turn
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn pending(&self) -> Option<&PendingAssertion> {
        self.pending.as_ref()
    }

    pub fn pending_claim_id(&self) -> Option<&str> {
        match self.pending.as_ref().map(|pending| &pending.stage) {
            Some(PendingStage::ConfirmClaim { claim_id }) => Some(claim_id.as_str()),
            _ => None,
        }
    }

    pub fn clear_pending(&mut self) {
        self.pending = None;
    }

    pub fn mentions(&self) -> impl Iterator<Item = &MentionCard> {
        self.mentions.values()
    }

    pub fn mention(&self, surface: &str) -> Option<&MentionCard> {
        self.mentions.get(surface.trim())
    }

    pub fn snapshot(&self) -> TarSnapshot {
        TarSnapshot {
            dialect: TAR_DIALECT.to_string(),
            turn: self.turn,
            pending: self.pending.clone(),
            mentions: self.mentions.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &TarSnapshot) -> bool {
        if snapshot.dialect != TAR_DIALECT {
            return false;
        }
        self.turn = snapshot.turn;
        self.pending = snapshot.pending.clone();
        self.mentions = snapshot.mentions.clone();
        true
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self.snapshot()).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
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
        let snapshot: TarSnapshot = serde_json::from_str(&text).ok()?;
        let mut reader = Self::new();
        reader.restore(&snapshot).then_some(reader)
    }

    /// 读取一句自由陈述。只有高精度候选才返回 `Propose`/`ClarifyType`；
    /// 其余情况返回 `Ignore`，由调用方继续原有蕴含/溯因管线。
    pub fn read_statement(
        &mut self,
        text: &str,
        grammar: &[GrammarTriple],
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> AssertionReading {
        self.turn += 1;
        let cleaned = clean_sentence(text);
        if cleaned.is_empty() || guard_reason(&cleaned).is_some() {
            return AssertionReading::ignore();
        }

        if let Some((subject, object, type_name)) = split_type_assertion(&cleaned, types) {
            if is_pronoun(&subject) {
                return AssertionReading::ignore();
            }
            let evidence = format!("用户断言「{}是{}」", subject, object);
            self.note_mention(&subject, &[type_name.clone()], &evidence, true);
            return AssertionReading {
                decision: TarDecision::TypeLearned {
                    mention: subject.clone(),
                    type_name: type_name.clone(),
                },
                message: format!(
                    "已记下：「{}」的类型是{}（提及卡，仅本会话；未写入知识图谱）。",
                    subject, type_name
                ),
                notes: vec![evidence],
            };
        }

        let mut candidates = extract_raw(&cleaned, registry);
        for triple in grammar {
            let object = triple.object.trim();
            if triple.subject.trim().is_empty() || object.is_empty() {
                continue;
            }
            candidates.push(RawAssertion {
                subject: triple.subject.trim().to_string(),
                predicate_surface: triple.predicate.trim().to_string(),
                object: object.to_string(),
                literal: looks_like_literal(object),
                evidence: "文法抽取（DCG）".to_string(),
                rule: "grammar",
                confidence: 78,
            });
        }
        if candidates.is_empty() {
            return AssertionReading::ignore();
        }
        candidates.sort_by(|left, right| {
            right
                .confidence
                .cmp(&left.confidence)
                .then_with(|| left.rule.cmp(right.rule))
                .then_with(|| left.subject.cmp(&right.subject))
                .then_with(|| left.predicate_surface.cmp(&right.predicate_surface))
        });
        candidates.dedup_by(|left, right| {
            left.subject == right.subject
                && left.predicate_surface == right.predicate_surface
                && left.object == right.object
        });

        let raw = candidates.remove(0);
        let resolution = registry.resolve(&raw.predicate_surface);
        let canonical = resolution.canonical.trim().to_string();
        if canonical.is_empty() || !registry.is_known(&canonical) {
            return AssertionReading::ignore();
        }

        let (subject, object, literal) = if resolution.inverted && !raw.literal {
            (raw.object.clone(), raw.subject.clone(), false)
        } else {
            (raw.subject.clone(), raw.object.clone(), raw.literal)
        };
        if is_pronoun(&subject) || subject.is_empty() || object.is_empty() {
            return AssertionReading::ignore();
        }
        if !literal && graph_supports(facts, registry, &subject, &canonical, &object) {
            return AssertionReading {
                decision: TarDecision::AlreadyKnown {
                    message: format!(
                        "知识图谱中已有：{} —{}→ {}（无需再记入会话）。",
                        subject, canonical, object
                    ),
                },
                message: format!(
                    "知识图谱中已有：{} —{}→ {}（无需再记入会话）。",
                    subject, canonical, object
                ),
                notes: vec!["图谱已有同向事实".to_string()],
            };
        }

        let subject_types = known_types(facts, &subject, types);
        let mut object_types = if literal {
            Vec::new()
        } else {
            known_types(facts, &object, types)
        };
        let mut type_evidence = if !object_types.is_empty() {
            TypeEvidence::Known
        } else {
            TypeEvidence::None
        };
        if !literal && object_types.is_empty() {
            if let Some(suggested) = surface_type(&object) {
                object_types.push(types.canonical(suggested));
                type_evidence = TypeEvidence::Suffix;
            }
        }
        if !literal && object_types.is_empty() {
            if let Some(card) = self.mentions.get(&object) {
                if card.status == MentionStatus::Confirmed && !card.types.is_empty() {
                    object_types.extend(card.types.iter().cloned());
                    type_evidence = TypeEvidence::Known;
                }
            }
        }
        if !literal && object_types.is_empty() {
            let range = expectation_types(registry, &canonical, false, types);
            if !range.is_empty() {
                object_types.extend(range);
                type_evidence = TypeEvidence::Range;
            }
        }
        object_types.sort();
        object_types.dedup();
        object_types.retain(|value| !value.is_empty());

        let mut notes = vec![format!("规则：{}", raw.evidence)];
        if !subject_types.is_empty() {
            notes.push(format!("主语类型：{}", subject_types.join(" / ")));
        }
        if !object_types.is_empty() {
            notes.push(format!("宾语类型：{}", object_types.join(" / ")));
        }
        self.note_mention(&subject, &subject_types, &raw.evidence, !subject_types.is_empty());
        if !literal {
            self.note_mention(
                &object,
                &object_types,
                &raw.evidence,
                type_evidence == TypeEvidence::Known,
            );
        }

        let frame = AssertionFrame {
            subject,
            predicate: canonical.clone(),
            predicate_surface: raw.predicate_surface.clone(),
            object: if literal {
                ClaimObject::Literal(object.clone())
            } else {
                ClaimObject::Entity(object.clone())
            },
            subject_types: subject_types.clone(),
            object_types: object_types.clone(),
            evidence: notes.clone(),
            rule: raw.rule.to_string(),
            source_text: cleaned.clone(),
        };

        if !literal
            && type_evidence != TypeEvidence::Known
            && type_evidence != TypeEvidence::Suffix
        {
            let question = format!(
                "我还不确定「{}」是哪类实体：{}。\n如果你确认它是{}，我就按「{}」记下。",
                object,
                type_options(&object_types),
                type_options(&object_types),
                frame.render_text()
            );
            self.pending = Some(PendingAssertion {
                frame: frame.clone(),
                stage: PendingStage::ClarifyType {
                    mention: object.clone(),
                },
                turn: self.turn,
            });
            return AssertionReading {
                decision: TarDecision::ClarifyType {
                    frame,
                    mention: object,
                    question: question.clone(),
                },
                message: question,
                notes,
            };
        }

        let message = format!(
            "我理解为：{}。要记入本轮会话吗？（对/不对）",
            frame.render_text()
        );
        AssertionReading {
            decision: TarDecision::Propose { frame },
            message,
            notes,
        }
    }

    /// 处理上一轮待确认断言。无法归入确认/否认/类型回答的输入会清空待确认，
    /// 返回 `None` 让调用方继续常规管线。
    pub fn respond(&mut self, input: &str, types: &TypeSystem) -> TarResponse {
        let Some(pending) = self.pending.clone() else {
            return TarResponse::None;
        };
        match pending.stage {
            PendingStage::ClarifyType { mention } => match parse_type_reply(input, types) {
                TypeReply::Type(type_name) => {
                    self.note_mention(
                        &mention,
                        &[type_name.clone()],
                        &format!("用户确认类型：{}", clean_sentence(input)),
                        true,
                    );
                    let mut frame = pending.frame;
                    frame.object_types = vec![type_name.clone()];
                    frame
                        .evidence
                        .push(format!("用户确认「{}」为{}", mention, type_name));
                    self.pending = None;
                    TarResponse::TypeResolved {
                        frame,
                        mention,
                        type_name,
                    }
                }
                TypeReply::Cancel => {
                    self.pending = None;
                    TarResponse::Cancelled
                }
                TypeReply::Unrelated => {
                    self.pending = None;
                    TarResponse::None
                }
            },
            PendingStage::ConfirmClaim { claim_id } => {
                if is_confirmation(input) {
                    self.pending = None;
                    TarResponse::Confirm { claim_id }
                } else if is_denial(input) {
                    self.pending = None;
                    TarResponse::Deny { claim_id }
                } else {
                    self.pending = None;
                    TarResponse::None
                }
            }
        }
    }

    /// 调用方把断言写入 QRA 后，用 claim id 挂起下一轮确认。
    pub fn mark_proposed(&mut self, frame: AssertionFrame, claim_id: Option<String>) {
        self.pending = claim_id.map(|claim_id| PendingAssertion {
            frame,
            stage: PendingStage::ConfirmClaim { claim_id },
            turn: self.turn,
        });
    }

    pub fn report(&self) -> String {
        let mut lines = vec![format!(
            "可教断言阅读器（TAR v{}）：轮次 {}，提及卡 {} 张，待确认 {}。",
            TAR_DIALECT.trim_start_matches("knowprolog-tar/"),
            self.turn,
            self.mentions.len(),
            if self.pending.is_some() { "1 条" } else { "无" }
        )];
        if let Some(pending) = &self.pending {
            match &pending.stage {
                PendingStage::ClarifyType { mention } => lines.push(format!(
                    "待补类型：「{}」· {}",
                    mention,
                    pending.frame.render_text()
                )),
                PendingStage::ConfirmClaim { claim_id } => lines.push(format!(
                    "待确认断言：{} · 账本 {}",
                    pending.frame.render_text(),
                    claim_id
                )),
            }
        } else {
            lines.push("待确认断言：（无；直接输入自由陈述句即可开始）。".to_string());
        }
        if self.mentions.is_empty() {
            lines.push("提及卡：（空）".to_string());
        } else {
            lines.push("提及卡：".to_string());
            for card in self.mentions.values() {
                lines.push(card.render_text());
            }
        }
        lines.push(format!("状态摘要：{}", self.digest()));
        lines.join("\n")
    }

    fn note_mention(
        &mut self,
        surface: &str,
        types: &[String],
        evidence: &str,
        confirmed: bool,
    ) {
        let surface = surface.trim();
        if surface.is_empty() {
            return;
        }
        if self.mentions.len() >= MAX_MENTIONS && !self.mentions.contains_key(surface) {
            if let Some(oldest) = self
                .mentions
                .iter()
                .min_by_key(|(_, card)| (card.last_turn, card.first_turn))
                .map(|(key, _)| key.clone())
            {
                self.mentions.remove(&oldest);
            }
        }
        let turn = self.turn;
        let card = self
            .mentions
            .entry(surface.to_string())
            .or_insert_with(|| MentionCard {
                surface: surface.to_string(),
                types: BTreeSet::new(),
                evidence: Vec::new(),
                status: MentionStatus::Proposed,
                first_turn: turn,
                last_turn: turn,
            });
        for type_name in types {
            if !type_name.trim().is_empty() {
                card.types.insert(type_name.trim().to_string());
            }
        }
        if confirmed {
            card.status = MentionStatus::Confirmed;
        }
        if !evidence.trim().is_empty() && !card.evidence.iter().any(|item| item == evidence) {
            card.evidence.push(evidence.trim().to_string());
            if card.evidence.len() > MAX_EVIDENCE {
                let drain = card.evidence.len() - MAX_EVIDENCE;
                card.evidence.drain(0..drain);
            }
        }
        card.last_turn = turn;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeEvidence {
    Known,
    Suffix,
    Range,
    None,
}

#[derive(Debug, Clone)]
struct RawAssertion {
    subject: String,
    predicate_surface: String,
    object: String,
    literal: bool,
    evidence: String,
    rule: &'static str,
    confidence: u8,
}

fn extract_raw(text: &str, registry: &PredicateRegistry) -> Vec<RawAssertion> {
    let mut out = Vec::new();

    if let Some((subject, place)) = split_at_work(text) {
        out.push(RawAssertion {
            subject,
            predicate_surface: "任职于".to_string(),
            object: place,
            literal: false,
            evidence: "「在…工作」句式".to_string(),
            rule: "at_work",
            confidence: 90,
        });
    }

    for lesson in parse_lessons(text).lessons {
        if lesson.kind != LessonKind::DeterministicFact
            || !lesson.head.is_ground()
            || lesson.head.terms.len() != 2
        {
            continue;
        }
        let predicate = lesson.head.predicate.trim();
        if predicate.is_empty() || predicate == "是" {
            continue;
        }
        let subject = lesson.head.terms[0].name().trim();
        let object = lesson.head.terms[1].name().trim();
        if subject.is_empty() || object.is_empty() {
            continue;
        }
        let resolution = registry.resolve(predicate);
        if !resolution.known && !registry.is_known(predicate) {
            continue;
        }
        out.push(RawAssertion {
            subject: subject.to_string(),
            predicate_surface: predicate.to_string(),
            object: object.to_string(),
            literal: looks_like_literal(object),
            evidence: format!("命题解析（{}）", predicate),
            rule: "proposition",
            confidence: 84,
        });
    }

    out.sort_by(|left, right| {
        right
            .confidence
            .cmp(&left.confidence)
            .then_with(|| left.subject.cmp(&right.subject))
            .then_with(|| left.predicate_surface.cmp(&right.predicate_surface))
    });
    out.dedup_by(|left, right| {
        left.subject == right.subject
            && left.predicate_surface == right.predicate_surface
            && left.object == right.object
    });
    out
}

fn split_at_work(text: &str) -> Option<(String, String)> {
    let suffix = ["工作", "上班", "任职", "就职", "供职"]
        .iter()
        .find(|suffix| text.ends_with(*suffix))?;
    let body = text[..text.len() - suffix.len()]
        .trim_end_matches(['了', '过'])
        .trim();
    let (subject, place) = body.rsplit_once('在')?;
    let subject = subject.trim();
    let place = place.trim();
    if subject.is_empty() || place.is_empty() || place.contains('，') || place.contains(',') {
        return None;
    }
    Some((subject.to_string(), place.to_string()))
}

fn split_type_assertion(text: &str, types: &TypeSystem) -> Option<(String, String, String)> {
    let (subject, object) = text.split_once('是')?;
    let subject = subject.trim();
    let object = object.trim();
    if subject.is_empty() || object.is_empty() || subject.contains('的') || object.contains('的') {
        return None;
    }
    let type_name = parse_type_word(object, types)?;
    Some((subject.to_string(), object.to_string(), type_name))
}

fn parse_type_word(word: &str, types: &TypeSystem) -> Option<String> {
    let text = clean_sentence(word);
    if text.is_empty() {
        return None;
    }
    let base = if text.ends_with("公司")
        || text.ends_with("集团")
        || text.ends_with("大学")
        || text.ends_with("学院")
        || text.ends_with("学校")
        || text.ends_with("中学")
        || text.ends_with("小学")
        || text.ends_with("医院")
        || text.ends_with("银行")
        || text.ends_with("研究所")
        || text.ends_with("研究院")
        || text.ends_with("机构")
        || text.ends_with("组织")
        || text.ends_with("企业")
        || text.ends_with("单位")
        || text.ends_with("工作室")
        || text.ends_with("协会")
        || text.ends_with("委员会")
        || text.ends_with("基金会")
    {
        "机构"
    } else if text.ends_with("省")
        || text.ends_with("市")
        || text.ends_with("县")
        || text.ends_with("区")
        || text.ends_with("镇")
        || text.ends_with("村")
        || text.ends_with("乡")
        || text.ends_with("国家")
        || text.ends_with("国")
    {
        "地点"
    } else if matches!(
        text.as_str(),
        "人物"
            | "人"
            | "人类"
            | "个人"
            | "学生"
            | "老师"
            | "教师"
            | "教授"
            | "医生"
            | "律师"
            | "演员"
            | "歌手"
            | "导演"
            | "作家"
    ) {
        "人物"
    } else if matches!(text.as_str(), "实体" | "事物" | "对象") {
        "实体"
    } else {
        return None;
    };
    let canonical = types.canonical(base);
    (!canonical.is_empty()).then_some(canonical)
}

fn surface_type(surface: &str) -> Option<&'static str> {
    let text = surface.trim();
    if text.is_empty() {
        return None;
    }
    if text.ends_with("先生")
        || text.ends_with("女士")
        || text.ends_with("老师")
        || text.ends_with("教授")
        || text.ends_with("医生")
        || text.ends_with("律师")
        || text.ends_with("演员")
        || text.ends_with("歌手")
        || text.ends_with("导演")
        || text.ends_with("作家")
    {
        return Some("人物");
    }
    if looks_like_person_name(text) {
        return Some("人物");
    }
    const ORG_SUFFIXES: &[&str] = &[
        "公司", "集团", "大学", "学院", "学校", "中学", "小学", "医院", "银行", "研究所",
        "研究院", "机构", "组织", "企业", "单位", "工作室", "协会", "委员会", "基金会", "政府",
        "法院", "出版社", "报社", "电视台", "机场", "车站",
    ];
    if ORG_SUFFIXES.iter().any(|suffix| text.ends_with(suffix)) {
        return Some("机构");
    }
    const LOCATION_SUFFIXES: &[&str] = &[
        "省", "市", "县", "区", "镇", "村", "乡", "街道", "路", "街", "山", "河", "湖", "海",
        "岛", "湾", "广场", "公园", "国家",
    ];
    if LOCATION_SUFFIXES.iter().any(|suffix| text.ends_with(suffix)) {
        return Some("地点");
    }
    None
}

fn looks_like_person_name(text: &str) -> bool {
    const SURNAMES: &str = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公";
    let chars: Vec<char> = text.chars().collect();
    (2..=4).contains(&chars.len())
        && SURNAMES.contains(chars[0])
        && chars
            .iter()
            .all(|ch| ch.is_alphabetic() && !matches!(ch, '的' | '是' | '在' | '和'))
}

fn known_types(facts: &[PrologFact], surface: &str, types: &TypeSystem) -> Vec<String> {
    let mut found = BTreeSet::new();
    for fact in facts {
        if let PrologFact::Entity {
            name, entity_type, ..
        } = fact
        {
            if name == surface && !entity_type.trim().is_empty() {
                found.insert(types.canonical(entity_type));
            }
        }
    }
    found.into_iter().filter(|value| !value.is_empty()).collect()
}

fn expectation_types(
    registry: &PredicateRegistry,
    canonical: &str,
    inverted: bool,
    types: &TypeSystem,
) -> Vec<String> {
    let Some(profile) = registry.profile(canonical) else {
        return Vec::new();
    };
    let source = if inverted { &profile.domain } else { &profile.range };
    source
        .iter()
        .map(|value| types.canonical(value))
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn graph_supports(
    facts: &[PrologFact],
    registry: &PredicateRegistry,
    subject: &str,
    canonical: &str,
    object: &str,
) -> bool {
    for fact in facts {
        match fact {
            PrologFact::Relation { from, rel_type, to } => {
                let resolution = registry.resolve(rel_type);
                if resolution.canonical != canonical {
                    continue;
                }
                let (actual_subject, actual_object) = if resolution.inverted {
                    (to.as_str(), from.as_str())
                } else {
                    (from.as_str(), to.as_str())
                };
                if actual_subject == subject && actual_object == object {
                    return true;
                }
            }
            PrologFact::Entity {
                name, properties, ..
            } => {
                if name != subject {
                    continue;
                }
                for (key, value) in properties {
                    if registry.resolve(key).canonical == canonical && value == object {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn looks_like_literal(value: &str) -> bool {
    let text = value.trim();
    if text.is_empty() {
        return false;
    }
    if text.parse::<f64>().is_ok() {
        return true;
    }
    let first = text.chars().next().unwrap_or(' ');
    let numeric_prefix = first.is_ascii_digit()
        || "零〇一二两三四五六七八九十百千万亿".contains(first);
    if numeric_prefix && text.chars().any(|ch| "岁年月日号元块个次分秒%％".contains(ch)) {
        return true;
    }
    matches!(text, "是" | "否" | "true" | "false" | "yes" | "no")
        || crate::shape_gate::datatype_conforms("date", text)
        || crate::shape_gate::datatype_conforms("time", text)
        || crate::shape_gate::datatype_conforms("datetime", text)
}

fn guard_reason(text: &str) -> Option<&'static str> {
    if crate::function_words::contains_question_marker(text) {
        return Some("疑问");
    }
    const NEGATIONS: &[&str] = &["不是", "不在", "没有", "没在", "并非", "从未", "不曾", "未必"];
    if NEGATIONS.iter().any(|word| text.contains(word)) {
        return Some("否定");
    }
    const CONDITIONALS: &[&str] = &["如果", "假如", "假设", "要是", "若是", "只要", "除非", "一旦", "那么", "则"];
    if CONDITIONALS.iter().any(|word| text.contains(word)) {
        return Some("条件");
    }
    const MODALS: &[&str] = &[
        "可能", "也许", "或许", "大概", "应该", "据说", "听说", "传闻", "将会", "将要", "想要",
        "希望", "打算", "似乎", "好像", "未必",
    ];
    if MODALS.iter().any(|word| text.contains(word)) {
        return Some("模态/传闻");
    }
    None
}

fn is_pronoun(surface: &str) -> bool {
    matches!(
        surface.trim(),
        "我" | "你" | "您" | "他" | "她" | "它" | "我们" | "你们" | "他们" | "她们" | "它们"
    )
}

fn clean_sentence(text: &str) -> String {
    text.trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '。' | '！'
                    | '!'
                    | '？'
                    | '?'
                    | '，'
                    | ','
                    | '；'
                    | ';'
                    | '：'
                    | ':'
                    | '“'
                    | '”'
                    | '"'
                    | '\''
                    | '\n'
                    | '\r'
                    | '\t'
                    | ' '
            )
        })
        .to_string()
}

fn type_options(types: &[String]) -> String {
    if types.is_empty() {
        "机构、地点，还是其他".to_string()
    } else {
        format!("{}，还是其他", types.join("、"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TypeReply {
    Type(String),
    Cancel,
    Unrelated,
}

fn parse_type_reply(input: &str, types: &TypeSystem) -> TypeReply {
    let mut text = clean_sentence(input);
    for prefix in ["它是", "他是", "她是", "这是", "类型是", "算是", "是"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim().to_string();
            break;
        }
    }
    if matches!(
        text.as_str(),
        "不确定" | "不知道" | "其他" | "其它" | "算了" | "不" | "不是" | "取消"
    ) {
        return TypeReply::Cancel;
    }
    if let Some(type_name) = parse_type_word(&text, types) {
        return TypeReply::Type(type_name);
    }
    let canonical = types.canonical(&text);
    if !canonical.is_empty()
        && text.chars().count() <= 6
        && types.ancestors(&text).iter().any(|name| name == &text)
    {
        return TypeReply::Type(canonical);
    }
    TypeReply::Unrelated
}

fn is_confirmation(input: &str) -> bool {
    let text = clean_sentence(input);
    matches!(
        text.as_str(),
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
            | "确认"
            | "yes"
            | "y"
            | "ok"
            | "okay"
    ) || text.starts_with("对，")
        || text.starts_with("对,")
        || text.starts_with("确认")
}

fn is_denial(input: &str) -> bool {
    let text = clean_sentence(input);
    matches!(
        text.as_str(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use std::collections::HashMap;

    fn entity(name: &str, etype: &str) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: HashMap::new(),
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "任职于".to_string(),
            aliases: vec!["工作于".to_string()],
            domain: vec!["人物".to_string()],
            range: vec!["机构".to_string()],
            ..Default::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            aliases: vec!["老师".to_string()],
            inverse: Some("徒弟".to_string()),
            domain: vec!["人物".to_string()],
            range: vec!["人物".to_string()],
            ..Default::default()
        });
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

    #[test]
    fn reads_at_work_statement_and_suggests_organization() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let reading = reader.read_statement(
            "张三在清华大学工作。",
            &[],
            &[],
            &registry,
            &types,
        );
        let TarDecision::Propose { frame } = reading.decision else {
            panic!("应提议确认：{:?}", reading);
        };
        assert_eq!(frame.subject, "张三");
        assert_eq!(frame.predicate, "任职于");
        assert_eq!(frame.object.text(), "清华大学");
        assert_eq!(frame.object_types, vec!["机构".to_string()]);
        assert!(reading.message.contains("要记入本轮会话吗"));
        assert!(reader.mention("清华大学").is_some());
    }

    #[test]
    fn asks_type_for_unknown_organization_and_accepts_reply() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let reading = reader.read_statement(
            "李四在字节跳动工作。",
            &[],
            &[],
            &registry,
            &types,
        );
        let TarDecision::ClarifyType { mention, .. } = reading.decision else {
            panic!("应追问类型：{:?}", reading);
        };
        assert_eq!(mention, "字节跳动");
        assert!(reader.has_pending());

        let response = reader.respond("机构", &types);
        let TarResponse::TypeResolved { frame, type_name, .. } = response else {
            panic!("应接受类型回答：{:?}", response);
        };
        assert_eq!(type_name, "机构");
        assert_eq!(frame.object_types, vec!["机构".to_string()]);
        assert_eq!(reader.mention("字节跳动").unwrap().status, MentionStatus::Confirmed);
    }

    #[test]
    fn guards_questions_negations_and_conditionals() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        for text in [
            "张三在清华大学工作吗？",
            "张三不在清华大学工作。",
            "如果张三在清华大学工作，那么李四也在。",
            "张三可能在清华大学工作。",
        ] {
            let reading = reader.read_statement(text, &[], &[], &registry, &types);
            assert_eq!(reading.decision, TarDecision::Ignore, "{}", text);
        }
    }

    #[test]
    fn possession_statement_is_oriented_to_canonical_predicate() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let reading = reader.read_statement(
            "张三是李四的老师。",
            &[],
            &[],
            &registry,
            &types,
        );
        let TarDecision::Propose { frame } = reading.decision else {
            panic!("应提议确认：{:?}", reading);
        };
        assert_eq!(frame.predicate, "师傅");
        assert_eq!(frame.subject, "李四");
        assert_eq!(frame.object.text(), "张三");
    }

    #[test]
    fn type_assertion_updates_mention_card_without_claim() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let reading = reader.read_statement("字节跳动是公司。", &[], &[], &registry, &types);
        let TarDecision::TypeLearned { mention, type_name } = reading.decision else {
            panic!("应学习类型：{:?}", reading);
        };
        assert_eq!(mention, "字节跳动");
        assert_eq!(type_name, "机构");
        assert_eq!(reader.mention("字节跳动").unwrap().status, MentionStatus::Confirmed);
    }

    #[test]
    fn snapshot_roundtrip_keeps_pending_and_mentions() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let reading = reader.read_statement(
            "李四在字节跳动工作。",
            &[],
            &[],
            &registry,
            &types,
        );
        assert!(matches!(reading.decision, TarDecision::ClarifyType { .. }));
        let snapshot = reader.snapshot();
        let mut restored = TeachableAssertionReader::new();
        assert!(restored.restore(&snapshot));
        assert_eq!(restored.digest(), reader.digest());
        assert!(restored.has_pending());
        assert!(restored.mention("字节跳动").is_some());
    }

    #[test]
    fn graph_fact_is_reported_as_already_known() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let facts = vec![PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "工作于".to_string(),
            to: "清华大学".to_string(),
        }];
        let reading = reader.read_statement(
            "张三在清华大学工作。",
            &[],
            &facts,
            &registry,
            &types,
        );
        assert!(matches!(reading.decision, TarDecision::AlreadyKnown { .. }));
    }

    #[test]
    fn grammar_triple_is_used_when_no_text_pattern_matches() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let triple = GrammarTriple {
            predicate: "师傅".to_string(),
            subject: "王五".to_string(),
            object: "赵六".to_string(),
        };
        let reading = reader.read_statement(
            "王五与赵六有师徒关系。",
            &[triple],
            &[],
            &registry,
            &types,
        );
        let TarDecision::Propose { frame } = reading.decision else {
            panic!("应使用文法三元组：{:?}", reading);
        };
        assert_eq!(frame.predicate, "师傅");
        assert_eq!(frame.subject, "王五");
        assert_eq!(frame.object.text(), "赵六");
    }

    #[test]
    fn known_entity_types_are_attached_to_frame() {
        let registry = registry();
        let types = types();
        let mut reader = TeachableAssertionReader::new();
        let facts = vec![entity("张三", "Person")];
        let reading = reader.read_statement(
            "张三在清华大学工作。",
            &[],
            &facts,
            &registry,
            &types,
        );
        let TarDecision::Propose { frame } = reading.decision else {
            panic!("应提议确认：{:?}", reading);
        };
        assert_eq!(frame.subject_types, vec!["人物".to_string()]);
    }
}

//! IRF：导入可逆帧（Import-Reversible Frames）。
//!
//! 文本导入本来只把事实写进图谱，却把「这句话怎么说」丢掉了。本模块把
//! 成功落库的 `(句子, 事实)` 对当作免费标注：把事实的主语/宾语回对齐到
//! 原句，夹在中间的确切词面就是该谓词的触发框架；同一份框架同时服务
//! 「读」「问」「说」：
//!
//! - **读**：框架只能从能被自身匹配器唯一还原的句子里学习（可逆闸门）；
//! - **说**：生成侧用框架把事实说回原句的措辞；
//! - **问**：把宾语/主语槽换成疑问词得到保守问法，先候选、后按见证数激活；
//! - **记**：每条框架保留原句见证与导入回执摘要，撤销导入时一并退场。
//!
//! 设计边界（宁可少学，也不学开放语法）：
//! - 只学字符级确切骨架，不做词性泛化、不做通配；
//! - 一句多事实/主宾重叠/同一最小跨度有歧义时跳过；
//! - 只接受回执里确实写入的事实，冲突/未准入的事实不产生语言记忆；
//! - 语言不是知识：框架不写图谱、不确认事实、不改变真值；
//! - 问法默认需要两条不同见证才激活，避免单例导入把问题路由带偏；
//! - 台账有界（框架数/每帧见证数），落盘复用 [`ContextImage`] 信封。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex, ContextImage};
use crate::identity::{local_name, DEFAULT_BASE_NS};
use crate::predicate_registry::{ImportAction, ImportReceipt};
use crate::KnowledgeItem;

/// 导入帧台账方言；版本不符时拒绝恢复。
pub const IMPORT_FRAMES_DIALECT: &str = "knowprolog-import-frames/1";
/// 台账在 ContextImage 信封中的分节名。
pub const IMPORT_FRAMES_SECTION: &str = "import_frames";
/// 当前格式版本。
pub const IMPORT_FRAMES_VERSION: u32 = 1;
/// 默认最多保留多少条框架。
pub const DEFAULT_MAX_FRAMES: usize = 256;
/// 每条框架默认保留多少条原句见证。
pub const DEFAULT_MAX_OBSERVATIONS: usize = 8;
/// 问法自动激活所需的「不同原句」条数。
pub const DEFAULT_MIN_ASK_SUPPORT: usize = 2;
/// 单次匹配最多收集多少个候选（防止歧义回溯爆炸）。
const MAX_MATCHES: usize = 64;
/// 报告中最多逐条列出的框架数。
const REPORT_FRAME_CAP: usize = 20;

const SLOT_SUBJECT: &str = "subject";
const SLOT_OBJECT: &str = "object";
const META_PREDICATES: [&str; 5] = ["name", "type", "类型", "rdf:type", "标签"];

/// 事实种类：属性（字面量）或关系（实体-实体）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    Attribute,
    Relation,
}

impl FactKind {
    pub fn label(self) -> &'static str {
        match self {
            FactKind::Attribute => "属性",
            FactKind::Relation => "关系",
        }
    }
}

/// 问句焦点：问宾语还是问主语。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameFocus {
    Subject,
    Object,
}

impl FrameFocus {
    pub fn label(self) -> &'static str {
        match self {
            FrameFocus::Subject => "主语",
            FrameFocus::Object => "宾语",
        }
    }
}

/// 陈述框架的一个位置：确切词面或实体槽。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrameAtom {
    Literal { surface: String },
    Subject,
    Object,
}

impl FrameAtom {
    fn literal(surface: impl Into<String>) -> Self {
        FrameAtom::Literal {
            surface: surface.into(),
        }
    }
}

/// 问法框架的一个位置：已知侧槽位、未知侧疑问词或确切词面。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AskAtom {
    Literal { surface: String },
    Slot,
    Question { focus: FrameFocus },
}

/// 一条原句见证。`import_digest` 把语言记忆绑到可撤销的导入回执上。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameObservation {
    pub witness: String,
    #[serde(default)]
    pub import_digest: String,
    #[serde(default)]
    pub turn: u64,
}

/// 一条导入可逆帧。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportFrame {
    pub id: String,
    pub kind: FactKind,
    pub predicate: String,
    pub atoms: Vec<FrameAtom>,
    pub example_subject: String,
    pub example_object: String,
    #[serde(default)]
    pub observations: Vec<FrameObservation>,
    /// 人工显式激活问法（默认仍由见证数阈值决定）。
    #[serde(default)]
    pub ask_activated: bool,
}

impl ImportFrame {
    /// 支持度 = 保留的原句见证条数。
    pub fn support(&self) -> usize {
        self.observations.len()
    }

    /// 不同原句条数（同一句重复导入不抬高问法支持度）。
    pub fn distinct_witnesses(&self) -> usize {
        self.observations
            .iter()
            .map(|observation| observation.witness.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn last_turn(&self) -> u64 {
        self.observations
            .iter()
            .map(|observation| observation.turn)
            .max()
            .unwrap_or(0)
    }

    /// 问法是否可参与路由：人工激活或达到见证数阈值。
    pub fn ask_ready(&self, min_support: usize) -> bool {
        self.ask_activated || self.distinct_witnesses() >= min_support.max(1)
    }

    pub fn latest_witness(&self) -> Option<&str> {
        self.observations
            .iter()
            .max_by_key(|observation| observation.turn)
            .map(|observation| observation.witness.as_str())
    }

    /// 按框架把事实说回人话；槽位为空时拒绝生成。
    pub fn realize(&self, subject: &str, object: &str) -> Option<String> {
        let subject = subject.trim();
        let object = object.trim();
        if subject.is_empty() || object.is_empty() {
            return None;
        }
        let mut output = String::new();
        for atom in &self.atoms {
            match atom {
                FrameAtom::Literal { surface } => output.push_str(surface),
                FrameAtom::Subject => output.push_str(subject),
                FrameAtom::Object => output.push_str(object),
            }
        }
        let output = output.trim().to_string();
        (!output.is_empty()).then_some(output)
    }

    /// 用陈述框架匹配一句话；只有唯一绑定才算可逆。
    pub fn bind_unique(&self, text: &str) -> Option<FrameBinding> {
        let chars: Vec<char> = normalize_text(text).chars().collect();
        let mut bindings = BTreeMap::new();
        let mut matches = Vec::new();
        match_declaration(&self.atoms, &chars, 0, &mut bindings, &mut matches);
        if matches.len() == 1 {
            matches.pop()
        } else {
            None
        }
    }

    /// 由陈述框架派生保守问法（不包含是非问，避免弱命中）。
    pub fn ask_patterns(&self) -> Vec<AskPattern> {
        let mut patterns = Vec::new();
        let object_interrogatives: Vec<String> = match self.kind {
            FactKind::Relation => vec!["谁", "什么人", "哪个人"],
            FactKind::Attribute => {
                vec!["多少", "什么", "几", "多大", "多高", "多重", "多长", "多远"]
            }
        }
        .into_iter()
        .map(str::to_string)
        .collect();
        let subject_interrogatives: Vec<String> = match self.kind {
            FactKind::Relation => vec!["谁", "什么人", "哪个人"],
            FactKind::Attribute => vec!["谁", "什么"],
        }
        .into_iter()
        .map(str::to_string)
        .collect();

        let mut object_atoms = Vec::new();
        let mut object_index = None;
        for (index, atom) in self.atoms.iter().enumerate() {
            match atom {
                FrameAtom::Literal { surface } => object_atoms.push(AskAtom::Literal {
                    surface: surface.clone(),
                }),
                FrameAtom::Subject => object_atoms.push(AskAtom::Slot),
                FrameAtom::Object => {
                    object_index = Some(index);
                    object_atoms.push(AskAtom::Question {
                        focus: FrameFocus::Object,
                    });
                }
            }
        }
        push_unique_ask(
            &mut patterns,
            AskPattern {
                focus: FrameFocus::Object,
                atoms: object_atoms.clone(),
                interrogatives: object_interrogatives.clone(),
            },
        );
        // 属性句的宾语后面常跟单位/量词（226cm、18岁）；问句通常省略单位。
        if self.kind == FactKind::Attribute {
            if let Some(index) = object_index {
                if self.atoms[index + 1..]
                    .iter()
                    .any(|atom| matches!(atom, FrameAtom::Literal { .. }))
                {
                    let mut trimmed = object_atoms;
                    trimmed.truncate(index + 1);
                    push_unique_ask(
                        &mut patterns,
                        AskPattern {
                            focus: FrameFocus::Object,
                            atoms: trimmed,
                            interrogatives: object_interrogatives,
                        },
                    );
                }
            }
        }

        let mut subject_atoms = Vec::new();
        for atom in &self.atoms {
            match atom {
                FrameAtom::Literal { surface } => subject_atoms.push(AskAtom::Literal {
                    surface: surface.clone(),
                }),
                FrameAtom::Subject => subject_atoms.push(AskAtom::Question {
                    focus: FrameFocus::Subject,
                }),
                FrameAtom::Object => subject_atoms.push(AskAtom::Slot),
            }
        }
        push_unique_ask(
            &mut patterns,
            AskPattern {
                focus: FrameFocus::Subject,
                atoms: subject_atoms,
                interrogatives: subject_interrogatives,
            },
        );
        patterns
    }
}

fn push_unique_ask(patterns: &mut Vec<AskPattern>, pattern: AskPattern) {
    if pattern.atoms.is_empty() {
        return;
    }
    if patterns
        .iter()
        .any(|existing| existing.atoms == pattern.atoms)
    {
        return;
    }
    patterns.push(pattern);
}

/// 陈述框架的一次唯一绑定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBinding {
    pub subject: String,
    pub object: String,
}

/// 一条派生问法。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskPattern {
    pub focus: FrameFocus,
    pub atoms: Vec<AskAtom>,
    pub interrogatives: Vec<String>,
}

impl AskPattern {
    fn bind(&self, text: &str) -> Option<AskBinding> {
        let chars: Vec<char> = normalize_text(text).chars().collect();
        let mut bindings = BTreeMap::new();
        let mut matches = Vec::new();
        match_question(
            &self.atoms,
            &self.interrogatives,
            &chars,
            0,
            &mut bindings,
            &mut matches,
        );
        matches.into_iter().next()
    }

    fn literal_len(&self) -> usize {
        self.atoms
            .iter()
            .filter_map(|atom| match atom {
                AskAtom::Literal { surface } => Some(surface.chars().count()),
                _ => None,
            })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AskBinding {
    known: String,
}

/// 一次问句命中：焦点侧已知，另一侧是待查值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionMatch {
    pub frame_id: String,
    pub kind: FactKind,
    pub predicate: String,
    pub focus: FrameFocus,
    pub subject: Option<String>,
    pub object: Option<String>,
    pub support: usize,
    pub witness: Option<String>,
}

/// 带原句见证的框架回答。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameAnswer {
    pub text: String,
    pub witness: String,
    pub frame_id: String,
    pub support: usize,
}

impl FrameAnswer {
    /// 生成侧措辞 + 可核查的原句依据。
    pub fn render_with_witness(&self) -> String {
        let mut text = self.text.trim().to_string();
        if !text.ends_with(&['。', '！', '？', '.', '!', '?'][..]) {
            text.push('。');
        }
        format!("{}（依据：{}）", text, self.witness)
    }
}

/// 一次导入的学习回执（只含计数与框架 id，不含整句报告）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LearnReport {
    pub candidates: usize,
    pub accepted: usize,
    pub learned: usize,
    pub updated: usize,
    pub duplicates: usize,
    pub skipped_unaccepted: usize,
    pub skipped_alignment: usize,
    pub skipped_roundtrip: usize,
    pub frames: Vec<String>,
}

impl LearnReport {
    pub fn changed(&self) -> bool {
        self.learned + self.updated > 0
    }

    pub fn summary(&self) -> String {
        if !self.changed() {
            if self.skipped_unaccepted > 0 {
                return "导入帧：本次没有可学习的已落库事实。".to_string();
            }
            return "导入帧：没有可对齐的事实。".to_string();
        }
        let mut output = format!(
            "导入帧：新学 {} 条，补充见证 {} 条",
            self.learned, self.updated
        );
        if self.duplicates > 0 {
            output.push_str(&format!("，重复见证 {} 条", self.duplicates));
        }
        if self.skipped_alignment > 0 || self.skipped_roundtrip > 0 {
            output.push_str(&format!(
                "，跳过歧义/不可逆 {} 条",
                self.skipped_alignment + self.skipped_roundtrip
            ));
        }
        output
    }
}

/// 台账加载结果：缺失 / 已加载 / 损坏。
#[derive(Debug)]
pub enum FrameLoad {
    Missing,
    Loaded(ImportFrameLedger),
    Damaged(String),
}

/// 有界的导入帧台账（按框架 id 稳定排序）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportFrameLedger {
    pub version: u32,
    pub turn: u64,
    pub max_frames: usize,
    pub max_observations: usize,
    pub min_ask_support: usize,
    pub frames: BTreeMap<String, ImportFrame>,
}

impl Default for ImportFrameLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportFrameLedger {
    pub fn new() -> Self {
        Self {
            version: IMPORT_FRAMES_VERSION,
            turn: 0,
            max_frames: DEFAULT_MAX_FRAMES,
            max_observations: DEFAULT_MAX_OBSERVATIONS,
            min_ask_support: DEFAULT_MIN_ASK_SUPPORT,
            frames: BTreeMap::new(),
        }
    }

    pub fn with_limits(max_frames: usize, max_observations: usize, min_ask_support: usize) -> Self {
        Self {
            version: IMPORT_FRAMES_VERSION,
            turn: 0,
            max_frames: max_frames.max(1),
            max_observations: max_observations.max(1),
            min_ask_support: min_ask_support.max(1),
            frames: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn frames(&self) -> impl Iterator<Item = &ImportFrame> {
        self.frames.values()
    }

    pub fn min_ask_support(&self) -> usize {
        self.min_ask_support
    }

    pub fn set_min_ask_support(&mut self, min_support: usize) {
        self.min_ask_support = min_support.max(1);
    }

    /// 从一次成功导入学习语言框架。
    ///
    /// `items` 是抽取器给出的候选事实；只有回执 `actions` 里确实写入的
    /// 事实才会被对齐。撤销回执摘要会写进每条见证，供 [`Self::retract_import`]
    /// 精确退场。
    pub fn learn_from_import(
        &mut self,
        turn: u64,
        witness: &str,
        items: &[KnowledgeItem],
        receipt: &ImportReceipt,
    ) -> LearnReport {
        self.turn = self.turn.max(turn);
        let mut report = LearnReport::default();
        let witness = normalize_witness(witness);
        if witness.is_empty() {
            return report;
        }
        let import_digest = receipt_digest(receipt);
        let candidates = candidate_facts(items);
        report.candidates = candidates.len();
        if candidates.is_empty() || receipt.actions.is_empty() {
            report.skipped_unaccepted = candidates.len();
            return report;
        }

        for candidate in candidates {
            if !accepted_by_receipt(&candidate, receipt) {
                report.skipped_unaccepted += 1;
                continue;
            }
            report.accepted += 1;
            let Some((atoms, subject, object)) = align_candidate(&witness, &candidate) else {
                report.skipped_alignment += 1;
                continue;
            };
            let id = frame_id(candidate.kind, &candidate.predicate, &atoms);
            let prospective = ImportFrame {
                id: id.clone(),
                kind: candidate.kind,
                predicate: candidate.predicate.clone(),
                atoms: atoms.clone(),
                example_subject: subject.clone(),
                example_object: object.clone(),
                observations: Vec::new(),
                ask_activated: false,
            };
            // 可逆闸门：用同一匹配器把原句读回来，必须唯一还原同一事实。
            let reversible = prospective
                .bind_unique(&witness)
                .map(|binding| binding.subject == subject && binding.object == object)
                .unwrap_or(false)
                && prospective.realize(&subject, &object).as_deref() == Some(witness.as_str());
            if !reversible {
                report.skipped_roundtrip += 1;
                continue;
            }
            let frame = self.frames.entry(id.clone()).or_insert(prospective);
            let duplicate = frame.observations.iter().any(|observation| {
                observation.witness == witness && observation.import_digest == import_digest
            });
            if duplicate {
                report.duplicates += 1;
                continue;
            }
            let existed = !frame.observations.is_empty();
            frame.observations.push(FrameObservation {
                witness: witness.clone(),
                import_digest: import_digest.clone(),
                turn,
            });
            while frame.observations.len() > self.max_observations {
                frame.observations.remove(0);
            }
            if existed {
                report.updated += 1;
            } else {
                report.learned += 1;
            }
            report.frames.push(id);
        }

        self.enforce_capacity();
        report
    }

    /// 撤销一次导入：删除该回执带来的见证；见证耗尽的框架整体退场。
    pub fn retract_import(&mut self, receipt: &ImportReceipt) -> usize {
        let digest = receipt_digest(receipt);
        let mut removed = 0usize;
        let mut empty = Vec::new();
        for (id, frame) in self.frames.iter_mut() {
            // 一次导入在同一框架上只留一条见证；重复导入的相同回执也只
            // 撤销最近的一条，避免把另一次导入的证据一并带走。
            if let Some(index) = frame
                .observations
                .iter()
                .rposition(|observation| observation.import_digest == digest)
            {
                frame.observations.remove(index);
                removed += 1;
            }
            if frame.observations.is_empty() {
                empty.push(id.clone());
            }
        }
        for id in empty {
            self.frames.remove(&id);
        }
        removed
    }

    /// 按当前图谱事实裁剪：示例事实已不在图上的框架退场。
    ///
    /// 供重建缓存/手工删除事实后的生命周期对齐使用；不做数值推理，
    /// 只按 `(主语, 谓词, 宾语)` 的字面存在性判断。
    pub fn retain_facts(&mut self, facts: &[(String, String, String)]) -> usize {
        let known: BTreeSet<(&str, &str, &str)> = facts
            .iter()
            .map(|(subject, predicate, object)| {
                (subject.as_str(), predicate.as_str(), object.as_str())
            })
            .collect();
        let before = self.frames.len();
        self.frames.retain(|_, frame| {
            known.contains(&(
                frame.example_subject.as_str(),
                frame.predicate.as_str(),
                frame.example_object.as_str(),
            ))
        });
        before - self.frames.len()
    }

    /// 生成侧：按已学措辞把事实说回去（谓词必须一致）。
    pub fn realize(&self, subject: &str, predicate: &str, object: &str) -> Option<String> {
        let frame = self
            .frames
            .values()
            .filter(|frame| frame.predicate == predicate && !frame.observations.is_empty())
            .max_by_key(|frame| (frame.support(), frame.last_turn()))?;
        frame.realize(subject, object)
    }

    /// 理解侧：匹配一条已激活问法；并列最高分视为歧义并弃权。
    pub fn match_question(&self, question: &str, min_support: usize) -> Option<QuestionMatch> {
        let normalized = normalize_text(question);
        if normalized.is_empty() {
            return None;
        }
        let mut candidates: Vec<(usize, usize, u64, QuestionMatch)> = Vec::new();
        for frame in self.frames.values() {
            if frame.observations.is_empty() || !frame.ask_ready(min_support) {
                continue;
            }
            for pattern in frame.ask_patterns() {
                let Some(binding) = pattern.bind(&normalized) else {
                    continue;
                };
                let (subject, object) = match pattern.focus {
                    FrameFocus::Object => (Some(binding.known), None),
                    FrameFocus::Subject => (None, Some(binding.known)),
                };
                candidates.push((
                    pattern.literal_len(),
                    frame.support(),
                    frame.last_turn(),
                    QuestionMatch {
                        frame_id: frame.id.clone(),
                        kind: frame.kind,
                        predicate: frame.predicate.clone(),
                        focus: pattern.focus,
                        subject,
                        object,
                        support: frame.support(),
                        witness: frame.latest_witness().map(str::to_string),
                    },
                ));
            }
        }
        candidates.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then(right.1.cmp(&left.1))
                .then(right.2.cmp(&left.2))
                .then(left.3.frame_id.cmp(&right.3.frame_id))
        });
        let best = candidates.first()?;
        if let Some(second) = candidates.get(1) {
            if second.0 == best.0 && second.1 == best.1 && second.3.frame_id != best.3.frame_id {
                return None;
            }
        }
        Some(best.3.clone())
    }

    /// 回答宾语问句：取值由调用方从图谱/生成器索引提供，本模块不写事实。
    pub fn answer<F>(&self, question: &str, min_support: usize, lookup: F) -> Option<FrameAnswer>
    where
        F: Fn(FactKind, &str, &str) -> Vec<String>,
    {
        let matched = self.match_question(question, min_support)?;
        if matched.focus != FrameFocus::Object {
            return None;
        }
        let subject = matched.subject.as_deref()?;
        let mut values = lookup(matched.kind, subject, &matched.predicate);
        values.retain(|value| !value.trim().is_empty());
        values.sort();
        values.dedup();
        if values.is_empty() {
            return None;
        }
        let frame = self.frames.get(&matched.frame_id)?;
        let text = frame.realize(subject, &values.join("、"))?;
        Some(FrameAnswer {
            text,
            witness: matched
                .witness
                .unwrap_or_else(|| "导入帧（原句见证已被容量淘汰）".to_string()),
            frame_id: matched.frame_id,
            support: matched.support,
        })
    }

    /// 人读台账（确定性顺序，最多列出 [`REPORT_FRAME_CAP`] 条）。
    pub fn report(&self) -> String {
        if self.frames.is_empty() {
            return "导入帧：空（用「录入：…」导入陈述句后自动学习）。".to_string();
        }
        let ready = self
            .frames
            .values()
            .filter(|frame| frame.ask_ready(self.min_ask_support))
            .count();
        let mut output = format!(
            "导入帧：{} 条（问法已激活 {} 条，阈值 {} 条不同见证）。\n",
            self.frames.len(),
            ready,
            self.min_ask_support
        );
        for (index, frame) in self.frames.values().take(REPORT_FRAME_CAP).enumerate() {
            let example = frame
                .realize(&frame.example_subject, &frame.example_object)
                .unwrap_or_default();
            output.push_str(&format!(
                "{}. [{}] {}（{}，支持 {} 条{}）\n   说法：{}\n",
                index + 1,
                frame.kind.label(),
                frame.predicate,
                frame.id,
                frame.support(),
                if frame.ask_ready(self.min_ask_support) {
                    "，问法已激活"
                } else {
                    ""
                },
                example
            ));
            if let Some(witness) = frame.latest_witness() {
                output.push_str(&format!("   原句：{}\n", witness));
            }
        }
        if self.frames.len() > REPORT_FRAME_CAP {
            output.push_str(&format!(
                "…另有 {} 条未列出。\n",
                self.frames.len() - REPORT_FRAME_CAP
            ));
        }
        output
    }

    /// 以 ContextImage 信封原子落盘。
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let value = serde_json::to_value(self)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        let mut image = ContextImage::new();
        image.set_section(IMPORT_FRAMES_SECTION, value);
        image.set_section("import_frames_meta", serde_json::json!(self.frames.len()));
        image.seal();
        image.save_atomic(path)
    }

    /// 读取台账：缺失/损坏都安全降级为空台账，绝不按坏数据路由。
    pub fn load(path: &str) -> FrameLoad {
        if !std::path::Path::new(path).exists() {
            return FrameLoad::Missing;
        }
        let image = match ContextImage::load_verified(path) {
            Ok(image) => image,
            Err(error) => return FrameLoad::Damaged(error),
        };
        let Some(value) = image.section(IMPORT_FRAMES_SECTION).cloned() else {
            return FrameLoad::Damaged("缺少 import_frames 分节".to_string());
        };
        match serde_json::from_value::<ImportFrameLedger>(value) {
            Ok(ledger) if ledger.version == IMPORT_FRAMES_VERSION => FrameLoad::Loaded(ledger),
            Ok(ledger) => FrameLoad::Damaged(format!(
                "不支持的导入帧版本 {}（期望 {}）",
                ledger.version, IMPORT_FRAMES_VERSION
            )),
            Err(error) => FrameLoad::Damaged(format!("导入帧解码失败：{}", error)),
        }
    }

    fn enforce_capacity(&mut self) {
        while self.frames.len() > self.max_frames {
            let victim = self
                .frames
                .values()
                .min_by_key(|frame| (frame.support(), frame.last_turn(), frame.id.clone()))
                .map(|frame| frame.id.clone());
            match victim {
                Some(id) => {
                    self.frames.remove(&id);
                }
                None => break,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CandidateFact {
    kind: FactKind,
    subject: String,
    predicate: String,
    object: String,
}

fn candidate_facts(items: &[KnowledgeItem]) -> Vec<CandidateFact> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for item in items {
        match item {
            KnowledgeItem::Entity { properties, .. } => {
                let Some(subject) = properties.get("name").map(|name| name.trim()) else {
                    continue;
                };
                if subject.is_empty() {
                    continue;
                }
                for (predicate, object) in properties {
                    let predicate = predicate.trim();
                    let object = object.trim();
                    if predicate.is_empty()
                        || object.is_empty()
                        || META_PREDICATES.contains(&predicate)
                    {
                        continue;
                    }
                    let key = (
                        FactKind::Attribute,
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                    );
                    if seen.insert(key.clone()) {
                        candidates.push(CandidateFact {
                            kind: key.0,
                            subject: key.1,
                            predicate: key.2,
                            object: key.3,
                        });
                    }
                }
            }
            KnowledgeItem::Relationship {
                from_id,
                to_id,
                rel_type,
                ..
            } => {
                let subject = from_id.trim();
                let object = to_id.trim();
                let predicate = rel_type.trim();
                if subject.is_empty()
                    || object.is_empty()
                    || predicate.is_empty()
                    || subject == object
                    || META_PREDICATES.contains(&predicate)
                {
                    continue;
                }
                let key = (
                    FactKind::Relation,
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                );
                if seen.insert(key.clone()) {
                    candidates.push(CandidateFact {
                        kind: key.0,
                        subject: key.1,
                        predicate: key.2,
                        object: key.3,
                    });
                }
            }
        }
    }
    candidates
}

fn accepted_by_receipt(candidate: &CandidateFact, receipt: &ImportReceipt) -> bool {
    let mut endpoint_matches = 0usize;
    for action in &receipt.actions {
        match (candidate.kind, action) {
            (
                FactKind::Attribute,
                ImportAction::Literal {
                    subject,
                    predicate,
                    value,
                },
            ) => {
                let Some(action_subject) = local_name(DEFAULT_BASE_NS, subject) else {
                    continue;
                };
                let Some(action_predicate) = local_name(DEFAULT_BASE_NS, predicate) else {
                    continue;
                };
                if action_subject == candidate.subject && value == &candidate.object {
                    if action_predicate == candidate.predicate {
                        return true;
                    }
                    if !META_PREDICATES.contains(&action_predicate.as_str()) {
                        endpoint_matches += 1;
                    }
                }
            }
            (
                FactKind::Relation,
                ImportAction::Named {
                    subject,
                    predicate,
                    object,
                },
            ) => {
                let Some(action_subject) = local_name(DEFAULT_BASE_NS, subject) else {
                    continue;
                };
                let Some(action_object) = local_name(DEFAULT_BASE_NS, object) else {
                    continue;
                };
                let Some(action_predicate) = local_name(DEFAULT_BASE_NS, predicate) else {
                    continue;
                };
                let same_direction =
                    action_subject == candidate.subject && action_object == candidate.object;
                let reversed =
                    action_subject == candidate.object && action_object == candidate.subject;
                if same_direction || reversed {
                    if action_predicate == candidate.predicate {
                        return true;
                    }
                    if !META_PREDICATES.contains(&action_predicate.as_str()) {
                        endpoint_matches += 1;
                    }
                }
            }
            _ => {}
        }
    }
    // 别名/逆向归一可能把谓词改成规范名；端点唯一命中时仍可安全对齐。
    endpoint_matches == 1
}

fn align_candidate(
    witness: &str,
    candidate: &CandidateFact,
) -> Option<(Vec<FrameAtom>, String, String)> {
    let chars: Vec<char> = witness.chars().collect();
    let subject_chars: Vec<char> = candidate.subject.chars().collect();
    let object_chars: Vec<char> = candidate.object.chars().collect();
    if subject_chars.is_empty() || object_chars.is_empty() || candidate.subject == candidate.object
    {
        return None;
    }
    let subject_occurrences = occurrences(&chars, &subject_chars);
    let object_occurrences = occurrences(&chars, &object_chars);
    if subject_occurrences.is_empty() || object_occurrences.is_empty() {
        return None;
    }

    let mut pairs: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    for &subject_start in &subject_occurrences {
        let subject_end = subject_start + subject_chars.len();
        for &object_start in &object_occurrences {
            let object_end = object_start + object_chars.len();
            if subject_end <= object_start {
                let span = object_end - subject_start;
                pairs.push((span, subject_start, subject_end, object_start, object_end));
            } else if object_end <= subject_start {
                let span = subject_end - object_start;
                pairs.push((span, subject_start, subject_end, object_start, object_end));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort();
    let best_span = pairs[0].0;
    let best: Vec<_> = pairs.iter().filter(|pair| pair.0 == best_span).collect();
    if best.len() != 1 {
        return None;
    }
    let (_, subject_start, subject_end, object_start, object_end) = *best[0];
    let (first_start, first_end, second_start, second_end, first_slot, second_slot) =
        if subject_start < object_start {
            (
                subject_start,
                subject_end,
                object_start,
                object_end,
                FrameAtom::Subject,
                FrameAtom::Object,
            )
        } else {
            (
                object_start,
                object_end,
                subject_start,
                subject_end,
                FrameAtom::Object,
                FrameAtom::Subject,
            )
        };
    let prefix = slice_chars(&chars, 0, first_start);
    let middle = slice_chars(&chars, first_end, second_start);
    let suffix = slice_chars(&chars, second_end, chars.len());
    if middle.trim().is_empty() {
        return None;
    }

    let mut atoms = Vec::new();
    if !prefix.trim().is_empty() {
        atoms.push(FrameAtom::literal(prefix.trim()));
    }
    atoms.push(first_slot);
    atoms.push(FrameAtom::literal(middle.trim()));
    atoms.push(second_slot);
    if !suffix.trim().is_empty() {
        atoms.push(FrameAtom::literal(suffix.trim()));
    }
    Some((atoms, candidate.subject.clone(), candidate.object.clone()))
}

fn frame_id(kind: FactKind, predicate: &str, atoms: &[FrameAtom]) -> String {
    let value = serde_json::json!({
        "kind": kind,
        "predicate": predicate,
        "atoms": atoms,
    });
    format!("ifr#{}", fnv1a_hex(canonical_json(&value).as_bytes()))
}

fn receipt_digest(receipt: &ImportReceipt) -> String {
    let value = serde_json::to_value(&receipt.actions).unwrap_or(serde_json::Value::Null);
    fnv1a_hex(canonical_json(&value).as_bytes())
}

fn normalize_witness(text: &str) -> String {
    normalize_text(text)
}

fn normalize_text(text: &str) -> String {
    text.trim()
        .trim_end_matches(&['。', '！', '？', '!', '?', '；', ';', '，', ',', ' '][..])
        .trim()
        .to_string()
}

fn slice_chars(chars: &[char], start: usize, end: usize) -> String {
    chars[start..end].iter().collect()
}

fn occurrences(haystack: &[char], needle: &[char]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .filter(|(_, window)| *window == needle)
        .map(|(index, _)| index)
        .collect()
}

fn starts_with(chars: &[char], pos: usize, needle: &[char]) -> bool {
    pos + needle.len() <= chars.len() && chars[pos..pos + needle.len()] == *needle
}

fn match_declaration(
    atoms: &[FrameAtom],
    chars: &[char],
    pos: usize,
    bindings: &mut BTreeMap<String, String>,
    out: &mut Vec<FrameBinding>,
) {
    if out.len() >= MAX_MATCHES {
        return;
    }
    let Some((atom, rest)) = atoms.split_first() else {
        if pos == chars.len() {
            let subject = bindings.get(SLOT_SUBJECT).cloned().unwrap_or_default();
            let object = bindings.get(SLOT_OBJECT).cloned().unwrap_or_default();
            if !subject.is_empty() && !object.is_empty() {
                out.push(FrameBinding { subject, object });
            }
        }
        return;
    };
    match atom {
        FrameAtom::Literal { surface } => {
            let literal: Vec<char> = surface.chars().collect();
            if starts_with(chars, pos, &literal) {
                match_declaration(rest, chars, pos + literal.len(), bindings, out);
            }
        }
        FrameAtom::Subject | FrameAtom::Object => {
            let key = if matches!(atom, FrameAtom::Subject) {
                SLOT_SUBJECT
            } else {
                SLOT_OBJECT
            };
            for end in pos + 1..=chars.len() {
                let value = slice_chars(chars, pos, end);
                let value = value.trim();
                if value.is_empty() {
                    continue;
                }
                let previous = bindings.insert(key.to_string(), value.to_string());
                match_declaration(rest, chars, end, bindings, out);
                match previous {
                    Some(previous) => {
                        bindings.insert(key.to_string(), previous);
                    }
                    None => {
                        bindings.remove(key);
                    }
                }
                if out.len() >= MAX_MATCHES {
                    return;
                }
            }
        }
    }
}

fn match_question(
    atoms: &[AskAtom],
    interrogatives: &[String],
    chars: &[char],
    pos: usize,
    bindings: &mut BTreeMap<String, String>,
    out: &mut Vec<AskBinding>,
) {
    if out.len() >= MAX_MATCHES {
        return;
    }
    let Some((atom, rest)) = atoms.split_first() else {
        if pos == chars.len() {
            let known = bindings.get("known").cloned().unwrap_or_default();
            if !known.is_empty() {
                out.push(AskBinding { known });
            }
        }
        return;
    };
    match atom {
        AskAtom::Literal { surface } => {
            let literal: Vec<char> = surface.chars().collect();
            if starts_with(chars, pos, &literal) {
                match_question(
                    rest,
                    interrogatives,
                    chars,
                    pos + literal.len(),
                    bindings,
                    out,
                );
            }
        }
        AskAtom::Slot => {
            for end in pos + 1..=chars.len() {
                let value = slice_chars(chars, pos, end);
                let value = value.trim();
                if value.is_empty() {
                    continue;
                }
                let previous = bindings.insert("known".to_string(), value.to_string());
                match_question(rest, interrogatives, chars, end, bindings, out);
                match previous {
                    Some(previous) => {
                        bindings.insert("known".to_string(), previous);
                    }
                    None => {
                        bindings.remove("known");
                    }
                }
                if out.len() >= MAX_MATCHES {
                    return;
                }
            }
        }
        AskAtom::Question { .. } => {
            for interrogative in interrogatives {
                let word: Vec<char> = interrogative.chars().collect();
                if starts_with(chars, pos, &word) {
                    match_question(rest, interrogatives, chars, pos + word.len(), bindings, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::entity_uri;
    use std::collections::HashMap;

    fn entity(name: &str, key: &str, value: &str) -> KnowledgeItem {
        KnowledgeItem::Entity {
            label: "Person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), name.to_string()),
                (key.to_string(), value.to_string()),
            ]),
        }
    }

    fn relation(subject: &str, predicate: &str, object: &str) -> KnowledgeItem {
        KnowledgeItem::Relationship {
            from_label: "Person".to_string(),
            from_id: subject.to_string(),
            to_label: "Person".to_string(),
            to_id: object.to_string(),
            rel_type: predicate.to_string(),
            properties: HashMap::new(),
        }
    }

    fn receipt(items: &[KnowledgeItem]) -> ImportReceipt {
        let mut receipt = ImportReceipt::default();
        for item in items {
            match item {
                KnowledgeItem::Entity { properties, .. } => {
                    let Some(name) = properties.get("name") else {
                        continue;
                    };
                    let subject = entity_uri(DEFAULT_BASE_NS, name);
                    for (predicate, value) in properties {
                        if predicate == "name" {
                            continue;
                        }
                        receipt.actions.push(ImportAction::Literal {
                            subject: subject.clone(),
                            predicate: format!("{}{}", DEFAULT_BASE_NS, predicate),
                            value: value.clone(),
                        });
                    }
                }
                KnowledgeItem::Relationship {
                    from_id,
                    to_id,
                    rel_type,
                    ..
                } => {
                    receipt.actions.push(ImportAction::Named {
                        subject: entity_uri(DEFAULT_BASE_NS, from_id),
                        predicate: format!("{}{}", DEFAULT_BASE_NS, rel_type),
                        object: entity_uri(DEFAULT_BASE_NS, to_id),
                    });
                }
            }
        }
        receipt.added = receipt.actions.len();
        receipt
    }

    #[test]
    fn learns_reversible_attribute_frame_and_activates_ask_after_two_witnesses() {
        let mut ledger = ImportFrameLedger::new();
        let first = entity("姚明", "height", "226");
        let first_receipt = receipt(std::slice::from_ref(&first));
        let report = ledger.learn_from_import(1, "姚明身高226cm", &[first], &first_receipt);
        assert_eq!(report.learned, 1);
        assert_eq!(ledger.len(), 1);
        let frame = ledger.frames().next().unwrap();
        assert_eq!(
            frame.realize("姚明", "226").as_deref(),
            Some("姚明身高226cm")
        );
        assert!(frame.ask_ready(2) == false);
        assert!(ledger.match_question("姚明身高多少", 2).is_none());

        let second = entity("叶莉", "height", "190");
        let second_receipt = receipt(std::slice::from_ref(&second));
        let report = ledger.learn_from_import(2, "叶莉身高190cm", &[second], &second_receipt);
        assert_eq!(report.updated, 1);
        let frame = ledger.frames().next().unwrap();
        assert_eq!(frame.support(), 2);
        assert!(frame.ask_ready(2));

        let matched = ledger
            .match_question("姚明身高多少", 2)
            .expect("两条见证后问法应激活");
        assert_eq!(matched.focus, FrameFocus::Object);
        assert_eq!(matched.subject.as_deref(), Some("姚明"));
        let answer = ledger
            .answer("姚明身高多少", 2, |_, subject, predicate| {
                assert_eq!(subject, "姚明");
                assert_eq!(predicate, "height");
                vec!["226".to_string()]
            })
            .expect("图谱有值时应回答");
        assert_eq!(answer.text, "姚明身高226cm");
        assert!(answer.render_with_witness().contains("依据："));
    }

    #[test]
    fn learns_relation_frame_and_asks_for_object() {
        let mut ledger = ImportFrameLedger::new();
        let first = relation("张三", "teacher_of", "李四");
        ledger.learn_from_import(
            1,
            "张三是李四的老师",
            std::slice::from_ref(&first),
            &receipt(std::slice::from_ref(&first)),
        );
        let second = relation("王五", "teacher_of", "赵六");
        ledger.learn_from_import(
            2,
            "王五是赵六的老师",
            std::slice::from_ref(&second),
            &receipt(std::slice::from_ref(&second)),
        );
        let matched = ledger
            .match_question("张三是谁的老师", 2)
            .expect("关系问法应命中");
        assert_eq!(matched.predicate, "teacher_of");
        assert_eq!(matched.subject.as_deref(), Some("张三"));
        let answer = ledger
            .answer("张三是谁的老师", 2, |_, subject, predicate| {
                assert_eq!(subject, "张三");
                assert_eq!(predicate, "teacher_of");
                vec!["李四".to_string()]
            })
            .unwrap();
        assert_eq!(answer.text, "张三是李四的老师");
    }

    #[test]
    fn ambiguous_alignment_is_skipped() {
        let mut ledger = ImportFrameLedger::new();
        let item = relation("张三", "teacher_of", "李四");
        let report = ledger.learn_from_import(
            1,
            "张三说李四是张三的老师",
            std::slice::from_ref(&item),
            &receipt(std::slice::from_ref(&item)),
        );
        assert_eq!(report.skipped_alignment, 1);
        assert!(ledger.is_empty());
    }

    #[test]
    fn unaccepted_facts_do_not_become_language() {
        let mut ledger = ImportFrameLedger::new();
        let item = entity("姚明", "height", "226");
        let report =
            ledger.learn_from_import(1, "姚明身高226cm", &[item], &ImportReceipt::default());
        assert_eq!(report.skipped_unaccepted, 1);
        assert!(ledger.is_empty());
    }

    #[test]
    fn no_trigger_literal_is_not_learned() {
        let mut ledger = ImportFrameLedger::new();
        let item = relation("张三", "friend", "李四");
        let report = ledger.learn_from_import(
            1,
            "张三李四",
            std::slice::from_ref(&item),
            &receipt(std::slice::from_ref(&item)),
        );
        assert_eq!(report.skipped_alignment, 1);
        assert!(ledger.is_empty());
    }

    #[test]
    fn undo_retracts_only_that_imports_witnesses() {
        let mut ledger = ImportFrameLedger::new();
        let first = entity("姚明", "height", "226");
        let first_receipt = receipt(std::slice::from_ref(&first));
        ledger.learn_from_import(1, "姚明身高226cm", &[first], &first_receipt);
        let second = entity("叶莉", "height", "190");
        let second_receipt = receipt(std::slice::from_ref(&second));
        ledger.learn_from_import(2, "叶莉身高190cm", &[second], &second_receipt);
        assert_eq!(ledger.frames().next().unwrap().support(), 2);

        assert_eq!(ledger.retract_import(&first_receipt), 1);
        assert_eq!(ledger.frames().next().unwrap().support(), 1);
        assert_eq!(ledger.retract_import(&second_receipt), 1);
        assert!(ledger.is_empty());
    }

    #[test]
    fn save_load_roundtrip_keeps_observations_and_witnesses() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("import_frames.json");
        let path = path.to_str().unwrap();
        let mut ledger = ImportFrameLedger::new();
        let item = entity("姚明", "height", "226");
        ledger.learn_from_import(
            1,
            "姚明身高226cm",
            std::slice::from_ref(&item),
            &receipt(std::slice::from_ref(&item)),
        );
        ledger.save(path).unwrap();
        let FrameLoad::Loaded(restored) = ImportFrameLedger::load(path) else {
            panic!("应能恢复导入帧台账");
        };
        assert_eq!(restored, ledger);
        assert_eq!(
            restored.frames().next().unwrap().latest_witness(),
            Some("姚明身高226cm")
        );
    }

    #[test]
    fn retain_facts_prunes_frames_whose_example_left_the_graph() {
        let mut ledger = ImportFrameLedger::new();
        let item = entity("姚明", "height", "226");
        ledger.learn_from_import(
            1,
            "姚明身高226cm",
            std::slice::from_ref(&item),
            &receipt(std::slice::from_ref(&item)),
        );
        assert_eq!(
            ledger.retain_facts(&[("姚明".to_string(), "height".to_string(), "226".to_string())]),
            0
        );
        assert_eq!(ledger.retain_facts(&[]), 1);
        assert!(ledger.is_empty());
    }

    #[test]
    fn capacity_evicts_lowest_support_frame() {
        let mut ledger = ImportFrameLedger::with_limits(1, 8, 2);
        let first = entity("姚明", "height", "226");
        ledger.learn_from_import(
            1,
            "姚明身高226cm",
            std::slice::from_ref(&first),
            &receipt(std::slice::from_ref(&first)),
        );
        let second = entity("张三", "age", "18");
        ledger.learn_from_import(
            2,
            "张三年龄18岁",
            std::slice::from_ref(&second),
            &receipt(std::slice::from_ref(&second)),
        );
        assert_eq!(ledger.len(), 1);
        assert!(ledger.frames().all(|frame| frame.predicate == "age"));
    }
}

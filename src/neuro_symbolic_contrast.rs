//! 神经符号对比仲裁器（Neuro-Symbolic Contrast Arbiter）
//!
//! 灵感：「神经符号对比」——让系统像 LLM 一样对自然语言陈述给出"带把握的
//! 判断"，但两个判断通道都是纯符号主义的，没有任何神经网络与权重训练：
//!
//! - A 通道（联想/分布通道，扮演"直觉"）：
//!     软逻辑（MLN-lite）对「主语是否具有谓词」的概率传播 + 关联规则置信度
//!     （谓词存在性）+ 图游走 RWR 相似度与语义嵌入余弦（对象契合度）。
//!     这些量全部来自计数、幂迭代与固定组合权重，可解释、可复算、可回溯；
//!     "合并联想分"只是固定的加权混合，不是训练出来的参数。
//! - B 通道（演绎/符号通道，扮演"逻辑"）：
//!     语义蕴含判定（蕴含/矛盾/未知）+ 归结 SLD 反驳证明。
//! - 仲裁规则：
//!     逻辑证实 且 联想支持 → 双通道一致；
//!     逻辑证实 但 联想贫弱 → 仅符号证实；
//!     逻辑未知 但 联想支持 → 仅联想推测（带对冲词输出，登记为待定推测）；
//!     逻辑反驳 但 联想支持 → 通道冲突（符号优先判假，标记为争议）；
//!     其余 → 双通道未知（系统坦诚地说"不知道"，并给出获取知识的途径）。
//! - 会话记忆（多轮）：
//!     案例日志（有界）、校准统计（一致率/争议率）、待定推测清单；
//!     用户可「对比确认 N」把推测采信为会话事实（无需改知识图谱），
//!     或「对比否认 N」把推测加入屏蔽集，联想通道此后不再提出同类推测——
//!     这是纯符号的"从反馈中校准"，不含梯度。
//!
//! 与 LLM 的对照：LLM 的"幻觉"来自分布通道没有符号约束；本模块把两条通道
//! 显式对照，凡逻辑未证实的输出一律降级为"推测"并要求确认，从而在保留
//! 生成式直觉的同时抑制幻觉。

use std::collections::{HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::association_miner::{attribute_item, display_item, relation_item, type_item};
use crate::chinese_reasoning_nlu::{parse_proposition, render_atom_zh};
use crate::resolution_engine::{Atom, ProofNode, ProofSource, Term};
use crate::semantic_entailment::Verdict;
use crate::symbolic_reasoner::{DialogueSnapshot, SymbolicReasoner};
use crate::ReasoningConfig;

const HAS_PREDICATE: &str = "__具有__";
const TYPE_PREDICATE: &str = "__类型__";
const TYPE_STATEMENT: &str = "是";
const NEGATION_MARKERS: [&str; 2] = ["不", "没"];

/// 双通道仲裁结论
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContrastVerdict {
    /// 逻辑证实且联想支持；或双通道一致否定
    Agree,
    /// 逻辑证实，联想通道证据不足
    SymbolicOnly,
    /// 逻辑未知，联想通道倾向为真（推测，需确认）
    DistributionalOnly,
    /// 逻辑反驳但联想通道倾向为真（争议，符号优先）
    Conflict,
    /// 两通道都无法判断
    BothAbstain,
}

impl ContrastVerdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Agree => "双通道一致",
            Self::SymbolicOnly => "仅符号证实",
            Self::DistributionalOnly => "仅联想推测",
            Self::Conflict => "通道冲突",
            Self::BothAbstain => "双通道未知",
        }
    }

    fn counter_slot(self, stats: &mut ContrastStats) {
        match self {
            Self::Agree => stats.agree += 1,
            Self::SymbolicOnly => stats.symbolic_only += 1,
            Self::DistributionalOnly => stats.distributional_only += 1,
            Self::Conflict => stats.conflict += 1,
            Self::BothAbstain => stats.both_abstain += 1,
        }
    }
}

impl Default for ContrastVerdict {
    fn default() -> Self {
        Self::BothAbstain
    }
}

/// 单条通道证据：来源、分数（0~1 或相似度）与人类可读细节
#[derive(Debug, Clone)]
pub struct ChannelEvidence {
    pub channel: &'static str,
    pub score: f64,
    pub detail: String,
}

/// B 通道（符号/演绎）的判定结果
#[derive(Debug, Clone, Default)]
pub struct SymbolicReading {
    pub formalized: bool,
    pub atom: Option<Atom>,
    pub proved: bool,
    pub contradicted: bool,
    pub confidence: f64,
    pub verdict_labels: Vec<String>,
    pub reasons: Vec<String>,
    pub proof_steps: Vec<String>,
    pub counter_evidence: Vec<String>,
}

/// A 通道（联想/分布）的判定结果
#[derive(Debug, Clone, Default)]
pub struct DistributionalReading {
    pub probed: bool,
    pub blocked: bool,
    pub soft: Option<ChannelEvidence>,
    pub association: Option<ChannelEvidence>,
    pub walk: Option<ChannelEvidence>,
    pub embedding: Option<ChannelEvidence>,
    pub score: f64,
    pub supported: bool,
    pub note: String,
}

/// 一次对比案例（会话记忆的基本单元）
#[derive(Debug, Clone)]
pub struct ContrastCase {
    pub id: usize,
    pub statement: String,
    pub formalization: String,
    pub key: String,
    pub symbolic: SymbolicReading,
    pub distributional: DistributionalReading,
    pub verdict: ContrastVerdict,
    pub answer: String,
    pub guess_id: Option<usize>,
}

/// 待定/已决推测
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GuessStatus {
    #[default]
    Pending,
    Confirmed,
    Refuted,
}

#[derive(Debug, Clone)]
pub struct ContrastGuess {
    pub id: usize,
    pub case_id: usize,
    pub statement: String,
    pub key: String,
    /// 联想通道的简短依据（用于生成澄清追问）
    pub evidence: String,
    /// 形式化原子的显示串（用于把用户否认传播进符号通道）
    pub atom_display: Option<String>,
    pub score: f64,
    pub status: GuessStatus,
}

/// 会话校准统计（可跨会话持久化）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContrastStats {
    pub total: usize,
    pub agree: usize,
    pub symbolic_only: usize,
    pub distributional_only: usize,
    pub conflict: usize,
    pub both_abstain: usize,
    pub confirmed: usize,
    pub refuted: usize,
    /// 联想通道被实际探测的案例数
    #[serde(default)]
    pub probed: usize,
    /// 联想通道被屏蔽集拦下的案例数
    #[serde(default)]
    pub blocked: usize,
    /// 累计经历过的会话数（用于跨会话记忆报告）
    #[serde(default)]
    pub sessions: usize,
}

/// 待定推测的持久化快照（跨会话延续推测生命周期）
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SnapshotGuess {
    pub id: usize,
    pub case_id: usize,
    pub statement: String,
    pub key: String,
    #[serde(default)]
    pub evidence: String,
    #[serde(default)]
    pub atom_display: Option<String>,
    #[serde(default)]
    pub score: f64,
    pub status: GuessStatus,
}

/// 案例日志条目的持久化快照（只保留报告所需字段，通道证据为会话内细节）
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SnapshotCase {
    pub id: usize,
    pub statement: String,
    #[serde(default)]
    pub verdict: ContrastVerdict,
}

/// 对比仲裁记忆的完整快照：随 dialogue_state.json 跨会话持久化
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContrastSnapshot {
    #[serde(default)]
    pub stats: ContrastStats,
    #[serde(default)]
    pub guesses: Vec<SnapshotGuess>,
    #[serde(default)]
    pub refuted_keys: Vec<String>,
    #[serde(default)]
    pub confirmed_keys: Vec<String>,
    /// 用户否认过的原子显示串：作为会话级否证进入符号通道
    #[serde(default)]
    pub refuted_atoms: Vec<String>,
    #[serde(default)]
    pub recent_cases: Vec<SnapshotCase>,
    #[serde(default)]
    pub next_case_id: usize,
    #[serde(default)]
    pub next_guess_id: usize,
}

/// 一次对比追问：系统主动把待定推测逆转为澄清问题
#[derive(Debug, Clone)]
pub struct ContrastFollowup {
    pub guess_id: usize,
    pub question: String,
}

/// 对立面阈值与固定组合权重（全部人工设定，非训练所得）
#[derive(Debug, Clone)]
pub struct ContrastTuning {
    pub min_score: f64,
    pub presence_floor: f64,
    pub walk_floor: f64,
    pub embedding_floor: f64,
    pub presence_weight: f64,
    pub affinity_weight: f64,
    pub max_cases: usize,
    pub max_guesses: usize,
}

impl Default for ContrastTuning {
    fn default() -> Self {
        Self {
            min_score: 0.45,
            presence_floor: 0.5,
            walk_floor: 0.2,
            embedding_floor: 0.35,
            presence_weight: 0.7,
            affinity_weight: 0.3,
            max_cases: 40,
            max_guesses: 12,
        }
    }
}

/// 一次形式化探测所需的信息：谓词存在性查询 + 关联规则目标物品
struct PresenceProbe {
    soft_query_predicate: &'static str,
    subject: String,
    target: String,
    object: String,
}

pub struct ContrastArbiter {
    tuning: ContrastTuning,
    cases: VecDeque<ContrastCase>,
    guesses: VecDeque<ContrastGuess>,
    refuted_keys: HashSet<String>,
    confirmed_keys: HashSet<String>,
    /// 用户否认过的原子（显示串）：进入符号通道作为会话级否证
    refuted_atoms: HashSet<String>,
    stats: ContrastStats,
    next_case_id: usize,
    next_guess_id: usize,
}

impl ContrastArbiter {
    pub fn new() -> Self {
        Self::with_tuning(ContrastTuning::default())
    }

    pub fn with_tuning(tuning: ContrastTuning) -> Self {
        let mut stats = ContrastStats::default();
        stats.sessions = 1;
        Self {
            tuning,
            cases: VecDeque::new(),
            guesses: VecDeque::new(),
            refuted_keys: HashSet::new(),
            confirmed_keys: HashSet::new(),
            refuted_atoms: HashSet::new(),
            stats,
            next_case_id: 1,
            next_guess_id: 1,
        }
    }

    /// 从既有 [reasoning] 配置读取阈值：软逻辑概率下限 = 谓词存在性下限，
    /// 相似度下限沿用图嵌入配置，其余为固定值
    pub fn from_reasoning_config(config: &ReasoningConfig) -> Self {
        let mut tuning = ContrastTuning::default();
        if config.soft_logic_min_probability > 0.0 {
            tuning.presence_floor = config.soft_logic_min_probability.clamp(0.05, 0.95);
        }
        if config.embedding_similarity_floor > 0.0 {
            tuning.embedding_floor = config.embedding_similarity_floor.clamp(0.05, 0.95);
        }
        if config.association_min_confidence > 0.0 {
            tuning.presence_floor =
                tuning.presence_floor.min(config.association_min_confidence.clamp(0.05, 0.95));
        }
        Self::with_tuning(tuning)
    }

    pub fn stats(&self) -> &ContrastStats {
        &self.stats
    }

    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    pub fn pending_guess_count(&self) -> usize {
        self.guesses
            .iter()
            .filter(|guess| guess.status == GuessStatus::Pending)
            .count()
    }

    pub fn refuted_keys(&self) -> impl Iterator<Item = &String> {
        self.refuted_keys.iter()
    }

    pub fn refuted_atom_count(&self) -> usize {
        self.refuted_atoms.len()
    }

    /// 记忆摘要：启动恢复 / 报告时的单行概述
    pub fn memory_summary(&self) -> String {
        if self.stats.total == 0 && self.guesses.is_empty() {
            return "对比记忆：空。".to_string();
        }
        format!(
            "对比记忆：累计 {} 个会话、{} 个案例（一致 {}，仅符号 {}，仅联想 {}，冲突 {}，未知 {}）；待定推测 {} 条，已屏蔽 {} 条。",
            self.stats.sessions,
            self.stats.total,
            self.stats.agree,
            self.stats.symbolic_only,
            self.stats.distributional_only,
            self.stats.conflict,
            self.stats.both_abstain,
            self.pending_guess_count(),
            self.refuted_atoms.len().max(self.refuted_keys.len())
        )
    }

    // ------------------------- 跨会话持久化 -------------------------

    /// 生成可写入 dialogue_state.json 的快照（只保留报告与生命周期所需信息）
    pub fn snapshot(&self) -> ContrastSnapshot {
        let mut refuted_keys: Vec<String> = self.refuted_keys.iter().cloned().collect();
        refuted_keys.sort();
        let mut confirmed_keys: Vec<String> = self.confirmed_keys.iter().cloned().collect();
        confirmed_keys.sort();
        let mut refuted_atoms: Vec<String> = self.refuted_atoms.iter().cloned().collect();
        refuted_atoms.sort();
        ContrastSnapshot {
            stats: self.stats.clone(),
            guesses: self
                .guesses
                .iter()
                .map(|guess| SnapshotGuess {
                    id: guess.id,
                    case_id: guess.case_id,
                    statement: guess.statement.clone(),
                    key: guess.key.clone(),
                    evidence: guess.evidence.clone(),
                    atom_display: guess.atom_display.clone(),
                    score: guess.score,
                    status: guess.status,
                })
                .collect(),
            refuted_keys,
            confirmed_keys,
            refuted_atoms,
            recent_cases: self
                .cases
                .iter()
                .map(|case| SnapshotCase {
                    id: case.id,
                    statement: case.statement.clone(),
                    verdict: case.verdict,
                })
                .collect(),
            next_case_id: self.next_case_id,
            next_guess_id: self.next_guess_id,
        }
    }

    /// 恢复快照：推测生命周期、屏蔽集与校准统计跨会话延续；
    /// 返回是否恢复出了实质内容（空快照返回 false）
    pub fn restore_snapshot(&mut self, snapshot: &ContrastSnapshot) -> bool {
        if snapshot.stats.total == 0
            && snapshot.guesses.is_empty()
            && snapshot.recent_cases.is_empty()
            && snapshot.refuted_keys.is_empty()
            && snapshot.refuted_atoms.is_empty()
        {
            return false;
        }

        self.stats = snapshot.stats.clone();
        self.stats.sessions += 1;

        self.guesses = snapshot
            .guesses
            .iter()
            .take(self.tuning.max_guesses)
            .map(|guess| ContrastGuess {
                id: guess.id,
                case_id: guess.case_id,
                statement: guess.statement.clone(),
                key: guess.key.clone(),
                evidence: guess.evidence.clone(),
                atom_display: guess.atom_display.clone(),
                score: guess.score,
                status: guess.status,
            })
            .collect();

        self.refuted_keys = snapshot.refuted_keys.iter().cloned().collect();
        self.confirmed_keys = snapshot.confirmed_keys.iter().cloned().collect();
        self.refuted_atoms = snapshot.refuted_atoms.iter().cloned().collect();

        // 案例日志按时间顺序恢复，超出上界时保留最近的
        self.cases = snapshot
            .recent_cases
            .iter()
            .rev()
            .take(self.tuning.max_cases)
            .rev()
            .map(|case| ContrastCase {
                id: case.id,
                statement: case.statement.clone(),
                formalization: String::new(),
                key: String::new(),
                symbolic: SymbolicReading::default(),
                distributional: DistributionalReading::default(),
                verdict: case.verdict,
                answer: String::new(),
                guess_id: None,
            })
            .collect();

        let max_case_id = snapshot
            .recent_cases
            .iter()
            .map(|case| case.id)
            .max()
            .unwrap_or(0);
        let max_guess_id = snapshot
            .guesses
            .iter()
            .map(|guess| guess.id)
            .max()
            .unwrap_or(0);
        self.next_case_id = snapshot.next_case_id.max(max_case_id + 1).max(1);
        self.next_guess_id = snapshot.next_guess_id.max(max_guess_id + 1).max(1);
        true
    }

    /// 从 dialogue_state.json 中提取对比记忆快照（文件缺失/无该字段时返回 None）
    pub fn load_snapshot_from_dialogue_state(path: &str) -> Option<ContrastSnapshot> {
        let content = std::fs::read_to_string(path).ok()?;
        let snapshot: DialogueSnapshot = serde_json::from_str(&content).ok()?;
        snapshot.contrast
    }

    // ------------------------- 主流程：对比仲裁 -------------------------

    /// 对一条自然语言陈述做双通道对比，返回案例（并写入会话记忆）
    pub fn contrast_statement(
        &mut self,
        statement: &str,
        reasoner: &SymbolicReasoner,
    ) -> ContrastCase {
        let trimmed = statement
            .trim()
            .trim_end_matches(['。', '．', '.', '！', '!', '？', '?'])
            .trim();
        let id = self.next_case_id;
        self.next_case_id += 1;

        let symbolic = self.read_symbolic(trimmed, reasoner);
        let distributional = self.read_distributional(trimmed, &symbolic, reasoner);
        let verdict = Self::arbitrate(&symbolic, &distributional);

        let key = symbolic
            .atom
            .as_ref()
            .map(|atom| format!("{}::{}", trimmed, atom.display()))
            .unwrap_or_else(|| trimmed.to_string());
        let formalization = symbolic
            .atom
            .as_ref()
            .map(|atom| format!("{}（{}）", atom.display(), render_atom_zh(atom)))
            .unwrap_or_else(|| "（未能形式化）".to_string());

        let guess_id = if verdict == ContrastVerdict::DistributionalOnly {
            let atom_display = symbolic.atom.as_ref().map(|atom| atom.display());
            Some(self.register_guess(
                id,
                trimmed,
                &key,
                distributional.score,
                &distributional,
                atom_display,
            ))
        } else {
            None
        };

        let answer = Self::compose_answer(verdict, &symbolic, &distributional, guess_id);

        let case = ContrastCase {
            id,
            statement: trimmed.to_string(),
            formalization,
            key,
            symbolic,
            distributional,
            verdict,
            answer,
            guess_id,
        };

        self.stats.total += 1;
        verdict.counter_slot(&mut self.stats);
        if case.distributional.probed {
            self.stats.probed += 1;
        }
        if case.distributional.blocked {
            self.stats.blocked += 1;
        }
        if self.cases.len() >= self.tuning.max_cases {
            self.cases.pop_front();
        }
        self.cases.push_back(case.clone());
        case
    }

    fn arbitrate(
        symbolic: &SymbolicReading,
        distributional: &DistributionalReading,
    ) -> ContrastVerdict {
        match (
            symbolic.proved,
            symbolic.contradicted,
            distributional.supported,
        ) {
            (true, _, true) => ContrastVerdict::Agree,
            (true, _, false) => ContrastVerdict::SymbolicOnly,
            (false, true, true) => ContrastVerdict::Conflict,
            (false, true, false) => ContrastVerdict::Agree,
            (false, false, true) => ContrastVerdict::DistributionalOnly,
            (false, false, false) => ContrastVerdict::BothAbstain,
        }
    }

    // ------------------------- B 通道：符号/演绎 -------------------------

    fn read_symbolic(&self, statement: &str, reasoner: &SymbolicReasoner) -> SymbolicReading {
        let mut reading = SymbolicReading::default();
        for result in reasoner.check_statement(statement) {
            reading.formalized = true;
            reading.verdict_labels.push(result.verdict.label().to_string());
            if !result.reason.is_empty() {
                reading.reasons.push(result.reason.clone());
            }
            for step in &result.proof {
                reading
                    .proof_steps
                    .push(format!("{}：{}", step.kind.label(), step.detail));
            }
            match result.verdict {
                Verdict::Entailed => {
                    reading.proved = true;
                    reading.confidence = reading.confidence.max(result.confidence);
                }
                Verdict::Contradicted => {
                    reading.contradicted = true;
                    reading.confidence = reading.confidence.max(result.confidence);
                    reading
                        .counter_evidence
                        .extend(result.counter_evidence.iter().cloned());
                }
                Verdict::Unknown => {}
            }
        }

        if reading.atom.is_none() {
            reading.atom = parse_proposition(statement).filter(Atom::is_ground);
        }
        reading.formalized = reading.formalized || reading.atom.is_some();

        let negated = contains_negation(statement);
        if !reading.proved && !reading.contradicted && !negated {
            if let Some(atom) = &reading.atom {
                let solutions = reasoner
                    .resolver()
                    .solve(std::slice::from_ref(atom), &[]);
                if let Some(solution) = solutions.first() {
                    reading.proved = true;
                    reading.confidence = solution.confidence;
                    for proof in &solution.proofs {
                        reading.proof_steps.push(render_proof_node(proof));
                    }
                    if reading
                        .verdict_labels
                        .iter()
                        .any(|label| label.contains("未知"))
                        || reading.verdict_labels.is_empty()
                    {
                        reading.reasons.push(
                            "语义蕴含未判定，但后向链归结（图谱事实 + 会话事实 + 规则）已给出证明"
                                .to_string(),
                        );
                    }
                }
            }
        }

        // 用户否认过该原子：作为会话级否证进入符号通道（除非知识图谱/规则已证实它）。
        // 这正是纯符号的"从反馈中校准"：一次否认即成为可持久化的负事实，
        // 此后同一陈述会被逻辑通道判为不成立，而不仅是屏蔽联想通道。
        if !reading.proved && !reading.contradicted && !negated {
            if let Some(atom) = &reading.atom {
                if self.refuted_atoms.contains(&atom.display()) {
                    reading.contradicted = true;
                    reading.confidence = reading.confidence.max(0.9);
                    reading.counter_evidence.push(
                        "用户已在对比流程中明确否认该陈述（会话级否证，随对话记忆持久化）"
                            .to_string(),
                    );
                    reading.verdict_labels.push("用户否认（会话记忆）".to_string());
                }
            }
        }

        if !reading.formalized {
            reading.note_push("形式化失败：需要「S的P是O / O是S的P / S是类型」等句式");
        }
        reading
    }

    // ------------------------- A 通道：联想/分布 -------------------------

    fn read_distributional(
        &self,
        statement: &str,
        symbolic: &SymbolicReading,
        reasoner: &SymbolicReasoner,
    ) -> DistributionalReading {
        let mut reading = DistributionalReading::default();
        if contains_negation(statement) {
            reading.note = "否定陈述不做联想通道评估（避免把否定误当肯定传播）".to_string();
            return reading;
        }
        let Some(atom) = symbolic.atom.as_ref() else {
            reading.note = "无法形式化，联想通道无从探测".to_string();
            return reading;
        };
        let Some(probe) = presence_probe(atom) else {
            reading.note = "该命题形态（一元/多元）暂不支持联想探测".to_string();
            return reading;
        };
        reading.probed = true;

        if self.refuted_keys.contains(&case_key(statement, atom)) {
            reading.blocked = true;
            reading.note = "该推测曾被用户否认，联想通道已将其屏蔽".to_string();
            return reading;
        }

        reading.soft = self.read_soft_presence(&probe, reasoner);
        reading.association = self.read_association_presence(&probe, reasoner);
        let (walk, embedding) = self.read_affinity(&probe, reasoner);
        reading.walk = walk;
        reading.embedding = embedding;

        let presence = [
            reading.soft.as_ref().map(|e| e.score),
            reading.association.as_ref().map(|e| e.score),
        ]
        .into_iter()
        .flatten()
        .fold(0.0f64, f64::max);
        let affinity = [
            reading.walk.as_ref().map(|e| e.score),
            reading.embedding.as_ref().map(|e| e.score),
        ]
        .into_iter()
        .flatten()
        .fold(0.0f64, f64::max);
        reading.score =
            self.tuning.presence_weight * presence + self.tuning.affinity_weight * affinity;
        reading.supported = presence >= self.tuning.presence_floor
            && reading.score >= self.tuning.min_score;

        if reading.soft.is_none()
            && reading.association.is_none()
            && reading.walk.is_none()
            && reading.embedding.is_none()
        {
            reading.note = "两通道证据均不足：本体中没有同型实体可供联想".to_string();
        } else if !reading.supported {
            reading.note = format!(
                "联想分 {:.2} 低于阈值 {:.2}（或谓词存在性证据不足），不足以形成推测",
                reading.score, self.tuning.min_score
            );
        }
        reading
    }

    /// 软逻辑：MLN-lite 对「主语具有谓词/类型」的概率传播（计数权重，无训练）
    fn read_soft_presence(
        &self,
        probe: &PresenceProbe,
        reasoner: &SymbolicReasoner,
    ) -> Option<ChannelEvidence> {
        let pattern = Atom::pattern(
            probe.soft_query_predicate,
            Term::constant(&probe.subject),
            Term::var("X"),
        );
        let candidates = reasoner.soft_logic().infer(&pattern, 8);
        let mut best: Option<(f64, bool, String)> = None;
        for fact in candidates {
            let Some(value) = fact.atom.terms.get(1).map(|term| term.name()) else {
                continue;
            };
            if value != probe.target {
                continue;
            }
            let source = if fact.from_fact {
                "知识图谱已有该谓词（对象另见）".to_string()
            } else {
                fact.evidence
                    .first()
                    .map(|evidence| evidence.source.clone())
                    .unwrap_or_else(|| "软逻辑概率传播".to_string())
            };
            let replace = best
                .as_ref()
                .map(|(score, _, _)| fact.probability > *score)
                .unwrap_or(true);
            if replace {
                best = Some((fact.probability, fact.from_fact, source));
            }
        }
        best.map(|(probability, from_fact, source)| ChannelEvidence {
            channel: "软逻辑",
            score: probability,
            detail: if from_fact {
                format!("P={:.2}；{}", probability, source)
            } else {
                format!("P={:.2}；由计数加权规则推出：{}", probability, source)
            },
        })
    }

    /// 关联规则：主语篮子里已有哪些物品 ⇒ 该谓词物品的出现置信度
    /// （谓词存在性，不含对象；对象契合度由图游走/嵌入补充）
    fn read_association_presence(
        &self,
        probe: &PresenceProbe,
        reasoner: &SymbolicReasoner,
    ) -> Option<ChannelEvidence> {
        let miner = reasoner.associations();
        let basket = miner.basket(&probe.subject)?;
        // 谓词可能以关系（三元组）、属性（键值）或类型标签形式入库，三种物品名都接受
        let relation_target = relation_item(&probe.target);
        let attribute_target = attribute_item(&probe.target);
        let type_target = type_item(&probe.target);
        let mut best: Option<&crate::association_miner::AssociationRule> = None;
        for rule in miner.rules() {
            if rule.consequent != relation_target
                && rule.consequent != attribute_target
                && rule.consequent != type_target
            {
                continue;
            }
            // 跳过同义反复规则（结论物品与前件物品同显示名），它们不提供新信息
            if rule
                .antecedent
                .iter()
                .any(|item| display_item(item) == display_item(&rule.consequent))
            {
                continue;
            }
            if !rule
                .antecedent
                .iter()
                .all(|item| basket.contains(item.as_str()))
            {
                continue;
            }
            if best
                .map(|current| rule.confidence > current.confidence)
                .unwrap_or(true)
            {
                best = Some(rule);
            }
        }
        best.map(|rule| ChannelEvidence {
            channel: "关联规则",
            score: rule.confidence,
            detail: format!(
                "{} ⇒ {}（置信度 {:.2}，提升度 {:.2}，支持 {}/{}）",
                rule.antecedent_text(),
                rule.consequent_text(),
                rule.confidence,
                rule.lift,
                rule.support,
                rule.total
            ),
        })
    }

    /// 对象契合度：图游走 RWR 相似度 + 语义嵌入余弦（均来自图结构统计）
    fn read_affinity(
        &self,
        probe: &PresenceProbe,
        reasoner: &SymbolicReasoner,
    ) -> (Option<ChannelEvidence>, Option<ChannelEvidence>) {
        let subject = probe.subject.as_str();
        let object = probe.object.as_str();
        let walk = reasoner
            .walks()
            .similarity(subject, object)
            .filter(|score| *score >= self.tuning.walk_floor)
            .map(|score| {
                let path = reasoner
                    .walks()
                    .best_path(subject, object, 3)
                    .map(|path| {
                        format!(
                            "；路径 {}",
                            crate::graph_walk_embedding::GraphWalkEmbedding::render_path(
                                subject, &path
                            )
                        )
                    })
                    .unwrap_or_default();
                ChannelEvidence {
                    channel: "图游走",
                    score,
                    detail: format!("RWR 结构相似度 {:.2}{}", score, path),
                }
            });
        let embedding = {
            let score = reasoner.embedding().similarity(subject, object);
            if score >= self.tuning.embedding_floor && score > 0.0 {
                Some(ChannelEvidence {
                    channel: "语义嵌入",
                    score,
                    detail: format!("特征向量余弦 {:.2}", score),
                })
            } else {
                None
            }
        };
        (walk, embedding)
    }

    // ------------------------- 推测的确认 / 否认 -------------------------

    fn register_guess(
        &mut self,
        case_id: usize,
        statement: &str,
        key: &str,
        score: f64,
        evidence: &DistributionalReading,
        atom_display: Option<String>,
    ) -> usize {
        let id = self.next_guess_id;
        self.next_guess_id += 1;
        if self.guesses.len() >= self.tuning.max_guesses {
            if let Some(position) = self
                .guesses
                .iter()
                .position(|guess| guess.status != GuessStatus::Pending)
            {
                self.guesses.remove(position);
            } else {
                self.guesses.pop_front();
            }
        }
        self.guesses.push_back(ContrastGuess {
            id,
            case_id,
            statement: statement.to_string(),
            key: key.to_string(),
            evidence: summarize_evidence(evidence),
            atom_display,
            score,
            status: GuessStatus::Pending,
        });
        id
    }

    /// 确认推测：写入会话事实（不修改知识图谱），此后逻辑通道也能证实
    pub fn confirm_guess(
        &mut self,
        id: usize,
        reasoner: &mut SymbolicReasoner,
    ) -> Option<String> {
        let index = self
            .guesses
            .iter()
            .position(|guess| guess.id == id && guess.status == GuessStatus::Pending)?;
        let statement = self.guesses[index].statement.clone();
        let key = self.guesses[index].key.clone();
        let report = reasoner.assert_nl_fact(&statement);
        self.guesses[index].status = GuessStatus::Confirmed;
        self.confirmed_keys.insert(key);
        self.stats.confirmed += 1;
        Some(format!(
            "已确认联想推测 #{}：{}（写入会话事实，逻辑通道从此可证实；如需写入知识图谱请用「记住：…」导入）\n{}",
            id, statement, report
        ))
    }

    /// 否认推测：加入联想通道屏蔽集（不再提出），同时升级为会话级否证
    /// （符号通道此后会明确反驳同一陈述）
    pub fn refute_guess(&mut self, id: usize) -> Option<String> {
        let index = self
            .guesses
            .iter()
            .position(|guess| guess.id == id && guess.status == GuessStatus::Pending)?;
        let statement = self.guesses[index].statement.clone();
        let key = self.guesses[index].key.clone();
        let atom_display = self.guesses[index].atom_display.clone();
        self.guesses[index].status = GuessStatus::Refuted;
        self.refuted_keys.insert(key);
        let negation_note = if let Some(atom) = atom_display {
            self.refuted_atoms.insert(atom);
            "；同时登记为会话级否证，逻辑通道此后会判同一陈述为不成立"
        } else {
            ""
        };
        self.stats.refuted += 1;
        Some(format!(
            "已否认联想推测 #{}：{}。该推测已加入联想通道屏蔽集，之后不再提出{}。",
            id, statement, negation_note
        ))
    }

    // ------------------------- 对比追问（主动澄清） -------------------------

    /// 取最旧的待定推测生成澄清问题：把"联想直觉"逆转为系统主动提问，
    /// 用户的是/否回答直接驱动推测生命周期（确认→会话事实 / 否认→会话否证）
    pub fn next_followup(&self) -> Option<ContrastFollowup> {
        let guess = self
            .guesses
            .iter()
            .find(|guess| guess.status == GuessStatus::Pending)?;
        let evidence = if guess.evidence.is_empty() {
            "（无更细证据）".to_string()
        } else {
            guess.evidence.clone()
        };
        let question = format!(
            "【对比追问·待定推测 #{}】「{}」\n  联想依据：{}\n  逻辑通道尚无证明，也无反例。这在您掌握的情况里属实吗？\n  请直接回答「是」或「不是」（输入「跳过」换下一条；「待定推测」查看全部清单）。",
            guess.id, guess.statement, evidence
        );
        Some(ContrastFollowup {
            guess_id: guess.id,
            question,
        })
    }

    // ------------------------- 渲染 -------------------------

    /// 完整案例报告：两通道证据 + 仲裁结论 + 后续操作提示
    pub fn render_case(&self, case: &ContrastCase) -> String {
        let mut output = String::new();
        output.push_str(&format!("═══ 神经符号对比 #{} ═══\n", case.id));
        output.push_str(&format!("陈述：{}\n", case.statement));
        output.push_str(&format!("形式化：{}\n", case.formalization));

        output.push_str("\n【B·符号/演绎通道】（蕴含判定 + SLD 归结反驳）\n");
        if !case.symbolic.formalized {
            output.push_str("  未能形式化为可判定命题。\n");
        } else {
            if case.symbolic.verdict_labels.is_empty() {
                output.push_str("  蕴含判定：（无可判定命题）\n");
            }
            for label in &case.symbolic.verdict_labels {
                output.push_str(&format!("  蕴含判定：{}\n", label));
            }
            for (index, reason) in case.symbolic.reasons.iter().enumerate() {
                output.push_str(&format!("  说明：{}\n", reason));
                if index >= 2 {
                    break;
                }
            }
            if case.symbolic.proved {
                output.push_str(&format!(
                    "  结论：已证实（证明置信度 {:.2}）\n",
                    case.symbolic.confidence
                ));
            } else if case.symbolic.contradicted {
                output.push_str(&format!(
                    "  结论：已反驳（反驳置信度 {:.2}）\n",
                    case.symbolic.confidence
                ));
            } else {
                output.push_str("  结论：未知（无反例，也无证明）\n");
            }
            for step in case.symbolic.proof_steps.iter().take(4) {
                output.push_str(&format!("  证明：{}\n", step));
            }
            for counter in case.symbolic.counter_evidence.iter().take(3) {
                output.push_str(&format!("  反证：{}\n", counter));
            }
        }

        output.push_str("\n【A·联想/分布通道】（软逻辑 ∩ 关联规则 ∩ 图游走；无神经网络）\n");
        if case.distributional.blocked {
            output.push_str(&format!("  {}\n", case.distributional.note));
        } else if !case.distributional.probed {
            output.push_str(&format!("  {}\n", case.distributional.note));
        } else {
            for evidence in [
                case.distributional.soft.as_ref(),
                case.distributional.association.as_ref(),
                case.distributional.walk.as_ref(),
                case.distributional.embedding.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                output.push_str(&format!(
                    "  {}：{}（{:.2}）\n",
                    evidence.channel, evidence.detail, evidence.score
                ));
            }
            if !case.distributional.note.is_empty() {
                output.push_str(&format!("  {}\n", case.distributional.note));
            }
            output.push_str(&format!(
                "  合并联想分：{:.2}（阈值 {:.2}）\n",
                case.distributional.score, self.tuning.min_score
            ));
        }

        output.push_str(&format!("\n【仲裁】{}\n", case.verdict.label()));
        output.push_str(&case.answer);
        output
    }

    /// 简版对比结论：用于常规问答路径的自动附加（控制篇幅）
    pub fn render_case_brief(&self, case: &ContrastCase) -> String {
        let mut output = format!(
            "【神经符号对比 #{}·{}】",
            case.id,
            case.verdict.label()
        );
        match case.verdict {
            ContrastVerdict::Agree => {
                output.push_str(&format!(
                    "逻辑与联想两通道一致；联想分 {:.2}。",
                    case.distributional.score
                ));
            }
            ContrastVerdict::SymbolicOnly => {
                output.push_str("由逻辑推导证实；联想通道证据不足。");
            }
            ContrastVerdict::DistributionalOnly => {
                output.push_str(&format!(
                    "逻辑未证实，联想通道倾向为真（联想分 {:.2}）。",
                    case.distributional.score
                ));
                if let Some(id) = case.guess_id {
                    output.push_str(&format!(
                        "可「对比确认 {}」采信为会话事实，或「对比否认 {}」排除。",
                        id, id
                    ));
                }
            }
            ContrastVerdict::Conflict => {
                output.push_str("联想通道倾向为真，但逻辑通道明确反驳；按符号优先判假。");
            }
            ContrastVerdict::BothAbstain => {
                output.push_str("两通道均无法判断；可「规则：…」教规则或「记住：…」补事实。");
            }
        }
        output
    }

    pub fn pending_report(&self) -> String {
        let pending: Vec<&ContrastGuess> = self
            .guesses
            .iter()
            .filter(|guess| guess.status == GuessStatus::Pending)
            .collect();
        if pending.is_empty() {
            return "当前没有待定联想推测。可输入「对比：<陈述句>」生成新的双通道对比。".to_string();
        }
        let mut output = String::from("待定联想推测（逻辑未证实；确认后成为会话事实）：\n");
        for guess in pending {
            output.push_str(&format!(
                "  #{}（联想分 {:.2}，来自案例 #{}）：{}\n",
                guess.id, guess.score, guess.case_id, guess.statement
            ));
        }
        output.push_str("  操作：对比确认 <编号> / 对比否认 <编号>");
        output
    }

    pub fn calibration_report(&self) -> String {
        let stats = &self.stats;
        if stats.total == 0 {
            return "神经符号对比·会话校准：尚无案例。可输入「对比：<陈述句>」开始（例如：对比：张三是李四的师傅）。"
                .to_string();
        }
        let percent = |value: usize| value as f64 * 100.0 / stats.total as f64;
        let mut output = String::new();
        output.push_str("神经符号对比·会话校准报告（全程纯符号，无神经网络）：\n");
        output.push_str(&format!("  累计会话 {} 个（跨会话持久化于对话记忆）\n", stats.sessions));
        output.push_str(&format!(
            "  案例总数 {}：双通道一致 {}（{:.0}%），仅符号证实 {}，仅联想推测 {}，通道冲突 {}，双通道未知 {}\n",
            stats.total,
            stats.agree,
            percent(stats.agree),
            stats.symbolic_only,
            stats.distributional_only,
            stats.conflict,
            stats.both_abstain
        ));
        output.push_str(&format!(
            "  联想通道：探测 {} 次，被屏蔽集拦截 {} 次，当前待定推测 {} 条\n",
            stats.probed,
            stats.blocked,
            self.pending_guess_count()
        ));
        output.push_str(&format!(
            "  推测反馈：确认 {} 条（写入会话事实），否认 {} 条（已屏蔽，联想通道不再重提）\n",
            stats.confirmed, stats.refuted
        ));
        output.push_str(&format!(
            "  一致率 {:.0}%：越高说明联想直觉与符号逻辑越吻合；冲突率 {:.0}% 是幻觉风险的可解释上界\n",
            percent(stats.agree),
            percent(stats.conflict)
        ));
        output.push_str("  最近案例：\n");
        for case in self.cases.iter().rev().take(6) {
            output.push_str(&format!(
                "    #{} [{}] {}\n",
                case.id,
                case.verdict.label(),
                case.statement
            ));
        }
        if !self.refuted_keys.is_empty() {
            let mut keys: Vec<&String> = self.refuted_keys.iter().collect();
            keys.sort();
            let shown: Vec<String> = keys
                .iter()
                .take(5)
                .map(|key| {
                    key.rsplit("::")
                        .next()
                        .unwrap_or(key.as_str())
                        .to_string()
                })
                .collect();
            output.push_str(&format!(
                "  已屏蔽推测 {} 条，例如：{}\n",
                keys.len(),
                shown.join("；")
            ));
        }
        output.trim_end().to_string()
    }

    // ------------------------- 仲裁文案 -------------------------

    fn compose_answer(
        verdict: ContrastVerdict,
        symbolic: &SymbolicReading,
        distributional: &DistributionalReading,
        guess_id: Option<usize>,
    ) -> String {
        match verdict {
            ContrastVerdict::Agree if symbolic.proved => {
                let affinity = distributional_score_note(distributional);
                format!(
                    "【结论】该陈述成立（逻辑已证实，联想通道独立支持{}）。双通道一致意味着它不是巧合共现，也不是无据猜测。",
                    affinity
                )
            }
            ContrastVerdict::Agree => {
                "【结论】该陈述不成立（逻辑明确反驳，联想通道同样不支持）。双通道一致否定。".to_string()
            }
            ContrastVerdict::SymbolicOnly => {
                let hint = if distributional.score > 0.0 {
                    format!("（联想分仅 {:.2}）", distributional.score)
                } else {
                    "（联想通道没有可用证据）".to_string()
                };
                format!(
                    "【结论】该陈述成立（仅符号证实）。逻辑推导给出了证明，{}——这类判断正是纯符号推理相对分布式模型的优势区。",
                    hint
                )
            }
            ContrastVerdict::DistributionalOnly => {
                let guess = guess_id
                    .map(|id| {
                        format!(
                            "\n  已登记为待定推测 #{}：可「对比确认 {}」采信为会话事实，或「对比否认 {}」永久排除。",
                            id, id, id
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "【结论】逻辑通道暂时无法证实；联想通道倾向为真（联想分 {:.2}，合并了软逻辑概率、关联规则置信度与图上游走契合度）。\n  这是可解释的推测而非事实：每条依据都可回溯；若采信请显式确认。{}",
                    distributional.score, guess
                )
            }
            ContrastVerdict::Conflict => {
                "【结论】通道冲突：联想通道倾向为真，但逻辑通道明确反驳（反证见上）。按符号优先原则判定为假，并标记该联想为争议——这正是「直觉可能出错、逻辑负责纠偏」的神经符号对比价值。".to_string()
            }
            ContrastVerdict::BothAbstain => {
                "【结论】两通道都无法判断。可尝试：用「规则：如果…那么…」教一条规则；用「记住：…」补充会话事实；或用「教学：…」给出概率知识后再问。".to_string()
            }
        }
    }
}

impl Default for ContrastArbiter {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------- 内部工具 -------------------------

fn case_key(statement: &str, atom: &Atom) -> String {
    format!("{}::{}", statement.trim(), atom.display())
}

fn contains_negation(statement: &str) -> bool {
    NEGATION_MARKERS
        .iter()
        .any(|marker| statement.contains(marker))
}

fn presence_probe(atom: &Atom) -> Option<PresenceProbe> {
    let names: Vec<&str> = atom.terms.iter().map(|term| term.name()).collect();
    match names.as_slice() {
        [subject, object] if &*atom.predicate == TYPE_STATEMENT => Some(PresenceProbe {
            soft_query_predicate: TYPE_PREDICATE,
            subject: subject.to_string(),
            target: object.to_string(),
            object: object.to_string(),
        }),
        [subject, object] => Some(PresenceProbe {
            soft_query_predicate: HAS_PREDICATE,
            subject: subject.to_string(),
            target: atom.predicate.to_string(),
            object: object.to_string(),
        }),
        _ => None,
    }
}

fn render_proof_node(node: &ProofNode) -> String {
    match &node.source {
        ProofSource::Fact => format!("{}（直接事实）", node.atom.display()),
        ProofSource::Rule { source } => format!(
            "{}（规则：{}；置信度 {:.2}）",
            node.atom.display(),
            source,
            node.confidence
        ),
    }
}

/// 把联想通道各证据压成一行，供推测条目与澄清追问使用
fn summarize_evidence(reading: &DistributionalReading) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(evidence) = &reading.soft {
        parts.push(format!("{} P={:.2}", evidence.channel, evidence.score));
    }
    if let Some(evidence) = &reading.association {
        parts.push(format!("{} {:.2}", evidence.channel, evidence.score));
    }
    if let Some(evidence) = &reading.walk {
        parts.push(format!("{} {:.2}", evidence.channel, evidence.score));
    }
    if let Some(evidence) = &reading.embedding {
        parts.push(format!("{} {:.2}", evidence.channel, evidence.score));
    }
    if parts.is_empty() {
        format!("合并联想分 {:.2}", reading.score)
    } else {
        format!("{}（合并 {:.2}）", parts.join("；"), reading.score)
    }
}

fn distributional_score_note(reading: &DistributionalReading) -> String {
    if reading.score > 0.0 {
        format!("，联想分 {:.2}", reading.score)
    } else {
        String::new()
    }
}

impl SymbolicReading {
    fn note_push(&mut self, note: &str) {
        self.reasons.push(note.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prolog_engine::PrologFact;
    use std::collections::HashMap;

    fn rel(from: &str, predicate: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: predicate.to_string(),
            to: to.to_string(),
        }
    }

    fn typed_entity(name: &str, etype: &str) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: HashMap::new(),
        }
    }

    /// 门派小图谱：6 位人物 + 3 个国家；
    /// 国籍只与部分人物共现，用于让关联规则「关系：师傅 ⇒ 关系：国籍」成立
    fn lineage_facts() -> Vec<PrologFact> {
        vec![
            typed_entity("张三", "人物"),
            typed_entity("李四", "人物"),
            typed_entity("王五", "人物"),
            typed_entity("赵六", "人物"),
            typed_entity("陈七", "人物"),
            typed_entity("周八", "人物"),
            typed_entity("中国", "国家"),
            typed_entity("美国", "国家"),
            typed_entity("法国", "国家"),
            rel("张三", "师傅", "李四"),
            rel("李四", "师傅", "王五"),
            rel("王五", "师傅", "赵六"),
            rel("陈七", "师傅", "周八"),
            rel("张三", "国籍", "中国"),
            rel("王五", "国籍", "美国"),
            rel("陈七", "国籍", "法国"),
        ]
    }

    fn reasoner() -> SymbolicReasoner {
        let mut reasoner =
            SymbolicReasoner::new(ReasoningConfig::default(), HashMap::new());
        reasoner.observe_facts(&lineage_facts());
        reasoner
    }

    #[test]
    fn arbitration_matrix_covers_all_channel_combinations() {
        let symbolic = |proved: bool, contradicted: bool| SymbolicReading {
            proved,
            contradicted,
            ..SymbolicReading::default()
        };
        let distributional = |supported: bool| DistributionalReading {
            supported,
            ..DistributionalReading::default()
        };
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(true, false), &distributional(true)),
            ContrastVerdict::Agree
        );
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(true, false), &distributional(false)),
            ContrastVerdict::SymbolicOnly
        );
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(false, true), &distributional(true)),
            ContrastVerdict::Conflict
        );
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(false, true), &distributional(false)),
            ContrastVerdict::Agree
        );
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(false, false), &distributional(true)),
            ContrastVerdict::DistributionalOnly
        );
        assert_eq!(
            ContrastArbiter::arbitrate(&symbolic(false, false), &distributional(false)),
            ContrastVerdict::BothAbstain
        );
    }

    #[test]
    fn factual_statement_gets_dual_channel_agreement() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的师傅是王五", &reasoner);
        assert!(case.symbolic.proved, "直接事实应被符号通道证实");
        assert_eq!(case.verdict, ContrastVerdict::Agree);
        assert!(case.answer.contains("该陈述成立"));
        assert!(case.distributional.supported, "直接事实的谓词存在性必然成立");
        let rendered = arbiter.render_case(&case);
        assert!(rendered.contains("B·符号/演绎通道"));
        assert!(rendered.contains("A·联想/分布通道"));
        assert!(rendered.contains("双通道一致"));
    }

    #[test]
    fn symbolic_only_when_association_channel_is_too_weak() {
        let reasoner = reasoner();
        let mut tuning = ContrastTuning::default();
        tuning.min_score = 0.99;
        let mut arbiter = ContrastArbiter::with_tuning(tuning);
        let case = arbiter.contrast_statement("李四的师傅是王五", &reasoner);
        assert!(case.symbolic.proved);
        assert!(!case.distributional.supported);
        assert_eq!(case.verdict, ContrastVerdict::SymbolicOnly);
        assert!(case.answer.contains("仅符号证实"));
    }

    #[test]
    fn association_guess_becomes_pending_distributional_only_case() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        assert!(!case.symbolic.proved, "图谱中没有李四的国籍记录");
        assert!(!case.symbolic.contradicted);
        assert!(
            case.distributional.supported,
            "关联规则应能从「师傅」推测「国籍」谓词存在性"
        );
        assert_eq!(case.verdict, ContrastVerdict::DistributionalOnly);
        let guess_id = case.guess_id.expect("仅联想推测应登记待定推测");
        assert_eq!(arbiter.pending_guess_count(), 1);
        assert!(case.answer.contains("逻辑通道暂时无法证实"));
        assert!(arbiter.pending_report().contains(&format!("#{}", guess_id)));
    }

    #[test]
    fn conflict_when_logic_refutes_but_association_supports() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("张三的国籍是美国", &reasoner);
        assert!(case.symbolic.contradicted, "张三的国籍已有相反记录");
        assert!(case.distributional.supported);
        assert_eq!(case.verdict, ContrastVerdict::Conflict);
        assert!(case.answer.contains("通道冲突"));
        assert!(case.guess_id.is_none(), "冲突案例不登记可采信的推测");
    }

    #[test]
    fn confirm_guess_turns_speculation_into_provable_session_fact() {
        let mut reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        let guess_id = case.guess_id.expect("应有待定推测");
        let confirmation = arbiter
            .confirm_guess(guess_id, &mut reasoner)
            .expect("确认应成功");
        assert!(confirmation.contains("已确认联想推测"));
        assert_eq!(arbiter.pending_guess_count(), 0);
        assert_eq!(arbiter.stats().confirmed, 1);

        let again = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        assert!(
            again.symbolic.proved,
            "会话事实应让符号通道（归结）随即能证实"
        );
    }

    #[test]
    fn refuted_guess_is_blocked_in_distributional_channel() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        let guess_id = case.guess_id.expect("应有待定推测");
        let denial = arbiter.refute_guess(guess_id).expect("否认应成功");
        assert!(denial.contains("已否认联想推测"));
        assert_eq!(arbiter.stats().refuted, 1);
        assert_eq!(arbiter.refuted_atom_count(), 1);

        let again = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        assert!(again.distributional.blocked);
        assert!(!again.distributional.supported);
        // 否认同时进入符号通道成为会话级否证，
        // 因此不是"双通道未知"，而是"双通道一致否定"
        assert!(again.symbolic.contradicted);
        assert_eq!(again.verdict, ContrastVerdict::Agree);
        assert!(again.answer.contains("不成立"));
        assert!(again.guess_id.is_none(), "被否认的推测不再登记");
    }

    #[test]
    fn refuted_atom_is_negated_by_symbolic_channel() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        arbiter.refute_guess(case.guess_id.expect("应有待定推测"));

        let again = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        assert!(again.symbolic.contradicted, "用户否认应作为否证被逻辑通道采用");
        assert!(
            again
                .symbolic
                .counter_evidence
                .iter()
                .any(|line| line.contains("用户")),
            "反证应标注来自用户否认"
        );
        assert!(
            again
                .symbolic
                .verdict_labels
                .iter()
                .any(|label| label.contains("用户否认")),
            "判定标签应说明这是会话记忆中的否证"
        );
    }

    #[test]
    fn knowledge_graph_fact_outranks_user_refutation_memory() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        // 构造"用户曾否认的事实恰好与图谱一致"的持久化记忆：
        // 图谱已有证明时，符号通道应优先采信图谱，而不是被会话否证覆盖
        let snapshot = ContrastSnapshot {
            refuted_atoms: vec!["师傅(李四, 王五)".to_string()],
            next_case_id: 1,
            next_guess_id: 1,
            ..ContrastSnapshot::default()
        };
        assert!(arbiter.restore_snapshot(&snapshot));
        let fact = arbiter.contrast_statement("李四的师傅是王五", &reasoner);
        assert!(fact.symbolic.proved, "图谱事实仍然可证");
        assert!(!fact.symbolic.contradicted, "图谱证明优先于会话否证");
    }

    #[test]
    fn followup_question_targets_oldest_pending_guess_with_evidence() {
        let mut reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let first = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        let second = arbiter.contrast_statement("李四的国籍是日本", &reasoner);
        let first_id = first.guess_id.expect("第一条应有待定推测");
        let second_id = second.guess_id.expect("第二条应有待定推测");
        assert_ne!(first_id, second_id);

        let followup = arbiter.next_followup().expect("应有可追问的推测");
        assert_eq!(followup.guess_id, first_id, "追问应取最旧的待定推测");
        assert!(followup.question.contains("是"));
        assert!(followup.question.contains("不是"));
        assert!(followup.question.contains("联想依据"), "追问应附上依据");
        assert!(
            followup.question.contains("关联规则") || followup.question.contains("软逻辑"),
            "追问应给出联想通道的具体证据来源"
        );

        arbiter.confirm_guess(first_id, &mut reasoner);
        let next = arbiter.next_followup().expect("仍有第二条可追问");
        assert_eq!(next.guess_id, second_id);
        arbiter.refute_guess(second_id);
        assert!(arbiter.next_followup().is_none(), "推测全部结算后不再追问");
    }

    #[test]
    fn snapshot_roundtrip_preserves_guess_lifecycle_and_screen() {
        let mut reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let first = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        let second = arbiter.contrast_statement("李四的国籍是日本", &reasoner);
        let third = arbiter.contrast_statement("李四的国籍是法国", &reasoner);
        let first_id = first.guess_id.expect("应有推测");
        let second_id = second.guess_id.expect("应有推测");
        let third_id = third.guess_id.expect("应有推测");
        arbiter.confirm_guess(first_id, &mut reasoner);
        arbiter.refute_guess(second_id);

        let snapshot = arbiter.snapshot();
        let mut restored = ContrastArbiter::new();
        assert!(restored.restore_snapshot(&snapshot));
        assert_eq!(restored.stats().sessions, 2);
        assert_eq!(restored.stats().total, 3);
        assert_eq!(restored.stats().confirmed, 1);
        assert_eq!(restored.stats().refuted, 1);
        assert_eq!(restored.pending_guess_count(), 1);
        assert_eq!(restored.refuted_atom_count(), 1);
        assert_eq!(
            restored.next_followup().map(|item| item.guess_id),
            Some(third_id),
            "跨会话恢复后待定推测仍可追问"
        );

        // 恢复后的编号继续递增；对恢复出的待定推测仍可结算
        assert!(
            restored.snapshot().next_guess_id > third_id,
            "推测编号应跨会话延续"
        );
        assert!(restored.refute_guess(third_id).is_some());
        assert_eq!(restored.refuted_atom_count(), 2);
        assert!(restored.next_followup().is_none());
    }

    #[test]
    fn contrast_snapshot_persists_inside_dialogue_state_file() {
        let dir = tempfile::tempdir().expect("临时目录");
        let path = dir.path().join("dialogue_state.json");
        let path_str = path.to_str().expect("路径");

        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        assert!(case.guess_id.is_some());
        arbiter.refute_guess(case.guess_id.expect("应有推测"));
        let pending = arbiter.contrast_statement("李四的国籍是日本", &reasoner);
        let pending_id = pending.guess_id.expect("应有待定推测");

        let saver = SymbolicReasoner::new(ReasoningConfig::default(), HashMap::new());
        saver
            .save_dialogue_state_with_contrast(path_str, Some(&arbiter.snapshot()))
            .expect("应能保存");

        let loaded = ContrastArbiter::load_snapshot_from_dialogue_state(path_str)
            .expect("应能从对话记忆中读回对比快照");
        let mut restored = ContrastArbiter::new();
        assert!(restored.restore_snapshot(&loaded));
        assert_eq!(restored.pending_guess_count(), 1);
        assert_eq!(restored.next_followup().map(|item| item.guess_id), Some(pending_id));
        assert_eq!(restored.stats().refuted, 1);

        // DialogueSnapshot 还应能被 load 路径完整消费（含对比字段）
        let mut loader = SymbolicReasoner::new(ReasoningConfig::default(), HashMap::new());
        let (ok, contrast) = loader.load_dialogue_state_with_contrast(path_str);
        assert!(ok);
        assert!(contrast.is_some());
        assert!(loader.load_dialogue_state(path_str));
    }

    #[test]
    fn dialogue_state_without_contrast_field_still_loads() {
        let legacy = r#"{"focus_entities":["张三"],"last_predicate":"师傅","history":["张三是谁"]}"#;
        let snapshot: DialogueSnapshot =
            serde_json::from_str(legacy).expect("缺少 contrast 字段的对话记忆应能反序列化");
        assert!(snapshot.contrast.is_none());
        assert_eq!(snapshot.focus_entities, vec!["张三".to_string()]);
    }

    #[test]
    fn memory_summary_reports_cross_session_calibration() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        assert!(arbiter.memory_summary().contains("空"));
        arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        let snapshot = arbiter.snapshot();
        let mut restored = ContrastArbiter::new();
        assert!(restored.restore_snapshot(&snapshot));
        let summary = restored.memory_summary();
        assert!(summary.contains("2 个会话"));
        assert!(summary.contains("1 个案例"));
    }

    #[test]
    fn negation_is_not_probed_by_distributional_channel() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        let case = arbiter.contrast_statement("李四不是王五的师傅", &reasoner);
        assert!(!case.distributional.probed);
        assert!(matches!(
            case.verdict,
            ContrastVerdict::Agree | ContrastVerdict::BothAbstain | ContrastVerdict::SymbolicOnly
        ));
    }

    #[test]
    fn calibration_report_tracks_session_verdicts() {
        let reasoner = reasoner();
        let mut arbiter = ContrastArbiter::new();
        assert!(arbiter.calibration_report().contains("尚无案例"));
        arbiter.contrast_statement("李四的师傅是王五", &reasoner);
        arbiter.contrast_statement("李四的国籍是德国", &reasoner);
        arbiter.contrast_statement("张三的国籍是美国", &reasoner);
        let report = arbiter.calibration_report();
        assert!(report.contains("案例总数 3"));
        assert!(report.contains("通道冲突 1"));
        assert!(report.contains("仅联想推测 1"));
    }
}

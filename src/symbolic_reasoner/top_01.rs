use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::abduction::{
    self, AbductionEngine, AbductionReport, ClarificationReply, ClarificationSession, Hypothesis,
    HypothesisKind, Scenario,
};
use crate::alignment_memory::{AlignmentKind, AlignmentMemory, AlignmentScope, LearnOutcome};
use crate::answer_plan::{AnswerPlan, ClaimEvidence, PlanClaim, StyleDirective, MAX_PLAN_CLAIMS};
use crate::association_miner::{display_item, AssociationMiner, AssociationRule};
use crate::batch_conflict::{
    self, BatchClaim, BatchMode, BatchOptions, BatchParseOutcome, BatchPendingTurn, BatchQuestion,
};
use crate::belief_revision::{BeliefState, ChangeOp, ChangeSet, RevisionConstraints};
use crate::cache::RevisionLru;
use crate::canonical::{self, DecodeError, DecodeReceipt};
use crate::causal_reasoning::{CausalAnalyzer, CausalGraph, CounterfactualDelta};
use crate::causal_tokenization::{attribute_tokens, AnchorKind, CausalTokenizer, MAX_TOKEN_CHARS};
use crate::chinese_reasoning_nlu::{self, LessonKind};
use crate::completion_episode::{CandidateStatus, CompletionLedger, CompletionQuestion, RuleScope};
use crate::confidence::{ClaimEnvelope, ConfidenceDecision, DecisionRecord, EnvelopeGrade};
use crate::constructive::{ConstructiveReasoner, FactKey, ObservedGraph, VirtualFact};
use crate::contextual_rag::{
    self, EvidenceAssembly, EvidenceKey, FrameInput, HoloEvidence, QueryFrame,
};
use crate::contextual_reference::{SenseCard, SenseSource};
use crate::contextual_segmentation::{
    ContextualSegmenter, ReadingKind, ReadingMemory, SegmentOptions,
};
use crate::conversational_graph::{
    ConversationalGraph, GraphCommand, GraphFact, GraphReply, MatchOptions, PatternError,
    PatternOutcome,
};
use crate::default_logic::{self, DefaultTheory, Literal};
use crate::dense_retrieval::{DenseRetriever, EvidenceSearch, SearchBudget, SearchConstraints};
use crate::derivation::Derivation;
use crate::dialogue_frame::{self, AskAttributeFrame, FrameMemory, OntologyView};
use crate::discourse::DiscoursePlanner;
use crate::entity_linker::{EntityLinker, LinkMethod, LinkedMention};
use crate::epistemic_turn::{EpistemicTurnBook, TurnEnvelope, TurnPolicy};
use crate::graph_diagnosis::diagnose_failure;
use crate::graph_ledger;
use crate::graph_question::{CompiledGraphQuestion, GraphQuestionKind, PatternLexicon};
use crate::graph_repair::{
    parse_repair_intent, suggest_repairs, RepairHints, RepairIntent, RepairLedger, RepairSuggestion,
};
use crate::graph_walk_embedding::GraphWalkEmbedding;
use crate::holographic_memory::HolographicMemory;
use crate::joint_parse::{
    joint_parse, verify_frames, FrameVerification, JointContext, JointFrame, JointParse,
    JointParseOptions,
};
use crate::knowledge_layers::{KnowledgeLayerManifest, LayerEntry};
use crate::law_book::{LawBook, LawStatus};
use crate::lexical_memory::{LexicalMemory, ObservationSummary, TypeExpectation};
use crate::lexical_trace::{
    self, LexicalTrace, ReadFailureKind, ReadStage, ReadTraceLedger, TraceFidelity, TraceReceipt,
};
use crate::link_prediction::{HornRule, LinkPredictor, Prediction, RuleTemplate, Triple};
use crate::lookahead::{self, BranchEvidence, BranchKey, BranchSeal, LookaheadBoard};
use crate::markov_logic::{MarkovLogicEngine, SoftFact, WeightedRule};
use crate::mention_lexicon::MentionLexicon;
use crate::neuro_symbolic_contrast::ContrastSnapshot;
use crate::ontology_alignment::{AlignmentDecision, AlignmentState, OntologyAligner, TermKind};
use crate::ontology_reasoning::{OntologyCapsule, OntologyLesson, OntologyProgram};
use crate::path_memory::{EdgeSource, PathPredicate, TraversalGraph, TraversalOptions};
use crate::pattern_closure::{evaluate_closure, ClosureBudget, ClosureOp, ClosureOutcome};
use crate::pending_question::{
    self, AnswerBinding, PendingHole, PendingQuestion, PendingQuestionLedger,
};
use crate::perf_ledger::PerfLedger;
use crate::plugin::{PluginOutcome, PluginRuntime, RecipeDirection};
use crate::predicate_registry::PredicateRegistry;
use crate::prediction_draft::PredictionDrafts;
use crate::foil_refinement::FoilLedger;
use crate::predictive_elaboration::{PredictionGuard, PredictiveElaboration};
use crate::probabilistic_query_intent::{
    Claim, EpistemicMode, EpistemicTarget, ProbabilisticQueryIntent,
};
use crate::problog::{ProblogAnswer, ProblogEngine};
use crate::prolog_engine::PrologFact;
use crate::proof_memory::{
    canonical_key, ProofFootprint, ProofMemory, ProofMemoryAccess, ProofMemoryHitAccess,
    ProofRecord, ProofSearchCost, ProofStatus,
};
use crate::query_anchor::{self, AnchorEvidence, TypeFit};
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::query_rewrite::{self, RelationLexicon};
use crate::reading::{
    Reading, RepairKind, Residual, ResidualKind, ResidualSeverity, ResidualStage, TextSpan,
};
use crate::relation_frame::RelationFrame;
use crate::resolution_engine::{
    render_term, Atom, Clause, DerivedFact, FailureFrontier, ProofNode, ProofSource,
    ResolutionEngine, SearchOutcome, SearchStats, SearchStatus, Solution, Term,
};
use crate::revision_guard::{GuardPlan, RevisionGuard};
use crate::rewrite_book::RewriteBook;
use crate::rewrite_retrieval::{
    self, Decision as RewriteRetrievalDecision, HypothesisCandidate, HypothesisSource,
};
use crate::schema_view::SchemaView;
use crate::semantic_embedding::SymbolicEmbedding;
use crate::semantic_entailment::{self, EntailmentResult, Knowledge, PropositionParser, Verdict};
use crate::semantic_monitor::{
    ControlMode, SearchAdvice, SemanticMonitor, SemanticObservation, TurnOutcome, WindowMetrics,
};
use crate::slot_ledger::{
    AssertionStrength, SlotAssertion, SlotKey, SlotLedger, SlotPolicy, SlotTransition,
};
use crate::stage_bus::BeliefLedger;
use crate::symbolic_generator::{
    GenerationCursor, GenerationOrder, GenerationOverlay, ProfileClause, SymbolicGenerator,
};
use crate::temporal::{
    self, validity_from_bounds, TemporalConstraint, UnknownTimePolicy, Validity,
};
use crate::view_render::ProofViewNode;
use crate::{NlpConfig, ReasoningConfig, SystemConfig};

// 归结引擎内部谓词：has(E, 谓词名) / type(E, 类型名)，用于承载关联规则的物品
const PROOF_HAS: &str = "__具有__";
const PROOF_TYPE: &str = "__类型__";

/// 在原始问句中定位词面，返回 Unicode 字符区间（供读法残差定位）。
fn term_span(question: &str, term: &str) -> Option<TextSpan> {
    if term.is_empty() {
        return None;
    }
    let byte_index = question.find(term)?;
    let start = question[..byte_index].chars().count();
    Some(TextSpan::new(start, start + term.chars().count()))
}

fn insert_type_hint(hints: &mut BTreeMap<String, BTreeSet<String>>, entity: &str, etype: &str) {
    let entity = entity.trim();
    let etype = etype.trim();
    if entity.is_empty() || etype.is_empty() {
        return;
    }
    hints
        .entry(entity.to_string())
        .or_default()
        .insert(etype.to_string());
}

// 默认逻辑（可反驳常识）的最大接地推导轮数
const DEFAULT_LOGIC_MAX_ROUNDS: usize = 8;

// 语义律闭包的有界扩张：深度与节点数双上限，超限即停止（不静默扩成全集）。
const LAW_CLOSURE_MAX_DEPTH: usize = 8;
const LAW_CLOSURE_MAX_NODES: usize = 512;

// 图谱属性里的有效时间约定字段（与 temporal::TemporalIndex 保持一致）。
const TEMPORAL_FROM_FIELDS: &[&str] =
    &["valid_from", "start_year", "since", "开始年份", "起始年份"];
const TEMPORAL_TO_FIELDS: &[&str] = &["valid_to", "end_year", "until", "结束年份", "终止年份"];

// 会话内纯函数记忆槽容量（提及/链接/谓词）：多轮交替询问时各自留槽，
// 知识版本推进（revision 自增）时整体失效；容量按条目数确定，LRU 序可审计。
const MEMO_CAPACITY: usize = 16;

// 轮末预测草稿容量：只缓存焦点实体/谓词的少量结果，纪元推进时整体失效。
const PREDICTION_DRAFT_CAPACITY: usize = 64;

// 会话修订回执容量：只保留最近若干次可撤销写入，历史仍可在对话编年史里查。
const MAX_SESSION_RECEIPTS: usize = 32;

// 中文教学知识类型的展示名
fn lesson_kind_label(kind: LessonKind) -> &'static str {
    match kind {
        LessonKind::DeterministicFact => "确定事实",
        LessonKind::ProbabilisticFact => "概率事实",
        LessonKind::ConditionalRule => "条件规则（如果…那么…）",
        LessonKind::CausalRule => "因果规则（因为…所以…）",
        LessonKind::DefaultRule => "默认规则（通常…除非…，可例外反驳）",
    }
}

// 把链接预测挖掘出的 Horn 规则（组合/逆向/共同邻居/共同主体）转成归结子句
fn clause_from_path_rule(rule: &HornRule) -> Clause {
    let a = || Term::var("A");
    let b = || Term::var("B");
    let c = || Term::var("C");
    let source = format!("链接预测规则 {}", rule.rule_text());
    match rule.template {
        RuleTemplate::Compose => Clause::new(
            Atom::pattern(&rule.head, a(), c()),
            vec![
                Atom::pattern(rule.body1(), a(), b()),
                Atom::pattern(rule.body2().unwrap_or(""), b(), c()),
            ],
            rule.confidence,
            &source,
        ),
        RuleTemplate::Inverse => Clause::new(
            Atom::pattern(&rule.head, b(), a()),
            vec![Atom::pattern(rule.body1(), a(), b())],
            rule.confidence,
            &source,
        ),
        RuleTemplate::CommonNeighbor => Clause::new(
            Atom::pattern(&rule.head, a(), b()),
            vec![
                Atom::pattern(rule.body1(), a(), c()),
                Atom::pattern(rule.body2().unwrap_or(""), b(), c()),
            ],
            rule.confidence,
            &source,
        ),
        RuleTemplate::CommonObject => Clause::new(
            Atom::pattern(&rule.head, a(), b()),
            vec![
                Atom::pattern(rule.body1(), c(), a()),
                Atom::pattern(rule.body2().unwrap_or(""), c(), b()),
            ],
            rule.confidence,
            &source,
        ),
    }
}

fn snapshot_clause_atom(atom: &Atom) -> SnapshotClauseAtom {
    SnapshotClauseAtom {
        predicate: atom.predicate.to_string(),
        terms: atom
            .terms
            .iter()
            .map(|term| match term {
                Term::Var(name) => SnapshotTerm {
                    name: name.to_string(),
                    is_var: true,
                },
                Term::Const(name) => SnapshotTerm {
                    name: name.to_string(),
                    is_var: false,
                },
            })
            .collect(),
    }
}

fn snapshot_clause(clause: &Clause) -> SnapshotClause {
    SnapshotClause {
        head: snapshot_clause_atom(&clause.head),
        body: clause.body.iter().map(snapshot_clause_atom).collect(),
        confidence: clause.confidence,
        source: clause.source.clone(),
    }
}

fn restore_snapshot_clause_atom(atom: &SnapshotClauseAtom) -> Option<Atom> {
    if atom.predicate.is_empty() || atom.terms.is_empty() {
        return None;
    }
    let terms: Vec<Term> = atom
        .terms
        .iter()
        .filter(|term| !term.name.is_empty())
        .map(|term| {
            if term.is_var {
                Term::var(&term.name)
            } else {
                Term::constant(&term.name)
            }
        })
        .collect();
    if terms.len() != atom.terms.len() {
        return None;
    }
    Some(Atom::new(&atom.predicate, terms))
}

fn restore_clause(snapshot: &SnapshotClause) -> Option<Clause> {
    let head = restore_snapshot_clause_atom(&snapshot.head)?;
    let mut body = Vec::with_capacity(snapshot.body.len());
    for atom in &snapshot.body {
        body.push(restore_snapshot_clause_atom(atom)?);
    }
    Some(Clause::new(
        head,
        body,
        snapshot.confidence,
        &snapshot.source,
    ))
}

fn snapshot_hypothesis(hypothesis: &Hypothesis) -> SnapshotHypothesis {
    SnapshotHypothesis {
        atom: snapshot_clause_atom(&hypothesis.atom),
        goal: snapshot_clause_atom(&hypothesis.goal),
        kind: match hypothesis.kind {
            HypothesisKind::MissingFact => "missing_fact".to_string(),
            HypothesisKind::CandidateValue => "candidate_value".to_string(),
        },
        confidence: hypothesis.confidence,
        reasons: hypothesis.reasons.clone(),
        conflicts: hypothesis.conflicts.clone(),
        contradicts_known: hypothesis.contradicts_known,
    }
}

fn restore_hypothesis(snapshot: &SnapshotHypothesis) -> Option<Hypothesis> {
    if !snapshot.confidence.is_finite() {
        return None;
    }
    let kind = match snapshot.kind.as_str() {
        "missing_fact" => HypothesisKind::MissingFact,
        "candidate_value" => HypothesisKind::CandidateValue,
        _ => return None,
    };
    Some(Hypothesis {
        atom: restore_snapshot_clause_atom(&snapshot.atom)?,
        goal: restore_snapshot_clause_atom(&snapshot.goal)?,
        kind,
        confidence: snapshot.confidence.clamp(0.0, 1.0),
        reasons: snapshot.reasons.clone(),
        conflicts: snapshot.conflicts.clone(),
        contradicts_known: snapshot.contradicts_known,
    })
}

fn snapshot_clarification(session: &ClarificationSession) -> SnapshotClarification {
    SnapshotClarification {
        queue: session.queue().iter().map(snapshot_hypothesis).collect(),
        current: session.current().map(snapshot_hypothesis),
        confirmed: session.confirmed.iter().map(snapshot_hypothesis).collect(),
        rejected: session.rejected.iter().map(snapshot_hypothesis).collect(),
        deferred: session.deferred.iter().map(snapshot_hypothesis).collect(),
    }
}

fn restore_clarification(snapshot: &SnapshotClarification) -> ClarificationSession {
    let restore = |items: &[SnapshotHypothesis]| -> Vec<Hypothesis> {
        items.iter().filter_map(restore_hypothesis).collect()
    };
    ClarificationSession::from_parts(
        restore(&snapshot.queue),
        snapshot.current.as_ref().and_then(restore_hypothesis),
        restore(&snapshot.confirmed),
        restore(&snapshot.rejected),
        restore(&snapshot.deferred),
    )
}

fn snapshot_atom(atom: &Atom) -> SnapshotAtom {
    SnapshotAtom {
        predicate: atom.predicate.to_string(),
        terms: atom
            .terms
            .iter()
            .map(|term| term.name().to_string())
            .collect(),
    }
}

fn restore_snapshot_atom(snapshot: &SnapshotAtom) -> Option<Atom> {
    if snapshot.predicate.is_empty() || snapshot.terms.is_empty() {
        return None;
    }
    if snapshot.terms.iter().any(|term| term.is_empty()) {
        return None;
    }
    Some(Atom::new(
        &snapshot.predicate,
        snapshot
            .terms
            .iter()
            .map(|term| Term::constant(term))
            .collect(),
    ))
}

/// 归结原子 → 图账本原子（规则/派生物化为 RDF 时的结构载体）。
fn ledger_atom(atom: &Atom) -> graph_ledger::LedgerAtom {
    graph_ledger::LedgerAtom {
        predicate: atom.predicate.to_string(),
        terms: atom
            .terms
            .iter()
            .map(|term| graph_ledger::LedgerTerm {
                name: term.name().to_string(),
                is_var: matches!(term, Term::Var(_)),
            })
            .collect(),
    }
}

/// 图账本原子 → 归结原子（从规则账本恢复会话规则时使用；空名/空谓词拒绝）。
fn atom_from_ledger(atom: &graph_ledger::LedgerAtom) -> Option<Atom> {
    if atom.predicate.is_empty() || atom.terms.is_empty() {
        return None;
    }
    let mut terms = Vec::with_capacity(atom.terms.len());
    for term in &atom.terms {
        if term.name.is_empty() {
            return None;
        }
        terms.push(if term.is_var {
            Term::var(&term.name)
        } else {
            Term::constant(&term.name)
        });
    }
    Some(Atom::new(&atom.predicate, terms))
}

// 上一段生成的记忆：实体、已讲句数、段落排序角度（支持"接着说/换个角度说"）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationState {
    pub entity: String,
    pub shown: usize,
    pub order: GenerationOrder,
    /// 增量话语账本：记住已经说出口的命题。旧快照缺省时从 `shown` 迁移。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<GenerationCursor>,
    /// 上一段生成的可重述计划：只说「说什么」，表达由风格指令选择。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<AnswerPlan>,
}

// 会话事实的持久化快照（定义域：全部为基事实，故只存谓词+常量名）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotAtom {
    pub predicate: String,
    pub terms: Vec<String>,
}

// 会话 Horn 规则的持久化快照（变量/常量需区分，便于跨会话还原后向链）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotTerm {
    pub name: String,
    pub is_var: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotClauseAtom {
    pub predicate: String,
    pub terms: Vec<SnapshotTerm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotClause {
    pub head: SnapshotClauseAtom,
    pub body: Vec<SnapshotClauseAtom>,
    pub confidence: f64,
    pub source: String,
}

/// 修订操作的持久化形式：事实用 `atom`，规则用 `clause`，二者按 `kind` 取用。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotRevisionOp {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atom: Option<SnapshotAtom>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clause: Option<SnapshotClause>,
}

/// 一条可跨会话撤销的修订回执；空回执不落盘，旧快照字节保持不变。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotRevision {
    pub id: String,
    pub label: String,
    pub revision: u64,
    pub ops: Vec<SnapshotRevisionOp>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restore_lessons: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove_lessons: Vec<String>,
}

/// 待定假设的持久化形式：原子可含变量（不可采纳的假设也要原样恢复）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotHypothesis {
    pub atom: SnapshotClauseAtom,
    pub goal: SnapshotClauseAtom,
    /// `missing_fact` / `candidate_value`
    pub kind: String,
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<String>,
    #[serde(default)]
    pub contradicts_known: bool,
}

/// 澄清会话的持久化形式：保留当前问题、未问队列与已裁决清单。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotClarification {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queue: Vec<SnapshotHypothesis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<SnapshotHypothesis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub confirmed: Vec<SnapshotHypothesis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<SnapshotHypothesis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred: Vec<SnapshotHypothesis>,
}

/// 一次会话写入的修订回执：真实生效的操作 + 可一并恢复的教学原句。
#[derive(Debug, Clone)]
struct SessionRevisionReceipt {
    id: String,
    label: String,
    revision: u64,
    ops: Vec<ChangeOp>,
    restore_lessons: Vec<String>,
    remove_lessons: Vec<String>,
}

impl SessionRevisionReceipt {
    fn to_snapshot(&self) -> SnapshotRevision {
        SnapshotRevision {
            id: self.id.clone(),
            label: self.label.clone(),
            revision: self.revision,
            ops: self
                .ops
                .iter()
                .map(|op| match op {
                    ChangeOp::AssertFact(atom) => SnapshotRevisionOp {
                        kind: "assert".to_string(),
                        atom: Some(snapshot_atom(atom)),
                        clause: None,
                    },
                    ChangeOp::RetractFact(atom) => SnapshotRevisionOp {
                        kind: "retract".to_string(),
                        atom: Some(snapshot_atom(atom)),
                        clause: None,
                    },
                    ChangeOp::AddRule(clause) => SnapshotRevisionOp {
                        kind: "add_rule".to_string(),
                        atom: None,
                        clause: Some(snapshot_clause(clause)),
                    },
                    ChangeOp::RemoveRule(clause) => SnapshotRevisionOp {
                        kind: "remove_rule".to_string(),
                        atom: None,
                        clause: Some(snapshot_clause(clause)),
                    },
                })
                .collect(),
            restore_lessons: self.restore_lessons.clone(),
            remove_lessons: self.remove_lessons.clone(),
        }
    }

    fn from_snapshot(snapshot: &SnapshotRevision) -> Option<Self> {
        if snapshot.id.trim().is_empty()
            || (snapshot.ops.is_empty()
                && snapshot.restore_lessons.is_empty()
                && snapshot.remove_lessons.is_empty())
        {
            return None;
        }
        let mut ops = Vec::with_capacity(snapshot.ops.len());
        for op in &snapshot.ops {
            let restored = match op.kind.as_str() {
                "assert" => ChangeOp::AssertFact(restore_snapshot_atom(op.atom.as_ref()?)?),
                "retract" => ChangeOp::RetractFact(restore_snapshot_atom(op.atom.as_ref()?)?),
                "add_rule" => ChangeOp::AddRule(restore_clause(op.clause.as_ref()?)?),
                "remove_rule" => ChangeOp::RemoveRule(restore_clause(op.clause.as_ref()?)?),
                _ => return None,
            };
            ops.push(restored);
        }
        Some(Self {
            id: snapshot.id.clone(),
            label: snapshot.label.clone(),
            revision: snapshot.revision,
            ops,
            restore_lessons: snapshot.restore_lessons.clone(),
            remove_lessons: snapshot.remove_lessons.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueSnapshot {
    pub focus_entities: Vec<String>,
    /// 焦点实体的类型证据（代词消解闸门；字段缺省为空）
    #[serde(default)]
    pub focus_types: HashMap<String, String>,
    pub last_predicate: Option<String>,
    /// 上一轮成功编译出的查询计划类型（`QueryPlanner::kind`）。
    /// 省略追问据此决定补「换主语的关系问」还是「换宾语的属性问」。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_plan_kind: Option<String>,
    pub history: Vec<String>,
    #[serde(default)]
    pub generation: Option<GenerationState>,
    /// 逐命题复核台账；旧快照缺省为空，空台账不写字段以保持字节兼容。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic_turns: Option<EpistemicTurnBook>,
    /// 跨模块/跨轮的槽位信念账本；空账本不落盘，旧快照字节保持不变。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beliefs: Option<BeliefLedger>,
    // 跨会话临时信念：被采纳的假设事实（会话级，不写知识图谱）
    #[serde(default)]
    pub session_facts: Vec<SnapshotAtom>,
    // 跨会话否定事实：路径遍历中作为阻断边（会话级，不写知识图谱）
    #[serde(default)]
    pub session_negative_facts: Vec<SnapshotAtom>,
    // 会话槽账本（改口史/压制值）；旧快照缺省时从 session_facts 重建，
    // 空账本不落盘以保持旧快照字节不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<SlotLedger>,
    // 用户用中文教过的概率知识原句（会话级，跨会话持久化后自动重放）
    #[serde(default)]
    pub lessons: Vec<String>,
    // 用户用中文教过的 Horn 规则（会话级后向链规则库，跨会话持久化）
    #[serde(default)]
    pub session_clauses: Vec<SnapshotClause>,
    // 神经符号对比仲裁记忆（校准统计、待定推测、屏蔽/否证集），跨会话持久化
    #[serde(default)]
    pub contrast: Option<ContrastSnapshot>,
    // 话语帧记忆（上一帧 + 挂账），跨会话持久化后可接着补全未竟之问
    #[serde(default)]
    pub frames: Option<FrameMemory>,
    // 悬问账本：没答上的目标 + 缺失前提；空账本不落盘，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_questions: Option<PendingQuestionLedger>,
    // 批次冲突裁决：保留原 claim 与冲突表，重启后可继续「批答」；空则不落盘
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_pending: Option<BatchPendingTurn>,
    // 本体胶囊（教过的公理 + 摘要 + 闭包缓存）：跨会话/跨重建保留教学本体；
    // 未教过公理时缺省为 None，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology: Option<OntologyCapsule>,
    // 补全回合台账（候选/裁决/规则/作用域），跨会话持久化后可继续裁定；
    // 空台账不落盘，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionLedger>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predictive: Option<PredictiveElaboration>,
    // 反例采样/规则守卫账本：跨会话保存显式否认、功能谓词与已采纳守卫；
    // 空账本不落盘，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foil: Option<FoilLedger>,
    // 当前假设世界的临时假设（未确认、未写入会话事实层）；
    // 空世界不落盘，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<SnapshotAtom>,
    // 会话修订回执：跨重启仍可撤销最新写入；空回执不落盘
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_receipts: Vec<SnapshotRevision>,
    // 溯因待定假设：跨重启保留候选，可继续「采纳假设 / 澄清」；空池不落盘
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_hypotheses: Vec<SnapshotHypothesis>,
    // 进行中的澄清会话：保留当前问题与队列，重启后不重复已答问题
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clarification: Option<SnapshotClarification>,
    // 最近一次溯因目标与查询变量：澄清确认后据此复核目标是否已可证
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub last_abduction_goal: Vec<SnapshotClauseAtom>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub last_abduction_vars: Vec<String>,
    // 会话级镜像对齐（词形规则 + 回声词面）；空记忆不落盘，旧快照字节不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<AlignmentMemory>,
}

#[derive(Clone)]
pub struct DialogueContext {
    focus_entities: Vec<String>,
    /// 焦点实体的类型证据（代词消解的类型闸门；未知=键不存在）
    focus_types: HashMap<String, String>,
    last_predicate: Option<String>,
    last_plan_kind: Option<String>,
    history: Vec<String>,
    generation: Option<GenerationState>,
    max_focus: usize,
    max_history: usize,
}

impl DialogueContext {
    pub fn new(max_focus: usize) -> Self {
        Self {
            focus_entities: Vec::new(),
            focus_types: HashMap::new(),
            last_predicate: None,
            last_plan_kind: None,
            history: Vec::new(),
            generation: None,
            max_focus,
            max_history: 20,
        }
    }

    pub fn focus(&self) -> Option<&str> {
        self.focus_entities.first().map(|s| s.as_str())
    }

    pub fn last_predicate(&self) -> Option<&str> {
        self.last_predicate.as_deref()
    }

    /// 上一轮查询计划类型；只记录成功编译的计划，省略句本身不会覆盖它。
    pub fn last_plan_kind(&self) -> Option<&str> {
        self.last_plan_kind.as_deref()
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn focus_entities(&self) -> &[String] {
        &self.focus_entities
    }

    pub fn record_entities(&mut self, entities: &[String]) {
        let typed: Vec<(String, Option<String>)> = entities
            .iter()
            .map(|entity| (entity.clone(), None))
            .collect();
        self.record_entities_typed(&typed);
    }

    /// 带类型证据的焦点登记：类型随实体入栈，供代词消解做类型过滤。
    pub fn record_entities_typed(&mut self, entities: &[(String, Option<String>)]) {
        for (entity, etype) in entities.iter().rev() {
            if entity.is_empty() {
                continue;
            }
            if let Some(etype) = etype.as_deref().filter(|etype| !etype.trim().is_empty()) {
                self.focus_types.insert(entity.clone(), etype.to_string());
            }
            self.focus_entities.retain(|e| e != entity);
            self.focus_entities.insert(0, entity.clone());
        }
        if self.focus_entities.len() > self.max_focus {
            self.focus_entities.truncate(self.max_focus);
        }
        // 类型表随焦点栈裁剪，避免无限增长
        self.focus_types
            .retain(|entity, _| self.focus_entities.contains(entity));
    }

    /// 焦点实体的类型证据（代词闸门用）。
    pub fn focus_type(&self, entity: &str) -> Option<&str> {
        self.focus_types.get(entity).map(String::as_str)
    }

    pub fn record_predicate(&mut self, predicate: &str) {
        if !predicate.is_empty() {
            self.last_predicate = Some(predicate.to_string());
        }
    }

    pub fn record_plan_kind(&mut self, kind: &str) {
        if !kind.is_empty() {
            self.last_plan_kind = Some(kind.to_string());
        }
    }

    pub fn record_question(&mut self, question: &str) {
        if question.trim().is_empty() {
            return;
        }
        self.history.insert(0, question.to_string());
        if self.history.len() > self.max_history {
            self.history.truncate(self.max_history);
        }
    }

    pub fn clear(&mut self) {
        self.focus_entities.clear();
        self.focus_types.clear();
        self.last_predicate = None;
        self.last_plan_kind = None;
        self.history.clear();
        self.generation = None;
    }

    pub fn generation_state(&self) -> Option<&GenerationState> {
        self.generation.as_ref()
    }

    pub fn record_generation(&mut self, entity: &str, shown: usize, order: GenerationOrder) {
        if entity.trim().is_empty() {
            return;
        }
        let plan = self.retained_plan(entity);
        self.generation = Some(GenerationState {
            entity: entity.to_string(),
            shown,
            order,
            cursor: None,
            plan,
        });
    }

    /// 记录一段生成及其话语账本；`shown` 由账本派生，保持旧字段可读。
    pub fn record_generation_with_cursor(&mut self, cursor: GenerationCursor) {
        if cursor.entity.trim().is_empty() {
            return;
        }
        let plan = self.retained_plan(&cursor.entity);
        self.generation = Some(GenerationState {
            entity: cursor.entity.clone(),
            shown: cursor.spoken_count(),
            order: cursor.order,
            cursor: Some(cursor),
            plan,
        });
    }

    /// 记录一段生成、话语账本与可重述计划。
    pub fn record_generation_with_plan(&mut self, cursor: GenerationCursor, plan: AnswerPlan) {
        if cursor.entity.trim().is_empty() {
            return;
        }
        self.generation = Some(GenerationState {
            entity: cursor.entity.clone(),
            shown: cursor.spoken_count(),
            order: cursor.order,
            cursor: Some(cursor),
            plan: Some(plan),
        });
    }

    /// 同一实体的计划在「接着说」等操作后继续保留；换实体则丢弃。
    fn retained_plan(&self, entity: &str) -> Option<AnswerPlan> {
        self.generation
            .as_ref()
            .filter(|state| state.entity == entity)
            .and_then(|state| state.plan.clone())
    }

    pub fn snapshot(&self) -> DialogueSnapshot {
        DialogueSnapshot {
            focus_entities: self.focus_entities.clone(),
            focus_types: self.focus_types.clone(),
            last_predicate: self.last_predicate.clone(),
            last_plan_kind: self.last_plan_kind.clone(),
            history: self.history.clone(),
            generation: self.generation.clone(),
            epistemic_turns: None,
            beliefs: None,
            session_facts: Vec::new(),
            session_negative_facts: Vec::new(),
            slots: None,
            lessons: Vec::new(),
            session_clauses: Vec::new(),
            contrast: None,
            frames: None,
            pending_questions: None,
            batch_pending: None,
            ontology: None,
            completion: None,
            predictive: None,
            foil: None,
            assumptions: Vec::new(),
            session_receipts: Vec::new(),
            pending_hypotheses: Vec::new(),
            clarification: None,
            last_abduction_goal: Vec::new(),
            last_abduction_vars: Vec::new(),
            alignment: None,
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueSnapshot) {
        self.focus_entities = snapshot.focus_entities.clone();
        self.focus_entities.truncate(self.max_focus);
        self.focus_types = snapshot.focus_types.clone();
        self.focus_types
            .retain(|entity, _| self.focus_entities.contains(entity));
        self.last_predicate = snapshot.last_predicate.clone();
        self.last_plan_kind = snapshot.last_plan_kind.clone();
        self.history = snapshot.history.clone();
        self.history.truncate(self.max_history);
        self.generation = snapshot.generation.clone();
    }

    /// 类型过滤 + 显著度排序地消解一个代词。
    ///
    /// - 序数代词：「前者」= 焦点栈第 1 个、「后者」= 第 2 个（按最近一次提及顺序）；
    /// - 类型代词：「他/她/这个人」要求人物类型（类型未知按开放世界放行），
    ///   「它/它们」要求已知非人物类型，「该公司」要求公司/组织类型；
    /// - 找不到唯一相容指称就返回 None：原样让位，绝不乱猜。
    pub fn resolve_pronoun_referent(&self, pronoun: &str) -> Option<String> {
        if let Some(index) = crate::identity::ordinal_pronoun_index(pronoun) {
            return self.focus_entities.get(index).cloned();
        }
        let filter = crate::identity::pronoun_filter(pronoun)?;
        self.focus_entities
            .iter()
            .find(|entity| crate::identity::type_allows(self.focus_type(entity), filter))
            .cloned()
    }

    pub fn resolve_pronouns(&self, input: &str) -> Option<String> {
        if self.focus_entities.is_empty() {
            return None;
        }
        let mut result = String::with_capacity(input.len() + 8);
        let mut replaced = false;
        let mut chars = input.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            // 在字节切片上直接判断前缀（代词表长词面优先，避免「他们」被「他」截断）
            let mut matched: Option<&str> = None;
            for pronoun in crate::identity::PRONOUNS {
                if input[i..].starts_with(pronoun) {
                    matched = Some(pronoun);
                    break;
                }
            }
            if let Some(pronoun) = matched {
                let previous = input[..i].chars().next_back();
                let blocked = matches!(previous, Some('其') | Some('无'));
                if !blocked {
                    if let Some(referent) = self.resolve_pronoun_referent(pronoun) {
                        result.push_str(&referent);
                        let end = i + pronoun.len();
                        while let Some((next, _)) = chars.peek() {
                            if *next < end {
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        replaced = true;
                        continue;
                    }
                    // 类型不相容 / 指称不足：代词原样保留（让位），不伪造指代
                }
            }
            result.push(ch);
        }
        if replaced {
            Some(result)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct ValueHit {
    value: String,
    prediction: Option<Prediction>,
    /// 该取值由哪条 Active 语义律推出（直接事实为 None）。
    law: Option<String>,
    /// 该取值来自建设性多跳推断（查询局部，未写图谱）。
    construction: Option<VirtualFact>,
}

#[derive(Debug, Clone)]
struct ChainStep {
    subject: String,
    predicate: String,
    hits: Vec<ValueHit>,
}

#[derive(Debug, Clone)]
struct ChainRun {
    steps: Vec<ChainStep>,
    last_hits: Vec<ValueHit>,
    closure_truncated: bool,
    construction_truncated: bool,
    construction_facts: usize,
}

/// 问句链上的一个谓词环节：词面（用户问法）+ 规范槽位（存储坐标）+ 是否逆向。
#[derive(Debug, Clone)]
struct ChainPredicate {
    surface: String,
    slot: String,
    inverted: bool,
}

/// 类型化配方的查询结果：取值 + 可复述的依据边。
///
/// 取值来自图谱事实或本体闭包（含逆关系/子类推断）；依据只复述实际存在的
/// 事实边与 TBox 推导理由，取不到值时返回 `None`，绝不凭空补值。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecipeLookup {
    pub values: Vec<String>,
    pub evidence: Vec<String>,
}

/// 插件配方的完整回答：回答文本之外的实体/属性/问句/依据一并返回，
/// 让 CLI 与 HTTP 共用同一条执行路径（接地、回指、模板、追踪不漂移）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginRecipeReply {
    pub text: String,
    pub plan: Vec<String>,
    pub provenance: Vec<String>,
    pub entity: Option<String>,
    pub attribute: Option<String>,
    pub question: String,
    pub kind: &'static str,
    /// 本轮结束后应进入对话焦点的实体：正向配方取答案（最后一个提及的
    /// 实体），逆向配方保留被查询的主语。用于「这位导演还拍过什么」这类
    /// 省略主语的下一轮追问。
    pub focus: Option<String>,
}

/// 插件本体并入统一推理器的增量回执：本次真正新增的别名/逆向词面/公理数。
/// 同一运行态重复应用返回全零，便于 CLI 与 HTTP 共用单调、幂等的合并路径。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PluginOntologySync {
    pub aliases: usize,
    pub inverse_surfaces: usize,
    pub lessons: usize,
    pub rules: usize,
    pub rules_removed: usize,
}

/// 语言策略热更新的回执。配置里的推理结构阈值不在本方法范围内：
/// 那些属于需要重启的层，绝不在一轮对话中半途换掉。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LanguagePolicySync {
    /// 新登记的同义词条数（已存在的映射不计）。
    pub synonyms_added: usize,
    /// 是否替换了分词/NLU 配置。
    pub nlp_updated: bool,
    /// 生成预算是否变化。
    pub generation_updated: bool,
    /// 当前是否挂接着双向语义规则包。
    pub semantic_grammar: bool,
}

/// 结构化证明视图：可视化（Mermaid/ASCII）消费的纯数据，不重复解析逻辑。
#[derive(Debug, Clone)]
pub struct ResolutionProofView {
    pub phrase: String,
    pub goal_display: String,
    pub facts: usize,
    pub clauses: usize,
    pub max_depth: usize,
    /// 三态搜索结论：得证 / 无解（已穷尽）/ 未得证（搜索被截断）。
    pub search_status: SearchStatus,
    /// 搜索是否触发深度、预算、解数或状态上限。
    pub search_truncated: bool,
    /// 本次搜索实际遇到的代表性知识缺口。
    pub frontier: Vec<FailureFrontier>,
    /// 本次搜索的确定性计数（供诊断面/日志消费，不含耗时）。
    pub stats: SearchStats,
    pub solutions: Vec<ProofSolutionView>,
}

/// 一个归结解：变量绑定 + 置信度 + 各目标的证明树。
#[derive(Debug, Clone)]
pub struct ProofSolutionView {
    pub bindings: Vec<(String, String)>,
    pub confidence: f64,
    pub proofs: Vec<ProofViewNode>,
}

/// 上下文改写收据：改写文本 + 来源说明（可打印、可审计）
#[derive(Debug, Clone)]
pub struct ContextRewrite {
    pub text: String,
    pub notes: Vec<String>,
}

impl ContextRewrite {
    fn from_text(text: String) -> Self {
        Self {
            text,
            notes: Vec::new(),
        }
    }
}

/// 一个有界的临时假设世界：跨轮追问共享同一组假设，退出即整体丢弃。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssumptionWorld {
    /// 由假设规范形哈希得到的稳定 id（与插入顺序无关）。
    pub id: String,
    pub assumptions: Vec<Atom>,
}

/// 本体视图适配器：帧解析与合一期间只读消费对齐器/谓词清单（排序判定）
struct FrameOntology<'a>(&'a SymbolicReasoner);

impl OntologyView for FrameOntology<'_> {
    fn canonical_entity(&self, raw: &str) -> Option<String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        let canonical = self.0.aligner.canonical_form(raw);
        if self.0.is_known_entity(&canonical) || self.0.is_known_entity(raw) {
            return Some(canonical);
        }
        None
    }

    fn canonical_attribute(&self, raw: &str) -> Option<String> {
        self.0.canonical_attribute_name(raw)
    }
}

#[derive(Clone)]
struct GraphSession {
    compiled: CompiledGraphQuestion,
    conversation: ConversationalGraph,
    /// 首次编译出的原模式与后续已确认的放宽，供累计标注与「恢复原条件」。
    ledger: RepairLedger,
    /// 最近一次零解诊断生成的有界放宽候选（尚未生效）。
    repairs: Vec<RepairSuggestion>,
}

impl RepairHints for SymbolicReasoner {
    fn predicate_alternatives(&self, predicate: &str) -> Vec<(String, String)> {
        let retrieval = self.retrieval.lock();
        let mut alternatives = Vec::new();
        for parent in retrieval.superproperties_reachable(predicate) {
            alternatives.push((
                parent.clone(),
                format!("「{}」是声明式中更一般的「{}」", predicate, parent),
            ));
        }
        for child in retrieval.subproperties_reachable(predicate) {
            alternatives.push((
                child.clone(),
                format!("「{}」是「{}」的声明子属性", child, predicate),
            ));
        }
        alternatives
    }

    fn type_supertypes(&self, entity_type: &str) -> Vec<(String, String)> {
        self.retrieval
            .lock()
            .type_ancestors(entity_type)
            .into_iter()
            .map(|parent| {
                (
                    parent.clone(),
                    format!("「{}」的父类型是「{}」", entity_type, parent),
                )
            })
            .collect()
    }

    fn entity_alternatives(&self, entity: &str) -> Vec<(String, String)> {
        if !self.config.enable_semantic_embedding || !self.embedding.is_embedded(entity) {
            return Vec::new();
        }
        self.embedding
            .nearest(entity, 2)
            .into_iter()
            .filter(|(candidate, _)| candidate != entity && self.is_known_entity(candidate))
            .map(|(candidate, score)| {
                (
                    candidate.clone(),
                    format!(
                        "「{}」与「{}」的结构角色相近（相似度 {:.2}）；这只是候选，需确认",
                        entity, candidate, score
                    ),
                )
            })
            .collect()
    }
}

fn describe_graph_subjects(subjects: &[String]) -> String {
    match subjects {
        [] => "「前者」和「后者」".to_string(),
        [only] => format!("「{}」", only),
        [left, right] => format!("「{}」和「{}」", left, right),
        many => {
            let quoted: Vec<String> = many.iter().map(|name| format!("「{}」", name)).collect();
            quoted.join("、")
        }
    }
}

/// 生成审计结果：`text` 只包含通过蕴含校验的行，`note` 为可回放的审计回执。
struct GenerationAudit {
    text: String,
    note: Option<String>,
}

/// 语义歧义澄清的一个可答读法：自然答案 + 可追溯的逻辑形式。
#[derive(Debug, Clone)]
struct FormalClarificationOption {
    logical_form: String,
    answer: String,
    evidence: Vec<String>,
}

impl FormalClarificationOption {
    fn full_answer(&self) -> String {
        if self.evidence.is_empty() {
            self.answer.clone()
        } else {
            format!("{}\n证据：{}", self.answer, self.evidence.join("；"))
        }
    }
}

/// 解析森林给出多个有证据读法时的会话态澄清帧。
///
/// 只在旧回答路径失败后建立；用户用序号/前者/后者选择，或明确放弃。
/// 不做默认选择，也不把多个候选合并成一条「都成立」的假答案。
#[derive(Debug, Clone)]
struct FormalClarification {
    question: String,
    options: Vec<FormalClarificationOption>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormalReply {
    Choice(usize),
    OutOfRange,
    Cancel,
}

impl FormalClarification {
    fn prompt(&self) -> String {
        let mut out = format!(
            "「{}」有 {} 种有证据的读法，请选择：",
            self.question,
            self.options.len()
        );
        for (index, option) in self.options.iter().enumerate() {
            out.push_str(&format!(
                "\n  {}. {}（读法：{}）",
                index + 1,
                option.answer.replace('\n', "；"),
                option.logical_form
            ));
        }
        out.push_str("\n输入序号选择；输入「都不是」放弃。");
        out
    }

    fn parse_reply(&self, text: &str) -> Option<FormalReply> {
        let trimmed = text
            .trim()
            .trim_end_matches(['。', '.', '，', ',', '!', '！', '?', '？'])
            .trim();
        if trimmed.is_empty() {
            return None;
        }
        if ["都不是", "都不要", "取消", "算了", "放弃", "quit", "exit"].contains(&trimmed)
        {
            return Some(FormalReply::Cancel);
        }
        let ordinal = |token: &str| -> Option<usize> {
            let mut digits = token.trim();
            for prefix in ["选择", "选", "第"] {
                digits = digits.strip_prefix(prefix).unwrap_or(digits).trim();
            }
            let mut digits = digits.trim();
            loop {
                let stripped = digits
                    .trim_end_matches(|c| matches!(c, '个' | '条' | '项' | '种' | '号' | '吧'));
                if stripped == digits {
                    break;
                }
                digits = stripped.trim_end();
            }
            let digits = digits.trim();
            let value = if let Ok(number) = digits.parse::<usize>() {
                Some(number)
            } else {
                match digits {
                    "一" => Some(1),
                    "二" | "两" => Some(2),
                    "三" => Some(3),
                    "四" => Some(4),
                    "五" => Some(5),
                    "六" => Some(6),
                    "七" => Some(7),
                    "八" => Some(8),
                    "九" => Some(9),
                    "十" => Some(10),
                    _ => None,
                }
            }?;
            Some(value)
        };
        if trimmed == "前者" {
            return Some(FormalReply::Choice(0));
        }
        if trimmed == "后者" {
            return Some(FormalReply::Choice(1));
        }
        if let Some(value) = ordinal(trimmed) {
            if value == 0 || value > self.options.len() {
                return Some(FormalReply::OutOfRange);
            }
            return Some(FormalReply::Choice(value - 1));
        }
        None
    }
}

/// 语义歧义澄清的一步结果：给出选中答案 / 越界重问 / 放弃 / 不是回答。
#[derive(Debug, Clone, PartialEq)]
pub enum FormalClarificationOutcome {
    Answered(String),
    Reprompt(String),
    Cancelled(String),
    NotAnAnswer,
}

pub struct SymbolicReasoner {
    config: ReasoningConfig,
    aligner: OntologyAligner,
    predictor: LinkPredictor,
    // 轮末预测答案草稿：按知识纪元失效，只复用纯函数结果，不写图谱。
    prediction_drafts: parking_lot::Mutex<PredictionDrafts>,
    // 补全回合：确认 → 归纳批次规则 → 有界闭包；默认不写图谱、不全局生效。
    completion: CompletionLedger,
    // 生成侧推测账本：链接预测 → 闸门 → 呈现 → 裁决 → 会话记忆。
    predictive: PredictiveElaboration,
    // 反例/守卫账本：来自 `foil` 子命令的显式决定；只过滤候选，不写图谱。
    foil_ledger: FoilLedger,
    // 律典：声明律 + 挖掘律的统一台账；仅 Active 的传递/对称/互逆律参与查询扩展。
    laws: LawBook,
    // 律典的声明来源快照；知识变化时据此重建，避免把配置知识散落在查询路径。
    law_registry: Option<PredicateRegistry>,
    // 显式持久化的采纳/退役决定；每次重建律典后重新盖到声明/挖掘结果上。
    persisted_laws: Option<LawBook>,
    context: DialogueContext,
    epistemic_turns: EpistemicTurnBook,
    // 跨模块/跨轮槽位信念：只记录带信封的命题，不写图谱。
    beliefs: BeliefLedger,
    embedding: SymbolicEmbedding,
    // 单一 RDFS 模式视图：类型闭包/子属性/域值域，嵌入与审计共用同一坐标系。
    schema_view: SchemaView,
    // 符号化图嵌入：随机游走重启（RWR）向量，做结构联想与话题延伸
    walks: GraphWalkEmbedding,
    // 轻量马尔可夫逻辑网络：加权规则软真值传播 + 加权 MAP 局部搜索
    soft: MarkovLogicEngine,
    // ProbLog 风格概率逻辑程序：概率事实/规则/注释析取 + 可能世界精确推理（BDD 编译）
    problog: ProblogEngine,
    // 用户用中文教过的概率知识原句（会话级记忆，随对话状态持久化）
    lessons: Vec<String>,
    // 上面原句的解析结果缓存：rebuild_problog 会随每次知识变更重放。
    lessons_parsed: Vec<chinese_reasoning_nlu::ChineseLesson>,
    planner: QueryPlanner,
    linker: EntityLinker,
    generator: SymbolicGenerator,
    // 已采纳的习得重写规则；只作为读法候选，不覆盖精确解析。
    rewrite_book: Option<RewriteBook>,
    // 统一证据索引：词面倒排 × 结构 PPR × 谓词匹配，支持约束与可审计剪枝回执。
    // 用 Mutex 承载惰性重建，使只读回答路径也能在不改变 &self 契约下检索。
    retrieval: parking_lot::Mutex<DenseRetriever>,
    // 对话级证据装配器的会话记忆：上次回答引用过的证据（追问去重、防复述）
    contextual_last: parking_lot::Mutex<Option<(String, Vec<EvidenceKey>)>>,
    // 语义检索端口的结果集会话：显式保存上一轮证据，支持确定性追问操作
    search_session: parking_lot::Mutex<crate::search_port::SearchSession>,
    // 语义歧义澄清帧：解析森林出现多个有证据读法时，等待用户选定其一
    formal_clarification: parking_lot::Mutex<Option<FormalClarification>>,
    // 语境分词读法记忆：同一词面在不同焦点/类型语境下的确认与纠正（会话态）
    contextual_memory: ReadingMemory,
    // 实体属性三元组（观察到的），供归结引擎编码事实
    attributes: HashSet<(String, String, String)>,
    // 属性查询索引：规范槽位 `(实体, 属性)` → 取值列表（随知识重建）
    attribute_index: HashMap<(String, String), Vec<String>>,
    // 属性本体档案：属性别名、量纲与规范单位；回答侧据此做单位化。
    attribute_profiles: crate::attribute_bridge::AttributeRegistry,
    // 关联规则挖掘器（类型/谓词/属性篮子的支持度-置信度-提升度）
    associations: AssociationMiner,
    // 一阶归结引擎（事实 + 链接预测规则 + 关联规则）
    resolver: ResolutionEngine,
    // 查询级证明记忆：同一会话内复用完整证明，按字节预算与知识变更失效。
    proof_memory: parking_lot::Mutex<ProofMemory<ResolutionProofView>>,
    // 溯因·假设生成引擎（后向链失败前沿 → 候选假设 → 反事实模拟）
    abduction: AbductionEngine,
    // 话语帧记忆：带洞查询帧 + 挂账（合一式多轮消解，跨会话持久化）
    frames: FrameMemory,
    // 悬问账本：没答上的目标 + 缺失前提；用户补上后重解原目标并延期回答。
    pending_questions: PendingQuestionLedger,
    // 结构型多轮：当前图谱模式、结果与翻页位置（会话态，不进知识图谱）
    graph_session: parking_lot::Mutex<Option<GraphSession>>,
    // 会话级临时事实（被采纳的假设；跨会话持久化，但永不写知识图谱）
    session_facts: Vec<Atom>,
    // 会话槽账本：把 session_facts 看成 (主语,谓词) 槽账本的投影，
    // 记录功能槽的 before/after 改口史（压制不等于否定，可撤销）。
    slots: SlotLedger,
    // 当前假设世界：临时前提只在世界内参与推理，退出/分叉时整体丢弃
    assumption_world: Option<AssumptionWorld>,
    // 会话级否定事实：路径遍历中的阻断边（不写知识图谱，可跨会话持久化）
    session_negative_facts: Vec<Atom>,
    // 会话级 Horn 规则（用户用中文教的一般规则/条件句；供后向链归结使用）
    session_clauses: Vec<Clause>,
    // 宿主插件规则包贡献（按当前启用集重算；停用或裁决会撤回落选规则）
    plugin_clauses: Vec<Clause>,
    // 会话写入的修订回执（预检通过后才记录；支持按最新回执撤销）
    session_receipts: Vec<SessionRevisionReceipt>,
    // 批次冲突裁决：一次输入编译成一批 claim，冲突未决时挂起一张冲突表
    batch_pending: Option<BatchPendingTurn>,
    // 当前澄清对话（把待定假设逆转为系统主动提问）
    clarification: Option<ClarificationSession>,
    // 澄清分支前瞻：下一句常见短答的派生证据包；按封印整体失效，不写图谱。
    clarification_lookahead: parking_lot::Mutex<LookaheadBoard>,
    // 最近一次溯因的目标合取与查询变量（澄清确认真后验证整个目标是否得证）
    last_abduction_goal: Option<Vec<Atom>>,
    last_abduction_vars: Vec<String>,
    // 谓词清单缓存：仅在 observe_facts 后重建（按对齐器剖面与预测规则生成，顺序敏感）
    known_predicates_cache: Vec<String>,
    // 与 known_predicates_cache 同序的（原形, 归一化形）缓存
    known_predicates_normalized_cache: Vec<(String, String)>,
    // 命题解析器的“表面积 → 规范谓词”索引缓存：随观察事实重建
    proposition_surfaces: Vec<(String, String)>,
    // 链接结果记忆：同一文本在一次查询流程中会被多处链接（上下文记录/链接说明/回答），
    // 多轮中交替询问的文本各自留槽（按知识版本失效的有界 LRU）；
    // 观察事实或重建缓存（线索变化）时整体失效
    link_memo: parking_lot::Mutex<RevisionLru<Arc<Vec<LinkedMention>>>>,
    // 实体提及记忆：record_vocabulary 对同一文本会多次调用，且每次需全量扫描实体剖面
    entity_mentions_memo: parking_lot::Mutex<RevisionLru<Vec<String>>>,
    // 同义词命中的规范谓词集合记忆
    predicate_canonicals_memo: parking_lot::Mutex<RevisionLru<Arc<HashSet<String>>>>,
    // 谓词画像逆向词面：词面 → (规范槽位, 逆向)。问句以逆向谓词发问时
    // 主宾对调命中同一批事实（「徒弟」问句 ↔「师傅」事实），零数据冗余
    inverse_query_map: HashMap<String, (String, bool)>,
    // 符号主义话语引擎：OWL 本体叙述（定义/继承/推导链）+ DCG 文法实现（多句式生成）
    discourse: DiscoursePlanner,
    // 全息符号记忆（VSA/HDC）：绑定/叠加/置换 + 净噪的软召回通道，
    // 与三元组登记集合（硬核查通道）交叉验证，无神经网络、零训练
    holo: HolographicMemory,
    // 知识版本：observe_facts 检测到内容确有变化时自增（纯重复观察不动）
    revision: u64,
    // 派生台账：每个派生器最后重建时的知识版本（版本键，可审计、可断言）
    derived_stamps: BTreeMap<&'static str, u64>,
    // 依赖台账：每类依赖最后发生变化的版本
    dependency_stamps: BTreeMap<&'static str, u64>,
    // 操作计数台账（可与其他模块共享）
    perf: Arc<PerfLedger>,
    // 读法观测：宿主注入 NLP 配置后，失败轮才做一次只读词法轨迹；
    // None 时零额外分词/链接成本，回答路径逐字节不变。
    nlp_config: Option<NlpConfig>,
    // 有界失败回执（只含 digest/计数/有限枚举，不保存原文，不进会话快照）
    read_traces: ReadTraceLedger,
    semantic: SemanticMonitor,
    // 跨会话词法记忆：词性来源证据 + 量词→类型期待（软约束；由宿主落图）
    lexical_memory: LexicalMemory,
    // 可生长的提及词表：图谱实体/确认别名/显式准入回灌切分覆盖层；
    // 待定只提名，墓碑永久挡板。派生自知识，不单独持久化。
    mention_lexicon: MentionLexicon,
    // 引擎层故障通道（Prolog / 存储等下游失败由宿主注入）：
    // 非空时对话管理器走 EngineFault 路径，绝不与「知识里没有」合并。
    engine_fault: Option<String>,
}

/// 锚定重写证据：把实体链接器的类型证据、谓词画像的定义域/值域，
/// 以及本体闭包中的子类/互斥声明合成一个只读视图。只在精确路径失败后使用。
struct AnchorEvidenceView<'a> {
    linker: &'a EntityLinker,
    registry: Option<&'a PredicateRegistry>,
    capsule: Option<&'a OntologyCapsule>,
}

impl AnchorEvidence for AnchorEvidenceView<'_> {
    fn entity_types(&self, entity: &str) -> Vec<String> {
        let mut types: Vec<String> = self
            .linker
            .entity_type(entity)
            .map(str::to_string)
            .into_iter()
            .collect();
        if let Some(capsule) = self.capsule {
            for (individual, class) in &capsule.closure.instances {
                if individual == entity {
                    types.push(class.clone());
                }
            }
        }
        types.sort();
        types.dedup();
        types
    }

    fn domain_of(&self, predicate: &str) -> Vec<String> {
        self.registry
            .and_then(|registry| registry.lookup(predicate))
            .map(|(profile, inverted)| {
                let classes = if inverted {
                    &profile.range
                } else {
                    &profile.domain
                };
                classes.iter().cloned().collect()
            })
            .unwrap_or_default()
    }

    fn range_of(&self, predicate: &str) -> Vec<String> {
        self.registry
            .and_then(|registry| registry.lookup(predicate))
            .map(|(profile, inverted)| {
                let classes = if inverted {
                    &profile.domain
                } else {
                    &profile.range
                };
                classes.iter().cloned().collect()
            })
            .unwrap_or_default()
    }

    fn type_fit(&self, actual: &str, expected: &str) -> TypeFit {
        let base = query_anchor::default_type_fit(actual, expected);
        if base == TypeFit::Compatible {
            return base;
        }
        let Some(capsule) = self.capsule else {
            return base;
        };
        let ancestors = |name: &str| {
            let mut set: BTreeSet<String> = BTreeSet::new();
            set.insert(name.to_string());
            if let Some((_, supers)) = capsule
                .closure
                .subclasses
                .iter()
                .find(|(sub, _)| sub == name)
            {
                set.extend(supers.iter().cloned());
            }
            set
        };
        let actual_ancestors = ancestors(actual.trim());
        let expected_ancestors = ancestors(expected.trim());
        if actual_ancestors.contains(expected.trim()) || expected_ancestors.contains(actual.trim())
        {
            return TypeFit::Compatible;
        }
        for lesson in &capsule.lessons {
            if let OntologyLesson::Disjoint { left, right, .. } = lesson {
                let (left, right) = (left.trim(), right.trim());
                if (actual_ancestors.contains(left) && expected_ancestors.contains(right))
                    || (actual_ancestors.contains(right) && expected_ancestors.contains(left))
                {
                    return TypeFit::Disjoint;
                }
            }
        }
        base
    }
}

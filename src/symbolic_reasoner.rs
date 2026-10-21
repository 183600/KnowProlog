use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::abduction::{
    self, AbductionEngine, AbductionReport, ClarificationReply, ClarificationSession, Hypothesis,
    Scenario,
};
use crate::association_miner::{display_item, AssociationMiner, AssociationRule};
use crate::cache::RevisionLru;
use crate::canonical::{self, DecodeError, DecodeReceipt};
use crate::causal_reasoning::{CausalAnalyzer, CausalGraph, CounterfactualDelta};
use crate::causal_tokenization::{attribute_tokens, AnchorKind, CausalTokenizer, MAX_TOKEN_CHARS};
use crate::chinese_reasoning_nlu::{self, LessonKind};
use crate::contextual_reference::{SenseCard, SenseSource};
use crate::default_logic::{self, DefaultTheory, Literal};
use crate::dense_retrieval::{DenseRetriever, EvidenceSearch, SearchBudget, SearchConstraints};
use crate::derivation::Derivation;
use crate::dialogue_frame::{self, AskAttributeFrame, FrameMemory, OntologyView};
use crate::discourse::DiscoursePlanner;
use crate::entity_linker::{EntityLinker, LinkMethod, LinkedMention};
use crate::graph_ledger;
use crate::graph_walk_embedding::GraphWalkEmbedding;
use crate::holographic_memory::HolographicMemory;
use crate::law_book::{LawBook, LawStatus};
use crate::lexical_trace::{
    self, LexicalTrace, ReadFailureKind, ReadStage, ReadTraceLedger, TraceFidelity, TraceReceipt,
};
use crate::link_prediction::{HornRule, LinkPredictor, Prediction, RuleTemplate, Triple};
use crate::markov_logic::{MarkovLogicEngine, SoftFact, WeightedRule};
use crate::neuro_symbolic_contrast::ContrastSnapshot;
use crate::ontology_alignment::{AlignmentDecision, AlignmentState, OntologyAligner, TermKind};
use crate::ontology_reasoning::{OntologyCapsule, OntologyLesson, OntologyProgram};
use crate::path_memory::{EdgeSource, PathPredicate, TraversalGraph, TraversalOptions};
use crate::pattern_closure::{evaluate_closure, ClosureBudget, ClosureOp, ClosureOutcome};
use crate::perf_ledger::PerfLedger;
use crate::plugin::{PluginOutcome, PluginRuntime, RecipeDirection};
use crate::predicate_registry::PredicateRegistry;
use crate::problog::{ProblogAnswer, ProblogEngine};
use crate::prolog_engine::PrologFact;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::query_rewrite::{self, RelationLexicon};
use crate::resolution_engine::{
    render_term, Atom, Clause, DerivedFact, FailureFrontier, ProofNode, ProofSource,
    ResolutionEngine, SearchOutcome, SearchStats, SearchStatus, Solution, Term,
};
use crate::semantic_embedding::SymbolicEmbedding;
use crate::semantic_entailment::{self, EntailmentResult, Knowledge, PropositionParser, Verdict};
use crate::symbolic_generator::{GenerationOrder, SymbolicGenerator};
use crate::temporal::{
    self, validity_from_bounds, TemporalConstraint, UnknownTimePolicy, Validity,
};
use crate::view_render::ProofViewNode;
use crate::{NlpConfig, ReasoningConfig};

// 归结引擎内部谓词：has(E, 谓词名) / type(E, 类型名)，用于承载关联规则的物品
const PROOF_HAS: &str = "__具有__";
const PROOF_TYPE: &str = "__类型__";

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

fn restore_clause(snapshot: &SnapshotClause) -> Option<Clause> {
    let restore_atom = |atom: &SnapshotClauseAtom| -> Option<Atom> {
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
    };
    let head = restore_atom(&snapshot.head)?;
    let mut body = Vec::with_capacity(snapshot.body.len());
    for atom in &snapshot.body {
        body.push(restore_atom(atom)?);
    }
    Some(Clause::new(
        head,
        body,
        snapshot.confidence,
        &snapshot.source,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueSnapshot {
    pub focus_entities: Vec<String>,
    /// 焦点实体的类型证据（代词消解闸门；字段缺省为空）
    #[serde(default)]
    pub focus_types: HashMap<String, String>,
    pub last_predicate: Option<String>,
    pub history: Vec<String>,
    #[serde(default)]
    pub generation: Option<GenerationState>,
    // 跨会话临时信念：被采纳的假设事实（会话级，不写知识图谱）
    #[serde(default)]
    pub session_facts: Vec<SnapshotAtom>,
    // 跨会话否定事实：路径遍历中作为阻断边（会话级，不写知识图谱）
    #[serde(default)]
    pub session_negative_facts: Vec<SnapshotAtom>,
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
    // 本体胶囊（教过的公理 + 摘要 + 闭包缓存）：跨会话/跨重建保留教学本体；
    // 未教过公理时缺省为 None，旧快照字节保持不变
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology: Option<OntologyCapsule>,
}

/// 概率降序、同概率按显示串升序的排序键
#[derive(PartialEq)]
struct ProbabilityDisplayKey {
    probability: f64,
    display: String,
}

impl Eq for ProbabilityDisplayKey {}

impl PartialOrd for ProbabilityDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbabilityDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .probability
            .partial_cmp(&self.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 事实标志升序 → 概率降序 → 显示串升序的排序键
#[derive(PartialEq)]
struct FromFactProbabilityKey {
    from_fact: bool,
    probability: f64,
    display: String,
}

impl Eq for FromFactProbabilityKey {}

impl PartialOrd for FromFactProbabilityKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FromFactProbabilityKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.from_fact
            .cmp(&other.from_fact)
            .then_with(|| {
                other
                    .probability
                    .partial_cmp(&self.probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 数值绝对值降序、同值按显示串升序的排序键
#[derive(PartialEq)]
struct MagnitudeDisplayKey {
    magnitude: f64,
    display: String,
}

impl Eq for MagnitudeDisplayKey {}

impl PartialOrd for MagnitudeDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MagnitudeDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .magnitude
            .partial_cmp(&self.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

#[derive(Clone)]
pub struct DialogueContext {
    focus_entities: Vec<String>,
    /// 焦点实体的类型证据（代词消解的类型闸门；未知=键不存在）
    focus_types: HashMap<String, String>,
    last_predicate: Option<String>,
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
        self.generation = Some(GenerationState {
            entity: entity.to_string(),
            shown,
            order,
        });
    }

    pub fn snapshot(&self) -> DialogueSnapshot {
        DialogueSnapshot {
            focus_entities: self.focus_entities.clone(),
            focus_types: self.focus_types.clone(),
            last_predicate: self.last_predicate.clone(),
            history: self.history.clone(),
            generation: self.generation.clone(),
            session_facts: Vec::new(),
            session_negative_facts: Vec::new(),
            lessons: Vec::new(),
            session_clauses: Vec::new(),
            contrast: None,
            frames: None,
            ontology: None,
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueSnapshot) {
        self.focus_entities = snapshot.focus_entities.clone();
        self.focus_entities.truncate(self.max_focus);
        self.focus_types = snapshot.focus_types.clone();
        self.focus_types
            .retain(|entity, _| self.focus_entities.contains(entity));
        self.last_predicate = snapshot.last_predicate.clone();
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
    fn new(text: String) -> Self {
        Self {
            text,
            notes: Vec::new(),
        }
    }
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
        self.0.canonical_attribute(raw)
    }
}

pub struct SymbolicReasoner {
    config: ReasoningConfig,
    aligner: OntologyAligner,
    predictor: LinkPredictor,
    // 律典：声明律 + 挖掘律的统一台账；仅 Active 的传递/对称/互逆律参与查询扩展。
    laws: LawBook,
    // 律典的声明来源快照；知识变化时据此重建，避免把配置知识散落在查询路径。
    law_registry: Option<PredicateRegistry>,
    // 显式持久化的采纳/退役决定；每次重建律典后重新盖到声明/挖掘结果上。
    persisted_laws: Option<LawBook>,
    context: DialogueContext,
    embedding: SymbolicEmbedding,
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
    // 统一证据索引：词面倒排 × 结构 PPR × 谓词匹配，支持约束与可审计剪枝回执。
    // 用 Mutex 承载惰性重建，使只读回答路径也能在不改变 &self 契约下检索。
    retrieval: parking_lot::Mutex<DenseRetriever>,
    // 实体属性三元组（观察到的），供归结引擎编码事实
    attributes: HashSet<(String, String, String)>,
    // 属性查询索引：规范槽位 `(实体, 属性)` → 取值列表（随知识重建）
    attribute_index: HashMap<(String, String), Vec<String>>,
    // 关联规则挖掘器（类型/谓词/属性篮子的支持度-置信度-提升度）
    associations: AssociationMiner,
    // 一阶归结引擎（事实 + 链接预测规则 + 关联规则）
    resolver: ResolutionEngine,
    // 溯因·假设生成引擎（后向链失败前沿 → 候选假设 → 反事实模拟）
    abduction: AbductionEngine,
    // 话语帧记忆：带洞查询帧 + 挂账（合一式多轮消解，跨会话持久化）
    frames: FrameMemory,
    // 会话级临时事实（被采纳的假设；跨会话持久化，但永不写知识图谱）
    session_facts: Vec<Atom>,
    // 会话级否定事实：路径遍历中的阻断边（不写知识图谱，可跨会话持久化）
    session_negative_facts: Vec<Atom>,
    // 会话级 Horn 规则（用户用中文教的一般规则/条件句；供后向链归结使用）
    session_clauses: Vec<Clause>,
    // 当前澄清对话（把待定假设逆转为系统主动提问）
    clarification: Option<ClarificationSession>,
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
    // 引擎层故障通道（Prolog / 存储等下游失败由宿主注入）：
    // 非空时对话管理器走 EngineFault 路径，绝不与「知识里没有」合并。
    engine_fault: Option<String>,
}

/// 深拷贝：所有子引擎必须实现 `Clone`（锁/原子量以值重建，不共享可变状态）。
/// 内存记忆缓存（提及/链接/谓词）按内容复制，语义不变。
impl Clone for SymbolicReasoner {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            aligner: self.aligner.clone(),
            predictor: self.predictor.clone(),
            laws: self.laws.clone(),
            law_registry: self.law_registry.clone(),
            persisted_laws: self.persisted_laws.clone(),
            context: self.context.clone(),
            embedding: self.embedding.clone(),
            walks: self.walks.clone(),
            soft: self.soft.clone(),
            problog: self.problog.clone(),
            lessons: self.lessons.clone(),
            lessons_parsed: self.lessons_parsed.clone(),
            planner: self.planner.clone(),
            linker: self.linker.clone(),
            generator: self.generator.clone(),
            retrieval: parking_lot::Mutex::new(self.retrieval.lock().clone()),
            attributes: self.attributes.clone(),
            attribute_index: self.attribute_index.clone(),
            associations: self.associations.clone(),
            resolver: self.resolver.clone(),
            abduction: self.abduction.clone(),
            frames: self.frames.clone(),
            session_facts: self.session_facts.clone(),
            session_negative_facts: self.session_negative_facts.clone(),
            session_clauses: self.session_clauses.clone(),
            clarification: self.clarification.clone(),
            last_abduction_goal: self.last_abduction_goal.clone(),
            last_abduction_vars: self.last_abduction_vars.clone(),
            known_predicates_cache: self.known_predicates_cache.clone(),
            known_predicates_normalized_cache: self.known_predicates_normalized_cache.clone(),
            proposition_surfaces: self.proposition_surfaces.clone(),
            // 记忆缓存是纯派生数据：分叉时留空，由新会话按需重建
            // （空缓存与「重建后的缓存」在固定纪元下读数一致）
            link_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            entity_mentions_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            predicate_canonicals_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            inverse_query_map: self.inverse_query_map.clone(),
            discourse: self.discourse.clone(),
            holo: self.holo.clone(),
            revision: self.revision,
            derived_stamps: self.derived_stamps.clone(),
            dependency_stamps: self.dependency_stamps.clone(),
            perf: Arc::clone(&self.perf),
            nlp_config: self.nlp_config.clone(),
            // 轨迹不是记忆：分叉会话从空台账开始，避免把上一会话的失败带入
            read_traces: ReadTraceLedger::new(),
            engine_fault: self.engine_fault.clone(),
        }
    }
}

/// 语境指称消歧（CRD）的只读义项来源：复用生成器（类型/谓词/取值）与
/// 链接器（邻居锚点/度数），不新建词表、不重写链接打分。
const SENSE_CARD_ANCHOR_LIMIT: usize = 8;

impl SenseSource for SymbolicReasoner {
    fn sense_card(&self, entity: &str) -> Option<SenseCard> {
        let known = self.generator.is_known_entity(entity) || self.linker.is_known_entity(entity);
        if !known {
            return None;
        }
        let etype = self
            .linker
            .entity_type(entity)
            .map(str::to_string)
            .or_else(|| self.generator.entity_type_of(entity).map(str::to_string));
        let mut out_predicates = self.generator.relation_predicates(entity);
        out_predicates.extend(self.generator.attribute_keys(entity));
        out_predicates.sort();
        out_predicates.dedup();
        let in_predicates = self.generator.relation_predicates_in(entity);
        let mut anchors = self.linker.neighbors_of(entity);
        anchors.truncate(SENSE_CARD_ANCHOR_LIMIT);
        Some(SenseCard {
            entity: entity.to_string(),
            etype,
            out_predicates,
            in_predicates,
            anchors,
            degree: self.linker.degree_of(entity),
        })
    }

    fn predicate_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.generator.relation_values(entity, predicate)
    }

    fn predicate_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.generator.relation_values_in(entity, predicate)
    }
}

impl SymbolicReasoner {
    /// 把本体的实体/谓词登记转成句法重写器可用的词表视图：
    /// 只有已登记项才放行，未登记的一律返回 None（保守弃权）。
    fn relation_lexicon(&self) -> impl RelationLexicon + '_ {
        struct View<'a>(&'a SymbolicReasoner);

        impl RelationLexicon for View<'_> {
            fn canonical_entity(&self, raw: &str) -> Option<String> {
                // 只接受精确登记的词面/会话事实或显式同义词；不用模糊对齐
                // 把整个短语（如「张三的师傅」）误吸附到一个短实体上。
                let canonical = self.0.aligner.canonical_form(raw);
                if self.0.is_known_entity(raw) {
                    return Some(canonical);
                }
                let alias = canonical != raw
                    && self
                        .0
                        .aligner
                        .synonyms()
                        .get(raw)
                        .is_some_and(|target| target == &canonical);
                (alias && self.0.is_known_entity(&canonical)).then_some(canonical)
            }

            fn canonical_predicate(&self, raw: &str) -> Option<String> {
                // 谓词同样先做精确判定：已登记关系、精确事实谓词、逆谓词
                // 或同义词别名，任一成立才归一；模糊对齐不参与边界切分。
                let canonical = self.0.aligner.canonical_form(raw);
                let alias = canonical != raw
                    && self
                        .0
                        .aligner
                        .synonyms()
                        .get(raw)
                        .is_some_and(|target| target == &canonical);
                let exact = self.0.predictor.has_predicate(raw)
                    || self.0.aligner.kind_of(raw) == Some(TermKind::Relation)
                    || self.0.inverse_query_map.contains_key(raw)
                    || alias;
                exact.then(|| self.0.resolve_question_predicate(raw).0)
            }
        }

        View(self)
    }
    /// 审计辅助：词条是否是任一实体的属性键、取值或实体名本身。
    /// 供对话忠信度检查（回答引用必须可回溯）使用。
    pub fn attribute_token_known(&self, token: &str) -> bool {
        self.attribute_index
            .iter()
            .any(|((entity, attribute), values)| {
                entity == token
                    || attribute == token
                    || crate::symbolic_generator::localized_predicate_key(attribute) == Some(token)
                    || values.iter().any(|value| value == token)
            })
    }

    pub fn new(config: ReasoningConfig, synonyms: HashMap<String, String>) -> Self {
        Self::with_perf_ledger(config, synonyms, Arc::new(PerfLedger::new()))
    }

    /// 与 `new` 相同，但可注入共享的性能台账（会话级操作计数）。
    pub fn with_perf_ledger(
        config: ReasoningConfig,
        synonyms: HashMap<String, String>,
        perf: Arc<PerfLedger>,
    ) -> Self {
        let aligner = OntologyAligner::new(synonyms, config.ontology_alignment_threshold);
        let embedding = SymbolicEmbedding::new(config.embedding_similarity_floor);
        let mut linker = EntityLinker::new();
        for (alias, canonical) in aligner.synonyms() {
            linker.register_alias(alias, canonical);
        }
        let generator =
            SymbolicGenerator::new(config.max_profile_sentences, config.completion_min_score);
        let associations = AssociationMiner::new(
            config.association_min_support,
            config.association_min_confidence,
            config.association_min_lift,
        );
        let resolver =
            ResolutionEngine::new(config.resolution_max_depth, config.resolution_max_solutions);
        let abduction =
            AbductionEngine::new(config.resolution_max_depth, config.abduction_max_hypotheses);
        let mut abduction = abduction;
        abduction.set_max_scenario_size(config.abduction_max_scenario_size);
        let walks = GraphWalkEmbedding::new(
            config.graph_walk_damping,
            config.graph_walk_iterations,
            config.embedding_similarity_floor,
        );
        let soft = MarkovLogicEngine::new(
            config.soft_logic_max_iterations,
            config.soft_logic_min_probability,
        );
        let problog = ProblogEngine::new(config.problog_max_depth, config.problog_max_proofs);
        let discourse =
            DiscoursePlanner::new(config.discourse_max_rounds, config.discourse_max_sentences);
        let holo = HolographicMemory::new(
            config.holographic_dimension,
            config.holographic_similarity_floor,
            config.holographic_max_hits,
        );
        let mut reasoner = Self {
            config,
            aligner,
            predictor: LinkPredictor::new(),
            laws: LawBook::new(),
            law_registry: None,
            persisted_laws: None,
            context: DialogueContext::new(5),
            embedding,
            walks,
            soft,
            problog,
            lessons: Vec::new(),
            lessons_parsed: Vec::new(),
            planner: QueryPlanner::new(),
            linker,
            generator,
            retrieval: parking_lot::Mutex::new(DenseRetriever::new(8, 0.85, 12)),
            attributes: HashSet::new(),
            attribute_index: HashMap::new(),
            associations,
            resolver,
            abduction,
            frames: FrameMemory::new(),
            session_facts: Vec::new(),
            session_negative_facts: Vec::new(),
            session_clauses: Vec::new(),
            clarification: None,
            last_abduction_goal: None,
            last_abduction_vars: Vec::new(),
            known_predicates_cache: Vec::new(),
            known_predicates_normalized_cache: Vec::new(),
            proposition_surfaces: Vec::new(),
            link_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            entity_mentions_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            predicate_canonicals_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            inverse_query_map: HashMap::new(),
            discourse,
            holo,
            revision: 0,
            derived_stamps: BTreeMap::new(),
            dependency_stamps: BTreeMap::new(),
            perf,
            nlp_config: None,
            read_traces: ReadTraceLedger::new(),
            engine_fault: None,
        };
        reasoner.rebuild_knowledge_caches();
        reasoner
    }

    /// 从预热基座分叉一个新会话：知识派生状态按值深拷贝，会话态清零。
    ///
    /// 正确性契约：分叉结果与「新建实例 + 重放同一批 `observe_facts`」在固定
    /// 查询集上逐字节一致（同一知识版本、同一配置、同一同义词表），
    /// 由 `fork_matches_fresh_observe_byte_for_byte` 等测试守护。
    /// 分叉基座以 `Arc<str>` 共享，可跨线程使用（`Send`）。
    pub fn fork_session(&self) -> Self {
        let mut fork = self.clone();
        fork.reset_session_state();
        fork
    }

    /// 用一份新的图谱快照替换推理器的全局知识，同时保留会话态。
    ///
    /// `observe_facts` 只增不减；这里以同一配置、同义词与性能台账重建基座，
    /// 再恢复会话快照（焦点/历史/会话事实/规则/帧），
    /// 使查询立即看到更新后的图谱。
    pub fn rebuild_from_graph(&mut self, facts: &[PrologFact]) {
        let synonyms = self.aligner.synonyms().clone();
        let snapshot = self.dialogue_snapshot();
        let law_registry = self.law_registry.clone();
        let persisted_laws = self.persisted_laws.clone();
        let clarification = self.clarification.clone();
        let last_goal = self.last_abduction_goal.clone();
        let last_vars = self.last_abduction_vars.clone();
        let engine_fault = self.engine_fault.clone();
        let mut fresh =
            Self::with_perf_ledger(self.config.clone(), synonyms, Arc::clone(&self.perf));
        fresh.observe_facts(facts);
        // 快照来自自身：胶囊公理摘要必然通过（除非内存中的快照被外部改写）。
        // 万一被拒，也保留新图谱与其余会话态，不让重建失败丢掉知识。
        let _ = fresh.restore_dialogue_snapshot(&snapshot);
        if let Some(registry) = law_registry {
            fresh.configure_laws(&registry);
        }
        if let Some(persisted) = persisted_laws {
            fresh.load_persisted_laws(persisted);
        }
        fresh.clarification = clarification;
        fresh.last_abduction_goal = last_goal;
        fresh.last_abduction_vars = last_vars;
        fresh.engine_fault = engine_fault;
        *self = fresh;
    }

    /// 清空会话态：焦点/历史、会话事实与临时规则、澄清、教学、记忆缓存、
    /// 注入故障。知识派生状态（剖面/规则/向量/台账）原样保留。
    fn reset_session_state(&mut self) {
        self.context.clear();
        let lessons_were_taught = !self.lessons_parsed.is_empty();
        self.lessons.clear();
        self.lessons_parsed.clear();
        self.frames = FrameMemory::new();
        // 会话事实可能已被索引为证据；分叉/重置时把非图谱来源的条目撤出索引。
        let session_facts = std::mem::take(&mut self.session_facts);
        for atom in &session_facts {
            if !self.graph_supports_atom(atom) {
                self.unindex_session_atom(atom);
            }
        }
        self.session_negative_facts.clear();
        self.session_clauses.clear();
        self.clarification = None;
        self.last_abduction_goal = None;
        self.last_abduction_vars.clear();
        self.abduction.reset_session();
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        self.read_traces.clear();
        self.engine_fault = None;
        // 教学规则会随 lessons_parsed 重放进概率程序；只有真的教过才需要重建
        if lessons_were_taught && self.config.enable_problog {
            self.rebuild_problog();
        }
    }

    /// 注入引擎层故障（宿主在 Prolog / 存储失败后调用）：
    /// 之后对话管理器的知识问句会走 EngineFault 路径。
    pub fn note_engine_fault(&mut self, message: impl Into<String>) {
        let message = message.into();
        let message = message.trim();
        if !message.is_empty() {
            self.engine_fault = Some(message.to_string());
        }
    }

    /// 清除引擎层故障（默认实现里故障是粘性的，仅在宿主确认恢复后清除）。
    pub fn clear_engine_fault(&mut self) {
        self.engine_fault = None;
    }

    /// 当前引擎层故障（无则 None）。
    pub fn engine_fault(&self) -> Option<String> {
        self.engine_fault.clone()
    }

    /// 当前知识版本（每次内容确有变化的观察 +1）。
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// 共享的性能台账（与 `OxigraphManager` 可为同一实例）。
    pub fn perf_ledger(&self) -> Arc<PerfLedger> {
        Arc::clone(&self.perf)
    }

    /// 注入 NLP 配置，启用读法观测。失败轮会留下只读轨迹回执；
    /// 成功轮不额外分词。未注入时失败轮保持原有的零开销回答路径。
    pub fn set_nlp_config(&mut self, config: NlpConfig) {
        self.nlp_config = Some(config);
        self.read_traces.clear();
    }

    /// 构建一次知识感知的读法轨迹（只读；不执行回答）。
    pub fn lexical_trace(&self, question: &str) -> Option<LexicalTrace> {
        let config = self.nlp_config.as_ref()?;
        let plan = self.planner.parse(question).or_else(|| {
            query_rewrite::rewrite(question, &self.relation_lexicon()).map(|rewrite| rewrite.plan)
        });
        let mut trace = lexical_trace::trace_question(question, config, plan.as_ref());
        trace.fidelity = TraceFidelity::Reconstructed;
        if let Some(plan) = &plan {
            self.enrich_trace_plan_terms(&mut trace, plan);
        }
        self.enrich_trace_oov(&mut trace, question);
        Some(trace)
    }

    /// 读法追踪报告：token/来源/边界/计划/失败分类，并标注回答管线是否命中。
    pub fn lexical_trace_report(&self, question: &str) -> String {
        let Some(mut trace) = self.lexical_trace(question) else {
            return "读法追踪未启用：推理器没有注入 NLP 配置。".to_string();
        };
        let answered = self.answer_impl(question).is_some();
        trace.answer_available = Some(answered);
        if !answered
            && trace.plan.is_some()
            && !trace.failures.iter().any(|failure| {
                matches!(
                    failure.kind,
                    ReadFailureKind::UnknownEntity
                        | ReadFailureKind::UnknownPredicate
                        | ReadFailureKind::ClosureMarkerInToken
                )
            })
        {
            trace.push_failure(
                ReadStage::Execute,
                ReadFailureKind::ExecutionMiss,
                "计划已识别，但回答管线没有产生结果（事实缺口/缺少推理律/截断需在推理侧继续区分）",
            );
        }
        self.perf.bump("lexical_trace_reports");
        self.perf.bump_by(
            "lexical_trace_unknown_tokens",
            trace.unknown_tokens().count() as u64,
        );
        self.perf.bump_by(
            "lexical_trace_marker_violations",
            trace.boundary_violations() as u64,
        );
        self.perf
            .bump_by("lexical_trace_oov", trace.oov.len() as u64);
        trace.render_report()
    }

    /// 有界失败回执台账（不含原文）；进程重启即失，不进入会话快照。
    pub fn read_trace_ledger_report(&self) -> String {
        self.read_traces.report()
    }

    pub fn read_trace_receipts(&self) -> Vec<TraceReceipt> {
        self.read_traces.snapshot()
    }

    fn enrich_trace_plan_terms(&self, trace: &mut LexicalTrace, plan: &QueryPlan) {
        let (entities, predicates) = lexical_trace::plan_terms(plan);
        for entity in entities {
            if !self.is_known_entity(&entity) {
                trace.push_failure(
                    ReadStage::Link,
                    ReadFailureKind::UnknownEntity,
                    format!("实体槽「{}」未在知识图谱登记", entity),
                );
            }
        }
        for predicate in predicates {
            if !self.is_known_predicate(&predicate) {
                trace.push_failure(
                    ReadStage::Link,
                    ReadFailureKind::UnknownPredicate,
                    format!("谓词槽「{}」未在谓词画像/事实中登记", predicate),
                );
            }
        }
    }

    fn enrich_trace_oov(&self, trace: &mut LexicalTrace, question: &str) {
        let expectations = crate::function_words::answer_type_keywords(question);
        let hints = self.link_hints(question);
        let receipts =
            self.linker
                .mention_receipts(question, &hints, &expectations, self.context.focus());
        trace.oov = crate::entity_linker::oov_suspects(question, &receipts)
            .into_iter()
            .map(|suspect| crate::lexical_trace::OovTrace {
                text: suspect.text,
                start: suspect.start,
                kind: match suspect.kind {
                    crate::entity_linker::OovKind::Quoted => "quoted".to_string(),
                    crate::entity_linker::OovKind::Suffix => "suffix".to_string(),
                },
                reason: suspect.reason,
                suggested_type: suspect.suggested_type,
            })
            .collect();
    }

    /// 失败轮的只读轨迹：有界回执 + 有限枚举指标。成功轮不调用。
    fn record_read_failure(&self, question: &str) {
        let Some(mut trace) = self.lexical_trace(question) else {
            return;
        };
        trace.answer_available = Some(false);
        if trace.plan.is_some()
            && trace.failures.iter().all(|failure| {
                !matches!(
                    failure.kind,
                    ReadFailureKind::UnknownEntity
                        | ReadFailureKind::UnknownPredicate
                        | ReadFailureKind::NoQuestionShape
                )
            })
        {
            trace.push_failure(
                ReadStage::Execute,
                ReadFailureKind::ExecutionMiss,
                "计划已识别，但回答管线没有产生结果",
            );
        }
        let receipt = self.read_traces.record(&trace);
        self.perf.bump("lexical_trace_failures");
        self.perf.bump_by(
            "lexical_trace_unknown_tokens",
            trace.unknown_tokens().count() as u64,
        );
        self.perf.bump_by(
            "lexical_trace_marker_violations",
            trace.boundary_violations() as u64,
        );
        self.perf
            .bump_by("lexical_trace_oov", trace.oov.len() as u64);
        for stage in &receipt.failure_stages {
            self.perf.bump(&format!("lexical_trace_stage_{}", stage));
        }
        for kind in &receipt.failure_kinds {
            self.perf.bump(&format!("lexical_trace_failure_{}", kind));
        }
    }

    /// 记录一次派生重建：派生器戳上当前知识版本，并在性能台账计一次数。
    fn stamp_rebuild(&mut self, deriver: &'static str) {
        self.derived_stamps.insert(deriver, self.revision);
        self.perf.bump(match deriver {
            "link_rules" => "rebuild_link_rules",
            "associations" => "rebuild_associations",
            "resolver" => "rebuild_resolver",
            "soft_logic" => "rebuild_soft_logic",
            "problog" => "rebuild_problog",
            "knowledge_caches" => "rebuild_knowledge_caches",
            _ => "rebuild_other",
        });
    }

    /// 记录某类依赖发生变化的版本。
    fn stamp_dependency(&mut self, dependency: &'static str) {
        self.dependency_stamps.insert(dependency, self.revision);
    }

    /// 派生台账报告：知识版本 + 各派生器最后重建版本 + 各依赖最后变化版本。
    ///
    /// 「重复观察不触发重建」因此可审计：版本不动 → 台账不动。
    pub fn derivation_report(&self) -> String {
        let mut output = format!("派生台账：知识版本 {}。\n", self.revision);
        if self.dependency_stamps.is_empty() {
            output.push_str("依赖变化：暂无。\n");
        } else {
            output.push_str("依赖变化（最后版本）：");
            let items: Vec<String> = self
                .dependency_stamps
                .iter()
                .map(|(name, revision)| format!("{}={}", name, revision))
                .collect();
            output.push_str(&items.join("；"));
            output.push('\n');
        }
        if self.derived_stamps.is_empty() {
            output.push_str("派生重建：暂无。\n");
        } else {
            output.push_str("派生重建（最后版本）：");
            let items: Vec<String> = self
                .derived_stamps
                .iter()
                .map(|(name, revision)| format!("{}={}", name, revision))
                .collect();
            output.push_str(&items.join("；"));
            output.push('\n');
        }
        output
    }

    /// 重建谓词清单与命题表面积索引（二者仅依赖对齐器/生成器/预测规则的状态）
    fn rebuild_knowledge_caches(&mut self) {
        self.known_predicates_cache = self.collect_known_predicates();
        self.known_predicates_normalized_cache = self
            .known_predicates_cache
            .iter()
            .map(|predicate| (predicate.clone(), self.aligner.normalize(predicate)))
            .collect();
        self.proposition_surfaces = semantic_entailment::build_predicate_surfaces(
            &self.aligner,
            self.generator.predicate_surfaces(),
        );
        // 属性查询索引按规范槽位重建（同义词/谓词别名变化后仍可命中）
        let mut attribute_index: HashMap<(String, String), Vec<String>> = HashMap::new();
        for (subject, key, value) in &self.attributes {
            let slot = self.aligner.canonical_form(key);
            attribute_index
                .entry((subject.clone(), slot))
                .or_default()
                .push(value.clone());
        }
        for values in attribute_index.values_mut() {
            values.sort();
            values.dedup();
        }
        self.attribute_index = attribute_index;
        // 词表变化后实体链接/实体提及的缓存结果可能过时，同步失效记忆缓存
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        // 律典是谓词画像与当前挖掘规则的纯派生：知识/规则变化后同批重建。
        self.rebuild_laws();
        self.stamp_rebuild("knowledge_caches");
    }

    pub fn context(&self) -> &DialogueContext {
        &self.context
    }

    /// 话语帧记忆（只读）：上一完整帧 + 挂账（未竟之问）
    pub fn frame_memory(&self) -> &FrameMemory {
        &self.frames
    }

    /// 话语帧记忆报告：帧、洞填充来源、挂账与剩余轮数（确定性、可断言）
    pub fn frame_memory_report(&self) -> String {
        self.frames.report()
    }

    /// 帧澄清是否进行中（无源可填的洞已转为等待一次显式绑定）
    pub fn frame_clarification_active(&self) -> bool {
        self.frames.clarification().is_some()
    }

    /// 帧澄清提示（候选编号列表；确定性渲染）
    pub fn frame_clarification_prompt(&self) -> Option<String> {
        self.frames
            .clarification()
            .map(dialogue_frame::ClarificationRequest::prompt)
    }

    /// 帧澄清入口：无上下文的「体重是多少」这类带洞问句在常规通道必然失败时，
    /// 不落进「无法理解」，而是挂成一次显式绑定请求（变量求解的第 4 种结局）。
    ///
    /// 开启条件（全部满足才接管）：
    ///   1) 配置启用澄清，且没有进行中的溯因/帧澄清（至多一条）；
    ///   2) 问句是无主语属性问句（`属性+是多少/是什么`），但不含「呢」
    ///      （「体重呢」属于常规省略/对话管理器通道）；
    ///   3) 常规通道必然失败：上一完整帧为空、挂账为空（否则应走合一补全）；
    ///   4) 候选非空：焦点实体优先，其余是知识库中确实拥有该属性的实体。
    pub fn start_frame_clarification(&mut self, input: &str) -> Option<String> {
        if !self.config.enable_clarification || self.clarification_active() {
            return None;
        }
        if self.frames.clarification().is_some()
            || self.frames.last().is_some()
            || !self.frames.pending().is_empty()
        {
            return None;
        }
        let trimmed = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if trimmed.ends_with('呢') {
            return None;
        }
        let turn = self.frames.turn();
        let fragment =
            dialogue_frame::parse_bare_attribute_question(input, &FrameOntology(self), turn)?;
        let attribute = fragment.attribute.clone()?;
        let candidates = self.frame_clarification_candidates(&attribute);
        if candidates.is_empty() {
            return None;
        }
        let request = self
            .frames
            .begin_clarification(input.trim(), &attribute, candidates);
        Some(request.prompt())
    }

    /// 处理对帧澄清的短答：序号/候选名 → 绑定；越界 → 重问；都不是 → 取消。
    /// 已知实体名（即使不在候选表里）同样接受为一次显式绑定；
    /// 其余回答返回 None（放行给常规管线，零状态变化）。
    pub fn answer_frame_clarification(
        &mut self,
        text: &str,
    ) -> Option<dialogue_frame::ClarificationOutcome> {
        let request = self.frames.clarification()?.clone();
        match request.parse_reply(text) {
            dialogue_frame::ClarificationReply::Candidate(entity) => {
                self.frames.commit_clarification(&entity)
            }
            dialogue_frame::ClarificationReply::OutOfRange => {
                Some(dialogue_frame::ClarificationOutcome::Reprompt {
                    prompt: request.prompt(),
                })
            }
            dialogue_frame::ClarificationReply::Cancel => self.frames.cancel_clarification(),
            dialogue_frame::ClarificationReply::Unmatched => {
                let entity = FrameOntology(self).canonical_entity(text)?;
                self.frames.commit_clarification(&entity)
            }
        }
    }

    /// 澄清候选：焦点实体 ∩ 有该属性的实体 → 有该属性的实体（名称序）→ 其余焦点实体。
    /// 上限 5 条；顺序确定（可 golden），焦点只作为先验来源，不引入新猜测。
    fn frame_clarification_candidates(&self, attribute: &str) -> Vec<String> {
        const MAX_CANDIDATES: usize = 5;
        let mut with_attribute: Vec<String> = self
            .attribute_index
            .iter()
            .filter(|((_, slot), values)| slot == attribute && !values.is_empty())
            .map(|((entity, _), _)| entity.clone())
            .collect();
        with_attribute.sort();
        with_attribute.dedup();
        let available: HashSet<&str> = with_attribute.iter().map(String::as_str).collect();
        let mut ranked: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for focus in self.context.focus_entities() {
            if available.contains(focus.as_str()) && seen.insert(focus.clone()) {
                ranked.push(focus.clone());
            }
        }
        for entity in &with_attribute {
            if ranked.len() >= MAX_CANDIDATES {
                break;
            }
            if seen.insert(entity.clone()) {
                ranked.push(entity.clone());
            }
        }
        for focus in self.context.focus_entities() {
            if ranked.len() >= MAX_CANDIDATES {
                break;
            }
            if self.is_known_entity(focus) && seen.insert(focus.clone()) {
                ranked.push(focus.clone());
            }
        }
        ranked.truncate(MAX_CANDIDATES);
        ranked
    }

    pub fn embedding(&self) -> &SymbolicEmbedding {
        &self.embedding
    }

    pub fn walks(&self) -> &GraphWalkEmbedding {
        &self.walks
    }

    /// 全息符号记忆（VSA/HDC 软召回通道 + 三元组硬核查通道）
    pub fn holographic(&self) -> &HolographicMemory {
        &self.holo
    }

    /// 把一段文本里的已知实体压入全息时序记忆（多轮对话的"注意力轨迹"）
    pub fn note_holo_context(&mut self, text: &str) {
        if !self.config.enable_holographic_memory {
            return;
        }
        for entity in self.entities_mentioned(text) {
            self.holo.push_focus(&entity);
        }
    }

    pub fn soft_logic(&self) -> &MarkovLogicEngine {
        &self.soft
    }

    pub fn problog(&self) -> &ProblogEngine {
        &self.problog
    }

    /// 已学的中文概率知识原句（教学记忆）
    pub fn lessons(&self) -> &[String] {
        &self.lessons
    }

    pub fn linker(&self) -> &EntityLinker {
        &self.linker
    }

    pub fn generator(&self) -> &SymbolicGenerator {
        &self.generator
    }

    /// 类型化配方查询：按方向（实体→答案 / 答案→实体）取关系值，
    /// 用当前单一本体的物化闭包补齐逆关系与子类推断，并按答案类型做
    /// **开放世界**过滤——未知类型放行，只有显式互斥的已知冲突才拒答。
    ///
    /// 返回的 `evidence` 逐条复述事实边/公理推导理由，供插件模板 `{evidence}`
    /// 接地生成；没有取值时返回 `None`，调用方应让位核心管线。
    pub fn recipe_lookup(
        &mut self,
        entity: &str,
        attribute: &str,
        direction: RecipeDirection,
        answer_types: &[String],
    ) -> Option<RecipeLookup> {
        let entity = entity.trim();
        let attribute = attribute.trim();
        if entity.is_empty() || attribute.is_empty() {
            return None;
        }
        let mut predicates: BTreeSet<String> = BTreeSet::new();
        predicates.insert(attribute.to_string());
        let canonical = self.aligner.canonical_form(attribute);
        if !canonical.trim().is_empty() {
            predicates.insert(canonical.trim().to_string());
        }

        let closure = self.discourse.narrator_mut().closure().clone();
        // 主语侧类型约束：正向问句的实体应落在谓词定义域内，逆向问句的
        // 实体应落在值域内。由「关系定义域/值域反推」得到的类型是循环
        // 证据（类型恰由被查关系自身反推），不作数；开放世界下有效类型
        // 未知即放行，已知类型与全部允许类互斥才拒答——减少「关系同名
        // 但类型不符」的误答。
        let mut expected_anchor_types: BTreeSet<String> = BTreeSet::new();
        for predicate in &predicates {
            if direction.is_inverse() {
                expected_anchor_types.extend(closure.range_classes(predicate));
            } else {
                expected_anchor_types.extend(closure.domain_classes(predicate));
            }
        }
        if !expected_anchor_types.is_empty() {
            let declared_types: Vec<String> = closure
                .types_of(entity)
                .into_iter()
                .filter(|class| {
                    !closure
                        .why_instance(entity, class)
                        .map(|reason| reason.contains("的定义域是") || reason.contains("的值域是"))
                        .unwrap_or(false)
                })
                .collect();
            if !declared_types.is_empty() {
                let compatible = declared_types.iter().any(|actual| {
                    expected_anchor_types
                        .iter()
                        .any(|expected| actual == expected || closure.is_subclass(actual, expected))
                });
                if !compatible
                    && declared_types.iter().any(|actual| {
                        expected_anchor_types
                            .iter()
                            .any(|expected| closure.are_disjoint(actual, expected))
                    })
                {
                    return None;
                }
            }
        }

        let mut raw_values: BTreeSet<String> = BTreeSet::new();
        for predicate in &predicates {
            if direction.is_inverse() {
                raw_values.extend(self.generator.relation_values_in(entity, predicate));
            } else {
                raw_values.extend(self.generator.attribute_values(entity, predicate));
                raw_values.extend(self.generator.relation_values(entity, predicate));
            }
        }

        for (predicate, object) in closure.properties_of(entity) {
            if predicates.contains(&predicate) {
                raw_values.insert(object);
            }
        }
        if direction.is_inverse() {
            for (subject, predicate, object) in closure.relations_snapshot() {
                if object == entity && predicates.contains(&predicate) {
                    raw_values.insert(subject);
                }
            }
        }

        if raw_values.is_empty() {
            return None;
        }
        if !answer_types.is_empty() {
            raw_values.retain(|value| {
                let known_types = closure.types_of(value);
                if known_types.is_empty() {
                    return true;
                }
                let compatible = answer_types.iter().any(|expected| {
                    closure.is_instance(value, expected)
                        || known_types
                            .iter()
                            .any(|actual| closure.is_subclass(actual, expected))
                });
                if compatible {
                    return true;
                }
                // 开放世界：已知类型但无互斥证据时仍放行，只有显式冲突才拒答。
                !known_types.iter().any(|actual| {
                    answer_types
                        .iter()
                        .any(|expected| closure.are_disjoint(actual, expected))
                })
            });
            if raw_values.is_empty() {
                return None;
            }
        }

        let mut evidence: Vec<String> = Vec::new();
        for (subject, predicate, object) in closure.relations_snapshot() {
            if !predicates.contains(&predicate) {
                continue;
            }
            let (anchor, answer) = if direction.is_inverse() {
                (object.as_str(), subject.clone())
            } else {
                (subject.as_str(), object.clone())
            };
            if anchor != entity || !raw_values.contains(&answer) {
                continue;
            }
            let why = closure
                .why_relation(&subject, &predicate, &object)
                .unwrap_or("直接声明");
            evidence.push(format!(
                "{} —{}→ {}（{}）",
                subject, predicate, object, why
            ));
        }
        if evidence.is_empty() {
            evidence.push(format!("图谱属性记录：{}的{}", entity, attribute));
        }
        // 事实边优先保留，类型证明只在剩余额度内补充；不能因为答案多值
        // 就把「哪条事实支持了这个值」挤出依据列表。
        evidence.sort();
        evidence.dedup();
        evidence.truncate(6);
        // 类型约束用的不是问答模板里的猜测：把每个答案的已知类型及其推导理由
        // 一并复述，用户能看到「为什么这个值符合 answer_types」。
        if !answer_types.is_empty() {
            let mut type_evidence = Vec::new();
            for value in &raw_values {
                let known_types = closure.types_of(value);
                for actual in &known_types {
                    let compatible = answer_types.iter().any(|expected| {
                        actual == expected || closure.is_subclass(actual, expected)
                    });
                    if !compatible {
                        continue;
                    }
                    let most_specific = !known_types.iter().any(|other| {
                        other != actual && closure.is_subclass(other, actual)
                    });
                    if !most_specific {
                        continue;
                    }
                    let why = closure
                        .why_instance(value, actual)
                        .unwrap_or("类型继承");
                    type_evidence.push(format!("{} 属于 {}（{}）", value, actual, why));
                }
            }
            type_evidence.sort();
            type_evidence.dedup();
            for line in type_evidence {
                if evidence.len() >= 8 {
                    break;
                }
                if !evidence.contains(&line) {
                    evidence.push(line);
                }
            }
        }

        let values: Vec<String> = raw_values
            .into_iter()
            .map(|value| {
                crate::identity::local_name(crate::identity::DEFAULT_BASE_NS, &value)
                    .unwrap_or(value)
            })
            .collect();
        Some(RecipeLookup { values, evidence })
    }

    /// 执行一次插件配方：回指焦点解析 → 类型化查值（含逆关系/子类/开放世界
    /// 类型约束）→ 核心管线兜底 → 模板接地。CLI 与 HTTP 都只调用这里，
    /// 避免两条入口在方向、依据或回指处理上各自演化。
    pub fn answer_plugin_recipe(
        &mut self,
        outcome: &PluginOutcome,
        runtime: &mut PluginRuntime,
        focus_entity: Option<&str>,
    ) -> Option<PluginRecipeReply> {
        let recipe = outcome.trace.recipe.clone()?;
        let raw_entity = recipe.slots.get("entity").cloned();
        let entity = raw_entity.as_ref().map(|raw| {
            if runtime.is_context_reference(raw) {
                focus_entity
                    .map(str::to_string)
                    .unwrap_or_else(|| raw.clone())
            } else {
                raw.clone()
            }
        });
        // 回指替换后同步改写问句，核心管线兜底时不再拿「他」去检索。
        let question = match (raw_entity.as_deref(), entity.as_deref()) {
            (Some(raw), Some(resolved)) if raw != resolved => {
                recipe.question.replace(raw, resolved)
            }
            _ => recipe.question.clone(),
        };
        let mut answer = None;
        let mut answer_values = Vec::new();
        let mut evidence = Vec::new();
        if let (Some(entity), Some(attribute)) = (entity.as_deref(), recipe.attribute.as_deref()) {
            if let Some(lookup) = self.recipe_lookup(
                entity,
                attribute,
                recipe.direction,
                &recipe.answer_types,
            ) {
                if !lookup.values.is_empty() {
                    answer_values = lookup.values.clone();
                    answer = Some(lookup.values.join("、"));
                    evidence = lookup.evidence;
                }
            }
        }
        if answer.is_none() {
            answer = self.answer(&question);
        }
        let answer = answer?;
        // 正向问句的答案通常是下一轮的话题（阿凡达 → 卡梅隆）；逆向问句
        // 的主语本来就是话题（卡梅隆 → 阿凡达），不能被答案覆盖。
        let focus = if recipe.direction.is_inverse() {
            entity.clone()
        } else {
            answer_values.first().cloned().or_else(|| entity.clone())
        };
        let composed = runtime.compose_answer_with_evidence(outcome, &answer, &evidence);
        let plan = vec![format!("plugin:{}:{}", recipe.plugin, recipe.contribution)];
        let provenance = vec![format!(
            "插件 {} / 配方 {} / 问句 {}",
            recipe.plugin, recipe.contribution, question
        )];
        Some(PluginRecipeReply {
            text: composed.text,
            plan,
            provenance,
            entity,
            attribute: recipe.attribute,
            question,
            kind: "attribute",
            focus,
        })
    }

    /// 在当前知识快照上执行带约束/预算的证据检索（含可审计回执）。
    ///
    /// 索引随 `observe_facts` 增量观察，首次查询时惰性重建倒排表；
    /// 结果与独立调用 `DenseRetriever::observe_facts + search` 一致。
    pub fn evidence_search(
        &self,
        query: &str,
        k: usize,
        constraints: &SearchConstraints,
        budget: SearchBudget,
    ) -> EvidenceSearch {
        self.retrieval.lock().search(query, k, constraints, budget)
    }

    /// 证据检索的文本视图：排序证据 + 检索回执。
    pub fn evidence_report(
        &self,
        query: &str,
        k: usize,
        constraints: &SearchConstraints,
        budget: SearchBudget,
    ) -> String {
        let search = self.evidence_search(query, k, constraints, budget);
        format!(
            "{}\n\n{}",
            self.retrieval.lock().format_hits(query, &search.hits),
            search.receipt.render_text()
        )
    }

    /// 运行时登记谓词别名（来自谓词画像准入）：查询解析与生成词面同步归一。
    pub fn register_predicate_alias(&mut self, alias: &str, canonical: &str) -> bool {
        if alias.trim().is_empty() || canonical.trim().is_empty() || alias == canonical {
            return false;
        }
        self.aligner.register_term(canonical, TermKind::Relation);
        let synonym_added = self.aligner.register_synonym(alias, canonical);
        let surface_added = self.generator.register_predicate_alias(alias, canonical);
        if synonym_added || surface_added {
            self.rebuild_knowledge_caches();
        }
        synonym_added || surface_added
    }

    /// 登记实体别名（来自实体身份注册表已确认的持久别名）：链接器按别名识别，
    /// 与 KB 里的规范名同权；歧义词面如实保留多候选，不静默择一。
    ///
    /// 已确认别名同时进对齐器：问答路径按规范名归一实体（「确认身份」签过字
    /// 的说法要能直接答出来），链接器仍负责候选枚举与接地引用，两处同源。
    pub fn register_entity_alias(&mut self, alias: &str, canonical: &str) -> bool {
        let (alias, canonical) = (alias.trim(), canonical.trim());
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return false;
        }
        self.linker.register_alias(alias, canonical);
        self.aligner.register_term(canonical, TermKind::Entity);
        self.aligner.register_term(alias, TermKind::Entity);
        self.aligner.register_synonym(alias, canonical);
        self.link_memo.lock().clear();
        true
    }

    /// 登记逆向谓词词面：问句用该词面发问时，命中规范槽位三元组的反方向
    ///（「徒弟」问句 ↔「师傅」事实）。只影响查询换向，不改词面/叙事表。
    /// 返回是否新增/改变；重复登记返回 `false`。
    pub fn register_predicate_query_surface(&mut self, surface: &str, slot: &str) -> bool {
        let (surface, slot) = (surface.trim(), slot.trim());
        if surface.is_empty() || slot.is_empty() || surface == slot {
            return false;
        }
        self.aligner.register_term(surface, TermKind::Relation);
        self.aligner.register_term(slot, TermKind::Relation);
        let changed = self.inverse_query_map.get(surface) != Some(&(slot.to_string(), true));
        self.inverse_query_map
            .insert(surface.to_string(), (slot.to_string(), true));
        if changed {
            self.rebuild_knowledge_caches();
        }
        changed
    }

    /// 把已启用插件的本体贡献单调并入同一个推理器：谓词别名、互逆词面与
    /// 结构性公理。返回本次真正新增的条数；停用插件的撤回不在本会话发生。
    pub fn apply_plugin_runtime(&mut self, runtime: &PluginRuntime) -> PluginOntologySync {
        let mut sync = PluginOntologySync::default();
        for (alias, canonical) in runtime.ontology_aliases() {
            if self.register_predicate_alias(&alias, &canonical) {
                sync.aliases += 1;
            }
        }
        for (surface, slot) in runtime.ontology_inverse_surfaces() {
            if self.register_predicate_query_surface(&surface, &slot) {
                sync.inverse_surfaces += 1;
            }
        }
        sync.lessons = self.configure_ontology_program(&runtime.ontology_program());
        sync
    }

    /// 谓词对齐批量注入（交互与服务共用同一入口）：三元组
    /// `(词面, 规范槽位, 是否换向)`；正向走别名归一，换向只登记查询转置。
    /// 返回本次真正生效的条数（重复/退化条目不计）。
    pub fn align_predicate_pairs(&mut self, pairs: &[(String, String, bool)]) -> usize {
        let mut applied = 0usize;
        for (surface, slot, inverted) in pairs {
            if surface == slot {
                continue;
            }
            let registered = if *inverted {
                self.register_predicate_query_surface(surface, slot)
            } else {
                self.register_predicate_alias(surface, slot)
            };
            if registered {
                applied += 1;
            }
        }
        applied
    }

    /// 实体身份别名批量注入（已确认的持久别名与规范名同权），返回生效条数。
    pub fn align_entity_aliases(&mut self, aliases: &[(String, String)]) -> usize {
        let mut applied = 0usize;
        for (surface, canonical) in aliases {
            if self.register_entity_alias(surface, canonical) {
                applied += 1;
            }
        }
        applied
    }

    /// 配置律典：把谓词画像中的显式声明编译为 Active 律，并吸收当前挖掘规则。
    /// 可反复调用；知识变化后由 `rebuild_knowledge_caches` 自动重放同一来源。
    pub fn configure_laws(&mut self, registry: &PredicateRegistry) {
        self.law_registry = Some(registry.clone());
        self.rebuild_laws();
    }

    /// 载入持久化账本中的显式采纳/退役决定，并立即重放律典。
    /// 返回载入的显式决定条数（Active/Retired）。
    pub fn load_persisted_laws(&mut self, persisted: LawBook) -> usize {
        let loaded = persisted
            .laws()
            .filter(|law| matches!(law.status, LawStatus::Active | LawStatus::Retired))
            .count();
        self.persisted_laws = Some(persisted);
        self.rebuild_laws();
        loaded
    }

    /// 律典快照（只读派生；调用方审计时在其副本上取反例）。
    pub fn law_book_snapshot(&self) -> LawBook {
        self.laws.clone()
    }

    /// 当前律典 + 对现有事实的确定性审计（交互命令与评测共用）。
    pub fn law_report(&self) -> String {
        let mut book = self.laws.clone();
        let audit = book.audit(self.predictor.triples());
        format!("{}{}", book.render_text(), audit.render_text())
    }

    fn rebuild_laws(&mut self) {
        let Some(registry) = &self.law_registry else {
            self.laws = self.persisted_laws.clone().unwrap_or_default();
            return;
        };
        let mut book = LawBook::from_registry(registry);
        book.add_mined_rules(self.predictor.rules());
        if let Some(persisted) = &self.persisted_laws {
            book.apply_persisted(persisted);
        }
        self.laws = book;
    }

    /// 问句谓词解析：对齐器归一后，若为反向词面则返回 (槽位, true)。
    /// 与「谓词画像」约定一致：逆向词面的答案是槽位三元组的反方向。
    fn resolve_question_predicate(&self, raw: &str) -> (String, bool) {
        let raw = raw.trim();
        if let Some((slot, inverted)) = self.inverse_query_map.get(raw) {
            return (slot.clone(), *inverted);
        }
        let canonical = self.aligner.canonical_form(raw);
        if let Some((slot, inverted)) = self.inverse_query_map.get(canonical.as_str()) {
            return (slot.clone(), *inverted);
        }
        (canonical, false)
    }

    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        self.perf.bump("observe_calls");
        self.perf.bump_by("observe_facts_input", facts.len() as u64);
        // 统一证据索引与知识图谱同源观察（内部去重；重建按需惰性触发）
        self.retrieval.lock().observe_facts(facts);
        // 话语引擎：本体叙事与 DCG 生成共用同一批事实（内部去重，可反复观察）
        if self.config.enable_discourse {
            self.discourse.observe_facts(facts);
        }
        // 溯因统计只累积计数（谓词取值、类型分布、特征），不参与学习
        self.abduction.observe_facts(facts);
        let mut relation_seen = 0usize;
        let mut entity_new = false;
        let mut entity_type_changed = false;
        let mut association_changed = false;
        let mut attribute_changed = false;
        let triples_before = self.predictor.triple_count();
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    self.aligner.observe_triple(from, rel_type, to);
                    self.predictor.add_triple(from, rel_type, to);
                    self.linker.observe_triple(from, rel_type, to);
                    self.generator.observe_triple(from, rel_type, to);
                    if self.config.enable_semantic_embedding {
                        self.embedding.observe_triple(from, rel_type, to);
                    }
                    if self.config.enable_graph_walk {
                        self.walks.observe_triple(from, rel_type, to);
                    }
                    if self.config.enable_holographic_memory {
                        self.holo.observe_triple(from, rel_type, to);
                    }
                    association_changed |= self.associations.observe_triple(from, rel_type);
                    relation_seen += 1;
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    self.aligner.register_term(name, TermKind::Entity);
                    self.aligner.register_term(entity_type, TermKind::Type);
                    for property in properties.keys() {
                        self.aligner.register_term(property, TermKind::Attribute);
                    }
                    // 新实体同样会改变实体提及/链接剖面，必须触发词表缓存重建
                    if !self.linker.is_known_entity(name) {
                        entity_new = true;
                    }
                    self.linker
                        .register_entity(name, Some(entity_type.as_str()));
                    // 生成器的实体类型会被后续观察覆盖，归结引擎的 __类型__ 事实依赖它：
                    // 类型变化时必须触发重建（关联篮子的变化无法覆盖此情形）
                    if !entity_type.trim().is_empty()
                        && self.generator.entity_type_of(name) != Some(entity_type.as_str())
                    {
                        entity_type_changed = true;
                    }
                    self.generator
                        .observe_entity_map(name, entity_type, properties);
                    if self.config.enable_semantic_embedding {
                        self.embedding
                            .observe_entity_map(name, entity_type, properties);
                    }
                    association_changed |=
                        self.associations
                            .observe_entity_map(name, entity_type, properties);
                    for (key, value) in properties {
                        if self
                            .attributes
                            .insert((name.clone(), key.clone(), value.clone()))
                        {
                            attribute_changed = true;
                        }
                        if self.config.enable_holographic_memory {
                            self.holo.observe_triple(name, key, value);
                        }
                    }
                }
            }
        }
        let triples_added = self.predictor.triple_count() > triples_before;
        self.perf.bump_by("observe_relations", relation_seen as u64);
        // 版本化增量：内容确有变化 → 版本 +1、依赖挂牌、派生重建；
        // 重复观察（快照重放）→ 版本不动、memo 不清、派生不重建
        let changed = triples_added
            || association_changed
            || attribute_changed
            || entity_type_changed
            || entity_new;
        if !changed {
            self.perf.bump("observe_no_change");
            return;
        }
        self.revision += 1;
        self.perf.bump("observe_changed");
        if triples_added {
            self.perf.bump("observe_new_triples");
            self.stamp_dependency("triples");
        }
        if attribute_changed {
            self.stamp_dependency("attributes");
        }
        if entity_type_changed {
            self.stamp_dependency("entity_types");
        }
        if association_changed {
            self.stamp_dependency("associations");
        }
        // 内容变化意味着链接剖面可能变化，单条记忆整体失效（在变更检测后统一做）
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        // 规则只依赖已积累的三元组与固定阈值：有新增三元组时才重新挖掘
        if triples_added && self.config.enable_link_prediction {
            self.predictor.mine_rules(
                self.config.link_prediction_min_support,
                self.config.link_prediction_min_confidence,
            );
            self.stamp_rebuild("link_rules");
        }
        // 关联规则与前向/反向归结都只依赖已观察的篮子与事实：有新增时才重建
        if self.config.enable_association_rules && (association_changed || triples_added) {
            self.associations.mine();
            self.stamp_rebuild("associations");
        }
        if self.config.enable_resolution
            && (association_changed || attribute_changed || entity_type_changed || triples_added)
        {
            self.rebuild_resolver();
        }
        // 加权软逻辑与归结引擎同源（同一批事实与规则），触发条件保持一致
        if self.config.enable_soft_logic
            && (association_changed || attribute_changed || entity_type_changed || triples_added)
        {
            self.rebuild_soft_logic();
        }
        // 图游走向量采用惰性重建：新边在 observe_triple 内标记失效，
        // 首次需要嵌入时才做全量幂迭代（查询结果与即时重建完全一致）
        if self.config.enable_semantic_embedding {
            if self.embedding.is_dirty() {
                self.perf.bump("embedding_rebuild");
                self.embedding.rebuild();
            } else {
                self.perf.bump("embedding_rebuild_skipped");
            }
        }
        // 谓词清单与命题表面积索引随剖面/规则变化，统一在变更后重建一次
        self.rebuild_knowledge_caches();
    }

    // 用当前知识（事实 + 链接预测规则 + 关联规则）重建归结引擎
    fn rebuild_resolver(&mut self) {
        let mut engine = ResolutionEngine::new(
            self.config.resolution_max_depth,
            self.config.resolution_max_solutions,
        );
        for triple in self.predictor.triples() {
            engine.add_fact(&triple.predicate, &triple.subject, &triple.object);
            engine.add_fact(PROOF_HAS, &triple.subject, &triple.predicate);
        }
        for (subject, key, value) in &self.attributes {
            engine.add_fact(key, subject, value);
            engine.add_fact(PROOF_HAS, subject, key);
        }
        for (entity, etype) in self.generator.typed_entities() {
            engine.add_fact(PROOF_TYPE, &entity, &etype);
        }
        if self.config.enable_association_rules {
            for clause in self.associations.to_clauses(PROOF_HAS, PROOF_TYPE) {
                engine.add_clause(clause);
            }
        }
        for rule in self.predictor.rules() {
            engine.add_clause(clause_from_path_rule(rule));
        }
        // 会话级假设事实必须在重建后重新注入（否则刷新知识图谱会丢失"采纳的假设"）
        for atom in &self.session_facts {
            engine.add_fact_atom(atom.clone());
        }
        // 会话级 Horn 规则同样在重建后重新注入（教过的一般规则不能因刷新而丢失）
        for clause in &self.session_clauses {
            engine.add_clause(clause.clone());
        }
        self.resolver = engine;
        self.stamp_rebuild("resolver");
    }

    /// 用当前知识重建加权软逻辑引擎：事实同上，规则权重 = 置信度的 log-odds
    fn rebuild_soft_logic(&mut self) {
        let mut engine = MarkovLogicEngine::new(
            self.config.soft_logic_max_iterations,
            self.config.soft_logic_min_probability,
        );
        for triple in self.predictor.triples() {
            engine.add_fact(Atom::rel(
                &triple.predicate,
                &triple.subject,
                &triple.object,
            ));
            engine.add_fact(Atom::rel(PROOF_HAS, &triple.subject, &triple.predicate));
        }
        for (subject, key, value) in &self.attributes {
            engine.add_fact(Atom::rel(key, subject, value));
            engine.add_fact(Atom::rel(PROOF_HAS, subject, key));
        }
        for (entity, etype) in self.generator.typed_entities() {
            engine.add_fact(Atom::rel(PROOF_TYPE, &entity, &etype));
        }
        for atom in &self.session_facts {
            engine.add_fact(atom.clone());
        }
        if self.config.enable_association_rules {
            for clause in self.associations.to_clauses(PROOF_HAS, PROOF_TYPE) {
                engine.add_rule(WeightedRule::new(
                    clause.head,
                    clause.body,
                    MarkovLogicEngine::weight_from_confidence(clause.confidence),
                    &clause.source,
                ));
            }
        }
        for rule in self.predictor.rules() {
            let clause = clause_from_path_rule(rule);
            engine.add_rule(WeightedRule::new(
                clause.head,
                clause.body,
                MarkovLogicEngine::weight_from_confidence(clause.confidence),
                &clause.source,
            ));
        }
        self.soft = engine;
        self.stamp_rebuild("soft_logic");
        // 概率逻辑程序与软逻辑同源：软逻辑重建后同步转换（含重放教过的中文知识）
        if self.config.enable_problog {
            self.rebuild_problog();
        }
    }

    /// 重建 ProbLog 风格概率程序：MLN-lite 规则按 σ(w) 转成概率规则，
    /// 再重放用户用中文教过的概率知识（会话级，不修改知识图谱）
    fn rebuild_problog(&mut self) {
        let mut program = ProblogEngine::from_markov_engine(
            &self.soft,
            self.config.problog_max_depth,
            self.config.problog_max_proofs,
        );
        // 重放缓存好的解析结果
        for lesson in &self.lessons_parsed {
            chinese_reasoning_nlu::apply_lesson(&mut program, lesson);
        }
        self.problog = program;
        self.stamp_rebuild("problog");
    }

    pub fn resolver(&self) -> &ResolutionEngine {
        &self.resolver
    }

    pub fn associations(&self) -> &AssociationMiner {
        &self.associations
    }

    pub fn contextualize(&mut self, input: &str) -> Option<String> {
        self.contextualize_with_receipt(input)
            .map(|rewrite| rewrite.text)
    }

    /// 与 `contextualize` 相同，但附带来源收据（哪个洞由哪一轮的哪个槽补上）。
    ///
    /// 处理顺序：代词消解 → 话语帧合一（前置尝试，命中才接管）→ 省略规则
    /// → 代词改写。帧通道不合法的合一逐字回退。
    pub fn contextualize_with_receipt(&mut self, input: &str) -> Option<ContextRewrite> {
        self.frames.begin_turn();
        let pronoun_resolved = self.context.resolve_pronouns(input);
        let base = pronoun_resolved.as_deref().unwrap_or(input);
        let rewrite = self
            .try_frame_rewrite(base)
            .or_else(|| {
                self.resolve_ellipsis(base)
                    .map(|text| ContextRewrite::new(text))
            })
            .or_else(|| pronoun_resolved.map(ContextRewrite::new));
        self.record_vocabulary(input);
        if let Some(rewrite) = &rewrite {
            self.record_vocabulary(&rewrite.text);
        }
        rewrite
    }

    /// 话语帧通道：复合问句拆帧（第一问照答、其余挂账）、省略追问合一补全、
    /// 完整 AskAttr 问句登记为上一帧。命中返回规范问句与来源收据。
    fn try_frame_rewrite(&mut self, input: &str) -> Option<ContextRewrite> {
        let turn = self.frames.turn();
        if let Some(compound) = dialogue_frame::parse_compound(input, &FrameOntology(self), turn) {
            let frames = compound.frames;
            let first_question = frames.first().and_then(|frame| frame.question())?;
            let mut notes = Vec::new();
            if frames.len() > 1 {
                notes.push(format!(
                    "话语帧：先答「{}」，其余挂账等待追问：{}",
                    first_question,
                    frames[1..]
                        .iter()
                        .map(|frame| frame.display())
                        .collect::<Vec<_>>()
                        .join("；")
                ));
            }
            self.frames.record_compound(frames);
            return Some(ContextRewrite {
                text: first_question,
                notes,
            });
        }

        let fragment =
            dialogue_frame::parse_followup(input, &FrameOntology(self), turn).or_else(|| {
                dialogue_frame::parse_bare_attribute_question(input, &FrameOntology(self), turn)
            });
        if let Some(fragment) = fragment {
            let plan = self.frames.plan_followup(&fragment)?;
            if !self.frame_slots_known(&plan.frame) {
                return None;
            }
            let resolution = self.frames.commit_followup(plan);
            return Some(ContextRewrite {
                text: resolution.question,
                notes: vec![resolution.note],
            });
        }

        if let Some(frame) = self.parse_full_ask_frame(input) {
            self.frames.record_last(frame);
        }
        None
    }

    /// 完整 AskAttr 问句（单跳、实体与谓词均在本体登记）→ 登记为上一帧
    fn parse_full_ask_frame(&self, input: &str) -> Option<AskAttributeFrame> {
        let plan = self.planner.parse(input)?;
        let QueryPlan::FindObject { segments } = plan else {
            return None;
        };
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.len() != 1 {
            return None;
        }
        if !self.is_known_entity(&entity) {
            return None;
        }
        let attribute = self.canonical_attribute(&predicates[0].surface)?;
        Some(AskAttributeFrame::new(
            input.trim(),
            self.frames.turn(),
            Some(entity),
            Some(attribute),
        ))
    }

    /// 合一产物的排序复核：实体/属性都必须是已登记项
    fn frame_slots_known(&self, frame: &AskAttributeFrame) -> bool {
        let entity_known = frame
            .entity
            .as_deref()
            .map(|entity| self.is_known_entity(entity))
            .unwrap_or(false);
        let attribute_known = frame
            .attribute
            .as_deref()
            .and_then(|attribute| self.canonical_attribute(attribute))
            .is_some();
        entity_known && attribute_known
    }

    /// 归一化属性词面：保留逆向词面的表述方向（「徒弟」问句 ↔「师傅」事实）
    fn canonical_attribute(&self, raw: &str) -> Option<String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        if self.inverse_query_map.contains_key(raw) {
            return Some(raw.to_string());
        }
        let canonical = self.aligner.canonical_form(raw);
        if self.inverse_query_map.contains_key(canonical.as_str()) {
            return Some(canonical);
        }
        let known = matches!(
            self.aligner.kind_of(&canonical),
            Some(TermKind::Relation) | Some(TermKind::Attribute)
        ) || self.predictor.has_predicate(&canonical);
        if known {
            Some(canonical)
        } else {
            None
        }
    }

    fn resolve_ellipsis(&self, input: &str) -> Option<String> {
        let trimmed = input.trim().trim_end_matches(['？', '?']).trim();
        let core = if let Some(rest) = trimmed.strip_prefix("那") {
            let rest = rest.strip_prefix("么").unwrap_or(rest);
            rest.strip_suffix("呢")?
        } else if trimmed.ends_with('呢') && trimmed.chars().count() <= 12 {
            trimmed.strip_suffix("呢")?
        } else {
            return None;
        };
        let core = core.trim();
        if core.is_empty() || core.contains(' ') {
            return None;
        }
        if core.contains('的') {
            return Some(format!("{}是什么", core));
        }
        let predicate = self.context.last_predicate()?;
        Some(format!("{}的{}是什么", core, predicate))
    }

    fn record_vocabulary(&mut self, text: &str) {
        let mut entities = self.entities_mentioned(text);
        if self.config.enable_generation {
            for entity in self.linked_entities(text) {
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
        }
        if !entities.is_empty() {
            // 焦点实体带类型证据入栈：代词消解据此做类型闸门（未知则不设限）
            let typed: Vec<(String, Option<String>)> = entities
                .iter()
                .map(|entity| {
                    (
                        entity.clone(),
                        self.linker.entity_type(entity).map(str::to_string),
                    )
                })
                .collect();
            self.context.record_entities_typed(&typed);
            if self.config.enable_holographic_memory {
                for entity in &entities {
                    self.holo.push_focus(entity);
                }
            }
        }
        let normalized_text = self.aligner.normalize_cow(text);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(text);
        let predicate =
            self.first_mentioned_known_predicate(normalized_text.as_ref(), &mentioned_canonicals);
        if let Some(predicate) = predicate {
            self.context.record_predicate(&predicate);
        }
        self.context.record_question(text);
    }

    fn collect_known_predicates(&self) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .aligner
            .profiles()
            .iter()
            .filter(|p| p.kind == TermKind::Relation)
            .map(|p| p.term.clone())
            .collect();
        let mut seen: HashSet<String> = predicates.iter().cloned().collect();
        for rule in self.predictor.rules() {
            if seen.insert(rule.head.clone()) {
                predicates.push(rule.head.clone());
            }
        }
        predicates
    }

    pub fn entities_mentioned(&self, text: &str) -> Vec<String> {
        // 同一文本会在一轮对话中被多次询问，做多槽记忆复用；
        // 实体剖面或同义词表只在 observe_facts 中变化，届时（revision 推进）整体失效
        if let Some(found) = self
            .entity_mentions_memo
            .lock()
            .get(self.revision, text)
            .cloned()
        {
            self.perf.bump("memo_entity_mentions_hit");
            return found;
        }
        self.perf.bump("memo_entity_mentions_miss");
        let found = self.entities_mentioned_uncached(text);
        self.entity_mentions_memo
            .lock()
            .put(self.revision, text.to_string(), found.clone());
        found
    }

    fn entities_mentioned_uncached(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize_cow(text);
        let mut found: Vec<(usize, String)> = Vec::new();
        for (i, profile) in self.aligner.profiles().iter().enumerate() {
            if profile.kind != TermKind::Entity {
                continue;
            }
            let Some(term_norm) = self.aligner.normalized_term_at(i) else {
                continue;
            };
            if term_norm.is_empty() {
                continue;
            }
            if normalized_text.contains(term_norm) {
                found.push((
                    profile.term.chars().count(),
                    self.aligner.canonical_form(&profile.term),
                ));
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if canonical.is_empty() {
                continue;
            }
            if text.contains(alias.as_str())
                && self.aligner.kind_of(canonical) == Some(TermKind::Entity)
            {
                found.push((alias.chars().count(), canonical.clone()));
            }
        }
        found.sort_by(|a, b| b.0.cmp(&a.0));
        let mut seen = HashSet::new();
        found
            .into_iter()
            .map(|(_, entity)| entity)
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    /// 收集文本中出现的同义词对应的规范谓词集合
    fn mentioned_predicate_canonicals(&self, text: &str) -> Arc<HashSet<String>> {
        if let Some(canonicals) = self
            .predicate_canonicals_memo
            .lock()
            .get(self.revision, text)
            .cloned()
        {
            self.perf.bump("memo_predicate_canonicals_hit");
            return canonicals;
        }
        self.perf.bump("memo_predicate_canonicals_miss");
        let mut mentioned: HashSet<String> = HashSet::new();
        for (alias, canonical) in self.aligner.synonyms() {
            if text.contains(alias.as_str()) {
                mentioned.insert(canonical.clone());
            }
        }
        let mentioned = Arc::new(mentioned);
        self.predicate_canonicals_memo.lock().put(
            self.revision,
            text.to_string(),
            mentioned.clone(),
        );
        mentioned
    }

    /// 按缓存顺序返回首个被文本提及的已知谓词（原形）
    fn first_mentioned_known_predicate(
        &self,
        normalized_text: &str,
        mentioned_canonicals: &HashSet<String>,
    ) -> Option<String> {
        self.known_predicates_normalized_cache
            .iter()
            .find_map(|(predicate, predicate_norm)| {
                if mentioned_canonicals.contains(predicate)
                    || (!predicate_norm.is_empty() && normalized_text.contains(predicate_norm))
                {
                    Some(predicate.clone())
                } else {
                    None
                }
            })
    }

    /// 判断谓词是否被文本提及（等价于 predicate_mentioned_normalized）
    fn predicate_mentioned_normalized_with(
        &self,
        normalized_text: &str,
        predicate: &str,
        mentioned_canonicals: &HashSet<String>,
    ) -> bool {
        if mentioned_canonicals.contains(predicate) {
            return true;
        }
        let predicate_norm = self.aligner.normalize_cow(predicate);
        !predicate_norm.is_empty() && normalized_text.contains(predicate_norm.as_ref())
    }

    // 实体链接：候选生成 + 模糊容错 + 上下文消歧（详见 entity_linker 模块）
    pub const LINK_ACCEPT_THRESHOLD: f64 = 0.5;

    /// 接地清单：词面 ↔ 实体 ↔ 类型（与链接路径同源，供回答附引用）。
    pub fn grounding_mentions(&self, text: &str) -> Vec<crate::entity_linker::GroundedMention> {
        let hints = self.link_hints(text);
        self.linker.grounded_mentions(text, &hints)
    }

    /// 显式准入一个 OOV 词面（提名 → 用户确认 → 词表增长）。
    ///
    /// 返回（词面，类型）；只改会话内词表（链接缓存即时失效）；
    /// 持久化由调用方写入知识库后重新观察事实完成。
    pub fn admit_vocabulary(&mut self, surface: &str) -> Option<(String, String)> {
        let admitted = self.linker.admit_entity(surface)?;
        self.link_memo.lock().clear();
        Some(admitted)
    }

    /// 提及格全局解码报告（只读审计面：不改变实际链接路径）。
    pub fn mention_lattice_report(&self, text: &str) -> String {
        let hints = self.link_hints(text);
        self.linker.lattice_report(text, &hints)
    }

    /// 身份台账：规范身份 ↔ 别名 ↔ 歧义（URI 身份的只读审计视图，确定排序）。
    pub fn identity_report(&self) -> String {
        self.linker.identity_registry().audit_lines().join("\n")
    }

    /// 文本的实体链接结果：一次查询流程内同一文本会被多处链接，做单条记忆复用
    fn linked_mentions(&self, text: &str) -> Arc<Vec<LinkedMention>> {
        if let Some(mentions) = self.link_memo.lock().get(self.revision, text).cloned() {
            self.perf.bump("memo_link_hit");
            return mentions;
        }
        self.perf.bump("memo_link_miss");
        let hints = self.link_hints(text);
        let mentions = Arc::new(self.linker.link_with_context(text, &hints));
        self.link_memo
            .lock()
            .put(self.revision, text.to_string(), mentions.clone());
        mentions
    }

    pub fn linked_entities(&self, text: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| mention.best())
            .filter(|candidate| candidate.score >= Self::LINK_ACCEPT_THRESHOLD)
            .map(|candidate| candidate.entity.clone())
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    pub fn best_linked_entity(&self, text: &str) -> Option<String> {
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| {
                mention
                    .best()
                    .map(|candidate| (candidate.score, candidate.entity.clone()))
            })
            .filter(|(score, _)| *score >= Self::LINK_ACCEPT_THRESHOLD)
            .max_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.1.cmp(&a.1))
            })
            .map(|(_, entity)| entity)
    }

    pub fn entity_link_report(&self, text: &str) -> String {
        let mentions = self.linked_mentions(text);
        self.linker.report_for_mentions(text, mentions.as_slice())
    }

    // 仅返回"模糊匹配"的链接说明，用于在回答前向用户提示容错链接
    pub fn entity_link_notes(&self, text: &str) -> Vec<String> {
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| {
                let candidate = mention.best()?;
                if candidate.method != LinkMethod::Fuzzy {
                    return None;
                }
                Some(format!(
                    "实体链接：「{}」→「{}」（置信度 {:.2}）",
                    mention.mention, candidate.entity, candidate.score
                ))
            })
            .collect()
    }

    /// 提及账本：每个指称一张证据收据（边界 + 类型证据 × 疑问词期待 + 焦点先验）。
    ///
    /// 只读审计面：不改变链接与作答行为，输出确定可 golden。
    pub fn mention_ledger_report(&self, text: &str) -> String {
        let hints = self.link_hints(text);
        let expectations = crate::function_words::answer_type_keywords(text);
        let receipts =
            self.linker
                .mention_receipts(text, &hints, &expectations, self.context.focus());
        let suspects = crate::entity_linker::oov_suspects(text, &receipts);
        crate::entity_linker::render_mention_ledger(text, &receipts, &expectations, &suspects)
    }

    fn link_hints(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize_cow(text);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(text);
        self.known_predicates_normalized_cache
            .iter()
            .filter(|(predicate, predicate_norm)| {
                mentioned_canonicals.contains(predicate)
                    || (!predicate_norm.is_empty() && normalized_text.contains(predicate_norm))
            })
            .map(|(predicate, _)| predicate.clone())
            .collect()
    }

    // 生成式介绍：把实体的相关事实组合成连贯段落（无神经网络）
    // 生成后自动做语义蕴含自检，并记录"上一段生成"以便多轮续说。
    pub fn generate_about(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let page = self
            .generator
            .profile_page(&entity, 0, GenerationOrder::Relevance)?;
        self.context
            .record_generation(&entity, page.next_offset, GenerationOrder::Relevance);
        let mut text = page.text;
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    pub fn discourse(&self) -> &DiscoursePlanner {
        &self.discourse
    }

    // 符号主义叙述：OWL 本体定义开场 + 层级/继承补充 + DCG 文法实现的多句连贯段落
    pub fn narrate(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_generation || !self.config.enable_discourse {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let focus = self.context.focus().map(|focus| focus.to_string());
        let text = self.discourse.narrate(
            &entity,
            focus.as_deref(),
            None,
            self.config.discourse_max_sentences,
            self.generator.predicate_surfaces(),
        )?;
        self.context.record_entities(&[entity.clone()]);
        self.context.record_question(&format!("叙述：{}", entity));
        Some(text)
    }

    // 围绕话题的叙述：命中话题的谓词优先（回答"关于X的P"式追问）
    pub fn narrate_topic(&mut self, entity_raw: &str, topic: &str) -> Option<String> {
        if !self.config.enable_generation || !self.config.enable_discourse {
            return None;
        }
        let trimmed = entity_raw.trim();
        let topic = topic.trim();
        if trimmed.is_empty() || topic.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let focus = self.context.focus().map(|focus| focus.to_string());
        self.discourse.narrate(
            &entity,
            focus.as_deref(),
            Some(topic),
            self.config.discourse_max_sentences,
            self.generator.predicate_surfaces(),
        )
    }

    // DCG 文法复述：同一事实的多种合规句式（纯符号改写，不改变语义）
    pub fn paraphrase_fact(
        &mut self,
        entity_raw: &str,
        predicate: &str,
        limit: usize,
    ) -> Option<String> {
        if !self.config.enable_discourse {
            return None;
        }
        let canonical = self.aligner.canonical_form(entity_raw.trim());
        let entity = if canonical.is_empty() {
            entity_raw.trim().to_string()
        } else {
            canonical
        };
        let variants = self.discourse.paraphrase(
            &entity,
            predicate.trim(),
            limit.max(2),
            self.generator.predicate_surfaces(),
        );
        if variants.is_empty() {
            return None;
        }
        let mut text = String::new();
        for (index, variant) in variants.iter().enumerate() {
            if index > 0 {
                text.push('\n');
            }
            text.push_str(&format!("{}. {}", index + 1, variant));
        }
        Some(text)
    }

    // OWL 本体教学：把中文公理加入本体（子类/互斥/定义域/固有属性……）
    pub fn teach_ontology(&mut self, axiom: &str) -> Result<String, String> {
        if !self.config.enable_discourse {
            return Err("话语引擎未启用（enable_discourse = false）".to_string());
        }
        let trimmed = axiom.trim();
        if trimmed.is_empty() {
            return Err("请提供本体公理，例如：人是一种动物".to_string());
        }
        let summary = self.discourse.teach_axiom(trimmed)?;
        self.sync_retrieval_type_hierarchy();
        Ok(summary)
    }

    /// 把单一本体（教学态 + 配置态）的子类/互斥公理同步进稠密检索的类型层级：
    /// 类型约束按 RDFS 展开（互斥是唯一硬剪证据），与本体推理器同源同簿。
    /// 观察是幂等的（集合插入），重复同步不产生重复边。
    fn sync_retrieval_type_hierarchy(&mut self) {
        let lessons: Vec<OntologyLesson> = self.discourse.taught_lessons().to_vec();
        let mut retrieval = self.retrieval.lock();
        for lesson in &lessons {
            match lesson {
                OntologyLesson::Subclass { sub, sup, .. } => retrieval.observe_subclass(sub, sup),
                OntologyLesson::Disjoint { left, right, .. } => {
                    retrieval.observe_disjoint(left, right)
                }
                _ => {}
            }
        }
    }

    /// 配置态本体并入**单一本体**：`[type_system]` 与 `[[predicate_profiles]]`
    /// 归约出的公理与教学公理共用同一个本体推理器与同一个胶囊，
    /// 落盘/重建/分叉都不丢（配置不再是第二套本体）。
    /// 公理集合是单调的：已存在的条目按整句（含原句）去重，重复调用幂等。
    /// 返回本次新登记的公理条数。
    pub fn configure_ontology_program(&mut self, program: &OntologyProgram) -> usize {
        let before = self.discourse.taught_lessons().len();
        for lesson in &program.lessons {
            self.discourse.install_lesson(lesson);
        }
        let added = self.discourse.taught_lessons().len().saturating_sub(before);
        if added > 0 {
            self.perf.bump_by("ontology_config_lessons", added as u64);
            self.sync_retrieval_type_hierarchy();
        }
        added
    }

    /// 已登记的本体公理（配置态 + 教学态同簿；供审计与断言）。
    pub fn taught_ontology_lessons(&self) -> &[OntologyLesson] {
        self.discourse.taught_lessons()
    }

    // 多轮生成：接着上一段介绍继续讲后面的句子
    pub fn continue_generation(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let page = self
            .generator
            .profile_page(&state.entity, state.shown, state.order)?;
        if page.shown == 0 {
            return Some(format!(
                "关于「{}」的知识已经介绍完毕。可以输入「换个角度说」按不同顺序重述，或介绍其它实体。",
                state.entity
            ));
        }
        self.context
            .record_generation(&state.entity, page.next_offset, state.order);
        let mut text = page.text;
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    // 多轮生成：换一个排序角度重述同一实体的介绍
    pub fn switch_generation_angle(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let order = match state.order {
            GenerationOrder::Relevance => GenerationOrder::Alphabetical,
            GenerationOrder::Alphabetical => GenerationOrder::Relevance,
        };
        let page = self.generator.profile_page(&state.entity, 0, order)?;
        if page.shown == 0 {
            return Some(format!("关于「{}」暂无可重述的事实。", state.entity));
        }
        self.context
            .record_generation(&state.entity, page.next_offset, order);
        let label = match order {
            GenerationOrder::Relevance => "信息量排序（由稀有谓词到常见谓词）",
            GenerationOrder::Alphabetical => "主题排序（按谓词字典序）",
        };
        let mut text = format!(
            "换个角度介绍「{}」（{}）：\n{}",
            state.entity, label, page.text
        );
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    fn knowledge(&self) -> Knowledge<'_> {
        Knowledge {
            predictor: &self.predictor,
            generator: &self.generator,
            min_rule_confidence: self.config.entailment_min_confidence,
            max_rule_candidates: 16,
        }
    }

    fn proposition_parser(&self) -> PropositionParser<'_> {
        PropositionParser::with_surfaces(&self.aligner, &self.linker, &self.proposition_surfaces)
    }

    // 形式化语义 + 语义蕴含：把陈述句解析为命题并逐条判定蕴含/矛盾/未知
    pub fn check_statement(&self, text: &str) -> Vec<EntailmentResult> {
        if !self.config.enable_entailment {
            return Vec::new();
        }
        let parser = self.proposition_parser();
        let knowledge = self.knowledge();
        parser
            .parse(text)
            .into_iter()
            .map(|proposition| semantic_entailment::entail(&proposition, &knowledge))
            .collect()
    }

    /// 语义蕴含判未知/形式化失败时的后向链补充判定：把陈述转成归结目标，
    /// 用「图谱事实 + 会话规则」做 SLD 反驳，得证则输出可追溯的证明树
    fn backward_chain_appendix(&self, text: &str) -> Option<String> {
        if !self.config.enable_resolution {
            return None;
        }
        let atom = self
            .statement_atom(text)
            .or_else(|| chinese_reasoning_nlu::parse_proposition(text))
            .filter(Atom::is_ground)?;
        let solutions = self.resolver.solve(std::slice::from_ref(&atom), &[]);
        let solution = solutions.first()?;
        let mut lines = Vec::new();
        for proof in &solution.proofs {
            self.render_proof_tree(proof, &mut lines, 2);
        }
        Some(format!(
            "【后向链补充判定】语义蕴含未能直接判定，但归结引擎证明了「{}」（证明置信度 {:.2}）：\n  后向链证明树：\n{}",
            self.render_atom_nl(&atom),
            solution.confidence,
            lines.join("\n")
        ))
    }

    pub fn entailment_report(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return "请输入要判定的陈述句，例如：判断：张三是李四的师傅".to_string();
        }
        let results = self.check_statement(trimmed);
        if results.is_empty() {
            if let Some(appendix) = self.backward_chain_appendix(trimmed) {
                return appendix;
            }
            return format!(
                "形式化失败：未能把「{}」解析为可判定命题。\n  可用句式：S的P是O / O是S的P / S是类型 / S P O（P 为图谱已知谓词）。",
                trimmed
            );
        }
        let mut output = format!(
            "形式化语义分析：「{}」解析出 {} 个候选命题，逐条给出蕴含判定（证据可追溯）：\n",
            trimmed,
            results.len()
        );
        output.push_str(&semantic_entailment::render_results(&results));
        if results
            .iter()
            .all(|result| result.verdict == Verdict::Unknown)
        {
            if let Some(appendix) = self.backward_chain_appendix(trimmed) {
                output.push('\n');
                output.push_str(&appendix);
            }
        }
        output
    }

    // ------------------------- 溯因推理·假设生成 -------------------------

    fn proposition_atom(&self, proposition: &semantic_entailment::Proposition) -> Atom {
        match proposition.kind {
            semantic_entailment::PropositionKind::TypeMembership => {
                Atom::rel(PROOF_TYPE, &proposition.subject, &proposition.object)
            }
            _ => Atom::rel(
                &proposition.predicate,
                &proposition.subject,
                &proposition.object,
            ),
        }
    }

    /// 把陈述句形式化为归结原子（否定句返回 None）
    fn statement_atom(&self, text: &str) -> Option<Atom> {
        let parser = self.proposition_parser();
        let propositions = parser.parse(text);
        let proposition = propositions.iter().find(|item| !item.negative)?;
        Some(self.proposition_atom(proposition))
    }

    /// 把（可能含多个条件的）文本切分为归结原子列表
    fn condition_atoms(&self, text: &str) -> Vec<Atom> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut candidates: Vec<String> = text
            .split(['；', ';', '，', ','])
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect();
        if candidates.is_empty() {
            candidates.push(text.trim().to_string());
        }
        let parser = self.proposition_parser();
        for part in &candidates {
            for proposition in parser.parse(part) {
                if proposition.negative {
                    continue;
                }
                let atom = self.proposition_atom(&proposition);
                if !atoms.contains(&atom) {
                    atoms.push(atom);
                }
            }
        }
        if atoms.is_empty() {
            if let Some(atom) = self.statement_atom(text) {
                atoms.push(atom);
            }
        }
        atoms
    }

    /// 把问句/陈述解析为溯因目标（开放问句 → 谓词链目标；陈述句 → 单原子目标）
    fn abduction_target(&self, trimmed: &str) -> Option<(Vec<Atom>, Vec<String>, String)> {
        if let Some(plan) = self.planner.parse(trimmed) {
            match plan {
                QueryPlan::FindObject { segments } => {
                    if let Some((entity, predicates)) = self.split_chain(&segments) {
                        if !predicates.is_empty()
                            && self.is_known_entity(&entity)
                            && predicates
                                .iter()
                                .all(|p| self.is_known_predicate(&p.surface))
                        {
                            let answer_var = "答案".to_string();
                            let mut goals = Vec::new();
                            let mut previous = Term::constant(&entity);
                            for (index, predicate) in predicates.iter().enumerate() {
                                let next = if index + 1 == predicates.len() {
                                    Term::var(&answer_var)
                                } else {
                                    Term::var(&format!("V{}", index + 1))
                                };
                                // 逆向环节以槽位反方向成立：slot(next, previous)
                                let atom = if predicate.inverted {
                                    Atom::pattern(&predicate.slot, next.clone(), previous.clone())
                                } else {
                                    Atom::pattern(&predicate.slot, previous.clone(), next.clone())
                                };
                                goals.push(atom);
                                previous = next;
                            }
                            let phrase = format!(
                                "{}的{}是谁",
                                entity,
                                predicates
                                    .iter()
                                    .map(|p| p.surface.as_str())
                                    .collect::<Vec<_>>()
                                    .join("的")
                            );
                            return Some((goals, vec![answer_var], phrase));
                        }
                    }
                }
                QueryPlan::Exist { subject, predicate } => {
                    let entity = self.aligner.canonical_form(subject.trim());
                    let (slot, inverted) = self.resolve_question_predicate(&predicate);
                    if !entity.is_empty() && !slot.is_empty() {
                        let atom = if inverted {
                            Atom::pattern(&slot, Term::var("答案"), Term::constant(&entity))
                        } else {
                            Atom::pattern(&slot, Term::constant(&entity), Term::var("答案"))
                        };
                        return Some((
                            vec![atom],
                            vec!["答案".to_string()],
                            format!("{}的{}是否存在", entity, slot),
                        ));
                    }
                }
                _ => {}
            }
        }
        let atom = self.statement_atom(trimmed)?;
        let phrase = self.render_atom_nl(&atom);
        Some((vec![atom], Vec::new(), phrase))
    }

    /// 统一渲染溯因报告：非空假设（含联合场景成员）存为待定，供「采纳假设 N / 澄清」
    fn finish_abduction_report(
        &mut self,
        report: AbductionReport,
        phrase: &str,
        goals: &[Atom],
        query_vars: &[String],
    ) -> String {
        if report.provable {
            let render = |atom: &Atom| self.render_atom_nl(atom);
            let mut output = abduction::render_report(&report, &render);
            if !phrase.is_empty() {
                output = format!("（问题：{}）\n{}", phrase, output);
            }
            return output;
        }
        if !goals.is_empty() {
            self.last_abduction_goal = Some(goals.to_vec());
            self.last_abduction_vars = query_vars.to_vec();
        }
        // 联合解释场景（多假设协同 + 注入验证）：成员也并入待定池
        //（`report.provable == false` 已确认目标不可证）
        let scenarios =
            self.abduction
                .scenarios_assuming_unprovable(&self.resolver, goals, query_vars);
        // 循环假设过滤："假设目标本身成立"不算解释，不进入待定池/澄清队列
        let mut hypotheses: Vec<abduction::Hypothesis> = report
            .hypotheses
            .iter()
            .filter(|hypothesis| !goals.iter().any(|goal| goal == &hypothesis.atom))
            .cloned()
            .collect();
        for scenario in &scenarios {
            for hypothesis in &scenario.hypotheses {
                if !hypotheses
                    .iter()
                    .any(|existing| existing.atom == hypothesis.atom)
                {
                    hypotheses.push(hypothesis.clone());
                }
            }
        }
        self.abduction.set_pending(hypotheses);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        let mut output = abduction::render_report(&report, &render);
        // 只有当"多条假设协同"确实能验证目标时才展示场景，避免噪音
        let joint: Vec<Scenario> = scenarios
            .into_iter()
            .filter(|scenario| scenario.verified && scenario.size() >= 2)
            .collect();
        if !joint.is_empty() {
            output.push('\n');
            output.push_str(&abduction::render_scenarios(
                &report.goal,
                &report.frontier,
                &joint,
                &render,
            ));
        }
        if !phrase.is_empty() {
            output = format!("（问题：{}）\n{}", phrase, output);
        }
        output
    }

    /// 溯因陈述：能证明就给证明；不能证明就做失败前沿分析并生成候选假设
    pub fn abduce_statement(&mut self, text: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let atom = self.statement_atom(trimmed)?;
        let goals = vec![atom.clone()];
        let report = self.abduction.analyze(&self.resolver, &goals, &[]);
        Some(self.finish_abduction_report(report, "", &goals, &[]))
    }

    /// 溯因问句：把开放问句拆成谓词链目标，失败前沿生成候选答案假设
    pub fn hypothesis_answer(&mut self, question: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = question
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        let (goals, query_vars, phrase) = self.abduction_target(trimmed)?;
        let report = self.abduction.analyze(&self.resolver, &goals, &query_vars);
        let solution = report.solutions.first().cloned();
        let mut output = self.finish_abduction_report(report, &phrase, &goals, &query_vars);
        if let Some(solution) = solution {
            output = if let Some(answer) = solution.bindings.get("答案") {
                format!(
                    "后向链推理给出答案：「{}」（证明置信度 {:.2}）\n{}",
                    answer, solution.confidence, output
                )
            } else {
                // 陈述句目标没有答案变量，展示"已证明"而不是伪造「?」答案
                format!(
                    "后向链推理已证明该陈述（证明置信度 {:.2}）\n{}",
                    solution.confidence, output
                )
            };
        }
        Some(output)
    }

    /// 生成式联合解释：枚举假设组合 + 注入克隆引擎验证 + 输出可证场景
    pub fn scenario_report(&mut self, text: &str) -> String {
        if !self.config.enable_abduction {
            return "溯因推理未启用（config.toml [reasoning] enable_abduction = false）。"
                .to_string();
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请输入要解释的陈述或问句，例如：场景：张三的师祖是赵六".to_string();
        }
        let Some((goals, query_vars, phrase)) = self.abduction_target(trimmed) else {
            return format!(
                "无法把「{}」形式化为可联合解释的目标。支持句式：S的P是O / S的P是谁。",
                trimmed
            );
        };
        let report = self.abduction.analyze(&self.resolver, &goals, &query_vars);
        if report.provable {
            let render = |atom: &Atom| self.render_atom_nl(atom);
            return format!(
                "目标「{}」已可由现有知识证明，无需联合假设：\n{}",
                phrase,
                abduction::render_report(&report, &render)
            );
        }
        self.last_abduction_goal = Some(goals.clone());
        self.last_abduction_vars = query_vars.clone();
        //（`report.provable == false` 已确认目标不可证）
        let scenarios =
            self.abduction
                .scenarios_assuming_unprovable(&self.resolver, &goals, &query_vars);
        // 循环假设过滤："假设目标本身成立"不算解释，不进入待定池/澄清队列
        let mut pending: Vec<Hypothesis> = report
            .hypotheses
            .iter()
            .filter(|hypothesis| !goals.iter().any(|goal| goal == &hypothesis.atom))
            .cloned()
            .collect();
        for scenario in &scenarios {
            for hypothesis in &scenario.hypotheses {
                if !pending
                    .iter()
                    .any(|existing| existing.atom == hypothesis.atom)
                {
                    pending.push(hypothesis.clone());
                }
            }
        }
        self.abduction.set_pending(pending);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        abduction::render_scenarios(&report.goal, &report.frontier, &scenarios, &render)
    }

    /// 把断言事实中的新实体/新谓词登记进词表：后续解析、意图规划与实体链接才能识别它们
    fn register_atom_vocabulary(&mut self, atom: &Atom) -> bool {
        let mut changed = false;
        if !atom.predicate.trim().is_empty() && !self.aligner.is_registered(&atom.predicate) {
            self.aligner
                .register_term(&atom.predicate, TermKind::Relation);
            changed = true;
        }
        for term in &atom.terms {
            if let Term::Const(name) = term {
                if !name.trim().is_empty() && !self.aligner.is_registered(name) {
                    self.aligner.register_term(name, TermKind::Entity);
                    self.linker.register_entity(name, None);
                    changed = true;
                }
            }
        }
        if changed {
            self.rebuild_knowledge_caches();
        }
        changed
    }

    /// 判断基原子是否已由知识图谱独立支持（三元组、属性或类型）。
    /// 会话事实可能与图谱事实同名；撤回会话副本时不能把图谱那一份一并删掉。
    fn graph_supports_atom(&self, atom: &Atom) -> bool {
        let [subject, object] = atom.terms.as_slice() else {
            return false;
        };
        let (Term::Const(subject), Term::Const(object)) = (subject, object) else {
            return false;
        };
        let subject: &str = subject;
        let object: &str = object;
        let predicate: &str = &atom.predicate;
        if predicate == PROOF_HAS {
            return self
                .predictor
                .triples()
                .iter()
                .any(|triple| triple.subject == subject && triple.predicate == object)
                || self
                    .attributes
                    .iter()
                    .any(|(entity, key, _)| entity == subject && key == object);
        }
        if predicate == PROOF_TYPE {
            return self
                .generator
                .typed_entities()
                .iter()
                .any(|(entity, etype)| entity == subject && etype == object);
        }
        self.predictor.triples().iter().any(|triple| {
            triple.subject == subject && triple.predicate == predicate && triple.object == object
        }) || self.attributes.contains(&(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ))
    }

    /// 二元常原子 → 证据三元组；其他元数或含变量的原子不进入证据索引。
    fn session_atom_triple(atom: &Atom) -> Option<(&str, &str, &str)> {
        match atom.terms.as_slice() {
            [Term::Const(subject), Term::Const(object)] => {
                Some((subject.as_ref(), atom.predicate.as_ref(), object.as_ref()))
            }
            _ => None,
        }
    }

    fn index_session_atom(&mut self, atom: &Atom) {
        if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
            self.retrieval
                .lock()
                .observe_triple(subject, predicate, object);
        }
    }

    fn unindex_session_atom(&mut self, atom: &Atom) {
        if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
            self.retrieval
                .lock()
                .remove_triple(subject, predicate, object);
        }
    }

    /// 把一条基事实从会话事实集中撤回（不触碰知识图谱）
    fn retract_atom(&mut self, atom: &Atom) -> bool {
        if !self.session_facts.iter().any(|existing| existing == atom) {
            return false;
        }
        self.session_facts.retain(|existing| existing != atom);
        // 图谱事实始终保留；仅当图谱不再支持该原子时才从归结引擎与证据索引
        // 增量删除，避免每次撤回都重建整库（同名的图谱事实与会话事实互不误删）。
        if !self.graph_supports_atom(atom) {
            self.unindex_session_atom(atom);
            if self.config.enable_resolution {
                self.resolver.remove_fact_atom(atom);
            }
        }
        // 软逻辑与概率程序同源：撤回后整体重建（概率程序随软逻辑一起重放）
        if self.config.enable_soft_logic {
            self.rebuild_soft_logic();
        }
        true
    }

    /// 把一条基事实加入会话事实集（不写知识图谱；归结引擎重建后依然保留）
    fn adopt_atom(&mut self, atom: Atom) -> bool {
        if !atom.is_ground() {
            return false;
        }
        let mut added = false;
        if self.config.enable_resolution && self.resolver.add_fact_atom(atom.clone()) {
            added = true;
        }
        if !self.session_facts.contains(&atom) {
            self.session_facts.push(atom.clone());
            self.index_session_atom(&atom);
            added = true;
        }
        // 会话假设同步注入软逻辑引擎
        if added && self.config.enable_soft_logic {
            self.soft.add_fact(atom.clone());
        }
        // 概率程序同步注入（ProbLog 的确定事实），保持两套语义事实一致
        if added && self.config.enable_problog {
            self.problog.add_deterministic_fact(atom);
        }
        added
    }

    /// 采纳第 index（1 起）条待定假设：加入会话归结事实集（不写知识图谱）
    pub fn adopt_hypothesis(&mut self, index: usize) -> Option<String> {
        // 先检查可采纳性再取用：不可采纳的假设不能从待定列表中丢失
        let position = index.checked_sub(1)?;
        if !self.abduction.pending().get(position)?.is_adoptable() {
            let hypothesis = &self.abduction.pending()[position];
            let rendered = self.render_atom_nl(&hypothesis.atom);
            return Some(format!(
                "假设 {}「{}」仍含未绑定变量，无法直接采纳；请提供更具体的陈述。",
                index, rendered
            ));
        }
        let hypothesis = self.abduction.take_pending(index)?;
        let atom = hypothesis.atom.clone();
        let rendered = self.render_atom_nl(&atom);
        self.adopt_atom(atom);
        let reasons = hypothesis.reasons.join("；");
        self.abduction.adopt(hypothesis);
        Some(format!(
            "已采纳假设 {}：{}。该事实已加入本轮会话的归结事实集（未写入知识图谱），后续「证明/推理链/溯因」会使用它。\n  依据：{}",
            index, rendered, reasons
        ))
    }

    pub fn pending_hypotheses_report(&self) -> String {
        let render = |atom: &Atom| self.render_atom_nl(atom);
        abduction::render_pending(self.abduction.pending(), &render)
    }

    /// 供外部查询当前待定假设
    pub fn pending_hypotheses(&self) -> &[Hypothesis] {
        self.abduction.pending()
    }

    /// 会话级假设事实（被采纳、跨会话持久化的临时信念）
    pub fn session_facts(&self) -> &[Atom] {
        &self.session_facts
    }

    // ------------------------- 对话式澄清（以问代猜） -------------------------

    /// 开始澄清对话：把待定假设逐条逆转为「是否成立」的提问
    pub fn start_clarification(&mut self) -> Option<String> {
        if !self.config.enable_abduction || !self.config.enable_clarification {
            return None;
        }
        let session = ClarificationSession::start(self.abduction.pending());
        if session.is_finished() {
            return None;
        }
        self.clarification = Some(session);
        Some(self.render_clarification_prompt())
    }

    fn render_clarification_prompt(&self) -> String {
        let render = |atom: &Atom| self.render_atom_nl(atom);
        match &self.clarification {
            Some(session) => abduction::render_clarification_question(session, &render),
            None => "【对话式澄清】没有进行中的澄清会话。".to_string(),
        }
    }

    pub fn clarification_active(&self) -> bool {
        self.clarification
            .as_ref()
            .map(|session| !session.is_finished())
            .unwrap_or(false)
    }

    /// 处理用户对当前澄清问题的回答（是 / 不是 / 不知道）
    pub fn answer_clarification(&mut self, text: &str) -> Option<String> {
        let reply = {
            let session = self.clarification.as_mut()?;
            session.respond(text)
        };
        match reply {
            ClarificationReply::Confirmed(hypothesis) => {
                self.adopt_atom(hypothesis.atom.clone());
                let mut output = format!(
                    "✓ 已确认假设：{}。已加入会话事实集（未写知识图谱）。",
                    self.render_atom_nl(&hypothesis.atom)
                );
                if !hypothesis.reasons.is_empty() {
                    output.push_str(&format!("\n  依据：{}", hypothesis.reasons.join("；")));
                }
                output.push_str(&self.clarification_follow_up());
                Some(output)
            }
            ClarificationReply::Rejected(hypothesis) => Some(format!(
                "✗ 已排除假设：{}。{}",
                self.render_atom_nl(&hypothesis.atom),
                self.clarification_follow_up()
            )),
            ClarificationReply::Deferred(hypothesis) => Some(format!(
                "？暂记为不确定：{}。{}",
                self.render_atom_nl(&hypothesis.atom),
                self.clarification_follow_up()
            )),
            ClarificationReply::Unrecognized => Some(format!(
                "没有理解回答（请用「是 / 不是 / 不知道」）。\n{}",
                self.render_clarification_prompt()
            )),
            ClarificationReply::Finished => Some(self.finish_clarification_session()),
        }
    }

    /// 澄清推进：若原始目标已得证则汇报答案并结束会话；否则继续下一问
    fn clarification_follow_up(&mut self) -> String {
        if let Some((goal, solution)) = self.clarified_goal_solution() {
            let mut text = format!(
                "\n  → 目标「{}」已可证明（置信度 {:.2}）",
                goal, solution.confidence
            );
            if let Some(answer) = solution.bindings.get("答案") {
                if !answer.is_empty() {
                    text.push_str(&format!("：答案是「{}」", answer));
                }
            }
            text.push_str("。\n");
            text.push_str(&self.finish_clarification_session());
            return text;
        }
        if self.clarification_active() {
            format!("\n{}", self.render_clarification_prompt())
        } else {
            format!("\n{}", self.finish_clarification_session())
        }
    }

    /// 原溯因目标合取是否已被会话事实证明（含"答案"绑定）；返回渲染后的目标描述
    fn clarified_goal_solution(&self) -> Option<(String, Solution)> {
        let goals = self.last_abduction_goal.clone()?;
        if goals.is_empty() {
            return None;
        }
        let vars = self.last_abduction_vars.clone();
        let solution = self.resolver.solve(&goals, &vars).into_iter().next()?;
        let description = goals
            .iter()
            .map(|goal| self.render_atom_nl(goal))
            .collect::<Vec<_>>()
            .join(" 且 ");
        Some((description, solution))
    }

    fn finish_clarification_session(&mut self) -> String {
        let Some(session) = self.clarification.take() else {
            return "【对话式澄清】没有进行中的澄清会话。".to_string();
        };
        let render = |atom: &Atom| self.render_atom_nl(atom);
        let mut output = abduction::render_clarification_summary(&session, &render);
        if let Some((goal, solution)) = self.clarified_goal_solution() {
            output.push_str(&format!(
                "\n  → 当前目标「{}」已可证明（置信度 {:.2}）",
                goal, solution.confidence
            ));
            if let Some(answer) = solution.bindings.get("答案") {
                if !answer.is_empty() {
                    output.push_str(&format!("：答案是「{}」", answer));
                }
            }
            output.push('。');
        }
        output
    }

    /// 手动结束澄清会话：输出已确认/否定/不确定清单
    pub fn stop_clarification(&mut self) -> Option<String> {
        self.clarification.as_ref()?;
        Some(self.finish_clarification_session())
    }

    pub fn clarification_progress(&self) -> Option<String> {
        self.clarification
            .as_ref()
            .map(ClarificationSession::progress_text)
    }

    /// 反事实模拟："如果……那么……" 把条件当临时假设，正向链推导后用后自动回滚
    pub fn what_if_report(&mut self, text: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['。', '！', '!', '？', '?', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        // 只取条件部分："如果A那么B" / "如果A，就B" / "如果A则B" → A
        let condition = trimmed
            .strip_prefix("如果")
            .or_else(|| trimmed.strip_prefix("假如"))
            .or_else(|| trimmed.strip_prefix("假设"))
            .unwrap_or(trimmed);
        let condition = ["那么", "则", "，就", ",就"]
            .iter()
            .filter_map(|separator| condition.find(separator))
            .min()
            .map(|index| &condition[..index])
            .unwrap_or(condition)
            .trim()
            .trim_end_matches(['，', ','])
            .trim();
        if condition.is_empty() {
            return None;
        }
        let assumptions = self.condition_atoms(condition);
        if assumptions.is_empty() {
            return None;
        }
        let result =
            self.abduction
                .what_if(&self.resolver, &assumptions, self.resolver.max_depth(), 20);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        Some(abduction::render_what_if(&result, &render))
    }

    // 生成自检：逐句回译为命题并验证是否被知识图谱蕴含（对抗"幻觉"）
    fn generation_self_check(&self, text: &str) -> Option<String> {
        if !self.config.enable_entailment {
            return None;
        }
        let parser = self.proposition_parser();
        let knowledge = self.knowledge();
        let mut total = 0usize;
        let mut entailed = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for raw_line in text.lines() {
            let sentence = semantic_entailment::strip_discourse_markers(raw_line.trim());
            if !sentence.ends_with('。') {
                continue;
            }
            let propositions = parser.parse(sentence);
            if propositions.is_empty() {
                continue;
            }
            total += 1;
            let supported = propositions.iter().any(|proposition| {
                semantic_entailment::entail(proposition, &knowledge).verdict == Verdict::Entailed
            });
            if supported {
                entailed += 1;
            } else {
                failed.push(sentence.to_string());
            }
        }
        if total == 0 {
            return None;
        }
        let mut note = format!("（生成自检：{}/{} 句可由知识图谱蕴含", entailed, total);
        if failed.is_empty() {
            note.push_str("，零幻觉）");
        } else {
            note.push_str(&format!(
                "，未通过 {} 句：{}）",
                failed.len(),
                failed.join("；")
            ));
        }
        Some(note)
    }

    // 符号续写：对「X的」做谓词预测，支持实体级→类型级→全局回退
    pub fn predict_continuation(&self, prefix: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = prefix.trim();
        let normalized = trimmed.strip_suffix('的').map(|root| {
            let root = root.trim();
            let canonical = self.aligner.canonical_form(root);
            if !canonical.is_empty() && canonical != root {
                format!("{}的", canonical)
            } else {
                trimmed.to_string()
            }
        });
        let base = self
            .generator
            .complete(normalized.as_deref().unwrap_or(trimmed))?;
        // 统计回退之外再挂一层"关联规则联想"：由已有特征推荐缺失特征（纯计数）
        if self.config.enable_association_rules {
            if let Some(root) = trimmed.strip_suffix('的') {
                if let Some(extra) = self.association_suggestions(root.trim()) {
                    let mut output = base;
                    output.push('\n');
                    output.push_str(&extra);
                    return Some(output);
                }
            }
        }
        Some(base)
    }

    // ===== 归结推理：证明、推理链与关联规则 =====

    fn resolve_known_entity(&self, raw: &str) -> Option<String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let candidate = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        if self.generator.is_known_entity(&candidate)
            || self.associations.basket(&candidate).is_some()
        {
            return Some(candidate);
        }
        self.best_linked_entity(trimmed).filter(|entity| {
            self.generator.is_known_entity(entity) || self.associations.basket(entity).is_some()
        })
    }

    // 解析 "S的P1的P2...是O"（谓词串可为空 ⇒ 类型断言）
    fn parse_chain_statement(&self, text: &str) -> Option<(String, Vec<ChainPredicate>, String)> {
        let trimmed = text.trim().trim_end_matches(['。', '！', '!']).trim();
        let (subject_part, object_part) = trimmed.rsplit_once('是')?;
        let object = self.aligner.canonical_form(object_part.trim());
        if object.is_empty() {
            return None;
        }
        let segments: Vec<String> = subject_part
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return None;
        }
        if segments.len() == 1 {
            let entity = self.aligner.canonical_form(&segments[0]);
            if entity.is_empty() {
                return None;
            }
            return Some((entity, Vec::new(), object));
        }
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty() {
            return None;
        }
        Some((entity, predicates, object))
    }

    fn render_atom_nl(&self, atom: &Atom) -> String {
        if !atom.is_ground() {
            return atom.display();
        }
        let names: Vec<String> = atom.terms.iter().map(render_term).collect();
        if &*atom.predicate == PROOF_HAS && names.len() == 2 {
            return format!("{} 具有 {}", names[0], names[1]);
        }
        if &*atom.predicate == PROOF_TYPE && names.len() == 2 {
            return format!("{} 属于类型 {}", names[0], names[1]);
        }
        if names.len() == 2 {
            let sentence =
                self.generator
                    .render_fact(&names[0], &atom.predicate, &[names[1].clone()]);
            return sentence.trim_end_matches('。').to_string();
        }
        atom.display()
    }

    fn render_proof_tree(&self, node: &ProofNode, lines: &mut Vec<String>, indent: usize) {
        let padding = "    ".repeat(indent);
        let fact = self.render_atom_nl(&node.atom);
        match &node.source {
            ProofSource::Fact => lines.push(format!("{}{}（事实）", padding, fact)),
            ProofSource::Rule { source } => lines.push(format!(
                "{}{}（由 {}；本步置信度 {:.2}）",
                padding, fact, source, node.confidence
            )),
        }
        for child in &node.children {
            self.render_proof_tree(child, lines, indent + 1);
        }
    }

    /// 结构化证明视图：与报告共用同一套解析/求解路径，供可视化消费。
    /// 不变量：每棵树的节点集 ≡ `Solution.proofs` 展平集合（不增不减）。
    fn build_resolution_view(&self, phrase: String, goals: Vec<Atom>) -> ResolutionProofView {
        // 目标中出现的变量即查询变量（按首现顺序，确定）：绑定表随视图返回
        let mut query_vars: Vec<String> = Vec::new();
        for goal in &goals {
            for term in &goal.terms {
                if term.is_var() {
                    let name = term.name().to_string();
                    if !query_vars.contains(&name) {
                        query_vars.push(name);
                    }
                }
            }
        }
        let outcome = self.resolver.solve_outcome(&goals, &query_vars);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let solutions = outcome
            .solutions
            .iter()
            .map(|solution| {
                let mut bindings: Vec<(String, String)> = solution
                    .bindings
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                bindings.sort();
                ProofSolutionView {
                    bindings,
                    confidence: solution.confidence,
                    proofs: solution
                        .proofs
                        .iter()
                        .map(|proof| {
                            ProofViewNode::from_proof_with(proof, |node| {
                                self.render_atom_nl(&node.atom)
                            })
                        })
                        .collect(),
                }
            })
            .collect();
        ResolutionProofView {
            phrase,
            goal_display,
            facts: self.resolver.fact_count(),
            clauses: self.resolver.clause_count(),
            max_depth: self.resolver.max_depth(),
            search_status: outcome.status,
            search_truncated: outcome.is_truncated(),
            frontier: outcome.frontier,
            stats: outcome.stats,
            solutions,
        }
    }

    /// 陈述句目标（"S有P" / "S的P…是O" / "S是类型"）→ (短语, 目标原子)。
    fn statement_goals(&self, text: &str) -> Option<(String, Vec<Atom>)> {
        if let Some(goals) = self.existence_goals(text) {
            return Some(goals);
        }
        let trimmed = text.trim().trim_end_matches(['。', '！', '!']).trim();
        let (entity, predicates, object) = self.parse_chain_statement(trimmed)?;
        let surfaces: Vec<&str> = predicates.iter().map(|p| p.surface.as_str()).collect();
        let (goals, phrase) = if predicates.is_empty() {
            (
                vec![Atom::rel(PROOF_TYPE, &entity, &object)],
                format!("{} 属于类型 {}", entity, object),
            )
        } else if predicates.len() == 1 {
            let predicate = &predicates[0];
            // 逆向谓词的陈述按槽位反方向编码（slot(object, entity)）
            let atom = if predicate.inverted {
                Atom::rel(&predicate.slot, &object, &entity)
            } else {
                Atom::rel(&predicate.slot, &entity, &object)
            };
            (
                vec![atom],
                format!("{}的{}是{}", entity, predicate.surface, object),
            )
        } else {
            let mut goals = Vec::new();
            let mut previous = Term::constant(&entity);
            for (index, predicate) in predicates.iter().enumerate() {
                let next = if index + 1 == predicates.len() {
                    Term::constant(&object)
                } else {
                    Term::var(&format!("V{}", index + 1))
                };
                let atom = if predicate.inverted {
                    Atom::pattern(&predicate.slot, next.clone(), previous.clone())
                } else {
                    Atom::pattern(&predicate.slot, previous.clone(), next.clone())
                };
                goals.push(atom);
                previous = next;
            }
            (
                goals,
                format!("{}的{}是{}", entity, surfaces.join("的"), object),
            )
        };
        Some((phrase, goals))
    }

    /// "X有P" 形式的存在性目标（由关联规则/前向推导证明）。
    fn existence_goals(&self, text: &str) -> Option<(String, Vec<Atom>)> {
        let (subject_part, object_part) = text.split_once('有')?;
        let subject_part = subject_part.trim();
        let object_part = object_part.trim();
        if subject_part.is_empty() || object_part.is_empty() {
            return None;
        }
        if subject_part.contains('的') || object_part.contains('的') {
            return None;
        }
        let entity = self.resolve_known_entity(subject_part)?;
        let predicate = self.aligner.canonical_form(object_part);
        if predicate.is_empty() {
            return None;
        }
        let goals = vec![Atom::rel(PROOF_HAS, &entity, &predicate)];
        Some((format!("{} 具有 {}", entity, predicate), goals))
    }

    /// 把主谓宾/被字句/把字句计划直接转成归结目标，供「为什么」重建证明。
    fn relation_plan_view(&self, plan: &QueryPlan) -> Option<ResolutionProofView> {
        let answer = || Term::var("答案");
        let (phrase, goals) = match plan {
            QueryPlan::FindObjectByPredicate { subject, predicate } => {
                let entity = self.resolve_known_entity(subject)?;
                let (slot, inverted) = self.resolve_question_predicate(predicate);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::pattern(&slot, answer(), Term::constant(&entity))
                } else {
                    Atom::pattern(&slot, Term::constant(&entity), answer())
                };
                (format!("{} {} 谁", entity, predicate), vec![goal])
            }
            QueryPlan::FindSubject { object, predicate } => {
                let target = self.resolve_known_entity(object)?;
                let (slot, inverted) = self.resolve_question_predicate(predicate);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::pattern(&slot, Term::constant(&target), answer())
                } else {
                    Atom::pattern(&slot, answer(), Term::constant(&target))
                };
                (format!("谁 {} {}", predicate, target), vec![goal])
            }
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => {
                if *negated {
                    // 否定问句的证明需要否定即失败语义；共享的归结视图
                    // 只重建肯定事实，这里保守弃权。
                    return None;
                }
                let subject = self.resolve_known_entity(subject)?;
                let object = self.resolve_known_entity(object)?;
                let (slot, inverted) = self.resolve_question_predicate(predicate);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::rel(&slot, &object, &subject)
                } else {
                    Atom::rel(&slot, &subject, &object)
                };
                (
                    format!("{} {} {}", subject, predicate, object),
                    vec![goal],
                )
            }
            _ => return None,
        };
        Some(self.build_resolution_view(phrase, goals))
    }

    /// 开放问句的谓词链目标构造（"X的P…是谁"），答案变量固定为「答案」。
    fn build_question_goals(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        answer_var: &str,
    ) -> Vec<Atom> {
        let mut goals = Vec::new();
        let mut previous = Term::constant(entity);
        for (index, predicate) in predicates.iter().enumerate() {
            let next = if index + 1 == predicates.len() {
                Term::var(answer_var)
            } else {
                Term::var(&format!("V{}", index + 1))
            };
            let atom = if predicate.inverted {
                Atom::pattern(&predicate.slot, next.clone(), previous.clone())
            } else {
                Atom::pattern(&predicate.slot, previous.clone(), next.clone())
            };
            goals.push(atom);
            previous = next;
        }
        goals
    }

    /// 把三态搜索结论与同树失败前沿附在推理报告后面：不把截断伪装成「不存在」。
    fn render_search_diagnostics(&self, outcome: &SearchOutcome) -> String {
        let mut output = format!("  搜索状态：{}\n", outcome.render_status());
        if outcome.is_truncated() {
            output.push_str(&format!(
                "  搜索统计：{} 个目标、{} 次目标展开、{} 次事实检查、{} 次规则尝试\n",
                outcome.stats.goals,
                outcome.stats.goal_attempts,
                outcome.stats.fact_checks,
                outcome.stats.rule_attempts
            ));
        }
        if outcome.solutions.is_empty() && !outcome.frontier.is_empty() {
            const SHOWN: usize = 5;
            output.push_str("  知识缺口（来自本次搜索的实际失败叶子）：\n");
            for entry in outcome.frontier.iter().take(SHOWN) {
                output.push_str(&format!("    - {}\n", entry.describe()));
            }
            let omitted =
                outcome.frontier.len().saturating_sub(SHOWN) + outcome.stats.frontier_omitted;
            if omitted > 0 {
                output.push_str(&format!("    - ……另有 {} 条缺口未展开\n", omitted));
            }
        }
        output
    }

    fn render_resolution_result(&self, phrase: &str, goals: &[Atom]) -> String {
        let outcome = self.resolver.solve_outcome(goals, &[]);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结证明（一阶 Horn 归结 + 回溯证明树，纯符号）：\n");
        output.push_str(&format!(
            "  目标：{}  〔逻辑形式：{}〕\n",
            phrase, goal_display
        ));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则（含链接预测与关联规则）\n",
            self.resolver.fact_count(),
            self.resolver.clause_count()
        ));
        if let Some(solution) = outcome.solutions.first() {
            output.push_str(&format!("  得证：证明置信度 {:.2}\n", solution.confidence));
            output.push_str("  证明树：\n");
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
        } else if outcome.status == SearchStatus::Incomplete {
            output.push_str(&format!(
                "  未得证：在深度上限 {} 内搜索被截断，不能据此断言目标不存在（开世界假设）。\n",
                self.resolver.max_depth()
            ));
            output
                .push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        } else {
            output.push_str(
                "  未得证：搜索已穷尽所有分支，未找到归结反驳（开世界假设：推不出 ≠ 为假）。\n",
            );
            output
                .push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        }
        output.push_str(&self.render_search_diagnostics(&outcome));
        output
    }

    /// 证明一条陈述（"S的P…是O" / "S是类型" / "S有P"），输出回归结证明树
    pub fn resolution_proof_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请提供要证明的陈述，例如：证明：张三的师祖是王五 或 证明：姚明有国籍"
                .to_string();
        }
        match self.statement_goals(trimmed) {
            Some((phrase, goals)) => self.render_resolution_result(&phrase, &goals),
            None => format!(
                "无法把「{}」解析为可归结的目标。支持句式：S的P…是O / S是类型 / S有P。",
                trimmed
            ),
        }
    }

    /// 结构化证明视图：陈述句（"证明"）与开放问句（"推理链"）统一入口。
    /// 返回 None 表示无法形式化为归结目标或归结未启用。
    pub fn resolution_proof_view(&self, text: &str) -> Option<ResolutionProofView> {
        if !self.config.enable_resolution {
            return None;
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        if let Some(plan) = self.planner.parse(trimmed) {
            if let Some(view) = self.relation_plan_view(&plan) {
                return Some(view);
            }
        }
        if let Some(rewritten) = query_rewrite::rewrite(trimmed, &self.relation_lexicon()) {
            if let Some(view) = self.relation_plan_view(&rewritten.plan) {
                return Some(view);
            }
        }
        // 开放问句优先：剥离问句后缀（"…是谁/是什么/…"）后按谓词链处理。
        // 否则 "张三的师傅的师傅是谁" 会被误判为陈述，把「谁」当常量求证。
        let mut core = trimmed.to_string();
        let mut stripped = false;
        for suffix in [
            "是谁",
            "是什么",
            "是多少",
            "是哪里",
            "是哪个",
            "是哪儿",
            "是哪位",
            "是哪些",
            "是几",
        ] {
            if let Some(rest) = core.strip_suffix(suffix) {
                core = rest.to_string();
                stripped = true;
                break;
            }
        }
        if stripped && !core.contains('是') {
            if let Some(view) = self.build_question_view(&core) {
                return Some(view);
            }
        }
        if let Some((phrase, goals)) = self.statement_goals(trimmed) {
            return Some(self.build_resolution_view(phrase, goals));
        }
        if !core.contains('是') {
            if let Some(view) = self.build_question_view(&core) {
                return Some(view);
            }
        }
        None
    }

    fn build_question_view(&self, core: &str) -> Option<ResolutionProofView> {
        let segments: Vec<String> = core
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty() {
            return None;
        }
        let surfaces: Vec<String> = predicates.iter().map(|p| p.surface.clone()).collect();
        let goals = self.build_question_goals(&entity, &predicates, "答案");
        Some(self.build_resolution_view(format!("{}的{}是谁", entity, surfaces.join("的")), goals))
    }

    /// 开放目标（"X的P…是谁"）：返回全部归结解与各解的证明链
    pub fn resolution_query_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', ' ']);
        if trimmed.is_empty() {
            return "请提供推理目标，例如：推理链：张三的师祖是谁".to_string();
        }
        let mut core = trimmed.to_string();
        for suffix in [
            "是谁",
            "是什么",
            "是多少",
            "是哪里",
            "是哪个",
            "是哪儿",
            "是哪位",
            "是哪些",
            "是几",
        ] {
            if let Some(rest) = core.strip_suffix(suffix) {
                core = rest.to_string();
                break;
            }
        }
        // 句中仍带"是"（如"推理链：张三的师祖是王五"）→ 按证明处理
        if core.contains('是') {
            return self.resolution_proof_report(&core);
        }
        let segments: Vec<String> = core
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        let Some((entity, predicates)) = self.split_chain(&segments) else {
            return format!("无法把「{}」解析为谓词链目标。", trimmed);
        };
        if predicates.is_empty() {
            return format!(
                "「{}」只有实体没有谓词链，请补充关系，例如：推理链：{}的师傅是谁",
                trimmed, entity
            );
        }
        let answer_var = "答案".to_string();
        let goals = self.build_question_goals(&entity, &predicates, &answer_var);
        let outcome = self
            .resolver
            .solve_outcome(&goals, std::slice::from_ref(&answer_var));
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结推理链（一阶 Horn 归结，解与证明均可回溯）：\n");
        output.push_str(&format!(
            "  目标：{}的{}是谁  〔逻辑形式：{}〕\n",
            entity,
            predicates
                .iter()
                .map(|p| p.surface.as_str())
                .collect::<Vec<_>>()
                .join("的"),
            goal_display
        ));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则；深度上限 {}\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            self.resolver.max_depth()
        ));
        if outcome.solutions.is_empty() {
            if outcome.status == SearchStatus::Incomplete {
                output.push_str("  没有找到解：搜索被截断，不能据此断言不存在。\n");
            } else {
                output.push_str(
                    "  没有找到解：搜索已穷尽所有分支（开世界假设：推不出 ≠ 不存在）。\n",
                );
            }
            output.push_str(&self.render_search_diagnostics(&outcome));
            return output.trim_end().to_string();
        }
        for (index, solution) in outcome.solutions.iter().enumerate() {
            let answer = solution
                .bindings
                .get(&answer_var)
                .cloned()
                .unwrap_or_else(|| "?".to_string());
            output.push_str(&format!(
                "  解 {}（证明置信度 {:.2}）：答案是「{}」\n",
                index + 1,
                solution.confidence,
                answer
            ));
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
            output.push('\n');
        }
        output.push_str(&self.render_search_diagnostics(&outcome));
        output.trim_end().to_string()
    }

    /// 关联规则报告（可带过滤词，比如实体/类型/谓词）
    pub fn association_report(&self, filter: Option<&str>) -> String {
        if !self.config.enable_association_rules {
            return "关联规则挖掘未启用（config.toml [reasoning] enable_association_rules = false）。"
                .to_string();
        }
        let mut output = String::from("关联规则报告（支持度/置信度/提升度/杠杆率，纯计数）：\n");
        output.push_str(&format!(
            "  实体篮子 {} 个，物品 {} 种，规则 {} 条（阈值：支持度≥{:.2}，置信度≥{:.2}，提升度≥{:.2}）\n",
            self.associations.basket_count(),
            self.associations.item_kinds(),
            self.associations.rules().len(),
            self.config.association_min_support,
            self.config.association_min_confidence,
            self.config.association_min_lift
        ));
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        let normalized = filter.map(|filter| self.aligner.canonical_form(filter));
        let rules: Vec<&AssociationRule> = self
            .associations
            .rules()
            .iter()
            .filter(|rule| match filter {
                None => true,
                Some(filter) => {
                    let normalized = normalized.as_deref().unwrap_or(filter);
                    rule.consequent.contains(filter)
                        || rule.consequent.contains(normalized)
                        || rule
                            .antecedent
                            .iter()
                            .any(|item| item.contains(filter) || item.contains(normalized))
                }
            })
            .collect();
        if rules.is_empty() {
            output.push_str("  （没有满足条件/阈值的规则：可多导入同类型实体，或降低阈值）\n");
            return output;
        }
        for (index, rule) in rules.iter().take(20).enumerate() {
            output.push_str(&format!(
                "  {}. {} —— {}\n",
                index + 1,
                rule.rule_text(),
                rule.metrics_text()
            ));
        }
        if rules.len() > 20 {
            output.push_str(&format!("  ……其余 {} 条省略\n", rules.len() - 20));
        }
        output.trim_end().to_string()
    }

    /// 实体补全联想："已有特征 ⇒ 可能还具备的缺失特征"
    pub fn association_suggestions(&self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_association_rules {
            return None;
        }
        let entity = self.resolve_known_entity(entity_raw)?;
        let suggestions = self.associations.suggestions_for(&entity, 3);
        if suggestions.is_empty() {
            return None;
        }
        let basket = self.associations.basket(&entity)?;
        let mut features: Vec<&str> = basket.iter().map(|item| display_item(item)).collect();
        features.sort();
        let mut output = format!(
            "关联规则联想（{} 个实体篮子的计数统计，无神经网络）：\n",
            self.associations.basket_count()
        );
        output.push_str(&format!(
            "  「{}」已有特征：{}\n",
            entity,
            features.join("、")
        ));
        for (index, rule) in suggestions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 可能还具备「{}」——{}\n",
                index + 1,
                rule.consequent_text(),
                self.associations.explain(rule)
            ));
        }
        Some(output.trim_end().to_string())
    }

    /// 恢复通道 L2：联想提名（链接预测）——「(实体, 属性) 还可能是哪个值」。
    ///
    /// 只提名、不下结论；每条附规则证据。返回顺序由预测器确定，
    /// 调用方（恢复引擎）会再做定点化排序与验证。
    pub fn recovery_nominations(
        &self,
        entity: &str,
        attribute: &str,
    ) -> Vec<crate::recovery::RecoveryNomination> {
        if !self.config.enable_link_prediction {
            return Vec::new();
        }
        let limit = self.config.max_predictions.saturating_mul(2).max(4);
        let mut nominations = Vec::new();
        for prediction in self
            .predictor
            .predict_for_entity(entity, Some(attribute), limit)
        {
            if prediction.triple.subject != entity || prediction.triple.predicate != attribute {
                continue;
            }
            nominations.push(crate::recovery::RecoveryNomination {
                value: prediction.triple.object.clone(),
                score: prediction.confidence,
                evidence: vec![self.predictor.explain(&prediction)],
            });
        }
        nominations
    }

    /// 恢复通道 L2：符号验证——提名值是否被图谱/生成器支持或反驳。
    pub fn recovery_verify(
        &self,
        entity: &str,
        attribute: &str,
        value: &str,
    ) -> crate::recovery::VerificationOutcome {
        if self.generator.has_attribute(entity, attribute, value)
            || self.predictor.has_triple(entity, attribute, value)
        {
            return crate::recovery::VerificationOutcome::Confirmed;
        }
        if !self
            .generator
            .attribute_values(entity, attribute)
            .is_empty()
        {
            return crate::recovery::VerificationOutcome::Refuted;
        }
        crate::recovery::VerificationOutcome::NoVerdict
    }

    /// 恢复通道 L3：实体可追问的属性/关系线索（确定性排序，取前几项）。
    pub fn recovery_clues(&self, entity: &str) -> Vec<String> {
        let mut clues = self.generator.attribute_keys(entity);
        for predicate in self.generator.relation_predicates(entity) {
            if !clues.contains(&predicate) {
                clues.push(predicate);
            }
        }
        clues.truncate(3);
        clues
    }

    /// 恢复通道 L1：换帧重解——换问法重新查询（不生成，只重查）。
    pub fn recovery_reframe(
        &self,
        entity: &str,
        attribute: &str,
    ) -> Vec<crate::recovery::ReframedReading> {
        let questions = [
            format!("{}的{}是多少", entity, attribute),
            format!("{}的{}是什么", entity, attribute),
            format!("{}的{}呢", entity, attribute),
        ];
        let mut readings = Vec::new();
        for question in questions {
            let answer = SymbolicReasoner::answer(self, &question);
            readings.push(crate::recovery::ReframedReading { question, answer });
        }
        readings
    }

    /// 前向饱和推导报告：多跳新事实（含前提与规则来源）
    pub fn derived_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let derived = self.resolver.forward_chain(3, 100);
        let mut output = String::from("推理发现（前向饱和，多跳优先；均可回溯前提）：\n");
        output.push_str(&format!(
            "  现有 {} 条事实、{} 条规则，本轮推出 {} 条新事实。\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            derived.len()
        ));
        if derived.is_empty() {
            output.push_str("  （没有可推导的新事实：可增加事实/规则或放宽规则阈值）\n");
            return output;
        }
        let mut items: Vec<(&DerivedFact, String)> = derived
            .iter()
            .map(|item| (item, item.atom.display()))
            .collect();
        items.sort_by(|(a, a_display), (b, b_display)| {
            b.depth
                .cmp(&a.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a_display.cmp(b_display))
        });
        for (index, (item, _)) in items.into_iter().take(limit.max(1)).enumerate() {
            let premises = item
                .premises
                .iter()
                .map(|premise| self.render_atom_nl(premise))
                .collect::<Vec<_>>()
                .join("；");
            output.push_str(&format!(
                "  {}. {}（第 {} 轮推导；前提：{}；规则：{}；置信度 {:.2}）\n",
                index + 1,
                self.render_atom_nl(&item.atom),
                item.depth,
                premises,
                item.rule_source,
                item.confidence
            ));
        }
        output.trim_end().to_string()
    }

    fn is_known_entity(&self, name: &str) -> bool {
        if self.aligner.kind_of(name) == Some(TermKind::Entity) {
            return true;
        }
        self.session_facts
            .iter()
            .chain(self.session_negative_facts.iter())
            .any(|atom| {
                atom.terms
                    .iter()
                    .any(|term| !term.is_var() && term.name() == name)
            })
    }

    fn is_known_predicate(&self, name: &str) -> bool {
        let canonical = self.aligner.canonical_form(name);
        self.aligner.kind_of(name) == Some(TermKind::Relation)
            || self.inverse_query_map.contains_key(name)
            || self.inverse_query_map.contains_key(canonical.as_str())
            || self.predictor.has_predicate(name)
            // 正向别名（同义词）按规范化槽位回查预测器
            || (canonical != name && self.predictor.has_predicate(&canonical))
            || self
                .session_facts
                .iter()
                .chain(self.session_negative_facts.iter())
                .any(|atom| {
                    atom.predicate.as_ref() == name || atom.predicate.as_ref() == canonical
                })
    }

    fn split_chain(&self, segments: &[String]) -> Option<(String, Vec<ChainPredicate>)> {
        if segments.is_empty() {
            return None;
        }
        // 在"精确登记"的实体名上做最长前缀切分
        for take in (1..=segments.len()).rev() {
            let candidate = segments[..take].join("的");
            if self.aligner.kind_of(&candidate) == Some(TermKind::Entity) {
                let predicates = segments[take..]
                    .iter()
                    .map(|segment| {
                        let (slot, inverted) = self.resolve_question_predicate(segment);
                        ChainPredicate {
                            surface: segment.trim().to_string(),
                            slot,
                            inverted,
                        }
                    })
                    .collect();
                return Some((self.aligner.canonical_form(&candidate), predicates));
            }
        }
        let entity = self.aligner.canonical_form(&segments[0]);
        let predicates = segments[1..]
            .iter()
            .map(|segment| {
                let (slot, inverted) = self.resolve_question_predicate(segment);
                ChainPredicate {
                    surface: segment.trim().to_string(),
                    slot,
                    inverted,
                }
            })
            .collect();
        Some((entity, predicates))
    }

    /// 取一个节点沿谓词的命中：正向给出宾语集合，逆向给出主语集合。
    /// 逆向命中以槽位反方向搜索（无预测分支：符号推断仅支持正向出边）。
    fn node_hits(&self, node: &str, predicate: &ChainPredicate) -> Vec<ValueHit> {
        let mut hits: Vec<ValueHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let objects: Vec<String> = if predicate.inverted {
            self.predictor.subjects_of(&predicate.slot, node)
        } else {
            self.predictor.objects_of(node, &predicate.slot)
        };
        for object in objects {
            if seen.insert(object.clone()) {
                hits.push(ValueHit {
                    value: object,
                    prediction: None,
                    law: None,
                });
            }
        }
        // 实体属性（非关系型三元组）同样沿属性槽位命中：属性查询与关系查询
        // 走同一条链，逆向（反查主体）暂不支持属性。
        if !predicate.inverted {
            let key = (node.to_string(), predicate.slot.clone());
            let mut values = self.attribute_index.get(&key);
            if values.is_none() {
                // 图谱里常见英文属性键（JSON-LD 的 name/age/height…），
                // 用户却按中文词面提问（姓名/年龄/身高…）。仅在中文槽位
                // 无命中时回退到对照表，避免覆盖真实的中文谓词。
                if let Some(canonical) =
                    crate::symbolic_generator::canonical_predicate_key(&predicate.surface)
                {
                    values = self
                        .attribute_index
                        .get(&(node.to_string(), canonical.to_string()));
                }
            }
            if let Some(values) = values {
                for value in values {
                    if seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value: value.clone(),
                            prediction: None,
                            law: None,
                        });
                    }
                }
            }
        }
        if !predicate.inverted && self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            for prediction in self
                .predictor
                .predict_for_entity(node, Some(&predicate.slot), limit)
            {
                // 只取 node 作为主体的出边预测
                if prediction.triple.subject != node {
                    continue;
                }
                let value = prediction.triple.object.clone();
                if seen.insert(value.clone()) {
                    hits.push(ValueHit {
                        value,
                        prediction: Some(prediction),
                        law: None,
                    });
                }
            }
        }
        hits.sort_by(|a, b| a.value.cmp(&b.value));
        hits
    }

    /// 主动声明律带来的查询扩展：传递闭包 / 对称反向 / 互逆反向。
    /// 只从已观测事实做有界遍历，不调用链接预测，因此不会编造未登记取值。
    fn law_hits(
        &self,
        node: &str,
        predicate: &ChainPredicate,
        direct_empty: bool,
    ) -> (Vec<ValueHit>, bool) {
        let mut hits: Vec<ValueHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut truncated = false;
        let neighbors = |current: &str| {
            if predicate.inverted {
                self.predictor.subjects_of(&predicate.slot, current)
            } else {
                self.predictor.objects_of(current, &predicate.slot)
            }
        };

        if let Some(law) = self.laws.active_transitive(&predicate.slot) {
            let mut visited: HashSet<String> = HashSet::from([node.to_string()]);
            let mut frontier = vec![node.to_string()];
            let mut expanded = 0usize;
            let mut depth = 0usize;
            loop {
                if depth >= LAW_CLOSURE_MAX_DEPTH {
                    truncated = frontier.iter().any(|current| {
                        neighbors(current)
                            .into_iter()
                            .any(|value| value != node && !visited.contains(&value))
                    });
                    break;
                }
                depth += 1;
                let mut next: Vec<String> = Vec::new();
                for current in &frontier {
                    for value in neighbors(current) {
                        if value == node || !visited.insert(value.clone()) {
                            continue;
                        }
                        expanded += 1;
                        if expanded > LAW_CLOSURE_MAX_NODES {
                            truncated = true;
                            break;
                        }
                        next.push(value);
                    }
                    if truncated {
                        break;
                    }
                }
                if truncated || next.is_empty() {
                    break;
                }
                for value in &next {
                    if seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value: value.clone(),
                            prediction: None,
                            law: Some(law.id.clone()),
                        });
                    }
                }
                frontier = next;
            }
        }

        if direct_empty {
            if let Some(law) = self.laws.active_symmetric(&predicate.slot) {
                let values = if predicate.inverted {
                    self.predictor.objects_of(node, &predicate.slot)
                } else {
                    self.predictor.subjects_of(&predicate.slot, node)
                };
                for value in values {
                    if value != node && seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value,
                            prediction: None,
                            law: Some(law.id.clone()),
                        });
                    }
                }
            }
        }

        if direct_empty {
            if let Some(other) = self.laws.inverse_partner(&predicate.slot) {
                if let Some(law) = self.laws.active_inverse(&predicate.slot, other) {
                    let values = if predicate.inverted {
                        self.predictor.objects_of(node, other)
                    } else {
                        self.predictor.subjects_of(other, node)
                    };
                    for value in values {
                        if value != node && seen.insert(value.clone()) {
                            hits.push(ValueHit {
                                value,
                                prediction: None,
                                law: Some(law.id.clone()),
                            });
                        }
                    }
                }
            }
        }

        hits.sort_by(|a, b| a.value.cmp(&b.value));
        (hits, truncated)
    }

    /// 链上所有谓词是否都是同一个声明的传递谓词（仅词面重复时启用闭包选路）。
    fn is_transitive_chain(&self, predicates: &[ChainPredicate]) -> bool {
        if predicates.len() < 2 {
            return false;
        }
        let first = &predicates[0];
        if self.laws.active_transitive(&first.slot).is_none() {
            return false;
        }
        predicates
            .iter()
            .all(|predicate| predicate.slot == first.slot && predicate.inverted == first.inverted)
    }

    fn run_chain(&self, entity: &str, predicates: &[ChainPredicate]) -> Option<ChainRun> {
        self.run_chain_mode(entity, predicates, false)
    }

    /// 指定律闭包通道：重复传递谓词走精确步数链，其余情况回退到逐跳律扩展。
    fn run_chain_via_laws(&self, entity: &str, predicates: &[ChainPredicate]) -> Option<ChainRun> {
        if self.is_transitive_chain(predicates) {
            return self.run_transitive_chain_exact(entity, predicates);
        }
        self.run_chain_mode(entity, predicates, true)
    }

    /// 传递链的精确语义：`P(P(x))` 是恰好两跳，而不是「所有可达后代」。
    /// 传递律在这里负责授权与归因；闭包可达性不能替代步数约束。
    fn run_transitive_chain_exact(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
    ) -> Option<ChainRun> {
        let law = self.laws.active_transitive(&predicates[0].slot)?;
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            let mut seen_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let mut hits = self.node_hits(node, predicate);
                for hit in &mut hits {
                    if hit.prediction.is_none() {
                        hit.law = Some(law.id.clone());
                    }
                }
                if hits.is_empty() {
                    continue;
                }
                for hit in &hits {
                    if seen_values.insert(hit.value.clone()) {
                        next_frontier.push(hit.value.clone());
                        round_hits.push(hit.clone());
                    }
                }
                steps.push(ChainStep {
                    subject: node.clone(),
                    predicate: predicate.surface.clone(),
                    hits,
                });
            }
            if next_frontier.is_empty() {
                return None;
            }
            frontier = next_frontier;
            last_hits = round_hits;
        }
        if last_hits.is_empty() {
            return None;
        }
        Some(ChainRun {
            steps,
            last_hits,
            closure_truncated: false,
        })
    }

    fn run_chain_mode(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        force_closure: bool,
    ) -> Option<ChainRun> {
        if predicates.is_empty() {
            return None;
        }
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        let mut closure_truncated = false;
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            // 多个分支可能命中同一实体，边收集边去重：两个集合插入条件相同，
            // 合并为一个即可（保留首次出现顺序）
            let mut seen_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let direct = self.node_hits(node, predicate);
                let hits = if force_closure || direct.is_empty() {
                    let (expanded, expanded_truncated) =
                        self.law_hits(node, predicate, direct.is_empty());
                    closure_truncated |= expanded_truncated;
                    if force_closure && !expanded.is_empty() {
                        // 闭包结果优先保留律归属；直接事实只补闭包未覆盖的取值。
                        let mut merged = expanded;
                        let mut seen_hits: HashSet<String> =
                            merged.iter().map(|hit| hit.value.clone()).collect();
                        for hit in direct {
                            if seen_hits.insert(hit.value.clone()) {
                                merged.push(hit);
                            }
                        }
                        merged.sort_by(|a, b| a.value.cmp(&b.value));
                        merged
                    } else if direct.is_empty() {
                        expanded
                    } else {
                        direct
                    }
                } else {
                    direct
                };
                if hits.is_empty() {
                    continue;
                }
                for hit in &hits {
                    if seen_values.insert(hit.value.clone()) {
                        next_frontier.push(hit.value.clone());
                        round_hits.push(hit.clone());
                    }
                }
                steps.push(ChainStep {
                    subject: node.clone(),
                    predicate: predicate.surface.clone(),
                    hits,
                });
            }
            if next_frontier.is_empty() {
                return None;
            }
            frontier = next_frontier;
            last_hits = round_hits;
        }
        if last_hits.is_empty() {
            return None;
        }
        Some(ChainRun {
            steps,
            last_hits,
            closure_truncated,
        })
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先"
        } else if index + 1 == total {
            "最后"
        } else {
            "此外"
        }
    }

    // 期望答案类型（形式语义中的问题类型约束）：谁→人物、哪里→地点、何时→时间
    // 疑问词识别与关键词表由虚词算符表提供（`function_words::answer_type_keywords`）
    fn expected_type_keywords(question: &str) -> Vec<&'static str> {
        crate::function_words::answer_type_keywords(question)
    }

    fn value_matches_expectation(&self, value: &str, keywords: &[&str]) -> bool {
        let Some(etype) = self.generator.entity_type_of(value) else {
            return false;
        };
        let etype_lower = etype.to_lowercase();
        // 期望类型关键词均为小写（或中文）
        keywords
            .iter()
            .any(|keyword| etype.contains(*keyword) || etype_lower.contains(*keyword))
    }

    fn generate_find_object(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        steps: &[ChainStep],
        last_hits: &[ValueHit],
        question: &str,
        closure_truncated: bool,
    ) -> String {
        let expectations = Self::expected_type_keywords(question);
        let mut ordered: Vec<&ValueHit> = last_hits.iter().collect();
        if !expectations.is_empty() {
            ordered.sort_by_cached_key(|hit| {
                if self.value_matches_expectation(&hit.value, &expectations) {
                    0usize
                } else {
                    1usize
                }
            });
        }
        let reordered = ordered
            .iter()
            .zip(last_hits.iter())
            .any(|(a, b)| !std::ptr::eq(*a, b));
        // 只复述真正用到的律；直接事实不冒充「推理得出」。
        let law_notes: Vec<String> = steps
            .iter()
            .flat_map(|step| step.hits.iter())
            .chain(last_hits.iter())
            .filter_map(|hit| hit.law.as_deref())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| self.laws.explain(id))
            .collect();
        let chain_text = format!(
            "{} 的 {}",
            entity,
            predicates
                .iter()
                .map(|p| p.surface.as_str())
                .collect::<Vec<_>>()
                .join(" 的 ")
        );
        let mut output = String::from("根据知识图谱");
        if steps.len() > 1 {
            output.push_str("（多跳推理）");
        }
        output.push_str("：\n");
        if steps.len() > 1 {
            for step in steps {
                let values = step
                    .hits
                    .iter()
                    .map(|hit| hit.value.clone())
                    .collect::<Vec<_>>()
                    .join("、");
                output.push_str(&format!(
                    "  {} 的 {} 是 {}。\n",
                    step.subject, step.predicate, values
                ));
            }
            output.push_str("  综合得出：\n");
        }
        if last_hits.len() == 1 {
            let hit = &last_hits[0];
            output.push_str(&format!("  {} 是 {}。", chain_text, hit.value));
            if let Some(prediction) = &hit.prediction {
                output.push_str(&format!(
                    "\n  推理依据：{}",
                    self.predictor.explain(prediction)
                ));
            } else if let Some(law_id) = &hit.law {
                if let Some(explain) = self.laws.explain(law_id) {
                    output.push_str(&format!("\n  推理依据：{}", explain));
                }
            } else if !law_notes.is_empty() {
                output.push_str(&format!("\n  推理依据：{}", law_notes.join("；")));
            }
        } else {
            for (index, hit) in ordered.iter().enumerate() {
                let marker = Self::discourse_marker(index, ordered.len());
                output.push_str(&format!(
                    "  {}{}{} 是 {}。",
                    marker,
                    if marker.is_empty() { "" } else { "，" },
                    chain_text,
                    hit.value
                ));
                if let Some(prediction) = &hit.prediction {
                    output.push_str(&format!("（推断，置信度 {:.2}）", prediction.confidence));
                }
                output.push('\n');
            }
            if last_hits.iter().any(|hit| hit.prediction.is_some()) {
                let evidence: Vec<String> = last_hits
                    .iter()
                    .filter_map(|hit| {
                        hit.prediction
                            .as_ref()
                            .map(|prediction| self.predictor.explain(prediction))
                    })
                    .take(2)
                    .collect();
                if !evidence.is_empty() {
                    output.push_str(&format!("  推理依据：{}", evidence.join("；")));
                }
            }
            if !law_notes.is_empty() {
                output.push_str(&format!("  推理依据：{}\n", law_notes.join("；")));
            }
            if reordered {
                output.push_str(&format!(
                    "\n  说明：已按问题期望的答案类型（{}）优先排序。",
                    expectations.join("/")
                ));
            }
        }
        if closure_truncated {
            output.push_str("\n  说明：传递闭包扩展达到深度或节点上限，未列出的取值仍可能存在。");
        }
        output
    }

    fn traversal_graph(&self) -> TraversalGraph {
        let mut graph = TraversalGraph::new();
        // 主语实体上的 valid_from/valid_to 等字段给出其出边的有效时间；
        // 没有字段的边保持 Unknown，带时间约束的查询按宽松策略标注 caveat。
        let mut bounds: BTreeMap<&str, (Option<&str>, Option<&str>)> = BTreeMap::new();
        for (entity, attribute, value) in &self.attributes {
            let entry = bounds.entry(entity.as_str()).or_default();
            if TEMPORAL_FROM_FIELDS.contains(&attribute.as_str()) {
                entry.0 = Some(value.as_str());
            } else if TEMPORAL_TO_FIELDS.contains(&attribute.as_str()) {
                entry.1 = Some(value.as_str());
            }
        }
        let mut validity_by_entity: BTreeMap<&str, Validity> = bounds
            .into_iter()
            .filter_map(|(entity, (start, end))| {
                let validity = validity_from_bounds(start, end);
                (!validity.is_unknown()).then_some((entity, validity))
            })
            .collect();
        // 实体键可能是 IRI 尾部，路径端点则用 name 属性字面量；登记别名，
        // 否则从知识图谱导入的历史有效期会在路径上丢失。
        let aliases: Vec<(&str, &str)> = self
            .attributes
            .iter()
            .filter(|(_, attribute, _)| attribute == "name")
            .map(|(entity, _, value)| (entity.as_str(), value.as_str()))
            .collect();
        for (entity, alias) in aliases {
            if let Some(validity) = validity_by_entity.get(entity).copied() {
                validity_by_entity.insert(alias, validity);
            }
        }
        for triple in self.predictor.triples() {
            let validity = validity_by_entity
                .get(triple.subject.as_str())
                .copied()
                .unwrap_or(Validity::Unknown);
            graph.add_edge_record(
                crate::path_memory::PathEdge::new(
                    &triple.subject,
                    &triple.predicate,
                    &triple.object,
                )
                .with_validity(validity),
            );
        }
        for (entity, attribute, value) in &self.attributes {
            if TEMPORAL_FROM_FIELDS.contains(&attribute.as_str())
                || TEMPORAL_TO_FIELDS.contains(&attribute.as_str())
            {
                continue;
            }
            let validity = validity_by_entity
                .get(entity.as_str())
                .copied()
                .unwrap_or(Validity::Unknown);
            graph.add_edge_record(
                crate::path_memory::PathEdge::new(entity, attribute, value)
                    .with_source(EdgeSource::Attribute)
                    .with_validity(validity),
            );
        }
        for atom in &self.session_facts {
            if atom.terms.len() == 2 {
                if let (Term::Const(subject), Term::Const(object)) =
                    (&atom.terms[0], &atom.terms[1])
                {
                    graph.add_edge_with_source(
                        subject,
                        &atom.predicate,
                        object,
                        EdgeSource::Session,
                    );
                }
            }
        }
        for atom in &self.session_negative_facts {
            if atom.terms.len() == 2 {
                if let (Term::Const(subject), Term::Const(object)) =
                    (&atom.terms[0], &atom.terms[1])
                {
                    graph.add_negative_edge(subject, &atom.predicate, object);
                }
            }
        }
        for (entity, entity_type) in self.generator.typed_entities() {
            graph.add_entity_type(&entity, &entity_type);
        }
        graph
    }

    fn session_negative_blocks(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.session_negative_facts.iter().any(|atom| {
            &*atom.predicate == predicate
                && atom.terms.len() == 2
                && atom.terms[0].name() == subject
                && atom.terms[1].name() == object
        })
    }

    fn path_entity(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        if self.is_known_entity(trimmed) || self.predictor.participates(trimmed) {
            trimmed.to_string()
        } else {
            self.aligner.canonical_form(trimmed)
        }
    }

    fn path_options(&self, max_depth: usize) -> TraversalOptions {
        TraversalOptions {
            max_depth: max_depth.clamp(1, 6),
            max_paths: 12,
            ..TraversalOptions::default()
        }
    }

    fn path_options_for_question(&self, max_depth: usize, question: &str) -> TraversalOptions {
        let mut options = self.path_options(max_depth);
        if let Some(span) = temporal::parse_time_span(question) {
            options.temporal = TemporalConstraint::from_span(span);
            // 现有知识库大多没有 valid-time 元数据：未知时间只作背景并标注，
            // 但已经写明有效期的边仍会被时间合一硬性阻断。
            options.unknown_time = UnknownTimePolicy::Background;
        }
        let keywords = Self::expected_type_keywords(question);
        if keywords.is_empty() {
            return options;
        }
        let mut expected = BTreeSet::new();
        for (_, entity_type) in self.generator.typed_entities() {
            let lower = entity_type.to_lowercase();
            if keywords
                .iter()
                .any(|keyword| entity_type.contains(*keyword) || lower.contains(*keyword))
            {
                expected.insert(entity_type);
            }
        }
        options.expected_end_types = expected;
        options
    }

    fn answer_chain_with_paths(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        question: &str,
    ) -> Option<String> {
        if predicates.len() < 2 {
            return None;
        }
        let graph = self.traversal_graph();
        let chain: Vec<PathPredicate> = predicates
            .iter()
            .map(|predicate| PathPredicate {
                predicate: predicate.slot.clone(),
                inverted: predicate.inverted,
            })
            .collect();
        let options = self.path_options_for_question(predicates.len(), question);
        let bundle = graph.enumerate_chain(entity, &chain, &options);
        if bundle.witnesses.is_empty() {
            if bundle.blocked.iter().any(|blocked| {
                matches!(
                    blocked.reason,
                    crate::path_memory::BlockReason::NegatedEdge { .. }
                )
            }) {
                return Some(bundle.render_text());
            }
            return None;
        }

        let expectations = Self::expected_type_keywords(question);
        let mut groups: Vec<(String, Vec<usize>)> = bundle.answer_groups().into_iter().collect();
        groups.sort_by(|(left, _), (right, _)| {
            let left_rank =
                if expectations.is_empty() || self.value_matches_expectation(left, &expectations) {
                    0
                } else {
                    1
                };
            let right_rank = if expectations.is_empty()
                || self.value_matches_expectation(right, &expectations)
            {
                0
            } else {
                1
            };
            left_rank.cmp(&right_rank).then(left.cmp(right))
        });

        let chain_text = format!(
            "{} 的 {}",
            entity,
            predicates
                .iter()
                .map(|predicate| predicate.surface.as_str())
                .collect::<Vec<_>>()
                .join(" 的 ")
        );
        let mut output = format!(
            "根据知识图谱（多跳路径，共 {} 条）：\n",
            bundle.witnesses.len()
        );
        let mut path_index = 0usize;
        for (value, indices) in &groups {
            output.push_str(&format!("  结论：{} 是 {}。\n", chain_text, value));
            for index in indices {
                if let Some(witness) = bundle.witnesses.get(*index) {
                    path_index += 1;
                    output.push_str(&format!(
                        "    路径 {}（{} 跳）：{}\n",
                        path_index,
                        witness.hop_count(),
                        witness.render()
                    ));
                }
            }
        }
        if !bundle.blocked.is_empty() {
            let blockers: Vec<String> = bundle
                .blocked
                .iter()
                .take(3)
                .map(|blocked| blocked.render())
                .collect();
            output.push_str(&format!("  阻断：{}\n", blockers.join("；")));
        }
        let law_notes: Vec<String> = predicates
            .iter()
            .filter_map(|predicate| self.laws.active_transitive(&predicate.slot))
            .map(|law| law.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| self.laws.explain(&id))
            .collect();
        if !law_notes.is_empty() {
            output.push_str(&format!("  推理依据：{}\n", law_notes.join("；")));
        }
        if bundle.truncated {
            output.push_str("  说明：路径结果已按上限截断。\n");
        }
        Some(output.trim_end().to_string())
    }

    fn answer_path_between(&self, start: &str, goal: &str, max_hops: usize) -> Option<String> {
        let start = self.path_entity(start);
        let goal = self.path_entity(goal);
        if start.is_empty() || goal.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.enumerate_between(&start, &goal, &self.path_options(max_hops));
        Some(bundle.render_text())
    }

    fn answer_common_neighbor(&self, left: &str, right: &str) -> Option<String> {
        let left = self.path_entity(left);
        let right = self.path_entity(right);
        if left.is_empty() || right.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.common_neighbors(&left, &right, &self.path_options(2));
        if bundle.witnesses.is_empty() {
            return Some(bundle.render_text());
        }
        let mut groups: Vec<(String, Vec<usize>)> =
            bundle.intermediary_groups().into_iter().collect();
        groups.sort_by(|(left, left_paths), (right, right_paths)| {
            right_paths
                .len()
                .cmp(&left_paths.len())
                .then(left.cmp(right))
        });
        let mut output = format!("共同邻居（{} 和 {}）：\n", left, right);
        for (hub, indices) in groups {
            output.push_str(&format!("  {}（{} 条路径）\n", hub, indices.len()));
            for index in indices {
                if let Some(witness) = bundle.witnesses.get(index) {
                    output.push_str(&format!("    {}\n", witness.render()));
                }
            }
        }
        if bundle.truncated {
            output.push_str("  说明：结果已按上限截断。\n");
        }
        Some(output.trim_end().to_string())
    }

    fn answer_intermediary(&self, start: &str, goal: &str, max_hops: usize) -> Option<String> {
        let start = self.path_entity(start);
        let goal = self.path_entity(goal);
        if start.is_empty() || goal.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.enumerate_between(&start, &goal, &self.path_options(max_hops));
        if bundle.witnesses.is_empty() {
            return Some(bundle.render_text());
        }
        let mut groups: Vec<(String, Vec<usize>)> =
            bundle.intermediary_groups().into_iter().collect();
        groups.sort_by(|(left, left_paths), (right, right_paths)| {
            right_paths
                .len()
                .cmp(&left_paths.len())
                .then(left.cmp(right))
        });
        let mut output = format!("从「{}」到「{}」的路径：\n", start, goal);
        if groups.is_empty() {
            output.push_str("  直接相连，不需要中间人。\n");
        } else {
            for (hub, indices) in groups {
                output.push_str(&format!("  中间人：{}（{} 条路径）\n", hub, indices.len()));
                for index in indices {
                    if let Some(witness) = bundle.witnesses.get(index) {
                        output.push_str(&format!("    {}\n", witness.render()));
                    }
                }
            }
        }
        output.push_str(&format!(
            "  路径枚举：{} 条，扩展 {} 次，访问 {} 个节点。\n",
            bundle.witnesses.len(),
            bundle.expansions,
            bundle.visited
        ));
        if !bundle.blocked.is_empty() {
            let blockers: Vec<String> = bundle
                .blocked
                .iter()
                .take(3)
                .map(|blocked| blocked.render())
                .collect();
            output.push_str(&format!("  阻断：{}\n", blockers.join("；")));
        }
        Some(output.trim_end().to_string())
    }

    fn answer_planned(&self, question: &str) -> Option<String> {
        let parsed = self.planner.parse(question);
        if let Some(answer) = parsed
            .as_ref()
            .and_then(|plan| self.answer_plan(plan, question))
        {
            return Some(answer);
        }
        // 已有计划可能只解析了句子的前半段（例如「X 有没有 V Y」被误读成
        // 存在性问句）。此时用句法重写器再编译一次，词表校验不过就弃权。
        let rewritten = query_rewrite::rewrite(question, &self.relation_lexicon())?;
        if parsed.as_ref() == Some(&rewritten.plan) {
            return None;
        }
        self.answer_plan(&rewritten.plan, question)
    }

    fn answer_plan(&self, plan: &QueryPlan, question: &str) -> Option<String> {
        let plan = plan.clone();
        match plan {
            QueryPlan::FindObject { segments } => {
                let (entity, predicates) = self.split_chain(&segments)?;
                if predicates.is_empty() {
                    return None;
                }
                if self.config.enable_clarification
                    && predicates.len() == 1
                    && self.is_known_entity(&entity)
                {
                    if let Some(decision) =
                        self.attribute_alignment_decision(&entity, &predicates[0].surface)
                    {
                        if decision.should_clarify() {
                            return Some(self.render_alignment_clarification(
                                &predicates[0].surface,
                                &decision,
                            ));
                        }
                    }
                }
                if predicates.len() >= 2 {
                    if let Some(answer) =
                        self.answer_chain_with_paths(&entity, &predicates, question)
                    {
                        return Some(answer);
                    }
                }
                let chain_result = if self.is_transitive_chain(&predicates) {
                    // 重复的传递谓词：用 Closure 算子压缩链，trace 会引用该律。
                    self.run_chain_via_laws(&entity, &predicates)
                } else {
                    self.run_chain(&entity, &predicates).or_else(|| {
                        if self.laws.has_inference_laws() {
                            self.run_chain_via_laws(&entity, &predicates)
                        } else {
                            None
                        }
                    })
                };
                if let Some(run) = chain_result {
                    return Some(self.generate_find_object(
                        &entity,
                        &predicates,
                        &run.steps,
                        &run.last_hits,
                        question,
                        run.closure_truncated,
                    ));
                }
                if self.is_known_entity(&entity)
                    && predicates
                        .iter()
                        .all(|p| self.is_known_predicate(&p.surface))
                {
                    return Some(format!(
                        "根据现有知识库，没有记录「{} 的 {}」。",
                        entity,
                        predicates
                            .iter()
                            .map(|p| p.surface.as_str())
                            .collect::<Vec<_>>()
                            .join(" 的 ")
                    ));
                }
                None
            }
            QueryPlan::Count { subject, predicate } => {
                self.answer_count(&subject, &predicate, false)
            }
            QueryPlan::Exist { subject, predicate } => {
                self.answer_count(&subject, &predicate, true)
            }
            QueryPlan::Closure {
                subject,
                predicate,
                operator,
                count_only,
                then,
            } => self.answer_closure(&subject, &predicate, operator, count_only, &then),
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => self.answer_check(&subject_segments, &object),
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => self.answer_check_relation(&subject, &predicate, &object, negated),
            QueryPlan::FindSubject { object, predicate } => {
                self.answer_find_subject(&object, &predicate)
            }
            QueryPlan::FindObjectByPredicate { subject, predicate } => {
                self.answer_relation_objects(&subject, &predicate)
            }
            QueryPlan::FindObjectExcept {
                subject,
                predicate,
                excluded,
            } => self.answer_relation_objects_filtered(&subject, &predicate, &excluded),
            QueryPlan::FindSubjectExcept {
                object,
                predicate,
                excluded,
            } => self.answer_find_subject_filtered(&object, &predicate, &excluded),
            QueryPlan::PathBetween {
                start,
                goal,
                max_hops,
            } => self.answer_path_between(&start, &goal, max_hops),
            QueryPlan::CommonNeighbor { left, right } => self.answer_common_neighbor(&left, &right),
            QueryPlan::Intermediary {
                start,
                goal,
                max_hops,
            } => self.answer_intermediary(&start, &goal, max_hops),
            QueryPlan::Similar { entity } => self.answer_similar(&entity),
            QueryPlan::Analogy { a, b, c } => self.answer_analogy(&a, &b, &c),
        }
    }

    fn closure_adjacency(&self, slot: &str, inverted: bool) -> BTreeMap<String, BTreeSet<String>> {
        fn add_edge(adjacency: &mut BTreeMap<String, BTreeSet<String>>, from: &str, to: &str) {
            adjacency
                .entry(from.to_string())
                .or_default()
                .insert(to.to_string());
        }

        let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for triple in self.predictor.triples() {
            if triple.predicate != slot {
                continue;
            }
            if inverted {
                add_edge(&mut adjacency, &triple.object, &triple.subject);
            } else {
                add_edge(&mut adjacency, &triple.subject, &triple.object);
            }
        }
        for atom in &self.session_facts {
            if atom.predicate.as_ref() != slot || atom.terms.len() != 2 {
                continue;
            }
            if let (Term::Const(subject), Term::Const(object)) = (&atom.terms[0], &atom.terms[1]) {
                if inverted {
                    add_edge(&mut adjacency, object, subject);
                } else {
                    add_edge(&mut adjacency, subject, object);
                }
            }
        }
        for atom in &self.session_negative_facts {
            if atom.predicate.as_ref() != slot || atom.terms.len() != 2 {
                continue;
            }
            if let (Term::Const(subject), Term::Const(object)) = (&atom.terms[0], &atom.terms[1]) {
                let (from, to) = if inverted {
                    (object.as_ref(), subject.as_ref())
                } else {
                    (subject.as_ref(), object.as_ref())
                };
                if let Some(values) = adjacency.get_mut(from) {
                    values.remove(to);
                }
            }
        }
        adjacency
    }

    fn answer_closure(
        &self,
        subject: &str,
        predicate: &str,
        operator: ClosureOp,
        count_only: bool,
        then: &[String],
    ) -> Option<String> {
        let entity = self.aligner.canonical_form(subject.trim());
        let surface = predicate.trim();
        if entity.is_empty() || surface.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) && !self.predictor.participates(&entity) {
            return None;
        }
        let (slot, inverted) = self.resolve_question_predicate(surface);
        if slot.is_empty() || (!self.is_known_predicate(&slot) && !self.is_known_predicate(surface))
        {
            return None;
        }
        let mut then_links: Vec<(String, String, bool)> = Vec::new();
        for raw in then {
            let raw = raw.trim();
            let (then_slot, then_inverted) = self.resolve_question_predicate(raw);
            if then_slot.is_empty()
                || (!self.is_known_predicate(&then_slot) && !self.is_known_predicate(raw))
            {
                return None;
            }
            then_links.push((raw.to_string(), then_slot, then_inverted));
        }

        let adjacency = self.closure_adjacency(&slot, inverted);
        let law = if operator.requires_transitive_law() {
            match self.laws.active_transitive(&slot) {
                Some(law) => Some(law.id.clone()),
                None => {
                    let direct: Vec<String> = adjacency
                        .get(&entity)
                        .map(|values| values.iter().cloned().collect())
                        .unwrap_or_default();
                    let direct_text = if direct.is_empty() {
                        "当前没有直接记录".to_string()
                    } else {
                        format!("当前只有直接记录：{}", direct.join("、"))
                    };
                    return Some(format!(
                        "无法安全回答「{}」的所有「{}」：谓词「{}」尚未声明为传递关系。{}。若要让「所有」按可达闭包回答，请先在语义律中启用传递律。",
                        entity, surface, slot, direct_text
                    ));
                }
            }
        } else {
            None
        };

        let mut budget = ClosureBudget::default();
        if let ClosureOp::Exact(hops) = operator {
            budget.max_depth = budget.max_depth.max(hops);
        }
        if !then_links.is_empty() {
            budget.max_witnesses = budget.max_nodes;
        }
        let outcome = evaluate_closure(&entity, operator, budget, law.as_deref(), |node| {
            adjacency
                .get(node)
                .map(|values| values.iter().cloned().collect())
                .unwrap_or_default()
        })
        .ok()?;
        if then_links.is_empty() {
            return Some(self.render_closure_answer(&entity, surface, &outcome, count_only));
        }
        let composed = self.extend_closure_targets(&outcome, &then_links);
        Some(self.render_closure_chain_answer(
            &entity,
            surface,
            &then_links,
            &outcome,
            count_only,
            &composed,
        ))
    }

    fn extend_closure_targets(
        &self,
        outcome: &ClosureOutcome,
        then: &[(String, String, bool)],
    ) -> Vec<(String, Vec<String>)> {
        let mut current: BTreeMap<String, Vec<String>> = outcome
            .witnesses
            .iter()
            .map(|(target, path)| (target.clone(), path.clone()))
            .collect();
        for (_, slot, inverted) in then {
            let adjacency = self.closure_adjacency(slot, *inverted);
            let mut ordered: Vec<(String, Vec<String>)> = current.into_iter().collect();
            ordered.sort_by(|(left_name, left_path), (right_name, right_path)| {
                left_path
                    .len()
                    .cmp(&right_path.len())
                    .then_with(|| left_name.cmp(right_name))
            });
            let mut next: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (node, path) in ordered {
                let values: Vec<String> = adjacency
                    .get(&node)
                    .map(|neighbors| neighbors.iter().cloned().collect())
                    .unwrap_or_default();
                for value in values {
                    let mut extended = path.clone();
                    extended.push(value.clone());
                    let replace = next
                        .get(&value)
                        .map_or(true, |existing| existing.len() > extended.len());
                    if replace {
                        next.insert(value, extended);
                    }
                }
            }
            current = next;
            if current.is_empty() {
                break;
            }
        }
        let mut result: Vec<(String, Vec<String>)> = current.into_iter().collect();
        result.sort_by(|(left_name, left_path), (right_name, right_path)| {
            left_path
                .len()
                .cmp(&right_path.len())
                .then_with(|| left_name.cmp(right_name))
        });
        result
    }

    fn render_closure_chain_answer(
        &self,
        entity: &str,
        surface: &str,
        then: &[(String, String, bool)],
        outcome: &ClosureOutcome,
        count_only: bool,
        composed: &[(String, Vec<String>)],
    ) -> String {
        let suffix = then
            .iter()
            .map(|(name, _, _)| format!("的「{}」", name))
            .collect::<String>();
        let target_phrase = format!("所有「{}」{}", surface, suffix);
        let basis = outcome
            .law
            .as_deref()
            .map(|law| format!("；依据 {}", law))
            .unwrap_or_default();
        if composed.is_empty() {
            if outcome.complete() {
                return format!("根据现有知识库，「{}」没有{}。", entity, target_phrase);
            }
            let mut output = format!(
                "根据知识图谱（{}{}）：「{}」暂未找到{}。",
                outcome.operator.label(),
                basis,
                entity,
                target_phrase
            );
            if let Some(truncation) = outcome.truncated {
                output.push_str(&format!(
                    "
  说明：闭包搜索在{}处停止，不能排除仍有更多结果。",
                    truncation.label()
                ));
            }
            return output;
        }

        let count = if outcome.count_is_lower_bound() {
            format!("至少 {} 个", composed.len())
        } else {
            format!("共 {} 个", composed.len())
        };
        let mut output = format!(
            "根据知识图谱（{}{}）：「{}」的{} {}。",
            outcome.operator.label(),
            basis,
            entity,
            target_phrase,
            count
        );
        if !count_only {
            for (index, (target, path)) in composed.iter().take(12).enumerate() {
                output.push_str(&format!(
                    "
  {}. {}（{} 跳：{}）",
                    index + 1,
                    target,
                    path.len().saturating_sub(1),
                    path.join(" → ")
                ));
            }
            if composed.len() > 12 {
                output.push_str(&format!(
                    "
  说明：其余 {} 个结果未展开。",
                    composed.len() - 12
                ));
            }
        } else {
            output.push_str(
                "
  计数口径：不同最终节点，不按路径重复计数。",
            );
        }
        if let Some(truncation) = outcome.truncated {
            output.push_str(&format!(
                "
  说明：闭包搜索在{}处停止，结果不完整，上述计数是下界（至少 {} 个）。",
                truncation.label(),
                composed.len()
            ));
        }
        output
    }

    fn render_closure_answer(
        &self,
        entity: &str,
        surface: &str,
        outcome: &ClosureOutcome,
        count_only: bool,
    ) -> String {
        let target_phrase = match outcome.operator {
            ClosureOp::Exact(hops) => format!("恰好 {} 跳的「{}」", hops, surface),
            ClosureOp::Plus => format!("所有「{}」", surface),
            ClosureOp::Star => format!("所有（含自身）「{}」", surface),
        };
        if outcome.reached_count == 0 && outcome.complete() {
            return format!("根据现有知识库，「{}」没有{}。", entity, target_phrase);
        }

        let basis = outcome
            .law
            .as_deref()
            .map(|law| format!("；依据 {}", law))
            .unwrap_or_default();
        let count = if outcome.count_is_lower_bound() {
            format!("至少 {} 个", outcome.reached_count)
        } else {
            format!("共 {} 个", outcome.reached_count)
        };
        let mut output = format!(
            "根据知识图谱（{}{}）：「{}」的{} {}。",
            outcome.operator.label(),
            basis,
            entity,
            target_phrase,
            count
        );
        if !count_only {
            for (index, (target, hops)) in outcome.targets_by_distance().into_iter().enumerate() {
                let witness = outcome
                    .render_witness(target)
                    .unwrap_or_else(|| target.to_string());
                output.push_str(&format!(
                    "
  {}. {}（{} 跳：{}）",
                    index + 1,
                    target,
                    hops,
                    witness
                ));
            }
        } else {
            output.push_str(
                "
  计数口径：不同可达节点，不按路径重复计数。",
            );
        }
        if outcome.witnesses_truncated {
            output.push_str(&format!(
                "
  说明：见证仅保留前 {} 个，计数仍按不同节点去重。",
                outcome.witnesses.len()
            ));
        }
        if let Some(truncation) = outcome.truncated {
            output.push_str(&format!(
                "
  说明：闭包搜索在{}处停止，结果不完整，上述计数是下界（至少 {} 个）。",
                truncation.label(),
                outcome.reached_count
            ));
        }
        output
    }

    fn answer_count(&self, subject: &str, predicate: &str, exist: bool) -> Option<String> {
        let entity = self.aligner.canonical_form(subject.trim());
        if entity.is_empty() {
            return None;
        }
        let (slot, inverted) = self.resolve_question_predicate(predicate);
        if slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) && !self.predictor.participates(&entity) {
            return None;
        }
        let chain_predicate = ChainPredicate {
            surface: predicate.trim().to_string(),
            slot: slot.clone(),
            inverted,
        };
        let hits = self.node_hits(&entity, &chain_predicate);
        if hits.is_empty() {
            if self.is_known_predicate(&chain_predicate.surface) {
                if exist {
                    return Some(format!(
                        "根据现有知识库，「{}」没有「{}」的记录。",
                        entity, chain_predicate.surface
                    ));
                }
                return Some(format!(
                    "根据现有知识库，「{}」的「{}」数量为 0（没有相关记录）。",
                    entity, chain_predicate.surface
                ));
            }
            return None;
        }
        let values: Vec<String> = hits.iter().map(|hit| hit.value.clone()).collect();
        let mut output = if exist {
            format!(
                "是的，根据知识图谱，「{}」有 {}（共 {} 个：{}）。",
                entity,
                chain_predicate.surface,
                values.len(),
                values.join("、")
            )
        } else {
            format!(
                "根据知识图谱：「{}」的「{}」共有 {} 个：{}。",
                entity,
                chain_predicate.surface,
                values.len(),
                values.join("、")
            )
        };
        if let Some(prediction) = hits.iter().find_map(|hit| hit.prediction.as_ref()) {
            output.push_str(&format!(
                "\n其中为符号推断结果（例如置信度 {:.2}）：{}",
                prediction.confidence,
                self.predictor.explain(prediction)
            ));
        }
        Some(output)
    }

    fn answer_check(&self, subject_segments: &[String], object_raw: &str) -> Option<String> {
        if subject_segments.len() < 2 {
            return None;
        }
        let predicate_raw = subject_segments.last()?.clone();
        let (entity, mut intermediate) =
            self.split_chain(&subject_segments[..subject_segments.len() - 1])?;
        let (slot, inverted) = self.resolve_question_predicate(&predicate_raw);
        let object = self.aligner.canonical_form(object_raw.trim());
        if entity.is_empty() || slot.is_empty() || object.is_empty() {
            return None;
        }
        let surface = predicate_raw.trim().to_string();
        intermediate.push(ChainPredicate {
            surface: surface.clone(),
            slot: slot.clone(),
            inverted,
        });

        if intermediate.len() == 1 {
            let (negative_subject, negative_object) = if inverted {
                (object.as_str(), entity.as_str())
            } else {
                (entity.as_str(), object.as_str())
            };
            if self.session_negative_blocks(negative_subject, &slot, negative_object) {
                return Some(format!(
                    "不是。会话中记录了否定事实：{} 的 {} 不是 {}。",
                    entity, surface, object
                ));
            }
        }

        // 单跳直查：逆向谓词按槽位反方向核对（triple(object, slot, entity)）
        if intermediate.len() == 1
            && (if inverted {
                self.predictor.has_triple(&object, &slot, &entity)
            } else {
                self.predictor.has_triple(&entity, &slot, &object)
            })
        {
            return Some(format!(
                "是的。根据知识图谱：{} 的 {} 是 {}。",
                entity, surface, object
            ));
        }

        if intermediate.len() >= 2 {
            let graph = self.traversal_graph();
            let chain: Vec<PathPredicate> = intermediate
                .iter()
                .map(|predicate| PathPredicate {
                    predicate: predicate.slot.clone(),
                    inverted: predicate.inverted,
                })
                .collect();
            let bundle =
                graph.enumerate_chain(&entity, &chain, &self.path_options(intermediate.len()));
            let negated_target = bundle.blocked.iter().any(|blocked| {
                matches!(
                    &blocked.reason,
                    crate::path_memory::BlockReason::NegatedEdge { to, .. } if to == &object
                )
            });
            if negated_target {
                return Some(format!(
                    "不是。会话中记录的否定事实阻断了通往 {} 的路径。",
                    object
                ));
            }
            let matched: Vec<&crate::path_memory::PathWitness> = bundle
                .witnesses
                .iter()
                .filter(|witness| witness.end() == Some(object.as_str()))
                .collect();
            if !matched.is_empty() {
                let chain_text = format!(
                    "{} 的 {}",
                    entity,
                    intermediate
                        .iter()
                        .map(|predicate| predicate.surface.as_str())
                        .collect::<Vec<_>>()
                        .join(" 的 ")
                );
                let mut output = format!("是的。根据知识图谱：{} 是 {}。\n", chain_text, object);
                for witness in matched.iter().take(3) {
                    output.push_str(&format!(
                        "  路径（{} 跳）：{}\n",
                        witness.hop_count(),
                        witness.render()
                    ));
                }
                return Some(output.trim_end().to_string());
            }
            if bundle.witnesses.is_empty()
                && bundle.blocked.iter().any(|blocked| {
                    matches!(
                        blocked.reason,
                        crate::path_memory::BlockReason::NegatedEdge { .. }
                    )
                })
            {
                return Some(bundle.render_text());
            }
        }

        let mut chain_result = self.run_chain(&entity, &intermediate);
        let need_laws = self.laws.has_inference_laws()
            && chain_result.as_ref().map_or(true, |run| {
                !run.last_hits.iter().any(|hit| hit.value == object)
            });
        if need_laws {
            if let Some(expanded) = self.run_chain_via_laws(&entity, &intermediate) {
                chain_result = Some(expanded);
            }
        }
        if let Some(run) = chain_result {
            if let Some(hit) = run.last_hits.iter().find(|hit| hit.value == object) {
                let step = run.steps.last()?;
                return Some(match &hit.prediction {
                    None => match &hit.law {
                        Some(law_id) => format!(
                            "是的。{}",
                            self.laws.explain(law_id).unwrap_or_else(|| {
                                format!(
                                    "根据知识图谱：{} 的 {} 是 {}。",
                                    step.subject, step.predicate, object
                                )
                            })
                        ),
                        None => format!(
                            "是的。根据知识图谱：{} 的 {} 是 {}。",
                            step.subject, step.predicate, object
                        ),
                    },
                    Some(prediction) => format!(
                        "很可能是的（符号推断）。{}",
                        self.predictor.explain(prediction)
                    ),
                });
            }
            let known_values: Vec<String> = run
                .last_hits
                .iter()
                .filter(|hit| hit.prediction.is_none())
                .map(|hit| hit.value.clone())
                .collect();
            if !known_values.is_empty() {
                let step = run.steps.last()?;
                if run.closure_truncated {
                    return Some(format!(
                        "无法确定。传递闭包扩展达到上限，未列出通往 {} 的路径；目前记录的是：{} 的 {} 是 {}。",
                        object,
                        step.subject,
                        step.predicate,
                        known_values.join("、")
                    ));
                }
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    step.subject,
                    step.predicate,
                    known_values.join("、")
                ));
            }
        } else if intermediate.len() == 1 {
            let known_values: Vec<String> = if inverted {
                self.predictor.subjects_of(&slot, &entity)
            } else {
                self.predictor.objects_of(&entity, &slot)
            };
            if !known_values.is_empty() {
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    entity,
                    surface,
                    known_values.join("、")
                ));
            }
        }

        if self.is_known_entity(&entity) && self.is_known_predicate(&surface) {
            return Some(format!(
                "根据现有知识库，无法确认「{} 的 {} 是 {}」（没有相关记录）。",
                entity, surface, object
            ));
        }
        None
    }

    /// 属性倒排反查：给定属性槽位与取值，返回属性值等于它的实体。
    ///
    /// literal 属性（身高/年龄/国籍…）不会成为关系边，因此 `predictor`
    /// 里查不到，只能扫 `attribute_index`。槽位键先用对齐后的规范词面，
    /// 再回退英文属性键对照表（与 `node_hits` 的正向回退同一套规则）。
    fn attribute_subjects(&self, slot: &str, surface: &str, value: &str) -> Vec<String> {
        let value = value.trim();
        if slot.is_empty() || value.is_empty() {
            return Vec::new();
        }
        let mut keys = vec![slot.to_string()];
        if let Some(canonical) = crate::symbolic_generator::canonical_predicate_key(surface) {
            if canonical != slot {
                keys.push(canonical.to_string());
            }
        }
        let mut subjects: Vec<String> = self
            .attribute_index
            .iter()
            .filter(|((_, key), values)| {
                keys.iter().any(|candidate| candidate == key)
                    && values.iter().any(|stored| stored.trim() == value)
            })
            .map(|((entity, _), _)| entity.clone())
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    fn answer_relation_objects(&self, subject_raw: &str, predicate_raw: &str) -> Option<String> {
        self.answer_relation_objects_filtered(subject_raw, predicate_raw, &[])
    }

    fn answer_relation_objects_filtered(
        &self,
        subject_raw: &str,
        predicate_raw: &str,
        excluded: &[String],
    ) -> Option<String> {
        let entity = self.aligner.canonical_form(subject_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        let surface = predicate_raw.trim().to_string();
        if entity.is_empty() || slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) || !self.is_known_predicate(&surface) {
            return None;
        }
        let excluded_display: Vec<String> = excluded.to_vec();
        let excluded_set: HashSet<String> = excluded
            .iter()
            .map(|item| self.aligner.canonical_form(item.trim()))
            .collect();
        let predicate = ChainPredicate {
            surface: surface.clone(),
            slot,
            inverted,
        };
        let mut hits = self.node_hits(&entity, &predicate);
        hits.sort_by(|left, right| left.value.cmp(&right.value));
        hits.dedup_by(|left, right| left.value == right.value);
        let had_hits = !hits.is_empty();
        if !excluded_set.is_empty() {
            hits.retain(|hit| !excluded_set.contains(&hit.value));
        }
        if hits.is_empty() {
            if had_hits && !excluded_display.is_empty() {
                return Some(format!(
                    "根据现有知识库，除「{}」外，没有其他记录「{}{}…」。",
                    excluded_display.join("、"),
                    entity,
                    surface
                ));
            }
            return Some(format!(
                "根据现有知识库，没有记录「{}{}…」。",
                entity, surface
            ));
        }
        let values: Vec<String> = hits.iter().map(|hit| hit.value.clone()).collect();
        let mut output = if inverted {
            format!("根据知识图谱：「{}」的「{}」是：{}。", entity, surface, values.join("、"))
        } else if values.len() == 1 {
            format!("根据知识图谱：{}{}{}。", entity, surface, values[0])
        } else {
            format!(
                "根据知识图谱：{}{}的有：{}。",
                entity,
                surface,
                values.join("、")
            )
        };
        if !excluded_display.is_empty() {
            output.push_str(&format!("\n已排除：{}。", excluded_display.join("、")));
        }
        if let Some(prediction) = hits.iter().find_map(|hit| hit.prediction.as_ref()) {
            output.push_str(&format!(
                "\n其中为符号推断结果（例如置信度 {:.2}）：{}",
                prediction.confidence,
                self.predictor.explain(prediction)
            ));
        }
        Some(output)
    }

    fn answer_check_relation(
        &self,
        subject_raw: &str,
        predicate_raw: &str,
        object_raw: &str,
        negated: bool,
    ) -> Option<String> {
        let subject = self.aligner.canonical_form(subject_raw.trim());
        let object = self.aligner.canonical_form(object_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        let surface = predicate_raw.trim().to_string();
        if subject.is_empty() || object.is_empty() || slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&subject)
            || !self.is_known_entity(&object)
            || !self.is_known_predicate(&surface)
        {
            return None;
        }
        let (negative_subject, negative_object) = if inverted {
            (object.as_str(), subject.as_str())
        } else {
            (subject.as_str(), object.as_str())
        };
        if self.session_negative_blocks(negative_subject, &slot, negative_object) {
            return Some(if negated {
                format!(
                    "是的。会话中记录了否定事实：{} {} {}。",
                    subject, surface, object
                )
            } else {
                format!(
                    "不是。会话中记录了否定事实：{} {} {}。",
                    subject, surface, object
                )
            });
        }
        let holds = if inverted {
            self.predictor.has_triple(&object, &slot, &subject)
        } else {
            self.predictor.has_triple(&subject, &slot, &object)
        };
        if negated {
            return Some(if holds {
                format!(
                    "不是。根据知识图谱，该关系成立：{} {} {}。",
                    subject, surface, object
                )
            } else {
                format!(
                    "是的。根据现有知识图谱，没有记录「{} {} {}」，与否定问法一致。",
                    subject, surface, object
                )
            });
        }
        if holds {
            return Some(format!(
                "是的。根据知识图谱，该关系成立：{} {} {}。",
                subject, surface, object
            ));
        }
        Some(format!(
            "不是。根据现有知识图谱，没有记录「{} {} {}」。",
            subject, surface, object
        ))
    }

    fn answer_find_subject(&self, object_raw: &str, predicate_raw: &str) -> Option<String> {
        self.answer_find_subject_filtered(object_raw, predicate_raw, &[])
    }

    fn answer_find_subject_filtered(
        &self,
        object_raw: &str,
        predicate_raw: &str,
        excluded: &[String],
    ) -> Option<String> {
        let object = self.aligner.canonical_form(object_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        if object.is_empty() || slot.is_empty() {
            return None;
        }
        let surface = predicate_raw.trim().to_string();
        let excluded_display: Vec<String> = excluded.to_vec();
        let excluded_set: HashSet<String> = excluded
            .iter()
            .map(|item| self.aligner.canonical_form(item.trim()))
            .collect();
        // 反查主体：正向找槽位三元组的主语；逆向词面（如「徒弟」）语义即
        // "谁的槽位值是 object"，等价于找槽位三元组的宾语（object 作为主语）
        let mut subjects: Vec<String> = if inverted {
            self.predictor.objects_of(&object, &slot)
        } else {
            self.predictor.subjects_of(&slot, &object)
        };
        // 实体属性（身高/年龄/国籍…）不进关系三元组索引，literal 取值只能
        // 走属性倒排反查；没有这一步，「谁的身高是226」永远查不到人。
        if !inverted {
            subjects.extend(self.attribute_subjects(&slot, &surface, &object));
        }
        subjects.sort();
        subjects.dedup();
        let mut evidence: Vec<String> = Vec::new();
        if !inverted && self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            // 用哈希集合做成员检查
            let mut seen_subjects: HashSet<String> = subjects.iter().cloned().collect();
            for prediction in self
                .predictor
                .predict_for_entity(&object, Some(&slot), limit)
            {
                if prediction.triple.object == object
                    && prediction.triple.subject != object
                    && !excluded_set.contains(&prediction.triple.subject)
                {
                    if seen_subjects.insert(prediction.triple.subject.clone()) {
                        subjects.push(prediction.triple.subject.clone());
                    }
                    evidence.push(self.predictor.explain(&prediction));
                }
            }
        }
        if !excluded_set.is_empty() {
            subjects.retain(|subject| !excluded_set.contains(subject));
        }
        if subjects.is_empty() {
            if !excluded_display.is_empty()
                && self.is_known_entity(&object)
                && self.is_known_predicate(&surface)
            {
                return Some(format!(
                    "根据现有知识库，除「{}」外，没有其他记录「谁的 {} 是 {}」。",
                    excluded_display.join("、"),
                    surface,
                    object
                ));
            }
            if self.is_known_entity(&object) && self.is_known_predicate(&surface) {
                return Some(format!(
                    "根据现有知识库，没有记录「谁的 {} 是 {}」。",
                    surface, object
                ));
            }
            return None;
        }
        subjects.sort();
        let mut output = format!(
            "根据知识图谱：{} 的 {} 是 {}。",
            subjects.join("、"),
            surface,
            object
        );
        if !excluded_display.is_empty() {
            output.push_str(&format!("\n已排除：{}。", excluded_display.join("、")));
        }
        if !evidence.is_empty() {
            output.push_str(&format!("\n推理依据：{}", evidence.join("；")));
        }
        Some(output)
    }

    fn answer_similar(&self, entity_raw: &str) -> Option<String> {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() || !self.embedding.is_embedded(&entity) {
            return None;
        }
        let neighbors = self.embedding.nearest(&entity, 3);
        if neighbors.is_empty() {
            return None;
        }
        let mut output = format!(
            "符号嵌入近邻检索（PPMI 角色向量，无神经网络）：与「{}」语义最接近的实体：\n",
            entity
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn answer_analogy(&self, a_raw: &str, b_raw: &str, c_raw: &str) -> Option<String> {
        let a = self.aligner.canonical_form(a_raw.trim());
        let b = self.aligner.canonical_form(b_raw.trim());
        let c = self.aligner.canonical_form(c_raw.trim());
        if a.is_empty() || b.is_empty() || c.is_empty() {
            return None;
        }
        let results = self.embedding.analogy(&a, &b, &c, 3);
        if results.is_empty() {
            return None;
        }
        let direction = self.embedding.describe_direction(&a, &b);
        let mut output = format!(
            "向量算术类比（{} : {} :: {} : ?）：\n  从「{}」到「{}」的语义方向是{}，应用到「{}」：\n",
            a, b, c, a, b,
            if direction.is_empty() {
                String::new()
            } else {
                format!("「{}」", direction)
            },
            c
        );
        for (index, (name, score)) in results.iter().enumerate() {
            let shared = self.embedding.shared_feature_labels(&b, name, 2);
            let shared_text = if shared.is_empty() {
                String::new()
            } else {
                format!("，与「{}」共享 {}", b, shared.join("、"))
            };
            output.push_str(&format!(
                "  {}. 「{}」（语义偏移相似度 {:.2}{}）\n",
                index + 1,
                name,
                score,
                shared_text
            ));
        }
        Some(output.trim_end().to_string())
    }

    pub fn answer(&self, question: &str) -> Option<String> {
        let answer = self.answer_impl(question);
        if answer.is_none() {
            // 只观测失败：成功轮不做额外分词，回答内容与原有路径逐字节一致。
            self.record_read_failure(question);
        }
        answer
    }

    /// 句法重写回执：展示「哪种句法外壳被压成哪个逻辑计划」。
    /// 只读；未命中或词表校验失败时返回 None。
    pub fn query_rewrite_receipt(&self, question: &str) -> Option<String> {
        query_rewrite::rewrite(question, &self.relation_lexicon())
            .map(|rewrite| rewrite.receipt())
    }

    /// 为问句重建有界推导视图（与回答共用同一份事实、规则与归结搜索）。
    ///
    /// 这是惰性通道：只在用户追问依据时调用，不进入回答热路径，也不写入
    /// 会话快照；没有可形式化的证明时返回 None，由调用方诚实降级。
    pub fn answer_derivation(
        &self,
        question: &str,
        expected_answer: Option<&str>,
    ) -> Option<Derivation> {
        let view = self.resolution_proof_view(question)?;
        Derivation::from_resolution_view(&view, expected_answer)
    }

    /// 无观测副作用的回答实现（报告路径复用，避免失败轨迹递归）。
    fn answer_impl(&self, question: &str) -> Option<String> {
        if let Some(answer) = self.answer_planned(question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_by_prediction(question) {
            return Some(answer);
        }
        self.answer_by_evidence(question)
    }

    /// 最后一道接地回答：计划与链接预测都无结论时，用统一证据索引取回
    /// 「已知实体 + 被提及谓词」的事实，并附检索回执。只引用索引中的真实
    /// 三元组，不生成无出处内容；没有同时满足锚定与谓词条件的证据就弃权。
    fn answer_by_evidence(&self, question: &str) -> Option<String> {
        if !self.config.enable_generation || self.retrieval.lock().is_empty() {
            return None;
        }
        let entities = self.entities_mentioned(question);
        if entities.is_empty() {
            return None;
        }
        let normalized_question = self.aligner.normalize_cow(question);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(question);
        let mentions_predicate = !mentioned_canonicals.is_empty()
            || self
                .known_predicates_normalized_cache
                .iter()
                .any(|(_, predicate_norm)| {
                    !predicate_norm.is_empty()
                        && normalized_question.contains(predicate_norm.as_str())
                });
        if !mentions_predicate {
            return None;
        }

        let search = self.evidence_search(
            question,
            self.config.max_predictions.clamp(1, 5),
            &SearchConstraints::default(),
            SearchBudget::default(),
        );
        let anchors: HashSet<&str> = entities.iter().map(String::as_str).collect();
        let hits: Vec<_> = search
            .hits
            .iter()
            .filter(|hit| {
                let anchored =
                    anchors.contains(hit.subject.as_str()) || anchors.contains(hit.object.as_str());
                if !anchored || hit.score < 0.2 {
                    return false;
                }
                let canonical = self.aligner.canonical_form(&hit.predicate);
                let predicate_norm = self.aligner.normalize_cow(&hit.predicate);
                mentioned_canonicals.contains(&hit.predicate)
                    || mentioned_canonicals.contains(&canonical)
                    || (!predicate_norm.is_empty()
                        && normalized_question.contains(predicate_norm.as_ref() as &str))
            })
            .take(3)
            .collect();
        if hits.is_empty() {
            return None;
        }

        let mut output = String::from("根据证据索引（带检索回执的符号检索）：\n");
        for (index, hit) in hits.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}（{}）\n",
                index + 1,
                hit.text(),
                hit.breakdown()
            ));
        }
        output.push_str(&format!(
            "  检索回执：候选 {}，打分 {}，上界剪枝 {}；{}。",
            search.receipt.candidates,
            search.receipt.scanned,
            search.receipt.bound_pruned,
            if search.receipt.certified_top_k {
                "top-k 已证明"
            } else {
                "未证明 top-k（结果按当前预算返回）"
            }
        ));
        Some(output.trim_end().to_string())
    }

    fn answer_by_prediction(&self, question: &str) -> Option<String> {
        if !self.config.enable_link_prediction || self.predictor.rules().is_empty() {
            return None;
        }
        let mut entities = self.entities_mentioned(question);
        if entities.is_empty() && self.config.enable_generation {
            // 精确匹配失败时，退回到带容错的实体链接结果（别名词表之外也能命中）
            entities = self.linked_entities(question);
        }
        if entities.is_empty() {
            return None;
        }

        let normalized_question = self.aligner.normalize_cow(question);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(question);
        let mut candidates: Vec<Prediction> = Vec::new();
        for entity in &entities {
            for prediction in
                self.predictor
                    .predict_for_entity(entity, None, self.config.max_predictions * 4)
            {
                if self.predicate_mentioned_normalized_with(
                    normalized_question.as_ref(),
                    &prediction.triple.predicate,
                    &mentioned_canonicals,
                ) {
                    candidates.push(prediction);
                }
            }
        }
        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.triple.subject.cmp(&b.triple.subject))
                .then_with(|| a.triple.object.cmp(&b.triple.object))
        });
        let mut seen: HashSet<Triple> = HashSet::new();
        candidates.retain(|p| seen.insert(p.triple.clone()));

        let limit = self.config.max_predictions.clamp(1, 5);
        let mut output = String::from("根据知识图谱的链接预测（符号推理）：\n");
        for prediction in candidates.iter().take(limit) {
            output.push_str("  • ");
            output.push_str(&self.predictor.explain(prediction));
            output.push('\n');
        }
        Some(output)
    }

    pub fn similarity_report(&self, entity_raw: &str) -> String {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() {
            return "请输入要检索的实体名，例如：相似实体：张三".to_string();
        }
        if !self.embedding.is_embedded(&entity) {
            return format!(
                "语义嵌入中没有「{}」的角色特征（可能是孤立节点或尚未导入关系数据）。",
                entity
            );
        }
        let neighbors = self.embedding.nearest(&entity, 5);
        if neighbors.is_empty() {
            return format!(
                "没有与「{}」相似度达到阈值 {:.2} 的实体。",
                entity,
                self.embedding.similarity_floor()
            );
        }
        let mut output = format!(
            "符号嵌入近邻（PPMI 角色向量；已嵌入 {} 个实体、{} 个特征）：\n",
            self.embedding.entity_count(),
            self.embedding.feature_count()
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        output.trim_end().to_string()
    }

    /// 结构联想：RWR 图嵌入检索 + 结构角色近邻 + 关联路径解释
    pub fn structural_related_report(&self, entity_raw: &str) -> String {
        if !self.config.enable_graph_walk {
            return "结构联想（图游走）未启用（config.toml [reasoning] enable_graph_walk = false）。"
                .to_string();
        }
        let raw = entity_raw.trim();
        if raw.is_empty() {
            return "请输入要联想的实体，例如：结构联想：张三".to_string();
        }
        let entity = self.aligner.canonical_form(raw);
        if !self.walks.is_embedded(&entity) {
            if let Some(linked) = self.best_linked_entity(raw) {
                if self.walks.is_embedded(&linked) {
                    return self.walks.report(&linked, 5);
                }
            }
            return format!(
                "结构联想中没有「{}」：它没有出现在任何关系三元组中。",
                entity
            );
        }
        self.walks.report(&entity, 5)
    }

    /// 话题延伸：以对话焦点实体栈为多源种子做重启随机游走（上下文敏感的联想）
    pub fn topic_extension_report(&self) -> String {
        if !self.config.enable_graph_walk {
            return "话题延伸（图游走）未启用（config.toml [reasoning] enable_graph_walk = false）。"
                .to_string();
        }
        let seeds: Vec<String> = self.context.focus_entities().to_vec();
        if seeds.is_empty() {
            return "还没有对话焦点：先聊到一个实体，再输入「话题延伸」。".to_string();
        }
        let related = self.walks.context_related(&seeds, 6);
        if related.is_empty() {
            return format!(
                "焦点实体「{}」不在关系图中（或无相邻实体），暂时无法延伸话题。",
                seeds.join("、")
            );
        }
        let mut output = format!(
            "话题延伸（以对话焦点 [{}] 为多源种子做 RWR 图游走；图规模 {} 实体 / {} 边）：\n",
            seeds.join(" → "),
            self.walks.entity_count(),
            self.walks.edge_count()
        );
        for (index, (name, weight)) in related.iter().enumerate() {
            let mut line = format!("  {}. {}（联合驻留概率 {:.3}）", index + 1, name, weight);
            if let Some((seed, path)) = seeds.iter().find_map(|seed| {
                self.walks
                    .best_path(seed, name, 4)
                    .map(|path| (seed.clone(), path))
            }) {
                line.push_str(&format!(
                    "；路径：{}",
                    GraphWalkEmbedding::render_path(&seed, &path)
                ));
            }
            output.push_str(&line);
            output.push('\n');
        }
        if let Some(focus) = seeds.first() {
            let similar = self.walks.nearest(focus, 3);
            if !similar.is_empty() {
                output.push_str(&format!("\n  与「{}」结构角色相近：", focus));
                output.push_str(
                    &similar
                        .iter()
                        .map(|(name, score)| format!("{}({:.2})", name, score))
                        .collect::<Vec<_>>()
                        .join("、"),
                );
            }
        }
        output
    }

    /// 全局软推理报告：加权规则推出的候选事实（按概率排序）
    pub fn soft_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_soft_logic {
            return "加权软逻辑未启用（config.toml [reasoning] enable_soft_logic = false）。"
                .to_string();
        }
        let limit = limit.clamp(1, 20);
        let all: Vec<SoftFact> = self.soft.all_predictions(limit * 4);
        // 优先展示"推出的新事实"；若没有（规则不足），退化为展示已知事实
        let inferred: Vec<SoftFact> = all.iter().filter(|fact| !fact.from_fact).cloned().collect();
        let showing_inferred = !inferred.is_empty();
        let mut predictions = if showing_inferred { inferred } else { all };
        predictions.truncate(limit);
        let mut output = format!(
            "加权软推理（MLN-lite：log-odds 累加 + sigmoid 归一；事实 {} 条，加权规则 {} 条，阈值 P≥{:.2}）：\n",
            self.soft.fact_count(),
            self.soft.rule_count(),
            self.soft.min_probability()
        );
        if predictions.is_empty() {
            output.push_str("  （暂无可推出的软事实：可增加事实/规则或降低阈值）\n");
            return output.trim_end().to_string();
        }
        if !showing_inferred {
            output.push_str("  （暂无可加权推出的新事实，以下为已确认事实）\n");
        }
        for (index, fact) in predictions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.render_soft_fact(fact)
            ));
        }
        output.trim_end().to_string()
    }

    /// 针对某个实体（可带「的谓词」）的软推理报告
    pub fn soft_inference_report(&self, target_raw: &str) -> String {
        if !self.config.enable_soft_logic {
            return "加权软逻辑未启用（config.toml [reasoning] enable_soft_logic = false）。"
                .to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要软推理的对象，例如：软推理：张三 或 软推理：张三的师傅".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let facts: Vec<SoftFact> = match predicate {
            Some(predicate) if !predicate.is_empty() => {
                let forward = self.soft.infer(
                    &Atom::pattern(&predicate, Term::constant(&entity), Term::var("X")),
                    8,
                );
                if !forward.is_empty() {
                    forward
                } else {
                    self.soft.infer(
                        &Atom::pattern(&predicate, Term::var("X"), Term::constant(&entity)),
                        8,
                    )
                }
            }
            _ => self.collect_soft_facts_for_entity(&entity, 8),
        };
        if facts.is_empty() {
            return format!(
                "软推理没有找到关于「{}」达到阈值（P≥{:.2}）的候选：可增加同类事实/规则，或降低阈值。",
                entity,
                self.soft.min_probability()
            );
        }
        let mut output = format!(
            "加权软推理（关于「{}」；事实 {} 条，加权规则 {} 条，无神经网络）：\n",
            entity,
            self.soft.fact_count(),
            self.soft.rule_count()
        );
        for (index, fact) in facts.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.render_soft_fact(fact)
            ));
            if let Some(evidence) = fact.evidence.first() {
                output.push_str(&format!(
                    "     依据：{}（权重 {:+.2}）\n",
                    evidence.source, evidence.weight
                ));
                if !evidence.premises.is_empty() {
                    let premises = evidence
                        .premises
                        .iter()
                        .map(|premise| self.render_atom_nl(premise))
                        .collect::<Vec<_>>()
                        .join(" 且 ");
                    output.push_str(&format!("     前提：{}\n", premises));
                }
            }
        }
        output.trim_end().to_string()
    }

    fn collect_soft_facts_for_entity(&self, entity: &str, limit: usize) -> Vec<SoftFact> {
        let mut merged: HashMap<String, SoftFact> = HashMap::new();
        for atom in self.soft.facts() {
            if atom.predicate.starts_with("__") {
                continue;
            }
            if atom.terms.iter().any(|term| term.name() == entity) {
                merged.insert(
                    atom.display(),
                    SoftFact {
                        atom: atom.clone(),
                        probability: 1.0,
                        log_odds: 6.0,
                        evidence: Vec::new(),
                        from_fact: true,
                    },
                );
            }
        }
        for fact in self.soft.all_predictions(256) {
            if fact.from_fact && fact.atom.predicate.starts_with("__") {
                continue;
            }
            if !fact.atom.terms.iter().any(|term| term.name() == entity) {
                continue;
            }
            merged
                .entry(fact.atom.display())
                .and_modify(|existing| {
                    if fact.probability > existing.probability {
                        *existing = fact.clone();
                    }
                })
                .or_insert(fact);
        }
        let mut results: Vec<SoftFact> = merged.into_values().collect();
        results.sort_by_cached_key(|item| ProbabilityDisplayKey {
            probability: item.probability,
            display: item.atom.display(),
        });
        results.truncate(limit.max(1));
        results
    }

    fn render_soft_fact(&self, fact: &SoftFact) -> String {
        let mut text = format!(
            "{} {}（P={:.2}",
            self.render_atom_nl(&fact.atom),
            MarkovLogicEngine::hedge(fact.probability),
            fact.probability
        );
        if fact.from_fact {
            text.push_str("，图谱事实");
        }
        if let Some(evidence) = fact.evidence.first() {
            text.push_str(&format!("；依据：{}", evidence.source));
        }
        text.push('）');
        text
    }

    /// 收集可用于因果解释的目标（优先非事实的加权推测，其次图谱事实）
    fn causal_targets(&self, entity: &str, predicate: Option<&str>, limit: usize) -> Vec<SoftFact> {
        let predicate_matches = |atom: &Atom| -> bool {
            match predicate {
                Some(pred) if !pred.is_empty() => atom.predicate.as_ref() == pred,
                _ => !atom.predicate.starts_with("__"),
            }
        };
        let mentions_entity = |atom: &Atom| atom.terms.iter().any(|term| term.name() == entity);
        let mut merged: HashMap<String, SoftFact> = HashMap::new();
        for atom in self.soft.facts() {
            if !mentions_entity(atom) || !predicate_matches(atom) {
                continue;
            }
            merged.insert(
                atom.display(),
                SoftFact {
                    atom: atom.clone(),
                    probability: 1.0,
                    log_odds: 6.0,
                    evidence: Vec::new(),
                    from_fact: true,
                },
            );
        }
        for fact in self.soft.all_predictions(256) {
            if !mentions_entity(&fact.atom) || !predicate_matches(&fact.atom) {
                continue;
            }
            merged
                .entry(fact.atom.display())
                .and_modify(|existing| {
                    if fact.probability > existing.probability {
                        *existing = fact.clone();
                    }
                })
                .or_insert(fact);
        }
        let mut results: Vec<SoftFact> = merged.into_values().collect();
        results.sort_by_cached_key(|item| FromFactProbabilityKey {
            from_fact: item.from_fact,
            probability: item.probability,
            display: item.atom.display(),
        });
        results.truncate(limit.max(1));
        results
    }

    /// 找出某个「实体[的谓词]」对应的可干预原子（优先图谱事实，其次加权推测；
    /// 实体作主语优先——"李四的师傅"应干预 师傅(李四, ?)，结果确定可复现）
    fn find_causal_atom(&self, entity: &str, predicate: Option<&str>) -> Option<Atom> {
        let predicate_matches = |atom: &Atom| -> bool {
            match predicate {
                Some(pred) if !pred.is_empty() => atom.predicate.as_ref() == pred,
                _ => !atom.predicate.starts_with("__"),
            }
        };
        let subject_matches = |atom: &Atom| -> bool {
            atom.terms
                .first()
                .map(|term| term.name() == entity)
                .unwrap_or(false)
                && predicate_matches(atom)
        };
        let any_matches = |atom: &Atom| -> bool {
            atom.terms.iter().any(|term| term.name() == entity) && predicate_matches(atom)
        };
        if let Some(atom) = self
            .soft
            .facts()
            .find(|atom| subject_matches(atom))
            .or_else(|| self.soft.facts().find(|atom| any_matches(atom)))
        {
            return Some(atom.clone());
        }
        // 软预测最多计算一次，两次查找共用同一结果
        let predictions = self.soft.all_predictions(256);
        predictions
            .iter()
            .find(|fact| subject_matches(&fact.atom))
            .or_else(|| predictions.iter().find(|fact| any_matches(&fact.atom)))
            .map(|fact| fact.atom.clone())
    }

    /// 因果推理报告：因果图 + 直接原因 + 事实根因链 + 混杂因子（纯符号 do-演算视角）
    pub fn causal_report(&self, target_raw: &str) -> String {
        if !self.config.enable_causal_reasoning {
            return "因果推理未启用（config.toml [reasoning] enable_causal_reasoning = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "因果推理依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要因果解释的对象，例如：因果：李四的师祖 或 因果：张三".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let candidates = self.causal_targets(&entity, predicate.as_deref(), 3);
        if candidates.is_empty() {
            return format!("没有找到关于「{}」可用于因果解释的事实或加权推测。", entity);
        }
        // 概率视图与因果图供所有候选目标共用
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let graph = analyzer.graph();
        let mut output = format!(
            "因果推理（因果图 {} 节点 / {} 条有符号机制；正权促进、负权抑制；规则基化自 {} 条事实）：\n",
            graph.node_count(),
            graph.edge_count(),
            self.soft.fact_count()
        );
        for (index, fact) in candidates.iter().enumerate() {
            let explanation =
                analyzer.explain_with_probabilities(&self.soft, &probabilities, &fact.atom, 3);
            output.push_str(&format!(
                "  {}. {} {}（P={:.2}）\n",
                index + 1,
                self.render_atom_nl(&explanation.target),
                if explanation.probability >= 0.999 {
                    "是图谱事实"
                } else {
                    MarkovLogicEngine::hedge(explanation.probability)
                },
                explanation.probability
            ));
            if explanation.direct_causes.is_empty() {
                output.push_str("     直接原因：（没有规则指向它，可能是一手事实）\n");
            } else {
                output.push_str("     直接原因：\n");
                for link in &explanation.direct_causes {
                    output.push_str(&format!(
                        "       • {}（P={:.2}）—[{:+.2}{}]→ 本结论；依据：{}\n",
                        self.render_atom_nl(&link.cause),
                        link.cause_probability,
                        link.weight,
                        if link.is_inhibitory() {
                            "，抑制"
                        } else {
                            ""
                        },
                        link.source
                    ));
                }
            }
            if let Some(chain) = explanation.chains.first() {
                output.push_str(&format!("     事实根因链：{}\n", chain.render()));
            }
            if !explanation.confounders.is_empty() {
                let names = explanation
                    .confounders
                    .iter()
                    .map(|atom| self.render_atom_nl(atom))
                    .collect::<Vec<_>>()
                    .join("、");
                output.push_str(&format!(
                    "     注意混杂因子（同为原因，相关≠因果）：{}\n",
                    names
                ));
            }
        }
        output.trim_end().to_string()
    }

    /// 因果反事实报告：do(干预原子=真/假) 后，下游结论在"事实世界"与"干预世界"的概率差
    pub fn counterfactual_report(&self, target_raw: &str, value: bool) -> String {
        if !self.config.enable_causal_reasoning {
            return "因果推理未启用（config.toml [reasoning] enable_causal_reasoning = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "反事实推理依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要干预的对象，例如：干预：张三的师傅 或 干预：张三的师傅=真".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let Some(intervention) = self.find_causal_atom(&entity, predicate.as_deref()) else {
            return format!(
                "没有找到关于「{}」的可干预原子（需要先有该事实或能推出它的规则）。",
                entity
            );
        };
        // 因果图基化使用同一份概率视图
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let probes = analyzer.direct_effect_probes(&intervention);
        if probes.is_empty() {
            return format!(
                "干预「{}」在当前因果图中没有直接下游（没有规则以它为前提），因此反事实世界与事实世界相同。",
                intervention.display()
            );
        }
        let mut deltas: Vec<CounterfactualDelta> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        // 事实世界与干预世界共用软真值传播结果
        let mut fixed = HashMap::new();
        fixed.insert(intervention.clone(), if value { 1.0 } else { 0.0 });
        let factual_beliefs = self.soft.propagate_with_fixed(&HashMap::new());
        let counterfactual_beliefs = self.soft.propagate_with_fixed(&fixed);
        for probe in probes.iter().take(6) {
            for delta in analyzer
                .counterfactual_from_beliefs(
                    &self.soft,
                    &factual_beliefs,
                    &counterfactual_beliefs,
                    &fixed,
                    probe,
                    &intervention,
                    value,
                    4,
                )
                .deltas
            {
                if seen.insert(delta.atom.display()) {
                    deltas.push(delta);
                }
            }
        }
        deltas.sort_by_cached_key(|item| MagnitudeDisplayKey {
            magnitude: item.delta.abs(),
            display: item.atom.display(),
        });
        deltas.truncate(8);
        let mut output = format!(
            "反事实推理（do：干预「{}」={}；符号实现 = 钉住真值 + 切断指向它的规则，其余规则重新软传播）：\n",
            self.render_atom_nl(&intervention),
            if value { "真" } else { "假" }
        );
        output.push_str("  事实世界 → 干预世界（Δ=干预后−事实）：\n");
        for (index, delta) in deltas.iter().enumerate() {
            let direction = if delta.delta > 1e-9 {
                "上升"
            } else if delta.delta < -1e-9 {
                "下降"
            } else {
                "不变"
            };
            let factual_hedge = if delta.factual <= 1e-9 {
                "未推出"
            } else {
                MarkovLogicEngine::hedge(delta.factual)
            };
            let counterfactual_hedge = if delta.counterfactual <= 1e-9 {
                "未推出"
            } else {
                MarkovLogicEngine::hedge(delta.counterfactual)
            };
            output.push_str(&format!(
                "  {}. {}：P {:.2}（{}）→ {:.2}（{}），Δ={:+.2}（{}）\n",
                index + 1,
                self.render_atom_nl(&delta.atom),
                delta.factual,
                factual_hedge,
                delta.counterfactual,
                counterfactual_hedge,
                delta.delta,
                direction
            ));
        }
        output.trim_end().to_string()
    }

    /// 符号梯度归因报告（"可微推理"的纯符号实现：解析求导，只解释与排序求证，不训练）
    pub fn gradient_report(&self, target_raw: &str) -> String {
        if !self.config.enable_gradient_attribution {
            return "符号梯度归因未启用（config.toml [reasoning] enable_gradient_attribution = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "符号梯度归因依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要归因的对象，例如：梯度归因：李四的师祖 或 梯度归因：张三".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let pattern = match predicate.as_deref() {
            Some(pred) if !pred.is_empty() => {
                Atom::pattern(pred, Term::constant(&entity), Term::var("X"))
            }
            _ => {
                let Some(candidate) = self
                    .collect_soft_facts_for_entity(&entity, 8)
                    .into_iter()
                    .find(|fact| !fact.from_fact)
                else {
                    return format!(
                        "「{}」目前只有图谱事实、没有加权推测，无需梯度归因。",
                        entity
                    );
                };
                candidate.atom
            }
        };
        let mut attributions = self.soft.gradient(&pattern, 3);
        if attributions.is_empty() {
            if let Some(pred) = predicate.as_deref().filter(|pred| !pred.is_empty()) {
                attributions = self.soft.gradient(
                    &Atom::pattern(pred, Term::var("X"), Term::constant(&entity)),
                    3,
                );
            }
        }
        if attributions.is_empty() {
            return format!(
                "没有可做梯度归因的加权推测（「{}」未被任何规则支持）。",
                entity
            );
        }
        let mut output = format!(
            "符号梯度归因（可微推理的纯符号版：∂P/∂w = P(1−P)·前件满足度；解析求导、不训练、无神经网络）\n"
        );
        for (index, attribution) in attributions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} {}（P={:.3}，log-odds {:+.3}）\n",
                index + 1,
                self.render_atom_nl(&attribution.atom),
                MarkovLogicEngine::hedge(attribution.probability),
                attribution.probability,
                attribution.log_odds
            ));
            for contribution in attribution.contributions.iter().take(4) {
                let premises = if contribution.premises.is_empty() {
                    "（无前件）".to_string()
                } else {
                    contribution
                        .premises
                        .iter()
                        .map(|premise| self.render_atom_nl(premise))
                        .collect::<Vec<_>>()
                        .join(" 且 ")
                };
                output.push_str(&format!(
                    "     规则贡献 ∂P/∂w={:+.4}（权重 {:+.2}，前件满足度 {:.2}）：{}\n       来源：{}\n",
                    contribution.gradient,
                    contribution.weight,
                    contribution.satisfaction,
                    premises,
                    contribution.source
                ));
            }
            let uncertain: Vec<_> = attribution
                .premise_sensitivity
                .iter()
                .filter(|sensitivity| sensitivity.truth < 0.999)
                .take(3)
                .collect();
            if !uncertain.is_empty() {
                output.push_str("     最值得求证的前提（按 |∂P/∂s|·(1−真值) 排序）：\n");
                for sensitivity in uncertain {
                    output.push_str(&format!(
                        "       • {}（当前软真值 {:.2}，∂P/∂s={:+.3}，求证优先级 {:.3}）\n",
                        self.render_atom_nl(&sensitivity.premise),
                        sensitivity.truth,
                        sensitivity.gradient,
                        sensitivity.verify_priority
                    ));
                }
            }
        }
        output.trim_end().to_string()
    }

    /// 从本体对齐器、生成器、会话事实与因果图构建因果分词器：
    /// 词表锚（实体/类型/谓词/同义词）+ 每条因果机制的权重质量
    fn build_causal_tokenizer(&self, graph: &CausalGraph) -> CausalTokenizer {
        let mut tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        // 本体对齐器词条：实体/类型/关系/属性
        let mut profiles: Vec<(&str, AnchorKind, String)> = self
            .aligner
            .profiles()
            .iter()
            .map(|profile| {
                let kind = match profile.kind {
                    TermKind::Entity => AnchorKind::Entity,
                    TermKind::Type => AnchorKind::Type,
                    TermKind::Relation | TermKind::Attribute => AnchorKind::Predicate,
                };
                (
                    profile.term.as_str(),
                    kind,
                    self.aligner.canonical_form(&profile.term),
                )
            })
            .collect();
        profiles.sort_by(|a, b| a.0.cmp(b.0));
        for (term, kind, canonical) in profiles {
            tokenizer.add_anchor(term, &canonical, kind, 0.95);
            if canonical != term {
                tokenizer.add_anchor(&canonical, &canonical, kind, 1.0);
            }
        }
        // 同义词表：别名 → 规范词
        let mut aliases: Vec<(&String, &String)> = self.aligner.synonyms().iter().collect();
        aliases.sort();
        for (alias, canonical) in aliases {
            tokenizer.add_alias(alias, canonical);
        }
        // 生成器的谓词表面形式（动词化表述也参与锚定）
        let mut surfaces: Vec<(&String, &String)> =
            self.generator.predicate_surfaces().iter().collect();
        surfaces.sort();
        for (surface, canonical) in surfaces {
            tokenizer.add_anchor(surface, canonical, AnchorKind::Predicate, 0.7);
        }
        // 图谱事实：孤立事实同样具有词表地位
        for atom in self.soft.facts() {
            tokenizer.observe_fact(atom);
        }
        // 因果图：每条有符号机制为两侧原子累计因果质量
        for edge in graph.edges() {
            tokenizer.observe_causal_edge(edge);
        }
        tokenizer
    }

    /// 因果分词报告：知识/因果图引导的 token 格 k-best DP 分词 + token 级因果归因
    /// （纯符号 attention：结论的概率质量沿因果链回溯摊到输入 token 上）
    pub fn causal_tokenize_report(&self, input_raw: &str) -> String {
        if !self.config.enable_causal_tokenization {
            return "因果分词未启用（config.toml [reasoning] enable_causal_tokenization = false）。"
                .to_string();
        }
        let text = input_raw.trim();
        if text.is_empty() {
            return "请输入要分词的文本，例如：因果分词：张三的师父是谁".to_string();
        }
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let graph = analyzer.graph();
        let tokenizer = self.build_causal_tokenizer(graph);
        let tokenization = tokenizer.tokenize(text);
        let mut output = format!(
            "因果分词（知识/因果图引导的 token 格 k-best DP：锚定支持度 + 因果质量加成；词表 {} 个锚、因果图 {} 条机制；无神经网络）\n输入：{}\n最优分词（得分 {:.2}）：{}\n",
            tokenizer.anchor_count(),
            graph.edge_count(),
            text,
            tokenization.score,
            tokenization.render_segmentation()
        );
        for token in &tokenization.tokens {
            match &token.anchor {
                Some(anchor) => output.push_str(&format!(
                    "   • {} → {}（锚定 {:.2}，机制 {} 条 / 因果质量 {:.2}）\n",
                    token.text,
                    anchor.text(),
                    anchor.support,
                    anchor.causal_edges,
                    anchor.causal_mass
                )),
                None => output.push_str(&format!("   • {} → 未锚定（回退）\n", token.text)),
            }
        }
        if !tokenization.alternatives.is_empty() {
            let rendered: Vec<String> = tokenization
                .alternatives
                .iter()
                .take(2)
                .map(|alt| format!("{}（{:.2}）", alt.tokens.join(" | "), alt.score))
                .collect();
            output.push_str(&format!("备选分词：{}\n", rendered.join("；")));
        }
        // 目标原子：优先按「实体[的谓词]」解析，退而取第一个实体锚
        let target = self
            .split_entity_predicate(text)
            .and_then(|(entity, predicate)| self.find_causal_atom(&entity, predicate.as_deref()))
            .or_else(|| {
                tokenization
                    .anchored_tokens()
                    .filter(|token| {
                        token
                            .anchor
                            .as_ref()
                            .map(|anchor| anchor.kind == AnchorKind::Entity)
                            .unwrap_or(false)
                    })
                    .map(|token| {
                        token
                            .anchor
                            .as_ref()
                            .expect("已按锚定过滤")
                            .canonical
                            .clone()
                    })
                    .next()
                    .and_then(|entity| self.find_causal_atom(&entity, None))
            });
        match target {
            Some(target) => {
                let attributions = attribute_tokens(&tokenization, &self.soft, graph, &target);
                output.push_str(&format!(
                    "token 因果归因（目标：{}）：\n",
                    self.render_atom_nl(&target)
                ));
                if attributions.is_empty() {
                    output.push_str(
                        "   • 目标未落在输入 token 的锚定路径上（结论来自输入之外的图谱事实）。\n",
                    );
                } else {
                    for item in &attributions {
                        output.push_str(&format!(
                            "   • 「{}」 +{:.3}（{:>5.1}%）锚：{}\n",
                            item.token,
                            item.contribution,
                            item.share * 100.0,
                            item.anchor
                        ));
                    }
                    output.push_str(
                        "归因说明：概率质量沿因果链回溯到锚定 token，纯符号 attention、逐行可回放。\n",
                    );
                }
            }
            None => {
                output.push_str("（未在知识库中找到与输入对应的可归因目标）\n");
            }
        }
        output.trim_end().to_string()
    }

    /// 中文概率知识教学：把「如果…那么…」「八成…」等句子解析成 ProbLog 概率知识，
    /// 只写入会话级概率程序（不修改知识图谱），可随对话状态持久化并在下次会话重放
    pub fn teach_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_problog {
            return "概率逻辑教学未启用（config.toml [reasoning] enable_problog = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "概率逻辑教学依赖加权软逻辑（事实与规则同源），请先启用 enable_soft_logic。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请用中文教一条概率知识，例如：教学：如果张三的师傅是李四，那么张三的师祖是王五。"
                .to_string();
        }
        if self.lessons.iter().any(|existing| existing == sentence) {
            return format!(
                "这条知识已经学过（重复教学不会重复计概率）：{}。可用「概率推理：实体」查看推断结果。",
                sentence
            );
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        if outcome.lessons.is_empty() {
            let mut output = String::from("没能把这句话解析成概率知识。支持句式：\n");
            output.push_str(
                "  • 条件句：如果S的P是O，那么S的P是O（可带 八成/很可能/百分之八十 等概率词）\n",
            );
            output.push_str("  • 因果句：因为S的P是O，所以S的P是O\n");
            output.push_str("  • 比例句：八成的S会V / S的P是O的可能性是百分之八十\n");
            output.push_str("  • 默认句（可反驳常识）：如果X是鸟，那么X会飞，除非X是企鹅\n");
            output.push_str("  • 直陈句：S的P是O（作为确定事实）");
            if !outcome.errors.is_empty() {
                output.push_str("\n  失败原因：\n");
                for error in &outcome.errors {
                    output.push_str(&format!("   • {}\n", error));
                }
            }
            return output.trim_end().to_string();
        }
        let mut output = format!(
            "已学会 {} 条中文概率知识（纯符号解析 → ProbLog 程序，可回放、不训练）：\n",
            outcome.lessons.len()
        );
        for (index, lesson) in outcome.lessons.iter().enumerate() {
            chinese_reasoning_nlu::apply_lesson(&mut self.problog, lesson);
            output.push_str(&format!(
                "  {}. 原始：{}\n     逻辑形式：{}\n     知识类型：{}；规则来源：{}\n",
                index + 1,
                lesson.original,
                chinese_reasoning_nlu::render_lesson_zh(lesson),
                lesson_kind_label(lesson.kind),
                lesson.source
            ));
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未学会：{}\n", error));
        }
        // 条件/因果知识同时登记为会话 Horn 子句：后向链（判断/证明/为什么/推理链）直接可用
        let added_clauses = self.register_lesson_clauses(&outcome.lessons);
        if added_clauses > 0 {
            output.push_str(&format!(
                "  其中 {} 条规则已进入会话后向链规则库（含占位符的自动变量化，可用「判断/为什么/推理链」验证）。\n",
                added_clauses
            ));
        }
        self.lessons.push(sentence.to_string());
        // 与上面 apply_lesson 的重放保持一致
        self.lessons_parsed.extend(outcome.lessons.iter().cloned());
        output.push_str(&format!(
            "  概率程序现有 {} 条确定事实、{} 条概率事实、{} 条概率规则；用「概率推理：实体」做可能世界精确查询，用「对话记忆」查看教学记忆。",
            self.problog.deterministic_fact_count(),
            self.problog.probabilistic_fact_count(),
            self.problog.rule_count()
        ));
        output
    }

    /// 把教过的条件/因果知识登记为会话 Horn 子句（含占位符的自动变量化），
    /// 重建归结算子库；返回新增子句条数
    fn register_lesson_clauses(
        &mut self,
        lessons: &[chinese_reasoning_nlu::ChineseLesson],
    ) -> usize {
        if !self.config.enable_resolution {
            return 0;
        }
        let mut added = 0;
        let mut new_predicates: Vec<String> = Vec::new();
        for lesson in lessons {
            let Some(clause) = chinese_reasoning_nlu::lesson_to_clause(lesson) else {
                continue;
            };
            let duplicate = self
                .session_clauses
                .iter()
                .any(|existing| existing.head == clause.head && existing.body == clause.body);
            if duplicate {
                continue;
            }
            for atom in std::iter::once(&clause.head).chain(clause.body.iter()) {
                if !self.aligner.is_registered(&atom.predicate)
                    && !new_predicates
                        .iter()
                        .any(|p| p.as_str() == atom.predicate.as_ref())
                {
                    new_predicates.push(atom.predicate.to_string());
                }
            }
            self.session_clauses.push(clause);
            added += 1;
        }
        if added > 0 {
            // 新谓词登记为关系：命题解析/意图规划才能识别规则里的谓词（如"老师"）
            for predicate in &new_predicates {
                self.aligner.register_term(predicate, TermKind::Relation);
            }
            if !new_predicates.is_empty() {
                self.rebuild_knowledge_caches();
            }
            self.rebuild_resolver();
        }
        added
    }

    fn render_clause_zh(&self, clause: &Clause) -> String {
        let head = chinese_reasoning_nlu::render_atom_zh(&clause.head);
        let body = clause
            .body
            .iter()
            .map(chinese_reasoning_nlu::render_atom_zh)
            .collect::<Vec<_>>()
            .join(" 且 ");
        format!("只要「{}」，就「{}」", body, head)
    }

    /// 中文通用规则学习：把「规则：如果X的师傅是Y，那么X的老师是Y」直接解析成
    /// Horn 子句（X/Y/Z、甲乙丙、A/B/C 等占位符自动成为逻辑变量），进入会话后向链规则库
    pub fn learn_rule_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出中文规则，例如：规则：如果X的师傅是Y，那么X的老师是Y".to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let rules: Vec<(&chinese_reasoning_nlu::ChineseLesson, Clause)> = outcome
            .lessons
            .iter()
            .filter_map(|lesson| {
                chinese_reasoning_nlu::lesson_to_clause(lesson).map(|clause| (lesson, clause))
            })
            .collect();
        if rules.is_empty() {
            let mut output = String::from("没能把这句话解析成规则。支持句式：\n");
            output.push_str(
                "  • 如果S的P是O，那么S的Q是O（S/O 用 X/Y/Z、甲乙丙、A/B/C 等占位符表示变量）\n",
            );
            output.push_str("  • 因为S的P是O，所以S的Q是O\n");
            for error in &outcome.errors {
                output.push_str(&format!("  失败原因：{}\n", error));
            }
            return output.trim_end().to_string();
        }
        let mut output =
            String::from("已学习中文规则（纯符号解析 → Horn 子句，直接进入后向链）：\n");
        for (index, (lesson, clause)) in rules.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 原始：{}\n     逻辑形式：{} :- {}\n     规则性质：{}；置信度 {:.2}\n",
                index + 1,
                lesson.original,
                clause.head.display(),
                clause
                    .body
                    .iter()
                    .map(|atom| atom.display())
                    .collect::<Vec<_>>()
                    .join(", "),
                if chinese_reasoning_nlu::lesson_has_variables(lesson) {
                    "通用规则（含逻辑变量，可套用到任意实体）"
                } else {
                    "具体规则（仅对提到的实体生效）"
                },
                clause.confidence
            ));
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未学习：{}\n", error));
        }
        let added = self.register_lesson_clauses(&outcome.lessons);
        if added == 0 {
            output.push_str("  （以上规则已存在，未重复添加）\n");
        } else {
            output.push_str(&format!("  新增 {} 条会话规则。", added));
        }
        output.push_str(&format!(
            "会话规则库现有 {} 条子句、{} 条事实；可用「判断/为什么/推理链/推理发现」验证后向链。",
            self.resolver.clause_count(),
            self.resolver.fact_count()
        ));
        output
    }

    /// 记住一条中文事实：解析为确定事实并加入会话归结事实集（不写知识图谱）
    /// 解析「A 不是 B 的 P」「A 不认识 B」这类否定陈述为被阻断的正向原子。
    fn parse_negative_atom(sentence: &str) -> Option<Atom> {
        let sentence = sentence.trim();
        let positive = if let Some(position) = sentence.find("不是") {
            let mut text = String::with_capacity(sentence.len());
            text.push_str(&sentence[..position]);
            text.push('是');
            text.push_str(&sentence[position + "不是".len()..]);
            text
        } else {
            let (position, _) = sentence
                .char_indices()
                .find(|(index, character)| *index > 0 && *character == '不')?;
            let after = &sentence[position + '不'.len_utf8()..];
            if ![
                "认识",
                "喜欢",
                "讨厌",
                "属于",
                "位于",
                "出生于",
                "隶属于",
                "工作于",
            ]
            .iter()
            .any(|verb| after.starts_with(verb))
            {
                return None;
            }
            let mut text = String::with_capacity(sentence.len());
            text.push_str(&sentence[..position]);
            text.push_str(after);
            text
        };
        let atom = chinese_reasoning_nlu::parse_proposition(&positive)?;
        if atom.terms.len() != 2 || !atom.is_ground() {
            return None;
        }
        Some(atom)
    }

    fn assert_negative_atom(&mut self, atom: Atom) -> String {
        self.register_atom_vocabulary(&atom);
        let rendered = self.render_atom_nl(&atom);
        let removed_positive = self.retract_atom(&atom);
        let added = if self.session_negative_facts.contains(&atom) {
            false
        } else {
            self.session_negative_facts.push(atom);
            true
        };
        let mut output = if added {
            format!(
                "已记住否定事实：{}。后续多跳路径会把它作为阻断边（未写知识图谱）。",
                rendered
            )
        } else {
            format!("否定事实「{}」已经记住。", rendered)
        };
        if removed_positive {
            output.push_str("已同时撤销同名会话肯定事实。");
        }
        output
    }

    /// 记住一条否定陈述：只作为会话级阻断边，不写知识图谱。
    pub fn assert_negative_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        let Some(atom) = Self::parse_negative_atom(sentence) else {
            return "没能把这句话解析为可阻断的否定事实；例如：李四不是王五的师傅".to_string();
        };
        self.assert_negative_atom(atom)
    }

    pub fn assert_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出要记住的事实，例如：记住：张三是李四的师傅".to_string();
        }
        if let Some(atom) = Self::parse_negative_atom(sentence) {
            return self.assert_negative_atom(atom);
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let mut remembered: Vec<String> = Vec::new();
        let mut refused: Vec<String> = Vec::new();
        let mut superseded_negatives: Vec<String> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact {
                let hint = match lesson.kind {
                    LessonKind::DefaultRule => "默认句请用「教学：」学习",
                    LessonKind::ConditionalRule | LessonKind::CausalRule => {
                        "条件句请用「规则：」或「教学：」学习"
                    }
                    LessonKind::ProbabilisticFact => "概率句请用「教学：」学习",
                    LessonKind::DeterministicFact => "",
                };
                refused.push(format!("「{}」不是确定事实（{}）", lesson.original, hint));
                continue;
            }
            if !lesson.head.is_ground() {
                refused.push(format!(
                    "「{}」含占位符，无法作为具体事实；请改用「规则：」",
                    lesson.original
                ));
                continue;
            }
            self.register_atom_vocabulary(&lesson.head);
            let rendered = self.render_atom_nl(&lesson.head);
            if let Some(position) = self
                .session_negative_facts
                .iter()
                .position(|negative| negative == &lesson.head)
            {
                self.session_negative_facts.remove(position);
                superseded_negatives.push(rendered.clone());
            }
            if self.adopt_atom(lesson.head.clone()) {
                remembered.push(rendered);
            } else {
                refused.push(format!("「{}」已经记住", rendered));
            }
        }
        let mut output = String::new();
        if remembered.is_empty() {
            output.push_str("没有新的会话事实被记住。\n");
        } else {
            output.push_str(&format!(
                "已记住 {} 条会话事实（不写知识图谱，随对话记忆持久化）：\n",
                remembered.len()
            ));
            for (index, fact) in remembered.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", index + 1, fact));
            }
        }
        for reason in &refused {
            output.push_str(&format!("  跳过：{}\n", reason));
        }
        if !superseded_negatives.is_empty() {
            output.push_str("已撤销同名会话否定事实：");
            output.push_str(&superseded_negatives.join("；"));
            output.push_str("。\n");
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未解析：{}\n", error));
        }
        output.push_str(&format!(
            "会话事实共 {} 条；后续「判断/为什么/推理链」会使用它们。",
            self.session_facts.len()
        ));
        output
    }

    /// 忘记一条会话事实：从会话归结事实集撤回（不触碰知识图谱）
    pub fn retract_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出要忘记的事实，例如：忘记：张三是李四的师傅".to_string();
        }
        if let Some(atom) = Self::parse_negative_atom(sentence) {
            let rendered = self.render_atom_nl(&atom);
            if let Some(position) = self
                .session_negative_facts
                .iter()
                .position(|existing| existing == &atom)
            {
                self.session_negative_facts.remove(position);
                return format!("已撤销否定事实：{}。", rendered);
            }
            return format!("否定事实中没有「{}」。", rendered);
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let mut removed: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
                continue;
            }
            let rendered = self.render_atom_nl(&lesson.head);
            let removed_negative = self
                .session_negative_facts
                .iter()
                .position(|negative| negative == &lesson.head)
                .map(|position| self.session_negative_facts.remove(position))
                .is_some();
            if self.retract_atom(&lesson.head) || removed_negative {
                removed.push(rendered);
            } else {
                missing.push(rendered);
            }
        }
        let mut output = String::new();
        if removed.is_empty() && missing.is_empty() {
            output.push_str(
                "没能把这句话解析为可撤回的事实（会话事实不会来自知识图谱，只撤回会话级记忆）。\n",
            );
        }
        for fact in &removed {
            output.push_str(&format!("已忘记会话事实：{}\n", fact));
        }
        for fact in &missing {
            output.push_str(&format!(
                "会话事实中没有「{}」（知识图谱事实不在此处撤回）。\n",
                fact
            ));
        }
        for error in &outcome.errors {
            output.push_str(&format!("未解析：{}\n", error));
        }
        output.push_str(&format!("会话事实剩余 {} 条。", self.session_facts.len()));
        output
    }

    /// 会话规则库报告：教过的 Horn 规则 + 记住的会话事实
    pub fn session_rules_report(&self) -> String {
        let mut output = format!(
            "会话后向链规则库（纯符号 Horn 子句，{} 条；事实 {} 条，其中教过的规则 {} 条）：\n",
            self.resolver.clause_count(),
            self.resolver.fact_count(),
            self.session_clauses.len()
        );
        if self.session_clauses.is_empty() {
            output.push_str(
                "  （还没有学习过中文规则；可输入「规则：如果X的师傅是Y，那么X的老师是Y」）\n",
            );
        }
        for (index, clause) in self.session_clauses.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}  〔{} :- {}；来源：{}；置信度 {:.2}〕\n",
                index + 1,
                self.render_clause_zh(clause),
                clause.head.display(),
                clause
                    .body
                    .iter()
                    .map(|atom| atom.display())
                    .collect::<Vec<_>>()
                    .join(", "),
                clause.source,
                clause.confidence
            ));
        }
        if self.session_facts.is_empty() {
            output.push_str("  会话事实：（无；可输入「记住：张三是李四的师傅」）");
        } else {
            output.push_str("  会话事实（未写知识图谱）：");
            for (index, atom) in self.session_facts.iter().enumerate() {
                if index > 0 {
                    output.push('；');
                }
                output.push_str(&self.render_atom_nl(atom));
            }
        }
        if !self.session_negative_facts.is_empty() {
            output.push_str("\n  会话否定事实（多跳路径阻断边）：");
            for (index, atom) in self.session_negative_facts.iter().enumerate() {
                if index > 0 {
                    output.push('；');
                }
                output.push_str(&self.render_atom_nl(atom));
            }
        }
        output
    }

    /// 会话规则 → 图账本条目（规则即数据：稳定 id、规范逻辑形式、来源、置信度）。
    pub fn rule_ledger_entries(&self) -> Vec<graph_ledger::RuleLedgerEntry> {
        self.session_clauses
            .iter()
            .map(|clause| {
                let head = ledger_atom(&clause.head);
                let body: Vec<graph_ledger::LedgerAtom> =
                    clause.body.iter().map(ledger_atom).collect();
                graph_ledger::RuleLedgerEntry {
                    id: graph_ledger::rule_entry_id(&head, &body),
                    head,
                    body,
                    confidence: clause.confidence,
                    source: clause.source.clone(),
                }
            })
            .collect()
    }

    /// 当前会话规则集指纹（派生账本过期判定的依据之一；与插入顺序无关）。
    pub fn rule_fingerprint(&self) -> String {
        graph_ledger::rule_fingerprint(&self.rule_ledger_entries())
    }

    /// 从规则账本恢复会话规则（启动时合并；重复规则不重复添加）。返回新增条数。
    ///
    /// 与 `apply_dialogue_snapshot` 的规则恢复同语义：新谓词登记为关系、
    /// 词表缓存与归结引擎整体重建，保证恢复的规则立即可用于后向链。
    pub fn restore_rules_from_ledger(
        &mut self,
        entries: &[graph_ledger::RuleLedgerEntry],
    ) -> usize {
        if !self.config.enable_resolution {
            return 0;
        }
        let mut added = 0usize;
        let mut new_predicates: Vec<String> = Vec::new();
        for entry in entries {
            let Some(head) = atom_from_ledger(&entry.head) else {
                continue;
            };
            let mut body = Vec::with_capacity(entry.body.len());
            let mut valid = true;
            for atom in &entry.body {
                match atom_from_ledger(atom) {
                    Some(restored) => body.push(restored),
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            if self
                .session_clauses
                .iter()
                .any(|existing| existing.head == head && existing.body == body)
            {
                continue;
            }
            for atom in std::iter::once(&head).chain(body.iter()) {
                if !self.aligner.is_registered(&atom.predicate)
                    && !new_predicates
                        .iter()
                        .any(|p| p.as_str() == atom.predicate.as_ref())
                {
                    new_predicates.push(atom.predicate.to_string());
                }
            }
            self.session_clauses
                .push(Clause::new(head, body, entry.confidence, &entry.source));
            added += 1;
        }
        if added > 0 {
            for predicate in &new_predicates {
                self.aligner.register_term(predicate, TermKind::Relation);
            }
            if !new_predicates.is_empty() {
                self.rebuild_knowledge_caches();
            }
            self.rebuild_resolver();
        }
        added
    }

    /// 忘记一条会话规则：子句库与教学原句两处一致撤回（图账本由调用方重同步）。
    ///
    /// 教学原句若解析出同一规则，整句一并撤回——否则重启时 `rebuild_problog`
    /// 重放原句会让规则经「教学」通道复活。
    pub fn forget_rule_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出要忘记的规则，例如：忘记规则：如果X的师傅是Y，那么X的老师是Y"
                .to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let targets: Vec<Clause> = outcome
            .lessons
            .iter()
            .filter_map(chinese_reasoning_nlu::lesson_to_clause)
            .collect();
        if targets.is_empty() {
            let mut output = String::from("没能把这句话解析成可忘记的规则。支持句式：\n");
            output.push_str("  • 如果S的P是O，那么S的Q是O（与「规则：」同款，占位符自动变量化）\n");
            for error in &outcome.errors {
                output.push_str(&format!("  失败原因：{}\n", error));
            }
            return output.trim_end().to_string();
        }
        let same_clause =
            |left: &Clause, right: &Clause| left.head == right.head && left.body == right.body;

        let existing_clauses = std::mem::take(&mut self.session_clauses);
        let mut removed_targets: Vec<Clause> = Vec::new();
        let mut kept_clauses: Vec<Clause> = Vec::new();
        for clause in existing_clauses {
            match targets.iter().find(|target| same_clause(target, &clause)) {
                Some(target) => removed_targets.push(target.clone()),
                None => kept_clauses.push(clause),
            }
        }
        self.session_clauses = kept_clauses;

        // 教学原句撤回：解析结果命中同一规则的整句一并移除
        let mut lessons_removed = 0usize;
        let mut kept_lessons: Vec<String> = Vec::new();
        for lesson_sentence in &self.lessons {
            let matches = chinese_reasoning_nlu::parse_lessons(lesson_sentence)
                .lessons
                .iter()
                .filter_map(chinese_reasoning_nlu::lesson_to_clause)
                .any(|parsed| targets.iter().any(|target| same_clause(target, &parsed)));
            if matches {
                lessons_removed += 1;
            } else {
                kept_lessons.push(lesson_sentence.clone());
            }
        }
        if lessons_removed > 0 {
            self.lessons = kept_lessons;
            self.lessons_parsed = self
                .lessons
                .iter()
                .flat_map(|lesson_sentence| {
                    chinese_reasoning_nlu::parse_lessons(lesson_sentence).lessons
                })
                .collect();
            if self.config.enable_problog && self.config.enable_soft_logic {
                self.rebuild_problog();
            }
        }

        let mut output = String::new();
        if removed_targets.is_empty() {
            output.push_str("会话规则库中没有匹配的规则（可能未学过或已被忘记）。\n");
        } else {
            output.push_str(&format!("已忘记 {} 条会话规则：\n", removed_targets.len()));
            for (index, clause) in removed_targets.iter().enumerate() {
                output.push_str(&format!(
                    "  {}. {} :- {}\n",
                    index + 1,
                    clause.head.display(),
                    clause
                        .body
                        .iter()
                        .map(|atom| atom.display())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            self.rebuild_resolver();
        }
        if lessons_removed > 0 {
            output.push_str(&format!(
                "同时撤回 {} 条对应教学原句（防止重启重放复活）。\n",
                lessons_removed
            ));
        }
        output.push_str(&format!(
            "会话规则库现有 {} 条子句。",
            self.session_clauses.len()
        ));
        output
    }

    /// 前向饱和的派生事实 → 图账本条目（含前提、规则来源、轮次、置信度与物化戳）。
    ///
    /// 与「推理发现」报告同排序（轮次降序 → 置信度降序 → 原子字典序），
    /// 截断到 `limit` 条；物化戳取当前知识版本与当前规则集指纹。
    pub fn derived_ledger_entries(&self, limit: usize) -> Vec<graph_ledger::DerivedLedgerEntry> {
        if !self.config.enable_resolution || limit == 0 {
            return Vec::new();
        }
        let fingerprint = self.rule_fingerprint();
        let derived = self.resolver.forward_chain(3, 100);
        let mut items: Vec<&DerivedFact> = derived.iter().collect();
        items.sort_by(|a, b| {
            b.depth
                .cmp(&a.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        items.truncate(limit);
        items
            .into_iter()
            .map(|item| {
                let atom = ledger_atom(&item.atom);
                graph_ledger::DerivedLedgerEntry {
                    id: graph_ledger::derived_entry_id(&atom),
                    atom,
                    rule_source: item.rule_source.clone(),
                    premises: item.premises.iter().map(ledger_atom).collect(),
                    depth: item.depth,
                    confidence: item.confidence,
                    revision: self.revision,
                    fingerprint: fingerprint.clone(),
                }
            })
            .collect()
    }

    /// ProbLog 风格可能世界精确推理报告：
    /// 概率事实/规则/注释析取 → BDD 加权模型计数（无采样），并列出证明路径与联合概率
    pub fn problog_report(&self, target_raw: &str) -> String {
        if !self.config.enable_problog {
            return "概率可能世界推理未启用（config.toml [reasoning] enable_problog = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "概率可能世界推理依赖加权软逻辑（图谱事实与规则同源），请先启用 enable_soft_logic。"
                .to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要精确查询的对象，例如：概率推理：李四的师祖 或 概率推理：李四"
                .to_string();
        }
        let (entity, predicate, known_entity) = match self.split_entity_predicate(target) {
            Some((entity, predicate)) => (entity, predicate, true),
            None => {
                let trimmed = target.trim();
                match trimmed.split_once('的') {
                    Some((entity, predicate)) if !entity.trim().is_empty() => (
                        entity.trim().to_string(),
                        Some(predicate.trim().to_string())
                            .filter(|predicate| !predicate.is_empty()),
                        false,
                    ),
                    _ if !trimmed.is_empty() => (trimmed.to_string(), None, false),
                    _ => return format!("无法在「{}」中识别出已知实体。", target),
                }
            }
        };
        let patterns = if known_entity {
            self.problog_patterns(&entity, predicate.as_deref())
        } else {
            let mut patterns = self.problog.patterns_for_subject(&entity);
            if let Some(predicate) = predicate.as_deref() {
                patterns.retain(|atom| {
                    atom.predicate.contains(predicate)
                        || predicate.contains(atom.predicate.as_ref())
                });
            }
            patterns.truncate(6);
            patterns
        };
        if patterns.is_empty() {
            return if known_entity {
                format!(
                    "没有找到关于「{}」可用于概率查询的目标（可先用「教学：」教一条知识，或确认实体已在知识图谱中）。",
                    entity
                )
            } else {
                format!(
                    "无法在「{}」中识别出已知实体（也没有教过与「{}」相关的概率知识）。",
                    target, entity
                )
            };
        }
        let mut merged: HashMap<String, ProblogAnswer> = HashMap::new();
        let mut stats: Option<(usize, usize, usize, usize, bool)> = None;
        for pattern in &patterns {
            let query =
                self.problog
                    .query(pattern, 8, self.config.soft_logic_min_probability.max(0.05));
            stats.get_or_insert((
                query.choice_variables,
                query.bdd_nodes,
                query.ground_instances,
                query.ground_rounds,
                query.exact,
            ));
            for answer in query.answers {
                merged
                    .entry(answer.atom.display())
                    .and_modify(|existing| {
                        if answer.probability > existing.probability {
                            *existing = answer.clone();
                        }
                    })
                    .or_insert(answer);
            }
        }
        let mut answers: Vec<ProblogAnswer> = merged.into_values().collect();
        answers.sort_by_cached_key(|answer| ProbabilityDisplayKey {
            probability: answer.probability,
            display: answer.atom.display(),
        });
        answers.truncate(8);
        if answers.is_empty() {
            return format!("关于「{}」目前没有概率 ≥ 阈值的精确推断结果。", entity);
        }
        let (choice_variables, bdd_nodes, ground_instances, ground_rounds, exact) =
            stats.unwrap_or((0, 0, 0, 0, true));
        let mut output = format!(
            "可能世界精确推理（ProbLog 风格：{} 轮基化得到 {} 条规则实例，{} 个选择变量编译成 {} 个 BDD 节点；加权模型计数，无采样、不训练）\n",
            ground_rounds, ground_instances, choice_variables, bdd_nodes
        );
        if !exact {
            output.push_str("  注意：BDD 超出编译上限，概率为证明路径联合概率上界近似。\n");
        }
        let soft_probabilities = self.soft.probability_map();
        for (index, answer) in answers.iter().enumerate() {
            let source_note = if answer.from_fact {
                if self.soft.has_fact(&answer.atom) {
                    "图谱事实"
                } else {
                    "概率事实/会话事实"
                }
            } else {
                "由概率规则编译推出"
            };
            output.push_str(&format!(
                "  {}. {} {}（P={:.3}，{}）\n",
                index + 1,
                self.render_atom_nl(&answer.atom),
                MarkovLogicEngine::hedge(answer.probability),
                answer.probability,
                source_note
            ));
            let soft_probability = if self.soft.has_fact(&answer.atom) {
                1.0
            } else {
                soft_probabilities.get(&answer.atom).copied().unwrap_or(0.0)
            };
            if soft_probability > 1e-9 {
                output.push_str(&format!(
                    "     与软逻辑（MLN-lite）对照：{:.3}，Δ={:+.3}（差异来自两套语义：log-odds 求和 vs 可能世界共享选择变量）\n",
                    soft_probability,
                    answer.probability - soft_probability
                ));
            }
            if answer.proofs.is_empty() {
                if !answer.from_fact {
                    output.push_str("     证明路径：（存在递归/循环依赖，未展开有限证明路径）\n");
                }
                continue;
            }
            for (proof_index, proof) in answer.proofs.iter().take(3).enumerate() {
                let steps = proof
                    .steps
                    .iter()
                    .map(|step| {
                        let premises = if step.premises.is_empty() {
                            "（无条件）".to_string()
                        } else {
                            step.premises
                                .iter()
                                .map(|premise| self.render_atom_nl(premise))
                                .collect::<Vec<_>>()
                                .join(" 且 ")
                        };
                        format!(
                            "{} —[{:.2}]→ {}（来源：{}）",
                            premises,
                            step.probability,
                            self.render_atom_nl(&step.head),
                            step.source
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("；");
                output.push_str(&format!(
                    "     证明路径 {}（联合概率 {:.3}）：{}\n",
                    proof_index + 1,
                    proof.probability,
                    steps
                ));
            }
            let path_sum: f64 = answer.proofs.iter().map(|proof| proof.probability).sum();
            if answer.proofs.len() > 1 && path_sum > answer.probability + 1e-6 {
                output.push_str(&format!(
                    "     路径联合概率之和 {:.3} > 精确概率 {:.3}：各路径共享同一批选择变量（证明重叠），精确值已做去重\n",
                    path_sum, answer.probability
                ));
            }
        }
        output.trim_end().to_string()
    }

    /// 从当前会话知识构建默认理论：图谱三元组/属性/实体类型 + 会话事实为事实，
    /// 教过的「除非」默认句为默认规则（cross-session 由 lessons 重放，无新增快照字段）
    fn build_default_theory(&self) -> DefaultTheory {
        let mut theory = DefaultTheory::new(DEFAULT_LOGIC_MAX_ROUNDS);
        for triple in self.predictor.triples() {
            theory.add_fact_atom(Atom::rel(
                &triple.predicate,
                &triple.subject,
                &triple.object,
            ));
        }
        for (subject, key, value) in &self.attributes {
            theory.add_fact_atom(Atom::rel(key, subject, value));
        }
        for (entity, entity_type) in self.generator.typed_entities() {
            if !entity.trim().is_empty() && !entity_type.trim().is_empty() {
                theory.add_fact_atom(Atom::rel("是", &entity, &entity_type));
            }
        }
        for atom in &self.session_facts {
            theory.add_fact_atom(atom.clone());
        }
        for lesson in &self.lessons_parsed {
            if lesson.kind != LessonKind::DefaultRule {
                continue;
            }
            let Some(exception) = &lesson.exception else {
                continue;
            };
            let prerequisite: Vec<Literal> = lesson
                .body
                .iter()
                .map(|atom| Literal::positive(chinese_reasoning_nlu::generalize_atom(atom)))
                .collect();
            let justifications = vec![Literal::negative(chinese_reasoning_nlu::generalize_atom(
                exception,
            ))];
            let consequent =
                Literal::positive(chinese_reasoning_nlu::generalize_atom(&lesson.head));
            theory.add_default(prerequisite, justifications, consequent, &lesson.source);
        }
        theory
    }

    /// 默认逻辑查询：把会话知识接地成良基扩展，报告触发/例外阻断，
    /// 并用 ProbLog 对同一目标给出「概率视图」对照（不建模例外的乐观倾向）
    pub fn default_logic_report(&mut self, target_raw: &str) -> String {
        let theory = self.build_default_theory();
        let closure = self.discourse.narrator_mut().closure().clone();
        let extension = theory.extend(&closure);
        let target_text = target_raw.trim();
        let target_atom = if target_text.is_empty() {
            None
        } else {
            chinese_reasoning_nlu::parse_proposition(target_text)
        };
        let target = target_atom
            .as_ref()
            .map(|atom| Literal::positive(atom.clone()));
        let mut output = default_logic::render_extension_zh(&theory, &extension, target.as_ref());
        if !target_text.is_empty() && target.is_none() {
            output.push_str(&format!(
                "\n无法把「{}」形式化为可查询命题（支持：S的P是O / S是O / S会V / 谓词(参数)）。",
                target_text
            ));
        }
        if let (Some(atom), true, true) = (
            target_atom.as_ref(),
            self.config.enable_problog,
            self.config.enable_soft_logic,
        ) {
            let query = self.problog.query(atom, 3, 0.0);
            match query.answers.first() {
                Some(answer) => output.push_str(&format!(
                    "\n概率视图（ProbLog 精确可能世界，不建模「除非」例外，仅作对照）：{} {}（P={:.3}）。",
                    self.render_atom_nl(&answer.atom),
                    MarkovLogicEngine::hedge(answer.probability),
                    answer.probability
                )),
                None => output
                    .push_str("\n概率视图（ProbLog）：概率程序中没有与目标相关的证据。"),
            }
        }
        output
    }

    /// 默认逻辑补充证据（结果抽取兜底入口）：主问答路径未命中「没有记录」或
    /// 无法作答时，用「除非」默认知识给出非单调补充；无默认规则、目标无法
    /// 形式化或默认扩展中无证据时返回 None（不产生任何输出与状态变化）
    pub fn default_logic_supplement(&mut self, question: &str) -> Option<String> {
        if !self
            .lessons_parsed
            .iter()
            .any(|lesson| lesson.kind == LessonKind::DefaultRule)
        {
            return None;
        }
        // 自然问句去掉句首礼貌语与句末疑问/标点，让「麻雀会飞吗」也能走默认推理
        let trimmed = question.trim();
        let trimmed = trimmed.strip_prefix("请问").unwrap_or(trimmed);
        let cleaned = trimmed
            .trim_end_matches(|c| {
                matches!(
                    c,
                    '吗' | '呢' | '吧' | '么' | '呀' | '？' | '?' | '。' | '！' | '!'
                )
            })
            .trim();
        let atom = chinese_reasoning_nlu::parse_proposition(cleaned)?;
        let target = Literal::positive(atom);
        let theory = self.build_default_theory();
        let closure = self.discourse.narrator_mut().closure().clone();
        let extension = theory.extend(&closure);
        if let Some(application) = extension.application_for(&target) {
            let mut output = format!(
                "默认推理补充：「{}」通常成立（{}；未发现例外）。",
                default_logic::Literal::render_zh(&target),
                default_logic::render_application_zh(application)
            );
            if extension.truncated {
                output.push_str("（注意：接地推导达到轮数上限，结果可能不完整。）");
            }
            return Some(output);
        }
        if let Some(blocked) = extension.blocked_for(&target).first() {
            return Some(format!(
                "默认推理补充：不能推出「{}」。{}",
                default_logic::Literal::render_zh(&target),
                default_logic::render_blocked_zh(blocked)
            ));
        }
        None
    }

    /// 概率查询模式：有谓词时二元/一元各试一次；无谓词时枚举主语相关的概率原子
    fn problog_patterns(&self, entity: &str, predicate: Option<&str>) -> Vec<Atom> {
        if let Some(pred) = predicate.filter(|pred| !pred.is_empty()) {
            let mut patterns = vec![Atom::pattern(pred, Term::constant(entity), Term::var("X"))];
            patterns.push(Atom::unary(pred, Term::constant(entity)));
            return patterns;
        }
        let mut patterns = self.problog.patterns_for_subject(entity);
        if patterns.is_empty() {
            for fact in self.collect_soft_facts_for_entity(entity, 6) {
                let pattern = if fact.atom.arity() == 2 && fact.atom.terms[0].name() == entity {
                    Atom::pattern(
                        fact.atom.predicate.as_ref(),
                        Term::constant(entity),
                        Term::var("X"),
                    )
                } else {
                    fact.atom.clone()
                };
                if !patterns.contains(&pattern) {
                    patterns.push(pattern);
                }
            }
        }
        patterns.truncate(6);
        patterns
    }

    /// 把「实体[的谓词]」拆成规范实体与可选谓词（最长前缀匹配，支持同义词）
    fn split_entity_predicate(&self, target: &str) -> Option<(String, Option<String>)> {
        let mut best: Option<(usize, String)> = None;
        let mut consider = |name: &str, canonical: &str| {
            if name.is_empty() || !target.starts_with(name) {
                return;
            }
            let length = name.chars().count();
            if best
                .as_ref()
                .map(|(best_length, _)| length > *best_length)
                .unwrap_or(true)
            {
                best = Some((length, canonical.to_string()));
            }
        };
        for profile in self.aligner.profiles() {
            // 前缀预筛：只处理可能命中目标前缀的实体画像
            if profile.kind == TermKind::Entity
                && !profile.term.is_empty()
                && target.starts_with(&profile.term)
            {
                let canonical = self.aligner.canonical_form(&profile.term);
                consider(&profile.term, &canonical);
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if target.starts_with(alias.as_str())
                && self.aligner.kind_of(canonical) == Some(TermKind::Entity)
            {
                consider(alias, canonical);
            }
        }
        let (length, entity) = best?;
        let remainder: String = target.chars().skip(length).collect();
        let remainder = remainder
            .trim()
            .trim_start_matches(['的', '：', ':', ' ', '是']);
        let predicate = if remainder.is_empty() {
            None
        } else {
            Some(self.aligner.canonical_form(remainder))
        };
        Some((entity, predicate))
    }

    pub fn dialogue_summary(&self) -> String {
        let focus = if self.context.focus_entities().is_empty() {
            "（空）".to_string()
        } else {
            self.context.focus_entities().join(" → ")
        };
        let predicate = self
            .context
            .last_predicate()
            .unwrap_or("（无）")
            .to_string();
        let generation = self
            .context
            .generation_state()
            .map(|state| {
                let order = match state.order {
                    GenerationOrder::Relevance => "信息量排序",
                    GenerationOrder::Alphabetical => "主题排序",
                };
                format!("「{}」（已讲 {} 句，{}）", state.entity, state.shown, order)
            })
            .unwrap_or_else(|| "（无）".to_string());
        let session_info =
            if self.session_facts.is_empty() && self.session_negative_facts.is_empty() {
                "（无）".to_string()
            } else {
                format!(
                    "{} 条，会话否定 {} 条",
                    self.session_facts.len(),
                    self.session_negative_facts.len()
                )
            };
        let clarify = self
            .clarification_progress()
            .map(|progress| format!("进行中（{}）", progress))
            .unwrap_or_else(|| "（无）".to_string());
        let lesson_info = if self.lessons.is_empty() {
            "（无）".to_string()
        } else {
            format!("{} 条", self.lessons.len())
        };
        let frame_info = {
            let clarification = match self.frames.clarification() {
                Some(request) => format!("，待绑定澄清「{}」", request.raw),
                None => String::new(),
            };
            match self.frames.last() {
                Some(frame) => format!(
                    "上一帧「{}」、挂账 {} 条{}",
                    frame.display(),
                    self.frames.pending().len(),
                    clarification
                ),
                None => format!(
                    "上一帧（无）、挂账 {} 条{}",
                    self.frames.pending().len(),
                    clarification
                ),
            }
        };
        format!(
            "对话记忆：焦点实体栈 [{}]，上一轮谓词「{}」，上一段生成 {}，历史问题 {} 条，会话假设 {}，会话规则 {} 条，澄清 {}，中文概率知识 {}，话语帧 {}。",
            focus,
            predicate,
            generation,
            self.context.history().len(),
            session_info,
            self.session_clauses.len(),
            clarify,
            lesson_info,
            frame_info
        )
    }

    /// 会话记忆的完整内存快照（焦点/历史/生成状态/会话事实/教学/规则/话语帧）。
    ///
    /// 与 [`Self::save_dialogue_state_with_contrast`] 落盘的内容一致（除对比仲裁
    /// 记忆外），供服务层在不触碰文件系统的前提下导出/导入会话。
    pub fn dialogue_snapshot(&self) -> DialogueSnapshot {
        let mut snapshot = self.context.snapshot();
        snapshot.session_facts = self
            .session_facts
            .iter()
            .filter(|atom| atom.is_ground())
            .map(|atom| SnapshotAtom {
                predicate: atom.predicate.to_string(),
                terms: atom
                    .terms
                    .iter()
                    .map(|term| term.name().to_string())
                    .collect(),
            })
            .collect();
        snapshot.session_negative_facts = self
            .session_negative_facts
            .iter()
            .filter(|atom| atom.is_ground())
            .map(|atom| SnapshotAtom {
                predicate: atom.predicate.to_string(),
                terms: atom
                    .terms
                    .iter()
                    .map(|term| term.name().to_string())
                    .collect(),
            })
            .collect();
        snapshot.lessons = self.lessons.clone();
        snapshot.session_clauses = self.session_clauses.iter().map(snapshot_clause).collect();
        snapshot.frames = Some(self.frames.clone());
        snapshot.ontology = self.discourse.ontology_capsule();
        snapshot
    }

    /// 把 [`Self::dialogue_snapshot`] 的内容恢复到当前推理器上。
    ///
    /// 恢复只重建会话级派生数据（会话事实/教学/会话规则/话语帧/本体胶囊），
    /// 全局知识图谱不受影响；同一快照恢复两次结果一致（幂等去重）。
    /// 本体胶囊的版本门或公理摘要不通过时整体拒绝并返回 `Err`（零半应用）。
    pub fn restore_dialogue_snapshot(&mut self, snapshot: &DialogueSnapshot) -> Result<(), String> {
        self.apply_dialogue_snapshot(snapshot)
    }

    pub fn save_dialogue_state(&self, path: &str) -> std::io::Result<()> {
        self.save_dialogue_state_with_contrast(path, None)
    }

    /// 保存对话状态，并额外写入神经符号对比仲裁记忆（校准统计/待定推测/否证集）。
    ///
    /// 落盘为**规范字节**（对象键递归排序）：同一会话状态永远同一字节串，
    /// 跨运行 diff / 审计不受 `HashMap` 迭代序影响（L2 字节不动点）。
    pub fn save_dialogue_state_with_contrast(
        &self,
        path: &str,
        contrast: Option<&ContrastSnapshot>,
    ) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut snapshot = self.dialogue_snapshot();
        snapshot.contrast = contrast.cloned();
        let bytes = canonical::canonical_bytes(&snapshot)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
        std::fs::write(path, bytes)
    }

    /// 读取并解码对话状态（不做任何应用），返回快照与解码回执。
    ///
    /// 读宽容：pretty 排布的文件仍可解码（回执 `canonical == false`）。
    pub fn read_dialogue_state(
        path: &str,
    ) -> Result<(DialogueSnapshot, DecodeReceipt), DecodeError> {
        let bytes = std::fs::read(path).map_err(|error| DecodeError::Io(error.to_string()))?;
        canonical::decode_json::<DialogueSnapshot>(
            &bytes,
            "dialogue_state",
            1,
            canonical::DEFAULT_MAX_DECODE_BYTES,
        )
    }

    pub fn load_dialogue_state(&mut self, path: &str) -> bool {
        self.load_dialogue_state_with_contrast(path).0
    }

    /// 加载对话状态，同时返回其中的对比仲裁记忆快照（无则 None）
    pub fn load_dialogue_state_with_contrast(
        &mut self,
        path: &str,
    ) -> (bool, Option<ContrastSnapshot>) {
        let Ok((snapshot, _)) = Self::read_dialogue_state(path) else {
            return (false, None);
        };
        let contrast = snapshot.contrast.clone();
        // 校验失败（版本门 / 公理摘要被改写）时零半应用，按「未加载」上报。
        match self.apply_dialogue_snapshot(&snapshot) {
            Ok(()) => (true, contrast),
            Err(_) => (false, None),
        }
    }

    fn apply_dialogue_snapshot(&mut self, snapshot: &DialogueSnapshot) -> Result<(), String> {
        // 本体胶囊最先恢复：版本门 / 公理摘要不通过就整体拒绝，
        // 绝不留下「会话态已换、本体还是旧的」这种半应用状态。
        if let Some(capsule) = &snapshot.ontology {
            self.discourse.restore_ontology_capsule(capsule)?;
            // 胶囊公理与检索类型层级同源：恢复后同步一次（幂等）
            self.sync_retrieval_type_hierarchy();
        }
        self.context.restore(snapshot);
        // 话语帧记忆：恢复上一帧与挂账（重启后接着补全未竟之问）
        if let Some(frames) = &snapshot.frames {
            self.frames = frames.clone();
        }
        // 跨会话临时信念：恢复被采纳的会话假设（仍不写知识图谱）
        let mut restored: Vec<Atom> = Vec::new();
        for entry in &snapshot.session_facts {
            if entry.predicate.is_empty() || entry.terms.is_empty() {
                continue;
            }
            let terms: Vec<Term> = entry
                .terms
                .iter()
                .map(|name| Term::constant(name))
                .collect();
            let atom = Atom::new(&entry.predicate, terms);
            if atom.is_ground() && !restored.contains(&atom) {
                restored.push(atom);
            }
        }
        if !restored.is_empty() {
            // 与 adopt_atom 保持一致：恢复的会话事实同样注入软逻辑/概率程序，
            // 否则恢复后 soft_inference_report 与 problog_report 缺少这些事实
            for atom in &restored {
                if self.config.enable_soft_logic {
                    self.soft.add_fact(atom.clone());
                }
                if self.config.enable_problog {
                    self.problog.add_deterministic_fact(atom.clone());
                }
            }
            self.abduction.restore_adopted(restored.clone());
            for atom in &restored {
                self.index_session_atom(atom);
            }
            self.session_facts = restored;
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
        }
        // 否定事实：恢复为路径阻断边（不进入正向归结，因此不触发引擎重建）
        let mut restored_negative: Vec<Atom> = Vec::new();
        for entry in &snapshot.session_negative_facts {
            if entry.predicate.is_empty() || entry.terms.len() != 2 {
                continue;
            }
            let terms: Vec<Term> = entry
                .terms
                .iter()
                .map(|name| Term::constant(name))
                .collect();
            let atom = Atom::new(&entry.predicate, terms);
            if atom.is_ground() && !restored_negative.contains(&atom) {
                restored_negative.push(atom);
            }
        }
        if !restored_negative.is_empty() {
            for atom in &restored_negative {
                self.register_atom_vocabulary(atom);
            }
            self.session_negative_facts = restored_negative;
        }
        // 教学记忆：恢复中文概率知识原句并在概率程序中重放（不写知识图谱）
        if !snapshot.lessons.is_empty() {
            self.lessons = snapshot.lessons.clone();
            // 恢复原句时重建解析缓存
            self.lessons_parsed = self
                .lessons
                .iter()
                .flat_map(|sentence| chinese_reasoning_nlu::parse_lessons(sentence).lessons)
                .collect();
            if self.config.enable_problog && self.config.enable_soft_logic {
                self.rebuild_problog();
            }
        }
        // 会话 Horn 规则：跨会话还原后向链规则库（谓词词表随之恢复）
        if !snapshot.session_clauses.is_empty() {
            let mut restored: Vec<Clause> = Vec::new();
            let mut new_predicates: Vec<String> = Vec::new();
            for entry in &snapshot.session_clauses {
                let Some(clause) = restore_clause(entry) else {
                    continue;
                };
                if restored.iter().any(|existing: &Clause| {
                    existing.head == clause.head && existing.body == clause.body
                }) {
                    continue;
                }
                for atom in std::iter::once(&clause.head).chain(clause.body.iter()) {
                    if !self.aligner.is_registered(&atom.predicate)
                        && !new_predicates
                            .iter()
                            .any(|p| p.as_str() == atom.predicate.as_ref())
                    {
                        new_predicates.push(atom.predicate.to_string());
                    }
                }
                restored.push(clause);
            }
            if !restored.is_empty() {
                for predicate in &new_predicates {
                    self.aligner.register_term(predicate, TermKind::Relation);
                }
                self.session_clauses = restored;
                if !new_predicates.is_empty() {
                    self.rebuild_knowledge_caches();
                }
                if self.config.enable_resolution {
                    self.rebuild_resolver();
                }
            }
        }
        Ok(())
    }

    pub fn alignment_report(&self) -> String {
        let pairs = self.aligner.align_all();
        let mut output = format!(
            "本体对齐报告：已登记 {} 个术语，发现 {} 组对齐关系。\n",
            self.aligner.term_count(),
            pairs.len()
        );
        for pair in pairs.iter().take(20) {
            output.push_str(&format!(
                "  • {} ⇒ {}（{:?}，置信度 {:.2}，证据：{}）\n",
                pair.source,
                pair.target,
                pair.kind,
                pair.confidence,
                pair.evidence.join("；")
            ));
        }
        if pairs.is_empty() {
            output.push_str("  （未发现跨词表对齐，可补充同义词或更多关系数据）\n");
        }
        let ambiguous = self.aligner.ambiguous_terms(8);
        if ambiguous.is_empty() {
            output.push_str("对齐歧义：未发现候选分数接近的术语。\n");
        } else {
            output.push_str(&format!(
                "对齐歧义：{} 个术语存在候选分数接近的对齐（可用「对齐候选：术语」查看）。\n",
                ambiguous.len()
            ));
            for resolution in ambiguous {
                output.push_str(&format!(
                    "  • {} → {}\n",
                    resolution.source,
                    resolution
                        .targets()
                        .into_iter()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" / ")
                ));
            }
        }
        output
    }

    /// 只读访问本体对齐器（诊断 / 外部知识属性准入等只读消费者）。
    pub fn aligner(&self) -> &OntologyAligner {
        &self.aligner
    }

    /// 候选格报告：展示 top-k 对齐候选、证据与（有上下文实体时的）答案等价性。
    pub fn alignment_candidates_report(&self, term_raw: &str) -> String {
        let term = term_raw.trim();
        if term.is_empty() {
            return "请提供要对齐的术语，例如：对齐候选：老婆".to_string();
        }
        let resolution = self.aligner.resolve_candidates(term, 5);
        let mut output = format!(
            "对齐候选「{}」：状态 {}，规范化形「{}」。\n",
            term,
            resolution.state.label_zh(),
            resolution.normalized
        );
        if resolution.is_empty() {
            if let Some(kind) = self.aligner.kind_of(term) {
                output.push_str(&format!("  精确登记为 {:?}，没有其他接近候选。\n", kind));
            } else {
                output.push_str("  （没有达到阈值的候选）\n");
            }
            return output;
        }
        for (index, candidate) in resolution.candidates.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}（{:?}，置信度 {:.2}，证据：{}）\n",
                index + 1,
                candidate.target(),
                candidate.kind(),
                candidate.confidence(),
                candidate.pair.evidence.join("；")
            ));
        }

        let Some(entity) = self.context.focus() else {
            return output;
        };
        let mut answerable = resolution.clone();
        answerable.retain(|candidate| self.is_answerable_predicate(candidate.target()));
        if answerable.is_empty() {
            return output;
        }
        let decision = self.aligner.probe_answers(answerable, |target| {
            self.predicate_values_for_entity(entity, target)
        });
        output.push_str(&format!(
            "\n答案等价性（上下文实体「{}」）：{}\n",
            entity,
            if decision.equivalent_answers {
                "候选给出相同答案，不需要澄清"
            } else if decision.should_clarify() {
                "候选给出不同答案，值得澄清"
            } else {
                "只有一种解释有答案，按最佳候选作答"
            }
        ));
        for answer in &decision.answers {
            output.push_str(&format!(
                "  • 「{}」→ {}\n",
                answer.target,
                if answer.answers.is_empty() {
                    "（没有记录）".to_string()
                } else {
                    answer.answers.join("、")
                }
            ));
        }
        output
    }

    /// 谓词是否能在当前知识库中产生答案（关系、属性或预测规则）。
    fn is_answerable_predicate(&self, target: &str) -> bool {
        self.canonical_attribute(target).is_some() || self.is_known_predicate(target)
    }

    /// 某实体在一个候选谓词下的全部取值（与正常查询同一条 `node_hits` 路径）。
    fn predicate_values_for_entity(&self, entity: &str, target: &str) -> Vec<String> {
        let (slot, inverted) = self.resolve_question_predicate(target);
        if slot.is_empty() {
            return Vec::new();
        }
        let predicate = ChainPredicate {
            surface: target.to_string(),
            slot,
            inverted,
        };
        self.node_hits(entity, &predicate)
            .into_iter()
            .map(|hit| hit.value)
            .collect()
    }

    /// 属性问句的对齐决策：只保留本库可作答的候选，再用答案集合判断是否真歧义。
    /// 精确登记的谓词是用户明确给出的词面，不因模糊候选而追问。
    fn attribute_alignment_decision(
        &self,
        entity: &str,
        surface: &str,
    ) -> Option<AlignmentDecision> {
        if self.aligner.is_registered(surface) && self.aligner.canonical_form(surface) == surface {
            return None;
        }
        let mut resolution = self.aligner.resolve_candidates(surface, 6);
        if resolution.state != AlignmentState::Ambiguous {
            return None;
        }
        resolution.retain(|candidate| self.is_answerable_predicate(candidate.target()));
        if resolution.candidates.len() < 2 {
            return None;
        }
        Some(self.aligner.probe_answers(resolution, |target| {
            self.predicate_values_for_entity(entity, target)
        }))
    }

    fn render_alignment_clarification(
        &self,
        surface: &str,
        decision: &AlignmentDecision,
    ) -> String {
        let mut output = format!(
            "「{}」有不止一种理解，而且在现有知识中会得到不同答案：\n",
            surface
        );
        for answer in decision.answers.iter().take(4) {
            output.push_str(&format!(
                "  • 「{}」→ {}\n",
                answer.target,
                if answer.answers.is_empty() {
                    "（没有记录）".to_string()
                } else {
                    answer.answers.join("、")
                }
            ));
        }
        output.push_str("请确认你指的是哪一种（可以换一种更具体的说法）。");
        output
    }

    pub fn rules_report(&self) -> String {
        let rules = self.predictor.rules();
        let mut output = format!(
            "已从 {} 条关系事实中挖掘出 {} 条符号推理规则（支持度阈值 {}，置信度阈值 {:.2}）：\n",
            self.predictor.triple_count(),
            rules.len(),
            self.config.link_prediction_min_support,
            self.config.link_prediction_min_confidence
        );
        for rule in rules.iter().take(30) {
            output.push_str(&format!(
                "  • [{}] {}（支持 {} 例，置信度 {:.2}，头部覆盖率 {:.2}）\n",
                rule.template.name(),
                rule.rule_text(),
                rule.support,
                rule.confidence,
                rule.head_coverage
            ));
        }
        if rules.is_empty() {
            output.push_str("  （暂无满足阈值的规则，可降低支持度/置信度阈值或增加关系数据）\n");
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ReasoningConfig {
        ReasoningConfig {
            enable_link_prediction: true,
            link_prediction_min_confidence: 0.5,
            link_prediction_min_support: 1,
            ontology_alignment_threshold: 0.72,
            max_predictions: 5,
            enable_semantic_embedding: true,
            embedding_similarity_floor: 0.6,
            dialogue_state_path: None,
            enable_generation: true,
            max_profile_sentences: 6,
            completion_min_score: 0.3,
            enable_entailment: true,
            entailment_min_confidence: 0.5,
            enable_resolution: true,
            resolution_max_depth: 6,
            resolution_max_solutions: 12,
            enable_association_rules: true,
            association_min_support: 0.2,
            association_min_confidence: 0.5,
            association_min_lift: 1.05,
            enable_abduction: true,
            abduction_max_hypotheses: 5,
            abduction_max_scenario_size: 2,
            enable_clarification: true,
            enable_graph_walk: true,
            graph_walk_damping: 0.85,
            graph_walk_iterations: 12,
            enable_soft_logic: true,
            soft_logic_max_iterations: 4,
            soft_logic_min_probability: 0.5,
            enable_causal_reasoning: true,
            causal_max_depth: 4,
            enable_gradient_attribution: true,
            enable_causal_tokenization: true,
            enable_problog: true,
            problog_max_depth: 6,
            problog_max_proofs: 12,
            enable_dialogue_manager: true,
            enable_discourse: true,
            discourse_max_sentences: 6,
            discourse_max_rounds: 4,
            enable_holographic_memory: true,
            holographic_dimension: 4096,
            holographic_similarity_floor: 0.01,
            holographic_max_hits: 6,
            require_predicate_admission: false,
        }
    }

    fn master_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师祖".to_string(),
                to: "王五".to_string(),
            },
        ]
    }

    fn pair_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]
    }

    #[test]
    fn test_pronoun_resolution_uses_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师傅是谁").is_none());
        let rewritten = reasoner
            .contextualize("他的师傅是谁")
            .expect("代词应被解析为上下文焦点实体");
        assert!(rewritten.contains("张三"));
        assert!(!rewritten.contains("他的"));
    }

    /// 类型闸门：「他/她」可解析人物（类型未知按开放世界放行），
    /// 「它」绝不指代人物，「该公司」顺位到组织，「前者/后者」按焦点栈序取。
    #[test]
    fn test_pronoun_type_gate_and_ordinals() {
        let mut context = DialogueContext::new(8);
        context.record_entities_typed(&[
            ("张三".to_string(), Some("人".to_string())),
            ("未名科技公司".to_string(), Some("组织".to_string())),
            ("某物".to_string(), None),
        ]);
        assert_eq!(
            context.resolve_pronouns("他的师傅是谁").as_deref(),
            Some("张三的师傅是谁")
        );
        assert_eq!(
            context.resolve_pronouns("它的创始人是谁").as_deref(),
            Some("未名科技公司的创始人是谁"),
            "「它」应优先解析到已知非人物焦点"
        );
        assert_eq!(
            context.resolve_pronouns("前者是谁").as_deref(),
            Some("张三是谁"),
            "「前者」= 焦点栈第 1 个（最近一次提及顺序）"
        );
        assert_eq!(
            context.resolve_pronouns("后者是谁").as_deref(),
            Some("未名科技公司是谁")
        );
        // 「该公司」只认公司/组织：焦点栈顶端是人时顺位到组织
        context.clear();
        context.record_entities_typed(&[
            ("张三".to_string(), Some("人".to_string())),
            ("未名科技公司".to_string(), Some("组织".to_string())),
        ]);
        assert_eq!(
            context.resolve_pronouns("该公司的创始人是谁").as_deref(),
            Some("未名科技公司的创始人是谁")
        );
        // 只有人物焦点时「它」让位（不替换、不乱猜）
        context.clear();
        context.record_entities_typed(&[("张三".to_string(), Some("人".to_string()))]);
        assert_eq!(
            context.resolve_pronouns("他的师傅是谁").as_deref(),
            Some("张三的师傅是谁")
        );
        assert!(
            context.resolve_pronouns("它的师傅是谁").is_none(),
            "「它」不得指代人物"
        );
    }

    #[test]
    fn test_ellipsis_completion_uses_last_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let rewritten = reasoner
            .contextualize("那李四呢")
            .expect("省略句应结合上文补全");
        assert_eq!(rewritten, "李四的师祖是什么");
    }

    #[test]
    fn test_contextualize_ignores_other_ren() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewritten = reasoner.contextualize("其他人的师傅是谁");
        assert!(rewritten.is_none() || rewritten.as_deref() == Some("其他张三的师傅是谁"));
    }

    fn attribute_facts() -> Vec<PrologFact> {
        let mut properties = HashMap::new();
        properties.insert("身高".to_string(), "226cm".to_string());
        properties.insert("体重".to_string(), "140kg".to_string());
        vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人".to_string(),
            properties,
        }]
    }

    #[test]
    fn test_frame_full_question_records_last_frame() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师祖是谁").is_none());
        assert_eq!(
            reasoner.frame_memory().last().map(|frame| frame.display()),
            Some("张三 的 师祖".to_string())
        );
    }

    #[test]
    fn test_frame_multi_hop_question_is_not_recorded() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师傅的师傅是谁").is_none());
        assert!(
            reasoner.frame_memory().last().is_none(),
            "多跳问句不是 AskAttr 帧，不应被登记"
        );
    }

    #[test]
    fn test_frame_followup_inherits_attribute_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let rewrite = reasoner
            .contextualize_with_receipt("那李四呢")
            .expect("帧合一应接管省略追问");
        assert_eq!(rewrite.text, "李四的师祖是什么");
        assert!(
            rewrite
                .notes
                .iter()
                .any(|note| note.contains("属性「师祖」←第 1 轮的完整帧")),
            "来源收据应标注属性洞的来源：{:?}",
            rewrite.notes
        );
    }

    #[test]
    fn test_frame_followup_inherits_entity_slot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewrite = reasoner
            .contextualize_with_receipt("师祖呢")
            .expect("帧合一应接管属性追问");
        assert_eq!(rewrite.text, "张三的师祖是什么");
    }

    #[test]
    fn test_frame_unknown_followup_falls_back_to_char_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewritten = reasoner
            .contextualize("某某某呢")
            .expect("未知实体仍应走省略规则");
        assert_eq!(rewritten, "某某某的师傅是什么");
    }

    #[test]
    fn test_frame_compound_defers_tail_then_pays_off() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let first = reasoner
            .contextualize_with_receipt("姚明的身高和体重是多少")
            .expect("复合问句应拆帧并先答第一问");
        assert_eq!(first.text, "姚明的身高是什么");
        assert!(
            first.notes.iter().any(|note| note.contains("挂账")),
            "第一答应提示其余挂账：{:?}",
            first.notes
        );
        assert!(reasoner.frame_memory_report().contains("姚明 的 体重"));

        let answer = reasoner
            .answer(&first.text)
            .expect("第一问应能命中属性索引");
        assert!(answer.contains("226cm"), "第一问答案：{}", answer);

        let second = reasoner
            .contextualize_with_receipt("体重呢")
            .expect("追问应结清挂账");
        assert_eq!(second.text, "姚明的体重是什么");
        assert!(
            second
                .notes
                .iter()
                .any(|note| note.contains("补全第 1 轮挂账")),
            "收据应标注挂账来源：{:?}",
            second.notes
        );
        assert!(reasoner.frame_memory().pending().is_empty());
        let answer2 = reasoner
            .answer(&second.text)
            .expect("结清的问句应能命中属性索引");
        assert!(answer2.contains("140kg"), "第二问答案：{}", answer2);
    }

    #[test]
    fn test_attribute_question_answers_from_index() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let answer = reasoner
            .answer("姚明的身高是什么")
            .expect("属性值应沿属性槽位命中");
        assert!(answer.contains("226cm"), "属性答案：{}", answer);
    }

    #[test]
    fn test_evidence_index_ranks_observed_facts_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let search = reasoner.evidence_search(
            "张三的师傅是谁",
            3,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(!search.hits.is_empty());
        assert_eq!(search.hits[0].subject, "张三");
        assert_eq!(search.hits[0].predicate, "师傅");
        assert!(search.receipt.certified_top_k);
        assert!(search
            .receipt
            .notes
            .iter()
            .any(|note| note.contains("top-k")));
    }

    #[test]
    fn test_evidence_index_honours_exact_constraints() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("师傅", 8, &constraints, SearchBudget::unlimited());
        assert!(!search.hits.is_empty());
        assert!(search.hits.iter().all(|hit| hit.predicate == "师傅"));
        assert!(search.receipt.constraint_pruned > 0);
    }

    #[test]
    fn test_evidence_fallback_answers_with_citation_and_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer_by_evidence("张三的师傅是谁")
            .expect("已锚定实体且已提及谓词时应由证据索引作答");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(answer.contains("师傅"), "{}", answer);
        assert!(answer.contains("李四"), "{}", answer);
        assert!(answer.contains("检索回执"), "{}", answer);
        assert!(answer.contains("top-k 已证明"), "{}", answer);
    }

    #[test]
    fn test_evidence_fallback_declines_without_anchored_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer_by_evidence("师傅是谁").is_none());
    }

    #[test]
    fn test_session_facts_enter_and_leave_evidence_index() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("钱七是孙八的师傅");

        let present = reasoner.evidence_search(
            "钱七 孙八 师傅",
            8,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(present.hits.iter().any(|hit| {
            (hit.subject == "钱七" || hit.object == "钱七")
                && (hit.subject == "孙八" || hit.object == "孙八")
        }));

        reasoner.retract_nl_fact("钱七是孙八的师傅");
        let absent = reasoner.evidence_search(
            "钱七 孙八 师傅",
            8,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(!absent.hits.iter().any(|hit| {
            (hit.subject == "钱七" || hit.object == "钱七")
                && (hit.subject == "孙八" || hit.object == "孙八")
        }));
    }

    #[test]
    fn test_frame_keeps_inverted_query_surface() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.register_predicate_query_surface("徒弟", "师傅");
        reasoner.contextualize("张三的徒弟是谁");
        assert_eq!(
            reasoner
                .frame_memory()
                .last()
                .and_then(|frame| frame.attribute.clone()),
            Some("徒弟".to_string()),
            "帧属性应保留逆向词面，不能塌缩成槽位「师傅」"
        );
        let rewrite = reasoner
            .contextualize_with_receipt("那李四呢")
            .expect("帧合一应保留逆向方向");
        assert_eq!(rewrite.text, "李四的徒弟是什么");
    }

    #[test]
    fn test_frame_memory_persists_across_sessions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.json");
        let state_path = path.to_str().unwrap();

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.contextualize_with_receipt("姚明的身高和体重是多少");
        reasoner.save_dialogue_state(state_path).unwrap();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&attribute_facts());
        assert!(restored.load_dialogue_state(state_path));
        let rewrite = restored
            .contextualize_with_receipt("体重呢")
            .expect("恢复后挂账应仍可结清");
        assert_eq!(rewrite.text, "姚明的体重是什么");
        assert!(rewrite
            .notes
            .iter()
            .any(|note| note.contains("第 1 轮挂账")));
    }

    fn attribute_candidates_facts() -> Vec<PrologFact> {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226cm".to_string());
        yao.insert("体重".to_string(), "140kg".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190cm".to_string());
        ye.insert("体重".to_string(), "85kg".to_string());
        vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人".to_string(),
                properties: ye,
            },
            // 已知但无体重的实体：用于「已知实体名也可作为一次显式绑定」的测试
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_frame_bare_attribute_without_context_starts_clarification() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        assert!(
            reasoner.contextualize("体重是多少").is_none(),
            "无上下文时帧通道不应凭空改写"
        );
        let prompt = reasoner
            .start_frame_clarification("体重是多少")
            .expect("无源可填的洞应转为一次显式绑定请求");
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "候选应来自知识库中拥有该属性的实体：{}",
            prompt
        );
        assert!(reasoner.frame_clarification_active());
        let report = reasoner.frame_memory_report();
        assert!(report.contains("等待绑定"), "{}", report);
        assert!(
            report.contains("1.叶莉；2.姚明"),
            "候选顺序应确定：{}",
            report
        );
        assert!(
            !prompt.contains("张三"),
            "无该属性且不在焦点的实体不应凭空成为候选：{}",
            prompt
        );
    }

    #[test]
    fn test_frame_clarification_binding_replays_and_remembers() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner
            .start_frame_clarification("体重是多少")
            .expect("应开启澄清");
        let outcome = reasoner
            .answer_frame_clarification("2")
            .expect("序号应被确定性解析为一次绑定");
        let question = match outcome {
            dialogue_frame::ClarificationOutcome::Bound { question, note } => {
                assert!(note.contains("第 1 轮澄清"), "{}", note);
                question
            }
            other => panic!("应为绑定：{:?}", other),
        };
        assert_eq!(question, "姚明的体重是什么");
        assert!(!reasoner.frame_clarification_active(), "绑定后澄清应关闭");
        let answer = reasoner.answer(&question).expect("重放问句应可作答");
        assert!(answer.contains("140kg"), "{}", answer);

        // 记忆回写：重放的问句成为上一帧，后续省略追问自然继承
        assert!(reasoner.contextualize(&question).is_none());
        let followup = reasoner
            .contextualize_with_receipt("身高呢")
            .expect("绑定后的实体槽应可被追问继承");
        assert_eq!(followup.text, "姚明的身高是什么");
    }

    #[test]
    fn test_frame_clarification_accepts_known_entity_outside_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("身高是多少");
        let prompt = reasoner
            .start_frame_clarification("身高是多少")
            .expect("应开启澄清");
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "{}",
            prompt
        );
        // 候选表外的已知实体名（张三没有身高记录）同样接受：绑定决定权在用户
        let outcome = reasoner
            .answer_frame_clarification("张三")
            .expect("已知实体名应可作为显式绑定");
        match outcome {
            dialogue_frame::ClarificationOutcome::Bound { question, .. } => {
                assert_eq!(question, "张三的身高是什么");
            }
            other => panic!("应为绑定：{:?}", other),
        }
    }

    #[test]
    fn test_frame_clarification_reprompts_out_of_range_and_passes_natural_language() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner.start_frame_clarification("体重是多少").unwrap();
        // 越界序号：重问，零状态变化
        match reasoner.answer_frame_clarification("9") {
            Some(dialogue_frame::ClarificationOutcome::Reprompt { prompt }) => {
                assert!(prompt.contains("姚明"), "{}", prompt);
            }
            other => panic!("越界序号应重问：{:?}", other),
        }
        assert!(reasoner.frame_clarification_active());
        // 自然语言非回答：放行给常规管线，澄清保持
        assert!(
            reasoner
                .answer_frame_clarification("北京是哪个省的")
                .is_none(),
            "无法确定性解析的输入应放行"
        );
        assert!(reasoner.frame_clarification_active());
        // 都不是：放弃
        match reasoner.answer_frame_clarification("都不是") {
            Some(dialogue_frame::ClarificationOutcome::Cancelled { .. }) => {}
            other => panic!("应取消：{:?}", other),
        }
        assert!(!reasoner.frame_clarification_active());
        assert!(reasoner.frame_memory().last().is_none());
    }

    #[test]
    fn test_frame_clarification_not_started_when_existing_channel_can_fill() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("姚明的身高是什么");
        assert!(
            reasoner.start_frame_clarification("体重是多少").is_none(),
            "有上一帧时应走合一补全，不另起澄清"
        );
        let rewrite = reasoner
            .contextualize_with_receipt("体重是多少")
            .expect("合一应补全实体洞");
        assert_eq!(rewrite.text, "姚明的体重是什么");
        assert!(!reasoner.frame_clarification_active());
        // 「呢」省略句属于常规通道，帧澄清不得抢答
        assert!(reasoner.start_frame_clarification("体重呢").is_none());
    }

    #[test]
    fn test_frame_clarification_expires_and_persists_across_sessions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.json");
        let state_path = path.to_str().unwrap();

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner.start_frame_clarification("体重是多少").unwrap();
        reasoner.save_dialogue_state(state_path).unwrap();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&attribute_candidates_facts());
        assert!(restored.load_dialogue_state(state_path));
        assert!(restored.frame_clarification_active(), "澄清应跨会话恢复");
        match restored.answer_frame_clarification("1") {
            Some(dialogue_frame::ClarificationOutcome::Bound { question, .. }) => {
                assert_eq!(question, "叶莉的体重是什么");
            }
            other => panic!("恢复后应可绑定：{:?}", other),
        }

        // TTL：3 轮内有效，超过后过期（零残留）
        let mut expiring = SymbolicReasoner::new(test_config(), HashMap::new());
        expiring.observe_facts(&attribute_candidates_facts());
        expiring.contextualize("体重是多少");
        expiring.start_frame_clarification("体重是多少").unwrap();
        for _ in 0..3 {
            let _ = expiring.contextualize("姚明的身高是什么");
        }
        assert!(
            !expiring.frame_clarification_active(),
            "超过 TTL 的澄清应过期"
        );
        assert!(expiring.frame_memory_report().contains("过期 1"));
    }

    #[test]
    fn test_answer_generates_prediction_with_explanation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的师祖是谁")
            .expect("应能通过组合规则预测出师祖");
        assert!(answer.contains("赵六"));
        assert!(answer.contains("师傅"));
        assert!(answer.contains("置信度"));
    }

    #[test]
    fn test_answer_requires_mentioned_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer("李四是谁").is_none());
    }

    #[test]
    fn test_answer_returns_none_without_facts() {
        let reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert!(reasoner.answer("李四的师祖是谁").is_none());
    }

    #[test]
    fn test_synonym_alias_matches_predicate() {
        let mut synonyms = HashMap::new();
        synonyms.insert("祖师".to_string(), "师祖".to_string());
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的祖师是谁")
            .expect("同义词别名应能命中师祖关系");
        assert!(answer.contains("赵六"));
    }

    #[test]
    fn test_alignment_and_rules_reports() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let rules_report = reasoner.rules_report();
        assert!(rules_report.contains("师傅"));
        let alignment_report = reasoner.alignment_report();
        assert!(alignment_report.contains("本体对齐报告"));
    }

    fn ambiguous_attribute_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果公司".to_string(),
                to: "甲公司".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果果实".to_string(),
                to: "红富士".to_string(),
            },
        ]
    }

    fn ambiguous_attribute_reasoner() -> SymbolicReasoner {
        let mut config = test_config();
        config.ontology_alignment_threshold = 0.5;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&ambiguous_attribute_facts());
        reasoner
    }

    #[test]
    fn test_ambiguous_attribute_alignment_asks_instead_of_guessing() {
        let reasoner = ambiguous_attribute_reasoner();
        let answer = reasoner
            .answer("张三的苹果是谁")
            .expect("两个候选谓词都有答案，应进入歧义决策");
        assert!(answer.contains("不止一种理解"), "实际回答：{}", answer);
        assert!(answer.contains("苹果公司"), "实际回答：{}", answer);
        assert!(answer.contains("苹果果实"), "实际回答：{}", answer);
        assert!(answer.contains("甲公司") && answer.contains("红富士"));
    }

    #[test]
    fn test_equivalent_attribute_candidates_do_not_ask() {
        let mut config = test_config();
        config.ontology_alignment_threshold = 0.5;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果公司".to_string(),
                to: "红色".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果果实".to_string(),
                to: "红色".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("张三的苹果是谁")
            .expect("答案等价的候选应静默选一个");
        assert!(!answer.contains("不止一种理解"), "实际回答：{}", answer);
        assert!(answer.contains("红色"), "实际回答：{}", answer);
    }

    #[test]
    fn test_alignment_candidates_report_explains_ambiguity() {
        let mut reasoner = ambiguous_attribute_reasoner();
        reasoner.context.record_entities(&["张三".to_string()]);
        let report = reasoner.alignment_candidates_report("苹果");
        assert!(report.contains("有歧义"), "{}", report);
        assert!(report.contains("苹果公司"), "{}", report);
        assert!(report.contains("苹果果实"), "{}", report);
        assert!(report.contains("答案等价性"), "{}", report);
        assert!(report.contains("值得澄清"), "{}", report);
    }

    #[test]
    fn test_planned_multi_hop_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问题应被逻辑计划执行");
        assert!(answer.contains("王五"));
        assert!(answer.contains("多跳"));
    }

    #[test]
    fn test_planned_multi_hop_answer_keeps_path_identity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问句应有路径答案");
        assert!(answer.contains("赵六"), "{}", answer);
        assert!(answer.contains("多跳"), "{}", answer);
        assert!(answer.contains("共 2 条"), "{}", answer);
        assert!(answer.contains("张三 --师傅--> 李四"), "{}", answer);
        assert!(answer.contains("张三 --师傅--> 王五"), "{}", answer);
    }

    #[test]
    fn test_planned_path_between_and_common_neighbor() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "赵六".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
        ]);
        let between = reasoner
            .answer("张三和王五之间是什么关系")
            .expect("路径问句应返回见证路径");
        assert!(
            between.contains("李四") && between.contains("赵六"),
            "{}",
            between
        );
        assert!(between.contains("路径"), "{}", between);

        let common = reasoner
            .answer("张三和赵六共同认识谁")
            .expect("共同邻居问句应返回中间实体");
        assert!(common.contains("王五"), "{}", common);

        let intermediary = reasoner
            .answer("张三通过谁认识王五")
            .expect("中间人问句应返回中间实体");
        assert!(
            intermediary.contains("李四") || intermediary.contains("赵六"),
            "{}",
            intermediary
        );
    }

    #[test]
    fn test_planned_count_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四有几个师傅")
            .expect("计数问题应被逻辑计划执行");
        assert!(answer.contains("共有 1 个"));
        assert!(answer.contains("王五"));
    }

    #[test]
    fn test_planned_yes_no_true_and_false() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let yes = reasoner
            .answer("李四的师傅是王五吗")
            .expect("是非问题应被逻辑计划执行");
        assert!(yes.starts_with("是的"), "实际回答：{}", yes);
        let no = reasoner
            .answer("李四的师傅是张三吗")
            .expect("错误的断言也应给出否定回答");
        assert!(no.starts_with("不是"), "实际回答：{}", no);
        assert!(no.contains("王五"));
    }

    #[test]
    fn test_planned_find_subject() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("谁的师傅是李四")
            .expect("反查主体应被逻辑计划执行");
        assert!(answer.contains("张三"));
    }

    #[test]
    fn test_planned_answer_reports_missing_record() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("赵六的师祖是谁")
            .expect("已知实体与已知谓词应给出明确的无记录回答");
        assert!(answer.contains("没有记录"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_exist_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三有没有师傅")
            .expect("存在性问题应被逻辑计划执行");
        assert!(answer.starts_with("是的"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_similar_entity_retrieval() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("谁和张三最像")
            .expect("近邻检索应命中最相似的实体");
        assert!(answer.contains("王五"));
        assert!(answer.contains("相似度"));
    }

    #[test]
    fn test_planned_analogy_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("张三对应李四，王五对应谁")
            .expect("类比问题应由向量算术回答");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_similarity_report_handles_unknown_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let report = reasoner.similarity_report("不存在的人");
        assert!(report.contains("没有"));
        let report = reasoner.similarity_report("张三");
        assert!(report.contains("王五"));
    }

    #[test]
    fn test_dialogue_state_persistence_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        reasoner.contextualize("那李四呢");
        let path =
            std::env::temp_dir().join(format!("knowprolog_dialogue_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner
            .save_dialogue_state(&path_str)
            .expect("应能保存对话状态");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.context().focus(), Some("李四"));
        assert_eq!(restored.context().last_predicate(), Some("师祖"));
        let rewritten = restored
            .contextualize("他的师祖是谁")
            .expect("恢复后的记忆应支持指代消解");
        assert!(rewritten.contains("李四"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dialogue_state_persistence_is_canonical_bytes_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        reasoner.contextualize("那李四呢");
        let path = std::env::temp_dir().join(format!(
            "knowprolog_canonical_state_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let bytes = std::fs::read(&path).unwrap();
        let (snapshot, receipt) = SymbolicReasoner::read_dialogue_state(&path_str).unwrap();
        assert!(receipt.canonical, "落盘必须是规范字节");
        assert_eq!(receipt.kind, "dialogue_state");
        // L1/L2/L3/L4：往返恒等 / 字节不动点 / 编码幂等 / 摘要交换
        assert_eq!(canonical::canonical_bytes(&snapshot).unwrap(), bytes);
        assert_eq!(
            canonical::canonical_digest(&snapshot).unwrap(),
            canonical::canonical_digest(&reasoner.dialogue_snapshot()).unwrap()
        );
        let (again, _) = SymbolicReasoner::read_dialogue_state(&path_str).unwrap();
        assert_eq!(
            canonical::canonical_bytes(&again).unwrap(),
            canonical::canonical_bytes(&snapshot).unwrap()
        );

        // 坏字节给 parse 码；缺失文件给 io 码（原因可回答）
        assert_eq!(
            SymbolicReasoner::read_dialogue_state("/nonexistent/state.json")
                .unwrap_err()
                .code(),
            "io"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_dialogue_summary_mentions_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let summary = reasoner.dialogue_summary();
        assert!(summary.contains("张三"));
        assert!(summary.contains("师祖"));
    }

    #[test]
    fn test_embedding_feeds_from_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        assert!(reasoner.embedding().is_embedded("张三"));
        assert!(reasoner.embedding().is_embedded("赵六"));
        assert!(reasoner.embedding().similarity("张三", "王五") > 0.9);
    }

    #[test]
    fn test_entity_link_notes_for_typo_mention() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let notes = reasoner.entity_link_notes("张山的师傅是谁");
        assert!(
            notes
                .iter()
                .any(|note| note.contains("张三") && note.contains("→")),
            "错别字指称应产生模糊链接说明：{:?}",
            notes
        );
        assert!(reasoner.linker().is_known_entity("张三"));
    }

    #[test]
    fn test_linked_entity_fallback_recovers_typo_query() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李寺的师祖是谁")
            .expect("错别字实体应能通过链接回退命中推理");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_generate_about_combines_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("师祖"));
        assert!(profile.contains("李四"));
        assert!(reasoner.generate_about("不存在的人").is_none());
    }

    #[test]
    fn test_narrate_uses_owl_definition_and_dcg_variants() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut properties = HashMap::new();
        properties.insert("职业".to_string(), "教师".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "北京大学".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "清华大学".to_string(),
            },
        ];
        reasoner.observe_facts(&facts);
        let text = reasoner.narrate("张三").expect("应能生成符号主义叙述");
        assert!(text.contains("是一种人"), "应包含 OWL 定义开场：{}", text);
        assert!(
            text.contains("北京大学") && text.contains("清华大学") && text.contains("、"),
            "多值事实应聚合：{}",
            text
        );
        assert!(reasoner.narrate("不存在的人").is_none());

        let variants = reasoner
            .paraphrase_fact("张三", "职业", 4)
            .expect("应能改写事实句式");
        assert!(
            variants.lines().count() >= 2,
            "应有多种合规句式：{}",
            variants
        );
        assert!(variants.contains("教师"));
    }

    #[test]
    fn test_teach_ontology_axiom_flows_into_narration() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }]);
        let summary = reasoner
            .teach_ontology("人是一种动物")
            .expect("本体公理应可学习");
        assert!(
            summary.contains("动物"),
            "教学回执应包含公理内容：{}",
            summary
        );
        let text = reasoner.narrate("苏格拉底").expect("叙述应可用");
        assert!(
            text.contains("动物"),
            "本体继承推理应进入叙述（定义/推导链）：{}",
            text
        );
    }

    /// 教学本体 → 快照胶囊：未教学时快照不携带胶囊（旧字节兼容），
    /// 教学后胶囊随快照走，且能通过自身的版本门与摘要校验。
    #[test]
    fn test_taught_ontology_lands_in_dialogue_snapshot_capsule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }]);
        let plain = reasoner.dialogue_snapshot();
        assert!(
            plain.ontology.is_none(),
            "未教学时不应携带本体胶囊（旧快照字节保持不变）"
        );
        let plain_text = String::from_utf8(canonical::canonical_bytes(&plain).unwrap()).unwrap();
        assert!(
            !plain_text.contains("ontology"),
            "无胶囊时规范字节不得出现 ontology 键：{}",
            plain_text
        );

        reasoner
            .teach_ontology("人是一种动物")
            .expect("本体公理应可学习");
        let snapshot = reasoner.dialogue_snapshot();
        let capsule = snapshot.ontology.expect("教学后快照应携带本体胶囊");
        capsule.verify().expect("胶囊应通过自身校验");
        assert_eq!(capsule.version, 1);
        assert!(!capsule.closure_digest.is_empty(), "闭包缓存应带摘要");
        assert!(
            capsule.closure_matches(reasoner.discourse.narrator_mut().closure()),
            "导出缓存应与当前闭包一致"
        );
    }

    /// 快照恢复到全新推理器：教过的本体 100% 保留（理解 = 反序列化）。
    #[test]
    fn test_taught_ontology_survives_snapshot_restore_on_fresh_reasoner() {
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        let snapshot = source.dialogue_snapshot();

        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("胶囊应通过校验并恢复");
        // 幂等：同一快照恢复两次，快照内容不变
        let after_first =
            canonical::canonical_digest(&target.dialogue_snapshot()).expect("快照可摘要");
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("重复恢复应幂等成功");
        let after_second =
            canonical::canonical_digest(&target.dialogue_snapshot()).expect("快照可摘要");
        assert_eq!(after_first, after_second, "重复恢复必须幂等");
        let text = target.narrate("苏格拉底").expect("叙述应可用");
        assert!(
            text.contains("动物"),
            "恢复后本体继承推理必须仍在：{}",
            text
        );
    }

    /// 重建基座（rebuild_from_graph）后教过的本体不得消失。
    #[test]
    fn test_taught_ontology_survives_rebuild_from_graph() {
        let facts = vec![
            PrologFact::Entity {
                name: "苏格拉底".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "苏格拉底".to_string(),
                rel_type: "徒弟".to_string(),
                to: "柏拉图".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        reasoner.teach_ontology("人是一种动物").expect("可教学");
        reasoner.rebuild_from_graph(&facts);
        let text = reasoner.narrate("苏格拉底").expect("重建后叙述应可用");
        assert!(
            text.contains("动物"),
            "重建图谱后教过的本体必须保留：{}",
            text
        );
    }

    /// 落盘 → 新进程读回：教过的本体随规范字节持久化。
    #[test]
    fn test_taught_ontology_survives_dialogue_state_file_roundtrip() {
        let dir = tempfile::TempDir::new().expect("临时目录");
        let state_path = dir.path().join("dialogue_state.json");
        let state_path_str = state_path.to_str().expect("路径可表示");
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        source
            .save_dialogue_state(state_path_str)
            .expect("应能保存");

        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        assert!(
            target.load_dialogue_state(state_path_str),
            "含胶囊的对话状态应可加载"
        );
        let text = target.narrate("苏格拉底").expect("叙述应可用");
        assert!(text.contains("动物"), "落盘恢复后本体应保留：{}", text);
    }

    /// 被改写 / 超前版本的胶囊：整体拒绝，零半应用。
    #[test]
    fn test_tampered_ontology_capsule_is_rejected_without_partial_apply() {
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        source.context.record_question("苏格拉底是谁");
        let mut snapshot = source.dialogue_snapshot();

        let mut tampered = snapshot.clone();
        {
            let capsule = tampered.ontology.as_mut().expect("应有胶囊");
            capsule.lessons.clear();
        }
        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        let error = target
            .restore_dialogue_snapshot(&tampered)
            .expect_err("改写公理必须被拒绝");
        assert!(error.contains("摘要"), "错误应说明摘要不匹配：{}", error);
        assert!(
            target.dialogue_snapshot().ontology.is_none(),
            "拒绝后不得留下半应用的胶囊"
        );
        assert!(
            target.context.history().is_empty(),
            "拒绝后其余会话态也不得被应用（零半应用）"
        );

        let mut future = snapshot.clone();
        future.ontology.as_mut().expect("应有胶囊").version = 99;
        let error = target
            .restore_dialogue_snapshot(&future)
            .expect_err("超前版本必须被版本门拒绝");
        assert!(error.contains("版本"), "错误应说明版本门：{}", error);

        snapshot.ontology = None;
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("无胶囊快照不受影响");
    }

    /// 类型化配方：逆方向查询、子类扩展、开放世界类型过滤与依据复述。
    #[test]
    fn recipe_lookup_uses_inverse_subclass_and_open_world_types() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "千与千寻".into(),
                entity_type: "动画电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "宫崎骏".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
            PrologFact::Relation {
                from: "千与千寻".into(),
                rel_type: "导演".into(),
                to: "宫崎骏".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "动画电影".into(),
                sup: "电影".into(),
                original: "动画电影是一种电影".into(),
            },
            OntologyLesson::Subclass {
                sub: "电影".into(),
                sup: "影视作品".into(),
                original: "电影是一种影视作品".into(),
            },
            OntologyLesson::Disjoint {
                left: "电影".into(),
                right: "人物".into(),
                original: "电影和人物互斥".into(),
            },
            OntologyLesson::Domain {
                property: "导演".into(),
                class: "电影".into(),
                original: "导演关系的定义域是电影".into(),
            },
            OntologyLesson::Range {
                property: "导演".into(),
                class: "人物".into(),
                original: "导演关系的值域是人物".into(),
            },
            OntologyLesson::Inverse {
                left: "导演".into(),
                right: "执导".into(),
                original: "导演和执导互为逆关系".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let forward = reasoner
            .recipe_lookup("阿凡达", "导演", RecipeDirection::Forward, &["人物".into()])
            .expect("正向应命中导演");
        assert_eq!(forward.values, vec!["卡梅隆".to_string()]);
        assert!(
            forward.evidence.iter().any(|line| line.contains("导演")),
            "依据必须复述事实边：{:?}",
            forward.evidence
        );

        let inverse = reasoner
            .recipe_lookup("宫崎骏", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("逆向应命中作品");
        assert_eq!(
            inverse.values,
            vec!["千与千寻".to_string()],
            "动画电影 ⊑ 电影，答案类型必须沿子类扩展"
        );

        let unknown = reasoner
            .recipe_lookup(
                "卡梅隆",
                "导演",
                RecipeDirection::Inverse,
                &["未知类型".into()],
            )
            .expect("开放世界：未知类型必须放行");
        assert_eq!(unknown.values, vec!["阿凡达".to_string()]);

        assert!(
            reasoner
                .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["人物".into()])
                .is_none(),
            "已知互斥类型（电影 ⊥ 人物）必须拒答"
        );
    }

    /// 主语侧类型约束：正向问句查定义域、逆向问句查值域；由关系反推的
    /// 循环类型不作数，未知类型放行，已知互斥才拒答。
    #[test]
    fn recipe_lookup_gates_subject_types_by_domain_and_range() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
            // 反向错录：把人物记成了导演关系的主语、把电影记成了宾语——
            // 同名关系用在了类型不符的实体上，主语侧门控必须挡住两侧误答。
            PrologFact::Relation {
                from: "卡梅隆".into(),
                rel_type: "导演".into(),
                to: "阿凡达".into(),
            },
            // 类型完全未知的关系主语：开放世界必须放行而不是猜测拒答。
            PrologFact::Relation {
                from: "某无名氏".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Domain {
                property: "导演".into(),
                class: "电影".into(),
                original: "导演关系的定义域是电影".into(),
            },
            OntologyLesson::Range {
                property: "导演".into(),
                class: "人物".into(),
                original: "导演关系的值域是人物".into(),
            },
            OntologyLesson::Disjoint {
                left: "电影".into(),
                right: "人物".into(),
                original: "电影和人物互斥".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let forward = reasoner
            .recipe_lookup("阿凡达", "导演", RecipeDirection::Forward, &["人物".into()])
            .expect("声明类型与定义域相符的正向查询应命中");
        assert_eq!(forward.values, vec!["卡梅隆".to_string()]);

        let inverse = reasoner
            .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("声明类型与值域相符的逆向查询应命中");
        assert!(inverse.values.contains(&"阿凡达".to_string()));

        assert!(
            reasoner
                .recipe_lookup("卡梅隆", "导演", RecipeDirection::Forward, &["人物".into()])
                .is_none(),
            "人物作为导演关系的主语与定义域电影互斥，正向误答必须拒答"
        );
        assert!(
            reasoner
                .recipe_lookup("阿凡达", "导演", RecipeDirection::Inverse, &["电影".into()])
                .is_none(),
            "电影作为导演关系的宾语与值域人物互斥，逆向误答必须拒答"
        );

        let unknown = reasoner
            .recipe_lookup(
                "某无名氏",
                "导演",
                RecipeDirection::Forward,
                &["人物".into()],
            )
            .expect("有效类型未知（关系反推的循环类型不作数）时必须放行");
        assert_eq!(unknown.values, vec!["卡梅隆".to_string()]);
    }

    /// 配方执行单一入口：CLI 与 HTTP 共用；角色指称接续焦点，依据随模板接地。
    #[test]
    fn answer_plugin_recipe_resolves_role_reference_and_grounds_evidence() {
        use crate::plugin::{
            PluginManifest, PluginRegistry, PluginRuntime, PluginSettings,
        };

        let manifest: PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "reference"
id = "film.reference"
quantifiers = ["这位", "那位"]
role_words = ["导演"]

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}还拍过什么", "那{entity}还拍过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"

[[contributions]]
kind = "recipe"
id = "film.directed_by"
patterns = ["谁导演了{entity}"]
question = "{entity}的导演是什么"
attribute = "导演"
direction = "forward"
answer_types = ["人物"]
template = "film.directed.template"

[[contributions]]
kind = "template"
id = "film.directed.template"
template = "「{entity}」的导演是{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let outcome = runtime.rewrite("这位导演还拍过什么");
        assert!(outcome.trace.recipe.is_some(), "配方应命中角色指称问法");
        let reply = reasoner
            .answer_plugin_recipe(&outcome, &mut runtime, Some("卡梅隆"))
            .expect("焦点实体应能完成逆向配方");
        assert_eq!(reply.entity.as_deref(), Some("卡梅隆"));
        assert_eq!(reply.question, "卡梅隆导演过哪些作品");
        assert_eq!(
            reply.focus.as_deref(),
            Some("卡梅隆"),
            "逆向问句的焦点应留在被查询的主语"
        );
        assert!(reply.text.contains("阿凡达"), "{}", reply.text);
        assert!(reply.text.contains("依据"), "{}", reply.text);
        assert!(
            reply.text.contains("属于 电影"),
            "依据应复述答案的类型约束：{}",
            reply.text
        );

        let forward = runtime.rewrite("谁导演了阿凡达");
        let forward_reply = reasoner
            .answer_plugin_recipe(&forward, &mut runtime, None)
            .expect("正向配方应命中");
        assert_eq!(
            forward_reply.focus.as_deref(),
            Some("卡梅隆"),
            "正向问句的焦点应转到答案，供「这位导演」继续追问"
        );
        assert!(
            reply.plan.iter().any(|plan| plan.contains("film.works_by_director")),
            "{:?}",
            reply.plan
        );
    }

    /// 别名公理让历史数据无需重写：谓词别名沿子关系归一，类型别名沿子类归一。
    #[test]
    fn recipe_lookup_reaches_alias_predicates_and_types() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "影片".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "director".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "影片".into(),
                sup: "电影".into(),
                original: "影片是电影的别名".into(),
            },
            OntologyLesson::Subproperty {
                sub: "director".into(),
                sup: "导演".into(),
                original: "director是导演的别名".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let inverse = reasoner
            .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("别名谓词与别名类型都应被规范词面命中");
        assert_eq!(inverse.values, vec!["阿凡达".to_string()]);
        assert!(
            inverse.evidence.iter().any(|line| line.contains("director ⊑ 导演")),
            "依据应说明别名归一：{:?}",
            inverse.evidence
        );
    }

    /// 运行态启用插件时增量合并本体：首次返回新增数，重复应用为零。
    #[test]
    fn applying_plugin_runtime_is_incremental_and_idempotent() {
        use crate::plugin::{PluginManifest, PluginRegistry, PluginRuntime, PluginSettings};

        let manifest: PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "ontology"
id = "film.ontology"
predicates = [
  { name = "导演", aliases = ["director"], inverse = "执导", domain = ["电影"], range = ["人物"] },
]
types = [
  { name = "电影", aliases = ["影片"] },
]
"#,
        )
        .expect("测试清单应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        let runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());

        let first = reasoner.apply_plugin_runtime(&runtime);
        assert_eq!(first.aliases, 1, "谓词别名 director");
        assert_eq!(first.inverse_surfaces, 1, "互逆词面 执导");
        assert_eq!(first.lessons, 5, "子类1 + 别名1 + 定义域1 + 值域1 + 逆1");

        assert_eq!(
            reasoner.apply_plugin_runtime(&runtime),
            PluginOntologySync::default(),
            "同一运行态重复应用必须无新增"
        );
    }

    /// 配置本体并入单一本体：注入幂等、与教学公理同簿记、
    /// 随快照/重建/分叉一并保留（配置不再是第二套本体）。
    #[test]
    fn test_configure_ontology_program_merges_into_single_program() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let types = TypeSystemConfig {
            subclasses: vec![TypeSubclassConfig {
                sub: "学生".into(),
                sup: "人物".into(),
            }],
            ..TypeSystemConfig::default()
        };
        let profiles = vec![PredicateProfileConfig {
            name: "师傅".into(),
            transitive: true,
            domain: vec!["人物".into()],
            ..PredicateProfileConfig::default()
        }];
        let program = OntologyProgram::from_config(&types, &profiles);

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert_eq!(
            reasoner.configure_ontology_program(&program),
            3,
            "子类1 + 定义域1 + 传递1"
        );
        assert_eq!(
            reasoner.configure_ontology_program(&program),
            0,
            "重复注入必须按整句去重"
        );
        reasoner
            .teach_ontology("学生是一种人物")
            .expect("教学同一条公理应成功");
        assert_eq!(
            reasoner.taught_ontology_lessons().len(),
            3,
            "配置与教学共用一份公理簿（同原句去重）"
        );

        // 配置本体随快照走：验证 = 反序列化
        let snapshot = reasoner.dialogue_snapshot();
        let capsule = snapshot.ontology.clone().expect("配置本体应随快照携带");
        capsule.verify().expect("胶囊应通过校验");
        assert_eq!(capsule.lessons.len(), 3);

        let mut fresh = SymbolicReasoner::new(test_config(), HashMap::new());
        fresh
            .restore_dialogue_snapshot(&snapshot)
            .expect("配置本体应可恢复");
        assert_eq!(
            fresh.taught_ontology_lessons().len(),
            3,
            "快照恢复后公理全在"
        );

        fresh.rebuild_from_graph(&master_facts());
        assert_eq!(
            fresh.taught_ontology_lessons().len(),
            3,
            "重建图谱后配置本体不得丢"
        );

        let fork = fresh.fork_session();
        assert_eq!(
            fork.taught_ontology_lessons().len(),
            3,
            "会话分叉保留单一本体"
        );
    }

    /// 单一本体 → 检索类型层级：教学的子类/互斥公理进入稠密检索的
    /// 类型约束（RDFS 展开；互斥是唯一硬剪证据）。
    #[test]
    fn test_taught_ontology_expands_retrieval_type_constraints() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "苏格拉底".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "雅典".to_string(),
                entity_type: "地点".to_string(),
                properties: HashMap::new(),
            },
        ]);
        reasoner.teach_ontology("人是一种动物").expect("可教学");
        reasoner.teach_ontology("地点和动物互斥").expect("可教学");

        let constraints = SearchConstraints {
            entity_type: Some("动物".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("", 8, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "苏格拉底"),
            "人 ⊑ 动物：沿教学子类边应命中类型约束：{:?}",
            search.hits
        );
        assert!(
            search.hits.iter().all(|hit| hit.subject != "雅典"),
            "互斥证据必须硬剪：{:?}",
            search.hits
        );
    }

    /// 配置态子类（config.toml 的 type_system）经单一本体进入检索类型层级。
    #[test]
    fn test_configured_subclass_expands_retrieval_type_constraints() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "学生".to_string(),
            properties: HashMap::new(),
        }]);
        let program = OntologyProgram::from_config(
            &TypeSystemConfig {
                subclasses: vec![TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[],
        );
        assert!(reasoner.configure_ontology_program(&program) > 0);
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("", 8, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "张三"),
            "配置子类（学生 ⊑ 人物）应展开类型约束：{:?}",
            search.hits
        );
    }

    #[test]
    fn test_check_statement_entailed_contradicted_unknown() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let entailed = reasoner.check_statement("张三的师傅是李四");
        assert_eq!(entailed.len(), 1);
        assert_eq!(entailed[0].verdict, Verdict::Entailed);
        assert_eq!(
            entailed[0].proof[0].kind,
            semantic_entailment::ProofKind::DirectFact
        );

        let contradicted = reasoner.check_statement("李四的师傅是张三");
        assert_eq!(contradicted[0].verdict, Verdict::Contradicted);
        assert!(contradicted[0]
            .counter_evidence
            .iter()
            .any(|evidence| evidence.contains("王五")));

        let unknown = reasoner.check_statement("钱七的师傅是李四");
        assert_eq!(unknown[0].verdict, Verdict::Unknown);
        assert!(
            unknown[0].reason.contains("尚未收录"),
            "实际原因：{}",
            unknown[0].reason
        );

        assert!(reasoner.check_statement("张三的师傅是谁").is_empty());
    }

    #[test]
    fn test_check_statement_recovers_typo_via_linker() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let results = reasoner.check_statement("张山的师傅是李四");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verdict, Verdict::Entailed);
        assert!(
            results[0]
                .proposition
                .notes
                .iter()
                .any(|note| note.contains("模糊链接") && note.contains("张三")),
            "解析证据应记录容错链接：{:?}",
            results[0].proposition.notes
        );
    }

    #[test]
    fn test_entailment_report_renders_logical_form_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.entailment_report("张三的师傅是李四");
        assert!(report.contains("逻辑形式：relation(张三, 师傅, 李四)"));
        assert!(report.contains("蕴含"));
        assert!(report.contains("直接事实"));

        let failure = reasoner.entailment_report("今天天气如何");
        assert!(failure.contains("形式化失败"));
    }

    #[test]
    fn test_generate_about_appends_entailment_self_check() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("生成自检"), "应附蕴含自检：{}", profile);
        assert!(profile.contains("零幻觉") || profile.contains("未通过"));
        assert_eq!(
            reasoner
                .context()
                .generation_state()
                .map(|s| s.entity.as_str()),
            Some("张三")
        );
    }

    #[test]
    fn test_generation_self_check_with_bare_english_predicates() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "张三".to_string()),
                    ("age".to_string(), "25".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "李四".to_string()),
                    ("age".to_string(), "30".to_string()),
                ]),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "knows".to_string(),
                to: "李四".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let profile = reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(profile.contains("张三认识李四"), "{}", profile);
        assert!(
            profile.contains("零幻觉"),
            "裸谓词生成应通过自检：{}",
            profile
        );
    }

    /// 英文属性键（JSON-LD 常见）应能按中文词面直接检索，且中文键仍优先。
    #[test]
    fn test_english_attribute_keys_answer_chinese_labels() {
        let facts = vec![PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "张三".to_string()),
                ("age".to_string(), "25".to_string()),
                ("height".to_string(), "180".to_string()),
            ]),
        }];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let age = reasoner.answer("张三的年龄是多少").expect("中文标签应命中 age");
        assert!(age.contains("25"), "{}", age);
        let height = reasoner
            .answer("张三的身高是多少")
            .expect("中文标签应命中 height");
        assert!(height.contains("180"), "{}", height);
        let english = reasoner.answer("张三的age是多少").expect("英文键也应可查");
        assert!(english.contains("25"), "{}", english);
    }

    /// literal 属性（不进关系三元组）也必须支持「谁的X是Y」反查主体，
    /// 且只回属性值真正相等的那个实体；英文属性键同样按中文词面反查。
    #[test]
    fn test_find_subject_by_attribute_value() {
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("身高".to_string(), "226".to_string()),
                    ("国籍".to_string(), "中国".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("身高".to_string(), "190".to_string())]),
            },
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("height".to_string(), "180".to_string())]),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let hit = reasoner
            .answer("谁的身高是226")
            .expect("属性反查应给出主体");
        assert!(hit.contains("姚明"), "{}", hit);
        assert!(!hit.contains("叶莉"), "{}", hit);

        let chinese_key = reasoner
            .answer("谁的国籍是中国")
            .expect("文本型属性同样可反查");
        assert!(chinese_key.contains("姚明"), "{}", chinese_key);

        let english_key = reasoner
            .answer("谁的身高是180")
            .expect("英文属性键按中文词面反查");
        assert!(english_key.contains("张三"), "{}", english_key);

        assert!(
            reasoner.answer("谁的身高是999").is_none(),
            "没有该取值时不应编造主体"
        );
    }

    /// 批量对齐入口（交互与服务共用）：别名/换向词面即时参与回答，
    /// 实体持久别名与规范名同权进入链接；退化与重复条目不计入生效数。
    #[test]
    fn test_batch_alignment_entry_points() {
        let facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let applied = reasoner.align_predicate_pairs(&[
            ("老师".to_string(), "师傅".to_string(), false),
            ("徒弟".to_string(), "师傅".to_string(), true),
            ("老师".to_string(), "老师".to_string(), false),
            ("  ".to_string(), "师傅".to_string(), false),
        ]);
        assert_eq!(applied, 2, "自指与空词面不计生效");

        let forward = reasoner
            .answer("张三的老师是谁")
            .expect("正向别名应命中师傅事实");
        assert!(forward.contains("李四"), "{}", forward);

        let inverse = reasoner
            .answer("李四的徒弟是谁")
            .expect("换向词面应按反方向检索");
        assert!(inverse.contains("张三"), "{}", inverse);

        let entities = reasoner.align_entity_aliases(&[
            ("明哥".to_string(), "姚明".to_string()),
            ("明哥".to_string(), "明哥".to_string()),
        ]);
        assert_eq!(entities, 1, "自指别名不计生效");
        assert_eq!(
            reasoner.best_linked_entity("明哥").as_deref(),
            Some("姚明"),
            "持久别名应与规范名同权参与链接"
        );
        let alias_answer = reasoner
            .answer("明哥的身高是多少")
            .expect("已确认的实体别名应能直接作答");
        assert!(alias_answer.contains("226"), "{}", alias_answer);
    }

    #[test]
    fn test_continue_generation_and_switch_angle() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());

        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        assert!(first.contains("共 1 条事实"), "实际：{}", first);
        let second = reasoner.continue_generation().expect("接着说应返回下一页");
        assert!(second.contains("续：第 2–2 句"), "实际：{}", second);
        assert!(second.contains("接着，"));
        assert_ne!(first, second);

        let done = reasoner.continue_generation().expect("讲完后应给出提示");
        assert!(done.contains("已经介绍完毕"));

        let angled = reasoner
            .switch_generation_angle()
            .expect("换个角度说应重新生成");
        assert!(angled.contains("换个角度"), "实际：{}", angled);
        assert!(reasoner.dialogue_summary().contains("上一段生成"));
    }

    #[test]
    fn test_expected_answer_type_reorders_find_object() {
        let mut facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "星海公司".to_string(),
            },
        ];
        facts.push(PrologFact::Entity {
            name: "李四".to_string(),
            entity_type: "人物".to_string(),
            properties: HashMap::new(),
        });
        facts.push(PrologFact::Entity {
            name: "星海公司".to_string(),
            entity_type: "公司".to_string(),
            properties: HashMap::new(),
        });
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner
            .answer("张三的徒弟是谁")
            .expect("应能回答反查对象问题");
        let person_at = answer.find("李四").expect("应包含李四");
        let company_at = answer.find("星海公司").expect("应包含星海公司");
        assert!(person_at < company_at, "人物类型答案应排在前面：{}", answer);
        assert!(answer.contains("期望"), "应说明排序依据：{}", answer);
    }

    #[test]
    fn test_predict_continuation_uses_facts_and_backoff() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let completion = reasoner
            .predict_continuation("张三的")
            .expect("已知实体应能续写");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));

        let mut synonyms = HashMap::new();
        synonyms.insert("老三".to_string(), "张三".to_string());
        let mut aliased = SymbolicReasoner::new(test_config(), synonyms);
        aliased.observe_facts(&master_facts());
        let aliased_completion = aliased
            .predict_continuation("老三的")
            .expect("同义词前缀应先归一化再续写");
        assert!(aliased_completion.contains("张三"));
    }

    fn nationality_facts() -> Vec<PrologFact> {
        let mut facts = Vec::new();
        for (name, has_nationality) in [
            ("张三", true),
            ("李四", true),
            ("王五", true),
            ("赵六", false),
        ] {
            let mut properties = HashMap::new();
            properties.insert("出生地".to_string(), format!("{}市", name));
            if has_nationality {
                properties.insert("国籍".to_string(), "中国".to_string());
            }
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "人物".to_string(),
                properties,
            });
        }
        // 一个无特征的地点篮子：拉低国籍先验，使关联规则的提升度可判
        facts.push(PrologFact::Entity {
            name: "无名城".to_string(),
            entity_type: "地点".to_string(),
            properties: HashMap::new(),
        });
        // 关系事实：让生成器有全局统计可回退（属性不进入关系计数）
        facts.push(PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        });
        facts
    }

    #[test]
    fn test_resolution_proof_from_facts_and_chain_goals() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("张三的师傅的师傅是王五");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(report.contains("王五"));
        assert!(report.contains("证明树"));
        assert!(report.contains("事实"));
        let failed = reasoner.resolution_proof_report("张三的师傅的师傅是赵六");
        assert!(failed.contains("未得证"), "实际：{}", failed);
        assert!(
            failed.contains("搜索状态："),
            "失败报告应披露搜索认识论状态：{}",
            failed
        );
        assert!(
            failed.contains("已穷尽") || failed.contains("搜索被截断"),
            "搜索状态必须区分无解与截断：{}",
            failed
        );
        assert!(
            failed.contains("搜索统计："),
            "搜索报告应披露确定性计数：{}",
            failed
        );
        let view = reasoner
            .resolution_proof_view("张三的师傅的师傅是赵六")
            .expect("应得到结构化证明视图");
        assert!(view.stats.goals >= 1);
        assert!(view.stats.goal_attempts > 0);
        assert!(!view.frontier.is_empty(), "穷尽无解应保留失败前沿");
    }

    #[test]
    fn test_resolution_proof_uses_mined_horn_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("李四的师祖是赵六");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(
            report.contains("链接预测规则"),
            "证明树应标出规则来源：{}",
            report
        );
        assert!(report.contains("师傅"));
    }

    #[test]
    fn test_resolution_query_returns_solution_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_query_report("张三的师傅的师傅是谁");
        assert!(report.contains("答案是「王五」"), "实际：{}", report);
        assert!(report.contains("事实"));
        assert!(report.contains("知识库"));
    }

    #[test]
    fn test_resolution_proof_view_is_structured_and_complete() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        // 陈述句：与报告同路径求解，结构可视图化
        let statement = reasoner
            .resolution_proof_view("张三的师傅的师傅是王五")
            .expect("陈述句应可视图化");
        assert!(statement.phrase.contains("王五"), "{}", statement.phrase);
        assert_eq!(statement.solutions.len(), 1);
        let nodes = crate::view_render::proof_forest_node_count(&statement.solutions[0].proofs);
        assert!(nodes >= 2, "链式证明至少覆盖两个事实：{}", nodes);
        let mermaid = crate::view_render::render_proof_mermaid(&statement.solutions[0].proofs);
        let rendered_nodes = mermaid
            .lines()
            .filter(|line| line.contains('[') && line.trim_start().starts_with('n'))
            .count();
        assert_eq!(
            rendered_nodes, nodes,
            "渲染节点集必须等于展平集合（不增不减）"
        );
        assert_eq!(
            mermaid,
            crate::view_render::render_proof_mermaid(&statement.solutions[0].proofs)
        );

        // 开放问句：绑定含答案变量
        let question = reasoner
            .resolution_proof_view("张三的师傅的师傅是谁")
            .expect("开放问句应可视图化");
        let solution = &question.solutions[0];
        assert!(
            solution
                .bindings
                .iter()
                .any(|(name, value)| name == "答案" && value == "王五"),
            "{:?}",
            solution.bindings
        );

        // 未得证：结构仍在，解为空（开世界假设）
        let unknown = reasoner
            .resolution_proof_view("张三的师傅的师傅是赵六")
            .expect("可解析但未得证");
        assert!(unknown.solutions.is_empty());

        // 不可形式化：None，而不是伪造空证明
        assert!(reasoner.resolution_proof_view("@@@").is_none());
    }

    #[test]
    fn test_answer_derivation_is_bounded_and_anchored_to_answer() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let derivation = reasoner
            .answer_derivation("张三的师傅的师傅是谁", Some("王五是张三的师祖"))
            .expect("应能从同一份归结数据重建推导");
        assert!(derivation.is_proved());
        assert_eq!(derivation.answer_value(), Some("王五"));
        assert!(!derivation.steps.is_empty());
        let text = derivation.render_explanation();
        assert!(text.contains("王五"), "{}", text);
        assert!(text.contains("证明步骤"), "{}", text);

        // 证明与上一轮回答不匹配时宁可不解释
        assert!(reasoner
            .answer_derivation("张三的师傅的师傅是谁", Some("赵六"))
            .is_none());
        // 无法形式化时诚实返回 None
        assert!(reasoner.answer_derivation("@@@", None).is_none());
    }

    #[test]
    fn test_dialogue_why_uses_resolution_proof_end_to_end() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let mut manager = crate::dialogue_manager::DialogueManager::new();
        let answer = manager
            .plan_reply("张三的师傅的师傅是谁", &reasoner)
            .expect("链式问句应被对话管理器接管");
        assert!(answer.text.contains("王五"), "{}", answer.text);

        let why = manager
            .plan_pure_dialogue("怎么推出来的？", &reasoner)
            .expect("推导追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(why.text.contains("事实"), "{}", why.text);
        assert!(why.text.contains("王五"), "{}", why.text);
    }

    #[test]
    fn test_resolution_proves_existence_via_association_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.resolution_proof_report("赵六有国籍");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(
            report.contains("关联规则"),
            "应有归结算子句来源：{}",
            report
        );
        assert!(report.contains("国籍"));
        let known = reasoner.resolution_proof_report("张三有国籍");
        assert!(known.contains("得证"));
        assert!(known.contains("事实"));
    }

    #[test]
    fn test_association_report_and_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.association_report(None);
        assert!(report.contains("支持度"), "实际：{}", report);
        assert!(report.contains("提升度"));
        assert!(report.contains("出生地"));
        let filtered = reasoner.association_report(Some("国籍"));
        assert!(filtered.contains("国籍"), "实际：{}", filtered);
        let suggestions = reasoner
            .association_suggestions("赵六")
            .expect("赵六缺国籍应有关联联想");
        assert!(suggestions.contains("国籍"), "实际：{}", suggestions);
        assert!(suggestions.contains("提升度"));
        assert!(!suggestions.contains("已有特征：国籍"));
    }

    #[test]
    fn test_derived_facts_report_discovers_multi_hop() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.derived_facts_report(10);
        assert!(report.contains("推理发现"), "实际：{}", report);
        assert!(report.contains("第 1 轮推导"), "应给出推导轮次：{}", report);
        assert!(report.contains("李四") && report.contains("赵六"));
        assert!(report.contains("前提"));
    }

    #[test]
    fn test_continuation_appends_association_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let text = reasoner
            .predict_continuation("赵六的")
            .expect("已知实体应能续写");
        assert!(text.contains("关联规则联想"), "实际：{}", text);
        assert!(text.contains("国籍"));
    }

    #[test]
    fn test_abduce_statement_generates_explained_hypotheses() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 张三的师祖是赵六：需要补全 师傅(李四, 赵六) 才能沿规则链证明
        let report = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("陈述应可形式化为溯因目标");
        assert!(report.contains("假设生成"), "实际：{}", report);
        assert!(report.contains("失败前沿"), "实际：{}", report);
        assert!(report.contains("师傅") && report.contains("李四") && report.contains("赵六"));
        assert!(report.contains("可信度"));
        let pending = reasoner.pending_hypotheses();
        assert!(!pending.is_empty(), "应暂存候选假设供采纳");
        assert!(pending.iter().all(|hypothesis| hypothesis.atom.is_ground()));
    }

    #[test]
    fn test_adopt_hypothesis_enables_session_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.adopt_hypothesis(1).is_none(),
            "无待定假设时应返回 None"
        );
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("未得证"));
        let message = reasoner.adopt_hypothesis(1).expect("应能采纳第 1 条假设");
        assert!(message.contains("已采纳假设"), "实际：{}", message);
        // 采纳后：假设进入会话归结事实集，后向链即可证明
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert!(proof.contains("赵六"));
    }

    #[test]
    fn test_hypothesis_answer_for_missing_link() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 赵六没有师傅记录：开放问句应生成候选取值假设
        let report = reasoner
            .hypothesis_answer("赵六的师傅是谁")
            .expect("已知实体+已知谓词应可溯因");
        assert!(report.contains("假设生成"), "实际：{}", report);
        assert!(report.contains("候选假设"), "实际：{}", report);
        assert!(report.contains("师傅"));
        let pending = reasoner.pending_hypotheses();
        assert!(!pending.is_empty());
        let displays: Vec<String> = pending.iter().map(|h| h.atom.display()).collect();
        assert!(
            pending
                .iter()
                .any(|hypothesis| hypothesis.atom.display().contains("师傅")),
            "候选假设应补全师傅关系，实际：{:?}",
            displays
        );
    }

    #[test]
    fn test_what_if_report_derives_and_rolls_back() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .what_if_report("如果李四的师傅是赵六")
            .expect("条件句应可解析为反事实假设");
        assert!(report.contains("反事实模拟"), "实际：{}", report);
        assert!(report.contains("赵六"), "实际：{}", report);
        assert!(
            report.contains("师祖"),
            "应沿师傅→师祖规则推出新结论：{}",
            report
        );
        // 模拟回滚：原知识库不能被改变
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("未得证"), "反事实模拟必须回滚：{}", proof);
    }

    #[test]
    fn test_provable_statement_gets_proof_not_hypotheses() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .abduce_statement("张三的师祖是王五")
            .expect("可证陈述应返回证明");
        assert!(report.contains("后向链得证"), "实际：{}", report);
        assert!(report.contains("证明树"));
        assert!(reasoner.pending_hypotheses().is_empty());
    }

    // ---- 生成式联合解释 + 对话式澄清 + 跨会话临时信念 ----

    fn sparse_master_facts() -> Vec<PrologFact> {
        // 只有 李四→王五 一条师傅关系：证明 张三的师祖是赵六 需要同时补两条边
        let mut facts = vec![
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            // 让"师祖"成为已知谓词（解析需要），与目标无关
            PrologFact::Relation {
                from: "钱七".to_string(),
                rel_type: "师祖".to_string(),
                to: "孙八".to_string(),
            },
        ];
        for name in ["张三", "李四", "王五", "赵六", "钱七", "孙八"] {
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            });
        }
        facts
    }

    fn teacher_rule() -> Clause {
        Clause::new(
            Atom::pattern("师祖", Term::var("x"), Term::var("y")),
            vec![
                Atom::pattern("师傅", Term::var("x"), Term::var("z")),
                Atom::pattern("师傅", Term::var("z"), Term::var("y")),
            ],
            0.9,
            "测试规则：师傅传递",
        )
    }

    #[test]
    fn test_scenario_report_verifies_joint_explanation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let report = reasoner.scenario_report("张三的师祖是赵六");
        assert!(report.contains("生成式联合解释"), "实际：{}", report);
        assert!(
            report.contains("2 条假设"),
            "应枚举出两条假设的联合场景：{}",
            report
        );
        assert!(
            report.contains("注入后目标可证"),
            "联合场景应通过注入验证：{}",
            report
        );
        assert!(report.contains("验证证明"), "应展示验证证明：{}", report);
        assert!(
            !reasoner.pending_hypotheses().is_empty(),
            "场景成员应进入待定池，供采纳/澄清"
        );
    }

    #[test]
    fn test_clarification_confirms_hypothesis_and_proves_goal() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("应能溯因");
        assert!(report.contains("假设生成"), "实际：{}", report);
        let question = reasoner.start_clarification().expect("有待定假设即可澄清");
        assert!(
            question.contains("请确认") || question.contains("是否成立"),
            "实际：{}",
            question
        );
        let response = reasoner.answer_clarification("是的").expect("应处理确认");
        assert!(response.contains("已确认假设"), "实际：{}", response);
        assert!(
            response.contains("已可证明"),
            "确认后目标应可证明：{}",
            response
        );
        assert!(!reasoner.clarification_active(), "目标得证后澄清应结束");
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert_eq!(reasoner.session_facts().len(), 1);
    }

    #[test]
    fn test_clarification_reject_and_defer_flow() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("应能溯因");
        reasoner.start_clarification().expect("应能开始澄清");
        let rejected = reasoner.answer_clarification("不是").expect("应处理否定");
        assert!(rejected.contains("已排除假设"), "实际：{}", rejected);
        if reasoner.clarification_active() {
            let deferred = reasoner
                .answer_clarification("不确定")
                .expect("应处理不确定");
            assert!(
                deferred.contains("暂记为不确定") || deferred.contains("澄清结束"),
                "实际：{}",
                deferred
            );
        }
        assert!(
            reasoner.session_facts().is_empty(),
            "否定/不确定都不应产生会话事实"
        );
    }

    #[test]
    fn test_session_facts_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        let message = reasoner.adopt_hypothesis(1).expect("应能采纳");
        assert!(message.contains("已采纳假设"), "实际：{}", message);
        let path =
            std::env::temp_dir().join(format!("knowprolog_session_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "跨会话恢复的假设应参与证明：{}",
            proof
        );
        let soft = restored.soft_inference_report("张三");
        assert!(
            soft.contains("师祖") && soft.contains("赵六"),
            "跨会话恢复的假设应注入软逻辑：{}",
            soft
        );
        let problog = restored.problog_report("张三");
        assert!(
            problog.contains("师祖") && problog.contains("赵六"),
            "跨会话恢复的假设应注入概率程序：{}",
            problog
        );
        assert!(
            restored.dialogue_summary().contains("会话假设 1 条"),
            "对话记忆应显示会话假设：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_session_facts_survive_resolver_rebuild() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        reasoner.adopt_hypothesis(1).expect("应能采纳");
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("得证"));
        // 再观察新关系会触发归结引擎重建：会话事实必须在重建后重新注入
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "赵六".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        }]);
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "重建后采纳的假设不应丢失：{}",
            proof
        );
    }

    #[test]
    fn test_structural_related_report_uses_random_walk() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.structural_related_report("张三");
        assert!(report.contains("随机游走重启"), "实际：{}", report);
        assert!(
            report.contains("李四"),
            "直接邻居应在结构联想中：{}",
            report
        );
        assert!(report.contains("游走驻留概率"));
    }

    #[test]
    fn test_topic_extension_uses_dialogue_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.topic_extension_report().contains("还没有对话焦点"),
            "无焦点时应给出提示"
        );
        let _ = reasoner.contextualize("张三的师傅是谁");
        let report = reasoner.topic_extension_report();
        assert!(report.contains("话题延伸"), "实际：{}", report);
        assert!(
            report.contains("张三"),
            "焦点实体应出现在报告中：{}",
            report
        );
        assert!(report.contains("李四"), "焦点邻居应被延伸出来：{}", report);
    }

    #[test]
    fn test_soft_inference_report_gives_probability_and_evidence() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let known = reasoner.soft_inference_report("张三的师祖");
        assert!(known.contains("王五"), "实际：{}", known);
        assert!(known.contains("P=1.00"), "图谱事实概率应为 1：{}", known);
        let inferred = reasoner.soft_inference_report("李四的师祖");
        assert!(inferred.contains("赵六"), "实际：{}", inferred);
        assert!(inferred.contains("依据"), "软推理应给出证据：{}", inferred);
        assert!(inferred.contains("P="), "软推理应给出概率：{}", inferred);
    }

    #[test]
    fn test_soft_facts_report_lists_weighted_rules() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.soft_facts_report(5);
        assert!(report.contains("加权软推理"), "实际：{}", report);
        assert!(report.contains("加权规则"), "实际：{}", report);
    }

    #[test]
    fn test_graph_walk_and_soft_logic_respect_disabled_config() {
        let mut config = test_config();
        config.enable_graph_walk = false;
        config.enable_soft_logic = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .structural_related_report("张三")
            .contains("未启用"));
        assert!(reasoner.topic_extension_report().contains("未启用"));
        assert!(reasoner.soft_inference_report("张三").contains("未启用"));
        assert!(reasoner.soft_facts_report(5).contains("未启用"));
    }

    #[test]
    fn test_session_assumption_feeds_soft_inference() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        reasoner.adopt_hypothesis(1).expect("应能采纳假设");
        let report = reasoner.soft_inference_report("张三的师祖");
        assert!(
            report.contains("赵六"),
            "采纳的会话假设应进入软推理：{}",
            report
        );
    }

    #[test]
    fn test_causal_report_explains_soft_prediction() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.causal_report("李四的师祖");
        assert!(report.contains("因果推理"), "实际：{}", report);
        assert!(report.contains("赵六"), "应解释推出的师祖：{}", report);
        assert!(report.contains("直接原因"), "应列出直接原因：{}", report);
        assert!(report.contains("依据："), "应给出规则来源：{}", report);
        assert!(
            report.contains("事实根因链") && report.contains("—["),
            "应给出可回放的因果链：{}",
            report
        );
        // 没有可解释目标时优雅降级
        assert!(reasoner.causal_report("不存在的人").contains("无法"));
    }

    #[test]
    fn test_counterfactual_report_quantifies_do_effect() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.counterfactual_report("李四的师傅", false);
        assert!(report.contains("反事实推理"), "实际：{}", report);
        assert!(report.contains("do"), "应说明干预语义：{}", report);
        assert!(report.contains("师傅"), "应指出被干预的关系：{}", report);
        assert!(report.contains("Δ="), "应给出概率差：{}", report);
        assert!(
            report.contains("下降"),
            "移除师傅应使下游加权推测下降：{}",
            report
        );
        let boosted = reasoner.counterfactual_report("李四的师傅", true);
        assert!(
            boosted.contains("不变") || boosted.contains("上升"),
            "置真干预不应使下游下降：{}",
            boosted
        );
    }

    #[test]
    fn test_gradient_report_shows_symbolic_derivatives() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.gradient_report("李四的师祖");
        assert!(report.contains("符号梯度归因"), "实际：{}", report);
        assert!(report.contains("∂P/∂w"), "应展示解析梯度：{}", report);
        assert!(
            report.contains("规则贡献"),
            "应逐条列出规则贡献：{}",
            report
        );
        assert!(report.contains("权重"), "应展示规则权重：{}", report);
        assert!(report.contains("来源："), "应展示规则来源：{}", report);
        assert!(report.contains("P="), "应展示概率：{}", report);
        // 未知实体时优雅提示
        assert!(reasoner.gradient_report("钱七").contains("无法"));
    }

    #[test]
    fn test_causal_and_gradient_respect_disabled_config() {
        let mut config = test_config();
        config.enable_causal_reasoning = false;
        config.enable_gradient_attribution = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.causal_report("李四的师祖").contains("未启用"));
        assert!(reasoner
            .counterfactual_report("张三的师傅", false)
            .contains("未启用"));
        assert!(reasoner.gradient_report("李四的师祖").contains("未启用"));
    }

    #[test]
    fn test_causal_tokenize_report_segments_with_alias_and_attributes() {
        let synonyms = HashMap::from([("师父".to_string(), "师傅".to_string())]);
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&master_facts());
        let report = reasoner.causal_tokenize_report("张三的师父是谁");
        assert!(report.contains("因果分词"), "实际：{}", report);
        assert!(report.contains("最优分词"), "实际：{}", report);
        assert!(
            report.contains("「师傅」（别名）"),
            "同义词「师父」应锚定到规范谓词「师傅」：{}",
            report
        );
        assert!(report.contains("token 因果归因"), "实际：{}", report);
        assert!(
            report.contains("张三"),
            "输入实体应出现在归因中：{}",
            report
        );
        assert!(
            report.contains("纯符号 attention"),
            "应声明归因是纯符号可回放：{}",
            report
        );
    }

    #[test]
    fn test_causal_tokenization_respects_disabled_config() {
        let mut config = test_config();
        config.enable_causal_tokenization = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .causal_tokenize_report("张三的师傅是谁")
            .contains("未启用"));
    }

    #[test]
    fn test_intervention_boosts_downstream_probability() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 干预一个"推导出来的中间节点"（李四的师傅=王五）为真：
        // 王五被钉真后，李四的师祖（赵六）概率应不低于事实世界
        let report = reasoner.counterfactual_report("李四的师傅", true);
        assert!(report.contains("师傅"), "实际：{}", report);
        assert!(report.contains("反事实推理"));
        assert!(
            report.contains("不变") || report.contains("上升"),
            "干预中间节点置真不应降低下游：{}",
            report
        );
    }

    #[test]
    fn test_problog_report_gives_exact_probability_and_proof_paths() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.problog_report("李四的师祖");
        assert!(report.contains("可能世界精确推理"), "实际：{}", report);
        assert!(report.contains("赵六"), "应推出赵六：{}", report);
        assert!(report.contains("证明路径"), "应给出证明路径：{}", report);
        assert!(
            report.contains("与软逻辑"),
            "应给出与 MLN-lite 的对照：{}",
            report
        );
        assert!(
            report.contains("P=0.") || report.contains("P=1."),
            "应给出概率：{}",
            report
        );
        assert!(reasoner.problog_report("不存在的人").contains("无法"));
    }

    #[test]
    fn test_teach_chinese_conditional_rule_affects_problog() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.teach_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(taught.contains("逻辑形式"), "实际：{}", taught);
        assert!(taught.contains("中文教学·条件句"), "应标注来源：{}", taught);
        assert!(taught.contains("95%"), "默认条件句概率应为 95%：{}", taught);
        let report = reasoner.problog_report("张三的师祖");
        assert!(report.contains("赵六"), "教学规则应推出赵六：{}", report);
        assert!(
            report.contains("中文教学"),
            "证明路径应保留教学来源：{}",
            report
        );
        let again = reasoner.teach_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(again.contains("已经学过"), "重复教学应被识别：{}", again);
    }

    #[test]
    fn test_teach_probabilistic_fact_percentage_query() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        assert!(taught.contains("概率事实"), "实际：{}", taught);
        assert!(taught.contains("80%"), "应为百分比概率：{}", taught);
        let report = reasoner.problog_report("李四的喜欢");
        assert!(
            report.contains("P=0.800"),
            "应给出精确概率 0.800：{}",
            report
        );
        assert!(report.contains("概率事实"), "应说明来源：{}", report);
    }

    #[test]
    fn test_problog_and_teaching_respect_disabled_config() {
        let mut config = test_config();
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.problog_report("李四的师祖").contains("未启用"));
        assert!(reasoner.teach_report("八成的鸟会飞").contains("未启用"));
    }

    #[test]
    fn test_problog_lessons_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        let path =
            std::env::temp_dir().join(format!("knowprolog_lessons_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.lessons().len(), 1, "教学记忆应跨会话恢复");
        let report = restored.problog_report("李四的喜欢");
        assert!(
            report.contains("P=0.800"),
            "恢复后应重放概率知识：{}",
            report
        );
        assert!(
            restored.dialogue_summary().contains("中文概率知识 1 条"),
            "对话记忆应显示教学记忆：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_problog_query_accepts_taught_entity_outside_graph() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("八成的鸟会飞");
        let report = reasoner.problog_report("鸟");
        assert!(report.contains("会飞"), "教学事实应可查询：{}", report);
        assert!(report.contains("P=0.800"), "应给出精确概率：{}", report);
        assert!(report.contains("概率事实"), "应说明来源：{}", report);
    }

    #[test]
    fn test_default_logic_applies_and_blocks_with_exception() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.assert_nl_fact("蒂娜是鸟");
        reasoner.assert_nl_fact("蒂娜是企鹅");
        let taught = reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        assert!(taught.contains("默认规则"), "应识别默认规则：{}", taught);
        assert!(taught.contains("除非"), "应展示例外：{}", taught);

        let applied = reasoner.default_logic_report("麻雀会飞");
        assert!(
            applied.contains("成立（默认推理触发）"),
            "实际：{}",
            applied
        );
        assert!(applied.contains("通常推出"), "应给出触发默认：{}", applied);
        assert!(
            applied.contains("P=0.900"),
            "应给出概率视图对照：{}",
            applied
        );

        let blocked = reasoner.default_logic_report("蒂娜会飞");
        assert!(
            blocked.contains("未成立（被默认阻断）"),
            "实际：{}",
            blocked
        );
        assert!(blocked.contains("例外"), "应说明例外：{}", blocked);
        assert!(
            blocked.contains("蒂娜是企鹅"),
            "应点名例外事实：{}",
            blocked
        );
    }

    #[test]
    fn test_default_logic_reports_missing_evidence_and_unformalizable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let unknown = reasoner.default_logic_report("阿猫会飞");
        assert!(unknown.contains("未找到证据"), "实际：{}", unknown);
        let unformalizable = reasoner.default_logic_report("今天天气怎么样");
        assert!(
            unformalizable.contains("无法把"),
            "应提示无法形式化：{}",
            unformalizable
        );
    }

    #[test]
    fn test_default_logic_supplement_only_fires_with_defaults() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.assert_nl_fact("蒂娜是鸟");
        reasoner.assert_nl_fact("蒂娜是企鹅");
        assert!(
            reasoner.default_logic_supplement("麻雀会飞").is_none(),
            "没有默认规则时不应产生补充"
        );
        reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        let supplement = reasoner
            .default_logic_supplement("麻雀会飞")
            .expect("默认规则就位后应给出补充");
        assert!(supplement.contains("通常成立"), "实际：{}", supplement);
        let question = reasoner
            .default_logic_supplement("请问麻雀会飞吗")
            .expect("疑问句形式也应给出补充");
        assert!(question.contains("通常成立"), "实际：{}", question);
        let blocked = reasoner
            .default_logic_supplement("蒂娜会飞吗")
            .expect("例外应给出阻断补充");
        assert!(blocked.contains("不能推出"), "实际：{}", blocked);
        assert!(
            reasoner
                .default_logic_supplement("今天天气怎么样")
                .is_none(),
            "无法形式化时不应产生补充"
        );
    }

    #[test]
    fn test_default_lessons_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        let path =
            std::env::temp_dir().join(format!("knowprolog_defaults_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let report = restored.default_logic_report("麻雀会飞");
        assert!(
            report.contains("成立（默认推理触发）"),
            "恢复后默认规则与会话事实应重放生效：{}",
            report
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_learn_general_rule_enables_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            taught.contains("通用规则"),
            "应识别为含变量的通用规则：{}",
            taught
        );
        assert!(taught.contains("老师"), "应展示逻辑形式：{}", taught);
        // 判断走语义蕴含；图谱里没有"老师"事实时由后向链接管
        let judged = reasoner.entailment_report("李四的老师是王五");
        assert!(judged.contains("后向链补充判定"), "实际：{}", judged);
        assert!(judged.contains("王五"), "证明树应落到王五：{}", judged);
        // 一般规则可套用到任意实体：张三的师傅是李四
        let queried = reasoner.resolution_query_report("张三的老师是谁");
        assert!(queried.contains("答案是「李四」"), "实际：{}", queried);
        let proof = reasoner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert!(
            proof.contains("中文教学"),
            "证明树应标出规则来源：{}",
            proof
        );
    }

    #[test]
    fn test_learn_ground_rule_joins_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(taught.contains("具体规则"), "应识别为具体规则：{}", taught);
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "教过的具体规则应参与后向链：{}",
            proof
        );
    }

    #[test]
    fn test_learn_rule_report_rejects_non_rules_gracefully() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.learn_rule_report("张三是李四的师傅");
        assert!(
            report.contains("没能把这句话解析成规则"),
            "实际：{}",
            report
        );
        let empty = reasoner.learn_rule_report("   ");
        assert!(empty.contains("请给出中文规则"), "实际：{}", empty);
    }

    #[test]
    fn test_chain_answer_carries_witness_paths() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问句应给出路径回答");
        assert!(answer.contains("路径"), "实际：{}", answer);
        assert!(
            answer.contains("张三 --师傅--> 李四 --师傅--> 王五"),
            "路径应保留每一跳：{}",
            answer
        );
        assert!(
            answer.matches("推理依据").count() <= 1,
            "推理依据不应重复：{}",
            answer
        );
    }

    #[test]
    fn test_check_triple_uses_path_witness() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是王五吗")
            .expect("多跳校验应给出证明路径");
        assert!(answer.contains("是的"), "实际：{}", answer);
        assert!(
            answer.contains("张三 --师傅--> 李四 --师傅--> 王五"),
            "校验应附带路径：{}",
            answer
        );
    }

    #[test]
    fn test_session_overlay_extends_paths_and_retract_restores() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        let before = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            !before.contains("赵六"),
            "未断言前不应出现跨分支答案：{}",
            before
        );

        let message = reasoner.assert_nl_fact("王五是李四的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        let answer = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("会话事实应补齐中间边");
        assert!(answer.contains("赵六"), "实际：{}", answer);
        assert!(answer.contains("路径"), "实际：{}", answer);

        let forgotten = reasoner.retract_nl_fact("王五是李四的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let after = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(!after.contains("赵六"), "撤回后路径应失效：{}", after);
    }

    #[test]
    fn test_negative_session_fact_blocks_paths_and_can_be_retracted() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("赵六不是王五的师傅");
        assert!(message.contains("已记住否定事实"), "实际：{}", message);

        let blocked = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            blocked.contains("否定记录阻断"),
            "应报告否定阻断原因：{}",
            blocked
        );

        let checked = reasoner.answer("王五的师傅是赵六吗").unwrap_or_default();
        assert!(
            checked.contains("不是"),
            "直接校验应尊重会话否定：{}",
            checked
        );

        let forgotten = reasoner.retract_nl_fact("赵六不是王五的师傅");
        assert!(forgotten.contains("已撤销否定事实"), "实际：{}", forgotten);
        let restored = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("撤销否定后路径应恢复");
        assert!(restored.contains("赵六"), "实际：{}", restored);
    }

    #[test]
    fn test_positive_session_fact_supersedes_negative() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("赵六不是王五的师傅");
        let message = reasoner.assert_nl_fact("王五的师傅是赵六");
        assert!(
            message.contains("已撤销同名会话否定事实"),
            "实际：{}",
            message
        );

        let answer = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("肯定事实应恢复路径");
        assert!(answer.contains("赵六"), "实际：{}", answer);
        assert!(!answer.contains("否定记录阻断"), "实际：{}", answer);
        let checked = reasoner.answer("王五的师傅是赵六吗").unwrap_or_default();
        assert!(checked.contains("是的"), "实际：{}", checked);
    }

    #[test]
    fn test_negative_session_fact_roundtrips_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("赵六不是王五的师傅");
        let path =
            std::env::temp_dir().join(format!("knowprolog_negative_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let answer = restored
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            answer.contains("否定记录阻断"),
            "恢复的否定边仍应阻断：{}",
            answer
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_assert_and_retract_session_fact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("得证"), "记住的事实应参与后向链：{}", proof);
        let rules = reasoner.session_rules_report();
        assert!(rules.contains("会话事实"), "实际：{}", rules);
        assert!(
            rules.contains("钱七"),
            "会话事实清单应包含新事实：{}",
            rules
        );

        let forgotten = reasoner.retract_nl_fact("钱七是孙八的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤回后不应再得证：{}", proof);
    }

    #[test]
    fn retract_session_fact_is_incremental_and_keeps_graph_duplicates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let perf = reasoner.perf_ledger();
        let rebuilds = perf.count("rebuild_resolver");

        let message = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        assert_eq!(
            perf.count("rebuild_resolver"),
            rebuilds,
            "采纳会话事实不应重建归结引擎"
        );

        // 与图谱同名的会话副本：撤回后图谱事实必须仍然可证
        let duplicate = reasoner.assert_nl_fact("张三是李四的师傅");
        assert!(duplicate.contains("已记住"), "实际：{}", duplicate);
        let forgotten = reasoner.retract_nl_fact("张三是李四的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("李四的师傅是张三");
        assert!(proof.contains("得证"), "同名图谱事实不得被误删：{}", proof);

        let forgotten = reasoner.retract_nl_fact("钱七是孙八的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤回后不应再得证：{}", proof);
        assert_eq!(
            perf.count("rebuild_resolver"),
            rebuilds,
            "撤回会话事实应增量删除而不是整库重建"
        );
    }

    #[test]
    fn rebuild_from_graph_drops_old_values_and_keeps_session_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.assert_nl_fact("钱七是孙八的师傅");
        let before = reasoner.answer("姚明的身高是什么").expect("重建前应可回答");
        assert!(before.contains("226cm"), "实际：{}", before);

        let mut properties = HashMap::new();
        properties.insert("身高".to_string(), "229cm".to_string());
        reasoner.rebuild_from_graph(&[PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人".to_string(),
            properties,
        }]);

        let after = reasoner.answer("姚明的身高是什么").expect("重建后应可回答");
        assert!(after.contains("229cm"), "实际：{}", after);
        assert!(!after.contains("226cm"), "旧值不应残留：{}", after);
        assert_eq!(reasoner.session_facts().len(), 1, "会话事实应保留");
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(
            proof.contains("得证"),
            "会话事实应继续参与后向链：{}",
            proof
        );
    }

    #[test]
    fn test_assert_nl_fact_rejects_conditional_sentence() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(message.contains("不是确定事实"), "实际：{}", message);
        assert!(reasoner.session_facts().is_empty());
    }

    #[test]
    fn test_session_clauses_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let path =
            std::env::temp_dir().join(format!("knowprolog_clauses_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("李四的老师是王五");
        assert!(
            proof.contains("得证"),
            "恢复后的会话规则应参与后向链：{}",
            proof
        );
        assert!(
            restored.dialogue_summary().contains("会话规则 1 条"),
            "对话记忆应显示会话规则：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_rule_ledger_entries_and_fingerprint_track_session_rules() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let empty = reasoner.rule_fingerprint();
        assert!(reasoner.rule_ledger_entries().is_empty());

        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = reasoner.rule_ledger_entries();
        assert_eq!(entries.len(), 1, "教过的规则应成为账本条目");
        assert!(
            entries[0].head.display().contains("老师"),
            "实际：{}",
            entries[0].head.display()
        );
        assert_eq!(entries[0].body.len(), 1);
        assert!(
            entries[0].source.contains("中文教学"),
            "来源应可审计：{}",
            entries[0].source
        );
        assert!(!entries[0].id.is_empty());
        assert_eq!(
            reasoner.rule_fingerprint(),
            graph_ledger::rule_fingerprint(&entries),
            "指纹必须由当前规则集确定"
        );
        assert_ne!(reasoner.rule_fingerprint(), empty, "规则增删必须改变指纹");
    }

    #[test]
    fn test_forget_rule_retracts_clause_and_lesson_consistently() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("如果X的师傅是Y，那么X的老师是Y");
        assert_eq!(
            reasoner.rule_ledger_entries().len(),
            1,
            "教学规则应立即入账"
        );
        assert!(!reasoner.lessons().is_empty());

        let forgotten = reasoner.forget_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            forgotten.contains("已忘记 1 条会话规则"),
            "实际：{}",
            forgotten
        );
        assert!(
            forgotten.contains("撤回 1 条对应教学原句"),
            "实际：{}",
            forgotten
        );
        assert!(
            reasoner.rule_ledger_entries().is_empty(),
            "忘记后账本条目清空"
        );
        assert!(reasoner.lessons().is_empty(), "教学原句应一并撤回");
        let proof = reasoner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("未得证"), "忘记后规则不应再得证：{}", proof);

        let again = reasoner.forget_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            again.contains("没有匹配的规则"),
            "重复忘记应零状态变化：{}",
            again
        );
    }

    #[test]
    fn test_restore_rules_from_ledger_enables_backward_chain() {
        let mut teacher = SymbolicReasoner::new(test_config(), HashMap::new());
        teacher.observe_facts(&master_facts());
        teacher.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = teacher.rule_ledger_entries();

        let mut learner = SymbolicReasoner::new(test_config(), HashMap::new());
        learner.observe_facts(&master_facts());
        assert_eq!(learner.restore_rules_from_ledger(&entries), 1);
        assert_eq!(
            learner.restore_rules_from_ledger(&entries),
            0,
            "重复恢复不重复添加"
        );
        let proof = learner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "恢复的规则应参与后向链：{}", proof);
        assert_eq!(
            learner.rule_fingerprint(),
            teacher.rule_fingerprint(),
            "恢复后指纹与原会话一致"
        );
    }

    #[test]
    fn test_derived_ledger_entries_carry_provenance_and_stamp() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = reasoner.derived_ledger_entries(10);
        assert!(!entries.is_empty(), "教过的规则应推出新事实");
        let derived = entries
            .iter()
            .find(|entry| entry.atom.predicate == "老师" && entry.atom.display().contains("李四"))
            .unwrap_or_else(|| panic!("应推出李四的老师：{:?}", entries));
        assert_eq!(derived.depth, 1);
        assert!(
            derived.rule_source.contains("中文教学"),
            "规则来源应可审计：{}",
            derived.rule_source
        );
        assert_eq!(
            derived.premises.len(),
            1,
            "应记录前提：{:?}",
            derived.premises
        );
        assert_eq!(derived.revision, reasoner.revision(), "物化戳应记知识版本");
        assert_eq!(
            derived.fingerprint,
            reasoner.rule_fingerprint(),
            "物化戳应记规则集指纹"
        );
        assert!(
            reasoner.derived_ledger_entries(0).is_empty(),
            "limit=0 不产出条目"
        );
    }

    #[test]
    fn test_problog_multi_sentence_teaching_reports_errors() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report =
            reasoner.teach_report("如果李四的师傅是王五，那么李四的师祖是赵六。这句话是谁说的？");
        assert!(report.contains("已学会 1 条"), "实际：{}", report);
        assert!(report.contains("未学会"), "应逐条报告失败原因：{}", report);
        assert!(report.contains("疑问句"), "疑问句应被拒绝：{}", report);
    }

    #[test]
    fn test_holographic_memory_feeds_from_facts_and_recalls() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let holo = reasoner.holographic();
        assert_eq!(holo.triple_count(), 4, "全部三元组应进入全息图");
        assert!(holo.dimension() >= 256);

        let objects = holo.recall_object("张三", "师傅");
        assert_eq!(
            objects.first().map(|hit| hit.symbol.as_str()),
            Some("李四"),
            "全息客体召回 top-1 应为李四：{:?}",
            objects
        );
        assert!(objects[0].verified, "事实中的三元组应通过硬核查");

        let subjects = holo.recall_subject("师傅", "王五");
        assert_eq!(
            subjects.first().map(|hit| hit.symbol.as_str()),
            Some("李四")
        );

        let predicates = holo.recall_predicate("张三", "王五");
        assert_eq!(
            predicates.first().map(|hit| hit.symbol.as_str()),
            Some("师祖")
        );

        let analogy = holo.analogy("张三", "李四", "王五");
        assert_eq!(
            analogy.first().map(|hit| hit.symbol.as_str()),
            Some("赵六"),
            "张三:李四 :: 王五:? 应迁移师傅关系召回赵六：{:?}",
            analogy
        );
        assert!(analogy[0].verified);
    }

    #[test]
    fn test_holographic_context_records_mention_trace() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.note_holo_context("张三的师傅是谁？");
        reasoner.note_holo_context("李四的国籍是什么？");
        let holo = reasoner.holographic();
        assert_eq!(holo.context_len(), 2, "两轮不同焦点应产生两条时序记忆");
        let recent = holo.recent_focus(0).expect("应能回读最近焦点");
        assert_eq!(recent.symbol, "李四");
        let previous = holo.recent_focus(1).expect("应能回读上一轮焦点");
        assert_eq!(previous.symbol, "张三");
    }

    #[test]
    fn test_holographic_memory_respects_disabled_config() {
        let mut config = test_config();
        config.enable_holographic_memory = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.note_holo_context("张三的师傅是谁？");
        assert_eq!(reasoner.holographic().triple_count(), 0);
        assert_eq!(reasoner.holographic().context_len(), 0);
    }

    /// VID（版本化增量派生）：重复观察同一批事实必须版本不动、派生不重建。
    #[test]
    fn test_repeat_observation_is_version_stable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert_eq!(reasoner.revision(), 1, "首次观察应产生版本 1");
        let perf = reasoner.perf_ledger();
        let resolver_rebuilds = perf.count("rebuild_resolver");
        let soft_rebuilds = perf.count("rebuild_soft_logic");
        let cache_rebuilds = perf.count("rebuild_knowledge_caches");
        let embedding_rebuilds = perf.count("embedding_rebuild");

        // 整图快照重放：版本与派生台账必须纹丝不动
        reasoner.observe_facts(&master_facts());
        reasoner.observe_facts(&master_facts());
        assert_eq!(reasoner.revision(), 1, "纯重复观察不应推进版本");
        assert_eq!(perf.count("rebuild_resolver"), resolver_rebuilds);
        assert_eq!(perf.count("rebuild_soft_logic"), soft_rebuilds);
        assert_eq!(perf.count("rebuild_knowledge_caches"), cache_rebuilds);
        assert_eq!(perf.count("embedding_rebuild"), embedding_rebuilds);
        assert_eq!(perf.count("observe_no_change"), 2);

        // 加入一条事实：版本推进，派生台账重新挂牌
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        assert_eq!(reasoner.revision(), 2, "新增事实应推进版本");
        assert_eq!(perf.count("rebuild_resolver"), resolver_rebuilds + 1);
        assert!(reasoner.derivation_report().contains("知识版本 2"));
    }

    /// 失效的 memo 必须在知识变化后清空（版本感应的缓存一致性）。
    #[test]
    fn test_memo_invalidated_only_on_change() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let perf = reasoner.perf_ledger();
        let hits = perf.count("memo_entity_mentions_hit");
        // 同一文本重复查询：命中记忆
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 1);

        // 纯重复观察不清缓存：仍应命中
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 2);

        // 知识变化后缓存失效：同文本查询不能再命中
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(
            perf.count("memo_entity_mentions_hit"),
            hits + 2,
            "版本变化后过期记忆必须失效"
        );
    }

    /// 多槽记忆：同一知识版本内交替询问多个文本，各自留槽、重复命中；
    /// 版本推进后整体不可命中（纪元失效，不靠 TTL）。
    #[test]
    fn test_memo_keeps_multiple_texts_within_revision() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let perf = reasoner.perf_ledger();

        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let _ = reasoner.entities_mentioned("李四的师傅是谁");
        let hits = perf.count("memo_entity_mentions_hit");
        // 交替再问：两个文本都应命中（不得互相挤掉）
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let _ = reasoner.entities_mentioned("李四的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 2);

        // 链接记忆同样多槽
        let _ = reasoner.linked_mentions("张三的师傅是谁");
        let _ = reasoner.linked_mentions("李四的师傅是谁");
        let link_hits = perf.count("memo_link_hit");
        let _ = reasoner.linked_mentions("张三的师傅是谁");
        let _ = reasoner.linked_mentions("李四的师傅是谁");
        assert_eq!(perf.count("memo_link_hit"), link_hits + 2);

        // 知识版本推进：原文本不得命中
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        let before = perf.count("memo_entity_mentions_hit");
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), before);
    }

    /// WarmBase 的跨线程前提：分叉基座必须可 `Send`。
    #[test]
    fn reasoner_is_send_for_cross_thread_fork() {
        fn assert_send<T: Send>() {}
        assert_send::<SymbolicReasoner>();
    }

    /// 分叉契约：从「观察完毕的基座」分叉，与「新建 + 重放同一批事实」在
    /// 固定查询集与固定对话序列上逐字节一致（回答、焦点、历史、接地清单）。
    #[test]
    fn fork_matches_fresh_observe_byte_for_byte() {
        let facts = master_facts();
        let config = test_config();

        let mut base = SymbolicReasoner::new(config.clone(), HashMap::new());
        base.observe_facts(&facts);
        let mut forked = base.fork_session();

        let mut fresh = SymbolicReasoner::new(config, HashMap::new());
        fresh.observe_facts(&facts);

        assert_eq!(forked.revision(), fresh.revision(), "知识版本必须一致");

        // 固定查询集：覆盖关系查询、逆向谓词、属性、链式与未知实体
        let questions = [
            "张三的师傅是谁",
            "李四的师祖是谁",
            "王五的师傅是谁",
            "赵六的师傅是谁",
            "张三的师傅的师傅是谁",
            "姚明的身高是什么",
            "秦始皇是谁",
            "张三多大",
        ];
        for question in questions {
            assert_eq!(
                forked.answer(question),
                fresh.answer(question),
                "回答必须逐字节一致：{}",
                question
            );
        }
        // 接地清单（提及解码 + 链接记忆路径）也必须一致
        let forked_grounding = forked.grounding_mentions("张三的师傅是谁");
        let fresh_grounding = fresh.grounding_mentions("张三的师傅是谁");
        assert_eq!(
            format!("{:?}", forked_grounding),
            format!("{:?}", fresh_grounding),
            "接地清单必须一致"
        );

        // 固定对话序列：多轮上下文状态（焦点/历史）必须一致
        for turn in ["张三的师傅是谁", "那李四呢", "他的师傅是谁"] {
            let _ = forked.contextualize(turn);
            let _ = fresh.contextualize(turn);
        }
        assert_eq!(
            forked.contextualize("王五的师傅是谁"),
            fresh.contextualize("王五的师傅是谁")
        );
        assert_eq!(
            format!("{:?}", forked.context.focus_entities()),
            format!("{:?}", fresh.context.focus_entities())
        );
    }

    /// 分叉必须清零会话态：源实例用过的焦点/教学不得泄漏进新会话。
    #[test]
    fn fork_session_clears_session_state_only() {
        let config = test_config();
        let mut used = SymbolicReasoner::new(config, HashMap::new());
        used.observe_facts(&master_facts());
        let _ = used.contextualize("张三的师傅是谁");
        let _ = used.answer("张三的师傅是谁");
        assert!(!used.context.focus_entities().is_empty(), "源实例应有焦点");

        let forked = used.fork_session();
        assert!(
            forked.context.focus_entities().is_empty(),
            "分叉不得继承焦点"
        );
        assert_eq!(forked.revision(), used.revision(), "知识版本随分叉保留");
        // 知识仍在：同一问句回答与源实例一致
        assert_eq!(
            forked.answer("张三的师傅是谁"),
            used.answer("张三的师傅是谁")
        );
    }

    fn transitive_registry(
        predicate: &str,
        symmetric: bool,
    ) -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: predicate.to_string(),
            transitive: !symmetric,
            symmetric,
            ..Default::default()
        });
        registry
    }

    fn inverse_registry() -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            inverse: Some("徒弟".to_string()),
            ..Default::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "徒弟".to_string(),
            inverse: Some("师傅".to_string()),
            ..Default::default()
        });
        registry
    }

    /// 重复的传递谓词由 Closure 算子压缩，回答必须引用具体定律。
    #[test]
    fn law_closure_answers_repeated_transitive_chain() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        reasoner.configure_laws(&transitive_registry("上司", false));

        let answer = reasoner
            .answer("甲的上司的上司的上司是谁")
            .expect("传递律应支持重复谓词链");
        assert!(answer.contains("丁"), "{}", answer);
        assert!(answer.contains("推理依据"), "{}", answer);
        assert!(answer.contains("传递律"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);

        // 两跳问句必须恰好两跳：不能把三跳才可达的「丁」也算进来。
        let two_hops = reasoner
            .answer("甲的上司的上司是谁")
            .expect("两跳链应有答案");
        assert!(two_hops.contains("丙"), "{}", two_hops);
        assert!(!two_hops.contains("丁"), "{}", two_hops);
    }

    fn closure_reasoner() -> SymbolicReasoner {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        reasoner.configure_laws(&transitive_registry("上司", false));
        reasoner
    }

    #[test]
    fn all_quantifier_uses_closure_even_when_direct_edge_exists() {
        let reasoner = closure_reasoner();
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("所有问句应由闭包回答");
        assert!(answer.contains("乙"), "{}", answer);
        assert!(answer.contains("丙"), "{}", answer);
        assert!(answer.contains("丁"), "{}", answer);
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
        assert!(answer.contains("甲 → 乙 → 丙"), "{}", answer);
        assert!(!answer.contains("下界"), "{}", answer);
    }

    #[test]
    fn exact_hop_closure_does_not_need_transitive_law() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲恰好两跳的上司是谁")
            .expect("精确跳数无需传递律");
        assert!(answer.contains("丙"), "{}", answer);
        assert!(!answer.contains("1. 乙"), "{}", answer);
        assert!(!answer.contains("丁"), "{}", answer);
        assert!(answer.contains("共 1 个"), "{}", answer);
        assert!(answer.contains("精确跳数"), "{}", answer);
        assert!(!answer.contains("law:transitive"), "{}", answer);

        let count = reasoner
            .answer("甲恰好 2 跳的上司有几个")
            .expect("精确跳数计数");
        assert!(count.contains("共 1 个"), "{}", count);
    }

    #[test]
    fn closure_respects_session_negative_edges() {
        let mut reasoner = closure_reasoner();
        let message = reasoner.assert_nl_fact("乙不是甲的上司");
        assert!(message.contains("已记住否定事实"), "{}", message);
        let answer = reasoner.answer("甲的所有上司是谁").expect("否定后仍应回答");
        assert!(!answer.contains("乙"), "{}", answer);
        assert!(!answer.contains("丙"), "{}", answer);
        assert!(answer.contains("没有所有「上司」"), "{}", answer);
    }

    #[test]
    fn closure_includes_session_facts_in_reachability() {
        let mut reasoner = closure_reasoner();
        let message = reasoner.assert_nl_fact("丙的上司是丁");
        assert!(message.contains("已记住"), "{}", message);
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "丁".to_string(),
            rel_type: "上司".to_string(),
            to: "戊".to_string(),
        }]);
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("会话事实应参与闭包");
        assert!(answer.contains("戊"), "{}", answer);
        assert!(!answer.contains("共 3 个"), "{}", answer);
    }

    #[test]
    fn all_quantifier_without_transitive_law_refuses_to_claim_closure() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("无传递律时必须给出可解释的拒绝");
        assert!(answer.contains("尚未声明为传递关系"), "{}", answer);
        assert!(answer.contains("直接记录：乙"), "{}", answer);
        assert!(!answer.contains("共 2 个"), "{}", answer);
    }

    #[test]
    fn closure_truncation_reports_a_lower_bound() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        let names = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let facts: Vec<PrologFact> = names
            .windows(2)
            .map(|pair| PrologFact::Relation {
                from: pair[0].to_string(),
                rel_type: "上司".to_string(),
                to: pair[1].to_string(),
            })
            .collect();
        reasoner.observe_facts(&facts);
        reasoner.configure_laws(&transitive_registry("上司", false));

        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("超深链仍应给出有界答案");
        assert!(answer.contains("至少"), "{}", answer);
        assert!(answer.contains("深度上限"), "{}", answer);
        assert!(answer.contains("下界"), "{}", answer);
        assert!(answer.contains("至少 8 个"), "{}", answer);
        assert!(answer.contains("壬"), "{}", answer);
        assert!(!answer.contains("癸"), "{}", answer);
    }

    fn inverse_transitive_registry() -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "上司".to_string(),
            transitive: true,
            inverse: Some("下属".to_string()),
            ..Default::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "下属".to_string(),
            inverse: Some("上司".to_string()),
            ..Default::default()
        });
        registry
    }

    fn transitive_predicate_registry(
        predicate: &str,
    ) -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: predicate.to_string(),
            transitive: true,
            ..Default::default()
        });
        registry
    }

    #[test]
    fn truncated_law_closure_is_disclosed_in_chain_answer() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        let mut facts = vec![PrologFact::Relation {
            from: "甲".to_string(),
            rel_type: "上司".to_string(),
            to: "乙".to_string(),
        }];
        let names = ["乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        for pair in names.windows(2) {
            facts.push(PrologFact::Relation {
                from: pair[0].to_string(),
                rel_type: "上司".to_string(),
                to: pair[1].to_string(),
            });
        }
        facts.push(PrologFact::Relation {
            from: "壬".to_string(),
            rel_type: "师傅".to_string(),
            to: "子".to_string(),
        });
        reasoner.observe_facts(&facts);
        reasoner.configure_laws(&transitive_predicate_registry("上司"));

        let answer = reasoner
            .answer("甲的上司的师傅是谁")
            .expect("混合链应由律闭包补充");
        assert!(
            answer.contains("传递闭包扩展达到深度或节点上限"),
            "{}",
            answer
        );
        assert!(answer.contains("子"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
    }

    #[test]
    fn closure_respects_inverted_predicate_direction() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
        ]);
        reasoner.configure_laws(&inverse_transitive_registry());
        reasoner.register_predicate_query_surface("下属", "上司");
        let answer = reasoner
            .answer("丙的所有下属是谁")
            .expect("逆向闭包应有答案");
        assert!(answer.contains("甲"), "{}", answer);
        assert!(answer.contains("乙"), "{}", answer);
        assert!(answer.contains("共 2 个"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
    }

    #[test]
    fn closure_composes_with_following_relation() {
        let mut reasoner = closure_reasoner();
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "师傅".to_string(),
                to: "戊".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "师傅".to_string(),
                to: "己".to_string(),
            },
            PrologFact::Relation {
                from: "丁".to_string(),
                rel_type: "师傅".to_string(),
                to: "庚".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲的所有上司的师傅是谁")
            .expect("闭包后接关系应可复合");
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("戊"), "{}", answer);
        assert!(answer.contains("己"), "{}", answer);
        assert!(answer.contains("庚"), "{}", answer);
        assert!(answer.contains("甲 → 乙 → 戊"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);

        let count = reasoner
            .answer("甲的所有上司的师傅有几个")
            .expect("复合计数");
        assert!(count.contains("共 3 个"), "{}", count);
        assert!(count.contains("不同最终节点"), "{}", count);

        let blocked = reasoner.assert_nl_fact("戊不是乙的师傅");
        assert!(blocked.contains("已记住否定事实"), "{}", blocked);
        let after_negative = reasoner
            .answer("甲的所有上司的师傅是谁")
            .expect("否定边应阻断复合链");
        assert!(!after_negative.contains("戊"), "{}", after_negative);
        assert!(after_negative.contains("共 2 个"), "{}", after_negative);
        assert!(after_negative.contains("己"), "{}", after_negative);
        assert!(after_negative.contains("庚"), "{}", after_negative);
    }

    #[test]
    fn all_quantifier_count_uses_distinct_nodes() {
        let reasoner = closure_reasoner();
        let answer = reasoner
            .answer("甲的所有上司一共有几个")
            .expect("所有+计数应走闭包计数");
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("不同可达节点"), "{}", answer);
        assert!(!answer.contains("1. 乙"), "{}", answer);
    }

    /// 对称律补足单向事实：反向问句给出反方向取值，且说明依据。
    #[test]
    fn symmetric_law_recovers_reverse_direction() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "配偶".to_string(),
            to: "李四".to_string(),
        }]);
        reasoner.configure_laws(&transitive_registry("配偶", true));

        let answer = reasoner
            .answer("李四的配偶是谁")
            .expect("对称律应恢复反向命中");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(answer.contains("对称律"), "{}", answer);

        let check = reasoner
            .answer("李四的配偶是张三吗")
            .expect("校验问句也应使用对称律");
        assert!(check.starts_with("是的"), "{}", check);
        assert!(check.contains("对称律"), "{}", check);
    }

    /// 持久化账本的退役决定必须压过配置声明，并在后续知识重建中保持。
    #[test]
    fn persisted_retirement_disables_inverse_law_and_survives_rebuild() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        }]);
        reasoner.configure_laws(&inverse_registry());

        let before = reasoner
            .answer("李四的徒弟是谁")
            .expect("声明互逆律应支持反向问句");
        assert!(before.contains("张三"), "{}", before);

        let mut persisted = LawBook::from_registry(&inverse_registry());
        assert!(persisted.retire("law:inverse:师傅|徒弟"));
        assert!(reasoner.load_persisted_laws(persisted) >= 1);

        let after = reasoner.answer("李四的徒弟是谁");
        assert!(
            after.as_deref().map_or(true, |text| !text.contains("张三")),
            "{:?}",
            after
        );

        // 新事实触发缓存重建，退役决定不能被重建冲掉。
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "赵六".to_string(),
        }]);
        let rebuilt = reasoner.answer("李四的徒弟是谁");
        assert!(
            rebuilt
                .as_deref()
                .map_or(true, |text| !text.contains("张三")),
            "{:?}",
            rebuilt
        );
    }

    #[test]
    fn query_board_scans_reasoner_entities_and_relations() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
            PrologFact::Entity {
                name: "王治郅".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "214".to_string())]),
            },
            PrologFact::Relation {
                from: "姚明".to_string(),
                rel_type: "球队".to_string(),
                to: "火箭队".to_string(),
            },
        ]);
        let mut manager = crate::dialogue_manager::DialogueManager::new();
        let reply = manager
            .plan_reply("列出所有人的身高", &reasoner)
            .expect("查询板应读取推理器索引");
        assert!(reply.text.contains("姚明"), "{}", reply.text);
        assert!(reply.text.contains("226"), "{}", reply.text);
        let selected = manager
            .plan_reply("第一个呢", &reasoner)
            .expect("查询板应支持序号选择");
        assert!(selected.text.contains("当前选中"), "{}", selected.text);
        let compare = manager
            .plan_reply("跟王治郅比呢", &reasoner)
            .expect("查询板应支持实体比较");
        assert!(compare.text.contains("王治郅"), "{}", compare.text);
        assert!(compare.text.contains("214"), "{}", compare.text);
    }

    #[test]
    fn svo_and_passive_questions_compile_to_shared_relation_plans() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "认识".into(),
                to: "张三".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "王五".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let object = reasoner.answer("张三认识谁").expect("正向问宾语");
        assert!(object.contains("王五"), "{}", object);
        assert!(!object.contains("李四"), "{}", object);

        let subject = reasoner.answer("谁认识张三").expect("正向问主语");
        assert!(subject.contains("李四"), "{}", subject);

        let yes = reasoner.answer("张三认识王五吗").expect("是非问");
        assert!(yes.contains("是的"), "{}", yes);
        let no = reasoner.answer("张三认识李四吗").expect("是非问");
        assert!(no.contains("不是"), "{}", no);

        let passive_object = reasoner.answer("谁被张三认识").expect("被字句问受事");
        assert!(passive_object.contains("王五"), "{}", passive_object);
        let passive_subject = reasoner.answer("张三被谁认识").expect("被字句问施事");
        assert!(passive_subject.contains("李四"), "{}", passive_subject);
        let passive_yes = reasoner.answer("张三被李四认识吗").expect("被字句是非问");
        assert!(passive_yes.contains("是的"), "{}", passive_yes);
        let existence = reasoner
            .answer("张三有没有认识王五")
            .expect("有没有是非问");
        assert!(existence.contains("是的"), "{}", existence);
        let relative = reasoner
            .answer("认识张三的是谁")
            .expect("关系从句问施事");
        assert!(relative.contains("李四"), "{}", relative);
        let passive_relative = reasoner
            .answer("被张三认识的是谁")
            .expect("被字关系从句问受事");
        assert!(passive_relative.contains("王五"), "{}", passive_relative);
        let negated_false = reasoner
            .answer("张三不认识王五吗")
            .expect("否定是非问");
        assert!(negated_false.starts_with("不是"), "{}", negated_false);
        let negated_true = reasoner
            .answer("张三不认识李四吗")
            .expect("否定是非问");
        assert!(negated_true.starts_with("是的"), "{}", negated_true);
        let a_not_a = reasoner
            .answer("张三认不认识王五")
            .expect("A 不 A 是非问");
        assert!(a_not_a.starts_with("是的"), "{}", a_not_a);
        let passive_negated = reasoner
            .answer("张三没有被李四认识吗")
            .expect("被字句标记否定");
        assert!(passive_negated.starts_with("不是"), "{}", passive_negated);

        let synonyms = HashMap::from([("老张".to_string(), "张三".to_string())]);
        let mut aliased = SymbolicReasoner::new(test_config(), synonyms);
        aliased.observe_facts(&facts);
        let alias = aliased.answer("老张认识谁").expect("同义词别名问句");
        assert!(alias.contains("王五"), "{}", alias);

        let receipt = reasoner
            .query_rewrite_receipt("谁被张三认识")
            .expect("重写回执");
        assert!(receipt.contains("passive-wh-patient"), "{}", receipt);
        assert!(receipt.contains("FindObjectByPredicate"), "{}", receipt);

        let derivation = reasoner
            .answer_derivation("张三认识谁", Some("王五"))
            .expect("关系问句应可重建证明");
        assert!(derivation.is_proved(), "{:?}", derivation);
        assert_eq!(derivation.answer_value(), Some("王五"));
    }

    #[test]
    fn quantified_and_relative_questions_use_relation_plans() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "老师".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "学生".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "学生".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "教".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "教".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "教".into(),
                to: "张三".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "认识".into(),
                to: "李四".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let objects = reasoner
            .answer("张三教过哪些学生")
            .expect("量词名词短语问宾语");
        assert!(objects.contains("李四"), "{}", objects);
        assert!(objects.contains("王五"), "{}", objects);

        let subjects = reasoner
            .answer("哪些老师教过张三")
            .expect("量词名词短语问主语");
        assert!(subjects.contains("李四"), "{}", subjects);

        let relative_object = reasoner
            .answer("张三教过的学生是谁")
            .expect("关系从句问宾语");
        assert!(relative_object.contains("李四"), "{}", relative_object);
        assert!(relative_object.contains("王五"), "{}", relative_object);

        let relative_subject = reasoner
            .answer("教过张三的老师有哪些")
            .expect("关系从句问主语");
        assert!(relative_subject.contains("李四"), "{}", relative_subject);

        let count = reasoner
            .answer("张三教过多少个学生")
            .expect("关系计数问句");
        assert!(count.contains("共有 2 个"), "{}", count);
        assert!(count.contains("李四"), "{}", count);
        assert!(count.contains("王五"), "{}", count);

        let except_object = reasoner
            .answer("除了李四，张三还认识谁")
            .expect("集合差问宾语");
        assert!(except_object.contains("王五"), "{}", except_object);
        assert!(except_object.contains("已排除：李四"), "{}", except_object);

        let except_subject = reasoner
            .answer("除了张三，还有谁认识李四")
            .expect("集合差问主语");
        assert!(except_subject.contains("王五"), "{}", except_subject);
        assert!(except_subject.contains("已排除：张三"), "{}", except_subject);

        let receipt = reasoner
            .query_rewrite_receipt("张三教过的学生是谁")
            .expect("关系从句回执");
        assert!(receipt.contains("typed-relative-object"), "{}", receipt);
        let except_receipt = reasoner
            .query_rewrite_receipt("除了李四，张三还认识谁")
            .expect("集合差回执");
        assert!(except_receipt.contains("except-wh-object"), "{}", except_receipt);
    }

    fn trace_nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: HashMap::new(),
            name_patterns: crate::NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: vec!["张三".to_string(), "李四".to_string(), "王五".to_string()],
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn failed_answer_records_classified_read_trace_without_raw_text() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());
        assert!(reasoner.read_trace_receipts().is_empty());

        let answer = reasoner.answer("未名科技公司的老板是谁");
        assert!(answer.is_none(), "未知实体不应被静默猜测：{:?}", answer);
        let receipts = reasoner.read_trace_receipts();
        assert_eq!(receipts.len(), 1, "失败轮应留下一条有界回执");
        let receipt = &receipts[0];
        assert!(
            receipt
                .failure_kinds
                .iter()
                .any(|kind| kind == "unknown_entity"),
            "失败应分类为未知实体：{:?}",
            receipt
        );
        assert_eq!(receipt.answered, Some(false));
        assert!(
            receipt.source_counts.values().sum::<u64>() > 0,
            "回执应记录词性来源分布"
        );
        let report = reasoner.read_trace_ledger_report();
        assert!(!report.contains("未名科技"), "台账不得保存原文：{}", report);

        // 成功轮不做额外失败记录（也只观测失败）。
        let answer = reasoner.answer("张三的师傅是谁");
        assert!(answer.is_some(), "已登记实体应正常回答");
        assert_eq!(reasoner.read_trace_receipts().len(), 1);
    }

    #[test]
    fn lexical_trace_report_exposes_marker_boundary_without_changing_answer() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());

        let report = reasoner.lexical_trace_report("张三的所有权归公司");
        assert!(report.contains("读法追踪"), "{}", report);
        assert!(report.contains("边界越界"), "{}", report);
        assert!(report.contains("事后重建"), "{}", report);
        assert!(!report.contains("失败分类：无"), "{}", report);

        // 观测不改变回答路径：普通闭包/多跳问句仍走原实现。
        let answer = reasoner.answer("张三的师傅的师傅是谁");
        assert!(answer.is_some(), "原多跳回答不应被追踪改变");
    }
}

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::abduction::{
    self, AbductionEngine, AbductionReport, ClarificationReply, ClarificationSession, Hypothesis,
    Scenario,
};
use crate::association_miner::{display_item, AssociationMiner, AssociationRule};
use crate::causal_reasoning::{CausalAnalyzer, CausalGraph, CounterfactualDelta};
use crate::causal_tokenization::{
    attribute_tokens, AnchorKind, CausalTokenizer, MAX_TOKEN_CHARS,
};
use crate::chinese_reasoning_nlu::{self, LessonKind};
use crate::discourse::DiscoursePlanner;
use crate::entity_linker::{EntityLinker, LinkMethod, LinkedMention};
use crate::graph_walk_embedding::GraphWalkEmbedding;
use crate::holographic_memory::HolographicMemory;
use crate::link_prediction::{HornRule, LinkPredictor, Prediction, RuleTemplate, Triple};
use crate::markov_logic::{MarkovLogicEngine, SoftFact, WeightedRule};
use crate::neuro_symbolic_contrast::ContrastSnapshot;
use crate::problog::{ProblogAnswer, ProblogEngine};
use crate::ontology_alignment::{OntologyAligner, TermKind};
use crate::perf_ledger::PerfLedger;
use crate::prolog_engine::PrologFact;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::resolution_engine::{
    render_term, Atom, Clause, DerivedFact, ProofNode, ProofSource, ResolutionEngine, Solution,
    Term,
};
use crate::semantic_embedding::SymbolicEmbedding;
use crate::semantic_entailment::{
    self, EntailmentResult, Knowledge, PropositionParser, Verdict,
};
use crate::symbolic_generator::{GenerationOrder, SymbolicGenerator};
use crate::ReasoningConfig;

// 归结引擎内部谓词：has(E, 谓词名) / type(E, 类型名)，用于承载关联规则的物品
const PROOF_HAS: &str = "__具有__";
const PROOF_TYPE: &str = "__类型__";

// 中文教学知识类型的展示名
fn lesson_kind_label(kind: LessonKind) -> &'static str {
    match kind {
        LessonKind::DeterministicFact => "确定事实",
        LessonKind::ProbabilisticFact => "概率事实",
        LessonKind::ConditionalRule => "条件规则（如果…那么…）",
        LessonKind::CausalRule => "因果规则（因为…所以…）",
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
    pub last_predicate: Option<String>,
    pub history: Vec<String>,
    #[serde(default)]
    pub generation: Option<GenerationState>,
    // 跨会话临时信念：被采纳的假设事实（会话级，不写知识图谱）
    #[serde(default)]
    pub session_facts: Vec<SnapshotAtom>,
    // 用户用中文教过的概率知识原句（会话级，跨会话持久化后自动重放）
    #[serde(default)]
    pub lessons: Vec<String>,
    // 用户用中文教过的 Horn 规则（会话级后向链规则库，跨会话持久化）
    #[serde(default)]
    pub session_clauses: Vec<SnapshotClause>,
    // 神经符号对比仲裁记忆（校准统计、待定推测、屏蔽/否证集），跨会话持久化
    #[serde(default)]
    pub contrast: Option<ContrastSnapshot>,
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

pub struct DialogueContext {
    focus_entities: Vec<String>,
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
        for entity in entities.iter().rev() {
            if entity.is_empty() {
                continue;
            }
            self.focus_entities.retain(|e| e != entity);
            self.focus_entities.insert(0, entity.clone());
        }
        if self.focus_entities.len() > self.max_focus {
            self.focus_entities.truncate(self.max_focus);
        }
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
            last_predicate: self.last_predicate.clone(),
            history: self.history.clone(),
            generation: self.generation.clone(),
            session_facts: Vec::new(),
            lessons: Vec::new(),
            session_clauses: Vec::new(),
            contrast: None,
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueSnapshot) {
        self.focus_entities = snapshot.focus_entities.clone();
        self.focus_entities.truncate(self.max_focus);
        self.last_predicate = snapshot.last_predicate.clone();
        self.history = snapshot.history.clone();
        self.history.truncate(self.max_history);
        self.generation = snapshot.generation.clone();
    }

    pub fn resolve_pronouns(&self, input: &str) -> Option<String> {
        let focus = self.focus()?;
        const PRONOUNS: [&str; 7] = ["他们", "她们", "它们", "这个人", "他", "她", "它"];
        let mut result = String::with_capacity(input.len() + focus.len());
        let mut replaced = false;
        let mut chars = input.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            // 在字节切片上直接判断前缀
            let mut matched: Option<&str> = None;
            for pronoun in PRONOUNS {
                if input[i..].starts_with(pronoun) {
                    matched = Some(pronoun);
                    break;
                }
            }
            if let Some(pronoun) = matched {
                let previous = input[..i].chars().next_back();
                let blocked = matches!(previous, Some('其') | Some('无'));
                if !blocked {
                    result.push_str(focus);
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
}

#[derive(Debug, Clone)]
struct ChainStep {
    subject: String,
    predicate: String,
    hits: Vec<ValueHit>,
}

/// 问句链上的一个谓词环节：词面（用户问法）+ 规范槽位（存储坐标）+ 是否逆向。
#[derive(Debug, Clone)]
struct ChainPredicate {
    surface: String,
    slot: String,
    inverted: bool,
}

pub struct SymbolicReasoner {
    config: ReasoningConfig,
    aligner: OntologyAligner,
    predictor: LinkPredictor,
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
    // 实体属性三元组（观察到的），供归结引擎编码事实
    attributes: HashSet<(String, String, String)>,
    // 关联规则挖掘器（类型/谓词/属性篮子的支持度-置信度-提升度）
    associations: AssociationMiner,
    // 一阶归结引擎（事实 + 链接预测规则 + 关联规则）
    resolver: ResolutionEngine,
    // 溯因·假设生成引擎（后向链失败前沿 → 候选假设 → 反事实模拟）
    abduction: AbductionEngine,
    // 会话级临时事实（被采纳的假设；跨会话持久化，但永不写知识图谱）
    session_facts: Vec<Atom>,
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
    // 单条链接结果记忆：同一文本在一次查询流程中会被多处链接（上下文记录/链接说明/回答），
    // 观察事实或重建缓存（线索变化）时整体失效
    link_memo: parking_lot::Mutex<Option<(String, Arc<Vec<LinkedMention>>)>>,
    // 实体提及记忆：record_vocabulary 对同一文本会多次调用，且每次需全量扫描实体剖面
    entity_mentions_memo: parking_lot::Mutex<Option<(String, Vec<String>)>>,
    // 同义词命中的规范谓词集合记忆
    predicate_canonicals_memo: parking_lot::Mutex<Option<(String, Arc<HashSet<String>>)>>,
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
}

impl SymbolicReasoner {
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
        let generator = SymbolicGenerator::new(
            config.max_profile_sentences,
            config.completion_min_score,
        );
        let associations = AssociationMiner::new(
            config.association_min_support,
            config.association_min_confidence,
            config.association_min_lift,
        );
        let resolver = ResolutionEngine::new(
            config.resolution_max_depth,
            config.resolution_max_solutions,
        );
        let abduction = AbductionEngine::new(
            config.resolution_max_depth,
            config.abduction_max_hypotheses,
        );
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
            attributes: HashSet::new(),
            associations,
            resolver,
            abduction,
            session_facts: Vec::new(),
            session_clauses: Vec::new(),
            clarification: None,
            last_abduction_goal: None,
            last_abduction_vars: Vec::new(),
            known_predicates_cache: Vec::new(),
            known_predicates_normalized_cache: Vec::new(),
            proposition_surfaces: Vec::new(),
            link_memo: parking_lot::Mutex::new(None),
            entity_mentions_memo: parking_lot::Mutex::new(None),
            predicate_canonicals_memo: parking_lot::Mutex::new(None),
            inverse_query_map: HashMap::new(),
            discourse,
            holo,
            revision: 0,
            derived_stamps: BTreeMap::new(),
            dependency_stamps: BTreeMap::new(),
            perf,
        };
        reasoner.rebuild_knowledge_caches();
        reasoner
    }

    /// 当前知识版本（每次内容确有变化的观察 +1）。
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// 共享的性能台账（与 `OxigraphManager` 可为同一实例）。
    pub fn perf_ledger(&self) -> Arc<PerfLedger> {
        Arc::clone(&self.perf)
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
    /// 「纯重复观察零重建」因此可审计：版本不动 → 台账不动。
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
        // 词表变化后实体链接/实体提及的旧结果可能过时，同步失效记忆缓存
        *self.link_memo.lock() = None;
        *self.entity_mentions_memo.lock() = None;
        *self.predicate_canonicals_memo.lock() = None;
        self.stamp_rebuild("knowledge_caches");
    }

    pub fn context(&self) -> &DialogueContext {
        &self.context
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

    /// 登记逆向谓词词面：问句用该词面发问时，命中规范槽位三元组的反方向
    ///（「徒弟」问句 ↔「师傅」事实）。只影响查询换向，不改词面/叙事表。
    pub fn register_predicate_query_surface(&mut self, surface: &str, slot: &str) {
        let (surface, slot) = (surface.trim(), slot.trim());
        if surface.is_empty() || slot.is_empty() || surface == slot {
            return;
        }
        self.aligner.register_term(surface, TermKind::Relation);
        self.aligner.register_term(slot, TermKind::Relation);
        let changed = self.inverse_query_map.get(surface) != Some(&(slot.to_string(), true));
        self.inverse_query_map
            .insert(surface.to_string(), (slot.to_string(), true));
        if changed {
            self.rebuild_knowledge_caches();
        }
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
                    self.linker.register_entity(name, Some(entity_type.as_str()));
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
        // 纯重复观察（快照重放）→ 版本不动、memo 不清、派生零重建
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
        *self.link_memo.lock() = None;
        *self.entity_mentions_memo.lock() = None;
        *self.predicate_canonicals_memo.lock() = None;
        // 规则只依赖已积累的三元组与固定阈值：若本次没有新增三元组，
        // 重新挖掘必然得到相同规则，直接跳过
        if triples_added && self.config.enable_link_prediction {
            self.predictor.mine_rules(
                self.config.link_prediction_min_support,
                self.config.link_prediction_min_confidence,
            );
            self.stamp_rebuild("link_rules");
        }
        // 关联规则与前向/反向归结都只依赖已观察的篮子与事实：无新增则跳过重建
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
        // 首次需要嵌入时才做全量幂迭代（查询结果与即时重建完全一致）；
        // PPMI 嵌入自带 dirty 跳过，纯重复观察时连调用都省去
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
            engine.add_fact(Atom::rel(&triple.predicate, &triple.subject, &triple.object));
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
        let pronoun_resolved = self.context.resolve_pronouns(input);
        let ellipsis_base = pronoun_resolved.as_deref().unwrap_or(input);
        let rewritten = self
            .resolve_ellipsis(ellipsis_base)
            .or(pronoun_resolved);
        self.record_vocabulary(input);
        if let Some(rewritten_text) = &rewritten {
            self.record_vocabulary(rewritten_text);
        }
        rewritten
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
            self.context.record_entities(&entities);
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
        // 同一文本会在一轮对话中被多次询问，做单条记忆复用；
        // 实体剖面或同义词表只在 observe_facts 中变化，届时整体失效
        if let Some((cached_text, found)) = self.entity_mentions_memo.lock().as_ref() {
            if cached_text == text {
                self.perf.bump("memo_entity_mentions_hit");
                return found.clone();
            }
        }
        self.perf.bump("memo_entity_mentions_miss");
        let found = self.entities_mentioned_uncached(text);
        *self.entity_mentions_memo.lock() = Some((text.to_string(), found.clone()));
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
                found.push((profile.term.chars().count(), self.aligner.canonical_form(&profile.term)));
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
        if let Some((cached_text, canonicals)) = self.predicate_canonicals_memo.lock().as_ref() {
            if cached_text == text {
                self.perf.bump("memo_predicate_canonicals_hit");
                return canonicals.clone();
            }
        }
        self.perf.bump("memo_predicate_canonicals_miss");
        let mut mentioned: HashSet<String> = HashSet::new();
        for (alias, canonical) in self.aligner.synonyms() {
            if text.contains(alias.as_str()) {
                mentioned.insert(canonical.clone());
            }
        }
        let mentioned = Arc::new(mentioned);
        *self.predicate_canonicals_memo.lock() = Some((text.to_string(), mentioned.clone()));
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
    const LINK_ACCEPT_THRESHOLD: f64 = 0.5;

    /// 文本的实体链接结果：一次查询流程内同一文本会被多处链接，做单条记忆复用
    fn linked_mentions(&self, text: &str) -> Arc<Vec<LinkedMention>> {
        if let Some((cached_text, mentions)) = self.link_memo.lock().as_ref() {
            if cached_text == text {
                self.perf.bump("memo_link_hit");
                return mentions.clone();
            }
        }
        self.perf.bump("memo_link_miss");
        let hints = self.link_hints(text);
        let mentions = Arc::new(self.linker.link_with_context(text, &hints));
        *self.link_memo.lock() = Some((text.to_string(), mentions.clone()));
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
        self.discourse.teach_axiom(trimmed)
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
        let mut text = format!("换个角度介绍「{}」（{}）：\n{}", state.entity, label, page.text);
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
        PropositionParser::with_surfaces(
            &self.aligner,
            &self.linker,
            &self.proposition_surfaces,
        )
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
            _ => Atom::rel(&proposition.predicate, &proposition.subject, &proposition.object),
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
                                    Atom::pattern(
                                        &predicate.slot,
                                        next.clone(),
                                        previous.clone(),
                                    )
                                } else {
                                    Atom::pattern(
                                        &predicate.slot,
                                        previous.clone(),
                                        next.clone(),
                                    )
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
        //（`report.provable == false` 已确认目标不可证，跳过重复的可证性检查）
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
            return "溯因推理未启用（config.toml [reasoning] enable_abduction = false）。".to_string();
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
        //（`report.provable == false` 已确认目标不可证，跳过重复的可证性检查）
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
            self.aligner.register_term(&atom.predicate, TermKind::Relation);
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

    /// 把一条基事实从会话事实集中撤回（不触碰知识图谱；软逻辑/概率程序同步重建）
    fn retract_atom(&mut self, atom: &Atom) -> bool {
        // 只撤回会话事实：归结引擎里同样载入知识图谱事实，直接 remove 会误删图谱知识
        if !self.session_facts.iter().any(|existing| existing == atom) {
            return false;
        }
        self.session_facts.retain(|existing| existing != atom);
        // 归结引擎重建后图谱事实仍在、被撤回的会话事实消失
        if self.config.enable_resolution {
            self.rebuild_resolver();
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

    // ------------------------- 对话式澄清（把"猜"升级为"问"） -------------------------

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
                "\n  → 目标「{}」现在已可证明（置信度 {:.2}）",
                goal,
                solution.confidence
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
                goal,
                solution.confidence
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
        let result = self.abduction.what_if(
            &self.resolver,
            &assumptions,
            self.resolver.max_depth(),
            20,
        );
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
            note.push_str(&format!("，未通过 {} 句：{}）", failed.len(), failed.join("；")));
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
        if self.generator.is_known_entity(&candidate) || self.associations.basket(&candidate).is_some()
        {
            return Some(candidate);
        }
        self.best_linked_entity(trimmed).filter(|entity| {
            self.generator.is_known_entity(entity) || self.associations.basket(entity).is_some()
        })
    }

    // 解析 "S的P1的P2...是O"（谓词串可为空 ⇒ 类型断言）
    fn parse_chain_statement(
        &self,
        text: &str,
    ) -> Option<(String, Vec<ChainPredicate>, String)> {
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
            let sentence = self
                .generator
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

    fn render_resolution_result(&self, phrase: &str, goals: &[Atom]) -> String {
        let solutions = self.resolver.solve(goals, &[]);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结证明（一阶 Horn 归结 + 回溯证明树，纯符号）：\n");
        output.push_str(&format!("  目标：{}  〔逻辑形式：{}〕\n", phrase, goal_display));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则（含链接预测与关联规则）\n",
            self.resolver.fact_count(),
            self.resolver.clause_count()
        ));
        if let Some(solution) = solutions.first() {
            output.push_str(&format!("  得证：证明置信度 {:.2}\n", solution.confidence));
            output.push_str("  证明树：\n");
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
        } else {
            output.push_str(&format!(
                "  未得证：在深度上限 {} 内未找到归结反驳（开世界假设：推不出 ≠ 为假）。\n",
                self.resolver.max_depth()
            ));
            output.push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        }
        output
    }

    // "X有P" 形式的存在性目标：由关联规则/前向推导证明
    fn prove_existence_statement(&self, text: &str) -> Option<String> {
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
        Some(self.render_resolution_result(&format!("{} 具有 {}", entity, predicate), &goals))
    }

    /// 证明一条陈述（"S的P…是O" / "S是类型" / "S有P"），输出回归结证明树
    pub fn resolution_proof_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请提供要证明的陈述，例如：证明：张三的师祖是王五 或 证明：姚明有国籍".to_string();
        }
        if let Some(report) = self.prove_existence_statement(trimmed) {
            return report;
        }
        match self.parse_chain_statement(trimmed) {
            Some((entity, predicates, object)) => {
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
                self.render_resolution_result(&phrase, &goals)
            }
            None => format!(
                "无法把「{}」解析为可归结的目标。支持句式：S的P…是O / S是类型 / S有P。",
                trimmed
            ),
        }
    }

    /// 开放目标（"X的P…是谁"）：返回全部归结解与各解的证明链
    pub fn resolution_query_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', ' ']);
        if trimmed.is_empty() {
            return "请提供推理目标，例如：推理链：张三的师祖是谁".to_string();
        }
        let mut core = trimmed.to_string();
        for suffix in [
            "是谁", "是什么", "是多少", "是哪里", "是哪个", "是哪儿", "是哪位", "是哪些", "是几",
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
        let mut goals = Vec::new();
        let mut previous = Term::constant(&entity);
        for (index, predicate) in predicates.iter().enumerate() {
            let next = if index + 1 == predicates.len() {
                Term::var(&answer_var)
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
        let solutions = self.resolver.solve(&goals, std::slice::from_ref(&answer_var));
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
        if solutions.is_empty() {
            output.push_str("  没有找到解（开世界假设：推不出 ≠ 不存在）。\n");
            return output;
        }
        for (index, solution) in solutions.iter().enumerate() {
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

    /// 前向饱和推导报告：多跳新事实（含前提与规则来源）
    pub fn derived_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
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
        self.aligner.kind_of(name) == Some(TermKind::Entity)
    }

    fn is_known_predicate(&self, name: &str) -> bool {
        self.aligner.kind_of(name) == Some(TermKind::Relation)
            || self.inverse_query_map.contains_key(name)
            || self.inverse_query_map.contains_key(self.aligner.canonical_form(name).as_str())
            || self.predictor.has_predicate(name)
            // 正向别名（同义词）按规范化槽位回查预测器
            || {
                let canonical = self.aligner.canonical_form(name);
                canonical != name && self.predictor.has_predicate(&canonical)
            }
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
            self.predictor
                .subjects_of(&predicate.slot, node)
        } else {
            self.predictor
                .objects_of(node, &predicate.slot)
        };
        for object in objects {
            if seen.insert(object.clone()) {
                hits.push(ValueHit {
                    value: object,
                    prediction: None,
                });
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
                    });
                }
            }
        }
        hits.sort_by(|a, b| a.value.cmp(&b.value));
        hits
    }

    fn run_chain(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
    ) -> Option<(Vec<ChainStep>, Vec<ValueHit>)> {
        if predicates.is_empty() {
            return None;
        }
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            // 多个分支可能命中同一实体，边收集边去重：两个集合插入条件相同，
            // 合并为一个即可（保留首次出现顺序）
            let mut seen_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let hits = self.node_hits(node, predicate);
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
        Some((steps, last_hits))
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
    fn expected_type_keywords(question: &str) -> Vec<&'static str> {
        if question.contains('谁') || question.contains("何人") || question.contains("哪位") {
            vec!["人", "person"]
        } else if question.contains("哪里")
            || question.contains("哪儿")
            || question.contains("什么地方")
            || question.contains("何处")
        {
            vec!["地", "城市", "国家", "place", "location", "country", "city"]
        } else if question.contains("什么时候") || question.contains("哪年") || question.contains("何时") {
            vec!["日期", "时间", "date", "time", "年"]
        } else {
            Vec::new()
        }
    }

    fn value_matches_expectation(&self, value: &str, keywords: &[&str]) -> bool {
        let Some(etype) = self.generator.entity_type_of(value) else {
            return false;
        };
        let etype_lower = etype.to_lowercase();
        // 期望类型关键词均为小写（或中文）
        keywords.iter().any(|keyword| {
            etype.contains(*keyword) || etype_lower.contains(*keyword)
        })
    }

    fn generate_find_object(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        steps: &[ChainStep],
        last_hits: &[ValueHit],
        question: &str,
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
                    output.push_str(&format!(
                        "（推断，置信度 {:.2}）",
                        prediction.confidence
                    ));
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
            if reordered {
                output.push_str(&format!(
                    "\n  说明：已按问题期望的答案类型（{}）优先排序。",
                    expectations.join("/")
                ));
            }
        }
        output
    }

    fn answer_planned(&self, question: &str) -> Option<String> {
        let plan = self.planner.parse(question)?;
        match plan {
            QueryPlan::FindObject { segments } => {
                let (entity, predicates) = self.split_chain(&segments)?;
                if predicates.is_empty() {
                    return None;
                }
                if let Some((steps, last_hits)) = self.run_chain(&entity, &predicates) {
                    return Some(self.generate_find_object(
                        &entity,
                        &predicates,
                        &steps,
                        &last_hits,
                        question,
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
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => self.answer_check(&subject_segments, &object),
            QueryPlan::FindSubject { object, predicate } => {
                self.answer_find_subject(&object, &predicate)
            }
            QueryPlan::Similar { entity } => self.answer_similar(&entity),
            QueryPlan::Analogy { a, b, c } => self.answer_analogy(&a, &b, &c),
        }
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

        if let Some((steps, last_hits)) = self.run_chain(&entity, &intermediate) {
            if let Some(hit) = last_hits.iter().find(|hit| hit.value == object) {
                let step = steps.last()?;
                return Some(match &hit.prediction {
                    None => format!(
                        "是的。根据知识图谱：{} 的 {} 是 {}。",
                        step.subject, step.predicate, object
                    ),
                    Some(prediction) => format!(
                        "很可能是的（符号推断）。{}",
                        self.predictor.explain(prediction)
                    ),
                });
            }
            let known_values: Vec<String> = last_hits
                .iter()
                .filter(|hit| hit.prediction.is_none())
                .map(|hit| hit.value.clone())
                .collect();
            if !known_values.is_empty() {
                let step = steps.last()?;
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

    fn answer_find_subject(&self, object_raw: &str, predicate_raw: &str) -> Option<String> {
        let object = self.aligner.canonical_form(object_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        if object.is_empty() || slot.is_empty() {
            return None;
        }
        let surface = predicate_raw.trim().to_string();
        // 反查主体：正向找槽位三元组的主语；逆向词面（如「徒弟」）语义即
        // "谁的槽位值是 object"，等价于找槽位三元组的宾语（object 作为主语）
        let mut subjects: Vec<String> = if inverted {
            self.predictor.objects_of(&object, &slot)
        } else {
            self.predictor.subjects_of(&slot, &object)
        };
        let mut evidence: Vec<String> = Vec::new();
        if !inverted && self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            // 用哈希集合做成员检查
            let mut seen_subjects: HashSet<String> = subjects.iter().cloned().collect();
            for prediction in self
                .predictor
                .predict_for_entity(&object, Some(&slot), limit)
            {
                if prediction.triple.object == object && prediction.triple.subject != object {
                    if seen_subjects.insert(prediction.triple.subject.clone()) {
                        subjects.push(prediction.triple.subject.clone());
                    }
                    evidence.push(self.predictor.explain(&prediction));
                }
            }
        }
        if subjects.is_empty() {
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
        if let Some(answer) = self.answer_planned(question) {
            return Some(answer);
        }
        self.answer_by_prediction(question)
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
            for prediction in self.predictor.predict_for_entity(
                entity,
                None,
                self.config.max_predictions * 4,
            ) {
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
            return format!("结构联想中没有「{}」：它没有出现在任何关系三元组中。", entity);
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
        let inferred: Vec<SoftFact> = all
            .iter()
            .filter(|fact| !fact.from_fact)
            .cloned()
            .collect();
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
            output.push_str(&format!("  {}. {}\n", index + 1, self.render_soft_fact(fact)));
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
        let mentions_entity =
            |atom: &Atom| atom.terms.iter().any(|term| term.name() == entity);
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
                        if link.is_inhibitory() { "，抑制" } else { "" },
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
            return format!("没有可做梯度归因的加权推测（「{}」未被任何规则支持）。", entity);
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
            output.push_str("  • 条件句：如果S的P是O，那么S的P是O（可带 八成/很可能/百分之八十 等概率词）\n");
            output.push_str("  • 因果句：因为S的P是O，所以S的P是O\n");
            output.push_str("  • 比例句：八成的S会V / S的P是O的可能性是百分之八十\n");
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
    fn register_lesson_clauses(&mut self, lessons: &[chinese_reasoning_nlu::ChineseLesson]) -> usize {
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
                    && !new_predicates.iter().any(|p| p.as_str() == atom.predicate.as_ref())
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
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
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
            output.push_str("  • 如果S的P是O，那么S的Q是O（S/O 用 X/Y/Z、甲乙丙、A/B/C 等占位符表示变量）\n");
            output.push_str("  • 因为S的P是O，所以S的Q是O\n");
            for error in &outcome.errors {
                output.push_str(&format!("  失败原因：{}\n", error));
            }
            return output.trim_end().to_string();
        }
        let mut output = String::from("已学习中文规则（纯符号解析 → Horn 子句，直接进入后向链）：\n");
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
    pub fn assert_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出要记住的事实，例如：记住：张三是李四的师傅".to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let mut remembered: Vec<String> = Vec::new();
        let mut refused: Vec<String> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact {
                refused.push(format!(
                    "「{}」不是确定事实（条件句请用「规则：」学习，概率句请用「教学：」）",
                    lesson.original
                ));
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
            if self.adopt_atom(lesson.head.clone()) {
                remembered.push(rendered);
            } else {
                refused.push(format!("「{}」此前已记住", rendered));
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
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let mut removed: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
                continue;
            }
            let rendered = self.render_atom_nl(&lesson.head);
            if self.retract_atom(&lesson.head) {
                removed.push(rendered);
            } else {
                missing.push(rendered);
            }
        }
        let mut output = String::new();
        if removed.is_empty() && missing.is_empty() {
            output.push_str("没能把这句话解析为可撤回的事实（会话事实不会来自知识图谱，只撤回会话级记忆）。\n");
        }
        for fact in &removed {
            output.push_str(&format!("已忘记会话事实：{}\n", fact));
        }
        for fact in &missing {
            output.push_str(&format!("会话事实中没有「{}」（知识图谱事实不在此处撤回）。\n", fact));
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
            output.push_str("  （还没有学习过中文规则；可输入「规则：如果X的师傅是Y，那么X的老师是Y」）\n");
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
        output
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
            return "请输入要精确查询的对象，例如：概率推理：李四的师祖 或 概率推理：李四".to_string();
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
                    atom.predicate.contains(predicate) || predicate.contains(atom.predicate.as_ref())
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
                format!("无法在「{}」中识别出已知实体（也没有教过与「{}」相关的概率知识）。", target, entity)
            };
        }
        let mut merged: HashMap<String, ProblogAnswer> = HashMap::new();
        let mut stats: Option<(usize, usize, usize, usize, bool)> = None;
        for pattern in &patterns {
            let query = self.problog.query(
                pattern,
                8,
                self.config.soft_logic_min_probability.max(0.05),
            );
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
                soft_probabilities
                    .get(&answer.atom)
                    .copied()
                    .unwrap_or(0.0)
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
            // 先做前缀预筛（`consider` 内部同样要求），避免为不匹配的
            // profile 做规范化与缓存查询
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
        let remainder = remainder.trim().trim_start_matches(['的', '：', ':', ' ', '是']);
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
        let session_info = if self.session_facts.is_empty() {
            "（无）".to_string()
        } else {
            format!("{} 条", self.session_facts.len())
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
        format!(
            "对话记忆：焦点实体栈 [{}]，上一轮谓词「{}」，上一段生成 {}，历史问题 {} 条，会话假设 {}，会话规则 {} 条，澄清 {}，中文概率知识 {}。",
            focus,
            predicate,
            generation,
            self.context.history().len(),
            session_info,
            self.session_clauses.len(),
            clarify,
            lesson_info
        )
    }

    pub fn save_dialogue_state(&self, path: &str) -> std::io::Result<()> {
        self.save_dialogue_state_with_contrast(path, None)
    }

    /// 保存对话状态，并额外写入神经符号对比仲裁记忆（校准统计/待定推测/否证集）
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
        snapshot.lessons = self.lessons.clone();
        snapshot.session_clauses = self.session_clauses.iter().map(snapshot_clause).collect();
        snapshot.contrast = contrast.cloned();
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load_dialogue_state(&mut self, path: &str) -> bool {
        self.load_dialogue_state_with_contrast(path).0
    }

    /// 加载对话状态，同时返回其中的对比仲裁记忆快照（无则 None）
    pub fn load_dialogue_state_with_contrast(
        &mut self,
        path: &str,
    ) -> (bool, Option<ContrastSnapshot>) {
        let Ok(content) = std::fs::read_to_string(path) else {
            return (false, None);
        };
        let Ok(snapshot) = serde_json::from_str::<DialogueSnapshot>(&content) else {
            return (false, None);
        };
        let contrast = snapshot.contrast.clone();
        self.apply_dialogue_snapshot(&snapshot);
        (true, contrast)
    }

    fn apply_dialogue_snapshot(&mut self, snapshot: &DialogueSnapshot) {
        self.context.restore(snapshot);
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
            self.session_facts = restored;
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
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
                if restored
                    .iter()
                    .any(|existing: &Clause| existing.head == clause.head && existing.body == clause.body)
                {
                    continue;
                }
                for atom in std::iter::once(&clause.head).chain(clause.body.iter()) {
                    if !self.aligner.is_registered(&atom.predicate)
                        && !new_predicates.iter().any(|p| p.as_str() == atom.predicate.as_ref())
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
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存对话状态");

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
            notes.iter().any(|note| note.contains("张三") && note.contains("→")),
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
        assert!(variants.lines().count() >= 2, "应有多种合规句式：{}", variants);
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
        assert!(summary.contains("动物"), "教学回执应包含公理内容：{}", summary);
        let text = reasoner.narrate("苏格拉底").expect("叙述应可用");
        assert!(
            text.contains("动物"),
            "本体继承推理应进入叙述（定义/推导链）：{}",
            text
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
        assert!(unknown[0].reason.contains("尚未收录"), "实际原因：{}", unknown[0].reason);

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
        assert_eq!(reasoner.context().generation_state().map(|s| s.entity.as_str()), Some("张三"));
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
        assert!(profile.contains("零幻觉"), "裸谓词生成应通过自检：{}", profile);
    }

    #[test]
    fn test_continue_generation_and_switch_angle() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());

        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        assert!(first.contains("共 1 条事实"), "实际：{}", first);
        let second = reasoner
            .continue_generation()
            .expect("接着说应返回下一页");
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
        assert!(
            person_at < company_at,
            "人物类型答案应排在前面：{}",
            answer
        );
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
    fn test_resolution_proves_existence_via_association_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.resolution_proof_report("赵六有国籍");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(report.contains("关联规则"), "应有归结算子句来源：{}", report);
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
        assert!(
            report.contains("第 1 轮推导"),
            "应给出推导轮次：{}",
            report
        );
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
        assert!(reasoner.adopt_hypothesis(1).is_none(), "无待定假设时应返回 None");
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("未得证"));
        let message = reasoner
            .adopt_hypothesis(1)
            .expect("应能采纳第 1 条假设");
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
        assert!(report.contains("师祖"), "应沿师傅→师祖规则推出新结论：{}", report);
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
        assert!(response.contains("已可证明"), "确认后目标应可证明：{}", response);
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
        let path = std::env::temp_dir().join(format!(
            "knowprolog_session_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "跨会话恢复的假设应参与证明：{}", proof);
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
        assert!(report.contains("李四"), "直接邻居应在结构联想中：{}", report);
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
        assert!(report.contains("张三"), "焦点实体应出现在报告中：{}", report);
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
        assert!(reasoner.structural_related_report("张三").contains("未启用"));
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
        assert!(report.contains("规则贡献"), "应逐条列出规则贡献：{}", report);
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
        assert!(reasoner.counterfactual_report("张三的师傅", false).contains("未启用"));
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
        assert!(report.contains("张三"), "输入实体应出现在归因中：{}", report);
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
        assert!(report.contains("与软逻辑"), "应给出与 MLN-lite 的对照：{}", report);
        assert!(report.contains("P=0.") || report.contains("P=1."), "应给出概率：{}", report);
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
        assert!(report.contains("中文教学"), "证明路径应保留教学来源：{}", report);
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
        assert!(report.contains("P=0.800"), "应给出精确概率 0.800：{}", report);
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
        let path = std::env::temp_dir().join(format!(
            "knowprolog_lessons_{}.json",
            std::process::id()
        ));
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
    fn test_learn_general_rule_enables_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(taught.contains("通用规则"), "应识别为含变量的通用规则：{}", taught);
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
        assert!(proof.contains("中文教学"), "证明树应标出规则来源：{}", proof);
    }

    #[test]
    fn test_learn_ground_rule_joins_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(taught.contains("具体规则"), "应识别为具体规则：{}", taught);
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "教过的具体规则应参与后向链：{}", proof);
    }

    #[test]
    fn test_learn_rule_report_rejects_non_rules_gracefully() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.learn_rule_report("张三是李四的师傅");
        assert!(report.contains("没能把这句话解析成规则"), "实际：{}", report);
        let empty = reasoner.learn_rule_report("   ");
        assert!(empty.contains("请给出中文规则"), "实际：{}", empty);
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
        assert!(rules.contains("钱七"), "会话事实清单应包含新事实：{}", rules);

        let forgotten = reasoner.retract_nl_fact("钱七是孙八的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤回后不应再得证：{}", proof);
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
        let path = std::env::temp_dir().join(format!(
            "knowprolog_clauses_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "恢复后的会话规则应参与后向链：{}", proof);
        assert!(
            restored.dialogue_summary().contains("会话规则 1 条"),
            "对话记忆应显示会话规则：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_problog_multi_sentence_teaching_reports_errors() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.teach_report("如果李四的师傅是王五，那么李四的师祖是赵六。这句话是谁说的？");
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
        assert_eq!(subjects.first().map(|hit| hit.symbol.as_str()), Some("李四"));

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

    /// VID（版本化增量派生）：重复观察同一批事实必须版本不动、派生零重建。
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

        // 新增一条事实：版本推进，派生台账重新挂牌
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
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 2, "版本变化后旧记忆必须失效");
    }
}

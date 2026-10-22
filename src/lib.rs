use oxigraph::model::*;
use oxigraph::sparql::{Query, QueryResults};
use oxigraph::store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use toml;
use uuid::Uuid;

// 使模块公开，以便 main.rs 和 tests 可以访问
pub mod abduction;
pub mod advanced_nlu;
pub mod artifact_audit;
pub mod association_miner;
pub mod belief_revision;
pub mod binding_frames;
pub mod bindings;
pub mod cache;
pub mod canonical;
pub mod causal_reasoning;
pub mod causal_tokenization;
pub mod chinese_reasoning_nlu;
pub mod chronicle;
pub mod completion_episode;
pub mod config_contract;
pub mod context_image;
pub mod contextual_rag;
pub mod contextual_reference;
pub mod conversation_theory;
pub mod conversational_graph;
pub mod default_logic;
pub mod dense_retrieval;
pub mod derivation;
pub mod dialogue_eval;
pub mod dialogue_frame;
pub mod dialogue_fuzz;
pub mod dialogue_manager;
pub mod discourse;
pub mod embedding_eval;
pub mod entity_linker;
pub mod entity_registry;
pub mod entity_relation_extractor;
pub mod epistemic;
pub mod eval_recovery;
pub mod external_knowledge;
pub mod external_ledger;
pub mod formal_semantics;
pub mod function_words;
pub mod grammar_extractor;
pub mod graph_ledger;
pub mod graph_question;
pub mod graph_walk_embedding;
pub mod history_trust;
pub mod holographic_memory;
pub mod identity;
pub mod identity_dialogue;
pub mod identity_eval;
pub mod import_journal;
pub mod knowledge_fusion;
pub mod knowledge_layers;
pub mod law_book;
pub mod lexical_trace;
pub mod link_eval;
pub mod link_prediction;
pub mod local_semantic_adapter;
pub mod logic_program;
pub mod markov_logic;
pub mod neuro_symbolic_contrast;
pub mod nlp;
pub mod nlu;
pub mod ontology_alignment;
pub mod ontology_reasoning;
pub mod ownthink_api;
pub mod path_memory;
pub mod pattern_closure;
pub mod pattern_ir;
pub mod perf_ledger;
pub mod planning;
pub mod plugin;
pub mod predicate_registry;
pub mod problog;
pub mod prolog_engine;
pub mod prompt_spec;
pub mod proof_journal;
pub mod quality;
pub mod query_board;
pub mod query_planner;
pub mod query_rewrite;
pub mod query_scenario;
pub mod quantity;
pub mod question_analyzer;
pub mod reading;
pub mod recovery;
pub mod relation_acquisition;
pub mod relation_frame;
pub mod release;
pub mod resolution_engine;
pub mod revision_guard;
pub mod semantic_embedding;
pub mod semantic_engine;
pub mod semantic_entailment;
pub mod serve_api;
pub mod shape_dialogue;
pub mod shape_eval;
pub mod shape_gate;
pub mod similarity_calculator;
pub mod startup_ledger;
pub mod symbolic_generator;
pub mod symbolic_reasoner;
pub mod temporal;
pub mod trace_graph;
pub mod turn_runtime;
pub mod type_system;
pub mod view_render;
pub mod weak_construction;

#[cfg(test)]
mod dialogue_conformance;

use nlp::{normalize_words, pattern_from_strings, preprocess, PrologRule, TaggedWord};
use prolog_engine::PrologFact;

// RDF命名空间（与 identity::DEFAULT_BASE_NS 单一正本）
const BASE_NS: &str = crate::identity::DEFAULT_BASE_NS;
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const NAME_PREDICATE_IRI: &str = "http://example.org/kg/name";

/// 转义 SPARQL 字符串字面量中的特殊字符（引号/反斜杠/换行等），
/// 避免名称里的特殊字符破坏查询语法或造成查询注入
fn sparql_literal(text: &str) -> std::borrow::Cow<'_, str> {
    if !text
        .as_bytes()
        .iter()
        .any(|b| matches!(b, b'\\' | b'"' | b'\n' | b'\r' | b'\t'))
    {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut escaped = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    std::borrow::Cow::Owned(escaped)
}

/// 实体在图中已声明的类型（rdf:type 的本地名集合，用于定义域/值域校验）。
fn entity_type_labels(
    store: &Store,
    name: &str,
) -> Result<BTreeSet<String>, Box<dyn std::error::Error + Send + Sync>> {
    let subject = NamedNode::new(crate::identity::entity_uri(BASE_NS, name))?;
    let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
    let mut labels = BTreeSet::new();
    for quad in store.quads_for_pattern(
        Some(subject.as_ref().into()),
        Some(rdf_type.as_ref()),
        None,
        None,
    ) {
        if let Term::NamedNode(type_node) = quad?.object {
            if let Some(label) = type_node.as_str().strip_prefix(BASE_NS) {
                labels.insert(label.to_string());
            }
        }
    }
    Ok(labels)
}

#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub success: bool,
    pub data: Vec<HashMap<String, Value>>,
}

impl QueryResult {
    /// 表格化渲染任意 SPARQL 结果（列按变量名字典序，行保持查询顺序）。
    ///
    /// 列宽按字符（而非字节）计算，中文实体名也能对齐；空结果给固定措辞，
    /// 与 `ASK false`（一行 `boolean = false`）在输出上可区分。
    pub fn render_text(&self) -> String {
        if self.data.is_empty() {
            return "（查询成功，无结果）".to_string();
        }
        let mut columns: Vec<String> = Vec::new();
        for row in &self.data {
            for key in row.keys() {
                if !columns.iter().any(|column| column == key) {
                    columns.push(key.clone());
                }
            }
        }
        columns.sort();
        if columns.is_empty() {
            return "（查询成功，无变量绑定）".to_string();
        }
        let cells: Vec<Vec<String>> = self
            .data
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|column| match row.get(column) {
                        Some(Value::String(text)) => text.clone(),
                        Some(value) => value.to_string(),
                        None => String::new(),
                    })
                    .collect()
            })
            .collect();
        let mut widths: Vec<usize> = columns
            .iter()
            .map(|column| column.chars().count())
            .collect();
        for row in &cells {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(cell.chars().count());
            }
        }
        let mut lines = Vec::with_capacity(cells.len() + 2);
        lines.push(render_table_row(&columns, &widths));
        lines.push(
            widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("-+-"),
        );
        for row in &cells {
            lines.push(render_table_row(row, &widths));
        }
        lines.join("\n")
    }
}

/// 单行表格：单元格左对齐、按字符宽度补空格。
fn render_table_row(cells: &[String], widths: &[usize]) -> String {
    cells
        .iter()
        .zip(widths)
        .map(|(cell, width)| {
            let padding = width.saturating_sub(cell.chars().count());
            format!("{}{}", cell, " ".repeat(padding))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// RDF 项 → JSON：IRI/空白节点用字符串；带类型的字面量按 XSD 类型映射为
/// 数字/布尔（其它保持字符串），`SELECT` 结果可直接被下游消费。
fn term_to_json(term: &Term) -> Value {
    match term {
        Term::NamedNode(node) => Value::String(node.as_str().to_string()),
        Term::BlankNode(node) => Value::String(format!("_:{}", node.as_str())),
        Term::Triple(triple) => Value::String(format!("<<{}>>", triple)),
        Term::Literal(literal) => {
            let text = literal.value();
            match literal.datatype().as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long"
                | "http://www.w3.org/2001/XMLSchema#short"
                | "http://www.w3.org/2001/XMLSchema#byte"
                | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                | "http://www.w3.org/2001/XMLSchema#unsignedLong" => text
                    .parse::<i64>()
                    .map(Value::from)
                    .unwrap_or_else(|_| Value::String(text.to_string())),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => text
                    .parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::String(text.to_string())),
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    Value::Bool(text == "true" || text == "1")
                }
                _ => Value::String(text.to_string()),
            }
        }
    }
}

/// JSON-LD 导入统计：节点（实体）数、指向命名节点的关系边数、
/// 本次实际新增的三元组数（与库中已有三元组重复者不计入）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct JsonldImportStats {
    pub entities: usize,
    pub relations: usize,
    pub triples: usize,
}

impl JsonldImportStats {
    /// 单行中文摘要（CLI 与交互模式共用同一措辞）。
    pub fn summary(&self) -> String {
        format!(
            "解析了 {} 个实体和 {} 个关系，新增 {} 条三元组。",
            self.entities, self.relations, self.triples
        )
    }
}

// 所有配置结构体都需要是 pub，以便在测试中使用
#[derive(Debug, Deserialize)]
pub struct SystemConfig {
    pub system: SystemSettings,
    pub api: Option<ApiConfig>,
    pub nlp: NlpConfig,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    pub entity_types: Vec<EntityTypeConfig>,
    pub extraction_rules: Vec<ExtractionRule>,
    pub prolog_rules: Vec<PrologRuleConfig>,
    pub messages: Messages,
    pub synonyms: Option<SynonymsConfig>,
    /// 谓词画像声明（可选）：导入准入、别名/逆向归一、功能谓词冲突裁决的共同依据
    #[serde(default)]
    pub predicate_profiles: Vec<predicate_registry::PredicateProfileConfig>,
    /// 实体身份声明（可选）：规范名 + 别名 + 类型，声明即已确认（显式承诺）
    #[serde(default)]
    pub entity_profiles: Vec<entity_registry::EntityProfileConfig>,
    /// 类型体系声明（可选）：别名/子类/互斥 + 模式闸门的闭包档
    #[serde(default)]
    pub type_system: type_system::TypeSystemConfig,
    /// 声明式插件（符号专家）目录、默认启用与逐包模式
    #[serde(default)]
    pub plugins: plugin::PluginSettings,
    /// 分层追踪图：多轮决策的显式工作记忆（问据栈/指代轨迹/更正失效）
    #[serde(default)]
    pub trace_graph: trace_graph::TraceGraphConfig,
    /// 配置编译的绑定帧：模式 -> 槽位 -> 目标 -> 措辞（默认关闭）。
    #[serde(default)]
    pub dialogue_frames: binding_frames::DialogueFramesConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReasoningConfig {
    #[serde(default = "default_enable_link_prediction")]
    pub enable_link_prediction: bool,
    #[serde(default = "default_link_prediction_min_confidence")]
    pub link_prediction_min_confidence: f64,
    #[serde(default = "default_link_prediction_min_support")]
    pub link_prediction_min_support: usize,
    #[serde(default = "default_ontology_alignment_threshold")]
    pub ontology_alignment_threshold: f64,
    #[serde(default = "default_max_predictions")]
    pub max_predictions: usize,
    #[serde(default = "default_enable_semantic_embedding")]
    pub enable_semantic_embedding: bool,
    #[serde(default = "default_embedding_similarity_floor")]
    pub embedding_similarity_floor: f64,
    #[serde(default)]
    pub dialogue_state_path: Option<String>,
    #[serde(default = "default_enable_generation")]
    pub enable_generation: bool,
    #[serde(default = "default_max_profile_sentences")]
    pub max_profile_sentences: usize,
    #[serde(default = "default_completion_min_score")]
    pub completion_min_score: f64,
    #[serde(default = "default_enable_entailment")]
    pub enable_entailment: bool,
    #[serde(default = "default_entailment_min_confidence")]
    pub entailment_min_confidence: f64,
    #[serde(default = "default_enable_resolution")]
    pub enable_resolution: bool,
    #[serde(default = "default_resolution_max_depth")]
    pub resolution_max_depth: usize,
    #[serde(default = "default_resolution_max_solutions")]
    pub resolution_max_solutions: usize,
    #[serde(default = "default_enable_association_rules")]
    pub enable_association_rules: bool,
    #[serde(default = "default_association_min_support")]
    pub association_min_support: f64,
    #[serde(default = "default_association_min_confidence")]
    pub association_min_confidence: f64,
    #[serde(default = "default_association_min_lift")]
    pub association_min_lift: f64,
    #[serde(default = "default_enable_abduction")]
    pub enable_abduction: bool,
    #[serde(default = "default_abduction_max_hypotheses")]
    pub abduction_max_hypotheses: usize,
    #[serde(default = "default_abduction_max_scenario_size")]
    pub abduction_max_scenario_size: usize,
    #[serde(default = "default_enable_clarification")]
    pub enable_clarification: bool,
    #[serde(default = "default_enable_graph_walk")]
    pub enable_graph_walk: bool,
    #[serde(default = "default_graph_walk_damping")]
    pub graph_walk_damping: f64,
    #[serde(default = "default_graph_walk_iterations")]
    pub graph_walk_iterations: usize,
    #[serde(default = "default_enable_soft_logic")]
    pub enable_soft_logic: bool,
    #[serde(default = "default_soft_logic_max_iterations")]
    pub soft_logic_max_iterations: usize,
    #[serde(default = "default_soft_logic_min_probability")]
    pub soft_logic_min_probability: f64,
    #[serde(default = "default_enable_causal_reasoning")]
    pub enable_causal_reasoning: bool,
    #[serde(default = "default_causal_max_depth")]
    pub causal_max_depth: usize,
    #[serde(default = "default_enable_gradient_attribution")]
    pub enable_gradient_attribution: bool,
    #[serde(default = "default_enable_causal_tokenization")]
    pub enable_causal_tokenization: bool,
    #[serde(default = "default_enable_problog")]
    pub enable_problog: bool,
    #[serde(default = "default_problog_max_depth")]
    pub problog_max_depth: usize,
    #[serde(default = "default_problog_max_proofs")]
    pub problog_max_proofs: usize,
    #[serde(default = "default_enable_dialogue_manager")]
    pub enable_dialogue_manager: bool,
    #[serde(default = "default_enable_discourse")]
    pub enable_discourse: bool,
    #[serde(default = "default_discourse_max_sentences")]
    pub discourse_max_sentences: usize,
    #[serde(default = "default_discourse_max_rounds")]
    pub discourse_max_rounds: usize,
    #[serde(default = "default_enable_holographic_memory")]
    pub enable_holographic_memory: bool,
    #[serde(default = "default_holographic_dimension")]
    pub holographic_dimension: usize,
    #[serde(default = "default_holographic_similarity_floor")]
    pub holographic_similarity_floor: f64,
    #[serde(default = "default_holographic_max_hits")]
    pub holographic_max_hits: usize,
    /// 对话级证据装配（焦点帧 × 全息召回 × 证据束生成）；默认关闭。
    #[serde(default)]
    pub enable_contextual_rag: bool,
    /// 严格谓词准入：未登记谓词不自动准入、其事实不落库（默认自动准入 + 待确认）
    #[serde(default)]
    pub require_predicate_admission: bool,
}

fn default_enable_holographic_memory() -> bool {
    true
}

fn default_holographic_dimension() -> usize {
    holographic_memory::DEFAULT_DIMENSION
}

fn default_holographic_similarity_floor() -> f64 {
    0.01
}

fn default_holographic_max_hits() -> usize {
    6
}

fn default_enable_problog() -> bool {
    true
}

fn default_problog_max_depth() -> usize {
    6
}

fn default_problog_max_proofs() -> usize {
    12
}

fn default_enable_discourse() -> bool {
    true
}

fn default_enable_dialogue_manager() -> bool {
    true
}

fn default_discourse_max_sentences() -> usize {
    6
}

fn default_discourse_max_rounds() -> usize {
    4
}

fn default_enable_causal_reasoning() -> bool {
    true
}

fn default_causal_max_depth() -> usize {
    4
}

fn default_enable_gradient_attribution() -> bool {
    true
}

fn default_enable_causal_tokenization() -> bool {
    true
}

fn default_enable_resolution() -> bool {
    true
}

fn default_resolution_max_depth() -> usize {
    6
}

fn default_resolution_max_solutions() -> usize {
    12
}

fn default_enable_association_rules() -> bool {
    true
}

fn default_association_min_support() -> f64 {
    0.2
}

fn default_association_min_confidence() -> f64 {
    0.5
}

fn default_association_min_lift() -> f64 {
    1.05
}

fn default_enable_abduction() -> bool {
    true
}

fn default_abduction_max_hypotheses() -> usize {
    5
}

fn default_abduction_max_scenario_size() -> usize {
    2
}

fn default_enable_clarification() -> bool {
    true
}

fn default_enable_graph_walk() -> bool {
    true
}

fn default_graph_walk_damping() -> f64 {
    0.85
}

fn default_graph_walk_iterations() -> usize {
    12
}

fn default_enable_soft_logic() -> bool {
    true
}

fn default_soft_logic_max_iterations() -> usize {
    4
}

fn default_soft_logic_min_probability() -> f64 {
    0.5
}

fn default_enable_entailment() -> bool {
    true
}

fn default_entailment_min_confidence() -> f64 {
    0.5
}

fn default_enable_generation() -> bool {
    true
}

fn default_max_profile_sentences() -> usize {
    6
}

fn default_completion_min_score() -> f64 {
    0.3
}

fn default_enable_semantic_embedding() -> bool {
    true
}

fn default_embedding_similarity_floor() -> f64 {
    0.6
}

fn default_enable_link_prediction() -> bool {
    true
}

fn default_link_prediction_min_confidence() -> f64 {
    0.6
}

fn default_link_prediction_min_support() -> usize {
    2
}

fn default_ontology_alignment_threshold() -> f64 {
    0.72
}

fn default_max_predictions() -> usize {
    5
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enable_link_prediction: default_enable_link_prediction(),
            link_prediction_min_confidence: default_link_prediction_min_confidence(),
            link_prediction_min_support: default_link_prediction_min_support(),
            ontology_alignment_threshold: default_ontology_alignment_threshold(),
            max_predictions: default_max_predictions(),
            enable_semantic_embedding: default_enable_semantic_embedding(),
            embedding_similarity_floor: default_embedding_similarity_floor(),
            dialogue_state_path: None,
            enable_generation: default_enable_generation(),
            max_profile_sentences: default_max_profile_sentences(),
            completion_min_score: default_completion_min_score(),
            enable_entailment: default_enable_entailment(),
            entailment_min_confidence: default_entailment_min_confidence(),
            enable_resolution: default_enable_resolution(),
            resolution_max_depth: default_resolution_max_depth(),
            resolution_max_solutions: default_resolution_max_solutions(),
            enable_association_rules: default_enable_association_rules(),
            association_min_support: default_association_min_support(),
            association_min_confidence: default_association_min_confidence(),
            association_min_lift: default_association_min_lift(),
            enable_abduction: default_enable_abduction(),
            abduction_max_hypotheses: default_abduction_max_hypotheses(),
            abduction_max_scenario_size: default_abduction_max_scenario_size(),
            enable_clarification: default_enable_clarification(),
            enable_graph_walk: default_enable_graph_walk(),
            graph_walk_damping: default_graph_walk_damping(),
            graph_walk_iterations: default_graph_walk_iterations(),
            enable_soft_logic: default_enable_soft_logic(),
            soft_logic_max_iterations: default_soft_logic_max_iterations(),
            soft_logic_min_probability: default_soft_logic_min_probability(),
            enable_causal_reasoning: default_enable_causal_reasoning(),
            causal_max_depth: default_causal_max_depth(),
            enable_gradient_attribution: default_enable_gradient_attribution(),
            enable_causal_tokenization: default_enable_causal_tokenization(),
            enable_problog: default_enable_problog(),
            problog_max_depth: default_problog_max_depth(),
            problog_max_proofs: default_problog_max_proofs(),
            enable_dialogue_manager: default_enable_dialogue_manager(),
            enable_discourse: default_enable_discourse(),
            discourse_max_sentences: default_discourse_max_sentences(),
            discourse_max_rounds: default_discourse_max_rounds(),
            enable_holographic_memory: default_enable_holographic_memory(),
            holographic_dimension: default_holographic_dimension(),
            holographic_similarity_floor: default_holographic_similarity_floor(),
            holographic_max_hits: default_holographic_max_hits(),
            enable_contextual_rag: false,
            require_predicate_admission: false,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiConfig {
    pub ownthink_base_url: String,
    pub ownthink_timeout_seconds: u64,
    pub enable_entity_disambiguation: bool,
    pub max_ambiguous_entities: usize,
    pub auto_select_best_entity: bool,
    /// 外链名册落盘路径（缺省为 `<store>/external_ledger.json`）
    #[serde(default)]
    pub external_ledger_path: Option<String>,
    /// 外链名册容量（条目数上限，LRU 淘汰）
    #[serde(default = "default_external_ledger_capacity")]
    pub external_ledger_capacity: usize,
    /// 外链名册时限（逻辑轮次；0 = 不过期，刷新交给显式命令）
    #[serde(default)]
    pub external_ledger_ttl_ticks: u64,
}

fn default_external_ledger_capacity() -> usize {
    external_ledger::DEFAULT_CAPACITY
}

#[derive(Debug, Deserialize)]
pub struct SynonymsConfig {
    pub mappings: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct SystemSettings {
    pub oxigraph_endpoint: String,
    pub oxigraph_store_path: String,
    pub default_language: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NlpConfig {
    pub import_keywords: Vec<String>,
    pub pos_dict: HashMap<String, String>,
    pub name_patterns: NamePatterns,
    pub enable_nlu: Option<bool>,
    pub nlu_confidence_threshold: Option<f64>,
    pub enable_local_semantic: Option<bool>,
    pub disable_ownthink_api: Option<bool>,
    pub semantic_similarity_threshold: Option<f64>,
    pub max_entity_candidates: Option<usize>,
    pub enable_advanced_entity_extraction: Option<bool>,
    pub enable_relation_extraction: Option<bool>,
    pub context_window_size: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamePatterns {
    pub prefixes: Vec<String>,
    pub suffixes: Vec<String>,
    pub full_names: Vec<String>,
    pub profession_suffix: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct EntityTypeConfig {
    pub label: String,
    pub properties: Vec<PropertyConfig>,
    pub examples: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct PropertyConfig {
    pub name: String,
    pub data_type: String,
    pub required: bool,
    #[serde(default)]
    pub extraction_patterns: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ExtractionRule {
    pub name: String,
    pub entity_type: String,
    pub patterns: Vec<PatternConfig>,
    pub priority: i32,
}

#[derive(Debug, Deserialize)]
pub struct PatternConfig {
    pub pos_pattern: Vec<String>,
    pub keyword_patterns: Option<Vec<String>>,
    pub property_mappings: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct PrologRuleConfig {
    pub name: String,
    pub pattern: Vec<String>,
    pub query_type: String,
    pub priority: i32,
}

#[derive(Debug, Deserialize)]
pub struct Messages {
    pub startup: String,
    pub ready: String,
    pub query_prompt: String,
    pub import_prompt: String,
    pub quit_prompt: String,
    pub importing: String,
    pub no_knowledge_extracted: String,
    pub cannot_understand: String,
    pub process_error: String,
}

#[derive(Debug)]
pub enum InputMode {
    Query,
    KnowledgeImport,
}

#[derive(Debug)]
pub enum KnowledgeItem {
    Entity {
        label: String,
        properties: HashMap<String, String>,
    },
    Relationship {
        from_label: String,
        from_id: String,
        to_label: String,
        to_id: String,
        rel_type: String,
        properties: HashMap<String, String>,
    },
}

pub struct KnowledgeExtractor {
    config: Arc<SystemConfig>,
}

impl KnowledgeExtractor {
    pub fn new(config: Arc<SystemConfig>) -> Self {
        Self { config }
    }

    pub fn extract(&self, text: &str, tagged_words: &[TaggedWord]) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();
        let tagged_words = nlp::merge_name_fragments(tagged_words);
        let tagged_words = tagged_words.as_slice();

        for rule in &self.config.extraction_rules {
            items.extend(self.apply_extraction_rule(text, tagged_words, rule));
        }

        if text.contains("，") || text.contains("；") || text.contains("。") {
            let parts: Vec<&str> = text.split(&['，', '；', '。'][..]).collect();
            for part in parts {
                if part.trim().is_empty() {
                    continue;
                }
                let part_tagged = preprocess(part, &self.config.nlp);
                let normalized_part_tagged = if let Some(synonyms_config) = &self.config.synonyms {
                    normalize_words(part_tagged, &synonyms_config.mappings)
                } else {
                    part_tagged
                };
                items.extend(self.extract(part, &normalized_part_tagged));
            }
        }

        items
    }

    fn apply_extraction_rule(
        &self,
        text: &str,
        tagged_words: &[TaggedWord],
        rule: &ExtractionRule,
    ) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();

        for pattern in &rule.patterns {
            if let Some(item) = self.match_pattern(text, tagged_words, pattern, &rule.entity_type) {
                items.push(item);
            }
        }

        items
    }

    fn match_pattern(
        &self,
        text: &str,
        tagged_words: &[TaggedWord],
        pattern: &PatternConfig,
        entity_type: &str,
    ) -> Option<KnowledgeItem> {
        if !self.check_pos_pattern(tagged_words, &pattern.pos_pattern) {
            return None;
        }

        // 检查关键词模式
        if let Some(keyword_patterns) = &pattern.keyword_patterns {
            if !self.check_keyword_patterns(text, keyword_patterns) {
                return None;
            }
        }

        let properties = self.extract_properties(tagged_words, &pattern.property_mappings);

        if properties.is_empty() {
            return None;
        }

        Some(KnowledgeItem::Entity {
            label: entity_type.to_string(),
            properties,
        })
    }

    fn check_keyword_patterns(&self, text: &str, keyword_patterns: &[String]) -> bool {
        for keyword in keyword_patterns {
            if text.contains(keyword) {
                return true;
            }
        }
        false
    }

    fn check_pos_pattern(&self, tagged_words: &[TaggedWord], pos_pattern: &[String]) -> bool {
        if pos_pattern.len() > tagged_words.len() {
            return false;
        }

        for i in 0..=(tagged_words.len() - pos_pattern.len()) {
            let mut matched = true;
            for (j, expected_pos) in pos_pattern.iter().enumerate() {
                if expected_pos != "*" && tagged_words[i + j].pos != *expected_pos {
                    matched = false;
                    break;
                }
            }
            if matched {
                return true;
            }
        }

        false
    }

    fn extract_properties(
        &self,
        tagged_words: &[TaggedWord],
        mappings: &HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut properties = HashMap::new();

        for (prop_name, extraction_rule) in mappings {
            if let Some(value) = self.extract_property_value(tagged_words, extraction_rule) {
                properties.insert(prop_name.clone(), value);
            }
        }

        properties
    }

    fn extract_property_value(&self, tagged_words: &[TaggedWord], rule: &str) -> Option<String> {
        if rule.starts_with("pos:") {
            let pos_rule = rule.strip_prefix("pos:")?;

            // 处理索引语法，如 pos:nr[0]
            if pos_rule.contains('[') && pos_rule.contains(']') {
                let parts: Vec<&str> = pos_rule.split(['[', ']']).collect();
                if parts.len() >= 2 {
                    let pos = parts[0];
                    // 索引语法畸形时视为无效规则
                    let index: usize = parts[1].trim().parse().ok()?;

                    let mut matching_words = Vec::new();
                    for word in tagged_words {
                        if word.pos == pos {
                            matching_words.push(word.text.clone());
                        }
                    }

                    if index < matching_words.len() {
                        return Some(matching_words[index].clone());
                    }
                }
            }

            // 人名提取：相邻的 nr + 单字 unknown 已在预处理阶段合并
            // （见 nlp::merge_name_fragments），这里只取首个 nr 词，
            // 避免把「姚明身高226」这类句式里紧跟的未知属性词并进人名。
            if pos_rule == "nr" {
                for word in tagged_words {
                    if word.pos == "nr" {
                        return Some(word.text.clone());
                    }
                }
            }

            // 处理通配符 *
            if pos_rule == "*" {
                // 对于通配符，返回所有不是功能词的词的组合
                let content_words: Vec<String> = tagged_words
                    .iter()
                    .filter(|word| {
                        !matches!(word.pos.as_str(), "u" | "c" | "p")
                            && !matches!(word.text.as_str(), "的" | "是" | "有")
                    })
                    .map(|word| word.text.clone())
                    .collect();
                if !content_words.is_empty() {
                    return Some(content_words.join(""));
                }
            }

            // 普通词性查找
            for word in tagged_words {
                if word.pos == pos_rule {
                    return Some(word.text.clone());
                }
            }
        } else if rule.starts_with("text:") {
            let text = rule.strip_prefix("text:")?;
            for word in tagged_words {
                if word.text == text {
                    return Some(word.text.clone());
                }
            }
        }

        None
    }
}

// Oxigraph操作管理器
pub struct OxigraphManager {
    store: Arc<Mutex<Store>>,
    // 每次写操作自增，用于判定缓存的整图快照是否仍然有效
    revision: AtomicU64,
    snapshot_cache: parking_lot::Mutex<Option<(u64, Arc<Vec<PrologFact>>)>>,
    // 相关事实查询缓存：同一知识版本内按查询键多槽缓存（LRU），
    // 交替询问不同实体互不挤占；写后整体失效
    relevant_cache: parking_lot::Mutex<RevisionLru<Arc<Vec<PrologFact>>>>,
    // 直答缓存：同一知识版本内按查询键多槽缓存（LRU，含"确无答案"的负缓存）
    direct_answer_cache: parking_lot::Mutex<RevisionLru<Option<String>>>,
    // 操作计数台账（可与会话推理器共享同一实例）
    perf: Arc<perf_ledger::PerfLedger>,
    // 谓词画像注册表：导入准入、别名/逆向归一、功能谓词冲突裁决
    predicate_registry: parking_lot::RwLock<predicate_registry::PredicateRegistry>,
    // 实体身份注册表：提名（待确认）/ 确认 / 拒绝 / 忘记 / 合并的显式台账
    entity_registry: parking_lot::RwLock<entity_registry::EntityRegistry>,
    // 类型体系：别名/子类/互斥声明的类型格，模式闸门与质量护照共同消费
    type_system: parking_lot::RwLock<type_system::TypeSystem>,
}

/// 每次写操作缓存容量上限（条目数）。对话中交替询问少量实体，
/// 32 槽足以消除往返抖动，同时让内存占用可计算。
const QUERY_CACHE_CAPACITY: usize = 32;

use crate::cache::RevisionLru;

impl OxigraphManager {
    pub fn new(store_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_perf_ledger(store_path, Arc::new(perf_ledger::PerfLedger::new()))
    }

    /// 与 `new` 相同，但可注入共享的性能台账（会话级操作计数）。
    pub fn with_perf_ledger(
        store_path: &str,
        perf: Arc<perf_ledger::PerfLedger>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 展开波浪号路径
        let expanded_path = if store_path.starts_with("~/") {
            if let Some(home_dir) = std::env::var_os("HOME") {
                let home_path = std::path::Path::new(&home_dir);
                home_path
                    .join(&store_path[2..])
                    .to_string_lossy()
                    .to_string()
            } else {
                store_path.to_string()
            }
        } else {
            store_path.to_string()
        };

        // 如果目录不存在则创建
        if let Some(parent) = std::path::Path::new(&expanded_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let store = Store::open(&expanded_path)?;
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            revision: AtomicU64::new(0),
            snapshot_cache: parking_lot::Mutex::new(None),
            relevant_cache: parking_lot::Mutex::new(RevisionLru::new(QUERY_CACHE_CAPACITY)),
            direct_answer_cache: parking_lot::Mutex::new(RevisionLru::new(QUERY_CACHE_CAPACITY)),
            perf,
            predicate_registry: parking_lot::RwLock::new(
                predicate_registry::PredicateRegistry::new(),
            ),
            entity_registry: parking_lot::RwLock::new(entity_registry::EntityRegistry::new()),
            type_system: parking_lot::RwLock::new(type_system::TypeSystem::new()),
        })
    }

    /// 共享的性能台账（与 `SymbolicReasoner` 可为同一实例）。
    pub fn perf_ledger(&self) -> Arc<perf_ledger::PerfLedger> {
        Arc::clone(&self.perf)
    }

    /// 标记知识版本变化：调用方必须持有存储锁，且在本次实际写入完成之后
    /// （version-after-write）。这样并发读者绝不会把过期数据缓存到新版本号上，
    /// 且重复导入不推进版本。
    fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.perf.bump("revision_bump");
    }

    pub async fn import_jsonld(
        &self,
        jsonld_content: &str,
    ) -> Result<JsonldImportStats, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        // 持锁后、写前失效整图快照缓存：JSON-LD 导入可能分批写入，
        // 版本号先行可保证中途失败也不会让读者把半成品缓存成完整快照
        self.bump_revision();

        // 解析 JSON-LD 内容
        let jsonld_value: Value = serde_json::from_str(jsonld_content)?;

        // 直接处理 JSON 结构；三元组数按库长度差统计（重复导入计 0）
        let before = store.len()?;
        let mut stats = JsonldImportStats::default();
        self.process_jsonld_simple(&store, &jsonld_value, &mut stats)
            .await?;
        let after = store.len()?;
        stats.triples = after.saturating_sub(before);

        Ok(stats)
    }

    async fn process_jsonld_simple(
        &self,
        store: &Store,
        jsonld: &Value,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 处理直接的 JSON-LD 结构
        if let Some(array) = jsonld.as_array() {
            for item in array {
                if let Some(obj) = item.as_object() {
                    // 处理每个对象
                    let context = Self::extract_jsonld_context(obj);
                    self.process_jsonld_object(store, obj, &context, stats)?;
                }
            }
        } else if let Some(obj) = jsonld.as_object() {
            // 顶层 @context 中的字符串映射用于展开类型与属性名
            let context = Self::extract_jsonld_context(obj);
            // 顶层为包含 @graph 的 JSON-LD 文档时，递归处理图中的每个节点
            if let Some(graph) = obj.get("@graph") {
                if let Some(graph_array) = graph.as_array() {
                    for item in graph_array {
                        if let Some(inner) = item.as_object() {
                            self.process_jsonld_object(store, inner, &context, stats)?;
                        }
                    }
                } else if let Some(inner) = graph.as_object() {
                    self.process_jsonld_object(store, inner, &context, stats)?;
                }
            } else {
                // 单个对象
                self.process_jsonld_object(store, obj, &context, stats)?;
            }
        }
        Ok(())
    }

    /// 提取 @context 中的映射（term -> IRI / @base / @vocab），供类型与属性名展开使用。
    ///
    /// 支持三种常见写法：
    /// - 字符串映射：`"name": "http://example.org/kg/name"`
    /// - 上下文数组：`"@context": [{...}, {...}]`（后者覆盖前者）
    /// - 对象映射：`"ex": { "@id": "http://example.org/kg/" }`
    fn extract_jsonld_context(obj: &serde_json::Map<String, Value>) -> HashMap<String, String> {
        let mut context = HashMap::new();
        let Some(ctx) = obj.get("@context") else {
            return context;
        };
        let apply = |ctx: &serde_json::Map<String, Value>,
                     context: &mut HashMap<String, String>| {
            for (term, value) in ctx {
                if let Some(iri) = value.as_str() {
                    context.insert(term.clone(), iri.to_string());
                } else if let Some(iri) = value
                    .as_object()
                    .and_then(|definition| definition.get("@id"))
                    .and_then(Value::as_str)
                {
                    context.insert(term.clone(), iri.to_string());
                }
            }
        };
        if let Some(ctx) = ctx.as_object() {
            apply(ctx, &mut context);
        } else if let Some(items) = ctx.as_array() {
            for item in items {
                if let Some(ctx) = item.as_object() {
                    apply(ctx, &mut context);
                }
            }
        }
        context
    }

    /// 按 @context 展开 JSON-LD 词项：已是 IRI 的保留，映射词项替换为对应 IRI，否则回退到 @vocab。
    ///
    /// 支持前缀映射（`ex:name`）与「词项 → 紧凑 IRI → 绝对 IRI」的链式展开；
    /// 设深度上限避免自引用上下文导致死循环。
    fn expand_jsonld_term(term: &str, context: &HashMap<String, String>) -> String {
        let mut current = term.to_string();
        for _ in 0..8 {
            if let Some((prefix, suffix)) = current.split_once(':') {
                if let Some(base) = context.get(prefix) {
                    return format!("{}{}", base, suffix);
                }
                return current;
            }
            let Some(iri) = context.get(&current) else {
                break;
            };
            if iri == &current {
                break;
            }
            current = iri.clone();
        }
        if !current.contains(':') {
            if let Some(vocab) = context.get("@vocab") {
                return format!("{}{}", vocab, current);
            }
        }
        current
    }

    /// 展开节点 `@id`：裸标识符按 `@base` 解析为绝对 IRI（JSON-LD 文档相对 IRI 语义）。
    fn expand_jsonld_id(id: &str, context: &HashMap<String, String>) -> String {
        if id.contains(':') || context.contains_key(id) {
            return Self::expand_jsonld_term(id, context);
        }
        if let Some(base) = context.get("@base") {
            return format!("{}{}", base, id);
        }
        id.to_string()
    }

    /// 把 JSON-LD 节点引用解析为命名节点或空白节点。
    ///
    /// `_:label` 是 JSON-LD 的空白节点标识符（不是 IRI）。同一标签在重复导入时
    /// 必须映射到同一空白节点，才能保持导入幂等；标签含非法字符时用稳定哈希兜底。
    fn jsonld_node(
        id: &str,
        context: &HashMap<String, String>,
    ) -> Result<NamedOrBlankNode, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(label) = id.strip_prefix("_:") {
            if let Ok(node) = BlankNode::new(label.to_string()) {
                return Ok(node.into());
            }
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in label.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            return Ok(BlankNode::new(format!("b{:016x}", hash))?.into());
        }
        Ok(NamedNode::new(Self::expand_jsonld_id(id, context))?.into())
    }

    fn process_jsonld_object(
        &self,
        store: &Store,
        obj: &serde_json::Map<String, Value>,
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 节点级 @context 是作用域上下文：只影响本节点及其嵌套值。
        let mut context = context.clone();
        if obj.contains_key("@context") {
            context.extend(Self::extract_jsonld_context(obj));
        }
        let context = &context;

        // 纯图容器（只有 @graph/@included 等关键字）本身不是实体：
        // 直接展开其中的节点，避免生成随机空白实体。
        let has_node_payload =
            obj.contains_key("@type") || obj.keys().any(|key| !key.starts_with("@"));
        if !has_node_payload {
            let graph = obj.get("@graph");
            let included = obj.get("@included");
            if graph.is_none() && included.is_none() {
                return Ok(());
            }
            if let Some(graph) = graph {
                self.process_jsonld_graph(store, graph, context, stats)?;
            }
            if let Some(included) = included {
                self.process_jsonld_graph(store, included, context, stats)?;
            }
            return Ok(());
        }

        // 获取主语 (subject)：@id 可以是 IRI，也可以是 `_:label` 空白节点。
        let subject: NamedOrBlankNode = if let Some(id) = obj.get("@id") {
            let id = id.as_str().unwrap_or("");
            if id.is_empty() {
                return Ok(());
            }
            Self::jsonld_node(id, context)?
        } else {
            NamedNode::new(crate::identity::entity_uri(
                BASE_NS,
                &Uuid::new_v4().to_string(),
            ))?
            .into()
        };
        stats.entities += 1;

        // 处理类型信息
        if let Some(types) = obj.get("@type") {
            if let Some(type_array) = types.as_array() {
                for type_val in type_array {
                    if let Some(type_uri) = type_val.as_str() {
                        let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
                        let object_type =
                            NamedNode::new(Self::expand_jsonld_term(type_uri, context))?;

                        store.insert(&Quad::new(
                            subject.clone(),
                            rdf_type,
                            object_type,
                            GraphName::DefaultGraph,
                        ))?;
                    }
                }
            } else if let Some(type_uri) = types.as_str() {
                let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
                let object_type = NamedNode::new(Self::expand_jsonld_term(type_uri, context))?;

                store.insert(&Quad::new(
                    subject.clone(),
                    rdf_type,
                    object_type,
                    GraphName::DefaultGraph,
                ))?;
            }
        }

        // 处理属性
        for (key, value) in obj {
            if key.starts_with("@") {
                continue; // 跳过 JSON-LD 关键字
            }

            let predicate = NamedNode::new(Self::expand_jsonld_term(key, context))?;

            if let Some(value_array) = value.as_array() {
                for val in value_array {
                    self.process_property_value(store, &subject, &predicate, val, context, stats)?;
                }
            } else {
                self.process_property_value(store, &subject, &predicate, value, context, stats)?;
            }
        }

        // @reverse：值里的节点是主语，当前节点是宾语（JSON-LD 反向属性）。
        if let Some(reverse) = obj.get("@reverse").and_then(Value::as_object) {
            for (key, value) in reverse {
                if key.starts_with("@") {
                    continue;
                }
                let predicate = NamedNode::new(Self::expand_jsonld_term(key, context))?;
                if let Some(value_array) = value.as_array() {
                    for val in value_array {
                        self.process_reverse_value(
                            store, &subject, &predicate, val, context, stats,
                        )?;
                    }
                } else {
                    self.process_reverse_value(
                        store, &subject, &predicate, value, context, stats,
                    )?;
                }
            }
        }

        // 带 @id/属性的节点也可以顺带承载命名图或 @included 节点；
        // 知识库只使用默认图，这里把其内容展开到默认图而不是丢弃。
        if let Some(graph) = obj.get("@graph") {
            self.process_jsonld_graph(store, graph, context, stats)?;
        }
        if let Some(included) = obj.get("@included") {
            self.process_jsonld_graph(store, included, context, stats)?;
        }

        Ok(())
    }

    /// 展开 `@graph` / `@included` 的值：接受单个节点或节点数组。
    fn process_jsonld_graph(
        &self,
        store: &Store,
        value: &Value,
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            Value::Array(items) => {
                for item in items {
                    if let Some(obj) = item.as_object() {
                        self.process_jsonld_object(store, obj, context, stats)?;
                    }
                }
            }
            Value::Object(obj) => self.process_jsonld_object(store, obj, context, stats)?,
            _ => {}
        }
        Ok(())
    }

    /// 处理 `@reverse` 中的节点引用：节点 → 谓词 → 当前节点。
    fn process_reverse_value(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        value: &Value,
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(obj) = value.as_object() else {
            return Ok(());
        };
        let Some(id) = obj.get("@id").and_then(Value::as_str) else {
            return Ok(());
        };
        if id.is_empty() {
            return Ok(());
        }
        let object = Self::jsonld_node(id, context)?;
        store.insert(&Quad::new(
            object.clone(),
            predicate.clone(),
            subject.clone(),
            GraphName::DefaultGraph,
        ))?;
        stats.relations += 1;
        // 带额外属性的反向节点本身也是节点（与正向嵌套对象一致）。
        if obj.keys().any(|key| key != "@id" && key != "@context") {
            self.process_jsonld_object(store, obj, context, stats)?;
        }
        Ok(())
    }

    fn process_property_value(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        value: &Value,
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            Value::String(s) => {
                let object = Literal::new_simple_literal(s);
                store.insert(&Quad::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                    GraphName::DefaultGraph,
                ))?;
            }
            Value::Number(n) => {
                let object = Literal::new_simple_literal(&n.to_string());
                store.insert(&Quad::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                    GraphName::DefaultGraph,
                ))?;
            }
            Value::Bool(b) => {
                let object = Literal::new_simple_literal(&b.to_string());
                store.insert(&Quad::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                    GraphName::DefaultGraph,
                ))?;
            }
            Value::Object(obj) => {
                // @set 只是数组容器：逐个展开，不产生额外节点。
                if let Some(set) = obj.get("@set") {
                    if let Some(items) = set.as_array() {
                        for item in items {
                            self.process_property_value(
                                store, subject, predicate, item, context, stats,
                            )?;
                        }
                    } else {
                        self.process_property_value(
                            store, subject, predicate, set, context, stats,
                        )?;
                    }
                    return Ok(());
                }
                // @list：编码为 RDF 集合（rdf:first / rdf:rest / rdf:nil）。
                if let Some(list) = obj.get("@list") {
                    let items: Vec<Value> = match list {
                        Value::Array(items) => items.clone(),
                        other => vec![other.clone()],
                    };
                    self.process_jsonld_list(
                        store, subject, predicate, &items, context, stats,
                    )?;
                    return Ok(());
                }
                // 处理嵌套对象
                if let Some(id) = obj.get("@id") {
                    if let Some(uri) = id.as_str() {
                        let object = Self::jsonld_node(uri, context)?;
                        store.insert(&Quad::new(
                            subject.clone(),
                            predicate.clone(),
                            object,
                            GraphName::DefaultGraph,
                        ))?;
                        // 对象属性指向另一实体（命名节点）时计一条关系
                        stats.relations += 1;
                        // 嵌套对象除 @id 外还带类型/属性时，它本身也是一个节点
                        // （JSON-LD 展开语义），继续落库而不是丢掉这些描述。
                        if obj.keys().any(|key| key != "@id" && key != "@context") {
                            self.process_jsonld_object(store, obj, context, stats)?;
                        }
                    }
                } else if let Some(val) = obj.get("@value") {
                    // 字符串值取其内容本身，其他类型使用 JSON 表示
                    let value_text = match val {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };

                    // 处理带数据类型的值
                    let literal = if let Some(datatype) = obj.get("@type") {
                        if let Some(datatype_uri) = datatype.as_str() {
                            let dt =
                                NamedNode::new(Self::expand_jsonld_term(datatype_uri, context))?;
                            Literal::new_typed_literal(&value_text, dt)
                        } else {
                            Literal::new_simple_literal(&value_text)
                        }
                    } else if let Some(lang) = obj.get("@language") {
                        if let Some(lang_tag) = lang.as_str() {
                            Literal::new_language_tagged_literal(&value_text, lang_tag)?
                        } else {
                            Literal::new_simple_literal(&value_text)
                        }
                    } else {
                        Literal::new_simple_literal(&value_text)
                    };

                    store.insert(&Quad::new(
                        subject.clone(),
                        predicate.clone(),
                        literal,
                        GraphName::DefaultGraph,
                    ))?;
                }
            }
            _ => {
                // 其他类型转换为字符串
                let object = Literal::new_simple_literal(&value.to_string());
                store.insert(&Quad::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                    GraphName::DefaultGraph,
                ))?;
            }
        }
        Ok(())
    }

    /// 把 `@list` 编码为 RDF 集合，并接到当前主语上。
    ///
    /// 单元格使用内容哈希命名：同一文档重复导入仍指向同一组空白节点，
    /// 不会因为每次生成随机空白节点而破坏幂等。
    fn process_jsonld_list(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        items: &[Value],
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rdf_nil = NamedNode::new(format!("{}nil", RDF_NS))?;
        if items.is_empty() {
            store.insert(&Quad::new(
                subject.clone(),
                predicate.clone(),
                rdf_nil,
                GraphName::DefaultGraph,
            ))?;
            return Ok(());
        }
        let rdf_first = NamedNode::new(format!("{}first", RDF_NS))?;
        let rdf_rest = NamedNode::new(format!("{}rest", RDF_NS))?;
        let cells: Vec<NamedOrBlankNode> = items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                let material = format!("{}|{}|{}|{}", subject, predicate.as_str(), index, item);
                for byte in material.as_bytes() {
                    hash ^= u64::from(*byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
                BlankNode::new(format!("l{:016x}", hash))
                    .expect("生成的集合单元格标签必须合法")
                    .into()
            })
            .collect();
        store.insert(&Quad::new(
            subject.clone(),
            predicate.clone(),
            cells[0].clone(),
            GraphName::DefaultGraph,
        ))?;
        for (index, cell) in cells.iter().enumerate() {
            let item = self.process_jsonld_list_item(store, &items[index], context, stats)?;
            store.insert(&Quad::new(
                cell.clone(),
                rdf_first.clone(),
                item,
                GraphName::DefaultGraph,
            ))?;
            let rest: NamedOrBlankNode = cells
                .get(index + 1)
                .cloned()
                .unwrap_or_else(|| rdf_nil.clone().into());
            store.insert(&Quad::new(
                cell.clone(),
                rdf_rest.clone(),
                rest,
                GraphName::DefaultGraph,
            ))?;
        }
        Ok(())
    }

    /// 集合项：节点引用保持节点身份，值对象按字面量落库。
    fn process_jsonld_list_item(
        &self,
        store: &Store,
        item: &Value,
        context: &HashMap<String, String>,
        stats: &mut JsonldImportStats,
    ) -> Result<Term, Box<dyn std::error::Error + Send + Sync>> {
        match item {
            Value::String(text) => Ok(Literal::new_simple_literal(text).into()),
            Value::Number(number) => Ok(Literal::new_simple_literal(&number.to_string()).into()),
            Value::Bool(value) => Ok(Literal::new_simple_literal(&value.to_string()).into()),
            Value::Object(obj) => {
                if let Some(id) = obj.get("@id").and_then(Value::as_str) {
                    if id.is_empty() {
                        return Ok(Literal::new_simple_literal("").into());
                    }
                    let node = Self::jsonld_node(id, context)?;
                    // 带描述的对象同时作为独立节点落库。
                    if obj.keys().any(|key| key != "@id" && key != "@context") {
                        self.process_jsonld_object(store, obj, context, stats)?;
                    }
                    return Ok(node.into());
                }
                if let Some(value) = obj.get("@value") {
                    let text = match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                    if let Some(datatype) = obj.get("@type").and_then(Value::as_str) {
                        let datatype =
                            NamedNode::new(Self::expand_jsonld_term(datatype, context))?;
                        return Ok(Literal::new_typed_literal(&text, datatype).into());
                    }
                    if let Some(language) = obj.get("@language").and_then(Value::as_str) {
                        return Ok(Literal::new_language_tagged_literal(&text, language)?.into());
                    }
                    return Ok(Literal::new_simple_literal(&text).into());
                }
                Ok(Literal::new_simple_literal(&item.to_string()).into())
            }
            other => Ok(Literal::new_simple_literal(&other.to_string()).into()),
        }
    }

    pub async fn import_jsonld_from_file(
        &self,
        file_path: &str,
    ) -> Result<JsonldImportStats, Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(file_path)?;
        self.import_jsonld(&content).await
    }

    /// 从文件扩展名识别 RDF 序列化格式（ttl / nt / nq / trig / rdf / xml）。
    ///
    /// JSON-LD 不在此列：请使用 `import_jsonld` / `import_jsonld_from_file`，
    /// 它们支持 `@context` 展开与更宽松的文档结构。
    pub fn detect_rdf_format(
        file_path: &str,
    ) -> Result<oxigraph::io::RdfFormat, Box<dyn std::error::Error + Send + Sync>> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        match extension.to_ascii_lowercase().as_str() {
            "json" | "jsonld" => Err(format!(
                "「{}」是 JSON-LD 文件：请改用 import-jsonld（或交互命令「导入jsonld文件：」）",
                file_path
            )
            .into()),
            _ => oxigraph::io::RdfFormat::from_extension(extension).ok_or_else(|| {
                format!(
                    "无法从扩展名「{}」识别 RDF 格式（支持 ttl / nt / nq / trig / rdf / xml）",
                    if extension.is_empty() {
                        "（无）"
                    } else {
                        extension
                    }
                )
                .into()
            }),
        }
    }

    /// 从 RDF 文件导入三元组（支持 Turtle / N-Triples / N-Quads / TriG / RDF/XML），
    /// 返回本次新增的三元组数（与库中已有三元组重复者不计入）。
    pub async fn import_rdf_file(
        &self,
        file_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let format = Self::detect_rdf_format(file_path)?;
        let file = fs::File::open(file_path)?;
        self.import_rdf_reader(format, file).await
    }

    /// 从任意 reader 导入 RDF 到默认图，返回新增三元组数。
    ///
    /// 与 JSON-LD 导入一致：持锁后、写前推进知识版本，保证并发读者不会把
    /// 半成品缓存成完整快照；解析或写入失败时版本号也已推进（保守失效）。
    pub async fn import_rdf_reader<R: std::io::Read>(
        &self,
        format: oxigraph::io::RdfFormat,
        reader: R,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let before = store.len()?;
        self.bump_revision();
        store.load_from_reader(format, reader)?;
        let after = store.len()?;
        Ok(after.saturating_sub(before))
    }

    /// 执行任意 SPARQL 1.1 查询（SELECT / ASK / CONSTRUCT / DESCRIBE）。
    ///
    /// 这是命令行 `sparql` 子命令与交互模式「SPARQL：…」的公共实现：
    /// - `SELECT`：每行解一条记录（变量名 → JSON 值，未绑定变量缺席）；
    /// - `ASK`：单行 `boolean`；
    /// - `CONSTRUCT` / `DESCRIBE`：每行一条三元组（`subject`/`predicate`/`object`）。
    ///
    /// 查询只读、不推进知识版本；语法/执行错误按错误返回，绝不静默为空结果。
    pub async fn query_sparql(
        &self,
        query_text: &str,
    ) -> Result<QueryResult, Box<dyn std::error::Error + Send + Sync>> {
        self.perf.bump("sparql_query");
        let query = Query::parse(query_text, None).map_err(
            |error| -> Box<dyn std::error::Error + Send + Sync> {
                // sparql 语法错误的多行期望 token 列表很长：只保留首行并限长，
                // 位置信息（行:列）已足够定位，终端不被刷屏。
                let text = error.to_string();
                let first_line = text.lines().next().unwrap_or("未知错误");
                let mut detail: String = first_line.chars().take(240).collect();
                if first_line.chars().count() > 240 {
                    detail.push('…');
                }
                format!("SPARQL 语法错误：{}", detail).into()
            },
        )?;
        let store = self.store.lock().await;
        let results =
            store
                .query(query)
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("SPARQL 执行失败：{}", error).into()
                })?;
        let mut data: Vec<HashMap<String, Value>> = Vec::new();
        match results {
            QueryResults::Solutions(solutions) => {
                for solution in solutions {
                    let solution =
                        solution.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("SPARQL 求解失败：{}", error).into()
                        })?;
                    let mut row = HashMap::new();
                    for (variable, term) in solution.iter() {
                        row.insert(variable.as_str().to_string(), term_to_json(term));
                    }
                    data.push(row);
                }
            }
            QueryResults::Boolean(value) => {
                data.push(HashMap::from([("boolean".to_string(), Value::Bool(value))]));
            }
            QueryResults::Graph(triples) => {
                for triple in triples {
                    let triple =
                        triple.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("SPARQL 图求解失败：{}", error).into()
                        })?;
                    data.push(HashMap::from([
                        (
                            "subject".to_string(),
                            term_to_json(&Term::from(triple.subject)),
                        ),
                        (
                            "predicate".to_string(),
                            term_to_json(&Term::from(triple.predicate)),
                        ),
                        ("object".to_string(), term_to_json(&triple.object)),
                    ]));
                }
            }
        }
        Ok(QueryResult {
            success: true,
            data,
        })
    }

    /// 从文件扩展名识别导出格式（ttl / nt / nq / trig / rdf / xml / jsonld）。
    ///
    /// 与 [`Self::detect_rdf_format`] 不同，JSON-LD 是合法导出格式
    /// （oxigraph 序列化器支持 Streaming Profile）。
    pub fn detect_export_format(
        file_path: &str,
    ) -> Result<oxigraph::io::RdfFormat, Box<dyn std::error::Error + Send + Sync>> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        match extension.to_ascii_lowercase().as_str() {
            "json" | "jsonld" => Ok(oxigraph::io::RdfFormat::JsonLd {
                profile: oxigraph::io::JsonLdProfileSet::empty(),
            }),
            _ => oxigraph::io::RdfFormat::from_extension(extension).ok_or_else(|| {
                format!(
                    "无法从扩展名「{}」识别 RDF 格式（支持 ttl / nt / nq / trig / rdf / xml / jsonld）",
                    if extension.is_empty() { "（无）" } else { extension }
                )
                .into()
            }),
        }
    }

    /// 把默认图导出为 RDF 文件，返回导出的三元组数。
    ///
    /// 导出只读、不推进知识版本；命名图不在导出范围内（与 RDF 导入的默认图
    /// 语义对称）。
    pub async fn export_rdf_file(
        &self,
        file_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let format = Self::detect_export_format(file_path)?;
        let file = fs::File::create(file_path)?;
        self.export_rdf_to_writer(format, file).await
    }

    /// 把默认图序列化到任意 writer，返回写出的三元组数。
    pub async fn export_rdf_to_writer<W: std::io::Write>(
        &self,
        format: oxigraph::io::RdfFormat,
        writer: W,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        self.perf.bump("rdf_export");
        let count = store
            .quads_for_pattern(None, None, None, Some(GraphNameRef::DefaultGraph))
            .count();
        store
            .dump_graph_to_writer(GraphNameRef::DefaultGraph, format, writer)
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                format!("RDF 导出失败：{}", error).into()
            })?;
        Ok(count)
    }

    pub async fn save_knowledge(
        &self,
        items: Vec<KnowledgeItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.save_knowledge_with_receipt(items).await.map(|_| ())
    }

    /// 带回执的知识保存：谓词准入、别名/逆向归一、去重合并、功能谓词冲突
    /// 标记（不覆盖）、撤销动作记录。返回的回执可交给 `undo_import` 撤回。
    pub async fn save_knowledge_with_receipt(
        &self,
        items: Vec<KnowledgeItem>,
    ) -> Result<predicate_registry::ImportReceipt, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;

        let mut receipt = predicate_registry::ImportReceipt::default();
        for item in items {
            match item {
                KnowledgeItem::Entity { label, properties } => {
                    self.save_entity_with_receipt(&store, &label, &properties, &mut receipt)?;
                }
                KnowledgeItem::Relationship {
                    from_id,
                    to_id,
                    rel_type,
                    ..
                } => {
                    self.save_relationship_with_receipt(
                        &store,
                        &from_id,
                        &to_id,
                        &rel_type,
                        &mut receipt,
                    )?;
                }
            }
        }
        // 只有确有写入才推进版本
        if receipt.added > 0 {
            self.bump_revision();
        }
        Ok(receipt)
    }

    fn save_entity_with_receipt(
        &self,
        store: &Store,
        label: &str,
        properties: &HashMap<String, String>,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let subject_name = properties
            .get("name")
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        // 图谱观测的 name 证据（属性迭代前先取，避免被排序视图遮蔽）
        let name_literal = properties.get("name").cloned();
        let subject_iri = crate::identity::entity_uri(BASE_NS, &subject_name);
        let subject = NamedNode::new(subject_iri.clone())?;
        let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
        let entity_type_iri = format!("{}{}", BASE_NS, label);
        let entity_type = NamedNode::new(entity_type_iri.clone())?;

        let type_quad = Quad::new(
            subject.clone(),
            rdf_type,
            entity_type,
            GraphName::DefaultGraph,
        );
        if store.contains(&type_quad)? {
            receipt.duplicates += 1;
        } else {
            store.insert(&type_quad)?;
            receipt.added += 1;
            receipt
                .actions
                .push(predicate_registry::ImportAction::Named {
                    subject: subject_iri.clone(),
                    predicate: format!("{}type", RDF_NS),
                    object: entity_type_iri,
                });
        }

        // 属性迭代顺序固定，回执与撤销动作可复现
        let mut properties: Vec<(&String, &String)> = properties.iter().collect();
        properties.sort();
        for (key, value) in properties {
            let predicate_iri = format!("{}{}", BASE_NS, key);
            let conflict = {
                let registry = self.predicate_registry.read();
                registry.conflict(&subject_name, key, value)
            };
            if let Some(existing) = conflict {
                receipt.conflicts.push(predicate_registry::ImportConflict {
                    subject: subject_name.clone(),
                    predicate: key.clone(),
                    existing,
                    incoming: value.clone(),
                });
                self.predicate_registry
                    .write()
                    .note_conflict(&subject_name, key, value);
                continue;
            }
            let predicate = NamedNode::new(predicate_iri.clone())?;
            let object = Literal::new_simple_literal(value);
            let quad = Quad::new(subject.clone(), predicate, object, GraphName::DefaultGraph);
            if store.contains(&quad)? {
                receipt.duplicates += 1;
                receipt.merged += 1;
            } else {
                store.insert(&quad)?;
                receipt.added += 1;
                receipt
                    .actions
                    .push(predicate_registry::ImportAction::Literal {
                        subject: subject_iri.clone(),
                        predicate: predicate_iri,
                        value: value.clone(),
                    });
                self.predicate_registry.write().observe(
                    &subject_name,
                    key,
                    value,
                    Some("知识导入"),
                );
            }
        }
        // 图谱观测：实体身份进入注册表（唯一自动确认来源；name 属性即别名证据）
        if let Some(name) = name_literal {
            let mut types = BTreeSet::new();
            types.insert(label.to_string());
            self.entity_registry.write().observe_kb_entity(
                &subject_name,
                &types,
                &[name],
                "文本导入",
            );
        }
        Ok(())
    }

    fn save_relationship_with_receipt(
        &self,
        store: &Store,
        from_id: &str,
        to_id: &str,
        rel_type: &str,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let from_id = from_id.trim();
        let to_id = to_id.trim();
        let rel_type = rel_type.trim();
        if from_id.is_empty() || to_id.is_empty() || rel_type.is_empty() {
            return Ok(());
        }

        // 谓词准入：未登记谓词先登记为「待确认」；严格准入下不落库
        let resolution = self.predicate_registry.read().resolve(rel_type);
        if !resolution.known {
            let strict = self.predicate_registry.read().strict_admission();
            if strict {
                if !receipt.pending_admissions.iter().any(|p| p == rel_type) {
                    receipt.pending_admissions.push(rel_type.to_string());
                }
                return Ok(());
            }
            let admitted = self.predicate_registry.write().admit(rel_type, "文本导入");
            if admitted && !receipt.provisional_predicates.iter().any(|p| p == rel_type) {
                receipt.provisional_predicates.push(rel_type.to_string());
                receipt.registry_changed = true;
            }
        }

        // 逆向谓词统一方向落库：徒弟(A,B) ≡ 师傅(B,A)
        let (subject_name, object_name) = if resolution.inverted {
            (to_id, from_id)
        } else {
            (from_id, to_id)
        };
        let predicate_name = resolution.canonical.clone();

        // 关系两端至少要有 name 字面量，保证按名称检索能找到它们
        self.ensure_entity_name(store, subject_name, receipt)?;
        self.ensure_entity_name(store, object_name, receipt)?;
        // 图谱观测：关系端点身份进入注册表（规范名即图谱键）
        {
            let empty_types = BTreeSet::new();
            let mut registry = self.entity_registry.write();
            registry.observe_kb_entity(
                subject_name,
                &empty_types,
                &[subject_name.to_string()],
                "文本导入",
            );
            registry.observe_kb_entity(
                object_name,
                &empty_types,
                &[object_name.to_string()],
                "文本导入",
            );
        }

        let conflict = {
            let registry = self.predicate_registry.read();
            registry.conflict(subject_name, &predicate_name, object_name)
        };
        if let Some(existing) = conflict {
            receipt.conflicts.push(predicate_registry::ImportConflict {
                subject: subject_name.to_string(),
                predicate: predicate_name.clone(),
                existing,
                incoming: object_name.to_string(),
            });
            self.predicate_registry.write().note_conflict(
                subject_name,
                &predicate_name,
                object_name,
            );
            return Ok(());
        }

        // 模式闸门：定义域/值域校验（实体类型未知时按开放世界放行）。
        // 类型体系开启闭包档时按别名/子类/互斥判定，并记录与精确集合判定的
        // 差异（误杀消解 / 收紧），差异只统计、不改变本次结果。
        let subject_types = entity_type_labels(store, subject_name)?;
        let object_types = entity_type_labels(store, object_name)?;
        let violation = {
            let registry = self.predicate_registry.read();
            let types = self.type_system.read();
            let exact = registry.check_schema(
                subject_name,
                &predicate_name,
                object_name,
                &subject_types,
                &object_types,
            );
            if types.closure_gate() {
                let closure = registry.check_schema_with_types(
                    subject_name,
                    &predicate_name,
                    object_name,
                    &subject_types,
                    &object_types,
                    &types,
                );
                if exact.is_some() && closure.is_none() {
                    receipt.gate_relaxed += 1;
                } else if exact.is_none() && closure.is_some() {
                    receipt.gate_tightened += 1;
                }
                closure
            } else {
                exact
            }
        };
        if let Some(violation) = violation {
            receipt.schema_violations.push(violation);
            return Ok(());
        }

        let subject_iri = crate::identity::entity_uri(BASE_NS, &subject_name);
        let object_iri = crate::identity::entity_uri(BASE_NS, &object_name);
        let predicate_iri = format!("{}{}", BASE_NS, predicate_name);
        let subject = NamedNode::new(subject_iri.clone())?;
        let predicate = NamedNode::new(predicate_iri.clone())?;
        let object = NamedNode::new(object_iri.clone())?;
        let quad = Quad::new(subject, predicate, object, GraphName::DefaultGraph);
        if store.contains(&quad)? {
            receipt.duplicates += 1;
        } else {
            store.insert(&quad)?;
            receipt.added += 1;
            receipt
                .actions
                .push(predicate_registry::ImportAction::Named {
                    subject: subject_iri,
                    predicate: predicate_iri,
                    object: object_iri,
                });
            self.predicate_registry.write().observe(
                subject_name,
                &predicate_name,
                object_name,
                Some("文本导入"),
            );
        }
        Ok(())
    }

    /// 确保实体有 name 字面量（辅助三元组，仅在缺失时插入并记录撤销动作）。
    fn ensure_entity_name(
        &self,
        store: &Store,
        name: &str,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let subject_iri = crate::identity::entity_uri(BASE_NS, name);
        let predicate_iri = format!("{}name", BASE_NS);
        let subject = NamedNode::new(subject_iri.clone())?;
        let predicate = NamedNode::new(predicate_iri.clone())?;
        let quad = Quad::new(
            subject,
            predicate,
            Literal::new_simple_literal(name),
            GraphName::DefaultGraph,
        );
        if store.contains(&quad)? {
            return Ok(());
        }
        store.insert(&quad)?;
        receipt.added += 1;
        receipt
            .actions
            .push(predicate_registry::ImportAction::Literal {
                subject: subject_iri,
                predicate: predicate_iri,
                value: name.to_string(),
            });
        self.predicate_registry
            .write()
            .observe(name, "name", name, Some("文本导入"));
        Ok(())
    }

    /// 撤回一次导入：逐条移除回执中记录的三元组，并同步回撤画像观测。
    pub async fn undo_import(
        &self,
        receipt: &predicate_registry::ImportReceipt,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let mut removed = 0usize;
        for action in &receipt.actions {
            let quad = match action {
                predicate_registry::ImportAction::Named {
                    subject,
                    predicate,
                    object,
                } => Quad::new(
                    NamedNode::new(subject.clone())?,
                    NamedNode::new(predicate.clone())?,
                    NamedNode::new(object.clone())?,
                    GraphName::DefaultGraph,
                ),
                predicate_registry::ImportAction::Literal {
                    subject,
                    predicate,
                    value,
                } => Quad::new(
                    NamedNode::new(subject.clone())?,
                    NamedNode::new(predicate.clone())?,
                    Literal::new_simple_literal(value),
                    GraphName::DefaultGraph,
                ),
            };
            if store.remove(&quad).unwrap_or(false) {
                removed += 1;
            }
        }
        {
            let mut registry = self.predicate_registry.write();
            for action in &receipt.actions {
                match action {
                    predicate_registry::ImportAction::Named {
                        subject,
                        predicate,
                        object,
                    } => {
                        registry.retract(
                            &Self::local_kg_name(subject),
                            &Self::local_kg_name(predicate),
                            &Self::local_kg_name(object),
                        );
                    }
                    predicate_registry::ImportAction::Literal {
                        subject,
                        predicate,
                        value,
                    } => {
                        registry.retract(
                            &Self::local_kg_name(subject),
                            &Self::local_kg_name(predicate),
                            value,
                        );
                    }
                }
            }
        }
        // 只有确有条目被移除才推进版本；重复撤销是无操作
        if removed > 0 {
            self.bump_revision();
        }
        Ok(removed)
    }

    /// 更正一个字面量属性：删除旧值三元组、写入新值三元组，并同步谓词画像。
    ///
    /// 返回 `(removed, inserted)`。旧值必须能在图谱中找到才会写入新值：
    /// 「更正」是说旧值错了，不是无条件覆盖；找不到旧值就诚实地不改。
    pub async fn correct_object_value(
        &self,
        entity: &str,
        predicate: &str,
        old_value: &str,
        new_value: &str,
    ) -> Result<(usize, bool), Box<dyn std::error::Error + Send + Sync>> {
        let entity = entity.trim();
        let old_value = old_value.trim();
        let new_value = new_value.trim();
        if entity.is_empty() || predicate.trim().is_empty() || old_value.is_empty() {
            return Ok((0, false));
        }
        let (canonical, inverted) = {
            let registry = self.predicate_registry.read();
            let resolution = registry.resolve(predicate.trim());
            (resolution.canonical.clone(), resolution.inverted)
        };
        if inverted {
            return Ok((0, false));
        }

        let subject_iri = crate::identity::entity_uri(BASE_NS, entity);
        let predicate_iri = format!("{}{}", BASE_NS, canonical);
        let subject = NamedNode::new(subject_iri)?;
        let predicate_node = NamedNode::new(predicate_iri)?;
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(
                Some(subject.as_ref().into()),
                Some(predicate_node.as_ref().into()),
                None,
                None,
            )
            .collect::<Result<Vec<_>, _>>()?;
        let mut removed = 0usize;
        let mut had_literal = false;
        let mut had_iri = false;
        for quad in existing {
            let matched = match &quad.object {
                Term::Literal(literal) => literal.value() == old_value,
                Term::NamedNode(node) => {
                    node.as_str() == old_value || Self::local_kg_name(node.as_str()) == old_value
                }
                _ => false,
            };
            if !matched {
                continue;
            }
            if store.remove(&quad)? {
                removed += 1;
                match &quad.object {
                    Term::Literal(_) => had_literal = true,
                    Term::NamedNode(_) => had_iri = true,
                    _ => {}
                }
            }
        }
        if removed == 0 {
            return Ok((0, false));
        }
        let mut inserted = false;
        if !new_value.is_empty() {
            if had_literal {
                let new_quad = Quad::new(
                    subject.clone(),
                    predicate_node.clone(),
                    Literal::new_simple_literal(new_value),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&new_quad)? {
                    store.insert(&new_quad)?;
                    inserted = true;
                }
            }
            if had_iri {
                let object = NamedNode::new(crate::identity::entity_uri(BASE_NS, new_value))?;
                let new_quad = Quad::new(
                    subject,
                    predicate_node,
                    object.clone(),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&new_quad)? {
                    store.insert(&new_quad)?;
                    inserted = true;
                }
                // 快照只保留「两端都有 kg:name」的关系：新宾语必须自带名称，
                // 否则更正后的关系会在推理快照里凭空消失。
                let name_iri = format!("{}name", BASE_NS);
                let name_quad = Quad::new(
                    object,
                    NamedNode::new(name_iri)?,
                    Literal::new_simple_literal(new_value),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&name_quad)? {
                    store.insert(&name_quad)?;
                    inserted = true;
                }
            }
        }
        drop(store);
        self.predicate_registry.write().correct(
            entity,
            &canonical,
            old_value,
            if new_value.is_empty() {
                old_value
            } else {
                new_value
            },
        );
        self.bump_revision();
        Ok((removed, inserted))
    }

    /// `http://example.org/kg/entity/张三` → `张三`（百分号编码无损反解），其它 IRI 原样返回。
    fn local_kg_name(iri: &str) -> String {
        crate::identity::local_name(BASE_NS, iri).unwrap_or_else(|| iri.to_string())
    }

    /// 主体 IRI → 图谱键：优先按 kg 实体路径无损反解；外部 IRI 退回「尾部段」语义。
    fn entity_key_from_iri(iri: &str) -> String {
        if let Some(name) = crate::identity::local_name(BASE_NS, iri) {
            return name;
        }
        match iri.rsplit_once("entity/") {
            Some((_, tail)) => tail.to_string(),
            None => iri.to_string(),
        }
    }

    /// 把知识库中现存三元组观测进谓词画像（供「谓词报告」与候选发现使用）。
    pub async fn refresh_predicate_registry(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let observed: Vec<(String, String, String)> = {
            let store = self.store.lock().await;
            let mut observed = Vec::new();
            for quad in store.iter() {
                let quad = match quad {
                    Ok(quad) => quad,
                    Err(_) => continue,
                };
                // 命名图（规则/派生账本）不是知识图谱：谓词画像只观测默认图
                if quad.graph_name != GraphName::DefaultGraph {
                    continue;
                }
                let predicate = Self::local_kg_name(quad.predicate.as_str());
                // rdf:type 不是领域谓词，跳过
                if predicate == "type" {
                    continue;
                }
                let subject = match &quad.subject {
                    Subject::NamedNode(node) => Self::local_kg_name(node.as_str()),
                    _ => continue,
                };
                let object = match &quad.object {
                    Term::NamedNode(node) => Self::local_kg_name(node.as_str()),
                    Term::Literal(literal) => literal.value().to_string(),
                    _ => continue,
                };
                observed.push((subject, predicate, object));
            }
            observed
        };
        let mut registry = self.predicate_registry.write();
        for (subject, predicate, object) in observed {
            registry.observe(&subject, &predicate, &object, Some("知识图谱"));
        }
        Ok(())
    }

    /// 从配置声明谓词画像，并设置准入策略。
    pub fn configure_predicates(
        &self,
        profiles: &[predicate_registry::PredicateProfileConfig],
        strict_admission: bool,
    ) {
        let mut registry = self.predicate_registry.write();
        registry.set_strict_admission(strict_admission);
        for profile in profiles {
            registry.declare(profile);
        }
    }

    /// 从配置装载类型体系（别名/子类/互斥 + 闭包闸门开关）。
    pub fn configure_type_system(&self, config: &type_system::TypeSystemConfig) {
        *self.type_system.write() = type_system::TypeSystem::from_config(config);
    }

    /// 类型体系快照（质量护照等只读消费者使用）。
    pub fn type_system_snapshot(&self) -> type_system::TypeSystem {
        self.type_system.read().clone()
    }

    /// 类型体系报告（确定性）。
    pub fn type_system_report(&self) -> String {
        self.type_system.read().report()
    }

    /// 模式闸门是否处于闭包档。
    pub fn closure_gate(&self) -> bool {
        self.type_system.read().closure_gate()
    }

    /// 谓词注册表快照（质量护照等只读消费者使用）。
    pub fn predicate_registry_snapshot(&self) -> predicate_registry::PredicateRegistry {
        self.predicate_registry.read().clone()
    }

    /// 全部功能谓词冲突笔记（被拒入值；图谱快照里看不到的「不同说法」）。
    pub fn predicate_conflicts(&self) -> Vec<predicate_registry::ImportConflict> {
        self.predicate_registry.read().conflicts()
    }

    /// 谓词画像 + 候选发现报告。
    pub fn predicate_registry_report(&self) -> String {
        let registry = self.predicate_registry.read();
        let mut output = registry.report();
        output.push_str(&registry.findings_report());
        output
    }

    /// 全部（词面 → 规范谓词, 是否逆向）映射，供对话层把问句归一到规范方向。
    pub fn predicate_query_pairs(&self) -> Vec<(String, String, bool)> {
        self.predicate_registry.read().alias_pairs()
    }

    pub fn register_predicate_alias(&self, alias: &str, canonical: &str) -> bool {
        self.predicate_registry
            .write()
            .register_alias(alias, canonical, "人工确认")
    }

    pub fn register_predicate_inverse(&self, left: &str, right: &str) -> bool {
        self.predicate_registry
            .write()
            .register_inverse(left, right, "人工确认")
    }

    pub fn mark_predicate_functional(&self, predicate: &str, functional: bool) -> bool {
        let mut registry = self.predicate_registry.write();
        if !registry.is_known(predicate) {
            registry.admit(predicate, "人工确认");
            registry.confirm(predicate);
        }
        registry.mark_functional(predicate, functional)
    }

    pub fn confirm_predicate(&self, predicate: &str) -> bool {
        self.predicate_registry.write().confirm(predicate)
    }

    /// 拒绝待定谓词：移除画像并删除图谱中该谓词的全部三元组。
    pub async fn reject_predicate(
        &self,
        predicate: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let canonical = self
            .predicate_registry
            .read()
            .resolve(predicate)
            .canonical
            .clone();
        let provisional = self
            .predicate_registry
            .read()
            .profile(&canonical)
            .map_or(false, |profile| profile.provisional);
        if !provisional {
            return Ok(0);
        }
        let removed = {
            let store = self.store.lock().await;
            let predicate_iri = NamedNode::new(format!("{}{}", BASE_NS, canonical))?;
            let quads: Vec<Quad> = store
                .quads_for_pattern(None, Some(predicate_iri.as_ref()), None, None)
                .collect::<Result<Vec<_>, _>>()?;
            let mut removed = 0usize;
            for quad in quads {
                if store.remove(&quad).unwrap_or(false) {
                    removed += 1;
                }
            }
            if removed > 0 {
                self.bump_revision();
            }
            removed
        };
        self.predicate_registry.write().reject(&canonical);
        Ok(removed)
    }

    pub fn save_predicate_registry(&self, path: &str) -> std::io::Result<()> {
        self.predicate_registry.read().save_to_path(path)
    }

    pub fn load_predicate_registry(&self, path: &str) -> bool {
        match predicate_registry::PredicateRegistry::load_from_path(path) {
            Some(loaded) => {
                *self.predicate_registry.write() = loaded;
                true
            }
            None => false,
        }
    }

    // ------------------------------------------------------------------
    // 实体身份注册表：谓词注册表的实体侧对偶（提名 / 确认 / 拒绝 / 忘记 / 合并）
    // ------------------------------------------------------------------

    /// 从配置声明实体画像（声明即已确认的显式承诺）。
    pub fn configure_entities(&self, profiles: &[entity_registry::EntityProfileConfig]) {
        let mut registry = self.entity_registry.write();
        for profile in profiles {
            registry.declare(profile);
        }
    }

    pub fn entity_registry_snapshot(&self) -> entity_registry::EntityRegistry {
        self.entity_registry.read().clone()
    }

    /// 身份台账报告（确定性；含待确认提名与墓碑）。
    pub fn entity_registry_report(&self) -> String {
        self.entity_registry.read().report()
    }

    /// 铸 URI 前的宪法检查（Exists / Redirect / Variant / Ambiguous / Fresh）。
    pub fn entity_mint_check(&self, name: &str) -> entity_registry::MintVerdict {
        self.entity_registry.read().mint_check(name)
    }

    pub fn resolve_entity_identity(&self, surface: &str) -> entity_registry::EntityResolution {
        self.entity_registry.read().resolve(surface)
    }

    /// 已确认别名对（词面 → 规范名），供链接器按持久别名识别。
    pub fn confirmed_entity_aliases(&self) -> Vec<(String, String)> {
        self.entity_registry.read().confirmed_alias_pairs()
    }

    /// 确认一个别名（显式承诺：词义改写需要第二次签字）。
    pub fn confirm_entity_alias(&self, surface: &str, canonical: &str) -> bool {
        let changed = self
            .entity_registry
            .write()
            .confirm_alias(surface, canonical);
        if changed {
            self.perf.bump("identity_alias_confirm");
        }
        changed
    }

    /// 拒绝一个别名提名（墓碑：不予自动提名，也不参与解析）。
    pub fn reject_entity_alias(&self, surface: &str, canonical: &str) -> bool {
        let changed = self
            .entity_registry
            .write()
            .reject_alias(surface, canonical);
        if changed {
            self.perf.bump("identity_alias_reject");
        }
        changed
    }

    /// 忘记一个已确认别名（撤销 + 墓碑）；返回受影响的规范身份。
    pub fn forget_entity_alias(&self, surface: &str) -> Vec<String> {
        let affected = self.entity_registry.write().forget_alias(surface);
        if !affected.is_empty() {
            self.perf.bump("identity_alias_forget");
        }
        affected
    }

    /// 合并身份（重命名 = 合并 + 重定向；URI 永不回收）。
    pub fn merge_entity_identities(
        &self,
        from: &str,
        into: &str,
    ) -> Result<entity_registry::MergeRecord, String> {
        let record = self.entity_registry.write().merge(from, into)?;
        self.perf.bump("identity_merge");
        Ok(record)
    }

    /// 提名文本模式抽出的别名候选（只提名，不写库）。
    pub fn nominate_entity_aliases(
        &self,
        candidates: &[entity_registry::AliasCandidate],
    ) -> Vec<entity_registry::NominationOutcome> {
        let types = self.type_system.read().clone();
        let mut registry = self.entity_registry.write();
        let mut outcomes = Vec::new();
        for candidate in candidates {
            let evidence = if candidate.evidence.is_empty() {
                candidate.pattern.clone()
            } else {
                format!("{}：{}", candidate.pattern, candidate.evidence)
            };
            let outcome = registry.nominate_alias_with_types(
                &candidate.surface,
                &candidate.canonical,
                entity_registry::NominationOrigin::Import,
                &evidence,
                Some(&types),
            );
            match outcome.kind {
                entity_registry::NominationKind::Confirmed => {
                    self.perf.bump("identity_nominate_confirmed")
                }
                entity_registry::NominationKind::Pending => {
                    self.perf.bump("identity_nominate_pending")
                }
                entity_registry::NominationKind::TypeConflict => {
                    self.perf.bump("identity_nominate_type_conflict")
                }
                entity_registry::NominationKind::Tombstoned => {
                    self.perf.bump("identity_nominate_tombstoned")
                }
                _ => {}
            }
            outcomes.push(outcome);
        }
        outcomes
    }

    /// 语境指称绑定经二次确认后的别名提名（来源＝用户确认，仍走类型闸门）。
    pub fn nominate_dialogue_aliases(
        &self,
        candidates: &[entity_registry::AliasCandidate],
    ) -> Vec<entity_registry::NominationOutcome> {
        let types = self.type_system.read().clone();
        let mut registry = self.entity_registry.write();
        let mut outcomes = Vec::new();
        for candidate in candidates {
            let evidence = if candidate.evidence.is_empty() {
                candidate.pattern.clone()
            } else {
                format!("{}：{}", candidate.pattern, candidate.evidence)
            };
            let outcome = registry.nominate_alias_with_types(
                &candidate.surface,
                &candidate.canonical,
                entity_registry::NominationOrigin::Dialogue,
                &evidence,
                Some(&types),
            );
            match outcome.kind {
                entity_registry::NominationKind::Confirmed => {
                    self.perf.bump("identity_dialogue_confirmed")
                }
                entity_registry::NominationKind::Pending => {
                    self.perf.bump("identity_dialogue_pending")
                }
                entity_registry::NominationKind::TypeConflict => {
                    self.perf.bump("identity_dialogue_type_conflict")
                }
                entity_registry::NominationKind::Tombstoned => {
                    self.perf.bump("identity_dialogue_tombstoned")
                }
                _ => {}
            }
            outcomes.push(outcome);
        }
        outcomes
    }

    /// 从知识图谱刷新身份观测：`rdf:type` 与 `kg:name`（唯一自动确认来源）。
    ///
    /// 按规范名排序处理，保证同一图谱得到逐字节一致的身份台账。
    pub async fn refresh_entity_registry(
        &self,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut observed: std::collections::BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> =
            std::collections::BTreeMap::new();
        {
            let store = self.store.lock().await;
            let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
            for quad in store.iter() {
                let quad = match quad {
                    Ok(quad) => quad,
                    Err(_) => continue,
                };
                if quad.graph_name != GraphName::DefaultGraph {
                    continue;
                }
                let Subject::NamedNode(subject) = &quad.subject else {
                    continue;
                };
                let Some(canonical) = crate::identity::entity_local_name(BASE_NS, subject.as_str())
                else {
                    continue;
                };
                let entry = observed.entry(canonical).or_default();
                if quad.predicate == rdf_type {
                    if let Term::NamedNode(type_node) = &quad.object {
                        if let Some(label) = type_node.as_str().strip_prefix(BASE_NS) {
                            entry.0.insert(label.to_string());
                        }
                    }
                } else if quad.predicate.as_str() == NAME_PREDICATE_IRI {
                    if let Term::Literal(literal) = &quad.object {
                        entry.1.insert(literal.value().to_string());
                    }
                }
            }
        }
        let mut registry = self.entity_registry.write();
        let mut changed = 0usize;
        for (canonical, (types, names)) in observed {
            let names: Vec<String> = names.into_iter().collect();
            if registry.observe_kb_entity(&canonical, &types, &names, "知识图谱") {
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// 从已加载的事实观测实体身份（与 `refresh_entity_registry` 等价，省一次全库扫描）。
    pub fn observe_entity_facts(&self, facts: &[PrologFact]) -> usize {
        let mut by_name: std::collections::BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> =
            std::collections::BTreeMap::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    let entry = by_name.entry(name.clone()).or_default();
                    if !entity_type.trim().is_empty() {
                        entry.0.insert(entity_type.clone());
                    }
                    if let Some(value) = properties.get("name") {
                        if !value.trim().is_empty() {
                            entry.1.insert(value.clone());
                        }
                    }
                }
                PrologFact::Relation { from, to, .. } => {
                    by_name.entry(from.clone()).or_default();
                    by_name.entry(to.clone()).or_default();
                }
            }
        }
        let mut registry = self.entity_registry.write();
        let mut changed = 0usize;
        for (canonical, (types, names)) in by_name {
            let names: Vec<String> = names.into_iter().collect();
            if registry.observe_kb_entity(&canonical, &types, &names, "知识图谱") {
                changed += 1;
            }
        }
        changed
    }

    pub fn save_entity_registry(&self, path: &str) -> std::io::Result<()> {
        self.entity_registry.read().save_to_path(path)
    }

    /// 加载身份注册表（版本门：折叠算法版本高于支持版本时不加载）。
    pub fn load_entity_registry(&self, path: &str) -> bool {
        match entity_registry::EntityRegistry::load_from_path(path) {
            Some(loaded) => {
                *self.entity_registry.write() = loaded;
                true
            }
            None => false,
        }
    }

    pub async fn get_relevant_facts(
        &self,
        query_entities: &[String],
        max_hops: usize,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        if query_entities.is_empty() {
            // 使用整图快照缓存，仅保留实体事实
            let snapshot = self.load_graph_snapshot().await?;
            return Ok(snapshot
                .iter()
                .filter(|fact| matches!(fact, PrologFact::Entity { .. }))
                .cloned()
                .collect());
        }

        // 同一知识版本下的同一组实体（排序后）与跳数 → 直接复用上次结果。
        // 问句实体顺序不影响语义，键里统一排序消除排列差异
        let mut sorted_entities: Vec<&str> = query_entities.iter().map(String::as_str).collect();
        sorted_entities.sort_unstable();
        let cache_key = format!("{}|{}", max_hops, sorted_entities.join("\u{1}"));
        let revision = self.revision.load(Ordering::Acquire);
        {
            let mut cache = self.relevant_cache.lock();
            if let Some(facts) = cache.get(revision, &cache_key) {
                self.perf.bump("relevant_facts_cache_hit");
                return Ok(facts.as_ref().clone());
            }
        }
        self.perf.bump("relevant_facts_cache_miss");
        self.perf.bump("relevant_facts_graph_scan");

        let mut facts = Vec::new();
        let store = self.store.lock().await;

        // 构建实体名称的匹配模式
        let mut combined_pattern = String::new();
        for (index, entity) in query_entities.iter().enumerate() {
            if index > 0 {
                combined_pattern.push('|');
            }
            combined_pattern.push_str(".*");
            combined_pattern.push_str(&regex::escape(entity));
            combined_pattern.push_str(".*");
        }

        // 第一步：找到直接匹配的实体
        let direct_matches = self
            .find_entities_by_pattern(&store, &combined_pattern)
            .await?;
        let mut relevant_entities: std::collections::HashSet<String> =
            direct_matches.into_iter().collect();

        // 第二步：扩展到相邻实体（基于指定跳数，仅扩展每轮新增的前沿节点）
        let mut frontier: Vec<String> = relevant_entities.iter().cloned().collect();
        for _hop in 0..max_hops {
            if frontier.is_empty() {
                break;
            }
            let neighbors = self.find_neighbors(&store, &frontier).await?;
            let mut next_frontier: Vec<String> = Vec::new();
            for neighbor in neighbors {
                if relevant_entities.insert(neighbor.clone()) {
                    next_frontier.push(neighbor);
                }
            }
            frontier = next_frontier;
        }

        // 第三步：为相关实体构建Prolog事实
        facts.extend(self.build_entity_facts(&store, &relevant_entities).await?);

        // 第四步：添加实体间的关系
        let relations = self
            .find_relations_between(&store, &relevant_entities)
            .await?;
        facts.extend(relations);

        let facts = Arc::new(facts);
        if self
            .relevant_cache
            .lock()
            .put(revision, cache_key, facts.clone())
            > 0
        {
            self.perf.bump("relevant_facts_cache_evict");
        }
        Ok(facts.as_ref().clone())
    }

    pub async fn load_all_facts(
        &self,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        // 与 get_relevant_facts 一致：直接复用整图快照缓存中的实体事实
        let snapshot = self.load_graph_snapshot().await?;
        Ok(snapshot
            .iter()
            .filter(|fact| matches!(fact, PrologFact::Entity { .. }))
            .cloned()
            .collect())
    }

    /// 整图快照（实体属性 + 关系），在两次写操作之间缓存。
    ///
    /// 正确性依赖：所有写入口在持有存储锁期间、且在实际写入完成之后自增
    /// `revision`（version-after-write）；因此命中缓存意味着自缓存构建以来
    /// 没有任何成功写入，内容必然一致。
    pub async fn load_graph_snapshot(
        &self,
    ) -> Result<Arc<Vec<PrologFact>>, Box<dyn std::error::Error + Send + Sync>> {
        let revision = self.revision.load(Ordering::Acquire);
        {
            let cache = self.snapshot_cache.lock();
            if let Some((cached_revision, snapshot)) = cache.as_ref() {
                if *cached_revision == revision {
                    self.perf.bump("graph_snapshot_cache_hit");
                    return Ok(snapshot.clone());
                }
            }
        }
        self.perf.bump("graph_snapshot_scan");

        let (facts, relations) = {
            let store = self.store.lock().await;
            Self::scan_graph_parts(&store)
        };
        let mut snapshot_facts = facts;
        snapshot_facts.extend(relations);
        let snapshot = Arc::new(snapshot_facts);

        let mut cache = self.snapshot_cache.lock();
        *cache = Some((revision, snapshot.clone()));
        Ok(snapshot)
    }

    /// 单次遍历存储，同时构建整图的实体事实与关系事实。
    ///
    /// 语义与 `load_all_facts` + `load_all_relations` 的组合保持一致：
    /// - 实体只保留拥有 `rdf:type` 的主体；同一实体多个类型时取最先遇到者，
    ///   重复属性取最后遇到者（按主体聚簇的存储序）；
    /// - 关系仅保留两端都有 `kg:name` 的命名主体，且排除自环与 `rdf:type`。
    fn scan_graph_parts(store: &Store) -> (Vec<PrologFact>, Vec<PrologFact>) {
        let mut entity_types: HashMap<String, String> = HashMap::new();
        let mut entity_props: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut subject_names: HashMap<Term, Vec<String>> = HashMap::new();
        let mut relation_candidates: Vec<(Term, Term, String)> = Vec::new();

        for quad in store.iter() {
            let quad = match quad {
                Ok(quad) => quad,
                Err(_) => continue,
            };
            // 命名图（规则/派生账本）不参与知识快照：只拍平默认图
            if quad.graph_name != GraphName::DefaultGraph {
                continue;
            }

            // 命名节点直接取 IRI 尾部（kg 实体 IRI 经百分号解码反解；外部 IRI 按尾部段语义）
            let entity_name = match &quad.subject {
                Subject::NamedNode(node) => Self::entity_key_from_iri(node.as_str()),
                _ => {
                    let subject_str = quad.subject.to_string();
                    let trimmed = subject_str.trim_matches(|c| c == '<' || c == '>');
                    Self::entity_key_from_iri(trimmed)
                }
            };

            if quad.predicate.as_str() == RDF_TYPE_IRI {
                let type_name = match &quad.object {
                    Term::NamedNode(node) => {
                        let iri = node.as_str();
                        if iri.contains(BASE_NS) {
                            iri.replace(BASE_NS, "").to_lowercase()
                        } else {
                            iri.to_lowercase()
                        }
                    }
                    object => {
                        let type_str = object.to_string();
                        if type_str.contains(BASE_NS) {
                            type_str
                                .replace(BASE_NS, "")
                                .trim_matches(|c| c == '<' || c == '>')
                                .to_lowercase()
                        } else {
                            type_str
                                .trim_matches(|c| c == '<' || c == '>')
                                .to_lowercase()
                        }
                    }
                };
                entity_types.entry(entity_name).or_insert(type_name);
                continue;
            }

            let is_name_predicate = quad.predicate.as_str() == NAME_PREDICATE_IRI;
            let predicate_iri = quad.predicate.as_str();
            let prop_name = if predicate_iri.contains(BASE_NS) {
                predicate_iri.replace(BASE_NS, "")
            } else {
                predicate_iri.to_string()
            };

            let value_str = if let Term::Literal(lit) = &quad.object {
                lit.value().to_string()
            } else {
                match &quad.object {
                    Term::NamedNode(node) => node.as_str().to_string(),
                    object => object
                        .to_string()
                        .trim_matches(|c| c == '<' || c == '>')
                        .to_string(),
                }
            };

            let name_value = if is_name_predicate {
                Some(if let Term::Literal(_) = &quad.object {
                    value_str.clone()
                } else {
                    quad.object.to_string()
                })
            } else {
                None
            };

            entity_props
                .entry(entity_name)
                .or_default()
                .insert(prop_name.clone(), value_str);

            if matches!(&quad.object, Term::Literal(_)) {
                // 字面量宾语只登记名称，不生成关系候选
                if let Some(name_value) = name_value {
                    subject_names
                        .entry(Term::from(quad.subject))
                        .or_default()
                        .push(name_value);
                }
            } else {
                let subject_term = Term::from(quad.subject);
                if let Some(name_value) = name_value {
                    subject_names
                        .entry(subject_term.clone())
                        .or_default()
                        .push(name_value);
                }
                relation_candidates.push((subject_term, quad.object, prop_name));
            }
        }

        let mut facts = Vec::with_capacity(entity_types.len());
        for (entity_name, entity_type) in entity_types {
            let properties = entity_props.remove(&entity_name).unwrap_or_default();
            facts.push(PrologFact::Entity {
                name: entity_name,
                entity_type,
                properties,
            });
        }

        let mut relations = Vec::with_capacity(relation_candidates.len());
        for (from, to, rel_type) in relation_candidates {
            if from == to {
                continue;
            }
            if let (Some(from_names), Some(to_names)) =
                (subject_names.get(&from), subject_names.get(&to))
            {
                for from_name in from_names {
                    for to_name in to_names {
                        relations.push(PrologFact::Relation {
                            from: from_name.clone(),
                            rel_type: rel_type.clone(),
                            to: to_name.clone(),
                        });
                    }
                }
            }
        }

        (facts, relations)
    }

    /// 直答入口：同一知识版本下同一（查询类型 + 实体）复用结果（含负缓存）。
    pub async fn get_direct_answer(&self, query_type: &str, entities: &[String]) -> Option<String> {
        let cache_key = format!("{}|{}", query_type, entities.join("\u{1}"));
        let revision = self.revision.load(Ordering::Acquire);
        {
            let mut cache = self.direct_answer_cache.lock();
            if let Some(answer) = cache.get(revision, &cache_key) {
                self.perf.bump("direct_answer_cache_hit");
                return answer.clone();
            }
        }
        self.perf.bump("direct_answer_cache_miss");
        let answer = self.get_direct_answer_uncached(query_type, entities).await;
        if self
            .direct_answer_cache
            .lock()
            .put(revision, cache_key, answer.clone())
            > 0
        {
            self.perf.bump("direct_answer_cache_evict");
        }
        answer
    }

    async fn get_direct_answer_uncached(
        &self,
        query_type: &str,
        entities: &[String],
    ) -> Option<String> {
        let store = self.store.lock().await;
        self.perf.bump("direct_answer_query");

        // 根据查询类型构造不同的SPARQL查询
        let query = match query_type {
            "capital_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?capital WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:capital ?capital .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "altitude_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?altitude WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:altitude ?altitude .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "chemical_formula_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?formula WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:formula ?formula .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "author_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?author WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:author ?author .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "champion_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?champion WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:champion ?champion .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "birth_year_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?year WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:birth_year ?year .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "concept_query" | "explanation_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?definition WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:definition ?definition .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "quantity_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?quantity WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:quantity ?quantity .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "planet_query" | "largest_query" => {
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?name WHERE {{
                        ?entity kg:type "planet" .
                        ?entity kg:size_rank "1" .
                        ?entity kg:name ?name .
                    }}
                "#
                )
            }
            "time_before_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?time WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:time ?time .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "principle_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?principle WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:principle ?principle .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "comparison_query" => {
                if entities.len() < 2 {
                    return None;
                }
                let entity1 = &entities[0];
                let entity2 = &entities[1];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?difference WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?comparison kg:entity1 ?entity1 .
                        ?comparison kg:entity2 ?entity2 .
                        ?comparison kg:difference ?difference .
                    }}
                "#,
                    sparql_literal(entity1),
                    sparql_literal(entity2)
                )
            }
            "philosophy_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?philosophy WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:philosophy ?philosophy .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "impact_query" => {
                if entities.len() < 2 {
                    return None;
                }
                let subject = &entities[0];
                let target = &entities[1];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?impact WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?entity1 kg:impact ?impact .
                        ?impact kg:target ?entity2 .
                    }}
                "#,
                    sparql_literal(subject),
                    sparql_literal(target)
                )
            }
            "cause_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?cause WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:cause ?cause .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "reason_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?reason WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:reason ?reason .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            _ => return None,
        };

        // 执行SPARQL查询
        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    // 获取第一个变量的值
                                    if let Some((_, value)) = solution.iter().next() {
                                        let value_str = if let Term::Literal(lit) = value {
                                            lit.value().to_string()
                                        } else {
                                            value
                                                .to_string()
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        // 根据查询类型格式化答案
                                        let answer = match query_type {
                                            "capital_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的首都是{}", entity_name, value_str)
                                            }
                                            "altitude_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该地点"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!(
                                                    "{}的海拔高度是{}米",
                                                    entity_name, value_str
                                                )
                                            }
                                            "chemical_formula_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!(
                                                    "{}的化学分子式是{}",
                                                    entity_name, value_str
                                                )
                                            }
                                            "author_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该作品"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的作者是{}", entity_name, value_str)
                                            }
                                            "champion_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该比赛"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的冠军是{}", entity_name, value_str)
                                            }
                                            "quantity_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该项目"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}有{}个", entity_name, value_str)
                                            }
                                            "birth_year_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}诞生于{}年", entity_name, value_str)
                                            }
                                            "planet_query" | "largest_query" => {
                                                format!("最大的行星是{}", value_str)
                                            }
                                            "time_before_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该事件"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}在{}年之前", entity_name, value_str)
                                            }
                                            "concept_query" | "explanation_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}是{}", entity_name, value_str)
                                            }
                                            "principle_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的工作原理是{}", entity_name, value_str)
                                            }
                                            "comparison_query" => {
                                                if entities.len() >= 2 {
                                                    format!(
                                                        "{}和{}的区别是{}",
                                                        entities[0], entities[1], value_str
                                                    )
                                                } else {
                                                    format!("区别是{}", value_str)
                                                }
                                            }
                                            "philosophy_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的哲学思想是{}", entity_name, value_str)
                                            }
                                            "impact_query" => {
                                                if entities.len() >= 2 {
                                                    format!(
                                                        "{}对{}的影响是{}",
                                                        entities[0], entities[1], value_str
                                                    )
                                                } else {
                                                    format!("影响是{}", value_str)
                                                }
                                            }
                                            "cause_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            }
                                            "reason_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            }
                                            _ => value_str,
                                        };

                                        return Some(answer);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => return None,
                }
            }
            Err(_) => return None,
        }

        None
    }

    async fn find_entities_by_pattern(
        &self,
        store: &Store,
        pattern: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut entities = Vec::new();

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT DISTINCT ?entity ?name WHERE {{
                ?entity kg:name ?name .
                FILTER(REGEX(?name, "{}", "i"))
            }}
        "#,
            sparql_literal(pattern)
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let Some(name) = solution.get("name") {
                                    if let Term::Literal(lit) = name {
                                        entities.push(lit.value().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(entities)
    }

    async fn find_neighbors(
        &self,
        store: &Store,
        entities: &[String],
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut neighbors = Vec::new();

        if entities.is_empty() {
            return Ok(neighbors);
        }

        let mut entity_filter = String::new();
        for (index, entity) in entities.iter().enumerate() {
            if index > 0 {
                entity_filter.push_str(", ");
            }
            entity_filter.push('"');
            entity_filter.push_str(&sparql_literal(entity));
            entity_filter.push('"');
        }

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT DISTINCT ?neighbor_name WHERE {{
                {{
                    ?entity kg:name ?entity_name .
                    ?neighbor kg:name ?neighbor_name .
                    ?entity ?rel ?neighbor .
                    FILTER(?entity_name IN ({}))
                    FILTER(?rel != rdf:type)
                    FILTER(?neighbor_name NOT IN ({}))
                }} UNION {{
                    ?entity kg:name ?entity_name .
                    ?neighbor kg:name ?neighbor_name .
                    ?neighbor ?rel ?entity .
                    FILTER(?entity_name IN ({}))
                    FILTER(?rel != rdf:type)
                    FILTER(?neighbor_name NOT IN ({}))
                }}
            }}
        "#,
            entity_filter, entity_filter, entity_filter, entity_filter
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let Some(name) = solution.get("neighbor_name") {
                                    if let Term::Literal(lit) = name {
                                        neighbors.push(lit.value().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(neighbors)
    }

    async fn build_entity_facts(
        &self,
        store: &Store,
        entity_names: &HashSet<String>,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut facts = Vec::new();

        if entity_names.is_empty() {
            return Ok(facts);
        }

        let mut name_filter = String::new();
        for (index, entity) in entity_names.iter().enumerate() {
            if index > 0 {
                name_filter.push_str(", ");
            }
            name_filter.push('"');
            name_filter.push_str(&sparql_literal(entity));
            name_filter.push('"');
        }

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?name ?type ?prop ?value WHERE {{
                ?entity kg:name ?name .
                ?entity rdf:type ?type .
                FILTER(?name IN ({}))
                OPTIONAL {{ ?entity ?prop ?value . FILTER(?prop != rdf:type && ?prop != kg:name) }}
            }}
        "#,
            name_filter
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            let mut entity_data: HashMap<
                                String,
                                (String, HashMap<String, String>),
                            > = HashMap::new();

                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    let entity_name = match solution.get("name") {
                                        Some(Term::Literal(lit)) => lit.value().to_string(),
                                        _ => continue,
                                    };

                                    let entry = entity_data
                                        .entry(entity_name)
                                        .or_insert_with(|| (String::new(), HashMap::new()));

                                    if let Some(type_term) = solution.get("type") {
                                        let type_str = match type_term {
                                            Term::NamedNode(node) => node.as_str().to_string(),
                                            other => other.to_string(),
                                        };
                                        // 后续类型覆盖先前类型
                                        entry.0 = if type_str.contains(BASE_NS) {
                                            type_str
                                                .replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        } else {
                                            type_str
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        };
                                    }

                                    if let (Some(prop), Some(value)) =
                                        (solution.get("prop"), solution.get("value"))
                                    {
                                        let prop_str = match prop {
                                            Term::NamedNode(node) => node.as_str().to_string(),
                                            other => other.to_string(),
                                        };
                                        let prop_name = if prop_str.contains(BASE_NS) {
                                            prop_str
                                                .replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        } else {
                                            prop_str
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        let value_str = if let Term::Literal(lit) = value {
                                            lit.value().to_string()
                                        } else {
                                            value
                                                .to_string()
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        entry.1.insert(prop_name, value_str);
                                    }
                                }
                            }

                            // 按传入顺序输出，跳过没有类型信息的实体
                            for name in entity_names {
                                if let Some((entity_type, properties)) = entity_data.remove(name) {
                                    if !entity_type.is_empty() {
                                        facts.push(PrologFact::Entity {
                                            name: name.clone(),
                                            entity_type,
                                            properties,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }

        Ok(facts)
    }

    pub async fn load_all_relations(
        &self,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        // 使用整图快照缓存
        let snapshot = self.load_graph_snapshot().await?;
        Ok(snapshot
            .iter()
            .filter(|fact| matches!(fact, PrologFact::Relation { .. }))
            .cloned()
            .collect())
    }

    async fn find_relations_between(
        &self,
        store: &Store,
        entities: &std::collections::HashSet<String>,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let mut entity_filter = String::new();
        for (index, entity) in entities.iter().enumerate() {
            if index > 0 {
                entity_filter.push_str(", ");
            }
            entity_filter.push('"');
            entity_filter.push_str(&sparql_literal(entity));
            entity_filter.push('"');
        }

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?from_name ?to_name ?rel WHERE {{
                ?from kg:name ?from_name .
                ?to kg:name ?to_name .
                ?from ?rel ?to .
                FILTER(?from_name IN ({}))
                FILTER(?to_name IN ({}))
                FILTER(?rel != rdf:type)
                FILTER(?from != ?to)
            }}
        "#,
            entity_filter, entity_filter
        );

        self.query_relations(store, &query).await
    }

    async fn query_relations(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut relations = Vec::new();

        match Query::parse(query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let (Some(from), Some(to), Some(rel)) = (
                                    solution.get("from_name"),
                                    solution.get("to_name"),
                                    solution.get("rel"),
                                ) {
                                    let from_name = if let Term::Literal(lit) = from {
                                        lit.value().to_string()
                                    } else {
                                        from.to_string()
                                    };

                                    let to_name = if let Term::Literal(lit) = to {
                                        lit.value().to_string()
                                    } else {
                                        to.to_string()
                                    };

                                    let rel_str = match rel {
                                        Term::NamedNode(node) => node.as_str().to_string(),
                                        other => other.to_string(),
                                    };
                                    let rel_type = if rel_str.contains(BASE_NS) {
                                        rel_str
                                            .replace(BASE_NS, "")
                                            .trim_matches(|c| c == '<' || c == '>')
                                            .to_string()
                                    } else {
                                        rel_str.trim_matches(|c| c == '<' || c == '>').to_string()
                                    };

                                    relations.push(PrologFact::Relation {
                                        from: from_name,
                                        rel_type,
                                        to: to_name,
                                    });
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(relations)
    }

    /// 当前知识版本（写操作 version-after-write 自增；账本写入不推进版本）。
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// 同步规则账本：清空 `kg:graphs/rules` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 规则即数据：稳定 id + 规范逻辑形式（JSON）+ 可读词面 + 来源 + 置信度，
    /// 加上账本戳（规则集指纹 + 条数）。返回写入的规则条数。
    pub async fn sync_rule_ledger(
        &self,
        entries: &[graph_ledger::RuleLedgerEntry],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = graph_ledger::rule_quads(entries)?;
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::RULES_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("rule_ledger_sync");
        Ok(entries.len())
    }

    /// 读取规则账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_rule_ledger(
        &self,
    ) -> Result<graph_ledger::RuleLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::RULES_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("rule_ledger_load");
        Ok(graph_ledger::parse_rule_graph(quads.into_iter()))
    }

    /// 同步模式账本：清空 `kg:graphs/patterns` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 模式即数据：稳定 id + IR（JSON）+ 来源 + 路由/优先级，
    /// 加上模式集指纹戳。返回写入的模式条数。
    pub async fn sync_pattern_ledger(
        &self,
        entries: &[pattern_ir::PatternEntry],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = pattern_ir::pattern_quads(entries)?;
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::PATTERNS_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("pattern_ledger_sync");
        Ok(entries.len())
    }

    /// 读取模式账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_pattern_ledger(
        &self,
    ) -> Result<Vec<pattern_ir::PatternEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::PATTERNS_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("pattern_ledger_load");
        Ok(pattern_ir::parse_pattern_graph(quads.into_iter()))
    }

    /// 同步模式经验账本：清空 `kg:graphs/pattern-episodes` 后全量重写
    /// （幂等；不推进知识版本）。经验与模式定义分图存放，互不污染。
    pub async fn sync_pattern_episodes(
        &self,
        episodes: &[pattern_ir::PatternEpisode],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = pattern_ir::episode_quads(episodes)?;
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::EPISODES_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("pattern_episode_sync");
        Ok(episodes.len())
    }

    /// 读取模式经验账本（按经验 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_pattern_episodes(
        &self,
    ) -> Result<Vec<pattern_ir::PatternEpisode>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::EPISODES_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("pattern_episode_load");
        Ok(pattern_ir::parse_episode_graph(quads.into_iter()))
    }

    /// 物化派生账本：清空 `kg:graphs/derived` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 每条派生事实带规则来源、前提、推导轮次、置信度与物化戳
    /// （知识版本 + 规则集指纹），过期与否可审计。
    pub async fn materialize_derived_ledger(
        &self,
        entries: &[graph_ledger::DerivedLedgerEntry],
        revision: u64,
        fingerprint: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = graph_ledger::derived_quads(entries, revision, fingerprint)?;
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::DERIVED_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("derived_ledger_materialize");
        Ok(entries.len())
    }

    /// 读取派生账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_derived_ledger(
        &self,
    ) -> Result<graph_ledger::DerivedLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::DERIVED_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("derived_ledger_load");
        Ok(graph_ledger::parse_derived_graph(quads.into_iter()))
    }

    // ------------------------------------------------------------------
    // 知识融合：命名空间源图 + 融合账本 + 受审默认视图
    // ------------------------------------------------------------------

    /// 断言谓词 → RDF 谓词 IRI：`type` 归一到 rdf:type，外部 IRI 原样保留，
    /// 其余按 `BASE_NS` 展开。
    fn assertion_predicate_iri(predicate: &str) -> String {
        if predicate == "type" {
            format!("{}type", RDF_NS)
        } else if predicate.starts_with("http://") || predicate.starts_with("https://") {
            predicate.to_string()
        } else {
            format!("{}{}", BASE_NS, predicate)
        }
    }

    /// 断言宾语 → 实体 IRI：外部 IRI 原样保留，其余编码为 `kg:entity/...`。
    fn assertion_object_iri(object: &str) -> String {
        if object.starts_with("http://") || object.starts_with("https://") {
            object.to_string()
        } else {
            crate::identity::entity_uri(BASE_NS, object)
        }
    }

    /// 把命名空间断言编码为四元组（写源图或默认图共用）。
    fn assertion_quad(
        subject: &str,
        predicate: &str,
        object: &str,
        object_kind: knowledge_fusion::ObjectKind,
        graph: GraphName,
    ) -> Result<Quad, Box<dyn std::error::Error + Send + Sync>> {
        let subject_node = NamedNode::new(crate::identity::entity_uri(BASE_NS, subject))?;
        let predicate_node = NamedNode::new(Self::assertion_predicate_iri(predicate))?;
        let object_term = match object_kind {
            knowledge_fusion::ObjectKind::Literal => {
                Term::Literal(Literal::new_simple_literal(object))
            }
            knowledge_fusion::ObjectKind::Entity => {
                let iri = if predicate == "type" && !object.starts_with("http") {
                    format!("{}{}", BASE_NS, object)
                } else {
                    Self::assertion_object_iri(object)
                };
                Term::NamedNode(NamedNode::new(iri)?)
            }
        };
        Ok(Quad::new(subject_node, predicate_node, object_term, graph))
    }

    /// 从 RDF 四元组构造命名空间断言（`import namespace` 的解析辅助）：
    /// `rdf:type` 归一为谓词 `type`，kg 词面还原为本地名，外部 IRI 原样保留；
    /// 空白节点主语或词表宾语返回 None（不猜测）。
    pub fn assertion_from_rdf_quad(
        quad: &Quad,
        namespace: &str,
        source: &str,
    ) -> Option<knowledge_fusion::ScopedAssertion> {
        let subject = match &quad.subject {
            Subject::NamedNode(node) => crate::identity::entity_local_name(BASE_NS, node.as_str())
                .or_else(|| crate::identity::local_name(BASE_NS, node.as_str()))
                .unwrap_or_else(|| node.as_str().to_string()),
            _ => return None,
        };
        let predicate_iri = quad.predicate.as_str();
        let predicate = if predicate_iri == format!("{}type", RDF_NS) {
            "type".to_string()
        } else {
            crate::identity::local_name(BASE_NS, predicate_iri)
                .unwrap_or_else(|| predicate_iri.to_string())
        };
        let (object, object_kind) = match &quad.object {
            Term::NamedNode(node) => {
                let iri = node.as_str();
                let name = crate::identity::entity_local_name(BASE_NS, iri)
                    .or_else(|| crate::identity::local_name(BASE_NS, iri))
                    .unwrap_or_else(|| iri.to_string());
                (name, knowledge_fusion::ObjectKind::Entity)
            }
            Term::Literal(literal) => (
                literal.value().to_string(),
                knowledge_fusion::ObjectKind::Literal,
            ),
            _ => return None,
        };
        Some(knowledge_fusion::ScopedAssertion::new(
            namespace,
            &subject,
            &predicate,
            &object,
            object_kind,
            source,
        ))
    }

    /// 从融合命名图读取账本（不触碰缓存，供持锁内部调用）。
    fn fusion_ledger_from_store(
        store: &Store,
    ) -> Result<knowledge_fusion::FusionLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(knowledge_fusion::FUSION_GRAPH)?);
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(knowledge_fusion::parse_fusion_graph(quads.into_iter()))
    }

    /// 全量重写融合账本命名图（幂等；不推进知识版本）。
    fn sync_fusion_ledger_locked(
        &self,
        store: &Store,
        ledger: &knowledge_fusion::FusionLedger,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = knowledge_fusion::fusion_quads(ledger)?;
        let graph = GraphName::NamedNode(NamedNode::new(knowledge_fusion::FUSION_GRAPH)?);
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("fusion_ledger_sync");
        Ok(ledger.assertions.len())
    }

    /// 全量重写融合账本（幂等；默认图与知识版本不变）。
    pub async fn sync_fusion_ledger(
        &self,
        ledger: &knowledge_fusion::FusionLedger,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        self.sync_fusion_ledger_locked(&store, ledger)
    }

    /// 读取融合账本（按 id 排序去重，确定性）。
    pub async fn load_fusion_ledger(
        &self,
    ) -> Result<knowledge_fusion::FusionLedger, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let ledger = Self::fusion_ledger_from_store(&store)?;
        self.perf.bump("fusion_ledger_load");
        Ok(ledger)
    }

    /// 把断言写入命名空间源图，并记入融合账本。
    ///
    /// 源图与账本都是命名图：默认图不变、知识版本不推进，因此现有回答路径
    /// 与缓存不受影响；返回本次新增的源图四元组数（重复导入计 0）。
    pub async fn record_namespace_assertions(
        &self,
        namespace: &knowledge_fusion::NamespaceRecord,
        assertions: &[knowledge_fusion::ScopedAssertion],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let graph = GraphName::NamedNode(NamedNode::new(namespace.graph_iri())?);
        let mut added = 0usize;
        for assertion in assertions {
            let quad = Self::assertion_quad(
                &assertion.subject,
                &assertion.predicate,
                &assertion.object,
                assertion.object_kind,
                graph.clone(),
            )?;
            if store.insert(&quad)? {
                added += 1;
            }
        }
        let mut ledger = Self::fusion_ledger_from_store(&store)?;
        ledger.upsert_namespace(namespace.clone());
        for assertion in assertions {
            ledger.record_assertion(assertion.clone());
        }
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        self.perf.bump("namespace_assertions");
        Ok(added)
    }

    /// 设置命名空间状态（启用 / 隔离 / 退役）；返回是否存在该命名空间。
    ///
    /// 只改融合账本元数据：源图证据与默认图都不动，退役后可随时恢复。
    pub async fn set_namespace_status(
        &self,
        namespace: &str,
        status: knowledge_fusion::NamespaceStatus,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let mut ledger = Self::fusion_ledger_from_store(&store)?;
        let id = knowledge_fusion::sanitize_namespace_id(namespace);
        let Some(record) = ledger.namespaces.iter_mut().find(|record| record.id == id) else {
            return Ok(false);
        };
        record.status = status;
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        Ok(true)
    }

    /// 按策略计算融合计划（只读，不写默认图）。
    pub async fn fusion_plan(
        &self,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<knowledge_fusion::FusionPlan, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        Ok(self.fusion_plan_locked(&store, policy)?.1)
    }

    fn fusion_plan_locked(
        &self,
        store: &Store,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<
        (knowledge_fusion::FusionLedger, knowledge_fusion::FusionPlan),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let ledger = Self::fusion_ledger_from_store(store)?;
        let plan = knowledge_fusion::fusion_plan(
            &ledger.namespaces,
            &ledger.assertions,
            &ledger.adjudications,
            policy,
            |subject, predicate, object, object_kind| match Self::assertion_quad(
                subject,
                predicate,
                object,
                object_kind,
                GraphName::DefaultGraph,
            ) {
                Ok(quad) => store.contains(&quad).unwrap_or(false),
                Err(_) => false,
            },
        );
        Ok((ledger, plan))
    }

    /// 把融合视图物化进默认图：写入赢家、回收被替换的旧赢家。
    ///
    /// 只回收本账本 `applied` 记录过的赢家，绝不触碰未纳入融合的历史数据；
    /// 默认图确有变化才推进知识版本，重复融合幂等。
    pub async fn apply_fusion(
        &self,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<knowledge_fusion::FusionApplyOutcome, Box<dyn std::error::Error + Send + Sync>>
    {
        let store = self.store.lock().await;
        let (mut ledger, plan) = self.fusion_plan_locked(&store, policy)?;
        let winners = plan.materialize_keys();

        let mut removed = 0usize;
        for previous in &ledger.applied {
            if winners.contains(&(
                previous.subject.clone(),
                previous.predicate.clone(),
                previous.object.clone(),
                previous.object_kind,
            )) {
                continue;
            }
            let quad = Self::assertion_quad(
                &previous.subject,
                &previous.predicate,
                &previous.object,
                previous.object_kind,
                GraphName::DefaultGraph,
            )?;
            if store.remove(&quad)? {
                removed += 1;
            }
        }

        let mut inserted = 0usize;
        for assertion in &plan.materialize {
            let quad = Self::assertion_quad(
                &assertion.subject,
                &assertion.predicate,
                &assertion.object,
                assertion.object_kind,
                GraphName::DefaultGraph,
            )?;
            if store.insert(&quad)? {
                inserted += 1;
            }
        }

        ledger.set_applied(
            plan.materialize
                .iter()
                .map(knowledge_fusion::AppliedWinner::from_assertion)
                .collect(),
        );
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        if inserted + removed > 0 {
            self.bump_revision();
        }
        self.perf.bump("fusion_apply");
        Ok(knowledge_fusion::FusionApplyOutcome {
            plan,
            inserted,
            removed,
        })
    }
}

pub fn extract_query_entities(query: &str, config: &SystemConfig) -> Vec<String> {
    let tagged_words = preprocess(query, &config.nlp);
    extract_query_entities_from_tagged(&tagged_words, config)
}

/// 使用已分词结果提取查询实体
pub fn extract_query_entities_from_tagged(
    tagged_words: &[TaggedWord],
    config: &SystemConfig,
) -> Vec<String> {
    let tagged_words = nlp::merge_name_fragments(tagged_words);
    let mut entities = Vec::new();

    for word in &tagged_words {
        if word.pos == "nr"
            || word.pos == "ns"
            || word.pos == "nz"
            || word.pos == "n"
            || word.pos == "nt"
        {
            entities.push(word.text.clone());
        }
    }

    if let Some(synonyms_config) = &config.synonyms {
        for entity in entities.iter_mut() {
            if let Some(normalized) = synonyms_config.mappings.get(entity) {
                *entity = normalized.clone();
            }
        }
    }

    // 去重时保留首次出现顺序
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| e.len() > 1 && seen.insert(e.clone()));
    entities
}

pub struct ResponseFormatter {
    _config: Arc<SystemConfig>,
}

impl ResponseFormatter {
    pub fn new(config: Arc<SystemConfig>) -> Self {
        Self { _config: config }
    }

    /// 取首个结果集中指定变量的绑定值（Prolog 变量名为首字母大写）。
    fn result_value<'a>(
        results: &'a [HashMap<String, String>],
        variable: &str,
    ) -> Option<&'a String> {
        results[0].get(variable)
    }

    pub fn format_prolog_results(
        &self,
        results: &[HashMap<String, String>],
        query_type: &str,
    ) -> String {
        if results.is_empty() {
            return "根据当前知识库，无法回答您的问题。请先导入相关知识。".to_string();
        }

        match query_type {
            "age_query" => {
                if let Some(age) = results[0].get("Age") {
                    format!("年龄是 {} 岁", age)
                } else {
                    "未找到年龄信息".to_string()
                }
            }
            "relation_query" => {
                let relations: Vec<&str> = results
                    .iter()
                    .filter_map(|r| r.get("Relation"))
                    .map(String::as_str)
                    .collect();

                if relations.is_empty() {
                    "未找到关系信息".to_string()
                } else {
                    format!("关系是：{}", relations.join(", "))
                }
            }
            "count_query" => {
                if let Some(count) = results[0].get("Count") {
                    format!("共有 {} 个", count)
                } else {
                    "无法计算数量".to_string()
                }
            }
            "spouse_query" => {
                if let Some(spouse) = results[0].get("Spouse") {
                    format!("配偶是 {}", spouse)
                } else {
                    "未找到配偶信息".to_string()
                }
            }
            "capital_query" => {
                if let Some(capital) = Self::result_value(results, "Capital") {
                    format!("首都是{}", capital)
                } else {
                    "未找到首都信息".to_string()
                }
            }
            "altitude_query" => {
                if let Some(altitude) = Self::result_value(results, "Altitude") {
                    format!("海拔高度是{}米", altitude)
                } else {
                    "未找到海拔信息".to_string()
                }
            }
            "chemical_formula_query" => {
                if let Some(formula) = Self::result_value(results, "Formula") {
                    format!("化学分子式是{}", formula)
                } else {
                    "未找到分子式信息".to_string()
                }
            }
            "author_query" => {
                if let Some(author) = Self::result_value(results, "Author") {
                    format!("作者是{}", author)
                } else {
                    "未找到作者信息".to_string()
                }
            }
            "champion_query" => {
                if let Some(champion) = Self::result_value(results, "Champion") {
                    format!("冠军是{}", champion)
                } else {
                    "未找到冠军信息".to_string()
                }
            }
            "birth_year_query" => {
                if let Some(year) = Self::result_value(results, "Year") {
                    format!("诞生于{}年", year)
                } else {
                    "未找到诞生年份信息".to_string()
                }
            }
            "largest_query" => {
                if let Some(entity) = Self::result_value(results, "Entity") {
                    format!("最大的是{}", entity)
                } else {
                    "未找到相关信息".to_string()
                }
            }
            "quantity_query" => {
                if let Some(quantity) = Self::result_value(results, "Quantity") {
                    format!("数量是{}", quantity)
                } else {
                    "未找到数量信息".to_string()
                }
            }
            "concept_query" | "explanation_query" => {
                if let Some(definition) = results[0].get("definition") {
                    format!("{}", definition)
                } else {
                    "未找到定义信息".to_string()
                }
            }
            "time_before_query" => {
                if let Some(year) = results[0].get("year") {
                    format!("时间是在{}年之前", year)
                } else {
                    "未找到时间信息".to_string()
                }
            }
            "principle_query" => {
                if let Some(principle) = results[0].get("principle") {
                    format!("工作原理是{}", principle)
                } else {
                    "未找到原理信息".to_string()
                }
            }
            "comparison_query" => {
                if let Some(difference) = results[0].get("difference") {
                    format!("区别是{}", difference)
                } else {
                    "未找到比较信息".to_string()
                }
            }
            "philosophy_query" => {
                if let Some(philosophy) = results[0].get("philosophy") {
                    format!("哲学思想是{}", philosophy)
                } else {
                    "未找到哲学思想信息".to_string()
                }
            }
            "impact_query" => {
                if let Some(impact) = results[0].get("impact") {
                    format!("影响是{}", impact)
                } else {
                    "未找到影响信息".to_string()
                }
            }
            "cause_query" | "reason_query" => {
                if let Some(cause) = results[0].get("cause").or_else(|| results[0].get("reason")) {
                    format!("原因是{}", cause)
                } else {
                    "未找到原因信息".to_string()
                }
            }
            "population_query" => {
                if let Some(population) = results[0].get("Population") {
                    format!("人口是 {} 人", population)
                } else {
                    "未找到人口信息".to_string()
                }
            }
            "planet_query" => {
                if let Some(name) = results[0].get("name") {
                    format!("最大的行星是{}", name)
                } else {
                    "未找到行星信息".to_string()
                }
            }
            _ => "根据现有信息，找到了相关内容，但无法生成详细回答。".to_string(),
        }
    }
}

pub async fn load_config() -> Result<SystemConfig, Box<dyn std::error::Error + Send + Sync>> {
    let config_content = fs::read_to_string("config.toml")?;
    let config: SystemConfig = toml::from_str(&config_content)?;
    Ok(config)
}

pub fn detect_input_mode(input: &str, keywords: &[String]) -> InputMode {
    if keywords.iter().any(|keyword| input.contains(keyword)) {
        InputMode::KnowledgeImport
    } else {
        InputMode::Query
    }
}

pub fn create_prolog_rules(config: &SystemConfig) -> Vec<PrologRule> {
    config
        .prolog_rules
        .iter()
        .map(|rule_cfg| PrologRule {
            pattern: pattern_from_strings(rule_cfg.pattern.clone()),
            query_type: rule_cfg.query_type.clone(),
            priority: rule_cfg.priority,
        })
        .collect()
}

pub fn create_default_synonyms() -> HashMap<String, String> {
    let mut synonyms = HashMap::new();

    // 基础同义词映射
    synonyms.insert("老婆".to_string(), "妻子".to_string());
    synonyms.insert("老公".to_string(), "丈夫".to_string());

    synonyms
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::perf_ledger::PerfLedger;

    /// 有界 LRU：版本失效整体清空、命中触碰、溢出淘汰最久未用
    #[test]
    fn revision_lru_invalidates_touches_and_evicts() {
        let mut cache: RevisionLru<u64> = RevisionLru::new(2);
        assert!(cache.get(1, "a").is_none(), "空缓存不命中");
        assert_eq!(cache.put(1, "a".to_string(), 1), 0);
        assert_eq!(cache.put(1, "b".to_string(), 2), 0);
        assert_eq!(cache.get(1, "a"), Some(&1));
        // 触碰 a 后，写入 c 应淘汰 b
        assert_eq!(cache.put(1, "c".to_string(), 3), 1);
        assert_eq!(cache.get(1, "b"), None);
        assert_eq!(cache.get(1, "a"), Some(&1));
        assert_eq!(cache.get(1, "c"), Some(&3));
        // 版本推进：全部失效，原键不得命中
        assert!(cache.get(2, "a").is_none());
        assert_eq!(cache.put(2, "d".to_string(), 4), 0);
        assert_eq!(cache.len(), 1, "版本变化时先清空");
        assert_eq!(cache.get(2, "d"), Some(&4));
    }

    /// 多槽直答缓存：交替询问两个同版本问题，第二次都可命中
    #[tokio::test]
    async fn direct_answer_cache_keeps_multiple_keys_per_revision() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![
                KnowledgeItem::Entity {
                    label: "City".to_string(),
                    properties: HashMap::from([
                        ("name".to_string(), "北京".to_string()),
                        ("capital".to_string(), "中国".to_string()),
                    ]),
                },
                KnowledgeItem::Entity {
                    label: "Person".to_string(),
                    properties: HashMap::from([("name".to_string(), "张三".to_string())]),
                },
            ])
            .await
            .unwrap();
        let perf = manager.perf_ledger();
        let entities = vec!["北京".to_string()];

        let first = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(first.as_deref(), Some("北京的首都是中国"));
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        let queries = perf.count("direct_answer_query");

        // 两个键都在：来回询问均命中缓存
        assert_eq!(
            manager.get_direct_answer("capital_query", &entities).await,
            first
        );
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        assert_eq!(perf.count("direct_answer_query"), queries);
        assert_eq!(perf.count("direct_answer_cache_hit"), 2);
    }

    /// 多槽相关事实缓存：同一版本交替询问两个实体，第二次都命中
    #[tokio::test]
    async fn relevant_facts_cache_keeps_multiple_keys_per_revision() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();

        let _ = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        let _ = manager
            .get_relevant_facts(&["李四".to_string()], 2)
            .await
            .unwrap();
        let scans = perf.count("relevant_facts_graph_scan");

        let second_zhang = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(!second_zhang.is_empty());
        let second_li = manager
            .get_relevant_facts(&["李四".to_string()], 2)
            .await
            .unwrap();
        assert!(!second_li.is_empty());
        assert_eq!(
            perf.count("relevant_facts_graph_scan"),
            scans,
            "两个键同版本都应在缓存中"
        );
        assert_eq!(perf.count("relevant_facts_cache_hit"), 2);
    }

    async fn temp_manager() -> (OxigraphManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = OxigraphManager::with_perf_ledger(
            dir.path().to_str().unwrap(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        (manager, dir)
    }

    fn items(first: &str, second: &str) -> Vec<KnowledgeItem> {
        vec![KnowledgeItem::Relationship {
            from_label: "人物".to_string(),
            from_id: first.to_string(),
            to_label: "人物".to_string(),
            to_id: second.to_string(),
            rel_type: "师傅".to_string(),
            properties: HashMap::new(),
        }]
    }

    fn person(name: &str) -> KnowledgeItem {
        KnowledgeItem::Entity {
            label: "Person".to_string(),
            properties: HashMap::from([("name".to_string(), name.to_string())]),
        }
    }

    /// 身份 URI 往返：名字含 `/`、空格时 IRI 合法，且图谱键能无损反解（编码/解码往返律）。
    #[tokio::test]
    async fn entity_uri_roundtrip_handles_awkward_names() {
        let (manager, _dir) = temp_manager().await;
        let name = "北京/大学 附中";
        manager.save_knowledge(vec![person(name)]).await.unwrap();
        let facts = manager.load_all_facts().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Entity { name: fact_name, .. } if fact_name == name
            )),
            "图谱键应无损往返：{:?}",
            facts
        );
        let iri = crate::identity::entity_uri(crate::identity::DEFAULT_BASE_NS, name);
        assert!(!iri.contains(' '), "URI 不允许空格：{}", iri);
        assert_eq!(
            crate::identity::entity_local_name(crate::identity::DEFAULT_BASE_NS, &iri).as_deref(),
            Some(name)
        );
    }

    /// 版本键查询缓存：同版本重复问同一实体命中缓存；写入后自动失效。
    #[tokio::test]
    async fn relevant_facts_cache_is_revision_keyed() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();

        let first = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(!first.is_empty());
        let scans_after_first = perf.count("relevant_facts_graph_scan");
        let misses_after_first = perf.count("relevant_facts_cache_miss");

        // 同一版本、同一实体（顺序不同也不影响）：命中缓存
        let second = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(perf.count("relevant_facts_graph_scan"), scans_after_first);
        assert_eq!(perf.count("relevant_facts_cache_miss"), misses_after_first);
        assert_eq!(perf.count("relevant_facts_cache_hit"), 1);

        // 写入新关系 → 版本变化 → 必须重新扫描并看到新数据
        manager.save_knowledge(items("李四", "王五")).await.unwrap();
        let third = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(perf.count("relevant_facts_graph_scan") > scans_after_first);
        let relations: Vec<&str> = third
            .iter()
            .filter_map(|fact| match fact {
                PrologFact::Relation { rel_type, .. } => Some(rel_type.as_str()),
                _ => None,
            })
            .collect();
        assert!(relations.contains(&"师傅"));
    }

    /// 重复导入不推进知识版本。
    #[tokio::test]
    async fn duplicate_import_does_not_bump_revision() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();
        let bumps = perf.count("revision_bump");

        let replay = manager
            .save_knowledge_with_receipt(items("张三", "李四"))
            .await
            .unwrap();
        assert_eq!(replay.added, 0);
        assert_eq!(perf.count("revision_bump"), bumps, "纯重复导入不应推进版本");

        let _ = manager.load_graph_snapshot().await.unwrap();
        let scans = perf.count("graph_snapshot_scan");
        let _ = manager.load_graph_snapshot().await.unwrap();
        assert_eq!(
            perf.count("graph_snapshot_scan"),
            scans,
            "同版本快照应命中缓存"
        );
        assert!(perf.count("graph_snapshot_cache_hit") >= 1);
    }

    /// 直答缓存：同版本命中（含负缓存）；写入后重查看到新值。
    #[tokio::test]
    async fn direct_answer_cache_is_revision_keyed() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![KnowledgeItem::Entity {
                label: "City".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "北京".to_string()),
                    ("capital".to_string(), "中国".to_string()),
                ]),
            }])
            .await
            .unwrap();
        let perf = manager.perf_ledger();
        let entities = vec!["北京".to_string()];

        let first = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(first.as_deref(), Some("北京的首都是中国"));
        let queries = perf.count("direct_answer_query");

        // 同版本同问句：命中缓存
        let second = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(second, first);
        assert_eq!(perf.count("direct_answer_query"), queries);
        assert_eq!(perf.count("direct_answer_cache_hit"), 1);

        // 负缓存：同版本问不存在的属性同样命中缓存
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        let queries_with_negative = perf.count("direct_answer_query");
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        assert_eq!(perf.count("direct_answer_query"), queries_with_negative);

        // 写入后版本变化 → 缓存失效并重查
        manager.save_knowledge(vec![person("李四")]).await.unwrap();
        let third = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(third.as_deref(), Some("北京的首都是中国"));
        assert!(perf.count("direct_answer_query") > queries_with_negative);
    }

    fn ledger_var(name: &str) -> graph_ledger::LedgerTerm {
        graph_ledger::LedgerTerm {
            name: name.to_string(),
            is_var: true,
        }
    }

    fn ledger_const(name: &str) -> graph_ledger::LedgerTerm {
        graph_ledger::LedgerTerm {
            name: name.to_string(),
            is_var: false,
        }
    }

    fn ledger_rule_entry() -> graph_ledger::RuleLedgerEntry {
        let head = graph_ledger::LedgerAtom {
            predicate: "老师".to_string(),
            terms: vec![ledger_var("X"), ledger_var("Y")],
        };
        let body = vec![graph_ledger::LedgerAtom {
            predicate: "师傅".to_string(),
            terms: vec![ledger_var("X"), ledger_var("Y")],
        }];
        graph_ledger::RuleLedgerEntry {
            id: graph_ledger::rule_entry_id(&head, &body),
            head,
            body,
            confidence: 0.95,
            source: "中文规则".to_string(),
        }
    }

    fn ledger_derived_entry(revision: u64, fingerprint: &str) -> graph_ledger::DerivedLedgerEntry {
        let atom = graph_ledger::LedgerAtom {
            predicate: "老师".to_string(),
            terms: vec![ledger_const("张三"), ledger_const("李四")],
        };
        graph_ledger::DerivedLedgerEntry {
            id: graph_ledger::derived_entry_id(&atom),
            atom,
            rule_source: "中文规则".to_string(),
            premises: vec![graph_ledger::LedgerAtom {
                predicate: "师傅".to_string(),
                terms: vec![ledger_const("张三"), ledger_const("李四")],
            }],
            depth: 1,
            confidence: 0.95,
            revision,
            fingerprint: fingerprint.to_string(),
        }
    }

    /// 图账本与知识图谱隔离：账本写入不进快照/谓词画像，也不推进知识版本。
    #[tokio::test]
    async fn ledger_graphs_are_isolated_from_knowledge_snapshot() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let revision = manager.revision();
        let snapshot_before = manager.load_graph_snapshot().await.unwrap();

        let rule = ledger_rule_entry();
        let fingerprint = graph_ledger::rule_fingerprint(&[rule.clone()]);
        manager.sync_rule_ledger(&[rule]).await.unwrap();
        manager
            .materialize_derived_ledger(
                &[ledger_derived_entry(revision, &fingerprint)],
                revision,
                &fingerprint,
            )
            .await
            .unwrap();

        assert_eq!(manager.revision(), revision, "账本写入不推进知识版本");
        let snapshot_after = manager.load_graph_snapshot().await.unwrap();
        assert_eq!(
            snapshot_before.as_ref(),
            snapshot_after.as_ref(),
            "命名图里的账本四元组不得进入知识快照"
        );

        manager.refresh_predicate_registry().await.unwrap();
        let report = manager.predicate_registry_report();
        assert!(
            !report.contains("ledger"),
            "谓词画像不得观测账本谓词：{}",
            report
        );
    }

    /// 规则账本经存储往返一致；重复同步幂等；空集同步清空账本。
    #[tokio::test]
    async fn rule_ledger_store_roundtrip_and_idempotent_resync() {
        let (manager, _dir) = temp_manager().await;
        let entry = ledger_rule_entry();
        assert_eq!(manager.sync_rule_ledger(&[entry.clone()]).await.unwrap(), 1);

        let loaded = manager.load_rule_ledger().await.unwrap();
        assert!(loaded.synced);
        assert_eq!(loaded.entries, vec![entry.clone()]);
        assert_eq!(
            loaded.fingerprint,
            graph_ledger::rule_fingerprint(&[entry.clone()])
        );

        manager.sync_rule_ledger(&[entry.clone()]).await.unwrap();
        assert_eq!(
            manager.load_rule_ledger().await.unwrap().entries,
            vec![entry.clone()],
            "同内容重复同步应幂等"
        );

        manager.sync_rule_ledger(&[]).await.unwrap();
        let cleared = manager.load_rule_ledger().await.unwrap();
        assert!(cleared.synced);
        assert!(cleared.entries.is_empty(), "空集同步必须清空账本");
    }

    /// 模式账本经存储往返一致；重复同步幂等；空集同步清空账本；不推进知识版本。
    #[tokio::test]
    async fn pattern_ledger_store_roundtrip_and_idempotent_resync() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let revision = manager.revision();

        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let mut library = crate::pattern_ir::PatternLibrary::from_config(&config);
        assert!(!library.is_empty());
        library
            .teach_with_guard(
                "省略身高",
                "身高 呢",
                "height_query",
                30,
                Some(crate::pattern_ir::ContextGuard::focus()),
            )
            .expect("守卫模式应可教学");
        // 落图读回按 subject（= id）排序，与库内配置序无关 → 比较前统一按 id 排
        let by_id = |entries: &[crate::pattern_ir::PatternEntry]| {
            let mut v = entries.to_vec();
            v.sort_by(|a, b| a.id().cmp(&b.id()));
            v
        };

        assert_eq!(
            manager
                .sync_pattern_ledger(library.entries())
                .await
                .unwrap(),
            library.len()
        );
        let loaded = manager.load_pattern_ledger().await.unwrap();
        assert_eq!(
            by_id(&loaded),
            by_id(library.entries()),
            "模式账本往返应逐条一致（与顺序无关）"
        );
        assert_eq!(
            loaded
                .iter()
                .find(|entry| entry.name == "省略身高")
                .and_then(|entry| entry.guard.as_ref()),
            Some(&crate::pattern_ir::ContextGuard::focus()),
            "上下文守卫必须随模式账本持久化"
        );
        assert_eq!(manager.revision(), revision, "模式账本写入不推进知识版本");

        manager
            .sync_pattern_ledger(library.entries())
            .await
            .unwrap();
        assert_eq!(
            by_id(&manager.load_pattern_ledger().await.unwrap()),
            by_id(library.entries()),
            "同内容重复同步应幂等"
        );

        manager.sync_pattern_ledger(&[]).await.unwrap();
        let cleared = manager.load_pattern_ledger().await.unwrap();
        assert!(cleared.is_empty(), "空集同步必须清空模式账本");
    }

    /// 模式经验账本经存储往返一致；经验与模式定义分图；空集同步清空。
    #[tokio::test]
    async fn pattern_episode_ledger_store_roundtrip_and_isolated_graph() {
        let (manager, _dir) = temp_manager().await;
        let revision = manager.revision();
        let tokens = vec![crate::nlp::TaggedWord {
            text: "法国".to_string(),
            pos: "ns".to_string(),
        }];
        let mut log = crate::pattern_ir::PatternEpisodeLog::with_capacity(8);
        log.record_unmatched("法国的首都是哪里", &tokens, Some("法国"), 3);
        log.record_unmatched("这是另一句", &tokens, None, 4);

        assert_eq!(
            manager.sync_pattern_episodes(log.entries()).await.unwrap(),
            2
        );
        let loaded = manager.load_pattern_episodes().await.unwrap();
        let mut expected: Vec<_> = log.entries().to_vec();
        expected.sort_by_key(crate::pattern_ir::PatternEpisode::id);
        assert_eq!(loaded, expected, "经验账本往返应逐条一致");
        assert_eq!(manager.revision(), revision, "经验写入不推进知识版本");

        // 模式定义图不受经验图影响
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let library = crate::pattern_ir::PatternLibrary::from_config(&config);
        manager
            .sync_pattern_ledger(library.entries())
            .await
            .unwrap();
        assert_eq!(
            manager.load_pattern_ledger().await.unwrap().len(),
            library.len()
        );
        assert_eq!(manager.load_pattern_episodes().await.unwrap().len(), 2);

        manager.sync_pattern_episodes(&[]).await.unwrap();
        assert!(
            manager.load_pattern_episodes().await.unwrap().is_empty(),
            "空集同步必须清空经验账本"
        );
        assert_eq!(
            manager.load_pattern_ledger().await.unwrap().len(),
            library.len(),
            "清空经验不得影响模式定义"
        );
    }

    /// 派生账本经存储往返一致，物化戳（版本 + 指纹）可审计；重物化全量替换。
    #[tokio::test]
    async fn derived_ledger_store_roundtrip_and_stamp() {
        let (manager, _dir) = temp_manager().await;
        let entry = ledger_derived_entry(5, "fp-one");
        assert_eq!(
            manager
                .materialize_derived_ledger(&[entry.clone()], 5, "fp-one")
                .await
                .unwrap(),
            1
        );
        let loaded = manager.load_derived_ledger().await.unwrap();
        assert!(loaded.materialized);
        assert_eq!(loaded.revision, 5);
        assert_eq!(loaded.fingerprint, "fp-one");
        assert_eq!(loaded.entries, vec![entry.clone()]);

        let replacement = ledger_derived_entry(6, "fp-two");
        manager
            .materialize_derived_ledger(&[replacement.clone()], 6, "fp-two")
            .await
            .unwrap();
        let reloaded = manager.load_derived_ledger().await.unwrap();
        assert_eq!(reloaded.entries, vec![replacement], "重物化应全量替换");
        assert_eq!(reloaded.revision, 6);
        assert_eq!(reloaded.fingerprint, "fp-two");
    }

    /// RDF 格式识别：常见扩展名映射到对应解析器；JSON-LD 与未知扩展名给出可操作错误。
    #[test]
    fn detect_rdf_format_maps_extensions_and_rejects_jsonld() {
        use oxigraph::io::RdfFormat;
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.ttl").unwrap(),
            RdfFormat::Turtle
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("KB.TTL").unwrap(),
            RdfFormat::Turtle,
            "扩展名大小写不敏感"
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.nt").unwrap(),
            RdfFormat::NTriples
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.nq").unwrap(),
            RdfFormat::NQuads
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.trig").unwrap(),
            RdfFormat::TriG
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.rdf").unwrap(),
            RdfFormat::RdfXml
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.xml").unwrap(),
            RdfFormat::RdfXml
        );

        let jsonld_error = OxigraphManager::detect_rdf_format("kb.jsonld")
            .unwrap_err()
            .to_string();
        assert!(jsonld_error.contains("import-jsonld"), "{}", jsonld_error);
        let unknown_error = OxigraphManager::detect_rdf_format("kb.bin")
            .unwrap_err()
            .to_string();
        assert!(unknown_error.contains("ttl"), "{}", unknown_error);
        assert!(
            OxigraphManager::detect_rdf_format("kb").is_err(),
            "无扩展名应报错"
        );
    }

    /// RDF 文件导入：新增三元组计数正确、重复导入幂等、数据进入知识快照。
    #[tokio::test]
    async fn import_rdf_file_counts_new_triples_and_feeds_snapshot() {
        let (manager, dir) = temp_manager().await;
        let ttl_path = dir.path().join("sample.ttl");
        fs::write(
            &ttl_path,
            r#"@prefix kg: <http://example.org/kg/> .
<http://example.org/kg/entity/王五> a kg:Person ;
    kg:name "王五" ;
    kg:age "20" .
<http://example.org/kg/entity/王五> kg:师父 <http://example.org/kg/entity/赵六> .
"#,
        )
        .unwrap();

        let added = manager
            .import_rdf_file(ttl_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(added, 4, "应报告 4 条新增三元组");

        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Entity { name, properties, .. }
                    if name == "王五" && properties.get("age").map(String::as_str) == Some("20")
            )),
            "导入的实体应进入知识快照：{:?}",
            facts
        );

        let revision_after_first = manager.revision();
        let replay = manager
            .import_rdf_file(ttl_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(replay, 0, "重复导入不应新增三元组");
        assert!(
            manager.revision() >= revision_after_first,
            "导入后知识版本推进（严格正版本号）"
        );

        let nt_path = dir.path().join("extra.nt");
        fs::write(
            &nt_path,
            "<http://example.org/kg/entity/赵六> <http://example.org/kg/name> \"赵六\" .\n",
        )
        .unwrap();
        assert_eq!(
            manager
                .import_rdf_file(nt_path.to_str().unwrap())
                .await
                .unwrap(),
            1
        );
    }

    /// RDF/XML 经 reader 导入同样落入默认图。
    #[tokio::test]
    async fn import_rdf_reader_accepts_rdfxml() {
        let (manager, _dir) = temp_manager().await;
        let rdfxml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:kg="http://example.org/kg/">
  <rdf:Description rdf:about="http://example.org/kg/entity/孙七">
    <rdf:type rdf:resource="http://example.org/kg/Person"/>
    <kg:name>孙七</kg:name>
  </rdf:Description>
</rdf:RDF>
"#;
        let added = manager
            .import_rdf_reader(oxigraph::io::RdfFormat::RdfXml, rdfxml.as_bytes())
            .await
            .unwrap();
        assert_eq!(added, 2);
        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(facts.iter().any(|fact| matches!(
            fact,
            PrologFact::Entity { name, .. } if name == "孙七"
        )));
    }

    /// 实体身份注册表：保存即观测（name 别名自动确认）、显式裁决、落盘往返。
    #[tokio::test]
    async fn entity_registry_observes_saves_and_roundtrips() {
        let (manager, dir) = temp_manager().await;
        let mut knowledge = vec![person("张三"), person("李四")];
        knowledge.extend(items("张三", "李四"));
        manager.save_knowledge(knowledge).await.unwrap();

        let facts = manager.load_all_facts().await.unwrap();
        assert!(manager.observe_entity_facts(&facts) > 0, "首次观测应有变化");
        assert_eq!(
            manager.resolve_entity_identity("张三"),
            crate::entity_registry::EntityResolution::Canonical("张三".to_string())
        );
        assert!(matches!(
            manager.entity_mint_check("张三"),
            crate::entity_registry::MintVerdict::Exists { .. }
        ));

        // 显式确认别名 → 链接器可见；重复确认无变化
        assert!(manager.confirm_entity_alias("老张", "张三"));
        assert!(!manager.confirm_entity_alias("老张", "张三"));
        assert_eq!(
            manager.resolve_entity_identity("老张"),
            crate::entity_registry::EntityResolution::Canonical("张三".to_string())
        );
        assert_eq!(
            manager.confirmed_entity_aliases(),
            vec![("老张".to_string(), "张三".to_string())]
        );

        // 落盘 → 新实例加载逐字节一致（规范 JSON）；库目录另起（RocksDB 单写锁）
        let path = dir.path().join("entity_registry.json");
        let path_text = path.to_str().unwrap();
        manager.save_entity_registry(path_text).unwrap();
        let (restored, _other_dir) = temp_manager().await;
        assert!(restored.load_entity_registry(path_text));
        assert_eq!(
            restored.entity_registry_snapshot().to_json(),
            manager.entity_registry_snapshot().to_json()
        );

        // 提名 → 拒绝：墓碑阻止解析；合并：改名后重定向
        let nominations =
            manager.nominate_entity_aliases(&[crate::entity_registry::AliasCandidate {
                surface: "小张".to_string(),
                canonical: "张三".to_string(),
                pattern: "简称".to_string(),
                evidence: "测试".to_string(),
            }]);
        assert_eq!(
            nominations[0].kind,
            crate::entity_registry::NominationKind::Pending
        );
        assert!(manager.reject_entity_alias("小张", "张三"));
        assert_eq!(
            manager.resolve_entity_identity("小张"),
            crate::entity_registry::EntityResolution::Unknown
        );
        let merged = manager.merge_entity_identities("张三", "张老师").unwrap();
        assert_eq!(merged.into, "张老师");
        assert_eq!(
            manager.resolve_entity_identity("张三"),
            crate::entity_registry::EntityResolution::Redirected {
                from: "张三".to_string(),
                to: "张老师".to_string(),
            }
        );
        let perf = manager.perf_ledger();
        assert!(perf.count("identity_alias_confirm") >= 1);
        assert_eq!(perf.count("identity_merge"), 1);
    }

    /// JSON-LD 导入统计：实体数 / 命名节点关系数 / 实际新增三元组数；
    /// 重复导入新增为 0（与 RDF 导入一致的确定性口径）。
    #[tokio::test]
    async fn import_jsonld_reports_entity_relation_and_triple_counts() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "teacher_of": "http://example.org/kg/teacher_of",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/小明",
                    "@type": "Person",
                    "name": "小明"
                },
                {
                    "@id": "http://example.org/kg/entity/张三",
                    "@type": "Person",
                    "name": "张三",
                    "teacher_of": { "@id": "http://example.org/kg/entity/李四" }
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "两个节点");
        assert_eq!(stats.relations, 1, "一条指向命名节点的关系边");
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 关系 1");
        assert_eq!(
            stats.summary(),
            "解析了 2 个实体和 1 个关系，新增 5 条三元组。"
        );

        let replay = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(replay.entities, 2);
        assert_eq!(replay.relations, 1);
        assert_eq!(replay.triples, 0, "重复导入不新增三元组");
    }

    /// JSON-LD：前缀/对象上下文、@base 相对 @id，以及带属性的嵌套节点都要落库。
    #[tokio::test]
    async fn import_jsonld_expands_context_and_nested_nodes() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "Person": { "@id": "ex:Person" },
                "@base": "http://example.org/kg/entity/",
                "name": "ex:name",
                "friend": { "@id": "ex:friend", "@type": "@id" }
            },
            "@graph": [
                {
                    "@id": "张三",
                    "@type": "Person",
                    "name": "张三",
                    "friend": {
                        "@id": "李四",
                        "@type": "Person",
                        "name": "李四"
                    }
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "嵌套命名节点也应计为实体");
        assert_eq!(stats.relations, 1);
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 好友关系 1");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?friendName WHERE {
                    <http://example.org/kg/entity/张三> kg:friend ?friend .
                    ?friend kg:name ?friendName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(
            result.data[0].get("friendName"),
            Some(&Value::String("李四".to_string()))
        );
    }

    /// JSON-LD 空白节点：`_:label` 作为主语与宾语都落库，重复导入保持幂等。
    #[tokio::test]
    async fn import_jsonld_supports_blank_node_ids_and_references() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "knows": "http://example.org/kg/knows",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "_:a",
                    "@type": "Person",
                    "name": "匿名甲",
                    "knows": { "@id": "_:b" }
                },
                {
                    "@id": "_:b",
                    "@type": "Person",
                    "name": "匿名乙"
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.relations, 1);
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 关系 1");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?subject ?targetName WHERE {
                    ?subject kg:name "匿名甲" ;
                             kg:knows ?target .
                    ?target kg:name ?targetName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(
            result.data[0].get("subject"),
            Some(&Value::String("_:a".to_string())),
            "空白节点应保留 `_:` 形式：{:?}",
            result.data
        );
        assert_eq!(
            result.data[0].get("targetName"),
            Some(&Value::String("匿名乙".to_string()))
        );

        let replay = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(replay.triples, 0, "重复导入不应新增三元组");
    }

    /// JSON-LD @reverse 与 @set：反向边方向正确，集合值逐项展开。
    #[tokio::test]
    async fn import_jsonld_supports_reverse_and_set_values() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "nick": "http://example.org/kg/nick",
                "parent_of": "http://example.org/kg/parent_of",
                "knows": "http://example.org/kg/knows"
            },
            "@id": "http://example.org/kg/entity/甲",
            "name": "甲",
            "nick": { "@set": ["阿甲", "甲哥"] },
            "@reverse": {
                "parent_of": { "@id": "http://example.org/kg/entity/乙" },
                "knows": {
                    "@id": "http://example.org/kg/entity/丙",
                    "name": "丙"
                }
            }
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "甲与带属性的反向节点丙");
        assert_eq!(stats.relations, 2, "两条反向边");
        assert_eq!(stats.triples, 6, "name 2 + nick 2 + 反向边 2");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                    FILTER(?p IN (kg:parent_of, kg:knows))
                }"#,
            )
            .await
            .unwrap();
        let pairs: Vec<(String, String)> = result
            .data
            .iter()
            .map(|row| {
                (
                    match row.get("s") {
                        Some(Value::String(text)) => text.clone(),
                        other => format!("{:?}", other),
                    },
                    match row.get("o") {
                        Some(Value::String(text)) => text.clone(),
                        other => format!("{:?}", other),
                    },
                )
            })
            .collect();
        assert!(
            pairs.contains(&(
                "http://example.org/kg/entity/乙".to_string(),
                "http://example.org/kg/entity/甲".to_string()
            )),
            "反向边应为 乙 parent_of 甲：{:?}",
            pairs
        );
        assert!(
            pairs.contains(&(
                "http://example.org/kg/entity/丙".to_string(),
                "http://example.org/kg/entity/甲".to_string()
            )),
            "反向边应为 丙 knows 甲：{:?}",
            pairs
        );

        let nicks = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?nick WHERE {
                    <http://example.org/kg/entity/甲> kg:nick ?nick
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(nicks.data.len(), 2, "{:?}", nicks.data);
        assert_eq!(
            manager
                .import_jsonld(jsonld)
                .await
                .unwrap()
                .triples,
            0,
            "重复导入保持幂等"
        );
    }

    /// JSON-LD @list：编码为 rdf:first/rdf:rest 链，空表指向 rdf:nil，重复导入幂等。
    #[tokio::test]
    async fn import_jsonld_supports_rdf_lists() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "items": "http://example.org/kg/items",
                "friend": "http://example.org/kg/friend"
            },
            "@id": "http://example.org/kg/entity/甲",
            "items": { "@list": ["一", "二"] },
            "friend": { "@list": [] }
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.triples, 6, "非空表 1+4 条，空表 1 条");

        let firsts = manager
            .query_sparql(
                r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?value WHERE { ?cell rdf:first ?value }"#,
            )
            .await
            .unwrap();
        let mut values: Vec<String> = firsts
            .data
            .iter()
            .filter_map(|row| match row.get("value") {
                Some(Value::String(text)) => Some(text.clone()),
                _ => None,
            })
            .collect();
        values.sort();
        assert_eq!(values, vec!["一".to_string(), "二".to_string()]);

        let empty = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?tail WHERE {
                    <http://example.org/kg/entity/甲> kg:friend ?tail .
                    FILTER(?tail = rdf:nil)
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(empty.data.len(), 1, "{:?}", empty.data);

        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "集合单元格按内容命名，重复导入不应新增三元组"
        );
    }

    /// 嵌套 @graph 与 @included：内容展开到默认图，纯图容器不生成实体。
    #[tokio::test]
    async fn import_jsonld_supports_nested_graphs_and_included() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": { "name": "http://example.org/kg/name" },
            "@graph": [
                {
                    "@id": "http://example.org/kg/graph/outer",
                    "@graph": [
                        { "@id": "http://example.org/kg/entity/甲", "name": "甲" }
                    ]
                },
                {
                    "@id": "http://example.org/kg/entity/乙",
                    "name": "乙",
                    "@included": [
                        { "@id": "http://example.org/kg/entity/丙", "name": "丙" }
                    ]
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 3, "纯图容器不计实体，只统计甲乙丙");
        assert_eq!(stats.triples, 3);

        let names = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?name WHERE { ?entity kg:name ?name }"#,
            )
            .await
            .unwrap();
        assert_eq!(names.data.len(), 3, "{:?}", names.data);
        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "重复导入保持幂等"
        );
    }

    /// 节点级 @context 只作用于该节点：嵌套节点可用不同词项映射。
    #[tokio::test]
    async fn import_jsonld_respects_scoped_context() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "name": "ex:name"
            },
            "@id": "http://example.org/kg/entity/甲",
            "ex:parent": {
                "@context": { "name": "http://example.org/kg/childName" },
                "@id": "http://example.org/kg/entity/乙",
                "name": "乙"
            }
        }"#;

        manager.import_jsonld(jsonld).await.unwrap();
        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?childName WHERE {
                    <http://example.org/kg/entity/乙> kg:childName ?childName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(
            result.data[0].get("childName"),
            Some(&Value::String("乙".to_string()))
        );
    }

    /// 任意 SPARQL SELECT：变量绑定转 JSON，带类型字面量映射为数字/布尔。
    #[tokio::test]
    async fn sparql_select_maps_typed_literals_to_json() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "age": "http://example.org/kg/age",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/小明",
                    "@type": "Person",
                    "name": "小明",
                    "age": {"@value": "18", "@type": "http://www.w3.org/2001/XMLSchema#integer"}
                }
            ]
        }"#;
        manager.import_jsonld(jsonld).await.unwrap();

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?name ?age WHERE { ?s kg:name ?name . ?s kg:age ?age }"#,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.data.len(), 1);
        assert_eq!(
            result.data[0].get("name"),
            Some(&Value::String("小明".to_string()))
        );
        assert_eq!(result.data[0].get("age"), Some(&serde_json::json!(18)));
        let rendered = result.render_text();
        assert!(
            rendered.contains("?age") || rendered.contains("age"),
            "{}",
            rendered
        );
        assert!(rendered.contains("小明"), "{}", rendered);
        assert!(rendered.contains("18"), "{}", rendered);
    }

    /// ASK 返回确定布尔；CONSTRUCT 按三元组逐行返回；语法错误不静默为空。
    #[tokio::test]
    async fn sparql_ask_construct_and_syntax_error() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(vec![person("张三")]).await.unwrap();

        let ask = manager
            .query_sparql(r#"ASK { ?s <http://example.org/kg/name> "张三" }"#)
            .await
            .unwrap();
        assert_eq!(ask.data.len(), 1);
        assert_eq!(ask.data[0].get("boolean"), Some(&Value::Bool(true)));

        let construct = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                CONSTRUCT { ?s kg:name ?name } WHERE { ?s kg:name ?name }"#,
            )
            .await
            .unwrap();
        assert!(!construct.data.is_empty());
        for row in &construct.data {
            assert!(row.contains_key("subject"), "{:?}", row);
            assert!(row.contains_key("predicate"), "{:?}", row);
            assert!(row.contains_key("object"), "{:?}", row);
        }

        let error = manager.query_sparql("SELECT ?s WHERE { ?s ?p").await;
        assert!(error.is_err());
        let message = error.unwrap_err().to_string();
        assert!(message.contains("SPARQL 语法错误"), "{}", message);
    }

    /// 无结果的 SELECT 与空表渲染：固定措辞，不伪装成故障。
    #[tokio::test]
    async fn sparql_empty_result_and_render_are_explicit() {
        let (manager, _dir) = temp_manager().await;
        let result = manager
            .query_sparql(r#"SELECT ?s WHERE { ?s <http://example.org/kg/name> "不存在" }"#)
            .await
            .unwrap();
        assert!(result.data.is_empty());
        assert_eq!(result.render_text(), "（查询成功，无结果）");
    }

    /// RDF 导出：Turtle/JSON-LD 逐条写出且能往返导入（导入计数一致）。
    #[tokio::test]
    async fn export_rdf_roundtrips_turtle_and_jsonld() {
        let (manager, dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();

        let turtle_path = dir.path().join("dump.ttl");
        let exported = manager
            .export_rdf_file(turtle_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(exported >= 4, "实体类型/名字/关系至少 4 条：{}", exported);
        let turtle = std::fs::read_to_string(&turtle_path).unwrap();
        assert!(turtle.contains("张三"), "{}", turtle);

        let (restored, _other) = temp_manager().await;
        let imported = restored
            .import_rdf_file(turtle_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(imported, exported, "往返导入应写回同等数量三元组");
        let facts = restored.load_all_facts().await.unwrap();
        assert!(facts
            .iter()
            .any(|fact| matches!(fact, PrologFact::Entity { name, .. } if name == "张三")));

        let jsonld_path = dir.path().join("dump.jsonld");
        let jsonld_exported = manager
            .export_rdf_file(jsonld_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(jsonld_exported, exported);
        let text = std::fs::read_to_string(&jsonld_path).unwrap();
        assert!(
            text.contains('\u{5f20}') || text.contains("\\\\u5f20"),
            "JSON-LD 应包含实体文本：{}",
            text
        );
    }

    /// 更正关系值：删旧宾语（实体 IRI）并写新宾语，同时补 kg:name，
    /// 保证推理快照仍能把新宾语识别为关系端点。
    #[tokio::test]
    async fn correct_object_value_replaces_entity_and_keeps_relation_visible() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("姚明"), person("226")])
            .await
            .unwrap();
        manager.save_knowledge(items("姚明", "226")).await.unwrap();
        // items() 的谓词是「师傅」，这里直接改用身高关系
        manager
            .save_knowledge(vec![KnowledgeItem::Relationship {
                from_label: "Person".to_string(),
                from_id: "姚明".to_string(),
                to_label: "Person".to_string(),
                to_id: "226".to_string(),
                rel_type: "身高".to_string(),
                properties: HashMap::new(),
            }])
            .await
            .unwrap();
        let (removed, inserted) = manager
            .correct_object_value("姚明", "身高", "226", "229")
            .await
            .unwrap();
        assert_eq!(removed, 1);
        assert!(inserted);
        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if from == "姚明" && rel_type == "身高" && to == "229"
            )),
            "新关系应可见：{:?}",
            facts
        );
        assert!(
            !facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if from == "姚明" && rel_type == "身高" && to == "226"
            )),
            "旧关系不应残留：{:?}",
            facts
        );
        // 旧值不存在时拒绝「更正」：不能变成无条件覆盖
        let (removed_again, inserted_again) = manager
            .correct_object_value("姚明", "身高", "226", "230")
            .await
            .unwrap();
        assert_eq!((removed_again, inserted_again), (0, false));
    }

    /// 未知扩展名拒绝导出，绝不猜测格式。
    #[tokio::test]
    async fn export_rdf_rejects_unknown_extension() {
        let (manager, dir) = temp_manager().await;
        let path = dir.path().join("dump.unknown");
        let error = manager.export_rdf_file(path.to_str().unwrap()).await;
        assert!(error.is_err());
        assert!(error.unwrap_err().to_string().contains("识别"));
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;
    use crate::knowledge_fusion::{
        Adjudication, FusionPolicy, NamespaceKind, NamespaceRecord, NamespaceStatus, ObjectKind,
        ScopedAssertion, TrustLevel,
    };
    use crate::perf_ledger::PerfLedger;

    async fn temp_manager() -> (OxigraphManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = OxigraphManager::with_perf_ledger(
            dir.path().to_str().unwrap(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        (manager, dir)
    }

    fn ns(id: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(id, id, NamespaceKind::Import, trust)
    }

    fn literal(namespace: &str, subject: &str, predicate: &str, object: &str) -> ScopedAssertion {
        ScopedAssertion::new(
            namespace,
            subject,
            predicate,
            object,
            ObjectKind::Literal,
            "test",
        )
    }

    /// 一个带类型与名字的实体断言集合（实体事实的物化需要 kg:name + rdf:type）。
    fn person_assertions(namespace: &str, name: &str, age: &str) -> Vec<ScopedAssertion> {
        vec![
            ScopedAssertion::new(
                namespace,
                name,
                "type",
                "Person",
                ObjectKind::Entity,
                "test",
            ),
            literal(namespace, name, "name", name),
            literal(namespace, name, "年龄", age),
        ]
    }

    fn has_property(facts: &[PrologFact], name: &str, key: &str, value: &str) -> bool {
        facts.iter().any(|fact| match fact {
            PrologFact::Entity {
                name: entity,
                properties,
                ..
            } if entity == name => properties.get(key).map(String::as_str) == Some(value),
            _ => false,
        })
    }

    /// 源图断言不进入默认图、不推进知识版本；重复导入幂等；账本可往返。
    #[tokio::test]
    async fn namespace_assertions_stay_out_of_default_graph() {
        let (manager, _dir) = temp_manager().await;
        let revision_before = manager.revision();
        let namespace = ns("alice", TrustLevel::Curated);
        let assertions = vec![literal("alice", "小明", "年龄", "18")];
        let added = manager
            .record_namespace_assertions(&namespace, &assertions)
            .await
            .unwrap();
        assert_eq!(added, 1);
        assert_eq!(
            manager.revision(),
            revision_before,
            "源图写入不推进知识版本"
        );
        assert!(
            manager.load_all_facts().await.unwrap().is_empty(),
            "默认图不受源图写入影响"
        );

        let rows = manager
            .query_sparql(
                "SELECT ?o WHERE { GRAPH <http://example.org/kg/graphs/source/alice> { \
                 ?s <http://example.org/kg/年龄> ?o } }",
            )
            .await
            .unwrap();
        assert_eq!(rows.data.len(), 1, "源图应可被 SPARQL 查询");

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert!(ledger.synced);
        assert_eq!(ledger.namespaces.len(), 1);
        assert_eq!(ledger.assertions, assertions);

        let added_again = manager
            .record_namespace_assertions(&namespace, &assertions)
            .await
            .unwrap();
        assert_eq!(added_again, 0, "重复导入幂等");
        assert_eq!(
            manager.load_fusion_ledger().await.unwrap().assertions.len(),
            1
        );
    }

    /// 功能谓词冲突：受管来源获胜、少数派隔离、默认图只写入赢家；重复融合幂等。
    #[tokio::test]
    async fn apply_fusion_materializes_winner_and_quarantines_loser() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();

        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = manager.fusion_plan(&policy).await.unwrap();
        assert_eq!(plan.conflicts().len(), 1);
        assert_eq!(plan.stats.quarantined, 1);

        let revision_before = manager.revision();
        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 3, "类型 / 名字 / 年龄三条都应物化");
        assert_eq!(outcome.removed, 0);
        assert!(outcome.changed());
        assert!(manager.revision() > revision_before, "默认图变化应推进版本");

        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "name", "小明"));
        assert!(has_property(&facts, "小明", "年龄", "18"));
        assert!(
            !has_property(&facts, "小明", "年龄", "19"),
            "少数派不得进入默认图"
        );

        let again = manager.apply_fusion(&policy).await.unwrap();
        assert!(!again.changed(), "重复融合应幂等");
        let revision_after = manager.revision();
        manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(manager.revision(), revision_after);

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert_eq!(ledger.applied.len(), 3);
        assert!(ledger
            .applied
            .iter()
            .any(|winner| winner.predicate == "年龄" && winner.object == "18"));
    }

    /// 用户裁决翻转赢家：旧赢家被回收、新赢家写入，且只影响声明的谓词槽位。
    #[tokio::test]
    async fn adjudication_flip_reclaims_previous_winner() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        manager.apply_fusion(&policy).await.unwrap();

        let mut ledger = manager.load_fusion_ledger().await.unwrap();
        ledger.record_adjudication(Adjudication::new("年龄", "friend", "本人确认", "user"));
        manager.sync_fusion_ledger(&ledger).await.unwrap();

        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 1, "只有年龄槽位翻转");
        assert_eq!(outcome.removed, 1, "旧赢家应被回收");
        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "年龄", "19"));
        assert!(!has_property(&facts, "小明", "年龄", "18"));

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert_eq!(ledger.adjudications.len(), 1);
        assert_eq!(ledger.applied.len(), 3);
        assert!(ledger
            .applied
            .iter()
            .any(|winner| winner.predicate == "年龄" && winner.object == "19"));
    }

    /// 历史默认图事实不被融合触碰（只增不删非托管数据）。
    #[tokio::test]
    async fn fusion_apply_keeps_legacy_default_facts() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![KnowledgeItem::Entity {
                label: "Person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "小明".to_string()),
                    ("年龄".to_string(), "17".to_string()),
                ]),
            }])
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 1, "只有年龄是新事实（类型/名字已存在）");
        assert_eq!(outcome.removed, 0);
        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "name", "小明"));

        // 同槽位的历史值（17）与融合赢家（18）都保留在默认图：不做静默删除
        for value in ["17", "18"] {
            let ask = manager
                .query_sparql(&format!(
                    "ASK {{ <http://example.org/kg/entity/小明> <http://example.org/kg/年龄> \"{}\" }}",
                    value
                ))
                .await
                .unwrap();
            assert_eq!(
                ask.data[0]["boolean"],
                serde_json::Value::Bool(true),
                "默认图应保留「{}」",
                value
            );
        }
    }

    /// 退役命名空间退出视图（其断言计为孤儿），只改账本元数据。
    #[tokio::test]
    async fn retiring_namespace_removes_it_from_view() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &[literal("kb", "小明", "年龄", "18")],
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();
        assert!(manager
            .set_namespace_status("friend", NamespaceStatus::Retired)
            .await
            .unwrap());
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = manager.fusion_plan(&policy).await.unwrap();
        assert_eq!(plan.stats.namespaces, 1);
        assert_eq!(plan.stats.orphans, 1);
        assert_eq!(
            plan.slots[0].delta,
            crate::knowledge_fusion::FusionDelta::New
        );
        assert_eq!(plan.slots[0].accepted[0].object, "18");
        assert!(!manager
            .set_namespace_status("ghost", NamespaceStatus::Retired)
            .await
            .unwrap());
    }

    /// RDF 四元组 → 命名空间断言：kg 词面还原本地名、rdf:type 归一、外部 IRI 保留。
    #[test]
    fn assertion_from_rdf_quad_normalizes_terms() {
        let quad = Quad::new(
            NamedNode::new("http://example.org/kg/entity/小明").unwrap(),
            NamedNode::new("http://example.org/kg/年龄").unwrap(),
            Literal::new_simple_literal("18"),
            GraphName::DefaultGraph,
        );
        let assertion = OxigraphManager::assertion_from_rdf_quad(&quad, "alice", "a.ttl").unwrap();
        assert_eq!(assertion.subject, "小明");
        assert_eq!(assertion.predicate, "年龄");
        assert_eq!(assertion.object, "18");
        assert_eq!(assertion.object_kind, ObjectKind::Literal);
        assert_eq!(assertion.namespace, "alice");
        assert_eq!(assertion.source, "a.ttl");

        let type_quad = Quad::new(
            NamedNode::new("http://example.org/kg/entity/小明").unwrap(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
            NamedNode::new("http://example.org/kg/Person").unwrap(),
            GraphName::DefaultGraph,
        );
        let assertion =
            OxigraphManager::assertion_from_rdf_quad(&type_quad, "alice", "a.ttl").unwrap();
        assert_eq!(assertion.predicate, "type");
        assert_eq!(assertion.object, "Person");
        assert_eq!(assertion.object_kind, ObjectKind::Entity);

        let blank = Quad::new(
            BlankNode::default(),
            NamedNode::new("http://example.org/kg/年龄").unwrap(),
            Literal::new_simple_literal("1"),
            GraphName::DefaultGraph,
        );
        assert!(OxigraphManager::assertion_from_rdf_quad(&blank, "alice", "a.ttl").is_none());
    }
}

//! KnowProlog 核心库。
//!
//! 库按职责划分为若干层，公开模块供 CLI、HTTP 服务与集成测试共同使用：
//!
//! - **语言与理解**：`nlp`、`question_analyzer`、`advanced_nlu`、`kg_pattern` 等；
//! - **抽取与存储**：`entity_relation_extractor`、`OxigraphManager`（本文件分片实现）；
//! - **推理与规划**：`prolog_engine`、`resolution_engine`、`planning`、`query_planner`；
//! - **对话与恢复**：`dialogue_manager`、`contextual_reference`、`recovery`；
//! - **质量与观测**：`quality`、`release`、`perf_ledger`、`provenance`。
//!
//! 模块声明保持字典序，便于查找；实现细节按功能拆分到 `lib/` 下的分片文件。

#![forbid(unsafe_code)]

use oxigraph::model::*;
use oxigraph::sparql::{Query, QueryResults};
use oxigraph::store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

// 公开模块：CLI、HTTP 服务与集成测试共享同一套类型与行为。
pub mod abduction;
pub mod advanced_nlu;
pub mod alignment_memory;
pub mod answer_plan;
pub mod answer_taxonomy;
pub mod artifact_audit;
pub mod attribute_bridge;
pub mod association_miner;
pub mod batch_conflict;
pub mod batch_teaching;
pub mod belief_revision;
pub mod binding_frames;
pub mod binding_table;
pub mod bindings;
pub mod cache;
pub mod canonical;
pub mod causal_reasoning;
pub mod causal_tokenization;
pub mod chinese_reasoning_nlu;
pub mod chronicle;
pub mod completion_episode;
pub mod confidence;
pub mod config_contract;
pub mod constructive;
pub mod constructive_eval;
pub mod context_image;
pub mod context_window;
pub mod contextual_closure;
pub mod contextual_rag;
pub mod contextual_reference;
pub mod contextual_segmentation;
pub mod conversation_theory;
pub mod conversational_graph;
pub mod coref_closure;
pub mod default_logic;
pub mod delta_cache;
pub mod dense_retrieval;
pub mod dependency_spine;
pub mod derivation;
pub mod dialogue_eval;
pub mod dialogue_frame;
pub mod dialogue_fuzz;
pub mod dialogue_manager;
pub mod dialogue_uri;
pub mod discourse;
pub mod disclosure_fusion;
pub mod embedding_eval;
pub mod entailment_eval;
pub mod entity_inventory;
pub mod entity_linker;
pub mod entity_registry;
pub mod entity_relation_extractor;
pub mod entity_thread;
pub mod epistemic;
pub mod epistemic_turn;
pub mod eval_recovery;
pub mod event_graph;
pub mod external_knowledge;
pub mod external_ledger;
pub mod fact_snapshot;
pub mod formal_semantics;
pub mod frame_semantics;
pub mod frontier;
pub mod fulfillment;
pub mod function_words;
pub mod fusion_arbitration;
pub mod grammar_extractor;
pub mod graph_diagnosis;
pub mod graph_ledger;
pub mod graph_question;
pub mod graph_repair;
pub mod graph_walk_embedding;
pub mod history_trust;
pub mod holographic_memory;
pub mod identity;
pub mod identity_dialogue;
pub mod identity_eval;
pub mod identity_risk;
pub mod import_episode;
pub mod import_frames;
pub mod import_journal;
pub mod intent_continuation;
pub mod joint_parse;
pub mod kg_pattern;
pub mod knowledge_delta;
pub mod knowledge_fusion;
pub mod knowledge_layers;
pub mod lang_runtime;
pub mod law_book;
pub mod lex_gate;
pub mod lexical_memory;
pub mod lexical_trace;
pub mod lexical_transducer;
pub mod link_eval;
pub mod link_prediction;
pub mod local_semantic_adapter;
pub mod logic_program;
pub mod lookahead;
pub mod markov_logic;
pub mod media_evidence;
pub mod mention_lexicon;
pub mod neuro_symbolic_contrast;
pub mod nlp;
pub mod nlu;
pub mod ontology_alignment;
pub mod ontology_reasoning;
pub mod open_closure;
pub mod owl_annotation;
pub mod ownthink_api;
pub mod path_memory;
pub mod pattern_closure;
pub mod pattern_evolution;
pub mod pattern_forge;
pub mod pattern_ir;
pub mod pending_question;
pub mod perf_ledger;
pub mod planning;
pub mod plugin;
pub mod pos_ontology;
pub mod predicate_registry;
pub mod prediction_draft;
pub mod predictive_elaboration;
pub mod privacy_flow;
pub mod probabilistic_query_intent;
pub mod problog;
pub mod prolog_engine;
pub mod prompt_spec;
pub mod proof_bench;
pub mod proof_journal;
pub mod proof_memory;
pub mod provenance;
pub mod quality;
pub mod quantifier;
pub mod quantity;
pub mod query_anchor;
pub mod query_board;
pub mod query_canonical;
pub mod query_planner;
pub mod query_rewrite;
pub mod query_scenario;
pub mod question_analyzer;
pub mod reading;
pub mod recall_audit;
pub mod recovery;
pub mod reference_probe;
pub mod referent_firewall;
pub mod relation_acquisition;
pub mod relation_frame;
pub mod release;
pub mod resolution_engine;
pub mod resource_ledger;
pub mod revision_guard;
pub mod rewrite_book;
pub mod rewrite_retrieval;
pub mod rule_pack;
pub mod schema_view;
pub mod search_port;
pub mod semantic_embedding;
pub mod semantic_engine;
pub mod semantic_entailment;
pub mod semantic_grammar;
pub mod semantic_monitor;
pub mod serve_api;
pub mod session_contract;
pub mod shape_dialogue;
pub mod shape_eval;
pub mod shape_gate;
pub mod shared_memory;
pub mod similarity_calculator;
pub mod slot_ledger;
pub mod soft_route;
pub mod stage_bus;
pub mod startup_ledger;
pub mod streaming_draft;
pub mod symbolic_generator;
pub mod symbolic_reasoner;
pub mod synonym_graph;
pub mod teachable_assertion;
pub mod temporal;
pub mod text_anchor;
pub mod text_identity;
pub mod trace_graph;
pub mod turn_runtime;
pub mod type_memory;
pub mod type_system;
pub mod variant_closure;
pub mod view_render;
pub mod viewpoint;
pub mod weak_construction;
pub mod working_memory;

#[cfg(test)]
mod dialogue_conformance;

// 配置、抽取规则、RDF 查询结果与通用辅助。
include!("lib/lib_part_01.rs");
// OxigraphManager：JSON-LD/RDF 导入、批次写入、查询与导出。
include!("lib/lib_part_02.rs");
// 直答缓存与事实快照相关实现。
include!("lib/lib_part_03.rs");
// 查询实体提取、响应格式化、配置加载与输入模式判定。
include!("lib/lib_part_04.rs");

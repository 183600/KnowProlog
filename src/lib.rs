use oxigraph::model::*;
use oxigraph::sparql::{Query, QueryResults};
use oxigraph::store::Store;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use tokio::sync::Mutex;
use toml;
use uuid::Uuid;

// 使模块公开，以便 main.rs 和 tests 可以访问
pub mod nlp;
pub mod nlu;
pub mod prolog_engine;
pub mod ownthink_api;
pub mod semantic_engine;
pub mod similarity_calculator;
pub mod advanced_nlu;
pub mod entity_relation_extractor;
pub mod local_semantic_adapter;
pub mod question_analyzer;
pub mod external_knowledge;
pub mod ontology_alignment;
pub mod link_prediction;
pub mod semantic_embedding;
pub mod query_planner;
pub mod symbolic_reasoner;
pub mod entity_linker;
pub mod symbolic_generator;
pub mod semantic_entailment;

use nlp::{preprocess, normalize_words, PrologRule, pattern_from_strings, TaggedWord};
use prolog_engine::PrologFact;

// RDF命名空间
const BASE_NS: &str = "http://example.org/kg/";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

#[derive(Debug)]
pub struct QueryResult {
    pub success: bool,
    pub data: Vec<HashMap<String, Value>>,
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
                // 避免递归调用时重复提取
                items.extend(self.extract(part, &normalized_part_tagged));
            }
        }
        
        items
    }

    fn apply_extraction_rule(&self, text: &str, tagged_words: &[TaggedWord], rule: &ExtractionRule) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();
        
        for pattern in &rule.patterns {
            if let Some(item) = self.match_pattern(text, tagged_words, pattern, &rule.entity_type) {
                items.push(item);
            }
        }
        
        items
    }

    fn match_pattern(&self, text: &str, tagged_words: &[TaggedWord], pattern: &PatternConfig, entity_type: &str) -> Option<KnowledgeItem> {
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

    fn extract_properties(&self, tagged_words: &[TaggedWord], mappings: &HashMap<String, String>) -> HashMap<String, String> {
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
                    let index: usize = parts[1].parse().unwrap_or(0);
                    
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
            
            // 特殊处理人名提取：如果是nr类型，尝试组合相邻的nr和unknown词
            if pos_rule == "nr" {
                let mut name_parts = Vec::new();
                let mut i = 0;
                while i < tagged_words.len() {
                    if tagged_words[i].pos == "nr" {
                        name_parts.push(tagged_words[i].text.clone());
                        // 检查后续的unknown词是否应该合并
                        i += 1;
                        while i < tagged_words.len() && tagged_words[i].pos == "unknown" {
                            name_parts.push(tagged_words[i].text.clone());
                            i += 1;
                        }
                        return Some(name_parts.join(""));
                    }
                    i += 1;
                }
            }
            
            // 处理通配符 *
            if pos_rule == "*" {
                // 对于通配符，返回所有不是功能词的词的组合
                let content_words: Vec<String> = tagged_words.iter()
                    .filter(|word| !matches!(word.pos.as_str(), "u" | "c" | "p" | "的" | "是" | "有"))
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
}

impl OxigraphManager {
    pub fn new(store_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 展开波浪号路径
        let expanded_path = if store_path.starts_with("~/") {
            if let Some(home_dir) = std::env::var_os("HOME") {
                let home_path = std::path::Path::new(&home_dir);
                home_path.join(&store_path[2..]).to_string_lossy().to_string()
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
        })
    }

    pub async fn import_jsonld(&self, jsonld_content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        
        // 解析 JSON-LD 内容
        let jsonld_value: Value = serde_json::from_str(jsonld_content)?;
        
        // 简化的 JSON-LD 处理 - 直接解析 JSON 结构
        // 暂时跳过复杂的 JSON-LD 展开，直接处理 JSON 结构
        self.process_jsonld_simple(&store, &jsonld_value).await?;
        
        Ok(())
    }

    async fn process_jsonld_simple(&self, store: &Store, jsonld: &Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 处理直接的 JSON-LD 结构
        if let Some(array) = jsonld.as_array() {
            for item in array {
                if let Some(obj) = item.as_object() {
                    // 处理每个对象
                    let context = Self::extract_jsonld_context(obj);
                    self.process_jsonld_object(store, obj, &context).await?;
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
                            self.process_jsonld_object(store, inner, &context).await?;
                        }
                    }
                } else if let Some(inner) = graph.as_object() {
                    self.process_jsonld_object(store, inner, &context).await?;
                }
            } else {
                // 单个对象
                self.process_jsonld_object(store, obj, &context).await?;
            }
        }
        Ok(())
    }

    /// 提取顶层 @context 中的字符串映射（term -> IRI），供类型与属性名展开使用
    fn extract_jsonld_context(obj: &serde_json::Map<String, Value>) -> HashMap<String, String> {
        let mut context = HashMap::new();
        if let Some(ctx) = obj.get("@context").and_then(|c| c.as_object()) {
            for (term, value) in ctx {
                if let Some(iri) = value.as_str() {
                    context.insert(term.clone(), iri.to_string());
                }
            }
        }
        context
    }

    /// 按 @context 展开 JSON-LD 词项：已是 IRI 的保留，映射词项替换为对应 IRI，否则回退到 @vocab
    fn expand_jsonld_term(term: &str, context: &HashMap<String, String>) -> String {
        if term.contains(':') {
            return term.to_string();
        }
        if let Some(iri) = context.get(term) {
            return iri.clone();
        }
        if let Some(vocab) = context.get("@vocab") {
            return format!("{}{}", vocab, term);
        }
        term.to_string()
    }

    async fn process_jsonld_object(&self, store: &Store, obj: &serde_json::Map<String, Value>, context: &HashMap<String, String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 获取主语 (subject)
        let subject_uri = if let Some(id) = obj.get("@id") {
            id.as_str().unwrap_or("").to_string()
        } else {
            format!("{}entity/{}", BASE_NS, Uuid::new_v4())
        };
        
        if subject_uri.is_empty() {
            return Ok(());
        }
        
        let subject = NamedNode::new(&subject_uri)?;
        
        // 处理类型信息
        if let Some(types) = obj.get("@type") {
            if let Some(type_array) = types.as_array() {
                for type_val in type_array {
                    if let Some(type_uri) = type_val.as_str() {
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
                    self.process_property_value(store, &subject, &predicate, val)?;
                }
            } else {
                self.process_property_value(store, &subject, &predicate, value)?;
            }
        }
        
        Ok(())
    }

    fn process_property_value(
        &self,
        store: &Store,
        subject: &NamedNode,
        predicate: &NamedNode,
        value: &Value,
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
                // 处理嵌套对象
                if let Some(id) = obj.get("@id") {
                    if let Some(uri) = id.as_str() {
                        let object = NamedNode::new(uri)?;
                        store.insert(&Quad::new(
                            subject.clone(),
                            predicate.clone(),
                            object,
                            GraphName::DefaultGraph,
                        ))?;
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
                            let dt = NamedNode::new(datatype_uri)?;
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

    pub async fn import_jsonld_from_file(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(file_path)?;
        self.import_jsonld(&content).await
    }

    pub async fn save_knowledge(&self, items: Vec<KnowledgeItem>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        
        for item in items {
            match item {
                KnowledgeItem::Entity { label, properties } => {
                    self.create_or_merge_entity(&store, &label, &properties)?;
                }
                KnowledgeItem::Relationship {
                    from_label, from_id, to_label, to_id, rel_type, properties
                } => {
                    self.create_relationship(
                        &store, &from_label, &from_id, &to_label, &to_id, &rel_type, &properties
                    )?;
                }
            }
        }
        Ok(())
    }

    fn create_or_merge_entity(&self, store: &Store, label: &str, properties: &HashMap<String, String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let entity_id = if let Some(name) = properties.get("name") {
            format!("{}entity/{}", BASE_NS, name)
        } else {
            format!("{}entity/{}", BASE_NS, Uuid::new_v4())
        };
        
        let subject = NamedNode::new(entity_id)?;
        let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
        let entity_type = NamedNode::new(format!("{}{}", BASE_NS, label))?;
        
        // 添加类型三元组
        store.insert(&Quad::new(
            subject.clone(),
            rdf_type,
            entity_type,
            GraphName::DefaultGraph,
        ))?;
        
        // 添加属性三元组
        for (key, value) in properties {
            let predicate = NamedNode::new(format!("{}{}", BASE_NS, key))?;
            let object = Literal::new_simple_literal(value);
            
            store.insert(&Quad::new(
                subject.clone(),
                predicate,
                object,
                GraphName::DefaultGraph,
            ))?;
        }
        
        Ok(())
    }

    fn create_relationship(
        &self,
        store: &Store,
        _from_label: &str, // 未使用
        from_id: &str,
        _to_label: &str, // 未使用
        to_id: &str,
        rel_type: &str,
        _properties: &HashMap<String, String>
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let from_subject = NamedNode::new(format!("{}entity/{}", BASE_NS, from_id))?;
        let to_object = NamedNode::new(format!("{}entity/{}", BASE_NS, to_id))?;
        let predicate = NamedNode::new(format!("{}{}", BASE_NS, rel_type))?;
        
        store.insert(&Quad::new(
            from_subject,
            predicate,
            to_object,
            GraphName::DefaultGraph,
        ))?;
        
        Ok(())
    }

    pub async fn get_relevant_facts(&self, query_entities: &[String], max_hops: usize) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut facts = Vec::new();
        let store = self.store.lock().await;
        
        if query_entities.is_empty() {
            return self.load_all_facts_internal(&store).await;
        }
        
        // 构建实体名称的匹配模式（合并为单次查询，避免逐实体重复扫描整库）
        let combined_pattern = query_entities.iter()
            .map(|e| format!(".*{}.*", regex::escape(e)))
            .collect::<Vec<_>>()
            .join("|");
        
        // 第一步：找到直接匹配的实体
        let direct_matches = self.find_entities_by_pattern(&store, &combined_pattern).await?;
        let mut relevant_entities: std::collections::HashSet<String> = direct_matches.into_iter().collect();
        
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
        let relevant_entity_list: Vec<String> = relevant_entities.iter().cloned().collect();
        facts.extend(self.build_entity_facts(&store, &relevant_entity_list).await?);
        
        // 第四步：添加实体间的关系
        let relations = self.find_relations_between(&store, &relevant_entities).await?;
        facts.extend(relations);
        
        Ok(facts)
    }

    pub async fn load_all_facts(&self) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        self.load_all_facts_internal(&store).await
    }

    async fn load_all_facts_internal(&self, store: &Store) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut facts = Vec::new();
        
        // 查询所有实体
        let query = r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?entity ?type ?prop ?value
            WHERE {
                ?entity rdf:type ?type .
                OPTIONAL { ?entity ?prop ?value . FILTER(?prop != rdf:type) }
            }
        "#;
        
        match Query::parse(query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            let mut entity_data: HashMap<String, (String, HashMap<String, String>)> = HashMap::new();
                            
                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    if let (Some(entity), Some(entity_type)) = (solution.get("entity"), solution.get("type")) {
                                        let entity_str = entity.to_string();
                                        let type_str = entity_type.to_string();
                                        
                                        // 提取实体名称（去掉URI前缀）
                                        let entity_name = if entity_str.contains("entity/") {
                                            entity_str.split("entity/").last().unwrap_or(&entity_str)
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        } else {
                                            entity_str.clone()
                                        };
                                        
                                        // 提取类型名称（去掉URI前缀）
                                        let type_name = if type_str.contains(BASE_NS) {
                                            type_str.replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        } else {
                                            type_str.to_lowercase()
                                        };
                                        
                                        // 获取或创建实体数据
                                        let entry = entity_data.entry(entity_name.clone())
                                            .or_insert_with(|| (type_name, HashMap::new()));
                                        
                                        // 添加属性
                                        if let (Some(prop), Some(value)) = (solution.get("prop"), solution.get("value")) {
                                            let prop_str = prop.to_string();
                                            let prop_name = if prop_str.contains(BASE_NS) {
                                                prop_str.replace(BASE_NS, "")
                                                    .trim_matches(|c| c == '<' || c == '>')
                                                    .to_string()
                                            } else {
                                                prop_str
                                            };
                                            
                                            let value_str = if let Term::Literal(lit) = value {
                                                lit.value().to_string()
                                            } else {
                                                value.to_string()
                                                    .trim_matches(|c| c == '<' || c == '>')
                                                    .to_string()
                                            };
                                            
                                            entry.1.insert(prop_name, value_str);
                                        }
                                    }
                                }
                            }
                            
                            // 转换为Prolog事实
                            for (entity_name, (entity_type, properties)) in entity_data {
                                facts.push(PrologFact::Entity {
                                    name: entity_name,
                                    entity_type,
                                    properties,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("SPARQL查询执行失败: {}", e);
                        return Err(Box::new(e));
                    }
                }
            }
            Err(e) => {
                eprintln!("SPARQL查询解析失败: {}", e);
                return Err(Box::new(e));
            }
        }
        
        Ok(facts)
    }
    
    pub async fn get_direct_answer(&self, query_type: &str, entities: &[String]) -> Option<String> {
        let store = self.store.lock().await;
        
        // 根据查询类型构造不同的SPARQL查询
        let query = match query_type {
            "capital_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?capital WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:capital ?capital .
                    }}
                "#, entity_name)
            }
            "altitude_query" => {
                let entity_name = if entities.is_empty() { return None; } else { &entities[0] };
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?altitude WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:altitude ?altitude .
                    }}
                "#, entity_name)
            }
            "chemical_formula_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?formula WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:formula ?formula .
                    }}
                "#, entity_name)
            }
            "author_query" => {
                let entity_name = if entities.is_empty() { return None; } else { &entities[0] };
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?author WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:author ?author .
                    }}
                "#, entity_name)
            }
            "champion_query" => {
                let entity_name = if entities.is_empty() { return None; } else { &entities[0] };
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?champion WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:champion ?champion .
                    }}
                "#, entity_name)
            }
            "birth_year_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?year WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:birth_year ?year .
                    }}
                "#, entity_name)
            }
            "concept_query" | "explanation_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?definition WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:definition ?definition .
                    }}
                "#, entity_name)
            }
            "quantity_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?quantity WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:quantity ?quantity .
                    }}
                "#, entity_name)
            }
            "planet_query" | "largest_query" => {
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?name WHERE {{
                        ?entity kg:type "planet" .
                        ?entity kg:size_rank "1" .
                        ?entity kg:name ?name .
                    }}
                "#)
            }
            "time_before_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?time WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:time ?time .
                    }}
                "#, entity_name)
            }
            "principle_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?principle WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:principle ?principle .
                    }}
                "#, entity_name)
            }
            "comparison_query" => {
                if entities.len() < 2 { return None; }
                let entity1 = &entities[0];
                let entity2 = &entities[1];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?difference WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?comparison kg:entity1 ?entity1 .
                        ?comparison kg:entity2 ?entity2 .
                        ?comparison kg:difference ?difference .
                    }}
                "#, entity1, entity2)
            }
            "philosophy_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?philosophy WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:philosophy ?philosophy .
                    }}
                "#, entity_name)
            }
            "impact_query" => {
                if entities.len() < 2 { return None; }
                let subject = &entities[0];
                let target = &entities[1];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?impact WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?entity1 kg:impact ?impact .
                        ?impact kg:target ?entity2 .
                    }}
                "#, subject, target)
            }
            "cause_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?cause WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:cause ?cause .
                    }}
                "#, entity_name)
            }
            "reason_query" => {
                if entities.is_empty() { return None; }
                let entity_name = &entities[0];
                format!(r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?reason WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:reason ?reason .
                    }}
                "#, entity_name)
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
                                            value.to_string()
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };
                                        
                                        // 根据查询类型格式化答案
                                        let answer = match query_type {
                                            "capital_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的首都是{}", entity_name, value_str)
                                            },
                                            "altitude_query" => {
                                                let entity_name = if entities.is_empty() { "该地点" } else { &entities[0] };
                                                format!("{}的海拔高度是{}米", entity_name, value_str)
                                            },
                                            "chemical_formula_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的化学分子式是{}", entity_name, value_str)
                                            },
                                            "author_query" => {
                                                let entity_name = if entities.is_empty() { "该作品" } else { &entities[0] };
                                                format!("{}的作者是{}", entity_name, value_str)
                                            },
                                            "champion_query" => {
                                                let entity_name = if entities.is_empty() { "该比赛" } else { &entities[0] };
                                                format!("{}的冠军是{}", entity_name, value_str)
                                            },
                                            "quantity_query" => {
                                                let entity_name = if entities.is_empty() { "该项目" } else { &entities[0] };
                                                format!("{}有{}个", entity_name, value_str)
                                            },
                                            "birth_year_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}诞生于{}年", entity_name, value_str)
                                            },
                                            "planet_query" | "largest_query" => format!("最大的行星是{}", value_str),
                                            "time_before_query" => {
                                                let entity_name = if entities.is_empty() { "该事件" } else { &entities[0] };
                                                format!("{}在{}年之前", entity_name, value_str)
                                            },
                                            "concept_query" | "explanation_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}是{}", entity_name, value_str)
                                            },
                                            "principle_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的工作原理是{}", entity_name, value_str)
                                            },
                                            "comparison_query" => {
                                                if entities.len() >= 2 {
                                                    format!("{}和{}的区别是{}", entities[0], entities[1], value_str)
                                                } else {
                                                    format!("区别是{}", value_str)
                                                }
                                            },
                                            "philosophy_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的哲学思想是{}", entity_name, value_str)
                                            },
                                            "impact_query" => {
                                                if entities.len() >= 2 {
                                                    format!("{}对{}的影响是{}", entities[0], entities[1], value_str)
                                                } else {
                                                    format!("影响是{}", value_str)
                                                }
                                            },
                                            "cause_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            },
                                            "reason_query" => {
                                                let entity_name = if entities.is_empty() { "" } else { &entities[0] };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            },
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

    async fn find_entities_by_pattern(&self, store: &Store, pattern: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut entities = Vec::new();
        
        let query = format!(r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT DISTINCT ?entity ?name WHERE {{
                ?entity rdf:type ?type .
                ?entity kg:name ?name .
                FILTER(REGEX(?name, "{}", "i"))
            }}
        "#, pattern);
        
        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
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
                }
            }
            Err(_) => {}
        }
        
        Ok(entities)
}

pub fn extract_query_entities(query: &str, config: &SystemConfig) -> Vec<String> {
    let mut entities = Vec::new();
    
    let tagged_words = preprocess(query, &config.nlp);
    
    for word in &tagged_words {
        if word.pos == "nr" || word.pos == "ns" || word.pos == "nz" || 
           word.pos == "n" || word.pos == "nt" {
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

    async fn find_neighbors(&self, store: &Store, entities: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut neighbors = Vec::new();
        
        if entities.is_empty() {
            return Ok(neighbors);
        }
        
        let entity_filter = entities.iter()
            .map(|e| format!("\"{}\"", e))
            .collect::<Vec<_>>()
            .join(", ");
        
        let query = format!(r#"
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
        "#, entity_filter, entity_filter, entity_filter, entity_filter);
        
        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
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
                }
            }
            Err(_) => {}
        }
        
        Ok(neighbors)
    }

    async fn build_entity_facts(&self, store: &Store, entity_names: &[String]) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut facts = Vec::new();
        
        if entity_names.is_empty() {
            return Ok(facts);
        }
        
        let name_filter = entity_names.iter()
            .map(|e| format!("\"{}\"", e))
            .collect::<Vec<_>>()
            .join(", ");
        
        let query = format!(r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?name ?type ?prop ?value WHERE {{
                ?entity kg:name ?name .
                ?entity rdf:type ?type .
                FILTER(?name IN ({}))
                OPTIONAL {{ ?entity ?prop ?value . FILTER(?prop != rdf:type && ?prop != kg:name) }}
            }}
        "#, name_filter);
        
        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            let mut entity_data: HashMap<String, (String, HashMap<String, String>)> = HashMap::new();
                            
                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    let entity_name = match solution.get("name") {
                                        Some(Term::Literal(lit)) => lit.value().to_string(),
                                        _ => continue,
                                    };
                                    
                                    let entry = entity_data.entry(entity_name)
                                        .or_insert_with(|| (String::new(), HashMap::new()));
                                    
                                    if let Some(type_term) = solution.get("type") {
                                        let type_str = type_term.to_string();
                                        // 后续类型覆盖先前类型
                                        entry.0 = if type_str.contains(BASE_NS) {
                                            type_str.replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        } else {
                                            type_str.to_lowercase()
                                        };
                                    }
                                    
                                    if let (Some(prop), Some(value)) = (solution.get("prop"), solution.get("value")) {
                                        let prop_str = prop.to_string();
                                        let prop_name = if prop_str.contains(BASE_NS) {
                                            prop_str.replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        } else {
                                            prop_str
                                        };
                                        
                                        let value_str = if let Term::Literal(lit) = value {
                                            lit.value().to_string()
                                        } else {
                                            value.to_string()
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

    pub async fn load_all_relations(&self) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let query = r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?from_name ?to_name ?rel WHERE {
                ?from kg:name ?from_name .
                ?to kg:name ?to_name .
                ?from ?rel ?to .
                FILTER(?rel != rdf:type)
                FILTER(?from != ?to)
            }
        "#;
        self.query_relations(&store, query).await
    }

    async fn find_relations_between(&self, store: &Store, entities: &std::collections::HashSet<String>) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }
        
        let entity_names: Vec<String> = entities.iter()
            .map(|e| format!("\"{}\"", e))
            .collect();
        let entity_filter = entity_names.join(", ");
        
        let query = format!(r#"
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
        "#, entity_filter, entity_filter);

        self.query_relations(store, &query).await
    }

    async fn query_relations(&self, store: &Store, query: &str) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut relations = Vec::new();

        match Query::parse(query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    if let (Some(from), Some(to), Some(rel)) = (
                                        solution.get("from_name"),
                                        solution.get("to_name"),
                                        solution.get("rel")
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
                                        
                                        let rel_str = rel.to_string();
                                        let rel_type = if rel_str.contains(BASE_NS) {
                                            rel_str.replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        } else {
                                            rel_str
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
                }
            }
            Err(_) => {}
        }
        
        Ok(relations)
    }
}

pub fn extract_query_entities(query: &str, config: &SystemConfig) -> Vec<String> {
    let mut entities = Vec::new();
    
    let tagged_words = preprocess(query, &config.nlp);
    
    for word in &tagged_words {
        if word.pos == "nr" || word.pos == "ns" || word.pos == "nz" || 
           word.pos == "n" || word.pos == "nt" {
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

    pub fn format_prolog_results(&self, results: &[HashMap<String, String>], query_type: &str) -> String {
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
                let relations: Vec<String> = results.iter()
                    .filter_map(|r| r.get("Relation"))
                    .map(|r| r.clone())
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
                if let Some(capital) = results[0].get("capital") {
                    format!("首都是{}", capital)
                } else {
                    "未找到首都信息".to_string()
                }
            }
            "altitude_query" => {
                if let Some(altitude) = results[0].get("altitude") {
                    format!("海拔高度是{}米", altitude)
                } else {
                    "未找到海拔信息".to_string()
                }
            }
            "chemical_formula_query" => {
                if let Some(formula) = results[0].get("formula") {
                    format!("化学分子式是{}", formula)
                } else {
                    "未找到分子式信息".to_string()
                }
            }
            "author_query" => {
                if let Some(author) = results[0].get("author") {
                    format!("作者是{}", author)
                } else {
                    "未找到作者信息".to_string()
                }
            }
            "champion_query" => {
                if let Some(champion) = results[0].get("champion") {
                    format!("冠军是{}", champion)
                } else {
                    "未找到冠军信息".to_string()
                }
            }
            "birth_year_query" => {
                if let Some(year) = results[0].get("birth_year") {
                    format!("诞生于{}年", year)
                } else {
                    "未找到诞生年份信息".to_string()
                }
            }
            "largest_query" => {
                if let Some(entity) = results[0].get("entity") {
                    format!("最大的是{}", entity)
                } else {
                    "未找到相关信息".to_string()
                }
            }
            "quantity_query" => {
                if let Some(quantity) = results[0].get("quantity") {
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
            _ => "根据现有信息，找到了相关内容，但无法生成详细回答。".to_string()
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
    config.prolog_rules.iter().map(|rule_cfg| {
        PrologRule {
            pattern: pattern_from_strings(rule_cfg.pattern.clone()),
            query_type: rule_cfg.query_type.clone(),
            priority: rule_cfg.priority,
        }
    }).collect()
}

pub fn create_default_synonyms() -> HashMap<String, String> {
    let mut synonyms = HashMap::new();
    
    // 基础同义词映射
    synonyms.insert("老婆".to_string(), "妻子".to_string());
    synonyms.insert("老公".to_string(), "丈夫".to_string());
    
    synonyms
}

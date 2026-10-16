use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use crate::semantic_engine::{SemanticEngine, SemanticKnowledge};
use crate::similarity_calculator::{SimilarityCalculator, SimilarityResult, WordEmbedding, ConceptNode};
use crate::advanced_nlu::{AdvancedNLU, IntentResult};
use crate::entity_relation_extractor::{EntityRelationExtractor, EntityExtraction, KnowledgeTriple};
use crate::question_analyzer::{QuestionAnalyzer, QuestionAnalysis};
use crate::ownthink_api::{AmbiguousResponse, KnowledgeResponse, KnowledgeData, EavResponse, EavData};

pub struct LocalSemanticAdapter {
    // 各子引擎按需惰性构建，避免未启用或未使用时的初始化开销
    semantic_engine_cell: OnceLock<SemanticEngine>,
    similarity_calculator_cell: OnceLock<SimilarityCalculator>,
    advanced_nlu_cell: OnceLock<AdvancedNLU>,
    entity_extractor_cell: OnceLock<EntityRelationExtractor>,
    question_analyzer_cell: OnceLock<QuestionAnalyzer>,
    enabled: bool,
}

impl LocalSemanticAdapter {
    pub fn new(enabled: bool) -> Self {
        Self {
            semantic_engine_cell: OnceLock::new(),
            similarity_calculator_cell: OnceLock::new(),
            advanced_nlu_cell: OnceLock::new(),
            entity_extractor_cell: OnceLock::new(),
            question_analyzer_cell: OnceLock::new(),
            enabled,
        }
    }

    fn semantic_engine(&self) -> &SemanticEngine {
        self.semantic_engine_cell.get_or_init(SemanticEngine::new)
    }

    fn similarity_calculator(&self) -> &SimilarityCalculator {
        self.similarity_calculator_cell.get_or_init(SimilarityCalculator::new)
    }

    fn advanced_nlu(&self) -> &AdvancedNLU {
        self.advanced_nlu_cell.get_or_init(AdvancedNLU::new)
    }

    fn entity_extractor(&self) -> &EntityRelationExtractor {
        self.entity_extractor_cell.get_or_init(EntityRelationExtractor::new)
    }

    fn question_analyzer(&self) -> &QuestionAnalyzer {
        self.question_analyzer_cell.get_or_init(QuestionAnalyzer::new)
    }

    pub fn initialize_with_knowledge(&self, entities: Vec<SemanticKnowledge>) -> Result<(), Box<dyn std::error::Error>> {
        if !self.enabled {
            return Ok(());
        }
        
        self.semantic_engine().initialize_knowledge_base(entities)?;
        Ok(())
    }

    pub fn initialize_embeddings(&self, embeddings: Vec<WordEmbedding>) -> Result<(), Box<dyn std::error::Error>> {
        if !self.enabled {
            return Ok(());
        }
        
        self.similarity_calculator().initialize_embeddings(embeddings)?;
        Ok(())
    }

    pub fn initialize_concepts(&self, concepts: Vec<ConceptNode>) -> Result<(), Box<dyn std::error::Error>> {
        if !self.enabled {
            return Ok(());
        }
        
        self.similarity_calculator().initialize_concept_hierarchy(concepts)?;
        Ok(())
    }

    pub async fn get_ambiguous_entities(&self, mention: &str) -> Result<AmbiguousResponse, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        let candidates = self.semantic_engine().get_ambiguous_entities(mention).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?;
        
        let data: Vec<(String, u64)> = candidates.into_iter()
            .map(|candidate| (candidate.entity, candidate.weight))
            .collect();

        Ok(AmbiguousResponse {
            message: "success".to_string(),
            data,
        })
    }

    pub async fn get_entity_knowledge(&self, entity: &str) -> Result<KnowledgeResponse, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        let knowledge = self.semantic_engine().get_entity_knowledge(entity).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?;
        
        let mut avp = Vec::new();
        for (key, values) in &knowledge.attributes {
            for value in values {
                avp.push(vec![key.clone(), value.clone()]);
            }
        }

        let data = KnowledgeData {
            entity: knowledge.entity,
            desc: knowledge.description,
            tag: knowledge.tags,
            avp,
        };

        Ok(KnowledgeResponse {
            message: "success".to_string(),
            data,
        })
    }

    pub async fn get_entity_attribute_value(&self, entity: &str, attribute: &str) -> Result<EavResponse, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        let values = self.semantic_engine().get_entity_attribute_value(entity, attribute).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?;

        let data = EavData {
            entity: entity.to_string(),
            attribute: attribute.to_string(),
            value: values,
        };

        Ok(EavResponse {
            message: "success".to_string(),
            data,
        })
    }

    pub async fn resolve_entity_ambiguity(&self, mention: &str, context: Option<&str>) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(None);
        }

        Ok(self.semantic_engine().resolve_entity_ambiguity(mention, context).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn get_simplified_knowledge(&self, entity: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(HashMap::new());
        }

        Ok(self.semantic_engine().get_simplified_knowledge(entity).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn batch_resolve_entities(&self, mentions: &[String]) -> Result<HashMap<String, Option<String>>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(HashMap::new());
        }

        Ok(self.semantic_engine().batch_resolve_entities(mentions).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn analyze_question(&self, question: &str) -> Result<QuestionAnalysis, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        Ok(self.question_analyzer().analyze_question(question).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn analyze_intent(&self, query: &str) -> Result<IntentResult, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        Ok(self.advanced_nlu().analyze_query(query).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn extract_entities_and_relations(&self, text: &str) -> Result<(Vec<EntityExtraction>, Vec<KnowledgeTriple>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok((Vec::new(), Vec::new()));
        }

        Ok(self.entity_extractor().extract_entities_and_relations(text).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn calculate_semantic_similarity(&self, text1: &str, text2: &str) -> Result<SimilarityResult, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Err("本地语义理解功能未启用".into());
        }

        Ok(self.similarity_calculator().calculate_semantic_similarity(text1, text2).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?)
    }

    pub async fn calculate_simple_similarity(&self, text1: &str, text2: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(0.0);
        }

        Ok(self.similarity_calculator().calculate_simple_similarity(text1, text2).await)
    }

    pub fn add_entity_knowledge(&self, entity: SemanticKnowledge) {
        if self.enabled {
            self.semantic_engine().add_entity_knowledge(entity);
        }
    }

    pub fn add_word_embedding(&self, word: String, vector: Vec<f32>) {
        if self.enabled {
            self.similarity_calculator().add_word_embedding(word, vector);
        }
    }

    pub fn add_concept(&self, concept: ConceptNode) {
        if self.enabled {
            self.similarity_calculator().add_concept(concept);
        }
    }

    pub fn get_entity_count(&self) -> usize {
        if self.enabled {
            self.semantic_engine().get_entity_count()
        } else {
            0
        }
    }

    pub fn get_embedding_count(&self) -> usize {
        if self.enabled {
            self.similarity_calculator().get_embedding_count()
        } else {
            0
        }
    }

    pub fn get_concept_count(&self) -> usize {
        if self.enabled {
            self.similarity_calculator().get_concept_count()
        } else {
            0
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub async fn enhance_query_understanding(&self, query: &str) -> Result<EnhancedQueryResult, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(EnhancedQueryResult::default());
        }

        // 意图分析
        let intent_result = self.analyze_intent(query).await?;
        let intent = intent_result.intent.clone();
        let confidence = intent_result.confidence;
        
        // 实体和关系抽取
        let (entities, relations) = self.extract_entities_and_relations(query).await?;
        
        // 语义相似度计算（与已知模式比较）
        let similarity_scores = self.calculate_pattern_similarities(query).await?;
        
        Ok(EnhancedQueryResult {
            intent,
            confidence,
            entities: entities.into_iter().map(|e| e.entity).collect(),
            relations: relations.into_iter().map(|r| format!("{}-{}-{}", r.subject, r.predicate, r.object)).collect(),
            similarity_scores,
            enhanced_query: self.generate_enhanced_query(query, &intent_result).await?,
        })
    }

    async fn calculate_pattern_similarities(&self, query: &str) -> Result<HashMap<String, f64>, Box<dyn std::error::Error + Send + Sync>> {
        let mut similarities = HashMap::new();
        
        // 与常见查询模式的相似度计算
        let common_patterns = vec![
            "谁是什么",
            "什么是谁",
            "多少年龄",
            "哪里出生",
            "什么关系",
            "多少数量",
            "哪个更好",
            "如何使用",
            "为什么这样",
        ];
        
        for pattern in common_patterns {
            let similarity = self.calculate_simple_similarity(query, pattern).await?;
            similarities.insert(pattern.to_string(), similarity);
        }
        
        Ok(similarities)
    }

    async fn generate_enhanced_query(&self, original_query: &str, intent_result: &IntentResult) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // 基于意图和实体生成增强查询
        let mut enhanced = original_query.to_string();
        
        // 根据意图类型添加上下文信息
        match intent_result.intent.as_str() {
            "entity_query" => {
                if !intent_result.entities.is_empty() {
                    let entity_names: Vec<String> = intent_result.entities.iter()
                        .map(|e| e.text.clone())
                        .collect();
                    enhanced = format!("查询实体信息: {} - {}", entity_names.join(", "), enhanced);
                }
            }
            "attribute_query" => {
                enhanced = format!("查询属性信息: {}", enhanced);
            }
            "relation_query" => {
                enhanced = format!("查询关系信息: {}", enhanced);
            }
            _ => {}
        }
        
        Ok(enhanced)
    }

    pub async fn analyze_complex_query(&self, query: &str) -> Result<ComplexQueryResult, Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(ComplexQueryResult::default());
        }

        // 第一阶段：问题分析
        let question_analysis = self.analyze_question(query).await?;
        
        // 第二阶段：意图分析
        let intent_result = self.analyze_intent(query).await?;
        
        // 第三阶段：实体和关系抽取
        let (entities, relations) = self.extract_entities_and_relations(query).await?;
        
        // 第四阶段：确定查询类型
        let query_type = self.determine_complex_query_type(&question_analysis, &intent_result, query).await?;
        
        // 第五阶段：计算置信度
        let confidence = self.calculate_complex_confidence(&question_analysis, &intent_result, &entities).await?;
        
        // 提取实体名称字符串
        let mut entity_names: Vec<String> = entities.into_iter().map(|e| e.entity).collect();
        
        // 补充知识库中已知实体的识别结果：通用抽取器可能识别不出
        // 采样知识库中已收录的实体（如"珠穆朗玛峰"），这里通过别名做最长匹配补齐
        let knowledge_entities = self.semantic_engine().analyze_intent(query).await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { format!("{}", e).into() })?;
        let mut seen_entities: HashSet<String> = entity_names.iter().cloned().collect();
        for entity_name in knowledge_entities.entities {
            if seen_entities.insert(entity_name.clone()) {
                entity_names.push(entity_name);
            }
        }
        
        // 第六阶段：生成查询模式
        let query_patterns = self.generate_complex_query_patterns(&query_type, &entity_names).await?;
        
        // 第七阶段：尝试本地回答
        let suggested_answer = if confidence > 0.6 {
            self.try_local_answer(&query_type, &entity_names, query).await?
        } else {
            None
        };
        
        Ok(ComplexQueryResult {
            original_query: query.to_string(),
            query_type,
            confidence,
            question_analysis,
            intent_result,
            entities: entity_names,
            relations: relations.into_iter().map(|r| format!("{}-{}-{}", r.subject, r.predicate, r.object)).collect(),
            query_patterns,
            suggested_answer,
        })
    }

    async fn determine_complex_query_type(&self, question_analysis: &QuestionAnalysis, intent_result: &IntentResult, query: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut scores = HashMap::new();
        
        // 基于问题类型的评分
        let question_type_score = question_analysis.confidence;
        let question_based_type = match question_analysis.question_type {
            crate::question_analyzer::QuestionType::FactualQuery => {
                if query.contains("首都") {
                    "capital_query"
                } else if query.contains("海拔") {
                    "altitude_query"
                } else if query.contains("分子式") {
                    "chemical_formula_query"
                } else if query.contains("作者") {
                    "author_query"
                } else if query.contains("冠军") {
                    "champion_query"
                } else if query.contains("诞生") || query.contains("发明") {
                    "birth_year_query"
                } else if query.contains("多少") && (query.contains("块") || query.contains("个")) {
                    "quantity_query"
                } else if query.contains("最大") && query.contains("行星") {
                    "planet_query"
                } else {
                    "property_query"
                }
            },
            crate::question_analyzer::QuestionType::ConceptExplanation => "explanation_query",
            crate::question_analyzer::QuestionType::ComparisonAnalysis => "comparison_query",
            crate::question_analyzer::QuestionType::HistoricalDescription => "impact_query",
            crate::question_analyzer::QuestionType::ScientificPrinciple => "principle_query",
            crate::question_analyzer::QuestionType::DefinitionQuestion => "concept_query",
            crate::question_analyzer::QuestionType::CausalExplanation => "cause_query",
            crate::question_analyzer::QuestionType::ProcessDescription => "principle_query",
            crate::question_analyzer::QuestionType::ListQuery => "quantity_query",
        };
        
        scores.insert(question_based_type.to_string(), question_type_score);
        
        // 基于意图的评分
        let intent_score = intent_result.confidence * 0.8;
        let intent_based_type = match intent_result.intent.as_str() {
            "entity_query" => "property_query",
            "attribute_query" => "property_query",
            "relation_query" => "relation_query",
            "comparison_query" => "comparison_query",
            _ => "property_query",
        };
        
        *scores.entry(intent_based_type.to_string()).or_insert(0.0) += intent_score;
        
        // 基于关键词的直接匹配
        let keyword_matches = vec![
            ("首都", "capital_query"),
            ("海拔", "altitude_query"),
            ("化学", "chemical_formula_query"),
            ("分子式", "chemical_formula_query"),
            ("作者", "author_query"),
            ("冠军", "champion_query"),
            ("诞生", "birth_year_query"),
            ("解释", "explanation_query"),
            ("什么是", "concept_query"),
            ("比较", "comparison_query"),
            ("区别", "comparison_query"),
            ("影响", "impact_query"),
            ("原理", "principle_query"),
            ("哲学", "philosophy_query"),
            ("思想", "philosophy_query"),
            ("原因", "cause_query"),
            ("为何", "reason_query"),
            ("为什么", "cause_query"),
            ("最大", "largest_query"),
            ("行星", "planet_query"),
        ];
        
        for (keyword, query_type) in keyword_matches {
            if query.contains(keyword) {
                *scores.entry(query_type.to_string()).or_insert(0.0) += 0.9;
            }
        }
        
        // 选择得分最高的查询类型
        let best_type = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(query_type, _)| query_type)
            .unwrap_or_else(|| "property_query".to_string());
        
        Ok(best_type)
    }

    async fn calculate_complex_confidence(&self, question_analysis: &QuestionAnalysis, intent_result: &IntentResult, entities: &[crate::entity_relation_extractor::EntityExtraction]) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut confidence = 0.0;
        let mut weight_sum = 0.0;
        
        // 问题分析置信度
        confidence += question_analysis.confidence * 0.4;
        weight_sum += 0.4;
        
        // 意图分析置信度
        confidence += intent_result.confidence * 0.3;
        weight_sum += 0.3;
        
        // 实体识别置信度
        if !entities.is_empty() {
            let entity_confidence = entities.iter()
                .map(|e| e.confidence)
                .sum::<f64>() / entities.len() as f64;
            confidence += entity_confidence * 0.3;
            weight_sum += 0.3;
        }
        
        if weight_sum > 0.0 {
            confidence /= weight_sum;
        }
        
        Ok(confidence.min(1.0))
    }

    async fn generate_complex_query_patterns(&self, query_type: &str, _entities: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut patterns = Vec::new();
        
        match query_type {
            "capital_query" => {
                patterns.push("SPARQL: SELECT ?capital WHERE { ?country kg:capital ?capital }".to_string());
                patterns.push("Prolog: capital_of(Country, Capital)".to_string());
            },
            "altitude_query" => {
                patterns.push("SPARQL: SELECT ?altitude WHERE { ?place kg:altitude ?altitude }".to_string());
                patterns.push("Prolog: altitude(Place, Height)".to_string());
            },
            "chemical_formula_query" => {
                patterns.push("SPARQL: SELECT ?formula WHERE { ?compound kg:chemical_formula ?formula }".to_string());
                patterns.push("Prolog: chemical_formula(Compound, Formula)".to_string());
            },
            "author_query" => {
                patterns.push("SPARQL: SELECT ?author WHERE { ?work kg:author ?author }".to_string());
                patterns.push("Prolog: author_of(Work, Author)".to_string());
            },
            "champion_query" => {
                patterns.push("SPARQL: SELECT ?champion WHERE { ?event kg:champion ?champion }".to_string());
                patterns.push("Prolog: champion_of(Event, Champion)".to_string());
            },
            "birth_year_query" => {
                patterns.push("SPARQL: SELECT ?year WHERE { ?entity kg:birth_year ?year }".to_string());
                patterns.push("Prolog: birth_year(Entity, Year)".to_string());
            },
            "explanation_query" | "concept_query" => {
                patterns.push("SPARQL: SELECT ?definition WHERE { ?concept kg:definition ?definition }".to_string());
                patterns.push("Prolog: definition(Concept, Definition)".to_string());
            },
            "comparison_query" => {
                patterns.push("SPARQL: 比较查询需要复杂的SPARQL联合查询".to_string());
                patterns.push("Prolog: compare(Entity1, Entity2, Differences)".to_string());
            },
            "quantity_query" => {
                patterns.push("SPARQL: SELECT ?quantity WHERE { ?entity kg:quantity ?quantity }".to_string());
                patterns.push("Prolog: quantity(Entity, Amount)".to_string());
            },
            "planet_query" | "largest_query" => {
                patterns.push("SPARQL: SELECT ?name WHERE { ?entity kg:size_rank '1' ; kg:name ?name }".to_string());
                patterns.push("Prolog: largest(Category, Entity)".to_string());
            },
            _ => {
                patterns.push("通用模式: 实体-属性-值查询".to_string());
            }
        }
        
        Ok(patterns)
    }

    async fn try_local_answer(&self, query_type: &str, entities: &[String], _query: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        // 尝试从本地知识库获取答案
        if entities.is_empty() {
            return Ok(None);
        }

        let entity = &entities[0];
        
        match query_type {
            "capital_query" => {
                if let Ok(knowledge) = self.get_entity_knowledge(entity).await {
                    for avp in &knowledge.data.avp {
                        if avp.len() >= 2 && (avp[0] == "首都" || avp[0] == "capital") {
                            return Ok(Some(format!("{}的首都是{}", entity, avp[1])));
                        }
                    }
                }
            },
            "altitude_query" => {
                if let Ok(knowledge) = self.get_entity_knowledge(entity).await {
                    for avp in &knowledge.data.avp {
                        if avp.len() >= 2 && (avp[0] == "海拔" || avp[0] == "altitude") {
                            return Ok(Some(format!("{}的海拔高度是{}", entity, avp[1])));
                        }
                    }
                }
            },
            "chemical_formula_query" => {
                if let Ok(knowledge) = self.get_entity_knowledge(entity).await {
                    for avp in &knowledge.data.avp {
                        if avp.len() >= 2 && (avp[0] == "化学式" || avp[0] == "分子式" || avp[0] == "formula") {
                            return Ok(Some(format!("{}的化学分子式是{}", entity, avp[1])));
                        }
                    }
                }
            },
            "author_query" => {
                if let Ok(knowledge) = self.get_entity_knowledge(entity).await {
                    for avp in &knowledge.data.avp {
                        if avp.len() >= 2 && (avp[0] == "作者" || avp[0] == "author") {
                            return Ok(Some(format!("{}的作者是{}", entity, avp[1])));
                        }
                    }
                }
            },
            "concept_query" | "explanation_query" => {
                if let Ok(knowledge) = self.get_entity_knowledge(entity).await {
                    if !knowledge.data.desc.is_empty() {
                        return Ok(Some(format!("{}是{}", entity, knowledge.data.desc)));
                    }
                }
            },
            _ => {}
        }

        Ok(None)
    }

    pub async fn add_sample_knowledge_for_tests(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.enabled {
            return Ok(());
        }

        let sample_entities = vec![
            SemanticKnowledge {
                entity: "法国".to_string(),
                description: "西欧国家".to_string(),
                tags: vec!["国家".to_string()],
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("首都".to_string(), vec!["巴黎".to_string()]);
                    attrs.insert("人口".to_string(), vec!["约6700万".to_string()]);
                    attrs
                },
                relations: HashMap::new(),
            },
            SemanticKnowledge {
                entity: "珠穆朗玛峰".to_string(),
                description: "世界最高峰".to_string(),
                tags: vec!["山峰".to_string()],
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("海拔".to_string(), vec!["8848.86米".to_string()]);
                    attrs.insert("位置".to_string(), vec!["中国西藏".to_string()]);
                    attrs
                },
                relations: HashMap::new(),
            },
            SemanticKnowledge {
                entity: "水".to_string(),
                description: "最常见的化合物".to_string(),
                tags: vec!["化学物质".to_string()],
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("化学式".to_string(), vec!["H2O".to_string()]);
                    attrs.insert("分子式".to_string(), vec!["H2O".to_string()]);
                    attrs
                },
                relations: HashMap::new(),
            },
            SemanticKnowledge {
                entity: "哈姆雷特".to_string(),
                description: "莎士比亚的著名悲剧".to_string(),
                tags: vec!["文学作品".to_string()],
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("作者".to_string(), vec!["威廉·莎士比亚".to_string()]);
                    attrs.insert("类型".to_string(), vec!["悲剧".to_string()]);
                    attrs
                },
                relations: HashMap::new(),
            },
            SemanticKnowledge {
                entity: "2022年世界杯".to_string(),
                description: "第22届FIFA世界杯".to_string(),
                tags: vec!["体育赛事".to_string()],
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("冠军".to_string(), vec!["阿根廷".to_string()]);
                    attrs.insert("举办地".to_string(), vec!["卡塔尔".to_string()]);
                    attrs
                },
                relations: HashMap::new(),
            },
        ];

        for entity in sample_entities {
            self.add_entity_knowledge(entity);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexQueryResult {
    pub original_query: String,
    pub query_type: String,
    pub confidence: f64,
    pub question_analysis: QuestionAnalysis,
    pub intent_result: IntentResult,
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub query_patterns: Vec<String>,
    pub suggested_answer: Option<String>,
}

impl Default for ComplexQueryResult {
    fn default() -> Self {
        Self {
            original_query: String::new(),
            query_type: "unknown".to_string(),
            confidence: 0.0,
            question_analysis: QuestionAnalysis {
                original_question: String::new(),
                question_type: crate::question_analyzer::QuestionType::FactualQuery,
                domain: crate::question_analyzer::Domain::General,
                complexity: crate::question_analyzer::ComplexityLevel::Simple,
                intent: crate::question_analyzer::Intent::SeekInformation,
                entities: Vec::new(),
                keywords: Vec::new(),
                confidence: 0.0,
                suggested_query_patterns: Vec::new(),
            },
            intent_result: IntentResult {
                intent: "unknown".to_string(),
                confidence: 0.0,
                entities: Vec::new(),
                slots: HashMap::new(),
                context: crate::advanced_nlu::QueryContext {
                    query_type: "unknown".to_string(),
                    domain: "general".to_string(),
                    temporal_info: None,
                    spatial_info: None,
                    modifiers: Vec::new(),
                },
            },
            entities: Vec::new(),
            relations: Vec::new(),
            query_patterns: Vec::new(),
            suggested_answer: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQueryResult {
    pub intent: String,
    pub confidence: f64,
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub similarity_scores: HashMap<String, f64>,
    pub enhanced_query: String,
}

impl Default for EnhancedQueryResult {
    fn default() -> Self {
        Self {
            intent: "unknown".to_string(),
            confidence: 0.0,
            entities: Vec::new(),
            relations: Vec::new(),
            similarity_scores: HashMap::new(),
            enhanced_query: String::new(),
        }
    }
}

impl Default for LocalSemanticAdapter {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_query_analysis() {
        let adapter = LocalSemanticAdapter::new(true);
        
        // 添加测试知识
        adapter.add_sample_knowledge_for_tests().await.unwrap();
        
        let test_queries = vec![
            "法国的首都是哪里？",
            "珠穆朗玛峰的海拔高度是多少？",
            "水的化学分子式是什么？",
            "《哈姆雷特》的作者是谁？",
            "2022年世界杯的冠军是哪支队伍？",
        ];
        
        for query in test_queries {
            println!("测试查询: {}", query);
            let result = adapter.analyze_complex_query(query).await.unwrap();
            println!("查询类型: {}", result.query_type);
            println!("置信度: {:.2}", result.confidence);
            if let Some(answer) = &result.suggested_answer {
                println!("建议答案: {}", answer);
            }
            println!("---");
        }
    }

    #[tokio::test]
    async fn test_local_semantic_adapter_creation() {
        let adapter = LocalSemanticAdapter::new(true);
        assert!(adapter.is_enabled());
        assert_eq!(adapter.get_entity_count(), 0);
    }

    #[tokio::test]
    async fn test_entity_resolution() {
        let adapter = LocalSemanticAdapter::new(true);
        
        // 添加测试实体
        let entity = SemanticKnowledge {
            entity: "测试实体".to_string(),
            description: "这是一个测试实体".to_string(),
            tags: vec!["测试".to_string()],
            attributes: HashMap::new(),
            relations: HashMap::new(),
        };
        
        adapter.add_entity_knowledge(entity);
        assert_eq!(adapter.get_entity_count(), 1);
    }

    #[tokio::test]
    async fn test_intent_analysis() {
        let adapter = LocalSemanticAdapter::new(true);
        
        let result = adapter.analyze_intent("刘德华是谁").await.unwrap();
        assert_eq!(result.intent, "entity_query");
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_enhanced_query_understanding() {
        let adapter = LocalSemanticAdapter::new(true);
        
        let result = adapter.enhance_query_understanding("刘德华的年龄是多少").await.unwrap();
        assert!(!result.intent.is_empty());
        assert!(!result.enhanced_query.is_empty());
    }

    #[tokio::test]
    async fn test_disabled_adapter() {
        let adapter = LocalSemanticAdapter::new(false);
        assert!(!adapter.is_enabled());
        
        let result = adapter.analyze_intent("测试查询").await;
        assert!(result.is_err());
    }
}
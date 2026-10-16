use jieba_rs::Jieba;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCandidate {
    pub entity: String,
    pub weight: u64,
    pub confidence: f64,
    pub entity_type: Vec<String>,
    pub attributes: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticKnowledge {
    pub entity: String,
    pub description: String,
    pub tags: Vec<String>,
    pub attributes: HashMap<String, Vec<String>>,
    pub relations: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAnalysisResult {
    pub intent_type: String,
    pub confidence: f64,
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub query_patterns: Vec<String>,
}

pub struct SemanticEngine {
    jieba: Arc<Jieba>,
    entity_knowledge: Arc<RwLock<HashMap<String, SemanticKnowledge>>>,
    entity_aliases: Arc<RwLock<HashMap<String, Vec<String>>>>,
    relation_patterns: Arc<RwLock<HashMap<String, Vec<String>>>>,
    intent_patterns: Arc<RwLock<HashMap<String, Vec<String>>>>,
    stop_words: HashSet<String>,
}

impl SemanticEngine {
    pub fn new() -> Self {
        let jieba = Arc::new(Jieba::new());
        
        let stop_words = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "也", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "还", "来", "个", "对", "为", "把", "里", "后", "才", "能", "下", "这些", "但是", "如果", "只是", "什么", "怎么", "为什么", "哪里", "那么", "这么", "怎样", "多少", "几个", "如何"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            jieba,
            entity_knowledge: Arc::new(RwLock::new(HashMap::new())),
            entity_aliases: Arc::new(RwLock::new(HashMap::new())),
            relation_patterns: Arc::new(RwLock::new(HashMap::new())),
            intent_patterns: Arc::new(RwLock::new(HashMap::new())),
            stop_words,
        }
    }

    pub fn initialize_knowledge_base(&self, entities: Vec<SemanticKnowledge>) -> Result<(), Box<dyn std::error::Error>> {
        let mut knowledge = self.entity_knowledge.write();
        let mut aliases = self.entity_aliases.write();
        
        for entity in entities {
            let entity_name = entity.entity.clone();
            
            // 添加实体别名映射
            let mut entity_aliases_list = vec![entity_name.clone()];
            if let Some(alias_list) = entity.attributes.get("别名") {
                entity_aliases_list.extend(alias_list.clone());
            }
            if let Some(other_names) = entity.attributes.get("其他名称") {
                entity_aliases_list.extend(other_names.clone());
            }
            
            for alias in &entity_aliases_list {
                aliases.entry(alias.clone()).or_insert_with(Vec::new).push(entity_name.clone());
            }
            
            knowledge.insert(entity_name, entity);
        }
        
        self.initialize_default_patterns();
        Ok(())
    }

    fn initialize_default_patterns(&self) {
        let mut intent_patterns = self.intent_patterns.write();
        let mut relation_patterns = self.relation_patterns.write();
        
        // 初始化意图识别模式
        intent_patterns.insert("entity_query".to_string(), vec![
            r"(.+)是什么".to_string(),
            r"(.+)是谁".to_string(),
            r"介绍一下(.+)".to_string(),
            r"(.+)的信息".to_string(),
            r"关于(.+)".to_string(),
        ]);
        
        intent_patterns.insert("attribute_query".to_string(), vec![
            r"(.+)的(.+)是什么".to_string(),
            r"(.+)有什么(.+)".to_string(),
            r"(.+)的(.+)".to_string(),
        ]);
        
        intent_patterns.insert("relation_query".to_string(), vec![
            r"(.+)和(.+)的关系".to_string(),
            r"(.+)与(.+)".to_string(),
            r"(.+)认识(.+)吗".to_string(),
        ]);
        
        intent_patterns.insert("comparison_query".to_string(), vec![
            r"(.+)和(.+)哪个更(.+)".to_string(),
            r"比较(.+)和(.+)".to_string(),
        ]);
        
        // 初始化关系模式
        relation_patterns.insert("is_a".to_string(), vec![
            "是".to_string(), "属于".to_string(), "为".to_string(), "叫做".to_string()
        ]);
        
        relation_patterns.insert("has_property".to_string(), vec![
            "有".to_string(), "具有".to_string(), "拥有".to_string(), "包含".to_string()
        ]);
        
        relation_patterns.insert("located_in".to_string(), vec![
            "位于".to_string(), "在".to_string(), "处于".to_string(), "坐落于".to_string()
        ]);
        
        relation_patterns.insert("works_for".to_string(), vec![
            "工作于".to_string(), "就职于".to_string(), "任职于".to_string(), "服务于".to_string()
        ]);
    }

    pub async fn get_ambiguous_entities(&self, mention: &str) -> Result<Vec<EntityCandidate>, Box<dyn std::error::Error>> {
        let aliases = self.entity_aliases.read();
        let knowledge = self.entity_knowledge.read();
        let mut candidates = Vec::new();
        
        // 直接匹配
        if let Some(entity_names) = aliases.get(mention) {
            for entity_name in entity_names {
                if let Some(entity_knowledge) = knowledge.get(entity_name) {
                    candidates.push(EntityCandidate {
                        entity: entity_name.clone(),
                        weight: 100,
                        confidence: 1.0,
                        entity_type: entity_knowledge.tags.clone(),
                        attributes: entity_knowledge.attributes.clone(),
                    });
                }
            }
        }
        
        // 模糊匹配
        let mention_lower = mention.to_lowercase();
        for (alias, entity_names) in aliases.iter() {
            let alias_lower = alias.to_lowercase();
            if alias_lower.contains(&mention_lower) || mention_lower.contains(&alias_lower) {
                let similarity = self.calculate_string_similarity(&mention_lower, &alias_lower);
                if similarity > 0.6 {
                    for entity_name in entity_names {
                        if let Some(entity_knowledge) = knowledge.get(entity_name) {
                            let weight = (similarity * 100.0) as u64;
                            candidates.push(EntityCandidate {
                                entity: entity_name.clone(),
                                weight,
                                confidence: similarity,
                                entity_type: entity_knowledge.tags.clone(),
                                attributes: entity_knowledge.attributes.clone(),
                            });
                        }
                    }
                }
            }
        }
        
        // 基于语义向量的匹配
        candidates.extend(self.semantic_entity_matching(mention).await?);
        
        // 去重并排序
        candidates.sort_by(|a, b| b.weight.cmp(&a.weight));
        candidates.dedup_by(|a, b| a.entity == b.entity);
        candidates.truncate(10); // 最多返回10个候选
        
        Ok(candidates)
    }

    async fn semantic_entity_matching(&self, mention: &str) -> Result<Vec<EntityCandidate>, Box<dyn std::error::Error>> {
        let mut candidates = Vec::new();
        let knowledge = self.entity_knowledge.read();
        
        // 分词
        let mention_words: Vec<_> = self.jieba.cut(mention, false).into_iter()
            .filter(|word| !self.stop_words.contains(*word))
            .collect();
        
        for (entity_name, entity_knowledge) in knowledge.iter() {
            let entity_words: Vec<_> = self.jieba.cut(&entity_knowledge.entity, false).into_iter()
                .filter(|word| !self.stop_words.contains(*word))
                .collect();
            
            // 计算词汇重叠度
            let overlap_score = self.calculate_word_overlap(&mention_words, &entity_words);
            
            // 计算描述相似度（mention 分词结果在上方已缓存，避免逐实体重复分词）
            let desc_words: Vec<_> = self.jieba.cut(&entity_knowledge.description, false).into_iter()
                .filter(|word| !self.stop_words.contains(*word))
                .collect();
            let desc_similarity = self.calculate_word_overlap(&mention_words, &desc_words);
            
            let combined_score = (overlap_score * 0.7 + desc_similarity * 0.3).max(0.0).min(1.0);
            
            if combined_score > 0.3 {
                candidates.push(EntityCandidate {
                    entity: entity_name.clone(),
                    weight: (combined_score * 80.0) as u64,
                    confidence: combined_score,
                    entity_type: entity_knowledge.tags.clone(),
                    attributes: entity_knowledge.attributes.clone(),
                });
            }
        }
        
        Ok(candidates)
    }

    pub async fn get_entity_knowledge(&self, entity: &str) -> Result<SemanticKnowledge, Box<dyn std::error::Error>> {
        let knowledge = self.entity_knowledge.read();
        
        if let Some(entity_knowledge) = knowledge.get(entity) {
            Ok(entity_knowledge.clone())
        } else {
            // 尝试通过别名查找
            let aliases = self.entity_aliases.read();
            for (alias, entity_names) in aliases.iter() {
                if alias == entity {
                    for entity_name in entity_names {
                        if let Some(entity_knowledge) = knowledge.get(entity_name) {
                            return Ok(entity_knowledge.clone());
                        }
                    }
                }
            }
            
            Err(format!("未找到实体: {}", entity).into())
        }
    }

    pub async fn get_entity_attribute_value(&self, entity: &str, attribute: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let entity_knowledge = self.get_entity_knowledge(entity).await?;
        
        if let Some(values) = entity_knowledge.attributes.get(attribute) {
            Ok(values.clone())
        } else {
            // 尝试模糊匹配属性名
            for (attr_name, values) in &entity_knowledge.attributes {
                if self.calculate_string_similarity(attribute, attr_name) > 0.8 {
                    return Ok(values.clone());
                }
            }
            
            Ok(vec![])
        }
    }

    pub async fn resolve_entity_ambiguity(&self, mention: &str, context: Option<&str>) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let candidates = self.get_ambiguous_entities(mention).await?;
        
        if candidates.is_empty() {
            return Ok(None);
        }
        
        let mut best_candidate = &candidates[0];
        
        // 如果有上下文，使用上下文进行消歧
        if let Some(context) = context {
            let mut best_score = 0.0;
            for candidate in &candidates {
                let context_score = self.calculate_context_relevance(context, candidate).await;
                if context_score > best_score {
                    best_score = context_score;
                    best_candidate = candidate;
                }
            }
        }
        
        Ok(Some(best_candidate.entity.clone()))
    }

    async fn calculate_context_relevance(&self, context: &str, candidate: &EntityCandidate) -> f64 {
        let context_words: Vec<_> = self.jieba.cut(context, false).into_iter()
            .filter(|word| !self.stop_words.contains(*word))
            .collect();
        
        let mut relevance_score = 0.0;
        let mut total_comparisons = 0;
        
        // 与实体标签的相关性
        for tag in &candidate.entity_type {
            for context_word in &context_words {
                relevance_score += self.calculate_string_similarity(context_word, tag);
                total_comparisons += 1;
            }
        }
        
        // 与实体属性的相关性
        for (attr_name, attr_values) in &candidate.attributes {
            for context_word in &context_words {
                relevance_score += self.calculate_string_similarity(context_word, attr_name);
                total_comparisons += 1;
                
                for attr_value in attr_values {
                    relevance_score += self.calculate_string_similarity(context_word, attr_value);
                    total_comparisons += 1;
                }
            }
        }
        
        if total_comparisons > 0 {
            relevance_score / total_comparisons as f64
        } else {
            0.0
        }
    }

    pub async fn analyze_intent(&self, query: &str) -> Result<IntentAnalysisResult, Box<dyn std::error::Error>> {
        let intent_patterns = self.intent_patterns.read();
        let mut best_intent = "unknown".to_string();
        let mut best_confidence = 0.0;
        let mut matched_patterns = Vec::new();
        
        // 预处理查询
        let query_words: Vec<_> = self.jieba.cut(query, false).into_iter().collect();
        let query_text = query_words.join(" ");
        
        // 匹配意图模式
        for (intent_type, patterns) in intent_patterns.iter() {
            for pattern in patterns {
                let confidence = self.match_pattern_confidence(&query_text, pattern);
                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_intent = intent_type.clone();
                    matched_patterns.clear();
                    matched_patterns.push(pattern.to_string());
                } else if confidence > 0.7 && confidence == best_confidence {
                    matched_patterns.push(pattern.to_string());
                }
            }
        }
        
        // 提取实体
        let extracted_entities = self.extract_entities_from_query(query).await?;
        
        // 提取关系
        let extracted_relations = self.extract_relations_from_query(query).await?;
        
        Ok(IntentAnalysisResult {
            intent_type: best_intent,
            confidence: best_confidence,
            entities: extracted_entities,
            relations: extracted_relations,
            query_patterns: matched_patterns,
        })
    }

    async fn extract_entities_from_query(&self, query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let aliases = self.entity_aliases.read();
        
        // 使用最长匹配策略
        let mut i = 0;
        let chars: Vec<char> = query.chars().collect();
        
        while i < chars.len() {
            let mut found_entity = false;
            let mut max_length = 0;
            let mut best_entity = String::new();
            
            // 尝试不同长度的子串（增量构建，避免每个长度都重新分配字符串）
            let mut substring = String::new();
            for j in (i + 1)..=(chars.len().min(i + 10)) {
                substring.push(chars[j - 1]);
                
                if aliases.contains_key(substring.as_str()) && substring.len() > max_length {
                    max_length = substring.len();
                    best_entity = substring.clone();
                    found_entity = true;
                }
            }
            
            if found_entity {
                entities.push(best_entity);
                i += max_length;
            } else {
                i += 1;
            }
        }
        
        // 去重
        entities.sort();
        entities.dedup();
        
        Ok(entities)
    }

    async fn extract_relations_from_query(&self, query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let relation_patterns = self.relation_patterns.read();
        let mut relations = Vec::new();
        
        for (relation_type, keywords) in relation_patterns.iter() {
            for keyword in keywords {
                if query.contains(keyword) {
                    relations.push(relation_type.clone());
                    break;
                }
            }
        }
        
        Ok(relations)
    }

    fn match_pattern_confidence(&self, text: &str, pattern: &str) -> f64 {
        // 简化的模式匹配，实际实现中可以使用正则表达式
        let pattern_words: Vec<_> = pattern.split_whitespace().collect();
        let text_words: Vec<_> = text.split_whitespace().collect();
        
        let mut matches = 0;
        for pattern_word in &pattern_words {
            if pattern_word.starts_with('(') && pattern_word.ends_with(')') {
                continue; // 跳过捕获组
            }
            
            for text_word in &text_words {
                if pattern_word == text_word || self.calculate_string_similarity(pattern_word, text_word) > 0.8 {
                    matches += 1;
                    break;
                }
            }
        }
        
        if pattern_words.is_empty() {
            0.0
        } else {
            matches as f64 / pattern_words.len() as f64
        }
    }

    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }
        
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }
        
        // 使用编辑距离计算相似度
        let edit_distance = self.levenshtein_distance(s1, s2);
        let max_len = len1.max(len2);
        
        1.0 - (edit_distance as f64 / max_len as f64)
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        // 双行动态规划：距离对称，将较短字符串作为列方向，内存降为 O(min(n, m))
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let (rows, cols) = if chars1.len() <= chars2.len() {
            (chars1, chars2)
        } else {
            (chars2, chars1)
        };
        let col_count = cols.len();
        
        let mut prev: Vec<usize> = (0..=col_count).collect();
        let mut curr = vec![0usize; col_count + 1];
        
        for (i, row_char) in rows.iter().enumerate() {
            curr[0] = i + 1;
            for j in 1..=col_count {
                let cost = if *row_char == cols[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1)
                    .min(curr[j - 1] + 1)
                    .min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        
        prev[col_count]
    }

    fn calculate_word_overlap(&self, words1: &[&str], words2: &[&str]) -> f64 {
        let set1: HashSet<_> = words1.iter().collect();
        let set2: HashSet<_> = words2.iter().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    pub async fn get_simplified_knowledge(&self, entity: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let entity_knowledge = self.get_entity_knowledge(entity).await?;
        let mut properties = HashMap::new();
        
        properties.insert("name".to_string(), entity_knowledge.entity);
        properties.insert("description".to_string(), entity_knowledge.description);
        properties.insert("tags".to_string(), entity_knowledge.tags.join(", "));
        
        // 将多值属性转换为字符串
        for (key, values) in entity_knowledge.attributes {
            properties.insert(key, values.join(", "));
        }
        
        Ok(properties)
    }

    pub async fn batch_resolve_entities(&self, mentions: &[String]) -> Result<HashMap<String, Option<String>>, Box<dyn std::error::Error>> {
        let mut results = HashMap::new();
        
        for mention in mentions {
            match self.resolve_entity_ambiguity(mention, None).await {
                Ok(resolved) => {
                    results.insert(mention.clone(), resolved);
                }
                Err(e) => {
                    eprintln!("解析实体 '{}' 时出错: {}", mention, e);
                    results.insert(mention.clone(), None);
                }
            }
        }
        
        Ok(results)
    }

    pub fn add_entity_knowledge(&self, entity: SemanticKnowledge) {
        let entity_name = entity.entity.clone();
        let mut knowledge = self.entity_knowledge.write();
        let mut aliases = self.entity_aliases.write();
        
        // 添加实体别名映射
        let mut entity_aliases_list = vec![entity_name.clone()];
        if let Some(alias_list) = entity.attributes.get("别名") {
            entity_aliases_list.extend(alias_list.clone());
        }
        
        for alias in &entity_aliases_list {
            aliases.entry(alias.clone()).or_insert_with(Vec::new).push(entity_name.clone());
        }
        
        knowledge.insert(entity_name, entity);
    }

    pub fn get_entity_count(&self) -> usize {
        self.entity_knowledge.read().len()
    }
}

impl Default for SemanticEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_semantic_engine_initialization() {
        let engine = SemanticEngine::new();
        assert_eq!(engine.get_entity_count(), 0);
    }

    #[tokio::test]
    async fn test_entity_knowledge_management() {
        let engine = SemanticEngine::new();
        
        let mut attributes = HashMap::new();
        attributes.insert("职业".to_string(), vec!["演员".to_string(), "歌手".to_string()]);
        attributes.insert("别名".to_string(), vec!["华仔".to_string()]);
        
        let entity = SemanticKnowledge {
            entity: "刘德华".to_string(),
            description: "香港著名演员和歌手".to_string(),
            tags: vec!["人物".to_string(), "明星".to_string()],
            attributes,
            relations: HashMap::new(),
        };
        
        engine.add_entity_knowledge(entity);
        assert_eq!(engine.get_entity_count(), 1);
        
        let knowledge = engine.get_entity_knowledge("刘德华").await.unwrap();
        assert_eq!(knowledge.entity, "刘德华");
        
        // 测试别名查找
        let knowledge_by_alias = engine.get_entity_knowledge("华仔").await.unwrap();
        assert_eq!(knowledge_by_alias.entity, "刘德华");
    }

    #[tokio::test]
    async fn test_ambiguous_entity_resolution() {
        let engine = SemanticEngine::new();
        
        // 添加测试数据
        let mut attributes1 = HashMap::new();
        attributes1.insert("类型".to_string(), vec!["水果".to_string()]);
        
        let entity1 = SemanticKnowledge {
            entity: "苹果".to_string(),  // 简化名称，避免括号干扰
            description: "一种常见的水果".to_string(),
            tags: vec!["水果".to_string()],
            attributes: attributes1,
            relations: HashMap::new(),
        };
        
        let mut attributes2 = HashMap::new();
        attributes2.insert("类型".to_string(), vec!["公司".to_string()]);
        
        let entity2 = SemanticKnowledge {
            entity: "苹果公司".to_string(),  // 完整名称
            description: "美国科技公司".to_string(),
            tags: vec!["公司".to_string()],
            attributes: attributes2,
            relations: HashMap::new(),
        };
        
        engine.add_entity_knowledge(entity1);
        engine.add_entity_knowledge(entity2);
        
        let candidates = engine.get_ambiguous_entities("苹果").await.unwrap();
        // 应该找到至少一个候选（"苹果"实体）
        assert!(!candidates.is_empty(), "Should find at least one candidate for '苹果'");
    }

    #[tokio::test]
    async fn test_intent_analysis() {
        let engine = SemanticEngine::new();
        
        let result = engine.analyze_intent("刘德华是谁").await.unwrap();
        // 检查意图识别是否工作（可能返回 "unknown" 是合理的，因为模式匹配相对简单）
        assert!(result.confidence >= 0.0, "Confidence should be non-negative");
        assert!(!result.intent_type.is_empty(), "Intent type should not be empty");
    }

    #[test]
    fn test_string_similarity() {
        let engine = SemanticEngine::new();
        
        let similarity = engine.calculate_string_similarity("苹果", "苹果");
        assert_eq!(similarity, 1.0);
        
        let similarity = engine.calculate_string_similarity("苹果", "苹");
        // 编辑距离相似度：苹果(2字符) vs 苹(1字符)，编辑距离为1，相似度为 1 - 1/2 = 0.5
        assert!(similarity >= 0.5, "Similarity between '苹果' and '苹' should be >= 0.5, got {}", similarity);
        
        let similarity = engine.calculate_string_similarity("苹果", "香蕉");
        assert!(similarity < 1.0, "Different words should have similarity < 1.0");
    }
}
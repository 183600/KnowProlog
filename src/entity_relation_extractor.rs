use jieba_rs::Jieba;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtraction {
    pub entity: String,
    pub entity_type: String,
    pub confidence: f64,
    pub position: EntityPosition,
    pub attributes: HashMap<String, Vec<String>>,
    pub relations: Vec<EntityRelation>,
    pub canonical_form: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPosition {
    pub start: usize,
    pub end: usize,
    pub sentence_id: usize,
    pub paragraph_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    pub relation_type: String,
    pub target_entity: String,
    pub confidence: f64,
    pub evidence: Vec<String>,
    pub direction: RelationDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationDirection {
    Forward,   // 主语 -> 谓语 -> 宾语
    Backward,  // 宾语 <- 谓语 <- 主语
    Bidirectional, // 双向关系
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source_text: String,
    pub extraction_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionPattern {
    pub name: String,
    pub pattern: String,
    pub relation_type: String,
    pub subject_group: usize,
    pub object_group: usize,
    pub predicate_template: String,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    pub name: String,
    pub aliases: Vec<String>,
    pub parent_types: Vec<String>,
    pub properties: Vec<String>,
    pub recognition_patterns: Vec<String>,
    pub context_keywords: Vec<String>,
}

pub struct EntityRelationExtractor {
    jieba: &'static Jieba,
    entity_types: Arc<RwLock<HashMap<String, EntityType>>>,
    extraction_patterns: Arc<RwLock<Vec<ExtractionPattern>>>,
    named_entity_dict: Arc<RwLock<HashMap<String, Vec<String>>>>,
    relation_keywords: Arc<RwLock<HashMap<String, Vec<String>>>>,
    stop_words: HashSet<String>,
    dependency_patterns: Arc<RwLock<Vec<DependencyPattern>>>,
    regex_cache: Arc<RwLock<HashMap<String, Regex>>>,
}

#[derive(Debug, Clone)]
pub struct DependencyPattern {
    pub pattern_name: String,
    pub subject_pos: Vec<String>,
    pub verb_pos: Vec<String>,
    pub object_pos: Vec<String>,
    pub relation_mapping: HashMap<String, String>,
}

impl EntityRelationExtractor {
    pub fn new() -> Self {
        let jieba = crate::nlp::shared_jieba();
        
        let stop_words = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "也", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "还", "来", "个", "对", "为", "把", "里", "后", "才", "能", "下", "这些", "但是", "如果", "只是", "什么", "怎么", "为什么", "哪里", "那么", "这么", "怎样", "多少", "几个", "如何", "这个", "那个", "这样", "那样", "因为", "所以", "然后", "接着", "最后", "首先", "其次", "另外", "此外", "而且", "并且", "或者", "还是", "虽然", "但是", "可是", "然而", "不过", "只要", "除非", "无论", "不管", "即使", "哪怕", "只有", "只是", "仅仅", "刚刚", "正在", "已经", "曾经", "将要", "可能", "应该", "必须", "需要", "想要", "希望", "愿意", "喜欢", "讨厌", "觉得", "认为", "以为", "知道", "了解", "明白", "理解", "发现", "意识"
        ].iter().map(|s| s.to_string()).collect();

        let extractor = Self {
            jieba,
            entity_types: Arc::new(RwLock::new(HashMap::new())),
            extraction_patterns: Arc::new(RwLock::new(Vec::new())),
            named_entity_dict: Arc::new(RwLock::new(HashMap::new())),
            relation_keywords: Arc::new(RwLock::new(HashMap::new())),
            stop_words,
            dependency_patterns: Arc::new(RwLock::new(Vec::new())),
            regex_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        extractor.initialize_default_types();
        extractor.initialize_extraction_patterns();
        extractor.initialize_relation_keywords();
        extractor.initialize_dependency_patterns();
        
        extractor
    }

    fn initialize_default_types(&self) {
        let mut types = self.entity_types.write();
        
        // 人物类型
        types.insert("PERSON".to_string(), EntityType {
            name: "PERSON".to_string(),
            aliases: vec!["人物".to_string(), "人".to_string(), "个人".to_string()],
            parent_types: vec!["ENTITY".to_string()],
            properties: vec![
                "姓名".to_string(), "年龄".to_string(), "职业".to_string(), "出生地".to_string(),
                "教育背景".to_string(), "工作单位".to_string(), "家庭关系".to_string(),
            ],
            recognition_patterns: vec![
                r"[王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董潘袁蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤].{1,3}".to_string(),
            ],
            context_keywords: vec![
                "先生".to_string(), "女士".to_string(), "老师".to_string(), "教授".to_string(),
                "博士".to_string(), "演员".to_string(), "歌手".to_string(), "导演".to_string(),
            ],
        });

        // 组织类型
        types.insert("ORGANIZATION".to_string(), EntityType {
            name: "ORGANIZATION".to_string(),
            aliases: vec!["组织".to_string(), "机构".to_string(), "单位".to_string()],
            parent_types: vec!["ENTITY".to_string()],
            properties: vec![
                "名称".to_string(), "类型".to_string(), "成立时间".to_string(), "总部".to_string(),
                "规模".to_string(), "业务范围".to_string(), "领导人".to_string(),
            ],
            recognition_patterns: vec![
                r".+公司".to_string(), r".+集团".to_string(), r".+学院".to_string(),
                r".+大学".to_string(), r".+研究所".to_string(), r".+医院".to_string(),
            ],
            context_keywords: vec![
                "公司".to_string(), "集团".to_string(), "企业".to_string(), "学校".to_string(),
                "大学".to_string(), "医院".to_string(), "银行".to_string(),
            ],
        });

        // 地点类型
        types.insert("LOCATION".to_string(), EntityType {
            name: "LOCATION".to_string(),
            aliases: vec!["地点".to_string(), "位置".to_string(), "地方".to_string()],
            parent_types: vec!["ENTITY".to_string()],
            properties: vec![
                "名称".to_string(), "类型".to_string(), "位置".to_string(), "面积".to_string(),
                "人口".to_string(), "气候".to_string(), "特产".to_string(),
            ],
            recognition_patterns: vec![
                r".+省".to_string(), r".+市".to_string(), r".+县".to_string(),
                r".+区".to_string(), r".+镇".to_string(), r".+村".to_string(),
            ],
            context_keywords: vec![
                "省".to_string(), "市".to_string(), "县".to_string(), "区".to_string(),
                "位于".to_string(), "坐落".to_string(), "地处".to_string(),
            ],
        });

        // 概念类型
        types.insert("CONCEPT".to_string(), EntityType {
            name: "CONCEPT".to_string(),
            aliases: vec!["概念".to_string(), "想法".to_string(), "理念".to_string()],
            parent_types: vec!["ENTITY".to_string()],
            properties: vec![
                "定义".to_string(), "特征".to_string(), "分类".to_string(), "应用".to_string(),
                "相关概念".to_string(), "历史".to_string(),
            ],
            recognition_patterns: vec![
                r".+主义".to_string(), r".+理论".to_string(), r".+方法".to_string(),
                r".+技术".to_string(), r".+系统".to_string(),
            ],
            context_keywords: vec![
                "理论".to_string(), "概念".to_string(), "方法".to_string(), "技术".to_string(),
                "系统".to_string(), "原理".to_string(), "思想".to_string(),
            ],
        });
    }

    fn initialize_extraction_patterns(&self) {
        let mut patterns = self.extraction_patterns.write();
        
        // 基本关系模式
        patterns.extend(vec![
            ExtractionPattern {
                name: "is_a_pattern".to_string(),
                pattern: r"(.+?)是(.+)".to_string(),
                relation_type: "is_a".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "是".to_string(),
                examples: vec!["刘德华是演员".to_string(), "苹果是水果".to_string()],
            },
            
            ExtractionPattern {
                name: "born_in_pattern".to_string(),
                pattern: r"(.+?)出生于(.+)".to_string(),
                relation_type: "born_in".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "出生于".to_string(),
                examples: vec!["刘德华出生于香港".to_string()],
            },
            
            ExtractionPattern {
                name: "work_for_pattern".to_string(),
                pattern: r"(.+?)在(.+?)工作".to_string(),
                relation_type: "work_for".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "工作于".to_string(),
                examples: vec!["张三在苹果公司工作".to_string()],
            },
            
            ExtractionPattern {
                name: "located_in_pattern".to_string(),
                pattern: r"(.+?)位于(.+)".to_string(),
                relation_type: "located_in".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "位于".to_string(),
                examples: vec!["北京大学位于北京".to_string()],
            },
            
            ExtractionPattern {
                name: "founded_by_pattern".to_string(),
                pattern: r"(.+?)由(.+?)创立".to_string(),
                relation_type: "founded_by".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "由...创立".to_string(),
                examples: vec!["苹果公司由乔布斯创立".to_string()],
            },
            
            ExtractionPattern {
                name: "has_property_pattern".to_string(),
                pattern: r"(.+?)的(.+?)是(.+)".to_string(),
                relation_type: "has_property".to_string(),
                subject_group: 1,
                object_group: 3,
                predicate_template: "的{2}是".to_string(),
                examples: vec!["刘德华的年龄是60岁".to_string()],
            },
            
            ExtractionPattern {
                name: "member_of_pattern".to_string(),
                pattern: r"(.+?)是(.+?)的成员".to_string(),
                relation_type: "member_of".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "是...的成员".to_string(),
                examples: vec!["张三是研发团队的成员".to_string()],
            },
            
            ExtractionPattern {
                name: "owns_pattern".to_string(),
                pattern: r"(.+?)拥有(.+)".to_string(),
                relation_type: "owns".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "拥有".to_string(),
                examples: vec!["公司拥有多项专利".to_string()],
            },
            
            ExtractionPattern {
                name: "married_to_pattern".to_string(),
                pattern: r"(.+?)与(.+?)结婚".to_string(),
                relation_type: "married_to".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "与...结婚".to_string(),
                examples: vec!["张三与李四结婚".to_string()],
            },
            
            ExtractionPattern {
                name: "graduated_from_pattern".to_string(),
                pattern: r"(.+?)毕业于(.+)".to_string(),
                relation_type: "graduated_from".to_string(),
                subject_group: 1,
                object_group: 2,
                predicate_template: "毕业于".to_string(),
                examples: vec!["张三毕业于清华大学".to_string()],
            },
        ]);
    }

    fn initialize_relation_keywords(&self) {
        let mut keywords = self.relation_keywords.write();
        
        keywords.insert("is_a".to_string(), vec![
            "是".to_string(), "为".to_string(), "属于".to_string(), "作为".to_string(),
        ]);
        
        keywords.insert("work_for".to_string(), vec![
            "工作".to_string(), "就职".to_string(), "任职".to_string(), "服务".to_string(),
            "供职".to_string(), "受雇".to_string(),
        ]);
        
        keywords.insert("located_in".to_string(), vec![
            "位于".to_string(), "在".to_string(), "坐落".to_string(), "地处".to_string(),
            "处于".to_string(), "设在".to_string(),
        ]);
        
        keywords.insert("has_property".to_string(), vec![
            "有".to_string(), "具有".to_string(), "拥有".to_string(), "包含".to_string(),
            "含有".to_string(), "带有".to_string(),
        ]);
        
        keywords.insert("born_in".to_string(), vec![
            "出生".to_string(), "诞生".to_string(), "生于".to_string(),
        ]);
        
        keywords.insert("founded_by".to_string(), vec![
            "创立".to_string(), "创建".to_string(), "建立".to_string(), "成立".to_string(),
            "创办".to_string(), "设立".to_string(),
        ]);
    }

    fn initialize_dependency_patterns(&self) {
        let mut patterns = self.dependency_patterns.write();
        
        patterns.extend(vec![
            DependencyPattern {
                pattern_name: "subject_verb_object".to_string(),
                subject_pos: vec!["n".to_string(), "nr".to_string(), "ns".to_string()],
                verb_pos: vec!["v".to_string(), "vn".to_string()],
                object_pos: vec!["n".to_string(), "nr".to_string(), "ns".to_string()],
                relation_mapping: [
                    ("是".to_string(), "is_a".to_string()),
                    ("工作".to_string(), "work_for".to_string()),
                    ("位于".to_string(), "located_in".to_string()),
                ].iter().cloned().collect(),
            },
        ]);
    }

    pub async fn extract_entities_and_relations(&self, text: &str) -> Result<(Vec<EntityExtraction>, Vec<KnowledgeTriple>), Box<dyn std::error::Error>> {
        // 文本预处理
        let sentences = self.split_into_sentences(text);
        
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        
        for (sentence_id, sentence) in sentences.iter().enumerate() {
            // 实体抽取
            let sentence_entities = self.extract_entities_from_sentence(sentence, sentence_id).await?;
            
            // 关系抽取
            let sentence_relations = self.extract_relations_from_sentence(sentence, &sentence_entities).await?;
            
            all_entities.extend(sentence_entities);
            all_relations.extend(sentence_relations);
        }
        
        // 实体去重和合并
        let merged_entities = self.merge_duplicate_entities(all_entities);
        
        // 实体链接和规范化
        let linked_entities = self.link_and_normalize_entities(merged_entities).await?;
        
        // 关系验证和过滤
        let validated_relations = self.validate_relations(all_relations, &linked_entities).await?;
        
        Ok((linked_entities, validated_relations))
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // 简单的句子分割，基于标点符号
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();
        
        for ch in text.chars() {
            current_sentence.push(ch);
            
            if matches!(ch, '。' | '！' | '？' | '；' | '\n') {
                let sentence = current_sentence.trim().to_string();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                current_sentence.clear();
            }
        }
        
        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
        
        sentences
    }

    async fn extract_entities_from_sentence(&self, sentence: &str, sentence_id: usize) -> Result<Vec<EntityExtraction>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let entity_types = self.entity_types.read();
        
        // 使用每种实体类型的模式进行匹配
        for (type_name, entity_type) in entity_types.iter() {
            let type_entities = self.extract_entities_by_type(sentence, type_name, entity_type, sentence_id).await?;
            entities.extend(type_entities);
        }
        
        // 基于命名实体词典的匹配
        entities.extend(self.extract_entities_from_dict(sentence, sentence_id).await?);
        
        // 去除重叠的实体
        self.remove_overlapping_entities(&mut entities);
        
        Ok(entities)
    }

    /// 获取缓存的正则
    fn get_compiled_regex(&self, pattern: &str) -> Option<Regex> {
        {
            let cache = self.regex_cache.read();
            if let Some(regex) = cache.get(pattern) {
                return Some(regex.clone());
            }
        }
        
        match Regex::new(pattern) {
            Ok(regex) => {
                self.regex_cache.write().insert(pattern.to_string(), regex.clone());
                Some(regex)
            }
            Err(_) => None,
        }
    }

    async fn extract_entities_by_type(&self, sentence: &str, type_name: &str, entity_type: &EntityType, sentence_id: usize) -> Result<Vec<EntityExtraction>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        
        // 正则模式匹配
        for pattern in &entity_type.recognition_patterns {
            if let Some(regex) = self.get_compiled_regex(pattern) {
                for mat in regex.find_iter(sentence) {
                    let entity_text = mat.as_str();
                    let confidence = self.calculate_entity_confidence(entity_text, entity_type, sentence);
                    
                    if confidence > 0.5 {
                        entities.push(EntityExtraction {
                            entity: entity_text.to_string(),
                            entity_type: type_name.to_string(),
                            confidence,
                            position: EntityPosition {
                                start: mat.start(),
                                end: mat.end(),
                                sentence_id,
                                paragraph_id: 0,
                            },
                            attributes: HashMap::new(),
                            relations: Vec::new(),
                            canonical_form: None,
                        });
                    }
                }
            }
        }
        
        // 上下文关键词匹配
        entities.extend(self.extract_by_context_keywords(sentence, type_name, entity_type, sentence_id).await?);
        
        Ok(entities)
    }

    async fn extract_by_context_keywords(&self, sentence: &str, type_name: &str, entity_type: &EntityType, sentence_id: usize) -> Result<Vec<EntityExtraction>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let words: Vec<_> = self.jieba.cut(sentence, false).into_iter().collect();
        
        // 每个词在句子中的字节偏移
        let mut word_offsets = Vec::with_capacity(words.len() + 1);
        word_offsets.push(0usize);
        for word in &words {
            word_offsets.push(word_offsets.last().copied().unwrap_or(0) + word.len());
        }
        
        // 查找上下文关键词
        for (i, word) in words.iter().enumerate() {
            if entity_type.context_keywords.iter().any(|keyword| keyword.as_str() == *word) {
                // 在关键词附近查找可能的实体
                let search_range = 3; // 前后3个词的范围
                let start_idx = if i >= search_range { i - search_range } else { 0 };
                let end_idx = (i + search_range + 1).min(words.len());
                
                for j in start_idx..end_idx {
                    if j != i && !self.stop_words.contains(words[j]) {
                        entities.push(EntityExtraction {
                            entity: words[j].to_string(),
                            entity_type: type_name.to_string(),
                            confidence: 0.6, // 基于上下文的基础置信度
                            position: EntityPosition {
                                start: word_offsets[j],
                                end: word_offsets[j + 1],
                                sentence_id,
                                paragraph_id: 0,
                            },
                            attributes: HashMap::new(),
                            relations: Vec::new(),
                            canonical_form: None,
                        });
                    }
                }
            }
        }
        
        Ok(entities)
    }

    async fn extract_entities_from_dict(&self, sentence: &str, sentence_id: usize) -> Result<Vec<EntityExtraction>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let named_entity_dict = self.named_entity_dict.read();
        
        // 最长匹配算法
        let chars: Vec<char> = sentence.chars().collect();
        
        // 每个字符对应的字节偏移（EntityPosition 使用字节偏移）
        let mut byte_offsets = Vec::with_capacity(chars.len() + 1);
        byte_offsets.push(0usize);
        for ch in &chars {
            byte_offsets.push(byte_offsets.last().copied().unwrap_or(0) + ch.len_utf8());
        }
        
        let mut i = 0;
        
        while i < chars.len() {
            let mut found_entity = false;
            let mut max_length = 0;
            let mut best_entity = String::new();
            let mut best_types = Vec::new();
            
            // 尝试不同长度的子串，增量构建
            let mut substring = String::new();
            for j in (i + 1)..=(chars.len().min(i + 10)) {
                substring.push(chars[j - 1]);
                
                if let Some(types) = named_entity_dict.get(substring.as_str()) {
                    // i 为字符索引，匹配长度按字符数（j - i）计算
                    if j - i > max_length {
                        max_length = j - i;
                        best_entity = substring.clone();
                        best_types = types.clone();
                        found_entity = true;
                    }
                }
            }
            
            if found_entity {
                for entity_type in best_types {
                    entities.push(EntityExtraction {
                        entity: best_entity.clone(),
                        entity_type,
                        confidence: 0.9, // 词典匹配的高置信度
                        position: EntityPosition {
                            start: byte_offsets[i],
                            end: byte_offsets[i + max_length],
                            sentence_id,
                            paragraph_id: 0,
                        },
                        attributes: HashMap::new(),
                        relations: Vec::new(),
                        canonical_form: None,
                    });
                }
                i += max_length;
            } else {
                i += 1;
            }
        }
        
        Ok(entities)
    }

    fn calculate_entity_confidence(&self, entity_text: &str, entity_type: &EntityType, sentence: &str) -> f64 {
        let mut confidence: f64 = 0.5;
        
        // 检查是否包含类型关键词
        for keyword in &entity_type.context_keywords {
            if sentence.contains(keyword) {
                confidence += 0.2;
            }
        }
        
        // 检查实体文本长度和格式
        if entity_text.len() >= 2 && entity_text.len() <= 10 {
            confidence += 0.1;
        }
        
        // 检查是否包含停用词（降低置信度）
        if self.stop_words.contains(entity_text) {
            confidence -= 0.3;
        }
        
        confidence.max(0.0).min(1.0)
    }

    fn remove_overlapping_entities(&self, entities: &mut Vec<EntityExtraction>) {
        entities.sort_by(|a, b| a.position.start.cmp(&b.position.start));
        
        let mut i = 0;
        while i < entities.len() {
            let mut j = i + 1;
            while j < entities.len() {
                if self.entities_overlap(&entities[i], &entities[j]) {
                    // 保留置信度更高的实体
                    if entities[i].confidence >= entities[j].confidence {
                        entities.remove(j);
                    } else {
                        entities.remove(i);
                        j = i + 1;
                    }
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    fn entities_overlap(&self, entity1: &EntityExtraction, entity2: &EntityExtraction) -> bool {
        !(entity1.position.end <= entity2.position.start || entity2.position.end <= entity1.position.start)
    }

    async fn extract_relations_from_sentence(&self, sentence: &str, entities: &[EntityExtraction]) -> Result<Vec<KnowledgeTriple>, Box<dyn std::error::Error>> {
        let mut relations = Vec::new();
        
        // 基于模式的关系抽取
        relations.extend(self.extract_relations_by_patterns(sentence).await?);
        
        // 基于实体对的关系抽取
        relations.extend(self.extract_relations_between_entities(sentence, entities).await?);
        
        // 基于依存句法的关系抽取
        relations.extend(self.extract_relations_by_dependency(sentence, entities).await?);
        
        Ok(relations)
    }

    async fn extract_relations_by_patterns(&self, sentence: &str) -> Result<Vec<KnowledgeTriple>, Box<dyn std::error::Error>> {
        let mut relations = Vec::new();
        let extraction_patterns = self.extraction_patterns.read();
        
        for pattern in extraction_patterns.iter() {
            if let Some(regex) = self.get_compiled_regex(&pattern.pattern) {
                for captures in regex.captures_iter(sentence) {
                    if let (Some(subject), Some(object)) = (captures.get(pattern.subject_group), captures.get(pattern.object_group)) {
                        let subject_text = subject.as_str().trim().to_string();
                        let object_text = object.as_str().trim().to_string();
                        
                        if !subject_text.is_empty() && !object_text.is_empty() && 
                           !self.stop_words.contains(&subject_text) && 
                           !self.stop_words.contains(&object_text) {
                            
                            relations.push(KnowledgeTriple {
                                subject: subject_text,
                                predicate: pattern.relation_type.clone(),
                                object: object_text,
                                confidence: 0.8,
                                source_text: sentence.to_string(),
                                extraction_method: "pattern_based".to_string(),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(relations)
    }

    async fn extract_relations_between_entities(&self, sentence: &str, entities: &[EntityExtraction]) -> Result<Vec<KnowledgeTriple>, Box<dyn std::error::Error>> {
        let mut relations = Vec::new();
        let relation_keywords = self.relation_keywords.read();
        
        // 遍历实体对
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];
                
                // 查找实体间的关系词
                let text_between = self.extract_text_between_entities(sentence, entity1, entity2);
                
                for (relation_type, keywords) in relation_keywords.iter() {
                    for keyword in keywords {
                        if text_between.contains(keyword) {
                            let confidence = 0.7;
                            
                            relations.push(KnowledgeTriple {
                                subject: entity1.entity.clone(),
                                predicate: relation_type.clone(),
                                object: entity2.entity.clone(),
                                confidence,
                                source_text: sentence.to_string(),
                                extraction_method: "entity_pair_based".to_string(),
                            });
                            
                            // 如果是双向关系，也添加反向关系
                            if self.is_bidirectional_relation(relation_type) {
                                relations.push(KnowledgeTriple {
                                    subject: entity2.entity.clone(),
                                    predicate: relation_type.clone(),
                                    object: entity1.entity.clone(),
                                    confidence,
                                    source_text: sentence.to_string(),
                                    extraction_method: "entity_pair_based_reverse".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(relations)
    }

    fn extract_text_between_entities<'a>(&self, sentence: &'a str, entity1: &EntityExtraction, entity2: &EntityExtraction) -> &'a str {
        let start = entity1.position.end.min(entity2.position.end);
        let end = entity1.position.start.max(entity2.position.start);
        
        if start < end && end <= sentence.len() {
            &sentence[start..end]
        } else {
            ""
        }
    }

    fn is_bidirectional_relation(&self, relation_type: &str) -> bool {
        matches!(relation_type, "married_to" | "friends_with" | "siblings")
    }

    async fn extract_relations_by_dependency(&self, sentence: &str, _entities: &[EntityExtraction]) -> Result<Vec<KnowledgeTriple>, Box<dyn std::error::Error>> {
        let mut relations = Vec::new();
        
        // 简化的依存句法分析
        let words: Vec<_> = self.jieba.cut_for_search(sentence, false).into_iter().collect();
        let pos_tags = self.simple_pos_tagging(&words);
        
        // 寻找主谓宾结构
        for i in 0..words.len() {
            if self.is_verb(&pos_tags[i]) {
                let verb = &words[i];
                
                // 寻找主语
                let subject = self.find_subject(&words, &pos_tags, i);
                
                // 寻找宾语
                let object = self.find_object(&words, &pos_tags, i);
                
                if let (Some(subj), Some(obj)) = (subject, object) {
                    // 将动词映射为关系类型
                    let relation_type = self.map_verb_to_relation(verb);
                    
                    relations.push(KnowledgeTriple {
                        subject: subj,
                        predicate: relation_type,
                        object: obj,
                        confidence: 0.6,
                        source_text: sentence.to_string(),
                        extraction_method: "dependency_based".to_string(),
                    });
                }
            }
        }
        
        Ok(relations)
    }

    fn simple_pos_tagging(&self, words: &[&str]) -> Vec<String> {
        // 简化的词性标注
        words.iter().map(|word| {
            if self.is_noun(word) {
                "n".to_string()
            } else if self.is_verb_word(word) {
                "v".to_string()
            } else if self.is_adjective(word) {
                "a".to_string()
            } else {
                "x".to_string()
            }
        }).collect()
    }

    fn is_noun(&self, word: &str) -> bool {
        // 简单的名词判断规则
        let noun_suffixes = ["公司", "学院", "大学", "医院", "银行", "集团", "省", "市", "县", "区"];
        noun_suffixes.iter().any(|suffix| word.ends_with(suffix))
    }

    fn is_verb_word(&self, word: &str) -> bool {
        // 简单的动词判断规则
        let common_verbs = ["是", "有", "在", "做", "去", "来", "工作", "学习", "生活", "创立", "建立", "位于"];
        common_verbs.contains(&word)
    }

    fn is_adjective(&self, word: &str) -> bool {
        // 简单的形容词判断规则
        let adjective_suffixes = ["的", "性", "化"];
        adjective_suffixes.iter().any(|suffix| word.ends_with(suffix))
    }

    fn is_verb(&self, pos_tag: &str) -> bool {
        pos_tag == "v"
    }

    fn find_subject(&self, words: &[&str], pos_tags: &[String], verb_index: usize) -> Option<String> {
        // 在动词前查找主语
        for i in (0..verb_index).rev() {
            if pos_tags[i] == "n" && !self.stop_words.contains(words[i]) {
                return Some(words[i].to_string());
            }
        }
        None
    }

    fn find_object(&self, words: &[&str], pos_tags: &[String], verb_index: usize) -> Option<String> {
        // 在动词后查找宾语
        for i in (verb_index + 1)..words.len() {
            if pos_tags[i] == "n" && !self.stop_words.contains(words[i]) {
                return Some(words[i].to_string());
            }
        }
        None
    }

    fn map_verb_to_relation(&self, verb: &str) -> String {
        match verb {
            "是" => "is_a".to_string(),
            "有" | "拥有" => "has".to_string(),
            "在" | "位于" => "located_in".to_string(),
            "工作" => "work_for".to_string(),
            "学习" => "study_at".to_string(),
            "创立" | "建立" => "founded_by".to_string(),
            _ => format!("relation_{}", verb),
        }
    }

    fn merge_duplicate_entities(&self, entities: Vec<EntityExtraction>) -> Vec<EntityExtraction> {
        let mut merged: HashMap<String, EntityExtraction> = HashMap::new();
        
        for entity in entities {
            let key = format!("{}_{}", entity.entity, entity.entity_type);
            
            if let Some(existing) = merged.get_mut(&key) {
                // 合并相同实体，保留较高的置信度
                if entity.confidence > existing.confidence {
                    *existing = entity;
                }
            } else {
                merged.insert(key, entity);
            }
        }
        
        merged.into_values().collect()
    }

    async fn link_and_normalize_entities(&self, entities: Vec<EntityExtraction>) -> Result<Vec<EntityExtraction>, Box<dyn std::error::Error>> {
        let mut linked_entities = Vec::new();
        
        for mut entity in entities {
            // 实体规范化
            entity.canonical_form = self.normalize_entity_name(&entity.entity, &entity.entity_type).await;
            
            // 提取实体属性
            entity.attributes = self.extract_entity_attributes(&entity).await?;
            
            linked_entities.push(entity);
        }
        
        Ok(linked_entities)
    }

    async fn normalize_entity_name(&self, entity_name: &str, entity_type: &str) -> Option<String> {
        // 实体名称规范化
        let normalized = entity_name.trim().to_string();
        
        match entity_type {
            "PERSON" => {
                // 人名规范化：去除称谓
                let titles = ["先生", "女士", "老师", "教授", "博士"];
                let mut result = normalized;
                for title in &titles {
                    result = result.replace(title, "");
                }
                Some(result.trim().to_string())
            }
            "ORGANIZATION" => {
                // 机构名称规范化
                Some(normalized)
            }
            "LOCATION" => {
                // 地点名称规范化
                Some(normalized)
            }
            _ => Some(normalized),
        }
    }

    async fn extract_entity_attributes(&self, entity: &EntityExtraction) -> Result<HashMap<String, Vec<String>>, Box<dyn std::error::Error>> {
        let mut attributes = HashMap::new();
        
        // 基于实体类型添加默认属性
        let entity_types = self.entity_types.read();
        if let Some(entity_type) = entity_types.get(&entity.entity_type) {
            for property in &entity_type.properties {
                attributes.insert(property.clone(), vec![]);
            }
        }
        
        Ok(attributes)
    }

    async fn validate_relations(&self, relations: Vec<KnowledgeTriple>, entities: &[EntityExtraction]) -> Result<Vec<KnowledgeTriple>, Box<dyn std::error::Error>> {
        let mut validated_relations = Vec::new();
        let entity_names: HashSet<_> = entities.iter().map(|e| &e.entity).collect();
        
        for relation in relations {
            // 验证主语和宾语是否为识别出的实体
            if entity_names.contains(&relation.subject) && entity_names.contains(&relation.object) {
                // 验证关系类型的合理性
                if self.is_valid_relation(&relation.subject, &relation.predicate, &relation.object, entities) {
                    validated_relations.push(relation);
                }
            }
        }
        
        Ok(validated_relations)
    }

    fn is_valid_relation(&self, subject: &str, predicate: &str, object: &str, entities: &[EntityExtraction]) -> bool {
        // 获取主语和宾语的实体类型
        let subject_type = entities.iter()
            .find(|e| e.entity == subject)
            .map(|e| &e.entity_type);
        
        let object_type = entities.iter()
            .find(|e| e.entity == object)
            .map(|e| &e.entity_type);
        
        // 基于实体类型验证关系的合理性
        match (subject_type.map(|s| s.as_str()), predicate, object_type.map(|s| s.as_str())) {
            (Some("PERSON"), "work_for", Some("ORGANIZATION")) => true,
            (Some("PERSON"), "born_in", Some("LOCATION")) => true,
            (Some("ORGANIZATION"), "located_in", Some("LOCATION")) => true,
            (Some("PERSON"), "graduated_from", Some("ORGANIZATION")) => true,
            (Some("PERSON"), "married_to", Some("PERSON")) => true,
            _ => true, // 默认允许，可以根据需要调整
        }
    }

    pub fn add_entity_type(&self, entity_type: EntityType) {
        let mut types = self.entity_types.write();
        types.insert(entity_type.name.clone(), entity_type);
    }

    pub fn add_extraction_pattern(&self, pattern: ExtractionPattern) {
        let mut patterns = self.extraction_patterns.write();
        patterns.push(pattern);
    }

    pub fn add_named_entity(&self, entity_name: String, entity_types: Vec<String>) {
        let mut dict = self.named_entity_dict.write();
        dict.insert(entity_name, entity_types);
    }

    pub fn get_entity_types(&self) -> Vec<String> {
        self.entity_types.read().keys().cloned().collect()
    }

    pub fn get_extraction_patterns(&self) -> Vec<ExtractionPattern> {
        self.extraction_patterns.read().clone()
    }
}

impl Default for EntityRelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_entity_extraction() {
        let extractor = EntityRelationExtractor::new();
        
        let text = "刘德华是香港著名演员，出生于香港。他毕业于香港演艺学院。";
        let (entities, _) = extractor.extract_entities_and_relations(text).await.unwrap();
        
        assert!(!entities.is_empty());
        
        // 应该提取到人物实体
        let person_entities: Vec<_> = entities.iter()
            .filter(|e| e.entity_type == "PERSON")
            .collect();
        assert!(!person_entities.is_empty());
    }

    #[tokio::test]
    async fn test_relation_extraction() {
        let extractor = EntityRelationExtractor::new();
        
        let text = "苹果公司由史蒂夫·乔布斯创立，总部位于美国加利福尼亚州。";
        let (_, relations) = extractor.extract_entities_and_relations(text).await.unwrap();
        
        assert!(!relations.is_empty());
        
        // 应该提取到创立关系
        let founded_relations: Vec<_> = relations.iter()
            .filter(|r| r.predicate.contains("founded") || r.predicate.contains("创立"))
            .collect();
        assert!(!founded_relations.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let extractor = EntityRelationExtractor::new();
        
        let text = "张三是工程师";
        let relations = extractor.extract_relations_by_patterns(text).await.unwrap();
        
        assert!(!relations.is_empty());
        assert_eq!(relations[0].predicate, "is_a");
    }

    #[test]
    fn test_sentence_splitting() {
        let extractor = EntityRelationExtractor::new();
        
        let text = "这是第一句。这是第二句！这是第三句？";
        let sentences = extractor.split_into_sentences(text);
        
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_entity_overlap_detection() {
        let extractor = EntityRelationExtractor::new();
        
        let entity1 = EntityExtraction {
            entity: "苹果公司".to_string(),
            entity_type: "ORGANIZATION".to_string(),
            confidence: 0.9,
            position: EntityPosition { start: 0, end: 4, sentence_id: 0, paragraph_id: 0 },
            attributes: HashMap::new(),
            relations: Vec::new(),
            canonical_form: None,
        };
        
        let entity2 = EntityExtraction {
            entity: "苹果".to_string(),
            entity_type: "ORGANIZATION".to_string(),
            confidence: 0.7,
            position: EntityPosition { start: 0, end: 2, sentence_id: 0, paragraph_id: 0 },
            attributes: HashMap::new(),
            relations: Vec::new(),
            canonical_form: None,
        };
        
        assert!(extractor.entities_overlap(&entity1, &entity2));
    }
}
use jieba_rs::Jieba;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use parking_lot::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    pub intent: String,
    pub confidence: f64,
    pub entities: Vec<ExtractedEntity>,
    pub slots: HashMap<String, String>,
    pub context: QueryContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f64,
    pub normalized_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub query_type: String,
    pub domain: String,
    pub temporal_info: Option<TemporalInfo>,
    pub spatial_info: Option<SpatialInfo>,
    pub modifiers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    pub time_type: String,
    pub time_value: String,
    pub is_relative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialInfo {
    pub location_type: String,
    pub location_value: String,
    pub is_relative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPattern {
    pub intent: String,
    pub patterns: Vec<String>,
    pub entities: Vec<String>,
    pub required_slots: Vec<String>,
    pub examples: Vec<String>,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPattern {
    pub entity_type: String,
    pub patterns: Vec<String>,
    pub keywords: Vec<String>,
    pub context_clues: Vec<String>,
}

pub struct AdvancedNLU {
    jieba: &'static Jieba,
    intent_patterns: RwLock<HashMap<String, IntentPattern>>,
    entity_patterns: RwLock<HashMap<String, EntityPattern>>,
    domain_keywords: HashMap<String, Vec<String>>,
    temporal_patterns: Vec<(Regex, String)>,
    spatial_patterns: Vec<(Regex, String)>,
    number_patterns: Vec<(Regex, String)>,
    context_memory: RwLock<Vec<QueryContext>>,
    entity_regex_cache: RwLock<HashMap<String, Regex>>,
}

impl AdvancedNLU {
    pub fn new() -> Self {
        let jieba = crate::nlp::shared_jieba();

        Self {
            jieba,
            intent_patterns: RwLock::new(Self::initialize_intent_patterns()),
            entity_patterns: RwLock::new(Self::initialize_entity_patterns()),
            domain_keywords: Self::initialize_domain_keywords(),
            temporal_patterns: Self::initialize_temporal_patterns(),
            spatial_patterns: Self::initialize_spatial_patterns(),
            number_patterns: Self::initialize_number_patterns(),
            context_memory: RwLock::new(Vec::new()),
            entity_regex_cache: RwLock::new(HashMap::new()),
        }
    }

    fn initialize_intent_patterns() -> HashMap<String, IntentPattern> {
        let mut patterns = HashMap::new();
        
        // 实体查询意图
        patterns.insert("entity_query".to_string(), IntentPattern {
            intent: "entity_query".to_string(),
            patterns: vec![
                r"(.+)是什么".to_string(),
                r"(.+)是谁".to_string(),
                r"介绍(.+)".to_string(),
                r"(.+)的信息".to_string(),
                r"关于(.+)".to_string(),
                r"告诉我(.+)".to_string(),
                r"(.+)简介".to_string(),
            ],
            entities: vec!["PERSON".to_string(), "ORGANIZATION".to_string(), "CONCEPT".to_string()],
            required_slots: vec!["entity".to_string()],
            examples: vec![
                "刘德华是谁".to_string(),
                "苹果公司是什么".to_string(),
                "介绍一下北京".to_string(),
            ],
            priority: 10,
        });

        // 属性查询意图
        patterns.insert("attribute_query".to_string(), IntentPattern {
            intent: "attribute_query".to_string(),
            patterns: vec![
                r"(.+)的(.+)是什么".to_string(),
                r"(.+)有什么(.+)".to_string(),
                r"(.+)的(.+)".to_string(),
                r"(.+)(.+)多少".to_string(),
                r"(.+)多(.+)".to_string(),
            ],
            entities: vec!["PERSON".to_string(), "ATTRIBUTE".to_string()],
            required_slots: vec!["entity".to_string(), "attribute".to_string()],
            examples: vec![
                "刘德华的年龄是多少".to_string(),
                "北京的人口".to_string(),
                "苹果公司的CEO".to_string(),
            ],
            priority: 15,
        });

        // 关系查询意图
        patterns.insert("relation_query".to_string(), IntentPattern {
            intent: "relation_query".to_string(),
            patterns: vec![
                r"(.+)和(.+)的关系".to_string(),
                r"(.+)与(.+)".to_string(),
                r"(.+)认识(.+)吗".to_string(),
                r"(.+)是(.+)的(.+)".to_string(),
            ],
            entities: vec!["PERSON".to_string(), "ORGANIZATION".to_string()],
            required_slots: vec!["entity1".to_string(), "entity2".to_string()],
            examples: vec![
                "刘德华和张学友的关系".to_string(),
                "苹果与微软".to_string(),
            ],
            priority: 12,
        });

        // 比较查询意图
        patterns.insert("comparison_query".to_string(), IntentPattern {
            intent: "comparison_query".to_string(),
            patterns: vec![
                r"(.+)和(.+)哪个更(.+)".to_string(),
                r"比较(.+)和(.+)".to_string(),
                r"(.+)与(.+)相比".to_string(),
                r"(.+)(.+)还是(.+)(.+)".to_string(),
            ],
            entities: vec!["PERSON".to_string(), "ORGANIZATION".to_string(), "CONCEPT".to_string()],
            required_slots: vec!["entity1".to_string(), "entity2".to_string(), "aspect".to_string()],
            examples: vec![
                "苹果和华为哪个更好".to_string(),
                "比较北京和上海的经济".to_string(),
            ],
            priority: 8,
        });

        // 列表查询意图
        patterns.insert("list_query".to_string(), IntentPattern {
            intent: "list_query".to_string(),
            patterns: vec![
                r"有哪些(.+)".to_string(),
                r"(.+)有哪些".to_string(),
                r"列出(.+)".to_string(),
                r"所有的(.+)".to_string(),
                r"(.+)包括什么".to_string(),
            ],
            entities: vec!["CATEGORY".to_string(), "ORGANIZATION".to_string()],
            required_slots: vec!["category".to_string()],
            examples: vec![
                "有哪些中国演员".to_string(),
                "苹果公司有哪些产品".to_string(),
            ],
            priority: 9,
        });

        // 计数查询意图
        patterns.insert("count_query".to_string(), IntentPattern {
            intent: "count_query".to_string(),
            patterns: vec![
                r"有多少(.+)".to_string(),
                r"(.+)有多少个".to_string(),
                r"(.+)的数量".to_string(),
                r"一共有几个(.+)".to_string(),
            ],
            entities: vec!["CATEGORY".to_string()],
            required_slots: vec!["category".to_string()],
            examples: vec![
                "中国有多少个省".to_string(),
                "苹果公司有多少员工".to_string(),
            ],
            priority: 7,
        });

        patterns
    }

    fn initialize_entity_patterns() -> HashMap<String, EntityPattern> {
        let mut patterns = HashMap::new();
        
        patterns.insert("PERSON".to_string(), EntityPattern {
            entity_type: "PERSON".to_string(),
            patterns: vec![
                r"[王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董潘袁蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤].{1,3}".to_string(),
            ],
            keywords: vec![
                "先生".to_string(), "女士".to_string(), "老师".to_string(), "教授".to_string(),
                "博士".to_string(), "硕士".to_string(), "院士".to_string(), "专家".to_string(),
                "演员".to_string(), "歌手".to_string(), "导演".to_string(), "作家".to_string(),
                "画家".to_string(), "科学家".to_string(), "企业家".to_string(), "运动员".to_string(),
            ],
            context_clues: vec![
                "出生".to_string(), "年龄".to_string(), "职业".to_string(), "作品".to_string(),
                "获奖".to_string(), "毕业".to_string(), "就读".to_string(),
            ],
        });

        patterns.insert("ORGANIZATION".to_string(), EntityPattern {
            entity_type: "ORGANIZATION".to_string(),
            patterns: vec![
                r".+公司".to_string(),
                r".+集团".to_string(),
                r".+学院".to_string(),
                r".+大学".to_string(),
                r".+研究所".to_string(),
                r".+医院".to_string(),
                r".+银行".to_string(),
            ],
            keywords: vec![
                "公司".to_string(), "集团".to_string(), "企业".to_string(), "机构".to_string(),
                "组织".to_string(), "学校".to_string(), "大学".to_string(), "学院".to_string(),
                "研究所".to_string(), "医院".to_string(), "银行".to_string(),
            ],
            context_clues: vec![
                "成立".to_string(), "总部".to_string(), "员工".to_string(), "业务".to_string(),
                "产品".to_string(), "服务".to_string(), "规模".to_string(),
            ],
        });

        patterns.insert("LOCATION".to_string(), EntityPattern {
            entity_type: "LOCATION".to_string(),
            patterns: vec![
                r".+省".to_string(),
                r".+市".to_string(),
                r".+县".to_string(),
                r".+区".to_string(),
                r".+镇".to_string(),
                r".+村".to_string(),
                r".+国".to_string(),
            ],
            keywords: vec![
                "省".to_string(), "市".to_string(), "县".to_string(), "区".to_string(),
                "镇".to_string(), "村".to_string(), "国家".to_string(), "城市".to_string(),
                "地区".to_string(), "地方".to_string(), "位置".to_string(),
            ],
            context_clues: vec![
                "位于".to_string(), "在".to_string(), "地处".to_string(), "坐落".to_string(),
                "面积".to_string(), "人口".to_string(), "气候".to_string(),
            ],
        });

        patterns.insert("NUMBER".to_string(), EntityPattern {
            entity_type: "NUMBER".to_string(),
            patterns: vec![
                r"\d+".to_string(),
                r"[一二三四五六七八九十百千万亿两]+".to_string(),
                r"\d+\.\d+".to_string(),
            ],
            keywords: vec![
                "个".to_string(), "只".to_string(), "人".to_string(), "年".to_string(),
                "月".to_string(), "日".to_string(), "元".to_string(), "万".to_string(),
                "亿".to_string(), "千".to_string(), "百".to_string(),
            ],
            context_clues: vec![
                "数量".to_string(), "年龄".to_string(), "价格".to_string(), "时间".to_string(),
                "重量".to_string(), "长度".to_string(), "面积".to_string(),
            ],
        });

        patterns
    }

    fn initialize_temporal_patterns() -> Vec<(Regex, String)> {
        vec![
            (Regex::new(r"(\d{4})年").unwrap(), "YEAR".to_string()),
            (Regex::new(r"(\d{1,2})月").unwrap(), "MONTH".to_string()),
            (Regex::new(r"(\d{1,2})日").unwrap(), "DAY".to_string()),
            (Regex::new(r"今天|今日").unwrap(), "TODAY".to_string()),
            (Regex::new(r"明天|明日").unwrap(), "TOMORROW".to_string()),
            (Regex::new(r"昨天|昨日").unwrap(), "YESTERDAY".to_string()),
            (Regex::new(r"现在|目前|当前").unwrap(), "NOW".to_string()),
            (Regex::new(r"以前|过去|之前").unwrap(), "PAST".to_string()),
            (Regex::new(r"将来|未来|以后").unwrap(), "FUTURE".to_string()),
            (Regex::new(r"上周|上个星期").unwrap(), "LAST_WEEK".to_string()),
            (Regex::new(r"下周|下个星期").unwrap(), "NEXT_WEEK".to_string()),
            (Regex::new(r"上个月").unwrap(), "LAST_MONTH".to_string()),
            (Regex::new(r"下个月").unwrap(), "NEXT_MONTH".to_string()),
            (Regex::new(r"去年").unwrap(), "LAST_YEAR".to_string()),
            (Regex::new(r"明年").unwrap(), "NEXT_YEAR".to_string()),
        ]
    }

    fn initialize_spatial_patterns() -> Vec<(Regex, String)> {
        vec![
            (Regex::new(r"这里|此处").unwrap(), "HERE".to_string()),
            (Regex::new(r"那里|彼处").unwrap(), "THERE".to_string()),
            (Regex::new(r"附近|周围|周边").unwrap(), "NEARBY".to_string()),
            (Regex::new(r"远处|远方").unwrap(), "FAR".to_string()),
            (Regex::new(r"上面|上方").unwrap(), "ABOVE".to_string()),
            (Regex::new(r"下面|下方").unwrap(), "BELOW".to_string()),
            (Regex::new(r"左边|左侧").unwrap(), "LEFT".to_string()),
            (Regex::new(r"右边|右侧").unwrap(), "RIGHT".to_string()),
            (Regex::new(r"前面|前方").unwrap(), "FRONT".to_string()),
            (Regex::new(r"后面|后方").unwrap(), "BACK".to_string()),
            (Regex::new(r"中间|中央").unwrap(), "CENTER".to_string()),
        ]
    }

    fn initialize_number_patterns() -> Vec<(Regex, String)> {
        vec![
            (Regex::new(r"(\d+)").unwrap(), "ARABIC_NUMBER".to_string()),
            (Regex::new(r"([一二三四五六七八九十百千万亿两]+)").unwrap(), "CHINESE_NUMBER".to_string()),
            (Regex::new(r"(\d+\.\d+)").unwrap(), "DECIMAL".to_string()),
            (Regex::new(r"第(\d+)").unwrap(), "ORDINAL".to_string()),
            (Regex::new(r"(\d+)%").unwrap(), "PERCENTAGE".to_string()),
        ]
    }

    fn initialize_domain_keywords() -> HashMap<String, Vec<String>> {
        let mut domains = HashMap::new();
        
        domains.insert("ENTERTAINMENT".to_string(), vec![
            "演员".to_string(), "歌手".to_string(), "电影".to_string(), "音乐".to_string(),
            "导演".to_string(), "明星".to_string(), "艺人".to_string(), "作品".to_string(),
            "专辑".to_string(), "电视剧".to_string(), "综艺".to_string(),
        ]);

        domains.insert("TECHNOLOGY".to_string(), vec![
            "公司".to_string(), "科技".to_string(), "产品".to_string(), "软件".to_string(),
            "硬件".to_string(), "系统".to_string(), "平台".to_string(), "应用".to_string(),
            "技术".to_string(), "创新".to_string(), "研发".to_string(),
        ]);

        domains.insert("GEOGRAPHY".to_string(), vec![
            "国家".to_string(), "城市".to_string(), "省份".to_string(), "地区".to_string(),
            "位置".to_string(), "气候".to_string(), "人口".to_string(), "面积".to_string(),
            "经济".to_string(), "文化".to_string(), "历史".to_string(),
        ]);

        domains.insert("EDUCATION".to_string(), vec![
            "学校".to_string(), "大学".to_string(), "学院".to_string(), "专业".to_string(),
            "教授".to_string(), "学生".to_string(), "研究".to_string(), "学术".to_string(),
            "教育".to_string(), "课程".to_string(), "学位".to_string(),
        ]);

        domains
    }

    pub async fn analyze_query(&self, query: &str) -> Result<IntentResult, Box<dyn std::error::Error>> {
        // 预处理查询
        let normalized_query = self.normalize_query(query);
        
        // 意图识别
        let intent_info = self.recognize_intent(&normalized_query).await?;
        
        // 实体抽取
        let entities = self.extract_entities(&normalized_query).await?;
        
        // 槽位填充
        let slots = self.extract_slots(&normalized_query, &intent_info, &entities).await?;
        
        // 上下文分析
        let context = self.analyze_context(&normalized_query, &entities).await?;
        
        // 保存上下文到记忆
        self.update_context_memory(&context);
        
        Ok(IntentResult {
            intent: intent_info.0,
            confidence: intent_info.1,
            entities,
            slots,
            context,
        })
    }

    fn normalize_query(&self, query: &str) -> String {
        // 去除多余空白字符
        let mut normalized = query.trim().to_string();

        // 统一标点符号（仅在确有需要时分配新字符串）
        if normalized.contains('？') {
            normalized = normalized.replace('？', "?");
        }
        if normalized.contains('！') {
            normalized = normalized.replace('！', "!");
        }

        // 转换为小写（但保留中文）
        normalized
    }

    async fn recognize_intent(&self, query: &str) -> Result<(String, f64), Box<dyn std::error::Error>> {
        let mut best_intent = "unknown".to_string();
        let mut best_confidence = 0.0;
        
        // 分词（词长一次算好，供全部模式复用）
        let query_words: Vec<(&str, usize)> = self
            .jieba
            .cut(query, false)
            .into_iter()
            .map(|word| (word, word.chars().count()))
            .collect();
        
        {
            let intent_patterns = self.intent_patterns.read();
            for (intent_name, pattern_info) in intent_patterns.iter() {
                let mut max_pattern_confidence: f64 = 0.0;
                
                for pattern in &pattern_info.patterns {
                    let confidence = self.calculate_pattern_match_confidence(&query_words, pattern);
                    max_pattern_confidence = max_pattern_confidence.max(confidence);
                }
                
                // 考虑优先级权重
                let weighted_confidence = max_pattern_confidence * (pattern_info.priority as f64 / 20.0);
                
                if weighted_confidence > best_confidence {
                    best_confidence = weighted_confidence;
                    best_intent = intent_name.clone();
                }
            }
        }
        
        // 如果置信度太低，尝试基于关键词的模糊匹配
        if best_confidence < 0.3 {
            let fallback_result = self.fallback_intent_recognition(query).await?;
            if fallback_result.1 > best_confidence {
                return Ok(fallback_result);
            }
        }
        
        Ok((best_intent, best_confidence))
    }

    async fn fallback_intent_recognition(&self, query: &str) -> Result<(String, f64), Box<dyn std::error::Error>> {
        let mut intent_scores = HashMap::new();
        
        // 基于关键词推断意图
        if query.contains("是什么") || query.contains("是谁") || query.contains("介绍") {
            intent_scores.insert("entity_query".to_string(), 0.6);
        }
        
        if query.contains("的") && (query.contains("多少") || query.contains("什么")) {
            intent_scores.insert("attribute_query".to_string(), 0.5);
        }
        
        if query.contains("和") && query.contains("关系") {
            intent_scores.insert("relation_query".to_string(), 0.7);
        }
        
        if query.contains("比较") || query.contains("哪个更") {
            intent_scores.insert("comparison_query".to_string(), 0.6);
        }
        
        if query.contains("有哪些") || query.contains("包括") {
            intent_scores.insert("list_query".to_string(), 0.6);
        }
        
        if query.contains("多少") && (query.contains("个") || query.contains("数量")) {
            intent_scores.insert("count_query".to_string(), 0.6);
        }
        
        // 同分时按意图名字典序打破平局，避免 HashMap 迭代顺序导致结果不可复现
        let best = intent_scores.into_iter()
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap()
                    .then_with(|| b.0.cmp(&a.0))
            })
            .unwrap_or(("unknown".to_string(), 0.0));
        
        Ok(best)
    }

    fn calculate_pattern_match_confidence(&self, query_words: &[(&str, usize)], pattern: &str) -> f64 {
        // 简化的模式匹配
        let mut matches = 0;
        let mut total = 0;

        for pattern_word in pattern
            .split_whitespace()
            .filter(|word| !word.starts_with('(') || !word.ends_with(')'))
        {
            total += 1;
            // 模式词长度只算一次，供与该模式下所有查询词的比较复用
            let pattern_len = pattern_word.chars().count();
            for (query_word, query_len) in query_words {
                if pattern_word == *query_word
                    || self.words_similarity_above_08_with_lens(
                        pattern_word,
                        pattern_len,
                        query_word,
                        *query_len,
                    )
                {
                    matches += 1;
                    break;
                }
            }
        }

        if total == 0 {
            0.0
        } else {
            matches as f64 / total as f64
        }
    }

    /// 与 `calculate_word_similarity(...) > 0.8` 完全等价；
    /// 先用长度差下界剪枝（编辑距离 ≥ 长度差），避免无谓的 DP 分配
    fn words_similarity_above_08_with_lens(
        &self,
        word1: &str,
        len1: usize,
        word2: &str,
        len2: usize,
    ) -> bool {
        if len1 == 0 || len2 == 0 {
            return false;
        }
        let max_len = len1.max(len2);
        let min_len = len1.min(len2);
        // 相似度 ≤ 1 - (max-min)/max；当长度差达到 20% 上界时必然不满足 > 0.8
        if (max_len - min_len) * 5 >= max_len {
            return false;
        }
        self.word_similarity_with_lens(word1, len1, word2, len2) > 0.8
    }

    #[cfg(test)]
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        self.word_similarity_with_lens(
            word1,
            word1.chars().count(),
            word2,
            word2.chars().count(),
        )
    }

    fn word_similarity_with_lens(&self, word1: &str, len1: usize, word2: &str, len2: usize) -> f64 {
        if word1 == word2 {
            return 1.0;
        }
        
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }
        
        let distance = self.levenshtein_distance_with_lens(word1, len1, word2, len2);
        let max_len = len1.max(len2);
        
        1.0 - (distance as f64 / max_len as f64)
    }

    fn levenshtein_distance_with_lens(&self, s1: &str, len1: usize, s2: &str, len2: usize) -> usize {
        // 双行动态规划：将较短字符串作为列方向；行方向直接迭代字符，省去一份 Vec 收集
        let (rows, cols) = if len1 <= len2 { (s1, s2) } else { (s2, s1) };
        let cols: Vec<char> = cols.chars().collect();
        let col_count = cols.len();
        
        let mut prev: Vec<usize> = (0..=col_count).collect();
        let mut curr = vec![0usize; col_count + 1];
        
        for (i, row_char) in rows.chars().enumerate() {
            curr[0] = i + 1;
            for j in 1..=col_count {
                let cost = if row_char == cols[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1)
                    .min(curr[j - 1] + 1)
                    .min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        
        prev[col_count]
    }

    async fn extract_entities(&self, query: &str) -> Result<Vec<ExtractedEntity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        {
            let entity_patterns = self.entity_patterns.read();
            // 按类型名排序后抽取，重叠实体的取舍不依赖 HashMap 迭代顺序
            let mut entity_types: Vec<&String> = entity_patterns.keys().collect();
            entity_types.sort();
            for entity_type in entity_types {
                let Some(pattern_info) = entity_patterns.get(entity_type) else {
                    continue;
                };
                let extracted = self.extract_entities_by_type(query, entity_type, pattern_info)?;
                entities.extend(extracted);
            }
        }
        
        // 提取时间实体
        entities.extend(self.extract_temporal_entities(query).await?);
        
        // 提取空间实体
        entities.extend(self.extract_spatial_entities(query).await?);
        
        // 提取数字实体
        entities.extend(self.extract_number_entities(query).await?);
        
        // 去重和排序
        entities.sort_by(|a, b| a.start_pos.cmp(&b.start_pos));
        self.remove_overlapping_entities(&mut entities);
        
        Ok(entities)
    }

    /// 获取缓存的实体识别正则
    fn get_compiled_entity_regex(&self, pattern: &str) -> Option<Regex> {
        {
            let cache = self.entity_regex_cache.read();
            if let Some(regex) = cache.get(pattern) {
                return Some(regex.clone());
            }
        }
        
        match Regex::new(pattern) {
            Ok(regex) => {
                self.entity_regex_cache.write().insert(pattern.to_string(), regex.clone());
                Some(regex)
            }
            Err(_) => None,
        }
    }

    fn extract_entities_by_type(&self, query: &str, entity_type: &str, pattern_info: &EntityPattern) -> Result<Vec<ExtractedEntity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        
        // 使用正则模式匹配
        for pattern in &pattern_info.patterns {
            if let Some(regex) = self.get_compiled_entity_regex(pattern) {
                for mat in regex.find_iter(query) {
                    let text = mat.as_str();
                    let confidence = self.calculate_entity_confidence(text, query, pattern_info);
                    
                    entities.push(ExtractedEntity {
                        text: text.to_string(),
                        entity_type: entity_type.to_string(),
                        start_pos: mat.start(),
                        end_pos: mat.end(),
                        confidence,
                        normalized_value: self.normalize_entity_value(text, entity_type),
                    });
                }
            }
        }
        
        // 基于关键词的实体识别（同一关键词的多次出现都要产出）
        for keyword in &pattern_info.keywords {
            for (start, matched) in query.match_indices(keyword.as_str()) {
                let confidence = 0.7; // 关键词匹配的基础置信度
                
                entities.push(ExtractedEntity {
                    text: matched.to_string(),
                    entity_type: entity_type.to_string(),
                    start_pos: start,
                    end_pos: start + matched.len(),
                    confidence,
                    normalized_value: self.normalize_entity_value(matched, entity_type),
                });
            }
        }
        
        Ok(entities)
    }

    fn calculate_entity_confidence(&self, text: &str, context: &str, pattern_info: &EntityPattern) -> f64 {
        let mut confidence: f64 = 0.5; // 基础置信度
        
        // 检查实体文本是否包含类型关键词
        for keyword in &pattern_info.keywords {
            if text.contains(keyword) {
                confidence += 0.2;
            }
        }
        
        // 上下文线索通常出现在实体以外的句中语境（如「刘德华的年龄」中的「年龄」），须在整句上匹配
        for clue in &pattern_info.context_clues {
            if context.contains(clue) {
                confidence += 0.1;
            }
        }
        
        confidence.min(1.0)
    }

    fn normalize_entity_value(&self, text: &str, entity_type: &str) -> Option<String> {
        match entity_type {
            "NUMBER" => self.normalize_number(text),
            "PERSON" => Some(text.trim().to_string()),
            "ORGANIZATION" => Some(text.trim().to_string()),
            "LOCATION" => Some(text.trim().to_string()),
            _ => Some(text.trim().to_string()),
        }
    }

    fn normalize_number(&self, text: &str) -> Option<String> {
        // 中文数字按位值换算（五十 → 50），阿拉伯数字等保持原样
        match Self::chinese_numeral_to_arabic(text) {
            Some(value) => Some(value.to_string()),
            None => Some(text.to_string()),
        }
    }

    fn chinese_numeral_to_arabic(text: &str) -> Option<u64> {
        fn digit(ch: char) -> Option<u64> {
            match ch {
                '零' | '〇' => Some(0),
                '一' | '壹' => Some(1),
                '二' | '两' | '贰' => Some(2),
                '三' | '叁' => Some(3),
                '四' | '肆' => Some(4),
                '五' | '伍' => Some(5),
                '六' | '陆' => Some(6),
                '七' | '柒' => Some(7),
                '八' | '捌' => Some(8),
                '九' | '玖' => Some(9),
                _ => None,
            }
        }
        fn unit(ch: char) -> Option<u64> {
            match ch {
                '十' | '拾' => Some(10),
                '百' | '佰' => Some(100),
                '千' | '仟' => Some(1000),
                _ => None,
            }
        }
        if text.is_empty() {
            return None;
        }
        let mut total: u64 = 0;
        let mut section: u64 = 0;
        let mut current: u64 = 0;
        for ch in text.chars() {
            if let Some(d) = digit(ch) {
                // 连续数字按位拼接（「一九九八」→ 1998），而非只保留最后一位
                current = current.checked_mul(10)?.checked_add(d)?;
            } else if let Some(u) = unit(ch) {
                if current == 0 {
                    current = 1;
                }
                section = section.checked_add(current.checked_mul(u)?)?;
                current = 0;
            } else if ch == '万' {
                // 与十/百/千 一致：整串只有「万」时省略的前导数字按 1 计
                // （累计值非零时「一万亿」等写法已有前导数值，不能补 1）
                if total == 0 && section == 0 && current == 0 {
                    current = 1;
                }
                section = section.checked_add(current)?;
                total = total.checked_add(section.checked_mul(10000)?)?;
                section = 0;
                current = 0;
            } else if ch == '亿' {
                // 亿作用于其之前累积的整个数值段（如「一万亿」= 1 万亿）
                if total == 0 && section == 0 && current == 0 {
                    current = 1;
                }
                section = section.checked_add(current)?;
                total = total.checked_add(section)?.checked_mul(100_000_000)?;
                section = 0;
                current = 0;
            } else {
                return None;
            }
        }
        total.checked_add(section)?.checked_add(current)
    }

    async fn extract_temporal_entities(&self, query: &str) -> Result<Vec<ExtractedEntity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let temporal_patterns = &self.temporal_patterns;
        
        for (regex, time_type) in temporal_patterns.iter() {
            for mat in regex.find_iter(query) {
                entities.push(ExtractedEntity {
                    text: mat.as_str().to_string(),
                    entity_type: "TEMPORAL".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.9,
                    normalized_value: Some(time_type.clone()),
                });
            }
        }
        
        Ok(entities)
    }

    async fn extract_spatial_entities(&self, query: &str) -> Result<Vec<ExtractedEntity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let spatial_patterns = &self.spatial_patterns;
        
        for (regex, location_type) in spatial_patterns.iter() {
            for mat in regex.find_iter(query) {
                entities.push(ExtractedEntity {
                    text: mat.as_str().to_string(),
                    entity_type: "SPATIAL".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.8,
                    normalized_value: Some(location_type.clone()),
                });
            }
        }
        
        Ok(entities)
    }

    async fn extract_number_entities(&self, query: &str) -> Result<Vec<ExtractedEntity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let number_patterns = &self.number_patterns;
        
        for (regex, _number_type) in number_patterns.iter() {
            for mat in regex.find_iter(query) {
                let text = mat.as_str();
                entities.push(ExtractedEntity {
                    text: text.to_string(),
                    entity_type: "NUMBER".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.95,
                    normalized_value: self.normalize_number(text),
                });
            }
        }
        
        Ok(entities)
    }

    fn remove_overlapping_entities(&self, entities: &mut Vec<ExtractedEntity>) {
        let mut i = 0;
        while i < entities.len() {
            let mut j = i + 1;
            while j < entities.len() {
                if entities[i].end_pos <= entities[j].start_pos {
                    // 列表按起点升序：i 完全位于 j 之前，后续实体更不可能与 i 重叠
                    break;
                }
                if self.entities_overlap(&entities[i], &entities[j]) {
                    // 嵌套时保留更完整的复合实体，否则保留置信度更高的实体
                    let i_contains_j = entities[i].start_pos <= entities[j].start_pos
                        && entities[i].end_pos >= entities[j].end_pos;
                    let j_contains_i = entities[j].start_pos <= entities[i].start_pos
                        && entities[j].end_pos >= entities[i].end_pos;
                    let keep_i = if i_contains_j && !j_contains_i {
                        true
                    } else if j_contains_i && !i_contains_j {
                        false
                    } else {
                        entities[i].confidence >= entities[j].confidence
                    };
                    if keep_i {
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

    fn entities_overlap(&self, entity1: &ExtractedEntity, entity2: &ExtractedEntity) -> bool {
        !(entity1.end_pos <= entity2.start_pos || entity2.end_pos <= entity1.start_pos)
    }

    async fn extract_slots(&self, query: &str, intent_info: &(String, f64), entities: &[ExtractedEntity]) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let mut slots = HashMap::new();
        
        {
            let intent_patterns = self.intent_patterns.read();
            if let Some(pattern_info) = intent_patterns.get(&intent_info.0) {
                for required_slot in &pattern_info.required_slots {
                    let slot_value = self.extract_slot_value(query, required_slot, entities)?;
                    if let Some(value) = slot_value {
                        slots.insert(required_slot.clone(), value);
                    }
                }
            }
        }
        
        Ok(slots)
    }

    fn extract_slot_value(&self, query: &str, slot_name: &str, entities: &[ExtractedEntity]) -> Result<Option<String>, Box<dyn std::error::Error>> {
        match slot_name {
            "entity" => {
                // 查找第一个人物、组织或概念实体
                for entity in entities {
                    if matches!(entity.entity_type.as_str(), "PERSON" | "ORGANIZATION" | "CONCEPT") {
                        return Ok(Some(entity.text.clone()));
                    }
                }
            }
            "entity1" | "entity2" => {
                // 查找实体对
                let relevant_entities: Vec<_> = entities.iter()
                    .filter(|e| matches!(e.entity_type.as_str(), "PERSON" | "ORGANIZATION" | "CONCEPT"))
                    .collect();
                
                if slot_name == "entity1" && !relevant_entities.is_empty() {
                    return Ok(Some(relevant_entities[0].text.clone()));
                } else if slot_name == "entity2" && relevant_entities.len() > 1 {
                    return Ok(Some(relevant_entities[1].text.clone()));
                }
            }
            "attribute" => {
                // 基于常见属性词汇提取
                let attribute_keywords = ["年龄", "身高", "体重", "职业", "出生", "学历", "作品", "成就"];
                for keyword in &attribute_keywords {
                    if query.contains(keyword) {
                        return Ok(Some(keyword.to_string()));
                    }
                }
            }
            "category" => {
                // 提取类别信息
                for entity in entities {
                    if entity.entity_type == "CATEGORY" {
                        return Ok(Some(entity.text.clone()));
                    }
                }
            }
            _ => {}
        }
        
        Ok(None)
    }

    async fn analyze_context(&self, query: &str, entities: &[ExtractedEntity]) -> Result<QueryContext, Box<dyn std::error::Error>> {
        let query_type = self.determine_query_type(query, entities).await?;
        let domain = self.determine_domain(query, entities).await?;
        let temporal_info = self.extract_temporal_context(entities);
        let spatial_info = self.extract_spatial_context(entities);
        let modifiers = self.extract_modifiers(query).await?;
        
        Ok(QueryContext {
            query_type,
            domain,
            temporal_info,
            spatial_info,
            modifiers,
        })
    }

    async fn determine_query_type(&self, query: &str, _entities: &[ExtractedEntity]) -> Result<String, Box<dyn std::error::Error>> {
        if query.contains("是什么") || query.contains("是谁") {
            Ok("definition".to_string())
        } else if query.contains("怎么") || query.contains("如何") {
            Ok("how_to".to_string())
        } else if query.contains("为什么") {
            Ok("why".to_string())
        } else if query.contains("什么时候") {
            Ok("when".to_string())
        } else if query.contains("哪里") || query.contains("在哪") {
            Ok("where".to_string())
        } else {
            Ok("general".to_string())
        }
    }

    async fn determine_domain(&self, query: &str, entities: &[ExtractedEntity]) -> Result<String, Box<dyn std::error::Error>> {
        let domain_keywords = &self.domain_keywords;
        
        // 命中关键词最多的领域胜出；同分按领域名字典序打破平局，
        // 避免 HashMap 迭代顺序导致同一查询得到不同领域
        let mut best: Option<(&String, usize)> = None;
        for (domain, keywords) in domain_keywords.iter() {
            let hits = keywords
                .iter()
                .filter(|keyword| query.contains(keyword.as_str()))
                .count();
            if hits == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((best_domain, best_hits)) => {
                    hits > best_hits || (hits == best_hits && domain < best_domain)
                }
            };
            if better {
                best = Some((domain, hits));
            }
        }
        if let Some((domain, _)) = best {
            return Ok(domain.clone());
        }
        
        // 基于实体类型推断领域
        for entity in entities {
            match entity.entity_type.as_str() {
                "PERSON" => return Ok("PEOPLE".to_string()),
                "ORGANIZATION" => return Ok("BUSINESS".to_string()),
                "LOCATION" => return Ok("GEOGRAPHY".to_string()),
                _ => {}
            }
        }
        
        Ok("GENERAL".to_string())
    }

    fn extract_temporal_context(&self, entities: &[ExtractedEntity]) -> Option<TemporalInfo> {
        for entity in entities {
            if entity.entity_type == "TEMPORAL" {
                let is_relative = matches!(entity.normalized_value.as_deref(), 
                    Some("TODAY") | Some("TOMORROW") | Some("YESTERDAY") | Some("NOW") | Some("PAST") | Some("FUTURE")
                    | Some("LAST_WEEK") | Some("NEXT_WEEK") | Some("LAST_MONTH") | Some("NEXT_MONTH")
                    | Some("LAST_YEAR") | Some("NEXT_YEAR"));
                
                return Some(TemporalInfo {
                    time_type: entity.normalized_value.clone().unwrap_or_default(),
                    time_value: entity.text.clone(),
                    is_relative,
                });
            }
        }
        None
    }

    fn extract_spatial_context(&self, entities: &[ExtractedEntity]) -> Option<SpatialInfo> {
        for entity in entities {
            if entity.entity_type == "SPATIAL" {
                let is_relative = matches!(entity.normalized_value.as_deref(),
                    Some("HERE") | Some("THERE") | Some("NEARBY") | Some("FAR"));
                
                return Some(SpatialInfo {
                    location_type: entity.normalized_value.clone().unwrap_or_default(),
                    location_value: entity.text.clone(),
                    is_relative,
                });
            }
        }
        None
    }

    async fn extract_modifiers(&self, query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // 数组字面量直接在栈上迭代，避免每次调用堆分配 Vec
        let modifiers = [
            "最", "更", "非常", "特别", "很", "极其", "相当", "十分", "比较", "稍微", "略", "大概", "大约", "可能", "也许", "或许", "一定", "肯定", "必须", "应该", "需要"
        ];
        
        let mut found_modifiers = Vec::new();
        for modifier in modifiers {
            if query.contains(modifier) {
                found_modifiers.push(modifier.to_string());
            }
        }
        
        Ok(found_modifiers)
    }

    fn update_context_memory(&self, context: &QueryContext) {
        let mut memory = self.context_memory.write();
        memory.push(context.clone());
        
        // 保持最近的10个上下文
        if memory.len() > 10 {
            memory.remove(0);
        }
    }

    pub fn get_context_history(&self) -> Vec<QueryContext> {
        self.context_memory.read().clone()
    }

    pub fn clear_context(&self) {
        self.context_memory.write().clear();
    }

    pub fn add_intent_pattern(&self, pattern: IntentPattern) {
        let mut patterns = self.intent_patterns.write();
        patterns.insert(pattern.intent.clone(), pattern);
    }

    pub fn add_entity_pattern(&self, pattern: EntityPattern) {
        let mut patterns = self.entity_patterns.write();
        patterns.insert(pattern.entity_type.clone(), pattern);
    }
}

impl Default for AdvancedNLU {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_intent_recognition() {
        let nlu = AdvancedNLU::new();
        
        let result = nlu.analyze_query("刘德华是谁").await.unwrap();
        assert_eq!(result.intent, "entity_query");
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let nlu = AdvancedNLU::new();
        
        let result = nlu.analyze_query("刘德华的年龄是多少").await.unwrap();
        assert!(!result.entities.is_empty());
        
        // 应该提取到人物实体
        let person_entities: Vec<_> = result.entities.iter()
            .filter(|e| e.entity_type == "PERSON")
            .collect();
        assert!(!person_entities.is_empty());
    }

    #[tokio::test]
    async fn test_slot_extraction() {
        let nlu = AdvancedNLU::new();
        
        let result = nlu.analyze_query("刘德华的年龄是多少").await.unwrap();
        assert!(result.slots.contains_key("entity"));
        assert!(result.slots.contains_key("attribute"));
    }

    #[tokio::test]
    async fn test_context_analysis() {
        let nlu = AdvancedNLU::new();
        
        let result = nlu.analyze_query("明天北京的天气怎么样").await.unwrap();
        assert!(result.context.temporal_info.is_some());
        assert_eq!(result.context.query_type, "how_to");
    }

    #[test]
    fn test_word_similarity() {
        let nlu = AdvancedNLU::new();
        
        let similarity = nlu.calculate_word_similarity("苹果", "苹果");
        assert_eq!(similarity, 1.0);
        
        let similarity = nlu.calculate_word_similarity("苹果", "苹");
        assert!(similarity >= 0.5); // 编辑距离为1，最大长度为2，相似度为0.5
    }
}
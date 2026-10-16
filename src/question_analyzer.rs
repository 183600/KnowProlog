use jieba_rs::Jieba;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnalysis {
    pub original_question: String,
    pub question_type: QuestionType,
    pub domain: Domain,
    pub complexity: ComplexityLevel,
    pub intent: Intent,
    pub entities: Vec<Entity>,
    pub keywords: Vec<String>,
    pub confidence: f64,
    pub suggested_query_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuestionType {
    FactualQuery,        // 事实查询：法国的首都是哪里？
    ConceptExplanation,  // 概念解释：什么是黑洞？
    ComparisonAnalysis,  // 比较分析：比较斯多葛主义和享乐主义
    HistoricalDescription, // 历史描述：文艺复兴的影响
    ScientificPrinciple, // 科学原理：牛顿第三定律
    DefinitionQuestion,  // 定义问题：什么是光合作用？
    CausalExplanation,   // 因果解释：为什么会发生？
    ProcessDescription,  // 过程描述：如何工作？
    ListQuery,           // 列表查询：有哪些？
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Domain {
    Geography,      // 地理
    History,        // 历史  
    Science,        // 科学
    Technology,     // 技术
    Literature,     // 文学
    Philosophy,     // 哲学
    Sports,         // 体育
    Politics,       // 政治
    Economics,      // 经济
    Culture,        // 文化
    General,        // 通用
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ComplexityLevel {
    Simple,         // 简单：直接事实查询
    Intermediate,   // 中等：需要一定解释
    Complex,        // 复杂：需要深入分析
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Intent {
    SeekInformation,    // 寻求信息
    UnderstandConcept,  // 理解概念
    CompareEntities,    // 比较实体
    AnalyzeRelation,    // 分析关系
    ExplainProcess,     // 解释过程
    DescribeImpact,     // 描述影响
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f64,
}

pub struct QuestionAnalyzer {
    jieba: Arc<Jieba>,
    question_patterns: Arc<RwLock<HashMap<QuestionType, Vec<Regex>>>>,
    domain_keywords: Arc<RwLock<HashMap<Domain, Vec<String>>>>,
    intent_indicators: Arc<RwLock<HashMap<Intent, Vec<String>>>>,
    entity_patterns: Arc<RwLock<Vec<(Regex, String)>>>,
    stop_words: HashSet<String>,
}

impl QuestionAnalyzer {
    pub fn new() -> Self {
        let jieba = Arc::new(Jieba::new());
        
        let stop_words = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "也", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "还", "来", "个", "对", "为", "把", "里", "后", "才", "能", "下", "这些", "但是", "如果", "只是"
        ].iter().map(|s| s.to_string()).collect();

        let analyzer = Self {
            jieba,
            question_patterns: Arc::new(RwLock::new(HashMap::new())),
            domain_keywords: Arc::new(RwLock::new(HashMap::new())),
            intent_indicators: Arc::new(RwLock::new(HashMap::new())),
            entity_patterns: Arc::new(RwLock::new(Vec::new())),
            stop_words,
        };
        
        analyzer.initialize_patterns();
        analyzer
    }

    fn initialize_patterns(&self) {
        self.initialize_question_patterns();
        self.initialize_domain_keywords();
        self.initialize_intent_indicators();
        self.initialize_entity_patterns();
    }

    fn initialize_question_patterns(&self) {
        let mut patterns = self.question_patterns.write();
        
        // 事实查询模式
        patterns.insert(QuestionType::FactualQuery, vec![
            Regex::new(r"(.+)的(.+)是(.+)").unwrap(),
            Regex::new(r"(.+)是哪里").unwrap(),
            Regex::new(r"(.+)是什么").unwrap(),
            Regex::new(r"(.+)是谁").unwrap(),
            Regex::new(r"(.+)的首都").unwrap(),
            Regex::new(r"(.+)的海拔").unwrap(),
            Regex::new(r"(.+)有多少").unwrap(),
            Regex::new(r"(.+)是哪一年").unwrap(),
            Regex::new(r"(.+)的冠军").unwrap(),
            Regex::new(r"(.+)的作者").unwrap(),
            Regex::new(r"(.+)的分子式").unwrap(),
        ]);

        // 概念解释模式
        patterns.insert(QuestionType::ConceptExplanation, vec![
            Regex::new(r"什么是(.+)").unwrap(),
            Regex::new(r"解释一下(.+)").unwrap(),
            Regex::new(r"简述(.+)").unwrap(),
            Regex::new(r"描述(.+)").unwrap(),
            Regex::new(r"(.+)的工作原理").unwrap(),
            Regex::new(r"(.+)是什么意思").unwrap(),
        ]);

        // 比较分析模式  
        patterns.insert(QuestionType::ComparisonAnalysis, vec![
            Regex::new(r"比较(.+)和(.+)").unwrap(),
            Regex::new(r"(.+)和(.+)的区别").unwrap(),
            Regex::new(r"(.+)相比(.+)").unwrap(),
            Regex::new(r"(.+)与(.+)的差异").unwrap(),
        ]);

        // 历史描述模式
        patterns.insert(QuestionType::HistoricalDescription, vec![
            Regex::new(r"(.+)对(.+)产生了(.+)影响").unwrap(),
            Regex::new(r"(.+)的历史意义").unwrap(),
            Regex::new(r"(.+)在(.+)中扮演了(.+)角色").unwrap(),
            Regex::new(r"(.+)的背景").unwrap(),
        ]);

        // 科学原理模式
        patterns.insert(QuestionType::ScientificPrinciple, vec![
            Regex::new(r"(.+)定律").unwrap(),
            Regex::new(r"(.+)原理").unwrap(),
            Regex::new(r"(.+)理论").unwrap(),
            Regex::new(r"解释(.+)定律").unwrap(),
        ]);

        // 定义问题模式
        patterns.insert(QuestionType::DefinitionQuestion, vec![
            Regex::new(r"什么是(.+)").unwrap(),
            Regex::new(r"(.+)的定义").unwrap(),
            Regex::new(r"(.+)是指什么").unwrap(),
        ]);

        // 因果解释模式
        patterns.insert(QuestionType::CausalExplanation, vec![
            Regex::new(r"为什么(.+)").unwrap(),
            Regex::new(r"(.+)的原因").unwrap(),
            Regex::new(r"(.+)为何(.+)").unwrap(),
        ]);

        // 过程描述模式
        patterns.insert(QuestionType::ProcessDescription, vec![
            Regex::new(r"如何(.+)").unwrap(),
            Regex::new(r"怎么(.+)").unwrap(),
            Regex::new(r"(.+)的过程").unwrap(),
            Regex::new(r"(.+)的步骤").unwrap(),
        ]);
    }

    fn initialize_domain_keywords(&self) {
        let mut domains = self.domain_keywords.write();
        
        domains.insert(Domain::Geography, vec![
            "国家".to_string(), "首都".to_string(), "城市".to_string(), "海拔".to_string(), 
            "高度".to_string(), "地理".to_string(), "位置".to_string(), "山峰".to_string(),
            "法国".to_string(), "珠穆朗玛峰".to_string(), "北京".to_string(), "东京".to_string(),
            "京都".to_string(), "太阳系".to_string(), "行星".to_string(),
        ]);

        domains.insert(Domain::Science, vec![
            "化学".to_string(), "分子式".to_string(), "物理".to_string(), "定律".to_string(),
            "原理".to_string(), "理论".to_string(), "科学".to_string(), "实验".to_string(),
            "黑洞".to_string(), "进化论".to_string(), "光合作用".to_string(), "量子".to_string(),
            "基因".to_string(), "CRISPR".to_string(), "摩尔定律".to_string(), "牛顿".to_string(),
        ]);

        domains.insert(Domain::Technology, vec![
            "技术".to_string(), "计算机".to_string(), "网络".to_string(), "软件".to_string(),
            "区块链".to_string(), "5G".to_string(), "机器学习".to_string(), "深度学习".to_string(),
            "人工智能".to_string(), "量子计算".to_string(), "Python".to_string(), "编程".to_string(),
        ]);

        domains.insert(Domain::History, vec![
            "历史".to_string(), "古代".to_string(), "朝代".to_string(), "战争".to_string(),
            "文艺复兴".to_string(), "第二次世界大战".to_string(), "金字塔".to_string(), "古埃及".to_string(),
            "独立宣言".to_string(), "美国".to_string(), "丝绸之路".to_string(),
        ]);

        domains.insert(Domain::Literature, vec![
            "文学".to_string(), "作者".to_string(), "作品".to_string(), "小说".to_string(),
            "诗歌".to_string(), "哈姆雷特".to_string(), "莎士比亚".to_string(), "著作".to_string(),
        ]);

        domains.insert(Domain::Philosophy, vec![
            "哲学".to_string(), "思想".to_string(), "主义".to_string(), "观点".to_string(),
            "孔子".to_string(), "斯多葛".to_string(), "享乐主义".to_string(), "普罗米修斯".to_string(),
        ]);

        domains.insert(Domain::Sports, vec![
            "体育".to_string(), "运动".to_string(), "世界杯".to_string(), "冠军".to_string(),
            "足球".to_string(), "比赛".to_string(), "队伍".to_string(),
        ]);

        domains.insert(Domain::Culture, vec![
            "文化".to_string(), "艺术".to_string(), "绘画".to_string(), "音乐".to_string(),
            "梵高".to_string(), "星夜".to_string(), "希腊神话".to_string(), "内卷".to_string(),
        ]);
    }

    fn initialize_intent_indicators(&self) {
        let mut intents = self.intent_indicators.write();
        
        intents.insert(Intent::SeekInformation, vec![
            "是什么".to_string(), "是谁".to_string(), "是哪里".to_string(), "多少".to_string(),
            "哪一年".to_string(), "什么时候".to_string(), "多高".to_string(), "有多少".to_string(),
        ]);

        intents.insert(Intent::UnderstandConcept, vec![
            "什么是".to_string(), "解释".to_string(), "描述".to_string(), "简述".to_string(),
            "介绍".to_string(), "说明".to_string(),
        ]);

        intents.insert(Intent::CompareEntities, vec![
            "比较".to_string(), "区别".to_string(), "差异".to_string(), "相比".to_string(),
            "哪个更".to_string(), "与".to_string(),
        ]);

        intents.insert(Intent::AnalyzeRelation, vec![
            "关系".to_string(), "联系".to_string(), "相互".to_string(), "之间".to_string(),
        ]);

        intents.insert(Intent::ExplainProcess, vec![
            "如何".to_string(), "怎么".to_string(), "过程".to_string(), "步骤".to_string(),
            "方法".to_string(), "工作原理".to_string(),
        ]);

        intents.insert(Intent::DescribeImpact, vec![
            "影响".to_string(), "作用".to_string(), "效果".to_string(), "结果".to_string(),
            "意义".to_string(), "价值".to_string(),
        ]);
    }

    fn initialize_entity_patterns(&self) {
        let mut patterns = self.entity_patterns.write();
        
        // 地名实体
        patterns.push((Regex::new(r"[法德英美中日韩俄印澳][国]").unwrap(), "COUNTRY".to_string()));
        patterns.push((Regex::new(r"[北上广深][京海州圳]|东京|巴黎|伦敦|纽约").unwrap(), "CITY".to_string()));
        
        // 人名实体
        patterns.push((Regex::new(r"[孔老庄孟荀]子|牛顿|达尔文|梵高|莎士比亚").unwrap(), "PERSON".to_string()));
        
        // 概念实体
        patterns.push((Regex::new(r"黑洞|光合作用|进化论|量子计算|区块链|机器学习").unwrap(), "CONCEPT".to_string()));
        
        // 组织实体（多字符词必须用分组/选择，字符类只会匹配单个字符）
        patterns.push((Regex::new(r"联合国|世界杯|奥运会").unwrap(), "ORGANIZATION".to_string()));
        
        // 时间实体
        patterns.push((Regex::new(r"\d{4}年|\d{1,2}世纪|古代|现代|当代").unwrap(), "TIME".to_string()));
        
        // 数量实体
        patterns.push((Regex::new(r"\d+(\.\d+)?(米|公里|千米|厘米|毫米|块|个|人|年|月|日)").unwrap(), "QUANTITY".to_string()));
    }

    pub async fn analyze_question(&self, question: &str) -> Result<QuestionAnalysis, Box<dyn std::error::Error + Send + Sync>> {
        // 预处理
        let normalized_question = self.normalize_question(question);
        
        // 识别问题类型
        let question_type = self.identify_question_type(&normalized_question).await?;
        
        // 识别领域
        let domain = self.identify_domain(&normalized_question).await?;
        
        // 评估复杂度
        let complexity = self.assess_complexity(&normalized_question, &question_type).await?;
        
        // 识别意图
        let intent = self.identify_intent(&normalized_question).await?;
        
        // 提取实体
        let entities = self.extract_entities(&normalized_question).await?;
        
        // 提取关键词
        let keywords = self.extract_keywords(&normalized_question).await?;
        
        // 计算置信度
        let confidence = self.calculate_confidence(&question_type, &domain, &entities).await?;
        
        // 生成建议的查询模式
        let suggested_query_patterns = self.generate_query_patterns(&question_type, &intent, &entities).await?;
        
        Ok(QuestionAnalysis {
            original_question: question.to_string(),
            question_type,
            domain,
            complexity,
            intent,
            entities,
            keywords,
            confidence,
            suggested_query_patterns,
        })
    }

    fn normalize_question(&self, question: &str) -> String {
        question.trim()
            .replace("？", "?")
            .replace("！", "!")
            .replace("，", ",")
            .replace("。", ".")
            .to_string()
    }

    async fn identify_question_type(&self, question: &str) -> Result<QuestionType, Box<dyn std::error::Error + Send + Sync>> {
        let patterns = self.question_patterns.read();
        
        let mut scores: HashMap<QuestionType, f64> = HashMap::new();
        
        for (q_type, regexes) in patterns.iter() {
            let mut max_score: f64 = 0.0;
            for regex in regexes {
                if regex.is_match(question) {
                    max_score = max_score.max(1.0);
                }
            }
            scores.insert(q_type.clone(), max_score);
        }
        
        // 基于关键词的额外评分
        if question.contains("什么是") || question.contains("解释") {
            *scores.entry(QuestionType::ConceptExplanation).or_insert(0.0) += 0.8;
        }
        
        if question.contains("比较") || question.contains("区别") {
            *scores.entry(QuestionType::ComparisonAnalysis).or_insert(0.0) += 0.8;
        }
        
        if question.contains("影响") || question.contains("意义") {
            *scores.entry(QuestionType::HistoricalDescription).or_insert(0.0) += 0.8;
        }
        
        if question.contains("为什么") || question.contains("原因") {
            *scores.entry(QuestionType::CausalExplanation).or_insert(0.0) += 0.8;
        }
        
        if question.contains("如何") || question.contains("怎么") {
            *scores.entry(QuestionType::ProcessDescription).or_insert(0.0) += 0.8;
        }
        
        // 选择得分最高的类型
        let best_type = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(q_type, _)| q_type)
            .unwrap_or(QuestionType::FactualQuery);
            
        Ok(best_type)
    }

    async fn identify_domain(&self, question: &str) -> Result<Domain, Box<dyn std::error::Error + Send + Sync>> {
        let domains = self.domain_keywords.read();
        
        let mut scores: HashMap<Domain, f64> = HashMap::new();
        
        for (domain, keywords) in domains.iter() {
            let mut score = 0.0;
            for keyword in keywords {
                if question.contains(keyword) {
                    score += 1.0;
                }
            }
            scores.insert(domain.clone(), score);
        }
        
        let best_domain = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(domain, _)| domain)
            .unwrap_or(Domain::General);
            
        Ok(best_domain)
    }

    async fn assess_complexity(&self, question: &str, question_type: &QuestionType) -> Result<ComplexityLevel, Box<dyn std::error::Error + Send + Sync>> {
        let word_count = question.split_whitespace().count();
        
        let complexity = match question_type {
            QuestionType::FactualQuery => {
                if word_count <= 10 {
                    ComplexityLevel::Simple
                } else {
                    ComplexityLevel::Intermediate
                }
            }
            QuestionType::ConceptExplanation | QuestionType::ComparisonAnalysis | 
            QuestionType::HistoricalDescription | QuestionType::ScientificPrinciple => {
                if word_count <= 15 {
                    ComplexityLevel::Intermediate
                } else {
                    ComplexityLevel::Complex
                }
            }
            _ => ComplexityLevel::Intermediate,
        };
        
        Ok(complexity)
    }

    async fn identify_intent(&self, question: &str) -> Result<Intent, Box<dyn std::error::Error + Send + Sync>> {
        let intents = self.intent_indicators.read();
        
        let mut scores: HashMap<Intent, f64> = HashMap::new();
        
        for (intent, indicators) in intents.iter() {
            let mut score = 0.0;
            for indicator in indicators {
                if question.contains(indicator) {
                    score += 1.0;
                }
            }
            scores.insert(intent.clone(), score);
        }
        
        let best_intent = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(intent, _)| intent)
            .unwrap_or(Intent::SeekInformation);
            
        Ok(best_intent)
    }

    async fn extract_entities(&self, question: &str) -> Result<Vec<Entity>, Box<dyn std::error::Error + Send + Sync>> {
        let patterns = self.entity_patterns.read();
        let mut entities = Vec::new();
        
        for (regex, entity_type) in patterns.iter() {
            for mat in regex.find_iter(question) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: entity_type.clone(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.9,
                });
            }
        }
        
        Ok(entities)
    }

    async fn extract_keywords(&self, question: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let words: Vec<_> = self.jieba.cut(question, false).into_iter()
            .filter(|word| !self.stop_words.contains(*word) && word.len() > 1)
            .map(|word| word.to_string())
            .collect();
            
        Ok(words)
    }

    async fn calculate_confidence(&self, question_type: &QuestionType, domain: &Domain, entities: &[Entity]) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut confidence: f64 = 0.5; // 基础置信度
        
        // 如果识别到明确的问题类型，增加置信度
        if !matches!(question_type, QuestionType::FactualQuery) {
            confidence += 0.2;
        }
        
        // 如果识别到明确的领域，增加置信度
        if !matches!(domain, Domain::General) {
            confidence += 0.2;
        }
        
        // 如果识别到实体，增加置信度
        if !entities.is_empty() {
            confidence += 0.1;
        }
        
        Ok(confidence.min(1.0))
    }

    async fn generate_query_patterns(&self, question_type: &QuestionType, _intent: &Intent, _entities: &[Entity]) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut patterns = Vec::new();
        
        match question_type {
            QuestionType::FactualQuery => {
                patterns.push("SPARQL查询: SELECT ?property WHERE { ?entity ?relation ?property }".to_string());
                patterns.push("Prolog查询: entity_property(Entity, Property)".to_string());
            }
            QuestionType::ConceptExplanation => {
                patterns.push("知识图谱查询: 实体定义和描述".to_string());
                patterns.push("本体查询: 概念层次和属性".to_string());
            }
            QuestionType::ComparisonAnalysis => {
                patterns.push("对比查询: 实体A vs 实体B 的属性差异".to_string());
                patterns.push("关系查询: 两个实体的共同点和不同点".to_string());
            }
            _ => {
                patterns.push("通用查询: 基于关键词的语义搜索".to_string());
            }
        }
        
        Ok(patterns)
    }
}

impl Default for QuestionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_factual_query_analysis() {
        let analyzer = QuestionAnalyzer::new();
        
        let result = analyzer.analyze_question("法国的首都是哪里？").await.unwrap();
        assert_eq!(result.question_type, QuestionType::FactualQuery);
        assert_eq!(result.domain, Domain::Geography);
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_concept_explanation_analysis() {
        let analyzer = QuestionAnalyzer::new();
        
        let result = analyzer.analyze_question("什么是黑洞？").await.unwrap();
        assert_eq!(result.question_type, QuestionType::ConceptExplanation);
        assert_eq!(result.domain, Domain::Science);
    }

    #[tokio::test]
    async fn test_comparison_analysis() {
        let analyzer = QuestionAnalyzer::new();
        
        let result = analyzer.analyze_question("比较斯多葛主义和享乐主义的主要区别").await.unwrap();
        assert_eq!(result.question_type, QuestionType::ComparisonAnalysis);
        assert_eq!(result.domain, Domain::Philosophy);
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let analyzer = QuestionAnalyzer::new();
        
        let result = analyzer.analyze_question("牛顿第三定律是什么？").await.unwrap();
        assert!(!result.entities.is_empty());
        assert!(!result.keywords.is_empty());
    }
}
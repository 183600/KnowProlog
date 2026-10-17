use crate::nlp::{preprocess, normalize_words, TaggedWord};
use crate::{SystemConfig, NlpConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    pub input: String,
    pub semantics: Vec<SemanticResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticResult {
    pub domain: String,
    pub score: f64,
    pub intent: String,
    pub slots: Vec<(String, String)>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DomainConfig {
    pub name: String,
    pub description: String,
    pub intents: Vec<IntentConfig>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentConfig {
    pub name: String,
    pub patterns: Vec<String>,
    pub slots: Vec<SlotConfig>,
    pub priority: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SlotConfig {
    pub name: String,
    pub slot_type: String,
    pub required: bool,
    pub patterns: Vec<String>,
}

pub struct NLUEngine {
    domains: Vec<DomainConfig>,
    nlp_config: NlpConfig,
    synonyms: Option<HashMap<String, String>>,
}

impl NLUEngine {
    pub fn new(config: &SystemConfig) -> Self {
        let domains = create_default_domains();
        Self {
            domains,
            nlp_config: config.nlp.clone(),
            synonyms: config.synonyms.as_ref().map(|s| s.mappings.clone()),
        }
    }

    pub fn analyze(&self, input: &str) -> SemanticAnalysis {
        let tagged_words = preprocess(input, &self.nlp_config);
        self.analyze_tagged(input, tagged_words)
    }

    /// 使用已分词结果进行语义分析，避免重复分词
    pub fn analyze_tagged(&self, input: &str, tagged_words: Vec<TaggedWord>) -> SemanticAnalysis {
        let normalized_words = if let Some(synonyms) = &self.synonyms {
            normalize_words(tagged_words, synonyms)
        } else {
            tagged_words
        };

        let mut semantics = Vec::new();

        // 分词文本在一轮分析内不变，提前展开一次供所有模式复用
        let input_words: Vec<&str> = normalized_words.iter().map(|w| w.text.as_str()).collect();

        for domain in &self.domains {
            if let Some(result) =
                self.analyze_domain(input, &normalized_words, &input_words, domain)
            {
                semantics.push(result);
            }
        }

        semantics.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        SemanticAnalysis {
            input: input.to_string(),
            semantics,
        }
    }

    fn analyze_domain(
        &self,
        input: &str,
        tagged_words: &[TaggedWord],
        input_words: &[&str],
        domain: &DomainConfig,
    ) -> Option<SemanticResult> {
        let mut best_intent = None;
        let mut best_score = 0.0;
        let mut best_slots = Vec::new();

        for intent in &domain.intents {
            if let Some((score, slots)) = self.match_intent(input, tagged_words, input_words, intent) {
                if score > best_score {
                    best_score = score;
                    best_intent = Some(intent.name.clone());
                    best_slots = slots;
                }
            }
        }

        if let Some(intent) = best_intent {
            Some(SemanticResult {
                domain: domain.name.clone(),
                score: best_score,
                intent,
                slots: best_slots,
            })
        } else {
            None
        }
    }

    fn match_intent(
        &self,
        input: &str,
        tagged_words: &[TaggedWord],
        input_words: &[&str],
        intent: &IntentConfig,
    ) -> Option<(f64, Vec<(String, String)>)> {
        let mut best_score = 0.0;
        let mut best_slots = Vec::new();

        for pattern in &intent.patterns {
            if let Some((score, slots)) =
                self.match_pattern(input, tagged_words, input_words, pattern, &intent.slots)
            {
                if score > best_score {
                    best_score = score;
                    best_slots = slots;
                }
            }
        }

        if best_score > 0.0 {
            Some((best_score, best_slots))
        } else {
            None
        }
    }

    fn match_pattern(
        &self,
        _input: &str,
        tagged_words: &[TaggedWord],
        input_words: &[&str],
        pattern: &str,
        slot_configs: &[SlotConfig],
    ) -> Option<(f64, Vec<(String, String)>)> {
        let pattern_words: Vec<&str> = pattern.split_whitespace().collect();
        let total_count = pattern_words.len();

        let mut matched_count = 0;
        let mut slots = Vec::new();

        for pattern_word in pattern_words {
            if pattern_word.starts_with('{') && pattern_word.ends_with('}') {
                let slot_name = &pattern_word[1..pattern_word.len() - 1];
                if let Some(slot_config) = slot_configs.iter().find(|s| s.name == slot_name) {
                    if let Some(value) = self.extract_slot_value(tagged_words, slot_config) {
                        slots.push((slot_name.to_string(), value));
                        matched_count += 1;
                    }
                }
            } else {
                for input_word in input_words {
                    if self.word_similarity(pattern_word, input_word) >= 0.8 {
                        matched_count += 1;
                        break;
                    }
                }
            }
        }

        if total_count == 0 {
            return None;
        }

        let score = matched_count as f64 / total_count as f64;
        
        if score > 0.6 {
            Some((score, slots))
        } else {
            None
        }
    }

    fn extract_slot_value(&self, tagged_words: &[TaggedWord], slot_config: &SlotConfig) -> Option<String> {
        match slot_config.slot_type.as_str() {
            "城市" | "地点" => {
                for word in tagged_words {
                    if word.pos == "ns" || word.pos == "n" {
                        return Some(word.text.clone());
                    }
                    // 如果没有标记为地点，检查是否是已知城市名
                    if ["厦门", "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉"].contains(&word.text.as_str()) {
                        return Some(word.text.clone());
                    }
                }
            }
            "日期" | "时间" => {
                for pattern in &slot_config.patterns {
                    for word in tagged_words {
                        if word.text.contains(pattern) {
                            return Some(word.text.clone());
                        }
                    }
                }
                
                for word in tagged_words {
                    if word.pos == "m" && (word.text.contains("天") || word.text.contains("月") || word.text.contains("年")) {
                        return Some(word.text.clone());
                    }
                    // 检查常见时间词
                    if ["明天", "后天", "今天", "昨天", "大后天"].contains(&word.text.as_str()) {
                        return Some(word.text.clone());
                    }
                }
            }
            "人名" => {
                for word in tagged_words {
                    if word.pos == "nr" {
                        return Some(word.text.clone());
                    }
                    // 检查常见人名
                    if ["姚明", "刘德华", "张三", "李四", "王五", "李明"].contains(&word.text.as_str()) {
                        return Some(word.text.clone());
                    }
                }
            }
            "数量" => {
                for word in tagged_words {
                    if word.pos == "m" || word.text.chars().all(|c| c.is_numeric()) {
                        return Some(word.text.clone());
                    }
                }
            }
            "实体" => {
                for word in tagged_words {
                    if word.pos == "nr" || word.pos == "n" {
                        return Some(word.text.clone());
                    }
                    // 检查常见实体名
                    if ["李明", "姚明", "刘德华", "张三", "李四", "王五"].contains(&word.text.as_str()) {
                        return Some(word.text.clone());
                    }
                }
            }
            _ => {}
        }
        None
    }

    fn word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }
        
        if word1.contains(word2) || word2.contains(word1) {
            return 0.8;
        }
        
        if let Some(synonyms) = &self.synonyms {
            if let Some(normalized1) = synonyms.get(word1) {
                if normalized1 == word2 {
                    return 0.9;
                }
            }
            if let Some(normalized2) = synonyms.get(word2) {
                if normalized2 == word1 {
                    return 0.9;
                }
            }
        }
        
        0.0
    }
}

fn create_default_domains() -> Vec<DomainConfig> {
    vec![
        DomainConfig {
            name: "天气".to_string(),
            description: "天气查询相关".to_string(),
            intents: vec![
                IntentConfig {
                    name: "天气查询".to_string(),
                    patterns: vec![
                        "{城市} {日期} 天气".to_string(),
                        "{城市} {日期} 会不会下雨".to_string(),
                        "{日期} {城市} 几度".to_string(),
                        "{城市} 天气怎么样".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "城市".to_string(),
                            slot_type: "城市".to_string(),
                            required: true,
                            patterns: vec!["厦门".to_string(), "北京".to_string(), "上海".to_string()],
                        },
                        SlotConfig {
                            name: "日期".to_string(),
                            slot_type: "日期".to_string(),
                            required: false,
                            patterns: vec!["明天".to_string(), "后天".to_string(), "今天".to_string()],
                        },
                    ],
                    priority: 10,
                },
                IntentConfig {
                    name: "雨".to_string(),
                    patterns: vec![
                        "会不会下雨".to_string(),
                        "有雨吗".to_string(),
                        "下雨".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "城市".to_string(),
                            slot_type: "城市".to_string(),
                            required: false,
                            patterns: vec![],
                        },
                        SlotConfig {
                            name: "日期".to_string(),
                            slot_type: "日期".to_string(),
                            required: false,
                            patterns: vec![],
                        },
                    ],
                    priority: 9,
                },
            ],
            examples: vec!["厦门明天会不会下雨".to_string()],
        },
        DomainConfig {
            name: "人物".to_string(),
            description: "人物信息查询".to_string(),
            intents: vec![
                IntentConfig {
                    name: "身高查询".to_string(),
                    patterns: vec![
                        "{人名} 身高".to_string(),
                        "{人名} 多高".to_string(),
                        "{人名} 真正的身高".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "人名".to_string(),
                            slot_type: "人名".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                    ],
                    priority: 10,
                },
                IntentConfig {
                    name: "年龄查询".to_string(),
                    patterns: vec![
                        "{人名} 年龄".to_string(),
                        "{人名} 多大".to_string(),
                        "{人名} 几岁".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "人名".to_string(),
                            slot_type: "人名".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                    ],
                    priority: 9,
                },
            ],
            examples: vec!["姚明真正的身高是多少".to_string()],
        },
        DomainConfig {
            name: "聊天".to_string(),
            description: "日常聊天对话".to_string(),
            intents: vec![
                IntentConfig {
                    name: "问候".to_string(),
                    patterns: vec![
                        "你好".to_string(),
                        "早上好".to_string(),
                        "晚上好".to_string(),
                        "hi".to_string(),
                        "hello".to_string(),
                    ],
                    slots: vec![],
                    priority: 5,
                },
                IntentConfig {
                    name: "感谢".to_string(),
                    patterns: vec![
                        "谢谢".to_string(),
                        "感谢".to_string(),
                        "thanks".to_string(),
                    ],
                    slots: vec![],
                    priority: 5,
                },
            ],
            examples: vec!["你好".to_string(), "谢谢".to_string()],
        },
        DomainConfig {
            name: "知识查询".to_string(),
            description: "知识图谱查询".to_string(),
            intents: vec![
                IntentConfig {
                    name: "属性查询".to_string(),
                    patterns: vec![
                        "{实体} 的 {属性}".to_string(),
                        "{实体} {属性} 是什么".to_string(),
                        "什么是 {实体} 的 {属性}".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "实体".to_string(),
                            slot_type: "实体".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                        SlotConfig {
                            name: "属性".to_string(),
                            slot_type: "属性".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                    ],
                    priority: 8,
                },
                IntentConfig {
                    name: "信息查询".to_string(),
                    patterns: vec![
                        "{实体} 的 信息".to_string(),
                        "{人名} 的 信息".to_string(),
                        "关于 {实体}".to_string(),
                        "告诉我 {实体}".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "实体".to_string(),
                            slot_type: "实体".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                        SlotConfig {
                            name: "人名".to_string(),
                            slot_type: "人名".to_string(),
                            required: false,
                            patterns: vec![],
                        },
                    ],
                    priority: 9,
                },
                IntentConfig {
                    name: "关系查询".to_string(),
                    patterns: vec![
                        "{实体1} 和 {实体2} 什么关系".to_string(),
                        "{实体1} {实体2} 关系".to_string(),
                    ],
                    slots: vec![
                        SlotConfig {
                            name: "实体1".to_string(),
                            slot_type: "实体".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                        SlotConfig {
                            name: "实体2".to_string(),
                            slot_type: "实体".to_string(),
                            required: true,
                            patterns: vec![],
                        },
                    ],
                    priority: 8,
                },
            ],
            examples: vec!["北京的首都".to_string(), "张三和李四什么关系".to_string()],
        },
    ]
}
// src/nlp.rs
use jieba_rs::Jieba; // 导入结巴分词库，用于中文分词
use std::collections::HashMap; // 导入哈希映射，用于存储键值对
use std::sync::OnceLock; // 用于全局缓存分词器实例

// 结巴分词器初始化开销较大（需加载内置词典），全局复用同一实例
static JIEBA: OnceLock<Jieba> = OnceLock::new();

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

// 标注了词性的单词结构体
#[derive(Debug, Clone)]
pub struct TaggedWord {
    pub text: String, // 词语文本
    pub pos: String, // 词性标记（如n表示名词，v表示动词）
}

// Prolog推理规则结构体
#[derive(Debug, Clone)]
pub struct PrologRule {
    pub pattern: Vec<PatternElement>, // 模式元素序列
    pub query_type: String, // 查询类型标识
    pub priority: i32, // 优先级，数值越大越优先
}

// 模式元素枚举，定义模式匹配的各种元素类型
#[derive(Debug, Clone)]
pub enum PatternElement {
    Pos(String), // 匹配特定词性
    Word(String), // 匹配特定词语
    PosOrWord(String, String), // 匹配词性或词语
    Optional(Box<PatternElement>), // 可选元素
    Any, // 匹配任意元素
    Number, // 匹配数字
    Name, // 匹配人名
    Capture(String, Box<PatternElement>), // 捕获元素并命名
}

// 预处理函数：分词并进行词性标注
// 注意：crate::NlpConfig 现在引用的是 lib.rs 中的 NlpConfig
pub fn preprocess(text: &str, config: &crate::NlpConfig) -> Vec<TaggedWord> {
    let words = jieba().cut(text, false); // 执行分词，false表示精准模式

    // 结巴分词可能把配置中的常见人名（如"张三"）切分成多个单字，
    // 这里按最长匹配合并相邻词，使 full_names 中的人名能被正确识别
    let mut merged_words: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let mut matched = None;
        let max_end = (i + 4).min(words.len()); // 人名一般不超过4个分词
        for end in (i + 2..=max_end).rev() {
            let candidate: String = words[i..end].concat();
            if config.name_patterns.full_names.contains(&candidate) {
                matched = Some((candidate, end));
                break;
            }
        }
        if let Some((name, end)) = matched {
            merged_words.push(name);
            i = end;
        } else {
            merged_words.push(words[i].to_string());
            i += 1;
        }
    }

    let mut tagged_words = Vec::with_capacity(merged_words.len());
    for word in &merged_words {
        let word = word.as_str();
        // 确定词性
        let pos = if word.chars().all(|c| c.is_numeric()) { // 全是数字
            "m".to_string() // m表示数词
        } else if is_name(word, &config.name_patterns) { // 判断是否为人名
            "nr".to_string() // nr表示人名
        } else {
            // 从词性字典查找，找不到则标记为unknown
            config.pos_dict.get(word).cloned().unwrap_or_else(|| "unknown".to_string())
        };
        
        tagged_words.push(TaggedWord {
            text: word.to_string(),
            pos,
        });
    }
    
    tagged_words
}

// 词语归一化函数，将同义词转换为标准形式
pub fn normalize_words(words: Vec<TaggedWord>, synonyms: &HashMap<String, String>) -> Vec<TaggedWord> {
    words.into_iter().map(|mut word| {
        // 如果词语在同义词表中，替换为标准形式
        if let Some(canonical_word) = synonyms.get(&word.text) {
            word.text = canonical_word.clone();
        }
        word
    }).collect()
}

// 判断一个词是否为人名
// 注意：crate::NamePatterns 现在引用的是 lib.rs 中的 NamePatterns
fn is_name(word: &str, patterns: &crate::NamePatterns) -> bool {
    patterns.prefixes.iter().any(|p| word.starts_with(p)) || // 以姓氏前缀开头
    patterns.suffixes.iter().any(|s| word.ends_with(s)) || // 以名字后缀结尾
    patterns.full_names.iter().any(|n| n == word) || // 在常见人名列表中（避免为每个词分配临时字符串）
    patterns.profession_suffix.iter().any(|ps| word.ends_with(ps) && word.len() > ps.len()) // 以职业后缀结尾且不只是后缀本身
}

// 分析查询意图并提取实体
pub fn analyze_query_intent(tagged_words: &[TaggedWord], rules: &[PrologRule]) -> Option<(String, Vec<String>)> {
    // 仅对引用排序，避免深拷贝规则数据；稳定排序保证同优先级顺序与原来一致
    let mut sorted_rules: Vec<&PrologRule> = rules.iter().collect();
    sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority)); // 按优先级降序排序
    
    // 按优先级尝试匹配每个规则
    for rule in sorted_rules {
        if let Some(entities) = match_pattern_for_prolog(tagged_words, &rule.pattern) {
            return Some((rule.query_type.clone(), entities)); // 返回查询类型和提取的实体
        }
    }
    
    None // 没有匹配的规则
}

// 为Prolog推理匹配模式并提取实体
fn match_pattern_for_prolog(tagged_words: &[TaggedWord], pattern: &[PatternElement]) -> Option<Vec<String>> {
    let mut entities = Vec::new(); // 存储提取的实体
    let mut pos = 0; // 当前位置索引
    
    // 如果模式为空，返回空实体列表
    if pattern.is_empty() {
        return Some(entities);
    }
    
    // 遍历模式中的每个元素
    for pattern_elem in pattern {
        // 跳过空白词语（标点符号等）
        while pos < tagged_words.len() && 
              (tagged_words[pos].text.trim().is_empty() || 
               tagged_words[pos].text == "？" || 
               tagged_words[pos].text == "。" ||
               tagged_words[pos].text == "，" ||
               tagged_words[pos].text == "、") {
            pos += 1;
        }
        
        if pos >= tagged_words.len() {
            // 检查剩余模式是否都是可选的，如果不是，则匹配失败
            if !matches!(pattern_elem, PatternElement::Optional(_)) {
                return None;
            }
            continue;
        }
        
        match pattern_elem {
            PatternElement::Pos(expected_pos) => { // 匹配词性
                if expected_pos == "nr" {
                    // 特殊处理人名：尝试组合相邻的nr和unknown词
                    if pos < tagged_words.len() && 
                       (tagged_words[pos].pos == "nr" || tagged_words[pos].pos == "unknown") {
                        let mut name_parts = vec![tagged_words[pos].text.clone()];
                        let mut next_pos = pos + 1;
                        
                        // 检查后续的unknown词是否应该合并
                        while next_pos < tagged_words.len() && 
                              (tagged_words[next_pos].pos == "unknown" || 
                               tagged_words[next_pos].pos == "nr") {
                            name_parts.push(tagged_words[next_pos].text.clone());
                            next_pos += 1;
                        }
                        
                        entities.push(name_parts.join(""));
                        pos = next_pos;
                    } else {
                        return None; // 不是人名
                    }
                } else if pos < tagged_words.len() && tagged_words[pos].pos == *expected_pos {
                    // 如果是人名或名词，提取为实体
                    if expected_pos == "n" || expected_pos == "nr" {
                        entities.push(tagged_words[pos].text.clone());
                    }
                    pos += 1;
                } else {
                    return None; // 词性不匹配
                }
            }
            PatternElement::Word(expected_word) => { // 匹配具体词语
                // 更灵活的词语匹配
                if pos < tagged_words.len() {
                    let current_word = &tagged_words[pos].text;
                    
                    // 直接匹配
                    if current_word == expected_word {
                        pos += 1;
                        continue;
                    }
                    
                    // 检查是否是被分词的复合词
                    if expected_word.len() > 1 {
                        let expected_chars: Vec<char> = expected_word.chars().collect();
                        let mut matched_chars = Vec::new();
                        let mut temp_pos = pos;
                        
                        for expected_char in &expected_chars {
                            if temp_pos < tagged_words.len() {
                                let word_text = tagged_words[temp_pos].text.as_str();
                                // 等价于 word_text == expected_char.to_string()，但不分配临时字符串
                                if word_text.len() == expected_char.len_utf8()
                                    && word_text.starts_with(*expected_char)
                                {
                                    matched_chars.push(expected_char);
                                    temp_pos += 1;
                                    continue;
                                }
                            }
                            break;
                        }
                        
                        if matched_chars.len() == expected_chars.len() {
                            pos = temp_pos;
                            continue;
                        }
                    }
                    
                    // 部分匹配或包含匹配
                    if current_word.contains(expected_word) || expected_word.contains(current_word) {
                        pos += 1;
                        continue;
                    }
                    
                    return None; // 词语不匹配
                } else {
                    return None;
                }
            }
            PatternElement::Name => { // 匹配人名
                if pos < tagged_words.len() && 
                   (tagged_words[pos].pos == "nr" || tagged_words[pos].pos == "unknown") {
                    entities.push(tagged_words[pos].text.clone());
                    pos += 1;
                } else {
                    return None; // 不是人名
                }
            }
            PatternElement::Number => { // 匹配数字
                if pos < tagged_words.len() && 
                   (tagged_words[pos].pos == "m" || tagged_words[pos].text.chars().all(|c| c.is_numeric())) {
                    entities.push(tagged_words[pos].text.clone());
                    pos += 1;
                } else {
                    return None; // 不是数字
                }
            }
            _ => { // 其他情况直接跳过 (如 Any, Optional, Capture, PosOrWord)
                pos += 1;
            }
        }
    }
    
    Some(entities)
}

// 从字符串数组创建模式元素
pub fn pattern_from_strings(strings: Vec<String>) -> Vec<PatternElement> {
    strings.into_iter().map(|s| {
        match s.as_str() {
            "r" => PatternElement::Pos("r".to_string()), // 代词
            "n" => PatternElement::Pos("n".to_string()), // 名词
            "nr" => PatternElement::Pos("nr".to_string()), // 人名
            "m" => PatternElement::Pos("m".to_string()), // 数词
            "q" => PatternElement::Pos("q".to_string()), // 量词
            "v" => PatternElement::Pos("v".to_string()), // 动词
            "u" => PatternElement::Pos("u".to_string()), // 助词
            "f" => PatternElement::Pos("f".to_string()), // 方位词
            "a" => PatternElement::Pos("a".to_string()), // 形容词
            "d" => PatternElement::Pos("d".to_string()), // 副词
            "*" => PatternElement::Any, // 任意匹配
            _ => PatternElement::Word(s), // 具体词语
        }
    }).collect()
}

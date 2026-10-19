use jieba_rs::Jieba; // 导入结巴分词库，用于中文分词
use std::collections::HashMap; // 导入哈希映射，用于存储键值对
use std::sync::OnceLock;

static JIEBA: OnceLock<Jieba> = OnceLock::new();

/// 进程级共享的结巴分词器。
pub fn shared_jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

fn jieba() -> &'static Jieba {
    shared_jieba()
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
// 注意：crate::NlpConfig 引用的是 lib.rs 中的 NlpConfig
pub fn preprocess(text: &str, config: &crate::NlpConfig) -> Vec<TaggedWord> {
    // 结巴自带词性标注（nr 人名 / ns 地名 / nt 机构名 / nz 专名）：
    // 用作未知词的兜底，让「姚明」「诸葛亮」等词无需写进配置也能落型；
    // 配置 pos_dict 与 name_patterns 的优先级更高。
    let tagged = jieba().tag(text, false);
    let words: Vec<&str> = tagged.iter().map(|tag| tag.word).collect();
    let jieba_tags: Vec<&str> = tagged.iter().map(|tag| tag.tag).collect();

    // 常见人名集合
    let full_names: std::collections::HashSet<&str> = config
        .name_patterns
        .full_names
        .iter()
        .map(String::as_str)
        .collect();

    // 结巴分词可能把配置中的常见人名（如"张三"）切分成多个单字，
    // 这里按最长匹配合并相邻词，使 full_names 中的人名能被正确识别；
    // 合并与标注在同一次遍历内完成
    let mut tagged_words: Vec<TaggedWord> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        // 增量拼接候选词并记录最长匹配
        let mut matched: Option<(String, usize)> = None;
        let mut candidate = String::from(words[i]);
        let max_end = (i + 4).min(words.len()); // 人名一般不超过4个分词
        for end in i + 1..max_end {
            candidate.push_str(words[end]);
            if full_names.contains(candidate.as_str()) {
                matched = Some((candidate.clone(), end + 1));
            }
        }
        if let Some((name, end)) = matched {
            tagged_words.push(tag_word(name, config, &full_names, None));
            i = end;
        } else {
            tagged_words.push(tag_word(
                words[i].to_string(),
                config,
                &full_names,
                jieba_tags.get(i).copied(),
            ));
            i += 1;
        }
    }

    tagged_words
}

// 确定单个词语的词性并封装为标注词
fn tag_word(
    word: String,
    config: &crate::NlpConfig,
    full_names: &std::collections::HashSet<&str>,
    jieba_tag: Option<&str>,
) -> TaggedWord {
    let pos = if word.chars().all(|c| c.is_numeric()) { // 全是数字
        "m".to_string() // m表示数词
    } else if is_name(&word, &config.name_patterns, full_names) { // 判断是否为人名
        "nr".to_string() // nr表示人名
    } else if let Some(pos) = config.pos_dict.get(&word) {
        // 从词性字典查找（配置优先）
        pos.clone()
    } else {
        // 结巴标注的人名/地名/机构名/专名兜底，其余标记为unknown
        match jieba_tag {
            Some("nr") => "nr".to_string(),
            Some("ns") => "ns".to_string(),
            Some("nt") => "nt".to_string(),
            Some("nz") => "nz".to_string(),
            _ => "unknown".to_string(),
        }
    };

    TaggedWord { text: word, pos }
}

// 词语归一化函数，将同义词转换为标准形式
pub fn normalize_words(
    mut words: Vec<TaggedWord>,
    synonyms: &HashMap<String, String>,
) -> Vec<TaggedWord> {
    // 原地替换
    for word in &mut words {
        if let Some(canonical_word) = synonyms.get(&word.text) {
            word.text = canonical_word.clone();
        }
    }
    words
}

/// 人名碎片合并：结巴可能把「小明」切成「小/nr + 明/unknown」，
/// 配置里的提取模式（如 ["nr","m","q"]）与查询槽位便难以命中。
/// 这里把「nr + 连续单字 unknown」按最长 4 个字符合并成单一人名 token；
/// 遇到标点、数字、空白、多字词或常见疑问/功能词即停止，不吞并其它成分。
pub fn merge_name_fragments(tagged_words: &[TaggedWord]) -> Vec<TaggedWord> {
    // 单字碎片中不允许并入人名的常见疑问/功能/语气字
    const BLOCKED_FRAGMENTS: &str =
        "多少谁哪什么吗呢吧的了是有岁年月日个位名人上下前后里外中来去想要会能也都还就才\
         很最更太真不没与和跟把被给对从向为以于之而且或但在高矮胖瘦长短老新好坏男女\
         家们些时候地方怎样如何做干用说问答错该应当可需请让使叫称看听吃喝玩学生师医院";

    fn is_mergeable_fragment(word: &TaggedWord) -> bool {
        word.pos == "unknown"
            && word.text.chars().count() == 1
            && !word.text.is_empty()
            && word
                .text
                .chars()
                .all(|ch| !ch.is_numeric() && !ch.is_whitespace() && !is_boundary_char(ch))
            && !word
                .text
                .chars()
                .all(|ch| BLOCKED_FRAGMENTS.contains(ch))
    }

    fn is_boundary_char(ch: char) -> bool {
        "，。、；：？！「」『』《》【】（）()[]{}<>\"'`~,.;:!?-_|/\\".contains(ch)
    }

    let mut merged: Vec<TaggedWord> = Vec::with_capacity(tagged_words.len());
    let mut index = 0;
    while index < tagged_words.len() {
        let word = &tagged_words[index];
        if word.pos != "nr" {
            merged.push(word.clone());
            index += 1;
            continue;
        }
        let mut text = word.text.clone();
        let mut end = index + 1;
        while end < tagged_words.len() && text.chars().count() < 4 {
            let fragment = &tagged_words[end];
            if !is_mergeable_fragment(fragment) {
                break;
            }
            let candidate = format!("{}{}", text, fragment.text);
            if candidate.chars().count() > 4 {
                break;
            }
            text = candidate;
            end += 1;
        }
        merged.push(TaggedWord {
            text,
            pos: word.pos.clone(),
        });
        index = end;
    }
    merged
}

// 判断一个词是否为人名
// 注意：crate::NamePatterns 引用的是 lib.rs 中的 NamePatterns
fn is_name(
    word: &str,
    patterns: &crate::NamePatterns,
    full_names: &std::collections::HashSet<&str>,
) -> bool {
    patterns.prefixes.iter().any(|p| word.starts_with(p)) || // 以姓氏前缀开头
    patterns.suffixes.iter().any(|s| word.ends_with(s)) || // 以名字后缀结尾
    full_names.contains(word) || // 在常见人名列表中
    patterns.profession_suffix.iter().any(|ps| word.ends_with(ps) && word.len() > ps.len()) // 以职业后缀结尾且不只是后缀本身
}

// 分析查询意图并提取实体
pub fn analyze_query_intent(tagged_words: &[TaggedWord], rules: &[PrologRule]) -> Option<(String, Vec<String>)> {
    // 规则通常已按优先级降序排列：先做一次线性检查
    if rules.windows(2).all(|pair| pair[0].priority >= pair[1].priority) {
        for rule in rules {
            if let Some(entities) = match_pattern_for_prolog(tagged_words, &rule.pattern) {
                return Some((rule.query_type.clone(), entities)); // 返回查询类型和提取的实体
            }
        }
        return None; // 没有匹配的规则
    }

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
    for (pattern_index, pattern_elem) in pattern.iter().enumerate() {
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
                        // 下一个模式元素若是具体词语，合并人名时不能把它吞掉
                        // （如「张三和李四」中的人名合并不能吃掉「和」）
                        let next_literal = match pattern.get(pattern_index + 1) {
                            Some(PatternElement::Word(word)) => Some(word.as_str()),
                            _ => None,
                        };
                        
                        // 检查后续的unknown词是否应该合并
                        while next_pos < tagged_words.len() && 
                              (tagged_words[next_pos].pos == "unknown" || 
                               tagged_words[next_pos].pos == "nr") {
                            if next_literal == Some(tagged_words[next_pos].text.as_str()) {
                                break;
                            }
                            name_parts.push(tagged_words[next_pos].text.clone());
                            next_pos += 1;
                        }
                        
                        entities.push(name_parts.join(""));
                        pos = next_pos;
                    } else {
                        return None; // 不是人名
                    }
                } else if pos < tagged_words.len() && tagged_words[pos].pos == *expected_pos {
                    // 如果是名词性成分，提取为实体
                    if expected_pos == "n" || expected_pos == "nr" || expected_pos == "ns" {
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
                        let mut matched_count = 0usize;
                        let mut temp_pos = pos;
                        
                        for expected_char in &expected_chars {
                            if temp_pos < tagged_words.len() {
                                let word_text = tagged_words[temp_pos].text.as_str();
                                // 等价于 word_text == expected_char.to_string()
                                if word_text.len() == expected_char.len_utf8()
                                    && word_text.starts_with(*expected_char)
                                {
                                    matched_count += 1;
                                    temp_pos += 1;
                                    continue;
                                }
                            }
                            break;
                        }
                        
                        if matched_count == expected_chars.len() {
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
            "ns" => PatternElement::Pos("ns".to_string()), // 地名
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged(pairs: &[(&str, &str)]) -> Vec<TaggedWord> {
        pairs
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: text.to_string(),
                pos: pos.to_string(),
            })
            .collect()
    }

    #[test]
    fn test_ns_pattern_is_matched_and_captured() {
        let pattern = pattern_from_strings(vec![
            "ns".to_string(),
            "的".to_string(),
            "首都".to_string(),
            "是".to_string(),
            "哪里".to_string(),
        ]);
        let words = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = match_pattern_for_prolog(&words, &pattern).expect("ns 模式应能匹配");
        assert_eq!(matched, vec!["法国".to_string()]);
    }

    #[test]
    fn test_name_merge_stops_before_literal_conjunction() {
        let pattern = pattern_from_strings(vec![
            "nr".to_string(),
            "和".to_string(),
            "nr".to_string(),
            "什么".to_string(),
            "关系".to_string(),
        ]);
        let words = tagged(&[
            ("张三", "nr"),
            ("和", "unknown"),
            ("李四", "nr"),
            ("什么", "r"),
            ("关系", "unknown"),
        ]);
        let matched = match_pattern_for_prolog(&words, &pattern).expect("关系模式应能匹配");
        assert_eq!(matched, vec!["张三".to_string(), "李四".to_string()]);
    }
}

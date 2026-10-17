use jieba_rs::Jieba;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

thread_local! {
    /// 编辑距离计算的可复用缓冲：避免每次调用产生 4 次堆分配
    static LEVENSHTEIN_SCRATCH: RefCell<(Vec<char>, Vec<char>, Vec<usize>, Vec<usize>)> =
        RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub similarity: f64,
    pub method: String,
    pub details: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordEmbedding {
    pub word: String,
    pub vector: Vec<f32>,
    pub frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub concept: String,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub synonyms: Vec<String>,
    pub weight: f64,
}

pub struct SimilarityCalculator {
    jieba: &'static Jieba,
    word_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    word_frequencies: Arc<RwLock<HashMap<String, u32>>>,
    concept_hierarchy: Arc<RwLock<HashMap<String, ConceptNode>>>,
    synonym_dict: Arc<RwLock<HashMap<String, Vec<String>>>>,
    stop_words: HashSet<String>,
    idf_cache: Arc<RwLock<HashMap<String, f64>>>,
}

impl SimilarityCalculator {
    pub fn new() -> Self {
        let jieba = crate::nlp::shared_jieba();
        
        let stop_words = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "也", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "还", "来", "个", "对", "为", "把", "里", "后", "才", "能", "下", "这些", "但是", "如果", "只是", "什么", "怎么", "为什么", "哪里", "那么", "这么", "怎样", "多少", "几个", "如何", "这个", "那个", "这样", "那样", "因为", "所以", "然后", "接着", "最后", "首先", "其次", "另外", "此外", "而且", "并且", "或者", "还是", "虽然", "但是", "可是", "然而", "不过", "只要", "除非", "无论", "不管", "即使", "哪怕", "只有", "只是", "仅仅", "刚刚", "正在", "已经", "曾经", "将要", "可能", "应该", "必须", "需要", "想要", "希望", "愿意", "喜欢", "讨厌", "觉得", "认为", "以为", "知道", "了解", "明白", "理解", "发现", "意识"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            jieba,
            word_embeddings: Arc::new(RwLock::new(HashMap::new())),
            word_frequencies: Arc::new(RwLock::new(HashMap::new())),
            concept_hierarchy: Arc::new(RwLock::new(HashMap::new())),
            synonym_dict: Arc::new(RwLock::new(HashMap::new())),
            stop_words,
            idf_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn initialize_embeddings(&self, embeddings: Vec<WordEmbedding>) -> Result<(), Box<dyn std::error::Error>> {
        let mut word_embeddings = self.word_embeddings.write();
        let mut word_frequencies = self.word_frequencies.write();
        
        for embedding in embeddings {
            word_embeddings.insert(embedding.word.clone(), embedding.vector);
            // 频率为 0 会让 IDF 出现 ln(∞)，整个 TF-IDF 结果变成 NaN，按「未知频率」处理
            if embedding.frequency > 0 {
                word_frequencies.insert(embedding.word, embedding.frequency);
            }
        }
        
        // IDF 依赖词频统计，初始化新语料后必须失效旧缓存，否则继续使用上一份语料的 IDF
        self.idf_cache.write().clear();
        
        Ok(())
    }

    pub fn initialize_concept_hierarchy(&self, concepts: Vec<ConceptNode>) -> Result<(), Box<dyn std::error::Error>> {
        let mut concept_hierarchy = self.concept_hierarchy.write();
        let mut synonym_dict = self.synonym_dict.write();
        
        for concept in concepts {
            let concept_name = concept.concept.clone();
            
            // 添加同义词映射
            for synonym in &concept.synonyms {
                synonym_dict.entry(synonym.clone()).or_insert_with(Vec::new).push(concept_name.clone());
            }
            
            concept_hierarchy.insert(concept_name, concept);
        }
        
        Ok(())
    }

    pub async fn calculate_semantic_similarity(&self, text1: &str, text2: &str) -> Result<SimilarityResult, Box<dyn std::error::Error>> {
        let mut details = HashMap::new();

        let words1 = self.tokenize_and_filter(text1);
        let words2 = self.tokenize_and_filter(text2);

        // 1. 词汇相似度 (Jaccard similarity)
        let jaccard_sim = self.jaccard_similarity_from_words(&words1, &words2);
        details.insert("jaccard".to_string(), jaccard_sim);
        
        // 2. 余弦相似度 (基于词向量)
        let cosine_sim = self.cosine_similarity_from_words(&words1, text1, &words2, text2);
        details.insert("cosine".to_string(), cosine_sim);
        
        // 3. 编辑距离相似度
        let edit_sim = self.calculate_edit_distance_similarity(text1, text2);
        details.insert("edit_distance".to_string(), edit_sim);
        
        // 4. TF-IDF 相似度
        let tfidf_sim = self.tfidf_similarity_from_words(&words1, &words2).await;
        details.insert("tfidf".to_string(), tfidf_sim);
        
        // 5. 语义概念相似度
        let concept_sim = self.concept_similarity_from_words(&words1, &words2);
        details.insert("concept".to_string(), concept_sim);
        
        // 6. N-gram 相似度
        let ngram_sim = self.calculate_ngram_similarity(text1, text2).await;
        details.insert("ngram".to_string(), ngram_sim);
        
        // 综合相似度计算 (加权平均)
        let combined_similarity = jaccard_sim * 0.15 
            + cosine_sim * 0.25 
            + edit_sim * 0.1 
            + tfidf_sim * 0.2 
            + concept_sim * 0.2 
            + ngram_sim * 0.1;
        
        Ok(SimilarityResult {
            similarity: combined_similarity.max(0.0).min(1.0),
            method: "combined".to_string(),
            details,
        })
    }

    async fn calculate_jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1 = self.tokenize_and_filter(text1);
        let words2 = self.tokenize_and_filter(text2);
        self.jaccard_similarity_from_words(&words1, &words2)
    }

    fn jaccard_similarity_from_words(&self, words1: &[&str], words2: &[&str]) -> f64 {
        let set1: HashSet<_> = words1.iter().copied().collect();
        let set2: HashSet<_> = words2.iter().copied().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.len() + set2.len() - intersection;
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn cosine_similarity_from_words(
        &self,
        words1: &[&str],
        text1: &str,
        words2: &[&str],
        text2: &str,
    ) -> f64 {
        let vec1 = self.text_to_vector_from_words(words1, text1);
        let vec2 = self.text_to_vector_from_words(words2, text2);
        
        if vec1.is_empty() || vec2.is_empty() {
            return 0.0;
        }
        
        self.cosine_similarity(&vec1, &vec2)
    }

    fn text_to_vector_from_words(&self, words: &[&str], text: &str) -> Vec<f32> {
        if words.is_empty() {
            return vec![];
        }
        let embeddings = self.word_embeddings.read();
        
        let mut total_vector: Option<Vec<f32>> = None;
        let mut word_count = 0;
        
        for word in words {
            if let Some(embedding) = embeddings.get(*word) {
                match &mut total_vector {
                    Some(ref mut vec) => {
                        for (i, &val) in embedding.iter().enumerate() {
                            if i < vec.len() {
                                vec[i] += val;
                            }
                        }
                    }
                    None => {
                        total_vector = Some(embedding.clone());
                    }
                }
                word_count += 1;
            }
        }
        
        if let Some(mut vec) = total_vector {
            if word_count > 0 {
                for val in &mut vec {
                    *val /= word_count as f32;
                }
            }
            vec
        } else {
            // 如果没有找到词向量，使用随机向量或零向量
            self.generate_fallback_vector(text)
        }
    }

    fn generate_fallback_vector(&self, text: &str) -> Vec<f32> {
        // 基于文本特征生成简单的向量表示
        let mut vector = vec![0.0; 100]; // 100维向量
        
        for ch in text.chars() {
            let char_code = ch as u32;
            let index = (char_code % 100) as usize;
            vector[index] += 1.0;
        }
        
        // 归一化
        let magnitude: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut vector {
                *val /= magnitude;
            }
        }
        
        vector
    }

    fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let magnitude1: f32 = vec1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = vec2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            (dot_product / (magnitude1 * magnitude2)) as f64
        }
    }

    fn calculate_edit_distance_similarity(&self, text1: &str, text2: &str) -> f64 {
        let distance = self.levenshtein_distance(text1, text2);
        let max_len = text1.chars().count().max(text2.chars().count());
        
        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        // 双行动态规划：将较短字符串作为列方向；复用线程本地缓冲避免反复分配
        LEVENSHTEIN_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (chars1, chars2, prev, curr) = &mut *scratch;
            chars1.clear();
            chars1.extend(s1.chars());
            chars2.clear();
            chars2.extend(s2.chars());
            let (rows, cols) = if chars1.len() <= chars2.len() {
                (&*chars1, &*chars2)
            } else {
                (&*chars2, &*chars1)
            };
            let col_count = cols.len();

            prev.clear();
            prev.extend(0..=col_count);
            curr.clear();
            curr.resize(col_count + 1, 0);

            for (i, row_char) in rows.iter().enumerate() {
                curr[0] = i + 1;
                for j in 1..=col_count {
                    let cost = if *row_char == cols[j - 1] { 0 } else { 1 };
                    curr[j] = (prev[j] + 1)
                        .min(curr[j - 1] + 1)
                        .min(prev[j - 1] + cost);
                }
                std::mem::swap(prev, curr);
            }

            prev[col_count]
        })
    }

    async fn tfidf_similarity_from_words(&self, words1: &[&str], words2: &[&str]) -> f64 {
        let tfidf1 = self.calculate_tfidf_vector(words1).await;
        let tfidf2 = self.calculate_tfidf_vector(words2).await;
        
        self.calculate_vector_similarity(&tfidf1, &tfidf2)
    }

    async fn calculate_tfidf_vector<'a>(&self, words: &[&'a str]) -> HashMap<&'a str, f64> {
        let mut tf_map: HashMap<&'a str, f64> = HashMap::new();
        let total_words = words.len() as f64;
        
        // 计算 TF (Term Frequency)
        for word in words {
            *tf_map.entry(*word).or_insert(0.0) += 1.0;
        }
        
        for (_, tf) in tf_map.iter_mut() {
            *tf /= total_words;
        }
        
        // 计算 TF-IDF；先整批命中缓存（一次读锁），剩余未命中词再统一计算
        let mut tfidf_vector: HashMap<&'a str, f64> = HashMap::with_capacity(tf_map.len());
        {
            let cache = self.idf_cache.read();
            for (&word, &tf) in &tf_map {
                if let Some(&idf) = cache.get(word) {
                    tfidf_vector.insert(word, tf * idf);
                }
            }
        }
        if tfidf_vector.len() < tf_map.len() {
            let word_frequencies = self.word_frequencies.read();
            let total_documents = word_frequencies.len() as f64;
            let mut freshly_computed: Vec<(&'a str, f64)> = Vec::new();
            for (&word, &tf) in &tf_map {
                if tfidf_vector.contains_key(word) {
                    continue;
                }
                if total_documents == 0.0 {
                    tfidf_vector.insert(word, tf);
                } else {
                    let document_frequency = (*word_frequencies.get(word).unwrap_or(&1) as f64).max(1.0);
                    let idf = (total_documents / document_frequency).ln();
                    tfidf_vector.insert(word, tf * idf);
                    freshly_computed.push((word, idf));
                }
            }
            drop(word_frequencies);
            if !freshly_computed.is_empty() {
                let mut cache = self.idf_cache.write();
                for (word, idf) in freshly_computed {
                    cache.insert(word.to_string(), idf);
                }
            }
        }
        
        tfidf_vector
    }

    fn calculate_vector_similarity(&self, vec1: &HashMap<&str, f64>, vec2: &HashMap<&str, f64>) -> f64 {
        if vec1.is_empty() || vec2.is_empty() {
            return 0.0;
        }
        
        let mut dot_product = 0.0;
        let mut magnitude1 = 0.0;
        let mut magnitude2 = 0.0;
        
        // 共用词在两次遍历中只需各贡献一次；仅 vec2 独有的词在第二遍补齐
        for (term, val1) in vec1 {
            let val2 = vec2.get(term).copied().unwrap_or(0.0);
            dot_product += val1 * val2;
            magnitude1 += val1 * val1;
            magnitude2 += val2 * val2;
        }
        for (term, val2) in vec2 {
            if !vec1.contains_key(term) {
                magnitude2 += val2 * val2;
            }
        }
        
        magnitude1 = magnitude1.sqrt();
        magnitude2 = magnitude2.sqrt();
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }

    fn concept_similarity_from_words(&self, words1: &[&str], words2: &[&str]) -> f64 {
        let concepts1 = self.extract_concepts_from_words(words1);
        let concepts2 = self.extract_concepts_from_words(words2);
        
        if concepts1.is_empty() || concepts2.is_empty() {
            return 0.0;
        }
        
        // 整个配对循环只取一次读锁
        let concept_hierarchy = self.concept_hierarchy.read();
        let mut total_similarity = 0.0;
        let mut comparisons = 0;
        
        for concept1 in &concepts1 {
            for concept2 in &concepts2 {
                let similarity = if concept1 == concept2 {
                    1.0
                } else {
                    match (concept_hierarchy.get(concept1), concept_hierarchy.get(concept2)) {
                        (Some(n1), Some(n2)) => {
                            // 计算概念层次相似度
                            let hierarchy_sim = Self::calculate_hierarchy_similarity(n1, n2);
                            // 计算同义词相似度
                            let synonym_sim = Self::calculate_synonym_similarity(n1, n2);
                            (hierarchy_sim + synonym_sim) / 2.0
                        }
                        _ => 0.0,
                    }
                };
                total_similarity += similarity;
                comparisons += 1;
            }
        }
        
        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    fn extract_concepts_from_words(&self, words: &[&str]) -> Vec<String> {
        let concept_hierarchy = self.concept_hierarchy.read();
        let synonym_dict = self.synonym_dict.read();
        let mut concepts = Vec::new();
        
        for word in words {
            // 直接概念匹配
            if concept_hierarchy.contains_key(*word) {
                concepts.push(word.to_string());
            }
            
            // 通过同义词查找概念
            if let Some(concept_list) = synonym_dict.get(*word) {
                concepts.extend(concept_list.clone());
            }
        }
        
        concepts.sort();
        concepts.dedup();
        concepts
    }

    fn calculate_hierarchy_similarity(node1: &ConceptNode, node2: &ConceptNode) -> f64 {
        // 简化的层次相似度计算
        // 实际实现中可以使用更复杂的算法，如 Wu-Palmer 相似度
        
        if let (Some(parent1), Some(parent2)) = (&node1.parent, &node2.parent) {
            if parent1 == parent2 {
                return 0.8; // 同一父概念下的兄弟概念
            }
        }
        
        // 检查是否为父子关系
        if node1.children.contains(&node2.concept) || node2.children.contains(&node1.concept) {
            return 0.6;
        }
        
        0.0
    }

    fn calculate_synonym_similarity(node1: &ConceptNode, node2: &ConceptNode) -> f64 {
        let synonyms1: HashSet<_> = node1.synonyms.iter().collect();
        let synonyms2: HashSet<_> = node2.synonyms.iter().collect();
        
        let intersection = synonyms1.intersection(&synonyms2).count();
        let union = synonyms1.len() + synonyms2.len() - intersection;
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    async fn calculate_ngram_similarity(&self, text1: &str, text2: &str) -> f64 {
        // 收集字符序列用于 bigram/trigram 相似度
        let chars1: Vec<char> = text1.chars().collect();
        let chars2: Vec<char> = text2.chars().collect();
        
        let bigram_sim = Self::calculate_ngram_set_similarity(
            &Self::bigrams_from_chars(&chars1),
            &Self::bigrams_from_chars(&chars2),
        );
        let trigram_sim = Self::calculate_ngram_set_similarity(
            &Self::trigrams_from_chars(&chars1),
            &Self::trigrams_from_chars(&chars2),
        );
        
        (bigram_sim + trigram_sim) / 2.0
    }

    /// 二元组集合：用 (char, char) 作为键
    fn bigrams_from_chars(chars: &[char]) -> HashSet<(char, char)> {
        let mut ngrams = HashSet::new();
        if chars.len() >= 2 {
            for window in chars.windows(2) {
                ngrams.insert((window[0], window[1]));
            }
        }
        ngrams
    }

    /// 三元组集合：用 (char, char, char) 作为键
    fn trigrams_from_chars(chars: &[char]) -> HashSet<(char, char, char)> {
        let mut ngrams = HashSet::new();
        if chars.len() >= 3 {
            for window in chars.windows(3) {
                ngrams.insert((window[0], window[1], window[2]));
            }
        }
        ngrams
    }

    #[cfg(test)]
    fn ngrams_from_chars(chars: &[char], n: usize) -> HashSet<String> {
        let mut ngrams = HashSet::new();
        
        if chars.len() >= n {
            for i in 0..=chars.len() - n {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.insert(ngram);
            }
        }
        
        ngrams
    }

    #[cfg(test)]
    fn generate_character_ngrams(&self, text: &str, n: usize) -> HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        Self::ngrams_from_chars(&chars, n)
    }

    fn calculate_ngram_set_similarity<T: std::hash::Hash + Eq>(
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> f64 {
        let intersection = set1.intersection(set2).count();
        let union = set1.len() + set2.len() - intersection;
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn tokenize_and_filter<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.jieba.cut(text, false)
            .into_iter()
            .filter(|word| !self.stop_words.contains(*word) && word.trim().chars().count() > 1)
            .collect()
    }

    pub async fn calculate_simple_similarity(&self, text1: &str, text2: &str) -> f64 {
        if text1 == text2 {
            return 1.0;
        }
        
        // 快速相似度计算，适用于实时查询
        let words1 = self.tokenize_and_filter(text1);
        let words2 = self.tokenize_and_filter(text2);
        let jaccard = self.jaccard_similarity_from_words(&words1, &words2);
        let edit = self.calculate_edit_distance_similarity(text1, text2);
        
        (jaccard * 0.7 + edit * 0.3).max(0.0).min(1.0)
    }

    pub fn add_word_embedding(&self, word: String, vector: Vec<f32>) {
        let mut embeddings = self.word_embeddings.write();
        embeddings.insert(word, vector);
    }

    pub fn add_concept(&self, concept: ConceptNode) {
        let concept_name = concept.concept.clone();
        let mut concept_hierarchy = self.concept_hierarchy.write();
        let mut synonym_dict = self.synonym_dict.write();
        
        // 添加同义词映射
        for synonym in &concept.synonyms {
            synonym_dict.entry(synonym.clone()).or_insert_with(Vec::new).push(concept_name.clone());
        }
        
        concept_hierarchy.insert(concept_name, concept);
    }

    pub fn get_embedding_count(&self) -> usize {
        self.word_embeddings.read().len()
    }

    pub fn get_concept_count(&self) -> usize {
        self.concept_hierarchy.read().len()
    }
}

impl Default for SimilarityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jaccard_similarity() {
        let calculator = SimilarityCalculator::new();
        
        let sim = calculator.calculate_jaccard_similarity("苹果很好吃", "苹果非常美味").await;
        assert!(sim > 0.0);
        
        let sim = calculator.calculate_jaccard_similarity("苹果", "香蕉").await;
        assert!(sim < 0.5);
    }

    #[tokio::test]
    async fn test_edit_distance_similarity() {
        let calculator = SimilarityCalculator::new();
        
        let sim = calculator.calculate_edit_distance_similarity("苹果", "苹果");
        assert_eq!(sim, 1.0);
        
        let sim = calculator.calculate_edit_distance_similarity("苹果", "苹");
        assert!(sim >= 0.5); // 编辑距离为1，最大长度为2，相似度为0.5
    }

    #[tokio::test]
    async fn test_semantic_similarity() {
        let calculator = SimilarityCalculator::new();
        
        let result = calculator.calculate_semantic_similarity("刘德华是演员", "华仔是明星").await.unwrap();
        assert!(result.similarity >= 0.0 && result.similarity <= 1.0);
        assert_eq!(result.method, "combined");
        assert!(!result.details.is_empty());
    }

    #[test]
    fn test_ngram_generation() {
        let calculator = SimilarityCalculator::new();
        
        let bigrams = calculator.generate_character_ngrams("苹果", 2);
        assert!(bigrams.contains("苹果"));
        
        let trigrams = calculator.generate_character_ngrams("苹果很好", 3);
        assert!(trigrams.len() > 0);
    }

    #[tokio::test]
    async fn test_simple_similarity() {
        let calculator = SimilarityCalculator::new();
        
        let sim = calculator.calculate_simple_similarity("测试文本", "测试文本").await;
        assert_eq!(sim, 1.0);
        
        let sim = calculator.calculate_simple_similarity("测试文本", "测试").await;
        assert!(sim > 0.0 && sim < 1.0);
    }
}
// src/dense_retrieval.rs
//! 符号主义「稠密检索」：把每条知识当作一篇文档，用纯统计与图游走信号编码成
//! 稀疏高维向量（TF-IDF 词面 × 结构 PPR × 谓词匹配），查询与文档做加权内积。
//!
//! 与神经网络检索器不同，这里没有任何可学习参数：
//! - 词面向量：TF-IDF 倒排索引（df 来自语料统计，无训练）；
//! - 结构向量：在实体-实体图上做随机游走重启（Personalized PageRank）得到
//!   与查询实体结构相关的分布，作为「稠密」语义信号；
//! - 谓词匹配：查询词与谓词的归一化匹配作为精确信号。
//! 三种信号线性融合，输出可解释的证据排序，供生成式回答接地（RAG 的符号版）。

use std::collections::{HashMap, HashSet};

use crate::nlp::shared_jieba;

/// 查询/文档共用的停用词（功能词不携带检索信息）
const STOPWORDS: [&str; 34] = [
    "的", "了", "是", "在", "和", "与", "也", "都", "就", "吗", "呢", "吧", "啊", "什么", "多少",
    "几", "哪", "哪个", "哪些", "谁", "怎么", "为什么", "如何", "请问", "帮我", "告诉", "我",
    "一下", "这个", "那个", "他", "她", "它", "们",
];

/// 单条被检索的证据（三元组或实体属性）
#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
    pub lexical: f64,
    pub structural: f64,
    pub predicate_match: f64,
    pub reasons: Vec<String>,
}

impl RetrievalHit {
    /// 证据的线性化文本（用于回答引用）
    pub fn text(&self) -> String {
        format!("{} 的 {} 是 {}", self.subject, self.predicate, self.object)
    }

    /// 融合打分的分解说明
    pub fn breakdown(&self) -> String {
        format!(
            "总 {:.3} = 词面 {:.3} + 结构 {:.3} + 谓词 {:.3}",
            self.score, self.lexical, self.structural, self.predicate_match
        )
    }
}

/// 一篇文档的内部记录
struct DocRecord {
    subject: String,
    predicate: String,
    object: String,
    terms: Vec<String>,
}

/// 符号稠密检索器
pub struct DenseRetriever {
    top_k: usize,
    damping: f64,
    iterations: usize,
    lexical_weight: f64,
    structural_weight: f64,
    predicate_weight: f64,
    docs: Vec<DocRecord>,
    seen: HashSet<(String, String, String)>,
    df: HashMap<String, usize>,
    postings: HashMap<String, Vec<(usize, f64)>>,
    doc_norms: Vec<f64>,
    entity_docs: HashMap<String, Vec<usize>>,
    adjacency: HashMap<String, HashMap<String, f64>>,
    dirty: bool,
    built: bool,
}

impl DenseRetriever {
    pub fn new(top_k: usize, damping: f64, iterations: usize) -> Self {
        Self {
            top_k: top_k.clamp(1, 50),
            damping: damping.clamp(0.1, 0.99),
            iterations: iterations.clamp(1, 64),
            lexical_weight: 0.5,
            structural_weight: 0.35,
            predicate_weight: 0.15,
            docs: Vec::new(),
            seen: HashSet::new(),
            df: HashMap::new(),
            postings: HashMap::new(),
            doc_norms: Vec::new(),
            entity_docs: HashMap::new(),
            adjacency: HashMap::new(),
            dirty: false,
            built: false,
        }
    }

    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    pub fn entity_count(&self) -> usize {
        self.entity_docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// 观察一条关系三元组（重复观察会被去重）
    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        let key = (subject.to_string(), predicate.to_string(), object.to_string());
        if !self.seen.insert(key) {
            return;
        }
        let mut terms = index_terms(subject);
        terms.extend(index_terms(predicate));
        terms.extend(index_terms(object));
        let mut unique = Vec::with_capacity(terms.len());
        let mut seen_terms: HashSet<String> = HashSet::new();
        for term in terms {
            if seen_terms.insert(term.clone()) {
                unique.push(term);
            }
        }
        let id = self.docs.len();
        self.docs.push(DocRecord {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            terms: unique,
        });
        self.entity_docs.entry(subject.to_string()).or_default().push(id);
        if object != subject {
            self.entity_docs.entry(object.to_string()).or_default().push(id);
        }
        // 无向一跳邻接（结构游走的图）：主体与客体互为邻居
        if subject != object {
            self.adjacency
                .entry(subject.to_string())
                .or_default()
                .entry(object.to_string())
                .and_modify(|weight| *weight += 1.0)
                .or_insert(1.0);
            self.adjacency
                .entry(object.to_string())
                .or_default()
                .entry(subject.to_string())
                .and_modify(|weight| *weight += 1.0)
                .or_insert(1.0);
        }
        self.dirty = true;
    }

    /// 观察一个实体的类型与属性（属性逐条转成证据文档）
    pub fn observe_entity(&mut self, name: &str, entity_type: &str, properties: &[(String, String)]) {
        if !entity_type.trim().is_empty() {
            self.observe_triple(name, "类型", entity_type);
        }
        for (key, value) in properties {
            self.observe_triple(name, key, value);
        }
    }

    /// 惰性重建倒排索引（只有新事实时才需要重算）
    fn rebuild(&mut self) {
        if self.built && !self.dirty {
            return;
        }
        let n = self.docs.len();
        self.df.clear();
        self.postings.clear();
        self.doc_norms = vec![0.0; n];
        let mut term_freq: Vec<HashMap<&str, f64>> = Vec::with_capacity(n);
        for (id, doc) in self.docs.iter().enumerate() {
            let mut tf: HashMap<&str, f64> = HashMap::new();
            for term in &doc.terms {
                *tf.entry(term.as_str()).or_insert(0.0) += 1.0;
            }
            for term in tf.keys() {
                *self.df.entry((*term).to_string()).or_insert(0) += 1;
            }
            let _ = id;
            term_freq.push(tf);
        }
        // 第二遍：用 df 计算 idf 与归一化权重
        for (id, tf) in term_freq.iter().enumerate() {
            let mut norm_sq = 0.0;
            for (term, freq) in tf {
                let idf = self.idf(term, n);
                let weight = (1.0 + freq.ln()) * idf;
                if weight > 0.0 {
                    self.postings
                        .entry((*term).to_string())
                        .or_default()
                        .push((id, weight));
                    norm_sq += weight * weight;
                }
            }
            self.doc_norms[id] = norm_sq.sqrt();
        }
        self.dirty = false;
        self.built = true;
    }

    fn idf(&self, term: &str, total: usize) -> f64 {
        let df = self.df.get(term).copied().unwrap_or(0);
        (1.0 + total as f64 / (1.0 + df as f64)).ln()
    }

    /// 在查询串中找出已知实体作为结构游走的种子（最长优先）
    pub fn seeds(&self, query: &str) -> Vec<String> {
        let mut names: Vec<&String> = self.entity_docs.keys().collect();
        names.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then_with(|| a.cmp(b)));
        let mut seeds = Vec::new();
        for name in names {
            if name.chars().count() < 2 {
                continue;
            }
            if query.contains(name.as_str()) && !seeds.contains(name) {
                seeds.push(name.clone());
                if seeds.len() >= 8 {
                    break;
                }
            }
        }
        seeds
    }

    /// 查询词向量的 tf-idf 权重（tf 按查询内出现次数）
    fn query_vector(&self, query: &str, seeds: &[String]) -> HashMap<String, f64> {
        let total = self.docs.len().max(1);
        let mut weights: HashMap<String, f64> = HashMap::new();
        for token in tokenize(query) {
            let idf = if self.df.contains_key(&token) {
                self.idf(&token, total)
            } else {
                (1.0 + total as f64).ln()
            };
            *weights.entry(token).or_insert(0.0) += idf;
        }
        for seed in seeds {
            let idf = if self.df.contains_key(seed) {
                self.idf(seed, total)
            } else {
                (1.0 + total as f64).ln()
            };
            *weights.entry(seed.clone()).or_insert(0.0) += idf * 2.0;
        }
        weights
    }

    fn query_norm(vector: &HashMap<String, f64>) -> f64 {
        vector.values().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// 结构信号：以查询实体为种子的随机游走重启分布（PPR）
    fn structural_scores(&self, seeds: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = HashMap::new();
        if seeds.is_empty() {
            return scores;
        }
        let personalization: HashMap<&str, f64> = seeds
            .iter()
            .map(|seed| (seed.as_str(), 1.0 / seeds.len() as f64))
            .collect();
        let mut current: HashMap<String, f64> = personalization
            .iter()
            .map(|(name, weight)| ((*name).to_string(), *weight))
            .collect();
        for _ in 0..self.iterations {
            let mut next: HashMap<String, f64> = personalization
                .iter()
                .map(|(name, weight)| ((*name).to_string(), (1.0 - self.damping) * weight))
                .collect();
            for (node, mass) in &current {
                if *mass <= 1e-12 {
                    continue;
                }
                let neighbors = match self.adjacency.get(node) {
                    Some(map) if !map.is_empty() => map,
                    _ => {
                        // 悬挂节点：质量留在原地，避免概率流失
                        *next.entry(node.clone()).or_insert(0.0) += self.damping * mass;
                        continue;
                    }
                };
                let total: f64 = neighbors.values().sum();
                if total <= 0.0 {
                    *next.entry(node.clone()).or_insert(0.0) += self.damping * mass;
                    continue;
                }
                for (neighbor, weight) in neighbors {
                    *next.entry(neighbor.clone()).or_insert(0.0) +=
                        self.damping * mass * (weight / total);
                }
            }
            current = next;
        }
        let max = current.values().cloned().fold(0.0_f64, f64::max);
        if max <= 0.0 {
            return scores;
        }
        for (node, mass) in current {
            scores.insert(node, mass / max);
        }
        scores
    }

    /// 检索与查询最相关的证据（融合三种符号信号），结果按分数降序
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<RetrievalHit> {
        self.rebuild();
        if self.docs.is_empty() {
            return Vec::new();
        }
        let query = query.trim();
        if query.is_empty() {
            return Vec::new();
        }
        let seeds = self.seeds(query);
        let query_vector = self.query_vector(query, &seeds);
        if query_vector.is_empty() && seeds.is_empty() {
            return Vec::new();
        }
        let query_norm = Self::query_norm(&query_vector).max(1e-9);
        let structural = self.structural_scores(&seeds);
        let query_tokens: HashSet<String> = tokenize(query).into_iter().collect();

        let mut candidate_ids: HashSet<usize> = HashSet::new();
        for term in query_vector.keys() {
            if let Some(postings) = self.postings.get(term) {
                for (id, _) in postings {
                    candidate_ids.insert(*id);
                }
            }
        }
        for seed in &seeds {
            if let Some(ids) = self.entity_docs.get(seed) {
                for id in ids.iter().take(400) {
                    candidate_ids.insert(*id);
                }
            }
        }
        // 结构游走扩散出的高权重实体也进入候选（保证结构信号可召回词面不重合的证据）
        let mut structural_ranked: Vec<(&String, &f64)> = structural.iter().collect();
        structural_ranked.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        for (entity, score) in structural_ranked.into_iter().take(20) {
            if *score < 0.05 {
                break;
            }
            if let Some(ids) = self.entity_docs.get(entity) {
                for id in ids.iter().take(50) {
                    candidate_ids.insert(*id);
                }
            }
        }

        let mut hits: Vec<RetrievalHit> = Vec::new();
        for id in candidate_ids {
            let doc = &self.docs[id];
            let mut dot = 0.0;
            for term in query_vector.keys() {
                if doc.terms.iter().any(|candidate| candidate == term) {
                    let idf = self.idf(term, self.docs.len());
                    dot += query_vector[term] * idf;
                }
            }
            let lexical = if self.doc_norms[id] > 0.0 {
                dot / (self.doc_norms[id] * query_norm)
            } else {
                0.0
            };
            let structural_score = (structural
                .get(&doc.subject)
                .copied()
                .unwrap_or(0.0)
                + structural.get(&doc.object).copied().unwrap_or(0.0))
            .min(1.0);
            let predicate_hit = if query_tokens.contains(&doc.predicate)
                || query.contains(doc.predicate.as_str())
            {
                1.0
            } else {
                0.0
            };
            let score = self.lexical_weight * lexical
                + self.structural_weight * structural_score
                + self.predicate_weight * predicate_hit;
            if score <= 0.0 {
                continue;
            }
            let mut reasons = Vec::new();
            if predicate_hit > 0.0 {
                reasons.push(format!("谓词「{}」直接命中", doc.predicate));
            }
            if structural_score > 0.1 {
                reasons.push(format!("与查询实体结构关联（PPR {:.2}）", structural_score));
            }
            if lexical > 0.05 {
                reasons.push(format!("词面重合（余弦 {:.2}）", lexical));
            }
            hits.push(RetrievalHit {
                subject: doc.subject.clone(),
                predicate: doc.predicate.clone(),
                object: doc.object.clone(),
                score,
                lexical,
                structural: structural_score,
                predicate_match: predicate_hit,
                reasons,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.subject.cmp(&b.subject))
                .then_with(|| a.predicate.cmp(&b.predicate))
                .then_with(|| a.object.cmp(&b.object))
        });
        hits.truncate(k.clamp(1, 50));
        hits
    }

    /// 默认 top-k 检索
    pub fn retrieve_top(&mut self, query: &str) -> Vec<RetrievalHit> {
        self.retrieve(query, self.top_k)
    }

    /// 渲染检索结果（含每种信号的分解与理由，便于解释生成依据）
    pub fn format_hits(&self, query: &str, hits: &[RetrievalHit]) -> String {
        if hits.is_empty() {
            return format!("没有找到与「{}」相关的证据。", query.trim());
        }
        let mut output = format!(
            "符号稠密检索（词面 TF-IDF × 结构 PPR × 谓词匹配，共 {} 条文档）：\n",
            self.docs.len()
        );
        for (index, hit) in hits.iter().enumerate() {
            let reasons = if hit.reasons.is_empty() {
                String::new()
            } else {
                format!("（{}）", hit.reasons.join("；"))
            };
            output.push_str(&format!(
                "  {}. {} ［{}］{}\n",
                index + 1,
                hit.text(),
                hit.breakdown(),
                reasons
            ));
        }
        output.trim_end().to_string()
    }
}

/// 分词并构造索引词项：结巴切词 + 中文二元组（提升词面召回）
pub fn tokenize(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut tokens: Vec<String> = Vec::new();
    for word in shared_jieba().cut(text, false) {
        let word = word.trim();
        if word.is_empty() || is_punctuation(word) || STOPWORDS.contains(&word) {
            continue;
        }
        let normalized = word.to_lowercase();
        if !tokens.contains(&normalized) {
            tokens.push(normalized);
        }
    }
    tokens
}

/// 词项集合（分词 + 二元组），用于文档向量
fn index_terms(text: &str) -> Vec<String> {
    let mut terms = tokenize(text);
    let compact = text.trim();
    let chars: Vec<char> = compact.chars().filter(|ch| !is_punctuation_char(*ch)).collect();
    if chars.len() >= 2 && chars.len() <= 24 {
        if !terms.contains(&compact.to_string()) {
            terms.push(compact.to_string());
        }
        for window in chars.windows(2) {
            let bigram: String = window.iter().collect();
            if !terms.contains(&bigram) {
                terms.push(bigram);
            }
        }
    }
    terms
}

fn is_punctuation(word: &str) -> bool {
    word.chars().all(is_punctuation_char)
}

fn is_punctuation_char(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。'
            | '？'
            | '！'
            | '、'
            | '；'
            | '：'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '（'
            | '）'
            | '《'
            | '》'
            | '—'
            | '…'
            | ','
            | '.'
            | '?'
            | '!'
            | ';'
            | ':'
            | '"'
            | '\''
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | ' '
            | '\t'
            | '\n'
            | '\r'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_retriever() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_entity("张三", "人", &[
            ("年龄".to_string(), "30".to_string()),
            ("职业".to_string(), "演员".to_string()),
            ("国籍".to_string(), "中国".to_string()),
        ]);
        retriever.observe_entity("李四", "人", &[
            ("年龄".to_string(), "28".to_string()),
            ("职业".to_string(), "导演".to_string()),
            ("国籍".to_string(), "德国".to_string()),
        ]);
        retriever.observe_triple("张三", "老师", "王五");
        retriever.observe_triple("李四", "老师", "王五");
        retriever.observe_triple("王五", "职业", "教授");
        retriever
    }

    #[test]
    fn retrieves_by_predicate_and_entity() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("张三的年龄是多少", 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].subject, "张三");
        assert_eq!(hits[0].predicate, "年龄");
        assert_eq!(hits[0].object, "30");
        assert!(hits[0].predicate_match > 0.0);
    }

    #[test]
    fn structural_signal_links_neighbors() {
        let mut retriever = build_retriever();
        // 「王五」不在查询词面中，但结构上游走可以召回王五的证据
        let hits = retriever.retrieve("张三的老师是谁", 10);
        assert!(hits.iter().any(|hit| hit.subject == "王五"));
    }

    #[test]
    fn explicit_entity_beats_unrelated_doc() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("李四 国籍", 5);
        assert_eq!(hits[0].subject, "李四");
        assert_eq!(hits[0].predicate, "国籍");
    }

    #[test]
    fn empty_or_unknown_query_yields_no_or_low_hits() {
        let mut retriever = build_retriever();
        assert!(retriever.retrieve("", 5).is_empty());
        let hits = retriever.retrieve("量子色动力学", 5);
        assert!(hits.is_empty());
    }

    #[test]
    fn format_hits_contains_breakdown() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("张三的年龄是多少", 2);
        let text = retriever.format_hits("张三的年龄是多少", &hits);
        assert!(text.contains("词面"));
        assert!(text.contains("结构"));
        assert!(text.contains("张三"));
    }

    #[test]
    fn deduplicates_repeated_observations() {
        let mut retriever = DenseRetriever::new(5, 0.85, 8);
        retriever.observe_triple("张三", "年龄", "30");
        retriever.observe_triple("张三", "年龄", "30");
        assert_eq!(retriever.doc_count(), 1);
    }
}

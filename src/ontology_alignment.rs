use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TermKind {
    #[default]
    Entity,
    Relation,
    Attribute,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentKind {
    Synonym,
    Normalized,
    Similarity,
    Structural,
}

#[derive(Debug, Clone)]
pub struct AlignmentPair {
    pub source: String,
    pub target: String,
    pub kind: AlignmentKind,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TermProfile {
    pub term: String,
    pub kind: TermKind,
    pub neighbors: HashSet<String>,
    pub types: HashSet<String>,
}

pub struct OntologyAligner {
    synonyms: HashMap<String, String>,
    profiles: Vec<TermProfile>,
    // 与 profiles 一一对应，注册时预计算规范化结果
    profile_norms: Vec<String>,
    index: HashMap<String, usize>,
    similarity_threshold: f64,
    structural_min_shared: usize,
    structural_min_jaccard: f64,
    // 术语解析结果缓存：注册新术语时整体失效；用 Arc 避免命中时的深拷贝
    resolve_cache: RwLock<HashMap<String, Option<Arc<(AlignmentPair, Option<usize>)>>>>,
}

impl OntologyAligner {
    pub fn new(synonyms: HashMap<String, String>, similarity_threshold: f64) -> Self {
        Self {
            synonyms,
            profiles: Vec::new(),
            profile_norms: Vec::new(),
            index: HashMap::new(),
            similarity_threshold,
            structural_min_shared: 2,
            structural_min_jaccard: 0.3,
            resolve_cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn term_count(&self) -> usize {
        self.profiles.len()
    }

    pub fn profiles(&self) -> &[TermProfile] {
        &self.profiles
    }

    pub fn synonyms(&self) -> &HashMap<String, String> {
        &self.synonyms
    }

    pub fn is_registered(&self, term: &str) -> bool {
        self.index.contains_key(&self.normalize(term))
    }

    pub fn kind_of(&self, term: &str) -> Option<TermKind> {
        self.index
            .get(&self.normalize(term))
            .map(|&idx| self.profiles[idx].kind)
    }

    pub fn register_term(&mut self, term: &str, kind: TermKind) -> usize {
        let normalized = self.normalize(term);
        if let Some(&idx) = self.index.get(&normalized) {
            // 同一术语先以属性注册、后以关系出现时（RDF 对象属性），升级为关系，
            // 以便「关系/属性」判定与谓词问答使用正确语义。
            if kind == TermKind::Relation && self.profiles[idx].kind == TermKind::Attribute {
                self.profiles[idx].kind = TermKind::Relation;
                self.resolve_cache.write().clear();
            }
            return idx;
        }
        self.profiles.push(TermProfile {
            term: term.to_string(),
            kind,
            ..Default::default()
        });
        let idx = self.profiles.len() - 1;
        self.profile_norms.push(normalized.clone());
        self.index.insert(normalized, idx);
        self.resolve_cache.write().clear();
        idx
    }

    /// 返回第 `idx` 个术语的预计算规范化形式
    pub fn normalized_term_at(&self, idx: usize) -> Option<&str> {
        self.profile_norms.get(idx).map(String::as_str)
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let s_idx = self.register_term(subject, TermKind::Entity);
        let o_idx = self.register_term(object, TermKind::Entity);
        self.register_term(predicate, TermKind::Relation);
        if s_idx != o_idx {
            let forward_inserted = self.profiles[s_idx]
                .neighbors
                .insert(format!("{}{}{}", predicate, '\u{2192}', object));
            let backward_inserted = self.profiles[o_idx]
                .neighbors
                .insert(format!("{}{}{}", predicate, '\u{2190}', subject));
            if forward_inserted || backward_inserted {
                self.resolve_cache.write().clear();
            }
        }
    }

    pub fn normalize(&self, term: &str) -> String {
        let mut out = String::with_capacity(term.len());
        for ch in term.trim().chars() {
            if ch.is_whitespace() {
                continue;
            }
            let mapped = match ch {
                '（' => '(',
                '）' => ')',
                '：' => ':',
                '，' => ',',
                '。' => '.',
                '、' => ',',
                '；' => ';',
                '！' => '!',
                '？' => '?',
                '《' => '<',
                '》' => '>',
                '【' => '[',
                '】' => ']',
                '“' | '”' => '"',
                '‘' | '’' => '\'',
                '－' | '—' | '–' => '-',
                '·' => '.',
                _ => ch,
            };
            if matches!(
                mapped,
                '(' | ')'
                    | ':'
                    | ','
                    | '.'
                    | ';'
                    | '!'
                    | '?'
                    | '<'
                    | '>'
                    | '['
                    | ']'
                    | '"'
                    | '\''
                    | '-'
                    | '_'
                    | '/'
                    | '\\'
                    | '#'
                    | '@'
                    | '&'
                    | '+'
                    | '*'
            ) {
                continue;
            }
            if mapped.is_ascii_uppercase() {
                out.push(mapped.to_ascii_lowercase());
            } else {
                out.push(mapped);
            }
        }
        out
    }

    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        self.similarity_normalized(&self.normalize(a), &self.normalize(b))
    }

    /// 对已规范化的字符串直接计算相似度
    fn similarity_normalized(&self, na: &str, nb: &str) -> f64 {
        if na.is_empty() || nb.is_empty() {
            return 0.0;
        }
        if na == nb {
            return 1.0;
        }
        let na_len = na.chars().count();
        let nb_len = nb.chars().count();
        let (short, long, short_len, long_len) = if na_len <= nb_len {
            (na, nb, na_len, nb_len)
        } else {
            (nb, na, nb_len, na_len)
        };
        if short_len >= 2 && long.contains(short) {
            return 0.7 + 0.3 * (short_len as f64 / long_len as f64);
        }
        Self::bigram_dice(short, long)
    }

    fn bigram_dice(a: &str, b: &str) -> f64 {
        Self::bigram_dice_keys(&Self::bigram_keys(a), &Self::bigram_keys(b))
    }

    fn bigram_dice_keys(ga: &[(char, Option<char>)], gb: &[(char, Option<char>)]) -> f64 {
        if ga.is_empty() || gb.is_empty() {
            return 0.0;
        }
        let total = ga.len() + gb.len();
        // 计数映射实现多重集交集，避免逐元素线性查找的 O(n²)
        let mut counts: HashMap<(char, Option<char>), usize> = HashMap::with_capacity(ga.len());
        for key in ga {
            *counts.entry(*key).or_insert(0) += 1;
        }
        let mut overlap = 0usize;
        for key in gb {
            if let Some(count) = counts.get_mut(key) {
                if *count > 0 {
                    *count -= 1;
                    overlap += 1;
                }
            }
        }
        2.0 * overlap as f64 / total as f64
    }

    /// 与 `similarity_normalized` 等价，但复用调用方预计算的查询侧 bigram，
    /// 避免在一批候选中反复重建同一个字符串的 bigram 表。
    fn similarity_normalized_with_keys(
        &self,
        query: &str,
        query_keys: &[(char, Option<char>)],
        other: &str,
    ) -> f64 {
        if query.is_empty() || other.is_empty() {
            return 0.0;
        }
        if query == other {
            return 1.0;
        }
        let q_len = query.chars().count();
        let o_len = other.chars().count();
        let (short, long, short_len, long_len) = if q_len <= o_len {
            (query, other, q_len, o_len)
        } else {
            (other, query, o_len, q_len)
        };
        if short_len >= 2 && long.contains(short) {
            return 0.7 + 0.3 * (short_len as f64 / long_len as f64);
        }
        let other_keys = Self::bigram_keys(other);
        if short == query {
            Self::bigram_dice_keys(query_keys, &other_keys)
        } else {
            Self::bigram_dice_keys(&other_keys, query_keys)
        }
    }

    /// 以 (字符, 下一字符) 作为 bigram 键；
    /// 长度不足 2 的字符串使用单元素语义（第二项为 None）。
    fn bigram_keys(s: &str) -> Vec<(char, Option<char>)> {
        let mut keys = Vec::new();
        let mut prev: Option<char> = None;
        for ch in s.chars() {
            if let Some(p) = prev {
                keys.push((p, Some(ch)));
            }
            prev = Some(ch);
        }
        if keys.is_empty() {
            if let Some(single) = prev {
                keys.push((single, None));
            }
        }
        keys
    }

    fn synonym_chain(&self, term: &str) -> Option<String> {
        let mut current = term.to_string();
        let mut current_norm = self.normalize(term);
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(current_norm.clone());
        let mut changed = false;
        for _ in 0..8 {
            let next = self
                .synonyms
                .get(current.as_str())
                .cloned()
                .or_else(|| self.synonyms.get(&current_norm).cloned());
            match next {
                Some(target) if !target.is_empty() => {
                    let target_norm = self.normalize(&target);
                    if seen.contains(&target_norm) {
                        break;
                    }
                    seen.insert(target_norm.clone());
                    current = target;
                    current_norm = target_norm;
                    changed = true;
                }
                _ => break,
            }
        }
        if changed {
            Some(current)
        } else {
            None
        }
    }

    fn resolve_arc(&self, term: &str) -> Option<Arc<(AlignmentPair, Option<usize>)>> {
        {
            let cache = self.resolve_cache.read();
            if let Some(cached) = cache.get(term) {
                return cached.clone();
            }
        }
        let resolved = self.resolve_with_index_uncached(term).map(Arc::new);
        self.resolve_cache
            .write()
            .insert(term.to_string(), resolved.clone());
        resolved
    }

    fn resolve_with_index(&self, term: &str) -> Option<(AlignmentPair, Option<usize>)> {
        self.resolve_arc(term).map(|resolved| (*resolved).clone())
    }

    fn resolve_with_index_uncached(&self, term: &str) -> Option<(AlignmentPair, Option<usize>)> {
        if let Some(target) = self.synonym_chain(term) {
            let target_norm = self.normalize(&target);
            if target_norm != self.normalize(term) {
                let idx = self.index.get(&target_norm).copied();
                return Some((
                    AlignmentPair {
                        source: term.to_string(),
                        target,
                        kind: AlignmentKind::Synonym,
                        confidence: 0.99,
                        evidence: vec![format!("同义词映射 {} → {}", term, self.synonyms.get(term).map(|s| s.as_str()).unwrap_or(""))],
                    },
                    idx,
                ));
            }
        }

        let normalized = self.normalize(term);
        let self_idx = self.index.get(&normalized).copied();
        if let Some(idx) = self_idx {
            let canonical = self.profiles[idx].term.clone();
            if canonical != term {
                return Some((
                    AlignmentPair {
                        source: term.to_string(),
                        target: canonical.clone(),
                        kind: AlignmentKind::Normalized,
                        confidence: 0.98,
                        evidence: vec![format!("规范化后与「{}」相同", canonical)],
                    },
                    Some(idx),
                ));
            }
        }

        let mut best_similarity: Option<(f64, usize)> = None;
        // 查询侧 bigram 只需计算一次，供全部候选复用
        let query_keys = Self::bigram_keys(&normalized);
        for i in 0..self.profiles.len() {
            if Some(i) == self_idx {
                continue;
            }
            if let Some(self_i) = self_idx {
                if i > self_i {
                    continue;
                }
            }
            let sim =
                self.similarity_normalized_with_keys(&normalized, &query_keys, &self.profile_norms[i]);
            if sim >= self.similarity_threshold
                && best_similarity.map_or(true, |(b, _)| sim > b)
            {
                best_similarity = Some((sim, i));
            }
        }
        if let Some((sim, i)) = best_similarity {
            return Some((
                AlignmentPair {
                    source: term.to_string(),
                    target: self.profiles[i].term.clone(),
                    kind: AlignmentKind::Similarity,
                    confidence: sim,
                    evidence: vec![format!("字面相似度 {:.2}", sim)],
                },
                Some(i),
            ));
        }

        if let Some(profile) = self_idx.map(|i| &self.profiles[i]) {
            let mut best_structural: Option<(f64, usize, Vec<String>)> = None;
            for (i, other) in self.profiles.iter().enumerate() {
                if Some(i) == self_idx || other.kind != profile.kind {
                    continue;
                }
                if let Some(self_i) = self_idx {
                    if i > self_i {
                        continue;
                    }
                }
                // 先只统计交集大小，胜出后再重建证据，避免为每个候选克隆邻居
                let shared_count = profile.neighbors.intersection(&other.neighbors).count();
                if shared_count < self.structural_min_shared {
                    continue;
                }
                let union = profile.neighbors.union(&other.neighbors).count();
                if union == 0 {
                    continue;
                }
                let jaccard = shared_count as f64 / union as f64;
                if jaccard < self.structural_min_jaccard {
                    continue;
                }
                let confidence = (0.5 + 0.5 * jaccard).min(0.95);
                if best_structural
                    .as_ref()
                    .map_or(true, |(b, _, _)| confidence > *b)
                {
                    let mut evidence: Vec<String> = profile
                        .neighbors
                        .intersection(&other.neighbors)
                        .take(3)
                        .map(|n| format!("共同关系「{}」", n))
                        .collect();
                    evidence.push(format!("Jaccard {:.2}", jaccard));
                    best_structural = Some((confidence, i, evidence));
                }
            }
            if let Some((confidence, i, evidence)) = best_structural {
                return Some((
                    AlignmentPair {
                        source: term.to_string(),
                        target: self.profiles[i].term.clone(),
                        kind: AlignmentKind::Structural,
                        confidence,
                        evidence,
                    },
                    Some(i),
                ));
            }
        }

        None
    }

    pub fn resolve(&self, term: &str) -> Option<AlignmentPair> {
        self.resolve_with_index(term).map(|(pair, _)| pair)
    }

    pub fn canonical_form(&self, term: &str) -> String {
        match self.resolve_arc(term) {
            Some(pair) => pair.0.target.clone(),
            None => term.to_string(),
        }
    }

    pub fn align_all(&self) -> Vec<AlignmentPair> {
        let mut pairs = Vec::new();
        for (i, profile) in self.profiles.iter().enumerate() {
            if let Some(resolved) = self.resolve_arc(&profile.term) {
                match resolved.1 {
                    Some(t) if t >= i => continue,
                    _ => pairs.push(resolved.0.clone()),
                }
            }
        }
        pairs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source.cmp(&b.source))
        });
        pairs
    }

    pub fn alignment_triples(&self) -> Vec<(String, String, String)> {
        self.align_all()
            .into_iter()
            .map(|p| (p.source, "sameAs".to_string(), p.target))
            .collect()
    }

    pub fn rewrite_triples(
        &self,
        triples: &[(String, String, String)],
    ) -> (Vec<(String, String, String)>, Vec<AlignmentPair>) {
        let mut rewritten = Vec::new();
        let mut pairs = Vec::new();
        let mut seen_triples: HashSet<(String, String, String)> = HashSet::new();
        let mut seen_pairs: HashSet<(String, String)> = HashSet::new();

        for (s, p, o) in triples {
            // 部分解析结果供对齐证据与规范化三元组共用
            let resolved = [self.resolve(s), self.resolve(p), self.resolve(o)];
            for pair in resolved.iter().flatten() {
                if seen_pairs.insert((pair.source.clone(), pair.target.clone())) {
                    pairs.push(pair.clone());
                }
            }
            let canonical = (
                match &resolved[0] {
                    Some(pair) => pair.target.clone(),
                    None => s.clone(),
                },
                match &resolved[1] {
                    Some(pair) => pair.target.clone(),
                    None => p.clone(),
                },
                match &resolved[2] {
                    Some(pair) => pair.target.clone(),
                    None => o.clone(),
                },
            );
            if seen_triples.insert(canonical.clone()) {
                rewritten.push(canonical);
            }
        }

        pairs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source.cmp(&b.source))
        });
        (rewritten, pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synonyms() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("老婆".to_string(), "妻子".to_string());
        m.insert("老公".to_string(), "丈夫".to_string());
        m.insert("北大".to_string(), "北京大学".to_string());
        m.insert("北京大学".to_string(), "北大".to_string());
        m
    }

    #[test]
    fn test_synonym_resolution() {
        let aligner = OntologyAligner::new(synonyms(), 0.72);
        let pair = aligner.resolve("老婆").expect("老婆应能对齐");
        assert_eq!(pair.target, "妻子");
        assert_eq!(pair.kind, AlignmentKind::Synonym);
        assert!(pair.confidence > 0.9);
        assert_eq!(aligner.canonical_form("老公"), "丈夫");
    }

    #[test]
    fn test_synonym_cycle_terminates() {
        let aligner = OntologyAligner::new(synonyms(), 0.72);
        let form = aligner.canonical_form("北大");
        assert!(form == "北大" || form == "北京大学");
        let form2 = aligner.canonical_form("北京大学");
        assert!(form2 == "北大" || form2 == "北京大学");
    }

    #[test]
    fn test_normalized_alignment() {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.9);
        aligner.register_term("Python 语言", TermKind::Entity);
        let pair = aligner.resolve("python语言").expect("空格与大小写差异应被规范化");
        assert_eq!(pair.target, "Python 语言");
        assert_eq!(pair.kind, AlignmentKind::Normalized);
    }

    #[test]
    fn test_similarity_alignment() {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.72);
        aligner.register_term("北京大学", TermKind::Entity);
        let pair = aligner.resolve("北京大学校").expect("包含关系应触发相似度对齐");
        assert_eq!(pair.target, "北京大学");
        assert_eq!(pair.kind, AlignmentKind::Similarity);
        assert!(pair.confidence >= 0.72);
    }

    #[test]
    fn test_structural_alignment() {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.99);
        aligner.observe_triple("苹果公司", "创始人", "乔布斯");
        aligner.observe_triple("苹果公司", "产品", "iPhone");
        aligner.observe_triple("苹果公司", "总部", "库比蒂诺");
        aligner.observe_triple("apple", "创始人", "乔布斯");
        aligner.observe_triple("apple", "产品", "iPhone");
        aligner.observe_triple("apple", "总部", "加利福尼亚");
        let pair = aligner.resolve("apple").expect("共享邻居应触发结构对齐");
        assert_eq!(pair.target, "苹果公司");
        assert_eq!(pair.kind, AlignmentKind::Structural);
        assert!(pair.evidence.iter().any(|e| e.contains("乔布斯")));
    }

    #[test]
    fn test_observe_triple_invalidates_cached_resolution() {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.99);
        aligner.register_term("甲", TermKind::Entity);
        aligner.register_term("乙", TermKind::Entity);
        aligner.register_term("丙", TermKind::Entity);
        aligner.observe_triple("甲", "关系", "乙");
        aligner.observe_triple("丁", "关系", "乙");
        assert!(aligner.resolve("丁").is_none());
        aligner.observe_triple("甲", "关系", "丙");
        aligner.observe_triple("丁", "关系", "丙");
        let pair = aligner.resolve("丁").expect("邻居变化后缓存应失效并触发结构对齐");
        assert_eq!(pair.target, "甲");
        assert_eq!(pair.kind, AlignmentKind::Structural);
    }

    #[test]
    fn test_align_all_and_triples() {
        let mut aligner = OntologyAligner::new(synonyms(), 0.8);
        aligner.observe_triple("张三", "妻子", "李四");
        aligner.observe_triple("张三的老婆", "妻子", "李四");
        let pairs = aligner.align_all();
        assert!(pairs.iter().any(|p| p.source == "张三的老婆" && p.target == "张三"));
        let triples = aligner.alignment_triples();
        assert!(triples
            .iter()
            .any(|(s, p, o)| s == "张三的老婆" && p == "sameAs" && o == "张三"));
    }

    #[test]
    fn test_rewrite_triples_merges_duplicates() {
        let mut aligner = OntologyAligner::new(synonyms(), 0.72);
        aligner.observe_triple("张三", "妻子", "李四");
        aligner.observe_triple("张三的老婆", "妻子", "李四");
        let triples = vec![
            (
                "张三的老婆".to_string(),
                "妻子".to_string(),
                "李四".to_string(),
            ),
            ("张三".to_string(), "妻子".to_string(), "李四".to_string()),
            ("张三".to_string(), "老婆".to_string(), "李四".to_string()),
        ];
        let (rewritten, pairs) = aligner.rewrite_triples(&triples);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].0, "张三");
        assert_eq!(rewritten[0].1, "妻子");
        assert!(pairs.iter().any(|p| p.source == "老婆" && p.target == "妻子"));
        assert!(pairs.iter().any(|p| p.source == "张三的老婆" && p.target == "张三"));
    }

    #[test]
    fn test_unknown_term_returns_none() {
        let aligner = OntologyAligner::new(synonyms(), 0.95);
        assert!(aligner.resolve("完全不存在的术语").is_none());
        assert_eq!(aligner.canonical_form("完全不存在的术语"), "完全不存在的术语");
    }
}

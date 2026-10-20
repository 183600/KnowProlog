use parking_lot::RwLock;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 候选分数差不超过该值时视为「分数接近」，交给答案探测决定是否澄清。
pub const DEFAULT_AMBIGUITY_MARGIN: f64 = 0.05;

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

impl AlignmentKind {
    /// 对齐通道的确定性优先级：显式同义词 > 规范化 > 字面相似 > 结构相似。
    /// 数值越大越优先；仅用于候选排序，不参与置信度计算。
    pub fn priority(&self) -> u8 {
        match self {
            AlignmentKind::Synonym => 3,
            AlignmentKind::Normalized => 2,
            AlignmentKind::Similarity => 1,
            AlignmentKind::Structural => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignmentPair {
    pub source: String,
    pub target: String,
    pub kind: AlignmentKind,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// 一个术语的对齐决策状态：有唯一最佳、存在值得澄清的候选、或完全未知。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignmentState {
    #[default]
    Unknown,
    Resolved,
    Ambiguous,
}

impl AlignmentState {
    pub fn label_zh(self) -> &'static str {
        match self {
            AlignmentState::Unknown => "未知",
            AlignmentState::Resolved => "已解析",
            AlignmentState::Ambiguous => "有歧义",
        }
    }
}

/// 候选格中的一个对齐候选（保留来源通道、置信度、证据与剖面下标）。
#[derive(Debug, Clone)]
pub struct AlignmentCandidate {
    pub pair: AlignmentPair,
    pub index: Option<usize>,
}

impl AlignmentCandidate {
    pub fn target(&self) -> &str {
        &self.pair.target
    }

    pub fn kind(&self) -> AlignmentKind {
        self.pair.kind
    }

    pub fn confidence(&self) -> f64 {
        self.pair.confidence
    }
}

/// 对齐候选格：`resolve` 的 additive 版本，保留 top-k 而不是只留最佳。
///
/// 候选按「通道优先级 → 置信度降序 → 目标字典序」稳定排序，因此同样的
/// 术语表与查询在任何调用顺序下都得到相同结果。`state` 只表达是否存在
/// 值得进一步处理的歧义；答案级等价性可由 [`OntologyAligner::probe_answers`]
/// 进一步消解（两种解释给出同一答案时不打扰用户）。
#[derive(Debug, Clone)]
pub struct AlignmentResolution {
    pub source: String,
    pub normalized: String,
    pub state: AlignmentState,
    pub ambiguity_margin: f64,
    pub candidates: Vec<AlignmentCandidate>,
}

impl AlignmentResolution {
    pub fn best(&self) -> Option<&AlignmentCandidate> {
        self.candidates.first()
    }

    pub fn targets(&self) -> Vec<&str> {
        self.candidates
            .iter()
            .map(AlignmentCandidate::target)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    fn recompute_state(&mut self) {
        self.state = Self::state_for(&self.candidates, self.ambiguity_margin);
    }

    fn state_for(
        candidates: &[AlignmentCandidate],
        ambiguity_margin: f64,
    ) -> AlignmentState {
        let Some(top) = candidates.first() else {
            return AlignmentState::Unknown;
        };
        // 显式同义词/规范化是确定性映射：即使存在相似候选也不要求澄清。
        if matches!(
            top.pair.kind,
            AlignmentKind::Synonym | AlignmentKind::Normalized
        ) {
            return AlignmentState::Resolved;
        }
        match candidates.get(1) {
            Some(second)
                if top.pair.confidence - second.pair.confidence <= ambiguity_margin =>
            {
                AlignmentState::Ambiguous
            }
            _ => AlignmentState::Resolved,
        }
    }

    /// 过滤候选后重新计算状态（例如只保留本知识库中真实存在的谓词）。
    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&AlignmentCandidate) -> bool,
    {
        self.candidates.retain(predicate);
        self.recompute_state();
    }
}

/// 单个候选在某一实体/问题下实际得到的答案集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAnswers {
    pub target: String,
    pub answers: Vec<String>,
}

/// 带答案探测的对齐决策：先看候选，再用「两种解释是否给出同一答案」消解歧义。
#[derive(Debug, Clone)]
pub struct AlignmentDecision {
    pub resolution: AlignmentResolution,
    pub answers: Vec<CandidateAnswers>,
    /// 与最佳候选分数接近的所有候选是否给出完全相同的答案集合。
    pub equivalent_answers: bool,
}

impl AlignmentDecision {
    pub fn best(&self) -> Option<&AlignmentCandidate> {
        self.resolution.best()
    }

    /// 只有真正会改变答案的歧义才值得打扰用户。
    pub fn should_clarify(&self) -> bool {
        self.resolution.state == AlignmentState::Ambiguous
    }
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
    // 与 profiles 一一对应的规范化结果
    profile_norms: Vec<String>,
    // 与 profiles 一一对应的有序 bigram 键
    profile_keys: Vec<Vec<(char, Option<char>)>>,
    // 与 profiles 一一对应的规范化字符长度
    profile_lens: Vec<usize>,
    index: HashMap<String, usize>,
    similarity_threshold: f64,
    structural_min_shared: usize,
    structural_min_jaccard: f64,
    // 术语解析结果缓存：注册新术语时整体失效
    resolve_cache: RwLock<HashMap<String, Option<Arc<(AlignmentPair, Option<usize>)>>>>,
}

/// 深拷贝：解析缓存一并复制（缓存内容是不可变 `Arc`，克隆只是指针复制）。
/// 供 WarmBase 会话分叉使用：派生状态按值分叉，绝不与源实例共享可变锁。
impl Clone for OntologyAligner {
    fn clone(&self) -> Self {
        Self {
            synonyms: self.synonyms.clone(),
            profiles: self.profiles.clone(),
            profile_norms: self.profile_norms.clone(),
            profile_keys: self.profile_keys.clone(),
            profile_lens: self.profile_lens.clone(),
            index: self.index.clone(),
            similarity_threshold: self.similarity_threshold,
            structural_min_shared: self.structural_min_shared,
            structural_min_jaccard: self.structural_min_jaccard,
            resolve_cache: RwLock::new(self.resolve_cache.read().clone()),
        }
    }
}

impl OntologyAligner {
    pub fn new(synonyms: HashMap<String, String>, similarity_threshold: f64) -> Self {
        Self {
            synonyms,
            profiles: Vec::new(),
            profile_norms: Vec::new(),
            profile_keys: Vec::new(),
            profile_lens: Vec::new(),
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

    /// 运行时登记同义词映射（谓词别名准入后同步到对齐器，供查询归一）。
    pub fn register_synonym(&mut self, alias: &str, canonical: &str) -> bool {
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return false;
        }
        if self.synonyms.get(alias).map(String::as_str) == Some(canonical) {
            return false;
        }
        self.synonyms.insert(alias.to_string(), canonical.to_string());
        self.resolve_cache.write().clear();
        true
    }

    pub fn is_registered(&self, term: &str) -> bool {
        self.index.contains_key(self.normalize_cow(term).as_ref())
    }

    pub fn kind_of(&self, term: &str) -> Option<TermKind> {
        self.index
            .get(self.normalize_cow(term).as_ref())
            .map(|&idx| self.profiles[idx].kind)
    }

    pub fn register_term(&mut self, term: &str, kind: TermKind) -> usize {
        let normalized = self.normalize_cow(term);
        if let Some(&idx) = self.index.get(normalized.as_ref()) {
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
        let mut keys = Self::bigram_keys(normalized.as_ref());
        keys.sort_unstable();
        self.profile_lens.push(normalized.chars().count());
        let normalized = normalized.into_owned();
        self.profile_norms.push(normalized.clone());
        self.profile_keys.push(keys);
        self.index.insert(normalized, idx);
        self.resolve_cache.write().clear();
        idx
    }

    /// 返回第 `idx` 个术语的规范化形式
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

    /// 规范化术语；无需改写时直接借用原字符串。
    pub fn normalize_cow<'a>(&self, term: &'a str) -> Cow<'a, str> {
        let needs_work = term.chars().any(|ch| {
            ch.is_whitespace()
                || ch.is_ascii_uppercase()
                || matches!(
                    ch,
                    '（' | '）'
                        | '：'
                        | '，'
                        | '。'
                        | '、'
                        | '；'
                        | '！'
                        | '？'
                        | '《'
                        | '》'
                        | '【'
                        | '】'
                        | '“'
                        | '”'
                        | '‘'
                        | '’'
                        | '－'
                        | '—'
                        | '–'
                        | '·'
                        | '('
                        | ')'
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
                )
        });
        if !needs_work {
            return Cow::Borrowed(term);
        }
        Cow::Owned(self.normalize(term))
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
        let mut ga = Self::bigram_keys(a);
        let mut gb = Self::bigram_keys(b);
        ga.sort_unstable();
        gb.sort_unstable();
        Self::bigram_dice_sorted(&ga, &gb)
    }

    /// 对两个已排序的 bigram 多重集做归并求交，得到 Dice 系数。
    fn bigram_dice_sorted(ga: &[(char, Option<char>)], gb: &[(char, Option<char>)]) -> f64 {
        if ga.is_empty() || gb.is_empty() {
            return 0.0;
        }
        let mut i = 0usize;
        let mut j = 0usize;
        let mut overlap = 0usize;
        while i < ga.len() && j < gb.len() {
            match ga[i].cmp(&gb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    overlap += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        2.0 * overlap as f64 / (ga.len() + gb.len()) as f64
    }

    /// 使用查询侧与候选侧 bigram 计算相似度。
    fn similarity_normalized_with_keys(
        &self,
        query: &str,
        query_keys: &[(char, Option<char>)],
        other: &str,
        other_keys: &[(char, Option<char>)],
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
        if short == query {
            Self::bigram_dice_sorted(query_keys, other_keys)
        } else {
            Self::bigram_dice_sorted(other_keys, query_keys)
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

    /// `term_norm` 必须是 `self.normalize(term)` 的结果
    fn synonym_chain(&self, term: &str, term_norm: &str) -> Option<String> {
        let mut current = term.to_string();
        let mut current_norm = term_norm.to_string();
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

    fn synonym_candidate(
        &self,
        term: &str,
        normalized: &str,
    ) -> Option<(AlignmentPair, Option<usize>)> {
        let target = self.synonym_chain(term, normalized)?;
        let target_norm = self.normalize(&target);
        if target_norm == normalized {
            return None;
        }
        Some((
            AlignmentPair {
                source: term.to_string(),
                target,
                kind: AlignmentKind::Synonym,
                confidence: 0.99,
                evidence: vec![format!(
                    "同义词映射 {} → {}",
                    term,
                    self.synonyms.get(term).map(|s| s.as_str()).unwrap_or("")
                )],
            },
            self.index.get(&target_norm).copied(),
        ))
    }

    fn normalized_candidate(
        &self,
        term: &str,
        normalized: &str,
    ) -> Option<(AlignmentPair, Option<usize>)> {
        let idx = self.index.get(normalized).copied()?;
        let canonical = self.profiles[idx].term.clone();
        if canonical == term {
            return None;
        }
        Some((
            AlignmentPair {
                source: term.to_string(),
                target: canonical.clone(),
                kind: AlignmentKind::Normalized,
                confidence: 0.98,
                evidence: vec![format!("规范化后与「{}」相同", canonical)],
            },
            Some(idx),
        ))
    }

    /// 全部字面相似候选（按相似度降序、剖面下标升序）。
    /// `resolve` 取第一个；候选格保留全部，供歧义判定。
    fn similarity_candidates(
        &self,
        normalized: &str,
        self_idx: Option<usize>,
    ) -> Vec<(f64, usize)> {
        let mut found: Vec<(f64, usize)> = Vec::new();
        let mut query_keys = Self::bigram_keys(normalized);
        query_keys.sort_unstable();
        let query_len = normalized.chars().count();
        let query_key_len = query_keys.len();
        let scan_end = self_idx.map_or(self.profiles.len(), |i| i + 1);
        for i in 0..scan_end {
            if Some(i) == self_idx {
                continue;
            }
            // 上界剪枝：包含关系与 Dice 的取值上限都不可能达到阈值时直接跳过
            if self.similarity_threshold > 0.0 {
                let other_len = self.profile_lens[i];
                if query_len == 0 || other_len == 0 {
                    continue;
                }
                let (min_len, max_len) = if query_len <= other_len {
                    (query_len, other_len)
                } else {
                    (other_len, query_len)
                };
                let mut upper_bound = if min_len >= 2 {
                    0.7 + 0.3 * (min_len as f64 / max_len as f64)
                } else {
                    0.0
                };
                let other_key_len = self.profile_keys[i].len();
                if other_key_len > 0 && query_key_len > 0 {
                    let dice_bound = 2.0 * (query_key_len.min(other_key_len) as f64)
                        / (query_key_len + other_key_len) as f64;
                    if dice_bound > upper_bound {
                        upper_bound = dice_bound;
                    }
                }
                if upper_bound < self.similarity_threshold {
                    continue;
                }
            }
            let sim = self.similarity_normalized_with_keys(
                normalized,
                &query_keys,
                &self.profile_norms[i],
                &self.profile_keys[i],
            );
            if sim >= self.similarity_threshold {
                found.push((sim, i));
            }
        }
        found.sort_by(|(left_score, left_index), (right_score, right_index)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_index.cmp(right_index))
        });
        found
    }

    /// 全部结构相似候选（按置信度降序、剖面下标升序）。
    fn structural_candidates(
        &self,
        self_idx: Option<usize>,
    ) -> Vec<(f64, usize, Vec<String>)> {
        let Some(idx) = self_idx else {
            return Vec::new();
        };
        let profile = &self.profiles[idx];
        let mut found: Vec<(f64, usize, Vec<String>)> = Vec::new();
        let scan_end = idx + 1;
        let own_degree = profile.neighbors.len();
        for (i, other) in self.profiles.iter().enumerate().take(scan_end) {
            if i == idx || other.kind != profile.kind {
                continue;
            }
            let other_degree = other.neighbors.len();
            // 长度上界剪枝：共享数不可能达到下限或 Jaccard 不可能达标时跳过
            let (smaller, larger) = if own_degree <= other_degree {
                (own_degree, other_degree)
            } else {
                (other_degree, own_degree)
            };
            if smaller < self.structural_min_shared
                || (smaller as f64) < self.structural_min_jaccard * (larger as f64)
            {
                continue;
            }
            let shared_count = profile.neighbors.intersection(&other.neighbors).count();
            if shared_count < self.structural_min_shared {
                continue;
            }
            // |A∪B| = |A| + |B| − |A∩B|
            let union = own_degree + other_degree - shared_count;
            if union == 0 {
                continue;
            }
            let jaccard = shared_count as f64 / union as f64;
            if jaccard < self.structural_min_jaccard {
                continue;
            }
            let confidence = (0.5 + 0.5 * jaccard).min(0.95);
            let mut evidence: Vec<String> = profile
                .neighbors
                .intersection(&other.neighbors)
                .take(3)
                .map(|n| format!("共同关系「{}」", n))
                .collect();
            evidence.push(format!("Jaccard {:.2}", jaccard));
            found.push((confidence, i, evidence));
        }
        found.sort_by(|(left_score, left_index, _), (right_score, right_index, _)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_index.cmp(right_index))
        });
        found
    }

    fn resolve_with_index_uncached(&self, term: &str) -> Option<(AlignmentPair, Option<usize>)> {
        let normalized = self.normalize(term);
        if let Some(candidate) = self.synonym_candidate(term, &normalized) {
            return Some(candidate);
        }
        if let Some(candidate) = self.normalized_candidate(term, &normalized) {
            return Some(candidate);
        }
        let self_idx = self.index.get(&normalized).copied();
        if let Some((sim, i)) = self
            .similarity_candidates(&normalized, self_idx)
            .into_iter()
            .next()
        {
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
        if let Some((confidence, i, evidence)) = self
            .structural_candidates(self_idx)
            .into_iter()
            .next()
        {
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
        None
    }

    pub fn resolve(&self, term: &str) -> Option<AlignmentPair> {
        self.resolve_with_index(term).map(|(pair, _)| pair)
    }

    /// 返回 top-k 对齐候选格（additive：不改变 [`Self::resolve`] 的结果）。
    ///
    /// 候选来源与 `resolve` 完全一致；同一目标只保留最高优先级的证据。
    /// `limit == 0` 时按 1 处理，保证非未知术语至少有一个候选。
    pub fn resolve_candidates(&self, term: &str, limit: usize) -> AlignmentResolution {
        let normalized = self.normalize(term);
        let self_idx = self.index.get(&normalized).copied();
        let mut candidates: Vec<AlignmentCandidate> = Vec::new();
        if let Some((pair, index)) = self.synonym_candidate(term, &normalized) {
            candidates.push(AlignmentCandidate { pair, index });
        }
        if let Some((pair, index)) = self.normalized_candidate(term, &normalized) {
            candidates.push(AlignmentCandidate { pair, index });
        }
        for (sim, i) in self.similarity_candidates(&normalized, self_idx) {
            candidates.push(AlignmentCandidate {
                pair: AlignmentPair {
                    source: term.to_string(),
                    target: self.profiles[i].term.clone(),
                    kind: AlignmentKind::Similarity,
                    confidence: sim,
                    evidence: vec![format!("字面相似度 {:.2}", sim)],
                },
                index: Some(i),
            });
        }
        for (confidence, i, evidence) in self.structural_candidates(self_idx) {
            candidates.push(AlignmentCandidate {
                pair: AlignmentPair {
                    source: term.to_string(),
                    target: self.profiles[i].term.clone(),
                    kind: AlignmentKind::Structural,
                    confidence,
                    evidence,
                },
                index: Some(i),
            });
        }

        // 同一目标只留证据最强的一条；显式通道优先于模糊通道。
        let mut deduped: HashMap<String, AlignmentCandidate> = HashMap::new();
        for candidate in candidates {
            let target = candidate.pair.target.clone();
            match deduped.get(&target) {
                Some(existing)
                    if existing.pair.kind.priority() > candidate.pair.kind.priority()
                        || (existing.pair.kind.priority() == candidate.pair.kind.priority()
                            && existing.pair.confidence >= candidate.pair.confidence) => {}
                _ => {
                    deduped.insert(target, candidate);
                }
            }
        }
        let mut candidates: Vec<AlignmentCandidate> = deduped.into_values().collect();
        candidates.sort_by(|left, right| {
            right
                .pair
                .kind
                .priority()
                .cmp(&left.pair.kind.priority())
                .then_with(|| {
                    right
                        .pair
                        .confidence
                        .partial_cmp(&left.pair.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.pair.target.cmp(&right.pair.target))
        });
        candidates.truncate(limit.max(1));

        let state = AlignmentResolution::state_for(&candidates, DEFAULT_AMBIGUITY_MARGIN);
        AlignmentResolution {
            source: term.to_string(),
            normalized,
            state,
            ambiguity_margin: DEFAULT_AMBIGUITY_MARGIN,
            candidates,
        }
    }

    /// 用答案探测消解候选歧义：与最佳候选分数接近的解释若给出同一答案，
    /// 则静默采用最佳候选；只有答案集合确实不同才保留 `Ambiguous`。
    pub fn probe_answers<F>(
        &self,
        resolution: AlignmentResolution,
        probe: F,
    ) -> AlignmentDecision
    where
        F: Fn(&str) -> Vec<String>,
    {
        let mut resolution = resolution;
        let answers: Vec<CandidateAnswers> = resolution
            .candidates
            .iter()
            .map(|candidate| {
                let mut values = probe(candidate.target());
                values.sort();
                values.dedup();
                CandidateAnswers {
                    target: candidate.target().to_string(),
                    answers: values,
                }
            })
            .collect();

        let top_confidence = resolution
            .candidates
            .first()
            .map(AlignmentCandidate::confidence)
            .unwrap_or(0.0);
        let contenders: Vec<&CandidateAnswers> = answers
            .iter()
            .filter(|answer| {
                resolution
                    .candidates
                    .iter()
                    .find(|candidate| candidate.target() == answer.target)
                    .map(|candidate| {
                        top_confidence - candidate.confidence() <= resolution.ambiguity_margin
                    })
                    .unwrap_or(false)
            })
            .collect();
        let equivalent_answers = contenders.windows(2).all(|pair| pair[0].answers == pair[1].answers);
        let non_empty = contenders
            .iter()
            .filter(|answer| !answer.answers.is_empty())
            .count();

        // 等价的解释不构成歧义；只有一种解释有答案时也优先作答而不是追问。
        if resolution.state == AlignmentState::Ambiguous && (equivalent_answers || non_empty < 2) {
            resolution.state = AlignmentState::Resolved;
        }
        AlignmentDecision {
            resolution,
            answers,
            equivalent_answers,
        }
    }

    /// 对所有术语做候选扫描，返回有歧义的项（确定性顺序，供诊断报告）。
    pub fn ambiguous_terms(&self, limit: usize) -> Vec<AlignmentResolution> {
        let mut found: Vec<AlignmentResolution> = self
            .profiles
            .iter()
            .map(|profile| self.resolve_candidates(&profile.term, 3))
            .filter(|resolution| resolution.state == AlignmentState::Ambiguous)
            .collect();
        found.sort_by(|left, right| left.source.cmp(&right.source));
        found.truncate(limit.max(1));
        found
    }

    pub fn canonical_form(&self, term: &str) -> String {
        match self.resolve_arc(term) {
            Some(pair) => pair.0.target.clone(),
            None => term.to_string(),
        }
    }

    pub fn align_all(&self) -> Vec<AlignmentPair> {
        // 对齐是对称关系（sameAs），同一对术语只保留一次；
        // 不能按「目标索引 >= 源索引」跳过，否则同义词目标注册得比源晚时会被整体丢弃
        let mut pairs = Vec::new();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for profile in &self.profiles {
            if let Some(resolved) = self.resolve_arc(&profile.term) {
                let pair = resolved.0.clone();
                if pair.source == pair.target {
                    continue;
                }
                let (left, right) = if pair.source <= pair.target {
                    (pair.source.clone(), pair.target.clone())
                } else {
                    (pair.target.clone(), pair.source.clone())
                };
                if seen.insert((left, right)) {
                    pairs.push(pair);
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
            let resolved = [self.resolve_arc(s), self.resolve_arc(p), self.resolve_arc(o)];
            for pair in resolved.iter().flatten() {
                if seen_pairs.insert((pair.0.source.clone(), pair.0.target.clone())) {
                    pairs.push(pair.0.clone());
                }
            }
            let canonical = (
                match &resolved[0] {
                    Some(pair) => pair.0.target.clone(),
                    None => s.clone(),
                },
                match &resolved[1] {
                    Some(pair) => pair.0.target.clone(),
                    None => p.clone(),
                },
                match &resolved[2] {
                    Some(pair) => pair.0.target.clone(),
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
    fn test_align_all_keeps_synonym_pair_regardless_of_registration_order() {
        let mut synonyms = HashMap::new();
        synonyms.insert("老婆".to_string(), "妻子".to_string());
        let mut aligner = OntologyAligner::new(synonyms, 0.99);
        // 目标术语比源术语更晚注册
        aligner.register_term("老婆", TermKind::Entity);
        aligner.register_term("妻子", TermKind::Entity);
        assert_eq!(aligner.resolve("老婆").map(|p| p.target), Some("妻子".to_string()));
        let pairs = aligner.align_all();
        assert!(
            pairs
                .iter()
                .any(|p| p.source == "老婆" && p.target == "妻子"),
            "同义词对不应因注册顺序而被丢弃：{:?}",
            pairs
        );
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

    fn ambiguous_aligner() -> OntologyAligner {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.5);
        aligner.register_term("苹果公司", TermKind::Entity);
        aligner.register_term("苹果果实", TermKind::Entity);
        aligner
    }

    #[test]
    fn test_candidate_lattice_keeps_top_k_and_matches_resolve() {
        let aligner = ambiguous_aligner();
        let resolution = aligner.resolve_candidates("苹果", 5);
        assert_eq!(resolution.state, AlignmentState::Ambiguous);
        assert_eq!(resolution.targets(), vec!["苹果公司", "苹果果实"]);
        assert_eq!(
            aligner.resolve("苹果").map(|pair| pair.target),
            resolution.best().map(|candidate| candidate.target().to_string())
        );
        assert!(resolution
            .candidates
            .iter()
            .all(|candidate| candidate.confidence() >= 0.5));
    }

    #[test]
    fn test_candidate_lattice_is_deterministic_and_explicit_channel_wins() {
        let aligner = OntologyAligner::new(synonyms(), 0.72);
        let first = aligner.resolve_candidates("老婆", 5);
        let second = aligner.resolve_candidates("老婆", 5);
        assert_eq!(first.targets(), second.targets());
        let best = first.best().expect("同义词应产生候选");
        assert_eq!(best.target(), "妻子");
        assert_eq!(best.kind(), AlignmentKind::Synonym);
        assert_eq!(first.state, AlignmentState::Resolved);
    }

    #[test]
    fn test_answer_probe_collapses_equivalent_candidates() {
        let aligner = ambiguous_aligner();
        let resolution = aligner.resolve_candidates("苹果", 5);
        let decision = aligner.probe_answers(resolution, |_| vec!["同一答案".to_string()]);
        assert!(decision.equivalent_answers);
        assert_eq!(decision.resolution.state, AlignmentState::Resolved);
        assert!(!decision.should_clarify());
        assert_eq!(decision.answers.len(), decision.resolution.candidates.len());
    }

    #[test]
    fn test_answer_probe_keeps_ambiguous_when_answers_differ() {
        let aligner = ambiguous_aligner();
        let resolution = aligner.resolve_candidates("苹果", 5);
        let decision =
            aligner.probe_answers(resolution, |target| vec![target.to_string()]);
        assert!(!decision.equivalent_answers);
        assert_eq!(decision.resolution.state, AlignmentState::Ambiguous);
        assert!(decision.should_clarify());
        assert!(decision
            .answers
            .iter()
            .all(|answer| !answer.answers.is_empty()));
    }

    #[test]
    fn test_answer_probe_does_not_ask_when_only_one_interpretation_has_answers() {
        let aligner = ambiguous_aligner();
        let resolution = aligner.resolve_candidates("苹果", 5);
        let decision = aligner.probe_answers(resolution, |target| {
            if target == "苹果公司" {
                vec!["库比蒂诺".to_string()]
            } else {
                Vec::new()
            }
        });
        assert!(!decision.equivalent_answers);
        assert_eq!(decision.resolution.state, AlignmentState::Resolved);
        assert!(!decision.should_clarify());
    }

    #[test]
    fn test_ambiguous_terms_report_is_deterministic() {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.99);
        for subject in ["甲", "乙", "丙"] {
            aligner.observe_triple(subject, "关系", "目标一");
            aligner.observe_triple(subject, "关系", "目标二");
        }
        let terms = aligner.ambiguous_terms(10);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].source, "丙");
        assert_eq!(terms[0].state, AlignmentState::Ambiguous);
        assert_eq!(aligner.ambiguous_terms(10)[0].targets(), terms[0].targets());
    }
}

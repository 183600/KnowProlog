// 符号化实体链接器（Entity Linker）
//
// 目标：把问句中的自然语言指称（mention）链接到知识图谱中的规范实体，
// 全程不依赖神经网络：
//   1) 候选生成：规范名/别名精确匹配 + 字符二元组相似度的模糊匹配（容忍错别字）；
//   2) 候选消歧：结合共现邻居、问题谓词/类型线索、实体度数做可解释打分；
//   3) 可解释输出：每个候选附带置信度与打分理由，便于人工检查链接是否可信。
// 这对应 LLM 的"输入理解/实体识别"，但实现是字符串索引 + 计数 + 规则。

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMethod {
    ExactName,
    Alias,
    Fuzzy,
}

impl LinkMethod {
    pub fn label(self) -> &'static str {
        match self {
            LinkMethod::ExactName => "规范名精确匹配",
            LinkMethod::Alias => "别名匹配",
            LinkMethod::Fuzzy => "模糊匹配",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinkCandidate {
    pub entity: String,
    pub score: f64,
    pub method: LinkMethod,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LinkedMention {
    pub mention: String,
    pub start: usize,
    pub candidates: Vec<LinkCandidate>,
}

impl LinkedMention {
    pub fn best(&self) -> Option<&LinkCandidate> {
        self.candidates.first()
    }
}

struct EntityProfile {
    etype: Option<String>,
    degree: usize,
    char_count: usize,
    char_set: HashSet<char>,
    bigrams: HashSet<(char, char)>,
}

pub struct EntityLinker {
    profiles: HashMap<String, EntityProfile>,
    surface_candidates: HashMap<String, Vec<String>>,
    // 表层名及其字符序列，按首字符分桶；桶内保持（长度降序、字典序升序），
    // 注册期维护，精确匹配时无需每次重建索引
    surfaces_by_first: HashMap<char, Vec<(String, Vec<char>)>>,
    neighbors: HashMap<String, HashSet<String>>,
    out_predicates: HashMap<String, HashMap<String, usize>>,
    in_predicates: HashMap<String, HashMap<String, usize>>,
    fuzzy_threshold: f64,
}

const FUZZY_MIN_SCORE: f64 = 0.45;
const FUZZY_MAX_CHARS: usize = 12;
const MAX_CANDIDATES_PER_MENTION: usize = 3;
const SEGMENT_DELIMITERS: [char; 22] = [
    '的', '是', '有', '吗', '呢', '谁', '什', '么', '多', '少', '几', '和', '与', '在', '为', '把',
    '？', '?', '，', ',', '。', '：',
];

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityLinker {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            surface_candidates: HashMap::new(),
            surfaces_by_first: HashMap::new(),
            neighbors: HashMap::new(),
            out_predicates: HashMap::new(),
            in_predicates: HashMap::new(),
            fuzzy_threshold: FUZZY_MIN_SCORE,
        }
    }

    pub fn with_fuzzy_threshold(mut self, threshold: f64) -> Self {
        self.fuzzy_threshold = threshold;
        self
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn is_known_entity(&self, name: &str) -> bool {
        self.profiles.contains_key(name)
    }

    pub fn register_entity(&mut self, name: &str, etype: Option<&str>) {
        if name.trim().is_empty() {
            return;
        }
        let degree = self.profiles.get(name).map(|p| p.degree).unwrap_or(0);
        let chars: Vec<char> = name.chars().collect();
        let char_set: HashSet<char> = chars.iter().copied().collect();
        let bigrams: HashSet<(char, char)> = chars.windows(2).map(|w| (w[0], w[1])).collect();
        self.profiles.insert(
            name.to_string(),
            EntityProfile {
                etype: etype.filter(|t| !t.trim().is_empty()).map(str::to_string),
                degree,
                char_count: chars.len(),
                char_set,
                bigrams,
            },
        );
        self.add_surface(name, name);
    }

    pub fn register_alias(&mut self, alias: &str, canonical: &str) {
        if alias.trim().is_empty() || canonical.trim().is_empty() || alias == canonical {
            return;
        }
        let is_new = match self.surface_candidates.entry(alias.to_string()) {
            Entry::Occupied(mut entry) => {
                if !entry.get().iter().any(|c| c == canonical) {
                    entry.get_mut().push(canonical.to_string());
                }
                false
            }
            Entry::Vacant(entry) => {
                entry.insert(vec![canonical.to_string()]);
                true
            }
        };
        if is_new {
            self.insert_surface_sorted(alias);
        }
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        for (name, other, predicate) in [(subject, object, predicate), (object, subject, predicate)] {
            if !self.profiles.contains_key(name) {
                self.register_entity(name, None);
            }
            let profile = self.profiles.get_mut(name).expect("实体刚注册");
            profile.degree += 1;
            match self.neighbors.get_mut(name) {
                Some(neighbors) => {
                    // 先借用检查，避免对已存在的邻居重复分配 String
                    if !neighbors.contains(other) {
                        neighbors.insert(other.to_string());
                    }
                }
                None => {
                    self.neighbors
                        .entry(name.to_string())
                        .or_default()
                        .insert(other.to_string());
                }
            }
            let predicates = if name == subject {
                &mut self.out_predicates
            } else {
                &mut self.in_predicates
            };
            if !predicates.contains_key(name) {
                predicates.insert(name.to_string(), HashMap::new());
            }
            let bucket = predicates.get_mut(name).expect("刚插入");
            match bucket.get_mut(predicate) {
                Some(count) => *count += 1,
                None => {
                    bucket.insert(predicate.to_string(), 1);
                }
            }
        }
    }

    fn add_surface(&mut self, surface: &str, canonical: &str) {
        let is_new = match self.surface_candidates.entry(surface.to_string()) {
            Entry::Occupied(mut entry) => {
                if !entry.get().iter().any(|c| c == canonical) {
                    entry.get_mut().push(canonical.to_string());
                }
                false
            }
            Entry::Vacant(entry) => {
                entry.insert(vec![canonical.to_string()]);
                true
            }
        };
        if is_new {
            self.insert_surface_sorted(surface);
        }
    }

    // 表层名按首字符分桶，桶内按字符数降序、字典序升序维护，新项二分插入保持有序。
    fn insert_surface_sorted(&mut self, surface: &str) {
        let chars: Vec<char> = surface.chars().collect();
        let Some(&first) = chars.first() else {
            return;
        };
        let len = chars.len();
        let bucket = self.surfaces_by_first.entry(first).or_default();
        let pos = bucket
            .binary_search_by(|probe| {
                probe
                    .1
                    .len()
                    .cmp(&len)
                    .reverse()
                    .then_with(|| probe.0.as_str().cmp(surface))
            })
            .unwrap_or_else(|pos| pos);
        bucket.insert(pos, (surface.to_string(), chars));
    }

    pub fn link(&self, text: &str) -> Vec<LinkedMention> {
        self.link_with_context(text, &[])
    }

    pub fn link_with_context(&self, text: &str, hints: &[String]) -> Vec<LinkedMention> {
        if self.profiles.is_empty() || text.trim().is_empty() {
            return Vec::new();
        }
        let mut mentions = self.exact_mentions(text);
        if mentions.is_empty() {
            mentions = self.fuzzy_mentions(text);
        }
        self.disambiguate(&mut mentions, hints);
        mentions
    }

    fn exact_mentions(&self, text: &str) -> Vec<LinkedMention> {
        let chars: Vec<char> = text.chars().collect();
        let mut mentions: Vec<LinkedMention> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched: Option<(&str, usize)> = None;
            if let Some(candidates) = self.surfaces_by_first.get(&chars[i]) {
                for (surface, surface_chars) in candidates {
                    let len = surface_chars.len();
                    if i + len > chars.len() {
                        continue;
                    }
                    if chars[i..i + len] == surface_chars[..] {
                        matched = Some((surface.as_str(), len));
                        break;
                    }
                }
            }
            if let Some((surface, len)) = matched {
                let candidates = self.candidates_for_surface(surface);
                if !candidates.is_empty() {
                    mentions.push(LinkedMention {
                        mention: surface.to_string(),
                        start: i,
                        candidates,
                    });
                    i += len;
                    continue;
                }
            }
            i += 1;
        }
        mentions
    }

    fn candidates_for_surface(&self, surface: &str) -> Vec<LinkCandidate> {
        let Some(canonicals) = self.surface_candidates.get(surface) else {
            return Vec::new();
        };
        let mut out: Vec<LinkCandidate> = Vec::new();
        for canonical in canonicals {
            let Some(profile) = self.profiles.get(canonical) else {
                continue;
            };
            let method = if canonical == surface {
                LinkMethod::ExactName
            } else {
                LinkMethod::Alias
            };
            let mut score = if method == LinkMethod::ExactName { 1.0 } else { 0.85 };
            let degree_bonus = ((1.0 + profile.degree as f64).ln() / 10.0).min(0.1);
            score += degree_bonus;
            let mut reasons = vec![format!("{}「{}」", method.label(), surface)];
            if degree_bonus > 0.0 {
                reasons.push(format!("图谱度数 {}（常用实体优先）", profile.degree));
            }
            out.push(LinkCandidate {
                entity: canonical.clone(),
                score: score.min(0.99),
                method,
                reasons,
            });
        }
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.entity.cmp(&b.entity))
        });
        out.truncate(MAX_CANDIDATES_PER_MENTION);
        out
    }

    fn fuzzy_mentions(&self, text: &str) -> Vec<LinkedMention> {
        let mut out: Vec<LinkedMention> = Vec::new();
        for (start, segment) in segment_spans(text) {
            let trimmed = segment.trim();
            let char_count = trimmed.chars().count();
            if char_count < 2 || char_count > FUZZY_MAX_CHARS || is_stop_segment(trimmed) {
                continue;
            }
            let trimmed_chars: Vec<char> = trimmed.chars().collect();
            let trimmed_char_set: HashSet<char> = trimmed_chars.iter().copied().collect();
            let trimmed_bigrams: HashSet<(char, char)> = trimmed_chars
                .windows(2)
                .map(|w| (w[0], w[1]))
                .collect();
            let mut scored: Vec<(f64, f64, &String)> = self
                .profiles
                .iter()
                .filter_map(|(canonical, profile)| {
                    // 上界低于阈值时直接跳过昂贵集合运算（须用真实集合大小，
                    // 重复字符会令集合远小于字符数，按字符数估界会误剪）
                    let upper = similarity_upper_bound(
                        trimmed_chars.len(),
                        profile.char_count,
                        trimmed_char_set.len(),
                        profile.char_set.len(),
                        trimmed_bigrams.len(),
                        profile.bigrams.len(),
                        trimmed == canonical.as_str(),
                    );
                    if upper + 0.1 < self.fuzzy_threshold {
                        return None;
                    }
                    let similarity = dice_similarity_precomputed(
                        trimmed,
                        &trimmed_chars,
                        &trimmed_char_set,
                        &trimmed_bigrams,
                        canonical,
                        profile,
                    );
                    let degree_bonus = ((1.0 + profile.degree as f64).ln() / 10.0).min(0.1);
                    Some((similarity + degree_bonus, similarity, canonical))
                })
                .filter(|(combined, _, _)| *combined >= self.fuzzy_threshold)
                .collect();
            if scored.is_empty() {
                continue;
            }
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.2.cmp(b.2))
            });
            let candidates: Vec<LinkCandidate> = scored
                .iter()
                .take(MAX_CANDIDATES_PER_MENTION)
                .map(|(combined, similarity, canonical)| LinkCandidate {
                    entity: (*canonical).clone(),
                    score: combined.min(0.9),
                    method: LinkMethod::Fuzzy,
                    reasons: vec![format!(
                        "字符相似度 {:.2}（「{}」≈「{}」，容错匹配）",
                        similarity, trimmed, canonical
                    )],
                })
                .collect();
            out.push(LinkedMention {
                mention: trimmed.to_string(),
                start,
                candidates,
            });
        }
        out
    }

    fn disambiguate(&self, mentions: &mut [LinkedMention], hints: &[String]) {
        let top_entities: Vec<String> = mentions
            .iter()
            .filter_map(|m| m.best().map(|c| c.entity.clone()))
            .collect();
        for mention in mentions.iter_mut() {
            for candidate in mention.candidates.iter_mut() {
                let mut boost = 0.0;
                let mut reasons: Vec<String> = Vec::new();
                if let Some(neighbors) = self.neighbors.get(&candidate.entity) {
                    for other in &top_entities {
                        if other != &candidate.entity && neighbors.contains(other) {
                            boost += 0.08;
                            reasons.push(format!("与共现实体「{}」存在知识关联", other));
                        }
                    }
                }
                for hint in hints {
                    let predicate_hit = self
                        .out_predicates
                        .get(&candidate.entity)
                        .map_or(false, |p| p.contains_key(hint))
                        || self
                            .in_predicates
                            .get(&candidate.entity)
                            .map_or(false, |p| p.contains_key(hint));
                    let type_hit = self
                        .profiles
                        .get(&candidate.entity)
                        .and_then(|p| p.etype.as_deref())
                        == Some(hint.as_str());
                    if predicate_hit || type_hit {
                        boost += 0.1;
                        reasons.push(format!("与问题线索「{}」相容", hint));
                    }
                }
                if boost > 0.0 {
                    candidate.score = (candidate.score + boost).min(0.99);
                    candidate.reasons.extend(reasons);
                }
            }
            mention.candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.entity.cmp(&b.entity))
            });
            mention.candidates.truncate(MAX_CANDIDATES_PER_MENTION);
        }
    }

    pub fn report(&self, text: &str, hints: &[String]) -> String {
        let mentions = self.link_with_context(text, hints);
        self.report_for_mentions(text, &mentions)
    }

    pub fn report_for_mentions(&self, text: &str, mentions: &[LinkedMention]) -> String {
        if mentions.is_empty() {
            return format!("实体链接：「{}」中未识别到知识图谱中的实体。", text);
        }
        let mut output = format!(
            "实体链接报告：「{}」识别到 {} 个指称（知识图谱已登记 {} 个实体）。\n",
            text,
            mentions.len(),
            self.profiles.len()
        );
        for mention in mentions {
            let method = mention
                .best()
                .map(|c| c.method.label())
                .unwrap_or("无候选");
            output.push_str(&format!(
                "  • 「{}」（第 {} 字起；{}）\n",
                mention.mention,
                mention.start + 1,
                method
            ));
            for (index, candidate) in mention.candidates.iter().enumerate() {
                output.push_str(&format!(
                    "      {}. {}（置信度 {:.2}；{}）\n",
                    index + 1,
                    candidate.entity,
                    candidate.score,
                    candidate.reasons.join("；")
                ));
            }
        }
        output.trim_end().to_string()
    }
}

fn segment_spans(text: &str) -> Vec<(usize, String)> {
    let mut segments: Vec<(usize, String)> = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;
    for (index, ch) in text.chars().enumerate() {
        if SEGMENT_DELIMITERS.contains(&ch) {
            if !current.trim().is_empty() {
                segments.push((current_start, std::mem::take(&mut current)));
            } else {
                current.clear();
            }
            current_start = index + 1;
        } else {
            if current.is_empty() {
                current_start = index;
            }
            current.push(ch);
        }
    }
    if !current.trim().is_empty() {
        segments.push((current_start, current));
    }
    segments
}

fn is_stop_segment(segment: &str) -> bool {
    const STOP: [&str; 14] = [
        "什么", "多少", "哪里", "哪个", "哪些", "怎么", "如何", "为什么", "介绍", "一下",
        "告诉我", "请问", "你好", "一下吗",
    ];
    STOP.contains(&segment)
}

#[cfg(test)]
fn bigrams(text: &str) -> HashSet<(char, char)> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return HashSet::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(test)]
fn dice_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();
    // 单字指称不做模糊匹配，避免"张"链接到"张三"之类的过度链接
    if chars_a.len() < 2 || chars_b.len() < 2 {
        return 0.0;
    }
    let bigram_score = {
        let bigram_a = bigrams(a);
        let bigram_b = bigrams(b);
        let intersection = bigram_a.intersection(&bigram_b).count() as f64;
        2.0 * intersection / (bigram_a.len() + bigram_b.len()) as f64
    };
    let unigram_score = {
        let set_a: HashSet<char> = chars_a.iter().copied().collect();
        let set_b: HashSet<char> = chars_b.iter().copied().collect();
        let intersection = set_a.intersection(&set_b).count() as f64;
        // 单字重叠较宽松，打折后仅作为二元组相似度的补充（处理双字名错别字）
        2.0 * intersection / (set_a.len() + set_b.len()) as f64 * 0.8
    };
    bigram_score.max(unigram_score)
}

/// 给定两侧字符数与字符/二元组集合大小，相似度（二元组/单字 Dice 取大者）的
/// 理论上界：交集不超过较小集合，故 `2*min(|A|,|B|)/(|A|+|B|)` 是可达最大值；
/// 文本完全相同时按 1.0 计（`dice_similarity_precomputed` 的相等捷径）。
fn similarity_upper_bound(
    len_a: usize,
    len_b: usize,
    char_set_a: usize,
    char_set_b: usize,
    bigrams_a: usize,
    bigrams_b: usize,
    equal_text: bool,
) -> f64 {
    if equal_text {
        return 1.0;
    }
    if len_a < 2 || len_b < 2 {
        return 0.0;
    }
    let mut upper: f64 = 0.0;
    if bigrams_a + bigrams_b > 0 {
        upper = upper.max(
            2.0 * bigrams_a.min(bigrams_b) as f64 / (bigrams_a + bigrams_b) as f64,
        );
    }
    if char_set_a + char_set_b > 0 {
        upper = upper.max(
            2.0 * char_set_a.min(char_set_b) as f64 / (char_set_a + char_set_b) as f64 * 0.8,
        );
    }
    upper
}

// 与 dice_similarity 完全同值，但复用已预计算的指称字符集合与实体画像特征集合
fn dice_similarity_precomputed(
    trimmed: &str,
    trimmed_chars: &[char],
    trimmed_char_set: &HashSet<char>,
    trimmed_bigrams: &HashSet<(char, char)>,
    canonical: &str,
    profile: &EntityProfile,
) -> f64 {
    if trimmed == canonical {
        return 1.0;
    }
    if trimmed_chars.len() < 2 || profile.char_count < 2 {
        return 0.0;
    }
    let bigram_score = {
        let intersection = trimmed_bigrams.intersection(&profile.bigrams).count() as f64;
        2.0 * intersection / (trimmed_bigrams.len() + profile.bigrams.len()) as f64
    };
    let unigram_score = {
        let intersection = trimmed_char_set.intersection(&profile.char_set).count() as f64;
        2.0 * intersection / (trimmed_char_set.len() + profile.char_set.len()) as f64 * 0.8
    };
    bigram_score.max(unigram_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_linker() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.register_entity("刘德华", Some("人物"));
        linker.register_entity("刘德凯", Some("人物"));
        linker.register_entity("珠穆朗玛峰", Some("山峰"));
        linker.register_entity("苹果公司", Some("公司"));
        linker.register_entity("苹果果实", Some("水果"));
        linker.observe_triple("刘德华", "出生地", "香港");
        linker.observe_triple("刘德华", "职业", "演员");
        linker.observe_triple("苹果公司", "创始人", "乔布斯");
        linker.observe_triple("苹果果实", "颜色", "红色");
        linker.register_alias("苹果", "苹果公司");
        linker.register_alias("苹果", "苹果果实");
        linker
    }

    #[test]
    fn test_exact_and_alias_linking() {
        let linker = sample_linker();
        let mentions = linker.link("刘德华的出生地在哪里");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "刘德华");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::ExactName);

        let alias_mentions = linker.link("苹果创始人的创始人是谁");
        assert!(!alias_mentions.is_empty());
        let entities: Vec<String> = alias_mentions
            .iter()
            .filter_map(|m| m.best().map(|c| c.entity.clone()))
            .collect();
        assert!(entities.contains(&"苹果公司".to_string()));
    }

    #[test]
    fn test_fuzzy_typo_linking() {
        let linker = sample_linker();
        let mentions = linker.link("刘德桦的出生地是哪里");
        assert_eq!(mentions.len(), 1, "错别字应被模糊匹配容忍");
        let best = mentions[0].best().unwrap();
        assert_eq!(best.entity, "刘德华");
        assert_eq!(best.method, LinkMethod::Fuzzy);
        assert!(best.reasons[0].contains("相似度"));

        let long_typo = linker.link("珠穆朗玛锋有多高");
        assert_eq!(long_typo[0].best().unwrap().entity, "珠穆朗玛峰");
    }

    #[test]
    fn test_context_disambiguation_prefers_hint_compatible_candidate() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let mentions = linker.link_with_context("苹果的创始人是谁", &hints);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "苹果公司");

        let hints2 = vec!["颜色".to_string()];
        let mentions2 = linker.link_with_context("苹果的颜色是什么", &hints2);
        assert_eq!(mentions2[0].best().unwrap().entity, "苹果果实");
    }

    #[test]
    fn test_fuzzy_matching_survives_repeated_characters() {
        // 重复字符令 bigram/字符集合远小于字符数：真实相似度仍可达到阈值，
        // 剪枝的上界必须按集合大小估计，否则会漏掉这些匹配
        let mut linker = EntityLinker::new();
        linker.register_entity("哈哈哈哈哈哈哈哈", Some("人物"));
        let mentions = linker.link("哈哈的来历");
        assert_eq!(mentions.len(), 1, "重复字符文本不应被上界剪枝漏掉");
        assert_eq!(mentions[0].best().unwrap().entity, "哈哈哈哈哈哈哈哈");
    }

    #[test]
    fn test_no_false_link_for_unknown_text() {
        let linker = sample_linker();
        let mentions = linker.link("今天天气怎么样");
        assert!(mentions.is_empty(), "无关文本不应产生链接：{:?}", mentions);
    }

    #[test]
    fn test_report_explains_candidates() {
        let linker = sample_linker();
        let report = linker.report("刘德桦的出生地是哪里", &[]);
        assert!(report.contains("实体链接报告"));
        assert!(report.contains("刘德华"));
        assert!(report.contains("置信度"));
    }

    #[test]
    fn test_dice_similarity_bounds() {
        assert_eq!(dice_similarity("张三", "张三"), 1.0);
        assert!(dice_similarity("刘德桦", "刘德华") > 0.4);
        assert_eq!(dice_similarity("张三", "苹果"), 0.0);
        assert_eq!(dice_similarity("张", "张三"), 0.0);
    }
}

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
use std::fmt::Write as _;

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

/// 类型核对结论：疑问词的「类型期待」× 候选实体的「类型证据」。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeVerdict {
    /// 有期待且候选类型命中
    Match,
    /// 有期待但候选类型不符
    Mismatch,
    /// 有期待但候选类型未知（证据不足，不阻断）
    Unknown,
    /// 无疑问词期待
    NotExpected,
}

impl TypeVerdict {
    pub fn label(self) -> &'static str {
        match self {
            TypeVerdict::Match => "命中期待",
            TypeVerdict::Mismatch => "与期待不符",
            TypeVerdict::Unknown => "类型未知",
            TypeVerdict::NotExpected => "无类型期待",
        }
    }
}

/// 单个候选的证据收据（只读审计，不参与链接打分）。
#[derive(Debug, Clone)]
pub struct CandidateReceipt {
    pub entity: String,
    pub score: f64,
    pub method: LinkMethod,
    /// 候选实体的图谱登记类型（类型证据）
    pub entity_type: Option<String>,
    /// 当前问题类型期待与该候选的核对结论（诊断用）
    pub expectation_check: TypeVerdict,
    /// 该候选是否恰为当前对话焦点（先验标注，不抬分）
    pub focus_prior: bool,
    pub reasons: Vec<String>,
}

/// 接地条目：回答引用的最小单位（词面 ↔ 实体 ↔ 类型 ↔ 置信度）。
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedMention {
    pub mention: String,
    pub entity: String,
    pub etype: Option<String>,
    pub score: f64,
}

/// 一个指称的完整证据收据：边界证据 + 链接候选 + 类型证据 + 焦点先验。
#[derive(Debug, Clone)]
pub struct MentionReceipt {
    pub mention: String,
    pub start: usize,
    pub candidates: Vec<CandidateReceipt>,
}

/// 类型核对：实体类型未知→Unknown；关键词命中（忽略大小写）→Match；否则 Mismatch。
fn expectation_check(entity_type: Option<&str>, expectations: &[&str]) -> TypeVerdict {
    if expectations.is_empty() {
        return TypeVerdict::NotExpected;
    }
    let Some(etype) = entity_type else {
        return TypeVerdict::Unknown;
    };
    let etype_lower = etype.to_lowercase();
    if expectations.iter().any(|keyword| {
        etype.contains(keyword) || etype_lower.contains(&keyword.to_lowercase())
    }) {
        TypeVerdict::Match
    } else {
        TypeVerdict::Mismatch
    }
}

/// OOV（未登录）疑点的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OovKind {
    /// 书名号 / 引号包裹的词
    Quoted,
    /// 命中实体后缀表的词
    Suffix,
}

/// OOV 疑点：只报告、不落库；采纳须走人工确认（编年史）流程。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OovSuspect {
    pub text: String,
    pub start: usize,
    pub kind: OovKind,
    pub reason: String,
    /// 后缀表给出的暂定类型（如「未名科技公司」→ 组织）；只标注，不进 KB
    pub suggested_type: Option<String>,
}

/// 实体后缀表（机构/地点/人物角色）：用于把未登录词提名为疑点。
pub const ENTITY_SUFFIXES: [&str; 20] = [
    "公司", "集团", "大学", "学院", "医院", "银行", "博物馆", "图书馆", "研究院", "委员会",
    "政府", "法院", "城市", "自治区", "老师", "教授", "博士", "医生", "律师", "导演",
];

fn is_covered(covered: &[(usize, usize)], start: usize, end: usize) -> bool {
    covered.iter().any(|&(s, e)| start < e && s < end)
}

fn is_han_char(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

/// OOV 后缀词前缀扫描的边界虚词：向前取名字时遇到这些字即截断。
const OOV_BOUNDARY_CHARS: [char; 18] = [
    '的', '是', '和', '与', '或', '及', '在', '为', '把', '被', '由', '将', '有', '了', '地', '得',
    '对', '从',
];

fn is_oov_boundary(ch: char) -> bool {
    OOV_BOUNDARY_CHARS.contains(&ch)
}

/// 扫描文本中的 OOV 疑点：引号词 + 后缀词，排除已链接区间。
///
/// 只提名不落库；输出按（起始位置，来源）确定排序、按文本去重。
pub fn oov_suspects(text: &str, receipts: &[MentionReceipt]) -> Vec<OovSuspect> {
    let chars: Vec<char> = text.chars().collect();
    let covered: Vec<(usize, usize)> = receipts
        .iter()
        .map(|receipt| {
            (
                receipt.start,
                receipt.start + receipt.mention.chars().count(),
            )
        })
        .collect();
    let mut suspects: Vec<OovSuspect> = Vec::new();

    // 1) 引号词：书名号 / 各类引号
    let mut i = 0;
    while i < chars.len() {
        let close = match chars[i] {
            '《' => Some('》'),
            '“' => Some('”'),
            '「' => Some('」'),
            '『' => Some('』'),
            _ => None,
        };
        let Some(close) = close else {
            i += 1;
            continue;
        };
        if let Some(offset) = chars[i + 1..].iter().position(|ch| *ch == close) {
            let end = i + 1 + offset;
            let inner: String = chars[i + 1..end].iter().collect();
            let trimmed = inner.trim();
            let count = trimmed.chars().count();
            if count > 0 && count <= 12 && !is_covered(&covered, i + 1, end) {
                suspects.push(OovSuspect {
                    text: trimmed.to_string(),
                    start: i + 1,
                    kind: OovKind::Quoted,
                    reason: "未在知识图谱登记".to_string(),
                    suggested_type: crate::identity::infer_suffix_type(trimmed)
                        .map(|(_, etype)| etype.to_string()),
                });
            }
            i = end + 1;
            continue;
        }
        i += 1;
    }

    // 2) 实体后缀词：向前取最多 4 个连续汉字作前缀（功能词处截断）
    let suffix_chars: Vec<Vec<char>> = ENTITY_SUFFIXES
        .iter()
        .map(|suffix| suffix.chars().collect())
        .collect();
    for (suffix_index, _) in ENTITY_SUFFIXES.iter().enumerate() {
        let suffix = &suffix_chars[suffix_index];
        let len = suffix.len();
        if len == 0 || chars.len() < len {
            continue;
        }
        for start in 0..=chars.len() - len {
            if chars[start..start + len] != suffix[..] {
                continue;
            }
            let mut prefix_start = start;
            let mut taken = 0;
            while prefix_start > 0 && taken < 4 {
                let prev = chars[prefix_start - 1];
                if !is_han_char(prev) || is_oov_boundary(prev) {
                    break;
                }
                prefix_start -= 1;
                taken += 1;
            }
            if taken == 0 {
                continue; // 裸后缀（公司/大学…）不提名
            }
            let end = start + len;
            if is_covered(&covered, prefix_start, end) {
                continue;
            }
            let candidate: String = chars[prefix_start..end].iter().collect();
            let suggested_type = crate::identity::infer_suffix_type(&candidate)
                .map(|(_, etype)| etype.to_string());
            suspects.push(OovSuspect {
                text: candidate,
                start: prefix_start,
                kind: OovKind::Suffix,
                reason: format!("实体后缀「{}」，未在知识图谱登记", ENTITY_SUFFIXES[suffix_index]),
                suggested_type,
            });
        }
    }

    suspects.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.text.cmp(&b.text)));
    let mut seen = HashSet::new();
    suspects.retain(|suspect| seen.insert(suspect.text.clone()));
    suspects
}

/// 渲染提及账本：每个指称一张证据收据，输出确定性可 golden。
pub fn render_mention_ledger(
    text: &str,
    receipts: &[MentionReceipt],
    expectations: &[&str],
    suspects: &[OovSuspect],
) -> String {
    let expectation_text = if expectations.is_empty() {
        "无".to_string()
    } else {
        expectations.join("/")
    };
    let mut output = String::new();
    if receipts.is_empty() {
        output.push_str(&format!(
            "提及账本：「{}」中未识别到知识图谱中的实体（类型期待：{}）。\n",
            text, expectation_text
        ));
    } else {
        output.push_str(&format!(
            "提及账本：「{}」识别到 {} 个指称（类型期待：{}）。\n",
            text,
            receipts.len(),
            expectation_text
        ));
        for receipt in receipts {
            let _ = writeln!(
                output,
                "  • 「{}」（第 {} 字起）",
                receipt.mention,
                receipt.start + 1
            );
            for (index, candidate) in receipt.candidates.iter().enumerate() {
                let etype = candidate.entity_type.as_deref().unwrap_or("未登记类型");
                let _ = writeln!(
                    output,
                    "      {}. {}（置信度 {:.2}；{}；类型证据：{} → {}）",
                    index + 1,
                    candidate.entity,
                    candidate.score,
                    candidate.method.label(),
                    etype,
                    candidate.expectation_check.label()
                );
                for reason in &candidate.reasons {
                    let _ = writeln!(output, "         - {}", reason);
                }
                if candidate.focus_prior {
                    let _ = writeln!(output, "         - 焦点先验：当前对话焦点");
                }
            }
        }
    }
    if !suspects.is_empty() {
        output.push_str(&format!("OOV 疑点（只报告，不落库）：{} 个\n", suspects.len()));
        for suspect in suspects {
            let kind = match suspect.kind {
                OovKind::Quoted => "引号词",
                OovKind::Suffix => "后缀词",
            };
            let mut line = format!(
                "  • 「{}」（第 {} 字起；{}；{}",
                suspect.text,
                suspect.start + 1,
                kind,
                suspect.reason
            );
            if let Some(etype) = &suspect.suggested_type {
                line.push_str(&format!("；暂定类型：{}（准入后生效）", etype));
            }
            line.push('）');
            let _ = writeln!(output, "{}", line);
        }
    }
    output.trim_end().to_string()
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
    // 注册期维护
    surfaces_by_first: HashMap<char, Vec<(String, Vec<char>)>>,
    neighbors: HashMap<String, HashSet<String>>,
    out_predicates: HashMap<String, HashMap<String, usize>>,
    in_predicates: HashMap<String, HashMap<String, usize>>,
    // 已观测三元组（去重）：整图快照会被反复观察，度数/谓词计数不得因重放膨胀
    seen_triples: HashSet<(String, String, String)>,
    fuzzy_threshold: f64,
}

const FUZZY_MIN_SCORE: f64 = 0.45;
const FUZZY_MAX_CHARS: usize = 12;
const MAX_CANDIDATES_PER_MENTION: usize = 3;

#[inline]
fn is_segment_delimiter(ch: char) -> bool {
    matches!(
        ch,
        '的' | '是'
            | '有'
            | '吗'
            | '呢'
            | '谁'
            | '什'
            | '么'
            | '多'
            | '少'
            | '几'
            | '和'
            | '与'
            | '在'
            | '为'
            | '把'
            | '？'
            | '?'
            | '，'
            | ','
            | '。'
            | '：'
    )
}

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
            seen_triples: HashSet::new(),
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
        let new_etype = etype.filter(|t| !t.trim().is_empty());
        // 重复观察同一实体且类型未变：画像（字符集/bigram/表面映射）一致
        if let Some(existing) = self.profiles.get(name) {
            if existing.etype.as_deref() == new_etype {
                return;
            }
        }
        let degree = self.profiles.get(name).map(|p| p.degree).unwrap_or(0);
        let chars: Vec<char> = name.chars().collect();
        let char_set: HashSet<char> = chars.iter().copied().collect();
        let bigrams: HashSet<(char, char)> = chars.windows(2).map(|w| (w[0], w[1])).collect();
        self.profiles.insert(
            name.to_string(),
            EntityProfile {
                etype: new_etype.map(str::to_string),
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
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        // 空白端点无法登记画像（register_entity 会跳过），整条边忽略避免 panic
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        // 重复观察（快照重放）是无操作：度数、谓词计数与邻居都不变
        if !self
            .seen_triples
            .insert((subject.to_string(), predicate.to_string(), object.to_string()))
        {
            return;
        }
        for (name, other, predicate) in [(subject, object, predicate), (object, subject, predicate)] {
            if !self.profiles.contains_key(name) {
                self.register_entity(name, None);
            }
            let profile = self.profiles.get_mut(name).expect("实体刚注册");
            profile.degree += 1;
            match self.neighbors.get_mut(name) {
                Some(neighbors) => {
                    // 先借用检查
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

    /// 已观测的唯一三元组数（去重后）。
    pub fn triple_count(&self) -> usize {
        self.seen_triples.len()
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
        // 模糊匹配补充精确匹配未覆盖的片段：同一句中既有已知实体又有错别字时，
        // 不能因为有精确命中就整体跳过模糊匹配
        let exact_spans: Vec<(usize, usize)> = mentions
            .iter()
            .map(|mention| (mention.start, mention.start + mention.mention.chars().count()))
            .collect();
        for fuzzy in self.fuzzy_mentions(text) {
            let fuzzy_end = fuzzy.start + fuzzy.mention.chars().count();
            let overlaps_exact = exact_spans
                .iter()
                .any(|&(start, end)| fuzzy.start < end && start < fuzzy_end);
            if !overlaps_exact {
                mentions.push(fuzzy);
            }
        }
        mentions.sort_by_key(|mention| mention.start);
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
                    // 上界低于阈值时直接跳过集合运算（须用真实集合大小，
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

    /// 问题线索加成：候选实体的出/入谓词或登记类型命中线索时 +0.1（可解释理由）。
    fn hint_boost(&self, entity: &str, hints: &[String]) -> (f64, Vec<String>) {
        let mut boost = 0.0;
        let mut reasons: Vec<String> = Vec::new();
        for hint in hints {
            let predicate_hit = self
                .out_predicates
                .get(entity)
                .map_or(false, |p| p.contains_key(hint))
                || self
                    .in_predicates
                    .get(entity)
                    .map_or(false, |p| p.contains_key(hint));
            let type_hit = self
                .profiles
                .get(entity)
                .and_then(|p| p.etype.as_deref())
                == Some(hint.as_str());
            if predicate_hit || type_hit {
                boost += 0.1;
                reasons.push(format!("与问题线索「{}」相容", hint));
            }
        }
        (boost, reasons)
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
                let (hint_gain, hint_reasons) = self.hint_boost(&candidate.entity, hints);
                boost += hint_gain;
                reasons.extend(hint_reasons);
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

    /// 显式准入一个 OOV 词面（提名的唯一出口）：按后缀表推断暂定类型并登记实体。
    ///
    /// 幂等：已登记实体直接返回既有类型，不覆盖。返回（词面，类型）；
    /// 空词面返回 None；无后缀线索时类型为通用的 `Entity`（准入动作本身是显式的）。
    pub fn admit_entity(&mut self, surface: &str) -> Option<(String, String)> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        if self.is_known_entity(surface) {
            let etype = self.entity_type(surface).unwrap_or("Entity").to_string();
            return Some((surface.to_string(), etype));
        }
        let etype = crate::identity::infer_suffix_type(surface)
            .map(|(_, etype)| etype.to_string())
            .unwrap_or_else(|| "Entity".to_string());
        self.register_entity(surface, Some(&etype));
        Some((surface.to_string(), etype))
    }

    /// 提及格全局解码：把所有候选跨度（精确/别名/模糊/嵌套）放进格，
    /// 用 Viterbi 式动态规划选一条「不重叠、总分最大」的路径（对应 LLM 的分词/束搜索）。
    ///
    /// 打分口径（全部符号化、可复现）：
    /// - 节点权重 = 候选置信度 × 覆盖字数（置信度 × 证据长度）；
    /// - 相邻选中实体在知识图谱中有关联时 +0.08 共现加成；
    /// - 问题线索加成沿用 `hint_boost`（+0.1/条）；
    /// - 平局裁决：总分高 → 节点少（长词优先）→ 迭代序（确定性）。
    /// OOV 提名不进入格：没有候选就没有节点，保持「只报告、不落库」。
    pub fn link_lattice(&self, text: &str, hints: &[String]) -> Vec<LinkedMention> {
        if self.profiles.is_empty() || text.trim().is_empty() {
            return Vec::new();
        }
        let chars: Vec<char> = text.chars().collect();

        #[derive(Debug, Clone)]
        struct LatticeNode {
            start: usize,
            end: usize,
            surface: String,
            candidates: Vec<LinkCandidate>,
        }

        let mut nodes: Vec<LatticeNode> = Vec::new();
        // 1) 精确/别名：位置 i 上收集全部命中词面（不提前截断，保留嵌套候选）
        for i in 0..chars.len() {
            let Some(bucket) = self.surfaces_by_first.get(&chars[i]) else {
                continue;
            };
            for (surface, surface_chars) in bucket {
                let len = surface_chars.len();
                if i + len > chars.len() || chars[i..i + len] != surface_chars[..] {
                    continue;
                }
                let candidates = self.candidates_for_surface(surface);
                if candidates.is_empty() {
                    continue;
                }
                nodes.push(LatticeNode {
                    start: i,
                    end: i + len,
                    surface: surface.clone(),
                    candidates,
                });
            }
        }
        // 2) 模糊：按分隔段的容错候选（与精确匹配共用同一候选生成器）
        for fuzzy in self.fuzzy_mentions(text) {
            let end = fuzzy.start + fuzzy.mention.chars().count();
            nodes.push(LatticeNode {
                start: fuzzy.start,
                end,
                surface: fuzzy.mention.clone(),
                candidates: fuzzy.candidates,
            });
        }
        if nodes.is_empty() {
            return Vec::new();
        }
        // 线索加成先落到候选分上（与 disambiguate 的线索部分同口径）
        for node in nodes.iter_mut() {
            for candidate in node.candidates.iter_mut() {
                let (boost, reasons) = self.hint_boost(&candidate.entity, hints);
                if boost > 0.0 {
                    candidate.score = (candidate.score + boost).min(0.99);
                    candidate.reasons.extend(reasons);
                }
            }
            node.candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.entity.cmp(&b.entity))
            });
        }

        // 起点 → 节点下标，保持构造序（确定性）
        let mut nodes_by_start: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (index, node) in nodes.iter().enumerate() {
            nodes_by_start.entry(node.start).or_default().push(index);
        }

        #[derive(Debug, Clone)]
        struct DpCell {
            score: f64,
            nodes: u32,
            from_pos: usize,
            from_state: String,
            chosen: Option<(usize, usize)>,
        }

        fn better(a: &DpCell, b: &DpCell) -> bool {
            if (a.score - b.score).abs() > 1e-9 {
                return a.score > b.score;
            }
            a.nodes < b.nodes
        }

        fn offer(
            layer: &mut std::collections::BTreeMap<String, DpCell>,
            state: String,
            cell: DpCell,
        ) {
            match layer.get(&state) {
                Some(existing) if !better(&cell, existing) => {}
                _ => {
                    layer.insert(state, cell);
                }
            }
        }

        // dp[pos][上次选中的实体] = 覆盖前 pos 个字的最优路径；空串键表示尚未选中实体
        let mut dp: Vec<std::collections::BTreeMap<String, DpCell>> =
            vec![std::collections::BTreeMap::new(); chars.len() + 1];
        dp[0].insert(
            String::new(),
            DpCell {
                score: 0.0,
                nodes: 0,
                from_pos: 0,
                from_state: String::new(),
                chosen: None,
            },
        );
        for pos in 0..=chars.len() {
            let states: Vec<(String, DpCell)> = dp[pos]
                .iter()
                .map(|(state, cell)| (state.clone(), cell.clone()))
                .collect();
            for (state, cell) in states {
                // 跳过当前字符（不产生指称）
                if pos < chars.len() {
                    offer(
                        &mut dp[pos + 1],
                        state.clone(),
                        DpCell {
                            score: cell.score,
                            nodes: cell.nodes,
                            from_pos: pos,
                            from_state: state.clone(),
                            chosen: None,
                        },
                    );
                }
                let Some(indices) = nodes_by_start.get(&pos) else {
                    continue;
                };
                let prev_entity = if state.is_empty() {
                    None
                } else {
                    Some(state.as_str())
                };
                for &node_index in indices {
                    let node = &nodes[node_index];
                    let mut best: Option<(f64, usize)> = None;
                    for (candidate_index, candidate) in node.candidates.iter().enumerate() {
                        let mut weight = candidate.score * (node.end - node.start) as f64;
                        if let Some(prev) = prev_entity {
                            if self
                                .neighbors
                                .get(prev)
                                .map_or(false, |neighbors| neighbors.contains(&candidate.entity))
                            {
                                weight += 0.08;
                            }
                        }
                        if best.map_or(true, |(best_weight, _)| weight > best_weight) {
                            best = Some((weight, candidate_index));
                        }
                    }
                    let Some((weight, candidate_index)) = best else {
                        continue;
                    };
                    let entity = node.candidates[candidate_index].entity.clone();
                    offer(
                        &mut dp[node.end],
                        entity,
                        DpCell {
                            score: cell.score + weight,
                            nodes: cell.nodes + 1,
                            from_pos: pos,
                            from_state: state.clone(),
                            chosen: Some((node_index, candidate_index)),
                        },
                    );
                }
            }
        }

        // 终点：全文本覆盖下总分最优（平局取节点少者）
        let final_pos = chars.len();
        let mut best_state: Option<String> = None;
        for (state, cell) in dp[final_pos].iter() {
            let replace = match &best_state {
                None => true,
                Some(current) => dp[final_pos]
                    .get(current)
                    .map_or(true, |current_cell| better(cell, current_cell)),
            };
            if replace {
                best_state = Some(state.clone());
            }
        }
        let Some(mut state) = best_state else {
            return Vec::new();
        };
        let mut selected: Vec<(usize, usize, Option<String>)> = Vec::new();
        let mut pos = final_pos;
        while pos > 0 {
            let Some(cell) = dp[pos].get(&state).cloned() else {
                break;
            };
            if let Some((node_index, candidate_index)) = cell.chosen {
                let prev_entity = if cell.from_state.is_empty() {
                    None
                } else {
                    Some(cell.from_state.clone())
                };
                selected.push((node_index, candidate_index, prev_entity));
            }
            pos = cell.from_pos;
            state = cell.from_state;
        }
        selected.reverse();

        let mut mentions: Vec<LinkedMention> = Vec::with_capacity(selected.len());
        for (node_index, candidate_index, prev_entity) in selected {
            let node = &nodes[node_index];
            let mut candidates = node.candidates.clone();
            let chosen = &mut candidates[candidate_index];
            if let Some(prev) = &prev_entity {
                if self
                    .neighbors
                    .get(prev)
                    .map_or(false, |neighbors| neighbors.contains(&chosen.entity))
                {
                    chosen.score = (chosen.score + 0.08).min(0.99);
                    chosen
                        .reasons
                        .push(format!("全局选路：与相邻指称「{}」共现（+0.08）", prev));
                }
            }
            chosen
                .reasons
                .push("全局选路：置信度×覆盖字数在该路径上最优".to_string());
            let chosen_candidate = candidates.remove(candidate_index);
            candidates.insert(0, chosen_candidate);
            mentions.push(LinkedMention {
                mention: node.surface.clone(),
                start: node.start,
                candidates,
            });
        }
        mentions.sort_by_key(|mention| mention.start);
        mentions
    }

    /// 格解码的可读报告：选中的指称与理由（只读展示，不改链接行为）。
    pub fn lattice_report(&self, text: &str, hints: &[String]) -> String {
        let mentions = self.link_lattice(text, hints);
        if mentions.is_empty() {
            return format!("格解码：「{}」未选出任何不重叠指称。", text);
        }
        let covered: usize = mentions
            .iter()
            .map(|mention| mention.mention.chars().count())
            .sum();
        let mut output = format!(
            "格解码（提及格全局选路）：「{}」选出 {} 个不重叠指称，覆盖 {} 字。\n",
            text,
            mentions.len(),
            covered
        );
        for mention in &mentions {
            let _ = writeln!(
                output,
                "  • 「{}」（第 {} 字起）",
                mention.mention,
                mention.start + 1
            );
            if let Some(candidate) = mention.best() {
                let _ = writeln!(
                    output,
                    "      → {}（置信度 {:.2}；{}）",
                    candidate.entity,
                    candidate.score,
                    candidate.method.label()
                );
                for reason in &candidate.reasons {
                    let _ = writeln!(output, "         - {}", reason);
                }
            }
        }
        output.trim_end().to_string()
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
            let _ = write!(
                output,
                "  • 「{}」（第 {} 字起；{}）\n",
                mention.mention,
                mention.start + 1,
                method
            );
            for (index, candidate) in mention.candidates.iter().enumerate() {
                let _ = write!(
                    output,
                    "      {}. {}（置信度 {:.2}；{}）\n",
                    index + 1,
                    candidate.entity,
                    candidate.score,
                    candidate.reasons.join("；")
                );
            }
        }
        output.trim_end().to_string()
    }

    /// 查询实体的登记类型（提及账本的「类型证据」来源）。
    pub fn entity_type(&self, entity: &str) -> Option<&str> {
        self.profiles
            .get(entity)
            .and_then(|profile| profile.etype.as_deref())
    }

    /// 从当前词表导出身份注册表（只读审计视图；歧义别名原样保留，不静默取一）。
    pub fn identity_registry(&self) -> crate::identity::IdentityRegistry {
        let mut registry = crate::identity::IdentityRegistry::new();
        for name in self.profiles.keys() {
            registry.register(name);
        }
        for (surface, canonicals) in &self.surface_candidates {
            for canonical in canonicals {
                if surface != canonical && self.profiles.contains_key(canonical) {
                    let _ = registry.register_alias(surface, canonical);
                }
            }
        }
        registry
    }

    /// 接地清单：每个指称的最佳候选（词面 ↔ 实体 ↔ 类型 ↔ 置信度）。
    ///
    /// 只读视图，与链接路径同源；置信度阈值由调用方决定（回答附引用用）。
    pub fn grounded_mentions(&self, text: &str, hints: &[String]) -> Vec<GroundedMention> {
        self.link_with_context(text, hints)
            .into_iter()
            .filter_map(|mention| {
                mention.best().map(|candidate| GroundedMention {
                    mention: mention.mention.clone(),
                    entity: candidate.entity.clone(),
                    etype: self.entity_type(&candidate.entity).map(str::to_string),
                    score: candidate.score,
                })
            })
            .collect()
    }

    /// 为文本中的每个指称生成证据收据（只读审计，不改变链接行为）。
    ///
    /// - 边界证据：链接方法 + 位置 + 打分理由（`reasons`）；
    /// - 类型证据：候选实体的图谱登记类型 → 与问题期待核对的结论；
    /// - 对话先验：候选恰为当前焦点实体时标注（不抬分，焦点不得压倒证据）。
    pub fn mention_receipts(
        &self,
        text: &str,
        hints: &[String],
        expectations: &[&str],
        focus: Option<&str>,
    ) -> Vec<MentionReceipt> {
        self.link_with_context(text, hints)
            .into_iter()
            .map(|mention| MentionReceipt {
                mention: mention.mention.clone(),
                start: mention.start,
                candidates: mention
                    .candidates
                    .iter()
                    .map(|candidate| {
                        let entity_type = self.entity_type(&candidate.entity).map(str::to_string);
                        CandidateReceipt {
                            entity: candidate.entity.clone(),
                            score: candidate.score,
                            method: candidate.method,
                            expectation_check: expectation_check(
                                entity_type.as_deref(),
                                expectations,
                            ),
                            entity_type,
                            focus_prior: focus == Some(candidate.entity.as_str()),
                            reasons: candidate.reasons.clone(),
                        }
                    })
                    .collect(),
            })
            .collect()
    }
}

fn segment_spans(text: &str) -> Vec<(usize, String)> {
    let mut segments: Vec<(usize, String)> = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;
    for (index, ch) in text.chars().enumerate() {
        if is_segment_delimiter(ch) {
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
    crate::function_words::projection_set(crate::function_words::Projection::LinkerSegment)
        .contains(segment)
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

// 使用已预计算的指称字符集合与实体画像特征集合
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
    fn test_observe_triple_ignores_blank_endpoints() {
        let mut linker = EntityLinker::new();
        linker.observe_triple("", "师傅", "李四");
        linker.observe_triple("张三", "师傅", " ");
        assert_eq!(linker.len(), 0, "空白端点不应登记任何画像");
        linker.observe_triple("张三", "师傅", "李四");
        assert!(linker.is_known_entity("张三"));
        assert!(linker.is_known_entity("李四"));
    }

    #[test]
    fn test_repeat_observation_does_not_inflate_degree() {
        let mut linker = EntityLinker::new();
        linker.observe_triple("张三", "师傅", "李四");
        let degree = linker.profiles.get("张三").unwrap().degree;
        let predicate_count = linker
            .out_predicates
            .get("张三")
            .and_then(|bucket| bucket.get("师傅"))
            .copied()
            .unwrap();
        // 整图快照重放：度数与谓词计数必须保持不变
        linker.observe_triple("张三", "师傅", "李四");
        assert_eq!(linker.profiles.get("张三").unwrap().degree, degree);
        assert_eq!(
            linker
                .out_predicates
                .get("张三")
                .and_then(|bucket| bucket.get("师傅"))
                .copied(),
            Some(predicate_count)
        );
        assert_eq!(linker.triple_count(), 1);
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

    /// 提及账本：候选带类型证据、期待核对与焦点先验；不改变链接排序。
    #[test]
    fn test_mention_receipts_type_evidence_and_focus() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let receipts = linker.mention_receipts(
            "苹果的创始人是谁",
            &hints,
            &["人", "person"],
            Some("苹果公司"),
        );
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].mention, "苹果");
        let best = &receipts[0].candidates[0];
        assert_eq!(best.entity, "苹果公司");
        assert_eq!(best.method, LinkMethod::Alias);
        assert_eq!(best.entity_type.as_deref(), Some("公司"));
        // 「公司」不含「人/person」→ 与疑问词期待不符；类型证据仍如实展示
        assert_eq!(best.expectation_check, TypeVerdict::Mismatch);
        assert!(best.focus_prior, "候选中含当前焦点实体应标注先验");
        assert!(
            receipts[0]
                .candidates
                .iter()
                .any(|candidate| candidate.entity == "苹果果实" && !candidate.focus_prior)
        );
    }

    /// 类型未知→Unknown；无期待→NotExpected；命中→Match（忽略大小写）。
    #[test]
    fn test_expectation_check_verdicts() {
        assert_eq!(expectation_check(Some("人物"), &["人", "person"]), TypeVerdict::Match);
        assert_eq!(expectation_check(Some("Person"), &["person"]), TypeVerdict::Match);
        assert_eq!(expectation_check(Some("公司"), &["人"]), TypeVerdict::Mismatch);
        assert_eq!(expectation_check(None, &["人"]), TypeVerdict::Unknown);
        assert_eq!(expectation_check(Some("公司"), &[]), TypeVerdict::NotExpected);
    }

    /// 账本渲染确定性：含类型证据、焦点先验与候选理由；两次渲染同字节。
    #[test]
    fn test_render_mention_ledger_is_deterministic() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let expectations = ["人", "person"];
        let receipts =
            linker.mention_receipts("苹果的创始人是谁", &hints, &expectations, Some("苹果公司"));
        let first = render_mention_ledger("苹果的创始人是谁", &receipts, &expectations, &[]);
        let second = render_mention_ledger("苹果的创始人是谁", &receipts, &expectations, &[]);
        assert_eq!(first, second);
        assert!(first.contains("提及账本"));
        assert!(first.contains("类型证据：公司 → 与期待不符"));
        assert!(first.contains("焦点先验：当前对话焦点"));

        let empty = render_mention_ledger("今天天气怎么样", &[], &[], &[]);
        assert!(empty.contains("未识别到知识图谱中的实体"));
    }

    /// OOV 疑点：引号词与后缀词被提名，已链接区间不重复提名，KB 不变。
    #[test]
    fn test_oov_suspects_quoted_and_suffix() {
        let linker = sample_linker();
        let before = linker.len();
        let text = "想了解未名科技公司和《背影》的作者";
        let receipts = linker.mention_receipts(text, &[], &[], None);
        let suspects = oov_suspects(text, &receipts);
        let names: Vec<&str> = suspects.iter().map(|s| s.text.as_str()).collect();
        assert!(names.contains(&"未名科技公司"), "后缀词应被提名：{:?}", names);
        assert!(names.contains(&"背影"), "书名号词应被提名：{:?}", names);
        assert!(
            suspects
                .iter()
                .any(|s| s.text == "未名科技公司" && s.kind == OovKind::Suffix),
        );
        assert!(suspects
            .iter()
            .any(|s| s.text == "背影" && s.kind == OovKind::Quoted));
        // 已链接实体（刘德华）不进入 OOV 疑点
        let linked_text = "刘德华的公司";
        let linked_receipts = linker.mention_receipts(linked_text, &[], &[], None);
        let linked_suspects = oov_suspects(linked_text, &linked_receipts);
        assert!(
            linked_suspects.iter().all(|s| s.text != "刘德华的" && s.text != "刘德华的公司"),
            "已链接区间不应提名：{:?}",
            linked_suspects
        );
        // KB 不变：账本只读
        assert_eq!(linker.len(), before);
    }

    /// 渲染含 OOV 段落时仍确定；无疑点时不出现该段落。
    #[test]
    fn test_render_mention_ledger_oov_section() {
        let linker = sample_linker();
        let text = "未名研究院的院长是谁";
        let receipts = linker.mention_receipts(text, &[], &["人"], None);
        let suspects = oov_suspects(text, &receipts);
        let rendered = render_mention_ledger(text, &receipts, &["人"], &suspects);
        assert!(rendered.contains("OOV 疑点（只报告，不落库）"));
        assert!(rendered.contains("未名研究院"));
        assert!(rendered.contains("暂定类型：组织"), "后缀表应给出暂定类型：{}", rendered);
        assert_eq!(render_mention_ledger(text, &receipts, &["人"], &suspects), rendered);

        let no_oov = render_mention_ledger("苹果的创始人是谁", &[], &[], &[]);
        assert!(!no_oov.contains("OOV 疑点"));
    }

    /// 格解码：嵌套指称等分时由「长词优先」（节点少者）裁决，选不重叠的最大证据路径。
    #[test]
    fn test_lattice_decoding_prefers_longest_exact_path_on_tie() {
        let mut linker = EntityLinker::new();
        linker.register_entity("北京大学", Some("组织"));
        linker.register_entity("北京", Some("地点"));
        linker.register_entity("大学", Some("组织"));
        let mentions = linker.link_lattice("北京大学的校长是谁", &[]);
        assert_eq!(mentions.len(), 1, "等分时长词应压过拆分路径：{:?}", mentions);
        assert_eq!(mentions[0].mention, "北京大学");
        assert_eq!(mentions[0].best().unwrap().entity, "北京大学");
        assert!(mentions[0]
            .best()
            .unwrap()
            .reasons
            .iter()
            .any(|reason| reason.contains("全局选路")));
    }

    #[test]
    fn test_lattice_decoding_splits_when_long_form_unknown() {
        let mut linker = EntityLinker::new();
        linker.register_entity("北京", Some("地点"));
        linker.register_entity("大学", Some("组织"));
        let mentions = linker.link_lattice("北京大学的校长是谁", &[]);
        let surfaces: Vec<&str> = mentions.iter().map(|m| m.mention.as_str()).collect();
        assert_eq!(surfaces, vec!["北京", "大学"]);
        assert_eq!(
            linker.lattice_report("北京大学的校长是谁", &[]),
            linker.lattice_report("北京大学的校长是谁", &[]),
            "格解码报告必须确定可复现"
        );
        assert!(linker
            .lattice_report("今天天气怎么样", &[])
            .contains("未选出"));
    }

    #[test]
    fn test_lattice_decoding_includes_fuzzy_overlaps() {
        let linker = sample_linker();
        let mentions = linker.link_lattice("刘德桦的出生地是哪里", &[]);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "刘德华");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::Fuzzy);
    }

    /// 身份注册表快照：别名与歧义原样保留（不静默取一），审计行确定。
    #[test]
    fn test_identity_registry_snapshot_keeps_aliases_and_ambiguity() {
        let linker = sample_linker();
        let registry = linker.identity_registry();
        assert_eq!(registry.len(), linker.len());
        assert_eq!(
            registry.resolve("苹果"),
            crate::identity::Resolution::Ambiguous(vec![
                "苹果公司".to_string(),
                "苹果果实".to_string(),
            ])
        );
        assert_eq!(
            registry.resolve("苹果公司"),
            crate::identity::Resolution::Canonical("苹果公司".to_string())
        );
        let report = registry.audit_lines().join("\n");
        assert!(report.contains("身份注册表"));
        assert!(report.contains("苹果公司"));
        assert!(report.contains("苹果果实"));
    }

    /// OOV 提名带暂定类型；准入是显式动作且幂等，未准入前 KB 不变。
    #[test]
    fn test_oov_suggested_type_and_explicit_admission() {
        let mut linker = sample_linker();
        let before = linker.len();
        let text = "未名科技公司发布了新书";
        let receipts = linker.mention_receipts(text, &[], &[], None);
        let suspects = oov_suspects(text, &receipts);
        let company = suspects
            .iter()
            .find(|suspect| suspect.text == "未名科技公司")
            .expect("后缀词应被提名");
        assert_eq!(company.suggested_type.as_deref(), Some("组织"));
        assert!(!linker.is_known_entity("未名科技公司"));
        assert_eq!(linker.len(), before, "提名不落库");

        let (name, etype) = linker.admit_entity("未名科技公司").expect("准入应成功");
        assert_eq!(name, "未名科技公司");
        assert_eq!(etype, "组织");
        assert!(linker.is_known_entity("未名科技公司"));
        assert_eq!(linker.len(), before + 1);
        assert_eq!(
            linker.admit_entity("未名科技公司").unwrap().1,
            "组织",
            "重复准入幂等"
        );
        assert!(linker.admit_entity("   ").is_none());
        assert_eq!(
            linker.admit_entity("张三丰").unwrap().1,
            "Entity",
            "无后缀线索时显式准入为通用类型"
        );
    }
}

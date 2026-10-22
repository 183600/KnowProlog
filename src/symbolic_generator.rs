// 符号化生成器（Symbolic Generator）——"语言模型即知识库"
//
// 本模块不训练任何权重，而是把知识图谱本身当作语言模型：
//   • 生成：把与实体相关的三元组按"信息量（谓词稀有度）"排序，由模板
//     组合成连贯的介绍段落（类似 LLM 的资料生成，但每句话都可回溯到事实）；
//   • 续写：对前缀「X的」用计数统计预测下一个谓词，并支持
//     实体级 → 类型级 → 全局 的统计回退（对应 n-gram backoff，对象是事实），
//     预测句子会给出支持度与回退层级，完全可解释。
//
// 全程零神经网络、零梯度：只有计数、排序与模板组合。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

// 段落生成时的排序策略：信息量（谓词稀有度）或主题（谓词字典序）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationOrder {
    Relevance,
    Alphabetical,
}

#[derive(Debug, Clone)]
pub struct ProfilePage {
    pub entity: String,
    pub text: String,
    pub total_lines: usize,
    pub shown: usize,
    pub next_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionLevel {
    Entity,
    Type,
    Global,
}

impl CompletionLevel {
    pub fn label(self) -> &'static str {
        match self {
            CompletionLevel::Entity => "实体级统计",
            CompletionLevel::Type => "类型级回退",
            CompletionLevel::Global => "全局回退",
        }
    }
}

#[derive(Debug, Clone)]
struct CompletionPrediction {
    predicate: String,
    level: CompletionLevel,
    score: f64,
    support: String,
    sentence: String,
}

#[derive(Debug, Clone)]
struct FactLine {
    rank: f64,
    sentence: String,
}

/// 一句话的稳定身份：排序、插入顺序或同谓词对象增减都不会改变它。
///
/// 关系句按谓词成句，所以对象变化只改内容指纹、不换 key；属性句按
/// `(key, value)` 成句，每个取值都有自己的身份；反向句按谓词成句。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClauseKey {
    pub entity: String,
    pub kind: ClauseKind,
    pub predicate: String,
    /// 属性句的取值；关系/反向句为空。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub qualifier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ClauseKind {
    Relation,
    Attribute,
    Reverse,
}

impl ClauseKey {
    fn encode(&self) -> String {
        let kind = match self.kind {
            ClauseKind::Relation => "r",
            ClauseKind::Attribute => "a",
            ClauseKind::Reverse => "v",
        };
        format!(
            "{}#{}#{}#{}#{}#{}#",
            self.entity.len(),
            self.predicate.len(),
            self.qualifier.len(),
            kind,
            self.entity,
            self.predicate
        ) + &self.qualifier
    }
}

/// 一条当前可生成的命题（句子 + 稳定键 + 内容指纹）。
#[derive(Debug, Clone)]
pub struct ProfileClause {
    pub key: ClauseKey,
    pub sentence: String,
    /// 是否含会话层事实（自检时可把这些句子视为已证事实）
    pub from_overlay: bool,
    rank_milli: i64,
}

impl ProfileClause {
    pub fn fingerprint(&self) -> &str {
        &self.sentence
    }
}

/// 生成时的分层事实视图：图谱值 ∪ 会话值，先压制、再替换。
///
/// `replace` 用于功能槽（会话已有唯一赢家）；`add` 是普通会话事实；
/// `suppress` 是被更正/否定掉的取值；`hide` 让整个谓词不再出现在介绍里。
#[derive(Debug, Clone, Default)]
pub struct GenerationOverlay {
    pub add: BTreeMap<(String, String), BTreeSet<String>>,
    pub replace: BTreeMap<(String, String), Vec<String>>,
    pub suppress: BTreeMap<(String, String), BTreeSet<String>>,
    pub hide: BTreeSet<(String, String)>,
}

impl GenerationOverlay {
    pub fn is_empty(&self) -> bool {
        self.add.is_empty()
            && self.replace.is_empty()
            && self.suppress.is_empty()
            && self.hide.is_empty()
    }

    fn touches(&self, subject: &str, predicate: &str) -> bool {
        let key = (subject.to_string(), predicate.to_string());
        self.add.contains_key(&key) || self.replace.contains_key(&key)
    }
}

/// 已经对用户说出口的一句话：只记身份、内容指纹与原句，不记整份生成计划。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpokenRecord {
    pub key: ClauseKey,
    pub fingerprint: String,
    pub sentence: String,
    pub seq: u64,
}

/// 话语账本：按稳定键记住「说过什么、说时是什么内容」。
///
/// 它随对话快照持久化，恢复后不依赖当时的排序结果；每轮从当前知识
/// 重建命题并与账本做差分，得到「新讲 / 更新 / 撤回」三类补丁。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationCursor {
    pub entity: String,
    pub order: GenerationOrder,
    #[serde(default)]
    pub spoken: BTreeMap<String, SpokenRecord>,
    #[serde(default)]
    pub seq: u64,
    #[serde(default)]
    pub turns: u64,
    #[serde(default)]
    pub last_index: usize,
}

#[derive(Debug, Clone)]
pub struct CursorLine {
    pub text: String,
    pub correction: bool,
    pub trusted: bool,
}

#[derive(Debug, Clone)]
pub struct CursorAdvance {
    pub lines: Vec<CursorLine>,
    pub total_lines: usize,
    pub fresh: usize,
    pub corrected: usize,
    pub removed: usize,
    pub span: Option<(usize, usize)>,
    pub continuation: bool,
    pub next_offset: usize,
}

impl GenerationCursor {
    pub fn new(entity: &str, order: GenerationOrder) -> Self {
        Self {
            entity: entity.to_string(),
            order,
            spoken: BTreeMap::new(),
            seq: 0,
            turns: 0,
            last_index: 0,
        }
    }

    /// 显式「换个角度说」：同一实体重新叙述，账本从零开始。
    pub fn reset_for_order(&mut self, order: GenerationOrder) {
        self.order = order;
        self.spoken.clear();
        self.seq = 0;
        self.turns = 0;
        self.last_index = 0;
    }

    pub fn spoken_count(&self) -> usize {
        self.spoken.len()
    }

    /// 旧快照只有 `shown` 位置、没有账本时，把当时已讲过的前 N 句补记为已说，
    /// 避免升级后的第一次「接着说」重复用户已经听过的内容。
    pub fn seed_spoken(&mut self, clauses: &[ProfileClause], shown: usize) {
        for clause in clauses.iter().take(shown) {
            self.seq += 1;
            self.spoken.insert(
                clause.key.encode(),
                SpokenRecord {
                    key: clause.key.clone(),
                    fingerprint: clause.fingerprint().to_string(),
                    sentence: clause.sentence.clone(),
                    seq: self.seq,
                },
            );
        }
        self.last_index = shown.min(clauses.len());
    }

    /// 从当前命题重建视图并与账本差分，产出本轮补丁式叙述。
    ///
    /// 已说过的 key 不会作为新句重复出现；内容指纹变化会生成更正；
    /// 消失的 key 会生成撤回声明。更正有上限，未播出的更正留在账本差异里，
    /// 下一轮继续播（不会静默丢失）。
    pub fn advance(&mut self, clauses: &[ProfileClause], limit: usize) -> CursorAdvance {
        let limit = limit.max(1);
        let continuation = !(self.spoken.is_empty() && self.turns == 0);
        let current: BTreeMap<String, &ProfileClause> = clauses
            .iter()
            .map(|clause| (clause.key.encode(), clause))
            .collect();
        let mut order_index: BTreeMap<String, usize> = BTreeMap::new();
        for (index, clause) in clauses.iter().enumerate() {
            order_index.insert(clause.key.encode(), index);
        }

        let mut updates: Vec<(String, &ProfileClause)> = self
            .spoken
            .iter()
            .filter_map(|(encoded, record)| {
                let clause = current.get(encoded)?;
                (record.fingerprint != clause.fingerprint())
                    .then(|| (encoded.clone(), *clause))
            })
            .collect();
        updates.sort_by_key(|(encoded, _)| order_index.get(encoded).copied().unwrap_or(usize::MAX));

        let mut removals: Vec<(String, SpokenRecord)> = self
            .spoken
            .iter()
            .filter(|(encoded, _)| !current.contains_key(*encoded))
            .map(|(encoded, record)| (encoded.clone(), record.clone()))
            .collect();
        removals.sort_by_key(|(_, record)| record.seq);

        let fresh_count = clauses
            .iter()
            .filter(|clause| !self.spoken.contains_key(&clause.key.encode()))
            .count();
        // 有补讲任务时给新句子留一半版面，避免更正把新知识永远挤在后面。
        let correction_budget = if fresh_count > 0 {
            limit.div_ceil(2).max(1)
        } else {
            limit
        };

        let mut lines: Vec<CursorLine> = Vec::new();
        let mut corrected = 0usize;
        let mut removed = 0usize;

        for (encoded, clause) in updates {
            if lines.len() >= correction_budget {
                break;
            }
            let Some(record) = self.spoken.get(&encoded).cloned() else {
                continue;
            };
            let old = record.sentence.trim().trim_end_matches('。');
            let new = clause.sentence.trim().trim_end_matches('。');
            lines.push(CursorLine {
                text: format!("更正：此前说「{}」；现在更新为「{}」。", old, new),
                correction: true,
                trusted: false,
            });
            corrected += 1;
            self.seq += 1;
            if let Some(record) = self.spoken.get_mut(&encoded) {
                record.fingerprint = clause.fingerprint().to_string();
                record.sentence = clause.sentence.clone();
                record.seq = self.seq;
            }
        }

        for (encoded, record) in removals {
            if lines.len() >= correction_budget {
                break;
            }
            let old = record.sentence.trim().trim_end_matches('。');
            lines.push(CursorLine {
                text: format!("更正：此前说「{}」；该说法已不再成立。", old),
                correction: true,
                trusted: false,
            });
            removed += 1;
            self.spoken.remove(&encoded);
        }

        let mut fresh = 0usize;
        let mut selected_indices: Vec<usize> = Vec::new();
        let mut trusted_lines: Vec<String> = Vec::new();
        for (index, clause) in clauses.iter().enumerate() {
            if lines.len() >= limit {
                break;
            }
            let encoded = clause.key.encode();
            if self.spoken.contains_key(&encoded) {
                continue;
            }
            self.seq += 1;
            self.spoken.insert(
                encoded,
                SpokenRecord {
                    key: clause.key.clone(),
                    fingerprint: clause.fingerprint().to_string(),
                    sentence: clause.sentence.clone(),
                    seq: self.seq,
                },
            );
            if clause.from_overlay {
                trusted_lines.push(clause.sentence.clone());
            }
            lines.push(CursorLine {
                text: clause.sentence.clone(),
                correction: false,
                trusted: clause.from_overlay,
            });
            selected_indices.push(index);
            fresh += 1;
        }

        let span = if selected_indices.is_empty() {
            None
        } else {
            let start = selected_indices[0];
            let end = selected_indices[selected_indices.len() - 1] + 1;
            self.last_index = end;
            Some((start, end))
        };
        let next_offset = span
            .map(|(_, end)| end)
            .unwrap_or_else(|| self.last_index.min(clauses.len()));
        if !lines.is_empty() {
            self.turns += 1;
        }
        // 供调用方在自检中信任会话层生成的句子（按整句精确匹配）。
        if !trusted_lines.is_empty() {
            for line in &mut lines {
                if trusted_lines.iter().any(|sentence| sentence == &line.text) {
                    line.trusted = true;
                }
            }
        }
        CursorAdvance {
            lines,
            total_lines: clauses.len(),
            fresh,
            corrected,
            removed,
            span,
            continuation,
            next_offset,
        }
    }
}

/// 一页带账本的分页结果（`text` 已含话语标记与页眉）。
#[derive(Debug, Clone)]
pub struct CursorPage {
    pub entity: String,
    pub text: String,
    pub total_lines: usize,
    pub shown: usize,
    pub next_offset: usize,
    pub fresh: usize,
    pub corrected: usize,
    pub removed: usize,
    pub done: bool,
    /// 含会话层事实、已由账本背书的句子（自检白名单）
    pub trusted: Vec<String>,
}

pub struct SymbolicGenerator {
    known_entities: HashSet<String>,
    entity_types: HashMap<String, String>,
    // 类型 → 成员实体索引
    type_members: HashMap<String, BTreeSet<String>>,
    attributes: HashMap<String, Vec<(String, String)>>,
    by_subject: HashMap<String, HashMap<String, Vec<String>>>,
    by_object: HashMap<String, HashMap<String, Vec<String>>>,
    predicate_totals: HashMap<String, usize>,
    entity_predicate_counts: HashMap<String, HashMap<String, usize>>,
    entity_relation_total: HashMap<String, usize>,
    // 谓词 → 拥有该谓词的主体中字典序最小者（全局回退原型，增量维护）
    predicate_min_entity: HashMap<String, String>,
    triples: HashSet<String>,
    total_relations: usize,
    max_profile_sentences: usize,
    completion_min_score: f64,
    // 谓词的中文表述（含词汇化动词）→ 规范谓词，供形式化语义解析复用
    predicate_surfaces: HashMap<String, String>,
    lexicalized_predicates: HashSet<String>,
    // 类型 → 谓词计数聚合缓存；知识变更（revision 自增）后失效
    revision: u64,
    type_count_cache: parking_lot::RwLock<HashMap<String, (u64, HashMap<String, usize>, usize)>>,
}

// 三元组的去重键：长度前缀保证无歧义
fn triple_key(subject: &str, predicate: &str, object: &str) -> String {
    use std::fmt::Write as _;
    let mut key = String::with_capacity(subject.len() + predicate.len() + object.len() + 24);
    let _ = write!(
        key,
        "{}#{}#{}#",
        subject.len(),
        predicate.len(),
        object.len()
    );
    key.push_str(subject);
    key.push_str(predicate);
    key.push_str(object);
    key
}

/// 深拷贝：类型计数缓存一并复制。供 WarmBase 会话分叉使用，
/// 复制后的读数与重算结果一致（缓存键含 revision，跨版本不会串味）。
impl Clone for SymbolicGenerator {
    fn clone(&self) -> Self {
        Self {
            known_entities: self.known_entities.clone(),
            entity_types: self.entity_types.clone(),
            type_members: self.type_members.clone(),
            attributes: self.attributes.clone(),
            by_subject: self.by_subject.clone(),
            by_object: self.by_object.clone(),
            predicate_totals: self.predicate_totals.clone(),
            entity_predicate_counts: self.entity_predicate_counts.clone(),
            entity_relation_total: self.entity_relation_total.clone(),
            predicate_min_entity: self.predicate_min_entity.clone(),
            triples: self.triples.clone(),
            total_relations: self.total_relations,
            max_profile_sentences: self.max_profile_sentences,
            completion_min_score: self.completion_min_score,
            predicate_surfaces: self.predicate_surfaces.clone(),
            lexicalized_predicates: self.lexicalized_predicates.clone(),
            revision: self.revision,
            type_count_cache: parking_lot::RwLock::new(self.type_count_cache.read().clone()),
        }
    }
}

impl SymbolicGenerator {
    pub fn new(max_profile_sentences: usize, completion_min_score: f64) -> Self {
        Self {
            known_entities: HashSet::new(),
            entity_types: HashMap::new(),
            type_members: HashMap::new(),
            attributes: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            predicate_totals: HashMap::new(),
            entity_predicate_counts: HashMap::new(),
            entity_relation_total: HashMap::new(),
            predicate_min_entity: HashMap::new(),
            triples: HashSet::new(),
            total_relations: 0,
            max_profile_sentences: max_profile_sentences.max(1),
            completion_min_score: completion_min_score.clamp(0.0, 1.0),
            predicate_surfaces: HashMap::new(),
            lexicalized_predicates: HashSet::new(),
            revision: 0,
            type_count_cache: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn is_known_entity(&self, name: &str) -> bool {
        self.known_entities.contains(name)
    }

    pub fn relation_count(&self) -> usize {
        self.total_relations
    }

    pub fn entity_type_of(&self, name: &str) -> Option<&str> {
        self.entity_types.get(name).map(String::as_str)
    }

    /// 已登记类型的所有实体（实体名, 类型名），供归结引擎编码 type 事实
    pub fn typed_entities(&self) -> Vec<(String, String)> {
        self.entity_types
            .iter()
            .map(|(entity, etype)| (entity.clone(), etype.clone()))
            .collect()
    }

    pub fn all_entities(&self) -> Vec<String> {
        let mut entities: Vec<String> = self.known_entities.iter().cloned().collect();
        entities.sort();
        entities
    }

    pub fn all_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.type_members.keys().cloned().collect();
        types.sort();
        types
    }

    pub fn entities_of_type(&self, etype: &str) -> Vec<String> {
        self.type_members
            .get(etype)
            .map(|members| members.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn attribute_pairs(&self, entity: &str) -> Vec<(String, String)> {
        let mut pairs = self.attributes.get(entity).cloned().unwrap_or_default();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    pub fn relation_pairs(&self, entity: &str) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .by_subject
            .get(entity)
            .map(|predicates| {
                predicates
                    .iter()
                    .flat_map(|(predicate, objects)| {
                        objects
                            .iter()
                            .map(move |object| (predicate.clone(), object.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    pub fn all_predicates(&self) -> Vec<String> {
        let mut predicates: Vec<String> = self.predicate_totals.keys().cloned().collect();
        predicates.sort();
        predicates
    }

    pub fn canonical_predicate(&self, surface: &str) -> Option<String> {
        self.predicate_surfaces.get(surface).cloned()
    }

    /// 把一条事实渲染成自然中文句子（对外复用动词化词典）
    pub fn render_fact(&self, subject: &str, predicate: &str, objects: &[String]) -> String {
        self.lexicalize(subject, predicate, objects)
    }

    pub fn has_attribute(&self, entity: &str, key: &str, value: &str) -> bool {
        self.attributes.get(entity).map_or(false, |pairs| {
            pairs.iter().any(|(k, v)| k == key && v == value)
        })
    }

    pub fn attribute_values(&self, entity: &str, key: &str) -> Vec<String> {
        let mut values = self.values_for_key(entity, key);
        if values.is_empty() {
            // 中文词面（「身高」）回落英文图谱键（height）
            if let Some(canonical) = canonical_predicate_key(key) {
                values = self.values_for_key(entity, canonical);
            }
        }
        values
    }

    fn values_for_key(&self, entity: &str, key: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .attributes
            .get(entity)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .collect()
            })
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    pub fn predicate_surfaces(&self) -> &HashMap<String, String> {
        &self.predicate_surfaces
    }

    /// 运行时登记谓词别名（谓词画像准入后同步，查询/生成共用同一词面表）。
    pub fn register_predicate_alias(&mut self, alias: &str, canonical: &str) -> bool {
        if alias.trim().is_empty() || canonical.trim().is_empty() || alias == canonical {
            return false;
        }
        if self.predicate_surfaces.get(alias).map(String::as_str) == Some(canonical) {
            return false;
        }
        self.predicate_surfaces
            .insert(canonical.to_string(), canonical.to_string());
        self.predicate_surfaces
            .insert(alias.to_string(), canonical.to_string());
        self.lexicalized_predicates.insert(alias.to_string());
        self.revision += 1;
        true
    }

    /// 实体的属性键（去重排序）：供对话管理器生成"我还知道它的……"式追问建议
    pub fn attribute_keys(&self, entity: &str) -> Vec<String> {
        let mut keys: Vec<String> = self
            .attributes
            .get(entity)
            .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
            .unwrap_or_default();
        keys.sort();
        keys.dedup();
        keys
    }

    /// 实体作为主语出现过的关系谓词（去重排序）：与属性键合并后作为可追问主题
    pub fn relation_predicates(&self, entity: &str) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .by_subject
            .get(entity)
            .map(|subjects| subjects.keys().cloned().collect())
            .unwrap_or_default();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// 实体作为宾语出现过的关系谓词（入向；去重排序）。
    /// 供语境指称消歧的义项卡比较「同一类型、不同角色」的候选。
    pub fn relation_predicates_in(&self, entity: &str) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .by_object
            .get(entity)
            .map(|objects| objects.keys().cloned().collect())
            .unwrap_or_default();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// 出向关系取值（主语, 谓词 → 宾语；去重排序）。
    pub fn relation_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .by_subject
            .get(entity)
            .and_then(|subjects| subjects.get(predicate))
            .cloned()
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    /// 入向关系取值（宾语, 谓词 → 主语；去重排序）。
    pub fn relation_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .by_object
            .get(entity)
            .and_then(|objects| objects.get(predicate))
            .cloned()
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    pub fn observe_entity(&mut self, name: &str, etype: &str, properties: &[(String, String)]) {
        let refs: Vec<(&str, &str)> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        self.observe_entity_refs(name, etype, &refs);
    }

    /// 与 `observe_entity` 等价，但直接以属性表为输入，迭代顺序与之一致。
    pub fn observe_entity_map(
        &mut self,
        name: &str,
        etype: &str,
        properties: &HashMap<String, String>,
    ) {
        let refs: Vec<(&str, &str)> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        self.observe_entity_refs(name, etype, &refs);
    }

    fn observe_entity_refs(&mut self, name: &str, etype: &str, properties: &[(&str, &str)]) {
        if name.trim().is_empty() {
            return;
        }
        // 只在知识确有变化时自增 revision
        let mut changed = false;
        if self.known_entities.insert(name.to_string()) {
            changed = true;
        }
        if !etype.trim().is_empty() {
            match self
                .entity_types
                .insert(name.to_string(), etype.to_string())
            {
                Some(previous) => {
                    if previous != etype {
                        changed = true;
                        if let Some(members) = self.type_members.get_mut(&previous) {
                            members.remove(name);
                            if members.is_empty() {
                                self.type_members.remove(&previous);
                            }
                        }
                        self.type_members
                            .entry(etype.to_string())
                            .or_default()
                            .insert(name.to_string());
                    }
                }
                None => {
                    changed = true;
                    self.type_members
                        .entry(etype.to_string())
                        .or_default()
                        .insert(name.to_string());
                }
            }
        }
        if !properties.is_empty() {
            let bucket = self.attributes.entry(name.to_string()).or_default();
            for &(key, value) in properties {
                if key.trim().is_empty() {
                    continue;
                }
                if !bucket.iter().any(|(k, v)| k == key && v == value) {
                    bucket.push((key.to_string(), value.to_string()));
                    changed = true;
                }
            }
        }
        for &(key, _) in properties {
            if !key.trim().is_empty() && self.register_predicate_surface(key) {
                changed = true;
            }
        }
        if changed {
            self.revision += 1;
        }
    }

    fn register_predicate_surface(&mut self, predicate: &str) -> bool {
        if !self.lexicalized_predicates.insert(predicate.to_string()) {
            return false;
        }
        // 规范谓词自身总是可解析；规范形式优先于其它谓词的词汇化表述
        self.predicate_surfaces
            .insert(predicate.to_string(), predicate.to_string());
        for surface in lexical_surfaces(predicate) {
            if *surface == predicate {
                continue;
            }
            // 图谱里实际出现的内置词面优先于先前登记的配置别名：例如配置
            // 把「年龄」登记成谓词画像后，再观察属性键 age 时，必须把
            // 「年龄」重新指回 age，否则形式化语义会把属性误判成关系。
            self.predicate_surfaces
                .insert(surface.to_string(), predicate.to_string());
        }
        true
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.trim().is_empty() || predicate.trim().is_empty() || object.trim().is_empty() {
            return;
        }
        let key = triple_key(subject, predicate, object);
        if !self.triples.insert(key) {
            return;
        }
        self.revision += 1;
        self.known_entities.insert(subject.to_string());
        self.known_entities.insert(object.to_string());

        // triples 已按 (s,p,o) 去重，此处到达即保证是新事实
        let subjects = match self.by_subject.get_mut(subject) {
            Some(subjects) => subjects,
            None => self.by_subject.entry(subject.to_string()).or_default(),
        };
        match subjects.get_mut(predicate) {
            Some(objects) => objects.push(object.to_string()),
            None => subjects
                .entry(predicate.to_string())
                .or_default()
                .push(object.to_string()),
        }
        let object_bucket = match self.by_object.get_mut(object) {
            Some(bucket) => bucket,
            None => self.by_object.entry(object.to_string()).or_default(),
        };
        match object_bucket.get_mut(predicate) {
            Some(subjects_rev) => subjects_rev.push(subject.to_string()),
            None => object_bucket
                .entry(predicate.to_string())
                .or_default()
                .push(subject.to_string()),
        }

        match self.predicate_totals.get_mut(predicate) {
            Some(total) => *total += 1,
            None => {
                self.predicate_totals.insert(predicate.to_string(), 1);
            }
        }
        let predicate_counts = match self.entity_predicate_counts.get_mut(subject) {
            Some(counts) => counts,
            None => self
                .entity_predicate_counts
                .entry(subject.to_string())
                .or_default(),
        };
        match predicate_counts.get_mut(predicate) {
            Some(count) => *count += 1,
            None => {
                predicate_counts.insert(predicate.to_string(), 1);
                // 该主体首次拥有此谓词：增量维护字典序最小的原型主体
                let replace = match self.predicate_min_entity.get(predicate) {
                    Some(current) => current.as_str() > subject,
                    None => true,
                };
                if replace {
                    self.predicate_min_entity
                        .insert(predicate.to_string(), subject.to_string());
                }
            }
        }
        match self.entity_relation_total.get_mut(subject) {
            Some(total) => *total += 1,
            None => {
                self.entity_relation_total.insert(subject.to_string(), 1);
            }
        }
        self.total_relations += 1;

        self.register_predicate_surface(predicate);
    }

    /// 从生成器索引中撤回一条三元组（知识图谱侧删除后调用）。
    ///
    /// 与 `observe_triple` 严格对称：计数、正/反向倒排、谓词原型一起回退，
    /// 重复撤回返回 `false` 且不改变任何状态（幂等）。实体本身保留，
    /// 因为同名的属性/类型证据可能仍然存在。
    pub fn retract_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        if !self.triples.remove(&triple_key(subject, predicate, object)) {
            return false;
        }
        self.revision += 1;

        if let Some(subjects) = self.by_subject.get_mut(subject) {
            if let Some(objects) = subjects.get_mut(predicate) {
                if let Some(position) = objects.iter().position(|existing| existing == object) {
                    objects.remove(position);
                }
                if objects.is_empty() {
                    subjects.remove(predicate);
                }
            }
            if subjects.is_empty() {
                self.by_subject.remove(subject);
            }
        }
        if let Some(bucket) = self.by_object.get_mut(object) {
            if let Some(subjects) = bucket.get_mut(predicate) {
                if let Some(position) = subjects.iter().position(|existing| existing == subject) {
                    subjects.remove(position);
                }
                if subjects.is_empty() {
                    bucket.remove(predicate);
                }
            }
            if bucket.is_empty() {
                self.by_object.remove(object);
            }
        }

        let mut predicate_gone = false;
        if let Some(total) = self.predicate_totals.get_mut(predicate) {
            *total = total.saturating_sub(1);
            if *total == 0 {
                self.predicate_totals.remove(predicate);
                predicate_gone = true;
            }
        }
        if let Some(counts) = self.entity_predicate_counts.get_mut(subject) {
            if let Some(count) = counts.get_mut(predicate) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(predicate);
                }
            }
            if counts.is_empty() {
                self.entity_predicate_counts.remove(subject);
            }
        }
        if let Some(total) = self.entity_relation_total.get_mut(subject) {
            *total = total.saturating_sub(1);
            if *total == 0 {
                self.entity_relation_total.remove(subject);
            }
        }
        self.total_relations = self.total_relations.saturating_sub(1);

        // 原型主体只能从现存主体中重新选取：只增不减会让回退指向幽灵实体。
        if predicate_gone || !self.predicate_min_entity.contains_key(predicate) {
            self.refresh_predicate_min_entity(predicate);
        } else if self.predicate_min_entity.get(predicate).map(String::as_str) == Some(subject) {
            self.refresh_predicate_min_entity(predicate);
        }
        true
    }

    /// 重算某谓词的全局回退原型（字典序最小的现存主体）。
    fn refresh_predicate_min_entity(&mut self, predicate: &str) {
        let min = self
            .by_subject
            .iter()
            .filter(|(_, predicates)| {
                predicates
                    .get(predicate)
                    .map_or(false, |objects| !objects.is_empty())
            })
            .map(|(subject, _)| subject.clone())
            .min();
        match min {
            Some(subject) => {
                self.predicate_min_entity
                    .insert(predicate.to_string(), subject);
            }
            None => {
                self.predicate_min_entity.remove(predicate);
            }
        }
    }

    // 谓词动词化：把裸谓词（含 JSON-LD 英文谓词）渲染成自然中文句子
    fn lexicalize(&self, subject: &str, predicate: &str, objects: &[String]) -> String {
        let joined = objects.join("、");
        if let Some(template) = lexical_template(predicate) {
            return template.replace("{s}", subject).replace("{o}", &joined);
        }
        let surface = localized_predicate_key(predicate).unwrap_or(predicate);
        format!("{}的{}是{}。", subject, surface, joined)
    }

    // 生成式介绍：把实体相关事实排序、模板化、加话语标记后组合成段落。
    pub fn profile(&self, entity: &str) -> Option<String> {
        let lines = self.profile_lines(entity, GenerationOrder::Relevance)?;
        Some(self.render_profile(entity, &lines, 0))
    }

    // 分页生成：支持"接着说"（offset 续航）与"换个角度说"（换排序）
    pub fn profile_page(
        &self,
        entity: &str,
        offset: usize,
        order: GenerationOrder,
    ) -> Option<ProfilePage> {
        let lines = self.profile_lines(entity, order)?;
        // offset 超出末尾时视为请求最后一页之后的内容，避免切片越界 panic
        let offset = offset.min(lines.len());
        let end = offset
            .saturating_add(self.max_profile_sentences)
            .min(lines.len());
        let shown = end.saturating_sub(offset);
        Some(ProfilePage {
            entity: entity.to_string(),
            text: self.render_profile(entity, &lines, offset),
            total_lines: lines.len(),
            shown,
            next_offset: end,
        })
    }

    /// 当前可生成的稳定命题列表（纯图谱视图，等价于空叠加层）。
    pub fn profile_clauses(
        &self,
        entity: &str,
        order: GenerationOrder,
    ) -> Option<Vec<ProfileClause>> {
        self.profile_clauses_with_overlay(entity, order, &GenerationOverlay::default())
    }

    /// 图谱视图 ∪ 会话视图，生成稳定命题列表。
    ///
    /// 功能槽已有会话赢家时以赢家为准（`replace`），否则取并集；被更正/
    /// 否定的取值先删除（`suppress`）。这样生成器本身保持「图谱视图」，
    /// 会话层的裁决由调用方以叠加层传入，避免两份事实互相污染。
    pub fn profile_clauses_with_overlay(
        &self,
        entity: &str,
        order: GenerationOrder,
        overlay: &GenerationOverlay,
    ) -> Option<Vec<ProfileClause>> {
        // 会话层可能引入图谱尚未收录的实体：它既可能是新事实的主语，
        // 也可能只作为宾语出现（此时生成反向句）。
        let overlay_has_entity = overlay
            .add
            .iter()
            .any(|((subject, _), values)| {
                subject == entity || values.iter().any(|value| value == entity)
            })
            || overlay.replace.iter().any(|((subject, _), values)| {
                subject == entity || values.iter().any(|value| value == entity)
            });
        if !self.is_known_entity(entity) && !overlay_has_entity {
            return None;
        }
        let mut predicates: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(relations) = self.by_subject.get(entity) {
            for (predicate, objects) in relations {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(objects.iter().cloned());
            }
        }
        if let Some(attributes) = self.attributes.get(entity) {
            for (key, value) in attributes {
                let relation = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |relations| relations.contains_key(key));
                if !relation {
                    predicates
                        .entry(key.clone())
                        .or_default()
                        .push(value.clone());
                }
            }
        }
        for ((subject, predicate), values) in &overlay.add {
            if subject == entity {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(values.iter().cloned());
            }
        }
        for ((subject, predicate), values) in &overlay.replace {
            if subject == entity {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(values.iter().cloned());
            }
        }

        let mut clauses: Vec<ProfileClause> = Vec::new();
        for (predicate, values) in predicates {
            let slot = (entity.to_string(), predicate.clone());
            if overlay.hide.contains(&slot) {
                continue;
            }
            let mut values = match overlay.replace.get(&slot) {
                Some(replacement) => replacement.clone(),
                None => values,
            };
            if let Some(suppressed) = overlay.suppress.get(&slot) {
                values.retain(|value| !suppressed.contains(value));
            }
            values.sort();
            values.dedup();
            if values.is_empty() {
                continue;
            }
            let relation = self
                .by_subject
                .get(entity)
                .map_or(false, |relations| relations.contains_key(&predicate));
            let attribute = !relation
                && self.attributes.get(entity).map_or(false, |attributes| {
                    attributes.iter().any(|(key, _)| key == &predicate)
                });
            let from_overlay = overlay.touches(entity, &predicate);
            if attribute {
                for value in values {
                    if is_name_key(&predicate) && value == entity {
                        continue;
                    }
                    let key = ClauseKey {
                        entity: entity.to_string(),
                        kind: ClauseKind::Attribute,
                        predicate: predicate.clone(),
                        qualifier: value.clone(),
                    };
                    let sentence =
                        self.lexicalize(entity, &predicate, std::slice::from_ref(&value));
                    clauses.push(ProfileClause {
                        key,
                        sentence,
                        from_overlay,
                        rank_milli: 500,
                    });
                }
            } else {
                let key = ClauseKey {
                    entity: entity.to_string(),
                    kind: ClauseKind::Relation,
                    predicate: predicate.clone(),
                    qualifier: String::new(),
                };
                let sentence = self.lexicalize(entity, &predicate, &values);
                clauses.push(ProfileClause {
                    key,
                    sentence,
                    from_overlay,
                    rank_milli: (self.predicate_idf(&predicate) * 1000.0).round() as i64,
                });
            }
        }

        // 反向句：只补「主语→本实体」的入向事实，且谓词已在正向出现时跳过。
        let mut reverse: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut reverse_overlay: BTreeSet<String> = BTreeSet::new();
        if let Some(relations) = self.by_object.get(entity) {
            for (predicate, subjects) in relations {
                let duplicated = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |forward| forward.contains_key(predicate));
                if duplicated {
                    continue;
                }
                reverse
                    .entry(predicate.clone())
                    .or_default()
                    .extend(subjects.iter().cloned());
            }
        }
        for ((subject, predicate), values) in &overlay.add {
            if subject != entity && values.contains(entity) {
                reverse
                    .entry(predicate.clone())
                    .or_default()
                    .insert(subject.clone());
                reverse_overlay.insert(predicate.clone());
            }
        }
        for ((subject, predicate), values) in &overlay.suppress {
            if subject != entity && values.contains(entity) {
                if let Some(subjects) = reverse.get_mut(predicate) {
                    subjects.remove(subject);
                }
            }
        }
        let mut reverse_added = 0usize;
        for (predicate, subjects) in reverse {
            if reverse_added >= 2 {
                break;
            }
            let duplicated = self
                .by_subject
                .get(entity)
                .map_or(false, |forward| forward.contains_key(&predicate));
            if duplicated || subjects.is_empty() {
                continue;
            }
            let subject_list: Vec<String> = subjects.into_iter().collect();
            let key = ClauseKey {
                entity: entity.to_string(),
                kind: ClauseKind::Reverse,
                predicate: predicate.clone(),
                qualifier: String::new(),
            };
            let sentence = self.lexicalize(
                &subject_list.join("、"),
                &predicate,
                std::slice::from_ref(&entity.to_string()),
            );
            clauses.push(ProfileClause {
                key,
                sentence,
                from_overlay: reverse_overlay.contains(&predicate),
                rank_milli: -1000,
            });
            reverse_added += 1;
        }

        match order {
            GenerationOrder::Relevance => clauses.sort_by(|a, b| {
                b.rank_milli
                    .cmp(&a.rank_milli)
                    .then_with(|| a.sentence.cmp(&b.sentence))
                    .then_with(|| a.key.cmp(&b.key))
            }),
            GenerationOrder::Alphabetical => clauses.sort_by(|a, b| {
                a.sentence
                    .cmp(&b.sentence)
                    .then_with(|| a.key.cmp(&b.key))
            }),
        }
        Some(clauses)
    }

    /// 用话语账本生成一页：先补更正/撤回，再讲尚未说过的命题。
    pub fn cursor_page(
        &self,
        entity: &str,
        order: GenerationOrder,
        overlay: &GenerationOverlay,
        cursor: &mut GenerationCursor,
    ) -> Option<CursorPage> {
        let clauses = self.profile_clauses_with_overlay(entity, order, overlay)?;
        let advance = cursor.advance(&clauses, self.max_profile_sentences);
        let text = if advance.lines.is_empty() {
            if clauses.is_empty() {
                format!(
                    "关于「{}」的知识（{}符号检索生成）：知识图谱中暂无可用事实。",
                    entity,
                    self.type_note(entity)
                )
            } else {
                String::new()
            }
        } else {
            self.render_cursor_advance(entity, &advance)
        };
        let trusted: Vec<String> = advance
            .lines
            .iter()
            .filter(|line| line.trusted)
            .map(|line| line.text.clone())
            .collect();
        let done = clauses.is_empty()
            || (advance.fresh == 0 && advance.corrected == 0 && advance.removed == 0);
        Some(CursorPage {
            entity: entity.to_string(),
            text,
            total_lines: advance.total_lines,
            shown: advance.lines.len(),
            next_offset: advance.next_offset,
            fresh: advance.fresh,
            corrected: advance.corrected,
            removed: advance.removed,
            done,
            trusted,
        })
    }

    fn type_note(&self, entity: &str) -> String {
        self.entity_types
            .get(entity)
            .filter(|t| t.as_str() != "Entity" && t.as_str() != "实体")
            .map(|t| format!("类型：{}；", localized_type_label(t)))
            .unwrap_or_default()
    }

    fn render_cursor_advance(&self, entity: &str, advance: &CursorAdvance) -> String {
        let mut body = String::new();
        let mut fresh_ordinal = 0usize;
        for line in &advance.lines {
            body.push_str("  ");
            if line.correction {
                body.push_str(&line.text);
            } else {
                if advance.continuation && fresh_ordinal == 0 {
                    body.push_str("接着，");
                } else {
                    body.push_str(Self::discourse_marker(fresh_ordinal, advance.fresh));
                }
                body.push_str(&line.text);
                fresh_ordinal += 1;
            }
            body.push('\n');
        }
        let header = if advance.continuation {
            match advance.span {
                Some((start, end)) => format!(
                    "关于「{}」的知识（续：第 {}–{} 句，共 {} 条事实）：\n",
                    entity,
                    start + 1,
                    end,
                    advance.total_lines
                ),
                None => format!(
                    "关于「{}」的知识（续：更正 {} 条，共 {} 条事实）：\n",
                    entity,
                    advance.corrected + advance.removed,
                    advance.total_lines
                ),
            }
        } else {
            format!(
                "关于「{}」的知识（{}符号检索生成，共 {} 条事实）：\n",
                entity,
                self.type_note(entity),
                advance.lines.len()
            )
        };
        format!("{}{}", header, body.trim_end())
    }

    fn profile_lines(&self, entity: &str, order: GenerationOrder) -> Option<Vec<FactLine>> {
        if !self.is_known_entity(entity) {
            return None;
        }
        let mut lines: Vec<FactLine> = Vec::new();

        if let Some(predicates) = self.by_subject.get(entity) {
            for (predicate, objects) in predicates {
                let idf = self.predicate_idf(predicate);
                lines.push(FactLine {
                    rank: idf,
                    sentence: self.lexicalize(entity, predicate, objects),
                });
            }
        }
        if let Some(attributes) = self.attributes.get(entity) {
            for (key, value) in attributes {
                // 属性与关系是同一谓词时只保留关系句，避免重复生成
                let duplicated = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |predicates| predicates.contains_key(key));
                if duplicated {
                    continue;
                }
                // 姓名与实体名相同属于同义反复，跳过以免生成「X的名字是X」
                if is_name_key(key) && value == entity {
                    continue;
                }
                lines.push(FactLine {
                    rank: 0.5,
                    sentence: self.lexicalize(entity, key, std::slice::from_ref(value)),
                });
            }
        }
        let reverse_sentences = self.reverse_sentences(entity, 2);
        for sentence in reverse_sentences {
            lines.push(FactLine {
                rank: -1.0,
                sentence,
            });
        }

        match order {
            GenerationOrder::Relevance => lines.sort_by(|a, b| {
                b.rank
                    .partial_cmp(&a.rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.sentence.cmp(&b.sentence))
            }),
            GenerationOrder::Alphabetical => lines.sort_by(|a, b| a.sentence.cmp(&b.sentence)),
        }
        Some(lines)
    }

    fn render_profile(&self, entity: &str, lines: &[FactLine], offset: usize) -> String {
        let type_note = self
            .entity_types
            .get(entity)
            .filter(|t| t.as_str() != "Entity" && t.as_str() != "实体")
            .map(|t| format!("类型：{}；", localized_type_label(t)))
            .unwrap_or_default();
        if lines.is_empty() {
            return format!(
                "关于「{}」的知识（{}符号检索生成）：知识图谱中暂无可用事实。",
                entity, type_note
            );
        }
        let end = offset
            .saturating_add(self.max_profile_sentences)
            .min(lines.len());
        let page = &lines[offset..end];
        let continuation = offset > 0;

        let mut body = String::new();
        for (index, line) in page.iter().enumerate() {
            body.push_str("  ");
            if continuation && index == 0 {
                body.push_str("接着，");
            } else {
                body.push_str(&Self::discourse_marker(index, page.len()));
            }
            body.push_str(&line.sentence);
            body.push('\n');
        }
        let header = if continuation {
            format!(
                "关于「{}」的知识（续：第 {}–{} 句，共 {} 条事实）：\n",
                entity,
                offset + 1,
                end,
                lines.len()
            )
        } else {
            format!(
                "关于「{}」的知识（{}符号检索生成，共 {} 条事实）：\n",
                entity,
                type_note,
                page.len()
            )
        };
        format!("{}{}", header, body.trim_end())
    }

    fn reverse_sentences(&self, entity: &str, limit: usize) -> Vec<String> {
        let Some(predicates) = self.by_object.get(entity) else {
            return Vec::new();
        };
        let mut sentences: Vec<String> = Vec::new();
        let mut predicate_names: Vec<&String> = predicates.keys().collect();
        predicate_names.sort();
        for predicate in predicate_names {
            let duplicated = self
                .by_subject
                .get(entity)
                .map_or(false, |forward| forward.contains_key(predicate));
            if duplicated {
                continue;
            }
            let subjects = &predicates[predicate];
            sentences.push(self.lexicalize(&subjects.join("、"), predicate, &[entity.to_string()]));
            if sentences.len() >= limit {
                break;
            }
        }
        sentences
    }

    fn predicate_idf(&self, predicate: &str) -> f64 {
        let count = self.predicate_totals.get(predicate).copied().unwrap_or(0);
        ((1.0 + self.total_relations as f64) / (1.0 + count as f64)).ln()
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先，"
        } else if index + 1 == total {
            "最后，"
        } else {
            "此外，"
        }
    }

    // 统计式续写：给定前缀「X的」，预测下一个谓词并生成句子。
    pub fn complete(&self, prefix: &str) -> Option<String> {
        let trimmed = prefix
            .trim()
            .trim_end_matches(['。', '？', '?', '！', '!', ' ']);
        let entity = trimmed.strip_suffix('的')?.trim();
        if entity.is_empty() || !self.known_entities.contains(entity) {
            return None;
        }

        let predictions = self
            .entity_level_predictions(entity)
            .or_else(|| self.type_level_predictions(entity))
            .or_else(|| self.global_predictions());
        let predictions = predictions?;
        if predictions.is_empty() {
            return None;
        }

        let mut output = String::from("符号续写（知识图谱统计回退，无神经网络）：\n");
        output.push_str(&format!("  前缀：「{}的」\n", entity));
        for (index, prediction) in predictions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 谓词「{}」（{}，支持度 {}，置信度 {:.2}）：{}\n",
                index + 1,
                prediction.predicate,
                prediction.level.label(),
                prediction.support,
                prediction.score,
                prediction.sentence
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn entity_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let predicate_counts = self.entity_predicate_counts.get(entity)?;
        let total = self
            .entity_relation_total
            .get(entity)
            .copied()
            .unwrap_or(1)
            .max(1);
        let mut items: Vec<(&String, &usize)> = predicate_counts.iter().collect();
        // 取前 3 名
        if items.len() > 3 {
            items.select_nth_unstable_by(2, |a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            items.truncate(3);
        }
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter().take(3) {
            let objects = self
                .by_subject
                .get(entity)
                .and_then(|m| m.get(predicate))
                .cloned()
                .unwrap_or_default();
            if objects.is_empty() {
                continue;
            }
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Entity,
                score: *count as f64 / total as f64,
                support: format!("{}/{}", count, total),
                sentence: self.lexicalize(entity, predicate, &objects),
            });
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn type_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let etype = self.entity_types.get(entity)?;
        self.with_type_counts(etype, |type_counts, type_total| {
            if type_counts.is_empty() || type_total == 0 {
                return None;
            }
            let mut items: Vec<(&String, &usize)> = type_counts.iter().collect();
            items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let mut predictions: Vec<CompletionPrediction> = Vec::new();
            for (predicate, count) in items.into_iter() {
                let score = *count as f64 / type_total as f64;
                if score < self.completion_min_score {
                    // 已按计数降序，后续得分只会更低
                    break;
                }
                let Some((prototype, objects)) = self.prototype_for(etype, entity, predicate)
                else {
                    continue;
                };
                predictions.push(CompletionPrediction {
                    predicate: predicate.clone(),
                    level: CompletionLevel::Type,
                    score,
                    support: format!("{}/{}", count, type_total),
                    sentence: format!(
                        "参照同类实体「{}」：{}的{}很可能是{}。",
                        prototype,
                        entity,
                        predicate,
                        objects.join("、")
                    ),
                });
                if predictions.len() >= 3 {
                    break;
                }
            }
            if predictions.is_empty() {
                None
            } else {
                Some(predictions)
            }
        })
    }

    /// 用带失效机制的缓存提供某类型的谓词计数聚合结果
    fn with_type_counts<R>(
        &self,
        etype: &str,
        f: impl FnOnce(&HashMap<String, usize>, usize) -> R,
    ) -> R {
        let revision = self.revision;
        {
            let cache = self.type_count_cache.read();
            if let Some((cached_revision, counts, total)) = cache.get(etype) {
                if *cached_revision == revision {
                    return f(counts, *total);
                }
            }
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        if let Some(members) = self.type_members.get(etype) {
            for entity in members {
                if let Some(predicate_counts) = self.entity_predicate_counts.get(entity) {
                    for (predicate, count) in predicate_counts {
                        // 键通常已存在：命中时直接累加
                        match counts.get_mut(predicate) {
                            Some(existing) => *existing += count,
                            None => {
                                counts.insert(predicate.clone(), *count);
                            }
                        }
                        total += count;
                    }
                }
            }
        }
        let result = f(&counts, total);
        self.type_count_cache
            .write()
            .insert(etype.to_string(), (revision, counts, total));
        result
    }

    fn prototype_for(
        &self,
        etype: &str,
        entity: &str,
        predicate: &str,
    ) -> Option<(String, Vec<String>)> {
        let members = self.type_members.get(etype)?;
        let mut best: Option<(usize, &str)> = None;
        for other in members {
            if other == entity {
                continue;
            }
            let count = self
                .entity_predicate_counts
                .get(other)
                .and_then(|m| m.get(predicate))
                .copied()
                .unwrap_or(0);
            if count == 0 {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((best_count, best_name)) => {
                    count > *best_count || (count == *best_count && other.as_str() < *best_name)
                }
            };
            if replace {
                best = Some((count, other.as_str()));
            }
        }
        let (_, prototype) = best?;
        let objects = self
            .by_subject
            .get(prototype)
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some((prototype.to_string(), objects))
    }

    fn global_predictions(&self) -> Option<Vec<CompletionPrediction>> {
        if self.total_relations == 0 {
            return None;
        }
        let mut items: Vec<(&String, &usize)> = self.predicate_totals.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter() {
            let score = *count as f64 / self.total_relations as f64;
            if score < self.completion_min_score {
                // 已按计数降序，后续得分只会更低
                break;
            }
            let Some((prototype, objects)) = self.global_prototype(predicate) else {
                continue;
            };
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Global,
                score,
                support: format!("{}/{}", count, self.total_relations),
                sentence: format!(
                    "（全局回退）知识图谱中最常见的关系是「{}」，例如「{}」的「{}」是「{}」。",
                    predicate,
                    prototype,
                    predicate,
                    objects.join("、")
                ),
            });
            if predictions.len() >= 3 {
                break;
            }
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn global_prototype(&self, predicate: &str) -> Option<(String, Vec<String>)> {
        // 增量的“字典序最小主体”索引
        let prototype = self.predicate_min_entity.get(predicate)?;
        let objects = self
            .by_subject
            .get(prototype.as_str())
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some((prototype.clone(), objects))
    }
}

// 谓词动词化词典：规范谓词 → 中文句模板（{s}=主语，{o}=宾语）
fn lexical_template(predicate: &str) -> Option<&'static str> {
    Some(match predicate {
        "knows" | "认识" => "{s}认识{o}。",
        "age" | "年龄" => "{s}的年龄是{o}。",
        "capital" | "首都" => "{s}的首都是{o}。",
        "author" | "作者" => "{s}的作者是{o}。",
        "birthPlace" | "出生地" | "出生于" => "{s}出生于{o}。",
        "founder" | "创始人" => "{s}的创始人是{o}。",
        "memberOf" | "隶属于" => "{s}隶属于{o}。",
        "locatedIn" | "位于" => "{s}位于{o}。",
        "worksAt" | "工作于" => "{s}工作于{o}。",
        "spouse" | "配偶" | "妻子" | "丈夫" => "{s}的配偶是{o}。",
        "child" | "孩子" | "子女" => "{s}的子女是{o}。",
        "likes" | "喜欢" => "{s}喜欢{o}。",
        "nationality" | "国籍" => "{s}的国籍是{o}。",
        "occupation" | "职业" => "{s}的职业是{o}。",
        _ => return None,
    })
}

/// 常见英文谓词键的中文表述（生成叙事用；图谱内的谓词本身不变）。
pub fn localized_predicate_key(key: &str) -> Option<&'static str> {
    PREDICATE_LABELS
        .iter()
        .find(|(canonical, _)| *canonical == key)
        .map(|(_, label)| *label)
}

/// 中文谓词词面 → 图谱英文键（「身高」→ height）：
/// 供按中文属性名检索字面量取值时做归一。
pub fn canonical_predicate_key(label: &str) -> Option<&'static str> {
    PREDICATE_LABELS
        .iter()
        .find(|(_, surface)| *surface == label)
        .map(|(canonical, _)| *canonical)
}

/// 图谱属性键与中文词面的对照表只读视图（英文键 → 中文），
/// 供查询规划器识别「实体 + 属性 + 多少」这类省略「的/是」的问法。
pub fn predicate_labels() -> &'static [(&'static str, &'static str)] {
    PREDICATE_LABELS
}

/// 图谱属性键与中文词面的对照表（英文键 → 中文）。
const PREDICATE_LABELS: &[(&str, &str)] = &[
    ("name", "姓名"),
    ("gender", "性别"),
    ("height", "身高"),
    ("weight", "体重"),
    ("birthDate", "出生日期"),
    ("capital", "首都"),
    ("population", "人口"),
    ("area", "面积"),
    ("author", "作者"),
    ("founder", "创始人"),
    ("location", "所在地"),
    ("locatedIn", "所在地"),
    ("worksAt", "工作单位"),
    ("memberOf", "隶属于"),
    ("nationality", "国籍"),
    ("occupation", "职业"),
    ("spouse", "配偶"),
    ("child", "子女"),
    ("age", "年龄"),
    ("birthPlace", "出生地"),
];

/// 是否姓名类谓词（生成画像时与实体名相同的取值属于同义反复）。
fn is_name_key(key: &str) -> bool {
    matches!(key, "name" | "姓名" | "名字")
}

/// 常见英文类型标签的中文表述（仅用于生成与展示兜底）。
pub fn localized_type_label(etype: &str) -> &str {
    match etype.to_ascii_lowercase().as_str() {
        "person" | "people" | "human" => "人物",
        "entity" => "实体",
        "location" | "place" => "地点",
        "city" => "城市",
        "country" => "国家",
        "organization" | "organisation" | "company" => "组织",
        "event" => "事件",
        "work" => "作品",
        "animal" => "动物",
        "food" => "食物",
        _ => etype,
    }
}

// 中文表述集合（首个为动词化时选择的面）→ 供形式化语义解析识别
fn lexical_surfaces(predicate: &str) -> &'static [&'static str] {
    match predicate {
        "knows" | "认识" => &["认识"],
        "age" | "年龄" => &["年龄"],
        "capital" | "首都" => &["首都"],
        "population" | "人口" => &["人口"],
        "area" | "面积" => &["面积"],
        "author" | "作者" => &["作者"],
        "birthPlace" | "出生地" | "出生于" => &["出生地", "出生于"],
        "founder" | "创始人" => &["创始人"],
        "memberOf" | "隶属于" => &["隶属于"],
        "locatedIn" | "位于" => &["位于", "所在地"],
        "worksAt" | "工作于" => &["工作于", "任职于"],
        "spouse" | "配偶" | "妻子" | "丈夫" => &["配偶"],
        "child" | "孩子" | "子女" => &["孩子", "子女"],
        "likes" | "喜欢" => &["喜欢"],
        "nationality" | "国籍" => &["国籍"],
        "occupation" | "职业" => &["职业"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_generator() -> SymbolicGenerator {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_entity("李四", "人物", &[]);
        generator.observe_entity("王五", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("李四", "师傅", "王五");
        generator.observe_triple("张三", "师祖", "王五");
        generator
    }

    #[test]
    fn test_profile_generation_uses_facts_and_markers() {
        let generator = sample_generator();
        let profile = generator.profile("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("类型：人物"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("李四"));
        assert!(profile.contains("年龄"));
        assert!(profile.contains("30"));
        assert!(profile.contains("首先"));
    }

    #[test]
    fn test_profile_unknown_entity_is_none() {
        let generator = sample_generator();
        assert!(generator.profile("不存在的人").is_none());
    }

    #[test]
    fn test_complete_entity_level_prediction() {
        let generator = sample_generator();
        let completion = generator.complete("张三的").expect("实体级统计应有预测");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));
        assert!(completion.contains("张三的师傅是李四"));
    }

    #[test]
    fn test_complete_type_level_backoff() {
        let mut generator = sample_generator();
        generator.observe_entity("赵六", "人物", &[]);
        let completion = generator.complete("赵六的").expect("类型级回退应有预测");
        assert!(
            completion.contains("类型级回退"),
            "实际输出：{}",
            completion
        );
        assert!(completion.contains("师傅"));
        assert!(completion.contains("参照同类实体"));
    }

    #[test]
    fn test_complete_global_backoff() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "类型A", &[]);
        generator.observe_triple("乙", "朋友", "丙");
        let completion = generator.complete("甲的").expect("全局回退应有预测");
        assert!(completion.contains("全局回退"), "实际输出：{}", completion);
        assert!(completion.contains("朋友"));
    }

    #[test]
    fn test_complete_requires_known_entity_and_de_suffix() {
        let generator = sample_generator();
        assert!(generator.complete("不存在的人的").is_none());
        assert!(generator.complete("张三").is_none());
    }

    #[test]
    fn test_profile_respects_sentence_limit() {
        let mut generator = SymbolicGenerator::new(2, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let profile = generator.profile("张三").unwrap();
        let sentences = profile.matches('。').count();
        assert!(sentences <= 2, "截断后应不超过 2 句：{}", profile);
    }

    #[test]
    fn test_lexicalization_maps_bare_predicates_to_chinese() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_triple("张三", "knows", "李四");
        generator.observe_triple("李四", "age", "30");
        let profile = generator.profile("张三").unwrap();
        assert!(
            profile.contains("张三认识李四"),
            "裸谓词应动词化：{}",
            profile
        );
        let profile2 = generator.profile("李四").unwrap();
        assert!(
            profile2.contains("李四的年龄是30"),
            "英文属性应中文化：{}",
            profile2
        );
        let surfaces = generator.predicate_surfaces();
        assert_eq!(surfaces.get("认识").map(String::as_str), Some("knows"));
        assert_eq!(surfaces.get("age").map(String::as_str), Some("age"));
    }

    #[test]
    fn test_observed_attribute_surface_overrides_earlier_alias() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        // 配置谓词画像先登记「岁数 → 年龄」。
        assert!(generator.register_predicate_alias("岁数", "年龄"));
        assert_eq!(generator.canonical_predicate("年龄").as_deref(), Some("年龄"));
        // 随后图谱出现英文属性 age：内置中文词面「年龄」应重新指回 age。
        generator.observe_entity("张三", "人物", &[("age".to_string(), "30".to_string())]);
        assert_eq!(generator.canonical_predicate("年龄").as_deref(), Some("age"));
        assert_eq!(generator.canonical_predicate("岁数").as_deref(), Some("年龄"));
    }

    #[test]
    fn test_chinese_attribute_alias_reaches_english_key() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("北京", "City", &[("population".to_string(), "2189万".to_string())]);
        assert_eq!(canonical_predicate_key("人口"), Some("population"));
        assert_eq!(localized_predicate_key("population"), Some("人口"));
        assert_eq!(generator.attribute_values("北京", "人口"), vec!["2189万"]);
    }

    #[test]
    fn test_profile_page_supports_continuation_and_reorder() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let first = generator
            .profile_page("张三", 0, GenerationOrder::Relevance)
            .expect("第一页应可生成");
        assert_eq!(first.shown, 1);
        assert_eq!(first.total_lines, 2);
        assert_eq!(first.next_offset, 1);
        let second = generator
            .profile_page("张三", first.next_offset, GenerationOrder::Relevance)
            .expect("第二页应可生成");
        assert_eq!(second.shown, 1);
        assert!(
            second.text.contains("续：第 2–2 句"),
            "续页应有页码：{}",
            second.text
        );
        assert!(second.text.contains("接着，"));
        assert_ne!(first.text, second.text);
        let alphabetical_first = generator
            .profile_page("张三", 0, GenerationOrder::Alphabetical)
            .expect("换排序也应可生成");
        assert_eq!(alphabetical_first.total_lines, 2);
        let done = generator
            .profile_page("张三", 2, GenerationOrder::Relevance)
            .expect("翻到末页应返回空页");
        assert_eq!(done.shown, 0);
    }

    #[test]
    fn test_retract_triple_is_incremental_and_idempotent() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_triple("甲", "朋友", "乙");
        generator.observe_triple("丙", "朋友", "丁");
        assert_eq!(generator.relation_count(), 2);

        assert!(generator.retract_triple("甲", "朋友", "乙"));
        assert_eq!(generator.relation_count(), 1);
        assert!(generator.relation_values("甲", "朋友").is_empty());
        // 反向索引同步撤回，不再把甲当作「丁的朋友」
        assert!(generator.relation_values_in("丁", "朋友").contains(&"丙".to_string()));
        assert!(!generator.retract_triple("甲", "朋友", "乙"));

        // 全局回退原型必须从现存主体重算，不能指向已撤回的甲。
        let completion = generator.complete("甲的").expect("仍应给出全局回退");
        assert!(completion.contains("全局回退"), "实际：{}", completion);
        assert!(completion.contains("丙"), "原型不应指向幽灵实体：{}", completion);
        assert!(!completion.contains("甲的朋友"), "{}", completion);
    }

    #[test]
    fn test_cursor_does_not_repeat_when_insertion_reorders() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let clauses = generator
            .profile_clauses("张三", GenerationOrder::Relevance)
            .unwrap();
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert_eq!(first.shown, 1);
        let first_sentence = first.trusted.first().cloned().unwrap_or_else(|| {
            first
                .text
                .lines()
                .find(|line| line.contains('。'))
                .unwrap()
                .trim()
                .to_string()
        });

        // 新事实插入到排序最前，旧实现会让 offset 指向另一句，导致重复/漏讲。
        generator.observe_triple("张三", "朋友", "赵六");
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert_eq!(second.shown, 1);
        assert!(
            !second.text.contains(first_sentence.trim_start_matches("接着，")),
            "已说过的句子不应重复：第一页={}；第二页={}",
            first.text,
            second.text
        );
        assert!(second.text.contains("接着，"), "{}", second.text);
        // 两轮之后仍未讲完，账本里不能有重复 key。
        assert_eq!(cursor.spoken_count(), 2);
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_cursor_reports_updates_and_removals() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(first.text.contains("李四"));

        assert!(generator.retract_triple("张三", "师傅", "李四"));
        generator.observe_triple("张三", "师傅", "王五");
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(second.text.contains("更正"), "{}", second.text);
        assert!(second.text.contains("此前说"), "{}", second.text);
        assert!(second.text.contains("现在更新为"), "{}", second.text);
        assert!(second.text.contains("王五"), "{}", second.text);

        assert!(generator.retract_triple("张三", "师傅", "王五"));
        let third = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(third.text.contains("已不再成立"), "{}", third.text);
    }

    #[test]
    fn test_cursor_uses_layered_session_overlay() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        let mut overlay = GenerationOverlay::default();
        overlay
            .replace
            .insert(("张三".to_string(), "师傅".to_string()), vec!["王五".to_string()]);
        overlay
            .suppress
            .entry(("张三".to_string(), "师傅".to_string()))
            .or_default()
            .insert("李四".to_string());

        let clauses = generator
            .profile_clauses_with_overlay("张三", GenerationOrder::Relevance, &overlay)
            .unwrap();
        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].sentence.contains("王五"));
        assert!(!clauses[0].sentence.contains("李四"));
        assert!(clauses[0].from_overlay);

        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let page = generator
            .cursor_page("张三", GenerationOrder::Relevance, &overlay, &mut cursor)
            .unwrap();
        assert!(page.text.contains("王五"));
        assert!(!page.text.contains("李四"));
        assert_eq!(page.trusted.len(), 1);
    }

    #[test]
    fn test_cursor_survives_serialization_and_continues_without_repeat() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        let encoded = serde_json::to_string(&cursor).expect("账本应可序列化");
        let mut restored: GenerationCursor = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.spoken_count(), 1);
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut restored,
            )
            .unwrap();
        assert_ne!(first.text, second.text);
        assert!(second.text.contains("续：第 2–2 句"), "{}", second.text);
        assert!(!second.text.contains(first.text.lines().last().unwrap()));
    }

    #[test]
    fn test_type_members_are_ordered_and_follow_type_changes() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("乙", "人物", &[]);
        generator.observe_entity("甲", "人物", &[]);
        assert_eq!(generator.entities_of_type("人物"), vec!["乙", "甲"]);

        generator.observe_entity("乙", "组织", &[]);
        assert_eq!(generator.entities_of_type("人物"), vec!["甲"]);
        assert_eq!(generator.entities_of_type("组织"), vec!["乙"]);
    }
}

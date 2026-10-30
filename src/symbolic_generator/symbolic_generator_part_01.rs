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
/// 类型画像中一个谓词的来源：关系句、属性句，或两种用法在成员间并存。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeFactKind {
    Relation,
    Attribute,
    Mixed,
}
impl TypeFactKind {
    pub fn label(self) -> &'static str {
        match self {
            TypeFactKind::Relation => "关系",
            TypeFactKind::Attribute => "属性",
            TypeFactKind::Mixed => "关系/属性",
        }
    }
}
/// 类型画像中的一条样例：某个成员在该谓词上的取值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeFactSample {
    pub entity: String,
    pub values: Vec<String>,
}
/// 类型画像的谓词切片（成员覆盖度 + 取值数 + 有界样例）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypePredicateSummary {
    pub predicate: String,
    pub label: String,
    pub kind: TypeFactKind,
    pub member_count: usize,
    pub value_count: usize,
    /// 拥有该谓词的成员占比（千分比，四舍五入）。
    pub coverage_milli: u32,
    pub samples: Vec<TypeFactSample>,
}
/// 一个类型的物化画像：成员、谓词分布、覆盖度与确定性样例。
///
/// 画像是纯派生数据：`revision` 是该类型自己的版本戳，只有影响该类型
/// 成员事实的写入才推进它，因此不同类型的缓存可以独立失效。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeProfile {
    pub type_name: String,
    pub revision: u64,
    pub total_members: usize,
    pub members: Vec<String>,
    pub members_truncated: bool,
    pub covered_members: usize,
    pub total_predicates: usize,
    pub predicates: Vec<TypePredicateSummary>,
    pub predicates_truncated: bool,
    pub samples: Vec<String>,
}
/// 类型成员的一页（稳定字典序）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeMemberPage {
    pub type_name: String,
    pub offset: usize,
    pub next_offset: usize,
    pub total: usize,
    pub done: bool,
    pub members: Vec<String>,
}
/// 两个类型的画像对比：共享谓词与各自独有的谓词。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeComparison {
    pub left: TypeProfile,
    pub right: TypeProfile,
    pub shared_predicates: Vec<String>,
    pub left_only_predicates: Vec<String>,
    pub right_only_predicates: Vec<String>,
}
const TYPE_PROFILE_MAX_MEMBERS: usize = 256;
const TYPE_PROFILE_SAMPLES_PER_PREDICATE: usize = 2;
const TYPE_PROFILE_DEFAULT_PREDICATES: usize = 12;
const TYPE_PROFILE_DEFAULT_SAMPLES: usize = 6;
impl TypeProfile {
    /// 人读报告（确定性排序；超长成员表只展示前 20 个并显式披露）。
    pub fn render(&self) -> String {
        let mut output = format!(
            "类型画像「{}」（类型版本 v{}；成员 {} 个）：\n",
            self.type_name, self.revision, self.total_members
        );
        if self.members.is_empty() {
            output.push_str("  （没有显式成员。）");
            return output;
        }
        let shown = self.members.iter().take(20).cloned().collect::<Vec<_>>();
        output.push_str(&format!("  成员：{}", shown.join("、")));
        if self.total_members > shown.len() {
            output.push_str(&format!(" 等 {} 个", self.total_members));
        }
        if self.members_truncated {
            output.push_str(&format!(
                "（画像成员表仅保留前 {} 个，覆盖度仍按全部成员统计）",
                TYPE_PROFILE_MAX_MEMBERS
            ));
        }
        output.push('\n');

        let coverage = if self.total_members == 0 {
            0
        } else {
            (self.covered_members * 1000 / self.total_members).min(1000)
        };
        let calibrated = crate::quantifier::calibrate_counts(
            self.covered_members as u64,
            self.total_members as u64,
        )
        .map(|estimate| estimate.label_zh())
        .filter(|label| !label.is_empty())
        .map(|label| format!("，{}", label))
        .unwrap_or_default();
        output.push_str(&format!(
            "  覆盖：{}/{} 个成员至少有 1 条事实（{}%{}）；谓词 {} 种。\n",
            self.covered_members,
            self.total_members,
            coverage / 10,
            calibrated,
            self.total_predicates
        ));
        if self.predicates.is_empty() {
            output.push_str("  （这些成员尚无可生成的事实。）");
            return output;
        }
        output.push_str("  常见谓词：\n");
        for (index, summary) in self.predicates.iter().enumerate() {
            output.push_str(&format!(
                "    {}. {}（{}）：{}/{} 个成员（{}%），共 {} 个取值",
                index + 1,
                summary.label,
                summary.kind.label(),
                summary.member_count,
                self.total_members,
                summary.coverage_milli / 10,
                summary.value_count
            ));
            if let Some(sample) = summary.samples.first() {
                output.push_str(&format!(
                    "；样例：{} → {}",
                    sample.entity,
                    sample.values.join("、")
                ));
            }
            output.push('\n');
        }
        if self.predicates_truncated {
            output.push_str(&format!(
                "    （其余 {} 种谓词未展开。）\n",
                self.total_predicates.saturating_sub(self.predicates.len())
            ));
        }
        if !self.samples.is_empty() {
            output.push_str("  样例事实：\n");
            for sentence in &self.samples {
                output.push_str(&format!("    - {}\n", sentence));
            }
        }
        output.trim_end().to_string()
    }
}
impl TypeMemberPage {
    pub fn render(&self) -> String {
        let shown = self.members.len();
        let first = self.offset + usize::from(shown > 0);
        let last = self.offset + shown;
        let mut output = format!(
            "类型「{}」的成员（第 {}–{} 个，共 {} 个）：\n",
            self.type_name, first, last, self.total
        );
        if self.members.is_empty() {
            output.push_str("  （这一页没有成员。）");
            return output;
        }
        for (index, member) in self.members.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", self.offset + index + 1, member));
        }
        if self.done {
            output.push_str("  已到成员表末尾。");
        } else {
            output.push_str(&format!(
                "  还有 {} 个未显示；可用 --offset {} 继续查看。",
                self.total.saturating_sub(self.next_offset),
                self.next_offset
            ));
        }
        output.trim_end().to_string()
    }
}
impl TypeComparison {
    pub fn render(&self) -> String {
        let mut output = format!(
            "类型对比「{}」与「{}」：\n",
            self.left.type_name, self.right.type_name
        );
        output.push_str(&format!(
            "  成员：{} {} 个；{} {} 个。\n",
            self.left.type_name,
            self.left.total_members,
            self.right.type_name,
            self.right.total_members
        ));
        let label = |predicate: &str| {
            localized_predicate_key(predicate)
                .unwrap_or(predicate)
                .to_string()
        };
        let shared = self
            .shared_predicates
            .iter()
            .map(|predicate| label(predicate))
            .collect::<Vec<_>>();
        let left_only = self
            .left_only_predicates
            .iter()
            .map(|predicate| label(predicate))
            .collect::<Vec<_>>();
        let right_only = self
            .right_only_predicates
            .iter()
            .map(|predicate| label(predicate))
            .collect::<Vec<_>>();
        output.push_str(&format!(
            "  共同谓词（{}）：{}\n",
            shared.len(),
            if shared.is_empty() {
                "无".to_string()
            } else {
                shared.join("、")
            }
        ));
        output.push_str(&format!(
            "  仅「{}」（{}）：{}\n",
            self.left.type_name,
            left_only.len(),
            if left_only.is_empty() {
                "无".to_string()
            } else {
                left_only.join("、")
            }
        ));
        output.push_str(&format!(
            "  仅「{}」（{}）：{}\n",
            self.right.type_name,
            right_only.len(),
            if right_only.is_empty() {
                "无".to_string()
            } else {
                right_only.join("、")
            }
        ));
        output.push_str(
            "  说明：共同谓词是共享范畴线索，独有谓词是区分线索；均只基于显式 rdf:type 成员。",
        );
        output
    }
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
    /// 本句依据的登记三元组（主体, 谓词, 客体）；用于回答计划的证据引用。
    pub evidence: Vec<(String, String, String)>,
    rank_milli: i64,
}
impl ProfileClause {
    pub fn fingerprint(&self) -> &str {
        &self.sentence
    }

    /// 稳定槽位编码：回答计划与话语账本共用同一身份。
    pub fn slot_id(&self) -> String {
        self.key.encode()
    }

    pub fn predicate(&self) -> &str {
        &self.key.predicate
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
                (record.fingerprint != clause.fingerprint()).then(|| (encoded.clone(), *clause))
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
    // 配置态属性规范名 → 首选中文词面（新属性无需改代码即可被说出口）
    attribute_labels: HashMap<String, String>,
    lexicalized_predicates: HashSet<String>,
    // 可选的同一份双向语义规则包：生成侧复用抽取侧配置的 realize 模板
    semantic_grammar: Option<crate::semantic_grammar::RulePack>,
    // 导入可逆帧：把导入句学到的措辞用于生成与保守问法回答。
    import_frames: Option<crate::import_frames::ImportFrameLedger>,
    // 谓词中心语义角色帧：同一份 schema 驱动角色标注与反向生成。
    frame_book: Option<crate::frame_semantics::FrameBook>,
    // 类型 → 谓词计数聚合缓存；知识变更（revision 自增）后失效
    revision: u64,
    type_count_cache: parking_lot::RwLock<HashMap<String, (u64, HashMap<String, usize>, usize)>>,
    // 类型 → 该类型自己的版本戳：只有影响该类型成员事实的写入才推进
    type_versions: HashMap<String, u64>,
    type_version_counter: u64,
    // 类型画像缓存：键含「类型 + 有界参数」，只按该类型的版本戳失效
    type_profile_cache: parking_lot::RwLock<HashMap<(String, usize, usize), (u64, TypeProfile)>>,
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
            attribute_labels: self.attribute_labels.clone(),
            lexicalized_predicates: self.lexicalized_predicates.clone(),
            semantic_grammar: self.semantic_grammar.clone(),
            import_frames: self.import_frames.clone(),
            frame_book: self.frame_book.clone(),
            revision: self.revision,
            type_count_cache: parking_lot::RwLock::new(self.type_count_cache.read().clone()),
            type_versions: self.type_versions.clone(),
            type_version_counter: self.type_version_counter,
            type_profile_cache: parking_lot::RwLock::new(self.type_profile_cache.read().clone()),
        }
    }
}

use std::collections::{HashMap, HashSet};
use serde::Serialize;
use crate::cache::RevisionLru;
use crate::function_words::{projection_set, Projection};
use crate::nlp::shared_jieba;
use crate::prolog_engine::PrologFact;
/// 查询/文档共用的停用词（唯一正本见 `function_words`）
fn stopwords() -> &'static HashSet<&'static str> {
    projection_set(Projection::Retrieval)
}
/// RDFS 子类谓词：观察到这类三元组时顺手登记类型层级边（与话语引擎
/// [`crate::discourse::is_hierarchy_predicate`] 同词面，但排除「类型」/「rdf:type」
/// ——那是实例断言，由 [`DenseRetriever::observe_entity`] 处理）。
const SUBCLASS_PREDICATES: [&str; 5] = ["子类", "子类于", "是一种", "属于", "subClassOf"];
/// 图谱中表示子属性边的谓词：观察到即登记谓词层级（与子类谓词对称）。
const SUBPROPERTY_PREDICATES: [&str; 4] = ["子关系", "子属性", "子关系于", "subPropertyOf"];
/// 子类可达性的层数上限（中文领域层级很浅；上限只为防病态图）
const MAX_SUBCLASS_DEPTH: usize = 16;
/// 同一检索状态下可复用的查询结果缓存容量。
const QUERY_CACHE_CAPACITY: usize = 32;

/// 单条被检索的证据（三元组或实体属性）
#[derive(Debug, Clone, Serialize)]
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
/// 检索时的精确约束：全部字段都是可选合取条件。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SearchConstraints {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// 实体的声明类型（主体或客体任一匹配即可）。
    /// 按 RDFS 展开：声明类型与约束类型相等、互为子类路径均算匹配；
    /// 仅当两类型存在显式互斥证据时才硬剪（开放世界，见
    /// [`DenseRetriever::type_compatible`]）；未声明类型的实体仍保守剪掉。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
}
impl SearchConstraints {
    pub fn is_empty(&self) -> bool {
        self.subject.is_none()
            && self.predicate.is_none()
            && self.object.is_none()
            && self.entity_type.is_none()
    }
}
/// 扫描预算：`None` 表示不设上限；预算耗尽时返回当前最优并标记未证明。
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SearchBudget {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_scanned: Option<usize>,
}
impl SearchBudget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn scanning(max_scanned: usize) -> Self {
        Self {
            max_scanned: Some(max_scanned),
        }
    }
}
/// 可审计的检索回执：扫描量、剪枝原因与 top-k 证书。
#[derive(Debug, Clone, Default, Serialize)]
pub struct SearchReceipt {
    pub docs: usize,
    pub query_terms: usize,
    pub candidates: usize,
    /// 被主体/谓词/客体/类型约束排除的文档数。
    pub constraint_pruned: usize,
    /// 实际打分的候选数。
    pub scanned: usize,
    /// 分数大于零并进入候选结果的文档数。
    pub scored: usize,
    /// 因上界低于当前第 k 名而被跳过的候选数（不改变最优性）。
    pub bound_pruned: usize,
    /// 因候选生成截断或预算耗尽而未考虑的文档数。
    pub omitted_by_truncation: usize,
    pub budget_exhausted: bool,
    /// 候选完整且未耗尽预算时为 true：返回结果已证明为全局 top-k。
    pub certified_top_k: bool,
    pub top_k: usize,
    pub returned: usize,
    pub notes: Vec<String>,
}
impl SearchReceipt {
    pub fn render_text(&self) -> String {
        let certificate = if self.certified_top_k {
            "已证明 top-k"
        } else {
            "未证明 top-k（见下方省略原因）"
        };
        let mut text = format!(
            "检索回执：文档 {}，候选 {}，约束剪枝 {}，打分 {}，上界剪枝 {}，返回 {}/{}；{}。",
            self.docs,
            self.candidates,
            self.constraint_pruned,
            self.scanned,
            self.bound_pruned,
            self.returned,
            self.top_k,
            certificate
        );
        for note in &self.notes {
            text.push_str("\n  - ");
            text.push_str(note);
        }
        text
    }
}
/// 一次带约束与预算的证据检索结果。
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceSearch {
    pub hits: Vec<RetrievalHit>,
    pub receipt: SearchReceipt,
}
/// 一篇文档的内部记录
#[derive(Debug, Clone)]
struct DocRecord {
    subject: String,
    predicate: String,
    object: String,
    terms: HashSet<String>,
}
/// 符号稠密检索器
#[derive(Clone)]
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
    /// 每篇文档中最大的词项权重，用于计算词面得分的保守上界。
    doc_max_weights: Vec<f64>,
    entity_docs: HashMap<String, Vec<usize>>,
    /// 精确约束索引（主体/谓词/客体），避免全量扫描。
    by_subject: HashMap<String, Vec<usize>>,
    by_predicate: HashMap<String, Vec<usize>>,
    by_object: HashMap<String, Vec<usize>>,
    /// 实体声明类型（来自 `observe_entity`），用于类型约束与审计。
    entity_types: HashMap<String, String>,
    type_entities: HashMap<String, HashSet<String>>,
    /// RDFS 子类层级（直接边；可达性按有界 BFS 取传递闭包）。
    type_hierarchy: HashMap<String, HashSet<String>>,
    /// 互斥类型对（规范化为左 ≤ 右的有序对）：唯一可证明不兼容的来源。
    type_disjoint: HashSet<(String, String)>,
    /// RDFS 谓词层级：sub → 直接父谓词集合（`sub ⊑ sup`）。
    predicate_hierarchy: HashMap<String, HashSet<String>>,
    /// 谓词层级的反向边：sup → 直接子谓词集合（向下召回用）。
    predicate_children: HashMap<String, HashSet<String>>,
    /// 谓词直接声明的定义域类型（沿父谓词继承的声明在判定时按层级展开）。
    predicate_domain: HashMap<String, HashSet<String>>,
    /// 谓词直接声明的值域类型。
    predicate_range: HashMap<String, HashSet<String>>,
    adjacency: HashMap<String, HashMap<String, f64>>,
    /// 实体名按长度降序的缓存（与 `entity_docs` 的键集合同步，`rebuild` 时刷新）
    seed_candidates: Vec<String>,
    dirty: bool,
    built: bool,
    /// 精确查询缓存：键覆盖查询、k、约束与预算，纪元随索引/层级变化推进。
    search_cache: RevisionLru<EvidenceSearch>,
    cache_epoch: u64,
    cache_hits: u64,
    cache_misses: u64,
}

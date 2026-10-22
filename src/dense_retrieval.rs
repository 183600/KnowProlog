//! 符号主义「稠密检索」：把每条知识当作一篇文档，用纯统计与图游走信号编码成
//! 稀疏高维向量（TF-IDF 词面 × 结构 PPR × 谓词匹配），查询与文档做加权内积。
//!
//! 与神经网络检索器不同，这里没有任何可学习参数：
//! - 词面向量：TF-IDF 倒排索引（df 来自语料统计，无训练）；
//! - 结构向量：在实体-实体图上做随机游走重启（Personalized PageRank）得到
//!   与查询实体结构相关的分布，作为「稠密」语义信号；
//! - 谓词匹配：查询词与谓词的归一化匹配作为精确信号。
//! 三种信号线性融合，输出可解释的证据排序，供生成式回答接地（RAG 的符号版）。
//!
//! [`DenseRetriever::search`] 额外提供精确约束（主体/谓词/客体/类型）与
//! 带证书的 top-k：倒排索引中的候选先按保守上界排序，上界低于当前第 k 名
//! 的文档整段跳过；扫描预算耗尽或候选生成被截断时，[`SearchReceipt`] 会
//! 明确标记 `certified_top_k = false`，保证「省略」始终可审计。

use std::collections::{HashMap, HashSet};

use serde::Serialize;

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
            doc_max_weights: Vec::new(),
            entity_docs: HashMap::new(),
            by_subject: HashMap::new(),
            by_predicate: HashMap::new(),
            by_object: HashMap::new(),
            entity_types: HashMap::new(),
            type_entities: HashMap::new(),
            type_hierarchy: HashMap::new(),
            type_disjoint: HashSet::new(),
            predicate_hierarchy: HashMap::new(),
            predicate_children: HashMap::new(),
            predicate_domain: HashMap::new(),
            predicate_range: HashMap::new(),
            adjacency: HashMap::new(),
            seed_candidates: Vec::new(),
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
        let key = (
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        );
        if !self.seen.insert(key) {
            return;
        }
        let mut terms: HashSet<String> = HashSet::new();
        terms.extend(index_terms(subject));
        terms.extend(index_terms(predicate));
        terms.extend(index_terms(object));
        let id = self.docs.len();
        self.docs.push(DocRecord {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            terms,
        });
        self.entity_docs
            .entry(subject.to_string())
            .or_default()
            .push(id);
        if object != subject {
            self.entity_docs
                .entry(object.to_string())
                .or_default()
                .push(id);
        }
        self.by_subject
            .entry(subject.to_string())
            .or_default()
            .push(id);
        self.by_predicate
            .entry(predicate.to_string())
            .or_default()
            .push(id);
        self.by_object
            .entry(object.to_string())
            .or_default()
            .push(id);
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
        // 子类谓词顺手登记类型层级：检索约束按 RDFS 沿层级展开
        if SUBCLASS_PREDICATES.contains(&predicate) {
            self.observe_subclass(subject, object);
        }
        // 子属性谓词顺手登记谓词层级：谓词约束按 RDFS 沿层级向下展开
        if SUBPROPERTY_PREDICATES.contains(&predicate) {
            self.observe_subproperty(subject, object);
        }
        self.dirty = true;
    }

    /// 撤回一条已观察的三元组；下一次检索前会按需重建倒排与结构索引。
    /// 返回是否确实移除了证据（未观察过的三元组为 false）。
    pub fn remove_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        let key = (
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        );
        if !self.seen.remove(&key) {
            return false;
        }
        let before = self.docs.len();
        self.docs.retain(|doc| {
            doc.subject != subject || doc.predicate != predicate || doc.object != object
        });
        let removed = self.docs.len() != before;
        if removed {
            // 类型证据与实体类型映射同源：撤回类型文档时同步注销声明类型。
            if predicate == "类型"
                && self.entity_types.get(subject).map(String::as_str) == Some(object)
            {
                self.entity_types.remove(subject);
                if let Some(entities) = self.type_entities.get_mut(object) {
                    entities.remove(subject);
                }
            }
            // 子类证据撤回时同步注销层级边（与类型断言注销对称）
            if SUBCLASS_PREDICATES.contains(&predicate) {
                let empty = match self.type_hierarchy.get_mut(subject) {
                    Some(supers) => {
                        supers.remove(object);
                        supers.is_empty()
                    }
                    None => false,
                };
                if empty {
                    self.type_hierarchy.remove(subject);
                }
            }
            // 子属性证据撤回时同步注销谓词层级边
            if SUBPROPERTY_PREDICATES.contains(&predicate) {
                let empty = match self.predicate_hierarchy.get_mut(subject) {
                    Some(supers) => {
                        supers.remove(object);
                        supers.is_empty()
                    }
                    None => false,
                };
                if empty {
                    self.predicate_hierarchy.remove(subject);
                }
                if let Some(children) = self.predicate_children.get_mut(object) {
                    children.remove(subject);
                    if children.is_empty() {
                        self.predicate_children.remove(object);
                    }
                }
            }
            self.dirty = true;
        }
        removed
    }

    /// 观察一个实体的类型与属性（属性逐条转成证据文档）
    pub fn observe_entity(
        &mut self,
        name: &str,
        entity_type: &str,
        properties: &[(String, String)],
    ) {
        let name = name.trim();
        let entity_type = entity_type.trim();
        if name.is_empty() {
            return;
        }
        if !entity_type.is_empty() {
            let previous = self
                .entity_types
                .insert(name.to_string(), entity_type.to_string());
            if previous.as_deref() != Some(entity_type) {
                if let Some(old_type) = previous {
                    if let Some(entities) = self.type_entities.get_mut(&old_type) {
                        entities.remove(name);
                    }
                }
                self.type_entities
                    .entry(entity_type.to_string())
                    .or_default()
                    .insert(name.to_string());
                self.dirty = true;
            }
            self.observe_triple(name, "类型", entity_type);
        }
        for (key, value) in properties {
            self.observe_triple(name, key, value);
        }
    }

    /// 登记一条子类边 `sub ⊑ sup`（幂等）：类型约束据此沿层级展开。
    pub fn observe_subclass(&mut self, sub: &str, sup: &str) {
        let sub = sub.trim();
        let sup = sup.trim();
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return;
        }
        self.type_hierarchy
            .entry(sub.to_string())
            .or_default()
            .insert(sup.to_string());
    }

    /// 登记一对互斥类型（幂等，顺序无关）：互斥是类型约束里唯一的硬剪证据。
    pub fn observe_disjoint(&mut self, left: &str, right: &str) {
        let left = left.trim();
        let right = right.trim();
        if left.is_empty() || right.is_empty() || left == right {
            return;
        }
        let pair = if left <= right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        };
        self.type_disjoint.insert(pair);
    }

    /// 子类可达（含传递闭包）：`sub ⊑* sup`。
    pub fn subclass_reachable(&self, sub: &str, sup: &str) -> bool {
        if sub == sup {
            return true;
        }
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(sub);
        let mut frontier: Vec<&str> = vec![sub];
        for _ in 0..MAX_SUBCLASS_DEPTH {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                let Some(supers) = self.type_hierarchy.get(node) else {
                    continue;
                };
                for parent in supers {
                    if parent == sup {
                        return true;
                    }
                    if visited.insert(parent.as_str()) {
                        next.push(parent.as_str());
                    }
                }
            }
            frontier = next;
        }
        false
    }

    /// 全部严格父类型（传递闭包，有界 BFS，字典序）。
    ///
    /// 与 [`Self::type_pool`] 的向下展开互补：这里给出「已知类型上浮到
    /// 哪些父类型」，供图谱问句在只读匹配里物化类型闭包。
    pub fn type_ancestors(&self, name: &str) -> Vec<String> {
        let name = name.trim();
        if name.is_empty() {
            return Vec::new();
        }
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(name);
        let mut frontier: Vec<&str> = vec![name];
        let mut found: HashSet<String> = HashSet::new();
        for _ in 0..MAX_SUBCLASS_DEPTH {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                let Some(supers) = self.type_hierarchy.get(node) else {
                    continue;
                };
                for parent in supers {
                    if visited.insert(parent.as_str()) {
                        found.insert(parent.clone());
                        next.push(parent.as_str());
                    }
                }
            }
            frontier = next;
        }
        let mut result: Vec<String> = found.into_iter().collect();
        result.sort();
        result
    }

    /// 两类型是否有显式互斥证据（规范化有序对判定）。
    pub fn types_disjoint(&self, left: &str, right: &str) -> bool {
        if left == right {
            return false;
        }
        let pair = if left <= right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        };
        self.type_disjoint.contains(&pair)
    }

    /// 类型相容判定（检索约束的 RDFS 语义，开放世界，与模式闸门
    /// `closure_gate` 同一套判定）：
    /// - 相等 → 相容；
    /// - 存在显式互斥证据 → 不相容（**唯一**硬剪：不凭空发明不相容）；
    /// - 其余（含子类路径、无层级证据）→ 相容。
    /// 子类路径本身不改变布尔结论，但 [`Self::subclass_reachable`] 供上层
    /// 逐条复述「类型为何相容」，并参与类型池展开的审计回执。
    pub fn type_compatible(&self, declared: &str, constraint: &str) -> bool {
        if declared == constraint {
            return true;
        }
        !self.types_disjoint(declared, constraint)
    }

    /// 与约束类型相容的已声明类型（按字典序）：类型池展开用。
    pub fn type_pool(&self, constraint: &str) -> Vec<String> {
        let mut types: Vec<String> = self
            .type_entities
            .keys()
            .filter(|declared| self.type_compatible(declared, constraint))
            .cloned()
            .collect();
        types.sort();
        types
    }

    /// 登记一条子属性边 `sub ⊑ sup`（幂等）：谓词约束据此沿层级向下展开
    /// （子谓词的事实沿边上浮，蕴含父谓词）。
    pub fn observe_subproperty(&mut self, sub: &str, sup: &str) {
        let sub = sub.trim();
        let sup = sup.trim();
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return;
        }
        self.predicate_hierarchy
            .entry(sub.to_string())
            .or_default()
            .insert(sup.to_string());
        self.predicate_children
            .entry(sup.to_string())
            .or_default()
            .insert(sub.to_string());
    }

    /// 登记谓词的定义域类型（幂等）：定义域/值域是 schema 级类型约束，
    /// 与显式互斥证据组合后可硬剪类型讲不通的证据。
    pub fn observe_domain(&mut self, predicate: &str, class: &str) {
        let predicate = predicate.trim();
        let class = class.trim();
        if predicate.is_empty() || class.is_empty() {
            return;
        }
        self.predicate_domain
            .entry(predicate.to_string())
            .or_default()
            .insert(class.to_string());
    }

    /// 登记谓词的值域类型（幂等）。
    pub fn observe_range(&mut self, predicate: &str, class: &str) {
        let predicate = predicate.trim();
        let class = class.trim();
        if predicate.is_empty() || class.is_empty() {
            return;
        }
        self.predicate_range
            .entry(predicate.to_string())
            .or_default()
            .insert(class.to_string());
    }

    /// 谓词的全部严格父谓词（传递闭包，有界 BFS，字典序）。
    pub fn superproperties_reachable(&self, predicate: &str) -> Vec<String> {
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(predicate);
        let mut frontier: Vec<&str> = vec![predicate];
        let mut found: HashSet<String> = HashSet::new();
        for _ in 0..MAX_SUBCLASS_DEPTH {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                let Some(supers) = self.predicate_hierarchy.get(node) else {
                    continue;
                };
                for parent in supers {
                    if visited.insert(parent.as_str()) {
                        found.insert(parent.clone());
                        next.push(parent.as_str());
                    }
                }
            }
            frontier = next;
        }
        let mut out: Vec<String> = found.into_iter().collect();
        out.sort();
        out
    }

    /// 谓词的全部严格子谓词（传递闭包，有界 BFS，字典序）。
    pub fn subproperties_reachable(&self, predicate: &str) -> Vec<String> {
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(predicate);
        let mut frontier: Vec<&str> = vec![predicate];
        let mut found: HashSet<String> = HashSet::new();
        for _ in 0..MAX_SUBCLASS_DEPTH {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                let Some(children) = self.predicate_children.get(node) else {
                    continue;
                };
                for child in children {
                    if visited.insert(child.as_str()) {
                        found.insert(child.clone());
                        next.push(child.as_str());
                    }
                }
            }
            frontier = next;
        }
        let mut out: Vec<String> = found.into_iter().collect();
        out.sort();
        out
    }

    /// 谓词蕴含：`doc` 谓词的事实是否（沿子属性上浮）蕴含 `constraint` 谓词。
    /// 方向是单向的：子事实蕴含父谓词，父事实不蕴含子谓词。
    pub fn predicate_entails(&self, doc: &str, constraint: &str) -> bool {
        doc == constraint
            || self
                .superproperties_reachable(doc)
                .iter()
                .any(|p| p == constraint)
    }

    /// 祖先闭包感知的互斥判定：两侧各沿子类层级上溯，任一对显式互斥即成立。
    /// 与 [`Self::types_disjoint`]（仅显式有序对）相比更强：学生 ⊑ 人物 ⊥ 地点
    /// ⟹ 学生 ⊥ 地点。用于 schema（定义域/值域）硬剪，不改动类型约束的既有语义。
    fn types_proven_disjoint(&self, left: &str, right: &str) -> bool {
        if left == right {
            return false;
        }
        let left_ancestors = self.type_ancestors_or_self(left);
        let right_ancestors = self.type_ancestors_or_self(right);
        for a in &left_ancestors {
            for b in &right_ancestors {
                if self.types_disjoint(a, b) {
                    return true;
                }
            }
        }
        false
    }

    /// 类型自身及其全部祖先（含自身，字典序）。
    fn type_ancestors_or_self(&self, name: &str) -> Vec<String> {
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(name);
        let mut frontier: Vec<&str> = vec![name];
        let mut found: HashSet<String> = HashSet::from([name.to_string()]);
        for _ in 0..MAX_SUBCLASS_DEPTH {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                let Some(supers) = self.type_hierarchy.get(node) else {
                    continue;
                };
                for parent in supers {
                    if visited.insert(parent.as_str()) {
                        found.insert(parent.clone());
                        next.push(parent.as_str());
                    }
                }
            }
            frontier = next;
        }
        let mut out: Vec<String> = found.into_iter().collect();
        out.sort();
        out
    }

    /// 谓词（含其祖先谓词）声明的定义域类型集合（排序去重）。
    fn domain_types_for(&self, predicate: &str) -> Vec<String> {
        let mut classes: HashSet<String> = self
            .predicate_domain
            .get(predicate)
            .cloned()
            .unwrap_or_default();
        for parent in self.superproperties_reachable(predicate) {
            if let Some(declared) = self.predicate_domain.get(&parent) {
                classes.extend(declared.iter().cloned());
            }
        }
        let mut out: Vec<String> = classes.into_iter().collect();
        out.sort();
        out
    }

    /// 谓词（含其祖先谓词）声明的值域类型集合（排序去重）。
    fn range_types_for(&self, predicate: &str) -> Vec<String> {
        let mut classes: HashSet<String> = self
            .predicate_range
            .get(predicate)
            .cloned()
            .unwrap_or_default();
        for parent in self.superproperties_reachable(predicate) {
            if let Some(declared) = self.predicate_range.get(&parent) {
                classes.extend(declared.iter().cloned());
            }
        }
        let mut out: Vec<String> = classes.into_iter().collect();
        out.sort();
        out
    }

    /// schema 硬剪：三元组某侧的声明类型与谓词定义域/值域的**全部**允许类型
    /// 都可证明互斥（祖先闭包感知）时返回 true。开放世界：类型未知、无声明、
    /// 或与任一允许类型不互斥都不剪。
    fn schema_prunes_doc(&self, doc: &DocRecord) -> bool {
        if let Some(declared) = self.entity_types.get(&doc.subject) {
            let domains = self.domain_types_for(&doc.predicate);
            if !domains.is_empty()
                && domains
                    .iter()
                    .all(|allowed| self.types_proven_disjoint(declared, allowed))
            {
                return true;
            }
        }
        if let Some(declared) = self.entity_types.get(&doc.object) {
            let ranges = self.range_types_for(&doc.predicate);
            if !ranges.is_empty()
                && ranges
                    .iter()
                    .all(|allowed| self.types_proven_disjoint(declared, allowed))
            {
                return true;
            }
        }
        false
    }

    /// 一次性观察图谱快照（关系三元组 + 实体类型/属性）。
    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    self.observe_triple(from, rel_type, to)
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    let properties: Vec<(String, String)> = properties
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect();
                    self.observe_entity(name, entity_type, &properties);
                }
            }
        }
    }

    /// 惰性重建倒排索引
    fn rebuild(&mut self) {
        if self.built && !self.dirty {
            return;
        }
        let n = self.docs.len();
        self.df.clear();
        self.postings.clear();
        self.entity_docs.clear();
        self.by_subject.clear();
        self.by_predicate.clear();
        self.by_object.clear();
        self.adjacency.clear();
        self.doc_norms = vec![0.0; n];
        self.doc_max_weights = vec![0.0; n];
        let mut term_freq: Vec<HashMap<&str, f64>> = Vec::with_capacity(n);
        for (id, doc) in self.docs.iter().enumerate() {
            let mut tf: HashMap<&str, f64> = HashMap::new();
            for term in &doc.terms {
                *tf.entry(term.as_str()).or_insert(0.0) += 1.0;
            }
            for term in tf.keys() {
                *self.df.entry((*term).to_string()).or_insert(0) += 1;
            }
            // 全部派生索引在这里从文档正本重建：撤回后不会残留旧文档 id。
            self.entity_docs
                .entry(doc.subject.clone())
                .or_default()
                .push(id);
            if doc.object != doc.subject {
                self.entity_docs
                    .entry(doc.object.clone())
                    .or_default()
                    .push(id);
            }
            self.by_subject
                .entry(doc.subject.clone())
                .or_default()
                .push(id);
            self.by_predicate
                .entry(doc.predicate.clone())
                .or_default()
                .push(id);
            self.by_object
                .entry(doc.object.clone())
                .or_default()
                .push(id);
            if doc.subject != doc.object {
                self.adjacency
                    .entry(doc.subject.clone())
                    .or_default()
                    .entry(doc.object.clone())
                    .and_modify(|weight| *weight += 1.0)
                    .or_insert(1.0);
                self.adjacency
                    .entry(doc.object.clone())
                    .or_default()
                    .entry(doc.subject.clone())
                    .and_modify(|weight| *weight += 1.0)
                    .or_insert(1.0);
            }
            term_freq.push(tf);
        }
        // 第二遍：用 df 计算 idf 与归一化权重
        for (id, tf) in term_freq.iter().enumerate() {
            let mut norm_sq = 0.0;
            let mut max_weight = 0.0_f64;
            for (term, freq) in tf {
                let idf = self.idf(term, n);
                let weight = (1.0 + freq.ln()) * idf;
                if weight > 0.0 {
                    self.postings
                        .entry((*term).to_string())
                        .or_default()
                        .push((id, weight));
                    norm_sq += weight * weight;
                    max_weight = max_weight.max(weight);
                }
            }
            self.doc_norms[id] = norm_sq.sqrt();
            self.doc_max_weights[id] = max_weight;
        }
        // 倒排表按权重降序：段首即当前段的权重上界，便于后续分段剪枝与确定性复现。
        for postings in self.postings.values_mut() {
            postings.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        // 实体名按键长降序缓存，供 `seeds_cached` 的最长优先匹配复用
        let mut names: Vec<String> = self.entity_docs.keys().cloned().collect();
        names.sort_by(|a, b| {
            b.chars()
                .count()
                .cmp(&a.chars().count())
                .then_with(|| a.cmp(b))
        });
        self.seed_candidates = names;
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
        names.sort_by(|a, b| {
            b.chars()
                .count()
                .cmp(&a.chars().count())
                .then_with(|| a.cmp(b))
        });
        Self::seeds_from_names(&names, query)
    }

    /// 复用重建时缓存的实体名（`rebuild` 后与键集合一致）
    fn seeds_cached(&self, query: &str) -> Vec<String> {
        if self.seed_candidates.len() != self.entity_docs.len() {
            return self.seeds(query);
        }
        let names: Vec<&String> = self.seed_candidates.iter().collect();
        Self::seeds_from_names(&names, query)
    }

    fn seeds_from_names(names: &[&String], query: &str) -> Vec<String> {
        let mut seeds = Vec::new();
        for name in names {
            let name: &String = name;
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
    fn query_vector(&self, tokens: &[String], seeds: &[String]) -> HashMap<String, f64> {
        let total = self.docs.len().max(1);
        let mut weights: HashMap<String, f64> = HashMap::new();
        for token in tokens {
            let idf = if self.df.contains_key(token) {
                self.idf(token, total)
            } else {
                (1.0 + total as f64).ln()
            };
            *weights.entry(token.clone()).or_insert(0.0) += idf;
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

    /// 两个概率分布之间的 L1 距离（用于判断幂迭代是否收敛）
    fn distribution_delta(left: &HashMap<String, f64>, right: &HashMap<String, f64>) -> f64 {
        let mut delta: f64 = left
            .iter()
            .map(|(node, mass)| (mass - right.get(node).copied().unwrap_or(0.0)).abs())
            .sum();
        for (node, mass) in right {
            if !left.contains_key(node) {
                delta += mass.abs();
            }
        }
        delta
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
        // 幂迭代至少执行 `iterations` 轮；实体图常为（近）二分图，固定轮数可能恰好
        // 停在奇偶振荡的反相上，使距离 2 的节点压过直接邻居。达到最少轮数后继续
        // 迭代到相邻两轮分布收敛为止，并以轮数上限兜底（防止高阻尼下振荡不收敛）。
        let min_iterations = self.iterations.max(1);
        let max_iterations = min_iterations.saturating_mul(8).max(64).min(1024);
        for iteration in 0..max_iterations {
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
                        // 悬挂节点：质量留在原地
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
            let delta = Self::distribution_delta(&current, &next);
            current = next;
            if iteration + 1 >= min_iterations && delta <= 1e-9 {
                break;
            }
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

    fn predicate_matches(&self, id: usize, query: &str, query_tokens: &HashSet<String>) -> bool {
        let doc = &self.docs[id];
        if query_tokens.contains(&doc.predicate) || query.contains(doc.predicate.as_str()) {
            return true;
        }
        // 查询提到父谓词时，子谓词证据按蕴含关系同样算命中
        self.superproperties_reachable(&doc.predicate)
            .iter()
            .any(|parent| query_tokens.contains(parent) || query.contains(parent.as_str()))
    }

    fn score_doc(
        &self,
        id: usize,
        dot_terms: &[(&String, f64)],
        query_tokens: &HashSet<String>,
        query: &str,
        query_norm: f64,
        structural: &HashMap<String, f64>,
    ) -> Option<RetrievalHit> {
        let doc = &self.docs[id];
        let mut dot = 0.0;
        for (term, weighted) in dot_terms {
            if doc.terms.contains(term.as_str()) {
                dot += weighted;
            }
        }
        let lexical = if self.doc_norms[id] > 0.0 {
            dot / (self.doc_norms[id] * query_norm)
        } else {
            0.0
        };
        let structural_score = (structural.get(&doc.subject).copied().unwrap_or(0.0)
            + structural.get(&doc.object).copied().unwrap_or(0.0))
        .min(1.0);
        let predicate_hit = if self.predicate_matches(id, query, query_tokens) {
            1.0
        } else {
            0.0
        };
        let score = self.lexical_weight * lexical
            + self.structural_weight * structural_score
            + self.predicate_weight * predicate_hit;
        if score <= 0.0 {
            return None;
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
        Some(RetrievalHit {
            subject: doc.subject.clone(),
            predicate: doc.predicate.clone(),
            object: doc.object.clone(),
            score,
            lexical,
            structural: structural_score,
            predicate_match: predicate_hit,
            reasons,
        })
    }

    /// 单篇文档的保守分数上界：词面用最大词项权重 × 查询总权重，
    /// 结构用归一化 PPR 的全局最大值，谓词命中可直接精确判定。
    fn doc_upper_bound(
        &self,
        id: usize,
        dot_weight_sum: f64,
        query_norm: f64,
        structural: &HashMap<String, f64>,
        max_structural: f64,
        query: &str,
        query_tokens: &HashSet<String>,
    ) -> f64 {
        let doc = &self.docs[id];
        let lexical_ub = if self.doc_norms[id] > 0.0 && query_norm > 0.0 {
            (self.doc_max_weights[id] * dot_weight_sum / (self.doc_norms[id] * query_norm)).min(1.0)
        } else {
            0.0
        };
        let structural_ub = match (
            structural.get(&doc.subject).copied(),
            structural.get(&doc.object).copied(),
        ) {
            (Some(left), Some(right)) => (left + right).min(1.0),
            (Some(one), None) | (None, Some(one)) => (one + max_structural).min(1.0),
            (None, None) => (2.0 * max_structural).min(1.0),
        };
        let predicate_ub = if self.predicate_matches(id, query, query_tokens) {
            1.0
        } else {
            0.0
        };
        self.lexical_weight * lexical_ub
            + self.structural_weight * structural_ub
            + self.predicate_weight * predicate_ub
    }

    fn compare_hits(a: &RetrievalHit, b: &RetrievalHit) -> std::cmp::Ordering {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.subject.cmp(&b.subject))
            .then_with(|| a.predicate.cmp(&b.predicate))
            .then_with(|| a.object.cmp(&b.object))
    }

    fn doc_matches_constraints(&self, id: usize, constraints: &SearchConstraints) -> bool {
        let doc = &self.docs[id];
        if let Some(subject) = &constraints.subject {
            if doc.subject != *subject {
                return false;
            }
        }
        if let Some(predicate) = &constraints.predicate {
            // RDFS 谓词语义：子谓词事实上浮蕴含父谓词（单向，父不蕴含子）。
            if !self.predicate_entails(&doc.predicate, predicate) {
                return false;
            }
        }
        if let Some(object) = &constraints.object {
            if doc.object != *object {
                return false;
            }
        }
        // schema 硬剪：声明类型与定义域/值域全部允许类型互斥的证据不入结果。
        if self.schema_prunes_doc(doc) {
            return false;
        }
        if let Some(entity_type) = &constraints.entity_type {
            // RDFS 语义：声明类型与约束类型相容（子类路径/开放世界，互斥硬剪）；
            // 未声明类型的实体仍保守剪掉。
            let subject_matches = self
                .entity_types
                .get(&doc.subject)
                .map(|declared| self.type_compatible(declared, entity_type))
                .unwrap_or(false);
            let object_matches = self
                .entity_types
                .get(&doc.object)
                .map(|declared| self.type_compatible(declared, entity_type))
                .unwrap_or(false);
            if !subject_matches && !object_matches {
                return false;
            }
        }
        true
    }

    /// 用最窄的约束倒排表做交集，避免为了一个主体/谓词扫描全部文档。
    fn constrained_candidates(&self, constraints: &SearchConstraints) -> Vec<usize> {
        let mut base: Option<Vec<usize>> = None;
        if let Some(subject) = &constraints.subject {
            let ids = match self.by_subject.get(subject) {
                Some(ids) => ids,
                None => return Vec::new(),
            };
            if base
                .as_ref()
                .map_or(true, |current| ids.len() < current.len())
            {
                base = Some(ids.clone());
            }
        }
        if let Some(predicate) = &constraints.predicate {
            // 谓词倒排按 RDFS 向下展开：子谓词的事实上浮蕴含父谓词，一并入池。
            let mut ids: Vec<usize> = self
                .by_predicate
                .get(predicate)
                .cloned()
                .unwrap_or_default();
            for sub in self.subproperties_reachable(predicate) {
                if let Some(sub_ids) = self.by_predicate.get(&sub) {
                    ids.extend(sub_ids.iter().copied());
                }
            }
            ids.sort_unstable();
            ids.dedup();
            if base
                .as_ref()
                .map_or(true, |current| ids.len() < current.len())
            {
                base = Some(ids);
            }
        }
        if let Some(object) = &constraints.object {
            let ids = match self.by_object.get(object) {
                Some(ids) => ids,
                None => return Vec::new(),
            };
            if base
                .as_ref()
                .map_or(true, |current| ids.len() < current.len())
            {
                base = Some(ids.clone());
            }
        }
        let pool = match base {
            Some(ids) => ids,
            None => {
                let Some(entity_type) = &constraints.entity_type else {
                    return (0..self.docs.len()).collect();
                };
                // 类型池按相容类型展开：子类/无互斥证据的类型一并入池（开放世界）；
                // 互斥类型不入池（唯一硬剪）。
                let compatible = self.type_pool(entity_type);
                let mut seen = HashSet::new();
                let mut ids = Vec::new();
                for declared in &compatible {
                    let Some(entities) = self.type_entities.get(declared) else {
                        continue;
                    };
                    for entity in entities {
                        if let Some(entity_docs) = self.entity_docs.get(entity) {
                            for id in entity_docs {
                                if seen.insert(*id) {
                                    ids.push(*id);
                                }
                            }
                        }
                    }
                }
                ids
            }
        };
        pool.into_iter()
            .filter(|id| self.doc_matches_constraints(*id, constraints))
            .collect()
    }

    /// 带约束、上界剪枝与预算的检索：返回排序证据和可审计回执。
    ///
    /// 正确性契约：上界只用于跳过不可能进入前 k 的文档；预算耗尽或候选生成
    /// 截断时回执显式标记 `certified_top_k = false`，绝不把近似结果冒充精确结果。
    pub fn search(
        &mut self,
        query: &str,
        k: usize,
        constraints: &SearchConstraints,
        budget: SearchBudget,
    ) -> EvidenceSearch {
        self.rebuild();
        let k = k.clamp(1, 50);
        let query = query.trim();
        let mut receipt = SearchReceipt {
            docs: self.docs.len(),
            top_k: k,
            ..SearchReceipt::default()
        };
        if self.docs.is_empty() || (query.is_empty() && constraints.is_empty()) {
            receipt.certified_top_k = true;
            return EvidenceSearch {
                hits: Vec::new(),
                receipt,
            };
        }

        let constrained = if constraints.is_empty() {
            None
        } else {
            let ids = self.constrained_candidates(constraints);
            receipt.constraint_pruned = self.docs.len().saturating_sub(ids.len());
            Some(ids)
        };

        // 类型约束的 RDFS 展开写进回执：展开过程可审计（互斥已排除）
        if let Some(entity_type) = &constraints.entity_type {
            let pool = self.type_pool(entity_type);
            if pool.len() > 1 {
                receipt.notes.push(format!(
                    "类型约束「{}」按 RDFS 展开为相容类型池（{}）；互斥类型已排除。",
                    entity_type,
                    pool.join("、")
                ));
            }
        }

        // 谓词约束的子属性展开写进回执：子谓词事实蕴含父谓词（单向）
        if let Some(predicate) = &constraints.predicate {
            let subs = self.subproperties_reachable(predicate);
            if !subs.is_empty() {
                receipt.notes.push(format!(
                    "谓词约束「{}」按 RDFS 向下展开，纳入子谓词证据（{}）。",
                    predicate,
                    subs.join("、")
                ));
            }
        }

        if query.is_empty() {
            let mut ids = constrained.unwrap_or_default();
            ids.sort_by(|left, right| {
                let left = &self.docs[*left];
                let right = &self.docs[*right];
                left.subject
                    .cmp(&right.subject)
                    .then_with(|| left.predicate.cmp(&right.predicate))
                    .then_with(|| left.object.cmp(&right.object))
            });
            receipt.candidates = ids.len();
            receipt.scanned = ids.len().min(k);
            let hits: Vec<RetrievalHit> = ids
                .into_iter()
                .take(k)
                .map(|id| {
                    let doc = &self.docs[id];
                    RetrievalHit {
                        subject: doc.subject.clone(),
                        predicate: doc.predicate.clone(),
                        object: doc.object.clone(),
                        score: 0.0,
                        lexical: 0.0,
                        structural: 0.0,
                        predicate_match: 0.0,
                        reasons: vec!["约束精确匹配".to_string()],
                    }
                })
                .collect();
            receipt.returned = hits.len();
            receipt.certified_top_k = true;
            receipt.notes.push(format!(
                "空查询按约束精确匹配，按主体/谓词/客体字典序返回 {} 条。",
                hits.len()
            ));
            return EvidenceSearch { hits, receipt };
        }

        let seeds = self.seeds_cached(query);
        let tokens = tokenize(query);
        let query_vector = self.query_vector(&tokens, &seeds);
        if query_vector.is_empty() && seeds.is_empty() {
            receipt.certified_top_k = true;
            return EvidenceSearch {
                hits: Vec::new(),
                receipt,
            };
        }
        receipt.query_terms = query_vector.len();
        let query_norm = Self::query_norm(&query_vector).max(1e-9);
        let structural = self.structural_scores(&seeds);
        let query_tokens: HashSet<String> = tokens.into_iter().collect();
        let total_docs = self.docs.len();
        let dot_terms: Vec<(&String, f64)> = query_vector
            .iter()
            .map(|(term, weight)| {
                let idf = if self.df.contains_key(term) {
                    self.idf(term, total_docs)
                } else {
                    (1.0 + total_docs as f64).ln()
                };
                (term, weight * idf)
            })
            .collect();

        let mut candidate_ids: HashSet<usize> = HashSet::new();
        let mut omitted_by_truncation = 0usize;
        if let Some(ids) = constrained {
            candidate_ids.extend(ids);
        } else {
            for term in query_vector.keys() {
                if let Some(postings) = self.postings.get(term) {
                    for (id, _) in postings {
                        candidate_ids.insert(*id);
                    }
                }
            }
            for seed in &seeds {
                if let Some(ids) = self.entity_docs.get(seed) {
                    omitted_by_truncation += ids.len().saturating_sub(400);
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
            let mut structural_entities = 0usize;
            for (entity, score) in structural_ranked {
                if *score < 0.05 {
                    break;
                }
                if structural_entities >= 20 {
                    if let Some(ids) = self.entity_docs.get(entity) {
                        omitted_by_truncation += ids.len();
                    }
                    continue;
                }
                structural_entities += 1;
                if let Some(ids) = self.entity_docs.get(entity) {
                    omitted_by_truncation += ids.len().saturating_sub(50);
                    for id in ids.iter().take(50) {
                        candidate_ids.insert(*id);
                    }
                }
            }
            // 谓词词面直接出现在查询里时，把该谓词的全部证据纳入候选；
            // 避免「谓词命中但分词未命中」的文档被倒排表漏掉。
            // 查询提到父谓词时，子谓词证据按蕴含关系一并纳入。
            for (predicate, ids) in &self.by_predicate {
                let mentioned = query.contains(predicate.as_str())
                    || self
                        .superproperties_reachable(predicate)
                        .iter()
                        .any(|parent| query.contains(parent.as_str()));
                if mentioned {
                    candidate_ids.extend(ids.iter().copied());
                }
            }
        }

        receipt.candidates = candidate_ids.len();
        receipt.omitted_by_truncation = omitted_by_truncation;
        if candidate_ids.is_empty() {
            receipt.certified_top_k = omitted_by_truncation == 0;
            if !receipt.certified_top_k {
                receipt.notes.push(format!(
                    "候选生成截断了 {} 条证据，未证明 top-k 最优。",
                    omitted_by_truncation
                ));
            }
            return EvidenceSearch {
                hits: Vec::new(),
                receipt,
            };
        }

        let max_structural = structural.values().cloned().fold(0.0_f64, f64::max);
        let dot_weight_sum: f64 = dot_terms.iter().map(|(_, weight)| *weight).sum();
        let mut ranked: Vec<(usize, f64)> = candidate_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    self.doc_upper_bound(
                        *id,
                        dot_weight_sum,
                        query_norm,
                        &structural,
                        max_structural,
                        query,
                        &query_tokens,
                    ),
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let max_scanned = budget.max_scanned.unwrap_or(usize::MAX);
        let mut hits: Vec<RetrievalHit> = Vec::new();
        let mut threshold = f64::NEG_INFINITY;
        let mut exhausted = false;
        for (index, (id, upper_bound)) in ranked.iter().enumerate() {
            if hits.len() >= k && *upper_bound < threshold {
                receipt.bound_pruned += 1;
                continue;
            }
            if receipt.scanned >= max_scanned {
                receipt.omitted_by_truncation += ranked.len() - index;
                exhausted = true;
                break;
            }
            receipt.scanned += 1;
            if let Some(hit) = self.score_doc(
                *id,
                &dot_terms,
                &query_tokens,
                query,
                query_norm,
                &structural,
            ) {
                receipt.scored += 1;
                hits.push(hit);
                hits.sort_by(Self::compare_hits);
                if hits.len() > k {
                    hits.truncate(k);
                }
                if hits.len() == k {
                    threshold = hits[k - 1].score;
                }
            }
        }

        receipt.returned = hits.len();
        receipt.budget_exhausted = exhausted;
        receipt.certified_top_k = !exhausted && receipt.omitted_by_truncation == 0;
        if receipt.bound_pruned > 0 {
            receipt.notes.push(format!(
                "上界剪枝跳过 {} 条候选（其上界低于当前第 {} 名，不影响最优性）。",
                receipt.bound_pruned, k
            ));
        }
        if exhausted {
            receipt.notes.push(format!(
                "扫描预算 {} 耗尽，返回当前最优；未证明 top-k 最优。",
                max_scanned
            ));
        }
        if receipt.omitted_by_truncation > 0 {
            receipt.notes.push(format!(
                "候选生成截断 {} 条证据，未证明 top-k 最优。",
                receipt.omitted_by_truncation
            ));
        }
        if receipt.certified_top_k {
            receipt
                .notes
                .push("候选完整且未被预算截断：top-k 已证明。".to_string());
        }
        EvidenceSearch { hits, receipt }
    }

    /// 检索与查询最相关的证据（融合三种符号信号），结果按分数降序
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<RetrievalHit> {
        self.search(
            query,
            k,
            &SearchConstraints::default(),
            SearchBudget::default(),
        )
        .hits
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

/// 分词并构造索引词项：结巴切词 + 中文二元组
pub fn tokenize(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut tokens: Vec<String> = Vec::new();
    for word in shared_jieba().cut(text, false) {
        let word = word.trim();
        if word.is_empty() || is_punctuation(word) || stopwords().contains(word) {
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
    let chars: Vec<char> = compact
        .chars()
        .filter(|ch| !is_punctuation_char(*ch))
        .collect();
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
        retriever.observe_entity(
            "张三",
            "人",
            &[
                ("年龄".to_string(), "30".to_string()),
                ("职业".to_string(), "演员".to_string()),
                ("国籍".to_string(), "中国".to_string()),
            ],
        );
        retriever.observe_entity(
            "李四",
            "人",
            &[
                ("年龄".to_string(), "28".to_string()),
                ("职业".to_string(), "导演".to_string()),
                ("国籍".to_string(), "德国".to_string()),
            ],
        );
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

    fn exhaustive_hits(retriever: &DenseRetriever, query: &str, k: usize) -> Vec<RetrievalHit> {
        let seeds = retriever.seeds(query);
        let tokens = tokenize(query);
        let query_vector = retriever.query_vector(&tokens, &seeds);
        let query_norm = DenseRetriever::query_norm(&query_vector).max(1e-9);
        let structural = retriever.structural_scores(&seeds);
        let query_tokens: HashSet<String> = tokens.into_iter().collect();
        let total = retriever.docs.len();
        let dot_terms: Vec<(&String, f64)> = query_vector
            .iter()
            .map(|(term, weight)| {
                let idf = if retriever.df.contains_key(term) {
                    retriever.idf(term, total)
                } else {
                    (1.0 + total as f64).ln()
                };
                (term, weight * idf)
            })
            .collect();
        let mut hits: Vec<RetrievalHit> = (0..retriever.docs.len())
            .filter_map(|id| {
                retriever.score_doc(
                    id,
                    &dot_terms,
                    &query_tokens,
                    query,
                    query_norm,
                    &structural,
                )
            })
            .collect();
        hits.sort_by(DenseRetriever::compare_hits);
        hits.truncate(k);
        hits
    }

    #[test]
    fn bounded_search_matches_exhaustive_scoring() {
        let mut retriever = build_retriever();
        retriever.rebuild();
        let search = retriever.search(
            "张三的年龄是多少",
            5,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        let exhaustive = exhaustive_hits(&retriever, "张三的年龄是多少", 5);
        let keys = |hits: &[RetrievalHit]| {
            hits.iter()
                .map(|hit| {
                    (
                        hit.subject.clone(),
                        hit.predicate.clone(),
                        hit.object.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&search.hits), keys(&exhaustive));
        assert!(search
            .hits
            .iter()
            .zip(exhaustive.iter())
            .all(|(left, right)| (left.score - right.score).abs() < 1e-9));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn upper_bound_dominates_exact_score() {
        let mut retriever = build_retriever();
        retriever.rebuild();
        let query = "张三的年龄是多少";
        let seeds = retriever.seeds(query);
        let tokens = tokenize(query);
        let query_vector = retriever.query_vector(&tokens, &seeds);
        let query_norm = DenseRetriever::query_norm(&query_vector).max(1e-9);
        let structural = retriever.structural_scores(&seeds);
        let max_structural = structural.values().cloned().fold(0.0_f64, f64::max);
        let query_tokens: HashSet<String> = tokens.into_iter().collect();
        let total = retriever.docs.len();
        let dot_terms: Vec<(&String, f64)> = query_vector
            .iter()
            .map(|(term, weight)| {
                let idf = if retriever.df.contains_key(term) {
                    retriever.idf(term, total)
                } else {
                    (1.0 + total as f64).ln()
                };
                (term, weight * idf)
            })
            .collect();
        let dot_weight_sum: f64 = dot_terms.iter().map(|(_, weight)| *weight).sum();
        for id in 0..retriever.docs.len() {
            let upper = retriever.doc_upper_bound(
                id,
                dot_weight_sum,
                query_norm,
                &structural,
                max_structural,
                query,
                &query_tokens,
            );
            if let Some(hit) = retriever.score_doc(
                id,
                &dot_terms,
                &query_tokens,
                query,
                query_norm,
                &structural,
            ) {
                assert!(
                    hit.score <= upper + 1e-9,
                    "上界 {upper} 低于精确分数 {}（文档 {id}）",
                    hit.score
                );
            }
        }
    }

    #[test]
    fn constraints_filter_before_scoring_and_certify() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            subject: Some("张三".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("年龄", 5, &constraints, SearchBudget::unlimited());
        assert!(!search.hits.is_empty());
        assert!(search.hits.iter().all(|hit| hit.subject == "张三"));
        assert!(search.receipt.constraint_pruned > 0);
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn type_constraint_uses_declared_entity_types() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            predicate: Some("职业".to_string()),
            entity_type: Some("人".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(subjects.contains("张三"));
        assert!(subjects.contains("李四"));
        assert!(!subjects.contains("王五"));
        assert!(search.hits.iter().all(|hit| hit.predicate == "职业"));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn empty_query_returns_constraint_lookup() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            predicate: Some("老师".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert_eq!(search.hits.len(), 2);
        assert!(search.hits.iter().all(|hit| hit.predicate == "老师"));
        assert!(search
            .hits
            .iter()
            .all(|hit| hit.reasons == vec!["约束精确匹配"]));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn removal_drops_evidence_and_invalidates_index() {
        let mut retriever = build_retriever();
        assert!(retriever.remove_triple("张三", "老师", "王五"));
        assert!(!retriever.remove_triple("张三", "老师", "王五"));
        let removed = retriever.search(
            "张三 老师 王五",
            10,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(removed.hits.iter().all(|hit| !(hit.subject == "张三"
            && hit.predicate == "老师"
            && hit.object == "王五")));

        retriever.observe_triple("张三", "老师", "王五");
        let restored = retriever.search(
            "张三 老师 王五",
            10,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(restored
            .hits
            .iter()
            .any(|hit| hit.subject == "张三" && hit.predicate == "老师" && hit.object == "王五"));
    }

    #[test]
    fn removing_type_evidence_unregisters_declared_type() {
        let mut retriever = build_retriever();
        assert!(retriever.remove_triple("张三", "类型", "人"));
        let constraints = SearchConstraints {
            predicate: Some("职业".to_string()),
            entity_type: Some("人".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(!subjects.contains("张三"));
        assert!(subjects.contains("李四"));
    }

    #[test]
    fn budget_exhaustion_is_explicitly_uncertified() {
        let mut retriever = build_retriever();
        let search = retriever.search(
            "张三",
            3,
            &SearchConstraints::default(),
            SearchBudget::scanning(1),
        );
        assert!(search.receipt.budget_exhausted);
        assert!(!search.receipt.certified_top_k);
        assert!(search.receipt.scanned <= 1);
        assert!(search
            .receipt
            .notes
            .iter()
            .any(|note| note.contains("预算")));
    }

    fn rdfs_retriever() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(8, 0.85, 12);
        retriever.observe_entity("张三", "学生", &[("专业".to_string(), "哲学".to_string())]);
        retriever.observe_entity(
            "李四",
            "教师",
            &[("专业".to_string(), "逻辑学".to_string())],
        );
        retriever.observe_entity("王五", "地点", &[("名称".to_string(), "长安".to_string())]);
        retriever.observe_subclass("学生", "人物");
        retriever.observe_subclass("教师", "人物");
        retriever.observe_disjoint("地点", "人物");
        retriever
    }

    #[test]
    fn type_pool_expands_to_subclasses_and_excludes_disjoint() {
        let retriever = rdfs_retriever();
        let pool = retriever.type_pool("人物");
        assert_eq!(pool, vec!["学生".to_string(), "教师".to_string()]);
        assert!(
            retriever.subclass_reachable("学生", "人物"),
            "子类可达性应成立"
        );
        assert!(retriever.type_compatible("学生", "人物"), "子类与父类相容");
        assert!(retriever.type_compatible("学生", "学生"), "自反相容");
        assert!(
            !retriever.type_compatible("地点", "人物"),
            "互斥证据是唯一硬剪"
        );
    }

    #[test]
    fn type_constraint_matches_subclass_declared_types() {
        let mut retriever = rdfs_retriever();
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(
            subjects.contains("张三"),
            "学生 ⊑ 人物 应命中：{:?}",
            subjects
        );
        assert!(
            subjects.contains("李四"),
            "教师 ⊑ 人物 应命中：{:?}",
            subjects
        );
        assert!(
            !subjects.contains("王五"),
            "互斥类型（地点）必须被排除：{:?}",
            subjects
        );
        assert!(
            search
                .receipt
                .notes
                .iter()
                .any(|note| note.contains("相容类型池")),
            "类型池展开应写进回执：{:?}",
            search.receipt.notes
        );
    }

    #[test]
    fn unknown_declared_types_are_still_pruned() {
        let mut retriever = rdfs_retriever();
        // 主客体都未声明类型：不臆造类型，保守剪掉
        retriever.observe_triple("赵六", "同事", "钱七");
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search
                .hits
                .iter()
                .all(|hit| hit.subject != "赵六" && hit.object != "赵六"),
            "未声明类型不得混入类型约束结果"
        );
        // 一端有相容类型即可命中（主体或客体任一匹配）
        retriever.observe_entity("钱七", "学生", &[]);
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search
                .hits
                .iter()
                .any(|hit| hit.subject == "赵六" && hit.object == "钱七"),
            "客体类型相容（学生 ⊑ 人物）时应命中"
        );
    }

    #[test]
    fn subclass_predicates_from_triples_feed_hierarchy() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_entity("苏格拉底", "人", &[]);
        // 图谱里的子类三元组应被识别为层级边
        retriever.observe_triple("人", "是一种", "动物");
        assert!(retriever.subclass_reachable("人", "动物"));
        let constraints = SearchConstraints {
            entity_type: Some("动物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 5, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "苏格拉底"),
            "沿图谱子类边应命中：{:?}",
            search.hits
        );
    }

    #[test]
    fn removing_subclass_triple_drops_hierarchy_edge() {
        // 用 observe_triple 建立的层级边随三元组撤回（层级与证据同源）
        let mut graph = DenseRetriever::new(5, 0.85, 12);
        graph.observe_triple("学生", "是一种", "人物");
        assert!(graph.subclass_reachable("学生", "人物"));
        assert!(graph.remove_triple("学生", "是一种", "人物"));
        assert!(!graph.subclass_reachable("学生", "人物"));
    }

    #[test]
    fn unrelated_types_are_open_world_compatible() {
        let retriever = rdfs_retriever();
        // 无层级也无互斥证据：不凭空发明不相容（开放世界）
        assert!(retriever.type_compatible("学生", "公司"));
        assert!(!retriever.types_disjoint("学生", "公司"));
        assert!(retriever.types_disjoint("人物", "地点"), "顺序无关判定");
        assert!(retriever.types_disjoint("地点", "人物"));
    }

    #[test]
    fn predicate_constraint_expands_to_subproperties() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_subproperty("师承", "师傅");
        retriever.observe_triple("张三", "师承", "李四");
        retriever.observe_triple("王五", "师傅", "赵六");
        assert!(retriever.predicate_entails("师承", "师傅"));
        assert!(!retriever.predicate_entails("师傅", "师承"), "父不蕴含子");

        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let pairs: HashSet<(&str, &str)> = search
            .hits
            .iter()
            .map(|hit| (hit.subject.as_str(), hit.predicate.as_str()))
            .collect();
        assert!(
            pairs.contains(&("张三", "师承")),
            "子谓词事实应被父谓词约束召回：{:?}",
            pairs
        );
        assert!(pairs.contains(&("王五", "师傅")), "直接事实照常召回");
        assert!(
            search
                .receipt
                .notes
                .iter()
                .any(|note| note.contains("子谓词")),
            "谓词展开应写进回执：{:?}",
            search.receipt.notes
        );

        // 单向性：子谓词约束不召回父谓词事实
        let constraints = SearchConstraints {
            predicate: Some("师承".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().all(|hit| hit.subject != "王五"),
            "父谓词事实不得混入子谓词约束：{:?}",
            search.hits
        );
    }

    #[test]
    fn schema_disjoint_types_prune_evidence() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_subclass("学生", "人物");
        retriever.observe_disjoint("人物", "地点");
        retriever.observe_domain("师傅", "人物");
        retriever.observe_range("师傅", "人物");
        // 主体类型（地点）与定义域（人物）可证明互斥：硬剪
        retriever.observe_entity("故宫", "地点", &[]);
        retriever.observe_triple("故宫", "师傅", "导游");
        // 主体类型（学生）沿子类相容：保留
        retriever.observe_entity("张三", "学生", &[]);
        retriever.observe_triple("张三", "师傅", "李四");
        // 主体类型未知：开放世界，不剪
        retriever.observe_triple("无名氏", "师傅", "某人");

        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(
            !subjects.contains("故宫"),
            "定义域互斥的证据必须被硬剪：{:?}",
            subjects
        );
        assert!(
            subjects.contains("张三"),
            "子类相容证据保留：{:?}",
            subjects
        );
        assert!(
            subjects.contains("无名氏"),
            "未知类型不凭空剪（开放世界）：{:?}",
            subjects
        );
    }

    #[test]
    fn subproperty_triples_feed_and_unfeed_hierarchy() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_triple("师承", "subPropertyOf", "师傅");
        assert!(retriever.predicate_entails("师承", "师傅"));
        assert!(retriever.remove_triple("师承", "subPropertyOf", "师傅"));
        assert!(!retriever.predicate_entails("师承", "师傅"));
    }
}

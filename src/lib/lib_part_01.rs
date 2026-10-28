use nlp::{normalize_words, pattern_from_strings, preprocess, PrologRule, TaggedWord};
use prolog_engine::PrologFact;
// RDF命名空间（与 identity::DEFAULT_BASE_NS 单一正本）
const BASE_NS: &str = crate::identity::DEFAULT_BASE_NS;
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const NAME_PREDICATE_IRI: &str = "http://example.org/kg/name";
/// 转义 SPARQL 字符串字面量中的特殊字符（引号/反斜杠/换行等），
/// 避免名称里的特殊字符破坏查询语法或造成查询注入
fn sparql_literal(text: &str) -> std::borrow::Cow<'_, str> {
    if !text
        .as_bytes()
        .iter()
        .any(|b| matches!(b, b'\\' | b'"' | b'\n' | b'\r' | b'\t'))
    {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut escaped = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    std::borrow::Cow::Owned(escaped)
}
/// 把 IRI 的 kg 本地名（或实体路径名）计入增量足迹；外部 IRI 原样计入。
fn touch_iri_name(delta: &mut DeltaFootprint, iri: &str) {
    if let Some(name) = crate::identity::entity_local_name(BASE_NS, iri)
        .or_else(|| crate::identity::local_name(BASE_NS, iri))
    {
        delta.touch_name(name);
    } else {
        delta.touch_name(iri);
    }
}

/// 从 RDF 四元组提取「这次写入可能碰到哪些名称/谓词」。
///
/// 这是保守足迹：名称字面量、IRI 本地名与完整的非 kg IRI 都计入，宁可多失效
/// 一条缓存，也不让新事实绕过依赖复核。
fn footprint_from_quads(quads: &[Quad]) -> DeltaFootprint {
    let mut delta = DeltaFootprint::new(0);
    for quad in quads {
        if let Subject::NamedNode(node) = &quad.subject {
            touch_iri_name(&mut delta, node.as_str());
        }
        let predicate_iri = quad.predicate.as_str();
        delta.touch_predicate(
            crate::identity::local_name(BASE_NS, predicate_iri)
                .unwrap_or_else(|| predicate_iri.to_string()),
        );
        match &quad.object {
            Term::NamedNode(node) => touch_iri_name(&mut delta, node.as_str()),
            Term::Literal(literal) => {
                delta.touch_name(literal.value());
            }
            _ => {}
        }
    }
    delta
}

/// 从导入回执的实际动作提取增量足迹（只含成功写入的三元组）。
fn footprint_from_actions(actions: &[predicate_registry::ImportAction]) -> DeltaFootprint {
    let mut delta = DeltaFootprint::new(0);
    for action in actions {
        match action {
            predicate_registry::ImportAction::Named {
                subject,
                predicate,
                object,
            } => {
                touch_iri_name(&mut delta, subject);
                touch_iri_name(&mut delta, object);
                delta.touch_predicate(
                    crate::identity::local_name(BASE_NS, predicate)
                        .unwrap_or_else(|| predicate.clone()),
                );
            }
            predicate_registry::ImportAction::Literal {
                subject,
                predicate,
                value,
            } => {
                touch_iri_name(&mut delta, subject);
                delta.touch_name(value.clone());
                delta.touch_predicate(
                    crate::identity::local_name(BASE_NS, predicate)
                        .unwrap_or_else(|| predicate.clone()),
                );
            }
        }
    }
    delta
}

/// 实体在图中已声明的类型（rdf:type 的本地名集合，用于定义域/值域校验）。
fn entity_type_labels(
    store: &Store,
    name: &str,
) -> Result<BTreeSet<String>, Box<dyn std::error::Error + Send + Sync>> {
    let subject = NamedNode::new(crate::identity::entity_uri(BASE_NS, name))?;
    let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
    let mut labels = BTreeSet::new();
    for quad in store.quads_for_pattern(
        Some(subject.as_ref().into()),
        Some(rdf_type.as_ref()),
        None,
        None,
    ) {
        if let Term::NamedNode(type_node) = quad?.object {
            if let Some(label) = type_node.as_str().strip_prefix(BASE_NS) {
                labels.insert(label.to_string());
            }
        }
    }
    Ok(labels)
}
#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub success: bool,
    /// 查询声明的变量顺序（`SELECT ?s ?p ?o` → s、p、o）；
    /// 构造型结果（CONSTRUCT / DESCRIBE）固定为 subject、predicate、object，
    /// ASK 为 boolean。渲染与 JSON 消费都应尊重这一顺序，而不是按字典序重排。
    pub columns: Vec<String>,
    pub data: Vec<HashMap<String, Value>>,
}
impl QueryResult {
    /// 表格化渲染任意 SPARQL 结果（列按查询声明顺序，行保持查询顺序）。
    ///
    /// 列宽按字符（而非字节）计算，中文实体名也能对齐；空结果给固定措辞，
    /// 与 `ASK false`（一行 `boolean = false`）在输出上可区分。
    pub fn render_text(&self) -> String {
        if self.data.is_empty() {
            return "（查询成功，无结果）".to_string();
        }
        let mut columns = self.columns.clone();
        let mut extras: Vec<String> = Vec::new();
        for row in &self.data {
            for key in row.keys() {
                if !columns.iter().any(|column| column == key)
                    && !extras.iter().any(|column| column == key)
                {
                    extras.push(key.clone());
                }
            }
        }
        if columns.is_empty() {
            // 兼容手工构造、未声明列顺序的结果：退回确定性的字典序。
            extras.sort();
            columns = extras;
        } else {
            extras.sort();
            columns.extend(extras);
        }
        if columns.is_empty() {
            return "（查询成功，无变量绑定）".to_string();
        }
        let cells: Vec<Vec<String>> = self
            .data
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|column| match row.get(column) {
                        Some(Value::String(text)) => text.clone(),
                        Some(value) => value.to_string(),
                        None => String::new(),
                    })
                    .collect()
            })
            .collect();
        let mut widths: Vec<usize> = columns
            .iter()
            .map(|column| column.chars().count())
            .collect();
        for row in &cells {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(cell.chars().count());
            }
        }
        let mut lines = Vec::with_capacity(cells.len() + 2);
        lines.push(render_table_row(&columns, &widths));
        lines.push(
            widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("-+-"),
        );
        for row in &cells {
            lines.push(render_table_row(row, &widths));
        }
        lines.join("\n")
    }
}
/// 单行表格：单元格左对齐、按字符宽度补空格。
fn render_table_row(cells: &[String], widths: &[usize]) -> String {
    cells
        .iter()
        .zip(widths)
        .map(|(cell, width)| {
            let padding = width.saturating_sub(cell.chars().count());
            format!("{}{}", cell, " ".repeat(padding))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
/// RDF 项 → JSON：IRI/空白节点用字符串；带类型的字面量按 XSD 类型映射为
/// 数字/布尔（其它保持字符串），`SELECT` 结果可直接被下游消费。
fn term_to_json(term: &Term) -> Value {
    match term {
        Term::NamedNode(node) => Value::String(node.as_str().to_string()),
        Term::BlankNode(node) => Value::String(format!("_:{}", node.as_str())),
        Term::Triple(triple) => Value::String(format!("<<{}>>", triple)),
        Term::Literal(literal) => {
            let text = literal.value();
            match literal.datatype().as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long"
                | "http://www.w3.org/2001/XMLSchema#short"
                | "http://www.w3.org/2001/XMLSchema#byte"
                | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                | "http://www.w3.org/2001/XMLSchema#unsignedLong" => text
                    .parse::<i64>()
                    .map(Value::from)
                    .unwrap_or_else(|_| Value::String(text.to_string())),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => text
                    .parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::String(text.to_string())),
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    Value::Bool(text == "true" || text == "1")
                }
                _ => Value::String(text.to_string()),
            }
        }
    }
}
/// 属性值到节点引用的上下文强制（`"@type": "@id"` / `"@type": "@vocab"`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeReferenceCoercion {
    /// 字符串按 `@base`（或上下文词项映射）解析为节点 IRI。
    Id,
    /// 字符串按 `@vocab`（或上下文词项映射）解析为节点 IRI。
    Vocab,
}
/// JSON-LD 上下文：词项 → IRI 映射，以及节点引用的类型强制声明。
///
/// 词项定义里的 `@type` 不只是元数据：`"@type": "@id"` 表示该属性的字符串值
/// 是节点引用而不是字面量；不记录它就会把「friend: "李四"」导入成字符串，
/// 后续 SPARQL/推理再也找不到这条边。
#[derive(Debug, Clone, Default)]
struct JsonLdContext {
    terms: HashMap<String, String>,
    id_coerced: HashSet<String>,
    vocab_coerced: HashSet<String>,
    /// 上下文级默认语言（`"@language": "zh"`）：未显式声明语言的字符串
    /// 字面量应带此语言标签，而不是退化成无语言的简单字面量。
    default_language: Option<String>,
    /// 本上下文重新定义的词项；合并到父级时必须先清掉继承的旧定义
    ///（含 `null` 清除与非 `null` 覆盖）。
    redefined: HashSet<String>,
}
impl JsonLdContext {
    fn clear_term(&mut self, term: &str) {
        if term == "@language" {
            self.default_language = None;
        }
        self.terms.remove(term);
        self.id_coerced.remove(term);
        self.vocab_coerced.remove(term);
        self.redefined.remove(term);
    }

    fn merge(&mut self, other: Self) {
        for term in other.redefined {
            self.clear_term(&term);
        }
        self.terms.extend(other.terms);
        self.id_coerced.extend(other.id_coerced);
        self.vocab_coerced.extend(other.vocab_coerced);
        if other.default_language.is_some() {
            self.default_language = other.default_language;
        }
    }

    fn coercion(&self, term: &str) -> Option<NodeReferenceCoercion> {
        if self.id_coerced.contains(term) {
            Some(NodeReferenceCoercion::Id)
        } else if self.vocab_coerced.contains(term) {
            Some(NodeReferenceCoercion::Vocab)
        } else {
            None
        }
    }
}
/// JSON-LD 导入统计：节点（实体）数、指向命名节点的关系边数、
/// 本次实际新增的三元组数（与库中已有三元组重复者不计入）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct JsonldImportStats {
    pub entities: usize,
    pub relations: usize,
    pub triples: usize,
}
impl JsonldImportStats {
    /// 单行中文摘要（CLI 与交互模式共用同一措辞）。
    pub fn summary(&self) -> String {
        format!(
            "解析了 {} 个实体和 {} 个关系，新增 {} 条三元组。",
            self.entities, self.relations, self.triples
        )
    }
}
// 所有配置结构体都需要是 pub，以便在测试中使用；
// 派生 Clone 让语言运行时能在不重启进程的前提下重建语言配置视图。
#[derive(Debug, Deserialize, Clone)]
pub struct SystemConfig {
    pub system: SystemSettings,
    pub api: Option<ApiConfig>,
    pub nlp: NlpConfig,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    pub entity_types: Vec<EntityTypeConfig>,
    pub extraction_rules: Vec<ExtractionRule>,
    pub prolog_rules: Vec<PrologRuleConfig>,
    pub messages: Messages,
    pub synonyms: Option<SynonymsConfig>,
    /// 谓词画像声明（可选）：导入准入、别名/逆向归一、功能谓词冲突裁决的共同依据
    #[serde(default)]
    pub predicate_profiles: Vec<predicate_registry::PredicateProfileConfig>,
    /// 实体身份声明（可选）：规范名 + 别名 + 类型，声明即已确认（显式承诺）
    #[serde(default)]
    pub entity_profiles: Vec<entity_registry::EntityProfileConfig>,
    /// 类型体系声明（可选）：别名/子类/互斥 + 模式闸门的闭包档
    #[serde(default)]
    pub type_system: type_system::TypeSystemConfig,
    /// 声明式插件（符号专家）目录、默认启用与逐包模式
    #[serde(default)]
    pub plugins: plugin::PluginSettings,
    /// 分层追踪图：多轮决策的显式工作记忆（问据栈/指代轨迹/更正失效）
    #[serde(default)]
    pub trace_graph: trace_graph::TraceGraphConfig,
    /// 配置编译的绑定帧：模式 -> 槽位 -> 目标 -> 措辞（默认关闭）。
    #[serde(default)]
    pub dialogue_frames: binding_frames::DialogueFramesConfig,
    /// 双向语义规则包：同一份规则负责正向抽取与反向实现（默认关闭）。
    #[serde(default)]
    pub semantic_grammar: semantic_grammar::SemanticGrammarConfig,
    /// 谓词中心语义角色帧：schema 同时驱动角色标注与反向生成（默认关闭）。
    #[serde(default)]
    pub frame_semantics: frame_semantics::FrameSemanticsConfig,
}
#[derive(Debug, Deserialize, Clone)]
pub struct SemanticMonitorConfig {
    #[serde(default = "default_semantic_monitor_window")]
    pub window: usize,
    #[serde(default)]
    pub mode: semantic_monitor::ControlMode,
    #[serde(default)]
    pub budget: semantic_monitor::BudgetConfig,
}

fn default_semantic_monitor_window() -> usize {
    semantic_monitor::DEFAULT_WINDOW
}

impl Default for SemanticMonitorConfig {
    fn default() -> Self {
        Self {
            window: default_semantic_monitor_window(),
            mode: semantic_monitor::ControlMode::Observe,
            budget: semantic_monitor::BudgetConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReasoningConfig {
    #[serde(default = "default_enable_link_prediction")]
    pub enable_link_prediction: bool,
    #[serde(default = "default_link_prediction_min_confidence")]
    pub link_prediction_min_confidence: f64,
    #[serde(default = "default_link_prediction_min_support")]
    pub link_prediction_min_support: usize,
    #[serde(default = "default_ontology_alignment_threshold")]
    pub ontology_alignment_threshold: f64,
    #[serde(default = "default_max_predictions")]
    pub max_predictions: usize,
    #[serde(default = "default_enable_semantic_embedding")]
    pub enable_semantic_embedding: bool,
    #[serde(default = "default_embedding_similarity_floor")]
    pub embedding_similarity_floor: f64,
    #[serde(default)]
    pub dialogue_state_path: Option<String>,
    #[serde(default = "default_enable_generation")]
    pub enable_generation: bool,
    #[serde(default = "default_max_profile_sentences")]
    pub max_profile_sentences: usize,
    #[serde(default = "default_completion_min_score")]
    pub completion_min_score: f64,
    #[serde(default = "default_enable_entailment")]
    pub enable_entailment: bool,
    #[serde(default = "default_entailment_min_confidence")]
    pub entailment_min_confidence: f64,
    #[serde(default = "default_enable_resolution")]
    pub enable_resolution: bool,
    #[serde(default = "default_resolution_max_depth")]
    pub resolution_max_depth: usize,
    #[serde(default = "default_resolution_max_solutions")]
    pub resolution_max_solutions: usize,
    #[serde(default = "default_enable_association_rules")]
    pub enable_association_rules: bool,
    #[serde(default = "default_association_min_support")]
    pub association_min_support: f64,
    #[serde(default = "default_association_min_confidence")]
    pub association_min_confidence: f64,
    #[serde(default = "default_association_min_lift")]
    pub association_min_lift: f64,
    #[serde(default = "default_enable_abduction")]
    pub enable_abduction: bool,
    #[serde(default = "default_abduction_max_hypotheses")]
    pub abduction_max_hypotheses: usize,
    #[serde(default = "default_abduction_max_scenario_size")]
    pub abduction_max_scenario_size: usize,
    #[serde(default = "default_enable_clarification")]
    pub enable_clarification: bool,
    #[serde(default = "default_enable_graph_walk")]
    pub enable_graph_walk: bool,
    #[serde(default = "default_graph_walk_damping")]
    pub graph_walk_damping: f64,
    #[serde(default = "default_graph_walk_iterations")]
    pub graph_walk_iterations: usize,
    #[serde(default = "default_enable_soft_logic")]
    pub enable_soft_logic: bool,
    #[serde(default = "default_soft_logic_max_iterations")]
    pub soft_logic_max_iterations: usize,
    #[serde(default = "default_soft_logic_min_probability")]
    pub soft_logic_min_probability: f64,
    #[serde(default = "default_enable_causal_reasoning")]
    pub enable_causal_reasoning: bool,
    #[serde(default = "default_causal_max_depth")]
    pub causal_max_depth: usize,
    #[serde(default = "default_enable_gradient_attribution")]
    pub enable_gradient_attribution: bool,
    #[serde(default = "default_enable_causal_tokenization")]
    pub enable_causal_tokenization: bool,
    #[serde(default = "default_enable_problog")]
    pub enable_problog: bool,
    #[serde(default = "default_problog_max_depth")]
    pub problog_max_depth: usize,
    #[serde(default = "default_problog_max_proofs")]
    pub problog_max_proofs: usize,
    #[serde(default = "default_enable_dialogue_manager")]
    pub enable_dialogue_manager: bool,
    #[serde(default = "default_enable_discourse")]
    pub enable_discourse: bool,
    #[serde(default = "default_discourse_max_sentences")]
    pub discourse_max_sentences: usize,
    #[serde(default = "default_discourse_max_rounds")]
    pub discourse_max_rounds: usize,
    #[serde(default = "default_enable_holographic_memory")]
    pub enable_holographic_memory: bool,
    #[serde(default = "default_holographic_dimension")]
    pub holographic_dimension: usize,
    #[serde(default = "default_holographic_similarity_floor")]
    pub holographic_similarity_floor: f64,
    #[serde(default = "default_holographic_max_hits")]
    pub holographic_max_hits: usize,
    /// 对话级证据装配（焦点帧 × 全息召回 × 证据束生成）；默认关闭。
    #[serde(default)]
    pub enable_contextual_rag: bool,
    /// 严格谓词准入：未登记谓词不自动准入、其事实不落库（默认自动准入 + 待确认）
    #[serde(default)]
    pub require_predicate_admission: bool,
    /// 可携带证明的置信度层（PCC）：校准断点、log-odds 融合、风险证书与四档决策。
    /// 默认关闭；只有显式启用且证书通过校验时才接管情态回答。
    #[serde(default)]
    pub confidence: confidence::ConfidenceConfig,
    /// 建设性多跳推理：在链式查询缺口处合成有证明的虚拟事实。
    /// 默认关闭；关闭时回答路径与旧行为逐字节一致。
    #[serde(default)]
    pub constructive: constructive::ConstructionConfig,
    /// 软路由：精确模式路由失败后，用软对齐与 log-odds 证据竞争解释。
    /// 默认开启；只有通过守卫/覆盖闸门且领先的候选才会被接受。
    #[serde(default)]
    pub soft_route: soft_route::SoftRouteConfig,
    /// 溯因式重写-检索互证：保留多个读法，让稠密检索证据裁决。
    /// 默认关闭；关闭时回答路径与旧行为逐字节一致。
    #[serde(default)]
    pub rewrite_retrieval: rewrite_retrieval::RewriteRetrievalConfig,
    #[serde(default)]
    pub semantic_monitor: SemanticMonitorConfig,
}
fn default_enable_holographic_memory() -> bool {
    true
}
fn default_holographic_dimension() -> usize {
    holographic_memory::DEFAULT_DIMENSION
}
fn default_holographic_similarity_floor() -> f64 {
    0.01
}
fn default_holographic_max_hits() -> usize {
    6
}
fn default_enable_problog() -> bool {
    true
}
fn default_problog_max_depth() -> usize {
    6
}
fn default_problog_max_proofs() -> usize {
    12
}
fn default_enable_discourse() -> bool {
    true
}
fn default_enable_dialogue_manager() -> bool {
    true
}
fn default_discourse_max_sentences() -> usize {
    6
}
fn default_discourse_max_rounds() -> usize {
    4
}
fn default_enable_causal_reasoning() -> bool {
    true
}
fn default_causal_max_depth() -> usize {
    4
}
fn default_enable_gradient_attribution() -> bool {
    true
}
fn default_enable_causal_tokenization() -> bool {
    true
}
fn default_enable_resolution() -> bool {
    true
}
fn default_resolution_max_depth() -> usize {
    6
}
fn default_resolution_max_solutions() -> usize {
    12
}
fn default_enable_association_rules() -> bool {
    true
}
fn default_association_min_support() -> f64 {
    0.2
}
fn default_association_min_confidence() -> f64 {
    0.5
}
fn default_association_min_lift() -> f64 {
    1.05
}
fn default_enable_abduction() -> bool {
    true
}
fn default_abduction_max_hypotheses() -> usize {
    5
}
fn default_abduction_max_scenario_size() -> usize {
    2
}
fn default_enable_clarification() -> bool {
    true
}
fn default_enable_graph_walk() -> bool {
    true
}
fn default_graph_walk_damping() -> f64 {
    0.85
}
fn default_graph_walk_iterations() -> usize {
    12
}
fn default_enable_soft_logic() -> bool {
    true
}
fn default_soft_logic_max_iterations() -> usize {
    4
}
fn default_soft_logic_min_probability() -> f64 {
    0.5
}
fn default_enable_entailment() -> bool {
    true
}
fn default_entailment_min_confidence() -> f64 {
    0.5
}
fn default_enable_generation() -> bool {
    true
}
fn default_max_profile_sentences() -> usize {
    6
}
fn default_completion_min_score() -> f64 {
    0.3
}
fn default_enable_semantic_embedding() -> bool {
    true
}
fn default_embedding_similarity_floor() -> f64 {
    0.6
}
fn default_enable_link_prediction() -> bool {
    true
}
fn default_link_prediction_min_confidence() -> f64 {
    0.6
}
fn default_link_prediction_min_support() -> usize {
    2
}
fn default_ontology_alignment_threshold() -> f64 {
    0.72
}
fn default_max_predictions() -> usize {
    5
}
impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enable_link_prediction: default_enable_link_prediction(),
            link_prediction_min_confidence: default_link_prediction_min_confidence(),
            link_prediction_min_support: default_link_prediction_min_support(),
            ontology_alignment_threshold: default_ontology_alignment_threshold(),
            max_predictions: default_max_predictions(),
            enable_semantic_embedding: default_enable_semantic_embedding(),
            embedding_similarity_floor: default_embedding_similarity_floor(),
            dialogue_state_path: None,
            enable_generation: default_enable_generation(),
            max_profile_sentences: default_max_profile_sentences(),
            completion_min_score: default_completion_min_score(),
            enable_entailment: default_enable_entailment(),
            entailment_min_confidence: default_entailment_min_confidence(),
            enable_resolution: default_enable_resolution(),
            resolution_max_depth: default_resolution_max_depth(),
            resolution_max_solutions: default_resolution_max_solutions(),
            enable_association_rules: default_enable_association_rules(),
            association_min_support: default_association_min_support(),
            association_min_confidence: default_association_min_confidence(),
            association_min_lift: default_association_min_lift(),
            enable_abduction: default_enable_abduction(),
            abduction_max_hypotheses: default_abduction_max_hypotheses(),
            abduction_max_scenario_size: default_abduction_max_scenario_size(),
            enable_clarification: default_enable_clarification(),
            enable_graph_walk: default_enable_graph_walk(),
            graph_walk_damping: default_graph_walk_damping(),
            graph_walk_iterations: default_graph_walk_iterations(),
            enable_soft_logic: default_enable_soft_logic(),
            soft_logic_max_iterations: default_soft_logic_max_iterations(),
            soft_logic_min_probability: default_soft_logic_min_probability(),
            enable_causal_reasoning: default_enable_causal_reasoning(),
            causal_max_depth: default_causal_max_depth(),
            enable_gradient_attribution: default_enable_gradient_attribution(),
            enable_causal_tokenization: default_enable_causal_tokenization(),
            enable_problog: default_enable_problog(),
            problog_max_depth: default_problog_max_depth(),
            problog_max_proofs: default_problog_max_proofs(),
            enable_dialogue_manager: default_enable_dialogue_manager(),
            enable_discourse: default_enable_discourse(),
            discourse_max_sentences: default_discourse_max_sentences(),
            discourse_max_rounds: default_discourse_max_rounds(),
            enable_holographic_memory: default_enable_holographic_memory(),
            holographic_dimension: default_holographic_dimension(),
            holographic_similarity_floor: default_holographic_similarity_floor(),
            holographic_max_hits: default_holographic_max_hits(),
            enable_contextual_rag: false,
            require_predicate_admission: false,
            confidence: confidence::ConfidenceConfig::default(),
            constructive: constructive::ConstructionConfig::default(),
            soft_route: soft_route::SoftRouteConfig::default(),
            rewrite_retrieval: rewrite_retrieval::RewriteRetrievalConfig::default(),
            semantic_monitor: SemanticMonitorConfig::default(),
        }
    }
}
#[derive(Debug, Deserialize, Clone)]
pub struct ApiConfig {
    pub ownthink_base_url: String,
    pub ownthink_timeout_seconds: u64,
    pub enable_entity_disambiguation: bool,
    pub max_ambiguous_entities: usize,
    pub auto_select_best_entity: bool,
    /// 外链名册落盘路径（缺省为 `<store>/external_ledger.json`）
    #[serde(default)]
    pub external_ledger_path: Option<String>,
    /// 外链名册容量（条目数上限，LRU 淘汰）
    #[serde(default = "default_external_ledger_capacity")]
    pub external_ledger_capacity: usize,
    /// 外链名册时限（逻辑轮次；0 = 不过期，刷新交给显式命令）
    #[serde(default)]
    pub external_ledger_ttl_ticks: u64,
}
fn default_external_ledger_capacity() -> usize {
    external_ledger::DEFAULT_CAPACITY
}
#[derive(Debug, Deserialize, Clone)]
pub struct SynonymsConfig {
    pub mappings: HashMap<String, String>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct SystemSettings {
    pub oxigraph_endpoint: String,
    pub oxigraph_store_path: String,
    pub default_language: String,
}
#[derive(Debug, Deserialize, Clone)]
pub struct NlpConfig {
    pub import_keywords: Vec<String>,
    pub pos_dict: HashMap<String, String>,
    pub name_patterns: NamePatterns,
    pub enable_nlu: Option<bool>,
    pub nlu_confidence_threshold: Option<f64>,
    pub enable_local_semantic: Option<bool>,
    pub disable_ownthink_api: Option<bool>,
    pub semantic_similarity_threshold: Option<f64>,
    pub max_entity_candidates: Option<usize>,
    pub enable_advanced_entity_extraction: Option<bool>,
    pub enable_relation_extraction: Option<bool>,
    pub context_window_size: Option<usize>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct NamePatterns {
    pub prefixes: Vec<String>,
    pub suffixes: Vec<String>,
    pub full_names: Vec<String>,
    pub profession_suffix: Vec<String>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct EntityTypeConfig {
    pub label: String,
    pub properties: Vec<PropertyConfig>,
    pub examples: Vec<String>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct PropertyConfig {
    pub name: String,
    pub data_type: String,
    pub required: bool,
    #[serde(default)]
    pub extraction_patterns: Vec<String>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct ExtractionRule {
    pub name: String,
    pub entity_type: String,
    pub patterns: Vec<PatternConfig>,
    pub priority: i32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct PatternConfig {
    pub pos_pattern: Vec<String>,
    pub keyword_patterns: Option<Vec<String>>,
    pub property_mappings: HashMap<String, String>,
}
#[derive(Debug, Deserialize, Clone)]
pub struct PrologRuleConfig {
    pub name: String,
    pub pattern: Vec<String>,
    pub query_type: String,
    pub priority: i32,
    /// 泛化规则命中后仍交给软路由复核；软层拒绝或澄清时保留本规则。
    /// 用于「属性查询」这类只消费前缀的兜底模式，避免它们挡住更具体的
    /// 配置规则在近失句上的软对齐。
    #[serde(default)]
    pub defer_to_soft: bool,
}
#[derive(Debug, Deserialize, Clone)]
pub struct Messages {
    pub startup: String,
    pub ready: String,
    pub query_prompt: String,
    pub import_prompt: String,
    pub quit_prompt: String,
    pub importing: String,
    pub no_knowledge_extracted: String,
    pub cannot_understand: String,
    pub process_error: String,
}
#[derive(Debug)]
pub enum InputMode {
    Query,
    KnowledgeImport,
}
#[derive(Debug, Clone)]
pub enum KnowledgeItem {
    Entity {
        label: String,
        properties: HashMap<String, String>,
    },
    Relationship {
        from_label: String,
        from_id: String,
        to_label: String,
        to_id: String,
        rel_type: String,
        properties: HashMap<String, String>,
    },
}
pub struct KnowledgeExtractor {
    config: Arc<SystemConfig>,
    semantic_grammar: semantic_grammar::RulePack,
    semantic_grammar_enabled: bool,
}
impl KnowledgeExtractor {
    pub fn new(config: Arc<SystemConfig>) -> Self {
        let semantic_grammar = semantic_grammar::RulePack::from_config(&config);
        let semantic_grammar_enabled = config.semantic_grammar.enabled;
        Self {
            config,
            semantic_grammar,
            semantic_grammar_enabled,
        }
    }

    pub fn extract(&self, text: &str, tagged_words: &[TaggedWord]) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();
        let tagged_words = nlp::merge_name_fragments(tagged_words);
        let tagged_words = tagged_words.as_slice();

        for rule in &self.config.extraction_rules {
            items.extend(self.apply_extraction_rule(text, tagged_words, rule));
        }

        if self.semantic_grammar_enabled {
            items.extend(self.semantic_grammar.extract_items(tagged_words));
        }

        if text.contains("，") || text.contains("；") || text.contains("。") {
            let parts: Vec<&str> = text.split(&['，', '；', '。'][..]).collect();
            for part in parts {
                if part.trim().is_empty() {
                    continue;
                }
                let part_tagged = preprocess(part, &self.config.nlp);
                let normalized_part_tagged = if let Some(synonyms_config) = &self.config.synonyms {
                    normalize_words(part_tagged, &synonyms_config.mappings)
                } else {
                    part_tagged
                };
                items.extend(self.extract(part, &normalized_part_tagged));
            }
        }

        items
    }

    fn apply_extraction_rule(
        &self,
        text: &str,
        tagged_words: &[TaggedWord],
        rule: &ExtractionRule,
    ) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();

        for pattern in &rule.patterns {
            if let Some(item) = self.match_pattern(text, tagged_words, pattern, &rule.entity_type) {
                items.push(item);
            }
        }

        items
    }

    fn match_pattern(
        &self,
        text: &str,
        tagged_words: &[TaggedWord],
        pattern: &PatternConfig,
        entity_type: &str,
    ) -> Option<KnowledgeItem> {
        if !self.check_pos_pattern(tagged_words, &pattern.pos_pattern) {
            return None;
        }

        // 检查关键词模式
        if let Some(keyword_patterns) = &pattern.keyword_patterns {
            if !self.check_keyword_patterns(text, keyword_patterns) {
                return None;
            }
        }

        let properties = self.extract_properties(tagged_words, &pattern.property_mappings);

        if properties.is_empty() {
            return None;
        }

        Some(KnowledgeItem::Entity {
            label: entity_type.to_string(),
            properties,
        })
    }

    fn check_keyword_patterns(&self, text: &str, keyword_patterns: &[String]) -> bool {
        for keyword in keyword_patterns {
            if text.contains(keyword) {
                return true;
            }
        }
        false
    }

    fn check_pos_pattern(&self, tagged_words: &[TaggedWord], pos_pattern: &[String]) -> bool {
        if pos_pattern.len() > tagged_words.len() {
            return false;
        }

        for i in 0..=(tagged_words.len() - pos_pattern.len()) {
            let mut matched = true;
            for (j, expected_pos) in pos_pattern.iter().enumerate() {
                if expected_pos != "*" && tagged_words[i + j].pos != *expected_pos {
                    matched = false;
                    break;
                }
            }
            if matched {
                return true;
            }
        }

        false
    }

    fn extract_properties(
        &self,
        tagged_words: &[TaggedWord],
        mappings: &HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut properties = HashMap::new();

        for (prop_name, extraction_rule) in mappings {
            if let Some(value) = self.extract_property_value(tagged_words, extraction_rule) {
                properties.insert(prop_name.clone(), value);
            }
        }

        properties
    }

    fn extract_property_value(&self, tagged_words: &[TaggedWord], rule: &str) -> Option<String> {
        if rule.starts_with("pos:") {
            let pos_rule = rule.strip_prefix("pos:")?;

            // 处理索引语法，如 pos:nr[0]
            if pos_rule.contains('[') && pos_rule.contains(']') {
                let parts: Vec<&str> = pos_rule.split(['[', ']']).collect();
                if parts.len() >= 2 {
                    let pos = parts[0];
                    // 索引语法畸形时视为无效规则
                    let index: usize = parts[1].trim().parse().ok()?;

                    let mut matching_words = Vec::new();
                    for word in tagged_words {
                        if word.pos == pos {
                            matching_words.push(word.text.clone());
                        }
                    }

                    if index < matching_words.len() {
                        return Some(matching_words[index].clone());
                    }
                }
            }

            // 人名提取：相邻的 nr + 单字 unknown 已在预处理阶段合并
            // （见 nlp::merge_name_fragments），这里只取首个 nr 词，
            // 避免把「姚明身高226」这类句式里紧跟的未知属性词并进人名。
            if pos_rule == "nr" {
                for word in tagged_words {
                    if word.pos == "nr" {
                        return Some(word.text.clone());
                    }
                }
            }

            // 处理通配符 *
            if pos_rule == "*" {
                // 对于通配符，返回所有不是功能词的词的组合
                let content_words: Vec<String> = tagged_words
                    .iter()
                    .filter(|word| {
                        !matches!(word.pos.as_str(), "u" | "c" | "p")
                            && !matches!(word.text.as_str(), "的" | "是" | "有")
                    })
                    .map(|word| word.text.clone())
                    .collect();
                if !content_words.is_empty() {
                    return Some(content_words.join(""));
                }
            }

            // 普通词性查找
            for word in tagged_words {
                if word.pos == pos_rule {
                    return Some(word.text.clone());
                }
            }
        } else if rule.starts_with("text:") {
            let text = rule.strip_prefix("text:")?;
            for word in tagged_words {
                if word.text == text {
                    return Some(word.text.clone());
                }
            }
        }

        None
    }
}
// Oxigraph操作管理器
pub struct OxigraphManager {
    store: Arc<Mutex<Store>>,
    // 每次写操作自增，用于判定缓存的整图快照是否仍然有效
    revision: AtomicU64,
    snapshot_cache: parking_lot::Mutex<Option<(u64, Arc<Vec<PrologFact>>)>>,
    // 相关事实查询缓存：同一知识版本内按查询键多槽缓存（LRU），
    // 交替询问不同实体互不挤占；写后整体失效
    relevant_cache: parking_lot::Mutex<DeltaQualifiedCache<Arc<Vec<PrologFact>>>>,
    // 直答缓存：同一知识版本内按查询键多槽缓存（LRU，含"确无答案"的负缓存）
    direct_answer_cache: parking_lot::Mutex<DeltaQualifiedCache<Option<String>>>,
    // 操作计数台账（可与会话推理器共享同一实例）
    perf: Arc<perf_ledger::PerfLedger>,
    // 谓词画像注册表：导入准入、别名/逆向归一、功能谓词冲突裁决
    predicate_registry: parking_lot::RwLock<predicate_registry::PredicateRegistry>,
    // 实体身份注册表：提名（待确认）/ 确认 / 拒绝 / 忘记 / 合并的显式台账
    entity_registry: parking_lot::RwLock<entity_registry::EntityRegistry>,
    // 类型体系：别名/子类/互斥声明的类型格，模式闸门与质量护照共同消费
    type_system: parking_lot::RwLock<type_system::TypeSystem>,
}
/// 每次写操作缓存容量上限（条目数）。对话中交替询问少量实体，
/// 32 槽足以消除往返抖动，同时让内存占用可计算。
const QUERY_CACHE_CAPACITY: usize = 32;
use crate::cache::RevisionLru;
use crate::delta_cache::{
    ClosureFootprint, DeltaFootprint, DeltaQualifiedCache, FactPattern, Qualification,
};

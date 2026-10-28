/// 批次写入事务的错误：只用于把非存储错误（IRI / JSON）带进 Oxigraph 事务闭包。
#[derive(Debug)]
struct ImportBatchTransactionError(String);

impl std::fmt::Display for ImportBatchTransactionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ImportBatchTransactionError {}

impl From<oxigraph::store::StorageError> for ImportBatchTransactionError {
    fn from(error: oxigraph::store::StorageError) -> Self {
        Self(error.to_string())
    }
}

fn push_unique_episode_fact(
    facts: &mut Vec<crate::import_episode::GistFact>,
    seen: &mut BTreeSet<(String, String, String)>,
    quad: &Quad,
) {
    let fact = crate::import_episode::gist_fact(quad);
    if seen.insert((
        fact.subject.clone(),
        fact.predicate.clone(),
        fact.object.clone(),
    )) {
        facts.push(fact);
    }
}

/// 一次批量写入的结果（RDF 与 JSON-LD 导入共用，保证两者的批次语义一致）。
struct EpisodeWriteOutcome {
    added: usize,
    duplicates: usize,
    episode: crate::import_episode::ImportEpisode,
    recorded: bool,
    source_aliases_added: bool,
}

/// JSON-LD 收据：保留原有实体/关系统计，并附上本次导入情景。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonldEpisodeReceipt {
    pub stats: JsonldImportStats,
    pub episode: crate::import_episode::ImportEpisode,
    pub recorded: bool,
    pub source_aliases_added: bool,
}

impl JsonldEpisodeReceipt {
    pub fn summary(&self) -> String {
        format!(
            "{}\n批次 B{}｜id {}｜批次图 {}",
            self.stats.summary(),
            self.episode.seq,
            self.episode.id,
            self.episode.batch_graph
        )
    }
}

impl OxigraphManager {
    pub fn new(store_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_perf_ledger(store_path, Arc::new(perf_ledger::PerfLedger::new()))
    }

    /// 与 `new` 相同，但可注入共享的性能台账（会话级操作计数）。
    pub fn with_perf_ledger(
        store_path: &str,
        perf: Arc<perf_ledger::PerfLedger>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 展开波浪号路径
        let expanded_path = if store_path.starts_with("~/") {
            if let Some(home_dir) = std::env::var_os("HOME") {
                let home_path = std::path::Path::new(&home_dir);
                home_path
                    .join(&store_path[2..])
                    .to_string_lossy()
                    .to_string()
            } else {
                store_path.to_string()
            }
        } else {
            store_path.to_string()
        };

        // 如果目录不存在则创建
        if let Some(parent) = std::path::Path::new(&expanded_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let store = Store::open(&expanded_path)?;
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            revision: AtomicU64::new(0),
            snapshot_cache: parking_lot::Mutex::new(None),
            relevant_cache: parking_lot::Mutex::new(RevisionLru::new(QUERY_CACHE_CAPACITY)),
            direct_answer_cache: parking_lot::Mutex::new(RevisionLru::new(QUERY_CACHE_CAPACITY)),
            perf,
            predicate_registry: parking_lot::RwLock::new(
                predicate_registry::PredicateRegistry::new(),
            ),
            entity_registry: parking_lot::RwLock::new(entity_registry::EntityRegistry::new()),
            type_system: parking_lot::RwLock::new(type_system::TypeSystem::new()),
        })
    }

    /// 共享的性能台账（与 `SymbolicReasoner` 可为同一实例）。
    pub fn perf_ledger(&self) -> Arc<perf_ledger::PerfLedger> {
        Arc::clone(&self.perf)
    }

    /// 标记知识版本变化：调用方必须持有存储锁，且在本次实际写入完成之后
    /// （version-after-write）。这样并发读者绝不会把过期数据缓存到新版本号上，
    /// 且重复导入不推进版本。
    fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.perf.bump("revision_bump");
    }

    pub async fn import_jsonld(
        &self,
        jsonld_content: &str,
    ) -> Result<JsonldImportStats, Box<dyn std::error::Error + Send + Sync>> {
        // 先在内存暂存区完整展开，再一次性写入：JSON-LD 结构错误不会留下半批
        // 三元组，也不会在失败时推进知识版本。
        let (quads, mut stats) = self.stage_jsonld(jsonld_content).await?;
        let (added, _) = self.insert_flattened_quads(&quads).await?;
        stats.triples = added;
        Ok(stats)
    }

    /// 在内存暂存区展开 JSON-LD，返回默认图四元组与解析统计。
    ///
    /// 暂存避免结构化导入直接改写主库：解析/展开失败时主库不受影响。
    async fn stage_jsonld(
        &self,
        jsonld_content: &str,
    ) -> Result<(Vec<Quad>, JsonldImportStats), Box<dyn std::error::Error + Send + Sync>> {
        let jsonld_value: Value = serde_json::from_str(jsonld_content)?;
        let staging = Store::new()?;
        let mut stats = JsonldImportStats::default();
        self.process_jsonld_simple(&staging, &jsonld_value, &mut stats)
            .await?;
        let quads = staging.iter().collect::<Result<Vec<_>, _>>()?;
        Ok((quads, stats))
    }

    /// 把已展开的四元组写入默认图；返回（新增，重复）。
    async fn insert_flattened_quads(
        &self,
        quads: &[Quad],
    ) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let (added, duplicates) = store.transaction(move |mut transaction| {
            let mut added = 0usize;
            let mut duplicates = 0usize;
            for quad in quads {
                if transaction.insert(quad.as_ref())? {
                    added += 1;
                } else {
                    duplicates += 1;
                }
            }
            Ok::<(usize, usize), oxigraph::store::StorageError>((added, duplicates))
        })?;
        if added > 0 {
            self.bump_revision();
        }
        Ok((added, duplicates))
    }

    async fn process_jsonld_simple(
        &self,
        store: &Store,
        jsonld: &Value,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 处理直接的 JSON-LD 结构
        if let Some(array) = jsonld.as_array() {
            for item in array {
                if let Some(obj) = item.as_object() {
                    // 处理每个对象
                    let context = Self::extract_jsonld_context(obj);
                    self.process_jsonld_object(store, obj, &context, stats)?;
                }
            }
        } else if let Some(obj) = jsonld.as_object() {
            // 顶层 @context 中的字符串映射用于展开类型与属性名
            let context = Self::extract_jsonld_context(obj);
            // 顶层为包含 @graph 的 JSON-LD 文档时，递归处理图中的每个节点
            if let Some(graph) = obj.get("@graph") {
                if let Some(graph_array) = graph.as_array() {
                    for item in graph_array {
                        if let Some(inner) = item.as_object() {
                            self.process_jsonld_object(store, inner, &context, stats)?;
                        }
                    }
                } else if let Some(inner) = graph.as_object() {
                    self.process_jsonld_object(store, inner, &context, stats)?;
                }
            } else {
                // 单个对象
                self.process_jsonld_object(store, obj, &context, stats)?;
            }
        }
        Ok(())
    }

    /// 提取 @context 中的映射（term -> IRI / @base / @vocab）与节点引用强制，
    /// 供类型与属性名展开使用。
    ///
    /// 支持三种常见写法：
    /// - 字符串映射：`"name": "http://example.org/kg/name"`
    /// - 上下文数组：`"@context": [{...}, {...}]`（后者覆盖前者）
    /// - 对象映射：`"ex": { "@id": "http://example.org/kg/" }`，并可带
    ///   `"@type": "@id"` / `"@type": "@vocab"` 声明节点引用
    ///
    /// `null` 词项定义按 JSON-LD 语义清除先前映射，避免作用域上下文里
    /// 「显式清空」被误读成继承。
    fn extract_jsonld_context(obj: &serde_json::Map<String, Value>) -> JsonLdContext {
        let mut context = JsonLdContext::default();
        let Some(ctx) = obj.get("@context") else {
            return context;
        };
        let apply = |ctx: &serde_json::Map<String, Value>, context: &mut JsonLdContext| {
            for (term, value) in ctx {
                context.clear_term(term);
                context.redefined.insert(term.clone());
                if value.is_null() {
                    continue;
                }
                if let Some(iri) = value.as_str() {
                    context.terms.insert(term.clone(), iri.to_string());
                    continue;
                }
                let Some(definition) = value.as_object() else {
                    continue;
                };
                if let Some(iri) = definition.get("@id").and_then(Value::as_str) {
                    context.terms.insert(term.clone(), iri.to_string());
                }
                match definition.get("@type").and_then(Value::as_str) {
                    Some("@id") => {
                        context.id_coerced.insert(term.clone());
                    }
                    Some("@vocab") => {
                        context.vocab_coerced.insert(term.clone());
                    }
                    _ => {}
                }
            }
        };
        if let Some(ctx) = ctx.as_object() {
            apply(ctx, &mut context);
        } else if let Some(items) = ctx.as_array() {
            for item in items {
                if let Some(ctx) = item.as_object() {
                    apply(ctx, &mut context);
                }
            }
        }
        context
    }

    /// 按 @context 展开 JSON-LD 词项：已是 IRI 的保留，映射词项替换为对应 IRI，否则回退到 @vocab。
    ///
    /// 支持前缀映射（`ex:name`）与「词项 → 紧凑 IRI → 绝对 IRI」的链式展开；
    /// 设深度上限避免自引用上下文导致死循环。
    fn expand_jsonld_term(term: &str, context: &JsonLdContext) -> String {
        let mut current = term.to_string();
        for _ in 0..8 {
            if let Some((prefix, suffix)) = current.split_once(':') {
                if let Some(base) = context.terms.get(prefix) {
                    return format!("{}{}", base, suffix);
                }
                return current;
            }
            let Some(iri) = context.terms.get(&current) else {
                break;
            };
            if iri == &current {
                break;
            }
            current = iri.clone();
        }
        if !current.contains(':') {
            if let Some(vocab) = context.terms.get("@vocab") {
                return format!("{}{}", vocab, current);
            }
        }
        current
    }

    /// 展开节点 `@id`：裸标识符按 `@base` 解析为绝对 IRI（JSON-LD 文档相对 IRI 语义）。
    fn expand_jsonld_id(id: &str, context: &JsonLdContext) -> String {
        if id.contains(':') || context.terms.contains_key(id) {
            return Self::expand_jsonld_term(id, context);
        }
        if let Some(base) = context.terms.get("@base") {
            return format!("{}{}", base, id);
        }
        id.to_string()
    }

    /// 把被声明为节点引用的字符串解析为节点：`@id` 按 `@base`，`@vocab` 按词表；
    /// 两者都没有可用基准时回落到本知识库的实体 URI 规则，而不是产出相对 IRI。
    fn jsonld_coerced_node(
        value: &str,
        coercion: NodeReferenceCoercion,
        context: &JsonLdContext,
    ) -> Result<NamedOrBlankNode, Box<dyn std::error::Error + Send + Sync>> {
        let expanded = match coercion {
            NodeReferenceCoercion::Id => Self::expand_jsonld_id(value, context),
            NodeReferenceCoercion::Vocab => Self::expand_jsonld_term(value, context),
        };
        if expanded.contains(':') {
            Self::jsonld_node(&expanded, context)
        } else {
            Self::jsonld_node(&crate::identity::entity_uri(BASE_NS, &expanded), context)
        }
    }

    /// JSON 原生标量 → RDF 字面量：数字/布尔保留 XSD 类型（与 JSON-LD 展开一致）。
    ///
    /// `null` 不是 RDF 值：JSON-LD 语义要求忽略，返回 `None`。对象/数组不应
    /// 走到这里（它们由上层按 `@value` / `@list` / 节点展开），保留字符串兜底。
    fn jsonld_scalar_literal(value: &Value) -> Option<Literal> {
        const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";
        let datatype = |local: &str| {
            NamedNode::new(format!("{}{}", XSD_NS, local)).expect("XSD 常量 IRI 必须合法")
        };
        match value {
            Value::Null => None,
            Value::String(text) => Some(Literal::new_simple_literal(text)),
            Value::Bool(boolean) => Some(Literal::new_typed_literal(
                boolean.to_string(),
                datatype("boolean"),
            )),
            Value::Number(number) => {
                let local = if number.is_i64() || number.is_u64() {
                    "integer"
                } else {
                    "double"
                };
                Some(Literal::new_typed_literal(
                    number.to_string(),
                    datatype(local),
                ))
            }
            other => Some(Literal::new_simple_literal(&other.to_string())),
        }
    }

    /// 把 JSON-LD 节点引用解析为命名节点或空白节点。
    ///
    /// `_:label` 是 JSON-LD 的空白节点标识符（不是 IRI）。同一标签在重复导入时
    /// 必须映射到同一空白节点，才能保持导入幂等；标签含非法字符时用稳定哈希兜底。
    fn jsonld_node(
        id: &str,
        context: &JsonLdContext,
    ) -> Result<NamedOrBlankNode, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(label) = id.strip_prefix("_:") {
            if let Ok(node) = BlankNode::new(label.to_string()) {
                return Ok(node.into());
            }
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in label.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            return Ok(BlankNode::new(format!("b{:016x}", hash))?.into());
        }
        Ok(NamedNode::new(Self::expand_jsonld_id(id, context))?.into())
    }

    fn process_jsonld_object(
        &self,
        store: &Store,
        obj: &serde_json::Map<String, Value>,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 节点级 @context 是作用域上下文：只影响本节点及其嵌套值。
        let mut context = context.clone();
        if obj.contains_key("@context") {
            context.merge(Self::extract_jsonld_context(obj));
        }
        let context = &context;

        // 纯图容器（只有 @graph/@included 等关键字）本身不是实体：
        // 直接展开其中的节点，避免生成随机空白实体。
        let has_node_payload =
            obj.contains_key("@type") || obj.keys().any(|key| !key.starts_with("@"));
        if !has_node_payload {
            let graph = obj.get("@graph");
            let included = obj.get("@included");
            if graph.is_none() && included.is_none() {
                return Ok(());
            }
            if let Some(graph) = graph {
                self.process_jsonld_graph(store, graph, context, stats)?;
            }
            if let Some(included) = included {
                self.process_jsonld_graph(store, included, context, stats)?;
            }
            return Ok(());
        }

        // 获取主语 (subject)：@id 可以是 IRI，也可以是 `_:label` 空白节点。
        let subject: NamedOrBlankNode = if let Some(id) = obj.get("@id") {
            let id = id.as_str().unwrap_or("");
            if id.is_empty() {
                return Ok(());
            }
            Self::jsonld_node(id, context)?
        } else {
            NamedNode::new(crate::identity::entity_uri(
                BASE_NS,
                &Uuid::new_v4().to_string(),
            ))?
            .into()
        };
        stats.entities += 1;

        // 处理类型信息
        if let Some(types) = obj.get("@type") {
            if let Some(type_array) = types.as_array() {
                for type_val in type_array {
                    if let Some(type_uri) = type_val.as_str() {
                        let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
                        let object_type =
                            NamedNode::new(Self::expand_jsonld_term(type_uri, context))?;

                        store.insert(&Quad::new(
                            subject.clone(),
                            rdf_type,
                            object_type,
                            GraphName::DefaultGraph,
                        ))?;
                    }
                }
            } else if let Some(type_uri) = types.as_str() {
                let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
                let object_type = NamedNode::new(Self::expand_jsonld_term(type_uri, context))?;

                store.insert(&Quad::new(
                    subject.clone(),
                    rdf_type,
                    object_type,
                    GraphName::DefaultGraph,
                ))?;
            }
        }

        // 处理属性
        for (key, value) in obj {
            if key.starts_with("@") {
                continue; // 跳过 JSON-LD 关键字
            }

            let predicate = NamedNode::new(Self::expand_jsonld_term(key, context))?;
            let coercion = context.coercion(key);

            if let Some(value_array) = value.as_array() {
                for val in value_array {
                    self.process_property_value(
                        store, &subject, &predicate, val, coercion, context, stats,
                    )?;
                }
            } else {
                self.process_property_value(
                    store, &subject, &predicate, value, coercion, context, stats,
                )?;
            }
        }

        // @reverse：值里的节点是主语，当前节点是宾语（JSON-LD 反向属性）。
        if let Some(reverse) = obj.get("@reverse").and_then(Value::as_object) {
            for (key, value) in reverse {
                if key.starts_with("@") {
                    continue;
                }
                let predicate = NamedNode::new(Self::expand_jsonld_term(key, context))?;
                let coercion = context.coercion(key);
                if let Some(value_array) = value.as_array() {
                    for val in value_array {
                        self.process_reverse_value(
                            store, &subject, &predicate, val, coercion, context, stats,
                        )?;
                    }
                } else {
                    self.process_reverse_value(
                        store, &subject, &predicate, value, coercion, context, stats,
                    )?;
                }
            }
        }

        // 带 @id/属性的节点也可以顺带承载命名图或 @included 节点；
        // 知识库只使用默认图，这里把其内容展开到默认图而不是丢弃。
        if let Some(graph) = obj.get("@graph") {
            self.process_jsonld_graph(store, graph, context, stats)?;
        }
        if let Some(included) = obj.get("@included") {
            self.process_jsonld_graph(store, included, context, stats)?;
        }

        Ok(())
    }

    /// 展开 `@graph` / `@included` 的值：接受单个节点或节点数组。
    fn process_jsonld_graph(
        &self,
        store: &Store,
        value: &Value,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            Value::Array(items) => {
                for item in items {
                    if let Some(obj) = item.as_object() {
                        self.process_jsonld_object(store, obj, context, stats)?;
                    }
                }
            }
            Value::Object(obj) => self.process_jsonld_object(store, obj, context, stats)?,
            _ => {}
        }
        Ok(())
    }

    /// 处理 `@reverse` 中的节点引用：节点 → 谓词 → 当前节点。
    fn process_reverse_value(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        value: &Value,
        coercion: Option<NodeReferenceCoercion>,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(obj) = value.as_object() else {
            if let (Some(text), Some(coercion)) = (value.as_str(), coercion) {
                let object = Self::jsonld_coerced_node(text, coercion, context)?;
                store.insert(&Quad::new(
                    object,
                    predicate.clone(),
                    subject.clone(),
                    GraphName::DefaultGraph,
                ))?;
                stats.relations += 1;
            }
            return Ok(());
        };
        let Some(id) = obj.get("@id").and_then(Value::as_str) else {
            return Ok(());
        };
        if id.is_empty() {
            return Ok(());
        }
        let object = Self::jsonld_node(id, context)?;
        store.insert(&Quad::new(
            object.clone(),
            predicate.clone(),
            subject.clone(),
            GraphName::DefaultGraph,
        ))?;
        stats.relations += 1;
        // 带额外属性的反向节点本身也是节点（与正向嵌套对象一致）。
        if obj.keys().any(|key| key != "@id" && key != "@context") {
            self.process_jsonld_object(store, obj, context, stats)?;
        }
        Ok(())
    }

    fn process_property_value(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        value: &Value,
        coercion: Option<NodeReferenceCoercion>,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            Value::Null => {}
            Value::String(s) => {
                if let Some(coercion) = coercion {
                    let object = Self::jsonld_coerced_node(s, coercion, context)?;
                    store.insert(&Quad::new(
                        subject.clone(),
                        predicate.clone(),
                        object,
                        GraphName::DefaultGraph,
                    ))?;
                    stats.relations += 1;
                } else {
                    let object = Literal::new_simple_literal(s);
                    store.insert(&Quad::new(
                        subject.clone(),
                        predicate.clone(),
                        object,
                        GraphName::DefaultGraph,
                    ))?;
                }
            }
            Value::Number(_) | Value::Bool(_) => {
                if let Some(object) = Self::jsonld_scalar_literal(value) {
                    store.insert(&Quad::new(
                        subject.clone(),
                        predicate.clone(),
                        object,
                        GraphName::DefaultGraph,
                    ))?;
                }
            }
            Value::Object(obj) => {
                // @set 只是数组容器：逐个展开，不产生额外节点。
                if let Some(set) = obj.get("@set") {
                    if let Some(items) = set.as_array() {
                        for item in items {
                            self.process_property_value(
                                store, subject, predicate, item, coercion, context, stats,
                            )?;
                        }
                    } else {
                        self.process_property_value(
                            store, subject, predicate, set, coercion, context, stats,
                        )?;
                    }
                    return Ok(());
                }
                // @list：编码为 RDF 集合（rdf:first / rdf:rest / rdf:nil）。
                if let Some(list) = obj.get("@list") {
                    let items: Vec<Value> = match list {
                        Value::Array(items) => items.clone(),
                        other => vec![other.clone()],
                    };
                    self.process_jsonld_list(
                        store, subject, predicate, &items, coercion, context, stats,
                    )?;
                    return Ok(());
                }
                // 处理嵌套对象
                if let Some(id) = obj.get("@id") {
                    if let Some(uri) = id.as_str() {
                        let object = Self::jsonld_node(uri, context)?;
                        store.insert(&Quad::new(
                            subject.clone(),
                            predicate.clone(),
                            object,
                            GraphName::DefaultGraph,
                        ))?;
                        // 对象属性指向另一实体（命名节点）时计一条关系
                        stats.relations += 1;
                        // 嵌套对象除 @id 外还带类型/属性时，它本身也是一个节点
                        // （JSON-LD 展开语义），继续落库而不是丢掉这些描述。
                        if obj.keys().any(|key| key != "@id" && key != "@context") {
                            self.process_jsonld_object(store, obj, context, stats)?;
                        }
                    }
                } else if let Some(val) = obj.get("@value") {
                    // 显式 @type / @language 优先；没有时按 JSON 原生标量定型
                    // （数字 → xsd:integer/xsd:double，布尔 → xsd:boolean）。
                    let literal = if let Some(datatype) = obj.get("@type") {
                        if let Some(datatype_uri) = datatype.as_str() {
                            let value_text = match val {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            let dt =
                                NamedNode::new(Self::expand_jsonld_term(datatype_uri, context))?;
                            Some(Literal::new_typed_literal(&value_text, dt))
                        } else {
                            Self::jsonld_scalar_literal(val)
                        }
                    } else if let Some(lang) = obj.get("@language") {
                        if let Some(lang_tag) = lang.as_str() {
                            let value_text = match val {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            Some(Literal::new_language_tagged_literal(&value_text, lang_tag)?)
                        } else {
                            Self::jsonld_scalar_literal(val)
                        }
                    } else {
                        Self::jsonld_scalar_literal(val)
                    };

                    if let Some(literal) = literal {
                        store.insert(&Quad::new(
                            subject.clone(),
                            predicate.clone(),
                            literal,
                            GraphName::DefaultGraph,
                        ))?;
                    }
                }
            }
            _ => {
                // 其他类型转换为字符串
                let object = Literal::new_simple_literal(&value.to_string());
                store.insert(&Quad::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                    GraphName::DefaultGraph,
                ))?;
            }
        }
        Ok(())
    }

    /// 把 `@list` 编码为 RDF 集合，并接到当前主语上。
    ///
    /// 单元格使用内容哈希命名：同一文档重复导入仍指向同一组空白节点，
    /// 不会因为每次生成随机空白节点而破坏幂等。
    fn process_jsonld_list(
        &self,
        store: &Store,
        subject: &NamedOrBlankNode,
        predicate: &NamedNode,
        items: &[Value],
        coercion: Option<NodeReferenceCoercion>,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // JSON-LD 规定 null 不产生 RDF 项：先剔除，避免把「空位」编码进链表。
        let items: Vec<Value> = items
            .iter()
            .filter(|item| !item.is_null())
            .cloned()
            .collect();
        let items = items.as_slice();
        let rdf_nil = NamedNode::new(format!("{}nil", RDF_NS))?;
        if items.is_empty() {
            store.insert(&Quad::new(
                subject.clone(),
                predicate.clone(),
                rdf_nil,
                GraphName::DefaultGraph,
            ))?;
            return Ok(());
        }
        let rdf_first = NamedNode::new(format!("{}first", RDF_NS))?;
        let rdf_rest = NamedNode::new(format!("{}rest", RDF_NS))?;
        let cells: Vec<NamedOrBlankNode> = items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                let material = format!("{}|{}|{}|{}", subject, predicate.as_str(), index, item);
                for byte in material.as_bytes() {
                    hash ^= u64::from(*byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
                BlankNode::new(format!("l{:016x}", hash))
                    .expect("生成的集合单元格标签必须合法")
                    .into()
            })
            .collect();
        store.insert(&Quad::new(
            subject.clone(),
            predicate.clone(),
            cells[0].clone(),
            GraphName::DefaultGraph,
        ))?;
        for (index, cell) in cells.iter().enumerate() {
            let item =
                self.process_jsonld_list_item(store, &items[index], coercion, context, stats)?;
            store.insert(&Quad::new(
                cell.clone(),
                rdf_first.clone(),
                item,
                GraphName::DefaultGraph,
            ))?;
            let rest: NamedOrBlankNode = cells
                .get(index + 1)
                .cloned()
                .unwrap_or_else(|| rdf_nil.clone().into());
            store.insert(&Quad::new(
                cell.clone(),
                rdf_rest.clone(),
                rest,
                GraphName::DefaultGraph,
            ))?;
        }
        Ok(())
    }

    /// 集合项：节点引用保持节点身份，值对象按字面量落库。
    fn process_jsonld_list_item(
        &self,
        store: &Store,
        item: &Value,
        coercion: Option<NodeReferenceCoercion>,
        context: &JsonLdContext,
        stats: &mut JsonldImportStats,
    ) -> Result<Term, Box<dyn std::error::Error + Send + Sync>> {
        match item {
            Value::String(text) => {
                if let Some(coercion) = coercion {
                    Ok(Self::jsonld_coerced_node(text, coercion, context)?.into())
                } else {
                    Ok(Literal::new_simple_literal(text).into())
                }
            }
            Value::Number(_) | Value::Bool(_) => Ok(Self::jsonld_scalar_literal(item)
                .map(Term::from)
                .unwrap_or_else(|| Literal::new_simple_literal(&item.to_string()).into())),
            Value::Object(obj) => {
                if let Some(id) = obj.get("@id").and_then(Value::as_str) {
                    if id.is_empty() {
                        return Ok(Literal::new_simple_literal("").into());
                    }
                    let node = Self::jsonld_node(id, context)?;
                    // 带描述的对象同时作为独立节点落库。
                    if obj.keys().any(|key| key != "@id" && key != "@context") {
                        self.process_jsonld_object(store, obj, context, stats)?;
                    }
                    return Ok(node.into());
                }
                if let Some(value) = obj.get("@value") {
                    let text = match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                    if let Some(datatype) = obj.get("@type").and_then(Value::as_str) {
                        let datatype = NamedNode::new(Self::expand_jsonld_term(datatype, context))?;
                        return Ok(Literal::new_typed_literal(&text, datatype).into());
                    }
                    if let Some(language) = obj.get("@language").and_then(Value::as_str) {
                        return Ok(Literal::new_language_tagged_literal(&text, language)?.into());
                    }
                    return Ok(Self::jsonld_scalar_literal(value)
                        .map(Term::from)
                        .unwrap_or_else(|| Literal::new_simple_literal(&text).into()));
                }
                Ok(Literal::new_simple_literal(&item.to_string()).into())
            }
            other => Ok(Literal::new_simple_literal(&other.to_string()).into()),
        }
    }

    pub async fn import_jsonld_from_file(
        &self,
        file_path: &str,
    ) -> Result<JsonldImportStats, Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(file_path)?;
        self.import_jsonld(&content).await
    }

    /// 导入 JSON-LD 文件并记入导入情景（默认图 + 批次命名图 + 元数据图）。
    pub async fn import_jsonld_file_batch(
        &self,
        file_path: &str,
    ) -> Result<JsonldEpisodeReceipt, Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(file_path)?;
        self.import_jsonld_batch(&content, file_path).await
    }

    /// 导入 JSON-LD 内容并记入导入情景。
    ///
    /// 与 `import_jsonld` 共用同一套展开逻辑（含 @context、@reverse、@list
    /// 与 @graph），但成功写入后会保留本批断言与确定性 gist，供「这批 /
    /// 上一批 / 来源文件」回指。JSON-LD 的 @graph 不是 RDF 命名图，因此
    /// `input_named_graphs` 恒为 0。
    pub async fn import_jsonld_batch(
        &self,
        jsonld_content: &str,
        source: &str,
    ) -> Result<JsonldEpisodeReceipt, Box<dyn std::error::Error + Send + Sync>> {
        let (quads, mut stats) = self.stage_jsonld(jsonld_content).await?;
        let blank_nodes = crate::import_episode::has_blank_nodes(&quads);
        let outcome = self
            .write_episode_batch(quads, "JSON-LD".to_string(), source, 0, blank_nodes)
            .await?;
        stats.triples = outcome.added;
        Ok(JsonldEpisodeReceipt {
            stats,
            episode: outcome.episode,
            recorded: outcome.recorded,
            source_aliases_added: outcome.source_aliases_added,
        })
    }

    /// 按名称识别 RDF 序列化格式，供标准输入等没有扩展名的场景使用。
    pub fn parse_rdf_format(
        name: &str,
    ) -> Result<oxigraph::io::RdfFormat, Box<dyn std::error::Error + Send + Sync>> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ttl" | "turtle" => Ok(oxigraph::io::RdfFormat::Turtle),
            "nt" | "ntriples" | "n-triples" => Ok(oxigraph::io::RdfFormat::NTriples),
            "nq" | "nquads" | "n-quads" => Ok(oxigraph::io::RdfFormat::NQuads),
            "trig" => Ok(oxigraph::io::RdfFormat::TriG),
            "rdf" | "xml" | "rdfxml" | "rdf/xml" => Ok(oxigraph::io::RdfFormat::RdfXml),
            _ => Err(format!(
                "无法识别 RDF 格式「{}」（支持 ttl / nt / nq / trig / rdf / xml）",
                name
            )
            .into()),
        }
    }

    /// 从文件扩展名识别 RDF 序列化格式（ttl / nt / nq / trig / rdf / xml）。
    ///
    /// JSON-LD 不在此列：请使用 `import_jsonld` / `import_jsonld_from_file`，
    /// 它们支持 `@context` 展开与更宽松的文档结构。
    pub fn detect_rdf_format(
        file_path: &str,
    ) -> Result<oxigraph::io::RdfFormat, Box<dyn std::error::Error + Send + Sync>> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        match extension.to_ascii_lowercase().as_str() {
            "json" | "jsonld" => Err(format!(
                "「{}」是 JSON-LD 文件：请改用 import-jsonld（或交互命令「导入jsonld文件：」）",
                file_path
            )
            .into()),
            _ => oxigraph::io::RdfFormat::from_extension(extension).ok_or_else(|| {
                format!(
                    "无法从扩展名「{}」识别 RDF 格式（支持 ttl / nt / nq / trig / rdf / xml）",
                    if extension.is_empty() {
                        "（无）"
                    } else {
                        extension
                    }
                )
                .into()
            }),
        }
    }

    /// 从 RDF 文件导入三元组（支持 Turtle / N-Triples / N-Quads / TriG / RDF/XML），
    /// 返回本次新增的三元组数（与库中已有三元组重复者不计入）。
    pub async fn import_rdf_file(
        &self,
        file_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let format = Self::detect_rdf_format(file_path)?;
        let file = fs::File::open(file_path)?;
        self.import_rdf_reader(format, file).await
    }

    /// 从任意 reader 导入 RDF 到默认图，返回新增三元组数。
    ///
    /// N-Quads / TriG 等数据集格式里的命名图不会被保留：所有三元组统一并入
    /// 默认图。这样 `import-rdf` 与只导出默认图的 `export-rdf` 语义对称，
    /// 导入的知识也能立即被推理器与 SPARQL 默认图查询看到；需要保留来源
    /// 隔离时请使用 `namespace` 命令。
    ///
    /// 与 JSON-LD 导入一致：持锁后、写前推进知识版本，保证并发读者不会把
    /// 半成品缓存成完整快照；解析或写入失败时版本号也已推进（保守失效）。
    pub async fn import_rdf_reader<R: std::io::Read>(
        &self,
        format: oxigraph::io::RdfFormat,
        reader: R,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // 先完整解析再写入：语法错误不会留下半批三元组。空白节点重命名
        // 与 Store::load_from_reader 一致，避免不同文档的 `_:b0` 相互碰撞。
        let quads: Vec<Quad> = oxigraph::io::RdfParser::from_format(format)
            .rename_blank_nodes()
            .for_reader(reader)
            .map(|quad| {
                quad.map(|quad| {
                    Quad::new(
                        quad.subject,
                        quad.predicate,
                        quad.object,
                        GraphName::DefaultGraph,
                    )
                })
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("RDF 解析失败：{}", error).into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let store = self.store.lock().await;
        self.bump_revision();
        let added = store.transaction(move |mut transaction| {
            let mut added = 0usize;
            for quad in &quads {
                if transaction.insert(quad.as_ref())? {
                    added += 1;
                }
            }
            Ok::<usize, oxigraph::store::StorageError>(added)
        })?;
        Ok(added)
    }

    /// 从 RDF 文件导入并记录导入情景（默认图 + 批次命名图 + 元数据图）。
    pub async fn import_rdf_file_batch(
        &self,
        file_path: &str,
    ) -> Result<crate::import_episode::RdfImportReceipt, Box<dyn std::error::Error + Send + Sync>>
    {
        let format = Self::detect_rdf_format(file_path)?;
        let file = fs::File::open(file_path)?;
        self.import_rdf_batch(format, file, file_path).await
    }

    /// 从任意 reader 导入 RDF，并把本次导入提升为可回指、可审计的导入情景。
    ///
    /// 断言仍全部并入默认图（与 `import_rdf_reader` 一致，旧查询不受影响），
    /// 同时写入：
    /// - 批次命名图 `kg:graphs/import-batches/{id}`：本批压平后的断言；
    /// - 元数据图 `kg:graphs/import-episodes`：序号、来源、格式、计数与 gist。
    ///
    /// 批次 id 由内容寻址；同一内容重复导入不会新建批次记录，也不会重复写
    /// 元数据。空白节点因解析时会重命名，是幂等的已知例外，回执会显式披露。
    pub async fn import_rdf_batch<R: std::io::Read>(
        &self,
        format: oxigraph::io::RdfFormat,
        reader: R,
        source: &str,
    ) -> Result<crate::import_episode::RdfImportReceipt, Box<dyn std::error::Error + Send + Sync>>
    {
        // 先完整解析：语法错误不会留下半批三元组。
        let parsed: Vec<Quad> = oxigraph::io::RdfParser::from_format(format)
            .rename_blank_nodes()
            .for_reader(reader)
            .map(|quad| {
                quad.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("RDF 解析失败：{}", error).into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_named_graphs: BTreeSet<String> = parsed
            .iter()
            .filter_map(|quad| match &quad.graph_name {
                GraphName::DefaultGraph => None,
                GraphName::NamedNode(node) => Some(node.as_str().to_string()),
                GraphName::BlankNode(node) => Some(format!("_:{}", node.as_str())),
            })
            .collect();
        let blank_nodes = crate::import_episode::has_blank_nodes(&parsed);
        let flattened: Vec<Quad> = parsed
            .into_iter()
            .map(|quad| {
                Quad::new(
                    quad.subject,
                    quad.predicate,
                    quad.object,
                    GraphName::DefaultGraph,
                )
            })
            .collect();

        let outcome = self
            .write_episode_batch(
                flattened,
                crate::import_episode::rdf_format_label(format).to_string(),
                source,
                input_named_graphs.len(),
                blank_nodes,
            )
            .await?;
        self.perf.bump("rdf_import_batch");
        Ok(crate::import_episode::RdfImportReceipt {
            episode: outcome.episode,
            added: outcome.added,
            duplicates: outcome.duplicates,
            recorded: outcome.recorded,
            source_aliases_added: outcome.source_aliases_added,
        })
    }

    /// 把一批已压平的四元组写入默认图、批次命名图与情景元数据图。
    ///
    /// RDF 与 JSON-LD 导入共用：内容寻址、同批次来源别名、幂等回执与
    /// gist 全部在这里完成，避免两条导入路径出现语义漂移。
    async fn write_episode_batch(
        &self,
        flattened: Vec<Quad>,
        format_label: String,
        source: &str,
        input_named_graphs: usize,
        blank_nodes: bool,
    ) -> Result<EpisodeWriteOutcome, Box<dyn std::error::Error + Send + Sync>> {
        let id = crate::import_episode::batch_id(&flattened);
        let gist = crate::import_episode::build_gist(
            &flattened,
            crate::import_episode::DEFAULT_TOP_N,
            crate::import_episode::DEFAULT_SAMPLE_LIMIT,
        );
        let store = self.store.lock().await;
        let existing = Self::import_episodes_from_store(&store)?;
        let known_episode = existing.iter().find(|episode| episode.id == id).cloned();
        let alias_needed = match known_episode.as_ref() {
            Some(episode) => !episode.all_sources().iter().any(|known| known == source),
            None => false,
        };
        let old_metadata: Vec<Quad> = if alias_needed {
            let subject = Subject::NamedNode(NamedNode::new(
                crate::import_episode::episode_subject_iri(&id),
            )?);
            let graph =
                GraphName::NamedNode(NamedNode::new(crate::import_episode::EPISODE_LEDGER_GRAPH)?);
            store
                .quads_for_pattern(Some(subject.as_ref()), None, None, Some(graph.as_ref()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let seq = existing
            .iter()
            .map(|episode| episode.seq)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let batch_graph = crate::import_episode::batch_graph_iri(&id);
        let fresh_episode = crate::import_episode::ImportEpisode {
            seq,
            id: id.clone(),
            source: source.to_string(),
            sources: Vec::new(),
            format: format_label,
            imported_at: crate::import_episode::now_unix_seconds(),
            total: flattened.len(),
            added: 0,
            duplicates: 0,
            blank_nodes,
            input_named_graphs,
            batch_graph: batch_graph.clone(),
            gist,
        };
        let known_for_closure = known_episode.clone();
        let fallback_episode = fresh_episode.clone();
        let source_for_closure = source.to_string();

        let (added, duplicates, recorded_episode, alias_episode) = store
            .transaction(
                move |mut transaction| -> Result<
                    (
                        usize,
                        usize,
                        Option<crate::import_episode::ImportEpisode>,
                        Option<crate::import_episode::ImportEpisode>,
                    ),
                    ImportBatchTransactionError,
                > {
                    let mut added = 0usize;
                    let mut duplicates = 0usize;
                    for quad in &flattened {
                        if transaction.insert(quad.as_ref())? {
                            added += 1;
                        } else {
                            duplicates += 1;
                        }
                    }
                    let batch_graph_node = GraphName::NamedNode(
                        NamedNode::new(&batch_graph)
                            .map_err(|error| ImportBatchTransactionError(error.to_string()))?,
                    );
                    for quad in &flattened {
                        let batch_quad = Quad::new(
                            quad.subject.clone(),
                            quad.predicate.clone(),
                            quad.object.clone(),
                            batch_graph_node.clone(),
                        );
                        transaction.insert(batch_quad.as_ref())?;
                    }
                    let mut recorded_episode = None;
                    let mut alias_episode = None;
                    if let Some(known) = known_for_closure.clone() {
                        if alias_needed {
                            let mut episode = known;
                            episode.record_source(&source_for_closure);
                            for quad in &old_metadata {
                                transaction.remove(quad.as_ref())?;
                            }
                            let metadata = crate::import_episode::episode_quads(&episode)
                                .map_err(|error| ImportBatchTransactionError(error.to_string()))?;
                            for quad in metadata {
                                transaction.insert(quad.as_ref())?;
                            }
                            alias_episode = Some(episode);
                        }
                    } else {
                        let mut episode = fresh_episode.clone();
                        episode.added = added;
                        episode.duplicates = duplicates;
                        let metadata = crate::import_episode::episode_quads(&episode)
                            .map_err(|error| ImportBatchTransactionError(error.to_string()))?;
                        for quad in metadata {
                            transaction.insert(quad.as_ref())?;
                        }
                        recorded_episode = Some(episode);
                    }
                    Ok((added, duplicates, recorded_episode, alias_episode))
                },
            )
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> { Box::new(error) })?;

        // 只在实际改变默认图时推进知识版本；批次元数据与命名图不污染快照缓存。
        if added > 0 {
            self.bump_revision();
        }
        let recorded = recorded_episode.is_some();
        let source_aliases_added = alias_episode.is_some();
        let episode = recorded_episode
            .or(alias_episode)
            .or(known_episode)
            .unwrap_or(fallback_episode);
        Ok(EpisodeWriteOutcome {
            added,
            duplicates,
            episode,
            recorded,
            source_aliases_added,
        })
    }

    /// 读取全部 RDF 导入情景（按 seq 升序）。
    pub async fn load_import_episodes(
        &self,
    ) -> Result<Vec<crate::import_episode::ImportEpisode>, Box<dyn std::error::Error + Send + Sync>>
    {
        let store = self.store.lock().await;
        self.perf.bump("import_episode_load");
        Self::import_episodes_from_store(&store)
    }

    fn import_episodes_from_store(
        store: &Store,
    ) -> Result<Vec<crate::import_episode::ImportEpisode>, Box<dyn std::error::Error + Send + Sync>>
    {
        let graph = NamedNode::new(crate::import_episode::EPISODE_LEDGER_GRAPH)?;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(GraphName::NamedNode(graph).as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(crate::import_episode::parse_episode_graph(
            quads.into_iter(),
        ))
    }

    /// 回答带批次线索的问题：先解析批次，再按意图从 gist 或批次命名图取数。
    ///
    /// 只有出现批次线索（这批 / 上一批 / 第 N 批 / 来源文件 / 批次 id）时才返回
    /// `Some`；普通问题返回 `None`，仍走全局图谱，避免最近导入偏置。
    pub async fn answer_import_episode_question(
        &self,
        text: &str,
        episodes: &[crate::import_episode::ImportEpisode],
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let Some(episode) = crate::import_episode::resolve_episode_reference(text, episodes) else {
            return Ok(None);
        };
        let question = crate::import_episode::classify_episode_question(text);
        let mentions_scope =
            text.contains("关于") || text.contains("批里") || text.contains("批中");
        let wants_subject = matches!(
            question,
            crate::import_episode::EpisodeQuestion::SubjectFacts
        ) || (mentions_scope
            && matches!(
                question,
                crate::import_episode::EpisodeQuestion::Brief
                    | crate::import_episode::EpisodeQuestion::Types
                    | crate::import_episode::EpisodeQuestion::Predicates
            ));
        if wants_subject {
            if let Some(answer) = self.batch_subject_facts_answer(episode, text).await? {
                return Ok(Some(answer));
            }
            if matches!(
                question,
                crate::import_episode::EpisodeQuestion::SubjectFacts
            ) {
                return Ok(Some(crate::import_episode::render_episode_missing_subject(
                    episode,
                )));
            }
        }
        if let crate::import_episode::EpisodeQuestion::TypeEntities(focus) = question {
            if let Some(answer) = self.batch_type_entities_answer(episode, focus).await? {
                return Ok(Some(answer));
            }
            return Ok(Some(crate::import_episode::render_episode_missing_type(
                episode, focus,
            )));
        }
        Ok(Some(crate::import_episode::render_episode_answer(
            episode, question,
        )))
    }

    /// 从批次命名图列出某类实体的有界名单（类型标签按焦点别名匹配）。
    async fn batch_type_entities_answer(
        &self,
        episode: &crate::import_episode::ImportEpisode,
        focus: crate::import_episode::TypeFocus,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(&episode.batch_graph)?);
        let rdf_type = NamedNode::new(RDF_TYPE_IRI)?;
        let store = self.store.lock().await;
        let quads = store
            .quads_for_pattern(None, Some(rdf_type.as_ref()), None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut by_type: std::collections::BTreeMap<String, BTreeSet<String>> =
            std::collections::BTreeMap::new();
        for quad in &quads {
            if !matches!(quad.subject, Subject::NamedNode(_)) {
                continue;
            }
            let Term::NamedNode(type_node) = &quad.object else {
                continue;
            };
            by_type
                .entry(crate::import_episode::display_named_node(type_node))
                .or_default()
                .insert(crate::import_episode::display_subject(&quad.subject));
        }
        let mut labels = Vec::new();
        let mut entities = BTreeSet::new();
        for (label, subjects) in &by_type {
            if crate::import_episode::type_label_matches(label, focus) {
                labels.push(label.clone());
                entities.extend(subjects.iter().cloned());
            }
        }
        if labels.is_empty() {
            return Ok(None);
        }
        let total = entities.len();
        let shown: Vec<String> = entities
            .into_iter()
            .take(crate::import_episode::DEFAULT_ENTITY_LIMIT)
            .collect();
        Ok(Some(crate::import_episode::render_episode_type_entities(
            episode,
            &labels.join(" / "),
            total,
            &shown,
        )))
    }

    /// 从批次命名图回答单主体问题：匹配问题中最长的已知实体名，返回出入边。
    async fn batch_subject_facts_answer(
        &self,
        episode: &crate::import_episode::ImportEpisode,
        text: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(&episode.batch_graph)?);
        let store = self.store.lock().await;
        let quads = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut best: Option<(usize, NamedNode)> = None;
        for quad in &quads {
            let mut candidates: Vec<&NamedNode> = Vec::new();
            if let Subject::NamedNode(node) = &quad.subject {
                candidates.push(node);
            }
            if quad.predicate.as_str() != RDF_TYPE_IRI {
                if let Term::NamedNode(node) = &quad.object {
                    candidates.push(node);
                }
            }
            for node in candidates {
                let name = crate::import_episode::display_named_node(node);
                let length = name.chars().count();
                if length < 2 || !text.contains(&name) {
                    continue;
                }
                let better = match best.as_ref() {
                    Some((best_length, _)) => length > *best_length,
                    None => true,
                };
                if better {
                    best = Some((length, node.clone()));
                }
            }
        }
        let Some((_, node)) = best else {
            return Ok(None);
        };
        let subject = Subject::NamedNode(node.clone());
        let mut facts = Vec::new();
        let mut seen = BTreeSet::new();
        for quad in store
            .quads_for_pattern(Some(subject.as_ref()), None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?
        {
            push_unique_episode_fact(&mut facts, &mut seen, &quad);
        }
        let object = Term::NamedNode(node.clone());
        for quad in store
            .quads_for_pattern(None, None, Some(object.as_ref()), Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?
        {
            push_unique_episode_fact(&mut facts, &mut seen, &quad);
        }
        facts.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
        });
        let total = facts.len();
        facts.truncate(crate::import_episode::DEFAULT_ENTITY_LIMIT);
        Ok(Some(crate::import_episode::render_episode_subject_facts(
            episode,
            &crate::import_episode::display_named_node(&node),
            total,
            &facts,
        )))
    }

    /// 执行任意 SPARQL 1.1 查询（SELECT / ASK / CONSTRUCT / DESCRIBE）。
    ///
    /// 这是命令行 `sparql` 子命令与交互模式「SPARQL：…」的公共实现：
    /// - `SELECT`：每行解一条记录（变量名 → JSON 值，未绑定变量缺席）；
    /// - `ASK`：单行 `boolean`；
    /// - `CONSTRUCT` / `DESCRIBE`：每行一条三元组（`subject`/`predicate`/`object`）。
    ///
    /// 查询只读、不推进知识版本；语法/执行错误按错误返回，绝不静默为空结果。
    pub async fn query_sparql(
        &self,
        query_text: &str,
    ) -> Result<QueryResult, Box<dyn std::error::Error + Send + Sync>> {
        self.perf.bump("sparql_query");
        let query = Query::parse(query_text, None).map_err(
            |error| -> Box<dyn std::error::Error + Send + Sync> {
                // sparql 语法错误的多行期望 token 列表很长：只保留首行并限长，
                // 位置信息（行:列）已足够定位，终端不被刷屏。
                let text = error.to_string();
                let first_line = text.lines().next().unwrap_or("未知错误");
                let mut detail: String = first_line.chars().take(240).collect();
                if first_line.chars().count() > 240 {
                    detail.push('…');
                }
                format!("SPARQL 语法错误：{}", detail).into()
            },
        )?;
        let store = self.store.lock().await;
        let results =
            store
                .query(query)
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("SPARQL 执行失败：{}", error).into()
                })?;
        let mut data: Vec<HashMap<String, Value>> = Vec::new();
        match results {
            QueryResults::Solutions(solutions) => {
                for solution in solutions {
                    let solution =
                        solution.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("SPARQL 求解失败：{}", error).into()
                        })?;
                    let mut row = HashMap::new();
                    for (variable, term) in solution.iter() {
                        row.insert(variable.as_str().to_string(), term_to_json(term));
                    }
                    data.push(row);
                }
            }
            QueryResults::Boolean(value) => {
                data.push(HashMap::from([("boolean".to_string(), Value::Bool(value))]));
            }
            QueryResults::Graph(triples) => {
                for triple in triples {
                    let triple =
                        triple.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("SPARQL 图求解失败：{}", error).into()
                        })?;
                    data.push(HashMap::from([
                        (
                            "subject".to_string(),
                            term_to_json(&Term::from(triple.subject)),
                        ),
                        (
                            "predicate".to_string(),
                            term_to_json(&Term::from(triple.predicate)),
                        ),
                        ("object".to_string(), term_to_json(&triple.object)),
                    ]));
                }
            }
        }
        Ok(QueryResult {
            success: true,
            data,
        })
    }

    /// 从文件扩展名识别导出格式（ttl / nt / nq / trig / rdf / xml / jsonld）。
    ///
    /// 与 [`Self::detect_rdf_format`] 不同，JSON-LD 是合法导出格式
    /// （oxigraph 序列化器支持 Streaming Profile）。
    pub fn detect_export_format(
        file_path: &str,
    ) -> Result<oxigraph::io::RdfFormat, Box<dyn std::error::Error + Send + Sync>> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        match extension.to_ascii_lowercase().as_str() {
            "json" | "jsonld" => Ok(oxigraph::io::RdfFormat::JsonLd {
                profile: oxigraph::io::JsonLdProfileSet::empty(),
            }),
            _ => oxigraph::io::RdfFormat::from_extension(extension).ok_or_else(|| {
                format!(
                    "无法从扩展名「{}」识别 RDF 格式（支持 ttl / nt / nq / trig / rdf / xml / jsonld）",
                    if extension.is_empty() { "（无）" } else { extension }
                )
                .into()
            }),
        }
    }

    /// 把默认图导出为 RDF 文件，返回导出的三元组数。
    ///
    /// 导出只读、不推进知识版本；命名图不在导出范围内（与 RDF 导入的默认图
    /// 语义对称）。
    pub async fn export_rdf_file(
        &self,
        file_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let format = Self::detect_export_format(file_path)?;
        let file = fs::File::create(file_path)?;
        self.export_rdf_to_writer(format, file).await
    }

    /// 把默认图序列化到任意 writer，返回写出的三元组数。
    pub async fn export_rdf_to_writer<W: std::io::Write>(
        &self,
        format: oxigraph::io::RdfFormat,
        writer: W,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        self.perf.bump("rdf_export");
        let count = store
            .quads_for_pattern(None, None, None, Some(GraphNameRef::DefaultGraph))
            .count();
        store
            .dump_graph_to_writer(GraphNameRef::DefaultGraph, format, writer)
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                format!("RDF 导出失败：{}", error).into()
            })?;
        Ok(count)
    }

    pub async fn save_knowledge(
        &self,
        items: Vec<KnowledgeItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.save_knowledge_with_receipt(items).await.map(|_| ())
    }

    /// 带回执的知识保存：谓词准入、别名/逆向归一、去重合并、功能谓词冲突
    /// 标记（不覆盖）、撤销动作记录。返回的回执可交给 `undo_import` 撤回。
    pub async fn save_knowledge_with_receipt(
        &self,
        items: Vec<KnowledgeItem>,
    ) -> Result<predicate_registry::ImportReceipt, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;

        let mut receipt = predicate_registry::ImportReceipt::default();
        for item in items {
            match item {
                KnowledgeItem::Entity { label, properties } => {
                    self.save_entity_with_receipt(&store, &label, &properties, &mut receipt)?;
                }
                KnowledgeItem::Relationship {
                    from_id,
                    to_id,
                    rel_type,
                    ..
                } => {
                    self.save_relationship_with_receipt(
                        &store,
                        &from_id,
                        &to_id,
                        &rel_type,
                        &mut receipt,
                    )?;
                }
            }
        }
        // 只有确有写入才推进版本
        if receipt.added > 0 {
            self.bump_revision();
        }
        Ok(receipt)
    }

    fn save_entity_with_receipt(
        &self,
        store: &Store,
        label: &str,
        properties: &HashMap<String, String>,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let subject_name = properties
            .get("name")
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        // 图谱观测的 name 证据（属性迭代前先取，避免被排序视图遮蔽）
        let name_literal = properties.get("name").cloned();
        let subject_iri = crate::identity::entity_uri(BASE_NS, &subject_name);
        let subject = NamedNode::new(subject_iri.clone())?;
        let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
        let entity_type_iri = format!("{}{}", BASE_NS, label);
        let entity_type = NamedNode::new(entity_type_iri.clone())?;

        let type_quad = Quad::new(
            subject.clone(),
            rdf_type,
            entity_type,
            GraphName::DefaultGraph,
        );
        if store.contains(&type_quad)? {
            receipt.duplicates += 1;
        } else {
            store.insert(&type_quad)?;
            receipt.added += 1;
            receipt
                .actions
                .push(predicate_registry::ImportAction::Named {
                    subject: subject_iri.clone(),
                    predicate: format!("{}type", RDF_NS),
                    object: entity_type_iri,
                });
        }

        // 属性迭代顺序固定，回执与撤销动作可复现
        let mut properties: Vec<(&String, &String)> = properties.iter().collect();
        properties.sort();
        for (key, value) in properties {
            let predicate_iri = format!("{}{}", BASE_NS, key);
            let conflict = {
                let registry = self.predicate_registry.read();
                registry.conflict(&subject_name, key, value)
            };
            if let Some(existing) = conflict {
                receipt.conflicts.push(predicate_registry::ImportConflict {
                    subject: subject_name.clone(),
                    predicate: key.clone(),
                    existing,
                    incoming: value.clone(),
                });
                self.predicate_registry
                    .write()
                    .note_conflict(&subject_name, key, value);
                continue;
            }
            let predicate = NamedNode::new(predicate_iri.clone())?;
            let object = Literal::new_simple_literal(value);
            let quad = Quad::new(subject.clone(), predicate, object, GraphName::DefaultGraph);
            if store.contains(&quad)? {
                receipt.duplicates += 1;
                receipt.merged += 1;
            } else {
                store.insert(&quad)?;
                receipt.added += 1;
                receipt
                    .actions
                    .push(predicate_registry::ImportAction::Literal {
                        subject: subject_iri.clone(),
                        predicate: predicate_iri,
                        value: value.clone(),
                    });
                self.predicate_registry.write().observe(
                    &subject_name,
                    key,
                    value,
                    Some("知识导入"),
                );
            }
        }
        // 图谱观测：实体身份进入注册表（唯一自动确认来源；name 属性即别名证据）
        if let Some(name) = name_literal {
            let mut types = BTreeSet::new();
            types.insert(label.to_string());
            self.entity_registry.write().observe_kb_entity(
                &subject_name,
                &types,
                &[name],
                "文本导入",
            );
        }
        Ok(())
    }

    fn save_relationship_with_receipt(
        &self,
        store: &Store,
        from_id: &str,
        to_id: &str,
        rel_type: &str,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let from_id = from_id.trim();
        let to_id = to_id.trim();
        let rel_type = rel_type.trim();
        if from_id.is_empty() || to_id.is_empty() || rel_type.is_empty() {
            return Ok(());
        }

        // 谓词准入：未登记谓词先登记为「待确认」；严格准入下不落库
        let resolution = self.predicate_registry.read().resolve(rel_type);
        if !resolution.known {
            let strict = self.predicate_registry.read().strict_admission();
            if strict {
                if !receipt.pending_admissions.iter().any(|p| p == rel_type) {
                    receipt.pending_admissions.push(rel_type.to_string());
                }
                return Ok(());
            }
            let admitted = self.predicate_registry.write().admit(rel_type, "文本导入");
            if admitted && !receipt.provisional_predicates.iter().any(|p| p == rel_type) {
                receipt.provisional_predicates.push(rel_type.to_string());
                receipt.registry_changed = true;
            }
        }

        // 逆向谓词统一方向落库：徒弟(A,B) ≡ 师傅(B,A)
        let (subject_name, object_name) = if resolution.inverted {
            (to_id, from_id)
        } else {
            (from_id, to_id)
        };
        let predicate_name = resolution.canonical.clone();

        // 关系两端至少要有 name 字面量，保证按名称检索能找到它们
        self.ensure_entity_name(store, subject_name, receipt)?;
        self.ensure_entity_name(store, object_name, receipt)?;
        // 图谱观测：关系端点身份进入注册表（规范名即图谱键）
        {
            let empty_types = BTreeSet::new();
            let mut registry = self.entity_registry.write();
            registry.observe_kb_entity(
                subject_name,
                &empty_types,
                &[subject_name.to_string()],
                "文本导入",
            );
            registry.observe_kb_entity(
                object_name,
                &empty_types,
                &[object_name.to_string()],
                "文本导入",
            );
        }

        let conflict = {
            let registry = self.predicate_registry.read();
            registry.conflict(subject_name, &predicate_name, object_name)
        };
        if let Some(existing) = conflict {
            receipt.conflicts.push(predicate_registry::ImportConflict {
                subject: subject_name.to_string(),
                predicate: predicate_name.clone(),
                existing,
                incoming: object_name.to_string(),
            });
            self.predicate_registry.write().note_conflict(
                subject_name,
                &predicate_name,
                object_name,
            );
            return Ok(());
        }

        // 模式闸门：定义域/值域校验（实体类型未知时按开放世界放行）。
        // 类型体系开启闭包档时按别名/子类/互斥判定，并记录与精确集合判定的
        // 差异（误杀消解 / 收紧），差异只统计、不改变本次结果。
        let subject_types = entity_type_labels(store, subject_name)?;
        let object_types = entity_type_labels(store, object_name)?;
        let violation = {
            let registry = self.predicate_registry.read();
            let types = self.type_system.read();
            let exact = registry.check_schema(
                subject_name,
                &predicate_name,
                object_name,
                &subject_types,
                &object_types,
            );
            if types.closure_gate() {
                let closure = registry.check_schema_with_types(
                    subject_name,
                    &predicate_name,
                    object_name,
                    &subject_types,
                    &object_types,
                    &types,
                );
                if exact.is_some() && closure.is_none() {
                    receipt.gate_relaxed += 1;
                } else if exact.is_none() && closure.is_some() {
                    receipt.gate_tightened += 1;
                }
                closure
            } else {
                exact
            }
        };
        if let Some(violation) = violation {
            receipt.schema_violations.push(violation);
            return Ok(());
        }

        let subject_iri = crate::identity::entity_uri(BASE_NS, &subject_name);
        let object_iri = crate::identity::entity_uri(BASE_NS, &object_name);
        let predicate_iri = format!("{}{}", BASE_NS, predicate_name);
        let subject = NamedNode::new(subject_iri.clone())?;
        let predicate = NamedNode::new(predicate_iri.clone())?;
        let object = NamedNode::new(object_iri.clone())?;
        let quad = Quad::new(subject, predicate, object, GraphName::DefaultGraph);
        if store.contains(&quad)? {
            receipt.duplicates += 1;
        } else {
            store.insert(&quad)?;
            receipt.added += 1;
            receipt
                .actions
                .push(predicate_registry::ImportAction::Named {
                    subject: subject_iri,
                    predicate: predicate_iri,
                    object: object_iri,
                });
            self.predicate_registry.write().observe(
                subject_name,
                &predicate_name,
                object_name,
                Some("文本导入"),
            );
        }
        Ok(())
    }

    /// 确保实体有 name 字面量（辅助三元组，仅在缺失时插入并记录撤销动作）。
    fn ensure_entity_name(
        &self,
        store: &Store,
        name: &str,
        receipt: &mut predicate_registry::ImportReceipt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let subject_iri = crate::identity::entity_uri(BASE_NS, name);
        let predicate_iri = format!("{}name", BASE_NS);
        let subject = NamedNode::new(subject_iri.clone())?;
        let predicate = NamedNode::new(predicate_iri.clone())?;
        let quad = Quad::new(
            subject,
            predicate,
            Literal::new_simple_literal(name),
            GraphName::DefaultGraph,
        );
        if store.contains(&quad)? {
            return Ok(());
        }
        store.insert(&quad)?;
        receipt.added += 1;
        receipt
            .actions
            .push(predicate_registry::ImportAction::Literal {
                subject: subject_iri,
                predicate: predicate_iri,
                value: name.to_string(),
            });
        self.predicate_registry
            .write()
            .observe(name, "name", name, Some("文本导入"));
        Ok(())
    }

    /// 撤回一次导入：逐条移除回执中记录的三元组，并同步回撤画像观测。
    pub async fn undo_import(
        &self,
        receipt: &predicate_registry::ImportReceipt,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let mut removed = 0usize;
        for action in &receipt.actions {
            let quad = match action {
                predicate_registry::ImportAction::Named {
                    subject,
                    predicate,
                    object,
                } => Quad::new(
                    NamedNode::new(subject.clone())?,
                    NamedNode::new(predicate.clone())?,
                    NamedNode::new(object.clone())?,
                    GraphName::DefaultGraph,
                ),
                predicate_registry::ImportAction::Literal {
                    subject,
                    predicate,
                    value,
                } => Quad::new(
                    NamedNode::new(subject.clone())?,
                    NamedNode::new(predicate.clone())?,
                    Literal::new_simple_literal(value),
                    GraphName::DefaultGraph,
                ),
            };
            if store.remove(&quad).unwrap_or(false) {
                removed += 1;
            }
        }
        {
            let mut registry = self.predicate_registry.write();
            for action in &receipt.actions {
                match action {
                    predicate_registry::ImportAction::Named {
                        subject,
                        predicate,
                        object,
                    } => {
                        registry.retract(
                            &Self::local_kg_name(subject),
                            &Self::local_kg_name(predicate),
                            &Self::local_kg_name(object),
                        );
                    }
                    predicate_registry::ImportAction::Literal {
                        subject,
                        predicate,
                        value,
                    } => {
                        registry.retract(
                            &Self::local_kg_name(subject),
                            &Self::local_kg_name(predicate),
                            value,
                        );
                    }
                }
            }
        }
        // 只有确有条目被移除才推进版本；重复撤销是无操作
        if removed > 0 {
            self.bump_revision();
        }
        Ok(removed)
    }

    /// 更正一个字面量属性：删除旧值三元组、写入新值三元组，并同步谓词画像。
    ///
    /// 返回 `(removed, inserted)`。旧值必须能在图谱中找到才会写入新值：
    /// 「更正」是说旧值错了，不是无条件覆盖；找不到旧值就诚实地不改。
    pub async fn correct_object_value(
        &self,
        entity: &str,
        predicate: &str,
        old_value: &str,
        new_value: &str,
    ) -> Result<(usize, bool), Box<dyn std::error::Error + Send + Sync>> {
        let entity = entity.trim();
        let old_value = old_value.trim();
        let new_value = new_value.trim();
        if entity.is_empty() || predicate.trim().is_empty() || old_value.is_empty() {
            return Ok((0, false));
        }
        let (canonical, inverted) = {
            let registry = self.predicate_registry.read();
            let resolution = registry.resolve(predicate.trim());
            (resolution.canonical.clone(), resolution.inverted)
        };
        if inverted {
            return Ok((0, false));
        }

        let subject_iri = crate::identity::entity_uri(BASE_NS, entity);
        let predicate_iri = format!("{}{}", BASE_NS, canonical);
        let subject = NamedNode::new(subject_iri)?;
        let predicate_node = NamedNode::new(predicate_iri)?;
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(
                Some(subject.as_ref().into()),
                Some(predicate_node.as_ref().into()),
                None,
                None,
            )
            .collect::<Result<Vec<_>, _>>()?;
        let mut removed = 0usize;
        let mut had_literal = false;
        let mut had_iri = false;
        for quad in existing {
            let matched = match &quad.object {
                Term::Literal(literal) => literal.value() == old_value,
                Term::NamedNode(node) => {
                    node.as_str() == old_value || Self::local_kg_name(node.as_str()) == old_value
                }
                _ => false,
            };
            if !matched {
                continue;
            }
            if store.remove(&quad)? {
                removed += 1;
                match &quad.object {
                    Term::Literal(_) => had_literal = true,
                    Term::NamedNode(_) => had_iri = true,
                    _ => {}
                }
            }
        }
        if removed == 0 {
            return Ok((0, false));
        }
        let mut inserted = false;
        if !new_value.is_empty() {
            if had_literal {
                let new_quad = Quad::new(
                    subject.clone(),
                    predicate_node.clone(),
                    Literal::new_simple_literal(new_value),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&new_quad)? {
                    store.insert(&new_quad)?;
                    inserted = true;
                }
            }
            if had_iri {
                let object = NamedNode::new(crate::identity::entity_uri(BASE_NS, new_value))?;
                let new_quad = Quad::new(
                    subject,
                    predicate_node,
                    object.clone(),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&new_quad)? {
                    store.insert(&new_quad)?;
                    inserted = true;
                }
                // 快照只保留「两端都有 kg:name」的关系：新宾语必须自带名称，
                // 否则更正后的关系会在推理快照里凭空消失。
                let name_iri = format!("{}name", BASE_NS);
                let name_quad = Quad::new(
                    object,
                    NamedNode::new(name_iri)?,
                    Literal::new_simple_literal(new_value),
                    GraphName::DefaultGraph,
                );
                if !store.contains(&name_quad)? {
                    store.insert(&name_quad)?;
                    inserted = true;
                }
            }
        }
        drop(store);
        self.predicate_registry.write().correct(
            entity,
            &canonical,
            old_value,
            if new_value.is_empty() {
                old_value
            } else {
                new_value
            },
        );
        self.bump_revision();
        Ok((removed, inserted))
    }
}

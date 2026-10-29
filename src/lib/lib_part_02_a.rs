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

/// JSON-LD 匿名节点（没有 `@id` 的对象）的确定性命名作用域。
///
/// 文档指纹 + 节点序号 + 节点内容共同决定 IRI：同一文档重复导入得到同一节点，
/// 不会像随机 UUID 那样破坏幂等；不同文档/不同位置也不会相互碰撞。
struct JsonLdAnonymousScope {
    fingerprint: u64,
    next: std::cell::Cell<usize>,
}

impl JsonLdAnonymousScope {
    fn new(document: &Value) -> Self {
        let text = serde_json::to_string(document).unwrap_or_default();
        Self {
            fingerprint: fnv1a64(text.as_bytes()),
            next: std::cell::Cell::new(0),
        }
    }

    fn fresh(
        &self,
        obj: &serde_json::Map<String, Value>,
    ) -> Result<NamedNode, Box<dyn std::error::Error + Send + Sync>> {
        let index = self.next.get();
        self.next.set(index.saturating_add(1));
        let material = serde_json::to_string(obj).unwrap_or_default();
        let mut hash = self.fingerprint;
        hash = fnv1a64_update(hash, &index.to_le_bytes());
        hash = fnv1a64_update(hash, material.as_bytes());
        Ok(NamedNode::new(crate::identity::entity_uri(
            BASE_NS,
            &format!("anon-{:016x}", hash),
        ))?)
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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
            relevant_cache: parking_lot::Mutex::new(DeltaQualifiedCache::new(QUERY_CACHE_CAPACITY)),
            direct_answer_cache: parking_lot::Mutex::new(DeltaQualifiedCache::new(
                QUERY_CACHE_CAPACITY,
            )),
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

    /// 带增量足迹的版本推进：相关事实缓存按依赖逐条复核，而不是整体失忆。
    ///
    /// 调用方必须持有存储锁，并在实际写入完成之后调用（version-after-write），
    /// 这样并发读者绝不会把过期数据缓存到新版本号上。只有能给出完整增量足迹
    /// 的写入路径才应传入 `Some`；整体替换、结构迁移等只能传 `None` 全量清空。
    fn bump_revision_with(&self, delta: Option<DeltaFootprint>) {
        let revision = self
            .revision
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        self.perf.bump("revision_bump");
        let mut relevant_cache = self.relevant_cache.lock();
        let mut direct_cache = self.direct_answer_cache.lock();
        match delta {
            Some(mut delta) => {
                delta.generation = revision;
                let relevant_report = relevant_cache.observe(&delta);
                self.perf.bump_by(
                    "relevant_cache_delta_retained",
                    relevant_report.retained as u64,
                );
                self.perf.bump_by(
                    "relevant_cache_delta_incomplete",
                    relevant_report.marked_incomplete as u64,
                );
                self.perf.bump_by(
                    "relevant_cache_delta_invalidated",
                    relevant_report.invalidated as u64,
                );
                if relevant_report.gap {
                    self.perf.bump("relevant_cache_delta_gap");
                }
                if relevant_report.cleared_all {
                    self.perf.bump("relevant_cache_delta_cleared");
                }
                let direct_report = direct_cache.observe(&delta);
                self.perf.bump_by(
                    "direct_answer_cache_delta_retained",
                    direct_report.retained as u64,
                );
                self.perf.bump_by(
                    "direct_answer_cache_delta_invalidated",
                    direct_report.invalidated as u64,
                );
                if direct_report.gap {
                    self.perf.bump("direct_answer_cache_delta_gap");
                }
            }
            None => {
                if !relevant_cache.is_empty() {
                    self.perf.bump("relevant_cache_full_invalidations");
                }
                if !direct_cache.is_empty() {
                    self.perf.bump("direct_answer_cache_full_invalidations");
                }
                relevant_cache.clear();
                direct_cache.clear();
            }
        }
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
        let anonymous = JsonLdAnonymousScope::new(&jsonld_value);
        let mut stats = JsonldImportStats::default();
        self.process_jsonld_simple(&staging, &jsonld_value, &anonymous, &mut stats)
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
        let (added, duplicates, added_quads) = store.transaction(move |mut transaction| {
            let mut added = 0usize;
            let mut duplicates = 0usize;
            let mut added_quads = Vec::new();
            for quad in quads {
                if transaction.insert(quad.as_ref())? {
                    added += 1;
                    added_quads.push(quad.clone());
                } else {
                    duplicates += 1;
                }
            }
            Ok::<(usize, usize, Vec<Quad>), oxigraph::store::StorageError>((
                added,
                duplicates,
                added_quads,
            ))
        })?;
        if added > 0 {
            let delta = footprint_from_quads(&added_quads);
            self.bump_revision_with(Some(delta));
        }
        Ok((added, duplicates))
    }

    async fn process_jsonld_simple(
        &self,
        store: &Store,
        jsonld: &Value,
        anonymous: &JsonLdAnonymousScope,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 处理直接的 JSON-LD 结构
        if let Some(array) = jsonld.as_array() {
            for item in array {
                if let Some(obj) = item.as_object() {
                    // 处理每个对象
                    let context = Self::extract_jsonld_context(obj);
                    self.process_jsonld_object(store, obj, &context, anonymous, stats)?;
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
                            self.process_jsonld_object(store, inner, &context, anonymous, stats)?;
                        }
                    }
                } else if let Some(inner) = graph.as_object() {
                    self.process_jsonld_object(store, inner, &context, anonymous, stats)?;
                }
            } else {
                // 单个对象
                self.process_jsonld_object(store, obj, &context, anonymous, stats)?;
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
                if term == "@language" {
                    if let Some(language) = value.as_str() {
                        context.default_language = Some(language.to_string());
                    }
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
        anonymous: &JsonLdAnonymousScope,
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
                self.process_jsonld_graph(store, graph, context, anonymous, stats)?;
            }
            if let Some(included) = included {
                self.process_jsonld_graph(store, included, context, anonymous, stats)?;
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
            anonymous.fresh(obj)?.into()
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
                        store, &subject, &predicate, val, coercion, context, anonymous, stats,
                    )?;
                }
            } else {
                self.process_property_value(
                    store, &subject, &predicate, value, coercion, context, anonymous, stats,
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
                            store, &subject, &predicate, val, coercion, context, anonymous, stats,
                        )?;
                    }
                } else {
                    self.process_reverse_value(
                        store, &subject, &predicate, value, coercion, context, anonymous, stats,
                    )?;
                }
            }
        }

        // 带 @id/属性的节点也可以顺带承载命名图或 @included 节点；
        // 知识库只使用默认图，这里把其内容展开到默认图而不是丢弃。
        if let Some(graph) = obj.get("@graph") {
            self.process_jsonld_graph(store, graph, context, anonymous, stats)?;
        }
        if let Some(included) = obj.get("@included") {
            self.process_jsonld_graph(store, included, context, anonymous, stats)?;
        }

        Ok(())
    }

    /// 展开 `@graph` / `@included` 的值：接受单个节点或节点数组。
    fn process_jsonld_graph(
        &self,
        store: &Store,
        value: &Value,
        context: &JsonLdContext,
        anonymous: &JsonLdAnonymousScope,
        stats: &mut JsonldImportStats,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            Value::Array(items) => {
                for item in items {
                    if let Some(obj) = item.as_object() {
                        self.process_jsonld_object(store, obj, context, anonymous, stats)?;
                    }
                }
            }
            Value::Object(obj) => {
                self.process_jsonld_object(store, obj, context, anonymous, stats)?
            }
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
        anonymous: &JsonLdAnonymousScope,
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
            self.process_jsonld_object(store, obj, context, anonymous, stats)?;
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
        anonymous: &JsonLdAnonymousScope,
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
                    let object = match context.default_language.as_deref() {
                        Some(language) => Literal::new_language_tagged_literal(s, language)?.into(),
                        None => Literal::new_simple_literal(s),
                    };
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
                                store, subject, predicate, item, coercion, context, anonymous,
                                stats,
                            )?;
                        }
                    } else {
                        self.process_property_value(
                            store, subject, predicate, set, coercion, context, anonymous, stats,
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
                        store, subject, predicate, &items, coercion, context, anonymous, stats,
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
                            self.process_jsonld_object(store, obj, context, anonymous, stats)?;
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
                    } else if let (Value::String(text), Some(language)) =
                        (val, context.default_language.as_deref())
                    {
                        Some(Literal::new_language_tagged_literal(text, language)?)
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
        anonymous: &JsonLdAnonymousScope,
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
            let item = self.process_jsonld_list_item(
                store,
                &items[index],
                coercion,
                context,
                anonymous,
                stats,
            )?;
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
        anonymous: &JsonLdAnonymousScope,
        stats: &mut JsonldImportStats,
    ) -> Result<Term, Box<dyn std::error::Error + Send + Sync>> {
        match item {
            Value::String(text) => {
                if let Some(coercion) = coercion {
                    Ok(Self::jsonld_coerced_node(text, coercion, context)?.into())
                } else if let Some(language) = context.default_language.as_deref() {
                    Ok(Literal::new_language_tagged_literal(text, language)?.into())
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
                        self.process_jsonld_object(store, obj, context, anonymous, stats)?;
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
                    if value.is_string() {
                        if let Some(language) = context.default_language.as_deref() {
                            return Ok(
                                Literal::new_language_tagged_literal(&text, language)?.into()
                            );
                        }
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
}

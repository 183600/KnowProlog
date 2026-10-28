impl OxigraphManager {
    pub async fn load_all_relations(
        &self,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        // 使用整图快照缓存
        let snapshot = self.load_graph_snapshot().await?;
        Ok(snapshot
            .iter()
            .filter(|fact| matches!(fact, PrologFact::Relation { .. }))
            .cloned()
            .collect())
    }

    async fn find_relations_between(
        &self,
        store: &Store,
        entities: &std::collections::HashSet<String>,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let mut entity_filter = String::new();
        for (index, entity) in entities.iter().enumerate() {
            if index > 0 {
                entity_filter.push_str(", ");
            }
            entity_filter.push('"');
            entity_filter.push_str(&sparql_literal(entity));
            entity_filter.push('"');
        }

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?from_name ?to_name ?rel WHERE {{
                ?from kg:name ?from_name .
                ?to kg:name ?to_name .
                ?from ?rel ?to .
                FILTER(?from_name IN ({}))
                FILTER(?to_name IN ({}))
                FILTER(?rel != rdf:type)
                FILTER(?from != ?to)
            }}
        "#,
            entity_filter, entity_filter
        );

        self.query_relations(store, &query).await
    }

    async fn query_relations(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut relations = Vec::new();

        match Query::parse(query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let (Some(from), Some(to), Some(rel)) = (
                                    solution.get("from_name"),
                                    solution.get("to_name"),
                                    solution.get("rel"),
                                ) {
                                    let from_name = if let Term::Literal(lit) = from {
                                        lit.value().to_string()
                                    } else {
                                        from.to_string()
                                    };

                                    let to_name = if let Term::Literal(lit) = to {
                                        lit.value().to_string()
                                    } else {
                                        to.to_string()
                                    };

                                    let rel_str = match rel {
                                        Term::NamedNode(node) => node.as_str().to_string(),
                                        other => other.to_string(),
                                    };
                                    let rel_type = if rel_str.contains(BASE_NS) {
                                        rel_str
                                            .replace(BASE_NS, "")
                                            .trim_matches(|c| c == '<' || c == '>')
                                            .to_string()
                                    } else {
                                        rel_str.trim_matches(|c| c == '<' || c == '>').to_string()
                                    };

                                    relations.push(PrologFact::Relation {
                                        from: from_name,
                                        rel_type,
                                        to: to_name,
                                    });
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(relations)
    }

    /// 当前知识版本（写操作 version-after-write 自增；账本写入不推进版本）。
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// 同步规则账本：清空 `kg:graphs/rules` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 规则即数据：稳定 id + 规范逻辑形式（JSON）+ 可读词面 + 来源 + 置信度，
    /// 加上账本戳（规则集指纹 + 条数）。返回写入的规则条数。
    pub async fn sync_rule_ledger(
        &self,
        entries: &[graph_ledger::RuleLedgerEntry],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = graph_ledger::rule_quads(entries)?;
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::RULES_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("rule_ledger_sync");
        Ok(entries.len())
    }

    /// 读取规则账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_rule_ledger(
        &self,
    ) -> Result<graph_ledger::RuleLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::RULES_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("rule_ledger_load");
        Ok(graph_ledger::parse_rule_graph(quads.into_iter()))
    }

    /// 同步模式账本：清空 `kg:graphs/patterns` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 模式即数据：稳定 id + IR（JSON）+ 来源 + 路由/优先级，
    /// 加上模式集指纹戳。返回写入的模式条数。
    pub async fn sync_pattern_ledger(
        &self,
        entries: &[pattern_ir::PatternEntry],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = pattern_ir::pattern_quads(entries)?;
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::PATTERNS_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("pattern_ledger_sync");
        Ok(entries.len())
    }

    /// 读取模式账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_pattern_ledger(
        &self,
    ) -> Result<Vec<pattern_ir::PatternEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::PATTERNS_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("pattern_ledger_load");
        Ok(pattern_ir::parse_pattern_graph(quads.into_iter()))
    }

    /// 同步模式经验账本：清空 `kg:graphs/pattern-episodes` 后全量重写
    /// （幂等；不推进知识版本）。经验与模式定义分图存放，互不污染。
    pub async fn sync_pattern_episodes(
        &self,
        episodes: &[pattern_ir::PatternEpisode],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = pattern_ir::episode_quads(episodes)?;
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::EPISODES_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("pattern_episode_sync");
        Ok(episodes.len())
    }

    /// 读取模式经验账本（按经验 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_pattern_episodes(
        &self,
    ) -> Result<Vec<pattern_ir::PatternEpisode>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(pattern_ir::EPISODES_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("pattern_episode_load");
        Ok(pattern_ir::parse_episode_graph(quads.into_iter()))
    }

    /// 同步词法记忆：清空 `kg:graphs/lexical-memory` 后全量重写
    /// （幂等；不推进知识版本）。词法证据是账本，不是事实。
    pub async fn sync_lexical_memory(
        &self,
        memory: &lexical_memory::LexicalMemory,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = memory.to_quads()?;
        let graph = GraphName::NamedNode(NamedNode::new(lexical_memory::LEXICAL_MEMORY_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("lexical_memory_sync");
        Ok(memory.len())
    }

    /// 读取词法记忆（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_lexical_memory(
        &self,
    ) -> Result<lexical_memory::LexicalMemory, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(lexical_memory::LEXICAL_MEMORY_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("lexical_memory_load");
        Ok(lexical_memory::LexicalMemory::from_quads(quads.into_iter()))
    }

    /// 同步同义词图：清空 `kg:graphs/synonym-graph` 后全量重写
    /// （幂等；不推进知识版本）。词义证据是账本，不是事实。
    pub async fn sync_synonym_graph(
        &self,
        graph: &synonym_graph::SynonymGraph,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = graph.to_quads()?;
        let graph_name = GraphName::NamedNode(NamedNode::new(
            synonym_graph::SYNONYM_GRAPH_IRI,
        )?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph_name.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("synonym_graph_sync");
        Ok(graph.len())
    }

    /// 读取同义词图（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_synonym_graph(
        &self,
    ) -> Result<synonym_graph::SynonymGraph, Box<dyn std::error::Error + Send + Sync>> {
        let graph_name = GraphName::NamedNode(NamedNode::new(
            synonym_graph::SYNONYM_GRAPH_IRI,
        )?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph_name.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("synonym_graph_load");
        Ok(synonym_graph::SynonymGraph::from_quads(
            quads.into_iter(),
        ))
    }

    /// 物化派生账本：清空 `kg:graphs/derived` 后全量重写（幂等；不推进知识版本）。
    ///
    /// 每条派生事实带规则来源、前提、推导轮次、置信度与物化戳
    /// （知识版本 + 规则集指纹），过期与否可审计。
    pub async fn materialize_derived_ledger(
        &self,
        entries: &[graph_ledger::DerivedLedgerEntry],
        revision: u64,
        fingerprint: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = graph_ledger::derived_quads(entries, revision, fingerprint)?;
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::DERIVED_GRAPH)?);
        let store = self.store.lock().await;
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("derived_ledger_materialize");
        Ok(entries.len())
    }

    /// 读取派生账本（按条目 id 升序，确定性；不触碰任何缓存）。
    pub async fn load_derived_ledger(
        &self,
    ) -> Result<graph_ledger::DerivedLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(graph_ledger::DERIVED_GRAPH)?);
        let store = self.store.lock().await;
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.perf.bump("derived_ledger_load");
        Ok(graph_ledger::parse_derived_graph(quads.into_iter()))
    }

    // ------------------------------------------------------------------
    // 知识融合：命名空间源图 + 融合账本 + 受审默认视图
    // ------------------------------------------------------------------

    /// 断言谓词 → RDF 谓词 IRI：`type` 归一到 rdf:type，外部 IRI 原样保留，
    /// 其余按 `BASE_NS` 展开。
    fn assertion_predicate_iri(predicate: &str) -> String {
        if predicate == "type" {
            format!("{}type", RDF_NS)
        } else if predicate.starts_with("http://") || predicate.starts_with("https://") {
            predicate.to_string()
        } else {
            format!("{}{}", BASE_NS, predicate)
        }
    }

    /// 断言宾语 → 实体 IRI：外部 IRI 原样保留，其余编码为 `kg:entity/...`。
    fn assertion_object_iri(object: &str) -> String {
        if object.starts_with("http://") || object.starts_with("https://") {
            object.to_string()
        } else {
            crate::identity::entity_uri(BASE_NS, object)
        }
    }

    /// 把命名空间断言编码为四元组（写源图或默认图共用）。
    fn assertion_quad(
        subject: &str,
        predicate: &str,
        object: &str,
        object_kind: knowledge_fusion::ObjectKind,
        graph: GraphName,
    ) -> Result<Quad, Box<dyn std::error::Error + Send + Sync>> {
        let subject_node = NamedNode::new(crate::identity::entity_uri(BASE_NS, subject))?;
        let predicate_node = NamedNode::new(Self::assertion_predicate_iri(predicate))?;
        let object_term = match object_kind {
            knowledge_fusion::ObjectKind::Literal => {
                Term::Literal(Literal::new_simple_literal(object))
            }
            knowledge_fusion::ObjectKind::Entity => {
                let iri = if predicate == "type" && !object.starts_with("http") {
                    format!("{}{}", BASE_NS, object)
                } else {
                    Self::assertion_object_iri(object)
                };
                Term::NamedNode(NamedNode::new(iri)?)
            }
        };
        Ok(Quad::new(subject_node, predicate_node, object_term, graph))
    }

    /// 从 RDF 四元组构造命名空间断言（`import namespace` 的解析辅助）：
    /// `rdf:type` 归一为谓词 `type`，kg 词面还原为本地名，外部 IRI 原样保留；
    /// 空白节点主语或词表宾语返回 None（不猜测）。
    pub fn assertion_from_rdf_quad(
        quad: &Quad,
        namespace: &str,
        source: &str,
    ) -> Option<knowledge_fusion::ScopedAssertion> {
        let subject = match &quad.subject {
            Subject::NamedNode(node) => crate::identity::entity_local_name(BASE_NS, node.as_str())
                .or_else(|| crate::identity::local_name(BASE_NS, node.as_str()))
                .unwrap_or_else(|| node.as_str().to_string()),
            _ => return None,
        };
        let predicate_iri = quad.predicate.as_str();
        let predicate = if predicate_iri == format!("{}type", RDF_NS) {
            "type".to_string()
        } else {
            crate::identity::local_name(BASE_NS, predicate_iri)
                .unwrap_or_else(|| predicate_iri.to_string())
        };
        let (object, object_kind) = match &quad.object {
            Term::NamedNode(node) => {
                let iri = node.as_str();
                let name = crate::identity::entity_local_name(BASE_NS, iri)
                    .or_else(|| crate::identity::local_name(BASE_NS, iri))
                    .unwrap_or_else(|| iri.to_string());
                (name, knowledge_fusion::ObjectKind::Entity)
            }
            Term::Literal(literal) => (
                literal.value().to_string(),
                knowledge_fusion::ObjectKind::Literal,
            ),
            _ => return None,
        };
        Some(knowledge_fusion::ScopedAssertion::new(
            namespace,
            &subject,
            &predicate,
            &object,
            object_kind,
            source,
        ))
    }

    /// 从融合命名图读取账本（不触碰缓存，供持锁内部调用）。
    fn fusion_ledger_from_store(
        store: &Store,
    ) -> Result<knowledge_fusion::FusionLedger, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(knowledge_fusion::FUSION_GRAPH)?);
        let quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(knowledge_fusion::parse_fusion_graph(quads.into_iter()))
    }

    /// 全量重写融合账本命名图（幂等；不推进知识版本）。
    fn sync_fusion_ledger_locked(
        &self,
        store: &Store,
        ledger: &knowledge_fusion::FusionLedger,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = knowledge_fusion::fusion_quads(ledger)?;
        let graph = GraphName::NamedNode(NamedNode::new(knowledge_fusion::FUSION_GRAPH)?);
        let existing: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for quad in existing {
            store.remove(&quad)?;
        }
        for quad in &quads {
            store.insert(quad)?;
        }
        self.perf.bump("fusion_ledger_sync");
        Ok(ledger.assertions.len())
    }

    /// 全量重写融合账本（幂等；默认图与知识版本不变）。
    pub async fn sync_fusion_ledger(
        &self,
        ledger: &knowledge_fusion::FusionLedger,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        self.sync_fusion_ledger_locked(&store, ledger)
    }

    /// 读取融合账本（按 id 排序去重，确定性）。
    pub async fn load_fusion_ledger(
        &self,
    ) -> Result<knowledge_fusion::FusionLedger, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let ledger = Self::fusion_ledger_from_store(&store)?;
        self.perf.bump("fusion_ledger_load");
        Ok(ledger)
    }

    /// 把断言写入命名空间源图，并记入融合账本。
    ///
    /// 源图与账本都是命名图：默认图不变、知识版本不推进，因此现有回答路径
    /// 与缓存不受影响；返回本次新增的源图四元组数（重复导入计 0）。
    pub async fn record_namespace_assertions(
        &self,
        namespace: &knowledge_fusion::NamespaceRecord,
        assertions: &[knowledge_fusion::ScopedAssertion],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let graph = GraphName::NamedNode(NamedNode::new(namespace.graph_iri())?);
        let mut added = 0usize;
        for assertion in assertions {
            let quad = Self::assertion_quad(
                &assertion.subject,
                &assertion.predicate,
                &assertion.object,
                assertion.object_kind,
                graph.clone(),
            )?;
            if store.insert(&quad)? {
                added += 1;
            }
        }
        let mut ledger = Self::fusion_ledger_from_store(&store)?;
        ledger.upsert_namespace(namespace.clone());
        for assertion in assertions {
            ledger.record_assertion(assertion.clone());
        }
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        self.perf.bump("namespace_assertions");
        Ok(added)
    }

    /// 设置命名空间状态（启用 / 隔离 / 退役）；返回是否存在该命名空间。
    ///
    /// 只改融合账本元数据：源图证据与默认图都不动，退役后可随时恢复。
    pub async fn set_namespace_status(
        &self,
        namespace: &str,
        status: knowledge_fusion::NamespaceStatus,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let mut ledger = Self::fusion_ledger_from_store(&store)?;
        let id = knowledge_fusion::sanitize_namespace_id(namespace);
        let Some(record) = ledger.namespaces.iter_mut().find(|record| record.id == id) else {
            return Ok(false);
        };
        record.status = status;
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        Ok(true)
    }

    /// 按策略计算融合计划（只读，不写默认图）。
    pub async fn fusion_plan(
        &self,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<knowledge_fusion::FusionPlan, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        Ok(self.fusion_plan_locked(&store, policy)?.1)
    }

    fn fusion_plan_locked(
        &self,
        store: &Store,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<
        (knowledge_fusion::FusionLedger, knowledge_fusion::FusionPlan),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let ledger = Self::fusion_ledger_from_store(store)?;
        let plan = knowledge_fusion::fusion_plan(
            &ledger.namespaces,
            &ledger.assertions,
            &ledger.adjudications,
            policy,
            |subject, predicate, object, object_kind| match Self::assertion_quad(
                subject,
                predicate,
                object,
                object_kind,
                GraphName::DefaultGraph,
            ) {
                Ok(quad) => store.contains(&quad).unwrap_or(false),
                Err(_) => false,
            },
        );
        Ok((ledger, plan))
    }

    /// 把融合视图物化进默认图：写入赢家、回收被替换的旧赢家。
    ///
    /// 只回收本账本 `applied` 记录过的赢家，绝不触碰未纳入融合的历史数据；
    /// 默认图确有变化才推进知识版本，重复融合幂等。
    pub async fn apply_fusion(
        &self,
        policy: &knowledge_fusion::FusionPolicy,
    ) -> Result<knowledge_fusion::FusionApplyOutcome, Box<dyn std::error::Error + Send + Sync>>
    {
        let store = self.store.lock().await;
        let (mut ledger, plan) = self.fusion_plan_locked(&store, policy)?;
        let winners = plan.materialize_keys();

        let mut removed = 0usize;
        let mut removed_quads = Vec::new();
        for previous in &ledger.applied {
            if winners.contains(&(
                previous.subject.clone(),
                previous.predicate.clone(),
                previous.object.clone(),
                previous.object_kind,
            )) {
                continue;
            }
            let quad = Self::assertion_quad(
                &previous.subject,
                &previous.predicate,
                &previous.object,
                previous.object_kind,
                GraphName::DefaultGraph,
            )?;
            if store.remove(&quad)? {
                removed += 1;
                removed_quads.push(quad);
            }
        }

        let mut inserted = 0usize;
        let mut inserted_quads = Vec::new();
        for assertion in &plan.materialize {
            let quad = Self::assertion_quad(
                &assertion.subject,
                &assertion.predicate,
                &assertion.object,
                assertion.object_kind,
                GraphName::DefaultGraph,
            )?;
            if store.insert(&quad)? {
                inserted += 1;
                inserted_quads.push(quad);
            }
        }

        ledger.set_applied(
            plan.materialize
                .iter()
                .map(knowledge_fusion::AppliedWinner::from_assertion)
                .collect(),
        );
        ledger.sort_and_dedup();
        self.sync_fusion_ledger_locked(&store, &ledger)?;
        if inserted + removed > 0 {
            let delta = footprint_from_quad_sets(&inserted_quads, &removed_quads);
            self.bump_revision_with(Some(delta));
        }
        self.perf.bump("fusion_apply");
        Ok(knowledge_fusion::FusionApplyOutcome {
            plan,
            inserted,
            removed,
        })
    }
}
pub fn extract_query_entities(query: &str, config: &SystemConfig) -> Vec<String> {
    let tagged_words = preprocess(query, &config.nlp);
    extract_query_entities_from_tagged(&tagged_words, config)
}
/// 使用已分词结果提取查询实体
pub fn extract_query_entities_from_tagged(
    tagged_words: &[TaggedWord],
    config: &SystemConfig,
) -> Vec<String> {
    let tagged_words = nlp::merge_name_fragments(tagged_words);
    let mut entities = Vec::new();

    for word in &tagged_words {
        if word.pos == "nr"
            || word.pos == "ns"
            || word.pos == "nz"
            || word.pos == "n"
            || word.pos == "nt"
        {
            entities.push(word.text.clone());
        }
    }

    if let Some(synonyms_config) = &config.synonyms {
        for entity in entities.iter_mut() {
            if let Some(normalized) = synonyms_config.mappings.get(entity) {
                *entity = normalized.clone();
            }
        }
    }

    // 去重时保留首次出现顺序
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| e.len() > 1 && seen.insert(e.clone()));
    entities
}
pub struct ResponseFormatter {
    _config: Arc<SystemConfig>,
}
impl ResponseFormatter {
    pub fn new(config: Arc<SystemConfig>) -> Self {
        Self { _config: config }
    }

    /// 取首个结果集中指定变量的绑定值（Prolog 变量名为首字母大写）。
    fn result_value<'a>(
        results: &'a [HashMap<String, String>],
        variable: &str,
    ) -> Option<&'a String> {
        results[0].get(variable)
    }

    pub fn format_prolog_results(
        &self,
        results: &[HashMap<String, String>],
        query_type: &str,
    ) -> String {
        if results.is_empty() {
            return "根据当前知识库，无法回答您的问题。请先导入相关知识。".to_string();
        }

        match query_type {
            "age_query" => {
                if let Some(age) = results[0].get("Age") {
                    format!("年龄是 {} 岁", age)
                } else {
                    "未找到年龄信息".to_string()
                }
            }
            "relation_query" => {
                let relations: Vec<&str> = results
                    .iter()
                    .filter_map(|r| r.get("Relation"))
                    .map(String::as_str)
                    .collect();

                if relations.is_empty() {
                    "未找到关系信息".to_string()
                } else {
                    format!("关系是：{}", relations.join(", "))
                }
            }
            "count_query" => {
                if let Some(count) = results[0].get("Count") {
                    format!("共有 {} 个", count)
                } else {
                    "无法计算数量".to_string()
                }
            }
            "spouse_query" => {
                if let Some(spouse) = results[0].get("Spouse") {
                    format!("配偶是 {}", spouse)
                } else {
                    "未找到配偶信息".to_string()
                }
            }
            "capital_query" => {
                if let Some(capital) = Self::result_value(results, "Capital") {
                    format!("首都是{}", capital)
                } else {
                    "未找到首都信息".to_string()
                }
            }
            "altitude_query" => {
                if let Some(altitude) = Self::result_value(results, "Altitude") {
                    format!("海拔高度是{}米", altitude)
                } else {
                    "未找到海拔信息".to_string()
                }
            }
            "chemical_formula_query" => {
                if let Some(formula) = Self::result_value(results, "Formula") {
                    format!("化学分子式是{}", formula)
                } else {
                    "未找到分子式信息".to_string()
                }
            }
            "author_query" => {
                if let Some(author) = Self::result_value(results, "Author") {
                    format!("作者是{}", author)
                } else {
                    "未找到作者信息".to_string()
                }
            }
            "champion_query" => {
                if let Some(champion) = Self::result_value(results, "Champion") {
                    format!("冠军是{}", champion)
                } else {
                    "未找到冠军信息".to_string()
                }
            }
            "birth_year_query" => {
                if let Some(year) = Self::result_value(results, "Year") {
                    format!("诞生于{}年", year)
                } else {
                    "未找到诞生年份信息".to_string()
                }
            }
            "largest_query" => {
                if let Some(entity) = Self::result_value(results, "Entity") {
                    format!("最大的是{}", entity)
                } else {
                    "未找到相关信息".to_string()
                }
            }
            "quantity_query" => {
                if let Some(quantity) = Self::result_value(results, "Quantity") {
                    format!("数量是{}", quantity)
                } else {
                    "未找到数量信息".to_string()
                }
            }
            "concept_query" | "explanation_query" => {
                if let Some(definition) = results[0].get("definition") {
                    format!("{}", definition)
                } else {
                    "未找到定义信息".to_string()
                }
            }
            "time_before_query" => {
                if let Some(year) = results[0].get("year") {
                    format!("时间是在{}年之前", year)
                } else {
                    "未找到时间信息".to_string()
                }
            }
            "principle_query" => {
                if let Some(principle) = results[0].get("principle") {
                    format!("工作原理是{}", principle)
                } else {
                    "未找到原理信息".to_string()
                }
            }
            "comparison_query" => {
                if let Some(difference) = results[0].get("difference") {
                    format!("区别是{}", difference)
                } else {
                    "未找到比较信息".to_string()
                }
            }
            "philosophy_query" => {
                if let Some(philosophy) = results[0].get("philosophy") {
                    format!("哲学思想是{}", philosophy)
                } else {
                    "未找到哲学思想信息".to_string()
                }
            }
            "impact_query" => {
                if let Some(impact) = results[0].get("impact") {
                    format!("影响是{}", impact)
                } else {
                    "未找到影响信息".to_string()
                }
            }
            "cause_query" | "reason_query" => {
                if let Some(cause) = results[0].get("cause").or_else(|| results[0].get("reason")) {
                    format!("原因是{}", cause)
                } else {
                    "未找到原因信息".to_string()
                }
            }
            "population_query" => {
                if let Some(population) = results[0].get("Population") {
                    format!("人口是 {} 人", population)
                } else {
                    "未找到人口信息".to_string()
                }
            }
            "planet_query" => {
                if let Some(name) = results[0].get("name") {
                    format!("最大的行星是{}", name)
                } else {
                    "未找到行星信息".to_string()
                }
            }
            _ => "根据现有信息，找到了相关内容，但无法生成详细回答。".to_string(),
        }
    }
}
pub async fn load_config() -> Result<SystemConfig, Box<dyn std::error::Error + Send + Sync>> {
    load_config_from("config.toml").await
}
/// 从显式路径加载配置；交互与所有子命令共用同一份解析/校验语义。
pub async fn load_config_from(
    path: &str,
) -> Result<SystemConfig, Box<dyn std::error::Error + Send + Sync>> {
    let config_content = fs::read_to_string(path)?;
    let config: SystemConfig = toml::from_str(&config_content)?;
    Ok(config)
}
#[cfg(test)]
mod config_path_tests {
    use super::*;

    #[tokio::test]
    async fn explicit_config_path_is_loaded_and_missing_path_is_an_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("alternate.toml");
        fs::write(&path, include_str!("../../config.toml")).unwrap();
        let config = load_config_from(path.to_str().unwrap()).await.unwrap();
        assert_eq!(config.system.default_language, "zh");
        assert!(
            load_config_from(dir.path().join("missing.toml").to_str().unwrap())
                .await
                .is_err()
        );
    }
}
pub fn detect_input_mode(input: &str, keywords: &[String]) -> InputMode {
    if keywords.iter().any(|keyword| input.contains(keyword)) {
        InputMode::KnowledgeImport
    } else {
        InputMode::Query
    }
}
pub fn create_prolog_rules(config: &SystemConfig) -> Vec<PrologRule> {
    config
        .prolog_rules
        .iter()
        .map(|rule_cfg| PrologRule {
            pattern: pattern_from_strings(rule_cfg.pattern.clone()),
            query_type: rule_cfg.query_type.clone(),
            priority: rule_cfg.priority,
        })
        .collect()
}
pub fn create_default_synonyms() -> HashMap<String, String> {
    let mut synonyms = HashMap::new();

    // 基础同义词映射
    synonyms.insert("老婆".to_string(), "妻子".to_string());
    synonyms.insert("老公".to_string(), "丈夫".to_string());

    synonyms
}
#[cfg(test)]
mod cache_tests {
    include!("lib__cache_tests_01.rs");
    include!("lib__cache_tests_02.rs");
    include!("lib__delta_cache_tests.rs");
}
#[cfg(test)]
mod fusion_tests {
    use super::*;
    use crate::knowledge_fusion::{
        Adjudication, FusionPolicy, NamespaceKind, NamespaceRecord, NamespaceStatus, ObjectKind,
        ScopedAssertion, TrustLevel,
    };
    use crate::perf_ledger::PerfLedger;

    async fn temp_manager() -> (OxigraphManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = OxigraphManager::with_perf_ledger(
            dir.path().to_str().unwrap(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        (manager, dir)
    }

    fn ns(id: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(id, id, NamespaceKind::Import, trust)
    }

    fn literal(namespace: &str, subject: &str, predicate: &str, object: &str) -> ScopedAssertion {
        ScopedAssertion::new(
            namespace,
            subject,
            predicate,
            object,
            ObjectKind::Literal,
            "test",
        )
    }

    /// 一个带类型与名字的实体断言集合（实体事实的物化需要 kg:name + rdf:type）。
    fn person_assertions(namespace: &str, name: &str, age: &str) -> Vec<ScopedAssertion> {
        vec![
            ScopedAssertion::new(
                namespace,
                name,
                "type",
                "Person",
                ObjectKind::Entity,
                "test",
            ),
            literal(namespace, name, "name", name),
            literal(namespace, name, "年龄", age),
        ]
    }

    fn has_property(facts: &[PrologFact], name: &str, key: &str, value: &str) -> bool {
        facts.iter().any(|fact| match fact {
            PrologFact::Entity {
                name: entity,
                properties,
                ..
            } if entity == name => properties.get(key).map(String::as_str) == Some(value),
            _ => false,
        })
    }

    /// 源图断言不进入默认图、不推进知识版本；重复导入幂等；账本可往返。
    #[tokio::test]
    async fn namespace_assertions_stay_out_of_default_graph() {
        let (manager, _dir) = temp_manager().await;
        let revision_before = manager.revision();
        let namespace = ns("alice", TrustLevel::Curated);
        let assertions = vec![literal("alice", "小明", "年龄", "18")];
        let added = manager
            .record_namespace_assertions(&namespace, &assertions)
            .await
            .unwrap();
        assert_eq!(added, 1);
        assert_eq!(
            manager.revision(),
            revision_before,
            "源图写入不推进知识版本"
        );
        assert!(
            manager.load_all_facts().await.unwrap().is_empty(),
            "默认图不受源图写入影响"
        );

        let rows = manager
            .query_sparql(
                "SELECT ?o WHERE { GRAPH <http://example.org/kg/graphs/source/alice> { \
                 ?s <http://example.org/kg/年龄> ?o } }",
            )
            .await
            .unwrap();
        assert_eq!(rows.data.len(), 1, "源图应可被 SPARQL 查询");

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert!(ledger.synced);
        assert_eq!(ledger.namespaces.len(), 1);
        assert_eq!(ledger.assertions, assertions);

        let added_again = manager
            .record_namespace_assertions(&namespace, &assertions)
            .await
            .unwrap();
        assert_eq!(added_again, 0, "重复导入幂等");
        assert_eq!(
            manager.load_fusion_ledger().await.unwrap().assertions.len(),
            1
        );
    }

    /// 功能谓词冲突：受管来源获胜、少数派隔离、默认图只写入赢家；重复融合幂等。
    #[tokio::test]
    async fn apply_fusion_materializes_winner_and_quarantines_loser() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();

        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = manager.fusion_plan(&policy).await.unwrap();
        assert_eq!(plan.conflicts().len(), 1);
        assert_eq!(plan.stats.quarantined, 1);

        let revision_before = manager.revision();
        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 3, "类型 / 名字 / 年龄三条都应物化");
        assert_eq!(outcome.removed, 0);
        assert!(outcome.changed());
        assert!(manager.revision() > revision_before, "默认图变化应推进版本");

        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "name", "小明"));
        assert!(has_property(&facts, "小明", "年龄", "18"));
        assert!(
            !has_property(&facts, "小明", "年龄", "19"),
            "少数派不得进入默认图"
        );

        let again = manager.apply_fusion(&policy).await.unwrap();
        assert!(!again.changed(), "重复融合应幂等");
        let revision_after = manager.revision();
        manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(manager.revision(), revision_after);

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert_eq!(ledger.applied.len(), 3);
        assert!(ledger
            .applied
            .iter()
            .any(|winner| winner.predicate == "年龄" && winner.object == "18"));
    }

    /// 用户裁决翻转赢家：旧赢家被回收、新赢家写入，且只影响声明的谓词槽位。
    #[tokio::test]
    async fn adjudication_flip_reclaims_previous_winner() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        manager.apply_fusion(&policy).await.unwrap();

        let mut ledger = manager.load_fusion_ledger().await.unwrap();
        ledger.record_adjudication(Adjudication::new("年龄", "friend", "本人确认", "user"));
        manager.sync_fusion_ledger(&ledger).await.unwrap();

        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 1, "只有年龄槽位翻转");
        assert_eq!(outcome.removed, 1, "旧赢家应被回收");
        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "年龄", "19"));
        assert!(!has_property(&facts, "小明", "年龄", "18"));

        let ledger = manager.load_fusion_ledger().await.unwrap();
        assert_eq!(ledger.adjudications.len(), 1);
        assert_eq!(ledger.applied.len(), 3);
        assert!(ledger
            .applied
            .iter()
            .any(|winner| winner.predicate == "年龄" && winner.object == "19"));
    }

    /// 历史默认图事实不被融合触碰（只增不删非托管数据）。
    #[tokio::test]
    async fn fusion_apply_keeps_legacy_default_facts() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![KnowledgeItem::Entity {
                label: "Person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "小明".to_string()),
                    ("年龄".to_string(), "17".to_string()),
                ]),
            }])
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &person_assertions("kb", "小明", "18"),
            )
            .await
            .unwrap();
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let outcome = manager.apply_fusion(&policy).await.unwrap();
        assert_eq!(outcome.inserted, 1, "只有年龄是新事实（类型/名字已存在）");
        assert_eq!(outcome.removed, 0);
        let facts = manager.load_all_facts().await.unwrap();
        assert!(has_property(&facts, "小明", "name", "小明"));

        // 同槽位的历史值（17）与融合赢家（18）都保留在默认图：不做静默删除
        for value in ["17", "18"] {
            let ask = manager
                .query_sparql(&format!(
                    "ASK {{ <http://example.org/kg/entity/小明> <http://example.org/kg/年龄> \"{}\" }}",
                    value
                ))
                .await
                .unwrap();
            assert_eq!(
                ask.data[0]["boolean"],
                serde_json::Value::Bool(true),
                "默认图应保留「{}」",
                value
            );
        }
    }

    /// 退役命名空间退出视图（其断言计为孤儿），只改账本元数据。
    #[tokio::test]
    async fn retiring_namespace_removes_it_from_view() {
        let (manager, _dir) = temp_manager().await;
        manager
            .record_namespace_assertions(
                &ns("kb", TrustLevel::Curated),
                &[literal("kb", "小明", "年龄", "18")],
            )
            .await
            .unwrap();
        manager
            .record_namespace_assertions(
                &ns("friend", TrustLevel::Imported),
                &[literal("friend", "小明", "年龄", "19")],
            )
            .await
            .unwrap();
        assert!(manager
            .set_namespace_status("friend", NamespaceStatus::Retired)
            .await
            .unwrap());
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = manager.fusion_plan(&policy).await.unwrap();
        assert_eq!(plan.stats.namespaces, 1);
        assert_eq!(plan.stats.orphans, 1);
        assert_eq!(
            plan.slots[0].delta,
            crate::knowledge_fusion::FusionDelta::New
        );
        assert_eq!(plan.slots[0].accepted[0].object, "18");
        assert!(!manager
            .set_namespace_status("ghost", NamespaceStatus::Retired)
            .await
            .unwrap());
    }

    /// RDF 四元组 → 命名空间断言：kg 词面还原本地名、rdf:type 归一、外部 IRI 保留。
    #[test]
    fn assertion_from_rdf_quad_normalizes_terms() {
        let quad = Quad::new(
            NamedNode::new("http://example.org/kg/entity/小明").unwrap(),
            NamedNode::new("http://example.org/kg/年龄").unwrap(),
            Literal::new_simple_literal("18"),
            GraphName::DefaultGraph,
        );
        let assertion = OxigraphManager::assertion_from_rdf_quad(&quad, "alice", "a.ttl").unwrap();
        assert_eq!(assertion.subject, "小明");
        assert_eq!(assertion.predicate, "年龄");
        assert_eq!(assertion.object, "18");
        assert_eq!(assertion.object_kind, ObjectKind::Literal);
        assert_eq!(assertion.namespace, "alice");
        assert_eq!(assertion.source, "a.ttl");

        let type_quad = Quad::new(
            NamedNode::new("http://example.org/kg/entity/小明").unwrap(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
            NamedNode::new("http://example.org/kg/Person").unwrap(),
            GraphName::DefaultGraph,
        );
        let assertion =
            OxigraphManager::assertion_from_rdf_quad(&type_quad, "alice", "a.ttl").unwrap();
        assert_eq!(assertion.predicate, "type");
        assert_eq!(assertion.object, "Person");
        assert_eq!(assertion.object_kind, ObjectKind::Entity);

        let blank = Quad::new(
            BlankNode::default(),
            NamedNode::new("http://example.org/kg/年龄").unwrap(),
            Literal::new_simple_literal("1"),
            GraphName::DefaultGraph,
        );
        assert!(OxigraphManager::assertion_from_rdf_quad(&blank, "alice", "a.ttl").is_none());
    }
}

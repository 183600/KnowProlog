impl OxigraphManager {
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
    /// 持锁写入完成后才推进知识版本：事务失败原样回滚，不留下半批数据；
    /// 重复导入仍推进版本，但按空增量足迹保留无关缓存。
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
        let (added, added_quads) = store.transaction(move |mut transaction| {
            let mut added = 0usize;
            let mut added_quads = Vec::new();
            for quad in &quads {
                if transaction.insert(quad.as_ref())? {
                    added += 1;
                    added_quads.push(quad.clone());
                }
            }
            Ok::<(usize, Vec<Quad>), oxigraph::store::StorageError>((added, added_quads))
        })?;
        let delta = footprint_from_quads(&added_quads);
        self.bump_revision_with(Some(delta));
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
        let delta = footprint_from_quads(&flattened);
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
                    let mut added_quads = Vec::new();
                    for quad in &flattened {
                        if transaction.insert(quad.as_ref())? {
                            added += 1;
                            added_quads.push(quad.clone());
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
                        let additions_graph_node = GraphName::NamedNode(
                            NamedNode::new(crate::import_episode::additions_graph_iri(&id))
                                .map_err(|error| ImportBatchTransactionError(error.to_string()))?,
                        );
                        for quad in &added_quads {
                            let addition_quad = Quad::new(
                                quad.subject.clone(),
                                quad.predicate.clone(),
                                quad.object.clone(),
                                additions_graph_node.clone(),
                            );
                            transaction.insert(addition_quad.as_ref())?;
                        }
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
            self.bump_revision_with(Some(delta));
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

    /// 读取某批次的断言（批次命名图内容，保留批次图名）。
    pub async fn load_import_batch_quads(
        &self,
        episode: &crate::import_episode::ImportEpisode,
    ) -> Result<Vec<Quad>, Box<dyn std::error::Error + Send + Sync>> {
        let graph = GraphName::NamedNode(NamedNode::new(&episode.batch_graph)?);
        let store = self.store.lock().await;
        self.perf.bump("import_batch_load");
        store
            .quads_for_pattern(None, None, None, Some(graph.as_ref()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> { Box::new(error) })
    }

    /// 把某批断言导出为独立 RDF 文件；数据集格式保留批次图名，图格式压平到默认图。
    pub async fn export_import_batch_file(
        &self,
        episode: &crate::import_episode::ImportEpisode,
        file_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let format = Self::detect_export_format(file_path)?;
        let file = fs::File::create(file_path)?;
        self.export_import_batch_to_writer(episode, format, file)
            .await
    }

    /// 把某批断言序列化到任意 writer，返回写出的断言数。
    pub async fn export_import_batch_to_writer<W: std::io::Write>(
        &self,
        episode: &crate::import_episode::ImportEpisode,
        format: oxigraph::io::RdfFormat,
        writer: W,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let quads = self.load_import_batch_quads(episode).await?;
        let keep_graphs = format.supports_datasets();
        let mut serializer = oxigraph::io::RdfSerializer::from_format(format).for_writer(writer);
        for quad in &quads {
            let quad = if keep_graphs {
                quad.clone()
            } else {
                Quad::new(
                    quad.subject.clone(),
                    quad.predicate.clone(),
                    quad.object.clone(),
                    GraphName::DefaultGraph,
                )
            };
            serializer.serialize_quad(&quad).map_err(
                |error| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("批次导出失败：{}", error).into()
                },
            )?;
        }
        serializer
            .finish()
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                format!("批次导出失败：{}", error).into()
            })?;
        Ok(quads.len())
    }

    /// 撤销一个批次：回撤本批首次新增、且未被其他批次共享的默认图断言。
    ///
    /// 安全边界：
    /// - 只删除 `import-additions/{id}` 里记录过的「首次新增」三元组；
    ///   导入前就已在默认图中的手动断言不会被误删；
    /// - 若其他批次仍包含同一断言，则保留默认图，并把该断言转移到那些批次的
    ///   新增图，保证日后撤销最后一个持有者时仍能正确回撤；
    /// - 旧版本批次没有新增图时，只移除批次图与元数据，默认图保持不动。
    pub async fn retract_import_batch(
        &self,
        episode: &crate::import_episode::ImportEpisode,
    ) -> Result<crate::import_episode::BatchRetraction, Box<dyn std::error::Error + Send + Sync>>
    {
        let store = self.store.lock().await;
        let existing = Self::import_episodes_from_store(&store)?;
        let Some(current) = existing
            .iter()
            .find(|known| known.id == episode.id)
            .cloned()
        else {
            return Err(format!("没有匹配的导入批次：B{}（{}）", episode.seq, episode.id).into());
        };

        let mut other_additions: Vec<(GraphName, HashSet<String>)> = Vec::new();
        for other in existing.iter().filter(|known| known.id != current.id) {
            let graph = GraphName::NamedNode(NamedNode::new(&other.batch_graph)?);
            let keys = store
                .quads_for_pattern(None, None, None, Some(graph.as_ref()))
                .map(|quad| {
                    quad.map(|quad| crate::import_episode::assertion_key(&quad))
                        .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(error)
                        })
                })
                .collect::<Result<HashSet<_>, _>>()?;
            if !keys.is_empty() {
                let additions = GraphName::NamedNode(NamedNode::new(
                    crate::import_episode::additions_graph_iri(&other.id),
                )?);
                other_additions.push((additions, keys));
            }
        }

        let batch_graph = GraphName::NamedNode(NamedNode::new(&current.batch_graph)?);
        let batch_quads: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(batch_graph.as_ref()))
            .collect::<Result<_, _>>()?;
        let additions_graph = GraphName::NamedNode(NamedNode::new(
            crate::import_episode::additions_graph_iri(&current.id),
        )?);
        let additions: Vec<Quad> = store
            .quads_for_pattern(None, None, None, Some(additions_graph.as_ref()))
            .collect::<Result<_, _>>()?;
        let ledger_graph =
            GraphName::NamedNode(NamedNode::new(crate::import_episode::EPISODE_LEDGER_GRAPH)?);
        let metadata_subject = Subject::NamedNode(NamedNode::new(
            crate::import_episode::episode_subject_iri(&current.id),
        )?);
        let metadata: Vec<Quad> = store
            .quads_for_pattern(
                Some(metadata_subject.as_ref()),
                None,
                None,
                Some(ledger_graph.as_ref()),
            )
            .collect::<Result<_, _>>()?;

        let additions_len = additions.len();
        let (removed, kept_elsewhere, batch_assertions, removed_default_quads) =
            store.transaction(move |mut transaction| {
                let mut removed = 0usize;
                let mut kept_elsewhere = 0usize;
                let mut removed_default_quads = Vec::new();
                for addition in &additions {
                    let key = crate::import_episode::assertion_key(addition);
                    let owners: Vec<&GraphName> = other_additions
                        .iter()
                        .filter(|(_, keys)| keys.contains(&key))
                        .map(|(graph, _)| graph)
                        .collect();
                    if owners.is_empty() {
                        let default_quad = Quad::new(
                            addition.subject.clone(),
                            addition.predicate.clone(),
                            addition.object.clone(),
                            GraphName::DefaultGraph,
                        );
                        if transaction.remove(default_quad.as_ref())? {
                            removed += 1;
                            removed_default_quads.push(default_quad);
                        }
                    } else {
                        kept_elsewhere += 1;
                        for owner in owners {
                            let transferred = Quad::new(
                                addition.subject.clone(),
                                addition.predicate.clone(),
                                addition.object.clone(),
                                owner.clone(),
                            );
                            transaction.insert(transferred.as_ref())?;
                        }
                    }
                }
                let mut batch_assertions = 0usize;
                for quad in &batch_quads {
                    if transaction.remove(quad.as_ref())? {
                        batch_assertions += 1;
                    }
                }
                for quad in &additions {
                    transaction.remove(quad.as_ref())?;
                }
                for quad in &metadata {
                    transaction.remove(quad.as_ref())?;
                }
                Ok::<(usize, usize, usize, Vec<Quad>), ImportBatchTransactionError>((
                    removed,
                    kept_elsewhere,
                    batch_assertions,
                    removed_default_quads,
                ))
            })?;

        if removed > 0 {
            let delta = footprint_from_removed_quads(&removed_default_quads);
            self.bump_revision_with(Some(delta));
        }
        self.perf.bump("import_batch_retract");
        Ok(crate::import_episode::BatchRetraction {
            seq: current.seq,
            id: current.id,
            added: additions_len,
            removed,
            kept_elsewhere,
            batch_assertions,
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
}

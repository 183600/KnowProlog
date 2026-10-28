impl OxigraphManager {
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
        let columns: Vec<String>;
        match results {
            QueryResults::Solutions(solutions) => {
                columns = solutions
                    .variables()
                    .iter()
                    .map(|variable| variable.as_str().to_string())
                    .collect();
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
                columns = vec!["boolean".to_string()];
                data.push(HashMap::from([("boolean".to_string(), Value::Bool(value))]));
            }
            QueryResults::Graph(triples) => {
                columns = vec![
                    "subject".to_string(),
                    "predicate".to_string(),
                    "object".to_string(),
                ];
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
            columns,
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

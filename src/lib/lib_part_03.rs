impl OxigraphManager {
    /// `http://example.org/kg/entity/张三` → `张三`（百分号编码无损反解），其它 IRI 原样返回。
    fn local_kg_name(iri: &str) -> String {
        crate::identity::local_name(BASE_NS, iri).unwrap_or_else(|| iri.to_string())
    }

    /// 主体 IRI → 图谱键：优先按 kg 实体路径无损反解；外部 IRI 退回「尾部段」语义。
    fn entity_key_from_iri(iri: &str) -> String {
        if let Some(name) = crate::identity::local_name(BASE_NS, iri) {
            return name;
        }
        match iri.rsplit_once("entity/") {
            Some((_, tail)) => tail.to_string(),
            None => iri.to_string(),
        }
    }

    /// 把知识库中现存三元组观测进谓词画像（供「谓词报告」与候选发现使用）。
    pub async fn refresh_predicate_registry(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let observed: Vec<(String, String, String)> = {
            let store = self.store.lock().await;
            let mut observed = Vec::new();
            for quad in store.iter() {
                let quad = match quad {
                    Ok(quad) => quad,
                    Err(_) => continue,
                };
                // 命名图（规则/派生账本）不是知识图谱：谓词画像只观测默认图
                if quad.graph_name != GraphName::DefaultGraph {
                    continue;
                }
                let predicate = Self::local_kg_name(quad.predicate.as_str());
                // rdf:type 不是领域谓词，跳过
                if predicate == "type" {
                    continue;
                }
                let subject = match &quad.subject {
                    Subject::NamedNode(node) => Self::local_kg_name(node.as_str()),
                    _ => continue,
                };
                let object = match &quad.object {
                    Term::NamedNode(node) => Self::local_kg_name(node.as_str()),
                    Term::Literal(literal) => literal.value().to_string(),
                    _ => continue,
                };
                observed.push((subject, predicate, object));
            }
            observed
        };
        let mut registry = self.predicate_registry.write();
        for (subject, predicate, object) in observed {
            registry.observe(&subject, &predicate, &object, Some("知识图谱"));
        }
        Ok(())
    }

    /// 从配置声明谓词画像，并设置准入策略。
    pub fn configure_predicates(
        &self,
        profiles: &[predicate_registry::PredicateProfileConfig],
        strict_admission: bool,
    ) {
        let mut registry = self.predicate_registry.write();
        registry.set_strict_admission(strict_admission);
        for profile in profiles {
            registry.declare(profile);
        }
    }

    /// 从配置装载类型体系（别名/子类/互斥 + 闭包闸门开关）。
    pub fn configure_type_system(&self, config: &type_system::TypeSystemConfig) {
        *self.type_system.write() = type_system::TypeSystem::from_config(config);
    }

    /// 类型体系快照（质量护照等只读消费者使用）。
    pub fn type_system_snapshot(&self) -> type_system::TypeSystem {
        self.type_system.read().clone()
    }

    /// 类型体系报告（确定性）。
    pub fn type_system_report(&self) -> String {
        self.type_system.read().report()
    }

    /// 模式闸门是否处于闭包档。
    pub fn closure_gate(&self) -> bool {
        self.type_system.read().closure_gate()
    }

    /// 谓词注册表快照（质量护照等只读消费者使用）。
    pub fn predicate_registry_snapshot(&self) -> predicate_registry::PredicateRegistry {
        self.predicate_registry.read().clone()
    }

    /// 全部功能谓词冲突笔记（被拒入值；图谱快照里看不到的「不同说法」）。
    pub fn predicate_conflicts(&self) -> Vec<predicate_registry::ImportConflict> {
        self.predicate_registry.read().conflicts()
    }

    /// 谓词画像 + 候选发现报告。
    pub fn predicate_registry_report(&self) -> String {
        let registry = self.predicate_registry.read();
        let mut output = registry.report();
        output.push_str(&registry.findings_report());
        output
    }

    /// 全部（词面 → 规范谓词, 是否逆向）映射，供对话层把问句归一到规范方向。
    pub fn predicate_query_pairs(&self) -> Vec<(String, String, bool)> {
        self.predicate_registry.read().alias_pairs()
    }

    pub fn register_predicate_alias(&self, alias: &str, canonical: &str) -> bool {
        self.predicate_registry
            .write()
            .register_alias(alias, canonical, "人工确认")
    }

    pub fn register_predicate_inverse(&self, left: &str, right: &str) -> bool {
        self.predicate_registry
            .write()
            .register_inverse(left, right, "人工确认")
    }

    pub fn mark_predicate_functional(&self, predicate: &str, functional: bool) -> bool {
        let mut registry = self.predicate_registry.write();
        if !registry.is_known(predicate) {
            registry.admit(predicate, "人工确认");
            registry.confirm(predicate);
        }
        registry.mark_functional(predicate, functional)
    }

    pub fn confirm_predicate(&self, predicate: &str) -> bool {
        self.predicate_registry.write().confirm(predicate)
    }

    /// 拒绝待定谓词：移除画像并删除图谱中该谓词的全部三元组。
    pub async fn reject_predicate(
        &self,
        predicate: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let canonical = self
            .predicate_registry
            .read()
            .resolve(predicate)
            .canonical
            .clone();
        let provisional = self
            .predicate_registry
            .read()
            .profile(&canonical)
            .map_or(false, |profile| profile.provisional);
        if !provisional {
            return Ok(0);
        }
        let removed = {
            let store = self.store.lock().await;
            let predicate_iri = NamedNode::new(format!("{}{}", BASE_NS, canonical))?;
            let quads: Vec<Quad> = store
                .quads_for_pattern(None, Some(predicate_iri.as_ref()), None, None)
                .collect::<Result<Vec<_>, _>>()?;
            let mut removed = 0usize;
            for quad in quads {
                if store.remove(&quad).unwrap_or(false) {
                    removed += 1;
                }
            }
            if removed > 0 {
                self.bump_revision();
            }
            removed
        };
        self.predicate_registry.write().reject(&canonical);
        Ok(removed)
    }

    pub fn save_predicate_registry(&self, path: &str) -> std::io::Result<()> {
        self.predicate_registry.read().save_to_path(path)
    }

    pub fn load_predicate_registry(&self, path: &str) -> bool {
        match predicate_registry::PredicateRegistry::load_from_path(path) {
            Some(loaded) => {
                *self.predicate_registry.write() = loaded;
                true
            }
            None => false,
        }
    }

    // ------------------------------------------------------------------
    // 实体身份注册表：谓词注册表的实体侧对偶（提名 / 确认 / 拒绝 / 忘记 / 合并）
    // ------------------------------------------------------------------

    /// 从配置声明实体画像（声明即已确认的显式承诺）。
    pub fn configure_entities(&self, profiles: &[entity_registry::EntityProfileConfig]) {
        let mut registry = self.entity_registry.write();
        for profile in profiles {
            registry.declare(profile);
        }
    }

    pub fn entity_registry_snapshot(&self) -> entity_registry::EntityRegistry {
        self.entity_registry.read().clone()
    }

    /// 身份台账报告（确定性；含待确认提名与墓碑）。
    pub fn entity_registry_report(&self) -> String {
        self.entity_registry.read().report()
    }

    /// 铸 URI 前的宪法检查（Exists / Redirect / Variant / Ambiguous / Fresh）。
    pub fn entity_mint_check(&self, name: &str) -> entity_registry::MintVerdict {
        self.entity_registry.read().mint_check(name)
    }

    pub fn resolve_entity_identity(&self, surface: &str) -> entity_registry::EntityResolution {
        self.entity_registry.read().resolve(surface)
    }

    /// 已确认别名对（词面 → 规范名），供链接器按持久别名识别。
    pub fn confirmed_entity_aliases(&self) -> Vec<(String, String)> {
        self.entity_registry.read().confirmed_alias_pairs()
    }

    /// 确认一个别名（显式承诺：词义改写需要第二次签字）。
    pub fn confirm_entity_alias(&self, surface: &str, canonical: &str) -> bool {
        let changed = self
            .entity_registry
            .write()
            .confirm_alias(surface, canonical);
        if changed {
            self.perf.bump("identity_alias_confirm");
        }
        changed
    }

    /// 拒绝一个别名提名（墓碑：不予自动提名，也不参与解析）。
    pub fn reject_entity_alias(&self, surface: &str, canonical: &str) -> bool {
        let changed = self
            .entity_registry
            .write()
            .reject_alias(surface, canonical);
        if changed {
            self.perf.bump("identity_alias_reject");
        }
        changed
    }

    /// 忘记一个已确认别名（撤销 + 墓碑）；返回受影响的规范身份。
    pub fn forget_entity_alias(&self, surface: &str) -> Vec<String> {
        let affected = self.entity_registry.write().forget_alias(surface);
        if !affected.is_empty() {
            self.perf.bump("identity_alias_forget");
        }
        affected
    }

    /// 合并身份（重命名 = 合并 + 重定向；URI 永不回收）。
    pub fn merge_entity_identities(
        &self,
        from: &str,
        into: &str,
    ) -> Result<entity_registry::MergeRecord, String> {
        let record = self.entity_registry.write().merge(from, into)?;
        self.perf.bump("identity_merge");
        Ok(record)
    }

    /// 提名文本模式抽出的别名候选（只提名，不写库）。
    pub fn nominate_entity_aliases(
        &self,
        candidates: &[entity_registry::AliasCandidate],
    ) -> Vec<entity_registry::NominationOutcome> {
        let types = self.type_system.read().clone();
        let mut registry = self.entity_registry.write();
        let mut outcomes = Vec::new();
        for candidate in candidates {
            let evidence = if candidate.evidence.is_empty() {
                candidate.pattern.clone()
            } else {
                format!("{}：{}", candidate.pattern, candidate.evidence)
            };
            let outcome = registry.nominate_alias_with_types(
                &candidate.surface,
                &candidate.canonical,
                entity_registry::NominationOrigin::Import,
                &evidence,
                Some(&types),
            );
            match outcome.kind {
                entity_registry::NominationKind::Confirmed => {
                    self.perf.bump("identity_nominate_confirmed")
                }
                entity_registry::NominationKind::Pending => {
                    self.perf.bump("identity_nominate_pending")
                }
                entity_registry::NominationKind::TypeConflict => {
                    self.perf.bump("identity_nominate_type_conflict")
                }
                entity_registry::NominationKind::Tombstoned => {
                    self.perf.bump("identity_nominate_tombstoned")
                }
                _ => {}
            }
            outcomes.push(outcome);
        }
        outcomes
    }

    /// 语境指称绑定经二次确认后的别名提名（来源＝用户确认，仍走类型闸门）。
    pub fn nominate_dialogue_aliases(
        &self,
        candidates: &[entity_registry::AliasCandidate],
    ) -> Vec<entity_registry::NominationOutcome> {
        let types = self.type_system.read().clone();
        let mut registry = self.entity_registry.write();
        let mut outcomes = Vec::new();
        for candidate in candidates {
            let evidence = if candidate.evidence.is_empty() {
                candidate.pattern.clone()
            } else {
                format!("{}：{}", candidate.pattern, candidate.evidence)
            };
            let outcome = registry.nominate_alias_with_types(
                &candidate.surface,
                &candidate.canonical,
                entity_registry::NominationOrigin::Dialogue,
                &evidence,
                Some(&types),
            );
            match outcome.kind {
                entity_registry::NominationKind::Confirmed => {
                    self.perf.bump("identity_dialogue_confirmed")
                }
                entity_registry::NominationKind::Pending => {
                    self.perf.bump("identity_dialogue_pending")
                }
                entity_registry::NominationKind::TypeConflict => {
                    self.perf.bump("identity_dialogue_type_conflict")
                }
                entity_registry::NominationKind::Tombstoned => {
                    self.perf.bump("identity_dialogue_tombstoned")
                }
                _ => {}
            }
            outcomes.push(outcome);
        }
        outcomes
    }

    /// 从知识图谱刷新身份观测：`rdf:type` 与 `kg:name`（唯一自动确认来源）。
    ///
    /// 按规范名排序处理，保证同一图谱得到逐字节一致的身份台账。
    pub async fn refresh_entity_registry(
        &self,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut observed: std::collections::BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> =
            std::collections::BTreeMap::new();
        {
            let store = self.store.lock().await;
            let rdf_type = NamedNode::new(format!("{}type", RDF_NS))?;
            for quad in store.iter() {
                let quad = match quad {
                    Ok(quad) => quad,
                    Err(_) => continue,
                };
                if quad.graph_name != GraphName::DefaultGraph {
                    continue;
                }
                let Subject::NamedNode(subject) = &quad.subject else {
                    continue;
                };
                let Some(canonical) = crate::identity::entity_local_name(BASE_NS, subject.as_str())
                else {
                    continue;
                };
                let entry = observed.entry(canonical).or_default();
                if quad.predicate == rdf_type {
                    if let Term::NamedNode(type_node) = &quad.object {
                        if let Some(label) = type_node.as_str().strip_prefix(BASE_NS) {
                            entry.0.insert(label.to_string());
                        }
                    }
                } else if quad.predicate.as_str() == NAME_PREDICATE_IRI {
                    if let Term::Literal(literal) = &quad.object {
                        entry.1.insert(literal.value().to_string());
                    }
                }
            }
        }
        let mut registry = self.entity_registry.write();
        let mut changed = 0usize;
        for (canonical, (types, names)) in observed {
            let names: Vec<String> = names.into_iter().collect();
            if registry.observe_kb_entity(&canonical, &types, &names, "知识图谱") {
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// 从已加载的事实观测实体身份（与 `refresh_entity_registry` 等价，省一次全库扫描）。
    pub fn observe_entity_facts(&self, facts: &[PrologFact]) -> usize {
        let mut by_name: std::collections::BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> =
            std::collections::BTreeMap::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    let entry = by_name.entry(name.clone()).or_default();
                    if !entity_type.trim().is_empty() {
                        entry.0.insert(entity_type.clone());
                    }
                    if let Some(value) = properties.get("name") {
                        if !value.trim().is_empty() {
                            entry.1.insert(value.clone());
                        }
                    }
                }
                PrologFact::Relation { from, to, .. } => {
                    by_name.entry(from.clone()).or_default();
                    by_name.entry(to.clone()).or_default();
                }
            }
        }
        let mut registry = self.entity_registry.write();
        let mut changed = 0usize;
        for (canonical, (types, names)) in by_name {
            let names: Vec<String> = names.into_iter().collect();
            if registry.observe_kb_entity(&canonical, &types, &names, "知识图谱") {
                changed += 1;
            }
        }
        changed
    }

    pub fn save_entity_registry(&self, path: &str) -> std::io::Result<()> {
        self.entity_registry.read().save_to_path(path)
    }

    /// 加载身份注册表（版本门：折叠算法版本高于支持版本时不加载）。
    pub fn load_entity_registry(&self, path: &str) -> bool {
        match entity_registry::EntityRegistry::load_from_path(path) {
            Some(loaded) => {
                *self.entity_registry.write() = loaded;
                true
            }
            None => false,
        }
    }

    pub async fn get_relevant_facts(
        &self,
        query_entities: &[String],
        max_hops: usize,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        if query_entities.is_empty() {
            // 使用整图快照缓存，仅保留实体事实
            let snapshot = self.load_graph_snapshot().await?;
            return Ok(snapshot
                .iter()
                .filter(|fact| matches!(fact, PrologFact::Entity { .. }))
                .cloned()
                .collect());
        }

        // 同一知识版本下的同一组实体（排序后）与跳数 → 直接复用上次结果。
        // 问句实体顺序不影响语义，键里统一排序消除排列差异
        let mut sorted_entities: Vec<&str> = query_entities.iter().map(String::as_str).collect();
        sorted_entities.sort_unstable();
        let cache_key = format!("{}|{}", max_hops, sorted_entities.join("\u{1}"));
        let revision = self.revision.load(Ordering::Acquire);
        {
            let mut cache = self.relevant_cache.lock();
            if let Some(facts) = cache.get(revision, &cache_key) {
                self.perf.bump("relevant_facts_cache_hit");
                return Ok(facts.as_ref().clone());
            }
        }
        self.perf.bump("relevant_facts_cache_miss");
        self.perf.bump("relevant_facts_graph_scan");

        let mut facts = Vec::new();
        let store = self.store.lock().await;

        // 构建实体名称的匹配模式
        let mut combined_pattern = String::new();
        for (index, entity) in query_entities.iter().enumerate() {
            if index > 0 {
                combined_pattern.push('|');
            }
            combined_pattern.push_str(".*");
            combined_pattern.push_str(&regex::escape(entity));
            combined_pattern.push_str(".*");
        }

        // 第一步：找到直接匹配的实体
        let direct_matches = self
            .find_entities_by_pattern(&store, &combined_pattern)
            .await?;
        let mut relevant_entities: std::collections::HashSet<String> =
            direct_matches.into_iter().collect();

        // 第二步：扩展到相邻实体（基于指定跳数，仅扩展每轮新增的前沿节点）
        let mut frontier: Vec<String> = relevant_entities.iter().cloned().collect();
        for _hop in 0..max_hops {
            if frontier.is_empty() {
                break;
            }
            let neighbors = self.find_neighbors(&store, &frontier).await?;
            let mut next_frontier: Vec<String> = Vec::new();
            for neighbor in neighbors {
                if relevant_entities.insert(neighbor.clone()) {
                    next_frontier.push(neighbor);
                }
            }
            frontier = next_frontier;
        }

        // 第三步：为相关实体构建Prolog事实
        facts.extend(self.build_entity_facts(&store, &relevant_entities).await?);

        // 第四步：添加实体间的关系
        let relations = self
            .find_relations_between(&store, &relevant_entities)
            .await?;
        facts.extend(relations);

        let facts = Arc::new(facts);
        if self
            .relevant_cache
            .lock()
            .put(revision, cache_key, facts.clone())
            > 0
        {
            self.perf.bump("relevant_facts_cache_evict");
        }
        Ok(facts.as_ref().clone())
    }

    pub async fn load_all_facts(
        &self,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        // 与 get_relevant_facts 一致：直接复用整图快照缓存中的实体事实
        let snapshot = self.load_graph_snapshot().await?;
        Ok(snapshot
            .iter()
            .filter(|fact| matches!(fact, PrologFact::Entity { .. }))
            .cloned()
            .collect())
    }

    /// 整图快照（实体属性 + 关系），在两次写操作之间缓存。
    ///
    /// 正确性依赖：所有写入口在持有存储锁期间、且在实际写入完成之后自增
    /// `revision`（version-after-write）；因此命中缓存意味着自缓存构建以来
    /// 没有任何成功写入，内容必然一致。
    pub async fn load_graph_snapshot(
        &self,
    ) -> Result<Arc<Vec<PrologFact>>, Box<dyn std::error::Error + Send + Sync>> {
        let revision = self.revision.load(Ordering::Acquire);
        {
            let cache = self.snapshot_cache.lock();
            if let Some((cached_revision, snapshot)) = cache.as_ref() {
                if *cached_revision == revision {
                    self.perf.bump("graph_snapshot_cache_hit");
                    return Ok(snapshot.clone());
                }
            }
        }
        self.perf.bump("graph_snapshot_scan");

        let (facts, relations) = {
            let store = self.store.lock().await;
            Self::scan_graph_parts(&store)
        };
        let mut snapshot_facts = facts;
        snapshot_facts.extend(relations);
        let snapshot = Arc::new(snapshot_facts);

        let mut cache = self.snapshot_cache.lock();
        *cache = Some((revision, snapshot.clone()));
        Ok(snapshot)
    }

    /// 读取默认图的**全部**事实引用（不做主谓折叠、不丢多值）。
    ///
    /// `load_graph_snapshot` 把实体属性放进 `HashMap`，同一谓词的多个取值会
    /// 互相覆盖——这对回答路径是历史兼容行为，但会让「同一主语的两个取值」
    /// 在冲突检测里结构上不可见。本方法保留每一行，供溯源/冲突证书消费。
    pub async fn load_fact_refs(
        &self,
    ) -> Result<Vec<crate::provenance::FactRef>, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.store.lock().await;
        let mut out: Vec<crate::provenance::FactRef> = Self::scan_fact_refs(&store)
            .into_iter()
            .map(|(fact, _)| fact)
            .collect();
        out.sort();
        out.dedup();
        self.perf.bump("fact_ref_scan");
        Ok(out)
    }

    /// 读取按层组织的事实快照：默认图的全部行 + 谓词注册表里的功能声明。
    ///
    /// 这是只读 adapter；写入路径、缓存与导入行为不受影响。快照的 revision
    /// 与知识版本一致，同一 revision 下重复读取得到相同内容。
    pub async fn load_fact_snapshot(
        &self,
    ) -> Result<crate::fact_snapshot::FactSnapshot, Box<dyn std::error::Error + Send + Sync>> {
        let revision = self.revision.load(Ordering::Acquire);
        let rows = {
            let store = self.store.lock().await;
            Self::scan_fact_refs(&store)
        };
        let source = crate::provenance::SourceId::new(
            crate::provenance::SourceKind::Import,
            "oxigraph:default",
            &format!("revision={revision};rows={}", rows.len()),
        );
        let mut builder = crate::fact_snapshot::FactSnapshotBuilder::new(revision);
        builder.grammar_digest("oxigraph");
        for predicate in self.predicate_registry.read().functional_predicates() {
            builder.mark_functional(&predicate);
        }
        for (fact, kind) in rows {
            builder.add_base(fact, kind, source.clone());
        }
        self.perf.bump("fact_snapshot_load");
        Ok(builder.build())
    }

    /// 单次遍历默认图，保留每一行并区分类型/属性/关系。
    fn scan_fact_refs(
        store: &Store,
    ) -> Vec<(crate::provenance::FactRef, crate::fact_snapshot::FactKind)> {
        use crate::fact_snapshot::FactKind;
        let mut out: Vec<(crate::provenance::FactRef, FactKind)> = Vec::new();
        for quad in store.iter() {
            let quad = match quad {
                Ok(quad) => quad,
                Err(_) => continue,
            };
            if quad.graph_name != GraphName::DefaultGraph {
                continue;
            }
            let subject = match &quad.subject {
                Subject::NamedNode(node) => Self::entity_key_from_iri(node.as_str()),
                _ => {
                    let text = quad.subject.to_string();
                    Self::entity_key_from_iri(text.trim_matches(|c| c == '<' || c == '>'))
                }
            };
            let predicate = if quad.predicate.as_str() == RDF_TYPE_IRI {
                "type".to_string()
            } else if quad.predicate.as_str().contains(BASE_NS) {
                quad.predicate.as_str().replace(BASE_NS, "")
            } else {
                quad.predicate.as_str().to_string()
            };
            let (value, kind) = match &quad.object {
                Term::Literal(literal) => (literal.value().to_string(), FactKind::Attribute),
                Term::NamedNode(node) => (
                    Self::entity_key_from_iri(node.as_str()),
                    if predicate == "type" {
                        FactKind::Type
                    } else {
                        FactKind::Relation
                    },
                ),
                object => (
                    object
                        .to_string()
                        .trim_matches(|c| c == '<' || c == '>')
                        .to_string(),
                    FactKind::Attribute,
                ),
            };
            out.push((
                crate::provenance::FactRef::new(&subject, &predicate, &value),
                kind,
            ));
        }
        out.sort();
        out.dedup();
        out
    }

    /// 单次遍历存储，同时构建整图的实体事实与关系事实。
    ///
    /// 语义与 `load_all_facts` + `load_all_relations` 的组合保持一致：
    /// - 实体只保留拥有 `rdf:type` 的主体；同一实体多个类型时取最先遇到者，
    ///   重复属性取最后遇到者（按主体聚簇的存储序）；
    /// - 关系仅保留两端都有 `kg:name` 的命名主体，且排除自环与 `rdf:type`。
    fn scan_graph_parts(store: &Store) -> (Vec<PrologFact>, Vec<PrologFact>) {
        let mut entity_types: HashMap<String, String> = HashMap::new();
        let mut entity_props: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut subject_names: HashMap<Term, Vec<String>> = HashMap::new();
        let mut relation_candidates: Vec<(Term, Term, String)> = Vec::new();

        for quad in store.iter() {
            let quad = match quad {
                Ok(quad) => quad,
                Err(_) => continue,
            };
            // 命名图（规则/派生账本）不参与知识快照：只拍平默认图
            if quad.graph_name != GraphName::DefaultGraph {
                continue;
            }

            // 命名节点直接取 IRI 尾部（kg 实体 IRI 经百分号解码反解；外部 IRI 按尾部段语义）
            let entity_name = match &quad.subject {
                Subject::NamedNode(node) => Self::entity_key_from_iri(node.as_str()),
                _ => {
                    let subject_str = quad.subject.to_string();
                    let trimmed = subject_str.trim_matches(|c| c == '<' || c == '>');
                    Self::entity_key_from_iri(trimmed)
                }
            };

            if quad.predicate.as_str() == RDF_TYPE_IRI {
                let type_name = match &quad.object {
                    Term::NamedNode(node) => {
                        let iri = node.as_str();
                        if iri.contains(BASE_NS) {
                            iri.replace(BASE_NS, "").to_lowercase()
                        } else {
                            iri.to_lowercase()
                        }
                    }
                    object => {
                        let type_str = object.to_string();
                        if type_str.contains(BASE_NS) {
                            type_str
                                .replace(BASE_NS, "")
                                .trim_matches(|c| c == '<' || c == '>')
                                .to_lowercase()
                        } else {
                            type_str
                                .trim_matches(|c| c == '<' || c == '>')
                                .to_lowercase()
                        }
                    }
                };
                entity_types.entry(entity_name).or_insert(type_name);
                continue;
            }

            let is_name_predicate = quad.predicate.as_str() == NAME_PREDICATE_IRI;
            let predicate_iri = quad.predicate.as_str();
            let prop_name = if predicate_iri.contains(BASE_NS) {
                predicate_iri.replace(BASE_NS, "")
            } else {
                predicate_iri.to_string()
            };

            let value_str = if let Term::Literal(lit) = &quad.object {
                lit.value().to_string()
            } else {
                match &quad.object {
                    // kg 实体宾语在快照里统一用本地名，与关系事实的可读取值
                    // 一致；外部 IRI 原样保留，避免把别人的命名空间截断。
                    Term::NamedNode(node) => {
                        crate::identity::entity_local_name(BASE_NS, node.as_str())
                            .unwrap_or_else(|| node.as_str().to_string())
                    }
                    object => object
                        .to_string()
                        .trim_matches(|c| c == '<' || c == '>')
                        .to_string(),
                }
            };

            let name_value = if is_name_predicate {
                Some(if let Term::Literal(_) = &quad.object {
                    value_str.clone()
                } else {
                    quad.object.to_string()
                })
            } else {
                None
            };

            entity_props
                .entry(entity_name)
                .or_default()
                .insert(prop_name.clone(), value_str);

            if matches!(&quad.object, Term::Literal(_)) {
                // 字面量宾语只登记名称，不生成关系候选
                if let Some(name_value) = name_value {
                    subject_names
                        .entry(Term::from(quad.subject))
                        .or_default()
                        .push(name_value);
                }
            } else {
                let subject_term = Term::from(quad.subject);
                if let Some(name_value) = name_value {
                    subject_names
                        .entry(subject_term.clone())
                        .or_default()
                        .push(name_value);
                }
                relation_candidates.push((subject_term, quad.object, prop_name));
            }
        }

        let mut facts = Vec::with_capacity(entity_types.len());
        for (entity_name, entity_type) in entity_types {
            let properties = entity_props.remove(&entity_name).unwrap_or_default();
            facts.push(PrologFact::Entity {
                name: entity_name,
                entity_type,
                properties,
            });
        }

        let mut relations = Vec::with_capacity(relation_candidates.len());
        for (from, to, rel_type) in relation_candidates {
            if from == to {
                continue;
            }
            if let (Some(from_names), Some(to_names)) =
                (subject_names.get(&from), subject_names.get(&to))
            {
                for from_name in from_names {
                    for to_name in to_names {
                        relations.push(PrologFact::Relation {
                            from: from_name.clone(),
                            rel_type: rel_type.clone(),
                            to: to_name.clone(),
                        });
                    }
                }
            }
        }

        (facts, relations)
    }

    /// 直答入口：同一知识版本下同一（查询类型 + 实体）复用结果（含负缓存）。
    pub async fn get_direct_answer(&self, query_type: &str, entities: &[String]) -> Option<String> {
        let cache_key = format!("{}|{}", query_type, entities.join("\u{1}"));
        let revision = self.revision.load(Ordering::Acquire);
        {
            let mut cache = self.direct_answer_cache.lock();
            if let Some(answer) = cache.get(revision, &cache_key) {
                self.perf.bump("direct_answer_cache_hit");
                return answer.clone();
            }
        }
        self.perf.bump("direct_answer_cache_miss");
        let answer = self.get_direct_answer_uncached(query_type, entities).await;
        if self
            .direct_answer_cache
            .lock()
            .put(revision, cache_key, answer.clone())
            > 0
        {
            self.perf.bump("direct_answer_cache_evict");
        }
        answer
    }

    async fn get_direct_answer_uncached(
        &self,
        query_type: &str,
        entities: &[String],
    ) -> Option<String> {
        let store = self.store.lock().await;
        self.perf.bump("direct_answer_query");

        // 根据查询类型构造不同的SPARQL查询
        let query = match query_type {
            "capital_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?capital WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:capital ?capital .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "altitude_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?altitude WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:altitude ?altitude .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "chemical_formula_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?formula WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:formula ?formula .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "author_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?author WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:author ?author .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "champion_query" => {
                let entity_name = if entities.is_empty() {
                    return None;
                } else {
                    &entities[0]
                };
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?champion WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:champion ?champion .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "birth_year_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?year WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:birth_year ?year .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "concept_query" | "explanation_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?definition WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:definition ?definition .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "quantity_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?quantity WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:quantity ?quantity .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "planet_query" | "largest_query" => {
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?name WHERE {{
                        ?entity kg:type "planet" .
                        ?entity kg:size_rank "1" .
                        ?entity kg:name ?name .
                    }}
                "#
                )
            }
            "time_before_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?time WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:time ?time .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "principle_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?principle WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:principle ?principle .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "comparison_query" => {
                if entities.len() < 2 {
                    return None;
                }
                let entity1 = &entities[0];
                let entity2 = &entities[1];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?difference WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?comparison kg:entity1 ?entity1 .
                        ?comparison kg:entity2 ?entity2 .
                        ?comparison kg:difference ?difference .
                    }}
                "#,
                    sparql_literal(entity1),
                    sparql_literal(entity2)
                )
            }
            "philosophy_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?philosophy WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:philosophy ?philosophy .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "impact_query" => {
                if entities.len() < 2 {
                    return None;
                }
                let subject = &entities[0];
                let target = &entities[1];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?impact WHERE {{
                        ?entity1 kg:name "{}" .
                        ?entity2 kg:name "{}" .
                        ?entity1 kg:impact ?impact .
                        ?impact kg:target ?entity2 .
                    }}
                "#,
                    sparql_literal(subject),
                    sparql_literal(target)
                )
            }
            "cause_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?cause WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:cause ?cause .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            "reason_query" => {
                if entities.is_empty() {
                    return None;
                }
                let entity_name = &entities[0];
                format!(
                    r#"
                    PREFIX kg: <http://example.org/kg/>
                    SELECT ?reason WHERE {{
                        ?entity kg:name "{}" .
                        ?entity kg:reason ?reason .
                    }}
                "#,
                    sparql_literal(entity_name)
                )
            }
            _ => return None,
        };

        // 执行SPARQL查询
        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    // 获取第一个变量的值
                                    if let Some((_, value)) = solution.iter().next() {
                                        let value_str = if let Term::Literal(lit) = value {
                                            lit.value().to_string()
                                        } else {
                                            value
                                                .to_string()
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        // 根据查询类型格式化答案
                                        let answer = match query_type {
                                            "capital_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的首都是{}", entity_name, value_str)
                                            }
                                            "altitude_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该地点"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!(
                                                    "{}的海拔高度是{}米",
                                                    entity_name, value_str
                                                )
                                            }
                                            "chemical_formula_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!(
                                                    "{}的化学分子式是{}",
                                                    entity_name, value_str
                                                )
                                            }
                                            "author_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该作品"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的作者是{}", entity_name, value_str)
                                            }
                                            "champion_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该比赛"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的冠军是{}", entity_name, value_str)
                                            }
                                            "quantity_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该项目"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}有{}个", entity_name, value_str)
                                            }
                                            "birth_year_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}诞生于{}年", entity_name, value_str)
                                            }
                                            "planet_query" | "largest_query" => {
                                                format!("最大的行星是{}", value_str)
                                            }
                                            "time_before_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    "该事件"
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}在{}年之前", entity_name, value_str)
                                            }
                                            "concept_query" | "explanation_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}是{}", entity_name, value_str)
                                            }
                                            "principle_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的工作原理是{}", entity_name, value_str)
                                            }
                                            "comparison_query" => {
                                                if entities.len() >= 2 {
                                                    format!(
                                                        "{}和{}的区别是{}",
                                                        entities[0], entities[1], value_str
                                                    )
                                                } else {
                                                    format!("区别是{}", value_str)
                                                }
                                            }
                                            "philosophy_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的哲学思想是{}", entity_name, value_str)
                                            }
                                            "impact_query" => {
                                                if entities.len() >= 2 {
                                                    format!(
                                                        "{}对{}的影响是{}",
                                                        entities[0], entities[1], value_str
                                                    )
                                                } else {
                                                    format!("影响是{}", value_str)
                                                }
                                            }
                                            "cause_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            }
                                            "reason_query" => {
                                                let entity_name = if entities.is_empty() {
                                                    ""
                                                } else {
                                                    &entities[0]
                                                };
                                                format!("{}的原因是{}", entity_name, value_str)
                                            }
                                            _ => value_str,
                                        };

                                        return Some(answer);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => return None,
                }
            }
            Err(_) => return None,
        }

        None
    }

    async fn find_entities_by_pattern(
        &self,
        store: &Store,
        pattern: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut entities = Vec::new();

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT DISTINCT ?entity ?name WHERE {{
                ?entity kg:name ?name .
                FILTER(REGEX(?name, "{}", "i"))
            }}
        "#,
            sparql_literal(pattern)
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let Some(name) = solution.get("name") {
                                    if let Term::Literal(lit) = name {
                                        entities.push(lit.value().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(entities)
    }

    async fn find_neighbors(
        &self,
        store: &Store,
        entities: &[String],
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut neighbors = Vec::new();

        if entities.is_empty() {
            return Ok(neighbors);
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
            
            SELECT DISTINCT ?neighbor_name WHERE {{
                {{
                    ?entity kg:name ?entity_name .
                    ?neighbor kg:name ?neighbor_name .
                    ?entity ?rel ?neighbor .
                    FILTER(?entity_name IN ({}))
                    FILTER(?rel != rdf:type)
                    FILTER(?neighbor_name NOT IN ({}))
                }} UNION {{
                    ?entity kg:name ?entity_name .
                    ?neighbor kg:name ?neighbor_name .
                    ?neighbor ?rel ?entity .
                    FILTER(?entity_name IN ({}))
                    FILTER(?rel != rdf:type)
                    FILTER(?neighbor_name NOT IN ({}))
                }}
            }}
        "#,
            entity_filter, entity_filter, entity_filter, entity_filter
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => match store.query(query_obj) {
                Ok(results) => {
                    if let QueryResults::Solutions(solutions) = results {
                        for solution in solutions {
                            if let Ok(solution) = solution {
                                if let Some(name) = solution.get("neighbor_name") {
                                    if let Term::Literal(lit) = name {
                                        neighbors.push(lit.value().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            },
            Err(_) => {}
        }

        Ok(neighbors)
    }

    async fn build_entity_facts(
        &self,
        store: &Store,
        entity_names: &HashSet<String>,
    ) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
        let mut facts = Vec::new();

        if entity_names.is_empty() {
            return Ok(facts);
        }

        let mut name_filter = String::new();
        for (index, entity) in entity_names.iter().enumerate() {
            if index > 0 {
                name_filter.push_str(", ");
            }
            name_filter.push('"');
            name_filter.push_str(&sparql_literal(entity));
            name_filter.push('"');
        }

        let query = format!(
            r#"
            PREFIX kg: <http://example.org/kg/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?name ?type ?prop ?value WHERE {{
                ?entity kg:name ?name .
                ?entity rdf:type ?type .
                FILTER(?name IN ({}))
                OPTIONAL {{ ?entity ?prop ?value . FILTER(?prop != rdf:type && ?prop != kg:name) }}
            }}
        "#,
            name_filter
        );

        match Query::parse(&query, None) {
            Ok(query_obj) => {
                match store.query(query_obj) {
                    Ok(results) => {
                        if let QueryResults::Solutions(solutions) = results {
                            let mut entity_data: HashMap<
                                String,
                                (String, HashMap<String, String>),
                            > = HashMap::new();

                            for solution in solutions {
                                if let Ok(solution) = solution {
                                    let entity_name = match solution.get("name") {
                                        Some(Term::Literal(lit)) => lit.value().to_string(),
                                        _ => continue,
                                    };

                                    let entry = entity_data
                                        .entry(entity_name)
                                        .or_insert_with(|| (String::new(), HashMap::new()));

                                    if let Some(type_term) = solution.get("type") {
                                        let type_str = match type_term {
                                            Term::NamedNode(node) => node.as_str().to_string(),
                                            other => other.to_string(),
                                        };
                                        // 后续类型覆盖先前类型
                                        entry.0 = if type_str.contains(BASE_NS) {
                                            type_str
                                                .replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        } else {
                                            type_str
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_lowercase()
                                        };
                                    }

                                    if let (Some(prop), Some(value)) =
                                        (solution.get("prop"), solution.get("value"))
                                    {
                                        let prop_str = match prop {
                                            Term::NamedNode(node) => node.as_str().to_string(),
                                            other => other.to_string(),
                                        };
                                        let prop_name = if prop_str.contains(BASE_NS) {
                                            prop_str
                                                .replace(BASE_NS, "")
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        } else {
                                            prop_str
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        let value_str = if let Term::Literal(lit) = value {
                                            lit.value().to_string()
                                        } else {
                                            value
                                                .to_string()
                                                .trim_matches(|c| c == '<' || c == '>')
                                                .to_string()
                                        };

                                        entry.1.insert(prop_name, value_str);
                                    }
                                }
                            }

                            // 按传入顺序输出，跳过没有类型信息的实体
                            for name in entity_names {
                                if let Some((entity_type, properties)) = entity_data.remove(name) {
                                    if !entity_type.is_empty() {
                                        facts.push(PrologFact::Entity {
                                            name: name.clone(),
                                            entity_type,
                                            properties,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }

        Ok(facts)
    }
}

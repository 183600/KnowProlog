impl SymbolicGenerator {
    pub fn new(max_profile_sentences: usize, completion_min_score: f64) -> Self {
        Self {
            known_entities: HashSet::new(),
            entity_types: HashMap::new(),
            type_members: HashMap::new(),
            attributes: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            predicate_totals: HashMap::new(),
            entity_predicate_counts: HashMap::new(),
            entity_relation_total: HashMap::new(),
            predicate_min_entity: HashMap::new(),
            triples: HashSet::new(),
            total_relations: 0,
            max_profile_sentences: max_profile_sentences.max(1),
            completion_min_score: completion_min_score.clamp(0.0, 1.0),
            predicate_surfaces: HashMap::new(),
            lexicalized_predicates: HashSet::new(),
            semantic_grammar: None,
            import_frames: None,
            frame_book: None,
            revision: 0,
            type_count_cache: parking_lot::RwLock::new(HashMap::new()),
            type_versions: HashMap::new(),
            type_version_counter: 0,
            type_profile_cache: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn is_known_entity(&self, name: &str) -> bool {
        self.known_entities.contains(name)
    }

    /// 挂接双向语义规则包：之后生成优先使用同一份规则里的 `realize` 模板。
    pub fn attach_semantic_grammar(&mut self, grammar: crate::semantic_grammar::RulePack) {
        self.semantic_grammar = Some(grammar);
    }

    /// 摘除双向语义规则包（配置热加载关闭该能力时使用）。
    pub fn detach_semantic_grammar(&mut self) {
        self.semantic_grammar = None;
    }

    /// 挂接导入可逆帧：生成优先复用导入句学到的措辞。
    pub fn attach_import_frames(&mut self, frames: crate::import_frames::ImportFrameLedger) {
        self.import_frames = Some(frames);
    }

    /// 摘除导入可逆帧（撤销全部导入或热加载关闭时使用）。
    pub fn detach_import_frames(&mut self) {
        self.import_frames = None;
    }

    /// 当前导入帧台账（未挂接时为 `None`）。
    pub fn import_frames(&self) -> Option<&crate::import_frames::ImportFrameLedger> {
        self.import_frames.as_ref()
    }

    /// 挂接谓词中心语义角色帧；之后可用同一份 schema 标注与实现帧。
    pub fn attach_frame_book(&mut self, book: crate::frame_semantics::FrameBook) {
        self.frame_book = Some(book);
    }

    pub fn detach_frame_book(&mut self) {
        self.frame_book = None;
    }

    pub fn frame_book(&self) -> Option<&crate::frame_semantics::FrameBook> {
        self.frame_book.as_ref()
    }

    /// 用已挂接的 schema 实现一个语义角色帧；没有匹配 schema 时返回 `None`。
    pub fn realize_frame(&self, frame: &crate::frame_semantics::FrameInstance) -> Option<String> {
        self.frame_book.as_ref()?.realize(frame)
    }

    /// 用已挂接的 schema 从文本标注语义角色帧。
    pub fn annotate_frames(&self, text: &str) -> Vec<crate::frame_semantics::FrameInstance> {
        let Some(book) = self.frame_book.as_ref() else {
            return Vec::new();
        };
        crate::frame_semantics::FrameAnnotator::new(book).annotate(text)
    }

    /// 用导入帧回答宾语问句：只有图谱取值存在时才作答，并附原句见证。
    pub fn answer_import_frame(&self, question: &str) -> Option<String> {
        let frames = self.import_frames.as_ref()?;
        let answer = frames.answer(
            question,
            frames.min_ask_support(),
            |kind, subject, predicate| match kind {
                crate::import_frames::FactKind::Attribute => {
                    self.attribute_values(subject, predicate)
                }
                crate::import_frames::FactKind::Relation => {
                    self.relation_values(subject, predicate)
                }
            },
        )?;
        Some(answer.render_with_witness())
    }

    /// 热更新生成预算：只改「说多少」，不重建已观察的事实与类型索引。
    pub fn set_generation_budget(
        &mut self,
        max_profile_sentences: usize,
        completion_min_score: f64,
    ) {
        self.max_profile_sentences = max_profile_sentences.max(1);
        self.completion_min_score = completion_min_score.clamp(0.0, 1.0);
    }

    /// 当前挂接的语义规则包（未挂接时为 `None`）。
    pub fn semantic_grammar(&self) -> Option<&crate::semantic_grammar::RulePack> {
        self.semantic_grammar.as_ref()
    }

    pub fn relation_count(&self) -> usize {
        self.total_relations
    }

    pub fn entity_type_of(&self, name: &str) -> Option<&str> {
        self.entity_types.get(name).map(String::as_str)
    }

    /// 已登记类型的所有实体（实体名, 类型名），供归结引擎编码 type 事实
    pub fn typed_entities(&self) -> Vec<(String, String)> {
        self.entity_types
            .iter()
            .map(|(entity, etype)| (entity.clone(), etype.clone()))
            .collect()
    }

    pub fn all_entities(&self) -> Vec<String> {
        let mut entities: Vec<String> = self.known_entities.iter().cloned().collect();
        entities.sort();
        entities
    }

    pub fn all_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.type_members.keys().cloned().collect();
        types.sort();
        types
    }

    pub fn entities_of_type(&self, etype: &str) -> Vec<String> {
        self.type_members
            .get(etype)
            .map(|members| members.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 某类型自己的版本戳（未见过的类型为 0）。
    pub fn type_revision(&self, etype: &str) -> u64 {
        self.type_versions.get(etype).copied().unwrap_or(0)
    }

    /// 把用户词面解析成图中的类型名：精确名 → 大小写不敏感 → 本地化标签。
    /// 只在显式成员类型中查找，避免把普通实体名误当类型。
    pub fn resolve_type_name(&self, mention: &str) -> Option<String> {
        let mention = mention.trim().trim_matches(|c: char| {
            matches!(
                c,
                '「' | '」' | '"' | '\'' | '“' | '”' | '？' | '?' | '。' | '，' | ',' | '：' | ':'
            )
        });
        if mention.is_empty() {
            return None;
        }
        let types = self.all_types();
        if let Some(exact) = types.iter().find(|etype| etype.as_str() == mention) {
            return Some(exact.clone());
        }
        if let Some(insensitive) = types
            .iter()
            .find(|etype| etype.eq_ignore_ascii_case(mention))
        {
            return Some(insensitive.clone());
        }
        let label = localized_type_label(mention);
        types
            .iter()
            .find(|etype| localized_type_label(etype) == label)
            .cloned()
    }

    /// 默认粒度的类型画像（12 种谓词、6 条样例）。
    pub fn type_profile(&self, etype: &str) -> Option<TypeProfile> {
        self.type_profile_bounded(
            etype,
            TYPE_PROFILE_DEFAULT_PREDICATES,
            TYPE_PROFILE_DEFAULT_SAMPLES,
        )
    }

    /// 有界类型画像；缓存键包含界参数，只随该类型的版本戳失效。
    pub fn type_profile_bounded(
        &self,
        etype: &str,
        max_predicates: usize,
        max_samples: usize,
    ) -> Option<TypeProfile> {
        let revision = self.type_revision(etype);
        let cache_key = (etype.to_string(), max_predicates, max_samples);
        {
            let cache = self.type_profile_cache.read();
            if let Some((cached_revision, profile)) = cache.get(&cache_key) {
                if *cached_revision == revision {
                    return Some(profile.clone());
                }
            }
        }
        let profile = self.build_type_profile(etype, revision, max_predicates, max_samples)?;
        self.type_profile_cache
            .write()
            .insert(cache_key, (revision, profile.clone()));
        Some(profile)
    }

    fn build_type_profile(
        &self,
        etype: &str,
        revision: u64,
        max_predicates: usize,
        max_samples: usize,
    ) -> Option<TypeProfile> {
        let members_ref = self.type_members.get(etype)?;
        if members_ref.is_empty() {
            return None;
        }
        struct Accumulator {
            members: usize,
            values: usize,
            relation: bool,
            attribute: bool,
            samples: Vec<TypeFactSample>,
        }
        let mut accumulators: BTreeMap<String, Accumulator> = BTreeMap::new();
        let mut covered: BTreeSet<String> = BTreeSet::new();
        for entity in members_ref {
            // 同一实体同一谓词先合并：关系优先，姓名自指句不计入画像。
            let mut facts: BTreeMap<String, (BTreeSet<String>, bool)> = BTreeMap::new();
            if let Some(relations) = self.by_subject.get(entity) {
                for (predicate, objects) in relations {
                    let values = facts
                        .entry(predicate.clone())
                        .or_insert_with(|| (BTreeSet::new(), true));
                    values.1 = true;
                    values.0.extend(objects.iter().cloned());
                }
            }
            if let Some(attributes) = self.attributes.get(entity) {
                for (key, value) in attributes {
                    if facts.contains_key(key) {
                        continue;
                    }
                    if is_name_key(key) && value == entity {
                        continue;
                    }
                    facts
                        .entry(key.clone())
                        .or_insert_with(|| (BTreeSet::new(), false))
                        .0
                        .insert(value.clone());
                }
            }
            for (predicate, (values, relation)) in facts {
                if values.is_empty() {
                    continue;
                }
                covered.insert(entity.clone());
                let entry = accumulators
                    .entry(predicate)
                    .or_insert_with(|| Accumulator {
                        members: 0,
                        values: 0,
                        relation: false,
                        attribute: false,
                        samples: Vec::new(),
                    });
                entry.members += 1;
                entry.values += values.len();
                if relation {
                    entry.relation = true;
                } else {
                    entry.attribute = true;
                }
                if entry.samples.len() < TYPE_PROFILE_SAMPLES_PER_PREDICATE {
                    entry.samples.push(TypeFactSample {
                        entity: entity.clone(),
                        values: values.into_iter().collect(),
                    });
                }
            }
        }

        let total_members = members_ref.len();
        let mut summaries: Vec<TypePredicateSummary> = accumulators
            .into_iter()
            .map(|(predicate, accumulator)| {
                let kind = match (accumulator.relation, accumulator.attribute) {
                    (true, true) => TypeFactKind::Mixed,
                    (true, false) => TypeFactKind::Relation,
                    (false, true) => TypeFactKind::Attribute,
                    (false, false) => TypeFactKind::Attribute,
                };
                let coverage_milli = if total_members == 0 {
                    0
                } else {
                    ((accumulator.members * 1000 + total_members / 2) / total_members).min(1000)
                        as u32
                };
                TypePredicateSummary {
                    label: localized_predicate_key(&predicate)
                        .unwrap_or(predicate.as_str())
                        .to_string(),
                    predicate,
                    kind,
                    member_count: accumulator.members,
                    value_count: accumulator.values,
                    coverage_milli,
                    samples: accumulator.samples,
                }
            })
            .collect();
        summaries.sort_by(|left, right| {
            right
                .member_count
                .cmp(&left.member_count)
                .then_with(|| right.value_count.cmp(&left.value_count))
                .then_with(|| left.predicate.cmp(&right.predicate))
        });
        let total_predicates = summaries.len();
        let predicates_truncated = total_predicates > max_predicates;
        summaries.truncate(max_predicates);

        let mut samples = Vec::new();
        'outer: for summary in &summaries {
            for sample in &summary.samples {
                let sentence = self.lexicalize(&sample.entity, &summary.predicate, &sample.values);
                if !samples.contains(&sentence) {
                    samples.push(sentence);
                }
                if samples.len() >= max_samples {
                    break 'outer;
                }
            }
        }

        let members: Vec<String> = members_ref
            .iter()
            .take(TYPE_PROFILE_MAX_MEMBERS)
            .cloned()
            .collect();
        Some(TypeProfile {
            type_name: etype.to_string(),
            revision,
            total_members,
            members_truncated: total_members > members.len(),
            members,
            covered_members: covered.len(),
            total_predicates,
            predicates: summaries,
            predicates_truncated,
            samples,
        })
    }

    /// 类型成员分页（字典序，越界返回空页而不是报错）。
    pub fn type_member_page(
        &self,
        etype: &str,
        offset: usize,
        limit: usize,
    ) -> Option<TypeMemberPage> {
        let members = self.type_members.get(etype)?;
        let total = members.len();
        let limit = limit.max(1);
        let offset = offset.min(total);
        let page: Vec<String> = members.iter().skip(offset).take(limit).cloned().collect();
        let next_offset = offset + page.len();
        Some(TypeMemberPage {
            type_name: etype.to_string(),
            offset,
            next_offset,
            total,
            done: next_offset >= total,
            members: page,
        })
    }

    /// 两个类型的画像对比（全量谓词，不做默认截断）。
    pub fn compare_types(&self, left: &str, right: &str) -> Option<TypeComparison> {
        let left_profile = self.type_profile_bounded(left, usize::MAX, 0)?;
        let right_profile = self.type_profile_bounded(right, usize::MAX, 0)?;
        let left_predicates: BTreeSet<String> = left_profile
            .predicates
            .iter()
            .map(|summary| summary.predicate.clone())
            .collect();
        let right_predicates: BTreeSet<String> = right_profile
            .predicates
            .iter()
            .map(|summary| summary.predicate.clone())
            .collect();
        let shared = left_predicates
            .intersection(&right_predicates)
            .cloned()
            .collect();
        let left_only = left_predicates
            .difference(&right_predicates)
            .cloned()
            .collect();
        let right_only = right_predicates
            .difference(&left_predicates)
            .cloned()
            .collect();
        Some(TypeComparison {
            left: left_profile,
            right: right_profile,
            shared_predicates: shared,
            left_only_predicates: left_only,
            right_only_predicates: right_only,
        })
    }

    /// 默认类型画像的人读报告。
    pub fn type_profile_report(&self, etype: &str) -> Option<String> {
        self.type_profile(etype).map(|profile| profile.render())
    }

    /// 类型对比的人读报告。
    pub fn type_comparison_report(&self, left: &str, right: &str) -> Option<String> {
        self.compare_types(left, right)
            .map(|comparison| comparison.render())
    }

    pub fn attribute_pairs(&self, entity: &str) -> Vec<(String, String)> {
        let mut pairs = self.attributes.get(entity).cloned().unwrap_or_default();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    pub fn relation_pairs(&self, entity: &str) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .by_subject
            .get(entity)
            .map(|predicates| {
                predicates
                    .iter()
                    .flat_map(|(predicate, objects)| {
                        objects
                            .iter()
                            .map(move |object| (predicate.clone(), object.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    pub fn all_predicates(&self) -> Vec<String> {
        let mut predicates: Vec<String> = self.predicate_totals.keys().cloned().collect();
        predicates.sort();
        predicates
    }

    pub fn canonical_predicate(&self, surface: &str) -> Option<String> {
        self.predicate_surfaces.get(surface).cloned()
    }

    /// 把一条事实渲染成自然中文句子（对外复用动词化词典）
    pub fn render_fact(&self, subject: &str, predicate: &str, objects: &[String]) -> String {
        self.lexicalize(subject, predicate, objects)
    }

    pub fn has_attribute(&self, entity: &str, key: &str, value: &str) -> bool {
        self.attributes.get(entity).map_or(false, |pairs| {
            pairs.iter().any(|(k, v)| k == key && v == value)
        })
    }

    pub fn attribute_values(&self, entity: &str, key: &str) -> Vec<String> {
        let mut values = self.values_for_key(entity, key);
        if values.is_empty() {
            // 中文词面（「身高」）回落英文图谱键（height）
            if let Some(canonical) = canonical_predicate_key(key) {
                values = self.values_for_key(entity, canonical);
            }
        }
        values
    }

    fn values_for_key(&self, entity: &str, key: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .attributes
            .get(entity)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .collect()
            })
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    pub fn predicate_surfaces(&self) -> &HashMap<String, String> {
        &self.predicate_surfaces
    }

    /// 规范谓词键的稳定中文展示面：观察事实时登记的非同名词面里，
    /// 先取最短、再按字典序，避免 HashMap 迭代顺序影响回答措辞。
    pub fn preferred_predicate_surface(&self, canonical: &str) -> Option<&str> {
        self.predicate_surfaces
            .iter()
            .filter(|(surface, target)| {
                target.as_str() == canonical && surface.as_str() != canonical
            })
            .map(|(surface, _)| surface.as_str())
            .min_by_key(|surface| (surface.chars().count(), *surface))
    }

    /// 运行时登记谓词别名（谓词画像准入后同步，查询/生成共用同一词面表）。
    pub fn register_predicate_alias(&mut self, alias: &str, canonical: &str) -> bool {
        if alias.trim().is_empty() || canonical.trim().is_empty() || alias == canonical {
            return false;
        }
        if self.predicate_surfaces.get(alias).map(String::as_str) == Some(canonical) {
            return false;
        }
        self.predicate_surfaces
            .insert(canonical.to_string(), canonical.to_string());
        self.predicate_surfaces
            .insert(alias.to_string(), canonical.to_string());
        self.lexicalized_predicates.insert(alias.to_string());
        self.revision += 1;
        self.type_profile_cache.write().clear();
        true
    }

    /// 实体的属性键（去重排序）：供对话管理器生成"我还知道它的……"式追问建议
    pub fn attribute_keys(&self, entity: &str) -> Vec<String> {
        let mut keys: Vec<String> = self
            .attributes
            .get(entity)
            .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
            .unwrap_or_default();
        keys.sort();
        keys.dedup();
        keys
    }

    /// 实体作为主语出现过的关系谓词（去重排序）：与属性键合并后作为可追问主题
    pub fn relation_predicates(&self, entity: &str) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .by_subject
            .get(entity)
            .map(|subjects| subjects.keys().cloned().collect())
            .unwrap_or_default();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// 实体作为宾语出现过的关系谓词（入向；去重排序）。
    /// 供语境指称消歧的义项卡比较「同一类型、不同角色」的候选。
    pub fn relation_predicates_in(&self, entity: &str) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .by_object
            .get(entity)
            .map(|objects| objects.keys().cloned().collect())
            .unwrap_or_default();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// 出向关系取值（主语, 谓词 → 宾语；去重排序）。
    pub fn relation_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .by_subject
            .get(entity)
            .and_then(|subjects| subjects.get(predicate))
            .cloned()
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    /// 入向关系取值（宾语, 谓词 → 主语；去重排序）。
    pub fn relation_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .by_object
            .get(entity)
            .and_then(|objects| objects.get(predicate))
            .cloned()
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    pub fn observe_entity(&mut self, name: &str, etype: &str, properties: &[(String, String)]) {
        let refs: Vec<(&str, &str)> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        self.observe_entity_refs(name, etype, &refs);
    }

    /// 与 `observe_entity` 等价，但直接以属性表为输入，迭代顺序与之一致。
    pub fn observe_entity_map(
        &mut self,
        name: &str,
        etype: &str,
        properties: &HashMap<String, String>,
    ) {
        let refs: Vec<(&str, &str)> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        self.observe_entity_refs(name, etype, &refs);
    }

    fn observe_entity_refs(&mut self, name: &str, etype: &str, properties: &[(&str, &str)]) {
        if name.trim().is_empty() {
            return;
        }
        // 只在知识确有变化时自增 revision
        let mut changed = false;
        let mut touched_types: BTreeSet<String> = BTreeSet::new();
        if self.known_entities.insert(name.to_string()) {
            changed = true;
        }
        if !etype.trim().is_empty() {
            match self
                .entity_types
                .insert(name.to_string(), etype.to_string())
            {
                Some(previous) => {
                    if previous != etype {
                        changed = true;
                        touched_types.insert(previous.clone());
                        touched_types.insert(etype.to_string());
                        if let Some(members) = self.type_members.get_mut(&previous) {
                            members.remove(name);
                            if members.is_empty() {
                                self.type_members.remove(&previous);
                            }
                        }
                        self.type_members
                            .entry(etype.to_string())
                            .or_default()
                            .insert(name.to_string());
                    }
                }
                None => {
                    changed = true;
                    touched_types.insert(etype.to_string());
                    self.type_members
                        .entry(etype.to_string())
                        .or_default()
                        .insert(name.to_string());
                }
            }
        }
        let mut facts_changed = false;
        if !properties.is_empty() {
            let bucket = self.attributes.entry(name.to_string()).or_default();
            for &(key, value) in properties {
                if key.trim().is_empty() {
                    continue;
                }
                if !bucket.iter().any(|(k, v)| k == key && v == value) {
                    bucket.push((key.to_string(), value.to_string()));
                    changed = true;
                    facts_changed = true;
                }
            }
        }
        for &(key, _) in properties {
            if !key.trim().is_empty() && self.register_predicate_surface(key) {
                changed = true;
            }
        }
        if facts_changed {
            if let Some(current) = self.entity_types.get(name) {
                touched_types.insert(current.clone());
            }
        }
        for etype in touched_types {
            self.bump_type_version(&etype);
        }
        if changed {
            self.revision += 1;
        }
    }

    fn register_predicate_surface(&mut self, predicate: &str) -> bool {
        if !self.lexicalized_predicates.insert(predicate.to_string()) {
            return false;
        }
        // 规范谓词自身总是可解析；规范形式优先于其它谓词的词汇化表述
        self.predicate_surfaces
            .insert(predicate.to_string(), predicate.to_string());
        for surface in lexical_surfaces(predicate) {
            if *surface == predicate {
                continue;
            }
            // 图谱里实际出现的内置词面优先于先前登记的配置别名：例如配置
            // 把「年龄」登记成谓词画像后，再观察属性键 age 时，必须把
            // 「年龄」重新指回 age，否则形式化语义会把属性误判成关系。
            self.predicate_surfaces
                .insert(surface.to_string(), predicate.to_string());
        }
        self.type_profile_cache.write().clear();
        true
    }

    /// 推进一个类型自己的版本戳。只有成员集合或成员事实变化时才调用，
    /// 因而同一轮写入不会把无关类型的画像缓存一并打掉。
    fn bump_type_version(&mut self, etype: &str) {
        if etype.trim().is_empty() {
            return;
        }
        self.type_version_counter = self.type_version_counter.saturating_add(1);
        let stamp = self.type_version_counter;
        self.type_versions.insert(etype.to_string(), stamp);
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.trim().is_empty() || predicate.trim().is_empty() || object.trim().is_empty() {
            return;
        }
        let key = triple_key(subject, predicate, object);
        if !self.triples.insert(key) {
            return;
        }
        self.revision += 1;
        let subject_type = self.entity_types.get(subject).cloned();
        self.known_entities.insert(subject.to_string());
        self.known_entities.insert(object.to_string());

        // triples 已按 (s,p,o) 去重，此处到达即保证是新事实
        let subjects = match self.by_subject.get_mut(subject) {
            Some(subjects) => subjects,
            None => self.by_subject.entry(subject.to_string()).or_default(),
        };
        match subjects.get_mut(predicate) {
            Some(objects) => objects.push(object.to_string()),
            None => subjects
                .entry(predicate.to_string())
                .or_default()
                .push(object.to_string()),
        }
        let object_bucket = match self.by_object.get_mut(object) {
            Some(bucket) => bucket,
            None => self.by_object.entry(object.to_string()).or_default(),
        };
        match object_bucket.get_mut(predicate) {
            Some(subjects_rev) => subjects_rev.push(subject.to_string()),
            None => object_bucket
                .entry(predicate.to_string())
                .or_default()
                .push(subject.to_string()),
        }

        match self.predicate_totals.get_mut(predicate) {
            Some(total) => *total += 1,
            None => {
                self.predicate_totals.insert(predicate.to_string(), 1);
            }
        }
        let predicate_counts = match self.entity_predicate_counts.get_mut(subject) {
            Some(counts) => counts,
            None => self
                .entity_predicate_counts
                .entry(subject.to_string())
                .or_default(),
        };
        match predicate_counts.get_mut(predicate) {
            Some(count) => *count += 1,
            None => {
                predicate_counts.insert(predicate.to_string(), 1);
                // 该主体首次拥有此谓词：增量维护字典序最小的原型主体
                let replace = match self.predicate_min_entity.get(predicate) {
                    Some(current) => current.as_str() > subject,
                    None => true,
                };
                if replace {
                    self.predicate_min_entity
                        .insert(predicate.to_string(), subject.to_string());
                }
            }
        }
        match self.entity_relation_total.get_mut(subject) {
            Some(total) => *total += 1,
            None => {
                self.entity_relation_total.insert(subject.to_string(), 1);
            }
        }
        self.total_relations += 1;

        self.register_predicate_surface(predicate);
        if let Some(etype) = subject_type {
            self.bump_type_version(&etype);
        }
    }

    /// 从生成器索引中撤回一条三元组（知识图谱侧删除后调用）。
    ///
    /// 与 `observe_triple` 严格对称：计数、正/反向倒排、谓词原型一起回退，
    /// 重复撤回返回 `false` 且不改变任何状态（幂等）。实体本身保留，
    /// 因为同名的属性/类型证据可能仍然存在。
    pub fn retract_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        if !self.triples.remove(&triple_key(subject, predicate, object)) {
            return false;
        }
        self.revision += 1;
        let subject_type = self.entity_types.get(subject).cloned();

        if let Some(subjects) = self.by_subject.get_mut(subject) {
            if let Some(objects) = subjects.get_mut(predicate) {
                if let Some(position) = objects.iter().position(|existing| existing == object) {
                    objects.remove(position);
                }
                if objects.is_empty() {
                    subjects.remove(predicate);
                }
            }
            if subjects.is_empty() {
                self.by_subject.remove(subject);
            }
        }
        if let Some(bucket) = self.by_object.get_mut(object) {
            if let Some(subjects) = bucket.get_mut(predicate) {
                if let Some(position) = subjects.iter().position(|existing| existing == subject) {
                    subjects.remove(position);
                }
                if subjects.is_empty() {
                    bucket.remove(predicate);
                }
            }
            if bucket.is_empty() {
                self.by_object.remove(object);
            }
        }

        let mut predicate_gone = false;
        if let Some(total) = self.predicate_totals.get_mut(predicate) {
            *total = total.saturating_sub(1);
            if *total == 0 {
                self.predicate_totals.remove(predicate);
                predicate_gone = true;
            }
        }
        if let Some(counts) = self.entity_predicate_counts.get_mut(subject) {
            if let Some(count) = counts.get_mut(predicate) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(predicate);
                }
            }
            if counts.is_empty() {
                self.entity_predicate_counts.remove(subject);
            }
        }
        if let Some(total) = self.entity_relation_total.get_mut(subject) {
            *total = total.saturating_sub(1);
            if *total == 0 {
                self.entity_relation_total.remove(subject);
            }
        }
        self.total_relations = self.total_relations.saturating_sub(1);

        // 原型主体只能从现存主体中重新选取：只增不减会让回退指向幽灵实体。
        if predicate_gone || !self.predicate_min_entity.contains_key(predicate) {
            self.refresh_predicate_min_entity(predicate);
        } else if self.predicate_min_entity.get(predicate).map(String::as_str) == Some(subject) {
            self.refresh_predicate_min_entity(predicate);
        }
        if let Some(etype) = subject_type {
            self.bump_type_version(&etype);
        }
        true
    }

    /// 重算某谓词的全局回退原型（字典序最小的现存主体）。
    fn refresh_predicate_min_entity(&mut self, predicate: &str) {
        let min = self
            .by_subject
            .iter()
            .filter(|(_, predicates)| {
                predicates
                    .get(predicate)
                    .map_or(false, |objects| !objects.is_empty())
            })
            .map(|(subject, _)| subject.clone())
            .min();
        match min {
            Some(subject) => {
                self.predicate_min_entity
                    .insert(predicate.to_string(), subject);
            }
            None => {
                self.predicate_min_entity.remove(predicate);
            }
        }
    }

    // 谓词动词化：把裸谓词（含 JSON-LD 英文谓词）渲染成自然中文句子
    fn lexicalize(&self, subject: &str, predicate: &str, objects: &[String]) -> String {
        let joined = objects.join("、");
        if let Some(realization) = self
            .semantic_grammar
            .as_ref()
            .and_then(|grammar| grammar.realize_values(subject, predicate, objects))
        {
            return realization.text;
        }
        if objects.len() == 1 {
            if let Some(realization) = self
                .import_frames
                .as_ref()
                .and_then(|frames| frames.realize(subject, predicate, &objects[0]))
            {
                return realization;
            }
        }
        if let Some(template) = lexical_template(predicate) {
            return template.replace("{s}", subject).replace("{o}", &joined);
        }
        let surface = localized_predicate_key(predicate).unwrap_or(predicate);
        format!("{}的{}是{}。", subject, surface, joined)
    }

    // 生成式介绍：把实体相关事实排序、模板化、加话语标记后组合成段落。
    pub fn profile(&self, entity: &str) -> Option<String> {
        let lines = self.profile_lines(entity, GenerationOrder::Relevance)?;
        Some(self.render_profile(entity, &lines, 0))
    }

    // 分页生成：支持"接着说"（offset 续航）与"换个角度说"（换排序）
    pub fn profile_page(
        &self,
        entity: &str,
        offset: usize,
        order: GenerationOrder,
    ) -> Option<ProfilePage> {
        let lines = self.profile_lines(entity, order)?;
        // offset 超出末尾时视为请求最后一页之后的内容，避免切片越界 panic
        let offset = offset.min(lines.len());
        let end = offset
            .saturating_add(self.max_profile_sentences)
            .min(lines.len());
        let shown = end.saturating_sub(offset);
        Some(ProfilePage {
            entity: entity.to_string(),
            text: self.render_profile(entity, &lines, offset),
            total_lines: lines.len(),
            shown,
            next_offset: end,
        })
    }

    /// 当前可生成的稳定命题列表（纯图谱视图，等价于空叠加层）。
    pub fn profile_clauses(
        &self,
        entity: &str,
        order: GenerationOrder,
    ) -> Option<Vec<ProfileClause>> {
        self.profile_clauses_with_overlay(entity, order, &GenerationOverlay::default())
    }

    /// 图谱视图 ∪ 会话视图，生成稳定命题列表。
    ///
    /// 功能槽已有会话赢家时以赢家为准（`replace`），否则取并集；被更正/
    /// 否定的取值先删除（`suppress`）。这样生成器本身保持「图谱视图」，
    /// 会话层的裁决由调用方以叠加层传入，避免两份事实互相污染。
    pub fn profile_clauses_with_overlay(
        &self,
        entity: &str,
        order: GenerationOrder,
        overlay: &GenerationOverlay,
    ) -> Option<Vec<ProfileClause>> {
        // 会话层可能引入图谱尚未收录的实体：它既可能是新事实的主语，
        // 也可能只作为宾语出现（此时生成反向句）。
        let overlay_has_entity = overlay.add.iter().any(|((subject, _), values)| {
            subject == entity || values.iter().any(|value| value == entity)
        }) || overlay.replace.iter().any(|((subject, _), values)| {
            subject == entity || values.iter().any(|value| value == entity)
        });
        if !self.is_known_entity(entity) && !overlay_has_entity {
            return None;
        }
        let mut predicates: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(relations) = self.by_subject.get(entity) {
            for (predicate, objects) in relations {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(objects.iter().cloned());
            }
        }
        if let Some(attributes) = self.attributes.get(entity) {
            for (key, value) in attributes {
                let relation = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |relations| relations.contains_key(key));
                if !relation {
                    predicates
                        .entry(key.clone())
                        .or_default()
                        .push(value.clone());
                }
            }
        }
        for ((subject, predicate), values) in &overlay.add {
            if subject == entity {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(values.iter().cloned());
            }
        }
        for ((subject, predicate), values) in &overlay.replace {
            if subject == entity {
                predicates
                    .entry(predicate.clone())
                    .or_default()
                    .extend(values.iter().cloned());
            }
        }

        let mut clauses: Vec<ProfileClause> = Vec::new();
        for (predicate, values) in predicates {
            let slot = (entity.to_string(), predicate.clone());
            if overlay.hide.contains(&slot) {
                continue;
            }
            let mut values = match overlay.replace.get(&slot) {
                Some(replacement) => replacement.clone(),
                None => values,
            };
            if let Some(suppressed) = overlay.suppress.get(&slot) {
                values.retain(|value| !suppressed.contains(value));
            }
            values.sort();
            values.dedup();
            if values.is_empty() {
                continue;
            }
            let relation = self
                .by_subject
                .get(entity)
                .map_or(false, |relations| relations.contains_key(&predicate));
            let attribute = !relation
                && self.attributes.get(entity).map_or(false, |attributes| {
                    attributes.iter().any(|(key, _)| key == &predicate)
                });
            let from_overlay = overlay.touches(entity, &predicate);
            if attribute {
                for value in values {
                    if is_name_key(&predicate) && value == entity {
                        continue;
                    }
                    let key = ClauseKey {
                        entity: entity.to_string(),
                        kind: ClauseKind::Attribute,
                        predicate: predicate.clone(),
                        qualifier: value.clone(),
                    };
                    let sentence =
                        self.lexicalize(entity, &predicate, std::slice::from_ref(&value));
                    let evidence = vec![(entity.to_string(), predicate.clone(), value.clone())];
                    clauses.push(ProfileClause {
                        key,
                        sentence,
                        from_overlay,
                        evidence,
                        rank_milli: 500,
                    });
                }
            } else {
                let key = ClauseKey {
                    entity: entity.to_string(),
                    kind: ClauseKind::Relation,
                    predicate: predicate.clone(),
                    qualifier: String::new(),
                };
                let sentence = self.lexicalize(entity, &predicate, &values);
                let evidence = values
                    .iter()
                    .map(|value| (entity.to_string(), predicate.clone(), value.clone()))
                    .collect();
                clauses.push(ProfileClause {
                    key,
                    sentence,
                    from_overlay,
                    evidence,
                    rank_milli: (self.predicate_idf(&predicate) * 1000.0).round() as i64,
                });
            }
        }

        // 反向句：只补「主语→本实体」的入向事实，且谓词已在正向出现时跳过。
        let mut reverse: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut reverse_overlay: BTreeSet<String> = BTreeSet::new();
        if let Some(relations) = self.by_object.get(entity) {
            for (predicate, subjects) in relations {
                let duplicated = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |forward| forward.contains_key(predicate));
                if duplicated {
                    continue;
                }
                reverse
                    .entry(predicate.clone())
                    .or_default()
                    .extend(subjects.iter().cloned());
            }
        }
        for ((subject, predicate), values) in &overlay.add {
            if subject != entity && values.contains(entity) {
                reverse
                    .entry(predicate.clone())
                    .or_default()
                    .insert(subject.clone());
                reverse_overlay.insert(predicate.clone());
            }
        }
        for ((subject, predicate), values) in &overlay.suppress {
            if subject != entity && values.contains(entity) {
                if let Some(subjects) = reverse.get_mut(predicate) {
                    subjects.remove(subject);
                }
            }
        }
        let mut reverse_added = 0usize;
        for (predicate, subjects) in reverse {
            if reverse_added >= 2 {
                break;
            }
            let duplicated = self
                .by_subject
                .get(entity)
                .map_or(false, |forward| forward.contains_key(&predicate));
            if duplicated || subjects.is_empty() {
                continue;
            }
            let subject_list: Vec<String> = subjects.into_iter().collect();
            let key = ClauseKey {
                entity: entity.to_string(),
                kind: ClauseKind::Reverse,
                predicate: predicate.clone(),
                qualifier: String::new(),
            };
            let sentence = self.lexicalize(
                &subject_list.join("、"),
                &predicate,
                std::slice::from_ref(&entity.to_string()),
            );
            let evidence = subject_list
                .iter()
                .map(|subject| (subject.clone(), predicate.clone(), entity.to_string()))
                .collect();
            clauses.push(ProfileClause {
                key,
                sentence,
                from_overlay: reverse_overlay.contains(&predicate),
                evidence,
                rank_milli: -1000,
            });
            reverse_added += 1;
        }

        match order {
            GenerationOrder::Relevance => clauses.sort_by(|a, b| {
                b.rank_milli
                    .cmp(&a.rank_milli)
                    .then_with(|| a.sentence.cmp(&b.sentence))
                    .then_with(|| a.key.cmp(&b.key))
            }),
            GenerationOrder::Alphabetical => {
                clauses.sort_by(|a, b| a.sentence.cmp(&b.sentence).then_with(|| a.key.cmp(&b.key)))
            }
        }
        Some(clauses)
    }

    /// 用话语账本生成一页：先补更正/撤回，再讲尚未说过的命题。
    pub fn cursor_page(
        &self,
        entity: &str,
        order: GenerationOrder,
        overlay: &GenerationOverlay,
        cursor: &mut GenerationCursor,
    ) -> Option<CursorPage> {
        let clauses = self.profile_clauses_with_overlay(entity, order, overlay)?;
        let advance = cursor.advance(&clauses, self.max_profile_sentences);
        let text = if advance.lines.is_empty() {
            if clauses.is_empty() {
                format!(
                    "关于「{}」的知识（{}符号检索生成）：知识图谱中暂无可用事实。",
                    entity,
                    self.type_note(entity)
                )
            } else {
                String::new()
            }
        } else {
            self.render_cursor_advance(entity, &advance)
        };
        let trusted: Vec<String> = advance
            .lines
            .iter()
            .filter(|line| line.trusted)
            .map(|line| line.text.clone())
            .collect();
        let done = clauses.is_empty()
            || (advance.fresh == 0 && advance.corrected == 0 && advance.removed == 0);
        Some(CursorPage {
            entity: entity.to_string(),
            text,
            total_lines: advance.total_lines,
            shown: advance.lines.len(),
            next_offset: advance.next_offset,
            fresh: advance.fresh,
            corrected: advance.corrected,
            removed: advance.removed,
            done,
            trusted,
        })
    }

    fn type_note(&self, entity: &str) -> String {
        self.entity_types
            .get(entity)
            .filter(|t| t.as_str() != "Entity" && t.as_str() != "实体")
            .map(|t| format!("类型：{}；", localized_type_label(t)))
            .unwrap_or_default()
    }

    fn render_cursor_advance(&self, entity: &str, advance: &CursorAdvance) -> String {
        let mut body = String::new();
        let mut fresh_ordinal = 0usize;
        for line in &advance.lines {
            body.push_str("  ");
            if line.correction {
                body.push_str(&line.text);
            } else {
                if advance.continuation && fresh_ordinal == 0 {
                    body.push_str("接着，");
                } else {
                    body.push_str(Self::discourse_marker(fresh_ordinal, advance.fresh));
                }
                body.push_str(&line.text);
                fresh_ordinal += 1;
            }
            body.push('\n');
        }
        let header = if advance.continuation {
            match advance.span {
                Some((start, end)) => format!(
                    "关于「{}」的知识（续：第 {}–{} 句，共 {} 条事实）：\n",
                    entity,
                    start + 1,
                    end,
                    advance.total_lines
                ),
                None => format!(
                    "关于「{}」的知识（续：更正 {} 条，共 {} 条事实）：\n",
                    entity,
                    advance.corrected + advance.removed,
                    advance.total_lines
                ),
            }
        } else {
            format!(
                "关于「{}」的知识（{}符号检索生成，共 {} 条事实）：\n",
                entity,
                self.type_note(entity),
                advance.lines.len()
            )
        };
        format!("{}{}", header, body.trim_end())
    }

    fn profile_lines(&self, entity: &str, order: GenerationOrder) -> Option<Vec<FactLine>> {
        if !self.is_known_entity(entity) {
            return None;
        }
        let mut lines: Vec<FactLine> = Vec::new();

        if let Some(predicates) = self.by_subject.get(entity) {
            for (predicate, objects) in predicates {
                let idf = self.predicate_idf(predicate);
                lines.push(FactLine {
                    rank: idf,
                    sentence: self.lexicalize(entity, predicate, objects),
                });
            }
        }
        if let Some(attributes) = self.attributes.get(entity) {
            for (key, value) in attributes {
                // 属性与关系是同一谓词时只保留关系句，避免重复生成
                let duplicated = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |predicates| predicates.contains_key(key));
                if duplicated {
                    continue;
                }
                // 姓名与实体名相同属于同义反复，跳过以免生成「X的名字是X」
                if is_name_key(key) && value == entity {
                    continue;
                }
                lines.push(FactLine {
                    rank: 0.5,
                    sentence: self.lexicalize(entity, key, std::slice::from_ref(value)),
                });
            }
        }
        let reverse_sentences = self.reverse_sentences(entity, 2);
        for sentence in reverse_sentences {
            lines.push(FactLine {
                rank: -1.0,
                sentence,
            });
        }

        match order {
            GenerationOrder::Relevance => lines.sort_by(|a, b| {
                b.rank
                    .partial_cmp(&a.rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.sentence.cmp(&b.sentence))
            }),
            GenerationOrder::Alphabetical => lines.sort_by(|a, b| a.sentence.cmp(&b.sentence)),
        }
        Some(lines)
    }

    fn render_profile(&self, entity: &str, lines: &[FactLine], offset: usize) -> String {
        let type_note = self
            .entity_types
            .get(entity)
            .filter(|t| t.as_str() != "Entity" && t.as_str() != "实体")
            .map(|t| format!("类型：{}；", localized_type_label(t)))
            .unwrap_or_default();
        if lines.is_empty() {
            return format!(
                "关于「{}」的知识（{}符号检索生成）：知识图谱中暂无可用事实。",
                entity, type_note
            );
        }
        let end = offset
            .saturating_add(self.max_profile_sentences)
            .min(lines.len());
        let page = &lines[offset..end];
        let continuation = offset > 0;

        let mut body = String::new();
        for (index, line) in page.iter().enumerate() {
            body.push_str("  ");
            if continuation && index == 0 {
                body.push_str("接着，");
            } else {
                body.push_str(&Self::discourse_marker(index, page.len()));
            }
            body.push_str(&line.sentence);
            body.push('\n');
        }
        let header = if continuation {
            format!(
                "关于「{}」的知识（续：第 {}–{} 句，共 {} 条事实）：\n",
                entity,
                offset + 1,
                end,
                lines.len()
            )
        } else {
            format!(
                "关于「{}」的知识（{}符号检索生成，共 {} 条事实）：\n",
                entity,
                type_note,
                page.len()
            )
        };
        format!("{}{}", header, body.trim_end())
    }

    fn reverse_sentences(&self, entity: &str, limit: usize) -> Vec<String> {
        let Some(predicates) = self.by_object.get(entity) else {
            return Vec::new();
        };
        let mut sentences: Vec<String> = Vec::new();
        let mut predicate_names: Vec<&String> = predicates.keys().collect();
        predicate_names.sort();
        for predicate in predicate_names {
            let duplicated = self
                .by_subject
                .get(entity)
                .map_or(false, |forward| forward.contains_key(predicate));
            if duplicated {
                continue;
            }
            let subjects = &predicates[predicate];
            sentences.push(self.lexicalize(&subjects.join("、"), predicate, &[entity.to_string()]));
            if sentences.len() >= limit {
                break;
            }
        }
        sentences
    }

    fn predicate_idf(&self, predicate: &str) -> f64 {
        let count = self.predicate_totals.get(predicate).copied().unwrap_or(0);
        ((1.0 + self.total_relations as f64) / (1.0 + count as f64)).ln()
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先，"
        } else if index + 1 == total {
            "最后，"
        } else {
            "此外，"
        }
    }

    // 统计式续写：给定前缀「X的」，预测下一个谓词并生成句子。
    pub fn complete(&self, prefix: &str) -> Option<String> {
        let trimmed = prefix
            .trim()
            .trim_end_matches(['。', '？', '?', '！', '!', ' ']);
        let entity = trimmed.strip_suffix('的')?.trim();
        if entity.is_empty() || !self.known_entities.contains(entity) {
            return None;
        }

        let predictions = self
            .entity_level_predictions(entity)
            .or_else(|| self.type_level_predictions(entity))
            .or_else(|| self.global_predictions());
        let predictions = predictions?;
        if predictions.is_empty() {
            return None;
        }

        let mut output = String::from("符号续写（知识图谱统计回退，无神经网络）：\n");
        output.push_str(&format!("  前缀：「{}的」\n", entity));
        for (index, prediction) in predictions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 谓词「{}」（{}，支持度 {}，置信度 {:.2}）：{}\n",
                index + 1,
                prediction.predicate,
                prediction.level.label(),
                prediction.support,
                prediction.score,
                prediction.sentence
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn entity_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let predicate_counts = self.entity_predicate_counts.get(entity)?;
        let total = self
            .entity_relation_total
            .get(entity)
            .copied()
            .unwrap_or(1)
            .max(1);
        let mut items: Vec<(&String, &usize)> = predicate_counts.iter().collect();
        // 取前 3 名
        if items.len() > 3 {
            items.select_nth_unstable_by(2, |a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            items.truncate(3);
        }
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter().take(3) {
            let objects = self
                .by_subject
                .get(entity)
                .and_then(|m| m.get(predicate))
                .cloned()
                .unwrap_or_default();
            if objects.is_empty() {
                continue;
            }
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Entity,
                score: *count as f64 / total as f64,
                support: format!("{}/{}", count, total),
                sentence: self.lexicalize(entity, predicate, &objects),
            });
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn type_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let etype = self.entity_types.get(entity)?;
        self.with_type_counts(etype, |type_counts, type_total| {
            if type_counts.is_empty() || type_total == 0 {
                return None;
            }
            let mut items: Vec<(&String, &usize)> = type_counts.iter().collect();
            items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let mut predictions: Vec<CompletionPrediction> = Vec::new();
            for (predicate, count) in items.into_iter() {
                let score = *count as f64 / type_total as f64;
                if score < self.completion_min_score {
                    // 已按计数降序，后续得分只会更低
                    break;
                }
                let Some((prototype, objects)) = self.prototype_for(etype, entity, predicate)
                else {
                    continue;
                };
                predictions.push(CompletionPrediction {
                    predicate: predicate.clone(),
                    level: CompletionLevel::Type,
                    score,
                    support: format!("{}/{}", count, type_total),
                    sentence: format!(
                        "参照同类实体「{}」：{}的{}很可能是{}。",
                        prototype,
                        entity,
                        predicate,
                        objects.join("、")
                    ),
                });
                if predictions.len() >= 3 {
                    break;
                }
            }
            if predictions.is_empty() {
                None
            } else {
                Some(predictions)
            }
        })
    }
}

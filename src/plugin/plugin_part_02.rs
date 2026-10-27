impl PluginRuntime {
    pub fn with_settings(registry: PluginRegistry, settings: &PluginSettings) -> Self {
        let manifests = registry.manifests().to_vec();
        let mut desired = BTreeMap::new();
        for manifest in &manifests {
            desired.insert(manifest.id.clone(), manifest.default_mode);
        }
        for id in &settings.enabled {
            if manifests.iter().any(|manifest| &manifest.id == id) {
                desired.insert(id.clone(), PluginMode::On);
            }
        }
        for (id, mode) in &settings.modes {
            if manifests.iter().any(|manifest| &manifest.id == id) {
                desired.insert(id.clone(), *mode);
            }
        }
        let unknown: BTreeSet<&str> = settings
            .enabled
            .iter()
            .map(String::as_str)
            .chain(settings.modes.keys().map(String::as_str))
            .filter(|id| !manifests.iter().any(|manifest| manifest.id == *id))
            .collect();
        let mut runtime = PluginRuntime {
            registry,
            manifests,
            modes: BTreeMap::new(),
            overrides: BTreeMap::new(),
            activation_issues: Vec::new(),
            last_trace: None,
            last_used: Vec::new(),
        };
        runtime.resolve_initial(desired);
        runtime
            .activation_issues
            .extend(unknown.into_iter().map(|id| {
                PluginIssue::warning(
                    Some(id),
                    "config.plugins",
                    format!("配置引用了未安装的插件「{}」，已忽略。", id),
                )
            }));
        runtime
    }

    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    pub fn installed_count(&self) -> usize {
        self.manifests.len()
    }

    pub fn enabled_count(&self) -> usize {
        self.modes
            .values()
            .filter(|mode| **mode == PluginMode::On)
            .count()
    }

    pub fn mode(&self, id: &str) -> Option<PluginMode> {
        self.modes.get(id).copied()
    }

    pub fn modes(&self) -> &BTreeMap<String, PluginMode> {
        &self.modes
    }

    pub fn manifests(&self) -> &[PluginManifest] {
        &self.manifests
    }

    pub fn issues(&self) -> Vec<PluginIssue> {
        let mut issues = self.registry.issues().to_vec();
        issues.extend(self.activation_issues.iter().cloned());
        issues
    }

    pub fn last_trace(&self) -> Option<&PluginTrace> {
        self.last_trace.as_ref()
    }

    pub fn session_state(&self) -> PluginSessionState {
        PluginSessionState {
            modes: self.overrides.clone(),
            last_used: self.last_used.clone(),
        }
    }

    /// 恢复用户显式覆盖；返回成功应用的条数。旧包/冲突包会留在诊断里。
    pub fn restore_session_state(&mut self, state: PluginSessionState) -> usize {
        let mut applied = 0;
        for (id, mode) in state.modes {
            match self.set_mode(&id, mode) {
                Ok(_) => applied += 1,
                Err(message) => self.activation_issues.push(PluginIssue::warning(
                    Some(&id),
                    "session",
                    format!("会话状态未恢复：{}", message),
                )),
            }
        }
        self.last_used = state
            .last_used
            .into_iter()
            .filter(|id| self.modes.contains_key(id))
            .collect();
        applied
    }

    pub fn manifest(&self, id: &str) -> Option<&PluginManifest> {
        self.manifests.iter().find(|manifest| manifest.id == id)
    }

    /// 显式切换模式。启用会递归拉起依赖；停用若被其他启用包依赖则拒绝。
    pub fn set_mode(&mut self, id: &str, mode: PluginMode) -> Result<Vec<String>, String> {
        if !self.modes.contains_key(id) {
            return Err(format!("没有已安装的插件「{}」。", id));
        }
        let previous_modes = self.modes.clone();
        let previous_overrides = self.overrides.clone();
        let mut proposed = self.modes.clone();
        let mut auto_enabled = Vec::new();

        match mode {
            PluginMode::Off | PluginMode::Observe => {
                let dependents = self.enabled_dependents(id);
                if !dependents.is_empty() {
                    return Err(format!(
                        "「{}」仍被启用中的插件依赖：{}。请先停用依赖它的插件。",
                        id,
                        dependents.join("、")
                    ));
                }
                proposed.insert(id.to_string(), mode);
            }
            PluginMode::On => {
                proposed.insert(id.to_string(), PluginMode::On);
                let mut queue = vec![id.to_string()];
                let mut seen = BTreeSet::new();
                while let Some(current) = queue.pop() {
                    if !seen.insert(current.clone()) {
                        continue;
                    }
                    let Some(manifest) = self.manifest(&current) else {
                        return Err(format!("依赖的插件「{}」未安装。", current));
                    };
                    for required in manifest.requires.clone() {
                        if self.manifest(&required).is_none() {
                            return Err(format!(
                                "插件「{}」依赖的「{}」未安装，无法启用。",
                                manifest.id, required
                            ));
                        }
                        if proposed.get(&required) != Some(&PluginMode::On) {
                            proposed.insert(required.clone(), PluginMode::On);
                            if required != id {
                                auto_enabled.push(required.clone());
                            }
                        }
                        queue.push(required);
                    }
                }
                if let Some((left, right)) = self.first_conflict(&proposed) {
                    return Err(format!(
                        "插件「{}」与「{}」声明冲突，不能同时启用。",
                        left, right
                    ));
                }
            }
        }

        self.modes = proposed;
        self.overrides.insert(id.to_string(), mode);
        for dependency in &auto_enabled {
            self.overrides.insert(dependency.clone(), PluginMode::On);
        }
        self.resolve_activation_issues();
        if self.modes.get(id) != Some(&mode) {
            self.modes = previous_modes;
            self.overrides = previous_overrides;
            return Err(format!("启用「{}」会破坏依赖/冲突约束，已回滚。", id));
        }
        auto_enabled.sort();
        auto_enabled.dedup();
        Ok(auto_enabled)
    }

    fn enabled_dependents(&self, id: &str) -> Vec<String> {
        self.manifests
            .iter()
            .filter(|manifest| {
                manifest.id != id
                    && self.modes.get(&manifest.id) == Some(&PluginMode::On)
                    && manifest.requires.iter().any(|required| required == id)
            })
            .map(|manifest| manifest.display_name().to_string())
            .collect()
    }

    fn resolve_initial(&mut self, desired: BTreeMap<String, PluginMode>) {
        let mut modes = desired;
        let mut blocked = BTreeSet::new();
        let mut issues = Vec::new();
        let ids: Vec<String> = self.manifests.iter().map(|m| m.id.clone()).collect();
        let mut rounds = 0;
        loop {
            rounds += 1;
            let mut changed = false;
            for id in &ids {
                if modes.get(id) != Some(&PluginMode::On) {
                    continue;
                }
                let Some(manifest) = self.manifest(id) else {
                    continue;
                };
                for required in manifest.requires.clone() {
                    if self.manifest(&required).is_none() {
                        modes.insert(id.clone(), PluginMode::Off);
                        blocked.insert(id.clone());
                        issues.push(PluginIssue::warning(
                            Some(id),
                            "requires",
                            format!("依赖的插件「{}」未安装，已停用。", required),
                        ));
                        changed = true;
                        break;
                    }
                    if blocked.contains(&required) {
                        modes.insert(id.clone(), PluginMode::Off);
                        blocked.insert(id.clone());
                        issues.push(PluginIssue::warning(
                            Some(id),
                            "requires",
                            format!("依赖的插件「{}」不可用，已停用。", required),
                        ));
                        changed = true;
                        break;
                    }
                    if modes.get(&required) != Some(&PluginMode::On) {
                        modes.insert(required.clone(), PluginMode::On);
                        changed = true;
                    }
                }
            }
            for left_index in 0..ids.len() {
                for right_index in left_index + 1..ids.len() {
                    let left = &ids[left_index];
                    let right = &ids[right_index];
                    if modes.get(left) != Some(&PluginMode::On)
                        || modes.get(right) != Some(&PluginMode::On)
                    {
                        continue;
                    }
                    if self.conflict_between(left, right) {
                        let loser = right.clone();
                        modes.insert(loser.clone(), PluginMode::Off);
                        blocked.insert(loser.clone());
                        issues.push(PluginIssue::warning(
                            Some(&loser),
                            "conflicts",
                            format!("与「{}」冲突，按 id 排序停用后者。", left),
                        ));
                        changed = true;
                    }
                }
            }
            if !changed || rounds > ids.len() * 2 + 2 {
                break;
            }
        }
        self.modes = modes;
        self.activation_issues = dedup_issues(issues);
    }

    fn resolve_activation_issues(&mut self) {
        // 显式 set_mode 已做前置检查；这里只同步诊断，不改动模式。
        let mut issues = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for required in &manifest.requires {
                if self.modes.get(required) != Some(&PluginMode::On) {
                    issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "requires",
                        format!("依赖的插件「{}」未启用。", required),
                    ));
                }
            }
        }
        if let Some((left, right)) = self.first_conflict(&self.modes) {
            issues.push(PluginIssue::warning(
                Some(&left),
                "conflicts",
                format!("与「{}」同时启用。", right),
            ));
        }
        self.activation_issues = dedup_issues(issues);
    }

    fn first_conflict(&self, modes: &BTreeMap<String, PluginMode>) -> Option<(String, String)> {
        let ids: Vec<&String> = modes
            .iter()
            .filter(|(_, mode)| **mode == PluginMode::On)
            .map(|(id, _)| id)
            .collect();
        for left_index in 0..ids.len() {
            for right_index in left_index + 1..ids.len() {
                if self.conflict_between(ids[left_index], ids[right_index]) {
                    return Some((ids[left_index].clone(), ids[right_index].clone()));
                }
            }
        }
        None
    }

    fn conflict_between(&self, left: &str, right: &str) -> bool {
        let left_conflicts = self
            .manifest(left)
            .map(|manifest| manifest.conflicts.iter().any(|id| id == right))
            .unwrap_or(false);
        let right_conflicts = self
            .manifest(right)
            .map(|manifest| manifest.conflicts.iter().any(|id| id == left))
            .unwrap_or(false);
        left_conflicts || right_conflicts
    }

    /// 词法 → 改写/配方 →（可选）观察。返回生效文本与完整 trace。
    pub fn rewrite(&mut self, input: &str) -> PluginOutcome {
        let original = input;
        // 问句末尾的标点是排版不是语义：只用于匹配，不把「？」带进槽位。
        // 若没有任何插件实际改写，最终仍原样返回，避免影响核心管线。
        let match_input = trim_query_surface(input);
        let mut trace = PluginTrace {
            input: original.to_string(),
            effective: original.to_string(),
            ..PluginTrace::default()
        };
        if self.manifests.is_empty() {
            self.last_trace = Some(trace.clone());
            return PluginOutcome {
                text: trace.effective.clone(),
                trace,
            };
        }

        // 1. Lexicon + ontology aliases：最长匹配、单遍替换；观察模式只记账，
        //    且匹配的是原文（不会出现「电影」再被「影」二次替换的堆叠错误）。
        let lexicon = self.lexicon_pairs();
        let mut text = match_input.to_string();
        if !lexicon.is_empty() {
            let (normalized, matches) = apply_lexicon(match_input, &lexicon);
            text = normalized;
            trace.matches.extend(matches);
        }
        trace.effective = text.clone();

        // 2. Rewrite / Recipe：最多连续三轮，避免插件互相改写形成环。
        let mut seen = BTreeSet::new();
        seen.insert(text.clone());
        for _ in 0..MAX_REWRITE_PASSES {
            let candidates = self.match_frame(&text);
            if candidates.is_empty() {
                break;
            }
            let observed: Vec<&FrameCandidate> = candidates
                .iter()
                .filter(|candidate| candidate.mode == PluginMode::Observe)
                .collect();
            for candidate in observed {
                trace.matches.push(PluginMatch {
                    plugin: candidate.plugin.clone(),
                    contribution: candidate.contribution.clone(),
                    kind: candidate.kind.to_string(),
                    mode: PluginMode::Observe,
                    applied: false,
                    detail: candidate.detail.clone(),
                });
            }
            let on_candidates: Vec<&FrameCandidate> = candidates
                .iter()
                .filter(|candidate| candidate.mode == PluginMode::On)
                .collect();
            let Some(chosen) = on_candidates.iter().copied().max_by(|left, right| {
                left.specificity
                    .cmp(&right.specificity)
                    .then_with(|| left.recipe.cmp(&right.recipe))
                    .then_with(|| right.plugin.cmp(&left.plugin))
                    .then_with(|| right.contribution.cmp(&left.contribution))
            }) else {
                break;
            };
            let distinct: BTreeSet<&str> = on_candidates
                .iter()
                .map(|candidate| candidate.output.as_str())
                .collect();
            if distinct.len() > 1 {
                trace.notes.push(format!(
                    "多个启用插件命中同一问法，按具体度与 id 选择「{} / {}」。",
                    chosen.plugin, chosen.contribution
                ));
            }
            trace.matches.push(PluginMatch {
                plugin: chosen.plugin.clone(),
                contribution: chosen.contribution.clone(),
                kind: chosen.kind.to_string(),
                mode: PluginMode::On,
                applied: true,
                detail: chosen.detail.clone(),
            });
            if let Some(recipe) = &chosen.recipe_invocation {
                trace.recipe = Some(recipe.clone());
            } else {
                // 后续普通改写改变了规范问句：先前配方的答案与当前问法不一致。
                trace.recipe = None;
            }
            let recipe_applied = chosen.recipe_invocation.is_some();
            let next = chosen.output.clone();
            if next == text || !seen.insert(next.clone()) {
                trace.notes.push("改写出现环或空转，已停止。".to_string());
                break;
            }
            text = next;
            trace.effective = text.clone();
            // 配方输出就是规范问句，是改写链的终点；继续套用其它模式会把
            // 「主演过…」里的「演过」再次当成槽位，叠出「主主主演」。
            if recipe_applied {
                break;
            }
        }

        if text == match_input && text != original {
            text = original.to_string();
            trace.effective = text.clone();
        }

        let used = trace.used_plugins();
        if !used.is_empty() {
            self.last_used = used;
        }
        self.last_trace = Some(trace.clone());
        PluginOutcome { text, trace }
    }

    /// 启动时合并进既有谓词对齐器的本体别名（仅启用中的插件）。
    /// 与 `lexicon_pairs` 不同，这里提交给推理器，使图谱侧查询也能识别别名。
    pub fn ontology_aliases(&self) -> Vec<(String, String)> {
        let mut aliases = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Ontology { predicates, .. } = contribution {
                    for predicate in predicates {
                        for alias in &predicate.aliases {
                            if !alias.trim().is_empty()
                                && alias != &predicate.name
                                && !predicate.name.trim().is_empty()
                            {
                                aliases.push((alias.clone(), predicate.name.clone()));
                            }
                        }
                    }
                }
            }
        }
        aliases.sort();
        aliases.dedup();
        aliases
    }

    /// 启动时把插件的本体贡献归约为统一本体程序（仅启用中的插件）。
    ///
    /// 别名不再只是词面归一：类型别名编译成 `别名 ⊑ 规范类`，谓词别名编译成
    /// `别名 ⊑ 规范关系`，让**已落库的数据**使用别名书写时也能被规范词面
    /// 查询命中；子类、互斥、定义域/值域、逆关系、传递、对称同样编译成
    /// `OntologyLesson`，交给宿主的单一本体推理器。重复声明由程序自身的
    /// 规范去重吸收，合并是幂等的。
    pub fn ontology_program(&self) -> OntologyProgram {
        let mut lessons = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                let Contribution::Ontology {
                    predicates, types, ..
                } = contribution
                else {
                    continue;
                };
                for entity_type in types {
                    let name = entity_type.name.trim();
                    if name.is_empty() {
                        continue;
                    }
                    for alias in &entity_type.aliases {
                        let alias = alias.trim();
                        if alias.is_empty() || alias == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Subclass {
                            sub: alias.to_string(),
                            sup: name.to_string(),
                            original: format!("{}是{}的别名", alias, name),
                        });
                    }
                    for sup in &entity_type.subclass_of {
                        let sup = sup.trim();
                        if sup.is_empty() || sup == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Subclass {
                            sub: name.to_string(),
                            sup: sup.to_string(),
                            original: format!("{}是一种{}", name, sup),
                        });
                    }
                    for right in &entity_type.disjoint_with {
                        let right = right.trim();
                        if right.is_empty() || right == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Disjoint {
                            left: name.to_string(),
                            right: right.to_string(),
                            original: format!("{}和{}互斥", name, right),
                        });
                    }
                }
                for predicate in predicates {
                    let name = predicate.name.trim();
                    if name.is_empty() {
                        continue;
                    }
                    for alias in &predicate.aliases {
                        let alias = alias.trim();
                        if alias.is_empty() || alias == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Subproperty {
                            sub: alias.to_string(),
                            sup: name.to_string(),
                            original: format!("{}是{}的别名", alias, name),
                        });
                    }
                    for class in &predicate.domain {
                        let class = class.trim();
                        if !class.is_empty() {
                            lessons.push(OntologyLesson::Domain {
                                property: name.to_string(),
                                class: class.to_string(),
                                original: format!("{}关系的定义域是{}", name, class),
                            });
                        }
                    }
                    for class in &predicate.range {
                        let class = class.trim();
                        if !class.is_empty() {
                            lessons.push(OntologyLesson::Range {
                                property: name.to_string(),
                                class: class.to_string(),
                                original: format!("{}关系的值域是{}", name, class),
                            });
                        }
                    }
                    if let Some(inverse) = predicate.inverse.as_deref() {
                        let inverse = inverse.trim();
                        if !inverse.is_empty() && inverse != name {
                            lessons.push(OntologyLesson::Inverse {
                                left: name.to_string(),
                                right: inverse.to_string(),
                                original: format!("{}和{}互为逆关系", name, inverse),
                            });
                        }
                    }
                    if predicate.transitive {
                        lessons.push(OntologyLesson::Transitive {
                            property: name.to_string(),
                            original: format!("{}关系可以传递", name),
                        });
                    }
                    if predicate.symmetric {
                        lessons.push(OntologyLesson::Symmetric {
                            property: name.to_string(),
                            original: format!("{}是对称的", name),
                        });
                    }
                }
            }
        }
        OntologyProgram::from_lessons(lessons)
    }

    /// 插件声明的互逆谓词中「需要换向查询」的词面（仅启用中的插件）。
    /// 与 `ontology_aliases` 不同：这里登记的词面在发问时把主宾对调，
    /// 例如「执导」问句命中「导演」事实的反方向。
    pub fn ontology_inverse_surfaces(&self) -> Vec<(String, String)> {
        let mut surfaces = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Ontology { predicates, .. } = contribution {
                    for predicate in predicates {
                        let name = predicate.name.trim();
                        let Some(inverse) = predicate.inverse.as_deref() else {
                            continue;
                        };
                        let inverse = inverse.trim();
                        if !name.is_empty() && !inverse.is_empty() && inverse != name {
                            surfaces.push((inverse.to_string(), name.to_string()));
                        }
                    }
                }
            }
        }
        surfaces.sort();
        surfaces.dedup();
        surfaces
    }

    /// 当前启用插件声明的指称词是否可回指对话焦点。
    /// 静态代词之外，插件 `Reference.quantifiers` 也参与判定；量词后紧跟
    /// 本包 `role_words` 的偏正短语（如「这位导演」「该演员」）同样视为
    /// 回指，而不是一个需要检索的新实体。
    /// 只认启用中的包，观察/停用包不改变消解行为。
    pub fn is_context_reference(&self, surface: &str) -> bool {
        if is_reference_surface(surface) {
            return true;
        }
        let surface = surface.trim();
        if surface.is_empty() {
            return false;
        }
        self.manifests.iter().any(|manifest| {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                return false;
            }
            manifest.contributions.iter().any(|contribution| {
                let Contribution::Reference {
                    quantifiers,
                    role_words,
                    ..
                } = contribution
                else {
                    return false;
                };
                quantifiers.iter().any(|quantifier| {
                    let quantifier = quantifier.trim();
                    if quantifier.is_empty() {
                        return false;
                    }
                    if quantifier == surface {
                        return true;
                    }
                    surface
                        .strip_prefix(quantifier)
                        .map(str::trim)
                        .is_some_and(|rest| {
                            role_words
                                .iter()
                                .any(|role| !role.trim().is_empty() && role.trim() == rest)
                        })
                })
            })
        })
    }

    /// 启动时合并进中文分词/词性词典的指称线索（仅启用中的插件）。
    /// 量词按代词 `r`、角色词按名词 `n`、姓氏按人名 `nr` 标注。
    pub fn reference_lexicon(&self) -> Vec<(String, String)> {
        let mut entries = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Reference {
                    quantifiers,
                    role_words,
                    surnames,
                    ..
                } = contribution
                {
                    for quantifier in quantifiers {
                        if !quantifier.trim().is_empty() {
                            entries.push((quantifier.clone(), "r".to_string()));
                        }
                    }
                    for role in role_words {
                        if !role.trim().is_empty() {
                            entries.push((role.clone(), "n".to_string()));
                        }
                    }
                    for surname in surnames {
                        if !surname.trim().is_empty() {
                            entries.push((surname.clone(), "nr".to_string()));
                        }
                    }
                }
            }
        }
        entries.sort();
        entries.dedup();
        entries
    }

    /// 派生 NLP 配置：保留宿主原有词性标注，只补入插件指称线索。
    pub fn augmented_nlp_config(&self, base: &crate::NlpConfig) -> crate::NlpConfig {
        let mut config = base.clone();
        for (surface, pos) in self.reference_lexicon() {
            config.pos_dict.entry(surface).or_insert(pos);
        }
        config
    }

    fn has_ontology(&self, id: &str) -> bool {
        self.manifest(id)
            .map(|manifest| {
                manifest
                    .contributions
                    .iter()
                    .any(|contribution| matches!(contribution, Contribution::Ontology { .. }))
            })
            .unwrap_or(false)
    }

    fn lexicon_pairs(&self) -> Vec<(String, String, String, String, PluginMode)> {
        let mut pairs = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                match contribution {
                    Contribution::Lexicon { id, entries, .. } => {
                        for entry in entries {
                            // 空词面永远匹配（starts_with("")恒真）却不推进游标，
                            // 会让单遍替换原地打转：校验层已按 Error 废包，这里再兜底。
                            if entry.surface.trim().is_empty() || entry.canonical.trim().is_empty() {
                                continue;
                            }
                            pairs.push((
                                entry.surface.clone(),
                                entry.canonical.clone(),
                                manifest.id.clone(),
                                id.clone(),
                                mode,
                            ));
                        }
                    }
                    Contribution::Ontology {
                        id,
                        predicates,
                        types,
                        ..
                    } => {
                        for predicate in predicates {
                            let name = predicate.name.trim();
                            if name.is_empty() {
                                continue;
                            }
                            for alias in &predicate.aliases {
                                let alias = alias.trim();
                                // 与校验层「空别名/自指别名已忽略」的承诺保持一致：
                                // 这些词面不进词法替换，既不空转也不产生假命中。
                                if alias.is_empty() || alias == name {
                                    continue;
                                }
                                pairs.push((
                                    alias.to_string(),
                                    name.to_string(),
                                    manifest.id.clone(),
                                    id.clone(),
                                    mode,
                                ));
                            }
                        }
                        for entity_type in types {
                            let name = entity_type.name.trim();
                            if name.is_empty() {
                                continue;
                            }
                            for alias in &entity_type.aliases {
                                let alias = alias.trim();
                                if alias.is_empty() || alias == name {
                                    continue;
                                }
                                pairs.push((
                                    alias.to_string(),
                                    name.to_string(),
                                    manifest.id.clone(),
                                    id.clone(),
                                    mode,
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        pairs.sort_by(|left, right| {
            right
                .0
                .chars()
                .count()
                .cmp(&left.0.chars().count())
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
        });
        pairs
    }

    fn match_frame(&self, text: &str) -> Vec<FrameCandidate> {
        let mut candidates = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                match contribution {
                    Contribution::Rewrite {
                        id,
                        pattern,
                        replacement,
                        ..
                    } => {
                        let Ok(pattern_segments) = parse_template(pattern) else {
                            continue;
                        };
                        let Ok(replacement_segments) = parse_template(replacement) else {
                            continue;
                        };
                        let Some(captures) = match_segments(&pattern_segments, text) else {
                            continue;
                        };
                        let Ok(output) = render_segments(&replacement_segments, &captures) else {
                            continue;
                        };
                        candidates.push(FrameCandidate {
                            plugin: manifest.id.clone(),
                            contribution: id.clone(),
                            kind: "rewrite",
                            mode,
                            specificity: literal_length(&pattern_segments),
                            output: output.clone(),
                            detail: format!("{} → {}", text, output),
                            recipe_invocation: None,
                            recipe: false,
                        });
                    }
                    Contribution::Recipe {
                        id,
                        patterns,
                        question,
                        template,
                        attribute,
                        direction,
                        answer_types,
                        ..
                    } => {
                        for pattern in patterns {
                            let Ok(pattern_segments) = parse_template(pattern) else {
                                continue;
                            };
                            let Some(captures) = match_segments(&pattern_segments, text) else {
                                continue;
                            };
                            let Ok(question_segments) = parse_template(question) else {
                                continue;
                            };
                            let Ok(output) = render_segments(&question_segments, &captures) else {
                                continue;
                            };
                            candidates.push(FrameCandidate {
                                plugin: manifest.id.clone(),
                                contribution: id.clone(),
                                kind: "recipe",
                                mode,
                                specificity: literal_length(&pattern_segments),
                                output: output.clone(),
                                detail: format!("{} → {}（配方）", text, output),
                                recipe_invocation: Some(RecipeInvocation {
                                    plugin: manifest.id.clone(),
                                    contribution: id.clone(),
                                    question: output,
                                    template: template.clone(),
                                    attribute: attribute.clone(),
                                    direction: *direction,
                                    answer_types: answer_types.clone(),
                                    slots: captures,
                                }),
                                recipe: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        candidates
    }

    /// 用配方选中的模板渲染答案；任一槽位未接地则回退原答案并记 trace。
    pub fn compose_answer(
        &mut self,
        outcome: &PluginOutcome,
        answer: &str,
    ) -> PluginComposedAnswer {
        self.compose_answer_with_evidence(outcome, answer, &[])
    }

    /// 与 [`Self::compose_answer`] 相同，但额外提供 `{evidence}` 槽位。
    /// 依据必须由调用方从事实/公理闭包取出，插件模板只能复述、不能编造。
    pub fn compose_answer_with_evidence(
        &mut self,
        outcome: &PluginOutcome,
        answer: &str,
        evidence: &[String],
    ) -> PluginComposedAnswer {
        let mut trace = outcome.trace.clone();
        let mut text = answer.to_string();
        let mut notes = Vec::new();
        let mut template_applied = None;
        if let Some(recipe) = &outcome.trace.recipe {
            if let Some(template_id) = &recipe.template {
                let template = self.template_contribution(template_id);
                match template {
                    Some((plugin, mode, segments)) if mode == PluginMode::On => {
                        let mut slots = recipe.slots.clone();
                        if !answer.trim().is_empty() {
                            slots.insert("answer".to_string(), answer.to_string());
                        }
                        if let Some(attribute) = &recipe.attribute {
                            slots.insert("attribute".to_string(), attribute.clone());
                        }
                        if !evidence.is_empty() {
                            slots.insert("evidence".to_string(), evidence.join("；"));
                        }
                        slots.insert("plugin".to_string(), recipe.plugin.clone());
                        slots.insert("recipe".to_string(), recipe.contribution.clone());
                        match render_segments(&segments, &slots) {
                            Ok(rendered) => {
                                text = rendered;
                                template_applied = Some(template_id.clone());
                                trace.matches.push(PluginMatch {
                                    plugin: plugin.clone(),
                                    contribution: template_id.clone(),
                                    kind: "template".to_string(),
                                    mode,
                                    applied: true,
                                    detail: "槽位接地通过".to_string(),
                                });
                            }
                            Err(message) => {
                                notes.push(format!(
                                    "模板「{}」未渲染，回退原答案：{}",
                                    template_id, message
                                ));
                            }
                        }
                    }
                    Some((plugin, mode, _)) => {
                        notes.push(format!(
                            "模板「{}」属于{}模式插件「{}」，未接管。",
                            template_id,
                            mode.label(),
                            plugin
                        ));
                    }
                    None => notes.push(format!("模板「{}」不存在，回退原答案。", template_id)),
                }
            }
        }

        let (styled, style_matches) = self.style_with_matches(&text);
        text = styled;
        trace.matches.extend(style_matches);
        trace.notes.extend(notes.clone());
        self.last_trace = Some(trace);
        PluginComposedAnswer {
            text,
            template: template_applied,
            notes,
        }
    }

    fn template_contribution(&self, id: &str) -> Option<(String, PluginMode, Vec<Segment>)> {
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Template {
                    id: template_id,
                    template,
                    ..
                } = contribution
                {
                    if template_id == id {
                        if let Ok(segments) = parse_template(template) {
                            return Some((manifest.id.clone(), mode, segments));
                        }
                    }
                }
            }
        }
        None
    }

    /// 核心管线答不了时的声明式兜底策略：按触发词长度、插件 id、贡献 id 择优。
    ///
    /// 只返回文本与可审计的贡献路径，不伪造知识；调用方仍需决定是否采纳。
    pub fn policy_reply(&mut self, input: &str) -> Option<(String, String)> {
        let mut best: Option<(String, String, String, usize, String)> = None;
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                let Contribution::Policy {
                    id, when, reply, ..
                } = contribution
                else {
                    continue;
                };
                for trigger in when {
                    if trigger.trim().is_empty() || !input.contains(trigger.as_str()) {
                        continue;
                    }
                    let score = trigger.chars().count();
                    let better = best.as_ref().is_none_or(|current| {
                        score > current.3
                            || (score == current.3
                                && (manifest.id.as_str(), id.as_str())
                                    < (current.0.as_str(), current.1.as_str()))
                    });
                    if better {
                        best = Some((
                            manifest.id.clone(),
                            id.clone(),
                            reply.clone(),
                            score,
                            trigger.clone(),
                        ));
                    }
                }
            }
        }
        let (plugin, contribution, reply, _, trigger) = best?;
        let trace = self.last_trace.get_or_insert_with(|| PluginTrace {
            input: input.to_string(),
            effective: input.to_string(),
            ..PluginTrace::default()
        });
        trace.matches.push(PluginMatch {
            plugin: plugin.clone(),
            contribution: contribution.clone(),
            kind: "policy".to_string(),
            mode: PluginMode::On,
            applied: true,
            detail: format!("触发词「{}」", trigger),
        });
        if !self.last_used.contains(&plugin) {
            self.last_used.push(plugin.clone());
            self.last_used.sort();
            self.last_used.dedup();
        }
        Some((reply, format!("plugin:{}:{}", plugin, contribution)))
    }

    /// 只给文本加固定前后缀；不改变事实句。非启用风格包不参与。
    pub fn style_wrap(&self, text: &str) -> String {
        self.style_with_matches(text).0
    }

    fn style_with_matches(&self, text: &str) -> (String, Vec<PluginMatch>) {
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let mut matches = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if mode != PluginMode::On {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Style {
                    id, prefix, suffix, ..
                } = contribution
                {
                    if prefix.is_empty() && suffix.is_empty() {
                        continue;
                    }
                    if !prefix.is_empty() {
                        prefixes.push(prefix.clone());
                    }
                    if !suffix.is_empty() {
                        suffixes.push(suffix.clone());
                    }
                    matches.push(PluginMatch {
                        plugin: manifest.id.clone(),
                        contribution: id.clone(),
                        kind: "style".to_string(),
                        mode,
                        applied: true,
                        detail: format!(
                            "{}{}",
                            if prefix.is_empty() { "" } else { "前缀" },
                            if suffix.is_empty() { "" } else { "后缀" }
                        ),
                    });
                }
            }
        }
        if prefixes.is_empty() && suffixes.is_empty() {
            return (text.to_string(), matches);
        }
        let mut out = String::new();
        for prefix in prefixes {
            out.push_str(&prefix);
        }
        out.push_str(text);
        for suffix in suffixes {
            out.push_str(&suffix);
        }
        (out, matches)
    }

    /// 处理一条插件命令；不匹配时返回 None（放行常规查询）。
    pub fn handle_command(&mut self, input: &str) -> Option<PluginCommandOutcome> {
        let command = parse_plugin_command(input)?;
        let outcome = match command {
            ParsedPluginCommand::Report => PluginCommandOutcome {
                text: self.render_report(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Diagnose => PluginCommandOutcome {
                text: self.render_diagnose(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Evaluate => {
                let results = self.registry.evaluate();
                let failures = results.iter().filter(|result| !result.passed).count();
                PluginCommandOutcome {
                    text: self.registry.render_eval(&results),
                    state_changed: false,
                    success: failures == 0,
                }
            }
            ParsedPluginCommand::Why => PluginCommandOutcome {
                text: self.render_trace(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Enable(id) => match self.set_mode(&id, PluginMode::On) {
                Ok(auto) => PluginCommandOutcome {
                    text: {
                        let mut text = format!(
                            "已启用插件「{}」{}。",
                            self.manifest(&id)
                                .map(PluginManifest::display_name)
                                .unwrap_or(&id),
                            if auto.is_empty() {
                                String::new()
                            } else {
                                format!("（同时启用依赖：{}）", auto.join("、"))
                            }
                        );
                        if self.has_ontology(&id) {
                            text.push_str(
                                "\n注意：该包的本体声明在下次启动时合并；本轮先启用词表/改写/配方/模板。",
                            );
                        }
                        text.push_str("\n可用「插件」查看全部能力，「为什么这么答」查看本轮依据。");
                        text
                    },
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("启用失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
            ParsedPluginCommand::Disable(id) => match self.set_mode(&id, PluginMode::Off) {
                Ok(_) => PluginCommandOutcome {
                    text: format!(
                        "已停用插件「{}」。",
                        self.manifest(&id)
                            .map(PluginManifest::display_name)
                            .unwrap_or(&id)
                    ),
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("停用失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
            ParsedPluginCommand::SetMode(id, mode) => match self.set_mode(&id, mode) {
                Ok(auto) => PluginCommandOutcome {
                    text: format!(
                        "插件「{}」已切换为{}模式{}。",
                        id,
                        mode.label(),
                        if auto.is_empty() {
                            String::new()
                        } else {
                            format!("（同时启用依赖：{}）", auto.join("、"))
                        }
                    ),
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("模式切换失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
        };
        Some(outcome)
    }

    pub fn render_report(&self) -> String {
        let mut lines = Vec::new();
        match self.registry.directory() {
            Some(directory) => lines.push(format!(
                "插件目录：{}（已安装 {} 个，启用 {} 个，观察 {} 个）",
                directory,
                self.manifests.len(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::On)
                    .count(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::Observe)
                    .count()
            )),
            None => lines.push(format!(
                "插件：已安装 {} 个，启用 {} 个",
                self.manifests.len(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::On)
                    .count()
            )),
        }
        if self.manifests.is_empty() {
            lines.push(
                "没有已安装的插件。可在插件目录放入 manifest.toml，或用「插件」查看加载诊断。"
                    .to_string(),
            );
        }
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            let counts = manifest
                .contribution_counts()
                .iter()
                .map(|(kind, count)| format!("{}×{}", kind, count))
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!(
                "- {} v{} [{}] {}：{}{}",
                manifest.display_name(),
                manifest.version,
                mode.label(),
                manifest.id,
                if manifest.description.trim().is_empty() {
                    ""
                } else {
                    manifest.description.trim()
                },
                if counts.is_empty() {
                    String::new()
                } else {
                    format!("\n    贡献：{}", counts)
                }
            ));
            if !manifest.requires.is_empty() {
                lines.push(format!("    依赖：{}", manifest.requires.join("、")));
            }
            if !manifest.conflicts.is_empty() {
                lines.push(format!("    冲突：{}", manifest.conflicts.join("、")));
            }
        }
        if !self.last_used.is_empty() {
            lines.push(format!("最近命中：{}", self.last_used.join("、")));
        }
        let issues = self.issues();
        if !issues.is_empty() {
            lines.push(format!("诊断：{} 条", issues.len()));
            for issue in issues.iter().take(8) {
                lines.push(format!("  {}", issue.render()));
            }
            if issues.len() > 8 {
                lines.push(format!("  …共 {} 条，用「插件诊断」查看。", issues.len()));
            }
        }
        lines.push(
            "命令：启用插件：<id> / 停用插件：<id> / 插件模式：<id>=observe|on|off / 插件诊断 / 插件评测 / 为什么这么答"
                .to_string(),
        );
        lines.join("\n")
    }

    pub fn render_diagnose(&self) -> String {
        let mut lines = vec![self.render_report()];
        lines.push("清单与启用诊断：".to_string());
        let issues = self.issues();
        if issues.is_empty() {
            lines.push("  未发现问题。".to_string());
        } else {
            for issue in &issues {
                lines.push(format!("  {}", issue.render()));
            }
        }
        lines.push("本轮插件轨迹：".to_string());
        lines.push(
            self.last_trace
                .as_ref()
                .map(render_trace_body)
                .unwrap_or_else(|| "  本轮尚未处理查询。".to_string()),
        );
        lines.join("\n")
    }

    pub fn render_trace(&self) -> String {
        match &self.last_trace {
            Some(trace) => format!("本轮插件轨迹：\n{}", render_trace_body(trace)),
            None => "本轮尚未处理查询，还没有插件轨迹。".to_string(),
        }
    }
}
type LexiconPair = (String, String, String, String, PluginMode);
fn apply_lexicon(input: &str, pairs: &[LexiconPair]) -> (String, Vec<PluginMatch>) {
    let mut out = String::with_capacity(input.len());
    let mut matches = Vec::new();
    let mut observed = BTreeSet::new();
    let mut pos = 0usize;
    while pos < input.len() {
        let rest = &input[pos..];
        // 空词面必须排除：starts_with("") 恒真且 len()==0 不推进 pos，
        // 哪怕上游漏过滤也会在这里兜底，保证每轮至少吐一个字符。
        let on_match = pairs.iter().find(|(surface, _, _, _, mode)| {
            *mode == PluginMode::On && !surface.is_empty() && rest.starts_with(surface.as_str())
        });
        let observe_match = pairs.iter().find(|(surface, _, _, _, mode)| {
            *mode == PluginMode::Observe && !surface.is_empty() && rest.starts_with(surface.as_str())
        });
        if let Some((surface, canonical, plugin, contribution, mode)) = on_match {
            out.push_str(canonical);
            matches.push(PluginMatch {
                plugin: plugin.clone(),
                contribution: contribution.clone(),
                kind: "lexicon".to_string(),
                mode: *mode,
                applied: true,
                detail: format!("{} → {}", surface, canonical),
            });
            pos += surface.len();
        } else if let Some((surface, canonical, plugin, contribution, mode)) = observe_match {
            if observed.insert(surface.clone()) {
                matches.push(PluginMatch {
                    plugin: plugin.clone(),
                    contribution: contribution.clone(),
                    kind: "lexicon".to_string(),
                    mode: *mode,
                    applied: false,
                    detail: format!("{} → {}", surface, canonical),
                });
            }
            out.push_str(surface);
            pos += surface.len();
        } else {
            let ch = rest.chars().next().expect("non-empty rest");
            out.push(ch);
            pos += ch.len_utf8();
        }
    }
    (out, matches)
}
fn render_trace_body(trace: &PluginTrace) -> String {
    let mut lines = vec![
        format!("  输入：{}", trace.input),
        format!(
            "  生效：{}{}",
            trace.effective,
            if trace.changed() {
                "（已改写）"
            } else {
                "（未改写）"
            }
        ),
    ];
    if trace.matches.is_empty() {
        lines.push("  没有插件命中。".to_string());
    } else {
        for entry in &trace.matches {
            lines.push(format!(
                "  {} [{}] {} / {}（{}）{}",
                if entry.applied { "✓" } else { "○" },
                entry.mode.label(),
                entry.plugin,
                entry.contribution,
                entry.kind,
                entry.detail
            ));
        }
    }
    if let Some(recipe) = &trace.recipe {
        lines.push(format!(
            "  配方：{} / {} → {}",
            recipe.plugin, recipe.contribution, recipe.question
        ));
        if let Some(attribute) = &recipe.attribute {
            lines.push(format!("  属性：{}", attribute));
        }
    }
    for note in &trace.notes {
        lines.push(format!("  注：{}", note));
    }
    lines.join("\n")
}
fn dedup_issues(issues: Vec<PluginIssue>) -> Vec<PluginIssue> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for issue in issues {
        let key = format!(
            "{:?}|{:?}|{}|{}",
            issue.severity, issue.plugin, issue.field, issue.message
        );
        if seen.insert(key) {
            out.push(issue);
        }
    }
    out
}
#[derive(Clone, Debug)]
struct FrameCandidate {
    plugin: String,
    contribution: String,
    kind: &'static str,
    mode: PluginMode,
    specificity: usize,
    output: String,
    detail: String,
    recipe_invocation: Option<RecipeInvocation>,
    recipe: bool,
}

impl SymbolicReasoner {
    /// 开放问句的谓词链目标构造（"X的P…是谁"），答案变量固定为「答案」。
    fn build_question_goals(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        answer_var: &str,
    ) -> Vec<Atom> {
        let mut goals = Vec::new();
        let mut previous = Term::constant(entity);
        for (index, predicate) in predicates.iter().enumerate() {
            let next = if index + 1 == predicates.len() {
                Term::var(answer_var)
            } else {
                Term::var(&format!("V{}", index + 1))
            };
            let slot =
                self.proof_predicate_slot(&predicate.surface, &predicate.slot, predicate.inverted);
            let atom = if predicate.inverted {
                Atom::pattern(&slot, next.clone(), previous.clone())
            } else {
                Atom::pattern(&slot, previous.clone(), next.clone())
            };
            goals.push(atom);
            previous = next;
        }
        goals
    }

    /// 证明视图里的谓词落键：图谱常把属性存成英文键（JSON-LD 的 age/height），
    /// 用户却按中文词面（年龄/身高）提问。只有当原槽位在归结知识库里完全没有
    /// 事实与规则时，才回退到对照表的规范键，避免覆盖真实的中文谓词。
    fn proof_predicate_slot(&self, surface: &str, slot: &str, inverted: bool) -> String {
        if inverted || slot.is_empty() {
            return slot.to_string();
        }
        if !self.resolver.facts_for(slot).is_empty() || !self.resolver.clauses_for(slot).is_empty()
        {
            return slot.to_string();
        }
        if let Some(canonical) = crate::symbolic_generator::canonical_predicate_key(surface) {
            if !self.resolver.facts_for(canonical).is_empty()
                || !self.resolver.clauses_for(canonical).is_empty()
            {
                return canonical.to_string();
            }
        }
        slot.to_string()
    }

    /// 把三态搜索结论与同树失败前沿附在推理报告后面：不把截断伪装成「不存在」。
    fn render_search_diagnostics(&self, outcome: &SearchOutcome) -> String {
        let mut output = format!("  搜索状态：{}\n", outcome.render_status());
        if outcome.is_truncated() {
            output.push_str(&format!(
                "  搜索统计：{} 个目标、{} 次目标展开、{} 次事实检查、{} 次规则尝试\n",
                outcome.stats.goals,
                outcome.stats.goal_attempts,
                outcome.stats.fact_checks,
                outcome.stats.rule_attempts
            ));
        }
        if outcome.solutions.is_empty() && !outcome.frontier.is_empty() {
            const SHOWN: usize = 5;
            output.push_str("  知识缺口（来自本次搜索的实际失败叶子）：\n");
            for entry in outcome.frontier.iter().take(SHOWN) {
                output.push_str(&format!("    - {}\n", entry.describe()));
            }
            let omitted =
                outcome.frontier.len().saturating_sub(SHOWN) + outcome.stats.frontier_omitted;
            if omitted > 0 {
                output.push_str(&format!("    - ……另有 {} 条缺口未展开\n", omitted));
            }
        }
        output
    }

    fn render_resolution_result(&self, phrase: &str, goals: &[Atom]) -> String {
        let outcome = self.resolver.solve_outcome(goals, &[]);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结证明（一阶 Horn 归结 + 回溯证明树，纯符号）：\n");
        output.push_str(&format!(
            "  目标：{}  〔逻辑形式：{}〕\n",
            phrase, goal_display
        ));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则（含链接预测与关联规则）\n",
            self.resolver.fact_count(),
            self.resolver.clause_count()
        ));
        if let Some(solution) = outcome.solutions.first() {
            output.push_str(&format!("  得证：证明置信度 {:.2}\n", solution.confidence));
            output.push_str("  证明树：\n");
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
        } else if outcome.status == SearchStatus::Incomplete {
            output.push_str(&format!(
                "  未得证：在深度上限 {} 内搜索被截断，不能据此断言目标不存在（开世界假设）。\n",
                self.resolver.max_depth()
            ));
            output
                .push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        } else {
            output.push_str(
                "  未得证：搜索已穷尽所有分支，未找到归结反驳（开世界假设：推不出 ≠ 为假）。\n",
            );
            output
                .push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        }
        output.push_str(&self.render_search_diagnostics(&outcome));
        output
    }

    /// 证明一条陈述（"S的P…是O" / "S是类型" / "S有P"），输出回归结证明树
    pub fn resolution_proof_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请提供要证明的陈述，例如：证明：张三的师祖是王五 或 证明：姚明有国籍"
                .to_string();
        }
        match self.statement_goals(trimmed) {
            Some((phrase, goals)) => self.render_resolution_result(&phrase, &goals),
            None => format!(
                "无法把「{}」解析为可归结的目标。支持句式：S的P…是O / S是类型 / S有P。",
                trimmed
            ),
        }
    }

    /// 证明记忆：当前会话的命中、字节与淘汰回执（确定性，可直接断言）。
    pub fn proof_memory_report(&self) -> String {
        self.proof_memory.lock().render_receipt()
    }

    /// 清空当前会话的证明记忆；知识图谱、会话事实与编年史不受影响。
    pub fn clear_proof_memory(&self) -> String {
        let mut memory = self.proof_memory.lock();
        let before = memory.len();
        memory.clear();
        format!("证明记忆已清空：移除 {} 条。", before)
    }

    /// 与 [`Self::resolution_proof_view`] 相同，并给出缓存命中或写入回执。
    ///
    /// 只复用搜索完整的证明；被截断的搜索只作为一次性视图返回，不写入
    /// 可复用记忆。缓存键是去首尾标点后的规范请求键，同一会话内文字
    /// 变化（换说法/换实体）不会误命中。
    pub fn resolution_proof_view_with_receipt(
        &self,
        text: &str,
    ) -> Option<(ResolutionProofView, ProofMemoryAccess)> {
        if !self.config.enable_resolution {
            return None;
        }
        let key = canonical_key(text);
        if !key.is_empty() {
            let hit = self.proof_memory.lock().lookup_complete(&key);
            if let Some(hit) = hit {
                let saved_steps = hit.cost.steps().saturating_mul(hit.hits);
                let access = ProofMemoryHitAccess {
                    hits: hit.hits,
                    payload_bytes: hit.payload_bytes,
                    saved_steps,
                    status: hit.status,
                };
                return Some((hit.payload, ProofMemoryAccess::hit(key, access)));
            }
        }
        let view = self.resolution_proof_view_uncached(text)?;
        let incomplete = view.search_truncated || view.search_status == SearchStatus::Incomplete;
        if incomplete || key.is_empty() {
            let mut access = ProofMemoryAccess::miss(key);
            if incomplete {
                access.status = ProofStatus::MayBeIncomplete;
            }
            return Some((view, access));
        }
        let record = ProofRecord::new(
            key.clone(),
            view.clone(),
            estimate_resolution_view_bytes(&view),
        )
        .with_footprint(resolution_view_footprint(&view))
        .with_cost(proof_search_cost(&view.stats))
        .with_status(ProofStatus::Complete);
        let receipt = self.proof_memory.lock().store(record);
        Some((view, ProofMemoryAccess::miss(key).with_store(&receipt)))
    }

    /// 结构化证明视图：陈述句（"证明"）与开放问句（"推理链"）统一入口。
    /// 返回 None 表示无法形式化为归结目标或归结未启用。
    ///
    /// 同一会话内重复的完整查询走 [`Self::resolution_proof_view_with_receipt`]
    /// 的证明记忆；需要命中信息时直接调用后者。
    pub fn resolution_proof_view(&self, text: &str) -> Option<ResolutionProofView> {
        self.resolution_proof_view_with_receipt(text)
            .map(|(view, _)| view)
    }

    fn resolution_proof_view_uncached(&self, text: &str) -> Option<ResolutionProofView> {
        if !self.config.enable_resolution {
            return None;
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        if let Some(plan) = self.planner.parse(trimmed) {
            if let Some(view) = self.relation_plan_view(&plan) {
                return Some(view);
            }
        }
        if let Some(rewritten) = query_rewrite::rewrite(trimmed, &self.relation_lexicon()) {
            if let Some(view) = self.relation_plan_view(&rewritten.plan) {
                return Some(view);
            }
        }
        // 开放问句优先：剥离问句后缀（"…是谁/是什么/…"）后按谓词链处理。
        // 否则 "张三的师傅的师傅是谁" 会被误判为陈述，把「谁」当常量求证。
        let mut core = trimmed.to_string();
        let mut stripped = false;
        for suffix in [
            "是谁",
            "是什么",
            "是多少",
            "是哪里",
            "是哪个",
            "是哪儿",
            "是哪位",
            "是哪些",
            "是几",
        ] {
            if let Some(rest) = core.strip_suffix(suffix) {
                core = rest.to_string();
                stripped = true;
                break;
            }
        }
        if stripped && !core.contains('是') {
            if let Some(view) = self.build_question_view(&core) {
                return Some(view);
            }
        }
        if let Some((phrase, goals)) = self.statement_goals(trimmed) {
            return Some(self.build_resolution_view(phrase, goals));
        }
        if !core.contains('是') {
            if let Some(view) = self.build_question_view(&core) {
                return Some(view);
            }
        }
        None
    }

    fn build_question_view(&self, core: &str) -> Option<ResolutionProofView> {
        let segments: Vec<String> = core
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty() {
            return None;
        }
        let surfaces: Vec<String> = predicates.iter().map(|p| p.surface.clone()).collect();
        let goals = self.build_question_goals(&entity, &predicates, "答案");
        Some(self.build_resolution_view(format!("{}的{}是谁", entity, surfaces.join("的")), goals))
    }

    /// 开放目标（"X的P…是谁"）：返回全部归结解与各解的证明链
    pub fn resolution_query_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', ' ']);
        if trimmed.is_empty() {
            return "请提供推理目标，例如：推理链：张三的师祖是谁".to_string();
        }
        let mut core = trimmed.to_string();
        for suffix in [
            "是谁",
            "是什么",
            "是多少",
            "是哪里",
            "是哪个",
            "是哪儿",
            "是哪位",
            "是哪些",
            "是几",
        ] {
            if let Some(rest) = core.strip_suffix(suffix) {
                core = rest.to_string();
                break;
            }
        }
        // 句中仍带"是"（如"推理链：张三的师祖是王五"）→ 按证明处理
        if core.contains('是') {
            return self.resolution_proof_report(&core);
        }
        let segments: Vec<String> = core
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        let Some((entity, predicates)) = self.split_chain(&segments) else {
            return format!("无法把「{}」解析为谓词链目标。", trimmed);
        };
        if predicates.is_empty() {
            return format!(
                "「{}」只有实体没有谓词链，请补充关系，例如：推理链：{}的师傅是谁",
                trimmed, entity
            );
        }
        let answer_var = "答案".to_string();
        let goals = self.build_question_goals(&entity, &predicates, &answer_var);
        let outcome = self
            .resolver
            .solve_outcome(&goals, std::slice::from_ref(&answer_var));
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结推理链（一阶 Horn 归结，解与证明均可回溯）：\n");
        output.push_str(&format!(
            "  目标：{}的{}是谁  〔逻辑形式：{}〕\n",
            entity,
            predicates
                .iter()
                .map(|p| p.surface.as_str())
                .collect::<Vec<_>>()
                .join("的"),
            goal_display
        ));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则；深度上限 {}\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            self.resolver.max_depth()
        ));
        if outcome.solutions.is_empty() {
            if outcome.status == SearchStatus::Incomplete {
                output.push_str("  没有找到解：搜索被截断，不能据此断言不存在。\n");
            } else {
                output.push_str(
                    "  没有找到解：搜索已穷尽所有分支（开世界假设：推不出 ≠ 不存在）。\n",
                );
            }
            output.push_str(&self.render_search_diagnostics(&outcome));
            return output.trim_end().to_string();
        }
        for (index, solution) in outcome.solutions.iter().enumerate() {
            let answer = solution
                .bindings
                .get(&answer_var)
                .cloned()
                .unwrap_or_else(|| "?".to_string());
            output.push_str(&format!(
                "  解 {}（证明置信度 {:.2}）：答案是「{}」\n",
                index + 1,
                solution.confidence,
                answer
            ));
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
            output.push('\n');
        }
        output.push_str(&self.render_search_diagnostics(&outcome));
        output.trim_end().to_string()
    }

    /// 关联规则报告（可带过滤词，比如实体/类型/谓词）
    pub fn association_report(&self, filter: Option<&str>) -> String {
        if !self.config.enable_association_rules {
            return "关联规则挖掘未启用（config.toml [reasoning] enable_association_rules = false）。"
                .to_string();
        }
        let mut output = String::from("关联规则报告（支持度/置信度/提升度/杠杆率，纯计数）：\n");
        output.push_str(&format!(
            "  实体篮子 {} 个，物品 {} 种，规则 {} 条（阈值：支持度≥{:.2}，置信度≥{:.2}，提升度≥{:.2}）\n",
            self.associations.basket_count(),
            self.associations.item_kinds(),
            self.associations.rules().len(),
            self.config.association_min_support,
            self.config.association_min_confidence,
            self.config.association_min_lift
        ));
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        let normalized = filter.map(|filter| self.aligner.canonical_form(filter));
        let rules: Vec<&AssociationRule> = self
            .associations
            .rules()
            .iter()
            .filter(|rule| match filter {
                None => true,
                Some(filter) => {
                    let normalized = normalized.as_deref().unwrap_or(filter);
                    rule.consequent.contains(filter)
                        || rule.consequent.contains(normalized)
                        || rule
                            .antecedent
                            .iter()
                            .any(|item| item.contains(filter) || item.contains(normalized))
                }
            })
            .collect();
        if rules.is_empty() {
            output.push_str("  （没有满足条件/阈值的规则：可多导入同类型实体，或降低阈值）\n");
            return output;
        }
        for (index, rule) in rules.iter().take(20).enumerate() {
            output.push_str(&format!(
                "  {}. {} —— {}\n",
                index + 1,
                rule.rule_text(),
                rule.metrics_text()
            ));
        }
        if rules.len() > 20 {
            output.push_str(&format!("  ……其余 {} 条省略\n", rules.len() - 20));
        }
        output.trim_end().to_string()
    }

    /// 实体补全联想："已有特征 ⇒ 可能还具备的缺失特征"
    pub fn association_suggestions(&self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_association_rules {
            return None;
        }
        let entity = self.resolve_known_entity(entity_raw)?;
        let suggestions = self.associations.suggestions_for(&entity, 3);
        if suggestions.is_empty() {
            return None;
        }
        let basket = self.associations.basket(&entity)?;
        let mut features: Vec<&str> = basket.iter().map(|item| display_item(item)).collect();
        features.sort();
        let mut output = format!(
            "关联规则联想（{} 个实体篮子的计数统计，无神经网络）：\n",
            self.associations.basket_count()
        );
        output.push_str(&format!(
            "  「{}」已有特征：{}\n",
            entity,
            features.join("、")
        ));
        for (index, rule) in suggestions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 可能还具备「{}」——{}\n",
                index + 1,
                rule.consequent_text(),
                self.associations.explain(rule)
            ));
        }
        Some(output.trim_end().to_string())
    }

    /// 补全回合的只读视图（供命令、评测与会话快照复用）。
    pub fn completion_ledger(&self) -> &CompletionLedger {
        &self.completion
    }

    /// 本批范围：焦点实体 + 同类型实体 + 一跳邻居 + 与焦点共享谓词的实体。
    /// 有界且确定；无类型数据时仍能覆盖同一关系族的其他实体。
    fn completion_batch_scope(&self, entity: &str) -> Vec<String> {
        let mut scope: BTreeSet<String> = BTreeSet::new();
        scope.insert(entity.to_string());
        if let Some(etype) = self.generator.entity_type_of(entity) {
            for other in self.generator.entities_of_type(etype).into_iter().take(32) {
                scope.insert(other);
            }
        }
        let mut predicates: BTreeSet<String> = self
            .generator
            .relation_predicates(entity)
            .into_iter()
            .collect();
        predicates.extend(self.generator.relation_predicates_in(entity));
        for predicate in &predicates {
            for value in self
                .generator
                .relation_values(entity, predicate)
                .into_iter()
                .take(8)
            {
                scope.insert(value);
            }
            for value in self
                .generator
                .relation_values_in(entity, predicate)
                .into_iter()
                .take(8)
            {
                scope.insert(value);
            }
        }
        for triple in self.predictor.triples() {
            if !predicates.contains(&triple.predicate) {
                continue;
            }
            if scope.len() >= 64 {
                break;
            }
            scope.insert(triple.subject.clone());
            scope.insert(triple.object.clone());
        }
        scope.into_iter().take(64).collect()
    }

    /// 打开一个补全回合：收集链接预测候选，建立本批范围。
    pub fn completion_open(&mut self, entity_raw: &str) -> String {
        let entity = self
            .resolve_known_entity(entity_raw)
            .unwrap_or_else(|| self.aligner.canonical_form(entity_raw.trim()));
        if entity.is_empty() {
            return "请提供要补全的实体，例如：补全回合：张三".to_string();
        }
        if !self.predictor.participates(&entity) {
            return format!(
                "「{}」不在知识图谱的关系中：补全回合需要至少一条已导入关系。",
                entity
            );
        }
        let limit = self.config.max_predictions.saturating_mul(3).clamp(6, 24);
        let predictions = self.predictions_for_entity(&entity, None, limit);
        let scope = self.completion_batch_scope(&entity);
        self.completion.open_episode(&entity, None);
        self.completion.set_batch_scope(scope);
        for prediction in &predictions {
            let rule = self.predictor.rules().get(prediction.rule_index);
            let explanation = self.predictor.explain(prediction);
            self.completion
                .add_prediction(prediction, rule, explanation);
        }
        if predictions.is_empty() {
            self.completion
                .note("当前没有满足阈值的链接预测候选：可导入更多同类实体或降低推理阈值。");
        }
        let mut output = self.completion.render_episode();
        if let Some(question) = self.completion_ask_line() {
            output.push('\n');
            output.push_str(question.trim_end());
        }
        output
    }

    /// 确认一条补全：归纳本批规则、做有界闭包，并把确认事实加入本轮会话事实。
    /// 会话事实不会写入知识图谱；规则默认也只在本批生效。
    pub fn completion_confirm(&mut self, selector: &str) -> String {
        let facts = self.predictor.triples().to_vec();
        let outcome = match self.completion.confirm(selector, &facts) {
            Ok(outcome) => outcome,
            Err(error) => return error.to_string(),
        };
        let atom = Atom::rel(
            &outcome.candidate.predicate,
            &outcome.candidate.subject,
            &outcome.candidate.object,
        );
        let adopted = self.adopt_atom(atom);
        self.completion.note(if adopted {
            "确认结果已加入本轮会话事实（未写入知识图谱）。"
        } else {
            "确认结果已在会话事实中。"
        });
        let mut output = format!(
            "已确认补全：{}({}, {})。\n",
            outcome.candidate.predicate, outcome.candidate.subject, outcome.candidate.object
        );
        if outcome.learned {
            output.push_str(&format!(
                "  已归纳候选规则（作用域：{}），本批新补全 {} 条{}。\n",
                outcome.rule_scope.map(RuleScope::label).unwrap_or("本批"),
                outcome.new_candidates,
                if outcome.closure_truncated {
                    "（达到上限，未补全部分不等于不存在）"
                } else {
                    ""
                }
            ));
        } else {
            output.push_str("  该候选没有可归纳的规则，只记录这一条会话事实。\n");
        }
        output.push('\n');
        output.push_str(&self.completion.render_episode());
        if let Some(question) = self.completion_ask_line() {
            output.push('\n');
            output.push_str(question.trim_end());
        }
        output
    }

    /// 驳回一条补全：写入负例；同一规则反复被驳回会自动隔离。
    pub fn completion_refute(&mut self, selector: &str) -> String {
        let outcome = match self.completion.refute(selector) {
            Ok(outcome) => outcome,
            Err(error) => return error.to_string(),
        };
        let mut output = format!(
            "已驳回补全：{}({}, {})。\n",
            outcome.candidate.predicate, outcome.candidate.subject, outcome.candidate.object
        );
        if let Some(status) = outcome.rule_status {
            output.push_str(&format!("  相关规则状态：{}。\n", status.label()));
        }
        if outcome.quarantined {
            output.push_str(&format!(
                "  规则已被隔离，并撤回同一规则的 {} 条待定候选。\n",
                outcome.removed_pending
            ));
        }
        output.push('\n');
        output.push_str(&self.completion.render_episode());
        if let Some(question) = self.completion_ask_line() {
            output.push('\n');
            output.push_str(question.trim_end());
        }
        output
    }

    /// 显式晋升补全规则的作用域；只有全局规则进入全局归结。
    ///
    /// 命名空间规则留在补全回合内按实体范围使用；要让它参与全局问答，
    /// 必须再晋升到「全局」。这样命名空间不会被静默地当成全局。
    pub fn completion_promote(&mut self, rule_id: &str, scope: &str) -> String {
        let target = match scope.trim().to_lowercase().as_str() {
            "example" | "本例" => RuleScope::Example,
            "batch" | "本批" => RuleScope::Batch,
            "namespace" | "命名空间" => RuleScope::Namespace,
            "global" | "全局" => RuleScope::Global,
            _ => {
                return "作用域只支持：本例 / 本批 / 命名空间 / 全局".to_string();
            }
        };
        let outcome = match self.completion.promote(rule_id, target) {
            Ok(outcome) => outcome,
            Err(error) => return error.to_string(),
        };
        if target == RuleScope::Global {
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
            if self.config.enable_soft_logic {
                self.rebuild_soft_logic();
            }
        }
        let mut output = format!(
            "补全规则 {} 作用域已从「{}」晋升为「{}」。\n",
            rule_id,
            outcome.from.label(),
            outcome.to.label()
        );
        if let Some(warning) = outcome.warning {
            output.push_str(&format!("  注意：{}。\n", warning));
        }
        match target {
            RuleScope::Global => output.push_str(
                "  该规则已进入全局归结与软逻辑；回答仍会标注归纳来源。\n",
            ),
            RuleScope::Namespace => output.push_str(
                "  该规则只在补全回合内按命名空间范围参与候选生成，尚未进入全局归结；要参与全局问答请再晋升到「全局」。\n",
            ),
            _ => {}
        }
        output
    }

    /// 显式退役补全规则：停止派生新结论；已确认的会话事实保留。
    pub fn completion_retire(&mut self, rule_id: &str) -> String {
        let outcome = match self.completion.retire(rule_id) {
            Ok(outcome) => outcome,
            Err(error) => return error.to_string(),
        };
        if outcome.scope == RuleScope::Global {
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
            if self.config.enable_soft_logic {
                self.rebuild_soft_logic();
            }
        }
        let mut output = format!(
            "补全规则 {} 已退役（原状态：{}，作用域：{}）。\n",
            outcome.rule_id,
            outcome.previous.label(),
            outcome.scope.label()
        );
        output.push_str("  已确认的会话事实保留；该规则不再派生新结论。\n");
        if outcome.scope == RuleScope::Global {
            output.push_str("  已从全局归结与软逻辑中移除。\n");
        }
        output
    }

    /// 恢复已退役补全规则：按确认次数回到候选或生效状态。
    pub fn completion_reactivate(&mut self, rule_id: &str) -> String {
        let outcome = match self.completion.reactivate(rule_id) {
            Ok(outcome) => outcome,
            Err(error) => return error.to_string(),
        };
        if outcome.scope == RuleScope::Global {
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
            if self.config.enable_soft_logic {
                self.rebuild_soft_logic();
            }
        }
        let mut output = format!(
            "补全规则 {} 已恢复为「{}」（作用域：{}）。\n",
            outcome.rule_id,
            outcome.current.label(),
            outcome.scope.label()
        );
        if outcome.scope == RuleScope::Global {
            output.push_str("  该规则重新进入全局归结与软逻辑。\n");
        }
        output
    }

    pub fn completion_report(&self) -> String {
        self.completion.render_episode()
    }

    pub fn completion_rules_report(&self) -> String {
        self.completion.render_rules()
    }

    pub fn completion_regression_report(&self) -> String {
        self.completion.render_regression_report()
    }

    /// 选一个候选向用户追问；每轮最多一个，重复候选会被降权。
    pub fn completion_next_question(&mut self) -> Option<CompletionQuestion> {
        self.completion.next_question()
    }

    pub fn completion_question_report(&mut self) -> String {
        if self.completion.questions_muted() {
            return "主动追问已静音。用「允许提问」恢复。".to_string();
        }
        match self.completion.next_question() {
            Some(question) => format!(
                "补全提问（{}，效用 {:.1}）：{}",
                question.kind.label(),
                question.utility,
                question.text
            ),
            None => "当前没有值得追问的候选（证据不足、已裁定或已静音）。".to_string(),
        }
    }

    fn completion_ask_line(&mut self) -> Option<String> {
        self.completion
            .next_question()
            .map(|question| format!("  下一问（{}）：{}\n", question.kind.label(), question.text))
    }

    /// 草稿命中前的双重复核：纪元已由 `RevisionLru` 把关，这里再确认
    /// 规则下标仍然有效、头部谓词一致、置信度未变、证据三元组仍在图中。
    /// 任何一项不符都按未命中处理，绝不让旧规则文本借缓存回流。
    fn draft_is_valid(&self, predictions: &[Prediction]) -> bool {
        predictions.iter().all(|prediction| {
            let Some(rule) = self.predictor.rules().get(prediction.rule_index) else {
                return false;
            };
            rule.head == prediction.triple.predicate
                && (prediction.confidence - rule.confidence).abs() < f64::EPSILON
                && !self.predictor.has_triple(
                    &prediction.triple.subject,
                    &prediction.triple.predicate,
                    &prediction.triple.object,
                )
                && prediction.evidence.iter().all(|triple| {
                    self.predictor
                        .has_triple(&triple.subject, &triple.predicate, &triple.object)
                })
        })
    }

    /// 预测候选的会话缓存入口：纪元相同且证据复核通过才复用，否则重算。
    fn predictions_for_entity(
        &self,
        entity: &str,
        predicate: Option<&str>,
        limit: usize,
    ) -> Vec<Prediction> {
        let cached = {
            let mut drafts = self.prediction_drafts.lock();
            drafts.get(self.revision, entity, predicate, limit)
        };
        if let Some(predictions) = cached {
            if self.draft_is_valid(&predictions) {
                self.perf.bump("prediction_draft_hits");
                return predictions;
            }
            self.perf.bump("prediction_draft_stale");
            self.prediction_drafts.lock().clear();
        }
        self.perf.bump("prediction_draft_misses");
        let predictions = self.predictor.predict_for_entity(entity, predicate, limit);
        let evicted = self.prediction_drafts.lock().put(
            self.revision,
            entity,
            predicate,
            limit,
            predictions.clone(),
        );
        self.perf
            .bump_by("prediction_draft_evictions", evicted as u64);
        predictions
    }

    /// 轮末预取：把当前焦点实体（以及上一轮谓词）的候选提前算好。
    ///
    /// 预取只是把下一轮本来就要做的纯计算提前；不写入图谱、不确认候选、
    /// 不改变任何回答文本。返回预先物化的候选条数。
    pub fn prefetch_answer_drafts(&mut self) -> usize {
        if !self.config.enable_link_prediction || self.predictor.rules().is_empty() {
            return 0;
        }
        let Some(entity) = self.context.focus().map(str::to_string) else {
            return 0;
        };
        let limit = self.config.max_predictions.saturating_mul(4).max(4);
        let mut warmed = self.predictions_for_entity(&entity, None, limit).len();
        if let Some(predicate) = self.context.last_predicate().map(str::to_string) {
            warmed += self
                .predictions_for_entity(&entity, Some(&predicate), limit)
                .len();
        }
        self.perf.bump("prediction_draft_prefetches");
        self.perf
            .bump_by("prediction_draft_prefetched", warmed as u64);
        warmed
    }

    /// 草稿台账：只读观测，展示纪元、驻留条目与命中/预取计数。
    pub fn prediction_draft_report(&self) -> String {
        let drafts = self.prediction_drafts.lock();
        let hits = self.perf.count("prediction_draft_hits");
        let misses = self.perf.count("prediction_draft_misses");
        let stale = self.perf.count("prediction_draft_stale");
        let prefetches = self.perf.count("prediction_draft_prefetches");
        let prefetched = self.perf.count("prediction_draft_prefetched");
        let evictions = self.perf.count("prediction_draft_evictions");
        format!(
            "预测答案草稿（链接预测的轮末预取）：知识版本 {}，驻留 {} 条。\n  命中 {} 次，未命中 {} 次，证据复核拒绝 {} 次；预取 {} 轮（{} 条候选），淘汰 {} 条。\n  草稿不是知识：不写图谱、不确认候选；知识版本推进时整体失效。",
            self.revision,
            drafts.len(),
            hits,
            misses,
            stale,
            prefetches,
            prefetched,
            evictions
        )
    }

    pub fn predictive_ledger(&self) -> &PredictiveElaboration {
        &self.predictive
    }

    fn predictive_guard(&mut self) -> PredictionGuard {
        let mut guard = PredictionGuard::new();
        for triple in self.predictor.triples() {
            guard.insert_known(&triple.subject, &triple.predicate, &triple.object);
        }
        for triple in self.completion.refuted_triples() {
            guard.insert_refuted(&triple.subject, &triple.predicate, &triple.object);
        }
        for atom in &self.session_facts {
            if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
                guard.insert_session(subject, predicate, object);
            }
        }
        for atom in &self.session_negative_facts {
            if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
                guard.insert_refuted(subject, predicate, object);
            }
        }
        for predicate in self.slots.policy().functional_predicates() {
            guard.mark_functional(predicate);
        }
        for (entity, etype) in self.generator.typed_entities() {
            guard.insert_type(&entity, &etype);
        }
        let closure = self.discourse.narrator_mut().closure().clone();
        guard.set_closure(closure);
        guard
    }

    pub fn predictive_elaboration(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_link_prediction {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        if !self.predictor.participates(&entity) && !self.generator.is_known_entity(&entity) {
            return Some(format!(
                "「{}」不在知识图谱的关系中：推测需要至少一条已导入关系。",
                entity
            ));
        }
        let limit = self.config.max_predictions.saturating_mul(4).clamp(6, 24);
        let predictions = self.predictions_for_entity(&entity, None, limit);
        let guard = self.predictive_guard();
        let plan = self
            .predictive
            .propose(&entity, &predictions, &self.predictor, &guard);
        Some(plan.render())
    }

    /// 事实介绍 + 推测段。推测是未确认假设，不进入 [`AnswerPlan`]；
    /// 后续风格重述只覆盖已审计事实，避免把假设当成计划内命题。
    pub fn generate_about_with_prediction(&mut self, entity_raw: &str) -> Option<String> {
        let facts = self.generate_about(entity_raw);
        let speculation = self.predictive_elaboration(entity_raw);
        match (facts, speculation) {
            (Some(facts), Some(speculation)) => Some(format!(
                "{}\n\n{}",
                facts.trim_end(),
                speculation.trim_end()
            )),
            (Some(facts), None) => Some(facts),
            (None, Some(speculation)) => Some(speculation),
            (None, None) => None,
        }
    }

    pub fn predictive_confirm(&mut self, selector: &str) -> String {
        let hypothesis = match self.predictive.resolve(selector, true) {
            Ok(hypothesis) => hypothesis,
            Err(error) => return format!("无法确认预测：{}。", error),
        };
        let atom = Atom::rel(
            &hypothesis.key.predicate,
            &hypothesis.key.subject,
            &hypothesis.key.object,
        );
        let label = format!("确认预测：{}", hypothesis.key.display());
        match self.commit_session_assertion(label, &atom) {
            Ok(_) => {
                self.note_dialogue_verdict(
                    &hypothesis.key.subject,
                    &hypothesis.key.predicate,
                    &hypothesis.key.object,
                    true,
                    &hypothesis.id,
                );
                format!(
                    "已确认预测：{}。该事实只进入本次会话事实层（未写入知识图谱），后续介绍会把它当作会话事实。",
                    hypothesis.sentence()
                )
            }
            Err(error) => {
                self.predictive.reopen(&hypothesis.id);
                format!("确认未写入：{}", error)
            }
        }
    }

    pub fn predictive_refute(&mut self, selector: &str) -> String {
        let hypothesis = match self.predictive.resolve(selector, false) {
            Ok(hypothesis) => hypothesis,
            Err(error) => return format!("无法否定预测：{}。", error),
        };
        let atom = Atom::rel(
            &hypothesis.key.predicate,
            &hypothesis.key.subject,
            &hypothesis.key.object,
        );
        let message = self.assert_negative_atom(atom);
        self.note_dialogue_verdict(
            &hypothesis.key.subject,
            &hypothesis.key.predicate,
            &hypothesis.key.object,
            false,
            &hypothesis.id,
        );
        format!(
            "已否定预测：{}。\n  {}\n本会话内不再提名该候选。",
            hypothesis.sentence(),
            message.trim()
        )
    }

    pub fn predictive_report(&self) -> String {
        self.predictive.render_report()
    }

    pub fn handle_predictive_command(&mut self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        for prefix in ["预测展开：", "预测展开:"] {
            if let Some(entity) = trimmed.strip_prefix(prefix) {
                let entity = entity.trim();
                return Some(if entity.is_empty() {
                    "请提供要展开的实体，例如：预测展开：李四".to_string()
                } else {
                    self.generate_about_with_prediction(entity)
                        .unwrap_or_else(|| format!("无法为「{}」生成推测。", entity))
                });
            }
        }
        for prefix in ["推测：", "推测:", "预测推测：", "预测推测:"] {
            if let Some(entity) = trimmed.strip_prefix(prefix) {
                let entity = entity.trim();
                return Some(if entity.is_empty() {
                    "请提供要推测的实体，例如：推测：李四".to_string()
                } else {
                    self.predictive_elaboration(entity)
                        .unwrap_or_else(|| format!("无法为「{}」生成推测。", entity))
                });
            }
        }
        for prefix in [
            "确认预测：",
            "确认预测:",
            "确认推测：",
            "确认推测:",
            "采纳预测：",
            "采纳预测:",
        ] {
            if let Some(selector) = trimmed.strip_prefix(prefix) {
                let selector = selector.trim();
                return Some(if selector.is_empty() {
                    "请给出推测序号或编号，例如：确认预测：1".to_string()
                } else {
                    self.predictive_confirm(selector)
                });
            }
        }
        for prefix in [
            "否定预测：",
            "否定预测:",
            "否定推测：",
            "否定推测:",
            "驳回预测：",
            "驳回预测:",
        ] {
            if let Some(selector) = trimmed.strip_prefix(prefix) {
                let selector = selector.trim();
                return Some(if selector.is_empty() {
                    "请给出推测序号或编号，例如：否定预测：1".to_string()
                } else {
                    self.predictive_refute(selector)
                });
            }
        }
        if matches!(trimmed, "预测台账" | "预测报告" | "推测台账" | "推测报告") {
            return Some(self.predictive_report());
        }
        if matches!(trimmed, "预测草稿" | "预测缓存" | "草稿台账") {
            return Some(self.prediction_draft_report());
        }
        None
    }

    /// 把恢复阶梯（或宿主脚本）的一次用户裁决写入补全台账。
    ///
    /// 若该三元组可由链接预测解释，就把预测规则一并登记；确认会归纳出
    /// 本批规则，否决会进入全局拒绝集。只写会话台账，不写知识图谱。
    pub fn note_dialogue_verdict(
        &mut self,
        entity: &str,
        attribute: &str,
        value: &str,
        confirmed: bool,
        source_id: &str,
    ) -> bool {
        let triple = crate::link_prediction::Triple::new(entity, attribute, value);
        if triple.subject.trim().is_empty()
            || triple.predicate.trim().is_empty()
            || triple.object.trim().is_empty()
        {
            return false;
        }
        let limit = self.config.max_predictions.saturating_mul(2).max(4);
        let prediction = self
            .predictions_for_entity(&triple.subject, Some(&triple.predicate), limit)
            .into_iter()
            .find(|prediction| prediction.triple == triple);
        let (rule, evidence) = prediction
            .map(|prediction| {
                let rule = self
                    .predictor
                    .rules()
                    .get(prediction.rule_index)
                    .and_then(crate::completion_episode::RuleSpec::from_horn);
                (rule, prediction.evidence)
            })
            .unwrap_or((None, Vec::new()));
        self.completion.record_external_verdict(
            triple,
            confirmed,
            rule,
            evidence,
            "恢复阶梯",
            source_id,
        )
    }

    /// 导出四层知识容器清单：基础层 / 会话层 / 推测层 / 假设层。
    ///
    /// 每层只读当前状态，不做推理；摘要覆盖该层全量条目，样例有界。
    pub fn knowledge_layer_manifest(&self) -> KnowledgeLayerManifest {
        let mut base: Vec<String> = self
            .predictor
            .triples()
            .iter()
            .map(|triple| {
                format!(
                    "{} --{}--> {}",
                    triple.subject, triple.predicate, triple.object
                )
            })
            .collect();
        base.extend(
            self.attributes.iter().map(|(entity, attribute, value)| {
                format!("{} --{}--> {}", entity, attribute, value)
            }),
        );

        let mut session = Vec::new();
        for atom in &self.session_facts {
            session.push(format!("+ {}", atom.display()));
        }
        for atom in &self.session_negative_facts {
            session.push(format!("- {}", atom.display()));
        }

        let mut speculative = Vec::new();
        for episode in self.completion.episodes() {
            for candidate in &episode.candidates {
                if candidate.status == CandidateStatus::Pending {
                    speculative.push(format!(
                        "{} [{}，置信度 {:.2}，回合 {}]",
                        candidate.display_triple(),
                        candidate.warrant.label(),
                        candidate.confidence,
                        &episode.id[..episode.id.len().min(21)]
                    ));
                }
            }
        }
        for rule in self.completion.rules() {
            if rule.is_usable() {
                speculative.push(format!(
                    "规则[{}] {}",
                    rule.status.label(),
                    rule.spec.rule_text()
                ));
            }
        }
        for hypothesis in self.predictive.pending() {
            speculative.push(format!(
                "预测 {} [{}，置信度 {:.2}]",
                hypothesis.key.display(),
                hypothesis.status.label(),
                hypothesis.confidence()
            ));
        }

        let hypotheses: Vec<String> = self
            .assumption_world
            .as_ref()
            .map(|world| {
                world
                    .assumptions
                    .iter()
                    .map(|atom| atom.display())
                    .collect()
            })
            .unwrap_or_default();

        KnowledgeLayerManifest::new(
            self.revision,
            vec![
                LayerEntry::new("L0", "基础层", "只读", base),
                LayerEntry::new("L1", "会话层", "会话可写", session),
                LayerEntry::new("L2", "推测层", "待确认", speculative),
                LayerEntry::new("L3", "假设层", "临时", hypotheses),
            ],
        )
    }

    /// 人读的层次清单（命令行与宿主自检共用）。
    pub fn knowledge_layers_report(&self) -> String {
        self.knowledge_layer_manifest().render_text()
    }

    /// 解析建设性推断命令（查询推断 / 确认 / 否定）。
    ///
    /// 构建性推断默认只是查询内假设；这里给出显式裁决入口，确认后的事实
    /// 进入会话事实层并参与后续推理，但始终不写知识图谱。
    pub fn handle_constructive_command(&mut self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        for prefix in ["建设性推断：", "建设性推断:", "推断依据：", "推断依据:"]
        {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let spec = spec.trim();
                return Some(if spec.is_empty() {
                    "请给出起点与目标谓词，例如：建设性推断：孙七|朋友".to_string()
                } else {
                    self.constructive_probe_report(spec)
                });
            }
        }
        for prefix in ["确认推断：", "确认推断:", "采纳推断：", "采纳推断:"] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let spec = spec.trim();
                return Some(if spec.is_empty() {
                    "请给出三元组，例如：确认推断：孙七|朋友|吴九".to_string()
                } else {
                    self.confirm_constructive_triple(spec)
                });
            }
        }
        for prefix in ["否定推断：", "否定推断:", "驳回推断：", "驳回推断:"] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let spec = spec.trim();
                return Some(if spec.is_empty() {
                    "请给出三元组，例如：否定推断：孙七|同事|周八".to_string()
                } else {
                    self.refute_constructive_triple(spec)
                });
            }
        }
        None
    }

    fn parse_constructive_triple(&self, spec: &str) -> Result<(String, String, String), String> {
        let parts: Vec<&str> = spec.split('|').map(str::trim).collect();
        if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
            return Err("需要「主体|谓词|客体」三段，且都不能为空".to_string());
        }
        let subject = self.aligner.canonical_form(parts[0]);
        let object = self.aligner.canonical_form(parts[2]);
        let (predicate, inverted) = self.resolve_question_predicate(parts[1]);
        if !self.is_known_predicate(&predicate) {
            return Err(format!("谓词「{}」不在当前词表或规则中", parts[1]));
        }
        Ok(if inverted {
            (object, predicate, subject)
        } else {
            (subject, predicate, object)
        })
    }

    fn confirm_constructive_triple(&mut self, spec: &str) -> String {
        let (subject, predicate, object) = match self.parse_constructive_triple(spec) {
            Ok(parts) => parts,
            Err(error) => return format!("无法确认建设性推断：{}。", error),
        };
        let atom = Atom::rel(&predicate, &subject, &object);
        let label = format!("确认建设性推断：{}|{}|{}", subject, predicate, object);
        match self.commit_session_assertion(label, &atom) {
            Ok(true) => format!(
                "✓ 已确认建设性推断：{} 的 {} 是 {}。该事实进入本轮会话事实，未写入知识图谱。",
                subject, predicate, object
            ),
            Ok(false) => format!(
                "该建设性推断已在会话事实中：{} 的 {} 是 {}（幂等确认）。",
                subject, predicate, object
            ),
            Err(error) => format!("确认未写入：{}", error),
        }
    }

    fn refute_constructive_triple(&mut self, spec: &str) -> String {
        let (subject, predicate, object) = match self.parse_constructive_triple(spec) {
            Ok(parts) => parts,
            Err(error) => return format!("无法否定建设性推断：{}。", error),
        };
        let atom = Atom::rel(&predicate, &subject, &object);
        let rendered = self.render_atom_nl(&atom);
        let message = self.assert_negative_atom(atom);
        format!(
            "已否定建设性推断：{}。
{}",
            rendered, message
        )
    }

    fn constructive_probe_report(&self, spec: &str) -> String {
        let parts: Vec<&str> = spec.split('|').map(str::trim).collect();
        if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
            return "需要「起点|目标谓词」，例如：建设性推断：孙七|朋友".to_string();
        }
        if parts.len() > 3 {
            return "最多三段：起点|目标谓词|正向或逆向".to_string();
        }
        let seed = self.aligner.canonical_form(parts[0]);
        let (predicate, mut inverted) = self.resolve_question_predicate(parts[1]);
        if parts.len() == 3 {
            match parts[2] {
                "逆向" | "反向" | "inverse" => inverted = !inverted,
                "正向" | "forward" | "" => {}
                other => return format!("方向「{}」无法识别；可用「正向」或「逆向」。", other),
            }
        }
        if !self.is_known_predicate(&predicate) {
            return format!("谓词「{}」不在当前词表或规则中。", parts[1]);
        }
        let Some(mut engine) = self.new_construction_reasoner() else {
            return "建设性推断未启用。请在 config.toml 的 [reasoning.constructive] 中设置 enabled = true。".to_string();
        };
        let report = engine.derive(&seed, &predicate, inverted);
        let audit = engine.support_audit();
        let mut output = report.render_text();
        output.push_str(&format!(
            "\n证明审计：检查 {} 条虚拟事实，无效 {}，循环 {}。",
            audit.inspected,
            audit.invalid.len(),
            audit.cyclic.len()
        ));
        if audit.is_clean() && audit.inspected > 0 {
            output.push_str(" 所有推断都能回溯到观察事实。");
        }
        if !report.is_empty() {
            let key = &report.target_facts[0].key;
            output.push_str(&format!(
                "\n如需采纳，回复「确认推断：{}|{}|{}」；否认回复「否定推断：{}|{}|{}」。",
                key.subject, key.predicate, key.object, key.subject, key.predicate, key.object
            ));
        }
        engine.end_query();
        output
    }

    /// 解析并执行补全回合命令（打开 / 确认 / 驳回 / 报告 / 规则生命周期）。
    ///
    /// 交互命令行与 HTTP 服务共用这一入口，避免两套解析漂移；返回 `None`
    /// 表示不是补全命令，调用方应继续常规管线。
    pub fn handle_completion_command(&mut self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        for prefix in ["补全回合：", "补全回合:", "学习回合：", "学习回合:"] {
            if let Some(entity) = trimmed.strip_prefix(prefix) {
                let entity = entity.trim();
                return Some(if entity.is_empty() {
                    "请提供要补全的实体，例如：补全回合：张三".to_string()
                } else {
                    self.completion_open(entity)
                });
            }
        }
        for prefix in ["确认补全：", "确认补全:", "采纳补全：", "采纳补全:"] {
            if let Some(selector) = trimmed.strip_prefix(prefix) {
                let selector = selector.trim();
                return Some(if selector.is_empty() {
                    "请给出候选序号、候选 id 或 主体|谓词|客体，例如：确认补全：1".to_string()
                } else {
                    self.completion_confirm(selector)
                });
            }
        }
        for prefix in ["否定补全：", "否定补全:", "驳回补全：", "驳回补全:"] {
            if let Some(selector) = trimmed.strip_prefix(prefix) {
                let selector = selector.trim();
                return Some(if selector.is_empty() {
                    "请给出候选序号、候选 id 或 主体|谓词|客体，例如：否定补全：2".to_string()
                } else {
                    self.completion_refute(selector)
                });
            }
        }
        if matches!(trimmed, "补全规则" | "补全规则报告" | "补全台账") {
            return Some(self.completion_rules_report());
        }
        if matches!(trimmed, "补全报告" | "补全回合报告" | "补全状态") {
            return Some(self.completion_report());
        }
        if matches!(
            trimmed,
            "补全测试" | "补全回归" | "知识回归" | "补全测试报告"
        ) {
            return Some(self.completion_regression_report());
        }
        if matches!(trimmed, "补全重放" | "重放补全测试" | "补全测试重放") {
            let report = self.completion.enforce_regression();
            let mut output = format!("{}\n", report.summary());
            output.push_str(&self.completion.render_regression_report());
            return Some(output);
        }
        if matches!(
            trimmed,
            "补全提问" | "下一个问题" | "该问什么" | "主动提问" | "问我"
        ) {
            return Some(self.completion_question_report());
        }
        if matches!(trimmed, "别问了" | "不要追问" | "静音补全提问" | "停止提问") {
            let changed = self.completion.mute_questions();
            return Some(if changed {
                "好，主动追问已静音；已有候选仍可用「补全报告」查看。说「允许提问」可恢复。"
                    .to_string()
            } else {
                "主动追问已经是静音状态。".to_string()
            });
        }
        if matches!(trimmed, "允许提问" | "恢复提问" | "解除静音") {
            let changed = self.completion.unmute_questions();
            return Some(if changed {
                "已恢复主动追问。".to_string()
            } else {
                "主动追问本来就是开启的。".to_string()
            });
        }
        for prefix in [
            "晋升补全规则：",
            "晋升补全规则:",
            "补全规则晋升：",
            "补全规则晋升:",
        ] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let parsed = spec
                    .trim()
                    .split_once('|')
                    .or_else(|| spec.trim().split_once('｜'))
                    .or_else(|| spec.trim().rsplit_once(' '));
                return Some(match parsed {
                    Some((rule_id, scope))
                        if !rule_id.trim().is_empty() && !scope.trim().is_empty() =>
                    {
                        self.completion_promote(rule_id.trim(), scope.trim())
                    }
                    _ => "用法：晋升补全规则：<规则id>|<作用域>，作用域为本例/本批/命名空间/全局"
                        .to_string(),
                });
            }
        }
        for prefix in [
            "退役补全规则：",
            "退役补全规则:",
            "补全规则退役：",
            "补全规则退役:",
        ] {
            if let Some(rule_id) = trimmed.strip_prefix(prefix) {
                let rule_id = rule_id.trim();
                return Some(if rule_id.is_empty() {
                    "用法：退役补全规则：<规则id>".to_string()
                } else {
                    self.completion_retire(rule_id)
                });
            }
        }
        for prefix in [
            "恢复补全规则：",
            "恢复补全规则:",
            "补全规则恢复：",
            "补全规则恢复:",
        ] {
            if let Some(rule_id) = trimmed.strip_prefix(prefix) {
                let rule_id = rule_id.trim();
                return Some(if rule_id.is_empty() {
                    "用法：恢复补全规则：<规则id>".to_string()
                } else {
                    self.completion_reactivate(rule_id)
                });
            }
        }
        None
    }

    /// 恢复通道 L2：联想提名（链接预测）——「(实体, 属性) 还可能是哪个值」。
    ///
    /// 只提名、不下结论；每条附规则证据。返回顺序由预测器确定，
    /// 调用方（恢复引擎）会再做定点化排序与验证。
    pub fn recovery_nominations(
        &self,
        entity: &str,
        attribute: &str,
    ) -> Vec<crate::recovery::RecoveryNomination> {
        if !self.config.enable_link_prediction {
            return Vec::new();
        }
        let limit = self.config.max_predictions.saturating_mul(2).max(4);
        let mut nominations = Vec::new();
        for prediction in self.predictions_for_entity(entity, Some(attribute), limit) {
            if prediction.triple.subject != entity || prediction.triple.predicate != attribute {
                continue;
            }
            if self.completion.is_adjudicated(&prediction.triple) {
                continue;
            }
            nominations.push(crate::recovery::RecoveryNomination {
                value: prediction.triple.object.clone(),
                score: prediction.confidence,
                evidence: vec![self.predictor.explain(&prediction)],
            });
        }
        nominations
    }

    /// 恢复通道 L2：符号验证——提名值是否被图谱/生成器支持或反驳。
    pub fn recovery_verify(
        &self,
        entity: &str,
        attribute: &str,
        value: &str,
    ) -> crate::recovery::VerificationOutcome {
        if self.generator.has_attribute(entity, attribute, value)
            || self.predictor.has_triple(entity, attribute, value)
        {
            return crate::recovery::VerificationOutcome::Confirmed;
        }
        if !self
            .generator
            .attribute_values(entity, attribute)
            .is_empty()
        {
            return crate::recovery::VerificationOutcome::Refuted;
        }
        crate::recovery::VerificationOutcome::NoVerdict
    }

    /// 恢复通道 L3：实体可追问的属性/关系线索（确定性排序，取前几项）。
    pub fn recovery_clues(&self, entity: &str) -> Vec<String> {
        let mut clues = self.generator.attribute_keys(entity);
        for predicate in self.generator.relation_predicates(entity) {
            if !clues.contains(&predicate) {
                clues.push(predicate);
            }
        }
        clues.truncate(3);
        clues
    }

    /// 恢复通道 L1：换帧重解——换问法重新查询（不生成，只重查）。
    pub fn recovery_reframe(
        &self,
        entity: &str,
        attribute: &str,
    ) -> Vec<crate::recovery::ReframedReading> {
        let questions = [
            format!("{}的{}是多少", entity, attribute),
            format!("{}的{}是什么", entity, attribute),
            format!("{}的{}呢", entity, attribute),
        ];
        let mut readings = Vec::new();
        for question in questions {
            let answer = SymbolicReasoner::answer(self, &question);
            readings.push(crate::recovery::ReframedReading { question, answer });
        }
        readings
    }
}

fn proof_search_cost(stats: &SearchStats) -> ProofSearchCost {
    ProofSearchCost {
        goals: stats.goals,
        goal_attempts: stats.goal_attempts,
        fact_checks: stats.fact_checks,
        rule_attempts: stats.rule_attempts,
        depth_cutoffs: stats.depth_cutoffs,
        budget_cutoffs: stats.budget_cutoffs,
        solution_cutoffs: stats.solution_cutoffs,
        state_cutoffs: stats.state_cutoffs,
    }
}

fn estimate_resolution_view_bytes(view: &ResolutionProofView) -> usize {
    fn node_bytes(node: &ProofViewNode) -> usize {
        let source = match &node.source {
            crate::view_render::ViewProofSource::Fact => 4,
            crate::view_render::ViewProofSource::Rule(source) => source.len(),
        };
        node.label
            .len()
            .saturating_add(node.canonical.len())
            .saturating_add(source)
            .saturating_add(32)
            .saturating_add(node.children.iter().map(node_bytes).sum::<usize>())
    }
    let mut bytes = 128usize
        .saturating_add(view.phrase.len())
        .saturating_add(view.goal_display.len());
    for solution in &view.solutions {
        for (name, value) in &solution.bindings {
            bytes = bytes
                .saturating_add(name.len())
                .saturating_add(value.len())
                .saturating_add(3);
        }
        bytes = bytes.saturating_add(24);
        bytes = bytes.saturating_add(solution.proofs.iter().map(node_bytes).sum::<usize>());
    }
    for frontier in &view.frontier {
        bytes = bytes
            .saturating_add(frontier.goal.len())
            .saturating_add(64);
    }
    bytes
}

fn resolution_view_footprint(view: &ResolutionProofView) -> ProofFootprint {
    fn collect<'a>(node: &'a ProofViewNode, out: &mut Vec<&'a str>) {
        out.push(node.canonical.as_str());
        for child in &node.children {
            collect(child, out);
        }
    }
    let mut texts: Vec<&str> = Vec::new();
    for goal in view.goal_display.split('，') {
        let goal = goal.trim();
        if !goal.is_empty() {
            texts.push(goal);
        }
    }
    for solution in &view.solutions {
        for proof in &solution.proofs {
            collect(proof, &mut texts);
        }
    }
    for frontier in &view.frontier {
        texts.push(frontier.goal.as_str());
    }
    let mut footprint = ProofFootprint::from_canonical_atoms(texts);
    if footprint.is_empty() {
        footprint.watch_any = true;
    }
    footprint
}

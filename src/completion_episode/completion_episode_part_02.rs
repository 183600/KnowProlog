impl CompletionLedger {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_MAX_EPISODE_CANDIDATES,
            DEFAULT_MAX_CLOSURE_CANDIDATES,
            DEFAULT_MAX_EPISODES,
            DEFAULT_MAX_RULES,
        )
    }

    pub fn with_limits(
        max_candidates: usize,
        max_closure: usize,
        max_episodes: usize,
        max_rules: usize,
    ) -> Self {
        Self {
            episodes: Vec::new(),
            active: None,
            rules: Vec::new(),
            refuted_triples: BTreeSet::new(),
            confirmed_triples: BTreeSet::new(),
            regression_tests: Vec::new(),
            asked_questions: BTreeMap::new(),
            question_muted: false,
            tick: 0,
            max_candidates: max_candidates.max(1),
            max_closure: max_closure.max(1),
            max_episodes: max_episodes.max(1),
            max_rules: max_rules.max(1),
            max_batch_entities: DEFAULT_MAX_BATCH_ENTITIES,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
            && self.rules.is_empty()
            && self.refuted_triples.is_empty()
            && self.confirmed_triples.is_empty()
            && self.regression_tests.is_empty()
            && self.asked_questions.is_empty()
            && !self.question_muted
    }

    /// 从持久化快照恢复后重新夹紧上限：快照文件不可信，不能让其中的
    /// 超大上限把有界闭包变成无界计算。
    pub fn clamp_limits(&mut self) {
        self.max_candidates = self
            .max_candidates
            .clamp(1, DEFAULT_MAX_EPISODE_CANDIDATES);
        self.max_closure = self.max_closure.clamp(1, DEFAULT_MAX_CLOSURE_CANDIDATES);
        self.max_episodes = self.max_episodes.clamp(1, DEFAULT_MAX_EPISODES);
        self.max_rules = self.max_rules.clamp(1, DEFAULT_MAX_RULES);
        self.max_batch_entities = self
            .max_batch_entities
            .clamp(1, DEFAULT_MAX_BATCH_ENTITIES);
        self.episodes.truncate(self.max_episodes);
        self.rules.truncate(self.max_rules);
        self.regression_tests.truncate(DEFAULT_MAX_TESTS);
        if self.asked_questions.len() > DEFAULT_MAX_TESTS {
            self.asked_questions.clear();
        }
        self.active = self.active.filter(|index| *index < self.episodes.len());
    }

    pub fn rules(&self) -> &[LearnedRule] {
        &self.rules
    }

    pub fn episodes(&self) -> &[CompletionEpisode] {
        &self.episodes
    }

    pub fn rule(&self, rule_id: &str) -> Option<&LearnedRule> {
        self.rules.iter().find(|rule| rule.id == rule_id)
    }

    pub fn current_episode(&self) -> Option<&CompletionEpisode> {
        self.active.and_then(|index| self.episodes.get(index))
    }

    pub fn current_episode_mut(&mut self) -> Option<&mut CompletionEpisode> {
        self.active
            .and_then(move |index| self.episodes.get_mut(index))
    }

    pub fn refuted_triples(&self) -> &BTreeSet<Triple> {
        &self.refuted_triples
    }

    pub fn confirmed_triples(&self) -> &BTreeSet<Triple> {
        &self.confirmed_triples
    }

    pub fn regression_tests(&self) -> &[CompletionTest] {
        &self.regression_tests
    }

    /// 该三元组是否已经被用户裁定过（确认或否决）。
    pub fn is_adjudicated(&self, triple: &Triple) -> bool {
        self.confirmed_triples.contains(triple) || self.refuted_triples.contains(triple)
    }

    /// 该三元组是否被用户否决过（任何通道）。
    pub fn is_refuted(&self, triple: &Triple) -> bool {
        self.refuted_triples.contains(triple)
    }

    pub fn questions_muted(&self) -> bool {
        self.question_muted
    }

    /// 用户说「别问了」：长期静音主动追问（随快照持久化）。
    pub fn mute_questions(&mut self) -> bool {
        if self.question_muted {
            return false;
        }
        self.question_muted = true;
        true
    }

    pub fn unmute_questions(&mut self) -> bool {
        if !self.question_muted {
            return false;
        }
        self.question_muted = false;
        true
    }

    /// 选一个最有判别力的问题；每轮最多一个，重复提问会被扣分。
    ///
    /// 效用是确定性的：置信度 + 证据强度 + 规则杠杆 - 重复惩罚。
    /// 没有候选、证据太弱或已静音时返回 `None`，绝不为了追问而追问。
    pub fn next_question(&mut self) -> Option<CompletionQuestion> {
        if self.question_muted {
            return None;
        }
        let episode = self.current_episode()?;
        let mut scored: Vec<(f64, &CompletionCandidate)> = episode
            .candidates
            .iter()
            .filter(|candidate| candidate.status == CandidateStatus::Pending)
            .filter(|candidate| !self.is_adjudicated(&candidate.triple))
            .filter_map(|candidate| {
                let asked = self
                    .asked_questions
                    .get(&candidate.id)
                    .copied()
                    .unwrap_or(0);
                if asked >= QUESTION_MAX_REPEATS {
                    return None;
                }
                let mut utility = candidate.confidence.clamp(0.0, 1.0) * 60.0;
                utility += match candidate.warrant {
                    Warrant::Entailed => 12.0,
                    Warrant::Induced => 8.0,
                    Warrant::Analogical => 0.0,
                };
                utility += match candidate.origin {
                    CandidateOrigin::BatchClosure => 4.0,
                    CandidateOrigin::PathPrediction => 2.0,
                    CandidateOrigin::Manual => 1.0,
                    CandidateOrigin::Analogical => 0.0,
                };
                if candidate.rule.is_some() {
                    utility += 5.0;
                }
                if candidate
                    .rule_id
                    .as_deref()
                    .and_then(|id| self.rule(id))
                    .is_some_and(|rule| rule.confirmations > 0)
                {
                    utility += 3.0;
                }
                utility += (candidate.evidence.len().min(3) as f64) * 0.5;
                utility -= asked as f64 * QUESTION_REPEAT_PENALTY;
                (utility >= QUESTION_MIN_UTILITY).then_some((utility, candidate))
            })
            .collect();
        scored.sort_by(|(left_utility, left), (right_utility, right)| {
            right_utility
                .total_cmp(left_utility)
                .then_with(|| left.id.cmp(&right.id))
        });
        let (utility, candidate) = scored.first().map(|(utility, candidate)| (*utility, *candidate))?;
        let alternatives: Vec<String> = episode
            .candidates
            .iter()
            .filter(|other| {
                other.status == CandidateStatus::Pending
                    && other.id != candidate.id
                    && other.triple.subject == candidate.triple.subject
                    && other.triple.predicate == candidate.triple.predicate
                    && other.triple.object != candidate.triple.object
                    && !self.is_adjudicated(&other.triple)
            })
            .map(|other| other.triple.object.clone())
            .collect();
        let kind = if alternatives.is_empty() {
            QuestionKind::Confirm
        } else {
            QuestionKind::Choice
        };
        let text = match kind {
            QuestionKind::Confirm => format!(
                "「{}」的「{}」是「{}」吗？（{}来源，置信度 {:.2}；确认补全：{} / 否定补全：{}）",
                candidate.triple.subject,
                candidate.triple.predicate,
                candidate.triple.object,
                candidate.origin.label(),
                candidate.confidence,
                short_id(&candidate.id),
                short_id(&candidate.id)
            ),
            QuestionKind::Choice => format!(
                "「{}」的「{}」更可能是「{}」还是「{}」？（确认补全：{}）",
                candidate.triple.subject,
                candidate.triple.predicate,
                candidate.triple.object,
                alternatives.join("」/「"),
                short_id(&candidate.id)
            ),
            QuestionKind::Open => format!(
                "「{}」的「{}」应该是多少？",
                candidate.triple.subject, candidate.triple.predicate
            ),
        };
        let question = CompletionQuestion {
            candidate_id: candidate.id.clone(),
            triple: candidate.triple.clone(),
            kind,
            utility,
            text,
            alternatives,
        };
        *self
            .asked_questions
            .entry(question.candidate_id.clone())
            .or_insert(0) += 1;
        Some(question)
    }

    /// 推进逻辑轮次（无挂钟，TTL/审计均用此计数）。
    pub fn begin_turn(&mut self) {
        self.tick += 1;
    }

    /// 打开一个新回合，返回其在台账中的索引。旧回合超出上限时淘汰最旧的。
    pub fn open_episode(&mut self, focus: &str, question: Option<&str>) -> usize {
        self.tick += 1;
        if self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
            self.active = self.active.and_then(|index| index.checked_sub(1));
        }
        let id = stable_id(
            "ep-",
            &format!("{}@{}:{}", focus.trim(), self.tick, question.unwrap_or("")),
        );
        let index = self.episodes.len();
        self.episodes.push(CompletionEpisode {
            id,
            focus: focus.trim().to_string(),
            question: question.map(|value| value.trim().to_string()),
            opened_tick: self.tick,
            batch_entities: BTreeSet::new(),
            namespace: namespace_of(focus),
            candidates: Vec::new(),
            notes: Vec::new(),
            truncated: false,
        });
        self.active = Some(index);
        index
    }

    /// 设置本批实体范围（超出上限时截断并记录）。
    pub fn set_batch_scope<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = String>,
    {
        let max_batch_entities = self.max_batch_entities;
        let Some(episode) = self.current_episode_mut() else {
            return;
        };
        episode.batch_entities.clear();
        for entity in entities {
            let entity = entity.trim();
            if entity.is_empty() {
                continue;
            }
            if episode.batch_entities.len() >= max_batch_entities {
                episode.truncated = true;
                episode
                    .notes
                    .push(format!("本批实体超过 {} 个，已截断", max_batch_entities));
                break;
            }
            episode.batch_entities.insert(entity.to_string());
        }
    }

    pub fn note(&mut self, text: impl Into<String>) {
        if let Some(episode) = self.current_episode_mut() {
            episode.notes.push(text.into());
        }
    }

    /// 追加一条链接预测候选（warrant=Induced）。
    pub fn add_prediction(
        &mut self,
        prediction: &Prediction,
        rule: Option<&HornRule>,
        explanation: String,
    ) -> bool {
        let rule_spec = rule.and_then(RuleSpec::from_horn);
        let rule_id = rule_spec.as_ref().map(RuleSpec::id);
        self.insert_candidate(
            prediction.triple.clone(),
            Warrant::Induced,
            prediction.confidence,
            prediction.evidence.clone(),
            explanation,
            rule_spec,
            rule_id,
            CandidateOrigin::PathPrediction,
        )
    }

    /// 追加一条类比候选（warrant=Analogical，只能提示）。
    pub fn add_analogical(
        &mut self,
        triple: Triple,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
    ) -> bool {
        self.insert_candidate(
            triple,
            Warrant::Analogical,
            confidence,
            evidence,
            explanation,
            None,
            None,
            CandidateOrigin::Analogical,
        )
    }

    /// 手工提名（warrant 由调用方负责，适合接入其他候选来源）。
    pub fn add_candidate(
        &mut self,
        triple: Triple,
        warrant: Warrant,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
    ) -> bool {
        self.insert_candidate(
            triple,
            warrant,
            confidence,
            evidence,
            explanation,
            None,
            None,
            CandidateOrigin::Manual,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_candidate(
        &mut self,
        triple: Triple,
        warrant: Warrant,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
        rule: Option<RuleSpec>,
        rule_id: Option<String>,
        origin: CandidateOrigin,
    ) -> bool {
        if self.is_adjudicated(&triple) {
            return false;
        }
        let tick = self.tick;
        let max_candidates = self.max_candidates;
        let id = stable_id(
            "cand-",
            &format!("{}|{}|{}", triple.subject, triple.predicate, triple.object),
        );
        let Some(episode) = self.current_episode_mut() else {
            return false;
        };
        if let Some(existing) = episode
            .candidates
            .iter_mut()
            .find(|candidate| candidate.triple == triple)
        {
            if existing.status != CandidateStatus::Refuted {
                existing.warrant = existing.warrant.stronger(warrant);
                existing.confidence = existing.confidence.max(confidence);
                if explanation.len() > existing.explanation.len() {
                    existing.explanation = explanation;
                }
                merge_evidence(&mut existing.evidence, evidence);
                if existing.rule.is_none() {
                    existing.rule = rule;
                }
                if existing.rule_id.is_none() {
                    existing.rule_id = rule_id;
                }
            }
            return false;
        }
        if episode.candidates.len() >= max_candidates {
            episode.truncated = true;
            return false;
        }
        episode.candidates.push(CompletionCandidate {
            id,
            triple,
            warrant,
            status: CandidateStatus::Pending,
            confidence: confidence.clamp(0.0, 1.0),
            evidence,
            explanation,
            rule_id,
            rule,
            origin,
            added_tick: tick,
        });
        true
    }

    fn find_candidate_index(&self, selector: &str) -> Result<usize, EpisodeError> {
        let episode = self
            .current_episode()
            .ok_or(EpisodeError::NoActiveEpisode)?;
        let selector = selector.trim();
        if selector.is_empty() {
            return Err(EpisodeError::CandidateNotFound(selector.to_string()));
        }
        if let Ok(position) = selector.trim_start_matches('#').parse::<usize>() {
            if position == 0 || position > episode.candidates.len() {
                return Err(EpisodeError::CandidateNotFound(selector.to_string()));
            }
            return Ok(position - 1);
        }
        if let Some(position) = episode
            .candidates
            .iter()
            .position(|candidate| candidate.id == selector)
        {
            return Ok(position);
        }
        let prefix_matches: Vec<usize> = episode
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.id.starts_with(selector))
            .map(|(position, _)| position)
            .collect();
        match prefix_matches.as_slice() {
            [] => {}
            [position] => return Ok(*position),
            _ => {
                return Err(EpisodeError::AmbiguousSelector {
                    selector: selector.to_string(),
                    count: prefix_matches.len(),
                });
            }
        }
        if let Some((subject, predicate, object)) = parse_triple_selector(selector) {
            if let Some(position) = episode.candidates.iter().position(|candidate| {
                candidate.triple.subject == subject
                    && candidate.triple.predicate == predicate
                    && candidate.triple.object == object
            }) {
                return Ok(position);
            }
        }
        Err(EpisodeError::CandidateNotFound(selector.to_string()))
    }

    fn push_regression_test(
        &mut self,
        triple: Triple,
        verdict: RegressionVerdict,
        rule: Option<RuleSpec>,
        evidence: Vec<Triple>,
        source: &str,
        source_id: &str,
    ) -> String {
        let id = stable_id(
            "test-",
            &format!(
                "{}|{}|{}|{}",
                verdict.label(),
                triple.subject,
                triple.predicate,
                triple.object
            ),
        );
        self.regression_tests
            .retain(|test| test.triple != triple && test.id != id);
        self.regression_tests.push(CompletionTest {
            id: id.clone(),
            triple,
            verdict,
            rule,
            evidence,
            source: source.to_string(),
            source_id: source_id.to_string(),
            created_tick: self.tick,
        });
        if self.regression_tests.len() > DEFAULT_MAX_TESTS {
            let overflow = self.regression_tests.len() - DEFAULT_MAX_TESTS;
            self.regression_tests.drain(0..overflow);
        }
        id
    }

    /// 记录一次来自补全回合之外的裁决（恢复阶梯、导入检查或宿主脚本）。
    ///
    /// 确认会撤销同一三元组的旧否决并把归纳规则纳入台账；否决会把三元组
    /// 加入全局拒绝集，后续任何回合都不再提名。返回台账是否发生变化。
    pub fn record_external_verdict(
        &mut self,
        triple: Triple,
        confirmed: bool,
        rule: Option<RuleSpec>,
        evidence: Vec<Triple>,
        source: &str,
        source_id: &str,
    ) -> bool {
        if triple.subject.trim().is_empty()
            || triple.predicate.trim().is_empty()
            || triple.object.trim().is_empty()
        {
            return false;
        }
        let verdict = if confirmed {
            RegressionVerdict::Confirmed
        } else {
            RegressionVerdict::Refuted
        };
        let source_id = if source_id.trim().is_empty() {
            source.trim()
        } else {
            source_id.trim()
        };
        if self.regression_tests.iter().any(|test| {
            test.triple == triple && test.verdict == verdict && test.source_id == source_id
        }) {
            return false;
        }
        let tick = self.tick;
        let valid_rule = rule.filter(RuleSpec::is_valid);
        if confirmed {
            self.refuted_triples.remove(&triple);
            self.confirmed_triples.insert(triple.clone());
            self.regression_tests
                .retain(|test| !(test.triple == triple && test.verdict == RegressionVerdict::Refuted));
            if let Some(spec) = valid_rule.clone() {
                let id = spec.id();
                if let Some(rule) = self.rules.iter_mut().find(|rule| rule.id == id) {
                    rule.note_reinstatement(&triple, tick);
                }
                let episode_ref = format!("外部裁决:{}:{}", source.trim(), source_id);
                self.upsert_rule(spec, &triple, &episode_ref, 0.8, tick);
            }
        } else {
            self.confirmed_triples.remove(&triple);
            self.refuted_triples.insert(triple.clone());
            self.regression_tests
                .retain(|test| !(test.triple == triple && test.verdict == RegressionVerdict::Confirmed));
            if let Some(spec) = valid_rule.clone() {
                let id = spec.id();
                if let Some(rule) = self.rules.iter_mut().find(|rule| rule.id == id) {
                    rule.note_refutation(&triple, tick);
                }
            }
        }
        self.push_regression_test(
            triple,
            verdict,
            valid_rule,
            evidence,
            source.trim(),
            source_id,
        );
        true
    }

    /// 重放该规则相关的运行期知识测试。
    pub fn replay_rule(&self, rule_id: &str) -> Result<RegressionReport, EpisodeError> {
        let position = self.find_rule_index(rule_id)?;
        Ok(self.replay_rule_at(position))
    }

    /// 重放全部可用规则；任何违反历史裁决的规则立即隔离。
    ///
    /// 用于快照恢复、导入或显式自检：宁可少一条规则，也不让系统
    /// 带着与用户裁决冲突的知识继续回答。
    pub fn enforce_regression(&mut self) -> RegressionReport {
        let positions: Vec<usize> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.is_usable())
            .map(|(position, _)| position)
            .collect();
        let mut report = RegressionReport::default();
        let mut failing = Vec::new();
        for position in positions {
            let current = self.replay_rule_at(position);
            report.checked += current.checked;
            if !current.passed() {
                report.failures.extend(current.failures);
                failing.push(position);
            }
        }
        let tick = self.tick;
        for position in failing {
            if let Some(rule) = self.rules.get_mut(position) {
                rule.status = RuleStatus::Quarantined;
                rule.updated_tick = tick;
            }
        }
        report
    }

    fn replay_rule_at(&self, position: usize) -> RegressionReport {
        let Some(rule) = self.rules.get(position) else {
            return RegressionReport::default();
        };
        let mut report = RegressionReport::default();
        for test in &self.regression_tests {
            report.checked += 1;
            let rule_matches = test
                .rule
                .as_ref()
                .map(|spec| spec.id() == rule.id)
                .unwrap_or(false);
            match test.verdict {
                RegressionVerdict::Refuted => {
                    let rederives = !test.evidence.is_empty()
                        && rule_derives(&rule.spec, &test.evidence, &test.triple);
                    if rule_matches || rederives {
                        report.failures.push(RegressionFailure {
                            rule_id: rule.id.clone(),
                            test_id: test.id.clone(),
                            triple: test.triple.clone(),
                            reason: if rederives {
                                "规则会从当前证据重新推出已被否决的候选".to_string()
                            } else {
                                "规则曾产出被用户否决的候选".to_string()
                            },
                        });
                    }
                }
                RegressionVerdict::Confirmed => {
                    if rule_matches
                        && !test.evidence.is_empty()
                        && !rule_derives(&rule.spec, &test.evidence, &test.triple)
                    {
                        report.failures.push(RegressionFailure {
                            rule_id: rule.id.clone(),
                            test_id: test.id.clone(),
                            triple: test.triple.clone(),
                            reason: "规则不再能从确认证据推出已确认事实".to_string(),
                        });
                    }
                }
            }
        }
        report
    }

    pub fn render_regression_report(&self) -> String {
        if self.regression_tests.is_empty() {
            return "还没有运行期知识测试。用户确认或驳回补全候选后会自动登记。"
                .to_string();
        }
        let confirmed = self
            .regression_tests
            .iter()
            .filter(|test| test.verdict == RegressionVerdict::Confirmed)
            .count();
        let mut output = format!(
            "运行期知识测试 {} 条（确认 {} / 驳回 {}）\n",
            self.regression_tests.len(),
            confirmed,
            self.regression_tests.len() - confirmed
        );
        for test in self.regression_tests.iter().rev().take(10) {
            output.push_str(&format!(
                "  {}（来源：{}，轮次 {}）\n",
                test.display_line(),
                test.source,
                test.created_tick
            ));
        }
        output.trim_end().to_string()
    }

    /// 确认一条候选：标记状态、归纳规则，并在本批范围上做有界闭包。
    pub fn confirm(
        &mut self,
        selector: &str,
        facts: &[Triple],
    ) -> Result<ConfirmOutcome, EpisodeError> {
        let position = self.find_candidate_index(selector)?;
        let active = self.active.ok_or(EpisodeError::NoActiveEpisode)?;
        let candidate = self.episodes[active].candidates[position].clone();
        if candidate.status != CandidateStatus::Pending {
            return Err(EpisodeError::CandidateAlreadyClosed(selector.to_string()));
        }
        let tick = self.tick;
        let episode_id = self.episodes[active].id.clone();
        let rule_id = if let Some(spec) = candidate.rule.clone() {
            let id = self.upsert_rule(
                spec,
                &candidate.triple,
                &episode_id,
                candidate.confidence,
                tick,
            );
            Some(id)
        } else {
            None
        };
        self.episodes[active].candidates[position].status = CandidateStatus::Confirmed;
        self.episodes[active].candidates[position].rule_id = rule_id.clone();
        self.refuted_triples.remove(&candidate.triple);
        self.confirmed_triples.insert(candidate.triple.clone());
        self.push_regression_test(
            candidate.triple.clone(),
            RegressionVerdict::Confirmed,
            candidate.rule.clone(),
            candidate.evidence.clone(),
            "补全回合",
            &episode_id,
        );
        let rule_scope = rule_id
            .as_deref()
            .and_then(|id| self.rule(id))
            .map(|rule| rule.scope);

        let mut closure_truncated = false;
        let mut new_candidates = 0usize;
        let learned: Vec<LearnedRule> = self
            .rules
            .iter()
            .filter(|rule| rule.is_usable())
            .cloned()
            .collect();
        let remaining = self.max_closure;
        let mut produced = 0usize;
        for rule in &learned {
            if produced >= remaining {
                closure_truncated = true;
                break;
            }
            let filter = self.scope_filter(rule.scope, active, facts);
            let (hits, truncated) = apply_rule(
                &rule.spec,
                facts,
                &filter,
                remaining.saturating_sub(produced),
            );
            closure_truncated |= truncated;
            for hit in hits {
                if produced >= remaining {
                    closure_truncated = true;
                    break;
                }
                let explanation = format!(
                    "由本回合规则「{}」（{}作用域）推出；证据：{}",
                    rule.spec.rule_text(),
                    rule.scope.label(),
                    hit.evidence
                        .iter()
                        .map(display_triple)
                        .collect::<Vec<_>>()
                        .join("；")
                );
                let accepted = self.insert_candidate(
                    hit.triple,
                    Warrant::Induced,
                    rule.confidence,
                    hit.evidence,
                    explanation,
                    Some(rule.spec.clone()),
                    Some(rule.id.clone()),
                    CandidateOrigin::BatchClosure,
                );
                if accepted {
                    new_candidates += 1;
                    produced += 1;
                }
            }
        }
        if closure_truncated {
            if let Some(episode) = self.current_episode_mut() {
                episode.truncated = true;
                episode
                    .notes
                    .push("本批闭包达到上限，未补全部分不会假装完整".to_string());
            }
        }
        Ok(ConfirmOutcome {
            candidate: candidate.triple,
            rule_id,
            rule_scope,
            learned: candidate.rule.is_some(),
            new_candidates,
            closure_truncated,
        })
    }

    /// 驳回一条候选：写入负例；同一规则被反复驳回时自动隔离。
    pub fn refute(&mut self, selector: &str) -> Result<RefuteOutcome, EpisodeError> {
        let position = self.find_candidate_index(selector)?;
        let active = self.active.ok_or(EpisodeError::NoActiveEpisode)?;
        let candidate = self.episodes[active].candidates[position].clone();
        if candidate.status != CandidateStatus::Pending {
            return Err(EpisodeError::CandidateAlreadyClosed(selector.to_string()));
        }
        let tick = self.tick;
        self.episodes[active].candidates[position].status = CandidateStatus::Refuted;
        self.refuted_triples.insert(candidate.triple.clone());
        self.confirmed_triples.remove(&candidate.triple);
        let episode_id = self.episodes[active].id.clone();
        self.push_regression_test(
            candidate.triple.clone(),
            RegressionVerdict::Refuted,
            candidate.rule.clone(),
            candidate.evidence.clone(),
            "补全回合",
            &episode_id,
        );
        let mut quarantined = false;
        let mut rule_id = candidate.rule_id.clone();
        if rule_id.is_none() {
            rule_id = candidate.rule.as_ref().map(RuleSpec::id);
        }
        if let Some(id) = &rule_id {
            if let Some(rule) = self.rules.iter_mut().find(|rule| &rule.id == id) {
                quarantined = rule.note_refutation(&candidate.triple, tick);
            }
        }
        let mut removed_pending = 0usize;
        if quarantined {
            let rule_id = rule_id.clone();
            if let Some(episode) = self.episodes.get_mut(active) {
                let before = episode.candidates.len();
                episode.candidates.retain(|candidate| {
                    !(candidate.status == CandidateStatus::Pending
                        && candidate.rule_id.as_deref() == rule_id.as_deref())
                });
                removed_pending = before - episode.candidates.len();
            }
        }
        let rule_status = rule_id
            .as_deref()
            .and_then(|id| self.rule(id))
            .map(|rule| rule.status);
        Ok(RefuteOutcome {
            candidate: candidate.triple,
            rule_id,
            rule_status,
            quarantined,
            removed_pending,
        })
    }

    /// 按完整 id 或唯一前缀查找规则（展示层给的是短 id）。
    fn find_rule_index(&self, selector: &str) -> Result<usize, EpisodeError> {
        let selector = selector.trim().trim_start_matches('#');
        if selector.is_empty() {
            return Err(EpisodeError::RuleNotFound(selector.to_string()));
        }
        if let Some(position) = self.rules.iter().position(|rule| rule.id == selector) {
            return Ok(position);
        }
        let prefix_matches: Vec<usize> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.id.starts_with(selector))
            .map(|(position, _)| position)
            .collect();
        match prefix_matches.as_slice() {
            [] => Err(EpisodeError::RuleNotFound(selector.to_string())),
            [position] => Ok(*position),
            _ => Err(EpisodeError::AmbiguousRuleSelector {
                selector: selector.to_string(),
                count: prefix_matches.len(),
            }),
        }
    }

    /// 显式晋升规则作用域。晋升是用户动作，但仍拒绝已隔离/退役的规则。
    pub fn promote(
        &mut self,
        rule_id: &str,
        target: RuleScope,
    ) -> Result<PromotionOutcome, EpisodeError> {
        let tick = self.tick;
        let position = self.find_rule_index(rule_id)?;
        let regression = self.replay_rule_at(position);
        let rule = &mut self.rules[position];
        if !rule.is_usable() {
            return Err(EpisodeError::RuleUnavailable(rule.id.clone()));
        }
        if !regression.passed() {
            return Err(EpisodeError::PromotionBlocked(regression.summary()));
        }
        if target.rank() <= rule.scope.rank() {
            return Err(EpisodeError::PromotionNotHigher {
                from: rule.scope,
                to: target,
            });
        }
        if target == RuleScope::Global && rule.refutations > 0 {
            return Err(EpisodeError::PromotionBlocked(format!(
                "全局晋升要求零反例，但该规则已有 {} 条驳回",
                rule.refutations
            )));
        }
        let from = rule.scope;
        let recommended = target.recommended_episodes();
        let warning = if rule.episodes.len() < recommended {
            Some(format!(
                "证据偏薄：只在 {} 个回合确认，建议至少 {} 个",
                rule.episodes.len(),
                recommended
            ))
        } else {
            None
        };
        rule.scope = target;
        rule.promoted_tick = Some(tick);
        rule.updated_tick = tick;
        if rule.confirmations >= 2 {
            rule.status = RuleStatus::Active;
        }
        Ok(PromotionOutcome {
            rule_id: rule.id.clone(),
            from,
            to: target,
            warning,
        })
    }

    /// 显式退役规则：停止参与闭包与全局归结，但保留审计记录。
    ///
    /// 已确认的补全事实属于会话事实，不会因规则退役而被撤回；
    /// 退役只阻止该规则继续派生新结论。
    pub fn retire(&mut self, rule_id: &str) -> Result<RuleLifecycleOutcome, EpisodeError> {
        let position = self.find_rule_index(rule_id)?;
        let tick = self.tick;
        let rule = &mut self.rules[position];
        if rule.status == RuleStatus::Retired {
            return Err(EpisodeError::RuleAlreadyRetired(rule.id.clone()));
        }
        let previous = rule.status;
        rule.status = RuleStatus::Retired;
        rule.updated_tick = tick;
        Ok(RuleLifecycleOutcome {
            rule_id: rule.id.clone(),
            scope: rule.scope,
            previous,
            current: rule.status,
        })
    }

    /// 恢复已退役规则：按确认次数回到候选（<2）或生效（>=2）。
    pub fn reactivate(&mut self, rule_id: &str) -> Result<RuleLifecycleOutcome, EpisodeError> {
        let position = self.find_rule_index(rule_id)?;
        let regression = self.replay_rule_at(position);
        if !regression.passed() {
            return Err(EpisodeError::RegressionBlocked(regression.summary()));
        }
        let tick = self.tick;
        let rule = &mut self.rules[position];
        if rule.status != RuleStatus::Retired {
            return Err(EpisodeError::RuleNotRetired(rule.id.clone()));
        }
        let previous = rule.status;
        rule.status = if rule.confirmations >= 2 {
            RuleStatus::Active
        } else {
            RuleStatus::Candidate
        };
        rule.updated_tick = tick;
        Ok(RuleLifecycleOutcome {
            rule_id: rule.id.clone(),
            scope: rule.scope,
            previous,
            current: rule.status,
        })
    }

    /// 只有显式晋升到全局的规则才进入全局归结与软逻辑。
    ///
    /// 命名空间规则继续在补全回合内按实体范围参与候选生成，但不进入
    /// 全局归结：归结引擎目前没有事实级命名空间标签，把命名空间规则倒进
    /// 全局引擎会让它跨域 join，等于把「命名空间」退化成全局。
    pub fn reasoning_clauses(&self) -> Vec<Clause> {
        self.rules
            .iter()
            .filter(|rule| rule.is_usable() && rule.scope == RuleScope::Global)
            .filter_map(|rule| {
                rule.spec.to_clause(
                    rule.confidence,
                    &format!(
                        "补全回合规则 {}（{}）",
                        rule.spec.rule_text(),
                        rule.scope.label()
                    ),
                )
            })
            .collect()
    }

    fn upsert_rule(
        &mut self,
        spec: RuleSpec,
        example: &Triple,
        episode_id: &str,
        confidence: f64,
        tick: u64,
    ) -> String {
        let id = spec.id();
        if let Some(rule) = self.rules.iter_mut().find(|rule| rule.id == id) {
            rule.note_confirmation(episode_id, example, tick, confidence);
            return id;
        }
        if self.rules.len() >= self.max_rules {
            let position = self
                .rules
                .iter()
                .position(|rule| {
                    matches!(rule.status, RuleStatus::Quarantined | RuleStatus::Retired)
                })
                .unwrap_or(0);
            self.rules.remove(position);
        }
        self.rules.push(LearnedRule {
            id: id.clone(),
            spec,
            scope: RuleScope::Batch,
            status: RuleStatus::Candidate,
            confirmations: 1,
            refutations: 0,
            episodes: BTreeSet::from([episode_id.to_string()]),
            positive_examples: vec![example.clone()],
            negative_examples: Vec::new(),
            confidence: confidence.clamp(0.0, 1.0),
            created_tick: tick,
            updated_tick: tick,
            source_episode: episode_id.to_string(),
            promoted_tick: None,
        });
        id
    }

    fn scope_filter(
        &self,
        scope: RuleScope,
        active: usize,
        facts: &[Triple],
    ) -> ScopeFilter {
        let Some(episode) = self.episodes.get(active) else {
            return ScopeFilter::Entities(HashSet::new());
        };
        match scope {
            RuleScope::Example => ScopeFilter::Entities(HashSet::from([episode.focus.clone()])),
            RuleScope::Batch => {
                if episode.batch_entities.is_empty() {
                    ScopeFilter::Entities(HashSet::from([episode.focus.clone()]))
                } else {
                    ScopeFilter::Entities(
                        episode.batch_entities.iter().cloned().collect::<HashSet<_>>(),
                    )
                }
            }
            RuleScope::Namespace => {
                if episode.namespace.trim().is_empty() {
                    // 无显式命名空间时退回本批，绝不意外放大到全图。
                    return self.scope_filter(RuleScope::Batch, active, facts);
                }
                let entities = facts
                    .iter()
                    .flat_map(|triple| [triple.subject.as_str(), triple.object.as_str()])
                    .filter(|entity| namespace_of(entity) == episode.namespace)
                    .map(str::to_string)
                    .collect();
                ScopeFilter::Entities(entities)
            }
            RuleScope::Global => ScopeFilter::All,
        }
    }

    pub fn render_episode(&self) -> String {
        let Some(episode) = self.current_episode() else {
            return "还没有打开的补全回合。用「补全回合：<实体>」开始一次学习回合。"
                .to_string();
        };
        let mut output = format!(
            "补全回合 {}（焦点「{}」，候选 {}，待确认 {}，已确认 {}，已驳回 {}）\n",
            short_id(&episode.id),
            episode.focus,
            episode.candidates.len(),
            episode.pending_count(),
            episode.confirmed_count(),
            episode.refuted_count()
        );
        if !episode.batch_entities.is_empty() {
            let shown: Vec<&str> = episode
                .batch_entities
                .iter()
                .take(8)
                .map(String::as_str)
                .collect();
            output.push_str(&format!(
                "  本批实体 {} 个：{}{}\n",
                episode.batch_entities.len(),
                shown.join("、"),
                if episode.batch_entities.len() > shown.len() {
                    "……"
                } else {
                    ""
                }
            ));
        }
        if episode.candidates.is_empty() {
            output.push_str("  （没有候选：可先导入更多关系，或降低链接预测阈值）\n");
        }
        for (index, candidate) in episode.candidates.iter().enumerate() {
            let marker = match candidate.status {
                CandidateStatus::Pending => format!(
                    "[{} · {}]",
                    candidate.warrant.label(),
                    candidate.warrant.hedge().trim_matches(['（', '）'])
                ),
                CandidateStatus::Confirmed => "[已确认]".to_string(),
                CandidateStatus::Refuted => "[已驳回]".to_string(),
            };
            output.push_str(&format!(
                "  {}. {} {}（{}，来源：{}，置信度 {:.2}）\n",
                index + 1,
                marker,
                candidate.display_triple(),
                candidate.status.label(),
                candidate.origin.label(),
                candidate.confidence
            ));
            if !candidate.explanation.trim().is_empty() {
                output.push_str(&format!("     {}\n", candidate.explanation.trim()));
            }
        }
        for note in &episode.notes {
            output.push_str(&format!("  注：{}\n", note));
        }
        if episode.truncated {
            output.push_str("  本批结果已截断：未补全部分不等于不存在。\n");
        }
        output.push_str(
            "  操作：确认补全：<序号> / 否定补全：<序号> / 补全规则 / 晋升补全规则：<规则id>|<作用域>",
        );
        output.trim_end().to_string()
    }

    pub fn render_rules(&self) -> String {
        if self.rules.is_empty() {
            return "还没有归纳出的补全规则。先打开回合并确认一条候选。".to_string();
        }
        let mut output = format!("补全规则（{} 条，默认本批作用域）\n", self.rules.len());
        for (index, rule) in self.rules.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} id={}\n",
                index + 1,
                rule.display_line(),
                short_id(&rule.id)
            ));
            if !rule.positive_examples.is_empty() {
                let examples: Vec<String> = rule
                    .positive_examples
                    .iter()
                    .take(3)
                    .map(display_triple)
                    .collect();
                output.push_str(&format!("     正例：{}\n", examples.join("；")));
            }
            if !rule.negative_examples.is_empty() {
                let examples: Vec<String> = rule
                    .negative_examples
                    .iter()
                    .take(3)
                    .map(display_triple)
                    .collect();
                output.push_str(&format!("     负例：{}\n", examples.join("；")));
            }
        }
        output.trim_end().to_string()
    }
}
#[derive(Debug, Clone)]
enum ScopeFilter {
    Entities(HashSet<String>),
    All,
}
impl ScopeFilter {
    fn allows(&self, entity: &str) -> bool {
        match self {
            ScopeFilter::Entities(entities) => entities.contains(entity),
            ScopeFilter::All => true,
        }
    }
}
#[derive(Debug, Clone)]
struct ClosureHit {
    triple: Triple,
    evidence: Vec<Triple>,
}
/// 在一个明确的作用域内做规则闭包；返回（新三元组, 是否因上限截断）。
fn apply_rule(
    spec: &RuleSpec,
    facts: &[Triple],
    filter: &ScopeFilter,
    max: usize,
) -> (Vec<ClosureHit>, bool) {
    if !spec.is_valid() || max == 0 {
        return (Vec::new(), false);
    }
    let mut out: HashMap<&str, HashMap<&str, HashSet<&str>>> = HashMap::new();
    let mut into: HashMap<&str, HashMap<&str, HashSet<&str>>> = HashMap::new();
    let mut pairs: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
    for triple in facts {
        let subject = triple.subject.as_str();
        let object = triple.object.as_str();
        let predicate = triple.predicate.as_str();
        out.entry(predicate)
            .or_default()
            .entry(subject)
            .or_default()
            .insert(object);
        into.entry(predicate)
            .or_default()
            .entry(object)
            .or_default()
            .insert(subject);
        pairs
            .entry(predicate)
            .or_default()
            .push((subject, object));
    }

    let max_scan = max.saturating_mul(CLOSURE_SCAN_FACTOR);
    let mut scanned = 0usize;
    let mut truncated = false;
    let mut hits: BTreeMap<Triple, Vec<Triple>> = BTreeMap::new();
    let already_exists = |subject: &str, object: &str| {
        out.get(spec.head.as_str())
            .and_then(|by_subject| by_subject.get(subject))
            .map_or(false, |objects| objects.contains(object))
    };
    let push = |triple: Triple, evidence: Vec<Triple>, hits: &mut BTreeMap<Triple, Vec<Triple>>| {
        match hits.get_mut(&triple) {
            Some(current) if &evidence < current => {
                *current = evidence;
            }
            Some(_) => {}
            None => {
                hits.insert(triple, evidence);
            }
        }
    };

    match spec.pattern {
        RulePattern::Compose => {
            let Some(first) = out.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = out.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (subject, middle_entities) in first {
                if !filter.allows(subject) {
                    continue;
                }
                for middle in middle_entities {
                    let Some(objects) = second.get(*middle) else {
                        continue;
                    };
                    for object in objects {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == object || already_exists(subject, object) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, object),
                            vec![
                                Triple::new(subject, &spec.body[0], middle),
                                Triple::new(middle, &spec.body[1], object),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        RulePattern::Inverse => {
            let Some(body_pairs) = pairs.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            for (subject, object) in body_pairs {
                scanned += 1;
                if scanned > max_scan {
                    truncated = true;
                    break;
                }
                if subject == object
                    || !filter.allows(object)
                    || already_exists(object, subject)
                {
                    continue;
                }
                push(
                    Triple::new(object, &spec.head, subject),
                    vec![Triple::new(subject, &spec.body[0], object)],
                    &mut hits,
                );
                if hits.len() >= max {
                    truncated = true;
                    break;
                }
            }
        }
        RulePattern::CommonNeighbor => {
            let Some(first) = into.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = into.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (anchor, subjects) in first {
                let Some(others) = second.get(*anchor) else {
                    continue;
                };
                for subject in subjects {
                    if !filter.allows(subject) {
                        continue;
                    }
                    for other in others {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == other || already_exists(subject, other) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, other),
                            vec![
                                Triple::new(subject, &spec.body[0], anchor),
                                Triple::new(other, &spec.body[1], anchor),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        RulePattern::CommonObject => {
            let Some(first) = out.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = out.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (anchor, subjects) in first {
                let Some(others) = second.get(*anchor) else {
                    continue;
                };
                for subject in subjects {
                    if !filter.allows(subject) {
                        continue;
                    }
                    for other in others {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == other || already_exists(subject, other) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, other),
                            vec![
                                Triple::new(anchor, &spec.body[0], subject),
                                Triple::new(anchor, &spec.body[1], other),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    (
        hits.into_iter()
            .map(|(triple, evidence)| ClosureHit { triple, evidence })
            .collect(),
        truncated,
    )
}
/// 在给定证据上验证规则是否能推出目标三元组。
fn rule_derives(spec: &RuleSpec, evidence: &[Triple], target: &Triple) -> bool {
    if !spec.is_valid() {
        return false;
    }
    let (hits, _) = apply_rule(spec, evidence, &ScopeFilter::All, 8);
    hits.iter().any(|hit| &hit.triple == target)
}
fn merge_evidence(current: &mut Vec<Triple>, incoming: Vec<Triple>) {
    for triple in incoming {
        if !current.contains(&triple) {
            current.push(triple);
        }
    }
    current.sort();
}
fn parse_triple_selector(selector: &str) -> Option<(String, String, String)> {
    if let Some((subject, rest)) = selector.split_once('|') {
        if let Some((predicate, object)) = rest.split_once('|') {
            let subject = subject.trim();
            let predicate = predicate.trim();
            let object = object.trim();
            if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                return Some((
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                ));
            }
        }
    }
    let parts: Vec<&str> = selector.split_whitespace().collect();
    if parts.len() == 3 && parts.iter().all(|part| !part.is_empty()) {
        return Some((
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        ));
    }
    None
}
/// 命名空间前缀：取最后一个 `:`、`/` 或 `#` 之前的部分；无分隔符则为空。
fn namespace_of(entity: &str) -> String {
    let entity = entity.trim();
    match entity.rfind([':', '/', '#']) {
        Some(index) if index > 0 => entity[..=index].to_string(),
        _ => String::new(),
    }
}
fn display_triple(triple: &Triple) -> String {
    format!(
        "{}({}, {})",
        triple.predicate, triple.subject, triple.object
    )
}

impl PatternEpisodeLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
            clock: 0,
            revision: 0,
        }
    }

    pub fn entries(&self) -> &[PatternEpisode] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clock(&self) -> usize {
        self.clock
    }

    /// 进程内变更计数：任何增删改（含反馈）都会推进；用于决定是否落图。
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn matched_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|episode| episode.outcome == EpisodeOutcome::Matched)
            .count()
    }

    pub fn unmatched_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|episode| episode.outcome == EpisodeOutcome::Unmatched)
            .count()
    }

    pub fn clear(&mut self) {
        if !self.entries.is_empty() {
            self.entries.clear();
            self.touch();
        }
    }

    /// 删除单条经验：按完整 id、短 id、完整原句或至少 4 位十六进制 id 前缀匹配。
    /// 返回删除条数；用于撤销误确认/误否定而不清空整本经验。
    pub fn forget(&mut self, needle: &str) -> usize {
        let needle = needle.trim();
        if needle.is_empty() {
            return 0;
        }
        let prefix = needle.chars().count() >= 4
            && needle.chars().all(|ch| ch.is_ascii_hexdigit())
            && needle.chars().count() < 16;
        let before = self.entries.len();
        self.entries.retain(|episode| {
            let id = episode.id();
            !(id == needle
                || episode.short_id() == needle
                || episode.input.trim() == needle
                || (prefix && id.starts_with(needle)))
        });
        let removed = before - self.entries.len();
        if removed > 0 {
            self.touch();
        }
        removed
    }

    /// 记录一次成功理解（同身份经验累计次数并刷新最后轮次）。
    pub fn record_match(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        matched: &PatternMatch,
        focus: Option<&str>,
        turn: usize,
    ) -> PatternEpisode {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            turn,
            ..PatternContext::default()
        };
        self.record_match_with_context(input, tokens, matched, &context)
    }

    /// 记录一次带完整对话上下文的成功理解。
    pub fn record_match_with_context(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        matched: &PatternMatch,
        context: &PatternContext,
    ) -> PatternEpisode {
        let bindings = matched
            .bindings
            .iter()
            .filter_map(|(name, term)| term.render_raw().map(|value| (name.clone(), value)))
            .collect();
        let episode = PatternEpisode {
            input: input.to_string(),
            tokens: token_pairs(tokens),
            winner_id: Some(matched.entry.id()),
            winner_name: Some(matched.entry.name.clone()),
            query_type: Some(matched.entry.query_type.clone()),
            entities: matched.entities.clone(),
            bindings,
            focus: context.focus.clone(),
            focus_type: context.focus_type.clone(),
            last_move: context.last_move.clone(),
            pending_clarify: Some(context.pending_clarify),
            outcome: EpisodeOutcome::Matched,
            count: 1,
            acceptances: 0,
            rejections: 0,
            first_turn: self.effective_turn(context.turn),
            last_turn: self.effective_turn(context.turn),
            frontier: None,
        };
        self.upsert(episode.clone());
        self.entries
            .iter()
            .find(|entry| entry.id() == episode.id())
            .cloned()
            .unwrap_or(episode)
    }

    /// 记录一次未命中（用于「上次为什么没懂」与候选骨架检索）。
    pub fn record_unmatched(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        turn: usize,
    ) -> PatternEpisode {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            turn,
            ..PatternContext::default()
        };
        self.record_unmatched_with_context(input, tokens, &context)
    }

    /// 记录一次带完整对话上下文的未命中。
    pub fn record_unmatched_with_context(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> PatternEpisode {
        self.record_unmatched_with_frontier(input, tokens, context, None)
    }

    /// 记录未命中，并保存最近受阻模式的决策前沿。
    ///
    /// `frontier` 只描述**为什么没懂**（停在哪个要素、期望/实见什么），
    /// 不参与经验身份与检索排序；旧账本没有该字段时按 `None` 恢复。
    pub fn record_unmatched_with_frontier(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        frontier: Option<EpisodeFrontier>,
    ) -> PatternEpisode {
        let episode = PatternEpisode {
            input: input.to_string(),
            tokens: token_pairs(tokens),
            winner_id: None,
            winner_name: None,
            query_type: None,
            entities: Vec::new(),
            bindings: Vec::new(),
            focus: context.focus.clone(),
            focus_type: context.focus_type.clone(),
            last_move: context.last_move.clone(),
            pending_clarify: Some(context.pending_clarify),
            outcome: EpisodeOutcome::Unmatched,
            count: 1,
            acceptances: 0,
            rejections: 0,
            first_turn: self.effective_turn(context.turn),
            last_turn: self.effective_turn(context.turn),
            frontier,
        };
        self.upsert(episode.clone());
        self.entries
            .iter()
            .find(|entry| entry.id() == episode.id())
            .cloned()
            .unwrap_or(episode)
    }

    fn effective_turn(&mut self, turn: usize) -> usize {
        let effective = if turn == 0 { self.clock + 1 } else { turn };
        self.clock = self.clock.max(effective);
        effective
    }

    fn upsert(&mut self, episode: PatternEpisode) -> bool {
        let id = episode.id();
        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            existing.count = existing.count.saturating_add(1);
            existing.last_turn = existing.last_turn.max(episode.last_turn);
            existing.tokens = episode.tokens;
            existing.entities = episode.entities;
            existing.bindings = episode.bindings;
            existing.acceptances = existing.acceptances.max(episode.acceptances);
            existing.rejections = existing.rejections.max(episode.rejections);
            if episode.frontier.is_some() {
                existing.frontier = episode.frontier;
            }
            self.touch();
            return false;
        }
        self.entries.push(episode);
        self.enforce_capacity();
        self.touch();
        true
    }

    fn enforce_capacity(&mut self) {
        while self.entries.len() > self.capacity {
            // 失败经验先被淘汰；同为失败/成功时再淘汰支持度低、最久未用的。
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, episode)| {
                    (
                        u8::from(episode.outcome == EpisodeOutcome::Matched),
                        episode.count,
                        episode.last_turn,
                        episode.id(),
                    )
                })
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    self.entries.remove(index);
                }
                None => break,
            }
        }
    }

    /// 并入从图账本恢复的经验（同身份取较大支持度与较新轮次）。
    pub fn merge_persisted(&mut self, episodes: &[PatternEpisode]) -> usize {
        let mut added = 0usize;
        for episode in episodes {
            let id = episode.id();
            if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id() == id) {
                existing.count = existing.count.max(episode.count);
                existing.acceptances = existing.acceptances.max(episode.acceptances);
                existing.rejections = existing.rejections.max(episode.rejections);
                existing.first_turn = existing.first_turn.min(episode.first_turn);
                existing.last_turn = existing.last_turn.max(episode.last_turn);
                if episode.frontier.is_some() {
                    existing.frontier = episode.frontier.clone();
                }
            } else {
                self.entries.push(episode.clone());
                added += 1;
            }
        }
        self.clock = self.clock.max(
            self.entries
                .iter()
                .map(|episode| episode.last_turn)
                .max()
                .unwrap_or(0),
        );
        self.enforce_capacity();
        if !episodes.is_empty() {
            self.touch();
        }
        added
    }

    /// 按共享词面 + 焦点 + 结果检索相似经验（确定性排序）。
    pub fn nearest(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        limit: usize,
    ) -> Vec<EpisodeNeighbor> {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            ..PatternContext::default()
        };
        self.nearest_with_context(tokens, &context, limit)
    }

    /// 按共享词面 + 完整对话上下文 + 结果检索相似经验（确定性排序）。
    ///
    /// 上下文是**加性弱条件**：词面结构仍是主信号，焦点/类型/上轮动作/
    /// 待澄清只在相近结构之间做消歧，避免同句式在错误对话状态下被召回。
    pub fn nearest_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        limit: usize,
    ) -> Vec<EpisodeNeighbor> {
        if limit == 0 {
            return Vec::new();
        }
        let current: BTreeSet<&str> = tokens
            .iter()
            .map(|token| token.text.as_str())
            .filter(|text| !is_skippable(text))
            .collect();
        let mut neighbors: Vec<EpisodeNeighbor> = Vec::new();
        for (index, episode) in self.entries.iter().enumerate() {
            let stored = episode.surface_set();
            let shared: Vec<String> = episode
                .tokens
                .iter()
                .map(|(text, _)| text.as_str())
                .filter(|text| current.contains(text))
                .fold(Vec::new(), |mut acc, text| {
                    if !acc.iter().any(|seen| seen == text) {
                        acc.push(text.to_string());
                    }
                    acc
                });
            let union = current.union(&stored).count().max(1);
            let overlap = current.intersection(&stored).count() as f64 / union as f64;
            let context_bonus = episode.context_affinity(context);
            let outcome_bonus = if episode.outcome == EpisodeOutcome::Matched {
                0.05
            } else {
                0.0
            };
            // 被用户否定过的成功经验仍可检索（解释为什么不再推荐），
            // 但排序时扣分；净否定越多扣得越重，最多扣到 0。
            let rejection_penalty = episode
                .rejections
                .saturating_sub(episode.acceptances)
                .min(3) as f64
                * 0.2;
            if shared.is_empty() && context_bonus == 0.0 {
                continue;
            }
            neighbors.push(EpisodeNeighbor {
                index,
                score: (overlap + context_bonus + outcome_bonus - rejection_penalty)
                    .clamp(0.0, 1.0),
                shared,
            });
        }
        neighbors.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.entries[b.index]
                        .context_affinity(context)
                        .partial_cmp(&self.entries[a.index].context_affinity(context))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    self.entries[b.index]
                        .count
                        .cmp(&self.entries[a.index].count)
                })
                .then_with(|| self.entries[a.index].id().cmp(&self.entries[b.index].id()))
        });
        neighbors.truncate(limit);
        neighbors
    }

    /// 由最相似的成功经验泛化出「临时模式」：共享词保留为词面，
    /// 当前输入中的新内容词升为槽位。候选项 `route=false`，只能被确认后教学激活。
    pub fn suggest_case(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        min_score: f64,
    ) -> Option<CaseSuggestion> {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            ..PatternContext::default()
        };
        self.suggest_case_with_context(tokens, &context, min_score)
    }

    /// 带完整对话上下文的案例联想。
    pub fn suggest_case_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        min_score: f64,
    ) -> Option<CaseSuggestion> {
        // 取前若干个邻居逐个尝试：最近的一条可能是否定过的骨架、
        // 未命中的经验，或泛化后无法解释当前输入；不应因此放弃次近的经验。
        for neighbor in self.nearest_with_context(tokens, context, 5) {
            if neighbor.score < min_score {
                break;
            }
            let Some(episode) = self.entries.get(neighbor.index).cloned() else {
                continue;
            };
            if episode.outcome != EpisodeOutcome::Matched
                || episode.rejections > episode.acceptances
            {
                continue;
            }
            let shared = episode.surface_set();
            let mut elems = Vec::new();
            let mut slot_number = 0usize;
            for token in tokens {
                let text = token.text.trim();
                if text.is_empty() || is_skippable(text) {
                    continue;
                }
                if shared.contains(text) {
                    elems.push(Elem::word(text));
                    continue;
                }
                let sort = match token.pos.as_str() {
                    "nr" => "人名",
                    "ns" | "nt" => "地点",
                    "n" | "nz" | "vn" => "实体",
                    "m" => "数量",
                    _ => "",
                };
                if sort.is_empty() {
                    elems.push(Elem::word(text));
                } else {
                    slot_number += 1;
                    elems.push(Elem::slot(format!("槽{}", slot_number), sort));
                }
            }
            if elems.is_empty() {
                continue;
            }
            let Some(query_type) = episode.query_type.clone() else {
                continue;
            };
            let mut entry = PatternEntry::new(
                format!(
                    "案例候选：{}",
                    episode.winner_name.as_deref().unwrap_or("经验")
                ),
                format!("case#{}", episode.short_id()),
                query_type,
                5,
                false,
                elems,
            );
            // 当前输入没有显式实体（靠焦点补全的省略/指代问句）时，
            // 候选必须带上焦点守卫；确认转正后也不会在无焦点时误答。
            let has_explicit_entity = tokens
                .iter()
                .any(|token| matches!(token.pos.as_str(), "nr" | "ns" | "nt"));
            if !has_explicit_entity {
                if let Some(sort) = &context.focus_type {
                    entry = entry.with_guard(ContextGuard::focus_type(sort.clone()));
                } else if context.focus.is_some() {
                    entry = entry.with_guard(ContextGuard::focus());
                }
            }
            // 候选必须能解释当前输入；否则继续看下一条经验。
            if match_entry(&entry, tokens).is_err() {
                continue;
            }
            return Some(CaseSuggestion {
                episode,
                score: neighbor.score,
                shared: neighbor.shared,
                pattern: entry,
            });
        }
        None
    }

    /// 记录用户确认了某条案例联想（正向反馈，随经验账本持久化）。
    pub fn confirm_case(&mut self, suggestion: &CaseSuggestion) -> bool {
        let id = suggestion.episode.id();
        if let Some(episode) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            episode.acceptances = episode.acceptances.saturating_add(1);
            self.touch();
            return true;
        }
        false
    }

    /// 记录用户否定了某条案例联想（负向反馈；净否定的骨架不再主动推荐）。
    pub fn reject_case(&mut self, suggestion: &CaseSuggestion) -> bool {
        let id = suggestion.episode.id();
        if let Some(episode) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            episode.rejections = episode.rejections.saturating_add(1);
            self.touch();
            return true;
        }
        false
    }
}
fn token_pairs(tokens: &[crate::nlp::TaggedWord]) -> Vec<(String, String)> {
    tokens
        .iter()
        .map(|token| (token.text.clone(), token.pos.clone()))
        .collect()
}
/// 单个候选的评估结果（报告用；不携带完整解，避免大对象复制）。
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateOutcome {
    pub id: String,
    pub name: String,
    pub query_type: String,
    pub source: String,
    pub priority: i32,
    pub route: bool,
    pub status: CandidateStatus,
}
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    /// 命中（`rank` 为命中序，从 0 起；按优先级+配置序）。
    Matched {
        rank: usize,
        specificity: f64,
        span: Span,
        bindings: usize,
    },
    /// 未命中（首个失配要素的原因）。
    Failed { reason: String },
}
/// 快速路由的一次扫描：胜者 + 实际完整评估/被首要素预筛跳过的条数。
#[derive(Debug, Clone, PartialEq)]
pub struct RouteScan {
    pub winner: Option<PatternMatch>,
    /// 进入完整 `match_entry` 的候选数。
    pub evaluated: usize,
    /// 首要素必然失配、无需完整匹配的候选数。
    pub skipped: usize,
}
impl RouteScan {
    pub fn scanned(&self) -> usize {
        self.evaluated + self.skipped
    }
}
/// 一次匹配的完整账目：胜者、全部命中、全部失败原因。
#[derive(Debug, Clone, PartialEq)]
pub struct MatchReport {
    pub winner: Option<PatternMatch>,
    pub matched: Vec<CandidateOutcome>,
    pub failed: Vec<CandidateOutcome>,
}
impl MatchReport {
    pub fn matched_count(&self) -> usize {
        self.matched.len()
    }

    /// 同覆盖区间但路由不同的命中（潜在冲突，体检/解释用）。
    pub fn conflicts(&self) -> Vec<(String, String, Span)> {
        let mut out = Vec::new();
        for (i, a) in self.matched.iter().enumerate() {
            let span_a = match a.status {
                CandidateStatus::Matched { span, .. } => span,
                _ => continue,
            };
            for b in self.matched.iter().skip(i + 1) {
                if a.query_type == b.query_type {
                    continue;
                }
                if let CandidateStatus::Matched { span, .. } = b.status {
                    if span == span_a {
                        out.push((a.name.clone(), b.name.clone(), span_a));
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 匹配器
// ---------------------------------------------------------------------------
/// 标点/空白 token：匹配推进时跳过（与既有语义一致）。
fn is_skippable(text: &str) -> bool {
    text.trim().is_empty() || matches!(text, "？" | "。" | "，" | "、")
}
/// 句首礼貌/话语标记：不承载查询语义，但会破坏锚定匹配。
///
/// 只在**句首**跳过，且当模式首要素正好需要该词面时不跳；避免把
/// 「那一年」里的「那」当成可丢弃填充词。
const LEADING_FILLERS: &[&str] = &[
    "请问",
    "请问一下",
    "我想问",
    "想问",
    "问一下",
    "麻烦",
    "那么",
    "嗯",
    "呃",
];
fn skip_leading_fillers(entry: &PatternEntry, tokens: &[crate::nlp::TaggedWord]) -> usize {
    let first_literal = match entry.elems.first() {
        Some(Elem::Word { surface }) => Some(surface.as_str()),
        _ => None,
    };
    let mut pos = 0usize;
    while pos < tokens.len() {
        let text = tokens[pos].text.as_str();
        if !LEADING_FILLERS.contains(&text) {
            break;
        }
        if first_literal == Some(text) {
            break;
        }
        pos += 1;
    }
    pos
}
const CITY_LITERALS: &[&str] = &[
    "厦门", "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉",
];
const PERSON_LITERALS: &[&str] = &["姚明", "刘德华", "张三", "李四", "王五", "李明"];
const DATE_WORDS: &[&str] = &["明天", "后天", "今天", "昨天", "大后天"];
/// 槽位排序约束：与 `nlu::NLUEngine::extract_slot_value` 同口径
/// （词性集 + 词面提示），锚定在当前位置的 token 上。
fn sort_matches(sort: &str, hints: &[String], token: &crate::nlp::TaggedWord) -> bool {
    match sort {
        "城市" | "地点" => {
            crate::pos_ontology::tag_matches("ns", &token.pos)
                || crate::pos_ontology::tag_matches("n", &token.pos)
                || CITY_LITERALS.contains(&token.text.as_str())
                || hints.iter().any(|hint| token.text.contains(hint.as_str()))
        }
        "日期" | "时间" => {
            DATE_WORDS.contains(&token.text.as_str())
                || (crate::pos_ontology::tag_matches("m", &token.pos)
                    && (token.text.contains('天')
                        || token.text.contains('月')
                        || token.text.contains('年')))
                || hints.iter().any(|hint| token.text.contains(hint.as_str()))
        }
        "人名" => {
            crate::pos_ontology::tag_matches("nr", &token.pos)
                || PERSON_LITERALS.contains(&token.text.as_str())
        }
        "数量" => {
            crate::pos_ontology::tag_matches("m", &token.pos)
                || token.text.chars().all(|c| c.is_numeric())
        }
        "实体" => {
            crate::pos_ontology::is_entity_tag(&token.pos)
                || PERSON_LITERALS.contains(&token.text.as_str())
        }
        "属性" => {
            let profile = crate::pos_ontology::profile(&token.pos);
            profile.is_nominal()
                && !profile.has_sort(crate::pos_ontology::Sort::Person)
                && !profile.has_sort(crate::pos_ontology::Sort::Place)
                && !profile.has_sort(crate::pos_ontology::Sort::Time)
        }
        _ => true,
    }
}
/// 单要素尝试的结果（只在成功时产生副作用）。
enum Attempt {
    Consumed {
        new_pos: usize,
        /// 停留在 `new_pos` 这个 token 内的字节偏移（0 = 已越过 token 边界）。
        new_offset: usize,
        span: Span,
        entity: Option<String>,
        binding: Option<(String, String)>,
    },
    /// 可选要素未占位。
    Absent,
}
/// 当前 token 是否是「下一个词面要素」的开头片段。
///
/// 用于人名/人名槽的跨词合并：结巴把「张三多大」切成 `张三/nr 多/unknown
/// 大/nr` 时，若不看下一个词面，`nr` 会把「多」「大」一起吞掉。
fn next_literal_starts_with(next_literal: Option<&str>, token: &str) -> bool {
    next_literal
        .map(|literal| literal.len() > token.len() && literal.starts_with(token))
        .unwrap_or(false)
}
fn attempt_elem(
    elem: &Elem,
    pattern: &[Elem],
    index: usize,
    tokens: &[crate::nlp::TaggedWord],
    pos: usize,
    offset: usize,
) -> Result<Attempt, ElemFailure> {
    let token = &tokens[pos];
    let remainder = token.text.get(offset..).unwrap_or("");
    match elem {
        Elem::Optional(inner) => {
            attempt_elem(inner, pattern, index, tokens, pos, offset).or(Ok(Attempt::Absent))
        }
        Elem::Pos { pos: expected } if expected == "nr" => {
            // 人名/整词词性只能从 token 边界开始，避免把半个词当人名。
            if offset != 0
                || (!crate::pos_ontology::tag_matches("nr", &token.pos)
                    && token.pos != "unknown")
            {
                return Err(ElemFailure::new(
                    MatchFailureKind::PosMismatch,
                    "期望人名(nr)",
                    token,
                ));
            }
            let mut parts = vec![token.text.clone()];
            let mut end = pos + 1;
            let next_literal = match pattern.get(index + 1) {
                Some(Elem::Word { surface }) => Some(surface.as_str()),
                _ => None,
            };
            while end < tokens.len()
                && (tokens[end].pos == "unknown"
                    || crate::pos_ontology::tag_matches("nr", &tokens[end].pos))
            {
                if next_literal == Some(tokens[end].text.as_str())
                    || next_literal_starts_with(next_literal, tokens[end].text.as_str())
                {
                    // 「张三多大」被切成 `张三/nr 多/unknown 大/nr` 时，
                    // 人名合并不能把下一个词面的首字吞进名字。
                    break;
                }
                parts.push(tokens[end].text.clone());
                end += 1;
            }
            Ok(Attempt::Consumed {
                new_pos: end,
                new_offset: 0,
                span: Span::new(pos, end),
                entity: Some(parts.join("")),
                binding: None,
            })
        }
        Elem::Pos { pos: expected } => {
            if offset != 0 || !crate::pos_ontology::tag_matches(expected, &token.pos) {
                return Err(ElemFailure::new(
                    MatchFailureKind::PosMismatch,
                    format!("期望词性「{}」", expected),
                    token,
                ));
            }
            let entity = if crate::pos_ontology::captures_entity(expected) {
                Some(token.text.clone())
            } else {
                None
            };
            Ok(Attempt::Consumed {
                new_pos: pos + 1,
                new_offset: 0,
                span: Span::new(pos, pos + 1),
                entity,
                binding: None,
            })
        }
        Elem::Word { surface } => {
            // 同一 token 内已经消费过前缀时，只允许从剩余部分继续；
            // 这样「解释一下」可以先吃「解释」再吃「一下」。
            if remainder == surface {
                return Ok(Attempt::Consumed {
                    new_pos: pos + 1,
                    new_offset: 0,
                    span: Span::new(pos, pos + 1),
                    entity: None,
                    binding: None,
                });
            }
            if !remainder.is_empty() && remainder.starts_with(surface.as_str()) {
                let new_offset = offset + surface.len();
                if new_offset >= token.text.len() {
                    return Ok(Attempt::Consumed {
                        new_pos: pos + 1,
                        new_offset: 0,
                        span: Span::new(pos, pos + 1),
                        entity: None,
                        binding: None,
                    });
                }
                return Ok(Attempt::Consumed {
                    new_pos: pos,
                    new_offset,
                    span: Span::new(pos, pos + 1),
                    entity: None,
                    binding: None,
                });
            }
            // 跨词拼接：词面被分词器切成多个 token 时，按 token 边界
            // 累积，只有**恰好拼成完整词面**才消费。这样「多大」可以是
            // `多 + 大`，「有什么」可以是 `有 + 什么`；但不会把只是
            // 前缀相同的 token 误吞（例如「什么」对「什么关系」）。
            if surface.chars().count() > 1 {
                let mut concat = String::from(remainder);
                let mut temp = pos + 1;
                while temp < tokens.len() {
                    concat.push_str(tokens[temp].text.as_str());
                    temp += 1;
                    if concat == surface.as_str() {
                        return Ok(Attempt::Consumed {
                            new_pos: temp,
                            new_offset: 0,
                            span: Span::new(pos, temp),
                            entity: None,
                            binding: None,
                        });
                    }
                    if !surface.starts_with(concat.as_str()) {
                        break;
                    }
                }
            }
            // 兼容既有「包含匹配」：只在 token 边界且没有更精确解时使用；
            // 它会消费整个 token，因此把词面拆到同一 token 内时优先走上面的
            // 前缀/跨词路径。
            if offset == 0
                && (token.text.contains(surface.as_str()) || surface.contains(&token.text))
            {
                return Ok(Attempt::Consumed {
                    new_pos: pos + 1,
                    new_offset: 0,
                    span: Span::new(pos, pos + 1),
                    entity: None,
                    binding: None,
                });
            }
            Err(ElemFailure::new(
                MatchFailureKind::WordMismatch,
                surface.clone(),
                token,
            ))
        }
        Elem::Slot { name, sort, hints } => {
            if offset == 0 && sort_matches(sort, hints, token) {
                // 人名槽与 `nr` 一样要能跨过结巴切开的后续字：
                // 「王强」可能分成「王/nr」「强/unknown」，但槽仍应绑定
                // 完整词面，而不是只吃掉第一个字后让后续要素失配。
                let mut end = pos + 1;
                if sort == "人名" {
                    let next_literal = match pattern.get(index + 1) {
                        Some(Elem::Word { surface }) => Some(surface.as_str()),
                        _ => None,
                    };
                    while end < tokens.len() {
                        let next = &tokens[end];
                        if is_skippable(next.text.as_str())
                            || !(next.pos == "unknown" || next.pos == "nr")
                            || next_literal == Some(next.text.as_str())
                            || next_literal_starts_with(next_literal, next.text.as_str())
                        {
                            break;
                        }
                        end += 1;
                    }
                }
                let surface: String = tokens[pos..end]
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect();
                Ok(Attempt::Consumed {
                    new_pos: end,
                    new_offset: 0,
                    span: Span::new(pos, end),
                    entity: Some(surface.clone()),
                    binding: Some((name.clone(), surface)),
                })
            } else {
                Err(ElemFailure::new(
                    MatchFailureKind::SortMismatch,
                    format!("词槽「{}」排序不匹配（期望{}）", name, sort),
                    token,
                ))
            }
        }
        Elem::Any => Ok(Attempt::Consumed {
            new_pos: pos + 1,
            new_offset: 0,
            span: Span::new(pos, pos + 1),
            entity: None,
            binding: None,
        }),
    }
}
/// 对一条模式做锚定匹配，失败时返回结构化的决策前沿。
///
/// 这是匹配语义的**唯一实现**：日志、评估和路由都从这里取结果，避免
/// 「解释器」与「匹配器」各写一遍而漂移。旧的字符串失败原因只是它的
/// [`Display`] 投影。
pub fn match_entry_detailed(
    entry: &PatternEntry,
    tokens: &[crate::nlp::TaggedWord],
) -> Result<PatternMatch, MatchFailure> {
    let elems = &entry.elems;
    if elems.is_empty() {
        // 与既有语义一致：空模式命中空解
        return Ok(PatternMatch {
            entry: entry.clone(),
            bindings: BindingSet::new(),
            entities: Vec::new(),
            entity_spans: Vec::new(),
            elem_spans: Vec::new(),
            span: Span::new(0, 0),
        });
    }

    let mut pos = skip_leading_fillers(entry, tokens);
    let mut offset = 0usize;
    let mut bindings = BindingSet::new();
    let mut entities = Vec::new();
    let mut entity_spans = Vec::new();
    let mut elem_spans: Vec<Option<Span>> = Vec::with_capacity(elems.len());
    let mut span_start: Option<usize> = None;
    let mut span_end: usize = 0;
    let mut entity_counter = 0usize;

    for (index, elem) in elems.iter().enumerate() {
        while pos < tokens.len() && offset == 0 && is_skippable(tokens[pos].text.as_str()) {
            pos += 1;
        }
        if pos < tokens.len() && offset >= tokens[pos].text.len() {
            pos += 1;
            offset = 0;
        }
        if pos >= tokens.len() {
            if matches!(elem, Elem::Optional(_)) {
                elem_spans.push(None);
                continue;
            }
            return Err(MatchFailure {
                index,
                total_elems: elems.len(),
                position: pos,
                kind: MatchFailureKind::InputExhausted,
                expected: elem.render(),
                actual: None,
                actual_pos: None,
                matched_elems: elem_spans,
                remaining_tokens: 0,
            });
        }

        let attempt = match attempt_elem(elem, elems, index, tokens, pos, offset) {
            Ok(attempt) => attempt,
            Err(error) => {
                return Err(MatchFailure {
                    index,
                    total_elems: elems.len(),
                    position: pos,
                    kind: error.kind,
                    expected: error.expected,
                    actual: error.actual,
                    actual_pos: error.actual_pos,
                    matched_elems: elem_spans,
                    remaining_tokens: tokens.len().saturating_sub(pos),
                });
            }
        };
        match attempt {
            Attempt::Absent => elem_spans.push(None),
            Attempt::Consumed {
                new_pos,
                new_offset,
                span,
                entity,
                binding,
            } => {
                if let Some(entity) = entity {
                    entity_counter += 1;
                    let key = format!("entity.{}", entity_counter);
                    bindings.bind(&key, &entity);
                    entities.push(entity);
                    entity_spans.push(span);
                }
                if let Some((name, value)) = binding {
                    bindings.bind(name, &value);
                }
                span_start.get_or_insert(span.start);
                span_end = span_end.max(span.end);
                pos = new_pos;
                offset = new_offset;
                elem_spans.push(Some(span));
            }
        }
    }

    Ok(PatternMatch {
        entry: entry.clone(),
        bindings,
        entities,
        entity_spans,
        elem_spans,
        span: Span::new(span_start.unwrap_or(0), span_end),
    })
}
/// 内部路由/报告使用的字符串失败口；语义完全来自
/// [`match_entry_detailed`]。
fn match_entry(
    entry: &PatternEntry,
    tokens: &[crate::nlp::TaggedWord],
) -> Result<PatternMatch, String> {
    match_entry_detailed(entry, tokens).map_err(|failure| failure.to_string())
}
/// 首要素预筛：只判断「首要素是否必然失配」，成功集必须是完整匹配的超集。
///
/// 复用 [`attempt_elem`] 的匹配语义（含人名合并、逐字跨词、包含匹配），
/// 因此不会把本可命中的条目误跳过；可选首要素一律保守放行。
fn first_elem_may_miss(entry: &PatternEntry, tokens: &[crate::nlp::TaggedWord]) -> bool {
    let Some(first) = entry.elems.first() else {
        return false;
    };
    let mut pos = skip_leading_fillers(entry, tokens);
    while pos < tokens.len() && is_skippable(tokens[pos].text.as_str()) {
        pos += 1;
    }
    if pos >= tokens.len() {
        return !matches!(first, Elem::Optional(_));
    }
    attempt_elem(first, &entry.elems, 0, tokens, pos, 0).is_err()
}

// ---------------------------------------------------------------------------
// 模式库
// ---------------------------------------------------------------------------
/// 配置层热替换的回执：旧基础条数、新基础条数、保留下来的会话条数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternLayerRefresh {
    pub removed: usize,
    pub added: usize,
    pub kept: usize,
}
/// 模式库：配置查询 + NLU 意图 + 教学模式的统一登记处。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PatternLibrary {
    entries: Vec<PatternEntry>,
}
impl PatternLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// 由一组 `[[prolog_rules]]` 编译（全部参与路由）。
    pub fn from_prolog_rules(rules: &[PrologRuleConfig]) -> Self {
        Self {
            entries: rules.iter().map(PatternEntry::from_prolog_rule).collect(),
        }
    }

    /// 全量编译：配置查询（可路由）+ NLU 默认意图（同库展示，不路由）。
    pub fn from_config(config: &SystemConfig) -> Self {
        let mut library = Self::from_prolog_rules(&config.prolog_rules);
        for domain in crate::nlu::default_domains() {
            for intent in &domain.intents {
                // NLU 意图与同名配置规则对齐时，复用配置的 query_type，
                // 作为主路由未命中后的回退（route=false 仍不抢配置匹配）。
                let aligned_type = config
                    .prolog_rules
                    .iter()
                    .find(|rule| rule.name == intent.name)
                    .map(|rule| rule.query_type.clone());
                for pattern in &intent.patterns {
                    let mut entry = PatternEntry::from_nlu_intent(&domain, intent, pattern);
                    if let Some(query_type) = &aligned_type {
                        entry.query_type = query_type.clone();
                        entry.fallback = true;
                    }
                    library.entries.push(entry);
                }
            }
        }
        // 输入在路由前已归一化；模式词面必须落在同一规范形上，否则
        // 「老婆 → 妻子」这类映射会让规则永远匹配不到自己的词面。
        if let Some(synonyms) = &config.synonyms {
            for entry in &mut library.entries {
                entry.normalize_literals(&synonyms.mappings);
            }
        }
        library
    }

    pub fn entries(&self) -> &[PatternEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn route_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.route).count()
    }

    /// 未对齐到可执行路由的 NLU 意图数（回退条目单独计入 [`Self::fallback_count`]）。
    pub fn nlu_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| !entry.route && !entry.fallback)
            .count()
    }

    pub fn fallback_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.fallback).count()
    }

    /// 按来源前缀计数（`taught#` / `config.toml#` / `nlu#` …）。
    pub fn count_with_source_prefix(&self, prefix: &str) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.source.starts_with(prefix))
            .count()
    }

    /// 登记一条模式：同 id（IR+路由+来源）原地替换，返回是否为新增。
    pub fn add(&mut self, entry: PatternEntry) -> bool {
        let id = entry.id();
        if let Some(slot) = self.entries.iter_mut().find(|existing| existing.id() == id) {
            *slot = entry;
            false
        } else {
            self.entries.push(entry);
            true
        }
    }

    /// 用户确认一条案例候选：把 `route=false` 的临时骨架转正为可路由模式。
    ///
    /// 转正后的来源为 `taught#案例#<经验 id>`，因此与显式教学一样：
    /// - 参与后续路由与匹配解释；
    /// - 随模式账本落图、跨会话恢复；
    /// - 可被 [`PatternLibrary::forget`] 遗忘。
    ///
    /// 候选本身不可路由（避免未经确认的泛化直接答题）；重复确认同一案例
    /// 幂等（同 id 原地替换）。
    pub fn accept_case(&mut self, suggestion: &CaseSuggestion) -> Result<PatternEntry, String> {
        if suggestion.pattern.route {
            return Err("案例候选已经处于可路由状态".to_string());
        }
        if suggestion.pattern.elems.is_empty() {
            return Err("案例候选没有模式要素".to_string());
        }
        if suggestion.pattern.query_type.trim().is_empty() {
            return Err("案例候选缺少路由类型".to_string());
        }
        let mut entry = suggestion.pattern.clone();
        entry.route = true;
        entry.source = format!("taught#案例#{}", suggestion.episode.short_id());
        entry.name = entry
            .name
            .strip_prefix("案例候选：")
            .map(|name| format!("案例确认：{}", name))
            .unwrap_or_else(|| format!("案例确认：{}", entry.name));
        self.add(entry.clone());
        Ok(entry)
    }

    /// 教一条模式（运行时即刻参与匹配与路由；来源 `taught#会话`）。
    pub fn teach(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
    ) -> Result<PatternEntry, String> {
        self.teach_with_guard(name, pattern, query_type, priority, None)
    }

    /// 教学并附带上下文守卫（省略/指代模式只在相应对话状态下成立）。
    pub fn teach_with_guard(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
        guard: Option<ContextGuard>,
    ) -> Result<PatternEntry, String> {
        if name.trim().is_empty() {
            return Err("模式名称不能为空".to_string());
        }
        if query_type.trim().is_empty() {
            return Err("路由类型（query_type）不能为空".to_string());
        }
        let elems = parse_teach_pattern(pattern)?;
        let mut entry = PatternEntry::new(
            name.trim(),
            "taught#会话",
            query_type.trim(),
            priority,
            true,
            elems,
        );
        if let Some(guard) = guard {
            entry = entry.with_guard(guard);
        }
        self.add(entry.clone());
        Ok(entry)
    }

    /// 忘掉一条教学模式（按 id 前缀或精确名；只删 `taught#` 来源）。
    pub fn forget(&mut self, id_or_name: &str) -> Option<PatternEntry> {
        let key = id_or_name.trim();
        if key.is_empty() {
            return None;
        }
        let position = self.entries.iter().position(|entry| {
            entry.source.starts_with("taught#")
                && (entry.id().starts_with(key) || entry.name == key)
        })?;
        Some(self.entries.remove(position))
    }

    /// 按精确来源移除条目（如一条锻造规则的全部展开）。
    pub fn remove_source(&mut self, source: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.source != source);
        before - self.entries.len()
    }

    /// 按来源前缀移除条目（如 `induced#forge#` 的全部会话规则）。
    ///
    /// 只影响运行时模式库；调用方仍负责同步落图与台账。
    pub fn remove_source_prefix(&mut self, prefix: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|entry| !entry.source.starts_with(prefix));
        before - self.entries.len()
    }

    /// 配置层条目的来源前缀：`config.toml` 查询规则与内置 NLU 意图。
    pub fn is_config_layer_source(source: &str) -> bool {
        source.starts_with("config.toml#") || source.starts_with("nlu#")
    }

    /// 原子替换配置层：重编译查询规则与 NLU 回退，保留教学/归纳/锻造来源。
    ///
    /// 热加载只换「语言策略」这一层；会话中学到的模式、已确认案例与锻造
    /// 展开不随配置重建。返回移除了多少旧基础条目、加入了多少新基础条目。
    pub fn replace_config_layer(&mut self, config: &SystemConfig) -> PatternLayerRefresh {
        let removed = self
            .entries
            .iter()
            .filter(|entry| Self::is_config_layer_source(&entry.source))
            .count();
        let runtime: Vec<PatternEntry> = self
            .entries
            .iter()
            .filter(|entry| !Self::is_config_layer_source(&entry.source))
            .cloned()
            .collect();
        let kept = runtime.len();
        let mut next = Self::from_config(config);
        let added = next.entries.len();
        for entry in runtime {
            next.add(entry);
        }
        *self = next;
        PatternLayerRefresh {
            removed,
            added,
            kept,
        }
    }

    /// 并入外部载入的条目（教学与归纳来源；配置/NLU 每次启动重建）。
    pub fn merge_taught(&mut self, entries: &[PatternEntry]) -> usize {
        let mut merged = 0;
        for entry in entries {
            if (entry.source.starts_with("taught#") || entry.source.starts_with("induced#"))
                && self.add(entry.clone())
            {
                merged += 1;
            }
        }
        merged
    }

    /// 归纳一条模式（来源 `induced#<seed>`；参与路由）。
    ///
    /// 与 [`PatternLibrary::teach`] 同一落库路径，只是来源不同：
    /// 归纳出的模式来自对话诱导/最小泛化，而不是显式教学。
    pub fn induce(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
        seed: &str,
    ) -> Result<PatternEntry, String> {
        if name.trim().is_empty() {
            return Err("模式名称不能为空".to_string());
        }
        if query_type.trim().is_empty() {
            return Err("路由类型（query_type）不能为空".to_string());
        }
        let elems = parse_teach_pattern(pattern)?;
        let entry = PatternEntry::new(
            name.trim(),
            format!("induced#{}", seed.trim()),
            query_type.trim(),
            priority,
            true,
            elems,
        );
        self.add(entry.clone());
        Ok(entry)
    }

    /// 评估序：优先级降序 + 配置序（稳定排序，与既有路由一致）。
    fn evaluation_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by(|a, b| self.entries[*b].priority.cmp(&self.entries[*a].priority));
        order
    }

    /// 全量评估：胜者（第一条可路由的命中）+ 命中榜 + 失败原因榜。
    pub fn rank(&self, tokens: &[crate::nlp::TaggedWord]) -> MatchReport {
        self.rank_with_context(tokens, &PatternContext::default())
    }

    /// 带对话上下文的全量评估：守卫不满足的模式进入失败榜并给出原因。
    pub fn rank_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> MatchReport {
        let mut winner: Option<PatternMatch> = None;
        let mut matched = Vec::new();
        let mut failed = Vec::new();
        let mut rank = 0usize;

        for index in self.evaluation_order() {
            let entry = &self.entries[index];
            if !entry.context_satisfied(context) {
                let guard = entry
                    .guard
                    .as_ref()
                    .map(ContextGuard::render)
                    .unwrap_or_default();
                failed.push(CandidateOutcome {
                    id: entry.id(),
                    name: entry.name.clone(),
                    query_type: entry.query_type.clone(),
                    source: entry.source.clone(),
                    priority: entry.priority,
                    route: entry.route,
                    status: CandidateStatus::Failed {
                        reason: format!("上下文守卫不满足（{}）", guard),
                    },
                });
                continue;
            }
            match match_entry(entry, tokens) {
                Ok(m) => {
                    if winner.is_none() && entry.route {
                        winner = Some(m.clone());
                    }
                    matched.push(CandidateOutcome {
                        id: entry.id(),
                        name: entry.name.clone(),
                        query_type: entry.query_type.clone(),
                        source: entry.source.clone(),
                        priority: entry.priority,
                        route: entry.route,
                        status: CandidateStatus::Matched {
                            rank,
                            specificity: entry.specificity(),
                            span: m.span,
                            bindings: m.bindings.len(),
                        },
                    });
                    rank += 1;
                }
                Err(reason) => failed.push(CandidateOutcome {
                    id: entry.id(),
                    name: entry.name.clone(),
                    query_type: entry.query_type.clone(),
                    source: entry.source.clone(),
                    priority: entry.priority,
                    route: entry.route,
                    status: CandidateStatus::Failed { reason },
                }),
            }
        }

        MatchReport {
            winner,
            matched,
            failed,
        }
    }

    /// 取第一条可路由的命中（查询路径热路径；不必算完整报告）。
    pub fn match_route(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<PatternMatch> {
        self.match_route_with_context(tokens, &PatternContext::default())
    }

    /// 带上下文守卫的路由：守卫不满足的模式在本次路由中被跳过。
    pub fn match_route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<PatternMatch> {
        self.scan_with_context(tokens, context, false).winner
    }

    /// 快速路由并返回扫描统计（首要素预筛省掉了多少次完整匹配）。
    pub fn scan_route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> RouteScan {
        self.scan_with_context(tokens, context, false)
    }

    /// NLU 回退路由：仅当主路由无命中时使用，按同样优先级与守卫评估。
    pub fn match_fallback(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<PatternMatch> {
        self.match_fallback_with_context(tokens, &PatternContext::default())
    }

    /// 带上下文守卫的 NLU 回退路由。
    pub fn match_fallback_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<PatternMatch> {
        self.scan_with_context(tokens, context, true).winner
    }

    /// 主路由/回退路由共用的扫描器：保留评估序，首要素必然失配的直接跳过。
    fn scan_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        fallback: bool,
    ) -> RouteScan {
        let mut evaluated = 0usize;
        let mut skipped = 0usize;
        for index in self.evaluation_order() {
            let entry = &self.entries[index];
            let eligible = if fallback {
                entry.fallback
            } else {
                entry.route
            };
            if !eligible || !entry.context_satisfied(context) {
                continue;
            }
            if first_elem_may_miss(entry, tokens) {
                skipped += 1;
                continue;
            }
            evaluated += 1;
            if let Ok(m) = match_entry(entry, tokens) {
                return RouteScan {
                    winner: Some(m),
                    evaluated,
                    skipped,
                };
            }
        }
        RouteScan {
            winner: None,
            evaluated,
            skipped,
        }
    }

    /// 与既有 `nlp::analyze_query_intent` 同构的路由口。
    pub fn route(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<(String, Vec<String>)> {
        self.match_route(tokens)
            .map(|m| (m.entry.query_type.clone(), m.entities.clone()))
    }

    /// 带上下文守卫的路由口。
    pub fn route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<(String, Vec<String>)> {
        self.match_route_with_context(tokens, context)
            .map(|m| (m.entry.query_type.clone(), m.entities.clone()))
    }

    /// 按 id 前缀或名称查条目。
    pub fn get(&self, id_or_name: &str) -> Option<&PatternEntry> {
        let key = id_or_name.trim();
        self.entries
            .iter()
            .find(|entry| entry.id().starts_with(key) || entry.name == key)
    }
}

// ---------------------------------------------------------------------------
// 报告渲染
// ---------------------------------------------------------------------------

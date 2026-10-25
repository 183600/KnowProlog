//! Entity Thread Digest（实体脉络账本）：编年史 × 身份注册表的只读投影。
//!
//! 设计法则：
//!   1. 写端只追加「词面 + 写时身份」（`EntityRef`），读端用**当前**注册表折叠
//!      别名、合并与重定向；日志本身永不回写，哈希链保持完整。
//!   2. 旧日志没有 refs 也不丢：读端按注册表的已确认词面做确定性扫描，把旧词面
//!      延迟绑定到今天的身份（标注为「迟到绑定」，不冒充当时的确认）。
//!   3. 解析不了就保持歧义/未解析，绝不静默压掉；未解析词面单独审计。
//!   4. 线程有界（保留最近 [`MAX_EVENTS_PER_THREAD`] 条），折叠数量显式披露。
//!
//! 与相邻模块的边界：Chronicle 是唯一真相源；EntityRegistry 是身份镜头；
//! 本模块不维护第二份可变对话状态，也不写 RDF。每次查询全量重投影，输出确定、
//! 可回放到 `seq`。

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::chronicle::{Chronicle, ChronicleEntry, EntityRef};
use crate::entity_registry::{fold_surface, EntityRegistry, EntityResolution, EntityStatus};
use crate::identity::{entity_uri, DEFAULT_BASE_NS};
use crate::prolog_engine::PrologFact;

/// 单个实体线程保留的最近事件上限；更早的事件折叠计数仍可审计。
pub const MAX_EVENTS_PER_THREAD: usize = 128;

/// 事件在对话中的角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadRole {
    Turn,
    Reply,
    Other,
}

impl ThreadRole {
    pub const fn label(self) -> &'static str {
        match self {
            ThreadRole::Turn => "提问",
            ThreadRole::Reply => "回答",
            ThreadRole::Other => "其他",
        }
    }
}

/// 同一实体在某一条编年史记录里的词面证据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadMention {
    /// 日志中实际出现的词面。
    pub surface: String,
    /// 写端记录的身份（旧日志可能为空）。
    pub stored_entity: Option<String>,
    /// 证据来源：`link` / `focus` / `text` / `ref`。
    pub sources: BTreeSet<String>,
    /// 该词面只在读端扫描/焦点时才落到身份（当时并未写死）。
    pub late_bound: bool,
    /// 写时身份在今天的注册表里已重定向（旧身份仍保留展示）。
    pub redirected_from: Option<String>,
}

/// 一条编年史记录在某个实体线程中的投影。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadEvent {
    pub seq: u64,
    pub kind: String,
    pub role: ThreadRole,
    pub question: Option<String>,
    pub answer: Option<String>,
    pub focus: Option<String>,
    pub kb_revision: u64,
    pub mentions: Vec<ThreadMention>,
}

impl ThreadEvent {
    /// 事件正文：优先展示当轮输入，其次回答。
    pub fn text(&self) -> &str {
        self.question
            .as_deref()
            .or(self.answer.as_deref())
            .unwrap_or_default()
    }
}

/// 一个实体的跨会话脉络。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityThread {
    /// 当前规范身份（读端折叠后的最终目标）。
    pub canonical: String,
    /// 与规范身份一一对应的稳定 URI。
    pub uri: String,
    pub types: BTreeSet<String>,
    pub observed: bool,
    /// 注册表里可解析到该身份的当前词面（规范名 + 已确认别名 + 曾用名）。
    pub known_surfaces: BTreeSet<String>,
    /// 日志中实际出现过的词面。
    pub surfaces: BTreeSet<String>,
    /// 日志里记载、如今已重定向到当前身份的旧身份。
    pub old_identities: BTreeSet<String>,
    /// 最近事件（有界；更早的折叠到 `omitted_events`）。
    pub events: Vec<ThreadEvent>,
    pub total_events: usize,
    pub omitted_events: usize,
    pub first_seq: Option<u64>,
    pub last_seq: Option<u64>,
    pub question_events: usize,
    pub reply_events: usize,
    pub late_bound_events: usize,
}

/// 未解析（或歧义）的提及：保留在审计视图里，不强行归到某个身份。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnresolvedMention {
    pub seqs: BTreeSet<u64>,
    pub sources: BTreeSet<String>,
    pub candidates: BTreeSet<String>,
}

/// 实体线索索引：全量编年史的只读投影。
#[derive(Debug, Clone)]
pub struct EntityThreadDigest {
    pub threads: BTreeMap<String, EntityThread>,
    pub unresolved: BTreeMap<String, UnresolvedMention>,
    pub entries: usize,
    pub mentions: usize,
    registry: EntityRegistry,
}

/// 查询结论：唯一线程 / 歧义候选 / 无记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadLookup<'a> {
    Found(&'a EntityThread),
    Ambiguous(Vec<String>),
    Unknown,
}

impl EntityThreadDigest {
    /// 从编年史与当前身份注册表构建投影（纯读，不修改日志）。
    pub fn build(chronicle: &Chronicle, registry: &EntityRegistry) -> Self {
        let surfaces = RegistrySurfaces::build(registry);
        let mut digest = EntityThreadDigest {
            threads: BTreeMap::new(),
            unresolved: BTreeMap::new(),
            entries: chronicle.len(),
            mentions: 0,
            registry: registry.clone(),
        };
        for entry in chronicle.entries() {
            digest.ingest(entry, &surfaces);
        }
        digest.finish(&surfaces);
        digest
    }

    /// 按名称/别名/曾用名查找线程（不猜：多解时返回候选）。
    pub fn lookup(&self, query: &str) -> ThreadLookup<'_> {
        let query = query.trim();
        if query.is_empty() {
            return ThreadLookup::Unknown;
        }
        match self.registry.resolve(query) {
            EntityResolution::Canonical(canonical)
            | EntityResolution::Redirected { to: canonical, .. } => {
                return match self.threads.get(&canonical) {
                    Some(thread) => ThreadLookup::Found(thread),
                    None => ThreadLookup::Unknown,
                }
            }
            EntityResolution::Ambiguous(candidates) => {
                return ThreadLookup::Ambiguous(
                    candidates
                        .into_iter()
                        .filter(|candidate| self.threads.contains_key(candidate))
                        .collect(),
                )
            }
            EntityResolution::Unknown => {}
        }
        // 注册表未命中：再按线程已见词面精确/折叠匹配（覆盖旧日志与未入册词面）。
        let folded = fold_surface(query);
        let mut matches: Vec<&EntityThread> = self
            .threads
            .values()
            .filter(|thread| {
                fold_surface(&thread.canonical) == folded
                    || thread
                        .known_surfaces
                        .iter()
                        .any(|surface| fold_surface(surface) == folded)
                    || thread
                        .surfaces
                        .iter()
                        .any(|surface| fold_surface(surface) == folded)
            })
            .collect();
        matches.sort_by(|left, right| left.canonical.cmp(&right.canonical));
        match matches.as_slice() {
            [] => ThreadLookup::Unknown,
            [thread] => ThreadLookup::Found(thread),
            _ => ThreadLookup::Ambiguous(
                matches
                    .iter()
                    .map(|thread| thread.canonical.clone())
                    .collect(),
            ),
        }
    }

    pub fn thread(&self, canonical: &str) -> Option<&EntityThread> {
        self.threads.get(canonical.trim())
    }

    /// 实体足迹：跨会话提及、回答、身份变更与当前图谱现状。
    pub fn render_footprint(&self, query: &str, facts: &[PrologFact]) -> String {
        let query = query.trim();
        if query.is_empty() {
            return "用法：实体足迹：<实体>，例如：实体足迹：张三".to_string();
        }
        match self.lookup(query) {
            ThreadLookup::Found(thread) => self.render_thread(thread, facts),
            ThreadLookup::Ambiguous(candidates) if candidates.is_empty() => format!(
                "「{}」在身份注册表里有多个候选，但编年史里都没有留下线索。",
                query
            ),
            ThreadLookup::Ambiguous(candidates) => format!(
                "「{}」可解析到多个身份：{}。请用规范名重试（不会静默折叠歧义）。",
                query,
                candidates.join(" / ")
            ),
            ThreadLookup::Unknown => {
                let mut output = format!("编年史里没有关于「{}」的实体线索。", query);
                let folded = fold_surface(query);
                let unresolved: Vec<&String> = self
                    .unresolved
                    .keys()
                    .filter(|surface| fold_surface(surface) == folded)
                    .collect();
                if !unresolved.is_empty() {
                    output.push_str(&format!(
                        " 但有一个未解析词面「{}」；可用「未解析提及」查看上下文。",
                        unresolved[0]
                    ));
                }
                output
            }
        }
    }

    /// 索引审计：活跃线程、静默身份、旧 URI 足迹与未解析提及。
    pub fn render_audit(&self) -> String {
        let mut lines = vec![
            "实体线索索引（编年史 × 身份注册表，只读投影）".to_string(),
            format!(
                "  编年史 {} 条；实体线索 {} 个；可归属提及 {} 条；未解析词面 {} 个。",
                self.entries,
                self.threads.len(),
                self.mentions,
                self.unresolved.len()
            ),
        ];
        let mut threads: Vec<&EntityThread> = self.threads.values().collect();
        threads.sort_by(|left, right| {
            right
                .last_seq
                .cmp(&left.last_seq)
                .then_with(|| left.canonical.cmp(&right.canonical))
        });
        if threads.is_empty() {
            lines.push("  还没有可归属的实体线索：先用自然语言聊到图谱里的实体。".to_string());
        } else {
            lines.push("  活跃线索（按最近轮次）：".to_string());
            for thread in threads.iter().take(12) {
                lines.push(format!(
                    "    {:>4}  {}（{} 条记录，seq {}–{}；{}）",
                    thread.last_seq.unwrap_or(0),
                    thread.canonical,
                    thread.total_events,
                    thread.first_seq.unwrap_or(0),
                    thread.last_seq.unwrap_or(0),
                    thread.uri
                ));
                if !thread.old_identities.is_empty() {
                    lines.push(format!(
                        "          旧身份已折叠：{}",
                        thread
                            .old_identities
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("、")
                    ));
                }
            }
            if threads.len() > 12 {
                lines.push(format!("    …共 {} 个线索。", threads.len()));
            }
        }

        let silent: Vec<&str> = self
            .registry
            .records()
            .filter(|record| {
                record.status == EntityStatus::Confirmed
                    && record.merged_into.is_none()
                    && !self.threads.contains_key(&record.canonical)
            })
            .map(|record| record.canonical.as_str())
            .collect();
        if !silent.is_empty() {
            let shown = silent
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join("、");
            let suffix = if silent.len() > 8 {
                format!(" 等 {} 个", silent.len())
            } else {
                String::new()
            };
            lines.push(format!(
                "  静默身份（已在图谱注册表、编年史未提及）：{}{}。",
                shown, suffix
            ));
        }

        let redirected: Vec<&EntityThread> = threads
            .iter()
            .copied()
            .filter(|thread| !thread.old_identities.is_empty())
            .collect();
        if !redirected.is_empty() {
            let shown = redirected
                .iter()
                .take(6)
                .map(|thread| {
                    format!(
                        "{}→{}",
                        thread
                            .old_identities
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("/"),
                        thread.canonical
                    )
                })
                .collect::<Vec<_>>()
                .join("；");
            lines.push(format!(
                "  旧 URI 足迹（读端折叠，原日志未改写）：{}。",
                shown
            ));
        }

        if !self.unresolved.is_empty() {
            lines.push("  未解析提及（保持歧义，不静默归并）：".to_string());
            for (surface, mention) in self.unresolved.iter().take(10) {
                let candidates = if mention.candidates.is_empty() {
                    "无候选".to_string()
                } else {
                    mention
                        .candidates
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" / ")
                };
                lines.push(format!(
                    "    {}（{} 次，seq {}；候选：{}）",
                    surface,
                    mention.seqs.len(),
                    mention
                        .seqs
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    candidates
                ));
            }
            if self.unresolved.len() > 10 {
                lines.push(format!("    …共 {} 个未解析词面。", self.unresolved.len()));
            }
        }
        lines.join("\n")
    }

    /// 未解析提及的独立审计视图。
    pub fn render_unresolved(&self) -> String {
        if self.unresolved.is_empty() {
            return "没有未解析的实体提及。".to_string();
        }
        let mut lines = vec![format!(
            "未解析实体提及：{} 个词面。",
            self.unresolved.len()
        )];
        for (surface, mention) in &self.unresolved {
            let sources = mention
                .sources
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("、");
            let candidates = if mention.candidates.is_empty() {
                "无候选".to_string()
            } else {
                mention
                    .candidates
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            lines.push(format!(
                "  {}（来源：{}；seq：{}；候选：{}）",
                surface,
                sources,
                mention
                    .seqs
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                candidates
            ));
        }
        lines.join("\n")
    }

    fn ingest(&mut self, entry: &ChronicleEntry, surfaces: &RegistrySurfaces) {
        let role = match entry.kind.as_str() {
            "reply" => ThreadRole::Reply,
            "turn" => ThreadRole::Turn,
            _ => ThreadRole::Other,
        };
        let question = entry
            .payload
            .get("user")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        let answer = entry
            .payload
            .get("answer")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        let focus = entry
            .payload
            .get("focus")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);

        let mut raw: Vec<RawMention> = Vec::new();
        for reference in entity_refs(entry) {
            let surface = reference.surface.trim();
            if surface.is_empty() {
                continue;
            }
            raw.push(RawMention {
                surface: surface.to_string(),
                entity: reference
                    .entity
                    .as_deref()
                    .map(str::trim)
                    .filter(|entity| !entity.is_empty())
                    .map(str::to_string),
                source: if reference.source.trim().is_empty() {
                    "ref".to_string()
                } else {
                    reference.source.trim().to_string()
                },
                late_bound: reference
                    .entity
                    .as_deref()
                    .map(str::trim)
                    .filter(|entity| !entity.is_empty())
                    .is_none(),
                seq: entry.seq,
            });
        }
        if let Some(focus) = focus.as_deref() {
            raw.push(RawMention {
                surface: focus.to_string(),
                entity: None,
                source: "focus".to_string(),
                late_bound: true,
                seq: entry.seq,
            });
        }
        // 旧日志没有 refs：按今天的已确认词面做确定性扫描，实现延迟绑定。
        if entry.kind != "command" {
            for text in [question.as_deref(), answer.as_deref()]
                .into_iter()
                .flatten()
            {
                for hit in surfaces.scan(text) {
                    raw.push(RawMention {
                        surface: hit.surface.to_string(),
                        entity: Some(hit.canonical.to_string()),
                        source: "text".to_string(),
                        late_bound: true,
                        seq: entry.seq,
                    });
                }
            }
        }

        let mut seen: BTreeSet<(String, Option<String>, String)> = BTreeSet::new();
        for mention in raw {
            if !seen.insert((
                mention.surface.clone(),
                mention.entity.clone(),
                mention.source.clone(),
            )) {
                continue;
            }
            let (canonical, redirected_from) = match surfaces.resolve(&self.registry, &mention) {
                MentionOutcome::Resolved {
                    canonical,
                    redirected_from,
                } => (canonical, redirected_from),
                MentionOutcome::Ambiguous(candidates) => {
                    let entry = self.unresolved.entry(mention.surface).or_default();
                    entry.seqs.insert(mention.seq);
                    entry.sources.insert(mention.source);
                    entry.candidates.extend(candidates);
                    continue;
                }
                MentionOutcome::Unknown => {
                    let entry = self.unresolved.entry(mention.surface).or_default();
                    entry.seqs.insert(mention.seq);
                    entry.sources.insert(mention.source);
                    continue;
                }
            };
            self.mentions += 1;
            let fresh = self.new_thread(&canonical);
            let thread = self.threads.entry(canonical.clone()).or_insert(fresh);
            let event = match thread.events.last_mut() {
                Some(event) if event.seq == entry.seq => event,
                _ => {
                    thread.events.push(ThreadEvent {
                        seq: entry.seq,
                        kind: entry.kind.clone(),
                        role,
                        question: question.clone(),
                        answer: answer.clone(),
                        focus: focus.clone(),
                        kb_revision: entry.kb_revision,
                        mentions: Vec::new(),
                    });
                    thread.events.last_mut().expect("刚插入")
                }
            };
            match event
                .mentions
                .iter_mut()
                .find(|existing| existing.surface == mention.surface)
            {
                Some(existing) => {
                    existing.sources.insert(mention.source);
                    existing.late_bound &= mention.late_bound;
                    if existing.stored_entity.is_none() {
                        existing.stored_entity = mention.entity.clone();
                    }
                    if existing.redirected_from.is_none() {
                        existing.redirected_from = redirected_from.clone();
                    }
                }
                None => event.mentions.push(ThreadMention {
                    surface: mention.surface,
                    stored_entity: mention.entity,
                    sources: [mention.source].into_iter().collect(),
                    late_bound: mention.late_bound,
                    redirected_from,
                }),
            }
        }
    }

    fn new_thread(&self, canonical: &str) -> EntityThread {
        let record = self.registry.record(canonical);
        EntityThread {
            canonical: canonical.to_string(),
            uri: entity_uri(DEFAULT_BASE_NS, canonical),
            types: record
                .map(|record| record.types.clone())
                .unwrap_or_default(),
            observed: record.map(|record| record.observed).unwrap_or(false),
            known_surfaces: BTreeSet::new(),
            surfaces: BTreeSet::new(),
            old_identities: BTreeSet::new(),
            events: Vec::new(),
            total_events: 0,
            omitted_events: 0,
            first_seq: None,
            last_seq: None,
            question_events: 0,
            reply_events: 0,
            late_bound_events: 0,
        }
    }

    fn finish(&mut self, surfaces: &RegistrySurfaces) {
        for (canonical, thread) in self.threads.iter_mut() {
            thread.known_surfaces = surfaces
                .by_canonical
                .get(canonical)
                .cloned()
                .unwrap_or_default();
            thread.surfaces = thread
                .events
                .iter()
                .flat_map(|event| event.mentions.iter())
                .map(|mention| mention.surface.clone())
                .collect();
            for event in &thread.events {
                for mention in &event.mentions {
                    if let Some(stored) = &mention.stored_entity {
                        if stored != canonical {
                            thread.old_identities.insert(stored.clone());
                        }
                    }
                    if let Some(from) = &mention.redirected_from {
                        if from != canonical {
                            thread.old_identities.insert(from.clone());
                        }
                    }
                }
            }
            thread.events.sort_by_key(|event| event.seq);
            thread.total_events = thread.events.len();
            thread.first_seq = thread.events.first().map(|event| event.seq);
            thread.last_seq = thread.events.last().map(|event| event.seq);
            thread.question_events = thread
                .events
                .iter()
                .filter(|event| event.role == ThreadRole::Turn)
                .count();
            thread.reply_events = thread
                .events
                .iter()
                .filter(|event| event.role == ThreadRole::Reply)
                .count();
            thread.late_bound_events = thread
                .events
                .iter()
                .filter(|event| event.mentions.iter().any(|mention| mention.late_bound))
                .count();
            if thread.events.len() > MAX_EVENTS_PER_THREAD {
                let drop = thread.events.len() - MAX_EVENTS_PER_THREAD;
                thread.events.drain(0..drop);
                thread.omitted_events = drop;
            }
        }
    }

    fn render_thread(&self, thread: &EntityThread, facts: &[PrologFact]) -> String {
        let mut lines = vec![format!("实体足迹：{}", thread.canonical)];
        lines.push(format!("  URI：{}", thread.uri));
        let types = if thread.types.is_empty() {
            "类型未知".to_string()
        } else {
            format!(
                "类型 {}",
                thread.types.iter().cloned().collect::<Vec<_>>().join("、")
            )
        };
        lines.push(format!(
            "  注册表：{}；图谱观测：{}；可解析词面 {} 个。",
            types,
            if thread.observed { "是" } else { "否" },
            thread.known_surfaces.len()
        ));
        let mut aliases: Vec<String> = thread
            .known_surfaces
            .iter()
            .filter(|surface| *surface != &thread.canonical)
            .cloned()
            .collect();
        aliases.sort();
        if !aliases.is_empty() {
            lines.push(format!("  别名/曾用名：{}", aliases.join("、")));
        }
        if !thread.old_identities.is_empty() {
            lines.push(format!(
                "  读端折叠：{} → {}（旧日志未改写，哈希链保持完整）",
                thread
                    .old_identities
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("、"),
                thread.canonical
            ));
        }
        let mut seen: Vec<String> = thread
            .surfaces
            .iter()
            .filter(|surface| !thread.known_surfaces.contains(*surface))
            .cloned()
            .collect();
        seen.sort();
        if !seen.is_empty() {
            lines.push(format!("  对话中出现的其它词面：{}", seen.join("、")));
        }
        let recent_cap = 8usize;
        let span = match (thread.first_seq, thread.last_seq) {
            (Some(first), Some(last)) => format!("seq {}–{}", first, last),
            _ => "无记录".to_string(),
        };
        lines.push(format!(
            "  对话：{} 条记录（提问 {}，回答 {}，迟到绑定 {}），{}。",
            thread.total_events,
            thread.question_events,
            thread.reply_events,
            thread.late_bound_events,
            span
        ));
        if thread.events.is_empty() {
            lines.push("  最近记录：无（线程只剩身份变更足迹）。".to_string());
        } else {
            lines.push("  最近记录（由新到旧）：".to_string());
            for event in thread.events.iter().rev().take(recent_cap) {
                lines.push(format!("    {}", render_event_line(thread, event)));
            }
            let omitted = thread.omitted_events + thread.events.len().saturating_sub(recent_cap);
            if omitted > 0 {
                lines.push(format!(
                    "    …另有 {} 条较早记录已折叠（完整历史仍可「聊过」检索）。",
                    omitted
                ));
            }
        }
        let names: BTreeSet<String> = thread
            .surfaces
            .iter()
            .chain(thread.known_surfaces.iter())
            .chain(std::iter::once(&thread.canonical))
            .cloned()
            .collect();
        let spoken = thread
            .events
            .iter()
            .flat_map(|event| [event.question.as_deref(), event.answer.as_deref()])
            .flatten()
            .collect::<Vec<_>>()
            .join("\n");
        let mut related: Vec<(String, bool)> = facts
            .iter()
            .filter(|fact| fact_touches(fact, &names))
            .map(|fact| (render_fact(fact), fact_surfaced(fact, &spoken)))
            .collect();
        related.sort();
        if related.is_empty() {
            lines.push("  图谱现状：当前快照中没有与该实体相关的事实。".to_string());
        } else {
            let unmentioned = related.iter().filter(|(_, surfaced)| !surfaced).count();
            let note = if unmentioned > 0 {
                format!("（其中 {} 条未在对话文本中出现）", unmentioned)
            } else {
                "（均已在对话文本中出现）".to_string()
            };
            lines.push(format!("  图谱现状：{} 条相关事实{}。", related.len(), note));
            for (fact, surfaced) in related.iter().take(6) {
                lines.push(format!(
                    "    {}{}",
                    fact,
                    if *surfaced { "" } else { "（对话未提及）" }
                ));
            }
            if related.len() > 6 {
                lines.push(format!("    …另有 {} 条。", related.len() - 6));
            }
        }
        lines.join("\n")
    }
}

struct RawMention {
    surface: String,
    entity: Option<String>,
    source: String,
    late_bound: bool,
    seq: u64,
}

enum MentionOutcome {
    Resolved {
        canonical: String,
        redirected_from: Option<String>,
    },
    Ambiguous(Vec<String>),
    Unknown,
}

fn entity_refs(entry: &ChronicleEntry) -> Vec<EntityRef> {
    let Some(items) = entry.payload.get("refs").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| match item {
            Value::String(surface) => Some(EntityRef {
                surface: surface.clone(),
                entity: None,
                source: "ref".to_string(),
            }),
            Value::Object(object) => {
                let surface = object
                    .get("surface")
                    .or_else(|| object.get("mention"))
                    .or_else(|| object.get("name"))
                    .or_else(|| object.get("entity"))
                    .and_then(Value::as_str)?
                    .trim()
                    .to_string();
                if surface.is_empty() {
                    return None;
                }
                let entity = object
                    .get("entity")
                    .or_else(|| object.get("canonical"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|entity| !entity.is_empty())
                    .map(str::to_string);
                let source = object
                    .get("source")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|source| !source.is_empty())
                    .unwrap_or("ref")
                    .to_string();
                Some(EntityRef {
                    surface,
                    entity,
                    source,
                })
            }
            _ => None,
        })
        .collect()
}

/// 单条线程事件的人类可读行（供实体足迹与别名感知检索共用）。
pub fn render_event_line(thread: &EntityThread, event: &ThreadEvent) -> String {
    let mut text = format!("第 {} 条 [{}]", event.seq, event.role.label());
    if event.role == ThreadRole::Reply {
        if let Some(question) = &event.question {
            text.push_str(&format!(" 问：{}", excerpt(question, 32)));
        }
        if let Some(answer) = &event.answer {
            text.push_str(&format!(" 答：{}", excerpt(answer, 56)));
        }
    } else if let Some(question) = &event.question {
        text.push_str(&format!(" {}", excerpt(question, 72)));
    } else if let Some(answer) = &event.answer {
        text.push_str(&format!(" {}", excerpt(answer, 72)));
    }
    let mentions: Vec<String> = event
        .mentions
        .iter()
        .map(|mention| match &mention.redirected_from {
            Some(from) if from != &thread.canonical => {
                format!(
                    "{}（旧身份 {}→{}）",
                    mention.surface, from, thread.canonical
                )
            }
            _ => mention.surface.clone(),
        })
        .collect();
    if !mentions.is_empty() {
        text.push_str(&format!("（提及：{}）", mentions.join("、")));
    }
    text
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let text = text.replace(['\n', '\r'], " ");
    let text = text.trim();
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push('…');
    }
    out
}

/// 事实是否在对话文本中被明确提到（宽松但只作「未出现」标注，不宣称真假）。
fn fact_surfaced(fact: &PrologFact, text: &str) -> bool {
    match fact {
        PrologFact::Entity { properties, .. } => properties
            .iter()
            .any(|(key, value)| text.contains(key) || text.contains(value)),
        PrologFact::Relation {
            from, rel_type, to, ..
        } => text.contains(rel_type) || (text.contains(from) && text.contains(to)),
    }
}

fn fact_touches(fact: &PrologFact, names: &BTreeSet<String>) -> bool {
    match fact {
        PrologFact::Entity {
            name, properties, ..
        } => names.contains(name) || properties.values().any(|value| names.contains(value)),
        PrologFact::Relation { from, to, .. } => names.contains(from) || names.contains(to),
    }
}

fn render_fact(fact: &PrologFact) -> String {
    match fact {
        PrologFact::Entity {
            name,
            entity_type,
            properties,
        } => {
            let mut fields: Vec<String> = properties
                .iter()
                .map(|(key, value)| format!("{}={}", key, excerpt(value, 40)))
                .collect();
            fields.sort();
            if fields.is_empty() {
                format!("{} 是 {}", name, entity_type)
            } else {
                format!("{}（{}）：{}", name, entity_type, fields.join("；"))
            }
        }
        PrologFact::Relation { from, rel_type, to } => {
            format!("{} --{}--> {}", from, rel_type, to)
        }
    }
}

/// 注册表词面表：用于读端扫描与「可解析词面」展示。
struct RegistrySurfaces {
    by_first: BTreeMap<char, Vec<SurfaceHit>>,
    by_canonical: BTreeMap<String, BTreeSet<String>>,
}

struct SurfaceHit {
    surface: String,
    canonical: String,
    chars: Vec<char>,
}

impl RegistrySurfaces {
    fn build(registry: &EntityRegistry) -> Self {
        let mut pairs: BTreeSet<(String, String, bool)> = BTreeSet::new();
        for record in registry.records() {
            let target = registry
                .resolve(&record.canonical)
                .canonical()
                .map(str::to_string);
            if let Some(target) = target {
                let is_canonical = true;
                pairs.insert((record.canonical.clone(), target, is_canonical));
            }
            for alias in record.aliases.values() {
                if alias.status != EntityStatus::Confirmed {
                    continue;
                }
                for surface in std::iter::once(&alias.surface).chain(alias.variants.iter()) {
                    if let Some(target) = registry.resolve(surface).canonical() {
                        pairs.insert((surface.clone(), target.to_string(), false));
                    }
                }
            }
        }
        for merge in registry.merges() {
            if let Some(target) = registry.resolve(&merge.from).canonical() {
                pairs.insert((merge.from.clone(), target.to_string(), true));
            }
            if let Some(target) = registry.resolve(&merge.into).canonical() {
                pairs.insert((merge.into.clone(), target.to_string(), true));
            }
        }
        let mut by_first: BTreeMap<char, Vec<SurfaceHit>> = BTreeMap::new();
        let mut by_canonical: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for (surface, canonical, is_canonical) in pairs {
            if surface.trim().is_empty() || !seen.insert((surface.clone(), canonical.clone())) {
                continue;
            }
            let chars: Vec<char> = surface.chars().collect();
            if !is_canonical && chars.len() < 2 {
                continue;
            }
            by_canonical
                .entry(canonical.clone())
                .or_default()
                .insert(surface.clone());
            if let Some(first) = chars.first().copied() {
                by_first.entry(first).or_default().push(SurfaceHit {
                    surface,
                    canonical,
                    chars,
                });
            }
        }
        for hits in by_first.values_mut() {
            hits.sort_by(|left, right| {
                right
                    .chars
                    .len()
                    .cmp(&left.chars.len())
                    .then_with(|| left.surface.cmp(&right.surface))
            });
        }
        RegistrySurfaces {
            by_first,
            by_canonical,
        }
    }

    fn scan<'a>(&'a self, text: &str) -> Vec<&'a SurfaceHit> {
        let chars: Vec<char> = text.chars().collect();
        let mut hits = Vec::new();
        let mut index = 0usize;
        while index < chars.len() {
            let Some(candidates) = self.by_first.get(&chars[index]) else {
                index += 1;
                continue;
            };
            let mut matched: Option<&SurfaceHit> = None;
            for candidate in candidates {
                let len = candidate.chars.len();
                if index + len <= chars.len() && chars[index..index + len] == candidate.chars[..] {
                    matched = Some(candidate);
                    break;
                }
            }
            match matched {
                Some(hit) => {
                    index += hit.chars.len();
                    hits.push(hit);
                }
                None => index += 1,
            }
        }
        hits
    }

    fn resolve(&self, registry: &EntityRegistry, mention: &RawMention) -> MentionOutcome {
        let mut ambiguous: Vec<String> = Vec::new();
        if let Some(entity) = mention.entity.as_deref() {
            match registry.resolve(entity) {
                EntityResolution::Canonical(canonical) => {
                    return MentionOutcome::Resolved {
                        canonical,
                        redirected_from: None,
                    }
                }
                EntityResolution::Redirected { from, to } => {
                    return MentionOutcome::Resolved {
                        canonical: to,
                        redirected_from: Some(from),
                    }
                }
                EntityResolution::Ambiguous(candidates) => ambiguous.extend(candidates),
                EntityResolution::Unknown => {}
            }
        }
        match registry.resolve(&mention.surface) {
            EntityResolution::Canonical(canonical) => MentionOutcome::Resolved {
                canonical,
                redirected_from: None,
            },
            EntityResolution::Redirected { from, to } => MentionOutcome::Resolved {
                canonical: to,
                redirected_from: Some(from),
            },
            EntityResolution::Ambiguous(candidates) => {
                ambiguous.extend(candidates);
                ambiguous.sort();
                ambiguous.dedup();
                MentionOutcome::Ambiguous(ambiguous)
            }
            EntityResolution::Unknown => {
                if ambiguous.is_empty() {
                    MentionOutcome::Unknown
                } else {
                    MentionOutcome::Ambiguous(ambiguous)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::FsyncPolicy;
    use tempfile::TempDir;

    fn registry_with(names: &[(&str, &[&str], &[&str])]) -> EntityRegistry {
        let mut registry = EntityRegistry::new();
        let types: BTreeSet<String> = BTreeSet::new();
        for (canonical, aliases, names_literals) in names {
            let literals: Vec<String> =
                names_literals.iter().map(|name| name.to_string()).collect();
            registry.observe_kb_entity(canonical, &types, &literals, "测试图谱");
            for alias in *aliases {
                registry.confirm_alias(alias, canonical);
            }
        }
        registry
    }

    fn chronicle(dir: &TempDir) -> (Chronicle, String) {
        let path = dir.path().join("entity-thread.chronicle.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (chronicle, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        (chronicle, path_text)
    }

    #[test]
    fn refs_fold_to_current_identity_without_rewriting_log() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, path_text) = chronicle(&dir);
        let refs = vec![EntityRef {
            surface: "小李".to_string(),
            entity: Some("李四".to_string()),
            source: "link".to_string(),
        }];
        chronicle
            .append_turn_with_refs("小李是谁", "query", None, 0, &refs)
            .unwrap();
        chronicle
            .append_reply("李四是张三的曾用名。", &[], &[])
            .unwrap();
        let bytes_before = std::fs::read(&path_text).unwrap();

        let mut registry = registry_with(&[("李四", &["小李"], &[]), ("张三", &[], &[])]);
        assert!(registry.merge("李四", "张三").is_ok());
        let (reloaded, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        let digest = EntityThreadDigest::build(&reloaded, &registry);

        let thread = digest.thread("张三").expect("旧身份应折叠到当前身份");
        assert_eq!(thread.total_events, 2, "{:?}", thread.events);
        assert_eq!(thread.first_seq, Some(1));
        assert_eq!(thread.last_seq, Some(2));
        assert!(thread.surfaces.contains("小李"));
        assert!(thread.old_identities.contains("李四"));
        assert_eq!(thread.uri, "http://example.org/kg/entity/张三");
        assert_eq!(
            std::fs::read(&path_text).unwrap(),
            bytes_before,
            "投影不得改写日志字节"
        );
        assert!(digest.thread("李四").is_none(), "线程只按最终身份聚合");
    }

    #[test]
    fn old_log_late_binds_after_alias_confirmation() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, path_text) = chronicle(&dir);
        chronicle
            .append_turn("小李后来怎么样了", "query", None, 0)
            .unwrap();
        chronicle
            .append_reply("小李最近去了杭州。", &[], &[])
            .unwrap();

        let empty = EntityRegistry::new();
        let before = EntityThreadDigest::build(&chronicle, &empty);
        assert!(before.threads.is_empty(), "未登记时不得猜测身份");

        let registry = registry_with(&[("李四", &["小李"], &[])]);
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        let thread = digest.thread("李四").expect("确认别名后旧词面应回链");
        assert_eq!(thread.total_events, 2);
        assert!(thread.late_bound_events >= 2, "{:?}", thread.events);
        let event = &thread.events[0];
        assert!(event.mentions.iter().any(|mention| {
            mention.surface == "小李" && mention.sources.contains("text") && mention.late_bound
        }));
        assert!(digest.render_footprint("小李", &[]).contains("李四"));
        assert!(Chronicle::open_verified(&path_text).is_ok());
    }

    #[test]
    fn lookup_is_alias_aware_and_keeps_ambiguity_visible() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        chronicle
            .append_turn_with_refs(
                "老张是谁",
                "query",
                None,
                0,
                &[EntityRef {
                    surface: "老张".to_string(),
                    entity: Some("张三".to_string()),
                    source: "link".to_string(),
                }],
            )
            .unwrap();
        let mut registry = registry_with(&[("张三", &["老张", "小张"], &[])]);
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        assert!(matches!(digest.lookup("小张"), ThreadLookup::Found(_)));
        assert!(matches!(
            digest.lookup("老张"),
            ThreadLookup::Found(thread) if thread.canonical == "张三"
        ));

        registry.confirm_alias("老张", "李四");
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        assert!(matches!(digest.lookup("老张"), ThreadLookup::Ambiguous(_)));
        let report = digest.render_footprint("老张", &[]);
        assert!(report.contains("多个身份"), "{}", report);
    }

    #[test]
    fn unresolved_refs_are_reported_not_collapsed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        chronicle
            .append_turn_with_refs(
                "小赵是谁",
                "query",
                Some("小赵"),
                0,
                &[EntityRef {
                    surface: "小赵".to_string(),
                    entity: Some("赵六".to_string()),
                    source: "link".to_string(),
                }],
            )
            .unwrap();
        let registry = EntityRegistry::new();
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        assert!(digest.threads.is_empty());
        let mention = digest.unresolved.get("小赵").expect("未解析词面必须保留");
        assert!(mention.seqs.contains(&1));
        assert!(mention.sources.contains("focus"));
        let report = digest.render_unresolved();
        assert!(report.contains("小赵"), "{}", report);
    }

    #[test]
    fn audit_separates_silent_identities_and_threads() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        chronicle
            .append_turn_with_refs(
                "张三是谁",
                "query",
                None,
                0,
                &[EntityRef {
                    surface: "张三".to_string(),
                    entity: Some("张三".to_string()),
                    source: "link".to_string(),
                }],
            )
            .unwrap();
        let registry = registry_with(&[("张三", &[], &[]), ("王五", &[], &[])]);
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        let report = digest.render_audit();
        assert!(report.contains("实体线索 1 个"), "{}", report);
        assert!(report.contains("静默身份"), "{}", report);
        assert!(report.contains("王五"), "{}", report);
        assert!(!report.contains("旧 URI 足迹"), "没有合并时不应伪造旧 URI");
    }

    #[test]
    fn footprint_marks_graph_facts_absent_from_dialogue_text() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let refs = vec![EntityRef {
            surface: "张三".to_string(),
            entity: Some("张三".to_string()),
            source: "link".to_string(),
        }];
        chronicle
            .append_turn_with_refs("张三的身高是多少", "query", None, 0, &refs)
            .unwrap();
        chronicle
            .append_reply("张三的身高是 226 厘米。", &[], &[])
            .unwrap();
        let registry = registry_with(&[("张三", &[], &[])]);
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        let mut properties = std::collections::HashMap::new();
        properties.insert("身高".to_string(), "226 厘米".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "Person".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
        ];
        let report = digest.render_footprint("张三", &facts);
        assert!(report.contains("226 厘米"), "{}", report);
        assert!(
            report.contains("师傅") && report.contains("对话未提及"),
            "{}",
            report
        );
        assert!(report.contains("1 条未在对话文本中出现"), "{}", report);
    }

    #[test]
    fn thread_event_cap_is_disclosed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let refs = vec![EntityRef {
            surface: "张三".to_string(),
            entity: Some("张三".to_string()),
            source: "link".to_string(),
        }];
        for turn in 0..(MAX_EVENTS_PER_THREAD + 3) {
            chronicle
                .append_turn_with_refs(&format!("第 {} 问", turn), "query", None, turn, &refs)
                .unwrap();
        }
        let registry = registry_with(&[("张三", &[], &[])]);
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        let thread = digest.thread("张三").unwrap();
        assert_eq!(thread.total_events, MAX_EVENTS_PER_THREAD + 3);
        assert_eq!(thread.events.len(), MAX_EVENTS_PER_THREAD);
        assert_eq!(thread.omitted_events, 3);
        assert_eq!(thread.last_seq, Some((MAX_EVENTS_PER_THREAD + 3) as u64));
        assert!(digest.render_footprint("张三", &[]).contains("已折叠"));
    }
}

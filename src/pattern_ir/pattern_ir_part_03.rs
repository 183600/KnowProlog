/// 模式库清单 + 静态体检的渲染。
pub fn render_library_report(library: &PatternLibrary) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let taught = library.count_with_source_prefix("taught#");
    let _ = writeln!(
        out,
        "模式库：{} 条（可路由查询 {} + NLU 回退 {} + NLU 意图 {} + 教学 {}）",
        library.len(),
        library.route_count(),
        library.fallback_count(),
        library.nlu_count(),
        taught
    );
    let _ = writeln!(
        out,
        " {:<9} {:>4} {:<22} {:<12} {}",
        "#id", "优先", "路由", "名称", "模式"
    );
    for entry in library.entries() {
        let guard = entry
            .guard
            .as_ref()
            .map(|guard| format!("  [守卫：{}]", guard.render()))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            " {:<9} {:>4} {:<22} {:<12} {}{}",
            entry.short_id(),
            entry.priority,
            entry.query_type,
            entry.name,
            entry.render(),
            guard
        );
    }
    let findings = lint(library);
    if findings.is_empty() {
        let _ = writeln!(out, "静态体检：无重复、遮蔽或冲突。");
    } else {
        let _ = writeln!(out, "静态体检：{} 项", findings.len());
        for finding in &findings {
            let _ = writeln!(out, " - {}", finding.render());
        }
    }
    out
}
/// 模式关系报告：泛化/冗余/冲突的图边清单 + 一条可复现的图体检查询 + 端点 id。
///
/// 与 [`render_library_report`] 展示同一份体检结果，但以「模式命名图里的边」
/// 为主视角：每条关系都是可 SPARQL 查询的数据，末尾附上计数查询与端点 id，
/// 便于把结论复制到「SPARQL：…」里复核。
pub fn render_relation_report(library: &PatternLibrary) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let edges = relation_edges(library.entries());
    let counts = relation_counts(&edges);
    let generalizes = counts.get("泛化").copied().unwrap_or(0);
    let redundant = counts.get("冗余").copied().unwrap_or(0);
    let conflicts = counts.get("冲突").copied().unwrap_or(0);
    let _ = writeln!(
        out,
        "模式关系图 {}：{} 条边（泛化 {} + 冗余 {} + 冲突 {}）",
        PATTERNS_GRAPH,
        edges.len(),
        generalizes,
        redundant,
        conflicts
    );
    if edges.is_empty() {
        let _ = writeln!(out, "（无关系边：模式间无包含、重复或冲突）");
    } else {
        let _ = writeln!(
            out,
            " {:<6} {:<9} {:<9} {}",
            "关系", "主语#id", "宾语#id", "端点"
        );
        for edge in &edges {
            let _ = writeln!(
                out,
                " {:<6} {:<9} {:<9} {}",
                edge.kind.label(),
                &edge.subject_id.chars().take(8).collect::<String>(),
                &edge.object_id.chars().take(8).collect::<String>(),
                edge.render()
            );
        }
    }
    let _ = writeln!(out, "图体检查询（直接执行即得各关系计数）：");
    let _ = writeln!(out, "  {}", relation_count_sparql());
    out
}
/// 匹配报告渲染：胜者带绑定与区间，命中未胜带差额，失败带首个失配。
pub fn render_match_report(input: &str, report: &MatchReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "模式匹配「{}」：", input);

    match &report.winner {
        Some(winner) => {
            let _ = writeln!(
                out,
                " 胜者 {}（#{}）优先 {} → 路由 {}，覆盖 {}，特化度 {:.2}",
                winner.entry.name,
                winner.entry.short_id(),
                winner.entry.priority,
                winner.entry.query_type,
                winner.span,
                winner.entry.specificity()
            );
            if let Some(guard) = &winner.entry.guard {
                let _ = writeln!(out, " 上下文守卫：{}", guard.render());
            }
            let bindings: Vec<String> = winner
                .bindings
                .iter()
                .filter(|(_, term)| term.is_bound())
                .map(|(name, term)| match entity_span_for(&winner, name) {
                    Some(span) => format!("{}={}{}", name, term, span),
                    None => format!("{}={}", name, term),
                })
                .collect();
            if bindings.is_empty() {
                let _ = writeln!(out, " 绑定：（无槽位）");
            } else {
                let _ = writeln!(out, " 绑定：{}", bindings.join(" "));
            }
        }
        None => {
            let _ = writeln!(out, " 无可路由命中（教学模式或配置模式均未匹配）。");
        }
    }

    let others: Vec<&CandidateOutcome> = report
        .matched
        .iter()
        .filter(|outcome| report.winner.as_ref().map(|w| w.entry.id()) != Some(outcome.id.clone()))
        .collect();
    if !others.is_empty() {
        let _ = writeln!(out, " 命中未胜（{}）：", others.len());
        for outcome in others.iter().take(5) {
            if let CandidateStatus::Matched {
                rank,
                specificity,
                span,
                ..
            } = &outcome.status
            {
                let _ = writeln!(
                    out,
                    "  {}. {}（#{}）优先 {} → {}，{}，特化度 {:.2}{}",
                    rank + 1,
                    outcome.name,
                    &outcome.id.chars().take(8).collect::<String>(),
                    outcome.priority,
                    outcome.query_type,
                    span,
                    specificity,
                    if outcome.route {
                        ""
                    } else {
                        "（意图展示，不路由）"
                    }
                );
            }
        }
    }

    if !report.failed.is_empty() {
        let _ = writeln!(out, " 未命中（{}，前 3 条原因）：", report.failed.len());
        for outcome in report.failed.iter().take(3) {
            if let CandidateStatus::Failed { reason } = &outcome.status {
                let _ = writeln!(
                    out,
                    "  - {}（#{}）：{}",
                    outcome.name,
                    &outcome.id.chars().take(8).collect::<String>(),
                    reason
                );
            }
        }
    }

    for (a, b, span) in report.conflicts() {
        let _ = writeln!(
            out,
            " 冲突：「{}」与「{}」同覆盖 {} 但路由不同。",
            a, b, span
        );
    }
    out
}
/// 绑定名 → 实体区间（展示绑定的区间证据）。
fn entity_span_for<'a>(m: &'a PatternMatch, name: &str) -> Option<&'a Span> {
    let index: usize = name.strip_prefix("entity.")?.parse().ok()?;
    m.entity_spans.get(index.checked_sub(1)?)
}
/// 模式经验台账的文本报告（总量 + 高频 + 最近）。
pub fn render_episode_report(log: &PatternEpisodeLog) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let total_support: u32 = log.entries().iter().map(|episode| episode.count).sum();
    let _ = writeln!(
        out,
        "模式经验：{} 条（命中 {} + 未命中 {}），容量 {}，累计支持度 {}。",
        log.len(),
        log.matched_count(),
        log.unmatched_count(),
        log.capacity(),
        total_support
    );
    if log.is_empty() {
        let _ = writeln!(out, "（还没有经验：成功或失败的解析都会留下可检索的记录）");
        return out;
    }
    let mut frequent: Vec<&PatternEpisode> = log.entries().iter().collect();
    frequent.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| b.last_turn.cmp(&a.last_turn))
            .then_with(|| a.id().cmp(&b.id()))
    });
    let _ = writeln!(out, " 高频经验：");
    for episode in frequent.iter().take(5) {
        let target = episode
            .winner_name
            .as_deref()
            .map(|name| format!("{} → {}", name, episode.query_type.as_deref().unwrap_or("")))
            .unwrap_or_else(|| "（无可路由命中）".to_string());
        let context = episode
            .context_note()
            .map(|note| format!("，{}", note))
            .unwrap_or_default();
        let feedback = if episode.acceptances > 0 || episode.rejections > 0 {
            format!("，确认×{} 否定×{}", episode.acceptances, episode.rejections)
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  [{}] {} ×{} 第{}轮{}{}：{}",
            episode.short_id(),
            target,
            episode.count,
            episode.last_turn,
            context,
            feedback,
            episode.input
        );
        if let Some(frontier) = &episode.frontier {
            let _ = writeln!(out, "      ↳ {}", frontier.render());
        }
    }
    let mut recent: Vec<&PatternEpisode> = log.entries().iter().collect();
    recent.sort_by(|a, b| {
        b.last_turn
            .cmp(&a.last_turn)
            .then_with(|| a.id().cmp(&b.id()))
    });
    let _ = writeln!(out, " 最近经验：");
    for episode in recent.iter().take(5) {
        let context = episode
            .context_note()
            .map(|note| format!("，{}", note))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  [{}] 第{}轮 {}{}：{}",
            episode.outcome.label(),
            episode.last_turn,
            episode.short_id(),
            context,
            episode.input
        );
    }
    let _ = writeln!(
        out,
        "（可用「忘经验：<短 id 或原句>」删除单条经验，撤销误确认/误否定。）"
    );
    out
}
/// 候选骨架的文本报告：相似经验 + 候选模式 + 显式激活方式。
pub fn render_case_suggestion(input: &str, suggestion: &CaseSuggestion) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let episode = &suggestion.episode;
    let _ = writeln!(
        out,
        "模式联想「{}」：相似经验 {}（{:.2}，共享：{}）",
        input,
        episode.input,
        suggestion.score,
        if suggestion.shared.is_empty() {
            "无".to_string()
        } else {
            suggestion.shared.join("、")
        }
    );
    let _ = writeln!(
        out,
        " 经验命中：{} → {}（第{}轮，×{}）",
        episode.winner_name.as_deref().unwrap_or("未知模式"),
        episode.query_type.as_deref().unwrap_or("未知路由"),
        episode.last_turn,
        episode.count
    );
    if episode.acceptances > 0 || episode.rejections > 0 {
        let _ = writeln!(
            out,
            " 用户反馈：确认 ×{}，否定 ×{}（净否定的骨架不再主动推荐）",
            episode.acceptances, episode.rejections
        );
    }
    if let Some(context) = episode.context_note() {
        let _ = writeln!(out, " 发生上下文：{}", context);
    }
    let _ = writeln!(out, " 候选模式：{}", suggestion.pattern.render());
    let guard = suggestion
        .pattern
        .guard
        .as_ref()
        .map(|guard| format!("，守卫 {}", guard.render()))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        " 候选路由：{}（来源 {}，未激活{}）",
        suggestion.pattern.query_type, suggestion.pattern.source, guard
    );
    let _ = writeln!(
        out,
        " 确认方式：确认联想：<句子>（或 教模式：{} | {} | {} | {}）",
        suggestion.pattern.name,
        suggestion.pattern.render(),
        suggestion.pattern.query_type,
        suggestion.pattern.priority
    );
    let _ = writeln!(
        out,
        " 边界：候选只用于确认，不直接回答；确认后才进入可路由模式库。"
    );
    let _ = writeln!(out, " 口语确认：直接回答「是」或「不是」即可。");
    out
}

// ---------------------------------------------------------------------------
// 静态体检：重复 / 遮蔽 / 冲突
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintKind {
    /// 同一模式语言（IR）+ 同路由，两处定义。
    Redundant,
    /// 同一模式语言（IR）但路由不同。
    Conflict,
    /// 高位模式覆盖低位模式的语言，低位永不胜出。
    Shadowed,
}
/// 单要素包含：`a` 的语言 ⊇ `b` 的语言（保守近似，按位对齐）。
fn elem_subsumes(a: &Elem, b: &Elem) -> bool {
    match (a, b) {
        (Elem::Any, _) => true,
        // 「任意」只被「任意」覆盖
        (_, Elem::Any) => false,
        (Elem::Optional(inner), _) => elem_subsumes(inner, b),
        (Elem::Pos { pos: pa }, Elem::Pos { pos: pb }) => pa == pb,
        // 词性/词面接受集互不可判，保守不覆盖
        (Elem::Pos { .. }, Elem::Word { .. }) | (Elem::Pos { .. }, Elem::Slot { .. }) => false,
        (Elem::Word { surface: wa }, Elem::Word { surface: wb }) => wa == wb,
        (Elem::Word { .. }, Elem::Pos { .. }) | (Elem::Word { .. }, Elem::Slot { .. }) => false,
        (Elem::Slot { sort: sa, .. }, Elem::Pos { pos }) => sort_pos_set(sa)
            .map(|set| set.contains(&pos.as_str()))
            .unwrap_or(false),
        (Elem::Slot { sort: sa, .. }, Elem::Slot { sort: sb, .. }) => sa == sb,
        (Elem::Slot { .. }, Elem::Word { .. }) => false,
        // b 可缺席而 a 不可，单步对齐下不构成覆盖
        (_, Elem::Optional(_)) => false,
    }
}
/// 槽位排序 → 词性集（保守估计；供包含判定）。
fn sort_pos_set(sort: &str) -> Option<&'static [&'static str]> {
    match sort {
        "城市" | "地点" => Some(&["ns", "n"]),
        "日期" | "时间" => Some(&["m"]),
        "人名" => Some(&["nr"]),
        "数量" => Some(&["m"]),
        "实体" => Some(&["nr", "ns", "n"]),
        "属性" => Some(&["n", "nz", "vn"]),
        "*" | "" => Some(&["r", "n", "nr", "ns", "m", "q", "v", "u", "f", "a", "d"]),
        _ => None,
    }
}
/// 模式 A 是否（按位对齐地）覆盖模式 B 的语言。
pub fn subsumes(a: &PatternEntry, b: &PatternEntry) -> bool {
    if a.elems.len() > b.elems.len() {
        return false;
    }
    a.elems
        .iter()
        .enumerate()
        .all(|(i, elem)| elem_subsumes(elem, &b.elems[i]))
        && a.elems != b.elems
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LintFinding {
    pub kind: LintKind,
    pub subject_id: String,
    pub subject_name: String,
    pub object_id: String,
    pub object_name: String,
    pub detail: String,
}
impl LintFinding {
    pub fn render(&self) -> String {
        let kind = match self.kind {
            LintKind::Redundant => "冗余",
            LintKind::Conflict => "冲突",
            LintKind::Shadowed => "遮蔽",
        };
        format!(
            "[{}] {}（#{}）↔ {}（#{}）：{}",
            kind,
            self.subject_name,
            &self.subject_id.chars().take(8).collect::<String>(),
            self.object_name,
            &self.object_id.chars().take(8).collect::<String>(),
            self.detail
        )
    }
}
/// 静态体检：按评估序两两检查重复/冲突/遮蔽（只看参与条目=全部条目）。
pub fn lint(library: &PatternLibrary) -> Vec<LintFinding> {
    let mut findings = Vec::new();
    let order = library.evaluation_order();
    for (pos, &a_index) in order.iter().enumerate() {
        let a = &library.entries[a_index];
        for &b_index in order.iter().skip(pos + 1) {
            let b = &library.entries[b_index];
            if a.ir_fingerprint() == b.ir_fingerprint() {
                let kind = if a.query_type == b.query_type {
                    LintKind::Redundant
                } else {
                    LintKind::Conflict
                };
                let detail = if kind == LintKind::Redundant {
                    format!("IR 完全相同且同路由（{}），两条定义只需其一", a.query_type)
                } else {
                    format!(
                        "IR 完全相同但路由不同：{} vs {}",
                        a.query_type, b.query_type
                    )
                };
                findings.push(LintFinding {
                    kind,
                    subject_id: a.id(),
                    subject_name: a.name.clone(),
                    object_id: b.id(),
                    object_name: b.name.clone(),
                    detail,
                });
                continue;
            }
            if subsumes(a, b) {
                let detail = if a.query_type == b.query_type {
                    format!(
                        "「{}」覆盖「{}」的语言（优先 {} ≥ {}），后者永不胜出",
                        a.name, b.name, a.priority, b.priority
                    )
                } else {
                    format!(
                        "「{}」（{}）命中时「{}」（{}）被遮蔽，路由到 {}",
                        a.name, a.query_type, b.name, b.query_type, a.query_type
                    )
                };
                findings.push(LintFinding {
                    kind: LintKind::Shadowed,
                    subject_id: a.id(),
                    subject_name: a.name.clone(),
                    object_id: b.id(),
                    object_name: b.name.clone(),
                    detail,
                });
            }
        }
    }
    findings
}

// ---------------------------------------------------------------------------
// 模式关系：体检结果落成图边（关系本身是可查询的数据）
// ---------------------------------------------------------------------------
/// 模式之间的关系：静态体检结论的图视角。
///
/// 落图后这些关系就是 `kg:graphs/patterns` 里的普通三元组，
/// 「哪些模式互斥/冗余/互相遮蔽」由一条 SPARQL 就能问出来，
/// 不必再靠手写遍历（见 [`relation_edges`] 与 [`relation_quads`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// 语言包含：subject 覆盖 object 的语言（评估序下 object 被遮蔽）。
    Generalizes,
    /// 同一模式语言且同路由，两处定义只需其一。
    Redundant,
    /// 同一模式语言但路由不同：命中即冲突。
    Conflicts,
}
impl RelationKind {
    pub fn label(self) -> &'static str {
        match self {
            RelationKind::Generalizes => "泛化",
            RelationKind::Redundant => "冗余",
            RelationKind::Conflicts => "冲突",
        }
    }

    /// 关系谓词 IRI（`kg:ledger/` 命名空间）。
    pub fn predicate(self) -> &'static str {
        match self {
            RelationKind::Generalizes => P_GENERALIZES,
            RelationKind::Redundant => P_REDUNDANT,
            RelationKind::Conflicts => P_CONFLICTS,
        }
    }

    /// 是否对称（冗余/冲突双向各一条；泛化有方向）。
    pub fn is_symmetric(self) -> bool {
        !matches!(self, RelationKind::Generalizes)
    }
}
/// 一条模式关系边（端点带稳定 id 与可读名）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub kind: RelationKind,
    pub subject_id: String,
    pub subject_name: String,
    pub object_id: String,
    pub object_name: String,
}
impl RelationEdge {
    pub fn render(&self) -> String {
        format!(
            "{}（{}）{}（{}）",
            self.subject_name,
            &self.subject_id.chars().take(8).collect::<String>(),
            self.kind.label(),
            self.object_name
        )
    }
}
/// 由条目集计算全部模式关系（即静态体检的图视角；确定性、与顺序无关的集合）。
pub fn relation_edges(entries: &[PatternEntry]) -> Vec<RelationEdge> {
    let library = PatternLibrary {
        entries: entries.to_vec(),
    };
    lint(&library)
        .into_iter()
        .map(|finding| {
            let kind = match finding.kind {
                LintKind::Shadowed => RelationKind::Generalizes,
                LintKind::Redundant => RelationKind::Redundant,
                LintKind::Conflict => RelationKind::Conflicts,
            };
            RelationEdge {
                kind,
                subject_id: finding.subject_id,
                subject_name: finding.subject_name,
                object_id: finding.object_id,
                object_name: finding.object_name,
            }
        })
        .collect()
}
/// 按关系种类计数（图上体检摘要用）。
pub fn relation_counts(edges: &[RelationEdge]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for edge in edges {
        *counts.entry(edge.kind.label()).or_insert(0usize) += 1;
    }
    counts
}
/// 图上体检的 SPARQL：直接在模式命名图里数三种关系的边数。
pub fn relation_count_sparql() -> String {
    let predicates = [
        RelationKind::Generalizes.predicate(),
        RelationKind::Redundant.predicate(),
        RelationKind::Conflicts.predicate(),
    ]
    .iter()
    .map(|iri| format!("<{}>", iri))
    .collect::<Vec<_>>()
    .join(", ");
    format!(
        "SELECT ?rel (COUNT(*) AS ?n) WHERE {{ \
         GRAPH <{graph}> {{ ?subject ?rel ?object FILTER(?rel IN ({predicates})) }} \
         }} GROUP BY ?rel",
        graph = PATTERNS_GRAPH,
        predicates = predicates,
    )
}

// ---------------------------------------------------------------------------
// 模式落图：pattern IR → RDF 命名图（kg:graphs/patterns）
// ---------------------------------------------------------------------------
use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Term};
/// 模式账本命名图。
pub const PATTERNS_GRAPH: &str = "http://example.org/kg/graphs/patterns";
use crate::graph_ledger::LEDGER_NS;
type LedgerResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const PATTERN_CLASS: &str = "http://example.org/kg/ledger/Pattern";
const STAMP_CLASS: &str = "http://example.org/kg/ledger/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/ledger/stamp";
const P_JSON: &str = "http://example.org/kg/ledger/json";
const P_NAME: &str = "http://example.org/kg/ledger/name";
const P_QUERY_TYPE: &str = "http://example.org/kg/ledger/queryType";
const P_PRIORITY: &str = "http://example.org/kg/ledger/priority";
const P_SOURCE: &str = "http://example.org/kg/ledger/source";
const P_IR: &str = "http://example.org/kg/ledger/ir";
const P_GUARD: &str = "http://example.org/kg/ledger/contextGuard";
const P_FINGERPRINT: &str = "http://example.org/kg/ledger/fingerprint";
const P_COUNT: &str = "http://example.org/kg/ledger/count";
const P_GENERALIZES: &str = "http://example.org/kg/ledger/generalizes";
const P_REDUNDANT: &str = "http://example.org/kg/ledger/redundantWith";
const P_CONFLICTS: &str = "http://example.org/kg/ledger/conflictsWith";
/// 模式经验命名图：一次理解的经验与图谱/模式定义分开存放。
pub const EPISODES_GRAPH: &str = "http://example.org/kg/graphs/pattern-episodes";
const EPISODE_CLASS: &str = "http://example.org/kg/ledger/Episode";
const EPISODE_STAMP_SUBJECT: &str = "http://example.org/kg/ledger/episodeStamp";
const P_EPISODE_JSON: &str = "http://example.org/kg/ledger/episodeJson";
const P_EPISODE_INPUT: &str = "http://example.org/kg/ledger/input";
const P_EPISODE_OUTCOME: &str = "http://example.org/kg/ledger/outcome";
const P_EPISODE_LAST_TURN: &str = "http://example.org/kg/ledger/lastTurn";
fn named_node(iri: &str) -> LedgerResult<NamedNode> {
    Ok(NamedNode::new(iri)?)
}
fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> LedgerResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}
/// 模式集指纹（按条目 id 升序折叠，与插入顺序无关）。
pub fn library_fingerprint(entries: &[PatternEntry]) -> String {
    let mut ids: Vec<String> = entries.iter().map(PatternEntry::id).collect();
    ids.sort();
    fnv1a_hex(ids.join("\n").as_bytes())
}
/// 生成模式账本的全部四元组（含物化戳与模式关系边）。
///
/// 模式关系（泛化/冗余/冲突）来自 [`relation_edges`]：对称关系双向各一条，
/// 于是「哪两个模式互斥」在图上是一条反向也走得通的三元组。
pub fn pattern_quads(entries: &[PatternEntry]) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(PATTERNS_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let pattern_class = Term::NamedNode(named_node(PATTERN_CLASS)?);
    let edges = relation_edges(entries);
    let mut quads: Vec<Quad> = Vec::with_capacity(entries.len() * 8 + edges.len() * 2 + 3);
    for entry in entries {
        let subject = named_node(&format!("{}pattern/{}", LEDGER_NS, entry.id()))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            pattern_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(entry)?,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_NAME, &entry.name, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_QUERY_TYPE,
            &entry.query_type,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PRIORITY,
            &entry.priority.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_SOURCE, &entry.source, &graph)?);
        quads.push(literal_quad(&subject, P_IR, &entry.render(), &graph)?);
        quads.push(literal_quad(&subject, P_FINGERPRINT, &entry.id(), &graph)?);
        if let Some(guard) = &entry.guard {
            quads.push(literal_quad(&subject, P_GUARD, &guard.render(), &graph)?);
        }
    }
    for edge in &edges {
        let subject = named_node(&format!("{}pattern/{}", LEDGER_NS, edge.subject_id))?;
        let object = named_node(&format!("{}pattern/{}", LEDGER_NS, edge.object_id))?;
        let predicate = named_node(edge.kind.predicate())?;
        quads.push(Quad::new(
            subject.clone(),
            predicate.clone(),
            object.clone(),
            graph.clone(),
        ));
        if edge.kind.is_symmetric() {
            quads.push(Quad::new(object, predicate, subject, graph.clone()));
        }
    }
    let stamp = named_node(STAMP_SUBJECT)?;
    quads.push(Quad::new(
        stamp.clone(),
        rdf_type,
        Term::NamedNode(named_node(STAMP_CLASS)?),
        graph.clone(),
    ));
    quads.push(literal_quad(
        &stamp,
        P_FINGERPRINT,
        &library_fingerprint(entries),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &entries.len().to_string(),
        &graph,
    )?);
    Ok(quads)
}
/// 从模式命名图的四元组还原条目（按 id 升序，确定性；物化戳忽略）。
pub fn parse_pattern_graph(quads: impl Iterator<Item = Quad>) -> Vec<PatternEntry> {
    let mut by_subject: BTreeMap<String, String> = BTreeMap::new();
    for quad in quads {
        if quad.predicate.as_str() != P_JSON {
            continue;
        }
        let subject = quad.subject.to_string();
        if let Term::Literal(value) = quad.object {
            by_subject.insert(subject, value.value().to_string());
        }
    }
    by_subject
        .into_values()
        .filter_map(|json| serde_json::from_str::<PatternEntry>(&json).ok())
        .collect()
}
/// 经验集指纹（按经验 id 升序折叠，与插入顺序无关）。
pub fn episode_fingerprint(episodes: &[PatternEpisode]) -> String {
    let mut ids: Vec<String> = episodes.iter().map(PatternEpisode::id).collect();
    ids.sort();
    fnv1a_hex(ids.join("\n").as_bytes())
}
/// 生成模式经验账本的全部四元组（独立命名图 + 物化戳）。
pub fn episode_quads(episodes: &[PatternEpisode]) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(EPISODES_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let episode_class = Term::NamedNode(named_node(EPISODE_CLASS)?);
    let mut quads: Vec<Quad> = Vec::with_capacity(episodes.len() * 5 + 3);
    for episode in episodes {
        let subject = named_node(&format!("{}episode/{}", LEDGER_NS, episode.id()))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            episode_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_EPISODE_JSON,
            &serde_json::to_string(episode)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_INPUT,
            &episode.input,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_OUTCOME,
            episode.outcome.label(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_LAST_TURN,
            &episode.last_turn.to_string(),
            &graph,
        )?);
    }
    let stamp = named_node(EPISODE_STAMP_SUBJECT)?;
    quads.push(Quad::new(
        stamp.clone(),
        rdf_type,
        Term::NamedNode(named_node(STAMP_CLASS)?),
        graph.clone(),
    ));
    quads.push(literal_quad(
        &stamp,
        P_FINGERPRINT,
        &episode_fingerprint(episodes),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &episodes.len().to_string(),
        &graph,
    )?);
    Ok(quads)
}
/// 从经验命名图还原经验（按 id 升序，确定性；物化戳忽略）。
pub fn parse_episode_graph(quads: impl Iterator<Item = Quad>) -> Vec<PatternEpisode> {
    let mut by_subject: BTreeMap<String, String> = BTreeMap::new();
    for quad in quads {
        if quad.predicate.as_str() != P_EPISODE_JSON {
            continue;
        }
        let subject = quad.subject.to_string();
        if let Term::Literal(value) = quad.object {
            by_subject.insert(subject, value.value().to_string());
        }
    }
    by_subject
        .into_values()
        .filter_map(|json| serde_json::from_str::<PatternEpisode>(&json).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    include!("pattern_ir__tests_01.rs");
}

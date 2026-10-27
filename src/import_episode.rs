//! 导入情景记忆（Import Episode Memory）：把一次 RDF 批量导入记成可回指、
//! 可审计、可复述的对象。
//!
//! 设计立场：
//!   - **默认图仍是主视图**：断言照旧并入默认图，旧查询与旧测试不变；
//!   - **批次是 RDF 里的一等公民**：同一批断言另存进 `kg:graphs/import-batches/{id}`
//!     命名图，可用 `GRAPH <批次图> { ... }` 做范围查询与审计；
//!   - **元数据单独成图**：序号、来源、格式、解析/新增/重复计数与确定性 gist
//!     写入 `kg:graphs/import-episodes`，不污染默认图；
//!   - **内容寻址、幂等**：批次 id 由规范化、排序、去重后的断言集合哈希得到；
//!     空白节点因解析时会重命名，是已知的幂等例外，必须在回执里显式披露；
//!   - **gist 有损但诚实**：只给计数、Top-N 与带引用的样例；列表截断时说明
//!     「前 N / 共 M」，样例绝不伪装成全量；
//!   - **只做模板组合**：每句都可回溯到 `[B{seq}-{ordinal}]`，不猜、不补全。

use oxigraph::model::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// 导入情景元数据命名图。
pub const EPISODE_LEDGER_GRAPH: &str = "http://example.org/kg/graphs/import-episodes";
/// 批次断言命名图前缀：`{prefix}{batch_id}`。
pub const BATCH_GRAPH_PREFIX: &str = "http://example.org/kg/graphs/import-batches/";
/// 情景元数据节点前缀：`{prefix}{batch_id}`。
pub const EPISODE_SUBJECT_PREFIX: &str = "http://example.org/kg/ledger/import-episode/";

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const BASE_NS: &str = "http://example.org/kg/";
const EPISODE_CLASS: &str = "http://example.org/kg/ledger/ImportEpisode";

const P_JSON: &str = "http://example.org/kg/ledger/json";
const P_SEQ: &str = "http://example.org/kg/ledger/seq";
const P_SOURCE: &str = "http://example.org/kg/ledger/source";
const P_FORMAT: &str = "http://example.org/kg/ledger/format";
const P_TOTAL: &str = "http://example.org/kg/ledger/total";
const P_ADDED: &str = "http://example.org/kg/ledger/added";
const P_DUPLICATES: &str = "http://example.org/kg/ledger/duplicates";
const P_BLANK_NODES: &str = "http://example.org/kg/ledger/blankNodes";
const P_BATCH_GRAPH: &str = "http://example.org/kg/ledger/batchGraph";
const P_IMPORTED_AT: &str = "http://example.org/kg/ledger/importedAt";
const P_INPUT_NAMED_GRAPHS: &str = "http://example.org/kg/ledger/inputNamedGraphs";
const P_TYPE_COUNT: &str = "http://example.org/kg/ledger/typeCount";
const P_PREDICATE_COUNT: &str = "http://example.org/kg/ledger/predicateCount";
const P_TOP_SUBJECT: &str = "http://example.org/kg/ledger/topSubject";
const P_SAMPLE: &str = "http://example.org/kg/ledger/sample";

/// 默认 Top-N 长度（类型 / 谓词 / 主体）。
pub const DEFAULT_TOP_N: usize = 5;
/// 默认样例条数。
pub const DEFAULT_SAMPLE_LIMIT: usize = 8;
/// 台账一次最多展示的批次数。
pub const MAX_LEDGER_ROWS: usize = 50;

type EpisodeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 一条可展示、可引用的样例事实（词面已本地化；不用于内容哈希）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GistFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl GistFact {
    /// 人类可读的一行断言。
    pub fn line(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// 一次导入的确定性 gist：有损、可复述、可回溯。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BatchGist {
    /// 断言总数（批内去重前；与导入回执的 total 一致）。
    pub total: usize,
    /// 不同主体数。
    pub distinct_subjects: usize,
    /// 不同谓词数（全量，不因 Top-N 截断而丢失计数）。
    pub distinct_predicates: usize,
    /// 不同 rdf:type 宾语数（全量）。
    pub distinct_types: usize,
    pub type_counts: Vec<(String, usize)>,
    pub predicate_counts: Vec<(String, usize)>,
    pub top_subjects: Vec<(String, usize)>,
    pub samples: Vec<GistFact>,
}

/// 一次批量导入的完整情景记录（元数据图与 JSON 往返的唯一结构）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportEpisode {
    /// 单调递增批次号（从 1 开始）。
    pub seq: usize,
    /// 内容寻址批次 id。
    pub id: String,
    pub source: String,
    pub format: String,
    /// Unix 秒；只作审计，不参与 gist 的确定性。
    pub imported_at: u64,
    /// 解析出的断言总数（保留原输入命名图的数量只记在 input_named_graphs）。
    pub total: usize,
    /// 本批首次写入默认图的新增数。
    pub added: usize,
    /// 本批与默认图已有断言重复的数量。
    pub duplicates: usize,
    /// 本批是否含空白节点（内容寻址的已知例外）。
    pub blank_nodes: bool,
    /// 输入数据集里出现过的命名图数量（落库时统一压平）。
    pub input_named_graphs: usize,
    /// 本批断言所在的命名图 IRI。
    pub batch_graph: String,
    pub gist: BatchGist,
}

impl ImportEpisode {
    /// 批次的一句式摘要（台账 / CLI 共用）。
    pub fn line(&self) -> String {
        let mut line = format!(
            "B{}｜{}｜{}｜{} 条断言（新增 {}，重复 {}）｜{}",
            self.seq,
            self.source,
            self.format,
            self.total,
            self.added,
            self.duplicates,
            self.id
        );
        if self.blank_nodes {
            line.push_str("｜含空白节点：批次 id 可能随解析变化");
        }
        line
    }
}

/// 导入回执：episode 是历史记录，added/duplicates 是本次实际写入统计。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RdfImportReceipt {
    pub episode: ImportEpisode,
    pub added: usize,
    pub duplicates: usize,
    /// 本次是否新建了批次记录（同内容重复导入为 false）。
    pub recorded: bool,
}

impl RdfImportReceipt {
    pub fn summary(&self) -> String {
        if self.recorded {
            format!(
                "新增 {} 条三元组；批次 B{} 已记为 {} 条断言（重复 {} 条）。",
                self.added, self.episode.seq, self.episode.total, self.duplicates
            )
        } else {
            format!(
                "新增 {} 条三元组；批次 B{} 已存在，未重复记账（重复 {} 条）。",
                self.added, self.episode.seq, self.duplicates
            )
        }
    }
}

/// 批次命名图 IRI。
pub fn batch_graph_iri(id: &str) -> String {
    format!("{}{}", BATCH_GRAPH_PREFIX, id)
}

/// 情景元数据主语 IRI。
pub fn episode_subject_iri(id: &str) -> String {
    format!("{}{}", EPISODE_SUBJECT_PREFIX, id)
}

/// RDF 序列化格式的稳定中文/英文标签（随回执与台账展示）。
pub fn rdf_format_label(format: oxigraph::io::RdfFormat) -> &'static str {
    match format {
        oxigraph::io::RdfFormat::Turtle => "Turtle",
        oxigraph::io::RdfFormat::NTriples => "N-Triples",
        oxigraph::io::RdfFormat::NQuads => "N-Quads",
        oxigraph::io::RdfFormat::TriG => "TriG",
        oxigraph::io::RdfFormat::RdfXml => "RDF/XML",
        oxigraph::io::RdfFormat::N3 => "N3",
        oxigraph::io::RdfFormat::JsonLd { .. } => "JSON-LD",
        _ => "RDF",
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn short_hash(text: &str) -> String {
    format!("{:016x}{:x}", fnv1a64(text.as_bytes()), text.len())
}

fn escape_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn subject_key(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(node) => format!("<{}>", node.as_str()),
        Subject::BlankNode(node) => format!("_:{}", node.as_str()),
        Subject::Triple(triple) => format!("<<{}>>", triple),
    }
}

fn term_key(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => format!("<{}>", node.as_str()),
        Term::BlankNode(node) => format!("_:{}", node.as_str()),
        Term::Literal(literal) => {
            let language = literal
                .language()
                .map(|language| format!("@{}", language))
                .unwrap_or_default();
            format!(
                "\"{}\"^^<{}>{}",
                escape_literal(literal.value()),
                literal.datatype().as_str(),
                language
            )
        }
        Term::Triple(triple) => format!("<<{}>>", triple),
    }
}

/// 与插入顺序无关的断言规范形式（不包含原图名，因为落库时会压平）。
fn assertion_key(quad: &Quad) -> String {
    format!(
        "{} <{}> {}",
        subject_key(&quad.subject),
        quad.predicate.as_str(),
        term_key(&quad.object)
    )
}

fn subject_has_blank(subject: &Subject) -> bool {
    match subject {
        Subject::BlankNode(_) => true,
        Subject::Triple(triple) => triple_has_blank(triple),
        _ => false,
    }
}

fn triple_has_blank(triple: &Triple) -> bool {
    subject_has_blank(&triple.subject)
        || term_has_blank(&triple.object)
        || matches!(triple.subject, Subject::BlankNode(_))
}

fn term_has_blank(term: &Term) -> bool {
    match term {
        Term::BlankNode(_) => true,
        Term::Triple(triple) => triple_has_blank(triple),
        _ => false,
    }
}

/// 批次内是否含空白节点（主语或宾语；空白图名已在落库时替换为批次图）。
pub fn has_blank_nodes(quads: &[Quad]) -> bool {
    quads
        .iter()
        .any(|quad| subject_has_blank(&quad.subject) || term_has_blank(&quad.object))
}

/// 内容寻址批次 id：对规范化、排序、去重后的断言集合做 FNV-1a。
///
/// 原图名被有意排除（批次图保存压平后的断言）；空白节点的标签来自解析器
/// 重命名，因此含空白节点的同一文档可能得到不同 id，调用方应显式披露。
pub fn batch_id(quads: &[Quad]) -> String {
    let mut keys: Vec<String> = quads.iter().map(assertion_key).collect();
    keys.sort();
    keys.dedup();
    short_hash(&keys.join("\n"))
}

fn local_name(iri: &str) -> Option<String> {
    crate::identity::local_name(BASE_NS, iri)
}

fn display_named_node(node: &NamedNode) -> String {
    local_name(node.as_str()).unwrap_or_else(|| node.as_str().to_string())
}

fn display_subject(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(node) => display_named_node(node),
        Subject::BlankNode(node) => format!("_:{}", node.as_str()),
        Subject::Triple(triple) => format!("<<{}>>", triple),
    }
}

fn display_term(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => display_named_node(node),
        Term::BlankNode(node) => format!("_:{}", node.as_str()),
        Term::Literal(literal) => literal.value().to_string(),
        Term::Triple(triple) => format!("<<{}>>", triple),
    }
}

fn top_counts(counts: BTreeMap<String, usize>, limit: usize) -> Vec<(String, usize)> {
    let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
    entries.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    entries.truncate(limit);
    entries
}

/// 从断言集合构造确定性 gist（Top-N 与样例都会按 `limit` 截断）。
pub fn build_gist(quads: &[Quad], top_n: usize, sample_limit: usize) -> BatchGist {
    let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut predicate_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut subject_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut subjects: BTreeSet<String> = BTreeSet::new();

    for quad in quads {
        let subject = display_subject(&quad.subject);
        subjects.insert(subject.clone());
        *subject_counts.entry(subject).or_insert(0) += 1;
        let predicate = display_named_node(&quad.predicate);
        *predicate_counts.entry(predicate).or_insert(0) += 1;
        if quad.predicate.as_str() == RDF_TYPE_IRI {
            if let Term::NamedNode(node) = &quad.object {
                *type_counts.entry(display_named_node(node)).or_insert(0) += 1;
            }
        }
    }

    let distinct_predicates = predicate_counts.len();
    let distinct_types = type_counts.len();

    // 样例选择：每个主体优先一条非 type 断言，再用 type 断言补位，最后按规范
    // 顺序填满。规则简单、可复现，且尽量覆盖更多主体而不是堆同一实体的边。
    let mut sorted: Vec<&Quad> = quads.iter().collect();
    sorted.sort_by_key(|quad| assertion_key(quad));
    sorted.dedup_by_key(|quad| assertion_key(quad));
    let mut picked: BTreeSet<String> = BTreeSet::new();
    let mut samples: Vec<GistFact> = Vec::new();
    let mut seen_subjects: BTreeSet<String> = BTreeSet::new();
    for quad in &sorted {
        if quad.predicate.as_str() == RDF_TYPE_IRI {
            continue;
        }
        let subject = display_subject(&quad.subject);
        if seen_subjects.insert(subject) {
            let key = assertion_key(quad);
            if picked.insert(key) {
                samples.push(GistFact {
                    subject: display_subject(&quad.subject),
                    predicate: display_named_node(&quad.predicate),
                    object: display_term(&quad.object),
                });
            }
            if samples.len() >= sample_limit {
                break;
            }
        }
    }
    if samples.len() < sample_limit {
        seen_subjects.clear();
        for quad in &sorted {
            if quad.predicate.as_str() != RDF_TYPE_IRI {
                continue;
            }
            let subject = display_subject(&quad.subject);
            if seen_subjects.insert(subject) {
                let key = assertion_key(quad);
                if picked.insert(key) {
                    samples.push(GistFact {
                        subject: display_subject(&quad.subject),
                        predicate: display_named_node(&quad.predicate),
                        object: display_term(&quad.object),
                    });
                }
                if samples.len() >= sample_limit {
                    break;
                }
            }
        }
    }
    if samples.len() < sample_limit {
        for quad in &sorted {
            let key = assertion_key(quad);
            if picked.insert(key) {
                samples.push(GistFact {
                    subject: display_subject(&quad.subject),
                    predicate: display_named_node(&quad.predicate),
                    object: display_term(&quad.object),
                });
                if samples.len() >= sample_limit {
                    break;
                }
            }
        }
    }

    BatchGist {
        total: quads.len(),
        distinct_subjects: subjects.len(),
        distinct_predicates,
        distinct_types,
        type_counts: top_counts(type_counts, top_n),
        predicate_counts: top_counts(predicate_counts, top_n),
        top_subjects: top_counts(subject_counts, top_n),
        samples,
    }
}

fn named_node(iri: &str) -> EpisodeResult<NamedNode> {
    Ok(NamedNode::new(iri)?)
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> EpisodeResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}

/// 生成情景元数据图的全部四元组（JSON 是往返正本，其余字段供 SPARQL 直接审计）。
pub fn episode_quads(episode: &ImportEpisode) -> EpisodeResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(EPISODE_LEDGER_GRAPH)?);
    let subject = named_node(&episode_subject_iri(&episode.id))?;
    let mut quads = Vec::with_capacity(12 + episode.gist.samples.len() + 24);
    quads.push(Quad::new(
        subject.clone(),
        named_node(RDF_TYPE)?,
        Term::NamedNode(named_node(EPISODE_CLASS)?),
        graph.clone(),
    ));
    quads.push(literal_quad(
        &subject,
        P_JSON,
        &serde_json::to_string(episode)?,
        &graph,
    )?);
    for (predicate, value) in [
        (P_SEQ, episode.seq.to_string()),
        (P_SOURCE, episode.source.clone()),
        (P_FORMAT, episode.format.clone()),
        (P_TOTAL, episode.total.to_string()),
        (P_ADDED, episode.added.to_string()),
        (P_DUPLICATES, episode.duplicates.to_string()),
        (P_BLANK_NODES, episode.blank_nodes.to_string()),
        (P_IMPORTED_AT, episode.imported_at.to_string()),
        (
            P_INPUT_NAMED_GRAPHS,
            episode.input_named_graphs.to_string(),
        ),
    ] {
        quads.push(literal_quad(&subject, predicate, &value, &graph)?);
    }
    quads.push(Quad::new(
        subject.clone(),
        named_node(P_BATCH_GRAPH)?,
        Term::NamedNode(named_node(&episode.batch_graph)?),
        graph.clone(),
    ));
    for (label, count) in &episode.gist.type_counts {
        quads.push(literal_quad(
            &subject,
            P_TYPE_COUNT,
            &format!("{}|{}", label, count),
            &graph,
        )?);
    }
    for (label, count) in &episode.gist.predicate_counts {
        quads.push(literal_quad(
            &subject,
            P_PREDICATE_COUNT,
            &format!("{}|{}", label, count),
            &graph,
        )?);
    }
    for (label, count) in &episode.gist.top_subjects {
        quads.push(literal_quad(
            &subject,
            P_TOP_SUBJECT,
            &format!("{}|{}", label, count),
            &graph,
        )?);
    }
    for (index, sample) in episode.gist.samples.iter().enumerate() {
        quads.push(literal_quad(
            &subject,
            P_SAMPLE,
            &format!("[B{}-{}] {}", episode.seq, index + 1, sample.line()),
            &graph,
        )?);
    }
    Ok(quads)
}

/// 从情景元数据图还原批次记录（按 seq、id 升序；损坏的 JSON 条目被跳过）。
pub fn parse_episode_graph(quads: impl Iterator<Item = Quad>) -> Vec<ImportEpisode> {
    let mut episodes: Vec<ImportEpisode> = Vec::new();
    for quad in quads {
        if quad.predicate.as_str() != P_JSON {
            continue;
        }
        if let Term::Literal(literal) = &quad.object {
            if let Ok(episode) = serde_json::from_str::<ImportEpisode>(literal.value()) {
                episodes.push(episode);
            }
        }
    }
    episodes.sort_by(|left, right| left.seq.cmp(&right.seq).then_with(|| left.id.cmp(&right.id)));
    episodes.dedup_by(|left, right| left.id == right.id);
    episodes
}

/// 批次引用：只在问题出现批次线索时使用，避免普通问题被最近导入偏置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeReference {
    Latest,
    Previous,
    Ordinal(usize),
    Id(String),
}

fn is_numeral_char(ch: char) -> bool {
    ch.is_ascii_digit() || "零〇一二两三四五六七八九十百".contains(ch)
}

fn parse_chinese_number(text: &str) -> Option<usize> {
    if text.chars().all(|ch| ch.is_ascii_digit()) {
        return text.parse().ok();
    }
    let mut total = 0usize;
    let mut current = 0usize;
    let mut saw = false;
    for ch in text.chars() {
        saw = true;
        match ch {
            '零' | '〇' => current = 0,
            '一' => current = 1,
            '二' | '两' => current = 2,
            '三' => current = 3,
            '四' => current = 4,
            '五' => current = 5,
            '六' => current = 6,
            '七' => current = 7,
            '八' => current = 8,
            '九' => current = 9,
            '十' => {
                if current == 0 {
                    current = 1;
                }
                total = total.saturating_add(current.saturating_mul(10));
                current = 0;
            }
            '百' => {
                if current == 0 {
                    current = 1;
                }
                total = total.saturating_add(current.saturating_mul(100));
                current = 0;
            }
            _ => return None,
        }
    }
    if !saw {
        return None;
    }
    Some(total.saturating_add(current))
}

fn parse_ordinal_batch(text: &str) -> Option<usize> {
    let mut chars = text.char_indices().peekable();
    while let Some((_, ch)) = chars.next() {
        if ch != '第' {
            continue;
        }
        let mut number = String::new();
        while let Some(&(_, next)) = chars.peek() {
            if next.is_whitespace() {
                chars.next();
            } else if is_numeral_char(next) {
                number.push(next);
                chars.next();
            } else {
                break;
            }
        }
        if number.is_empty() {
            continue;
        }
        if let Some(&(_, '个')) = chars.peek() {
            chars.next();
        }
        if let Some(&(_, '批')) = chars.peek() {
            return parse_chinese_number(&number).filter(|seq| *seq > 0);
        }
    }
    None
}

fn parse_english_ordinal(text: &str) -> Option<usize> {
    let lower = text.to_ascii_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    for window in tokens.windows(2) {
        if matches!(window[0], "batch" | "episode") {
            if let Ok(seq) = window[1].parse::<usize>() {
                if seq > 0 {
                    return Some(seq);
                }
            }
        }
    }
    None
}

/// 解析显式批次线索：这批 / 上一批 / 第 N 批 / batch N。
///
/// 普通问句（没有批次线索）返回 `None`，仍走全局图谱路径。
pub fn parse_episode_reference(text: &str) -> Option<EpisodeReference> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.contains("上一批")
        || trimmed.contains("上一次导入")
        || trimmed.contains("上一份导入")
        || trimmed.contains("前一批")
    {
        return Some(EpisodeReference::Previous);
    }
    if trimmed.contains("这批")
        || trimmed.contains("本批")
        || trimmed.contains("这一批")
        || trimmed.contains("最近一批")
        || trimmed.contains("最近那次导入")
        || trimmed.contains("刚才那批")
        || trimmed.contains("刚才的导入")
        || trimmed.contains("刚才导入")
    {
        return Some(EpisodeReference::Latest);
    }
    if let Some(seq) = parse_ordinal_batch(trimmed).or_else(|| parse_english_ordinal(trimmed)) {
        return Some(EpisodeReference::Ordinal(seq));
    }
    if let Some(rest) = trimmed
        .split_once("批次id：")
        .or_else(|| trimmed.split_once("批次id:"))
        .or_else(|| trimmed.split_once("batch id:"))
        .or_else(|| trimmed.split_once("batch id："))
    {
        let id = rest.1.trim().split_whitespace().next().unwrap_or("");
        if !id.is_empty() {
            return Some(EpisodeReference::Id(id.to_string()));
        }
    }
    None
}

/// 问题是否带批次/来源线索（决定是否走情景路径）。
pub fn looks_like_episode_reference(text: &str) -> bool {
    if parse_episode_reference(text).is_some() {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    [".ttl", ".nt", ".nq", ".trig", ".rdf", ".xml"]
        .iter()
        .any(|extension| lower.contains(extension))
        || text.contains("批次台账")
        || text.contains("导入批次")
        || text.contains("批次列表")
}

fn source_matches(source: &str, text: &str) -> bool {
    let source = source.trim();
    if source.is_empty() {
        return false;
    }
    if text.contains(source) {
        return true;
    }
    let base = source
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source)
        .trim();
    base.chars().count() >= 3 && text.contains(base)
}

fn id_matches(id: &str, needle: &str) -> bool {
    id == needle
        || (needle.chars().count() >= 4 && id.starts_with(needle))
        || (id.starts_with(needle) && needle.chars().count() >= 4)
}

impl EpisodeReference {
    pub fn resolve<'a>(&self, episodes: &'a [ImportEpisode]) -> Option<&'a ImportEpisode> {
        match self {
            Self::Latest => episodes.iter().max_by_key(|episode| episode.seq),
            Self::Previous => {
                let mut ordered: Vec<&ImportEpisode> = episodes.iter().collect();
                ordered.sort_by_key(|episode| episode.seq);
                if ordered.len() < 2 {
                    None
                } else {
                    ordered.get(ordered.len() - 2).copied()
                }
            }
            Self::Ordinal(seq) => episodes.iter().find(|episode| episode.seq == *seq),
            Self::Id(id) => {
                let mut matches = episodes
                    .iter()
                    .filter(|episode| id_matches(&episode.id, id));
                let first = matches.next()?;
                if matches.next().is_some() {
                    None
                } else {
                    Some(first)
                }
            }
        }
    }
}

/// 把自然语言/CLI 引用解析到具体批次：显式线索优先，其次来源文件名，最后 id。
pub fn resolve_episode_reference<'a>(
    text: &str,
    episodes: &'a [ImportEpisode],
) -> Option<&'a ImportEpisode> {
    if let Some(reference) = parse_episode_reference(text) {
        return reference.resolve(episodes);
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(episode) = episodes
        .iter()
        .rev()
        .find(|episode| source_matches(&episode.source, trimmed))
    {
        return Some(episode);
    }
    let needle = trimmed.split_whitespace().next().unwrap_or(trimmed);
    episodes
        .iter()
        .rev()
        .find(|episode| id_matches(&episode.id, needle))
}

fn truncated_note(shown: usize, total: usize, unit: &str) -> String {
    if total > shown {
        format!("前 {} {} / 共 {} {}", shown, unit, total, unit)
    } else {
        format!("共 {} {}", total, unit)
    }
}

fn render_counter_list(title: &str, entries: &[(String, usize)], total: usize) -> String {
    if total == 0 {
        return format!("{}：无", title);
    }
    let body = entries
        .iter()
        .map(|(label, count)| format!("{}×{}", label, count))
        .collect::<Vec<_>>()
        .join("、");
    format!(
        "{}：{}（{}）",
        title,
        body,
        truncated_note(entries.len(), total, "项")
    )
}

/// 批次简报：所有数字来自 episode，所有样例带 `[B{seq}-{n}]` 引用。
pub fn render_episode_brief(episode: &ImportEpisode) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "批次 B{}｜来源 {}｜格式 {}｜{} 条断言（新增 {}，重复 {}）",
        episode.seq, episode.source, episode.format, episode.total, episode.added, episode.duplicates
    ));
    lines.push(format!(
        "稳定 id {}｜批次图 {}",
        episode.id, episode.batch_graph
    ));
    if episode.input_named_graphs > 0 {
        lines.push(format!(
            "输入含 {} 个命名图，已按既有语义压平后写入默认图与本批命名图。",
            episode.input_named_graphs
        ));
    }
    if episode.blank_nodes {
        lines.push(
            "注意：本批含空白节点；解析器会重命名空白节点，同一文档重复导入可能得到不同批次 id。"
                .to_string(),
        );
    }
    if episode.total == 0 {
        lines.push("本批没有可陈述的断言。".to_string());
        return lines.join("\n");
    }
    lines.push(format!(
        "不同主体 {} 个；{}。",
        episode.gist.distinct_subjects,
        render_counter_list(
            "类型分布",
            &episode.gist.type_counts,
            episode.gist.distinct_types
        )
    ));
    lines.push(render_counter_list(
        "谓词分布",
        &episode.gist.predicate_counts,
        episode.gist.distinct_predicates,
    ));
    if !episode.gist.top_subjects.is_empty() {
        let body = episode
            .gist
            .top_subjects
            .iter()
            .map(|(label, count)| format!("{}（{} 条）", label, count))
            .collect::<Vec<_>>()
            .join("、");
        lines.push(format!(
            "主体 Top：{}（{}）",
            body,
            truncated_note(
                episode.gist.top_subjects.len(),
                episode.gist.distinct_subjects,
                "个"
            )
        ));
    }
    lines.push(format!(
        "样例（{}）：",
        truncated_note(episode.gist.samples.len(), episode.total, "条断言")
    ));
    for (index, sample) in episode.gist.samples.iter().enumerate() {
        lines.push(format!(
            "  [B{}-{}] {}",
            episode.seq,
            index + 1,
            sample.line()
        ));
    }
    lines.join("\n")
}

/// 批次台账：按 seq 升序；超过上限时只展示最近 N 批并说明总数。
pub fn render_episode_ledger(episodes: &[ImportEpisode]) -> String {
    if episodes.is_empty() {
        return "还没有 RDF 导入批次。用 import-rdf 或「导入rdf文件：」导入后会自动记账。".to_string();
    }
    let mut ordered: Vec<&ImportEpisode> = episodes.iter().collect();
    ordered.sort_by_key(|episode| episode.seq);
    let total = ordered.len();
    let start = total.saturating_sub(MAX_LEDGER_ROWS);
    let mut lines = Vec::new();
    if start > 0 {
        lines.push(format!(
            "共 {} 批；仅显示最近 {} 批（B{}–B{}）。",
            total,
            MAX_LEDGER_ROWS,
            ordered[start].seq,
            ordered[total - 1].seq
        ));
    } else {
        lines.push(format!("共 {} 批。", total));
    }
    for episode in &ordered[start..] {
        lines.push(episode.line());
    }
    lines.join("\n")
}

/// 当前 Unix 秒（审计用；失败时回落为 0，不影响导入）。
pub fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(value: &str) -> NamedNode {
        NamedNode::new(value).unwrap()
    }

    fn quad(subject: &str, predicate: &str, object: &str, graph: GraphName) -> Quad {
        Quad::new(
            iri(&crate::identity::entity_uri(BASE_NS, subject)),
            iri(&format!("{}{}", BASE_NS, predicate)),
            Term::NamedNode(iri(&crate::identity::entity_uri(BASE_NS, object))),
            graph,
        )
    }

    fn literal_quad(subject: &str, predicate: &str, value: &str) -> Quad {
        Quad::new(
            iri(&crate::identity::entity_uri(BASE_NS, subject)),
            iri(&format!("{}{}", BASE_NS, predicate)),
            Literal::new_simple_literal(value),
            GraphName::DefaultGraph,
        )
    }

    fn episode(seq: usize, id: &str, source: &str) -> ImportEpisode {
        ImportEpisode {
            seq,
            id: id.to_string(),
            source: source.to_string(),
            format: "Turtle".to_string(),
            imported_at: 1700000000 + seq as u64,
            total: 2,
            added: 2,
            duplicates: 0,
            blank_nodes: false,
            input_named_graphs: 0,
            batch_graph: batch_graph_iri(id),
            gist: BatchGist {
                total: 2,
                distinct_subjects: 1,
                distinct_predicates: 2,
                distinct_types: 0,
                type_counts: vec![],
                predicate_counts: vec![("name".to_string(), 1)],
                top_subjects: vec![("张三".to_string(), 2)],
                samples: vec![GistFact {
                    subject: "张三".to_string(),
                    predicate: "name".to_string(),
                    object: "张三".to_string(),
                }],
            },
        }
    }

    #[test]
    fn batch_id_is_order_independent_and_content_addressed() {
        let first = vec![
            quad("张三", "认识", "李四", GraphName::DefaultGraph),
            literal_quad("张三", "name", "张三"),
        ];
        let second = vec![first[1].clone(), first[0].clone()];
        assert_eq!(batch_id(&first), batch_id(&second));
        let mut changed = first.clone();
        changed.push(literal_quad("张三", "age", "30"));
        assert_ne!(batch_id(&first), batch_id(&changed));
        assert!(!has_blank_nodes(&first));
    }

    #[test]
    fn gist_is_bounded_and_samples_use_distinct_subjects() {
        let mut quads = Vec::new();
        for index in 0..20 {
            let subject = format!("实体{}", index);
            quads.push(literal_quad(&subject, "name", &subject));
        }
        let gist = build_gist(&quads, 3, 4);
        assert_eq!(gist.total, 20);
        assert_eq!(gist.distinct_subjects, 20);
        assert_eq!(gist.distinct_predicates, 1);
        assert_eq!(gist.predicate_counts.len(), 1);
        assert_eq!(gist.samples.len(), 4);
        assert_eq!(
            gist.samples
                .iter()
                .map(|sample| sample.subject.clone())
                .collect::<BTreeSet<_>>()
                .len(),
            4
        );
        let rendered = render_episode_brief(&episode(1, "abc123", "sample.ttl"));
        assert!(rendered.contains("[B1-1]"));
        assert!(rendered.contains("前 1 项 / 共 2 项"), "{}", rendered);
    }

    #[test]
    fn references_resolve_latest_previous_ordinal_and_source() {
        let episodes = vec![
            episode(1, "aaaa1111", "data/first.ttl"),
            episode(2, "bbbb2222", "data/second.ttl"),
        ];
        assert_eq!(
            parse_episode_reference("这批讲了什么"),
            Some(EpisodeReference::Latest)
        );
        assert_eq!(
            parse_episode_reference("上一批有哪些人物"),
            Some(EpisodeReference::Previous)
        );
        assert_eq!(
            parse_episode_reference("第十二批简报"),
            Some(EpisodeReference::Ordinal(12))
        );
        assert_eq!(
            parse_episode_reference("batch 2 summary"),
            Some(EpisodeReference::Ordinal(2))
        );
        assert_eq!(
            resolve_episode_reference("上一批", &episodes).map(|item| item.seq),
            Some(1)
        );
        assert_eq!(
            resolve_episode_reference("第2批", &episodes).map(|item| item.seq),
            Some(2)
        );
        assert_eq!(
            resolve_episode_reference("second.ttl 讲了什么", &episodes).map(|item| item.seq),
            Some(2)
        );
        assert!(resolve_episode_reference("张三多大", &episodes).is_none());
    }

    #[test]
    fn metadata_quads_roundtrip_episode_json() {
        let original = episode(3, "cccc3333", "data/third.trig");
        let quads = episode_quads(&original).unwrap();
        assert!(quads
            .iter()
            .all(|quad| quad.graph_name == GraphName::NamedNode(iri(EPISODE_LEDGER_GRAPH))));
        let restored = parse_episode_graph(quads.into_iter());
        assert_eq!(restored, vec![original]);
    }

    #[test]
    fn ledger_states_when_rows_are_truncated() {
        let episodes: Vec<ImportEpisode> = (1..=(MAX_LEDGER_ROWS + 2))
            .map(|seq| episode(seq, &format!("id{:04}", seq), "batch.ttl"))
            .collect();
        let rendered = render_episode_ledger(&episodes);
        assert!(rendered.contains("共 52 批"));
        assert!(rendered.contains("仅显示最近 50 批"));
        assert!(rendered.contains("B52"));
        assert!(!rendered.contains("B1｜"));
    }
}

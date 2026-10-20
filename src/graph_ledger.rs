//! 图账本（Graph Ledger）：把会话规则与派生事实物化为 RDF 命名图。
//!
//! 设计立场：
//!   - **规则是知识**：教过的 Horn 规则落入 `kg:graphs/rules`，带稳定 id、
//!     规范逻辑形式、来源与置信度；跨会话、跨进程、可用 SPARQL 直接审计；
//!   - **派生是半成品**：前向饱和推出的新事实落入 `kg:graphs/derived`，
//!     每条自带规则来源、前提、推导轮次、置信度，以及物化戳
//!     （知识版本 revision + 规则集指纹 fingerprint），过期与否可判定；
//!   - **不抢回答路径**：账本节点使用独立命名空间 `kg:ledger/`，不使用 `kg:name`；
//!     全图扫描（快照/谓词画像）只看默认图，账本命名图与之天然隔离；
//!   - **幂等写入**：同步 = 清空命名图 + 全量重写，不推进知识版本；
//!   - **确定性**：id 与指纹由规范字符串哈希得出（FNV-1a），与插入顺序无关，
//!     可逐字节断言。

use oxigraph::model::*;
use serde::{Deserialize, Serialize};

/// 规则账本命名图。
pub const RULES_GRAPH: &str = "http://example.org/kg/graphs/rules";
/// 派生账本命名图。
pub const DERIVED_GRAPH: &str = "http://example.org/kg/graphs/derived";
/// 账本专用命名空间（与知识图谱 `kg:` 词面隔离）。
pub const LEDGER_NS: &str = "http://example.org/kg/ledger/";

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RULE_CLASS: &str = "http://example.org/kg/ledger/Rule";
const DERIVED_CLASS: &str = "http://example.org/kg/ledger/DerivedFact";
const STAMP_CLASS: &str = "http://example.org/kg/ledger/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/ledger/stamp";

const P_JSON: &str = "http://example.org/kg/ledger/json";
const P_HEAD: &str = "http://example.org/kg/ledger/head";
const P_BODY: &str = "http://example.org/kg/ledger/body";
const P_SOURCE: &str = "http://example.org/kg/ledger/source";
const P_CONFIDENCE: &str = "http://example.org/kg/ledger/confidence";
const P_ATOM: &str = "http://example.org/kg/ledger/atom";
const P_RULE_SOURCE: &str = "http://example.org/kg/ledger/ruleSource";
const P_PREMISE: &str = "http://example.org/kg/ledger/premise";
const P_DEPTH: &str = "http://example.org/kg/ledger/depth";
const P_REVISION: &str = "http://example.org/kg/ledger/revision";
const P_FINGERPRINT: &str = "http://example.org/kg/ledger/fingerprint";
const P_COUNT: &str = "http://example.org/kg/ledger/count";

type LedgerResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 账本项：逻辑项（变量 / 常量）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerTerm {
    pub name: String,
    pub is_var: bool,
}

/// 账本原子：谓词 + 参数。序列化为 JSON 存入账本，保证结构精确往返。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerAtom {
    pub predicate: String,
    pub terms: Vec<LedgerTerm>,
}

impl LedgerAtom {
    /// 逻辑形式显示：`师傅(?X, ?Y)`（变量带 `?` 前缀，与归结引擎一致）。
    pub fn display(&self) -> String {
        let terms: Vec<String> = self
            .terms
            .iter()
            .map(|term| {
                if term.is_var {
                    format!("?{}", term.name)
                } else {
                    term.name.clone()
                }
            })
            .collect();
        format!("{}({})", self.predicate, terms.join(", "))
    }

    fn canonical(&self) -> String {
        let terms: Vec<String> = self
            .terms
            .iter()
            .map(|term| {
                if term.is_var {
                    format!("?{}", term.name)
                } else {
                    format!("={}", term.name)
                }
            })
            .collect();
        format!("{}({})", self.predicate, terms.join(","))
    }
}

/// 规则账本条目：一条 Horn 子句 + 出处与置信度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleLedgerEntry {
    /// 稳定 id（由规范逻辑形式哈希；同一规则恒定，与插入顺序无关）。
    pub id: String,
    pub head: LedgerAtom,
    pub body: Vec<LedgerAtom>,
    pub confidence: f64,
    pub source: String,
}

impl RuleLedgerEntry {
    fn canonical(&self) -> String {
        let body: Vec<String> = self.body.iter().map(LedgerAtom::canonical).collect();
        format!("{}:-{}", self.head.canonical(), body.join("&"))
    }

    /// 规范中文规则显示：`只要「师傅(X, Y)」，就「老师(X, Y)」`。
    pub fn display_zh(&self) -> String {
        let body = self
            .body
            .iter()
            .map(LedgerAtom::display)
            .collect::<Vec<_>>()
            .join(" 且 ");
        format!("只要「{}」，就「{}」", body, self.head.display())
    }
}

/// 派生账本条目：一条前向饱和推出的新事实 + 完整出处。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedLedgerEntry {
    /// 稳定 id（由原子哈希）。
    pub id: String,
    pub atom: LedgerAtom,
    pub rule_source: String,
    pub premises: Vec<LedgerAtom>,
    pub depth: usize,
    pub confidence: f64,
    /// 物化时的知识版本（修订号）。
    pub revision: u64,
    /// 物化时的规则集指纹。
    pub fingerprint: String,
}

/// 规则账本读取结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleLedger {
    /// 账本是否写过（有账本戳或条目）。
    pub synced: bool,
    /// 账本戳上的规则集指纹。
    pub fingerprint: String,
    pub entries: Vec<RuleLedgerEntry>,
}

/// 派生账本读取结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedLedger {
    /// 账本是否物化过（有账本戳或条目）。
    pub materialized: bool,
    /// 物化时的知识版本。
    pub revision: u64,
    /// 物化时的规则集指纹。
    pub fingerprint: String,
    pub entries: Vec<DerivedLedgerEntry>,
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 稳定短哈希：FNV-1a 64 位 + 输入长度（消除同前缀截断歧义），全十六进制。
fn short_hash(text: &str) -> String {
    format!("{:016x}{:x}", fnv1a64(text.as_bytes()), text.len())
}

/// 规则条目的稳定 id。
pub fn rule_entry_id(head: &LedgerAtom, body: &[LedgerAtom]) -> String {
    let body: Vec<String> = body.iter().map(LedgerAtom::canonical).collect();
    short_hash(&format!("{}:-{}", head.canonical(), body.join("&")))
}

/// 派生条目的稳定 id。
pub fn derived_entry_id(atom: &LedgerAtom) -> String {
    short_hash(&atom.canonical())
}

/// 规则集指纹：对全部规则的规范形式 + 置信度排序后整体哈希。
///
/// 与条目顺序无关；规则增删改（头/体/置信度）都会改变指纹，
/// 从而让已物化的派生账本自动判定为「已过期」。
pub fn rule_fingerprint(entries: &[RuleLedgerEntry]) -> String {
    let mut canonicals: Vec<String> = entries
        .iter()
        .map(|entry| format!("{}@{:.6}", entry.canonical(), entry.confidence))
        .collect();
    canonicals.sort();
    short_hash(&canonicals.join("\n"))
}

/// 空规则集的固定指纹（未学过任何规则时的基准）。
pub fn empty_fingerprint() -> String {
    rule_fingerprint(&[])
}

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

/// 生成规则账本的全部四元组（含账本戳），写入指定的 `kg:graphs/rules`。
pub fn rule_quads(entries: &[RuleLedgerEntry]) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(RULES_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let rule_class = Term::NamedNode(named_node(RULE_CLASS)?);
    let mut quads: Vec<Quad> = Vec::with_capacity(entries.len() * 6 + 3);
    for entry in entries {
        let subject = named_node(&format!("{}rule/{}", LEDGER_NS, entry.id))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            rule_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(entry)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_HEAD,
            &entry.head.display(),
            &graph,
        )?);
        let body = entry
            .body
            .iter()
            .map(LedgerAtom::display)
            .collect::<Vec<_>>()
            .join(", ");
        quads.push(literal_quad(&subject, P_BODY, &body, &graph)?);
        quads.push(literal_quad(&subject, P_SOURCE, &entry.source, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_CONFIDENCE,
            &format!("{:.3}", entry.confidence),
            &graph,
        )?);
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
        &rule_fingerprint(entries),
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

/// 生成派生账本的全部四元组（含物化戳），写入指定的 `kg:graphs/derived`。
pub fn derived_quads(
    entries: &[DerivedLedgerEntry],
    revision: u64,
    fingerprint: &str,
) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(DERIVED_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let derived_class = Term::NamedNode(named_node(DERIVED_CLASS)?);
    let mut quads: Vec<Quad> = Vec::with_capacity(entries.len() * 9 + 4);
    for entry in entries {
        let subject = named_node(&format!("{}derived/{}", LEDGER_NS, entry.id))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            derived_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(entry)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_ATOM,
            &entry.atom.display(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_RULE_SOURCE,
            &entry.rule_source,
            &graph,
        )?);
        for premise in &entry.premises {
            quads.push(literal_quad(
                &subject,
                P_PREMISE,
                &premise.display(),
                &graph,
            )?);
        }
        quads.push(literal_quad(
            &subject,
            P_DEPTH,
            &entry.depth.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_CONFIDENCE,
            &format!("{:.3}", entry.confidence),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_REVISION,
            &entry.revision.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_FINGERPRINT,
            &entry.fingerprint,
            &graph,
        )?);
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
        P_REVISION,
        &revision.to_string(),
        &graph,
    )?);
    quads.push(literal_quad(&stamp, P_FINGERPRINT, fingerprint, &graph)?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &entries.len().to_string(),
        &graph,
    )?);
    Ok(quads)
}

/// 从规则命名图的四元组还原账本（按条目 id 升序，确定性）。
pub fn parse_rule_graph(quads: impl Iterator<Item = Quad>) -> RuleLedger {
    let mut synced = false;
    let mut fingerprint = empty_fingerprint();
    let mut entries: Vec<RuleLedgerEntry> = Vec::new();
    for quad in quads {
        let subject = match &quad.subject {
            Subject::NamedNode(node) => node.as_str(),
            _ => continue,
        };
        let predicate = quad.predicate.as_str();
        if subject == STAMP_SUBJECT {
            synced = true;
            if predicate == P_FINGERPRINT {
                if let Term::Literal(lit) = &quad.object {
                    fingerprint = lit.value().to_string();
                }
            }
            continue;
        }
        if predicate != P_JSON {
            continue;
        }
        if let Term::Literal(lit) = &quad.object {
            if let Ok(entry) = serde_json::from_str::<RuleLedgerEntry>(lit.value()) {
                entries.push(entry);
            }
        }
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries.dedup_by(|a, b| a.id == b.id);
    RuleLedger {
        synced,
        fingerprint,
        entries,
    }
}

/// 从派生命名图的四元组还原账本（按条目 id 升序，确定性）。
pub fn parse_derived_graph(quads: impl Iterator<Item = Quad>) -> DerivedLedger {
    let mut materialized = false;
    let mut revision: u64 = 0;
    let mut fingerprint = String::new();
    let mut entries: Vec<DerivedLedgerEntry> = Vec::new();
    for quad in quads {
        let subject = match &quad.subject {
            Subject::NamedNode(node) => node.as_str(),
            _ => continue,
        };
        let predicate = quad.predicate.as_str();
        if subject == STAMP_SUBJECT {
            materialized = true;
            match predicate {
                P_FINGERPRINT => {
                    if let Term::Literal(lit) = &quad.object {
                        fingerprint = lit.value().to_string();
                    }
                }
                P_REVISION => {
                    if let Term::Literal(lit) = &quad.object {
                        revision = lit.value().parse().unwrap_or(0);
                    }
                }
                _ => {}
            }
            continue;
        }
        if predicate != P_JSON {
            continue;
        }
        if let Term::Literal(lit) = &quad.object {
            if let Ok(entry) = serde_json::from_str::<DerivedLedgerEntry>(lit.value()) {
                entries.push(entry);
            }
        }
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries.dedup_by(|a, b| a.id == b.id);
    if !materialized && !entries.is_empty() {
        // 无账本戳的账本：从条目自身恢复物化戳
        materialized = true;
        revision = entries
            .iter()
            .map(|entry| entry.revision)
            .max()
            .unwrap_or(0);
        fingerprint = entries
            .first()
            .map(|entry| entry.fingerprint.clone())
            .unwrap_or_default();
    }
    DerivedLedger {
        materialized,
        revision,
        fingerprint,
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> LedgerTerm {
        LedgerTerm {
            name: name.to_string(),
            is_var: true,
        }
    }

    fn constant(name: &str) -> LedgerTerm {
        LedgerTerm {
            name: name.to_string(),
            is_var: false,
        }
    }

    fn rule(
        head: &str,
        args: Vec<LedgerTerm>,
        body: Vec<LedgerAtom>,
        source: &str,
    ) -> RuleLedgerEntry {
        let head = LedgerAtom {
            predicate: head.to_string(),
            terms: args,
        };
        RuleLedgerEntry {
            id: rule_entry_id(&head, &body),
            head,
            body,
            confidence: 0.95,
            source: source.to_string(),
        }
    }

    fn atom(predicate: &str, terms: Vec<LedgerTerm>) -> LedgerAtom {
        LedgerAtom {
            predicate: predicate.to_string(),
            terms,
        }
    }

    #[test]
    fn atom_display_matches_resolution_form() {
        let display = atom("师傅", vec![var("X"), var("Y")]).display();
        assert_eq!(display, "师傅(?X, ?Y)");
        assert_eq!(
            atom("是", vec![constant("张三"), constant("学生")]).display(),
            "是(张三, 学生)"
        );
    }

    #[test]
    fn entry_ids_are_stable_and_content_sensitive() {
        let body = vec![atom("师傅", vec![var("X"), var("Y")])];
        let first = rule("老师", vec![var("X"), var("Y")], body.clone(), "中文规则");
        let second = rule("老师", vec![var("X"), var("Y")], body.clone(), "中文规则");
        assert_eq!(first.id, second.id, "同规则 id 必须稳定");
        let reordered = rule("老师", vec![var("Y"), var("X")], body, "中文规则");
        assert_ne!(first.id, reordered.id, "参数顺序不同应为不同条目");
    }

    #[test]
    fn fingerprint_is_order_independent_and_content_sensitive() {
        let body_a = vec![atom("师傅", vec![var("X"), var("Y")])];
        let entry_a = rule("老师", vec![var("X"), var("Y")], body_a, "中文规则");
        let body_b = vec![atom("籍贯", vec![var("X"), var("Y")])];
        let entry_b = rule("同乡", vec![var("X"), var("Y")], body_b, "中文规则");

        let forward = rule_fingerprint(&[entry_a.clone(), entry_b.clone()]);
        let backward = rule_fingerprint(&[entry_b.clone(), entry_a.clone()]);
        assert_eq!(forward, backward, "指纹必须与插入顺序无关");
        assert_ne!(forward, empty_fingerprint());

        let mut changed = entry_a.clone();
        changed.confidence = 0.80;
        assert_ne!(
            rule_fingerprint(&[changed, entry_b]),
            forward,
            "置信度变化应改变指纹"
        );
    }

    #[test]
    fn rule_quads_roundtrip_with_stamp_and_json_payload() {
        let body = vec![atom("师傅", vec![var("X"), var("Y")])];
        let entry = rule("老师", vec![var("X"), var("Y")], body, "中文规则");
        let quads = rule_quads(&[entry.clone()]).unwrap();
        assert!(quads
            .iter()
            .all(|quad| quad.graph_name
                == GraphName::NamedNode(NamedNode::new(RULES_GRAPH).unwrap())));

        let ledger = parse_rule_graph(quads.into_iter());
        assert!(ledger.synced);
        assert_eq!(ledger.entries, vec![entry.clone()]);
        assert_eq!(ledger.fingerprint, rule_fingerprint(&[entry]));
    }

    #[test]
    fn derived_quads_roundtrip_with_provenance() {
        let head = atom("老师", vec![constant("张三"), constant("李四")]);
        let entry = DerivedLedgerEntry {
            id: derived_entry_id(&head),
            atom: head,
            rule_source: "中文规则".to_string(),
            premises: vec![atom("师傅", vec![constant("张三"), constant("李四")])],
            depth: 1,
            confidence: 0.95,
            revision: 7,
            fingerprint: "abc123".to_string(),
        };
        let quads = derived_quads(&[entry.clone()], 7, "abc123").unwrap();
        assert!(quads
            .iter()
            .all(|quad| quad.graph_name
                == GraphName::NamedNode(NamedNode::new(DERIVED_GRAPH).unwrap())));

        let ledger = parse_derived_graph(quads.into_iter());
        assert!(ledger.materialized);
        assert_eq!(ledger.revision, 7);
        assert_eq!(ledger.fingerprint, "abc123");
        assert_eq!(ledger.entries, vec![entry]);
    }

    #[test]
    fn empty_derived_ledger_keeps_materialization_stamp() {
        let quads = derived_quads(&[], 3, "fp").unwrap();
        let ledger = parse_derived_graph(quads.into_iter());
        assert!(ledger.materialized);
        assert!(ledger.entries.is_empty());
        assert_eq!(ledger.revision, 3);
        assert_eq!(ledger.fingerprint, "fp");
    }

    #[test]
    fn parse_dedupes_by_id_and_sorts() {
        let body = vec![atom("师傅", vec![var("X"), var("Y")])];
        let entry_a = rule("老师", vec![var("X"), var("Y")], body.clone(), "中文规则");
        let entry_b = rule("师祖", vec![var("X"), var("Y")], body, "中文规则");
        let mut quads = rule_quads(&[entry_b.clone(), entry_a.clone()]).unwrap();
        quads.extend(rule_quads(&[entry_a.clone()]).unwrap().into_iter().take(6));
        let ledger = parse_rule_graph(quads.into_iter());
        assert_eq!(ledger.entries.len(), 2, "同 id 重复四元组应只还原为一条");
        let mut expected = vec![entry_a, entry_b];
        expected.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(ledger.entries, expected);
    }
}

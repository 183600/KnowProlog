//! PRV：数据溯源与冲突证书（Provenance & Conflict Certificates）。
//!
//! 现有系统里的「来源」几乎都是类别（`ProofSource::Fact`、`EdgeSource::Observed`、
//! 谓词画像上的 `sources` 标签）——它们回答「这是哪一类」，不回答「这是哪一条」。
//! 结果是：质量护照的溯源维度由「谓词是否被声明」决定，与事实本身无关；证明树
//! 走到事实叶子就断了。本模块补上三件缺一不可的东西：
//!
//! 1. **来源有身份**：[`SourceId`] = 类别 + 定位符（文件/批次/轮次）+ 内容摘要。
//!    同一份内容重复登记得到同一身份，不同内容即使定位符相同也不会被合并。
//! 2. **来源能传递**：[`Support`] 是一棵支撑树，叶子是事实，节点是规则；每个
//!    事实各自携带自己的来源集合。派生事实的来源来自规则账本，而不是谓词画像。
//! 3. **来源能求交/求差**：[`ConflictCertificate`] 把两个冲突结论的支撑树求交
//!    得到共同前提，求差得到各自独有前提；共同前提不是肇事者，独有前提才是候选
//!    肇事者。当任一侧的叶子没有登记来源时，证书显式进入**归因不足**状态，而不是
//!    随便挑一边。
//!
//! 冲突有两个时间轴：已见证（闭包里两个结论都在）与潜在（两条规则的头部可合一、
//! 体部可同时满足，但图上还没有实例）。后者是本模块把输入从「闭包」换成「规则对」
//! 之后才可判定的对象。规则是 Datalog（无函数符号），所以功能冲突只在头部含常量
//! 的规则对之间报告；宁可少报，也不把 `P(x, z)` 与 `P(x, z')` 误报成冲突。
//!
//! 冲突条数是错误的度量：一个功能槽 3 个值时两两计数是 3，但修复只需删 2 条；
//! 计数经 `clamp` 后连序都丢了。本模块给出**最小撤回集**（命中所有证书支撑集的
//! 最小事实集合）：可计算、可指认、可执行。候选 ≤ [`MAX_EXACT_CANDIDATES`] 时
//! 精确枚举，超过时退化为贪心并显式标记 `exact = false`，绝不假装最优。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex, ContextImage};
use crate::graph_ledger::{DerivedLedger, LedgerAtom, RuleLedger};
use crate::import_journal::ImportJournal;
use crate::knowledge_fusion::FusionLedger;
use crate::predicate_registry::ImportAction;
use crate::predicate_registry::PredicateRegistry;
use crate::resolution_engine::{unify_atoms, Atom, Clause, Substitution, Term};
use crate::type_system::TypeSystem;

/// 来源台账在 ContextImage 信封中的分节名。
pub const PROVENANCE_SECTION: &str = "fact_provenance";
/// 精确最小命中集的最大候选事实数；超过则贪心并标记非精确。
pub const MAX_EXACT_CANDIDATES: usize = 16;
/// 支撑树展开的最大深度（防止派生账本出现环时无限递归）。
pub const MAX_SUPPORT_DEPTH: usize = 16;
/// 报告中最多逐条展开的未登记事实数。
pub const REPORT_LIST_CAP: usize = 8;
/// 类型断言默认使用的谓词名（项目内 `type` / `类型` / `rdf:type` 都有使用）。
pub const DEFAULT_TYPE_PREDICATES: [&str; 3] = ["type", "类型", "rdf:type"];

// ---------------------------------------------------------------------------
// 来源身份
// ---------------------------------------------------------------------------

/// 来源类别。类别只用于分组与措辞，不承担身份。
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// 配置/规则文件中的声明。
    Config,
    /// 文件或批量导入。
    Import,
    /// 人工确认或人工录入。
    Manual,
    /// 对话轮次中的断言。
    Conversation,
    /// Horn 规则（规则账本中的来源）。
    Rule,
    /// 由规则派生。
    Derived,
    /// 未登记来源（诚实缺省，不得当作「可信」）。
    #[default]
    Unknown,
}

impl SourceKind {
    /// 解析人读/配置写法；未知写法返回 `None`（调用方应拒绝而不是当成 Unknown）。
    pub fn parse(raw: &str) -> Option<SourceKind> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "config" | "配置" | "配置声明" => Some(SourceKind::Config),
            "import" | "导入" => Some(SourceKind::Import),
            "manual" | "人工" | "人工确认" => Some(SourceKind::Manual),
            "conversation" | "对话" | "对话录入" => Some(SourceKind::Conversation),
            "rule" | "规则" => Some(SourceKind::Rule),
            "derived" | "派生" => Some(SourceKind::Derived),
            "unknown" | "未登记" => Some(SourceKind::Unknown),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Config => "配置声明",
            SourceKind::Import => "导入",
            SourceKind::Manual => "人工确认",
            SourceKind::Conversation => "对话",
            SourceKind::Rule => "规则",
            SourceKind::Derived => "派生",
            SourceKind::Unknown => "未登记",
        }
    }
}

/// 一条来源的**身份**：类别 + 定位符 + 内容摘要。
///
/// - `locator` 回答「来自哪一批/哪个文件/第几轮」；
/// - `digest` 回答「是不是同一份内容」——定位符相同但内容不同会被区分；
/// - 只有 `kind != Unknown` 且定位符或摘要非空时才算已登记。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceId {
    pub kind: SourceKind,
    #[serde(default)]
    pub locator: String,
    #[serde(default)]
    pub digest: String,
}

impl SourceId {
    /// 构造一个有身份的来源；`content` 用于计算内容摘要。
    pub fn new(kind: SourceKind, locator: &str, content: &str) -> Self {
        SourceId {
            kind,
            locator: locator.trim().to_string(),
            digest: if content.is_empty() {
                String::new()
            } else {
                fnv1a_hex(content.as_bytes())
            },
        }
    }

    /// 只有内容摘要、没有定位符的来源（例如规则账本只保留了来源标签）。
    pub fn anonymous(kind: SourceKind, content: &str) -> Self {
        Self::new(kind, "", content)
    }

    /// 未登记来源的显式表示。
    pub fn unknown() -> Self {
        SourceId {
            kind: SourceKind::Unknown,
            locator: String::new(),
            digest: String::new(),
        }
    }

    /// 是否登记了可引用的身份（`Unknown` 永远不算）。
    pub fn is_registered(&self) -> bool {
        self.kind != SourceKind::Unknown && (!self.locator.is_empty() || !self.digest.is_empty())
    }

    /// 人读形式：「导入:2002-体测表.csv#a1b2c3d4e5f6」。
    pub fn render(&self) -> String {
        if !self.is_registered() {
            return "来源未登记".to_string();
        }
        let short: String = self.digest.chars().take(12).collect();
        match (self.locator.is_empty(), short.is_empty()) {
            (false, false) => format!("{}:{}#{}", self.kind.label(), self.locator, short),
            (false, true) => format!("{}:{}", self.kind.label(), self.locator),
            (true, false) => format!("{}:#{}", self.kind.label(), short),
            (true, true) => self.kind.label().to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// 事实引用与支撑树
// ---------------------------------------------------------------------------

/// 一条事实的规范引用（主语 + 谓词 + 取值）。属性与关系共用同一形状：
/// 属性事实的取值是字面量，关系事实的取值是实体名。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactRef {
    pub subject: String,
    pub predicate: String,
    pub value: String,
}

impl FactRef {
    pub fn new(subject: &str, predicate: &str, value: &str) -> Self {
        FactRef {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            value: value.to_string(),
        }
    }

    pub fn relation(subject: &str, predicate: &str, object: &str) -> Self {
        Self::new(subject, predicate, object)
    }

    pub fn attribute(subject: &str, predicate: &str, value: &str) -> Self {
        Self::new(subject, predicate, value)
    }

    /// 类型断言（统一用 `type` 作为谓词键）。
    pub fn type_assertion(entity: &str, type_name: &str) -> Self {
        Self::new(entity, "type", type_name)
    }

    pub fn key(&self) -> String {
        format!(
            "{}\u{1}{}\u{1}{}",
            self.subject, self.predicate, self.value
        )
    }

    pub fn display(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.value)
    }
}

/// 规则出处：稳定 id + 该规则自己的来源 + 置信度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleOrigin {
    pub id: String,
    pub source: SourceId,
    pub confidence: f64,
}

/// 支撑树节点：结论事实 + 本节点来源 + 可选规则出处 + 前提支撑。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Support {
    pub fact: FactRef,
    #[serde(default)]
    pub sources: BTreeSet<SourceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<RuleOrigin>,
    #[serde(default)]
    pub premises: Vec<Support>,
}

impl Support {
    /// 事实叶子：没有规则前提，来源就是台账里登记的事实来源。
    pub fn fact(fact: FactRef, sources: BTreeSet<SourceId>) -> Self {
        Support {
            fact,
            sources,
            rule: None,
            premises: Vec::new(),
        }
    }

    /// 由规则推出的节点：来源是规则来源，前提是规则体部的支撑。
    pub fn derived(
        fact: FactRef,
        source: SourceId,
        rule_id: &str,
        confidence: f64,
        premises: Vec<Support>,
    ) -> Self {
        Support {
            fact,
            sources: if source.is_registered() {
                [source.clone()].into_iter().collect()
            } else {
                BTreeSet::new()
            },
            rule: Some(RuleOrigin {
                id: rule_id.to_string(),
                source,
                confidence,
            }),
            premises,
        }
    }

    pub fn is_rule(&self) -> bool {
        self.rule.is_some()
    }

    /// 本节点及其全部前提的来源并集。
    pub fn all_sources(&self) -> BTreeSet<SourceId> {
        let mut out = self.sources.clone();
        for premise in &self.premises {
            out.extend(premise.all_sources());
        }
        out
    }

    /// 支撑树的事实叶子（递归；规则节点的结论本身不是叶子）。
    pub fn leaf_facts(&self) -> BTreeSet<FactRef> {
        let mut out = BTreeSet::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves(&self, out: &mut BTreeSet<FactRef>) {
        if self.premises.is_empty() {
            out.insert(self.fact.clone());
            return;
        }
        for premise in &self.premises {
            premise.collect_leaves(out);
        }
    }

    /// 归因/撤回使用的前提集：规则节点取其叶子，事实节点取自身。
    pub fn premise_facts(&self) -> BTreeSet<FactRef> {
        self.leaf_facts()
    }

    /// 叶子中没有登记来源的事实（空表示该侧归因有据）。
    pub fn unregistered_leaves(&self) -> Vec<FactRef> {
        let mut out = Vec::new();
        self.collect_unregistered(&mut out);
        out.sort();
        out.dedup();
        out
    }

    fn collect_unregistered(&self, out: &mut Vec<FactRef>) {
        if self.premises.is_empty() {
            if self.sources.iter().all(|source| !source.is_registered()) {
                out.push(self.fact.clone());
            }
            return;
        }
        for premise in &self.premises {
            premise.collect_unregistered(out);
        }
    }

    /// 某个事实在本支撑树里登记的全部来源（找不到时为空）。
    pub fn sources_of(&self, fact: &FactRef) -> BTreeSet<SourceId> {
        if &self.fact == fact {
            return self.sources.clone();
        }
        let mut out = BTreeSet::new();
        for premise in &self.premises {
            out.extend(premise.sources_of(fact));
        }
        out
    }

    pub fn depth(&self) -> usize {
        1 + self
            .premises
            .iter()
            .map(Support::depth)
            .max()
            .unwrap_or(0)
    }

    fn render_lines(&self, indent: usize, out: &mut Vec<String>) {
        let pad = " ".repeat(indent);
        let mut line = format!("{}- {}", pad, self.fact.display());
        if let Some(rule) = &self.rule {
            line.push_str(&format!("（规则 {}）", rule.id));
        }
        let sources: Vec<String> = self
            .sources
            .iter()
            .filter(|source| source.is_registered())
            .map(SourceId::render)
            .collect();
        if sources.is_empty() {
            line.push_str("［来源未登记］");
        } else {
            line.push_str(&format!("［{}］", sources.join("、")));
        }
        out.push(line);
        for premise in &self.premises {
            premise.render_lines(indent + 2, out);
        }
    }
}

// ---------------------------------------------------------------------------
// 来源台账（可持久化）
// ---------------------------------------------------------------------------

/// 台账中的一条：事实 + 它的全部来源（同一事实多来源时保留全部）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    pub fact: FactRef,
    #[serde(default)]
    pub sources: BTreeSet<SourceId>,
}

/// 事实 → 来源集合的登记台账。
///
/// 这是「事实级来源」的唯一写入点：谓词画像里的 `sources` 只是谓词属性，
/// 不能替代本台账。台账有界策略由调用方决定（本结构只负责确定性）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceLedger {
    #[serde(default)]
    entries: BTreeMap<String, ProvenanceEntry>,
}

/// 来源台账加载结果：缺失 / 已加载 / 损坏（损坏时调用方以空台账降级）。
#[derive(Debug)]
pub enum ProvenanceLoad {
    Missing,
    Loaded(ProvenanceLedger),
    Damaged(String),
}

impl ProvenanceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 为一条事实登记来源；返回是否新增了来源（重复登记幂等）。
    pub fn record(&mut self, fact: FactRef, source: SourceId) -> bool {
        let entry = self.entries.entry(fact.key()).or_insert_with(|| ProvenanceEntry {
            fact: fact.clone(),
            sources: BTreeSet::new(),
        });
        entry.sources.insert(source)
    }

    pub fn record_many(&mut self, facts: &[FactRef], source: SourceId) -> usize {
        facts
            .iter()
            .filter(|fact| self.record((*fact).clone(), source.clone()))
            .count()
    }

    pub fn sources(&self, fact: &FactRef) -> BTreeSet<SourceId> {
        self.entries
            .get(&fact.key())
            .map(|entry| entry.sources.clone())
            .unwrap_or_default()
    }

    pub fn get(&self, fact: &FactRef) -> Option<&ProvenanceEntry> {
        self.entries.get(&fact.key())
    }

    pub fn entries(&self) -> impl Iterator<Item = &ProvenanceEntry> {
        self.entries.values()
    }

    /// 只保留一条事实，供没有派生信息时构造事实支撑。
    pub fn support(&self, fact: FactRef) -> Support {
        Support::fact(fact.clone(), self.sources(&fact))
    }

    /// 合并另一份台账（同一事实的多来源求并集）。
    pub fn merge(&mut self, other: &ProvenanceLedger) {
        for entry in other.entries.values() {
            for source in &entry.sources {
                self.record(entry.fact.clone(), source.clone());
            }
        }
    }

    /// 覆盖率：给定事实表中登记了来源的事实比例 + 未登记清单。
    pub fn coverage(&self, facts: &[FactRef]) -> Coverage {
        let unique: BTreeSet<&FactRef> = facts.iter().collect();
        let total = unique.len();
        let unregistered: Vec<FactRef> = unique
            .iter()
            .filter(|fact| {
                self.sources(fact)
                    .iter()
                    .all(|source| !source.is_registered())
            })
            .map(|fact| (*fact).clone())
            .collect();
        Coverage {
            total,
            registered: total.saturating_sub(unregistered.len()),
            ratio: if total == 0 {
                0.0
            } else {
                (total - unregistered.len()) as f64 / total as f64
            },
            unregistered,
        }
    }

    /// 以 ContextImage 信封原子落盘（规范字节、逐节自检）。
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let entries: Vec<&ProvenanceEntry> = self.entries.values().collect();
        let value = serde_json::to_value(&entries)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        let mut image = ContextImage::new();
        image.set_section(PROVENANCE_SECTION, value);
        image.seal();
        image.save_atomic(path)
    }

    /// 读取台账：缺失 → Missing；损坏 → Damaged（调用方以空台账降级，绝不误信）。
    pub fn load(path: &str) -> ProvenanceLoad {
        if !std::path::Path::new(path).exists() {
            return ProvenanceLoad::Missing;
        }
        let image = match ContextImage::load(path) {
            Ok(image) => image,
            Err(error) => return ProvenanceLoad::Damaged(error),
        };
        let audit = image.verify();
        if audit.global_damaged {
            return ProvenanceLoad::Damaged("来源台账全局哈希不匹配".to_string());
        }
        if audit
            .damaged_sections
            .iter()
            .any(|section| section == PROVENANCE_SECTION)
        {
            return ProvenanceLoad::Damaged("来源台账分节哈希不匹配".to_string());
        }
        let Some(value) = image.section(PROVENANCE_SECTION) else {
            return ProvenanceLoad::Loaded(ProvenanceLedger::new());
        };
        match serde_json::from_value::<Vec<ProvenanceEntry>>(value.clone()) {
            Ok(entries) => {
                let mut ledger = ProvenanceLedger::new();
                for entry in entries {
                    for source in entry.sources {
                        ledger.record(entry.fact.clone(), source);
                    }
                }
                ProvenanceLoad::Loaded(ledger)
            }
            Err(error) => ProvenanceLoad::Damaged(format!("来源台账解析失败：{}", error)),
        }
    }

    /// 从跨会话导入日志回填事实级来源。
    ///
    /// 日志是**有界**的（默认最近 32 笔），所以覆盖率必须如实报：更早的导入
    /// 无从回溯，不能默认 100%。每笔导入的身份 = 序号 + 回执内容摘要；同一
    /// 事实被多笔导入写入时保留多个来源（join 语义）。
    pub fn seed_from_import_journal(&mut self, journal: &ImportJournal) -> usize {
        let mut recorded = 0usize;
        for (index, receipt) in journal.entries().iter().enumerate() {
            let content = serde_json::to_string(receipt).unwrap_or_default();
            let source = SourceId::new(
                SourceKind::Import,
                &format!("导入日志#{}", index + 1),
                &content,
            );
            for action in &receipt.actions {
                let fact = match action {
                    ImportAction::Named {
                        subject,
                        predicate,
                        object,
                    } => FactRef::new(
                        &local_from_iri(subject),
                        &predicate_from_iri(predicate),
                        &local_from_iri(object),
                    ),
                    ImportAction::Literal {
                        subject,
                        predicate,
                        value,
                    } => FactRef::new(
                        &local_from_iri(subject),
                        &predicate_from_iri(predicate),
                        value,
                    ),
                };
                self.record(fact, source.clone());
                recorded += 1;
            }
        }
        recorded
    }

    /// 从融合账本的命名空间断言回填来源。
    ///
    /// 每个命名空间就是一批来源：定位符取「命名空间标签 @ 证据指针」，
    /// 内容摘要取断言稳定 id。调用方决定是否把命名空间事实纳入冲突检测
    /// （默认图与隔离源图是两套视图，不能静默混在一起）。
    pub fn seed_from_fusion_ledger(&mut self, ledger: &FusionLedger) -> usize {
        let mut recorded = 0usize;
        for assertion in &ledger.assertions {
            let label = ledger
                .namespace(&assertion.namespace)
                .map(|record| record.label.clone())
                .unwrap_or_else(|| assertion.namespace.clone());
            let locator = if assertion.source.is_empty() {
                label
            } else {
                format!("{}@{}", label, assertion.source)
            };
            let source = SourceId::new(SourceKind::Import, &locator, &assertion.id);
            self.record(
                FactRef::new(&assertion.subject, &assertion.predicate, &assertion.object),
                source,
            );
            recorded += 1;
        }
        recorded
    }

    /// 规范 JSON（确定性摘要/审计用）。
    pub fn to_json(&self) -> String {
        let entries: Vec<&ProvenanceEntry> = self.entries.values().collect();
        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn digest(&self) -> String {
        let entries: Vec<&ProvenanceEntry> = self.entries.values().collect();
        fnv1a_hex(canonical_json(&serde_json::json!(entries)).as_bytes())
    }
}

/// 解析来源台账 TSV：`主语\t谓词\t取值\t类别\t定位符[\t内容]`。
///
/// 空行与 `#` 注释跳过；类别只接受 [`SourceKind::parse`] 认识的写法，且拒绝
/// `unknown`（登记一个「未登记来源」没有意义）。内容列缺省时只用定位符，
/// 此时同一批次的所有事实共享同一来源身份——这正是「批次号」该有的语义。
pub fn parse_provenance_tsv(text: &str) -> Result<Vec<(FactRef, SourceId)>, String> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 || fields.len() > 6 {
            return Err(format!(
                "第 {} 行应为 5-6 个制表符分隔字段（主语/谓词/取值/类别/定位符[/内容]），实际 {} 个",
                index + 1,
                fields.len()
            ));
        }
        if fields[0].trim().is_empty() || fields[1].trim().is_empty() {
            return Err(format!("第 {} 行主语或谓词为空", index + 1));
        }
        let kind = SourceKind::parse(fields[3]).ok_or_else(|| {
            format!(
                "第 {} 行类别「{}」无法识别（import/manual/conversation/config/rule/derived）",
                index + 1,
                fields[3]
            )
        })?;
        if kind == SourceKind::Unknown {
            return Err(format!(
                "第 {} 行类别不能是 unknown：登记「未登记来源」没有意义",
                index + 1
            ));
        }
        let content = fields.get(5).copied().unwrap_or("");
        out.push((
            FactRef::new(fields[0], fields[1], fields[2]),
            SourceId::new(kind, fields[4], content),
        ));
    }
    Ok(out)
}

/// 来源覆盖率（诚实报数：未登记就写未登记，不默认 100%）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coverage {
    pub total: usize,
    pub registered: usize,
    pub ratio: f64,
    pub unregistered: Vec<FactRef>,
}

impl Coverage {
    pub fn render_zh(&self) -> String {
        let mut line = format!(
            "来源覆盖：{}/{} 条事实已登记（{:.1}%）。",
            self.registered,
            self.total,
            self.ratio * 100.0
        );
        if !self.unregistered.is_empty() {
            let sample: Vec<String> = self
                .unregistered
                .iter()
                .take(REPORT_LIST_CAP)
                .map(FactRef::display)
                .collect();
            line.push_str(&format!(
                "未登记：{}{}",
                sample.join("、"),
                if self.unregistered.len() > REPORT_LIST_CAP {
                    format!(" 等 {} 条", self.unregistered.len())
                } else {
                    String::new()
                }
            ));
        }
        line
    }
}

// ---------------------------------------------------------------------------
// 冲突点与冲突证书
// ---------------------------------------------------------------------------

/// 冲突的「点」：功能槽多值，或互斥类型共存。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConflictPoint {
    Functional {
        subject: String,
        predicate: String,
        left_value: String,
        right_value: String,
    },
    DisjointTypes {
        entity: String,
        left_type: String,
        right_type: String,
    },
}

impl ConflictPoint {
    pub fn key(&self) -> String {
        match self {
            ConflictPoint::Functional {
                subject,
                predicate,
                left_value,
                right_value,
            } => format!(
                "functional\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
                subject, predicate, left_value, right_value
            ),
            ConflictPoint::DisjointTypes {
                entity,
                left_type,
                right_type,
            } => format!(
                "disjoint\u{1}{}\u{1}{}\u{1}{}",
                entity, left_type, right_type
            ),
        }
    }

    /// 冲突点所在的槽（同一槽的多个取值只算一个冲突点）。
    ///
    /// 一个功能槽出现 3 个值时会生成 3 张两两证书，但只有 1 个冲突点、
    /// 最少删 2 条事实即可修复。计数与代价必须分开报告。
    pub fn slot_key(&self) -> String {
        match self {
            ConflictPoint::Functional {
                subject,
                predicate,
                ..
            } => format!("functional\u{1}{}\u{1}{}", subject, predicate),
            ConflictPoint::DisjointTypes { entity, .. } => {
                format!("disjoint\u{1}{}", entity)
            }
        }
    }

    pub fn left_fact(&self) -> FactRef {
        match self {
            ConflictPoint::Functional {
                subject,
                predicate,
                left_value,
                ..
            } => FactRef::new(subject, predicate, left_value),
            ConflictPoint::DisjointTypes {
                entity,
                left_type,
                ..
            } => FactRef::type_assertion(entity, left_type),
        }
    }

    pub fn right_fact(&self) -> FactRef {
        match self {
            ConflictPoint::Functional {
                subject,
                predicate,
                right_value,
                ..
            } => FactRef::new(subject, predicate, right_value),
            ConflictPoint::DisjointTypes {
                entity,
                right_type,
                ..
            } => FactRef::type_assertion(entity, right_type),
        }
    }

    pub fn render_zh(&self) -> String {
        match self {
            ConflictPoint::Functional {
                subject,
                predicate,
                left_value,
                right_value,
            } => format!(
                "功能槽「{}」的主语「{}」有两个取值：「{}」与「{}」",
                predicate, subject, left_value, right_value
            ),
            ConflictPoint::DisjointTypes {
                entity,
                left_type,
                right_type,
            } => format!(
                "实体「{}」同时被断言为互斥类型「{}」与「{}」",
                entity, left_type, right_type
            ),
        }
    }
}

/// 归因结论：冲突是谁的错。
///
/// 注意 `Candidates` 只表示「有候选肇事前提」，不表示已裁决；`SameOrigin` 与
/// `Insufficient` 是两种必须显式说出口的失败态，不能静默择一。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attribution {
    /// 两侧各有独有前提：独有前提是候选肇事者，共同前提不是。
    Candidates {
        left_only: Vec<FactRef>,
        right_only: Vec<FactRef>,
    },
    /// 双方共享全部前提：同一来源自相矛盾，不能怪某一条独有记录。
    SameOrigin { sources: Vec<SourceId> },
    /// 至少一侧的叶子没有登记来源：有冲突，但无法归因。
    Insufficient {
        left_missing: Vec<FactRef>,
        right_missing: Vec<FactRef>,
    },
    /// 双方都有来源、但没有独有前提，也没有可指认的共同来源。
    Unattributed { reason: String },
}

impl Attribution {
    pub fn render_zh(&self) -> String {
        match self {
            Attribution::Candidates {
                left_only,
                right_only,
            } => {
                let mut parts = Vec::new();
                if !left_only.is_empty() {
                    parts.push(format!(
                        "左侧独有：{}",
                        left_only.iter().map(FactRef::display).collect::<Vec<_>>().join("、")
                    ));
                }
                if !right_only.is_empty() {
                    parts.push(format!(
                        "右侧独有：{}",
                        right_only.iter().map(FactRef::display).collect::<Vec<_>>().join("、")
                    ));
                }
                format!("可归因（候选肇事前提）：{}", parts.join("；"))
            }
            Attribution::SameOrigin { sources } => format!(
                "同源矛盾：双方共享全部前提（{}），不是两条独立记录打架",
                if sources.is_empty() {
                    "来源未登记".to_string()
                } else {
                    sources.iter().map(SourceId::render).collect::<Vec<_>>().join("、")
                }
            ),
            Attribution::Insufficient {
                left_missing,
                right_missing,
            } => {
                let mut parts = Vec::new();
                if !left_missing.is_empty() {
                    parts.push(format!(
                        "左侧未登记：{}",
                        left_missing.iter().map(FactRef::display).collect::<Vec<_>>().join("、")
                    ));
                }
                if !right_missing.is_empty() {
                    parts.push(format!(
                        "右侧未登记：{}",
                        right_missing.iter().map(FactRef::display).collect::<Vec<_>>().join("、")
                    ));
                }
                format!("有冲突，但无法归因：{}", parts.join("；"))
            }
            Attribution::Unattributed { reason } => format!("无法归因：{}", reason),
        }
    }
}

/// 最小撤回集：让所有证书的支撑集同时被命中的最小事实集合。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetractionPlan {
    pub facts: Vec<FactRef>,
    /// 是否为精确最小集（候选超过上限时退化为贪心并标记 false）。
    pub exact: bool,
    /// 参与命中集计算的候选事实数。
    pub candidates: usize,
    /// 被该撤回集解决的证书数。
    pub resolved: usize,
    /// 无法通过撤回解决（支撑集为空）的证书数。
    pub unresolved: usize,
}

impl RetractionPlan {
    pub fn render_zh(&self) -> String {
        if self.resolved == 0 && self.unresolved == 0 {
            return "最小撤回集：无需撤回（没有冲突证书）。".to_string();
        }
        let mut line = if self.facts.is_empty() {
            "最小撤回集：空（没有可撤回的支撑事实）。".to_string()
        } else {
            format!(
                "最小撤回集（{} 条，{}）：{}",
                self.facts.len(),
                if self.exact { "精确" } else { "贪心，非最优" },
                self.facts
                    .iter()
                    .map(FactRef::display)
                    .collect::<Vec<_>>()
                    .join("、")
            )
        };
        line.push_str(&format!(
            "；解决 {}/{} 个证书",
            self.resolved,
            self.resolved + self.unresolved
        ));
        if self.unresolved > 0 {
            line.push_str(&format!(
                "，{} 个证书没有可撤回的支撑集（来源缺失）",
                self.unresolved
            ));
        }
        line.push('。');
        line
    }
}

/// 冲突证书：两个结论 + 各自支撑树 + 共同/独有前提 + 归因 + 最小撤回集。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictCertificate {
    pub id: String,
    pub point: ConflictPoint,
    pub left: Support,
    pub right: Support,
    /// 两棵支撑树的事实叶子交集（共同前提，不是肇事者）。
    pub common: Vec<FactRef>,
    /// 左侧独有的前提（候选肇事者）。
    pub left_only: Vec<FactRef>,
    /// 右侧独有的前提（候选肇事者）。
    pub right_only: Vec<FactRef>,
    pub attribution: Attribution,
    /// 已见证（闭包中有实例）还是潜在（规则对可冲突但暂无实例）。
    pub latent: bool,
    /// 由报告统一计算后回填；单张证书构造时为空计划。
    #[serde(default)]
    pub retraction: RetractionPlan,
}

impl ConflictCertificate {
    pub fn new(point: ConflictPoint, left: Support, right: Support, latent: bool) -> Self {
        let left_facts = left.premise_facts();
        let right_facts = right.premise_facts();
        let common: Vec<FactRef> = left_facts.intersection(&right_facts).cloned().collect();
        let left_only: Vec<FactRef> = left_facts.difference(&right_facts).cloned().collect();
        let right_only: Vec<FactRef> = right_facts.difference(&left_facts).cloned().collect();
        let attribution = attribute(&left, &right, &left_only, &right_only);
        let id = certificate_id(&point, &left, &right, latent);
        ConflictCertificate {
            id,
            point,
            left,
            right,
            common,
            left_only,
            right_only,
            attribution,
            latent,
            retraction: RetractionPlan::default(),
        }
    }

    /// 该证书的支撑事实并集（撤回命中集的输入）。
    pub fn support_facts(&self) -> BTreeSet<FactRef> {
        let mut out = self.left.premise_facts();
        out.extend(self.right.premise_facts());
        out
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "冲突证书 {}（{}）：{}。",
            self.id,
            if self.latent { "潜在" } else { "已见证" },
            self.point.render_zh()
        )];
        lines.push("  左支撑：".to_string());
        self.left.render_lines(4, &mut lines);
        lines.push("  右支撑：".to_string());
        self.right.render_lines(4, &mut lines);
        if !self.common.is_empty() {
            lines.push(format!(
                "  共同前提（不是肇事者）：{}",
                self.common
                    .iter()
                    .map(FactRef::display)
                    .collect::<Vec<_>>()
                    .join("、")
            ));
        }
        lines.push(format!("  {}", self.attribution.render_zh()));
        lines.join("\n")
    }
}

fn certificate_id(
    point: &ConflictPoint,
    left: &Support,
    right: &Support,
    latent: bool,
) -> String {
    let left_facts: Vec<String> = left.premise_facts().iter().map(FactRef::key).collect();
    let right_facts: Vec<String> = right.premise_facts().iter().map(FactRef::key).collect();
    let value = serde_json::json!({
        "point": point.key(),
        "left": left_facts,
        "right": right_facts,
        "latent": latent,
    });
    fnv1a_hex(canonical_json(&value).as_bytes())
}

fn attribute(
    left: &Support,
    right: &Support,
    left_only: &[FactRef],
    right_only: &[FactRef],
) -> Attribution {
    let left_missing = left.unregistered_leaves();
    let right_missing = right.unregistered_leaves();
    if !left_missing.is_empty() || !right_missing.is_empty() {
        return Attribution::Insufficient {
            left_missing,
            right_missing,
        };
    }
    if left_only.is_empty() && right_only.is_empty() {
        let left_sources = left.all_sources();
        let right_sources = right.all_sources();
        let mut sources: Vec<SourceId> = left_sources.intersection(&right_sources).cloned().collect();
        sources.retain(SourceId::is_registered);
        if !sources.is_empty() {
            return Attribution::SameOrigin { sources };
        }
        return Attribution::Unattributed {
            reason: "双方共享全部前提，但共同来源为空".to_string(),
        };
    }
    Attribution::Candidates {
        left_only: left_only.to_vec(),
        right_only: right_only.to_vec(),
    }
}

/// 一次冲突分析的总报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictReport {
    pub certificates: Vec<ConflictCertificate>,
    pub retraction: RetractionPlan,
    pub coverage: Coverage,
}

impl ConflictReport {
    /// 去重、稳定排序、统一计算最小撤回集并回填每张证书。
    pub fn new(mut certificates: Vec<ConflictCertificate>, coverage: Coverage) -> Self {
        certificates.sort_by(|left, right| {
            left.point
                .key()
                .cmp(&right.point.key())
                .then_with(|| left.id.cmp(&right.id))
        });
        certificates.dedup_by(|left, right| left.id == right.id);
        let retraction = plan_retraction(&certificates);
        for certificate in &mut certificates {
            certificate.retraction = retraction.clone();
        }
        ConflictReport {
            certificates,
            retraction,
            coverage,
        }
    }

    /// 去重后的冲突点数（与两两证书数分开）。
    pub fn conflict_points(&self) -> usize {
        self.certificates
            .iter()
            .map(|certificate| certificate.point.slot_key())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn witnessed(&self) -> usize {
        self.certificates.iter().filter(|cert| !cert.latent).count()
    }

    pub fn latent(&self) -> usize {
        self.certificates.iter().filter(|cert| cert.latent).count()
    }

    pub fn insufficient_attribution(&self) -> usize {
        self.certificates
            .iter()
            .filter(|cert| matches!(cert.attribution, Attribution::Insufficient { .. }))
            .count()
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "冲突证书：{} 张，覆盖 {} 个冲突点（已见证 {} / 潜在 {}），其中归因不足 {} 张。",
            self.certificates.len(),
            self.conflict_points(),
            self.witnessed(),
            self.latent(),
            self.insufficient_attribution()
        )];
        lines.push(self.coverage.render_zh());
        if self.certificates.is_empty() {
            lines.push("未发现冲突证书。".to_string());
            return lines.join("\n");
        }
        for certificate in &self.certificates {
            lines.push(certificate.render_zh());
        }
        lines.push(self.retraction.render_zh());
        lines.join("\n")
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// 检测：已见证冲突（闭包）与潜在冲突（规则对）
// ---------------------------------------------------------------------------

/// 冲突判定使用的声明面：功能谓词、互斥类型对、类型断言谓词。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictSchema {
    #[serde(default)]
    pub functional: BTreeSet<String>,
    #[serde(default)]
    pub disjoint: BTreeSet<(String, String)>,
    #[serde(default)]
    pub type_predicates: BTreeSet<String>,
    /// 输入类型名 → 规范类型名（别名归一；空表时按原样比较）。
    #[serde(default)]
    pub type_aliases: BTreeMap<String, String>,
    /// 输入谓词名 → 规范谓词名（别名归一；逆向别名不参与方向换算）。
    #[serde(default)]
    pub predicate_aliases: BTreeMap<String, String>,
}

impl ConflictSchema {
    pub fn new() -> Self {
        ConflictSchema {
            functional: BTreeSet::new(),
            disjoint: BTreeSet::new(),
            type_predicates: DEFAULT_TYPE_PREDICATES
                .iter()
                .map(|name| name.to_string())
                .collect(),
            type_aliases: BTreeMap::new(),
            predicate_aliases: BTreeMap::new(),
        }
    }

    /// 从谓词画像与类型体系构建判定面。
    pub fn from_registry_and_types(registry: &PredicateRegistry, types: &TypeSystem) -> Self {
        let mut schema = Self::new();
        for predicate in registry.canonical_names() {
            schema
                .predicate_aliases
                .insert(predicate.to_string(), predicate.to_string());
            if registry.is_functional(predicate) {
                schema.functional.insert(predicate.to_string());
            }
        }
        for (alias, canonical, inverted) in registry.alias_pairs() {
            // 逆向别名交换主宾语，不能按同一方向归组；保留原名，避免误判。
            if !inverted {
                schema
                    .predicate_aliases
                    .insert(alias, canonical);
            }
        }
        for (left, right) in types.disjoint_pairs() {
            schema.disjoint.insert((left, right));
        }
        schema
    }

    /// 登记一个输入类型名的别名归一结果。
    pub fn observe_type(&mut self, types: &TypeSystem, name: &str) {
        self.type_aliases
            .insert(name.to_string(), types.canonical(name));
    }

    pub fn canonical_predicate(&self, name: &str) -> String {
        self.predicate_aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    /// 谓词是否为声明的功能谓词（先做别名归一）。
    pub fn is_functional(&self, name: &str) -> bool {
        self.functional.contains(&self.canonical_predicate(name))
    }

    pub fn canonical_type(&self, name: &str) -> String {
        self.type_aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    /// 是否存在声明的互斥（两侧都先做别名归一）。
    pub fn declares_disjoint(&self, left: &str, right: &str) -> bool {
        let left = self.canonical_type(left);
        let right = self.canonical_type(right);
        self.disjoint.contains(&(left.clone(), right.clone()))
            || self.disjoint.contains(&(right, left))
    }
}

/// 从整图事实检测已见证冲突（功能槽多值 + 互斥类型共存）。
///
/// `supports` 是事实键 → 支撑树；缺失时退化为「来源未登记」的事实支撑。
pub fn detect_witnessed(
    facts: &[FactRef],
    supports: &BTreeMap<String, Support>,
    schema: &ConflictSchema,
) -> Vec<ConflictCertificate> {
    let lookup = |fact: &FactRef| -> Support {
        supports
            .get(&fact.key())
            .cloned()
            .unwrap_or_else(|| Support::fact(fact.clone(), BTreeSet::new()))
    };

    let mut out = Vec::new();

    // 功能槽：按（规范谓词， 主语）分组，取值两两成证书。
    // 保留每条原始事实引用，别名谓词也能找回自己的支撑树。
    let mut groups: BTreeMap<(String, String), BTreeMap<String, FactRef>> = BTreeMap::new();
    for fact in facts {
        if !schema.is_functional(&fact.predicate) {
            continue;
        }
        groups
            .entry((
                schema.canonical_predicate(&fact.predicate),
                fact.subject.clone(),
            ))
            .or_default()
            .entry(fact.value.clone())
            .or_insert_with(|| fact.clone());
    }
    for ((predicate, subject), values) in groups {
        let values: Vec<(String, FactRef)> = values.into_iter().collect();
        for (index, (left_value, left_fact)) in values.iter().enumerate() {
            for (right_value, right_fact) in values.iter().skip(index + 1) {
                let point = ConflictPoint::Functional {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    left_value: left_value.clone(),
                    right_value: right_value.clone(),
                };
                out.push(ConflictCertificate::new(
                    point,
                    lookup(left_fact),
                    lookup(right_fact),
                    false,
                ));
            }
        }
    }

    // 互斥类型：同一实体被断言为两个互斥类型。
    let mut type_groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for fact in facts {
        if schema.type_predicates.contains(&fact.predicate) {
            type_groups
                .entry(fact.subject.clone())
                .or_default()
                .insert(fact.value.clone());
        }
    }
    for (entity, types) in type_groups {
        let types: Vec<String> = types.into_iter().collect();
        for (index, left_type) in types.iter().enumerate() {
            for right_type in types.iter().skip(index + 1) {
                if !schema.declares_disjoint(left_type, right_type) {
                    continue;
                }
                let point = ConflictPoint::DisjointTypes {
                    entity: entity.clone(),
                    left_type: left_type.clone(),
                    right_type: right_type.clone(),
                };
                out.push(ConflictCertificate::new(
                    point.clone(),
                    lookup(&point.left_fact()),
                    lookup(&point.right_fact()),
                    false,
                ));
            }
        }
    }

    out
}

/// 从规则对检测潜在冲突（规则已存在、图上尚无实例）。
///
/// 只报告能确定「同一主语可能得到不同常量」的规则对：
/// - 功能冲突：头部值位置两侧都是常量且不同；
/// - 类型冲突：类型断言谓词的值位置两侧都是常量且互斥；
/// - 主语位置必须可合一，且体部没有可直接检出的常量冲突。
pub fn detect_latent(rules: &[Clause], schema: &ConflictSchema) -> Vec<ConflictCertificate> {
    let mut sorted: Vec<&Clause> = rules.iter().collect();
    sorted.sort_by(|left, right| clause_key(left).cmp(&clause_key(right)));

    let mut out: BTreeMap<String, ConflictCertificate> = BTreeMap::new();
    for (index, left) in sorted.iter().enumerate() {
        for right in sorted.iter().skip(index + 1) {
            if let Some(certificate) = latent_pair(left, right, schema) {
                out.entry(certificate.id.clone()).or_insert(certificate);
            }
        }
    }
    out.into_values().collect()
}

fn latent_pair(left: &Clause, right: &Clause, schema: &ConflictSchema) -> Option<ConflictCertificate> {
    if left.head.predicate != right.head.predicate || left.head.arity() != 2 {
        return None;
    }
    let right_head = freshen_atom(&right.head, "#r");
    let right_body = right
        .body
        .iter()
        .map(|atom| freshen_atom(atom, "#r"))
        .collect::<Vec<_>>();
    let left_subject = Atom::new("$subject", vec![left.head.terms[0].clone()]);
    let right_subject = Atom::new("$subject", vec![right_head.terms[0].clone()]);
    let mut subst = Substitution::new();
    if !unify_atoms(&left_subject, &right_subject, &mut subst) {
        return None;
    }
    let subject_display = term_display(&resolve_term(&left.head.terms[0], &subst));
    let predicate = left.head.predicate.to_string();
    let left_value = &left.head.terms[1];
    let right_value = &right_head.terms[1];

    let point = if schema.is_functional(&predicate) {
        let (Term::Const(left_constant), Term::Const(right_constant)) = (left_value, right_value)
        else {
            return None;
        };
        if left_constant == right_constant {
            return None;
        }
        ConflictPoint::Functional {
            subject: subject_display,
            predicate,
            left_value: left_constant.to_string(),
            right_value: right_constant.to_string(),
        }
    } else if schema.type_predicates.contains(&predicate) {
        let (Term::Const(left_constant), Term::Const(right_constant)) = (left_value, right_value)
        else {
            return None;
        };
        if !schema.declares_disjoint(left_constant, right_constant) {
            return None;
        }
        ConflictPoint::DisjointTypes {
            entity: subject_display,
            left_type: left_constant.to_string(),
            right_type: right_constant.to_string(),
        }
    } else {
        return None;
    };

    if !bodies_compatible(&left.body, &right_body, &subst) {
        return None;
    }

    let left_support = clause_support(left, "latent");
    let right_support = clause_support(right, "latent");
    Some(ConflictCertificate::new(
        point,
        left_support,
        right_support,
        true,
    ))
}

fn clause_support(clause: &Clause, tag: &str) -> Support {
    let head = &clause.head;
    let (subject, value) = match head.terms.as_slice() {
        [subject, value] => (term_display(subject), term_display(value)),
        [subject] => (term_display(subject), String::new()),
        _ => (String::new(), String::new()),
    };
    let fact = FactRef::new(&subject, &head.predicate, &value);
    let canonical = format!(
        "{}:-{}@{}",
        head.display(),
        clause
            .body
            .iter()
            .map(Atom::display)
            .collect::<Vec<_>>()
            .join("&"),
        tag
    );
    let source = SourceId::new(SourceKind::Rule, &clause.source, &canonical);
    let premises: Vec<Support> = clause
        .body
        .iter()
        .map(|atom| {
            let (subject, value) = match atom.terms.as_slice() {
                [subject, value] => (term_display(subject), term_display(value)),
                [subject] => (term_display(subject), String::new()),
                _ => (String::new(), String::new()),
            };
            Support::fact(
                FactRef::new(&subject, &atom.predicate, &value),
                BTreeSet::new(),
            )
        })
        .collect();
    Support::derived(
        fact,
        source,
        &format!("{}@{}", clause.source, tag),
        clause.confidence,
        premises,
    )
}

fn bodies_compatible(left: &[Atom], right: &[Atom], subst: &Substitution) -> bool {
    for left_atom in left {
        let left_atom = apply_subst_atom(left_atom, subst);
        for right_atom in right {
            let right_atom = apply_subst_atom(right_atom, subst);
            if left_atom.predicate != right_atom.predicate || left_atom.arity() != right_atom.arity() {
                continue;
            }
            let mut trial = subst.clone();
            if !unify_atoms(&left_atom, &right_atom, &mut trial) {
                return false;
            }
        }
    }
    true
}

fn apply_subst_atom(atom: &Atom, subst: &Substitution) -> Atom {
    Atom::new(
        &atom.predicate,
        atom.terms
            .iter()
            .map(|term| resolve_term(term, subst))
            .collect(),
    )
}

fn resolve_term(term: &Term, subst: &Substitution) -> Term {
    let mut current = term.clone();
    for _ in 0..MAX_SUPPORT_DEPTH {
        let Term::Var(name) = &current else {
            return current;
        };
        match subst.get(name) {
            Some(next) => current = next.clone(),
            None => return current,
        }
    }
    current
}

fn freshen_atom(atom: &Atom, suffix: &str) -> Atom {
    Atom::new(
        &atom.predicate,
        atom.terms
            .iter()
            .map(|term| match term {
                Term::Var(name) => Term::Var(format!("{}#{}", name, suffix).into()),
                Term::Const(name) => Term::Const(name.clone()),
            })
            .collect(),
    )
}

fn term_display(term: &Term) -> String {
    match term {
        Term::Var(name) => format!("?{}", name),
        Term::Const(name) => name.to_string(),
    }
}

fn clause_key(clause: &Clause) -> String {
    format!(
        "{}|{}",
        clause.head.display(),
        clause
            .body
            .iter()
            .map(Atom::display)
            .collect::<Vec<_>>()
            .join("&")
    )
}

// ---------------------------------------------------------------------------
// 最小撤回集
// ---------------------------------------------------------------------------

/// 计算命中所有证书支撑集的最小事实集合。
///
/// 候选 ≤ [`MAX_EXACT_CANDIDATES`] 时精确枚举（同一大小取字典序最小，保证
/// 确定性）；超过时按覆盖数贪心并标记 `exact = false`。支撑集为空的证书
/// （来源与事实都缺失）计入 `unresolved`，不被静默忽略。
pub fn plan_retraction(certificates: &[ConflictCertificate]) -> RetractionPlan {
    let mut sets: Vec<BTreeSet<FactRef>> = Vec::new();
    let mut unresolved = 0usize;
    let mut candidates: BTreeSet<FactRef> = BTreeSet::new();
    for certificate in certificates {
        let support = certificate.support_facts();
        if support.is_empty() {
            unresolved += 1;
            continue;
        }
        candidates.extend(support.iter().cloned());
        sets.push(support);
    }
    if sets.is_empty() {
        return RetractionPlan {
            facts: Vec::new(),
            exact: true,
            candidates: 0,
            resolved: 0,
            unresolved,
        };
    }

    let candidates: Vec<FactRef> = candidates.into_iter().collect();
    let (selected, exact) = if candidates.len() <= MAX_EXACT_CANDIDATES {
        (
            exact_hitting_set(&sets, &candidates).unwrap_or_default(),
            true,
        )
    } else {
        (greedy_hitting_set(&sets, &candidates), false)
    };
    let selected_sets: BTreeSet<&FactRef> = selected.iter().collect();
    let resolved = sets
        .iter()
        .filter(|set| set.iter().any(|fact| selected_sets.contains(fact)))
        .count();
    RetractionPlan {
        facts: selected,
        exact,
        candidates: candidates.len(),
        resolved,
        unresolved,
    }
}

fn exact_hitting_set(
    sets: &[BTreeSet<FactRef>],
    candidates: &[FactRef],
) -> Option<Vec<FactRef>> {
    let count = candidates.len();
    if count == 0 {
        return None;
    }
    let mut best: Option<(u32, Vec<usize>)> = None;
    for mask in 1u32..(1u32 << count) {
        let size = mask.count_ones();
        if let Some((best_size, _)) = &best {
            if size > *best_size {
                continue;
            }
        }
        let hits = sets.iter().all(|set| {
            set.iter().any(|fact| {
                candidates
                    .iter()
                    .position(|candidate| candidate == fact)
                    .map(|index| mask & (1 << index) != 0)
                    .unwrap_or(false)
            })
        });
        if !hits {
            continue;
        }
        let indices: Vec<usize> = (0..count).filter(|index| mask & (1 << index) != 0).collect();
        let replace = match &best {
            None => true,
            Some((best_size, best_indices)) => {
                size < *best_size || (size == *best_size && indices < *best_indices)
            }
        };
        if replace {
            best = Some((size, indices));
        }
    }
    best.map(|(_, indices)| indices.into_iter().map(|index| candidates[index].clone()).collect())
}

fn greedy_hitting_set(sets: &[BTreeSet<FactRef>], candidates: &[FactRef]) -> Vec<FactRef> {
    let mut uncovered: Vec<&BTreeSet<FactRef>> = sets.iter().collect();
    let mut selected: Vec<FactRef> = Vec::new();
    while !uncovered.is_empty() {
        let mut best: Option<(usize, &FactRef)> = None;
        for candidate in candidates {
            let covered = uncovered.iter().filter(|set| set.contains(candidate)).count();
            let replace = match best {
                None => covered > 0,
                Some((best_covered, best_candidate)) => {
                    covered > best_covered
                        || (covered == best_covered && candidate < best_candidate)
                }
            };
            if replace {
                best = Some((covered, candidate));
            }
        }
        match best {
            Some((_, candidate)) if !selected.contains(candidate) => {
                selected.push(candidate.clone());
                uncovered.retain(|set| !set.contains(candidate));
            }
            _ => break,
        }
    }
    selected
}

// ---------------------------------------------------------------------------
// 与规则/派生账本对接
// ---------------------------------------------------------------------------

/// 把图谱事实、事实来源台账、规则账本与派生账本拼成支撑树。
///
/// 派生事实的来源由规则账本给出（规则出处 + 前提链），而不是谓词画像标签；
/// 没有派生记录的普通事实回落到 [`ProvenanceLedger`] 的事实级来源。两者都没有
/// 时保持空来源，让后续证书如实报告「来源未登记」。
pub fn build_supports(
    facts: &[FactRef],
    ledger: &ProvenanceLedger,
    rules: &RuleLedger,
    derived: &DerivedLedger,
) -> BTreeMap<String, Support> {
    let mut derived_by_fact: BTreeMap<String, &crate::graph_ledger::DerivedLedgerEntry> =
        BTreeMap::new();
    for entry in &derived.entries {
        if let Some(fact) = ledger_atom_fact(&entry.atom) {
            derived_by_fact.entry(fact.key()).or_insert(entry);
        }
    }
    let mut rules_by_source: BTreeMap<&str, &crate::graph_ledger::RuleLedgerEntry> = BTreeMap::new();
    for entry in &rules.entries {
        rules_by_source.entry(entry.source.as_str()).or_insert(entry);
    }

    let mut out: BTreeMap<String, Support> = BTreeMap::new();
    for fact in facts {
        let support = build_support_with(
            fact,
            ledger,
            &derived_by_fact,
            &rules_by_source,
            &mut BTreeSet::new(),
            0,
        );
        out.insert(fact.key(), support);
    }
    // 派生账本里的结论即使尚未出现在快照里也纳入（潜在/已见证判定都用得到）。
    for entry in &derived.entries {
        if let Some(fact) = ledger_atom_fact(&entry.atom) {
            if !out.contains_key(&fact.key()) {
                let support = build_support_with(
                    &fact,
                    ledger,
                    &derived_by_fact,
                    &rules_by_source,
                    &mut BTreeSet::new(),
                    0,
                );
                out.insert(fact.key(), support);
            }
        }
    }
    out
}

fn build_support_with(
    fact: &FactRef,
    ledger: &ProvenanceLedger,
    derived_by_fact: &BTreeMap<String, &crate::graph_ledger::DerivedLedgerEntry>,
    rules_by_source: &BTreeMap<&str, &crate::graph_ledger::RuleLedgerEntry>,
    visiting: &mut BTreeSet<String>,
    depth: usize,
) -> Support {
    if depth >= MAX_SUPPORT_DEPTH || !visiting.insert(fact.key()) {
        return ledger.support(fact.clone());
    }
    let support = match derived_by_fact.get(&fact.key()) {
        Some(entry) => {
            let (source, rule_id, confidence) = match rules_by_source.get(entry.rule_source.as_str())
            {
                Some(rule) => (
                    SourceId::new(SourceKind::Rule, &rule.source, &rule_canonical(rule)),
                    rule.id.clone(),
                    rule.confidence,
                ),
                None => (
                    SourceId::new(SourceKind::Rule, &entry.rule_source, &entry.id),
                    entry.rule_source.clone(),
                    entry.confidence,
                ),
            };
            let premises: Vec<Support> = entry
                .premises
                .iter()
                .filter_map(|atom| ledger_atom_fact(atom))
                .map(|premise| {
                    build_support_with(
                        &premise,
                        ledger,
                        derived_by_fact,
                        rules_by_source,
                        visiting,
                        depth + 1,
                    )
                })
                .collect();
            if premises.is_empty() {
                ledger.support(fact.clone())
            } else {
                Support::derived(fact.clone(), source, &rule_id, confidence, premises)
            }
        }
        None => ledger.support(fact.clone()),
    };
    visiting.remove(&fact.key());
    support
}

fn rule_canonical(entry: &crate::graph_ledger::RuleLedgerEntry) -> String {
    let body = entry
        .body
        .iter()
        .map(LedgerAtom::display)
        .collect::<Vec<_>>()
        .join("&");
    format!("{}:-{}", entry.head.display(), body)
}

fn ledger_atom_fact(atom: &LedgerAtom) -> Option<FactRef> {
    match atom.terms.as_slice() {
        [subject, value] => Some(FactRef::new(
            &ledger_term_name(subject),
            &atom.predicate,
            &ledger_term_name(value),
        )),
        [subject] => Some(FactRef::new(
            &ledger_term_name(subject),
            &atom.predicate,
            "",
        )),
        _ => None,
    }
}

fn ledger_term_name(term: &crate::graph_ledger::LedgerTerm) -> String {
    if term.is_var {
        format!("?{}", term.name)
    } else {
        term.name.clone()
    }
}

fn local_from_iri(iri: &str) -> String {
    let base = crate::identity::DEFAULT_BASE_NS;
    crate::identity::entity_local_name(base, iri)
        .or_else(|| crate::identity::local_name(base, iri))
        .unwrap_or_else(|| iri.to_string())
}

fn predicate_from_iri(iri: &str) -> String {
    if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
        "type".to_string()
    } else {
        local_from_iri(iri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(locator: &str, content: &str) -> SourceId {
        SourceId::new(SourceKind::Import, locator, content)
    }

    fn fact(subject: &str, predicate: &str, value: &str) -> FactRef {
        FactRef::new(subject, predicate, value)
    }

    fn supported(fact: FactRef, sources: &[SourceId]) -> Support {
        Support::fact(fact, sources.iter().cloned().collect())
    }

    fn empty_rules() -> RuleLedger {
        RuleLedger {
            synced: false,
            fingerprint: String::new(),
            entries: Vec::new(),
        }
    }

    fn empty_derived() -> DerivedLedger {
        DerivedLedger {
            materialized: false,
            revision: 0,
            fingerprint: String::new(),
            entries: Vec::new(),
        }
    }

    fn schema() -> ConflictSchema {
        let mut schema = ConflictSchema::new();
        schema.functional.insert("年龄".to_string());
        schema.disjoint.insert(("人物".to_string(), "地点".to_string()));
        schema
    }

    #[test]
    fn source_identity_separates_category_locator_and_content() {
        let first = source("体测表", "张三 38");
        let same = source("体测表", "张三 38");
        let different_content = source("体测表", "张三 39");
        let different_locator = source("体检表", "张三 38");
        assert_eq!(first, same, "同一定位符 + 同一内容 = 同一身份");
        assert_ne!(first, different_content, "内容摘要必须参与身份");
        assert_ne!(first, different_locator, "定位符必须参与身份");
        assert!(first.is_registered());
        assert!(!SourceId::unknown().is_registered());
        assert_eq!(SourceId::unknown().render(), "来源未登记");
        assert!(first.render().contains("导入:体测表#"));
    }

    #[test]
    fn source_kind_parses_human_labels_and_rejects_typos() {
        assert_eq!(SourceKind::parse("import"), Some(SourceKind::Import));
        assert_eq!(SourceKind::parse("导入"), Some(SourceKind::Import));
        assert_eq!(SourceKind::parse("人工确认"), Some(SourceKind::Manual));
        assert_eq!(SourceKind::parse("unknown"), Some(SourceKind::Unknown));
        assert_eq!(SourceKind::parse("导入n"), None);
    }

    #[test]
    fn ledger_joins_sources_and_reports_honest_coverage() {
        let first = fact("张三", "年龄", "38");
        let second = fact("张三", "师傅", "李四");
        let mut ledger = ProvenanceLedger::new();
        let a = source("甲批", "a");
        let b = source("乙批", "b");
        assert!(ledger.record(first.clone(), a.clone()));
        assert!(!ledger.record(first.clone(), a.clone()), "重复登记幂等");
        assert!(ledger.record(first.clone(), b.clone()), "同一事实可多来源");
        assert_eq!(ledger.sources(&first).len(), 2, "多来源必须保留全部");
        let coverage = ledger.coverage(&[first.clone(), second.clone()]);
        assert_eq!(coverage.total, 2);
        assert_eq!(coverage.registered, 1);
        assert_eq!(coverage.unregistered, vec![second]);
        assert!((coverage.ratio - 0.5).abs() < 1e-9);
        assert!(coverage.render_zh().contains("1/2"));
    }

    #[test]
    fn tsv_parser_rejects_malformed_rows_and_accepts_batches() {
        let text = "# comment\n姚明\t年龄\t38\timport\t体测表\t姚明38\n张三\t师傅\t李四\tmanual\t人工确认\n";
        let rows = parse_provenance_tsv(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, fact("姚明", "年龄", "38"));
        assert_eq!(rows[0].1.kind, SourceKind::Import);
        assert!(rows[0].1.render().contains("体测表"));
        assert!(parse_provenance_tsv("a\tb\tc\ttypo\td").is_err());
        assert!(parse_provenance_tsv("a\tb\tc\tunknown\td").is_err());
        assert!(parse_provenance_tsv("a\tb\tc").is_err());
    }

    #[test]
    fn ledger_roundtrip_and_tamper_are_visible() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("provenance.json");
        let path_text = path.to_str().unwrap();
        assert!(matches!(ProvenanceLedger::load(path_text), ProvenanceLoad::Missing));

        let mut ledger = ProvenanceLedger::new();
        ledger.record(fact("张三", "年龄", "38"), source("体测表", "38"));
        ledger.save(path_text).unwrap();
        let ProvenanceLoad::Loaded(restored) = ProvenanceLedger::load(path_text) else {
            panic!("应能恢复来源台账");
        };
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.digest(), ledger.digest());

        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path_text).unwrap()).unwrap();
        value["sections"][PROVENANCE_SECTION][0]["fact"]["value"] = serde_json::json!("39");
        std::fs::write(path_text, value.to_string()).unwrap();
        assert!(matches!(
            ProvenanceLedger::load(path_text),
            ProvenanceLoad::Damaged(_)
        ));
    }

    #[test]
    fn witnessed_functional_conflict_has_candidates_and_exact_retraction() {
        let age = fact("姚明", "年龄", "38");
        let age_other = fact("姚明", "年龄", "39");
        let first = source("2002-体测表", "38");
        let second = source("2010-访谈", "39");
        let mut ledger = ProvenanceLedger::new();
        ledger.record(age.clone(), first.clone());
        ledger.record(age_other.clone(), second.clone());
        let supports = build_supports(
            &[age.clone(), age_other.clone()],
            &ledger,
            &empty_rules(),
            &empty_derived(),
        );
        let facts = vec![age, age_other];
        let certificates = detect_witnessed(&facts, &supports, &schema());
        assert_eq!(certificates.len(), 1);
        let certificate = &certificates[0];
        assert!(!certificate.latent);
        assert!(matches!(
            certificate.attribution,
            Attribution::Candidates { .. }
        ));
        assert_eq!(certificate.left_only.len(), 1);
        assert_eq!(certificate.right_only.len(), 1);
        assert!(certificate.common.is_empty());
        let report = ConflictReport::new(certificates, ledger.coverage(&facts));
        assert!(report.retraction.exact);
        assert_eq!(report.retraction.facts.len(), 1, "删一个值即可解决冲突");
        assert_eq!(report.retraction.resolved, 1);
        assert_eq!(report.retraction.unresolved, 0);
        let rendered = report.render_text();
        assert!(rendered.contains("已见证"), "{}", rendered);
        assert!(rendered.contains("候选肇事前提"), "{}", rendered);
        assert!(rendered.contains("2002-体测表"), "{}", rendered);
    }

    #[test]
    fn missing_sources_enter_insufficient_attribution_not_silent_choice() {
        let age = fact("姚明", "年龄", "38");
        let age_other = fact("姚明", "年龄", "39");
        let supports = build_supports(
            &[age.clone(), age_other.clone()],
            &ProvenanceLedger::new(),
            &empty_rules(),
            &empty_derived(),
        );
        let facts = vec![age, age_other];
        let report = ConflictReport::new(
            detect_witnessed(&facts, &supports, &schema()),
            Coverage {
                total: 2,
                registered: 0,
                ratio: 0.0,
                unregistered: facts,
            },
        );
        assert_eq!(report.insufficient_attribution(), 1);
        let rendered = report.render_text();
        assert!(rendered.contains("无法归因"), "{}", rendered);
        assert!(rendered.contains("来源未登记"), "{}", rendered);
        assert_eq!(
            report.retraction.facts.len(),
            1,
            "即使来源缺失，事实级撤回集仍可计算"
        );
        assert!(report.retraction.exact);
    }

    #[test]
    fn shared_premises_are_same_origin_not_independent_sources() {
        let premise = supported(
            fact("姚明", "出生年", "1986"),
            &[source("档案", "1986")],
        );
        let left = Support::derived(
            fact("姚明", "年龄", "38"),
            source("规则甲", "甲"),
            "R1",
            0.9,
            vec![premise.clone()],
        );
        let right = Support::derived(
            fact("姚明", "年龄", "39"),
            source("规则乙", "乙"),
            "R2",
            0.9,
            vec![premise],
        );
        let certificate = ConflictCertificate::new(
            ConflictPoint::Functional {
                subject: "姚明".to_string(),
                predicate: "年龄".to_string(),
                left_value: "38".to_string(),
                right_value: "39".to_string(),
            },
            left,
            right,
            false,
        );
        assert!(matches!(
            certificate.attribution,
            Attribution::SameOrigin { .. }
        ));
        let report = ConflictReport::new(
            vec![certificate],
            Coverage {
                total: 1,
                registered: 1,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        assert_eq!(
            report.retraction.facts,
            vec![fact("姚明", "出生年", "1986")],
            "同源矛盾只能撤回共同前提"
        );
    }

    #[test]
    fn minimal_hitting_set_covers_overlapping_certificates() {
        let a = supported(fact("A", "p", "1"), &[source("s", "a")]);
        let b = supported(fact("B", "p", "1"), &[source("s", "b")]);
        let c = supported(fact("C", "p", "1"), &[source("s", "c")]);
        let mut certificates = Vec::new();
        for (left, right) in [(&a, &b), (&b, &c)] {
            let point = ConflictPoint::Functional {
                subject: left.fact.subject.clone(),
                predicate: "p".to_string(),
                left_value: left.fact.value.clone(),
                right_value: right.fact.value.clone(),
            };
            certificates.push(ConflictCertificate::new(
                point,
                left.clone(),
                right.clone(),
                false,
            ));
        }
        let plan = plan_retraction(&certificates);
        assert!(plan.exact);
        assert_eq!(plan.facts.len(), 1, "两个共用 B 的证书只需撤回 B");
        assert_eq!(plan.facts, vec![fact("B", "p", "1")]);
        assert_eq!(plan.resolved, 2);
    }

    #[test]
    fn predicate_aliases_join_the_same_functional_slot() {
        let mut schema = schema();
        schema
            .predicate_aliases
            .insert("岁数".to_string(), "年龄".to_string());
        let first = fact("姚明", "年龄", "38");
        let second = fact("姚明", "岁数", "39");
        let supports = BTreeMap::new();
        let certificates = detect_witnessed(&[first.clone(), second.clone()], &supports, &schema);
        assert_eq!(certificates.len(), 1, "别名必须并入同一功能槽");
        assert_eq!(
            certificates[0].point,
            ConflictPoint::Functional {
                subject: "姚明".to_string(),
                predicate: "年龄".to_string(),
                left_value: "38".to_string(),
                right_value: "39".to_string(),
            }
        );
        assert_eq!(certificates[0].left.fact, first);
        assert_eq!(certificates[0].right.fact, second);
    }

    #[test]
    fn non_functional_predicates_and_same_values_are_not_conflicts() {
        let facts = vec![
            fact("张三", "喜欢", "李四"),
            fact("张三", "喜欢", "王五"),
            fact("张三", "年龄", "38"),
            fact("张三", "年龄", "38"),
        ];
        let supports = BTreeMap::new();
        let certificates = detect_witnessed(&facts, &supports, &schema());
        assert!(
            certificates.is_empty(),
            "非功能谓词多值、相同取值重复都不是冲突：{:?}",
            certificates
        );
    }

    #[test]
    fn witnessed_disjoint_types_use_type_aliases() {
        let mut schema = schema();
        schema
            .type_aliases
            .insert("Person".to_string(), "人物".to_string());
        let facts = vec![
            FactRef::type_assertion("姚明", "Person"),
            FactRef::type_assertion("姚明", "地点"),
        ];
        let certificates = detect_witnessed(&facts, &BTreeMap::new(), &schema);
        assert_eq!(certificates.len(), 1);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::DisjointTypes { .. }
        ));
        assert!(matches!(
            certificates[0].attribution,
            Attribution::Insufficient { .. }
        ));
    }

    fn clause(head: Atom, body: Vec<Atom>, source: &str) -> Clause {
        Clause::new(head, body, 0.9, source)
    }

    #[test]
    fn latent_functional_conflict_only_for_different_head_constants() {
        let left = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("38")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1986")],
            )],
            "规则甲",
        );
        let right = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("39")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1990")],
            )],
            "规则乙",
        );
        let schema = schema();
        let certificates = detect_latent(&[left.clone(), right.clone()], &schema);
        assert_eq!(
            certificates.len(),
            0,
            "出生年互相矛盾，两条规则不可能同时触发：{:?}",
            certificates
        );

        let compatible = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("39")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1986")],
            )],
            "规则乙",
        );
        let certificates = detect_latent(&[left, compatible], &schema);
        assert_eq!(certificates.len(), 1);
        assert!(certificates[0].latent);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::Functional { .. }
        ));

        let same = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("38")]),
            Vec::new(),
            "重复规则",
        );
        assert!(
            detect_latent(&[same.clone(), same], &schema).is_empty(),
            "头部常量相同不是冲突"
        );
    }

    #[test]
    fn latent_type_conflict_requires_declared_disjoint_types() {
        let left = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("人物")]),
            Vec::new(),
            "类型规则甲",
        );
        let right = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("地点")]),
            Vec::new(),
            "类型规则乙",
        );
        let certificates = detect_latent(&[left.clone(), right.clone()], &schema());
        assert_eq!(certificates.len(), 1);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::DisjointTypes { .. }
        ));

        let compatible = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("人物")]),
            Vec::new(),
            "类型规则丙",
        );
        assert!(detect_latent(&[left, compatible], &schema()).is_empty());
    }

    #[test]
    fn report_is_deterministic_and_serializable() {
        let a = supported(fact("A", "年龄", "1"), &[source("s", "a")]);
        let b = supported(fact("A", "年龄", "2"), &[source("s", "b")]);
        let cert = ConflictCertificate::new(
            ConflictPoint::Functional {
                subject: "A".to_string(),
                predicate: "年龄".to_string(),
                left_value: "1".to_string(),
                right_value: "2".to_string(),
            },
            a,
            b,
            false,
        );
        let first = ConflictReport::new(
            vec![cert.clone()],
            Coverage {
                total: 2,
                registered: 2,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        let second = ConflictReport::new(
            vec![cert],
            Coverage {
                total: 2,
                registered: 2,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        assert_eq!(first, second);
        assert_eq!(first.to_json(), second.to_json());
        assert!(!first.certificates[0].id.is_empty());
    }

    #[test]
    fn import_journal_seeding_uses_real_batch_identity() {
        use crate::import_journal::ImportJournal;
        use crate::predicate_registry::{ImportAction, ImportReceipt};

        let receipt = ImportReceipt {
            added: 2,
            actions: vec![
                ImportAction::Literal {
                    subject: "http://example.org/kg/entity/姚明".to_string(),
                    predicate: "http://example.org/kg/年龄".to_string(),
                    value: "38".to_string(),
                },
                ImportAction::Named {
                    subject: "http://example.org/kg/entity/姚明".to_string(),
                    predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    object: "http://example.org/kg/Person".to_string(),
                },
            ],
            ..Default::default()
        };
        let journal = ImportJournal::from_entries(vec![receipt]);
        let mut ledger = ProvenanceLedger::new();
        assert_eq!(ledger.seed_from_import_journal(&journal), 2);

        let age = fact("姚明", "年龄", "38");
        let sources = ledger.sources(&age);
        assert_eq!(sources.len(), 1);
        let source = sources.iter().next().unwrap();
        assert_eq!(source.kind, SourceKind::Import);
        assert!(source.locator.starts_with("导入日志#1"));
        assert!(source.is_registered());
        assert!(ledger.sources(&FactRef::type_assertion("姚明", "Person")).len() == 1);
    }

    #[test]
    fn fusion_seeding_keeps_namespace_and_evidence_pointer() {
        use crate::knowledge_fusion::{
            FusionLedger, NamespaceKind, NamespaceRecord, ObjectKind, ScopedAssertion, TrustLevel,
        };

        let namespace = NamespaceRecord::new("batch-a", "甲批", NamespaceKind::Import, TrustLevel::Imported);
        let assertion = ScopedAssertion::new(
            "batch-a",
            "姚明",
            "年龄",
            "40",
            ObjectKind::Literal,
            "/data/ns.ttl",
        );
        let mut fusion = FusionLedger::new();
        fusion.upsert_namespace(namespace);
        fusion.record_assertion(assertion);
        let mut ledger = ProvenanceLedger::new();
        assert_eq!(ledger.seed_from_fusion_ledger(&fusion), 1);
        let sources = ledger.sources(&fact("姚明", "年龄", "40"));
        assert_eq!(sources.len(), 1);
        let source = sources.iter().next().unwrap();
        assert!(source.locator.contains("甲批"));
        assert!(source.locator.contains("/data/ns.ttl"));
        assert!(source.render().contains("导入:甲批"));
    }

    #[test]
    fn three_values_make_three_certificates_but_one_conflict_point() {
        let mut facts = Vec::new();
        let mut ledger = ProvenanceLedger::new();
        for value in ["38", "39", "40"] {
            let fact = fact("姚明", "年龄", value);
            ledger.record(fact.clone(), source("批", value));
            facts.push(fact);
        }
        let supports = build_supports(
            &facts,
            &ledger,
            &empty_rules(),
            &empty_derived(),
        );
        let report = ConflictReport::new(
            detect_witnessed(&facts, &supports, &schema()),
            ledger.coverage(&facts),
        );
        assert_eq!(report.certificates.len(), 3, "两两证书");
        assert_eq!(report.conflict_points(), 1, "但只有一个功能槽冲突点");
        assert_eq!(report.retraction.facts.len(), 2, "修复代价是删 2 条，不是 3 条");
        assert!(report.render_text().contains("覆盖 1 个冲突点"));
    }

    #[test]
    fn retraction_falls_back_to_greedy_beyond_exact_limit() {
        let mut certificates = Vec::new();
        for index in 0..(MAX_EXACT_CANDIDATES + 1) {
            let left = supported(
                fact(&format!("E{}", index), "年龄", "1"),
                &[source("s", &format!("l{}", index))],
            );
            let right = supported(
                fact(&format!("E{}", index), "年龄", "2"),
                &[source("s", &format!("r{}", index))],
            );
            certificates.push(ConflictCertificate::new(
                ConflictPoint::Functional {
                    subject: format!("E{}", index),
                    predicate: "年龄".to_string(),
                    left_value: "1".to_string(),
                    right_value: "2".to_string(),
                },
                left,
                right,
                false,
            ));
        }
        let plan = plan_retraction(&certificates);
        assert!(!plan.exact, "超过精确上限必须标记非精确");
        assert_eq!(plan.resolved, certificates.len());
    }
}

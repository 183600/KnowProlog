//! 视图渲染：把内部符号结构渲染为「看得见的推理」。
//!
//! 四张视图，全部零依赖、确定性优先（同输入同字节，可做 golden 测试）：
//!   - 证明树：`ProofNode` → Mermaid / ASCII，节点集 ≡ 证明展平集合（不增不减）；
//!   - 知识子图：焦点实体 N 跳邻域的确定性 BFS 分层 → Mermaid / DOT / SVG / JSON，
//!     本轮实际用到的边可高亮（provenance 审计）；
//!   - 对话时间线：Chronicle → Markdown 表格 / Mermaid 流程，焦点迁移与出处逐轮可见；
//!   - 覆盖报告：悬空引用 / 孤立实体 / 未登记谓词，只做不依赖 schema 的确定性检查。
//!
//! 渲染→解析律（反渲染）：机器可读视图都有反向解析器——`render_json` ↔
//! `parse_json`（逐字节无损）、`render_timeline_markdown` ↔
//! `parse_timeline_markdown`、`render_timeline_flow` ↔ `parse_timeline_flow`
//! （后两者在安全域内逐字段无损，见各自文档）。渲染器不是「只写不读」的
//! 死角：看过的视图都能读回来，视图层因此可审计、可回归。
//!
//! 设计戒律：输入三元组打乱顺序输出仍须同字节——视图层否则不可测试、必然腐烂。

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::chronicle::ChronicleEntry;
use crate::context_image::canonical_json;
use crate::resolution_engine::{ProofNode, ProofSource};

/// SVG 手绘规模上限：超过则退化为 Mermaid 文本（避免浏览器卡死与布局失控）。
pub const MAX_SVG_NODES: usize = 200;

// ============================ 证明树 ============================

/// 证明树节点（与归结引擎解耦的视图类型；标签可用自然语言渲染）。
#[derive(Debug, Clone, PartialEq)]
pub enum ViewProofSource {
    Fact,
    Rule(String),
}

/// 证明树节点：标签 + 来源 + 置信度 + 子证明。
#[derive(Debug, Clone, PartialEq)]
pub struct ProofViewNode {
    /// 展示用标签（可以是自然语言）。
    pub label: String,
    /// 规范逻辑形式（`Atom::display()`）：证明日志的稳定身份只依赖该字段，
    /// 不受展示措辞变化影响。
    pub canonical: String,
    pub source: ViewProofSource,
    pub confidence: f64,
    pub children: Vec<ProofViewNode>,
}

impl ProofViewNode {
    /// 从归结证明节点转换，标签用逻辑形式 `atom.display()`。
    pub fn from_proof(node: &ProofNode) -> Self {
        Self::from_proof_with(node, |node| node.atom.display())
    }

    /// 从归结证明节点转换，标签由调用方给定（如自然语言渲染）。
    pub fn from_proof_with<F: Fn(&ProofNode) -> String>(node: &ProofNode, render: F) -> Self {
        fn build(node: &ProofNode, render: &dyn Fn(&ProofNode) -> String) -> ProofViewNode {
            ProofViewNode {
                label: render(node),
                canonical: node.atom.display(),
                source: match &node.source {
                    ProofSource::Fact => ViewProofSource::Fact,
                    ProofSource::Rule { source } => ViewProofSource::Rule(source.clone()),
                },
                confidence: node.confidence,
                children: node
                    .children
                    .iter()
                    .map(|child| build(child, render))
                    .collect(),
            }
        }
        build(node, &render)
    }

    /// 本子树（含自身）的节点数。
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(Self::node_count).sum::<usize>()
    }

    fn describe(&self) -> String {
        match &self.source {
            ViewProofSource::Fact => format!("{}（事实）", self.label),
            ViewProofSource::Rule(source) => format!(
                "{}（由 {}，置信度 {:.2}）",
                self.label, source, self.confidence
            ),
        }
    }
}

/// 证明森林的节点总数（≡ 展平集合大小）。
pub fn proof_forest_node_count(forest: &[ProofViewNode]) -> usize {
    forest.iter().map(ProofViewNode::node_count).sum()
}

fn mermaid_label(text: &str) -> String {
    text.replace('"', "#quot;")
        .replace('\n', "<br/>")
        .replace('\r', "")
}

/// 证明森林 → Mermaid（前序编号，边 = 子证明关系）。
pub fn render_proof_mermaid(forest: &[ProofViewNode]) -> String {
    let mut lines = vec!["graph TD".to_string()];
    let mut next_id = 0usize;
    fn emit(node: &ProofViewNode, id: usize, next_id: &mut usize, lines: &mut Vec<String>) {
        lines.push(format!(
            "    n{}[\"{}\"]",
            id,
            mermaid_label(&node.describe())
        ));
        for child in &node.children {
            let child_id = *next_id;
            *next_id += 1;
            lines.push(format!("    n{} --> n{}", id, child_id));
            emit(child, child_id, next_id, lines);
        }
    }
    for root in forest {
        let id = next_id;
        next_id += 1;
        emit(root, id, &mut next_id, &mut lines);
    }
    lines.join("\n")
}

fn ascii_children(children: &[ProofViewNode], prefix: &str, lines: &mut Vec<String>) {
    for (index, child) in children.iter().enumerate() {
        let last = index + 1 == children.len();
        lines.push(format!(
            "{}{} {}",
            prefix,
            if last { "└─" } else { "├─" },
            child.describe()
        ));
        let next_prefix = format!("{}{}", prefix, if last { "   " } else { "│  " });
        ascii_children(&child.children, &next_prefix, lines);
    }
}

/// 证明森林 → ASCII 树（终端可读）。
pub fn render_proof_ascii(forest: &[ProofViewNode]) -> String {
    let mut lines = Vec::new();
    for (index, root) in forest.iter().enumerate() {
        if forest.len() > 1 {
            lines.push(format!("证明 {}：", index + 1));
        }
        lines.push(format!("● {}", root.describe()));
        ascii_children(&root.children, "", &mut lines);
    }
    if lines.is_empty() {
        lines.push("（无证明）".to_string());
    }
    lines.join("\n")
}

// ============================ 知识子图 ============================

/// 子图节点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphNode {
    pub name: String,
    pub depth: usize,
    pub is_focus: bool,
}

/// 子图边；`highlight` 表示本轮推理实际用到（provenance 命中）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphEdge {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub highlight: bool,
}

/// 焦点实体邻域的确定性分层子图。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSubgraph {
    pub focus: String,
    pub max_depth: usize,
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<SubgraphEdge>,
}

/// `render_json` 的反渲染：JSON → 子图（逐字段无损，含任意引号/换行/中文）。
///
/// 解析律：`parse_json(&g.render_json())? == g`（nodes/edges 顺序也一致）。
pub fn parse_json(text: &str) -> Result<KnowledgeSubgraph, String> {
    #[derive(Deserialize)]
    struct JsonNode {
        name: String,
        depth: usize,
        focus: bool,
    }
    #[derive(Deserialize)]
    struct JsonEdge {
        subject: String,
        predicate: String,
        object: String,
        highlight: bool,
    }
    #[derive(Deserialize)]
    struct JsonSubgraph {
        focus: String,
        max_depth: usize,
        nodes: Vec<JsonNode>,
        edges: Vec<JsonEdge>,
    }
    let decoded: JsonSubgraph =
        serde_json::from_str(text).map_err(|error| format!("子图 JSON 解析失败：{}", error))?;
    Ok(KnowledgeSubgraph {
        focus: decoded.focus,
        max_depth: decoded.max_depth,
        nodes: decoded
            .nodes
            .into_iter()
            .map(|node| SubgraphNode {
                name: node.name,
                depth: node.depth,
                is_focus: node.focus,
            })
            .collect(),
        edges: decoded
            .edges
            .into_iter()
            .map(|edge| SubgraphEdge {
                subject: edge.subject,
                predicate: edge.predicate,
                object: edge.object,
                highlight: edge.highlight,
            })
            .collect(),
    })
}

impl KnowledgeSubgraph {
    /// 确定性 BFS：无向扩展（关系可反向追溯），节点按 (深度, 名称) 排序，
    /// 边集 = 两端都已发现的边，按 (主语, 谓词, 宾语) 排序。
    pub fn bfs(
        focus: &str,
        edges: &[(String, String, String)],
        max_depth: usize,
        highlight: &HashSet<(String, String, String)>,
    ) -> Self {
        let focus = focus.trim();
        let mut adjacency: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
        for (subject, predicate, object) in edges {
            adjacency
                .entry(subject.as_str())
                .or_default()
                .push((predicate.as_str(), object.as_str()));
            adjacency
                .entry(object.as_str())
                .or_default()
                .push((predicate.as_str(), subject.as_str()));
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort();
            neighbors.dedup();
        }

        let mut depth: BTreeMap<&str, usize> = BTreeMap::new();
        depth.insert(focus, 0);
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(focus);
        while let Some(node) = queue.pop_front() {
            let node_depth = depth[node];
            if node_depth >= max_depth {
                continue;
            }
            if let Some(neighbors) = adjacency.get(node) {
                for (_, other) in neighbors {
                    if !depth.contains_key(other) {
                        depth.insert(other, node_depth + 1);
                        queue.push_back(other);
                    }
                }
            }
        }

        let mut nodes: Vec<SubgraphNode> = depth
            .iter()
            .map(|(name, depth)| SubgraphNode {
                name: (*name).to_string(),
                depth: *depth,
                is_focus: *name == focus,
            })
            .collect();
        nodes.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.name.cmp(&right.name))
        });

        let discovered: BTreeSet<&str> = depth.keys().copied().collect();
        let mut selected: Vec<SubgraphEdge> = edges
            .iter()
            .filter(|(subject, _, object)| {
                discovered.contains(subject.as_str()) && discovered.contains(object.as_str())
            })
            .map(|(subject, predicate, object)| SubgraphEdge {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
                highlight: highlight.contains(&(
                    subject.clone(),
                    predicate.clone(),
                    object.clone(),
                )),
            })
            .collect();
        selected.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
        });
        selected.dedup_by(|left, right| {
            left.subject == right.subject
                && left.predicate == right.predicate
                && left.object == right.object
        });

        KnowledgeSubgraph {
            focus: focus.to_string(),
            max_depth,
            nodes,
            edges: selected,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn node_ids(&self) -> BTreeMap<&str, String> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.name.as_str(), format!("n{}", index)))
            .collect()
    }

    /// Mermaid（高亮边用粗箭头 `==>` 表达）。
    pub fn render_mermaid(&self) -> String {
        let ids = self.node_ids();
        let mut lines = vec!["graph LR".to_string()];
        for node in &self.nodes {
            let label = if node.is_focus {
                format!("★ {}", node.name)
            } else {
                node.name.clone()
            };
            lines.push(format!(
                "    {}[\"{}\"]",
                ids[node.name.as_str()],
                mermaid_label(&label)
            ));
        }
        for edge in &self.edges {
            let arrow = if edge.highlight { "==>" } else { "-->" };
            lines.push(format!(
                "    {} {}|\"{}\"| {}",
                ids[edge.subject.as_str()],
                arrow,
                mermaid_label(&edge.predicate),
                ids[edge.object.as_str()]
            ));
        }
        lines.join("\n")
    }

    /// Graphviz DOT（可离线渲染为图片）。
    pub fn render_dot(&self) -> String {
        let mutate = |text: &str| text.replace('\\', "\\\\").replace('"', "\\\"");
        let mut lines = vec!["digraph kg {".to_string(), "    rankdir=LR;".to_string()];
        for node in &self.nodes {
            let style = if node.is_focus {
                "shape=box, style=filled, fillcolor=\"#fff2b2\""
            } else {
                "shape=box"
            };
            lines.push(format!("    \"{}\" [{}];", mutate(&node.name), style));
        }
        for edge in &self.edges {
            let attrs = if edge.highlight {
                "color=\"#cc3333\", penwidth=2"
            } else {
                "color=\"#666666\""
            };
            lines.push(format!(
                "    \"{}\" -> \"{}\" [label=\"{}\", {}];",
                mutate(&edge.subject),
                mutate(&edge.object),
                mutate(&edge.predicate),
                attrs
            ));
        }
        lines.push("}".to_string());
        lines.join("\n")
    }

    /// JSON（规范序；机器消费/前端二次渲染）。
    pub fn render_json(&self) -> String {
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|node| {
                json!({
                    "name": node.name,
                    "depth": node.depth,
                    "focus": node.is_focus,
                })
            })
            .collect();
        let edges: Vec<Value> = self
            .edges
            .iter()
            .map(|edge| {
                json!({
                    "subject": edge.subject,
                    "predicate": edge.predicate,
                    "object": edge.object,
                    "highlight": edge.highlight,
                })
            })
            .collect();
        canonical_json(&json!({
            "focus": self.focus,
            "max_depth": self.max_depth,
            "nodes": nodes,
            "edges": edges,
        }))
    }

    /// 零依赖手绘 SVG（确定性分层布局）；超过 `MAX_SVG_NODES` 退化为 Mermaid。
    pub fn render_svg(&self) -> String {
        if self.nodes.len() > MAX_SVG_NODES {
            return format!(
                "<!-- 节点数 {} 超过 SVG 上限 {}：已退化为 Mermaid 视图 -->\n{}",
                self.nodes.len(),
                MAX_SVG_NODES,
                self.render_mermaid()
            );
        }
        let ids = self.node_ids();
        let width = 260 + self.nodes.iter().map(|n| n.depth).max().unwrap_or(0) * 240;
        let mut per_layer: BTreeMap<usize, usize> = BTreeMap::new();
        let mut y_of: BTreeMap<&str, usize> = BTreeMap::new();
        for node in &self.nodes {
            let slot = per_layer.entry(node.depth).or_insert(0);
            y_of.insert(node.name.as_str(), 30 + *slot * 70);
            *slot += 1;
        }
        let height = 60 + per_layer.values().copied().max().unwrap_or(1) * 70;
        let x_of = |depth: usize| 40 + depth * 240;

        let escape = |text: &str| {
            text.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
        };

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
            width, height, width, height
        ));
        svg.push_str(&format!(
            "  <title>知识子图：{}（{} 个节点，{} 条边）</title>\n",
            escape(&self.focus),
            self.nodes.len(),
            self.edges.len()
        ));
        for edge in &self.edges {
            let subject = &self.nodes[self.nodes.iter().position(|n| n.name == edge.subject).unwrap()];
            let object = &self.nodes[self.nodes.iter().position(|n| n.name == edge.object).unwrap()];
            let color = if edge.highlight { "#cc3333" } else { "#999999" };
            let stroke = if edge.highlight { 3 } else { 1 };
            svg.push_str(&format!(
                "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                x_of(subject.depth) + 90,
                y_of[edge.subject.as_str()] + 20,
                x_of(object.depth) + 90,
                y_of[edge.object.as_str()] + 20,
                color,
                stroke
            ));
        }
        for node in &self.nodes {
            let x = x_of(node.depth);
            let y = y_of[node.name.as_str()];
            let fill = if node.is_focus { "#fff2b2" } else { "#f5f5f5" };
            svg.push_str(&format!(
                "  <g><rect x=\"{}\" y=\"{}\" width=\"180\" height=\"40\" rx=\"6\" fill=\"{}\" stroke=\"#333333\"/><text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"14\">{}{}</text></g>\n",
                x,
                y,
                fill,
                x + 10,
                y + 25,
                if node.is_focus { "★ " } else { "" },
                escape(&node.name)
            ));
        }
        let _ = ids;
        svg.push_str("</svg>");
        svg
    }
}

// ============================ 对话时间线 ============================

/// 时间线的一轮（从 Chronicle 记录投影而来）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineTurn {
    pub seq: u64,
    pub kind: String,
    pub text: String,
    pub focus: Option<String>,
    pub inherited: bool,
    pub provenance: Vec<String>,
}

impl TimelineTurn {
    /// 从编年史记录投影；无「user/answer」文本的记录返回 None（如 note）。
    pub fn from_entry(entry: &ChronicleEntry) -> Option<Self> {
        let payload = entry.payload.as_object()?;
        let user = payload.get("user").and_then(Value::as_str);
        let answer = payload.get("answer").and_then(Value::as_str);
        let text = match (user, answer) {
            (Some(user), _) if !user.trim().is_empty() => user.to_string(),
            (_, Some(answer)) if !answer.trim().is_empty() => format!("答：{}", answer),
            _ => return None,
        };
        let focus = payload
            .get("focus")
            .and_then(Value::as_str)
            .filter(|focus| !focus.is_empty())
            .map(str::to_string);
        let inherited = payload
            .get("inherited")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let provenance = payload
            .get("provenance")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Some(TimelineTurn {
            seq: entry.seq,
            kind: entry.kind.clone(),
            text,
            focus,
            inherited,
            provenance,
        })
    }
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// 时间线 → Markdown 表格（审计视图：轮次/输入/类型/焦点/出处）。
pub fn render_timeline_markdown(turns: &[TimelineTurn]) -> String {
    if turns.is_empty() {
        return "（编年史为空：还没有可回放的对话）".to_string();
    }
    let mut lines = vec![
        "| 序号 | 输入/回答 | 类型 | 焦点 | 出处 |".to_string(),
        "| --- | --- | --- | --- | --- |".to_string(),
    ];
    for turn in turns {
        let focus = match (&turn.focus, turn.inherited) {
            (Some(focus), true) => format!("{}（继承）", focus),
            (Some(focus), false) => focus.clone(),
            (None, _) => "—".to_string(),
        };
        let provenance = if turn.provenance.is_empty() {
            "—".to_string()
        } else {
            turn.provenance.join("、")
        };
        lines.push(format!(
            "| {} | {} | {} | {} | {} |",
            turn.seq,
            cell(&turn.text),
            cell(&turn.kind),
            cell(&focus),
            cell(&provenance)
        ));
    }
    lines.join("\n")
}

/// `render_timeline_markdown` 的反渲染：Markdown 表格 → 时间线。
///
/// 解析律：在「安全域」（字段不包含换行；制表带来的换行→空格归一化见
/// [`cell`]）内 `parse_timeline_markdown(&render_timeline_markdown(t)) == t`。
/// 空视图文本解析为空列表；表头/列数不符一律报错而非静默丢失。
pub fn parse_timeline_markdown(text: &str) -> Result<Vec<TimelineTurn>, String> {
    if text.trim() == "（编年史为空：还没有可回放的对话）" {
        return Ok(Vec::new());
    }
    let mut lines = text.lines();
    let header = lines.next().ok_or("缺少表头")?;
    if header.trim() != "| 序号 | 输入/回答 | 类型 | 焦点 | 出处 |" {
        return Err(format!("表头不匹配：{}", header));
    }
    let separator = lines.next().ok_or("缺少分隔行")?;
    if separator.trim() != "| --- | --- | --- | --- | --- |" {
        return Err(format!("分隔行不匹配：{}", separator));
    }
    let mut turns = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let inner = line
            .trim()
            .strip_prefix("| ")
            .and_then(|rest| rest.strip_suffix(" |"))
            .ok_or_else(|| format!("第 {} 行不是表格行：{}", index + 3, line))?;
        let cells = split_escaped_pipes(inner);
        if cells.len() != 5 {
            return Err(format!(
                "第 {} 行应为 5 列，实为 {} 列：{}",
                index + 3,
                cells.len(),
                line
            ));
        }
        let seq: u64 = cells[0]
            .trim()
            .parse()
            .map_err(|error| format!("第 {} 行序号非法：{}", index + 3, error))?;
        let text_cell = cells[1].trim().replace("\\|", "|");
        let kind = cells[2].trim().replace("\\|", "|");
        let focus_cell = cells[3].trim();
        let (focus, inherited) = if focus_cell == "—" {
            (None, false)
        } else if let Some(stripped) = focus_cell.strip_suffix("（继承）") {
            (Some(stripped.replace("\\|", "|")), true)
        } else {
            (Some(focus_cell.replace("\\|", "|")), false)
        };
        let provenance = if cells[4].trim() == "—" {
            Vec::new()
        } else {
            cells[4]
                .trim()
                .split('、')
                .map(|item| item.trim().replace("\\|", "|"))
                .collect()
        };
        turns.push(TimelineTurn {
            seq,
            kind,
            text: text_cell,
            focus,
            inherited,
            provenance,
        });
    }
    Ok(turns)
}

/// 按未转义的 `|` 切分：`\|` 属于单元格内容（与渲染端 `cell` 转义互逆）。
fn split_escaped_pipes(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut cells = Vec::new();
    let mut start = 0;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'|' {
            continue;
        }
        let mut backslashes = 0;
        let mut probe = index;
        while probe > 0 && bytes[probe - 1] == b'\\' {
            backslashes += 1;
            probe -= 1;
        }
        if backslashes % 2 == 0 {
            cells.push(&text[start..index]);
            start = index + 1;
        }
    }
    cells.push(&text[start..]);
    cells
}

/// 时间线 → Mermaid 流程（焦点迁移一眼可见）。
pub fn render_timeline_flow(turns: &[TimelineTurn]) -> String {
    if turns.is_empty() {
        return "（编年史为空）".to_string();
    }
    let mut lines = vec!["graph LR".to_string()];
    let mut previous: Option<u64> = None;
    for turn in turns {
        let focus = match (&turn.focus, turn.inherited) {
            (Some(focus), true) => format!("<br/>焦点→{}（继承）", focus),
            (Some(focus), false) => format!("<br/>焦点→{}", focus),
            (None, _) => String::new(),
        };
        lines.push(format!(
            "    T{}[\"{}. {}{}\"]",
            turn.seq,
            turn.seq,
            mermaid_label(&turn.text),
            mermaid_label(&focus)
        ));
        if let Some(previous) = previous {
            lines.push(format!("    T{} --> T{}", previous, turn.seq));
        }
        previous = Some(turn.seq);
    }
    lines.join("\n")
}

/// `mermaid_label` 的逆：`#quot;` → `"`，`<br/>` → 换行。
fn mermaid_unescape(label: &str) -> String {
    label.replace("#quot;", "\"").replace("<br/>", "\n")
}

/// `render_timeline_flow` 的反渲染：Mermaid 流程 → 时间线。
///
/// 解析律：在「安全域」（正文不含字面 `<br/>焦点→`；焦点不以 `（继承）`
/// 结尾）内 `parse_timeline_flow(&render_timeline_flow(t)) == t`，其中
/// `kind` 固定为 `"turn"`——流程图无损承载 seq/正文/焦点/继承，但不编码
/// `kind`（该字段由表格视图承载）。同时校验流程结构本身：`T{n}` 单调
/// 递增且逐条以 `-->` 链到前一轮——渲染出的图若断链，解析必须报错而不是
/// 假装成功。
pub fn parse_timeline_flow(text: &str) -> Result<Vec<TimelineTurn>, String> {
    if text.trim() == "（编年史为空）" {
        return Ok(Vec::new());
    }
    let mut lines = text.lines();
    if lines.next() != Some("graph LR") {
        return Err("流程缺少 graph LR 头".to_string());
    }
    let mut turns: Vec<TimelineTurn> = Vec::new();
    // 渲染序：T1；T2；T1-->T2；T3；T2-->T3；……即每个 n≥2 的节点后必跟其入边。
    let mut expecting_edge = false;
    for line in lines {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let body = line
            .strip_prefix("    ")
            .ok_or_else(|| format!("流程行缺少缩进：{}", line))?;
        if let Some(edge) = body.strip_prefix('T') {
            if let Some((from, to)) = edge.split_once(" --> T") {
                if !expecting_edge {
                    return Err(format!("意外的边（上一行不是节点）：{}", line));
                }
                let from: u64 = from
                    .parse()
                    .map_err(|error| format!("边起点非法：{} ({})", from, error))?;
                let to: u64 = to
                    .parse()
                    .map_err(|error| format!("边终点非法：{} ({})", to, error))?;
                let expected_from = turns[turns.len() - 2].seq;
                let expected_to = turns[turns.len() - 1].seq;
                if from != expected_from || to != expected_to {
                    return Err(format!(
                        "流程断链：T{} --> T{}（应为 T{} --> T{}）",
                        from, to, expected_from, expected_to
                    ));
                }
                expecting_edge = false;
                continue;
            }
            if expecting_edge {
                return Err(format!(
                    "流程断链：T{} 后缺少 T{} --> T{} 边",
                    turns.last().map(|turn| turn.seq).unwrap_or(0),
                    turns[turns.len() - 2].seq,
                    turns.last().map(|turn| turn.seq).unwrap_or(0)
                ));
            }
            let open = edge.find("[\"").ok_or_else(|| format!("节点行缺少标签：{}", line))?;
            let seq_text = &edge[..open];
            let seq: u64 = seq_text
                .parse()
                .map_err(|error| format!("节点序号非法：{} ({})", seq_text, error))?;
            let label = edge[open + 2..]
                .strip_suffix("\"]")
                .ok_or_else(|| format!("节点标签未闭合：{}", line))?;
            let expected_prefix = format!("{}. ", seq);
            let rest = label
                .strip_prefix(&expected_prefix)
                .ok_or_else(|| format!("节点标签前缀不匹配：{}", label))?;
            let (raw_text, focus_part) = match rest.rfind("<br/>焦点→") {
                Some(position) => (
                    &rest[..position],
                    Some(&rest[position + "<br/>焦点→".len()..]),
                ),
                None => (rest, None),
            };
            let (focus, inherited) = match focus_part {
                None => (None, false),
                Some(part) => match part.strip_suffix("（继承）") {
                    Some(stripped) => (Some(mermaid_unescape(stripped)), true),
                    None => (Some(mermaid_unescape(part)), false),
                },
            };
            turns.push(TimelineTurn {
                seq,
                kind: "turn".to_string(),
                text: mermaid_unescape(raw_text),
                focus,
                inherited,
                provenance: Vec::new(),
            });
            if turns.len() >= 2 {
                expecting_edge = true;
            }
        } else {
            return Err(format!("无法识别的流程行：{}", line));
        }
    }
    if expecting_edge {
        return Err("流程缺少末尾入边".to_string());
    }
    Ok(turns)
}

// ============================ 覆盖报告 ============================

/// 覆盖检查输入（从导入事实清单投影，不依赖 schema/domain/range）。
#[derive(Debug, Clone, Default)]
pub struct CoverageInput {
    /// 显式登记的实体（如 JSON-LD 的 Entity 条目）。
    pub entities: Vec<String>,
    /// 关系三元组。
    pub edges: Vec<(String, String, String)>,
    /// 字面量属性：(实体, 属性键)。
    pub attributes: Vec<(String, String)>,
    /// 已登记谓词（谓词画像 + 配置声明），为空则跳过未登记检查。
    pub registered_predicates: Vec<String>,
}

const COVERAGE_LIST_CAP: usize = 20;

fn push_capped(lines: &mut Vec<String>, items: &[String]) {
    for item in items.iter().take(COVERAGE_LIST_CAP) {
        lines.push(format!("    · {}", item));
    }
    if items.len() > COVERAGE_LIST_CAP {
        lines.push(format!("    ……其余 {} 个省略", items.len() - COVERAGE_LIST_CAP));
    }
}

/// 确定性覆盖/质量报告。
pub fn render_coverage_report(input: &CoverageInput) -> String {
    let entities: BTreeSet<&str> = input.entities.iter().map(String::as_str).collect();
    let mut lines = vec!["知识覆盖报告（不依赖 schema 的确定性检查）：".to_string()];
    lines.push(format!(
        "  实体 {} 个，关系 {} 条，字面量属性 {} 项，已登记谓词 {} 种",
        entities.len(),
        input.edges.len(),
        input.attributes.len(),
        input.registered_predicates.len()
    ));

    if entities.is_empty() {
        lines.push("  悬空引用/孤立实体：无实体元数据可对照，检查跳过。".to_string());
    } else {
        let mut dangling: BTreeSet<&str> = BTreeSet::new();
        let mut connected: BTreeSet<&str> = BTreeSet::new();
        for (subject, _, object) in &input.edges {
            for endpoint in [subject.as_str(), object.as_str()] {
                connected.insert(endpoint);
                if !entities.contains(endpoint) {
                    dangling.insert(endpoint);
                }
            }
        }
        let dangling: Vec<String> = dangling.iter().map(|name| name.to_string()).collect();
        lines.push(format!(
            "  悬空引用（出现在关系中但未作为实体登记）：{} 个",
            dangling.len()
        ));
        push_capped(&mut lines, &dangling);

        let attributed: BTreeSet<&str> = input
            .attributes
            .iter()
            .map(|(entity, _)| entity.as_str())
            .collect();
        let isolated: Vec<String> = entities
            .iter()
            .filter(|entity| {
                !connected.contains(**entity) && !attributed.contains(**entity)
            })
            .map(|name| name.to_string())
            .collect();
        lines.push(format!("  孤立实体（既无属性也无关系）：{} 个", isolated.len()));
        push_capped(&mut lines, &isolated);
    }

    if input.registered_predicates.is_empty() {
        lines.push("  未登记谓词：谓词画像为空，检查跳过。".to_string());
    } else {
        let registered: BTreeSet<&str> =
            input.registered_predicates.iter().map(String::as_str).collect();
        let mut unregistered: BTreeSet<&str> = BTreeSet::new();
        for (_, predicate, _) in &input.edges {
            if !registered.contains(predicate.as_str()) {
                unregistered.insert(predicate);
            }
        }
        for (_, key) in &input.attributes {
            if !registered.contains(key.as_str()) {
                unregistered.insert(key);
            }
        }
        let unregistered: Vec<String> =
            unregistered.iter().map(|name| name.to_string()).collect();
        lines.push(format!("  未登记谓词（画像/配置均未声明）：{} 种", unregistered.len()));
        push_capped(&mut lines, &unregistered);
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::{Chronicle, FsyncPolicy};
    use crate::resolution_engine::{Atom, ProofNode, ProofSource};
    use tempfile::TempDir;

    fn leaf(label: &str) -> ProofViewNode {
        ProofViewNode {
            label: label.to_string(),
            canonical: label.to_string(),
            source: ViewProofSource::Fact,
            confidence: 1.0,
            children: Vec::new(),
        }
    }

    fn sample_forest() -> Vec<ProofViewNode> {
        vec![ProofViewNode {
            label: "师祖(张三, 王五)".to_string(),
            canonical: "师祖(张三, 王五)".to_string(),
            source: ViewProofSource::Rule("链接预测规则 师傅∘师傅".to_string()),
            confidence: 0.81,
            children: vec![
                leaf("师傅(张三, 李四)"),
                ProofViewNode {
                    label: "师傅(李四, 王五)".to_string(),
                    canonical: "师傅(李四, 王五)".to_string(),
                    source: ViewProofSource::Rule("用户规则".to_string()),
                    confidence: 0.9,
                    children: vec![leaf("师傅(李四, 王五)")],
                },
            ],
        }]
    }

    #[test]
    fn proof_node_count_matches_flattened_set() {
        let forest = sample_forest();
        let flattened = {
            fn walk(node: &ProofViewNode, count: &mut usize) {
                *count += 1;
                for child in &node.children {
                    walk(child, count);
                }
            }
            let mut count = 0;
            for root in &forest {
                walk(root, &mut count);
            }
            count
        };
        assert_eq!(proof_forest_node_count(&forest), flattened);
        assert_eq!(flattened, 4);
    }

    #[test]
    fn proof_mermaid_is_deterministic_and_escapes() {
        let forest = sample_forest();
        let first = render_proof_mermaid(&forest);
        let second = render_proof_mermaid(&forest);
        assert_eq!(first, second, "同输入两次渲染必须同字节");
        let expected = "graph TD\n    n0[\"师祖(张三, 王五)（由 链接预测规则 师傅∘师傅，置信度 0.81）\"]\n    n0 --> n1\n    n1[\"师傅(张三, 李四)（事实）\"]\n    n0 --> n2\n    n2[\"师傅(李四, 王五)（由 用户规则，置信度 0.90）\"]\n    n2 --> n3\n    n3[\"师傅(李四, 王五)（事实）\"]";
        assert_eq!(first, expected);
        assert!(first.contains("n3"));
    }

    #[test]
    fn proof_mermaid_escapes_quotes_and_newlines() {
        let forest = vec![ProofViewNode {
            label: "带\"引号\"\n与换行".to_string(),
            canonical: "带引号与换行".to_string(),
            source: ViewProofSource::Fact,
            confidence: 1.0,
            children: Vec::new(),
        }];
        let rendered = render_proof_mermaid(&forest);
        assert!(rendered.contains("#quot;"));
        assert!(rendered.contains("<br/>"));
        assert!(!rendered.contains('\n') || rendered.lines().count() == 2);
    }

    #[test]
    fn proof_ascii_golden() {
        let forest = sample_forest();
        let expected = "● 师祖(张三, 王五)（由 链接预测规则 师傅∘师傅，置信度 0.81）\n\
                        ├─ 师傅(张三, 李四)（事实）\n\
                        └─ 师傅(李四, 王五)（由 用户规则，置信度 0.90）\n\
                        \u{20}  └─ 师傅(李四, 王五)（事实）";
        assert_eq!(render_proof_ascii(&forest), expected);
    }

    #[test]
    fn proof_view_from_engine_node_uses_custom_labels() {
        let engine_node = ProofNode {
            atom: Atom::rel("师傅", "张三", "李四"),
            source: ProofSource::Rule {
                source: "测试规则".to_string(),
            },
            confidence: 0.5,
            children: vec![ProofNode {
                atom: Atom::rel("师傅", "张三", "李四"),
                source: ProofSource::Fact,
                confidence: 1.0,
                children: Vec::new(),
            }],
        };
        let view = ProofViewNode::from_proof_with(&engine_node, |node| {
            format!("NL:{}", node.atom.display())
        });
        assert_eq!(view.label, "NL:师傅(张三, 李四)");
        assert_eq!(view.children.len(), 1);
        assert!(matches!(view.source, ViewProofSource::Rule(ref source) if source == "测试规则"));
        assert_eq!(view.node_count(), 2);
    }

    #[test]
    fn subgraph_is_order_invariant() {
        let edges = vec![
            ("张三".to_string(), "师傅".to_string(), "李四".to_string()),
            ("李四".to_string(), "师傅".to_string(), "王五".to_string()),
            ("赵六".to_string(), "同事".to_string(), "张三".to_string()),
            ("王五".to_string(), "队友".to_string(), "赵六".to_string()),
        ];
        let highlight: HashSet<(String, String, String)> =
            [("张三".to_string(), "师傅".to_string(), "李四".to_string())]
                .into_iter()
                .collect();
        let mut shuffled = edges.clone();
        shuffled.reverse();
        shuffled.rotate_left(1);
        let left = KnowledgeSubgraph::bfs("张三", &edges, 2, &highlight);
        let right = KnowledgeSubgraph::bfs("张三", &shuffled, 2, &highlight);
        assert_eq!(left.render_mermaid(), right.render_mermaid());
        assert_eq!(left.render_dot(), right.render_dot());
        assert_eq!(left.render_json(), right.render_json());
        assert_eq!(left.render_svg(), right.render_svg());
    }

    #[test]
    fn subgraph_bfs_respects_depth_and_highlight() {
        let edges = vec![
            ("A".to_string(), "p".to_string(), "B".to_string()),
            ("B".to_string(), "p".to_string(), "C".to_string()),
            ("C".to_string(), "p".to_string(), "D".to_string()),
        ];
        let highlight = HashSet::new();
        let depth1 = KnowledgeSubgraph::bfs("A", &edges, 1, &highlight);
        assert_eq!(depth1.node_count(), 2);
        assert_eq!(depth1.edge_count(), 1);
        assert_eq!(depth1.nodes[0].depth, 0);
        assert!(depth1.nodes[0].is_focus, "焦点节点应标记 ★");
        assert!(depth1.render_mermaid().contains("★ A"));

        let depth2 = KnowledgeSubgraph::bfs("A", &edges, 2, &highlight);
        assert_eq!(depth2.node_count(), 3);
        assert_eq!(depth2.edge_count(), 2);
        assert_eq!(depth2.nodes[1].name, "B");

        let highlight: HashSet<(String, String, String)> =
            [("B".to_string(), "p".to_string(), "C".to_string())]
                .into_iter()
                .collect();
        let highlighted = KnowledgeSubgraph::bfs("A", &edges, 2, &highlight);
        assert!(highlighted.render_mermaid().contains("==>"));
        assert!(highlighted.render_dot().contains("penwidth=2"));
    }

    #[test]
    fn subgraph_svg_degrades_past_limit() {
        let mut edges = Vec::new();
        for index in 1..=(MAX_SVG_NODES + 20) {
            edges.push((
                "实体0".to_string(),
                "链".to_string(),
                format!("实体{}", index),
            ));
        }
        let subgraph = KnowledgeSubgraph::bfs("实体0", &edges, 1, &HashSet::new());
        let svg = subgraph.render_svg();
        assert!(svg.contains("退化为 Mermaid"), "{}", &svg[..80]);
        assert!(svg.contains("graph LR"));

        let small = KnowledgeSubgraph::bfs("实体0", &edges[..2], 2, &HashSet::new());
        let small_svg = small.render_svg();
        assert!(small_svg.starts_with("<svg"));
        assert!(small_svg.ends_with("</svg>"));
        assert!(small_svg.contains("★ 实体0"));
    }

    #[test]
    fn coverage_report_finds_three_classes() {
        let input = CoverageInput {
            entities: vec![
                "张三".to_string(),
                "李四".to_string(),
                "孤立".to_string(),
            ],
            edges: vec![
                ("张三".to_string(), "师傅".to_string(), "李四".to_string()),
                ("李四".to_string(), "师父".to_string(), "王五".to_string()),
            ],
            attributes: vec![("张三".to_string(), "身高".to_string())],
            registered_predicates: vec!["师傅".to_string(), "身高".to_string()],
        };
        let report = render_coverage_report(&input);
        assert!(report.contains("悬空引用"), "{}", report);
        assert!(report.contains("王五"), "{}", report);
        assert!(report.contains("孤立实体"), "{}", report);
        assert!(report.contains("孤立"), "{}", report);
        assert!(report.contains("未登记谓词"), "{}", report);
        assert!(report.contains("师父"), "{}", report);
        assert!(!report.contains("· 张三"), "已登记的实体不应误报：{}", report);
    }

    #[test]
    fn timeline_projects_chronicle_entries() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("timeline.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (mut chronicle, _) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        chronicle
            .append(
                "turn",
                json!({"user": "姚明是谁", "mode": "query", "focus": null, "inherited": false}),
            )
            .unwrap();
        chronicle
            .append(
                "turn",
                json!({"user": "他的身高呢", "mode": "query", "focus": "姚明", "inherited": true}),
            )
            .unwrap();
        chronicle
            .append_reply("姚明的身高是 226。", &["direct_answer".to_string()], &["身高".to_string()])
            .unwrap();

        let turns: Vec<TimelineTurn> = chronicle
            .entries()
            .iter()
            .filter_map(TimelineTurn::from_entry)
            .collect();
        assert_eq!(turns.len(), 3);
        assert!(turns[1].inherited);
        let markdown = render_timeline_markdown(&turns);
        assert!(markdown.contains("| 1 | 姚明是谁 | turn | — | — |"), "{}", markdown);
        assert!(markdown.contains("姚明（继承）"), "{}", markdown);
        assert!(markdown.contains("答：姚明的身高是 226。"), "{}", markdown);
        let flow = render_timeline_flow(&turns);
        assert!(flow.contains("T1 --> T2"), "{}", flow);
        assert!(flow.contains("T2 --> T3"), "{}", flow);
        assert_eq!(flow, render_timeline_flow(&turns), "渲染必须确定");
    }

    #[test]
    fn timeline_markdown_escapes_pipes() {
        let turns = vec![TimelineTurn {
            seq: 1,
            kind: "turn".to_string(),
            text: "A|B".to_string(),
            focus: None,
            inherited: false,
            provenance: Vec::new(),
        }];
        let markdown = render_timeline_markdown(&turns);
        assert!(markdown.contains("A\\|B"), "{}", markdown);
        assert!(render_timeline_markdown(&[]).contains("编年史为空"));
    }

    #[test]
    fn subgraph_json_parse_law_holds_for_hostile_strings() {
        let subgraph = KnowledgeSubgraph {
            focus: "焦点\"X\"\n多行".to_string(),
            max_depth: 2,
            nodes: vec![
                SubgraphNode {
                    name: "焦点\"X\"\n多行".to_string(),
                    depth: 0,
                    is_focus: true,
                },
                SubgraphNode {
                    name: "A|B\"]逃逸".to_string(),
                    depth: 1,
                    is_focus: false,
                },
                SubgraphNode {
                    name: "<br/>焦点→伪装".to_string(),
                    depth: 2,
                    is_focus: false,
                },
            ],
            edges: vec![
                SubgraphEdge {
                    subject: "焦点\"X\"\n多行".to_string(),
                    predicate: "谓词|带|竖线".to_string(),
                    object: "A|B\"]逃逸".to_string(),
                    highlight: true,
                },
                SubgraphEdge {
                    subject: "A|B\"]逃逸".to_string(),
                    predicate: "p".to_string(),
                    object: "<br/>焦点→伪装".to_string(),
                    highlight: false,
                },
            ],
        };
        let text = subgraph.render_json();
        let parsed = parse_json(&text).unwrap();
        assert_eq!(parsed, subgraph, "parse(render(x)) 必须逐字段等于 x");
        assert_eq!(parsed.render_json(), text, "解析后再渲染必须同字节（L2 不动点）");
        assert!(parse_json("{").is_err());
        assert!(parse_json("{\"focus\":1}").is_err());
    }

    #[test]
    fn timeline_markdown_parse_law_holds_on_safe_corpus() {
        let turns = vec![
            TimelineTurn {
                seq: 1,
                kind: "turn".to_string(),
                text: "姚明是谁".to_string(),
                focus: None,
                inherited: false,
                provenance: Vec::new(),
            },
            TimelineTurn {
                seq: 2,
                kind: "turn".to_string(),
                text: "A|B 与 \"引号\"（伪）".to_string(),
                focus: Some("姚明".to_string()),
                inherited: true,
                provenance: vec!["师傅".to_string(), "身高|别名".to_string()],
            },
            TimelineTurn {
                seq: 7,
                kind: "reply".to_string(),
                text: "答：226".to_string(),
                focus: Some("姚 明".to_string()),
                inherited: false,
                provenance: Vec::new(),
            },
        ];
        let markdown = render_timeline_markdown(&turns);
        assert_eq!(
            parse_timeline_markdown(&markdown).unwrap(),
            turns,
            "\n{}",
            markdown
        );
        assert_eq!(
            parse_timeline_markdown(&render_timeline_markdown(&[])).unwrap(),
            Vec::new()
        );
        assert!(parse_timeline_markdown("表头错误").is_err());
        assert!(parse_timeline_markdown("| 序号 | 输入/回答 | 类型 | 焦点 | 出处 |\n| --- | --- | --- | --- | --- |\n| x | y | z | — | — |").is_err());
    }

    #[test]
    fn timeline_flow_parse_law_holds_on_safe_corpus() {
        let turns = vec![
            TimelineTurn {
                seq: 1,
                kind: "turn".to_string(),
                text: "姚明是谁".to_string(),
                focus: None,
                inherited: false,
                provenance: Vec::new(),
            },
            TimelineTurn {
                seq: 2,
                kind: "turn".to_string(),
                text: "A|B 与 \"引号\"\n第二行".to_string(),
                focus: Some("姚明".to_string()),
                inherited: true,
                provenance: Vec::new(),
            },
            TimelineTurn {
                seq: 9,
                kind: "turn".to_string(),
                text: "他的身高呢".to_string(),
                focus: Some("姚 明".to_string()),
                inherited: false,
                provenance: Vec::new(),
            },
        ];
        let flow = render_timeline_flow(&turns);
        assert_eq!(parse_timeline_flow(&flow).unwrap(), turns, "\n{}", flow);
        assert_eq!(
            parse_timeline_flow(&render_timeline_flow(&[])).unwrap(),
            Vec::new()
        );
        let broken = flow.replace("T1 --> T2", "T1 --> T8");
        assert!(
            parse_timeline_flow(&broken).is_err(),
            "断链流程解析必须报错：{}",
            broken
        );
        assert!(parse_timeline_flow("graph LR\n    T1[\"无序号前缀\"]").is_err());
        assert!(parse_timeline_flow("graph LR\n    T1 --> T2").is_err());
    }
}

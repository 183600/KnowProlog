use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const DEFAULT_CLOSURE_MAX_DEPTH: usize = 8;
pub const DEFAULT_CLOSURE_MAX_NODES: usize = 512;
pub const DEFAULT_CLOSURE_MAX_EXPANSIONS: usize = 20_000;
pub const DEFAULT_CLOSURE_MAX_WITNESSES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ClosureOp {
    Exact(usize),
    Plus,
    Star,
}

impl ClosureOp {
    pub fn label(self) -> &'static str {
        match self {
            ClosureOp::Exact(_) => "精确跳数",
            ClosureOp::Plus => "传递闭包",
            ClosureOp::Star => "自反传递闭包",
        }
    }

    pub fn requires_transitive_law(self) -> bool {
        !matches!(self, ClosureOp::Exact(_))
    }

    pub fn requested_depth(self, max_depth: usize) -> usize {
        match self {
            ClosureOp::Exact(hops) => hops.min(max_depth),
            ClosureOp::Plus | ClosureOp::Star => max_depth,
        }
    }

    /// 解析用户词面：`plus`/`star`/`exact:N`，同时接受 `+`、`*` 与中文别名。
    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim().to_ascii_lowercase();
        match raw.as_str() {
            "plus" | "+" | "传递" => Some(ClosureOp::Plus),
            "star" | "*" | "自反" => Some(ClosureOp::Star),
            other => {
                let digits = other
                    .strip_prefix("exact:")
                    .or_else(|| other.strip_prefix("exact="))
                    .or_else(|| other.strip_prefix("exact"))
                    .unwrap_or(other);
                digits
                    .parse::<usize>()
                    .ok()
                    .filter(|hops| *hops > 0)
                    .map(ClosureOp::Exact)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureBudget {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_expansions: usize,
    pub max_witnesses: usize,
}

impl Default for ClosureBudget {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_CLOSURE_MAX_DEPTH,
            max_nodes: DEFAULT_CLOSURE_MAX_NODES,
            max_expansions: DEFAULT_CLOSURE_MAX_EXPANSIONS,
            max_witnesses: DEFAULT_CLOSURE_MAX_WITNESSES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosureTruncation {
    Depth,
    Nodes,
    Expansions,
}

impl ClosureTruncation {
    pub fn label(self) -> &'static str {
        match self {
            ClosureTruncation::Depth => "深度上限",
            ClosureTruncation::Nodes => "节点上限",
            ClosureTruncation::Expansions => "扩展上限",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClosureError {
    MissingTransitiveLaw { operator: ClosureOp },
}

impl std::fmt::Display for ClosureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClosureError::MissingTransitiveLaw { operator } => write!(
                formatter,
                "{}需要一条已启用的传递律，精确跳数不需要",
                operator.label()
            ),
        }
    }
}

impl std::error::Error for ClosureError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureOutcome {
    pub start: String,
    pub operator: ClosureOp,
    pub witnesses: BTreeMap<String, Vec<String>>,
    pub reached_count: usize,
    pub expansions: usize,
    pub visited: usize,
    pub truncated: Option<ClosureTruncation>,
    pub witnesses_truncated: bool,
    pub law: Option<String>,
}

impl ClosureOutcome {
    pub fn complete(&self) -> bool {
        self.truncated.is_none()
    }

    pub fn count_is_lower_bound(&self) -> bool {
        !self.complete()
    }

    pub fn hop_count_of(&self, target: &str) -> Option<usize> {
        self.witnesses
            .get(target)
            .map(|path| path.len().saturating_sub(1))
    }

    pub fn render_witness(&self, target: &str) -> Option<String> {
        let path = self.witnesses.get(target)?;
        if path.is_empty() {
            return Some(self.start.clone());
        }
        Some(path.join(" → "))
    }

    pub fn targets_by_distance(&self) -> Vec<(&str, usize)> {
        let mut targets: Vec<(&str, usize)> = self
            .witnesses
            .iter()
            .map(|(target, path)| (target.as_str(), path.len().saturating_sub(1)))
            .collect();
        targets.sort_by(|(left_name, left_hops), (right_name, right_hops)| {
            left_hops
                .cmp(right_hops)
                .then_with(|| left_name.cmp(right_name))
        });
        targets
    }
}

pub fn evaluate_closure<F>(
    start: &str,
    operator: ClosureOp,
    budget: ClosureBudget,
    law: Option<&str>,
    mut neighbors: F,
) -> Result<ClosureOutcome, ClosureError>
where
    F: FnMut(&str) -> Vec<String>,
{
    if operator.requires_transitive_law() && law.is_none() {
        return Err(ClosureError::MissingTransitiveLaw { operator });
    }

    let exact_depth = match operator {
        ClosureOp::Exact(hops) => Some(hops),
        ClosureOp::Plus | ClosureOp::Star => None,
    };
    let requested_depth = operator.requested_depth(budget.max_depth);
    let mut truncated = match exact_depth {
        Some(hops) if hops > budget.max_depth => Some(ClosureTruncation::Depth),
        _ => None,
    };

    let mut witnesses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut retained_witnesses = 0usize;
    let mut witnesses_truncated = false;
    let mut discovered: BTreeSet<String> = BTreeSet::from([start.to_string()]);
    let mut expansion_count = 0usize;
    let mut exact_targets: BTreeSet<String> = BTreeSet::new();

    if operator == ClosureOp::Star {
        witnesses.insert(start.to_string(), Vec::new());
        retained_witnesses += 1;
    }

    let mut frontier: Vec<Vec<String>> = vec![vec![start.to_string()]];
    let mut depth = 0usize;
    let mut stopped_by_budget = false;

    'levels: while depth < requested_depth && !frontier.is_empty() {
        depth += 1;
        let mut next_frontier: Vec<Vec<String>> = Vec::new();
        let mut layer_targets: BTreeSet<String> = BTreeSet::new();

        for path in &frontier {
            let node = path.last().map(String::as_str).unwrap_or(start);
            let mut candidates = neighbors(node);
            candidates.sort();
            candidates.dedup();
            for candidate in candidates {
                expansion_count += 1;
                if expansion_count > budget.max_expansions {
                    truncated = Some(ClosureTruncation::Expansions);
                    stopped_by_budget = true;
                    break 'levels;
                }

                let is_new = !discovered.contains(&candidate);
                if is_new && discovered.len() >= budget.max_nodes {
                    truncated = Some(ClosureTruncation::Nodes);
                    stopped_by_budget = true;
                    break 'levels;
                }
                if is_new {
                    discovered.insert(candidate.clone());
                } else if exact_depth.is_none() {
                    continue;
                }

                let mut next_path = path.clone();
                next_path.push(candidate.clone());
                if !layer_targets.insert(candidate.clone()) {
                    continue;
                }
                if exact_depth == Some(depth) {
                    exact_targets.insert(candidate.clone());
                    if retained_witnesses < budget.max_witnesses {
                        witnesses.insert(candidate.clone(), next_path.clone());
                        retained_witnesses += 1;
                    } else {
                        witnesses_truncated = true;
                    }
                    continue;
                }
                if exact_depth.is_none() {
                    if retained_witnesses < budget.max_witnesses {
                        witnesses.insert(candidate.clone(), next_path.clone());
                        retained_witnesses += 1;
                    } else {
                        witnesses_truncated = true;
                    }
                }
                next_frontier.push(next_path);
            }
        }
        frontier = next_frontier;
    }

    if !stopped_by_budget
        && truncated.is_none()
        && exact_depth.is_none()
        && depth >= budget.max_depth
    {
        let has_more = frontier.iter().any(|path| {
            let node = path.last().map(String::as_str).unwrap_or(start);
            neighbors(node)
                .into_iter()
                .any(|candidate| !discovered.contains(&candidate))
        });
        if has_more {
            truncated = Some(ClosureTruncation::Depth);
        }
    }

    let reached_count = match operator {
        ClosureOp::Star => discovered.len(),
        ClosureOp::Plus => discovered.len().saturating_sub(1),
        ClosureOp::Exact(_) => exact_targets.len(),
    };

    Ok(ClosureOutcome {
        start: start.to_string(),
        operator,
        witnesses,
        reached_count,
        expansions: expansion_count,
        visited: discovered.len().saturating_sub(1),
        truncated,
        witnesses_truncated,
        law: law.map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> BTreeMap<&'static str, Vec<&'static str>> {
        BTreeMap::from([
            ("甲", vec!["乙"]),
            ("乙", vec!["丙", "丁"]),
            ("丙", vec!["戊"]),
            ("丁", vec!["戊"]),
            ("戊", vec![]),
            ("己", vec!["甲", "己"]),
        ])
    }

    fn neighbors<'a>(
        graph: &'a BTreeMap<&'static str, Vec<&'static str>>,
    ) -> impl FnMut(&str) -> Vec<String> + 'a {
        move |node| {
            graph
                .get(node)
                .map(|values| values.iter().map(|value| value.to_string()).collect())
                .unwrap_or_default()
        }
    }

    #[test]
    fn plus_requires_law_and_reports_distinct_reachable_nodes() {
        let graph = graph();
        let missing = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget::default(),
            None,
            neighbors(&graph),
        );
        assert_eq!(
            missing,
            Err(ClosureError::MissingTransitiveLaw {
                operator: ClosureOp::Plus
            })
        );

        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget::default(),
            Some("law:transitive:上司"),
            neighbors(&graph),
        )
        .expect("law authorizes plus closure");
        assert_eq!(outcome.reached_count, 4);
        assert!(outcome.complete());
        assert_eq!(
            outcome.witnesses.get("戊"),
            Some(&vec![
                "甲".to_string(),
                "乙".to_string(),
                "丁".to_string(),
                "戊".to_string()
            ])
        );
        assert_eq!(outcome.hop_count_of("戊"), Some(3));
        assert_eq!(outcome.law.as_deref(), Some("law:transitive:上司"));
    }

    #[test]
    fn star_includes_start_and_plus_does_not() {
        let graph = graph();
        let plus = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget::default(),
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert!(!plus.witnesses.contains_key("甲"));

        let star = evaluate_closure(
            "甲",
            ClosureOp::Star,
            ClosureBudget::default(),
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert_eq!(star.reached_count, 5);
        assert_eq!(star.witnesses.get("甲"), Some(&Vec::new()));
        assert_eq!(star.render_witness("甲").as_deref(), Some("甲"));
    }

    #[test]
    fn exact_hops_do_not_need_a_transitive_law() {
        let graph = graph();
        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Exact(2),
            ClosureBudget::default(),
            None,
            neighbors(&graph),
        )
        .expect("exact composition is plain relation composition");
        let targets: Vec<&str> = outcome.witnesses.keys().map(String::as_str).collect();
        assert_eq!(targets, vec!["丁", "丙"]);
        assert!(!outcome.witnesses.contains_key("乙"));
        assert!(!outcome.witnesses.contains_key("戊"));
    }

    #[test]
    fn depth_and_node_budgets_are_visible() {
        let deep = BTreeMap::from([
            ("甲", vec!["乙"]),
            ("乙", vec!["丙"]),
            ("丙", vec!["丁"]),
            ("丁", vec!["戊"]),
        ]);
        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget {
                max_depth: 2,
                ..ClosureBudget::default()
            },
            Some("law"),
            neighbors(&deep),
        )
        .unwrap();
        assert!(!outcome.complete());
        assert_eq!(outcome.truncated, Some(ClosureTruncation::Depth));
        assert_eq!(outcome.reached_count, 2);
        assert!(outcome.count_is_lower_bound());

        let node_limited = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget {
                max_nodes: 2,
                ..ClosureBudget::default()
            },
            Some("law"),
            neighbors(&deep),
        )
        .unwrap();
        assert_eq!(node_limited.truncated, Some(ClosureTruncation::Nodes));
        assert_eq!(node_limited.reached_count, 1);
        assert!(node_limited.count_is_lower_bound());
    }

    #[test]
    fn depth_budget_that_exactly_fits_is_complete() {
        let graph = BTreeMap::from([("甲", vec!["乙"]), ("乙", vec!["丙"]), ("丙", vec![])]);
        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget {
                max_depth: 2,
                ..ClosureBudget::default()
            },
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert!(outcome.complete());
        assert_eq!(outcome.truncated, None);
        assert_eq!(outcome.reached_count, 2);
    }

    #[test]
    fn expansion_budget_stops_search_and_marks_lower_bound() {
        let graph = graph();
        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget {
                max_expansions: 1,
                ..ClosureBudget::default()
            },
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert_eq!(outcome.truncated, Some(ClosureTruncation::Expansions));
        assert!(outcome.count_is_lower_bound());
        assert_eq!(outcome.reached_count, 1);
    }

    #[test]
    fn witness_limit_does_not_shrink_the_distinct_count() {
        let graph = graph();
        let outcome = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget {
                max_witnesses: 1,
                ..ClosureBudget::default()
            },
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert_eq!(outcome.reached_count, 4);
        assert_eq!(outcome.witnesses.len(), 1);
        assert!(outcome.witnesses_truncated);
        assert!(outcome.complete());
    }

    #[test]
    fn cycles_are_deduplicated_and_ordered_by_shortest_distance() {
        let graph = graph();
        let outcome = evaluate_closure(
            "己",
            ClosureOp::Plus,
            ClosureBudget::default(),
            Some("law"),
            neighbors(&graph),
        )
        .unwrap();
        assert_eq!(outcome.reached_count, 5);
        let ordered = outcome.targets_by_distance();
        assert_eq!(ordered.first().map(|(name, _)| *name), Some("甲"));
        assert_eq!(
            ordered.last().map(|(name, hops)| (*name, *hops)),
            Some(("戊", 4))
        );
    }

    #[test]
    fn operator_parse_accepts_english_chinese_and_exact_forms() {
        assert_eq!(ClosureOp::parse("plus"), Some(ClosureOp::Plus));
        assert_eq!(ClosureOp::parse("+"), Some(ClosureOp::Plus));
        assert_eq!(ClosureOp::parse("传递"), Some(ClosureOp::Plus));
        assert_eq!(ClosureOp::parse("star"), Some(ClosureOp::Star));
        assert_eq!(ClosureOp::parse("*"), Some(ClosureOp::Star));
        assert_eq!(ClosureOp::parse("自反"), Some(ClosureOp::Star));
        assert_eq!(ClosureOp::parse("exact:2"), Some(ClosureOp::Exact(2)));
        assert_eq!(ClosureOp::parse("exact=3"), Some(ClosureOp::Exact(3)));
        assert_eq!(ClosureOp::parse("exact4"), Some(ClosureOp::Exact(4)));
        assert_eq!(ClosureOp::parse("4"), Some(ClosureOp::Exact(4)));
        assert_eq!(ClosureOp::parse("exact:0"), None);
        assert_eq!(ClosureOp::parse("sometimes"), None);
    }
}

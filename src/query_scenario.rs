//! 查询剧本：把「多轮交互式查询」写成可执行的回合轨迹契约。
//!
//! 查询板（[`crate::query_board`]）把「当前查询」变成可编辑的工作记忆；
//! 本模块给它配一份可回放的证据：每个剧本声明一组事实、一串用户输入，
//! 以及每一步期望的状态转移（算子 / 行数 / 页号 / 选中行 / 焦点 / 证据）。
//!
//! - 表现层断言（`answer_contains`）之外，还断言**结构与状态**：
//!   翻页有没有前进、筛选有没有丢约束、选行有没有错位、知识版本推进后
//!   旧帧有没有被标旧——这些用字符串断言测不出来，只能对轨迹下契约；
//! - 剧本是自包含的：内置一个从剧本事实构建的 [`ScenarioKnowledge`]，
//!   不依赖 Oxigraph、不依赖网络，因此既是 CI 回归资产，也是用户可读的
//!   查询脚本与 bug 的最小复现；
//! - 同一份剧本可对任意 [`DialogueKnowledge`] 后端回放
//!   （[`run_scenario_against`]），便于把「真实推理器」的行为与脚本对齐。
//!
//! 边界：运行器覆盖对话管理、查询板执行与回答渲染，不覆盖 CLI 命令分发
//! 与 HTTP 层；CLI/HTTP 的一致性由各自的集成测试负责。

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::{DialogueKnowledge, DialogueManager};
use crate::query_board::BoardView;

/// 一个查询剧本：事实 + 回合轨迹契约。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryScenario {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// 剧本自带的实体（属性为单值映射，够表达集合/数值查询即可）。
    #[serde(default)]
    pub entities: Vec<ScenarioEntity>,
    /// 剧本自带的关系。
    #[serde(default)]
    pub relations: Vec<ScenarioRelation>,
    pub turns: Vec<ScenarioTurn>,
}

/// 剧本实体：名字 + 类型 + 属性表。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioEntity {
    pub name: String,
    #[serde(default, rename = "type")]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// 剧本关系：from -[predicate]-> to。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioRelation {
    pub from: String,
    pub predicate: String,
    pub to: String,
}

/// 一轮输入与它的轨迹契约。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioTurn {
    pub input: String,
    #[serde(default)]
    pub expect: TurnExpectation,
    /// 执行本轮前把知识版本推进到该值（用于验证「旧帧标旧」）。
    #[serde(default)]
    pub set_revision: Option<u64>,
}

/// 一轮期望：全部字段可选，缺省即不断言。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnExpectation {
    /// 查询板算子（`scan / filter / sort / select / compare / page / undo / ...`）。
    #[serde(default)]
    pub op: Option<String>,
    /// 是否应被接管（`false` 表示应让位给其他管线）。
    #[serde(default)]
    pub answered: Option<bool>,
    /// 回答必须包含的子串。
    #[serde(default)]
    pub answer_contains: Vec<String>,
    /// 回答不得包含的子串。
    #[serde(default)]
    pub answer_excludes: Vec<String>,
    /// 当前帧的行数。
    #[serde(default)]
    pub rows: Option<usize>,
    /// 当前页码（1 起）。
    #[serde(default)]
    pub page: Option<usize>,
    /// 总页数。
    #[serde(default)]
    pub pages: Option<usize>,
    /// 当前选中行的主语。
    #[serde(default)]
    pub selected: Option<String>,
    /// 焦点实体。
    #[serde(default)]
    pub focus: Option<String>,
    /// 查询板帧数。
    #[serde(default)]
    pub frames: Option<usize>,
    /// 当前帧是否应被标记为旧版本。
    #[serde(default)]
    pub stale: Option<bool>,
    /// 本轮证据必须包含的子串。
    #[serde(default)]
    pub provenance_contains: Vec<String>,
    /// 查询板是否处于激活状态。
    #[serde(default)]
    pub board_active: Option<bool>,
}

/// 单步回放结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioStepReport {
    pub input: String,
    pub passed: bool,
    pub op: Option<String>,
    pub focus: Option<String>,
    pub answer: String,
    pub board: Option<BoardView>,
    pub failures: Vec<String>,
}

/// 剧本回放报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub name: String,
    pub passed: bool,
    pub passed_steps: usize,
    pub total_steps: usize,
    pub steps: Vec<ScenarioStepReport>,
}

impl ScenarioReport {
    /// 面向终端的中文报告：每步一行，失败给出原因与实际回答。
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "剧本「{}」：{}/{} 步通过{}",
            self.name,
            self.passed_steps,
            self.total_steps,
            if self.passed {
                "。"
            } else {
                "（有失败）。"
            }
        );
        for (index, step) in self.steps.iter().enumerate() {
            let marker = if step.passed { '✓' } else { '✗' };
            let op = step.op.as_deref().unwrap_or("未接管");
            output.push_str(&format!(
                "\n  {} {}. {} [{}]",
                marker,
                index + 1,
                step.input,
                op
            ));
            if !step.passed {
                for failure in &step.failures {
                    output.push_str(&format!("\n       {}", failure));
                }
                let excerpt = excerpt(&step.answer, 60);
                if !excerpt.is_empty() {
                    output.push_str(&format!("\n       实际回答：{}", excerpt));
                }
            }
        }
        output
    }
}

/// 解析剧本 TOML。
pub fn parse_scenario(text: &str) -> Result<QueryScenario, toml::de::Error> {
    toml::from_str(text)
}

/// 读取并运行剧本文件。
pub fn run_file(path: impl AsRef<Path>) -> Result<ScenarioReport, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("读取 {} 失败：{}", path.display(), error))?;
    let scenario = parse_scenario(&text).map_err(|error| format!("解析剧本失败：{}", error))?;
    Ok(run_scenario(&scenario))
}

/// 运行剧本：用剧本内自带的事实构建知识源。
pub fn run_scenario(scenario: &QueryScenario) -> ScenarioReport {
    let knowledge = ScenarioKnowledge::from_scenario(scenario);
    run_scenario_with(scenario, &knowledge, |turn| {
        if let Some(revision) = turn.set_revision {
            knowledge.set_revision(revision);
        }
    })
}

/// 对任意知识源回放剧本（忽略 `set_revision`，知识版本由后端决定）。
pub fn run_scenario_against(
    scenario: &QueryScenario,
    knowledge: &dyn DialogueKnowledge,
) -> ScenarioReport {
    run_scenario_with(scenario, knowledge, |_| {})
}

fn run_scenario_with<F>(
    scenario: &QueryScenario,
    knowledge: &dyn DialogueKnowledge,
    mut before_turn: F,
) -> ScenarioReport
where
    F: FnMut(&ScenarioTurn),
{
    let mut manager = DialogueManager::new();
    let mut steps = Vec::with_capacity(scenario.turns.len());
    for turn in &scenario.turns {
        before_turn(turn);
        let reply = manager
            .plan_pure_dialogue(&turn.input, knowledge)
            .or_else(|| manager.plan_reply(&turn.input, knowledge));
        let view = manager.board().view(knowledge.revision());
        let op = reply
            .as_ref()
            .and_then(|reply| reply.plan.first())
            .map(|plan| plan.strip_prefix("board:").unwrap_or(plan).to_string());
        let focus = manager.focus_entity().map(str::to_string);
        let answer = reply
            .as_ref()
            .map(|reply| reply.text.clone())
            .unwrap_or_default();
        let provenance = manager
            .turns()
            .last()
            .map(|turn| turn.provenance.clone())
            .unwrap_or_default();
        let actual = StepActual {
            reply_answered: reply.is_some(),
            op: op.as_deref(),
            answer: &answer,
            view: view.as_ref(),
            board_active: manager.board().is_active(),
            frame_count: manager.board().frame_count(),
            selected: manager.board().current().and_then(|frame| {
                frame
                    .selected
                    .and_then(|index| frame.rows.get(index))
                    .map(|row| row.subject.as_str())
            }),
            focus: focus.as_deref(),
            provenance: &provenance,
        };
        let failures = check_expectations(&turn.expect, &actual);
        steps.push(ScenarioStepReport {
            input: turn.input.clone(),
            passed: failures.is_empty(),
            op,
            focus,
            answer,
            board: view,
            failures,
        });
    }
    let passed_steps = steps.iter().filter(|step| step.passed).count();
    ScenarioReport {
        name: scenario.name.clone(),
        passed: passed_steps == steps.len(),
        passed_steps,
        total_steps: steps.len(),
        steps,
    }
}

/// 一步回放中可观察到的事实（断言只读这个快照）。
struct StepActual<'a> {
    reply_answered: bool,
    op: Option<&'a str>,
    answer: &'a str,
    view: Option<&'a BoardView>,
    board_active: bool,
    frame_count: usize,
    selected: Option<&'a str>,
    focus: Option<&'a str>,
    provenance: &'a [String],
}

fn check_expectations(expect: &TurnExpectation, actual: &StepActual<'_>) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(expected) = &expect.op {
        match actual.op {
            Some(op) if op == expected => {}
            Some(op) => failures.push(format!("期望算子「{}」，实际「{}」", expected, op)),
            None => failures.push(format!("期望算子「{}」，但本轮未被查询板接管", expected)),
        }
    }
    if let Some(expected) = expect.answered {
        if expected != actual.reply_answered {
            failures.push(format!(
                "期望{}回答，实际{}回答",
                if expected { "有" } else { "无" },
                if actual.reply_answered { "有" } else { "无" }
            ));
        }
    }
    for needle in &expect.answer_contains {
        if !actual.answer.contains(needle.as_str()) {
            failures.push(format!("回答缺少「{}」", needle));
        }
    }
    for needle in &expect.answer_excludes {
        if actual.answer.contains(needle.as_str()) {
            failures.push(format!("回答不应包含「{}」", needle));
        }
    }
    if let Some(expected) = expect.rows {
        let actual_rows = actual.view.map(|view| view.rows);
        if actual_rows != Some(expected) {
            failures.push(format!(
                "期望 {} 行，实际 {}",
                expected,
                actual_rows
                    .map(|rows| rows.to_string())
                    .unwrap_or_else(|| "无查询帧".to_string())
            ));
        }
    }
    if let Some(expected) = expect.page {
        let actual_page = actual.view.map(|view| view.page);
        if actual_page != Some(expected) {
            failures.push(format!(
                "期望第 {} 页，实际 {}",
                expected,
                actual_page
                    .map(|page| page.to_string())
                    .unwrap_or_else(|| "无查询帧".to_string())
            ));
        }
    }
    if let Some(expected) = expect.pages {
        let actual_pages = actual.view.map(|view| view.pages);
        if actual_pages != Some(expected) {
            failures.push(format!("期望共 {} 页，实际 {:?}", expected, actual_pages));
        }
    }
    if let Some(expected) = &expect.selected {
        match actual.selected {
            Some(selected) if selected == expected => {}
            Some(selected) => {
                failures.push(format!("期望选中「{}」，实际「{}」", expected, selected))
            }
            None => failures.push(format!("期望选中「{}」，实际没有选中行", expected)),
        }
    }
    if let Some(expected) = &expect.focus {
        match actual.focus {
            Some(focus) if focus == expected => {}
            Some(focus) => failures.push(format!("期望焦点「{}」，实际「{}」", expected, focus)),
            None => failures.push(format!("期望焦点「{}」，实际焦点为空", expected)),
        }
    }
    if let Some(expected) = expect.frames {
        if actual.frame_count != expected {
            failures.push(format!("期望 {} 帧，实际 {}", expected, actual.frame_count));
        }
    }
    if let Some(expected) = expect.stale {
        match actual.view {
            Some(view) if view.stale == expected => {}
            Some(view) => failures.push(format!("期望旧帧标记 {}，实际 {}", expected, view.stale)),
            None => failures.push("期望查询板有当前帧，实际为空".to_string()),
        }
    }
    for needle in &expect.provenance_contains {
        if !actual
            .provenance
            .iter()
            .any(|line| line.contains(needle.as_str()))
        {
            failures.push(format!("证据链缺少「{}」", needle));
        }
    }
    if let Some(expected) = expect.board_active {
        if actual.board_active != expected {
            failures.push(format!(
                "期望查询板{}，实际{}",
                if expected { "激活" } else { "未激活" },
                if actual.board_active {
                    "激活"
                } else {
                    "未激活"
                }
            ));
        }
    }
    failures
}

fn excerpt(text: &str, max: usize) -> String {
    let cleaned = text.trim().replace('\n', " / ");
    if cleaned.chars().count() <= max {
        cleaned
    } else {
        cleaned.chars().take(max).collect::<String>() + "…"
    }
}

/// 剧本内自带的确定性知识源。
#[derive(Debug, Clone)]
pub struct ScenarioKnowledge {
    entities: Vec<ScenarioEntity>,
    relations: Vec<ScenarioRelation>,
    revision: Cell<u64>,
}

impl ScenarioKnowledge {
    pub fn new(entities: Vec<ScenarioEntity>, relations: Vec<ScenarioRelation>) -> Self {
        ScenarioKnowledge {
            entities,
            relations,
            revision: Cell::new(1),
        }
    }

    pub fn from_scenario(scenario: &QueryScenario) -> Self {
        Self::new(scenario.entities.clone(), scenario.relations.clone())
    }

    pub fn set_revision(&self, revision: u64) {
        self.revision.set(revision);
    }

    fn entity(&self, name: &str) -> Option<&ScenarioEntity> {
        self.entities.iter().find(|entity| entity.name == name)
    }

    fn type_of(&self, name: &str) -> String {
        self.entity(name)
            .and_then(|entity| entity.entity_type.clone())
            .unwrap_or_else(|| "实体".to_string())
    }
}

impl DialogueKnowledge for ScenarioKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        let mut found: Vec<String> = self
            .entities
            .iter()
            .filter(|entity| text.contains(entity.name.as_str()))
            .map(|entity| entity.name.clone())
            .collect();
        found.sort_by_key(|entity| std::cmp::Reverse(entity.chars().count()));
        found.dedup();
        found
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.entity(entity).is_some()
    }

    fn answer(&self, _question: &str) -> Option<String> {
        None
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        let mut keys: Vec<String> = self
            .entity(entity)
            .map(|entity| entity.attributes.keys().cloned().collect())
            .unwrap_or_default();
        for relation in &self.relations {
            if relation.from == entity && !keys.contains(&relation.predicate) {
                keys.push(relation.predicate.clone());
            }
        }
        keys.sort();
        keys
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.entity(entity)
            .and_then(|entity| entity.attributes.get(attribute).cloned())
            .into_iter()
            .collect()
    }

    fn all_entities(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entities
            .iter()
            .map(|entity| entity.name.clone())
            .collect();
        names.sort();
        names
    }

    fn all_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .entities
            .iter()
            .map(|entity| self.type_of(&entity.name))
            .collect();
        types.sort();
        types.dedup();
        types
    }

    fn entities_of_type(&self, entity_type: &str) -> Vec<String> {
        let mut members: Vec<String> = self
            .entities
            .iter()
            .filter(|entity| self.type_of(&entity.name) == entity_type)
            .map(|entity| entity.name.clone())
            .collect();
        members.sort();
        members
    }

    fn entity_type(&self, entity: &str) -> Option<String> {
        self.entity(entity).map(|entity| self.type_of(&entity.name))
    }

    fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
        self.entity(entity)
            .map(|entity| {
                entity
                    .attributes
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
        self.relations
            .iter()
            .filter(|relation| relation.from == entity)
            .map(|relation| (relation.predicate.clone(), relation.to.clone()))
            .collect()
    }

    fn predicate_label(&self, predicate: &str) -> String {
        predicate.to_string()
    }

    fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn traceable_token(&self, token: &str) -> bool {
        self.entities.iter().any(|entity| entity.name == token)
            || self.entities.iter().any(|entity| {
                entity
                    .attributes
                    .iter()
                    .any(|(key, value)| key == token || value == token)
            })
            || self
                .relations
                .iter()
                .any(|relation| relation.predicate == token || relation.to == token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENARIO: &str = r#"
name = "身高链"
description = "列举 → 筛选 → 排序 → 选行 → 比较 → 撤销 → 知识更新标旧"

[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226", "国籍" = "中国" }

[[entities]]
name = "王治郅"
type = "Person"
attributes = { "身高" = "214", "国籍" = "中国" }

[[entities]]
name = "诺维茨基"
type = "Person"
attributes = { "身高" = "213", "国籍" = "德国" }

[[relations]]
from = "姚明"
predicate = "球队"
to = "火箭队"

[[turns]]
input = "列出所有人的身高"
expect = { op = "scan", rows = 3, board_active = true, answer_contains = ["姚明"], provenance_contains = ["身高"] }

[[turns]]
input = "只看中国的"
expect = { op = "filter", rows = 2, answer_contains = ["姚明"], answer_excludes = ["诺维茨基"] }

[[turns]]
input = "按身高从低到高排"
expect = { op = "sort", rows = 2 }

[[turns]]
input = "第二个呢"
expect = { op = "select", selected = "姚明", focus = "姚明" }

[[turns]]
input = "跟王治郅比呢"
expect = { op = "compare", answer_contains = ["226", "214", "高 12"] }

[[turns]]
input = "回到上一步"
expect = { op = "undo", selected = "姚明" }

[[turns]]
input = "依据"
expect = { op = "explain", provenance_contains = ["身高"] }

[[turns]]
set_revision = 9
input = "依据"
expect = { op = "explain", stale = true, answer_contains = ["v1 → v9"] }
"#;

    #[test]
    fn scenario_replays_query_board_state_transitions() {
        let scenario = parse_scenario(SCENARIO).expect("剧本应可解析");
        let report = run_scenario(&scenario);
        assert!(report.passed, "{}", report.render_text());
        assert_eq!(report.passed_steps, 8);
        assert_eq!(report.steps[0].op.as_deref(), Some("scan"));
        assert_eq!(
            report.steps[3]
                .board
                .as_ref()
                .and_then(|view| view.selected_row),
            Some(2)
        );
    }

    #[test]
    fn scenario_failure_is_reported_with_reason() {
        let mut scenario = parse_scenario(SCENARIO).expect("剧本应可解析");
        scenario.turns[1].expect.rows = Some(99);
        let report = run_scenario(&scenario);
        assert!(!report.passed);
        let failures = &report.steps[1].failures;
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("期望 99 行")),
            "{:?}",
            failures
        );
        assert!(report.render_text().contains("✗"));
    }

    #[test]
    fn scenario_against_external_knowledge_ignores_revision_bumps() {
        let mut scenario = parse_scenario(SCENARIO).expect("剧本应可解析");
        scenario.turns.last_mut().unwrap().expect.stale = None;
        scenario
            .turns
            .last_mut()
            .unwrap()
            .expect
            .answer_contains
            .clear();
        let knowledge = ScenarioKnowledge::from_scenario(&scenario);
        let report = run_scenario_against(&scenario, &knowledge);
        assert!(report.passed, "{}", report.render_text());
        // 外部知识源自己决定修订号：剧本里的 set_revision 不生效，旧帧不会标旧。
        assert!(!report.steps.last().unwrap().board.as_ref().unwrap().stale);
    }

    #[test]
    fn parse_rejects_scenario_without_name() {
        assert!(parse_scenario("turns = []").is_err());
    }

    #[test]
    fn bundled_example_scenario_passes() {
        let report =
            super::run_file("examples/query_board_scenario.toml").expect("示例剧本应可读取并运行");
        assert!(report.passed, "{}", report.render_text());
    }

    #[test]
    fn scenario_replays_against_real_reasoner() {
        use crate::prolog_engine::PrologFact;
        use crate::symbolic_reasoner::SymbolicReasoner;
        use std::collections::HashMap;

        let mut reasoner = SymbolicReasoner::new(crate::ReasoningConfig::default(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
            PrologFact::Entity {
                name: "王治郅".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "214".to_string())]),
            },
        ]);
        let scenario = QueryScenario {
            name: "真实推理器".to_string(),
            description: None,
            entities: Vec::new(),
            relations: Vec::new(),
            turns: vec![
                ScenarioTurn {
                    input: "列出所有人的身高".to_string(),
                    expect: TurnExpectation {
                        op: Some("scan".to_string()),
                        rows: Some(2),
                        answer_contains: vec!["226".to_string()],
                        ..TurnExpectation::default()
                    },
                    set_revision: None,
                },
                ScenarioTurn {
                    input: "第二个呢".to_string(),
                    expect: TurnExpectation {
                        op: Some("select".to_string()),
                        selected: Some("王治郅".to_string()),
                        ..TurnExpectation::default()
                    },
                    set_revision: None,
                },
            ],
        };
        let report = run_scenario_against(&scenario, &reasoner);
        assert!(report.passed, "{}", report.render_text());
    }
}

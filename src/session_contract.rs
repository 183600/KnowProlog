//! SC：会话契约与过程回执（Session Contract & Process Receipts）。
//!
//! `query_scenario` 把「查询板」写成可执行的回合轨迹；`dialogue_conformance`
//! 把对话策略写成进程内断言。两者之间还缺一条可集成的公共投影：一次多轮
//! 交互结束后，留下可版本化、可比较、可回放的过程回执。
//!
//! 本模块提供最小闭环：
//!
//! - [`TurnReceipt`]：每轮确定性的过程回执——输入、意图、回答动作、焦点前后、
//!   是否继承、计划、证据、缺口、状态摘要与内容摘要；
//! - [`SessionContract`]：自带事实与增量的 TOML 会话契约，断言结构不变量
//!   （焦点接力、计划、证据、缺口、确定性），而不是钉死整句措辞；
//! - [`run_contract`] / [`run_directory`]：离线回放器；同一契约重复运行的回执
//!   摘要逐字节一致，失败用例给出最小原因；
//! - [`discover_contracts`]：按文件名排序发现 `examples/sessions/*.toml`，
//!   新契约放入目录即进入 `cargo test` 门禁。
//!
//! 边界：回执是有界投影，不是第二份真相源；完整过程仍以编年史 / 追踪图为准。
//! 回执只在显式运行契约时生成，不改变既有对话路径的任何字节。

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::canonical_digest;
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager, DialogueReply};
use crate::query_scenario::{ScenarioAlias, ScenarioEntity, ScenarioKnowledge, ScenarioRelation};

/// 会话契约格式版本；未知版本直接拒绝，不做静默降级。
pub const SESSION_CONTRACT_VERSION: u32 = 1;
/// 默认契约目录（相对仓库根）。
pub const SESSION_CONTRACT_DIR: &str = "examples/sessions";
/// 单条回执保留的证据行数上限。
pub const RECEIPT_EVIDENCE_LIMIT: usize = 24;
/// 单条回执保留的计划 / 缺口条数上限。
pub const RECEIPT_TOKEN_LIMIT: usize = 32;
/// 回执文本字段的字符上限。
pub const RECEIPT_TEXT_LIMIT: usize = 240;

/// 一份会话契约：事实 + 增量 + 每轮结构断言。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionContract {
    #[serde(default = "default_version")]
    pub format_version: u32,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub entities: Vec<ScenarioEntity>,
    #[serde(default)]
    pub relations: Vec<ScenarioRelation>,
    #[serde(default)]
    pub aliases: Vec<ScenarioAlias>,
    pub turns: Vec<ContractTurn>,
    /// 是否要求两次独立回放的会话摘要一致（默认是）。
    #[serde(default = "default_true")]
    pub require_stable: bool,
    /// 会话级摘要锁：契约通过后可把当时的摘要写回，防止无意漂移。
    #[serde(default)]
    pub expect_digest: Option<String>,
}

fn default_version() -> u32 {
    SESSION_CONTRACT_VERSION
}

fn default_true() -> bool {
    true
}

/// 一轮输入与它的回执断言。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractTurn {
    pub input: String,
    #[serde(default)]
    pub expect: ReceiptExpectation,
    #[serde(default)]
    pub set_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_entities: Vec<ScenarioEntity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_relations: Vec<ScenarioRelation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_aliases: Vec<ScenarioAlias>,
}

/// 一轮回执期望；缺省即不断言。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReceiptExpectation {
    /// 是否应产生回答（`false` 表示应让位给其他管线）。
    #[serde(default)]
    pub answered: Option<bool>,
    /// 本体化解读得到的意图（`MoveKind::label`）。
    #[serde(default)]
    pub intent: Option<String>,
    /// 实际落地的回答动作；未回答时为 `unhandled`。
    #[serde(default)]
    pub reply_kind: Option<String>,
    /// 本轮结束后的焦点；空字符串表示期望无焦点。
    #[serde(default)]
    pub focus: Option<String>,
    /// 本轮开始前的焦点；空字符串表示期望无焦点。
    #[serde(default)]
    pub focus_before: Option<String>,
    /// 实体是否来自指代 / 省略继承。
    #[serde(default)]
    pub inherited: Option<bool>,
    /// 计划必须包含的子串（如 `direct_answer`、`board:filter`）。
    #[serde(default)]
    pub plan_contains: Vec<String>,
    /// 证据链必须包含的子串。
    #[serde(default)]
    pub evidence_contains: Vec<String>,
    /// 回答必须包含的子串。
    #[serde(default)]
    pub answer_contains: Vec<String>,
    /// 回答不得包含的子串。
    #[serde(default)]
    pub answer_excludes: Vec<String>,
    /// 缺口必须包含的稳定标签（`unhandled` / `knowledge_gap` / ...）。
    #[serde(default)]
    pub gaps_contains: Vec<String>,
    /// 回答中的引号词条是否全部可回溯到知识源（`true` = 无未接地词条）。
    #[serde(default)]
    pub grounded: Option<bool>,
    /// 回答必须引用的词条（逐字匹配）。
    #[serde(default)]
    pub quoted_contains: Vec<String>,
    /// 本轮回执摘要锁。
    #[serde(default)]
    pub digest: Option<String>,
}

/// 一轮的确定性过程回执（有界投影）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnReceipt {
    pub turn: usize,
    pub input: String,
    pub answered: bool,
    pub intent: String,
    pub reply_kind: String,
    pub focus_before: Option<String>,
    pub focus_after: Option<String>,
    pub inherited: bool,
    pub plan: Vec<String>,
    pub evidence: Vec<String>,
    pub gaps: Vec<String>,
    /// 回答中出现的引号词条（去重、有界）。
    #[serde(default)]
    pub quoted_terms: Vec<String>,
    /// 无法在知识源中回溯的引号词条（空 = 全部接地）。
    #[serde(default)]
    pub ungrounded_terms: Vec<String>,
    pub answer_excerpt: String,
    pub answer_digest: String,
    pub state_digest: String,
    pub digest: String,
}

/// 单步契约报告：回执 + 失败原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStep {
    pub receipt: TurnReceipt,
    pub passed: bool,
    pub failures: Vec<String>,
}

/// 契约回放报告。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionReport {
    pub name: String,
    pub passed: bool,
    pub passed_steps: usize,
    pub total_steps: usize,
    pub stable: bool,
    pub digest: String,
    pub failures: Vec<String>,
    pub steps: Vec<SessionStep>,
}

/// 两次回放的差异：只列变化，不重复整份回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportComparison {
    pub identical: bool,
    pub baseline_digest: String,
    pub current_digest: String,
    pub sessions: Vec<SessionDiff>,
}

impl ReportComparison {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "会话基线比较：{}（{} → {}）",
            if self.identical {
                "一致 ✓"
            } else {
                "有差异"
            },
            self.baseline_digest,
            self.current_digest
        );
        for session in &self.sessions {
            if session.changes.is_empty() && session.steps.is_empty() {
                continue;
            }
            output.push_str(&format!("\n  会话「{}」", session.name));
            for change in &session.changes {
                output.push_str(&format!("\n    - {}", change));
            }
            for step in &session.steps {
                output.push_str(&format!(
                    "\n    - 第 {} 轮「{}」：{}",
                    step.turn,
                    step.input,
                    step.changes.join("；")
                ));
            }
        }
        output
    }
}

/// 单个会话的差异。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDiff {
    pub name: String,
    pub changes: Vec<String>,
    pub steps: Vec<StepDiff>,
}

/// 单轮差异：轮次、输入与变化列表。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDiff {
    pub turn: usize,
    pub input: String,
    pub changes: Vec<String>,
}

impl SessionReport {
    /// 面向终端的中文报告：每步一行，失败给出最小原因。
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "会话「{}」：{}/{} 步通过；摘要 {}；{}",
            self.name,
            self.passed_steps,
            self.total_steps,
            self.digest,
            if self.stable {
                "确定性 ✓"
            } else {
                "确定性 ✗"
            }
        );
        if self.passed {
            output.push('。');
        } else {
            output.push_str("（有失败）。");
        }
        for failure in &self.failures {
            output.push_str(&format!("\n  ✗ {}", failure));
        }
        for (index, step) in self.steps.iter().enumerate() {
            let marker = if step.passed { '✓' } else { '✗' };
            let receipt = &step.receipt;
            output.push_str(&format!(
                "\n  {} {}. {} [{} → {}]",
                marker,
                index + 1,
                receipt.input,
                receipt.intent,
                receipt.reply_kind
            ));
            if receipt.focus_before != receipt.focus_after {
                output.push_str(&format!(
                    " 焦点 {} → {}",
                    receipt.focus_before.as_deref().unwrap_or("（空）"),
                    receipt.focus_after.as_deref().unwrap_or("（空）")
                ));
            } else if let Some(focus) = receipt.focus_after.as_deref() {
                output.push_str(&format!(" 焦点 {}", focus));
            }
            if receipt.inherited {
                output.push_str("（继承）");
            }
            if !receipt.gaps.is_empty() {
                output.push_str(&format!(" 缺口 {}", receipt.gaps.join("、")));
            }
            if !receipt.ungrounded_terms.is_empty() {
                output.push_str(&format!(" 未接地 {}", receipt.ungrounded_terms.join("、")));
            }
            for failure in &step.failures {
                output.push_str(&format!("\n       {}", failure));
            }
        }
        output
    }
}

/// 解析契约 TOML，并做版本门校验。
pub fn parse_contract(text: &str) -> Result<SessionContract, String> {
    let contract: SessionContract =
        toml::from_str(text).map_err(|error| format!("解析会话契约失败：{}", error))?;
    if contract.format_version != SESSION_CONTRACT_VERSION {
        return Err(format!(
            "会话契约版本 {} 不受支持（当前 {}）",
            contract.format_version, SESSION_CONTRACT_VERSION
        ));
    }
    if contract.name.trim().is_empty() {
        return Err("会话契约缺少 name".to_string());
    }
    if contract.turns.is_empty() {
        return Err(format!("会话契约「{}」没有回合", contract.name));
    }
    Ok(contract)
}

/// 读取并运行单个契约文件。
pub fn run_file(path: impl AsRef<Path>) -> Result<SessionReport, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("读取 {} 失败：{}", path.display(), error))?;
    let contract = parse_contract(&text)?;
    Ok(run_contract(&contract))
}

/// 运行契约：用契约自带的事实构建确定性知识源。
pub fn run_contract(contract: &SessionContract) -> SessionReport {
    finish(contract, || {
        let knowledge =
            ScenarioKnowledge::new(contract.entities.clone(), contract.relations.clone());
        if !contract.aliases.is_empty() {
            knowledge.inject(&[], &[], &contract.aliases);
            knowledge.set_revision(1);
        }
        replay(contract, &knowledge, |turn| {
            knowledge.inject(&turn.add_entities, &turn.add_relations, &turn.add_aliases);
            if let Some(revision) = turn.set_revision {
                knowledge.set_revision(revision);
            }
        })
    })
}

/// 对任意知识源回放契约（忽略回合内增量，知识版本由后端决定）。
pub fn run_contract_against(
    contract: &SessionContract,
    knowledge: &dyn DialogueKnowledge,
) -> SessionReport {
    finish(contract, || replay(contract, knowledge, |_| {}))
}

fn finish<F>(contract: &SessionContract, mut play: F) -> SessionReport
where
    F: FnMut() -> (Vec<SessionStep>, String),
{
    let (steps, digest) = play();
    let stable = if contract.require_stable {
        let (_, again) = play();
        again == digest
    } else {
        true
    };
    let passed_steps = steps.iter().filter(|step| step.passed).count();
    let mut failures = Vec::new();
    if !stable {
        failures.push("两次独立回放的会话摘要不一致".to_string());
    }
    if let Some(expected) = contract.expect_digest.as_deref() {
        if expected != digest {
            failures.push(format!(
                "会话摘要不匹配：期望 {}，实际 {}",
                expected, digest
            ));
        }
    }
    SessionReport {
        name: contract.name.clone(),
        passed: failures.is_empty() && passed_steps == steps.len(),
        passed_steps,
        total_steps: steps.len(),
        stable,
        digest,
        failures,
        steps,
    }
}

fn replay(
    contract: &SessionContract,
    knowledge: &dyn DialogueKnowledge,
    mut before_turn: impl FnMut(&ContractTurn),
) -> (Vec<SessionStep>, String) {
    let mut manager = DialogueManager::new();
    let mut steps = Vec::with_capacity(contract.turns.len());
    for (index, turn) in contract.turns.iter().enumerate() {
        before_turn(turn);
        let (receipt, answer) = capture_turn(&mut manager, knowledge, index + 1, &turn.input);
        let failures = check_expectations(&turn.expect, &receipt, &answer);
        steps.push(SessionStep {
            passed: failures.is_empty(),
            receipt,
            failures,
        });
    }
    let receipts: Vec<&TurnReceipt> = steps.iter().map(|step| &step.receipt).collect();
    let digest = canonical_digest(&(contract.name.as_str(), &receipts)).unwrap_or_default();
    (steps, digest)
}

/// 走与 CLI/HTTP 相同的对话入口，捕获一轮的有界回执。
fn capture_turn(
    manager: &mut DialogueManager,
    knowledge: &dyn DialogueKnowledge,
    turn: usize,
    input: &str,
) -> (TurnReceipt, String) {
    let focus_before = manager.focus_entity().map(str::to_string);
    let frame = manager.analyze(input, knowledge);
    let turns_before = manager.turns().len();
    let reply = manager
        .plan_pure_dialogue(input, knowledge)
        .or_else(|| manager.plan_reply(input, knowledge));
    let focus_after = manager.focus_entity().map(str::to_string);
    let appended = manager.turns().len() > turns_before;
    let plan = reply
        .as_ref()
        .map(|reply| reply.plan.clone())
        .or_else(|| {
            appended
                .then(|| manager.turns().last().map(|turn| turn.plan.clone()))
                .flatten()
        })
        .unwrap_or_default();
    let evidence = if appended {
        manager
            .turns()
            .last()
            .map(|turn| turn.provenance.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let answer = reply
        .as_ref()
        .map(|reply| reply.text.clone())
        .unwrap_or_default();
    let answer_digest = canonical_digest(&answer).unwrap_or_default();
    let quoted = quoted_terms(&answer);
    let ungrounded = quoted
        .iter()
        .filter(|term| !knowledge.traceable_token(term))
        .cloned()
        .collect();
    let state_digest = canonical_digest(&manager.snapshot()).unwrap_or_default();
    let reply_kind = reply
        .as_ref()
        .map(|reply| reply.kind.label().to_string())
        .unwrap_or_else(|| "unhandled".to_string());
    let mut receipt = TurnReceipt {
        turn,
        input: bounded(input.trim(), RECEIPT_TEXT_LIMIT),
        answered: reply.is_some(),
        intent: frame.kind.label().to_string(),
        reply_kind,
        focus_before,
        focus_after,
        inherited: frame.inherited,
        plan: bounded_tokens(plan.clone()),
        evidence: bounded_tokens(evidence),
        gaps: gaps_for(reply.as_ref(), &plan),
        quoted_terms: bounded_tokens(quoted),
        ungrounded_terms: bounded_tokens(ungrounded),
        answer_excerpt: excerpt(&answer, 80),
        answer_digest,
        state_digest,
        digest: String::new(),
    };
    receipt.digest = receipt_digest(&receipt);
    (receipt, answer)
}

fn gaps_for(reply: Option<&DialogueReply>, plan: &[String]) -> Vec<String> {
    let mut gaps = BTreeSet::new();
    if reply.is_none() {
        gaps.insert("unhandled");
    }
    for entry in plan {
        let label = entry.strip_prefix("board:").unwrap_or(entry);
        match label {
            "admit_ignorance" => {
                gaps.insert("knowledge_gap");
            }
            "why_no_evidence" => {
                gaps.insert("evidence_gap");
            }
            "clarify_entity" | "clarify_entity_no_focus" => {
                gaps.insert("referent_gap");
            }
            "continue_no_focus" => {
                gaps.insert("focus_gap");
            }
            "rank_pending" => {
                gaps.insert("attribute_gap");
            }
            _ => {}
        }
    }
    gaps.into_iter().map(str::to_string).collect()
}

fn check_expectations(
    expect: &ReceiptExpectation,
    receipt: &TurnReceipt,
    answer: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(expected) = expect.answered {
        if expected != receipt.answered {
            failures.push(format!(
                "期望{}回答，实际{}回答",
                if expected { "有" } else { "无" },
                if receipt.answered { "有" } else { "无" }
            ));
        }
    }
    if let Some(expected) = expect.intent.as_deref() {
        if expected != receipt.intent {
            failures.push(format!(
                "期望意图「{}」，实际「{}」",
                expected, receipt.intent
            ));
        }
    }
    if let Some(expected) = expect.reply_kind.as_deref() {
        if expected != receipt.reply_kind {
            failures.push(format!(
                "期望回答动作「{}」，实际「{}」",
                expected, receipt.reply_kind
            ));
        }
    }
    check_focus(
        "焦点",
        expect.focus.as_deref(),
        receipt.focus_after.as_deref(),
        &mut failures,
    );
    check_focus(
        "改动前焦点",
        expect.focus_before.as_deref(),
        receipt.focus_before.as_deref(),
        &mut failures,
    );
    if let Some(expected) = expect.inherited {
        if expected != receipt.inherited {
            failures.push(format!(
                "期望继承={}，实际继承={}",
                expected, receipt.inherited
            ));
        }
    }
    for needle in &expect.plan_contains {
        if !receipt.plan.iter().any(|plan| plan.contains(needle)) {
            failures.push(format!("计划缺少「{}」（实际 {:?}）", needle, receipt.plan));
        }
    }
    for needle in &expect.evidence_contains {
        if !receipt
            .evidence
            .iter()
            .any(|line| line.contains(needle.as_str()))
        {
            failures.push(format!("证据缺少「{}」", needle));
        }
    }
    for needle in &expect.answer_contains {
        if !answer.contains(needle.as_str()) {
            failures.push(format!(
                "回答缺少「{}」（实际 {}）",
                needle,
                excerpt(answer, 80)
            ));
        }
    }
    for needle in &expect.answer_excludes {
        if answer.contains(needle.as_str()) {
            failures.push(format!("回答不应包含「{}」", needle));
        }
    }
    for needle in &expect.gaps_contains {
        if !receipt.gaps.iter().any(|gap| gap == needle) {
            failures.push(format!("缺口缺少「{}」（实际 {:?}）", needle, receipt.gaps));
        }
    }
    if let Some(expected) = expect.grounded {
        let actual = receipt.ungrounded_terms.is_empty();
        if expected != actual {
            failures.push(format!(
                "期望引号词条{}，实际未接地 {:?}",
                if expected {
                    "全部接地"
                } else {
                    "存在未接地"
                },
                receipt.ungrounded_terms
            ));
        }
    }
    for needle in &expect.quoted_contains {
        if !receipt.quoted_terms.iter().any(|term| term == needle) {
            failures.push(format!(
                "回答未引用「{}」（引号词条 {:?}）",
                needle, receipt.quoted_terms
            ));
        }
    }
    if let Some(expected) = expect.digest.as_deref() {
        if expected != receipt.digest {
            failures.push(format!(
                "回执摘要不匹配：期望 {}，实际 {}",
                expected, receipt.digest
            ));
        }
    }
    failures
}

fn check_focus(
    label: &str,
    expected: Option<&str>,
    actual: Option<&str>,
    failures: &mut Vec<String>,
) {
    let Some(expected) = expected else {
        return;
    };
    let expected = (!expected.is_empty()).then_some(expected);
    if expected != actual {
        failures.push(format!(
            "期望{}「{}」，实际「{}」",
            label,
            expected.unwrap_or("（空）"),
            actual.unwrap_or("（空）")
        ));
    }
}

/// 发现目录下的契约文件，按路径排序，保证 `cargo test` 与 CLI 顺序一致。
pub fn discover_contracts(dir: impl AsRef<Path>) -> Result<Vec<SessionContract>, String> {
    let dir = dir.as_ref();
    if !dir.is_dir() {
        return Err(format!("契约目录不存在：{}", dir.display()));
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|error| format!("读取契约目录 {} 失败：{}", dir.display(), error))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .collect();
    paths.sort();
    let mut contracts = Vec::with_capacity(paths.len());
    for path in paths {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("读取 {} 失败：{}", path.display(), error))?;
        let contract =
            parse_contract(&text).map_err(|error| format!("{}：{}", path.display(), error))?;
        contracts.push(contract);
    }
    Ok(contracts)
}

/// 运行目录下的全部契约。
pub fn run_directory(dir: impl AsRef<Path>) -> Result<Vec<SessionReport>, String> {
    let contracts = discover_contracts(dir)?;
    Ok(contracts.iter().map(run_contract).collect())
}

/// 比较两份报告集：按会话名对齐，逐轮列出结构差异。
pub fn compare_reports(baseline: &[SessionReport], current: &[SessionReport]) -> ReportComparison {
    use std::collections::BTreeMap;
    let baseline_digest = reports_digest(baseline);
    let current_digest = reports_digest(current);
    let before: BTreeMap<&str, &SessionReport> = baseline
        .iter()
        .map(|report| (report.name.as_str(), report))
        .collect();
    let after: BTreeMap<&str, &SessionReport> = current
        .iter()
        .map(|report| (report.name.as_str(), report))
        .collect();
    let mut names: Vec<&str> = before.keys().chain(after.keys()).copied().collect();
    names.sort();
    names.dedup();
    let mut sessions = Vec::new();
    for name in names {
        let mut changes = Vec::new();
        let mut steps = Vec::new();
        match (before.get(name), after.get(name)) {
            (None, Some(_)) => changes.push("新增会话".to_string()),
            (Some(_), None) => changes.push("移除会话".to_string()),
            (Some(old), Some(new)) => {
                if old.passed != new.passed {
                    changes.push(format!("通过状态 {} → {}", old.passed, new.passed));
                }
                if old.stable != new.stable {
                    changes.push(format!("确定性 {} → {}", old.stable, new.stable));
                }
                if old.digest != new.digest {
                    changes.push(format!("会话摘要 {} → {}", old.digest, new.digest));
                }
                let total = old.steps.len().max(new.steps.len());
                for index in 0..total {
                    match (old.steps.get(index), new.steps.get(index)) {
                        (None, Some(new)) => steps.push(StepDiff {
                            turn: new.receipt.turn,
                            input: new.receipt.input.clone(),
                            changes: vec!["新增回合".to_string()],
                        }),
                        (Some(old), None) => steps.push(StepDiff {
                            turn: old.receipt.turn,
                            input: old.receipt.input.clone(),
                            changes: vec!["移除回合".to_string()],
                        }),
                        (Some(old), Some(new)) => {
                            let changes = compare_receipts(&old.receipt, &new.receipt);
                            if !changes.is_empty() {
                                steps.push(StepDiff {
                                    turn: new.receipt.turn,
                                    input: new.receipt.input.clone(),
                                    changes,
                                });
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
            (None, None) => {}
        }
        if !changes.is_empty() || !steps.is_empty() {
            sessions.push(SessionDiff {
                name: name.to_string(),
                changes,
                steps,
            });
        }
    }
    let identical = baseline_digest == current_digest && sessions.is_empty();
    ReportComparison {
        identical,
        baseline_digest,
        current_digest,
        sessions,
    }
}

fn compare_receipts(baseline: &TurnReceipt, current: &TurnReceipt) -> Vec<String> {
    let mut changes = Vec::new();
    if baseline.answered != current.answered {
        changes.push(format!("回答 {} → {}", baseline.answered, current.answered));
    }
    if baseline.intent != current.intent {
        changes.push(format!("意图 {} → {}", baseline.intent, current.intent));
    }
    if baseline.reply_kind != current.reply_kind {
        changes.push(format!(
            "回答动作 {} → {}",
            baseline.reply_kind, current.reply_kind
        ));
    }
    if baseline.focus_before != current.focus_before {
        changes.push(format!(
            "改动前焦点 {} → {}",
            focus_label(baseline.focus_before.as_deref()),
            focus_label(current.focus_before.as_deref())
        ));
    }
    if baseline.focus_after != current.focus_after {
        changes.push(format!(
            "焦点 {} → {}",
            focus_label(baseline.focus_after.as_deref()),
            focus_label(current.focus_after.as_deref())
        ));
    }
    if baseline.inherited != current.inherited {
        changes.push(format!(
            "继承 {} → {}",
            baseline.inherited, current.inherited
        ));
    }
    changes.extend(compare_tokens("计划", &baseline.plan, &current.plan));
    changes.extend(compare_tokens(
        "证据",
        &baseline.evidence,
        &current.evidence,
    ));
    changes.extend(compare_tokens("缺口", &baseline.gaps, &current.gaps));
    changes.extend(compare_tokens(
        "引号词条",
        &baseline.quoted_terms,
        &current.quoted_terms,
    ));
    changes.extend(compare_tokens(
        "未接地词条",
        &baseline.ungrounded_terms,
        &current.ungrounded_terms,
    ));
    if baseline.answer_digest != current.answer_digest {
        changes.push(format!(
            "回答 {} → {}",
            excerpt(&baseline.answer_excerpt, 24),
            excerpt(&current.answer_excerpt, 24)
        ));
    }
    if baseline.state_digest != current.state_digest {
        changes.push("状态摘要变化".to_string());
    }
    if baseline.digest != current.digest {
        changes.push("回执摘要变化".to_string());
    }
    changes
}

fn compare_tokens(label: &str, baseline: &[String], current: &[String]) -> Vec<String> {
    let removed: Vec<&str> = baseline
        .iter()
        .filter(|token| !current.contains(token))
        .map(String::as_str)
        .collect();
    let added: Vec<&str> = current
        .iter()
        .filter(|token| !baseline.contains(token))
        .map(String::as_str)
        .collect();
    let mut changes = Vec::new();
    if !removed.is_empty() {
        changes.push(format!("{} -{}", label, removed.join("、")));
    }
    if !added.is_empty() {
        changes.push(format!("{} +{}", label, added.join("、")));
    }
    changes
}

fn focus_label(focus: Option<&str>) -> &str {
    focus.unwrap_or("（空）")
}

fn reports_digest(reports: &[SessionReport]) -> String {
    let mut rows: Vec<(&str, &str)> = reports
        .iter()
        .map(|report| (report.name.as_str(), report.digest.as_str()))
        .collect();
    rows.sort();
    canonical_digest(&rows).unwrap_or_default()
}

/// 读取报告集：既接受 `session --json` 的数组，也接受单份报告对象。
pub fn parse_reports(text: &str) -> Result<Vec<SessionReport>, String> {
    if let Ok(reports) = serde_json::from_str::<Vec<SessionReport>>(text) {
        return Ok(reports);
    }
    serde_json::from_str::<SessionReport>(text)
        .map(|report| vec![report])
        .map_err(|error| format!("解析会话报告失败：{}", error))
}

/// 从路径读取报告集。
pub fn load_reports(path: impl AsRef<Path>) -> Result<Vec<SessionReport>, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("读取 {} 失败：{}", path.display(), error))?;
    parse_reports(&text)
}

/// 把报告集写成稳定的 JSON（升序会话名 + 两空格缩进）。
pub fn save_reports(path: impl AsRef<Path>, reports: &[SessionReport]) -> Result<(), String> {
    let path = path.as_ref();
    let mut sorted: Vec<&SessionReport> = reports.iter().collect();
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    let text = serde_json::to_string_pretty(&sorted)
        .map_err(|error| format!("序列化会话报告失败：{}", error))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("创建 {} 失败：{}", parent.display(), error))?;
        }
    }
    std::fs::write(path, text).map_err(|error| format!("写入 {} 失败：{}", path.display(), error))
}

fn receipt_digest(receipt: &TurnReceipt) -> String {
    #[derive(Serialize)]
    struct Body<'a> {
        turn: usize,
        input: &'a str,
        answered: bool,
        intent: &'a str,
        reply_kind: &'a str,
        focus_before: &'a Option<String>,
        focus_after: &'a Option<String>,
        inherited: bool,
        plan: &'a [String],
        evidence: &'a [String],
        gaps: &'a [String],
        quoted_terms: &'a [String],
        ungrounded_terms: &'a [String],
        answer_excerpt: &'a str,
        answer_digest: &'a str,
        state_digest: &'a str,
    }
    canonical_digest(&Body {
        turn: receipt.turn,
        input: &receipt.input,
        answered: receipt.answered,
        intent: &receipt.intent,
        reply_kind: &receipt.reply_kind,
        focus_before: &receipt.focus_before,
        focus_after: &receipt.focus_after,
        inherited: receipt.inherited,
        plan: &receipt.plan,
        evidence: &receipt.evidence,
        gaps: &receipt.gaps,
        quoted_terms: &receipt.quoted_terms,
        ungrounded_terms: &receipt.ungrounded_terms,
        answer_excerpt: &receipt.answer_excerpt,
        answer_digest: &receipt.answer_digest,
        state_digest: &receipt.state_digest,
    })
    .unwrap_or_default()
}

fn quoted_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for (open, close) in [('「', '」'), ('『', '』'), ('“', '”'), ('"', '"')] {
        let mut rest = text;
        while let Some(start) = rest.find(open) {
            let after = &rest[start + open.len_utf8()..];
            let Some(end) = after.find(close) else {
                break;
            };
            let term = after[..end].trim();
            if !term.is_empty() && !terms.iter().any(|known| known == term) {
                terms.push(term.to_string());
            }
            rest = &after[end + close.len_utf8()..];
        }
    }
    terms
}

fn bounded(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        text.chars().take(limit).collect::<String>() + "…"
    }
}

fn bounded_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut output = Vec::new();
    for token in tokens {
        let token = bounded(&token, RECEIPT_TEXT_LIMIT);
        if !output.contains(&token) {
            output.push(token);
        }
        if output.len() >= RECEIPT_TOKEN_LIMIT {
            break;
        }
    }
    output
}

fn excerpt(text: &str, limit: usize) -> String {
    let cleaned = text.trim().replace('\n', " / ");
    bounded(&cleaned, limit.min(RECEIPT_TEXT_LIMIT))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(source: &str) -> SessionContract {
        parse_contract(source).expect("契约应可解析")
    }

    #[test]
    fn example_contracts_are_discovered_and_pass() {
        let reports = run_directory(SESSION_CONTRACT_DIR).expect("示例契约应可发现");
        assert!(!reports.is_empty(), "示例契约目录不应为空");
        for report in &reports {
            assert!(
                report.passed,
                "契约「{}」失败：\n{}",
                report.name,
                report.render_text()
            );
            assert!(report.stable, "契约「{}」应确定性回放", report.name);
            assert!(!report.digest.is_empty());
        }
    }

    #[test]
    fn receipts_are_deterministic_across_independent_runs() {
        let source = r#"
name = "确定性"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "姚明的身高是多少"
[[turns]]
input = "他的国籍呢"
"#;
        let contract = contract(source);
        let first = run_contract(&contract);
        let second = run_contract(&contract);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.steps, second.steps);
        assert!(first.stable);
    }

    #[test]
    fn focus_inheritance_is_asserted_by_structure() {
        let source = r#"
name = "焦点接力"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226", "国籍" = "中国" }
[[turns]]
input = "姚明的身高是多少"
expect = { answered = true, focus = "姚明", focus_before = "", inherited = false, answer_contains = ["226"], plan_contains = ["direct_answer"] }
[[turns]]
input = "他的国籍呢"
expect = { answered = true, focus = "姚明", focus_before = "姚明", inherited = true, answer_contains = ["中国"], plan_contains = ["direct_answer"] }
"#;
        let report = run_contract(&contract(source));
        assert!(report.passed, "{}", report.render_text());
    }

    #[test]
    fn contract_failure_carries_minimal_reason() {
        let source = r#"
name = "故意失败"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "姚明的身高是多少"
expect = { focus = "叶莉", plan_contains = ["board:filter"] }
"#;
        let report = run_contract(&contract(source));
        assert!(!report.passed);
        let failures = &report.steps[0].failures;
        assert!(failures.iter().any(|failure| failure.contains("焦点")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("board:filter")));
        assert!(report.render_text().contains("✗"));
    }

    #[test]
    fn digest_pin_detects_drift() {
        let source = r#"
name = "摘要锁"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "姚明的身高是多少"
"#;
        let mut contract = contract(source);
        contract.expect_digest = Some("deadbeef".to_string());
        let report = run_contract(&contract);
        assert!(!report.passed);
        assert!(report
            .failures
            .iter()
            .any(|failure| failure.contains("摘要不匹配")));
    }

    #[test]
    fn unknown_version_is_rejected() {
        let source = r#"
format_version = 99
name = "未来契约"
[[turns]]
input = "你好"
"#;
        let error = parse_contract(source).expect_err("未知版本应被拒绝");
        assert!(error.contains("版本"));
    }

    #[test]
    fn discovery_is_sorted_and_rejects_missing_directory() {
        let dir = tempfile::TempDir::new().expect("临时目录");
        std::fs::write(
            dir.path().join("b.toml"),
            "name = \"乙\"\n[[turns]]\ninput = \"你好\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("a.toml"),
            "name = \"甲\"\n[[turns]]\ninput = \"你好\"\n",
        )
        .unwrap();
        let contracts = discover_contracts(dir.path()).expect("发现契约");
        assert_eq!(
            contracts
                .iter()
                .map(|contract| contract.name.as_str())
                .collect::<Vec<_>>(),
            vec!["甲", "乙"]
        );
        assert!(discover_contracts(dir.path().join("missing")).is_err());
    }

    #[test]
    fn unhandled_input_is_a_first_class_gap() {
        let source = r#"
name = "缺口"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "张三的身高是多少"
expect = { answered = false, reply_kind = "unhandled", gaps_contains = ["unhandled"] }
"#;
        let report = run_contract(&contract(source));
        assert!(report.passed, "{}", report.render_text());
    }

    #[test]
    fn mid_session_increments_replay_from_fresh_state() {
        let source = r#"
name = "增量稳定"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "列出所有人的身高"
expect = { answered = true, plan_contains = ["board:scan"], answer_excludes = ["王治郅"] }
[[turns]]
add_entities = [{ name = "王治郅", type = "Person", attributes = { "身高" = "214" } }]
input = "列出所有人的身高"
expect = { answered = true, plan_contains = ["board:scan"], answer_contains = ["王治郅"] }
"#;
        let report = run_contract(&contract(source));
        assert!(report.passed, "{}", report.render_text());
        assert!(report.stable, "增量契约必须每轮回放都从初始事实开始");
    }

    #[test]
    fn quoted_terms_are_deduplicated_and_grounded() {
        assert_eq!(
            quoted_terms("「姚明」的「身高」是 226；再说一次「姚明」。"),
            vec!["姚明", "身高"]
        );
        let source = r#"
name = "接地"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "姚明的身高是多少"
expect = { answered = true, grounded = true, quoted_contains = ["姚明", "身高"] }
"#;
        let report = run_contract(&contract(source));
        assert!(report.passed, "{}", report.render_text());
        assert!(report.steps[0].receipt.ungrounded_terms.is_empty());
    }

    #[test]
    fn baseline_comparison_lists_structural_changes() {
        let source = r#"
name = "比较"
[[entities]]
name = "姚明"
type = "Person"
attributes = { "身高" = "226" }
[[turns]]
input = "姚明的身高是多少"
"#;
        let baseline = run_contract(&contract(source));
        let identical = compare_reports(&[baseline.clone()], &[baseline.clone()]);
        assert!(identical.identical, "{}", identical.render_text());

        let mut current = baseline.clone();
        current.steps[0].receipt.focus_after = Some("叶莉".to_string());
        current.steps[0].receipt.plan = vec!["board:filter".to_string()];
        current.steps[0].receipt.digest = receipt_digest(&current.steps[0].receipt);
        current.digest = canonical_digest(
            &current
                .steps
                .iter()
                .map(|step| &step.receipt)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let diff = compare_reports(&[baseline], &[current]);
        assert!(!diff.identical);
        let text = diff.render_text();
        assert!(text.contains("焦点"));
        assert!(text.contains("计划"));
        assert!(text.contains("回执摘要变化"));
    }

    #[test]
    fn session_digest_identifies_the_contract() {
        let left = contract(
            "name = \"甲\"\n[[entities]]\nname = \"姚明\"\ntype = \"Person\"\nattributes = { \"身高\" = \"226\" }\n[[turns]]\ninput = \"姚明的身高是多少\"\n",
        );
        let right = contract(
            "name = \"乙\"\n[[entities]]\nname = \"姚明\"\ntype = \"Person\"\nattributes = { \"身高\" = \"226\" }\n[[turns]]\ninput = \"姚明的身高是多少\"\n",
        );
        assert_eq!(
            run_contract(&left).steps,
            run_contract(&right).steps,
            "同一事实与回合的行为回执应相同"
        );
        assert_ne!(
            run_contract(&left).digest,
            run_contract(&right).digest,
            "会话摘要必须包含契约身份"
        );
    }

    #[test]
    fn reports_roundtrip_through_json() {
        let reports = run_directory(SESSION_CONTRACT_DIR).expect("示例契约");
        let dir = tempfile::TempDir::new().expect("临时目录");
        let path = dir.path().join("baseline.json");
        save_reports(&path, &reports).expect("写入基线");
        let loaded = load_reports(&path).expect("读取基线");
        assert_eq!(loaded.len(), reports.len());
        let comparison = compare_reports(&reports, &loaded);
        assert!(comparison.identical, "{}", comparison.render_text());
        let single = serde_json::to_string(&reports[0]).unwrap();
        assert_eq!(parse_reports(&single).unwrap().len(), 1);
    }

    #[test]
    fn receipts_are_bounded() {
        let long = "很".repeat(RECEIPT_TEXT_LIMIT * 2);
        let receipt = TurnReceipt {
            turn: 1,
            input: bounded(&long, RECEIPT_TEXT_LIMIT),
            answered: false,
            intent: "other".to_string(),
            reply_kind: "unhandled".to_string(),
            focus_before: None,
            focus_after: None,
            inherited: false,
            plan: bounded_tokens(
                (0..RECEIPT_TOKEN_LIMIT * 2)
                    .map(|i| format!("p{}", i))
                    .collect(),
            ),
            evidence: Vec::new(),
            gaps: vec!["unhandled".to_string()],
            quoted_terms: Vec::new(),
            ungrounded_terms: Vec::new(),
            answer_excerpt: excerpt(&long, 80),
            answer_digest: canonical_digest(&long).unwrap(),
            state_digest: String::new(),
            digest: String::new(),
        };
        assert_eq!(receipt.input.chars().count(), RECEIPT_TEXT_LIMIT + 1);
        assert!(receipt.answer_excerpt.chars().count() <= 81);
        assert_eq!(receipt.plan.len(), RECEIPT_TOKEN_LIMIT);
    }
}

//! 词门（LexGate）：生词要进入系统语言，必须先得到一条**可执行定义**。
//!
//! 覆盖审计（[`crate::lexical_trace`]）只负责把缺口显式化；本模块负责门后
//! 的第一步：接受离线定义文件（JSON），逐条验证 alias / compose，并把通过
//! 验证的定义展开成规范谓词链。被接受的定义只进入静态词表（覆盖审计与
//! `read-trace --lexicon --definitions`），不写知识库、不改变回答。
//!
//! 设计约束：
//!   - 无网络、无权重：定义来自本地 JSON，运行时可完全复现；
//!   - 只接受可归约的定义：alias 的目标、compose 的每一步都必须在当前静态
//!     词表中可执行，或引用本文件中**更早**通过准入的定义；
//!   - 有界：链长 ≤ [`MAX_COMPOSE_STEPS`]，词面长度 ≤ [`MAX_SURFACE_CHARS`]；
//!   - 无环：只允许引用已准入的定义，禁止自指/后向引用/递归；
//!   - 拒绝也是一等输出：每条拒绝都带确定性理由，供人或 LLM 修正后重提。

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::lexical_trace::LexiconInventory;

pub const LEX_GATE_VERSION: u32 = 1;
pub const MAX_COMPOSE_STEPS: usize = 4;
pub const MAX_SURFACE_CHARS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexRuleKind {
    /// 词面是某个规范谓词的别名（「先生」→「配偶」）。
    Alias,
    /// 词面可归约为一条规范谓词链（「师祖」→「师傅」∘「师傅」）。
    Compose,
}

impl LexRuleKind {
    pub fn label(self) -> &'static str {
        match self {
            LexRuleKind::Alias => "别名",
            LexRuleKind::Compose => "组合",
        }
    }
}

/// 文件中的一条定义提议。字段宽松解析，验证器负责给出可操作理由。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexProposal {
    pub surface: String,
    pub kind: LexRuleKind,
    #[serde(default)]
    pub canonical: Option<String>,
    #[serde(default)]
    pub chain: Vec<String>,
    /// 词义/用法说明（可选；只作证据展示，不参与结构验证）。
    #[serde(default)]
    pub sense: Option<String>,
    /// 提议来源（可选；例如人工、离线模型、词典）。
    #[serde(default)]
    pub evidence: Option<String>,
}

/// 定义文件：既接受 `{"version":1,"entries":[...]}`，也接受裸数组。
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum LexiconFile {
    Wrapped {
        #[serde(default)]
        version: Option<u32>,
        entries: Vec<LexProposal>,
    },
    Bare(Vec<LexProposal>),
}

impl LexiconFile {
    fn entries(&self) -> &[LexProposal] {
        match self {
            LexiconFile::Wrapped { entries, .. } => entries,
            LexiconFile::Bare(entries) => entries,
        }
    }

    fn version(&self) -> u32 {
        match self {
            LexiconFile::Wrapped { version, .. } => version.unwrap_or(LEX_GATE_VERSION),
            LexiconFile::Bare(_) => LEX_GATE_VERSION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexAccepted {
    pub surface: String,
    pub kind: LexRuleKind,
    /// 已展开的规范谓词链（alias 为单步，compose 为多步）。
    pub expansion: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sense: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexRejection {
    pub surface: String,
    pub reason: String,
}

/// 一次离线准入的结果：接受与拒绝都保留，顺序稳定。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexAdmission {
    pub version: u32,
    pub accepted: Vec<LexAccepted>,
    pub rejected: Vec<LexRejection>,
}

impl LexAdmission {
    pub fn is_empty(&self) -> bool {
        self.accepted.is_empty() && self.rejected.is_empty()
    }

    /// 把已接受的定义注入静态词表；只影响覆盖审计，不写知识库。
    pub fn apply(&self, inventory: &mut LexiconInventory) {
        for entry in &self.accepted {
            inventory.insert_definition(&entry.surface, entry.expansion.clone());
        }
    }

    pub fn render_report(&self) -> String {
        let mut output = format!(
            "词门定义准入：接受 {} 条，拒绝 {} 条。\n",
            self.accepted.len(),
            self.rejected.len()
        );
        if self.accepted.is_empty() && self.rejected.is_empty() {
            output.push_str("文件没有定义条目。\n");
            return output.trim_end().to_string();
        }
        for entry in &self.accepted {
            output.push_str(&format!(
                "  ✓ 「{}」（{}）→ {}\n",
                entry.surface,
                entry.kind.label(),
                entry.expansion.join(" ∘ ")
            ));
        }
        for entry in &self.rejected {
            output.push_str(&format!("  ✗ 「{}」：{}\n", entry.surface, entry.reason));
        }
        output.trim_end().to_string()
    }
}

/// 从文件加载并验证定义；解析失败返回错误，语义拒绝进入报告。
pub fn load_file(path: impl AsRef<Path>, inventory: &LexiconInventory) -> Result<LexAdmission, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("读取定义文件失败（{}）：{}", path.display(), error))?;
    admit_json(&text, inventory)
}

pub fn admit_json(text: &str, inventory: &LexiconInventory) -> Result<LexAdmission, String> {
    let file: LexiconFile =
        serde_json::from_str(text).map_err(|error| format!("解析定义文件失败：{}", error))?;
    let version = file.version();
    if version > LEX_GATE_VERSION {
        return Err(format!(
            "定义文件版本 {} 高于当前支持的 {}",
            version, LEX_GATE_VERSION
        ));
    }
    Ok(admit_proposals(file.entries(), inventory))
}

/// 逐条验证；只有引用更早接受条目的定义才能通过（因此天然无环）。
pub fn admit_proposals(proposals: &[LexProposal], inventory: &LexiconInventory) -> LexAdmission {
    let mut accepted: Vec<LexAccepted> = Vec::new();
    let mut rejected: Vec<LexRejection> = Vec::new();
    let mut expansions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for proposal in proposals {
        let surface = proposal.surface.trim().to_string();
        if surface.is_empty() {
            rejected.push(LexRejection {
                surface: proposal.surface.clone(),
                reason: "词面为空".to_string(),
            });
            continue;
        }
        if surface.chars().count() > MAX_SURFACE_CHARS {
            rejected.push(LexRejection {
                surface,
                reason: format!("词面超过 {} 字上限", MAX_SURFACE_CHARS),
            });
            continue;
        }
        if !surface.chars().any(char::is_alphanumeric) {
            rejected.push(LexRejection {
                surface,
                reason: "词面不含可读字符".to_string(),
            });
            continue;
        }
        if !seen.insert(surface.clone()) {
            rejected.push(LexRejection {
                surface,
                reason: "词面在本文件中重复（同形多义需先拆分 sense）".to_string(),
            });
            continue;
        }

        let expansion = match proposal.kind {
            LexRuleKind::Alias => validate_alias(proposal, &surface, inventory, &expansions),
            LexRuleKind::Compose => validate_compose(proposal, &surface, inventory, &expansions),
        };
        match expansion {
            Ok(expansion) => {
                expansions.insert(surface.clone(), expansion.clone());
                accepted.push(LexAccepted {
                    surface,
                    kind: proposal.kind,
                    expansion,
                    sense: proposal.sense.clone(),
                    evidence: proposal.evidence.clone(),
                });
            }
            Err(reason) => rejected.push(LexRejection { surface, reason }),
        }
    }

    LexAdmission {
        version: LEX_GATE_VERSION,
        accepted,
        rejected,
    }
}

fn validate_alias(
    proposal: &LexProposal,
    surface: &str,
    inventory: &LexiconInventory,
    expansions: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    let canonical = proposal
        .canonical
        .as_deref()
        .map(str::trim)
        .filter(|canonical| !canonical.is_empty())
        .ok_or_else(|| "alias 缺少 canonical".to_string())?;
    if canonical == surface {
        return Err("alias 的 canonical 不能等于词面本身".to_string());
    }
    flatten(canonical, inventory, expansions)
}

fn validate_compose(
    proposal: &LexProposal,
    surface: &str,
    inventory: &LexiconInventory,
    expansions: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    if proposal.chain.is_empty() {
        return Err("compose 缺少 chain".to_string());
    }
    if proposal.chain.len() > MAX_COMPOSE_STEPS {
        return Err(format!(
            "compose 链长 {} 超过 {} 步上限",
            proposal.chain.len(),
            MAX_COMPOSE_STEPS
        ));
    }
    let mut expansion = Vec::new();
    for step in &proposal.chain {
        let step = step.trim();
        if step.is_empty() {
            return Err("compose 链含空步骤".to_string());
        }
        if step == surface {
            return Err("compose 不能直接或间接引用自身".to_string());
        }
        let mut part = flatten(step, inventory, expansions)?;
        expansion.append(&mut part);
        if expansion.len() > MAX_COMPOSE_STEPS {
            return Err(format!(
                "compose 展开后超过 {} 步上限",
                MAX_COMPOSE_STEPS
            ));
        }
    }
    if expansion.len() < 2 {
        return Err("compose 展开后至少需要两步；单步请用 alias".to_string());
    }
    Ok(expansion)
}

fn flatten(
    word: &str,
    inventory: &LexiconInventory,
    expansions: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    if let Some(expansion) = expansions.get(word) {
        return Ok(expansion.clone());
    }
    if inventory.is_predicate(word) {
        return Ok(vec![word.to_string()]);
    }
    Err(format!(
        "「{}」不在当前可执行谓词表中；先定义它，或改用已有谓词",
        word
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemConfig;

    fn inventory() -> LexiconInventory {
        let config: SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        LexiconInventory::from_config(&config)
    }

    #[test]
    fn alias_and_compose_admit_and_expand() {
        let inventory = inventory();
        let text = r#"{
            "version": 1,
            "entries": [
                {"surface": "先生", "kind": "alias", "canonical": "配偶", "sense": "丈夫"},
                {"surface": "师祖", "kind": "compose", "chain": ["师傅", "师傅"]}
            ]
        }"#;
        let admission = admit_json(text, &inventory).unwrap();
        assert_eq!(admission.accepted.len(), 2, "{:?}", admission.rejected);
        assert_eq!(admission.accepted[0].expansion, vec!["配偶".to_string()]);
        assert_eq!(
            admission.accepted[1].expansion,
            vec!["师傅".to_string(), "师傅".to_string()]
        );
        let mut inventory = inventory;
        admission.apply(&mut inventory);
        assert!(inventory.is_predicate("先生"));
        assert_eq!(
            inventory.definition_expansion("师祖"),
            Some(["师傅".to_string(), "师傅".to_string()].as_slice())
        );
    }

    #[test]
    fn unknown_targets_are_rejected_with_actionable_reason() {
        let inventory = inventory();
        let text = r#"[
            {"surface": "曾祖父", "kind": "compose", "chain": ["父亲", "父亲", "父亲"]},
            {"surface": "先生", "kind": "alias", "canonical": "郎君"},
            {"surface": "", "kind": "alias", "canonical": "配偶"}
        ]"#;
        let admission = admit_json(text, &inventory).unwrap();
        assert!(admission.accepted.is_empty());
        assert_eq!(admission.rejected.len(), 3);
        assert!(admission.rejected[0].reason.contains("父亲"));
        assert!(admission.rejected[1].reason.contains("郎君"));
        assert!(admission.rejected[2].reason.contains("词面为空"));
    }

    #[test]
    fn duplicate_and_self_reference_are_rejected() {
        let inventory = inventory();
        let text = r#"[
            {"surface": "先生", "kind": "alias", "canonical": "配偶"},
            {"surface": "先生", "kind": "alias", "canonical": "配偶"},
            {"surface": "师祖", "kind": "compose", "chain": ["师祖"]},
            {"surface": "高祖父", "kind": "compose", "chain": ["师傅"]}
        ]"#;
        let admission = admit_json(text, &inventory).unwrap();
        assert_eq!(admission.accepted.len(), 1);
        assert_eq!(admission.rejected.len(), 3);
        assert!(admission.rejected[0].reason.contains("重复"));
        assert!(admission.rejected[1].reason.contains("自身"));
        assert!(admission.rejected[2].reason.contains("至少需要两步"));
    }

    #[test]
    fn forward_reference_is_rejected_but_backward_reference_admits() {
        let inventory = inventory();
        let forward = r#"[
            {"surface": "太祖", "kind": "compose", "chain": ["师祖", "师傅"]}
        ]"#;
        let admission = admit_json(forward, &inventory).unwrap();
        assert!(admission.accepted.is_empty());
        assert!(admission.rejected[0].reason.contains("师祖"));

        let backward = r#"[
            {"surface": "师祖", "kind": "compose", "chain": ["师傅", "师傅"]},
            {"surface": "太祖", "kind": "compose", "chain": ["师祖", "师傅"]}
        ]"#;
        let admission = admit_json(backward, &inventory).unwrap();
        assert_eq!(admission.accepted.len(), 2);
        assert_eq!(
            admission.accepted[1].expansion,
            vec!["师傅".to_string(), "师傅".to_string(), "师傅".to_string()]
        );
    }

    #[test]
    fn file_version_gate_and_bare_array_parse() {
        let inventory = inventory();
        let bare = r#"[{"surface": "先生", "kind": "alias", "canonical": "配偶"}]"#;
        assert_eq!(admit_json(bare, &inventory).unwrap().accepted.len(), 1);
        let future = r#"{"version": 99, "entries": []}"#;
        assert!(admit_json(future, &inventory)
            .unwrap_err()
            .contains("高于当前支持"));
        assert_eq!(
            LEX_GATE_VERSION,
            admit_json(bare, &inventory).unwrap().version
        );
    }
}

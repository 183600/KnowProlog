//! 提示资产（PromptSpec）：把「模板措辞」物化为可编译、可测试、可归因、
//! 可版本的符号资产，而不是散落在 Rust 源码里的 `match` 分支。
//!
//! 设计要点（纯符号主义，无神经网络）：
//! - **模板 = 带类型槽位的渲染程序**：`{slot}` 占位符在 lint 期即校验声明，
//!   渲染期每个占位符要么取到绑定值，要么触发接地闸门；
//! - **示例即测试**：每个 spec 自带 `example` 绑定与期望文本，lint 会实际
//!   渲染并比对——示例不是装饰，是可执行的符合性测试；
//! - **接地闸门（grounding gate）**：spec 声明的 `required` 槽位必须全部
//!   拿到绑定才允许渲染；任一未绑定即返回 [`RenderOutcome::Ungrounded`]，
//!   调用方沿 fallback 图回退（如实回退到组合生成器 / 澄清追问），
//!   从结构上排除「模板里编造内容」；
//! - **版本化**：注册表带 [`PROMPT_SPEC_VERSION`]，逐 spec 指标可归因、
//!   可对比。
//!
//! 与 [`crate::dialogue_manager`] 的关系：管理器负责「选策略 + 选 spec 变体 +
//! 绑定槽位」，本模块只负责「校验 + 渲染」。三档模式
//! （[`PromptMode::Off`] / [`PromptMode::Parity`] / [`PromptMode::On`]）由管理器生效。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bindings::{BindingSet, BoundTerm};

/// 提示资产注册表的语义版本：模板结构 / 槽位契约变化时递增。
pub const PROMPT_SPEC_VERSION: u32 = 1;

/// 对话策略名全表（与 `dialogue_manager::POLICY_SOURCE` 推出的 `respond/1` 对应）。
pub const POLICY_NAMES: [&str; 13] = [
    "recap",
    "explain_why",
    "why_no_evidence",
    "continue_topic",
    "continue_no_focus",
    "elaborate",
    "clarify_entity",
    "clarify_entity_no_focus",
    "direct_answer",
    "admit_ignorance",
    "greet",
    "thanks",
    "farewell",
];

/// 一条模板资产。
#[derive(Debug, Clone, PartialEq)]
pub struct PromptSpec {
    /// 全注册表唯一 id，如 `direct_answer/more/0`。
    pub id: &'static str,
    /// 该模板对应哪条对话策略（[`POLICY_NAMES`] 之一）。
    pub policy: &'static str,
    /// 模板正文：字面量与 `{slot}` 占位符交替。
    pub template: &'static str,
    /// 接地闸门：必须全部拿到绑定才允许渲染的槽位。
    pub required: &'static [&'static str],
    /// 可缺省槽位：缺省时渲染为空串。
    pub optional: &'static [&'static str],
    /// 示例绑定（few-shot 的符号等价物：可执行符合性测试）。
    pub example: &'static [(&'static str, &'static str)],
    /// 示例绑定下应逐字渲染出的文本。
    pub example_rendered: &'static str,
    /// 人读说明（出现在 `prompt-lint` 报告与 `/v1/prompts` 中）。
    pub description: &'static str,
}

/// 模板片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Literal(String),
    Slot(String),
}

impl PromptSpec {
    /// 该 spec 的全部声明槽位（required + optional）。
    pub fn declared_slots(&self) -> Vec<&'static str> {
        let mut slots: Vec<&'static str> = self.required.to_vec();
        for slot in self.optional {
            if !slots.contains(slot) {
                slots.push(slot);
            }
        }
        slots
    }
}

/// 渲染结果：成功 / 接地失败 / 资产无效（lint 应拦住的错误）。
#[derive(Debug, Clone, PartialEq)]
pub enum RenderOutcome {
    Rendered {
        spec_id: &'static str,
        text: String,
    },
    Ungrounded {
        spec_id: &'static str,
        missing: Vec<String>,
    },
    Invalid {
        spec_id: &'static str,
        message: String,
    },
}

impl RenderOutcome {
    pub fn text(&self) -> Option<&str> {
        match self {
            RenderOutcome::Rendered { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// 解析模板为片段序列；`{{`/`}}` 为字面花括号转义。
pub fn parse_template(template: &str) -> Result<Vec<Part>, String> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    literal.push('{');
                    continue;
                }
                if !literal.is_empty() {
                    parts.push(Part::Literal(std::mem::take(&mut literal)));
                }
                let mut name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    if inner == '{' {
                        return Err(format!("槽位「{}」内出现未转义的左花括号", name));
                    }
                    name.push(inner);
                }
                if !closed {
                    return Err(format!("槽位「{}」缺少右花括号", name));
                }
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err("出现空槽位「{}」".to_string());
                }
                if !name.chars().all(|ch| ch.is_alphanumeric() || ch == '_') {
                    return Err(format!(
                        "槽位名「{}」含非法字符（允许字母/数字/下划线）",
                        name
                    ));
                }
                parts.push(Part::Slot(name));
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    literal.push('}');
                    continue;
                }
                return Err("出现未配对的右花括号".to_string());
            }
            _ => literal.push(ch),
        }
    }
    if !literal.is_empty() {
        parts.push(Part::Literal(literal));
    }
    Ok(parts)
}

/// 接地闸门渲染：required 槽位全绑定 → 渲染；否则 [`RenderOutcome::Ungrounded`]。
pub fn render_spec(spec: &PromptSpec, bindings: &BindingSet) -> RenderOutcome {
    let parts = match parse_template(spec.template) {
        Ok(parts) => parts,
        Err(message) => {
            return RenderOutcome::Invalid {
                spec_id: spec.id,
                message,
            }
        }
    };
    let missing = bindings.missing(spec.required);
    if !missing.is_empty() {
        return RenderOutcome::Ungrounded {
            spec_id: spec.id,
            missing,
        };
    }
    let mut text = String::with_capacity(spec.template.len() + 16);
    for part in &parts {
        match part {
            Part::Literal(literal) => text.push_str(literal),
            Part::Slot(name) => match bindings.get(name) {
                Some(BoundTerm::Unbound) | None => {}
                Some(term) => {
                    if let Some(raw) = term.render_raw() {
                        text.push_str(&raw);
                    }
                }
            },
        }
    }
    RenderOutcome::Rendered {
        spec_id: spec.id,
        text,
    }
}

/// 默认注册表：覆盖 10/13 条策略的 20 个模板变体。
///
/// 未外化的策略（`recap` / `continue_topic` / `elaborate`）保留组合生成器，
/// 由 [`coverage`] 显式报告——「有没有外化」本身是可观测状态。
pub static DEFAULT_SPECS: &[PromptSpec] = &[
    PromptSpec {
        id: "direct_answer/plain",
        policy: "direct_answer",
        template: "{answer}",
        required: &["answer"],
        optional: &[],
        example: &[("answer", "姚明的身高是 226 厘米。")],
        example_rendered: "姚明的身高是 226 厘米。",
        description: "直答且没有可推荐的相关属性时，原样呈现结论。",
    },
    PromptSpec {
        id: "direct_answer/more/0",
        policy: "direct_answer",
        template: "{answer}\n我还知道它的「{listed}」等 {count} 项信息，想了解可以直接问。",
        required: &["answer", "listed", "count"],
        optional: &[],
        example: &[
            ("answer", "姚明的身高是 226 厘米。"),
            ("listed", "国籍」「出生地"),
            ("count", "2"),
        ],
        example_rendered: "姚明的身高是 226 厘米。\n我还知道它的「国籍」「出生地」等 2 项信息，想了解可以直接问。",
        description: "直答 + 追问建议（变体 0，按轮次轮换）。",
    },
    PromptSpec {
        id: "direct_answer/more/1",
        policy: "direct_answer",
        template: "{answer}\n关于「{entity}」，我还能补充「{listed}」等 {count} 项。",
        required: &["answer", "entity", "listed", "count"],
        optional: &[],
        example: &[
            ("answer", "姚明的身高是 226 厘米。"),
            ("entity", "姚明"),
            ("listed", "国籍」「出生地"),
            ("count", "2"),
        ],
        example_rendered: "姚明的身高是 226 厘米。\n关于「姚明」，我还能补充「国籍」「出生地」等 2 项。",
        description: "直答 + 追问建议（变体 1）。",
    },
    PromptSpec {
        id: "direct_answer/more/2",
        policy: "direct_answer",
        template: "{answer}\n除此之外，「{entity}」的「{listed}」我也知道（共 {count} 项），继续问我就好。",
        required: &["answer", "entity", "listed", "count"],
        optional: &[],
        example: &[
            ("answer", "姚明的身高是 226 厘米。"),
            ("entity", "姚明"),
            ("listed", "国籍」「出生地"),
            ("count", "2"),
        ],
        example_rendered: "姚明的身高是 226 厘米。\n除此之外，「姚明」的「国籍」「出生地」我也知道（共 2 项），继续问我就好。",
        description: "直答 + 追问建议（变体 2）。",
    },
    PromptSpec {
        id: "admit_ignorance/attribute/plain",
        policy: "admit_ignorance",
        template: "关于「{entity}」的「{attribute}」，我的知识图谱里暂时没有记录。\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。",
        required: &["entity", "attribute"],
        optional: &[],
        example: &[("entity", "姚明"), ("attribute", "体重")],
        example_rendered: "关于「姚明」的「体重」，我的知识图谱里暂时没有记录。\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。",
        description: "属性未知坦白（无相关属性可推荐）。",
    },
    PromptSpec {
        id: "admit_ignorance/attribute/more",
        policy: "admit_ignorance",
        template: "关于「{entity}」的「{attribute}」，我的知识图谱里暂时没有记录。\n不过我知道它的「{listed}」等 {count} 项信息，你想了解哪一个？",
        required: &["entity", "attribute", "listed", "count"],
        optional: &[],
        example: &[
            ("entity", "姚明"),
            ("attribute", "体重"),
            ("listed", "身高」「国籍"),
            ("count", "2"),
        ],
        example_rendered: "关于「姚明」的「体重」，我的知识图谱里暂时没有记录。\n不过我知道它的「身高」「国籍」等 2 项信息，你想了解哪一个？",
        description: "属性未知坦白 + 可选项推荐。",
    },
    PromptSpec {
        id: "admit_ignorance/identity/plain",
        policy: "admit_ignorance",
        template: "我的知识图谱里暂时没有关于「{entity}」的介绍信息。\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。",
        required: &["entity"],
        optional: &[],
        example: &[("entity", "诸葛亮")],
        example_rendered: "我的知识图谱里暂时没有关于「诸葛亮」的介绍信息。\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。",
        description: "身份未知坦白（无相关属性可推荐）。",
    },
    PromptSpec {
        id: "admit_ignorance/identity/more",
        policy: "admit_ignorance",
        template: "我的知识图谱里暂时没有关于「{entity}」的介绍信息。\n不过我知道它的「{listed}」等 {count} 项信息，你想了解哪一个？",
        required: &["entity", "listed", "count"],
        optional: &[],
        example: &[("entity", "诸葛亮"), ("listed", "身高」「国籍"), ("count", "2")],
        example_rendered: "我的知识图谱里暂时没有关于「诸葛亮」的介绍信息。\n不过我知道它的「身高」「国籍」等 2 项信息，你想了解哪一个？",
        description: "身份未知坦白 + 可选项推荐。",
    },
    PromptSpec {
        id: "clarify_entity/multi",
        policy: "clarify_entity",
        template: "你想问的「{attribute}」是指「{candidates}」还是「{last}」？告诉我名字我就接着答。",
        required: &["attribute", "candidates", "last"],
        optional: &[],
        example: &[("attribute", "身高"), ("candidates", "叶莉"), ("last", "姚明")],
        example_rendered: "你想问的「身高」是指「叶莉」还是「姚明」？告诉我名字我就接着答。",
        description: "多焦点消歧：列出焦点栈中的候选实体。",
    },
    PromptSpec {
        id: "clarify_entity/single",
        policy: "clarify_entity",
        template: "你想问谁的「{attribute}」？先告诉我实体名吧。",
        required: &["attribute"],
        optional: &[],
        example: &[("attribute", "身高")],
        example_rendered: "你想问谁的「身高」？先告诉我实体名吧。",
        description: "消歧但焦点不足时的退化问法。",
    },
    PromptSpec {
        id: "clarify_entity_no_focus/plain",
        policy: "clarify_entity_no_focus",
        template: "你想问谁的「{attribute}」？先告诉我实体名吧。",
        required: &["attribute"],
        optional: &[],
        example: &[("attribute", "首都")],
        example_rendered: "你想问谁的「首都」？先告诉我实体名吧。",
        description: "只有属性没有实体时主动要实体名。",
    },
    PromptSpec {
        id: "greet/first",
        policy: "greet",
        template: "你好！我是 KnowProlog 的符号主义对话层：知识来自图谱，推理依靠规则与逻辑程序，回答由模板组合生成。你可以直接问「姚明的身高是多少」这样的问题。",
        required: &[],
        optional: &[],
        example: &[],
        example_rendered: "你好！我是 KnowProlog 的符号主义对话层：知识来自图谱，推理依靠规则与逻辑程序，回答由模板组合生成。你可以直接问「姚明的身高是多少」这样的问题。",
        description: "首次寒暄：自述系统能力边界。",
    },
    PromptSpec {
        id: "greet/returning_focus",
        policy: "greet",
        template: "又见面了！我们接着聊「{focus}」也行，换个话题也行。",
        required: &["focus"],
        optional: &[],
        example: &[("focus", "姚明")],
        example_rendered: "又见面了！我们接着聊「姚明」也行，换个话题也行。",
        description: "再次寒暄：带着当前焦点。",
    },
    PromptSpec {
        id: "greet/returning_plain",
        policy: "greet",
        template: "又见面了！继续问我吧。",
        required: &[],
        optional: &[],
        example: &[],
        example_rendered: "又见面了！继续问我吧。",
        description: "再次寒暄：无焦点。",
    },
    PromptSpec {
        id: "thanks/focus",
        policy: "thanks",
        template: "不客气！关于「{focus}」还有想问的，随时说。",
        required: &["focus"],
        optional: &[],
        example: &[("focus", "姚明")],
        example_rendered: "不客气！关于「姚明」还有想问的，随时说。",
        description: "致谢回应：带着当前焦点。",
    },
    PromptSpec {
        id: "thanks/plain",
        policy: "thanks",
        template: "不客气！随时继续问我。",
        required: &[],
        optional: &[],
        example: &[],
        example_rendered: "不客气！随时继续问我。",
        description: "致谢回应：无焦点。",
    },
    PromptSpec {
        id: "farewell/plain",
        policy: "farewell",
        template: "好的，再见！这次我们聊了 {turn_count} 轮，对话本体已经记入记忆。",
        required: &["turn_count"],
        optional: &[],
        example: &[("turn_count", "3")],
        example_rendered: "好的，再见！这次我们聊了 3 轮，对话本体已经记入记忆。",
        description: "告别：汇报已记录的对话轮数。",
    },
    PromptSpec {
        id: "why_no_evidence/plain",
        policy: "why_no_evidence",
        template: "我还没有答过什么问题，暂时没有可解释的依据。你可以先问我一个知识问题，再问我「你怎么知道的」。",
        required: &[],
        optional: &[],
        example: &[],
        example_rendered: "我还没有答过什么问题，暂时没有可解释的依据。你可以先问我一个知识问题，再问我「你怎么知道的」。",
        description: "追问依据但尚无回答可解释时的坦白。",
    },
    PromptSpec {
        id: "continue_no_focus/plain",
        policy: "continue_no_focus",
        template: "我还不知道你想接着聊哪个实体。先给我一个名字吧，比如「姚明」。",
        required: &[],
        optional: &[],
        example: &[],
        example_rendered: "我还不知道你想接着聊哪个实体。先给我一个名字吧，比如「姚明」。",
        description: "续聊但没有焦点时主动要实体名。",
    },
    PromptSpec {
        id: "explain_why/plain",
        policy: "explain_why",
        template: "刚才关于{topic}的回答，依据是：{provenance}。这些结论都来自知识图谱与规则程序（逐条符号匹配、逻辑推导），没有任何神经网络参与。",
        required: &["topic", "provenance"],
        optional: &[],
        example: &[("topic", "「姚明」的「身高」"), ("provenance", "知识图谱中的直接事实")],
        example_rendered: "刚才关于「姚明」的「身高」的回答，依据是：知识图谱中的直接事实。这些结论都来自知识图谱与规则程序（逐条符号匹配、逻辑推导），没有任何神经网络参与。",
        description: "元话语：宣讲上一轮回答的推理出处。",
    },
];

/// 默认注册表。
pub fn default_specs() -> &'static [PromptSpec] {
    DEFAULT_SPECS
}

/// 按 id 查找默认注册表。
pub fn find_spec(id: &str) -> Option<&'static PromptSpec> {
    DEFAULT_SPECS.iter().find(|spec| spec.id == id)
}

/// 按 id 渲染（接地闸门生效）。
pub fn render(id: &str, bindings: &BindingSet) -> RenderOutcome {
    match find_spec(id) {
        Some(spec) => render_spec(spec, bindings),
        None => RenderOutcome::Invalid {
            spec_id: "",
            message: format!("未知提示资产 id：{}", id),
        },
    }
}

/// 某策略下的全部 spec id（默认注册表顺序）。
pub fn spec_ids_for_policy(policy: &str) -> Vec<&'static str> {
    DEFAULT_SPECS
        .iter()
        .filter(|spec| spec.policy == policy)
        .map(|spec| spec.id)
        .collect()
}

// ---------------------------------------------------------------------------
// lint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptLintIssue {
    pub severity: LintSeverity,
    pub spec_id: String,
    pub message: String,
}

/// 校验注册表：结构错误为 `Error`，可疑约定为 `Warning`。
pub fn lint_specs(specs: &[PromptSpec]) -> Vec<PromptLintIssue> {
    let mut issues = Vec::new();
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    for spec in specs {
        let mut push = |severity: LintSeverity, message: String| {
            issues.push(PromptLintIssue {
                severity,
                spec_id: spec.id.to_string(),
                message,
            });
        };
        if spec.id.trim().is_empty() {
            push(LintSeverity::Error, "id 为空".to_string());
        } else if seen.insert(spec.id, ()).is_some() {
            push(LintSeverity::Error, "id 重复".to_string());
        }
        if !POLICY_NAMES.contains(&spec.policy) {
            push(LintSeverity::Error, format!("未知策略「{}」", spec.policy));
        }
        let parts = match parse_template(spec.template) {
            Ok(parts) => parts,
            Err(message) => {
                push(LintSeverity::Error, format!("模板解析失败：{}", message));
                continue;
            }
        };
        let mut placeholders: Vec<&str> = Vec::new();
        for part in &parts {
            if let Part::Slot(name) = part {
                if !placeholders.contains(&name.as_str()) {
                    placeholders.push(name.as_str());
                }
            }
        }
        let declared = spec.declared_slots();
        for placeholder in &placeholders {
            if !declared.contains(placeholder) {
                push(
                    LintSeverity::Error,
                    format!("占位符「{}」未在 required/optional 中声明", placeholder),
                );
            }
        }
        for required in spec.required {
            if !placeholders.contains(required) {
                push(
                    LintSeverity::Error,
                    format!("required 槽位「{}」未出现在模板中", required),
                );
            }
        }
        for required in spec.required {
            if spec.optional.contains(required) {
                push(
                    LintSeverity::Warning,
                    format!("槽位「{}」同时声明为 required 与 optional", required),
                );
            }
        }
        if spec.template.is_empty() {
            push(LintSeverity::Error, "模板为空".to_string());
        }
        if spec.description.trim().is_empty() {
            push(LintSeverity::Warning, "缺少说明文字".to_string());
        }
        // 示例即测试：实际渲染必须与期望文本逐字一致
        if spec.example_rendered.is_empty() {
            push(
                LintSeverity::Error,
                "缺少 example_rendered（示例不可执行）".to_string(),
            );
        } else {
            let mut bindings = BindingSet::new();
            for (name, value) in spec.example {
                bindings.bind((*name).to_string(), value);
            }
            match render_spec(spec, &bindings) {
                RenderOutcome::Rendered { text, .. } => {
                    if text != spec.example_rendered {
                        push(
                            LintSeverity::Error,
                            format!(
                                "示例渲染不一致：\n    期望: {}\n    实际: {}",
                                spec.example_rendered.replace('\n', "⏎"),
                                text.replace('\n', "⏎")
                            ),
                        );
                    }
                }
                RenderOutcome::Ungrounded { missing, .. } => {
                    push(
                        LintSeverity::Error,
                        format!("示例缺少 required 槽位绑定：{:?}", missing),
                    );
                }
                RenderOutcome::Invalid { message, .. } => {
                    push(LintSeverity::Error, format!("示例渲染无效：{}", message));
                }
            }
        }
    }
    issues
}

/// 默认注册表的 lint 结果。
pub fn lint_default() -> Vec<PromptLintIssue> {
    lint_specs(DEFAULT_SPECS)
}

/// 策略覆盖情况：(已外化, 未外化)。
pub fn coverage() -> (Vec<&'static str>, Vec<&'static str>) {
    let mut covered = Vec::new();
    let mut composed = Vec::new();
    for policy in POLICY_NAMES {
        if DEFAULT_SPECS.iter().any(|spec| spec.policy == policy) {
            covered.push(policy);
        } else {
            composed.push(policy);
        }
    }
    (covered, composed)
}

// ---------------------------------------------------------------------------
// 运行模式与统计
// ---------------------------------------------------------------------------

/// 模板运行三档：关 / 双跑 / 启用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptMode {
    /// 只走组合生成器。
    Off,
    /// 双跑：文本照常返回，同时用 spec 渲染并计数差异。
    Parity,
    /// 启用 spec 渲染；接地失败时回退组合生成器（结构保证不幻觉）。
    On,
}

impl PromptMode {
    pub fn label(&self) -> &'static str {
        match self {
            PromptMode::Off => "off",
            PromptMode::Parity => "parity",
            PromptMode::On => "on",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(PromptMode::Off),
            "parity" => Some(PromptMode::Parity),
            "on" => Some(PromptMode::On),
            _ => None,
        }
    }
}

impl Default for PromptMode {
    fn default() -> Self {
        PromptMode::Off
    }
}

/// 模板运行统计（确定性计数，供棘轮 / 报告 / 测试消费）。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptStats {
    /// 尝试走 spec 渲染的策略次数。
    pub checked: usize,
    /// required 槽位全部绑定，成功渲染。
    pub grounded: usize,
    /// required 槽位缺失，回退组合生成器。
    pub ungrounded: usize,
    /// 无 spec 映射、保留组合生成器的策略次数。
    pub composed: usize,
    /// 双跑模式下 spec 渲染与组合生成器文本不一致的次数。
    pub parity_mismatches: usize,
    /// 缺失槽位计数（槽位名 → 次数）。
    pub missing_slots: BTreeMap<String, usize>,
    /// 差异样例（最多 3 条，供诊断）。
    pub mismatch_examples: Vec<String>,
}

impl PromptStats {
    pub fn attempted(&self) -> usize {
        self.grounded + self.ungrounded
    }

    /// 接地率：grounded / attempted；没有尝试过时视为 1.0（无样本不构成错误）。
    pub fn grounding_rate(&self) -> f64 {
        if self.attempted() == 0 {
            1.0
        } else {
            self.grounded as f64 / self.attempted() as f64
        }
    }

    pub fn record_missing(&mut self, slots: &[String]) {
        for slot in slots {
            *self.missing_slots.entry(slot.clone()).or_insert(0) += 1;
        }
    }

    pub fn record_mismatch(&mut self, example: String) {
        if self.mismatch_examples.len() < 3 {
            self.mismatch_examples.push(example);
        }
    }

    /// 合并另一个统计（评测期逐场景累计）。
    pub fn absorb(&mut self, other: &PromptStats) {
        self.checked += other.checked;
        self.grounded += other.grounded;
        self.ungrounded += other.ungrounded;
        self.composed += other.composed;
        self.parity_mismatches += other.parity_mismatches;
        for (slot, count) in &other.missing_slots {
            *self.missing_slots.entry(slot.clone()).or_insert(0) += count;
        }
        for example in &other.mismatch_examples {
            self.record_mismatch(example.clone());
        }
    }

    pub fn render_text(&self) -> String {
        format!(
            "提示模板：尝试 {} 次（接地 {} / 未接地 {}），组合生成器 {} 次，双跑差异 {} 次，接地率 {:.4}。",
            self.attempted(),
            self.grounded,
            self.ungrounded,
            self.composed,
            self.parity_mismatches,
            self.grounding_rate()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::BoundTerm;

    #[test]
    fn test_parse_template_escapes_and_errors() {
        assert_eq!(
            parse_template("a{s}b").unwrap(),
            vec![
                Part::Literal("a".into()),
                Part::Slot("s".into()),
                Part::Literal("b".into())
            ]
        );
        assert_eq!(
            parse_template("{{literal}}").unwrap(),
            vec![Part::Literal("{literal}".into())]
        );
        assert!(parse_template("{unclosed").is_err());
        assert!(parse_template("stray}").is_err());
        assert!(parse_template("{}").is_err());
        assert!(parse_template("{bad name}").is_err());
    }

    #[test]
    fn test_default_registry_passes_lint() {
        let issues = lint_default();
        let errors: Vec<&PromptLintIssue> = issues
            .iter()
            .filter(|issue| issue.severity == LintSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "默认注册表不应有 lint 错误：{:#?}",
            errors
        );
    }

    #[test]
    fn test_examples_are_executable_tests() {
        for spec in default_specs() {
            let mut bindings = BindingSet::new();
            for (name, value) in spec.example {
                bindings.bind((*name).to_string(), value);
            }
            match render_spec(spec, &bindings) {
                RenderOutcome::Rendered { text, .. } => {
                    assert_eq!(
                        text, spec.example_rendered,
                        "spec {} 示例渲染不一致",
                        spec.id
                    );
                }
                other => panic!("spec {} 示例应可渲染：{:?}", spec.id, other),
            }
        }
    }

    #[test]
    fn test_grounding_gate_blocks_unbound_required_slot() {
        let mut bindings = BindingSet::new();
        bindings.bind("answer", "姚明的身高是 226 厘米。");
        bindings.bind("listed", "国籍");
        // 缺 count → 接地失败
        let outcome = render("direct_answer/more/0", &bindings);
        match outcome {
            RenderOutcome::Ungrounded { missing, .. } => {
                assert_eq!(missing, vec!["count".to_string()])
            }
            other => panic!("应触发接地闸门：{:?}", other),
        }

        // 显式 Unbound 同样视为未接地
        bindings.insert("count", BoundTerm::Unbound);
        match render("direct_answer/more/0", &bindings) {
            RenderOutcome::Ungrounded { missing, .. } => {
                assert_eq!(missing, vec!["count".to_string()])
            }
            other => panic!("显式 Unbound 应触发接地闸门：{:?}", other),
        }

        // 全绑定 → 渲染，数值槽位渲染 raw
        bindings.bind("count", "2");
        let text = render("direct_answer/more/0", &bindings)
            .text()
            .expect("应可渲染")
            .to_string();
        assert!(text.contains("等 2 项信息"), "{}", text);
    }

    #[test]
    fn test_unknown_spec_is_invalid() {
        let outcome = render("no/such", &BindingSet::new());
        assert!(matches!(outcome, RenderOutcome::Invalid { .. }));
    }

    #[test]
    fn test_lint_flags_unknown_placeholder_and_policy() {
        static BAD: [PromptSpec; 1] = [PromptSpec {
            id: "bad",
            policy: "no_such_policy",
            template: "hi {undeclared}",
            required: &["missing_required"],
            optional: &[],
            example: &[],
            example_rendered: "",
            description: "bad",
        }];
        let issues = lint_specs(&BAD);
        let messages: Vec<&str> = issues.iter().map(|issue| issue.message.as_str()).collect();
        assert!(messages.iter().any(|message| message.contains("未知策略")));
        assert!(messages
            .iter()
            .any(|message| message.contains("未在 required/optional 中声明")));
        assert!(messages
            .iter()
            .any(|message| message.contains("未出现在模板中")));
        assert!(messages
            .iter()
            .any(|message| message.contains("example_rendered")));
    }

    #[test]
    fn test_coverage_reports_composed_policies() {
        let (covered, composed) = coverage();
        assert!(covered.contains(&"direct_answer"));
        assert!(composed.contains(&"recap"));
        assert_eq!(covered.len() + composed.len(), POLICY_NAMES.len());
    }

    #[test]
    fn test_stats_grounding_rate_and_absorb() {
        let mut stats = PromptStats::default();
        assert_eq!(stats.grounding_rate(), 1.0);
        stats.grounded = 3;
        stats.ungrounded = 1;
        stats.record_missing(&["attribute".to_string()]);
        assert!((stats.grounding_rate() - 0.75).abs() < 1e-12);
        let mut total = PromptStats::default();
        total.absorb(&stats);
        assert_eq!(total.grounded, 3);
        assert_eq!(total.missing_slots.get("attribute"), Some(&1));
    }
}

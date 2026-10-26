//! 回答计划（Answer Plan）与风格实现（Realization）。
//!
//! 生成式回答被拆成两层：
//!
//! - **计划层**：一组已经过生成自检、带稳定槽位与来源的命题。它只回答
//!   「说什么」，随会话快照持久化，重启后仍可复述；
//! - **表达层**：`StyleDirective` 只决定取哪些点、按什么顺序、是否披露
//!   来源与容量，绝不引入计划之外的 claim。
//!
//! 因此「简短点」「完整讲讲」「换种表达」「第二点是什么」都是对同一份
//! 计划的符号操作，而不是重新检索或继续枚举。计划带内容指纹、知识版本
//! 与容量上限：知识版本一变就拒绝旧计划，避免两代事实混在一段回答里。

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;

/// 回答计划的结构版本；不兼容变更时递增。
pub const ANSWER_PLAN_VERSION: u32 = 1;

/// 单份计划最多保留的命题数：限制快照体积与重述成本。
pub const MAX_PLAN_CLAIMS: usize = 24;

/// 简短模式默认展示的命题数。
pub const BRIEF_CLAIM_LIMIT: usize = 3;

/// 一条可被表达层选择的命题。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanClaim {
    /// 稳定槽位（由生成器的 `ClauseKey` 编码而来）。
    pub slot: String,
    /// 规范化谓词；用于详细模式的出处标签与换序表达。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub predicate: String,
    /// 已经过生成自检的完整句子。
    pub text: String,
    /// 是否来自本会话叠加层（未写入图谱）。
    #[serde(default, skip_serializing_if = "is_false")]
    pub session: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl PlanClaim {
    pub fn new(
        slot: impl Into<String>,
        predicate: impl Into<String>,
        text: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            slot: slot.into(),
            predicate: predicate.into(),
            text: text.into(),
            session,
        }
    }

    fn source_label(&self) -> &'static str {
        if self.session {
            "会话事实"
        } else {
            "图谱事实"
        }
    }
}

/// 一份可序列化、可追溯的回答计划。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerPlan {
    #[serde(default = "current_version")]
    pub version: u32,
    /// 内容指纹（与命题顺序无关，同内容同 id）。
    pub id: String,
    /// 触发本次计划的问句或请求。
    #[serde(default)]
    pub question: String,
    pub subject: String,
    /// 计划生成时的知识版本。
    pub revision: u64,
    pub claims: Vec<PlanClaim>,
    /// 因容量上限未纳入计划的命题数。
    #[serde(default)]
    pub omitted: usize,
    /// 因生成自检未通过而剔除的命题数。
    #[serde(default)]
    pub suppressed: usize,
}

fn current_version() -> u32 {
    ANSWER_PLAN_VERSION
}

impl AnswerPlan {
    /// 从已过滤的命题建立计划；容量上限在这里统一执行。
    pub fn new(
        question: impl Into<String>,
        subject: impl Into<String>,
        revision: u64,
        claims: Vec<PlanClaim>,
        omitted: usize,
        suppressed: usize,
    ) -> Self {
        let mut claims = claims;
        let overflow = claims.len().saturating_sub(MAX_PLAN_CLAIMS);
        claims.truncate(MAX_PLAN_CLAIMS);
        let mut plan = Self {
            version: ANSWER_PLAN_VERSION,
            id: String::new(),
            question: question.into(),
            subject: subject.into(),
            revision,
            claims,
            omitted: omitted.saturating_add(overflow),
            suppressed,
        };
        plan.id = plan.compute_id();
        plan
    }

    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    pub fn is_current_schema(&self) -> bool {
        self.version == ANSWER_PLAN_VERSION
    }

    /// 计划是否仍然对应当前知识版本。
    pub fn is_fresh_for(&self, revision: u64) -> bool {
        self.revision == revision
    }

    /// 内容指纹：命题按槽位与文本排序后参与哈希，展示顺序不影响 id。
    fn compute_id(&self) -> String {
        let mut claims: Vec<&PlanClaim> = self.claims.iter().collect();
        claims.sort_by(|left, right| {
            left.slot
                .cmp(&right.slot)
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| left.predicate.cmp(&right.predicate))
        });
        let mut material = String::new();
        let _ = write!(
            material,
            "v{}|{}|{}|{}|{}",
            self.version, self.subject, self.revision, self.omitted, self.suppressed
        );
        for claim in claims {
            let _ = write!(
                material,
                "\n{}|{}|{}|{}",
                claim.slot, claim.predicate, claim.text, claim.session
            );
        }
        fnv1a_hex(material.as_bytes())
    }

    /// 按风格指令表达同一份计划。
    pub fn realize(&self, directive: StyleDirective) -> Result<String, PlanRenderError> {
        if self.claims.is_empty() {
            return Err(PlanRenderError::Empty {
                subject: self.subject.clone(),
            });
        }
        match directive {
            StyleDirective::Brief => Ok(self.render_window(BRIEF_CLAIM_LIMIT, false)),
            StyleDirective::Detailed => Ok(self.render_window(usize::MAX, true)),
            StyleDirective::Restate => Ok(self.render_restate()),
            StyleDirective::Focus(index) => self.render_focus(index),
        }
    }

    fn render_window(&self, limit: usize, detailed: bool) -> String {
        let total = self.claims.len();
        let shown = limit.min(total);
        let mut output = if detailed {
            format!(
                "完整版（{}/{} 点，计划 {}，知识版本 v{}）：\n",
                shown, total, self.id, self.revision
            )
        } else {
            format!("简短版（取前 {}/{} 点，未新增事实）：\n", shown, total)
        };
        for (index, claim) in self.claims.iter().take(shown).enumerate() {
            let _ = writeln!(
                output,
                "  {}. {}{}",
                index + 1,
                claim.text.trim(),
                if detailed {
                    format!("（谓词：{}；来源：{}）", claim.predicate, claim.source_label())
                } else {
                    String::new()
                }
            );
        }
        if shown < total {
            let _ = writeln!(
                output,
                "  ……其余 {} 点未展开；输入「完整讲讲」查看全部。",
                total - shown
            );
        }
        self.append_plan_notes(&mut output, detailed);
        output.trim_end().to_string()
    }

    fn render_focus(&self, index: usize) -> Result<String, PlanRenderError> {
        let total = self.claims.len();
        let Some(claim) = self.claims.get(index.saturating_sub(1)) else {
            return Err(PlanRenderError::FocusOutOfRange {
                subject: self.subject.clone(),
                requested: index,
                total,
            });
        };
        let mut output = format!(
            "第 {} 点（共 {} 点，计划 {}，知识版本 v{}）：\n",
            index, total, self.id, self.revision
        );
        let _ = writeln!(
            output,
            "  {}（谓词：{}；来源：{}）",
            claim.text.trim(),
            claim.predicate,
            claim.source_label()
        );
        if total > 1 {
            let _ = writeln!(output, "  输入「完整讲讲」查看全部 {} 点。", total);
        }
        self.append_plan_notes(&mut output, true);
        Ok(output.trim_end().to_string())
    }

    fn render_restate(&self) -> String {
        let mut indices: Vec<usize> = (0..self.claims.len()).collect();
        indices.sort_by(|left, right| {
            let left = &self.claims[*left];
            let right = &self.claims[*right];
            left.predicate
                .cmp(&right.predicate)
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        let mut output = format!(
            "换种表达（同一份计划按主题重排，共 {} 点，未新增事实）：\n",
            self.claims.len()
        );
        for index in indices {
            let _ = writeln!(output, "  • {}", self.claims[index].text.trim());
        }
        self.append_plan_notes(&mut output, false);
        output.trim_end().to_string()
    }

    fn append_plan_notes(&self, output: &mut String, detailed: bool) {
        if self.suppressed > 0 {
            let _ = writeln!(
                output,
                "  （另有 {} 句未通过生成自检，未纳入计划。）",
                self.suppressed
            );
        }
        if self.omitted > 0 {
            let _ = writeln!(
                output,
                "  （计划容量上限 {} 点，另有 {} 点未纳入。）",
                MAX_PLAN_CLAIMS, self.omitted
            );
        }
        if detailed && self.claims.len() > BRIEF_CLAIM_LIMIT {
            let _ = writeln!(
                output,
                "  （说「简短点」可回到前 {} 点摘要。）",
                BRIEF_CLAIM_LIMIT
            );
        }
    }
}

/// 一次表达请求；只影响计划的选择与呈现。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleDirective {
    /// 只取计划前几点的简短版。
    Brief,
    /// 展开计划中的全部命题，并披露谓词与来源。
    Detailed,
    /// 同一组命题按主题重排，换一种说法。
    Restate,
    /// 只取第 N 点（从 1 开始）。
    Focus(usize),
}

impl StyleDirective {
    /// 解析自然语言或显式前缀（`风格：…` / `回答计划：…`）。
    ///
    /// 刻意不接管已有的「详细一点」「展开说说」「换个说法」：这些短语
    /// 在对话管理器中已有明确的续说/换角度语义，不能悄悄改成重述。
    pub fn parse(input: &str) -> Option<Self> {
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', '；', ';', '，', ',', ' '])
            .trim();
        if stripped.is_empty() {
            return None;
        }
        for prefix in ["风格：", "风格:", "回答计划：", "回答计划:"] {
            if let Some(rest) = stripped.strip_prefix(prefix) {
                let rest = rest.trim();
                if rest.is_empty() {
                    return None;
                }
                if let Some(index) = parse_focus(rest, true) {
                    return Some(StyleDirective::Focus(index));
                }
                return match rest {
                    "简短" | "短" | "精简" | "要点" | "重点" => Some(StyleDirective::Brief),
                    "详细" | "完整" | "全部" | "展开" => Some(StyleDirective::Detailed),
                    "换种表达" | "换一种表达" | "换种说法" | "重排" | "重新组织" => {
                        Some(StyleDirective::Restate)
                    }
                    _ => None,
                };
            }
        }
        if BRIEF_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Brief);
        }
        if DETAILED_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Detailed);
        }
        if RESTATE_PHRASES.contains(&stripped) {
            return Some(StyleDirective::Restate);
        }
        parse_focus(stripped, false).map(StyleDirective::Focus)
    }

    pub fn label(self) -> &'static str {
        match self {
            StyleDirective::Brief => "简短版",
            StyleDirective::Detailed => "完整版",
            StyleDirective::Restate => "换种表达",
            StyleDirective::Focus(_) => "定位单点",
        }
    }
}

const BRIEF_PHRASES: &[&str] = &[
    "简短点",
    "简短一点",
    "简短些",
    "说简短点",
    "讲简短点",
    "精简点",
    "精简一下",
    "只讲重点",
    "只讲要点",
    "说重点",
    "讲重点",
    "一句话说说",
    "概括一下",
    "概括说说",
];

const DETAILED_PHRASES: &[&str] = &[
    "说详细点",
    "讲详细点",
    "完整讲讲",
    "完整说说",
    "完整说一遍",
    "全部展开",
    "展开全部",
    "详细展开",
    "详细版",
    "完整版",
    "回到详细版",
    "回到完整版",
    "全都说说",
];

const RESTATE_PHRASES: &[&str] = &[
    "换种表达",
    "换一种表达",
    "重新表达",
    "换种说法",
    "再表达一次",
    "重排一下",
    "换个叙述顺序",
];

fn parse_focus(text: &str, explicit: bool) -> Option<usize> {
    let mut body = text.trim();
    let mut prefixed = false;
    for prefix in ["展开", "回到", "看看", "讲讲", "说说", "看"] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim();
            prefixed = true;
            break;
        }
    }
    let had_di = if let Some(rest) = body.strip_prefix('第') {
        body = rest.trim();
        true
    } else {
        false
    };
    let (index, rest) = split_ordinal(body)?;
    let rest = rest.trim();
    let named_point = matches!(
        rest,
        "点" | "点呢"
            | "点是什么"
            | "点什么"
            | "点的内容"
            | "点内容"
            | "点讲讲"
            | "点说说"
            | "条"
            | "条呢"
    );
    // 裸数字「1」是澄清短答或普通数字，不是风格指令；只有显式前缀
    // 或「展开/第N点」这类带点标记的说法才进入计划定位。
    let suffix_ok = named_point || (rest.is_empty() && (explicit || prefixed));
    if !suffix_ok {
        return None;
    }
    let bare = had_di && rest == "点";
    if bare && !explicit && !prefixed {
        return None;
    }
    Some(index)
}

/// 读取开头的中文/阿拉伯序数，返回数值与剩余文本。
fn split_ordinal(text: &str) -> Option<(usize, &str)> {
    let mut chars = text.char_indices().peekable();
    let (_, first) = chars.next()?;
    if first.is_ascii_digit() {
        let end = text
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(text.len());
        let value: usize = text[..end].parse().ok()?;
        return (value > 0).then_some((value, &text[end..]));
    }
    let mut value = 0usize;
    let mut current = 0usize;
    let mut consumed = 0usize;
    for (offset, ch) in text.char_indices() {
        if let Some(digit) = chinese_digit(ch) {
            current = digit;
            consumed = offset + ch.len_utf8();
        } else if ch == '十' {
            let base = if current == 0 { 1 } else { current };
            value = value.saturating_add(base.saturating_mul(10));
            current = 0;
            consumed = offset + ch.len_utf8();
        } else if ch == '两' {
            current = 2;
            consumed = offset + ch.len_utf8();
        } else {
            break;
        }
    }
    if consumed == 0 {
        return None;
    }
    let value = value.saturating_add(current);
    (value > 0).then_some((value, &text[consumed..]))
}

fn chinese_digit(ch: char) -> Option<usize> {
    match ch {
        '一' => Some(1),
        '二' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

/// 表达失败：计划为空或定位越界。调用方据此给出可行动的提示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRenderError {
    Empty {
        subject: String,
    },
    FocusOutOfRange {
        subject: String,
        requested: usize,
        total: usize,
    },
}

impl std::fmt::Display for PlanRenderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanRenderError::Empty { subject } => {
                write!(formatter, "关于「{}」的回答计划为空，没有可复述的命题。", subject)
            }
            PlanRenderError::FocusOutOfRange {
                subject,
                requested,
                total,
            } => write!(
                formatter,
                "关于「{}」的回答计划只有 {} 点，无法定位第 {} 点。",
                subject, total, requested
            ),
        }
    }
}

impl std::error::Error for PlanRenderError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_claims() -> Vec<PlanClaim> {
        vec![
            PlanClaim::new("a#age", "age", "张三的年龄是18岁。", false),
            PlanClaim::new("r#师傅", "师傅", "张三的师傅是李四。", false),
            PlanClaim::new("a#出生地", "出生地", "张三的出生地是北京。", true),
            PlanClaim::new("r#徒弟", "徒弟", "张三的徒弟是王五。", false),
            PlanClaim::new("a#身高", "身高", "张三的身高是175厘米。", false),
        ]
    }

    fn sample_plan() -> AnswerPlan {
        AnswerPlan::new("介绍：张三", "张三", 7, sample_claims(), 0, 0)
    }

    #[test]
    fn plan_id_is_order_independent_and_revision_sensitive() {
        let plan = sample_plan();
        let mut reversed = sample_claims();
        reversed.reverse();
        let other = AnswerPlan::new("换个问法", "张三", 7, reversed, 0, 0);
        assert_eq!(plan.id, other.id, "同内容不同展示顺序应共享计划 id");
        let newer = AnswerPlan::new("介绍：张三", "张三", 8, sample_claims(), 0, 0);
        assert_ne!(plan.id, newer.id, "知识版本必须进入内容指纹");
        assert!(plan.is_fresh_for(7));
        assert!(!plan.is_fresh_for(8));
    }

    #[test]
    fn brief_uses_only_prefix_claims_and_never_invents() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Brief).unwrap();
        assert!(text.contains("简短版"), "{}", text);
        for claim in plan.claims.iter().take(BRIEF_CLAIM_LIMIT) {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        for claim in plan.claims.iter().skip(BRIEF_CLAIM_LIMIT) {
            assert!(!text.contains(claim.text.trim()), "越界 claim：{}", text);
        }
        assert!(text.contains("完整讲讲"), "{}", text);
    }

    #[test]
    fn detailed_discloses_every_claim_and_its_source() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Detailed).unwrap();
        assert!(text.contains("完整版"), "{}", text);
        assert!(text.contains(&plan.id), "{}", text);
        assert!(text.contains("知识版本 v7"), "{}", text);
        for claim in &plan.claims {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        assert!(text.contains("来源：会话事实"), "{}", text);
        assert!(text.contains("来源：图谱事实"), "{}", text);
    }

    #[test]
    fn focus_returns_exactly_one_claim_and_reports_range_errors() {
        let plan = sample_plan();
        let text = plan.realize(StyleDirective::Focus(2)).unwrap();
        assert!(text.contains("第 2 点"), "{}", text);
        assert!(text.contains(plan.claims[1].text.trim()), "{}", text);
        assert!(!text.contains(plan.claims[0].text.trim()), "{}", text);
        assert!(!text.contains(plan.claims[2].text.trim()), "{}", text);

        let error = plan.realize(StyleDirective::Focus(9)).unwrap_err();
        assert!(error.to_string().contains("只有 5 点"), "{}", error);
        assert!(error.to_string().contains("第 9 点"), "{}", error);
    }

    #[test]
    fn restate_preserves_the_claim_set_but_reorders_it() {
        let plan = AnswerPlan::new(
            "介绍：张三",
            "张三",
            7,
            vec![
                PlanClaim::new("z", "zeta", "先讲的命题。", false),
                PlanClaim::new("a", "alpha", "后讲的命题。", false),
            ],
            0,
            0,
        );
        let text = plan.realize(StyleDirective::Restate).unwrap();
        assert!(text.contains("换种表达"), "{}", text);
        for claim in &plan.claims {
            assert!(text.contains(claim.text.trim()), "{}", text);
        }
        let first_restated = text
            .lines()
            .find(|line| line.trim_start().starts_with('•'))
            .expect("换种表达应输出条目");
        assert!(
            first_restated.contains("后讲的命题"),
            "按谓词重排后应把 alpha 放到最前：{}",
            text
        );
    }

    #[test]
    fn empty_plan_is_an_explicit_error_not_a_blank_answer() {
        let plan = AnswerPlan::new("介绍：无人", "无人", 1, Vec::new(), 0, 0);
        let error = plan.realize(StyleDirective::Brief).unwrap_err();
        assert!(error.to_string().contains("无人"), "{}", error);
    }

    #[test]
    fn capacity_and_suppression_are_disclosed() {
        let mut claims = Vec::new();
        for index in 0..(MAX_PLAN_CLAIMS + 2) {
            claims.push(PlanClaim::new(
                format!("s{}", index),
                "p",
                format!("第{}条事实。", index),
                false,
            ));
        }
        let plan = AnswerPlan::new("介绍：很多", "很多", 1, claims, 0, 3);
        assert_eq!(plan.claim_count(), MAX_PLAN_CLAIMS);
        assert_eq!(plan.omitted, 2);
        let text = plan.realize(StyleDirective::Detailed).unwrap();
        assert!(text.contains("未通过生成自检"), "{}", text);
        assert!(text.contains("容量上限"), "{}", text);
    }

    #[test]
    fn parser_accepts_new_style_words_and_leaves_existing_followups_alone() {
        for (input, expected) in [
            ("简短点", StyleDirective::Brief),
            ("只讲重点", StyleDirective::Brief),
            ("完整讲讲", StyleDirective::Detailed),
            ("回到详细版", StyleDirective::Detailed),
            ("换种表达", StyleDirective::Restate),
            ("第二点是什么", StyleDirective::Focus(2)),
            ("展开第三点", StyleDirective::Focus(3)),
            ("风格：第2点", StyleDirective::Focus(2)),
            ("回答计划：简短", StyleDirective::Brief),
            ("风格：完整", StyleDirective::Detailed),
            ("风格：换种表达", StyleDirective::Restate),
        ] {
            assert_eq!(
                StyleDirective::parse(input),
                Some(expected),
                "应识别「{}」",
                input
            );
        }
        for input in [
            "详细一点",
            "展开说说",
            "展开讲",
            "换个说法",
            "换个角度说",
            "接着说",
            "第二点",
            "1",
            "2",
            "二十",
            "介绍一下北京",
        ] {
            assert_eq!(StyleDirective::parse(input), None, "不应接管「{}」", input);
        }
    }

    #[test]
    fn plan_survives_json_roundtrip_and_old_payloads_default_to_none() {
        let plan = sample_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let restored: AnswerPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, plan);
        assert!(restored.is_current_schema());

        #[derive(Deserialize)]
        struct Holder {
            #[serde(default)]
            plan: Option<AnswerPlan>,
        }
        let old: Holder = serde_json::from_str("{}").unwrap();
        assert!(old.plan.is_none());
    }
}

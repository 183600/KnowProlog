//! AT：回答分类学与回答回合（Answer Taxonomy & Answer Episode）。
//!
//! 项目里已有三套彼此独立、但缺少交叉引用的词表：`MoveKind` 描述话语动作，
//! `query_type` 描述查询规则，生成器与查询板又各自用计划标签表达回答形态。
//! 本模块补上中间那一层：**这个问题应该按哪一种回答图式来组织**。
//!
//! - [`AnswerSchema`]：回答图式（身份 / 属性 / 关系 / 对比 / 定义 / 解释 …）；
//! - [`SchemaSpec`]：每类图式对应的 `query_type`、必需槽位、可选槽位与合法追问；
//! - [`AnswerEpisode`]：一次回答实例的只读投影——图式、主体、已填槽位、
//!   仍欠哪些槽位、证据版本与连续轮次；
//! - [`FollowUpKind`]：把「还有呢 / 为什么 / 换个说法 / 那 X 呢」读成对同一
//!   回答回合的受类型约束操作，而不是重新检索。
//!
//! 设计纪律：
//! - 分类只做确定性映射，不引入概率、权重或训练；
//! - 未知问题宁可落到 [`AnswerSchema::Unknown`] 并显式欠一个澄清，
//!   也不猜成某个具体图式；
//! - `query_type` 覆盖由 [`lint_query_types`] 守卫：新增查询类型而忘记
//!   登记回答图式会得到稳定的 `unmapped_query_type`；
//! - 回合是投影而非第二份真相源：它从对话轮次重建，不改变既有回答字节。

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::{DialogueTurn, MoveKind};

/// 回答分类学结构版本；不兼容变更时递增。
pub const ANSWER_TAXONOMY_VERSION: u32 = 1;

/// 回答图式：决定一段回答要按什么结构组织。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum AnswerSchema {
    /// 实体身份：这是谁 / 是什么。
    Identity,
    /// 单值或少量取值属性：年龄、身高、国籍。
    Attribute,
    /// 两个实体之间的关系。
    Relation,
    /// 两个及以上对象的对比、排序或极值。
    Comparison,
    /// 概念定义与上下位归属。
    Definition,
    /// 因果、原理、影响等解释。
    Explanation,
    /// 步骤、方法或操作过程。
    Procedure,
    /// 列举、筛选、分页与集合扫描。
    Enumeration,
    /// 计数。
    Count,
    /// 是非判断。
    Truth,
    /// 回顾本轮之前说过什么。
    Recap,
    /// 复核计划或知识版本。
    Review,
    /// 寒暄、致谢与告别。
    Social,
    /// 无法归入以上图式：保留澄清义务，不硬猜。
    Unknown,
}

impl AnswerSchema {
    /// 稳定机器标签（回执、契约与基线使用）。
    pub fn label(self) -> &'static str {
        match self {
            AnswerSchema::Identity => "identity",
            AnswerSchema::Attribute => "attribute",
            AnswerSchema::Relation => "relation",
            AnswerSchema::Comparison => "comparison",
            AnswerSchema::Definition => "definition",
            AnswerSchema::Explanation => "explanation",
            AnswerSchema::Procedure => "procedure",
            AnswerSchema::Enumeration => "enumeration",
            AnswerSchema::Count => "count",
            AnswerSchema::Truth => "truth",
            AnswerSchema::Recap => "recap",
            AnswerSchema::Review => "review",
            AnswerSchema::Social => "social",
            AnswerSchema::Unknown => "unknown",
        }
    }

    /// 面向用户的中文标签。
    pub fn label_zh(self) -> &'static str {
        match self {
            AnswerSchema::Identity => "身份",
            AnswerSchema::Attribute => "属性",
            AnswerSchema::Relation => "关系",
            AnswerSchema::Comparison => "对比",
            AnswerSchema::Definition => "定义",
            AnswerSchema::Explanation => "解释",
            AnswerSchema::Procedure => "步骤",
            AnswerSchema::Enumeration => "列举",
            AnswerSchema::Count => "计数",
            AnswerSchema::Truth => "判断",
            AnswerSchema::Recap => "回顾",
            AnswerSchema::Review => "复核",
            AnswerSchema::Social => "寒暄",
            AnswerSchema::Unknown => "未分类",
        }
    }

    /// 是否是知识型回答（元话语与寒暄不算）。
    pub fn is_knowledge(self) -> bool {
        !matches!(
            self,
            AnswerSchema::Recap | AnswerSchema::Review | AnswerSchema::Social
        )
    }

    pub fn from_label(label: &str) -> Option<Self> {
        TAXONOMY
            .iter()
            .map(|spec| spec.schema)
            .find(|schema| schema.label() == label.trim())
    }
}

/// 话语角色：一句话在回答结构中承担什么功能。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DiscourseRole {
    /// 定义或上位归属。
    Definition,
    /// 直接回答。
    Answer,
    /// 证据、出处与推导依据。
    Evidence,
    /// 对比或差异。
    Comparison,
    /// 边界、例外与容量说明。
    Boundary,
    /// 追问建议或下一步。
    FollowUp,
    /// 不确定性、冲突与保留意见。
    Caveat,
}

impl DiscourseRole {
    pub fn label(self) -> &'static str {
        match self {
            DiscourseRole::Definition => "definition",
            DiscourseRole::Answer => "answer",
            DiscourseRole::Evidence => "evidence",
            DiscourseRole::Comparison => "comparison",
            DiscourseRole::Boundary => "boundary",
            DiscourseRole::FollowUp => "follow_up",
            DiscourseRole::Caveat => "caveat",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            DiscourseRole::Definition => "定义",
            DiscourseRole::Answer => "回答",
            DiscourseRole::Evidence => "证据",
            DiscourseRole::Comparison => "对比",
            DiscourseRole::Boundary => "边界",
            DiscourseRole::FollowUp => "追问",
            DiscourseRole::Caveat => "保留",
        }
    }
}

/// 回答回合需要填的槽位。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Obligation {
    /// 主体实体。
    Subject,
    /// 谓词或属性槽。
    Predicate,
    /// 实际回答内容。
    Answer,
    /// 可核查依据。
    Evidence,
    /// 对比维度或差异。
    Contrast,
    /// 边界与例外。
    Boundary,
    /// 消歧或澄清。
    Clarification,
}

impl Obligation {
    pub fn label(self) -> &'static str {
        match self {
            Obligation::Subject => "subject",
            Obligation::Predicate => "predicate",
            Obligation::Answer => "answer",
            Obligation::Evidence => "evidence",
            Obligation::Contrast => "contrast",
            Obligation::Boundary => "boundary",
            Obligation::Clarification => "clarification",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Obligation::Subject => "主体",
            Obligation::Predicate => "属性",
            Obligation::Answer => "回答",
            Obligation::Evidence => "依据",
            Obligation::Contrast => "对比",
            Obligation::Boundary => "边界",
            Obligation::Clarification => "澄清",
        }
    }

    /// 每轮都要重新履行、不随追问继承的槽位。
    pub fn is_transient(self) -> bool {
        matches!(
            self,
            Obligation::Answer
                | Obligation::Evidence
                | Obligation::Contrast
                | Obligation::Clarification
        )
    }
}

/// 追问类型：对当前回答回合的合法操作。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FollowUpKind {
    /// 继续展开同一回答。
    More,
    /// 追问依据或原因。
    Why,
    /// 下一步 / 下一点。
    Next,
    /// 换一种说法。
    Paraphrase,
    /// 对比或换个维度。
    Compare,
    /// 需要澄清指称或问题。
    Clarify,
    /// 切换到新的主体或问题。
    Switch,
}

impl FollowUpKind {
    pub fn label(self) -> &'static str {
        match self {
            FollowUpKind::More => "more",
            FollowUpKind::Why => "why",
            FollowUpKind::Next => "next",
            FollowUpKind::Paraphrase => "paraphrase",
            FollowUpKind::Compare => "compare",
            FollowUpKind::Clarify => "clarify",
            FollowUpKind::Switch => "switch",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            FollowUpKind::More => "继续",
            FollowUpKind::Why => "依据",
            FollowUpKind::Next => "下一步",
            FollowUpKind::Paraphrase => "换说法",
            FollowUpKind::Compare => "对比",
            FollowUpKind::Clarify => "澄清",
            FollowUpKind::Switch => "换题",
        }
    }

    /// 从字面识别追问意图；识别不到时返回 `None`（按新问题处理）。
    pub fn from_input(input: &str) -> Option<Self> {
        let text = input.trim().trim_end_matches([
            '？', '?', '。', '.', '！', '!', '，', ',', ' ', '呢',
        ]);
        if text.is_empty() {
            return None;
        }
        const WHY: [&str; 8] = [
            "为什么",
            "为何",
            "怎么知道",
            "怎么推",
            "依据",
            "根据什么",
            "凭什么",
            "证据",
        ];
        const PARAPHRASE: [&str; 6] = [
            "换个说法",
            "换种说法",
            "换一种说法",
            "改写",
            "再说一遍",
            "重说",
        ];
        const MORE: [&str; 8] = [
            "还有",
            "继续",
            "展开",
            "详细",
            "多说",
            "然后",
            "补充",
            "讲完",
        ];
        const NEXT: [&str; 5] = ["下一步", "下一点", "下一个", "第几", "第二步"];
        const COMPARE: [&str; 5] = ["对比", "比较", "区别", "差异", "哪个更"];
        const CLARIFY: [&str; 7] = [
            "澄清",
            "你指",
            "哪一个",
            "什么意思",
            "是指",
            "确认一下",
            "再说清楚",
        ];
        if WHY.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::Why);
        }
        if PARAPHRASE.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::Paraphrase);
        }
        if COMPARE.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::Compare);
        }
        if CLARIFY.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::Clarify);
        }
        if NEXT.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::Next);
        }
        if MORE.iter().any(|marker| text.contains(marker)) {
            return Some(FollowUpKind::More);
        }
        None
    }

    /// 该追问是否属于同一回答回合（切换话题除外）。
    pub fn stays_in_episode(self) -> bool {
        !matches!(self, FollowUpKind::Switch)
    }
}

/// 一类回答图式的完整规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaSpec {
    pub schema: AnswerSchema,
    /// 该图式接受的稳定 `query_type`。
    pub query_types: &'static [&'static str],
    /// 必须填的槽位。
    pub required: &'static [Obligation],
    /// 有则更好、但缺失不算未完成的槽位。
    pub optional: &'static [Obligation],
    /// 合法追问。
    pub followups: &'static [FollowUpKind],
}

const ALL_FOLLOWUPS: &[FollowUpKind] = &[
    FollowUpKind::More,
    FollowUpKind::Why,
    FollowUpKind::Next,
    FollowUpKind::Paraphrase,
    FollowUpKind::Compare,
    FollowUpKind::Clarify,
    FollowUpKind::Switch,
];

const ATTRIBUTE_QUERY_TYPES: &[&str] = &[
    "age_query",
    "altitude_query",
    "author_query",
    "birth_year_query",
    "capital_query",
    "champion_query",
    "chemical_formula_query",
    "planet_query",
    "population_query",
    "property_query",
    "quantity_query",
    "spouse_query",
    "time_before_query",
];

const RELATION_QUERY_TYPES: &[&str] = &["relation_query"];
const COMPARISON_QUERY_TYPES: &[&str] = &["comparison_query", "largest_query"];
const DEFINITION_QUERY_TYPES: &[&str] = &["concept_query", "philosophy_query"];
const EXPLANATION_QUERY_TYPES: &[&str] = &[
    "cause_query",
    "explanation_query",
    "impact_query",
    "principle_query",
    "reason_query",
];
const COUNT_QUERY_TYPES: &[&str] = &["count_query"];

/// 静态回答分类学。顺序即报告顺序。
pub const TAXONOMY: &[SchemaSpec] = &[
    SchemaSpec {
        schema: AnswerSchema::Identity,
        query_types: &[],
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Attribute,
        query_types: ATTRIBUTE_QUERY_TYPES,
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Predicate, Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Relation,
        query_types: RELATION_QUERY_TYPES,
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Predicate, Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Comparison,
        query_types: COMPARISON_QUERY_TYPES,
        required: &[
            Obligation::Subject,
            Obligation::Contrast,
            Obligation::Answer,
            Obligation::Evidence,
        ],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Definition,
        query_types: DEFINITION_QUERY_TYPES,
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Explanation,
        query_types: EXPLANATION_QUERY_TYPES,
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Procedure,
        query_types: &[],
        required: &[Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Subject, Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Enumeration,
        query_types: &[],
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Count,
        query_types: COUNT_QUERY_TYPES,
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Truth,
        query_types: &[],
        required: &[Obligation::Subject, Obligation::Answer, Obligation::Evidence],
        optional: &[Obligation::Boundary],
        followups: ALL_FOLLOWUPS,
    },
    SchemaSpec {
        schema: AnswerSchema::Recap,
        query_types: &[],
        required: &[],
        optional: &[Obligation::Answer],
        followups: &[FollowUpKind::More, FollowUpKind::Paraphrase, FollowUpKind::Switch],
    },
    SchemaSpec {
        schema: AnswerSchema::Review,
        query_types: &[],
        required: &[],
        optional: &[Obligation::Answer, Obligation::Evidence],
        followups: &[FollowUpKind::More, FollowUpKind::Why, FollowUpKind::Switch],
    },
    SchemaSpec {
        schema: AnswerSchema::Social,
        query_types: &[],
        required: &[],
        optional: &[],
        followups: &[FollowUpKind::Switch],
    },
    SchemaSpec {
        schema: AnswerSchema::Unknown,
        query_types: &[],
        required: &[Obligation::Clarification],
        optional: &[Obligation::Subject, Obligation::Answer],
        followups: &[FollowUpKind::Clarify, FollowUpKind::Switch],
    },
];

/// 取某图式的规格；分类学封闭，总能命中。
pub fn spec(schema: AnswerSchema) -> &'static SchemaSpec {
    TAXONOMY
        .iter()
        .find(|spec| spec.schema == schema)
        .expect("回答分类学必须覆盖所有 AnswerSchema")
}

/// 把稳定 `query_type` 映射到回答图式；未知类型返回 `None`，由调用方显式处理。
pub fn classify_query_type(query_type: &str) -> Option<AnswerSchema> {
    let query_type = query_type.trim();
    TAXONOMY
        .iter()
        .find(|spec| spec.query_types.contains(&query_type))
        .map(|spec| spec.schema)
}

/// 把话语动作映射到回答图式；追问继承由 [`classify_input`] 处理。
pub fn classify_move(kind: &MoveKind) -> AnswerSchema {
    match kind {
        MoveKind::AskIdentity => AnswerSchema::Identity,
        MoveKind::AskAttribute => AnswerSchema::Attribute,
        MoveKind::AskBinary => AnswerSchema::Truth,
        MoveKind::MetaRecap => AnswerSchema::Recap,
        MoveKind::MetaReview => AnswerSchema::Review,
        MoveKind::MetaWhy => AnswerSchema::Explanation,
        MoveKind::Greet | MoveKind::Thanks | MoveKind::Farewell => AnswerSchema::Social,
        MoveKind::ContinueRequest | MoveKind::ElaborateRequest => AnswerSchema::Unknown,
        MoveKind::Assert | MoveKind::Other => AnswerSchema::Unknown,
    }
}

/// 结合上一回合与追问字面做分类。
///
/// 同一回答回合内的追问（继续 / 依据 / 换说法 / 对比 / 澄清）继承上一图式；
/// 显式切换话题或没有上一回合时，按本轮话语动作重新分类。
pub fn classify_input(input: &str, kind: &MoveKind, previous: Option<AnswerSchema>) -> AnswerSchema {
    if let (Some(previous), Some(followup)) = (previous, FollowUpKind::from_input(input)) {
        if followup.stays_in_episode() {
            return previous;
        }
    }
    match kind {
        MoveKind::ContinueRequest | MoveKind::ElaborateRequest => {
            previous.unwrap_or(AnswerSchema::Unknown)
        }
        _ => classify_move(kind),
    }
}

/// 分类学覆盖的问题：稳定机器码 + 具体 `query_type`。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaxonomyIssue {
    pub code: String,
    pub query_type: String,
    pub detail: String,
}

/// 检查一组 `query_type` 是否都有唯一的回答图式。
///
/// 未登记 -> `unmapped_query_type`；同一类型被多个图式认领 ->
/// `ambiguous_query_type`。结果按类型名排序，便于基线比较。
pub fn lint_query_types<'a, I>(query_types: I) -> Vec<TaxonomyIssue>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut owners: std::collections::BTreeMap<String, BTreeSet<AnswerSchema>> =
        std::collections::BTreeMap::new();
    for query_type in query_types {
        let query_type = query_type.trim();
        if query_type.is_empty() {
            continue;
        }
        let entry = owners.entry(query_type.to_string()).or_default();
        for spec in TAXONOMY {
            if spec.query_types.contains(&query_type) {
                entry.insert(spec.schema);
            }
        }
    }
    let mut issues = Vec::new();
    for (query_type, schemas) in owners {
        match schemas.len() {
            0 => issues.push(TaxonomyIssue {
                code: "unmapped_query_type".to_string(),
                query_type: query_type.clone(),
                detail: format!(
                    "query_type「{}」没有回答图式；请在 TAXONOMY 中登记，或显式归入 Unknown",
                    query_type
                ),
            }),
            1 => {}
            _ => issues.push(TaxonomyIssue {
                code: "ambiguous_query_type".to_string(),
                query_type: query_type.clone(),
                detail: format!(
                    "query_type「{}」被多个回答图式认领：{}",
                    query_type,
                    schemas
                        .iter()
                        .map(|schema| schema.label())
                        .collect::<Vec<_>>()
                        .join(" / ")
                ),
            }),
        }
    }
    issues
}

/// 分类学报告：图式、query_type 覆盖、必需槽位与合法追问。
pub fn render_table() -> String {
    let mut lines = vec![format!(
        "回答分类学 v{}（AT）：{} 个图式，{} 个 query_type。",
        ANSWER_TAXONOMY_VERSION,
        TAXONOMY.len(),
        TAXONOMY
            .iter()
            .map(|spec| spec.query_types.len())
            .sum::<usize>()
    )];
    for spec in TAXONOMY {
        let required = spec
            .required
            .iter()
            .map(|obligation| obligation.label())
            .collect::<Vec<_>>()
            .join(",");
        let followups = spec
            .followups
            .iter()
            .map(|followup| followup.label())
            .collect::<Vec<_>>()
            .join(",");
        let query_types = if spec.query_types.is_empty() {
            "-".to_string()
        } else {
            spec.query_types.join(",")
        };
        lines.push(format!(
            "  {:<12} 查询：{}；必需：{}；追问：{}",
            spec.schema.label(),
            query_types,
            if required.is_empty() { "-" } else { &required },
            followups
        ));
    }
    lines.join("\n")
}

/// 一次回答回合的观察输入。
#[derive(Debug, Clone, Copy)]
pub struct EpisodeObservation<'a> {
    pub question: &'a str,
    pub kind: &'a MoveKind,
    pub subject: Option<&'a str>,
    pub predicate: Option<&'a str>,
    pub plan: &'a [String],
    pub answer: Option<&'a str>,
    pub evidence: &'a [String],
    pub revision: u64,
}

/// 一次回答实例：图式 + 已填槽位 + 仍欠槽位。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerEpisode {
    #[serde(default = "current_version")]
    pub version: u32,
    pub schema: AnswerSchema,
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filled: Vec<Obligation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing: Vec<Obligation>,
    /// 追加的追问义务（例如「为什么」要求补齐依据）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra: Vec<Obligation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub followups: Vec<FollowUpKind>,
    pub revision: u64,
    pub turns: u32,
}

fn current_version() -> u32 {
    ANSWER_TAXONOMY_VERSION
}

impl AnswerEpisode {
    /// 建立或推进一个回答回合。
    pub fn observe(observation: &EpisodeObservation<'_>, previous: Option<&Self>) -> Self {
        let previous_schema = previous.map(|episode| episode.schema);
        let followup = FollowUpKind::from_input(observation.question);
        let schema = classify_input(observation.question, observation.kind, previous_schema);
        let subject = observation
            .subject
            .map(str::to_string)
            .or_else(|| {
                previous.and_then(|episode| {
                    followup
                        .map(FollowUpKind::stays_in_episode)
                        .unwrap_or(false)
                        .then(|| episode.subject.clone())
                        .flatten()
                })
            });
        let predicate = observation.predicate.map(str::to_string).or_else(|| {
            previous.and_then(|episode| {
                followup
                    .map(FollowUpKind::stays_in_episode)
                    .unwrap_or(false)
                    .then(|| episode.predicate.clone())
                    .flatten()
            })
        });

        let continues = previous.is_some_and(|episode| {
            episode.schema == schema
                && followup
                    .map(FollowUpKind::stays_in_episode)
                    .unwrap_or(false)
                && (subject.is_none()
                    || episode.subject.is_none()
                    || episode.subject == subject)
                && (predicate.is_none()
                    || episode.predicate.is_none()
                    || episode.predicate == predicate)
        });

        let mut extra = if continues {
            previous.map(|episode| episode.extra.clone()).unwrap_or_default()
        } else {
            Vec::new()
        };
        if let Some(followup) = followup {
            match followup {
                FollowUpKind::More => {
                    extra.push(Obligation::Answer);
                    extra.push(Obligation::Evidence);
                }
                FollowUpKind::Next | FollowUpKind::Paraphrase => {
                    extra.push(Obligation::Answer)
                }
                FollowUpKind::Why => extra.push(Obligation::Evidence),
                FollowUpKind::Compare => {
                    extra.push(Obligation::Contrast);
                    extra.push(Obligation::Evidence);
                }
                FollowUpKind::Clarify => extra.push(Obligation::Clarification),
                FollowUpKind::Switch => {}
            }
        }
        extra.sort();
        extra.dedup();

        let mut filled = if continues {
            previous
                .map(|episode| {
                    episode
                        .filled
                        .iter()
                        .copied()
                        .filter(|obligation| !obligation.is_transient())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut mark = |obligation: Obligation| {
            if !filled.contains(&obligation) {
                filled.push(obligation);
            }
        };
        if subject.is_some() {
            mark(Obligation::Subject);
        }
        if predicate.is_some() {
            mark(Obligation::Predicate);
        }
        let answered = observation
            .answer
            .map(|answer| !answer.trim().is_empty())
            .unwrap_or(false)
            && !plan_has_gap(observation.plan);
        if answered {
            mark(Obligation::Answer);
            if schema == AnswerSchema::Comparison {
                mark(Obligation::Contrast);
            }
        }
        if !observation.evidence.is_empty() {
            mark(Obligation::Evidence);
        }
        if plan_has_clarification(observation.plan) {
            mark(Obligation::Clarification);
        }
        filled.sort();
        filled.dedup();

        let mut required = spec(schema).required.to_vec();
        required.extend(extra.iter().copied());
        required.sort();
        required.dedup();
        let missing: Vec<Obligation> = required
            .into_iter()
            .filter(|obligation| !filled.contains(obligation))
            .collect();

        let turns = if continues {
            previous.map(|episode| episode.turns.saturating_add(1)).unwrap_or(1)
        } else {
            1
        };
        let mut followups = if continues {
            previous.map(|episode| episode.followups.clone()).unwrap_or_default()
        } else {
            Vec::new()
        };
        if let Some(followup) = followup {
            if !followups.contains(&followup) {
                followups.push(followup);
            }
        }
        Self {
            version: ANSWER_TAXONOMY_VERSION,
            schema,
            question: observation.question.trim().to_string(),
            subject,
            predicate,
            filled,
            missing,
            extra,
            followups,
            revision: observation.revision,
            turns,
        }
    }

    /// 用本轮实际回答文本推进回合。
    ///
    /// 生成管线并不总把最终文本写回 `DialogueTurn.answer_excerpt`（例如续聊
    /// 轮只记录策略层的原始答案），因此回执应以实际回复为准。本函数不写任何
    /// 会话状态，只在只读投影上合并本轮观察。
    #[allow(clippy::too_many_arguments)]
    pub fn observe_reply(
        previous: Option<&Self>,
        question: &str,
        kind: &MoveKind,
        subject: Option<&str>,
        predicate: Option<&str>,
        plan: &[String],
        reply: &str,
        evidence: &[String],
        revision: u64,
    ) -> Self {
        let observation = EpisodeObservation {
            question,
            kind,
            subject,
            predicate,
            plan,
            answer: Some(reply),
            evidence,
            revision,
        };
        Self::observe(&observation, previous)
    }

    /// 当前回合是否已无必需槽位欠账。
    pub fn is_closed(&self) -> bool {
        self.missing.is_empty()
    }

    /// 合法追问（来自图式规格）。
    pub fn allows(&self, followup: FollowUpKind) -> bool {
        spec(self.schema).followups.contains(&followup)
    }

    /// 从已持久化的对话轮次重建最近一次回答回合。
    ///
    /// 回合是投影：不写快照、不改回答，只在读取时按同一套规则重放。
    pub fn from_turns(turns: &[DialogueTurn]) -> Option<Self> {
        let mut episode: Option<Self> = None;
        for turn in turns {
            let observation = EpisodeObservation {
                question: &turn.user,
                kind: &turn.kind,
                subject: turn.entity.as_deref(),
                predicate: turn.attribute.as_deref(),
                plan: &turn.plan,
                answer: turn.answer_excerpt.as_deref(),
                evidence: &turn.provenance,
                revision: turn.kb_revision.unwrap_or(0),
            };
            episode = Some(Self::observe(&observation, episode.as_ref()));
        }
        episode
    }

    /// 面向用户的一行摘要。
    pub fn render_zh(&self) -> String {
        let subject = self.subject.as_deref().unwrap_or("（无主体）");
        let missing = if self.missing.is_empty() {
            "无欠账".to_string()
        } else {
            self.missing
                .iter()
                .map(|obligation| obligation.label_zh())
                .collect::<Vec<_>>()
                .join("、")
        };
        format!(
            "回答回合：{}（主体：{}；第 {} 轮；{}；欠：{}）",
            self.schema.label_zh(),
            subject,
            self.turns,
            if self.is_closed() { "已闭合" } else { "未闭合" },
            missing
        )
    }
}

fn plan_has_gap(plan: &[String]) -> bool {
    plan.iter().any(|entry| {
        let label = entry.strip_prefix("board:").unwrap_or(entry);
        matches!(
            label,
            "admit_ignorance"
                | "why_no_evidence"
                | "clarify_entity"
                | "clarify_entity_no_focus"
                | "rank_pending"
        )
    })
}

fn plan_has_clarification(plan: &[String]) -> bool {
    plan.iter().any(|entry| {
        let label = entry.strip_prefix("board:").unwrap_or(entry);
        label.contains("clarify") || label == "rank_pending"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation<'a>(
        question: &'a str,
        kind: &'a MoveKind,
        subject: Option<&'a str>,
        answer: Option<&'a str>,
        evidence: &'a [String],
        plan: &'a [String],
    ) -> EpisodeObservation<'a> {
        EpisodeObservation {
            question,
            kind,
            subject,
            predicate: None,
            plan,
            answer,
            evidence,
            revision: 1,
        }
    }

    #[test]
    fn every_supported_query_type_maps_to_one_schema() {
        let issues = lint_query_types(crate::config_contract::supported_query_types());
        assert!(issues.is_empty(), "{:?}", issues);
        for query_type in crate::config_contract::supported_query_types() {
            assert!(
                classify_query_type(query_type).is_some(),
                "query_type「{}」缺少回答图式",
                query_type
            );
        }
    }

    #[test]
    fn unmapped_query_type_is_reported() {
        let issues = lint_query_types(["age_query", "unknown_query"]);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, "unmapped_query_type");
        assert_eq!(issues[0].query_type, "unknown_query");
    }

    #[test]
    fn followups_stay_in_the_same_episode() {
        let direct = MoveKind::AskAttribute;
        let plan = vec!["direct_answer".to_string()];
        let evidence = vec!["图谱 姚明 身高 226".to_string()];
        let first = AnswerEpisode::observe(
            &observation(
                "姚明的身高是多少",
                &direct,
                Some("姚明"),
                Some("226 厘米"),
                &evidence,
                &plan,
            ),
            None,
        );
        assert_eq!(first.schema, AnswerSchema::Attribute);
        assert!(first.is_closed());

        let continuation = MoveKind::ContinueRequest;
        let second = AnswerEpisode::observe(
            &observation("还有呢", &continuation, Some("姚明"), None, &[], &plan),
            Some(&first),
        );
        assert_eq!(second.schema, AnswerSchema::Attribute);
        assert_eq!(second.subject.as_deref(), Some("姚明"));
        assert_eq!(second.turns, 2);
        assert_eq!(second.missing, vec![Obligation::Answer, Obligation::Evidence]);
    }

    #[test]
    fn explicit_new_topic_starts_a_new_episode() {
        let ask = MoveKind::AskAttribute;
        let plan = vec!["direct_answer".to_string()];
        let evidence = vec!["图谱 姚明 身高 226".to_string()];
        let first = AnswerEpisode::observe(
            &observation(
                "姚明的身高是多少",
                &ask,
                Some("姚明"),
                Some("226 厘米"),
                &evidence,
                &plan,
            ),
            None,
        );
        let second = AnswerEpisode::observe(
            &observation(
                "叶莉的身高是多少",
                &ask,
                Some("叶莉"),
                Some("190 厘米"),
                &evidence,
                &plan,
            ),
            Some(&first),
        );
        assert_eq!(second.schema, AnswerSchema::Attribute);
        assert_eq!(second.subject.as_deref(), Some("叶莉"));
        assert_eq!(second.turns, 1);
        assert!(second.is_closed());
    }

    #[test]
    fn why_followup_adds_an_evidence_debt() {
        let ask = MoveKind::AskAttribute;
        let plan = vec!["direct_answer".to_string()];
        let evidence = vec!["图谱 姚明 身高 226".to_string()];
        let first = AnswerEpisode::observe(
            &observation(
                "姚明的身高是多少",
                &ask,
                Some("姚明"),
                Some("226 厘米"),
                &evidence,
                &plan,
            ),
            None,
        );
        let why = MoveKind::MetaWhy;
        let second = AnswerEpisode::observe(
            &observation("为什么？", &why, Some("姚明"), Some("依据是登记事实"), &[], &plan),
            Some(&first),
        );
        assert_eq!(second.schema, AnswerSchema::Attribute);
        assert!(second.extra.contains(&Obligation::Evidence));
        assert!(second.missing.contains(&Obligation::Evidence));
    }

    #[test]
    fn unknown_questions_owe_a_clarification() {
        let other = MoveKind::Other;
        let episode = AnswerEpisode::observe(
            &observation("嗯嗯", &other, None, None, &[], &[]),
            None,
        );
        assert_eq!(episode.schema, AnswerSchema::Unknown);
        assert_eq!(episode.missing, vec![Obligation::Clarification]);
        assert!(!episode.is_closed());
    }

    #[test]
    fn gap_answers_do_not_fill_the_answer_slot() {
        let ask = MoveKind::AskAttribute;
        let plan = vec!["admit_ignorance".to_string()];
        let episode = AnswerEpisode::observe(
            &observation(
                "姚明的体重是多少",
                &ask,
                Some("姚明"),
                Some("知识图谱里没有记录"),
                &[],
                &plan,
            ),
            None,
        );
        assert_eq!(episode.schema, AnswerSchema::Attribute);
        assert!(episode.missing.contains(&Obligation::Answer));
        assert!(episode.missing.contains(&Obligation::Evidence));
    }

    #[test]
    fn table_and_episode_projection_are_deterministic() {
        let table = render_table();
        assert_eq!(table, render_table());
        assert!(table.contains("attribute"));
        assert!(table.contains("age_query"));

        let ask = MoveKind::AskAttribute;
        let plan = vec!["direct_answer".to_string()];
        let evidence = vec!["图谱 姚明 身高 226".to_string()];
        let observation = observation(
            "姚明的身高是多少",
            &ask,
            Some("姚明"),
            Some("226 厘米"),
            &evidence,
            &plan,
        );
        assert_eq!(
            AnswerEpisode::observe(&observation, None),
            AnswerEpisode::observe(&observation, None)
        );
    }
}

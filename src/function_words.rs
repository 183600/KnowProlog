//! 虚词算符表：功能词的唯一正本与投影快照。
//!
//! 在符号栈里，功能词不是「不携带信息」，而是语法经过的地方——
//! 的=属性访问、是=类型断言、被/由=边反向、
//! 把/将=话题前置、和/与=合取、不/没=否定（闭世界）、吗/呢=句类、
//! 谁/什么/哪里/几/多少=答案类型约束与聚合。它们驱动结构，不该被丢弃。
//!
//! 各模块的词面过滤名单以本模块为唯一正本，按「投影快照」分发：
//!
//! 1. `REGISTRY`：带角色标注的词条正本（一词一记录，可多重角色）；
//! 2. `Projection`：五个「投影快照」+ 一个片段投影，逐词固定，
//!    可对账（见 `tests` 中的投影校验）；
//! 3. 共享函数：`contains_question_marker`（含同形守卫）、
//!    `wh_kind` / `answer_type_keywords`（疑问词 → 答案类型期待）。
//!
//! 设计戒律：投影与正本同源；新角色先在正本登记，再让模块显式消费。

use std::collections::HashSet;
use std::sync::OnceLock;

/// 功能词在句法-语义中的角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// 的：属性访问 / 多跳连接 / 定语标记
    Attribute,
    /// 是：类型断言 / 定义 / 等同
    TypeAssertion,
    /// 被 / 由：关系边反向（受事主语）
    Passive,
    /// 把 / 将：话题（宾语）前置
    Topic,
    /// 和 / 与 / 、：合取查询 / 并列展开
    Conjunction,
    /// 不 / 没 / 没有 / 无 / 未：否定（闭世界判定的触发词）
    Negation,
    /// 吗 / 吧 / 是不是 / 是否 / 有没有：真值校验句类
    TruthCheck,
    /// 呢：追询 / 省略补全
    FollowUp,
    /// 啊 / 呀：纯语气
    Mood,
    /// 谁 / 何人 / 哪位：人类型答案期待
    WhPerson,
    /// 哪里 / 哪儿 / 何处 / 什么地方：地点型答案期待
    WhPlace,
    /// 什么时候 / 哪年 / 何时 / 几时：时间型答案期待
    WhTime,
    /// 多少 / 几 / 几个：数量聚合
    WhQuantity,
    /// 什么 / 哪 / 哪个 / 哪些：未知实体槽
    WhThing,
    /// 为什么 / 怎么 / 怎样 / 如何：原因 / 方法
    WhReason,
    /// 个 / 位 / 名 / 门 / 项 / 条 / 件 / 座：量词（计数 + 分类词一致）
    Quantifier,
    /// 我 / 你 / 他 / 她 / 它 / 这 / 那 / 自己：回指消解
    Reference,
    /// 了 / 着 / 已经 / 曾经 …：体标记
    Aspect,
    /// 可能 / 应该 / 必须 / 需要 / 要 / 会 / 能：情态
    Modal,
    /// 请问 / 帮我 / 告诉 / 介绍 / 你好：请求 / 礼貌包装
    Request,
    /// 但是 / 因为 / 所以 / 首先 …：语篇连接
    Discourse,
    /// 说 / 看 / 去 / 来 / 到 / 有：泛化动词
    LightVerb,
    /// 觉得 / 认为 / 知道 / 希望 / 喜欢 …：心理与认知动词
    MentalVerb,
    /// 在 / 为 / 对：介词
    Preposition,
    /// 里 / 后 / 下：方位 / 处所
    Locative,
    /// 高频泛化词：不作命名实体，无句法角色断言
    Generic,
}

impl Role {
    /// 角色的中文标签（渲染审计视图用，保持稳定）。
    pub fn label(self) -> &'static str {
        match self {
            Role::Attribute => "属性访问",
            Role::TypeAssertion => "类型断言",
            Role::Passive => "边反向",
            Role::Topic => "话题前置",
            Role::Conjunction => "合取并列",
            Role::Negation => "否定",
            Role::TruthCheck => "真值校验",
            Role::FollowUp => "追询",
            Role::Mood => "语气",
            Role::WhPerson => "答案期待:人",
            Role::WhPlace => "答案期待:地点",
            Role::WhTime => "答案期待:时间",
            Role::WhQuantity => "数量聚合",
            Role::WhThing => "未知实体槽",
            Role::WhReason => "原因方法",
            Role::Quantifier => "量词",
            Role::Reference => "回指指代",
            Role::Aspect => "体标记",
            Role::Modal => "情态",
            Role::Request => "请求包装",
            Role::Discourse => "语篇连接",
            Role::LightVerb => "泛化动词",
            Role::MentalVerb => "心理认知动词",
            Role::Preposition => "介词",
            Role::Locative => "方位处所",
            Role::Generic => "泛化词",
        }
    }
}

/// 词条正本中的一条记录：一词一记录，角色可多重。
#[derive(Debug, Clone, Copy)]
pub struct FunctionWord {
    pub text: &'static str,
    pub roles: &'static [Role],
}

/// 全部注册词条（唯一正本）。
///
/// 组织顺序：结构算符 → 句类 → 疑问词 → 量词 → 回指 → 体/情态 →
/// 请求 → 语篇 → 动词 → 介词/方位 → 泛化词。渲染输出按此顺序，可 golden。
pub static REGISTRY: &[FunctionWord] = &[
    // 结构算符
    FunctionWord {
        text: "的",
        roles: &[Role::Attribute],
    },
    FunctionWord {
        text: "是",
        roles: &[Role::TypeAssertion],
    },
    FunctionWord {
        text: "被",
        roles: &[Role::Passive],
    },
    FunctionWord {
        text: "由",
        roles: &[Role::Passive],
    },
    FunctionWord {
        text: "把",
        roles: &[Role::Topic],
    },
    FunctionWord {
        text: "将",
        roles: &[Role::Topic],
    },
    FunctionWord {
        text: "和",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "与",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "以及",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "或",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "或者",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "并且",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "而且",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "还是",
        roles: &[Role::Conjunction],
    },
    FunctionWord {
        text: "不",
        roles: &[Role::Negation],
    },
    FunctionWord {
        text: "没",
        roles: &[Role::Negation],
    },
    FunctionWord {
        text: "没有",
        roles: &[Role::Negation],
    },
    FunctionWord {
        text: "无",
        roles: &[Role::Negation],
    },
    FunctionWord {
        text: "未",
        roles: &[Role::Negation],
    },
    // 句类
    FunctionWord {
        text: "吗",
        roles: &[Role::TruthCheck],
    },
    FunctionWord {
        text: "吧",
        roles: &[Role::TruthCheck],
    },
    FunctionWord {
        text: "是不是",
        roles: &[Role::TruthCheck],
    },
    FunctionWord {
        text: "是否",
        roles: &[Role::TruthCheck],
    },
    FunctionWord {
        text: "有没有",
        roles: &[Role::TruthCheck],
    },
    FunctionWord {
        text: "呢",
        roles: &[Role::FollowUp],
    },
    FunctionWord {
        text: "啊",
        roles: &[Role::Mood],
    },
    FunctionWord {
        text: "呀",
        roles: &[Role::Mood],
    },
    // 疑问词
    FunctionWord {
        text: "谁",
        roles: &[Role::WhPerson],
    },
    FunctionWord {
        text: "何人",
        roles: &[Role::WhPerson],
    },
    FunctionWord {
        text: "哪位",
        roles: &[Role::WhPerson],
    },
    FunctionWord {
        text: "哪里",
        roles: &[Role::WhPlace],
    },
    FunctionWord {
        text: "哪儿",
        roles: &[Role::WhPlace],
    },
    FunctionWord {
        text: "何处",
        roles: &[Role::WhPlace],
    },
    FunctionWord {
        text: "什么地方",
        roles: &[Role::WhPlace],
    },
    FunctionWord {
        text: "什么时候",
        roles: &[Role::WhTime],
    },
    FunctionWord {
        text: "哪年",
        roles: &[Role::WhTime],
    },
    FunctionWord {
        text: "何时",
        roles: &[Role::WhTime],
    },
    FunctionWord {
        text: "几时",
        roles: &[Role::WhTime],
    },
    FunctionWord {
        text: "多少",
        roles: &[Role::WhQuantity],
    },
    FunctionWord {
        text: "几",
        roles: &[Role::WhQuantity],
    },
    FunctionWord {
        text: "几个",
        roles: &[Role::WhQuantity],
    },
    FunctionWord {
        text: "什么",
        roles: &[Role::WhThing],
    },
    FunctionWord {
        text: "哪",
        roles: &[Role::WhThing],
    },
    FunctionWord {
        text: "哪个",
        roles: &[Role::WhThing],
    },
    FunctionWord {
        text: "哪些",
        roles: &[Role::WhThing],
    },
    FunctionWord {
        text: "为什么",
        roles: &[Role::WhReason],
    },
    FunctionWord {
        text: "怎么",
        roles: &[Role::WhReason],
    },
    FunctionWord {
        text: "怎样",
        roles: &[Role::WhReason],
    },
    FunctionWord {
        text: "如何",
        roles: &[Role::WhReason],
    },
    // 量词
    FunctionWord {
        text: "个",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "位",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "名",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "门",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "项",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "条",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "件",
        roles: &[Role::Quantifier],
    },
    FunctionWord {
        text: "座",
        roles: &[Role::Quantifier],
    },
    // 回指指代
    FunctionWord {
        text: "我",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "你",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "他",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "她",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "它",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "们",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "自己",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "这",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "这个",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "那个",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "这些",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "那么",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "这么",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "这样",
        roles: &[Role::Reference],
    },
    FunctionWord {
        text: "那样",
        roles: &[Role::Reference],
    },
    // 体标记
    FunctionWord {
        text: "了",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "着",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "刚刚",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "正在",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "已经",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "曾经",
        roles: &[Role::Aspect],
    },
    FunctionWord {
        text: "将要",
        roles: &[Role::Aspect],
    },
    // 情态
    FunctionWord {
        text: "可能",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "应该",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "必须",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "需要",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "要",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "会",
        roles: &[Role::Modal],
    },
    FunctionWord {
        text: "能",
        roles: &[Role::Modal],
    },
    // 请求包装
    FunctionWord {
        text: "请问",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "帮我",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "告诉",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "告诉我",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "介绍",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "你好",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "一下",
        roles: &[Role::Request],
    },
    FunctionWord {
        text: "一下吗",
        roles: &[Role::Request, Role::TruthCheck],
    },
    // 语篇连接
    FunctionWord {
        text: "但是",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "只是",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "如果",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "因为",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "所以",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "然后",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "接着",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "最后",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "首先",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "其次",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "另外",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "此外",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "虽然",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "可是",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "然而",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "不过",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "只要",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "除非",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "无论",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "不管",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "即使",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "哪怕",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "只有",
        roles: &[Role::Discourse],
    },
    FunctionWord {
        text: "仅仅",
        roles: &[Role::Discourse],
    },
    // 动词
    FunctionWord {
        text: "说",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "看",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "去",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "来",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "到",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "有",
        roles: &[Role::LightVerb],
    },
    FunctionWord {
        text: "想要",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "希望",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "愿意",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "喜欢",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "讨厌",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "觉得",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "认为",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "以为",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "知道",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "了解",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "明白",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "理解",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "发现",
        roles: &[Role::MentalVerb],
    },
    FunctionWord {
        text: "意识",
        roles: &[Role::MentalVerb],
    },
    // 介词 / 方位
    FunctionWord {
        text: "在",
        roles: &[Role::Preposition],
    },
    FunctionWord {
        text: "为",
        roles: &[Role::Preposition],
    },
    FunctionWord {
        text: "对",
        roles: &[Role::Preposition],
    },
    FunctionWord {
        text: "里",
        roles: &[Role::Locative],
    },
    FunctionWord {
        text: "后",
        roles: &[Role::Locative],
    },
    FunctionWord {
        text: "下",
        roles: &[Role::Locative],
    },
    // 泛化词
    FunctionWord {
        text: "就",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "都",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "也",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "人",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "一",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "好",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "还",
        roles: &[Role::Generic],
    },
    FunctionWord {
        text: "才",
        roles: &[Role::Generic],
    },
];

/// 查询某词的角色；未登记返回空切片。
///
/// 线性扫描 ~150 条词表，对每次调用都是缓存友好的；
/// 热路径若需要集合语义请使用 `is_registered` 或投影切片。
pub fn roles_of(word: &str) -> &'static [Role] {
    REGISTRY
        .iter()
        .find(|entry| entry.text == word)
        .map(|entry| entry.roles)
        .unwrap_or(&[])
}

/// 词条是否已登记。
pub fn is_registered(word: &str) -> bool {
    REGISTRY.iter().any(|entry| entry.text == word)
}

/// 投影名：各模块词面名单在正本中的快照坐标。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Projection {
    /// dense_retrieval 的查询/文档停用词（34 词）
    Retrieval,
    /// question_analyzer 的关键词停用词（41 词，为抽取名单前缀）
    Question,
    /// semantic_engine 的词面过滤（51 词，为抽取名单前缀）
    Semantic,
    /// similarity_calculator 的词面过滤（105 词）
    Similarity,
    /// entity_relation_extractor 的抽取停用词（105 词）
    Extraction,
    /// entity_linker 的片段级停用词（14 词）
    LinkerSegment,
}

impl Projection {
    /// 投影的稳定名称（渲染与对账用）。
    pub fn name(self) -> &'static str {
        match self {
            Projection::Retrieval => "Retrieval",
            Projection::Question => "Question",
            Projection::Semantic => "Semantic",
            Projection::Similarity => "Similarity",
            Projection::Extraction => "Extraction",
            Projection::LinkerSegment => "LinkerSegment",
        }
    }

    /// 全部投影（渲染与遍历的确定顺序）。
    pub const ALL: [Projection; 6] = [
        Projection::Retrieval,
        Projection::Question,
        Projection::Semantic,
        Projection::Similarity,
        Projection::Extraction,
        Projection::LinkerSegment,
    ];
}

/// 稠密检索的停用词快照。
pub const RETRIEVAL_WORDS: [&str; 34] = [
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "也",
    "都",
    "就",
    "吗",
    "呢",
    "吧",
    "啊",
    "什么",
    "多少",
    "几",
    "哪",
    "哪个",
    "哪些",
    "谁",
    "怎么",
    "为什么",
    "如何",
    "请问",
    "帮我",
    "告诉",
    "我",
    "一下",
    "这个",
    "那个",
    "他",
    "她",
    "它",
    "们",
];

/// 问题分析的关键词停用词快照。
pub const QUESTION_WORDS: [&str; 41] = [
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "也", "到", "说", "要",
    "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "还", "来", "个", "对", "为",
    "把", "里", "后", "才", "能", "下", "这些", "但是", "如果", "只是",
];

/// 语义引擎的词面过滤快照。
pub const SEMANTIC_WORDS: [&str; 51] = [
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "也",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "他",
    "还",
    "来",
    "个",
    "对",
    "为",
    "把",
    "里",
    "后",
    "才",
    "能",
    "下",
    "这些",
    "但是",
    "如果",
    "只是",
    "什么",
    "怎么",
    "为什么",
    "哪里",
    "那么",
    "这么",
    "怎样",
    "多少",
    "几个",
    "如何",
];

/// 相似度计算的词面过滤快照。
pub const SIMILARITY_WORDS: [&str; 105] = [
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "也",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "他",
    "还",
    "来",
    "个",
    "对",
    "为",
    "把",
    "里",
    "后",
    "才",
    "能",
    "下",
    "这些",
    "但是",
    "如果",
    "只是",
    "什么",
    "怎么",
    "为什么",
    "哪里",
    "那么",
    "这么",
    "怎样",
    "多少",
    "几个",
    "如何",
    "这个",
    "那个",
    "这样",
    "那样",
    "因为",
    "所以",
    "然后",
    "接着",
    "最后",
    "首先",
    "其次",
    "另外",
    "此外",
    "而且",
    "并且",
    "或者",
    "还是",
    "虽然",
    "但是",
    "可是",
    "然而",
    "不过",
    "只要",
    "除非",
    "无论",
    "不管",
    "即使",
    "哪怕",
    "只有",
    "只是",
    "仅仅",
    "刚刚",
    "正在",
    "已经",
    "曾经",
    "将要",
    "可能",
    "应该",
    "必须",
    "需要",
    "想要",
    "希望",
    "愿意",
    "喜欢",
    "讨厌",
    "觉得",
    "认为",
    "以为",
    "知道",
    "了解",
    "明白",
    "理解",
    "发现",
    "意识",
];

/// 关系抽取的停用词快照。
pub const EXTRACTION_WORDS: [&str; 105] = SIMILARITY_WORDS;

/// 实体链接的片段级停用词快照。
pub const LINKER_SEGMENT_WORDS: [&str; 14] = [
    "什么",
    "多少",
    "哪里",
    "哪个",
    "哪些",
    "怎么",
    "如何",
    "为什么",
    "介绍",
    "一下",
    "告诉我",
    "请问",
    "你好",
    "一下吗",
];

/// 取得投影的逐词快照。
pub fn projection_words(projection: Projection) -> &'static [&'static str] {
    match projection {
        Projection::Retrieval => &RETRIEVAL_WORDS,
        Projection::Question => &QUESTION_WORDS,
        Projection::Semantic => &SEMANTIC_WORDS,
        Projection::Similarity => &SIMILARITY_WORDS,
        Projection::Extraction => &EXTRACTION_WORDS,
        Projection::LinkerSegment => &LINKER_SEGMENT_WORDS,
    }
}

/// 取得投影的集合视图（一次性构建并缓存，供逐词过滤消费）。
pub fn projection_set(projection: Projection) -> &'static HashSet<&'static str> {
    static CACHE: OnceLock<Vec<HashSet<&'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        Projection::ALL
            .iter()
            .map(|p| projection_words(*p).iter().copied().collect())
            .collect()
    });
    let index = Projection::ALL
        .iter()
        .position(|p| *p == projection)
        .expect("投影必在 ALL 中");
    &cache[index]
}

/// 多字疑问标记：不会与普通词冲突，直接子串匹配。
pub const QUESTION_MARKERS: [&str; 12] = [
    "谁",
    "什么",
    "怎么",
    "为什么",
    "如何",
    "多少",
    "何人",
    "何时",
    "何处",
    "是不是",
    "是否",
    "有没有",
];

/// 判断文本是否含疑问标记。
///
/// 多字标记直接子串匹配，单字标记带同形守卫：
/// - 几（几乎/几何/几内亚）、哪（哪吒/哪咤）不误报；
/// - 吗啡（吗）、呢子/呢绒/呢喃（呢）不误报；
/// - 「几何」仅在句末是疑问（价值几何），句首/句中作名词（几何是数学分支）。
pub fn contains_question_marker(text: &str) -> bool {
    if QUESTION_MARKERS.iter().any(|marker| text.contains(marker)) {
        return true;
    }
    for (pos, ch) in text.char_indices() {
        let rest = &text[pos + ch.len_utf8()..];
        match ch {
            '吗' => {
                if !rest.starts_with('啡') {
                    return true;
                }
            }
            '呢' => {
                if !(rest.starts_with('子') || rest.starts_with('绒') || rest.starts_with('喃'))
                {
                    return true;
                }
            }
            '几' => {
                if !(rest.starts_with('乎') || rest.starts_with('何') || rest.starts_with('内'))
                {
                    return true;
                }
            }
            '哪' => {
                if !(rest.starts_with('吒') || rest.starts_with('咤')) {
                    return true;
                }
            }
            _ => {}
        }
    }
    text.ends_with("几何")
}

/// 疑问词给出的答案类型期待（第一个命中的疑问词优先）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhKind {
    Person,
    Place,
    Time,
    Quantity,
    Thing,
    Reason,
}

/// 识别文本中的疑问词类型；返回 None 表示无类型期待。
pub fn wh_kind(text: &str) -> Option<WhKind> {
    if text.contains('谁') || text.contains("何人") || text.contains("哪位") {
        return Some(WhKind::Person);
    }
    if text.contains("哪里")
        || text.contains("哪儿")
        || text.contains("什么地方")
        || text.contains("何处")
    {
        return Some(WhKind::Place);
    }
    if text.contains("什么时候")
        || text.contains("哪年")
        || text.contains("何时")
        || text.contains("几时")
    {
        return Some(WhKind::Time);
    }
    if text.contains("多少") || text.contains('几') || text.contains("几个") {
        return Some(WhKind::Quantity);
    }
    if text.contains('哪') && !text.contains("哪吒") && !text.contains("哪咤") {
        return Some(WhKind::Thing);
    }
    if text.contains("什么") {
        return Some(WhKind::Thing);
    }
    if text.contains("为什么")
        || text.contains("怎么")
        || text.contains("怎样")
        || text.contains("如何")
    {
        return Some(WhKind::Reason);
    }
    None
}

/// 疑问词的「类型期待」→ 实体类型关键词：谁→人、哪里→地点、何时→时间。
///
/// 数量/实体槽/原因类不给类型期待（返回空表）。
pub fn answer_type_keywords(question: &str) -> Vec<&'static str> {
    match wh_kind(question) {
        Some(WhKind::Person) => vec!["人", "person"],
        Some(WhKind::Place) => vec!["地", "城市", "国家", "place", "location", "country", "city"],
        Some(WhKind::Time) => vec!["日期", "时间", "date", "time", "年"],
        _ => Vec::new(),
    }
}

/// 渲染虚词算符表审计视图（确定性：同版本同字节，可 golden）。
pub fn render_registry_report() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "虚词算符表（唯一正本）：{} 条词条\n",
        REGISTRY.len()
    ));
    for entry in REGISTRY {
        let roles = entry
            .roles
            .iter()
            .map(|role| role.label())
            .collect::<Vec<_>>()
            .join("、");
        out.push_str(&format!("  {:<6} [{}]\n", entry.text, roles));
    }
    out.push_str("投影快照（逐词固定）：\n");
    for projection in Projection::ALL {
        let words = projection_words(projection);
        out.push_str(&format!(
            "  {:<13} {:>3} 词：{}\n",
            projection.name(),
            words.len(),
            words.join(" ")
        ));
    }
    let unregistered: Vec<&str> = Projection::ALL
        .iter()
        .flat_map(|p| projection_words(*p).iter().copied())
        .filter(|word| !is_registered(word))
        .collect();
    out.push_str(&format!(
        "对账：{}；未登记词条 {} 个\n",
        if unregistered.is_empty() {
            "全部投影词条均已在正本登记".to_string()
        } else {
            format!("发现未登记词条 {}", unregistered.join(" "))
        },
        unregistered.len()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 投影快照：五个投影逐词固定（长度 + 前缀关系 + 内容）。
    #[test]
    fn projections_are_frozen() {
        assert_eq!(RETRIEVAL_WORDS.len(), 34);
        assert_eq!(
            RETRIEVAL_WORDS,
            [
                "的",
                "了",
                "是",
                "在",
                "和",
                "与",
                "也",
                "都",
                "就",
                "吗",
                "呢",
                "吧",
                "啊",
                "什么",
                "多少",
                "几",
                "哪",
                "哪个",
                "哪些",
                "谁",
                "怎么",
                "为什么",
                "如何",
                "请问",
                "帮我",
                "告诉",
                "我",
                "一下",
                "这个",
                "那个",
                "他",
                "她",
                "它",
                "们"
            ]
        );

        assert_eq!(QUESTION_WORDS.len(), 41);
        assert_eq!(&EXTRACTION_WORDS[..41], &QUESTION_WORDS);
        assert_eq!(QUESTION_WORDS.last(), Some(&"只是"));

        assert_eq!(SEMANTIC_WORDS.len(), 51);
        assert_eq!(&EXTRACTION_WORDS[..51], &SEMANTIC_WORDS);
        assert_eq!(SEMANTIC_WORDS[40], "只是");
        assert_eq!(SEMANTIC_WORDS[50], "如何");

        // 相似度与抽取共用同一份名单
        assert_eq!(SIMILARITY_WORDS, EXTRACTION_WORDS);
        assert_eq!(EXTRACTION_WORDS.len(), 105);
        // 「但是」「只是」各出现两次；集合消费不受影响，快照保留原样
        assert_eq!(EXTRACTION_WORDS.iter().filter(|w| **w == "但是").count(), 2);
        assert_eq!(EXTRACTION_WORDS.iter().filter(|w| **w == "只是").count(), 2);

        assert_eq!(LINKER_SEGMENT_WORDS.len(), 14);
        assert!(LINKER_SEGMENT_WORDS.contains(&"一下吗"));
        assert!(LINKER_SEGMENT_WORDS.contains(&"告诉我"));
    }

    /// 投影中的每个词都必须在正本登记；正本无重复词条。
    #[test]
    fn registry_covers_every_projection_word() {
        for projection in Projection::ALL {
            for word in projection_words(projection) {
                assert!(
                    is_registered(word),
                    "投影 {} 的词条「{}」未在正本登记",
                    projection.name(),
                    word
                );
                assert!(
                    !roles_of(word).is_empty(),
                    "词条「{}」至少应有一个角色",
                    word
                );
            }
        }
        let mut seen = HashSet::new();
        for entry in REGISTRY {
            assert!(
                seen.insert(entry.text),
                "正本出现重复词条「{}」",
                entry.text
            );
        }
    }

    /// 关键角色标注。
    #[test]
    fn roles_are_typed() {
        assert!(roles_of("的").contains(&Role::Attribute));
        assert!(roles_of("是").contains(&Role::TypeAssertion));
        assert!(roles_of("和").contains(&Role::Conjunction));
        assert!(roles_of("不").contains(&Role::Negation));
        assert!(roles_of("吗").contains(&Role::TruthCheck));
        assert!(roles_of("呢").contains(&Role::FollowUp));
        assert!(roles_of("谁").contains(&Role::WhPerson));
        assert!(roles_of("哪里").contains(&Role::WhPlace));
        assert!(roles_of("什么时候").contains(&Role::WhTime));
        assert!(roles_of("多少").contains(&Role::WhQuantity));
        assert!(roles_of("个").contains(&Role::Quantifier));
        assert!(roles_of("他").contains(&Role::Reference));
        assert!(roles_of("被").contains(&Role::Passive));
        assert!(roles_of("把").contains(&Role::Topic));
        assert!(roles_of("几乎").is_empty());
    }

    /// 疑问标记识别：疑问词全绿，同形词零误报。
    #[test]
    fn question_marker_union_and_guards() {
        for marker in [
            "他是不是老师",
            "这块表是否防水",
            "商店有没有可乐",
            "何人到此",
            "何时出发",
            "去何处",
            "你好吗",
            "呢",
            "价值几何",
        ] {
            assert!(contains_question_marker(marker), "「{}」应判为疑问", marker);
        }
        for statement in [
            "几乎所有人都到了",
            "几何是数学分支",
            "几内亚是非洲国家",
            "哪吒是神话人物",
            "吗啡是镇痛药",
            "呢子大衣很保暖",
            "不锈钢是合金",
            "他们不但来了还带了礼物",
            "这两件差不多",
        ] {
            assert!(
                !contains_question_marker(statement),
                "「{}」不应判为疑问",
                statement
            );
        }
    }

    /// 疑问词 → 答案类型期待。
    #[test]
    fn answer_type_expectation_keywords() {
        assert_eq!(
            answer_type_keywords("张三的师傅是谁？"),
            vec!["人", "person"]
        );
        assert_eq!(
            answer_type_keywords("姚明在哪里出生"),
            vec!["地", "城市", "国家", "place", "location", "country", "city"]
        );
        assert_eq!(
            answer_type_keywords("公司是什么时候成立的"),
            vec!["日期", "时间", "date", "time", "年"]
        );
        // 人优先于地点
        assert_eq!(answer_type_keywords("谁在哪里"), vec!["人", "person"]);
        // 数量/实体槽/原因不产生类型期待
        assert!(answer_type_keywords("有多少员工").is_empty());
        assert!(answer_type_keywords("什么是黑洞").is_empty());
        assert!(answer_type_keywords("为什么会下雨").is_empty());
    }

    /// 渲染审计视图确定性：两次渲染逐字节相同，且含投影对账行。
    #[test]
    fn registry_report_is_deterministic() {
        let first = render_registry_report();
        let second = render_registry_report();
        assert_eq!(first, second);
        assert!(first.contains("全部投影词条均已在正本登记"));
        assert!(first.contains("Retrieval"));
        assert!(first.contains("LinkerSegment"));
    }
}

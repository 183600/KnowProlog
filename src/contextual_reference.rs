//! CRD：语境指称消歧（Contextual Reference Disambiguation）。
//!
//! 把消歧做成**语境约束求解**：
//! 歧义不是解析失败，而是一个显式的、可求解的约束问题；求解结果不是一条
//! 答案，而是一条**有作用域、有支持度、可撤销、可审计的绑定**。约束不足以
//! 裁决时，系统用图谱事实生成一个「能分裂候选集」的区分性问题（符号化的
//! 20 questions），而不是把候选名字罗列一遍。
//!
//! 数据流（M → C → E → D → Q → A → B → G）：
//! 1. **M 提及**：复用 `entity_linker::link_lattice` 与全部候选，不重写打分；
//! 2. **C 语境素描**：为每个指称抽中文特征（量词、同位角色、关系名词、
//!    疑问词期待），为候选实体建 `SenseCard`（类型、出/入谓词、邻居锚点、度数）；
//! 3. **E 证据表决**：硬约束（类型不相容、人名护栏）先否决，其余候选进入
//!    整数证据表决（精确 +4、已确认别名 +3、类型相容 +2、谓词相容 +2……
//!    f64 分数只经 `score_bucket` 量化为整数）；
//! 4. **D 裁决**：五选一——`Bound`（命中活跃绑定）/ `Unique`（存活候选唯一且
//!    领先次优 ≥ 边际）/ `Ambiguous`（需澄清）/ `Description`（描述性指称）/
//!    `Nil`（零候选）。歧义是一等状态，而不是被静默压掉；
//! 5. **Q 追问**：从义项卡找「把候选两两分开」的特征生成问题（类型标签 /
//!    功能谓词的不同取值 / 邻居锚点），选项 ≤3，可答「不知道/别问了」；
//! 6. **A 回答行为**：`Choose / Restate / Correct / Unknown / Decline`；
//! 7. **B 绑定账本**：`ReferenceBinding{key, entity, scope, status, support,
//!    turn, evidence}`，作用域 `Turn/Topic/Session`，随快照持久化；
//! 8. **G 生成与学习**：回答先查绑定再查图谱；同一绑定第二次被独立确认 →
//!    提名为待确认别名（不自动确认、不写图谱）。
//!
//! 设计边界：本模块**只读**实体链接器的候选与身份台账的折叠函数，不新建
//! 词表、不重写打分、不写知识图谱。零权重、零训练。

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::entity_linker::{EntityLinker, LinkCandidate, LinkMethod, LinkedMention};
use crate::entity_registry::fold_surface;
use crate::identity::infer_suffix_type;

/// 领先次优多少票算「唯一」（默认边际）。低于边际即升级为显式歧义。
pub const DEFAULT_MARGIN_VOTES: i32 = 2;
/// 区分性问题最多给出的候选数。
pub const MAX_QUESTION_OPTIONS: usize = 3;

// 证据表决票值（整数，可解释、可复现）。
const VOTE_EXACT: i32 = 4;
const VOTE_ALIAS: i32 = 3;
const VOTE_FUZZY_STRONG: i32 = 1;
const VOTE_TYPE_MATCH: i32 = 2;
const VOTE_PREDICATE_MATCH: i32 = 2;
const VOTE_BINDING: i32 = 3;
const VOTE_FOCUS: i32 = 1;
const VOTE_ANCHOR: i32 = 1;
const VOTE_DEGREE: i32 = 1;

/// 量词 → 类型期待（「那位导演」→ 人物；「那家公司」→ 组织）。
const QUANTIFIER_TYPES: [(&str, &str); 12] = [
    ("那位", "人物"),
    ("这位", "人物"),
    ("哪位", "人物"),
    ("那家", "组织"),
    ("这家", "组织"),
    ("哪家", "组织"),
    ("那所", "组织"),
    ("这所", "组织"),
    ("那个", ""),
    ("这个", ""),
    ("哪个", ""),
    ("那座", "地点"),
];

/// 同位语角色词（「演员梁朝伟」）：默认给出「人物」类型期待。
const ROLE_WORDS: [&str; 18] = [
    "演员",
    "导演",
    "歌手",
    "作家",
    "诗人",
    "画家",
    "教授",
    "医生",
    "律师",
    "老师",
    "记者",
    "运动员",
    "主持人",
    "科学家",
    "企业家",
    "音乐家",
    "翻译家",
    "运动员",
];

/// 关系名词（「姚明的妻子」）：描述性指称 → 路径查询。
const RELATION_NOUNS: [&str; 40] = [
    "妻子",
    "丈夫",
    "太太",
    "夫人",
    "配偶",
    "伴侣",
    "儿子",
    "女儿",
    "父亲",
    "母亲",
    "爸爸",
    "妈妈",
    "哥哥",
    "弟弟",
    "姐姐",
    "妹妹",
    "祖父",
    "祖母",
    "爷爷",
    "奶奶",
    "外公",
    "外婆",
    "岳父",
    "岳母",
    "师傅",
    "师父",
    "老师",
    "学生",
    "徒弟",
    "朋友",
    "同事",
    "搭档",
    "老板",
    "创始人",
    "董事长",
    "总裁",
    "总经理",
    "校长",
    "院长",
    "CEO",
];

/// 描述性指称的动词（「演《无间道》的演员」→ 出演）。
const DESCRIPTION_VERBS: [(&str, &str); 7] = [
    ("演", "出演"),
    ("导", "执导"),
    ("唱", "演唱"),
    ("写", "著作"),
    ("著", "著作"),
    ("编", "编剧"),
    ("拍", "拍摄"),
];

/// 裁决五态：`Bound / Unique / Ambiguous / Description / Nil`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 命中活跃绑定（语境记忆直接生效）
    Bound,
    /// 存活候选唯一且领先次优 ≥ 边际
    Unique,
    /// 候选分裂且领先不足：升级为显式歧义，需澄清
    Ambiguous,
    /// 描述性指称（关系名词 / 作品-角色），转路径查询
    Description,
    /// 零候选：留给新实体识别/识别流程，不静默造实体
    Nil,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Bound => "绑定生效",
            Verdict::Unique => "唯一裁决",
            Verdict::Ambiguous => "显式歧义",
            Verdict::Description => "描述性指称",
            Verdict::Nil => "无候选",
        }
    }
}

/// 指称素描：提及侧的中文特征（集合，不是向量）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionSketch {
    pub mention: String,
    pub start: usize,
    /// 量词词面（那位/这家/……）
    pub quantifier: Option<String>,
    /// 同位语角色词（「演员梁朝伟」里的「演员」）
    pub appositive_role: Option<String>,
    /// 从量词/同位语/角色词推出的类型期待
    pub expectation_types: Vec<String>,
    /// 紧跟的关系名词（「姚明的妻子」里的「妻子」）
    pub relational_relation: Option<String>,
    /// 剥掉量词/同位语前缀后的有效指称面（绑定键与改写用）
    pub key_surface: String,
    /// 有效指称面相对 `mention.start` 的字符偏移
    pub key_offset: usize,
    /// 是否描述性指称
    pub is_description: bool,
}

/// 义项卡：实体侧可比较的集合画像。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SenseCard {
    pub entity: String,
    pub etype: Option<String>,
    pub out_predicates: Vec<String>,
    pub in_predicates: Vec<String>,
    /// 邻居锚点（排序、截断）
    pub anchors: Vec<String>,
    pub degree: usize,
}

impl SenseCard {
    pub fn all_predicates(&self) -> Vec<String> {
        let mut out = self.out_predicates.clone();
        out.extend(self.in_predicates.iter().cloned());
        out.sort();
        out.dedup();
        out
    }

    pub fn has_predicate(&self, predicate: &str) -> bool {
        self.out_predicates.iter().any(|p| p == predicate)
            || self.in_predicates.iter().any(|p| p == predicate)
    }
}

/// 义项来源：消歧只读知识源，不重写链接打分。
pub trait SenseSource {
    /// 实体义项卡；未登记实体返回 None。
    fn sense_card(&self, entity: &str) -> Option<SenseCard>;
    /// 实体是否在本体/图谱中登记。缺省由义项卡判定。
    fn known_entity(&self, entity: &str) -> bool {
        self.sense_card(entity).is_some()
    }
    /// 出向关系取值（候选区分特征用；缺省为空）。
    fn predicate_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        let _ = (entity, predicate);
        Vec::new()
    }
    /// 入向关系取值（缺省为空）。
    fn predicate_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        let _ = (entity, predicate);
        Vec::new()
    }
}

/// 单个候选的表决记录（整数票 + 硬约束结论）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateVote {
    pub entity: String,
    pub method: LinkMethod,
    pub votes: i32,
    /// 硬约束否决理由（`Some` 表示不出现在存活候选里）
    pub hard_rejected: Option<String>,
    pub etype: Option<String>,
    pub evidence: Vec<String>,
}

impl CandidateVote {
    pub fn survived(&self) -> bool {
        self.hard_rejected.is_none()
    }
}

/// 一个指称的完整裁决。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionResolution {
    pub sketch: MentionSketch,
    pub key: String,
    pub verdict: Verdict,
    pub chosen: Option<String>,
    pub candidates: Vec<CandidateVote>,
    pub question: Option<ClarifyQuestion>,
    pub notes: Vec<String>,
}

/// 区分性特征：问题由「分裂候选集」的需要决定，不是固定话术。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscriminatingFeature {
    /// 类型标签能分开候选（水果 vs 公司）
    Type,
    /// 谓词在候选间存在/取值不同
    Predicate { predicate: String },
    /// 邻居锚点不同
    Anchor { entity: String },
    /// 问不出特征时的退化：罗列名字（仍 ≤3 项）
    Names,
}

/// 澄清选项：候选实体 + 展示标签。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClarifyOption {
    pub entity: String,
    pub label: String,
    pub etype: Option<String>,
}

/// 区分性问题（选项 ≤3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClarifyQuestion {
    pub mention: String,
    pub feature: DiscriminatingFeature,
    pub options: Vec<ClarifyOption>,
}

impl ClarifyQuestion {
    /// 确定性渲染（同版本同字节，可 golden）。
    pub fn render(&self) -> String {
        let mut out = format!("你说的「{}」是指哪一个？\n", self.mention);
        for (index, option) in self.options.iter().enumerate() {
            let _ = writeln!(out, "  {}. {}", index + 1, option.label);
        }
        out.push_str("（可回答序号、名字，或说「不是」「不知道」「别问了」。）");
        out
    }
}

/// 澄清回答行为（A：回答行为）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClarifyAct {
    /// 选定：序号或名字命中某个选项
    Choose { index: usize, entity: String },
    /// 重述：给出一个选项外/选项内的名字（调用方决定是否可绑定）
    Restate { entity: String },
    /// 纠错：「不是 A，是 B」
    Correct { from: String, to: String },
    /// 不知道
    Unknown,
    /// 明确不追问
    Decline,
}

/// 待澄清请求：原始输入 + 指称跨度 + 问题。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClarifyRequest {
    pub original_input: String,
    pub mention: String,
    pub start: usize,
    pub key: String,
    pub question: ClarifyQuestion,
}

/// 描述性指称 → 路径查询。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptionKind {
    /// 「姚明的妻子」→ 头实体 + 关系名词
    Relational,
    /// 「演《无间道》的演员」→ 作品 + 角色
    ByTitle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptionQuery {
    pub kind: DescriptionKind,
    pub head: Option<String>,
    pub relation: Option<String>,
    pub role: Option<String>,
    pub title: Option<String>,
    pub rewritten: Option<String>,
    pub note: String,
}

/// 一次消歧的完整产物。
#[derive(Debug, Clone, Default)]
pub struct CrdOutcome {
    pub resolutions: Vec<MentionResolution>,
    pub descriptions: Vec<DescriptionQuery>,
    pub rewrite: Option<String>,
    pub ask: Option<ClarifyRequest>,
    pub notes: Vec<String>,
}

/// 绑定作用域：Turn（本回合）/ Topic（当前话题）/ Session（会话级）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingScope {
    Turn,
    Topic,
    Session,
}

impl BindingScope {
    pub fn label(self) -> &'static str {
        match self {
            BindingScope::Turn => "本回合",
            BindingScope::Topic => "当前话题",
            BindingScope::Session => "整个会话",
        }
    }
}

/// 绑定状态：活跃 / 被覆盖 / 已过期。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingStatus {
    Active,
    Overridden,
    Expired,
}

impl BindingStatus {
    pub fn label(self) -> &'static str {
        match self {
            BindingStatus::Active => "活跃",
            BindingStatus::Overridden => "已覆盖",
            BindingStatus::Expired => "已过期",
        }
    }
}

/// 一条指称绑定：有作用域、有支持度、可撤销、可审计。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceBinding {
    pub id: String,
    /// 折叠后的词面键
    pub key: String,
    /// 绑定到的规范实体
    pub entity: String,
    pub scope: BindingScope,
    pub status: BindingStatus,
    /// 支持度（含同回合重复）
    pub support: u32,
    /// 独立确认次数（不同回合才算第二次签字）
    pub confirmations: u32,
    /// 最近一次确认的回合号
    pub turn: u64,
    pub evidence: Vec<String>,
    /// 是否已向上提名过别名（每次提名只报一次）
    #[serde(default)]
    pub alias_nominated: bool,
}

impl ReferenceBinding {
    pub fn active(&self) -> bool {
        self.status == BindingStatus::Active
    }
}

/// 一次绑定的结果（回执式，回答可直接引用 id）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingConfirmation {
    pub binding: ReferenceBinding,
    /// 这是不是第二次独立确认（可提名别名）
    pub second_signature: bool,
}

/// 别名提名：绑定账本向身份台账的提议（不自动确认、不写图谱）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasNomination {
    pub surface: String,
    pub canonical: String,
    pub confirmations: u32,
    pub evidence: String,
}

/// 绑定账本：会话记忆的显式承载（随快照持久化）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BindingLedger {
    #[serde(default)]
    turn: u64,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    bindings: Vec<ReferenceBinding>,
}

impl BindingLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// 推进一回合：Turn 作用域的活跃绑定过期。
    pub fn begin_turn(&mut self) {
        self.turn += 1;
        for binding in self.bindings.iter_mut() {
            if binding.active() && binding.scope == BindingScope::Turn {
                binding.status = BindingStatus::Expired;
            }
        }
    }

    /// 话题切换：Topic 作用域的活跃绑定过期（会话级保留）。
    pub fn end_topic(&mut self) {
        for binding in self.bindings.iter_mut() {
            if binding.active() && binding.scope == BindingScope::Topic {
                binding.status = BindingStatus::Expired;
            }
        }
    }

    /// 查活跃绑定（键 = 折叠词面）。
    pub fn lookup(&self, key: &str) -> Option<&ReferenceBinding> {
        let folded = fold_surface(key);
        self.bindings
            .iter()
            .find(|binding| binding.active() && binding.key == folded)
    }

    /// 建立/追加一次绑定。
    ///
    /// - 同键同实体：同回合只加支持度；跨回合追加一次独立确认；
    /// - 同键异实体：旧绑定转「已覆盖」，新绑定从 1 次确认起算；
    /// - 返回回执，回答可显式引用 binding id。
    pub fn bind(
        &mut self,
        key: &str,
        entity: &str,
        scope: BindingScope,
        evidence: &str,
    ) -> Option<BindingConfirmation> {
        let folded = fold_surface(key);
        let entity = entity.trim();
        if folded.is_empty() || entity.is_empty() {
            return None;
        }
        if let Some(index) = self.bindings.iter().position(|binding| {
            binding.active() && binding.key == folded && binding.entity == entity
        }) {
            let binding = &mut self.bindings[index];
            let second_signature = binding.turn != self.turn;
            binding.support += 1;
            if second_signature {
                binding.confirmations += 1;
                binding.turn = self.turn;
            }
            if !evidence.is_empty() && !binding.evidence.iter().any(|e| e == evidence) {
                binding.evidence.push(evidence.to_string());
            }
            return Some(BindingConfirmation {
                binding: binding.clone(),
                second_signature,
            });
        }
        // 同键异实体：覆盖（旧的转 Overridden，留痕不删除）
        for binding in self.bindings.iter_mut() {
            if binding.active() && binding.key == folded && binding.entity != entity {
                binding.status = BindingStatus::Overridden;
            }
        }
        self.next_id += 1;
        let binding = ReferenceBinding {
            id: format!("rb-{}-{}", self.turn.max(1), self.next_id),
            key: folded,
            entity: entity.to_string(),
            scope,
            status: BindingStatus::Active,
            support: 1,
            confirmations: 1,
            turn: self.turn,
            evidence: if evidence.is_empty() {
                Vec::new()
            } else {
                vec![evidence.to_string()]
            },
            alias_nominated: false,
        };
        self.bindings.push(binding.clone());
        Some(BindingConfirmation {
            binding,
            second_signature: false,
        })
    }

    /// 纠错：把词面改绑到另一个实体（旧的转 Overridden）。
    pub fn correct(
        &mut self,
        key: &str,
        entity: &str,
        evidence: &str,
    ) -> Option<BindingConfirmation> {
        let folded = fold_surface(key);
        for binding in self.bindings.iter_mut() {
            if binding.active() && binding.key == folded && binding.entity != entity {
                binding.status = BindingStatus::Overridden;
            }
        }
        let mut confirmation = self.bind(key, entity, BindingScope::Topic, evidence)?;
        // 纠错本身是一次新的独立确认，不应计为「第二次签字」的自动提名
        confirmation.second_signature = false;
        confirmation.binding.confirmations = 1;
        confirmation.binding.turn = self.turn;
        if let Some(binding) = self.bindings.last_mut() {
            binding.confirmations = 1;
            binding.turn = self.turn;
        }
        Some(confirmation)
    }

    /// 撤销一条活跃绑定（用户显式否定）。
    pub fn forget(&mut self, key: &str) -> bool {
        let folded = fold_surface(key);
        let mut changed = false;
        for binding in self.bindings.iter_mut() {
            if binding.active() && binding.key == folded {
                binding.status = BindingStatus::Expired;
                changed = true;
            }
        }
        changed
    }

    /// 活跃绑定（按 id 稳定排序）。
    pub fn active_bindings(&self) -> Vec<&ReferenceBinding> {
        let mut out: Vec<&ReferenceBinding> = self
            .bindings
            .iter()
            .filter(|binding| binding.active())
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// 别名提名：同键跨回合确认 ≥2 次且词面 ≠ 规范名；每次提名只报一次。
    pub fn take_alias_nominations(&mut self) -> Vec<AliasNomination> {
        let mut out = Vec::new();
        for binding in self.bindings.iter_mut() {
            if !binding.active()
                || binding.alias_nominated
                || binding.confirmations < 2
                || binding.key == fold_surface(&binding.entity)
            {
                continue;
            }
            binding.alias_nominated = true;
            out.push(AliasNomination {
                surface: binding.key.clone(),
                canonical: binding.entity.clone(),
                confirmations: binding.confirmations,
                evidence: format!(
                    "语境指称绑定 {}：{} 个回合独立确认（最新第 {} 轮）",
                    binding.id, binding.confirmations, binding.turn
                ),
            });
        }
        out
    }

    /// 确定性审计视图。
    pub fn report(&self) -> String {
        let mut out = format!(
            "指称绑定账本：{} 条（活跃 {} 条，当前第 {} 回合）\n",
            self.bindings.len(),
            self.bindings.iter().filter(|b| b.active()).count(),
            self.turn
        );
        if self.bindings.is_empty() {
            out.push_str("  （空：发生一次澄清回答后，这里会记下「谁说哪个词指哪个实体」）");
            return out;
        }
        let mut bindings: Vec<&ReferenceBinding> = self.bindings.iter().collect();
        bindings.sort_by(|a, b| a.id.cmp(&b.id));
        for binding in bindings {
            let _ = writeln!(
                out,
                "  • {}「{}」→「{}」（{}，支持度 {}，确认 {} 次，第 {} 轮）",
                binding.status.label(),
                binding.key,
                binding.entity,
                binding.scope.label(),
                binding.support,
                binding.confirmations,
                binding.turn
            );
            for evidence in &binding.evidence {
                let _ = writeln!(out, "      - {}", evidence);
            }
        }
        out.trim_end().to_string()
    }

    pub fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn from_snapshot_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|error| error.to_string())
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        let text = self.snapshot_json().map_err(std::io::Error::other)?;
        std::fs::write(path, text)
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_snapshot_json(&text).ok()
    }
}

/// 语境指称消歧器（无状态；状态在 [`BindingLedger`]）。
#[derive(Debug, Clone)]
pub struct ContextualResolver {
    margin: i32,
    max_options: usize,
}

impl Default for ContextualResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextualResolver {
    pub fn new() -> Self {
        Self {
            margin: DEFAULT_MARGIN_VOTES,
            max_options: MAX_QUESTION_OPTIONS,
        }
    }

    pub fn with_margin(mut self, margin: i32) -> Self {
        self.margin = margin.max(1);
        self
    }

    pub fn margin(&self) -> i32 {
        self.margin
    }

    /// 完整裁决：提及 → 素描 → 表决 → 五态。
    pub fn resolve_text(
        &self,
        text: &str,
        linker: &EntityLinker,
        source: &dyn SenseSource,
        ledger: &BindingLedger,
        hints: &[String],
        focus: Option<&str>,
    ) -> CrdOutcome {
        let mentions = linker.link_lattice(text, hints);
        let descriptions = detect_descriptions(text, source);
        let global_expectations: Vec<String> = crate::function_words::answer_type_keywords(text)
            .into_iter()
            .map(str::to_string)
            .collect();
        let mut outcome = CrdOutcome {
            descriptions,
            ..CrdOutcome::default()
        };
        if mentions.is_empty() {
            self.attach_description_rewrite(&mut outcome);
            return outcome;
        }

        // 第一遍：不含锚点票，得到临时胜者（供第二遍的邻居锚点核对）。
        let provisional: Vec<Option<String>> = mentions
            .iter()
            .map(|mention| {
                self.evaluate(
                    mention,
                    linker,
                    text,
                    source,
                    ledger,
                    hints,
                    focus,
                    &[],
                    &global_expectations,
                )
                .chosen
            })
            .collect();
        let peers: Vec<String> = provisional.iter().flatten().cloned().collect();

        for mention in &mentions {
            let resolution = self.evaluate(
                mention,
                linker,
                text,
                source,
                ledger,
                hints,
                focus,
                &peers,
                &global_expectations,
            );
            if outcome.ask.is_none() && resolution.verdict == Verdict::Ambiguous {
                if let Some(question) = resolution.question.clone() {
                    outcome.ask = Some(ClarifyRequest {
                        original_input: text.to_string(),
                        mention: mention.mention.clone(),
                        start: mention.start,
                        key: resolution.key.clone(),
                        question,
                    });
                }
            }
            if resolution.verdict == Verdict::Bound || resolution.verdict == Verdict::Unique {
                if let Some(chosen) = &resolution.chosen {
                    if chosen != &mention.mention {
                        outcome.rewrite = Some(rewrite_at(
                            text,
                            mention.start,
                            mention.mention.chars().count(),
                            chosen,
                        ));
                    }
                }
            }
            outcome.notes.extend(resolution.notes.iter().cloned());
            outcome.resolutions.push(resolution);
        }
        if outcome.ask.is_none() {
            self.attach_description_rewrite(&mut outcome);
        }
        outcome
    }

    fn attach_description_rewrite(&self, outcome: &mut CrdOutcome) {
        if outcome.ask.is_some() {
            return;
        }
        for description in &outcome.descriptions {
            outcome.notes.push(description.note.clone());
            if outcome.rewrite.is_none() {
                if let Some(rewritten) = &description.rewritten {
                    outcome.rewrite = Some(rewritten.clone());
                }
            }
        }
    }

    /// 单个指称的裁决。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        mention: &LinkedMention,
        linker: &EntityLinker,
        text: &str,
        source: &dyn SenseSource,
        ledger: &BindingLedger,
        hints: &[String],
        focus: Option<&str>,
        peers: &[String],
        global_expectations: &[String],
    ) -> MentionResolution {
        let sketch = sketch_mention(text, mention);
        let key = fold_surface(&sketch.key_surface);
        let mut notes: Vec<String> = Vec::new();

        // 格解码可能把量词/同位语吞进模糊提及（「那位张三」），
        // 剥离前缀后按有效词面复核精确/别名候选。
        let mut candidates: Vec<LinkCandidate> = mention.candidates.clone();
        if sketch.key_offset > 0 && sketch.key_surface != mention.mention {
            for candidate in linker.surface_candidates(&sketch.key_surface) {
                if !candidates
                    .iter()
                    .any(|existing| existing.entity == candidate.entity)
                {
                    candidates.push(candidate);
                }
            }
        }

        // 0) 绑定优先：语境记忆直接生效（但仍做完整表决以出审计视图）。
        let binding = ledger.lookup(&key);
        let bound_entity = binding.and_then(|binding| {
            if source.known_entity(&binding.entity) {
                Some((binding.entity.clone(), binding.clone()))
            } else {
                None
            }
        });

        let mut expectations = sketch.expectation_types.clone();
        if let Some(relation) = &sketch.relational_relation {
            if let Some(keywords) = relation_expectation(relation) {
                expectations.extend(keywords.iter().map(|k| k.to_string()));
            }
        }
        expectations.sort();
        expectations.dedup();

        // 1) 证据表决 + 硬约束否决。
        let mut votes: Vec<CandidateVote> = candidates
            .iter()
            .map(|candidate| {
                self.vote(
                    candidate,
                    &sketch,
                    source,
                    ledger,
                    hints,
                    focus,
                    peers,
                    &expectations,
                    global_expectations,
                )
            })
            .collect();
        votes.sort_by(|a, b| b.votes.cmp(&a.votes).then_with(|| a.entity.cmp(&b.entity)));

        let survived: Vec<&CandidateVote> = votes.iter().filter(|vote| vote.survived()).collect();
        let question = self.plan_question(&sketch, &votes, source);

        if let Some((entity, binding)) = bound_entity {
            notes.push(format!(
                "「{}」按绑定账本（{}）理解为「{}」（支持度 {}，第 {} 轮确认{}）",
                mention.mention,
                binding.id,
                entity,
                binding.support,
                binding.turn,
                if binding.confirmations >= 2 {
                    format!("，已确认 {} 次", binding.confirmations)
                } else {
                    String::new()
                }
            ));
            return MentionResolution {
                sketch,
                key,
                verdict: Verdict::Bound,
                chosen: Some(entity),
                candidates: votes,
                question: None,
                notes,
            };
        }

        let (verdict, chosen, question) = match survived.len() {
            0 => (Verdict::Nil, None, None),
            1 => {
                let leader = survived[0];
                if leader.votes < 0 {
                    (Verdict::Nil, None, None)
                } else {
                    let contested = mention.candidates.len() > 1;
                    if contested {
                        notes.push(format!(
                            "「{}」语境证据唯一指向「{}」（{} 票）",
                            mention.mention, leader.entity, leader.votes
                        ));
                    }
                    (Verdict::Unique, Some(leader.entity.clone()), None)
                }
            }
            _ => {
                let leader = survived[0];
                let runner_up = survived[1];
                if leader.votes - runner_up.votes >= self.margin {
                    let contested = mention.candidates.len() > 1;
                    if contested {
                        notes.push(format!(
                            "「{}」语境证据指向「{}」（{} 票，领先次优 {} 票）",
                            mention.mention,
                            leader.entity,
                            leader.votes,
                            leader.votes - runner_up.votes
                        ));
                    }
                    (Verdict::Unique, Some(leader.entity.clone()), None)
                } else {
                    (Verdict::Ambiguous, None, question)
                }
            }
        };
        MentionResolution {
            sketch,
            key,
            verdict,
            chosen,
            candidates: votes,
            question,
            notes,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vote(
        &self,
        candidate: &LinkCandidate,
        sketch: &MentionSketch,
        source: &dyn SenseSource,
        ledger: &BindingLedger,
        hints: &[String],
        focus: Option<&str>,
        peers: &[String],
        expectations: &[String],
        global_expectations: &[String],
    ) -> CandidateVote {
        let mut votes = 0i32;
        let mut evidence: Vec<String> = Vec::new();
        let card = source.sense_card(&candidate.entity);
        let etype = card.as_ref().and_then(|card| card.etype.clone());
        let mut hard_rejected: Option<String> = None;

        // 链接方法票（f64 只经量化后进入整数表决）
        match candidate.method {
            LinkMethod::ExactName => {
                votes += VOTE_EXACT;
                evidence.push(format!("规范名精确匹配 +{}", VOTE_EXACT));
            }
            LinkMethod::Alias => {
                votes += VOTE_ALIAS;
                evidence.push(format!("别名匹配 +{}", VOTE_ALIAS));
            }
            LinkMethod::Fuzzy => {
                let bucket = if candidate.score >= 0.75 {
                    VOTE_FUZZY_STRONG
                } else if candidate.score >= 0.5 {
                    0
                } else {
                    -1
                };
                votes += bucket;
                evidence.push(format!(
                    "模糊匹配 {:.2} 量化 {:+}（容错仅补充证据）",
                    candidate.score, bucket
                ));
            }
        }

        // 硬约束 1：类型不相容一票否决（只由指称侧特征决定：量词/同位/关系名词/
        // 后缀；疑问词期待只做软票，避免「刘德桦的出生地是哪里」把人物主语否决）。
        if !expectations.is_empty() {
            if let Some(etype) = etype.as_deref() {
                if type_matches(etype, expectations) {
                    votes += VOTE_TYPE_MATCH;
                    evidence.push(format!("类型「{}」命中期待 +{}", etype, VOTE_TYPE_MATCH));
                } else {
                    hard_rejected = Some(format!(
                        "类型「{}」与语境期待「{}」不相容（硬约束否决）",
                        etype,
                        expectations.join("/")
                    ));
                }
            } else {
                evidence.push("类型未知：不阻断，也不加分".to_string());
            }
        }
        // 疑问词答案类型：软票（+1），只用于同分裁决倾向，不否决。
        if hard_rejected.is_none() && !global_expectations.is_empty() {
            if let Some(etype) = etype.as_deref() {
                if type_matches(etype, global_expectations) {
                    votes += 1;
                    evidence.push(format!("类型「{}」合乎疑问词答案期待 +1", etype));
                }
            }
        }

        // 硬约束 2：人名护栏——模糊匹配不得只靠单字相同把中国人名链错。
        if hard_rejected.is_none()
            && candidate.method == LinkMethod::Fuzzy
            && looks_like_person_name(&sketch.key_surface)
            && etype.as_deref().map(is_person_type).unwrap_or(false)
            && !shares_bigram(&sketch.key_surface, &candidate.entity)
        {
            hard_rejected = Some(format!(
                "人名护栏：模糊候选「{}」与「{}」无相邻二元组共享（仅单字近似不足以容错）",
                candidate.entity, sketch.key_surface
            ));
        }

        if hard_rejected.is_none() {
            // 绑定票（同键活跃绑定）
            if let Some(binding) = ledger.lookup(&fold_surface(&sketch.key_surface)) {
                if binding.entity == candidate.entity {
                    votes += VOTE_BINDING;
                    evidence.push(format!("活跃绑定 {} +{}", binding.id, VOTE_BINDING));
                }
            }
            // 谓词线索票
            if let Some(card) = &card {
                let hits: Vec<&String> = hints
                    .iter()
                    .filter(|hint| {
                        card.has_predicate(hint) || etype.as_deref() == Some(hint.as_str())
                    })
                    .collect();
                if !hits.is_empty() {
                    votes += VOTE_PREDICATE_MATCH;
                    evidence.push(format!(
                        "与问题线索「{}」相容 +{}",
                        hits.iter()
                            .map(|h| h.as_str())
                            .collect::<Vec<_>>()
                            .join("/"),
                        VOTE_PREDICATE_MATCH
                    ));
                }
                // 邻居锚点票：与同句其它指称的临时胜者存在知识关联
                let anchor_hits: Vec<&String> = peers
                    .iter()
                    .filter(|peer| *peer != &candidate.entity && card.anchors.contains(peer))
                    .collect();
                if !anchor_hits.is_empty() {
                    votes += VOTE_ANCHOR;
                    evidence.push(format!(
                        "与共现指称「{}」存在知识关联 +{}",
                        anchor_hits
                            .iter()
                            .map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join("/"),
                        VOTE_ANCHOR
                    ));
                }
                if card.degree >= 4 {
                    votes += VOTE_DEGREE;
                    evidence.push(format!(
                        "图谱度数 {}（常用实体）+{}",
                        card.degree, VOTE_DEGREE
                    ));
                }
            } else {
                evidence.push("实体未登记义项卡：仅按链接证据表决".to_string());
            }
            // 对话焦点先验（低权重，焦点不得压倒证据）
            if focus == Some(candidate.entity.as_str()) {
                votes += VOTE_FOCUS;
                evidence.push(format!("恰为当前对话焦点 +{}", VOTE_FOCUS));
            }
        }

        CandidateVote {
            entity: candidate.entity.clone(),
            method: candidate.method,
            votes,
            hard_rejected,
            etype,
            evidence,
        }
    }

    /// 区分性问题规划：先用类型，再用谓词取值，再用锚点，最后退化罗列名字。
    fn plan_question(
        &self,
        sketch: &MentionSketch,
        votes: &[CandidateVote],
        source: &dyn SenseSource,
    ) -> Option<ClarifyQuestion> {
        let alive: Vec<&CandidateVote> = {
            let mut alive: Vec<&CandidateVote> = votes.iter().filter(|v| v.survived()).collect();
            alive.sort_by(|a, b| b.votes.cmp(&a.votes).then_with(|| a.entity.cmp(&b.entity)));
            alive
        };
        if alive.len() < 2 {
            return None;
        }
        let options: Vec<&CandidateVote> = alive.iter().take(self.max_options).copied().collect();

        // 特征 1：类型标签能分开候选
        let distinct_types: Vec<Option<&str>> = {
            let mut types: Vec<Option<&str>> = Vec::new();
            for vote in &options {
                let t = vote.etype.as_deref();
                if !types.contains(&t) {
                    types.push(t);
                }
            }
            types
        };
        if distinct_types.len() >= 2 && distinct_types.iter().filter(|t| t.is_some()).count() >= 2 {
            let clarify_options: Vec<ClarifyOption> = options
                .iter()
                .map(|vote| ClarifyOption {
                    entity: vote.entity.clone(),
                    label: match &vote.etype {
                        Some(etype) if !vote.entity.contains(etype.as_str()) => {
                            format!("{}（{}）", vote.entity, etype)
                        }
                        _ => vote.entity.clone(),
                    },
                    etype: vote.etype.clone(),
                })
                .collect();
            return Some(ClarifyQuestion {
                mention: sketch.mention.clone(),
                feature: DiscriminatingFeature::Type,
                options: clarify_options,
            });
        }

        // 特征 2：某个谓词在候选间存在性/取值不同
        if let Some(predicate) = discriminating_predicate(&options, source) {
            let clarify_options: Vec<ClarifyOption> = options
                .iter()
                .map(|vote| {
                    let values = source.predicate_values(&vote.entity, &predicate);
                    let inverse = source.predicate_values_in(&vote.entity, &predicate);
                    let label = if !values.is_empty() {
                        format!("{}（{}={}）", vote.entity, predicate, values.join("/"))
                    } else if !inverse.is_empty() {
                        format!("{}（{}：{}）", vote.entity, predicate, inverse.join("/"))
                    } else if vote
                        .etype
                        .as_ref()
                        .map(|t| !vote.entity.contains(t.as_str()))
                        .unwrap_or(false)
                    {
                        format!(
                            "{}（{}）",
                            vote.entity,
                            vote.etype.clone().unwrap_or_default()
                        )
                    } else {
                        vote.entity.clone()
                    };
                    ClarifyOption {
                        entity: vote.entity.clone(),
                        label,
                        etype: vote.etype.clone(),
                    }
                })
                .collect();
            return Some(ClarifyQuestion {
                mention: sketch.mention.clone(),
                feature: DiscriminatingFeature::Predicate { predicate },
                options: clarify_options,
            });
        }

        // 特征 3：邻居锚点不同
        if let Some(anchor) = discriminating_anchor(&options, source) {
            return Some(ClarifyQuestion {
                mention: sketch.mention.clone(),
                feature: DiscriminatingFeature::Anchor { entity: anchor },
                options: options
                    .iter()
                    .map(|vote| ClarifyOption {
                        entity: vote.entity.clone(),
                        label: vote.entity.clone(),
                        etype: vote.etype.clone(),
                    })
                    .collect(),
            });
        }

        // 退化：名字列举（仍 ≤3 项）
        Some(ClarifyQuestion {
            mention: sketch.mention.clone(),
            feature: DiscriminatingFeature::Names,
            options: options
                .iter()
                .map(|vote| ClarifyOption {
                    entity: vote.entity.clone(),
                    label: vote.entity.clone(),
                    etype: vote.etype.clone(),
                })
                .collect(),
        })
    }
}

/// 指称素描：从提及及其前后文抽中文特征。
pub fn sketch_mention(text: &str, mention: &LinkedMention) -> MentionSketch {
    let chars: Vec<char> = text.chars().collect();
    let start = mention.start.min(chars.len());
    let mut expectation_types: Vec<String> = Vec::new();
    let mut key_offset = 0usize;

    // 量词（「那位」「这家」）：先看紧邻指称前的 2 字，再看指称内部前缀
    let mut quantifier = None;
    if start >= 2 {
        let window: String = chars[start - 2..start].iter().collect();
        if let Some((word, etype)) = QUANTIFIER_TYPES
            .iter()
            .filter(|(word, _)| window == *word)
            .max_by_key(|(word, _)| word.chars().count())
        {
            quantifier = Some((*word).to_string());
            if !etype.is_empty() {
                expectation_types.push((*etype).to_string());
            }
        }
    }
    if quantifier.is_none() {
        if let Some((word, etype)) = QUANTIFIER_TYPES
            .iter()
            .filter(|(word, _)| mention.mention.starts_with(*word))
            .max_by_key(|(word, _)| word.chars().count())
        {
            quantifier = Some((*word).to_string());
            key_offset += word.chars().count();
            if !etype.is_empty() {
                expectation_types.push((*etype).to_string());
            }
        }
    }

    // 同位语角色（「演员梁朝伟」）：紧邻指称前最长角色词，或指称内部前缀
    let mut appositive_role = None;
    for role in ROLE_WORDS {
        let len = role.chars().count();
        if start >= len {
            let window: String = chars[start - len..start].iter().collect();
            if window == role {
                appositive_role = Some(role.to_string());
                expectation_types.push("人物".to_string());
                break;
            }
        }
    }
    if appositive_role.is_none() {
        let stripped: String = mention.mention.chars().skip(key_offset).collect();
        for role in ROLE_WORDS {
            if stripped.starts_with(role) {
                appositive_role = Some(role.to_string());
                key_offset += role.chars().count();
                expectation_types.push("人物".to_string());
                break;
            }
        }
    }

    let key_surface: String = mention.mention.chars().skip(key_offset).collect();
    if let Some((_, etype)) = infer_suffix_type(&key_surface) {
        expectation_types.push(etype.to_string());
    }

    // 关系名词（「姚明的妻子」）：紧邻指称后的「的」+ 关系词
    let after_index = start + mention.mention.chars().count();
    let mut relational_relation = None;
    if chars.get(after_index) == Some(&'的') {
        let tail: String = chars[after_index + 1..].iter().collect();
        if let Some(relation) = RELATION_NOUNS
            .iter()
            .filter(|relation| tail.starts_with(*relation))
            .max_by_key(|relation| relation.chars().count())
        {
            relational_relation = Some((*relation).to_string());
        }
    }

    expectation_types.sort();
    expectation_types.dedup();
    let is_description = relational_relation.is_some();
    MentionSketch {
        mention: mention.mention.clone(),
        start,
        quantifier,
        appositive_role,
        expectation_types,
        relational_relation,
        key_surface,
        key_offset,
        is_description,
    }
}

/// 描述性指称识别：关系名词（「姚明的妻子」）与作品-角色（「演《无间道》的演员」）。
pub fn detect_descriptions(text: &str, source: &dyn SenseSource) -> Vec<DescriptionQuery> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<DescriptionQuery> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();

    // 1) 关系名词：头实体（≤4 字的已登记实体）+ 的 + 关系词
    for relation in RELATION_NOUNS {
        let relation_chars: Vec<char> = relation.chars().collect();
        let rel_len = relation_chars.len();
        if rel_len == 0 || chars.len() < rel_len {
            continue;
        }
        for index in 0..=chars.len() - rel_len {
            if chars[index..index + rel_len] != relation_chars[..] {
                continue;
            }
            if index == 0 || chars[index - 1] != '的' {
                continue;
            }
            let de_index = index - 1;
            let mut head: Option<String> = None;
            for head_len in (1..=4.min(de_index)).rev() {
                let candidate: String = chars[de_index - head_len..de_index].iter().collect();
                if candidate.chars().all(is_han_char) && source.known_entity(&candidate) {
                    head = Some(candidate);
                    break;
                }
            }
            let Some(head) = head else {
                continue;
            };
            if seen.iter().any(|(h, r)| h == &head && r == relation) {
                continue;
            }
            seen.push((head.clone(), (*relation).to_string()));
            let rewritten = format!("{}的{}是谁", head, relation);
            out.push(DescriptionQuery {
                kind: DescriptionKind::Relational,
                head: Some(head.clone()),
                relation: Some((*relation).to_string()),
                role: None,
                title: None,
                rewritten: Some(rewritten.clone()),
                note: format!(
                    "描述性指称（关系名词）：「{}的{}」→ 路径查询「{}」",
                    head, relation, rewritten
                ),
            });
        }
    }

    // 2) 作品-角色：动词 + 《作品》 + 的 + 角色词
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '《' {
            index += 1;
            continue;
        }
        let Some(close_offset) = chars[index + 1..].iter().position(|ch| *ch == '》') else {
            index += 1;
            continue;
        };
        let close = index + 1 + close_offset;
        let title: String = chars[index + 1..close].iter().collect();
        // 角色词紧随「》的」之后
        if chars.get(close + 1) == Some(&'的') {
            let tail: String = chars[close + 2..].iter().collect();
            if let Some(role) = ROLE_WORDS
                .iter()
                .filter(|role| tail.starts_with(*role))
                .max_by_key(|role| role.chars().count())
            {
                let verb = if index > 0 {
                    DESCRIPTION_VERBS
                        .iter()
                        .find(|(verb, _)| {
                            chars.get(index - 1) == Some(&verb.chars().next().unwrap())
                        })
                        .map(|(_, relation)| (*relation).to_string())
                        .unwrap_or_else(|| "出演".to_string())
                } else {
                    "出演".to_string()
                };
                let rewritten = format!("{}的{}是谁", title, role);
                out.push(DescriptionQuery {
                    kind: DescriptionKind::ByTitle,
                    head: None,
                    relation: Some(verb),
                    role: Some((*role).to_string()),
                    title: Some(title.clone()),
                    rewritten: Some(rewritten.clone()),
                    note: format!(
                        "描述性指称（作品-角色）：「《{}》的{}」→ 路径查询「{}」",
                        title, role, rewritten
                    ),
                });
            }
        }
        index = close + 1;
    }

    out
}

/// 澄清回答解析：`Choose / Restate / Correct / Unknown / Decline`。
///
/// 返回 `None` 表示这不像对当前问题的回答（应放行给常规管线）。
pub fn parse_clarify_act(input: &str, question: &ClarifyQuestion) -> Option<ClarifyAct> {
    let trimmed = input
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    const DECLINE: [&str; 8] = [
        "别问了",
        "不问了",
        "算了",
        "取消",
        "跳过",
        "不用了",
        "没事",
        "换个话题",
    ];
    const UNKNOWN: [&str; 8] = [
        "不知道",
        "不清楚",
        "不确定",
        "都行",
        "随便",
        "无所谓",
        "你说呢",
        "都可能是",
    ];
    if DECLINE.iter().any(|marker| trimmed.contains(marker)) {
        return Some(ClarifyAct::Decline);
    }
    if UNKNOWN.iter().any(|marker| trimmed.contains(marker)) {
        return Some(ClarifyAct::Unknown);
    }
    // 新问题混入（「那苹果的CEO呢」）：不把它当成对澄清问题的一次回答。
    if trimmed.ends_with('呢')
        || ["什么", "为什么", "怎么", "谁", "哪", "多少"]
            .iter()
            .any(|marker| trimmed.contains(marker))
    {
        return None;
    }

    // 序号：1 / 第1个 / 一
    if let Some(index) = parse_option_index(trimmed, question.options.len()) {
        return Some(ClarifyAct::Choose {
            index,
            entity: question.options[index].entity.clone(),
        });
    }

    let mentioned: Vec<&ClarifyOption> = {
        let mut hits: Vec<(&ClarifyOption, usize)> = question
            .options
            .iter()
            .filter_map(|option| trimmed.find(&option.entity).map(|index| (option, index)))
            .collect();
        hits.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.0.entity.chars().count().cmp(&a.0.entity.chars().count()))
        });
        hits.into_iter().map(|(option, _)| option).collect()
    };
    let after_is = extract_after_is(trimmed);

    if trimmed.contains("不是") {
        let negated = mentioned.first();
        let affirmed = mentioned
            .last()
            .filter(|option| negated.map(|n| n.entity != option.entity).unwrap_or(true));
        if let (Some(negated), Some(affirmed)) = (negated, affirmed) {
            return Some(ClarifyAct::Correct {
                from: negated.entity.clone(),
                to: affirmed.entity.clone(),
            });
        }
        if let Some(negated) = negated {
            // 「不是 A」：只有两个选项时取另一个；否则需更多信息
            if question.options.len() == 2 {
                let other = question
                    .options
                    .iter()
                    .find(|option| option.entity != negated.entity)?;
                return Some(ClarifyAct::Choose {
                    index: question
                        .options
                        .iter()
                        .position(|option| option.entity == other.entity)?,
                    entity: other.entity.clone(),
                });
            }
            if let Some(entity) = after_is {
                return Some(ClarifyAct::Restate { entity });
            }
            return Some(ClarifyAct::Unknown);
        }
        if let Some(entity) = after_is {
            return Some(ClarifyAct::Restate { entity });
        }
        return None;
    }

    if let Some(option) = mentioned.first() {
        let index = question
            .options
            .iter()
            .position(|candidate| candidate.entity == option.entity)?;
        return Some(ClarifyAct::Choose {
            index,
            entity: option.entity.clone(),
        });
    }

    // 类型线索（「是公司」「水果」）：短回答里出现的唯一类型标签
    if trimmed.chars().count() <= 10 {
        let typed: Vec<&ClarifyOption> = question
            .options
            .iter()
            .filter(|option| {
                option
                    .etype
                    .as_deref()
                    .map(|etype| type_cue_hit(etype, trimmed))
                    .unwrap_or(false)
            })
            .collect();
        if typed.len() == 1 {
            let option = typed[0];
            let index = question
                .options
                .iter()
                .position(|candidate| candidate.entity == option.entity)?;
            return Some(ClarifyAct::Choose {
                index,
                entity: option.entity.clone(),
            });
        }
    }

    // 「就是那个演……」式重述
    if let Some(entity) = after_is {
        return Some(ClarifyAct::Restate { entity });
    }
    None
}

/// 按下标重写指称（跨字节安全；越界/不匹配时回退为首次出现替换）。
pub fn rewrite_at(text: &str, start: usize, mention_len: usize, replacement: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if mention_len == 0 || start + mention_len > chars.len() {
        return text.to_string();
    }
    let mut out: String = chars[..start].iter().collect();
    out.push_str(replacement);
    out.extend(chars[start + mention_len..].iter());
    out
}

/// 按词面重写首次出现（用于无法确定跨度的场合）。
pub fn rewrite_mention(text: &str, mention: &str, replacement: &str) -> String {
    match text.find(mention) {
        Some(byte_index) => {
            let start = text[..byte_index].chars().count();
            rewrite_at(text, start, mention.chars().count(), replacement)
        }
        None => text.to_string(),
    }
}

fn parse_option_index(trimmed: &str, len: usize) -> Option<usize> {
    let core = trimmed
        .trim_start_matches(['第', '选'])
        .trim_end_matches(['个', '项', '号', '是', '的', '。', '.'])
        .trim();
    if core.is_empty() {
        return None;
    }
    if core.chars().all(|ch| ch.is_ascii_digit()) {
        let value: usize = core.parse().ok()?;
        if value >= 1 && value <= len {
            return Some(value - 1);
        }
        return None;
    }
    let chinese = ["一", "二", "三", "四", "五"];
    chinese
        .iter()
        .position(|word| core == *word)
        .filter(|index| *index < len)
}

/// 类型标签的词面线索（用户说「是公司」也能命中「组织」）。
fn type_cue_hit(etype: &str, text: &str) -> bool {
    if text.contains(etype) {
        return true;
    }
    let cues: &[&str] = if etype.contains("组织") {
        &["公司", "企业", "机构", "集团", "组织"]
    } else if etype.contains("人物") || etype.to_lowercase().contains("person") {
        &["人物", "人"]
    } else if etype.contains("地点") {
        &["地点", "城市", "地方"]
    } else if etype.contains("时间") {
        &["时间", "日期", "时候"]
    } else {
        &[]
    };
    cues.iter().any(|cue| text.contains(cue))
}

fn extract_after_is(text: &str) -> Option<String> {
    let position = text.rfind('是')?;
    let tail: String = text[position + '是'.len_utf8()..].chars().collect();
    let tail = tail.split(['，', ',', '、']).next().unwrap_or("");
    let cleaned = tail
        .trim()
        .trim_start_matches('的')
        .trim_end_matches(['的', '啊', '吧', '呀', '呢', '哦', '了', '！', '!', '。'])
        .trim();
    if cleaned.is_empty() || cleaned.chars().count() > 12 {
        return None;
    }
    Some(cleaned.to_string())
}

fn discriminating_predicate(
    options: &[&CandidateVote],
    source: &dyn SenseSource,
) -> Option<String> {
    let mut predicates: Vec<String> = Vec::new();
    for vote in options {
        if let Some(card) = source.sense_card(&vote.entity) {
            for predicate in card.all_predicates() {
                if !predicates.contains(&predicate) {
                    predicates.push(predicate);
                }
            }
        }
    }
    predicates.sort();
    for predicate in predicates {
        let present: Vec<bool> = options
            .iter()
            .map(|vote| {
                source
                    .sense_card(&vote.entity)
                    .map(|card| card.has_predicate(&predicate))
                    .unwrap_or(false)
            })
            .collect();
        if present.iter().all(|p| *p) {
            // 都有该谓词：看取值是否不同
            let values: Vec<Vec<String>> = options
                .iter()
                .map(|vote| {
                    let mut out = source.predicate_values(&vote.entity, &predicate);
                    out.extend(source.predicate_values_in(&vote.entity, &predicate));
                    out.sort();
                    out.dedup();
                    out
                })
                .collect();
            let distinct: Vec<&Vec<String>> = {
                let mut distinct: Vec<&Vec<String>> = Vec::new();
                for value in &values {
                    if !value.is_empty() && !distinct.contains(&value) {
                        distinct.push(value);
                    }
                }
                distinct
            };
            if distinct.len() >= 2 {
                return Some(predicate);
            }
            continue;
        }
        if present.iter().any(|p| *p) {
            return Some(predicate);
        }
    }
    None
}

fn discriminating_anchor(options: &[&CandidateVote], source: &dyn SenseSource) -> Option<String> {
    let mut anchors: Vec<String> = Vec::new();
    for vote in options {
        if let Some(card) = source.sense_card(&vote.entity) {
            for anchor in card.anchors {
                if !anchors.contains(&anchor) {
                    anchors.push(anchor);
                }
            }
        }
    }
    anchors.sort();
    for anchor in anchors {
        let present: Vec<bool> = options
            .iter()
            .map(|vote| {
                source
                    .sense_card(&vote.entity)
                    .map(|card| card.anchors.contains(&anchor))
                    .unwrap_or(false)
            })
            .collect();
        if present.iter().any(|p| *p) && !present.iter().all(|p| *p) {
            return Some(anchor);
        }
    }
    None
}

fn relation_expectation(relation: &str) -> Option<&'static [&'static str]> {
    const PERSON: [&str; 2] = ["人", "person"];
    const ORG: [&str; 4] = ["组织", "公司", "机构", "organization"];
    match relation {
        "创始人" | "董事长" | "总裁" | "总经理" | "CEO" | "校长" | "院长" => {
            Some(&ORG)
        }
        _ => Some(&PERSON),
    }
}

fn type_matches(etype: &str, expectations: &[String]) -> bool {
    let lowered = etype.to_lowercase();
    expectations.iter().any(|keyword| {
        etype.contains(keyword.as_str()) || lowered.contains(&keyword.to_lowercase())
    })
}

fn is_person_type(etype: &str) -> bool {
    etype.contains('人') || etype.to_lowercase().contains("person")
}

fn looks_like_person_name(surface: &str) -> bool {
    let chars: Vec<char> = surface.chars().collect();
    (2..=4).contains(&chars.len())
        && chars.iter().all(|ch| is_han_char(*ch))
        && infer_suffix_type(surface).is_none()
}

fn shares_bigram(a: &str, b: &str) -> bool {
    let grams = |text: &str| -> Vec<(char, char)> {
        let chars: Vec<char> = text.chars().collect();
        chars
            .windows(2)
            .map(|window| (window[0], window[1]))
            .collect()
    };
    let a_grams = grams(a);
    let b_grams = grams(b);
    a_grams
        .iter()
        .any(|gram| b_grams.iter().any(|other| other == gram))
}

fn is_han_char(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct MapSenseSource {
        cards: HashMap<String, SenseCard>,
        values: HashMap<(String, String), Vec<String>>,
        values_in: HashMap<(String, String), Vec<String>>,
    }

    impl MapSenseSource {
        fn add(
            &mut self,
            entity: &str,
            etype: &str,
            out: &[&str],
            anchors: &[&str],
            degree: usize,
        ) {
            self.cards.insert(
                entity.to_string(),
                SenseCard {
                    entity: entity.to_string(),
                    etype: Some(etype.to_string()),
                    out_predicates: out.iter().map(|p| p.to_string()).collect(),
                    in_predicates: Vec::new(),
                    anchors: anchors.iter().map(|a| a.to_string()).collect(),
                    degree,
                },
            );
        }

        fn add_value(&mut self, entity: &str, predicate: &str, value: &str) {
            self.values
                .entry((entity.to_string(), predicate.to_string()))
                .or_default()
                .push(value.to_string());
        }
    }

    impl SenseSource for MapSenseSource {
        fn sense_card(&self, entity: &str) -> Option<SenseCard> {
            self.cards.get(entity).cloned()
        }

        fn predicate_values(&self, entity: &str, predicate: &str) -> Vec<String> {
            self.values
                .get(&(entity.to_string(), predicate.to_string()))
                .cloned()
                .unwrap_or_default()
        }

        fn predicate_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
            self.values_in
                .get(&(entity.to_string(), predicate.to_string()))
                .cloned()
                .unwrap_or_default()
        }
    }

    fn apple_linker() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果", Some("水果"));
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("苹果", "苹果公司");
        linker
    }

    fn resolutions(text: &str, linker: &EntityLinker, source: &dyn SenseSource) -> CrdOutcome {
        ContextualResolver::new().resolve_text(
            text,
            linker,
            source,
            &BindingLedger::new(),
            &[],
            None,
        )
    }

    #[test]
    fn ambiguous_mention_asks_type_discriminating_question() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &["颜色"], &[], 2);
        source.add("苹果公司", "组织", &["创始人"], &[], 3);
        let outcome = resolutions("苹果是什么", &linker, &source);
        let ask = outcome.ask.expect("应升级为显式歧义");
        assert_eq!(ask.mention, "苹果");
        assert_eq!(ask.question.options.len(), 2);
        assert!(matches!(ask.question.feature, DiscriminatingFeature::Type));
        let rendered = ask.question.render();
        assert!(rendered.contains("水果"));
        assert!(rendered.contains("组织") || rendered.contains("公司"));
    }

    #[test]
    fn relational_noun_constraint_resolves_executive_query() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &["颜色"], &[], 2);
        source.add("苹果公司", "组织", &["创始人"], &[], 3);
        let outcome = resolutions("苹果的CEO是谁", &linker, &source);
        assert!(outcome.ask.is_none(), "关系名词 CEO 期待组织，足以裁决");
        assert_eq!(outcome.resolutions[0].verdict, Verdict::Unique);
        assert_eq!(outcome.resolutions[0].chosen.as_deref(), Some("苹果公司"));
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司的CEO是谁"));
    }

    #[test]
    fn bound_mention_wins_without_asking() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &[], &[], 0);
        source.add("苹果公司", "组织", &[], &[], 0);
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        let confirmation = ledger
            .bind("苹果", "苹果公司", BindingScope::Topic, "第一轮澄清")
            .expect("绑定成功");
        assert_eq!(confirmation.binding.id, "rb-1-1");
        let outcome = ContextualResolver::new().resolve_text(
            "苹果的CEO是谁",
            &linker,
            &source,
            &ledger,
            &[],
            None,
        );
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司的CEO是谁"));
        assert!(outcome.notes.iter().any(|note| note.contains("rb-1-1")));
        let resolution = &outcome.resolutions[0];
        assert_eq!(resolution.verdict, Verdict::Bound);
        assert_eq!(resolution.chosen.as_deref(), Some("苹果公司"));
    }

    #[test]
    fn binding_expires_on_turn_and_topic_boundaries() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("它", "张三", BindingScope::Turn, "本回合");
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "话题");
        ledger.bind("小明", "小明", BindingScope::Session, "会话");
        ledger.begin_turn();
        assert!(ledger.lookup("它").is_none(), "Turn 绑定随回合过期");
        assert!(ledger.lookup("苹果").is_some());
        ledger.end_topic();
        assert!(ledger.lookup("苹果").is_none(), "Topic 绑定随话题过期");
        assert!(ledger.lookup("小明").is_some(), "Session 绑定保留");
    }

    #[test]
    fn override_marks_old_binding_and_correction_rebinds() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "第一轮");
        let confirmation = ledger
            .correct("苹果", "苹果果实", "用户纠错：不是苹果公司")
            .expect("改绑成功");
        assert_eq!(confirmation.binding.entity, "苹果果实");
        assert_eq!(
            ledger.lookup("苹果").map(|b| b.entity.as_str()),
            Some("苹果果实")
        );
        let report = ledger.report();
        assert!(report.contains("已覆盖"));
        assert!(report.contains("苹果公司"));
    }

    #[test]
    fn second_independent_confirmation_nominates_alias_once() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "第一轮");
        assert!(ledger.take_alias_nominations().is_empty());
        ledger.begin_turn();
        let confirmation = ledger
            .bind("苹果", "苹果公司", BindingScope::Topic, "第二轮")
            .expect("追加确认");
        assert!(confirmation.second_signature);
        let nominations = ledger.take_alias_nominations();
        assert_eq!(nominations.len(), 1);
        assert_eq!(nominations[0].surface, "苹果");
        assert_eq!(nominations[0].canonical, "苹果公司");
        assert_eq!(nominations[0].confirmations, 2);
        assert!(
            ledger.take_alias_nominations().is_empty(),
            "同一绑定只提名一次"
        );
    }

    #[test]
    fn ledger_snapshot_roundtrip() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Session, "快照");
        let json = ledger.snapshot_json().expect("序列化");
        let restored = BindingLedger::from_snapshot_json(&json).expect("反序列化");
        assert_eq!(restored.turn(), ledger.turn());
        assert_eq!(
            restored.lookup("苹果").map(|b| b.entity.as_str()),
            Some("苹果公司")
        );
    }

    #[test]
    fn type_mismatch_is_hard_rejected() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三", Some("人物"));
        linker.register_entity("张三公司", Some("组织"));
        linker.register_alias("张三", "张三公司");
        let mut source = MapSenseSource::default();
        source.add("张三", "人物", &[], &[], 0);
        source.add("张三公司", "组织", &[], &[], 0);
        let outcome = resolutions("那位张三是谁", &linker, &source);
        let resolution = &outcome.resolutions[0];
        let rejected = resolution
            .candidates
            .iter()
            .find(|vote| vote.entity == "张三公司")
            .expect("应有组织候选");
        assert!(rejected.hard_rejected.is_some());
        assert_eq!(resolution.verdict, Verdict::Unique);
        assert_eq!(resolution.chosen.as_deref(), Some("张三"));
    }

    #[test]
    fn person_guard_rejects_single_char_fuzzy_link() {
        let mut linker = EntityLinker::new();
        linker.register_entity("王大明", Some("人物"));
        let mut source = MapSenseSource::default();
        source.add("王大明", "人物", &[], &[], 0);
        let outcome = resolutions("王小明的生日是哪天", &linker, &source);
        if let Some(resolution) = outcome.resolutions.first() {
            assert_eq!(resolution.verdict, Verdict::Nil);
            let rejected = resolution
                .candidates
                .iter()
                .find(|vote| vote.entity == "王大明")
                .expect("模糊候选应进入表决视图");
            assert!(
                rejected
                    .hard_rejected
                    .as_deref()
                    .unwrap_or("")
                    .contains("人名护栏"),
                "人名护栏应否决仅单字近似的候选：{:?}",
                rejected.hard_rejected
            );
        } else {
            panic!("应至少有模糊提及进入视图");
        }
    }

    #[test]
    fn person_guard_allows_typo_with_shared_bigram() {
        let mut linker = EntityLinker::new();
        linker.register_entity("刘德华", Some("人物"));
        let mut source = MapSenseSource::default();
        source.add("刘德华", "人物", &[], &[], 0);
        let outcome = resolutions("刘德桦的出生地是哪里", &linker, &source);
        let resolution = outcome.resolutions.first().expect("应有模糊提及");
        assert_eq!(resolution.verdict, Verdict::Unique);
        assert_eq!(resolution.chosen.as_deref(), Some("刘德华"));
    }

    #[test]
    fn quantifier_and_appositive_add_type_expectation() {
        let mut linker = EntityLinker::new();
        linker.register_entity("梁朝伟", Some("人物"));
        linker.register_entity("梁朝伟公司", Some("组织"));
        linker.register_alias("梁朝伟", "梁朝伟公司");
        let mut source = MapSenseSource::default();
        source.add("梁朝伟", "人物", &[], &[], 0);
        source.add("梁朝伟公司", "组织", &[], &[], 0);
        let mention = &linker.link_lattice("演员梁朝伟是谁", &[])[0];
        let sketch = sketch_mention("演员梁朝伟是谁", mention);
        assert_eq!(sketch.appositive_role.as_deref(), Some("演员"));
        assert!(sketch.expectation_types.contains(&"人物".to_string()));
    }

    #[test]
    fn parse_clarify_act_variants() {
        let question = ClarifyQuestion {
            mention: "苹果".to_string(),
            feature: DiscriminatingFeature::Type,
            options: vec![
                ClarifyOption {
                    entity: "苹果".to_string(),
                    label: "苹果（水果）".to_string(),
                    etype: Some("水果".to_string()),
                },
                ClarifyOption {
                    entity: "苹果公司".to_string(),
                    label: "苹果公司（组织）".to_string(),
                    etype: Some("组织".to_string()),
                },
            ],
        };
        assert_eq!(
            parse_clarify_act("1", &question),
            Some(ClarifyAct::Choose {
                index: 0,
                entity: "苹果".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("第二个", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("苹果公司", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("是公司", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不是苹果，是苹果公司", &question),
            Some(ClarifyAct::Correct {
                from: "苹果".to_string(),
                to: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不是苹果", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不知道", &question),
            Some(ClarifyAct::Unknown)
        );
        assert_eq!(
            parse_clarify_act("别问了", &question),
            Some(ClarifyAct::Decline)
        );
        assert_eq!(parse_clarify_act("那苹果的CEO呢", &question), None);
    }

    #[test]
    fn description_relational_and_by_title() {
        let mut source = MapSenseSource::default();
        source.add("姚明", "人物", &[], &[], 0);
        let descriptions = detect_descriptions("姚明的妻子是谁", &source);
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].kind, DescriptionKind::Relational);
        assert_eq!(descriptions[0].head.as_deref(), Some("姚明"));
        assert_eq!(descriptions[0].relation.as_deref(), Some("妻子"));
        assert_eq!(descriptions[0].rewritten.as_deref(), Some("姚明的妻子是谁"));

        let descriptions = detect_descriptions("演《无间道》的演员是谁", &source);
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].kind, DescriptionKind::ByTitle);
        assert_eq!(descriptions[0].title.as_deref(), Some("无间道"));
        assert_eq!(descriptions[0].role.as_deref(), Some("演员"));
        assert_eq!(
            descriptions[0].rewritten.as_deref(),
            Some("无间道的演员是谁")
        );
    }

    #[test]
    fn predicate_feature_splits_same_type_candidates() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三丰", Some("人物"));
        linker.register_entity("张三丰演员", Some("人物"));
        linker.register_alias("张三丰", "张三丰演员");
        let mut source = MapSenseSource::default();
        source.add("张三丰", "人物", &["徒弟"], &[], 1);
        source.add("张三丰演员", "人物", &["演员"], &[], 1);
        let outcome = resolutions("张三丰是谁", &linker, &source);
        let ask = outcome.ask.expect("同类型候选应由谓词区分");
        assert!(matches!(
            ask.question.feature,
            DiscriminatingFeature::Predicate { .. }
        ));
    }

    #[test]
    fn unique_verdict_rewrites_alias_to_canonical() {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("果子公司", "苹果公司");
        let mut source = MapSenseSource::default();
        source.add("苹果公司", "组织", &[], &[], 0);
        let outcome = resolutions("果子公司有多少员工", &linker, &source);
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司有多少员工"));
        assert_eq!(outcome.resolutions[0].verdict, Verdict::Unique);
    }

    #[test]
    fn rewrite_at_is_char_safe() {
        assert_eq!(
            rewrite_at("苹果的CEO是谁", 0, 2, "苹果公司"),
            "苹果公司的CEO是谁"
        );
        assert_eq!(rewrite_at("abc", 0, 9, "x"), "abc");
        assert_eq!(rewrite_at("说苹果", 1, 2, "苹果公司"), "说苹果公司");
    }

    #[test]
    fn vote_quantization_prefers_exact_over_alias_with_margin() {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果", Some("水果"));
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("苹果", "苹果公司");
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &[], &[], 0);
        source.add("苹果公司", "组织", &[], &[], 0);
        // 无类型期待时两候选票差 = 精确 4 - 别名 3 = 1 < 边际 2 → 仍应追问
        let outcome = resolutions("苹果是什么", &linker, &source);
        assert!(outcome.ask.is_some(), "边际不足时不应静默选一个");
        // 边际降为 1 时精确匹配足以裁决
        let outcome = ContextualResolver::new().with_margin(1).resolve_text(
            "苹果是什么",
            &linker,
            &source,
            &BindingLedger::new(),
            &[],
            None,
        );
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.resolutions[0].chosen.as_deref(), Some("苹果"));
    }
}

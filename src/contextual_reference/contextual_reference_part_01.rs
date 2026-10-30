use crate::entity_linker::{EntityLinker, LinkCandidate, LinkMethod, LinkedMention};
use crate::entity_registry::fold_surface;
use crate::function_words::{roles_of, Role};
use crate::identity::infer_suffix_type;
use crate::lexical_memory::LexicalMemory;
use crate::nlp::AnnotatedWord;
use crate::pos_ontology::{self, Sort};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
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
    /// 提及跨度内的实际词性标签（稳定排序、去重）。
    pub pos_tags: Vec<String>,
    /// 词性本体投影出的语义排序。
    pub pos_sorts: Vec<Sort>,
    /// 排序投影出的默认领域类型（软证据）。
    pub pos_types: Vec<String>,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarifyOption {
    pub entity: String,
    pub label: String,
    pub etype: Option<String>,
}
/// 区分性问题（选项 ≤3）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarifyQuestion {
    pub mention: String,
    pub feature: DiscriminatingFeature,
    pub options: Vec<ClarifyOption>,
}
impl ClarifyQuestion {
    /// 选项类型一致时按类型选分类词；混合类型保守回退「个」。
    pub fn classifier(&self) -> String {
        let mut types = self
            .options
            .iter()
            .filter_map(|option| option.etype.as_deref())
            .map(str::trim)
            .filter(|etype| !etype.is_empty());
        let Some(first) = types.next() else {
            return "个".to_string();
        };
        if types.any(|etype| etype != first) {
            return "个".to_string();
        }
        crate::quantifier::classifier_for_type(first).classifier
    }

    /// 确定性渲染（同版本同字节，可 golden）。
    pub fn render(&self) -> String {
        let mut out = format!(
            "你说的「{}」是指哪一{}？\n",
            self.mention,
            self.classifier()
        );
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
        self.resolve_text_with_memory(text, linker, source, ledger, hints, focus, None)
    }

    /// 与 [`Self::resolve_text`] 相同，但允许用跨会话词法记忆补充量词期待。
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_text_with_memory(
        &self,
        text: &str,
        linker: &EntityLinker,
        source: &dyn SenseSource,
        ledger: &BindingLedger,
        hints: &[String],
        focus: Option<&str>,
        memory: Option<&LexicalMemory>,
    ) -> CrdOutcome {
        self.resolve_text_with_evidence(text, linker, source, ledger, hints, focus, memory, None)
    }

    /// 在词法记忆之外再注入已标注词，用词性本体为指称补充类型证据。
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_text_with_evidence(
        &self,
        text: &str,
        linker: &EntityLinker,
        source: &dyn SenseSource,
        ledger: &BindingLedger,
        hints: &[String],
        focus: Option<&str>,
        memory: Option<&LexicalMemory>,
        annotated: Option<&[AnnotatedWord]>,
    ) -> CrdOutcome {
        let mentions = linker.link_lattice(text, hints);
        let descriptions = detect_descriptions(text, source);
        let mut global_expectations: Vec<String> =
            crate::function_words::answer_type_keywords(text)
                .into_iter()
                .map(str::to_string)
                .collect();
        if let Some(memory) = memory {
            for (_, expectation) in memory.expectations_in_text(text) {
                global_expectations.push(expectation.expected_type);
            }
            global_expectations.sort();
            global_expectations.dedup();
        }
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
                self.evaluate_with_evidence(
                    mention,
                    linker,
                    text,
                    source,
                    ledger,
                    hints,
                    focus,
                    &[],
                    &global_expectations,
                    memory,
                    annotated,
                )
                .chosen
            })
            .collect();
        let peers: Vec<String> = provisional.iter().flatten().cloned().collect();

        for mention in &mentions {
            let resolution = self.evaluate_with_evidence(
                mention,
                linker,
                text,
                source,
                ledger,
                hints,
                focus,
                &peers,
                &global_expectations,
                memory,
                annotated,
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
        self.evaluate_with_memory(
            mention,
            linker,
            text,
            source,
            ledger,
            hints,
            focus,
            peers,
            global_expectations,
            None,
        )
    }

    /// 与 [`Self::evaluate`] 相同，但可注入词法记忆补充量词期待。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_memory(
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
        memory: Option<&LexicalMemory>,
    ) -> MentionResolution {
        self.evaluate_with_evidence(
            mention,
            linker,
            text,
            source,
            ledger,
            hints,
            focus,
            peers,
            global_expectations,
            memory,
            None,
        )
    }

    /// 与 [`Self::evaluate_with_memory`] 相同，并可注入已标注词补充词性证据。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_evidence(
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
        memory: Option<&LexicalMemory>,
        annotated: Option<&[AnnotatedWord]>,
    ) -> MentionResolution {
        let sketch = sketch_mention_with_evidence(text, mention, memory, annotated);
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
            LinkMethod::Alignment => {
                votes += VOTE_ALIAS;
                evidence.push(format!(
                    "镜像对齐「{}」→「{}」+{}（落地与类型守卫已通过）",
                    sketch.key_surface, candidate.entity, VOTE_ALIAS
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
/// 从词法记忆学习到的量词期待中找出紧邻指称的量词短语。
///
/// 先看指称前 2/1 字窗口，再看指称内部前缀；只在记忆里确有该量词期待时
/// 才返回，避免把任意「指别 + 名词」误当量词短语。
fn learned_quantifier(
    chars: &[char],
    start: usize,
    mention: &str,
    memory: &LexicalMemory,
) -> Option<(String, String, usize)> {
    for len in [2usize, 1] {
        if start >= len {
            let word: String = chars[start - len..start].iter().collect();
            if let Some(classifier) = classifier_of(&word) {
                if let Some(expectation) = memory.classifier_expectation(&classifier) {
                    return Some((word, expectation.expected_type, 0));
                }
            }
        }
    }
    for len in [2usize, 1] {
        let word: String = mention.chars().take(len).collect();
        if word.chars().count() != len {
            continue;
        }
        if let Some(classifier) = classifier_of(&word) {
            if let Some(expectation) = memory.classifier_expectation(&classifier) {
                return Some((word, expectation.expected_type, len));
            }
        }
    }
    None
}
/// 量词短语 → 量词词面（「那位」→「位」；独立「位」原样返回）。
fn classifier_of(word: &str) -> Option<String> {
    let mut chars = word.chars();
    let first = chars.next()?;
    let suffix: String = chars.collect();
    if suffix.is_empty() {
        let single = first.to_string();
        return roles_of(&single)
            .contains(&Role::Quantifier)
            .then_some(single);
    }
    if !matches!(first, '这' | '那' | '哪' | '每' | '某' | '几' | '数') {
        return None;
    }
    (suffix.chars().count() == 1 && roles_of(&suffix).contains(&Role::Quantifier)).then_some(suffix)
}
/// 指称素描：从提及及其前后文抽中文特征。
pub fn sketch_mention(text: &str, mention: &LinkedMention) -> MentionSketch {
    sketch_mention_with_evidence(text, mention, None, None)
}
/// 与 [`sketch_mention`] 相同，但允许用词法记忆补充量词期待。
pub fn sketch_mention_with_memory(
    text: &str,
    mention: &LinkedMention,
    memory: Option<&LexicalMemory>,
) -> MentionSketch {
    sketch_mention_with_evidence(text, mention, memory, None)
}
/// 在词法记忆之外再注入已标注词，用词性本体补充指称的语义排序。
pub fn sketch_mention_with_evidence(
    text: &str,
    mention: &LinkedMention,
    memory: Option<&LexicalMemory>,
    annotated: Option<&[AnnotatedWord]>,
) -> MentionSketch {
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
    // 词法记忆：已学到的量词期待不依赖硬编码表；只补充，不覆盖既有证据。
    if let Some(memory) = memory {
        if let Some(word) = quantifier.as_deref() {
            if let Some(classifier) = classifier_of(word) {
                if let Some(expectation) = memory.classifier_expectation(&classifier) {
                    expectation_types.push(expectation.expected_type);
                }
            }
        } else if let Some((word, expected, offset)) =
            learned_quantifier(&chars, start, &mention.mention, memory)
        {
            quantifier = Some(word);
            key_offset += offset;
            expectation_types.push(expected);
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

    let mut pos_tags: Vec<String> = Vec::new();
    let mut pos_sorts: Vec<Sort> = Vec::new();
    let mut pos_types: Vec<String> = Vec::new();
    if let Some(words) = annotated {
        let end = start.saturating_add(mention.mention.chars().count());
        for word in words
            .iter()
            .filter(|word| word.end > start && word.start < end)
        {
            let profile = pos_ontology::profile(&word.pos);
            if !word.pos.trim().is_empty() {
                pos_tags.push(word.pos.clone());
            }
            if profile.known {
                pos_sorts.extend(profile.sorts.iter().copied());
                pos_types.extend(profile.ontology_types());
            }
        }
        pos_tags.sort();
        pos_tags.dedup();
        pos_sorts.sort();
        pos_sorts.dedup();
        pos_types.sort();
        pos_types.dedup();
        expectation_types.extend(pos_types.iter().cloned());
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
        pos_tags,
        pos_sorts,
        pos_types,
    }
}

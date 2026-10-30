//! 重写手册（Rewrite Book）：把「同一逻辑计划的两种说法」挖成可复核的
//! 表面→规范重写规则。
//!
//! 与手写 `query_rewrite` 的关系是**底座与学习层**：解析器继续负责保守的
//! 首轮切分；本模块只从成对的 `surface/canonical` 回合里归纳带类型槽位的
//! 规则，再用执行计划是否等价做监督。规则先是候选，显式采纳后才可能进入
//! 运行时；离线评估只把规则当作基础解析失败后的 fallback，因此不会覆盖
//! 已有成功解析，也不会用覆盖率换错答。
//!
//! 训练数据不需要神经网络：每条回合声明两个问句；两句话编译出的
//! `QueryPlan` 相同即视为等价。槽位按计划角色命名（`{S}` 主语、`{O}` 宾语、
//! `{P}` 谓词、`{E0}`… 链实体），实体与谓词先经词表归一，因而规则可以跨
//! 实体、跨谓词复用。留出谓词评估用来检查「没见过的谓词」而不是复述训练集。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::query_rewrite::{self, RelationLexicon};

/// 手册的序列化方言；版本不符时拒绝恢复。
pub const REWRITE_DIALECT: &str = "knowprolog-rewrite-book/1";
/// 默认每本手册最多保留多少条规则。
pub const DEFAULT_MAX_RULES: usize = 128;
/// 一次重写最多链式应用多少条规则（防止规则爆炸）。
pub const MAX_CHAIN: usize = 2;
/// 匹配时最多保留多少组槽位绑定（相邻槽位会带来切分歧义）。
const MAX_BINDINGS: usize = 64;

/// 规则状态。学习只做证据累积，生效必须显式采纳。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteStatus {
    Proposed,
    Active,
    Retired,
}

impl RewriteStatus {
    pub fn label(self) -> &'static str {
        match self {
            RewriteStatus::Proposed => "Proposed",
            RewriteStatus::Active => "Active",
            RewriteStatus::Retired => "Retired",
        }
    }
}

/// 一条训练回合：同一个逻辑计划的表面问句与规范问句。
///
/// `entities` / `predicates` 可省略；缺省时从两边编译出的计划里抽取。
/// `negative = true` 的回合没有规范问句，只作为硬反例：任何规则命中它的
/// 表面形式都会被否决。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewriteEpisode {
    pub surface: String,
    #[serde(default)]
    pub canonical: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub predicates: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub negative: bool,
}

impl RewriteEpisode {
    pub fn new(surface: impl Into<String>, canonical: impl Into<String>) -> Self {
        Self {
            surface: surface.into(),
            canonical: canonical.into(),
            entities: Vec::new(),
            predicates: Vec::new(),
            source: String::new(),
            negative: false,
        }
    }

    pub fn with_terms(mut self, entities: &[&str], predicates: &[&str]) -> Self {
        self.entities = entities.iter().map(|value| value.to_string()).collect();
        self.predicates = predicates.iter().map(|value| value.to_string()).collect();
        self
    }

    pub fn negative(surface: impl Into<String>) -> Self {
        Self {
            surface: surface.into(),
            canonical: String::new(),
            entities: Vec::new(),
            predicates: Vec::new(),
            source: "negative".to_string(),
            negative: true,
        }
    }
}

/// 一条可复核的规则：表面模式 → 规范模式，带计划类型和证据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewriteRule {
    pub id: String,
    pub plan_kind: String,
    pub surface: String,
    pub canonical: String,
    pub status: RewriteStatus,
    pub support: usize,
    #[serde(default)]
    pub predicates: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub examples: Vec<(String, String)>,
    #[serde(default)]
    pub refutations: usize,
    #[serde(default)]
    pub source: String,
}

impl RewriteRule {
    pub fn receipt(&self) -> String {
        format!(
            "{} [{}] {} → {}（{}，支撑 {}，谓词 {}）",
            self.id,
            self.status.label(),
            self.surface,
            self.canonical,
            self.plan_kind,
            self.support,
            if self.predicates.is_empty() {
                "—".to_string()
            } else {
                self.predicates.join("、")
            }
        )
    }

    fn specificity(&self) -> usize {
        self.surface
            .chars()
            .filter(|ch| *ch != '{' && *ch != '}')
            .count()
    }
}

/// 一次规则应用的收据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteApplication {
    pub text: String,
    pub rules: Vec<String>,
}

/// 词表视图：评估时由语料声明构造，运行时由本体/谓词画像提供。
#[derive(Debug, Clone, Default)]
pub struct EpisodeLexicon {
    entities: BTreeMap<String, String>,
    predicates: BTreeMap<String, String>,
}

impl EpisodeLexicon {
    pub fn from_episodes(episodes: &[RewriteEpisode]) -> Self {
        let mut lexicon = Self::default();
        for episode in episodes {
            for entity in &episode.entities {
                lexicon.entities.insert(entity.clone(), entity.clone());
            }
            for predicate in &episode.predicates {
                lexicon
                    .predicates
                    .insert(predicate.clone(), predicate.clone());
            }
        }
        // 未声明词表时，从可编译的规范问句里逐步吸收计划项；吸收到不动点
        // 为止，保证外部语料只写问答对也能工作。
        loop {
            let before = (lexicon.entities.len(), lexicon.predicates.len());
            let snapshot = lexicon.clone();
            for episode in episodes.iter().filter(|episode| !episode.negative) {
                if let Some((plan, _)) = compile_pair(episode, &snapshot) {
                    lexicon.absorb_plan(&plan);
                }
            }
            if (lexicon.entities.len(), lexicon.predicates.len()) == before {
                break;
            }
        }
        lexicon
    }

    pub fn admit_entity(&mut self, surface: impl Into<String>) -> &mut Self {
        let surface = surface.into();
        self.entities.insert(surface.clone(), surface);
        self
    }

    pub fn admit_predicate(
        &mut self,
        surface: impl Into<String>,
        canonical: impl Into<String>,
    ) -> &mut Self {
        self.predicates.insert(surface.into(), canonical.into());
        self
    }

    pub fn entities(&self) -> impl Iterator<Item = &str> {
        self.entities.keys().map(String::as_str)
    }

    pub fn predicates(&self) -> impl Iterator<Item = &str> {
        self.predicates.values().map(String::as_str)
    }

    fn absorb_plan(&mut self, plan: &QueryPlan) {
        for (placeholder, value) in plan_slots(plan) {
            if placeholder.starts_with("{P}") {
                self.predicates.entry(value.clone()).or_insert(value);
            } else {
                self.entities.entry(value.clone()).or_insert(value);
            }
        }
    }
}

impl RelationLexicon for EpisodeLexicon {
    fn canonical_entity(&self, raw: &str) -> Option<String> {
        self.entities.get(raw.trim()).cloned()
    }

    fn canonical_predicate(&self, raw: &str) -> Option<String> {
        self.predicates.get(raw.trim()).cloned()
    }
}

/// 一条规则的候选分组（挖掘阶段的中间态）。
#[derive(Debug, Clone)]
struct RuleGroup {
    plan_kind: String,
    surface: String,
    canonical: String,
    support: usize,
    predicates: BTreeSet<String>,
    entities: BTreeSet<String>,
    examples: Vec<(String, String)>,
}

/// 手册本体：规则按 id 稳定排序，序列化后逐字节可复现。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewriteBook {
    pub dialect: String,
    pub version: u32,
    pub rules: Vec<RewriteRule>,
}

impl Default for RewriteBook {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteBook {
    pub fn new() -> Self {
        Self {
            dialect: REWRITE_DIALECT.to_string(),
            version: 1,
            rules: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn rules(&self) -> impl Iterator<Item = &RewriteRule> {
        self.rules.iter()
    }

    pub fn rule(&self, id: &str) -> Option<&RewriteRule> {
        self.rules.iter().find(|rule| rule.id == id)
    }

    pub fn status_count(&self, status: RewriteStatus) -> usize {
        self.rules
            .iter()
            .filter(|rule| rule.status == status)
            .count()
    }

    pub fn adopt(&mut self, selector: &str) -> Result<&RewriteRule, String> {
        self.set_status(selector, RewriteStatus::Active)
    }

    pub fn retire(&mut self, selector: &str) -> Result<&RewriteRule, String> {
        self.set_status(selector, RewriteStatus::Retired)
    }

    fn set_status(
        &mut self,
        selector: &str,
        status: RewriteStatus,
    ) -> Result<&RewriteRule, String> {
        let index = self.find_index(selector)?;
        self.rules[index].status = status;
        Ok(&self.rules[index])
    }

    fn find_index(&self, selector: &str) -> Result<usize, String> {
        if let Some(index) = self.rules.iter().position(|rule| rule.id == selector) {
            return Ok(index);
        }
        let matches: Vec<usize> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.id.starts_with(selector))
            .map(|(index, _)| index)
            .collect();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(format!("找不到重写规则「{}」", selector)),
            _ => Err(format!(
                "规则选择器「{}」有 {} 个匹配，请写完整 id",
                selector,
                matches.len()
            )),
        }
    }

    /// 只用已采纳规则重写；运行时（未来的 fallback 通道）应调用这个入口。
    pub fn resolve(&self, text: &str, lexicon: &dyn RelationLexicon) -> Option<RewriteApplication> {
        self.resolve_with(text, lexicon, false)
    }

    /// 离线评估可以带上候选规则；`include_proposed = false` 时只看已采纳规则。
    pub fn resolve_with(
        &self,
        text: &str,
        lexicon: &dyn RelationLexicon,
        include_proposed: bool,
    ) -> Option<RewriteApplication> {
        let mut current = text.trim().to_string();
        let mut applied = Vec::new();
        for _ in 0..MAX_CHAIN {
            let Some((rule, next)) = self.apply_once(&current, lexicon, include_proposed) else {
                break;
            };
            if next == current || applied.contains(&rule.id) {
                break;
            }
            applied.push(rule.id.clone());
            current = next;
        }
        if applied.is_empty() {
            None
        } else {
            Some(RewriteApplication {
                text: current,
                rules: applied,
            })
        }
    }

    fn apply_once(
        &self,
        text: &str,
        lexicon: &dyn RelationLexicon,
        include_proposed: bool,
    ) -> Option<(&RewriteRule, String)> {
        let mut candidates: Vec<&RewriteRule> = self
            .rules
            .iter()
            .filter(|rule| match rule.status {
                RewriteStatus::Active => true,
                RewriteStatus::Proposed => include_proposed,
                RewriteStatus::Retired => false,
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .specificity()
                .cmp(&left.specificity())
                .then(right.support.cmp(&left.support))
                .then(left.id.cmp(&right.id))
        });
        candidates
            .into_iter()
            .find_map(|rule| apply_rule(rule, text, lexicon).map(|next| (rule, next)))
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let bytes = crate::canonical::canonical_bytes(self)
            .map_err(|error| format!("重写手册序列化失败：{}", error))?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("创建重写手册目录失败：{}", error))?;
            }
        }
        fs::write(path, bytes).map_err(|error| format!("写入重写手册失败：{}", error))
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes =
            fs::read(path.as_ref()).map_err(|error| format!("读取重写手册失败：{}", error))?;
        let book: RewriteBook = serde_json::from_slice(&bytes)
            .map_err(|error| format!("解析重写手册失败：{}", error))?;
        if book.dialect != REWRITE_DIALECT {
            return Err(format!(
                "重写手册方言不匹配：期望 {}，实际 {}",
                REWRITE_DIALECT, book.dialect
            ));
        }
        Ok(book)
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "重写手册：{} 条规则（Active {} / Proposed {} / Retired {}）\n",
            self.len(),
            self.status_count(RewriteStatus::Active),
            self.status_count(RewriteStatus::Proposed),
            self.status_count(RewriteStatus::Retired)
        );
        for rule in &self.rules {
            output.push_str(&format!("  {}\n", rule.receipt()));
            if let Some((surface, canonical)) = rule.examples.first() {
                output.push_str(&format!("    例：{} ⇒ {}\n", surface, canonical));
            }
            if rule.refutations > 0 {
                output.push_str(&format!("    反例否决：{}\n", rule.refutations));
            }
        }
        output
    }
}

/// 挖掘：从训练回合归纳规则，并用全部回合（含反例）做硬校验。
pub fn mine(
    train: &[RewriteEpisode],
    all: &[RewriteEpisode],
    lexicon: &EpisodeLexicon,
    min_support: usize,
) -> RewriteBook {
    let min_support = min_support.max(1);
    let mut groups: BTreeMap<(String, String, String), RuleGroup> = BTreeMap::new();
    for episode in train.iter().filter(|episode| !episode.negative) {
        let Some((plan, _)) = compile_pair(episode, lexicon) else {
            continue;
        };
        let Some(surface) = pattern_from_question(&episode.surface, &plan) else {
            continue;
        };
        let Some(canonical) = pattern_from_question(&episode.canonical, &plan) else {
            continue;
        };
        if surface == canonical {
            continue;
        }
        if !slots(&canonical)
            .iter()
            .all(|slot| slots(&surface).contains(slot))
        {
            continue;
        }
        let key = (
            QueryPlanner::kind(&plan).to_string(),
            surface.clone(),
            canonical.clone(),
        );
        let group = groups.entry(key).or_insert_with(|| RuleGroup {
            plan_kind: QueryPlanner::kind(&plan).to_string(),
            surface,
            canonical,
            support: 0,
            predicates: BTreeSet::new(),
            entities: BTreeSet::new(),
            examples: Vec::new(),
        });
        group.support += 1;
        for (placeholder, value) in plan_slots(&plan) {
            if placeholder == "{P}" {
                group.predicates.insert(value);
            } else {
                group.entities.insert(value);
            }
        }
        if group.examples.len() < 4 {
            group
                .examples
                .push((episode.surface.clone(), episode.canonical.clone()));
        }
    }

    let mut rules: Vec<RewriteRule> = groups
        .into_values()
        .filter(|group| group.support >= min_support)
        .map(|group| {
            let id = rule_id(&group.plan_kind, &group.surface, &group.canonical);
            RewriteRule {
                id: id.clone(),
                plan_kind: group.plan_kind,
                surface: group.surface,
                canonical: group.canonical,
                status: RewriteStatus::Proposed,
                support: group.support,
                predicates: group.predicates.into_iter().collect(),
                entities: group.entities.into_iter().collect(),
                examples: group.examples,
                refutations: 0,
                source: format!("mined#rewrite#{}", id),
            }
        })
        .collect();

    for rule in &mut rules {
        rule.refutations = refutations(rule, all, lexicon);
    }
    rules.retain(|rule| rule.refutations == 0);
    rules.sort_by(|left, right| {
        left.plan_kind
            .cmp(&right.plan_kind)
            .then(left.surface.cmp(&right.surface))
            .then(left.canonical.cmp(&right.canonical))
    });
    rules.truncate(DEFAULT_MAX_RULES);
    RewriteBook {
        dialect: REWRITE_DIALECT.to_string(),
        version: 1,
        rules,
    }
}

fn refutations(rule: &RewriteRule, episodes: &[RewriteEpisode], lexicon: &EpisodeLexicon) -> usize {
    let mut count = 0;
    for episode in episodes {
        if rule.surface.trim() == episode.surface.trim() {
            continue;
        }
        let Some(rewritten) = apply_rule(rule, &episode.surface, lexicon) else {
            continue;
        };
        if episode.negative {
            count += 1;
            continue;
        }
        let Some((expected, _)) = compile_pair(episode, lexicon) else {
            count += 1;
            continue;
        };
        let Ok(actual) = compile_question(&rewritten, lexicon) else {
            count += 1;
            continue;
        };
        if !plans_equivalent(&actual, &expected) {
            count += 1;
        }
    }
    count
}

/// 一个评测切片。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RewriteSlice {
    pub episodes: usize,
    pub baseline_correct: usize,
    pub rewritten_correct: usize,
    pub rules_fired: usize,
    pub wrong_rewrites: usize,
    pub unresolved: usize,
}

impl RewriteSlice {
    pub fn coverage_gain(&self) -> f64 {
        if self.episodes == 0 {
            0.0
        } else {
            (self.rewritten_correct as f64 - self.baseline_correct as f64) / self.episodes as f64
        }
    }

    pub fn precision(&self) -> f64 {
        if self.rules_fired == 0 {
            1.0
        } else {
            (self.rules_fired - self.wrong_rewrites) as f64 / self.rules_fired as f64
        }
    }

    pub fn merge(&mut self, other: &RewriteSlice) {
        self.episodes += other.episodes;
        self.baseline_correct += other.baseline_correct;
        self.rewritten_correct += other.rewritten_correct;
        self.rules_fired += other.rules_fired;
        self.wrong_rewrites += other.wrong_rewrites;
        self.unresolved += other.unresolved;
    }
}

/// 一条具体失败样例。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewriteMiss {
    pub surface: String,
    pub canonical: String,
    pub reason: String,
}

/// 留出集评测报告。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RewriteEval {
    pub all: RewriteSlice,
    pub held_out: RewriteSlice,
    pub held_out_predicates: Vec<String>,
    pub misses: Vec<RewriteMiss>,
}

impl RewriteEval {
    pub fn render_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "留出谓词：{}\n",
            if self.held_out_predicates.is_empty() {
                "—".to_string()
            } else {
                self.held_out_predicates.join("、")
            }
        ));
        for (label, slice) in [("全部", &self.all), ("留出", &self.held_out)] {
            output.push_str(&format!(
                "{}：{} 句，基础命中 {}，重写命中 {}，规则触发 {}，错答 {}，未解 {}；覆盖率增益 {:+.1}%，精度 {:.1}%。\n",
                label,
                slice.episodes,
                slice.baseline_correct,
                slice.rewritten_correct,
                slice.rules_fired,
                slice.wrong_rewrites,
                slice.unresolved,
                slice.coverage_gain() * 100.0,
                slice.precision() * 100.0
            ));
        }
        for miss in self.misses.iter().take(12) {
            output.push_str(&format!(
                "  ✗ {} ⇒ {}（{}）\n",
                miss.surface, miss.canonical, miss.reason
            ));
        }
        output
    }
}

/// 评估：规则只在基础解析失败时接管，成功解析永远原样保留。
pub fn evaluate(
    book: &RewriteBook,
    episodes: &[RewriteEpisode],
    lexicon: &EpisodeLexicon,
    held_out_predicates: &BTreeSet<String>,
) -> RewriteEval {
    let mut report = RewriteEval {
        held_out_predicates: held_out_predicates.iter().cloned().collect(),
        ..RewriteEval::default()
    };
    for episode in episodes.iter().filter(|episode| !episode.negative) {
        let Some((expected, predicates)) = compile_pair(episode, lexicon) else {
            continue;
        };
        let baseline = compile_question(&episode.surface, lexicon).ok();
        let baseline_ok = baseline
            .as_ref()
            .is_some_and(|actual| plans_equivalent(actual, &expected));
        let held_out = predicates
            .iter()
            .any(|predicate| held_out_predicates.contains(predicate));
        let mut outcome = RewriteSlice {
            episodes: 1,
            ..RewriteSlice::default()
        };
        if baseline_ok {
            outcome.baseline_correct = 1;
            outcome.rewritten_correct = 1;
        } else if let Some(application) = book.resolve_with(&episode.surface, lexicon, true) {
            outcome.rules_fired = 1;
            match compile_question(&application.text, lexicon) {
                Ok(actual) if plans_equivalent(&actual, &expected) => {
                    outcome.rewritten_correct = 1;
                }
                Ok(_) => {
                    outcome.wrong_rewrites = 1;
                    push_miss(
                        &mut report,
                        episode,
                        held_out,
                        &format!("重写为 {} 后计划不同", application.text),
                    );
                }
                Err(error) => {
                    outcome.wrong_rewrites = 1;
                    push_miss(&mut report, episode, held_out, &error);
                }
            }
        } else {
            outcome.unresolved = 1;
            push_miss(&mut report, episode, held_out, "没有规则命中");
        }
        report.all.merge(&outcome);
        if held_out {
            report.held_out.merge(&outcome);
        }
    }
    report
}

fn push_miss(report: &mut RewriteEval, episode: &RewriteEpisode, held_out: bool, reason: &str) {
    if !held_out {
        return;
    }
    report.misses.push(RewriteMiss {
        surface: episode.surface.clone(),
        canonical: episode.canonical.clone(),
        reason: reason.to_string(),
    });
}

/// 从语料谓词里确定默认留出集：排序后每隔三个取一个。
pub fn default_holdout(predicates: &BTreeSet<String>) -> BTreeSet<String> {
    predicates
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 3 == 2)
        .map(|(_, predicate)| predicate.clone())
        .collect()
}

/// 单条回合涉及的谓词（声明值 ∪ 计划里的谓词）。
pub fn episode_predicates(episode: &RewriteEpisode, lexicon: &EpisodeLexicon) -> BTreeSet<String> {
    let mut predicates: BTreeSet<String> = episode.predicates.iter().cloned().collect();
    if let Some((plan, _)) = compile_pair(episode, lexicon) {
        for (placeholder, value) in plan_slots(&plan) {
            if placeholder == "{P}" {
                predicates.insert(value);
            }
        }
    }
    predicates
}

/// 语料里出现过的谓词（声明值 ∪ 计划里的谓词）。
pub fn corpus_predicates(
    episodes: &[RewriteEpisode],
    lexicon: &EpisodeLexicon,
) -> BTreeSet<String> {
    let mut predicates = BTreeSet::new();
    for episode in episodes {
        predicates.extend(episode_predicates(episode, lexicon));
    }
    predicates
}

/// 读取 JSONL（也接受一个 JSON 数组）格式的回合语料；`#` 行被忽略。
pub fn load_episodes(path: impl AsRef<Path>) -> Result<Vec<RewriteEpisode>, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|error| format!("读取重写语料失败：{}", error))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('[') {
        return serde_json::from_str(trimmed)
            .map_err(|error| format!("解析重写语料数组失败：{}", error));
    }
    let mut episodes = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let episode: RewriteEpisode = serde_json::from_str(line).map_err(|error| {
            format!(
                "解析重写语料第 {} 行失败（{}）：{}",
                line_number + 1,
                path.display(),
                error
            )
        })?;
        episodes.push(episode);
    }
    Ok(episodes)
}

/// 内置种子语料：六种谓词 × 三组实体 × 多种外壳。
///
/// 这些外壳故意包含基础解析器不认识的副词/关系从句脚手架，用于验证
/// 「规则作为 fallback」能带来真实覆盖率增益，而不是复述已有能力。
pub fn seed_episodes() -> Vec<RewriteEpisode> {
    let predicates = ["认识", "教", "喜欢", "拜访", "帮助", "邀请"];
    let pairs = [("张三", "李四"), ("王五", "赵六"), ("小明", "小红")];
    let mut episodes = Vec::new();
    for predicate in predicates {
        for (subject, object) in pairs {
            let entities = [subject, object];
            let predicate_list = [predicate];
            let canonical_object = format!("{}{}谁", subject, predicate);
            let canonical_subject = format!("谁{}{}", predicate, object);
            let canonical_yes_no = format!("{}{}{}吗", subject, predicate, object);
            let canonical_count = format!("{}有几个{}", subject, predicate);
            for (surface, canonical) in [
                (
                    format!("{}到底{}谁", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("{}究竟{}谁", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("{}到底{}过谁", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("被{}{}的那个人到底是谁", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("{}到底把谁{}了", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("到底谁被{}{}了", subject, predicate),
                    canonical_object.clone(),
                ),
                (
                    format!("{}到底被谁{}了", object, predicate),
                    canonical_subject.clone(),
                ),
                (
                    format!("到底谁{}{}", predicate, object),
                    canonical_subject.clone(),
                ),
                (
                    format!("{}究竟把谁{}了", object, predicate),
                    canonical_subject.clone(),
                ),
                (
                    format!("{}到底把{}{}了吗", subject, object, predicate),
                    canonical_yes_no.clone(),
                ),
                (
                    format!("{}到底被{}{}了吗", object, subject, predicate),
                    canonical_yes_no.clone(),
                ),
                (
                    format!("{}到底{}{}吗", subject, predicate, object),
                    canonical_yes_no.clone(),
                ),
                (
                    format!("{}到底有几个{}", subject, predicate),
                    canonical_count.clone(),
                ),
            ] {
                episodes.push(
                    RewriteEpisode::new(surface, canonical).with_terms(&entities, &predicate_list),
                );
            }
        }
    }
    episodes.push(RewriteEpisode::negative("张三到底不认识谁"));
    episodes.push(RewriteEpisode::negative("今天天气到底怎么样"));
    episodes
}

// ---------------------------------------------------------------------------
// 计划编译与槽位抽取
// ---------------------------------------------------------------------------

/// 编译一条回合：规范问句是监督信号；表面问句只需要能被同一词表识别。
fn compile_pair(
    episode: &RewriteEpisode,
    lexicon: &EpisodeLexicon,
) -> Option<(QueryPlan, BTreeSet<String>)> {
    if episode.negative {
        return None;
    }
    let plan = compile_question(&episode.canonical, lexicon).ok()?;
    let mut predicates = BTreeSet::new();
    for (placeholder, value) in plan_slots(&plan) {
        if placeholder == "{P}" {
            predicates.insert(value);
        }
    }
    Some((plan, predicates))
}

/// 语义等价监督：同一关系核心的计划形状不同也算同一问题；
/// 超出关系核心的计划仍要求完全相等，绝不放宽到「看起来像」。
fn plans_equivalent(left: &QueryPlan, right: &QueryPlan) -> bool {
    crate::query_canonical::plans_equivalent(
        left,
        right,
        &crate::query_canonical::CanonicalSchema::new(),
    )
}

fn compile_question(question: &str, lexicon: &EpisodeLexicon) -> Result<QueryPlan, String> {
    let planner = QueryPlanner::new();
    if let Some(plan) = planner.parse(question) {
        return Ok(plan);
    }
    query_rewrite::rewrite(question, lexicon)
        .map(|rewrite| rewrite.plan)
        .ok_or_else(|| "基础解析器未识别".to_string())
}

/// 计划里的角色槽位：占位符 → 规范值。
fn plan_slots(plan: &QueryPlan) -> Vec<(String, String)> {
    let mut slots = Vec::new();
    match plan {
        QueryPlan::FindObject { segments } => {
            if let Some((predicate, entities)) = segments.split_last() {
                for (index, entity) in entities.iter().enumerate() {
                    slots.push((format!("{{E{}}}", index), entity.clone()));
                }
                slots.push(("{P}".to_string(), predicate.clone()));
            }
        }
        QueryPlan::FindSubject { object, predicate }
        | QueryPlan::Count {
            subject: object,
            predicate,
        }
        | QueryPlan::Exist {
            subject: object,
            predicate,
        }
        | QueryPlan::Closure {
            subject: object,
            predicate,
            ..
        } => {
            slots.push(("{O}".to_string(), object.clone()));
            slots.push(("{P}".to_string(), predicate.clone()));
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => {
            slots.push(("{S}".to_string(), subject.clone()));
            slots.push(("{P}".to_string(), predicate.clone()));
        }
        QueryPlan::FindObjectExcept {
            subject,
            predicate,
            excluded,
        } => {
            slots.push(("{S}".to_string(), subject.clone()));
            slots.push(("{P}".to_string(), predicate.clone()));
            for (index, entity) in excluded.iter().enumerate() {
                slots.push((format!("{{E{}}}", index + 1), entity.clone()));
            }
        }
        QueryPlan::FindSubjectExcept {
            object,
            predicate,
            excluded,
        } => {
            slots.push(("{O}".to_string(), object.clone()));
            slots.push(("{P}".to_string(), predicate.clone()));
            for (index, entity) in excluded.iter().enumerate() {
                slots.push((format!("{{E{}}}", index + 1), entity.clone()));
            }
        }
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            ..
        } => {
            slots.push(("{S}".to_string(), subject.clone()));
            slots.push(("{P}".to_string(), predicate.clone()));
            slots.push(("{O}".to_string(), object.clone()));
        }
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } => {
            if let Some((predicate, entities)) = subject_segments.split_last() {
                for (index, entity) in entities.iter().enumerate() {
                    slots.push((format!("{{E{}}}", index), entity.clone()));
                }
                slots.push(("{P}".to_string(), predicate.clone()));
            }
            slots.push(("{O}".to_string(), object.clone()));
        }
        QueryPlan::PathBetween { start, goal, .. }
        | QueryPlan::CommonNeighbor {
            left: start,
            right: goal,
        }
        | QueryPlan::Intermediary { start, goal, .. } => {
            slots.push(("{E0}".to_string(), start.clone()));
            slots.push(("{E1}".to_string(), goal.clone()));
        }
        QueryPlan::Similar { entity } => {
            slots.push(("{E0}".to_string(), entity.clone()));
        }
        QueryPlan::Analogy { a, b, c } => {
            slots.push(("{E0}".to_string(), a.clone()));
            slots.push(("{E1}".to_string(), b.clone()));
            slots.push(("{E2}".to_string(), c.clone()));
        }
    }
    slots
}

/// 把问句中的角色值替换成占位符；至少保留一个槽位，否则视为死记硬背。
fn pattern_from_question(question: &str, plan: &QueryPlan) -> Option<String> {
    let mut terms: Vec<(String, String)> = plan_slots(plan)
        .into_iter()
        .map(|(placeholder, value)| (value, placeholder))
        .filter(|(value, _)| !value.trim().is_empty())
        .collect();
    if terms.is_empty() {
        return None;
    }
    terms.sort_by(|left, right| {
        right
            .0
            .chars()
            .count()
            .cmp(&left.0.chars().count())
            .then(left.0.cmp(&right.0))
    });
    let mut output = String::with_capacity(question.len());
    let mut rest = question.trim();
    let mut used = false;
    while !rest.is_empty() {
        let mut matched = None;
        for (value, placeholder) in &terms {
            if rest.starts_with(value) {
                matched = Some((value.len(), placeholder));
                break;
            }
        }
        if let Some((length, placeholder)) = matched {
            output.push_str(placeholder);
            rest = &rest[length..];
            used = true;
        } else {
            let ch = rest.chars().next()?;
            output.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    if !used {
        return None;
    }
    Some(output)
}

fn slots(pattern: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = pattern;
    while let Some(start) = rest.find('{') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            break;
        };
        let name = &after[..end];
        if !name.is_empty() {
            out.push(format!("{{{}}}", name));
        }
        rest = &after[end + 1..];
    }
    out
}

fn rule_id(plan_kind: &str, surface: &str, canonical: &str) -> String {
    let key = format!("{}\u{1f}{}\u{1f}{}", plan_kind, surface, canonical);
    format!("rw-{}", &fnv1a_hex(key.as_bytes())[..12])
}

// ---------------------------------------------------------------------------
// 模式匹配
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum PatternToken {
    Literal(String),
    Slot(String),
}

fn tokenize(pattern: &str) -> Option<Vec<PatternToken>> {
    let mut tokens = Vec::new();
    let mut literal = String::new();
    let mut rest = pattern;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('{') {
            let end = after.find('}')?;
            let name = &after[..end];
            if name.is_empty() {
                return None;
            }
            if !literal.is_empty() {
                tokens.push(PatternToken::Literal(std::mem::take(&mut literal)));
            }
            tokens.push(PatternToken::Slot(name.to_string()));
            rest = &after[end + 1..];
        } else {
            let ch = rest.chars().next()?;
            if ch == '}' {
                return None;
            }
            literal.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    if !literal.is_empty() {
        tokens.push(PatternToken::Literal(literal));
    }
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}

fn apply_rule(rule: &RewriteRule, text: &str, lexicon: &dyn RelationLexicon) -> Option<String> {
    let tokens = tokenize(&rule.surface)?;
    let mut bindings = Vec::new();
    match_tokens(&tokens, 0, text.trim(), BTreeMap::new(), &mut bindings);
    bindings.into_iter().find_map(|mut binding| {
        canonicalize_bindings(&mut binding, lexicon)?;
        let canonical = tokenize(&rule.canonical)?;
        Some(render_bindings(&canonical, &binding))
    })
}

fn match_tokens(
    tokens: &[PatternToken],
    index: usize,
    rest: &str,
    bindings: BTreeMap<String, String>,
    out: &mut Vec<BTreeMap<String, String>>,
) {
    if out.len() >= MAX_BINDINGS {
        return;
    }
    let Some(token) = tokens.get(index) else {
        if rest.is_empty() {
            out.push(bindings);
        }
        return;
    };
    match token {
        PatternToken::Literal(literal) => {
            if let Some(after) = rest.strip_prefix(literal.as_str()) {
                match_tokens(tokens, index + 1, after, bindings, out);
            }
        }
        PatternToken::Slot(name) => {
            if let Some(existing) = bindings.get(name) {
                let existing = existing.clone();
                if let Some(after) = rest.strip_prefix(existing.as_str()) {
                    match_tokens(tokens, index + 1, after, bindings, out);
                }
                return;
            }
            for (offset, _) in rest
                .char_indices()
                .skip(1)
                .chain(std::iter::once((rest.len(), '\0')))
            {
                if offset == 0 {
                    continue;
                }
                let captured = &rest[..offset];
                if captured.trim().is_empty() {
                    continue;
                }
                let mut next = bindings.clone();
                next.insert(name.clone(), captured.to_string());
                match_tokens(tokens, index + 1, &rest[offset..], next, out);
                if out.len() >= MAX_BINDINGS {
                    return;
                }
            }
        }
    }
}

fn canonicalize_bindings(
    bindings: &mut BTreeMap<String, String>,
    lexicon: &dyn RelationLexicon,
) -> Option<()> {
    for (placeholder, value) in bindings.iter_mut() {
        if placeholder == "P" {
            *value = lexicon.canonical_predicate(value)?;
        } else {
            *value = lexicon.canonical_entity(value)?;
        }
    }
    Some(())
}

fn render_bindings(tokens: &[PatternToken], bindings: &BTreeMap<String, String>) -> String {
    let mut output = String::new();
    for token in tokens {
        match token {
            PatternToken::Literal(literal) => output.push_str(literal),
            PatternToken::Slot(name) => {
                if let Some(value) = bindings.get(name) {
                    output.push_str(value);
                } else {
                    output.push('{');
                    output.push_str(name);
                    output.push('}');
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_lexicon() -> EpisodeLexicon {
        EpisodeLexicon::from_episodes(&seed_episodes())
    }

    #[test]
    fn seed_corpus_compiles_into_equivalent_plans() {
        let lexicon = seed_lexicon();
        for episode in seed_episodes().iter().filter(|episode| !episode.negative) {
            let canonical =
                compile_question(&episode.canonical, &lexicon).unwrap_or_else(|error| {
                    panic!("{} 的规范问句无法编译：{}", episode.canonical, error)
                });
            let (plan, _) = compile_pair(episode, &lexicon).expect("规范问句应可编译");
            assert_eq!(canonical, plan);
        }
    }

    #[test]
    fn semantic_equivalence_covers_shape_variants_only() {
        let chain = QueryPlan::FindObject {
            segments: vec!["张三".to_string(), "师傅".to_string()],
        };
        let direct = QueryPlan::FindObjectByPredicate {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
        };
        assert!(plans_equivalent(&chain, &direct));
        let check_triple = QueryPlan::CheckTriple {
            subject_segments: vec!["张三".to_string(), "师傅".to_string()],
            object: "李四".to_string(),
        };
        let check_relation = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            object: "李四".to_string(),
            negated: false,
        };
        assert!(plans_equivalent(&check_triple, &check_relation));
        let reverse = QueryPlan::FindSubject {
            object: "李四".to_string(),
            predicate: "师傅".to_string(),
        };
        assert!(!plans_equivalent(&chain, &reverse));
    }

    #[test]
    fn mining_generalizes_over_predicates() {
        let episodes = seed_episodes();
        let lexicon = seed_lexicon();
        let book = mine(&episodes, &episodes, &lexicon, 2);
        assert!(!book.is_empty(), "种子语料应挖出规则");
        let rule = book
            .rules()
            .find(|rule| rule.surface == "{S}到底{P}谁" && rule.canonical == "{S}{P}谁")
            .expect("应挖出「到底」外壳的泛化规则");
        assert!(rule.predicates.len() >= 2);
        assert_eq!(rule.status, RewriteStatus::Proposed);
        assert_eq!(rule.refutations, 0);
    }

    #[test]
    fn fallback_rewrites_held_out_predicate_without_touching_seen_surface() {
        let episodes = seed_episodes();
        let lexicon = seed_lexicon();
        let holdout = default_holdout(&corpus_predicates(&episodes, &lexicon));
        assert!(!holdout.is_empty());
        let train: Vec<RewriteEpisode> = episodes
            .iter()
            .filter(|episode| {
                !episode
                    .predicates
                    .iter()
                    .any(|predicate| holdout.contains(predicate))
            })
            .cloned()
            .collect();
        let book = mine(&train, &episodes, &lexicon, 2);
        let eval = evaluate(&book, &episodes, &lexicon, &holdout);
        assert!(eval.held_out.episodes > 0);
        assert_eq!(eval.held_out.wrong_rewrites, 0, "{}", eval.render_text());
        assert!(
            eval.held_out.rewritten_correct > eval.held_out.baseline_correct,
            "{}",
            eval.render_text()
        );
        assert!(
            eval.held_out.coverage_gain() > 0.0,
            "{}",
            eval.render_text()
        );
    }

    #[test]
    fn negative_surfaces_refute_rules() {
        let mut episodes = seed_episodes();
        episodes.push(RewriteEpisode::negative("张三到底认识谁"));
        let lexicon = EpisodeLexicon::from_episodes(&episodes);
        let book = mine(&episodes, &episodes, &lexicon, 2);
        assert!(
            book.rules().all(|rule| !(rule.surface == "{S}到底{P}谁")),
            "与硬反例同形的规则必须被否决"
        );
    }

    #[test]
    fn proposed_rules_only_apply_when_requested() {
        let episodes = seed_episodes();
        let lexicon = seed_lexicon();
        let book = mine(&episodes, &episodes, &lexicon, 2);
        assert!(book.resolve("张三到底认识谁", &lexicon).is_none());
        let applied = book
            .resolve_with("张三到底认识谁", &lexicon, true)
            .expect("候选规则应能离线重写");
        assert_eq!(applied.text, "张三认识谁");
        assert!(!applied.rules.is_empty());
    }

    #[test]
    fn master_and_retire_are_explicit() {
        let episodes = seed_episodes();
        let lexicon = seed_lexicon();
        let mut book = mine(&episodes, &episodes, &lexicon, 2);
        let id = book
            .rules()
            .find(|rule| rule.surface == "{S}到底{P}谁")
            .expect("应挖出规则")
            .id
            .clone();
        book.adopt(&id).expect("应可采纳");
        assert!(book.resolve("张三到底认识谁", &lexicon).is_some());
        book.retire(&id).expect("应可退役");
        assert!(book.resolve("张三到底认识谁", &lexicon).is_none());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let episodes = seed_episodes();
        let lexicon = seed_lexicon();
        let book = mine(&episodes, &episodes, &lexicon, 2);
        let directory =
            std::env::temp_dir().join(format!("knowprolog-rewrite-book-{}", std::process::id()));
        let path = directory.join("book.json");
        book.save_to_path(&path).expect("应可保存");
        let restored = RewriteBook::load_from_path(&path).expect("应可恢复");
        assert_eq!(book, restored);
        let _ = fs::remove_dir_all(directory);
    }
}

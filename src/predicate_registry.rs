//! PPR：谓词画像注册表（Predicate Profile Registry）。
//!
//! 谓词是一等公民：规范名 + 别名 + 互逆关系 + 功能/对称/传递性质 +
//! 观测统计（主语/宾语分布、支撑数、来源）。它是导入、生成、查询与推理共同
//! 消费的一本词典：
//!   - 导入准入：未登记谓词先登记为「待确认」画像，不静默落库；
//!   - 别名归一：同义说法命中同一事实；
//!   - 逆向归一：互逆谓词（师傅/徒弟）按统一方向落库；
//!   - 冲突裁决：功能谓词（每人只有一个取值）的新旧值冲突不覆盖、只标记；
//!   - 画像发现：从观测统计里挖功能/对称/互逆候选（复用现有统计口径，不引入新魔法数字）。
//!
//! 全部纯符号、可审计：每个字段都能追溯到配置声明或某条来源事实。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::type_system::{TypeSystem, TypeVerdict};

/// config.toml 中的 `[[predicate_profiles]]` 声明。
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PredicateProfileConfig {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub inverse: Option<String>,
    #[serde(default)]
    pub functional: bool,
    #[serde(default)]
    pub transitive: bool,
    #[serde(default)]
    pub symmetric: bool,
    /// 定义域：主语允许的实体类型（空 = 不限制）
    #[serde(default)]
    pub domain: Vec<String>,
    /// 值域：宾语允许的实体类型（空 = 不限制）
    #[serde(default)]
    pub range: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// 定义域/值域违规：实体类型已知且与声明类型不相交时拒绝落库。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaViolation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// 违规的一侧："定义域"（主语）或 "值域"（宾语）
    pub slot: String,
    /// 声明允许的类型
    pub expected: Vec<String>,
    /// 实体实际已有的类型
    pub actual: Vec<String>,
}

/// 谓词解析结果：规范谓词 + 是否需要翻转主宾语（逆向谓词）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateResolution {
    pub canonical: String,
    pub inverted: bool,
    pub known: bool,
}

impl PredicateResolution {
    fn unknown(predicate: &str) -> Self {
        Self {
            canonical: predicate.to_string(),
            inverted: false,
            known: false,
        }
    }
}

/// 单个谓词的画像。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateProfile {
    pub canonical: String,
    #[serde(default)]
    pub aliases: BTreeSet<String>,
    #[serde(default)]
    pub inverse: Option<String>,
    #[serde(default)]
    pub functional: bool,
    #[serde(default)]
    pub transitive: bool,
    #[serde(default)]
    pub symmetric: bool,
    /// 定义域：主语允许的实体类型（空 = 不限制）
    #[serde(default)]
    pub domain: BTreeSet<String>,
    /// 值域：宾语允许的实体类型（空 = 不限制）
    #[serde(default)]
    pub range: BTreeSet<String>,
    /// 由配置声明或人工确认（可稳定依赖）
    #[serde(default)]
    pub declared: bool,
    /// 自动准入、尚未确认（严格准入下的事实先不落库）
    #[serde(default)]
    pub provisional: bool,
    #[serde(default)]
    pub support: usize,
    #[serde(default)]
    pub subjects: BTreeSet<String>,
    #[serde(default)]
    pub objects: BTreeSet<String>,
    /// 主语 → 取值集合（功能谓词冲突检测与候选发现的原始统计）
    #[serde(default)]
    pub values: BTreeMap<String, BTreeSet<String>>,
    /// 来源：配置声明 / 文本导入 / 知识图谱 / 人工确认…
    #[serde(default)]
    pub sources: BTreeSet<String>,
    #[serde(default)]
    pub description: Option<String>,
}

impl PredicateProfile {
    fn new(canonical: &str) -> Self {
        Self {
            canonical: canonical.to_string(),
            aliases: BTreeSet::new(),
            inverse: None,
            functional: false,
            transitive: false,
            symmetric: false,
            domain: BTreeSet::new(),
            range: BTreeSet::new(),
            declared: false,
            provisional: false,
            support: 0,
            subjects: BTreeSet::new(),
            objects: BTreeSet::new(),
            values: BTreeMap::new(),
            sources: BTreeSet::new(),
            description: None,
        }
    }

    /// 单值主语占比达到 90% 且主语数 ≥3（与溯因模块的功能谓词判定同口径）
    fn functional_evidence(&self) -> Option<(usize, usize, usize)> {
        let subjects = self.values.len();
        if subjects < 3 {
            return None;
        }
        let single = self
            .values
            .values()
            .filter(|objects| objects.len() == 1)
            .count();
        if single as f64 / subjects as f64 >= 0.9 {
            return Some((subjects, single, self.support));
        }
        None
    }
}

/// 谓词画像注册表。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredicateRegistry {
    #[serde(default)]
    profiles: BTreeMap<String, PredicateProfile>,
    /// 别名 → (规范谓词, 是否逆向)。
    /// 互逆对的存储槽位由先登记画像的一方定格，另一方与它的别名都以
    /// 「逆向别名」的身份绑到同一槽位上——同一事实只落一个方向。
    #[serde(default)]
    alias_index: BTreeMap<String, (String, bool)>,
    /// 规范谓词 → 互逆谓词（双向登记，槽位画像间的关系记录）
    #[serde(default)]
    inverse_index: BTreeMap<String, String>,
    /// 已观测的规范事实 (主语, 规范谓词, 宾语)：支撑数去重的唯一依据
    #[serde(default)]
    facts: BTreeSet<(String, String, String)>,
    /// 冲突笔记：主语 → 规范谓词 → 被拒的入值集合（功能谓词冲突不覆盖、只登记）。
    /// 图谱快照看不到「被拒的值」，质量护照靠这本笔记披露「存在不同说法」。
    #[serde(default)]
    conflict_notes: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    /// 严格准入：未登记谓词不自动准入，也不落库
    #[serde(default)]
    strict_admission: bool,
}

impl PredicateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_strict_admission(&mut self, strict: bool) {
        self.strict_admission = strict;
    }

    pub fn strict_admission(&self) -> bool {
        self.strict_admission
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn canonical_names(&self) -> Vec<&str> {
        self.profiles.keys().map(String::as_str).collect()
    }

    pub fn profile(&self, predicate: &str) -> Option<&PredicateProfile> {
        let resolution = self.resolve(predicate);
        if !resolution.known || resolution.inverted {
            return None;
        }
        self.profiles.get(&resolution.canonical)
    }

    /// 画像查询：返回（规范画像, 是否以逆向身份命中）
    pub fn lookup(&self, predicate: &str) -> Option<(&PredicateProfile, bool)> {
        let resolution = self.resolve(predicate);
        if !resolution.known {
            return None;
        }
        self.profiles
            .get(&resolution.canonical)
            .map(|profile| (profile, resolution.inverted))
    }

    pub fn is_known(&self, predicate: &str) -> bool {
        self.resolve(predicate).known
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.profile(predicate).map_or(false, |p| p.functional)
    }

    pub fn is_transitive(&self, predicate: &str) -> bool {
        self.profile(predicate).map_or(false, |p| p.transitive)
    }

    pub fn is_symmetric(&self, predicate: &str) -> bool {
        self.profile(predicate).map_or(false, |p| p.symmetric)
    }

    /// 解析谓词：别名优先，其次规范画像，最后互逆索引。
    pub fn resolve(&self, predicate: &str) -> PredicateResolution {
        let predicate = predicate.trim();
        if let Some((canonical, inverted)) = self.alias_index.get(predicate) {
            return PredicateResolution {
                canonical: canonical.clone(),
                inverted: *inverted,
                known: true,
            };
        }
        if self.profiles.contains_key(predicate) {
            return PredicateResolution {
                canonical: predicate.to_string(),
                inverted: false,
                known: true,
            };
        }
        if let Some(canonical) = self.inverse_index.get(predicate) {
            if self.profiles.contains_key(canonical) {
                return PredicateResolution {
                    canonical: canonical.clone(),
                    inverted: true,
                    known: true,
                };
            }
        }
        PredicateResolution::unknown(predicate)
    }

    pub fn inverse_of(&self, predicate: &str) -> Option<&str> {
        let resolution = self.resolve(predicate);
        if !resolution.known {
            return None;
        }
        if resolution.inverted {
            // 逆向命中时，其互逆谓词就是规范谓词自身
            return Some(self.profiles.get(&resolution.canonical)?.canonical.as_str());
        }
        self.inverse_index.get(&resolution.canonical).map(String::as_str)
    }

    /// 别名列表（不包含规范名自身）
    pub fn aliases_of(&self, predicate: &str) -> Vec<String> {
        self.profile(predicate)
            .map(|profile| profile.aliases.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 全部（词面 → 规范谓词, 是否逆向）映射：别名与互逆词面共用一张表。
    /// 逆向词面（例如「徒弟」、"学生"）命中时主宾语需按槽位换向。
    pub fn alias_pairs(&self) -> Vec<(String, String, bool)> {
        self.alias_index
            .iter()
            .map(|(alias, (canonical, inverted))| {
                (alias.clone(), canonical.clone(), *inverted)
            })
            .collect()
    }

    /// 待确认的自动准入谓词
    pub fn pending_admissions(&self) -> Vec<String> {
        self.profiles
            .values()
            .filter(|profile| profile.provisional)
            .map(|profile| profile.canonical.clone())
            .collect()
    }

    fn ensure_profile(&mut self, canonical: &str, source: &str) -> &mut PredicateProfile {
        let profile = self
            .profiles
            .entry(canonical.to_string())
            .or_insert_with(|| PredicateProfile::new(canonical));
        if !source.is_empty() {
            profile.sources.insert(source.to_string());
        }
        profile
    }

    /// 配置声明：规范名 + 别名 + 互逆 + 性质。
    ///
    /// 配置名若已作为互逆别名绑定某槽位（例如「徒弟」先由「师傅」的声明绑定），
    /// 本次声明的性质会并入槽位画像——事实只落一个方向，这是设计意图。
    /// 别名一律以「配置名」本身登记：register_alias 会解析它是正/逆向坐标，
    ///从而使「学生」这类互逆谓词的别名也带上逆向标记。
    pub fn declare(&mut self, config: &PredicateProfileConfig) -> bool {
        let name = config.name.trim();
        if name.is_empty() {
            return false;
        }
        let resolution = self.resolve(name);
        let target = if resolution.known {
            resolution.canonical.clone()
        } else {
            name.to_string()
        };
        let existed = self.profiles.contains_key(&target);
        {
            let profile = self.ensure_profile(&target, "配置声明");
            profile.declared = true;
            profile.provisional = false;
            profile.functional |= config.functional;
            profile.transitive |= config.transitive;
            profile.symmetric |= config.symmetric;
            profile.domain.extend(
                config
                    .domain
                    .iter()
                    .map(|item| item.trim().to_string())
                    .filter(|item| !item.is_empty()),
            );
            profile.range.extend(
                config
                    .range
                    .iter()
                    .map(|item| item.trim().to_string())
                    .filter(|item| !item.is_empty()),
            );
            if let Some(description) = &config.description {
                profile.description = Some(description.clone());
            }
        }
        for alias in &config.aliases {
            self.register_alias(alias, name, "配置声明");
        }
        if let Some(inverse) = &config.inverse {
            self.register_inverse(&target, inverse, "配置声明");
        }
        !existed
    }

    /// 自动准入：未知谓词登记为「待确认」画像，返回是否新登记。
    pub fn admit(&mut self, predicate: &str, source: &str) -> bool {
        let predicate = predicate.trim();
        if predicate.is_empty() || self.resolve(predicate).known {
            return false;
        }
        let profile = self.ensure_profile(predicate, source);
        profile.provisional = true;
        true
    }

    /// 人工确认待定谓词：转为已声明画像。
    pub fn confirm(&mut self, predicate: &str) -> bool {
        let resolution = self.resolve(predicate);
        let changed = match self.profiles.get_mut(&resolution.canonical) {
            Some(profile) if profile.provisional => {
                profile.provisional = false;
                profile.declared = true;
                profile.sources.insert("人工确认".to_string());
                true
            }
            _ => false,
        };
        changed
    }

    /// 拒绝待定谓词：移除画像、别名与观测事实，返回是否移除。
    pub fn reject(&mut self, predicate: &str) -> bool {
        let resolution = self.resolve(predicate);
        let Some(profile) = self.profiles.get(&resolution.canonical) else {
            return false;
        };
        if !profile.provisional {
            return false;
        }
        let canonical = resolution.canonical.clone();
        let inverse = profile.inverse.clone();
        self.profiles.remove(&canonical);
        self.alias_index.retain(|_, (target, _)| target != &canonical);
        self.inverse_index.remove(&canonical);
        if let Some(inverse) = inverse {
            self.inverse_index.remove(&inverse);
        }
        self.facts.retain(|(_, predicate, _)| predicate != &canonical);
        self.conflict_notes.retain(|_, by_predicate| {
            by_predicate.remove(&canonical);
            !by_predicate.is_empty()
        });
        true
    }

    /// 登记别名 → 规范谓词；若别名已有画像，统计并入规范画像。
    pub fn register_alias(&mut self, alias: &str, canonical: &str, source: &str) -> bool {
        let alias = alias.trim();
        let canonical_input = canonical.trim();
        if alias.is_empty() || canonical_input.is_empty() || alias == canonical_input {
            return false;
        }
        let target = self.resolve(canonical_input);
        let target_key = (target.canonical.clone(), target.inverted);
        if self.alias_index.get(alias) == Some(&target_key) {
            return false;
        }

        let mut inherited: Vec<String> = Vec::new();
        if let Some(old) = self.profiles.remove(alias) {
            self.merge_profile_into(&target.canonical, old, target.inverted, alias);
            inherited = self
                .alias_index
                .iter()
                .filter(|(_, (target_name, _))| target_name == alias)
                .map(|(alias_name, _)| alias_name.clone())
                .collect();
        } else {
            let profile = self.ensure_profile(&target.canonical, source);
            profile.aliases.insert(alias.to_string());
            if !source.is_empty() {
                profile.sources.insert(source.to_string());
            }
        }
        for alias_name in &inherited {
            self.alias_index
                .insert(alias_name.clone(), (target.canonical.clone(), target.inverted));
        }
        self.alias_index
            .insert(alias.to_string(), (target.canonical, target.inverted));
        true
    }

    /// 把被合并画像的统计并入目标画像；逆转方向时交换主宾语。
    fn merge_profile_into(
        &mut self,
        canonical: &str,
        old: PredicateProfile,
        inverted: bool,
        alias: &str,
    ) {
        // 先重键别名下观测过的事实
        let moved: Vec<(String, String, String)> = self
            .facts
            .iter()
            .filter(|(_, predicate, _)| predicate == alias)
            .cloned()
            .collect();
        for (subject, _predicate, object) in moved {
            self.facts.remove(&(subject.clone(), alias.to_string(), object.clone()));
            if inverted {
                self.facts.insert((object, canonical.to_string(), subject));
            } else {
                self.facts.insert((subject, canonical.to_string(), object));
            }
        }
        let profile = self.ensure_profile(canonical, "");
        profile.support += old.support;
        profile.functional |= old.functional;
        profile.transitive |= old.transitive;
        profile.symmetric |= old.symmetric;
        if inverted {
            profile.domain.extend(old.range.iter().cloned());
            profile.range.extend(old.domain.iter().cloned());
        } else {
            profile.domain.extend(old.domain.iter().cloned());
            profile.range.extend(old.range.iter().cloned());
        }
        profile.declared |= old.declared;
        profile.provisional &= old.provisional;
        profile.sources.extend(old.sources.iter().cloned());
        profile.aliases.insert(alias.to_string());
        profile.aliases.extend(old.aliases.iter().cloned());
        if inverted {
            profile.subjects.extend(old.objects.iter().cloned());
            profile.objects.extend(old.subjects.iter().cloned());
            for (subject, objects) in &old.values {
                for object in objects {
                    profile
                        .values
                        .entry(object.clone())
                        .or_default()
                        .insert(subject.clone());
                }
            }
        } else {
            profile.subjects.extend(old.subjects.iter().cloned());
            profile.objects.extend(old.objects.iter().cloned());
            for (subject, objects) in &old.values {
                profile
                    .values
                    .entry(subject.clone())
                    .or_default()
                    .extend(objects.iter().cloned());
            }
        }
    }

    /// 登记互逆谓词（同义换向）：p(a,b) ≡ q(b,a)。
    ///
    /// 互逆对的「存储槽位」按可确定性选取：已声明画像者优先、已观测支撑者
    /// 优先、以上都平局则取左侧。非槽位词面与它的全部别名以「逆向别名」的
    /// 身份绑到槽位上，挑好吃的观测统计与事实会翻转方向并入槽位——同一
    /// 事实永远只落一个规范方向。
    pub fn register_inverse(&mut self, left: &str, right: &str, source: &str) -> bool {
        let left = left.trim();
        let right = right.trim();
        if left.is_empty() || right.is_empty() || left == right {
            return false;
        }
        let left_resolution = self.resolve(left);
        let right_resolution = self.resolve(right);
        let left_canonical = left_resolution.canonical.clone();
        let right_canonical = right_resolution.canonical.clone();
        if left_canonical == right_canonical {
            // 双方别名或互逆已指向同一槽位：称谓不同，语义同向，无需登记
            return false;
        }
        // 槽位裁决：声明画像 > 已观测支撑；全部平局取左侧
        let left_declared = self
            .profiles
            .get(&left_canonical)
            .map_or(false, |profile| profile.declared);
        let right_declared = self
            .profiles
            .get(&right_canonical)
            .map_or(false, |profile| profile.declared);
        let left_support = self
            .profiles
            .get(&left_canonical)
            .map_or(0, |profile| profile.support);
        let right_support = self
            .profiles
            .get(&right_canonical)
            .map_or(0, |profile| profile.support);
        let (slot, counterpart) = if left_declared && !right_declared {
            (left_canonical.clone(), right_canonical.clone())
        } else if right_declared && !left_declared {
            (right_canonical.clone(), left_canonical.clone())
        } else if right_support > left_support {
            (right_canonical.clone(), left_canonical.clone())
        } else {
            (left_canonical.clone(), right_canonical.clone())
        };

        let changed = self.inverse_index.get(&slot) != Some(&counterpart);
        self.inverse_index
            .insert(slot.clone(), counterpart.clone());
        self.inverse_index
            .insert(counterpart.clone(), slot.clone());
        // 非槽位词面以逆向别名并入槽位：画像/支撑/别名一并迁移（必要时翻转）
        self.bind_inverse_alias(&counterpart, &slot, source);
        {
            let profile = self.ensure_profile(&slot, source);
            profile.inverse = Some(counterpart.clone());
        }
        changed
    }

    /// 把 `alias` 绑成 `slot` 的逆向别名：p(alias 的坐标) ≡ slot(翻转)。
    /// 若 alias 已有画像（作为独立谓词观测过），其支撑/别名/事实一并翻转并入槽位。
    fn bind_inverse_alias(&mut self, alias: &str, slot: &str, source: &str) {
        if alias.trim().is_empty() || slot.trim().is_empty() || alias == slot {
            return;
        }
        let source = if source.trim().is_empty() {
            "互逆登记"
        } else {
            source.trim()
        };
        if let Some(old) = self.profiles.remove(alias) {
            self.merge_profile_into(slot, old, true, alias);
            let profile = self.ensure_profile(slot, "");
            profile.sources.insert(source.to_string());
        } else {
            let profile = self.ensure_profile(slot, source);
            profile.aliases.insert(alias.to_string());
        }
        // alias 名下原有的其它别名 → 以逆向坐标重新指向槽位
        let descendants: Vec<String> = self
            .alias_index
            .iter()
            .filter(|(_, (canonical, _))| canonical == alias)
            .map(|(surface, _)| surface.clone())
            .collect();
        for surface in descendants {
            self.alias_index.insert(surface, (slot.to_string(), true));
        }
        self.alias_index
            .insert(alias.to_string(), (slot.to_string(), true));
    }

    /// 标记功能谓词：同主语只允许一个取值。
    pub fn mark_functional(&mut self, predicate: &str, functional: bool) -> bool {
        let resolution = self.resolve(predicate);
        if !resolution.known {
            return false;
        }
        let profile = self.ensure_profile(&resolution.canonical, "人工确认");
        let changed = profile.functional != functional;
        profile.functional = functional;
        if changed && !functional {
            profile.sources.insert("取消功能谓词".to_string());
        }
        changed
    }

    /// 观测一条事实：非规范谓词先解析（别名/逆向）再归一落账。
    pub fn observe(&mut self, subject: &str, predicate: &str, object: &str, source: Option<&str>) {
        let subject = subject.trim();
        let object = object.trim();
        let predicate = predicate.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        let resolution = self.resolve(predicate);
        let (subject, object) = if resolution.inverted {
            (object, subject)
        } else {
            (subject, object)
        };
        let key = (
            subject.to_string(),
            resolution.canonical.clone(),
            object.to_string(),
        );
        if !self.facts.insert(key) {
            return;
        }
        let profile = self.ensure_profile(&resolution.canonical, "");
        profile.support += 1;
        profile.subjects.insert(subject.to_string());
        profile.objects.insert(object.to_string());
        profile
            .values
            .entry(subject.to_string())
            .or_default()
            .insert(object.to_string());
        if let Some(source) = source {
            if !source.is_empty() {
                profile.sources.insert(source.to_string());
            }
        }
    }

    /// 撤回一条已观测事实（撤销导入用）。
    pub fn retract(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let resolution = self.resolve(predicate);
        let (subject, object) = if resolution.inverted {
            (object, subject)
        } else {
            (subject, object)
        };
        let removed = self.facts.remove(&(
            subject.to_string(),
            resolution.canonical.clone(),
            object.to_string(),
        ));
        if removed {
            self.recompute(&resolution.canonical);
        }
        removed
    }

    /// 按现存事实重算画像统计。
    fn recompute(&mut self, canonical: &str) {
        let facts: Vec<(String, String, String)> = self
            .facts
            .iter()
            .filter(|(_, predicate, _)| predicate == canonical)
            .cloned()
            .collect();
        let Some(profile) = self.profiles.get_mut(canonical) else {
            return;
        };
        profile.support = facts.len();
        profile.subjects.clear();
        profile.objects.clear();
        profile.values.clear();
        for (subject, _, object) in facts {
            profile.subjects.insert(subject.clone());
            profile.objects.insert(object.clone());
            profile.values.entry(subject).or_default().insert(object);
        }
    }

    /// 功能谓词冲突检测：返回同主语已存在的其它取值。
    pub fn conflict(&self, subject: &str, predicate: &str, object: &str) -> Option<Vec<String>> {
        let resolution = self.resolve(predicate);
        let profile = self.profiles.get(&resolution.canonical)?;
        if !profile.functional {
            return None;
        }
        let (subject, object) = if resolution.inverted {
            (object, subject)
        } else {
            (subject, object)
        };
        let existing = profile.values.get(subject)?;
        if existing.contains(object) || existing.is_empty() {
            return None;
        }
        let mut existing: Vec<String> = existing.iter().cloned().collect();
        existing.sort();
        Some(existing)
    }

    /// 登记一条功能谓词冲突笔记（被拒的入值，不覆盖原值）。
    /// 主宾语按互逆归一方向记录，与图谱中的规范方向一致。
    pub fn note_conflict(&mut self, subject: &str, predicate: &str, incoming: &str) {
        let subject = subject.trim();
        let incoming = incoming.trim();
        let predicate = predicate.trim();
        if subject.is_empty() || predicate.is_empty() || incoming.is_empty() {
            return;
        }
        let resolution = self.resolve(predicate);
        let (subject, incoming) = if resolution.inverted {
            (incoming, subject)
        } else {
            (subject, incoming)
        };
        self.conflict_notes
            .entry(subject.to_string())
            .or_default()
            .entry(resolution.canonical)
            .or_default()
            .insert(incoming.to_string());
    }

    /// 全部冲突笔记 → 回执式冲突清单（现有值取自画像统计，确定性排序）。
    pub fn conflicts(&self) -> Vec<ImportConflict> {
        let mut conflicts = Vec::new();
        for (subject, by_predicate) in &self.conflict_notes {
            for (predicate, incoming) in by_predicate {
                let existing: Vec<String> = self
                    .profiles
                    .get(predicate)
                    .and_then(|profile| profile.values.get(subject))
                    .map(|values| values.iter().cloned().collect())
                    .unwrap_or_default();
                for value in incoming {
                    conflicts.push(ImportConflict {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        existing: existing.clone(),
                        incoming: value.clone(),
                    });
                }
            }
        }
        conflicts
    }

    /// 冲突笔记条数（被拒入值的总条数）。
    pub fn conflict_count(&self) -> usize {
        self.conflict_notes
            .values()
            .map(|by_predicate| by_predicate.values().map(BTreeSet::len).sum::<usize>())
            .sum()
    }

    /// 某主语是否存在任何冲突笔记。
    pub fn has_conflict(&self, subject: &str) -> bool {
        self.conflict_notes.contains_key(subject.trim())
    }

    /// 某（主语, 谓词）被拒入值清单（排序）；无笔记返回 None。
    pub fn conflict_incoming(&self, subject: &str, predicate: &str) -> Option<Vec<String>> {
        let resolution = self.resolve(predicate);
        let incoming = self
            .conflict_notes
            .get(subject.trim())?
            .get(&resolution.canonical)?;
        Some(incoming.iter().cloned().collect())
    }

    /// 定义域/值域校验（开放世界）：实体类型未知（空集）时放行；
    /// 类型已知但与声明的允许类型完全不相交时判违规。互逆谓词先翻转坐标。
    pub fn check_schema(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        subject_types: &BTreeSet<String>,
        object_types: &BTreeSet<String>,
    ) -> Option<SchemaViolation> {
        let resolution = self.resolve(predicate);
        let profile = self.profiles.get(&resolution.canonical)?;
        let (subject, object, subject_types, object_types) = if resolution.inverted {
            (object, subject, object_types, subject_types)
        } else {
            (subject, object, subject_types, object_types)
        };
        if !profile.domain.is_empty()
            && !subject_types.is_empty()
            && subject_types.is_disjoint(&profile.domain)
        {
            return Some(SchemaViolation {
                subject: subject.to_string(),
                predicate: resolution.canonical.clone(),
                object: object.to_string(),
                slot: "定义域".to_string(),
                expected: profile.domain.iter().cloned().collect(),
                actual: subject_types.iter().cloned().collect(),
            });
        }
        if !profile.range.is_empty()
            && !object_types.is_empty()
            && object_types.is_disjoint(&profile.range)
        {
            return Some(SchemaViolation {
                subject: subject.to_string(),
                predicate: resolution.canonical.clone(),
                object: object.to_string(),
                slot: "值域".to_string(),
                expected: profile.range.iter().cloned().collect(),
                actual: object_types.iter().cloned().collect(),
            });
        }
        None
    }

    /// 闭包档定义域/值域校验（需类型体系）：别名/子类闭包相容即放行；
    /// 只有闭包无交集且存在互斥证据（[`TypeVerdict::Disjoint`]）才判违规。
    /// 开放世界语义不变：类型未知、或两类型只是「互不相关」时一律放行。
    pub fn check_schema_with_types(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        subject_types: &BTreeSet<String>,
        object_types: &BTreeSet<String>,
        types: &TypeSystem,
    ) -> Option<SchemaViolation> {
        let resolution = self.resolve(predicate);
        let profile = self.profiles.get(&resolution.canonical)?;
        let (subject, object, subject_types, object_types) = if resolution.inverted {
            (object, subject, object_types, subject_types)
        } else {
            (subject, object, subject_types, object_types)
        };
        if !profile.domain.is_empty()
            && types.sets_admit(subject_types, &profile.domain) == TypeVerdict::Disjoint
        {
            return Some(SchemaViolation {
                subject: subject.to_string(),
                predicate: resolution.canonical.clone(),
                object: object.to_string(),
                slot: "定义域".to_string(),
                expected: profile.domain.iter().cloned().collect(),
                actual: subject_types.iter().cloned().collect(),
            });
        }
        if !profile.range.is_empty()
            && types.sets_admit(object_types, &profile.range) == TypeVerdict::Disjoint
        {
            return Some(SchemaViolation {
                subject: subject.to_string(),
                predicate: resolution.canonical.clone(),
                object: object.to_string(),
                slot: "值域".to_string(),
                expected: profile.range.iter().cloned().collect(),
                actual: object_types.iter().cloned().collect(),
            });
        }
        None
    }

    /// 功能谓词候选：≥3 主语且 90% 以上主语只有唯一取值（未标记功能的才建议）。
    pub fn functional_candidates(&self) -> Vec<String> {
        let mut candidates = Vec::new();
        for (name, profile) in &self.profiles {
            if profile.functional {
                continue;
            }
            if profile.functional_evidence().is_some() {
                candidates.push(name.clone());
            }
        }
        candidates
    }

    /// 对称谓词候选：p(a,b) 与 p(b,a) 同时对出现 ≥2 组。
    pub fn symmetric_candidates(&self) -> Vec<String> {
        let mut candidates = Vec::new();
        for (name, profile) in &self.profiles {
            if profile.symmetric {
                continue;
            }
            let mut reversed = 0usize;
            for (subject, objects) in &profile.values {
                for object in objects {
                    if object == subject {
                        continue;
                    }
                    if profile
                        .values
                        .get(object)
                        .map_or(false, |reverse| reverse.contains(subject))
                    {
                        reversed += 1;
                    }
                }
            }
            if reversed >= 2 {
                candidates.push(name.clone());
            }
        }
        candidates
    }

    /// 互逆谓词候选：p(a,b) 与 q(b,a) 同时成立的组数 ≥2。
    pub fn inverse_candidates(&self) -> Vec<(String, String, usize)> {
        let names: Vec<String> = self.profiles.keys().cloned().collect();
        let mut candidates = Vec::new();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let left = &names[i];
                let right = &names[j];
                if self.inverse_index.get(left) == Some(right)
                    || self.inverse_index.get(right) == Some(left)
                {
                    continue;
                }
                let left_profile = &self.profiles[left];
                let right_profile = &self.profiles[right];
                let mut overlap = 0usize;
                for (subject, objects) in &left_profile.values {
                    for object in objects {
                        if right_profile
                            .values
                            .get(object)
                            .map_or(false, |reverse| reverse.contains(subject))
                        {
                            overlap += 1;
                        }
                    }
                }
                if overlap >= 2 {
                    candidates.push((left.clone(), right.clone(), overlap));
                }
            }
        }
        candidates.sort();
        candidates
    }

    /// 谓词画像报告（确定性输出，便于回归对比）。
    pub fn report(&self) -> String {
        if self.profiles.is_empty() {
            return "谓词画像：暂无登记（可用 config.toml 的 [[predicate_profiles]] 声明，或在导入中自动准入）"
                .to_string();
        }
        let pending = self.pending_admissions().len();
        let mut output = format!(
            "谓词画像：共 {} 个规范谓词{}。\n",
            self.profiles.len(),
            if pending == 0 {
                String::new()
            } else {
                format!("，其中 {} 个待确认", pending)
            }
        );
        for (name, profile) in &self.profiles {
            let mut traits: Vec<String> = Vec::new();
            if profile.functional {
                traits.push("功能".to_string());
            }
            if profile.transitive {
                traits.push("传递".to_string());
            }
            if profile.symmetric {
                traits.push("对称".to_string());
            }
            if let Some(inverse) = &profile.inverse {
                traits.push(format!("互逆「{}」", inverse));
            }
            if !profile.domain.is_empty() {
                traits.push(format!(
                    "定义域「{}」",
                    profile.domain.iter().cloned().collect::<Vec<_>>().join("、")
                ));
            }
            if !profile.range.is_empty() {
                traits.push(format!(
                    "值域「{}」",
                    profile.range.iter().cloned().collect::<Vec<_>>().join("、")
                ));
            }
            if !profile.aliases.is_empty() {
                traits.push(format!(
                    "别名「{}」",
                    profile
                        .aliases
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                ));
            }
            let status = if profile.provisional {
                "待确认"
            } else if profile.declared {
                "已声明"
            } else {
                "自动观测"
            };
            output.push_str(&format!(
                "- {}（支撑 {}；主语 {} 宾语 {}；{}）",
                name,
                profile.support,
                profile.subjects.len(),
                profile.objects.len(),
                status
            ));
            if !traits.is_empty() {
                output.push('；');
                output.push_str(&traits.join("；"));
            }
            if !profile.sources.is_empty() {
                output.push_str(&format!(
                    "；来源：{}",
                    profile
                        .sources
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                ));
            }
            output.push('\n');
        }
        output
    }

    /// 候选发现报告：统计只是提示，需配置或对话确认后才生效。
    pub fn findings_report(&self) -> String {
        let functional = self.functional_candidates();
        let symmetric = self.symmetric_candidates();
        let inverse = self.inverse_candidates();
        if functional.is_empty() && symmetric.is_empty() && inverse.is_empty() {
            return "未发现新的谓词画像候选。".to_string();
        }
        let mut output = String::from("谓词画像候选（统计提示，确认后生效）：\n");
        if !functional.is_empty() {
            output.push_str(&format!(
                "- 功能谓词候选（≥3 主语且 90% 单值）：{}\n",
                functional.join("、")
            ));
        }
        if !symmetric.is_empty() {
            output.push_str(&format!(
                "- 对称谓词候选（双向事实 ≥2 组）：{}\n",
                symmetric.join("、")
            ));
        }
        if !inverse.is_empty() {
            let pairs: Vec<String> = inverse
                .iter()
                .map(|(left, right, overlap)| format!("{} ↔ {}（{} 组）", left, right, overlap))
                .collect();
            output.push_str(&format!("- 互逆谓词候选：{}\n", pairs.join("；")));
        }
        output.push_str(
            "提示：可用「登记别名：A = B」「登记逆谓词：A = B」「功能谓词：A」确认；\n候选项自动统计自现有知识库，不引入任何外部知识。",
        );
        output
    }

    /// 序列化快照（谓词准入与画像跨会话留存）。
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }
}

/// 一次导入的原子动作，供撤销（撤销导入 = 逐条移除这些动作）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportAction {
    /// 实体-实体关系（完整 IRI）
    Named {
        subject: String,
        predicate: String,
        object: String,
    },
    /// 实体-字面量属性（完整 IRI + 字面量值）
    Literal {
        subject: String,
        predicate: String,
        value: String,
    },
}

/// 功能谓词冲突：同主语已有取值，新值未写入。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportConflict {
    pub subject: String,
    pub predicate: String,
    pub existing: Vec<String>,
    pub incoming: String,
}

/// 导入回执：新增/合并/重复/冲突/模式违规/待确认谓词/待准入谓词 + 撤销动作。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportReceipt {
    pub added: usize,
    pub merged: usize,
    pub duplicates: usize,
    pub conflicts: Vec<ImportConflict>,
    #[serde(default)]
    pub schema_violations: Vec<SchemaViolation>,
    pub provisional_predicates: Vec<String>,
    pub pending_admissions: Vec<String>,
    pub actions: Vec<ImportAction>,
    pub registry_changed: bool,
    /// 闭包档相对精确集合判定「由误杀转为放行」的条数（只统计，不改变结果）。
    #[serde(default)]
    pub gate_relaxed: usize,
    /// 闭包档拒绝而精确集合放行的条数（收紧；有互斥证据才会发生）。
    #[serde(default)]
    pub gate_tightened: usize,
}

impl ImportReceipt {
    pub fn is_empty(&self) -> bool {
        self.added == 0
            && self.merged == 0
            && self.duplicates == 0
            && self.conflicts.is_empty()
            && self.schema_violations.is_empty()
            && self.provisional_predicates.is_empty()
            && self.pending_admissions.is_empty()
    }

    pub fn can_undo(&self) -> bool {
        !self.actions.is_empty()
    }

    pub fn summary(&self) -> String {
        let mut output = format!(
            "知识已保存：新增 {} 条，合并 {} 条，重复 {} 条。",
            self.added, self.merged, self.duplicates
        );
        if !self.conflicts.is_empty() {
            output.push_str(&format!(
                "\n功能谓词冲突 {} 处（不覆盖，保留原值）：",
                self.conflicts.len()
            ));
            for conflict in &self.conflicts {
                output.push_str(&format!(
                    "\n  • {}.{} 已有「{}」，新值「{}」未写入",
                    conflict.subject,
                    conflict.predicate,
                    conflict.existing.join("、"),
                    conflict.incoming
                ));
            }
        }
        if !self.schema_violations.is_empty() {
            output.push_str(&format!(
                "\n定义域/值域不符 {} 处（不落库）：",
                self.schema_violations.len()
            ));
            for violation in &self.schema_violations {
                output.push_str(&format!(
                    "\n  • {} -[{}]-> {} 违反{}：允许类型「{}」，实际类型「{}」",
                    violation.subject,
                    violation.predicate,
                    violation.object,
                    violation.slot,
                    violation.expected.join("、"),
                    violation.actual.join("、")
                ));
            }
        }
        if !self.provisional_predicates.is_empty() {
            output.push_str(&format!(
                "\n新谓词 {} 个（自动准入，待确认）：{}\n  可用「确认谓词：名称」转正，或「拒绝谓词：名称」移除。",
                self.provisional_predicates.len(),
                self.provisional_predicates.join("、")
            ));
        }
        if !self.pending_admissions.is_empty() {
            output.push_str(&format!(
                "\n严格准入下未落库的谓词：{}\n  可用「确认谓词：名称」登记后重新导入。",
                self.pending_admissions.join("、")
            ));
        }
        if self.gate_relaxed > 0 || self.gate_tightened > 0 {
            output.push_str(&format!(
                "\n类型闸门（闭包档）：误杀消解 {} 条，收紧 {} 条。",
                self.gate_relaxed, self.gate_tightened
            ));
        }
        if self.can_undo() {
            output.push_str("\n可用「撤销导入」撤回本次导入。");
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(name: &str) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn alias_and_inverse_resolution() {
        let mut registry = PredicateRegistry::new();
        let mut master = config("师傅");
        master.aliases = vec!["老师".to_string()];
        master.inverse = Some("徒弟".to_string());
        master.transitive = true;
        assert!(registry.declare(&master));

        let resolution = registry.resolve("老师");
        assert_eq!(resolution.canonical, "师傅");
        assert!(!resolution.inverted && resolution.known);

        let resolution = registry.resolve("徒弟");
        assert_eq!(resolution.canonical, "师傅");
        assert!(resolution.inverted, "互逆谓词应以逆向命中");

        assert_eq!(registry.inverse_of("师傅"), Some("徒弟"));
        assert_eq!(registry.inverse_of("徒弟"), Some("师傅"));
        assert!(registry.is_transitive("老师"));
    }

    #[test]
    fn inverse_observation_flips_direction() {
        let mut registry = PredicateRegistry::new();
        let mut master = config("师傅");
        master.inverse = Some("徒弟".to_string());
        master.functional = true;
        registry.declare(&master);

        registry.observe("张三", "徒弟", "李四", Some("文本导入"));
        // 徒弟(张三, 李四) ≡ 师傅(李四, 张三)
        let profile = registry.profile("师傅").unwrap();
        assert_eq!(profile.support, 1);
        assert!(profile.values.get("李四").unwrap().contains("张三"));
        assert!(!profile.values.contains_key("张三"));
        assert_eq!(registry.conflict("李四", "师傅", "王五"), Some(vec!["张三".to_string()]));
    }

    #[test]
    fn schema_domain_range_gate_is_open_world() {
        let mut registry = PredicateRegistry::new();
        let mut master = config("师傅");
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        registry.declare(&master);

        let people: BTreeSet<String> = ["人物".to_string()].into_iter().collect();
        let places: BTreeSet<String> = ["地点".to_string()].into_iter().collect();
        let empty = BTreeSet::new();

        assert!(
            registry
                .check_schema("张三", "师傅", "李四", &empty, &empty)
                .is_none(),
            "类型未知时应按开放世界放行"
        );
        assert!(
            registry
                .check_schema("张三", "师傅", "李四", &people, &people)
                .is_none(),
            "类型相符应放行"
        );
        let violation = registry
            .check_schema("北京", "师傅", "李四", &places, &people)
            .expect("定义域不符应判违规");
        assert_eq!(violation.slot, "定义域");
        assert_eq!(violation.expected, vec!["人物".to_string()]);
        assert_eq!(violation.actual, vec!["地点".to_string()]);
        let violation = registry
            .check_schema("张三", "师傅", "北京", &people, &places)
            .expect("值域不符应判违规");
        assert_eq!(violation.slot, "值域");

        let mut master = config("师傅");
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        assert!(
            !registry.declare(&master),
            "重复声明同一画像不新增"
        );
        assert!(registry.report().contains("定义域"));
    }

    #[test]
    fn functional_conflict_requires_flag() {
        let mut registry = PredicateRegistry::new();
        registry.observe("小明", "年龄", "18", None);
        registry.observe("小红", "年龄", "20", None);
        registry.observe("小刚", "年龄", "22", None);
        assert!(
            registry.conflict("小明", "年龄", "19").is_none(),
            "未标记功能谓词时不应判冲突"
        );
        assert!(registry.mark_functional("年龄", true));
        assert_eq!(registry.conflict("小明", "年龄", "19"), Some(vec!["18".to_string()]));
        assert!(registry.conflict("小明", "年龄", "18").is_none());
    }

    #[test]
    fn candidate_mining_reuses_existing_statistics() {
        let mut registry = PredicateRegistry::new();
        for (subject, object) in [("甲", "30"), ("乙", "25"), ("丙", "40")] {
            registry.observe(subject, "年龄", object, None);
        }
        assert_eq!(registry.functional_candidates(), vec!["年龄".to_string()]);

        for (left, right) in [("张三", "李四"), ("王五", "赵六")] {
            registry.observe(left, "同学", right, None);
            registry.observe(right, "同学", left, None);
        }
        assert_eq!(registry.symmetric_candidates(), vec!["同学".to_string()]);

        for (master, disciple) in [("张三", "李四"), ("王五", "赵六")] {
            registry.observe(disciple, "师傅", master, None);
            registry.observe(master, "徒弟", disciple, None);
        }
        let candidates = registry.inverse_candidates();
        assert!(
            candidates.contains(&("师傅".to_string(), "徒弟".to_string(), 2)),
            "应发现互逆候选：{:?}",
            candidates
        );
    }

    #[test]
    fn alias_merge_inherits_profile_statistics() {
        let mut registry = PredicateRegistry::new();
        registry.observe("张三", "老师", "李四", None);
        registry.observe("王五", "老师", "赵六", None);
        assert_eq!(registry.profile("老师").unwrap().support, 2);

        assert!(registry.register_alias("老师", "师傅", "人工确认"));
        assert!(
            !registry.canonical_names().contains(&"老师"),
            "别名不单独成画像"
        );
        let profile = registry.profile("师傅").unwrap();
        assert_eq!(profile.support, 2);
        assert!(profile.aliases.contains("老师"));
        assert!(profile.values.contains_key("张三"));
    }

    #[test]
    fn provisional_admission_confirm_and_reject() {
        let mut registry = PredicateRegistry::new();
        assert!(registry.admit("师从", "文本导入"));
        assert_eq!(registry.pending_admissions(), vec!["师从".to_string()]);
        assert!(registry.confirm("师从"));
        assert!(registry.pending_admissions().is_empty());
        assert!(registry.profile("师从").unwrap().declared);

        assert!(registry.admit("搭伙", "文本导入"));
        registry.observe("甲", "搭伙", "乙", Some("文本导入"));
        assert!(registry.reject("搭伙"));
        assert!(registry.profile("搭伙").is_none());
        assert!(!registry.is_known("搭伙"));
    }

    #[test]
    fn retract_recomputes_support() {
        let mut registry = PredicateRegistry::new();
        registry.observe("甲", "师傅", "乙", None);
        registry.observe("丙", "师傅", "丁", None);
        assert_eq!(registry.profile("师傅").unwrap().support, 2);
        assert!(registry.retract("甲", "师傅", "乙"));
        let profile = registry.profile("师傅").unwrap();
        assert_eq!(profile.support, 1);
        assert!(!profile.subjects.contains("甲"));
    }

    #[test]
    fn snapshot_roundtrip_preserves_profiles() {
        let mut registry = PredicateRegistry::new();
        let mut master = config("师傅");
        master.aliases = vec!["老师".to_string()];
        master.inverse = Some("徒弟".to_string());
        master.functional = true;
        registry.declare(&master);
        registry.observe("张三", "老师", "李四", Some("知识图谱"));
        registry.admit("师从", "文本导入");

        let restored = PredicateRegistry::from_json(&registry.to_json()).unwrap();
        assert_eq!(restored.len(), registry.len());
        assert_eq!(restored.resolve("老师").canonical, "师傅");
        assert!(restored.is_functional("师傅"));
        assert_eq!(restored.pending_admissions(), vec!["师从".to_string()]);
        assert_eq!(restored.profile("师傅").unwrap().support, 1);
        assert!(restored.report().contains("师傅"));
        assert!(restored.findings_report().contains("候选") || restored.findings_report().contains("未发现"));
    }

    #[test]
    fn receipt_summary_reports_every_gate() {
        let receipt = ImportReceipt {
            added: 3,
            merged: 1,
            duplicates: 2,
            conflicts: vec![ImportConflict {
                subject: "小明".to_string(),
                predicate: "年龄".to_string(),
                existing: vec!["18".to_string()],
                incoming: "19".to_string(),
            }],
            schema_violations: vec![SchemaViolation {
                subject: "北京".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                slot: "定义域".to_string(),
                expected: vec!["人物".to_string()],
                actual: vec!["地点".to_string()],
            }],
            provisional_predicates: vec!["师从".to_string()],
            pending_admissions: Vec::new(),
            actions: vec![ImportAction::Literal {
                subject: "s".to_string(),
                predicate: "p".to_string(),
                value: "v".to_string(),
            }],
            registry_changed: true,
            ..Default::default()
        };
        let summary = receipt.summary();
        assert!(summary.contains("新增 3 条"));
        assert!(summary.contains("冲突 1 处"));
        assert!(summary.contains("定义域/值域不符 1 处"));
        assert!(summary.contains("待确认"));
        assert!(summary.contains("撤销导入"));
        assert!(receipt.can_undo());
    }

    use crate::prolog_engine::PrologFact;
    use crate::{KnowledgeItem, OxigraphManager};
    use std::collections::HashMap;

    fn master_profile() -> PredicateProfileConfig {
        let mut master = config("师傅");
        master.aliases = vec!["老师".to_string()];
        master.inverse = Some("徒弟".to_string());
        master.functional = true;
        master
    }

    fn master_disciple_items(from: &str, to: &str) -> Vec<KnowledgeItem> {
        vec![KnowledgeItem::Relationship {
            from_label: "人物".to_string(),
            from_id: from.to_string(),
            to_label: "人物".to_string(),
            to_id: to.to_string(),
            rel_type: "师傅".to_string(),
            properties: HashMap::new(),
        }]
    }

    async fn temp_manager() -> (OxigraphManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = OxigraphManager::new(dir.path().to_str().unwrap()).unwrap();
        (manager, dir)
    }

    /// C10（导入闸门·幂等与撤销）：同文本重复导入不重复落库；
    /// 撤销导入把知识库恢复到导入前快照（包括辅助的 name 三元组）。
    #[tokio::test]
    async fn c10_import_gates_idempotence_and_undo() {
        let (manager, _dir) = temp_manager().await;
        manager.configure_predicates(&[master_profile()], false);

        let before = manager.load_all_relations().await.unwrap();
        let receipt = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "李四"))
            .await
            .unwrap();
        assert!(receipt.added > 0 && !receipt.actions.is_empty());
        let after = manager.load_all_relations().await.unwrap();
        assert_eq!(after.len(), before.len() + 1);

        // 同一文本再次导入：全部命中去重，不新增任何事实
        let replay = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "李四"))
            .await
            .unwrap();
        assert_eq!(replay.added, 0);
        assert!(replay.can_undo() == false || replay.actions.is_empty());
        assert!(replay.duplicates > 0);
        assert_eq!(manager.load_all_relations().await.unwrap(), after);

        // 撤销第一笔导入：动作逐条回撤，知识库回到导入前快照
        let removed = manager.undo_import(&receipt).await.unwrap();
        assert_eq!(removed, receipt.actions.len());
        assert_eq!(
            manager.load_all_relations().await.unwrap(),
            before
        );
    }

    /// C11（导入闸门·功能谓词冲突）：已确认唯一值的谓词出现矛盾值时不覆盖，
    /// 现有值保留在图谱中，回执明确指出冲突。
    #[tokio::test]
    async fn c11_import_gates_functional_conflicts_do_not_overwrite() {
        let (manager, _dir) = temp_manager().await;
        manager.configure_predicates(&[master_profile()], false);

        let first = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "李四"))
            .await
            .unwrap();
        assert!(first.conflicts.is_empty(), "{:?}", first.conflicts);

        // 与已存值矛盾的导入：关系被拒之门外（仅允许新实体补 name 辅助三元组），
        // 图谱维持 师傅(张三)=李四
        let clash = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "王五"))
            .await
            .unwrap();
        assert!(!clash.conflicts.is_empty(), "{:?}", clash.conflicts);
        assert_eq!(
            clash
                .actions
                .iter()
                .filter(|action| matches!(action, ImportAction::Named { .. }))
                .count(),
            0,
            "冲突值不得落库：{:?}",
            clash.actions
        );

        // 图谱维持原有关系
        let relations = manager.load_all_relations().await.unwrap();
        assert!(
            relations.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if rel_type == "师傅" && from == "张三" && to == "李四"
            )),
            "冲突后不得丢失原有关系：{:?}",
            relations
        );
        assert!(
            !relations.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if rel_type == "师傅" && from == "张三" && to == "王五"
            )),
            "冲突值不得改写图谱：{:?}",
            relations
        );
    }

    fn apprentice_items() -> Vec<KnowledgeItem> {
        vec![KnowledgeItem::Relationship {
            from_label: "人物".to_string(),
            from_id: "张三".to_string(),
            to_label: "人物".to_string(),
            to_id: "赵六".to_string(),
            rel_type: "拜师".to_string(),
            properties: HashMap::new(),
        }]
    }

    fn typed_entity(entity_type: &str, name: &str) -> KnowledgeItem {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), name.to_string());
        KnowledgeItem::Entity {
            label: entity_type.to_string(),
            properties,
        }
    }

    /// C13（导入闸门·定义域/值域）：两端实体类型已知且与画像声明不符时
    /// 关系不落库并入回执；类型未知按开放世界放行；类型相符正常落库。
    #[tokio::test]
    async fn c13_import_gates_schema_domain_range() {
        let (manager, _dir) = temp_manager().await;
        let mut master = master_profile();
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        manager.configure_predicates(&[master], false);

        manager
            .save_knowledge_with_receipt(vec![
                typed_entity("人物", "张三"),
                typed_entity("地点", "北京"),
                typed_entity("人物", "李四"),
            ])
            .await
            .unwrap();

        // 值域不符：北京 是「地点」而非「人物」，关系不落库
        let blocked = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "北京"))
            .await
            .unwrap();
        assert_eq!(blocked.schema_violations.len(), 1, "{:?}", blocked);
        assert!(
            blocked
                .actions
                .iter()
                .all(|action| !matches!(action, ImportAction::Named { .. })),
            "违规关系不得落库：{:?}",
            blocked.actions
        );
        assert!(blocked.summary().contains("定义域/值域不符"));
        let relations = manager.load_all_relations().await.unwrap();
        assert!(
            !relations.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { to, .. } if to == "北京"
            )),
            "违规关系不得进入图谱：{:?}",
            relations
        );

        // 类型相符：正常落库
        let admitted = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "李四"))
            .await
            .unwrap();
        assert!(admitted.schema_violations.is_empty());
        assert!(admitted.added > 0, "{:?}", admitted);
    }

    /// C12（导入闸门·严格准入）：未登记谓词在严格模式下不落库；
    /// 经人工登记（别名准入）并确认后再导入即放行。
    #[tokio::test]
    async fn c12_import_gates_strict_admission_requires_confirmation() {
        let (manager, _dir) = temp_manager().await;
        manager.configure_predicates(&[master_profile()], true);

        let held = manager
            .save_knowledge_with_receipt(apprentice_items())
            .await
            .unwrap();
        assert_eq!(held.added, 0, "未确认谓词不得落库：{:?}", held);
        assert_eq!(held.actions.len(), 0);
        assert_eq!(held.pending_admissions, vec!["拜师".to_string()]);
        assert!(manager.load_all_relations().await.unwrap().is_empty());

        // 人工登记别名（谓词候选进入注册表）后即可归一与放行
        assert!(manager.register_predicate_alias("从师", "拜师"));
        assert_eq!(
            manager
                .predicate_query_pairs()
                .into_iter()
                .find(|(surface, _, _)| surface == "从师")
                .map(|(_, slot, _)| slot),
            Some("拜师".to_string())
        );

        let admitted = manager
            .save_knowledge_with_receipt(apprentice_items())
            .await
            .unwrap();
        assert!(admitted.added > 0, "确认后导入应落库：{:?}", admitted);
        assert_eq!(admitted.pending_admissions, Vec::<String>::new());
        let relations = manager.load_all_relations().await.unwrap();
        assert!(
            relations.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { rel_type, .. } if rel_type == "拜师"
            )),
            "确认后关系应落库：{:?}",
            relations
        );
    }

    fn closure_type_system() -> crate::type_system::TypeSystemConfig {
        use crate::type_system::{
            TypeAliasConfig, TypeDisjointConfig, TypeSubclassConfig, TypeSystemConfig,
        };
        TypeSystemConfig {
            closure_gate: true,
            aliases: vec![
                TypeAliasConfig {
                    alias: "Person".to_string(),
                    canonical: "人物".to_string(),
                },
                TypeAliasConfig {
                    alias: "Entity".to_string(),
                    canonical: "实体".to_string(),
                },
            ],
            subclasses: vec![TypeSubclassConfig {
                sub: "人物".to_string(),
                sup: "实体".to_string(),
            }],
            disjoint: vec![TypeDisjointConfig {
                left: "人物".to_string(),
                right: "地点".to_string(),
            }],
        }
    }

    /// C14（类型格×模式闸门）：别名归一消解误杀（Person/人物），
    /// 互斥声明兜底拒绝（地点⊥人物），未知类型仍按开放世界放行；
    /// 与精确集合判定的差异计入回执（gate_relaxed / gate_tightened）。
    #[tokio::test]
    async fn c14_closure_gate_normalizes_aliases_and_keeps_disjoint_guard() {
        let (manager, _dir) = temp_manager().await;
        let mut master = master_profile();
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        manager.configure_predicates(&[master], false);
        manager.configure_type_system(&closure_type_system());

        manager
            .save_knowledge_with_receipt(vec![
                typed_entity("Person", "张三"),
                typed_entity("Person", "李四"),
                typed_entity("地点", "北京"),
                typed_entity("未知类型", "乌云"),
            ])
            .await
            .unwrap();

        // Person = 人物（别名）：精确集合会误杀，闭包档放行并计 relaxed
        let admitted = manager
            .save_knowledge_with_receipt(master_disciple_items("张三", "李四"))
            .await
            .unwrap();
        assert!(
            admitted.schema_violations.is_empty(),
            "别名归一后应放行：{:?}",
            admitted.schema_violations
        );
        assert!(admitted.added > 0, "{:?}", admitted);
        assert_eq!(admitted.gate_relaxed, 1, "{:?}", admitted);
        assert_eq!(admitted.gate_tightened, 0);

        // 北京 是「地点」，与值域「人物」有显式互斥声明：必拒
        let blocked = manager
            .save_knowledge_with_receipt(master_disciple_items("李四", "北京"))
            .await
            .unwrap();
        assert_eq!(blocked.schema_violations.len(), 1, "{:?}", blocked);
        assert_eq!(blocked.schema_violations[0].slot, "值域");
        assert!(
            blocked
                .actions
                .iter()
                .all(|action| !matches!(action, ImportAction::Named { .. })),
            "互斥证据下不得落库：{:?}",
            blocked.actions
        );

        // 未声明类型（乌云）：闭包不可达但无互斥证据 → 开放世界放行
        let unknown = manager
            .save_knowledge_with_receipt(master_disciple_items("李四", "乌云"))
            .await
            .unwrap();
        assert!(
            unknown.schema_violations.is_empty(),
            "未声明类型不得误杀：{:?}",
            unknown.schema_violations
        );
        assert!(unknown.added > 0, "{:?}", unknown);
    }

    /// C15（冲突笔记·在线查询）：功能谓词冲突不覆盖、入注册表，
    /// 供质量护照与对话层披露「存在不同说法」；拒绝谓词时笔记随画像清理。
    #[tokio::test]
    async fn c15_conflict_notes_are_queryable() {
        let (manager, _dir) = temp_manager().await;
        manager.configure_predicates(&[master_profile()], false);

        let mut items = master_disciple_items("张三", "李四");
        let receipt = manager.save_knowledge_with_receipt(items).await.unwrap();
        assert!(receipt.conflicts.is_empty(), "{:?}", receipt.conflicts);

        items = master_disciple_items("张三", "王五");
        let clash = manager.save_knowledge_with_receipt(items).await.unwrap();
        assert_eq!(clash.conflicts.len(), 1, "{:?}", clash);

        let notes = manager.predicate_conflicts();
        assert_eq!(notes.len(), 1, "{:?}", notes);
        assert_eq!(notes[0].subject, "张三");
        assert_eq!(notes[0].predicate, "师傅");
        assert_eq!(notes[0].existing, vec!["李四".to_string()]);
        assert_eq!(notes[0].incoming, "王五");
        let registry = manager.predicate_registry_snapshot();
        assert!(registry.has_conflict("张三"));
        assert_eq!(
            registry.conflict_incoming("张三", "师傅"),
            Some(vec!["王五".to_string()])
        );

        // 快照往返后笔记仍在（跨会话可披露）
        let restored = PredicateRegistry::from_json(&registry.to_json()).unwrap();
        assert_eq!(restored.conflict_count(), 1);

        // 拒绝待定谓词时，冲突笔记随画像一并清理（注册表级路径）
        let mut scratch = PredicateRegistry::new();
        scratch.admit("搭伙", "文本导入");
        scratch.note_conflict("张三", "搭伙", "王五");
        assert_eq!(scratch.conflict_count(), 1);
        assert_eq!(
            scratch.conflict_incoming("张三", "搭伙"),
            Some(vec!["王五".to_string()])
        );
        scratch.reject("搭伙");
        assert_eq!(scratch.conflict_count(), 0, "拒绝谓词后笔记应清理");
    }

    /// C16（类型体系·摘要与闭包判定）：类型格的判定与摘要确定性。
    #[test]
    fn c16_type_lattice_digest_and_closure() {
        let system = TypeSystem::from_config(&closure_type_system());
        assert!(system.is_subclass("Person", "实体"));
        assert!(system.types_disjoint("地点", "人物"));
        assert!(!system.types_disjoint("乌云", "人物"));
        assert_eq!(
            system.sets_admit(
                &["Person".to_string()].into_iter().collect(),
                &["人物".to_string()].into_iter().collect()
            ),
            TypeVerdict::Compatible
        );
        // 摘要锁定：同配置同字节、改一条声明即变化
        let clone = TypeSystem::from_config(&closure_type_system());
        assert_eq!(system.digest(), clone.digest());
        let mut changed = closure_type_system();
        changed.disjoint.clear();
        assert_ne!(system.digest(), TypeSystem::from_config(&changed).digest());
    }
}

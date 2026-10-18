//! TSL：类型体系（Type System / Lattice）。
//!
//! 把「实体类型」升为图谱的脊柱：一张带显式别名、子类边与互斥声明的类型格。
//! 它是模式闸门（`predicate_registry` 的定义域/值域校验）、质量护照
//! （`quality` 的完整性/一致性）与域/值域类型推断共同消费的坐标系。
//!
//! 设计要点（纯符号，零训练，同输入同字节）：
//! - **别名只认显式声明**：`Person = 人物`、`Entity = 实体` 都写死在配置里，
//!   不做任何模糊匹配——防止「苹果(公司) ≈ 苹果(水果)」式过度合并；
//! - **子类闭包**：`人物 ⊑ 实体` 后，所有判定都沿闭包传播（`学生 ⊑ 人物` 时
//!   `学生` 自动满足要求「人物」的定义域）；
//! - **互斥兜底**：`人物 ⊥ 地点` 是「可证明不兼容」的唯一来源。开放世界
//!   语义：两类型闭包无交集且无互斥证据时按「未知」放行，绝不误杀；
//! - **类型来源标记**：每条类型断言都带来源（显式 / 域推断 / 值域推断 /
//!   子类闭包），支撑质量报告里「这个类型是怎么来的」的逐条复述；
//! - **摘要稳定**：`digest()` 由规范 JSON + FNV-1a 得出，配置内容决定字节。
//!
//! 名称归一化为小写（`Person` → `person`）：RDF 类型标签在存储快照里本就
//! 小写化，别名/子类/互斥因此天然大小写不敏感；中文名不受影响。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};

/// 别名链最大跳数：配置里出现环时在有限步内退化为原样返回。
pub const MAX_ALIAS_HOPS: usize = 32;

/// `[[type_system.aliases]]`：别名 → 规范类型。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeAliasConfig {
    pub alias: String,
    pub canonical: String,
}

/// `[[type_system.subclasses]]`：子类 ⊑ 父类。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeSubclassConfig {
    pub sub: String,
    pub sup: String,
}

/// `[[type_system.disjoint]]`：互斥类型对（可证明不兼容的唯一来源）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeDisjointConfig {
    pub left: String,
    pub right: String,
}

/// config.toml 中的 `[type_system]` 段。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeSystemConfig {
    /// 模式闸门按闭包判定（默认关：按精确集合语义判定）。
    #[serde(default)]
    pub closure_gate: bool,
    #[serde(default)]
    pub aliases: Vec<TypeAliasConfig>,
    #[serde(default)]
    pub subclasses: Vec<TypeSubclassConfig>,
    #[serde(default)]
    pub disjoint: Vec<TypeDisjointConfig>,
}

/// 类型断言的来源（可多源并存，逐条可复述）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeSource {
    /// 图中显式 `rdf:type` 断言。
    Explicit,
    /// 由关系的定义域推断（主语继承）。
    DomainInference,
    /// 由关系的值域推断（宾语继承）。
    RangeInference,
    /// 子类闭包向上传播。
    SubclassClosure,
}

impl TypeSource {
    pub fn label(&self) -> &'static str {
        match self {
            TypeSource::Explicit => "显式",
            TypeSource::DomainInference => "定义域推断",
            TypeSource::RangeInference => "值域推断",
            TypeSource::SubclassClosure => "子类闭包",
        }
    }
}

/// 类型集合的判定结果（开放世界三值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeVerdict {
    /// 存在闭包可达（别名/子类）的相容类型对。
    Compatible,
    /// 闭包无交集且存在互斥证据：可证明不兼容。
    Disjoint,
    /// 闭包无交集但无互斥证据：未知，按开放世界放行。
    Unknown,
}

/// 类型格：别名 + 子类边 + 互斥对 + 闭包闸门开关。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeSystem {
    /// 归一化别名 → 归一化规范类型（链由 `canonical()` 展开）。
    #[serde(default)]
    aliases: BTreeMap<String, String>,
    /// 归一化子类 → 直接父类集合。
    #[serde(default)]
    subclasses: BTreeMap<String, BTreeSet<String>>,
    /// 归一化互斥对（字典序，去重）。
    #[serde(default)]
    disjoint: BTreeSet<(String, String)>,
    /// 模式闸门闭包档。
    #[serde(default)]
    closure_gate: bool,
    /// 归一化名 → 首次声明的书写形式（报告可读性，不参与判定）。
    #[serde(default)]
    display_names: BTreeMap<String, String>,
}

fn normalize(name: &str) -> String {
    name.trim().to_lowercase()
}

impl TypeSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从配置段构建：声明顺序固定，重复声明幂等。
    pub fn from_config(config: &TypeSystemConfig) -> Self {
        let mut system = Self::new();
        system.closure_gate = config.closure_gate;
        for alias in &config.aliases {
            system.declare_alias(&alias.alias, &alias.canonical);
        }
        for edge in &config.subclasses {
            system.declare_subclass(&edge.sub, &edge.sup);
        }
        for pair in &config.disjoint {
            system.declare_disjoint(&pair.left, &pair.right);
        }
        system
    }

    pub fn closure_gate(&self) -> bool {
        self.closure_gate
    }

    pub fn set_closure_gate(&mut self, enabled: bool) {
        self.closure_gate = enabled;
    }

    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    pub fn subclass_count(&self) -> usize {
        self.subclasses.values().map(BTreeSet::len).sum()
    }

    pub fn disjoint_count(&self) -> usize {
        self.disjoint.len()
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty() && self.subclasses.is_empty() && self.disjoint.is_empty()
    }

    fn remember_display(&mut self, raw: &str) {
        let normalized = normalize(raw);
        if !normalized.is_empty() {
            self.display_names
                .entry(normalized)
                .or_insert_with(|| raw.trim().to_string());
        }
    }

    /// 声明别名：拒绝空名、自指与成环；重复声明返回 false。
    pub fn declare_alias(&mut self, alias: &str, canonical: &str) -> bool {
        let alias_key = normalize(alias);
        let canonical_key = normalize(canonical);
        if alias_key.is_empty() || canonical_key.is_empty() || alias_key == canonical_key {
            return false;
        }
        if self.aliases.get(&alias_key) == Some(&canonical_key) {
            return false;
        }
        // 成环检测：沿 canonical 的别名链前进，若能碰到 alias 则拒绝
        let mut cursor = canonical_key.clone();
        for _ in 0..MAX_ALIAS_HOPS {
            if cursor == alias_key {
                return false;
            }
            match self.aliases.get(&cursor) {
                Some(next) if next != &cursor => cursor = next.clone(),
                _ => break,
            }
        }
        self.remember_display(alias);
        self.remember_display(canonical);
        self.aliases.insert(alias_key, canonical_key);
        true
    }

    /// 声明子类边：sub ⊑ sup；拒绝空名与自环。
    pub fn declare_subclass(&mut self, sub: &str, sup: &str) -> bool {
        let sub_key = normalize(sub);
        let sup_key = normalize(sup);
        if sub_key.is_empty() || sup_key.is_empty() || sub_key == sup_key {
            return false;
        }
        self.remember_display(sub);
        self.remember_display(sup);
        self.subclasses
            .entry(sub_key)
            .or_default()
            .insert(sup_key)
    }

    /// 声明互斥对；拒绝空名与自指。返回是否新登记。
    pub fn declare_disjoint(&mut self, left: &str, right: &str) -> bool {
        let left_key = normalize(left);
        let right_key = normalize(right);
        if left_key.is_empty() || right_key.is_empty() || left_key == right_key {
            return false;
        }
        self.remember_display(left);
        self.remember_display(right);
        let pair = if left_key <= right_key {
            (left_key, right_key)
        } else {
            (right_key, left_key)
        };
        self.disjoint.insert(pair)
    }

    /// 名称归一：依次展开别名链（有限步，防环）。
    pub fn canonical(&self, name: &str) -> String {
        let mut current = normalize(name);
        for _ in 0..MAX_ALIAS_HOPS {
            match self.aliases.get(&current) {
                Some(next) if next != &current => current = next.clone(),
                _ => break,
            }
        }
        current
    }

    /// 书写形式（首次声明的拼写；未声明名原样去空白）。
    pub fn display(&self, name: &str) -> String {
        let key = self.canonical(name);
        self.display_key(&key)
    }

    /// 归一化键的书写形式（别名自身等不经别名展开的场景使用）。
    fn display_key(&self, key: &str) -> String {
        self.display_names
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }

    /// 直接父类（对别名与子类边两侧做规范归一后匹配）。
    fn direct_supers(&self, node: &str) -> BTreeSet<String> {
        let mut supers = BTreeSet::new();
        for (sub, parents) in &self.subclasses {
            if self.canonical(sub) == node {
                for parent in parents {
                    supers.insert(self.canonical(parent));
                }
            }
        }
        supers
    }

    /// 祖先闭包（含自身）：沿子类边上溯到不动点。
    pub fn ancestors(&self, name: &str) -> BTreeSet<String> {
        let start = self.canonical(name);
        let mut result: BTreeSet<String> = BTreeSet::new();
        result.insert(start.clone());
        let mut frontier = vec![start];
        while let Some(node) = frontier.pop() {
            for parent in self.direct_supers(&node) {
                if result.insert(parent.clone()) {
                    frontier.push(parent);
                }
            }
        }
        result
    }

    /// a ⊑ b（含别名归一与传递闭包；a 与 b 为同一类型时恒真）。
    pub fn is_subclass(&self, a: &str, b: &str) -> bool {
        let target = self.canonical(b);
        self.ancestors(a).contains(&target)
    }

    /// 闭包相关：两者存在子类路径（任一方向，含别名归一后的同一类型）。
    ///
    /// 注意不能只看「祖先闭包相交」：任意两个类型都可能共享一个公共根
    /// （如 `地点 ⊑ 实体`、`人物 ⊑ 实体`），共享根不构成相容证据。
    pub fn types_compatible(&self, a: &str, b: &str) -> bool {
        self.is_subclass(a, b) || self.is_subclass(b, a)
    }

    /// 可证明互斥：存在声明的互斥对，其两侧分别被两类型的闭包涵盖，
    /// 且两者本身没有子类路径（子类关系优先于互斥声明）。
    pub fn types_disjoint(&self, a: &str, b: &str) -> bool {
        if self.types_compatible(a, b) {
            return false;
        }
        let left = self.ancestors(a);
        let right = self.ancestors(b);
        self.disjoint.iter().any(|(l, r)| {
            (left.contains(l) && right.contains(r)) || (left.contains(r) && right.contains(l))
        })
    }

    /// 集合判定：任一相容对 → 相容；否则任一互斥对 → 可证明不兼容；都没有 → 未知。
    pub fn sets_admit(&self, actual: &BTreeSet<String>, allowed: &BTreeSet<String>) -> TypeVerdict {
        if actual.is_empty() || allowed.is_empty() {
            return TypeVerdict::Unknown;
        }
        let mut any_disjoint = false;
        for a in actual {
            for d in allowed {
                if self.types_compatible(a, d) {
                    return TypeVerdict::Compatible;
                }
                if self.types_disjoint(a, d) {
                    any_disjoint = true;
                }
            }
        }
        if any_disjoint {
            TypeVerdict::Disjoint
        } else {
            TypeVerdict::Unknown
        }
    }

    /// 域/值域类型推断：显式类型 + 关系端点的域/值域投影 + 子类闭包展开。
    ///
    /// `relations`：(主语, 谓词, 宾语)；`domains`/`ranges`：谓词 → 声明的允许类型
    /// （可为别名写法，内部归一）。返回 实体 → 类型 → 来源集合（全部规范化、排序）。
    pub fn infer_entity_types(
        &self,
        relations: &[(String, String, String)],
        domains: &BTreeMap<String, BTreeSet<String>>,
        ranges: &BTreeMap<String, BTreeSet<String>>,
        explicit: &BTreeMap<String, BTreeSet<String>>,
    ) -> BTreeMap<String, BTreeMap<String, BTreeSet<TypeSource>>> {
        let mut inferred: BTreeMap<String, BTreeMap<String, BTreeSet<TypeSource>>> =
            BTreeMap::new();
        let mut add = |entity: &str, type_name: &str, source: TypeSource| {
            let key = self.canonical(type_name);
            if key.is_empty() {
                return;
            }
            inferred
                .entry(entity.to_string())
                .or_default()
                .entry(key)
                .or_default()
                .insert(source);
        };
        for (entity, types) in explicit {
            for type_name in types {
                add(entity, type_name, TypeSource::Explicit);
            }
        }
        for (subject, predicate, object) in relations {
            let key = self.canonical(predicate);
            if let Some(allowed) = domains.get(&key).or_else(|| domains.get(predicate)) {
                for type_name in allowed {
                    add(subject, type_name, TypeSource::DomainInference);
                }
            }
            if let Some(allowed) = ranges.get(&key).or_else(|| ranges.get(predicate)) {
                for type_name in allowed {
                    add(object, type_name, TypeSource::RangeInference);
                }
            }
        }
        // 子类闭包展开：每个实体的每个类型都补上其全部祖先（来源：子类闭包）
        let seeds: Vec<(String, String)> = inferred
            .iter()
            .flat_map(|(entity, types)| {
                types.keys().map(|type_name| (entity.clone(), type_name.clone()))
            })
            .collect();
        for (entity, type_name) in seeds {
            for ancestor in self.ancestors(&type_name) {
                if ancestor == type_name {
                    continue;
                }
                inferred
                    .entry(entity.clone())
                    .or_default()
                    .entry(ancestor)
                    .or_default()
                    .insert(TypeSource::SubclassClosure);
            }
        }
        inferred
    }

    /// 规范内容快照（摘要与测试的输入）。
    pub fn snapshot_json(&self) -> String {
        let value = serde_json::json!({
            "closure_gate": self.closure_gate,
            "aliases": self.aliases,
            "subclasses": self.subclasses,
            "disjoint": self.disjoint,
        });
        canonical_json(&value)
    }

    /// 确定性摘要（FNV-1a 64，十六进制）。
    pub fn digest(&self) -> String {
        fnv1a_hex(self.snapshot_json().as_bytes())
    }

    /// 人读报告（确定性排序）。
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "类型体系：别名 {} 条，子类边 {} 条，互斥对 {} 对；模式闸门 = {}。",
            self.aliases.len(),
            self.subclass_count(),
            self.disjoint.len(),
            if self.closure_gate {
                "闭包档（别名/子类/互斥判定）"
            } else {
                "legacy（精确集合）"
            }
        ));
        if self.aliases.is_empty() && self.subclasses.is_empty() && self.disjoint.is_empty() {
            lines.push(
                "  （未声明任何类型：可在 config.toml 的 [type_system] 段登记别名/子类/互斥）"
                    .to_string(),
            );
            return lines.join("\n");
        }
        if !self.aliases.is_empty() {
            let aliases: Vec<String> = self
                .aliases
                .iter()
                .map(|(alias, canonical)| {
                    format!(
                        "{} = {}",
                        self.display_key(alias),
                        self.display(canonical)
                    )
                })
                .collect();
            lines.push(format!("  别名：{}", aliases.join("；")));
        }
        if !self.subclasses.is_empty() {
            let mut edges: Vec<String> = Vec::new();
            for (sub, parents) in &self.subclasses {
                for parent in parents {
                    edges.push(format!(
                        "{} ⊑ {}",
                        self.display_key(sub),
                        self.display_key(parent)
                    ));
                }
            }
            edges.sort();
            lines.push(format!("  子类：{}", edges.join("；")));
        }
        if !self.disjoint.is_empty() {
            let pairs: Vec<String> = self
                .disjoint
                .iter()
                .map(|(left, right)| {
                    format!(
                        "{} ⊥ {}",
                        self.display_key(left),
                        self.display_key(right)
                    )
                })
                .collect();
            lines.push(format!("  互斥：{}", pairs.join("；")));
        }
        lines.join("\n")
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn sample() -> TypeSystem {
        TypeSystem::from_config(&TypeSystemConfig {
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
                TypeAliasConfig {
                    alias: "人".to_string(),
                    canonical: "人物".to_string(),
                },
            ],
            subclasses: vec![
                TypeSubclassConfig {
                    sub: "人物".to_string(),
                    sup: "实体".to_string(),
                },
                TypeSubclassConfig {
                    sub: "地点".to_string(),
                    sup: "实体".to_string(),
                },
                TypeSubclassConfig {
                    sub: "学生".to_string(),
                    sup: "人物".to_string(),
                },
            ],
            disjoint: vec![TypeDisjointConfig {
                left: "人物".to_string(),
                right: "地点".to_string(),
            }],
        })
    }

    #[test]
    fn alias_normalization_is_case_insensitive_and_chain_safe() {
        let system = sample();
        assert_eq!(system.canonical("Person"), "人物");
        assert_eq!(system.canonical("PERSON"), "人物");
        assert_eq!(system.canonical("人"), "人物");
        assert_eq!(system.canonical(" 实体 "), "实体");
        assert_eq!(system.canonical("未知类型"), "未知类型");

        let mut chain = TypeSystem::new();
        assert!(chain.declare_alias("A", "B"));
        assert!(chain.declare_alias("B", "C"));
        assert_eq!(chain.canonical("A"), "c");
        // B→C 已存在时 A→B 的链是 A→B→C；此时声明 C→A 成环被拒
        assert!(!chain.declare_alias("C", "A"));
        assert_eq!(chain.canonical("C"), "c");
    }

    #[test]
    fn subclass_closure_and_subtype_direction() {
        let system = sample();
        assert!(system.is_subclass("学生", "人物"));
        assert!(system.is_subclass("学生", "实体"));
        assert!(system.is_subclass("Person", "实体"), "别名参与子类闭包");
        assert!(!system.is_subclass("实体", "人物"), "实体不是人物的子类");
        let ancestors = system.ancestors("学生");
        assert!(ancestors.contains("学生"));
        assert!(ancestors.contains("人物"));
        assert!(ancestors.contains("实体"));
    }

    #[test]
    fn open_world_verdicts() {
        let system = sample();
        assert!(system.types_compatible("Person", "人物"));
        assert!(system.types_compatible("学生", "人物"));
        assert!(system.types_compatible("实体", "人物"), "上位类与下位类闭包相交");
        assert!(system.types_disjoint("地点", "人物"));
        assert!(system.types_disjoint("城市", "人物") == false, "未落型不构成互斥");

        assert_eq!(
            system.sets_admit(&typed(&["Person"]), &typed(&["人物"])),
            TypeVerdict::Compatible
        );
        assert_eq!(
            system.sets_admit(&typed(&["地点"]), &typed(&["人物"])),
            TypeVerdict::Disjoint
        );
        assert_eq!(
            system.sets_admit(&typed(&["城市"]), &typed(&["人物"])),
            TypeVerdict::Unknown,
            "未声明类型：闭包不可达但无互斥证据 → 开放世界未知"
        );
        assert_eq!(
            system.sets_admit(&typed(&[]), &typed(&["人物"])),
            TypeVerdict::Unknown
        );
    }

    #[test]
    fn disjoint_inherits_through_subclasses() {
        let system = sample();
        assert!(system.types_disjoint("学生", "地点"));
    }

    #[test]
    fn inference_marks_sources_and_closes_upwards() {
        let system = sample();
        let relations = vec![
            ("张三".to_string(), "师傅".to_string(), "李四".to_string()),
            ("北京大学".to_string(), "位于".to_string(), "海淀".to_string()),
        ];
        let mut domains = BTreeMap::new();
        domains.insert("师傅".to_string(), typed(&["人物"]));
        let mut ranges = BTreeMap::new();
        ranges.insert("师傅".to_string(), typed(&["人物"]));
        let mut explicit = BTreeMap::new();
        explicit.insert("张三".to_string(), typed(&["Person"]));

        let inferred = system.infer_entity_types(&relations, &domains, &ranges, &explicit);
        let zhangsan = &inferred["张三"];
        assert!(zhangsan["人物"].contains(&TypeSource::Explicit));
        assert!(zhangsan["人物"].contains(&TypeSource::DomainInference));
        assert!(zhangsan["实体"].contains(&TypeSource::SubclassClosure));
        let lisi = &inferred["李四"];
        assert!(lisi["人物"].contains(&TypeSource::RangeInference));

        // 没有域/值域声明的谓词不产生推断
        assert!(!inferred.contains_key("北京大学"));
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let system = sample();
        let clone = sample();
        assert_eq!(system.digest(), clone.digest());
        assert!(!system.digest().is_empty());

        let mut changed = sample();
        changed.declare_disjoint("天气", "地点");
        assert_ne!(system.digest(), changed.digest());

        let mut gate = sample();
        gate.set_closure_gate(false);
        assert_ne!(system.digest(), gate.digest());
    }

    #[test]
    fn report_is_deterministic_and_shows_declarations() {
        let system = sample();
        let report = system.report();
        assert!(report.contains("Person = 人物"), "{}", report);
        assert!(report.contains("人物 ⊑ 实体"), "{}", report);
        assert!(report.contains("人物 ⊥ 地点"), "{}", report);
        assert!(report.contains("闭包档"), "{}", report);
        assert_eq!(report, sample().report());
    }

    #[test]
    fn json_roundtrip_preserves_semantics() {
        let system = sample();
        let restored = TypeSystem::from_json(&system.to_json()).unwrap();
        assert_eq!(restored.digest(), system.digest());
        assert!(restored.types_disjoint("地点", "人物"));
        assert_eq!(restored.canonical("person"), "人物");
    }
}

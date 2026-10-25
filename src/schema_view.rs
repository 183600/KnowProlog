//! RDFS 模式视图（Schema View）：把散落在类型体系、谓词画像与本体公理里的
//! RDFS 语义收敛成一份只读坐标系，供嵌入、检索、评估与生成共同消费。
//!
//! 本模块只做「确定性的闭包查询」，不训练、不猜词：
//! - 子类闭包：`学生 ⊑ 人物 ⊑ 实体` 后，任何要求「人物」的位置都接受「学生」；
//! - 子属性闭包：`director ⊑ 导演` 后，`director` 的事实同时具备 `导演` 的角色特征；
//! - 定义域/值域：沿上位属性继承，端点类型按开放世界三值判定
//!   （`Compatible` / `Unknown` / `Disjoint`），只有显式互斥证据才算冲突；
//! - 覆盖率审计：如实报告「哪些谓词没有 schema、哪些实体没有类型」，
//!   避免在稀疏 schema 上假装有收益。
//!
//! 所有集合都用 `BTreeSet`/`BTreeMap`，同输入同字节；证明理由以字符串返回，
//! 可以逐条回溯到具体公理。

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};
use crate::predicate_registry::PredicateProfileConfig;
use crate::type_system::{TypeSystemConfig, TypeVerdict};

/// 闭包最多展开的跳数：配置成环时在有限步内退化，不会卡死。
pub const MAX_SCHEMA_HOPS: usize = 64;

fn trimmed(name: &str) -> &str {
    name.trim()
}

fn lookup_alias<'a>(map: &'a BTreeMap<String, String>, name: &str) -> Option<&'a String> {
    let name = trimmed(name);
    if let Some(target) = map.get(name) {
        return Some(target);
    }
    let lower = name.to_lowercase();
    map.get(&lower)
}

fn canonical_with(map: &BTreeMap<String, String>, name: &str) -> String {
    let mut current = trimmed(name).to_string();
    for _ in 0..MAX_SCHEMA_HOPS {
        let Some(next) = lookup_alias(map, &current) else {
            break;
        };
        if next == &current {
            break;
        }
        current = next.clone();
    }
    current
}

/// 类型/谓词的开放世界判定结果；复用类型体系的 `TypeVerdict`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeVerdict {
    /// 主语与谓词定义域的判定。
    pub subject: TypeVerdict,
    /// 宾语与谓词值域的判定。
    pub object: TypeVerdict,
    /// 两侧的综合判定：任一侧可证明互斥 → `Disjoint`，任一侧相容 → `Compatible`。
    pub overall: TypeVerdict,
    /// 判定理由（含用到的公理），确定性排序。
    pub reasons: Vec<String>,
}

impl EdgeVerdict {
    fn new(subject: TypeVerdict, object: TypeVerdict, reasons: Vec<String>) -> Self {
        let overall = if subject == TypeVerdict::Disjoint || object == TypeVerdict::Disjoint {
            TypeVerdict::Disjoint
        } else if subject == TypeVerdict::Compatible || object == TypeVerdict::Compatible {
            TypeVerdict::Compatible
        } else {
            TypeVerdict::Unknown
        };
        Self {
            subject,
            object,
            overall,
            reasons,
        }
    }
}

/// 只读 RDFS 模式视图。
#[derive(Debug, Clone, Default)]
pub struct SchemaView {
    type_aliases: BTreeMap<String, String>,
    property_aliases: BTreeMap<String, String>,
    /// 直接子类边：子 → 直接父类集合。
    subclasses: BTreeMap<String, BTreeSet<String>>,
    /// 直接子属性边：子 → 直接父属性集合。
    subproperties: BTreeMap<String, BTreeSet<String>>,
    /// 谓词 → 直接定义域类集合。
    domains: BTreeMap<String, BTreeSet<String>>,
    /// 谓词 → 直接值域类集合。
    ranges: BTreeMap<String, BTreeSet<String>>,
    /// 显式互斥类对（规范名，字典序）。
    disjoint: BTreeSet<(String, String)>,
    transitive: BTreeSet<String>,
    symmetric: BTreeSet<String>,
    inverses: BTreeSet<(String, String)>,
    declared_classes: BTreeSet<String>,
    declared_properties: BTreeSet<String>,
}

impl SchemaView {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从本体程序（配置态 + 教学态归约后的公理集）构造。
    pub fn from_ontology_program(program: &OntologyProgram) -> Self {
        let mut view = Self::new();
        for lesson in &program.lessons {
            view.observe_lesson(lesson);
        }
        view
    }

    /// 从 `config.toml` 的 `[type_system]` 与 `[[predicate_profiles]]` 构造，
    /// 包含别名归一（别名不进入本体公理，因此在这里单独登记）。
    pub fn from_config(types: &TypeSystemConfig, profiles: &[PredicateProfileConfig]) -> Self {
        let mut view = Self::new();
        for alias in &types.aliases {
            view.declare_type_alias(&alias.alias, &alias.canonical);
        }
        for edge in &types.subclasses {
            view.declare_subclass(&edge.sub, &edge.sup);
        }
        for pair in &types.disjoint {
            view.declare_disjoint(&pair.left, &pair.right);
        }
        for profile in profiles {
            for alias in &profile.aliases {
                view.declare_property_alias(alias, &profile.name);
            }
            if let Some(inverse) = &profile.inverse {
                view.declare_inverse(&profile.name, inverse);
            }
            for class in &profile.domain {
                view.declare_domain(&profile.name, class);
            }
            for class in &profile.range {
                view.declare_range(&profile.name, class);
            }
            if let Some(super_property) = &profile.subproperty_of {
                view.declare_subproperty(&profile.name, super_property);
            }
            if profile.transitive {
                view.declare_transitive(&profile.name);
            }
            if profile.symmetric {
                view.declare_symmetric(&profile.name);
            }
            if !profile.name.trim().is_empty() {
                view.declared_properties
                    .insert(view.canonical_property(&profile.name));
            }
        }
        view
    }

    /// 从一条本体公理登记（重复登记幂等）。
    pub fn observe_lesson(&mut self, lesson: &OntologyLesson) {
        match lesson {
            OntologyLesson::Subclass { sub, sup, .. } => {
                self.declare_subclass(sub, sup);
            }
            OntologyLesson::Instance { class, .. } => {
                self.declared_classes.insert(self.canonical_type(class));
            }
            OntologyLesson::Disjoint { left, right, .. } => {
                self.declare_disjoint(left, right);
            }
            OntologyLesson::Subproperty { sub, sup, .. } => {
                self.declare_subproperty(sub, sup);
            }
            OntologyLesson::Transitive { property, .. } => {
                self.declare_transitive(property);
            }
            OntologyLesson::Symmetric { property, .. } => {
                self.declare_symmetric(property);
            }
            OntologyLesson::Inverse { left, right, .. } => {
                self.declare_inverse(left, right);
            }
            OntologyLesson::Domain {
                property, class, ..
            } => {
                self.declare_domain(property, class);
            }
            OntologyLesson::Range {
                property, class, ..
            } => {
                self.declare_range(property, class);
            }
            OntologyLesson::ClassProperty {
                class, property, ..
            } => {
                self.declared_classes.insert(self.canonical_type(class));
                self.declared_properties
                    .insert(self.canonical_property(property));
            }
        }
    }

    pub fn declare_type_alias(&mut self, alias: &str, canonical: &str) -> bool {
        let alias = trimmed(alias);
        let canonical = trimmed(canonical);
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return false;
        }
        for key in [alias.to_string(), alias.to_lowercase()] {
            self.type_aliases
                .entry(key)
                .or_insert_with(|| canonical.to_string());
        }
        true
    }

    pub fn declare_property_alias(&mut self, alias: &str, canonical: &str) -> bool {
        let alias = trimmed(alias);
        let canonical = trimmed(canonical);
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return false;
        }
        for key in [alias.to_string(), alias.to_lowercase()] {
            self.property_aliases
                .entry(key)
                .or_insert_with(|| canonical.to_string());
        }
        true
    }

    pub fn declare_subclass(&mut self, sub: &str, sup: &str) -> bool {
        let sub = self.canonical_type(sub);
        let sup = self.canonical_type(sup);
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return false;
        }
        self.declared_classes.insert(sub.clone());
        self.declared_classes.insert(sup.clone());
        self.subclasses.entry(sub).or_default().insert(sup)
    }

    pub fn declare_subproperty(&mut self, sub: &str, sup: &str) -> bool {
        let sub = self.canonical_property(sub);
        let sup = self.canonical_property(sup);
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return false;
        }
        self.declared_properties.insert(sub.clone());
        self.declared_properties.insert(sup.clone());
        self.subproperties.entry(sub).or_default().insert(sup)
    }

    pub fn declare_disjoint(&mut self, left: &str, right: &str) -> bool {
        let left = self.canonical_type(left);
        let right = self.canonical_type(right);
        if left.is_empty() || right.is_empty() || left == right {
            return false;
        }
        self.declared_classes.insert(left.clone());
        self.declared_classes.insert(right.clone());
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        self.disjoint.insert(pair)
    }

    pub fn declare_domain(&mut self, property: &str, class: &str) -> bool {
        let property = self.canonical_property(property);
        let class = self.canonical_type(class);
        if property.is_empty() || class.is_empty() {
            return false;
        }
        self.declared_properties.insert(property.clone());
        self.declared_classes.insert(class.clone());
        self.domains.entry(property).or_default().insert(class)
    }

    pub fn declare_range(&mut self, property: &str, class: &str) -> bool {
        let property = self.canonical_property(property);
        let class = self.canonical_type(class);
        if property.is_empty() || class.is_empty() {
            return false;
        }
        self.declared_properties.insert(property.clone());
        self.declared_classes.insert(class.clone());
        self.ranges.entry(property).or_default().insert(class)
    }

    pub fn declare_transitive(&mut self, property: &str) -> bool {
        let property = self.canonical_property(property);
        if property.is_empty() {
            return false;
        }
        self.declared_properties.insert(property.clone());
        self.transitive.insert(property)
    }

    pub fn declare_symmetric(&mut self, property: &str) -> bool {
        let property = self.canonical_property(property);
        if property.is_empty() {
            return false;
        }
        self.declared_properties.insert(property.clone());
        self.symmetric.insert(property)
    }

    pub fn declare_inverse(&mut self, left: &str, right: &str) -> bool {
        let left = self.canonical_property(left);
        let right = self.canonical_property(right);
        if left.is_empty() || right.is_empty() || left == right {
            return false;
        }
        self.declared_properties.insert(left.clone());
        self.declared_properties.insert(right.clone());
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        self.inverses.insert(pair)
    }

    pub fn canonical_type(&self, name: &str) -> String {
        canonical_with(&self.type_aliases, name)
    }

    pub fn canonical_property(&self, name: &str) -> String {
        canonical_with(&self.property_aliases, name)
    }

    pub fn is_empty(&self) -> bool {
        self.subclasses.is_empty()
            && self.subproperties.is_empty()
            && self.domains.is_empty()
            && self.ranges.is_empty()
            && self.disjoint.is_empty()
    }

    pub fn declared_classes(&self) -> Vec<String> {
        self.declared_classes.iter().cloned().collect()
    }

    pub fn declared_properties(&self) -> Vec<String> {
        self.declared_properties.iter().cloned().collect()
    }

    pub fn subclass_edge_count(&self) -> usize {
        self.subclasses.values().map(BTreeSet::len).sum()
    }

    pub fn subproperty_edge_count(&self) -> usize {
        self.subproperties.values().map(BTreeSet::len).sum()
    }

    pub fn disjoint_pairs(&self) -> Vec<(String, String)> {
        self.disjoint.iter().cloned().collect()
    }

    /// 祖先闭包（含自身）。
    pub fn class_ancestors(&self, class: &str) -> BTreeSet<String> {
        let start = self.canonical_type(class);
        let mut result = BTreeSet::new();
        if start.is_empty() {
            return result;
        }
        let mut queue = VecDeque::new();
        result.insert(start.clone());
        queue.push_back(start);
        let mut steps = 0;
        while let Some(node) = queue.pop_front() {
            steps += 1;
            if steps > MAX_SCHEMA_HOPS * MAX_SCHEMA_HOPS {
                break;
            }
            let Some(parents) = self.subclasses.get(&node) else {
                continue;
            };
            for parent in parents {
                if result.insert(parent.clone()) {
                    queue.push_back(parent.clone());
                }
            }
        }
        result
    }

    /// `sub ⊑ sup`（含自反与别名归一）。
    pub fn is_subclass(&self, sub: &str, sup: &str) -> bool {
        let sup = self.canonical_type(sup);
        self.class_ancestors(sub).contains(&sup)
    }

    /// 两个类型是否有子类路径（任一方向）；共享祖先不算相容证据。
    pub fn types_compatible(&self, a: &str, b: &str) -> bool {
        self.is_subclass(a, b) || self.is_subclass(b, a)
    }

    /// 可证明互斥：显式互斥对的两侧分别被两个类型闭包涵盖，且两者无子类路径。
    pub fn types_disjoint(&self, a: &str, b: &str) -> bool {
        if self.types_compatible(a, b) {
            return false;
        }
        let left = self.class_ancestors(a);
        let right = self.class_ancestors(b);
        self.disjoint.iter().any(|(div_left, div_right)| {
            (left.contains(div_left) && right.contains(div_right))
                || (left.contains(div_right) && right.contains(div_left))
        })
    }

    /// 集合判定：相容优先，其次可证明互斥，否则未知。
    pub fn sets_admit(
        &self,
        actual: &BTreeSet<String>,
        expected: &BTreeSet<String>,
    ) -> TypeVerdict {
        if actual.is_empty() || expected.is_empty() {
            return TypeVerdict::Unknown;
        }
        let mut any_disjoint = false;
        for a in actual {
            for e in expected {
                if self.types_compatible(a, e) {
                    return TypeVerdict::Compatible;
                }
                if self.types_disjoint(a, e) {
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

    /// 属性的上位属性闭包（含自身；沿子属性边上浮）。
    pub fn superproperties(&self, property: &str) -> BTreeSet<String> {
        let start = self.canonical_property(property);
        let mut result = BTreeSet::new();
        if start.is_empty() {
            return result;
        }
        let mut queue = VecDeque::new();
        result.insert(start.clone());
        queue.push_back(start);
        let mut steps = 0;
        while let Some(node) = queue.pop_front() {
            steps += 1;
            if steps > MAX_SCHEMA_HOPS * MAX_SCHEMA_HOPS {
                break;
            }
            let Some(parents) = self.subproperties.get(&node) else {
                continue;
            };
            for parent in parents {
                if result.insert(parent.clone()) {
                    queue.push_back(parent.clone());
                }
            }
        }
        result
    }

    /// 定义域约束类集（沿上位属性继承；返回声明类本身而非其祖先，
    /// 否则 `人物` 的定义域会被误扩成 `实体`，把兄弟类 `地点` 也放进来）。
    pub fn domain_classes(&self, property: &str) -> BTreeSet<String> {
        let mut classes = BTreeSet::new();
        for predicate in self.superproperties(property) {
            if let Some(declared) = self.domains.get(&predicate) {
                classes.extend(declared.iter().cloned());
            }
        }
        classes
    }

    /// 值域约束类集（沿上位属性继承；同样不做祖先扩张）。
    pub fn range_classes(&self, property: &str) -> BTreeSet<String> {
        let mut classes = BTreeSet::new();
        for predicate in self.superproperties(property) {
            if let Some(declared) = self.ranges.get(&predicate) {
                classes.extend(declared.iter().cloned());
            }
        }
        classes
    }

    /// 谓词是否有任何 schema 声明（定义域/值域/上位属性/传递/对称/互逆/显式登记）。
    pub fn has_property_schema(&self, property: &str) -> bool {
        let predicates = self.superproperties(property);
        predicates.iter().any(|predicate| {
            self.declared_properties.contains(predicate)
                || self.domains.contains_key(predicate)
                || self.ranges.contains_key(predicate)
                || self.subproperties.contains_key(predicate)
                || self.transitive.contains(predicate)
                || self.symmetric.contains(predicate)
        }) || self
            .inverses
            .iter()
            .any(|(left, right)| predicates.contains(left) || predicates.contains(right))
    }

    pub fn is_transitive(&self, property: &str) -> bool {
        self.superproperties(property)
            .iter()
            .any(|predicate| self.transitive.contains(predicate))
    }

    pub fn is_symmetric(&self, property: &str) -> bool {
        let predicates = self.superproperties(property);
        predicates
            .iter()
            .any(|predicate| self.symmetric.contains(predicate))
    }

    pub fn inverse_of(&self, property: &str) -> Option<String> {
        let property = self.canonical_property(property);
        for (left, right) in &self.inverses {
            if left == &property {
                return Some(right.clone());
            }
            if right == &property {
                return Some(left.clone());
            }
        }
        None
    }

    /// 主语/宾语类型对谓词定义域/值域的开放世界裁决。
    pub fn edge_verdict(
        &self,
        subject_types: &BTreeSet<String>,
        predicate: &str,
        object_types: &BTreeSet<String>,
    ) -> EdgeVerdict {
        let property = self.canonical_property(predicate);
        let domains = self.domain_classes(&property);
        let ranges = self.range_classes(&property);
        let mut reasons = Vec::new();
        let subject = if domains.is_empty() {
            reasons.push(format!("{}：未声明定义域，按开放世界放行", property));
            TypeVerdict::Unknown
        } else {
            let verdict = self.sets_admit(
                &subject_types
                    .iter()
                    .map(|name| self.canonical_type(name))
                    .collect(),
                &domains,
            );
            reasons.push(format!(
                "{} 的定义域为 {{ {} }}；主语类型 {{ {} }} → {:?}",
                property,
                domains.iter().cloned().collect::<Vec<_>>().join("、"),
                subject_types
                    .iter()
                    .map(|name| self.canonical_type(name))
                    .collect::<Vec<_>>()
                    .join("、"),
                verdict
            ));
            verdict
        };
        let object = if ranges.is_empty() {
            reasons.push(format!("{}：未声明值域，按开放世界放行", property));
            TypeVerdict::Unknown
        } else {
            let verdict = self.sets_admit(
                &object_types
                    .iter()
                    .map(|name| self.canonical_type(name))
                    .collect(),
                &ranges,
            );
            reasons.push(format!(
                "{} 的值域为 {{ {} }}；宾语类型 {{ {} }} → {:?}",
                property,
                ranges.iter().cloned().collect::<Vec<_>>().join("、"),
                object_types
                    .iter()
                    .map(|name| self.canonical_type(name))
                    .collect::<Vec<_>>()
                    .join("、"),
                verdict
            ));
            verdict
        };
        EdgeVerdict::new(subject, object, reasons)
    }

    /// 只判宾语侧（链接预测最常用：`(s, p, ?)` 的值域约束）。
    pub fn range_verdict(&self, predicate: &str, object_types: &BTreeSet<String>) -> TypeVerdict {
        let property = self.canonical_property(predicate);
        let ranges = self.range_classes(&property);
        if ranges.is_empty() {
            return TypeVerdict::Unknown;
        }
        self.sets_admit(
            &object_types
                .iter()
                .map(|name| self.canonical_type(name))
                .collect(),
            &ranges,
        )
    }

    /// 类型推断：显式类型 + 关系的定义域/值域投影 + 子类闭包。
    ///
    /// `relations` 应当只包含训练侧事实；留出三元组必须先剔除，否则评估会泄漏。
    pub fn infer_entity_types(
        &self,
        relations: &[(String, String, String)],
        explicit: &BTreeMap<String, BTreeSet<String>>,
    ) -> BTreeMap<String, BTreeSet<String>> {
        let mut inferred: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut add = |entity: &str, class: &str| {
            let entity = trimmed(entity);
            if entity.is_empty() {
                return;
            }
            let canonical = self.canonical_type(class);
            if canonical.is_empty() {
                return;
            }
            let bucket = inferred.entry(entity.to_string()).or_default();
            bucket.extend(self.class_ancestors(&canonical));
        };
        for (entity, types) in explicit {
            for class in types {
                add(entity, class);
            }
        }
        for (subject, predicate, object) in relations {
            for class in self.domain_classes(predicate) {
                add(subject, &class);
            }
            for class in self.range_classes(predicate) {
                add(object, &class);
            }
        }
        inferred
    }

    /// schema 覆盖率审计：把「有声明」与「数据里真的用上」分开报。
    pub fn coverage(
        &self,
        relations: &[(String, String, String)],
        explicit: &BTreeMap<String, BTreeSet<String>>,
    ) -> SchemaCoverage {
        SchemaCoverage::from_relations(self, relations, explicit)
    }

    /// 规范摘要：只由声明内容决定，供嵌入缓存判断 schema 是否真的变化。
    pub fn digest(&self) -> String {
        let value = serde_json::json!({
            "type_aliases": self.type_aliases,
            "property_aliases": self.property_aliases,
            "subclasses": self.subclasses,
            "subproperties": self.subproperties,
            "domains": self.domains,
            "ranges": self.ranges,
            "disjoint": self.disjoint,
            "transitive": self.transitive,
            "symmetric": self.symmetric,
            "inverses": self.inverses,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    /// 确定性人读报告。
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "RDFS 模式视图：类 {} 个、子类边 {} 条、互斥对 {} 对；属性 {} 个、子属性边 {} 条、定义域 {} 个、值域 {} 个。",
            self.declared_classes.len(),
            self.subclass_edge_count(),
            self.disjoint.len(),
            self.declared_properties.len(),
            self.subproperty_edge_count(),
            self.domains.len(),
            self.ranges.len()
        ));
        if !self.subclasses.is_empty() {
            let mut edges: Vec<String> = Vec::new();
            for (sub, supers) in &self.subclasses {
                for sup in supers {
                    edges.push(format!("{} ⊑ {}", sub, sup));
                }
            }
            edges.sort();
            lines.push(format!("  子类：{}", edges.join("；")));
        }
        if !self.subproperties.is_empty() {
            let mut edges: Vec<String> = Vec::new();
            for (sub, supers) in &self.subproperties {
                for sup in supers {
                    edges.push(format!("{} ⊑ {}", sub, sup));
                }
            }
            edges.sort();
            lines.push(format!("  子属性：{}", edges.join("；")));
        }
        if !self.domains.is_empty() || !self.ranges.is_empty() {
            let mut entries: Vec<String> = Vec::new();
            for property in &self.declared_properties {
                let domains = self.domain_classes(property);
                let ranges = self.range_classes(property);
                if domains.is_empty() && ranges.is_empty() {
                    continue;
                }
                entries.push(format!(
                    "{}（域：{}；值域：{}）",
                    property,
                    if domains.is_empty() {
                        "未声明".to_string()
                    } else {
                        domains.iter().cloned().collect::<Vec<_>>().join("、")
                    },
                    if ranges.is_empty() {
                        "未声明".to_string()
                    } else {
                        ranges.iter().cloned().collect::<Vec<_>>().join("、")
                    }
                ));
            }
            if !entries.is_empty() {
                lines.push(format!("  定义域/值域：{}", entries.join("；")));
            }
        }
        if !self.disjoint.is_empty() {
            let pairs: Vec<String> = self
                .disjoint
                .iter()
                .map(|(left, right)| format!("{} ⊥ {}", left, right))
                .collect();
            lines.push(format!("  互斥：{}", pairs.join("；")));
        }
        lines.join("\n")
    }
}

/// schema 覆盖率报告（确定性、可序列化）。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchemaCoverage {
    /// schema 中声明的类数。
    pub classes: usize,
    pub subclass_edges: usize,
    pub disjoint_pairs: usize,
    /// schema 中声明的属性数。
    pub properties: usize,
    pub properties_with_domain: usize,
    pub properties_with_range: usize,
    pub properties_with_super: usize,
    /// 数据中出现的谓词数。
    pub data_predicates: usize,
    /// 数据谓词中有任何 schema 声明的数量。
    pub predicates_with_schema: usize,
    /// 数据中的实体数（关系端点并集）。
    pub entities: usize,
    /// 能推出至少一个类型的实体数。
    pub typed_entities: usize,
    /// 关系三元组数。
    pub relations: usize,
    /// 主语与宾语都已知类型的三元组数。
    pub typed_relations: usize,
    /// 数据中完全没有 schema 的谓词（排序、有界）。
    pub untyped_predicates: Vec<String>,
}

impl SchemaCoverage {
    pub fn from_relations(
        view: &SchemaView,
        relations: &[(String, String, String)],
        explicit: &BTreeMap<String, BTreeSet<String>>,
    ) -> Self {
        let mut predicates: BTreeSet<String> = BTreeSet::new();
        let mut entities: BTreeSet<String> = BTreeSet::new();
        for (subject, predicate, object) in relations {
            predicates.insert(view.canonical_property(predicate));
            if !subject.trim().is_empty() {
                entities.insert(subject.trim().to_string());
            }
            if !object.trim().is_empty() {
                entities.insert(object.trim().to_string());
            }
        }
        let inferred = view.infer_entity_types(relations, explicit);
        let typed_entities: BTreeSet<&String> = entities
            .iter()
            .filter(|entity| {
                inferred
                    .get(*entity)
                    .map(|types| !types.is_empty())
                    .unwrap_or(false)
            })
            .collect();
        let typed_relations = relations
            .iter()
            .filter(|(subject, _, object)| {
                inferred.contains_key(subject.trim()) && inferred.contains_key(object.trim())
            })
            .count();
        let properties_with_schema = predicates
            .iter()
            .filter(|predicate| view.has_property_schema(predicate))
            .count();
        let untyped_predicates = predicates
            .iter()
            .filter(|predicate| !view.has_property_schema(predicate))
            .take(16)
            .cloned()
            .collect();
        Self {
            classes: view.declared_classes.len(),
            subclass_edges: view.subclass_edge_count(),
            disjoint_pairs: view.disjoint.len(),
            properties: view.declared_properties.len(),
            properties_with_domain: view.domains.len(),
            properties_with_range: view.ranges.len(),
            properties_with_super: view.subproperties.len(),
            data_predicates: predicates.len(),
            predicates_with_schema: properties_with_schema,
            entities: entities.len(),
            typed_entities: typed_entities.len(),
            relations: relations.len(),
            typed_relations,
            untyped_predicates,
        }
    }

    pub fn typed_entity_rate(&self) -> f64 {
        rate(self.typed_entities, self.entities)
    }

    pub fn predicate_schema_rate(&self) -> f64 {
        rate(self.predicates_with_schema, self.data_predicates)
    }

    pub fn typed_relation_rate(&self) -> f64 {
        rate(self.typed_relations, self.relations)
    }

    /// schema 稀疏：没有任何类型证据，或过半数据谓词没有 schema。
    /// 这不是「失败」，只是提醒不要把 schema 通道的收益当作既成事实。
    pub fn is_sparse(&self) -> bool {
        self.entities == 0 || self.typed_entity_rate() <= 0.5 || self.predicate_schema_rate() <= 0.5
    }

    pub fn sparse_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.entities > 0 && self.typed_entity_rate() <= 0.5 {
            reasons.push(format!(
                "实体类型覆盖率 {:.0}%（{}/{}）：至少一半实体在开放世界里仍是未知类型",
                self.typed_entity_rate() * 100.0,
                self.typed_entities,
                self.entities
            ));
        }
        if self.data_predicates > 0 && self.predicate_schema_rate() <= 0.5 {
            reasons.push(format!(
                "谓词 schema 覆盖率 {:.0}%（{}/{}）：至少一半谓词没有定义域/值域声明",
                self.predicate_schema_rate() * 100.0,
                self.predicates_with_schema,
                self.data_predicates
            ));
        }
        if self.entities == 0 {
            reasons.push("数据中没有关系实体，无法审计 schema 覆盖".to_string());
        }
        reasons
    }

    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "RDFS 模式覆盖率：类 {}、子类边 {}、互斥对 {}；属性 {}（域 {} / 值域 {} / 上位 {}）。",
            self.classes,
            self.subclass_edges,
            self.disjoint_pairs,
            self.properties,
            self.properties_with_domain,
            self.properties_with_range,
            self.properties_with_super
        ));
        lines.push(format!(
            "  数据谓词：{}/{} 有 schema（{:.0}%）；实体类型：{}/{}（{:.0}%）；两端已知类型的关系：{}/{}（{:.0}%）。",
            self.predicates_with_schema,
            self.data_predicates,
            self.predicate_schema_rate() * 100.0,
            self.typed_entities,
            self.entities,
            self.typed_entity_rate() * 100.0,
            self.typed_relations,
            self.relations,
            self.typed_relation_rate() * 100.0
        ));
        if !self.untyped_predicates.is_empty() {
            lines.push(format!(
                "  无 schema 的谓词（前 {} 个）：{}",
                self.untyped_predicates.len(),
                self.untyped_predicates.join("、")
            ));
        }
        if self.is_sparse() {
            lines.push("  警告：schema 覆盖稀疏，类型化通道多数时候只会给出 Unknown；".to_string());
            for reason in self.sparse_reasons() {
                lines.push(format!("    - {}", reason));
            }
        }
        lines.join("\n")
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// 从 `[type_system]` 构造一个不含谓词画像的最小视图（测试与轻量调用）。
pub fn schema_from_type_system(types: &TypeSystemConfig) -> SchemaView {
    SchemaView::from_config(types, &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSubclassConfig};

    fn mixed_schema() -> SchemaView {
        SchemaView::from_config(
            &TypeSystemConfig {
                aliases: vec![TypeAliasConfig {
                    alias: "Person".into(),
                    canonical: "人物".into(),
                }],
                subclasses: vec![
                    TypeSubclassConfig {
                        sub: "学生".into(),
                        sup: "人物".into(),
                    },
                    TypeSubclassConfig {
                        sub: "人物".into(),
                        sup: "实体".into(),
                    },
                    TypeSubclassConfig {
                        sub: "地点".into(),
                        sup: "实体".into(),
                    },
                ],
                disjoint: vec![TypeDisjointConfig {
                    left: "人物".into(),
                    right: "地点".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[
                PredicateProfileConfig {
                    name: "师傅".into(),
                    aliases: vec!["老师".into()],
                    domain: vec!["人物".into()],
                    range: vec!["人物".into()],
                    ..PredicateProfileConfig::default()
                },
                PredicateProfileConfig {
                    name: "学生导师".into(),
                    subproperty_of: Some("师傅".into()),
                    domain: vec!["学生".into()],
                    ..PredicateProfileConfig::default()
                },
            ],
        )
    }

    #[test]
    fn subclass_and_alias_closure_is_transitive() {
        let view = mixed_schema();
        assert!(view.is_subclass("学生", "实体"));
        assert!(view.is_subclass("Person", "实体"));
        assert!(view.types_compatible("人物", "学生"));
        assert!(!view.types_compatible("人物", "地点"));
        assert_eq!(
            view.class_ancestors("学生"),
            ["人物", "学生", "实体"]
                .into_iter()
                .map(str::to_string)
                .collect()
        );
    }

    #[test]
    fn disjoint_requires_explicit_evidence_and_respects_subclasses() {
        let view = mixed_schema();
        assert!(view.types_disjoint("学生", "地点"));
        assert!(!view.types_disjoint("学生", "公司"));
        assert_eq!(
            view.sets_admit(
                &["学生".to_string()].into_iter().collect(),
                &["人物".to_string()].into_iter().collect()
            ),
            TypeVerdict::Compatible
        );
        assert_eq!(
            view.sets_admit(
                &["地点".to_string()].into_iter().collect(),
                &["人物".to_string()].into_iter().collect()
            ),
            TypeVerdict::Disjoint
        );
        assert_eq!(
            view.sets_admit(
                &["公司".to_string()].into_iter().collect(),
                &["人物".to_string()].into_iter().collect()
            ),
            TypeVerdict::Unknown
        );
    }

    #[test]
    fn subproperty_lift_and_domain_range_inheritance() {
        let view = mixed_schema();
        assert_eq!(view.canonical_property("老师"), "师傅");
        assert!(view
            .superproperties("学生导师")
            .contains(&"师傅".to_string()));
        assert!(view
            .domain_classes("学生导师")
            .contains(&"学生".to_string()));
        assert!(view.range_classes("学生导师").contains(&"人物".to_string()));
        assert_eq!(
            view.range_verdict("学生导师", &["地点".to_string()].into_iter().collect()),
            TypeVerdict::Disjoint
        );
        assert_eq!(
            view.range_verdict("学生导师", &["学生".to_string()].into_iter().collect()),
            TypeVerdict::Compatible
        );
    }

    #[test]
    fn edge_verdict_reports_reasons_and_overall() {
        let view = mixed_schema();
        let verdict = view.edge_verdict(
            &["学生".to_string()].into_iter().collect(),
            "学生导师",
            &["地点".to_string()].into_iter().collect(),
        );
        assert_eq!(verdict.subject, TypeVerdict::Compatible);
        assert_eq!(verdict.object, TypeVerdict::Disjoint);
        assert_eq!(verdict.overall, TypeVerdict::Disjoint);
        assert!(verdict.reasons.iter().any(|line| line.contains("学生导师")));
    }

    #[test]
    fn inference_uses_domain_and_range_without_test_leakage() {
        let view = mixed_schema();
        let relations = vec![
            ("小明".to_string(), "老师".to_string(), "王五".to_string()),
            (
                "某地".to_string(),
                "未知谓词".to_string(),
                "小明".to_string(),
            ),
        ];
        let inferred = view.infer_entity_types(&relations, &BTreeMap::new());
        assert!(inferred["小明"].contains("人物"));
        assert!(inferred["王五"].contains("人物"));
        assert!(!inferred.contains_key("某地"));
    }

    #[test]
    fn coverage_is_honest_about_sparse_schema() {
        let view = mixed_schema();
        let relations = vec![
            ("小明".to_string(), "老师".to_string(), "王五".to_string()),
            (
                "某地".to_string(),
                "未知谓词".to_string(),
                "某物".to_string(),
            ),
        ];
        let coverage = view.coverage(&relations, &BTreeMap::new());
        assert_eq!(coverage.data_predicates, 2);
        assert_eq!(coverage.predicates_with_schema, 1);
        assert_eq!(coverage.entities, 4);
        assert_eq!(coverage.typed_entities, 2);
        assert!(coverage.is_sparse());
        assert!(coverage.render_text().contains("schema 覆盖稀疏"));
        assert!(coverage
            .untyped_predicates
            .contains(&"未知谓词".to_string()));
    }

    #[test]
    fn digest_is_stable_and_content_sensitive() {
        let first = mixed_schema();
        let second = mixed_schema();
        assert_eq!(first.digest(), second.digest());
        let mut changed = mixed_schema();
        changed.declare_subclass("公司", "实体");
        assert_ne!(first.digest(), changed.digest());
    }

    #[test]
    fn ontology_program_lessons_are_idempotent() {
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "学生".into(),
                sup: "人物".into(),
                original: "学生是一种人物".into(),
            },
            OntologyLesson::Subproperty {
                sub: "导师".into(),
                sup: "师傅".into(),
                original: "导师是师傅的子属性".into(),
            },
            OntologyLesson::Domain {
                property: "导师".into(),
                class: "学生".into(),
                original: "导师关系的定义域是学生".into(),
            },
        ]);
        let first = SchemaView::from_ontology_program(&program);
        let second = SchemaView::from_ontology_program(&program);
        assert_eq!(first.report(), second.report());
        assert!(first.is_subclass("学生", "人物"));
        assert!(first.superproperties("导师").contains(&"师傅".to_string()));
    }
}

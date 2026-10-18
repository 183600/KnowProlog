// 本体推理（TBox + ABox 闭包；纯符号，零神经网络、零训练）
//
// 灵感：本体（ontology）用显式的类层级与属性公理描述领域知识：
//   • TBox：A ⊑ B（子类）、p ⊑ q（子关系）、p 的定义域/值域、
//           p 传递/对称、p 与 q 互逆、类 A 与 B 互斥、类的固有属性；
//   • ABox：个体属于某类、个体之间的事实关系。
// 本模块在内存中对这些公理做闭包物化（RDFS/OWL 规则的一个实用子集）：
//   1. A ⊑ B 的传递闭包（含自反）；    2. 子关系传播；
//   3. 个体类型沿子类上推；            4. 由关系定义域/值域反推个体类型；
//   5. 类固有属性被实例继承；          6. 传递/对称/逆关系闭包；
//   7. 一致性检查：互斥类冲突与不可满足类。
// 每一步都记录“为什么”（推导链），供生成式回答逐条复述——结论可溯源、可逐字回放。

use std::collections::{HashMap, HashSet};

const MAX_DERIVED_RELATIONS: usize = 4000;

/// 物化后的本体闭包：查询与报告都基于它
#[derive(Debug, Clone, Default)]
pub struct OntologyClosure {
    subclasses: HashMap<String, HashSet<String>>,
    subproperties: HashMap<String, HashSet<String>>,
    instances: HashSet<(String, String)>,
    relations: HashSet<(String, String, String)>,
    why_instances: HashMap<(String, String), String>,
    why_relations: HashMap<(String, String, String), String>,
    problems: Vec<String>,
    rounds: usize,
    truncated: bool,
}

impl OntologyClosure {
    /// a ⊑ b（闭包含自反：a ⊑ a 恒真）
    pub fn is_subclass(&self, a: &str, b: &str) -> bool {
        self.subclasses
            .get(a)
            .map(|supers| supers.contains(b))
            .unwrap_or(a == b)
    }

    /// 个体 x 是否（直接或推导地）属于类 c；类名自身也视为该类的个体（自反）
    pub fn is_instance(&self, x: &str, c: &str) -> bool {
        self.instances.contains(&(x.to_string(), c.to_string())) || self.is_subclass(x, c)
    }

    /// 个体 x 的全部所属概念（排序去重）
    pub fn types_of(&self, x: &str) -> Vec<String> {
        let mut types: Vec<String> = self
            .instances
            .iter()
            .filter(|(individual, _)| individual == x)
            .map(|(_, class)| class.clone())
            .collect();
        types.sort();
        types.dedup();
        types
    }

    /// 类 c 的全部已知实例（排序去重）
    pub fn instances_of(&self, c: &str) -> Vec<String> {
        let mut individuals: Vec<String> = self
            .instances
            .iter()
            .filter(|(_, class)| class == c)
            .map(|(individual, _)| individual.clone())
            .collect();
        individuals.sort();
        individuals.dedup();
        individuals
    }

    /// 类 c 的全部严格上位类（不含自身，排序）
    pub fn ancestors_of(&self, c: &str) -> Vec<String> {
        let mut supers: Vec<String> = self
            .subclasses
            .get(c)
            .map(|supers| {
                supers
                    .iter()
                    .filter(|candidate| candidate.as_str() != c)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        supers.sort();
        supers
    }

    /// 类 c 的全部严格下位类（排序）
    pub fn descendants_of(&self, c: &str) -> Vec<String> {
        let mut found: HashSet<String> = HashSet::new();
        for (sub, supers) in &self.subclasses {
            if sub != c && supers.contains(c) {
                found.insert(sub.clone());
            }
        }
        let mut result: Vec<String> = found.into_iter().collect();
        result.sort();
        result
    }

    /// 个体/类 x 的全部属性关系（排序去重）
    pub fn properties_of(&self, x: &str) -> Vec<(String, String)> {
        let mut properties: Vec<(String, String)> = self
            .relations
            .iter()
            .filter(|(subject, _, _)| subject == x)
            .map(|(_, predicate, object)| (predicate.clone(), object.clone()))
            .collect();
        properties.sort();
        properties.dedup();
        properties
    }

    /// 关系三元组快照（排序，便于确定性报告与测试）
    pub fn relations_snapshot(&self) -> Vec<(String, String, String)> {
        let mut relations: Vec<(String, String, String)> =
            self.relations.iter().cloned().collect();
        relations.sort();
        relations
    }

    /// 实例断言快照（排序）
    pub fn instances_snapshot(&self) -> Vec<(String, String)> {
        let mut instances: Vec<(String, String)> = self.instances.iter().cloned().collect();
        instances.sort();
        instances
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    pub fn class_count(&self) -> usize {
        self.subclasses.len()
    }

    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    pub fn rounds(&self) -> usize {
        self.rounds
    }

    pub fn truncated(&self) -> bool {
        self.truncated
    }

    pub fn why_instance(&self, x: &str, c: &str) -> Option<&str> {
        self.why_instances
            .get(&(x.to_string(), c.to_string()))
            .map(|reason| reason.as_str())
    }

    pub fn why_relation(&self, s: &str, p: &str, o: &str) -> Option<&str> {
        self.why_relations
            .get(&(s.to_string(), p.to_string(), o.to_string()))
            .map(|reason| reason.as_str())
    }
}

/// 一个属性的本体画像（供「本体推理：X关系」报告）
#[derive(Debug, Clone, Default)]
pub struct PropertyProfile {
    pub domains: Vec<String>,
    pub ranges: Vec<String>,
    pub subproperties: Vec<String>,
    pub superproperties: Vec<String>,
    pub transitive: bool,
    pub symmetric: bool,
    pub inverses: Vec<String>,
    pub closure_relations: usize,
}

/// 本体推理器：收集 TBox/ABox 公理，按需做闭包物化
#[derive(Debug, Clone)]
pub struct OntologyReasoner {
    subclasses: HashMap<String, HashSet<String>>,
    subproperties: HashMap<String, HashSet<String>>,
    domains: HashMap<String, HashSet<String>>,
    ranges: HashMap<String, HashSet<String>>,
    transitive: HashSet<String>,
    symmetric: HashSet<String>,
    inverses: Vec<(String, String)>,
    disjoint: Vec<(String, String)>,
    instances: HashSet<(String, String)>,
    relations: HashSet<(String, String, String)>,
    class_properties: HashSet<(String, String, String)>,
    max_rounds: usize,
}

impl OntologyReasoner {
    pub fn new(max_rounds: usize) -> Self {
        Self {
            subclasses: HashMap::new(),
            subproperties: HashMap::new(),
            domains: HashMap::new(),
            ranges: HashMap::new(),
            transitive: HashSet::new(),
            symmetric: HashSet::new(),
            inverses: Vec::new(),
            disjoint: Vec::new(),
            instances: HashSet::new(),
            relations: HashSet::new(),
            class_properties: HashSet::new(),
            max_rounds: max_rounds.max(1),
        }
    }

    pub fn add_subclass(&mut self, sub: &str, sup: &str) {
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return;
        }
        self.subclasses
            .entry(sub.to_string())
            .or_default()
            .insert(sup.to_string());
    }

    pub fn add_instance(&mut self, individual: &str, class: &str) {
        if individual.is_empty() || class.is_empty() {
            return;
        }
        self.instances
            .insert((individual.to_string(), class.to_string()));
    }

    pub fn add_relation(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        self.relations.insert((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
    }

    pub fn add_disjoint(&mut self, left: &str, right: &str) {
        if left.is_empty() || right.is_empty() || left == right {
            return;
        }
        let pair = if left <= right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        };
        if !self.disjoint.contains(&pair) {
            self.disjoint.push(pair);
        }
    }

    pub fn add_subproperty(&mut self, sub: &str, sup: &str) {
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return;
        }
        self.subproperties
            .entry(sub.to_string())
            .or_default()
            .insert(sup.to_string());
    }

    pub fn add_domain(&mut self, property: &str, class: &str) {
        if property.is_empty() || class.is_empty() {
            return;
        }
        self.domains
            .entry(property.to_string())
            .or_default()
            .insert(class.to_string());
    }

    pub fn add_range(&mut self, property: &str, class: &str) {
        if property.is_empty() || class.is_empty() {
            return;
        }
        self.ranges
            .entry(property.to_string())
            .or_default()
            .insert(class.to_string());
    }

    pub fn add_transitive(&mut self, property: &str) {
        if !property.is_empty() {
            self.transitive.insert(property.to_string());
        }
    }

    pub fn add_symmetric(&mut self, property: &str) {
        if !property.is_empty() {
            self.symmetric.insert(property.to_string());
        }
    }

    pub fn add_inverse(&mut self, left: &str, right: &str) {
        if left.is_empty() || right.is_empty() || left == right {
            return;
        }
        let pair = if left <= right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        };
        if !self.inverses.contains(&pair) {
            self.inverses.push(pair);
        }
    }

    pub fn add_class_property(&mut self, class: &str, property: &str, object: &str) {
        if class.is_empty() || property.is_empty() || object.is_empty() {
            return;
        }
        self.class_properties.insert((
            class.to_string(),
            property.to_string(),
            object.to_string(),
        ));
    }

    pub fn declared_subclass_count(&self) -> usize {
        self.subclasses.values().map(|supers| supers.len()).sum()
    }

    pub fn declared_instance_count(&self) -> usize {
        self.instances.len()
    }

    pub fn declared_relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn axiom_count(&self) -> usize {
        self.disjoint.len()
            + self.subproperties.values().map(|s| s.len()).sum::<usize>()
            + self.domains.values().map(|s| s.len()).sum::<usize>()
            + self.ranges.values().map(|s| s.len()).sum::<usize>()
            + self.transitive.len()
            + self.symmetric.len()
            + self.inverses.len()
            + self.class_properties.len()
    }

    pub fn class_names(&self) -> Vec<String> {
        let mut names: HashSet<String> = HashSet::new();
        for (sub, supers) in &self.subclasses {
            names.insert(sub.clone());
            names.extend(supers.iter().cloned());
        }
        for (_, class) in &self.instances {
            names.insert(class.clone());
        }
        let mut result: Vec<String> = names.into_iter().collect();
        result.sort();
        result
    }

    pub fn has_property_knowledge(&self, property: &str) -> bool {
        self.domains.contains_key(property)
            || self.ranges.contains_key(property)
            || self.subproperties.contains_key(property)
            || self.transitive.contains(property)
            || self.symmetric.contains(property)
            || self
                .inverses
                .iter()
                .any(|(left, right)| left == property || right == property)
            || self.relations.iter().any(|(_, p, _)| p == property)
    }

    /// 属性画像：定义域/值域/子关系/父关系/传递/对称/互逆 + 闭包规模
    pub fn property_profile(&self, property: &str, closure: &OntologyClosure) -> PropertyProfile {
        let mut profile = PropertyProfile {
            transitive: self.transitive.contains(property),
            symmetric: self.symmetric.contains(property),
            closure_relations: closure
                .relations
                .iter()
                .filter(|(_, p, _)| p == property)
                .count(),
            ..PropertyProfile::default()
        };
        for (sub, supers) in &self.subproperties {
            if supers.contains(property) {
                profile.subproperties.push(sub.clone());
            }
            if sub == property {
                profile.superproperties.extend(supers.iter().cloned());
            }
        }
        if let Some(classes) = self.domains.get(property) {
            profile.domains.extend(classes.iter().cloned());
        }
        if let Some(classes) = self.ranges.get(property) {
            profile.ranges.extend(classes.iter().cloned());
        }
        for (left, right) in &self.inverses {
            if left == property {
                profile.inverses.push(right.clone());
            }
            if right == property {
                profile.inverses.push(left.clone());
            }
        }
        profile.domains.sort();
        profile.ranges.sort();
        profile.subproperties.sort();
        profile.superproperties.sort();
        profile.inverses.sort();
        profile
    }

    /// 闭包物化：把公理反复应用直到不动点；每步记录推导理由
    pub fn materialize(&self) -> OntologyClosure {
        let mut closure = OntologyClosure {
            subclasses: transitive_closure(&self.subclasses),
            subproperties: transitive_closure(&self.subproperties),
            instances: self.instances.clone(),
            relations: self.relations.clone(),
            ..OntologyClosure::default()
        };
        for (individual, class) in closure.instances.iter() {
            closure
                .why_instances
                .insert((individual.clone(), class.clone()), "直接声明".to_string());
        }
        for (subject, predicate, object) in closure.relations.iter() {
            closure.why_relations.insert(
                (subject.clone(), predicate.clone(), object.clone()),
                "直接声明".to_string(),
            );
        }
        // 超类/上位属性/定义域/值域来自固定的 TBox 闭包：整次物化内缓存
        let mut supers_cache_props: HashMap<String, Vec<String>> = HashMap::new();
        let mut supers_cache_classes: HashMap<String, Vec<String>> = HashMap::new();
        let mut domain_cache: HashMap<String, Vec<String>> = HashMap::new();
        let mut range_cache: HashMap<String, Vec<String>> = HashMap::new();
        for round in 0..self.max_rounds {
            closure.rounds = round + 1;
            let mut changed = false;
            // 1. 子关系传播
            let snapshot: Vec<(String, String, String)> =
                closure.relations.iter().cloned().collect();
            for (subject, predicate, object) in snapshot {
                let supers = supers_cache_props
                    .entry(predicate.clone())
                    .or_insert_with(|| supers_of(&closure.subproperties, &predicate));
                for super_predicate in supers.iter() {
                    if *super_predicate == predicate {
                        continue;
                    }
                    derive_relation(
                        &mut closure,
                        &subject,
                        super_predicate,
                        &object,
                        format!("{} ⊑ {}", predicate, super_predicate),
                        &mut changed,
                    );
                }
            }
            // 2. 个体类型沿子类上推
            let snapshot: Vec<(String, String)> = closure.instances.iter().cloned().collect();
            for (individual, class) in snapshot {
                let supers = supers_cache_classes
                    .entry(class.clone())
                    .or_insert_with(|| supers_of(&closure.subclasses, &class));
                for super_class in supers.iter() {
                    if *super_class == class {
                        continue;
                    }
                    derive_instance(
                        &mut closure,
                        &individual,
                        super_class,
                        format!("{} ⊑ {}", class, super_class),
                        &mut changed,
                    );
                }
            }
            // 3. 关系定义域/值域反推个体类型
            let snapshot: Vec<(String, String, String)> =
                closure.relations.iter().cloned().collect();
            for (subject, predicate, object) in snapshot {
                let classes = domain_cache
                    .entry(predicate.clone())
                    .or_insert_with(|| self.domain_classes(&predicate, &closure.subproperties));
                for class in classes.iter() {
                    derive_instance(
                        &mut closure,
                        &subject,
                        class,
                        format!("关系 {}({}, {}) 的定义域是 {}", predicate, subject, object, class),
                        &mut changed,
                    );
                }
                let classes = range_cache
                    .entry(predicate.clone())
                    .or_insert_with(|| self.range_classes(&predicate, &closure.subproperties));
                for class in classes.iter() {
                    derive_instance(
                        &mut closure,
                        &object,
                        class,
                        format!("关系 {}({}, {}) 的值域是 {}", predicate, subject, object, class),
                        &mut changed,
                    );
                }
            }
            // 4. 类固有属性被实例继承（类自身保留一份，便于类级查询）
            let class_properties: Vec<(String, String, String)> =
                self.class_properties.iter().cloned().collect();
            let mut round_members: HashMap<String, Vec<String>> = HashMap::new();
            for (class, predicate, object) in class_properties {
                derive_relation(
                    &mut closure,
                    &class,
                    &predicate,
                    &object,
                    format!("类 {} 的固有属性", class),
                    &mut changed,
                );
                let members = round_members.entry(class.clone()).or_insert_with(|| {
                    closure
                        .instances
                        .iter()
                        .filter(|(_, member_class)| closure.is_subclass(member_class, &class))
                        .map(|(individual, _)| individual.clone())
                        .collect()
                });
                for individual in members.iter() {
                    derive_relation(
                        &mut closure,
                        &individual,
                        &predicate,
                        &object,
                        format!("类 {} 的固有属性，被实例 {} 继承", class, individual),
                        &mut changed,
                    );
                }
            }
            // 5. 传递关系闭包：按中间节点分组做邻接交叉
            for property in &self.transitive {
                let pairs: Vec<(String, String)> = closure
                    .relations
                    .iter()
                    .filter(|(_, p, _)| p == property)
                    .map(|(s, _, o)| (s.clone(), o.clone()))
                    .collect();
                if pairs.len() < 2 {
                    continue;
                }
                let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();
                let mut incoming: HashMap<&str, Vec<&str>> = HashMap::new();
                for (left, right) in &pairs {
                    outgoing
                        .entry(left.as_str())
                        .or_default()
                        .push(right.as_str());
                    incoming
                        .entry(right.as_str())
                        .or_default()
                        .push(left.as_str());
                }
                for (middle, rights) in &outgoing {
                    let Some(lefts) = incoming.get(middle) else {
                        continue;
                    };
                    for left in lefts {
                        for right in rights {
                            if left != right {
                                derive_relation(
                                    &mut closure,
                                    left,
                                    property,
                                    right,
                                    format!("{} 具有传递性", property),
                                    &mut changed,
                                );
                            }
                        }
                    }
                }
            }
            // 6. 对称关系
            let snapshot: Vec<(String, String, String)> =
                closure.relations.iter().cloned().collect();
            for (subject, predicate, object) in snapshot {
                if self.symmetric.contains(&predicate) && subject != object {
                    derive_relation(
                        &mut closure,
                        &object,
                        &predicate,
                        &subject,
                        format!("{} 是对称关系", predicate),
                        &mut changed,
                    );
                }
            }
            // 7. 逆关系
            let snapshot: Vec<(String, String, String)> =
                closure.relations.iter().cloned().collect();
            for (subject, predicate, object) in snapshot {
                for (left, right) in &self.inverses {
                    let inverse = if *left == predicate {
                        Some(right.clone())
                    } else if *right == predicate {
                        Some(left.clone())
                    } else {
                        None
                    };
                    if let Some(inverse) = inverse {
                        derive_relation(
                            &mut closure,
                            &object,
                            &inverse,
                            &subject,
                            format!("{} 与 {} 互为逆关系", predicate, inverse),
                            &mut changed,
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }
        self.detect_problems(&mut closure);
        closure
    }

    fn domain_classes(
        &self,
        property: &str,
        subproperties: &HashMap<String, HashSet<String>>,
    ) -> Vec<String> {
        let mut classes: HashSet<String> = HashSet::new();
        for candidate in supers_of(subproperties, property) {
            if let Some(direct) = self.domains.get(&candidate) {
                classes.extend(direct.iter().cloned());
            }
        }
        let mut result: Vec<String> = classes.into_iter().collect();
        result.sort();
        result
    }

    fn range_classes(
        &self,
        property: &str,
        subproperties: &HashMap<String, HashSet<String>>,
    ) -> Vec<String> {
        let mut classes: HashSet<String> = HashSet::new();
        for candidate in supers_of(subproperties, property) {
            if let Some(direct) = self.ranges.get(&candidate) {
                classes.extend(direct.iter().cloned());
            }
        }
        let mut result: Vec<String> = classes.into_iter().collect();
        result.sort();
        result
    }

    /// 一致性检查：互斥类冲突 + 不可满足类
    fn detect_problems(&self, closure: &mut OntologyClosure) {
        let mut classes_of: HashMap<String, HashSet<String>> = HashMap::new();
        for (individual, class) in &closure.instances {
            classes_of
                .entry(individual.clone())
                .or_default()
                .insert(class.clone());
        }
        let mut seen: HashSet<String> = HashSet::new();
        for (left, right) in &self.disjoint {
            let left_supers = supers_set(&closure.subclasses, left);
            let right_supers = supers_set(&closure.subclasses, right);
            for (individual, classes) in &classes_of {
                let in_left = classes.iter().any(|class| left_supers.contains(class));
                let in_right = classes.iter().any(|class| right_supers.contains(class));
                if in_left && in_right {
                    let message = format!(
                        "个体「{}」同时属于互斥类「{}」与「{}」（不一致）",
                        individual, left, right
                    );
                    if seen.insert(message.clone()) {
                        closure.problems.push(message);
                    }
                }
            }
            for (class, supers) in &closure.subclasses {
                // 互斥成员自身也可能因继承对方而不可满足（如「鸟 ⊑ 鱼」且鸟/鱼互斥），
                // 因此这里不能跳过 left/right
                let in_left = supers.iter().any(|super_class| left_supers.contains(super_class));
                let in_right = supers.iter().any(|super_class| right_supers.contains(super_class));
                if in_left && in_right {
                    let message = format!(
                        "类「{}」不可满足：同时 ⊑「{}」与「{}」，而两者互斥",
                        class, left, right
                    );
                    if seen.insert(message.clone()) {
                        closure.problems.push(message);
                    }
                }
            }
        }
    }
}

fn derive_instance(
    closure: &mut OntologyClosure,
    individual: &str,
    class: &str,
    reason: String,
    changed: &mut bool,
) {
    let key = (individual.to_string(), class.to_string());
    if closure.instances.insert(key.clone()) {
        closure.why_instances.insert(key, reason);
        *changed = true;
    }
}

fn derive_relation(
    closure: &mut OntologyClosure,
    subject: &str,
    predicate: &str,
    object: &str,
    reason: String,
    changed: &mut bool,
) {
    if closure.relations.len() >= MAX_DERIVED_RELATIONS {
        closure.truncated = true;
        return;
    }
    let key = (
        subject.to_string(),
        predicate.to_string(),
        object.to_string(),
    );
    if closure.relations.insert(key.clone()) {
        closure.why_relations.insert(key, reason);
        *changed = true;
    }
}

fn supers_set(closure_map: &HashMap<String, HashSet<String>>, name: &str) -> HashSet<String> {
    let mut supers = closure_map.get(name).cloned().unwrap_or_default();
    supers.insert(name.to_string());
    supers
}

fn supers_of(closure_map: &HashMap<String, HashSet<String>>, name: &str) -> Vec<String> {
    let mut supers: Vec<String> = closure_map
        .get(name)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();
    if !supers.iter().any(|candidate| candidate == name) {
        supers.push(name.to_string());
    }
    supers.sort();
    supers
}

/// 自反 + 传递闭包（迭代扩展直到不动点）
fn transitive_closure(
    direct: &HashMap<String, HashSet<String>>,
) -> HashMap<String, HashSet<String>> {
    let mut names: HashSet<String> = HashSet::new();
    for (name, targets) in direct {
        names.insert(name.clone());
        names.extend(targets.iter().cloned());
    }
    let mut result: HashMap<String, HashSet<String>> = names
        .iter()
        .map(|name| {
            let mut reflexive: HashSet<String> = HashSet::new();
            reflexive.insert(name.clone());
            (name.clone(), reflexive)
        })
        .collect();
    loop {
        let mut changed = false;
        let snapshot: Vec<String> = names.iter().cloned().collect();
        for name in snapshot {
            let current = result.get(&name).cloned().unwrap_or_default();
            let mut next = current.clone();
            for reachable in &current {
                if let Some(targets) = direct.get(reachable) {
                    next.extend(targets.iter().cloned());
                }
            }
            if next != current {
                result.insert(name, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    result
}

/// 一条从中文解析出的本体知识
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OntologyLesson {
    Subclass {
        sub: String,
        sup: String,
        original: String,
    },
    Instance {
        individual: String,
        class: String,
        original: String,
    },
    Disjoint {
        left: String,
        right: String,
        original: String,
    },
    Subproperty {
        sub: String,
        sup: String,
        original: String,
    },
    Transitive {
        property: String,
        original: String,
    },
    Symmetric {
        property: String,
        original: String,
    },
    Inverse {
        left: String,
        right: String,
        original: String,
    },
    Domain {
        property: String,
        class: String,
        original: String,
    },
    Range {
        property: String,
        class: String,
        original: String,
    },
    ClassProperty {
        class: String,
        property: String,
        object: String,
        original: String,
    },
}

impl OntologyLesson {
    pub fn original(&self) -> &str {
        match self {
            OntologyLesson::Subclass { original, .. }
            | OntologyLesson::Instance { original, .. }
            | OntologyLesson::Disjoint { original, .. }
            | OntologyLesson::Subproperty { original, .. }
            | OntologyLesson::Transitive { original, .. }
            | OntologyLesson::Symmetric { original, .. }
            | OntologyLesson::Inverse { original, .. }
            | OntologyLesson::Domain { original, .. }
            | OntologyLesson::Range { original, .. }
            | OntologyLesson::ClassProperty { original, .. } => original,
        }
    }
}

/// 解析结果：成功的知识与逐条失败原因（用于教学报告）
#[derive(Debug, Clone, Default)]
pub struct OntologyParseOutcome {
    pub lessons: Vec<OntologyLesson>,
    pub errors: Vec<String>,
}

fn clean_lesson_text(text: &str) -> String {
    text.trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '。' | '！' | '!' | '？' | '?' | '，' | ',' | '、' | '：' | ':' | '“' | '”' | '"'
                    | '\'' | '\n' | '\r' | '\t' | ' '
            )
        })
        .to_string()
}

fn valid_name(name: &str) -> bool {
    let length = name.chars().count();
    (1..=24).contains(&length)
        && !name.contains(['。', '，', '？', '！', '：', '；', ' ', ',', '.', '?', '!', ';'])
}

fn split_pair(text: &str) -> Option<(String, String)> {
    for separator in ['和', '与', '跟', '、', '及'] {
        if let Some(position) = text.find(separator) {
            let left = text[..position].trim().to_string();
            let right = text[position + separator.len_utf8()..].trim().to_string();
            if valid_name(&left) && valid_name(&right) {
                return Some((left, right));
            }
        }
    }
    None
}

fn split_shi(text: &str) -> Option<(String, String)> {
    let position = text.rfind('是')?;
    let left = text[..position].trim().to_string();
    let right = text[position + '是'.len_utf8()..].trim().to_string();
    if valid_name(&left) && valid_name(&right) {
        Some((left, right))
    } else {
        None
    }
}

/// 断言标记词前是否紧邻否定词（如「不互斥」「不可以传递」「不具有对称性」）
fn assertion_negated(text: &str, position: usize) -> bool {
    let prefix = text[..position].trim_end();
    if prefix.ends_with('不')
        || prefix.ends_with('没')
        || prefix.ends_with('无')
        || prefix.ends_with('非')
    {
        return true;
    }
    [
        "没有", "不满足", "不构成", "不具备", "不具有", "不可以", "不能", "不会",
    ]
    .iter()
    .any(|negation| prefix.ends_with(negation))
}

fn extract_property_prefix(body: &str, suffixes: &[&str]) -> Option<String> {
    for suffix in suffixes {
        if let Some(position) = body.find(suffix) {
            if assertion_negated(body, position) {
                continue;
            }
            let mut property = body[..position].trim().to_string();
            for tail in ["的", "是", "为", "具有", "可以", "关系"] {
                property = property.trim_end_matches(tail).trim().to_string();
            }
            if valid_name(&property) {
                return Some(property);
            }
        }
    }
    None
}

/// 解析单句本体知识：类层级、互斥、子关系、传递/对称/逆、定义域/值域、类固有属性、个体归类
pub fn parse_ontology_lesson(text: &str) -> Result<OntologyLesson, String> {
    // 疑问句检查必须在清理标点之前：clean_lesson_text 会去掉句末的问号
    let raw = text.trim();
    if raw.contains('？') || raw.contains('?') {
        return Err(format!(
            "「{}」是疑问句，不能作为本体知识教学",
            clean_lesson_text(text)
        ));
    }
    let cleaned = clean_lesson_text(text);
    if cleaned.is_empty() {
        return Err("空句子".to_string());
    }
    let original = cleaned.clone();
    // 否定句检查：解析器只接受肯定知识，避免「企鹅不属于鸟」被解析成
    // 子类「企鹅不」这类被吞掉否定词的错误知识
    if cleaned.contains("不是")
        || cleaned.contains("不属于")
        || cleaned.contains("不一定")
        || cleaned.contains("并非")
        || cleaned.contains("不具有")
        || cleaned.contains("不满足")
        || cleaned.contains("不构成")
        || cleaned.contains("不可以")
        || cleaned.contains("不能")
    {
        return Err(format!("「{}」是否定句，本体教学只接受肯定知识", cleaned));
    }
    // 互斥：X和Y互斥 / X与Y不相容
    for marker in ["互不相容", "互相排斥", "彼此排斥", "互斥", "不相容", "排斥"] {
        if let Some(position) = cleaned.find(marker) {
            if assertion_negated(&cleaned, position) {
                continue;
            }
            if let Some((left, right)) = split_pair(&cleaned[..position]) {
                return Ok(OntologyLesson::Disjoint {
                    left,
                    right,
                    original,
                });
            }
        }
    }
    // 传递性
    if cleaned.contains("传递") {
        if let Some(property) = extract_property_prefix(
            &cleaned,
            &["关系可以传递", "具有传递性", "是可以传递的", "是可传递的", "是传递的", "可以传递", "传递"],
        ) {
            return Ok(OntologyLesson::Transitive {
                property,
                original,
            });
        }
    }
    // 对称性
    if cleaned.contains("对称") {
        if let Some(property) =
            extract_property_prefix(&cleaned, &["具有对称性", "是对称的", "对称"])
        {
            return Ok(OntologyLesson::Symmetric {
                property,
                original,
            });
        }
    }
    // 逆关系
    for marker in ["互为逆关系", "互为相反关系", "互为逆"] {
        if let Some(position) = cleaned.find(marker) {
            if assertion_negated(&cleaned, position) {
                continue;
            }
            if let Some((left, right)) = split_pair(&cleaned[..position]) {
                return Ok(OntologyLesson::Inverse {
                    left,
                    right,
                    original,
                });
            }
        }
    }
    // 子关系：X是Y的子关系
    if let Some(position) = cleaned.find("的子关系") {
        if let Some((sub, sup)) = split_shi(&cleaned[..position]) {
            return Ok(OntologyLesson::Subproperty { sub, sup, original });
        }
    }
    // 定义域 / 值域
    if let Some(position) = cleaned.find("定义域") {
        let mut property = cleaned[..position].trim().to_string();
        for tail in ["的", "关系"] {
            property = property.trim_end_matches(tail).trim().to_string();
        }
        let class = cleaned[position + "定义域".len()..]
            .trim()
            .trim_start_matches(['是', '为', '：', ':', ' '])
            .trim()
            .to_string();
        if valid_name(&property) && valid_name(&class) {
            return Ok(OntologyLesson::Domain {
                property,
                class,
                original,
            });
        }
    }
    if let Some(position) = cleaned.find("值域") {
        let mut property = cleaned[..position].trim().to_string();
        for tail in ["的", "关系"] {
            property = property.trim_end_matches(tail).trim().to_string();
        }
        let class = cleaned[position + "值域".len()..]
            .trim()
            .trim_start_matches(['是', '为', '：', ':', ' '])
            .trim()
            .to_string();
        if valid_name(&property) && valid_name(&class) {
            return Ok(OntologyLesson::Range {
                property,
                class,
                original,
            });
        }
    }
    // 子类：X是一种Y / X是一类Y / X属于Y / X是Y的一种 / X是Y的子类
    for marker in ["是一种", "是一类", "属于"] {
        if let Some(position) = cleaned.find(marker) {
            let sub = cleaned[..position].trim().to_string();
            let mut sup = cleaned[position + marker.len()..].trim().to_string();
            // 「X属于Y的一种」等后置分类词写法：剥离后缀，避免把「Y的一种」整体当作父类
            for suffix in ["的一种", "的一类", "的子类", "的子集"] {
                if let Some(stripped) = sup.strip_suffix(suffix) {
                    sup = stripped.trim().to_string();
                    break;
                }
            }
            if valid_name(&sub) && valid_name(&sup) {
                return Ok(OntologyLesson::Subclass { sub, sup, original });
            }
        }
    }
    for marker in ["的一种", "的一类", "的子类", "的子集"] {
        if let Some(position) = cleaned.find(marker) {
            if let Some((sub, sup)) = split_shi(&cleaned[..position]) {
                return Ok(OntologyLesson::Subclass { sub, sup, original });
            }
        }
    }
    // 类固有属性：X的P是O
    if let Some(position_de) = cleaned.find('的') {
        let after_de = &cleaned[position_de + '的'.len_utf8()..];
        if let Some(position_shi) = after_de.find('是') {
            let class = cleaned[..position_de].trim().to_string();
            let property = after_de[..position_shi].trim().to_string();
            let object = after_de[position_shi + '是'.len_utf8()..].trim().to_string();
            if valid_name(&class) && valid_name(&property) && valid_name(&object) {
                return Ok(OntologyLesson::ClassProperty {
                    class,
                    property,
                    object,
                    original,
                });
            }
        }
    }
    // 个体归类：X是Y
    if let Some((individual, class)) = split_shi(&cleaned) {
        return Ok(OntologyLesson::Instance {
            individual,
            class,
            original,
        });
    }
    Err(format!(
        "无法解析「{}」（支持：X是一种Y / X属于Y / X和Y互斥 / X是Y的子关系 / X关系可以传递 / X是对称的 / X和Y互为逆关系 / X关系的定义域是Y / X的P是O / X是Y）",
        cleaned
    ))
}

/// 解析多句文本（按 。；; 换行 切分），收集成功与失败
pub fn parse_ontology_lessons(text: &str) -> OntologyParseOutcome {
    let mut outcome = OntologyParseOutcome::default();
    let segments: Vec<&str> = text
        .split(['。', '；', ';', '\n'])
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in segments {
        match parse_ontology_lesson(segment) {
            Ok(lesson) => outcome.lessons.push(lesson),
            Err(reason) => outcome.errors.push(reason),
        }
    }
    outcome
}

/// 把解析出的本体知识写入本体推理器
pub fn apply_ontology_lesson(ontology: &mut OntologyReasoner, lesson: &OntologyLesson) {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => ontology.add_subclass(sub, sup),
        OntologyLesson::Instance {
            individual, class, ..
        } => ontology.add_instance(individual, class),
        OntologyLesson::Disjoint { left, right, .. } => ontology.add_disjoint(left, right),
        OntologyLesson::Subproperty { sub, sup, .. } => ontology.add_subproperty(sub, sup),
        OntologyLesson::Transitive { property, .. } => ontology.add_transitive(property),
        OntologyLesson::Symmetric { property, .. } => ontology.add_symmetric(property),
        OntologyLesson::Inverse { left, right, .. } => ontology.add_inverse(left, right),
        OntologyLesson::Domain {
            property, class, ..
        } => ontology.add_domain(property, class),
        OntologyLesson::Range {
            property, class, ..
        } => ontology.add_range(property, class),
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => ontology.add_class_property(class, property, object),
    }
}

/// 本体知识的生成式回读（中文复述）
pub fn render_ontology_lesson_zh(lesson: &OntologyLesson) -> String {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => {
            format!("「{}」是「{}」的一种（子类关系，实例与属性会沿层级继承）", sub, sup)
        }
        OntologyLesson::Instance {
            individual, class, ..
        } => format!("「{}」属于「{}」（个体归类的硬事实）", individual, class),
        OntologyLesson::Disjoint { left, right, .. } => format!(
            "「{}」与「{}」互不相容（同一事物不能同时属于两者，用于一致性检查）",
            left, right
        ),
        OntologyLesson::Subproperty { sub, sup, .. } => {
            format!("「{}」是「{}」的子关系（子关系事实自动升级为父关系事实）", sub, sup)
        }
        OntologyLesson::Transitive { property, .. } => format!(
            "「{}」关系可以传递（a{}b、b{}c ⇒ a{}c）",
            property, property, property, property
        ),
        OntologyLesson::Symmetric { property, .. } => format!(
            "「{}」关系是对称的（a{}b ⇒ b{}a）",
            property, property, property
        ),
        OntologyLesson::Inverse { left, right, .. } => {
            format!("「{}」与「{}」互为逆关系（a{}b ⇒ b{}a）", left, right, left, right)
        }
        OntologyLesson::Domain {
            property, class, ..
        } => format!(
            "「{}」关系的定义域是「{}」（主体由关系事实自动归入该类）",
            property, class
        ),
        OntologyLesson::Range {
            property, class, ..
        } => format!(
            "「{}」关系的值域是「{}」（客体由关系事实自动归入该类）",
            property, class
        ),
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => format!(
            "「{}」的{}是「{}」（类固有属性，会被全部实例继承）",
            class, property, object
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subclass_closure_is_transitive_and_reflexive() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("皇帝企鹅", "企鹅");
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_subclass("鸟", "动物");
        let closure = ontology.materialize();
        assert!(closure.is_subclass("皇帝企鹅", "动物"), "子类应传递");
        assert!(closure.is_subclass("企鹅", "企鹅"), "子类应自反");
        assert!(closure.ancestors_of("企鹅").contains(&"动物".to_string()));
        assert!(closure.descendants_of("鸟").contains(&"企鹅".to_string()));
        assert!(closure.descendants_of("鸟").contains(&"皇帝企鹅".to_string()));
    }

    #[test]
    fn test_instance_propagates_along_subclasses() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_instance("阿德", "企鹅");
        ontology.add_subclass("企鹅", "鸟");
        let closure = ontology.materialize();
        assert!(closure.is_instance("阿德", "鸟"), "个体类型应沿子类上推");
        assert!(closure.types_of("阿德").contains(&"企鹅".to_string()));
        assert!(closure.types_of("阿德").contains(&"鸟".to_string()));
        let reason = closure.why_instance("阿德", "鸟").unwrap_or("");
        assert!(reason.contains("企鹅"), "应记录推导链：{}", reason);
    }

    #[test]
    fn test_domain_and_range_infer_types() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_relation("张三", "师傅", "李四");
        ontology.add_domain("师傅", "人");
        ontology.add_range("师傅", "人");
        let closure = ontology.materialize();
        assert!(closure.is_instance("张三", "人"), "主体应由定义域归入「人」");
        assert!(closure.is_instance("李四", "人"), "客体应由值域归入「人」");
        let reason = closure.why_instance("张三", "人").unwrap_or("");
        assert!(reason.contains("定义域"), "应记录定义域推导：{}", reason);
    }

    #[test]
    fn test_class_property_inheritance_and_subproperty() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_instance("阿德", "企鹅");
        ontology.add_class_property("企鹅", "栖息地", "南极");
        ontology.add_subproperty("师徒", "师傅");
        ontology.add_relation("张三", "师徒", "李四");
        let closure = ontology.materialize();
        assert!(
            closure.properties_of("阿德").contains(&("栖息地".to_string(), "南极".to_string())),
            "类固有属性应被实例继承"
        );
        assert!(
            closure
                .relations_snapshot()
                .contains(&("张三".to_string(), "师傅".to_string(), "李四".to_string())),
            "子关系事实应升级为父关系"
        );
        let reason = closure
            .why_relation("阿德", "栖息地", "南极")
            .unwrap_or("");
        assert!(reason.contains("继承"), "应记录继承来源：{}", reason);
    }

    #[test]
    fn test_transitive_symmetric_inverse_relations() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_transitive("师傅");
        ontology.add_relation("张三", "师傅", "李四");
        ontology.add_relation("李四", "师傅", "王五");
        ontology.add_symmetric("同学");
        ontology.add_relation("甲", "同学", "乙");
        ontology.add_inverse("师傅", "徒弟");
        let closure = ontology.materialize();
        let relations = closure.relations_snapshot();
        assert!(
            relations.contains(&("张三".to_string(), "师傅".to_string(), "王五".to_string())),
            "传递闭包应产生 张三→王五"
        );
        assert!(
            relations.contains(&("乙".to_string(), "同学".to_string(), "甲".to_string())),
            "对称关系应补齐反向"
        );
        assert!(
            relations.contains(&("李四".to_string(), "徒弟".to_string(), "张三".to_string())),
            "逆关系应补齐反向"
        );
        let reason = closure.why_relation("张三", "师傅", "王五").unwrap_or("");
        assert!(reason.contains("传递"), "应记录传递性来源：{}", reason);
    }

    #[test]
    fn test_disjoint_inconsistency_reported() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_instance("怪东西", "鸟");
        ontology.add_instance("怪东西", "鱼");
        ontology.add_disjoint("鸟", "鱼");
        let closure = ontology.materialize();
        assert_eq!(closure.problems().len(), 1, "应报出一个不一致");
        assert!(
            closure.problems()[0].contains("互斥"),
            "问题应说明互斥冲突：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_unsatisfiable_class_reported() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_subclass("企鹅", "鱼");
        ontology.add_disjoint("鸟", "鱼");
        let closure = ontology.materialize();
        assert!(
            closure.problems().iter().any(|p| p.contains("不可满足")),
            "应报出不可满足类：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_disjoint_member_subclass_is_unsatisfiable() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_disjoint("鸟", "鱼");
        ontology.add_subclass("鸟", "鱼");
        let closure = ontology.materialize();
        assert!(
            closure.problems().iter().any(|p| p.contains("不可满足")),
            "互斥类自身成为对方子类时应报出不可满足：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_negated_ontology_lessons_are_rejected() {
        assert!(parse_ontology_lesson("企鹅不属于鸟").is_err());
        assert!(parse_ontology_lesson("张三不是老师").is_err());
        // 「不相容」是合法的互斥知识，不应被否定句检查误伤
        assert!(matches!(
            parse_ontology_lesson("猫和狗互不相容"),
            Ok(OntologyLesson::Disjoint { .. })
        ));
        // 含「不」的肯定类名不应被误伤
        assert!(matches!(
            parse_ontology_lesson("不锈钢是一种合金"),
            Ok(OntologyLesson::Subclass { .. })
        ));
    }

    #[test]
    fn test_parse_and_apply_chinese_ontology_lessons() {
        let outcome = parse_ontology_lessons(
            "企鹅是一种鸟；企鹅和鱼互斥。师傅关系可以传递。张三的师傅是李四。",
        );
        assert_eq!(outcome.lessons.len(), 4, "应解析出 4 条：{:?}", outcome.errors);
        assert!(outcome.errors.is_empty(), "不应有失败：{:?}", outcome.errors);
        let mut ontology = OntologyReasoner::new(6);
        for lesson in &outcome.lessons {
            apply_ontology_lesson(&mut ontology, lesson);
            assert!(!render_ontology_lesson_zh(lesson).is_empty());
        }
        let closure = ontology.materialize();
        assert!(closure.is_subclass("企鹅", "鸟"));
        assert!(
            closure
                .relations_snapshot()
                .contains(&("张三".to_string(), "师傅".to_string(), "李四".to_string()))
        );
        assert!(parse_ontology_lesson("这条知识是谁说的？").is_err());
        assert!(parse_ontology_lesson("完全看不懂的句子").is_err());
    }

    #[test]
    fn test_property_profile_collects_axioms() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_domain("师傅", "人");
        ontology.add_transitive("师傅");
        ontology.add_inverse("师傅", "徒弟");
        ontology.add_subproperty("师徒", "师傅");
        ontology.add_relation("张三", "师傅", "李四");
        let closure = ontology.materialize();
        let profile = ontology.property_profile("师傅", &closure);
        assert!(profile.transitive, "应标记传递性");
        assert!(profile.domains.contains(&"人".to_string()));
        assert!(profile.inverses.contains(&"徒弟".to_string()));
        assert!(profile.subproperties.contains(&"师徒".to_string()));
        assert!(profile.closure_relations >= 1);
    }
}

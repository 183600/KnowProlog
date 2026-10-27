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
use serde::{Deserialize, Serialize};
use crate::canonical;
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
    /// 显式互斥类对（TBox 声明，供查询按开放世界做类型约束时使用）。
    disjoint: Vec<(String, String)>,
    /// 属性 → 定义域类集（TBox 声明；供配方做主语侧类型约束）。
    domains: HashMap<String, HashSet<String>>,
    /// 属性 → 值域类集（TBox 声明）。
    ranges: HashMap<String, HashSet<String>>,
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

    /// p ⊑ q（闭包含自反：p ⊑ p 恒真）
    pub fn is_subproperty(&self, p: &str, q: &str) -> bool {
        self.subproperties
            .get(p)
            .map(|supers| supers.contains(q))
            .unwrap_or(p == q)
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
        let mut relations: Vec<(String, String, String)> = self.relations.iter().cloned().collect();
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

    /// 两个类是否可证明互斥（含子类继承）。开放世界下只有显式互斥
    /// 才算「已知冲突」，未知类型不得据此拒答。
    pub fn are_disjoint(&self, left: &str, right: &str) -> bool {
        if left == right {
            return false;
        }
        self.disjoint.iter().any(|(div_left, div_right)| {
            (self.is_subclass(left, div_left) && self.is_subclass(right, div_right))
                || (self.is_subclass(left, div_right) && self.is_subclass(right, div_left))
        })
    }

    /// 属性 p 的全部定义域类（沿上位属性继承；排序去重）。
    pub fn domain_classes(&self, property: &str) -> Vec<String> {
        classes_of(&self.domains, &self.subproperties, property)
    }

    /// 属性 p 的全部值域类（沿上位属性继承；排序去重）。
    pub fn range_classes(&self, property: &str) -> Vec<String> {
        classes_of(&self.ranges, &self.subproperties, property)
    }
}
/// 闭包的可序列化导出：全部集合按字典序规约，推导理由与截断标记一并带出。
///
/// 这是「证明离开进程」的最小工件：同一闭包永远对应同一字节串（集合迭代序
/// 不得泄漏进字节），供胶囊缓存、跨进程审计与测试比对使用。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ClosureSnapshot {
    /// 子类边的传递闭包（超类集合已排序）
    pub subclasses: Vec<(String, Vec<String>)>,
    /// 子属性边的传递闭包
    pub subproperties: Vec<(String, Vec<String>)>,
    /// 个体归属断言（含推导）
    pub instances: Vec<(String, String)>,
    /// 关系三元组（含推导）
    pub relations: Vec<(String, String, String)>,
    /// (个体, 类, 推导理由)
    pub why_instances: Vec<(String, String, String)>,
    /// (主语, 谓词, 宾语, 推导理由)
    pub why_relations: Vec<(String, String, String, String)>,
    /// 一致性问题（排序去重）
    pub problems: Vec<String>,
    pub rounds: usize,
    /// 推导关系数达到上限被截断：闭包不完整，回答不得声称「全部」
    pub truncated: bool,
}
impl OntologyClosure {
    /// 规约后的可序列化快照（确定性排序）。
    pub fn snapshot(&self) -> ClosureSnapshot {
        let mut subclasses: Vec<(String, Vec<String>)> = self
            .subclasses
            .iter()
            .map(|(sub, supers)| {
                let mut supers: Vec<String> = supers.iter().cloned().collect();
                supers.sort();
                (sub.clone(), supers)
            })
            .collect();
        subclasses.sort();
        let mut subproperties: Vec<(String, Vec<String>)> = self
            .subproperties
            .iter()
            .map(|(sub, supers)| {
                let mut supers: Vec<String> = supers.iter().cloned().collect();
                supers.sort();
                (sub.clone(), supers)
            })
            .collect();
        subproperties.sort();
        let mut instances: Vec<(String, String)> = self.instances.iter().cloned().collect();
        instances.sort();
        instances.dedup();
        let mut relations: Vec<(String, String, String)> = self.relations.iter().cloned().collect();
        relations.sort();
        relations.dedup();
        let mut why_instances: Vec<(String, String, String)> = self
            .why_instances
            .iter()
            .map(|((individual, class), why)| (individual.clone(), class.clone(), why.clone()))
            .collect();
        why_instances.sort();
        why_instances.dedup();
        let mut why_relations: Vec<(String, String, String, String)> = self
            .why_relations
            .iter()
            .map(|((subject, predicate, object), why)| {
                (
                    subject.clone(),
                    predicate.clone(),
                    object.clone(),
                    why.clone(),
                )
            })
            .collect();
        why_relations.sort();
        why_relations.dedup();
        let mut problems = self.problems.clone();
        problems.sort();
        problems.dedup();
        ClosureSnapshot {
            subclasses,
            subproperties,
            instances,
            relations,
            why_instances,
            why_relations,
            problems,
            rounds: self.rounds,
            truncated: self.truncated,
        }
    }

    /// 从快照重建闭包（缓存只作核对与展示，真相源始终是公理 + 观察事实）。
    pub fn from_snapshot(snapshot: &ClosureSnapshot) -> Self {
        let mut closure = Self {
            subclasses: snapshot
                .subclasses
                .iter()
                .map(|(sub, supers)| (sub.clone(), supers.iter().cloned().collect()))
                .collect(),
            subproperties: snapshot
                .subproperties
                .iter()
                .map(|(sub, supers)| (sub.clone(), supers.iter().cloned().collect()))
                .collect(),
            instances: snapshot.instances.iter().cloned().collect(),
            relations: snapshot.relations.iter().cloned().collect(),
            why_instances: snapshot
                .why_instances
                .iter()
                .map(|(individual, class, why)| ((individual.clone(), class.clone()), why.clone()))
                .collect(),
            why_relations: snapshot
                .why_relations
                .iter()
                .map(|(subject, predicate, object, why)| {
                    (
                        (subject.clone(), predicate.clone(), object.clone()),
                        why.clone(),
                    )
                })
                .collect(),
            problems: snapshot.problems.clone(),
            disjoint: Vec::new(),
            domains: HashMap::new(),
            ranges: HashMap::new(),
            rounds: snapshot.rounds,
            truncated: snapshot.truncated,
        };
        closure.problems.sort();
        closure.problems.dedup();
        closure
    }

    /// 闭包的确定性摘要（规范字节上的 FNV-1a）。
    pub fn digest(&self) -> String {
        digest_of(&self.snapshot())
    }
}
/// 规范摘要：内容相等 ⇒ 摘要相等（L4 摘要交换）。序列化失败返回空串并
/// 由调用方按「不可校验」处理，绝不 panic。
fn digest_of<T: Serialize>(value: &T) -> String {
    canonical::canonical_digest(value).unwrap_or_default()
}
/// 本体胶囊的格式版本（每升一版必须提供迁移；解码按版本门裁决）。
pub const ONTOLOGY_CAPSULE_VERSION: u32 = 1;
/// 本体胶囊：公理 + 摘要 + 闭包缓存的可序列化工件。
///
/// 「理解是反序列化，生成是序列化」：对话快照携带它，保存 / 分叉 / 重建之后
/// 用户教过的本体 100% 保留；缓存是派生数据不是真相源，恢复时先校验
/// `program_digest`（公理被改写即拒绝），闭包一律按公理重算，缓存只用于
/// 核对（[`Self::closure_matches`]）与离开进程的证明展示。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyCapsule {
    pub version: u32,
    /// 规范规约后的公理（排序去重，摘要与教学顺序无关）
    pub lessons: Vec<OntologyLesson>,
    /// 公理集合的规范摘要
    pub program_digest: String,
    /// 导出时刻的闭包缓存（派生数据）
    pub closure: ClosureSnapshot,
    /// 闭包缓存的规范摘要
    pub closure_digest: String,
}
impl OntologyCapsule {
    /// 用公理与当前闭包构造胶囊；公理排序去重，摘要立即计算。
    pub fn new(lessons: Vec<OntologyLesson>, closure: &OntologyClosure) -> Self {
        let lessons = canonicalize_lessons(lessons);
        let program_digest = program_digest_of(&lessons);
        let closure = closure.snapshot();
        let closure_digest = digest_of(&closure);
        Self {
            version: ONTOLOGY_CAPSULE_VERSION,
            lessons,
            program_digest,
            closure,
            closure_digest,
        }
    }

    /// 版本门 + 公理摘要校验；被改写 / 超前版本一律拒绝。
    pub fn verify(&self) -> Result<(), String> {
        if self.version == 0 || self.version > ONTOLOGY_CAPSULE_VERSION {
            return Err(format!(
                "本体胶囊版本 {} 不在支持范围 1..={}（需迁移后才能恢复）",
                self.version, ONTOLOGY_CAPSULE_VERSION
            ));
        }
        let expected = program_digest_of(&self.lessons);
        if expected != self.program_digest {
            return Err(format!(
                "本体胶囊公理摘要不匹配（发现 {}，期望 {}）：公理被改写或截断",
                self.program_digest, expected
            ));
        }
        Ok(())
    }

    /// 校验缓存闭包是否与「按公理 + 事实重算」的闭包一致（缓存可核对）。
    pub fn closure_matches(&self, closure: &OntologyClosure) -> bool {
        digest_of(&closure.snapshot()) == self.closure_digest
    }
}
/// 公理集合的规范摘要（对规范规约后的序列计算，与教学顺序无关）。
pub fn program_digest_of(lessons: &[OntologyLesson]) -> String {
    digest_of(&canonicalize_lessons(lessons.to_vec()))
}
/// 公理集合的规范规约：按规范文本排序 + 去重（摘要与教学顺序无关）。
pub fn canonicalize_lessons(lessons: Vec<OntologyLesson>) -> Vec<OntologyLesson> {
    let mut keyed: Vec<(String, OntologyLesson)> = lessons
        .into_iter()
        .map(|lesson| {
            let key = canonical::canonical_text(&lesson).unwrap_or_default();
            (key, lesson)
        })
        .collect();
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    keyed.dedup_by(|left, right| left.0 == right.0);
    keyed.into_iter().map(|(_, lesson)| lesson).collect()
}
/// 配置态本体程序：把 config.toml 的 `[type_system]` 与 `[[predicate_profiles]]`
/// 归约成同一套公理（`OntologyLesson`），与「教学本体」共用**一个**推理器与
/// 一个胶囊——配置不再有第二套本体，重启/重建后同样保留。
///
/// 只并入结构性公理：子类、互斥、定义域/值域、逆关系、传递、对称、子属性。
/// `aliases` 不入本体（别名归一由 aligner / type_system 负责，混入会造成
/// 类身份重复）；`closure_gate` 是判定档位不是公理。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyProgram {
    pub lessons: Vec<OntologyLesson>,
}
impl OntologyProgram {
    /// 从配置构造（公理顺序按声明序，装入时仍会被规约去重）。
    pub fn from_config(
        types: &crate::type_system::TypeSystemConfig,
        profiles: &[crate::predicate_registry::PredicateProfileConfig],
    ) -> Self {
        let mut lessons = Vec::new();
        for subclass in &types.subclasses {
            lessons.push(OntologyLesson::Subclass {
                sub: subclass.sub.clone(),
                sup: subclass.sup.clone(),
                original: format!("{}是一种{}", subclass.sub, subclass.sup),
            });
        }
        for pair in &types.disjoint {
            lessons.push(OntologyLesson::Disjoint {
                left: pair.left.clone(),
                right: pair.right.clone(),
                original: format!("{}和{}互斥", pair.left, pair.right),
            });
        }
        for profile in profiles {
            for class in &profile.domain {
                lessons.push(OntologyLesson::Domain {
                    property: profile.name.clone(),
                    class: class.clone(),
                    original: format!("{}关系的定义域是{}", profile.name, class),
                });
            }
            for class in &profile.range {
                lessons.push(OntologyLesson::Range {
                    property: profile.name.clone(),
                    class: class.clone(),
                    original: format!("{}关系的值域是{}", profile.name, class),
                });
            }
            if let Some(inverse) = &profile.inverse {
                lessons.push(OntologyLesson::Inverse {
                    left: profile.name.clone(),
                    right: inverse.clone(),
                    original: format!("{}和{}互为逆关系", profile.name, inverse),
                });
            }
            if profile.transitive {
                lessons.push(OntologyLesson::Transitive {
                    property: profile.name.clone(),
                    original: format!("{}关系可以传递", profile.name),
                });
            }
            if profile.symmetric {
                lessons.push(OntologyLesson::Symmetric {
                    property: profile.name.clone(),
                    original: format!("{}是对称的", profile.name),
                });
            }
            if let Some(parent) = &profile.subproperty_of {
                let parent = parent.trim();
                if !parent.is_empty() && parent != profile.name {
                    lessons.push(OntologyLesson::Subproperty {
                        sub: profile.name.clone(),
                        sup: parent.to_string(),
                        original: format!("{}是{}的子关系", profile.name, parent),
                    });
                }
            }
        }
        Self { lessons }
    }

    /// 由一组公理构造（规范排序去重，与教学顺序无关）。
    pub fn from_lessons(lessons: Vec<OntologyLesson>) -> Self {
        Self {
            lessons: canonicalize_lessons(lessons),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.lessons.is_empty()
    }

    pub fn len(&self) -> usize {
        self.lessons.len()
    }

    /// 合并另一份程序（公理集合求并，去重规约后返回总条数）。
    pub fn merge(&mut self, other: &OntologyProgram) {
        self.lessons.extend(other.lessons.iter().cloned());
        self.lessons = canonicalize_lessons(std::mem::take(&mut self.lessons));
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
        self.class_properties
            .insert((class.to_string(), property.to_string(), object.to_string()));
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
            disjoint: self.disjoint.clone(),
            domains: self.domains.clone(),
            ranges: self.ranges.clone(),
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
                        format!(
                            "关系 {}({}, {}) 的定义域是 {}",
                            predicate, subject, object, class
                        ),
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
                        format!(
                            "关系 {}({}, {}) 的值域是 {}",
                            predicate, subject, object, class
                        ),
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
        classes_of(&self.domains, subproperties, property)
    }

    fn range_classes(
        &self,
        property: &str,
        subproperties: &HashMap<String, HashSet<String>>,
    ) -> Vec<String> {
        classes_of(&self.ranges, subproperties, property)
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
            for (individual, classes) in &classes_of {
                let in_left = classes.iter().any(|class| closure.is_subclass(class, left));
                let in_right = classes
                    .iter()
                    .any(|class| closure.is_subclass(class, right));
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
                let in_left = supers.contains(left);
                let in_right = supers.contains(right);
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
/// 属性沿上位属性继承的类集（定义域/值域共用；排序去重）。
fn classes_of(
    declared: &HashMap<String, HashSet<String>>,
    subproperties: &HashMap<String, HashSet<String>>,
    property: &str,
) -> Vec<String> {
    let mut classes: HashSet<String> = HashSet::new();
    for candidate in supers_of(subproperties, property) {
        if let Some(direct) = declared.get(&candidate) {
            classes.extend(direct.iter().cloned());
        }
    }
    let mut result: Vec<String> = classes.into_iter().collect();
    result.sort();
    result
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
/// 一条从中文解析出的本体知识（可序列化：随本体胶囊跨会话携带）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
                '。' | '！'
                    | '!'
                    | '？'
                    | '?'
                    | '，'
                    | ','
                    | '、'
                    | '：'
                    | ':'
                    | '“'
                    | '”'
                    | '"'
                    | '\''
                    | '\n'
                    | '\r'
                    | '\t'
                    | ' '
            )
        })
        .to_string()
}
fn valid_name(name: &str) -> bool {
    let length = name.chars().count();
    (1..=24).contains(&length)
        && !name.contains([
            '。', '，', '？', '！', '：', '；', ' ', ',', '.', '?', '!', ';',
        ])
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
        "没有",
        "不满足",
        "不构成",
        "不具备",
        "不具有",
        "不可以",
        "不能",
        "不会",
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

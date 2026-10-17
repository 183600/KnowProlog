// src/formal_semantics.rs
//! 形式语义学（Montague 风格）：把中文句子组合式地翻译为带类型的逻辑形式，
//! 再在由知识图谱事实构成的「微型模型」上求值，并支持从逻辑形式反向生成中文答句。
//!
//! 纯符号主义实现，不引入任何神经网络或权重训练：
//! - 组合原则：函数应用（前向/后向）、广义量词（所有/有些）、话题-述题「是」规则、
//!   合取/析取/否定/蕴含，全部按类型驱动；
//! - 模型论语义：谓词外延 = 图谱中满足该谓词的实体集合；类型断言沿子类闭包传播；
//! - 生成式回答：从求解结果组合中文答句（谓词表面形来自 SymbolicGenerator），
//!   检索到的证据以引用形式附加，保证可追溯。

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::nlp::shared_jieba;
use crate::prolog_engine::PrologFact;

/// 语义类型：e（实体）、t（真值）、n（数）、以及函数类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Entity,
    Truth,
    Number,
    Fun(Box<Type>, Box<Type>),
}

impl Type {
    pub fn fun(dom: Type, cod: Type) -> Self {
        Type::Fun(Box::new(dom), Box::new(cod))
    }

    pub fn render(&self) -> String {
        match self {
            Type::Entity => "e".to_string(),
            Type::Truth => "t".to_string(),
            Type::Number => "n".to_string(),
            Type::Fun(dom, cod) => format!("({}→{})", dom.render(), cod.render()),
        }
    }
}

/// 逻辑形式项（lambda 演算 + 逻辑连接词 + 量词）
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Const(String),
    Pred(String),
    Num(String),
    Var(String),
    App(Box<Term>, Box<Term>),
    Lam(String, Box<Term>),
    Attr(Box<Term>, Box<Term>),
    Not(Box<Term>),
    And(Vec<Term>),
    Or(Vec<Term>),
    Implies(Box<Term>, Box<Term>),
    Isa(Box<Term>, Box<Term>),
    Eq(Box<Term>, Box<Term>),
    Exists { var: String, body: Box<Term> },
    Forall { var: String, body: Box<Term> },
}

impl Term {
    /// 逻辑记法渲染（用于可解释性输出）
    pub fn render(&self) -> String {
        match self {
            Term::Const(name) => name.clone(),
            Term::Pred(name) => name.clone(),
            Term::Num(text) => text.clone(),
            Term::Var(name) => name.clone(),
            Term::App(f, a) => format!("{}({})", f.render(), a.render()),
            Term::Lam(var, body) => format!("λ{}.{}", var, body.render()),
            Term::Attr(pred, x) => format!("{}的{}", x.render(), pred.render()),
            Term::Not(t) => format!("¬{}", t.render()),
            Term::And(ts) => {
                let parts: Vec<String> = ts.iter().map(Term::render).collect();
                format!("({})", parts.join(" ∧ "))
            }
            Term::Or(ts) => {
                let parts: Vec<String> = ts.iter().map(Term::render).collect();
                format!("({})", parts.join(" ∨ "))
            }
            Term::Implies(a, b) => format!("({} → {})", a.render(), b.render()),
            Term::Isa(a, b) => format!("{}({})", b.render(), a.render()),
            Term::Eq(a, b) => format!("({} = {})", a.render(), b.render()),
            Term::Exists { var, body } => format!("∃{}.{}", var, body.render()),
            Term::Forall { var, body } => format!("∀{}.{}", var, body.render()),
        }
    }

    /// 变量替换（不做捕获规避：本片段中 lambda 变量与个体常量不重名）
    fn subst(term: &Term, var: &str, value: &Term) -> Term {
        match term {
            Term::Var(name) if name == var => value.clone(),
            Term::Const(_) | Term::Pred(_) | Term::Num(_) | Term::Var(_) => term.clone(),
            Term::App(f, a) => Term::App(
                Box::new(Term::subst(f, var, value)),
                Box::new(Term::subst(a, var, value)),
            ),
            Term::Lam(name, body) if name == var => term.clone(),
            Term::Lam(name, body) => {
                Term::Lam(name.clone(), Box::new(Term::subst(body, var, value)))
            }
            Term::Attr(p, x) => Term::Attr(
                Box::new(Term::subst(p, var, value)),
                Box::new(Term::subst(x, var, value)),
            ),
            Term::Not(t) => Term::Not(Box::new(Term::subst(t, var, value))),
            Term::And(ts) => Term::And(ts.iter().map(|t| Term::subst(t, var, value)).collect()),
            Term::Or(ts) => Term::Or(ts.iter().map(|t| Term::subst(t, var, value)).collect()),
            Term::Implies(a, b) => Term::Implies(
                Box::new(Term::subst(a, var, value)),
                Box::new(Term::subst(b, var, value)),
            ),
            Term::Isa(a, b) => Term::Isa(
                Box::new(Term::subst(a, var, value)),
                Box::new(Term::subst(b, var, value)),
            ),
            Term::Eq(a, b) => Term::Eq(
                Box::new(Term::subst(a, var, value)),
                Box::new(Term::subst(b, var, value)),
            ),
            Term::Exists { var: inner, body } if inner == var => term.clone(),
            Term::Exists { var: inner, body } => Term::Exists {
                var: inner.clone(),
                body: Box::new(Term::subst(body, var, value)),
            },
            Term::Forall { var: inner, body } if inner == var => term.clone(),
            Term::Forall { var: inner, body } => Term::Forall {
                var: inner.clone(),
                body: Box::new(Term::subst(body, var, value)),
            },
        }
    }

    /// beta 归一（在 App 处展开 lambda）
    pub fn reduce(term: Term) -> Term {
        match term {
            Term::App(f, a) => {
                let reduced_f = Term::reduce(*f);
                let reduced_a = Term::reduce(*a);
                match reduced_f {
                    Term::Lam(var, body) => Term::reduce(Term::subst(&body, &var, &reduced_a)),
                    other => Term::App(Box::new(other), Box::new(reduced_a)),
                }
            }
            Term::Lam(var, body) => Term::Lam(var, Box::new(Term::reduce(*body))),
            Term::Attr(p, x) => Term::Attr(Box::new(Term::reduce(*p)), Box::new(Term::reduce(*x))),
            Term::Not(t) => Term::Not(Box::new(Term::reduce(*t))),
            Term::And(ts) => Term::And(ts.into_iter().map(Term::reduce).collect()),
            Term::Or(ts) => Term::Or(ts.into_iter().map(Term::reduce).collect()),
            Term::Implies(a, b) => {
                Term::Implies(Box::new(Term::reduce(*a)), Box::new(Term::reduce(*b)))
            }
            Term::Isa(a, b) => Term::Isa(Box::new(Term::reduce(*a)), Box::new(Term::reduce(*b))),
            Term::Eq(a, b) => Term::Eq(Box::new(Term::reduce(*a)), Box::new(Term::reduce(*b))),
            Term::Exists { var, body } => Term::Exists {
                var,
                body: Box::new(Term::reduce(*body)),
            },
            Term::Forall { var, body } => Term::Forall {
                var,
                body: Box::new(Term::reduce(*body)),
            },
            other => other,
        }
    }

    /// 函数应用（自动 beta 归一）
    pub fn apply(function: Term, argument: Term) -> Term {
        Term::reduce(Term::App(Box::new(function), Box::new(argument)))
    }

    /// 收集项中出现的个体常量（用于对话记忆的焦点提取）
    pub fn collect_consts(&self, out: &mut Vec<String>) {
        match self {
            Term::Const(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Term::App(f, a) => {
                f.collect_consts(out);
                a.collect_consts(out);
            }
            Term::Lam(_, body)
            | Term::Not(body)
            | Term::Exists { body, .. }
            | Term::Forall { body, .. } => body.collect_consts(out),
            Term::Attr(p, x) => {
                p.collect_consts(out);
                x.collect_consts(out);
            }
            Term::And(ts) | Term::Or(ts) => {
                for t in ts {
                    t.collect_consts(out);
                }
            }
            Term::Implies(a, b) | Term::Isa(a, b) | Term::Eq(a, b) => {
                a.collect_consts(out);
                b.collect_consts(out);
            }
            _ => {}
        }
    }
}

/// 词条类别（用于可读性标注）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    NP,
    N,
    IV,
    TV,
    Det,
    Conj,
    Neg,
    Cop,
    Wh,
    Num,
    Func,
}

/// 词条：表面形 → 带类型的逻辑项
#[derive(Debug, Clone)]
pub struct LexEntry {
    pub surface: String,
    pub category: Category,
    pub term: Term,
    pub ty: Type,
}

/// 词典：功能词 + 从知识图谱注册的实体/类别/谓词
pub struct Lexicon {
    entries: HashMap<String, Vec<LexEntry>>,
    entities: HashSet<String>,
    classes: HashSet<String>,
    predicates: HashSet<String>,
    binary_predicates: HashSet<String>,
    predicate_surfaces: HashMap<String, String>,
    surface_to_predicate: HashMap<String, String>,
}

impl Lexicon {
    pub fn default_chinese() -> Self {
        let mut lexicon = Self {
            entries: HashMap::new(),
            entities: HashSet::new(),
            classes: HashSet::new(),
            predicates: HashSet::new(),
            binary_predicates: HashSet::new(),
            predicate_surfaces: HashMap::new(),
            surface_to_predicate: HashMap::new(),
        };
        let e = Type::Entity;
        let t = Type::Truth;
        // 系词「是/为/属于/不是」不进入词库，而是由图表分析中的
        // 系词规则按「主语-系词-表语」顺序直接构造 Isa(x, y)，避免函数应用的参数顺序歧义。
        // 助词「的」三种组合式：属性值化、一元谓词化、关系谓词化
        let de_value = Term::Lam(
            "x".into(),
            Box::new(Term::Lam(
                "P".into(),
                Box::new(Term::Attr(
                    Box::new(Term::Var("P".into())),
                    Box::new(Term::Var("x".into())),
                )),
            )),
        );
        lexicon.add(LexEntry {
            surface: "的".to_string(),
            category: Category::Func,
            term: de_value,
            ty: Type::fun(
                e.clone(),
                Type::fun(Type::fun(e.clone(), t.clone()), e.clone()),
            ),
        });
        let de_pred = Term::Lam(
            "x".into(),
            Box::new(Term::Lam(
                "P".into(),
                Box::new(Term::App(
                    Box::new(Term::Var("P".into())),
                    Box::new(Term::Var("x".into())),
                )),
            )),
        );
        lexicon.add(LexEntry {
            surface: "的".to_string(),
            category: Category::Func,
            term: de_pred,
            ty: Type::fun(
                e.clone(),
                Type::fun(Type::fun(e.clone(), t.clone()), t.clone()),
            ),
        });
        let de_rel = Term::Lam(
            "x".into(),
            Box::new(Term::Lam(
                "R".into(),
                Box::new(Term::Lam(
                    "y".into(),
                    Box::new(Term::App(
                        Box::new(Term::App(
                            Box::new(Term::Var("R".into())),
                            Box::new(Term::Var("y".into())),
                        )),
                        Box::new(Term::Var("x".into())),
                    )),
                )),
            )),
        );
        lexicon.add(LexEntry {
            surface: "的".to_string(),
            category: Category::Func,
            term: de_rel,
            ty: Type::fun(
                e.clone(),
                Type::fun(
                    Type::fun(e.clone(), Type::fun(e.clone(), t.clone())),
                    Type::fun(e.clone(), t.clone()),
                ),
            ),
        });
        lexicon
    }

    fn add(&mut self, entry: LexEntry) {
        let key = entry.ty.render();
        let bucket = self.entries.entry(entry.surface.clone()).or_default();
        if bucket.iter().any(|existing| existing.ty.render() == key) {
            return;
        }
        bucket.push(entry);
    }

    pub fn lookup(&self, token: &str) -> Vec<LexEntry> {
        self.entries.get(token).cloned().unwrap_or_default()
    }

    pub fn register_entity(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() || !self.entities.insert(name.to_string()) {
            return;
        }
        self.add(LexEntry {
            surface: name.to_string(),
            category: Category::NP,
            term: Term::Const(name.to_string()),
            ty: Type::Entity,
        });
        // 广义量词提升：λP. P(name)
        let raised = Term::Lam(
            "P".into(),
            Box::new(Term::App(
                Box::new(Term::Var("P".into())),
                Box::new(Term::Const(name.to_string())),
            )),
        );
        self.add(LexEntry {
            surface: name.to_string(),
            category: Category::NP,
            term: raised,
            ty: Type::fun(Type::fun(Type::Entity, Type::Truth), Type::Truth),
        });
    }

    pub fn register_class(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.classes.insert(name.to_string());
        self.register_entity(name);
        if self.predicates.insert(name.to_string()) {
            self.add(LexEntry {
                surface: name.to_string(),
                category: Category::N,
                term: Term::Pred(name.to_string()),
                ty: Type::fun(Type::Entity, Type::Truth),
            });
        }
    }

    pub fn register_unary_predicate(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.register_entity(name);
        if self.predicates.insert(name.to_string()) {
            self.add(LexEntry {
                surface: name.to_string(),
                category: Category::IV,
                term: Term::Pred(name.to_string()),
                ty: Type::fun(Type::Entity, Type::Truth),
            });
        }
    }

    pub fn register_binary_predicate(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.register_entity(name);
        if self.binary_predicates.insert(name.to_string()) {
            self.add(LexEntry {
                surface: name.to_string(),
                category: Category::TV,
                term: Term::Pred(name.to_string()),
                ty: Type::fun(Type::Entity, Type::fun(Type::Entity, Type::Truth)),
            });
        }
    }

    pub fn set_predicate_surfaces(&mut self, surfaces: &HashMap<String, String>) {
        for (canonical, surface) in surfaces {
            self.predicate_surfaces
                .entry(canonical.clone())
                .or_insert_with(|| surface.clone());
            self.surface_to_predicate
                .entry(surface.clone())
                .or_insert_with(|| canonical.clone());
        }
    }

    pub fn is_entity_name(&self, name: &str) -> bool {
        self.entities.contains(name)
    }

    pub fn is_class(&self, name: &str) -> bool {
        self.classes.contains(name)
    }

    pub fn is_predicate(&self, name: &str) -> bool {
        self.predicates.contains(name) || self.binary_predicates.contains(name)
    }

    /// 在字符串中寻找最长已知实体（用于问句切分）
    pub fn find_entity(&self, text: &str) -> Option<String> {
        let mut names: Vec<&String> = self.entities.iter().collect();
        names.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then_with(|| a.cmp(b)));
        names
            .into_iter()
            .find(|name| name.chars().count() >= 2 && text.contains(name.as_str()))
            .cloned()
    }

    /// 表面形 → 规范谓词名（未知则回退原文，便于检索接地）
    pub fn resolve_predicate_surface(&self, surface: &str) -> Option<String> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        if let Some(canonical) = self.surface_to_predicate.get(surface) {
            return Some(canonical.clone());
        }
        Some(surface.to_string())
    }

    /// 规范谓词 → 中文表面形（用于生成答句）
    pub fn surface_of(&self, predicate: &str) -> String {
        self.predicate_surfaces
            .get(predicate)
            .cloned()
            .unwrap_or_else(|| predicate.to_string())
    }
}

// ---------------------------------------------------------------------------
// 微型模型：知识图谱事实上的模型论语义
// ---------------------------------------------------------------------------

/// 求值结果
#[derive(Debug, Clone, PartialEq)]
pub enum Eval {
    Truth(bool),
    Entity(String),
    Entities(Vec<String>),
    Number(f64),
    Failure(String),
}

/// 子类关系谓词（沿此关系传播类型断言）
const SUPER_RELS: [&str; 6] = ["子类", "子类别", "subclass", "subClassOf", "is_a", "isa"];
/// 对称关系（双向匹配）
const SYMMETRIC_RELS: [&str; 5] = ["配偶", "夫妻", "兄弟", "姐妹", "同学"];
/// 个体域枚举上限（量词求值）
const DOMAIN_CAP: usize = 1000;

/// 由 PrologFact 增量构建的微型模型（谓词外延 + 类型闭包 + 属性值）
pub struct TinyModel {
    entities: BTreeSet<String>,
    unary: HashMap<String, BTreeSet<String>>,
    values: HashMap<(String, String), Vec<String>>,
    binary: HashSet<(String, String, String)>,
    types: HashMap<String, BTreeSet<String>>,
    supers: HashMap<String, BTreeSet<String>>,
    class_set: HashSet<String>,
    out_index: HashMap<String, Vec<(String, String)>>,
    in_index: HashMap<String, Vec<(String, String)>>,
    synonyms: HashMap<String, String>,
    fact_count: usize,
}

impl TinyModel {
    pub fn new(synonyms: HashMap<String, String>) -> Self {
        Self {
            entities: BTreeSet::new(),
            unary: HashMap::new(),
            values: HashMap::new(),
            binary: HashSet::new(),
            types: HashMap::new(),
            supers: HashMap::new(),
            class_set: HashSet::new(),
            out_index: HashMap::new(),
            in_index: HashMap::new(),
            synonyms,
            fact_count: 0,
        }
    }

    pub fn fact_count(&self) -> usize {
        self.fact_count
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    fn canon(&self, name: &str) -> String {
        self.synonyms
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    fn name_variants(&self, name: &str) -> Vec<String> {
        let canonical = self.canon(name);
        if canonical == name {
            vec![name.to_string()]
        } else {
            vec![name.to_string(), canonical]
        }
    }

    pub fn observe_fact(&mut self, fact: &PrologFact) {
        match fact {
            PrologFact::Relation { from, rel_type, to } => {
                self.observe_relation(from, rel_type, to);
            }
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                let pairs: Vec<(String, String)> = properties
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                self.observe_entity(name, entity_type, &pairs);
            }
        }
    }

    pub fn observe_relation(&mut self, from: &str, rel_type: &str, to: &str) {
        let from = from.trim();
        let rel_type = rel_type.trim();
        let to = to.trim();
        if from.is_empty() || rel_type.is_empty() || to.is_empty() {
            return;
        }
        if !self.binary.insert((
            from.to_string(),
            rel_type.to_string(),
            to.to_string(),
        )) {
            return;
        }
        self.fact_count += 1;
        self.note_entity(from);
        self.note_entity(to);
        self.out_index
            .entry(from.to_string())
            .or_default()
            .push((rel_type.to_string(), to.to_string()));
        self.in_index
            .entry(to.to_string())
            .or_default()
            .push((rel_type.to_string(), from.to_string()));
        if SUPER_RELS.contains(&self.canon(rel_type).as_str()) {
            self.class_set.insert(from.to_string());
            self.class_set.insert(to.to_string());
            self.supers
                .entry(from.to_string())
                .or_default()
                .insert(to.to_string());
        }
    }

    pub fn observe_entity(
        &mut self,
        name: &str,
        entity_type: &str,
        properties: &[(String, String)],
    ) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.note_entity(name);
        self.fact_count += 1;
        let entity_type = entity_type.trim();
        if !entity_type.is_empty() {
            self.class_set.insert(entity_type.to_string());
            self.note_entity(entity_type);
            self.types
                .entry(name.to_string())
                .or_default()
                .insert(entity_type.to_string());
            self.unary
                .entry(entity_type.to_string())
                .or_default()
                .insert(name.to_string());
        }
        for (key, value) in properties {
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() || value.is_empty() {
                continue;
            }
            self.unary
                .entry(key.to_string())
                .or_default()
                .insert(name.to_string());
            self.values
                .entry((name.to_string(), key.to_string()))
                .or_default()
                .push(value.to_string());
            if parse_number_token(value).is_none() {
                self.note_entity(value);
            }
        }
    }

    fn note_entity(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() || parse_number_token(name).is_some() {
            return;
        }
        self.entities.insert(name.to_string());
        let canonical = self.canon(name);
        if canonical != name && parse_number_token(&canonical).is_none() {
            self.entities.insert(canonical);
        }
    }

    fn merged_unary(&self, predicate: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for variant in self.name_variants(predicate) {
            if let Some(set) = self.unary.get(&variant) {
                out.extend(set.iter().cloned());
            }
        }
        out
    }

    pub fn values_of(&self, entity: &str, predicate: &str) -> Vec<String> {
        let mut out = Vec::new();
        for e in self.name_variants(entity) {
            for p in self.name_variants(predicate) {
                if let Some(values) = self.values.get(&(e.clone(), p.clone())) {
                    for value in values {
                        if !out.contains(value) {
                            out.push(value.clone());
                        }
                    }
                }
            }
        }
        for e in self.name_variants(entity) {
            if let Some(outgoing) = self.out_index.get(&e) {
                for (rel, object) in outgoing {
                    if self.name_variants(predicate).contains(rel) && !out.contains(object) {
                        out.push(object.clone());
                    }
                }
            }
            if let Some(incoming) = self.in_index.get(&e) {
                for (rel, subject) in incoming {
                    if self.name_variants(predicate).contains(rel) && !out.contains(subject) {
                        out.push(subject.clone());
                    }
                }
            }
        }
        out
    }

    pub fn holds_unary(&self, entity: &str, predicate: &str) -> bool {
        if self.merged_unary(predicate).contains(entity) {
            return true;
        }
        if !self.values_of(entity, predicate).is_empty() {
            return true;
        }
        for e in self.name_variants(entity) {
            if let Some(outgoing) = self.out_index.get(&e) {
                for (rel, _) in outgoing {
                    if self.name_variants(predicate).contains(rel) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn holds_binary(&self, predicate: &str, subject: &str, object: &str) -> bool {
        let predicates = self.name_variants(predicate);
        let subjects = self.name_variants(subject);
        let objects = self.name_variants(object);
        for s in &subjects {
            if let Some(outgoing) = self.out_index.get(s) {
                for (rel, obj) in outgoing {
                    if predicates.contains(rel) && objects.contains(obj) {
                        return true;
                    }
                }
            }
        }
        let symmetric = predicates
            .iter()
            .any(|p| SYMMETRIC_RELS.contains(&self.canon(p).as_str()));
        if symmetric {
            for o in &objects {
                if let Some(outgoing) = self.out_index.get(o) {
                    for (rel, obj) in outgoing {
                        if predicates.contains(rel) && subjects.contains(obj) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn entities_satisfying(&self, predicate: &str) -> Vec<String> {
        let mut out = self.merged_unary(predicate);
        for (entity, types) in &self.types {
            for ty in types {
                if self.name_variants(predicate).contains(ty) || self.isa(ty, predicate) {
                    out.insert(entity.clone());
                }
            }
        }
        for (entity, key) in self.values.keys() {
            if self.name_variants(predicate).contains(key) {
                out.insert(entity.clone());
            }
        }
        for (subject, outgoing) in &self.out_index {
            if outgoing
                .iter()
                .any(|(rel, _)| self.name_variants(predicate).contains(rel))
            {
                out.insert(subject.clone());
            }
        }
        out.into_iter().take(DOMAIN_CAP).collect()
    }

    fn isa(&self, entity: &str, class: &str) -> bool {
        let class_variants = self.name_variants(class);
        if class_variants
            .iter()
            .any(|c| c == "实体" || c == "事物")
        {
            return self.entities.contains(entity) || self.types.contains_key(entity);
        }
        let direct: BTreeSet<String> = self
            .name_variants(entity)
            .iter()
            .filter_map(|e| self.types.get(e).cloned())
            .flatten()
            .collect();
        for ty in &direct {
            if class_variants.contains(ty) || self.super_closure(ty, class, 0) {
                return true;
            }
        }
        false
    }

    fn super_closure(&self, class: &str, target: &str, depth: usize) -> bool {
        if depth > 8 {
            return false;
        }
        let Some(supers) = self.supers.get(class) else {
            return false;
        };
        for sup in supers {
            if self.name_variants(target).contains(sup)
                || self.super_closure(sup, target, depth + 1)
            {
                return true;
            }
        }
        false
    }

    pub fn class_like(&self, name: &str) -> bool {
        self.class_set.contains(name)
            || self.name_variants(name).iter().any(|n| {
                self.class_set.contains(n)
                    || self.types.values().any(|types| types.contains(n))
                    || n == "实体"
                    || n == "事物"
            })
    }

    fn name_of(&self, value: &Eval) -> Option<String> {
        match value {
            Eval::Entity(name) => Some(name.clone()),
            Eval::Number(number) => Some(format_number(*number)),
            Eval::Entities(names) if names.len() == 1 => Some(names[0].clone()),
            _ => None,
        }
    }

    /// 在模型上求值（空环境）
    pub fn eval(&self, term: &Term) -> Eval {
        let env = HashMap::new();
        self.eval_with(term, &env)
    }

    pub fn eval_with(&self, term: &Term, env: &HashMap<String, Term>) -> Eval {
        let reduced = Term::reduce(term.clone());
        self.eval_reduced(&reduced, env)
    }

    fn eval_reduced(&self, term: &Term, env: &HashMap<String, Term>) -> Eval {
        match term {
            Term::Const(name) => Eval::Entity(name.clone()),
            Term::Num(text) => match parse_number_token(text) {
                Some(number) => Eval::Number(number),
                None => Eval::Failure(format!("无法解析数字「{}」", text)),
            },
            Term::Var(name) => match env.get(name) {
                Some(value) => self.eval_with(value, env),
                None => Eval::Failure(format!("自由变量 {}", name)),
            },
            Term::Pred(predicate) => Eval::Entities(self.entities_satisfying(predicate)),
            Term::Attr(predicate, arg) => {
                let predicate_name = match &**predicate {
                    Term::Pred(name) | Term::Const(name) => name.clone(),
                    other => other.render(),
                };
                let subject = match self.name_of(&self.eval_reduced(arg, env)) {
                    Some(name) => name,
                    None => return Eval::Failure("属性主体无法求值".to_string()),
                };
                let values = self.values_of(&subject, &predicate_name);
                match values.len() {
                    0 => Eval::Failure(format!("没有「{}」的「{}」记录", subject, predicate_name)),
                    1 => Eval::Entity(values[0].clone()),
                    _ => Eval::Entities(values),
                }
            }
            Term::App(function, argument) => match &**function {
                Term::Pred(predicate) => {
                    let value = self.eval_reduced(argument, env);
                    match self.name_of(&value) {
                        Some(entity) => Eval::Truth(self.holds_unary(&entity, predicate)),
                        None => Eval::Failure(format!("{}(?) 主体无法求值", predicate)),
                    }
                }
                Term::App(inner, object_term) => match &**inner {
                    Term::Pred(predicate) => {
                        let object = self.name_of(&self.eval_reduced(object_term, env));
                        let subject = self.name_of(&self.eval_reduced(argument, env));
                        match (subject, object) {
                            (Some(subject), Some(object)) => {
                                Eval::Truth(self.holds_binary(predicate, &subject, &object))
                            }
                            _ => Eval::Failure(format!("{}(?, ?) 参数无法求值", predicate)),
                        }
                    }
                    _ => Eval::Failure("不支持的高阶应用".to_string()),
                },
                Term::Lam(var, body) => {
                    let applied = Term::apply(
                        Term::Lam(var.clone(), body.clone()),
                        argument.as_ref().clone(),
                    );
                    self.eval_reduced(&applied, env)
                }
                _ => Eval::Failure("函数位置求值失败".to_string()),
            },
            Term::Not(inner) => match self.eval_reduced(inner, env) {
                Eval::Truth(value) => Eval::Truth(!value),
                _ => Eval::Failure("否定作用于非真值".to_string()),
            },
            Term::And(terms) => {
                for term in terms {
                    match self.eval_reduced(term, env) {
                        Eval::Truth(false) => return Eval::Truth(false),
                        Eval::Truth(true) => {}
                        _ => return Eval::Failure("合取项无法判定".to_string()),
                    }
                }
                Eval::Truth(true)
            }
            Term::Or(terms) => {
                for term in terms {
                    if let Eval::Truth(true) = self.eval_reduced(term, env) {
                        return Eval::Truth(true);
                    }
                }
                Eval::Truth(false)
            }
            Term::Implies(a, b) => match (self.eval_reduced(a, env), self.eval_reduced(b, env)) {
                (Eval::Truth(false), _) => Eval::Truth(true),
                (Eval::Truth(true), Eval::Truth(value)) => Eval::Truth(value),
                _ => Eval::Failure("蕴含前件/后件无法判定".to_string()),
            },
            Term::Isa(a, b) => {
                let left = self.eval_reduced(a, env);
                let right = self.eval_reduced(b, env);
                self.eval_isa(left, right)
            }
            Term::Eq(a, b) => {
                let left = self.eval_reduced(a, env);
                let right = self.eval_reduced(b, env);
                self.eval_eq(left, right)
            }
            Term::Exists { var, body } => {
                for entity in self.entities.iter().take(DOMAIN_CAP) {
                    let mut env2 = env.clone();
                    env2.insert(var.clone(), Term::Const(entity.clone()));
                    if let Eval::Truth(true) = self.eval_reduced(body, &env2) {
                        return Eval::Truth(true);
                    }
                }
                Eval::Truth(false)
            }
            Term::Forall { var, body } => {
                for entity in self.entities.iter().take(DOMAIN_CAP) {
                    let mut env2 = env.clone();
                    env2.insert(var.clone(), Term::Const(entity.clone()));
                    match self.eval_reduced(body, &env2) {
                        Eval::Truth(true) => {}
                        Eval::Truth(false) => return Eval::Truth(false),
                        _ => return Eval::Failure("全称量化的个体无法判定".to_string()),
                    }
                }
                Eval::Truth(true)
            }
            Term::Lam(_, _) => Eval::Failure("未应用的函数（lambda 抽象）".to_string()),
        }
    }

    fn eval_isa(&self, left: Eval, right: Eval) -> Eval {
        let right_names: Vec<String> = match &right {
            Eval::Entity(name) => vec![name.clone()],
            Eval::Entities(names) => names.clone(),
            Eval::Number(number) => vec![format_number(*number)],
            Eval::Truth(_) => Vec::new(),
            Eval::Failure(reason) => return Eval::Failure(reason.clone()),
        };
        if right_names.is_empty() {
            return Eval::Failure("归类目标无法求值".to_string());
        }
        let subjects: Vec<String> = match &left {
            Eval::Entity(name) => vec![name.clone()],
            Eval::Entities(names) => names.clone(),
            Eval::Number(number) => vec![format_number(*number)],
            Eval::Truth(_) => Vec::new(),
            Eval::Failure(reason) => return Eval::Failure(reason.clone()),
        };
        if subjects.is_empty() {
            return Eval::Failure("归类主体无法求值".to_string());
        }
        if right_names.iter().all(|name| self.class_like(name)) {
            let holds = subjects
                .iter()
                .any(|subject| right_names.iter().any(|class| self.isa(subject, class)));
            return Eval::Truth(holds);
        }
        let values_match = |subject: &str, target: &str| -> bool {
            if self.canon(subject) == self.canon(target) {
                return true;
            }
            match (parse_number_token(subject), parse_number_token(target)) {
                (Some(a), Some(b)) => (a - b).abs() < 1e-9,
                _ => false,
            }
        };
        let holds = subjects
            .iter()
            .any(|subject| right_names.iter().any(|target| values_match(subject, target)));
        Eval::Truth(holds)
    }

    fn eval_eq(&self, left: Eval, right: Eval) -> Eval {
        let left_names: Vec<String> = match &left {
            Eval::Entity(name) => vec![name.clone()],
            Eval::Entities(names) => names.clone(),
            Eval::Number(number) => vec![format_number(*number)],
            _ => Vec::new(),
        };
        let right_names: Vec<String> = match &right {
            Eval::Entity(name) => vec![name.clone()],
            Eval::Entities(names) => names.clone(),
            Eval::Number(number) => vec![format_number(*number)],
            _ => Vec::new(),
        };
        let holds = left_names
            .iter()
            .any(|a| right_names.iter().any(|b| self.canon(a) == self.canon(b)));
        Eval::Truth(holds)
    }

    /// 为逻辑项收集可读的自然语言证据（用于回答引用）
    pub fn evidence_for(&self, term: &Term) -> Vec<String> {
        match term {
            Term::App(f, a) => match &**f {
                Term::Pred(predicate) => match &**a {
                    Term::Const(entity) => {
                        let values = self.values_of(entity, predicate);
                        if values.is_empty() {
                            vec![format!("图谱中没有「{}」的「{}」记录", entity, predicate)]
                        } else {
                            values
                                .into_iter()
                                .map(|value| format!("{} 的 {} 是 {}", entity, predicate, value))
                                .collect()
                        }
                    }
                    other => vec![format!("{}({})", predicate, other.render())],
                },
                Term::App(inner, object_term) => match &**inner {
                    Term::Pred(predicate) => match (&**object_term, &**a) {
                        (Term::Const(object), Term::Const(subject)) => {
                            if self.holds_binary(predicate, subject, object) {
                                vec![format!("{} —{}→ {}", subject, predicate, object)]
                            } else {
                                vec![format!(
                                    "图谱中没有「{} {} {}」的记录",
                                    subject, predicate, object
                                )]
                            }
                        }
                        _ => vec![term.render()],
                    },
                    _ => vec![term.render()],
                },
                _ => vec![term.render()],
            },
            Term::Not(inner) => {
                let mut out = self.evidence_for(inner);
                out.push("（否定判断）".to_string());
                out
            }
            Term::And(terms) | Term::Or(terms) => {
                terms.iter().flat_map(|t| self.evidence_for(t)).collect()
            }
            Term::Isa(a, b) => match (&**a, &**b) {
                (Term::Const(subject), Term::Const(target)) => {
                    if self.isa(subject, target) {
                        let types: Vec<String> = self
                            .name_variants(subject)
                            .iter()
                            .filter_map(|e| self.types.get(e).cloned())
                            .flatten()
                            .collect();
                        vec![format!("{} 的类型是 {}", subject, types.join("、"))]
                    } else {
                        vec![format!("{} 不是 {}", subject, target)]
                    }
                }
                _ => vec![term.render()],
            },
            Term::Exists { var, body } | Term::Forall { var, body } => {
                let quantifier = if matches!(term, Term::Exists { .. }) {
                    "存在量化"
                } else {
                    "全称量化"
                };
                vec![format!(
                    "{}：{}（个体域 {} 个实体）：{}",
                    quantifier,
                    var,
                    self.entities.len(),
                    body.render()
                )]
            }
            other => vec![other.render()],
        }
    }

    /// 从实体出发取属性/关系值（供外部检索接地使用）
    pub fn attribute_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.values_of(entity, predicate)
    }
}

fn format_number(number: f64) -> String {
    if (number - number.round()).abs() < 1e-9 {
        format!("{}", number.round() as i64)
    } else {
        format!("{}", number)
    }
}

/// 解析数字 token（阿拉伯数字或中文数词）
pub fn parse_number_token(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Ok(value) = text.parse::<f64>() {
        return Some(value);
    }
    crate::chinese_reasoning_nlu::parse_chinese_number(text).map(|value| value as f64)
}

// ---------------------------------------------------------------------------
// 逻辑形式与查询
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// 值查询：ENTITY 的 PREDICATE 是多少
    Value { entity: String, predicate: String },
    /// 实体查询：谁是 PREDICATE
    Entity { predicate: String },
    /// 真值查询：判断 BODY 是否成立
    Truth { body: Term },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LfKind {
    Statement(Term),
    Query(Query),
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalForm {
    pub source: String,
    pub tokens: Vec<String>,
    pub kind: LfKind,
    pub trace: Vec<String>,
}

impl LogicalForm {
    fn unknown(source: String, tokens: Vec<String>, trace: Vec<String>) -> Self {
        Self {
            source,
            tokens,
            kind: LfKind::Unknown,
            trace,
        }
    }

    pub fn is_known(&self) -> bool {
        !matches!(self.kind, LfKind::Unknown)
    }

    /// 逻辑形式的一行渲染
    pub fn render(&self) -> String {
        match &self.kind {
            LfKind::Statement(term) => format!("{} : t", term.render()),
            LfKind::Query(Query::Value { entity, predicate }) => {
                format!("查询 x：{} 的 {} = x", entity, predicate)
            }
            LfKind::Query(Query::Entity { predicate }) => format!("查询 x：{}(x)", predicate),
            LfKind::Query(Query::Truth { body }) => format!("验证：{}", body.render()),
            LfKind::Unknown => "（未能组合出逻辑形式）".to_string(),
        }
    }

    /// 提取焦点实体与谓词（供对话记忆）
    pub fn focus(&self) -> (Option<String>, Option<String>) {
        match &self.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                (Some(entity.clone()), Some(predicate.clone()))
            }
            LfKind::Query(Query::Entity { predicate }) => (None, Some(predicate.clone())),
            LfKind::Query(Query::Truth { body }) => first_atom(body)
                .map(|(e, p)| (Some(e), Some(p)))
                .unwrap_or((None, None)),
            LfKind::Statement(term) => first_atom(term)
                .map(|(e, p)| (Some(e), Some(p)))
                .unwrap_or((None, None)),
            LfKind::Unknown => (None, None),
        }
    }
}

/// 从逻辑项中找最左侧的原子谓词结构
fn first_atom(term: &Term) -> Option<(String, String)> {
    match term {
        Term::App(f, a) => match (&**f, &**a) {
            (Term::Pred(p), Term::Const(e)) => Some((e.clone(), p.clone())),
            (Term::App(inner, first), Term::Const(e)) => match &**inner {
                Term::Pred(p) => Some((e.clone(), p.clone())),
                _ => first_atom(f).or_else(|| first_atom(first)),
            },
            _ => first_atom(f).or_else(|| first_atom(a)),
        },
        Term::Not(t) => first_atom(t),
        Term::And(ts) | Term::Or(ts) => ts.iter().find_map(first_atom),
        Term::Implies(a, b) => first_atom(a).or_else(|| first_atom(b)),
        Term::Isa(a, b) => match (&**a, &**b) {
            (Term::Const(e), Term::Const(c)) => Some((e.clone(), c.clone())),
            _ => first_atom(a).or_else(|| first_atom(b)),
        },
        Term::Exists { body, .. } | Term::Forall { body, .. } => first_atom(body),
        Term::Attr(p, x) => first_atom(p).or_else(|| first_atom(x)),
        _ => None,
    }
}

/// 求解结果
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOutcome {
    Values {
        entity: String,
        predicate: String,
        values: Vec<String>,
        evidence: Vec<String>,
    },
    Entities {
        predicate: String,
        entities: Vec<String>,
        evidence: Vec<String>,
    },
    Truth {
        holds: bool,
        statement: String,
        evidence: Vec<String>,
    },
    Unsupported(String),
}

// ---------------------------------------------------------------------------
// 组合语义解析器
// ---------------------------------------------------------------------------

const QUESTION_WORDS: [&str; 12] = [
    "谁", "什么", "多少", "几", "哪", "哪个", "哪些", "哪里", "何人", "怎么", "为什么", "如何",
];
const ASSERTION_MARKERS: [&str; 8] = ["是", "为", "不", "没", "所有", "每个", "并且", "或者"];
const DET_ALL: [&str; 6] = ["所有", "每个", "每一个", "任何", "全部", "一切"];
const DET_SOME: [&str; 5] = ["有些", "有的", "某些", "某个", "存在"];
const CONJ_AND: [&str; 4] = ["并且", "而且", "且", "同时"];
const CONJ_OR: [&str; 3] = ["或者", "或", "要么"];
const COPULA_CONNECTORS: [&str; 5] = ["是", "为", "属于", "不是", "并非"];
const PARTICLES: [&str; 5] = ["都", "也", "而", "乃", "则"];
const QUESTION_SUFFIX: [&str; 2] = ["吗", "呢"];
const MAX_ITEMS_PER_CELL: usize = 8;

#[derive(Clone)]
struct Item {
    term: Term,
    ty: Type,
    label: String,
}

fn push_item(cells: &mut [Vec<Vec<Item>>], i: usize, j: usize, item: Item, max_items: usize) {
    let cell = &mut cells[i][j];
    let key = format!("{}:{}", item.ty.render(), item.label);
    if cell
        .iter()
        .any(|existing| format!("{}:{}", existing.ty.render(), existing.label) == key)
    {
        return;
    }
    if cell.len() < max_items {
        cell.push(item);
    }
}

pub struct FormalSemantics {
    lexicon: Lexicon,
    max_tokens: usize,
    max_items: usize,
}

impl Default for FormalSemantics {
    fn default() -> Self {
        Self::new()
    }
}

impl FormalSemantics {
    pub fn new() -> Self {
        Self {
            lexicon: Lexicon::default_chinese(),
            max_tokens: 24,
            max_items: MAX_ITEMS_PER_CELL,
        }
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    pub fn register_entity(&mut self, name: &str) {
        self.lexicon.register_entity(name);
    }

    pub fn register_class(&mut self, name: &str) {
        self.lexicon.register_class(name);
    }

    pub fn register_unary_predicate(&mut self, name: &str) {
        self.lexicon.register_unary_predicate(name);
    }

    pub fn register_binary_predicate(&mut self, name: &str) {
        self.lexicon.register_binary_predicate(name);
    }

    pub fn set_predicate_surfaces(&mut self, surfaces: &HashMap<String, String>) {
        self.lexicon.set_predicate_surfaces(surfaces);
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in shared_jieba().cut(text, false) {
            let word = word.trim();
            if word.is_empty() || is_punctuation(word) || PARTICLES.contains(&word) {
                continue;
            }
            tokens.push(word.to_string());
        }
        tokens
    }

    /// 解析中文句子为逻辑形式（问句优先走模式切分，陈述句走类型驱动图表分析）
    pub fn parse(&self, text: &str) -> LogicalForm {
        let source = text.trim().to_string();
        let raw_tokens = self.tokenize(&source);
        // 合并被切开的「不 + 是」→「不是」，保证系词否定进入同一条系词规则
        let mut tokens: Vec<String> = Vec::with_capacity(raw_tokens.len());
        let mut index = 0;
        while index < raw_tokens.len() {
            if raw_tokens[index] == "不"
                && index + 1 < raw_tokens.len()
                && raw_tokens[index + 1] == "是"
            {
                tokens.push("不是".to_string());
                index += 2;
            } else {
                tokens.push(raw_tokens[index].clone());
                index += 1;
            }
        }
        let mut trace = vec![format!("分词：{}", tokens.join(" / "))];
        if tokens.is_empty() {
            trace.push("空输入".to_string());
            return LogicalForm::unknown(source, tokens, trace);
        }
        if let Some(lf) = self.parse_conditional(&tokens, &source, &mut trace) {
            return lf;
        }
        let is_question = tokens
            .iter()
            .any(|token| QUESTION_WORDS.contains(&token.as_str()))
            || tokens
                .last()
                .map(|token| QUESTION_SUFFIX.contains(&token.as_str()))
                .unwrap_or(false)
            || source.ends_with('?')
            || source.ends_with('？');
        if is_question {
            if let Some(lf) = self.parse_question(&tokens, &source, &mut trace) {
                return lf;
            }
        }
        if tokens.len() <= self.max_tokens {
            let mut chart_trace = Vec::new();
            if let Some(item) = self.chart_parse(&tokens, &mut chart_trace) {
                if item.ty == Type::Truth {
                    trace.extend(chart_trace);
                    if !self.has_assertion_marker(&tokens) {
                        if let Some((entity, predicate)) = possessive_np(&item.term) {
                            trace.push(format!(
                                "无断言标记，解释为属性查询：{} 的 {}",
                                entity, predicate
                            ));
                            return LogicalForm {
                                source,
                                tokens,
                                kind: LfKind::Query(Query::Value { entity, predicate }),
                                trace,
                            };
                        }
                    }
                    return LogicalForm {
                        source,
                        tokens,
                        kind: LfKind::Statement(item.term),
                        trace,
                    };
                }
            }
        }
        trace.push("未能在当前片段内组合出类型为 t 的逻辑形式".to_string());
        LogicalForm::unknown(source, tokens, trace)
    }

    fn has_assertion_marker(&self, tokens: &[String]) -> bool {
        tokens
            .iter()
            .any(|token| ASSERTION_MARKERS.contains(&token.as_str()))
    }

    /// 「如果 P 那么/就/则 Q」→ P → Q
    fn parse_conditional(
        &self,
        tokens: &[String],
        source: &str,
        trace: &mut Vec<String>,
    ) -> Option<LogicalForm> {
        let start = tokens
            .iter()
            .position(|token| token == "如果" || token == "假如" || token == "若")?;
        let connect = tokens
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, token)| ["那么", "则", "就"].contains(&token.as_str()))
            .map(|(index, _)| index)?;
        let left_tokens = &tokens[start + 1..connect];
        let right_tokens: Vec<String> = tokens[connect + 1..]
            .iter()
            .filter(|token| !QUESTION_SUFFIX.contains(&token.as_str()))
            .cloned()
            .collect();
        let mut left_trace = Vec::new();
        let left = self.chart_parse(left_tokens, &mut left_trace)?;
        let mut right_trace = Vec::new();
        let right = self.chart_parse(&right_tokens, &mut right_trace)?;
        let left_term = saturate(left)?;
        let right_term = saturate(right)?;
        trace.push(format!("条件句：{} → {}", left_term.render(), right_term.render()));
        let term = Term::Implies(Box::new(left_term), Box::new(right_term));
        Some(LogicalForm {
            source: source.to_string(),
            tokens: tokens.to_vec(),
            kind: LfKind::Statement(term),
            trace: trace.clone(),
        })
    }

    fn parse_question(
        &self,
        tokens: &[String],
        source: &str,
        trace: &mut Vec<String>,
    ) -> Option<LogicalForm> {
        // 1) 「…吗/呢」：去掉句尾疑问词后按陈述句解析，再作为真值查询
        if let Some(last) = tokens.last() {
            if QUESTION_SUFFIX.contains(&last.as_str()) && tokens.len() > 1 {
                let body_tokens = &tokens[..tokens.len() - 1];
                let mut chart_trace = Vec::new();
                if let Some(item) = self.chart_parse(body_tokens, &mut chart_trace) {
                    if item.ty == Type::Truth {
                        trace.extend(chart_trace);
                        trace.push("句尾疑问词：转化为真值查询".to_string());
                        return Some(LogicalForm {
                            source: source.to_string(),
                            tokens: tokens.to_vec(),
                            kind: LfKind::Query(Query::Truth { body: item.term }),
                            trace: trace.clone(),
                        });
                    }
                }
            }
        }
        // 2) 「是不是 / 是否」：E 是不是 P → Isa(E, P) 真值查询
        if let Some(index) = tokens
            .iter()
            .position(|token| token == "是不是" || token == "是否")
        {
            let entity = self.entity_from_tokens(&tokens[..index]);
            let predicate = self.predicate_from_tokens(&tokens[index + 1..]);
            if let (Some(entity), Some(predicate)) = (entity, predicate) {
                trace.push(format!("是不是句式：Isa({}, {})", entity, predicate));
                return Some(LogicalForm {
                    source: source.to_string(),
                    tokens: tokens.to_vec(),
                    kind: LfKind::Query(Query::Truth {
                        body: Term::Isa(
                            Box::new(Term::Const(entity)),
                            Box::new(Term::Const(predicate)),
                        ),
                    }),
                    trace: trace.clone(),
                });
            }
        }
        // 3) 「谁 / 哪些人 / 何人 …」：实体查询
        if let Some(first) = tokens.first() {
            if ["谁", "哪些人", "哪些", "何人", "什么人", "哪个"].contains(&first.as_str()) {
                let mut rest = &tokens[1..];
                while rest
                    .first()
                    .map(|token| ["是", "为", "都", "的", "属于"].contains(&token.as_str()))
                    .unwrap_or(false)
                {
                    rest = &rest[1..];
                }
                if let Some(predicate) = self.predicate_from_tokens(rest) {
                    trace.push(format!("谁是句式：查询满足「{}」的实体", predicate));
                    return Some(LogicalForm {
                        source: source.to_string(),
                        tokens: tokens.to_vec(),
                        kind: LfKind::Query(Query::Entity { predicate }),
                        trace: trace.clone(),
                    });
                }
            }
        }
        // 4) 「X 的 P …」：值查询
        if let Some(index) = tokens.iter().position(|token| token == "的") {
            if index > 0 && index + 1 < tokens.len() {
                if let Some(entity) = self.entity_from_tokens(&tokens[..index]) {
                    if let Some(predicate) = self.predicate_from_tokens(&tokens[index + 1..]) {
                        if !predicate.is_empty() {
                            trace.push(format!(
                                "的-属格句式：查询「{}」的「{}」值",
                                entity, predicate
                            ));
                            return Some(LogicalForm {
                                source: source.to_string(),
                                tokens: tokens.to_vec(),
                                kind: LfKind::Query(Query::Value { entity, predicate }),
                                trace: trace.clone(),
                            });
                        }
                    }
                }
            }
        }
        // 5) 「E P …」（无「的」）：按最长实体前缀切分
        if let Some((entity, predicate)) = self.split_entity_predicate(tokens) {
            trace.push(format!(
                "实体-谓词切分：查询「{}」的「{}」值",
                entity, predicate
            ));
            return Some(LogicalForm {
                source: source.to_string(),
                tokens: tokens.to_vec(),
                kind: LfKind::Query(Query::Value { entity, predicate }),
                trace: trace.clone(),
            });
        }
        None
    }

    fn entity_from_tokens(&self, tokens: &[String]) -> Option<String> {
        let filtered: Vec<String> = tokens
            .iter()
            .filter(|token| {
                !["那", "那么", "这个", "那个", "请问", "帮我", "告诉"].contains(&token.as_str())
            })
            .cloned()
            .collect();
        if filtered.is_empty() {
            return None;
        }
        let joined: String = filtered.concat();
        if let Some(entity) = self.lexicon.find_entity(&joined) {
            return Some(entity);
        }
        if !joined.is_empty() {
            Some(joined)
        } else {
            None
        }
    }

    fn predicate_from_tokens(&self, tokens: &[String]) -> Option<String> {
        let filtered: Vec<String> = tokens
            .iter()
            .filter(|token| {
                !QUESTION_WORDS.contains(&token.as_str())
                    && !["的", "是", "为", "了", "时候", "哪里", "号码"].contains(&token.as_str())
            })
            .cloned()
            .collect();
        if filtered.is_empty() {
            return None;
        }
        let joined: String = filtered.concat();
        self.lexicon.resolve_predicate_surface(&joined)
    }

    fn split_entity_predicate(&self, tokens: &[String]) -> Option<(String, String)> {
        let mut best: Option<(String, String)> = None;
        for length in 1..tokens.len() {
            let prefix: String = tokens[..length].concat();
            if !self.lexicon.is_entity_name(&prefix) && self.lexicon.find_entity(&prefix).is_none() {
                continue;
            }
            let rest: Vec<String> = tokens[length..]
                .iter()
                .filter(|token| !["是", "为", "的"].contains(&token.as_str()))
                .cloned()
                .collect();
            if rest.is_empty() {
                continue;
            }
            let Some(predicate) = self.predicate_from_tokens(&rest) else {
                continue;
            };
            if predicate.is_empty() {
                continue;
            }
            best = Some((prefix, predicate));
        }
        best
    }

    /// 类型驱动的图表组合分析（CKY 风格：函数应用 + 量词 + 合取 + 系词规则）
    fn chart_parse(&self, tokens: &[String], trace: &mut Vec<String>) -> Option<Item> {
        let n = tokens.len();
        if n == 0 || n > self.max_tokens {
            return None;
        }
        let entity_ty = Type::Entity;
        let truth_ty = Type::Truth;
        let unary_ty = Type::fun(entity_ty.clone(), truth_ty.clone());
        let binary_ty = Type::fun(entity_ty.clone(), unary_ty.clone());
        let quant_ty = Type::fun(unary_ty.clone(), truth_ty.clone());

        let mut cells: Vec<Vec<Vec<Item>>> = vec![vec![Vec::new(); n + 1]; n + 1];

        // 词项装载
        for i in 0..n {
            let token = tokens[i].clone();
            for entry in self.lexicon.lookup(&token) {
                let label = entry.term.render();
                push_item(
                    &mut cells,
                    i,
                    i + 1,
                    Item {
                        term: entry.term,
                        ty: entry.ty,
                        label,
                    },
                    self.max_items,
                );
            }
            if parse_number_token(&token).is_some() {
                push_item(
                    &mut cells,
                    i,
                    i + 1,
                    Item {
                        term: Term::Num(token.clone()),
                        ty: entity_ty.clone(),
                        label: token.clone(),
                    },
                    self.max_items,
                );
            }
            // 否定前缀「不 + 谓词」
            if token == "不" && i + 1 < n {
                for entry in self.lexicon.lookup(&tokens[i + 1]) {
                    if entry.ty == unary_ty {
                        let neg = Term::Lam(
                            "x".into(),
                            Box::new(Term::Not(Box::new(Term::apply(
                                entry.term.clone(),
                                Term::Var("x".into()),
                            )))),
                        );
                        push_item(
                            &mut cells,
                            i,
                            i + 2,
                            Item {
                                label: format!("不{}", entry.surface),
                                term: neg,
                                ty: unary_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                    if entry.ty == binary_ty {
                        let neg = Term::Lam(
                            "x".into(),
                            Box::new(Term::Lam(
                                "y".into(),
                                Box::new(Term::Not(Box::new(Term::apply(
                                    Term::apply(entry.term.clone(), Term::Var("y".into())),
                                    Term::Var("x".into()),
                                )))),
                            )),
                        );
                        push_item(
                            &mut cells,
                            i,
                            i + 2,
                            Item {
                                label: format!("不{}", entry.surface),
                                term: neg,
                                ty: binary_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                }
            }
            // 多词实体/谓词（跨 2..=4 个 token 的连接串）
            for length in 2..=4.min(n - i) {
                let joined: String = tokens[i..i + length].concat();
                if joined == token {
                    continue;
                }
                for entry in self.lexicon.lookup(&joined) {
                    let label = entry.term.render();
                    push_item(
                        &mut cells,
                        i,
                        i + length,
                        Item {
                            term: entry.term,
                            ty: entry.ty,
                            label,
                        },
                        self.max_items,
                    );
                }
            }
        }

        // 组合
        for length in 2..=n {
            for i in 0..=n - length {
                let j = i + length;
                // 量词规则：DET + N
                if DET_ALL.contains(&tokens[i].as_str()) || DET_SOME.contains(&tokens[i].as_str()) {
                    let universal = DET_ALL.contains(&tokens[i].as_str());
                    for head in cells[i + 1][j].clone() {
                        if head.ty != unary_ty {
                            continue;
                        }
                        let body = Term::Implies(
                            Box::new(Term::apply(head.term.clone(), Term::Var("x".into()))),
                            Box::new(Term::apply(Term::Var("P".into()), Term::Var("x".into()))),
                        );
                        let key = if universal {
                            Term::Forall {
                                var: "x".into(),
                                body: Box::new(body),
                            }
                        } else {
                            Term::Exists {
                                var: "x".into(),
                                body: Box::new(body),
                            }
                        };
                        let quant = Term::Lam("P".into(), Box::new(key));
                        push_item(
                            &mut cells,
                            i,
                            j,
                            Item {
                                label: format!("{}{}", tokens[i], head.label),
                                term: quant,
                                ty: quant_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                }
                // 连接词规则（合取/析取 + 话题-述题「是」）
                for k in i + 1..j {
                    if k + 1 >= j {
                        continue;
                    }
                    let connector = tokens[k].as_str();
                    if CONJ_AND.contains(&connector) || CONJ_OR.contains(&connector) {
                        let conjunctive = CONJ_AND.contains(&connector);
                        for a in cells[i][k].clone() {
                            if a.ty != truth_ty {
                                continue;
                            }
                            for b in cells[k + 1][j].clone() {
                                if b.ty != truth_ty {
                                    continue;
                                }
                                let term = if conjunctive {
                                    Term::And(vec![a.term.clone(), b.term.clone()])
                                } else {
                                    Term::Or(vec![a.term.clone(), b.term.clone()])
                                };
                                push_item(
                                    &mut cells,
                                    i,
                                    j,
                                    Item {
                                        label: format!("({} {} {})", a.label, connector, b.label),
                                        term,
                                        ty: truth_ty.clone(),
                                    },
                                    self.max_items,
                                );
                            }
                        }
                    }
                    // 系词规则：⟨左项，系词，右项⟩ 直接按「主语-表语」顺序构造 Isa，
                    // 避免把系词当成普通二元谓词时的参数顺序歧义（话题-述题也由此覆盖）
                    if COPULA_CONNECTORS.contains(&connector) {
                        let negate = connector == "不是" || connector == "并非";
                        for a in cells[i][k].clone() {
                            for b in cells[k + 1][j].clone() {
                                let body = if b.ty == entity_ty {
                                    if a.ty == entity_ty {
                                        Some(Term::apply(
                                            Term::Lam(
                                                "z".into(),
                                                Box::new(Term::Isa(
                                                    Box::new(Term::Var("z".into())),
                                                    Box::new(b.term.clone()),
                                                )),
                                            ),
                                            a.term.clone(),
                                        ))
                                    } else if a.ty == quant_ty {
                                        Some(Term::apply(
                                            a.term.clone(),
                                            Term::Lam(
                                                "z".into(),
                                                Box::new(Term::Isa(
                                                    Box::new(Term::Var("z".into())),
                                                    Box::new(b.term.clone()),
                                                )),
                                            ),
                                        ))
                                    } else if a.ty == unary_ty {
                                        Some(Term::apply(a.term.clone(), b.term.clone()))
                                    } else {
                                        None
                                    }
                                } else if b.ty == unary_ty && a.ty == entity_ty {
                                    Some(Term::apply(b.term.clone(), a.term.clone()))
                                } else if b.ty == unary_ty && a.ty == quant_ty {
                                    Some(Term::apply(a.term.clone(), b.term.clone()))
                                } else {
                                    None
                                };
                                let Some(body) = body else { continue };
                                let term = if negate {
                                    Term::Not(Box::new(body))
                                } else {
                                    body
                                };
                                push_item(
                                    &mut cells,
                                    i,
                                    j,
                                    Item {
                                        label: format!("{} {} {}", a.label, connector, b.label),
                                        term,
                                        ty: truth_ty.clone(),
                                    },
                                    self.max_items,
                                );
                            }
                        }
                    }
                }
                // 二元组合：函数应用 + 客体广义量词
                for k in i + 1..j {
                    for a in cells[i][k].clone() {
                        for b in cells[k][j].clone() {
                            for item in combine_items(&a, &b) {
                                push_item(&mut cells, i, j, item, self.max_items);
                            }
                        }
                    }
                }
            }
        }

        // 优先取类型为 t 的完整解析，其次取一元谓词（无主语句），最后取任何解析
        let top = cells[0][n].clone();
        let truth = top.iter().find(|item| item.ty == truth_ty).cloned();
        let unary = top.iter().find(|item| item.ty == unary_ty).cloned();
        let chosen = truth.or(unary).or_else(|| top.first().cloned());
        if let Some(item) = &chosen {
            trace.push(format!("图表组合：{} : {}", item.label, item.ty.render()));
        }
        chosen
    }

    /// 在模型上求解逻辑形式
    pub fn solve(&self, lf: &LogicalForm, model: &TinyModel) -> QueryOutcome {
        match &lf.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                let values = model.values_of(entity, predicate);
                let evidence = if values.is_empty() {
                    vec![format!(
                        "知识图谱中没有「{}」的「{}」记录",
                        entity, predicate
                    )]
                } else {
                    values
                        .iter()
                        .map(|value| format!("{} 的 {} 是 {}", entity, predicate, value))
                        .collect()
                };
                QueryOutcome::Values {
                    entity: entity.clone(),
                    predicate: predicate.clone(),
                    values,
                    evidence,
                }
            }
            LfKind::Query(Query::Entity { predicate }) => {
                let entities = model.entities_satisfying(predicate);
                let evidence = entities
                    .iter()
                    .map(|entity| format!("{} 是 {}", entity, predicate))
                    .collect();
                QueryOutcome::Entities {
                    predicate: predicate.clone(),
                    entities,
                    evidence,
                }
            }
            LfKind::Query(Query::Truth { body }) => {
                let value = model.eval(body);
                match value {
                    Eval::Truth(holds) => QueryOutcome::Truth {
                        holds,
                        statement: body.render(),
                        evidence: model.evidence_for(body),
                    },
                    Eval::Failure(reason) => QueryOutcome::Unsupported(reason),
                    _ => QueryOutcome::Unsupported("真值查询未得到真假值".to_string()),
                }
            }
            LfKind::Statement(term) => {
                let value = model.eval(term);
                match value {
                    Eval::Truth(holds) => QueryOutcome::Truth {
                        holds,
                        statement: term.render(),
                        evidence: model.evidence_for(term),
                    },
                    Eval::Failure(reason) => QueryOutcome::Unsupported(reason),
                    _ => QueryOutcome::Unsupported("陈述求值未得到真假值".to_string()),
                }
            }
            LfKind::Unknown => QueryOutcome::Unsupported("未能组合出逻辑形式".to_string()),
        }
    }

    /// 渲染求解结果（中文生成）
    pub fn render_outcome(&self, outcome: &QueryOutcome) -> String {
        match outcome {
            QueryOutcome::Values {
                entity,
                predicate,
                values,
                ..
            } => {
                let surface = self.lexicon.surface_of(predicate);
                if values.is_empty() {
                    format!("知识图谱中没有「{}」的「{}」记录。", entity, surface)
                } else {
                    format!(
                        "根据知识图谱：{} 的 {} 是 {}。",
                        entity,
                        surface,
                        values.join("、")
                    )
                }
            }
            QueryOutcome::Entities {
                predicate,
                entities,
                ..
            } => {
                let surface = self.lexicon.surface_of(predicate);
                if entities.is_empty() {
                    format!("知识图谱中没有满足「{}」的实体。", surface)
                } else {
                    format!(
                        "满足「{}」的实体有 {} 个：{}。",
                        surface,
                        entities.len(),
                        entities.join("、")
                    )
                }
            }
            QueryOutcome::Truth {
                holds, statement, ..
            } => {
                format!(
                    "判定：{} —— 在知识图谱中{}。",
                    statement,
                    if *holds { "成立" } else { "不成立" }
                )
            }
            QueryOutcome::Unsupported(reason) => format!("无法求解：{}。", reason),
        }
    }

    /// 逻辑形式分析报告（解析 + 轨迹 + 模型求值）
    pub fn describe(&self, lf: &LogicalForm) -> String {
        let mut out = String::new();
        out.push_str(&format!("逻辑形式：{}\n", lf.render()));
        out.push_str("分析轨迹：\n");
        for step in &lf.trace {
            out.push_str(&format!("  • {}\n", step));
        }
        out.trim_end().to_string()
    }

    /// 求值报告（供「逻辑形式」命令展示）
    pub fn denotation_report(&self, lf: &LogicalForm, model: &TinyModel) -> String {
        if !lf.is_known() {
            return "未能得出可求值的逻辑形式（可尝试更短的说法，或先导入相关事实）。".to_string();
        }
        let outcome = self.solve(lf, model);
        let mut out = self.render_outcome(&outcome);
        let evidence = match &outcome {
            QueryOutcome::Values { evidence, .. }
            | QueryOutcome::Entities { evidence, .. }
            | QueryOutcome::Truth { evidence, .. } => evidence.clone(),
            QueryOutcome::Unsupported(_) => Vec::new(),
        };
        if !evidence.is_empty() {
            out.push_str(&format!("\n证据：{}", evidence.join("；")));
        }
        out
    }
}

/// 把完整解析饱和为真值：一元谓词做存在闭包（无主语句），其余仅接受 t
fn saturate(item: Item) -> Option<Term> {
    match &item.ty {
        Type::Truth => Some(item.term),
        Type::Fun(dom, cod) if **dom == Type::Entity && **cod == Type::Truth => Some(Term::Exists {
            var: "x".into(),
            body: Box::new(Term::apply(item.term, Term::Var("x".into()))),
        }),
        _ => None,
    }
}

/// 无断言标记时把 P(E) 形态识别为属性值查询
fn possessive_np(term: &Term) -> Option<(String, String)> {
    if let Term::App(f, a) = term {
        if let (Term::Pred(predicate), Term::Const(entity)) = (&**f, &**a) {
            return Some((entity.clone(), predicate.clone()));
        }
    }
    None
}

/// 两项的类型驱动组合：前向应用、后向应用、客体广义量词
fn combine_items(a: &Item, b: &Item) -> Vec<Item> {
    let mut out = Vec::new();
    let entity_ty = Type::Entity;
    let truth_ty = Type::Truth;
    let unary_ty = Type::fun(entity_ty.clone(), truth_ty.clone());
    let binary_ty = Type::fun(entity_ty.clone(), unary_ty.clone());
    let quant_ty = Type::fun(unary_ty.clone(), truth_ty.clone());
    // 前向应用：a 是函数，b 是参数
    if let Type::Fun(dom, cod) = &a.ty {
        if **dom == b.ty {
            out.push(Item {
                term: Term::apply(a.term.clone(), b.term.clone()),
                label: format!("{}({})", a.label, b.label),
                ty: (**cod).clone(),
            });
        }
    }
    // 后向应用：b 是函数，a 是参数
    if let Type::Fun(dom, cod) = &b.ty {
        if **dom == a.ty {
            out.push(Item {
                term: Term::apply(b.term.clone(), a.term.clone()),
                label: format!("{}({})", b.label, a.label),
                ty: (**cod).clone(),
            });
        }
    }
    // 客体广义量词：TV + 量化 NP → VP
    if a.ty == binary_ty && b.ty == quant_ty {
        let term = Term::Lam(
            "x".into(),
            Box::new(Term::App(
                Box::new(b.term.clone()),
                Box::new(Term::Lam(
                    "y".into(),
                    Box::new(Term::App(
                        Box::new(Term::App(
                            Box::new(a.term.clone()),
                            Box::new(Term::Var("y".into())),
                        )),
                        Box::new(Term::Var("x".into())),
                    )),
                )),
            )),
        );
        out.push(Item {
            label: format!("{}⟨{}⟩", a.label, b.label),
            term,
            ty: unary_ty.clone(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// 语义层对话记忆：逻辑形式栈 + 话题实体/谓词，支撑指代消解与省略补全
// ---------------------------------------------------------------------------

pub struct SemanticMemory {
    capacity: usize,
    forms: Vec<LogicalForm>,
    focus_entities: Vec<String>,
    last_predicate: Option<String>,
}

impl SemanticMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.clamp(1, 32),
            forms: Vec::new(),
            focus_entities: Vec::new(),
            last_predicate: None,
        }
    }

    /// 记录一条逻辑形式（未知形式不进入记忆，避免污染指代）
    pub fn record(&mut self, lf: &LogicalForm) {
        if !lf.is_known() {
            return;
        }
        let (entity, predicate) = lf.focus();
        if let Some(predicate) = predicate {
            self.last_predicate = Some(predicate);
        }
        if let Some(entity) = entity {
            self.focus_entities.retain(|existing| existing != &entity);
            self.focus_entities.push(entity);
            while self.focus_entities.len() > 8 {
                self.focus_entities.remove(0);
            }
        }
        self.forms.push(lf.clone());
        while self.forms.len() > self.capacity {
            self.forms.remove(0);
        }
    }

    pub fn forms_len(&self) -> usize {
        self.forms.len()
    }

    pub fn focus_entities(&self) -> &[String] {
        &self.focus_entities
    }

    pub fn last_entity(&self) -> Option<&str> {
        self.focus_entities.last().map(String::as_str)
    }

    pub fn last_predicate(&self) -> Option<&str> {
        self.last_predicate.as_deref()
    }

    pub fn clear(&mut self) {
        self.forms.clear();
        self.focus_entities.clear();
        self.last_predicate = None;
    }

    /// 多轮上下文改写：指代消解（他/她/它 → 话题实体）+ 省略补全（「X呢」→「X 的上一谓词 是多少」）
    pub fn contextualize(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        // 省略补全
        let stripped = trimmed.trim_end_matches(['？', '?', '。', '.']).trim();
        let mut candidate = stripped.to_string();
        if let Some(stem) = stripped.strip_suffix('呢') {
            let stem = stem.trim();
            let stem = stem.strip_prefix("那").map(str::trim).unwrap_or(stem);
            let predicate = self.last_predicate()?;
            candidate = if stem.is_empty() {
                let entity = self.last_entity()?;
                format!("{}的{}是多少", entity, predicate)
            } else {
                format!("{}的{}是多少", stem, predicate)
            };
        }
        // 指代消解
        if let Some(entity) = self.last_entity() {
            for pronoun in ["他们", "他", "她", "它"] {
                if candidate.contains(pronoun) {
                    let replaced = candidate.replacen(pronoun, entity, 1);
                    if replaced != candidate {
                        return Some(replaced);
                    }
                }
            }
        }
        if candidate != trimmed {
            return Some(candidate);
        }
        None
    }

    pub fn summary(&self) -> String {
        let focus = if self.focus_entities.is_empty() {
            "（空）".to_string()
        } else {
            self.focus_entities.join(" → ")
        };
        let predicate = self.last_predicate.clone().unwrap_or_else(|| "（无）".to_string());
        format!(
            "语义记忆：最近逻辑形式 {} 条，话题实体 [{}]，上一谓词「{}」。",
            self.forms.len(),
            focus,
            predicate
        )
    }
}

fn is_punctuation(word: &str) -> bool {
    word.chars().all(|ch| {
        matches!(
            ch,
            '，' | '。'
                | '？'
                | '！'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '《'
                | '》'
                | '—'
                | '…'
                | ','
                | '.'
                | '?'
                | '!'
                | ';'
                | ':'
                | '"'
                | '\''
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | ' '
                | '\t'
                | '\n'
                | '\r'
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(semantics: &mut FormalSemantics, model: &mut TinyModel, facts: &[PrologFact]) {
        for fact in facts {
            model.observe_fact(fact);
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    semantics.register_binary_predicate(rel_type);
                    semantics.register_entity(from);
                    semantics.register_entity(to);
                    if SUPER_RELS.contains(&rel_type.as_str()) {
                        semantics.register_class(from);
                        semantics.register_class(to);
                    }
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    semantics.register_entity(name);
                    semantics.register_class(entity_type);
                    for key in properties.keys() {
                        semantics.register_unary_predicate(key);
                    }
                }
            }
        }
    }

    fn people_setup() -> (FormalSemantics, TinyModel) {
        let mut semantics = FormalSemantics::new();
        let mut model = TinyModel::new(HashMap::new());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "中国人".to_string(),
                properties: HashMap::from([
                    ("年龄".to_string(), "30".to_string()),
                    ("国籍".to_string(), "中国".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "美国人".to_string(),
                properties: HashMap::from([("年龄".to_string(), "28".to_string())]),
            },
            PrologFact::Relation {
                from: "学生".to_string(),
                rel_type: "子类".to_string(),
                to: "人".to_string(),
            },
            PrologFact::Entity {
                name: "小明".to_string(),
                entity_type: "学生".to_string(),
                properties: HashMap::new(),
            },
        ];
        feed(&mut semantics, &mut model, &facts);
        (semantics, model)
    }

    #[test]
    fn type_assertion_and_negation() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三是中国人");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("张三不是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("张三是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
    }

    #[test]
    fn possessive_statement_and_value_query() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三的年龄是30");
        assert!(
            matches!(
                semantics.solve(&lf, &model),
                QueryOutcome::Truth { holds: true, .. }
            ),
            "lf = {:?}",
            lf
        );
        let lf = semantics.parse("李四的年龄是30");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
        let lf = semantics.parse("张三的年龄是多少");
        match semantics.solve(&lf, &model) {
            QueryOutcome::Values { values, .. } => assert_eq!(values, vec!["30".to_string()]),
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn bare_possessive_is_value_query() {
        let (semantics, _model) = people_setup();
        let lf = semantics.parse("张三的年龄");
        match lf.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                assert_eq!(entity, "张三");
                assert_eq!(predicate, "年龄");
            }
            other => panic!("unexpected kind: {:?}", other),
        }
    }

    #[test]
    fn entity_query_who_is_class() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("谁是中国人");
        match semantics.solve(&lf, &model) {
            QueryOutcome::Entities { entities, .. } => {
                assert_eq!(entities, vec!["张三".to_string()]);
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn subclass_closure_propagates() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("小明是人");
        assert!(
            matches!(
                semantics.solve(&lf, &model),
                QueryOutcome::Truth { holds: true, .. }
            ),
            "lf = {:?}",
            lf
        );
    }

    #[test]
    fn truth_question_with_ma() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三是中国人吗");
        match lf.kind {
            LfKind::Query(Query::Truth { .. }) => {}
            other => panic!("unexpected kind: {:?}", other),
        }
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
    }

    #[test]
    fn universal_quantifier_evaluates() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("所有中国人都是中国人");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("所有中国人都是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
    }

    #[test]
    fn conditional_combines_two_clauses() {
        let mut semantics = FormalSemantics::new();
        let model = TinyModel::new(HashMap::new());
        semantics.register_unary_predicate("下雨");
        semantics.register_unary_predicate("地湿");
        let lf = semantics.parse("如果下雨那么地湿");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        let rendered = lf.render();
        assert!(rendered.contains("→"), "rendered = {}", rendered);
        let _ = model.fact_count();
    }

    #[test]
    fn unknown_input_gives_unknown_kind() {
        let semantics = FormalSemantics::new();
        let lf = semantics.parse("量子色动力学");
        assert!(!lf.is_known(), "lf = {:?}", lf);
    }

    #[test]
    fn semantic_memory_contextualizes_ellipsis_and_pronoun() {
        let mut memory = SemanticMemory::new(6);
        let lf = LogicalForm {
            source: "张三的年龄是多少".to_string(),
            tokens: vec!["张三".to_string(), "的".to_string(), "年龄".to_string()],
            kind: LfKind::Query(Query::Value {
                entity: "张三".to_string(),
                predicate: "年龄".to_string(),
            }),
            trace: Vec::new(),
        };
        memory.record(&lf);
        assert_eq!(memory.last_entity(), Some("张三"));
        assert_eq!(memory.last_predicate(), Some("年龄"));
        assert_eq!(
            memory.contextualize("李四呢").as_deref(),
            Some("李四的年龄是多少")
        );
        assert_eq!(
            memory.contextualize("他的年龄是多少").as_deref(),
            Some("张三的年龄是多少")
        );
        assert_eq!(memory.contextualize("张三的年龄是多少"), None);
    }
}

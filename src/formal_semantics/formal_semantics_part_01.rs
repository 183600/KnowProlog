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
        names.sort_by(|a, b| {
            b.chars()
                .count()
                .cmp(&a.chars().count())
                .then_with(|| a.cmp(b))
        });
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
/// 判断关系名是否表示子类层级（供外部构建微型模型时登记类）。
pub fn is_subclass_relation(rel_type: &str) -> bool {
    SUPER_RELS.contains(&rel_type.trim())
}
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
    binary_predicates: HashSet<String>,
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
            binary_predicates: HashSet::new(),
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

    /// 从一批已加载事实构造微型模型（与逐条 `observe_fact` 等价）。
    pub fn from_facts(facts: &[PrologFact], synonyms: HashMap<String, String>) -> Self {
        let mut model = Self::new(synonyms);
        for fact in facts {
            model.observe_fact(fact);
        }
        model
    }

    /// 名称是否已有个体域/类/取值层面的记录（数字字面量视为已知）。
    pub fn knows_name(&self, name: &str) -> bool {
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        if parse_number_token(name).is_some() {
            return true;
        }
        if self
            .name_variants(name)
            .iter()
            .any(|n| self.entities.contains(n) || self.class_set.contains(n))
        {
            return true;
        }
        self.class_like(name) || self.types.contains_key(name)
    }

    /// 谓词是否有任何外延记录（一元谓词、类型、属性键或二元关系）。
    pub fn knows_predicate(&self, predicate: &str) -> bool {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return false;
        }
        self.name_variants(predicate).iter().any(|p| {
            self.unary.contains_key(p)
                || self.class_set.contains(p)
                || self.binary_predicates.contains(p)
                || self.values.keys().any(|(_, key)| key == p)
        })
    }

    /// 逻辑项中既不在个体域、也没有谓词/类型记录的词面（证据缺口）。
    pub fn unknown_mentions(&self, term: &Term) -> Vec<String> {
        let mut constants = Vec::new();
        term.collect_consts(&mut constants);
        let mut predicates = Vec::new();
        collect_predicate_names(term, &mut predicates);
        let mut unknown = Vec::new();
        for name in constants {
            if !self.knows_name(&name) && !unknown.contains(&name) {
                unknown.push(name);
            }
        }
        for predicate in predicates {
            if !self.knows_predicate(&predicate) && !unknown.contains(&predicate) {
                unknown.push(predicate);
            }
        }
        unknown
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
        if !self
            .binary
            .insert((from.to_string(), rel_type.to_string(), to.to_string()))
        {
            return;
        }
        self.fact_count += 1;
        self.binary_predicates.insert(rel_type.to_string());
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

    /// 反向关系检索：返回满足 PREDICATE(subject, object) 的全部主语。
    pub fn subjects_with_object(&self, predicate: &str, object: &str) -> Vec<String> {
        let mut out = Vec::new();
        for o in self.name_variants(object) {
            if let Some(incoming) = self.in_index.get(&o) {
                for (rel, subject) in incoming {
                    if self.name_variants(predicate).contains(rel) && !out.contains(subject) {
                        out.push(subject.clone());
                    }
                }
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
        let symmetric = self
            .name_variants(predicate)
            .iter()
            .any(|p| SYMMETRIC_RELS.contains(&self.canon(p).as_str()));
        for e in self.name_variants(entity) {
            if let Some(outgoing) = self.out_index.get(&e) {
                for (rel, object) in outgoing {
                    if self.name_variants(predicate).contains(rel) && !out.contains(object) {
                        out.push(object.clone());
                    }
                }
            }
            // 非对称关系只沿出边取值；入边必须由「谁的 P 是 O」这类反向查询显式表达。
            if symmetric {
                if let Some(incoming) = self.in_index.get(&e) {
                    for (rel, subject) in incoming {
                        if self.name_variants(predicate).contains(rel) && !out.contains(subject) {
                            out.push(subject.clone());
                        }
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
        if class_variants.iter().any(|c| c == "实体" || c == "事物") {
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
        let holds = subjects.iter().any(|subject| {
            right_names
                .iter()
                .any(|target| values_match(subject, target))
        });
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
    /// 反向实体查询：谁的 PREDICATE 是 OBJECT
    EntityByObject { predicate: String, object: String },
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

    /// 逻辑形式的规范签名：只含可求值的语义内容，不含来源与轨迹。
    /// 同一签名的不同解析视为同一读法（用于候选森林去重与歧义判定）。
    pub fn signature(&self) -> String {
        match &self.kind {
            LfKind::Statement(term) => format!("statement:{}", term.render()),
            LfKind::Query(Query::Value { entity, predicate }) => {
                format!("value:{}=>{}", entity, predicate)
            }
            LfKind::Query(Query::Entity { predicate }) => format!("entity:{}", predicate),
            LfKind::Query(Query::EntityByObject { predicate, object }) => {
                format!("entity-rev:{}<-{}", predicate, object)
            }
            LfKind::Query(Query::Truth { body }) => format!("truth:{}", body.render()),
            LfKind::Unknown => "unknown".to_string(),
        }
    }

    /// 逻辑形式的一行渲染
    pub fn render(&self) -> String {
        match &self.kind {
            LfKind::Statement(term) => format!("{} : t", term.render()),
            LfKind::Query(Query::Value { entity, predicate }) => {
                format!("查询 x：{} 的 {} = x", entity, predicate)
            }
            LfKind::Query(Query::Entity { predicate }) => format!("查询 x：{}(x)", predicate),
            LfKind::Query(Query::EntityByObject { predicate, object }) => {
                format!("查询 x：{}(x, {})", predicate, object)
            }
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
            LfKind::Query(Query::EntityByObject { predicate, object }) => {
                (Some(object.clone()), Some(predicate.clone()))
            }
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
/// 收集逻辑项里出现的谓词名（不包括常量与量词变量）。
fn collect_predicate_names(term: &Term, out: &mut Vec<String>) {
    match term {
        Term::Pred(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Term::App(f, a) => {
            collect_predicate_names(f, out);
            collect_predicate_names(a, out);
        }
        Term::Lam(_, body)
        | Term::Not(body)
        | Term::Exists { body, .. }
        | Term::Forall { body, .. } => collect_predicate_names(body, out),
        Term::Attr(p, x) => {
            collect_predicate_names(p, out);
            collect_predicate_names(x, out);
        }
        Term::And(terms) | Term::Or(terms) => {
            for item in terms {
                collect_predicate_names(item, out);
            }
        }
        Term::Implies(a, b) | Term::Isa(a, b) | Term::Eq(a, b) => {
            collect_predicate_names(a, out);
            collect_predicate_names(b, out);
        }
        Term::Const(_) | Term::Num(_) | Term::Var(_) => {}
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
    EntitiesByObject {
        predicate: String,
        object: String,
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
impl QueryOutcome {
    /// 可答内容的规范签名：两个候选签名相同即视为同一答案。
    pub fn signature(&self) -> String {
        match self {
            QueryOutcome::Values {
                entity,
                predicate,
                values,
                ..
            } => format!("values:{}:{}:{}", entity, predicate, values.join("\u{1f}")),
            QueryOutcome::Entities {
                predicate,
                entities,
                ..
            } => format!("entities:{}:{}", predicate, entities.join("\u{1f}")),
            QueryOutcome::EntitiesByObject {
                predicate,
                object,
                entities,
                ..
            } => format!(
                "entities-rev:{}<-{}:{}",
                predicate,
                object,
                entities.join("\u{1f}")
            ),
            QueryOutcome::Truth {
                holds, statement, ..
            } => {
                format!("truth:{}:{}", holds, statement)
            }
            QueryOutcome::Unsupported(reason) => format!("unsupported:{}", reason),
        }
    }
}
/// 候选逻辑形式的来源策略（成本越低越优先保留旧路径的读法）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseStrategy {
    Conditional,
    QuestionPattern,
    Chart,
}
impl ParseStrategy {
    pub fn label(self) -> &'static str {
        match self {
            ParseStrategy::Conditional => "条件句",
            ParseStrategy::QuestionPattern => "问句模式",
            ParseStrategy::Chart => "图表组合",
        }
    }
}
/// 解析候选：逻辑形式 + 来源 + 成本。森林保留全部读法，不在此处压平成一条。
#[derive(Debug, Clone)]
pub struct ParseCandidate {
    pub form: LogicalForm,
    pub strategy: ParseStrategy,
    pub cost: usize,
}
/// 解析森林：同一输入的全部候选读法（按签名去重，保持发现顺序）。
#[derive(Debug, Clone)]
pub struct ParseForest {
    pub source: String,
    pub tokens: Vec<String>,
    pub candidates: Vec<ParseCandidate>,
    pub trace: Vec<String>,
}
impl ParseForest {
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn signatures(&self) -> Vec<String> {
        self.candidates
            .iter()
            .map(|candidate| candidate.form.signature())
            .collect()
    }
}
/// 候选在给定微型模型上的接地状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateStatus {
    /// 图谱中有正面证据（值/实体/成立的真值）。
    Supported,
    /// 名称与谓词均已知，但按封闭世界判为不成立。
    Refuted,
    /// 证据缺口：实体/谓词未知，或图谱没有相关记录。
    Open,
}
impl CandidateStatus {
    pub fn label(self) -> &'static str {
        match self {
            CandidateStatus::Supported => "有证据",
            CandidateStatus::Refuted => "被反驳",
            CandidateStatus::Open => "证据缺口",
        }
    }
}
#[derive(Debug, Clone)]
pub struct GroundedCandidate {
    pub candidate: ParseCandidate,
    pub outcome: QueryOutcome,
    pub status: CandidateStatus,
    pub gaps: Vec<String>,
}
/// 联合解析-推理的判定：唯一答案、歧义（多个不同答案）、或证据不足。
#[derive(Debug, Clone, PartialEq)]
pub enum ForestDecision {
    Answer { index: usize },
    Ambiguous { indexes: Vec<usize> },
    Unknown { reason: String },
}
#[derive(Debug, Clone)]
pub struct ForestEvaluation {
    pub forest: ParseForest,
    pub grounded: Vec<GroundedCandidate>,
    pub decision: ForestDecision,
}

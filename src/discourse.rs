// 符号主义话语引擎（OWL 本体叙述 + Prolog DCG 文法实现 + 多句连贯生成）
// 纯符号：零神经网络、零权重训练，全部由知识图谱、本体公理与文法规则驱动。
//
// 灵感：
//   • OWL：类层级、实例归属、继承属性、互斥约束都是“可叙述的知识”。
//     叙述时先用本体闭包给出定义（X 是一种 Y），再沿层级补充更一般的归属，
//     并把推导链（“为什么”）复述出来，让生成内容可溯源、可回放。
//   • Prolog：DCG（Definite Clause Grammar）是 Prolog 描述语言结构的标准工具。
//     这里实现一个微型 DCG 解释器：文法规则 → 终结符序列的回溯展开，
//     同一事实可有多种合规句式（是/为、的/关于……其），实现“符号化改写”。
//
// 话语规划（discourse planning）流水线：
//   定义开场 → 推理/继承补充 → 事实聚合（同谓词多值合并）→ 上下文衔接（焦点回指）
//   → 话语标记（此外/另外/值得一提的是）。
//   所有句子由 DCG 文法实现，而非硬编码模板字符串。

use std::collections::{HashMap, HashSet};

use crate::ontology_reasoning::{
    apply_ontology_lesson, parse_ontology_lesson, render_ontology_lesson_zh, OntologyCapsule,
    OntologyClosure, OntologyLesson, OntologyReasoner,
};
use crate::prolog_engine::PrologFact;

/// 话语标记：用于串联多个事实句，模拟自然段落的衔接词
const DISCOURSE_MARKERS: [&str; 4] = ["此外，", "另外，", "值得一提的是，", "补充一点，"];

// ---------------------------------------------------------------------------
// 一、DCG 文法引擎（Prolog 灵感）
// ---------------------------------------------------------------------------

/// DCG 规则体中的一个符号：非终结符（继续展开）或终结符（输出文本，可含 {槽位}）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DcgSymbol {
    Nonterminal(String),
    Terminal(String),
}

/// 一条 DCG 规则：head --> body...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcgRule {
    pub head: String,
    pub body: Vec<DcgSymbol>,
}

/// 微型 DCG 文法：支持回溯展开、槽位替换、多句式生成与 Prolog 文本导出
#[derive(Debug, Clone)]
pub struct DcgGrammar {
    rules: HashMap<String, Vec<DcgRule>>,
    order: Vec<String>,
    max_depth: usize,
    max_results: usize,
}

impl DcgGrammar {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            order: Vec::new(),
            max_depth: 8,
            max_results: 24,
        }
    }

    pub fn with_limits(max_depth: usize, max_results: usize) -> Self {
        Self {
            rules: HashMap::new(),
            order: Vec::new(),
            max_depth: max_depth.max(1),
            max_results: max_results.max(1),
        }
    }

    pub fn add_rule(&mut self, head: &str, body: Vec<DcgSymbol>) {
        let head = head.trim();
        if head.is_empty() {
            return;
        }
        if !self.rules.contains_key(head) {
            self.order.push(head.to_string());
        }
        self.rules
            .entry(head.to_string())
            .or_default()
            .push(DcgRule {
                head: head.to_string(),
                body,
            });
    }

    pub fn add_terminal_rule(&mut self, head: &str, text: &str) {
        self.add_rule(head, vec![DcgSymbol::Terminal(text.to_string())]);
    }

    /// 解析文本规则：`sentence --> entity "的" pred end`
    /// 终结符写法：`"文本"`、`[文本]`、`{槽位名}`；其余 token 视为非终结符。
    pub fn parse_rule(&mut self, text: &str) -> Result<(), String> {
        let (head, body) = if let Some(parts) = text.split_once("-->") {
            parts
        } else if let Some(parts) = text.split_once('→') {
            parts
        } else {
            return Err(format!("规则缺少 --> 或 →：{}", text));
        };
        let head = head.trim();
        if head.is_empty() {
            return Err("规则头为空".to_string());
        }
        let mut symbols = Vec::new();
        for token in body.split_whitespace() {
            let token = token.trim_end_matches('.').trim();
            if token.is_empty() {
                continue;
            }
            symbols.push(parse_dcg_token(token));
        }
        self.add_rule(head, symbols);
        Ok(())
    }

    pub fn nonterminals(&self) -> Vec<String> {
        self.order.clone()
    }

    pub fn alternative_count(&self, head: &str) -> usize {
        self.rules.get(head).map(Vec::len).unwrap_or(0)
    }

    /// 从 start 非终结符出发，回填槽位后枚举全部（去重的）终结符串
    pub fn generate(
        &self,
        start: &str,
        slots: &HashMap<String, String>,
        limit: usize,
    ) -> Vec<String> {
        let limit = limit.max(1).min(self.max_results);
        let expanded = self.expand_all(start, slots, 0);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for text in expanded {
            if text.trim().is_empty() {
                continue;
            }
            if seen.insert(text.clone()) {
                out.push(text);
                if out.len() >= limit {
                    break;
                }
            }
        }
        out
    }

    /// 取第一个实现（用于确定性输出）
    pub fn realize(&self, start: &str, slots: &HashMap<String, String>) -> Option<String> {
        self.generate(start, slots, 1).into_iter().next()
    }

    fn expand_all(
        &self,
        symbol: &str,
        slots: &HashMap<String, String>,
        depth: usize,
    ) -> Vec<String> {
        if depth > self.max_depth {
            return Vec::new();
        }
        let Some(rules) = self.rules.get(symbol) else {
            // 未定义符号按字面终结符处理（容错：允许直接把中文词写进规则体）
            return vec![substitute_slots(symbol, slots)];
        };
        let mut out: Vec<String> = Vec::new();
        for rule in rules {
            let mut parts: Vec<String> = vec![String::new()];
            let mut ok = true;
            for sym in &rule.body {
                let choices: Vec<String> = match sym {
                    DcgSymbol::Terminal(text) => vec![substitute_slots(text, slots)],
                    DcgSymbol::Nonterminal(name) => self.expand_all(name, slots, depth + 1),
                };
                if choices.is_empty() {
                    ok = false;
                    break;
                }
                let mut next: Vec<String> = Vec::with_capacity(parts.len() * choices.len());
                'outer: for prefix in &parts {
                    for choice in &choices {
                        let mut combined = prefix.clone();
                        combined.push_str(choice);
                        next.push(combined);
                        if next.len() >= self.max_results {
                            break 'outer;
                        }
                    }
                }
                parts = next;
                if parts.len() >= self.max_results {
                    break;
                }
            }
            if !ok {
                continue;
            }
            out.extend(parts);
            if out.len() >= self.max_results {
                out.truncate(self.max_results);
                break;
            }
        }
        out
    }

    /// 导出为 Prolog DCG 文本（可读、可校验，便于人工审查文法）
    pub fn to_prolog_rules(&self) -> String {
        let mut out = String::new();
        for head in &self.order {
            if let Some(rules) = self.rules.get(head) {
                for rule in rules {
                    out.push_str(head);
                    out.push_str(" --> ");
                    if rule.body.is_empty() {
                        out.push_str("[]");
                    } else {
                        let parts: Vec<String> = rule
                            .body
                            .iter()
                            .map(|sym| match sym {
                                DcgSymbol::Nonterminal(name) => name.clone(),
                                DcgSymbol::Terminal(text) => format!("['{}']", text),
                            })
                            .collect();
                        out.push_str(&parts.join(", "));
                    }
                    out.push_str(".\n");
                }
            }
        }
        out
    }
}

impl Default for DcgGrammar {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_dcg_token(token: &str) -> DcgSymbol {
    if token.len() >= 2 && token.starts_with('"') && token.ends_with('"') {
        return DcgSymbol::Terminal(token[1..token.len() - 1].to_string());
    }
    if token.len() >= 2 && token.starts_with('[') && token.ends_with(']') {
        return DcgSymbol::Terminal(token[1..token.len() - 1].to_string());
    }
    if token.len() >= 2 && token.starts_with('{') && token.ends_with('}') {
        return DcgSymbol::Terminal(token.to_string());
    }
    DcgSymbol::Nonterminal(token.to_string())
}

/// 把 `{槽位}` 占位替换为实际文本；未知槽位原样保留（便于排查文法错误）
fn substitute_slots(text: &str, slots: &HashMap<String, String>) -> String {
    if !text.contains('{') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end) => {
                let key = &after[..end];
                match slots.get(key) {
                    Some(value) => out.push_str(value),
                    None => {
                        out.push('{');
                        out.push_str(key);
                        out.push('}');
                    }
                }
                rest = &after[end + 1..];
            }
            None => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// 内置中文文法：事实句、定义句、聚合句。
/// 同一语义多种合规句式的来源就是“同一非终结符的多条规则”。
pub fn default_chinese_grammar() -> DcgGrammar {
    let mut grammar = DcgGrammar::new();
    for rule in [
        "sentence --> entity \"的\" pred copula objects end",
        "sentence --> \"关于\" entity \"，\" \"其\" pred copula objects end",
        "copula --> \"是\"",
        "copula --> \"为\"",
        "pred --> \"{p}\"",
        "objects --> \"{o}\"",
        "entity --> \"{s}\"",
        "end --> \"。\"",
        "definition --> entity \"是一种\" category end",
        "definition --> entity \"属于\" category end",
        "definition --> entity \"是\" category \"的一种\" end",
        "definition --> entity \"是\" category \"的一类\" end",
        "category --> \"{c}\"",
        "aggregate --> entity \"的\" pred \"包括\" objects end",
        "aggregate --> entity \"的\" pred \"有\" objects end",
    ] {
        let _ = grammar.parse_rule(rule);
    }
    grammar
}

// ---------------------------------------------------------------------------
// 二、OWL 本体叙述器（OWL 灵感）
// ---------------------------------------------------------------------------

/// 判断谓词是否表达本体层级公理（这类事实进本体，不做普通叙述）
pub fn is_hierarchy_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "子类" | "子类于" | "是一种" | "属于" | "subClassOf" | "rdf:type" | "类型"
    )
}

/// 实体的本体画像：叙述生成的知识来源
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConceptBrief {
    pub entity: String,
    /// 全部所属类（含推导）
    pub types: Vec<String>,
    /// 最具体的类（叙述时作为“是一种 X”的 X）
    pub primary_type: Option<String>,
    /// primary_type 的更一般上位类
    pub ancestors: Vec<String>,
    /// 同类已知实例（不含自身）
    pub siblings: Vec<String>,
    /// 继承而来的属性 (谓词, 值, 推导理由)
    pub inherited: Vec<(String, String, String)>,
    /// 推导得到的类型 (类, 推导理由)
    pub derived_types: Vec<(String, String)>,
    /// 与实体相关的一致性问题
    pub problems: Vec<String>,
}

/// OWL 本体叙述器：观察事实 → 闭包物化 → 输出可叙述的画像与“为什么”
#[derive(Clone)]
pub struct OwlNarrator {
    ontology: OntologyReasoner,
    closure: OntologyClosure,
    known: HashSet<String>,
    seen_entities: HashMap<String, HashSet<String>>,
    seen_hierarchy: HashMap<String, HashSet<String>>,
    seen_relations: HashMap<String, HashMap<String, HashSet<String>>>,
    dirty: bool,
    /// 被教过的本体公理（含配置并入的结构化公理）：胶囊的公理来源，
    /// 保存 / 分叉 / 重建后据此重放，教过的本体不会消失。
    taught: Vec<OntologyLesson>,
}

impl OwlNarrator {
    pub fn new(max_rounds: usize) -> Self {
        Self {
            ontology: OntologyReasoner::new(max_rounds),
            closure: OntologyClosure::default(),
            known: HashSet::new(),
            seen_entities: HashMap::new(),
            seen_hierarchy: HashMap::new(),
            seen_relations: HashMap::new(),
            dirty: false,
            taught: Vec::new(),
        }
    }

    /// 观察一批事实：Entity 事实给出实例归属，层级谓词给出子类公理，其余进关系闭包。
    /// 内部去重，可安全地对同一批事实反复调用（主循环每轮都会重放知识图谱快照）。
    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    if name.trim().is_empty() {
                        continue;
                    }
                    if !self.known.contains(name.as_str()) {
                        self.known.insert(name.clone());
                    }
                    if !entity_type.trim().is_empty() {
                        if !self.known.contains(entity_type.as_str()) {
                            self.known.insert(entity_type.clone());
                        }
                        if self.mark_entity_seen(name, entity_type) {
                            self.ontology.add_instance(name, entity_type);
                            self.dirty = true;
                        }
                    }
                    for (key, value) in properties {
                        self.observe_relation(name, key, value);
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    self.observe_relation(from, rel_type, to);
                }
            }
        }
        self.materialize_if_dirty();
    }

    fn observe_relation(&mut self, from: &str, predicate: &str, to: &str) {
        if from.trim().is_empty() || predicate.trim().is_empty() || to.trim().is_empty() {
            return;
        }
        if !self.known.contains(from) {
            self.known.insert(from.to_string());
        }
        if !self.known.contains(to) {
            self.known.insert(to.to_string());
        }
        if is_hierarchy_predicate(predicate) {
            if self.mark_hierarchy_seen(from, to) {
                self.ontology.add_subclass(from, to);
                self.dirty = true;
            }
        } else if self.mark_relation_seen(from, predicate, to) {
            self.ontology.add_relation(from, predicate, to);
            self.dirty = true;
        }
    }

    /// 记录“名称-类型”二元组；返回是否为首次出现
    fn mark_entity_seen(&mut self, name: &str, entity_type: &str) -> bool {
        match self.seen_entities.get_mut(name) {
            Some(types) => {
                if types.contains(entity_type) {
                    false
                } else {
                    types.insert(entity_type.to_string());
                    true
                }
            }
            None => {
                let mut types = HashSet::new();
                types.insert(entity_type.to_string());
                self.seen_entities.insert(name.to_string(), types);
                true
            }
        }
    }

    /// 记录“子类-超类”二元组；返回是否为首次出现
    fn mark_hierarchy_seen(&mut self, from: &str, to: &str) -> bool {
        match self.seen_hierarchy.get_mut(from) {
            Some(targets) => {
                if targets.contains(to) {
                    false
                } else {
                    targets.insert(to.to_string());
                    true
                }
            }
            None => {
                let mut targets = HashSet::new();
                targets.insert(to.to_string());
                self.seen_hierarchy.insert(from.to_string(), targets);
                true
            }
        }
    }

    /// 记录“主语-谓词-宾语”三元组；返回是否为首次出现
    fn mark_relation_seen(&mut self, from: &str, predicate: &str, to: &str) -> bool {
        match self.seen_relations.get_mut(from) {
            Some(predicates) => match predicates.get_mut(predicate) {
                Some(objects) => {
                    if objects.contains(to) {
                        false
                    } else {
                        objects.insert(to.to_string());
                        true
                    }
                }
                None => {
                    let mut objects = HashSet::new();
                    objects.insert(to.to_string());
                    predicates.insert(predicate.to_string(), objects);
                    true
                }
            },
            None => {
                let mut objects = HashSet::new();
                objects.insert(to.to_string());
                let mut predicates = HashMap::new();
                predicates.insert(predicate.to_string(), objects);
                self.seen_relations.insert(from.to_string(), predicates);
                true
            }
        }
    }

    /// 教一条中文本体公理（子类/互斥/定义域/固有属性……），供继承推理与叙述使用
    pub fn teach_axiom(&mut self, text: &str) -> Result<String, String> {
        let lesson = parse_ontology_lesson(text)?;
        self.install_lesson(&lesson);
        self.materialize_if_dirty();
        Ok(render_ontology_lesson_zh(&lesson))
    }

    /// 写入一条公理并登记涉及的名字：教学与胶囊恢复共用的唯一入口。
    /// 公理是单调集合，重复写入幂等（`taught` 去重，闭包按集合物化）。
    pub fn install_lesson(&mut self, lesson: &OntologyLesson) {
        apply_ontology_lesson(&mut self.ontology, lesson);
        for name in self.ontology.class_names() {
            self.known.insert(name);
        }
        // 个体、属性等名字也要登记，否则 brief/叙述会误判为未知实体
        match lesson {
            OntologyLesson::Instance {
                individual, class, ..
            } => {
                self.known.insert(individual.clone());
                self.known.insert(class.clone());
            }
            OntologyLesson::Subclass { sub, sup, .. } => {
                self.known.insert(sub.clone());
                self.known.insert(sup.clone());
            }
            OntologyLesson::ClassProperty { class, .. } => {
                self.known.insert(class.clone());
            }
            OntologyLesson::Domain { class, .. } | OntologyLesson::Range { class, .. } => {
                self.known.insert(class.clone());
            }
            OntologyLesson::Disjoint { left, right, .. } => {
                self.known.insert(left.clone());
                self.known.insert(right.clone());
            }
            OntologyLesson::Subproperty { sub, sup, .. } => {
                self.known.insert(sub.clone());
                self.known.insert(sup.clone());
            }
            _ => {}
        }
        if !self.taught.iter().any(|known| known == lesson) {
            self.taught.push(lesson.clone());
        }
        self.dirty = true;
    }

    pub fn materialize_if_dirty(&mut self) {
        if self.dirty {
            self.closure = self.ontology.materialize();
            self.dirty = false;
        }
    }

    pub fn closure(&mut self) -> &OntologyClosure {
        self.materialize_if_dirty();
        &self.closure
    }

    pub fn is_known(&self, entity: &str) -> bool {
        self.known.contains(entity)
    }

    /// 已登记的公理数（去重后；含配置并入的结构化公理）。
    pub fn taught_axioms(&self) -> usize {
        self.taught.len()
    }

    /// 已登记的公理（供审计与程序合并）。
    pub fn taught_lessons(&self) -> &[OntologyLesson] {
        &self.taught
    }

    /// 导出本体胶囊：公理 + 规范摘要 + 当前闭包缓存。
    /// 从未教过公理时返回 `None`（旧快照的字节保持不变）。
    /// 只读：闭包脏时就地物化一次，不改动缓存状态。
    pub fn ontology_capsule(&self) -> Option<OntologyCapsule> {
        if self.taught.is_empty() {
            return None;
        }
        let closure = if self.dirty {
            self.ontology.materialize()
        } else {
            self.closure.clone()
        };
        Some(OntologyCapsule::new(self.taught.clone(), &closure))
    }

    /// 恢复本体胶囊：先过版本门与公理摘要校验，再按公理重放。
    /// 闭包一律重算（缓存只是派生数据），校验失败返回 `Err` 且不改动状态。
    pub fn restore_ontology_capsule(&mut self, capsule: &OntologyCapsule) -> Result<(), String> {
        capsule.verify()?;
        for lesson in &capsule.lessons {
            self.install_lesson(lesson);
        }
        self.materialize_if_dirty();
        Ok(())
    }

    pub fn types_of(&mut self, entity: &str) -> Vec<String> {
        self.materialize_if_dirty();
        self.closure.types_of(entity)
    }

    /// 生成实体的本体画像；未知实体返回 None
    pub fn brief(&mut self, entity: &str) -> Option<ConceptBrief> {
        if !self.known.contains(entity) {
            return None;
        }
        self.materialize_if_dirty();
        let closure = &self.closure;
        let mut all_types: Vec<String> = closure
            .types_of(entity)
            .into_iter()
            .filter(|t| t != entity)
            .collect();
        if all_types.is_empty() {
            // 实体本身可能是一个类：用它的上位类作为“所属”
            all_types = closure.ancestors_of(entity);
        }
        all_types.sort();
        all_types.dedup();
        let most_specific = most_specific_types(closure, &all_types);
        let primary_type = most_specific.first().cloned();
        let mut ancestors: Vec<String> = all_types
            .iter()
            .filter(|t| {
                primary_type
                    .as_deref()
                    .map(|primary| closure.is_subclass(primary, t))
                    .unwrap_or(false)
            })
            .filter(|t| Some(t.as_str()) != primary_type.as_deref())
            .cloned()
            .collect();
        ancestors.sort();
        ancestors.dedup();
        let siblings = match &primary_type {
            Some(primary) => {
                let mut members = closure.instances_of(primary);
                members.retain(|member| member != entity);
                members.truncate(5);
                members
            }
            None => Vec::new(),
        };
        let inherited: Vec<(String, String, String)> = closure
            .properties_of(entity)
            .into_iter()
            .filter_map(|(predicate, object)| {
                closure
                    .why_relation(entity, &predicate, &object)
                    .filter(|why| why.contains("继承"))
                    .map(|why| (predicate, object, why.to_string()))
            })
            .collect();
        let derived_types: Vec<(String, String)> = all_types
            .iter()
            .filter_map(|etype| {
                closure
                    .why_instance(entity, etype)
                    .filter(|why| *why != "直接声明")
                    .map(|why| (etype.clone(), why.to_string()))
            })
            .collect();
        let problems: Vec<String> = closure
            .problems()
            .iter()
            .filter(|problem| problem.contains(entity))
            .cloned()
            .collect();
        Some(ConceptBrief {
            entity: entity.to_string(),
            types: all_types,
            primary_type,
            ancestors,
            siblings,
            inherited,
            derived_types,
            problems,
        })
    }
}

/// 最具体类型：候选集中不存在其它候选的严格子类者
fn most_specific_types(closure: &OntologyClosure, candidates: &[String]) -> Vec<String> {
    let mut out: Vec<String> = candidates
        .iter()
        .filter(|candidate| {
            !candidates
                .iter()
                .any(|other| other != *candidate && closure.is_subclass(other, candidate))
        })
        .cloned()
        .collect();
    out.sort();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------
// 三、话语规划器：把本体画像 + 图谱事实规划成连贯段落
// ---------------------------------------------------------------------------

/// 叙述选项
#[derive(Debug, Clone, Default)]
pub struct NarrationOptions<'a> {
    /// 当前对话焦点实体：用于回指（用“它”代替实体名）与结尾衔接
    pub focus: Option<&'a str>,
    /// 话题关键词：命中的谓词优先叙述（用于“关于 X 的 P”式追问）
    pub topic: Option<&'a str>,
    /// 句子数上限；0 表示使用规划器默认值
    pub max_sentences: usize,
}

/// 话语规划器：索引事实 + 维护本体 + 生成多句连贯叙述与句式改写
#[derive(Clone)]
pub struct DiscoursePlanner {
    narrator: OwlNarrator,
    grammar: DcgGrammar,
    by_subject: HashMap<String, Vec<(String, Vec<String>)>>,
    predicate_freq: HashMap<String, usize>,
    known_entities: HashSet<String>,
    seen_triples: HashMap<String, HashMap<String, HashSet<String>>>,
    max_sentences: usize,
}

impl DiscoursePlanner {
    pub fn new(max_rounds: usize, max_sentences: usize) -> Self {
        Self {
            narrator: OwlNarrator::new(max_rounds),
            grammar: default_chinese_grammar(),
            by_subject: HashMap::new(),
            predicate_freq: HashMap::new(),
            known_entities: HashSet::new(),
            seen_triples: HashMap::new(),
            max_sentences: max_sentences.max(1),
        }
    }

    pub fn grammar(&self) -> &DcgGrammar {
        &self.grammar
    }

    pub fn narrator_mut(&mut self) -> &mut OwlNarrator {
        &mut self.narrator
    }

    pub fn relation_count(&self) -> usize {
        self.seen_triples
            .values()
            .flat_map(|predicates| {
                predicates
                    .iter()
                    .filter(|(predicate, _)| !is_hierarchy_predicate(predicate))
                    .map(|(_, objects)| objects.len())
            })
            .sum()
    }

    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name, properties, ..
                } => {
                    if !self.known_entities.contains(name.as_str()) {
                        self.known_entities.insert(name.clone());
                    }
                    for (key, value) in properties {
                        self.observe_triple(name, key, value);
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    self.observe_triple(from, rel_type, to);
                }
            }
        }
        self.narrator.observe_facts(facts);
    }

    fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.trim().is_empty() || predicate.trim().is_empty() || object.trim().is_empty() {
            return;
        }
        if !self.known_entities.contains(subject) {
            self.known_entities.insert(subject.to_string());
        }
        if !self.known_entities.contains(object) {
            self.known_entities.insert(object.to_string());
        }
        // 借用查找优先：重放快照时重复三元组直接跳过
        let already_seen = self
            .seen_triples
            .get(subject)
            .and_then(|predicates| predicates.get(predicate))
            .map(|objects| objects.contains(object))
            .unwrap_or(false);
        if already_seen {
            return;
        }
        self.seen_triples
            .entry(subject.to_string())
            .or_default()
            .entry(predicate.to_string())
            .or_default()
            .insert(object.to_string());
        *self
            .predicate_freq
            .entry(predicate.to_string())
            .or_insert(0) += 1;
        let entry = self.by_subject.entry(subject.to_string()).or_default();
        match entry.iter_mut().find(|(pred, _)| pred == predicate) {
            Some((_, objects)) => objects.push(object.to_string()),
            None => entry.push((predicate.to_string(), vec![object.to_string()])),
        }
    }

    pub fn teach_axiom(&mut self, text: &str) -> Result<String, String> {
        self.narrator.teach_axiom(text)
    }

    /// 直接写入一条结构化公理（配置并入单一本体程序的入口，幂等）。
    pub fn install_lesson(&mut self, lesson: &OntologyLesson) {
        self.narrator.install_lesson(lesson);
        self.narrator.materialize_if_dirty();
    }

    /// 已登记的公理（供审计与程序合并）。
    pub fn taught_lessons(&self) -> &[OntologyLesson] {
        self.narrator.taught_lessons()
    }

    /// 导出本体胶囊（未教过公理时为 `None`）。
    pub fn ontology_capsule(&self) -> Option<OntologyCapsule> {
        self.narrator.ontology_capsule()
    }

    /// 恢复本体胶囊：版本门 + 公理摘要校验通过才重放。
    pub fn restore_ontology_capsule(&mut self, capsule: &OntologyCapsule) -> Result<(), String> {
        self.narrator.restore_ontology_capsule(capsule)
    }

    /// 多句连贯叙述：定义开场 → 推理/继承补充 → 事实聚合 → 焦点衔接
    pub fn narrate(
        &mut self,
        entity: &str,
        focus: Option<&str>,
        topic: Option<&str>,
        max_sentences: usize,
        surfaces: &HashMap<String, String>,
    ) -> Option<String> {
        let limit = if max_sentences == 0 {
            self.max_sentences
        } else {
            max_sentences
        }
        .max(1);
        if !self.known_entities.contains(entity) && !self.narrator.is_known(entity) {
            return None;
        }
        let has_facts = self
            .by_subject
            .get(entity)
            .map(|facts| !facts.is_empty())
            .unwrap_or(false);
        let brief = self.narrator.brief(entity);
        if !has_facts && brief.is_none() {
            return None;
        }
        let reference = if focus == Some(entity) {
            "它".to_string()
        } else {
            entity.to_string()
        };
        let mut sentences: Vec<String> = Vec::new();

        // 1. 本体定义开场（最具体的类）
        if let Some(brief) = &brief {
            if let Some(primary) = brief.primary_type.as_deref() {
                if let Some(sentence) = self.definition_sentence(&reference, primary, 0) {
                    sentences.push(sentence);
                }
            }
        }

        // 2. 事实句：命中话题的谓词排序在前（话题追问优先作答）
        if let Some(facts) = self.by_subject.get(entity) {
            let mut ranked: Vec<&(String, Vec<String>)> = facts
                .iter()
                .filter(|(predicate, _)| !is_hierarchy_predicate(predicate))
                .collect();
            ranked.sort_by(|a, b| {
                let topic_rank_a = topic_rank(&a.0, surfaces, topic);
                let topic_rank_b = topic_rank(&b.0, surfaces, topic);
                topic_rank_a
                    .cmp(&topic_rank_b)
                    .then_with(|| {
                        self.predicate_freq
                            .get(&a.0)
                            .copied()
                            .unwrap_or(0)
                            .cmp(&self.predicate_freq.get(&b.0).copied().unwrap_or(0))
                    })
                    .then_with(|| a.0.cmp(&b.0))
            });
            let mut fact_index = 0usize;
            for (predicate, objects) in ranked {
                if sentences.len() >= limit {
                    break;
                }
                let surface = surface_of(predicate, surfaces);
                let variants = self.fact_variants(&reference, surface, objects);
                if variants.is_empty() {
                    continue;
                }
                let picked = &variants[fact_index % variants.len()];
                let sentence = if sentences.is_empty() {
                    picked.clone()
                } else {
                    format!(
                        "{}{}",
                        DISCOURSE_MARKERS[fact_index % DISCOURSE_MARKERS.len()],
                        picked
                    )
                };
                sentences.push(sentence);
                fact_index += 1;
            }
        }

        // 3. 推理与背景补充：更一般的归属、同类实例、继承属性、推导结论、一致性提示
        if let Some(brief) = &brief {
            if sentences.len() < limit {
                if let Some(primary) = brief.primary_type.as_deref() {
                    if let Some(ancestor) = brief.ancestors.first() {
                        if let Some(sentence) = self.definition_sentence(primary, ancestor, 1) {
                            sentences.push(format!("进一步说，{}", sentence));
                        }
                    }
                    if !brief.siblings.is_empty() && sentences.len() < limit {
                        sentences.push(format!(
                            "与{}同属{}的已知实体还有{}。",
                            entity,
                            crate::symbolic_generator::localized_type_label(primary),
                            brief.siblings.join("、")
                        ));
                    }
                }
            }
            if sentences.len() < limit {
                if let Some(sentence) = self.inherited_sentence(&reference, brief, surfaces) {
                    sentences.push(sentence);
                }
            }
            if let Some((class, why)) = brief.derived_types.first() {
                if sentences.len() < limit {
                    sentences.push(format!(
                        "由此可知，{}也属于{}（依据：{}）。",
                        reference, class, why
                    ));
                }
            }
            for problem in brief.problems.iter().take(1) {
                if sentences.len() < limit {
                    sentences.push(format!("注意：{}。", problem));
                }
            }
        }

        if sentences.len() < limit {
            if let Some(focus_entity) = focus.filter(|focus_entity| *focus_entity != entity) {
                if let Some(sentence) = self.bridge_sentence(entity, focus_entity, &brief, surfaces)
                {
                    sentences.push(sentence);
                }
            }
        }

        if sentences.is_empty() {
            return None;
        }
        Some(sentences.join(""))
    }

    /// 对同一条事实给出多种合规句式（DCG 回溯展开去重）
    pub fn paraphrase(
        &self,
        entity: &str,
        predicate: &str,
        limit: usize,
        surfaces: &HashMap<String, String>,
    ) -> Vec<String> {
        let Some(facts) = self.by_subject.get(entity) else {
            return Vec::new();
        };
        let Some((key, objects)) = facts.iter().find(|(canonical, _)| {
            canonical == predicate || surface_of(canonical, surfaces) == predicate
        }) else {
            return Vec::new();
        };
        let surface = surface_of(key, surfaces);
        self.fact_variants(entity, surface, objects)
            .into_iter()
            .take(limit.max(1))
            .collect()
    }

    fn fact_variants(&self, reference: &str, surface: &str, objects: &[String]) -> Vec<String> {
        let mut slots = HashMap::new();
        slots.insert("s".to_string(), reference.to_string());
        slots.insert("p".to_string(), surface.to_string());
        if objects.len() == 1 {
            slots.insert("o".to_string(), objects[0].clone());
            self.grammar.generate("sentence", &slots, 4)
        } else {
            slots.insert("o".to_string(), join_objects(objects, 3));
            self.grammar.generate("aggregate", &slots, 4)
        }
    }

    fn definition_sentence(&self, subject: &str, category: &str, variant: usize) -> Option<String> {
        let category = crate::symbolic_generator::localized_type_label(category);
        let mut slots = HashMap::new();
        slots.insert("s".to_string(), subject.to_string());
        slots.insert("c".to_string(), category.to_string());
        let variants = self.grammar.generate("definition", &slots, 8);
        if variants.is_empty() {
            None
        } else {
            Some(variants[variant % variants.len()].clone())
        }
    }

    fn inherited_sentence(
        &self,
        reference: &str,
        brief: &ConceptBrief,
        surfaces: &HashMap<String, String>,
    ) -> Option<String> {
        let (predicate, object, _) = brief.inherited.first()?;
        let surface = surface_of(predicate, surfaces);
        Some(format!(
            "此外，作为{}的一员，{}的{}是{}。",
            crate::symbolic_generator::localized_type_label(
                brief.primary_type.as_deref().unwrap_or("该类")
            ),
            reference,
            surface,
            object
        ))
    }

    fn bridge_sentence(
        &mut self,
        entity: &str,
        focus: &str,
        brief: &Option<ConceptBrief>,
        surfaces: &HashMap<String, String>,
    ) -> Option<String> {
        if let Some(brief) = brief {
            if let Some(primary) = brief.primary_type.as_deref() {
                let focus_types = self.narrator.types_of(focus);
                if focus_types.iter().any(|etype| etype == primary) {
                    return Some(format!(
                        "它与前面提到的「{}」同属{}。",
                        focus,
                        crate::symbolic_generator::localized_type_label(primary)
                    ));
                }
            }
        }
        let focus_facts = self.by_subject.get(focus)?;
        let entity_facts = self.by_subject.get(entity)?;
        for (predicate, objects) in entity_facts {
            if is_hierarchy_predicate(predicate) {
                continue;
            }
            for (focus_predicate, focus_objects) in focus_facts {
                if focus_predicate != predicate {
                    continue;
                }
                if let Some(shared) = objects.iter().find(|object| focus_objects.contains(object)) {
                    let surface = surface_of(predicate, surfaces);
                    return Some(format!(
                        "它和前面提到的「{}」的{}都是{}。",
                        focus, surface, shared
                    ));
                }
            }
        }
        None
    }
}

/// 谓词的中文表述（优先本地化键名，其次用生成器登记的动词化表述）
fn surface_of<'a>(predicate: &'a str, surfaces: &'a HashMap<String, String>) -> &'a str {
    if let Some(localized) = crate::symbolic_generator::localized_predicate_key(predicate) {
        return localized;
    }
    surfaces
        .get(predicate)
        .map(String::as_str)
        .unwrap_or(predicate)
}

/// 话题匹配度：命中话题的谓词排序在前
fn topic_rank(predicate: &str, surfaces: &HashMap<String, String>, topic: Option<&str>) -> u8 {
    match topic {
        None => 1,
        Some(topic) if topic.is_empty() => 1,
        Some(topic) => {
            let surface = surface_of(predicate, surfaces);
            if predicate.contains(topic) || surface.contains(topic) {
                0
            } else {
                1
            }
        }
    }
}

/// 合并多值对象：去重、最多展示 cap 个，超出用“等”收束
fn join_objects(objects: &[String], cap: usize) -> String {
    let mut unique: Vec<String> = Vec::new();
    for object in objects {
        if !unique.contains(object) {
            unique.push(object.clone());
        }
    }
    let shown = &unique[..unique.len().min(cap.max(1))];
    let mut text = shown.join("、");
    if unique.len() > cap.max(1) {
        text.push_str("等");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_facts() -> Vec<PrologFact> {
        let mut properties = HashMap::new();
        properties.insert("职业".to_string(), "教师".to_string());
        vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人".to_string(),
                properties,
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "北京大学".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "清华大学".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "任教于".to_string(),
                to: "北京大学".to_string(),
            },
            PrologFact::Relation {
                from: "人".to_string(),
                rel_type: "子类".to_string(),
                to: "动物".to_string(),
            },
        ]
    }

    fn sample_planner() -> DiscoursePlanner {
        let mut planner = DiscoursePlanner::new(4, 6);
        planner.observe_facts(&sample_facts());
        planner
    }

    fn slots(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn test_dcg_generates_all_sentence_variants() {
        let grammar = default_chinese_grammar();
        let filled = slots(&[("s", "张三"), ("p", "职业"), ("o", "教师")]);
        let variants = grammar.generate("sentence", &filled, 8);
        assert_eq!(variants.len(), 4, "两种句式 × 两种系动词 = 4 个变体");
        assert!(variants.iter().all(|text| text.ends_with('。')));
        assert!(variants
            .iter()
            .all(|text| text.contains("张三") && text.contains("教师")));
        let mut unique = variants.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), variants.len(), "变体必须互不相同");
    }

    #[test]
    fn test_dcg_rule_parsing_and_generation() {
        let mut grammar = DcgGrammar::new();
        grammar
            .parse_rule("greeting --> hello name")
            .expect("应能解析规则");
        grammar.add_terminal_rule("hello", "你好");
        grammar.add_terminal_rule("name", "{who}");
        let filled = slots(&[("who", "张三")]);
        assert_eq!(
            grammar.realize("greeting", &filled).as_deref(),
            Some("你好张三")
        );
    }

    #[test]
    fn test_dcg_to_prolog_rules_render() {
        let grammar = default_chinese_grammar();
        let text = grammar.to_prolog_rules();
        assert!(text.contains("sentence --> "));
        assert!(text.contains("['的']"));
        assert!(text.lines().count() >= 10);
    }

    #[test]
    fn test_dcg_depth_guard_terminates_on_left_recursion() {
        let mut grammar = DcgGrammar::with_limits(4, 8);
        let _ = grammar.parse_rule("loop --> loop tail");
        grammar.add_terminal_rule("tail", "尾");
        let _ = grammar.generate("loop", &HashMap::new(), 4);
        assert!(grammar
            .generate("loop", &HashMap::new(), 4)
            .iter()
            .all(|text| text.len() <= 4));
    }

    #[test]
    fn test_owl_narrator_brief_inherits_hierarchy() {
        let mut narrator = OwlNarrator::new(4);
        narrator.observe_facts(&sample_facts());
        let brief = narrator.brief("张三").expect("应能生成本体画像");
        assert_eq!(brief.primary_type.as_deref(), Some("人"));
        assert!(
            brief.types.contains(&"动物".to_string()),
            "类型应沿子类上推"
        );
        assert!(brief.ancestors.contains(&"动物".to_string()));
        assert!(brief.siblings.contains(&"李四".to_string()));
        assert!(brief
            .derived_types
            .iter()
            .any(|(class, why)| class == "动物" && why.contains("⊑")));
    }

    #[test]
    fn test_owl_narrator_teach_axiom_and_class_property_inheritance() {
        let mut narrator = OwlNarrator::new(4);
        narrator
            .teach_axiom("人是一种动物")
            .expect("子类公理应可学习");
        narrator
            .teach_axiom("人的寿命是100年")
            .expect("固有属性应可学习");
        narrator.observe_facts(&[PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }]);
        let brief = narrator.brief("苏格拉底").expect("应能生成画像");
        assert_eq!(brief.primary_type.as_deref(), Some("人"));
        assert!(
            brief
                .inherited
                .iter()
                .any(|(predicate, value, why)| predicate == "寿命"
                    && value == "100年"
                    && why.contains("继承")),
            "类固有属性应被实例继承：{:?}",
            brief.inherited
        );
    }

    #[test]
    fn test_discourse_narrate_is_coherent_and_aggregates() {
        let mut planner = sample_planner();
        let text = planner
            .narrate("张三", None, None, 6, &HashMap::new())
            .expect("应能生成叙述");
        assert!(text.contains("张三"));
        assert!(text.contains("是一种人"), "应以本体定义开场：{}", text);
        assert!(text.contains("动物"), "应沿层级补充更一般的归属：{}", text);
        assert!(
            text.contains("北京大学") && text.contains("、") && text.contains("清华大学"),
            "同谓词多值应聚合：{}",
            text
        );
        assert!(
            !text.contains("子类"),
            "层级谓词不应作为普通事实叙述：{}",
            text
        );
        assert!(
            text.matches('。').count() <= 6,
            "句数应受上限约束：{}",
            text
        );
    }

    #[test]
    fn test_discourse_narrate_uses_pronoun_for_focus_entity() {
        let mut planner = sample_planner();
        let text = planner
            .narrate("张三", Some("张三"), None, 4, &HashMap::new())
            .expect("应能生成叙述");
        assert!(
            text.starts_with("它是一种人"),
            "焦点实体应回指为“它”：{}",
            text
        );
    }

    #[test]
    fn test_discourse_narrate_bridges_to_previous_focus() {
        let mut planner = sample_planner();
        let text = planner
            .narrate("李四", Some("张三"), None, 6, &HashMap::new())
            .expect("应能生成叙述");
        assert!(
            text.contains("同属") || text.contains("都是"),
            "应与上一焦点建立衔接：{}",
            text
        );
    }

    #[test]
    fn test_discourse_topic_ranking_prioritizes_keyword() {
        let mut planner = sample_planner();
        let text = planner
            .narrate("张三", None, Some("任教"), 3, &HashMap::new())
            .expect("应能生成叙述");
        let teaching_pos = text.find("任教于").expect("话题谓词应出现");
        let profession_pos = text.find("职业").expect("其它事实也应出现");
        assert!(
            teaching_pos < profession_pos,
            "命中话题的谓词应排前：{}",
            text
        );
    }

    #[test]
    fn test_discourse_paraphrase_yields_distinct_variants() {
        let planner = sample_planner();
        let variants = planner.paraphrase("张三", "职业", 4, &HashMap::new());
        assert!(variants.len() >= 2, "应给出多种合规句式：{:?}", variants);
        let mut unique = variants.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), variants.len());
        assert!(variants
            .iter()
            .all(|text| text.contains("张三") && text.contains("教师")));
    }

    #[test]
    fn test_discourse_narrate_unknown_entity_returns_none() {
        let mut planner = sample_planner();
        assert!(planner
            .narrate("不存在的人", None, None, 4, &HashMap::new())
            .is_none());
    }

    #[test]
    fn test_repeated_observation_is_idempotent() {
        let mut planner = sample_planner();
        let before = planner.relation_count();
        planner.observe_facts(&sample_facts());
        planner.observe_facts(&sample_facts());
        assert_eq!(planner.relation_count(), before, "重复观察不应重复计数");
        let text = planner
            .narrate("张三", None, None, 6, &HashMap::new())
            .expect("重复观察后仍应能叙述");
        assert_eq!(
            text.matches("任教于").count(),
            1,
            "事实不应重复出现：{}",
            text
        );
    }
}

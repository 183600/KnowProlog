//! SG：双向语义语法（Semantic Grammar）——让抽取规则同时学会「读」与「说」。
//!
//! 旧的 `[[extraction_rules]]` 是单向投影：词性模式命中后，用
//! `property_mappings` 把若干词性槽位取出来，既不能命名捕获，也没有任何
//! 反向信息。生成端只能另维护一套谓词词典与模板，于是「同一句话」在读侧
//! 与写侧可以长出两套词汇。
//!
//! 本模块给出一份可逆的规则包：
//!
//! - 正向：`$s:pos:nr 身高 $o:pos:m` → `FactRef { subject, height, value }`；
//! - 反向：同一条规则的 `realize = "{s}身高{o}"` → 人话。
//!
//! 规则默认关闭（`[semantic_grammar] enabled = false`），旧配置零行为变化。
//! 每条命中都保留规则名与字符级跨度，抽取与生成可共享同一份出处。

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::nlp::{preprocess, TaggedWord};
use crate::provenance::FactRef;
use crate::{KnowledgeItem, SystemConfig};

/// 语义语法配置：[`semantic_grammar`] 段。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SemanticGrammarConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub rules: Vec<SemanticRuleConfig>,
}

/// 一条双向规则的原始配置。
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticRuleConfig {
    pub name: String,
    pub predicate: String,
    #[serde(default)]
    pub entity_type: Option<String>,
    /// 正向模式；token 形式见 [`RulePack::compile`]。
    #[serde(default)]
    pub pattern: Vec<String>,
    /// 反向模板，支持 `{s}` / `{o}` / `{p}`。
    #[serde(default)]
    pub realize: Option<String>,
    /// 关系事实（`value` 是实体）还是属性事实（`value` 是字面量）。
    #[serde(default)]
    pub relation: bool,
    #[serde(default)]
    pub priority: i32,
    /// 额外守门词：任一出现才尝试本规则。
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// 编译后的模式原子。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PatternAtom {
    /// 命名捕获；`pos` 为空表示任意词。
    Capture {
        slot: String,
        #[serde(default)]
        pos: Vec<String>,
        #[serde(default)]
        optional: bool,
    },
    /// 匿名词性断言。
    Pos {
        #[serde(default)]
        pos: Vec<String>,
        #[serde(default)]
        optional: bool,
    },
    /// 字面词；`|` 分隔多个写法。
    Literal {
        #[serde(default)]
        text: Vec<String>,
        #[serde(default)]
        optional: bool,
    },
    /// 任意一个词。
    Any {
        #[serde(default)]
        optional: bool,
    },
}

/// 编译后的一条双向规则。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticRule {
    pub name: String,
    pub predicate: String,
    pub entity_type: Option<String>,
    pub relation: bool,
    pub priority: i32,
    pub atoms: Vec<PatternAtom>,
    pub realize: Option<String>,
    pub keywords: Vec<String>,
}

/// 编译错误：坏规则不会让整个规则包失效，但必须显式可见。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleCompileError {
    pub rule: String,
    pub message: String,
}

impl RuleCompileError {
    pub fn render(&self) -> String {
        if self.rule.is_empty() {
            self.message.clone()
        } else {
            format!("{}：{}", self.rule, self.message)
        }
    }
}

/// 一次规则命中。`fact` 是规范事实，`start`/`end` 是 token 下标区间。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleMatch {
    pub rule: String,
    pub predicate: String,
    pub entity_type: Option<String>,
    pub relation: bool,
    pub priority: i32,
    pub fact: FactRef,
    pub start: usize,
    pub end: usize,
    pub bindings: BTreeMap<String, String>,
}

impl RuleMatch {
    pub fn to_knowledge_item(&self) -> KnowledgeItem {
        if self.relation {
            KnowledgeItem::Relationship {
                from_label: self.entity_type.clone().unwrap_or_default(),
                from_id: self.fact.subject.clone(),
                to_label: String::new(),
                to_id: self.fact.value.clone(),
                rel_type: self.fact.predicate.clone(),
                properties: HashMap::new(),
            }
        } else {
            let mut properties = HashMap::new();
            properties.insert("name".to_string(), self.fact.subject.clone());
            properties.insert(self.fact.predicate.clone(), self.fact.value.clone());
            KnowledgeItem::Entity {
                label: self.entity_type.clone().unwrap_or_else(|| "Entity".to_string()),
                properties,
            }
        }
    }
}

/// 一次反向实现的结果（句子 + 规则出处）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Realization {
    pub text: String,
    pub rule: String,
    pub predicate: String,
}

/// 抽取 → 生成 → 再抽取的往返回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundTrip {
    pub original: FactRef,
    pub text: String,
    pub rule: String,
    pub recovered: Option<FactRef>,
    pub consistent: bool,
}

/// 编译后的规则包（纯数据、可克隆、可跨会话共享）。
#[derive(Debug, Clone, Default)]
pub struct RulePack {
    rules: Vec<SemanticRule>,
    errors: Vec<RuleCompileError>,
}

const SUBJECT_SLOTS: [&str; 3] = ["s", "subject", "name"];
const VALUE_SLOTS: [&str; 3] = ["o", "object", "value"];

impl RulePack {
    /// 从系统配置编译；即使 `enabled = false` 也会编译，方便预览与测试。
    pub fn from_config(config: &SystemConfig) -> Self {
        Self::compile(config.semantic_grammar.rules.clone())
    }

    /// 编译一组规则。坏规则进入 `errors`，好规则照常可用。
    pub fn compile(configs: Vec<SemanticRuleConfig>) -> Self {
        let mut rules = Vec::new();
        let mut errors = Vec::new();
        for config in configs {
            match compile_rule(&config) {
                Ok(rule) => rules.push(rule),
                Err(message) => errors.push(RuleCompileError {
                    rule: config.name,
                    message,
                }),
            }
        }
        rules.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.name.cmp(&right.name))
        });
        Self { rules, errors }
    }

    pub fn rules(&self) -> &[SemanticRule] {
        &self.rules
    }

    pub fn errors(&self) -> &[RuleCompileError] {
        &self.errors
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 正向：从已分词输入抽取事实。同一事实由多条规则命中时保留优先级最高者。
    pub fn extract_tagged(&self, words: &[TaggedWord]) -> Vec<RuleMatch> {
        let joined: String = words.iter().map(|word| word.text.as_str()).collect();
        let mut best: BTreeMap<String, RuleMatch> = BTreeMap::new();
        for rule in &self.rules {
            if !rule.keywords.is_empty()
                && !rule
                    .keywords
                    .iter()
                    .any(|keyword| joined.contains(keyword.as_str()))
            {
                continue;
            }
            for start in 0..words.len() {
                let mut bindings = BTreeMap::new();
                if let Some((end, captures)) = match_atoms(&rule.atoms, words, 0, start, &mut bindings)
                {
                    let fact = match build_fact(rule, &captures) {
                        Some(fact) => fact,
                        None => continue,
                    };
                    let candidate = RuleMatch {
                        rule: rule.name.clone(),
                        predicate: rule.predicate.clone(),
                        entity_type: rule.entity_type.clone(),
                        relation: rule.relation,
                        priority: rule.priority,
                        fact,
                        start,
                        end,
                        bindings: captures,
                    };
                    let key = candidate.fact.key();
                    match best.get(&key) {
                        Some(existing)
                            if existing.priority > candidate.priority
                                || (existing.priority == candidate.priority
                                    && (existing.start, &existing.rule)
                                        <= (candidate.start, &candidate.rule)) => {}
                        _ => {
                            best.insert(key, candidate);
                        }
                    }
                }
            }
        }
        let mut matches: Vec<RuleMatch> = best.into_values().collect();
        matches.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.priority.cmp(&left.priority))
                .then_with(|| left.rule.cmp(&right.rule))
        });
        matches
    }

    /// 正向：先按配置分词，再抽取。
    pub fn extract_text(&self, text: &str, nlp: &crate::NlpConfig) -> Vec<RuleMatch> {
        let words = preprocess(text, nlp);
        self.extract_tagged(&words)
    }

    /// 正向：直接产出 `KnowledgeItem`，供 `KnowledgeExtractor` 接入。
    pub fn extract_items(&self, words: &[TaggedWord]) -> Vec<KnowledgeItem> {
        self.extract_tagged(words)
            .iter()
            .map(RuleMatch::to_knowledge_item)
            .collect()
    }

    /// 反向：用同一规则包的 `realize` 模板生成句子。
    ///
    /// 只使用显式配置了模板的规则；没有模板时返回 `None`，调用方保留原有
    /// 生成路径，避免新模块悄悄改变旧输出。
    pub fn realize_values(&self, subject: &str, predicate: &str, values: &[String]) -> Option<Realization> {
        let joined = values.join("、");
        for rule in &self.rules {
            if rule.predicate != predicate {
                continue;
            }
            let template = rule.realize.as_ref()?;
            let text = template
                .replace("{s}", subject)
                .replace("{o}", &joined)
                .replace("{p}", predicate);
            if text.trim().is_empty() {
                return None;
            }
            return Some(Realization {
                text,
                rule: rule.name.clone(),
                predicate: predicate.to_string(),
            });
        }
        None
    }

    pub fn realize_fact(&self, fact: &FactRef) -> Option<Realization> {
        self.realize_values(
            &fact.subject,
            &fact.predicate,
            std::slice::from_ref(&fact.value),
        )
    }

    /// 往返检查：事实 → 句子 → 分词 → 事实。
    pub fn roundtrip(
        &self,
        fact: &FactRef,
        nlp: &crate::NlpConfig,
    ) -> Option<RoundTrip> {
        let realization = self.realize_fact(fact)?;
        let recovered = self
            .extract_text(&realization.text, nlp)
            .into_iter()
            .find(|hit| {
                hit.fact.subject == fact.subject
                    && hit.fact.predicate == fact.predicate
                    && hit.fact.value == fact.value
            })
            .map(|hit| hit.fact);
        Some(RoundTrip {
            original: fact.clone(),
            text: realization.text,
            rule: realization.rule,
            consistent: recovered.is_some(),
            recovered,
        })
    }

    /// 人读报告（确定性顺序）。
    pub fn render(&self) -> String {
        let mut lines = vec![format!(
            "双向语义规则包：{} 条规则，{} 个编译错误",
            self.rules.len(),
            self.errors.len()
        )];
        for rule in &self.rules {
            let forward = rule
                .atoms
                .iter()
                .map(render_atom)
                .collect::<Vec<_>>()
                .join(" ");
            let direction = if rule.relation { "关系" } else { "属性" };
            lines.push(format!(
                "- {} [{}] {}（{}）→ {}",
                rule.name,
                rule.predicate,
                direction,
                forward,
                rule.realize.as_deref().unwrap_or("（无反向模板）")
            ));
        }
        for error in &self.errors {
            lines.push(format!("- 编译错误 {}", error.render()));
        }
        lines.join("\n")
    }

    /// 让 `SymbolicGenerator` 复用同一份反向模板。
    pub fn apply_to_generator(&self, generator: &mut crate::symbolic_generator::SymbolicGenerator) {
        generator.attach_semantic_grammar(self.clone());
    }
}

fn render_atom(atom: &PatternAtom) -> String {
    match atom {
        PatternAtom::Capture { slot, pos, optional } => {
            let suffix = if *optional { "?" } else { "" };
            if pos.is_empty() {
                format!("${}{}", slot, suffix)
            } else {
                format!("${}:pos:{}{}", slot, pos.join("|"), suffix)
            }
        }
        PatternAtom::Pos { pos, optional } => {
            format!("pos:{}{}", pos.join("|"), if *optional { "?" } else { "" })
        }
        PatternAtom::Literal { text, optional } => {
            format!("{}{}", text.join("|"), if *optional { "?" } else { "" })
        }
        PatternAtom::Any { optional } => {
            if *optional {
                "*?".to_string()
            } else {
                "*".to_string()
            }
        }
    }
}

fn compile_rule(config: &SemanticRuleConfig) -> Result<SemanticRule, String> {
    if config.name.trim().is_empty() {
        return Err("规则名为空".to_string());
    }
    if config.predicate.trim().is_empty() {
        return Err("predicate 为空".to_string());
    }
    if config.pattern.is_empty() {
        return Err("pattern 为空：规则永不命中".to_string());
    }
    let mut atoms = Vec::with_capacity(config.pattern.len());
    for raw in &config.pattern {
        atoms.push(parse_atom(raw)?);
    }
    let has_subject = atoms.iter().any(|atom| match atom {
        PatternAtom::Capture { slot, .. } => SUBJECT_SLOTS.contains(&slot.as_str()),
        _ => false,
    });
    let has_value = atoms.iter().any(|atom| match atom {
        PatternAtom::Capture { slot, .. } => VALUE_SLOTS.contains(&slot.as_str()),
        _ => false,
    });
    if !has_subject {
        return Err("缺少主语捕获：请使用 $s / $subject / $name".to_string());
    }
    if !has_value {
        return Err("缺少取值捕获：请使用 $o / $object / $value".to_string());
    }
    let realize = config.realize.as_ref().map(|template| template.trim().to_string());
    if let Some(template) = &realize {
        if !template.contains("{s}") || !template.contains("{o}") {
            return Err("realize 必须同时包含 {s} 与 {o}".to_string());
        }
    }
    Ok(SemanticRule {
        name: config.name.trim().to_string(),
        predicate: config.predicate.trim().to_string(),
        entity_type: config
            .entity_type
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        relation: config.relation,
        priority: config.priority,
        atoms,
        realize,
        keywords: config
            .keywords
            .iter()
            .map(|keyword| keyword.trim().to_string())
            .filter(|keyword| !keyword.is_empty())
            .collect(),
    })
}

fn parse_atom(raw: &str) -> Result<PatternAtom, String> {
    let mut token = raw.trim();
    if token.is_empty() {
        return Err("pattern 含空 token".to_string());
    }
    let mut optional = false;
    if let Some(stripped) = token.strip_suffix('?') {
        optional = true;
        token = stripped.trim();
        if token.is_empty() {
            return Err("可选 token 缺少内容".to_string());
        }
    }
    if token == "*" {
        return Ok(PatternAtom::Any { optional });
    }
    if let Some(slot) = token.strip_prefix('$') {
        let (slot, rest) = match slot.split_once(':') {
            Some((slot, rest)) => (slot.trim(), rest.trim()),
            None => (slot.trim(), ""),
        };
        if slot.is_empty() {
            return Err("捕获缺少槽位名".to_string());
        }
        let pos = if rest.is_empty() {
            Vec::new()
        } else if let Some(pos) = rest.strip_prefix("pos:") {
            split_alternatives(pos)
        } else {
            // 兼容 `$s:nr` 简写。
            split_alternatives(rest)
        };
        if pos.is_empty() && !rest.is_empty() {
            return Err(format!("捕获 ${} 缺少词性", slot));
        }
        return Ok(PatternAtom::Capture {
            slot: slot.to_string(),
            pos,
            optional,
        });
    }
    if let Some(pos) = token.strip_prefix("pos:") {
        let pos = split_alternatives(pos);
        if pos.is_empty() {
            return Err("pos: 缺少词性".to_string());
        }
        return Ok(PatternAtom::Pos { pos, optional });
    }
    if let Some(text) = token.strip_prefix("lit:") {
        let text = split_alternatives(text);
        if text.is_empty() {
            return Err("lit: 缺少字面词".to_string());
        }
        return Ok(PatternAtom::Literal { text, optional });
    }
    Ok(PatternAtom::Literal {
        text: split_alternatives(token),
        optional,
    })
}

fn split_alternatives(raw: &str) -> Vec<String> {
    raw.split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn build_fact(rule: &SemanticRule, bindings: &BTreeMap<String, String>) -> Option<FactRef> {
    let subject = SUBJECT_SLOTS
        .iter()
        .find_map(|slot| bindings.get(*slot))
        .cloned()?;
    let value = VALUE_SLOTS
        .iter()
        .find_map(|slot| bindings.get(*slot))
        .cloned()?;
    if subject.trim().is_empty() || value.trim().is_empty() {
        return None;
    }
    Some(FactRef::new(
        subject.trim(),
        rule.predicate.trim(),
        value.trim(),
    ))
}

fn match_atoms(
    atoms: &[PatternAtom],
    words: &[TaggedWord],
    atom_index: usize,
    word_index: usize,
    bindings: &mut BTreeMap<String, String>,
) -> Option<(usize, BTreeMap<String, String>)> {
    if atom_index == atoms.len() {
        return Some((word_index, bindings.clone()));
    }
    let atom = &atoms[atom_index];
    if atom_is_optional(atom) {
        if let Some(result) = match_atoms_required(atoms, words, atom_index, word_index, bindings) {
            return Some(result);
        }
        return match_atoms(atoms, words, atom_index + 1, word_index, bindings);
    }
    match_atoms_required(atoms, words, atom_index, word_index, bindings)
}

fn match_atoms_required(
    atoms: &[PatternAtom],
    words: &[TaggedWord],
    atom_index: usize,
    word_index: usize,
    bindings: &BTreeMap<String, String>,
) -> Option<(usize, BTreeMap<String, String>)> {
    let atom = &atoms[atom_index];
    for (advance, capture) in atom_candidates(atom, words, word_index) {
        let mut next = bindings.clone();
        if let Some((slot, value)) = capture {
            if let Some(existing) = next.get(&slot) {
                if existing != &value {
                    continue;
                }
            } else {
                next.insert(slot, value);
            }
        }
        if let Some(result) = match_atoms(atoms, words, atom_index + 1, word_index + advance, &mut next)
        {
            return Some(result);
        }
    }
    None
}

fn atom_is_optional(atom: &PatternAtom) -> bool {
    match atom {
        PatternAtom::Capture { optional, .. }
        | PatternAtom::Pos { optional, .. }
        | PatternAtom::Literal { optional, .. }
        | PatternAtom::Any { optional } => *optional,
    }
}

fn atom_candidates(
    atom: &PatternAtom,
    words: &[TaggedWord],
    word_index: usize,
) -> Vec<(usize, Option<(String, String)>)> {
    if word_index >= words.len() {
        return Vec::new();
    }
    match atom {
        PatternAtom::Any { .. } => vec![(1, None)],
        PatternAtom::Pos { pos, .. } => {
            if pos_match(pos, &words[word_index].pos) {
                vec![(1, None)]
            } else {
                Vec::new()
            }
        }
        PatternAtom::Literal { text, .. } => {
            let mut out = Vec::new();
            let mut joined = String::new();
            for advance in 1..=4 {
                let index = word_index + advance - 1;
                if index >= words.len() {
                    break;
                }
                joined.push_str(&words[index].text);
                if text.iter().any(|candidate| candidate == &joined) {
                    out.push((advance, None));
                }
            }
            out
        }
        PatternAtom::Capture { slot, pos, .. } => {
            if pos_match(pos, &words[word_index].pos) {
                vec![(1, Some((slot.clone(), words[word_index].text.clone())))]
            } else {
                Vec::new()
            }
        }
    }
}

fn pos_match(expected: &[String], actual: &str) -> bool {
    expected.is_empty()
        || expected
            .iter()
            .any(|pos| pos == "*" || pos == actual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamePatterns;

    fn rule(name: &str, predicate: &str, pattern: &[&str], realize: &str, relation: bool) -> SemanticRuleConfig {
        SemanticRuleConfig {
            name: name.to_string(),
            predicate: predicate.to_string(),
            entity_type: Some("Person".to_string()),
            pattern: pattern.iter().map(|item| item.to_string()).collect(),
            realize: Some(realize.to_string()),
            relation,
            priority: 10,
            keywords: Vec::new(),
        }
    }

    fn nlp_config() -> crate::NlpConfig {
        let mut pos_dict = std::collections::HashMap::new();
        pos_dict.insert("小明".to_string(), "nr".to_string());
        pos_dict.insert("小刚".to_string(), "nr".to_string());
        pos_dict.insert("身高".to_string(), "n".to_string());
        pos_dict.insert("认识".to_string(), "v".to_string());
        crate::NlpConfig {
            import_keywords: Vec::new(),
            pos_dict,
            name_patterns: NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: vec!["小明".to_string(), "小刚".to_string()],
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn forward_extraction_and_reverse_realization_share_one_rule() {
        let pack = RulePack::compile(vec![rule(
            "身高",
            "height",
            &["$s:pos:nr", "身高", "$o:pos:m"],
            "{s}身高{o}",
            false,
        )]);
        let words = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
        ];
        let hits = pack.extract_tagged(&words);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].fact, FactRef::attribute("小明", "height", "226"));
        let sentence = pack.realize_fact(&hits[0].fact).unwrap().text;
        assert_eq!(sentence, "小明身高226");
        let again = pack.extract_text(&sentence, &nlp_config());
        assert_eq!(again[0].fact, hits[0].fact);
    }

    #[test]
    fn relation_rule_emits_relationship_and_roundtrips() {
        let pack = RulePack::compile(vec![rule(
            "认识",
            "认识",
            &["$s:pos:nr", "认识", "$o:pos:nr"],
            "{s}认识{o}",
            true,
        )]);
        let words = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "认识".into(), pos: "v".into() },
            TaggedWord { text: "小刚".into(), pos: "nr".into() },
        ];
        let hits = pack.extract_tagged(&words);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].relation);
        assert_eq!(hits[0].fact, FactRef::relation("小明", "认识", "小刚"));
        let item = hits[0].to_knowledge_item();
        assert!(matches!(item, KnowledgeItem::Relationship { ref from_id, ref to_id, .. }
            if from_id == "小明" && to_id == "小刚"));
        let trip = pack.roundtrip(&hits[0].fact, &nlp_config()).unwrap();
        assert!(trip.consistent, "{trip:?}");
    }

    #[test]
    fn optional_and_alternative_atoms_backtrack() {
        let pack = RulePack::compile(vec![rule(
            "身高可选量词",
            "height",
            &["$s:pos:nr", "身高", "pos:q?", "$o:pos:m"],
            "{s}身高{o}",
            false,
        )]);
        let no_measure = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
        ];
        let with_measure = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
            TaggedWord { text: "厘米".into(), pos: "q".into() },
        ];
        assert_eq!(
            pack.extract_tagged(&no_measure)[0].fact,
            FactRef::attribute("小明", "height", "226")
        );
        assert_eq!(
            pack.extract_tagged(&with_measure)[0].fact,
            FactRef::attribute("小明", "height", "226")
        );
    }

    #[test]
    fn multi_token_literals_match_without_pos_dict() {
        let pack = RulePack::compile(vec![rule(
            "多词字面量",
            "height",
            &["$s:pos:nr", "身高为", "$o:pos:m"],
            "{s}身高为{o}",
            false,
        )]);
        let words = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "为".into(), pos: "v".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
        ];
        let hits = pack.extract_tagged(&words);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].fact.value, "226");
    }

    #[test]
    fn higher_priority_rule_wins_and_duplicates_collapse() {
        let mut low = rule(
            "低优先级",
            "height",
            &["$s:pos:nr", "身高", "$o:pos:m"],
            "{s}身高{o}",
            false,
        );
        low.priority = 1;
        let high = rule(
            "高优先级",
            "height",
            &["$s:pos:nr", "身高", "$o:pos:m"],
            "{s}的身高是{o}",
            false,
        );
        let pack = RulePack::compile(vec![low, high]);
        let words = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
        ];
        let hits = pack.extract_tagged(&words);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule, "高优先级");
    }

    #[test]
    fn invalid_rules_are_reported_not_silently_dropped() {
        let bad = SemanticRuleConfig {
            name: "坏规则".to_string(),
            predicate: "p".to_string(),
            entity_type: None,
            pattern: vec!["$s:pos:nr".to_string()],
            realize: Some("{s}".to_string()),
            relation: false,
            priority: 0,
            keywords: Vec::new(),
        };
        let pack = RulePack::compile(vec![bad]);
        assert!(pack.is_empty());
        assert_eq!(pack.errors().len(), 1);
        assert!(pack.errors()[0].message.contains("取值捕获"));
    }

    #[test]
    fn keywords_guard_rule_without_hiding_unrelated_hits() {
        let mut guarded = rule(
            "受保护",
            "height",
            &["$s:pos:nr", "身高", "$o:pos:m"],
            "{s}身高{o}",
            false,
        );
        guarded.keywords = vec!["体重".to_string()];
        let pack = RulePack::compile(vec![guarded]);
        let words = vec![
            TaggedWord { text: "小明".into(), pos: "nr".into() },
            TaggedWord { text: "身高".into(), pos: "n".into() },
            TaggedWord { text: "226".into(), pos: "m".into() },
        ];
        assert!(pack.extract_tagged(&words).is_empty());
    }
}

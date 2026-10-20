// 关联规则挖掘（纯计数，无神经网络、无权重训练）
//
// 灵感：Agrawal 的关联规则 / 购物篮分析。把每个实体看作一个"购物篮"，
// 篮中物品是它的类型标签、关系谓词与属性键（如「类型：人物」「关系：出生地」）：
//   • 支持度 support(A⇒B) = P(A∧B)        —— 这组特征在全图中出现得多不多；
//   • 置信度 confidence   = P(B|A)         —— 有 A 的实体里有多大比例也有 B；
//   • 提升度 lift         = P(B|A)/P(B)    —— 相比"随机共现"提升了多少倍；
//   • 杠杆率 leverage     = P(A∧B)-P(A)P(B)—— 绝对共现增量。
// 挖出的规则可直接转成 Horn 子句（见 to_clauses），交给归结引擎做前向推导，
// 用于实体补全建议（"赵六可能也有国籍"）与生成式续写的联想。

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::resolution_engine::{Atom, Clause, Term};

pub const TYPE_PREFIX: &str = "类型：";
pub const RELATION_PREFIX: &str = "关系：";
pub const ATTRIBUTE_PREFIX: &str = "属性：";

pub fn type_item(etype: &str) -> String {
    format!("{}{}", TYPE_PREFIX, etype)
}

pub fn relation_item(predicate: &str) -> String {
    format!("{}{}", RELATION_PREFIX, predicate)
}

pub fn attribute_item(key: &str) -> String {
    format!("{}{}", ATTRIBUTE_PREFIX, key)
}

/// 去掉分类前缀，展示为人类可读名称
pub fn display_item(item: &str) -> &str {
    for prefix in [TYPE_PREFIX, RELATION_PREFIX, ATTRIBUTE_PREFIX] {
        if let Some(rest) = item.strip_prefix(prefix) {
            return rest;
        }
    }
    item
}

#[derive(Debug, Clone)]
pub struct AssociationRule {
    pub antecedent: Vec<String>,
    pub consequent: String,
    pub support: usize,
    pub antecedent_support: usize,
    pub consequent_support: usize,
    pub total: usize,
    pub confidence: f64,
    pub lift: f64,
    pub leverage: f64,
}

impl AssociationRule {
    pub fn item_list_text(&self, items: &[String]) -> String {
        items
            .iter()
            .map(|item| display_item(item).to_string())
            .collect::<Vec<_>>()
            .join("、")
    }

    pub fn antecedent_text(&self) -> String {
        self.item_list_text(&self.antecedent)
    }

    pub fn consequent_text(&self) -> String {
        display_item(&self.consequent).to_string()
    }

    /// 形如 `类型：人物 ∧ 关系：出生地 ⇒ 关系：国籍`
    pub fn rule_text(&self) -> String {
        format!("{} ⇒ {}", self.antecedent.join(" ∧ "), self.consequent)
    }

    /// 形如 `支持度 3/6，置信度 0.75，提升度 2.50，杠杆率 0.17`
    pub fn metrics_text(&self) -> String {
        format!(
            "支持度 {}/{}，置信度 {:.2}，提升度 {:.2}，杠杆率 {:.2}",
            self.support, self.total, self.confidence, self.lift, self.leverage
        )
    }
}

#[derive(Clone)]
pub struct AssociationMiner {
    // 实体 → 物品篮（集合，天然去重）
    baskets: HashMap<String, HashSet<String>>,
    item_counts: HashMap<String, usize>,
    rules: Vec<AssociationRule>,
    min_support: f64,
    min_confidence: f64,
    min_lift: f64,
    // 二前件规则的组合上限（只在高频物品间枚举，防止组合爆炸）
    max_pair_items: usize,
}

impl AssociationMiner {
    pub fn new(min_support: f64, min_confidence: f64, min_lift: f64) -> Self {
        Self {
            baskets: HashMap::new(),
            item_counts: HashMap::new(),
            rules: Vec::new(),
            min_support: min_support.clamp(0.0, 1.0),
            min_confidence: min_confidence.clamp(0.0, 1.0),
            min_lift: min_lift.max(0.0),
            max_pair_items: 10,
        }
    }

    pub fn basket_count(&self) -> usize {
        self.baskets.len()
    }

    pub fn rules(&self) -> &[AssociationRule] {
        &self.rules
    }

    pub fn basket(&self, entity: &str) -> Option<&HashSet<String>> {
        self.baskets.get(entity)
    }

    pub fn item_count(&self, item: &str) -> usize {
        self.item_counts.get(item).copied().unwrap_or(0)
    }

    /// 不同物品的种类数（报告用）
    pub fn item_kinds(&self) -> usize {
        self.item_counts.len()
    }

    fn insert_item(&mut self, entity: &str, item: String) -> bool {
        if entity.trim().is_empty() || item.trim().is_empty() {
            return false;
        }
        // 已存在的篮子直接复用
        let basket = match self.baskets.get_mut(entity) {
            Some(basket) => basket,
            None => self.baskets.entry(entity.to_string()).or_default(),
        };
        if !basket.insert(item.clone()) {
            return false;
        }
        *self.item_counts.entry(item).or_insert(0) += 1;
        true
    }

    /// 观察一个带类型的实体：类型标签 + 属性键入篮
    pub fn observe_entity(&mut self, name: &str, etype: &str, property_keys: &[String]) -> bool {
        self.observe_entity_keys(name, etype, property_keys.iter().map(|k| k.as_str()))
    }

    /// 与 `observe_entity` 等价，但直接以属性表的键为输入。
    pub fn observe_entity_map(
        &mut self,
        name: &str,
        etype: &str,
        property_keys: &HashMap<String, String>,
    ) -> bool {
        self.observe_entity_keys(name, etype, property_keys.keys().map(|k| k.as_str()))
    }

    fn observe_entity_keys<'a, I>(&mut self, name: &str, etype: &str, property_keys: I) -> bool
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut changed = false;
        if !etype.trim().is_empty() && etype != "Entity" {
            changed |= self.insert_item(name, type_item(etype));
        } else {
            self.baskets.entry(name.to_string()).or_default();
        }
        for key in property_keys {
            changed |= self.insert_item(name, attribute_item(key));
        }
        changed
    }

    /// 观察一条关系三元组：谓词入主体篮
    pub fn observe_triple(&mut self, subject: &str, predicate: &str) -> bool {
        self.insert_item(subject, relation_item(predicate))
    }

    /// 重新挖掘规则（幂等；支持度阈值按"篮子占比"换算成绝对计数）
    pub fn mine(&mut self) {
        self.rules.clear();
        let total = self.baskets.len();
        if total == 0 {
            return;
        }
        let min_count = ((self.min_support * total as f64).ceil() as usize).max(1);
        let mut items: Vec<(String, usize)> = self
            .item_counts
            .iter()
            .map(|(item, count)| (item.clone(), *count))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // 倒排索引：物品 → 包含该物品的实体集合
        let mut item_entities: HashMap<&str, HashSet<&str>> = HashMap::with_capacity(items.len());
        for (entity, basket) in &self.baskets {
            for item in basket {
                item_entities
                    .entry(item.as_str())
                    .or_default()
                    .insert(entity.as_str());
            }
        }

        // 单前件规则 A ⇒ B（物品按出现次数降序，低于阈值的尾部直接截断）
        for (index, (a, a_count)) in items.iter().enumerate() {
            if *a_count < min_count {
                // 已按出现次数降序排列，后续项只会更少
                break;
            }
            for (other, (b, b_count)) in items.iter().enumerate() {
                if index == other {
                    continue;
                }
                if *b_count < min_count {
                    break;
                }
                if let Some(rule) = self.build_rule(
                    std::slice::from_ref(a),
                    a,
                    b,
                    *b_count,
                    *a_count,
                    total,
                    min_count,
                    &item_entities,
                ) {
                    self.rules.push(rule);
                }
            }
        }

        // 二前件规则 A ∧ C ⇒ B（只在高频物品间枚举）
        let top: Vec<(String, usize)> = items
            .iter()
            .filter(|(_, count)| *count >= min_count)
            .take(self.max_pair_items)
            .cloned()
            .collect();
        for i in 0..top.len() {
            for j in (i + 1)..top.len() {
                let first = &top[i].0;
                let second = &top[j].0;
                let pair_count = match (
                    item_entities.get(first.as_str()),
                    item_entities.get(second.as_str()),
                ) {
                    (Some(x), Some(y)) => intersection_count(x, y),
                    _ => 0,
                };
                if pair_count < min_count {
                    continue;
                }
                let antecedent = vec![first.clone(), second.clone()];
                let antecedent_text = if cfg!(debug_assertions) {
                    antecedent.join(" ∧ ")
                } else {
                    String::new()
                };
                for (b, b_count) in top.iter() {
                    if b == first || b == second {
                        continue;
                    }
                    if let Some(rule) = self.build_rule(
                        &antecedent,
                        &antecedent_text,
                        b,
                        *b_count,
                        pair_count,
                        total,
                        min_count,
                        &item_entities,
                    ) {
                        self.rules.push(rule);
                    }
                }
            }
        }

        self.rules.sort_by(|a, b| {
            b.lift
                .partial_cmp(&a.lift)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| a.consequent.cmp(&b.consequent))
        });
        self.rules
            .dedup_by(|a, b| a.antecedent == b.antecedent && a.consequent == b.consequent);
        // 规则数量上限：保持报告可读、后续推导可控
        self.rules.truncate(200);
    }

    fn build_rule(
        &self,
        antecedent: &[String],
        antecedent_text: &str,
        consequent: &str,
        consequent_count: usize,
        antecedent_count: usize,
        total: usize,
        min_count: usize,
        item_entities: &HashMap<&str, HashSet<&str>>,
    ) -> Option<AssociationRule> {
        if consequent_count < min_count {
            return None;
        }
        // 支持度：整个前件与结论共现的实体数（前件计数由调用方提供）
        let support = if antecedent.len() == 1 {
            match (
                item_entities.get(antecedent[0].as_str()),
                item_entities.get(consequent),
            ) {
                (Some(x), Some(y)) => intersection_count(x, y),
                _ => 0,
            }
        } else {
            match (
                item_entities.get(antecedent[0].as_str()),
                item_entities.get(antecedent[1].as_str()),
                item_entities.get(consequent),
            ) {
                (Some(x), Some(y), Some(z)) => triple_intersection_count(x, y, z),
                _ => 0,
            }
        };
        if support < min_count || antecedent_count < min_count || antecedent_count == 0 {
            return None;
        }
        // 保证前提集合非空且前件文本与项一致（防误用）
        debug_assert!(antecedent.iter().any(|item| antecedent_text.contains(item)));
        let confidence = support as f64 / antecedent_count as f64;
        if confidence < self.min_confidence {
            return None;
        }
        let p_b = consequent_count as f64 / total as f64;
        if p_b <= 0.0 {
            return None;
        }
        let lift = confidence / p_b;
        if lift < self.min_lift {
            return None;
        }
        let p_a = antecedent_count as f64 / total as f64;
        let leverage = support as f64 / total as f64 - p_a * p_b;
        Some(AssociationRule {
            antecedent: antecedent.to_vec(),
            consequent: consequent.to_string(),
            support,
            antecedent_support: antecedent_count,
            consequent_support: consequent_count,
            total,
            confidence,
            lift,
            leverage,
        })
    }

    /// 与某物品相关的规则（前件包含它或结论是它）
    pub fn rules_for_item(&self, item: &str, limit: usize) -> Vec<&AssociationRule> {
        self.rules
            .iter()
            .filter(|rule| {
                rule.antecedent.iter().any(|candidate| candidate == item) || rule.consequent == item
            })
            .take(limit)
            .collect()
    }

    /// 实体补全建议：前件已在篮中、结论尚缺的规则（按提升度×置信度排序）
    pub fn suggestions_for(&self, entity: &str, limit: usize) -> Vec<&AssociationRule> {
        let Some(basket) = self.baskets.get(entity) else {
            return Vec::new();
        };
        let mut suggestions: Vec<&AssociationRule> = self
            .rules
            .iter()
            .filter(|rule| {
                rule.antecedent.iter().all(|item| basket.contains(item))
                    && !basket.contains(&rule.consequent)
            })
            .collect();
        // 同一结论只保留最强规则
        suggestions.sort_by(|a, b| {
            (b.lift * b.confidence)
                .partial_cmp(&(a.lift * a.confidence))
                .unwrap_or(Ordering::Equal)
        });
        let mut seen: HashSet<&str> = HashSet::new();
        suggestions.retain(|rule| seen.insert(rule.consequent.as_str()));
        suggestions.truncate(limit);
        suggestions
    }

    pub fn explain(&self, rule: &AssociationRule) -> String {
        let antecedent = rule.antecedent_text();
        let consequent = rule.consequent_text();
        format!(
            "在该知识图谱的 {} 个实体中，「{}」与「{}」共现 {} 次；已知「{}」的实体有 {:.0}% 也具备「{}」（提升度 {:.2}，说明不是随机共现）",
            rule.total,
            antecedent,
            consequent,
            rule.support,
            antecedent,
            rule.confidence * 100.0,
            consequent,
            rule.lift
        )
    }

    /// 转 Horn 子句：物品映射为 `has(E, 谓词)` / `type(E, 类型)`，同一实体变量 E
    pub fn to_clauses(&self, has_predicate: &str, type_predicate: &str) -> Vec<Clause> {
        self.rules
            .iter()
            .map(|rule| {
                let head = item_atom(&rule.consequent, has_predicate, type_predicate);
                let body = rule
                    .antecedent
                    .iter()
                    .map(|item| item_atom(item, has_predicate, type_predicate))
                    .collect();
                let source = format!(
                    "关联规则 {}（支持 {}/{}，置信度 {:.2}，提升度 {:.2}）",
                    rule.rule_text(),
                    rule.support,
                    rule.total,
                    rule.confidence,
                    rule.lift
                );
                Clause::new(head, body, rule.confidence, &source)
            })
            .collect()
    }
}

fn intersection_count(a: &HashSet<&str>, b: &HashSet<&str>) -> usize {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().filter(|item| large.contains(*item)).count()
}

fn triple_intersection_count(a: &HashSet<&str>, b: &HashSet<&str>, c: &HashSet<&str>) -> usize {
    let sets = [a, b, c];
    let smallest = sets
        .iter()
        .min_by_key(|set| set.len())
        .expect("non-empty sets array");
    smallest
        .iter()
        .filter(|item| sets.iter().all(|set| set.contains(*item)))
        .count()
}

fn item_atom(item: &str, has_predicate: &str, type_predicate: &str) -> Atom {
    if let Some(etype) = item.strip_prefix(TYPE_PREFIX) {
        return Atom::pattern(type_predicate, Term::var("E"), Term::constant(etype));
    }
    let name = display_item(item);
    Atom::pattern(has_predicate, Term::var("E"), Term::constant(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 6 个实体：3 个"有出生地且有国籍"，1 个"只有出生地"，另 2 个无关
    fn sample_miner() -> AssociationMiner {
        let mut miner = AssociationMiner::new(0.3, 0.6, 1.0);
        for (name, has_nationality) in [
            ("张三", true),
            ("李四", true),
            ("王五", true),
            ("赵六", false),
        ] {
            let keys = if has_nationality {
                vec!["出生地".to_string(), "国籍".to_string()]
            } else {
                vec!["出生地".to_string()]
            };
            miner.observe_entity(name, "人物", &keys);
        }
        miner.observe_entity("西湖", "地点", &[]);
        miner.observe_entity("长江", "地点", &[]);
        miner
    }

    #[test]
    fn test_support_confidence_lift_leverage() {
        let mut miner = sample_miner();
        miner.mine();
        let rule = miner
            .rules()
            .iter()
            .find(|rule| {
                rule.antecedent == vec![relation_or_attr("出生地")]
                    && rule.consequent == relation_or_attr("国籍")
            })
            .expect("应挖出 出生地⇒国籍 规则");
        assert_eq!(rule.total, 6);
        assert_eq!(rule.antecedent_support, 4);
        assert_eq!(rule.support, 3);
        assert_eq!(rule.consequent_support, 3);
        assert!((rule.confidence - 0.75).abs() < 1e-9);
        assert!((rule.lift - 1.5).abs() < 1e-9);
        assert!((rule.leverage - (0.5 - (4.0 / 6.0) * 0.5)).abs() < 1e-9);
    }

    fn relation_or_attr(name: &str) -> String {
        attribute_item(name)
    }

    #[test]
    fn test_min_lift_filters_independent_items() {
        let mut miner = AssociationMiner::new(0.2, 0.0, 1.0);
        // A 与 B 负相关：P(A)=4/6，P(B)=4/6，P(B|A)=0.5 ⇒ 提升度 0.75
        miner.observe_triple("e1", "rA");
        miner.observe_triple("e2", "rA");
        miner.observe_triple("e2", "rB");
        miner.observe_triple("e3", "rA");
        miner.observe_triple("e4", "rA");
        miner.observe_triple("e4", "rB");
        miner.observe_triple("e5", "rB");
        miner.observe_triple("e6", "rB");
        miner.mine();
        assert!(
            miner
                .rules()
                .iter()
                .all(|rule| !(rule.antecedent[0] == relation_item("rA")
                    && rule.consequent == relation_item("rB"))),
            "提升度低于阈值的规则不应出现：{:?}",
            miner
                .rules()
                .iter()
                .map(|r| r.rule_text())
                .collect::<Vec<_>>()
        );
        // 阈值放宽后同一规则应被挖出，说明过滤发生在提升度而非支持度
        let mut relaxed = AssociationMiner::new(0.2, 0.0, 0.5);
        relaxed.observe_triple("e1", "rA");
        relaxed.observe_triple("e2", "rA");
        relaxed.observe_triple("e2", "rB");
        relaxed.observe_triple("e3", "rA");
        relaxed.observe_triple("e4", "rA");
        relaxed.observe_triple("e4", "rB");
        relaxed.observe_triple("e5", "rB");
        relaxed.observe_triple("e6", "rB");
        relaxed.mine();
        assert!(relaxed.rules().iter().any(|rule| {
            rule.antecedent[0] == relation_item("rA") && rule.consequent == relation_item("rB")
        }));
    }

    #[test]
    fn test_pair_antecedent_rule() {
        let mut miner = AssociationMiner::new(0.2, 1.0, 1.0);
        for name in ["a", "b", "c"] {
            miner.observe_entity(name, "人物", &[]);
            miner.observe_triple(name, "出生地");
            miner.observe_triple(name, "毕业院校");
            miner.observe_triple(name, "职业");
        }
        miner.mine();
        assert!(
            miner.rules().iter().any(|rule| {
                rule.antecedent.len() == 2
                    && rule.consequent == relation_item("职业")
                    && rule.confidence >= 0.99
            }),
            "应挖出二前件规则：{:?}",
            miner
                .rules()
                .iter()
                .map(|r| r.rule_text())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_suggestions_exclude_known_items() {
        let mut miner = sample_miner();
        miner.mine();
        let suggestions = miner.suggestions_for("赵六", 5);
        assert!(
            suggestions
                .iter()
                .any(|rule| rule.consequent == attribute_item("国籍")),
            "赵六缺国籍应被建议"
        );
        assert!(miner.suggestions_for("不存在的人", 5).is_empty());
    }

    #[test]
    fn test_explain_contains_metrics() {
        let mut miner = sample_miner();
        miner.mine();
        let rule = miner
            .rules()
            .iter()
            .find(|rule| rule.consequent == attribute_item("国籍"))
            .expect("应有国籍关联规则");
        let text = miner.explain(rule);
        assert!(text.contains("共现"));
        assert!(text.contains("提升度"));
        assert!(text.contains("出生地"));
    }

    #[test]
    fn test_to_clauses_maps_items_and_entity_var() {
        let mut miner = sample_miner();
        miner.mine();
        let clauses = miner.to_clauses("__has__", "__type__");
        assert!(!clauses.is_empty());
        let clause = clauses
            .iter()
            .find(|clause| clause.source.contains("出生地") && clause.source.contains("国籍"))
            .expect("应有 出生地⇒国籍 相关子句");
        assert_eq!(&*clause.head.predicate, "__has__");
        assert_eq!(clause.body.len(), 1);
        assert_eq!(&*clause.body[0].predicate, "__has__");
        // 同一实体变量 E，前提与结论可从同一实体实例化
        assert!(clause.head.terms.iter().any(|term| term.is_var()));
    }

    #[test]
    fn test_empty_miner_is_safe() {
        let mut miner = AssociationMiner::new(0.3, 0.5, 1.0);
        miner.mine();
        assert!(miner.rules().is_empty());
        assert!(miner.suggestions_for("任意", 3).is_empty());
        assert!(miner.to_clauses("__has__", "__type__").is_empty());
        assert_eq!(miner.basket_count(), 0);
    }

    #[test]
    fn test_observe_is_idempotent() {
        let mut miner = AssociationMiner::new(0.3, 0.5, 1.0);
        assert!(miner.observe_triple("张三", "师傅"));
        assert!(!miner.observe_triple("张三", "师傅"));
        assert_eq!(miner.item_count(&relation_item("师傅")), 1);
        assert!(miner.observe_entity("张三", "人物", &["年龄".to_string()]));
        assert!(!miner.observe_entity("张三", "人物", &["年龄".to_string()]));
        assert_eq!(miner.basket("张三").map(HashSet::len), Some(3));
    }
}

//! POCL：词性—本体耦合层（POS-Ontology Coupling Layer）。
//!
//! 把扁平的词性字符串升级为一个显式的词法本体：每个词性节点带父类边、
//! 语义排序（sort）与特征（指称/可数/谓词/修饰/数量/量词/代词/功能词）。
//! 模式里的 `n` 因此可以匹配 `nr/ns/nt/nz/vn/...`，`v` 可以匹配 `vn/vd`，
//! 而所有这些泛化都来自可枚举、可审计的闭包，而不是隐藏的模糊匹配。
//!
//! 本模块只做投影，不替换 `type_system` / `ontology_reasoning`：
//! 领域本体的真相源仍是配置与图谱；这里给出的是「词性允许哪些排序」以及
//! 「这个提及可能对应哪些默认类型」的软证据。

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::nlp::{AnnotatedWord, TagSource};

/// 词性本体格式版本。
pub const POS_ONTOLOGY_VERSION: u32 = 1;

const MAX_CLOSURE_HOPS: usize = 32;

/// 语义排序：词性到领域类型的粗粒度投影。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sort {
    Person,
    Organization,
    Place,
    Event,
    Object,
    Attribute,
    Quantity,
    Time,
    Relation,
    DiscourseReferent,
}

impl Sort {
    pub fn code(self) -> &'static str {
        match self {
            Sort::Person => "person",
            Sort::Organization => "organization",
            Sort::Place => "place",
            Sort::Event => "event",
            Sort::Object => "object",
            Sort::Attribute => "attribute",
            Sort::Quantity => "quantity",
            Sort::Time => "time",
            Sort::Relation => "relation",
            Sort::DiscourseReferent => "discourse_referent",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Sort::Person => "人物",
            Sort::Organization => "组织",
            Sort::Place => "地点",
            Sort::Event => "事件",
            Sort::Object => "对象",
            Sort::Attribute => "属性",
            Sort::Quantity => "数量",
            Sort::Time => "时间",
            Sort::Relation => "关系",
            Sort::DiscourseReferent => "指称",
        }
    }

    /// 默认领域类型标签；`DiscourseReferent` 不是可落图的实体类型。
    pub fn ontology_type(self) -> Option<&'static str> {
        match self {
            Sort::Person => Some("人物"),
            Sort::Organization => Some("组织"),
            Sort::Place => Some("地点"),
            Sort::Event => Some("事件"),
            Sort::Object => Some("实体"),
            Sort::Attribute => Some("属性"),
            Sort::Quantity => Some("数量"),
            Sort::Time => Some("时间"),
            Sort::Relation => Some("关系"),
            Sort::DiscourseReferent => None,
        }
    }
}

/// 词性节点携带的语法—语义特征。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosFeatures {
    pub referential: bool,
    pub countable: bool,
    pub predicate: bool,
    pub modifier: bool,
    pub numeral: bool,
    pub classifier: bool,
    pub pronoun: bool,
    pub function_word: bool,
}

impl PosFeatures {
    fn merge(&mut self, other: PosFeatures) {
        self.referential |= other.referential;
        self.countable |= other.countable;
        self.predicate |= other.predicate;
        self.modifier |= other.modifier;
        self.numeral |= other.numeral;
        self.classifier |= other.classifier;
        self.pronoun |= other.pronoun;
        self.function_word |= other.function_word;
    }
}

/// 一个词性节点：父类边、直接排序与直接特征。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PosNode {
    pub tag: String,
    pub parents: Vec<String>,
    pub sorts: Vec<Sort>,
    pub features: PosFeatures,
}

/// 闭包后的词性画像。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosProfile {
    pub tag: String,
    pub known: bool,
    pub ancestors: Vec<String>,
    pub sorts: Vec<Sort>,
    pub features: PosFeatures,
}

impl PosProfile {
    pub fn ontology_types(&self) -> Vec<String> {
        self.sorts
            .iter()
            .filter_map(|sort| sort.ontology_type())
            .map(str::to_string)
            .collect()
    }

    pub fn has_sort(&self, sort: Sort) -> bool {
        self.sorts.contains(&sort)
    }

    pub fn is_nominal(&self) -> bool {
        self.features.referential && !self.features.pronoun
    }

    pub fn is_verbal(&self) -> bool {
        self.features.predicate
    }

    pub fn is_countable(&self) -> bool {
        self.features.countable
    }
}

/// 带来源证据的类型化提及。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedMention {
    pub surface: String,
    pub pos: String,
    pub source: TagSource,
    pub profile: PosProfile,
    pub ontology_types: Vec<String>,
}

impl TypedMention {
    pub fn from_annotated(word: &AnnotatedWord, ontology: &PosOntology) -> Self {
        let profile = ontology.profile(&word.pos);
        let ontology_types = profile.ontology_types();
        Self {
            surface: word.text.clone(),
            pos: word.pos.clone(),
            source: word.source,
            profile,
            ontology_types,
        }
    }

    pub fn from_tagged(
        surface: &str,
        pos: &str,
        source: TagSource,
        ontology: &PosOntology,
    ) -> Self {
        let profile = ontology.profile(pos);
        let ontology_types = profile.ontology_types();
        Self {
            surface: surface.to_string(),
            pos: pos.to_string(),
            source,
            profile,
            ontology_types,
        }
    }

    pub fn is_nominal(&self) -> bool {
        self.profile.is_nominal()
    }
}

/// 显式词性本体：节点 + 父类边。
#[derive(Debug, Clone, Default)]
pub struct PosOntology {
    nodes: BTreeMap<String, PosNode>,
}

fn normalize_tag(tag: &str) -> String {
    tag.trim().to_ascii_lowercase()
}

fn features(
    referential: bool,
    countable: bool,
    predicate: bool,
    modifier: bool,
) -> PosFeatures {
    PosFeatures {
        referential,
        countable,
        predicate,
        modifier,
        ..PosFeatures::default()
    }
}

fn node(tag: &str, parents: &[&str], sorts: &[Sort], features: PosFeatures) -> PosNode {
    PosNode {
        tag: tag.to_string(),
        parents: parents.iter().map(|parent| parent.to_string()).collect(),
        sorts: sorts.to_vec(),
        features,
    }
}

impl PosOntology {
    pub fn new() -> Self {
        Self::default()
    }

    /// 默认词性本体：结巴/配置使用的常见标签及其闭包。
    pub fn standard() -> Self {
        let mut ontology = Self::new();
        let noun = features(true, true, false, false);
        let verb = features(false, false, true, false);
        let adjective = features(false, false, false, true);
        let function = PosFeatures {
            function_word: true,
            ..PosFeatures::default()
        };

        ontology.declare(node("n", &[], &[], noun));
        ontology.declare(node("nr", &["n"], &[Sort::Person], noun));
        ontology.declare(node("nrfg", &["nr"], &[], noun));
        ontology.declare(node("nrt", &["nr"], &[], noun));
        ontology.declare(node("ns", &["n"], &[Sort::Place], noun));
        ontology.declare(node("nt", &["n"], &[Sort::Organization], noun));
        ontology.declare(node("nz", &["n"], &[], noun));
        ontology.declare(node("ng", &["n"], &[], noun));
        ontology.declare(node("j", &["n"], &[], noun));
        ontology.declare(node("s", &["n"], &[Sort::Place], noun));
        ontology.declare(node("t", &["n"], &[Sort::Time], noun));
        ontology.declare(node(
            "vn",
            &["n", "v"],
            &[Sort::Object],
            PosFeatures {
                referential: true,
                predicate: true,
                ..PosFeatures::default()
            },
        ));

        ontology.declare(node("v", &[], &[], verb));
        ontology.declare(node(
            "vd",
            &["v"],
            &[],
            PosFeatures {
                predicate: true,
                modifier: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node("vg", &["v"], &[], verb));

        ontology.declare(node("a", &[], &[Sort::Attribute], adjective));
        ontology.declare(node("ad", &["a"], &[], adjective));
        ontology.declare(node("ag", &["a"], &[], adjective));
        ontology.declare(node(
            "an",
            &["a", "n"],
            &[Sort::Attribute],
            PosFeatures {
                referential: true,
                countable: true,
                modifier: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node("b", &[], &[], adjective));
        ontology.declare(node("z", &["a"], &[], adjective));

        ontology.declare(node(
            "m",
            &[],
            &[Sort::Quantity],
            PosFeatures {
                numeral: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node("mg", &["m"], &[], PosFeatures::default()));
        ontology.declare(node(
            "mq",
            &["m", "q"],
            &[Sort::Quantity],
            PosFeatures {
                numeral: true,
                classifier: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node(
            "q",
            &[],
            &[Sort::Quantity],
            PosFeatures {
                classifier: true,
                ..PosFeatures::default()
            },
        ));

        ontology.declare(node(
            "r",
            &[],
            &[Sort::DiscourseReferent],
            PosFeatures {
                referential: true,
                pronoun: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node("rr", &["r"], &[], PosFeatures::default()));
        ontology.declare(node("rz", &["r"], &[], PosFeatures::default()));
        ontology.declare(node("ry", &["r"], &[], PosFeatures::default()));

        ontology.declare(node("d", &[], &[], adjective));
        ontology.declare(node(
            "f",
            &[],
            &[Sort::Place],
            PosFeatures {
                referential: true,
                modifier: true,
                ..PosFeatures::default()
            },
        ));
        ontology.declare(node("p", &[], &[], function));
        ontology.declare(node("c", &[], &[], function));
        ontology.declare(node("u", &[], &[], function));
        ontology.declare(node("y", &[], &[], function));
        ontology.declare(node("e", &[], &[], function));
        ontology.declare(node("o", &[], &[], function));
        ontology.declare(node("w", &[], &[], function));
        ontology.declare(node("x", &[], &[], function));
        ontology.declare(node("unknown", &[], &[], PosFeatures::default()));
        ontology
    }

    pub fn declare(&mut self, node: PosNode) -> bool {
        let tag = normalize_tag(&node.tag);
        if tag.is_empty() {
            return false;
        }
        let parents: Vec<String> = node
            .parents
            .iter()
            .map(|parent| normalize_tag(parent))
            .filter(|parent| !parent.is_empty() && parent != &tag)
            .collect();
        let sorts = node.sorts.clone();
        let features = node.features;
        self.nodes.insert(
            tag.clone(),
            PosNode {
                tag,
                parents,
                sorts,
                features,
            },
        );
        true
    }

    pub fn declare_edge(&mut self, sub: &str, sup: &str) -> bool {
        let sub = normalize_tag(sub);
        let sup = normalize_tag(sup);
        if sub.is_empty() || sup.is_empty() || sub == sup {
            return false;
        }
        if self.ancestors(&sup).contains(&sub) {
            return false;
        }
        if !self.nodes.contains_key(&sub) {
            self.nodes.insert(
                sub.clone(),
                PosNode {
                    tag: sub.clone(),
                    parents: Vec::new(),
                    sorts: Vec::new(),
                    features: PosFeatures::default(),
                },
            );
        }
        let node = self.nodes.get_mut(&sub).expect("节点已插入");
        if node.parents.contains(&sup) {
            return false;
        }
        node.parents.push(sup);
        true
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.nodes.values().map(|node| node.parents.len()).sum()
    }

    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(String::as_str)
    }

    /// 闭包：包含自身的全部祖先标签，广度优先、去重、稳定排序。
    pub fn ancestors(&self, tag: &str) -> BTreeSet<String> {
        let tag = normalize_tag(tag);
        let mut seen = BTreeSet::new();
        if tag.is_empty() {
            return seen;
        }
        seen.insert(tag.clone());
        if !self.nodes.contains_key(&tag) {
            return seen;
        }
        let mut queue = VecDeque::new();
        queue.push_back((tag, 0usize));
        while let Some((current, hops)) = queue.pop_front() {
            if hops >= MAX_CLOSURE_HOPS {
                continue;
            }
            let Some(node) = self.nodes.get(&current) else {
                continue;
            };
            for parent in &node.parents {
                if seen.insert(parent.clone()) {
                    queue.push_back((parent.clone(), hops + 1));
                }
            }
        }
        seen
    }

    pub fn profile(&self, tag: &str) -> PosProfile {
        let normalized = normalize_tag(tag);
        let ancestors = self.ancestors(&normalized);
        let known = self.nodes.contains_key(&normalized);
        let mut sorts = BTreeSet::new();
        let mut merged = PosFeatures::default();
        for ancestor in &ancestors {
            if let Some(node) = self.nodes.get(ancestor) {
                sorts.extend(node.sorts.iter().copied());
                merged.merge(node.features);
            }
        }
        PosProfile {
            tag: normalized,
            known,
            ancestors: ancestors.into_iter().collect(),
            sorts: sorts.into_iter().collect(),
            features: merged,
        }
    }

    /// `actual` 是否为 `expected` 的后代（含自身）。
    pub fn subsumes(&self, expected: &str, actual: &str) -> bool {
        let expected = normalize_tag(expected);
        let actual = normalize_tag(actual);
        if expected.is_empty() || actual.is_empty() {
            return false;
        }
        if expected == actual {
            return true;
        }
        self.ancestors(&actual).contains(&expected)
    }

    pub fn is_nominal(&self, tag: &str) -> bool {
        self.profile(tag).is_nominal()
    }

    pub fn is_verbal(&self, tag: &str) -> bool {
        self.profile(tag).is_verbal()
    }

    pub fn is_referential(&self, tag: &str) -> bool {
        self.profile(tag).features.referential
    }

    pub fn is_countable(&self, tag: &str) -> bool {
        self.profile(tag).is_countable()
    }

    pub fn captures_entity(&self, expected: &str) -> bool {
        let profile = self.profile(expected);
        profile.is_nominal()
    }

    /// 实际词性是否可作为图谱实体提及（排除代词与纯时间词）。
    pub fn is_entity_tag(&self, tag: &str) -> bool {
        let profile = self.profile(tag);
        profile.is_nominal() && !profile.has_sort(Sort::Time)
    }
}

static STANDARD_ONTOLOGY: OnceLock<PosOntology> = OnceLock::new();

/// 进程级共享的默认词性本体。
pub fn standard() -> &'static PosOntology {
    STANDARD_ONTOLOGY.get_or_init(PosOntology::standard)
}

/// 模式词性是否接受实际词性（闭包语义）。
pub fn tag_matches(expected: &str, actual: &str) -> bool {
    standard().subsumes(expected, actual)
}

pub fn profile(tag: &str) -> PosProfile {
    standard().profile(tag)
}

pub fn typed_mention(word: &AnnotatedWord) -> TypedMention {
    TypedMention::from_annotated(word, standard())
}

pub fn is_nominal_tag(tag: &str) -> bool {
    standard().is_nominal(tag)
}

pub fn captures_entity(expected: &str) -> bool {
    standard().captures_entity(expected)
}

pub fn is_entity_tag(tag: &str) -> bool {
    standard().is_entity_tag(tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_closure_covers_specific_noun_tags() {
        let ontology = PosOntology::standard();
        for tag in ["n", "nr", "ns", "nt", "nz", "vn", "an", "j", "s", "t"] {
            assert!(ontology.subsumes("n", tag), "n 应覆盖 {}", tag);
        }
        assert!(ontology.subsumes("nr", "nrfg"));
        assert!(ontology.subsumes("nr", "nrt"));
        assert!(!ontology.subsumes("nr", "ns"));
        assert!(!ontology.subsumes("n", "v"));
        assert!(!ontology.subsumes("v", "n"));
    }

    #[test]
    fn verbal_and_adjectival_closure_cover_multi_category_tags() {
        let ontology = PosOntology::standard();
        assert!(ontology.subsumes("v", "vn"));
        assert!(ontology.subsumes("v", "vd"));
        assert!(ontology.subsumes("a", "an"));
        assert!(ontology.subsumes("a", "ad"));
        assert!(ontology.subsumes("q", "mq"));
        assert!(ontology.subsumes("m", "mq"));
    }

    #[test]
    fn profile_projects_sorts_and_features() {
        let ontology = PosOntology::standard();
        let person = ontology.profile("nrfg");
        assert!(person.known);
        assert!(person.has_sort(Sort::Person));
        assert!(person.is_nominal());
        assert!(person.is_countable());
        assert_eq!(person.ontology_types(), vec!["人物".to_string()]);

        let organization = ontology.profile("nt");
        assert_eq!(organization.ontology_types(), vec!["组织".to_string()]);
        assert!(!organization.is_verbal());

        let pronoun = ontology.profile("r");
        assert!(pronoun.features.pronoun);
        assert!(pronoun.features.referential);
        assert!(!pronoun.is_nominal());
    }

    #[test]
    fn unknown_tags_stay_unknown_and_match_only_themselves() {
        let ontology = PosOntology::standard();
        assert!(!ontology.subsumes("n", "unknown"));
        assert!(!ontology.subsumes("n", "custom"));
        assert!(!ontology.profile("custom").known);
        assert_eq!(ontology.profile("custom").ancestors, vec!["custom"]);
    }

    #[test]
    fn capture_policy_excludes_pronouns_and_verbs() {
        let ontology = PosOntology::standard();
        assert!(ontology.captures_entity("n"));
        assert!(ontology.captures_entity("nr"));
        assert!(ontology.captures_entity("nt"));
        assert!(ontology.captures_entity("vn"));
        assert!(!ontology.captures_entity("r"));
        assert!(!ontology.captures_entity("v"));
        assert!(!ontology.captures_entity("q"));
    }

    #[test]
    fn entity_tag_policy_excludes_time_and_pronouns() {
        let ontology = PosOntology::standard();
        assert!(ontology.is_entity_tag("n"));
        assert!(ontology.is_entity_tag("nt"));
        assert!(ontology.is_entity_tag("vn"));
        assert!(ontology.is_entity_tag("s"));
        assert!(!ontology.is_entity_tag("t"));
        assert!(!ontology.is_entity_tag("r"));
        assert!(!ontology.is_entity_tag("v"));
    }

    #[test]
    fn typed_mention_keeps_source_and_type_projection() {
        let word = AnnotatedWord {
            text: "北京".to_string(),
            pos: "ns".to_string(),
            source: TagSource::JiebaTag,
            jieba_pos: Some("ns".to_string()),
            start: 0,
            end: 2,
        };
        let mention = typed_mention(&word);
        assert_eq!(mention.surface, "北京");
        assert_eq!(mention.source, TagSource::JiebaTag);
        assert!(mention.is_nominal());
        assert_eq!(mention.ontology_types, vec!["地点".to_string()]);
    }

    #[test]
    fn custom_edges_extend_closure_without_touching_standard() {
        let mut ontology = PosOntology::standard();
        assert!(ontology.declare_edge("my_noun", "n"));
        assert!(ontology.subsumes("n", "my_noun"));
        assert!(!standard().subsumes("n", "my_noun"));
        assert!(!ontology.declare_edge("n", "my_noun"));
        assert!(ontology.declare_edge("my_noun", "my_parent"));
        assert!(ontology.subsumes("n", "my_noun"));
        assert!(ontology.subsumes("my_parent", "my_noun"));
    }

    #[test]
    fn tag_lookup_is_case_and_whitespace_insensitive() {
        let ontology = PosOntology::standard();
        assert!(ontology.subsumes(" N ", "NR"));
        assert_eq!(ontology.profile(" NS ").tag, "ns");
    }
}

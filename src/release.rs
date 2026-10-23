//! REL：语义发布（Semantic Release）——把语言资产编译成可差分、可回归、
//! 可回滚的发布单元。
//!
//! 符号系统没有梯度，但有一组显式资产：实体类型、属性、抽取规则、Prolog
//! 规则、谓词画像、类型格、同义词、提示模板与插件贡献。本模块把这些资产
//! 投影成 [`SchemaSnapshot`]（单一正本由配置派生，不手写），与上一版做
//! [`SchemaDiff`]，按 breaking / compatible / behavioral 分类，再用 semver
//! 规则约束版本号。
//!
//! [`check`] 是发布总闸门：config lint → prompt lint → plugin eval → schema
//! diff → NLU 评测 → 恢复评测 → 身份评测 → 质量差分模糊测试。所有报告只含确定性数据
//! （BTreeMap 排序、无时间戳），同一资产必然得到同一摘要。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::canonical;
use crate::config_contract;
use crate::dialogue_eval::{self, DialogueEvalBaseline, EvalKnowledge};
use crate::dialogue_fuzz::{self, FuzzBaseline, FuzzConfig};
use crate::entailment_eval::{self, EntailmentEvalBaseline};
use crate::eval_recovery::{self, EvalRecoveryBaseline};
use crate::identity_eval::{self, IdentityEvalBaseline};
use crate::plugin::{PluginRegistry, PluginRuntime};
use crate::prompt_spec::{self, LintSeverity, PromptMode};
use crate::shape_eval;
use crate::SystemConfig;

pub const SCHEMA_SNAPSHOT_VERSION: u32 = 1;
pub const RELEASE_LOCK_VERSION: u32 = 1;
pub const DEFAULT_LOCK_PATH: &str = "release.lock.json";

fn digest<T: Serialize>(value: &T) -> String {
    canonical::canonical_digest(value).unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyShape {
    pub data_type: String,
    pub required: bool,
    pub extraction_patterns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityTypeShape {
    pub examples: Vec<String>,
    pub properties: BTreeMap<String, PropertyShape>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternShape {
    pub pos: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,
    pub properties: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionRuleShape {
    pub entity_type: String,
    pub priority: i32,
    pub patterns: Vec<PatternShape>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrologRuleShape {
    pub query_type: String,
    pub priority: i32,
    pub pattern: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryTypeShape {
    pub direct: bool,
    pub generated: bool,
    pub formatted: bool,
    pub supported: bool,
    pub declared_by: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateShape {
    pub aliases: BTreeSet<String>,
    pub inverse: Option<String>,
    pub functional: bool,
    pub transitive: bool,
    pub symmetric: bool,
    pub domain: BTreeSet<String>,
    pub range: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityProfileShape {
    pub aliases: BTreeSet<String>,
    pub types: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeShape {
    pub closure_gate: bool,
    pub aliases: BTreeMap<String, String>,
    pub subclasses: BTreeMap<String, BTreeSet<String>>,
    pub disjoint: BTreeSet<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexiconShape {
    pub import_keywords: BTreeSet<String>,
    pub pos: BTreeMap<String, String>,
    pub name_prefixes: Vec<String>,
    pub name_suffixes: Vec<String>,
    pub full_names: Vec<String>,
    pub profession_suffixes: Vec<String>,
    pub synonyms: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptShape {
    pub policy: String,
    pub template: String,
    pub required: Vec<String>,
    pub optional: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionShape {
    pub kind: String,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginShape {
    pub version: String,
    pub api: u32,
    pub default_mode: String,
    pub mode: String,
    pub requires: BTreeSet<String>,
    pub conflicts: BTreeSet<String>,
    pub provides: BTreeSet<String>,
    pub contributions: BTreeMap<String, ContributionShape>,
    pub eval_cases_digest: String,
}

/// 语言能力的结构化投影：同一配置 + 插件目录 + 提示资产 ⇒ 同一 snapshot。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaSnapshot {
    pub snapshot_version: u32,
    pub entity_types: BTreeMap<String, EntityTypeShape>,
    pub extraction_rules: BTreeMap<String, ExtractionRuleShape>,
    pub prolog_rules: BTreeMap<String, PrologRuleShape>,
    pub query_types: BTreeMap<String, QueryTypeShape>,
    pub predicate_profiles: BTreeMap<String, PredicateShape>,
    pub entity_profiles: BTreeMap<String, EntityProfileShape>,
    pub type_system: TypeShape,
    pub lexicon: LexiconShape,
    pub prompts: BTreeMap<String, PromptShape>,
    pub plugins: BTreeMap<String, PluginShape>,
    pub behavior: BTreeMap<String, String>,
}

impl SchemaSnapshot {
    pub fn build(config: &SystemConfig) -> Self {
        let registry = PluginRegistry::load_dir(&config.plugins.directory);
        Self::from_parts(config, &registry)
    }

    pub fn from_parts(config: &SystemConfig, registry: &PluginRegistry) -> Self {
        let mut entity_types = BTreeMap::new();
        for entity in &config.entity_types {
            let mut properties = BTreeMap::new();
            for property in &entity.properties {
                properties.insert(
                    property.name.clone(),
                    PropertyShape {
                        data_type: property.data_type.clone(),
                        required: property.required,
                        extraction_patterns: property.extraction_patterns.clone(),
                    },
                );
            }
            entity_types.insert(
                entity.label.clone(),
                EntityTypeShape {
                    examples: entity.examples.clone(),
                    properties,
                },
            );
        }

        let mut extraction_rules = BTreeMap::new();
        for rule in &config.extraction_rules {
            extraction_rules.insert(
                rule.name.clone(),
                ExtractionRuleShape {
                    entity_type: rule.entity_type.clone(),
                    priority: rule.priority,
                    patterns: rule
                        .patterns
                        .iter()
                        .map(|pattern| PatternShape {
                            pos: pattern.pos_pattern.clone(),
                            keywords: pattern.keyword_patterns.clone(),
                            properties: pattern.property_mappings.clone().into_iter().collect(),
                        })
                        .collect(),
                },
            );
        }

        let mut prolog_rules = BTreeMap::new();
        for rule in &config.prolog_rules {
            prolog_rules.insert(
                rule.name.clone(),
                PrologRuleShape {
                    query_type: rule.query_type.clone(),
                    priority: rule.priority,
                    pattern: rule.pattern.clone(),
                },
            );
        }

        let mut query_types: BTreeMap<String, QueryTypeShape> = BTreeMap::new();
        for query_type in config_contract::supported_query_types() {
            query_types.insert(
                query_type.to_string(),
                QueryTypeShape {
                    direct: config_contract::DIRECT_ANSWER_QUERY_TYPES.contains(&query_type),
                    generated: config_contract::GENERATED_QUERY_TYPES.contains(&query_type),
                    formatted: config_contract::FORMATTED_QUERY_TYPES.contains(&query_type),
                    supported: true,
                    declared_by: BTreeSet::new(),
                },
            );
        }
        for query_type in config_contract::FORMATTED_QUERY_TYPES {
            let entry = query_types
                .entry(query_type.to_string())
                .or_insert_with(|| QueryTypeShape {
                    direct: false,
                    generated: false,
                    formatted: true,
                    supported: false,
                    declared_by: BTreeSet::new(),
                });
            entry.formatted = true;
        }
        for (index, rule) in config.prolog_rules.iter().enumerate() {
            let entry = query_types
                .entry(rule.query_type.clone())
                .or_insert_with(|| QueryTypeShape {
                    direct: false,
                    generated: false,
                    formatted: false,
                    supported: config_contract::is_query_type_supported(&rule.query_type),
                    declared_by: BTreeSet::new(),
                });
            entry
                .declared_by
                .insert(format!("{}[{}]", rule.name, index));
        }

        let mut predicate_profiles = BTreeMap::new();
        for profile in &config.predicate_profiles {
            predicate_profiles.insert(
                profile.name.clone(),
                PredicateShape {
                    aliases: profile.aliases.iter().cloned().collect(),
                    inverse: profile.inverse.clone(),
                    functional: profile.functional,
                    transitive: profile.transitive,
                    symmetric: profile.symmetric,
                    domain: profile.domain.iter().cloned().collect(),
                    range: profile.range.iter().cloned().collect(),
                },
            );
        }

        let mut entity_profiles = BTreeMap::new();
        for profile in &config.entity_profiles {
            entity_profiles.insert(
                profile.name.clone(),
                EntityProfileShape {
                    aliases: profile.aliases.iter().cloned().collect(),
                    types: profile.etype.iter().cloned().collect(),
                },
            );
        }

        let mut type_aliases = BTreeMap::new();
        for alias in &config.type_system.aliases {
            type_aliases.insert(
                normalize_type(&alias.alias),
                normalize_type(&alias.canonical),
            );
        }
        let mut subclasses: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for edge in &config.type_system.subclasses {
            subclasses
                .entry(normalize_type(&edge.sub))
                .or_default()
                .insert(normalize_type(&edge.sup));
        }
        let mut disjoint = BTreeSet::new();
        for pair in &config.type_system.disjoint {
            disjoint.insert(sorted_type_pair(&pair.left, &pair.right));
        }

        let lexicon = LexiconShape {
            import_keywords: config.nlp.import_keywords.iter().cloned().collect(),
            pos: config.nlp.pos_dict.clone().into_iter().collect(),
            name_prefixes: config.nlp.name_patterns.prefixes.clone(),
            name_suffixes: config.nlp.name_patterns.suffixes.clone(),
            full_names: config.nlp.name_patterns.full_names.clone(),
            profession_suffixes: config.nlp.name_patterns.profession_suffix.clone(),
            synonyms: config
                .synonyms
                .as_ref()
                .map(|synonyms| synonyms.mappings.clone().into_iter().collect())
                .unwrap_or_default(),
        };

        let prompts = prompt_spec::default_specs()
            .iter()
            .map(|spec| {
                (
                    spec.id.to_string(),
                    PromptShape {
                        policy: spec.policy.to_string(),
                        template: spec.template.to_string(),
                        required: spec.required.iter().map(|slot| slot.to_string()).collect(),
                        optional: spec.optional.iter().map(|slot| slot.to_string()).collect(),
                    },
                )
            })
            .collect();

        SchemaSnapshot {
            snapshot_version: SCHEMA_SNAPSHOT_VERSION,
            entity_types,
            extraction_rules,
            prolog_rules,
            query_types,
            predicate_profiles,
            entity_profiles,
            type_system: TypeShape {
                closure_gate: config.type_system.closure_gate,
                aliases: type_aliases,
                subclasses,
                disjoint,
            },
            lexicon,
            prompts,
            plugins: build_plugins(config, registry),
            behavior: build_behavior(config),
        }
    }

    pub fn digest(&self) -> String {
        digest(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.snapshot_version > SCHEMA_SNAPSHOT_VERSION {
            return Err(format!(
                "schema 快照版本 {} 高于支持版本 {}",
                self.snapshot_version, SCHEMA_SNAPSHOT_VERSION
            ));
        }
        Ok(())
    }
}

fn normalize_type(name: &str) -> String {
    name.trim().to_lowercase()
}

fn sorted_type_pair(left: &str, right: &str) -> (String, String) {
    let left = normalize_type(left);
    let right = normalize_type(right);
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn build_plugins(
    config: &SystemConfig,
    registry: &PluginRegistry,
) -> BTreeMap<String, PluginShape> {
    let runtime = PluginRuntime::with_settings(registry.clone(), &config.plugins);
    let mut plugins = BTreeMap::new();
    for manifest in registry.manifests() {
        let contributions = manifest
            .contributions
            .iter()
            .map(|contribution| {
                (
                    contribution.id().to_string(),
                    ContributionShape {
                        kind: contribution.kind().to_string(),
                        digest: digest(contribution),
                    },
                )
            })
            .collect();
        plugins.insert(
            manifest.id.clone(),
            PluginShape {
                version: manifest.version.clone(),
                api: manifest.api,
                default_mode: manifest.default_mode.code().to_string(),
                mode: runtime
                    .mode(&manifest.id)
                    .map(|mode| mode.code().to_string())
                    .unwrap_or_else(|| "off".to_string()),
                requires: manifest.requires.iter().cloned().collect(),
                conflicts: manifest.conflicts.iter().cloned().collect(),
                provides: manifest.provides.iter().cloned().collect(),
                contributions,
                eval_cases_digest: digest(&manifest.eval_cases),
            },
        );
    }
    plugins
}

fn build_behavior(config: &SystemConfig) -> BTreeMap<String, String> {
    let mut behavior = BTreeMap::new();
    let reasoning = &config.reasoning;
    let mut flag = |name: &str, value: bool| {
        behavior.insert(name.to_string(), value.to_string());
    };
    flag(
        "reasoning.enable_link_prediction",
        reasoning.enable_link_prediction,
    );
    flag(
        "reasoning.enable_semantic_embedding",
        reasoning.enable_semantic_embedding,
    );
    flag("reasoning.enable_generation", reasoning.enable_generation);
    flag("reasoning.enable_entailment", reasoning.enable_entailment);
    flag("reasoning.enable_resolution", reasoning.enable_resolution);
    flag(
        "reasoning.enable_association_rules",
        reasoning.enable_association_rules,
    );
    flag("reasoning.enable_abduction", reasoning.enable_abduction);
    flag(
        "reasoning.enable_clarification",
        reasoning.enable_clarification,
    );
    flag("reasoning.enable_graph_walk", reasoning.enable_graph_walk);
    flag("reasoning.enable_soft_logic", reasoning.enable_soft_logic);
    flag(
        "reasoning.enable_causal_reasoning",
        reasoning.enable_causal_reasoning,
    );
    flag(
        "reasoning.enable_gradient_attribution",
        reasoning.enable_gradient_attribution,
    );
    flag(
        "reasoning.enable_causal_tokenization",
        reasoning.enable_causal_tokenization,
    );
    flag("reasoning.enable_problog", reasoning.enable_problog);
    flag(
        "reasoning.enable_dialogue_manager",
        reasoning.enable_dialogue_manager,
    );
    flag("reasoning.enable_discourse", reasoning.enable_discourse);
    flag(
        "reasoning.enable_holographic_memory",
        reasoning.enable_holographic_memory,
    );
    flag(
        "reasoning.require_predicate_admission",
        reasoning.require_predicate_admission,
    );
    behavior.insert(
        "reasoning.link_prediction_min_confidence".to_string(),
        reasoning.link_prediction_min_confidence.to_string(),
    );
    behavior.insert(
        "reasoning.link_prediction_min_support".to_string(),
        reasoning.link_prediction_min_support.to_string(),
    );
    behavior.insert(
        "reasoning.ontology_alignment_threshold".to_string(),
        reasoning.ontology_alignment_threshold.to_string(),
    );
    behavior.insert(
        "reasoning.max_predictions".to_string(),
        reasoning.max_predictions.to_string(),
    );
    behavior.insert(
        "reasoning.embedding_similarity_floor".to_string(),
        reasoning.embedding_similarity_floor.to_string(),
    );
    behavior.insert(
        "reasoning.max_profile_sentences".to_string(),
        reasoning.max_profile_sentences.to_string(),
    );
    behavior.insert(
        "reasoning.completion_min_score".to_string(),
        reasoning.completion_min_score.to_string(),
    );
    behavior.insert(
        "reasoning.entailment_min_confidence".to_string(),
        reasoning.entailment_min_confidence.to_string(),
    );
    behavior.insert(
        "reasoning.resolution_max_depth".to_string(),
        reasoning.resolution_max_depth.to_string(),
    );
    behavior.insert(
        "reasoning.resolution_max_solutions".to_string(),
        reasoning.resolution_max_solutions.to_string(),
    );
    behavior.insert(
        "reasoning.association_min_support".to_string(),
        reasoning.association_min_support.to_string(),
    );
    behavior.insert(
        "reasoning.association_min_confidence".to_string(),
        reasoning.association_min_confidence.to_string(),
    );
    behavior.insert(
        "reasoning.association_min_lift".to_string(),
        reasoning.association_min_lift.to_string(),
    );
    behavior.insert(
        "reasoning.abduction_max_hypotheses".to_string(),
        reasoning.abduction_max_hypotheses.to_string(),
    );
    behavior.insert(
        "reasoning.abduction_max_scenario_size".to_string(),
        reasoning.abduction_max_scenario_size.to_string(),
    );
    behavior.insert(
        "reasoning.graph_walk_damping".to_string(),
        reasoning.graph_walk_damping.to_string(),
    );
    behavior.insert(
        "reasoning.graph_walk_iterations".to_string(),
        reasoning.graph_walk_iterations.to_string(),
    );
    behavior.insert(
        "reasoning.soft_logic_max_iterations".to_string(),
        reasoning.soft_logic_max_iterations.to_string(),
    );
    behavior.insert(
        "reasoning.soft_logic_min_probability".to_string(),
        reasoning.soft_logic_min_probability.to_string(),
    );
    behavior.insert(
        "reasoning.causal_max_depth".to_string(),
        reasoning.causal_max_depth.to_string(),
    );
    behavior.insert(
        "reasoning.problog_max_depth".to_string(),
        reasoning.problog_max_depth.to_string(),
    );
    behavior.insert(
        "reasoning.problog_max_proofs".to_string(),
        reasoning.problog_max_proofs.to_string(),
    );
    behavior.insert(
        "reasoning.discourse_max_sentences".to_string(),
        reasoning.discourse_max_sentences.to_string(),
    );
    behavior.insert(
        "reasoning.discourse_max_rounds".to_string(),
        reasoning.discourse_max_rounds.to_string(),
    );
    behavior.insert(
        "reasoning.holographic_dimension".to_string(),
        reasoning.holographic_dimension.to_string(),
    );
    behavior.insert(
        "reasoning.holographic_similarity_floor".to_string(),
        reasoning.holographic_similarity_floor.to_string(),
    );
    behavior.insert(
        "reasoning.holographic_max_hits".to_string(),
        reasoning.holographic_max_hits.to_string(),
    );
    behavior.insert(
        "system.default_language".to_string(),
        config.system.default_language.clone(),
    );
    let nlp = &config.nlp;
    let optional = [
        (
            "nlp.enable_nlu",
            nlp.enable_nlu.map(|value| value.to_string()),
        ),
        (
            "nlp.enable_local_semantic",
            nlp.enable_local_semantic.map(|value| value.to_string()),
        ),
        (
            "nlp.disable_ownthink_api",
            nlp.disable_ownthink_api.map(|value| value.to_string()),
        ),
        (
            "nlp.enable_advanced_entity_extraction",
            nlp.enable_advanced_entity_extraction
                .map(|value| value.to_string()),
        ),
        (
            "nlp.enable_relation_extraction",
            nlp.enable_relation_extraction
                .map(|value| value.to_string()),
        ),
        (
            "nlp.nlu_confidence_threshold",
            nlp.nlu_confidence_threshold.map(|value| value.to_string()),
        ),
        (
            "nlp.semantic_similarity_threshold",
            nlp.semantic_similarity_threshold
                .map(|value| value.to_string()),
        ),
        (
            "nlp.max_entity_candidates",
            nlp.max_entity_candidates.map(|value| value.to_string()),
        ),
        (
            "nlp.context_window_size",
            nlp.context_window_size.map(|value| value.to_string()),
        ),
    ];
    for (name, value) in optional {
        behavior.insert(
            name.to_string(),
            value.unwrap_or_else(|| "unset".to_string()),
        );
    }
    behavior
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeSeverity {
    Breaking,
    Compatible,
    Behavioral,
}

impl ChangeSeverity {
    pub fn label(self) -> &'static str {
        match self {
            ChangeSeverity::Breaking => "BREAKING",
            ChangeSeverity::Compatible => "COMPATIBLE",
            ChangeSeverity::Behavioral => "BEHAVIORAL",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaChange {
    pub severity: ChangeSeverity,
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaDiff {
    pub changes: Vec<SchemaChange>,
}

impl SchemaDiff {
    pub fn between(old: &SchemaSnapshot, new: &SchemaSnapshot) -> Self {
        let mut changes = Vec::new();
        if old.snapshot_version != new.snapshot_version {
            changes.push(change(
                ChangeSeverity::Breaking,
                "snapshot_version_changed",
                "snapshot_version",
                format!(
                    "快照版本 {} → {}",
                    old.snapshot_version, new.snapshot_version
                ),
            ));
        }
        diff_entity_types(old, new, &mut changes);
        diff_extraction_rules(old, new, &mut changes);
        diff_prolog_rules(old, new, &mut changes);
        diff_query_types(old, new, &mut changes);
        diff_predicates(old, new, &mut changes);
        diff_entity_profiles(old, new, &mut changes);
        diff_type_system(old, new, &mut changes);
        diff_lexicon(old, new, &mut changes);
        diff_prompts(old, new, &mut changes);
        diff_plugins(old, new, &mut changes);
        diff_behavior(old, new, &mut changes);
        changes.sort_by(|left, right| {
            (left.severity, &left.path, &left.code).cmp(&(right.severity, &right.path, &right.code))
        });
        SchemaDiff { changes }
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn count(&self, severity: ChangeSeverity) -> usize {
        self.changes
            .iter()
            .filter(|change| change.severity == severity)
            .count()
    }

    pub fn has_breaking(&self) -> bool {
        self.count(ChangeSeverity::Breaking) > 0
    }

    pub fn has_compatible(&self) -> bool {
        self.count(ChangeSeverity::Compatible) > 0
    }

    pub fn has_behavioral(&self) -> bool {
        self.count(ChangeSeverity::Behavioral) > 0
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "schema 差分：{} breaking / {} compatible / {} behavioral",
            self.count(ChangeSeverity::Breaking),
            self.count(ChangeSeverity::Compatible),
            self.count(ChangeSeverity::Behavioral)
        )];
        if self.changes.is_empty() {
            lines.push("  schema 未变化。".to_string());
            return lines.join("\n");
        }
        for change in &self.changes {
            lines.push(format!(
                "  [{}] {} {}：{}",
                change.severity.label(),
                change.code,
                change.path,
                change.message
            ));
        }
        lines.join("\n")
    }
}

fn change(
    severity: ChangeSeverity,
    code: &str,
    path: impl Into<String>,
    message: impl Into<String>,
) -> SchemaChange {
    SchemaChange {
        severity,
        code: code.to_string(),
        path: path.into(),
        message: message.into(),
    }
}

fn removed<T>(changes: &mut Vec<SchemaChange>, code: &str, path: &str, value: &T)
where
    T: std::fmt::Debug,
{
    changes.push(change(
        ChangeSeverity::Breaking,
        code,
        path,
        format!("已删除：{:?}", value),
    ));
}

fn added<T>(changes: &mut Vec<SchemaChange>, code: &str, path: &str, value: &T)
where
    T: std::fmt::Debug,
{
    changes.push(change(
        ChangeSeverity::Compatible,
        code,
        path,
        format!("已新增：{:?}", value),
    ));
}

fn changed<T: std::fmt::Debug>(
    changes: &mut Vec<SchemaChange>,
    severity: ChangeSeverity,
    code: &str,
    path: &str,
    old: &T,
    new: &T,
) {
    changes.push(change(
        severity,
        code,
        path,
        format!("{:?} → {:?}", old, new),
    ));
}

fn diff_entity_types(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (label, shape) in &old.entity_types {
        let path = format!("entity_types.{}", label);
        match new.entity_types.get(label) {
            None => removed(changes, "entity_type_removed", &path, &label),
            Some(next) => {
                if shape.examples != next.examples {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "entity_type_examples_changed",
                        &path,
                        &shape.examples,
                        &next.examples,
                    );
                }
                for (name, property) in &shape.properties {
                    let property_path = format!("{}.properties.{}", path, name);
                    match next.properties.get(name) {
                        None => removed(changes, "property_removed", &property_path, &name),
                        Some(property_next) => {
                            if property.data_type != property_next.data_type {
                                changed(
                                    changes,
                                    ChangeSeverity::Breaking,
                                    "property_type_changed",
                                    &property_path,
                                    &property.data_type,
                                    &property_next.data_type,
                                );
                            }
                            if property.required != property_next.required {
                                let severity = if property_next.required {
                                    ChangeSeverity::Breaking
                                } else {
                                    ChangeSeverity::Compatible
                                };
                                changed(
                                    changes,
                                    severity,
                                    "property_required_changed",
                                    &property_path,
                                    &property.required,
                                    &property_next.required,
                                );
                            }
                            if property.extraction_patterns != property_next.extraction_patterns {
                                changed(
                                    changes,
                                    ChangeSeverity::Behavioral,
                                    "property_patterns_changed",
                                    &property_path,
                                    &property.extraction_patterns,
                                    &property_next.extraction_patterns,
                                );
                            }
                        }
                    }
                }
                for (name, property) in &next.properties {
                    if !shape.properties.contains_key(name) {
                        let severity = if property.required {
                            ChangeSeverity::Breaking
                        } else {
                            ChangeSeverity::Compatible
                        };
                        changes.push(change(
                            severity,
                            "property_added",
                            format!("{}.properties.{}", path, name),
                            format!("已新增属性：{:?}", property),
                        ));
                    }
                }
            }
        }
    }
    for (label, shape) in &new.entity_types {
        if !old.entity_types.contains_key(label) {
            added(
                changes,
                "entity_type_added",
                &format!("entity_types.{}", label),
                shape,
            );
        }
    }
}

fn diff_extraction_rules(
    old: &SchemaSnapshot,
    new: &SchemaSnapshot,
    changes: &mut Vec<SchemaChange>,
) {
    for (name, rule) in &old.extraction_rules {
        let path = format!("extraction_rules.{}", name);
        match new.extraction_rules.get(name) {
            None => removed(changes, "extraction_rule_removed", &path, &name),
            Some(next) => {
                if rule.entity_type != next.entity_type {
                    changed(
                        changes,
                        ChangeSeverity::Breaking,
                        "extraction_rule_type_changed",
                        &path,
                        &rule.entity_type,
                        &next.entity_type,
                    );
                }
                if rule.priority != next.priority {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "extraction_rule_priority_changed",
                        &path,
                        &rule.priority,
                        &next.priority,
                    );
                }
                if rule.patterns != next.patterns {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "extraction_rule_patterns_changed",
                        &path,
                        &rule.patterns,
                        &next.patterns,
                    );
                }
            }
        }
    }
    for (name, rule) in &new.extraction_rules {
        if !old.extraction_rules.contains_key(name) {
            added(
                changes,
                "extraction_rule_added",
                &format!("extraction_rules.{}", name),
                rule,
            );
        }
    }
}

fn diff_prolog_rules(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (name, rule) in &old.prolog_rules {
        let path = format!("prolog_rules.{}", name);
        match new.prolog_rules.get(name) {
            None => removed(changes, "prolog_rule_removed", &path, &name),
            Some(next) => {
                if rule.query_type != next.query_type {
                    changed(
                        changes,
                        ChangeSeverity::Breaking,
                        "prolog_rule_query_type_changed",
                        &path,
                        &rule.query_type,
                        &next.query_type,
                    );
                }
                if rule.priority != next.priority {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "prolog_rule_priority_changed",
                        &path,
                        &rule.priority,
                        &next.priority,
                    );
                }
                if rule.pattern != next.pattern {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "prolog_rule_pattern_changed",
                        &path,
                        &rule.pattern,
                        &next.pattern,
                    );
                }
            }
        }
    }
    for (name, rule) in &new.prolog_rules {
        if !old.prolog_rules.contains_key(name) {
            added(
                changes,
                "prolog_rule_added",
                &format!("prolog_rules.{}", name),
                rule,
            );
        }
    }
}

fn diff_query_types(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (query_type, shape) in &old.query_types {
        let path = format!("query_types.{}", query_type);
        match new.query_types.get(query_type) {
            None => removed(changes, "query_type_removed", &path, &query_type),
            Some(next) => {
                if shape.supported != next.supported
                    || shape.direct != next.direct
                    || shape.generated != next.generated
                {
                    changed(
                        changes,
                        ChangeSeverity::Breaking,
                        "query_type_capability_changed",
                        &path,
                        shape,
                        next,
                    );
                }
                if shape.declared_by != next.declared_by {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "query_type_declarations_changed",
                        &path,
                        &shape.declared_by,
                        &next.declared_by,
                    );
                }
            }
        }
    }
    for (query_type, shape) in &new.query_types {
        if !old.query_types.contains_key(query_type) {
            added(
                changes,
                "query_type_added",
                &format!("query_types.{}", query_type),
                shape,
            );
        }
    }
}

fn diff_predicates(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (name, profile) in &old.predicate_profiles {
        let path = format!("predicate_profiles.{}", name);
        match new.predicate_profiles.get(name) {
            None => {
                removed(changes, "predicate_profile_removed", &path, &name);
                continue;
            }
            Some(next) => {
                for alias in &profile.aliases {
                    if !next.aliases.contains(alias) {
                        changes.push(change(
                            ChangeSeverity::Breaking,
                            "predicate_alias_removed",
                            format!("{}.aliases.{}", path, alias),
                            "已删除谓词别名，旧说法将不再归一。",
                        ));
                    }
                }
                for alias in &next.aliases {
                    if !profile.aliases.contains(alias) {
                        added(
                            changes,
                            "predicate_alias_added",
                            &format!("{}.aliases.{}", path, alias),
                            alias,
                        );
                    }
                }
                if profile.inverse != next.inverse {
                    changed(
                        changes,
                        ChangeSeverity::Breaking,
                        "predicate_inverse_changed",
                        &path,
                        &profile.inverse,
                        &next.inverse,
                    );
                }
                if profile.functional != next.functional {
                    let severity = if next.functional {
                        ChangeSeverity::Breaking
                    } else {
                        ChangeSeverity::Compatible
                    };
                    changed(
                        changes,
                        severity,
                        "predicate_functional_changed",
                        &path,
                        &profile.functional,
                        &next.functional,
                    );
                }
                if profile.transitive != next.transitive {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "predicate_transitive_changed",
                        &path,
                        &profile.transitive,
                        &next.transitive,
                    );
                }
                if profile.symmetric != next.symmetric {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "predicate_symmetric_changed",
                        &path,
                        &profile.symmetric,
                        &next.symmetric,
                    );
                }
                diff_domain_range(changes, &path, "domain", &profile.domain, &next.domain);
                diff_domain_range(changes, &path, "range", &profile.range, &next.range);
            }
        }
    }
    for (name, profile) in &new.predicate_profiles {
        if !old.predicate_profiles.contains_key(name) {
            added(
                changes,
                "predicate_profile_added",
                &format!("predicate_profiles.{}", name),
                profile,
            );
        }
    }
}

fn diff_domain_range(
    changes: &mut Vec<SchemaChange>,
    path: &str,
    slot: &str,
    old: &BTreeSet<String>,
    new: &BTreeSet<String>,
) {
    let narrowed: Vec<&String> = new.difference(old).collect();
    if !narrowed.is_empty() {
        changes.push(change(
            ChangeSeverity::Breaking,
            "predicate_restriction_added",
            format!("{}.{}", path, slot),
            format!("{} 收窄，新增限制：{:?}", slot, narrowed),
        ));
    }
    let relaxed: Vec<&String> = old.difference(new).collect();
    if !relaxed.is_empty() {
        changes.push(change(
            ChangeSeverity::Compatible,
            "predicate_restriction_removed",
            format!("{}.{}", path, slot),
            format!("{} 放宽，删除限制：{:?}", slot, relaxed),
        ));
    }
}

fn diff_entity_profiles(
    old: &SchemaSnapshot,
    new: &SchemaSnapshot,
    changes: &mut Vec<SchemaChange>,
) {
    for (name, profile) in &old.entity_profiles {
        let path = format!("entity_profiles.{}", name);
        match new.entity_profiles.get(name) {
            None => removed(changes, "entity_profile_removed", &path, &name),
            Some(next) => {
                for alias in &profile.aliases {
                    if !next.aliases.contains(alias) {
                        changes.push(change(
                            ChangeSeverity::Breaking,
                            "entity_alias_removed",
                            format!("{}.aliases.{}", path, alias),
                            "已删除实体别名。",
                        ));
                    }
                }
                for alias in &next.aliases {
                    if !profile.aliases.contains(alias) {
                        added(
                            changes,
                            "entity_alias_added",
                            &format!("{}.aliases.{}", path, alias),
                            alias,
                        );
                    }
                }
                if profile.types != next.types {
                    let severity = if next.types.is_subset(&profile.types) {
                        ChangeSeverity::Compatible
                    } else {
                        ChangeSeverity::Breaking
                    };
                    changed(
                        changes,
                        severity,
                        "entity_profile_types_changed",
                        &path,
                        &profile.types,
                        &next.types,
                    );
                }
            }
        }
    }
    for (name, profile) in &new.entity_profiles {
        if !old.entity_profiles.contains_key(name) {
            added(
                changes,
                "entity_profile_added",
                &format!("entity_profiles.{}", name),
                profile,
            );
        }
    }
}

fn diff_type_system(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    if old.type_system.closure_gate != new.type_system.closure_gate {
        let severity = if new.type_system.closure_gate {
            ChangeSeverity::Breaking
        } else {
            ChangeSeverity::Compatible
        };
        changed(
            changes,
            severity,
            "type_closure_gate_changed",
            "type_system.closure_gate",
            &old.type_system.closure_gate,
            &new.type_system.closure_gate,
        );
    }
    for (alias, canonical) in &old.type_system.aliases {
        let path = format!("type_system.aliases.{}", alias);
        match new.type_system.aliases.get(alias) {
            None => removed(changes, "type_alias_removed", &path, alias),
            Some(next) if canonical != next => changed(
                changes,
                ChangeSeverity::Breaking,
                "type_alias_retargeted",
                &path,
                canonical,
                next,
            ),
            Some(_) => {}
        }
    }
    for (alias, canonical) in &new.type_system.aliases {
        if !old.type_system.aliases.contains_key(alias) {
            added(
                changes,
                "type_alias_added",
                &format!("type_system.aliases.{}", alias),
                canonical,
            );
        }
    }
    for (sub, parents) in &old.type_system.subclasses {
        let path = format!("type_system.subclasses.{}", sub);
        match new.type_system.subclasses.get(sub) {
            None => removed(changes, "type_subclass_removed", &path, parents),
            Some(next) => {
                for parent in parents {
                    if !next.contains(parent) {
                        changes.push(change(
                            ChangeSeverity::Breaking,
                            "type_subclass_edge_removed",
                            format!("{}.{}", path, parent),
                            "已删除子类边，旧类型闭包不再成立。",
                        ));
                    }
                }
                for parent in next {
                    if !parents.contains(parent) {
                        added(
                            changes,
                            "type_subclass_edge_added",
                            &format!("{}.{}", path, parent),
                            parent,
                        );
                    }
                }
            }
        }
    }
    for (sub, parents) in &new.type_system.subclasses {
        if !old.type_system.subclasses.contains_key(sub) {
            added(
                changes,
                "type_subclass_added",
                &format!("type_system.subclasses.{}", sub),
                parents,
            );
        }
    }
    for pair in &old.type_system.disjoint {
        if !new.type_system.disjoint.contains(pair) {
            changes.push(change(
                ChangeSeverity::Compatible,
                "type_disjoint_removed",
                format!("type_system.disjoint.{:?}", pair),
                "已删除互斥声明，类型判定放宽。",
            ));
        }
    }
    for pair in &new.type_system.disjoint {
        if !old.type_system.disjoint.contains(pair) {
            changes.push(change(
                ChangeSeverity::Breaking,
                "type_disjoint_added",
                format!("type_system.disjoint.{:?}", pair),
                "新增互斥声明，部分旧事实可能被拒绝。",
            ));
        }
    }
}

fn diff_lexicon(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for keyword in &old.lexicon.import_keywords {
        if !new.lexicon.import_keywords.contains(keyword) {
            changes.push(change(
                ChangeSeverity::Breaking,
                "import_keyword_removed",
                format!("lexicon.import_keywords.{}", keyword),
                "已删除导入关键词。",
            ));
        }
    }
    for keyword in &new.lexicon.import_keywords {
        if !old.lexicon.import_keywords.contains(keyword) {
            added(
                changes,
                "import_keyword_added",
                &format!("lexicon.import_keywords.{}", keyword),
                keyword,
            );
        }
    }
    for (word, tag) in &old.lexicon.pos {
        let path = format!("lexicon.pos.{}", word);
        match new.lexicon.pos.get(word) {
            None => removed(changes, "pos_entry_removed", &path, word),
            Some(next) if tag != next => changed(
                changes,
                ChangeSeverity::Behavioral,
                "pos_tag_changed",
                &path,
                tag,
                next,
            ),
            Some(_) => {}
        }
    }
    for (word, tag) in &new.lexicon.pos {
        if !old.lexicon.pos.contains_key(word) {
            added(
                changes,
                "pos_entry_added",
                &format!("lexicon.pos.{}", word),
                tag,
            );
        }
    }
    diff_string_list(
        changes,
        "name_prefix",
        "lexicon.name_prefixes",
        &old.lexicon.name_prefixes,
        &new.lexicon.name_prefixes,
    );
    diff_string_list(
        changes,
        "name_suffix",
        "lexicon.name_suffixes",
        &old.lexicon.name_suffixes,
        &new.lexicon.name_suffixes,
    );
    diff_string_list(
        changes,
        "full_name",
        "lexicon.full_names",
        &old.lexicon.full_names,
        &new.lexicon.full_names,
    );
    diff_string_list(
        changes,
        "profession_suffix",
        "lexicon.profession_suffixes",
        &old.lexicon.profession_suffixes,
        &new.lexicon.profession_suffixes,
    );
    for (word, canonical) in &old.lexicon.synonyms {
        let path = format!("lexicon.synonyms.{}", word);
        match new.lexicon.synonyms.get(word) {
            None => removed(changes, "synonym_removed", &path, word),
            Some(next) if canonical != next => changed(
                changes,
                ChangeSeverity::Breaking,
                "synonym_retargeted",
                &path,
                canonical,
                next,
            ),
            Some(_) => {}
        }
    }
    for (word, canonical) in &new.lexicon.synonyms {
        if !old.lexicon.synonyms.contains_key(word) {
            added(
                changes,
                "synonym_added",
                &format!("lexicon.synonyms.{}", word),
                canonical,
            );
        }
    }
}

fn diff_string_list(
    changes: &mut Vec<SchemaChange>,
    code: &str,
    path: &str,
    old: &[String],
    new: &[String],
) {
    let old_set: BTreeSet<&String> = old.iter().collect();
    let new_set: BTreeSet<&String> = new.iter().collect();
    for value in old_set.difference(&new_set) {
        changes.push(change(
            ChangeSeverity::Breaking,
            &format!("{}_removed", code),
            format!("{}.{}", path, value),
            "已删除词表条目。",
        ));
    }
    for value in new_set.difference(&old_set) {
        changes.push(change(
            ChangeSeverity::Compatible,
            &format!("{}_added", code),
            format!("{}.{}", path, value),
            "已新增词表条目。",
        ));
    }
    if old_set == new_set && old != new {
        changes.push(change(
            ChangeSeverity::Behavioral,
            "lexicon_order_changed",
            path,
            "词表条目顺序变化（可能改变分词优先级）。",
        ));
    }
}

fn diff_prompts(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (id, prompt) in &old.prompts {
        let path = format!("prompts.{}", id);
        match new.prompts.get(id) {
            None => removed(changes, "prompt_removed", &path, id),
            Some(next) => {
                if prompt.policy != next.policy {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "prompt_policy_changed",
                        &path,
                        &prompt.policy,
                        &next.policy,
                    );
                }
                if prompt.template != next.template {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "prompt_template_changed",
                        &path,
                        &prompt.template,
                        &next.template,
                    );
                }
                for slot in &next.required {
                    if !prompt.required.contains(slot) {
                        changes.push(change(
                            ChangeSeverity::Breaking,
                            "prompt_required_slot_added",
                            format!("{}.required.{}", path, slot),
                            "新增必填槽位，旧绑定可能无法渲染。",
                        ));
                    }
                }
                for slot in &prompt.required {
                    if !next.required.contains(slot) {
                        changes.push(change(
                            ChangeSeverity::Compatible,
                            "prompt_required_slot_removed",
                            format!("{}.required.{}", path, slot),
                            "已删除必填槽位，接地约束放宽。",
                        ));
                    }
                }
                if prompt.optional != next.optional {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "prompt_optional_slots_changed",
                        &path,
                        &prompt.optional,
                        &next.optional,
                    );
                }
            }
        }
    }
    for (id, prompt) in &new.prompts {
        if !old.prompts.contains_key(id) {
            added(changes, "prompt_added", &format!("prompts.{}", id), prompt);
        }
    }
}

fn diff_plugins(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (id, plugin) in &old.plugins {
        let path = format!("plugins.{}", id);
        match new.plugins.get(id) {
            None => removed(changes, "plugin_removed", &path, id),
            Some(next) => {
                if plugin.version != next.version {
                    let severity =
                        match (parse_semver(&plugin.version), parse_semver(&next.version)) {
                            (Some(old_version), Some(new_version)) if new_version < old_version => {
                                ChangeSeverity::Breaking
                            }
                            _ => ChangeSeverity::Behavioral,
                        };
                    changed(
                        changes,
                        severity,
                        "plugin_version_changed",
                        &path,
                        &plugin.version,
                        &next.version,
                    );
                }
                if plugin.api != next.api {
                    changed(
                        changes,
                        ChangeSeverity::Breaking,
                        "plugin_api_changed",
                        &path,
                        &plugin.api,
                        &next.api,
                    );
                }
                if plugin.default_mode != next.default_mode {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "plugin_default_mode_changed",
                        &path,
                        &plugin.default_mode,
                        &next.default_mode,
                    );
                }
                if plugin.mode != next.mode {
                    changed(
                        changes,
                        ChangeSeverity::Behavioral,
                        "plugin_mode_changed",
                        &path,
                        &plugin.mode,
                        &next.mode,
                    );
                }
                if plugin.requires != next.requires
                    || plugin.conflicts != next.conflicts
                    || plugin.provides != next.provides
                {
                    changes.push(change(
                        ChangeSeverity::Behavioral,
                        "plugin_dependencies_changed",
                        &path,
                        "依赖 / 冲突 / 提供声明发生变化。",
                    ));
                }
                for (contribution_id, contribution) in &plugin.contributions {
                    let contribution_path = format!("{}.contributions.{}", path, contribution_id);
                    match next.contributions.get(contribution_id) {
                        None => removed(
                            changes,
                            "plugin_contribution_removed",
                            &contribution_path,
                            contribution_id,
                        ),
                        Some(next_contribution) if contribution != next_contribution => {
                            changes.push(change(
                                ChangeSeverity::Behavioral,
                                "plugin_contribution_changed",
                                &contribution_path,
                                format!(
                                    "贡献内容变化（{} → {}）。",
                                    contribution.kind, next_contribution.kind
                                ),
                            ));
                        }
                        Some(_) => {}
                    }
                }
                for (contribution_id, contribution) in &next.contributions {
                    if !plugin.contributions.contains_key(contribution_id) {
                        added(
                            changes,
                            "plugin_contribution_added",
                            &format!("{}.contributions.{}", path, contribution_id),
                            &contribution.kind,
                        );
                    }
                }
                if plugin.eval_cases_digest != next.eval_cases_digest {
                    changes.push(change(
                        ChangeSeverity::Behavioral,
                        "plugin_eval_cases_changed",
                        &path,
                        "包内评测用例发生变化。",
                    ));
                }
            }
        }
    }
    for (id, plugin) in &new.plugins {
        if !old.plugins.contains_key(id) {
            added(changes, "plugin_added", &format!("plugins.{}", id), plugin);
        }
    }
}

fn diff_behavior(old: &SchemaSnapshot, new: &SchemaSnapshot, changes: &mut Vec<SchemaChange>) {
    for (key, value) in &old.behavior {
        let path = format!("behavior.{}", key);
        match new.behavior.get(key) {
            None => changes.push(change(
                ChangeSeverity::Behavioral,
                "behavior_setting_removed",
                &path,
                format!("设置已删除（回落到默认值）：{}", value),
            )),
            Some(next) if value != next => {
                let severity = if key == "reasoning.require_predicate_admission"
                    && value == "false"
                    && next == "true"
                {
                    ChangeSeverity::Breaking
                } else {
                    ChangeSeverity::Behavioral
                };
                changed(
                    changes,
                    severity,
                    "behavior_setting_changed",
                    &path,
                    value,
                    next,
                );
            }
            Some(_) => {}
        }
    }
    for (key, value) in &new.behavior {
        if !old.behavior.contains_key(key) {
            added(
                changes,
                "behavior_setting_added",
                &format!("behavior.{}", key),
                value,
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Semver {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

pub fn parse_semver(text: &str) -> Option<Semver> {
    let core = text.trim().split(['-', '+']).next().unwrap_or_default();
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(Semver {
        major,
        minor,
        patch,
    })
}

/// semver 规则：breaking 需要主版本（0.x 为次版本）提升；compatible 至少提升
/// 次版本；behavioral 至少提升补丁号。
pub fn check_version_policy(
    old_version: &str,
    new_version: &str,
    diff: &SchemaDiff,
) -> Result<(), String> {
    let old = parse_semver(old_version)
        .ok_or_else(|| format!("旧语言包版本「{}」不是 X.Y.Z", old_version))?;
    let new = parse_semver(new_version)
        .ok_or_else(|| format!("新语言包版本「{}」不是 X.Y.Z", new_version))?;
    let bumped_breaking = if old.major == 0 {
        new.major > old.major || (new.major == 0 && new.minor > old.minor)
    } else {
        new.major > old.major
    };
    let bumped_minor = new.major > old.major || (new.major == old.major && new.minor > old.minor);
    let bumped_patch =
        bumped_minor || (new.major == old.major && new.minor == old.minor && new.patch > old.patch);
    if diff.has_breaking() && !bumped_breaking {
        return Err(format!(
            "存在 {} 项 breaking 变更，版本必须提升主版本（0.x 提升次版本）：{} → {}",
            diff.count(ChangeSeverity::Breaking),
            old_version,
            new_version
        ));
    }
    if !diff.has_breaking() && diff.has_compatible() && !bumped_minor {
        return Err(format!(
            "存在 {} 项 compatible 变更，版本必须至少提升次版本：{} → {}",
            diff.count(ChangeSeverity::Compatible),
            old_version,
            new_version
        ));
    }
    if !diff.has_breaking() && !diff.has_compatible() && diff.has_behavioral() && !bumped_patch {
        return Err(format!(
            "存在 {} 项 behavioral 变更，版本必须提升：{} → {}",
            diff.count(ChangeSeverity::Behavioral),
            old_version,
            new_version
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    pub name: String,
    pub passed: bool,
    pub skipped: bool,
    pub digest: String,
    pub summary: String,
    #[serde(default)]
    pub detail: Vec<String>,
}

impl GateReport {
    fn new(name: &str, passed: bool, digest: String, summary: impl Into<String>) -> Self {
        GateReport {
            name: name.to_string(),
            passed,
            skipped: false,
            digest,
            summary: summary.into(),
            detail: Vec::new(),
        }
    }

    fn skipped(name: &str, reason: &str) -> Self {
        GateReport {
            name: name.to_string(),
            passed: true,
            skipped: true,
            digest: String::new(),
            summary: reason.to_string(),
            detail: Vec::new(),
        }
    }

    fn with_detail(mut self, detail: Vec<String>) -> Self {
        self.detail = detail;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReleaseReport {
    pub lock_path: String,
    pub language_version: String,
    pub schema_digest: String,
    pub lock_updated: bool,
    pub passed: bool,
    pub changes: Vec<SchemaChange>,
    pub gates: Vec<GateReport>,
}

impl ReleaseReport {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "发布检查：语言包 {}，schema {}，锁文件 {}{}",
            self.language_version,
            self.schema_digest,
            self.lock_path,
            if self.lock_updated {
                "（已更新）"
            } else {
                ""
            }
        )];
        let diff = SchemaDiff {
            changes: self.changes.clone(),
        };
        lines.push(diff.render_text());
        for gate in &self.gates {
            lines.push(format!(
                "  [{}] {}：{}",
                if gate.skipped {
                    "SKIP"
                } else if gate.passed {
                    "PASS"
                } else {
                    "FAIL"
                },
                gate.name,
                gate.summary
            ));
            for detail in &gate.detail {
                lines.push(format!("    - {}", detail));
            }
        }
        lines.push(if self.passed {
            "发布检查通过。".to_string()
        } else {
            "发布检查未通过。".to_string()
        });
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReleaseLock {
    pub lock_version: u32,
    pub language_version: String,
    pub schema: SchemaSnapshot,
    pub schema_digest: String,
    #[serde(default)]
    pub nlu: Option<DialogueEvalBaseline>,
    #[serde(default)]
    pub recovery: Option<EvalRecoveryBaseline>,
    #[serde(default)]
    pub identity: Option<IdentityEvalBaseline>,
    #[serde(default)]
    pub fuzz: Option<FuzzBaseline>,
    #[serde(default)]
    pub entailment: Option<EntailmentEvalBaseline>,
}

impl ReleaseLock {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|error| format!("读取发布锁 {} 失败：{}", path.display(), error))?;
        let lock: ReleaseLock = serde_json::from_str(&text)
            .map_err(|error| format!("解析发布锁 {} 失败：{}", path.display(), error))?;
        lock.validate()?;
        Ok(lock)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lock_version > RELEASE_LOCK_VERSION {
            return Err(format!(
                "发布锁版本 {} 高于支持版本 {}",
                self.lock_version, RELEASE_LOCK_VERSION
            ));
        }
        self.schema.validate()?;
        let actual = self.schema.digest();
        if actual != self.schema_digest {
            return Err(format!(
                "发布锁 schema 摘要不匹配（文件 {}，实算 {}）",
                self.schema_digest, actual
            ));
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let mut text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("序列化发布锁失败：{}", error))?;
        text.push('\n');
        fs::write(path, text)
            .map_err(|error| format!("写入发布锁 {} 失败：{}", path.display(), error))
    }
}

pub struct CheckOptions<'a> {
    pub config_file: &'a str,
    pub lock_path: &'a str,
    pub language_version: Option<&'a str>,
    pub update_lock: bool,
    pub skip_evals: bool,
}

pub fn check(
    config: &SystemConfig,
    registry: &PluginRegistry,
    options: &CheckOptions<'_>,
) -> Result<ReleaseReport, String> {
    let snapshot = SchemaSnapshot::from_parts(config, registry);
    snapshot.validate()?;
    let schema_digest = snapshot.digest();

    let old_lock = if Path::new(options.lock_path).exists() {
        Some(ReleaseLock::load(options.lock_path)?)
    } else {
        None
    };
    let new_version = options
        .language_version
        .map(str::to_string)
        .or_else(|| old_lock.as_ref().map(|lock| lock.language_version.clone()))
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    parse_semver(&new_version).ok_or_else(|| format!("语言包版本「{}」不是 X.Y.Z", new_version))?;

    let mut gates = Vec::new();

    let issues = config_contract::lint(config);
    let errors = config_contract::error_count(&issues);
    let warnings = config_contract::warning_count(&issues);
    let mut detail: Vec<String> = issues.iter().map(|issue| issue.line()).collect();
    if detail.len() > 40 {
        let omitted = detail.len() - 40;
        detail.truncate(40);
        detail.push(format!("……其余 {} 条省略", omitted));
    }
    gates.push(
        GateReport::new(
            "config_lint",
            errors == 0,
            digest(&issues),
            format!("{} error / {} warning", errors, warnings),
        )
        .with_detail(detail),
    );

    let prompt_issues = prompt_spec::lint_default();
    let prompt_errors = prompt_issues
        .iter()
        .filter(|issue| issue.severity == LintSeverity::Error)
        .count();
    let prompt_warnings = prompt_issues
        .iter()
        .filter(|issue| issue.severity == LintSeverity::Warning)
        .count();
    let (covered, composed) = prompt_spec::coverage();
    gates.push(
        GateReport::new(
            "prompt_lint",
            prompt_errors == 0,
            digest(&prompt_issues),
            format!(
                "{} 条模板，{} error / {} warning，策略覆盖 {}/{}",
                prompt_spec::default_specs().len(),
                prompt_errors,
                prompt_warnings,
                covered.len(),
                covered.len() + composed.len()
            ),
        )
        .with_detail(
            prompt_issues
                .iter()
                .map(|issue| {
                    format!(
                        "[{}] {}：{}",
                        match issue.severity {
                            LintSeverity::Error => "ERROR",
                            LintSeverity::Warning => "WARN",
                        },
                        issue.spec_id,
                        issue.message
                    )
                })
                .collect(),
        ),
    );

    let plugin_issues = registry.issues().to_vec();
    let plugin_evals = registry.evaluate();
    let plugin_errors = plugin_issues
        .iter()
        .filter(|issue| issue.severity == crate::plugin::PluginSeverity::Error)
        .count();
    let plugin_warnings = plugin_issues.len() - plugin_errors;
    let plugin_failures: Vec<String> = plugin_evals
        .iter()
        .filter(|result| !result.passed)
        .map(|result| format!("[{}] {} → {}", result.plugin, result.input, result.detail))
        .collect();
    let plugin_passed = plugin_errors == 0 && plugin_failures.is_empty();
    gates.push(
        GateReport::new(
            "plugin_lint",
            plugin_passed,
            digest(&(&plugin_issues, &plugin_evals)),
            format!(
                "{} 个插件，{} error / {} warning，评测 {}/{} 通过",
                registry.manifests().len(),
                plugin_errors,
                plugin_warnings,
                plugin_evals.iter().filter(|result| result.passed).count(),
                plugin_evals.len()
            ),
        )
        .with_detail(plugin_failures),
    );

    let mut changes = Vec::new();
    let schema_gate = match &old_lock {
        None => {
            if options.update_lock {
                GateReport::new(
                    "schema_diff",
                    true,
                    schema_digest.clone(),
                    "首次发布：建立发布锁基线",
                )
            } else {
                GateReport::new(
                    "schema_diff",
                    false,
                    schema_digest.clone(),
                    "发布锁不存在；先运行 release-check --update-lock 固化基线",
                )
            }
        }
        Some(lock) => {
            let diff = SchemaDiff::between(&lock.schema, &snapshot);
            changes = diff.changes.clone();
            let version_result = check_version_policy(&lock.language_version, &new_version, &diff);
            let passed = version_result.is_ok();
            let mut detail = Vec::new();
            if let Err(message) = &version_result {
                detail.push(message.clone());
            }
            detail.extend(diff.changes.iter().map(|change| {
                format!(
                    "[{}] {} {}：{}",
                    change.severity.label(),
                    change.code,
                    change.path,
                    change.message
                )
            }));
            GateReport::new(
                "schema_diff",
                passed,
                digest(&diff),
                format!(
                    "{} → {}：{} breaking / {} compatible / {} behavioral",
                    lock.language_version,
                    new_version,
                    diff.count(ChangeSeverity::Breaking),
                    diff.count(ChangeSeverity::Compatible),
                    diff.count(ChangeSeverity::Behavioral)
                ),
            )
            .with_detail(detail)
        }
    };
    gates.push(schema_gate);

    let mut nlu_baseline = old_lock.as_ref().and_then(|lock| lock.nlu.clone());
    let mut recovery_baseline = old_lock.as_ref().and_then(|lock| lock.recovery.clone());
    let mut identity_baseline = old_lock.as_ref().and_then(|lock| lock.identity.clone());
    let mut fuzz_baseline = old_lock.as_ref().and_then(|lock| lock.fuzz.clone());

    if options.skip_evals {
        gates.push(GateReport::skipped("eval_nlu", "已跳过（--skip-evals）"));
        gates.push(GateReport::skipped(
            "eval_recovery",
            "已跳过（--skip-evals）",
        ));
        gates.push(GateReport::skipped(
            "eval_identity",
            "已跳过（--skip-evals）",
        ));
        gates.push(GateReport::skipped(
            "fuzz_dialogue",
            "已跳过（--skip-evals）",
        ));
    } else {
        let nlu_report = dialogue_eval::evaluate_seed_with(EvalKnowledge::Reasoner, PromptMode::On);
        let (passed, summary, detail) = match &nlu_baseline {
            Some(baseline) => {
                let verdict = dialogue_eval::compare_baseline(baseline, &nlu_report, 1e-6);
                (
                    verdict.passed,
                    format!(
                        "意图 macro-F1 {:.4} / 答案 char-F1 {:.4}",
                        nlu_report.intents.macro_f1, nlu_report.answers.char_f1
                    ),
                    verdict
                        .regressions
                        .iter()
                        .map(|delta| {
                            format!(
                                "{}: {:.6} → {:.6}",
                                delta.metric, delta.baseline, delta.current
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            }
            None if options.update_lock => (
                true,
                "首次运行：以当前 NLU 报告作为基线".to_string(),
                Vec::new(),
            ),
            None => (
                false,
                "发布锁没有 NLU 基线；先运行 release-check --update-lock".to_string(),
                Vec::new(),
            ),
        };
        gates.push(
            GateReport::new("eval_nlu", passed, digest(&nlu_report), summary).with_detail(detail),
        );
        if options.update_lock {
            nlu_baseline = Some(DialogueEvalBaseline::from_report(&nlu_report));
        }

        let recovery_report = eval_recovery::evaluate_seed();
        let (passed, summary, detail) = match &recovery_baseline {
            Some(baseline) => {
                let verdict = eval_recovery::compare_baseline(baseline, &recovery_report, 1e-6);
                (
                    verdict.passed,
                    format!(
                        "静默失败率 {:.4} / 回执覆盖 {:.4}",
                        recovery_report.silent_failure_rate, recovery_report.receipt_coverage
                    ),
                    verdict
                        .regressions
                        .iter()
                        .map(|delta| {
                            format!(
                                "{}: {:.6} → {:.6}",
                                delta.metric, delta.baseline, delta.current
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            }
            None if options.update_lock => (
                true,
                "首次运行：以当前恢复报告作为基线".to_string(),
                Vec::new(),
            ),
            None => (
                false,
                "发布锁没有恢复基线；先运行 release-check --update-lock".to_string(),
                Vec::new(),
            ),
        };
        gates.push(
            GateReport::new("eval_recovery", passed, digest(&recovery_report), summary)
                .with_detail(detail),
        );
        if options.update_lock {
            recovery_baseline = Some(EvalRecoveryBaseline::from_report(&recovery_report));
        }

        let identity_report = identity_eval::evaluate_seed();
        let (passed, summary, detail) = match &identity_baseline {
            Some(baseline) => {
                let verdict = identity_eval::compare_baseline(baseline, &identity_report, 1e-6);
                (
                    verdict.passed,
                    format!(
                        "作用域准确率 {:.4} / claim F1 {:.4} / 越权率 {:.4}",
                        identity_report.act_accuracy,
                        identity_report.claim_f1,
                        identity_report.overclaim_rate
                    ),
                    verdict
                        .regressions
                        .iter()
                        .map(|delta| {
                            format!(
                                "{}: {:.6} → {:.6}",
                                delta.metric, delta.baseline, delta.current
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            }
            None if options.update_lock => (
                true,
                "首次运行：以当前身份报告作为基线".to_string(),
                Vec::new(),
            ),
            None => (
                false,
                "发布锁没有身份基线；先运行 release-check --update-lock".to_string(),
                Vec::new(),
            ),
        };
        gates.push(
            GateReport::new("eval_identity", passed, digest(&identity_report), summary)
                .with_detail(detail),
        );
        if options.update_lock {
            identity_baseline = Some(IdentityEvalBaseline::from_report(&identity_report));
        }

        let fuzz_report = dialogue_fuzz::run(&FuzzConfig::default());
        let (passed, summary, detail) = match &fuzz_baseline {
            Some(baseline) => {
                let verdict = dialogue_fuzz::compare_baseline(baseline, &fuzz_report);
                (
                    verdict.passed,
                    format!(
                        "用例 {}，失败 {}，基线外失败 {}",
                        fuzz_report.cases,
                        fuzz_report.failed,
                        verdict.new_failures.len()
                    ),
                    verdict
                        .new_failures
                        .iter()
                        .map(|failure| {
                            format!(
                                "{} / {}：{}（{}）",
                                failure.mutation,
                                failure.question,
                                failure.contract,
                                failure.detail
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            }
            None if options.update_lock => (
                true,
                "首次运行：以当前质量差分指纹作为 known-gap 基线".to_string(),
                Vec::new(),
            ),
            None => (
                false,
                "发布锁没有模糊测试基线；先运行 release-check --update-lock".to_string(),
                Vec::new(),
            ),
        };
        gates.push(
            GateReport::new("fuzz_dialogue", passed, fuzz_report.digest.clone(), summary)
                .with_detail(detail),
        );
        if options.update_lock {
            fuzz_baseline = Some(FuzzBaseline::from_report(&fuzz_report));
        }
    }

    let shape_report = shape_eval::evaluate_seed();
    let shape_passed = shape_report.passed();
    gates.push(
        GateReport::new(
            "eval_shape_dialogue",
            shape_passed,
            shape_report.digest.clone(),
            format!(
                "缺口精度 {:.4} / 承认精度 {:.4} / 冲突披露 {:.4} / 未知克制 {:.4}",
                shape_report.gap_precision(),
                shape_report.claim_precision(),
                shape_report.conflict_disclosure_rate(),
                shape_report.abstain_rate()
            ),
        )
        .with_detail(if shape_passed {
            Vec::new()
        } else {
            vec![shape_report.render_text()]
        }),
    );

    let entailment_report = entailment_eval::evaluate_seed();
    let entailment_baseline = old_lock.as_ref().and_then(|lock| lock.entailment.clone());
    let (entailment_passed, entailment_summary, entailment_detail) = match &entailment_baseline {
        Some(baseline) => {
            let verdict = entailment_eval::compare_baseline(baseline, &entailment_report, 1e-9);
            (
                verdict.passed,
                format!(
                    "金标准确率 {:.4} / 变形关系通过率 {:.4}",
                    entailment_report.gold_accuracy(),
                    entailment_report.metamorphic_pass_rate()
                ),
                verdict
                    .regressions
                    .iter()
                    .map(|delta| {
                        format!(
                            "{}: {:.6} → {:.6}",
                            delta.metric, delta.baseline, delta.current
                        )
                    })
                    .chain(if verdict.digest_matches {
                        Vec::new()
                    } else {
                        vec!["蕴含语料摘要漂移：基线不可直接比较".to_string()]
                    })
                    .chain(entailment_report.failures.iter().cloned())
                    .collect::<Vec<_>>(),
            )
        }
        None if options.update_lock => (
            true,
            "首次运行：以当前蕴含评测作为基线".to_string(),
            Vec::new(),
        ),
        None => (
            false,
            "发布锁没有蕴含基线；先运行 release-check --update-lock".to_string(),
            Vec::new(),
        ),
    };
    gates.push(
        GateReport::new(
            "eval_entailment",
            entailment_passed,
            entailment_report.digest(),
            entailment_summary,
        )
        .with_detail(entailment_detail),
    );

    let passed = gates.iter().all(|gate| gate.passed);
    let mut lock_updated = false;
    if options.update_lock && passed {
        let lock = ReleaseLock {
            lock_version: RELEASE_LOCK_VERSION,
            language_version: new_version.clone(),
            schema: snapshot,
            schema_digest: schema_digest.clone(),
            nlu: nlu_baseline,
            recovery: recovery_baseline,
            identity: identity_baseline,
            fuzz: fuzz_baseline,
            entailment: Some(EntailmentEvalBaseline::from_report(&entailment_report)),
        };
        lock.save(options.lock_path)?;
        lock_updated = true;
    }

    let mut detail = Vec::new();
    if options.update_lock && !passed {
        detail.push("存在失败的 gate，发布锁未更新。".to_string());
    }
    detail.push(format!("配置文件：{}", options.config_file));

    Ok(ReleaseReport {
        lock_path: options.lock_path.to_string(),
        language_version: new_version,
        schema_digest,
        lock_updated,
        passed,
        changes,
        gates: {
            let mut gates = gates;
            if let Some(first) = gates.first_mut() {
                first.detail.extend(detail);
            }
            gates
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> SystemConfig {
        toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析")
    }

    fn snapshot() -> SchemaSnapshot {
        SchemaSnapshot::from_parts(&sample_config(), &PluginRegistry::empty())
    }

    #[test]
    fn snapshot_is_deterministic_and_covers_assets() {
        let first = snapshot();
        let second = snapshot();
        assert_eq!(first, second);
        assert_eq!(first.digest(), second.digest());
        assert!(!first.entity_types.is_empty(), "应投影实体类型");
        assert!(!first.predicate_profiles.is_empty(), "应投影谓词画像");
        assert!(!first.prompts.is_empty(), "应投影提示模板");
        assert!(!first.lexicon.pos.is_empty(), "应投影词表");
        assert!(
            first.query_types.values().any(|shape| shape.supported),
            "应投影查询能力"
        );
    }

    #[test]
    fn diff_classifies_removals_as_breaking_and_additions_as_compatible() {
        let old = snapshot();
        let mut new = old.clone();
        let removed_type = old
            .entity_types
            .keys()
            .next()
            .expect("至少一个实体类型")
            .clone();
        new.entity_types.remove(&removed_type);
        let removed_prompt = old.prompts.keys().next().expect("至少一个模板").clone();
        new.prompts.remove(&removed_prompt);
        let diff = SchemaDiff::between(&old, &new);
        assert!(diff.has_breaking());
        assert!(diff
            .changes
            .iter()
            .any(|change| change.code == "entity_type_removed"));
        assert!(diff
            .changes
            .iter()
            .any(|change| change.code == "prompt_removed"));

        let reverse = SchemaDiff::between(&new, &old);
        assert!(reverse
            .changes
            .iter()
            .any(|change| change.code == "entity_type_added"
                && change.severity == ChangeSeverity::Compatible));
    }

    #[test]
    fn required_slot_added_is_breaking() {
        let old = snapshot();
        let mut new = old.clone();
        let (id, _) = old.prompts.iter().next().expect("至少一个模板");
        new.prompts
            .get_mut(id)
            .unwrap()
            .required
            .push("new_slot".to_string());
        let diff = SchemaDiff::between(&old, &new);
        assert!(diff.changes.iter().any(|change| {
            change.code == "prompt_required_slot_added"
                && change.severity == ChangeSeverity::Breaking
        }));
    }

    #[test]
    fn domain_narrowing_is_breaking_and_widening_is_compatible() {
        let old = snapshot();
        let mut new = old.clone();
        let (name, _) = old
            .predicate_profiles
            .iter()
            .find(|(_, profile)| !profile.domain.is_empty())
            .expect("至少一个带定义域的谓词");
        new.predicate_profiles
            .get_mut(name)
            .unwrap()
            .domain
            .insert("新增类型".to_string());
        let diff = SchemaDiff::between(&old, &new);
        assert!(diff.changes.iter().any(|change| {
            change.code == "predicate_restriction_added"
                && change.severity == ChangeSeverity::Breaking
        }));
    }

    #[test]
    fn version_policy_enforces_semver() {
        let old = snapshot();
        let mut new = old.clone();
        new.prompts.remove(old.prompts.keys().next().unwrap());
        let diff = SchemaDiff::between(&old, &new);
        assert!(check_version_policy("0.1.0", "0.1.0", &diff).is_err());
        assert!(check_version_policy("0.1.0", "0.1.1", &diff).is_err());
        assert!(check_version_policy("0.1.0", "0.2.0", &diff).is_ok());
        assert!(check_version_policy("1.2.0", "2.0.0", &diff).is_ok());

        let mut compatible_new = old.clone();
        compatible_new
            .predicate_profiles
            .entry("新谓词".to_string())
            .or_insert_with(|| PredicateShape {
                aliases: BTreeSet::new(),
                inverse: None,
                functional: false,
                transitive: false,
                symmetric: false,
                domain: BTreeSet::new(),
                range: BTreeSet::new(),
            });
        let compatible = SchemaDiff::between(&old, &compatible_new);
        assert!(check_version_policy("1.2.0", "1.2.1", &compatible).is_err());
        assert!(check_version_policy("1.2.0", "1.3.0", &compatible).is_ok());
    }

    #[test]
    fn lock_roundtrip_and_digest_guard() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("release.lock.json");
        let snapshot = snapshot();
        let lock = ReleaseLock {
            lock_version: RELEASE_LOCK_VERSION,
            language_version: "0.1.0".to_string(),
            schema_digest: snapshot.digest(),
            schema: snapshot,
            nlu: None,
            recovery: None,
            identity: None,
            fuzz: None,
            entailment: None,
        };
        lock.save(&path).unwrap();
        assert_eq!(ReleaseLock::load(&path).unwrap(), lock);

        let mut tampered = lock.clone();
        tampered.schema_digest = "deadbeef".to_string();
        tampered.save(&path).unwrap();
        assert!(ReleaseLock::load(&path).is_err());
    }

    #[test]
    fn release_check_bootstraps_and_ratchets_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        let lock_path = dir.path().join("release.lock.json");
        let config = sample_config();
        let registry = PluginRegistry::empty();
        let options = CheckOptions {
            config_file: "config.toml",
            lock_path: lock_path.to_str().unwrap(),
            language_version: Some("0.1.0"),
            update_lock: true,
            skip_evals: true,
        };
        let report = check(&config, &registry, &options).unwrap();
        assert!(report.passed, "{}", report.render_text());
        assert!(report.lock_updated);
        assert!(lock_path.exists());
        let lock = ReleaseLock::load(&lock_path).unwrap();
        assert!(
            lock.entailment.is_some(),
            "首次固化基线应包含蕴含评测"
        );

        let second = check(&config, &registry, &options).unwrap();
        assert!(second.passed, "{}", second.render_text());

        let mut lock = ReleaseLock::load(&lock_path).unwrap();
        let prompt_id = lock
            .schema
            .prompts
            .keys()
            .next()
            .expect("至少一个模板")
            .clone();
        lock.schema.prompts.remove(&prompt_id);
        lock.schema_digest = lock.schema.digest();
        lock.save(&lock_path).unwrap();
        let drift = check(&config, &registry, &options).unwrap();
        assert!(!drift.passed, "删模板必须被 schema 门拦住");
        assert!(drift
            .gates
            .iter()
            .any(|gate| gate.name == "schema_diff" && !gate.passed));
        assert!(!drift.lock_updated);
    }
}

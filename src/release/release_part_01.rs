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

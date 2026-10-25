//! CC：配置契约（Config Contract）——把 config.toml 当作语言器官的源码来编译。
//!
//! config.toml 不是参数表，而是一门小语言的源码：意图模式、实体/属性模式、
//! `query_type`、谓词画像、类型格、同义词与阈值共同定义了「系统能听懂什么、
//! 能答什么」。手写这门语言时，最危险的失败不是报错，而是**安静地不会说话**：
//! 规则永不命中、查询永不生成、回答退回兜底，却没有一行日志。
//!
//! 本模块是这门语言的编译前端：把配置编译成能力图并做静态检查，覆盖
//! **引用完整性**（悬空 `entity_type` / `query_type` / `inverse` / 类型名）、
//! **能力可达性**（规则能否捕获到 handler 需要的实体槽位）、**形状正确性**
//! （别名冲突、子类环、同义词环、互斥自反）与**阈值在域内**（越界会被静默钳制）。
//!
//! 约定：
//! - `Error`：契约被破坏——引用不存在、成环、冲突、域外阈值；功能必然或极可能静默失效；
//! - `Warning`：配置将被钳制/退化/可疑——越界钳制、未声明属性、字面量词性、空槽位等；
//! - 纯函数、确定性顺序：同配置同输出字节，可直接作为回归断言对象；
//! - 校验只读，不在启动时改变任何运行路径（可先 lint 暴露断链，再决定是否接管）。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::binding_frames::FrameProgram;
use crate::trace_graph;
use crate::SystemConfig;

/// 契约版本：字段/检查语义变化时递增，便于基线与 CI 对齐。
pub const CONFIG_CONTRACT_VERSION: u32 = 1;

/// `generate_prolog_query` 能直接生成 Prolog 查询的 `query_type`。
///
/// 这是能力图的第一张表：与 `prolog_engine::PrologEngine::generate_prolog_query`
/// 的分支一一对应；`prolog_generated_list_matches_engine` 测试用真实调用守卫，
/// 防止表格与实现漂移。
pub const GENERATED_QUERY_TYPES: &[&str] = &[
    "age_query",
    "altitude_query",
    "author_query",
    "birth_year_query",
    "capital_query",
    "champion_query",
    "chemical_formula_query",
    "concept_query",
    "count_query",
    "largest_query",
    "property_query",
    "quantity_query",
    "relation_query",
    "spouse_query",
];

/// `get_direct_answer` 能用 SPARQL 直接回答的 `query_type`（命中时由 SPARQL 直接回答）。
pub const DIRECT_ANSWER_QUERY_TYPES: &[&str] = &[
    "altitude_query",
    "author_query",
    "birth_year_query",
    "capital_query",
    "cause_query",
    "champion_query",
    "chemical_formula_query",
    "comparison_query",
    "concept_query",
    "explanation_query",
    "impact_query",
    "largest_query",
    "philosophy_query",
    "planet_query",
    "principle_query",
    "quantity_query",
    "reason_query",
    "time_before_query",
];

/// `format_prolog_results` 有专门措辞的 `query_type`（只影响输出，不影响可达性）。
pub const FORMATTED_QUERY_TYPES: &[&str] = &[
    "age_query",
    "altitude_query",
    "author_query",
    "birth_year_query",
    "capital_query",
    "cause_query",
    "champion_query",
    "chemical_formula_query",
    "comparison_query",
    "concept_query",
    "count_query",
    "explanation_query",
    "impact_query",
    "largest_query",
    "philosophy_query",
    "planet_query",
    "population_query",
    "principle_query",
    "quantity_query",
    "reason_query",
    "relation_query",
    "spouse_query",
    "time_before_query",
];

/// 主流程真正可达的 `query_type` 全集：直接回答 ∪ 查询生成。
///
/// `format_prolog_results` 单独出现的类型（如 `population_query`）不可达——
/// 没有前两步就没有结果可格式化。
pub fn supported_query_types() -> BTreeSet<&'static str> {
    DIRECT_ANSWER_QUERY_TYPES
        .iter()
        .chain(GENERATED_QUERY_TYPES.iter())
        .copied()
        .collect()
}

pub fn is_query_type_supported(query_type: &str) -> bool {
    DIRECT_ANSWER_QUERY_TYPES.contains(&query_type) || GENERATED_QUERY_TYPES.contains(&query_type)
}

/// handler 所需的最小实体槽位数；`None` 表示该类型不在能力图上。
fn required_entity_arity(query_type: &str) -> Option<usize> {
    match query_type {
        "relation_query" | "comparison_query" => Some(2),
        // 行星/最大的直接回答是固定 SPARQL，不需要实体。
        "planet_query" | "largest_query" => Some(0),
        q if is_query_type_supported(q) => Some(1),
        _ => None,
    }
}

/// 严重度：Error 破坏契约，Warning 退化/可疑。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractSeverity {
    Error,
    Warning,
}

impl ContractSeverity {
    pub fn label(&self) -> &'static str {
        match self {
            ContractSeverity::Error => "ERROR",
            ContractSeverity::Warning => "WARN ",
        }
    }
}

/// 一条契约问题：稳定机器码 + 配置坐标 + 人话。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigIssue {
    pub severity: ContractSeverity,
    /// 稳定机器码（snake_case），供 CI/基线按类匹配。
    pub code: String,
    /// 配置坐标，如 `prolog_rules[5].query_type`。
    pub path: String,
    pub message: String,
}

impl ConfigIssue {
    /// 单行文本（CLI 与日志共用同一措辞）。
    pub fn line(&self) -> String {
        format!(
            "{} [{}] {}：{}",
            self.severity.label(),
            self.code,
            self.path,
            self.message
        )
    }
}

#[derive(Default)]
struct Collector {
    issues: Vec<ConfigIssue>,
}

impl Collector {
    fn error(&mut self, code: &str, path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ConfigIssue {
            severity: ContractSeverity::Error,
            code: code.to_string(),
            path: path.into(),
            message: message.into(),
        });
    }

    fn warn(&mut self, code: &str, path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ConfigIssue {
            severity: ContractSeverity::Warning,
            code: code.to_string(),
            path: path.into(),
            message: message.into(),
        });
    }
}

/// 编译校验整个配置：问题按固定顺序输出（配置顺序 + 字典序）。
pub fn lint(config: &SystemConfig) -> Vec<ConfigIssue> {
    let mut c = Collector::default();
    check_system(&mut c, config);
    check_messages(&mut c, config);
    check_nlp(&mut c, config);
    check_api(&mut c, config);
    check_entity_types(&mut c, config);
    let type_names = collect_type_names(config);
    check_extraction_rules(&mut c, config, &type_names);
    check_semantic_grammar(&mut c, config, &type_names);
    check_prolog_rules(&mut c, config);
    check_predicate_profiles(&mut c, config, &type_names);
    check_entity_profiles(&mut c, config, &type_names);
    check_type_system(&mut c, config);
    check_synonyms(&mut c, config);
    check_reasoning(&mut c, config);
    check_confidence(&mut c, config);
    check_soft_route(&mut c, config);
    check_trace_graph(&mut c, config);
    check_dialogue_frames(&mut c, config);
    c.issues
}

/// 解析并校验 TOML 文本；语法错误以 `Err` 返回（调用方决定如何呈现）。
pub fn lint_str(source: &str) -> Result<Vec<ConfigIssue>, String> {
    let config: SystemConfig = toml::from_str(source).map_err(|e| e.to_string())?;
    Ok(lint(&config))
}

pub fn error_count(issues: &[ConfigIssue]) -> usize {
    issues
        .iter()
        .filter(|issue| issue.severity == ContractSeverity::Error)
        .count()
}

pub fn warning_count(issues: &[ConfigIssue]) -> usize {
    issues
        .iter()
        .filter(|issue| issue.severity == ContractSeverity::Warning)
        .count()
}

pub fn has_errors(issues: &[ConfigIssue]) -> bool {
    error_count(issues) > 0
}

/// 多行文本报告（末尾不带换行）；问题为空时给出通过措辞。
pub fn render_text(issues: &[ConfigIssue]) -> String {
    let mut lines = Vec::with_capacity(issues.len() + 1);
    lines.push(format!(
        "配置契约 v{}（CC）：{} 个错误、{} 个警告",
        CONFIG_CONTRACT_VERSION,
        error_count(issues),
        warning_count(issues)
    ));
    if issues.is_empty() {
        lines.push("契约检查通过：引用完整、能力可达、阈值在域内。".to_string());
        return lines.join("\n");
    }
    for issue in issues {
        lines.push(issue.line());
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// 检查器
// ---------------------------------------------------------------------------

fn check_system(c: &mut Collector, config: &SystemConfig) {
    if config.system.oxigraph_store_path.trim().is_empty() {
        c.warn(
            "empty_store_path",
            "system.oxigraph_store_path",
            "存储路径为空，打开知识库时将以当前目录为数据目录",
        );
    }
    if config.system.default_language.trim().is_empty() {
        c.warn(
            "empty_default_language",
            "system.default_language",
            "默认语言为空",
        );
    }
    // `oxigraph_endpoint` 曾被读进 SystemSettings，却没有任何运行路径消费它：
    // 存储始终走内嵌 Oxigraph（`oxigraph_store_path`），HTTP 端点从未被拨号。
    // 保留字段是为了兼容既有 config.toml，但必须如实告诉作者它不会生效，
    // 否则「配了却不说」会让人误以为在指定远程存储。
    if !config.system.oxigraph_endpoint.trim().is_empty() {
        c.warn(
            "inert_oxigraph_endpoint",
            "system.oxigraph_endpoint",
            "该端点在运行时不生效：存储始终使用内嵌 Oxigraph（见 system.oxigraph_store_path），此字段仅作兼容保留",
        );
    }
}

fn check_messages(c: &mut Collector, config: &SystemConfig) {
    let named: [(&str, &str, &str); 9] = [
        ("startup", "messages.startup", &config.messages.startup),
        ("ready", "messages.ready", &config.messages.ready),
        (
            "query_prompt",
            "messages.query_prompt",
            &config.messages.query_prompt,
        ),
        (
            "import_prompt",
            "messages.import_prompt",
            &config.messages.import_prompt,
        ),
        (
            "quit_prompt",
            "messages.quit_prompt",
            &config.messages.quit_prompt,
        ),
        (
            "importing",
            "messages.importing",
            &config.messages.importing,
        ),
        (
            "no_knowledge_extracted",
            "messages.no_knowledge_extracted",
            &config.messages.no_knowledge_extracted,
        ),
        (
            "cannot_understand",
            "messages.cannot_understand",
            &config.messages.cannot_understand,
        ),
        (
            "process_error",
            "messages.process_error",
            &config.messages.process_error,
        ),
    ];
    for (name, path, value) in named {
        if value.trim().is_empty() {
            c.warn(
                "empty_message",
                path.to_string(),
                format!("界面文案「{}」为空，用户将看到空白提示", name),
            );
        }
    }
}

fn check_nlp(c: &mut Collector, config: &SystemConfig) {
    if config.nlp.import_keywords.is_empty() {
        c.warn(
            "no_import_keywords",
            "nlp.import_keywords",
            "导入关键词为空：任何输入都会被当作查询，无法从交互输入录入知识",
        );
    }
    if let Some(threshold) = config.nlp.nlu_confidence_threshold {
        check_probability(
            c,
            threshold,
            "nlp.nlu_confidence_threshold",
            "NLU 置信度阈值",
        );
    }
    if let Some(threshold) = config.nlp.semantic_similarity_threshold {
        check_probability(
            c,
            threshold,
            "nlp.semantic_similarity_threshold",
            "语义相似度阈值",
        );
    }
    if config.nlp.max_entity_candidates == Some(0) {
        c.warn(
            "zero_entity_candidates",
            "nlp.max_entity_candidates",
            "最大实体候选数为 0：歧义实体将没有可选项",
        );
    }
    if config.nlp.context_window_size == Some(0) {
        c.warn(
            "zero_context_window",
            "nlp.context_window_size",
            "上下文窗口为 0：实体抽取将看不到任何上下文",
        );
    }
    for (word, tag) in config.nlp.pos_dict.iter() {
        if word.trim().is_empty() {
            c.warn("empty_pos_dict_key", "nlp.pos_dict", "词性词典存在空词条");
        }
        if tag.trim().is_empty() {
            c.warn(
                "empty_pos_tag",
                format!("nlp.pos_dict[{}]", word),
                "词性标签为空：该词条不会命中任何词性模式",
            );
        } else if !tag
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            c.warn(
                "unknown_pos_tag",
                format!("nlp.pos_dict[{}]", word),
                format!(
                    "词性标签「{}」不是常见标签形式（小写字母/数字），可能永不命中",
                    tag
                ),
            );
        }
    }
    if config.nlp.name_patterns.full_names.is_empty()
        && config.nlp.name_patterns.prefixes.is_empty()
        && config.nlp.name_patterns.suffixes.is_empty()
        && config.nlp.name_patterns.profession_suffix.is_empty()
    {
        c.warn(
            "no_name_patterns",
            "nlp.name_patterns",
            "人名模式全为空：中文人名识别将只依赖分词器默认行为",
        );
    }
}

fn check_api(c: &mut Collector, config: &SystemConfig) {
    let Some(api) = &config.api else {
        return;
    };
    if api.ownthink_timeout_seconds == 0 {
        c.warn(
            "zero_api_timeout",
            "api.ownthink_timeout_seconds",
            "外部知识 API 超时为 0 秒：请求将立即超时",
        );
    }
    if api.max_ambiguous_entities == 0 {
        c.warn(
            "zero_ambiguous_entities",
            "api.max_ambiguous_entities",
            "歧义实体展示上限为 0：消歧候选将不可见",
        );
    }
    if api.external_ledger_capacity == 0 {
        c.warn(
            "zero_ledger_capacity",
            "api.external_ledger_capacity",
            "外链名册容量为 0：外部知识无法缓存",
        );
    }
}

/// 类型名的归一化坐标（类型体系大小写不敏感）。
fn fold(name: &str) -> String {
    name.trim().to_lowercase()
}

/// 已知类型全集：实体类型标签 + 类型体系全部出现过的名字 + 实体画像类型。
fn collect_type_names(config: &SystemConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for entity_type in &config.entity_types {
        names.insert(fold(&entity_type.label));
    }
    let types = &config.type_system;
    for alias in &types.aliases {
        names.insert(fold(&alias.alias));
        names.insert(fold(&alias.canonical));
    }
    for subclass in &types.subclasses {
        names.insert(fold(&subclass.sub));
        names.insert(fold(&subclass.sup));
    }
    for disjoint in &types.disjoint {
        names.insert(fold(&disjoint.left));
        names.insert(fold(&disjoint.right));
    }
    names.remove("");
    names
}

fn check_entity_types(c: &mut Collector, config: &SystemConfig) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (index, entity_type) in config.entity_types.iter().enumerate() {
        let path = format!("entity_types[{}]", index);
        if entity_type.label.trim().is_empty() {
            c.error(
                "empty_entity_type_label",
                format!("{}.label", path),
                "实体类型标签为空",
            );
        } else if let Some(first) = seen.insert(fold(&entity_type.label), index) {
            c.error(
                "duplicate_entity_type",
                format!("{}.label", path),
                format!(
                    "实体类型「{}」与 entity_types[{}] 重复（类型名大小写不敏感）",
                    entity_type.label, first
                ),
            );
        }
        if entity_type.properties.is_empty() {
            c.warn(
                "type_without_properties",
                path.clone(),
                format!("实体类型「{}」未声明任何属性", entity_type.label),
            );
        }
        let mut property_names: BTreeMap<String, usize> = BTreeMap::new();
        for (property_index, property) in entity_type.properties.iter().enumerate() {
            let property_path = format!("{}.properties[{}]", path, property_index);
            if property.name.trim().is_empty() {
                c.error(
                    "empty_property_name",
                    format!("{}.name", property_path),
                    "属性名为空",
                );
                continue;
            }
            if let Some(first) = property_names.insert(fold(&property.name), property_index) {
                c.error(
                    "duplicate_property",
                    format!("{}.name", property_path),
                    format!(
                        "属性「{}」与 {}.properties[{}] 重复（属性名大小写不敏感）",
                        property.name, path, first
                    ),
                );
            }
            if !KNOWN_DATA_TYPES.contains(&fold(&property.data_type).as_str()) {
                c.warn(
                    "unknown_data_type",
                    format!("{}.data_type", property_path),
                    format!(
                        "数据类型「{}」不在常见集合（string/integer/float/boolean/date/datetime/text）中",
                        property.data_type
                    ),
                );
            }
            if !property.extraction_patterns.is_empty() {
                c.warn(
                    "inert_property_extraction_patterns",
                    format!("{}.extraction_patterns", property_path),
                    format!(
                        "属性「{}」的 extraction_patterns 当前不会被读取：抽取由 [[extraction_rules]].property_mappings 驱动，这些模式不会生效",
                        property.name
                    ),
                );
            }
        }
    }
}

const KNOWN_DATA_TYPES: &[&str] = &[
    "string", "integer", "int", "float", "double", "number", "boolean", "bool", "date", "datetime",
    "time", "text",
];

/// 常见中文词性标签（ICTCLAS/jieba 体系）与自定义词典允许的形态。
/// 抽取规则里的 `pos_pattern` 是与分词结果做严格相等的，写错的标签是死规则。
const KNOWN_POS_TAGS: &[&str] = &[
    "ag", "an", "ad", "a", "b", "c", "d", "df", "dg", "e", "eng", "f", "g", "h", "i", "j", "k",
    "l", "m", "mg", "mq", "n", "ng", "nr", "nrfg", "nrt", "ns", "nt", "nz", "o", "p", "q", "r",
    "rg", "rr", "rz", "s", "t", "tg", "u", "ud", "ug", "uj", "ul", "uv", "uz", "v", "vd", "vg",
    "vi", "vn", "vq", "x", "y", "z", "zg", "unknown", "w", "un",
];

fn check_extraction_rules(c: &mut Collector, config: &SystemConfig, type_names: &BTreeSet<String>) {
    let mut declared_properties: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entity_type in &config.entity_types {
        declared_properties.insert(
            fold(&entity_type.label),
            entity_type
                .properties
                .iter()
                .map(|property| fold(&property.name))
                .collect(),
        );
    }
    let mut rule_names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, rule) in config.extraction_rules.iter().enumerate() {
        let path = format!("extraction_rules[{}]", index);
        if rule.name.trim().is_empty() {
            c.error(
                "empty_extraction_rule_name",
                format!("{}.name", path),
                "提取规则名为空",
            );
        } else if let Some(first) = rule_names.insert(rule.name.trim().to_string(), index) {
            c.warn(
                "duplicate_extraction_rule_name",
                format!("{}.name", path),
                format!(
                    "提取规则名「{}」与 extraction_rules[{}] 重复",
                    rule.name, first
                ),
            );
        }
        let entity_type_key = fold(&rule.entity_type);
        let entity_type_known =
            !entity_type_key.is_empty() && type_names.contains(&entity_type_key);
        if rule.entity_type.trim().is_empty() {
            c.error(
                "empty_entity_type",
                format!("{}.entity_type", path),
                "提取规则的 entity_type 为空：抽取结果无法归型",
            );
        } else if !entity_type_known {
            c.error(
                "dangling_entity_type",
                format!("{}.entity_type", path),
                format!(
                    "未知实体类型「{}」：规则永不产出（已知类型：{}）",
                    rule.entity_type,
                    type_names.iter().cloned().collect::<Vec<_>>().join("、")
                ),
            );
        }
        if rule.patterns.is_empty() {
            c.error(
                "rule_without_patterns",
                path.clone(),
                format!("提取规则「{}」没有任何模式，永不命中", rule.name),
            );
        }
        for (pattern_index, pattern) in rule.patterns.iter().enumerate() {
            let pattern_path = format!("{}.patterns[{}]", path, pattern_index);
            if pattern.property_mappings.is_empty() {
                c.error(
                    "pattern_without_property_mappings",
                    pattern_path.clone(),
                    "模式没有任何属性映射：即使命中也不会产出知识",
                );
            }
            if pattern.pos_pattern.is_empty() {
                c.warn(
                    "empty_pos_pattern",
                    format!("{}.pos_pattern", pattern_path),
                    "词性模式为空：将匹配任意输入（含无关句子）",
                );
            }
            let mut tokens: Vec<&str> = pattern
                .pos_pattern
                .iter()
                .map(|token| token.trim())
                .collect();
            tokens.sort_unstable();
            tokens.dedup();
            for token in tokens {
                if token.is_empty() {
                    c.error(
                        "empty_pos_token",
                        format!("{}.pos_pattern", pattern_path),
                        "词性模式含空 token：该规则永不命中",
                    );
                    continue;
                }
                if token == "*" {
                    continue;
                }
                if !KNOWN_POS_TAGS.contains(&token)
                    && !config.nlp.pos_dict.values().any(|tag| tag == token)
                {
                    c.warn(
                        "unknown_pos_token",
                        format!("{}.pos_pattern", pattern_path),
                        format!(
                            "词性 token「{}」不在常见标签与 nlp.pos_dict 中：与词性做严格相等，规则可能永不命中",
                            token
                        ),
                    );
                }
            }
            if entity_type_known {
                if let Some(properties) = declared_properties.get(&entity_type_key) {
                    for property_name in pattern.property_mappings.keys() {
                        if !properties.contains(&fold(property_name)) {
                            c.warn(
                                "undeclared_property_mapping",
                                format!("{}.property_mappings", pattern_path),
                                format!(
                                    "属性「{}」未在实体类型「{}」中声明（仍会入库，但类型契约与必填校验看不到它）",
                                    property_name, rule.entity_type
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// 双向语义规则包的静态检查：坏规则必须在 lint 中可见，而不是等运行时空转。
fn check_semantic_grammar(
    c: &mut Collector,
    config: &SystemConfig,
    type_names: &BTreeSet<String>,
) {
    let grammar = &config.semantic_grammar;
    if !grammar.enabled {
        if !grammar.rules.is_empty() {
            c.warn(
                "semantic_grammar_disabled",
                "semantic_grammar.enabled",
                "已配置双向规则但开关为 false：规则只可预览，运行时不参与抽取与生成",
            );
        }
        return;
    }
    if grammar.rules.is_empty() {
        c.warn(
            "semantic_grammar_without_rules",
            "semantic_grammar.rules",
            "双向语义语法已启用但没有任何规则",
        );
        return;
    }
    let pack = crate::semantic_grammar::RulePack::compile(grammar.rules.clone());
    for error in pack.errors() {
        c.error(
            "semantic_rule_compile_error",
            "semantic_grammar.rules",
            error.render(),
        );
    }
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, rule) in grammar.rules.iter().enumerate() {
        let path = format!("semantic_grammar.rules[{}]", index);
        let name = rule.name.trim();
        if name.is_empty() {
            c.error("empty_semantic_rule_name", format!("{}.name", path), "规则名为空");
        } else if let Some(first) = names.insert(name.to_string(), index) {
            c.warn(
                "duplicate_semantic_rule_name",
                format!("{}.name", path),
                format!("规则名「{}」与 semantic_grammar.rules[{}] 重复", name, first),
            );
        }
        if rule.predicate.trim().is_empty() {
            c.error(
                "empty_semantic_predicate",
                format!("{}.predicate", path),
                "predicate 为空：抽取结果没有谓词",
            );
        }
        if let Some(entity_type) = rule.entity_type.as_deref().map(str::trim) {
            if !entity_type.is_empty() && !type_names.contains(&fold(entity_type)) {
                c.warn(
                    "dangling_semantic_entity_type",
                    format!("{}.entity_type", path),
                    format!("实体类型「{}」未声明：抽取结果仍会入库，但类型契约看不到它", entity_type),
                );
            }
        }
        if rule.realize.is_none() && rule.relation {
            c.warn(
                "semantic_relation_without_realize",
                format!("{}.realize", path),
                "关系规则没有反向模板：抽取可用，但生成不会复用这条规则",
            );
        }
    }
}

/// `pattern_from_strings` 中会被解释为词性（而非字面词）的 token。
const PROLOG_POS_TOKENS: &[&str] = &["r", "n", "nr", "ns", "m", "q", "v", "u", "f", "a", "d", "*"];

/// 模式能捕获到实体槽位的 token（与 `match_pattern_for_prolog` 的捕获分支一致）。
fn captured_entity_slots(pattern: &[String]) -> usize {
    pattern
        .iter()
        .filter(|token| matches!(token.as_str(), "n" | "nr" | "ns" | "m"))
        .count()
}

fn check_prolog_rules(c: &mut Collector, config: &SystemConfig) {
    let mut rule_names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, rule) in config.prolog_rules.iter().enumerate() {
        let path = format!("prolog_rules[{}]", index);
        if rule.name.trim().is_empty() {
            c.error(
                "empty_prolog_rule_name",
                format!("{}.name", path),
                "查询规则名为空",
            );
        } else if let Some(first) = rule_names.insert(rule.name.trim().to_string(), index) {
            c.warn(
                "duplicate_prolog_rule_name",
                format!("{}.name", path),
                format!("查询规则名「{}」与 prolog_rules[{}] 重复", rule.name, first),
            );
        }
        if rule.pattern.is_empty() {
            c.error(
                "empty_query_pattern",
                format!("{}.pattern", path),
                format!(
                    "查询规则「{}」模式为空：会匹配任意输入并抢占后续规则",
                    rule.name
                ),
            );
        }
        let query_type = rule.query_type.trim();
        if query_type.is_empty() {
            c.error(
                "empty_query_type",
                format!("{}.query_type", path),
                "query_type 为空：规则命中后无法生成查询",
            );
        } else if !is_query_type_supported(query_type) {
            c.error(
                "dangling_query_type",
                format!("{}.query_type", path),
                format!(
                    "未知 query_type「{}」：命中后只会打印「无法生成有效的Prolog查询」；可用类型：{}",
                    query_type,
                    supported_query_types()
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                ),
            );
        }
        for token in &rule.pattern {
            let token = token.trim();
            if token.is_empty() {
                c.error(
                    "empty_pattern_token",
                    format!("{}.pattern", path),
                    "模式含空 token：空字面量会吞掉任意词，规则将过度匹配",
                );
                continue;
            }
            // 只对「像词性标签的短小写 ASCII token」报警：中文/英文实词是合法字面量。
            let looks_like_tag = token.chars().count() <= 3
                && token
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
                && !PROLOG_POS_TOKENS.contains(&token)
                && !config.nlp.pos_dict.contains_key(token);
            if looks_like_tag {
                c.warn(
                    "pattern_token_treated_as_word",
                    format!("{}.pattern", path),
                    format!(
                        "token「{}」不是已知词性（{}），将按字面词匹配：可能永不命中",
                        token,
                        PROLOG_POS_TOKENS.join("/")
                    ),
                );
            }
        }
        if let Some(arity) = required_entity_arity(query_type) {
            let captured = captured_entity_slots(&rule.pattern);
            if captured < arity {
                c.warn(
                    "insufficient_entity_slots",
                    format!("{}.pattern", path),
                    format!(
                        "query_type「{}」至少需要 {} 个实体槽位，当前模式只能捕获 {} 个：命中也无法生成查询",
                        query_type, arity, captured
                    ),
                );
            }
        }
    }
}

fn check_predicate_profiles(
    c: &mut Collector,
    config: &SystemConfig,
    type_names: &BTreeSet<String>,
) {
    let mut canonical_names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, profile) in config.predicate_profiles.iter().enumerate() {
        let path = format!("predicate_profiles[{}]", index);
        let name = profile.name.trim();
        if name.is_empty() {
            c.error(
                "empty_predicate_name",
                format!("{}.name", path),
                "谓词规范名为空",
            );
            continue;
        }
        if let Some(first) = canonical_names.insert(name.to_string(), index) {
            c.error(
                "duplicate_predicate_profile",
                format!("{}.name", path),
                format!("谓词「{}」与 predicate_profiles[{}] 重复", name, first),
            );
        }
    }

    let mut alias_owner: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut inverse_by_name: BTreeMap<String, String> = BTreeMap::new();
    for (index, profile) in config.predicate_profiles.iter().enumerate() {
        let path = format!("predicate_profiles[{}]", index);
        let name = profile.name.trim();
        if name.is_empty() {
            continue;
        }
        if let Some(inverse) = &profile.inverse {
            let inverse = inverse.trim();
            if inverse.is_empty() {
                c.warn(
                    "empty_inverse",
                    format!("{}.inverse", path),
                    "互逆谓词为空：该声明被忽略",
                );
            } else if inverse == name {
                c.error(
                    "self_inverse",
                    format!("{}.inverse", path),
                    format!("谓词「{}」的互逆谓词是自身", name),
                );
            } else {
                inverse_by_name.insert(name.to_string(), inverse.to_string());
            }
        }
    }

    let inverse_owner: BTreeMap<&str, &str> = config
        .predicate_profiles
        .iter()
        .filter_map(|profile| {
            let name = profile.name.trim();
            let inverse = profile
                .inverse
                .as_deref()
                .map(str::trim)
                .filter(|i| !i.is_empty())?;
            if name.is_empty() || inverse == name {
                None
            } else {
                Some((inverse, name))
            }
        })
        .collect();

    for (index, profile) in config.predicate_profiles.iter().enumerate() {
        let path = format!("predicate_profiles[{}]", index);
        let name = profile.name.trim();
        if name.is_empty() {
            continue;
        }
        let mut local_aliases: BTreeSet<String> = BTreeSet::new();
        for alias in &profile.aliases {
            let alias = alias.trim();
            if alias.is_empty() {
                c.warn(
                    "empty_predicate_alias",
                    format!("{}.aliases", path),
                    "空别名：该声明被忽略",
                );
                continue;
            }
            if alias == name {
                c.warn(
                    "self_alias",
                    format!("{}.aliases", path),
                    format!("谓词「{}」把规范名自身声明为别名", name),
                );
                continue;
            }
            if !local_aliases.insert(alias.to_string()) {
                c.warn(
                    "duplicate_predicate_alias",
                    format!("{}.aliases", path),
                    format!("别名「{}」重复声明", alias),
                );
                continue;
            }
            if let Some(canonical_index) = canonical_names.get(alias) {
                if canonical_index != &index {
                    let canonical = config.predicate_profiles[*canonical_index].name.trim();
                    c.error(
                        "alias_shadows_canonical",
                        format!("{}.aliases", path),
                        format!(
                            "别名「{}」与 predicate_profiles[{}] 的规范名「{}」冲突：归一方向取决于声明顺序",
                            alias, canonical_index, canonical
                        ),
                    );
                    continue;
                }
            }
            match alias_owner.get(alias) {
                Some((owner_index, owner)) if owner_index != &index => {
                    c.error(
                        "predicate_alias_conflict",
                        format!("{}.aliases", path),
                        format!("别名「{}」已归属谓词「{}」", alias, owner),
                    );
                }
                Some(_) => {}
                None => {
                    alias_owner.insert(alias.to_string(), (index, name.to_string()));
                }
            }
            if let Some(owner) = inverse_owner.get(alias) {
                if owner != &name {
                    c.error(
                        "alias_conflicts_with_inverse",
                        format!("{}.aliases", path),
                        format!("别名「{}」已作为谓词「{}」的互逆谓词绑定", alias, owner),
                    );
                }
            }
        }
        for domain in &profile.domain {
            if domain.trim().is_empty() {
                c.warn(
                    "empty_domain_type",
                    format!("{}.domain", path),
                    "定义域含空类型名",
                );
            } else if !type_names.contains(&fold(domain)) {
                c.warn(
                    "unknown_domain_type",
                    format!("{}.domain", path),
                    format!(
                        "定义域类型「{}」未在类型体系中出现：若为笔误将无法校验",
                        domain
                    ),
                );
            }
        }
        for range in &profile.range {
            if range.trim().is_empty() {
                c.warn(
                    "empty_range_type",
                    format!("{}.range", path),
                    "值域含空类型名",
                );
            } else if !type_names.contains(&fold(range)) {
                c.warn(
                    "unknown_range_type",
                    format!("{}.range", path),
                    format!(
                        "值域类型「{}」未在类型体系中出现：若为笔误将无法校验",
                        range
                    ),
                );
            }
        }
        if let Some(parent) = &profile.subproperty_of {
            let parent = parent.trim();
            if parent.is_empty() {
                c.warn(
                    "empty_subproperty_of",
                    format!("{}.subproperty_of", path),
                    "子属性父谓词为空：该声明被忽略",
                );
            } else if parent == name {
                c.error(
                    "self_subproperty",
                    format!("{}.subproperty_of", path),
                    format!("谓词「{}」声明自身为其子属性", name),
                );
            } else if !canonical_names.contains_key(parent) {
                c.warn(
                    "unknown_subproperty_parent",
                    format!("{}.subproperty_of", path),
                    format!(
                        "父谓词「{}」未在 predicate_profiles 中声明规范名：若为笔误将无法沿层级展开",
                        parent
                    ),
                );
            }
        }
    }

    for (index, profile) in config.predicate_profiles.iter().enumerate() {
        let name = profile.name.trim();
        let Some(inverse) = inverse_by_name.get(name) else {
            continue;
        };
        if let Some(back) = inverse_by_name.get(inverse) {
            if back != name {
                c.warn(
                    "inverse_not_reciprocated",
                    format!("predicate_profiles[{}].inverse", index),
                    format!(
                        "「{}」声明互逆为「{}」，但「{}」声明互逆为「{}」：方向可能不符预期",
                        name, inverse, inverse, back
                    ),
                );
            }
        }
    }

    if config.reasoning.require_predicate_admission && config.predicate_profiles.is_empty() {
        c.warn(
            "strict_admission_without_profiles",
            "reasoning.require_predicate_admission",
            "严格准入已开启但没有任何谓词画像：所有新谓词都不会落库",
        );
    }
}

fn check_entity_profiles(c: &mut Collector, config: &SystemConfig, type_names: &BTreeSet<String>) {
    let mut canonical_names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, profile) in config.entity_profiles.iter().enumerate() {
        let path = format!("entity_profiles[{}]", index);
        let name = profile.name.trim();
        if name.is_empty() {
            c.error(
                "empty_entity_profile_name",
                format!("{}.name", path),
                "实体规范名为空",
            );
            continue;
        }
        if let Some(first) = canonical_names.insert(name.to_string(), index) {
            c.error(
                "duplicate_entity_profile",
                format!("{}.name", path),
                format!("实体「{}」与 entity_profiles[{}] 重复", name, first),
            );
        }
        for etype in &profile.etype {
            if etype.trim().is_empty() {
                c.warn(
                    "empty_entity_profile_type",
                    format!("{}.etype", path),
                    "实体画像含空类型名",
                );
            } else if !type_names.contains(&fold(etype)) {
                c.warn(
                    "unknown_entity_profile_type",
                    format!("{}.etype", path),
                    format!("实体画像类型「{}」未在类型体系中出现", etype),
                );
            }
        }
    }

    let mut alias_owner: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (index, profile) in config.entity_profiles.iter().enumerate() {
        let path = format!("entity_profiles[{}]", index);
        let name = profile.name.trim();
        if name.is_empty() {
            continue;
        }
        let mut local_aliases: BTreeSet<String> = BTreeSet::new();
        for alias in &profile.aliases {
            let alias = alias.trim();
            if alias.is_empty() {
                c.warn(
                    "empty_entity_alias",
                    format!("{}.aliases", path),
                    "空别名：该声明被忽略",
                );
                continue;
            }
            if alias == name {
                c.warn(
                    "self_entity_alias",
                    format!("{}.aliases", path),
                    format!("实体「{}」把规范名自身声明为别名", name),
                );
                continue;
            }
            if !local_aliases.insert(alias.to_string()) {
                c.warn(
                    "duplicate_entity_alias",
                    format!("{}.aliases", path),
                    format!("别名「{}」重复声明", alias),
                );
                continue;
            }
            if let Some(canonical_index) = canonical_names.get(alias) {
                if canonical_index != &index {
                    let canonical = config.entity_profiles[*canonical_index].name.trim();
                    c.error(
                        "entity_alias_shadows_canonical",
                        format!("{}.aliases", path),
                        format!(
                            "别名「{}」与 entity_profiles[{}] 的规范名「{}」冲突",
                            alias, canonical_index, canonical
                        ),
                    );
                    continue;
                }
            }
            match alias_owner.get(alias) {
                Some((owner_index, owner)) if owner_index != &index => {
                    c.error(
                        "entity_alias_conflict",
                        format!("{}.aliases", path),
                        format!("别名「{}」已归属实体「{}」", alias, owner),
                    );
                }
                Some(_) => {}
                None => {
                    alias_owner.insert(alias.to_string(), (index, name.to_string()));
                }
            }
        }
    }
}

fn check_type_system(c: &mut Collector, config: &SystemConfig) {
    let types = &config.type_system;
    let mut alias_map: BTreeMap<String, String> = BTreeMap::new();
    for (index, alias) in types.aliases.iter().enumerate() {
        let path = format!("type_system.aliases[{}]", index);
        let alias_key = fold(&alias.alias);
        let canonical_key = fold(&alias.canonical);
        if alias_key.is_empty() || canonical_key.is_empty() {
            c.error(
                "empty_type_alias",
                path,
                "类型别名的 alias/canonical 为空：该声明被忽略",
            );
            continue;
        }
        if alias_key == canonical_key {
            c.warn(
                "self_type_alias",
                path,
                format!("类型别名「{}」与规范名相同：该声明无效果", alias.alias),
            );
            continue;
        }
        match alias_map.get(&alias_key) {
            Some(existing) if existing != &canonical_key => {
                c.error(
                    "type_alias_conflict",
                    format!("{}.alias", path),
                    format!(
                        "类型别名「{}」同时指向「{}」与「{}」",
                        alias.alias, existing, alias.canonical
                    ),
                );
            }
            Some(_) => {
                c.warn(
                    "duplicate_type_alias",
                    format!("{}.alias", path),
                    format!("类型别名「{}」重复声明", alias.alias),
                );
            }
            None => {
                alias_map.insert(alias_key, canonical_key);
            }
        }
    }
    let mut reported_alias_cycles: BTreeSet<String> = BTreeSet::new();
    for start in alias_map.keys() {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut current: &str = start;
        loop {
            seen.insert(current);
            let Some(next) = alias_map.get(current) else {
                break;
            };
            if seen.contains(next.as_str()) {
                let mut cycle: Vec<String> = seen.iter().map(|node| node.to_string()).collect();
                cycle.sort();
                let key = cycle.join("→");
                if reported_alias_cycles.insert(key) {
                    c.error(
                        "type_alias_cycle",
                        "type_system.aliases",
                        format!(
                            "类型别名成环：{} —— 归一化将在跳数上限处退化为原样返回",
                            seen.iter().cloned().collect::<Vec<_>>().join(" → ")
                        ),
                    );
                }
                break;
            }
            current = next;
        }
    }

    // 子类环：DFS 三色标记，每个环只报一次。
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (index, subclass) in types.subclasses.iter().enumerate() {
        let path = format!("type_system.subclasses[{}]", index);
        let sub = fold(&subclass.sub);
        let sup = fold(&subclass.sup);
        if sub.is_empty() || sup.is_empty() {
            c.error(
                "empty_subclass",
                path,
                "子类声明的 sub/sup 为空：该声明被忽略",
            );
            continue;
        }
        if sub == sup {
            c.error(
                "self_subclass",
                path,
                format!("类型「{}」声明为自身的子类", subclass.sub),
            );
            continue;
        }
        edges.entry(sub).or_default().push(sup);
    }
    let mut reported: BTreeSet<String> = BTreeSet::new();
    for start in edges.keys() {
        let mut stack: Vec<(String, usize)> = vec![(start.clone(), 0)];
        let mut path: Vec<String> = vec![start.clone()];
        let mut on_path: BTreeSet<String> = BTreeSet::from([start.clone()]);
        while let Some((node, child_index)) = stack.pop() {
            let children = edges.get(&node).cloned().unwrap_or_default();
            if child_index >= children.len() {
                on_path.remove(&node);
                path.pop();
                continue;
            }
            stack.push((node.clone(), child_index + 1));
            let child = children[child_index].clone();
            if on_path.contains(&child) {
                let cycle_start = path.iter().position(|node| node == &child).unwrap_or(0);
                let mut cycle: Vec<String> = path[cycle_start..].to_vec();
                cycle.push(child.clone());
                let canonical_cycle: Vec<String> = cycle
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if reported.insert(canonical_cycle.join("→")) {
                    c.error(
                        "subclass_cycle",
                        "type_system.subclasses",
                        format!("子类关系成环：{}", cycle.join(" → ")),
                    );
                }
                continue;
            }
            if edges.contains_key(&child) {
                on_path.insert(child.clone());
                path.push(child.clone());
                stack.push((child, 0));
            }
        }
    }

    fn canonicalize(map: &BTreeMap<String, String>, name: &str) -> String {
        let mut current = fold(name);
        for _ in 0..64 {
            match map.get(&current) {
                Some(next) => current = next.clone(),
                None => break,
            }
        }
        current
    }

    let mut seen_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    for (index, disjoint) in types.disjoint.iter().enumerate() {
        let path = format!("type_system.disjoint[{}]", index);
        let left = fold(&disjoint.left);
        let right = fold(&disjoint.right);
        if left.is_empty() || right.is_empty() {
            c.error(
                "empty_disjoint",
                path,
                "互斥声明的 left/right 为空：该声明被忽略",
            );
            continue;
        }
        if left == right {
            c.error(
                "self_disjoint",
                path,
                format!("类型「{}」与自身互斥", disjoint.left),
            );
            continue;
        }
        let canonical_left = canonicalize(&alias_map, &left);
        let canonical_right = canonicalize(&alias_map, &right);
        if canonical_left == canonical_right {
            c.error(
                "contradictory_disjoint",
                path,
                format!(
                    "类型「{}」与「{}」别名归一后是同一类型（{}）：互斥声明自相矛盾",
                    disjoint.left, disjoint.right, canonical_left
                ),
            );
            continue;
        }
        let pair = if canonical_left <= canonical_right {
            (canonical_left, canonical_right)
        } else {
            (canonical_right, canonical_left)
        };
        if !seen_pairs.insert(pair) {
            c.warn("duplicate_disjoint", path, "互斥类型对重复声明");
        }
    }
}

fn check_synonyms(c: &mut Collector, config: &SystemConfig) {
    let Some(synonyms) = &config.synonyms else {
        return;
    };
    let keys: BTreeSet<&String> = synonyms.mappings.keys().collect();
    let mut reported_cycles: BTreeSet<String> = BTreeSet::new();
    for key in keys {
        let value = &synonyms.mappings[key];
        let path = format!("synonyms.mappings[{}]", key);
        if key.trim().is_empty() || value.trim().is_empty() {
            c.warn(
                "empty_synonym",
                path.as_str(),
                "同义词映射含空词条：该映射被忽略",
            );
            continue;
        }
        if key == value {
            c.warn(
                "self_synonym",
                path.as_str(),
                format!("词「{}」的同义词是自身", key),
            );
            continue;
        }
        if synonyms.mappings.contains_key(value) {
            c.warn(
                "synonym_chain",
                path.as_str(),
                format!(
                    "「{}」→「{}」，但「{}」还有下一跳：归一化只替换一步，不会到达链尾",
                    key, value, value
                ),
            );
        }
        let mut chain: Vec<String> = vec![key.clone()];
        let mut current: &str = value;
        loop {
            if let Some(position) = chain.iter().position(|node| node == current) {
                let cycle: Vec<String> = chain[position..].to_vec();
                let canonical_cycle: Vec<String> = cycle
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if reported_cycles.insert(canonical_cycle.join("→")) {
                    c.warn(
                        "synonym_cycle",
                        path.as_str(),
                        format!("同义词成环：{}", cycle.join(" → ")),
                    );
                }
                break;
            }
            chain.push(current.to_string());
            match synonyms.mappings.get(current) {
                Some(next) => current = next,
                None => break,
            }
        }
    }
}

fn check_probability(c: &mut Collector, value: f64, path: &str, label: &str) {
    if !value.is_finite() {
        c.error(
            "non_finite_threshold",
            path.to_string(),
            format!("{}不是有限数（{}）", label, value),
        );
    } else if !(0.0..=1.0).contains(&value) {
        c.error(
            "threshold_out_of_range",
            path.to_string(),
            format!("{}={} 不在 [0,1]：概率语义失效", label, value),
        );
    }
}

fn check_clamped_range(
    c: &mut Collector,
    value: usize,
    min: usize,
    max: usize,
    path: &str,
    label: &str,
) {
    if value < min || value > max {
        c.warn(
            "value_will_be_clamped",
            path.to_string(),
            format!(
                "{}={} 超出运行时区间 [{}..{}]：将被静默钳制",
                label, value, min, max
            ),
        );
    }
}

fn check_reasoning(c: &mut Collector, config: &SystemConfig) {
    let r = &config.reasoning;
    check_probability(
        c,
        r.link_prediction_min_confidence,
        "reasoning.link_prediction_min_confidence",
        "链接预测置信度阈值",
    );
    check_probability(
        c,
        r.ontology_alignment_threshold,
        "reasoning.ontology_alignment_threshold",
        "本体对齐阈值",
    );
    check_probability(
        c,
        r.embedding_similarity_floor,
        "reasoning.embedding_similarity_floor",
        "嵌入相似度下限",
    );
    check_probability(
        c,
        r.completion_min_score,
        "reasoning.completion_min_score",
        "统计续写最低支持度",
    );
    check_probability(
        c,
        r.entailment_min_confidence,
        "reasoning.entailment_min_confidence",
        "蕴含证据最低置信度",
    );
    check_probability(
        c,
        r.association_min_support,
        "reasoning.association_min_support",
        "关联规则最小支持度",
    );
    check_probability(
        c,
        r.association_min_confidence,
        "reasoning.association_min_confidence",
        "关联规则最小置信度",
    );
    check_probability(
        c,
        r.soft_logic_min_probability,
        "reasoning.soft_logic_min_probability",
        "软逻辑最低概率",
    );
    check_probability(
        c,
        r.holographic_similarity_floor,
        "reasoning.holographic_similarity_floor",
        "全息召回相似度下限",
    );

    if r.link_prediction_min_support == 0 {
        c.warn(
            "zero_link_prediction_support",
            "reasoning.link_prediction_min_support",
            "链接预测最小支持度为 0：任意偶然共现都会成为规则",
        );
    }
    if r.max_predictions == 0 {
        c.warn(
            "zero_max_predictions",
            "reasoning.max_predictions",
            "max_predictions=0：推理结果将一条都不展示",
        );
    }
    if r.max_profile_sentences == 0 {
        c.warn(
            "zero_profile_sentences",
            "reasoning.max_profile_sentences",
            "max_profile_sentences=0：运行时会被抬升为 1 句",
        );
    }
    if r.association_min_lift < 1.0 {
        c.warn(
            "low_association_lift",
            "reasoning.association_min_lift",
            format!(
                "association_min_lift={} < 1.0：提升度低于 1 表示负相关，规则将失去统计意义",
                r.association_min_lift
            ),
        );
    }
    if !r.graph_walk_damping.is_finite()
        || r.graph_walk_damping <= 0.0
        || r.graph_walk_damping >= 1.0
    {
        c.error(
            "damping_out_of_range",
            "reasoning.graph_walk_damping",
            format!(
                "graph_walk_damping={} 不在 (0,1)：随机游走将退化",
                r.graph_walk_damping
            ),
        );
    } else if !(0.05..=0.99).contains(&r.graph_walk_damping) {
        c.warn(
            "damping_will_be_clamped",
            "reasoning.graph_walk_damping",
            format!(
                "graph_walk_damping={} 超出运行时区间 [0.05,0.99]：将被静默钳制",
                r.graph_walk_damping
            ),
        );
    }
    check_clamped_range(
        c,
        r.graph_walk_iterations,
        1,
        100,
        "reasoning.graph_walk_iterations",
        "随机游走迭代轮数",
    );
    if r.resolution_max_depth == 0 {
        c.warn(
            "zero_resolution_depth",
            "reasoning.resolution_max_depth",
            "resolution_max_depth=0：运行时会被抬升为 1 层",
        );
    }
    check_clamped_range(
        c,
        r.resolution_max_solutions,
        1,
        200,
        "reasoning.resolution_max_solutions",
        "归结查询最大解数",
    );
    check_clamped_range(
        c,
        r.abduction_max_hypotheses,
        1,
        20,
        "reasoning.abduction_max_hypotheses",
        "溯因最大假设数",
    );
    check_clamped_range(
        c,
        r.abduction_max_scenario_size,
        1,
        4,
        "reasoning.abduction_max_scenario_size",
        "溯因场景最大假设数",
    );
    if r.abduction_max_scenario_size > r.abduction_max_hypotheses {
        c.warn(
            "scenario_exceeds_hypotheses",
            "reasoning.abduction_max_scenario_size",
            "溯因场景大小超过最大假设数：多余槽位永远不会用满",
        );
    }
    check_clamped_range(
        c,
        r.soft_logic_max_iterations,
        1,
        32,
        "reasoning.soft_logic_max_iterations",
        "软逻辑迭代轮数",
    );
    check_clamped_range(
        c,
        r.causal_max_depth,
        1,
        8,
        "reasoning.causal_max_depth",
        "因果链最大深度",
    );
    check_clamped_range(
        c,
        r.problog_max_depth,
        1,
        16,
        "reasoning.problog_max_depth",
        "概率证明最大深度",
    );
    check_clamped_range(
        c,
        r.problog_max_proofs,
        1,
        64,
        "reasoning.problog_max_proofs",
        "概率查询最大证明路径数",
    );
    if r.discourse_max_sentences == 0 {
        c.warn(
            "zero_discourse_sentences",
            "reasoning.discourse_max_sentences",
            "discourse_max_sentences=0：运行时会被抬升为 1 句",
        );
    }
    if r.discourse_max_rounds == 0 {
        c.warn(
            "zero_discourse_rounds",
            "reasoning.discourse_max_rounds",
            "discourse_max_rounds=0：运行时会被抬升为 1 轮",
        );
    }
    check_clamped_range(
        c,
        r.holographic_dimension,
        256,
        16384,
        "reasoning.holographic_dimension",
        "全息超向量维度",
    );
    if r.holographic_similarity_floor < 0.0 {
        c.warn(
            "negative_holographic_floor",
            "reasoning.holographic_similarity_floor",
            format!(
                "holographic_similarity_floor={} 为负：运行时会被钳制到 0",
                r.holographic_similarity_floor
            ),
        );
    }
    if r.holographic_max_hits == 0 {
        c.warn(
            "zero_holographic_hits",
            "reasoning.holographic_max_hits",
            "holographic_max_hits=0：全息召回不可见",
        );
    }
    if let Some(path) = &r.dialogue_state_path {
        if path.trim().is_empty() {
            c.warn(
                "empty_dialogue_state_path",
                "reasoning.dialogue_state_path",
                "对话记忆路径为空：跨会话记忆无法落盘",
            );
        }
    }
}

fn check_soft_route(c: &mut Collector, config: &SystemConfig) {
    if let Err(issue) = config.reasoning.soft_route.validate() {
        c.error("invalid_soft_route", "reasoning.soft_route", issue);
    }
}

fn check_confidence(c: &mut Collector, config: &SystemConfig) {
    for issue in config.reasoning.confidence.validate() {
        match issue.severity {
            crate::confidence::ConfidenceSeverity::Error => c.error(
                &issue.code,
                issue.path,
                issue.message,
            ),
            crate::confidence::ConfidenceSeverity::Warning => c.warn(
                &issue.code,
                issue.path,
                issue.message,
            ),
        }
    }
}

fn check_trace_graph(c: &mut Collector, config: &SystemConfig) {
    let t = &config.trace_graph;
    if trace_graph::TraceMode::parse(&t.mode).is_none() {
        c.error(
            "unknown_trace_mode",
            "trace_graph.mode",
            format!(
                "未知的分层追踪模式「{}」：可选 off / observe / on（非法值将回落为 off）",
                t.mode
            ),
        );
    }
    if t.max_turns == 0 {
        c.warn(
            "zero_trace_turns",
            "trace_graph.max_turns",
            "max_turns=0：追踪图无法保留任何轮迹",
        );
    }
    if t.max_nodes_per_turn == 0 {
        c.warn(
            "zero_trace_nodes",
            "trace_graph.max_nodes_per_turn",
            "max_nodes_per_turn=0：单轮不记录任何决策节点",
        );
    }
    if t.max_events_per_turn == 0 {
        c.warn(
            "zero_trace_events",
            "trace_graph.max_events_per_turn",
            "max_events_per_turn=0：单轮不记录任何指代事件",
        );
    }
}

fn check_dialogue_frames(c: &mut Collector, config: &SystemConfig) {
    let frames = &config.dialogue_frames;
    if frames.frames.is_empty() {
        if frames.enabled {
            c.warn(
                "empty_dialogue_frames",
                "dialogue_frames.frames",
                "enabled=true 但未声明任何帧：绑定帧层不会接管任何输入",
            );
        }
        return;
    }
    if let Err(errors) = FrameProgram::compile(frames) {
        for error in errors {
            c.error(error.code, error.path, error.message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小可用配置：所有必需字段齐备，可被故障注入测试当作基线。
    const BASE_CONFIG: &str = r#"
[system]
oxigraph_endpoint = "http://localhost:7878"
oxigraph_store_path = "oxigraph_data"
default_language = "zh"

[nlp]
import_keywords = ["录入", "添加"]
nlu_confidence_threshold = 0.6
semantic_similarity_threshold = 0.7
max_entity_candidates = 10
context_window_size = 5

[nlp.pos_dict]
"是" = "v"
"的" = "u"
"年龄" = "n"
"多大" = "r"
"谁" = "r"

[nlp.name_patterns]
prefixes = ["小", "老"]
suffixes = ["总"]
full_names = ["张三"]
profession_suffix = ["医生"]

[[prolog_rules]]
name = "年龄查询"
pattern = ["nr", "多大"]
query_type = "age_query"
priority = 10

[[prolog_rules]]
name = "关系查询"
pattern = ["nr", "和", "nr", "什么", "关系"]
query_type = "relation_query"
priority = 9

[[entity_types]]
label = "Person"
examples = ["张三18岁"]

[[entity_types.properties]]
name = "name"
data_type = "string"
required = true
extraction_patterns = ["pos:nr"]

[[entity_types.properties]]
name = "age"
data_type = "integer"
required = false
extraction_patterns = ["pos:m"]

[[extraction_rules]]
name = "人物信息提取"
entity_type = "Person"
priority = 10

[[extraction_rules.patterns]]
pos_pattern = ["nr", "m", "q"]
property_mappings = { name = "pos:nr", age = "pos:m" }

[messages]
startup = "启动中"
ready = "就绪"
query_prompt = "请输入查询"
import_prompt = "输入导入"
quit_prompt = "输入quit退出"
importing = "导入中"
no_knowledge_extracted = "未提取到知识"
cannot_understand = "无法理解"
process_error = "处理出错"
"#;

    fn base() -> String {
        BASE_CONFIG.to_string()
    }

    fn lint_source(source: &str) -> Vec<ConfigIssue> {
        lint_str(source).expect("测试配置应能解析")
    }

    fn codes(issues: &[ConfigIssue]) -> Vec<String> {
        issues.iter().map(|issue| issue.code.clone()).collect()
    }

    fn assert_code(issues: &[ConfigIssue], code: &str) {
        assert!(
            issues.iter().any(|issue| issue.code == code),
            "期望问题码 {}，实际：{:?}\n{}",
            code,
            codes(issues),
            render_text(issues)
        );
    }

    #[test]
    fn base_config_is_clean() {
        let issues = lint_source(&base());
        assert!(
            !has_errors(&issues),
            "基线配置不应有 Error：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn shipped_config_has_no_errors() {
        let source = include_str!("../config.toml");
        let issues = lint_source(source);
        assert!(
            !has_errors(&issues),
            "出厂 config.toml 不应有 Error：\n{}",
            render_text(&issues)
        );
    }

    #[test]
    fn semantic_grammar_rules_are_compiled_and_reported() {
        let valid = format!(
            "{}\n[semantic_grammar]\nenabled = true\n\n[[semantic_grammar.rules]]\nname = \"身高\"\npredicate = \"height\"\nentity_type = \"Person\"\npattern = [\"$s:pos:nr\", \"身高\", \"$o:pos:m\"]\nrealize = \"{{s}}身高{{o}}\"\n",
            base()
        );
        let issues = lint_source(&valid);
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code.starts_with("semantic_")),
            "合法双向规则不应报语义问题：{}",
            render_text(&issues)
        );

        let invalid = format!(
            "{}\n[semantic_grammar]\nenabled = true\n\n[[semantic_grammar.rules]]\nname = \"坏规则\"\npredicate = \"height\"\npattern = [\"$s:pos:nr\"]\nrealize = \"{{s}}\"\n",
            base()
        );
        assert_code(&lint_source(&invalid), "semantic_rule_compile_error");

        let disabled = format!(
            "{}\n[semantic_grammar]\nenabled = false\n\n[[semantic_grammar.rules]]\nname = \"预览\"\npredicate = \"height\"\npattern = [\"$s:pos:nr\", \"身高\", \"$o:pos:m\"]\nrealize = \"{{s}}身高{{o}}\"\n",
            base()
        );
        assert_code(&lint_source(&disabled), "semantic_grammar_disabled");
    }

    #[test]
    fn trace_graph_mode_is_validated() {
        // 合法模式：不报错误。
        for mode in ["off", "observe", "on"] {
            let source = format!("{}\n[trace_graph]\nmode = \"{}\"\n", base(), mode);
            let issues = lint_source(&source);
            assert!(
                !issues
                    .iter()
                    .any(|issue| issue.code == "unknown_trace_mode"),
                "模式 {} 不应报 unknown_trace_mode：{}",
                mode,
                render_text(&issues)
            );
        }
        // 非法模式：报 unknown_trace_mode。
        let source = format!("{}\n[trace_graph]\nmode = \"bogus\"\n", base());
        assert_code(&lint_source(&source), "unknown_trace_mode");
    }

    #[test]
    fn trace_graph_zero_limits_are_warnings() {
        let source = format!(
            "{}\n[trace_graph]\nmode = \"off\"\nmax_turns = 0\nmax_nodes_per_turn = 0\nmax_events_per_turn = 0\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "zero_trace_turns");
        assert_code(&issues, "zero_trace_nodes");
        assert_code(&issues, "zero_trace_events");
    }

    #[test]
    fn shipped_config_has_no_errors_placeholder() {
        let source = include_str!("../config.toml");
        let issues = lint_source(source);
        let config: SystemConfig = toml::from_str(source).expect("config.toml 应能解析");
        let _ = &config;
        assert!(
            !has_errors(&issues),
            "出厂 config.toml 不应有 Error：\n{}",
            render_text(&issues)
        );
    }

    #[test]
    fn shipped_config_query_types_are_all_supported() {
        let source = include_str!("../config.toml");
        let config: SystemConfig = toml::from_str(source).expect("config.toml 应能解析");
        for rule in &config.prolog_rules {
            assert!(
                is_query_type_supported(rule.query_type.trim()),
                "出厂配置的 query_type「{}」应在能力图上",
                rule.query_type
            );
        }
    }

    #[test]
    fn prolog_generated_list_matches_engine() {
        for query_type in GENERATED_QUERY_TYPES {
            let query = crate::prolog_engine::PrologEngine::generate_prolog_query(
                query_type,
                &["测试实体".to_string(), "另一实体".to_string()],
            );
            assert!(
                query.is_some(),
                "能力表声称可生成查询，但引擎未实现：{}",
                query_type
            );
        }
    }

    #[test]
    fn dangling_entity_type_is_error() {
        let issues =
            lint_source(&base().replace("entity_type = \"Person\"", "entity_type = \"Persom\""));
        assert_code(&issues, "dangling_entity_type");
        assert!(issues
            .iter()
            .any(|issue| issue.code == "dangling_entity_type"
                && issue.severity == ContractSeverity::Error));
    }

    #[test]
    fn dangling_query_type_is_error() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"创始人查询\"\npattern = [\"n\", \"创始人\"]\nquery_type = \"founder_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "dangling_query_type");
    }

    #[test]
    fn empty_query_pattern_and_token_are_errors() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"坏规则\"\npattern = [\"\"]\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "empty_pattern_token");
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"空规则\"\npattern = []\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "empty_query_pattern");
    }

    #[test]
    fn insufficient_entity_slots_is_warning() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"关系查询2\"\npattern = [\"nr\", \"和\", \"什么\", \"关系\"]\nquery_type = \"relation_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "insufficient_entity_slots");
        assert!(issues
            .iter()
            .all(|issue| issue.code != "insufficient_entity_slots"
                || issue.severity == ContractSeverity::Warning));
    }

    #[test]
    fn ascii_pattern_token_treated_as_word_is_warning() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"笔误规则\"\npattern = [\"nn\", \"年龄\"]\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "pattern_token_treated_as_word");
    }

    #[test]
    fn duplicate_entity_type_and_property_are_errors() {
        let source = format!(
            "{}\n[[entity_types]]\nlabel = \"Person\"\nexamples = []\n\n[[entity_types.properties]]\nname = \"name\"\ndata_type = \"string\"\nrequired = true\nextraction_patterns = [\"pos:nr\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "duplicate_entity_type");

        let source = base().replace(
            "[[entity_types.properties]]\nname = \"age\"",
            "[[entity_types.properties]]\nname = \"name\"",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "duplicate_property");
    }

    #[test]
    fn undeclared_property_mapping_is_warning() {
        let source = base().replace(
            "property_mappings = { name = \"pos:nr\", age = \"pos:m\" }",
            "property_mappings = { name = \"pos:nr\", height = \"pos:m\" }",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "undeclared_property_mapping");
    }

    #[test]
    fn property_extraction_patterns_are_optional_but_inert() {
        let issues = lint_source(&base());
        assert_code(&issues, "inert_property_extraction_patterns");

        let without = base()
            .replace("extraction_patterns = [\"pos:nr\"]\n", "")
            .replace("extraction_patterns = [\"pos:m\"]\n", "");
        let issues = lint_source(&without);
        assert!(
            !codes(&issues)
                .iter()
                .any(|code| code == "inert_property_extraction_patterns"),
            "省略 extraction_patterns 后不应再提示：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn predicate_alias_conflicts_are_errors() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"师傅\"\naliases = [\"老师\"]\ninverse = \"徒弟\"\n\n[[predicate_profiles]]\nname = \"徒弟\"\ninverse = \"师傅\"\n\n[[predicate_profiles]]\nname = \"师父\"\naliases = [\"老师\"]\n\n[[predicate_profiles]]\nname = \"导师\"\naliases = [\"徒弟\"]\n\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"丙\"\naliases = [\"乙\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "predicate_alias_conflict");
        assert_code(&issues, "alias_shadows_canonical");
        assert_code(&issues, "alias_conflicts_with_inverse");
    }

    #[test]
    fn predicate_inverse_checks() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"甲\"\n\n[[predicate_profiles]]\nname = \"乙\"\ninverse = \"丙\"\n\n[[predicate_profiles]]\nname = \"丙\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"丁\"\ninverse = \"戊\"\n\n[[predicate_profiles]]\nname = \"戊\"\ninverse = \"丁\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "self_inverse");
        // 乙↔丙 互逆一致；丁→戊 未互报：戊声明丁，方向一致，不应误报
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code == "inverse_not_reciprocated"),
            "互逆一致不应误报：{}",
            render_text(&issues)
        );

        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"乙\"\ninverse = \"丙\"\n\n[[predicate_profiles]]\nname = \"丙\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "inverse_not_reciprocated");
    }

    #[test]
    fn unknown_domain_type_is_warning() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"师傅\"\ndomain = [\"人间\"]\nrange = [\"Person\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "unknown_domain_type");
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code == "unknown_range_type"),
            "已知类型 Person 不应误报：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn type_alias_cycle_is_error() {
        let source = format!(
            "{}\n[type_system]\n\n[[type_system.aliases]]\nalias = \"A\"\ncanonical = \"B\"\n\n[[type_system.aliases]]\nalias = \"B\"\ncanonical = \"A\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "type_alias_cycle");
    }

    #[test]
    fn subclass_cycle_is_error() {
        let source = format!(
            "{}\n[[type_system.subclasses]]\nsub = \"学生\"\nsup = \"人物\"\n\n[[type_system.subclasses]]\nsub = \"人物\"\nsup = \"学生\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "subclass_cycle");
    }

    #[test]
    fn contradictory_disjoint_is_error() {
        let source = format!(
            "{}\n[[type_system.aliases]]\nalias = \"Person\"\ncanonical = \"人物\"\n\n[[type_system.disjoint]]\nleft = \"人物\"\nright = \"Person\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "contradictory_disjoint");
    }

    #[test]
    fn synonym_self_chain_and_cycle_are_warnings() {
        let source = format!(
            "{}\n[synonyms.mappings]\n\"甲\" = \"甲\"\n\"乙\" = \"丙\"\n\"丙\" = \"丁\"\n\"戊\" = \"己\"\n\"己\" = \"戊\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "self_synonym");
        assert_code(&issues, "synonym_chain");
        assert_code(&issues, "synonym_cycle");
    }

    #[test]
    fn probability_out_of_range_is_error() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\nsoft_logic_min_probability = 1.5\n\n[nlp]\nimport_keywords",
        );
        let source = source.replace(
            "nlu_confidence_threshold = 0.6",
            "nlu_confidence_threshold = -0.2",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "threshold_out_of_range");
        assert_eq!(error_count(&issues), 2, "{}", render_text(&issues));
    }

    #[test]
    fn damping_out_of_range_is_error() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\ngraph_walk_damping = 1.5\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "damping_out_of_range");
    }

    #[test]
    fn clamping_band_is_warning() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\ncausal_max_depth = 99\nholographic_dimension = 128\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "value_will_be_clamped");
        assert!(issues
            .iter()
            .filter(|issue| issue.code == "value_will_be_clamped")
            .all(|issue| issue.severity == ContractSeverity::Warning));
    }

    #[test]
    fn inert_oxigraph_endpoint_is_warning() {
        let issues = lint_source(&base());
        assert_code(&issues, "inert_oxigraph_endpoint");
        assert!(issues
            .iter()
            .filter(|issue| issue.code == "inert_oxigraph_endpoint")
            .all(|issue| issue.severity == ContractSeverity::Warning));

        let blanked = base().replace(
            "oxigraph_endpoint = \"http://localhost:7878\"",
            "oxigraph_endpoint = \"\"",
        );
        let issues = lint_source(&blanked);
        assert!(
            !codes(&issues)
                .iter()
                .any(|code| code == "inert_oxigraph_endpoint"),
            "留空后不应再提示：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn strict_admission_without_profiles_is_warning() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\nrequire_predicate_admission = true\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "strict_admission_without_profiles");
    }

    #[test]
    fn entity_profile_conflicts_are_errors() {
        let source = format!(
            "{}\n[[entity_profiles]]\nname = \"张三\"\naliases = [\"老张\"]\netype = [\"Person\"]\n\n[[entity_profiles]]\nname = \"李四\"\naliases = [\"老张\"]\netype = [\"外星人\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "entity_alias_conflict");
        assert_code(&issues, "unknown_entity_profile_type");
    }

    #[test]
    fn stale_confidence_certificate_is_an_error() {
        let source = format!(
            "{}\n[reasoning.confidence]\nenabled = true\nanswer_threshold = 0.86\nhedge_threshold = 0.62\nclarify_threshold = 0.55\ncertificate = {{ n = 240, errors = 4, alpha = 0.05, delta = 0.001, max_error_rate = 0.05, claimed_upper_bound = 0.0100 }}\n\n[[reasoning.confidence.sources]]\nname = \"intent\"\ngamma = 1.0\ncalibration = [{{ score = 0.5, probability = 0.7 }}, {{ score = 0.9, probability = 0.9 }}]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "confidence_certificate_invalid");
    }

    #[test]
    fn parse_errors_are_returned() {
        assert!(lint_str("this is not toml = = =").is_err());
    }

    #[test]
    fn lint_is_deterministic() {
        let source = include_str!("../config.toml");
        let first = lint_source(source);
        let second = lint_source(source);
        assert_eq!(first, second);
        assert_eq!(render_text(&first), render_text(&second));
    }

    #[test]
    fn issues_are_json_round_trippable() {
        let issues = lint_source(&base());
        let json = serde_json::to_string(&issues).expect("问题列表应能序列化");
        let back: Vec<ConfigIssue> = serde_json::from_str(&json).expect("问题列表应能反序列化");
        assert_eq!(issues, back);
    }
}

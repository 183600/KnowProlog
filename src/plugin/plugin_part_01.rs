use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};
use crate::rule_pack::{ExclusiveGroup, FunctionalKey, RulePack, RuleSpec};
/// 当前宿主认识的插件清单 API 版本。
pub const PLUGIN_API_VERSION: u32 = 1;
/// 默认插件目录（相对工作目录）。
pub const DEFAULT_PLUGIN_DIRECTORY: &str = "plugins";
/// 一次用户输入最多连续改写的轮数（防止插件互相改写形成环）。
pub const MAX_REWRITE_PASSES: usize = 3;
fn default_plugin_directory() -> String {
    DEFAULT_PLUGIN_DIRECTORY.to_string()
}
fn default_api_version() -> u32 {
    PLUGIN_API_VERSION
}
fn default_manifest_version() -> String {
    "0.0.0".to_string()
}
/// 插件的运行模式。`Observe` 与对话/恢复模块的 parity/observe 语义一致：
/// 先观察，再接管；接管必须可回退。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginMode {
    #[default]
    Off,
    Observe,
    On,
}
impl PluginMode {
    pub fn label(self) -> &'static str {
        match self {
            PluginMode::Off => "停用",
            PluginMode::Observe => "观察",
            PluginMode::On => "启用",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            PluginMode::Off => "off",
            PluginMode::Observe => "observe",
            PluginMode::On => "on",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "off" | "disable" | "disabled" | "false" | "0" | "停用" | "关闭" | "禁用" => {
                Some(PluginMode::Off)
            }
            "observe" | "parity" | "dry-run" | "dryrun" | "观察" | "试运行" | "旁路" => {
                Some(PluginMode::Observe)
            }
            "on" | "enable" | "enabled" | "true" | "1" | "启用" | "开启" | "打开" => {
                Some(PluginMode::On)
            }
            _ => None,
        }
    }

    pub fn is_active(self) -> bool {
        !matches!(self, PluginMode::Off)
    }
}
/// `config.toml` 的 `[plugins]` 段：目录、启动时启用列表、逐包模式覆盖。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSettings {
    #[serde(default = "default_plugin_directory")]
    pub directory: String,
    #[serde(default)]
    pub enabled: Vec<String>,
    #[serde(default)]
    pub modes: BTreeMap<String, PluginMode>,
}
impl Default for PluginSettings {
    fn default() -> Self {
        PluginSettings {
            directory: default_plugin_directory(),
            enabled: Vec::new(),
            modes: BTreeMap::new(),
        }
    }
}
/// 插件清单。字段缺省值保证小清单也能工作，但 id/api 必须显式给出。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    #[serde(default = "default_manifest_version")]
    pub version: String,
    #[serde(default = "default_api_version")]
    pub api: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub default_mode: PluginMode,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub conflicts: Vec<String>,
    #[serde(default)]
    pub provides: Vec<String>,
    #[serde(default)]
    pub contributions: Vec<Contribution>,
    #[serde(default)]
    pub eval_cases: Vec<PluginEvalCase>,
}
impl PluginManifest {
    pub fn display_name(&self) -> &str {
        if self.name.trim().is_empty() {
            &self.id
        } else {
            &self.name
        }
    }

    pub fn contribution_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for contribution in &self.contributions {
            *counts.entry(contribution.kind()).or_insert(0) += 1;
        }
        counts
    }
}
/// 八类声明式贡献。每个贡献都有插件内唯一的 id，并携带一种类型化资产。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Contribution {
    /// 词表：词面 → 规范名（可带词性提示）。
    Lexicon {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        entries: Vec<LexiconEntry>,
    },
    /// 本体：谓词画像与类型别名/层级/互斥声明。
    Ontology {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        predicates: Vec<OntologyPredicate>,
        #[serde(default)]
        types: Vec<OntologyType>,
    },
    /// 指称线索：量词、角色词、姓氏等（CRD 的可配置扩展位）。
    Reference {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        quantifiers: Vec<String>,
        #[serde(default)]
        role_words: Vec<String>,
        #[serde(default)]
        surnames: Vec<String>,
    },
    /// 问法改写：受限槽位模式 → 已有语义帧。
    Rewrite {
        id: String,
        #[serde(default)]
        description: String,
        pattern: String,
        replacement: String,
    },
    /// 检索配方：问法模式 → 规范问句 + 可选模板。
    ///
    /// `direction` 描述「实体在关系中扮演哪一端」：`forward` 查实体 → 答案，
    /// `inverse` 查答案 → 实体（如「卡梅隆导演过哪些电影」）。`answer_types`
    /// 是答案的期望类型：只有已知类型冲突（显式互斥）才拒答，未知类型放行。
    Recipe {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        patterns: Vec<String>,
        question: String,
        #[serde(default)]
        template: Option<String>,
        #[serde(default)]
        attribute: Option<String>,
        #[serde(default)]
        direction: RecipeDirection,
        #[serde(default)]
        answer_types: Vec<String>,
    },
    /// 话语策略（声明；由宿主在核心策略答不了时择优使用）。
    Policy {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        when: Vec<String>,
        #[serde(default)]
        reply: String,
    },
    /// 生成模板：只允许使用已接地的槽位。
    Template {
        id: String,
        #[serde(default)]
        description: String,
        template: String,
    },
    /// 风格片段：固定前后缀，只能改变措辞，不能替换事实。
    Style {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        prefix: String,
        #[serde(default)]
        suffix: String,
    },
    /// 规则包：带层次、依赖、激活话题、功能键与互斥谓词的 Horn 规则。
    Rules {
        id: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        layer: i32,
        #[serde(default)]
        activation: Vec<String>,
        #[serde(default)]
        requires: Vec<String>,
        #[serde(default)]
        conflicts: Vec<String>,
        #[serde(default)]
        functional: Vec<FunctionalKey>,
        #[serde(default)]
        exclusive: Vec<ExclusiveGroup>,
        #[serde(default)]
        rules: Vec<RuleSpec>,
    },
}
impl Contribution {
    pub fn id(&self) -> &str {
        match self {
            Contribution::Lexicon { id, .. }
            | Contribution::Ontology { id, .. }
            | Contribution::Reference { id, .. }
            | Contribution::Rewrite { id, .. }
            | Contribution::Recipe { id, .. }
            | Contribution::Policy { id, .. }
            | Contribution::Template { id, .. }
            | Contribution::Style { id, .. }
            | Contribution::Rules { id, .. } => id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Contribution::Lexicon { .. } => "lexicon",
            Contribution::Ontology { .. } => "ontology",
            Contribution::Reference { .. } => "reference",
            Contribution::Rewrite { .. } => "rewrite",
            Contribution::Recipe { .. } => "recipe",
            Contribution::Policy { .. } => "policy",
            Contribution::Template { .. } => "template",
            Contribution::Style { .. } => "style",
            Contribution::Rules { .. } => "rules",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Contribution::Lexicon { description, .. }
            | Contribution::Ontology { description, .. }
            | Contribution::Reference { description, .. }
            | Contribution::Rewrite { description, .. }
            | Contribution::Recipe { description, .. }
            | Contribution::Policy { description, .. }
            | Contribution::Template { description, .. }
            | Contribution::Style { description, .. }
            | Contribution::Rules { description, .. } => description,
        }
    }

    pub fn rule_pack(&self, plugin: &str) -> Option<RulePack> {
        let Contribution::Rules {
            id,
            description,
            layer,
            activation,
            requires,
            conflicts,
            functional,
            exclusive,
            rules,
        } = self
        else {
            return None;
        };
        let qualify = |reference: &str| {
            if reference.contains('#') {
                reference.to_string()
            } else {
                format!("{}#{}", plugin, reference)
            }
        };
        Some(RulePack {
            id: format!("{}#{}", plugin, id),
            name: if description.trim().is_empty() {
                format!("{} / {}", plugin, id)
            } else {
                description.clone()
            },
            layer: *layer,
            activation: activation.clone(),
            requires: requires.iter().map(|entry| qualify(entry)).collect(),
            conflicts: conflicts.iter().map(|entry| qualify(entry)).collect(),
            functional: functional.clone(),
            exclusive: exclusive.clone(),
            rules: rules.clone(),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LexiconEntry {
    pub surface: String,
    pub canonical: String,
    #[serde(default)]
    pub pos: Option<String>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OntologyPredicate {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub inverse: Option<String>,
    #[serde(default)]
    pub symmetric: bool,
    #[serde(default)]
    pub transitive: bool,
    #[serde(default)]
    pub domain: Vec<String>,
    #[serde(default)]
    pub range: Vec<String>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OntologyType {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub subclass_of: Vec<String>,
    #[serde(default)]
    pub disjoint_with: Vec<String>,
}
/// 包内自测：验证「装了这个包，至少这些说法会按预期改写」。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginEvalCase {
    pub input: String,
    #[serde(default)]
    pub expect_rewrite: Option<String>,
    #[serde(default)]
    pub expect_plugin: Option<String>,
    #[serde(default)]
    pub expect_contains: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginSeverity {
    Error,
    Warning,
}
impl PluginSeverity {
    pub fn label(self) -> &'static str {
        match self {
            PluginSeverity::Error => "错误",
            PluginSeverity::Warning => "警告",
        }
    }
}
/// 清单 / 启用裁决中的一条诊断。错误会阻止该清单或该启用动作生效。
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PluginIssue {
    pub severity: PluginSeverity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    pub field: String,
    pub message: String,
}
impl PluginIssue {
    fn error(plugin: Option<&str>, file: Option<&Path>, field: &str, message: String) -> Self {
        PluginIssue {
            severity: PluginSeverity::Error,
            plugin: plugin.map(str::to_string),
            file: file.map(|path| path.display().to_string()),
            field: field.to_string(),
            message,
        }
    }

    fn warning(plugin: Option<&str>, field: &str, message: String) -> Self {
        PluginIssue {
            severity: PluginSeverity::Warning,
            plugin: plugin.map(str::to_string),
            file: None,
            field: field.to_string(),
            message,
        }
    }

    pub fn render(&self) -> String {
        let mut prefix = format!("[{}]", self.severity.label());
        if let Some(plugin) = &self.plugin {
            prefix.push_str(&format!("[{}]", plugin));
        }
        if let Some(file) = &self.file {
            prefix.push_str(&format!("[{}]", file));
        }
        format!("{} {}：{}", prefix, self.field, self.message)
    }
}
/// 已安装插件集合：负责发现、解析、lint 与跨包冲突检查。
#[derive(Clone, Debug, Default)]
pub struct PluginRegistry {
    directory: Option<String>,
    manifests: Vec<PluginManifest>,
    issues: Vec<PluginIssue>,
}
impl PluginRegistry {
    pub fn empty() -> Self {
        PluginRegistry::default()
    }

    /// 从目录加载插件。支持 `dir/*.toml` 与 `dir/<package>/manifest.toml`
    /// 两种布局；目录不存在时返回空注册表并记一条警告。
    pub fn load_dir(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        let mut registry = PluginRegistry {
            directory: Some(directory.display().to_string()),
            ..PluginRegistry::default()
        };
        if !directory.exists() {
            registry.issues.push(PluginIssue::warning(
                None,
                "directory",
                format!("插件目录 {} 不存在，按无插件运行。", directory.display()),
            ));
            return registry;
        }
        let mut files = Vec::new();
        match fs::read_dir(directory) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let manifest = path.join("manifest.toml");
                        if manifest.is_file() {
                            files.push(manifest);
                        }
                    } else if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                        files.push(path);
                    }
                }
            }
            Err(error) => {
                registry.issues.push(PluginIssue::error(
                    None,
                    Some(directory),
                    "directory",
                    format!("读取插件目录失败：{}", error),
                ));
                return registry;
            }
        }
        files.sort();
        let mut manifests = Vec::new();
        for file in files {
            let text = match fs::read_to_string(&file) {
                Ok(text) => text,
                Err(error) => {
                    registry.issues.push(PluginIssue::error(
                        None,
                        Some(&file),
                        "read",
                        format!("读取失败：{}", error),
                    ));
                    continue;
                }
            };
            match toml::from_str::<PluginManifest>(&text) {
                Ok(manifest) => manifests.push((file, manifest)),
                Err(error) => registry.issues.push(PluginIssue::error(
                    None,
                    Some(&file),
                    "toml",
                    format!("解析失败：{}", error),
                )),
            }
        }
        registry.finish(manifests);
        registry
    }

    /// 直接由清单构造（供测试与嵌入式使用），仍执行完整的跨包检查。
    pub fn from_manifests(manifests: Vec<PluginManifest>) -> Self {
        let mut registry = PluginRegistry::default();
        let manifests = manifests
            .into_iter()
            .map(|manifest| (PathBuf::from(format!("<{}>", manifest.id)), manifest))
            .collect();
        registry.finish(manifests);
        registry
    }

    fn finish(&mut self, raw: Vec<(PathBuf, PluginManifest)>) {
        let mut accepted: Vec<PluginManifest> = Vec::new();
        let mut seen_plugins: BTreeSet<String> = BTreeSet::new();
        let mut seen_contributions: BTreeSet<String> = BTreeSet::new();

        for (file, manifest) in raw {
            let issues = validate_manifest(&manifest, &file);
            let has_error = issues
                .iter()
                .any(|issue| issue.severity == PluginSeverity::Error);
            self.issues.extend(issues);
            if has_error {
                continue;
            }
            if !seen_plugins.insert(manifest.id.clone()) {
                self.issues.push(PluginIssue::error(
                    Some(&manifest.id),
                    Some(&file),
                    "id",
                    "插件 id 重复，后者被忽略。".to_string(),
                ));
                continue;
            }
            let mut duplicate = false;
            for contribution in &manifest.contributions {
                if !seen_contributions.insert(contribution.id().to_string()) {
                    self.issues.push(PluginIssue::error(
                        Some(&manifest.id),
                        Some(&file),
                        contribution.id(),
                        format!(
                            "贡献 id「{}」与已安装插件冲突，整个包被忽略。",
                            contribution.id()
                        ),
                    ));
                    duplicate = true;
                }
            }
            if duplicate {
                continue;
            }
            accepted.push(manifest);
        }

        accepted.sort_by(|left, right| left.id.cmp(&right.id));
        self.manifests = accepted;
        self.cross_validate();
    }

    fn cross_validate(&mut self) {
        let ids: BTreeSet<String> = self.manifests.iter().map(|m| m.id.clone()).collect();
        for manifest in &self.manifests {
            for required in &manifest.requires {
                if required == &manifest.id {
                    self.issues.push(PluginIssue::error(
                        Some(&manifest.id),
                        None,
                        "requires",
                        "插件不能依赖自身。".to_string(),
                    ));
                } else if !ids.contains(required) {
                    self.issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "requires",
                        format!("依赖的插件「{}」未安装；启用时会被拒绝。", required),
                    ));
                }
            }
            for conflict in &manifest.conflicts {
                if conflict == &manifest.id {
                    self.issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "conflicts",
                        "插件与自己冲突，已忽略该声明。".to_string(),
                    ));
                }
            }
            for eval_case in &manifest.eval_cases {
                if eval_case.input.trim().is_empty() {
                    self.issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "eval_cases",
                        "存在空的评测输入。".to_string(),
                    ));
                }
                if eval_case.expect_rewrite.is_none()
                    && eval_case.expect_plugin.is_none()
                    && eval_case.expect_contains.is_none()
                {
                    self.issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "eval_cases",
                        format!("评测输入「{}」没有断言。", eval_case.input),
                    ));
                }
            }
        }
        // 配方引用的模板必须存在，且模板槽位必须在配方可提供的槽位内。
        let templates: BTreeMap<String, (&str, Vec<Segment>)> = self
            .manifests
            .iter()
            .flat_map(|manifest| {
                manifest
                    .contributions
                    .iter()
                    .filter_map(move |contribution| {
                        if let Contribution::Template { id, template, .. } = contribution {
                            parse_template(template)
                                .ok()
                                .map(|segments| (id.clone(), (manifest.id.as_str(), segments)))
                        } else {
                            None
                        }
                    })
            })
            .collect();
        for manifest in &self.manifests {
            for contribution in &manifest.contributions {
                if let Contribution::Recipe {
                    id,
                    patterns,
                    template: Some(template_id),
                    ..
                } = contribution
                {
                    let Some((_, template_segments)) = templates.get(template_id) else {
                        self.issues.push(PluginIssue::error(
                            Some(&manifest.id),
                            None,
                            id,
                            format!("引用的模板「{}」不存在。", template_id),
                        ));
                        continue;
                    };
                    let mut available = BTreeSet::new();
                    for pattern in patterns {
                        if let Ok(segments) = parse_template(pattern) {
                            collect_slots(&segments, &mut available);
                        }
                    }
                    available.extend(
                        ["answer", "attribute", "evidence", "plugin", "recipe"]
                            .iter()
                            .map(|slot| slot.to_string()),
                    );
                    let mut required = BTreeSet::new();
                    collect_slots(template_segments, &mut required);
                    if let Some(missing) = required.difference(&available).next() {
                        self.issues.push(PluginIssue::error(
                            Some(&manifest.id),
                            None,
                            id,
                            format!(
                                "模板「{}」需要槽位「{}」，但配方无法提供。",
                                template_id, missing
                            ),
                        ));
                    }
                }
            }
        }
    }

    pub fn directory(&self) -> Option<&str> {
        self.directory.as_deref()
    }

    pub fn manifests(&self) -> &[PluginManifest] {
        &self.manifests
    }

    pub fn issues(&self) -> &[PluginIssue] {
        &self.issues
    }

    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == PluginSeverity::Error)
    }

    pub fn get(&self, id: &str) -> Option<&PluginManifest> {
        self.manifests.iter().find(|manifest| manifest.id == id)
    }

    pub fn render_issues(&self) -> String {
        if self.issues.is_empty() {
            return "插件清单检查：通过（无问题）。".to_string();
        }
        let mut lines = vec![format!("插件清单检查：{} 条问题。", self.issues.len())];
        for issue in &self.issues {
            lines.push(format!("  {}", issue.render()));
        }
        lines.join("\n")
    }

    /// 以「全部启用」的临时运行态跑包内自测。
    pub fn evaluate(&self) -> Vec<PluginEvalResult> {
        let settings = PluginSettings {
            enabled: self.manifests.iter().map(|m| m.id.clone()).collect(),
            ..PluginSettings::default()
        };
        let mut runtime = PluginRuntime::with_settings(self.clone(), &settings);
        let mut results = Vec::new();
        for manifest in &self.manifests {
            for case in &manifest.eval_cases {
                let outcome = runtime.rewrite(&case.input);
                let mut failures = Vec::new();
                if let Some(expected) = &case.expect_rewrite {
                    if &outcome.text != expected {
                        failures.push(format!(
                            "期望改写为「{}」，实际为「{}」",
                            expected, outcome.text
                        ));
                    }
                }
                if let Some(expected_plugin) = &case.expect_plugin {
                    if !outcome
                        .trace
                        .matches
                        .iter()
                        .any(|entry| &entry.plugin == expected_plugin && entry.applied)
                    {
                        failures.push(format!("期望命中插件「{}」", expected_plugin));
                    }
                }
                if let Some(expected) = &case.expect_contains {
                    if !outcome.text.contains(expected) {
                        failures.push(format!("期望结果包含「{}」", expected));
                    }
                }
                results.push(PluginEvalResult {
                    plugin: manifest.id.clone(),
                    input: case.input.clone(),
                    passed: failures.is_empty(),
                    detail: if failures.is_empty() {
                        format!("→ {}", outcome.text)
                    } else {
                        failures.join("；")
                    },
                });
            }
        }
        results
    }

    pub fn render_eval(&self, results: &[PluginEvalResult]) -> String {
        if results.is_empty() {
            return "插件评测：没有包内评测用例。".to_string();
        }
        let passed = results.iter().filter(|result| result.passed).count();
        let mut lines = vec![format!("插件评测：{}/{} 通过。", passed, results.len())];
        for result in results {
            lines.push(format!(
                "  {} [{}] {} {}",
                if result.passed { "✓" } else { "✗" },
                result.plugin,
                result.input,
                result.detail
            ));
        }
        lines.join("\n")
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PluginEvalResult {
    pub plugin: String,
    pub input: String,
    pub passed: bool,
    pub detail: String,
}
fn validate_manifest(manifest: &PluginManifest, file: &Path) -> Vec<PluginIssue> {
    let mut issues = Vec::new();
    let plugin = Some(manifest.id.as_str());
    if manifest.id.trim().is_empty() {
        issues.push(PluginIssue::error(
            plugin,
            Some(file),
            "id",
            "插件 id 不能为空。".to_string(),
        ));
    } else if !manifest
        .id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_'))
    {
        issues.push(PluginIssue::error(
            plugin,
            Some(file),
            "id",
            "插件 id 只能包含 ASCII 字母、数字、点、短横线和下划线。".to_string(),
        ));
    }
    if manifest.api != PLUGIN_API_VERSION {
        issues.push(PluginIssue::error(
            plugin,
            Some(file),
            "api",
            format!(
                "清单 API 版本 {} 不受支持（宿主支持 {}）。",
                manifest.api, PLUGIN_API_VERSION
            ),
        ));
    }
    let mut contribution_ids = BTreeSet::new();
    for contribution in &manifest.contributions {
        if contribution.id().trim().is_empty() {
            issues.push(PluginIssue::error(
                plugin,
                Some(file),
                "contributions",
                "贡献 id 不能为空。".to_string(),
            ));
        } else if !contribution_ids.insert(contribution.id().to_string()) {
            issues.push(PluginIssue::error(
                plugin,
                Some(file),
                contribution.id(),
                "同一插件内贡献 id 重复。".to_string(),
            ));
        }
        issues.extend(validate_contribution(manifest, contribution));
    }
    issues
}
fn validate_contribution(
    manifest: &PluginManifest,
    contribution: &Contribution,
) -> Vec<PluginIssue> {
    let plugin = Some(manifest.id.as_str());
    let id = contribution.id();
    let mut issues = Vec::new();
    match contribution {
        Contribution::Lexicon { entries, .. } => {
            if entries.is_empty() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "词表贡献没有词条。".to_string(),
                ));
            }
            for entry in entries {
                if entry.surface.trim().is_empty() || entry.canonical.trim().is_empty() {
                    issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        "词表词面与规范名都不能为空。".to_string(),
                    ));
                }
            }
        }
        Contribution::Ontology {
            predicates, types, ..
        } => {
            if predicates.is_empty() && types.is_empty() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "本体贡献没有谓词或类型声明。".to_string(),
                ));
            }
            for predicate in predicates {
                let name = predicate.name.trim();
                if name.is_empty() {
                    issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        "本体谓词 name 不能为空。".to_string(),
                    ));
                    continue;
                }
                for alias in &predicate.aliases {
                    if alias.trim().is_empty() {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体谓词「{}」声明了空别名，已忽略。", name),
                        ));
                    } else if alias.trim() == name {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体谓词「{}」的别名与规范名相同，已忽略。", name),
                        ));
                    }
                }
                if predicate.inverse.as_deref().map(str::trim) == Some(name) {
                    issues.push(PluginIssue::warning(
                        plugin,
                        id,
                        format!("本体谓词「{}」不能与自身互逆，已忽略该声明。", name),
                    ));
                }
                for class in predicate.domain.iter().chain(predicate.range.iter()) {
                    if class.trim().is_empty() {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体谓词「{}」声明了空的定义域/值域，已忽略。", name),
                        ));
                    }
                }
            }
            for entity_type in types {
                let name = entity_type.name.trim();
                if name.is_empty() {
                    issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        "本体类型 name 不能为空。".to_string(),
                    ));
                    continue;
                }
                for alias in &entity_type.aliases {
                    if alias.trim().is_empty() {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」声明了空别名，已忽略。", name),
                        ));
                    } else if alias.trim() == name {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」的别名与规范名相同，已忽略。", name),
                        ));
                    }
                }
                for sup in &entity_type.subclass_of {
                    if sup.trim().is_empty() {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」声明了空的父类，已忽略。", name),
                        ));
                    } else if sup.trim() == name {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」不能以自身为父类，已忽略。", name),
                        ));
                    }
                }
                for right in &entity_type.disjoint_with {
                    if right.trim().is_empty() {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」声明了空的互斥类，已忽略。", name),
                        ));
                    } else if right.trim() == name {
                        issues.push(PluginIssue::warning(
                            plugin,
                            id,
                            format!("本体类型「{}」不能与自身互斥，已忽略。", name),
                        ));
                    }
                }
            }
        }
        Contribution::Reference {
            quantifiers,
            role_words,
            surnames,
            ..
        } => {
            if quantifiers.is_empty() && role_words.is_empty() && surnames.is_empty() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "指称贡献没有任何线索词。".to_string(),
                ));
            }
        }
        Contribution::Rewrite {
            pattern,
            replacement,
            ..
        } => match (
            validate_slot_template(pattern, true),
            validate_slot_template(replacement, false),
        ) {
            (Err(message), _) | (_, Err(message)) => issues.push(PluginIssue::error(
                plugin,
                Some(Path::new("<manifest>")),
                id,
                message,
            )),
            (Ok(pattern_segments), Ok(replacement_segments)) => {
                let mut pattern_slots = BTreeSet::new();
                collect_slots(&pattern_segments, &mut pattern_slots);
                let mut replacement_slots = BTreeSet::new();
                collect_slots(&replacement_segments, &mut replacement_slots);
                if let Some(missing) = replacement_slots.difference(&pattern_slots).next() {
                    issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        format!("替换式引用未捕获的槽位「{}」。", missing),
                    ));
                }
            }
        },
        Contribution::Recipe {
            patterns,
            question,
            template,
            attribute,
            direction,
            answer_types,
            ..
        } => {
            if patterns.is_empty() {
                issues.push(PluginIssue::error(
                    plugin,
                    Some(Path::new("<manifest>")),
                    id,
                    "配方至少需要一个问法模式。".to_string(),
                ));
            }
            let mut pattern_slots = BTreeSet::new();
            for pattern in patterns {
                match validate_slot_template(pattern, true) {
                    Ok(segments) => collect_slots(&segments, &mut pattern_slots),
                    Err(message) => issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        message,
                    )),
                }
            }
            if !pattern_slots.contains("entity") {
                issues.push(PluginIssue::error(
                    plugin,
                    Some(Path::new("<manifest>")),
                    id,
                    "配方模式必须包含 {entity} 槽位。".to_string(),
                ));
            }
            match validate_slot_template(question, false) {
                Ok(segments) => {
                    let mut question_slots = BTreeSet::new();
                    collect_slots(&segments, &mut question_slots);
                    if let Some(missing) = question_slots.difference(&pattern_slots).next() {
                        issues.push(PluginIssue::error(
                            plugin,
                            Some(Path::new("<manifest>")),
                            id,
                            format!("规范问句引用未捕获的槽位「{}」。", missing),
                        ));
                    }
                }
                Err(message) => issues.push(PluginIssue::error(
                    plugin,
                    Some(Path::new("<manifest>")),
                    id,
                    message,
                )),
            }
            if let Some(attribute) = attribute {
                if attribute.trim().is_empty() {
                    issues.push(PluginIssue::warning(
                        plugin,
                        id,
                        "配方的 attribute 为空字符串，已忽略。".to_string(),
                    ));
                }
            }
            for answer_type in answer_types {
                if answer_type.trim().is_empty() {
                    issues.push(PluginIssue::warning(
                        plugin,
                        id,
                        "配方的 answer_types 含空字符串，已忽略。".to_string(),
                    ));
                }
            }
            if direction.is_inverse() && attribute.is_none() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "逆向配方没有 attribute，无法确定查询关系。".to_string(),
                ));
            }
            if let Some(template) = template {
                if template.trim().is_empty() {
                    issues.push(PluginIssue::error(
                        plugin,
                        Some(Path::new("<manifest>")),
                        id,
                        "配方模板 id 不能为空。".to_string(),
                    ));
                }
            }
        }
        Contribution::Policy { when, reply, .. } => {
            if when.is_empty() || reply.trim().is_empty() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "策略贡献缺少触发词或回复文本。".to_string(),
                ));
            }
        }
        Contribution::Template { template, .. } => {
            if let Err(message) = validate_slot_template(template, false) {
                issues.push(PluginIssue::error(
                    plugin,
                    Some(Path::new("<manifest>")),
                    id,
                    message,
                ));
            }
        }
        Contribution::Style { prefix, suffix, .. } => {
            if prefix.trim().is_empty() && suffix.trim().is_empty() {
                issues.push(PluginIssue::warning(
                    plugin,
                    id,
                    "风格贡献没有前后缀。".to_string(),
                ));
            }
        }
        Contribution::Rules { .. } => {
            if let Some(pack) = contribution.rule_pack(&manifest.id) {
                for issue in crate::rule_pack::validate_pack(&pack) {
                    issues.push(PluginIssue {
                        severity: match issue.severity {
                            crate::rule_pack::IssueSeverity::Error => PluginSeverity::Error,
                            crate::rule_pack::IssueSeverity::Warning => PluginSeverity::Warning,
                        },
                        plugin: Some(manifest.id.clone()),
                        file: None,
                        field: issue.field,
                        message: issue.message,
                    });
                }
            }
        }
    }
    issues
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Segment {
    Literal(String),
    Slot(String),
}
fn validate_slot_template(template: &str, require_slot: bool) -> Result<Vec<Segment>, String> {
    let segments = parse_template(template)?;
    if require_slot
        && !segments
            .iter()
            .any(|segment| matches!(segment, Segment::Slot(_)))
    {
        return Err("模式必须至少包含一个 {槽位}。".to_string());
    }
    Ok(segments)
}
fn parse_template(template: &str) -> Result<Vec<Segment>, String> {
    if template.trim().is_empty() {
        return Err("模板不能为空。".to_string());
    }
    let mut segments = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        if rest[..open].contains('}') {
            return Err(format!("模板「{}」存在未配对的 }}。", template));
        }
        let before = &rest[..open];
        if !before.is_empty() {
            segments.push(Segment::Literal(before.to_string()));
        }
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            return Err(format!("模板「{}」存在未闭合的 {{。", template));
        };
        let name = &after[..close];
        if name.is_empty() {
            return Err(format!("模板「{}」存在空槽位 {{}}。", template));
        }
        if !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Err(format!(
                "槽位「{}」只能包含 ASCII 字母、数字与下划线。",
                name
            ));
        }
        segments.push(Segment::Slot(name.to_string()));
        rest = &after[close + 1..];
    }
    if rest.contains('}') {
        return Err(format!("模板「{}」存在未配对的 }}。", template));
    }
    if !rest.is_empty() {
        segments.push(Segment::Literal(rest.to_string()));
    }
    for window in segments.windows(2) {
        if matches!(window[0], Segment::Slot(_)) && matches!(window[1], Segment::Slot(_)) {
            return Err(format!("模板「{}」存在相邻槽位，无法确定边界。", template));
        }
    }
    let mut seen = BTreeSet::new();
    for segment in &segments {
        if let Segment::Slot(name) = segment {
            if !seen.insert(name.clone()) {
                return Err(format!("模板「{}」重复使用槽位 {{{}}}。", template, name));
            }
        }
    }
    Ok(segments)
}
fn collect_slots(segments: &[Segment], out: &mut BTreeSet<String>) {
    for segment in segments {
        if let Segment::Slot(name) = segment {
            out.insert(name.clone());
        }
    }
}
fn literal_length(segments: &[Segment]) -> usize {
    segments
        .iter()
        .map(|segment| match segment {
            Segment::Literal(text) => text.chars().count(),
            Segment::Slot(_) => 0,
        })
        .sum()
}
/// 整串匹配模式；槽位非空，字面量按顺序出现。
fn trim_query_surface(text: &str) -> &str {
    let mut text = text.trim();
    for prefix in ["请问一下", "想问一下", "请告诉我", "请问"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim_start();
            break;
        }
    }
    loop {
        let trimmed = text.trim_end_matches(|ch: char| {
            matches!(
                ch,
                '？' | '?' | '。' | '！' | '!' | '；' | ';' | '，' | ',' | '…'
            )
        });
        let trimmed = trimmed
            .trim_end_matches(|ch: char| matches!(ch, '呢' | '吗' | '吧' | '啊' | '呀' | '嘛' | '哦' | '哈' | '啦'))
            .trim_end();
        if trimmed == text {
            break;
        }
        text = trimmed;
    }
    text
}
fn match_segments(segments: &[Segment], text: &str) -> Option<BTreeMap<String, String>> {
    let mut captures = BTreeMap::new();
    let mut pos = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        match segment {
            Segment::Literal(literal) => {
                let rest = text.get(pos..)?;
                if index == 0 {
                    if !rest.starts_with(literal) {
                        return None;
                    }
                    pos += literal.len();
                } else if index == segments.len() - 1 {
                    if !rest.ends_with(literal) {
                        return None;
                    }
                    pos = text.len();
                } else {
                    if !rest.starts_with(literal) {
                        return None;
                    }
                    pos += literal.len();
                }
            }
            Segment::Slot(name) => {
                let next_literal = segments[index + 1..].iter().find_map(|segment| {
                    if let Segment::Literal(literal) = segment {
                        Some(literal)
                    } else {
                        None
                    }
                });
                let end = match next_literal {
                    Some(literal) if index + 1 == segments.len() - 1 => {
                        pos + text.get(pos..)?.rfind(literal)?
                    }
                    Some(literal) => pos + text.get(pos..)?.find(literal)?,
                    None => text.len(),
                };
                if end <= pos {
                    return None;
                }
                captures.insert(name.clone(), text[pos..end].to_string());
                pos = end;
            }
        }
    }
    if pos != text.len() {
        return None;
    }
    Some(captures)
}
fn render_segments(
    segments: &[Segment],
    slots: &BTreeMap<String, String>,
) -> Result<String, String> {
    let mut out = String::new();
    for segment in segments {
        match segment {
            Segment::Literal(text) => out.push_str(text),
            Segment::Slot(name) => match slots.get(name) {
                Some(value) if !value.trim().is_empty() => out.push_str(value),
                _ => return Err(format!("槽位「{}」未接地。", name)),
            },
        }
    }
    Ok(out)
}
/// 一次改写/配方的命中记录。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginMatch {
    pub plugin: String,
    pub contribution: String,
    pub kind: String,
    pub mode: PluginMode,
    pub applied: bool,
    pub detail: String,
}
/// 配方查询方向：实体是关系的主语还是宾语。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecipeDirection {
    #[default]
    Forward,
    Inverse,
}
impl RecipeDirection {
    pub fn label(self) -> &'static str {
        match self {
            RecipeDirection::Forward => "正向",
            RecipeDirection::Inverse => "逆向",
        }
    }

    pub fn is_inverse(self) -> bool {
        matches!(self, RecipeDirection::Inverse)
    }
}
/// 配方 `{entity}` 捕获到的是否为可回指当前焦点的指称词。
/// 只认封闭的代词/指示词集合，不做模糊匹配，避免把实体名误当回指。
pub fn is_reference_surface(surface: &str) -> bool {
    matches!(
        surface.trim(),
        "他" | "她" | "它"
            | "他们"
            | "她们"
            | "它们"
            | "这位"
            | "那位"
            | "该"
            | "这个"
            | "那个"
            | "其"
    )
}
/// 配方被选中后的可执行调用信息。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeInvocation {
    pub plugin: String,
    pub contribution: String,
    pub question: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub attribute: Option<String>,
    #[serde(default)]
    pub direction: RecipeDirection,
    #[serde(default)]
    pub answer_types: Vec<String>,
    #[serde(default)]
    pub slots: BTreeMap<String, String>,
}
/// 一轮插件轨迹（Audit 阶段的可观察载体）。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginTrace {
    pub input: String,
    pub effective: String,
    #[serde(default)]
    pub matches: Vec<PluginMatch>,
    #[serde(default)]
    pub recipe: Option<RecipeInvocation>,
    #[serde(default)]
    pub notes: Vec<String>,
}
impl PluginTrace {
    pub fn changed(&self) -> bool {
        self.input != self.effective
    }

    pub fn used_plugins(&self) -> Vec<String> {
        let mut used: Vec<String> = self
            .matches
            .iter()
            .filter(|entry| entry.applied)
            .map(|entry| entry.plugin.clone())
            .collect();
        used.sort();
        used.dedup();
        used
    }
}
/// `PluginRuntime::rewrite` 的结果：生效文本 + 可审计轨迹。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginOutcome {
    pub text: String,
    pub trace: PluginTrace,
}
impl PluginOutcome {
    pub fn changed(&self) -> bool {
        self.trace.changed()
    }
}
/// 配方答案的组合结果。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PluginComposedAnswer {
    pub text: String,
    pub template: Option<String>,
    pub notes: Vec<String>,
}
/// 会话级插件状态：只持久化用户显式覆盖的模式，不固化清单默认值。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginSessionState {
    #[serde(default)]
    pub modes: BTreeMap<String, PluginMode>,
    #[serde(default)]
    pub last_used: Vec<String>,
}
impl PluginSessionState {
    pub fn load(path: impl AsRef<Path>) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|error| error.to_string())?;
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        fs::write(path, text).map_err(|error| error.to_string())
    }
}
/// 插件命令的执行结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginCommandOutcome {
    pub text: String,
    pub state_changed: bool,
    pub success: bool,
}
/// 插件命令解析结果的内部表示。
#[derive(Clone, Debug, PartialEq, Eq)]
enum ParsedPluginCommand {
    Report,
    Diagnose,
    Evaluate,
    Rules,
    Why,
    Enable(String),
    Disable(String),
    SetMode(String, PluginMode),
}
/// 运行时：在清单之上叠加模式、依赖裁决、改写路由、模板接地与会话状态。
#[derive(Clone, Debug)]
pub struct PluginRuntime {
    registry: PluginRegistry,
    manifests: Vec<PluginManifest>,
    modes: BTreeMap<String, PluginMode>,
    overrides: BTreeMap<String, PluginMode>,
    activation_issues: Vec<PluginIssue>,
    last_trace: Option<PluginTrace>,
    last_used: Vec<String>,
}

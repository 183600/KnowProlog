//! 声明式插件（符号专家）机制。
//!
//! 每个插件是一份 `manifest.toml`：身份（id/version/api）、依赖与冲突、
//! 以及若干**带类型的贡献**。宿主只认识固定的贡献类型与固定阶段，
//! 不加载动态库、不执行插件代码；插件只能声明「要什么」，不能自行行动。
//!
//! 本模块实现七个阶段中的前五个可执行阶段：
//! `Lexicon → Reference → Frame → Retrieve → Compose`（Reference 目前仅校验与登记），
//! 以及贯穿全程的 `Audit`（清单 lint、启用裁决、逐轮 trace、包内评测）。
//!
//! 运行模式与项目既有认识论一致：
//! - `off`：完全不加载贡献；
//! - `observe`：只计算「如果启用会怎样」并写入 trace，不影响回答；
//! - `on`：真正参与改写 / 配方 / 模板 / 风格。
//!
//! 安全边界：改写与模板都是受限的槽位语法（`{entity}` 等），不是正则、
//! 不是脚本；模板必须所有槽位都接地才渲染，否则回退原文，绝不凭空补值。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

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
            | Contribution::Style { id, .. } => id,
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
            | Contribution::Style { description, .. } => description,
        }
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

impl PluginRuntime {
    pub fn with_settings(registry: PluginRegistry, settings: &PluginSettings) -> Self {
        let manifests = registry.manifests().to_vec();
        let mut desired = BTreeMap::new();
        for manifest in &manifests {
            desired.insert(manifest.id.clone(), manifest.default_mode);
        }
        for id in &settings.enabled {
            if manifests.iter().any(|manifest| &manifest.id == id) {
                desired.insert(id.clone(), PluginMode::On);
            }
        }
        for (id, mode) in &settings.modes {
            if manifests.iter().any(|manifest| &manifest.id == id) {
                desired.insert(id.clone(), *mode);
            }
        }
        let unknown: BTreeSet<&str> = settings
            .enabled
            .iter()
            .map(String::as_str)
            .chain(settings.modes.keys().map(String::as_str))
            .filter(|id| !manifests.iter().any(|manifest| manifest.id == *id))
            .collect();
        let mut runtime = PluginRuntime {
            registry,
            manifests,
            modes: BTreeMap::new(),
            overrides: BTreeMap::new(),
            activation_issues: Vec::new(),
            last_trace: None,
            last_used: Vec::new(),
        };
        runtime.resolve_initial(desired);
        runtime
            .activation_issues
            .extend(unknown.into_iter().map(|id| {
                PluginIssue::warning(
                    Some(id),
                    "config.plugins",
                    format!("配置引用了未安装的插件「{}」，已忽略。", id),
                )
            }));
        runtime
    }

    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    pub fn installed_count(&self) -> usize {
        self.manifests.len()
    }

    pub fn enabled_count(&self) -> usize {
        self.modes
            .values()
            .filter(|mode| **mode == PluginMode::On)
            .count()
    }

    pub fn mode(&self, id: &str) -> Option<PluginMode> {
        self.modes.get(id).copied()
    }

    pub fn modes(&self) -> &BTreeMap<String, PluginMode> {
        &self.modes
    }

    pub fn manifests(&self) -> &[PluginManifest] {
        &self.manifests
    }

    pub fn issues(&self) -> Vec<PluginIssue> {
        let mut issues = self.registry.issues().to_vec();
        issues.extend(self.activation_issues.iter().cloned());
        issues
    }

    pub fn last_trace(&self) -> Option<&PluginTrace> {
        self.last_trace.as_ref()
    }

    pub fn session_state(&self) -> PluginSessionState {
        PluginSessionState {
            modes: self.overrides.clone(),
            last_used: self.last_used.clone(),
        }
    }

    /// 恢复用户显式覆盖；返回成功应用的条数。旧包/冲突包会留在诊断里。
    pub fn restore_session_state(&mut self, state: PluginSessionState) -> usize {
        let mut applied = 0;
        for (id, mode) in state.modes {
            match self.set_mode(&id, mode) {
                Ok(_) => applied += 1,
                Err(message) => self.activation_issues.push(PluginIssue::warning(
                    Some(&id),
                    "session",
                    format!("会话状态未恢复：{}", message),
                )),
            }
        }
        self.last_used = state
            .last_used
            .into_iter()
            .filter(|id| self.modes.contains_key(id))
            .collect();
        applied
    }

    pub fn manifest(&self, id: &str) -> Option<&PluginManifest> {
        self.manifests.iter().find(|manifest| manifest.id == id)
    }

    /// 显式切换模式。启用会递归拉起依赖；停用若被其他启用包依赖则拒绝。
    pub fn set_mode(&mut self, id: &str, mode: PluginMode) -> Result<Vec<String>, String> {
        if !self.modes.contains_key(id) {
            return Err(format!("没有已安装的插件「{}」。", id));
        }
        let previous_modes = self.modes.clone();
        let previous_overrides = self.overrides.clone();
        let mut proposed = self.modes.clone();
        let mut auto_enabled = Vec::new();

        match mode {
            PluginMode::Off | PluginMode::Observe => {
                let dependents = self.enabled_dependents(id);
                if !dependents.is_empty() {
                    return Err(format!(
                        "「{}」仍被启用中的插件依赖：{}。请先停用依赖它的插件。",
                        id,
                        dependents.join("、")
                    ));
                }
                proposed.insert(id.to_string(), mode);
            }
            PluginMode::On => {
                proposed.insert(id.to_string(), PluginMode::On);
                let mut queue = vec![id.to_string()];
                let mut seen = BTreeSet::new();
                while let Some(current) = queue.pop() {
                    if !seen.insert(current.clone()) {
                        continue;
                    }
                    let Some(manifest) = self.manifest(&current) else {
                        return Err(format!("依赖的插件「{}」未安装。", current));
                    };
                    for required in manifest.requires.clone() {
                        if self.manifest(&required).is_none() {
                            return Err(format!(
                                "插件「{}」依赖的「{}」未安装，无法启用。",
                                manifest.id, required
                            ));
                        }
                        if proposed.get(&required) != Some(&PluginMode::On) {
                            proposed.insert(required.clone(), PluginMode::On);
                            if required != id {
                                auto_enabled.push(required.clone());
                            }
                        }
                        queue.push(required);
                    }
                }
                if let Some((left, right)) = self.first_conflict(&proposed) {
                    return Err(format!(
                        "插件「{}」与「{}」声明冲突，不能同时启用。",
                        left, right
                    ));
                }
            }
        }

        self.modes = proposed;
        self.overrides.insert(id.to_string(), mode);
        for dependency in &auto_enabled {
            self.overrides.insert(dependency.clone(), PluginMode::On);
        }
        self.resolve_activation_issues();
        if self.modes.get(id) != Some(&mode) {
            self.modes = previous_modes;
            self.overrides = previous_overrides;
            return Err(format!("启用「{}」会破坏依赖/冲突约束，已回滚。", id));
        }
        auto_enabled.sort();
        auto_enabled.dedup();
        Ok(auto_enabled)
    }

    fn enabled_dependents(&self, id: &str) -> Vec<String> {
        self.manifests
            .iter()
            .filter(|manifest| {
                manifest.id != id
                    && self.modes.get(&manifest.id) == Some(&PluginMode::On)
                    && manifest.requires.iter().any(|required| required == id)
            })
            .map(|manifest| manifest.display_name().to_string())
            .collect()
    }

    fn resolve_initial(&mut self, desired: BTreeMap<String, PluginMode>) {
        let mut modes = desired;
        let mut blocked = BTreeSet::new();
        let mut issues = Vec::new();
        let ids: Vec<String> = self.manifests.iter().map(|m| m.id.clone()).collect();
        let mut rounds = 0;
        loop {
            rounds += 1;
            let mut changed = false;
            for id in &ids {
                if modes.get(id) != Some(&PluginMode::On) {
                    continue;
                }
                let Some(manifest) = self.manifest(id) else {
                    continue;
                };
                for required in manifest.requires.clone() {
                    if self.manifest(&required).is_none() {
                        modes.insert(id.clone(), PluginMode::Off);
                        blocked.insert(id.clone());
                        issues.push(PluginIssue::warning(
                            Some(id),
                            "requires",
                            format!("依赖的插件「{}」未安装，已停用。", required),
                        ));
                        changed = true;
                        break;
                    }
                    if blocked.contains(&required) {
                        modes.insert(id.clone(), PluginMode::Off);
                        blocked.insert(id.clone());
                        issues.push(PluginIssue::warning(
                            Some(id),
                            "requires",
                            format!("依赖的插件「{}」不可用，已停用。", required),
                        ));
                        changed = true;
                        break;
                    }
                    if modes.get(&required) != Some(&PluginMode::On) {
                        modes.insert(required.clone(), PluginMode::On);
                        changed = true;
                    }
                }
            }
            for left_index in 0..ids.len() {
                for right_index in left_index + 1..ids.len() {
                    let left = &ids[left_index];
                    let right = &ids[right_index];
                    if modes.get(left) != Some(&PluginMode::On)
                        || modes.get(right) != Some(&PluginMode::On)
                    {
                        continue;
                    }
                    if self.conflict_between(left, right) {
                        let loser = right.clone();
                        modes.insert(loser.clone(), PluginMode::Off);
                        blocked.insert(loser.clone());
                        issues.push(PluginIssue::warning(
                            Some(&loser),
                            "conflicts",
                            format!("与「{}」冲突，按 id 排序停用后者。", left),
                        ));
                        changed = true;
                    }
                }
            }
            if !changed || rounds > ids.len() * 2 + 2 {
                break;
            }
        }
        self.modes = modes;
        self.activation_issues = dedup_issues(issues);
    }

    fn resolve_activation_issues(&mut self) {
        // 显式 set_mode 已做前置检查；这里只同步诊断，不改动模式。
        let mut issues = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for required in &manifest.requires {
                if self.modes.get(required) != Some(&PluginMode::On) {
                    issues.push(PluginIssue::warning(
                        Some(&manifest.id),
                        "requires",
                        format!("依赖的插件「{}」未启用。", required),
                    ));
                }
            }
        }
        if let Some((left, right)) = self.first_conflict(&self.modes) {
            issues.push(PluginIssue::warning(
                Some(&left),
                "conflicts",
                format!("与「{}」同时启用。", right),
            ));
        }
        self.activation_issues = dedup_issues(issues);
    }

    fn first_conflict(&self, modes: &BTreeMap<String, PluginMode>) -> Option<(String, String)> {
        let ids: Vec<&String> = modes
            .iter()
            .filter(|(_, mode)| **mode == PluginMode::On)
            .map(|(id, _)| id)
            .collect();
        for left_index in 0..ids.len() {
            for right_index in left_index + 1..ids.len() {
                if self.conflict_between(ids[left_index], ids[right_index]) {
                    return Some((ids[left_index].clone(), ids[right_index].clone()));
                }
            }
        }
        None
    }

    fn conflict_between(&self, left: &str, right: &str) -> bool {
        let left_conflicts = self
            .manifest(left)
            .map(|manifest| manifest.conflicts.iter().any(|id| id == right))
            .unwrap_or(false);
        let right_conflicts = self
            .manifest(right)
            .map(|manifest| manifest.conflicts.iter().any(|id| id == left))
            .unwrap_or(false);
        left_conflicts || right_conflicts
    }

    /// 词法 → 改写/配方 →（可选）观察。返回生效文本与完整 trace。
    pub fn rewrite(&mut self, input: &str) -> PluginOutcome {
        let mut trace = PluginTrace {
            input: input.to_string(),
            effective: input.to_string(),
            ..PluginTrace::default()
        };
        if self.manifests.is_empty() {
            self.last_trace = Some(trace.clone());
            return PluginOutcome {
                text: trace.effective.clone(),
                trace,
            };
        }

        // 1. Lexicon + ontology aliases：最长匹配、单遍替换；观察模式只记账，
        //    且匹配的是原文（不会出现「电影」再被「影」二次替换的堆叠错误）。
        let lexicon = self.lexicon_pairs();
        let mut text = input.to_string();
        if !lexicon.is_empty() {
            let (normalized, matches) = apply_lexicon(input, &lexicon);
            text = normalized;
            trace.matches.extend(matches);
        }
        trace.effective = text.clone();

        // 2. Rewrite / Recipe：最多连续三轮，避免插件互相改写形成环。
        let mut seen = BTreeSet::new();
        seen.insert(text.clone());
        for _ in 0..MAX_REWRITE_PASSES {
            let candidates = self.match_frame(&text);
            if candidates.is_empty() {
                break;
            }
            let observed: Vec<&FrameCandidate> = candidates
                .iter()
                .filter(|candidate| candidate.mode == PluginMode::Observe)
                .collect();
            for candidate in observed {
                trace.matches.push(PluginMatch {
                    plugin: candidate.plugin.clone(),
                    contribution: candidate.contribution.clone(),
                    kind: candidate.kind.to_string(),
                    mode: PluginMode::Observe,
                    applied: false,
                    detail: candidate.detail.clone(),
                });
            }
            let on_candidates: Vec<&FrameCandidate> = candidates
                .iter()
                .filter(|candidate| candidate.mode == PluginMode::On)
                .collect();
            let Some(chosen) = on_candidates.iter().copied().max_by(|left, right| {
                left.specificity
                    .cmp(&right.specificity)
                    .then_with(|| left.recipe.cmp(&right.recipe))
                    .then_with(|| right.plugin.cmp(&left.plugin))
                    .then_with(|| right.contribution.cmp(&left.contribution))
            }) else {
                break;
            };
            let distinct: BTreeSet<&str> = on_candidates
                .iter()
                .map(|candidate| candidate.output.as_str())
                .collect();
            if distinct.len() > 1 {
                trace.notes.push(format!(
                    "多个启用插件命中同一问法，按具体度与 id 选择「{} / {}」。",
                    chosen.plugin, chosen.contribution
                ));
            }
            trace.matches.push(PluginMatch {
                plugin: chosen.plugin.clone(),
                contribution: chosen.contribution.clone(),
                kind: chosen.kind.to_string(),
                mode: PluginMode::On,
                applied: true,
                detail: chosen.detail.clone(),
            });
            if let Some(recipe) = &chosen.recipe_invocation {
                trace.recipe = Some(recipe.clone());
            } else {
                // 后续普通改写改变了规范问句：先前配方的答案与当前问法不一致。
                trace.recipe = None;
            }
            let next = chosen.output.clone();
            if next == text || !seen.insert(next.clone()) {
                trace.notes.push("改写出现环或空转，已停止。".to_string());
                break;
            }
            text = next;
            trace.effective = text.clone();
        }

        let used = trace.used_plugins();
        if !used.is_empty() {
            self.last_used = used;
        }
        self.last_trace = Some(trace.clone());
        PluginOutcome { text, trace }
    }

    /// 启动时合并进既有谓词对齐器的本体别名（仅启用中的插件）。
    /// 与 `lexicon_pairs` 不同，这里提交给推理器，使图谱侧查询也能识别别名。
    pub fn ontology_aliases(&self) -> Vec<(String, String)> {
        let mut aliases = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Ontology { predicates, .. } = contribution {
                    for predicate in predicates {
                        for alias in &predicate.aliases {
                            if !alias.trim().is_empty()
                                && alias != &predicate.name
                                && !predicate.name.trim().is_empty()
                            {
                                aliases.push((alias.clone(), predicate.name.clone()));
                            }
                        }
                    }
                }
            }
        }
        aliases.sort();
        aliases.dedup();
        aliases
    }

    /// 启动时把插件的本体贡献归约为统一本体程序（仅启用中的插件）。
    ///
    /// 与 `ontology_aliases` 分工：别名只做词面归一，不进公理；子类、互斥、
    /// 定义域/值域、逆关系、传递、对称全部编译成 `OntologyLesson`，交给宿主
    /// 的单一本体推理器。重复声明由程序自身的规范去重吸收，合并是幂等的。
    pub fn ontology_program(&self) -> OntologyProgram {
        let mut lessons = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                let Contribution::Ontology {
                    predicates, types, ..
                } = contribution
                else {
                    continue;
                };
                for entity_type in types {
                    let name = entity_type.name.trim();
                    if name.is_empty() {
                        continue;
                    }
                    for sup in &entity_type.subclass_of {
                        let sup = sup.trim();
                        if sup.is_empty() || sup == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Subclass {
                            sub: name.to_string(),
                            sup: sup.to_string(),
                            original: format!("{}是一种{}", name, sup),
                        });
                    }
                    for right in &entity_type.disjoint_with {
                        let right = right.trim();
                        if right.is_empty() || right == name {
                            continue;
                        }
                        lessons.push(OntologyLesson::Disjoint {
                            left: name.to_string(),
                            right: right.to_string(),
                            original: format!("{}和{}互斥", name, right),
                        });
                    }
                }
                for predicate in predicates {
                    let name = predicate.name.trim();
                    if name.is_empty() {
                        continue;
                    }
                    for class in &predicate.domain {
                        let class = class.trim();
                        if !class.is_empty() {
                            lessons.push(OntologyLesson::Domain {
                                property: name.to_string(),
                                class: class.to_string(),
                                original: format!("{}关系的定义域是{}", name, class),
                            });
                        }
                    }
                    for class in &predicate.range {
                        let class = class.trim();
                        if !class.is_empty() {
                            lessons.push(OntologyLesson::Range {
                                property: name.to_string(),
                                class: class.to_string(),
                                original: format!("{}关系的值域是{}", name, class),
                            });
                        }
                    }
                    if let Some(inverse) = predicate.inverse.as_deref() {
                        let inverse = inverse.trim();
                        if !inverse.is_empty() && inverse != name {
                            lessons.push(OntologyLesson::Inverse {
                                left: name.to_string(),
                                right: inverse.to_string(),
                                original: format!("{}和{}互为逆关系", name, inverse),
                            });
                        }
                    }
                    if predicate.transitive {
                        lessons.push(OntologyLesson::Transitive {
                            property: name.to_string(),
                            original: format!("{}关系可以传递", name),
                        });
                    }
                    if predicate.symmetric {
                        lessons.push(OntologyLesson::Symmetric {
                            property: name.to_string(),
                            original: format!("{}是对称的", name),
                        });
                    }
                }
            }
        }
        OntologyProgram::from_lessons(lessons)
    }

    /// 插件声明的互逆谓词中「需要换向查询」的词面（仅启用中的插件）。
    /// 与 `ontology_aliases` 不同：这里登记的词面在发问时把主宾对调，
    /// 例如「执导」问句命中「导演」事实的反方向。
    pub fn ontology_inverse_surfaces(&self) -> Vec<(String, String)> {
        let mut surfaces = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Ontology { predicates, .. } = contribution {
                    for predicate in predicates {
                        let name = predicate.name.trim();
                        let Some(inverse) = predicate.inverse.as_deref() else {
                            continue;
                        };
                        let inverse = inverse.trim();
                        if !name.is_empty() && !inverse.is_empty() && inverse != name {
                            surfaces.push((inverse.to_string(), name.to_string()));
                        }
                    }
                }
            }
        }
        surfaces.sort();
        surfaces.dedup();
        surfaces
    }

    /// 启动时合并进中文分词/词性词典的指称线索（仅启用中的插件）。
    /// 量词按代词 `r`、角色词按名词 `n`、姓氏按人名 `nr` 标注。
    pub fn reference_lexicon(&self) -> Vec<(String, String)> {
        let mut entries = Vec::new();
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Reference {
                    quantifiers,
                    role_words,
                    surnames,
                    ..
                } = contribution
                {
                    for quantifier in quantifiers {
                        if !quantifier.trim().is_empty() {
                            entries.push((quantifier.clone(), "r".to_string()));
                        }
                    }
                    for role in role_words {
                        if !role.trim().is_empty() {
                            entries.push((role.clone(), "n".to_string()));
                        }
                    }
                    for surname in surnames {
                        if !surname.trim().is_empty() {
                            entries.push((surname.clone(), "nr".to_string()));
                        }
                    }
                }
            }
        }
        entries.sort();
        entries.dedup();
        entries
    }

    /// 派生 NLP 配置：保留宿主原有词性标注，只补入插件指称线索。
    pub fn augmented_nlp_config(&self, base: &crate::NlpConfig) -> crate::NlpConfig {
        let mut config = base.clone();
        for (surface, pos) in self.reference_lexicon() {
            config.pos_dict.entry(surface).or_insert(pos);
        }
        config
    }

    fn has_ontology(&self, id: &str) -> bool {
        self.manifest(id)
            .map(|manifest| {
                manifest
                    .contributions
                    .iter()
                    .any(|contribution| matches!(contribution, Contribution::Ontology { .. }))
            })
            .unwrap_or(false)
    }

    fn lexicon_pairs(&self) -> Vec<(String, String, String, String, PluginMode)> {
        let mut pairs = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                match contribution {
                    Contribution::Lexicon { id, entries, .. } => {
                        for entry in entries {
                            pairs.push((
                                entry.surface.clone(),
                                entry.canonical.clone(),
                                manifest.id.clone(),
                                id.clone(),
                                mode,
                            ));
                        }
                    }
                    Contribution::Ontology {
                        id,
                        predicates,
                        types,
                        ..
                    } => {
                        for predicate in predicates {
                            for alias in &predicate.aliases {
                                pairs.push((
                                    alias.clone(),
                                    predicate.name.clone(),
                                    manifest.id.clone(),
                                    id.clone(),
                                    mode,
                                ));
                            }
                        }
                        for entity_type in types {
                            for alias in &entity_type.aliases {
                                pairs.push((
                                    alias.clone(),
                                    entity_type.name.clone(),
                                    manifest.id.clone(),
                                    id.clone(),
                                    mode,
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        pairs.sort_by(|left, right| {
            right
                .0
                .chars()
                .count()
                .cmp(&left.0.chars().count())
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
        });
        pairs
    }

    fn match_frame(&self, text: &str) -> Vec<FrameCandidate> {
        let mut candidates = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                match contribution {
                    Contribution::Rewrite {
                        id,
                        pattern,
                        replacement,
                        ..
                    } => {
                        let Ok(pattern_segments) = parse_template(pattern) else {
                            continue;
                        };
                        let Ok(replacement_segments) = parse_template(replacement) else {
                            continue;
                        };
                        let Some(captures) = match_segments(&pattern_segments, text) else {
                            continue;
                        };
                        let Ok(output) = render_segments(&replacement_segments, &captures) else {
                            continue;
                        };
                        candidates.push(FrameCandidate {
                            plugin: manifest.id.clone(),
                            contribution: id.clone(),
                            kind: "rewrite",
                            mode,
                            specificity: literal_length(&pattern_segments),
                            output: output.clone(),
                            detail: format!("{} → {}", text, output),
                            recipe_invocation: None,
                            recipe: false,
                        });
                    }
                    Contribution::Recipe {
                        id,
                        patterns,
                        question,
                        template,
                        attribute,
                        direction,
                        answer_types,
                        ..
                    } => {
                        for pattern in patterns {
                            let Ok(pattern_segments) = parse_template(pattern) else {
                                continue;
                            };
                            let Some(captures) = match_segments(&pattern_segments, text) else {
                                continue;
                            };
                            let Ok(question_segments) = parse_template(question) else {
                                continue;
                            };
                            let Ok(output) = render_segments(&question_segments, &captures) else {
                                continue;
                            };
                            candidates.push(FrameCandidate {
                                plugin: manifest.id.clone(),
                                contribution: id.clone(),
                                kind: "recipe",
                                mode,
                                specificity: literal_length(&pattern_segments),
                                output: output.clone(),
                                detail: format!("{} → {}（配方）", text, output),
                                recipe_invocation: Some(RecipeInvocation {
                                    plugin: manifest.id.clone(),
                                    contribution: id.clone(),
                                    question: output,
                                    template: template.clone(),
                                    attribute: attribute.clone(),
                                    direction: *direction,
                                    answer_types: answer_types.clone(),
                                    slots: captures,
                                }),
                                recipe: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        candidates
    }

    /// 用配方选中的模板渲染答案；任一槽位未接地则回退原答案并记 trace。
    pub fn compose_answer(
        &mut self,
        outcome: &PluginOutcome,
        answer: &str,
    ) -> PluginComposedAnswer {
        self.compose_answer_with_evidence(outcome, answer, &[])
    }

    /// 与 [`Self::compose_answer`] 相同，但额外提供 `{evidence}` 槽位。
    /// 依据必须由调用方从事实/公理闭包取出，插件模板只能复述、不能编造。
    pub fn compose_answer_with_evidence(
        &mut self,
        outcome: &PluginOutcome,
        answer: &str,
        evidence: &[String],
    ) -> PluginComposedAnswer {
        let mut trace = outcome.trace.clone();
        let mut text = answer.to_string();
        let mut notes = Vec::new();
        let mut template_applied = None;
        if let Some(recipe) = &outcome.trace.recipe {
            if let Some(template_id) = &recipe.template {
                let template = self.template_contribution(template_id);
                match template {
                    Some((plugin, mode, segments)) if mode == PluginMode::On => {
                        let mut slots = recipe.slots.clone();
                        if !answer.trim().is_empty() {
                            slots.insert("answer".to_string(), answer.to_string());
                        }
                        if let Some(attribute) = &recipe.attribute {
                            slots.insert("attribute".to_string(), attribute.clone());
                        }
                        if !evidence.is_empty() {
                            slots.insert("evidence".to_string(), evidence.join("；"));
                        }
                        slots.insert("plugin".to_string(), recipe.plugin.clone());
                        slots.insert("recipe".to_string(), recipe.contribution.clone());
                        match render_segments(&segments, &slots) {
                            Ok(rendered) => {
                                text = rendered;
                                template_applied = Some(template_id.clone());
                                trace.matches.push(PluginMatch {
                                    plugin: plugin.clone(),
                                    contribution: template_id.clone(),
                                    kind: "template".to_string(),
                                    mode,
                                    applied: true,
                                    detail: "槽位接地通过".to_string(),
                                });
                            }
                            Err(message) => {
                                notes.push(format!(
                                    "模板「{}」未渲染，回退原答案：{}",
                                    template_id, message
                                ));
                            }
                        }
                    }
                    Some((plugin, mode, _)) => {
                        notes.push(format!(
                            "模板「{}」属于{}模式插件「{}」，未接管。",
                            template_id,
                            mode.label(),
                            plugin
                        ));
                    }
                    None => notes.push(format!("模板「{}」不存在，回退原答案。", template_id)),
                }
            }
        }

        let (styled, style_matches) = self.style_with_matches(&text);
        text = styled;
        trace.matches.extend(style_matches);
        trace.notes.extend(notes.clone());
        self.last_trace = Some(trace);
        PluginComposedAnswer {
            text,
            template: template_applied,
            notes,
        }
    }

    fn template_contribution(&self, id: &str) -> Option<(String, PluginMode, Vec<Segment>)> {
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if !mode.is_active() {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Template {
                    id: template_id,
                    template,
                    ..
                } = contribution
                {
                    if template_id == id {
                        if let Ok(segments) = parse_template(template) {
                            return Some((manifest.id.clone(), mode, segments));
                        }
                    }
                }
            }
        }
        None
    }

    /// 核心管线答不了时的声明式兜底策略：按触发词长度、插件 id、贡献 id 择优。
    ///
    /// 只返回文本与可审计的贡献路径，不伪造知识；调用方仍需决定是否采纳。
    pub fn policy_reply(&mut self, input: &str) -> Option<(String, String)> {
        let mut best: Option<(String, String, String, usize, String)> = None;
        for manifest in &self.manifests {
            if self.modes.get(&manifest.id) != Some(&PluginMode::On) {
                continue;
            }
            for contribution in &manifest.contributions {
                let Contribution::Policy {
                    id, when, reply, ..
                } = contribution
                else {
                    continue;
                };
                for trigger in when {
                    if trigger.trim().is_empty() || !input.contains(trigger.as_str()) {
                        continue;
                    }
                    let score = trigger.chars().count();
                    let better = best.as_ref().is_none_or(|current| {
                        score > current.3
                            || (score == current.3
                                && (manifest.id.as_str(), id.as_str())
                                    < (current.0.as_str(), current.1.as_str()))
                    });
                    if better {
                        best = Some((
                            manifest.id.clone(),
                            id.clone(),
                            reply.clone(),
                            score,
                            trigger.clone(),
                        ));
                    }
                }
            }
        }
        let (plugin, contribution, reply, _, trigger) = best?;
        let trace = self.last_trace.get_or_insert_with(|| PluginTrace {
            input: input.to_string(),
            effective: input.to_string(),
            ..PluginTrace::default()
        });
        trace.matches.push(PluginMatch {
            plugin: plugin.clone(),
            contribution: contribution.clone(),
            kind: "policy".to_string(),
            mode: PluginMode::On,
            applied: true,
            detail: format!("触发词「{}」", trigger),
        });
        if !self.last_used.contains(&plugin) {
            self.last_used.push(plugin.clone());
            self.last_used.sort();
            self.last_used.dedup();
        }
        Some((reply, format!("plugin:{}:{}", plugin, contribution)))
    }

    /// 只给文本加固定前后缀；不改变事实句。非启用风格包不参与。
    pub fn style_wrap(&self, text: &str) -> String {
        self.style_with_matches(text).0
    }

    fn style_with_matches(&self, text: &str) -> (String, Vec<PluginMatch>) {
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let mut matches = Vec::new();
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            if mode != PluginMode::On {
                continue;
            }
            for contribution in &manifest.contributions {
                if let Contribution::Style {
                    id, prefix, suffix, ..
                } = contribution
                {
                    if prefix.is_empty() && suffix.is_empty() {
                        continue;
                    }
                    if !prefix.is_empty() {
                        prefixes.push(prefix.clone());
                    }
                    if !suffix.is_empty() {
                        suffixes.push(suffix.clone());
                    }
                    matches.push(PluginMatch {
                        plugin: manifest.id.clone(),
                        contribution: id.clone(),
                        kind: "style".to_string(),
                        mode,
                        applied: true,
                        detail: format!(
                            "{}{}",
                            if prefix.is_empty() { "" } else { "前缀" },
                            if suffix.is_empty() { "" } else { "后缀" }
                        ),
                    });
                }
            }
        }
        if prefixes.is_empty() && suffixes.is_empty() {
            return (text.to_string(), matches);
        }
        let mut out = String::new();
        for prefix in prefixes {
            out.push_str(&prefix);
        }
        out.push_str(text);
        for suffix in suffixes {
            out.push_str(&suffix);
        }
        (out, matches)
    }

    /// 处理一条插件命令；不匹配时返回 None（放行常规查询）。
    pub fn handle_command(&mut self, input: &str) -> Option<PluginCommandOutcome> {
        let command = parse_plugin_command(input)?;
        let outcome = match command {
            ParsedPluginCommand::Report => PluginCommandOutcome {
                text: self.render_report(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Diagnose => PluginCommandOutcome {
                text: self.render_diagnose(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Evaluate => {
                let results = self.registry.evaluate();
                let failures = results.iter().filter(|result| !result.passed).count();
                PluginCommandOutcome {
                    text: self.registry.render_eval(&results),
                    state_changed: false,
                    success: failures == 0,
                }
            }
            ParsedPluginCommand::Why => PluginCommandOutcome {
                text: self.render_trace(),
                state_changed: false,
                success: true,
            },
            ParsedPluginCommand::Enable(id) => match self.set_mode(&id, PluginMode::On) {
                Ok(auto) => PluginCommandOutcome {
                    text: {
                        let mut text = format!(
                            "已启用插件「{}」{}。",
                            self.manifest(&id)
                                .map(PluginManifest::display_name)
                                .unwrap_or(&id),
                            if auto.is_empty() {
                                String::new()
                            } else {
                                format!("（同时启用依赖：{}）", auto.join("、"))
                            }
                        );
                        if self.has_ontology(&id) {
                            text.push_str(
                                "\n注意：该包的本体声明在下次启动时合并；本轮先启用词表/改写/配方/模板。",
                            );
                        }
                        text.push_str("\n可用「插件」查看全部能力，「为什么这么答」查看本轮依据。");
                        text
                    },
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("启用失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
            ParsedPluginCommand::Disable(id) => match self.set_mode(&id, PluginMode::Off) {
                Ok(_) => PluginCommandOutcome {
                    text: format!(
                        "已停用插件「{}」。",
                        self.manifest(&id)
                            .map(PluginManifest::display_name)
                            .unwrap_or(&id)
                    ),
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("停用失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
            ParsedPluginCommand::SetMode(id, mode) => match self.set_mode(&id, mode) {
                Ok(auto) => PluginCommandOutcome {
                    text: format!(
                        "插件「{}」已切换为{}模式{}。",
                        id,
                        mode.label(),
                        if auto.is_empty() {
                            String::new()
                        } else {
                            format!("（同时启用依赖：{}）", auto.join("、"))
                        }
                    ),
                    state_changed: true,
                    success: true,
                },
                Err(message) => PluginCommandOutcome {
                    text: format!("模式切换失败：{}", message),
                    state_changed: false,
                    success: false,
                },
            },
        };
        Some(outcome)
    }

    pub fn render_report(&self) -> String {
        let mut lines = Vec::new();
        match self.registry.directory() {
            Some(directory) => lines.push(format!(
                "插件目录：{}（已安装 {} 个，启用 {} 个，观察 {} 个）",
                directory,
                self.manifests.len(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::On)
                    .count(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::Observe)
                    .count()
            )),
            None => lines.push(format!(
                "插件：已安装 {} 个，启用 {} 个",
                self.manifests.len(),
                self.modes
                    .values()
                    .filter(|mode| **mode == PluginMode::On)
                    .count()
            )),
        }
        if self.manifests.is_empty() {
            lines.push(
                "没有已安装的插件。可在插件目录放入 manifest.toml，或用「插件」查看加载诊断。"
                    .to_string(),
            );
        }
        for manifest in &self.manifests {
            let mode = self
                .modes
                .get(&manifest.id)
                .copied()
                .unwrap_or(PluginMode::Off);
            let counts = manifest
                .contribution_counts()
                .iter()
                .map(|(kind, count)| format!("{}×{}", kind, count))
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!(
                "- {} v{} [{}] {}：{}{}",
                manifest.display_name(),
                manifest.version,
                mode.label(),
                manifest.id,
                if manifest.description.trim().is_empty() {
                    ""
                } else {
                    manifest.description.trim()
                },
                if counts.is_empty() {
                    String::new()
                } else {
                    format!("\n    贡献：{}", counts)
                }
            ));
            if !manifest.requires.is_empty() {
                lines.push(format!("    依赖：{}", manifest.requires.join("、")));
            }
            if !manifest.conflicts.is_empty() {
                lines.push(format!("    冲突：{}", manifest.conflicts.join("、")));
            }
        }
        if !self.last_used.is_empty() {
            lines.push(format!("最近命中：{}", self.last_used.join("、")));
        }
        let issues = self.issues();
        if !issues.is_empty() {
            lines.push(format!("诊断：{} 条", issues.len()));
            for issue in issues.iter().take(8) {
                lines.push(format!("  {}", issue.render()));
            }
            if issues.len() > 8 {
                lines.push(format!("  …共 {} 条，用「插件诊断」查看。", issues.len()));
            }
        }
        lines.push(
            "命令：启用插件：<id> / 停用插件：<id> / 插件模式：<id>=observe|on|off / 插件诊断 / 插件评测 / 为什么这么答"
                .to_string(),
        );
        lines.join("\n")
    }

    pub fn render_diagnose(&self) -> String {
        let mut lines = vec![self.render_report()];
        lines.push("清单与启用诊断：".to_string());
        let issues = self.issues();
        if issues.is_empty() {
            lines.push("  未发现问题。".to_string());
        } else {
            for issue in &issues {
                lines.push(format!("  {}", issue.render()));
            }
        }
        lines.push("本轮插件轨迹：".to_string());
        lines.push(
            self.last_trace
                .as_ref()
                .map(render_trace_body)
                .unwrap_or_else(|| "  本轮尚未处理查询。".to_string()),
        );
        lines.join("\n")
    }

    pub fn render_trace(&self) -> String {
        match &self.last_trace {
            Some(trace) => format!("本轮插件轨迹：\n{}", render_trace_body(trace)),
            None => "本轮尚未处理查询，还没有插件轨迹。".to_string(),
        }
    }
}

type LexiconPair = (String, String, String, String, PluginMode);

fn apply_lexicon(input: &str, pairs: &[LexiconPair]) -> (String, Vec<PluginMatch>) {
    let mut out = String::with_capacity(input.len());
    let mut matches = Vec::new();
    let mut observed = BTreeSet::new();
    let mut pos = 0usize;
    while pos < input.len() {
        let rest = &input[pos..];
        let on_match = pairs
            .iter()
            .find(|(surface, _, _, _, mode)| *mode == PluginMode::On && rest.starts_with(surface));
        let observe_match = pairs.iter().find(|(surface, _, _, _, mode)| {
            *mode == PluginMode::Observe && rest.starts_with(surface)
        });
        if let Some((surface, canonical, plugin, contribution, mode)) = on_match {
            out.push_str(canonical);
            matches.push(PluginMatch {
                plugin: plugin.clone(),
                contribution: contribution.clone(),
                kind: "lexicon".to_string(),
                mode: *mode,
                applied: true,
                detail: format!("{} → {}", surface, canonical),
            });
            pos += surface.len();
        } else if let Some((surface, canonical, plugin, contribution, mode)) = observe_match {
            if observed.insert(surface.clone()) {
                matches.push(PluginMatch {
                    plugin: plugin.clone(),
                    contribution: contribution.clone(),
                    kind: "lexicon".to_string(),
                    mode: *mode,
                    applied: false,
                    detail: format!("{} → {}", surface, canonical),
                });
            }
            out.push_str(surface);
            pos += surface.len();
        } else {
            let ch = rest.chars().next().expect("non-empty rest");
            out.push(ch);
            pos += ch.len_utf8();
        }
    }
    (out, matches)
}

fn render_trace_body(trace: &PluginTrace) -> String {
    let mut lines = vec![
        format!("  输入：{}", trace.input),
        format!(
            "  生效：{}{}",
            trace.effective,
            if trace.changed() {
                "（已改写）"
            } else {
                "（未改写）"
            }
        ),
    ];
    if trace.matches.is_empty() {
        lines.push("  没有插件命中。".to_string());
    } else {
        for entry in &trace.matches {
            lines.push(format!(
                "  {} [{}] {} / {}（{}）{}",
                if entry.applied { "✓" } else { "○" },
                entry.mode.label(),
                entry.plugin,
                entry.contribution,
                entry.kind,
                entry.detail
            ));
        }
    }
    if let Some(recipe) = &trace.recipe {
        lines.push(format!(
            "  配方：{} / {} → {}",
            recipe.plugin, recipe.contribution, recipe.question
        ));
        if let Some(attribute) = &recipe.attribute {
            lines.push(format!("  属性：{}", attribute));
        }
    }
    for note in &trace.notes {
        lines.push(format!("  注：{}", note));
    }
    lines.join("\n")
}

fn dedup_issues(issues: Vec<PluginIssue>) -> Vec<PluginIssue> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for issue in issues {
        let key = format!(
            "{:?}|{:?}|{}|{}",
            issue.severity, issue.plugin, issue.field, issue.message
        );
        if seen.insert(key) {
            out.push(issue);
        }
    }
    out
}

#[derive(Clone, Debug)]
struct FrameCandidate {
    plugin: String,
    contribution: String,
    kind: &'static str,
    mode: PluginMode,
    specificity: usize,
    output: String,
    detail: String,
    recipe_invocation: Option<RecipeInvocation>,
    recipe: bool,
}

fn parse_plugin_command(input: &str) -> Option<ParsedPluginCommand> {
    let trimmed = input.trim().trim_end_matches(['。', '！', '!']).trim();
    let lower = trimmed.to_lowercase();
    if matches!(
        lower.as_str(),
        "插件" | "插件列表" | "插件报告" | "plugins" | "plugin list" | "plugin report"
    ) {
        return Some(ParsedPluginCommand::Report);
    }
    if matches!(
        lower.as_str(),
        "插件诊断" | "插件检查" | "plugin lint" | "plugins lint" | "plugin diagnose"
    ) {
        return Some(ParsedPluginCommand::Diagnose);
    }
    if matches!(
        lower.as_str(),
        "插件评测" | "插件自测" | "plugin eval" | "plugins eval"
    ) {
        return Some(ParsedPluginCommand::Evaluate);
    }
    if matches!(
        lower.as_str(),
        "为什么这么答" | "插件依据" | "插件轨迹" | "插件trace" | "plugin trace"
    ) {
        return Some(ParsedPluginCommand::Why);
    }
    for prefix in [
        "启用插件：",
        "启用插件:",
        "plugins enable ",
        "plugin enable ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let id = rest.trim().trim_matches(['「', '」', '"', '\'']);
            if !id.is_empty() {
                return Some(ParsedPluginCommand::Enable(id.to_string()));
            }
        }
    }
    for prefix in [
        "停用插件：",
        "停用插件:",
        "禁用插件：",
        "禁用插件:",
        "plugins disable ",
        "plugin disable ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let id = rest.trim().trim_matches(['「', '」', '"', '\'']);
            if !id.is_empty() {
                return Some(ParsedPluginCommand::Disable(id.to_string()));
            }
        }
    }
    for prefix in ["插件模式：", "插件模式:", "plugin mode ", "plugins mode "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim();
            let (id, mode) = rest
                .split_once('=')
                .or_else(|| rest.split_once('＝'))
                .or_else(|| rest.rsplit_once(char::is_whitespace))?;
            let id = id.trim().trim_matches(['「', '」', '"', '\'']);
            let mode = PluginMode::parse(mode.trim())?;
            if !id.is_empty() {
                return Some(ParsedPluginCommand::SetMode(id.to_string(), mode));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(id: &str, mode: PluginMode) -> PluginManifest {
        PluginManifest {
            id: id.to_string(),
            version: "0.1.0".to_string(),
            api: PLUGIN_API_VERSION,
            name: id.to_string(),
            description: String::new(),
            default_mode: mode,
            requires: Vec::new(),
            conflicts: Vec::new(),
            provides: Vec::new(),
            contributions: Vec::new(),
            eval_cases: Vec::new(),
        }
    }

    fn rewrite_contribution(id: &str, pattern: &str, replacement: &str) -> Contribution {
        Contribution::Rewrite {
            id: id.to_string(),
            description: String::new(),
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
        }
    }

    fn template(id: &str, text: &str) -> Contribution {
        Contribution::Template {
            id: id.to_string(),
            description: String::new(),
            template: text.to_string(),
        }
    }

    fn recipe(id: &str, pattern: &str, question: &str, template_id: &str) -> Contribution {
        Contribution::Recipe {
            id: id.to_string(),
            description: String::new(),
            patterns: vec![pattern.to_string()],
            question: question.to_string(),
            template: Some(template_id.to_string()),
            attribute: Some("导演".to_string()),
            direction: RecipeDirection::Forward,
            answer_types: Vec::new(),
        }
    }

    #[test]
    fn rewrite_applies_on_but_only_observes_parity() {
        let mut on = manifest("zh.film", PluginMode::On);
        on.contributions.push(rewrite_contribution(
            "film.directed",
            "{entity}是谁导演的",
            "{entity}的导演是谁",
        ));
        let mut observe = manifest("zh.observe", PluginMode::Observe);
        observe.contributions.push(rewrite_contribution(
            "observe.rewrite",
            "{entity}由谁导演",
            "{entity}的导演是谁",
        ));
        let registry = PluginRegistry::from_manifests(vec![on, observe]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let outcome = runtime.rewrite("阿凡达是谁导演的");
        assert_eq!(outcome.text, "阿凡达的导演是谁");
        assert!(outcome.trace.matches.iter().any(|entry| {
            entry.plugin == "zh.film" && entry.mode == PluginMode::On && entry.applied
        }));

        let observed = runtime.rewrite("阿凡达由谁导演");
        assert_eq!(observed.text, "阿凡达由谁导演");
        assert!(observed.trace.matches.iter().any(|entry| {
            entry.plugin == "zh.observe" && entry.mode == PluginMode::Observe && !entry.applied
        }));
    }

    #[test]
    fn lexicon_and_ontology_aliases_are_normalized() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(Contribution::Lexicon {
            id: "film.lexicon".to_string(),
            description: String::new(),
            entries: vec![
                LexiconEntry {
                    surface: "影片".to_string(),
                    canonical: "电影".to_string(),
                    pos: None,
                },
                LexiconEntry {
                    surface: "影".to_string(),
                    canonical: "电影".to_string(),
                    pos: None,
                },
            ],
        });
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec!["执导".to_string()],
                ..OntologyPredicate::default()
            }],
            types: Vec::new(),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let outcome = runtime.rewrite("这部影片由谁执导");
        assert_eq!(outcome.text, "这部电影由谁导演");
        assert!(
            outcome
                .trace
                .matches
                .iter()
                .filter(|entry| entry.kind == "lexicon")
                .count()
                >= 2
        );
    }

    #[test]
    fn settings_referencing_unknown_plugin_warns_instead_of_failing() {
        let registry = PluginRegistry::from_manifests(vec![manifest("zh.film", PluginMode::Off)]);
        let settings = PluginSettings {
            enabled: vec!["zh.drama".to_string()],
            modes: BTreeMap::from([("zh.music".to_string(), PluginMode::On)]),
            ..PluginSettings::default()
        };
        let runtime = PluginRuntime::with_settings(registry, &settings);
        assert_eq!(runtime.enabled_count(), 0);
        assert_eq!(
            runtime
                .issues()
                .iter()
                .filter(|issue| issue.severity == PluginSeverity::Warning)
                .count(),
            2
        );
    }

    #[test]
    fn style_wrap_applies_only_when_plugin_is_enabled() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(Contribution::Style {
            id: "film.style".to_string(),
            description: String::new(),
            prefix: "【电影】".to_string(),
            suffix: "（完）".to_string(),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert_eq!(runtime.style_wrap("答"), "答");
        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        assert_eq!(runtime.style_wrap("答"), "【电影】答（完）");
    }

    #[test]
    fn reference_hints_are_exported_only_when_enabled() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(Contribution::Reference {
            id: "film.reference".to_string(),
            description: String::new(),
            quantifiers: vec!["这位".to_string()],
            role_words: vec!["导演".to_string()],
            surnames: vec!["卡梅隆".to_string()],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert!(runtime.reference_lexicon().is_empty());
        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        assert_eq!(
            runtime.reference_lexicon(),
            vec![
                ("卡梅隆".to_string(), "nr".to_string()),
                ("导演".to_string(), "n".to_string()),
                ("这位".to_string(), "r".to_string()),
            ]
        );
    }

    #[test]
    fn policy_fallback_is_scoped_to_enabled_plugins_and_traced() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(Contribution::Policy {
            id: "film.fallback".to_string(),
            description: String::new(),
            when: vec!["不懂".to_string()],
            reply: "我可以帮你查电影。".to_string(),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert!(runtime.policy_reply("我不懂").is_none());
        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        let (reply, plan) = runtime.policy_reply("我不懂").expect("启用后应命中策略");
        assert_eq!(reply, "我可以帮你查电影。");
        assert_eq!(plan, "plugin:zh.film:film.fallback");
        let trace = runtime.last_trace().expect("策略命中应留痕");
        assert_eq!(trace.matches.len(), 1);
        assert_eq!(trace.matches[0].kind, "policy");
        assert!(trace.matches[0].detail.contains("不懂"));
    }

    #[test]
    fn ontology_aliases_are_exported_only_for_enabled_plugins() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec!["director".to_string()],
                ..OntologyPredicate::default()
            }],
            types: Vec::new(),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert!(runtime.ontology_aliases().is_empty());
        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        assert_eq!(
            runtime.ontology_aliases(),
            vec![("director".to_string(), "导演".to_string())]
        );
    }

    #[test]
    fn ontology_program_exports_structural_axioms_only_when_enabled() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec!["执导".to_string()],
                inverse: Some("执导".to_string()),
                transitive: true,
                symmetric: false,
                domain: vec!["电影".to_string()],
                range: vec!["人物".to_string()],
            }],
            types: vec![
                OntologyType {
                    name: "电影".to_string(),
                    aliases: Vec::new(),
                    subclass_of: vec!["影视作品".to_string()],
                    disjoint_with: Vec::new(),
                },
                OntologyType {
                    name: "动画电影".to_string(),
                    aliases: Vec::new(),
                    subclass_of: vec!["电影".to_string()],
                    disjoint_with: vec!["真人作品".to_string()],
                },
            ],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert!(runtime.ontology_program().is_empty());
        assert!(runtime.ontology_inverse_surfaces().is_empty());

        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        let program = runtime.ontology_program();
        assert_eq!(program.len(), 7, "子类2 + 互斥1 + 定义域1 + 值域1 + 逆1 + 传递1");
        assert!(program.lessons.iter().any(|lesson| matches!(
            lesson,
            OntologyLesson::Subclass { sub, sup, .. }
                if sub == "动画电影" && sup == "电影"
        )));
        assert!(program.lessons.iter().any(|lesson| matches!(
            lesson,
            OntologyLesson::Disjoint { left, right, .. }
                if left == "动画电影" && right == "真人作品"
        )));
        assert!(program.lessons.iter().any(|lesson| matches!(
            lesson,
            OntologyLesson::Domain { property, class, .. }
                if property == "导演" && class == "电影"
        )));
        assert!(program.lessons.iter().any(|lesson| matches!(
            lesson,
            OntologyLesson::Range { property, class, .. }
                if property == "导演" && class == "人物"
        )));
        assert!(program.lessons.iter().any(|lesson| matches!(
            lesson,
            OntologyLesson::Inverse { left, right, .. }
                if left == "导演" && right == "执导"
        )));
        assert_eq!(
            runtime.ontology_inverse_surfaces(),
            vec![("执导".to_string(), "导演".to_string())]
        );
        assert_eq!(
            runtime.ontology_program(),
            program,
            "重复导出必须幂等（规范排序去重）"
        );
    }

    #[test]
    fn ontology_lint_reports_degenerate_declarations_without_dropping_package() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec![String::new(), "导演".to_string()],
                inverse: Some("导演".to_string()),
                domain: vec![String::new()],
                ..OntologyPredicate::default()
            }],
            types: vec![OntologyType {
                name: "电影".to_string(),
                aliases: vec![String::new()],
                subclass_of: vec!["电影".to_string()],
                disjoint_with: vec![String::new()],
            }],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        assert!(!registry.has_errors(), "退化声明只记警告，不废整包");
        assert!(registry
            .issues()
            .iter()
            .filter(|issue| issue.severity == PluginSeverity::Warning)
            .count()
            >= 6);
        let runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let program = runtime.ontology_program();
        assert!(
            program.lessons.is_empty(),
            "空/自指声明不得进入统一本体：{:?}",
            program.lessons
        );
    }

    #[test]
    fn reference_film_plugin_lints_and_passes_its_own_evals() {
        let manifest: PluginManifest =
            toml::from_str(include_str!("../plugins/zh.film.toml")).expect("参考插件应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let results = registry.evaluate();
        assert!(!results.is_empty(), "参考插件应自带评测");
        assert!(
            results.iter().all(|result| result.passed),
            "{}",
            registry.render_eval(&results)
        );
        let runtime = PluginRuntime::with_settings(
            registry,
            &PluginSettings {
                enabled: vec!["zh.film".to_string()],
                ..PluginSettings::default()
            },
        );
        assert!(
            runtime.ontology_program().len() >= 7,
            "影视插件应贡献类型层级/定义域/值域/逆关系等公理"
        );
        assert_eq!(
            runtime.ontology_inverse_surfaces(),
            vec![("执导".to_string(), "导演".to_string())]
        );
    }

    #[test]
    fn typed_recipe_direction_and_evidence_are_parsed_and_rendered() {
        let manifest: PluginManifest = toml::from_str(
            r#"
id = "zh.film"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}导演过哪些电影"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过：{answer}。依据：{evidence}"
"#,
        )
        .expect("类型化配方应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let outcome = runtime.rewrite("卡梅隆导演过哪些电影");
        let recipe = outcome.trace.recipe.as_ref().expect("应命中配方");
        assert_eq!(recipe.direction, RecipeDirection::Inverse);
        assert_eq!(recipe.answer_types, vec!["电影".to_string()]);
        let composed = runtime.compose_answer_with_evidence(
            &outcome,
            "阿凡达",
            &["阿凡达 —导演→ 卡梅隆（直接声明）".to_string()],
        );
        assert_eq!(
            composed.text,
            "「卡梅隆」导演过：阿凡达。依据：阿凡达 —导演→ 卡梅隆（直接声明）"
        );
        assert!(
            runtime
                .last_trace()
                .expect("应留痕")
                .matches
                .iter()
                .any(|entry| entry.kind == "template" && entry.applied)
        );
    }

    #[test]
    fn recipe_composes_grounded_template_answer() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(recipe(
            "film.directed",
            "{entity}是谁导演的",
            "{entity}的导演是什么",
            "film.directed.template",
        ));
        film.contributions.push(template(
            "film.directed.template",
            "「{entity}」的导演是{answer}。",
        ));
        let registry = PluginRegistry::from_manifests(vec![film]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let outcome = runtime.rewrite("阿凡达是谁导演的");
        assert_eq!(
            outcome.trace.recipe.as_ref().unwrap().question,
            "阿凡达的导演是什么"
        );
        let composed = runtime.compose_answer(&outcome, "詹姆斯·卡梅隆");
        assert_eq!(composed.text, "「阿凡达」的导演是詹姆斯·卡梅隆。");
        assert_eq!(composed.template.as_deref(), Some("film.directed.template"));
    }

    #[test]
    fn template_without_slot_grounding_falls_back() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(Contribution::Style {
            id: "film.style".to_string(),
            description: String::new(),
            prefix: "（电影包）".to_string(),
            suffix: String::new(),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let outcome = runtime.rewrite("普通问题");
        let composed = runtime.compose_answer(&outcome, "答案");
        assert_eq!(composed.text, "（电影包）答案");
    }

    #[test]
    fn enabling_auto_enables_requires_and_disable_refuses_dependents() {
        let base = manifest("zh.base", PluginMode::Off);
        let mut child = manifest("zh.child", PluginMode::Off);
        child.requires = vec!["zh.base".to_string()];
        child.contributions.push(rewrite_contribution(
            "child.rewrite",
            "{entity}是谁",
            "{entity}是什么",
        ));
        let registry = PluginRegistry::from_manifests(vec![base, child]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let auto = runtime.set_mode("zh.child", PluginMode::On).unwrap();
        assert_eq!(auto, vec!["zh.base".to_string()]);
        assert_eq!(runtime.mode("zh.base"), Some(PluginMode::On));
        let error = runtime.set_mode("zh.base", PluginMode::Off).unwrap_err();
        assert!(error.contains("zh.child"));
        runtime.set_mode("zh.child", PluginMode::Off).unwrap();
        runtime.set_mode("zh.base", PluginMode::Off).unwrap();
    }

    #[test]
    fn conflicting_plugins_cannot_both_be_enabled() {
        let mut left = manifest("zh.left", PluginMode::Off);
        left.conflicts = vec!["zh.right".to_string()];
        let right = manifest("zh.right", PluginMode::Off);
        let registry = PluginRegistry::from_manifests(vec![left, right]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        runtime.set_mode("zh.left", PluginMode::On).unwrap();
        let error = runtime.set_mode("zh.right", PluginMode::On).unwrap_err();
        assert!(error.contains("冲突"));
        assert_eq!(runtime.mode("zh.right"), Some(PluginMode::Off));
    }

    #[test]
    fn session_state_round_trips_and_ignores_unknown_plugins() {
        let film = manifest("zh.film", PluginMode::Off);
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        runtime.set_mode("zh.film", PluginMode::Observe).unwrap();
        let state = runtime.session_state();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("plugin_state.json");
        state.save(&path).unwrap();
        let restored = PluginSessionState::load(&path).unwrap();
        assert_eq!(restored.modes.get("zh.film"), Some(&PluginMode::Observe));
        let registry = PluginRegistry::from_manifests(vec![manifest("zh.film", PluginMode::Off)]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let applied = runtime.restore_session_state(restored);
        assert_eq!(applied, 1);
        assert_eq!(runtime.mode("zh.film"), Some(PluginMode::Observe));
    }

    #[test]
    fn commands_are_parsed_and_report_modes() {
        let film = manifest("zh.film", PluginMode::Off);
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let outcome = runtime.handle_command("启用插件：zh.film").unwrap();
        assert!(outcome.success && outcome.state_changed);
        assert_eq!(runtime.mode("zh.film"), Some(PluginMode::On));
        let report = runtime.handle_command("插件").unwrap();
        assert!(report.text.contains("zh.film"));
        let mode = runtime.handle_command("插件模式：zh.film=observe").unwrap();
        assert!(mode.success && mode.state_changed);
        assert_eq!(runtime.mode("zh.film"), Some(PluginMode::Observe));
        assert!(runtime.handle_command("普通问题").is_none());
    }

    #[test]
    fn invalid_manifest_is_rejected_with_reason() {
        let mut broken = manifest("zh.broken", PluginMode::On);
        broken.contributions.push(rewrite_contribution(
            "broken.rewrite",
            "{entity}是谁",
            "{missing}是什么",
        ));
        let registry = PluginRegistry::from_manifests(vec![broken]);
        assert!(registry.has_errors());
        assert!(registry.manifests().is_empty());
        assert!(registry
            .issues()
            .iter()
            .any(|issue| issue.message.contains("missing")));
    }

    #[test]
    fn eval_cases_run_against_enabled_plugins() {
        let mut film = manifest("zh.film", PluginMode::Off);
        film.contributions.push(rewrite_contribution(
            "film.directed",
            "{entity}是谁导演的",
            "{entity}的导演是谁",
        ));
        film.eval_cases.push(PluginEvalCase {
            input: "阿凡达是谁导演的".to_string(),
            expect_rewrite: Some("阿凡达的导演是谁".to_string()),
            expect_plugin: Some("zh.film".to_string()),
            expect_contains: Some("导演".to_string()),
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let results = registry.evaluate();
        assert_eq!(results.len(), 1);
        assert!(results[0].passed, "{}", results[0].detail);
    }
}

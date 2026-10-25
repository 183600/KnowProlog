//! 语言运行时（Language Runtime）：把 `config.toml` 当作可修订的语言策略。
//!
//! 配置不是启动参数，而是一份会持续演化的「语言信念」。本模块把每次加载
//! 建模成一个不可变快照：
//!
//! - **事务提交**：读取 → 解析 → 配置契约 lint → 能力差异 → 原子替换；
//!   lint 有 Error 时整次修订被拒绝，旧快照（epoch）原样保留；
//! - **语义指纹**：按 TOML 结构递归排序后哈希，注释、空白、键序变化不算修订；
//! - **分层接管**：`nlp` / `messages` / 同义词 / 模式 / 抽取 / 双向语义这类
//!   语言资源可在会话中热替换；存储、推理结构、插件清单、类型体系等
//!   需要重启的节会从上一版语言配置继承，绝不半途混用两代结构；
//! - **会话不变**：本模块只持有配置快照，不拥有焦点、已学模式、承诺或
//!   回答计划。热替换的是「怎么说」，不是「记得什么」；
//! - **回滚是新纪元**：回滚不是时光倒流，而是把旧内容重新提交为一个新
//!   epoch，并留下可审计的回执。
//!
//! 约定：
//! - 快照与差异报告确定性排序，可逐字节断言；
//! - 错误/拒绝不改变 epoch，也不写入历史；
//! - 历史有界，够用即可，不把整份配置无限堆在内存里。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::sync::Arc;

use crate::config_contract::{self, ConfigIssue};
use crate::context_image::fnv1a_hex;
use crate::SystemConfig;

/// 语言运行时版本：快照/差异语义变化时递增。
pub const LANG_RUNTIME_VERSION: u32 = 1;

/// 默认保留的已提交快照数。
pub const DEFAULT_HISTORY_LIMIT: usize = 8;

/// 可在会话中热替换的顶层配置节（语言资源）。
///
/// 不在本表中的节默认按「需要重启」处理：未知节宁可保守，也不能把
/// 只重建了一半的结构当成已接管。
pub const HOT_SECTIONS: &[&str] = &[
    "nlp",
    "messages",
    "synonyms",
    "entity_types",
    "extraction_rules",
    "prolog_rules",
    "semantic_grammar",
];

/// 顶层节的稳定中文标签（回执与 CLI 共用）。
pub fn section_label_zh(key: &str) -> &'static str {
    match key {
        "system" => "系统与存储",
        "api" => "外部接口",
        "nlp" => "分词与 NLU",
        "reasoning" => "推理与生成结构",
        "entity_types" => "实体类型",
        "extraction_rules" => "知识抽取规则",
        "prolog_rules" => "查询规则",
        "messages" => "界面措辞",
        "synonyms" => "同义词",
        "predicate_profiles" => "谓词画像",
        "entity_profiles" => "实体身份",
        "type_system" => "类型体系",
        "plugins" => "插件清单",
        "trace_graph" => "分层追踪",
        "dialogue_frames" => "对话帧",
        "semantic_grammar" => "双向语义规则",
        _ => "其它配置",
    }
}

/// 该顶层节是否允许热替换。
pub fn is_hot_section(key: &str) -> bool {
    HOT_SECTIONS.contains(&key)
}

/// 某次加载的可热替换/需重启差异。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangDiff {
    /// 全部发生语义变化的顶层节（字典序）。
    pub changed: Vec<String>,
    /// 其中可在本会话接管的节。
    pub hot: Vec<String>,
    /// 其中必须重启才能接管的节。
    pub restart_required: Vec<String>,
}

impl LangDiff {
    fn between(previous: &BTreeMap<String, String>, next: &BTreeMap<String, String>) -> Self {
        let keys: BTreeSet<&str> = previous
            .keys()
            .chain(next.keys())
            .map(String::as_str)
            .collect();
        let changed: Vec<String> = keys
            .into_iter()
            .filter(|key| previous.get(*key) != next.get(*key))
            .map(str::to_string)
            .collect();
        let hot: Vec<String> = changed
            .iter()
            .filter(|key| is_hot_section(key))
            .cloned()
            .collect();
        let restart_required: Vec<String> = changed
            .iter()
            .filter(|key| !is_hot_section(key))
            .cloned()
            .collect();
        Self {
            changed,
            hot,
            restart_required,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }

    /// 人类可读的「正在热替换的节」。
    pub fn render_hot(&self) -> String {
        render_sections(&self.hot)
    }

    /// 人类可读的「需要重启的节」。
    pub fn render_restart_required(&self) -> String {
        render_sections(&self.restart_required)
    }
}

/// 把顶层节键渲染成中文列表；空列表给出「无」。
pub fn render_sections(sections: &[String]) -> String {
    if sections.is_empty() {
        return "无".to_string();
    }
    sections
        .iter()
        .map(|key| format!("{}（{}）", section_label_zh(key), key))
        .collect::<Vec<_>>()
        .join("、")
}

/// 一份已提交的语言策略快照。
///
/// `config` 是文件里的完整新配置（供审计与下一次差异）；
/// `language_config` 是本会话真正生效的语言视图：热节来自本快照，
/// 需重启的节沿用上一层，保证任何组件都不会看到「半新半旧」的结构。
#[derive(Debug, Clone)]
pub struct LangSnapshot {
    pub epoch: u64,
    pub fingerprint: String,
    pub path: String,
    config: Arc<SystemConfig>,
    language_config: Arc<SystemConfig>,
    sections: BTreeMap<String, String>,
}

impl LangSnapshot {
    /// 完整配置（含需重启的节；只读）。
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    /// 本会话有效的语言配置视图（需重启的节已继承上一层）。
    pub fn language_config(&self) -> &Arc<SystemConfig> {
        &self.language_config
    }

    /// 顶层节的规范摘要（只读；用于差异审计）。
    pub fn sections(&self) -> &BTreeMap<String, String> {
        &self.sections
    }

    /// 与另一快照的语义差异。
    pub fn diff_from(&self, previous: &LangSnapshot) -> LangDiff {
        LangDiff::between(&previous.sections, &self.sections)
    }
}

/// 一次成功提交的回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangApply {
    pub previous_epoch: u64,
    pub epoch: u64,
    pub diff: LangDiff,
    pub warnings: usize,
}

/// 一次回滚的回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangRollback {
    /// 回滚后产生的新 epoch（不是被恢复快照的旧 epoch）。
    pub epoch: u64,
    /// 被恢复内容原先所在的 epoch。
    pub restored_epoch: u64,
    /// 相对回滚前发生的变化。
    pub diff: LangDiff,
}

/// 加载/提交结果：只有 `Applied` 会推进 epoch。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LangReloadOutcome {
    Applied(LangApply),
    Unchanged { epoch: u64 },
    Rejected {
        reason: String,
        issues: Vec<ConfigIssue>,
    },
}

impl LangReloadOutcome {
    pub fn is_applied(&self) -> bool {
        matches!(self, LangReloadOutcome::Applied(_))
    }
}

/// 语言运行时错误。拒绝（契约不通过）不是错误，而是 [`LangReloadOutcome::Rejected`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LangRuntimeError {
    Read { path: String, message: String },
    Parse(String),
    NoHistory,
}

impl std::fmt::Display for LangRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LangRuntimeError::Read { path, message } => {
                write!(f, "读取语言配置「{}」失败：{}", path, message)
            }
            LangRuntimeError::Parse(message) => write!(f, "解析语言配置失败：{}", message),
            LangRuntimeError::NoHistory => write!(f, "没有可回滚的语言配置快照"),
        }
    }
}

impl std::error::Error for LangRuntimeError {}

/// 语言策略运行时：当前快照 + 有界历史。
#[derive(Debug)]
pub struct LangRuntime {
    path: String,
    current: LangSnapshot,
    history: Vec<LangSnapshot>,
    history_limit: usize,
}

impl LangRuntime {
    /// 以已经解析好的启动配置建立运行时（epoch = 1）。
    pub fn new(path: impl Into<String>, config: SystemConfig) -> Self {
        Self::with_history_limit(path, config, DEFAULT_HISTORY_LIMIT)
    }

    /// 指定历史容量。至少保留 1 份可回滚快照。
    pub fn with_history_limit(
        path: impl Into<String>,
        config: SystemConfig,
        history_limit: usize,
    ) -> Self {
        let path = path.into();
        let (fingerprint, sections) = match fs::read_to_string(&path) {
            Ok(source) => match canonical_source(&source) {
                Ok((canonical, sections)) => (fnv1a_hex(canonical.as_bytes()), sections),
                Err(_) => (unavailable_fingerprint(&path), BTreeMap::new()),
            },
            Err(_) => (unavailable_fingerprint(&path), BTreeMap::new()),
        };
        let config = Arc::new(config);
        Self {
            path: path.clone(),
            current: LangSnapshot {
                epoch: 1,
                fingerprint,
                path,
                language_config: Arc::clone(&config),
                config,
                sections,
            },
            history: Vec::new(),
            history_limit: history_limit.max(1),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn epoch(&self) -> u64 {
        self.current.epoch
    }

    pub fn current(&self) -> &LangSnapshot {
        &self.current
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// 从配置文件重新加载；读取/解析错误返回 `Err`，契约拒绝返回 `Rejected`。
    pub fn reload(&mut self) -> Result<LangReloadOutcome, LangRuntimeError> {
        let source = fs::read_to_string(&self.path).map_err(|error| LangRuntimeError::Read {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        self.apply_source(&source)
    }

    /// 从已读取的文本加载（供显式命令、测试与将来的 watcher 共用）。
    pub fn apply_source(&mut self, source: &str) -> Result<LangReloadOutcome, LangRuntimeError> {
        let (canonical, sections) =
            canonical_source(source).map_err(|error| LangRuntimeError::Parse(error.to_string()))?;
        let config: SystemConfig = toml::from_str(source).map_err(|error| {
            LangRuntimeError::Parse(format!("不是合法的配置结构：{}", error))
        })?;
        let issues = config_contract::lint(&config);
        if config_contract::has_errors(&issues) {
            return Ok(LangReloadOutcome::Rejected {
                reason: "配置契约检查未通过，本次修订未提交".to_string(),
                issues,
            });
        }
        let fingerprint = fnv1a_hex(canonical.as_bytes());
        if fingerprint == self.current.fingerprint {
            return Ok(LangReloadOutcome::Unchanged {
                epoch: self.current.epoch,
            });
        }
        let previous_epoch = self.current.epoch;
        let diff = LangDiff::between(&self.current.sections, &sections);
        let language_config = Arc::new(self.merge_language_config(&config));
        let warnings = config_contract::warning_count(&issues);
        let snapshot = LangSnapshot {
            epoch: previous_epoch + 1,
            fingerprint,
            path: self.path.clone(),
            config: Arc::new(config),
            language_config,
            sections,
        };
        let previous = std::mem::replace(&mut self.current, snapshot);
        self.history.push(previous);
        if self.history.len() > self.history_limit {
            self.history.remove(0);
        }
        Ok(LangReloadOutcome::Applied(LangApply {
            previous_epoch,
            epoch: self.current.epoch,
            diff,
            warnings,
        }))
    }

    /// 回滚到上一份已提交快照，并把旧内容提交为新 epoch。
    pub fn rollback(&mut self) -> Result<LangRollback, LangRuntimeError> {
        let Some(mut previous) = self.history.pop() else {
            return Err(LangRuntimeError::NoHistory);
        };
        let diff = LangDiff::between(&self.current.sections, &previous.sections);
        let restored_epoch = previous.epoch;
        let epoch = self.current.epoch + 1;
        previous.epoch = epoch;
        previous.path = self.path.clone();
        self.current = previous;
        Ok(LangRollback {
            epoch,
            restored_epoch,
            diff,
        })
    }

    /// 构造本会话语言视图：热节取新配置，需重启的结构节沿用上一层。
    fn merge_language_config(&self, next: &SystemConfig) -> SystemConfig {
        let mut merged = next.clone();
        let base = self.current.language_config.as_ref();
        merged.system = base.system.clone();
        merged.api = base.api.clone();
        merged.reasoning = base.reasoning.clone();
        merged.predicate_profiles = base.predicate_profiles.clone();
        merged.entity_profiles = base.entity_profiles.clone();
        merged.type_system = base.type_system.clone();
        merged.plugins = base.plugins.clone();
        merged.trace_graph = base.trace_graph.clone();
        merged.dialogue_frames = base.dialogue_frames.clone();
        merged
    }
}

fn unavailable_fingerprint(path: &str) -> String {
    fnv1a_hex(format!("unavailable\u{0}{}", path).as_bytes())
}

/// 把 TOML 解析成规范文本与逐顶层节的规范摘要。
///
/// 递归排序表键、保留数组顺序：注释/空白/键序不影响指纹，
/// 数组顺序（例如规则优先级列表）仍然算语义差异。
fn canonical_source(source: &str) -> Result<(String, BTreeMap<String, String>), toml::de::Error> {
    let value: toml::Value = toml::from_str(source)?;
    let canonical = canonical_toml(&value);
    let mut sections = BTreeMap::new();
    if let toml::Value::Table(table) = &value {
        for (key, value) in table {
            sections.insert(key.clone(), canonical_toml(value));
        }
    }
    Ok((canonical, sections))
}

fn canonical_toml(value: &toml::Value) -> String {
    let mut out = String::new();
    canonical_toml_into(value, &mut out);
    out
}

fn canonical_toml_into(value: &toml::Value, out: &mut String) {
    match value {
        toml::Value::String(text) => {
            let _ = write!(out, "s{}:{:?}", text.len(), text);
        }
        toml::Value::Integer(number) => {
            let _ = write!(out, "i{};", number);
        }
        toml::Value::Float(number) => {
            let _ = write!(out, "f{:016x};", number.to_bits());
        }
        toml::Value::Boolean(flag) => {
            let _ = write!(out, "b{};", flag);
        }
        toml::Value::Datetime(datetime) => {
            let text = datetime.to_string();
            let _ = write!(out, "d{}:{};", text.len(), text);
        }
        toml::Value::Array(items) => {
            let _ = write!(out, "a{}[", items.len());
            for item in items {
                canonical_toml_into(item, out);
            }
            out.push_str("];");
        }
        toml::Value::Table(table) => {
            let mut keys: Vec<&String> = table.keys().collect();
            keys.sort();
            let _ = write!(out, "t{}[", keys.len());
            for key in keys {
                let _ = write!(out, "k{}:{}=", key.len(), key);
                if let Some(value) = table.get(key) {
                    canonical_toml_into(value, out);
                }
                out.push(';');
            }
            out.push_str("];");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_source() -> String {
        include_str!("../config.toml").to_string()
    }

    fn config(source: &str) -> SystemConfig {
        toml::from_str(source).expect("测试配置应可解析")
    }

    fn mutate<T>(source: &str, edit: impl FnOnce(&mut toml::Value) -> T) -> (String, T) {
        let mut value: toml::Value = toml::from_str(source).expect("测试配置应可解析");
        let outcome = edit(&mut value);
        (toml::to_string(&value).expect("测试配置应可序列化"), outcome)
    }

    fn runtime_with(source: &str) -> (tempfile::TempDir, LangRuntime) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, source).unwrap();
        let runtime = LangRuntime::new(path.to_string_lossy().to_string(), config(source));
        (dir, runtime)
    }

    #[test]
    fn initial_snapshot_is_epoch_one_and_covers_file_sections() {
        let (_dir, runtime) = runtime_with(&base_source());
        assert_eq!(runtime.epoch(), 1);
        assert!(runtime.current().sections().contains_key("prolog_rules"));
        assert!(runtime.current().sections().contains_key("messages"));
        assert_eq!(runtime.history_len(), 0);
    }

    #[test]
    fn comments_and_formatting_do_not_advance_the_epoch() {
        let (_dir, mut runtime) = runtime_with(&base_source());
        let mut source = String::from("# 只改注释，不改语言\n\n");
        source.push_str(&base_source());
        let outcome = runtime.apply_source(&source).unwrap();
        assert_eq!(
            outcome,
            LangReloadOutcome::Unchanged { epoch: 1 },
            "注释不是语义修订"
        );
        assert_eq!(runtime.epoch(), 1);
    }

    #[test]
    fn hot_language_changes_apply_while_structural_sections_wait_for_restart() {
        let source = base_source();
        let (next, ()) = mutate(&source, |value| {
            value["messages"]["ready"] = toml::Value::String("语言已热更新。".to_string());
            value["reasoning"]["max_profile_sentences"] = toml::Value::Integer(2);
        });
        let (_dir, mut runtime) = runtime_with(&source);
        let before_reasoning = runtime
            .current
            .language_config
            .reasoning
            .max_profile_sentences;
        let outcome = runtime.apply_source(&next).unwrap();
        let LangReloadOutcome::Applied(apply) = outcome else {
            panic!("热语言修订应被提交：{outcome:?}");
        };
        assert_eq!(apply.previous_epoch, 1);
        assert_eq!(apply.epoch, 2);
        assert_eq!(runtime.epoch(), 2);
        assert_eq!(runtime.history_len(), 1);
        assert!(apply.diff.hot.contains(&"messages".to_string()));
        assert!(apply
            .diff
            .restart_required
            .contains(&"reasoning".to_string()));
        assert_eq!(
            runtime.current().language_config().messages.ready,
            "语言已热更新。"
        );
        assert_eq!(
            runtime
                .current()
                .language_config()
                .reasoning
                .max_profile_sentences,
            before_reasoning,
            "推理结构不能在一轮内半途换掉"
        );
        assert_eq!(
            runtime.current().config().reasoning.max_profile_sentences,
            2,
            "完整快照仍记录文件里的新值，供审计与下次启动"
        );
    }

    #[test]
    fn contract_errors_reject_the_whole_revision_without_touching_history() {
        let source = base_source();
        let (invalid, ()) = mutate(&source, |value| {
            let rules = value
                .get_mut("prolog_rules")
                .and_then(toml::Value::as_array_mut)
                .expect("测试配置应含 prolog_rules");
            let first = rules.first_mut().expect("测试配置应至少有一条查询规则");
            first["query_type"] = toml::Value::String("definitely_unknown_query".to_string());
        });
        let (_dir, mut runtime) = runtime_with(&source);
        let outcome = runtime.apply_source(&invalid).unwrap();
        match outcome {
            LangReloadOutcome::Rejected { reason, issues } => {
                assert!(reason.contains("配置契约"));
                assert!(config_contract::has_errors(&issues));
            }
            other => panic!("含悬空 query_type 的配置必须被拒绝：{other:?}"),
        }
        assert_eq!(runtime.epoch(), 1);
        assert_eq!(runtime.history_len(), 0);
        let (canonical, _) = canonical_source(&source).unwrap();
        assert_eq!(runtime.current().fingerprint, fnv1a_hex(canonical.as_bytes()));
    }

    #[test]
    fn syntax_errors_are_errors_not_silent_acceptance() {
        let (_dir, mut runtime) = runtime_with(&base_source());
        let error = runtime.apply_source("[messages\n").unwrap_err();
        assert!(matches!(error, LangRuntimeError::Parse(_)));
        assert_eq!(runtime.epoch(), 1);
    }

    #[test]
    fn rollback_commits_old_content_as_a_new_epoch() {
        let source = base_source();
        let (second, ()) = mutate(&source, |value| {
            value["messages"]["ready"] = toml::Value::String("第二版。".to_string());
        });
        let (third, ()) = mutate(&second, |value| {
            value["messages"]["ready"] = toml::Value::String("第三版。".to_string());
        });
        let (_dir, mut runtime) = runtime_with(&source);
        assert!(runtime.apply_source(&second).unwrap().is_applied());
        assert!(runtime.apply_source(&third).unwrap().is_applied());
        assert_eq!(runtime.epoch(), 3);
        let rollback = runtime.rollback().unwrap();
        assert_eq!(rollback.epoch, 4);
        assert_eq!(rollback.restored_epoch, 2);
        assert!(rollback.diff.changed.contains(&"messages".to_string()));
        assert_eq!(
            runtime.current().language_config().messages.ready,
            "第二版。"
        );
        assert_eq!(runtime.history_len(), 1);
        assert_eq!(runtime.epoch(), 4);
        assert!(runtime.rollback().is_ok());
        assert_eq!(runtime.epoch(), 5);
        assert_eq!(
            runtime.current().language_config().messages.ready,
            config(&source).messages.ready
        );
        assert!(matches!(
            runtime.rollback(),
            Err(LangRuntimeError::NoHistory)
        ));
    }

    #[test]
    fn history_is_bounded_and_old_snapshots_are_dropped_from_the_front() {
        let source = base_source();
        let (_dir, mut runtime) = runtime_with(&source);
        let mut current = source;
        for index in 0..12 {
            let (next, ()) = mutate(&current, |value| {
                value["messages"]["ready"] = toml::Value::String(format!("第 {} 版。", index));
            });
            assert!(runtime.apply_source(&next).unwrap().is_applied());
            current = next;
        }
        assert_eq!(runtime.epoch(), 13);
        assert_eq!(runtime.history_len(), DEFAULT_HISTORY_LIMIT);
        for _ in 0..DEFAULT_HISTORY_LIMIT {
            assert!(runtime.rollback().is_ok());
        }
        assert!(matches!(
            runtime.rollback(),
            Err(LangRuntimeError::NoHistory)
        ));
    }

    #[test]
    fn semantic_change_in_any_section_is_a_revision() {
        let source = base_source();
        let (next, ()) = mutate(&source, |value| {
            value["system"]["default_language"] = toml::Value::String("en".to_string());
        });
        let (_dir, mut runtime) = runtime_with(&source);
        let outcome = runtime.apply_source(&next).unwrap();
        let LangReloadOutcome::Applied(apply) = outcome else {
            panic!("语义变化应形成新快照：{outcome:?}");
        };
        assert!(apply.diff.changed.contains(&"system".to_string()));
        assert!(apply
            .diff
            .restart_required
            .contains(&"system".to_string()));
        assert_eq!(
            runtime.current().language_config().system.default_language,
            config(&source).system.default_language,
            "系统节需重启，语言视图保持旧值"
        );
    }
}

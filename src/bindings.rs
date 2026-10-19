//! 统一变量绑定契约（Bound bindings）：为三套符号引擎提供类型化、显式未绑定、
//! 确定性序的地面真值。
//!
//! 绑定以 [`BoundTerm`] / [`BindingSet`] 表达：
//! - [`BoundTerm::Number`] 双存 `raw`（原样保留）+ `value`（供数值比较）；
//! - [`BoundTerm::Unbound`] 显式表达「没有绑定」；
//! - [`BindingSet`] 以 `BTreeMap` 保证确定性序遍历，并提供「required 槽位
//!   全部绑定」的接地闸门判定，供提示模板渲染消费。
//!
//! 未绑定的字符串形态采用隐式约定编码（如 `?X` / `"null"`），
//! 由各引擎适配器识别；数值分类使绑定可供比较与接地判定。
//!
//! 三引擎适配器分别放在各自模块内（[`crate::resolution_engine::ResolutionEngine::solve_typed`]、
//! [`crate::prolog_engine::PrologEngine::query_typed`]、以及
//! [`BindingSet::from_env`]），本模块不依赖任何引擎，保持契约干净。

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// 类型化绑定值。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundTerm {
    /// 文本常量（原样保留）。
    Text { raw: String },
    /// 数值：`raw` 用于渲染（原样保留），`value` 用于比较。
    Number { raw: String, value: f64 },
    /// 列表项（Scryer `Value::List` 等）。
    List { items: Vec<BoundTerm> },
    /// 结构项（Scryer `Value::Structure(atom, args)` 等）。
    Struct {
        name: String,
        fields: Vec<BoundTerm>,
    },
    /// 显式未绑定：与该变量的隐式约定（`?X` / `"null"`）对应。
    Unbound,
}

impl BoundTerm {
    pub fn text(raw: impl Into<String>) -> Self {
        BoundTerm::Text { raw: raw.into() }
    }

    pub fn number(raw: impl Into<String>, value: f64) -> Self {
        BoundTerm::Number {
            raw: raw.into(),
            value,
        }
    }

    /// 从字符串绑定分类：整串可解析为有限 `f64` 即数值，否则文本。
    ///
    /// 分类只影响类型化 API 的能力（比较 / 接地），`raw` 原样保留，
    /// 渲染时逐字输出原字符串——"226" 依旧是 "226"，"张三" 依旧是 "张三"。
    pub fn classify(raw: &str) -> Self {
        if !raw.is_empty() && raw.trim() == raw {
            if let Ok(value) = raw.parse::<f64>() {
                if value.is_finite() {
                    return BoundTerm::Number {
                        raw: raw.to_string(),
                        value,
                    };
                }
            }
        }
        BoundTerm::Text {
            raw: raw.to_string(),
        }
    }

    pub fn is_bound(&self) -> bool {
        !matches!(self, BoundTerm::Unbound)
    }

    /// 数值视图：`Number` 直接给出，可解析为有限数值的 `Text` 也给数值。
    pub fn numeric_value(&self) -> Option<f64> {
        match self {
            BoundTerm::Number { value, .. } => Some(*value),
            BoundTerm::Text { raw } => raw.parse::<f64>().ok().filter(|value| value.is_finite()),
            _ => None,
        }
    }

    /// 渲染为字符串形式（对 `Text`/`Number` 原样输出；
    /// 复杂项按 Scryer 的 JSON 风格渲染；`Unbound` 无字符串文本，返回 None）。
    pub fn render_raw(&self) -> Option<String> {
        match self {
            BoundTerm::Text { raw } | BoundTerm::Number { raw, .. } => Some(raw.clone()),
            BoundTerm::List { items } => {
                let rendered: Vec<String> = items
                    .iter()
                    .map(|item| item.render_raw().unwrap_or_else(|| "null".to_string()))
                    .collect();
                Some(format!("[{}]", rendered.join(",")))
            }
            BoundTerm::Struct { name, fields } => {
                let rendered: Vec<String> = fields
                    .iter()
                    .map(|item| item.render_raw().unwrap_or_else(|| "null".to_string()))
                    .collect();
                Some(format!("\"{}\":[{}]", name, rendered.join(",")))
            }
            BoundTerm::Unbound => None,
        }
    }
}

impl fmt::Display for BoundTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.render_raw() {
            Some(text) => write!(f, "{}", text),
            None => write!(f, "未绑定"),
        }
    }
}

/// 确定性变量绑定集合（按变量名排序，保证逐字可复现的报告）。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BindingSet {
    entries: BTreeMap<String, BoundTerm>,
}

impl BindingSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// 由 `logic_program` 的 [`crate::logic_program::Env`]（字符串映射）构造。
    pub fn from_env(env: &BTreeMap<String, String>) -> Self {
        let mut bindings = BindingSet::new();
        for (name, value) in env {
            bindings.insert(name.clone(), BoundTerm::classify(value));
        }
        bindings
    }

    pub fn insert(&mut self, name: impl Into<String>, term: BoundTerm) {
        self.entries.insert(name.into(), term);
    }

    /// 便捷绑定：字符串按 [`BoundTerm::classify`] 分类后写入。
    pub fn bind(&mut self, name: impl Into<String>, raw: &str) {
        self.insert(name, BoundTerm::classify(raw));
    }

    pub fn bind_text(&mut self, name: impl Into<String>, raw: &str) {
        self.insert(name, BoundTerm::text(raw));
    }

    pub fn get(&self, name: &str) -> Option<&BoundTerm> {
        self.entries.get(name)
    }

    /// 绑定存在且非 [`BoundTerm::Unbound`]。
    pub fn is_bound(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .map(BoundTerm::is_bound)
            .unwrap_or(false)
    }

    /// required 中未绑定的槽位名（确定性序，可直接进报告）。
    pub fn missing(&self, required: &[&str]) -> Vec<String> {
        required
            .iter()
            .filter(|name| !self.is_bound(name))
            .map(|name| (*name).to_string())
            .collect()
    }

    /// 接地闸门：required 槽位是否全部绑定。
    pub fn all_bound(&self, required: &[&str]) -> bool {
        required.iter().all(|name| self.is_bound(name))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &BoundTerm)> {
        self.entries.iter()
    }

    pub fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

impl From<&BTreeMap<String, String>> for BindingSet {
    fn from(env: &BTreeMap<String, String>) -> Self {
        BindingSet::from_env(env)
    }
}

/// 数值比较：两侧都拿到数值时给出 `Ordering`，否则 `None`
/// （「姚明和叶莉谁更高」这类问题的最小可组合原语）。
pub fn compare_numeric(left: &BoundTerm, right: &BoundTerm) -> Option<std::cmp::Ordering> {
    left.numeric_value()
        .and_then(|a| right.numeric_value().map(|b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)))
}

/// 由变量名和可选值构造绑定（`None` / 隐式未绑定串 → [`BoundTerm::Unbound`]）。
pub fn bound_or_unbound(
    name: impl Into<String>,
    value: Option<&str>,
    unbound: Option<&str>,
) -> (String, BoundTerm) {
    let name = name.into();
    let term = match value {
        Some(value) if Some(value) != unbound => BoundTerm::classify(value),
        Some(_) | None => BoundTerm::Unbound,
    };
    (name, term)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_splits_numbers_and_text_raw_preserved() {
        assert_eq!(
            BoundTerm::classify("226"),
            BoundTerm::Number {
                raw: "226".to_string(),
                value: 226.0
            }
        );
        assert_eq!(
            BoundTerm::classify("-1.5"),
            BoundTerm::Number {
                raw: "-1.5".to_string(),
                value: -1.5
            }
        );
        assert_eq!(BoundTerm::classify("张三"), BoundTerm::text("张三"));
        assert_eq!(BoundTerm::classify(""), BoundTerm::text(""));
        assert_eq!(BoundTerm::classify(" 226"), BoundTerm::text(" 226"));
        assert_eq!(BoundTerm::classify("NaN"), BoundTerm::text("NaN"));
        assert_eq!(BoundTerm::classify("inf"), BoundTerm::text("inf"));
        // 数值渲染原样输出
        for raw in ["226", "-1.5", "张三", " 226"] {
            assert_eq!(BoundTerm::classify(raw).render_raw().as_deref(), Some(raw));
        }
    }

    #[test]
    fn numeric_compare_works_across_number_and_text() {
        let tall = BoundTerm::classify("226");
        let short = BoundTerm::classify("190");
        assert_eq!(
            compare_numeric(&tall, &short),
            Some(std::cmp::Ordering::Greater)
        );
        // 字符串字典序会给出错误答案，数值比较不会
        assert!(compare_numeric(&short, &tall) == Some(std::cmp::Ordering::Less));
        assert_eq!(compare_numeric(&BoundTerm::text("张三"), &tall), None);
    }

    #[test]
    fn unbound_is_explicit_and_gate_detects_it() {
        let mut bindings = BindingSet::new();
        bindings.bind("身高", "226");
        bindings.insert("焦点", BoundTerm::Unbound);
        assert!(bindings.is_bound("身高"));
        assert!(!bindings.is_bound("焦点"));
        assert!(!bindings.is_bound("缺失"));
        assert_eq!(
            bindings.missing(&["身高", "焦点", "缺失"]),
            vec!["焦点".to_string(), "缺失".to_string()]
        );
        assert!(!bindings.all_bound(&["身高", "焦点"]));
        assert!(bindings.all_bound(&["身高"]));
    }

    #[test]
    fn from_env_classifies_and_orders_deterministically() {
        let mut env = BTreeMap::new();
        env.insert("Z".to_string(), "兔子".to_string());
        env.insert("A".to_string(), "3".to_string());
        let bindings = BindingSet::from_env(&env);
        assert_eq!(bindings.names(), vec!["A", "Z"]);
        assert_eq!(bindings.get("A").and_then(BoundTerm::numeric_value), Some(3.0));
        assert_eq!(bindings.get("Z").and_then(BoundTerm::numeric_value), None);
    }

    #[test]
    fn complex_terms_render_json_shape() {
        let term = BoundTerm::Struct {
            name: "point".to_string(),
            fields: vec![BoundTerm::classify("1"), BoundTerm::classify("2")],
        };
        assert_eq!(term.render_raw().as_deref(), Some("\"point\":[1,2]"));
        let list = BoundTerm::List {
            items: vec![BoundTerm::text("a"), BoundTerm::Unbound],
        };
        assert_eq!(list.render_raw().as_deref(), Some("[a,null]"));
        assert_eq!(BoundTerm::Unbound.render_raw(), None);
    }

    #[test]
    fn bound_or_unbound_distinguishes_conventions() {
        // resolution_engine 的约定：未绑定渲染为 "?X"
        assert_eq!(
            bound_or_unbound("X", Some("?X"), Some("?X")).1,
            BoundTerm::Unbound
        );
        assert_eq!(
            bound_or_unbound("X", Some("姚明"), Some("?X")).1,
            BoundTerm::text("姚明")
        );
        // scryer 的约定：未绑定渲染为 "null"
        assert_eq!(
            bound_or_unbound("X", Some("null"), Some("null")).1,
            BoundTerm::Unbound
        );
        // 但数值分类优先于「null 约定」之外的普通文本
        assert_eq!(
            bound_or_unbound("Age", Some("226"), Some("null")).1,
            BoundTerm::number("226", 226.0)
        );
    }

    #[test]
    fn serde_roundtrip_is_stable() {
        let mut bindings = BindingSet::new();
        bindings.bind("Age", "226");
        bindings.insert("Nick", BoundTerm::Unbound);
        let json = serde_json::to_string(&bindings).unwrap();
        let restored: BindingSet = serde_json::from_str(&json).unwrap();
        assert_eq!(bindings, restored);
    }
}

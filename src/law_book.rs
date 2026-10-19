//! 律典（Semantic Law Book）：把「声明的知识」与「学到的规则」记在同一本台账上。
//!
//! 系统里同一个关系语义可能同时出现在四处：谓词画像的配置声明、导入时观测到的
//! 统计、链接预测挖出的 Horn 规则、以及查询/生成路径上的隐式约定。它们各自成立，
//! 却没有一本可审计的总账。律典提供这一层：
//!
//!   - 统一：谓词画像声明（别名/互逆/功能/传递/对称/定义域/值域）与链接挖掘规则
//!     编译成同一批 `SemanticLaw`，每条有稳定 id、状态与证据；
//!   - 可审计：功能谓词多值 = 矛盾；传递闭包缺口/互逆缺口 = 可物化缺口；
//!   - 可回滚：声明律为 `Active`，挖掘律最多到 `Corroborated`，永不自动生效；
//!   - 可复述：`digest` 与逐条证据都可打印，回答引用定律时能说清是哪一条。
//!
//! 律典只读图谱事实，不写知识库；持久化写入是显式动作（`--save`），
//! 启动时只回放已保存的显式采纳/退役决定，不重新审判知识库。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::link_prediction::{HornRule, RuleTemplate, Triple};
use crate::predicate_registry::{PredicateProfile, PredicateRegistry};

/// 单本律典最多收录多少条挖掘规则（声明律不受限；规则按置信度/支撑度截断）。
pub const MAX_MINED_LAWS: usize = 256;

/// 每条律最多保留多少个反例文本（计数仍是全量）。
pub const MAX_COUNTEREXAMPLES: usize = 8;

/// 定律种类。声明类（别名/互逆/功能/传递/对称/定义域/值域）来自谓词画像；
/// 结构类（组合/共同邻居/共同主体）来自链接预测规则挖掘。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LawKind {
    Alias,
    Inverse,
    Functional,
    Transitive,
    Symmetric,
    Domain,
    Range,
    Compose,
    CommonNeighbor,
    CommonObject,
}

impl LawKind {
    pub fn label(&self) -> &'static str {
        match self {
            LawKind::Alias => "别名",
            LawKind::Inverse => "互逆",
            LawKind::Functional => "功能",
            LawKind::Transitive => "传递",
            LawKind::Symmetric => "对称",
            LawKind::Domain => "定义域",
            LawKind::Range => "值域",
            LawKind::Compose => "组合",
            LawKind::CommonNeighbor => "共同邻居",
            LawKind::CommonObject => "共同主体",
        }
    }

    pub fn tag(&self) -> &'static str {
        match self {
            LawKind::Alias => "alias",
            LawKind::Inverse => "inverse",
            LawKind::Functional => "functional",
            LawKind::Transitive => "transitive",
            LawKind::Symmetric => "symmetric",
            LawKind::Domain => "domain",
            LawKind::Range => "range",
            LawKind::Compose => "compose",
            LawKind::CommonNeighbor => "common_neighbor",
            LawKind::CommonObject => "common_object",
        }
    }

    /// 是否来自配置/人工确认的显式声明（声明律可以直接参与推理）。
    pub fn is_declared(&self) -> bool {
        matches!(
            self,
            LawKind::Alias
                | LawKind::Inverse
                | LawKind::Functional
                | LawKind::Transitive
                | LawKind::Symmetric
                | LawKind::Domain
                | LawKind::Range
        )
    }
}

/// 定律状态。学习只做证据累积与状态迁移，不做梯度更新。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LawStatus {
    /// 候选：有统计，尚未确认（挖掘规则首次出现时的状态）
    Proposed,
    /// 有旁证：与声明方向一致，或被多次独立观测支撑
    Corroborated,
    /// 生效：只由显式声明或人工采纳进入，参与推理与规划
    Active,
    /// 退役：显式撤销/被新律取代（不做自动退役）
    Retired,
}

impl LawStatus {
    pub fn label(&self) -> &'static str {
        match self {
            LawStatus::Proposed => "Proposed",
            LawStatus::Corroborated => "Corroborated",
            LawStatus::Active => "Active",
            LawStatus::Retired => "Retired",
        }
    }

    fn rank(&self) -> u8 {
        match self {
            LawStatus::Proposed => 0,
            LawStatus::Corroborated => 1,
            LawStatus::Active => 2,
            LawStatus::Retired => 3,
        }
    }
}

/// 定律证据：声明来源、支撑数、规则统计与覆盖率。没有权重，只有账本。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LawEvidence {
    pub support: usize,
    pub body_count: usize,
    pub confidence: f64,
    pub head_coverage: f64,
    pub sources: BTreeSet<String>,
}

impl LawEvidence {
    fn absorb(&mut self, other: &LawEvidence) {
        self.support = self.support.max(other.support);
        self.body_count = self.body_count.max(other.body_count);
        if other.confidence.is_finite() {
            self.confidence = self.confidence.max(other.confidence);
        }
        if other.head_coverage.is_finite() {
            self.head_coverage = self.head_coverage.max(other.head_coverage);
        }
        self.sources.extend(other.sources.iter().cloned());
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![format!("支撑 {}", self.support)];
        if self.body_count > 0 {
            parts.push(format!("体部 {}", self.body_count));
        }
        if self.confidence > 0.0 {
            parts.push(format!("置信度 {:.2}", self.confidence));
        }
        if self.head_coverage > 0.0 {
            parts.push(format!("覆盖率 {:.2}", self.head_coverage));
        }
        if !self.sources.is_empty() {
            parts.push(format!(
                "来源 {}",
                self.sources.iter().cloned().collect::<Vec<_>>().join(" / ")
            ));
        }
        parts.join("；")
    }
}

/// 一条语义定律。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticLaw {
    pub id: String,
    pub kind: LawKind,
    /// 规范谓词（可含参数谓词，按种类解释）
    pub terms: Vec<String>,
    pub status: LawStatus,
    pub evidence: LawEvidence,
    pub detail: String,
    #[serde(default)]
    pub counterexample_count: usize,
    #[serde(default)]
    pub counterexamples: Vec<String>,
}

impl SemanticLaw {
    fn new(
        kind: LawKind,
        terms: Vec<String>,
        status: LawStatus,
        evidence: LawEvidence,
        detail: String,
    ) -> Self {
        Self {
            id: law_id(kind, &terms),
            kind,
            terms,
            status,
            evidence,
            detail,
            counterexample_count: 0,
            counterexamples: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == LawStatus::Active
    }

    pub fn evidence_line(&self) -> String {
        self.evidence.summary()
    }
}

fn law_id(kind: LawKind, terms: &[String]) -> String {
    let mut parts: Vec<&str> = terms.iter().map(String::as_str).collect();
    if kind == LawKind::Inverse {
        parts.sort_unstable();
    }
    format!("law:{}:{}", kind.tag(), parts.join("|"))
}

fn profile_evidence(profile: &PredicateProfile) -> LawEvidence {
    let mut sources = profile.sources.clone();
    sources.insert("配置声明".to_string());
    LawEvidence {
        support: profile.support,
        body_count: 0,
        confidence: 1.0,
        head_coverage: 1.0,
        sources,
    }
}

/// 律典：`id → SemanticLaw` 的确定性映射，外加谓词归一表（审计时把词面还原到槽位）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LawBook {
    #[serde(default)]
    laws: BTreeMap<String, SemanticLaw>,
    /// 词面 → (规范谓词, 是否逆向)，与谓词注册表共用同一套归一约定。
    #[serde(default)]
    aliases: BTreeMap<String, (String, bool)>,
}

impl LawBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.laws.len()
    }

    pub fn is_empty(&self) -> bool {
        self.laws.is_empty()
    }

    pub fn law(&self, id: &str) -> Option<&SemanticLaw> {
        self.laws.get(id)
    }

    #[cfg(test)]
    fn law_mut(&mut self, id: &str) -> Option<&mut SemanticLaw> {
        self.laws.get_mut(id)
    }

    pub fn laws(&self) -> impl Iterator<Item = &SemanticLaw> {
        self.laws.values()
    }

    pub fn alias_map(&self) -> &BTreeMap<String, (String, bool)> {
        &self.aliases
    }

    /// 从谓词画像编译声明律。只有 `declared` 画像进入 Active；
    /// 自动准入的待确认谓词只作为观测统计，不变成可执行定律。
    pub fn from_registry(registry: &PredicateRegistry) -> Self {
        let mut book = Self::new();
        book.aliases = registry
            .alias_pairs()
            .into_iter()
            .map(|(surface, canonical, inverted)| (surface, (canonical, inverted)))
            .collect();

        for canonical in registry.canonical_names() {
            let Some(profile) = registry.profile(canonical) else {
                continue;
            };
            if !profile.declared {
                continue;
            }
            let evidence = profile_evidence(profile);
            let inverse = profile.inverse.clone();

            for alias in &profile.aliases {
                if inverse.as_deref() == Some(alias.as_str()) {
                    // 互逆词面会单独编译成 Inverse 律，避免同一条语义记两遍。
                    continue;
                }
                if alias == canonical {
                    continue;
                }
                // 互逆词面的别名（如 徒弟 的别名 学生）在注册表里并入槽位画像，
                // 但方向是反的：记为逆向谓词的别名，不能冒充正向别名。
                if registry.resolve(alias).inverted {
                    if let Some(inverse) = &inverse {
                        book.insert_law(SemanticLaw::new(
                            LawKind::Alias,
                            vec![inverse.clone(), alias.clone()],
                            LawStatus::Active,
                            evidence.clone(),
                            format!(
                                "别名「{}」归一为逆向谓词「{}」（查询时主宾换向）",
                                alias, inverse
                            ),
                        ));
                    }
                    continue;
                }
                book.insert_law(SemanticLaw::new(
                    LawKind::Alias,
                    vec![canonical.to_string(), alias.clone()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!("别名「{}」归一为规范谓词「{}」", alias, canonical),
                ));
            }

            if let Some(inverse) = &inverse {
                let mut terms = vec![canonical.to_string(), inverse.clone()];
                terms.sort();
                let mut law_evidence = evidence.clone();
                if let Some(other) = registry.profile(inverse) {
                    law_evidence.absorb(&profile_evidence(other));
                }
                book.insert_law(SemanticLaw::new(
                    LawKind::Inverse,
                    terms,
                    LawStatus::Active,
                    law_evidence,
                    format!("「{}」与「{}」互逆：p(a,b) ≡ q(b,a)", canonical, inverse),
                ));
            }

            if profile.functional {
                book.insert_law(SemanticLaw::new(
                    LawKind::Functional,
                    vec![canonical.to_string()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!("「{}」是功能谓词：同一主语最多一个取值", canonical),
                ));
            }
            if profile.transitive {
                book.insert_law(SemanticLaw::new(
                    LawKind::Transitive,
                    vec![canonical.to_string()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!(
                        "「{}」可传递：{}(a,b) ∧ {}(b,c) ⇒ {}(a,c)",
                        canonical, canonical, canonical, canonical
                    ),
                ));
            }
            if profile.symmetric {
                book.insert_law(SemanticLaw::new(
                    LawKind::Symmetric,
                    vec![canonical.to_string()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!(
                        "「{}」对称：{}(a,b) ⇒ {}(b,a)",
                        canonical, canonical, canonical
                    ),
                ));
            }
            if !profile.domain.is_empty() {
                book.insert_law(SemanticLaw::new(
                    LawKind::Domain,
                    vec![canonical.to_string()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!(
                        "「{}」的定义域：{}",
                        canonical,
                        profile
                            .domain
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("、")
                    ),
                ));
            }
            if !profile.range.is_empty() {
                book.insert_law(SemanticLaw::new(
                    LawKind::Range,
                    vec![canonical.to_string()],
                    LawStatus::Active,
                    evidence.clone(),
                    format!(
                        "「{}」的值域：{}",
                        canonical,
                        profile.range.iter().cloned().collect::<Vec<_>>().join("、")
                    ),
                ));
            }
        }
        book
    }

    /// 收录链接挖掘规则。挖掘律永远不会自动变成 Active：
    /// 与声明同向的规则升为 Corroborated，其余停在 Proposed。
    pub fn add_mined_rules(&mut self, rules: &[HornRule]) {
        let mut selected: Vec<&HornRule> =
            rules.iter().filter(|rule| !rule.body.is_empty()).collect();
        selected.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| {
                    b.head_coverage
                        .partial_cmp(&a.head_coverage)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.rule_text().cmp(&b.rule_text()))
        });
        selected.truncate(MAX_MINED_LAWS);

        for rule in selected {
            let (kind, terms) = match rule.template {
                RuleTemplate::Compose => (
                    LawKind::Compose,
                    vec![
                        rule.body1().to_string(),
                        rule.body2().unwrap_or("").to_string(),
                        rule.head.clone(),
                    ],
                ),
                RuleTemplate::Inverse => (
                    LawKind::Inverse,
                    vec![rule.body1().to_string(), rule.head.clone()],
                ),
                RuleTemplate::CommonNeighbor => (
                    LawKind::CommonNeighbor,
                    vec![
                        rule.body1().to_string(),
                        rule.body2().unwrap_or("").to_string(),
                        rule.head.clone(),
                    ],
                ),
                RuleTemplate::CommonObject => (
                    LawKind::CommonObject,
                    vec![
                        rule.body1().to_string(),
                        rule.body2().unwrap_or("").to_string(),
                        rule.head.clone(),
                    ],
                ),
            };
            let mut status = LawStatus::Proposed;
            let mut detail = rule.rule_text();
            if kind == LawKind::Compose
                && rule.body1() == rule.head
                && rule.body2() == Some(rule.head.as_str())
                && self.active_transitive(&rule.head).is_some()
            {
                status = LawStatus::Corroborated;
                detail.push_str("（与传递声明一致）");
            }
            if kind == LawKind::Inverse && self.active_inverse(&rule.body1(), &rule.head).is_some()
            {
                status = LawStatus::Corroborated;
                detail.push_str("（与互逆声明一致）");
            }
            let evidence = LawEvidence {
                support: rule.support,
                body_count: rule.body_count,
                confidence: rule.confidence,
                head_coverage: rule.head_coverage,
                sources: BTreeSet::from(["链接挖掘".to_string()]),
            };
            self.insert_law(SemanticLaw::new(kind, terms, status, evidence, detail));
        }
    }

    /// 人工采纳：Proposed/Corroborated 律显式升为 Active。
    /// 返回是否发生状态变化；不存在的 id 返回 false。
    pub fn adopt(&mut self, id: &str) -> bool {
        let Some(law) = self.laws.get_mut(id) else {
            return false;
        };
        if law.status == LawStatus::Active {
            return false;
        }
        law.status = LawStatus::Active;
        law.evidence.sources.insert("人工采纳".to_string());
        true
    }

    /// 人工退役：显式撤销一条律（保留记录作为墓碑，挖掘不会自动复活）。
    pub fn retire(&mut self, id: &str) -> bool {
        let Some(law) = self.laws.get_mut(id) else {
            return false;
        };
        if law.status == LawStatus::Retired {
            return false;
        }
        law.status = LawStatus::Retired;
        law.evidence.sources.insert("人工退役".to_string());
        true
    }

    /// 载入持久化账本中的显式决定。只迁移 Active/Retired：
    /// Proposed/Corroborated 是当前事实的派生结论，必须每次重新计算。
    /// 返回发生变化的定律数。
    pub fn apply_persisted(&mut self, persisted: &LawBook) -> usize {
        let mut changed = 0usize;
        for (id, persisted_law) in &persisted.laws {
            if !matches!(persisted_law.status, LawStatus::Active | LawStatus::Retired) {
                continue;
            }
            match self.laws.get_mut(id) {
                Some(law) => {
                    if law.status != persisted_law.status {
                        law.status = persisted_law.status;
                        law.evidence.sources.insert("持久化账本".to_string());
                        changed += 1;
                    }
                    law.evidence.absorb(&persisted_law.evidence);
                }
                None => {
                    let mut adopted = persisted_law.clone();
                    adopted.evidence.sources.insert("持久化账本".to_string());
                    self.laws.insert(id.clone(), adopted);
                    changed += 1;
                }
            }
        }
        changed
    }

    fn insert_law(&mut self, law: SemanticLaw) {
        if let Some(existing) = self.laws.get_mut(&law.id) {
            existing.evidence.absorb(&law.evidence);
            if law.status.rank() > existing.status.rank() {
                existing.status = law.status;
            }
            if existing.detail.is_empty() {
                existing.detail = law.detail;
            }
            return;
        }
        self.laws.insert(law.id.clone(), law);
    }

    pub fn active_transitive(&self, predicate: &str) -> Option<&SemanticLaw> {
        self.active_law(LawKind::Transitive, predicate)
    }

    pub fn active_symmetric(&self, predicate: &str) -> Option<&SemanticLaw> {
        self.active_law(LawKind::Symmetric, predicate)
    }

    pub fn active_functional(&self, predicate: &str) -> Option<&SemanticLaw> {
        self.active_law(LawKind::Functional, predicate)
    }

    pub fn active_inverse(&self, left: &str, right: &str) -> Option<&SemanticLaw> {
        let mut terms = [left.to_string(), right.to_string()];
        terms.sort();
        self.law(&law_id(LawKind::Inverse, &terms))
            .filter(|law| law.is_active())
    }

    /// 某谓词的互逆伙伴（按声明方向，不区分左右）。
    pub fn inverse_partner(&self, predicate: &str) -> Option<&str> {
        self.laws
            .values()
            .filter(|law| law.kind == LawKind::Inverse && law.is_active())
            .find_map(|law| match law.terms.as_slice() {
                [left, right] if left == predicate => Some(right.as_str()),
                [left, right] if right == predicate => Some(left.as_str()),
                _ => None,
            })
    }

    fn active_law(&self, kind: LawKind, predicate: &str) -> Option<&SemanticLaw> {
        self.law(&law_id(kind, &[predicate.to_string()]))
            .filter(|law| law.is_active())
    }

    /// 是否存在任何可参与查询扩展的声明律（传递/对称/互逆）。
    pub fn has_inference_laws(&self) -> bool {
        self.laws.values().any(|law| {
            law.is_active()
                && matches!(
                    law.kind,
                    LawKind::Transitive | LawKind::Symmetric | LawKind::Inverse
                )
        })
    }

    pub fn explain(&self, id: &str) -> Option<String> {
        let law = self.law(id)?;
        Some(format!(
            "{}律 {}：{}（{}，{}）",
            law.kind.label(),
            law.id,
            law.detail,
            law.status.label(),
            law.evidence_line()
        ))
    }

    pub fn status_count(&self, status: LawStatus) -> usize {
        self.laws
            .values()
            .filter(|law| law.status == status)
            .count()
    }

    /// 稳定摘要：排序后的定律定义 + 审计反例计数，不依赖哈希迭代顺序。
    pub fn digest(&self) -> String {
        let mut material = String::new();
        for law in self.laws.values() {
            material.push_str(&law.id);
            material.push('|');
            material.push_str(law.status.label());
            material.push('|');
            material.push_str(&law.terms.join("|"));
            material.push('|');
            material.push_str(&law.evidence.summary());
            material.push('|');
            material.push_str(&law.counterexample_count.to_string());
            material.push('\n');
        }
        fnv1a(material.as_bytes())
    }

    /// 审计：功能谓词多值 = 矛盾；传递闭包缺口 / 显式互逆缺口 = 可物化缺口。
    ///
    /// 审计只更新反例计数，不自动改变律的状态。状态迁移（采纳/退役）必须显式。
    pub fn audit(&mut self, triples: &[Triple]) -> LawAuditReport {
        let normalized = self.normalized_edges(triples);
        let raw = raw_edges(triples);
        let mut updates: Vec<(String, usize, Vec<String>)> = Vec::new();

        for law in self.laws.values() {
            if law.status == LawStatus::Retired {
                // 退役是显式撤销：旧约束不参与当前事实的审计。
                continue;
            }
            let (count, examples) = match law.kind {
                LawKind::Functional => audit_functional(law, &normalized),
                LawKind::Transitive => audit_transitive(law, &normalized),
                LawKind::Inverse => audit_inverse(law, &raw),
                _ => (0, Vec::new()),
            };
            updates.push((law.id.clone(), count, examples));
        }
        // 先清零再写入：重复审计只反映当前事实，旧反例不得残留。
        for law in self.laws.values_mut() {
            law.counterexample_count = 0;
            law.counterexamples.clear();
        }
        for (id, count, examples) in updates {
            if let Some(law) = self.laws.get_mut(&id) {
                law.counterexample_count = count;
                law.counterexamples = examples;
            }
        }

        let entries: Vec<LawAuditEntry> = self
            .laws
            .values()
            .map(|law| {
                let severity = match law.kind {
                    LawKind::Functional if law.counterexample_count > 0 => AuditSeverity::Violation,
                    _ if law.counterexample_count > 0 => AuditSeverity::Gap,
                    _ => AuditSeverity::None,
                };
                LawAuditEntry {
                    law_id: law.id.clone(),
                    kind: law.kind,
                    status: law.status,
                    severity,
                    counterexample_count: law.counterexample_count,
                    counterexamples: law.counterexamples.clone(),
                }
            })
            .collect();

        let violation_count = entries
            .iter()
            .filter(|entry| entry.severity == AuditSeverity::Violation)
            .map(|entry| entry.counterexample_count)
            .sum();
        let gap_count = entries
            .iter()
            .filter(|entry| entry.severity == AuditSeverity::Gap)
            .map(|entry| entry.counterexample_count)
            .sum();
        let digest = self.digest();
        LawAuditReport {
            law_count: self.laws.len(),
            active_count: self.status_count(LawStatus::Active),
            corroborated_count: self.status_count(LawStatus::Corroborated),
            proposed_count: self.status_count(LawStatus::Proposed),
            retired_count: self.status_count(LawStatus::Retired),
            violation_count,
            gap_count,
            entries,
            digest,
        }
    }

    fn normalized_edges(&self, triples: &[Triple]) -> BTreeMap<String, BTreeSet<(String, String)>> {
        let mut edges: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();
        for triple in triples {
            let (predicate, inverted) = match self.aliases.get(&triple.predicate) {
                Some((canonical, inverted)) => (canonical.clone(), *inverted),
                None => (triple.predicate.clone(), false),
            };
            let (subject, object) = if inverted {
                (triple.object.clone(), triple.subject.clone())
            } else {
                (triple.subject.clone(), triple.object.clone())
            };
            edges
                .entry(predicate)
                .or_default()
                .insert((subject, object));
        }
        edges
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "律典：{} 条定律（Active {} / Corroborated {} / Proposed {} / Retired {}），digest {}。\n",
            self.laws.len(),
            self.status_count(LawStatus::Active),
            self.status_count(LawStatus::Corroborated),
            self.status_count(LawStatus::Proposed),
            self.status_count(LawStatus::Retired),
            self.digest()
        );
        if self.laws.is_empty() {
            output.push_str(
                "尚无定律：在 config.toml 的 [[predicate_profiles]] 中声明，或先挖掘链接规则。\n",
            );
            return output;
        }
        let mut proposed = 0usize;
        for law in self.laws.values() {
            match law.status {
                LawStatus::Active => {}
                LawStatus::Proposed | LawStatus::Corroborated => proposed += 1,
                LawStatus::Retired => continue,
            }
            // 挖掘律可能很多：Proposed 只展开前 12 条，账本其余仍可 JSON 查看。
            if !law.is_active() && proposed > 12 {
                continue;
            }
            output.push_str(&format!(
                "  [{}] {} {}  {}\n",
                law.status.label(),
                law.kind.label(),
                law.id,
                law.detail
            ));
            output.push_str(&format!("      证据：{}\n", law.evidence_line()));
        }
        if proposed > 12 {
            output.push_str(&format!(
                "  ……另有 {} 条候选/旁证律（用 --json 查看全量）。\n",
                proposed - 12
            ));
        }
        output
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditSeverity {
    None,
    Gap,
    Violation,
}

impl AuditSeverity {
    pub fn label(&self) -> &'static str {
        match self {
            AuditSeverity::None => "OK",
            AuditSeverity::Gap => "GAP",
            AuditSeverity::Violation => "VIOLATION",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawAuditEntry {
    pub law_id: String,
    pub kind: LawKind,
    pub status: LawStatus,
    pub severity: AuditSeverity,
    pub counterexample_count: usize,
    pub counterexamples: Vec<String>,
}

/// 审计报告：与律典正文同构、确定性排序；可直接 JSON 序列化为 CI 基线。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawAuditReport {
    pub law_count: usize,
    pub active_count: usize,
    pub corroborated_count: usize,
    pub proposed_count: usize,
    pub retired_count: usize,
    pub violation_count: usize,
    pub gap_count: usize,
    pub entries: Vec<LawAuditEntry>,
    pub digest: String,
}

impl LawAuditReport {
    /// 无矛盾。缺口（可物化但尚未物化）不算矛盾，单独计数。
    pub fn is_clean(&self) -> bool {
        self.violation_count == 0
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "审计：{} 条定律；矛盾 {} 处，可物化缺口 {} 处。{}\n",
            self.law_count,
            self.violation_count,
            self.gap_count,
            if self.is_clean() {
                "未发现违反声明的功能谓词。"
            } else {
                "存在功能谓词多值冲突，需要人工裁决（律典不自动覆盖/删除）。"
            }
        );
        for entry in &self.entries {
            if entry.counterexample_count == 0 {
                continue;
            }
            output.push_str(&format!(
                "  [{}] {} {}：{} 处\n",
                entry.severity.label(),
                entry.kind.label(),
                entry.law_id,
                entry.counterexample_count
            ));
            for example in &entry.counterexamples {
                output.push_str(&format!("      - {}\n", example));
            }
        }
        output
    }
}

fn audit_functional(
    law: &SemanticLaw,
    edges: &BTreeMap<String, BTreeSet<(String, String)>>,
) -> (usize, Vec<String>) {
    let Some(predicate) = law.terms.first() else {
        return (0, Vec::new());
    };
    let Some(pairs) = edges.get(predicate) else {
        return (0, Vec::new());
    };
    let mut values: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (subject, object) in pairs {
        values
            .entry(subject.as_str())
            .or_default()
            .insert(object.as_str());
    }
    let mut count = 0usize;
    let mut examples = Vec::new();
    for (subject, objects) in values {
        if objects.len() <= 1 {
            continue;
        }
        count += 1;
        if examples.len() < MAX_COUNTEREXAMPLES {
            examples.push(format!(
                "{} 的 {} 有 {} 个取值：{}",
                subject,
                predicate,
                objects.len(),
                objects.iter().cloned().collect::<Vec<_>>().join("、")
            ));
        }
    }
    (count, examples)
}

fn audit_transitive(
    law: &SemanticLaw,
    edges: &BTreeMap<String, BTreeSet<(String, String)>>,
) -> (usize, Vec<String>) {
    let Some(predicate) = law.terms.first() else {
        return (0, Vec::new());
    };
    let Some(pairs) = edges.get(predicate) else {
        return (0, Vec::new());
    };
    let mut successors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (subject, object) in pairs {
        successors
            .entry(subject.as_str())
            .or_default()
            .insert(object.as_str());
    }
    let mut missing: BTreeSet<(String, String)> = BTreeSet::new();
    let mut examples = Vec::new();
    for (subject, objects) in &successors {
        for object in objects {
            let Some(next) = successors.get(*object) else {
                continue;
            };
            for target in next {
                if subject == target {
                    continue;
                }
                if !pairs.contains(&(subject.to_string(), target.to_string())) {
                    let key = (subject.to_string(), target.to_string());
                    if missing.insert(key) && examples.len() < MAX_COUNTEREXAMPLES {
                        examples.push(format!(
                            "{} → {} → {}，缺 {}({}, {})",
                            subject, object, target, predicate, subject, target
                        ));
                    }
                }
            }
        }
    }
    (missing.len(), examples)
}

fn audit_inverse(
    law: &SemanticLaw,
    raw: &BTreeMap<String, BTreeSet<(String, String)>>,
) -> (usize, Vec<String>) {
    let [left, right] = match law.terms.as_slice() {
        [left, right] => [left, right],
        _ => return (0, Vec::new()),
    };
    // 只有两个方向都显式出现在图里时才审计互逆缺口。互逆归一后的知识
    // 通常只落一个方向，此时缺失的反向事实是推理可得的，不是矛盾。
    let (Some(left_edges), Some(right_edges)) = (raw.get(left), raw.get(right)) else {
        return (0, Vec::new());
    };
    let mut missing: BTreeSet<(String, String)> = BTreeSet::new();
    let mut examples = Vec::new();
    for (subject, object) in left_edges {
        if !right_edges.contains(&(object.clone(), subject.clone())) {
            let key = (subject.clone(), object.clone());
            if missing.insert(key) && examples.len() < MAX_COUNTEREXAMPLES {
                examples.push(format!(
                    "{} → {} 缺反向事实 {} → {}",
                    subject, object, object, subject
                ));
            }
        }
    }
    for (subject, object) in right_edges {
        if !left_edges.contains(&(object.clone(), subject.clone())) {
            let key = (subject.clone(), object.clone());
            if missing.insert(key) && examples.len() < MAX_COUNTEREXAMPLES {
                examples.push(format!(
                    "{} → {} 缺反向事实 {} → {}",
                    subject, object, object, subject
                ));
            }
        }
    }
    (missing.len(), examples)
}

fn raw_edges(triples: &[Triple]) -> BTreeMap<String, BTreeSet<(String, String)>> {
    let mut edges: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();
    for triple in triples {
        edges
            .entry(triple.predicate.clone())
            .or_default()
            .insert((triple.subject.clone(), triple.object.clone()));
    }
    edges
}

/// FNV-1a：小、稳定、无依赖；只用于账本摘要，不用于安全用途。
fn fnv1a(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_prediction::LinkPredictor;
    use crate::predicate_registry::PredicateProfileConfig;

    fn profile(name: &str) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        let mut master = profile("师傅");
        master.aliases = vec!["老师".to_string()];
        master.inverse = Some("徒弟".to_string());
        master.transitive = true;
        master.domain = vec!["人物".to_string()];
        registry.declare(&master);
        let mut apprentice = profile("徒弟");
        apprentice.aliases = vec!["学生".to_string()];
        registry.declare(&apprentice);
        let mut age = profile("年龄");
        age.functional = true;
        registry.declare(&age);
        let mut spouse = profile("配偶");
        spouse.symmetric = true;
        registry.declare(&spouse);
        registry
    }

    fn triple(subject: &str, predicate: &str, object: &str) -> Triple {
        Triple::new(subject, predicate, object)
    }

    #[test]
    fn declarations_become_active_with_stable_digest() {
        let book = LawBook::from_registry(&registry());
        assert!(book.active_transitive("师傅").is_some());
        assert!(book.active_functional("年龄").is_some());
        assert!(book.active_symmetric("配偶").is_some());
        assert!(book.active_inverse("师傅", "徒弟").is_some());
        assert_eq!(book.inverse_partner("徒弟"), Some("师傅"));
        assert!(book
            .laws()
            .any(|law| law.kind == LawKind::Alias && law.terms[1] == "老师"));
        // 互逆词面不应同时被记为普通别名；其别名应挂在逆向谓词下。
        assert!(!book
            .laws()
            .any(|law| law.kind == LawKind::Alias && law.terms[1] == "徒弟"));
        assert!(!book.laws().any(|law| law.kind == LawKind::Alias
            && law.terms[0] == "师傅"
            && law.terms[1] == "学生"));
        assert!(book.laws().any(|law| law.kind == LawKind::Alias
            && law.terms[0] == "徒弟"
            && law.terms[1] == "学生"));

        let twin = LawBook::from_registry(&registry());
        assert_eq!(book.digest(), twin.digest());
        assert_eq!(book.to_json(), twin.to_json());
    }

    /// 显式采纳/退役必须能穿过 JSON 往返；候选状态不得被持久化账本复活。
    #[test]
    fn adoption_and_retirement_survive_round_trip() {
        let mut book = LawBook::from_registry(&registry());
        let compose = SemanticLaw::new(
            LawKind::Compose,
            vec!["师傅".to_string(), "师傅".to_string(), "师祖".to_string()],
            LawStatus::Proposed,
            LawEvidence {
                support: 1,
                body_count: 2,
                confidence: 1.0,
                head_coverage: 1.0,
                sources: BTreeSet::from(["链接挖掘".to_string()]),
            },
            "手工构造的候选组合律".to_string(),
        );
        book.insert_law(compose.clone());
        assert_eq!(book.law(&compose.id).unwrap().status, LawStatus::Proposed);
        assert!(book.adopt(&compose.id), "显式采纳应改变候选状态");
        assert!(book.law(&compose.id).unwrap().is_active());
        assert!(!book.adopt(&compose.id), "重复采纳应无变化");

        let mut restored = LawBook::from_json(&book.to_json()).expect("往返应成功");
        assert!(book.retire("law:transitive:师傅"));
        assert!(restored.retire("law:transitive:师傅"));
        let restored = LawBook::from_json(&restored.to_json()).expect("往返应成功");

        let mut fresh = LawBook::from_registry(&registry());
        let changed = fresh.apply_persisted(&restored);
        assert!(changed >= 2, "采纳与退役都应迁移");
        assert_eq!(fresh.law(&compose.id).unwrap().status, LawStatus::Active);
        assert!(
            fresh.active_transitive("师傅").is_none(),
            "退役律不得参与推理"
        );
        assert!(fresh.retire("law:transitive:师傅") == false);
    }

    /// 只持久化显式决定：上次审计产生的 Proposed/Corroborated 不应覆盖当前计算。
    #[test]
    fn persisted_candidates_do_not_override_current_audit() {
        let mut persisted = LawBook::from_registry(&registry());
        let id = "law:transitive:师傅";
        persisted.law_mut(id).unwrap().status = LawStatus::Proposed;
        let mut current = LawBook::from_registry(&registry());
        current.apply_persisted(&persisted);
        assert!(current.active_transitive("师傅").is_some());
    }

    /// 退役律不产生矛盾或缺口：撤销必须是可执行的，而不只是账面上的状态。
    #[test]
    fn retired_law_is_not_audited() {
        let mut book = LawBook::from_registry(&registry());
        assert!(book.retire("law:functional:年龄"));
        let report = book.audit(&[triple("张三", "年龄", "20"), triple("张三", "年龄", "30")]);
        assert_eq!(report.violation_count, 0, "{}", report.render_text());
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.law_id == "law:functional:年龄")
            .expect("退役律仍应留在账本");
        assert_eq!(entry.status, LawStatus::Retired);
        assert_eq!(entry.counterexample_count, 0);
    }

    #[test]
    fn functional_multi_value_is_a_violation() {
        let mut book = LawBook::from_registry(&registry());
        let report = book.audit(&[
            triple("张三", "年龄", "18"),
            triple("张三", "年龄", "19"),
            triple("李四", "年龄", "20"),
        ]);
        assert!(!report.is_clean());
        assert_eq!(report.violation_count, 1);
        let law = book
            .law(&law_id(LawKind::Functional, &["年龄".to_string()]))
            .unwrap();
        assert_eq!(law.counterexample_count, 1);
        assert!(law.counterexamples[0].contains("张三"));
    }

    #[test]
    fn transitive_closure_gap_is_reported_and_closable() {
        let mut book = LawBook::from_registry(&registry());
        let incomplete = [
            triple("张三", "师傅", "李四"),
            triple("李四", "师傅", "王五"),
        ];
        let report = book.audit(&incomplete);
        assert!(report.is_clean(), "缺口不是矛盾");
        assert_eq!(report.gap_count, 1);
        let law = book
            .law(&law_id(LawKind::Transitive, &["师傅".to_string()]))
            .unwrap();
        assert_eq!(law.counterexample_count, 1);

        let complete = [
            triple("张三", "师傅", "李四"),
            triple("李四", "师傅", "王五"),
            triple("张三", "师傅", "王五"),
        ];
        let report = book.audit(&complete);
        assert_eq!(report.gap_count, 0);
    }

    #[test]
    fn inverse_gap_requires_both_explicit_directions() {
        let mut book = LawBook::from_registry(&registry());
        let normalized_only = [
            triple("张三", "师傅", "李四"),
            triple("李四", "师傅", "王五"),
            triple("张三", "师傅", "王五"),
        ];
        assert_eq!(book.audit(&normalized_only).gap_count, 0);

        let explicit_inverse = [
            triple("张三", "师傅", "李四"),
            triple("王五", "徒弟", "赵六"),
        ];
        let report = book.audit(&explicit_inverse);
        assert_eq!(report.gap_count, 2, "{}", report.render_text());
    }

    #[test]
    fn mined_rules_stay_proposed_or_corroborated() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.mine_rules(1, 0.5);

        let mut book = LawBook::from_registry(&registry());
        book.add_mined_rules(predictor.rules());
        assert!(book
            .laws()
            .any(|law| law.kind == LawKind::Compose && law.status != LawStatus::Active));
        assert!(book
            .laws()
            .filter(|law| !law.kind.is_declared())
            .all(|law| law.status != LawStatus::Active));
    }

    #[test]
    fn json_round_trip_preserves_laws_and_digest() {
        let book = LawBook::from_registry(&registry());
        let restored = LawBook::from_json(&book.to_json()).expect("律典 JSON 应可回读");
        assert_eq!(book.digest(), restored.digest());
        assert_eq!(book.len(), restored.len());
    }
}

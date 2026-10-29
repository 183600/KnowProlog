use crate::context_image::fnv1a_hex;
use crate::logic_program::{parse_rule, LAtom, LLiteral, LTerm};
use crate::resolution_engine::{
    apply_atom, freshen_clause, unify_atoms, Atom, Clause, Substitution, Term,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_CONFLICT_RULE_PAIRS: usize = 100_000;

fn default_confidence() -> f64 {
    1.0
}

fn default_key() -> usize {
    0
}

fn default_value() -> usize {
    1
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionalKey {
    pub predicate: String,
    #[serde(default = "default_key")]
    pub key: usize,
    #[serde(default = "default_value")]
    pub value: usize,
}

impl FunctionalKey {
    pub fn new(predicate: &str, key: usize, value: usize) -> Self {
        Self {
            predicate: predicate.to_string(),
            key,
            value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusiveGroup {
    pub predicates: Vec<String>,
    #[serde(default = "default_key")]
    pub key: usize,
}

impl ExclusiveGroup {
    pub fn new(predicates: &[&str], key: usize) -> Self {
        Self {
            predicates: predicates.iter().map(|value| value.to_string()).collect(),
            key,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleSpec {
    #[serde(default)]
    pub id: String,
    pub head: String,
    #[serde(default)]
    pub body: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

impl RuleSpec {
    pub fn new(id: &str, head: &str, body: &[&str]) -> Self {
        Self {
            id: id.to_string(),
            head: head.to_string(),
            body: body.iter().map(|value| value.to_string()).collect(),
            confidence: default_confidence(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RulePack {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub layer: i32,
    #[serde(default)]
    pub activation: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub conflicts: Vec<String>,
    #[serde(default)]
    pub functional: Vec<FunctionalKey>,
    #[serde(default)]
    pub exclusive: Vec<ExclusiveGroup>,
    #[serde(default)]
    pub rules: Vec<RuleSpec>,
}

impl RulePack {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            name: String::new(),
            layer: 0,
            activation: Vec::new(),
            requires: Vec::new(),
            conflicts: Vec::new(),
            functional: Vec::new(),
            exclusive: Vec::new(),
            rules: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    pub fn with_rule(mut self, rule: RuleSpec) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_functional(mut self, key: FunctionalKey) -> Self {
        self.functional.push(key);
        self
    }

    pub fn with_exclusive(mut self, group: ExclusiveGroup) -> Self {
        self.exclusive.push(group);
        self
    }

    pub fn with_requires(mut self, id: &str) -> Self {
        self.requires.push(id.to_string());
        self
    }

    pub fn with_conflicts(mut self, id: &str) -> Self {
        self.conflicts.push(id.to_string());
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Error,
    Warning,
}

impl IssueSeverity {
    pub fn label_zh(self) -> &'static str {
        match self {
            IssueSeverity::Error => "错误",
            IssueSeverity::Warning => "警告",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PackIssue {
    pub severity: IssueSeverity,
    pub pack: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    pub field: String,
    pub message: String,
}

impl PackIssue {
    fn new(
        severity: IssueSeverity,
        pack: &str,
        rule: Option<&str>,
        field: &str,
        message: String,
    ) -> Self {
        Self {
            severity,
            pack: pack.to_string(),
            rule: rule.map(str::to_string),
            field: field.to_string(),
            message,
        }
    }

    pub fn render(&self) -> String {
        let mut prefix = format!("[{}][{}]", self.severity.label_zh(), self.pack);
        if let Some(rule) = &self.rule {
            prefix.push_str(&format!("[{}]", rule));
        }
        format!("{} {}：{}", prefix, self.field, self.message)
    }
}

#[derive(Clone, Debug)]
pub struct CompiledRule {
    pub pack: String,
    pub rule_id: String,
    pub clause: Clause,
}

#[derive(Clone, Debug)]
pub struct CompiledPack {
    pub spec: RulePack,
    pub rules: Vec<CompiledRule>,
    pub issues: Vec<PackIssue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleConflictKind {
    Functional,
    Exclusive,
}

impl RuleConflictKind {
    pub fn label_zh(self) -> &'static str {
        match self {
            RuleConflictKind::Functional => "功能谓词分歧",
            RuleConflictKind::Exclusive => "互斥谓词冲突",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessStatus {
    Potential,
    Actual,
    Adjudicated,
}

impl WitnessStatus {
    pub fn label_zh(self) -> &'static str {
        match self {
            WitnessStatus::Potential => "潜在",
            WitnessStatus::Actual => "已见证",
            WitnessStatus::Adjudicated => "已裁决",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuleConflictWitness {
    pub id: String,
    pub kind: RuleConflictKind,
    pub status: WitnessStatus,
    pub left_pack: String,
    pub left_rule: String,
    pub right_pack: String,
    pub right_rule: String,
    pub subject: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_predicate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_predicate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_value: Option<String>,
    pub left_head: String,
    pub right_head: String,
    pub left_body: Vec<String>,
    pub right_body: Vec<String>,
    pub same_pack: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_pack: Option<String>,
}

impl RuleConflictWitness {
    pub fn render_zh(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "规则冲突 {}（{}，{}）：{}",
            self.id,
            self.kind.label_zh(),
            self.status.label_zh(),
            self.subject_line()
        ));
        lines.push(format!(
            "  左：{} [{} / {}] {}",
            self.left_head,
            self.left_pack,
            self.left_rule,
            render_body(&self.left_body)
        ));
        lines.push(format!(
            "  右：{} [{} / {}] {}",
            self.right_head,
            self.right_pack,
            self.right_rule,
            render_body(&self.right_body)
        ));
        if let Some(preferred) = &self.preferred_pack {
            lines.push(format!("  已裁决：优先采用「{}」。", preferred));
        } else {
            lines.push(format!("  {}", self.clarification_question()));
        }
        lines.join("\n")
    }

    pub fn clarification_question(&self) -> String {
        match self.kind {
            RuleConflictKind::Functional => format!(
                "关于「{}」的「{}」，规则包「{}」给出「{}」，规则包「{}」给出「{}」。你希望采用哪一个？",
                self.subject,
                self.predicate.as_deref().unwrap_or("未知谓词"),
                self.left_pack,
                self.left_value.as_deref().unwrap_or("?"),
                self.right_pack,
                self.right_value.as_deref().unwrap_or("?"),
            ),
            RuleConflictKind::Exclusive => format!(
                "关于「{}」，「{}」与「{}」被声明为互斥，但规则包「{}」和「{}」可能同时成立。你希望保留哪一边？",
                self.subject,
                self.left_predicate.as_deref().unwrap_or("?"),
                self.right_predicate.as_deref().unwrap_or("?"),
                self.left_pack,
                self.right_pack,
            ),
        }
    }

    fn subject_line(&self) -> String {
        match self.kind {
            RuleConflictKind::Functional => format!(
                "主语 {} 的「{}」可能同时取「{}」与「{}」",
                self.subject,
                self.predicate.as_deref().unwrap_or("?"),
                self.left_value.as_deref().unwrap_or("?"),
                self.right_value.as_deref().unwrap_or("?")
            ),
            RuleConflictKind::Exclusive => format!(
                "主语 {} 可能同时满足「{}」与「{}」",
                self.subject,
                self.left_predicate.as_deref().unwrap_or("?"),
                self.right_predicate.as_deref().unwrap_or("?")
            ),
        }
    }
}

fn render_body(body: &[String]) -> String {
    if body.is_empty() {
        "（无前提）".to_string()
    } else {
        format!(":- {}", body.join(", "))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompositionReport {
    pub packs: usize,
    pub rules: usize,
    pub issues: Vec<PackIssue>,
    pub conflicts: Vec<RuleConflictWitness>,
    pub truncated: bool,
}

impl CompositionReport {
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == IssueSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == IssueSeverity::Warning)
            .count()
    }

    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }

    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    pub fn adjudicate(&mut self, conflict_id: &str, prefer_pack: &str) -> bool {
        let mut changed = false;
        for conflict in &mut self.conflicts {
            if conflict.id != conflict_id {
                continue;
            }
            if conflict.left_pack == prefer_pack || conflict.right_pack == prefer_pack {
                conflict.preferred_pack = Some(prefer_pack.to_string());
                conflict.status = WitnessStatus::Adjudicated;
                changed = true;
            }
        }
        changed
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "规则包组合：{} 个包、{} 条规则；{} 个冲突，{} 个错误、{} 个警告{}。",
            self.packs,
            self.rules,
            self.conflicts.len(),
            self.error_count(),
            self.warning_count(),
            if self.truncated {
                "（规则对超过预算，冲突检测已截断）"
            } else {
                ""
            }
        )];
        if self.issues.is_empty() {
            lines.push("结构检查：通过。".to_string());
        } else {
            lines.push("结构检查：".to_string());
            for issue in &self.issues {
                lines.push(format!("  {}", issue.render()));
            }
        }
        if self.conflicts.is_empty() {
            lines.push("组合冲突：未发现。".to_string());
        } else {
            lines.push("组合冲突：".to_string());
            for conflict in &self.conflicts {
                lines.push(conflict.render_zh());
            }
            if self
                .conflicts
                .iter()
                .any(|conflict| conflict.preferred_pack.is_none())
            {
                lines.push(
                    "可用「规则裁决 <编号或冲突id>=<规则包>」记录选择，下一轮重建生效。"
                        .to_string(),
                );
            }
        }
        lines.join("\n")
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct RuleProgram {
    pub packs: Vec<CompiledPack>,
    pub report: CompositionReport,
}

impl RuleProgram {
    pub fn clauses(&self) -> Vec<CompiledRule> {
        self.packs
            .iter()
            .flat_map(|pack| pack.rules.iter().cloned())
            .collect()
    }

    /// 按已记录的裁决抑制落选规则，使选择在下一轮重建时真正生效。
    pub fn apply_adjudications(&mut self) {
        let mut suppressed: BTreeSet<(String, String)> = BTreeSet::new();
        let mut kept: BTreeSet<(String, String)> = BTreeSet::new();
        for conflict in &self.report.conflicts {
            let Some(preferred) = &conflict.preferred_pack else {
                continue;
            };
            let left = (conflict.left_pack.clone(), conflict.left_rule.clone());
            let right = (conflict.right_pack.clone(), conflict.right_rule.clone());
            if preferred == &conflict.left_pack {
                kept.insert(left);
                suppressed.insert(right);
            } else if preferred == &conflict.right_pack {
                kept.insert(right);
                suppressed.insert(left);
            }
        }
        for rule in &kept {
            suppressed.remove(rule);
        }
        if suppressed.is_empty() {
            return;
        }
        for pack in &mut self.packs {
            pack.rules
                .retain(|rule| !suppressed.contains(&(rule.pack.clone(), rule.rule_id.clone())));
        }
        self.report.rules = self.packs.iter().map(|pack| pack.rules.len()).sum();
    }
}

pub fn compile_packs(packs: &[RulePack]) -> RuleProgram {
    let mut report = CompositionReport {
        packs: packs.len(),
        ..CompositionReport::default()
    };
    let mut compiled: Vec<CompiledPack> = Vec::new();
    let mut seen_pack_ids: BTreeSet<String> = BTreeSet::new();

    for pack in packs {
        let mut pack = pack.clone();
        if pack.id.trim().is_empty() {
            report.issues.push(PackIssue::new(
                IssueSeverity::Error,
                "<unknown>",
                None,
                "id",
                "规则包 id 不能为空。".to_string(),
            ));
            continue;
        }
        if !seen_pack_ids.insert(pack.id.clone()) {
            report.issues.push(PackIssue::new(
                IssueSeverity::Error,
                &pack.id,
                None,
                "id",
                "规则包 id 重复，后者被忽略。".to_string(),
            ));
            continue;
        }
        if pack.name.trim().is_empty() {
            pack.name = pack.id.clone();
        }
        let mut issues = validate_declarations(&pack);
        let rules = compile_rules(&pack, &mut issues);
        report.rules += rules.len();
        report.issues.extend(issues.iter().cloned());
        compiled.push(CompiledPack {
            spec: pack,
            rules,
            issues,
        });
    }

    report.packs = compiled.len();
    report.issues.extend(cross_pack_issues(&compiled, packs));
    report.conflicts = detect_rule_conflicts(&compiled, &mut report.truncated);
    RuleProgram {
        packs: compiled,
        report,
    }
}

pub fn validate_pack(pack: &RulePack) -> Vec<PackIssue> {
    let mut issues = validate_declarations(pack);
    let _ = compile_rules(pack, &mut issues);
    issues
}

fn validate_declarations(pack: &RulePack) -> Vec<PackIssue> {
    let mut issues = Vec::new();
    if pack.rules.is_empty() {
        issues.push(PackIssue::new(
            IssueSeverity::Warning,
            &pack.id,
            None,
            "rules",
            "规则包没有规则。".to_string(),
        ));
    }
    if pack.requires.iter().any(|required| required == &pack.id) {
        issues.push(PackIssue::new(
            IssueSeverity::Error,
            &pack.id,
            None,
            "requires",
            "规则包不能依赖自身。".to_string(),
        ));
    }
    if pack.conflicts.iter().any(|other| other == &pack.id) {
        issues.push(PackIssue::new(
            IssueSeverity::Warning,
            &pack.id,
            None,
            "conflicts",
            "规则包与自己冲突，已忽略该声明。".to_string(),
        ));
    }

    let mut functional: BTreeMap<&str, &FunctionalKey> = BTreeMap::new();
    for key in &pack.functional {
        let predicate = key.predicate.trim();
        if predicate.is_empty() {
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                &pack.id,
                None,
                "functional",
                "功能谓词名不能为空。".to_string(),
            ));
            continue;
        }
        if key.key == key.value {
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                &pack.id,
                None,
                "functional",
                format!("功能谓词「{}」的 key 与 value 位置不能相同。", predicate),
            ));
        }
        if let Some(previous) = functional.insert(predicate, key) {
            if previous.key != key.key || previous.value != key.value {
                issues.push(PackIssue::new(
                    IssueSeverity::Error,
                    &pack.id,
                    None,
                    "functional",
                    format!("功能谓词「{}」存在互相矛盾的位置声明。", predicate),
                ));
            }
        }
    }

    for group in &pack.exclusive {
        if group.predicates.len() < 2 {
            issues.push(PackIssue::new(
                IssueSeverity::Warning,
                &pack.id,
                None,
                "exclusive",
                "互斥谓词组至少需要两个谓词。".to_string(),
            ));
        }
        let mut seen = BTreeSet::new();
        for predicate in &group.predicates {
            let predicate = predicate.trim();
            if predicate.is_empty() {
                issues.push(PackIssue::new(
                    IssueSeverity::Error,
                    &pack.id,
                    None,
                    "exclusive",
                    "互斥谓词名不能为空。".to_string(),
                ));
            } else if !seen.insert(predicate.to_string()) {
                issues.push(PackIssue::new(
                    IssueSeverity::Warning,
                    &pack.id,
                    None,
                    "exclusive",
                    format!("互斥谓词组中「{}」重复。", predicate),
                ));
            }
        }
    }
    issues
}

fn compile_rules(pack: &RulePack, issues: &mut Vec<PackIssue>) -> Vec<CompiledRule> {
    let functional: BTreeMap<&str, &FunctionalKey> = pack
        .functional
        .iter()
        .map(|key| (key.predicate.as_str(), key))
        .collect();
    let exclusive: BTreeMap<&str, &ExclusiveGroup> = pack
        .exclusive
        .iter()
        .flat_map(|group| {
            group
                .predicates
                .iter()
                .map(move |predicate| (predicate.as_str(), group))
        })
        .collect();
    let mut rules = Vec::new();
    let mut seen_ids: BTreeSet<String> = BTreeSet::new();
    let mut seen_signatures: BTreeSet<String> = BTreeSet::new();

    for (index, spec) in pack.rules.iter().enumerate() {
        let rule_id = if spec.id.trim().is_empty() {
            format!("rule-{}", index + 1)
        } else {
            spec.id.trim().to_string()
        };
        if !seen_ids.insert(rule_id.clone()) {
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                &pack.id,
                Some(&rule_id),
                "id",
                "同一规则包内规则 id 重复，后者被忽略。".to_string(),
            ));
            continue;
        }
        let clause = match compile_rule(&pack.id, &rule_id, spec, issues) {
            Some(clause) => clause,
            None => continue,
        };
        if let Some(key) = functional.get(clause.head.predicate.as_ref()) {
            let required = key.key.max(key.value);
            if required >= clause.head.arity() {
                issues.push(PackIssue::new(
                    IssueSeverity::Error,
                    &pack.id,
                    Some(&rule_id),
                    "functional",
                    format!(
                        "规则头「{}」的参数少于功能谓词「{}」声明的位置 {}。",
                        clause.head.display(),
                        key.predicate,
                        required
                    ),
                ));
                continue;
            }
        }
        if let Some(group) = exclusive.get(clause.head.predicate.as_ref()) {
            if group.key >= clause.head.arity() {
                issues.push(PackIssue::new(
                    IssueSeverity::Error,
                    &pack.id,
                    Some(&rule_id),
                    "exclusive",
                    format!(
                        "规则头「{}」的参数少于互斥组声明的主语位置 {}。",
                        clause.head.display(),
                        group.key
                    ),
                ));
                continue;
            }
        }
        let signature = clause_signature(&clause);
        if !seen_signatures.insert(signature) {
            issues.push(PackIssue::new(
                IssueSeverity::Warning,
                &pack.id,
                Some(&rule_id),
                "rules",
                "规则与同一包内另一条规则重复。".to_string(),
            ));
        }
        rules.push(CompiledRule {
            pack: pack.id.clone(),
            rule_id,
            clause,
        });
    }
    rules
}

fn compile_rule(
    pack: &str,
    rule_id: &str,
    spec: &RuleSpec,
    issues: &mut Vec<PackIssue>,
) -> Option<Clause> {
    let text = if spec.body.is_empty() {
        spec.head.clone()
    } else {
        format!("{} :- {}", spec.head, spec.body.join(", "))
    };
    let parsed = match parse_rule(&text) {
        Ok(parsed) => parsed,
        Err(error) => {
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                pack,
                Some(rule_id),
                "rules",
                format!("规则无法解析：{}", error),
            ));
            return None;
        }
    };
    let mut body = Vec::new();
    for literal in &parsed.body {
        match literal {
            LLiteral::Pos(atom) => body.push(convert_atom(atom)),
            LLiteral::Neg(atom) => {
                issues.push(PackIssue::new(
                    IssueSeverity::Error,
                    pack,
                    Some(rule_id),
                    "rules",
                    format!("规则体不支持否定文字「{}」。", atom.render()),
                ));
                return None;
            }
        }
    }
    let head = convert_atom(&parsed.head);
    let head_vars: BTreeSet<String> = head
        .terms
        .iter()
        .filter_map(|term| match term {
            Term::Var(name) => Some(name.to_string()),
            Term::Const(_) => None,
        })
        .collect();
    let body_vars: BTreeSet<String> = body
        .iter()
        .flat_map(|atom| atom.terms.iter())
        .filter_map(|term| match term {
            Term::Var(name) => Some(name.to_string()),
            Term::Const(_) => None,
        })
        .collect();
    if let Some(missing) = head_vars.difference(&body_vars).next() {
        issues.push(PackIssue::new(
            IssueSeverity::Error,
            pack,
            Some(rule_id),
            "rules",
            format!("规则头变量「?{}」未在规则体中绑定。", missing),
        ));
        return None;
    }
    if !(0.0..=1.0).contains(&spec.confidence) {
        issues.push(PackIssue::new(
            IssueSeverity::Warning,
            pack,
            Some(rule_id),
            "confidence",
            format!("置信度 {} 超出 [0,1]，按边界值使用。", spec.confidence),
        ));
    }
    Some(Clause::new(
        head,
        body,
        spec.confidence,
        &format!("rule_pack:{}/{}", pack, rule_id),
    ))
}

fn convert_atom(atom: &LAtom) -> Atom {
    Atom::new(
        &atom.predicate,
        atom.terms
            .iter()
            .map(|term| match term {
                LTerm::Var(name) => Term::var(name),
                LTerm::Atom(name) => Term::constant(name),
            })
            .collect(),
    )
}

fn clause_signature(clause: &Clause) -> String {
    format!(
        "{}:-{}",
        clause.head.display(),
        clause
            .body
            .iter()
            .map(Atom::display)
            .collect::<Vec<_>>()
            .join("&")
    )
}

fn cross_pack_issues(compiled: &[CompiledPack], raw: &[RulePack]) -> Vec<PackIssue> {
    let mut issues = Vec::new();
    let by_id: BTreeMap<&str, &RulePack> =
        raw.iter().map(|pack| (pack.id.as_str(), pack)).collect();
    let all_ids: BTreeSet<&str> = by_id.keys().copied().collect();

    for pack in raw {
        for required in &pack.requires {
            if required == &pack.id {
                continue;
            }
            match by_id.get(required.as_str()) {
                Some(required_pack) => {
                    if required_pack.layer > pack.layer {
                        issues.push(PackIssue::new(
                            IssueSeverity::Warning,
                            &pack.id,
                            None,
                            "layer",
                            format!(
                                "依赖「{}」的层 {} 高于本包的层 {}，组合顺序可能倒置。",
                                required, required_pack.layer, pack.layer
                            ),
                        ));
                    }
                }
                None => issues.push(PackIssue::new(
                    IssueSeverity::Warning,
                    &pack.id,
                    None,
                    "requires",
                    format!("依赖的规则包「{}」未参与组合。", required),
                )),
            }
        }
        for conflict in &pack.conflicts {
            if conflict != &pack.id && all_ids.contains(conflict.as_str()) {
                issues.push(PackIssue::new(
                    IssueSeverity::Warning,
                    &pack.id,
                    None,
                    "conflicts",
                    format!("声明互斥的规则包「{}」同时参与了组合。", conflict),
                ));
            }
        }
    }

    let mut visiting: BTreeSet<String> = BTreeSet::new();
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<String> = Vec::new();
    for pack in raw {
        detect_cycle(
            &pack.id,
            &by_id,
            &mut visiting,
            &mut visited,
            &mut stack,
            &mut issues,
        );
    }

    let mut functional_contracts: BTreeMap<&str, BTreeMap<(usize, usize), Vec<&str>>> =
        BTreeMap::new();
    for pack in compiled {
        for key in &pack.spec.functional {
            functional_contracts
                .entry(key.predicate.as_str())
                .or_default()
                .entry((key.key, key.value))
                .or_default()
                .push(pack.spec.id.as_str());
        }
    }
    for (predicate, positions) in functional_contracts {
        if positions.len() < 2 {
            continue;
        }
        let described = positions
            .iter()
            .map(|((key, value), packs)| {
                format!("({}, {}) 由「{}」声明", key, value, packs.join("、"))
            })
            .collect::<Vec<_>>()
            .join("；");
        let first_pack = positions.values().next().and_then(|packs| packs.first());
        if let Some(first_pack) = first_pack {
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                first_pack,
                None,
                "functional",
                format!(
                    "功能谓词「{}」在不同规则包中的位置声明不一致：{}。",
                    predicate, described
                ),
            ));
        }
    }

    let mut signatures: BTreeMap<String, (&str, &str)> = BTreeMap::new();
    for pack in compiled {
        for rule in &pack.rules {
            let signature = clause_signature(&rule.clause);
            if let Some((other_pack, other_rule)) = signatures.get(&signature) {
                if *other_pack != pack.spec.id {
                    issues.push(PackIssue::new(
                        IssueSeverity::Warning,
                        &pack.spec.id,
                        Some(&rule.rule_id),
                        "rules",
                        format!(
                            "规则与「{} / {}」重复；组合后可能重复推导。",
                            other_pack, other_rule
                        ),
                    ));
                }
            } else {
                signatures.insert(signature, (pack.spec.id.as_str(), rule.rule_id.as_str()));
            }
        }
    }
    issues
}

fn detect_cycle(
    id: &str,
    by_id: &BTreeMap<&str, &RulePack>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
    stack: &mut Vec<String>,
    issues: &mut Vec<PackIssue>,
) {
    if visited.contains(id) {
        return;
    }
    if !visiting.insert(id.to_string()) {
        if let Some(position) = stack.iter().position(|entry| entry == id) {
            let cycle = stack[position..]
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(id))
                .collect::<Vec<_>>()
                .join(" → ");
            issues.push(PackIssue::new(
                IssueSeverity::Error,
                id,
                None,
                "requires",
                format!("规则包依赖成环：{}。", cycle),
            ));
        }
        return;
    }
    stack.push(id.to_string());
    if let Some(pack) = by_id.get(id) {
        for required in &pack.requires {
            if by_id.contains_key(required.as_str()) {
                detect_cycle(required, by_id, visiting, visited, stack, issues);
            }
        }
    }
    stack.pop();
    visiting.remove(id);
    visited.insert(id.to_string());
}

fn detect_rule_conflicts(
    compiled: &[CompiledPack],
    truncated: &mut bool,
) -> Vec<RuleConflictWitness> {
    let mut rules: Vec<&CompiledRule> =
        compiled.iter().flat_map(|pack| pack.rules.iter()).collect();
    rules.sort_by(|left, right| {
        left.pack
            .cmp(&right.pack)
            .then_with(|| left.rule_id.cmp(&right.rule_id))
            .then_with(|| left.clause.head.display().cmp(&right.clause.head.display()))
    });

    let functional: BTreeMap<(&str, &str), &FunctionalKey> = compiled
        .iter()
        .flat_map(|pack| {
            pack.spec
                .functional
                .iter()
                .map(move |key| ((pack.spec.id.as_str(), key.predicate.as_str()), key))
        })
        .collect();
    let exclusive: Vec<&ExclusiveGroup> = compiled
        .iter()
        .flat_map(|pack| pack.spec.exclusive.iter())
        .collect();

    let mut witnesses: BTreeMap<String, RuleConflictWitness> = BTreeMap::new();
    let mut pairs = 0usize;
    'pairs: for (index, left) in rules.iter().enumerate() {
        for right in rules.iter().skip(index + 1) {
            if left.pack == right.pack && left.rule_id == right.rule_id {
                continue;
            }
            pairs += 1;
            if pairs > MAX_CONFLICT_RULE_PAIRS {
                *truncated = true;
                break 'pairs;
            }
            if left.clause.head.predicate == right.clause.head.predicate {
                let key = functional
                    .get(&(left.pack.as_str(), left.clause.head.predicate.as_ref()))
                    .or_else(|| {
                        functional.get(&(right.pack.as_str(), right.clause.head.predicate.as_ref()))
                    });
                if let Some(key) = key {
                    if let Some(witness) = functional_witness(left, right, key, index) {
                        witnesses.entry(witness.id.clone()).or_insert(witness);
                    }
                    continue;
                }
            }
            for group in &exclusive {
                if !contains_predicate(group, &left.clause.head.predicate)
                    || !contains_predicate(group, &right.clause.head.predicate)
                {
                    continue;
                }
                if left.clause.head.predicate == right.clause.head.predicate {
                    continue;
                }
                if let Some(witness) = exclusive_witness(left, right, group, index) {
                    witnesses.entry(witness.id.clone()).or_insert(witness);
                }
            }
        }
    }
    witnesses.into_values().collect()
}

fn contains_predicate(group: &ExclusiveGroup, predicate: &str) -> bool {
    group.predicates.iter().any(|entry| entry == predicate)
}

fn functional_witness(
    left: &CompiledRule,
    right: &CompiledRule,
    key: &FunctionalKey,
    tag: usize,
) -> Option<RuleConflictWitness> {
    let right_clause = freshen_clause(&right.clause, tag + 1);
    if key.key >= left.clause.head.arity()
        || key.value >= left.clause.head.arity()
        || key.key >= right_clause.head.arity()
        || key.value >= right_clause.head.arity()
    {
        return None;
    }
    let mut subst = Substitution::new();
    if !unify_terms_at(&left.clause.head, &right_clause.head, key.key, &mut subst) {
        return None;
    }
    let left_value = resolved_term(&left.clause.head.terms[key.value], &subst);
    let right_value = resolved_term(&right_clause.head.terms[key.value], &subst);
    let (Some(left_value), Some(right_value)) =
        (constant_of(&left_value), constant_of(&right_value))
    else {
        return None;
    };
    if left_value == right_value {
        return None;
    }
    let subject = subject_display(&left.clause.head.terms[key.key], &subst);
    let status = if left.clause.body.is_empty() && right.clause.body.is_empty() {
        WitnessStatus::Actual
    } else {
        WitnessStatus::Potential
    };
    let id = witness_id(
        RuleConflictKind::Functional,
        left,
        right,
        &subject,
        Some(&left_value),
        Some(&right_value),
    );
    Some(RuleConflictWitness {
        id,
        kind: RuleConflictKind::Functional,
        status,
        left_pack: left.pack.clone(),
        left_rule: left.rule_id.clone(),
        right_pack: right.pack.clone(),
        right_rule: right.rule_id.clone(),
        subject,
        predicate: Some(key.predicate.clone()),
        left_predicate: None,
        right_predicate: None,
        left_value: Some(left_value),
        right_value: Some(right_value),
        left_head: left.clause.head.display(),
        right_head: right.clause.head.display(),
        left_body: left.clause.body.iter().map(Atom::display).collect(),
        right_body: right.clause.body.iter().map(Atom::display).collect(),
        same_pack: left.pack == right.pack,
        preferred_pack: None,
    })
}

fn exclusive_witness(
    left: &CompiledRule,
    right: &CompiledRule,
    group: &ExclusiveGroup,
    tag: usize,
) -> Option<RuleConflictWitness> {
    let right_clause = freshen_clause(&right.clause, tag + 1);
    if group.key >= left.clause.head.arity() || group.key >= right_clause.head.arity() {
        return None;
    }
    let mut subst = Substitution::new();
    if !unify_terms_at(&left.clause.head, &right_clause.head, group.key, &mut subst) {
        return None;
    }
    let subject = subject_display(&left.clause.head.terms[group.key], &subst);
    let status = if left.clause.body.is_empty() && right.clause.body.is_empty() {
        WitnessStatus::Actual
    } else {
        WitnessStatus::Potential
    };
    let id = witness_id(
        RuleConflictKind::Exclusive,
        left,
        right,
        &subject,
        None,
        None,
    );
    Some(RuleConflictWitness {
        id,
        kind: RuleConflictKind::Exclusive,
        status,
        left_pack: left.pack.clone(),
        left_rule: left.rule_id.clone(),
        right_pack: right.pack.clone(),
        right_rule: right.rule_id.clone(),
        subject,
        predicate: None,
        left_predicate: Some(left.clause.head.predicate.to_string()),
        right_predicate: Some(right.clause.head.predicate.to_string()),
        left_value: None,
        right_value: None,
        left_head: left.clause.head.display(),
        right_head: right.clause.head.display(),
        left_body: left.clause.body.iter().map(Atom::display).collect(),
        right_body: right.clause.body.iter().map(Atom::display).collect(),
        same_pack: left.pack == right.pack,
        preferred_pack: None,
    })
}

fn unify_terms_at(left: &Atom, right: &Atom, index: usize, subst: &mut Substitution) -> bool {
    let left_atom = Atom::new("$key", vec![left.terms[index].clone()]);
    let right_atom = Atom::new("$key", vec![right.terms[index].clone()]);
    unify_atoms(&left_atom, &right_atom, subst)
}

fn resolved_term(term: &Term, subst: &Substitution) -> Term {
    let atom = Atom::new("$term", vec![term.clone()]);
    apply_atom(subst, &atom).terms[0].clone()
}

fn constant_of(term: &Term) -> Option<String> {
    match term {
        Term::Const(name) => Some(name.to_string()),
        Term::Var(_) => None,
    }
}

fn subject_display(term: &Term, subst: &Substitution) -> String {
    match resolved_term(term, subst) {
        Term::Const(name) => name.to_string(),
        Term::Var(name) => format!("?{}", name),
    }
}

fn witness_id(
    kind: RuleConflictKind,
    left: &CompiledRule,
    right: &CompiledRule,
    subject: &str,
    left_value: Option<&str>,
    right_value: Option<&str>,
) -> String {
    let seed = format!(
        "{:?}|{}|{}|{}|{}|{}|{}|{}",
        kind,
        left.pack,
        left.rule_id,
        right.pack,
        right.rule_id,
        subject,
        left_value.unwrap_or(""),
        right_value.unwrap_or("")
    );
    format!("rp-{}", &fnv1a_hex(seed.as_bytes())[..12])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(id: &str) -> RulePack {
        RulePack::new(id)
    }

    #[test]
    fn compiles_chinese_rules_and_rejects_unsafe_heads() {
        let safe =
            pack("safe").with_rule(RuleSpec::new("age", "年龄(X, 38)", &["出生年(X, 1986)"]));
        let program = compile_packs(&[safe]);
        assert_eq!(program.report.rules, 1);
        assert!(!program.report.has_errors(), "{:?}", program.report.issues);
        assert_eq!(program.clauses().len(), 1);

        let unsafe_pack = pack("unsafe").with_rule(RuleSpec::new("bad", "年龄(X, 38)", &[]));
        let program = compile_packs(&[unsafe_pack]);
        assert!(program.report.has_errors());
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("未在规则体中绑定")));
    }

    #[test]
    fn functional_contract_reports_cross_pack_divergence_with_witness() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new(
                "age-left",
                "年龄(X, 38)",
                &["出生年(X, 1986)"],
            ));
        let right = pack("right")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new(
                "age-right",
                "年龄(X, 39)",
                &["出生年(X, 1990)"],
            ));
        let program = compile_packs(&[left, right]);
        assert!(!program.report.has_errors(), "{:?}", program.report.issues);
        assert_eq!(program.report.conflicts.len(), 1);
        let conflict = &program.report.conflicts[0];
        assert_eq!(conflict.kind, RuleConflictKind::Functional);
        assert_eq!(conflict.status, WitnessStatus::Potential);
        assert_eq!(conflict.left_pack, "left");
        assert_eq!(conflict.right_pack, "right");
        assert_eq!(conflict.left_value.as_deref(), Some("38"));
        assert_eq!(conflict.right_value.as_deref(), Some("39"));
        assert!(!conflict.same_pack);
        let rendered = program.report.render_zh();
        assert!(rendered.contains("功能谓词分歧"));
        assert!(rendered.contains("38"));
        assert!(rendered.contains("39"));
        assert!(rendered.contains("你希望采用哪一个"));
    }

    #[test]
    fn same_value_or_bound_value_is_not_a_conflict() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("a", "年龄(X, 38)", &["出生年(X, 1986)"]));
        let same = pack("same")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("b", "年龄(X, 38)", &["出生年(X, 1990)"]));
        let bound = pack("bound")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("c", "年龄(X, Y)", &["年龄值(X, Y)"]));
        let program = compile_packs(&[left, same, bound]);
        assert!(
            program.report.conflicts.is_empty(),
            "{:?}",
            program.report.conflicts
        );
    }

    #[test]
    fn exclusive_group_reports_potential_conflict_and_can_be_adjudicated() {
        let left = pack("living")
            .with_exclusive(ExclusiveGroup::new(&["在世", "已故"], 0))
            .with_rule(RuleSpec::new("living-rule", "在世(X)", &["人物(X)"]));
        let right = pack("dead")
            .with_exclusive(ExclusiveGroup::new(&["在世", "已故"], 0))
            .with_rule(RuleSpec::new("dead-rule", "已故(X)", &["人物(X)"]));
        let mut program = compile_packs(&[left, right]);
        assert_eq!(program.report.conflicts.len(), 1);
        let conflict = &program.report.conflicts[0];
        assert_eq!(conflict.kind, RuleConflictKind::Exclusive);
        let predicates = BTreeSet::from([
            conflict.left_predicate.as_deref().unwrap_or(""),
            conflict.right_predicate.as_deref().unwrap_or(""),
        ]);
        assert!(predicates.contains("在世"));
        assert!(predicates.contains("已故"));
        let id = conflict.id.clone();
        assert!(program.report.adjudicate(&id, "living"));
        assert_eq!(
            program.report.conflicts[0].status,
            WitnessStatus::Adjudicated
        );
        assert_eq!(
            program.report.conflicts[0].preferred_pack.as_deref(),
            Some("living")
        );
        assert!(program.report.render_zh().contains("已裁决"));
    }

    #[test]
    fn dependency_cycle_and_layer_inversion_are_reported() {
        let mut base = pack("base").with_layer(3);
        base.requires.push("leaf".to_string());
        base.rules
            .push(RuleSpec::new("base-rule", "基础(X)", &["事实(X)"]));
        let mut leaf = pack("leaf").with_layer(1);
        leaf.requires.push("base".to_string());
        leaf.rules
            .push(RuleSpec::new("leaf-rule", "叶子(X)", &["事实(X)"]));
        let program = compile_packs(&[base, leaf]);
        assert!(program.report.has_errors());
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("依赖成环")));
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("层")));
    }

    #[test]
    fn actual_ground_rule_conflict_is_marked_witnessed() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("a", "年龄(张三, 38)", &[]));
        let right = pack("right")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("b", "年龄(张三, 39)", &[]));
        let program = compile_packs(&[left, right]);
        assert_eq!(program.report.conflicts.len(), 1);
        assert_eq!(program.report.conflicts[0].status, WitnessStatus::Actual);
        assert_eq!(program.report.conflicts[0].subject, "张三");
    }

    #[test]
    fn duplicate_rules_are_warned_and_contract_positions_are_checked() {
        let mut pack = pack("dup")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("one", "年龄(X, 38)", &["出生年(X, 1986)"]))
            .with_rule(RuleSpec::new("two", "年龄(X, 38)", &["出生年(X, 1986)"]));
        pack.rules
            .push(RuleSpec::new("bad", "年龄(X)", &["出生年(X, 1986)"]));
        let program = compile_packs(&[pack]);
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("重复")));
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("功能谓词")));
    }

    #[test]
    fn cross_pack_contract_position_mismatch_is_an_error() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new(
                "a",
                "年龄(X, 38, 备注)",
                &["出生年(X, 1986)"],
            ));
        let right = pack("right")
            .with_functional(FunctionalKey::new("年龄", 0, 2))
            .with_rule(RuleSpec::new(
                "b",
                "年龄(X, 备注, 39)",
                &["出生年(X, 1990)"],
            ));
        let program = compile_packs(&[left, right]);
        assert_eq!(program.report.packs, 2);
        assert!(program
            .report
            .issues
            .iter()
            .any(|issue| issue.message.contains("位置声明不一致")));
        assert!(program.report.has_errors());
    }

    #[test]
    fn adjudication_suppresses_the_losing_rule() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("a", "年龄(X, 38)", &["出生年(X, 1986)"]));
        let right = pack("right")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("b", "年龄(X, 39)", &["出生年(X, 1990)"]));
        let mut program = compile_packs(&[left, right]);
        let id = program.report.conflicts[0].id.clone();
        assert!(program.report.adjudicate(&id, "right"));
        program.apply_adjudications();
        assert_eq!(program.report.rules, 1);
        let clauses = program.clauses();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].pack, "right");
    }

    #[test]
    fn report_is_deterministic() {
        let left = pack("left")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("a", "年龄(X, 38)", &["出生年(X, 1986)"]));
        let right = pack("right")
            .with_functional(FunctionalKey::new("年龄", 0, 1))
            .with_rule(RuleSpec::new("b", "年龄(X, 39)", &["出生年(X, 1990)"]));
        let first = compile_packs(&[left.clone(), right.clone()]);
        let second = compile_packs(&[right, left]);
        assert_eq!(first.report.conflicts[0].id, second.report.conflicts[0].id);
        assert_eq!(first.report.to_json(), second.report.to_json());
    }
}

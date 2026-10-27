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
        toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析")
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

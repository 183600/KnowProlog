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
        "规则包" | "规则包诊断" | "规则冲突" | "rule packs" | "rulepack" | "rule conflicts"
    ) {
        return Some(ParsedPluginCommand::Rules);
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
    fn sentence_punctuation_and_polite_wrappers_are_ignored_for_matching() {
        let mut on = manifest("zh.film", PluginMode::On);
        on.contributions.push(rewrite_contribution(
            "film.directed",
            "{entity}是谁导演的",
            "{entity}的导演是谁",
        ));
        let registry = PluginRegistry::from_manifests(vec![on]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let outcome = runtime.rewrite("阿凡达是谁导演的？");
        assert_eq!(outcome.text, "阿凡达的导演是谁");
        assert_eq!(outcome.trace.input, "阿凡达是谁导演的？");

        let polite = runtime.rewrite("请问阿凡达是谁导演的呢？");
        assert_eq!(polite.text, "阿凡达的导演是谁");

        let untouched = runtime.rewrite("今天天气怎么样？");
        assert_eq!(untouched.text, "今天天气怎么样？");
    }

    #[test]
    fn recipe_output_is_terminal_for_rewrite_chain() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(recipe(
            "film.actor_works",
            "{entity}演过什么电影",
            "{entity}主演过哪些作品",
            "film.actor.template",
        ));
        film.contributions.push(template(
            "film.actor.template",
            "「{entity}」主演过的作品：{answer}。",
        ));
        film.contributions.push(rewrite_contribution(
            "film.actor_rewrite",
            "{entity}演过哪些作品",
            "{entity}主演过哪些作品",
        ));
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let outcome = runtime.rewrite("萨姆·沃辛顿演过什么电影");
        assert_eq!(outcome.text, "萨姆·沃辛顿主演过哪些作品");
        assert_eq!(
            outcome
                .trace
                .matches
                .iter()
                .filter(|entry| entry.applied)
                .count(),
            1,
            "配方输出已是规范问句，不应被后续改写再次拼接"
        );
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

    /// 空词面绝不能进单遍替换：`starts_with("")` 恒真且 `len()==0` 不推进
    /// 游标，一旦混入就会原地打转（进程挂死 + 无限分配）。
    #[test]
    fn apply_lexicon_ignores_empty_surfaces_and_always_terminates() {
        let pairs = vec![
            (
                String::new(),
                "电影".to_string(),
                "zh.broken".to_string(),
                "broken.lexicon".to_string(),
                PluginMode::On,
            ),
            (
                String::new(),
                "人物".to_string(),
                "zh.broken".to_string(),
                "broken.lexicon".to_string(),
                PluginMode::Observe,
            ),
        ];
        let (text, matches) = apply_lexicon("阿凡达是谁导演的", &pairs);
        assert_eq!(text, "阿凡达是谁导演的");
        assert!(
            matches.is_empty(),
            "空词面不应产生命中记录：{:?}",
            matches
        );

        // 空词面与真实词面混排时，真实词面仍要生效。
        let mixed = vec![
            (
                "影片".to_string(),
                "电影".to_string(),
                "zh.film".to_string(),
                "film.lexicon".to_string(),
                PluginMode::On,
            ),
            (
                String::new(),
                "垃圾".to_string(),
                "zh.broken".to_string(),
                "broken.lexicon".to_string(),
                PluginMode::On,
            ),
        ];
        let (text, matches) = apply_lexicon("这部影片很好", &mixed);
        assert_eq!(text, "这部电影很好");
        assert_eq!(matches.len(), 1);
    }

    /// 本体空别名/自指别名只记警告不废包：它们必须真的被「已忽略」，
    /// 既不进词法替换（否则挂死），也不产生自指改写噪声。
    #[test]
    fn degenerate_ontology_aliases_are_dropped_from_lexicon_pass() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec![String::new(), "导演".to_string(), "执导".to_string()],
                ..OntologyPredicate::default()
            }],
            types: vec![OntologyType {
                name: "电影".to_string(),
                aliases: vec![String::new(), "影片".to_string()],
                ..OntologyType::default()
            }],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let outcome = runtime.rewrite("这部影片是谁执导的");
        assert_eq!(
            outcome.text, "这部电影是谁导演的",
            "有效别名照常生效，空/自指别名被忽略"
        );
        let mut details: Vec<&str> = outcome
            .trace
            .matches
            .iter()
            .filter(|entry| entry.kind == "lexicon")
            .map(|entry| entry.detail.as_str())
            .collect();
        details.sort();
        assert_eq!(
            details,
            vec!["影片 → 电影", "执导 → 导演"],
            "空别名与自指别名都不得产生命中"
        );
    }

    #[test]
    fn ontology_aliases_become_structural_axioms_for_stored_data() {
        let mut film = manifest("zh.film", PluginMode::On);
        film.contributions.push(Contribution::Ontology {
            id: "film.ontology".to_string(),
            description: String::new(),
            predicates: vec![OntologyPredicate {
                name: "导演".to_string(),
                aliases: vec!["director".to_string()],
                ..OntologyPredicate::default()
            }],
            types: vec![OntologyType {
                name: "电影".to_string(),
                aliases: vec!["影片".to_string()],
                ..OntologyType::default()
            }],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let program = runtime.ontology_program();
        assert!(
            program.lessons.iter().any(|lesson| matches!(
                lesson,
                OntologyLesson::Subclass { sub, sup, .. }
                    if sub == "影片" && sup == "电影"
            )),
            "类型别名必须让已落库的别名类型满足规范类型：{:?}",
            program.lessons
        );
        assert!(
            program.lessons.iter().any(|lesson| matches!(
                lesson,
                OntologyLesson::Subproperty { sub, sup, .. }
                    if sub == "director" && sup == "导演"
            )),
            "谓词别名必须让已落库的别名关系满足规范关系：{:?}",
            program.lessons
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
            quantifiers: vec!["本片".to_string()],
            role_words: vec!["导演".to_string()],
            surnames: vec!["卡梅隆".to_string()],
        });
        let registry = PluginRegistry::from_manifests(vec![film]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        assert!(runtime.reference_lexicon().is_empty());
        assert!(runtime.is_context_reference("他"));
        assert!(!runtime.is_context_reference("本片"));
        runtime.set_mode("zh.film", PluginMode::On).unwrap();
        assert!(runtime.is_context_reference("本片"));
        assert!(runtime.is_context_reference("本片导演"));
        assert!(!runtime.is_context_reference("导演"));
        assert!(!runtime.is_context_reference("本片演员"));
        assert_eq!(
            runtime.reference_lexicon(),
            vec![
                ("卡梅隆".to_string(), "nr".to_string()),
                ("导演".to_string(), "n".to_string()),
                ("本片".to_string(), "r".to_string()),
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
        assert_eq!(
            program.len(),
            8,
            "子类2 + 互斥1 + 定义域1 + 值域1 + 逆1 + 传递1 + 谓词别名1"
        );
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
            toml::from_str(include_str!("../../plugins/zh.film.toml")).expect("参考插件应可解析");
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
    #[test]
    fn rules_contribution_compiles_and_reports_composition_conflicts() {
        let mut left = manifest("zh.rules.left", PluginMode::On);
        left.contributions.push(Contribution::Rules {
            id: "left.age".to_string(),
            description: "左年龄规则".to_string(),
            layer: 1,
            activation: Vec::new(),
            requires: Vec::new(),
            conflicts: Vec::new(),
            functional: vec![FunctionalKey::new("年龄", 0, 1)],
            exclusive: Vec::new(),
            rules: vec![RuleSpec::new(
                "age-left",
                "年龄(X, 38)",
                &["出生年(X, 1986)"],
            )],
        });
        let mut right = manifest("zh.rules.right", PluginMode::On);
        right.contributions.push(Contribution::Rules {
            id: "right.age".to_string(),
            description: "右年龄规则".to_string(),
            layer: 1,
            activation: Vec::new(),
            requires: Vec::new(),
            conflicts: Vec::new(),
            functional: vec![FunctionalKey::new("年龄", 0, 1)],
            exclusive: Vec::new(),
            rules: vec![RuleSpec::new(
                "age-right",
                "年龄(X, 39)",
                &["出生年(X, 1990)"],
            )],
        });
        let registry = PluginRegistry::from_manifests(vec![left, right]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let mut runtime = PluginRuntime::with_settings(
            registry,
            &PluginSettings {
                enabled: vec!["zh.rules.left".to_string(), "zh.rules.right".to_string()],
                ..PluginSettings::default()
            },
        );
        let report = runtime.rule_report();
        assert_eq!(report.packs, 2);
        assert_eq!(report.rules, 2);
        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(runtime.rule_clauses().len(), 2);
        assert!(runtime.render_report().contains("规则包"));
        let outcome = runtime.handle_command("规则包").unwrap();
        assert!(outcome.success);
        assert!(outcome.text.contains("功能谓词分歧"));
        assert!(outcome.text.contains("38"));
        assert!(outcome.text.contains("39"));
    }

}

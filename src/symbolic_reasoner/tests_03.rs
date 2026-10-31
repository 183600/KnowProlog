    /// 别名公理让历史数据无需重写：谓词别名沿子关系归一，类型别名沿子类归一。
    #[test]
    fn recipe_lookup_reaches_alias_predicates_and_types() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "影片".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "director".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "影片".into(),
                sup: "电影".into(),
                original: "影片是电影的别名".into(),
            },
            OntologyLesson::Subproperty {
                sub: "director".into(),
                sup: "导演".into(),
                original: "director是导演的别名".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let inverse = reasoner
            .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("别名谓词与别名类型都应被规范词面命中");
        assert_eq!(inverse.values, vec!["阿凡达".to_string()]);
        assert!(
            inverse
                .evidence
                .iter()
                .any(|line| line.contains("director ⊑ 导演")),
            "依据应说明别名归一：{:?}",
            inverse.evidence
        );
    }

    /// 运行态启用插件时增量合并本体：首次返回新增数，重复应用为零。
    #[test]
    fn applying_plugin_runtime_is_incremental_and_idempotent() {
        use crate::plugin::{PluginManifest, PluginRegistry, PluginRuntime, PluginSettings};

        let manifest: PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "ontology"
id = "film.ontology"
predicates = [
  { name = "导演", aliases = ["director"], inverse = "执导", domain = ["电影"], range = ["人物"] },
]
types = [
  { name = "电影", aliases = ["影片"] },
]
"#,
        )
        .expect("测试清单应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        let runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());

        let first = reasoner.apply_plugin_runtime(&runtime);
        assert_eq!(first.aliases, 1, "谓词别名 director");
        assert_eq!(first.inverse_surfaces, 1, "互逆词面 执导");
        assert_eq!(first.lessons, 5, "子类1 + 别名1 + 定义域1 + 值域1 + 逆1");

        assert_eq!(
            reasoner.apply_plugin_runtime(&runtime),
            PluginOntologySync::default(),
            "同一运行态重复应用必须无新增"
        );
    }

    #[test]
    fn plugin_rule_pack_joins_resolver_and_is_idempotent() {
        use crate::plugin::{PluginManifest, PluginRegistry, PluginRuntime, PluginSettings};
        use crate::resolution_engine::{Atom, Term};

        let manifest: PluginManifest = toml::from_str(
            r#"
id = "test.rules"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "rules"
id = "test.age"
functional = [{ predicate = "年龄", key = 0, value = 1 }]
rules = [
  { id = "age-1986", head = "年龄(X, 38)", body = ["出生年(X, 1986)"] },
]
"#,
        )
        .expect("规则包清单应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        assert!(!registry.has_errors(), "{}", registry.render_issues());
        let runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".into(),
            rel_type: "出生年".into(),
            to: "1986".into(),
        }]);

        let first = reasoner.apply_plugin_runtime(&runtime);
        assert_eq!(first.rules, 1);
        let goal = Atom::pattern("年龄", Term::constant("张三"), Term::var("Age"));
        let solutions = reasoner.resolver().solve(&[goal], &["Age".to_string()]);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].bindings.get("Age").map(String::as_str), Some("38"));
        assert_eq!(
            reasoner.apply_plugin_runtime(&runtime),
            PluginOntologySync::default(),
            "同一规则包重复应用必须无新增"
        );
    }

    #[test]
    fn rule_adjudication_removes_losing_clause_from_resolver() {
        use crate::plugin::{PluginManifest, PluginRegistry, PluginRuntime, PluginSettings};
        use crate::resolution_engine::{Atom, Term};

        let left: PluginManifest = toml::from_str(
            r#"
id = "zh.rules.left"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "rules"
id = "left.age"
functional = [{ predicate = "年龄", key = 0, value = 1 }]
rules = [
  { id = "age-left", head = "年龄(X, 38)", body = ["出生年(X, 1986)"] },
]
"#,
        )
        .expect("左侧规则包应可解析");
        let right: PluginManifest = toml::from_str(
            r#"
id = "zh.rules.right"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "rules"
id = "right.age"
functional = [{ predicate = "年龄", key = 0, value = 1 }]
rules = [
  { id = "age-right", head = "年龄(X, 39)", body = ["出生年(X, 1990)"] },
]
"#,
        )
        .expect("右侧规则包应可解析");
        let registry = PluginRegistry::from_manifests(vec![left, right]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "出生年".into(),
                to: "1986".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "出生年".into(),
                to: "1990".into(),
            },
        ]);

        let first = reasoner.apply_plugin_runtime(&runtime);
        assert_eq!(first.rules, 2);
        let goal = Atom::pattern("年龄", Term::constant("张三"), Term::var("Age"));
        let ages = reasoner
            .resolver()
            .solve(&[goal.clone()], &["Age".to_string()])
            .into_iter()
            .filter_map(|solution| solution.bindings.get("Age").cloned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            ages,
            BTreeSet::from(["38".to_string(), "39".to_string()])
        );

        let outcome = runtime
            .handle_command("规则裁决 1=right.age")
            .expect("裁决命令应被识别");
        assert!(outcome.success, "{}", outcome.text);
        let second = reasoner.apply_plugin_runtime(&runtime);
        assert_eq!(second.rules, 0);
        assert_eq!(second.rules_removed, 1);
        let ages = reasoner
            .resolver()
            .solve(&[goal], &["Age".to_string()])
            .into_iter()
            .filter_map(|solution| solution.bindings.get("Age").cloned())
            .collect::<BTreeSet<_>>();
        assert_eq!(ages, BTreeSet::from(["39".to_string()]));
    }

    /// 配置本体并入单一本体：注入幂等、与教学公理同簿记、
    /// 随快照/重建/分叉一并保留（配置不再是第二套本体）。
    #[test]
    fn test_configure_ontology_program_merges_into_single_program() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let types = TypeSystemConfig {
            subclasses: vec![TypeSubclassConfig {
                sub: "学生".into(),
                sup: "人物".into(),
            }],
            ..TypeSystemConfig::default()
        };
        let profiles = vec![PredicateProfileConfig {
            name: "师傅".into(),
            transitive: true,
            domain: vec!["人物".into()],
            ..PredicateProfileConfig::default()
        }];
        let program = OntologyProgram::from_config(&types, &profiles);

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert_eq!(
            reasoner.configure_ontology_program(&program),
            3,
            "子类1 + 定义域1 + 传递1"
        );
        assert_eq!(
            reasoner.configure_ontology_program(&program),
            0,
            "重复注入必须按整句去重"
        );
        reasoner
            .teach_ontology("学生是一种人物")
            .expect("教学同一条公理应成功");
        assert_eq!(
            reasoner.taught_ontology_lessons().len(),
            3,
            "配置与教学共用一份公理簿（同原句去重）"
        );

        // 配置本体随快照走：验证 = 反序列化
        let snapshot = reasoner.dialogue_snapshot();
        let capsule = snapshot.ontology.clone().expect("配置本体应随快照携带");
        capsule.verify().expect("胶囊应通过校验");
        assert_eq!(capsule.lessons.len(), 3);

        let mut fresh = SymbolicReasoner::new(test_config(), HashMap::new());
        fresh
            .restore_dialogue_snapshot(&snapshot)
            .expect("配置本体应可恢复");
        assert_eq!(
            fresh.taught_ontology_lessons().len(),
            3,
            "快照恢复后公理全在"
        );

        fresh.rebuild_from_graph(&master_facts());
        assert_eq!(
            fresh.taught_ontology_lessons().len(),
            3,
            "重建图谱后配置本体不得丢"
        );

        let fork = fresh.fork_session();
        assert_eq!(
            fork.taught_ontology_lessons().len(),
            3,
            "会话分叉保留单一本体"
        );
    }

    /// 单一本体 → 检索类型层级：教学的子类/互斥公理进入稠密检索的
    /// 类型约束（RDFS 展开；互斥是唯一硬剪证据）。
    #[test]
    fn test_taught_ontology_expands_retrieval_type_constraints() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "苏格拉底".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "雅典".to_string(),
                entity_type: "地点".to_string(),
                properties: HashMap::new(),
            },
        ]);
        reasoner.teach_ontology("人是一种动物").expect("可教学");
        reasoner.teach_ontology("地点和动物互斥").expect("可教学");

        let constraints = SearchConstraints {
            entity_type: Some("动物".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("", 8, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "苏格拉底"),
            "人 ⊑ 动物：沿教学子类边应命中类型约束：{:?}",
            search.hits
        );
        assert!(
            search.hits.iter().all(|hit| hit.subject != "雅典"),
            "互斥证据必须硬剪：{:?}",
            search.hits
        );
    }

    /// 配置态子类（config.toml 的 type_system）经单一本体进入检索类型层级。
    #[test]
    fn test_configured_subclass_expands_retrieval_type_constraints() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "学生".to_string(),
            properties: HashMap::new(),
        }]);
        let program = OntologyProgram::from_config(
            &TypeSystemConfig {
                subclasses: vec![TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[],
        );
        assert!(reasoner.configure_ontology_program(&program) > 0);
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("", 8, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "张三"),
            "配置子类（学生 ⊑ 人物）应展开类型约束：{:?}",
            search.hits
        );
    }

    #[test]
    fn test_check_statement_entailed_contradicted_unknown() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let entailed = reasoner.check_statement("张三的师傅是李四");
        assert_eq!(entailed.len(), 1);
        assert_eq!(entailed[0].verdict, Verdict::Entailed);
        assert_eq!(
            entailed[0].proof[0].kind,
            semantic_entailment::ProofKind::DirectFact
        );

        // 「师傅」是多值谓词：图谱里已有王五，并不能排除张三也是师傅，
        // 因此不再按功能谓词的封闭世界规则误判为矛盾。
        let multi_value = reasoner.check_statement("李四的师傅是张三");
        assert_eq!(multi_value[0].verdict, Verdict::Unknown);
        assert!(
            multi_value[0].reason.contains("多值") && multi_value[0].reason.contains("王五"),
            "实际原因：{}",
            multi_value[0].reason
        );

        let unknown = reasoner.check_statement("钱七的师傅是李四");
        assert_eq!(unknown[0].verdict, Verdict::Unknown);
        assert!(
            unknown[0].reason.contains("尚未收录"),
            "实际原因：{}",
            unknown[0].reason
        );

        assert!(reasoner.check_statement("张三的师傅是谁").is_empty());
    }

    #[test]
    fn test_check_statement_functional_predicate_still_contradicts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let contradicted = reasoner.check_statement("姚明的身高是180cm");
        assert_eq!(contradicted.len(), 1);
        assert_eq!(
            contradicted[0].verdict,
            Verdict::Contradicted,
            "功能谓词的不同取值仍应判矛盾：{:?}",
            contradicted[0]
        );
        assert!(contradicted[0]
            .counter_evidence
            .iter()
            .any(|evidence| evidence.contains("226cm")));

        let negative = reasoner.check_statement("姚明的身高不是180cm");
        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].verdict, Verdict::Entailed);
        assert_eq!(
            negative[0].proof[0].kind,
            semantic_entailment::ProofKind::ClosedWorld
        );
    }

    #[test]
    fn test_check_statement_sees_session_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.assert_nl_fact("张三的徒弟是钱七");
        assert!(taught.contains("已记住"), "{}", taught);
        let results = reasoner.check_statement("张三的徒弟是钱七");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].verdict,
            Verdict::Entailed,
            "会话层事实应立即通过蕴含判定：{:?}",
            results[0]
        );
        assert_eq!(
            results[0].proof[0].kind,
            semantic_entailment::ProofKind::DirectFact
        );
        // 反向对照：未教过的实体仍应为未知，而不是被会话层整体放行。
        let unknown = reasoner.check_statement("钱八的徒弟是钱七");
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].verdict, Verdict::Unknown);
    }

    #[test]
    fn test_check_statement_recovers_typo_via_linker() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let results = reasoner.check_statement("张山的师傅是李四");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verdict, Verdict::Entailed);
        assert!(
            results[0]
                .proposition
                .notes
                .iter()
                .any(|note| note.contains("模糊链接") && note.contains("张三")),
            "解析证据应记录容错链接：{:?}",
            results[0].proposition.notes
        );
    }

    #[test]
    fn test_entailment_report_renders_logical_form_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.entailment_report("张三的师傅是李四");
        assert!(report.contains("逻辑形式：relation(张三, 师傅, 李四)"));
        assert!(report.contains("蕴含"));
        assert!(report.contains("直接事实"));

        let failure = reasoner.entailment_report("今天天气如何");
        assert!(failure.contains("形式化失败"));
    }

    #[test]
    fn test_generate_about_appends_entailment_self_check() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("生成自检"), "应附蕴含自检：{}", profile);
        assert!(profile.contains("零幻觉") || profile.contains("未通过"));
        assert_eq!(
            reasoner
                .context()
                .generation_state()
                .map(|s| s.entity.as_str()),
            Some("张三")
        );
    }

    #[test]
    fn test_generation_audit_drops_unsupported_and_unparsed_sentences() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let audit = reasoner.audit_generation(
            "张三的师傅是李四。\n张三的师傅是赵六。\n今天天气不错。",
            &[],
        );
        assert!(audit.text.contains("张三的师傅是李四"), "{}", audit.text);
        assert!(
            !audit.text.contains("张三的师傅是赵六"),
            "与图谱矛盾的句子必须被丢弃：{}",
            audit.text
        );
        assert!(
            !audit.text.contains("今天天气不错"),
            "无法形式化的句子不得冒充事实：{}",
            audit.text
        );
        let note = audit.note.expect("应有审计回执");
        assert!(note.contains("1/3"), "{}", note);
        assert!(note.contains("未通过 2 句"), "{}", note);
        assert!(
            note.contains("张三的师傅是赵六") && note.contains("今天天气不错"),
            "{}",
            note
        );
    }

    #[test]
    fn test_generation_audit_accepts_trusted_unparsed_line() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let line = "张三的师傅是李四。";
        let audit = reasoner.audit_generation(line, &[line.to_string()]);
        assert_eq!(audit.text, line);
        assert!(audit.note.unwrap().contains("零幻觉"));
    }

    #[test]
    fn test_generation_self_check_with_bare_english_predicates() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "张三".to_string()),
                    ("age".to_string(), "25".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "李四".to_string()),
                    ("age".to_string(), "30".to_string()),
                ]),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "knows".to_string(),
                to: "李四".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let profile = reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(profile.contains("张三认识李四"), "{}", profile);
        assert!(
            profile.contains("零幻觉"),
            "裸谓词生成应通过自检：{}",
            profile
        );
    }

    /// 英文属性键存图时，「证明/为什么」通道也必须按中文词面找到同一事实，
    /// 不能出现「回答有值、推导却说无解」的自相矛盾。
    #[test]
    fn test_derivation_proves_chinese_label_of_english_attribute() {
        let facts = vec![PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "张三".to_string()),
                ("age".to_string(), "25".to_string()),
            ]),
        }];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let derivation = reasoner
            .answer_derivation("张三的年龄是多少", Some("25"))
            .expect("年龄问句应能形式化");
        assert!(derivation.is_proved(), "{:?}", derivation);
        assert!(
            derivation
                .steps
                .iter()
                .any(|step| step.canonical.contains("age")),
            "证明应引用图谱里的 age 事实：{:?}",
            derivation
        );
    }

    /// 英文属性键（JSON-LD 常见）应能按中文词面直接检索，且中文键仍优先。
    #[test]
    fn test_english_attribute_keys_answer_chinese_labels() {
        let facts = vec![PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "张三".to_string()),
                ("age".to_string(), "25".to_string()),
                ("height".to_string(), "180".to_string()),
            ]),
        }];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let age = reasoner
            .answer("张三的年龄是多少")
            .expect("中文标签应命中 age");
        assert!(age.contains("25"), "{}", age);
        let height = reasoner
            .answer("张三的身高是多少")
            .expect("中文标签应命中 height");
        assert!(height.contains("180"), "{}", height);
        let english = reasoner.answer("张三的age是多少").expect("英文键也应可查");
        assert!(english.contains("25"), "{}", english);
    }

    /// literal 属性（不进关系三元组）也必须支持「谁的X是Y」反查主体，
    /// 且只回属性值真正相等的那个实体；英文属性键同样按中文词面反查。
    #[test]
    fn test_find_subject_by_attribute_value() {
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("身高".to_string(), "226".to_string()),
                    ("国籍".to_string(), "中国".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("身高".to_string(), "190".to_string())]),
            },
            PrologFact::Entity {
                name: "易建联".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("身高".to_string(), "213".to_string())]),
            },
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("height".to_string(), "180".to_string())]),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let hit = reasoner
            .answer("谁的身高是226")
            .expect("属性反查应给出主体");
        assert!(hit.contains("姚明"), "{}", hit);
        assert!(!hit.contains("叶莉"), "{}", hit);

        let chinese_key = reasoner
            .answer("谁的国籍是中国")
            .expect("文本型属性同样可反查");
        assert!(chinese_key.contains("姚明"), "{}", chinese_key);

        let predicate_first = reasoner
            .answer("身高是226的人是谁")
            .expect("谓词在前的同义问法也应反查主体");
        assert!(predicate_first.contains("姚明"), "{}", predicate_first);
        assert!(!predicate_first.contains("叶莉"), "{}", predicate_first);

        let nationality = reasoner
            .answer("姚明是哪国人")
            .expect("国籍问法应落到国籍属性");
        assert!(nationality.contains("中国"), "{}", nationality);

        let except = reasoner
            .answer("除了姚明还有谁身高是213")
            .expect("带排除项的主体反查应可回答");
        assert!(except.contains("易建联"), "{}", except);
        assert!(
            !except.lines().next().unwrap_or_default().contains("姚明"),
            "排除项不得出现在答案主体中：{}",
            except
        );
        assert!(except.contains("已排除：姚明"), "{}", except);

        let english_key = reasoner
            .answer("谁的身高是180")
            .expect("英文属性键按中文词面反查");
        assert!(english_key.contains("张三"), "{}", english_key);

        assert!(
            reasoner.answer("谁的身高是999").is_none(),
            "没有该取值时不应编造主体"
        );
    }

    /// 批量对齐入口（交互与服务共用）：别名/换向词面即时参与回答，
    /// 实体持久别名与规范名同权进入链接；退化与重复条目不计入生效数。
    #[test]
    fn test_batch_alignment_entry_points() {
        let facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let applied = reasoner.align_predicate_pairs(&[
            ("老师".to_string(), "师傅".to_string(), false),
            ("徒弟".to_string(), "师傅".to_string(), true),
            ("老师".to_string(), "老师".to_string(), false),
            ("  ".to_string(), "师傅".to_string(), false),
        ]);
        assert_eq!(applied, 2, "自指与空词面不计生效");

        let forward = reasoner
            .answer("张三的老师是谁")
            .expect("正向别名应命中师傅事实");
        assert!(forward.contains("李四"), "{}", forward);

        let inverse = reasoner
            .answer("李四的徒弟是谁")
            .expect("换向词面应按反方向检索");
        assert!(inverse.contains("张三"), "{}", inverse);

        let entities = reasoner.align_entity_aliases(&[
            ("明哥".to_string(), "姚明".to_string()),
            ("明哥".to_string(), "明哥".to_string()),
        ]);
        assert_eq!(entities, 1, "自指别名不计生效");
        assert_eq!(
            reasoner.best_linked_entity("明哥").as_deref(),
            Some("姚明"),
            "持久别名应与规范名同权参与链接"
        );
        let alias_answer = reasoner
            .answer("明哥的身高是多少")
            .expect("已确认的实体别名应能直接作答");
        assert!(alias_answer.contains("226"), "{}", alias_answer);
    }

    #[test]
    fn test_continue_generation_does_not_repeat_after_insertion() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        let first_body = first
            .lines()
            .find(|line| line.trim().ends_with('。'))
            .expect("第一页应有一句事实")
            .trim()
            .to_string();
        let first_sentence: String = first_body
            .split_once('，')
            .map(|(_, sentence)| sentence.to_string())
            .unwrap_or(first_body);

        // 新事实插入后，旧实现按 offset 取句会漂移，续说可能重复第一句。
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "朋友".to_string(),
            to: "赵六".to_string(),
        }]);
        let second = reasoner.continue_generation().expect("应继续生成");
        assert!(second.contains("接着，"), "{}", second);
        assert!(
            !second.contains(first_sentence.trim_end_matches('。')),
            "已说过的句子不应重复：第一页={}；第二页={}",
            first,
            second
        );
        let state = reasoner.context().generation_state().unwrap();
        assert_eq!(
            state.cursor.as_ref().map(|cursor| cursor.spoken_count()),
            Some(2)
        );
    }

    #[test]
    fn test_generation_includes_taught_session_fact_and_self_check_is_clean() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.assert_nl_fact("张三的徒弟是钱七");
        let profile = reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(
            profile.contains("徒弟") && profile.contains("钱七"),
            "教过的会话事实应进入介绍：教学回执={}；介绍={}",
            taught,
            profile
        );
        assert!(profile.contains("零幻觉"), "自检不应误报：{}", profile);
        assert!(!profile.contains("未通过"), "{}", profile);
    }

    #[test]
    fn test_continue_generation_reports_functional_correction() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.slots.policy_mut().declare_functional("师傅", true);
        reasoner.observe_facts(&master_facts());
        let first = reasoner.generate_about("李四").expect("第一页应可生成");
        assert!(first.contains("王五"), "{}", first);

        reasoner.correct_nl_fact("改口：李四的师傅是赵六");
        let corrected = reasoner.continue_generation().expect("应输出更正");
        assert!(corrected.contains("更正"), "{}", corrected);
        assert!(corrected.contains("现在更新为"), "{}", corrected);
        assert!(corrected.contains("赵六"), "{}", corrected);
        assert!(
            corrected.contains("李四的师傅是赵六"),
            "更正应给出新说法：{}",
            corrected
        );
        assert!(
            !corrected.contains("未通过"),
            "更正不应被自检误报：{}",
            corrected
        );
    }

    #[test]
    fn test_generation_can_introduce_session_only_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.assert_nl_fact("钱七是孙八的师傅");
        let profile = reasoner
            .generate_about("钱七")
            .expect("会话层新实体也应可介绍");
        assert!(
            profile.contains("钱七") && profile.contains("孙八") && profile.contains("师傅"),
            "教学回执={}；介绍={}",
            taught,
            profile
        );
        assert!(profile.contains("零幻觉"), "{}", profile);
    }

    fn pending_prediction_id(reasoner: &SymbolicReasoner, object: &str) -> String {
        reasoner
            .predictive_ledger()
            .pending()
            .iter()
            .find(|hypothesis| hypothesis.key.object == object)
            .map(|hypothesis| hypothesis.id.clone())
            .unwrap_or_else(|| panic!("应存在指向「{}」的待定推测", object))
    }

    #[test]
    fn test_predictive_elaboration_is_marked_as_speculation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let text = reasoner
            .predictive_elaboration("李四")
            .expect("链接预测应生成推测段落");
        assert!(text.contains("推测，不是已确认事实"), "{}", text);
        assert!(text.contains("赵六"), "{}", text);
        assert!(reasoner
            .predictive_ledger()
            .pending()
            .iter()
            .any(|hypothesis| hypothesis.key.object == "赵六"));
    }

    #[test]
    fn test_prediction_drafts_prefetch_hit_and_invalidate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.context.record_entities(&["李四".to_string()]);
        reasoner.context.record_predicate("师祖");

        let warmed = reasoner.prefetch_answer_drafts();
        assert!(warmed > 0, "轮末预取应物化焦点实体的候选");
        let perf = reasoner.perf_ledger();
        let hits_before = perf.count("prediction_draft_hits");
        let predictions = reasoner.predictions_for_entity("李四", None, 20);
        assert!(!predictions.is_empty());
        assert!(
            perf.count("prediction_draft_hits") > hits_before,
            "相同的实体/谓词/上限应命中预取草稿"
        );

        let first = reasoner
            .answer_by_prediction("李四的师祖是谁")
            .expect("应能给出链接预测候选");
        let second = reasoner
            .answer_by_prediction("李四的师祖是谁")
            .expect("命中草稿后仍应给出候选");
        assert_eq!(first, second, "草稿命中与重算的输出必须逐字节一致");

        let report = reasoner
            .handle_predictive_command("预测草稿")
            .expect("预测草稿命令应可读");
        assert!(report.contains("预测答案草稿"), "{}", report);

        reasoner.observe_facts(&[PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "同事".to_string(),
            to: "钱七".to_string(),
        }]);
        let misses_before = perf.count("prediction_draft_misses");
        let refreshed = reasoner.predictions_for_entity("李四", None, 20);
        assert!(!refreshed.is_empty());
        assert!(
            perf.count("prediction_draft_misses") > misses_before,
            "知识版本推进后旧草稿必须整体失效"
        );
    }

    #[test]
    fn test_predictive_confirmation_feeds_session_generation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner
            .predictive_elaboration("李四")
            .expect("链接预测应生成推测段落");
        let id = pending_prediction_id(&reasoner, "赵六");
        let confirmed = reasoner.predictive_confirm(&id);
        assert!(confirmed.contains("已确认预测"), "{}", confirmed);
        let profile = reasoner
            .generate_about("李四")
            .expect("确认后的会话事实应参与介绍");
        assert!(profile.contains("赵六"), "{}", profile);
        assert!(profile.contains("零幻觉"), "{}", profile);
    }

    #[test]
    fn test_predictive_refutation_suppresses_candidate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner
            .predictive_elaboration("李四")
            .expect("链接预测应生成推测段落");
        let id = pending_prediction_id(&reasoner, "赵六");
        let refused = reasoner.predictive_refute(&id);
        assert!(refused.contains("已否定预测"), "{}", refused);
        assert!(refused.contains("不再提名"), "{}", refused);
        reasoner
            .predictive_elaboration("李四")
            .expect("否认后仍应能生成剩余推测");
        assert!(reasoner
            .predictive_ledger()
            .pending()
            .iter()
            .all(|hypothesis| hypothesis.key.object != "赵六"));
    }

    #[test]
    fn test_predictive_ledger_survives_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner
            .predictive_elaboration("李四")
            .expect("链接预测应生成推测段落");
        let id = pending_prediction_id(&reasoner, "赵六");
        let snapshot = reasoner.dialogue_snapshot();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("对话快照应可恢复");
        let confirmed = restored.predictive_confirm(&id);
        assert!(confirmed.contains("已确认预测"), "{}", confirmed);
    }

    #[test]
    fn test_generation_cursor_survives_snapshot_and_restores_identically() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config.clone(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("第一页应可生成");
        let snapshot = reasoner.dialogue_snapshot();

        let mut restored = SymbolicReasoner::new(config, HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("对话快照应可恢复");
        let update = [PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "朋友".to_string(),
            to: "赵六".to_string(),
        }];
        reasoner.observe_facts(&update);
        restored.observe_facts(&update);

        assert_eq!(
            reasoner.continue_generation(),
            restored.continue_generation(),
            "存档→恢复→知识更新→接着说 应与不中断会话一致"
        );
    }

    #[test]
    fn test_legacy_generation_state_without_cursor_migrates_without_repeat() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        let first_body = first
            .lines()
            .find(|line| line.trim().ends_with('。'))
            .expect("第一页应有一句事实")
            .trim()
            .to_string();
        let first_sentence = first_body
            .split_once('，')
            .map(|(_, sentence)| sentence.to_string())
            .unwrap_or(first_body);
        // 模拟旧快照：只有位置，没有话语账本。
        reasoner
            .context
            .record_generation("张三", 1, GenerationOrder::Relevance);
        assert!(reasoner
            .context
            .generation_state()
            .unwrap()
            .cursor
            .is_none());
        let second = reasoner.continue_generation().expect("应继续生成");
        assert!(
            !second.contains(first_sentence.trim_end_matches('。')),
            "旧快照迁移后不应重复：第一页={}；第二页={}",
            first,
            second
        );

        // 缺省 cursor 的旧 JSON 必须能解码（serde default，不破坏旧存档）。
        let legacy = r#"{
            "focus_entities": ["张三"],
            "last_predicate": null,
            "history": [],
            "generation": {"entity": "张三", "shown": 1, "order": "Relevance"}
        }"#;
        let snapshot: DialogueSnapshot = serde_json::from_str(legacy).expect("旧快照应可解码");
        let generation = snapshot.generation.unwrap();
        assert!(generation.cursor.is_none());
        assert!(generation.plan.is_none(), "旧快照缺省字段应为 None");
    }

    #[test]
    fn test_continue_generation_and_switch_angle() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());

        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        assert!(first.contains("共 1 条事实"), "实际：{}", first);
        let second = reasoner.continue_generation().expect("接着说应返回下一页");
        assert!(second.contains("续：第 2–2 句"), "实际：{}", second);
        assert!(second.contains("接着，"));
        assert_ne!(first, second);

        let done = reasoner.continue_generation().expect("讲完后应给出提示");
        assert!(done.contains("已经介绍完毕"));

        let angled = reasoner
            .switch_generation_angle()
            .expect("换个角度说应重新生成");
        assert!(angled.contains("换个角度"), "实际：{}", angled);
        assert!(reasoner.dialogue_summary().contains("上一段生成"));
    }

    #[test]
    fn generation_plan_supports_style_restyle_without_new_claims() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("生成后应留下回答计划");
        assert!(plan.claim_count() >= 2, "计划应覆盖全部可用命题");
        assert_eq!(plan.revision, reasoner.revision());

        let brief = reasoner
            .restyle_generation(StyleDirective::Brief)
            .expect("简短重述应可用");
        assert!(brief.contains("简短版"), "{}", brief);
        for claim in plan.claims.iter().take(3) {
            assert!(brief.contains(claim.text.trim()), "{}", brief);
        }
        for claim in plan.claims.iter().skip(3) {
            assert!(!brief.contains(claim.text.trim()), "越界：{}", brief);
        }

        let detailed = reasoner
            .restyle_generation(StyleDirective::Detailed)
            .expect("完整重述应可用");
        for claim in &plan.claims {
            assert!(detailed.contains(claim.text.trim()), "{}", detailed);
        }
        assert!(detailed.contains("来源："), "{}", detailed);

        let focused = reasoner
            .restyle_generation(StyleDirective::Focus(2))
            .expect("定位第二点应可用");
        assert!(focused.contains(plan.claims[1].text.trim()), "{}", focused);
        assert!(
            !focused.contains(plan.claims[0].text.trim()),
            "定位单点不得夹带其它命题：{}",
            focused
        );

        let restated = reasoner
            .restyle_generation(StyleDirective::Restate)
            .expect("换种表达应可用");
        for claim in &plan.claims {
            assert!(restated.contains(claim.text.trim()), "{}", restated);
        }
    }

    #[test]
    fn generation_plan_cites_registered_triples() {
        let mut config = test_config();
        config.max_profile_sentences = 6;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("生成后应留下回答计划");
        assert!(
            plan.claims.iter().all(|claim| !claim.evidence.is_empty()),
            "每条命题都应带证据引用：{:?}",
            plan.claims
        );
        assert!(
            plan.claims
                .iter()
                .any(|claim| claim.evidence.iter().any(|item| {
                    item.subject == "张三" && item.predicate == "师傅" && item.object == "李四"
                })),
            "师傅事实应出现在证据中：{:?}",
            plan.claims
        );
        let known: HashSet<(String, String, String)> = master_facts()
            .into_iter()
            .flat_map(|fact| match fact {
                PrologFact::Relation { from, rel_type, to } => vec![(from, rel_type, to)],
                PrologFact::Entity {
                    name, properties, ..
                } => properties
                    .into_iter()
                    .map(|(key, value)| (name.clone(), key, value))
                    .collect(),
            })
            .collect();
        for claim in &plan.claims {
            for item in &claim.evidence {
                assert!(
                    known.contains(&(
                        item.subject.clone(),
                        item.predicate.clone(),
                        item.object.clone()
                    )),
                    "证据必须是登记事实：{:?}",
                    item
                );
            }
        }
    }

    #[test]
    fn answer_plan_report_exposes_identity_version_and_actions() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("生成后应留下回答计划");
        let report = reasoner.answer_plan_report();
        assert!(report.contains(&plan.id), "{}", report);
        assert!(report.contains("与当前知识版本一致"), "{}", report);
        assert!(report.contains(plan.claims[0].text.trim()), "{}", report);
        assert!(report.contains("简短点"), "{}", report);

        reasoner.observe_facts(&[PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "徒弟".to_string(),
            to: "张三".to_string(),
        }]);
        let stale = reasoner.answer_plan_report();
        assert!(stale.contains("已过期"), "{}", stale);
        assert!(stale.contains(&format!("v{}", plan.revision)), "{}", stale);
    }

    #[test]
    fn generation_plan_attaches_claim_envelopes_only_when_pcc_is_enabled() {
        let mut config = test_config();
        config.max_profile_sentences = 2;
        let mut reasoner = SymbolicReasoner::new(config.clone(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plain = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("应有计划");
        assert!(
            plain.claims.iter().all(|claim| claim.envelope.is_none()),
            "关闭 PCC 时不得写入新区间字段"
        );

        config.confidence.enabled = true;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("应有计划");
        assert!(plan.claims.iter().all(|claim| claim.envelope.is_some()));
        let report = reasoner.answer_plan_report();
        assert!(report.contains("置信区间"), "{}", report);
        assert!(report.contains("可断言"), "{}", report);
        for claim in &plan.claims {
            let envelope = claim.envelope.expect("已附加区间");
            assert_eq!(
                reasoner.config.confidence.envelope_verdict(envelope),
                crate::confidence::EnvelopeVerdict::Assert,
                "纯图谱断言的默认区间应达到回答阈值：{}",
                report
            );
        }
    }

    #[test]
    fn epistemic_review_queue_survives_snapshot_and_is_visible_in_plan_report() {
        let mut config = test_config();
        config.max_profile_sentences = 4;
        config.confidence.enabled = true;
        let mut reasoner = SymbolicReasoner::new(config.clone(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("记住：张三的师傅是赵六");
        reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(
            reasoner.epistemic_turns.pending_count() > 0,
            "会话命题的跨阈值区间应进入复核队列"
        );
        let report = reasoner.answer_plan_report();
        assert!(report.contains("待复核"), "{}", report);
        assert!(report.contains("优先级"), "{}", report);

        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.epistemic_turns.is_some());
        let plain = SymbolicReasoner::new(config, HashMap::new());
        assert!(
            !serde_json::to_string(&plain.dialogue_snapshot())
                .unwrap()
                .contains("epistemic_turns"),
            "空台账不得写字段"
        );
        let mut restored = SymbolicReasoner::new(reasoner.config.clone(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("复核台账应可随快照恢复");
        assert_eq!(restored.epistemic_turns, reasoner.epistemic_turns);
        assert!(restored.answer_plan_report().contains("待复核"));
    }

    #[test]
    fn review_reports_a_confidence_correction_for_a_previously_asserted_claim() {
        let mut config = test_config();
        config.max_profile_sentences = 4;
        config.confidence.enabled = true;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.as_ref())
            .expect("应有计划");
        assert!(plan
            .claims
            .iter()
            .filter(|claim| claim.predicate == "师傅")
            .all(|claim| reasoner
                .config
                .confidence
                .envelope_verdict(claim.envelope.expect("图谱命题应带区间"))
                == crate::confidence::EnvelopeVerdict::Assert));

        reasoner.assert_nl_fact("记住：张三的师傅是赵六");
        reasoner
            .generate_about("张三")
            .expect("会话事实应可重新生成");
        let report = reasoner.review_generation_plan();
        assert!(report.contains("置信复核"), "{}", report);
        assert!(report.contains("更正"), "{}", report);
        assert!(report.contains("当前知识"), "{}", report);
        let claim = reasoner
            .epistemic_turns
            .last_claims
            .values()
            .find(|claim| claim.predicate == "师傅")
            .expect("复核后应记录当前命题");
        assert_ne!(
            reasoner.config.confidence.envelope_verdict(claim.envelope),
            crate::confidence::EnvelopeVerdict::Assert
        );
    }

    #[test]
    fn session_fact_lowers_its_claim_envelope_below_graph_assertions() {
        let mut config = test_config();
        config.max_profile_sentences = 4;
        config.confidence.enabled = true;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let graph_lower = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.as_ref())
            .and_then(|plan| {
                plan.claims
                    .iter()
                    .find(|claim| claim.predicate == "师傅")
                    .and_then(|claim| claim.envelope)
            })
            .expect("图谱师傅命题应带区间")
            .lower_bps();

        let reply = reasoner.assert_nl_fact("记住：张三的师傅是赵六");
        assert!(reply.contains("已记住"), "{}", reply);
        reasoner
            .generate_about("张三")
            .expect("会话事实应可重新生成");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("应有计划");
        let session_claim = plan
            .claims
            .iter()
            .find(|claim| claim.predicate == "师傅")
            .expect("师傅命题仍在");
        assert!(session_claim.session, "命题应标记为会话事实");
        let envelope = session_claim.envelope.expect("会话命题应带区间");
        assert!(envelope.lower_bps() < graph_lower);
        assert_ne!(
            reasoner.config.confidence.envelope_verdict(envelope),
            crate::confidence::EnvelopeVerdict::Assert,
            "会话直陈不应直接被视为可断言"
        );
    }

    #[test]
    fn plan_review_is_read_only_and_reports_current_differences() {
        let mut config = test_config();
        config.max_profile_sentences = 2;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let saved_plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("生成后应留下回答计划");

        let fresh = reasoner.review_generation_plan();
        assert!(fresh.contains("复核对象"), "{}", fresh);
        assert!(fresh.contains("计划内容未变"), "{}", fresh);
        assert!(fresh.contains(&saved_plan.id), "{}", fresh);
        let still_saved = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("复核不得替换旧计划");
        assert_eq!(still_saved, saved_plan, "复核必须是只读操作");

        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "赵六".to_string(),
        }]);
        let stale = reasoner.review_generation_plan();
        assert!(stale.contains("知识版本"), "{}", stale);
        assert!(stale.contains("更新"), "{}", stale);
        assert!(stale.contains("李四"), "{}", stale);
        assert!(stale.contains("赵六"), "{}", stale);
        assert!(stale.contains("旧计划仍保留"), "{}", stale);
        let after_review = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("旧计划仍应在");
        assert_eq!(after_review.revision, saved_plan.revision);
    }

    #[test]
    fn plan_review_without_a_plan_explains_how_to_create_one() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let report = reasoner.review_generation_plan();
        assert!(report.contains("还没有回答计划"), "{}", report);
        assert!(report.contains("介绍"), "{}", report);
    }

    #[test]
    fn restyle_refuses_a_plan_from_an_older_knowledge_revision() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let recorded = reasoner.revision();
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "徒弟".to_string(),
            to: "张三".to_string(),
        }]);
        assert!(reasoner.revision() > recorded);
        let text = reasoner
            .restyle_generation(StyleDirective::Brief)
            .expect("过期计划应给出明确提示");
        assert!(text.contains("知识版本"), "{}", text);
        assert!(text.contains(&format!("v{}", recorded)), "{}", text);
        assert!(text.contains("重新输入"), "{}", text);
    }

    #[test]
    fn generation_plan_survives_dialogue_snapshot_roundtrip() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config.clone(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        let plan = reasoner
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("生成后应留下回答计划");
        let snapshot = reasoner.dialogue_snapshot();
        let json = serde_json::to_string(&snapshot).expect("计划应可序列化");
        let restored_snapshot: DialogueSnapshot =
            serde_json::from_str(&json).expect("计划应可反序列化");

        let mut restored = SymbolicReasoner::new(config, HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&restored_snapshot)
            .expect("快照应能恢复");
        let restored_plan = restored
            .context()
            .generation_state()
            .and_then(|state| state.plan.clone())
            .expect("恢复后计划仍应在");
        assert_eq!(restored_plan, plan);
        let text = restored
            .restyle_generation(StyleDirective::Detailed)
            .expect("重启后仍应可完整重述");
        assert!(text.contains(plan.claims[0].text.trim()), "{}", text);
    }

    #[test]
    fn test_expected_answer_type_reorders_find_object() {
        let mut facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "星海公司".to_string(),
            },
        ];
        facts.push(PrologFact::Entity {
            name: "李四".to_string(),
            entity_type: "人物".to_string(),
            properties: HashMap::new(),
        });
        facts.push(PrologFact::Entity {
            name: "星海公司".to_string(),
            entity_type: "公司".to_string(),
            properties: HashMap::new(),
        });
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner
            .answer("张三的徒弟是谁")
            .expect("应能回答反查对象问题");
        let person_at = answer.find("李四").expect("应包含李四");
        let company_at = answer.find("星海公司").expect("应包含星海公司");
        assert!(person_at < company_at, "人物类型答案应排在前面：{}", answer);
        assert!(answer.contains("期望"), "应说明排序依据：{}", answer);
    }

    #[test]
    fn test_predict_continuation_uses_facts_and_backoff() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let completion = reasoner
            .predict_continuation("张三的")
            .expect("已知实体应能续写");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));

        let mut synonyms = HashMap::new();
        synonyms.insert("老三".to_string(), "张三".to_string());
        let mut aliased = SymbolicReasoner::new(test_config(), synonyms);
        aliased.observe_facts(&master_facts());
        let aliased_completion = aliased
            .predict_continuation("老三的")
            .expect("同义词前缀应先归一化再续写");
        assert!(aliased_completion.contains("张三"));
    }

    fn nationality_facts() -> Vec<PrologFact> {
        let mut facts = Vec::new();
        for (name, has_nationality) in [
            ("张三", true),
            ("李四", true),
            ("王五", true),
            ("赵六", false),
        ] {
            let mut properties = HashMap::new();
            properties.insert("出生地".to_string(), format!("{}市", name));
            if has_nationality {
                properties.insert("国籍".to_string(), "中国".to_string());
            }
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "人物".to_string(),
                properties,
            });
        }
        // 一个无特征的地点篮子：拉低国籍先验，使关联规则的提升度可判
        facts.push(PrologFact::Entity {
            name: "无名城".to_string(),
            entity_type: "地点".to_string(),
            properties: HashMap::new(),
        });
        // 关系事实：让生成器有全局统计可回退（属性不进入关系计数）
        facts.push(PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        });
        facts
    }

    #[test]
    fn test_resolution_proof_from_facts_and_chain_goals() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("张三的师傅的师傅是王五");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(report.contains("王五"));
        assert!(report.contains("证明树"));
        assert!(report.contains("事实"));
        let failed = reasoner.resolution_proof_report("张三的师傅的师傅是赵六");
        assert!(failed.contains("未得证"), "实际：{}", failed);
        assert!(
            failed.contains("搜索状态："),
            "失败报告应披露搜索认识论状态：{}",
            failed
        );
        assert!(
            failed.contains("已穷尽") || failed.contains("搜索被截断"),
            "搜索状态必须区分无解与截断：{}",
            failed
        );
        assert!(
            failed.contains("搜索统计："),
            "搜索报告应披露确定性计数：{}",
            failed
        );
        let view = reasoner
            .resolution_proof_view("张三的师傅的师傅是赵六")
            .expect("应得到结构化证明视图");
        assert!(view.stats.goals >= 1);
        assert!(view.stats.goal_attempts > 0);
        assert!(!view.frontier.is_empty(), "穷尽无解应保留失败前沿");
    }

    #[test]
    fn test_resolution_proof_uses_mined_horn_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("李四的师祖是赵六");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(
            report.contains("链接预测规则"),
            "证明树应标出规则来源：{}",
            report
        );
        assert!(report.contains("师傅"));
    }

    #[test]
    fn test_resolution_query_returns_solution_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_query_report("张三的师傅的师傅是谁");
        assert!(report.contains("答案是「王五」"), "实际：{}", report);
        assert!(report.contains("事实"));
        assert!(report.contains("知识库"));
    }

    #[test]
    fn belief_ledger_tracks_cross_turn_slots_and_survives_snapshot() {
        let mut config = test_config();
        config.max_profile_sentences = 4;
        config.confidence.enabled = true;
        let mut reasoner = SymbolicReasoner::new(config.clone(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(
            reasoner.belief_ledger().len() > 0,
            "生成计划后账本应记录槽位信念"
        );
        let report = reasoner.answer_plan_report();
        assert!(report.contains("信念账本"), "{}", report);
        assert!(report.contains("师傅"), "{}", report);

        reasoner.assert_nl_fact("记住：张三的师傅是赵六");
        reasoner.generate_about("张三").expect("会话事实应可重新生成");
        let masters = reasoner.belief_ledger().slot_beliefs("张三", "师傅");
        assert!(masters.len() >= 2, "同一槽位应保留冲突候选：{:?}", masters);
        assert!(
            masters.iter().any(|belief| belief.conflicts > 0),
            "不同取值应记冲突：{:?}",
            masters
        );

        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.beliefs.is_some());
        let plain = SymbolicReasoner::new(config, HashMap::new());
        assert!(
            !serde_json::to_string(&plain.dialogue_snapshot())
                .unwrap()
                .contains("\"beliefs\""),
            "空账本不得写字段"
        );
        let mut restored = SymbolicReasoner::new(reasoner.config.clone(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("信念账本应可随快照恢复");
        assert_eq!(
            restored.belief_ledger().len(),
            reasoner.belief_ledger().len()
        );
        assert!(restored.belief_ledger_report().contains("赵六"));
    }

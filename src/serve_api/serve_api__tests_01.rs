    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn turn_response_is_send() {
        assert_send::<TurnResponse>();
    }

    #[test]
    fn api_state_is_send_and_sync() {
        // actor 化后的核心契约：HTTP 层可以跨线程/跨任务共享服务状态
        assert_send_sync::<ApiState>();
    }

    #[tokio::test]
    async fn plugin_recipe_answers_across_http_sessions() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.height"
version = "0.1.0"
api = 1
name = "身高包"
default_mode = "on"

[[contributions]]
kind = "recipe"
id = "height.recipe"
patterns = ["{entity}有多高"]
question = "{entity}的身高是什么"
attribute = "身高"
template = "height.template"

[[contributions]]
kind = "template"
id = "height.template"
template = "「{entity}」的身高是{answer}。"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        assert_eq!(runtime.enabled_count(), 1);

        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let facts = vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人物".to_string(),
            properties: yao,
        }];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "姚明有多高")
            .await
            .expect("轮次应成功");
        assert_eq!(turn.source, "plugin");
        assert!(turn.reply.contains("226"), "{}", turn.reply);
        assert!(
            turn.plan.iter().any(|plan| plan.contains("height.recipe")),
            "{:?}",
            turn.plan
        );
    }

    /// 插件本体进入服务暖基座：逆向配方 + 子类类型约束 + 依据槽位都可跨会话工作。
    #[tokio::test]
    async fn plugin_typed_inverse_recipe_reaches_http_sessions() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
name = "电影包"
default_mode = "on"

[[contributions]]
kind = "ontology"
id = "film.ontology"
predicates = [
  { name = "导演", inverse = "执导", domain = ["电影"], range = ["人物"] },
]
types = [
  { name = "电影", subclass_of = ["影视作品"] },
  { name = "动画电影", subclass_of = ["电影"] },
]

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}导演过哪些电影", "那{entity}还导演过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let receipt = state.configure_plugin_runtime(&runtime);
        assert!(receipt.lessons > 0, "插件 TBox 必须并入暖基座");
        assert!(receipt.inverse_surfaces > 0, "逆关系词面必须登记");
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "卡梅隆导演过哪些电影")
            .await
            .expect("轮次应成功");
        assert_eq!(turn.source, "plugin", "{}", turn.reply);
        assert!(turn.reply.contains("阿凡达"), "{}", turn.reply);
        assert!(
            turn.reply.contains("导演") && turn.reply.contains("依据"),
            "依据必须接地：{}",
            turn.reply
        );

        // 多轮：省略主语时用对话焦点补全，再沿同一本体做逆向检索。
        let followup = state
            .take_turn(&session_id, "那他还导演过什么")
            .await
            .expect("追问轮次应成功");
        assert_eq!(followup.source, "plugin", "{}", followup.reply);
        assert!(followup.reply.contains("阿凡达"), "{}", followup.reply);
    }

    /// 正向配方回答后焦点转到答案实体：下一轮「这位导演…」可无缝续问。
    #[tokio::test]
    async fn plugin_forward_answer_moves_focus_for_role_followup() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
name = "电影包"
default_mode = "on"

[[contributions]]
kind = "reference"
id = "film.reference"
quantifiers = ["这位", "那位"]
role_words = ["导演"]

[[contributions]]
kind = "recipe"
id = "film.directed_by"
patterns = ["{entity}是谁导演的"]
question = "{entity}的导演是什么"
attribute = "导演"
direction = "forward"
answer_types = ["人物"]
template = "film.directed.template"

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}还拍过什么", "那{entity}还拍过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.directed.template"
template = "「{entity}」的导演是{answer}。依据：{evidence}"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let first = state
            .take_turn(&session_id, "阿凡达是谁导演的")
            .await
            .expect("首轮应成功");
        assert_eq!(first.source, "plugin", "{}", first.reply);
        assert!(first.reply.contains("卡梅隆"), "{}", first.reply);

        let followup = state
            .take_turn(&session_id, "这位导演还拍过什么")
            .await
            .expect("追问轮次应成功");
        assert_eq!(followup.source, "plugin", "{}", followup.reply);
        assert!(followup.reply.contains("阿凡达"), "{}", followup.reply);
    }

    fn test_state(max_sessions: usize) -> Arc<ApiState> {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        yao.insert("国籍".to_string(), "中国".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人物".to_string(),
                properties: ye,
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        Arc::new(ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            max_sessions,
            StartupLedger::shared(),
        ))
    }

    /// 配置态本体注入暖基座：分叉出的每个会话同源携带，且注入幂等。
    #[test]
    fn configure_ontology_reaches_forked_sessions() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let state = test_state(2);
        let program = OntologyProgram::from_config(
            &TypeSystemConfig {
                subclasses: vec![TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[PredicateProfileConfig {
                name: "师傅".into(),
                domain: vec!["人物".into()],
                ..PredicateProfileConfig::default()
            }],
        );
        assert_eq!(state.configure_ontology(&program), 2, "子类1 + 定义域1");
        assert_eq!(state.configure_ontology(&program), 0, "重复注入必须去重");
        let fork = state.warm.fork();
        assert_eq!(
            fork.taught_ontology_lessons().len(),
            2,
            "分叉会话应携带配置本体"
        );
        let snapshot = fork.dialogue_snapshot();
        assert!(
            snapshot.ontology.is_some(),
            "配置本体应随会话快照导出（生成 = 序列化）"
        );
    }

    fn test_state_with_quality() -> Arc<ApiState> {
        use crate::predicate_registry::PredicateRegistry;
        use crate::quality::{build_report, QualityIndex};
        use crate::type_system::TypeSystem;

        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人物".to_string(),
                properties: ye,
            },
        ];
        let mut registry = PredicateRegistry::new();
        registry.observe("姚明", "身高", "226", Some("体育百科"));
        let report = build_report(&facts, &registry, &TypeSystem::new(), &[]);
        let index = QualityIndex::from_report(&report, &facts, &registry);
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let index = index.with_revision(state.warm_revision());
        state.set_quality_index(Some(Arc::new(index)));
        Arc::new(state)
    }

    #[tokio::test]
    async fn quality_index_reaches_http_sessions() {
        let state = test_state_with_quality();
        assert!(state.quality_index().is_some());
        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "姚明的身高是多少")
            .await
            .expect("轮次应成功");
        assert!(turn.reply.contains("226"), "{}", turn.reply);
        assert!(
            turn.plan.iter().any(|plan| plan == "quality_cite_source"),
            "HTTP 回答必须带出处：{:?} / {}",
            turn.plan,
            turn.reply
        );
        assert!(turn.reply.contains("出处：体育百科"), "{}", turn.reply);
        assert_eq!(
            state.quality_index().map(|index| index.revision),
            Some(state.warm_revision())
        );
    }

    #[tokio::test]
    async fn generation_and_budget_directives_work_over_http_sessions() {
        let state = test_state(1);
        let session_id = state.create_session().await.expect("会话应创建成功");

        let generated = state
            .take_turn(&session_id, "介绍：姚明")
            .await
            .expect("生成轮应成功");
        assert_eq!(generated.status, "answered", "{}", generated.reply);
        assert_eq!(generated.source, "generation", "{}", generated.reply);
        assert!(generated.reply.contains("姚明"), "{}", generated.reply);

        let brief = state
            .take_turn(&session_id, "简短点")
            .await
            .expect("风格轮应成功");
        assert_eq!(brief.source, "generation_plan", "{}", brief.reply);
        assert!(brief.reply.contains("简短版"), "{}", brief.reply);

        let budgeted = state
            .take_turn(&session_id, "预算：80字")
            .await
            .expect("预算轮应成功");
        assert_eq!(budgeted.source, "generation_plan", "{}", budgeted.reply);
        assert!(budgeted.reply.contains("预算版"), "{}", budgeted.reply);
        assert!(
            budgeted.reply.chars().count() <= 80,
            "预算必须约束最终 HTTP 回复：{}",
            budgeted.reply
        );

        let before = state
            .take_turn(&session_id, "姚明的身高是多少")
            .await
            .expect("普通问答应成功");
        assert_ne!(before.source, "generation_plan");
        assert!(before.reply.contains("226"), "{}", before.reply);

        let report = state
            .take_turn(&session_id, "回答计划")
            .await
            .expect("计划报告应成功");
        assert_eq!(report.source, "generation_plan", "{}", report.reply);
        assert_eq!(report.kind.as_deref(), Some("plan_report"));
        assert!(report.reply.contains("回答计划"), "{}", report.reply);
        assert!(report.reply.contains("姚明"), "{}", report.reply);

        let review = state
            .take_turn(&session_id, "复核计划")
            .await
            .expect("计划复核应成功");
        assert_eq!(review.source, "generation_plan", "{}", review.reply);
        assert_eq!(review.kind.as_deref(), Some("plan_review"));
        assert!(review.reply.contains("复核对象"), "{}", review.reply);

        let ledger = state
            .take_turn(&session_id, "信念账本")
            .await
            .expect("信念账本应成功");
        assert_eq!(ledger.source, "belief_ledger", "{}", ledger.reply);
        assert!(!ledger.reply.is_empty(), "信念账本不应为空输出");
    }

    #[tokio::test]
    async fn knowledge_assertions_reach_existing_sessions() {
        let state = test_state(4);
        let first = state.create_session().await.expect("会话应创建成功");
        let second = state.create_session().await.expect("会话应创建成功");

        let before = state
            .take_turn(&first, "姚明的队友是谁")
            .await
            .expect("轮次应成功");
        assert!(!before.reply.contains("叶莉"), "{}", before.reply);
        let layers_before = state.session_layers(&first).await.expect("层清单应可用");
        let base_before = layers_before["manifest"]["layers"][0]["count"]
            .as_u64()
            .expect("L0 计数");

        let applied = state.assert_facts(
            &[PrologFact::Relation {
                from: "姚明".to_string(),
                rel_type: "队友".to_string(),
                to: "叶莉".to_string(),
            }],
            "test",
        );
        assert_eq!(applied, 1);
        assert_eq!(state.knowledge_snapshot().generation, 1);
        let layers_after = state.session_layers(&first).await.expect("层清单应可用");
        let base_after = layers_after["manifest"]["layers"][0]["count"]
            .as_u64()
            .expect("L0 计数");
        assert!(
            base_after > base_before,
            "读取层清单前应追赶知识增量：{base_before} -> {base_after}"
        );

        for session in [&first, &second] {
            let after = state
                .take_turn(session, "姚明的队友是谁")
                .await
                .expect("轮次应成功");
            assert!(after.reply.contains("叶莉"), "{}", after.reply);
        }
        assert_eq!(
            state.assert_facts(
                &[PrologFact::Relation {
                    from: "姚明".to_string(),
                    rel_type: "队友".to_string(),
                    to: "叶莉".to_string(),
                }],
                "test",
            ),
            0
        );
        assert_eq!(state.knowledge_snapshot().generation, 1);
    }

    #[tokio::test]
    async fn alias_admission_reaches_active_sessions() {
        let state = test_state(2);
        let session = state.create_session().await.expect("会话应创建成功");
        let before = state
            .take_turn(&session, "明哥的身高是多少")
            .await
            .expect("轮次应成功");
        assert!(!before.reply.contains("226"), "{}", before.reply);

        assert_eq!(
            state.align_entity_aliases(&[("明哥".to_string(), "姚明".to_string())]),
            1
        );
        let after = state
            .take_turn(&session, "明哥的身高是多少")
            .await
            .expect("轮次应成功");
        assert!(after.reply.contains("226"), "{}", after.reply);
    }

    /// 注册表对齐（谓词别名/换向词面 + 实体持久别名）必须进暖基座：
    /// CLI 里签过字的说法，走 HTTP 发问同样要能命中同一份知识。
    #[tokio::test]
    async fn registry_alignment_reaches_http_sessions() {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let applied = state.align_predicate_pairs(
            &[
                ("老师".to_string(), "师傅".to_string(), false),
                ("徒弟".to_string(), "师傅".to_string(), true),
            ],
            &crate::predicate_registry::PredicateRegistry::default(),
        );
        assert_eq!(applied, 2, "正向别名与换向词面都应生效");
        let entities = state.align_entity_aliases(&[("明哥".to_string(), "姚明".to_string())]);
        assert_eq!(entities, 1, "实体持久别名应生效");
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let forward = state
            .take_turn(&session_id, "张三的老师是谁")
            .await
            .expect("正向别名轮次应成功");
        assert!(forward.reply.contains("李四"), "{}", forward.reply);

        let inverse = state
            .take_turn(&session_id, "李四的徒弟是谁")
            .await
            .expect("换向词面轮次应成功");
        assert!(inverse.reply.contains("张三"), "{}", inverse.reply);

        let alias = state
            .take_turn(&session_id, "明哥的身高是多少")
            .await
            .expect("实体别名轮次应成功");
        assert!(
            alias.reply.contains("226"),
            "{:?} / {:?} / {}",
            alias.source,
            alias.plan,
            alias.reply
        );
    }

    /// 构造最小合法信封（`insert_order` 只改变焦点类型表的插入序，
    /// 焦点栈本身固定，用于验证摘要与 HashMap 迭代序无关）。
    fn envelope_with_focus(insert_order: &[&str]) -> SessionEnvelope {
        use crate::symbolic_reasoner::DialogueContext;
        let keys = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"];
        let mut context = DialogueContext::new(16);
        for key in &keys {
            context.record_entities_typed(&[(key.to_string(), Some("人物".to_string()))]);
        }
        let mut snapshot = context.snapshot();
        let mut focus_types = HashMap::new();
        for key in insert_order {
            focus_types.insert(key.to_string(), "人物".to_string());
        }
        snapshot.focus_types = focus_types;
        SessionEnvelope {
            format: ENVELOPE_FORMAT.to_string(),
            version: ENVELOPE_VERSION,
            session_id: "s-test".to_string(),
            turns: 2,
            history: vec!["姚明是谁".to_string()],
            policy: SessionPolicy::default(),
            manager: DialogueManager::new().snapshot(),
            reasoner: snapshot,
            engine: EngineStamp {
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                kb_fingerprint: "test-fingerprint".to_string(),
                kb_facts: 2,
                kb_generation: 0,
                prompt_mode: "on".to_string(),
                recovery_mode: "observe".to_string(),
            },
            digest: String::new(),
        }
    }

    #[test]
    fn envelope_digest_is_canonical_and_hashmap_order_invariant() {
        let forward_keys = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"];
        let reversed_keys: Vec<&str> = forward_keys.iter().rev().copied().collect();
        let mut forward = envelope_with_focus(&forward_keys);
        let mut backward = envelope_with_focus(&reversed_keys);
        forward.seal();
        backward.seal();
        assert!(forward.digest_valid() && backward.digest_valid());
        assert_eq!(
            forward.digest, backward.digest,
            "摘要不得依赖 HashMap 迭代序（跨进程换入不得误判 Stale）"
        );

        // L1/L2/L3/L4：规范字节往返 + 摘要交换
        let bytes = canonical::canonical_bytes(&forward).unwrap();
        let (decoded, receipt) = SessionEnvelope::decode(&bytes, 1 << 20).unwrap();
        assert!(receipt.canonical, "自产字节必须是规范形");
        assert_eq!(decoded.digest, forward.digest);
        assert!(decoded.digest_valid());
        assert_eq!(canonical::canonical_bytes(&decoded).unwrap(), bytes);
        assert_eq!(
            canonical::canonical_digest(&decoded).unwrap(),
            canonical::canonical_digest(&forward).unwrap()
        );
    }

    #[test]
    fn envelope_decode_error_taxonomy_is_stable() {
        let mut base = envelope_with_focus(&["甲"]);
        base.seal();
        let canonical_bytes = canonical::canonical_bytes(&base).unwrap();

        assert_eq!(
            SessionEnvelope::decode(b"{not json", 1 << 20)
                .unwrap_err()
                .code(),
            "parse"
        );
        assert_eq!(
            SessionEnvelope::decode(&canonical_bytes, 8)
                .unwrap_err()
                .code(),
            "too_large"
        );

        let mut wrong_format = base.clone();
        wrong_format.format = "other-format".to_string();
        wrong_format.seal();
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&wrong_format).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "format"
        );

        let mut future = base.clone();
        future.version = ENVELOPE_VERSION + 1;
        future.seal();
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&future).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "version"
        );

        let mut tampered = base.clone();
        tampered.history.push("伪造的历史".to_string());
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&tampered).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "digest"
        );
    }

    /// 服务跑在专属 OS 线程的 current-thread 运行时里（会话 actor 由 `run`
    /// 自行派生）；测试只通过 TCP 交互。
    fn start_server(max_sessions: usize) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_recovery(max_sessions, RecoveryMode::Observe)
    }

    #[tokio::test]
    async fn tampered_cold_session_surfaces_decode_code_without_side_effects() {
        let dir = temp_session_dir("decode");
        let (addr, _server) =
            start_server_with_options(1, RecoveryMode::Observe, Some(dir.clone()), None);
        let first = create_session(addr).await;
        take_turn_body(addr, &first, "姚明是谁").await;
        let _second = create_session(addr).await;
        let path = dir.join(format!("{}.json", first));
        assert!(path.exists(), "容量 1 应把 first 冻结到磁盘");

        // 换出写规范字节：解码回执必须标记 canonical（L2 字节不动点）
        let frozen_bytes = std::fs::read(&path).unwrap();
        let (frozen, receipt) = SessionEnvelope::decode(&frozen_bytes, 1 << 20).unwrap();
        assert!(receipt.canonical, "换出的冷信封必须写规范字节");
        assert!(frozen.digest_valid());
        assert_eq!(
            canonical::canonical_bytes(&frozen).unwrap(),
            frozen_bytes,
            "规范字节再编码逐字节一致"
        );

        // 篡改 turns 但不重算摘要 → 换入必须被拒，且给出稳定错误码
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        value["turns"] = serde_json::json!(99);
        std::fs::write(&path, value.to_string()).unwrap();
        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", first), &[], "").await;
        assert_eq!(status, 409, "{}", body);
        assert!(body.contains("decode_digest"), "{}", body);

        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(metrics["sessions"], 1, "被拒的冷换入不得产生会话");
        assert_eq!(metrics["perf"]["decode_reject_digest"], 1, "{}", metrics);
        assert_eq!(
            metrics["perf"]["decode_ok"].as_u64().unwrap_or(0),
            0,
            "{}",
            metrics
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn start_server_with_recovery(
        max_sessions: usize,
        recovery: RecoveryMode,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_options(max_sessions, recovery, None, None)
    }

    fn start_server_with_budget(
        max_sessions: usize,
        budget: TurnBudget,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_options(max_sessions, RecoveryMode::Observe, None, Some(budget))
    }

    /// 与服务启动路径一致：给暖基座注入 NLP 配置，失败轮才能生成读法回执。
    fn test_nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: HashMap::new(),
            name_patterns: crate::NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: Vec::new(),
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    fn start_server_with_options(
        max_sessions: usize,
        recovery: RecoveryMode,
        session_dir: Option<std::path::PathBuf>,
        turn_budget: Option<TurnBudget>,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = TcpListener::from_std(listener).unwrap();
                let mut state = test_state(max_sessions);
                // Arc::get_mut 不适用（已 Arc）：状态在启动前唯一持有，先解包
                let inner = Arc::get_mut(&mut state).expect("测试启动前状态唯一");
                inner.set_recovery_mode(recovery);
                inner.set_nlp_config(test_nlp_config());
                if let Some(dir) = session_dir {
                    inner.set_session_dir(dir);
                }
                if let Some(budget) = turn_budget {
                    inner.set_turn_budget(budget);
                }
                let _ = run(listener, state).await;
            });
        });
        (addr, handle)
    }

    fn temp_session_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kp-serve-{}-{}", std::process::id(), label));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn raw_request(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> (u16, String, String) {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let mut request = format!("{} {} HTTP/1.1\r\nHost: localhost\r\n", method, path);
        for (name, value) in headers {
            request.push_str(&format!("{}: {}\r\n", name, value));
        }
        request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        request.push_str(body);
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8_lossy(&response).to_string();
        let status = response
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let (head, body) = response
            .split_once("\r\n\r\n")
            .map(|(head, body)| (head.to_string(), body.to_string()))
            .unwrap_or((response.clone(), String::new()));
        (status, head, body)
    }

    async fn create_session(addr: std::net::SocketAddr) -> String {
        let (status, _, body) = raw_request(addr, "POST", "/sessions", &[], "").await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        value["session_id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn invalid_utf8_turn_body_is_rejected_with_offset() {
        let (addr, _server) = start_server(1);
        let session_id = create_session(addr).await;
        let body = b"{\"input\":\"\xE5\xBC".to_vec();
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let header = format!(
            "POST /sessions/{}/turns HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            session_id,
            body.len()
        );
        stream.write_all(header.as_bytes()).await.unwrap();
        stream.write_all(&body).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 400"), "{}", head);
        let value: serde_json::Value = serde_json::from_str(body).unwrap();
        assert!(
            value["error"].as_str().unwrap().contains("偏移 10"),
            "{}",
            value
        );
        assert!(!value["error"].as_str().unwrap().contains('�'), "{}", value);
        assert!(
            raw_request(addr, "GET", "/metrics", &[], "")
                .await
                .2
                .contains("\"sessions\":1"),
            "非法请求不得创建会话"
        );
    }

    #[tokio::test]
    async fn shared_memory_http_is_scoped_and_conflict_preserving() {
        let (addr, _server) = start_server(2);
        let first = create_session(addr).await;
        let second = create_session(addr).await;

        let (status, _, body) = raw_request(addr, "GET", "/v1/memory", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["claims"].as_array().unwrap().len(), 0);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/memory",
            &[("Content-Type", "application/json")],
            r#"{"subject":"姚明","predicate":"身高","value":"226cm","source":"seed"}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "published");
        assert_eq!(value["outcome"], "published");

        // 零宽字符 + 全角数字仍命中同一条身份键：幂等重复，不产生冲突。
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/memory",
            &[("Content-Type", "application/json")],
            r#"{"subject":"姚\u200b明","predicate":"身高","value":"２２６cm"}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "duplicate");

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/memory",
            &[("Content-Type", "application/json")],
            r#"{"subject":"姚明","predicate":"身高","value":"229cm"}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "conflict");
        assert_eq!(
            value["conflict"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>(),
            vec!["226cm", "229cm"]
        );

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/memory", first),
            &[("Content-Type", "application/json")],
            r#"{"subject":"叶莉","predicate":"身高","value":"190cm"}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/memory", second),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["claims"].as_array().unwrap().len(), 2);
        assert!(!body.contains("190cm"), "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/memory", first),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["claims"].as_array().unwrap().len(), 3);
        assert!(body.contains("190cm"), "{}", body);

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", first), &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["shared_memory"]["claims"], 3);
        let shared_items: Vec<&serde_json::Value> = value["working_memory"]["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| {
                item["id"]
                    .as_str()
                    .map(|id| id.starts_with("shared:"))
                    .unwrap_or(false)
            })
            .collect();
        assert_eq!(shared_items.len(), 3, "{}", body);
        assert!(shared_items.iter().any(|item| {
            item["entities"]
                .as_array()
                .map(|entities| entities.iter().any(|entity| entity == "叶莉"))
                .unwrap_or(false)
        }));

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", second), &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["shared_memory"]["claims"], 2);
        assert!(!body.contains("190cm"), "{}", body);

        let (status, _, _) = raw_request(
            addr,
            "GET",
            "/sessions/s-missing/memory",
            &[],
            "",
        )
        .await;
        assert_eq!(status, 404);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/memory",
            &[("Content-Type", "application/json")],
            r#"{"subject":"  ","predicate":"身高","value":"1cm"}"#,
        )
        .await;
        assert_eq!(status, 400, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["code"], "empty_field");

        let (status, _, body) = raw_request(addr, "GET", "/metrics", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["shared_memory"]["claims"], 3);
        assert_eq!(value["shared_memory"]["conflicts"], 1);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/memory",
            &[("Content-Type", "application/json")],
            r#"{"subject":"叶莉","predicate":"国籍","value":"中国"}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let global_id = value["claim"]["id"].as_str().unwrap().to_string();

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/memory", first),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let private_id = value["claims"]
            .as_array()
            .unwrap()
            .iter()
            .find(|claim| claim["value"] == "190cm")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}/memory/{}", first, global_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 404, "会话路由不得删除全局声明");
        let (status, _, body) = raw_request(
            addr,
            "DELETE",
            &format!("/v1/memory/{}", global_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "retracted");

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/v1/memory/{}", private_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 404, "全局路由不得删除会话私有声明");
        let (status, _, body) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}/memory/{}", first, private_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);

        let (status, _, body) = raw_request(addr, "GET", "/metrics", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["shared_memory"]["claims"], 2);

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", first),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200);
        let (status, _, body) = raw_request(addr, "GET", "/v1/memory", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["claims"].as_array().unwrap().len(), 2, "{}", body);
        assert!(!body.contains("190cm"), "{}", body);
    }

    #[tokio::test]
    async fn knowledge_http_writes_are_visible_to_active_sessions() {
        let (addr, _server) = start_server(2);
        let session = create_session(addr).await;

        let before = take_turn_body(addr, &session, "姚明的队友是谁").await;
        assert!(!before["reply"].as_str().unwrap_or("").contains("叶莉"));

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/knowledge/facts",
            &[("Content-Type", "application/json")],
            r#"{"facts":[{"kind":"relation","from":"姚明","rel_type":"队友","to":"叶莉"}],"source":"http-test"}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["applied"], 1);
        assert_eq!(value["generation"], 1);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/knowledge/aliases",
            &[("Content-Type", "application/json")],
            r#"{"aliases":[{"surface":"明哥","canonical":"姚明"}]}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["applied"], 1);
        assert_eq!(value["generation"], 2);

        let after = take_turn_body(addr, &session, "姚明的队友是谁").await;
        assert!(after["reply"].as_str().unwrap_or("").contains("叶莉"));
        let alias = take_turn_body(addr, &session, "明哥的身高是多少").await;
        assert!(alias["reply"].as_str().unwrap_or("").contains("226"));

        let (status, _, body) =
            raw_request(addr, "GET", "/v1/knowledge?include_facts=1", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["generation"], 2);
        assert_eq!(value["facts"], 3);
        assert_eq!(value["aliases"].as_array().unwrap().len(), 1);
        assert_eq!(value["cache_epoch"], 2);
        assert_eq!(value["data"].as_array().unwrap().len(), 3);
        let snapshot_data = value["data"].clone();

        let (status, _, body) = raw_request(
            addr,
            "GET",
            "/v1/knowledge/changes?since=0",
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["generation"], 2);
        assert_eq!(value["requires_rebuild"], false);
        assert_eq!(value["changes"].as_array().unwrap().len(), 2);
        assert_eq!(value["changes"][0]["kind"], "facts");
        assert_eq!(value["changes"][1]["kind"], "aliases");

        let (status, _, body) = raw_request(
            addr,
            "GET",
            "/v1/knowledge/changes?since=2",
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["changes"].as_array().unwrap().len(), 0);

        let roundtrip = serde_json::json!({"facts": snapshot_data}).to_string();
        let (status, _, body) = raw_request(
            addr,
            "PUT",
            "/v1/knowledge/facts",
            &[("Content-Type", "application/json")],
            &roundtrip,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "unchanged");
        assert_eq!(value["generation"], 2);

        let (status, _, body) = raw_request(
            addr,
            "PUT",
            "/v1/knowledge/facts",
            &[("Content-Type", "application/json")],
            r#"{"facts":[{"kind":"entity","name":"姚明","entity_type":"人物","properties":{"身高":"226","国籍":"中国"}},{"kind":"entity","name":"叶莉","entity_type":"人物","properties":{"身高":"190"}},{"kind":"relation","from":"姚明","rel_type":"队友","to":"王五"}]}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "replaced");
        assert_eq!(value["generation"], 3);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            "/v1/knowledge/changes?since=2",
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["requires_rebuild"], true);
        assert_eq!(value["changes"].as_array().unwrap().len(), 1);
        assert_eq!(value["changes"][0]["kind"], "reset");

        let corrected = take_turn_body(addr, &session, "姚明的队友是谁").await;
        let reply = corrected["reply"].as_str().unwrap_or("");
        assert!(reply.contains("王五"), "{}", reply);
        assert!(!reply.contains("叶莉"), "{}", reply);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/knowledge/facts",
            &[("Content-Type", "application/json")],
            r#"{"facts":[{"kind":"relation","from":"","rel_type":"队友","to":"叶莉"}]}"#,
        )
        .await;
        assert_eq!(status, 400, "{}", body);
    }

    #[tokio::test]
    async fn cold_session_resumes_after_knowledge_append() {
        let dir = tempfile::tempdir().unwrap();
        let (addr, _server) = start_server_with_options(
            1,
            RecoveryMode::Observe,
            Some(dir.path().to_path_buf()),
            None,
        );
        let first = create_session(addr).await;
        take_turn_body(addr, &first, "姚明的身高是多少").await;
        let _second = create_session(addr).await;
        assert!(
            dir.path().join(format!("{}.json", first)).exists(),
            "容量 1 应把 first 冻结到磁盘"
        );

        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/knowledge/facts",
            &[("Content-Type", "application/json")],
            r#"{"facts":[{"kind":"relation","from":"姚明","rel_type":"队友","to":"叶莉"}]}"#,
        )
        .await;
        assert_eq!(status, 201, "{}", body);

        let resumed = take_turn_body(addr, &first, "姚明的队友是谁").await;
        assert!(resumed["reply"].as_str().unwrap_or("").contains("叶莉"));
    }

    #[tokio::test]
    async fn health_metrics_and_index() {
        let (addr, _server) = start_server(4);
        let (status, _, body) = raw_request(addr, "GET", "/health", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "ok");

        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Content-Type", "application/json")],
            r#"{"input": "姚明的身高是多少"}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "answered");
        assert!(value["reply"].as_str().unwrap().contains("226"), "{}", body);

        let (status, _, body) = raw_request(addr, "GET", "/metrics", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["sessions"], 1);
        assert_eq!(value["turns_total"], 1);

        let (status, _, body) = raw_request(addr, "GET", "/", &[], "").await;
        assert_eq!(status, 200);
        assert!(body.contains("knowprolog"));
    }

    #[tokio::test]
    async fn semantic_route_exposes_bounded_metrics_without_raw_question() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        let question = "姚明的身高是多少";
        let body = take_turn_body(addr, &session_id, question).await;
        assert_eq!(body["status"], "answered", "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/semantic", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["session_id"], session_id);
        assert!(
            value["metrics"]["observations"].as_u64().unwrap_or(0) >= 1,
            "{}",
            body
        );
        assert!(value["report"]
            .as_str()
            .is_some_and(|report| report.contains("语义检索监控")));
        assert!(!body.contains(question), "监控不得泄露原始问句：{}", body);

        let (status, _, body) = raw_request(addr, "GET", "/metrics", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let metrics: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            metrics["perf"]["semantic_observations"]
                .as_u64()
                .unwrap_or(0)
                >= 1,
            "{}",
            body
        );
        assert!(!body.contains(question), "性能台账也不得泄露问题文本：{}", body);

        let second = create_session(addr).await;
        take_turn_body(addr, &second, "未名科技公司的老板是谁").await;
        let (status, _, body) = raw_request(addr, "GET", "/metrics/semantic", &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let fleet: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(fleet["sessions"], 2, "{}", body);
        assert_eq!(fleet["unavailable_sessions"], 0, "{}", body);
        assert!(
            fleet["window"]["observations"].as_u64().unwrap_or(0) >= 2,
            "{}",
            body
        );
        assert!(
            fleet["window"]["answered"]["denominator"]
                .as_u64()
                .unwrap_or(0)
                >= 2,
            "{}",
            body
        );
        assert!(
            fleet["alerting_sessions"].is_object(),
            "告警聚合必须是对象：{}",
            body
        );
        assert!(!body.contains(question), "聚合指标不得泄露问题文本：{}", body);
        assert!(
            !body.contains("未名科技公司的老板是谁"),
            "聚合指标不得泄露问题文本：{}",
            body
        );

        let (status, _, _) = raw_request(
            addr,
            "GET",
            "/sessions/nope/semantic",
            &[],
            "",
        )
        .await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn read_traces_route_returns_bounded_receipts_without_raw_question() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/read-traces", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["session_id"], session_id);
        assert!(value["report"]
            .as_str()
            .is_some_and(|report| report.contains("读法台账")));
        assert_eq!(value["receipts"].as_array().map(Vec::len), Some(0));

        let question = "未名科技公司的老板是谁";
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Content-Type", "application/json")],
            &serde_json::json!({ "input": question }).to_string(),
        )
        .await;
        assert_eq!(status, 200, "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/read-traces", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let receipts = value["receipts"].as_array().expect("回执必须是数组");
        assert_eq!(receipts.len(), 1, "失败轮应留下一条回执：{}", body);
        assert_eq!(receipts[0]["answered"], false);
        assert!(
            receipts[0]["failure_kinds"]
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "unknown_entity")),
            "未知实体应被分类：{}",
            body
        );
        assert!(
            !body.contains(question),
            "读法回执不得泄露原始问句：{}",
            body
        );

        let (status, _, _) = raw_request(addr, "GET", "/sessions/nope/read-traces", &[], "").await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn recovery_mode_surfaces_epistemic_view() {
        let (addr, _server) = start_server_with_recovery(4, RecoveryMode::On);
        let session_id = create_session(addr).await;

        // 已知实体 + 图谱里没有的属性：On 模式必须给出认识论视图与回执
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明的体重是多少",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let epistemic = &value["epistemic"];
        assert_eq!(epistemic["status"], "absent", "{}", body);
        assert_eq!(epistemic["label"], "确无");
        assert!(
            epistemic["receipt"]
                .as_str()
                .is_some_and(|receipt| receipt.starts_with('r')),
            "缺记录必须有回执 id：{}",
            body
        );
        assert!(value["reply"]
            .as_str()
            .is_some_and(|reply| !reply.contains("暂时") || reply.contains("没有记录")));

        // 无失败的动作轮只携带当前视图
        let (_, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明的身高是多少",
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            value.get("epistemic").is_none(),
            "成功回答轮不得携带陈旧认识论视图：{}",
            body
        );

        // 让位（defer）也必须可命名：Unknown 回执而非无声丢弃
        let (_, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "随便聊聊吧",
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "deferred", "{}", body);
        assert_eq!(value["epistemic"]["status"], "unknown", "{}", body);
        assert!(value["epistemic"]["receipt"].as_str().is_some(), "{}", body);
    }

    #[tokio::test]
    async fn session_layers_expose_four_level_manifest() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        take_turn_body(addr, &session_id, "姚明是谁").await;

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/layers", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["session_id"], session_id);
        let manifest = &value["manifest"];
        assert_eq!(manifest["manifest_version"], 1);
        let layers = manifest["layers"].as_array().expect("layers 数组");
        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0]["id"], "L0");
        assert!(layers[0]["count"].as_u64().unwrap_or(0) > 0);
        assert_eq!(layers[3]["id"], "L3");
        assert_eq!(
            manifest["digest"].as_str().map(str::len),
            Some(16),
            "{}",
            body
        );
    }

    #[tokio::test]
    async fn session_lifecycle_and_focus() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明是谁",
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["source"], "dialogue");
        assert_eq!(value["focus"], "姚明");

        // 指代继承：第二轮用「他的身高呢」应命中同一焦点
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "他的身高呢",
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("226"), "{}", body);

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", session_id), &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["turns"], 2);
        assert_eq!(value["focus_stack"][0], "姚明");
        assert_eq!(value["working_memory"]["version"], 1);
        let packet = value["working_memory"]["items"]
            .as_array()
            .expect("快照应包含工作记忆上下文包");
        assert!(
            packet.iter().any(|item| item["kind"] == "focus"),
            "{}",
            body
        );
        assert!(
            packet.iter().any(|item| item["id"] == "turn:1"),
            "{}",
            body
        );

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200);
        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "你好",
        )
        .await;
        assert_eq!(status, 404);
    }

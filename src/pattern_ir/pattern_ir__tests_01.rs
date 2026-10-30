    use super::*;
    use crate::bindings::BoundTerm;
    use crate::create_prolog_rules;
    use crate::nlp::{analyze_query_intent, preprocess, TaggedWord};

    fn tagged(pairs: &[(&str, &str)]) -> Vec<TaggedWord> {
        pairs
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: text.to_string(),
                pos: pos.to_string(),
            })
            .collect()
    }

    fn sample_library() -> PatternLibrary {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        library.add(PatternEntry::new(
            "首都查询2",
            "config.toml#首都查询2",
            "capital_query",
            14,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::word("首都"),
            ],
        ));
        library.add(PatternEntry::new(
            "属性查询",
            "config.toml#属性查询",
            "property_query",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library
    }

    #[test]
    fn fingerprint_is_stable_and_route_sensitive() {
        let a = PatternEntry::new("x", "config.toml#x", "q1", 1, true, vec![Elem::pos("ns")]);
        let b = PatternEntry::new("x", "config.toml#x", "q1", 9, true, vec![Elem::pos("ns")]);
        // IR+路由+来源构成身份：优先级变化不改身份（同 id 原地替换）
        assert_eq!(a.id(), b.id());
        let c = PatternEntry::new("x", "config.toml#x", "q2", 1, true, vec![Elem::pos("ns")]);
        assert_ne!(a.id(), c.id());
        // IR 指纹只看 IR
        assert_eq!(a.ir_fingerprint(), c.ir_fingerprint());
        // 序列不同 → 指纹不同
        let d = PatternEntry::new(
            "x",
            "config.toml#x",
            "q1",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        );
        assert_ne!(a.ir_fingerprint(), d.ir_fingerprint());
        // 指纹稳定（可逐字节断言）
        assert_eq!(a.ir_fingerprint().len(), 16);
        assert_eq!(
            a.ir_fingerprint(),
            PatternEntry::new("y", "s", "q1", 0, true, vec![Elem::pos("ns")]).ir_fingerprint()
        );
    }

    #[test]
    fn config_and_teach_tokens_parse() {
        assert_eq!(parse_config_token("nr"), Elem::pos("nr"));
        assert_eq!(parse_config_token("首都"), Elem::word("首都"));
        assert_eq!(parse_config_token("*"), Elem::Any);
        assert_eq!(
            parse_teach_token("{城市:城市}").unwrap(),
            Elem::Slot {
                name: "城市".to_string(),
                sort: "城市".to_string(),
                hints: Vec::new(),
            }
        );
        assert_eq!(parse_teach_token("_").unwrap(), Elem::Any);
        assert!(parse_teach_token("{}").is_err());
        let elems = parse_teach_pattern("ns 的 首都").unwrap();
        assert_eq!(elems.len(), 3);
        assert!(parse_teach_pattern("   ").is_err());
    }

    #[test]
    fn matcher_binds_entities_with_spans() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
            ("？", "w"),
        ]);
        let m = library.match_route(&tokens).expect("应命中首都查询");
        assert_eq!(m.entry.name, "首都查询");
        assert_eq!(m.entities, vec!["法国"]);
        assert_eq!(
            m.bindings.get("entity.1").map(|t| t.render_raw()),
            Some(Some("法国".to_string()))
        );
        assert_eq!(m.span, Span::new(0, 5));
        assert_eq!(m.entity_spans, vec![Span::new(0, 1)]);
        // 覆盖末尾标点不参与
        assert_eq!(m.elem_spans.len(), 5);
    }

    #[test]
    fn matcher_merges_person_names_and_skips_punctuation() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "关系查询",
            "config.toml#关系查询",
            "relation_query",
            10,
            true,
            vec![
                Elem::pos("nr"),
                Elem::word("和"),
                Elem::pos("nr"),
                Elem::word("什么"),
                Elem::word("关系"),
            ],
        ));
        let tokens = tagged(&[
            ("张三", "nr"),
            ("和", "unknown"),
            ("李四", "nr"),
            ("什么", "r"),
            ("关系", "unknown"),
        ]);
        let m = library.match_route(&tokens).expect("关系模式应命中");
        assert_eq!(m.entities, vec!["张三", "李四"]);
        assert_eq!(m.entity_spans, vec![Span::new(0, 1), Span::new(2, 3)]);
    }

    #[test]
    fn person_merge_stops_before_next_word_element() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        // 结巴把它切成 `张三/nr 多/unknown 大/nr`：人名合并不能吞掉「多大」。
        let tokens = preprocess("张三多大", &config.nlp);
        assert_eq!(
            tokens
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["张三", "多", "大"]
        );
        let matched = library.match_route(&tokens).expect("年龄规则应命中");
        assert_eq!(matched.entry.query_type, "age_query");
        assert_eq!(matched.entities, vec!["张三".to_string()]);
    }

    #[test]
    fn leading_politeness_is_skipped_but_never_stolen_from_a_pattern() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        for question in ["请问张三多大", "麻烦，张三多大？", "那么张三多大"] {
            let tokens = preprocess(question, &config.nlp);
            let matched = library
                .match_route(&tokens)
                .unwrap_or_else(|| panic!("礼貌前缀不应破坏路由：{} {:?}", question, tokens));
            assert_eq!(matched.entry.query_type, "age_query", "{}", question);
        }

        // 模式首要素就是「请问」时，它必须自己消费，而不是被当作填充词跳过。
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "请问查询",
            "taught#test",
            "ask_query",
            10,
            true,
            vec![Elem::word("请问"), Elem::pos("nr")],
        ));
        let tokens = tagged(&[("请问", "v"), ("张三", "nr")]);
        let matched = library.match_route(&tokens).expect("首要素词面应保留");
        assert_eq!(matched.entities, vec!["张三".to_string()]);
    }

    #[test]
    fn multi_token_literal_is_joined_across_token_boundaries() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let tokens = preprocess("苹果和书有什么区别", &config.nlp);
        let matched = library.match_route(&tokens).expect("比较规则应命中");
        assert_eq!(matched.entry.query_type, "comparison_query");
        assert_eq!(matched.entities, vec!["苹果".to_string(), "书".to_string()]);
    }

    #[test]
    fn slot_bindings_come_from_sort_constraints() {
        let mut library = PatternLibrary::new();
        let domain = crate::nlu::default_domains()
            .into_iter()
            .find(|d| d.name == "天气")
            .expect("天气领域存在");
        let intent = domain
            .intents
            .iter()
            .find(|i| i.name == "天气查询")
            .unwrap();
        for pattern in &intent.patterns {
            library.add(PatternEntry::from_nlu_intent(&domain, intent, pattern));
        }
        let tokens = tagged(&[("厦门", "ns"), ("明天", "t"), ("天气", "n")]);
        // NLU 意图不参与路由：winner 为 None，但应出现在命中榜并带槽位绑定
        let report = library.rank(&tokens);
        assert!(report.winner.is_none(), "NLU 意图不接管查询路由");
        assert_eq!(report.matched.len(), 1, "意图应出现在命中榜");
        let entry = library.entries()[0].clone();
        let m = match_entry(&entry, &tokens).expect("意图应命中");
        assert_eq!(
            m.bindings.get("城市").and_then(|t| t.render_raw()),
            Some("厦门".to_string())
        );
        assert_eq!(
            m.bindings.get("日期").and_then(|t| t.render_raw()),
            Some("明天".to_string())
        );
    }

    #[test]
    fn nlu_intents_rank_but_do_not_route() {
        let mut library = PatternLibrary::new();
        let domain = crate::nlu::default_domains()
            .into_iter()
            .find(|d| d.name == "天气")
            .expect("天气领域存在");
        let intent = domain
            .intents
            .iter()
            .find(|i| i.name == "天气查询")
            .unwrap();
        library.add(PatternEntry::from_nlu_intent(
            &domain,
            intent,
            intent.patterns[0].as_str(),
        ));
        let tokens = tagged(&[("厦门", "ns"), ("明天", "t"), ("天气", "n")]);
        let report = library.rank(&tokens);
        assert!(report.winner.is_none(), "NLU 意图不参与路由");
        assert_eq!(report.matched.len(), 1, "但应出现在命中榜");
        assert!(!report.matched[0].route);
        // 槽位绑定在胜者之外：从命中榜拿不到解，需要 match_entry——
        // 直接评估该条目验证槽位排序
        let entry = library.entries()[0].clone();
        let m = match_entry(&entry, &tokens).expect("意图应命中");
        assert_eq!(
            m.bindings.get("城市").and_then(|t| t.render_raw()),
            Some("厦门".to_string())
        );
        assert_eq!(
            m.bindings.get("日期").and_then(|t| t.render_raw()),
            Some("明天".to_string())
        );
    }

    #[test]
    fn anchored_mismatch_reports_first_failed_element() {
        let library = sample_library();
        let tokens = tagged(&[("这", "r"), ("是", "v"), ("什么", "r")]);
        let report = library.rank(&tokens);
        assert!(report.winner.is_none());
        assert_eq!(report.failed.len(), 3);
        let reason = match &report.failed[0].status {
            CandidateStatus::Failed { reason } => reason.clone(),
            other => panic!("期望失败，得到 {:?}", other),
        };
        assert!(
            reason.contains("第1要素"),
            "原因应指出首个失配要素：{}",
            reason
        );
    }

    #[test]
    fn detailed_failure_exposes_frontier_not_just_a_string() {
        let entry = PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        );
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let failure = match_entry_detailed(&entry, &tokens).expect_err("「在」不应命中「是」");
        assert_eq!(failure.index, 3);
        assert_eq!(failure.kind, MatchFailureKind::WordMismatch);
        assert_eq!(failure.expected, "是");
        assert_eq!(failure.actual.as_deref(), Some("在"));
        assert_eq!(failure.actual_pos.as_deref(), Some("v"));
        assert_eq!(failure.covered_elems(), 3);
        assert!((failure.coverage() - 0.6).abs() < f64::EPSILON);
        assert_eq!(failure.matched_elems.len(), 3);
        assert_eq!(failure.remaining_tokens, 2);
        // 稳定的字符串投影：旧调用方继续拿到同一句原因。
        assert_eq!(failure.to_string(), "第4要素期望词「是」，实得「在」");
        assert!(failure.render().contains("此前 3/5 要素已覆盖"));
        // 5 个要素覆盖 3 个：按 0.6 阈值算近似命中，按 0.75 则不算。
        assert!(failure.is_near_miss(0.6));
        assert!(!failure.is_near_miss(0.75));
    }

    #[test]
    fn exhausted_input_reports_zero_remaining_tokens() {
        let entry = PatternEntry::new(
            "年龄查询",
            "config.toml#年龄查询",
            "age_query",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("多大")],
        );
        let tokens = tagged(&[("张三", "nr")]);
        let failure = match_entry_detailed(&entry, &tokens).expect_err("缺少「多大」");
        assert_eq!(failure.kind, MatchFailureKind::InputExhausted);
        assert_eq!(failure.index, 1);
        assert_eq!(failure.covered_elems(), 1);
        assert_eq!(failure.remaining_tokens, 0);
        assert_eq!(failure.to_string(), "第2要素「多大」未匹配：输入已耗尽");
    }

    #[test]
    fn config_literals_follow_synonym_normalization() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let synonyms = config
            .synonyms
            .as_ref()
            .expect("配置应包含同义词表")
            .mappings
            .clone();
        let tagged = preprocess("张三的老婆", &config.nlp);
        let normalized = crate::nlp::normalize_words(tagged, &synonyms);
        assert!(
            normalized.iter().any(|word| word.text == "妻子"),
            "输入应先归一化为规范词：{:?}",
            normalized
        );
        let matched = library
            .match_route(&normalized)
            .expect("配偶规则应命中归一化后的词面");
        assert_eq!(matched.entry.query_type, "spouse_query");
        assert_eq!(matched.entities, vec!["张三".to_string()]);
    }

    #[test]
    fn ranking_follows_priority_then_config_order() {
        let library = sample_library();
        let tokens = tagged(&[
            ("阿根廷", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let report = library.rank(&tokens);
        assert_eq!(report.matched.len(), 2, "两条首都模式都应命中");
        // 优先级 15 的完整模式排前
        assert_eq!(report.matched[0].name, "首都查询");
        assert_eq!(report.matched[1].name, "首都查询2");
        assert!(report.winner.is_some());
        // 短模式是长模式前缀以外的独立匹配：特化度不同
        if let CandidateStatus::Matched { specificity, .. } = report.matched[0].status {
            assert!((specificity - 1.0).abs() < f64::EPSILON);
        } else {
            panic!("期望命中状态");
        }
        // 同覆盖同路由 → 无冲突
        assert!(report.conflicts().is_empty());
    }

    #[test]
    fn teaching_adds_and_forgets_patterns() {
        let mut library = sample_library();
        let entry = library
            .teach("老家", "{人} 的 老家", "hometown_query", 20)
            .expect("教学应成功");
        assert_eq!(library.route_count(), 4);
        assert_eq!(
            library.get(&entry.short_id()).map(|e| e.name.as_str()),
            Some("老家")
        );
        let tokens = tagged(&[("姚明", "nr"), ("的", "u"), ("老家", "n")]);
        let m = library
            .match_route(&tokens)
            .expect("教学模式应立即参与路由");
        assert_eq!(m.entry.query_type, "hometown_query");
        assert_eq!(m.entities, vec!["姚明"]);
        // 同 id 重复教学 → 替换不新增
        assert!(!library.add(entry.clone()));
        assert_eq!(library.len(), 4);
        let removed = library.forget("老家").expect("应可遗忘");
        assert_eq!(removed.name, "老家");
        assert!(library.get("老家").is_none());
        // 只删教学来源
        assert!(library.forget("首都查询").is_none());
    }

    #[test]
    fn lint_flags_duplicate_conflict_and_shadow() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "泛化",
            "config.toml#a",
            "q_gen",
            10,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        library.add(PatternEntry::new(
            "特化",
            "config.toml#b",
            "q_spec",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        // IR 相同、路由不同 → 冲突
        library.add(PatternEntry::new(
            "复刻",
            "taught#会话",
            "q_other",
            3,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        // IR+路由+来源不同的完全重复（相对另一条同 IR 同路由）
        library.add(PatternEntry::new(
            "重复",
            "taught#会话2",
            "q_gen",
            9,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        let findings = lint(&library);
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Conflict),
            "应检出 IR 同路由不同"
        );
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Redundant),
            "应检出重复定义"
        );
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Shadowed),
            "应检出泛化遮蔽特化：{:?}",
            findings
        );
        // 遮蔽方向：泛化在前（优先 10 > 5）
        let shadow = findings
            .iter()
            .find(|f| f.kind == LintKind::Shadowed)
            .unwrap();
        assert_eq!(shadow.subject_name, "泛化");
        assert_eq!(shadow.object_name, "特化");
    }

    #[test]
    fn subsumption_is_conservative() {
        let general = PatternEntry::new(
            "g",
            "s",
            "q",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        );
        let specific = PatternEntry::new(
            "sp",
            "s",
            "q",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("的"), Elem::word("首都")],
        );
        assert!(!subsumes(&general, &specific), "词面错位不构成覆盖");
        let prefix = PatternEntry::new("p", "s", "q", 1, true, vec![Elem::pos("ns")]);
        assert!(subsumes(&prefix, &specific));
        assert!(!subsumes(&specific, &prefix), "长不覆盖短");
        assert!(!subsumes(&prefix, &prefix), "自身不构成严格覆盖");
    }

    #[test]
    fn relation_edges_mirror_lint_findings() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "泛化",
            "config.toml#a",
            "q_gen",
            10,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        library.add(PatternEntry::new(
            "特化",
            "config.toml#b",
            "q_spec",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "复刻",
            "taught#会话",
            "q_other",
            3,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let edges = relation_edges(library.entries());
        assert!(
            edges.iter().any(|e| e.kind == RelationKind::Generalizes),
            "应产出泛化边：{:?}",
            edges
        );
        assert!(
            edges.iter().any(|e| e.kind == RelationKind::Conflicts),
            "应产出冲突边：{:?}",
            edges
        );
        // 与 lint 一一对应（图视角就是体检结果的另一面）
        assert_eq!(edges.len(), lint(&library).len());
        for edge in &edges {
            assert_eq!(
                library.get(&edge.subject_id).map(|e| e.name.as_str()),
                Some(edge.subject_name.as_str())
            );
        }
    }

    #[test]
    fn pattern_quads_carry_relations_and_ignore_them_on_parse() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "A",
            "config.toml#a",
            "q1",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "B",
            "config.toml#b",
            "q2",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let quads = pattern_quads(library.entries()).expect("生成四元组");
        // IR 同、路由不同 → 冲突，对称关系双向各一条
        let conflicts: Vec<_> = quads
            .iter()
            .filter(|q| q.predicate.as_str() == P_CONFLICTS)
            .collect();
        assert_eq!(conflicts.len(), 2, "冲突关系应对称落图");
        // 关系边不干扰条目还原
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn relation_count_sparql_targets_pattern_graph() {
        let query = relation_count_sparql();
        assert!(query.contains(PATTERNS_GRAPH));
        assert!(query.contains(P_CONFLICTS));
        assert!(query.contains("GROUP BY"));
    }

    #[test]
    fn relation_report_lists_edges_and_query() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "A",
            "config.toml#a",
            "q1",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "B",
            "config.toml#b",
            "q2",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let report = render_relation_report(&library);
        assert!(report.contains(PATTERNS_GRAPH));
        assert!(report.contains("冲突"));
        assert!(report.contains("图体检查询"));
    }

    #[test]
    fn quads_roundtrip_preserves_entries() {
        let mut library = sample_library();
        library
            .teach("老家", "{人} 的 老家", "hometown_query", 20)
            .unwrap();
        let quads = pattern_quads(library.entries()).expect("生成四元组");
        assert!(!quads.is_empty());
        // 主语都在模式命名图
        for quad in &quads {
            assert_eq!(
                quad.graph_name,
                oxigraph::model::GraphName::NamedNode(
                    oxigraph::model::NamedNode::new(PATTERNS_GRAPH).unwrap()
                )
            );
        }
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored.len(), library.len());
        let mut left: Vec<_> = library.entries().to_vec();
        left.sort_by_key(PatternEntry::id);
        let mut right = restored;
        right.sort_by_key(PatternEntry::id);
        assert_eq!(left, right);
        // 物化戳与集合指纹
        assert_eq!(
            library_fingerprint(library.entries()),
            library_fingerprint(&right)
        );
    }

    #[test]
    fn nlu_intent_only_routes_as_fallback_when_config_misses() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let legacy_rules = create_prolog_rules(&config);
        assert!(
            library.fallback_count() > 0,
            "应有与配置规则对齐的 NLU 回退意图"
        );

        let tokens = preprocess("姚明年龄", &config.nlp);
        assert!(
            analyze_query_intent(&tokens, &legacy_rules).is_none(),
            "配置规则不应命中这个问法"
        );
        assert!(library.match_route(&tokens).is_none(), "主路由不应命中");
        let fallback = library
            .match_fallback(&tokens)
            .expect("NLU 意图应作为回退命中");
        assert_eq!(fallback.entry.query_type, "age_query");
        assert_eq!(fallback.entities, vec!["姚明".to_string()]);
        assert!(
            fallback.entry.source.starts_with("nlu#"),
            "{}",
            fallback.entry.source
        );

        // 配置能命中的问法仍走主路由，回退不改变既有行为。
        let configured = preprocess("小明的年龄", &config.nlp);
        assert!(library.match_route(&configured).is_some());
        assert_eq!(
            library.route(&configured),
            analyze_query_intent(&configured, &legacy_rules)
        );
    }

    #[test]
    fn route_parity_with_legacy_analyze_query_intent() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let legacy_rules = create_prolog_rules(&config);

        let corpus: Vec<&str> = vec![
            "法国的首都是哪里",
            "阿根廷首都",
            "小明多大",
            "小明的年龄",
            "张三和李四什么关系",
            "有多少个人",
            "水的化学分子式是什么",
            "红楼梦的作者是谁",
            "什么是光合作用",
            "解释一下什么是相对论",
            "太阳系中最大的行星是哪一颗",
            "北京的海拔高度是多少",
            "这是一句毫无意义的话",
            "你好",
            "谢谢",
        ];
        let mut agreements = 0usize;
        for input in corpus {
            for tokens in [preprocess(input, &config.nlp), manual_tokens(input)] {
                let legacy = analyze_query_intent(&tokens, &legacy_rules);
                let unified = library.route(&tokens);
                // 统一匹配器是旧匹配器的超集：旧路由能命中的必须逐字一致；
                // 旧匹配器因「复合词不拆」而漏掉的（如「海拔高度」），
                // 统一匹配器允许新增命中，这是本轮的有意改进。
                if let Some(legacy_route) = legacy {
                    assert_eq!(
                        Some(legacy_route),
                        unified,
                        "统一匹配器不得改变旧路由：输入「{}」",
                        input
                    );
                    agreements += 1;
                }
            }
        }
        assert!(
            agreements >= 8,
            "语料应覆盖足够多的命中样本，实得 {}",
            agreements
        );
    }

    #[test]
    fn first_elem_prefilter_is_conservative_and_reduces_scan() {
        let config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let context = PatternContext::default();
        let corpus = [
            "法国的首都是哪里",
            "姚明的年龄",
            "张三和李四什么关系",
            "水的化学分子式是什么",
            "有多少个人",
            "你好",
            "这是一句毫无意义的话",
        ];
        for input in corpus {
            let tokens = preprocess(input, &config.nlp);
            let scan = library.scan_route_with_context(&tokens, &context);
            let mut reference = None;
            for index in library.evaluation_order() {
                let entry = &library.entries[index];
                if !entry.route || !entry.context_satisfied(&context) {
                    continue;
                }
                if let Ok(matched) = match_entry(entry, &tokens) {
                    reference = Some(matched);
                    break;
                }
            }
            assert_eq!(
                scan.winner, reference,
                "预筛不得改变路由结果：输入「{}」",
                input
            );
            assert!(
                scan.scanned() <= library.route_count(),
                "扫描数不能超过可路由条目数：输入「{}」",
                input
            );
        }

        let tokens = preprocess("姚明的年龄", &config.nlp);
        let scan = library.scan_route_with_context(&tokens, &context);
        assert!(scan.winner.is_some(), "该问法应可路由");
        assert!(scan.skipped > 0, "首要素预筛应跳过部分条目");
        assert!(
            scan.evaluated < library.route_count(),
            "完整评估数应少于全库：{} >= {}",
            scan.evaluated,
            library.route_count()
        );
    }

    /// 无词典依赖的确定性 token 序列（仅覆盖语料中的高频词）。
    fn manual_tokens(input: &str) -> Vec<TaggedWord> {
        const LEXICON: &[(&str, &str)] = &[
            ("法国", "ns"),
            ("阿根廷", "ns"),
            ("北京", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
            ("小明", "nr"),
            ("多大", "r"),
            ("年龄", "n"),
            ("张三", "nr"),
            ("和", "unknown"),
            ("李四", "nr"),
            ("什么", "r"),
            ("关系", "unknown"),
            ("有", "v"),
            ("多少", "m"),
            ("个", "q"),
            ("人", "n"),
            ("水", "n"),
            ("化学", "n"),
            ("分子式", "n"),
            ("红楼梦", "n"),
            ("作者", "n"),
            ("谁", "r"),
            ("光合作用", "n"),
            ("相对论", "n"),
            ("解释", "v"),
            ("一下", "v"),
            ("太阳系", "ns"),
            ("中", "f"),
            ("最大", "a"),
            ("行星", "n"),
            ("哪一颗", "r"),
            ("海拔", "n"),
            ("高度", "n"),
            ("这", "r"),
            ("一", "m"),
            ("句", "q"),
            ("毫无", "d"),
            ("意义", "n"),
            ("话", "n"),
            ("你好", "l"),
            ("谢谢", "v"),
        ];
        // 词面直配（最长优先），未命中字符归入 unknown 单字
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let rest: String = chars[i..].iter().collect();
            let mut matched = false;
            let mut surfaces: Vec<&(&str, &str)> = LEXICON.iter().collect();
            surfaces.sort_by_key(|(text, _)| std::cmp::Reverse(text.chars().count()));
            for &(text, pos) in surfaces {
                if rest.starts_with(text) {
                    tokens.push(TaggedWord {
                        text: text.to_string(),
                        pos: pos.to_string(),
                    });
                    i += text.chars().count();
                    matched = true;
                    break;
                }
            }
            if !matched {
                tokens.push(TaggedWord {
                    text: chars[i].to_string(),
                    pos: "unknown".to_string(),
                });
                i += 1;
            }
        }
        tokens
    }

    #[test]
    fn report_renders_winner_bindings_and_reasons() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let report = library.rank(&tokens);
        let text = render_match_report("法国的首都是哪里", &report);
        assert!(text.contains("首都查询"), "{}", text);
        assert!(text.contains("entity.1=法国"), "绑定应带区间证据：{}", text);
        assert!(text.contains("未命中"), "失败原因应展示：{}", text);
        let empty = library.rank(&tagged(&[("这", "r")]));
        let empty_text = render_match_report("这", &empty);
        assert!(empty_text.contains("无可路由命中"), "{}", empty_text);
        let lib_text = render_library_report(&library);
        assert!(lib_text.contains("模式库：3 条"), "{}", lib_text);
    }

    #[test]
    fn bound_terms_are_typed_in_bindings() {
        let mut library = PatternLibrary::new();
        library
            .teach("计数", "有 多少 {物}", "count_query", 8)
            .unwrap();
        let tokens = tagged(&[("有", "v"), ("多少", "m"), ("人", "n")]);
        let m = library.match_route(&tokens).expect("应命中");
        let term = m.bindings.get("物").expect("槽位应绑定");
        // BindingSet::bind 走分类：文本按数值/文本归类
        assert_eq!(term.render_raw().as_deref(), Some("人"));
        assert!(matches!(term, BoundTerm::Text { .. }));
    }

    #[test]
    fn episodes_merge_repeats_and_evict_by_recency() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let m = library.match_route(&tokens).expect("应命中");
        let mut log = PatternEpisodeLog::with_capacity(3);
        let first = log.record_match("法国的首都是哪里", &tokens, &m, Some("法国"), 2);
        assert_eq!(first.count, 1);
        assert_eq!(first.outcome, EpisodeOutcome::Matched);
        let again = log.record_match("法国的首都是哪里", &tokens, &m, Some("法国"), 3);
        assert_eq!(again.count, 2, "同一经验应累计支持度");
        assert_eq!(log.len(), 1, "相同经验不重复占容量");
        assert_eq!(log.clock(), 3);

        for turn in 4..8 {
            log.record_unmatched(
                &format!("未知句子{}", turn),
                &tagged(&[("未知", "n")]),
                None,
                turn,
            );
        }
        assert_eq!(log.len(), 3, "容量必须封顶");
        assert_eq!(log.matched_count(), 1);
        assert_eq!(log.unmatched_count(), 2);
        assert!(
            log.entries()
                .iter()
                .any(|episode| episode.input == "法国的首都是哪里"),
            "最近使用的经验不应被淘汰"
        );
        let report = render_episode_report(&log);
        assert!(report.contains("模式经验：3 条"), "{}", report);
        assert!(report.contains("×2"), "{}", report);
    }

    #[test]
    fn unmatched_episode_keeps_the_decision_frontier() {
        let library = sample_library();
        let entry = library.entries()[0].clone();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let failure = match_entry_detailed(&entry, &tokens).expect_err("「在」不应命中「是」");
        let snapshot = EpisodeFrontier {
            pattern_id: entry.id(),
            pattern_name: entry.name.clone(),
            query_type: entry.query_type.clone(),
            coverage: failure.coverage(),
            failure,
        };
        let mut log = PatternEpisodeLog::with_capacity(4);
        let episode = log.record_unmatched_with_frontier(
            "法国的首都在哪里",
            &tokens,
            &PatternContext::default(),
            Some(snapshot.clone()),
        );
        assert_eq!(episode.frontier.as_ref(), Some(&snapshot));
        assert_eq!(episode.outcome, EpisodeOutcome::Unmatched);

        // 落图/序列化后前沿仍然可恢复；旧账本没有该字段也应按 None 读入。
        let json = serde_json::to_string(&episode).expect("序列化经验");
        let restored: PatternEpisode = serde_json::from_str(&json).expect("反序列化经验");
        assert_eq!(restored.frontier.as_ref(), Some(&snapshot));
        let legacy: PatternEpisode =
            serde_json::from_str(r#"{"input":"旧句","tokens":[],"outcome":"unmatched"}"#)
                .expect("旧账本应兼容");
        assert!(legacy.frontier.is_none());

        let report = render_episode_report(&log);
        assert!(report.contains("最近受阻"), "{}", report);
        assert!(report.contains("第4要素"), "{}", report);
    }

    #[test]
    fn episode_quads_roundtrip_in_isolated_graph() {
        let tokens = tagged(&[("这", "r"), ("是", "v")]);
        let mut log = PatternEpisodeLog::with_capacity(8);
        let left = log.record_unmatched("这是", &tokens, Some("焦点"), 1);
        let right = log.record_unmatched("那是", &tokens, None, 2);
        let quads = episode_quads(log.entries()).expect("生成经验四元组");
        assert_eq!(
            quads[0].graph_name,
            oxigraph::model::GraphName::NamedNode(
                oxigraph::model::NamedNode::new(EPISODES_GRAPH).unwrap()
            )
        );
        let restored = parse_episode_graph(quads.into_iter());
        assert_eq!(restored.len(), 2);
        let mut ids: Vec<String> = restored.iter().map(PatternEpisode::id).collect();
        ids.sort();
        let mut expected = vec![left.id(), right.id()];
        expected.sort();
        assert_eq!(ids, expected);
        assert_eq!(
            episode_fingerprint(&restored),
            episode_fingerprint(log.entries()),
            "经验集指纹与插入顺序无关"
        );
    }

    #[test]
    fn case_suggestion_generalizes_unseen_entity_but_stays_inactive() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);

        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("相似经验应给出候选骨架");
        assert!(suggestion.score >= 0.2);
        assert_eq!(suggestion.episode.winner_name.as_deref(), Some("首都查询"));
        assert!(!suggestion.pattern.route, "候选不得直接参与路由");
        assert_eq!(
            suggestion.pattern.source,
            format!("case#{}", suggestion.episode.short_id())
        );
        assert!(
            suggestion.pattern.render().contains("槽1:地点"),
            "未见实体应升为地点槽：{}",
            suggestion.pattern.render()
        );
        assert!(
            match_entry(&suggestion.pattern, &current).is_ok(),
            "候选必须能解释当前输入"
        );
        let report = render_case_suggestion("德国的首都在哪里", &suggestion);
        assert!(report.contains("未激活"), "{}", report);
        assert!(report.contains("确认方式"), "{}", report);

        let unmatched = tagged(&[("今天", "t"), ("心情", "n"), ("不错", "a")]);
        assert!(
            log.suggest_case(&unmatched, None, 0.2).is_none(),
            "无共享词且无焦点时不应硬造候选"
        );
    }

    #[test]
    fn accepted_case_becomes_routable_taught_pattern() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);

        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("相似经验应给出候选骨架");
        assert!(library.match_route(&current).is_none(), "确认前不应路由");

        let mut active = library.clone();
        let entry = active.accept_case(&suggestion).expect("确认案例应成功");
        assert!(entry.route, "确认后必须可路由");
        assert_eq!(entry.query_type, "capital_query");
        assert!(entry.source.starts_with("taught#案例#"), "{}", entry.source);
        assert_eq!(
            active.accept_case(&suggestion).expect("重复确认幂等"),
            entry
        );
        assert_eq!(
            active.count_with_source_prefix("taught#"),
            1,
            "重复确认不得重复登记"
        );
        let routed = active.match_route(&current).expect("确认后应命中新实体");
        assert_eq!(routed.entry.name, entry.name);
        assert_eq!(routed.entities, vec!["德国".to_string()]);

        let quads = pattern_quads(active.entries()).expect("确认模式应可落图");
        let restored = parse_pattern_graph(quads.into_iter());
        let loaded = restored
            .iter()
            .find(|candidate| candidate.id() == entry.id())
            .expect("确认模式应随账本恢复");
        assert!(loaded.route);
        assert_eq!(loaded.source, entry.source);

        assert!(active.forget(&entry.name).is_some(), "确认模式应可遗忘");
    }

    #[test]
    fn episode_context_scopes_retrieval_and_suggestion() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        let asked = PatternContext {
            focus: Some("法国".to_string()),
            focus_type: Some("地点".to_string()),
            last_move: Some("ask_attribute".to_string()),
            pending_clarify: false,
            turn: 3,
        };
        let greet = PatternContext {
            focus: Some("法国".to_string()),
            focus_type: Some("地点".to_string()),
            last_move: Some("greet".to_string()),
            pending_clarify: false,
            turn: 4,
        };
        log.record_match_with_context("法国的首都是哪里", &stored, &matched, &asked);
        log.record_match_with_context("法国的首都是哪里", &stored, &matched, &greet);
        assert_eq!(log.len(), 2, "不同上轮动作应保留为不同上下文经验");
        assert_ne!(log.entries()[0].id(), log.entries()[1].id());

        let neighbors = log.nearest_with_context(&stored, &asked, 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(
            log.entries()[neighbors[0].index].last_move.as_deref(),
            Some("ask_attribute"),
            "上下文匹配的经验应排前"
        );
        let suggestion = log
            .suggest_case_with_context(&stored, &greet, 0.2)
            .expect("相似成功经验应给出候选");
        assert_eq!(suggestion.episode.last_move.as_deref(), Some("greet"));
        let report = render_case_suggestion("法国的首都是哪里", &suggestion);
        assert!(report.contains("发生上下文"), "{}", report);
        assert!(report.contains("上轮 greet"), "{}", report);
        let episode_report = render_episode_report(&log);
        assert!(
            episode_report.contains("上轮 ask_attribute"),
            "{}",
            episode_report
        );
        assert!(
            episode_report.contains("焦点 法国(地点)"),
            "{}",
            episode_report
        );
    }

    #[test]
    fn elliptical_case_suggestion_keeps_focus_guard_after_acceptance() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "身高查询",
            "config.toml#身高查询",
            "height_query",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::word("身高")],
        ));
        let stored = tagged(&[("姚明", "nr"), ("的", "u"), ("身高", "n")]);
        let matched = library.match_route(&stored).expect("应命中");
        let focus_context = PatternContext {
            focus: Some("姚明".to_string()),
            focus_type: Some("人物".to_string()),
            last_move: Some("ask_attribute".to_string()),
            ..PatternContext::default()
        };
        let mut log = PatternEpisodeLog::default();
        log.record_match_with_context("姚明的身高", &stored, &matched, &focus_context);

        let elliptical = tagged(&[("身高", "n"), ("呢", "u")]);
        let suggestion = log
            .suggest_case_with_context(&elliptical, &focus_context, 0.2)
            .expect("相似经验应给出省略问法候选");
        assert_eq!(
            suggestion.pattern.guard,
            Some(ContextGuard::focus_type("人物")),
            "省略问法候选必须带焦点类型守卫"
        );
        let mut active = library.clone();
        active.accept_case(&suggestion).expect("确认应成功");
        assert!(
            active.match_route(&elliptical).is_none(),
            "无焦点时守卫必须拦住省略问法"
        );
        assert_eq!(
            active.route_with_context(&elliptical, &focus_context),
            Some(("height_query".to_string(), Vec::new()))
        );
    }

    #[test]
    fn context_guard_parses_and_gates_routing() {
        assert_eq!(parse_context_guard("焦点").unwrap(), ContextGuard::focus());
        assert_eq!(
            parse_context_guard("焦点类型：人物").unwrap(),
            ContextGuard::focus_type("人物")
        );
        assert_eq!(
            parse_context_guard("上轮=ask_attribute；无待澄清").unwrap(),
            ContextGuard {
                focus_required: false,
                focus_type: None,
                last_move: Some("ask_attribute".to_string()),
                pending_clarify: Some(false),
            }
        );
        assert!(parse_context_guard("").is_err());
        assert!(parse_context_guard("未知守卫").is_err());

        let mut library = PatternLibrary::new();
        library
            .teach_with_guard(
                "省略身高",
                "身高 呢",
                "height_query",
                30,
                Some(ContextGuard::focus()),
            )
            .unwrap();
        let tokens = tagged(&[("身高", "n"), ("呢", "u")]);
        assert!(
            library.match_route(&tokens).is_none(),
            "无焦点时守卫必须拦住省略问句"
        );
        let blocked = library.rank(&tokens);
        assert!(blocked.winner.is_none());
        assert!(
            blocked.failed.iter().any(|outcome| matches!(
                &outcome.status,
                CandidateStatus::Failed { reason } if reason.contains("上下文守卫")
            )),
            "失败榜应披露守卫原因"
        );
        let context = PatternContext {
            focus: Some("姚明".to_string()),
            focus_type: Some("人物".to_string()),
            ..PatternContext::default()
        };
        assert_eq!(
            library.route_with_context(&tokens, &context),
            Some(("height_query".to_string(), Vec::new()))
        );
        let report = render_match_report("身高呢", &library.rank_with_context(&tokens, &context));
        assert!(report.contains("上下文守卫：焦点"), "{}", report);

        let guarded = &library.entries()[0];
        let quads = pattern_quads(library.entries()).expect("守卫条目应可落图");
        assert!(
            quads.iter().any(|quad| quad.predicate.as_str() == P_GUARD),
            "守卫应作为可查询字面量落图"
        );
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored[0].guard, guarded.guard);
    }

    #[test]
    fn case_reply_only_accepts_whole_short_answers() {
        for answer in ["是", "是的。", "对呀", "好的", "确认", "YES", "没错啊"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Accept,
                "「{}」应视为确认",
                answer
            );
        }
        for answer in ["不是", "不对！", "不", "没有", "算了", "不要", "no", "否吧"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Reject,
                "「{}」应视为否定",
                answer
            );
        }
        for answer in ["不知道", "不确定", "跳过", "下一个"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Unclear,
                "「{}」应视为暂不确定",
                answer
            );
        }
        for answer in ["是什么", "是的，首都在哪里", "帮助", "", "确认联想：德国"]
        {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::NewInput,
                "「{}」不是对联想提问的短答",
                answer
            );
        }
    }

    #[test]
    fn rejected_case_is_not_reoffered_and_feedback_persists() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);
        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("首次应给出候选");
        assert!(log.reject_case(&suggestion), "否定应命中经验");
        assert!(
            log.suggest_case(&current, Some("法国"), 0.2).is_none(),
            "净否定后的骨架不应再次主动推荐"
        );
        let latest = log.entries().last().expect("经验仍在账本中");
        assert_eq!(latest.rejections, 1);
        let report = render_episode_report(&log);
        assert!(report.contains("否定×1"), "{}", report);

        let quads = episode_quads(log.entries()).expect("反馈计数应可落图");
        let restored = parse_episode_graph(quads.into_iter());
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].rejections, 1);

        assert!(log.confirm_case(&suggestion), "确认也应命中经验");
        assert_eq!(
            log.entries().last().map(|episode| episode.acceptances),
            Some(1)
        );
        assert!(
            log.suggest_case(&current, Some("法国"), 0.2).is_some(),
            "确认抵消净否定后应允许再次联想"
        );
    }

    #[test]
    fn forget_removes_single_episode_by_short_id_or_exact_input() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        let episode = log.record_match("法国的首都是哪里", &stored, &matched, None, 1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.forget("deadbeef"), 0, "未知 id 不应误删");
        assert_eq!(log.forget(&episode.short_id()), 1);
        assert!(log.is_empty(), "按短 id 应删除单条经验");

        log.record_match("法国的首都是哪里", &stored, &matched, None, 2);
        assert_eq!(log.forget("法国的首都是哪里"), 1);
        assert!(log.is_empty(), "按完整原句也应删除单条经验");
    }

    #[test]
    fn episode_revision_advances_on_mutation_but_not_on_lookup() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&tokens).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        assert_eq!(log.revision(), 0);
        let episode = log.record_match("法国的首都是哪里", &tokens, &matched, None, 1);
        let after_record = log.revision();
        assert!(after_record > 0, "记录经验应推进版本");
        let _ = log.nearest(&tokens, None, 5);
        assert_eq!(log.revision(), after_record, "只读检索不应推进版本");
        assert_eq!(log.forget(&episode.short_id()), 1);
        assert!(log.revision() > after_record, "删除经验应推进版本");
    }

    #[test]
    fn config_layer_replacement_keeps_session_patterns_and_recompiles_nlu() {
        let mut config: SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let mut library = PatternLibrary::from_config(&config);
        library
            .teach("会话规则", "nr 今年 多大", "age_query", 30)
            .expect("教学规则应可编译");
        library.add(PatternEntry::new(
            "锻造规则",
            "induced#forge#test",
            "age_query",
            25,
            true,
            vec![Elem::pos("nr"), Elem::word("几岁")],
        ));
        let taught_before = library.entries().len();
        let removed_rule = config.prolog_rules.remove(0).name;
        let refresh = library.replace_config_layer(&config);

        assert!(refresh.removed > 0);
        assert!(refresh.added > 0);
        assert_eq!(refresh.kept, taught_before - refresh.removed);
        assert!(
            library.entries().iter().all(|entry| entry.name != removed_rule),
            "被删掉的配置规则不应在重建后复活"
        );
        assert!(library
            .entries()
            .iter()
            .any(|entry| entry.source == "taught#会话"));
        assert!(library
            .entries()
            .iter()
            .any(|entry| entry.source == "induced#forge#test"));
        assert_eq!(
            library.count_with_source_prefix("config.toml#"),
            config.prolog_rules.len()
        );
        assert!(
            library.count_with_source_prefix("nlu#") > 0,
            "NLU 回退层应随配置重建"
        );
    }

    #[test]
    fn pos_ontology_closure_extends_pos_and_entity_slots() {
        let entry = PatternEntry::new(
            "作者查询",
            "config.toml#作者查询",
            "author_query",
            10,
            true,
            vec![
                Elem::pos("n"),
                Elem::word("的"),
                Elem::word("作者"),
                Elem::word("是"),
                Elem::word("谁"),
            ],
        );
        let tokens = tagged(&[
            ("鲁迅", "nr"),
            ("的", "u"),
            ("作者", "n"),
            ("是", "v"),
            ("谁", "r"),
        ]);
        let matched = match_entry(&entry, &tokens).expect("n 应覆盖 nr");
        assert_eq!(matched.entities, vec!["鲁迅".to_string()]);

        let slot = PatternEntry::new(
            "实体槽",
            "test",
            "entity",
            1,
            true,
            vec![Elem::slot("实体", "实体")],
        );
        let tokens = tagged(&[("清华大学", "nt")]);
        let matched = match_entry(&slot, &tokens).expect("实体槽应接受 nt");
        assert_eq!(
            matched.bindings.get("实体").and_then(|t| t.render_raw()),
            Some("清华大学".to_string())
        );

        let time = tagged(&[("今天", "t")]);
        assert!(
            match_entry(&slot, &time).is_err(),
            "纯时间词不应被当成实体槽"
        );
    }

#[cfg(test)]
mod tests {
    use super::*;

    fn build_retriever() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_entity(
            "张三",
            "人",
            &[
                ("年龄".to_string(), "30".to_string()),
                ("职业".to_string(), "演员".to_string()),
                ("国籍".to_string(), "中国".to_string()),
            ],
        );
        retriever.observe_entity(
            "李四",
            "人",
            &[
                ("年龄".to_string(), "28".to_string()),
                ("职业".to_string(), "导演".to_string()),
                ("国籍".to_string(), "德国".to_string()),
            ],
        );
        retriever.observe_triple("张三", "老师", "王五");
        retriever.observe_triple("李四", "老师", "王五");
        retriever.observe_triple("王五", "职业", "教授");
        retriever
    }

    #[test]
    fn retrieves_by_predicate_and_entity() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("张三的年龄是多少", 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].subject, "张三");
        assert_eq!(hits[0].predicate, "年龄");
        assert_eq!(hits[0].object, "30");
        assert!(hits[0].predicate_match > 0.0);
    }

    #[test]
    fn structural_signal_links_neighbors() {
        let mut retriever = build_retriever();
        // 「王五」不在查询词面中，但结构上游走可以召回王五的证据
        let hits = retriever.retrieve("张三的老师是谁", 10);
        assert!(hits.iter().any(|hit| hit.subject == "王五"));
    }

    #[test]
    fn explicit_entity_beats_unrelated_doc() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("李四 国籍", 5);
        assert_eq!(hits[0].subject, "李四");
        assert_eq!(hits[0].predicate, "国籍");
    }

    #[test]
    fn empty_or_unknown_query_yields_no_or_low_hits() {
        let mut retriever = build_retriever();
        assert!(retriever.retrieve("", 5).is_empty());
        let hits = retriever.retrieve("量子色动力学", 5);
        assert!(hits.is_empty());
    }

    #[test]
    fn format_hits_contains_breakdown() {
        let mut retriever = build_retriever();
        let hits = retriever.retrieve("张三的年龄是多少", 2);
        let text = retriever.format_hits("张三的年龄是多少", &hits);
        assert!(text.contains("词面"));
        assert!(text.contains("结构"));
        assert!(text.contains("张三"));
    }

    #[test]
    fn deduplicates_repeated_observations() {
        let mut retriever = DenseRetriever::new(5, 0.85, 8);
        retriever.observe_triple("张三", "年龄", "30");
        retriever.observe_triple("张三", "年龄", "30");
        assert_eq!(retriever.doc_count(), 1);
    }

    #[test]
    fn repeated_search_reuses_exact_result_without_recomputing() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints::default();
        let first = retriever.search("张三的年龄", 5, &constraints, SearchBudget::unlimited());
        assert_eq!(retriever.cache_stats(), (0, 1));
        let second = retriever.search("张三的年龄", 5, &constraints, SearchBudget::unlimited());
        assert_eq!(retriever.cache_stats(), (1, 1));
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap(),
            "缓存命中必须与重新计算逐字节一致"
        );
        // 不同 k 是不同的逻辑请求，必须分别计算。
        let _ = retriever.search("张三的年龄", 3, &constraints, SearchBudget::unlimited());
        assert_eq!(retriever.cache_stats(), (1, 2));
    }

    #[test]
    fn evidence_mutation_invalidates_cached_results() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            predicate: Some("老师".to_string()),
            ..SearchConstraints::default()
        };
        let before = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(!before
            .hits
            .iter()
            .any(|hit| hit.subject == "赵六" && hit.predicate == "老师"));
        retriever.observe_triple("赵六", "老师", "王五");
        let after = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(after
            .hits
            .iter()
            .any(|hit| hit.subject == "赵六" && hit.predicate == "老师"));
        assert_eq!(retriever.cache_stats(), (0, 2), "写入后必须按未命中重算");
    }

    #[test]
    fn schema_mutation_invalidates_cached_results() {
        let mut retriever = DenseRetriever::new(8, 0.85, 8);
        retriever.observe_entity("故宫", "地点", &[]);
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let before = retriever.search("", 5, &constraints, SearchBudget::unlimited());
        assert!(
            before.hits.iter().any(|hit| hit.subject == "故宫"),
            "无互斥证据时开放世界应相容：{:?}",
            before.hits
        );
        retriever.observe_disjoint("地点", "人物");
        let after = retriever.search("", 5, &constraints, SearchBudget::unlimited());
        assert!(
            !after.hits.iter().any(|hit| hit.subject == "故宫"),
            "互斥类型登记后旧缓存必须失效：{:?}",
            after.hits
        );
        assert_eq!(retriever.cache_stats(), (0, 2));
    }

    fn exhaustive_hits(retriever: &DenseRetriever, query: &str, k: usize) -> Vec<RetrievalHit> {
        let seeds = retriever.seeds(query);
        let tokens = tokenize(query);
        let query_vector = retriever.query_vector(&tokens, &seeds);
        let query_norm = DenseRetriever::query_norm(&query_vector).max(1e-9);
        let structural = retriever.structural_scores(&seeds);
        let query_tokens: HashSet<String> = tokens.into_iter().collect();
        let total = retriever.docs.len();
        let dot_terms: Vec<(&String, f64)> = query_vector
            .iter()
            .map(|(term, weight)| {
                let idf = if retriever.df.contains_key(term) {
                    retriever.idf(term, total)
                } else {
                    (1.0 + total as f64).ln()
                };
                (term, weight * idf)
            })
            .collect();
        let mut hits: Vec<RetrievalHit> = (0..retriever.docs.len())
            .filter_map(|id| {
                retriever.score_doc(
                    id,
                    &dot_terms,
                    &query_tokens,
                    query,
                    query_norm,
                    &structural,
                )
            })
            .collect();
        hits.sort_by(DenseRetriever::compare_hits);
        hits.truncate(k);
        hits
    }

    #[test]
    fn bounded_search_matches_exhaustive_scoring() {
        let mut retriever = build_retriever();
        retriever.rebuild();
        let search = retriever.search(
            "张三的年龄是多少",
            5,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        let exhaustive = exhaustive_hits(&retriever, "张三的年龄是多少", 5);
        let keys = |hits: &[RetrievalHit]| {
            hits.iter()
                .map(|hit| {
                    (
                        hit.subject.clone(),
                        hit.predicate.clone(),
                        hit.object.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&search.hits), keys(&exhaustive));
        assert!(search
            .hits
            .iter()
            .zip(exhaustive.iter())
            .all(|(left, right)| (left.score - right.score).abs() < 1e-9));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn upper_bound_dominates_exact_score() {
        let mut retriever = build_retriever();
        retriever.rebuild();
        let query = "张三的年龄是多少";
        let seeds = retriever.seeds(query);
        let tokens = tokenize(query);
        let query_vector = retriever.query_vector(&tokens, &seeds);
        let query_norm = DenseRetriever::query_norm(&query_vector).max(1e-9);
        let structural = retriever.structural_scores(&seeds);
        let max_structural = structural.values().cloned().fold(0.0_f64, f64::max);
        let query_tokens: HashSet<String> = tokens.into_iter().collect();
        let total = retriever.docs.len();
        let dot_terms: Vec<(&String, f64)> = query_vector
            .iter()
            .map(|(term, weight)| {
                let idf = if retriever.df.contains_key(term) {
                    retriever.idf(term, total)
                } else {
                    (1.0 + total as f64).ln()
                };
                (term, weight * idf)
            })
            .collect();
        let dot_weight_sum: f64 = dot_terms.iter().map(|(_, weight)| *weight).sum();
        for id in 0..retriever.docs.len() {
            let upper = retriever.doc_upper_bound(
                id,
                dot_weight_sum,
                query_norm,
                &structural,
                max_structural,
                query,
                &query_tokens,
            );
            if let Some(hit) = retriever.score_doc(
                id,
                &dot_terms,
                &query_tokens,
                query,
                query_norm,
                &structural,
            ) {
                assert!(
                    hit.score <= upper + 1e-9,
                    "上界 {upper} 低于精确分数 {}（文档 {id}）",
                    hit.score
                );
            }
        }
    }

    #[test]
    fn constraints_filter_before_scoring_and_certify() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            subject: Some("张三".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("年龄", 5, &constraints, SearchBudget::unlimited());
        assert!(!search.hits.is_empty());
        assert!(search.hits.iter().all(|hit| hit.subject == "张三"));
        assert!(search.receipt.constraint_pruned > 0);
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn type_constraint_uses_declared_entity_types() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            predicate: Some("职业".to_string()),
            entity_type: Some("人".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(subjects.contains("张三"));
        assert!(subjects.contains("李四"));
        assert!(!subjects.contains("王五"));
        assert!(search.hits.iter().all(|hit| hit.predicate == "职业"));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn empty_query_returns_constraint_lookup() {
        let mut retriever = build_retriever();
        let constraints = SearchConstraints {
            predicate: Some("老师".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert_eq!(search.hits.len(), 2);
        assert!(search.hits.iter().all(|hit| hit.predicate == "老师"));
        assert!(search
            .hits
            .iter()
            .all(|hit| hit.reasons == vec!["约束精确匹配"]));
        assert!(search.receipt.certified_top_k);
    }

    #[test]
    fn removal_drops_evidence_and_invalidates_index() {
        let mut retriever = build_retriever();
        assert!(retriever.remove_triple("张三", "老师", "王五"));
        assert!(!retriever.remove_triple("张三", "老师", "王五"));
        let removed = retriever.search(
            "张三 老师 王五",
            10,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(removed.hits.iter().all(|hit| !(hit.subject == "张三"
            && hit.predicate == "老师"
            && hit.object == "王五")));

        retriever.observe_triple("张三", "老师", "王五");
        let restored = retriever.search(
            "张三 老师 王五",
            10,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(restored
            .hits
            .iter()
            .any(|hit| hit.subject == "张三" && hit.predicate == "老师" && hit.object == "王五"));
    }

    #[test]
    fn removing_type_evidence_unregisters_declared_type() {
        let mut retriever = build_retriever();
        assert!(retriever.remove_triple("张三", "类型", "人"));
        let constraints = SearchConstraints {
            predicate: Some("职业".to_string()),
            entity_type: Some("人".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(!subjects.contains("张三"));
        assert!(subjects.contains("李四"));
    }

    #[test]
    fn budget_exhaustion_is_explicitly_uncertified() {
        let mut retriever = build_retriever();
        let search = retriever.search(
            "张三",
            3,
            &SearchConstraints::default(),
            SearchBudget::scanning(1),
        );
        assert!(search.receipt.budget_exhausted);
        assert!(!search.receipt.certified_top_k);
        assert!(search.receipt.scanned <= 1);
        assert!(search
            .receipt
            .notes
            .iter()
            .any(|note| note.contains("预算")));
    }

    fn rdfs_retriever() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(8, 0.85, 12);
        retriever.observe_entity("张三", "学生", &[("专业".to_string(), "哲学".to_string())]);
        retriever.observe_entity(
            "李四",
            "教师",
            &[("专业".to_string(), "逻辑学".to_string())],
        );
        retriever.observe_entity("王五", "地点", &[("名称".to_string(), "长安".to_string())]);
        retriever.observe_subclass("学生", "人物");
        retriever.observe_subclass("教师", "人物");
        retriever.observe_disjoint("地点", "人物");
        retriever
    }

    #[test]
    fn type_pool_expands_to_subclasses_and_excludes_disjoint() {
        let retriever = rdfs_retriever();
        let pool = retriever.type_pool("人物");
        assert_eq!(pool, vec!["学生".to_string(), "教师".to_string()]);
        assert!(
            retriever.subclass_reachable("学生", "人物"),
            "子类可达性应成立"
        );
        assert!(retriever.type_compatible("学生", "人物"), "子类与父类相容");
        assert!(retriever.type_compatible("学生", "学生"), "自反相容");
        assert!(
            !retriever.type_compatible("地点", "人物"),
            "互斥证据是唯一硬剪"
        );
    }

    #[test]
    fn type_constraint_matches_subclass_declared_types() {
        let mut retriever = rdfs_retriever();
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(
            subjects.contains("张三"),
            "学生 ⊑ 人物 应命中：{:?}",
            subjects
        );
        assert!(
            subjects.contains("李四"),
            "教师 ⊑ 人物 应命中：{:?}",
            subjects
        );
        assert!(
            !subjects.contains("王五"),
            "互斥类型（地点）必须被排除：{:?}",
            subjects
        );
        assert!(
            search
                .receipt
                .notes
                .iter()
                .any(|note| note.contains("相容类型池")),
            "类型池展开应写进回执：{:?}",
            search.receipt.notes
        );
    }

    #[test]
    fn unknown_declared_types_are_still_pruned() {
        let mut retriever = rdfs_retriever();
        // 主客体都未声明类型：不臆造类型，保守剪掉
        retriever.observe_triple("赵六", "同事", "钱七");
        let constraints = SearchConstraints {
            entity_type: Some("人物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search
                .hits
                .iter()
                .all(|hit| hit.subject != "赵六" && hit.object != "赵六"),
            "未声明类型不得混入类型约束结果"
        );
        // 一端有相容类型即可命中（主体或客体任一匹配）
        retriever.observe_entity("钱七", "学生", &[]);
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search
                .hits
                .iter()
                .any(|hit| hit.subject == "赵六" && hit.object == "钱七"),
            "客体类型相容（学生 ⊑ 人物）时应命中"
        );
    }

    #[test]
    fn subclass_predicates_from_triples_feed_hierarchy() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_entity("苏格拉底", "人", &[]);
        // 图谱里的子类三元组应被识别为层级边
        retriever.observe_triple("人", "是一种", "动物");
        assert!(retriever.subclass_reachable("人", "动物"));
        let constraints = SearchConstraints {
            entity_type: Some("动物".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 5, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().any(|hit| hit.subject == "苏格拉底"),
            "沿图谱子类边应命中：{:?}",
            search.hits
        );
    }

    #[test]
    fn removing_subclass_triple_drops_hierarchy_edge() {
        // 用 observe_triple 建立的层级边随三元组撤回（层级与证据同源）
        let mut graph = DenseRetriever::new(5, 0.85, 12);
        graph.observe_triple("学生", "是一种", "人物");
        assert!(graph.subclass_reachable("学生", "人物"));
        assert!(graph.remove_triple("学生", "是一种", "人物"));
        assert!(!graph.subclass_reachable("学生", "人物"));
    }

    #[test]
    fn unrelated_types_are_open_world_compatible() {
        let retriever = rdfs_retriever();
        // 无层级也无互斥证据：不凭空发明不相容（开放世界）
        assert!(retriever.type_compatible("学生", "公司"));
        assert!(!retriever.types_disjoint("学生", "公司"));
        assert!(retriever.types_disjoint("人物", "地点"), "顺序无关判定");
        assert!(retriever.types_disjoint("地点", "人物"));
    }

    #[test]
    fn predicate_constraint_expands_to_subproperties() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_subproperty("师承", "师傅");
        retriever.observe_triple("张三", "师承", "李四");
        retriever.observe_triple("王五", "师傅", "赵六");
        assert!(retriever.predicate_entails("师承", "师傅"));
        assert!(!retriever.predicate_entails("师傅", "师承"), "父不蕴含子");

        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let pairs: HashSet<(&str, &str)> = search
            .hits
            .iter()
            .map(|hit| (hit.subject.as_str(), hit.predicate.as_str()))
            .collect();
        assert!(
            pairs.contains(&("张三", "师承")),
            "子谓词事实应被父谓词约束召回：{:?}",
            pairs
        );
        assert!(pairs.contains(&("王五", "师傅")), "直接事实照常召回");
        assert!(
            search
                .receipt
                .notes
                .iter()
                .any(|note| note.contains("子谓词")),
            "谓词展开应写进回执：{:?}",
            search.receipt.notes
        );

        // 单向性：子谓词约束不召回父谓词事实
        let constraints = SearchConstraints {
            predicate: Some("师承".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        assert!(
            search.hits.iter().all(|hit| hit.subject != "王五"),
            "父谓词事实不得混入子谓词约束：{:?}",
            search.hits
        );
    }

    #[test]
    fn schema_disjoint_types_prune_evidence() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_subclass("学生", "人物");
        retriever.observe_disjoint("人物", "地点");
        retriever.observe_domain("师傅", "人物");
        retriever.observe_range("师傅", "人物");
        // 主体类型（地点）与定义域（人物）可证明互斥：硬剪
        retriever.observe_entity("故宫", "地点", &[]);
        retriever.observe_triple("故宫", "师傅", "导游");
        // 主体类型（学生）沿子类相容：保留
        retriever.observe_entity("张三", "学生", &[]);
        retriever.observe_triple("张三", "师傅", "李四");
        // 主体类型未知：开放世界，不剪
        retriever.observe_triple("无名氏", "师傅", "某人");

        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = retriever.search("", 10, &constraints, SearchBudget::unlimited());
        let subjects: HashSet<&str> = search.hits.iter().map(|hit| hit.subject.as_str()).collect();
        assert!(
            !subjects.contains("故宫"),
            "定义域互斥的证据必须被硬剪：{:?}",
            subjects
        );
        assert!(
            subjects.contains("张三"),
            "子类相容证据保留：{:?}",
            subjects
        );
        assert!(
            subjects.contains("无名氏"),
            "未知类型不凭空剪（开放世界）：{:?}",
            subjects
        );
    }

    #[test]
    fn subproperty_triples_feed_and_unfeed_hierarchy() {
        let mut retriever = DenseRetriever::new(5, 0.85, 12);
        retriever.observe_triple("师承", "subPropertyOf", "师傅");
        assert!(retriever.predicate_entails("师承", "师傅"));
        assert!(retriever.remove_triple("师承", "subPropertyOf", "师傅"));
        assert!(!retriever.predicate_entails("师承", "师傅"));
    }
}

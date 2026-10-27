#[cfg(test)]
mod tests {
    use super::*;

    fn planner() -> QueryPlanner {
        QueryPlanner::new()
    }

    #[test]
    fn test_find_object_chain() {
        let plan = planner()
            .parse("张三的师傅的师傅是谁")
            .expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["张三".to_string(), "师傅".to_string(), "师傅".to_string()]
            }
        );
    }

    #[test]
    fn test_modal_words_do_not_change_the_query_goal() {
        for question in [
            "李四的老师应该是谁",
            "李四的老师可能是谁呢？",
            "李四的老师到底是谁",
            "李四的老师究竟是谁",
        ] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("情态问句应可编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec!["李四".to_string(), "老师".to_string()]
                },
                "{}",
                question
            );
        }
    }

    #[test]
    fn test_find_object_with_polite_prefix_and_particles() {
        let plan = planner()
            .parse("请问李四的师祖是什么呢？")
            .expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师祖".to_string()]
            }
        );
    }

    #[test]
    fn test_degree_attribute_questions_compile_to_attribute_chain() {
        for (question, subject, attribute) in [
            ("姚明多高", "姚明", "身高"),
            ("姚明有多高", "姚明", "身高"),
            ("姚明的身高有多高", "姚明", "身高"),
            ("姚明多重？", "姚明", "体重"),
            ("张三多大", "张三", "年龄"),
            ("张三多大了？", "张三", "年龄"),
            ("李四几岁", "李四", "年龄"),
            ("王五多少岁", "王五", "年龄"),
            ("王五多少岁了", "王五", "年龄"),
        ] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("程度问句应编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec![subject.to_string(), attribute.to_string()]
                },
                "{}",
                question
            );
        }
        let chained = planner()
            .parse("姚明的儿子多高")
            .expect("关系链上的程度问句应保留链");
        assert_eq!(
            chained,
            QueryPlan::FindObject {
                segments: vec![
                    "姚明".to_string(),
                    "儿子".to_string(),
                    "身高".to_string()
                ]
            }
        );
        assert!(planner().parse("今天天气怎么样").is_none());
        assert!(planner().parse("地球多大").is_some(), "映射后仍需由知识侧决定是否有「年龄」");
    }

    #[test]
    fn test_find_subject_except_accepts_predicate_first_phrasing() {
        let plan = planner()
            .parse("除了姚明还有谁身高是213")
            .expect("应编译为带排除项的主体反查");
        assert_eq!(
            plan,
            QueryPlan::FindSubjectExcept {
                object: "213".to_string(),
                predicate: "身高".to_string(),
                excluded: vec!["姚明".to_string()],
            }
        );
        let multiple = planner()
            .parse("除了姚明和王治郅，还有谁的身高是213")
            .expect("多个排除项应可解析");
        assert_eq!(
            multiple,
            QueryPlan::FindSubjectExcept {
                object: "213".to_string(),
                predicate: "身高".to_string(),
                excluded: vec!["姚明".to_string(), "王治郅".to_string()],
            }
        );
    }

    #[test]
    fn test_nationality_questions_compile_to_nationality_attribute() {
        for question in [
            "姚明是哪国人",
            "姚明是哪个国家的人？",
            "姚明来自哪里",
        ] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("国籍问句应编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec!["姚明".to_string(), "国籍".to_string()]
                },
                "{}",
                question
            );
        }
    }

    #[test]
    fn test_find_object_with_amount_tail() {
        let plan = planner()
            .parse("姚明的身高是多少")
            .expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["姚明".to_string(), "身高".to_string()]
            }
        );
        let plan2 = planner()
            .parse("李四的师傅有多少")
            .expect("应编译为查找计划");
        assert_eq!(
            plan2,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师傅".to_string()]
            }
        );
    }

    #[test]
    fn test_find_subject_pattern() {
        let plan = planner().parse("谁的师傅是李四").expect("应编译为反查主体");
        assert_eq!(
            plan,
            QueryPlan::FindSubject {
                object: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
    }

    #[test]
    fn name_questions_compile_to_name_attribute_chain() {
        for question in [
            "张三叫什么名字",
            "张三叫啥名字",
            "张三的名字是什么",
            "张三姓名是什么",
            "张三的师傅叫什么名字",
            "张三的师傅的名字是什么",
            "张三的师傅姓名是什么",
            "张三的师傅的姓名是什么",
        ] {
            let expected_segments = if question.contains("师傅") {
                vec!["张三".to_string(), "师傅".to_string(), "姓名".to_string()]
            } else {
                vec!["张三".to_string(), "姓名".to_string()]
            };
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("姓名问句应可编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: expected_segments
                },
                "{}",
                question
            );
        }
    }

    #[test]
    fn inverse_possessive_question_compiles_to_subject_lookup() {
        for question in ["张三是谁的徒弟", "张三是哪位的徒弟？"] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("应编译为反查主体：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindSubject {
                    object: "张三".to_string(),
                    predicate: "徒弟".to_string()
                },
                "{}",
                question
            );
        }
        assert!(
            !matches!(
                planner().parse("李四的师傅是谁的徒弟"),
                Some(QueryPlan::FindSubject { .. })
            ),
            "多跳链不应被压成单个实体反查"
        );
    }

    #[test]
    fn copula_statements_are_not_mistaken_for_subject_lookup() {
        for question in [
            "张三是李四的师傅",
            "张三是李四的师傅吗",
            "姚明是中国人",
        ] {
            let plan = planner().parse(question);
            assert!(
                !matches!(plan, Some(QueryPlan::FindSubject { .. })),
                "陈述句不得编译成反查：{} → {:?}",
                question,
                plan
            );
        }
        for question in ["张三是李四的师傅", "张三是李四的师傅吗"] {
            assert_eq!(
                planner().parse(question),
                Some(QueryPlan::CheckTriple {
                    subject_segments: vec!["李四".to_string(), "师傅".to_string()],
                    object: "张三".to_string(),
                }),
                "{}",
                question
            );
        }
    }

    #[test]
    fn test_find_subject_pattern_accepts_predicate_first_phrasing() {
        for (question, predicate, object) in [
            ("身高是226的人是谁", "身高", "226"),
            ("身高为226的人有谁", "身高", "226"),
            ("国籍是中国的人有哪些", "国籍", "中国"),
        ] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("应编译为反查主体：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindSubject {
                    object: object.to_string(),
                    predicate: predicate.to_string()
                },
                "{}",
                question
            );
        }
    }

    #[test]
    fn test_count_question() {
        let plan = planner().parse("李四有几个师傅").expect("应编译为计数");
        assert_eq!(
            plan,
            QueryPlan::Count {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
        let plan2 = planner().parse("苹果公司有多少员工").expect("应编译为计数");
        assert_eq!(
            plan2,
            QueryPlan::Count {
                subject: "苹果公司".to_string(),
                predicate: "员工".to_string()
            }
        );
    }

    #[test]
    fn test_closure_question() {
        let plan = planner()
            .parse("李四的所有上司是谁")
            .expect("应编译为闭包计划");
        assert_eq!(
            plan,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let list = planner()
            .parse("李四的全部上司分别是谁？")
            .expect("应编译为闭包计划");
        assert_eq!(
            list,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let count = planner()
            .parse("李四的所有上司一共有几个")
            .expect("应编译为闭包计数");
        assert_eq!(
            count,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: true,
                then: Vec::new(),
            }
        );
        let bare = planner()
            .parse("李四所有上司是谁")
            .expect("省略「的」的闭包问句");
        assert_eq!(
            bare,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let counted = planner()
            .parse("李四的所有上司有多少位")
            .expect("量词尾的闭包计数");
        assert_eq!(
            counted,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: true,
                then: Vec::new(),
            }
        );
        let all_people = planner()
            .parse("李四的所有上司都有哪些人")
            .expect("「都有哪些人」尾");
        assert_eq!(
            all_people,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let people_count = planner()
            .parse("李四的所有上司一共有多少位")
            .expect("「一共有多少位」尾");
        assert_eq!(
            people_count,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: true,
                then: Vec::new(),
            }
        );
        let composed = planner()
            .parse("李四的所有上司的师傅是谁")
            .expect("应编译为闭包+后续关系");
        assert_eq!(
            composed,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: vec!["师傅".to_string()],
            }
        );
    }

    #[test]
    fn test_closure_does_not_swallow_exist_or_count() {
        let exist = planner().parse("李四有没有上司").expect("存在性问句");
        assert_eq!(
            exist,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "上司".to_string()
            }
        );
        let count = planner().parse("李四有几个上司").expect("直接计数");
        assert_eq!(
            count,
            QueryPlan::Count {
                subject: "李四".to_string(),
                predicate: "上司".to_string()
            }
        );
        let negative = planner().parse("李四没有师傅");
        assert!(
            !matches!(
                negative,
                Some(QueryPlan::Exist { ref subject, .. }) if subject.contains('没')
            ),
            "{:?}",
            negative
        );
    }

    #[test]
    fn test_exist_question() {
        let plan = planner().parse("李四有没有师傅").expect("应编译为存在性");
        assert_eq!(
            plan,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
        let plan2 = planner().parse("李四有师傅吗").expect("应编译为存在性");
        assert_eq!(
            plan2,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
    }

    #[test]
    fn test_check_triple_yes_no() {
        let plan = planner()
            .parse("李四的师傅是王五吗")
            .expect("应编译为事实校验");
        assert_eq!(
            plan,
            QueryPlan::CheckTriple {
                subject_segments: vec!["李四".to_string(), "师傅".to_string()],
                object: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_check_triple_reversed() {
        let plan = planner()
            .parse("王五是不是李四的师傅")
            .expect("应编译为事实校验");
        assert_eq!(
            plan,
            QueryPlan::CheckTriple {
                subject_segments: vec!["李四".to_string(), "师傅".to_string()],
                object: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_similar_question() {
        let plan = planner().parse("谁和张三最像").expect("应编译为近邻检索");
        assert_eq!(
            plan,
            QueryPlan::Similar {
                entity: "张三".to_string()
            }
        );
        let plan2 = planner()
            .parse("和李四最像的实体是谁？")
            .expect("应编译为近邻检索");
        assert_eq!(
            plan2,
            QueryPlan::Similar {
                entity: "李四".to_string()
            }
        );
    }

    #[test]
    fn test_analogy_question() {
        let plan = planner()
            .parse("张三对应李四，王五对应谁")
            .expect("应编译为类比");
        assert_eq!(
            plan,
            QueryPlan::Analogy {
                a: "张三".to_string(),
                b: "李四".to_string(),
                c: "王五".to_string()
            }
        );
        let plan2 = planner()
            .parse("张三之于李四，王五呢")
            .expect("应编译为类比");
        assert_eq!(
            plan2,
            QueryPlan::Analogy {
                a: "张三".to_string(),
                b: "李四".to_string(),
                c: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_topic_without_question_word_is_find_object() {
        let plan = planner()
            .parse("李四的师祖")
            .expect("省略问句也应给出查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师祖".to_string()]
            }
        );
    }

    #[test]
    fn test_path_between_question() {
        let copula = planner()
            .parse("张三是李四的什么人")
            .expect("系表式关系问法应编译为路径计划");
        assert_eq!(
            copula,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("张三和王五之间是什么关系")
            .expect("两点之间的问句应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("张三和李四什么关系")
            .expect("省略是字的问句也应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("张三到王五怎么走")
            .expect("到字问句应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
    }

    #[test]
    fn test_exact_hop_closure_question() {
        let plan = planner()
            .parse("甲恰好 2 跳的上司是谁")
            .expect("精确跳数问句");
        assert_eq!(
            plan,
            QueryPlan::Closure {
                subject: "甲".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Exact(2),
                count_only: false,
                then: Vec::new(),
            }
        );
        let chinese = planner()
            .parse("甲正好两跳的上司有几个")
            .expect("中文数字精确跳数");
        assert_eq!(
            chinese,
            QueryPlan::Closure {
                subject: "甲".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Exact(2),
                count_only: true,
                then: Vec::new(),
            }
        );
    }

    #[test]
    fn test_explicit_hop_limit_overrides_default() {
        let plan = planner()
            .parse("张三和李四之间最多 2 跳是什么关系")
            .expect("路径问句");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 2,
            }
        );
        let chinese = planner()
            .parse("张三和王五之间最多三跳的中间人是谁")
            .expect("中间人问句");
        assert_eq!(
            chinese,
            QueryPlan::Intermediary {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 3,
            }
        );
        let suffix = planner()
            .parse("张三和李四之间三跳以内是什么关系")
            .expect("跳数后缀");
        assert_eq!(
            suffix,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 3,
            }
        );
        let default = planner()
            .parse("张三和李四之间是什么关系")
            .expect("默认路径问句");
        assert_eq!(
            default,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 4,
            }
        );
    }

    #[test]
    fn test_common_neighbor_question() {
        let plan = planner()
            .parse("张三和李四共同认识谁")
            .expect("共同邻居问句应被识别");
        assert_eq!(
            plan,
            QueryPlan::CommonNeighbor {
                left: "张三".to_string(),
                right: "李四".to_string(),
            }
        );
        let plan = planner()
            .parse("谁同时认识张三和李四")
            .expect("问题词在前也应识别共同邻居");
        assert_eq!(
            plan,
            QueryPlan::CommonNeighbor {
                left: "张三".to_string(),
                right: "李四".to_string(),
            }
        );
    }

    #[test]
    fn test_intermediary_question() {
        let plan = planner()
            .parse("张三通过谁认识王五")
            .expect("通过谁问句应编译为中间人计划");
        assert_eq!(
            plan,
            QueryPlan::Intermediary {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("谁把张三和王五连接起来")
            .expect("连接句应识别中间人");
        assert_eq!(
            plan,
            QueryPlan::Intermediary {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
    }

    #[test]
    fn test_path_questions_do_not_swallow_existing_plans() {
        assert!(matches!(
            planner().parse("李四有没有师傅"),
            Some(QueryPlan::Exist { .. })
        ));
        assert!(matches!(
            planner().parse("张三和李四是什么关系"),
            Some(QueryPlan::PathBetween { .. })
        ));
    }

    #[test]
    fn test_degree_question_uses_explicit_attribute() {
        for question in ["北京的面积多大", "北京面积多大", "北京的面积有多大"] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("程度问句应可编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec!["北京".to_string(), "面积".to_string()]
                },
                "{}",
                question
            );
        }
        // 明写属性优先于「多大」的默认年龄语义。
        let plan = planner().parse("李四的出生地有多大").expect("应命中出生地");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "出生地".to_string()]
            }
        );
    }

    #[test]
    fn test_which_relations_compile_to_plain_chain() {
        for question in ["李四有哪些徒弟", "李四的徒弟有哪些", "李四都有哪些徒弟"] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("复数问法应可编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec!["李四".to_string(), "徒弟".to_string()]
                },
                "{}",
                question
            );
        }
        assert!(matches!(
            planner().parse("李四有几个徒弟"),
            Some(QueryPlan::Count { .. })
        ));
    }

    #[test]
    fn test_attribute_quantity_without_particles() {
        for question in ["北京人口多少", "北京人口是多少", "北京有多少人口"] {
            let plan = planner()
                .parse(question)
                .unwrap_or_else(|| panic!("省略问法应可编译：{}", question));
            assert_eq!(
                plan,
                QueryPlan::FindObject {
                    segments: vec!["北京".to_string(), "人口".to_string()]
                },
                "{}",
                question
            );
        }
        // 没有命中属性词表时仍交给原有计数解析，不能把任意名词当属性。
        assert!(matches!(
            planner().parse("北京有多少光年"),
            Some(QueryPlan::Count { .. })
        ));
    }

    #[test]
    fn test_attribute_quantity_keeps_longest_label() {
        let plan = planner().parse("李四出生地是多少").expect("应命中出生地");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "出生地".to_string()]
            }
        );
        let chain = planner()
            .parse("叶莉的先生的身高是多少")
            .expect("多跳属性问句应可解析");
        assert_eq!(
            chain,
            QueryPlan::FindObject {
                segments: vec![
                    "叶莉".to_string(),
                    "先生".to_string(),
                    "身高".to_string()
                ]
            },
            "主体链必须按「的」拆段"
        );
    }

    #[test]
    fn test_unrelated_question_returns_none() {
        assert!(planner().parse("今天天气怎么样").is_none());
        assert!(planner().parse("帮我讲个笑话").is_none());
    }

    #[test]
    fn test_bare_entity_question_returns_none() {
        assert!(planner().parse("李四是谁").is_none());
        assert!(planner().parse("什么是知识图谱").is_none());
    }

    #[test]
    fn test_plan_formats_are_printable() {
        let plan = planner().parse("张三的师傅的师傅是谁").unwrap();
        let text = QueryPlanner::format_plan(&plan);
        assert!(text.contains("张三"));
        assert!(text.contains("师傅"));
    }
}

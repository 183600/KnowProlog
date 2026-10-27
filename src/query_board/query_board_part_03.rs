#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct TestKnowledge {
        entities: Vec<String>,
        types: Vec<(String, String)>,
        attributes: HashMap<String, Vec<(String, String)>>,
        relations: HashMap<String, Vec<(String, String)>>,
    }

    impl TestKnowledge {
        fn new() -> Self {
            let mut knowledge = TestKnowledge::default();
            for (entity, entity_type, height, country) in [
                ("姚明", "Person", "226", "中国"),
                ("王治郅", "Person", "214", "中国"),
                ("易建联", "Person", "213", "中国"),
                ("诺维茨基", "Person", "213", "德国"),
                ("加索尔", "Person", "216", "西班牙"),
                ("奥尼尔", "Person", "216", "美国"),
            ] {
                knowledge.entities.push(entity.to_string());
                knowledge
                    .types
                    .push((entity.to_string(), entity_type.to_string()));
                knowledge.attributes.insert(
                    entity.to_string(),
                    vec![
                        ("身高".to_string(), height.to_string()),
                        ("国籍".to_string(), country.to_string()),
                    ],
                );
                knowledge.relations.insert(
                    entity.to_string(),
                    vec![("球队".to_string(), "火箭队".to_string())],
                );
            }
            knowledge
        }
    }

    impl DialogueKnowledge for TestKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.entities.iter().any(|candidate| candidate == entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            let mut keys: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
                .unwrap_or_default();
            keys.sort();
            keys.dedup();
            keys
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            let mut values: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| {
                    pairs
                        .iter()
                        .filter(|(key, _)| key == attribute)
                        .map(|(_, value)| value.clone())
                        .collect()
                })
                .unwrap_or_default();
            values.sort();
            values.dedup();
            values
        }

        fn all_entities(&self) -> Vec<String> {
            let mut entities = self.entities.clone();
            entities.sort();
            entities
        }

        fn all_types(&self) -> Vec<String> {
            let mut types: Vec<String> = self.types.iter().map(|(_, t)| t.clone()).collect();
            types.sort();
            types.dedup();
            types
        }

        fn entities_of_type(&self, entity_type: &str) -> Vec<String> {
            let mut entities: Vec<String> = self
                .types
                .iter()
                .filter(|(_, t)| t == entity_type)
                .map(|(entity, _)| entity.clone())
                .collect();
            entities.sort();
            entities
        }

        fn entity_type(&self, entity: &str) -> Option<String> {
            self.types
                .iter()
                .find(|(candidate, _)| candidate == entity)
                .map(|(_, entity_type)| entity_type.clone())
        }

        fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
            self.attributes.get(entity).cloned().unwrap_or_default()
        }

        fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
            self.relations.get(entity).cloned().unwrap_or_default()
        }

        fn describe(&self, entity: &str) -> Option<String> {
            self.known_entity(entity)
                .then(|| format!("{}是测试实体。", entity))
        }

        fn revision(&self) -> u64 {
            7
        }
    }

    fn board() -> QueryBoard {
        QueryBoard::new()
    }

    #[test]
    fn scan_and_select_by_ordinal() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        let first = board
            .apply("列出所有人的身高", &knowledge, 7)
            .expect("应接管集合查询");
        assert!(first.text.contains("姚明"));
        let frame = board.current().expect("应留下查询帧");
        assert_eq!(frame.rows.len(), 6);

        let expected = frame.rows[1].subject.clone();
        let second = board
            .apply("第二个呢", &knowledge, 7)
            .expect("应选中第二行");
        assert_eq!(
            board
                .current()
                .and_then(|frame| frame.selected_row())
                .map(|row| row.subject.as_str()),
            Some(expected.as_str())
        );
        assert!(second.text.contains("当前选中"));
    }

    #[test]
    fn filter_sort_and_compare_keep_evidence() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("只看中国的", &knowledge, 7).unwrap();
        assert_eq!(board.current().unwrap().rows.len(), 3);

        board.apply("按身高从高到低排", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.first().unwrap().subject, "姚明");
        assert_eq!(frame.rows.last().unwrap().subject, "易建联");

        let compare = board.apply("跟易建联比呢", &knowledge, 7).unwrap();
        assert!(compare.text.contains("姚明"));
        assert!(compare.text.contains("易建联"));
        assert!(compare.text.contains("高 13"));
        assert!(!compare.provenance.is_empty());
    }

    #[test]
    fn pagination_undo_and_stale_revision() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let next = board.apply("下一页", &knowledge, 7).unwrap();
        assert!(next.text.contains("第 2 页"));
        let undone = board.apply("回到上一步", &knowledge, 7).unwrap();
        assert!(undone.text.contains("第 1 页"));
        assert_eq!(board.cursor(), Some(0));
        assert_eq!(board.frame_count(), 2);

        let stale = board.apply("依据", &knowledge, 9).unwrap();
        assert!(stale.text.contains("v7 → v9"));
    }

    #[test]
    fn numeric_filter_and_missing_records() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("只看身高超过215", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.len(), 3);
        assert!(frame
            .rows
            .iter()
            .all(|row| row.value.parse::<i64>().unwrap() > 215));

        let missing = board.apply("跟不存在比呢", &knowledge, 7).unwrap();
        assert!(missing.text.contains("没有记录"));
    }

    #[test]
    fn board_snapshot_round_trip_and_reset() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有实体", &knowledge, 7).unwrap();
        board.apply("只看中国", &knowledge, 7).unwrap();
        let snapshot = board.clone();
        let mut restored = snapshot;
        assert_eq!(restored.current().unwrap().rows.len(), 3);
        let reset = restored.apply("重新开始", &knowledge, 7).unwrap();
        assert!(reset.text.contains("已清空"));
        assert!(!restored.is_active());
    }

    #[test]
    fn localized_type_names_still_scan() {
        let mut knowledge = TestKnowledge::new();
        for (_, entity_type) in &mut knowledge.types {
            *entity_type = "人物".to_string();
        }
        let mut board = board();
        let scan = board
            .apply("列出所有人", &knowledge, 7)
            .expect("中文本体类型也应能扫描");
        assert!(scan.text.contains("姚明"));
        assert_eq!(board.current().unwrap().rows.len(), 6);
    }

    #[test]
    fn whole_table_phrases_scan_every_entity() {
        let knowledge = TestKnowledge::new();
        for phrase in ["哪些实体", "有哪些实体", "哪几个实体", "列出所有实体"] {
            let mut board = board();
            let scan = board
                .apply(phrase, &knowledge, 7)
                .unwrap_or_else(|| panic!("「{}」应触发整表扫描", phrase));
            assert!(
                scan.text.contains("姚明"),
                "「{}」的结果缺少实体：{}",
                phrase,
                scan.text
            );
            let frame = board.current().expect("应留下查询帧");
            assert_eq!(frame.rows.len(), 6, "「{}」应扫描全部实体", phrase);
            assert!(frame.rows.iter().all(|row| row.predicate == "类型"));
        }
    }

    #[test]
    fn predicate_phrase_still_scans_after_whole_table_trim() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board
            .apply("哪些实体有身高", &knowledge, 7)
            .expect("带谓词的整表问法应扫描身高");
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.len(), 6);
        assert!(frame.rows.iter().all(|row| row.predicate == "身高"));
    }

    #[test]
    fn empty_board_defers_normal_questions() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        assert!(board.apply("姚明有多高", &knowledge, 7).is_none());
        assert!(board.apply("姚明的所有孩子是谁", &knowledge, 7).is_none());
        assert!(board.apply("显示姚明的身高", &knowledge, 7).is_none());
        assert!(board.apply("列出姚明的身高", &knowledge, 7).is_none());
        assert!(board.apply("你好", &knowledge, 7).is_none());
    }

    #[test]
    fn chinese_ordinals_cover_common_forms() {
        assert_eq!(chinese_number("一"), Some(1));
        assert_eq!(chinese_number("十"), Some(10));
        assert_eq!(chinese_number("十二"), Some(12));
        assert_eq!(chinese_number("二十"), Some(20));
        assert_eq!(chinese_number("二十三"), Some(23));
        assert_eq!(chinese_number("3"), Some(3));
        assert_eq!(parse_page_number("第2页"), Some(1));
    }

    #[test]
    fn compare_op_matches_thresholds() {
        assert!(CompareOp::Gt.matches(Comparison::Greater));
        assert!(CompareOp::Le.matches(Comparison::Equal));
        assert!(CompareOp::Eq.matches(Comparison::Equal));
        assert!(!CompareOp::Gt.matches(Comparison::Uncertain));
        assert!(!CompareOp::Ge.matches(Comparison::Incomparable(
            crate::quantity::Incomparable::NotAQuantity
        )));
    }

    #[test]
    fn quantity_normalizes_common_units_and_refuses_mixed_units() {
        let left = quantity::parse_quantity("2.26米").unwrap();
        let right = quantity::parse_quantity("226厘米").unwrap();
        assert_eq!(left.dimension(), right.dimension());
        assert!((left.base_value() - right.base_value()).abs() < 1e-9);

        let weight = quantity::parse_quantity("100斤").unwrap();
        assert_eq!(weight.dimension(), Dimension::Mass);
        assert!((weight.base_value() - 50.0).abs() < 1e-9);

        let left = QueryRow::new("甲", "身高", "226");
        let right = QueryRow::new("乙", "身高", "2.26米");
        let text = render_comparison(&left, &right, &TestKnowledge::new());
        assert!(text.contains("单位不同"));

        // 差值保留用户熟悉的原始单位，而不是一律折算成基准单位。
        let left = QueryRow::new("甲", "身高", "226厘米");
        let right = QueryRow::new("乙", "身高", "216厘米");
        let text = render_comparison(&left, &right, &TestKnowledge::new());
        assert!(text.contains("10厘米"), "{}", text);
        assert!(!text.contains("0.1米"), "{}", text);
    }

    #[test]
    fn numeric_filter_compares_units_not_raw_numbers() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "226厘米"),
            ("王治郅", "2.14米"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "216厘米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let outcome = board.apply("只看身高超过2.15米", &knowledge, 7).unwrap();
        assert!(outcome.text.contains("226厘米"), "{}", outcome.text);
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.len(), 3);
        let subjects: Vec<&str> = frame.rows.iter().map(|row| row.subject.as_str()).collect();
        assert!(subjects.contains(&"姚明"));
        assert!(subjects.contains(&"加索尔"));
        assert!(subjects.contains(&"奥尼尔"));
        assert!(!subjects.contains(&"王治郅"), "2.14 米不应超过 2.15 米");
        assert!(!subjects.contains(&"易建联"));
    }

    #[test]
    fn sort_uses_unit_conversion_not_raw_numbers() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "2.26米"),
            ("王治郅", "214厘米"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "2.16米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("按身高从高到低排", &knowledge, 7).unwrap();
        let frame = board.current().unwrap();
        assert_eq!(frame.rows.first().unwrap().subject, "姚明");
        assert_eq!(frame.rows.get(1).unwrap().subject, "加索尔");
        let values: Vec<f64> = frame
            .rows
            .iter()
            .map(|row| quantity::parse_quantity(&row.value).unwrap().base_value())
            .collect();
        assert!(
            values.windows(2).all(|pair| pair[0] >= pair[1]),
            "换算后应按米降序：{:?}",
            values
        );
        assert!((values.last().copied().unwrap() - 2.13).abs() < 1e-9);
    }

    #[test]
    fn rank_answers_who_is_highest_and_second_highest() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let highest = board.apply("谁最高", &knowledge, 7).expect("应回答最高者");
        assert!(highest.text.contains("姚明"), "{}", highest.text);
        assert!(highest.text.contains("最"), "{}", highest.text);
        assert!(highest
            .provenance
            .iter()
            .any(|note| note.contains("226")));
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::Rank {
                descending: true,
                ..
            }
        ));

        let second = board.apply("第二高呢", &knowledge, 7).unwrap();
        assert_eq!(
            board
                .current()
                .and_then(|frame| frame.selected_row())
                .map(|row| row.subject.as_str()),
            Some("加索尔")
        );
        assert!(second.text.contains("当前选中"));

        let top3 = board.apply("前三名", &knowledge, 7).unwrap();
        assert!(top3.text.contains("姚明"), "{}", top3.text);
        assert!(top3.text.contains("加索尔"), "{}", top3.text);
        assert!(top3.text.contains("王治郅"), "{}", top3.text);

        let why = board.apply("为什么姚明最高", &knowledge, 7).unwrap();
        assert!(why.text.contains("证据"), "{}", why.text);
        assert!(why.text.contains("排名依据"), "{}", why.text);
        assert!(
            why.text.contains("姚明（226）比 加索尔（216）高，相差 10"),
            "解释应给出次席与差值：{}",
            why.text
        );
        assert!(
            why.provenance.iter().any(|line| line.contains("216")),
            "次席证据也应进入出处：{:?}",
            why.provenance
        );
    }

    #[test]
    fn rank_explanation_refuses_a_fake_difference_for_approximations() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [("地球", "46亿年"), ("月球", "约46亿年")] {
            knowledge.entities.push(entity.to_string());
            knowledge
                .types
                .push((entity.to_string(), "天体".to_string()));
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("年龄".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的年龄", &knowledge, 7).unwrap();
        board.apply("谁最年长", &knowledge, 7).unwrap();
        let why = board.apply("为什么", &knowledge, 7).unwrap();
        assert!(why.text.contains("约数区间重叠"), "{}", why.text);
        assert!(!why.text.contains("相差"), "{}", why.text);
    }

    #[test]
    fn rank_frames_and_pending_state_survive_snapshot() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("谁最高", &knowledge, 7).unwrap();
        let json = serde_json::to_string(&board).unwrap();
        let mut restored: QueryBoard = serde_json::from_str(&json).unwrap();
        assert!(restored.is_active());
        let second = restored.apply("第二高呢", &knowledge, 7).unwrap();
        assert!(second.text.contains("当前选中"));

        let mut pending_board = QueryBoard::new();
        pending_board.apply("谁最高", &knowledge, 7).unwrap();
        let json = serde_json::to_string(&pending_board).unwrap();
        let mut restored: QueryBoard = serde_json::from_str(&json).unwrap();
        let outcome = restored.apply("身高", &knowledge, 7).unwrap();
        assert!(outcome.text.contains("姚明"), "{}", outcome.text);
    }

    #[test]
    fn rank_without_attribute_asks_once_then_resolves_short_answer() {
        let knowledge = TestKnowledge::new();
        let mut board = board();
        let pending = board.apply("谁最高", &knowledge, 7).expect("应发起属性澄清");
        assert!(pending.text.contains("哪个属性"), "{}", pending.text);
        assert!(
            pending.text.contains("身高"),
            "澄清应列出可排名属性：{}",
            pending.text
        );
        assert!(
            !pending.text.contains("国籍"),
            "非数值属性不应冒充可排名项：{}",
            pending.text
        );
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::RankPending { descending: true }
        ));

        let resolved = board
            .apply("身高", &knowledge, 7)
            .expect("短答属性应完成排名");
        assert!(resolved.text.contains("姚明"), "{}", resolved.text);
        assert!(matches!(
            board.current().unwrap().source,
            QuerySource::Rank { .. }
        ));
    }

    #[test]
    fn extreme_parser_rejects_non_rank_words_and_normalizes_attribute() {
        assert!(parse_extreme("谁最高", false).is_some());
        assert_eq!(
            parse_extreme("谁是最高的", false).unwrap().attribute,
            None
        );
        assert_eq!(
            parse_extreme("谁的身高是最高的", false)
                .unwrap()
                .attribute
                .as_deref(),
            Some("身高")
        );
        assert!(parse_extreme("谁最高兴", false).is_none());
        assert!(parse_extreme("最大的行星", false).is_none());
        assert!(parse_extreme("前三名", false).is_some());

        let mut knowledge = TestKnowledge::new();
        for (entity, area) in [("姚明", "100平方米"), ("王治郅", "200平方米")] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("面积".to_string(), area.to_string())],
            );
        }
        let mut board = board();
        let outcome = board
            .apply("哪个城市面积最大", &knowledge, 7)
            .expect("「城市面积」应归一为图谱谓词「面积」");
        assert!(outcome.text.contains("王治郅"), "{}", outcome.text);
        assert!(outcome.text.contains("面积"), "{}", outcome.text);
    }

    #[test]
    fn rank_answers_real_unit_values_without_string_ordering() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "226厘米"),
            ("王治郅", "2.14米"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "2.16米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![
                    ("身高".to_string(), value.to_string()),
                    ("国籍".to_string(), "未知".to_string()),
                ],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let highest = board.apply("谁最高", &knowledge, 7).expect("应回答最高者");
        assert!(highest.text.contains("姚明"), "{}", highest.text);
        assert!(highest.text.contains("226厘米"), "{}", highest.text);

        let second = board.apply("第二高呢", &knowledge, 7).unwrap();
        assert_eq!(
            board
                .current()
                .and_then(|frame| frame.selected_row())
                .map(|row| row.subject.as_str()),
            Some("加索尔")
        );
        assert!(second.text.contains("当前选中"), "{}", second.text);

        let why = board.apply("为什么姚明最高", &knowledge, 7).unwrap();
        assert!(
            why.text.contains("相差 10厘米"),
            "真实单位差值应保留厘米：{}",
            why.text
        );
    }

    #[test]
    fn rank_explanation_follows_resorted_direction() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "226厘米"),
            ("王治郅", "214厘米"),
            ("易建联", "213厘米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        board.apply("谁最高", &knowledge, 7).unwrap();
        board.apply("按身高从低到高排", &knowledge, 7).unwrap();
        let why = board.apply("为什么", &knowledge, 7).unwrap();
        assert!(
            why.text.contains("低，相差 1厘米"),
            "倒序后解释应说「低」且换算正确：{}",
            why.text
        );
        assert!(
            !why.text.contains("高，相差 1厘米"),
            "不能沿用旧方向：{}",
            why.text
        );
    }

    #[test]
    fn rank_top_n_never_fills_up_with_incomparable_rows() {
        let mut knowledge = TestKnowledge::new();
        knowledge.attributes.insert(
            "姚明".to_string(),
            vec![
                ("身高".to_string(), "2.26米".to_string()),
                ("居住面积".to_string(), "300平方米".to_string()),
            ],
        );
        knowledge.attributes.insert(
            "王治郅".to_string(),
            vec![("身高".to_string(), "2.14米".to_string())],
        );
        knowledge.attributes.insert(
            "易建联".to_string(),
            vec![("身高".to_string(), "2.13米".to_string())],
        );
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let outcome = board.apply("前两名", &knowledge, 7).unwrap();
        let summary = outcome
            .text
            .split("\n查询板")
            .next()
            .expect("排名摘要应在前");
        assert!(summary.contains("姚明"), "{}", summary);
        assert!(summary.contains("王治郅"), "{}", summary);
        assert!(
            !summary.contains("300平方米"),
            "跨量纲行不能补进前两名：{}",
            summary
        );
        assert!(!summary.contains("易建联"), "{}", summary);
    }

    #[test]
    fn rank_skips_cross_dimension_values() {
        let mut knowledge = TestKnowledge::new();
        for (entity, value) in [
            ("姚明", "2.26米"),
            ("王治郅", "67万平方公里"),
            ("易建联", "2.13米"),
            ("诺维茨基", "2.13米"),
            ("加索尔", "2.16米"),
            ("奥尼尔", "2.16米"),
        ] {
            knowledge.attributes.insert(
                entity.to_string(),
                vec![("身高".to_string(), value.to_string())],
            );
        }
        let mut board = board();
        board.apply("列出所有人的身高", &knowledge, 7).unwrap();
        let outcome = board.apply("谁最高", &knowledge, 7).unwrap();
        assert!(outcome.text.contains("姚明"), "{}", outcome.text);
        assert!(
            outcome.text.contains("未参与排名"),
            "跨量纲行必须显式说明：{}",
            outcome.text
        );
        assert_eq!(board.current().unwrap().rows.first().unwrap().subject, "姚明");
    }
}

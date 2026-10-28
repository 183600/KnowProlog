impl SymbolicGenerator {
    /// 用带失效机制的缓存提供某类型的谓词计数聚合结果
    fn with_type_counts<R>(
        &self,
        etype: &str,
        f: impl FnOnce(&HashMap<String, usize>, usize) -> R,
    ) -> R {
        let revision = self.revision;
        {
            let cache = self.type_count_cache.read();
            if let Some((cached_revision, counts, total)) = cache.get(etype) {
                if *cached_revision == revision {
                    return f(counts, *total);
                }
            }
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        if let Some(members) = self.type_members.get(etype) {
            for entity in members {
                if let Some(predicate_counts) = self.entity_predicate_counts.get(entity) {
                    for (predicate, count) in predicate_counts {
                        // 键通常已存在：命中时直接累加
                        match counts.get_mut(predicate) {
                            Some(existing) => *existing += count,
                            None => {
                                counts.insert(predicate.clone(), *count);
                            }
                        }
                        total += count;
                    }
                }
            }
        }
        let result = f(&counts, total);
        self.type_count_cache
            .write()
            .insert(etype.to_string(), (revision, counts, total));
        result
    }

    fn prototype_for(
        &self,
        etype: &str,
        entity: &str,
        predicate: &str,
    ) -> Option<(String, Vec<String>)> {
        let members = self.type_members.get(etype)?;
        let mut best: Option<(usize, &str)> = None;
        for other in members {
            if other == entity {
                continue;
            }
            let count = self
                .entity_predicate_counts
                .get(other)
                .and_then(|m| m.get(predicate))
                .copied()
                .unwrap_or(0);
            if count == 0 {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((best_count, best_name)) => {
                    count > *best_count || (count == *best_count && other.as_str() < *best_name)
                }
            };
            if replace {
                best = Some((count, other.as_str()));
            }
        }
        let (_, prototype) = best?;
        let objects = self
            .by_subject
            .get(prototype)
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some((prototype.to_string(), objects))
    }

    fn global_predictions(&self) -> Option<Vec<CompletionPrediction>> {
        if self.total_relations == 0 {
            return None;
        }
        let mut items: Vec<(&String, &usize)> = self.predicate_totals.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter() {
            let score = *count as f64 / self.total_relations as f64;
            if score < self.completion_min_score {
                // 已按计数降序，后续得分只会更低
                break;
            }
            let Some((prototype, objects)) = self.global_prototype(predicate) else {
                continue;
            };
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Global,
                score,
                support: format!("{}/{}", count, self.total_relations),
                sentence: format!(
                    "（全局回退）知识图谱中最常见的关系是「{}」，例如「{}」的「{}」是「{}」。",
                    predicate,
                    prototype,
                    predicate,
                    objects.join("、")
                ),
            });
            if predictions.len() >= 3 {
                break;
            }
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn global_prototype(&self, predicate: &str) -> Option<(String, Vec<String>)> {
        // 增量的“字典序最小主体”索引
        let prototype = self.predicate_min_entity.get(predicate)?;
        let objects = self
            .by_subject
            .get(prototype.as_str())
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some((prototype.clone(), objects))
    }
}
// 谓词动词化词典：规范谓词 → 中文句模板（{s}=主语，{o}=宾语）
fn lexical_template(predicate: &str) -> Option<&'static str> {
    Some(match predicate {
        "knows" | "认识" => "{s}认识{o}。",
        "age" | "年龄" => "{s}的年龄是{o}。",
        "capital" | "首都" => "{s}的首都是{o}。",
        "author" | "作者" => "{s}的作者是{o}。",
        "birthPlace" | "出生地" | "出生于" => "{s}出生于{o}。",
        "founder" | "创始人" => "{s}的创始人是{o}。",
        "memberOf" | "隶属于" => "{s}隶属于{o}。",
        "locatedIn" | "位于" => "{s}位于{o}。",
        "worksAt" | "工作于" => "{s}工作于{o}。",
        "spouse" | "配偶" | "妻子" | "丈夫" => "{s}的配偶是{o}。",
        "child" | "孩子" | "子女" => "{s}的子女是{o}。",
        "likes" | "喜欢" => "{s}喜欢{o}。",
        "nationality" | "国籍" => "{s}的国籍是{o}。",
        "occupation" | "职业" => "{s}的职业是{o}。",
        _ => return None,
    })
}
/// 常见英文谓词键的中文表述（生成叙事用；图谱内的谓词本身不变）。
pub fn localized_predicate_key(key: &str) -> Option<&'static str> {
    PREDICATE_LABELS
        .iter()
        .find(|(canonical, _)| *canonical == key)
        .map(|(_, label)| *label)
}
/// 中文谓词词面 → 图谱英文键（「身高」→ height）：
/// 供按中文属性名检索字面量取值时做归一。
pub fn canonical_predicate_key(label: &str) -> Option<&'static str> {
    PREDICATE_LABELS
        .iter()
        .find(|(_, surface)| *surface == label)
        .map(|(canonical, _)| *canonical)
}
/// 图谱属性键与中文词面的对照表只读视图（英文键 → 中文），
/// 供查询规划器识别「实体 + 属性 + 多少」这类省略「的/是」的问法。
pub fn predicate_labels() -> &'static [(&'static str, &'static str)] {
    PREDICATE_LABELS
}
/// 图谱属性键与中文词面的对照表（英文键 → 中文）。
const PREDICATE_LABELS: &[(&str, &str)] = &[
    ("name", "姓名"),
    // 「名字」是同一属性的口语词面；保留「姓名」为规范展示面，
    // 两个词面都能回落到图谱里的 name 键。
    ("name", "名字"),
    ("gender", "性别"),
    ("height", "身高"),
    ("weight", "体重"),
    ("birthDate", "出生日期"),
    ("capital", "首都"),
    ("population", "人口"),
    ("area", "面积"),
    ("author", "作者"),
    ("founder", "创始人"),
    ("location", "所在地"),
    ("locatedIn", "所在地"),
    ("worksAt", "工作单位"),
    ("memberOf", "隶属于"),
    ("nationality", "国籍"),
    ("occupation", "职业"),
    ("spouse", "配偶"),
    ("child", "子女"),
    ("age", "年龄"),
    ("birthPlace", "出生地"),
];
/// 是否姓名类谓词（生成画像时与实体名相同的取值属于同义反复）。
fn is_name_key(key: &str) -> bool {
    matches!(key, "name" | "姓名" | "名字")
}
/// 常见英文类型标签的中文表述（仅用于生成与展示兜底）。
pub fn localized_type_label(etype: &str) -> &str {
    match etype.to_ascii_lowercase().as_str() {
        "person" | "people" | "human" => "人物",
        "entity" => "实体",
        "location" | "place" => "地点",
        "city" => "城市",
        "country" => "国家",
        "organization" | "organisation" | "company" => "组织",
        "event" => "事件",
        "work" => "作品",
        "animal" => "动物",
        "food" => "食物",
        _ => etype,
    }
}
// 中文表述集合（首个为动词化时选择的面）→ 供形式化语义解析识别
fn lexical_surfaces(predicate: &str) -> &'static [&'static str] {
    match predicate {
        "knows" | "认识" => &["认识"],
        "age" | "年龄" => &["年龄"],
        "capital" | "首都" => &["首都"],
        "population" | "人口" => &["人口"],
        "area" | "面积" => &["面积"],
        "author" | "作者" => &["作者"],
        "birthPlace" | "出生地" | "出生于" => &["出生地", "出生于"],
        "founder" | "创始人" => &["创始人"],
        "memberOf" | "隶属于" => &["隶属于"],
        "locatedIn" | "位于" => &["位于", "所在地"],
        "worksAt" | "工作于" => &["工作于", "任职于"],
        "spouse" | "配偶" | "妻子" | "丈夫" => &["配偶"],
        "child" | "孩子" | "子女" => &["孩子", "子女"],
        "likes" | "喜欢" => &["喜欢"],
        "nationality" | "国籍" => &["国籍"],
        "occupation" | "职业" => &["职业"],
        _ => &[],
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_generator() -> SymbolicGenerator {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_entity("李四", "人物", &[]);
        generator.observe_entity("王五", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("李四", "师傅", "王五");
        generator.observe_triple("张三", "师祖", "王五");
        generator
    }

    #[test]
    fn test_profile_generation_uses_facts_and_markers() {
        let generator = sample_generator();
        let profile = generator.profile("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("类型：人物"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("李四"));
        assert!(profile.contains("年龄"));
        assert!(profile.contains("30"));
        assert!(profile.contains("首先"));
    }

    fn import_frame_ledger() -> crate::import_frames::ImportFrameLedger {
        use crate::import_frames::{
            FactKind, FrameAtom, FrameObservation, ImportFrame, ImportFrameLedger,
        };
        let mut ledger = ImportFrameLedger::new();
        let frame = ImportFrame {
            id: "ifr#test".to_string(),
            kind: FactKind::Attribute,
            predicate: "年龄".to_string(),
            atoms: vec![
                FrameAtom::Subject,
                FrameAtom::Literal {
                    surface: "今年".to_string(),
                },
                FrameAtom::Object,
                FrameAtom::Literal {
                    surface: "岁".to_string(),
                },
            ],
            example_subject: "张三".to_string(),
            example_object: "30".to_string(),
            observations: vec![FrameObservation {
                witness: "张三今年30岁".to_string(),
                import_digest: "digest".to_string(),
                turn: 1,
            }],
            ask_activated: true,
        };
        ledger.frames.insert(frame.id.clone(), frame);
        ledger
    }

    #[test]
    fn import_frame_realizes_and_answers_with_witness() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.attach_import_frames(import_frame_ledger());
        assert_eq!(
            generator.lexicalize("张三", "年龄", &["30".to_string()]),
            "张三今年30岁"
        );
        let answer = generator
            .answer_import_frame("张三今年多少岁")
            .expect("导入帧应能回答宾语问句");
        assert!(answer.contains("张三今年30岁"));
        assert!(answer.contains("依据：张三今年30岁"));
        generator.detach_import_frames();
        assert!(generator.import_frames().is_none());
        assert_ne!(
            generator.lexicalize("张三", "年龄", &["30".to_string()]),
            "张三今年30岁"
        );
    }

    #[test]
    fn frame_book_drives_annotation_and_realization() {
        use crate::frame_semantics::{FrameAnnotator, FrameBook, Role};

        let book = FrameBook::with_core_schemas();
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.attach_frame_book(book.clone());
        let frames = generator.annotate_frames("张三在北京用扳手修好了电脑");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].usable(Role::Location), Some("北京"));
        assert_eq!(
            generator.realize_frame(&frames[0]).as_deref(),
            Some("张三在北京用扳手修好了电脑")
        );
        let direct = FrameAnnotator::new(&book)
            .annotate("张三在北京用扳手修好了电脑")
            .pop()
            .expect("annotated");
        assert_eq!(direct.project().expect("projected").qualifiers.len(), 2);
        generator.detach_frame_book();
        assert!(generator.frame_book().is_none());
        assert!(generator.annotate_frames("张三修好了电脑").is_empty());
    }

    #[test]
    fn test_profile_unknown_entity_is_none() {
        let generator = sample_generator();
        assert!(generator.profile("不存在的人").is_none());
    }

    #[test]
    fn test_complete_entity_level_prediction() {
        let generator = sample_generator();
        let completion = generator.complete("张三的").expect("实体级统计应有预测");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));
        assert!(completion.contains("张三的师傅是李四"));
    }

    #[test]
    fn test_complete_type_level_backoff() {
        let mut generator = sample_generator();
        generator.observe_entity("赵六", "人物", &[]);
        let completion = generator.complete("赵六的").expect("类型级回退应有预测");
        assert!(
            completion.contains("类型级回退"),
            "实际输出：{}",
            completion
        );
        assert!(completion.contains("师傅"));
        assert!(completion.contains("参照同类实体"));
    }

    #[test]
    fn test_complete_global_backoff() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "类型A", &[]);
        generator.observe_triple("乙", "朋友", "丙");
        let completion = generator.complete("甲的").expect("全局回退应有预测");
        assert!(completion.contains("全局回退"), "实际输出：{}", completion);
        assert!(completion.contains("朋友"));
    }

    #[test]
    fn test_complete_requires_known_entity_and_de_suffix() {
        let generator = sample_generator();
        assert!(generator.complete("不存在的人的").is_none());
        assert!(generator.complete("张三").is_none());
    }

    #[test]
    fn test_profile_respects_sentence_limit() {
        let mut generator = SymbolicGenerator::new(2, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let profile = generator.profile("张三").unwrap();
        let sentences = profile.matches('。').count();
        assert!(sentences <= 2, "截断后应不超过 2 句：{}", profile);
    }

    #[test]
    fn test_lexicalization_maps_bare_predicates_to_chinese() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_triple("张三", "knows", "李四");
        generator.observe_triple("李四", "age", "30");
        let profile = generator.profile("张三").unwrap();
        assert!(
            profile.contains("张三认识李四"),
            "裸谓词应动词化：{}",
            profile
        );
        let profile2 = generator.profile("李四").unwrap();
        assert!(
            profile2.contains("李四的年龄是30"),
            "英文属性应中文化：{}",
            profile2
        );
        let surfaces = generator.predicate_surfaces();
        assert_eq!(surfaces.get("认识").map(String::as_str), Some("knows"));
        assert_eq!(surfaces.get("age").map(String::as_str), Some("age"));
    }

    #[test]
    fn test_observed_attribute_surface_overrides_earlier_alias() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        // 配置谓词画像先登记「岁数 → 年龄」。
        assert!(generator.register_predicate_alias("岁数", "年龄"));
        assert_eq!(
            generator.canonical_predicate("年龄").as_deref(),
            Some("年龄")
        );
        // 随后图谱出现英文属性 age：内置中文词面「年龄」应重新指回 age。
        generator.observe_entity("张三", "人物", &[("age".to_string(), "30".to_string())]);
        assert_eq!(
            generator.canonical_predicate("年龄").as_deref(),
            Some("age")
        );
        assert_eq!(
            generator.canonical_predicate("岁数").as_deref(),
            Some("年龄")
        );
    }

    #[test]
    fn colloquial_name_surface_reaches_name_key() {
        assert_eq!(canonical_predicate_key("名字"), Some("name"));
        assert_eq!(canonical_predicate_key("姓名"), Some("name"));
        assert_eq!(localized_predicate_key("name"), Some("姓名"));
    }

    #[test]
    fn test_chinese_attribute_alias_reaches_english_key() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity(
            "北京",
            "City",
            &[("population".to_string(), "2189万".to_string())],
        );
        assert_eq!(canonical_predicate_key("人口"), Some("population"));
        assert_eq!(localized_predicate_key("population"), Some("人口"));
        assert_eq!(generator.attribute_values("北京", "人口"), vec!["2189万"]);
    }

    #[test]
    fn test_profile_page_supports_continuation_and_reorder() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let first = generator
            .profile_page("张三", 0, GenerationOrder::Relevance)
            .expect("第一页应可生成");
        assert_eq!(first.shown, 1);
        assert_eq!(first.total_lines, 2);
        assert_eq!(first.next_offset, 1);
        let second = generator
            .profile_page("张三", first.next_offset, GenerationOrder::Relevance)
            .expect("第二页应可生成");
        assert_eq!(second.shown, 1);
        assert!(
            second.text.contains("续：第 2–2 句"),
            "续页应有页码：{}",
            second.text
        );
        assert!(second.text.contains("接着，"));
        assert_ne!(first.text, second.text);
        let alphabetical_first = generator
            .profile_page("张三", 0, GenerationOrder::Alphabetical)
            .expect("换排序也应可生成");
        assert_eq!(alphabetical_first.total_lines, 2);
        let done = generator
            .profile_page("张三", 2, GenerationOrder::Relevance)
            .expect("翻到末页应返回空页");
        assert_eq!(done.shown, 0);
    }

    #[test]
    fn test_retract_triple_is_incremental_and_idempotent() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_triple("甲", "朋友", "乙");
        generator.observe_triple("丙", "朋友", "丁");
        assert_eq!(generator.relation_count(), 2);

        assert!(generator.retract_triple("甲", "朋友", "乙"));
        assert_eq!(generator.relation_count(), 1);
        assert!(generator.relation_values("甲", "朋友").is_empty());
        // 反向索引同步撤回，不再把甲当作「丁的朋友」
        assert!(generator
            .relation_values_in("丁", "朋友")
            .contains(&"丙".to_string()));
        assert!(!generator.retract_triple("甲", "朋友", "乙"));

        // 全局回退原型必须从现存主体重算，不能指向已撤回的甲。
        let completion = generator.complete("甲的").expect("仍应给出全局回退");
        assert!(completion.contains("全局回退"), "实际：{}", completion);
        assert!(
            completion.contains("丙"),
            "原型不应指向幽灵实体：{}",
            completion
        );
        assert!(!completion.contains("甲的朋友"), "{}", completion);
    }

    #[test]
    fn test_cursor_does_not_repeat_when_insertion_reorders() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let clauses = generator
            .profile_clauses("张三", GenerationOrder::Relevance)
            .unwrap();
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert_eq!(first.shown, 1);
        let first_sentence = first.trusted.first().cloned().unwrap_or_else(|| {
            first
                .text
                .lines()
                .find(|line| line.contains('。'))
                .unwrap()
                .trim()
                .to_string()
        });

        // 新事实插入到排序最前，旧实现会让 offset 指向另一句，导致重复/漏讲。
        generator.observe_triple("张三", "朋友", "赵六");
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert_eq!(second.shown, 1);
        assert!(
            !second
                .text
                .contains(first_sentence.trim_start_matches("接着，")),
            "已说过的句子不应重复：第一页={}；第二页={}",
            first.text,
            second.text
        );
        assert!(second.text.contains("接着，"), "{}", second.text);
        // 两轮之后仍未讲完，账本里不能有重复 key。
        assert_eq!(cursor.spoken_count(), 2);
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_cursor_reports_updates_and_removals() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(first.text.contains("李四"));

        assert!(generator.retract_triple("张三", "师傅", "李四"));
        generator.observe_triple("张三", "师傅", "王五");
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(second.text.contains("更正"), "{}", second.text);
        assert!(second.text.contains("此前说"), "{}", second.text);
        assert!(second.text.contains("现在更新为"), "{}", second.text);
        assert!(second.text.contains("王五"), "{}", second.text);

        assert!(generator.retract_triple("张三", "师傅", "王五"));
        let third = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        assert!(third.text.contains("已不再成立"), "{}", third.text);
    }

    #[test]
    fn test_cursor_uses_layered_session_overlay() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        let mut overlay = GenerationOverlay::default();
        overlay.replace.insert(
            ("张三".to_string(), "师傅".to_string()),
            vec!["王五".to_string()],
        );
        overlay
            .suppress
            .entry(("张三".to_string(), "师傅".to_string()))
            .or_default()
            .insert("李四".to_string());

        let clauses = generator
            .profile_clauses_with_overlay("张三", GenerationOrder::Relevance, &overlay)
            .unwrap();
        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].sentence.contains("王五"));
        assert!(!clauses[0].sentence.contains("李四"));
        assert!(clauses[0].from_overlay);

        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let page = generator
            .cursor_page("张三", GenerationOrder::Relevance, &overlay, &mut cursor)
            .unwrap();
        assert!(page.text.contains("王五"));
        assert!(!page.text.contains("李四"));
        assert_eq!(page.trusted.len(), 1);
    }

    #[test]
    fn test_cursor_survives_serialization_and_continues_without_repeat() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let mut cursor = GenerationCursor::new("张三", GenerationOrder::Relevance);
        let first = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut cursor,
            )
            .unwrap();
        let encoded = serde_json::to_string(&cursor).expect("账本应可序列化");
        let mut restored: GenerationCursor = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.spoken_count(), 1);
        let second = generator
            .cursor_page(
                "张三",
                GenerationOrder::Relevance,
                &GenerationOverlay::default(),
                &mut restored,
            )
            .unwrap();
        assert_ne!(first.text, second.text);
        assert!(second.text.contains("续：第 2–2 句"), "{}", second.text);
        assert!(!second.text.contains(first.text.lines().last().unwrap()));
    }

    #[test]
    fn test_type_members_are_ordered_and_follow_type_changes() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("乙", "人物", &[]);
        generator.observe_entity("甲", "人物", &[]);
        assert_eq!(generator.entities_of_type("人物"), vec!["乙", "甲"]);

        generator.observe_entity("乙", "组织", &[]);
        assert_eq!(generator.entities_of_type("人物"), vec!["甲"]);
        assert_eq!(generator.entities_of_type("组织"), vec!["乙"]);
    }

    #[test]
    fn test_type_profile_materializes_members_predicates_and_coverage() {
        let generator = sample_generator();
        let profile = generator.type_profile("人物").expect("人物应有类型画像");
        assert_eq!(profile.total_members, 3);
        assert_eq!(profile.members, vec!["张三", "李四", "王五"]);
        assert_eq!(profile.covered_members, 2);

        let master = profile
            .predicates
            .iter()
            .find(|summary| summary.predicate == "师傅")
            .expect("师傅应进入谓词切片");
        assert_eq!(master.member_count, 2);
        assert_eq!(master.value_count, 2);
        assert_eq!(master.coverage_milli, 667);
        assert_eq!(master.kind, TypeFactKind::Relation);
        assert!(!master.samples.is_empty());

        let report = generator.type_profile_report("人物").unwrap();
        assert!(report.contains("类型画像「人物」"), "{}", report);
        assert!(report.contains("成员：张三、李四、王五"), "{}", report);
        assert!(report.contains("覆盖：2/3"), "{}", report);
        assert!(report.contains("许多"), "{}", report);
        assert!(report.contains("师傅"), "{}", report);
        assert!(report.contains("样例事实"), "{}", report);
        assert_eq!(report, generator.type_profile_report("人物").unwrap());
    }

    #[test]
    fn test_type_profile_cache_invalidates_only_the_touched_type() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "甲类", &[("年龄".to_string(), "20".to_string())]);
        generator.observe_entity("乙", "乙类", &[("人口".to_string(), "100".to_string())]);
        let alpha = generator.type_profile("甲类").unwrap();
        let beta = generator.type_profile("乙类").unwrap();
        let beta_revision = beta.revision;

        generator.observe_entity("甲", "甲类", &[("身高".to_string(), "180".to_string())]);
        let alpha_after = generator.type_profile("甲类").unwrap();
        let beta_after = generator.type_profile("乙类").unwrap();
        assert!(
            alpha_after.revision > alpha.revision,
            "受影响类型必须换版本"
        );
        assert_eq!(beta_after.revision, beta_revision, "无关类型不得被整体失效");
        assert_eq!(beta_after, beta, "无关类型画像应逐字节保持");
        assert!(alpha_after
            .predicates
            .iter()
            .any(|summary| summary.predicate == "身高"));

        generator.observe_triple("Avat", "朋友", "Bod");
        let other = generator.type_profile("乙类").unwrap();
        assert_eq!(other, beta_after, "其它关系写入同样不得打掉无关类型");
    }

    #[test]
    fn test_type_profile_follows_member_moves_and_retractions() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "甲类", &[]);
        generator.observe_entity("乙", "乙类", &[]);
        generator.observe_triple("甲", "朋友", "丙");
        let before = generator.type_profile("甲类").unwrap();
        assert_eq!(before.total_members, 1);

        generator.observe_entity("甲", "乙类", &[]);
        assert!(
            generator.type_profile("甲类").is_none(),
            "成员迁出后空类型不再有画像"
        );
        let moved = generator.type_profile("乙类").unwrap();
        assert_eq!(moved.members, vec!["乙", "甲"]);
        assert!(moved.revision > before.revision);

        assert!(generator.retract_triple("甲", "朋友", "丙"));
        let after = generator.type_profile("乙类").unwrap();
        assert!(after.revision > moved.revision, "撤回成员事实必须推进版本");
        assert_eq!(after.covered_members, 0);
        assert!(after.predicates.is_empty());
    }

    #[test]
    fn test_type_member_page_and_comparison_are_deterministic() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "甲类", &[("说明".to_string(), "x".to_string())]);
        generator.observe_entity("乙", "甲类", &[("年龄".to_string(), "20".to_string())]);
        generator.observe_entity("丙", "乙类", &[("说明".to_string(), "y".to_string())]);
        generator.observe_entity("丁", "乙类", &[("人口".to_string(), "9".to_string())]);

        let first = generator.type_member_page("甲类", 0, 1).unwrap();
        assert_eq!(first.members, vec!["乙"]);
        assert_eq!(first.next_offset, 1);
        assert!(!first.done);
        let second = generator
            .type_member_page("甲类", first.next_offset, 1)
            .unwrap();
        assert_eq!(second.members, vec!["甲"]);
        assert!(second.done);

        let comparison = generator.compare_types("甲类", "乙类").unwrap();
        assert_eq!(comparison.shared_predicates, vec!["说明".to_string()]);
        assert_eq!(comparison.left_only_predicates, vec!["年龄".to_string()]);
        assert_eq!(comparison.right_only_predicates, vec!["人口".to_string()]);
        let report = comparison.render();
        assert!(report.contains("共同谓词（1）：说明"), "{}", report);
        assert!(report.contains("仅「甲类」（1）：年龄"), "{}", report);
        assert!(report.contains("仅「乙类」（1）：人口"), "{}", report);
    }

    #[test]
    fn test_type_profile_is_stable_across_insertion_orders() {
        fn build(reverse: bool) -> String {
            let mut generator = SymbolicGenerator::new(6, 0.3);
            let mut facts: Vec<(&str, &str, &str)> = vec![
                ("甲", "朋友", "丙"),
                ("乙", "朋友", "丁"),
                ("甲", "年龄", "30"),
                ("乙", "年龄", "40"),
            ];
            if reverse {
                facts.reverse();
            }
            for (subject, predicate, object) in facts {
                generator.observe_triple(subject, predicate, object);
            }
            generator.observe_entity("甲", "人物", &[]);
            generator.observe_entity("乙", "人物", &[]);
            generator.type_profile_report("人物").unwrap()
        }
        assert_eq!(build(false), build(true));
    }
}

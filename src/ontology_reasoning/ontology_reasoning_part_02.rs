/// 解析单句本体知识：类层级、互斥、子关系、传递/对称/逆、定义域/值域、类固有属性、个体归类
pub fn parse_ontology_lesson(text: &str) -> Result<OntologyLesson, String> {
    // 疑问句检查必须在清理标点之前：clean_lesson_text 会去掉句末的问号
    let raw = text.trim();
    if raw.contains('？') || raw.contains('?') {
        return Err(format!(
            "「{}」是疑问句，不能作为本体知识教学",
            clean_lesson_text(text)
        ));
    }
    let cleaned = clean_lesson_text(text);
    if cleaned.is_empty() {
        return Err("空句子".to_string());
    }
    let original = cleaned.clone();
    // 否定句检查：解析器只接受肯定知识，避免「企鹅不属于鸟」被解析成
    // 子类「企鹅不」这类被吞掉否定词的错误知识
    if cleaned.contains("不是")
        || cleaned.contains("不属于")
        || cleaned.contains("不一定")
        || cleaned.contains("并非")
        || cleaned.contains("不具有")
        || cleaned.contains("不满足")
        || cleaned.contains("不构成")
        || cleaned.contains("不可以")
        || cleaned.contains("不能")
    {
        return Err(format!("「{}」是否定句，本体教学只接受肯定知识", cleaned));
    }
    // 互斥：X和Y互斥 / X与Y不相容
    for marker in ["互不相容", "互相排斥", "彼此排斥", "互斥", "不相容", "排斥"]
    {
        if let Some(position) = cleaned.find(marker) {
            if assertion_negated(&cleaned, position) {
                continue;
            }
            if let Some((left, right)) = split_pair(&cleaned[..position]) {
                return Ok(OntologyLesson::Disjoint {
                    left,
                    right,
                    original,
                });
            }
        }
    }
    // 传递性
    if cleaned.contains("传递") {
        if let Some(property) = extract_property_prefix(
            &cleaned,
            &[
                "关系可以传递",
                "具有传递性",
                "是可以传递的",
                "是可传递的",
                "是传递的",
                "可以传递",
                "传递",
            ],
        ) {
            return Ok(OntologyLesson::Transitive { property, original });
        }
    }
    // 对称性
    if cleaned.contains("对称") {
        if let Some(property) =
            extract_property_prefix(&cleaned, &["具有对称性", "是对称的", "对称"])
        {
            return Ok(OntologyLesson::Symmetric { property, original });
        }
    }
    // 逆关系
    for marker in ["互为逆关系", "互为相反关系", "互为逆"] {
        if let Some(position) = cleaned.find(marker) {
            if assertion_negated(&cleaned, position) {
                continue;
            }
            if let Some((left, right)) = split_pair(&cleaned[..position]) {
                return Ok(OntologyLesson::Inverse {
                    left,
                    right,
                    original,
                });
            }
        }
    }
    // 子关系：X是Y的子关系
    if let Some(position) = cleaned.find("的子关系") {
        if let Some((sub, sup)) = split_shi(&cleaned[..position]) {
            return Ok(OntologyLesson::Subproperty { sub, sup, original });
        }
    }
    // 定义域 / 值域
    if let Some(position) = cleaned.find("定义域") {
        let mut property = cleaned[..position].trim().to_string();
        for tail in ["的", "关系"] {
            property = property.trim_end_matches(tail).trim().to_string();
        }
        let class = cleaned[position + "定义域".len()..]
            .trim()
            .trim_start_matches(['是', '为', '：', ':', ' '])
            .trim()
            .to_string();
        if valid_name(&property) && valid_name(&class) {
            return Ok(OntologyLesson::Domain {
                property,
                class,
                original,
            });
        }
    }
    if let Some(position) = cleaned.find("值域") {
        let mut property = cleaned[..position].trim().to_string();
        for tail in ["的", "关系"] {
            property = property.trim_end_matches(tail).trim().to_string();
        }
        let class = cleaned[position + "值域".len()..]
            .trim()
            .trim_start_matches(['是', '为', '：', ':', ' '])
            .trim()
            .to_string();
        if valid_name(&property) && valid_name(&class) {
            return Ok(OntologyLesson::Range {
                property,
                class,
                original,
            });
        }
    }
    // 子类：X是一种Y / X是一类Y / X属于Y / X是Y的一种 / X是Y的子类
    for marker in ["是一种", "是一类", "属于"] {
        if let Some(position) = cleaned.find(marker) {
            let sub = cleaned[..position].trim().to_string();
            let mut sup = cleaned[position + marker.len()..].trim().to_string();
            // 「X属于Y的一种」等后置分类词写法：剥离后缀，避免把「Y的一种」整体当作父类
            for suffix in ["的一种", "的一类", "的子类", "的子集"] {
                if let Some(stripped) = sup.strip_suffix(suffix) {
                    sup = stripped.trim().to_string();
                    break;
                }
            }
            if valid_name(&sub) && valid_name(&sup) {
                return Ok(OntologyLesson::Subclass { sub, sup, original });
            }
        }
    }
    for marker in ["的一种", "的一类", "的子类", "的子集"] {
        if let Some(position) = cleaned.find(marker) {
            if let Some((sub, sup)) = split_shi(&cleaned[..position]) {
                return Ok(OntologyLesson::Subclass { sub, sup, original });
            }
        }
    }
    // 类固有属性：X的P是O
    if let Some(position_de) = cleaned.find('的') {
        let after_de = &cleaned[position_de + '的'.len_utf8()..];
        if let Some(position_shi) = after_de.find('是') {
            let class = cleaned[..position_de].trim().to_string();
            let property = after_de[..position_shi].trim().to_string();
            let object = after_de[position_shi + '是'.len_utf8()..]
                .trim()
                .to_string();
            if valid_name(&class) && valid_name(&property) && valid_name(&object) {
                return Ok(OntologyLesson::ClassProperty {
                    class,
                    property,
                    object,
                    original,
                });
            }
        }
    }
    // 个体归类：X是Y
    if let Some((individual, class)) = split_shi(&cleaned) {
        return Ok(OntologyLesson::Instance {
            individual,
            class,
            original,
        });
    }
    Err(format!(
        "无法解析「{}」（支持：X是一种Y / X属于Y / X和Y互斥 / X是Y的子关系 / X关系可以传递 / X是对称的 / X和Y互为逆关系 / X关系的定义域是Y / X的P是O / X是Y）",
        cleaned
    ))
}
/// 解析多句文本（按 。；; 换行 切分），收集成功与失败
pub fn parse_ontology_lessons(text: &str) -> OntologyParseOutcome {
    let mut outcome = OntologyParseOutcome::default();
    let segments: Vec<&str> = text
        .split(['。', '；', ';', '\n'])
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in segments {
        match parse_ontology_lesson(segment) {
            Ok(lesson) => outcome.lessons.push(lesson),
            Err(reason) => outcome.errors.push(reason),
        }
    }
    outcome
}
/// 把解析出的本体知识写入本体推理器
pub fn apply_ontology_lesson(ontology: &mut OntologyReasoner, lesson: &OntologyLesson) {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => ontology.add_subclass(sub, sup),
        OntologyLesson::Instance {
            individual, class, ..
        } => ontology.add_instance(individual, class),
        OntologyLesson::Disjoint { left, right, .. } => ontology.add_disjoint(left, right),
        OntologyLesson::Subproperty { sub, sup, .. } => ontology.add_subproperty(sub, sup),
        OntologyLesson::Transitive { property, .. } => ontology.add_transitive(property),
        OntologyLesson::Symmetric { property, .. } => ontology.add_symmetric(property),
        OntologyLesson::Inverse { left, right, .. } => ontology.add_inverse(left, right),
        OntologyLesson::Domain {
            property, class, ..
        } => ontology.add_domain(property, class),
        OntologyLesson::Range {
            property, class, ..
        } => ontology.add_range(property, class),
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => ontology.add_class_property(class, property, object),
    }
}
/// 本体知识的生成式回读（中文复述）
pub fn render_ontology_lesson_zh(lesson: &OntologyLesson) -> String {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => {
            format!(
                "「{}」是「{}」的一种（子类关系，实例与属性会沿层级继承）",
                sub, sup
            )
        }
        OntologyLesson::Instance {
            individual, class, ..
        } => format!("「{}」属于「{}」（个体归类的硬事实）", individual, class),
        OntologyLesson::Disjoint { left, right, .. } => format!(
            "「{}」与「{}」互不相容（同一事物不能同时属于两者，用于一致性检查）",
            left, right
        ),
        OntologyLesson::Subproperty { sub, sup, .. } => {
            format!(
                "「{}」是「{}」的子关系（子关系事实自动升级为父关系事实）",
                sub, sup
            )
        }
        OntologyLesson::Transitive { property, .. } => format!(
            "「{}」关系可以传递（a{}b、b{}c ⇒ a{}c）",
            property, property, property, property
        ),
        OntologyLesson::Symmetric { property, .. } => format!(
            "「{}」关系是对称的（a{}b ⇒ b{}a）",
            property, property, property
        ),
        OntologyLesson::Inverse { left, right, .. } => {
            format!(
                "「{}」与「{}」互为逆关系（a{}b ⇒ b{}a）",
                left, right, left, right
            )
        }
        OntologyLesson::Domain {
            property, class, ..
        } => format!(
            "「{}」关系的定义域是「{}」（主体由关系事实自动归入该类）",
            property, class
        ),
        OntologyLesson::Range {
            property, class, ..
        } => format!(
            "「{}」关系的值域是「{}」（客体由关系事实自动归入该类）",
            property, class
        ),
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => format!(
            "「{}」的{}是「{}」（类固有属性，会被全部实例继承）",
            class, property, object
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subclass_closure_is_transitive_and_reflexive() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("皇帝企鹅", "企鹅");
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_subclass("鸟", "动物");
        let closure = ontology.materialize();
        assert!(closure.is_subclass("皇帝企鹅", "动物"), "子类应传递");
        assert!(closure.is_subclass("企鹅", "企鹅"), "子类应自反");
        assert!(closure.ancestors_of("企鹅").contains(&"动物".to_string()));
        assert!(closure.descendants_of("鸟").contains(&"企鹅".to_string()));
        assert!(closure
            .descendants_of("鸟")
            .contains(&"皇帝企鹅".to_string()));
    }

    #[test]
    fn test_instance_propagates_along_subclasses() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_instance("阿德", "企鹅");
        ontology.add_subclass("企鹅", "鸟");
        let closure = ontology.materialize();
        assert!(closure.is_instance("阿德", "鸟"), "个体类型应沿子类上推");
        assert!(closure.types_of("阿德").contains(&"企鹅".to_string()));
        assert!(closure.types_of("阿德").contains(&"鸟".to_string()));
        let reason = closure.why_instance("阿德", "鸟").unwrap_or("");
        assert!(reason.contains("企鹅"), "应记录推导链：{}", reason);
    }

    #[test]
    fn test_domain_and_range_infer_types() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_relation("张三", "师傅", "李四");
        ontology.add_domain("师傅", "人");
        ontology.add_range("师傅", "人");
        let closure = ontology.materialize();
        assert!(
            closure.is_instance("张三", "人"),
            "主体应由定义域归入「人」"
        );
        assert!(closure.is_instance("李四", "人"), "客体应由值域归入「人」");
        let reason = closure.why_instance("张三", "人").unwrap_or("");
        assert!(reason.contains("定义域"), "应记录定义域推导：{}", reason);
    }

    #[test]
    fn test_class_property_inheritance_and_subproperty() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_instance("阿德", "企鹅");
        ontology.add_class_property("企鹅", "栖息地", "南极");
        ontology.add_subproperty("师徒", "师傅");
        ontology.add_relation("张三", "师徒", "李四");
        let closure = ontology.materialize();
        assert!(
            closure
                .properties_of("阿德")
                .contains(&("栖息地".to_string(), "南极".to_string())),
            "类固有属性应被实例继承"
        );
        assert!(
            closure.relations_snapshot().contains(&(
                "张三".to_string(),
                "师傅".to_string(),
                "李四".to_string()
            )),
            "子关系事实应升级为父关系"
        );
        let reason = closure.why_relation("阿德", "栖息地", "南极").unwrap_or("");
        assert!(reason.contains("继承"), "应记录继承来源：{}", reason);
    }

    #[test]
    fn test_transitive_symmetric_inverse_relations() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_transitive("师傅");
        ontology.add_relation("张三", "师傅", "李四");
        ontology.add_relation("李四", "师傅", "王五");
        ontology.add_symmetric("同学");
        ontology.add_relation("甲", "同学", "乙");
        ontology.add_inverse("师傅", "徒弟");
        let closure = ontology.materialize();
        let relations = closure.relations_snapshot();
        assert!(
            relations.contains(&("张三".to_string(), "师傅".to_string(), "王五".to_string())),
            "传递闭包应产生 张三→王五"
        );
        assert!(
            relations.contains(&("乙".to_string(), "同学".to_string(), "甲".to_string())),
            "对称关系应补齐反向"
        );
        assert!(
            relations.contains(&("李四".to_string(), "徒弟".to_string(), "张三".to_string())),
            "逆关系应补齐反向"
        );
        let reason = closure.why_relation("张三", "师傅", "王五").unwrap_or("");
        assert!(reason.contains("传递"), "应记录传递性来源：{}", reason);
    }

    #[test]
    fn test_disjoint_inconsistency_reported() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_instance("怪东西", "鸟");
        ontology.add_instance("怪东西", "鱼");
        ontology.add_disjoint("鸟", "鱼");
        let closure = ontology.materialize();
        assert_eq!(closure.problems().len(), 1, "应报出一个不一致");
        assert!(
            closure.problems()[0].contains("互斥"),
            "问题应说明互斥冲突：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_shared_superclass_is_not_a_disjoint_conflict() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("狗", "动物");
        ontology.add_subclass("猫", "动物");
        ontology.add_disjoint("狗", "猫");
        ontology.add_instance("旺财", "动物");
        let closure = ontology.materialize();
        assert!(
            closure.problems().is_empty(),
            "个体仅属于共同上位类时不应报互斥冲突：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_unsatisfiable_class_reported() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("企鹅", "鸟");
        ontology.add_subclass("企鹅", "鱼");
        ontology.add_disjoint("鸟", "鱼");
        let closure = ontology.materialize();
        assert!(
            closure.problems().iter().any(|p| p.contains("不可满足")),
            "应报出不可满足类：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_disjoint_member_subclass_is_unsatisfiable() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_disjoint("鸟", "鱼");
        ontology.add_subclass("鸟", "鱼");
        let closure = ontology.materialize();
        assert!(
            closure.problems().iter().any(|p| p.contains("不可满足")),
            "互斥类自身成为对方子类时应报出不可满足：{:?}",
            closure.problems()
        );
    }

    #[test]
    fn test_negated_ontology_lessons_are_rejected() {
        assert!(parse_ontology_lesson("企鹅不属于鸟").is_err());
        assert!(parse_ontology_lesson("张三不是老师").is_err());
        // 「不相容」是合法的互斥知识，不应被否定句检查误伤
        assert!(matches!(
            parse_ontology_lesson("猫和狗互不相容"),
            Ok(OntologyLesson::Disjoint { .. })
        ));
        // 含「不」的肯定类名不应被误伤
        assert!(matches!(
            parse_ontology_lesson("不锈钢是一种合金"),
            Ok(OntologyLesson::Subclass { .. })
        ));
    }

    #[test]
    fn test_parse_and_apply_chinese_ontology_lessons() {
        let outcome = parse_ontology_lessons(
            "企鹅是一种鸟；企鹅和鱼互斥。师傅关系可以传递。张三的师傅是李四。",
        );
        assert_eq!(
            outcome.lessons.len(),
            4,
            "应解析出 4 条：{:?}",
            outcome.errors
        );
        assert!(
            outcome.errors.is_empty(),
            "不应有失败：{:?}",
            outcome.errors
        );
        let mut ontology = OntologyReasoner::new(6);
        for lesson in &outcome.lessons {
            apply_ontology_lesson(&mut ontology, lesson);
            assert!(!render_ontology_lesson_zh(lesson).is_empty());
        }
        let closure = ontology.materialize();
        assert!(closure.is_subclass("企鹅", "鸟"));
        assert!(closure.relations_snapshot().contains(&(
            "张三".to_string(),
            "师傅".to_string(),
            "李四".to_string()
        )));
        assert!(parse_ontology_lesson("这条知识是谁说的？").is_err());
        assert!(parse_ontology_lesson("完全看不懂的句子").is_err());
    }

    #[test]
    fn test_property_profile_collects_axioms() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_domain("师傅", "人");
        ontology.add_transitive("师傅");
        ontology.add_inverse("师傅", "徒弟");
        ontology.add_subproperty("师徒", "师傅");
        ontology.add_relation("张三", "师傅", "李四");
        let closure = ontology.materialize();
        let profile = ontology.property_profile("师傅", &closure);
        assert!(profile.transitive, "应标记传递性");
        assert!(profile.domains.contains(&"人".to_string()));
        assert!(profile.inverses.contains(&"徒弟".to_string()));
        assert!(profile.subproperties.contains(&"师徒".to_string()));
        assert!(profile.closure_relations >= 1);
    }

    #[test]
    fn closure_snapshot_roundtrip_preserves_derivation_and_digest() {
        let mut ontology = OntologyReasoner::new(6);
        ontology.add_subclass("学生", "人物");
        ontology.add_instance("张三", "学生");
        ontology.add_domain("师傅", "人物");
        ontology.add_relation("李四", "师傅", "王五");
        let closure = ontology.materialize();
        assert_eq!(
            closure.why_instance("张三", "人物"),
            Some("学生 ⊑ 人物"),
            "推导理由应随快照带出"
        );
        let snapshot = closure.snapshot();
        let restored = OntologyClosure::from_snapshot(&snapshot);
        assert_eq!(restored.snapshot(), snapshot, "快照往返应逐字一致");
        assert_eq!(restored.digest(), closure.digest(), "摘要交换（L4）");
        assert_eq!(
            restored.why_instance("张三", "人物"),
            closure.why_instance("张三", "人物")
        );
        assert_eq!(
            restored.why_relation("李四", "师傅", "王五"),
            closure.why_relation("李四", "师傅", "王五")
        );
    }

    #[test]
    fn closure_snapshot_is_set_order_invariant() {
        let mut forward = OntologyReasoner::new(6);
        for (sub, sup) in [("学生", "人物"), ("教师", "人物"), ("人物", "实体")] {
            forward.add_subclass(sub, sup);
        }
        forward.add_instance("张三", "学生");
        let mut reverse = OntologyReasoner::new(6);
        for (sub, sup) in [("人物", "实体"), ("教师", "人物"), ("学生", "人物")] {
            reverse.add_subclass(sub, sup);
        }
        reverse.add_instance("张三", "学生");
        let left = forward.materialize();
        let right = reverse.materialize();
        assert_eq!(
            canonical::canonical_bytes(&left.snapshot()).unwrap(),
            canonical::canonical_bytes(&right.snapshot()).unwrap(),
            "集合迭代序不得泄漏进快照：同一闭包必须同一字节"
        );
        assert_eq!(left.digest(), right.digest());
    }

    #[test]
    fn capsule_digest_ignores_lesson_order_and_duplicates() {
        let first = parse_ontology_lesson("人是一种动物").expect("可解析");
        let second = parse_ontology_lesson("猫是一种动物").expect("可解析");
        let mut ontology = OntologyReasoner::new(6);
        apply_ontology_lesson(&mut ontology, &first);
        let closure = ontology.materialize();
        let forward = OntologyCapsule::new(vec![first.clone(), second.clone()], &closure);
        let reverse = OntologyCapsule::new(vec![second, first.clone(), first.clone()], &closure);
        assert_eq!(forward.lessons.len(), 2, "重复公理应去重");
        assert_eq!(forward.lessons, reverse.lessons, "规约与教学顺序无关");
        assert_eq!(forward.program_digest, reverse.program_digest);
        assert_eq!(
            canonical::canonical_bytes(&forward).unwrap(),
            canonical::canonical_bytes(&reverse).unwrap(),
            "同一状态同一字节"
        );
        forward.verify().expect("自建胶囊应通过校验");
        reverse.verify().expect("自建胶囊应通过校验");
    }

    #[test]
    fn capsule_rejects_tampered_lessons_and_future_version() {
        let lesson = parse_ontology_lesson("人是一种动物").expect("可解析");
        let mut ontology = OntologyReasoner::new(6);
        apply_ontology_lesson(&mut ontology, &lesson);
        let closure = ontology.materialize();
        let mut capsule = OntologyCapsule::new(vec![lesson], &closure);
        capsule.verify().expect("原胶囊应通过校验");

        let mut tampered = capsule.clone();
        tampered.lessons.clear();
        let error = tampered.verify().expect_err("改写公理必须被拒绝");
        assert!(error.contains("摘要"), "错误应说明摘要不匹配：{}", error);

        let mut future = capsule.clone();
        future.version = ONTOLOGY_CAPSULE_VERSION + 1;
        let error = future.verify().expect_err("超前版本必须被版本门拒绝");
        assert!(error.contains("版本"), "错误应说明版本门：{}", error);

        capsule.version = 0;
        assert!(capsule.verify().is_err(), "版本 0 视为未知格式");

        // 缓存可核对：按公理重算的闭包与导出缓存一致
        let fresh = {
            let mut replay = OntologyReasoner::new(6);
            apply_ontology_lesson(&mut replay, &parse_ontology_lesson("人是一种动物").unwrap());
            replay.materialize()
        };
        assert!(
            OntologyCapsule::new(vec![parse_ontology_lesson("人是一种动物").unwrap()], &fresh)
                .closure_matches(&fresh),
            "同一公理 + 同一事实应得到同一闭包摘要"
        );
    }

    #[test]
    fn capsule_without_lessons_still_verifies_and_matches_empty_closure() {
        let closure = OntologyClosure::default();
        let capsule = OntologyCapsule::new(Vec::new(), &closure);
        capsule.verify().expect("空胶囊应可校验");
        assert!(capsule.closure_matches(&OntologyClosure::default()));
        assert_eq!(capsule.version, ONTOLOGY_CAPSULE_VERSION);
    }

    /// 配置态本体：`[type_system]` 与 `[[predicate_profiles]]` 归约成同一套公理，
    /// 且每条公理的原句都能被解析器回读成同一条公理（自描述、可审计）。
    #[test]
    fn config_program_roundtrips_through_lesson_parser() {
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::{TypeDisjointConfig, TypeSubclassConfig, TypeSystemConfig};

        let types = TypeSystemConfig {
            subclasses: vec![
                TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                },
                TypeSubclassConfig {
                    sub: "教师".into(),
                    sup: "人物".into(),
                },
            ],
            disjoint: vec![TypeDisjointConfig {
                left: "地点".into(),
                right: "人物".into(),
            }],
            ..TypeSystemConfig::default()
        };
        let profiles = vec![PredicateProfileConfig {
            name: "师傅".into(),
            aliases: vec!["师父".into()],
            inverse: Some("徒弟".into()),
            functional: true,
            transitive: true,
            domain: vec!["人物".into()],
            range: vec!["人物".into()],
            description: Some("师承".into()),
            ..PredicateProfileConfig::default()
        }];
        let program = OntologyProgram::from_config(&types, &profiles);
        assert_eq!(
            program.len(),
            7,
            "子类2 + 互斥1 + 域1 + 值1 + 逆1 + 传递1：{:?}",
            program.lessons
        );
        assert!(!program.is_empty());
        for lesson in &program.lessons {
            let reparsed = parse_ontology_lesson(lesson.original())
                .unwrap_or_else(|error| panic!("原句应可回读：{}（{}）", lesson.original(), error));
            assert_eq!(&reparsed, lesson, "原句回读必须得到同一公理");
        }
        // 配置里没有的类型不入本体（别名不产生类层级）
        let mut merged = program.clone();
        merged.merge(&program);
        assert_eq!(
            merged.lessons.len(),
            program.lessons.len(),
            "合并自身应去重"
        );
    }

    /// 配置公理装入本体推理器：类型层级与谓词画像立即参与闭包物化。
    #[test]
    fn config_program_drives_closure_materialization() {
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::TypeSubclassConfig;

        let types = crate::type_system::TypeSystemConfig {
            subclasses: vec![TypeSubclassConfig {
                sub: "学生".into(),
                sup: "人物".into(),
            }],
            ..Default::default()
        };
        let profiles = vec![PredicateProfileConfig {
            name: "师傅".into(),
            transitive: true,
            domain: vec!["人物".into()],
            range: vec!["人物".into()],
            ..PredicateProfileConfig::default()
        }];
        let program = OntologyProgram::from_config(&types, &profiles);
        let mut ontology = OntologyReasoner::new(6);
        for lesson in &program.lessons {
            apply_ontology_lesson(&mut ontology, lesson);
        }
        ontology.add_instance("张三", "学生");
        ontology.add_relation("张三", "师傅", "李四");
        let closure = ontology.materialize();
        assert!(
            closure.is_subclass("学生", "人物"),
            "配置子类应参与闭包：{:?}",
            closure.instances_snapshot()
        );
        assert!(
            closure.why_instance("张三", "人物").is_some(),
            "个体类型应沿配置子类上推"
        );
        assert!(
            closure.why_instance("李四", "人物").is_some(),
            "值域推断应生效：{:?}",
            closure.why_instance("李四", "人物")
        );
    }

    /// 配置态子属性通道：`subproperty_of` 归约为子关系公理，原句可回读，
    /// 且子关系事实沿层级上浮到父谓词（理解召回与生成概括的共同基础）。
    #[test]
    fn config_program_reduces_subproperty_and_lifts_facts() {
        use crate::predicate_registry::PredicateProfileConfig;

        let profiles = vec![
            PredicateProfileConfig {
                name: "师傅".into(),
                ..PredicateProfileConfig::default()
            },
            PredicateProfileConfig {
                name: "师承".into(),
                subproperty_of: Some("师傅".into()),
                ..PredicateProfileConfig::default()
            },
        ];
        let program = OntologyProgram::from_config(&Default::default(), &profiles);
        let lesson = program
            .lessons
            .iter()
            .find_map(|lesson| match lesson {
                OntologyLesson::Subproperty { sub, sup, .. } => Some((sub.clone(), sup.clone())),
                _ => None,
            })
            .expect("subproperty_of 应归约为子关系公理");
        assert_eq!(lesson, ("师承".to_string(), "师傅".to_string()));
        let sub_lesson = program
            .lessons
            .iter()
            .find(|lesson| matches!(lesson, OntologyLesson::Subproperty { .. }))
            .unwrap();
        let reparsed = parse_ontology_lesson(sub_lesson.original())
            .unwrap_or_else(|error| panic!("子关系原句应可回读：{}", error));
        assert_eq!(&reparsed, sub_lesson);

        let mut ontology = OntologyReasoner::new(6);
        for lesson in &program.lessons {
            apply_ontology_lesson(&mut ontology, lesson);
        }
        ontology.add_relation("张三", "师承", "李四");
        let closure = ontology.materialize();
        assert!(
            closure.why_relation("张三", "师傅", "李四").is_some(),
            "子关系事实应上浮到父谓词：{:?}",
            closure.relations_snapshot()
        );
        assert!(
            closure
                .why_relation("张三", "师傅", "李四")
                .map(|why| why.contains("师承"))
                .unwrap_or(false),
            "上浮推导理由应引用子关系边：{:?}",
            closure.why_relation("张三", "师傅", "李四")
        );
    }
}

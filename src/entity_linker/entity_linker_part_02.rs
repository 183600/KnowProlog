#[cfg(test)]
fn dice_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();
    // 单字指称不做模糊匹配，避免"张"链接到"张三"之类的过度链接
    if chars_a.len() < 2 || chars_b.len() < 2 {
        return 0.0;
    }
    let bigram_score = {
        let bigram_a = bigrams(a);
        let bigram_b = bigrams(b);
        let intersection = bigram_a.intersection(&bigram_b).count() as f64;
        2.0 * intersection / (bigram_a.len() + bigram_b.len()) as f64
    };
    let unigram_score = {
        let set_a: HashSet<char> = chars_a.iter().copied().collect();
        let set_b: HashSet<char> = chars_b.iter().copied().collect();
        let intersection = set_a.intersection(&set_b).count() as f64;
        // 单字重叠较宽松，打折后仅作为二元组相似度的补充（处理双字名错别字）
        2.0 * intersection / (set_a.len() + set_b.len()) as f64 * 0.8
    };
    bigram_score.max(unigram_score)
}
/// 给定两侧字符数与字符/二元组集合大小，相似度（二元组/单字 Dice 取大者）的
/// 理论上界：交集不超过较小集合，故 `2*min(|A|,|B|)/(|A|+|B|)` 是可达最大值；
/// 文本完全相同时按 1.0 计（`dice_similarity_precomputed` 的相等捷径）。
fn similarity_upper_bound(
    len_a: usize,
    len_b: usize,
    char_set_a: usize,
    char_set_b: usize,
    bigrams_a: usize,
    bigrams_b: usize,
    equal_text: bool,
) -> f64 {
    if equal_text {
        return 1.0;
    }
    if len_a < 2 || len_b < 2 {
        return 0.0;
    }
    let mut upper: f64 = 0.0;
    if bigrams_a + bigrams_b > 0 {
        upper = upper.max(2.0 * bigrams_a.min(bigrams_b) as f64 / (bigrams_a + bigrams_b) as f64);
    }
    if char_set_a + char_set_b > 0 {
        upper = upper
            .max(2.0 * char_set_a.min(char_set_b) as f64 / (char_set_a + char_set_b) as f64 * 0.8);
    }
    upper
}
// 使用已预计算的指称字符集合与实体画像特征集合
fn dice_similarity_precomputed(
    trimmed: &str,
    trimmed_chars: &[char],
    trimmed_char_set: &HashSet<char>,
    trimmed_bigrams: &HashSet<(char, char)>,
    canonical: &str,
    profile: &EntityProfile,
) -> f64 {
    if trimmed == canonical {
        return 1.0;
    }
    if trimmed_chars.len() < 2 || profile.char_count < 2 {
        return 0.0;
    }
    let bigram_score = {
        let intersection = trimmed_bigrams.intersection(&profile.bigrams).count() as f64;
        2.0 * intersection / (trimmed_bigrams.len() + profile.bigrams.len()) as f64
    };
    let unigram_score = {
        let intersection = trimmed_char_set.intersection(&profile.char_set).count() as f64;
        2.0 * intersection / (trimmed_char_set.len() + profile.char_set.len()) as f64 * 0.8
    };
    bigram_score.max(unigram_score)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_linker() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.register_entity("刘德华", Some("人物"));
        linker.register_entity("刘德凯", Some("人物"));
        linker.register_entity("珠穆朗玛峰", Some("山峰"));
        linker.register_entity("苹果公司", Some("公司"));
        linker.register_entity("苹果果实", Some("水果"));
        linker.observe_triple("刘德华", "出生地", "香港");
        linker.observe_triple("刘德华", "职业", "演员");
        linker.observe_triple("苹果公司", "创始人", "乔布斯");
        linker.observe_triple("苹果果实", "颜色", "红色");
        linker.register_alias("苹果", "苹果公司");
        linker.register_alias("苹果", "苹果果实");
        linker
    }

    #[test]
    fn test_exact_and_alias_linking() {
        let linker = sample_linker();
        let mentions = linker.link("刘德华的出生地在哪里");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "刘德华");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::ExactName);

        let alias_mentions = linker.link("苹果创始人的创始人是谁");
        assert!(!alias_mentions.is_empty());
        let entities: Vec<String> = alias_mentions
            .iter()
            .filter_map(|m| m.best().map(|c| c.entity.clone()))
            .collect();
        assert!(entities.contains(&"苹果公司".to_string()));
    }

    #[test]
    fn test_fuzzy_typo_linking() {
        let linker = sample_linker();
        let mentions = linker.link("刘德桦的出生地是哪里");
        assert_eq!(mentions.len(), 1, "错别字应被模糊匹配容忍");
        let best = mentions[0].best().unwrap();
        assert_eq!(best.entity, "刘德华");
        assert_eq!(best.method, LinkMethod::Fuzzy);
        assert!(best.reasons[0].contains("相似度"));

        let long_typo = linker.link("珠穆朗玛锋有多高");
        assert_eq!(long_typo[0].best().unwrap().entity, "珠穆朗玛峰");
    }

    #[test]
    fn test_observe_triple_ignores_blank_endpoints() {
        let mut linker = EntityLinker::new();
        linker.observe_triple("", "师傅", "李四");
        linker.observe_triple("张三", "师傅", " ");
        assert_eq!(linker.len(), 0, "空白端点不应登记任何画像");
        linker.observe_triple("张三", "师傅", "李四");
        assert!(linker.is_known_entity("张三"));
        assert!(linker.is_known_entity("李四"));
    }

    #[test]
    fn test_repeat_observation_does_not_inflate_degree() {
        let mut linker = EntityLinker::new();
        linker.observe_triple("张三", "师傅", "李四");
        let degree = linker.profiles.get("张三").unwrap().degree;
        let predicate_count = linker
            .out_predicates
            .get("张三")
            .and_then(|bucket| bucket.get("师傅"))
            .copied()
            .unwrap();
        // 整图快照重放：度数与谓词计数必须保持不变
        linker.observe_triple("张三", "师傅", "李四");
        assert_eq!(linker.profiles.get("张三").unwrap().degree, degree);
        assert_eq!(
            linker
                .out_predicates
                .get("张三")
                .and_then(|bucket| bucket.get("师傅"))
                .copied(),
            Some(predicate_count)
        );
        assert_eq!(linker.triple_count(), 1);
    }

    #[test]
    fn test_context_disambiguation_prefers_hint_compatible_candidate() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let mentions = linker.link_with_context("苹果的创始人是谁", &hints);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "苹果公司");

        let hints2 = vec!["颜色".to_string()];
        let mentions2 = linker.link_with_context("苹果的颜色是什么", &hints2);
        assert_eq!(mentions2[0].best().unwrap().entity, "苹果果实");
    }

    #[test]
    fn test_fuzzy_matching_survives_repeated_characters() {
        // 重复字符令 bigram/字符集合远小于字符数：真实相似度仍可达到阈值，
        // 剪枝的上界必须按集合大小估计，否则会漏掉这些匹配
        let mut linker = EntityLinker::new();
        linker.register_entity("哈哈哈哈哈哈哈哈", Some("人物"));
        let mentions = linker.link("哈哈的来历");
        assert_eq!(mentions.len(), 1, "重复字符文本不应被上界剪枝漏掉");
        assert_eq!(mentions[0].best().unwrap().entity, "哈哈哈哈哈哈哈哈");
    }

    #[test]
    fn test_no_false_link_for_unknown_text() {
        let linker = sample_linker();
        let mentions = linker.link("今天天气怎么样");
        assert!(mentions.is_empty(), "无关文本不应产生链接：{:?}", mentions);
    }

    #[test]
    fn test_report_explains_candidates() {
        let linker = sample_linker();
        let report = linker.report("刘德桦的出生地是哪里", &[]);
        assert!(report.contains("实体链接报告"));
        assert!(report.contains("刘德华"));
        assert!(report.contains("置信度"));
    }

    #[test]
    fn test_dice_similarity_bounds() {
        assert_eq!(dice_similarity("张三", "张三"), 1.0);
        assert!(dice_similarity("刘德桦", "刘德华") > 0.4);
        assert_eq!(dice_similarity("张三", "苹果"), 0.0);
        assert_eq!(dice_similarity("张", "张三"), 0.0);
    }

    /// 提及账本：候选带类型证据、期待核对与焦点先验；不改变链接排序。
    #[test]
    fn test_mention_receipts_type_evidence_and_focus() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let receipts = linker.mention_receipts(
            "苹果的创始人是谁",
            &hints,
            &["人", "person"],
            Some("苹果公司"),
        );
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].mention, "苹果");
        let best = &receipts[0].candidates[0];
        assert_eq!(best.entity, "苹果公司");
        assert_eq!(best.method, LinkMethod::Alias);
        assert_eq!(best.entity_type.as_deref(), Some("公司"));
        // 「公司」不含「人/person」→ 与疑问词期待不符；类型证据仍如实展示
        assert_eq!(best.expectation_check, TypeVerdict::Mismatch);
        assert!(best.focus_prior, "候选中含当前焦点实体应标注先验");
        assert!(receipts[0]
            .candidates
            .iter()
            .any(|candidate| candidate.entity == "苹果果实" && !candidate.focus_prior));
    }

    /// 类型未知→Unknown；无期待→NotExpected；命中→Match（忽略大小写）。
    #[test]
    fn test_expectation_check_verdicts() {
        assert_eq!(
            expectation_check(Some("人物"), &["人", "person"]),
            TypeVerdict::Match
        );
        assert_eq!(
            expectation_check(Some("Person"), &["person"]),
            TypeVerdict::Match
        );
        assert_eq!(
            expectation_check(Some("公司"), &["人"]),
            TypeVerdict::Mismatch
        );
        assert_eq!(expectation_check(None, &["人"]), TypeVerdict::Unknown);
        assert_eq!(
            expectation_check(Some("公司"), &[]),
            TypeVerdict::NotExpected
        );
    }

    /// 账本渲染确定性：含类型证据、焦点先验与候选理由；两次渲染同字节。
    #[test]
    fn test_render_mention_ledger_is_deterministic() {
        let linker = sample_linker();
        let hints = vec!["创始人".to_string()];
        let expectations = ["人", "person"];
        let receipts =
            linker.mention_receipts("苹果的创始人是谁", &hints, &expectations, Some("苹果公司"));
        let first = render_mention_ledger("苹果的创始人是谁", &receipts, &expectations, &[]);
        let second = render_mention_ledger("苹果的创始人是谁", &receipts, &expectations, &[]);
        assert_eq!(first, second);
        assert!(first.contains("提及账本"));
        assert!(first.contains("类型证据：公司 → 与期待不符"));
        assert!(first.contains("焦点先验：当前对话焦点"));

        let empty = render_mention_ledger("今天天气怎么样", &[], &[], &[]);
        assert!(empty.contains("未识别到知识图谱中的实体"));
    }

    /// OOV 疑点：引号词与后缀词被提名，已链接区间不重复提名，KB 不变。
    #[test]
    fn test_oov_suspects_quoted_and_suffix() {
        let linker = sample_linker();
        let before = linker.len();
        let text = "想了解未名科技公司和《背影》的作者";
        let receipts = linker.mention_receipts(text, &[], &[], None);
        let suspects = oov_suspects(text, &receipts);
        let names: Vec<&str> = suspects.iter().map(|s| s.text.as_str()).collect();
        assert!(
            names.contains(&"未名科技公司"),
            "后缀词应被提名：{:?}",
            names
        );
        assert!(names.contains(&"背影"), "书名号词应被提名：{:?}", names);
        assert!(suspects
            .iter()
            .any(|s| s.text == "未名科技公司" && s.kind == OovKind::Suffix),);
        assert!(suspects
            .iter()
            .any(|s| s.text == "背影" && s.kind == OovKind::Quoted));
        // 已链接实体（刘德华）不进入 OOV 疑点
        let linked_text = "刘德华的公司";
        let linked_receipts = linker.mention_receipts(linked_text, &[], &[], None);
        let linked_suspects = oov_suspects(linked_text, &linked_receipts);
        assert!(
            linked_suspects
                .iter()
                .all(|s| s.text != "刘德华的" && s.text != "刘德华的公司"),
            "已链接区间不应提名：{:?}",
            linked_suspects
        );
        // KB 不变：账本只读
        assert_eq!(linker.len(), before);
    }

    /// 渲染含 OOV 段落时仍确定；无疑点时不出现该段落。
    #[test]
    fn test_render_mention_ledger_oov_section() {
        let linker = sample_linker();
        let text = "未名研究院的院长是谁";
        let receipts = linker.mention_receipts(text, &[], &["人"], None);
        let suspects = oov_suspects(text, &receipts);
        let rendered = render_mention_ledger(text, &receipts, &["人"], &suspects);
        assert!(rendered.contains("OOV 疑点（只报告，不落库）"));
        assert!(rendered.contains("未名研究院"));
        assert!(
            rendered.contains("暂定类型：组织"),
            "后缀表应给出暂定类型：{}",
            rendered
        );
        assert_eq!(
            render_mention_ledger(text, &receipts, &["人"], &suspects),
            rendered
        );

        let no_oov = render_mention_ledger("苹果的创始人是谁", &[], &[], &[]);
        assert!(!no_oov.contains("OOV 疑点"));
    }

    /// 格解码：嵌套指称等分时由「长词优先」（节点少者）裁决，选不重叠的最大证据路径。
    #[test]
    fn test_lattice_decoding_prefers_longest_exact_path_on_tie() {
        let mut linker = EntityLinker::new();
        linker.register_entity("北京大学", Some("组织"));
        linker.register_entity("北京", Some("地点"));
        linker.register_entity("大学", Some("组织"));
        let mentions = linker.link_lattice("北京大学的校长是谁", &[]);
        assert_eq!(
            mentions.len(),
            1,
            "等分时长词应压过拆分路径：{:?}",
            mentions
        );
        assert_eq!(mentions[0].mention, "北京大学");
        assert_eq!(mentions[0].best().unwrap().entity, "北京大学");
        assert!(mentions[0]
            .best()
            .unwrap()
            .reasons
            .iter()
            .any(|reason| reason.contains("全局选路")));
    }

    #[test]
    fn test_lattice_decoding_splits_when_long_form_unknown() {
        let mut linker = EntityLinker::new();
        linker.register_entity("北京", Some("地点"));
        linker.register_entity("大学", Some("组织"));
        let mentions = linker.link_lattice("北京大学的校长是谁", &[]);
        let surfaces: Vec<&str> = mentions.iter().map(|m| m.mention.as_str()).collect();
        assert_eq!(surfaces, vec!["北京", "大学"]);
        assert_eq!(
            linker.lattice_report("北京大学的校长是谁", &[]),
            linker.lattice_report("北京大学的校长是谁", &[]),
            "格解码报告必须确定可复现"
        );
        assert!(linker
            .lattice_report("今天天气怎么样", &[])
            .contains("未选出"));
    }

    #[test]
    fn test_lattice_decoding_includes_fuzzy_overlaps() {
        let linker = sample_linker();
        let mentions = linker.link_lattice("刘德桦的出生地是哪里", &[]);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].best().unwrap().entity, "刘德华");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::Fuzzy);
    }

    /// 身份注册表快照：别名与歧义原样保留（不静默取一），审计行确定。
    #[test]
    fn test_identity_registry_snapshot_keeps_aliases_and_ambiguity() {
        let linker = sample_linker();
        let registry = linker.identity_registry();
        assert_eq!(registry.len(), linker.len());
        assert_eq!(
            registry.resolve("苹果"),
            crate::identity::Resolution::Ambiguous(vec![
                "苹果公司".to_string(),
                "苹果果实".to_string(),
            ])
        );
        assert_eq!(
            registry.resolve("苹果公司"),
            crate::identity::Resolution::Canonical("苹果公司".to_string())
        );
        let report = registry.audit_lines().join("\n");
        assert!(report.contains("身份注册表"));
        assert!(report.contains("苹果公司"));
        assert!(report.contains("苹果果实"));
    }

    #[test]
    fn alignment_shape_links_unseen_surface_and_yields_to_exact_match() {
        let mut linker = EntityLinker::new();
        linker.register_entity("未名科技", Some("组织"));
        linker.register_entity("远山科技", Some("组织"));
        linker.register_entity("张三", Some("人物"));
        let outcome = linker.learn_alignment(
            1,
            "未名科技公司",
            "未名科技",
            AlignmentKind::Entity,
            Some("组织"),
            "用户纠正",
            AlignmentScope::Session,
        );
        assert!(outcome.changed());

        let mentions = linker.link("远山科技公司发布了新芯片");
        let aligned = mentions
            .iter()
            .find(|mention| mention.mention == "远山科技公司")
            .expect("形状规则应命中未登录词面");
        let best = aligned.best().unwrap();
        assert_eq!(best.entity, "远山科技");
        assert_eq!(best.method, LinkMethod::Alignment);

        let type_blocked = linker.link("张三公司");
        assert!(
            type_blocked
                .iter()
                .all(|mention| mention.candidates.iter().all(|candidate| {
                    candidate.method != LinkMethod::Alignment
                })),
            "类型守卫应阻止组织后缀套到人物身上"
        );

        linker.register_entity("远山科技公司", Some("组织"));
        let exact = linker.link("远山科技公司");
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].mention, "远山科技公司");
        assert_eq!(exact[0].best().unwrap().method, LinkMethod::ExactName);
    }

    #[test]
    fn alignment_nodes_enter_the_global_lattice() {
        let mut linker = EntityLinker::new();
        linker.register_entity("未名科技", Some("组织"));
        linker.register_entity("远山科技", Some("组织"));
        linker.learn_alignment(
            1,
            "未名科技公司",
            "未名科技",
            AlignmentKind::Entity,
            Some("组织"),
            "用户说法",
            AlignmentScope::Session,
        );
        let mentions = linker.link_lattice("远山科技公司发布了新芯片", &[]);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].mention, "远山科技公司");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::Alignment);
    }

    #[test]
    fn alignment_reverse_surface_must_re_link() {
        let mut linker = EntityLinker::new();
        linker.register_entity("未名科技", Some("组织"));
        linker.learn_alignment(
            1,
            "未名科技公司",
            "未名科技",
            AlignmentKind::Entity,
            Some("组织"),
            "用户说法",
            AlignmentScope::Session,
        );
        linker.alignment_mut().remember_echo(2, "未名科技公司", "未名科技");
        let resolve = |surface: &str| -> Vec<String> {
            linker
                .surface_candidates(surface)
                .into_iter()
                .map(|candidate| candidate.entity)
                .collect()
        };
        let type_of =
            |entity: &str| -> Option<String> { linker.entity_type(entity).map(str::to_string) };
        let (surface, reason) = linker
            .alignment()
            .surface_for_with("未名科技", 2, resolve, type_of)
            .unwrap();
        assert_eq!(surface, "未名科技公司");
        assert!(reason.contains("回声"));
        assert_eq!(
            linker.link(&surface)[0].best().unwrap().entity,
            "未名科技"
        );
    }

    /// OOV 提名带暂定类型；准入是显式动作且幂等，未准入前 KB 不变。
    #[test]
    fn test_oov_suggested_type_and_explicit_admission() {
        let mut linker = sample_linker();
        let before = linker.len();
        let text = "未名科技公司发布了新书";
        let receipts = linker.mention_receipts(text, &[], &[], None);
        let suspects = oov_suspects(text, &receipts);
        let company = suspects
            .iter()
            .find(|suspect| suspect.text == "未名科技公司")
            .expect("后缀词应被提名");
        assert_eq!(company.suggested_type.as_deref(), Some("组织"));
        assert!(!linker.is_known_entity("未名科技公司"));
        assert_eq!(linker.len(), before, "提名不落库");

        let (name, etype) = linker.admit_entity("未名科技公司").expect("准入应成功");
        assert_eq!(name, "未名科技公司");
        assert_eq!(etype, "组织");
        assert!(linker.is_known_entity("未名科技公司"));
        assert_eq!(linker.len(), before + 1);
        assert_eq!(
            linker.admit_entity("未名科技公司").unwrap().1,
            "组织",
            "重复准入幂等"
        );
        assert!(linker.admit_entity("   ").is_none());
        assert_eq!(
            linker.admit_entity("张三丰").unwrap().1,
            "Entity",
            "无后缀线索时显式准入为通用类型"
        );
    }
}

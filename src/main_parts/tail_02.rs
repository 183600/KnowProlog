/// 用相似成功经验为一句输入生成候选骨架（确认前不参与路由）。
fn suggest_case_for_text(
    text: &str,
    config: &know_prolog::SystemConfig,
    synonyms: &std::collections::HashMap<String, String>,
    episodes: &pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
) -> Option<pattern_ir::CaseSuggestion> {
    let tagged = preprocess(text, &config.nlp);
    let normalized = normalize_words(tagged, synonyms);
    episodes.suggest_case_with_context(&normalized, context, 0.2)
}

/// 统一模式库路由：返回（查询类型、实体、命中模式解释行）。
///
/// 路由语义与既有 `analyze_query_intent(&words, &rules)` 一致（同排序、同匹配、
/// 同实体提取），额外给出「命中模式」的人类可读说明；无命中返回 `None`。
/// 每次尝试都会写入 [`pattern_ir::PatternEpisodeLog`]：成功记下命中模式与绑定，
/// 失败也留下可检索的情境经验（不改变路由结果）。
fn exact_pattern_result(
    input: &str,
    words: &[know_prolog::nlp::TaggedWord],
    matched: pattern_ir::PatternMatch,
    episodes: &mut pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
) -> (String, Vec<String>, Option<String>) {
    episodes.record_match_with_context(input, words, &matched, context);
    let guard = matched
        .entry
        .guard
        .as_ref()
        .map(|guard| format!("，守卫：{}", guard.render()))
        .unwrap_or_default();
    let note = format!(
        "{}（{}，优先级 {}，来源 {}{}）",
        matched.entry.name,
        matched.entry.render(),
        matched.entry.priority,
        matched.entry.source,
        guard
    );
    (
        matched.entry.query_type.clone(),
        matched.entities.clone(),
        Some(note),
    )
}

fn route_query_with_patterns(
    library: &PatternLibrary,
    input: &str,
    words: &[know_prolog::nlp::TaggedWord],
    episodes: &mut pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
    soft_config: &know_prolog::soft_route::SoftRouteConfig,
) -> Option<(String, Vec<String>, Option<String>)> {
    // 第一层：配置主路由。泛化前缀模式命中后仍让软层复核一次：
    // 近失句里它可能只吃掉了前缀，挡住更具体的配置规则。
    // 软层不接管时保留这条精确命中，召回不倒退。
    if let Some(matched) = library.match_route_with_context(words, context) {
        if matched.entry.defer_to_soft && soft_config.enabled {
            let outcome =
                know_prolog::soft_route::route_tokens(input, library, words, context, soft_config);
            if outcome.decision == know_prolog::soft_route::SoftDecision::Accept {
                if let Some(candidate) = outcome.selected {
                    if candidate.entry.id() != matched.entry.id() {
                        let refined = candidate.to_pattern_match();
                        episodes.record_match_with_context(input, words, &refined, context);
                        let note = format!(
                            "软路由复核：{}（{}，理解置信度 {:.2}，对齐 {:.0}%，要素 {}/{}，token 覆盖 {:.0}%；原命中 {}）",
                            candidate.name(),
                            candidate.entry.render(),
                            candidate.confidence,
                            candidate.alignment.score * 100.0,
                            candidate.alignment.matched_elems,
                            candidate.alignment.total_elems,
                            candidate.alignment.token_coverage() * 100.0,
                            matched.entry.name
                        );
                        return Some((
                            refined.entry.query_type.clone(),
                            refined.entities.clone(),
                            Some(note),
                        ));
                    }
                }
            }
        }
        return Some(exact_pattern_result(
            input, words, matched, episodes, context,
        ));
    }

    // 第二层：软路由。主路由未命中时先于 NLU 回退评估，避免泛化的
    // 回退槽位（如「属性查询」）再次挡住更具体的配置近失解释。
    if soft_config.enabled {
        let outcome =
            know_prolog::soft_route::route_tokens(input, library, words, context, soft_config);
        if outcome.decision == know_prolog::soft_route::SoftDecision::Accept {
            if let Some(candidate) = outcome.selected {
                let matched = candidate.to_pattern_match();
                episodes.record_match_with_context(input, words, &matched, context);
                let note = format!(
                    "软路由：{}（{}，理解置信度 {:.2}，对齐 {:.0}%，要素 {}/{}，token 覆盖 {:.0}%）",
                    candidate.name(),
                    candidate.entry.render(),
                    candidate.confidence,
                    candidate.alignment.score * 100.0,
                    candidate.alignment.matched_elems,
                    candidate.alignment.total_elems,
                    candidate.alignment.token_coverage() * 100.0
                );
                return Some((
                    matched.entry.query_type.clone(),
                    matched.entities.clone(),
                    Some(note),
                ));
            }
        }
    }

    // 第三层：NLU 回退。软层拒绝/澄清后仍保留原有回退召回。
    if let Some(matched) = library.match_fallback_with_context(words, context) {
        return Some(exact_pattern_result(
            input, words, matched, episodes, context,
        ));
    }

    // 失败也要留下可行动的粒度：记录最近受阻模式停在哪一要素。
    let frontier_report = frontier::analyze_tokens(input, words, library, context);
    let snapshot = frontier_report.closest_blocked().and_then(|row| {
        row.failure
            .as_ref()
            .map(|failure| pattern_ir::EpisodeFrontier {
                pattern_id: row.id.clone(),
                pattern_name: row.name.clone(),
                query_type: row.query_type.clone(),
                coverage: row.coverage(),
                failure: failure.clone(),
            })
    });
    episodes.record_unmatched_with_frontier(input, words, context, snapshot);
    None
}

/// 感知层优先、旧切分兜底的路由：词表覆盖只增加边界证据，不牺牲旧召回。
fn route_query_with_lexicon(
    library: &PatternLibrary,
    input: &str,
    primary: &[know_prolog::nlp::TaggedWord],
    fallback: &[know_prolog::nlp::TaggedWord],
    episodes: &mut pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
    soft_config: &know_prolog::soft_route::SoftRouteConfig,
) -> Option<(String, Vec<String>, Option<String>)> {
    if let Some(hit) =
        route_query_with_patterns(library, input, primary, episodes, context, soft_config)
    {
        return Some(hit);
    }
    if primary != fallback {
        return route_query_with_patterns(library, input, fallback, episodes, context, soft_config);
    }
    None
}

/// 解析更正命令：`实体|属性|旧值|新值` 或 `实体 的 属性 从 旧值 改为 新值`。
///
/// 只认「有旧值」的更正：修正一条不知道旧值的陈述不是更正，而是无条件覆盖。
fn parse_correction_spec(target: &str) -> Option<(String, String, String, String)> {
    let trimmed = target
        .trim()
        .trim_matches(|ch: char| matches!(ch, '。' | '！' | '!' | '；' | ';' | '，' | ','));
    if trimmed.is_empty() {
        return None;
    }
    for separator in ['|', '｜'] {
        if trimmed.contains(separator) {
            let parts: Vec<&str> = trimmed
                .split(separator)
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect();
            if parts.len() == 4 {
                return Some((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                ));
            }
        }
    }
    let body = trimmed
        .strip_prefix("把")
        .or_else(|| trimmed.strip_prefix("将"))
        .unwrap_or(trimmed)
        .trim();
    for marker in ["改为", "改成", "更正为", "修正为", "变为", "换成"] {
        let Some(position) = body.find(marker) else {
            continue;
        };
        let left = body[..position].trim();
        let new_value = body[position + marker.len()..]
            .trim()
            .trim_matches(|ch: char| matches!(ch, '。' | '！' | '!' | '；' | ';' | '，' | ','));
        if new_value.is_empty() {
            continue;
        }
        for old_marker in ["从", "由", "原值为", "原值", "原本为", "原本"] {
            let Some(index) = left.rfind(old_marker) else {
                continue;
            };
            let head = left[..index].trim();
            let old_value = left[index + old_marker.len()..]
                .trim()
                .trim_start_matches([':', '：', '='])
                .trim();
            if old_value.is_empty() {
                continue;
            }
            let head = head.strip_suffix('是').unwrap_or(head).trim();
            if let Some((entity, predicate)) = head.rsplit_once('的') {
                let entity = entity.trim().trim_start_matches(['把', '将']).trim();
                let predicate = predicate.trim();
                if !entity.is_empty() && !predicate.is_empty() {
                    return Some((
                        entity.to_string(),
                        predicate.to_string(),
                        old_value.to_string(),
                        new_value.to_string(),
                    ));
                }
            }
        }
    }
    None
}

/// 中文句子 → 关系三元组：只取确定事实、二元、无占位符的原子，
/// 跳过「是」断言（类型断言不落成关系）。
/// 两条抽取通道合并去重：
///   1. 模式解析器 `parse_lessons`（「记住：」同款）；
///   2. DCG 文法抽取（被字句/把字句/并列/所有格换向/词库归一）。
fn extract_relationships_from_text(
    text: &str,
    words: &[know_prolog::nlp::TaggedWord],
    grammar_extractor: &GrammarExtractor,
) -> Vec<KnowledgeItem> {
    use know_prolog::chinese_reasoning_nlu::{parse_lessons, LessonKind};
    let outcome = parse_lessons(text);
    let mut seen: std::collections::BTreeSet<(String, String, String)> =
        std::collections::BTreeSet::new();
    let mut items = Vec::new();
    let mut push_triple = |from: String, predicate: String, to: String| {
        if predicate == "是" || from.is_empty() || to.is_empty() || from == to {
            return;
        }
        let key = (from.clone(), predicate.clone(), to.clone());
        if !seen.insert(key) {
            return;
        }
        items.push(KnowledgeItem::Relationship {
            from_label: String::new(),
            from_id: from,
            to_label: String::new(),
            to_id: to,
            rel_type: predicate,
            properties: std::collections::HashMap::new(),
        });
    };

    for lesson in outcome.lessons {
        if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
            continue;
        }
        let terms = &lesson.head.terms;
        if terms.len() != 2 {
            continue;
        }
        push_triple(
            terms[0].name().trim().to_string(),
            lesson.head.predicate.to_string(),
            terms[1].name().trim().to_string(),
        );
    }

    for triple in grammar_extractor.extract_from_tagged(words) {
        push_triple(triple.subject, triple.predicate, triple.object);
    }

    items
}

/// 把谓词画像注册表的词面同步到符号推理器：
/// 正向词面（别名 → 槽位）走别名通道（对齐同义词 + 生成词面表），
/// 逆向词面只登记查询换向（不改生成叙事词面，避免叙事错向）。
fn sync_predicate_pairs(reasoner: &mut SymbolicReasoner, oxigraph_manager: &OxigraphManager) {
    // 别名/换向词面与画像律典走同一注入入口（服务启动复用同一实现）
    reasoner.align_predicate_pairs(&oxigraph_manager.predicate_query_pairs());
    // 谓词画像的显式声明同步进律典：传递/对称/互逆律从此进入查询选路。
    reasoner.configure_laws(&oxigraph_manager.predicate_registry_snapshot());
}

/// 把实体身份注册表里已确认的持久别名同步进链接器（别名与规范名同权）。
fn sync_entity_aliases(reasoner: &mut SymbolicReasoner, oxigraph_manager: &OxigraphManager) {
    reasoner.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
}

/// 把全息命令参数切成若干词项：支持空白/中英文逗号/冒号分隔
fn parse_holo_terms(target: &str, expected: usize) -> Option<Vec<String>> {
    let terms: Vec<String> = target
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':' | '、'))
        .filter(|term| !term.is_empty())
        .map(|term| term.to_string())
        .collect();
    if terms.len() == expected {
        Some(terms)
    } else {
        None
    }
}

/// 解析全息检索的两元组：支持「主体 谓词」与「主语的谓词」两种写法
fn parse_holo_pair(target: &str) -> Option<(String, String)> {
    if let Some(terms) = parse_holo_terms(target, 2) {
        return Some((terms[0].clone(), terms[1].clone()));
    }
    let text = target.trim();
    if let Some(index) = text.find('的') {
        let subject = text[..index].trim();
        let predicate = text[index + '的'.len_utf8()..].trim();
        if !subject.is_empty() && !predicate.is_empty() {
            return Some((subject.to_string(), predicate.to_string()));
        }
    }
    None
}

/// 渲染全息召回候选（软召回分数 + 硬核查标记）
fn render_holo_hits(title: &str, hits: &[HoloHit]) -> String {
    if hits.is_empty() {
        return format!(
            "{}：全息图未召回候选（实体/谓词未登记，或该结构在知识中不存在）。",
            title
        );
    }
    let verified = hits.iter().filter(|hit| hit.verified).count();
    let mut out = format!(
        "{}（候选 {} 个，通过三元组核查 {} 个）：",
        title,
        hits.len(),
        verified
    );
    for (index, hit) in hits.iter().enumerate() {
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（串扰/推测，不入结论）"
        };
        out.push_str(&format!(
            "\n  {}. {}  余弦 {:.3}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            mark
        ));
    }
    out
}

/// 渲染全息类比候选（关系迁移路径 + 核查标记）
fn render_holo_analogy(a: &str, b: &str, c: &str, hits: &[HoloAnalogyHit]) -> String {
    if hits.is_empty() {
        return format!(
            "全息类比 {}:{} :: {}:? 未产生候选（需先登记一组对应事实，关系才能被解缠迁移）。",
            a, b, c
        );
    }
    let mut out = format!(
        "全息类比 {}:{} :: {}:?（关系解缠 → 迁移绑定 → 净噪）：",
        a, b, c
    );
    for (index, hit) in hits.iter().enumerate() {
        let via: Vec<String> = hit
            .via
            .iter()
            .map(|(predicate, score)| format!("{}（{:.2}）", predicate, score))
            .collect();
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（全息推测，供参考）"
        };
        out.push_str(&format!(
            "\n  {}. {}  综合分 {:.3}  关系通道：{}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            via.join("、"),
            mark
        ));
    }
    out
}

/// 解析中文数词（1..=99）：十一、二十、二十三、九十九
fn parse_chinese_numeral(text: &str) -> Option<usize> {
    let digit = |ch: char| -> Option<usize> {
        Some(match ch {
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            _ => return None,
        })
    };
    let chars: Vec<char> = text.chars().collect();
    match chars.as_slice() {
        ['十'] => Some(10),
        [single] => digit(*single),
        ['十', ones] => Some(10 + digit(*ones)?),
        [tens, '十'] => Some(digit(*tens)? * 10),
        [tens, '十', ones] => Some(digit(*tens)? * 10 + digit(*ones)?),
        _ => None,
    }
}

/// 解析「采纳假设 N」中的编号：支持阿拉伯数字与「一、二、三…十、十二、二十三」
fn parse_hypothesis_index(text: &str) -> Option<usize> {
    let trimmed = text.trim().trim_start_matches(['：', ':', ' ', '第']);
    let trimmed = trimmed.trim_end_matches('条').trim();
    if let Ok(index) = trimmed.parse::<usize>() {
        return (index > 0).then_some(index);
    }
    let index = parse_chinese_numeral(trimmed)?;
    (index > 0).then_some(index)
}

/// 解析歧义义项选择输入（1 基编号）：返回候选表 0 基下标。
///
/// 空输入 / 非数字 / 越界一律返回 `None`（调用方回退权重最高项）。
fn parse_ambiguous_choice(text: &str, options: usize) -> Option<usize> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(index) if (1..=options).contains(&index) => Some(index - 1),
        _ => None,
    }
}

/// 歧义实体的人工选择：提示用户按编号选择义项。
///
/// 仅在交互模式调用；空输入 / 非法输入 / EOF 均回退到权重最高项（第 1 项），
/// 返回候选表下标。
fn prompt_ambiguous_selection(
    entity: &str,
    candidates: &[(String, u64)],
    max_options: usize,
) -> usize {
    let options = candidates.len().min(max_options.max(1));
    print!(
        "请选择「{}」的义项编号 [1-{}]（直接回车选权重最高项）: ",
        entity, options
    );
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return 0;
    }
    match parse_ambiguous_choice(&line, options) {
        Some(index) => index,
        None => {
            if !line.trim().is_empty() {
                println!("输入「{}」无效，使用权重最高项。", line.trim());
            }
            0
        }
    }
}

/// 对查询实体做歧义解析：自动模式结合整句上下文选择义项，
/// 人工模式在交互界面提示用户按编号选择。
///
/// 无候选或调用失败时保留原实体名，保证返回结果与输入实体一一对应。
async fn resolve_query_entities_with_disambiguation(
    input: &str,
    query_entities: &[String],
    api_config: &know_prolog::ApiConfig,
    adapter: &LocalSemanticAdapter,
) -> Vec<String> {
    println!("正在解析实体歧义...");
    let mut resolved_entities = Vec::new();
    let mut found_ambiguous = false;

    for entity in query_entities {
        match adapter.get_ambiguous_entities(entity).await {
            Ok(ambiguous_response) => {
                if ambiguous_response.data.len() > 1 {
                    found_ambiguous = true;
                    println!("发现实体 '{}' 有多种含义:", entity);
                    for (i, (entity_name, weight)) in ambiguous_response
                        .data
                        .iter()
                        .take(api_config.max_ambiguous_entities)
                        .enumerate()
                    {
                        println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                    }

                    let chosen_entity = if api_config.auto_select_best_entity {
                        let index =
                            select_candidate_by_context(&ambiguous_response.data, Some(input))
                                .unwrap_or(0);
                        let (best_entity, _) = &ambiguous_response.data[index];
                        println!("自动选择（结合上下文）: {}", best_entity);
                        best_entity.clone()
                    } else {
                        let index = prompt_ambiguous_selection(
                            entity,
                            &ambiguous_response.data,
                            api_config.max_ambiguous_entities,
                        );
                        let (chosen, _) = &ambiguous_response.data[index];
                        println!("已选择: {}", chosen);
                        chosen.clone()
                    };
                    resolved_entities.push(chosen_entity.clone());

                    match adapter.get_simplified_knowledge(&chosen_entity).await {
                        Ok(knowledge) => {
                            if let Some(desc) = knowledge.get("description") {
                                if !desc.is_empty() {
                                    println!("  描述: {}", desc);
                                }
                            }
                            if let Some(tags) = knowledge.get("tags") {
                                if !tags.is_empty() {
                                    println!("  标签: {}", tags);
                                }
                            }
                        }
                        Err(e) => eprintln!("获取实体知识时出错: {}", e),
                    }
                } else if ambiguous_response.data.len() == 1 {
                    if let Some((entity_name, _)) = ambiguous_response.data.first() {
                        resolved_entities.push(entity_name.clone());
                    }
                } else {
                    resolved_entities.push(entity.clone());
                }
            }
            Err(e) => {
                eprintln!("查询实体 '{}' 的歧义信息时出错: {}", entity, e);
                resolved_entities.push(entity.clone());
            }
        }
    }

    if found_ambiguous {
        println!("─────────────────────────────────");
    }
    resolved_entities
}

async fn continue_original_query(
    config: &know_prolog::SystemConfig,
    oxigraph_manager: &OxigraphManager,
    prolog_engine: &mut PrologEngine,
    response_formatter: &ResponseFormatter,
    local_semantic_adapter: &LocalSemanticAdapter,
    external_knowledge_service: &ExternalKnowledgeService,
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    plugin_runtime: &mut know_prolog::plugin::PluginRuntime,
    input: &str,
    query_type: &str,
    entities: &[String],
    precomputed_query_entities: &[String],
) {
    println!("查询类型: {}, 实体: {:?}", query_type, entities);

    // 提取查询中的实体
    let mut seen_entities = std::collections::HashSet::new();
    let mut query_entities: Vec<String> =
        Vec::with_capacity(precomputed_query_entities.len() + entities.len());
    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
        if seen_entities.insert(entity.as_str()) {
            query_entities.push(entity.clone());
        }
    }

    // 使用歧义解析功能（如果启用）
    if let Some(api_config) = &config.api {
        if api_config.enable_entity_disambiguation {
            let resolved_entities = resolve_query_entities_with_disambiguation(
                input,
                &query_entities,
                api_config,
                local_semantic_adapter,
            )
            .await;

            let final_entities = if !resolved_entities.is_empty() {
                resolved_entities
            } else {
                entities.to_vec()
            };

            let max_hops = 2;
            match oxigraph_manager
                .get_relevant_facts(&final_entities, max_hops)
                .await
            {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);

                    if let Some(answer) = oxigraph_manager
                        .get_direct_answer(query_type, &final_entities)
                        .await
                    {
                        println!("{}", answer);
                    } else if let Some(prolog_query) =
                        PrologEngine::generate_prolog_query(query_type, &final_entities)
                    {
                        println!("Prolog查询: {}", prolog_query);

                        let response = run_prolog_query(
                            prolog_engine,
                            symbolic_reasoner,
                            &prolog_query,
                            query_type,
                            response_formatter,
                        );
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) = try_symbolic_reasoning(
                        symbolic_reasoner,
                        contrast_arbiter,
                        oxigraph_manager,
                        input,
                        precomputed_query_entities,
                    )
                    .await
                    {
                        println!("{}", answer);
                    }
                }
            }
        } else {
            // 歧义解析功能被禁用
            let max_hops = 2;
            match oxigraph_manager
                .get_relevant_facts(&query_entities, max_hops)
                .await
            {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);

                    if let Some(answer) = oxigraph_manager
                        .get_direct_answer(query_type, entities)
                        .await
                    {
                        println!("{}", answer);
                    } else if let Some(prolog_query) =
                        PrologEngine::generate_prolog_query(query_type, entities)
                    {
                        println!("Prolog查询: {}", prolog_query);

                        let response = run_prolog_query(
                            prolog_engine,
                            symbolic_reasoner,
                            &prolog_query,
                            query_type,
                            response_formatter,
                        );
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) = try_symbolic_reasoning(
                        symbolic_reasoner,
                        contrast_arbiter,
                        oxigraph_manager,
                        input,
                        precomputed_query_entities,
                    )
                    .await
                    {
                        println!("{}", answer);
                    }
                }
            }
        }
    } else {
        // 没有API配置
        let max_hops = 2;
        match oxigraph_manager
            .get_relevant_facts(&query_entities, max_hops)
            .await
        {
            Ok(facts) => {
                println!("加载了 {} 个相关事实", facts.len());
                prolog_engine.load_relevant_facts(facts);

                if let Some(answer) = oxigraph_manager
                    .get_direct_answer(query_type, entities)
                    .await
                {
                    println!("{}", answer);
                } else if let Some(prolog_query) =
                    PrologEngine::generate_prolog_query(query_type, entities)
                {
                    println!("Prolog查询: {}", prolog_query);

                    let response = run_prolog_query(
                        prolog_engine,
                        symbolic_reasoner,
                        &prolog_query,
                        query_type,
                        response_formatter,
                    );
                    println!("{}", response);
                } else {
                    println!("无法生成有效的Prolog查询");
                }
            }
            Err(e) => {
                println!("{}: {}", config.messages.process_error, e);

                if let Some(answer) = try_symbolic_reasoning(
                    symbolic_reasoner,
                    contrast_arbiter,
                    oxigraph_manager,
                    input,
                    precomputed_query_entities,
                )
                .await
                {
                    println!("{}", answer);
                } else {
                    // 本地查询失败时，先给插件声明式兜底一次机会，再尝试外部知识API
                    println!("本地查询失败，尝试外部知识源...");

                    if let Some((text, _plan)) = plugin_policy_text(plugin_runtime, input) {
                        println!("{}", text);
                    } else if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                    } else {
                        match external_knowledge_service
                            .answer_question_with_entities(input, &query_entities)
                            .await
                        {
                            Ok(outcome) => match external_outcome_text(&outcome) {
                                Some(text) => println!("{}", text),
                                None => println!("很抱歉，我无法找到相关信息。"),
                            },
                            Err(e) => {
                                println!("查询外部知识时出错：{}。", e);
                            }
                        }
                    }
                }
            }
        }
    }
}

{
                {
                    // 知识导入逻辑
                    println!("{}", language_config.messages.importing);

                    let clean_text = language_config
                        .nlp
                        .import_keywords
                        .iter()
                        .fold(input.to_string(), |text, keyword| {
                            // 关键词未出现时原样返回
                            if !text.contains(keyword.as_str()) {
                                return text;
                            }
                            text.replace(&format!("{}：", keyword), "")
                                .replace(&format!("{}:", keyword), "")
                        })
                        .trim()
                        .to_string();

                    let tagged_words = preprocess(&clean_text, &plugin_nlp);
                    let normalized_words = normalize_words(tagged_words, &synonyms);
                    let mut items = knowledge_extractor.extract(&clean_text, &normalized_words);
                    let learned = batch_teaching.extract_active(
                        &language_config,
                        &clean_text,
                        &normalized_words,
                    );
                    let learned_added = append_learned_items(&mut items, learned);
                    if learned_added > 0 {
                        println!("批量教学规则命中 {} 条。", learned_added);
                    }
                    // 中文关系句 → 关系三元组（如「张三是李四的师傅」），
                    // 叠加 DCG 文法抽取（被字句/把字句/并列/所有格换向）
                    grammar_extractor
                        .set_lexicon_if_changed(&oxigraph_manager.predicate_query_pairs());
                    items.extend(extract_relationships_from_text(
                        &clean_text,
                        &normalized_words,
                        &grammar_extractor,
                    ));
                    // 已抽出显式关系时，通用兜底实体会把整句内容词拼成
                    // property（如「张三李四老师」），属于重复噪声，予以剔除；
                    // 具名类型实体（Person 等）保留。
                    if items
                        .iter()
                        .any(|item| matches!(item, KnowledgeItem::Relationship { .. }))
                    {
                        items.retain(|item| {
                            !matches!(item, KnowledgeItem::Entity { label, .. } if label == "Entity")
                        });
                    }

                    if items.is_empty() {
                        println!("{}", language_config.messages.no_knowledge_extracted);
                    } else {
                        for item in &items {
                            match item {
                                KnowledgeItem::Entity { label, properties } => {
                                    println!("实体 - 类型: {}, 属性: {:?}", label, properties);
                                }
                                KnowledgeItem::Relationship {
                                    from_label,
                                    from_id,
                                    to_label,
                                    to_id,
                                    rel_type,
                                    ..
                                } => {
                                    println!(
                                        "关系 - {}({}) -[{}]-> {}({})",
                                        from_label, from_id, rel_type, to_label, to_id
                                    );
                                }
                            }
                        }

                        let frame_items = items.clone();
                        match oxigraph_manager.save_knowledge_with_receipt(items).await {
                            Ok(receipt) => {
                                let summary = receipt.summary();
                                println!("{}", summary);
                                if let Some(chronicle) = chronicle.as_mut() {
                                    let _ = conversation_theory::append_import(
                                        chronicle,
                                        &receipt,
                                        "文本录入",
                                        kb_revision + 1,
                                    );
                                    if !receipt.conflicts.is_empty() {
                                        let _ = conversation_theory::append_conflict(
                                            chronicle,
                                            &receipt.conflicts,
                                            "文本录入",
                                            kb_revision + 1,
                                        );
                                    }
                                }
                                log_chronicle_reply(&mut chronicle, &summary, &[], &[]);
                                // 抽取即标注：把已落库事实回对齐到原句，学习可逆措辞/问法。
                                let learned = import_frames.learn_from_import(
                                    kb_revision + 1,
                                    &clean_text,
                                    &frame_items,
                                    &receipt,
                                );
                                if learned.changed() {
                                    if let Err(error) = import_frames.save(&import_frames_path) {
                                        eprintln!("保存导入帧失败：{}", error);
                                    }
                                    symbolic_reasoner.attach_import_frames(import_frames.clone());
                                    println!("{}", learned.summary());
                                }
                                if receipt.can_undo() {
                                    import_journal.push(receipt);
                                    let _ = import_journal.save(&journal_path);
                                }
                                reasoner_stale = true;
                                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                                // 文本别名模式（全称/简称/又称…）→ 只提名，不自动改写身份
                                let candidates = extract_alias_candidates(&clean_text);
                                if !candidates.is_empty() {
                                    let outcomes =
                                        oxigraph_manager.nominate_entity_aliases(&candidates);
                                    let pending = outcomes
                                        .iter()
                                        .filter(|o| o.kind == NominationKind::Pending)
                                        .count();
                                    let conflicts = outcomes
                                        .iter()
                                        .filter(|o| o.kind == NominationKind::TypeConflict)
                                        .count();
                                    if pending > 0 || conflicts > 0 {
                                        println!(
                                            "身份提名：待确认 {} 个、类型冲突 {} 个（用「身份台账」查看，「确认身份：A = B」裁决）。",
                                            pending, conflicts
                                        );
                                    }
                                }
                                let _ =
                                    oxigraph_manager.save_entity_registry(&entity_registry_path);
                            }
                            Err(e) => eprintln!("保存失败: {}", e),
                        }
                    }
                }
}

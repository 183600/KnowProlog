// src/main.rs
use std::io::{BufRead, Write};
use std::sync::{Arc, OnceLock};
use clap::{Parser, Subcommand};

// 从库（know_prolog）中导入所需的功能
use know_prolog::{
    load_config, 
    KnowledgeExtractor, 
    OxigraphManager,
    ResponseFormatter,
    InputMode,
    KnowledgeItem,
    create_prolog_rules,
    detect_input_mode,
    create_default_synonyms,
    extract_query_entities,
    nlp::{preprocess, normalize_words, analyze_query_intent},
    nlu::NLUEngine,
    prolog_engine::PrologEngine,
    local_semantic_adapter::LocalSemanticAdapter,
    external_knowledge::{ExternalKnowledgeService, get_direct_answer},
};

#[derive(Parser)]
#[command(name = "knowprolog")]
#[command(about = "KnowProlog - Chinese knowledge graph system")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

// 实体属性查询模式是固定不变的，全局编译一次即可
static ENTITY_ATTRIBUTE_RE: OnceLock<regex::Regex> = OnceLock::new();

#[derive(Subcommand, Debug)]
enum Commands {
    /// Import JSON-LD knowledge base from file
    ImportJsonld {
        /// Path to the JSON-LD file
        file_path: String,
    },
    /// Import JSON-LD knowledge base from content
    ImportJsonldContent {
        /// JSON-LD content as string
        content: String,
    },
    /// Run interactive mode (default)
    Interactive,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    let config = match load_config().await {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            eprintln!("加载配置文件失败: {}", e);
            return;
        }
    };

    // 创建Oxigraph存储
    let oxigraph_manager = match OxigraphManager::new(&config.system.oxigraph_store_path) {
        Ok(manager) => manager,
        Err(e) => {
            eprintln!("创建Oxigraph存储失败: {}", e);
            return;
        }
    };

    // 处理命令行参数
    println!("Debug: cli.command = {:?}", cli.command);
    match cli.command {
        Some(Commands::ImportJsonld { file_path }) => {
            println!("正在导入 JSON-LD 文件: {}", file_path);
            
            match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                Ok(_) => {
                    println!("JSON-LD 文件导入成功！");
                }
                Err(e) => {
                    eprintln!("JSON-LD 文件导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportJsonldContent { content }) => {
            println!("正在导入 JSON-LD 内容...");
            
            match oxigraph_manager.import_jsonld(&content).await {
                Ok(_) => {
                    println!("JSON-LD 内容导入成功！");
                }
                Err(e) => {
                    eprintln!("JSON-LD 内容导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Interactive) | None => {
            // 运行交互模式
            run_interactive_mode(config, oxigraph_manager).await;
        }
    }
}

async fn run_interactive_mode(config: Arc<know_prolog::SystemConfig>, oxigraph_manager: OxigraphManager) {
    println!("{}", config.messages.startup);

    let knowledge_extractor = KnowledgeExtractor::new(config.clone());
    let response_formatter = ResponseFormatter::new(config.clone());
    let nlu_engine = NLUEngine::new(&config);
    let prolog_rules = create_prolog_rules(&config);
    let mut prolog_engine = PrologEngine::new();
    let local_semantic_adapter = {
        let nlp_config = &config.nlp;
        LocalSemanticAdapter::new(
            nlp_config.enable_local_semantic.unwrap_or(true) && 
            nlp_config.disable_ownthink_api.unwrap_or(false)
        )
    };
    
    // 创建外部知识服务
    let external_knowledge_service = ExternalKnowledgeService::new();
    
    let synonyms = if let Some(synonyms_config) = &config.synonyms {
        synonyms_config.mappings.clone()
    } else {
        create_default_synonyms()
    };

    println!("{}", config.messages.ready);
    println!("{}", config.messages.query_prompt);
    println!("{}", config.messages.import_prompt);
    println!("{}", config.messages.quit_prompt);
    println!();

    let stdin_reader = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        match stdin_reader.lock().read_line(&mut input) {
            Ok(0) => break, // EOF reached, exit the loop
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        let input = input.trim();

        if input == "quit" {
            break;
        }

        let mode = detect_input_mode(input, &config.nlp.import_keywords);

        match mode {
            InputMode::KnowledgeImport => {
                // 检查是否是 JSON-LD 文件导入
                if input.starts_with("导入jsonld文件：") || input.starts_with("import jsonld:") {
                    let file_path = input.replace("导入jsonld文件：", "")
                                        .replace("import jsonld:", "")
                                        .trim()
                                        .to_string();
                    
                    if file_path.is_empty() {
                        println!("请提供 JSON-LD 文件路径");
                        continue;
                    }
                    
                    println!("正在导入 JSON-LD 文件: {}", file_path);
                    
                    match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                        Ok(_) => {
                            println!("JSON-LD 文件导入成功！");
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 文件导入失败: {}", e);
                        }
                    }
                } else if input.starts_with("导入jsonld内容：") || input.starts_with("import jsonld content:") {
                    let jsonld_content = input.replace("导入jsonld内容：", "")
                                             .replace("import jsonld content:", "")
                                             .trim()
                                             .to_string();
                    
                    if jsonld_content.is_empty() {
                        println!("请提供 JSON-LD 内容");
                        continue;
                    }
                    
                    println!("正在导入 JSON-LD 内容...");
                    
                    match oxigraph_manager.import_jsonld(&jsonld_content).await {
                        Ok(_) => {
                            println!("JSON-LD 内容导入成功！");
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 内容导入失败: {}", e);
                        }
                    }
                } else {
                    // 原有的知识导入逻辑
                    println!("{}", config.messages.importing);
                    
                    let clean_text = config.nlp.import_keywords.iter()
                        .fold(input.to_string(), |text, keyword| {
                            text.replace(&format!("{}：", keyword), "")
                                .replace(&format!("{}:", keyword), "")
                        })
                        .trim()
                        .to_string();
                    
                    let tagged_words = preprocess(&clean_text, &config.nlp);
                    let normalized_words = normalize_words(tagged_words, &synonyms);
                    let items = knowledge_extractor.extract(&clean_text, &normalized_words);
                    
                    if items.is_empty() {
                        println!("{}", config.messages.no_knowledge_extracted);
                    } else {
                        for item in &items {
                            match item {
                                KnowledgeItem::Entity { label, properties } => {
                                    println!("实体 - 类型: {}, 属性: {:?}", label, properties);
                                }
                                KnowledgeItem::Relationship { from_label, from_id, to_label, to_id, rel_type, .. } => {
                                    println!("关系 - {}({}) -[{}]-> {}({})",
                                        from_label, from_id, rel_type, to_label, to_id);
                                }
                            }
                        }
                        
                        match oxigraph_manager.save_knowledge(items).await {
                            Ok(_) => println!("知识已成功保存"),
                            Err(e) => eprintln!("保存失败: {}", e),
                        }
                    }
                }
            }
            InputMode::Query => {
                let tagged_words = preprocess(input, &config.nlp);
                let normalized_words = normalize_words(tagged_words, &synonyms);
                
                // 使用NLU进行语义理解
                if config.nlp.enable_nlu.unwrap_or(false) {
                    let semantic_analysis = nlu_engine.analyze(input);
                    
                    if !semantic_analysis.semantics.is_empty() {
                        let best_semantic = &semantic_analysis.semantics[0];
                        let confidence_threshold = config.nlp.nlu_confidence_threshold.unwrap_or(0.6);
                        
                        if best_semantic.score >= confidence_threshold {
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("NLU语义理解结果:");
                            println!("  领域: {}", best_semantic.domain);
                            println!("  意图: {}", best_semantic.intent);
                            println!("  置信度: {:.2}", best_semantic.score);
                            if !best_semantic.slots.is_empty() {
                                println!("  词槽:");
                                for (slot_name, slot_value) in &best_semantic.slots {
                                    println!("    {}: {}", slot_name, slot_value);
                                }
                            }
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            
                            // 根据NLU结果处理查询
                            match best_semantic.domain.as_str() {
                                "天气" => {
                                    let city = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "城市")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "未知城市".to_string());
                                    let date = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "日期")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "今天".to_string());
                                    
                                    match best_semantic.intent.as_str() {
                                        "雨" | "天气查询" => {
                                            println!("正在查询{}{}的天气信息...", city, date);
                                            println!("（此功能需要连接天气API服务）");
                                        }
                                        _ => {
                                            println!("暂不支持该天气查询类型: {}", best_semantic.intent);
                                        }
                                    }
                                }
                                "人物" => {
                                    let person = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "人名")
                                        .map(|(_, value)| value.clone());
                                    
                                    if let Some(person_name) = person {
                                        match best_semantic.intent.as_str() {
                                            "身高查询" => {
                                                println!("正在查询{}的身高信息...", person_name);
                                                // 这里可以调用现有的知识图谱查询逻辑
                                                match local_semantic_adapter.get_entity_knowledge(&person_name).await {
                                                    Ok(response) => {
                                                        println!("─────────────────────────────────");
                                                        println!("实体: {}", response.data.entity);
                                                        println!("描述: {}", response.data.desc);
                                                        for avp in &response.data.avp {
                                                            if avp.len() >= 2 && (avp[0].contains("身高") || avp[0].contains("高度")) {
                                                                println!("身高: {}", avp[1]);
                                                            }
                                                        }
                                                        println!("─────────────────────────────────");
                                                    }
                                                    Err(e) => {
                                                        println!("获取{}身高信息时出错: {}", person_name, e);
                                                    }
                                                }
                                            }
                                            "年龄查询" => {
                                                println!("正在查询{}的年龄信息...", person_name);
                                                // 使用现有的查询逻辑
                                            }
                                            _ => {
                                                println!("暂不支持该人物查询类型: {}", best_semantic.intent);
                                            }
                                        }
                                    } else {
                                        println!("未能识别人名，请重新输入");
                                    }
                                }
                                "聊天" => {
                                    match best_semantic.intent.as_str() {
                                        "问候" => {
                                            println!("你好！我是KnowProlog知识图谱系统，很高兴为您服务！");
                                        }
                                        "感谢" => {
                                            println!("不客气！如果您有其他问题，随时可以问我。");
                                        }
                                        _ => {
                                            println!("很高兴和您聊天！");
                                        }
                                    }
                                }
                                "知识查询" => {
                                    // 继续使用原有的查询逻辑
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        // 执行原有的查询流程
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, input, &query_type, &entities).await;
                                    } else {
                                        println!("无法理解您的查询，请重新描述您的问题。");
                                    }
                                }
                                _ => {
                                    println!("暂不支持该领域: {}", best_semantic.domain);
                                    // 如果NLU不能处理，回退到原有逻辑
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, input, &query_type, &entities).await;
                                    } else {
                                        println!("{}", config.messages.cannot_understand);
                                    }
                                }
                            }
                            continue; // 跳过原有的查询逻辑
                        }
                    }
                    
                    // 如果NLU置信度不够，回退到原有逻辑
                    println!("NLU置信度不足，使用传统模式处理查询...");
                }
                
                // 检查是否是直接的实体知识查询（例如："刘德华的信息" 或 "告诉我关于刘德华"）
                if input.contains("的信息") || input.contains("告诉我关于") || input.contains("什么是") {
                    // 提取实体名
                    let entity_name = input
                        .replace("的信息", "")
                        .replace("告诉我关于", "")
                        .replace("什么是", "")
                        .trim()
                        .to_string();
                    
                    if !entity_name.is_empty() {
                        println!("正在获取实体 '{}' 的知识...", entity_name);
                        
                        match local_semantic_adapter.get_entity_knowledge(&entity_name).await {
                            Ok(response) => {
                                println!("─────────────────────────────────");
                                println!("实体: {}", response.data.entity);
                                println!("描述: {}", response.data.desc);
                                if !response.data.tag.is_empty() {
                                    println!("标签: {}", response.data.tag.join(", "));
                                }
                                println!("\n属性信息:");
                                for avp in &response.data.avp {
                                    if avp.len() >= 2 {
                                        println!("  {}: {}", avp[0], avp[1]);
                                    }
                                }
                                println!("─────────────────────────────────");
                                continue;
                            }
                            Err(e) => {
                                println!("获取实体知识时出错: {}，尝试本地查询...", e);
                            }
                        }
                    }
                }
                
                // 检查是否是实体属性查询（例如："刘德华的年龄" 或 "苹果的颜色"）
                let attribute_re = ENTITY_ATTRIBUTE_RE.get_or_init(|| regex::Regex::new(r"(.+?)的(.+?)$").unwrap());
                if let Some(captures) = attribute_re.captures(input) {
                    if let (Some(entity_match), Some(attr_match)) = (captures.get(1), captures.get(2)) {
                        let entity_name = entity_match.as_str().trim();
                        let attribute = attr_match.as_str().trim();
                        
                        // 排除已经处理过的模式
                        if !input.contains("信息") && !input.contains("什么") {
                            println!("正在查询实体 '{}' 的属性 '{}'...", entity_name, attribute);
                            
                            match local_semantic_adapter.get_entity_attribute_value(entity_name, attribute).await {
                                Ok(response) => {
                                    println!("─────────────────────────────────");
                                    println!("实体: {}", response.data.entity);
                                    println!("属性: {}", response.data.attribute);
                                    println!("值: {}", response.data.value.join(", "));
                                    println!("─────────────────────────────────");
                                    continue;
                                }
                                Err(e) => {
                                    println!("获取属性值时出错: {}，尝试本地查询...", e);
                                }
                            }
                        }
                    }
                }
                
                if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                    println!("查询类型: {}, 实体: {:?}", query_type, entities);
                    
                    // 提取查询中的实体
                    let mut query_entities = extract_query_entities(input, &config);
                    query_entities.extend(entities.clone());
                    query_entities.dedup();
                    
                    // === 新增歧义解析功能 ===
                    if let Some(api_config) = &config.api {
                        if api_config.enable_entity_disambiguation {
                            println!("正在解析实体歧义...");
                            let mut resolved_entities = Vec::new();
                            let mut found_ambiguous = false;
                            
                            for entity in &query_entities {
                                match local_semantic_adapter.get_ambiguous_entities(entity).await {
                                    Ok(ambiguous_response) => {
                                        if ambiguous_response.data.len() > 1 {
                                            found_ambiguous = true;
                                            println!("发现实体 '{}' 有多种含义:", entity);
                                            for (i, (entity_name, weight)) in ambiguous_response.data.iter()
                                                .take(api_config.max_ambiguous_entities).enumerate() {
                                                println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                                            }
                                            
                                            // 根据配置决定是否自动选择
                                            if api_config.auto_select_best_entity {
                                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                                    println!("自动选择: {}", best_entity);
                                                    resolved_entities.push(best_entity.clone());
                                                    
                                                    // 尝试获取并显示该实体的知识
                                                    match local_semantic_adapter.get_simplified_knowledge(best_entity).await {
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
                                                        Err(e) => {
                                                            eprintln!("获取实体知识时出错: {}", e);
                                                        }
                                                    }
                                                }
                                            } else {
                                                // TODO: 实现用户交互选择逻辑
                                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                                    resolved_entities.push(best_entity.clone());
                                                }
                                            }
                                        } else if ambiguous_response.data.len() == 1 {
                                            // 只有一个匹配，直接使用
                                            if let Some((entity_name, _)) = ambiguous_response.data.first() {
                                                resolved_entities.push(entity_name.clone());
                                            }
                                        } else {
                                            // 没有找到匹配，使用原始实体名
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
                            
                            // 使用解析后的实体进行查询
                            let final_entities = if !resolved_entities.is_empty() {
                                resolved_entities
                            } else {
                                entities.clone()
                            };
                            
                            // 优化：仅加载相关知识子集
                            let max_hops = 2; // 可配置的跳跃数
                            match oxigraph_manager.get_relevant_facts(&final_entities, max_hops).await {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);
                                    
                                    // 先尝试直接从RDF数据获取答案
                                    if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &final_entities).await {
                                        println!("{}", answer);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &final_entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        } else {
                            // 歧义解析功能被禁用，使用原始流程
                            let max_hops = 2;
                            match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);
                                    
                                    if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &entities).await {
                                        println!("{}", answer);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        }
                    } else {
                        // 没有API配置，使用原始流程
                        let max_hops = 2;
                        match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                            Ok(facts) => {
                                println!("加载了 {} 个相关事实", facts.len());
                                prolog_engine.load_relevant_facts(facts);
                                
                                if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &entities).await {
                                    println!("{}", answer);
                                } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                    println!("Prolog查询: {}", prolog_query);
                                    
                                    let results = prolog_engine.query(&prolog_query);
                                    
                                    let response = response_formatter.format_prolog_results(&results, &query_type);
                                    println!("{}", response);
                                } else {
                                    println!("无法生成有效的Prolog查询");
                                }
                            }
                            Err(e) => {
                                println!("{}: {}", config.messages.process_error, e);
                            }
                        }
                    }
                } else {
                    println!("{}", config.messages.cannot_understand);
                    
                    // 尝试使用外部知识API
                    if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                    } else {
                        // 尝试使用OwnThink API查询
                        match external_knowledge_service.answer_question(input).await {
                            Ok(Some(answer)) => {
                                println!("根据外部知识：{}", answer);
                            }
                            Ok(None) => {
                                println!("很抱歉，我无法理解您的问题。请尝试重新描述。");
                            }
                            Err(e) => {
                                println!("查询外部知识时出错：{}。请检查网络连接。", e);
                            }
                        }
                    }
                }
            }
        }

        println!();
    }

    println!("系统已退出。");
}

async fn continue_original_query(
    config: &know_prolog::SystemConfig,
    oxigraph_manager: &OxigraphManager,
    prolog_engine: &mut PrologEngine,
    response_formatter: &ResponseFormatter,
    local_semantic_adapter: &LocalSemanticAdapter,
    external_knowledge_service: &ExternalKnowledgeService,
    input: &str,
    query_type: &str,
    entities: &[String]
) {
    println!("查询类型: {}, 实体: {:?}", query_type, entities);
    
    // 提取查询中的实体
    let mut query_entities = extract_query_entities(input, config);
    query_entities.extend(entities.iter().cloned());
    query_entities.dedup();
    
    // 使用歧义解析功能（如果启用）
    if let Some(api_config) = &config.api {
        if api_config.enable_entity_disambiguation {
            println!("正在解析实体歧义...");
            let mut resolved_entities = Vec::new();
            let mut found_ambiguous = false;
            
            for entity in &query_entities {
                match local_semantic_adapter.get_ambiguous_entities(entity).await {
                    Ok(ambiguous_response) => {
                        if ambiguous_response.data.len() > 1 {
                            found_ambiguous = true;
                            println!("发现实体 '{}' 有多种含义:", entity);
                            for (i, (entity_name, weight)) in ambiguous_response.data.iter()
                                .take(api_config.max_ambiguous_entities).enumerate() {
                                println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                            }
                            
                            if api_config.auto_select_best_entity {
                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                    println!("自动选择: {}", best_entity);
                                    resolved_entities.push(best_entity.clone());
                                    
                                    match local_semantic_adapter.get_simplified_knowledge(best_entity).await {
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
                                        Err(e) => {
                                            eprintln!("获取实体知识时出错: {}", e);
                                        }
                                    }
                                }
                            } else {
                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                    resolved_entities.push(best_entity.clone());
                                }
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
            
            let final_entities = if !resolved_entities.is_empty() {
                resolved_entities
            } else {
                entities.to_vec()
            };
            
            let max_hops = 2;
            match oxigraph_manager.get_relevant_facts(&final_entities, max_hops).await {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);
                    
                    if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, &final_entities).await {
                        println!("{}", answer);
                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, &final_entities) {
                        println!("Prolog查询: {}", prolog_query);
                        
                        let results = prolog_engine.query(&prolog_query);
                        let response = response_formatter.format_prolog_results(&results, query_type);
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                }
            }
        } else {
            // 歧义解析功能被禁用
            let max_hops = 2;
            match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);
                    
                    if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, entities).await {
                        println!("{}", answer);
                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, entities) {
                        println!("Prolog查询: {}", prolog_query);
                        
                        let results = prolog_engine.query(&prolog_query);
                        let response = response_formatter.format_prolog_results(&results, query_type);
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                }
            }
        }
    } else {
        // 没有API配置
        let max_hops = 2;
        match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
            Ok(facts) => {
                println!("加载了 {} 个相关事实", facts.len());
                prolog_engine.load_relevant_facts(facts);
                
                if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, entities).await {
                    println!("{}", answer);
                } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, entities) {
                    println!("Prolog查询: {}", prolog_query);
                    
                    let results = prolog_engine.query(&prolog_query);
                    let response = response_formatter.format_prolog_results(&results, query_type);
                    println!("{}", response);
                } else {
                    println!("无法生成有效的Prolog查询");
                }
            }
            Err(e) => {
                println!("{}: {}", config.messages.process_error, e);
                
                // 本地查询失败时，尝试外部知识API
                println!("本地查询失败，尝试外部知识源...");
                
                if let Some(direct_answer) = get_direct_answer(input) {
                    println!("根据常识知识库：{}", direct_answer);
                } else {
                    match external_knowledge_service.answer_question(input).await {
                        Ok(Some(answer)) => {
                            println!("根据外部知识：{}", answer);
                        }
                        Ok(None) => {
                            println!("很抱歉，我无法找到相关信息。");
                        }
                        Err(e) => {
                            println!("查询外部知识时出错：{}。", e);
                        }
                    }
                }
            }
        }
    }
}

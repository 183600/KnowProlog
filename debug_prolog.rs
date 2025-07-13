use know_prolog::{load_config, OxigraphManager, prolog_engine::PrologEngine};

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    let oxigraph_manager = OxigraphManager::new(&config.system.oxigraph_store_path).unwrap();
    
    let mut prolog_engine = PrologEngine::new();
    
    // 加载事实
    match oxigraph_manager.load_all_facts().await {
        Ok(facts) => {
            println!("加载的事实：");
            for (i, fact) in facts.iter().enumerate() {
                println!("  {}: {:?}", i, fact);
            }
            
            println!("\n正在加载事实到Prolog引擎...");
            prolog_engine.load_facts(facts);
            println!("事实加载完成。");
            
            // 测试简单查询
            println!("\n测试查询:");
            
            // 测试基础事实查询
            let test_queries = vec![
                "oxigraph_entity('水', Type, Props).",
                "entity('水', Type, Props).",
                "has_property('水', formula, Formula).",
            ];
            
            for query in test_queries {
                println!("\n查询: {}", query);
                let results = prolog_engine.query(query);
                println!("结果: {:?}", results);
            }
        }
        Err(e) => {
            println!("加载事实失败: {}", e);
        }
    }
}
use know_prolog::{load_config, OxigraphManager};

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    let oxigraph_manager = OxigraphManager::new(&config.system.oxigraph_store_path).unwrap();
    
    match oxigraph_manager.load_all_facts().await {
        Ok(facts) => {
            println!("从数据库加载的事实：");
            for (i, fact) in facts.iter().enumerate() {
                println!("  {}: {:?}", i, fact);
            }
        }
        Err(e) => {
            println!("加载事实失败: {}", e);
        }
    }
}
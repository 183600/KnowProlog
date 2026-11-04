use know_prolog::nlp::nlp::NLP;
use know_prolog::entity_relation_extractor::KnowledgeExtractor;

#[tokio::main]
async fn main() {
    // Initialize components
    let nlp = NLP::new();
    let config = know_prolog::config::Config::load().unwrap();
    let extractor = KnowledgeExtractor::new(config.clone());
    
    // Test the specific issue: "李四被张三批评了"
    let text = "李四被张三批评了";
    
    println!("Testing text: '{}'", text);
    
    // Test entity extraction
    let tagged_words = nlp.preprocess(text, &config.nlp);
    println!("Tagged words: {:?}", tagged_words);
    
    let entities = extractor.extract(text, &tagged_words);
    println!("Extracted entities: {:?}", entities);
    
    // Check if both "李四" and "张三" are extracted
    let names_found = entities.iter().map(|e| &e.entity).collect::<Vec<_>>();
    let has_li_si = names_found.contains(&"李四");
    let has_zhang_san = names_found.contains(&"张三");
    
    println!("Found '李四': {}", has_li_si);
    println!("Found '张三': {}", has_zhang_san);
    
    if has_li_si && has_zhang_san {
        println!("SUCCESS: Both names were extracted correctly!");
    } else {
        println!("ISSUE: Names not extracted properly");
    }
}

use std::sync::Arc;
use know_prolog::{load_config, nlp::preprocess};

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    let text = "法国的首都是巴黎";
    let tagged_words = preprocess(text, &config.nlp);
    
    println!("输入文本: {}", text);
    println!("词性标注结果:");
    for word in &tagged_words {
        println!("  {} -> {}", word.text, word.pos);
    }
}

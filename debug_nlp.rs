use know_prolog::{load_config, nlp::preprocess};

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    
    let test_texts = vec![
        "法国的首都是巴黎",
        "珠穆朗玛峰的海拔高度是8848米",
        "水的化学分子式是H2O",
        "哈姆雷特的作者是莎士比亚",
    ];
    
    for text in test_texts {
        println!("文本: {}", text);
        let tagged_words = preprocess(text, &config.nlp);
        for (i, word) in tagged_words.iter().enumerate() {
            println!("  {}: {} ({})", i, word.text, word.pos);
        }
        println!();
    }
}
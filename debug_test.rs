#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_tokenization() {
        let config = load_config().await.unwrap();
        let text = "法国的首都是巴黎";
        let tagged_words = nlp::preprocess(text, &config.nlp);
        
        println!("输入文本: {}", text);
        println!("词性标注结果:");
        for word in &tagged_words {
            println!("  {} -> {}", word.text, word.pos);
        }
        
        // 测试提取
        let extractor = KnowledgeExtractor::new(std::sync::Arc::new(config));
        let items = extractor.extract(text, &tagged_words);
        println!("提取的知识项: {:?}", items);
        
        assert!(false, "Debug测试");
    }
}
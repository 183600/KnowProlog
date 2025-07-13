#!/bin/bash

echo "=== Debug: 测试词性标注 ==="

# 创建临时测试程序
cat > test_tokenize.rs << 'EOF'
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
EOF

# 编译并运行
rustc --edition 2021 -L target/release/deps test_tokenize.rs -o test_tokenize --extern know_prolog=target/release/libknow_prolog.rlib --extern tokio=target/release/deps/libtokio-*.rlib || echo "编译失败，尝试简单的方法"

if [ -f test_tokenize ]; then
    ./test_tokenize
    rm test_tokenize test_tokenize.rs
else
    echo "尝试通过cargo运行测试"
fi
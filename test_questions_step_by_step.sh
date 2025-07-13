#!/bin/bash

# Test individual questions with proper setup
echo "Testing individual questions..."

# Create a test file with knowledge loading and one question
create_test_file() {
    local question="$1"
    cat > temp_test.txt << EOF
$(head -n 220 knowledge_base.txt)
$question
quit
EOF
}

# Test questions one by one
questions=(
    "法国的首都是哪里？"
    "珠穆朗玛峰的海拔高度是多少？"
    "水的化学分子式是什么？"
    "哈姆雷特的作者是谁？"
    "2022年世界杯的冠军是哪支队伍？"
    "人体有多少块骨头？"
    "Python语言是哪一年诞生的？"
    "日本的现任首都是东京，那么京都是在哪一年之前作为首都的？"
    "太阳系中最大的行星是哪一颗？"
    "什么是光合作用？"
)

for i in "${!questions[@]}"; do
    echo ""
    echo "Question $((i+1)): ${questions[$i]}"
    echo "Testing..."
    
    create_test_file "${questions[$i]}"
    timeout 30s ./target/release/know_prolog < temp_test.txt > test_result.txt 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Answer:"
        tail -n 5 test_result.txt | head -n 1
    else
        echo "Test timed out or failed"
    fi
    echo "---"
done

# Clean up
rm -f temp_test.txt test_result.txt
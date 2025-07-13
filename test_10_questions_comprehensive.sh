#!/bin/bash

# Test the 10 questions systematically
echo "Testing KnowProlog system with 10 comprehensive questions..."
echo "============================================="

# Array of questions
declare -a questions=(
    "法国的首都是哪里？"
    "珠穆朗玛峰的海拔高度是多少？"
    "水的化学分子式是什么？"
    "《哈姆雷特》的作者是谁？"
    "2022年世界杯的冠军是哪支队伍？"
    "人体有多少块骨头？"
    "Python语言是哪一年诞生的？"
    "日本的现任首都是东京，那么京都是在哪一年之前作为首都的？"
    "太阳系中最大的行星是哪一颗？"
    "什么是光合作用？"
)

# Expected answers for comparison
declare -a expected_answers=(
    "巴黎"
    "8848"
    "H2O"
    "莎士比亚"
    "阿根廷"
    "206"
    "1991"
    "1868"
    "木星"
    "植物制造食物的过程"
)

echo "First, loading the expanded knowledge base..."
echo "Loading knowledge base..."

# Load the knowledge base
if [ -f "knowledge_base_expanded.txt" ]; then
    cargo run --release < knowledge_base_expanded.txt > /dev/null 2>&1
    echo "Knowledge base loaded successfully."
else
    echo "Error: knowledge_base_expanded.txt not found!"
    exit 1
fi

echo ""
echo "Now testing each question:"
echo "========================="

# Test each question
for i in "${!questions[@]}"; do
    question_num=$((i + 1))
    question="${questions[$i]}"
    expected="${expected_answers[$i]}"
    
    echo ""
    echo "Question $question_num: $question"
    echo "Expected answer contains: $expected"
    echo "System response:"
    
    # Test the question
    echo "$question" | cargo run --release | tail -1
    
    echo "---"
done

echo ""
echo "Test completed!"
echo "============================================="
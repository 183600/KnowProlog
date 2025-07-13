#!/bin/bash

# Test all 10 questions individually
echo "Testing all 10 questions..."
echo "=========================="

# First load the knowledge base
echo "Loading knowledge base..."
./target/release/know_prolog < knowledge_base.txt > /dev/null 2>&1

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
    echo "Answer:"
    echo "${questions[$i]}" | ./target/release/know_prolog 2>/dev/null | grep -v "系统已就绪" | grep -v "输入查询" | grep -v "输入'quit'退出系统" | tail -n 1
    echo "---"
done

echo ""
echo "All questions tested!"
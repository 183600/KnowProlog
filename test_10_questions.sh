#!/bin/bash

echo "开始测试10个基础问题..."

# 问题列表
questions=(
    "法国的首都是哪里"
    "珠穆朗玛峰的海拔高度是多少"
    "水的化学分子式是什么"
    "《哈姆雷特》的作者是谁"
    "2022年世界杯的冠军是哪支队伍"
    "人体有多少块骨头"
    "Python语言是哪一年诞生的"
    "京都是在哪一年之前作为日本首都的"
    "太阳系中最大的行星是哪一颗"
    "什么是光合作用"
)

# 编译项目
echo "编译项目..."
cargo build --release

# 测试每个问题
for i in "${!questions[@]}"; do
    question_num=$((i + 1))
    question="${questions[$i]}"
    
    echo ""
    echo "=========================================="
    echo "问题 $question_num: $question"
    echo "=========================================="
    
    # 将问题输入系统并获取答案
    echo "$question" | timeout 10s cargo run --release 2>/dev/null | grep -A 5 "> " | tail -n +2
    
    echo ""
    sleep 1
done

echo ""
echo "测试完成！"
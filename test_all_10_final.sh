#!/bin/bash

echo "完整测试10个基础问题..."
echo ""

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

for i in "${!questions[@]}"; do
    question_num=$((i + 1))
    question="${questions[$i]}"
    
    echo "==== 问题 $question_num: $question ===="
    echo "$question" | timeout 5s cargo run --release 2>/dev/null | grep -v "知识图谱系统启动中" | grep -v "系统已就绪" | grep -v "输入查询" | grep -v "输入'录入'" | grep -v "输入'quit'" | grep -v "Loading Prolog facts" | grep -v "oxigraph_entity" | grep -v "执行Prolog查询" | grep -v "Prolog输出" | grep -v ">" | tail -1
    echo ""
done

echo "测试完成！"
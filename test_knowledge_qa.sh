#!/bin/bash

echo "测试知识库问答系统"
echo "====================="

questions=(
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

expected_answers=(
    "巴黎"
    "8848米"
    "H2O"
    "莎士比亚"
    "阿根廷"
    "206块"
    "1991年"
    "1868年"
    "木星"
    "植物制造食物的过程"
)

echo "开始测试..."
echo

for i in "${!questions[@]}"; do
    echo "问题 $((i+1)): ${questions[$i]}"
    echo "期望答案: ${expected_answers[$i]}"
    
    # 使用timeout来避免程序卡住
    result=$(timeout 10s bash -c "echo '${questions[$i]}' | cargo run --release --quiet" 2>/dev/null | grep -v "系统" | grep -v "输入" | grep -v ">" | grep -v "查询" | grep -v "Loading" | grep -v "oxigraph" | grep -v "Prolog" | tail -n 1)
    
    if [[ -n "$result" ]]; then
        echo "实际答案: $result"
        if [[ "$result" == *"${expected_answers[$i]}"* ]]; then
            echo "✓ 测试通过"
        else
            echo "✗ 测试失败"
        fi
    else
        echo "✗ 无法获取答案"
    fi
    echo "---"
done

echo "测试完成"
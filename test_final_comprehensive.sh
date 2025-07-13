#!/bin/bash

# Comprehensive test of all 10 questions with results

echo "=== KnowProlog 系统测试报告 ==="
echo "测试时间: $(date)"
echo "==============================="
echo

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

expected_keywords=(
    "巴黎"
    "8848"
    "H2O"
    "莎士比亚"
    "阿根廷"
    "206"
    "1991"
    "1868"
    "木星"
    "光合作用"
)

success_count=0
total_questions=${#questions[@]}

for i in "${!questions[@]}"; do
    echo "问题 $((i+1)): ${questions[i]}"
    echo -n "结果: "
    
    # Run the query and capture the output
    result=$(printf "${questions[i]}\nquit\n" | timeout 10s ./target/release/know_prolog 2>/dev/null)
    
    # Check for query type detection
    if echo "$result" | grep -q "查询类型:"; then
        query_type=$(echo "$result" | grep "查询类型:" | head -1)
        echo "✅ 识别查询 - $query_type"
        
        # Check if we get a meaningful answer containing expected keyword
        if echo "$result" | grep -q "${expected_keywords[i]}"; then
            echo "   ✅ 正确答案包含: ${expected_keywords[i]}"
            ((success_count++))
        else
            echo "   ⚠️  未找到预期关键词: ${expected_keywords[i]}"
        fi
    elif echo "$result" | grep -q "抱歉，我不理解您的查询"; then
        echo "❌ 未识别查询"
    else
        echo "⚠️  异常结果"
    fi
    
    echo "---"
done

echo
echo "=== 测试总结 ==="
echo "成功回答: $success_count / $total_questions"
echo "成功率: $(( success_count * 100 / total_questions ))%"

if [ $success_count -eq $total_questions ]; then
    echo "🎉 所有问题都正确回答！"
elif [ $success_count -gt $((total_questions / 2)) ]; then
    echo "👍 大部分问题正确回答"
else
    echo "⚠️  需要改进模式匹配"
fi
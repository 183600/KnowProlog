#!/bin/bash

# 扩展测试：测试系统回答各种问题的能力

echo "=== KnowProlog 扩展能力测试 ==="
echo "测试时间: $(date)"
echo "================================"
echo

additional_questions=(
    "中国的首都是哪里？"
    "日本的首都是哪里？"
    "哈姆雷特的作者是谁？"
    "Python诞生于哪一年？"
    "世界杯的冠军是谁？"
    "什么是水的分子式？"
    "光合作用是什么？"
    "珠穆朗玛峰有多高？"
    "人体骨头有多少块？"
    "最大的行星是什么？"
    "太阳系最大行星？"
    "世界杯冠军？"
    "水分子式？"
    "Python年份？"
    "法国首都？"
    "哈姆雷特作者？"
    "人体多少骨头？"
    "海拔高度？"
    "木星是什么？"
    "2022世界杯？"
)

success_count=0
total_questions=${#additional_questions[@]}

for i in "${!additional_questions[@]}"; do
    echo "问题 $((i+1)): ${additional_questions[i]}"
    echo -n "结果: "
    
    # Run the query and capture the output
    result=$(printf "${additional_questions[i]}\nquit\n" | timeout 10s ./target/release/know_prolog 2>/dev/null)
    
    # Check for query type detection
    if echo "$result" | grep -q "查询类型:"; then
        query_type=$(echo "$result" | grep "查询类型:" | head -1)
        echo "✅ 识别查询 - $query_type"
        ((success_count++))
    elif echo "$result" | grep -q "抱歉，我不理解您的查询"; then
        echo "❌ 未识别查询"
    else
        echo "⚠️  异常结果"
    fi
    
    echo "---"
done

echo
echo "=== 扩展测试总结 ==="
echo "成功识别: $success_count / $total_questions"
echo "识别率: $(( success_count * 100 / total_questions ))%"

if [ $success_count -gt $((total_questions * 8 / 10)) ]; then
    echo "🎉 系统具有优秀的问题识别能力！"
elif [ $success_count -gt $((total_questions / 2)) ]; then
    echo "👍 系统具有良好的问题识别能力"
else
    echo "⚠️  系统需要进一步改进"
fi
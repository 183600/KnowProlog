#!/bin/bash

echo "=== KnowProlog 10问题全面测试 ==="
echo "测试时间: $(date)"
echo "=========================================="

# 创建测试输入文件
cat > test_input_comprehensive.txt << 'EOF'
法国的首都是哪里？
珠穆朗玛峰的海拔高度是多少？
水的化学分子式是什么？
《哈姆雷特》的作者是谁？
2022年世界杯的冠军是哪支队伍？
人体有多少块骨头？
Python语言是哪一年诞生的？
日本的现任首都是东京，那么京都是在哪一年之前作为首都的？
太阳系中最大的行星是哪一颗？
什么是光合作用？
quit
EOF

echo "开始测试..."
echo "输入文件内容："
cat test_input_comprehensive.txt
echo "=========================================="

# 运行测试
echo "执行测试..."
timeout 60s cargo run --release < test_input_comprehensive.txt > test_output_comprehensive.txt 2>&1

echo "=========================================="
echo "测试结果："
cat test_output_comprehensive.txt

echo "=========================================="
echo "分析结果："

# 分析每个问题的回答
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

echo "问题回答分析："
for i in "${!questions[@]}"; do
    echo "问题 $((i+1)): ${questions[$i]}"
    echo "期望答案: ${expected_answers[$i]}"
    
    # 检查输出中是否包含期望答案的关键词
    if grep -q "${expected_answers[$i]}" test_output_comprehensive.txt; then
        echo "✓ 答案正确"
    else
        echo "✗ 答案可能不正确"
    fi
    echo "---"
done

echo "测试完成！"
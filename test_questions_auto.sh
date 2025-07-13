#!/bin/bash

# 自动测试脚本
echo "开始测试知识图谱系统的回答能力..."

# 测试输入文件路径
INPUT_FILE="knowledge_base.txt"
OUTPUT_FILE="test_output_auto.txt"

# 创建测试输入
cat > test_input_auto.txt << 'EOF'
法国的首都是哪里？
珠穆朗玛峰的海拔高度是多少？
水的化学分子式是什么？
哈姆雷特的作者是谁？
2022年世界杯的冠军是哪支队伍？
人体有多少块骨头？
Python语言是哪一年诞生的？
太阳系中最大的行星是哪一颗？
什么是光合作用？
京都在哪一年之前是日本首都？
解释什么是黑洞？
达尔文进化论的核心观点是什么？
什么是摩尔定律？
什么是区块链技术？
5G技术相比4G有哪些提升？
机器学习和深度学习有什么区别？
什么是量子计算？
牛顿第三定律是什么？
气候变化的主要原因是什么？
文艺复兴对欧洲产生了哪些影响？
quit
EOF

echo "测试输入文件已创建"
echo "开始运行测试..."

# 运行测试
timeout 60s cargo run --release < test_input_auto.txt > "$OUTPUT_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "测试完成！"
    echo "输出已保存到 $OUTPUT_FILE"
    echo ""
    echo "=== 测试结果摘要 ==="
    grep -E "(的首都|海拔高度|分子式|作者|冠军|骨头|诞生|行星|光合作用|黑洞|进化论|摩尔定律|区块链|5G|机器学习|量子计算|牛顿|气候变化|文艺复兴)" "$OUTPUT_FILE" || echo "未找到明确的回答"
else
    echo "测试超时或出现错误"
    echo "查看输出文件: $OUTPUT_FILE"
fi

echo ""
echo "完整输出内容："
cat "$OUTPUT_FILE"
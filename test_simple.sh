#!/bin/bash

# 简单测试基本功能
echo "开始测试基本功能..."

# 创建简单测试输入
cat > test_simple.txt << 'EOF'
录入：小明18岁
录入：张三是医生
小明多大？
张三的职业？
quit
EOF

echo "开始运行简单测试..."

# 运行测试
timeout 15s cargo run --release < test_simple.txt > test_simple_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "测试完成！"
    echo "输出已保存到 test_simple_output.txt"
else
    echo "测试超时或出现错误"
fi

echo ""
echo "完整输出内容："
cat test_simple_output.txt
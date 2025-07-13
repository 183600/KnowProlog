#!/bin/bash

# 测试导入和查询
echo "开始测试知识导入和查询..."

# 创建测试输入，先导入一些基础知识，然后查询
cat > test_import_query.txt << 'EOF'
录入：法国的首都是巴黎
录入：珠穆朗玛峰的海拔高度是8848米
录入：水的化学分子式是H2O
录入：哈姆雷特的作者是莎士比亚
录入：Python语言是1991年诞生的
录入：木星是太阳系中最大的行星
法国的首都是哪里？
珠穆朗玛峰的海拔高度是多少？
水的化学分子式是什么？
哈姆雷特的作者是谁？
Python语言是哪一年诞生的？
太阳系中最大的行星是哪一颗？
quit
EOF

echo "开始运行导入和查询测试..."

# 运行测试
timeout 30s cargo run --release < test_import_query.txt > test_import_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "测试完成！"
    echo "输出已保存到 test_import_output.txt"
else
    echo "测试超时或出现错误"
fi

echo ""
echo "完整输出内容："
cat test_import_output.txt
#!/bin/bash

echo "=== 完整的10题测试 ==="

# 首先导入所有知识
echo "导入知识库..."
timeout 30 cargo run --release < knowledge_base.txt > /dev/null 2>&1

echo ""
echo "开始测试10个问题的查询能力..."

# 测试1: 法国的首都是哪里？
echo "1. 测试：法国的首都是哪里？"
(
echo "法国的首都是哪里？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试2: 珠穆朗玛峰的海拔高度是多少？
echo "2. 测试：珠穆朗玛峰的海拔高度是多少？"
(
echo "珠穆朗玛峰的海拔高度是多少？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试3: 水的化学分子式是什么？
echo "3. 测试：水的化学分子式是什么？"
(
echo "水的化学分子式是什么？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试4: 哈姆雷特的作者是谁？
echo "4. 测试：哈姆雷特的作者是谁？"
(
echo "哈姆雷特的作者是谁？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试5: 2022年世界杯的冠军是哪支队伍？
echo "5. 测试：2022年世界杯的冠军是哪支队伍？"
(
echo "2022年世界杯的冠军是哪支队伍？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试6: 人体有多少块骨头？
echo "6. 测试：人体有多少块骨头？"
(
echo "人体有多少块骨头？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试7: Python语言是哪一年诞生的？
echo "7. 测试：Python语言是哪一年诞生的？"
(
echo "Python语言是哪一年诞生的？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试8: 京都是在哪一年之前作为首都的？
echo "8. 测试：京都是在哪一年之前作为首都的？"
(
echo "京都是在哪一年之前作为首都的？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试9: 太阳系中最大的行星是哪一颗？
echo "9. 测试：太阳系中最大的行星是哪一颗？"
(
echo "太阳系中最大的行星是哪一颗？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

# 测试10: 什么是光合作用？
echo "10. 测试：什么是光合作用？"
(
echo "什么是光合作用？"
sleep 1
echo "quit"
) | timeout 10 cargo run --release 2>&1 | grep -E "(查询类型|Prolog查询|抱歉)"
echo ""

echo "=== 测试完成 ==="
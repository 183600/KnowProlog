#\!/bin/bash

# 测试10个基础知识问题
echo "开始测试KnowProlog基础知识问答..."

echo "=== 问题 1: 法国的首都是哪里？ ==="
echo "法国的首都是哪里？"  < /dev/null |  timeout 10 cargo run --release 2>/dev/null
echo ""

echo "=== 问题 2: 珠穆朗玛峰的海拔高度是多少？ ==="
echo "珠穆朗玛峰的海拔高度是多少？" | timeout 10 cargo run --release 2>/dev/null
echo ""

echo "=== 问题 3: 水的化学分子式是什么？ ==="
echo "水的化学分子式是什么？" | timeout 10 cargo run --release 2>/dev/null
echo ""

echo "测试完成。"

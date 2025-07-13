#!/bin/bash

# 测试模式匹配的脚本

echo "=== 测试1：简单词语分词 ==="
echo "法国的首都是哪里" | timeout 5 cargo run --release 2>&1 | grep -E "(正在|抱歉|分词|词性)"

echo -e "\n=== 测试2：录入一条知识 ==="
echo "录入：法国的首都是巴黎" | timeout 5 cargo run --release 2>&1 | tail -10

echo -e "\n=== 测试3：查询 ==="
echo "法国的首都是哪里" | timeout 5 cargo run --release 2>&1 | tail -10
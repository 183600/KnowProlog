#!/bin/bash

echo "=== 测试完整流程 ==="

# 测试导入
echo "录入：法国的首都是巴黎" | timeout 10 cargo run --release
echo ""

# 测试查询 - 需要在同一个会话中
(
echo "录入：法国的首都是巴黎"
sleep 1
echo "法国的首都是哪里"
sleep 1 
echo "quit"
) | timeout 20 cargo run --release
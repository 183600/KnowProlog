#!/bin/bash

# 逐步导入和测试脚本
echo "=== 第一步：导入单条知识 ==="

echo "录入：法国的首都是巴黎" | timeout 10 cargo run --release
echo ""

echo "=== 第二步：测试第一个问题 ==="
echo "法国的首都是哪里？" | timeout 10 cargo run --release

echo "=== 完成 ==="
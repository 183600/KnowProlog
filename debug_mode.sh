#!/bin/bash

echo "=== Debug: 测试录入识别 ==="

echo "录入：测试" | cargo run --release &
PID=$!
sleep 2
kill $PID 2>/dev/null

echo -e "\n=== Debug: 测试查询识别 ==="

echo "查询测试" | cargo run --release &
PID=$!
sleep 2
kill $PID 2>/dev/null
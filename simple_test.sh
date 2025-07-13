#!/bin/bash

echo "简化测试..."

echo "问题1: 法国的首都是哪里"
echo "法国的首都是哪里" | cargo run --release | tail -n 5

echo ""
echo "问题2: 水的化学分子式是什么"
echo "水的化学分子式是什么" | cargo run --release | tail -n 5

echo ""
echo "问题3: 什么是光合作用"
echo "什么是光合作用" | cargo run --release | tail -n 5
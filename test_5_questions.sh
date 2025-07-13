#!/bin/bash

echo "测试单个问题..."
echo ""

echo "==== 测试问题1: 法国的首都是哪里 ===="
echo "法国的首都是哪里" | timeout 3s cargo run --release 2>/dev/null | grep -E "首都是|查询类型|Loading|执行"

echo ""
echo "==== 测试问题2: 珠穆朗玛峰的海拔高度是多少 ===="
echo "珠穆朗玛峰的海拔高度是多少" | timeout 3s cargo run --release 2>/dev/null | grep -E "海拔|查询类型|Loading|执行"

echo ""
echo "==== 测试问题3: 水的化学分子式是什么 ===="
echo "水的化学分子式是什么" | timeout 3s cargo run --release 2>/dev/null | grep -E "分子式|查询类型|Loading|执行"

echo ""
echo "==== 测试问题4: 哈姆雷特的作者是谁 ===="
echo "《哈姆雷特》的作者是谁" | timeout 3s cargo run --release 2>/dev/null | grep -E "作者是|查询类型|Loading|执行"

echo ""
echo "==== 测试问题5: 2022年世界杯的冠军是哪支队伍 ===="
echo "2022年世界杯的冠军是哪支队伍" | timeout 3s cargo run --release 2>/dev/null | grep -E "冠军是|查询类型|Loading|执行"

echo ""
echo "测试完成!"
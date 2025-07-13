#!/bin/bash

# KnowProlog 扩展演示脚本

echo "=== KnowProlog 扩展功能演示 ==="
echo

echo "1. 构建系统..."
cargo build --release

echo "2. 创建演示数据目录..."
mkdir -p oxigraph_data

echo "3. 系统现在支持的查询类型："
echo "   - 人物信息查询: '小明多大?'"
echo "   - 关系查询: '张三和李四什么关系?'"
echo "   - 地理查询: '法国的首都' (新增)"
echo "   - 人口查询: '中国有多少人' (新增)"
echo "   - 计数查询: '有多少个人?'"
echo

echo "4. 演示用法："
echo "   录入知识: '录入：法国的首都是巴黎'"
echo "   查询知识: '法国的首都'"
echo "   退出系统: 'quit'"
echo

echo "5. 启动系统..."
echo "   (系统将在下方启动，您可以开始输入)"
echo

# 启动系统
cargo run --release
#!/bin/bash

echo "测试KnowProlog与Ownthink API的集成功能"
echo "======================================"
echo ""

# 构建项目
echo "正在构建项目..."
cargo build

if [ $? -ne 0 ]; then
    echo "构建失败！"
    exit 1
fi

echo "构建成功！"
echo ""

# 测试直接的实体知识查询
echo "测试1: 查询刘德华的信息"
echo "命令: echo '刘德华的信息' | timeout 10 cargo run"
echo "--------------------------------"
echo "刘德华的信息" | timeout 10 cargo run
echo ""

echo "测试2: 查询苹果的信息（消歧）"
echo "命令: echo '苹果' | timeout 10 cargo run"
echo "--------------------------------" 
echo "苹果" | timeout 10 cargo run
echo ""

echo "测试3: 查询马云的信息"
echo "命令: echo '告诉我关于马云' | timeout 10 cargo run"
echo "--------------------------------"
echo "告诉我关于马云" | timeout 10 cargo run
echo ""

echo "测试完成！"
echo ""
echo "API功能说明："
echo "1. 直接实体知识查询：输入 '实体名的信息' 或 '告诉我关于实体名' 可获取完整知识"
echo "2. 自动消歧处理：对于有歧义的实体，系统会自动解析并选择最匹配的实体"
echo "3. 智能知识展示：显示实体的描述、标签和详细属性信息"
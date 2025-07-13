#!/bin/bash

# 导入关键知识数据

echo "导入核心知识数据..."

./target/release/know_prolog <<EOF
录入：人体有206块骨头
录入：木星是太阳系中最大的行星
录入：京都在1868年之前作为首都
quit
EOF

echo "数据导入完成！"
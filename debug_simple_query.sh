#!/bin/bash

echo "测试查询解析..."

# 简单测试一个问题
echo "法国的首都是哪里" | timeout 5s cargo run --release 2>&1 | head -20
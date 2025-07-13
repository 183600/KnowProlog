#!/usr/bin/env python3
# Simple test to see what tokens jieba would generate

import jieba

test_sentences = [
    "法国的首都是哪里？",
    "珠穆朗玛峰的海拔高度是多少？", 
    "水的化学分子式是什么？",
    "《哈姆雷特》的作者是谁？",
    "2022年世界杯的冠军是哪支队伍？"
]

for sentence in test_sentences:
    tokens = list(jieba.cut(sentence, cut_all=False))
    print(f"句子: {sentence}")
    print(f"分词: {tokens}")
    print()
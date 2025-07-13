#\!/bin/bash

echo "=== KnowProlog基础知识问答测试报告 ==="
echo ""

total=10
passed=0
failed=0

questions=(
    "法国的首都是哪里？"
    "珠穆朗玛峰的海拔高度是多少？"
    "水的化学分子式是什么？"
    "《哈姆雷特》的作者是谁？"
    "2022年世界杯的冠军是哪支队伍？"
    "人体有多少块骨头？"
    "Python语言是哪一年诞生的？"
    "日本的现任首都是东京，那么京都是在哪一年之前作为首都的？"
    "太阳系中最大的行星是哪一颗？"
    "什么是光合作用？"
)

for i in {0..9}; do
    echo "问题 $((i+1)): ${questions[i]}"
    result=$(echo "${questions[i]}"  < /dev/null |  timeout 15 cargo run --release 2>/dev/null | grep -v "Compiling\|Finished\|Running\|知识图谱系统启动中\|系统已就绪\|请输入您的查询\|输入'导入'\|输入'quit'\|Debug:\|JSON解析失败\|查询实体\|获取属性值\|查询类型\|正在解析\|加载了\|Loading\|Prolog查询\|执行Prolog查询\|清除动态事实\|系统已退出" | tail -1)
    
    if [[ "$result" == *"根据当前知识库，无法回答您的问题"* ]] || [[ "$result" == *"抱歉，我不理解您的查询"* ]] || [[ -z "$result" ]]; then
        echo "❌ 无法回答"
        ((failed++))
    else
        echo "✅ 系统回答: $result"
        ((passed++))
    fi
    echo ""
done

echo "=== 测试结果统计 ==="
echo "总问题数: $total"
echo "成功回答: $passed"
echo "无法回答: $failed"
echo "成功率: $((passed * 100 / total))%"

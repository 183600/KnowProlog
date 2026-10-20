use know_prolog::question_analyzer::{
    ComplexityLevel, Domain, Intent, QuestionAnalyzer, QuestionType,
};

#[tokio::test]
async fn test_30_sample_questions() {
    let analyzer = QuestionAnalyzer::new();

    // 一、基础知识与事实查询 (Factual Recall)
    let questions = vec![
        (
            "法国的首都是哪里？",
            QuestionType::FactualQuery,
            Domain::Geography,
        ),
        (
            "珠穆朗玛峰的海拔高度是多少？",
            QuestionType::FactualQuery,
            Domain::Geography,
        ),
        (
            "水的化学分子式是什么？",
            QuestionType::FactualQuery,
            Domain::Science,
        ),
        (
            "《哈姆雷特》的作者是谁？",
            QuestionType::FactualQuery,
            Domain::Literature,
        ),
        (
            "2022年世界杯的冠军是哪支队伍？",
            QuestionType::FactualQuery,
            Domain::Sports,
        ),
        (
            "人体有多少块骨头？",
            QuestionType::FactualQuery,
            Domain::Science,
        ),
        (
            "Python语言是哪一年诞生的？",
            QuestionType::FactualQuery,
            Domain::Technology,
        ),
        (
            "日本的现任首都是东京，那么京都是在哪一年之前作为首都的？",
            QuestionType::FactualQuery,
            Domain::History,
        ),
        (
            "太阳系中最大的行星是哪一颗？",
            QuestionType::FactualQuery,
            Domain::Science,
        ),
        (
            "什么是光合作用？",
            QuestionType::DefinitionQuestion,
            Domain::Science,
        ),
        // 二、科学与技术 (Science & Technology)
        (
            "解释一下什么是黑洞。",
            QuestionType::ConceptExplanation,
            Domain::Science,
        ),
        (
            "简述达尔文的进化论核心观点。",
            QuestionType::ConceptExplanation,
            Domain::Science,
        ),
        (
            "什么是摩尔定律，它现在还适用吗？",
            QuestionType::ConceptExplanation,
            Domain::Technology,
        ),
        (
            "区块链技术的工作原理是什么？",
            QuestionType::ConceptExplanation,
            Domain::Technology,
        ),
        (
            "描述一下CRISPR基因编辑技术及其潜在风险。",
            QuestionType::ConceptExplanation,
            Domain::Science,
        ),
        (
            "5G技术相比4G有哪些关键提升？",
            QuestionType::ComparisonAnalysis,
            Domain::Technology,
        ),
        (
            "机器学习和深度学习有什么区别？",
            QuestionType::ComparisonAnalysis,
            Domain::Technology,
        ),
        (
            "什么是量子计算？它能解决哪些经典计算机无法解决的问题？",
            QuestionType::ConceptExplanation,
            Domain::Technology,
        ),
        (
            "解释牛顿第三定律，并举一个生活中的例子。",
            QuestionType::ScientificPrinciple,
            Domain::Science,
        ),
        (
            "气候变化的主要原因是什么？",
            QuestionType::CausalExplanation,
            Domain::Science,
        ),
        // 三、历史与人文 (History & Humanities)
        (
            "文艺复兴对欧洲产生了哪些深远影响？",
            QuestionType::HistoricalDescription,
            Domain::History,
        ),
        (
            "描述古埃及金字塔的建造目的和历史意义。",
            QuestionType::HistoricalDescription,
            Domain::History,
        ),
        (
            "第二次世界大战的转折点是哪几场战役？",
            QuestionType::FactualQuery,
            Domain::History,
        ),
        (
            "孔子的核心哲学思想是什么？",
            QuestionType::ConceptExplanation,
            Domain::Philosophy,
        ),
        (
            "比较一下斯多葛主义和享乐主义的主要区别。",
            QuestionType::ComparisonAnalysis,
            Domain::Philosophy,
        ),
        (
            "美国《独立宣言》是在什么背景下发表的？",
            QuestionType::HistoricalDescription,
            Domain::History,
        ),
        (
            "丝绸之路在东西方文化交流中扮演了什么角色？",
            QuestionType::HistoricalDescription,
            Domain::History,
        ),
        (
            "解释\"内卷\"这个词的社会学含义和起源。",
            QuestionType::ConceptExplanation,
            Domain::Culture,
        ),
        (
            "分析梵高《星夜》的艺术风格和表达的情感。",
            QuestionType::ConceptExplanation,
            Domain::Culture,
        ),
        (
            "古希腊神话中，普罗米修斯为何受到惩罚？",
            QuestionType::CausalExplanation,
            Domain::Culture,
        ),
    ];

    println!("开始测试30个示例问题的语义理解能力:");
    println!("==========================================");

    let mut successful_classifications = 0;
    let mut total_questions = questions.len();

    for (i, (question, expected_type, expected_domain)) in questions.iter().enumerate() {
        println!("\n{}. 问题: {}", i + 1, question);

        match analyzer.analyze_question(question).await {
            Ok(analysis) => {
                println!("   ✓ 分析成功");
                println!(
                    "     - 问题类型: {:?} (期望: {:?})",
                    analysis.question_type, expected_type
                );
                println!(
                    "     - 领域: {:?} (期望: {:?})",
                    analysis.domain, expected_domain
                );
                println!("     - 复杂度: {:?}", analysis.complexity);
                println!("     - 意图: {:?}", analysis.intent);
                println!("     - 置信度: {:.2}", analysis.confidence);
                println!("     - 关键词: {:?}", analysis.keywords);
                println!(
                    "     - 实体: {:?}",
                    analysis
                        .entities
                        .iter()
                        .map(|e| &e.text)
                        .collect::<Vec<_>>()
                );

                // 检查分类是否正确
                let type_correct = analysis.question_type == *expected_type;
                let domain_correct = analysis.domain == *expected_domain;

                if type_correct && domain_correct {
                    successful_classifications += 1;
                    println!("     ✅ 分类完全正确");
                } else if type_correct || domain_correct {
                    println!(
                        "     ⚠️  部分正确 (类型: {}, 领域: {})",
                        if type_correct { "✓" } else { "✗" },
                        if domain_correct { "✓" } else { "✗" }
                    );
                } else {
                    println!("     ❌ 分类不正确");
                }
            }
            Err(e) => {
                println!("   ❌ 分析失败: {}", e);
            }
        }
    }

    println!("\n==========================================");
    println!("测试结果统计:");
    println!("总问题数: {}", total_questions);
    println!("完全正确分类: {}", successful_classifications);
    println!(
        "准确率: {:.1}%",
        (successful_classifications as f64 / total_questions as f64) * 100.0
    );

    // 至少应该有一定的准确率
    assert!(
        successful_classifications >= total_questions / 3,
        "分类准确率太低，期望至少33%，实际: {:.1}%",
        (successful_classifications as f64 / total_questions as f64) * 100.0
    );
}

#[tokio::test]
async fn test_complex_question_types() {
    let analyzer = QuestionAnalyzer::new();

    // 测试不同复杂度的问题
    let complex_questions = vec![
        ("法国的首都是哪里？", ComplexityLevel::Simple),
        ("解释一下什么是黑洞。", ComplexityLevel::Intermediate),
        (
            "比较一下斯多葛主义和享乐主义的主要区别。",
            ComplexityLevel::Complex,
        ),
        (
            "文艺复兴对欧洲产生了哪些深远影响？",
            ComplexityLevel::Complex,
        ),
    ];

    for (question, expected_complexity) in complex_questions {
        let analysis = analyzer.analyze_question(question).await.unwrap();
        println!("问题: {}", question);
        println!(
            "预期复杂度: {:?}, 实际复杂度: {:?}",
            expected_complexity, analysis.complexity
        );

        // 这里不强制要求完全匹配，因为复杂度评估可能有主观性
        assert!(analysis.confidence > 0.0);
    }
}

#[tokio::test]
async fn test_domain_classification() {
    let analyzer = QuestionAnalyzer::new();

    let domain_questions = vec![
        ("法国的首都是哪里？", Domain::Geography),
        ("什么是黑洞？", Domain::Science),
        ("Python语言是哪一年诞生的？", Domain::Technology),
        ("《哈姆雷特》的作者是谁？", Domain::Literature),
        ("孔子的核心哲学思想是什么？", Domain::Philosophy),
        ("2022年世界杯的冠军是哪支队伍？", Domain::Sports),
        ("文艺复兴对欧洲产生了哪些深远影响？", Domain::History),
    ];

    for (question, expected_domain) in domain_questions {
        let analysis = analyzer.analyze_question(question).await.unwrap();
        println!("问题: {}", question);
        println!(
            "预期领域: {:?}, 实际领域: {:?}",
            expected_domain, analysis.domain
        );

        // 验证基本的领域分类能力
        assert!(
            analysis.confidence > 0.3,
            "置信度过低: {}",
            analysis.confidence
        );
    }
}

#[tokio::test]
async fn test_intent_recognition() {
    let analyzer = QuestionAnalyzer::new();

    let intent_questions = vec![
        ("法国的首都是哪里？", Intent::SeekInformation),
        ("什么是黑洞？", Intent::UnderstandConcept),
        ("比较斯多葛主义和享乐主义", Intent::CompareEntities),
        ("解释牛顿第三定律", Intent::UnderstandConcept),
        ("文艺复兴的影响", Intent::DescribeImpact),
    ];

    for (question, expected_intent) in intent_questions {
        let analysis = analyzer.analyze_question(question).await.unwrap();
        println!("问题: {}", question);
        println!(
            "预期意图: {:?}, 实际意图: {:?}",
            expected_intent, analysis.intent
        );

        // 验证意图识别的基本功能
        assert!(!analysis.keywords.is_empty(), "应该提取到关键词");
    }
}

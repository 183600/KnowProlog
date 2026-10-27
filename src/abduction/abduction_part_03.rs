#[cfg(test)]
mod tests {
    use super::*;
    use crate::prolog_engine::PrologFact;
    use crate::resolution_engine::Clause;

    fn teacher_clause() -> Clause {
        // 师祖(x,y) :- 师傅(x,z), 师傅(z,y)
        Clause::new(
            Atom::pattern("师祖", Term::var("x"), Term::var("y")),
            vec![
                Atom::pattern("师傅", Term::var("x"), Term::var("z")),
                Atom::pattern("师傅", Term::var("z"), Term::var("y")),
            ],
            0.9,
            "测试规则：师傅传递",
        )
    }

    fn facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "师傅".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "师傅".into(),
                to: "赵六".into(),
            },
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: [("出生地".to_string(), "北京".to_string())]
                    .into_iter()
                    .collect(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "人物".into(),
                properties: [("出生地".to_string(), "上海".to_string())]
                    .into_iter()
                    .collect(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: [("出生地".to_string(), "上海".to_string())]
                    .into_iter()
                    .collect(),
            },
            PrologFact::Entity {
                name: "赵六".into(),
                entity_type: "人物".into(),
                properties: [("出生地".to_string(), "广州".to_string())]
                    .into_iter()
                    .collect(),
            },
        ]
    }

    fn engine_with_rule() -> ResolutionEngine {
        let mut engine = ResolutionEngine::new(6, 12);
        engine.add_fact("师傅", "张三", "李四");
        engine.add_fact("师傅", "李四", "王五");
        engine.add_fact("师傅", "王五", "赵六");
        engine.add_clause(teacher_clause());
        engine
    }

    #[test]
    fn provable_goal_returns_proof_not_hypotheses() {
        let engine = engine_with_rule();
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&facts());
        let report = abduction.analyze(&engine, &[Atom::rel("师祖", "张三", "王五")], &[]);
        assert!(report.provable);
        assert!(!report.solutions.is_empty());
        assert!(report.hypotheses.is_empty());
    }

    #[test]
    fn missing_ground_premise_generates_hypothesis_with_reasons() {
        let engine = engine_with_rule();
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&facts());
        // 师祖(张三, 赵六) 需要 师傅(李四, 赵六)（或经由其它中间人）——图中不存在
        let report = abduction.analyze(&engine, &[Atom::rel("师祖", "张三", "赵六")], &[]);
        assert!(!report.provable);
        assert!(!report.frontier.is_empty());
        assert!(!report.hypotheses.is_empty());
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("主后向链搜索结论")),
            "溯因报告必须披露主搜索的三态结论：{:?}",
            report.notes
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("已穷尽") || note.contains("搜索被截断")),
            "溯因不能把截断伪装成无解：{:?}",
            report.notes
        );
        let best = report.best_hypothesis().expect("应有可采纳的假设");
        assert!(best.atom.is_ground());
        assert!(best.confidence > 0.0);
        assert!(!best.reasons.is_empty());
    }

    #[test]
    fn open_goal_gets_candidate_values() {
        let engine = engine_with_rule();
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&facts());
        // 开放目标：师傅(赵六, ?X)，赵六在图谱中没有师傅事实
        let report = abduction.analyze(
            &engine,
            &[Atom::pattern(
                "师傅",
                Term::constant("赵六"),
                Term::var("X"),
            )],
            &["X".to_string()],
        );
        assert!(!report.provable);
        assert!(report.hypotheses.iter().all(|h| h.atom.is_ground()));
        let values: Vec<String> = report
            .hypotheses
            .iter()
            .filter_map(|h| h.atom.terms.get(1).map(|t| t.name().to_string()))
            .collect();
        assert!(
            values
                .iter()
                .any(|value| value == "李四" || value == "王五" || value == "赵六"),
            "候选应来自同谓词已知取值，实际：{:?}",
            values
        );
    }

    #[test]
    fn repeated_observations_do_not_inflate_counts_or_change_confidence() {
        let engine = engine_with_rule();
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&facts());
        let first = abduction.analyze(&engine, &[Atom::rel("师祖", "张三", "赵六")], &[]);
        // 快照重放 / 增量重载会再次提交同一批事实：统计必须保持不变。
        abduction.observe_facts(&facts());
        abduction.observe_facts(&facts());
        let second = abduction.analyze(&engine, &[Atom::rel("师祖", "张三", "赵六")], &[]);

        let before: Vec<_> = first
            .hypotheses
            .iter()
            .map(|hypothesis| {
                (
                    hypothesis.atom.display(),
                    hypothesis.confidence,
                    hypothesis.reasons.clone(),
                )
            })
            .collect();
        let after: Vec<_> = second
            .hypotheses
            .iter()
            .map(|hypothesis| {
                (
                    hypothesis.atom.display(),
                    hypothesis.confidence,
                    hypothesis.reasons.clone(),
                )
            })
            .collect();
        assert_eq!(before, after, "重复观察不得改变候选、置信度或计数理由");
    }

    #[test]
    fn what_if_derives_new_facts_and_rolls_back() {
        let engine = engine_with_rule();
        let abduction = AbductionEngine::new(6, 5);
        let assumption = Atom::rel("师傅", "赵六", "钱七");
        let result = abduction.what_if(&engine, &[assumption], 3, 20);
        assert!(!result.new_facts.is_empty());
        let derived: Vec<String> = result.new_facts.iter().map(|atom| atom.display()).collect();
        assert!(
            derived
                .iter()
                .any(|fact| fact.contains("师祖") && fact.contains("钱七")),
            "应推出与钱七相关的师祖事实，实际：{:?}",
            derived
        );
        // 回滚：原引擎不受影响
        assert!(!engine.has_fact(&Atom::rel("师傅", "赵六", "钱七")));
    }

    #[test]
    fn functional_predicate_conflict_is_penalized() {
        let mut engine = ResolutionEngine::new(6, 5);
        engine.add_fact("首都", "中国", "北京");
        engine.add_fact("首都", "法国", "巴黎");
        engine.add_fact("首都", "日本", "东京");
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&[
            PrologFact::Relation {
                from: "中国".into(),
                rel_type: "首都".into(),
                to: "北京".into(),
            },
            PrologFact::Relation {
                from: "法国".into(),
                rel_type: "首都".into(),
                to: "巴黎".into(),
            },
            PrologFact::Relation {
                from: "日本".into(),
                rel_type: "首都".into(),
                to: "东京".into(),
            },
        ]);
        let report = abduction.analyze(&engine, &[Atom::rel("首都", "中国", "上海")], &[]);
        assert!(!report.provable);
        let best = report.best_hypothesis().expect("应有假设");
        assert!(
            best.confidence < 0.1,
            "功能谓词冲突应被重罚：{}",
            best.confidence
        );
        assert!(!best.conflicts.is_empty());
    }

    #[test]
    fn take_pending_and_adopt_tracks_session_assumptions() {
        let mut abduction = AbductionEngine::new(6, 5);
        let hypothesis = Hypothesis {
            atom: Atom::rel("师傅", "张三", "李四"),
            goal: Atom::rel("师傅", "张三", "李四"),
            kind: HypothesisKind::MissingFact,
            confidence: 0.6,
            reasons: vec!["测试".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        };
        abduction.set_pending(vec![hypothesis]);
        let taken = abduction.take_pending(1).expect("应取到假设");
        assert!(taken.is_adoptable());
        abduction.adopt(taken);
        assert_eq!(abduction.adopted().len(), 1);
        assert!(abduction.take_pending(1).is_none());
    }

    // ---- 联合解释场景 + 对话式澄清 ----

    fn hyp(predicate: &str, subject: &str, object: &str, confidence: f64) -> Hypothesis {
        Hypothesis {
            atom: Atom::rel(predicate, subject, object),
            goal: Atom::rel(predicate, subject, object),
            kind: HypothesisKind::MissingFact,
            confidence,
            reasons: vec!["测试假设".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        }
    }

    fn sparse_facts() -> Vec<PrologFact> {
        // 只有 李四→王五 一条师傅关系：证明 师祖(张三,赵六) 必须同时补两条边
        let mut facts = vec![
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "师傅".into(),
                to: "王五".into(),
            },
            // 让"师祖"成为已知谓词（解析陈述句需要），与目标无关
            PrologFact::Relation {
                from: "钱七".into(),
                rel_type: "师祖".into(),
                to: "孙八".into(),
            },
        ];
        for name in ["张三", "李四", "王五", "赵六", "钱七", "孙八"] {
            facts.push(PrologFact::Entity {
                name: name.into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            });
        }
        facts
    }

    fn sparse_engine() -> ResolutionEngine {
        let mut engine = ResolutionEngine::new(8, 20);
        engine.add_fact("师傅", "李四", "王五");
        engine.add_clause(teacher_clause());
        engine
    }

    #[test]
    fn joint_scenario_requires_two_hypotheses_and_verifies() {
        let engine = sparse_engine();
        let mut abduction = AbductionEngine::new(8, 8);
        abduction.set_max_scenario_size(2);
        abduction.observe_facts(&sparse_facts());
        let goal = Atom::rel("师祖", "张三", "赵六");
        let scenarios = abduction.scenarios(&engine, std::slice::from_ref(&goal), &[]);
        assert!(!scenarios.is_empty(), "应能枚举出假设组合场景");
        // 任一单假设都不足以证明目标（缺两条边）
        assert!(
            scenarios
                .iter()
                .all(|scenario| !(scenario.size() == 1 && scenario.verified)),
            "不应存在可验证的单假设场景：{:?}",
            scenarios
                .iter()
                .map(|scenario| scenario
                    .hypotheses
                    .iter()
                    .map(|h| h.atom.display())
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        // 首选的联合场景：2 条假设、注入后目标可证、带验证证明
        let best = scenarios.first().expect("应有场景");
        assert_eq!(best.size(), 2);
        assert!(best.verified, "联合场景应通过注入验证");
        assert!(!best.solutions.is_empty(), "应有验证证明");
        assert!(best.joint_confidence > 0.0);
        // 循环解释过滤：场景成员不能就是目标本身
        assert!(scenarios.iter().all(|scenario| scenario
            .hypotheses
            .iter()
            .all(|hypothesis| hypothesis.atom != goal)));
    }

    fn answered_scenario(
        hypotheses: Vec<Hypothesis>,
        answer: &str,
        confidence: f64,
    ) -> Scenario {
        Scenario {
            hypotheses,
            verified: true,
            joint_confidence: confidence,
            solutions: vec![Solution {
                bindings: [("答案".to_string(), answer.to_string())]
                    .into_iter()
                    .collect(),
                proofs: Vec::new(),
                confidence,
            }],
        }
    }

    #[test]
    fn explanation_space_groups_answers_and_filters_rejections() {
        let h1 = hyp("师傅", "张三", "李四", 0.8);
        let h2 = hyp("师傅", "李四", "王五", 0.6);
        let h3 = hyp("师傅", "张三", "王五", 0.5);
        let mut space = ExplanationSpace::new(
            vec![
                answered_scenario(vec![h1.clone()], "王五", 0.8),
                answered_scenario(vec![h2.clone(), h3.clone()], "王五", 0.7),
                answered_scenario(vec![h2.clone()], "赵六", 0.6),
            ],
            vec![Atom::rel("师傅", "张三", "李四")],
            false,
        );
        assert_eq!(space.len(), 3);
        assert_eq!(space.split(&h1), (1, 2), "h1 只出现在 1 组解释中");
        assert_eq!(space.split(&h2), (2, 1), "h2 出现在 2 组解释中");
        let groups = space.answer_groups(&["答案".to_string()]);
        assert_eq!(groups.len(), 2, "应分成王五/赵六两组答案");
        assert_eq!(groups[0].answer, "王五");
        assert_eq!(groups[0].scenario_indices, vec![0, 1]);

        space.retain_compatible(&[h1.clone()]);
        assert_eq!(space.len(), 2, "含被否定假设的解释应整体删除");
        let groups = space.answer_groups(&["答案".to_string()]);
        assert_eq!(groups.len(), 2, "删除 h1 后仍有王五/赵六两组解释");

        space.retain_compatible(&[h3.clone()]);
        assert_eq!(space.len(), 1, "含 h3 的解释应删除");
        let groups = space.answer_groups(&["答案".to_string()]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].answer, "赵六");
        assert_eq!(groups[0].scenario_indices, vec![0]);
    }

    #[test]
    fn explanation_conclusion_distinguishes_robust_conditional_and_collapse() {
        let h1 = hyp("师傅", "张三", "李四", 0.8);
        let h2 = hyp("师傅", "李四", "王五", 0.6);
        let render = |atom: &Atom| atom.display();

        let robust = ExplanationSpace::new(
            vec![
                answered_scenario(vec![h1.clone()], "赵六", 0.8),
                answered_scenario(vec![h2.clone()], "赵六", 0.6),
            ],
            Vec::new(),
            false,
        );
        let text = render_explanation_conclusion(&robust, &["答案".to_string()], &render);
        assert!(text.contains("解释空间收敛"), "{}", text);
        assert!(text.contains("赵六"), "{}", text);
        assert!(text.contains("都成立"), "{}", text);

        let conditional = ExplanationSpace::new(
            vec![
                answered_scenario(vec![h1.clone()], "赵六", 0.8),
                answered_scenario(vec![h2.clone()], "王五", 0.6),
            ],
            Vec::new(),
            true,
        );
        let text = render_explanation_conclusion(&conditional, &["答案".to_string()], &render);
        assert!(text.contains("条件化回答"), "{}", text);
        assert!(text.contains("若") && text.contains("则答案是"), "{}", text);
        assert!(text.contains("不是完备枚举"), "有界枚举必须标注：{}", text);

        let empty = ExplanationSpace::new(Vec::new(), vec![h1.atom.clone()], false);
        let text = render_explanation_conclusion(&empty, &["答案".to_string()], &render);
        assert!(text.contains("解释空间崩塌"), "{}", text);
        assert!(text.contains("不是“知识库查无”"), "{}", text);
        assert!(text.contains("师傅(张三, 李四)"), "{}", text);
    }

    #[test]
    fn explanation_conclusion_keeps_multiple_answers_within_one_explanation() {
        let mut scenario = answered_scenario(vec![hyp("师傅", "张三", "李四", 0.8)], "赵六", 0.7);
        scenario.solutions.push(Solution {
            bindings: [("答案".to_string(), "王五".to_string())]
                .into_iter()
                .collect(),
            proofs: Vec::new(),
            confidence: 0.6,
        });
        let space = ExplanationSpace::new(vec![scenario], Vec::new(), false);
        let text = render_explanation_conclusion(&space, &["答案".to_string()], &|atom: &Atom| {
            atom.display()
        });
        assert!(
            text.contains("赵六 或 王五"),
            "同一解释的多个答案不能被压成单值：{}",
            text
        );
        assert!(text.contains("解释空间收敛"), "{}", text);
    }

    #[test]
    fn clarification_prefers_balanced_split_over_confidence() {
        let mut engine = ResolutionEngine::new(4, 4);
        engine.add_fact("师傅", "张三", "李四");
        let goals = [Atom::rel("师祖", "张三", "赵六")];
        let always = hyp("师傅", "张三", "李四", 0.95);
        let splitter = hyp("师傅", "李四", "王五", 0.1);
        let irrelevant = hyp("师傅", "张三", "钱七", 0.99);
        let space = ExplanationSpace::new(
            vec![
                answered_scenario(vec![always.clone()], "赵六", 0.9),
                answered_scenario(vec![always.clone(), splitter.clone()], "王五", 0.5),
            ],
            Vec::new(),
            false,
        );
        let mut session = ClarificationSession::start_with_goal(
            &[irrelevant.clone(), always.clone(), splitter.clone()],
            &engine,
            &goals,
            &["答案".to_string()],
        );
        session.attach_explanation_space(space);
        session.reprioritize(&engine, &goals, &["答案".to_string()]);
        assert_eq!(
            session.current().map(|hypothesis| hypothesis.atom.clone()),
            Some(splitter.atom.clone()),
            "应优先问能均衡划分解释空间的假设，而不是可信度最高但无判别力的那条"
        );
    }

    #[test]
    fn explanation_space_members_join_the_question_queue() {
        let stale = hyp("师傅", "张三", "钱七", 0.9);
        let decisive = hyp("师傅", "张三", "李四", 0.3);
        let space = ExplanationSpace::new(
            vec![
                answered_scenario(vec![decisive.clone()], "赵六", 0.7),
                answered_scenario(vec![decisive.clone()], "王五", 0.6),
            ],
            Vec::new(),
            false,
        );
        let mut engine = ResolutionEngine::new(4, 4);
        engine.add_fact("师傅", "李四", "赵六");
        engine.add_clause(teacher_clause());
        let goals = [Atom::rel("师祖", "张三", "赵六")];
        let mut session =
            ClarificationSession::start_with_goal(&[stale.clone()], &engine, &goals, &[]);
        session.attach_explanation_space(space);
        assert_eq!(
            session.include_explanation_hypotheses(),
            1,
            "解释空间中的新前提应并入队列"
        );
        assert_eq!(
            session.include_explanation_hypotheses(),
            0,
            "重复并入必须幂等"
        );
        session.reprioritize(&engine, &goals, &[]);
        assert_eq!(
            session.current().map(|hypothesis| hypothesis.atom.clone()),
            Some(decisive.atom.clone()),
            "新并入且能单独证明目标的前提应优先提问"
        );
    }

    #[test]
    fn collapsed_explanation_space_is_diagnosed_not_reported_as_unknown() {
        let h = hyp("师傅", "张三", "李四", 0.8);
        let space = ExplanationSpace::new(
            vec![answered_scenario(vec![h.clone()], "赵六", 0.8)],
            vec![h.atom.clone()],
            false,
        );
        let mut session = ClarificationSession::start(std::slice::from_ref(&h));
        session.attach_explanation_space(space);
        assert!(matches!(
            session.respond("不是"),
            ClarificationReply::Rejected(_)
        ));
        session.refresh_explanation_space(ExplanationSpace::new(
            Vec::new(),
            vec![h.atom.clone()],
            false,
        ));
        assert!(session.explanation_collapsed());
        assert!(session.progress_text().contains("解释空间已崩塌"));
        let render = |atom: &Atom| atom.display();
        let conclusion = session
            .explanation_conclusion(&["答案".to_string()], &render)
            .expect("崩塌应给出结论");
        assert!(conclusion.contains("解释空间崩塌"), "{}", conclusion);
        assert!(conclusion.contains("补充新事实"), "{}", conclusion);
    }

    #[test]
    fn batch_confirmation_answers_all_remaining_questions() {
        let hypotheses = vec![
            hyp("师傅", "张三", "李四", 0.9),
            hyp("师傅", "张三", "王五", 0.6),
            hyp("师傅", "张三", "赵六", 0.4),
        ];
        let mut session = ClarificationSession::start(&hypotheses);
        let reply = session.respond("全部确认");
        let ClarificationReply::ConfirmedAll(batch) = reply else {
            panic!("应识别为批量确认：{:?}", reply)
        };
        assert_eq!(batch.len(), 3);
        assert!(session.is_finished());
        assert!(session.queue().is_empty());
        assert_eq!(
            session.confirmed.len(),
            3,
            "批量确认要全部记入已确认"
        );
    }

    #[test]
    fn correction_wins_over_batch_rejection() {
        let hypotheses = vec![
            hyp("师傅", "张三", "李四", 0.9),
            hyp("师傅", "张三", "王五", 0.6),
        ];
        let mut session = ClarificationSession::start(&hypotheses);
        let reply = session.respond("都不是，是王五");
        let ClarificationReply::Corrected {
            rejected,
            confirmed,
        } = reply
        else {
            panic!("应识别为更正而不是批量否定：{:?}", reply)
        };
        assert_eq!(rejected.atom.display(), hypotheses[0].atom.display());
        assert_eq!(confirmed.atom.display(), hypotheses[1].atom.display());
        assert!(session.is_finished());
        assert_eq!(session.rejected.len(), 1);
        assert_eq!(session.confirmed.len(), 1);
    }

    #[test]
    fn plain_no_still_rejects_only_the_current_question() {
        let hypotheses = vec![
            hyp("师傅", "张三", "李四", 0.9),
            hyp("师傅", "张三", "王五", 0.6),
        ];
        let mut session = ClarificationSession::start(&hypotheses);
        let reply = session.respond("不是");
        assert!(matches!(reply, ClarificationReply::Rejected(_)));
        assert_eq!(session.rejected.len(), 1);
        assert_eq!(session.remaining(), 1, "普通否定不应批量排除剩余问题");

        let mut session = ClarificationSession::start(&hypotheses);
        let reply = session.respond("都不是");
        assert!(matches!(reply, ClarificationReply::RejectedAll(_)));
        assert!(session.is_finished());
        assert_eq!(session.rejected.len(), 2);
    }

    #[test]
    fn retry_keeps_the_rejected_question_current() {
        let first = hyp("师傅", "张三", "李四", 0.8);
        let second = hyp("师傅", "张三", "王五", 0.5);
        let mut session = ClarificationSession::start(&[first.clone(), second.clone()]);
        // 模拟一次失败写入：会话已经推进到第二问。
        let reply = session.respond("是");
        assert!(matches!(reply, ClarificationReply::Confirmed(_)));
        assert_eq!(
            session
                .current()
                .map(|hypothesis| hypothesis.atom.display()),
            Some(second.atom.display())
        );

        session.retry(first.clone());
        assert_eq!(
            session.current().map(|hypothesis| hypothesis.atom.clone()),
            Some(first.atom),
            "被预检拒绝的问题必须重新成为当前问题"
        );
        assert_eq!(
            session
                .queue()
                .first()
                .map(|hypothesis| hypothesis.atom.clone()),
            Some(second.atom),
            "原本的下一问回到队列，不能丢失"
        );
    }

    #[test]
    fn clarification_correction_switches_to_another_candidate() {
        let current = hyp("师傅", "张三", "李四", 0.8);
        let alternative = hyp("师傅", "张三", "王五", 0.5);
        let other_subject = hyp("师傅", "王五", "李四", 0.5);
        let mut session =
            ClarificationSession::start(&[current, alternative.clone(), other_subject]);

        let reply = session.respond("不是，是王五");
        match reply {
            ClarificationReply::Corrected {
                rejected,
                confirmed,
            } => {
                assert_eq!(rejected.atom.display(), "师傅(张三, 李四)");
                assert_eq!(confirmed.atom, alternative.atom);
            }
            other => panic!("应识别为更正，实际：{:?}", other),
        }
        assert_eq!(session.confirmed.len(), 1);
        assert_eq!(session.rejected.len(), 1);
        assert_eq!(
            session
                .current()
                .map(|hypothesis| hypothesis.atom.display()),
            Some("师傅(王五, 李四)".to_string()),
            "已更正的候选必须从队列中移除"
        );
        assert_eq!(session.confirmed[0].atom.display(), "师傅(张三, 王五)");
    }

    #[test]
    fn clarification_negated_alternative_is_not_a_correction() {
        let current = hyp("师傅", "张三", "李四", 0.8);
        let alternative = hyp("师傅", "张三", "王五", 0.5);
        let mut session = ClarificationSession::start(&[current, alternative]);
        let reply = session.respond("不是王五");
        assert!(
            matches!(reply, ClarificationReply::Rejected(_)),
            "否定的是替代取值而不是当前取值时应保持普通否定：{:?}",
            reply
        );
        assert!(session.confirmed.is_empty());
    }

    #[test]
    fn clarification_pool_keeps_the_strongest_duplicate() {
        let weak = hyp("师傅", "李四", "赵六", 0.3);
        let strong = hyp("师傅", "李四", "赵六", 0.8);
        let session = ClarificationSession::start(&[weak, strong]);
        assert_eq!(session.remaining(), 1, "同一原子只应提问一次");
        let current = session.current().expect("应有当前问题");
        assert_eq!(current.atom.display(), "师傅(李四, 赵六)");
        assert_eq!(current.confidence, 0.8, "应保留可信度更高的重复假设");
    }

    #[test]
    fn clarification_session_advances_on_answers() {
        let h1 = hyp("师傅", "李四", "赵六", 0.6);
        let h2 = hyp("师傅", "王五", "赵六", 0.5);
        let mut session = ClarificationSession::start(&[h2, h1]);
        // 高可信度优先提问
        assert!(session
            .current()
            .expect("应有当前问题")
            .atom
            .display()
            .contains("李四"));
        let reply = session.respond("是的，确实如此");
        assert!(matches!(reply, ClarificationReply::Confirmed(_)));
        assert_eq!(session.confirmed.len(), 1);
        // 第二问：否定后队列清空
        let reply = session.respond("不是");
        assert!(matches!(reply, ClarificationReply::Rejected(_)));
        assert!(session.is_finished());
        // 无法识别的回答不推进；"不确定"按 Unknown 处理
        let mut session = ClarificationSession::start(&[hyp("师傅", "李四", "赵六", 0.6)]);
        let reply = session.respond("今天天气不错");
        assert!(matches!(reply, ClarificationReply::Unrecognized));
        assert_eq!(session.remaining(), 1);
        let reply = session.respond("不知道");
        assert!(matches!(reply, ClarificationReply::Deferred(_)));
        assert!(session.is_finished());
    }

    #[test]
    fn clarification_prefers_hypothesis_that_decides_the_goal() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("师傅", "张三", "李四");
        engine.add_clause(teacher_clause());
        let decisive = hyp("师傅", "李四", "赵六", 0.2);
        let noisy = hyp("师傅", "张三", "钱七", 0.9);
        let goals = [Atom::rel("师祖", "张三", "赵六")];
        let session =
            ClarificationSession::start_with_goal(&[noisy, decisive.clone()], &engine, &goals, &[]);
        assert_eq!(
            session.current().map(|hypothesis| hypothesis.clone().atom),
            Some(decisive.atom.clone()),
            "应优先问能直接让目标可证的假设，即使它可信度更低"
        );
        assert_eq!(
            ClarificationSession::discrimination(&engine, &goals, &[], &decisive),
            2
        );
    }

    #[test]
    fn reprioritize_keeps_the_pending_question_in_the_pool() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("师傅", "张三", "李四");
        engine.add_clause(teacher_clause());
        let goals = [Atom::rel("师祖", "张三", "赵六")];
        let decisive = hyp("师傅", "李四", "赵六", 0.2);
        let noisy = hyp("师傅", "张三", "钱七", 0.9);
        let mut session =
            ClarificationSession::start_with_goal(&[noisy, decisive], &engine, &goals, &[]);
        let before = session.remaining();
        let reply = session.respond("不是");
        assert!(matches!(reply, ClarificationReply::Rejected(_)));
        session.reprioritize(&engine, &goals, &[]);
        assert_eq!(
            session.remaining(),
            before - 1,
            "重排不得丢失尚未回答的问题"
        );
    }

    #[test]
    fn classify_answer_prefers_longest_keyword() {
        assert_eq!(classify_answer("是"), AnswerKind::Yes);
        assert_eq!(classify_answer("嗯，对"), AnswerKind::Yes);
        assert_eq!(classify_answer("不是"), AnswerKind::No);
        // "不确定" 必须优先于 "不"，"不知道" 优先于 "不"
        assert_eq!(classify_answer("不确定"), AnswerKind::Unknown);
        assert_eq!(classify_answer("不知道呢"), AnswerKind::Unknown);
        assert_eq!(classify_answer("也许吧"), AnswerKind::Unknown);
        assert_eq!(classify_answer("banana"), AnswerKind::Unrecognized);
    }

    #[test]
    fn functional_hypotheses_are_mutually_exclusive() {
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&[
            PrologFact::Relation {
                from: "中国".into(),
                rel_type: "首都".into(),
                to: "北京".into(),
            },
            PrologFact::Relation {
                from: "法国".into(),
                rel_type: "首都".into(),
                to: "巴黎".into(),
            },
            PrologFact::Relation {
                from: "日本".into(),
                rel_type: "首都".into(),
                to: "东京".into(),
            },
        ]);
        let beijing = hyp("首都", "中国", "北京", 0.5);
        let shanghai = hyp("首都", "中国", "上海", 0.4);
        let paris = hyp("首都", "法国", "北京", 0.4);
        // 同一主体的功能谓词给出不同取值 ⇒ 互斥
        assert!(abduction.hypotheses_conflict(&beijing, &shanghai));
        // 不同主体并不互斥
        assert!(!abduction.hypotheses_conflict(&beijing, &paris));
    }

    #[test]
    fn object_type_mismatch_is_flagged() {
        let mut engine = ResolutionEngine::new(6, 5);
        engine.add_fact("首都", "中国", "北京");
        engine.add_fact("首都", "法国", "巴黎");
        engine.add_fact("首都", "日本", "东京");
        let facts = vec![
            PrologFact::Relation {
                from: "中国".into(),
                rel_type: "首都".into(),
                to: "北京".into(),
            },
            PrologFact::Relation {
                from: "法国".into(),
                rel_type: "首都".into(),
                to: "巴黎".into(),
            },
            PrologFact::Relation {
                from: "日本".into(),
                rel_type: "首都".into(),
                to: "东京".into(),
            },
            PrologFact::Entity {
                name: "中国".into(),
                entity_type: "国家".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "北京".into(),
                entity_type: "城市".into(),
                properties: HashMap::new(),
            },
        ];
        let mut abduction = AbductionEngine::new(6, 5);
        abduction.observe_facts(&facts);
        let report = abduction.analyze(&engine, &[Atom::rel("首都", "中国", "张三")], &[]);
        let best = report.best_hypothesis().expect("应有假设");
        assert!(
            best.conflicts
                .iter()
                .any(|conflict| conflict.contains("宾体类型")),
            "类型不相容应被记录为冲突：{:?}",
            best.conflicts
        );
    }

    #[test]
    fn dedupe_keeps_confidence_order_and_best_duplicate() {
        let mut hypotheses = vec![
            hyp("师傅", "张三", "李四", 0.9),
            hyp("师傅", "张三", "王五", 0.7),
            hyp("师傅", "张三", "李四", 0.3),
            hyp("师傅", "李四", "王五", 0.5),
        ];
        dedupe_hypotheses(&mut hypotheses);
        let displays: Vec<String> = hypotheses.iter().map(|h| h.atom.display()).collect();
        assert_eq!(
            displays,
            vec!["师傅(张三, 李四)", "师傅(张三, 王五)", "师傅(李四, 王五)"],
            "去重后应保持可信度降序"
        );
        assert_eq!(
            hypotheses[0].confidence, 0.9,
            "同一原子应保留置信度最高的一条"
        );
    }
}

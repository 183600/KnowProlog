impl SemanticMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.clamp(1, 32),
            forms: Vec::new(),
            focus_entities: Vec::new(),
            last_predicate: None,
        }
    }

    /// 记录一条逻辑形式（未知形式不进入记忆）
    pub fn record(&mut self, lf: &LogicalForm) {
        if !lf.is_known() {
            return;
        }
        let (entity, predicate) = lf.focus();
        if let Some(predicate) = predicate {
            self.last_predicate = Some(predicate);
        }
        if let Some(entity) = entity {
            self.focus_entities.retain(|existing| existing != &entity);
            self.focus_entities.push(entity);
            while self.focus_entities.len() > 8 {
                self.focus_entities.remove(0);
            }
        }
        self.forms.push(lf.clone());
        while self.forms.len() > self.capacity {
            self.forms.remove(0);
        }
    }

    pub fn forms_len(&self) -> usize {
        self.forms.len()
    }

    pub fn focus_entities(&self) -> &[String] {
        &self.focus_entities
    }

    pub fn last_entity(&self) -> Option<&str> {
        self.focus_entities.last().map(String::as_str)
    }

    pub fn last_predicate(&self) -> Option<&str> {
        self.last_predicate.as_deref()
    }

    pub fn clear(&mut self) {
        self.forms.clear();
        self.focus_entities.clear();
        self.last_predicate = None;
    }

    /// 多轮上下文改写：指代消解（他/她/它 → 话题实体）+ 省略补全（「X呢」→「X 的上一谓词 是多少」）
    pub fn contextualize(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        // 省略补全
        let stripped = trimmed.trim_end_matches(['？', '?', '。', '.']).trim();
        let mut candidate = stripped.to_string();
        if let Some(stem) = stripped.strip_suffix('呢') {
            let stem = stem.trim();
            let stem = stem.strip_prefix("那").map(str::trim).unwrap_or(stem);
            let predicate = self.last_predicate()?;
            candidate = if stem.is_empty() {
                let entity = self.last_entity()?;
                format!("{}的{}是多少", entity, predicate)
            } else {
                format!("{}的{}是多少", stem, predicate)
            };
        }
        // 指代消解
        if let Some(entity) = self.last_entity() {
            for pronoun in ["他们", "他", "她", "它"] {
                if candidate.contains(pronoun) {
                    let replaced = candidate.replacen(pronoun, entity, 1);
                    if replaced != candidate {
                        return Some(replaced);
                    }
                }
            }
        }
        if candidate != trimmed {
            return Some(candidate);
        }
        None
    }

    pub fn summary(&self) -> String {
        let focus = if self.focus_entities.is_empty() {
            "（空）".to_string()
        } else {
            self.focus_entities.join(" → ")
        };
        let predicate = self
            .last_predicate
            .clone()
            .unwrap_or_else(|| "（无）".to_string());
        format!(
            "语义记忆：最近逻辑形式 {} 条，话题实体 [{}]，上一谓词「{}」。",
            self.forms.len(),
            focus,
            predicate
        )
    }
}
fn is_punctuation(word: &str) -> bool {
    word.chars().all(|ch| {
        matches!(
            ch,
            '，' | '。'
                | '？'
                | '！'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '《'
                | '》'
                | '—'
                | '…'
                | ','
                | '.'
                | '?'
                | '!'
                | ';'
                | ':'
                | '"'
                | '\''
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | ' '
                | '\t'
                | '\n'
                | '\r'
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    fn feed(semantics: &mut FormalSemantics, model: &mut TinyModel, facts: &[PrologFact]) {
        for fact in facts {
            model.observe_fact(fact);
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    semantics.register_binary_predicate(rel_type);
                    semantics.register_entity(from);
                    semantics.register_entity(to);
                    if SUPER_RELS.contains(&rel_type.as_str()) {
                        semantics.register_class(from);
                        semantics.register_class(to);
                    }
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    semantics.register_entity(name);
                    semantics.register_class(entity_type);
                    for key in properties.keys() {
                        semantics.register_unary_predicate(key);
                    }
                }
            }
        }
    }

    fn people_setup() -> (FormalSemantics, TinyModel) {
        let mut semantics = FormalSemantics::new();
        let mut model = TinyModel::new(HashMap::new());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "中国人".to_string(),
                properties: HashMap::from([
                    ("年龄".to_string(), "30".to_string()),
                    ("国籍".to_string(), "中国".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "美国人".to_string(),
                properties: HashMap::from([("年龄".to_string(), "28".to_string())]),
            },
            PrologFact::Relation {
                from: "学生".to_string(),
                rel_type: "子类".to_string(),
                to: "人".to_string(),
            },
            PrologFact::Entity {
                name: "小明".to_string(),
                entity_type: "学生".to_string(),
                properties: HashMap::new(),
            },
        ];
        feed(&mut semantics, &mut model, &facts);
        (semantics, model)
    }

    #[test]
    fn type_assertion_and_negation() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三是中国人");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("张三不是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("张三是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
    }

    #[test]
    fn possessive_statement_and_value_query() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三的年龄是30");
        assert!(
            matches!(
                semantics.solve(&lf, &model),
                QueryOutcome::Truth { holds: true, .. }
            ),
            "lf = {:?}",
            lf
        );
        let lf = semantics.parse("李四的年龄是30");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
        let lf = semantics.parse("张三的年龄是多少");
        match semantics.solve(&lf, &model) {
            QueryOutcome::Values { values, .. } => assert_eq!(values, vec!["30".to_string()]),
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn bare_possessive_is_value_query() {
        let (semantics, _model) = people_setup();
        let lf = semantics.parse("张三的年龄");
        match lf.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                assert_eq!(entity, "张三");
                assert_eq!(predicate, "年龄");
            }
            other => panic!("unexpected kind: {:?}", other),
        }
    }

    #[test]
    fn entity_query_who_is_class() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("谁是中国人");
        match semantics.solve(&lf, &model) {
            QueryOutcome::Entities { entities, .. } => {
                assert_eq!(entities, vec!["张三".to_string()]);
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn reverse_entity_query_keeps_relation_direction() {
        let mut semantics = FormalSemantics::new();
        let mut model = TinyModel::new(HashMap::new());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "赵六".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ];
        feed(&mut semantics, &mut model, &facts);
        let lf = semantics.parse("谁的师傅是李四");
        assert!(
            matches!(
                lf.kind,
                LfKind::Query(Query::EntityByObject { .. })
            ),
            "lf = {:?}",
            lf
        );
        match semantics.solve(&lf, &model) {
            QueryOutcome::EntitiesByObject {
                entities, evidence, ..
            } => {
                assert_eq!(entities.len(), 2, "entities = {:?}", entities);
                assert!(entities.contains(&"张三".to_string()));
                assert!(entities.contains(&"王五".to_string()));
                assert!(!entities.contains(&"赵六".to_string()));
                assert!(evidence.iter().all(|line| line.contains("李四")));
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
        let forest = semantics.parse_forest("谁的师傅是李四");
        assert!(forest.candidates.iter().any(|candidate| matches!(
            candidate.form.kind,
            LfKind::Query(Query::EntityByObject { .. })
        )));

        let lf = semantics.parse("谁是李四的师傅");
        match semantics.solve(&lf, &model) {
            QueryOutcome::Values { values, .. } => {
                assert_eq!(values, vec!["赵六".to_string()], "lf = {:?}", lf);
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn subclass_closure_propagates() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("小明是人");
        assert!(
            matches!(
                semantics.solve(&lf, &model),
                QueryOutcome::Truth { holds: true, .. }
            ),
            "lf = {:?}",
            lf
        );
    }

    #[test]
    fn truth_question_with_ma() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("张三是中国人吗");
        match lf.kind {
            LfKind::Query(Query::Truth { .. }) => {}
            other => panic!("unexpected kind: {:?}", other),
        }
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
    }

    #[test]
    fn universal_quantifier_evaluates() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("所有中国人都是中国人");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
        let lf = semantics.parse("所有中国人都是美国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
    }

    #[test]
    fn conditional_combines_two_clauses() {
        let mut semantics = FormalSemantics::new();
        let model = TinyModel::new(HashMap::new());
        semantics.register_unary_predicate("下雨");
        semantics.register_unary_predicate("地湿");
        let lf = semantics.parse("如果下雨那么地湿");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        let rendered = lf.render();
        assert!(rendered.contains("→"), "rendered = {}", rendered);
        let _ = model.fact_count();
    }

    #[test]
    fn conditional_accepts_ze_connective() {
        let mut semantics = FormalSemantics::new();
        semantics.register_unary_predicate("下雨");
        semantics.register_unary_predicate("地湿");
        let lf = semantics.parse("如果下雨则地湿");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        let rendered = lf.render();
        assert!(rendered.contains("→"), "rendered = {}", rendered);
    }

    #[test]
    fn existential_quantifier_requires_witness() {
        let (semantics, model) = people_setup();
        let lf = semantics.parse("有些中国人是美国人");
        assert!(matches!(lf.kind, LfKind::Statement(_)), "lf = {:?}", lf);
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: false, .. }
        ));
        let lf = semantics.parse("有些中国人是中国人");
        assert!(matches!(
            semantics.solve(&lf, &model),
            QueryOutcome::Truth { holds: true, .. }
        ));
    }

    #[test]
    fn unknown_input_gives_unknown_kind() {
        let semantics = FormalSemantics::new();
        let lf = semantics.parse("量子色动力学");
        assert!(!lf.is_known(), "lf = {:?}", lf);
    }

    #[test]
    fn semantic_memory_contextualizes_ellipsis_and_pronoun() {
        let mut memory = SemanticMemory::new(6);
        let lf = LogicalForm {
            source: "张三的年龄是多少".to_string(),
            tokens: vec!["张三".to_string(), "的".to_string(), "年龄".to_string()],
            kind: LfKind::Query(Query::Value {
                entity: "张三".to_string(),
                predicate: "年龄".to_string(),
            }),
            trace: Vec::new(),
        };
        memory.record(&lf);
        assert_eq!(memory.last_entity(), Some("张三"));
        assert_eq!(memory.last_predicate(), Some("年龄"));
        assert_eq!(
            memory.contextualize("李四呢").as_deref(),
            Some("李四的年龄是多少")
        );
        assert_eq!(
            memory.contextualize("他的年龄是多少").as_deref(),
            Some("张三的年龄是多少")
        );
        assert_eq!(memory.contextualize("张三的年龄是多少"), None);
    }

    #[test]
    fn parse_forest_keeps_multiple_readings() {
        let (semantics, _model) = people_setup();
        let forest = semantics.parse_forest("张三的年龄");
        assert!(
            forest.candidates.len() >= 2,
            "森林不应压平成单条：{:?}",
            forest
        );
        let legacy = semantics.parse("张三的年龄");
        assert!(
            forest
                .candidates
                .iter()
                .any(|candidate| candidate.form.signature() == legacy.signature()),
            "旧路径读法必须保留在森林中"
        );
    }

    #[test]
    fn forest_answers_only_evidence_backed_reading() {
        let (semantics, model) = people_setup();
        let evaluation = semantics.evaluate("张三的年龄是多少", &model);
        assert!(
            matches!(evaluation.decision, ForestDecision::Answer { .. }),
            "evaluation = {:?}",
            evaluation
        );
        let answer = semantics
            .answer_evaluation(&evaluation, true)
            .expect("应给出唯一答案");
        assert!(answer.contains("30"), "answer = {}", answer);
        assert!(answer.contains("证据"), "answer = {}", answer);
    }

    #[test]
    fn unknown_entity_stays_an_open_gap() {
        let (semantics, model) = people_setup();
        let evaluation = semantics.evaluate("王五的年龄是多少", &model);
        assert!(
            matches!(evaluation.decision, ForestDecision::Unknown { .. }),
            "evaluation = {:?}",
            evaluation
        );
        assert!(evaluation
            .grounded
            .iter()
            .all(|candidate| candidate.status != CandidateStatus::Supported));
        assert!(evaluation
            .grounded
            .iter()
            .any(|candidate| candidate.gaps.iter().any(|gap| gap.contains("王五"))));
        assert!(semantics.answer_evaluation(&evaluation, true).is_none());
    }

    #[test]
    fn distinct_supported_readings_are_reported_as_ambiguous() {
        let (semantics, model) = people_setup();
        let make = |predicate: &str| LogicalForm {
            source: "张三的年龄和国籍".to_string(),
            tokens: vec!["张三".into(), "的".into(), predicate.into()],
            kind: LfKind::Query(Query::Value {
                entity: "张三".to_string(),
                predicate: predicate.to_string(),
            }),
            trace: Vec::new(),
        };
        let forest = ParseForest {
            source: "张三的年龄和国籍".to_string(),
            tokens: vec!["张三".into(), "的".into(), "年龄和国籍".into()],
            candidates: vec![
                ParseCandidate {
                    form: make("年龄"),
                    strategy: ParseStrategy::QuestionPattern,
                    cost: 10,
                },
                ParseCandidate {
                    form: make("国籍"),
                    strategy: ParseStrategy::Chart,
                    cost: 20,
                },
            ],
            trace: Vec::new(),
        };
        let evaluation = semantics.evaluate_forest(forest, &model);
        match evaluation.decision {
            ForestDecision::Ambiguous { ref indexes } => assert_eq!(indexes.len(), 2),
            other => panic!("unexpected decision: {:?}", other),
        }
        assert!(evaluation.answer().is_none());
        let report = semantics.render_forest_report(&evaluation);
        assert!(report.contains("歧义"), "report = {}", report);
    }
}

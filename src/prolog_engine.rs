use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as _;
use scryer_prolog::machine::Machine;
use scryer_prolog::machine::parsed_results::{prolog_value_to_json_string, QueryResolution, Value};

pub struct PrologEngine {
    machine: Machine,
    /// 已加载的动态事实（按规范序排序），用于跳过重复加载
    loaded_facts: Vec<PrologFact>,
}

impl PrologEngine {
    pub fn new() -> Self {
        let mut machine = Machine::new_lib();
        
        let base_rules = r#"
            % 列表谓词（member/2、length/2 等）
            :- use_module(library(lists)).
            
            % 动态事实谓词声明（支持 retractall 更新事实）
            :- dynamic(oxigraph_entity/3).
            :- dynamic(oxigraph_relation/3).
            
            % 基础谓词定义
            entity(Name, Type, Properties) :- 
                oxigraph_entity(Name, Type, Properties).
            
            % 关系谓词
            relation(From, Type, To) :- 
                oxigraph_relation(From, Type, To).
            
            % 年龄查询规则
            age_of(Person, Age) :- 
                entity(Person, person, Props),
                member(age-Age, Props).
            
            % 关系查询规则
            related(Person1, Relation, Person2) :- 
                relation(Person1, Relation, Person2).
            
            related(Person1, Relation, Person2) :- 
                relation(Person2, Relation, Person1).
            
            % 计数规则
            count_entities(Type, Count) :- 
                findall(Name, entity(Name, Type, _), Names),
                length(Names, Count).
            
            % 属性查询规则
            has_property(Entity, Property, Value) :- 
                entity(Entity, _, Props),
                member(Property-Value, Props).
            
            % 复合查询规则
            spouse_of(Person1, Person2) :- 
                related(Person1, spouse, Person2).
            
            parent_of(Parent, Child) :- 
                related(Parent, has_child, Child).
            
            % 传递关系
            ancestor_of(Ancestor, Person) :- 
                parent_of(Ancestor, Person).
            
            ancestor_of(Ancestor, Person) :- 
                parent_of(Ancestor, Parent),
                ancestor_of(Parent, Person).
        "#;
        
        machine.consult_module_string("user", base_rules.to_string());
        
        Self { machine, loaded_facts: Vec::new() }
    }
    
    pub fn load_facts(&mut self, facts: Vec<PrologFact>) {
        println!("Loading {} Prolog facts", facts.len());

        // 事实集与加载顺序都未变化时无需排序
        if facts == self.loaded_facts {
            return;
        }

        // 再按规范序排序，使相同事实集在不同调用间可直接比较（与加载顺序无关）
        let mut facts = facts;
        facts.sort_by(Self::compare_facts);

        // 事实集未变化时跳过清空与重载，机器状态保持一致
        if facts == self.loaded_facts {
            return;
        }

        self.clear_dynamic_facts();
        let mut fact_string = String::with_capacity(facts.len() * 64);
        
        for fact in &facts {
            match fact {
                PrologFact::Entity { name, entity_type, properties } => {
                    let _ = write!(
                        fact_string,
                        "oxigraph_entity('{}', '{}', [",
                        escape_atom(name),
                        escape_atom(entity_type)
                    );
                    for (i, (k, v)) in properties.iter().enumerate() {
                        if i > 0 {
                            fact_string.push_str(", ");
                        }
                        // 属性为 '键'-'值' 复合项
                        let _ = write!(fact_string, "'{}'-'{}'", escape_atom(k), escape_atom(v));
                    }
                    fact_string.push_str("]).\n");
                }
                PrologFact::Relation { from, rel_type, to } => {
                    let _ = write!(
                        fact_string,
                        "oxigraph_relation('{}', '{}', '{}').\n",
                        escape_atom(from),
                        escape_atom(rel_type),
                        escape_atom(to)
                    );
                }
            }
        }
        
        self.machine.consult_module_string("user", fact_string);
        self.loaded_facts = facts;
    }

    /// 事实的规范比较顺序：先实体后关系，各自按字段字典序
    fn compare_facts(a: &PrologFact, b: &PrologFact) -> Ordering {
        match (a, b) {
            (
                PrologFact::Entity { name: n1, entity_type: t1, .. },
                PrologFact::Entity { name: n2, entity_type: t2, .. },
            ) => n1.cmp(n2).then_with(|| t1.cmp(t2)),
            (PrologFact::Entity { .. }, PrologFact::Relation { .. }) => Ordering::Less,
            (PrologFact::Relation { .. }, PrologFact::Entity { .. }) => Ordering::Greater,
            (
                PrologFact::Relation { from: f1, rel_type: r1, to: o1 },
                PrologFact::Relation { from: f2, rel_type: r2, to: o2 },
            ) => f1.cmp(f2).then_with(|| r1.cmp(r2)).then_with(|| o1.cmp(o2)),
        }
    }

    pub fn load_relevant_facts(&mut self, facts: Vec<PrologFact>) {
        self.load_facts(facts);
    }

    fn clear_dynamic_facts(&mut self) {
        for goal in [
            "retractall(oxigraph_entity(_, _, _)).",
            "retractall(oxigraph_relation(_, _, _)).",
        ] {
            match self.machine.run_query(goal.to_string()) {
                Ok(_) => {},
                Err(e) => eprintln!("清除动态事实时出错: {}", e),
            }
        }
    }
    
    pub fn query(&mut self, query: &str) -> Vec<HashMap<String, String>> {
        let mut results = Vec::new();
        
        let query_str = format!("{}.", query);
        println!("执行Prolog查询: {}", query_str);
        
        match self.machine.run_query(query_str) {
            Ok(query_output) => {
                println!("Prolog输出: {:?}", query_output);
                
                let mut succeeded_without_bindings = false;
                
                match query_output {
                    QueryResolution::Matches(matches) => {
                        // 直接提取变量绑定（如 Age/Relation/Count/Spouse）
                        for query_match in matches {
                            if query_match.bindings.is_empty() {
                                continue;
                            }
                            let result_map: HashMap<String, String> = query_match.bindings
                                .into_iter()
                                .map(|(name, value)| (name, value_to_string(value)))
                                .collect();
                            results.push(result_map);
                        }
                    }
                    QueryResolution::True => {
                        succeeded_without_bindings = true;
                    }
                    QueryResolution::False => {
                        println!("查询未返回结果");
                    }
                }
                
                if succeeded_without_bindings {
                    // 查询成功但没有变量绑定 → 无可提取的答案。
                    // 不做任何兜底填空：宁可诚实地说「没有记录」，
                    // 也不返回与知识库无关的硬编码内容（无幻觉）。
                }
            }
            Err(err) => {
                eprintln!("Prolog查询错误: {}", err);
            }
        }
        
        results
    }
    
    pub fn generate_prolog_query(pattern: &str, entities: &[String]) -> Option<String> {
        match pattern {
            "age_query" => {
                if let Some(person) = entities.first() {
                    Some(format!("age_of('{}', Age)", escape_atom(person)))
                } else {
                    None
                }
            }
            "relation_query" => {
                if entities.len() >= 2 {
                    Some(format!(
                        "related('{}', Relation, '{}')",
                        escape_atom(&entities[0]),
                        escape_atom(&entities[1])
                    ))
                } else {
                    None
                }
            }
            "count_query" => {
                if let Some(entity_type) = entities.first() {
                    Some(format!("count_entities('{}', Count)", escape_atom(entity_type)))
                } else {
                    None
                }
            }
            "spouse_query" => {
                if let Some(person) = entities.first() {
                    Some(format!("spouse_of('{}', Spouse)", escape_atom(person)))
                } else {
                    None
                }
            }
            "capital_query" => {
                if let Some(country) = entities.first() {
                    Some(format!(
                        "oxigraph_entity('{}', 'country', Properties), member(capital-Capital, Properties)",
                        escape_atom(country)
                    ))
                } else {
                    None
                }
            }
            "altitude_query" => {
                if let Some(place) = entities.first() {
                    Some(format!("oxigraph_entity('{}', 'geographicfeature', Properties), member(altitude-Altitude, Properties)", escape_atom(place)))
                } else {
                    None
                }
            }
            "chemical_formula_query" => {
                if let Some(chemical) = entities.first() {
                    Some(format!("oxigraph_entity('{}', 'chemical', Properties), member(formula-Formula, Properties)", escape_atom(chemical)))
                } else {
                    None
                }
            }
            "author_query" => {
                if let Some(work) = entities.first() {
                    Some(format!("oxigraph_entity('{}', 'literature', Properties), member(author-Author, Properties)", escape_atom(work)))
                } else {
                    None
                }
            }
            "champion_query" => {
                if let Some(event) = entities.first() {
                    Some(format!("oxigraph_entity('{}', 'sportevent', Properties), member(champion-Champion, Properties)", escape_atom(event)))
                } else {
                    None
                }
            }
            "birth_year_query" => {
                if let Some(language) = entities.first() {
                    Some(format!("oxigraph_entity('{}', 'programminglanguage', Properties), member(birth_year-Year, Properties)", escape_atom(language)))
                } else {
                    None
                }
            }
            "largest_query" => {
                if entities.len() >= 2 {
                    Some(format!("oxigraph_entity(Entity, '{}', Properties), member(type-'{}', Properties), member(size_rank-'1', Properties)", escape_atom(&entities[0]), escape_atom(&entities[1])))
                } else {
                    None
                }
            }
            "quantity_query" => {
                if entities.len() >= 2 {
                    Some(format!("oxigraph_entity('{}', 'biologicalconcept', Properties), member(quantity-Quantity, Properties), member(unit-'{}', Properties)", escape_atom(&entities[0]), escape_atom(&entities[1])))
                } else {
                    None
                }
            }
            "concept_query" => {
                if let Some(concept) = entities.first() {
                    Some(format!("oxigraph_entity('{}', Type, Properties)", escape_atom(concept)))
                } else {
                    None
                }
            }
            "property_query" => {
                if entities.len() >= 2 {
                    // 为基础知识属性查询生成通用查询，调整实体顺序
                    Some(format!(
                        "oxigraph_relation('{}', Property, '{}')",
                        escape_atom(&entities[1]),
                        escape_atom(&entities[0])
                    ))
                } else if let Some(entity) = entities.first() {
                    // 单实体查询，查找所有属性
                    Some(format!("oxigraph_entity('{}', Type, Properties)", escape_atom(entity)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// 转义 Prolog 单引号原子中的特殊字符，避免名称里的引号/反斜杠破坏事实或查询语法
fn escape_atom(text: &str) -> std::borrow::Cow<'_, str> {
    // 常见情况无需转义：直接借用原字符串
    if !text
        .as_bytes()
        .iter()
        .any(|b| matches!(b, b'\\' | b'\'' | b'\n' | b'\r' | b'\t'))
    {
        return std::borrow::Cow::Borrowed(text);
    }

    let mut escaped = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\'' => escaped.push_str("''"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    std::borrow::Cow::Owned(escaped)
}

fn value_to_string(value: Value) -> String {
    match value {
        // 字符串字面量取内容本身
        Value::String(s) => s,
        other => prolog_value_to_json_string(other),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrologFact {
    Entity {
        name: String,
        entity_type: String,
        properties: HashMap<String, String>,
    },
    Relation {
        from: String,
        rel_type: String,
        to: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_atom_escapes_quotes_and_backslashes() {
        assert_eq!(escape_atom("O'Brien"), "O''Brien");
        assert_eq!(escape_atom("a\\b"), "a\\\\b");
        assert_eq!(escape_atom("line\nbreak"), "line\\nbreak");
        assert_eq!(escape_atom("张三"), "张三");
    }

    #[test]
    fn test_load_facts_with_quotes_does_not_break_engine() {
        let mut engine = PrologEngine::new();
        let facts = vec![
            PrologFact::Entity {
                name: "O'Brien".to_string(),
                entity_type: "person".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("note".to_string(), "it's fine".to_string());
                    props
                },
            },
            PrologFact::Relation {
                from: "O'Brien".to_string(),
                rel_type: "knows".to_string(),
                to: "back\\slash".to_string(),
            },
        ];
        engine.load_facts(facts);
        let results = engine.query("oxigraph_entity(Name, Type, _)");
        assert!(
            results
                .iter()
                .any(|result| result.get("Name").map(|name| name.contains("Brien")) == Some(true)),
            "带引号的事实应可查询（不崩、不丢失）：{:?}",
            results
        );
        let relations = engine.query("oxigraph_relation(From, Type, To)");
        assert!(
            relations.iter().any(|result| {
                result.get("From").map(|from| from.contains("Brien")) == Some(true)
                    && result.get("To").map(|to| to.contains("slash")) == Some(true)
            }),
            "带反斜杠的事实应可查询（不崩、不丢失）：{:?}",
            relations
        );
    }
}
// 逻辑查询计划编译器（Query Plan Compiler）
//
// 符号主义的"语义理解"：不把问题映射到稠密向量，而是把中文问句编译成
// 显式的逻辑查询计划（可打印、可检查、可执行）：
//
//   张三的师傅的师傅是谁   → FindObject { segments: [张三, 师傅, 师傅] }
//   李四的师傅是王五吗     → CheckTriple { subject_segments: [李四, 师傅], object: 王五 }
//   李四有几个师傅         → Count { subject: 李四, predicate: 师傅 }
//   李四有师傅吗           → Exist { subject: 李四, predicate: 师傅 }
//   谁的师傅是李四         → FindSubject { object: 李四, predicate: 师傅 }
//   谁和张三最像           → Similar { entity: 张三 }
//   张三对应李四，王五对应谁 → Analogy { a: 张三, b: 李四, c: 王五 }
//
// 计划只做句法/结构的模式枚举，不接触知识库；实体/谓词的归一化与执行
// 由 SymbolicReasoner 结合本体对齐与链接预测完成。

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryPlan {
    // 沿"的"链查找对象：张三 的 师傅 的 师傅 是谁
    FindObject { segments: Vec<String> },
    // 反向查找主体：谁的 师傅 是 李四
    FindSubject { object: String, predicate: String },
    // 三值校验：主语链最后一个词是谓词，如 [张三, 师傅] + object=李四
    CheckTriple { subject_segments: Vec<String>, object: String },
    // 计数：李四 有几个 师傅
    Count { subject: String, predicate: String },
    // 存在性：李四 有没有 师傅
    Exist { subject: String, predicate: String },
    // 最近邻语义检索：谁和 张三 最像
    Similar { entity: String },
    // 向量算术类比：a:b :: c:?
    Analogy { a: String, b: String, c: String },
}

pub struct QueryPlanner {
    max_question_chars: usize,
    max_chain: usize,
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanner {
    pub fn new() -> Self {
        Self {
            max_question_chars: 80,
            max_chain: 6,
        }
    }

    pub fn parse(&self, question: &str) -> Option<QueryPlan> {
        let raw = Self::clean(question)?;
        if raw.chars().count() > self.max_question_chars {
            return None;
        }

        // 存在句依赖句尾语气词（"有…吗"），必须在去语气词之前判断
        if let Some(plan) = self.parse_exist(&raw) {
            return Some(plan);
        }
        let q = Self::strip_particles(&raw);

        if let Some(plan) = self.parse_analogy(&q) {
            return Some(plan);
        }
        if let Some(plan) = self.parse_similar(&q) {
            return Some(plan);
        }
        if let Some(plan) = self.parse_count(&q) {
            return Some(plan);
        }
        if let Some(plan) = self.parse_find_subject(&q) {
            return Some(plan);
        }

        let (core, had_tail) = Self::strip_question_tail(&q);
        if had_tail {
            return self.build_find_object(core);
        }

        if let Some(plan) = self.parse_check_triple(&q) {
            return Some(plan);
        }

        if q.contains('的') {
            return self.build_find_object(&q);
        }
        None
    }

    pub fn format_plan(plan: &QueryPlan) -> String {
        match plan {
            QueryPlan::FindObject { segments } => {
                format!("查找对象：{}", segments.join(" 的 "))
            }
            QueryPlan::FindSubject { object, predicate } => {
                format!("反查主体：谁的 {} 是 {}", predicate, object)
            }
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => format!(
                "校验事实：{} 的 {} 是 {}",
                subject_segments[..subject_segments.len().saturating_sub(1)].join(" 的 "),
                subject_segments.last().cloned().unwrap_or_default(),
                object
            ),
            QueryPlan::Count { subject, predicate } => {
                format!("计数：{} 的 {} 个数", subject, predicate)
            }
            QueryPlan::Exist { subject, predicate } => {
                format!("存在性：{} 是否有 {}", subject, predicate)
            }
            QueryPlan::Similar { entity } => format!("语义近邻：与 {} 最像的实体", entity),
            QueryPlan::Analogy { a, b, c } => {
                format!("类比：{} : {} :: {} : ?", a, b, c)
            }
        }
    }

    fn clean(question: &str) -> Option<String> {
        let mut text = question.trim().to_string();
        for prefix in [
            "请问",
            "请告诉我一下",
            "请告诉我",
            "我想知道",
            "我想问一下",
            "帮我查一下",
            "帮我查查",
            "麻烦问一下",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim().to_string();
            }
        }
        let trimmed = text
            .trim_end_matches(|c: char| {
                matches!(
                    c,
                    '？' | '?' | '。' | '！' | '!' | '，' | ',' | '；' | ';' | '~' | '、' | ' '
                )
            })
            .trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(trimmed.to_string())
    }

    fn strip_particles(text: &str) -> String {
        text.trim_end_matches(|c: char| {
            matches!(
                c,
                '吗' | '呢' | '吧' | '啊' | '呀' | '？' | '?' | '。' | '！' | '!' | ' '
            )
        })
        .trim()
        .to_string()
    }

    fn strip_question_tail(q: &str) -> (&str, bool) {
        const TAILS: [&str; 9] = [
            "是什么人",
            "是哪一位",
            "是哪一个",
            "是什么",
            "是哪个",
            "是哪位",
            "是哪里",
            "是谁",
            "为谁",
        ];
        for tail in TAILS {
            if let Some(core) = q.strip_suffix(tail) {
                return (core.trim_end_matches('的'), true);
            }
        }
        for tail in ["谁", "什么", "哪个", "啥", "何人"] {
            if let Some(core) = q.strip_suffix(tail) {
                return (core.trim_end_matches('的'), true);
            }
        }
        (q, false)
    }

    fn segments_of(core: &str) -> Vec<String> {
        core.split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect()
    }

    fn build_find_object(&self, core: &str) -> Option<QueryPlan> {
        let segments = Self::segments_of(core);
        if segments.len() < 2 || segments.len() > self.max_chain {
            return None;
        }
        Some(QueryPlan::FindObject { segments })
    }

    fn parse_check_triple(&self, q: &str) -> Option<QueryPlan> {
        if let Some((left, right)) = q.split_once("是不是") {
            // 形式一：A 是不是 B 的 C  → 校验 B 的 C 是否为 A
            // 形式二：A 的 B 是不是 C  → 校验 A 的 B 是否为 C
            let left = left.trim();
            let right = right.trim();
            let (subject_segments, object) = if right.contains('的') {
                (Self::segments_of(right), left)
            } else if left.contains('的') {
                (Self::segments_of(left), right)
            } else {
                return None;
            };
            if subject_segments.len() >= 2 && !object.is_empty() && !Self::is_question_word(object) {
                return Some(QueryPlan::CheckTriple {
                    subject_segments,
                    object: object.to_string(),
                });
            }
            return None;
        }
        if let Some((left, right)) = q.rsplit_once('是') {
            let left = left.trim();
            let object = right.trim();
            if !left.contains('的') || object.is_empty() || Self::is_question_word(object) {
                return None;
            }
            let subject_segments = Self::segments_of(left);
            if subject_segments.len() < 2 || subject_segments.len() > self.max_chain {
                return None;
            }
            return Some(QueryPlan::CheckTriple {
                subject_segments,
                object: object.to_string(),
            });
        }
        None
    }

    fn is_question_word(text: &str) -> bool {
        [
            "谁", "什么", "哪个", "哪位", "哪里", "啥", "何人", "几点", "多少", "几",
        ]
        .iter()
        .any(|word| text.contains(word))
    }

    fn parse_find_subject(&self, q: &str) -> Option<QueryPlan> {
        let rest = q.strip_prefix("谁的").or_else(|| q.strip_prefix("哪位的"))?;
        let (predicate, object) = rest.split_once('是')?;
        let predicate = predicate.trim();
        let object = object.trim();
        if predicate.is_empty() || object.is_empty() || Self::is_question_word(object) {
            return None;
        }
        Some(QueryPlan::FindSubject {
            object: object.to_string(),
            predicate: predicate.to_string(),
        })
    }

    fn parse_count(&self, q: &str) -> Option<QueryPlan> {
        // 「有几个/几位/几名…」：量词可省略（有几个），也可跟上其它量词（有几位）
        if let Some((head, rest)) = q.split_once("有几") {
            let stripped = rest.trim_start_matches(|c: char| {
                matches!(c, '个' | '位' | '条' | '种' | '名' | '只')
            });
            if stripped.len() != rest.len() {
                if let Some(predicate) = Self::strip_measure(stripped) {
                    if !head.is_empty() && !predicate.is_empty() {
                        return Some(QueryPlan::Count {
                            subject: head.to_string(),
                            predicate,
                        });
                    }
                }
            }
        }
        if let Some((head, rest)) = q.split_once("有多少") {
            let rest = rest.strip_prefix('个').unwrap_or(rest);
            if let Some(predicate) = Self::strip_measure(rest) {
                if !head.is_empty() && !predicate.is_empty() {
                    return Some(QueryPlan::Count {
                        subject: head.to_string(),
                        predicate,
                    });
                }
            }
        }
        if let Some((subject, tail)) = q.split_once('的') {
            if let Some((predicate, _)) = tail
                .split_once("有几个")
                .or_else(|| tail.split_once("有多少个"))
            {
                if !subject.is_empty() && !predicate.is_empty() {
                    return Some(QueryPlan::Count {
                        subject: subject.to_string(),
                        predicate: predicate.to_string(),
                    });
                }
            }
        }
        None
    }

    fn strip_measure(text: &str) -> Option<String> {
        let mut rest = Self::strip_particles(text);
        for measure in ["个", "位", "条", "种", "名", "只"] {
            if let Some(stripped) = rest.strip_suffix(measure) {
                rest = stripped.trim().to_string();
            }
        }
        if rest.is_empty() {
            None
        } else {
            Some(rest)
        }
    }

    fn parse_exist(&self, q: &str) -> Option<QueryPlan> {
        if let Some((subject, predicate)) = q.split_once("有没有") {
            if let Some(predicate) = Self::strip_measure(predicate) {
                if !subject.trim().is_empty() && !predicate.is_empty() {
                    return Some(QueryPlan::Exist {
                        subject: subject.to_string(),
                        predicate,
                    });
                }
            }
        }
        if let Some((subject, tail)) = q.split_once("有") {
            // 计数问句（有几个/有多少）交给 Count 分支
            if tail.starts_with('几') || tail.starts_with("多少") {
                return None;
            }
            let tail = tail.trim_end_matches(|c: char| {
                matches!(c, '吗' | '么' | '嘛' | '吧' | '啊' | '？' | '?')
            });
            if tail.is_empty() || subject.trim().is_empty() {
                return None;
            }
            if let Some(predicate) = Self::strip_measure(tail) {
                if !predicate.is_empty() {
                    return Some(QueryPlan::Exist {
                        subject: subject.to_string(),
                        predicate,
                    });
                }
            }
        }
        None
    }

    fn parse_similar(&self, q: &str) -> Option<QueryPlan> {
        // 「谁和张三最像」中问题词在左侧，真正要检索的实体在右侧；
        // 「张三和谁最像」则相反，选非问题词的一侧作为实体。
        let resolve = |left: &str, entity: &str| -> Option<QueryPlan> {
            let entity = entity.trim();
            let candidate = if Self::is_question_word(entity) {
                left.trim()
            } else {
                entity
            };
            if candidate.is_empty() || Self::is_question_word(candidate) {
                None
            } else {
                Some(QueryPlan::Similar {
                    entity: candidate.to_string(),
                })
            }
        };
        if let Some((left, rest)) = q.split_once("和") {
            if let Some((entity, _)) = rest.split_once("最像") {
                if let Some(plan) = resolve(left, entity) {
                    return Some(plan);
                }
            }
            if let Some((entity, _)) = rest.split_once("相似") {
                if let Some(plan) = resolve(left, entity) {
                    return Some(plan);
                }
            }
        }
        if let Some((left, rest)) = q.split_once("跟") {
            if let Some((entity, _)) = rest.split_once("最像") {
                if let Some(plan) = resolve(left, entity) {
                    return Some(plan);
                }
            }
        }
        if let Some((left, rest)) = q.split_once("与") {
            if let Some((entity, _)) = rest.split_once("相似") {
                if let Some(plan) = resolve(left, entity) {
                    return Some(plan);
                }
            }
        }
        if q.ends_with("最像谁") {
            let entity = q
                .trim_end_matches("最像谁")
                .trim_end_matches('和')
                .trim_end_matches('跟')
                .trim_end_matches('与')
                .trim();
            if !entity.is_empty() && !Self::is_question_word(entity) {
                return Some(QueryPlan::Similar {
                    entity: entity.to_string(),
                });
            }
        }
        None
    }

    fn parse_analogy(&self, q: &str) -> Option<QueryPlan> {
        for separator in ['，', ',', '；', ';'] {
            if let Some((left, right)) = q.split_once(separator) {
                if let Some((a, b)) = left.split_once("对应") {
                    if let Some(c) = right.strip_suffix("对应谁") {
                        let (a, b, c) = (a.trim(), b.trim(), c.trim());
                        if !a.is_empty() && !b.is_empty() && !c.is_empty() {
                            return Some(QueryPlan::Analogy {
                                a: a.to_string(),
                                b: b.to_string(),
                                c: c.to_string(),
                            });
                        }
                    }
                }
                if let Some((a, b)) = left.split_once("之于") {
                    // 与「对应」句式一致：去掉右侧问句尾巴「对应谁/是谁」
                    let c = right
                        .strip_suffix("对应谁")
                        .or_else(|| right.strip_suffix("是谁"))
                        .unwrap_or(right)
                        .trim();
                    if !a.trim().is_empty() && !b.trim().is_empty() && !c.is_empty() {
                        return Some(QueryPlan::Analogy {
                            a: a.trim().to_string(),
                            b: b.trim().to_string(),
                            c: c.to_string(),
                        });
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planner() -> QueryPlanner {
        QueryPlanner::new()
    }

    #[test]
    fn test_find_object_chain() {
        let plan = planner().parse("张三的师傅的师傅是谁").expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["张三".to_string(), "师傅".to_string(), "师傅".to_string()]
            }
        );
    }

    #[test]
    fn test_find_object_with_polite_prefix_and_particles() {
        let plan = planner().parse("请问李四的师祖是什么呢？").expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师祖".to_string()]
            }
        );
    }

    #[test]
    fn test_find_subject_pattern() {
        let plan = planner().parse("谁的师傅是李四").expect("应编译为反查主体");
        assert_eq!(
            plan,
            QueryPlan::FindSubject {
                object: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
    }

    #[test]
    fn test_count_question() {
        let plan = planner().parse("李四有几个师傅").expect("应编译为计数");
        assert_eq!(
            plan,
            QueryPlan::Count {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
        let plan2 = planner().parse("苹果公司有多少员工").expect("应编译为计数");
        assert_eq!(
            plan2,
            QueryPlan::Count {
                subject: "苹果公司".to_string(),
                predicate: "员工".to_string()
            }
        );
    }

    #[test]
    fn test_exist_question() {
        let plan = planner().parse("李四有没有师傅").expect("应编译为存在性");
        assert_eq!(
            plan,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
        let plan2 = planner().parse("李四有师傅吗").expect("应编译为存在性");
        assert_eq!(
            plan2,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "师傅".to_string()
            }
        );
    }

    #[test]
    fn test_check_triple_yes_no() {
        let plan = planner().parse("李四的师傅是王五吗").expect("应编译为事实校验");
        assert_eq!(
            plan,
            QueryPlan::CheckTriple {
                subject_segments: vec!["李四".to_string(), "师傅".to_string()],
                object: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_check_triple_reversed() {
        let plan = planner().parse("王五是不是李四的师傅").expect("应编译为事实校验");
        assert_eq!(
            plan,
            QueryPlan::CheckTriple {
                subject_segments: vec!["李四".to_string(), "师傅".to_string()],
                object: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_similar_question() {
        let plan = planner().parse("谁和张三最像").expect("应编译为近邻检索");
        assert_eq!(
            plan,
            QueryPlan::Similar {
                entity: "张三".to_string()
            }
        );
        let plan2 = planner().parse("和李四最像的实体是谁？").expect("应编译为近邻检索");
        assert_eq!(
            plan2,
            QueryPlan::Similar {
                entity: "李四".to_string()
            }
        );
    }

    #[test]
    fn test_analogy_question() {
        let plan = planner().parse("张三对应李四，王五对应谁").expect("应编译为类比");
        assert_eq!(
            plan,
            QueryPlan::Analogy {
                a: "张三".to_string(),
                b: "李四".to_string(),
                c: "王五".to_string()
            }
        );
        let plan2 = planner().parse("张三之于李四，王五呢").expect("应编译为类比");
        assert_eq!(
            plan2,
            QueryPlan::Analogy {
                a: "张三".to_string(),
                b: "李四".to_string(),
                c: "王五".to_string()
            }
        );
    }

    #[test]
    fn test_topic_without_question_word_is_find_object() {
        let plan = planner().parse("李四的师祖").expect("省略问句也应给出查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师祖".to_string()]
            }
        );
    }

    #[test]
    fn test_unrelated_question_returns_none() {
        assert!(planner().parse("今天天气怎么样").is_none());
        assert!(planner().parse("帮我讲个笑话").is_none());
    }

    #[test]
    fn test_bare_entity_question_returns_none() {
        assert!(planner().parse("李四是谁").is_none());
        assert!(planner().parse("什么是知识图谱").is_none());
    }

    #[test]
    fn test_plan_formats_are_printable() {
        let plan = planner().parse("张三的师傅的师傅是谁").unwrap();
        let text = QueryPlanner::format_plan(&plan);
        assert!(text.contains("张三"));
        assert!(text.contains("师傅"));
    }
}

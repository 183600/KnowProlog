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

use crate::pattern_closure::ClosureOp;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryPlan {
    // 沿"的"链查找对象：张三 的 师傅 的 师傅 是谁
    FindObject { segments: Vec<String> },
    // 反向查找主体：谁的 师傅 是 李四
    FindSubject { object: String, predicate: String },
    // 两点之间的路径：张三和王五之间是什么关系
    PathBetween {
        start: String,
        goal: String,
        max_hops: usize,
    },
    // 共同邻居：张三和李四共同认识谁
    CommonNeighbor { left: String, right: String },
    // 路径上的中间人：张三通过谁认识王五
    Intermediary {
        start: String,
        goal: String,
        max_hops: usize,
    },
    // 三值校验：主语链最后一个词是谓词，如 [张三, 师傅] + object=李四
    CheckTriple { subject_segments: Vec<String>, object: String },
    // 计数：李四 有几个 师傅
    Count { subject: String, predicate: String },
    // 存在性：李四 有没有 师傅
    Exist { subject: String, predicate: String },
    // 闭包：李四 的 所有 上司（Plus 需要已启用的传递律）
    Closure {
        subject: String,
        predicate: String,
        operator: ClosureOp,
        count_only: bool,
        then: Vec<String>,
    },
    // 最近邻语义检索：谁和 张三 最像
    Similar { entity: String },
    // 向量算术类比：a:b :: c:?
    Analogy { a: String, b: String, c: String },
}

#[derive(Clone)]
pub struct QueryPlanner {
    max_question_chars: usize,
    max_chain: usize,
    max_path_hops: usize,
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
            max_path_hops: 4,
        }
    }

    pub fn parse(&self, question: &str) -> Option<QueryPlan> {
        let raw = Self::clean(question)?;
        if raw.chars().count() > self.max_question_chars {
            return None;
        }

        let path_question = Self::strip_particles(&raw);
        if let Some(plan) = self.parse_path_question(&path_question) {
            return Some(plan);
        }

        // 「所有」必须先于存在句解析，否则会被「有」误判成存在性问句。
        if let Some(plan) = self.parse_closure(&raw) {
            return Some(plan);
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

    pub fn kind(plan: &QueryPlan) -> &'static str {
        match plan {
            QueryPlan::FindObject { .. } => "FindObject",
            QueryPlan::FindSubject { .. } => "FindSubject",
            QueryPlan::PathBetween { .. } => "PathBetween",
            QueryPlan::CommonNeighbor { .. } => "CommonNeighbor",
            QueryPlan::Intermediary { .. } => "Intermediary",
            QueryPlan::CheckTriple { .. } => "CheckTriple",
            QueryPlan::Count { .. } => "Count",
            QueryPlan::Exist { .. } => "Exist",
            QueryPlan::Closure { .. } => "Closure",
            QueryPlan::Similar { .. } => "Similar",
            QueryPlan::Analogy { .. } => "Analogy",
        }
    }

    pub fn format_plan(plan: &QueryPlan) -> String {
        match plan {
            QueryPlan::FindObject { segments } => {
                format!("查找对象：{}", segments.join(" 的 "))
            }
            QueryPlan::FindSubject { object, predicate } => {
                format!("反查主体：谁的 {} 是 {}", predicate, object)
            }
            QueryPlan::PathBetween {
                start,
                goal,
                max_hops,
            } => format!("路径枚举：{} → {}（最多 {} 跳）", start, goal, max_hops),
            QueryPlan::CommonNeighbor { left, right } => {
                format!("共同邻居：{} 和 {} 共同指向的实体", left, right)
            }
            QueryPlan::Intermediary {
                start,
                goal,
                max_hops,
            } => format!(
                "中间人：{} → ? → {}（最多 {} 跳）",
                start, goal, max_hops
            ),
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
            QueryPlan::Closure {
                subject,
                predicate,
                operator,
                count_only,
                then,
            } => {
                let view = if *count_only { "计数" } else { "列出" };
                let chain = if then.is_empty() {
                    predicate.clone()
                } else {
                    format!("{} 的 {}", predicate, then.join(" 的 "))
                };
                format!(
                    "{}闭包：{} 的所有 {}（{}）",
                    view,
                    subject,
                    chain,
                    operator.label()
                )
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
        const TAILS: [&str; 11] = [
            "是什么人",
            "是哪一位",
            "是哪一个",
            "是多少",
            "有多少",
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

    fn clean_name(text: &str) -> String {
        text.trim()
            .trim_end_matches(|c: char| {
                matches!(
                    c,
                    ' ' | '的' | '是' | '，' | ',' | '。' | '?' | '？' | '！' | '!' | '、'
                )
            })
            .trim()
            .to_string()
    }

    fn valid_path_name(name: &str) -> bool {
        !name.is_empty()
            && name.chars().count() <= 40
            && !name.contains('的')
            && !Self::is_question_word(name)
            && !["怎么", "如何", "怎样", "关系", "路径", "路线", "中间人", "连接"]
                .iter()
                .any(|word| name.contains(word))
    }

    fn split_pair(text: &str) -> Option<(String, String)> {
        for connector in ['和', '跟', '与'] {
            if let Some((left, right)) = text.split_once(connector) {
                let left = Self::clean_name(left);
                let right = Self::clean_name(right);
                if Self::valid_path_name(&left) && Self::valid_path_name(&right) {
                    return Some((left, right));
                }
            }
        }
        None
    }

    fn cut_at_marker(text: &str, markers: &[&str]) -> String {
        let mut earliest: Option<usize> = None;
        for marker in markers {
            if let Some(position) = text.find(marker) {
                if earliest.is_none_or(|current| position < current) {
                    earliest = Some(position);
                }
            }
        }
        let cut = earliest.unwrap_or(text.len());
        Self::clean_name(&text[..cut])
    }

    fn strip_relation_prefix(text: &str) -> String {
        let mut value = text.trim().to_string();
        for _ in 0..3 {
            let mut changed = false;
            for prefix in [
                "认识", "结识", "找到", "联系到", "联系", "见到", "是", "和", "与", "跟",
            ] {
                if let Some(rest) = value.strip_prefix(prefix) {
                    value = rest.trim().to_string();
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for suffix in ["联系", "认识", "相连", "连接", "吗", "呢"] {
            if let Some(rest) = value.strip_suffix(suffix) {
                value = rest.trim().to_string();
            }
        }
        value
    }

    fn parse_path_question(&self, q: &str) -> Option<QueryPlan> {
        if let Some(plan) = self.parse_common_neighbor(q) {
            return Some(plan);
        }
        if let Some(plan) = self.parse_intermediary(q) {
            return Some(plan);
        }
        self.parse_path_between(q)
    }

    fn parse_common_neighbor(&self, q: &str) -> Option<QueryPlan> {
        const MARKERS: [&str; 7] = [
            "共同认识",
            "共同好友",
            "共同的朋友",
            "共同朋友",
            "共同关注",
            "都认识",
            "同时认识",
        ];
        for marker in MARKERS {
            let Some(position) = q.find(marker) else {
                continue;
            };
            let before = Self::clean_name(&q[..position]);
            let after = Self::clean_name(&q[position + marker.len()..]);
            if let Some((left, right)) = Self::split_pair(&before) {
                return Some(QueryPlan::CommonNeighbor { left, right });
            }
            if let Some((left, right)) = Self::split_pair(&after) {
                return Some(QueryPlan::CommonNeighbor { left, right });
            }
        }
        None
    }

    fn parse_intermediary(&self, q: &str) -> Option<QueryPlan> {
        if let Some(position) = q.find("通过谁") {
            let start = Self::clean_name(&q[..position]);
            let goal = Self::strip_relation_prefix(&q[position + "通过谁".len()..]);
            if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                return Some(QueryPlan::Intermediary {
                    start,
                    goal,
                    max_hops: self.max_path_hops,
                });
            }
        }
        if let Some(rest) = q.strip_prefix("谁把") {
            for marker in ["连接起来", "连起来", "联系起来", "连接", "联系"] {
                if let Some((pair, _)) = rest.split_once(marker) {
                    if let Some((start, goal)) = Self::split_pair(pair) {
                        return Some(QueryPlan::Intermediary {
                            start,
                            goal,
                            max_hops: self.max_path_hops,
                        });
                    }
                }
            }
        }
        if let Some(position) = q.find("中间人") {
            let before = &q[..position];
            let pair_text = before
                .find("之间")
                .map(|split| &before[..split])
                .unwrap_or(before);
            if let Some((start, goal)) = Self::split_pair(pair_text) {
                return Some(QueryPlan::Intermediary {
                    start,
                    goal,
                    max_hops: self.max_path_hops,
                });
            }
        }
        None
    }

    fn parse_path_between(&self, q: &str) -> Option<QueryPlan> {
        const MARKERS: [&str; 11] = [
            "之间",
            "什么关系",
            "有什么关系",
            "的关系",
            "怎么",
            "如何",
            "怎样",
            "路径",
            "路线",
            "连接起来",
            "连起来",
        ];
        if !MARKERS.iter().any(|marker| q.contains(marker)) {
            return None;
        }
        for connector in ['和', '跟', '与'] {
            let Some((left, right)) = q.split_once(connector) else {
                continue;
            };
            let start = Self::clean_name(left);
            let goal = Self::cut_at_marker(right, &MARKERS);
            if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                return Some(QueryPlan::PathBetween {
                    start,
                    goal,
                    max_hops: self.max_path_hops,
                });
            }
        }
        for marker in ["怎么找到", "如何找到", "怎么认识", "如何认识", "怎么联系", "如何联系"] {
            if let Some((start, rest)) = q.split_once(marker) {
                let start = Self::clean_name(start);
                let goal = Self::strip_relation_prefix(rest);
                if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                    return Some(QueryPlan::PathBetween {
                        start,
                        goal,
                        max_hops: self.max_path_hops,
                    });
                }
            }
        }
        for marker in ["怎么到", "如何到", "怎样到"] {
            if let Some((start, rest)) = q.split_once(marker) {
                let start = Self::clean_name(start);
                let goal = Self::strip_relation_prefix(rest);
                if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                    return Some(QueryPlan::PathBetween {
                        start,
                        goal,
                        max_hops: self.max_path_hops,
                    });
                }
            }
        }
        if let Some(rest) = q.strip_prefix("从") {
            if let Some((start, right)) = rest.split_once('到') {
                let start = Self::clean_name(start);
                let goal = Self::cut_at_marker(right, &MARKERS);
                if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                    return Some(QueryPlan::PathBetween {
                        start,
                        goal,
                        max_hops: self.max_path_hops,
                    });
                }
            }
        }
        if !q.contains("找到") && !q.contains("认识到") {
            if let Some((start, right)) = q.split_once('到') {
                let start = Self::clean_name(start);
                let goal = Self::cut_at_marker(right, &MARKERS);
                if Self::valid_path_name(&start) && Self::valid_path_name(&goal) {
                    return Some(QueryPlan::PathBetween {
                        start,
                        goal,
                        max_hops: self.max_path_hops,
                    });
                }
            }
        }
        None
    }

    fn parse_closure(&self, raw: &str) -> Option<QueryPlan> {
        for marker in ["的所有", "的全部", "的任意"] {
            let Some((subject, rest)) = raw.split_once(marker) else {
                continue;
            };
            let subject = subject.trim();
            if subject.is_empty() || subject.contains('的') || Self::is_question_word(subject) {
                continue;
            }
            let (chain, count_only) = Self::strip_closure_tail(rest);
            let parts: Vec<&str> = chain
                .split('的')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect();
            if parts.is_empty() || parts.len() > 6 {
                continue;
            }
            if parts.iter().any(|part| {
                part.chars().count() > 40 || Self::is_question_word(part)
            }) {
                continue;
            }
            let predicate = parts[0].trim_end_matches('的').trim();
            if predicate.is_empty() {
                continue;
            }
            return Some(QueryPlan::Closure {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                operator: ClosureOp::Plus,
                count_only,
                then: parts[1..].iter().map(|part| part.to_string()).collect(),
            });
        }
        None
    }

    fn strip_closure_tail(raw: &str) -> (String, bool) {
        let text = Self::strip_particles(raw);
        const TAILS: [(&str, bool); 30] = [
            ("分别有哪些", false),
            ("分别是什么", false),
            ("分别是谁", false),
            ("一共有多少个", true),
            ("一共有几位", true),
            ("一共有几个", true),
            ("一共有多少", true),
            ("共有多少个", true),
            ("共有几位", true),
            ("共有几个", true),
            ("共有多少", true),
            ("总共多少个", true),
            ("总共几位", true),
            ("总共几个", true),
            ("总共多少", true),
            ("有哪几个", false),
            ("有哪几位", false),
            ("有哪些", false),
            ("都是什么", false),
            ("都是谁", false),
            ("都有谁", false),
            ("是什么", false),
            ("是谁", false),
            ("有多少个", true),
            ("有几位", true),
            ("有几个", true),
            ("有多少", true),
            ("的总数", true),
            ("有谁", false),
            ("分别", false),
        ];
        for (tail, count_only) in TAILS {
            if let Some(core) = text.strip_suffix(tail) {
                return (core.trim().to_string(), count_only);
            }
        }
        (text, false)
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
    fn test_find_object_with_amount_tail() {
        let plan = planner().parse("姚明的身高是多少").expect("应编译为查找计划");
        assert_eq!(
            plan,
            QueryPlan::FindObject {
                segments: vec!["姚明".to_string(), "身高".to_string()]
            }
        );
        let plan2 = planner().parse("李四的师傅有多少").expect("应编译为查找计划");
        assert_eq!(
            plan2,
            QueryPlan::FindObject {
                segments: vec!["李四".to_string(), "师傅".to_string()]
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
    fn test_closure_question() {
        let plan = planner().parse("李四的所有上司是谁").expect("应编译为闭包计划");
        assert_eq!(
            plan,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let list = planner()
            .parse("李四的全部上司分别是谁？")
            .expect("应编译为闭包计划");
        assert_eq!(
            list,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: Vec::new(),
            }
        );
        let count = planner()
            .parse("李四的所有上司一共有几个")
            .expect("应编译为闭包计数");
        assert_eq!(
            count,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: true,
                then: Vec::new(),
            }
        );
        let composed = planner()
            .parse("李四的所有上司的师傅是谁")
            .expect("应编译为闭包+后续关系");
        assert_eq!(
            composed,
            QueryPlan::Closure {
                subject: "李四".to_string(),
                predicate: "上司".to_string(),
                operator: ClosureOp::Plus,
                count_only: false,
                then: vec!["师傅".to_string()],
            }
        );
    }

    #[test]
    fn test_closure_does_not_swallow_exist_or_count() {
        let exist = planner().parse("李四有没有上司").expect("存在性问句");
        assert_eq!(
            exist,
            QueryPlan::Exist {
                subject: "李四".to_string(),
                predicate: "上司".to_string()
            }
        );
        let count = planner().parse("李四有几个上司").expect("直接计数");
        assert_eq!(
            count,
            QueryPlan::Count {
                subject: "李四".to_string(),
                predicate: "上司".to_string()
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
    fn test_path_between_question() {
        let plan = planner()
            .parse("张三和王五之间是什么关系")
            .expect("两点之间的问句应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("张三和李四什么关系")
            .expect("省略是字的问句也应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "李四".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("张三到王五怎么走")
            .expect("到字问句应编译为路径计划");
        assert_eq!(
            plan,
            QueryPlan::PathBetween {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
    }

    #[test]
    fn test_common_neighbor_question() {
        let plan = planner()
            .parse("张三和李四共同认识谁")
            .expect("共同邻居问句应被识别");
        assert_eq!(
            plan,
            QueryPlan::CommonNeighbor {
                left: "张三".to_string(),
                right: "李四".to_string(),
            }
        );
        let plan = planner()
            .parse("谁同时认识张三和李四")
            .expect("问题词在前也应识别共同邻居");
        assert_eq!(
            plan,
            QueryPlan::CommonNeighbor {
                left: "张三".to_string(),
                right: "李四".to_string(),
            }
        );
    }

    #[test]
    fn test_intermediary_question() {
        let plan = planner()
            .parse("张三通过谁认识王五")
            .expect("通过谁问句应编译为中间人计划");
        assert_eq!(
            plan,
            QueryPlan::Intermediary {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
        let plan = planner()
            .parse("谁把张三和王五连接起来")
            .expect("连接句应识别中间人");
        assert_eq!(
            plan,
            QueryPlan::Intermediary {
                start: "张三".to_string(),
                goal: "王五".to_string(),
                max_hops: 4,
            }
        );
    }

    #[test]
    fn test_path_questions_do_not_swallow_existing_plans() {
        assert!(matches!(
            planner().parse("李四有没有师傅"),
            Some(QueryPlan::Exist { .. })
        ));
        assert!(matches!(
            planner().parse("张三和李四是什么关系"),
            Some(QueryPlan::PathBetween { .. })
        ));
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

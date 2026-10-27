impl ForestEvaluation {
    pub fn answer(&self) -> Option<&QueryOutcome> {
        match self.decision {
            ForestDecision::Answer { index } => self.grounded.get(index).map(|g| &g.outcome),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// 组合语义解析器
// ---------------------------------------------------------------------------
const QUESTION_WORDS: [&str; 12] = [
    "谁",
    "什么",
    "多少",
    "几",
    "哪",
    "哪个",
    "哪些",
    "哪里",
    "何人",
    "怎么",
    "为什么",
    "如何",
];
const ASSERTION_MARKERS: [&str; 8] = ["是", "为", "不", "没", "所有", "每个", "并且", "或者"];
const DET_ALL: [&str; 6] = ["所有", "每个", "每一个", "任何", "全部", "一切"];
const DET_SOME: [&str; 5] = ["有些", "有的", "某些", "某个", "存在"];
const CONJ_AND: [&str; 4] = ["并且", "而且", "且", "同时"];
const CONJ_OR: [&str; 3] = ["或者", "或", "要么"];
const COPULA_CONNECTORS: [&str; 5] = ["是", "为", "属于", "不是", "并非"];
const PARTICLES: [&str; 4] = ["都", "也", "而", "乃"];
const QUESTION_SUFFIX: [&str; 2] = ["吗", "呢"];
const MAX_ITEMS_PER_CELL: usize = 8;
#[derive(Clone)]
struct Item {
    term: Term,
    ty: Type,
    label: String,
}
fn push_item(cells: &mut [Vec<Vec<Item>>], i: usize, j: usize, item: Item, max_items: usize) {
    let cell = &mut cells[i][j];
    let key = format!("{}:{}", item.ty.render(), item.label);
    if cell
        .iter()
        .any(|existing| format!("{}:{}", existing.ty.render(), existing.label) == key)
    {
        return;
    }
    if cell.len() < max_items {
        cell.push(item);
    }
}
pub struct FormalSemantics {
    lexicon: Lexicon,
    max_tokens: usize,
    max_items: usize,
}
impl Default for FormalSemantics {
    fn default() -> Self {
        Self::new()
    }
}
impl FormalSemantics {
    pub fn new() -> Self {
        Self {
            lexicon: Lexicon::default_chinese(),
            max_tokens: 24,
            max_items: MAX_ITEMS_PER_CELL,
        }
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    pub fn register_entity(&mut self, name: &str) {
        self.lexicon.register_entity(name);
    }

    pub fn register_class(&mut self, name: &str) {
        self.lexicon.register_class(name);
    }

    pub fn register_unary_predicate(&mut self, name: &str) {
        self.lexicon.register_unary_predicate(name);
    }

    pub fn register_binary_predicate(&mut self, name: &str) {
        self.lexicon.register_binary_predicate(name);
    }

    pub fn set_predicate_surfaces(&mut self, surfaces: &HashMap<String, String>) {
        self.lexicon.set_predicate_surfaces(surfaces);
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in shared_jieba().cut(text, false) {
            let word = word.trim();
            if word.is_empty() || is_punctuation(word) || PARTICLES.contains(&word) {
                continue;
            }
            tokens.push(word.to_string());
        }
        tokens
    }

    /// 归一化分词：合并被切开的「不 + 是」→「不是」，保证系词否定进入同一条系词规则。
    fn normalize_tokens(&self, text: &str) -> Vec<String> {
        let raw_tokens = self.tokenize(text);
        let mut tokens: Vec<String> = Vec::with_capacity(raw_tokens.len());
        let mut index = 0;
        while index < raw_tokens.len() {
            if raw_tokens[index] == "不"
                && index + 1 < raw_tokens.len()
                && raw_tokens[index + 1] == "是"
            {
                tokens.push("不是".to_string());
                index += 2;
            } else {
                tokens.push(raw_tokens[index].clone());
                index += 1;
            }
        }
        tokens
    }

    /// 解析中文句子为逻辑形式（问句优先走模式切分，陈述句走类型驱动图表分析）
    pub fn parse(&self, text: &str) -> LogicalForm {
        let source = text.trim().to_string();
        let tokens = self.normalize_tokens(&source);
        let mut trace = vec![format!("分词：{}", tokens.join(" / "))];
        if tokens.is_empty() {
            trace.push("空输入".to_string());
            return LogicalForm::unknown(source, tokens, trace);
        }
        if let Some(lf) = self.parse_conditional(&tokens, &source, &mut trace) {
            return lf;
        }
        let is_question = tokens
            .iter()
            .any(|token| QUESTION_WORDS.contains(&token.as_str()))
            || tokens
                .last()
                .map(|token| QUESTION_SUFFIX.contains(&token.as_str()))
                .unwrap_or(false)
            || source.ends_with('?')
            || source.ends_with('？');
        if is_question {
            if let Some(lf) = self.parse_question(&tokens, &source, &mut trace) {
                return lf;
            }
        }
        if tokens.len() <= self.max_tokens {
            let mut chart_trace = Vec::new();
            if let Some(item) = self.chart_parse(&tokens, &mut chart_trace) {
                if item.ty == Type::Truth {
                    trace.extend(chart_trace);
                    if !self.has_assertion_marker(&tokens) {
                        if let Some((entity, predicate)) = possessive_np(&item.term) {
                            trace.push(format!(
                                "无断言标记，解释为属性查询：{} 的 {}",
                                entity, predicate
                            ));
                            return LogicalForm {
                                source,
                                tokens,
                                kind: LfKind::Query(Query::Value { entity, predicate }),
                                trace,
                            };
                        }
                    }
                    return LogicalForm {
                        source,
                        tokens,
                        kind: LfKind::Statement(item.term),
                        trace,
                    };
                }
            }
        }
        trace.push("未能在当前片段内组合出类型为 t 的逻辑形式".to_string());
        LogicalForm::unknown(source, tokens, trace)
    }

    /// 解析出候选森林：条件句、问句模式与图表组合都作为独立读法保留。
    ///
    /// 与 `parse` 的差别只在「不压平」：`parse` 仍按旧优先级返回单条逻辑形式，
    /// 本方法把同一输入的其它类型合法读法也留下来，供模型接地与歧义判定。
    pub fn parse_forest(&self, text: &str) -> ParseForest {
        let source = text.trim().to_string();
        let tokens = self.normalize_tokens(&source);
        let mut forest = ParseForest {
            source: source.clone(),
            tokens: tokens.clone(),
            candidates: Vec::new(),
            trace: vec![format!("分词：{}", tokens.join(" / "))],
        };
        if tokens.is_empty() {
            forest.trace.push("空输入".to_string());
            return forest;
        }
        let mut seen: HashSet<String> = HashSet::new();
        let mut conditional_trace = forest.trace.clone();
        if let Some(lf) = self.parse_conditional(&tokens, &source, &mut conditional_trace) {
            push_forest_candidate(&mut forest, &mut seen, lf, ParseStrategy::Conditional, 0);
        }
        let is_question = tokens
            .iter()
            .any(|token| QUESTION_WORDS.contains(&token.as_str()))
            || tokens
                .last()
                .map(|token| QUESTION_SUFFIX.contains(&token.as_str()))
                .unwrap_or(false)
            || source.ends_with('?')
            || source.ends_with('？');
        if is_question {
            for (index, lf) in self
                .question_candidates(&tokens, &source)
                .into_iter()
                .enumerate()
            {
                push_forest_candidate(
                    &mut forest,
                    &mut seen,
                    lf,
                    ParseStrategy::QuestionPattern,
                    10 + index,
                );
            }
        }
        if tokens.len() <= self.max_tokens {
            let chart_items = self.chart_complete(&tokens);
            for (index, item) in chart_items.into_iter().enumerate() {
                if item.ty != Type::Truth {
                    continue;
                }
                let chart_label = format!("图表组合：{} : {}", item.label, item.ty.render());
                if !self.has_assertion_marker(&tokens) {
                    if let Some((entity, predicate)) = possessive_np(&item.term) {
                        let form = LogicalForm {
                            source: source.clone(),
                            tokens: tokens.clone(),
                            kind: LfKind::Query(Query::Value {
                                entity: entity.clone(),
                                predicate: predicate.clone(),
                            }),
                            trace: vec![
                                format!("分词：{}", tokens.join(" / ")),
                                chart_label.clone(),
                                format!(
                                    "无断言标记，也可解释为属性查询：{} 的 {}",
                                    entity, predicate
                                ),
                            ],
                        };
                        push_forest_candidate(
                            &mut forest,
                            &mut seen,
                            form,
                            ParseStrategy::Chart,
                            20 + index,
                        );
                    }
                }
                let form = LogicalForm {
                    source: source.clone(),
                    tokens: tokens.clone(),
                    kind: LfKind::Statement(item.term),
                    trace: vec![format!("分词：{}", tokens.join(" / ")), chart_label],
                };
                push_forest_candidate(
                    &mut forest,
                    &mut seen,
                    form,
                    ParseStrategy::Chart,
                    40 + index,
                );
            }
        }
        if forest.candidates.is_empty() {
            forest
                .trace
                .push("未能在当前片段内组合出可求值的逻辑形式".to_string());
        }
        forest
    }

    fn has_assertion_marker(&self, tokens: &[String]) -> bool {
        tokens
            .iter()
            .any(|token| ASSERTION_MARKERS.contains(&token.as_str()))
    }

    /// 「如果 P 那么/就/则 Q」→ P → Q
    fn parse_conditional(
        &self,
        tokens: &[String],
        source: &str,
        trace: &mut Vec<String>,
    ) -> Option<LogicalForm> {
        let start = tokens
            .iter()
            .position(|token| token == "如果" || token == "假如" || token == "若")?;
        let connect = tokens
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, token)| ["那么", "则", "就"].contains(&token.as_str()))
            .map(|(index, _)| index)?;
        let left_tokens = &tokens[start + 1..connect];
        let right_tokens: Vec<String> = tokens[connect + 1..]
            .iter()
            .filter(|token| !QUESTION_SUFFIX.contains(&token.as_str()))
            .cloned()
            .collect();
        let mut left_trace = Vec::new();
        let left = self.chart_parse(left_tokens, &mut left_trace)?;
        let mut right_trace = Vec::new();
        let right = self.chart_parse(&right_tokens, &mut right_trace)?;
        let left_term = saturate(left)?;
        let right_term = saturate(right)?;
        trace.push(format!(
            "条件句：{} → {}",
            left_term.render(),
            right_term.render()
        ));
        let term = Term::Implies(Box::new(left_term), Box::new(right_term));
        Some(LogicalForm {
            source: source.to_string(),
            tokens: tokens.to_vec(),
            kind: LfKind::Statement(term),
            trace: trace.clone(),
        })
    }

    fn parse_question(
        &self,
        tokens: &[String],
        source: &str,
        trace: &mut Vec<String>,
    ) -> Option<LogicalForm> {
        let first = self
            .question_candidates(tokens, source)
            .into_iter()
            .next()?;
        *trace = first.trace.clone();
        Some(first)
    }

    /// 问句模式的全部候选：按旧路径的优先级顺序生成，但不提前返回。
    fn question_candidates(&self, tokens: &[String], source: &str) -> Vec<LogicalForm> {
        let base_trace = vec![format!("分词：{}", tokens.join(" / "))];
        let mut out = Vec::new();
        let make = |kind: LfKind, extra: Vec<String>| {
            let mut trace = base_trace.clone();
            trace.extend(extra);
            LogicalForm {
                source: source.to_string(),
                tokens: tokens.to_vec(),
                kind,
                trace,
            }
        };
        // 1) 「…吗/呢」：去掉句尾疑问词后按陈述句解析，再作为真值查询
        if let Some(last) = tokens.last() {
            if QUESTION_SUFFIX.contains(&last.as_str()) && tokens.len() > 1 {
                let body_tokens = &tokens[..tokens.len() - 1];
                for item in self.chart_complete(body_tokens) {
                    if item.ty != Type::Truth {
                        continue;
                    }
                    out.push(make(
                        LfKind::Query(Query::Truth { body: item.term }),
                        vec![
                            format!("图表组合：{} : {}", item.label, item.ty.render()),
                            "句尾疑问词：转化为真值查询".to_string(),
                        ],
                    ));
                }
            }
        }
        // 2) 「是不是 / 是否」：E 是不是 P → Isa(E, P) 真值查询
        if let Some(index) = tokens
            .iter()
            .position(|token| token == "是不是" || token == "是否")
        {
            let entity = self.entity_from_tokens(&tokens[..index]);
            let predicate = self.predicate_from_tokens(&tokens[index + 1..]);
            if let (Some(entity), Some(predicate)) = (entity, predicate) {
                out.push(make(
                    LfKind::Query(Query::Truth {
                        body: Term::Isa(
                            Box::new(Term::Const(entity.clone())),
                            Box::new(Term::Const(predicate.clone())),
                        ),
                    }),
                    vec![format!("是不是句式：Isa({}, {})", entity, predicate)],
                ));
            }
        }
        // 3a) 「谁是 X 的 P」：属格值查询（问 X 的 P 是谁）
        if tokens.len() >= 5
            && ["谁", "哪些人", "哪些", "何人", "什么人", "哪个", "哪个人"]
                .contains(&tokens[0].as_str())
            && tokens.get(1).map(|token| token == "是").unwrap_or(false)
        {
            if let Some(de) = tokens.iter().position(|token| token == "的") {
                if de > 2 {
                    if let (Some(entity), Some(predicate)) = (
                        self.entity_from_tokens(&tokens[2..de]),
                        self.predicate_from_tokens(&tokens[de + 1..]),
                    ) {
                        if !predicate.is_empty() {
                            out.push(make(
                                LfKind::Query(Query::Value {
                                    entity: entity.clone(),
                                    predicate: predicate.clone(),
                                }),
                                vec![format!(
                                    "谁是 X 的 P 句式：查询「{}」的「{}」值",
                                    entity, predicate
                                )],
                            ));
                        }
                    }
                }
            }
        }
        // 3) 「谁的 P 是 O」：反向实体查询（保留关系方向，不把入边当出边）
        if tokens.len() >= 5
            && ["谁", "哪些人", "哪些", "何人", "什么人", "哪个", "哪个人"]
                .contains(&tokens[0].as_str())
            && tokens.get(1).map(|token| token == "的").unwrap_or(false)
        {
            if let Some(copula) = tokens
                .iter()
                .position(|token| token == "是" || token == "为")
            {
                if copula > 2 {
                    let mut object_tokens = &tokens[copula + 1..];
                    while object_tokens
                        .last()
                        .map(|token| {
                            QUESTION_SUFFIX.contains(&token.as_str())
                                || token == "?"
                                || token == "？"
                        })
                        .unwrap_or(false)
                    {
                        object_tokens = &object_tokens[..object_tokens.len() - 1];
                    }
                    if let (Some(predicate), Some(object)) = (
                        self.predicate_from_tokens(&tokens[2..copula]),
                        self.entity_from_tokens(object_tokens),
                    ) {
                        out.push(make(
                            LfKind::Query(Query::EntityByObject {
                                predicate: predicate.clone(),
                                object: object.clone(),
                            }),
                            vec![format!(
                                "谁的 P 是 O 句式：查询 {}(x, {}) 中的 x",
                                predicate, object
                            )],
                        ));
                    }
                }
            }
        }
        // 4) 「谁 / 哪些人 / 何人 …」：实体查询
        if let Some(first) = tokens.first() {
            if ["谁", "哪些人", "哪些", "何人", "什么人", "哪个"].contains(&first.as_str())
            {
                let mut rest = &tokens[1..];
                while rest
                    .first()
                    .map(|token| ["是", "为", "都", "的", "属于"].contains(&token.as_str()))
                    .unwrap_or(false)
                {
                    rest = &rest[1..];
                }
                if let Some(predicate) = self.predicate_from_tokens(rest) {
                    out.push(make(
                        LfKind::Query(Query::Entity {
                            predicate: predicate.clone(),
                        }),
                        vec![format!("谁是句式：查询满足「{}」的实体", predicate)],
                    ));
                }
            }
        }
        // 5) 「X 的 P …」：值查询
        if let Some(index) = tokens.iter().position(|token| token == "的") {
            if index > 0 && index + 1 < tokens.len() {
                if let Some(entity) = self.entity_from_tokens(&tokens[..index]) {
                    if let Some(predicate) = self.predicate_from_tokens(&tokens[index + 1..]) {
                        if !predicate.is_empty() {
                            out.push(make(
                                LfKind::Query(Query::Value {
                                    entity: entity.clone(),
                                    predicate: predicate.clone(),
                                }),
                                vec![format!(
                                    "的-属格句式：查询「{}」的「{}」值",
                                    entity, predicate
                                )],
                            ));
                        }
                    }
                }
            }
        }
        // 6) 「E P …」（无「的」）：按最长实体前缀切分
        if let Some((entity, predicate)) = self.split_entity_predicate(tokens) {
            out.push(make(
                LfKind::Query(Query::Value {
                    entity: entity.clone(),
                    predicate: predicate.clone(),
                }),
                vec![format!(
                    "实体-谓词切分：查询「{}」的「{}」值",
                    entity, predicate
                )],
            ));
        }
        out
    }

    fn entity_from_tokens(&self, tokens: &[String]) -> Option<String> {
        let filtered: Vec<String> = tokens
            .iter()
            .filter(|token| {
                !["那", "那么", "这个", "那个", "请问", "帮我", "告诉"].contains(&token.as_str())
            })
            .cloned()
            .collect();
        if filtered.is_empty() {
            return None;
        }
        let joined: String = filtered.concat();
        if let Some(entity) = self.lexicon.find_entity(&joined) {
            return Some(entity);
        }
        if !joined.is_empty() {
            Some(joined)
        } else {
            None
        }
    }

    fn predicate_from_tokens(&self, tokens: &[String]) -> Option<String> {
        let filtered: Vec<String> = tokens
            .iter()
            .filter(|token| {
                !QUESTION_WORDS.contains(&token.as_str())
                    && !["的", "是", "为", "了", "时候", "哪里", "号码"].contains(&token.as_str())
            })
            .cloned()
            .collect();
        if filtered.is_empty() {
            return None;
        }
        let joined: String = filtered.concat();
        self.lexicon.resolve_predicate_surface(&joined)
    }

    fn split_entity_predicate(&self, tokens: &[String]) -> Option<(String, String)> {
        let mut best: Option<(String, String)> = None;
        for length in 1..tokens.len() {
            let prefix: String = tokens[..length].concat();
            if !self.lexicon.is_entity_name(&prefix) && self.lexicon.find_entity(&prefix).is_none()
            {
                continue;
            }
            let rest: Vec<String> = tokens[length..]
                .iter()
                .filter(|token| !["是", "为", "的"].contains(&token.as_str()))
                .cloned()
                .collect();
            if rest.is_empty() {
                continue;
            }
            let Some(predicate) = self.predicate_from_tokens(&rest) else {
                continue;
            };
            if predicate.is_empty() {
                continue;
            }
            best = Some((prefix, predicate));
        }
        best
    }

    /// 旧的单读法入口：从图表完整分析中按「真值 > 一元谓词 > 任意」取一条。
    /// 新代码应使用 `chart_complete` 或 `parse_forest`，避免静默丢掉其它读法。
    fn chart_parse(&self, tokens: &[String], trace: &mut Vec<String>) -> Option<Item> {
        let top = self.chart_complete(tokens);
        let truth = top.iter().find(|item| item.ty == Type::Truth).cloned();
        let unary = top
            .iter()
            .find(|item| item.ty == Type::fun(Type::Entity, Type::Truth))
            .cloned();
        let chosen = truth.or(unary).or_else(|| top.first().cloned());
        if let Some(item) = &chosen {
            trace.push(format!("图表组合：{} : {}", item.label, item.ty.render()));
        }
        chosen
    }

    /// 图表完整分析：返回顶层全部读法（按首次发现顺序，按项渲染去重）。
    fn chart_complete(&self, tokens: &[String]) -> Vec<Item> {
        let n = tokens.len();
        if n == 0 || n > self.max_tokens {
            return Vec::new();
        }
        let entity_ty = Type::Entity;
        let truth_ty = Type::Truth;
        let unary_ty = Type::fun(entity_ty.clone(), truth_ty.clone());
        let binary_ty = Type::fun(entity_ty.clone(), unary_ty.clone());
        let quant_ty = Type::fun(unary_ty.clone(), truth_ty.clone());

        let mut cells: Vec<Vec<Vec<Item>>> = vec![vec![Vec::new(); n + 1]; n + 1];

        // 词项装载
        for i in 0..n {
            let token = tokens[i].clone();
            for entry in self.lexicon.lookup(&token) {
                let label = entry.term.render();
                push_item(
                    &mut cells,
                    i,
                    i + 1,
                    Item {
                        term: entry.term,
                        ty: entry.ty,
                        label,
                    },
                    self.max_items,
                );
            }
            if parse_number_token(&token).is_some() {
                push_item(
                    &mut cells,
                    i,
                    i + 1,
                    Item {
                        term: Term::Num(token.clone()),
                        ty: entity_ty.clone(),
                        label: token.clone(),
                    },
                    self.max_items,
                );
            }
            // 否定前缀「不 + 谓词」
            if token == "不" && i + 1 < n {
                for entry in self.lexicon.lookup(&tokens[i + 1]) {
                    if entry.ty == unary_ty {
                        let neg = Term::Lam(
                            "x".into(),
                            Box::new(Term::Not(Box::new(Term::apply(
                                entry.term.clone(),
                                Term::Var("x".into()),
                            )))),
                        );
                        push_item(
                            &mut cells,
                            i,
                            i + 2,
                            Item {
                                label: format!("不{}", entry.surface),
                                term: neg,
                                ty: unary_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                    if entry.ty == binary_ty {
                        let neg = Term::Lam(
                            "x".into(),
                            Box::new(Term::Lam(
                                "y".into(),
                                Box::new(Term::Not(Box::new(Term::apply(
                                    Term::apply(entry.term.clone(), Term::Var("y".into())),
                                    Term::Var("x".into()),
                                )))),
                            )),
                        );
                        push_item(
                            &mut cells,
                            i,
                            i + 2,
                            Item {
                                label: format!("不{}", entry.surface),
                                term: neg,
                                ty: binary_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                }
            }
            // 多词实体/谓词（跨 2..=4 个 token 的连接串）
            for length in 2..=4.min(n - i) {
                let joined: String = tokens[i..i + length].concat();
                if joined == token {
                    continue;
                }
                for entry in self.lexicon.lookup(&joined) {
                    let label = entry.term.render();
                    push_item(
                        &mut cells,
                        i,
                        i + length,
                        Item {
                            term: entry.term,
                            ty: entry.ty,
                            label,
                        },
                        self.max_items,
                    );
                }
            }
        }

        // 组合
        for length in 2..=n {
            for i in 0..=n - length {
                let j = i + length;
                // 量词规则：DET + N
                if DET_ALL.contains(&tokens[i].as_str()) || DET_SOME.contains(&tokens[i].as_str()) {
                    let universal = DET_ALL.contains(&tokens[i].as_str());
                    for head in cells[i + 1][j].clone() {
                        if head.ty != unary_ty {
                            continue;
                        }
                        let restrictor = Term::apply(head.term.clone(), Term::Var("x".into()));
                        let scope = Term::apply(Term::Var("P".into()), Term::Var("x".into()));
                        let body = if universal {
                            Term::Implies(Box::new(restrictor), Box::new(scope))
                        } else {
                            Term::And(vec![restrictor, scope])
                        };
                        let key = if universal {
                            Term::Forall {
                                var: "x".into(),
                                body: Box::new(body),
                            }
                        } else {
                            Term::Exists {
                                var: "x".into(),
                                body: Box::new(body),
                            }
                        };
                        let quant = Term::Lam("P".into(), Box::new(key));
                        push_item(
                            &mut cells,
                            i,
                            j,
                            Item {
                                label: format!("{}{}", tokens[i], head.label),
                                term: quant,
                                ty: quant_ty.clone(),
                            },
                            self.max_items,
                        );
                    }
                }
                // 连接词规则（合取/析取 + 话题-述题「是」）
                for k in i + 1..j {
                    if k + 1 >= j {
                        continue;
                    }
                    let connector = tokens[k].as_str();
                    if CONJ_AND.contains(&connector) || CONJ_OR.contains(&connector) {
                        let conjunctive = CONJ_AND.contains(&connector);
                        for a in cells[i][k].clone() {
                            if a.ty != truth_ty {
                                continue;
                            }
                            for b in cells[k + 1][j].clone() {
                                if b.ty != truth_ty {
                                    continue;
                                }
                                let term = if conjunctive {
                                    Term::And(vec![a.term.clone(), b.term.clone()])
                                } else {
                                    Term::Or(vec![a.term.clone(), b.term.clone()])
                                };
                                push_item(
                                    &mut cells,
                                    i,
                                    j,
                                    Item {
                                        label: format!("({} {} {})", a.label, connector, b.label),
                                        term,
                                        ty: truth_ty.clone(),
                                    },
                                    self.max_items,
                                );
                            }
                        }
                    }
                    // 系词规则：⟨左项，系词，右项⟩ 直接按「主语-表语」顺序构造 Isa，
                    // 避免把系词当成普通二元谓词时的参数顺序歧义（话题-述题也由此覆盖）
                    if COPULA_CONNECTORS.contains(&connector) {
                        let negate = connector == "不是" || connector == "并非";
                        for a in cells[i][k].clone() {
                            for b in cells[k + 1][j].clone() {
                                let body = if b.ty == entity_ty {
                                    if a.ty == entity_ty {
                                        Some(Term::apply(
                                            Term::Lam(
                                                "z".into(),
                                                Box::new(Term::Isa(
                                                    Box::new(Term::Var("z".into())),
                                                    Box::new(b.term.clone()),
                                                )),
                                            ),
                                            a.term.clone(),
                                        ))
                                    } else if a.ty == quant_ty {
                                        Some(Term::apply(
                                            a.term.clone(),
                                            Term::Lam(
                                                "z".into(),
                                                Box::new(Term::Isa(
                                                    Box::new(Term::Var("z".into())),
                                                    Box::new(b.term.clone()),
                                                )),
                                            ),
                                        ))
                                    } else if a.ty == unary_ty {
                                        Some(Term::apply(a.term.clone(), b.term.clone()))
                                    } else {
                                        None
                                    }
                                } else if b.ty == unary_ty && a.ty == entity_ty {
                                    Some(Term::apply(b.term.clone(), a.term.clone()))
                                } else if b.ty == unary_ty && a.ty == quant_ty {
                                    Some(Term::apply(a.term.clone(), b.term.clone()))
                                } else {
                                    None
                                };
                                let Some(body) = body else { continue };
                                let term = if negate {
                                    Term::Not(Box::new(body))
                                } else {
                                    body
                                };
                                push_item(
                                    &mut cells,
                                    i,
                                    j,
                                    Item {
                                        label: format!("{} {} {}", a.label, connector, b.label),
                                        term,
                                        ty: truth_ty.clone(),
                                    },
                                    self.max_items,
                                );
                            }
                        }
                    }
                }
                // 二元组合：函数应用 + 客体广义量词
                for k in i + 1..j {
                    for a in cells[i][k].clone() {
                        for b in cells[k][j].clone() {
                            for item in combine_items(&a, &b) {
                                push_item(&mut cells, i, j, item, self.max_items);
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Item> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for item in cells[0][n].iter() {
            if seen.insert(format!("{}:{}", item.ty.render(), item.term.render())) {
                out.push(item.clone());
            }
        }
        out
    }

    /// 在模型上求解逻辑形式
    pub fn solve(&self, lf: &LogicalForm, model: &TinyModel) -> QueryOutcome {
        match &lf.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                let values = model.values_of(entity, predicate);
                let evidence = if values.is_empty() {
                    vec![format!(
                        "知识图谱中没有「{}」的「{}」记录",
                        entity, predicate
                    )]
                } else {
                    values
                        .iter()
                        .map(|value| format!("{} 的 {} 是 {}", entity, predicate, value))
                        .collect()
                };
                QueryOutcome::Values {
                    entity: entity.clone(),
                    predicate: predicate.clone(),
                    values,
                    evidence,
                }
            }
            LfKind::Query(Query::Entity { predicate }) => {
                let entities = model.entities_satisfying(predicate);
                let evidence = entities
                    .iter()
                    .map(|entity| format!("{} 是 {}", entity, predicate))
                    .collect();
                QueryOutcome::Entities {
                    predicate: predicate.clone(),
                    entities,
                    evidence,
                }
            }
            LfKind::Query(Query::EntityByObject { predicate, object }) => {
                let entities = model.subjects_with_object(predicate, object);
                let evidence = entities
                    .iter()
                    .map(|entity| format!("{} —{}→ {}", entity, predicate, object))
                    .collect();
                QueryOutcome::EntitiesByObject {
                    predicate: predicate.clone(),
                    object: object.clone(),
                    entities,
                    evidence,
                }
            }
            LfKind::Query(Query::Truth { body }) => {
                let value = model.eval(body);
                match value {
                    Eval::Truth(holds) => QueryOutcome::Truth {
                        holds,
                        statement: body.render(),
                        evidence: model.evidence_for(body),
                    },
                    Eval::Failure(reason) => QueryOutcome::Unsupported(reason),
                    _ => QueryOutcome::Unsupported("真值查询未得到真假值".to_string()),
                }
            }
            LfKind::Statement(term) => {
                let value = model.eval(term);
                match value {
                    Eval::Truth(holds) => QueryOutcome::Truth {
                        holds,
                        statement: term.render(),
                        evidence: model.evidence_for(term),
                    },
                    Eval::Failure(reason) => QueryOutcome::Unsupported(reason),
                    _ => QueryOutcome::Unsupported("陈述求值未得到真假值".to_string()),
                }
            }
            LfKind::Unknown => QueryOutcome::Unsupported("未能组合出逻辑形式".to_string()),
        }
    }

    /// 候选读法的证据缺口：实体/谓词不在模型里，或查询没有命中记录。
    fn candidate_gaps(&self, lf: &LogicalForm, model: &TinyModel) -> Vec<String> {
        let mut gaps = Vec::new();
        match &lf.kind {
            LfKind::Query(Query::Value { entity, predicate }) => {
                if !model.knows_name(entity) {
                    gaps.push(format!("实体「{}」不在已知个体域", entity));
                }
                if !model.knows_predicate(predicate) {
                    gaps.push(format!("谓词「{}」没有任何已知记录", predicate));
                }
            }
            LfKind::Query(Query::Entity { predicate }) => {
                if !model.knows_predicate(predicate) {
                    gaps.push(format!("谓词「{}」没有任何已知记录", predicate));
                }
            }
            LfKind::Query(Query::EntityByObject { predicate, object }) => {
                if !model.knows_predicate(predicate) {
                    gaps.push(format!("谓词「{}」没有任何已知记录", predicate));
                }
                if !model.knows_name(object) {
                    gaps.push(format!("实体「{}」不在已知个体域", object));
                }
            }
            LfKind::Query(Query::Truth { body }) | LfKind::Statement(body) => {
                for name in model.unknown_mentions(body) {
                    gaps.push(format!("「{}」不在个体域或谓词表中", name));
                }
            }
            LfKind::Unknown => gaps.push("未能组合出可求值的逻辑形式".to_string()),
        }
        gaps
    }

    /// 把单个候选放到模型上接地：有证据 / 被反驳 / 证据缺口。
    pub fn ground_candidate(
        &self,
        candidate: ParseCandidate,
        model: &TinyModel,
    ) -> GroundedCandidate {
        let gaps = self.candidate_gaps(&candidate.form, model);
        let outcome = self.solve(&candidate.form, model);
        let status = match &outcome {
            QueryOutcome::Values { values, .. } => {
                if values.is_empty() {
                    CandidateStatus::Open
                } else {
                    CandidateStatus::Supported
                }
            }
            QueryOutcome::Entities { entities, .. }
            | QueryOutcome::EntitiesByObject { entities, .. } => {
                if entities.is_empty() {
                    CandidateStatus::Open
                } else {
                    CandidateStatus::Supported
                }
            }
            QueryOutcome::Truth { holds: true, .. } => {
                if gaps.is_empty() {
                    CandidateStatus::Supported
                } else {
                    CandidateStatus::Open
                }
            }
            QueryOutcome::Truth { holds: false, .. } => {
                if gaps.is_empty() {
                    CandidateStatus::Refuted
                } else {
                    CandidateStatus::Open
                }
            }
            QueryOutcome::Unsupported(_) => CandidateStatus::Open,
        };
        GroundedCandidate {
            candidate,
            outcome,
            status,
            gaps,
        }
    }

    /// 联合解析-推理：保留森林、逐候选接地，再决定唯一答案/歧义/缺口。
    pub fn evaluate(&self, text: &str, model: &TinyModel) -> ForestEvaluation {
        let forest = self.parse_forest(text);
        self.evaluate_forest(forest, model)
    }

    /// 对已有森林做接地与判定（供外部候选来源合并后复用）。
    pub fn evaluate_forest(&self, forest: ParseForest, model: &TinyModel) -> ForestEvaluation {
        let grounded: Vec<GroundedCandidate> = forest
            .candidates
            .iter()
            .cloned()
            .map(|candidate| self.ground_candidate(candidate, model))
            .collect();
        let mut seen = HashSet::new();
        let mut supported_indexes = Vec::new();
        for (index, candidate) in grounded.iter().enumerate() {
            if candidate.status != CandidateStatus::Supported {
                continue;
            }
            if seen.insert(candidate.outcome.signature()) {
                supported_indexes.push(index);
            }
        }
        let decision = if supported_indexes.len() == 1 {
            ForestDecision::Answer {
                index: supported_indexes[0],
            }
        } else if supported_indexes.len() > 1 {
            ForestDecision::Ambiguous {
                indexes: supported_indexes,
            }
        } else if grounded.is_empty() {
            ForestDecision::Unknown {
                reason: "没有可求值的候选逻辑形式".to_string(),
            }
        } else if grounded
            .iter()
            .any(|candidate| candidate.status == CandidateStatus::Refuted)
        {
            ForestDecision::Unknown {
                reason: "候选均被知识图谱反驳；不把否定当作答案".to_string(),
            }
        } else {
            let mut gaps: Vec<String> = Vec::new();
            for candidate in &grounded {
                for gap in &candidate.gaps {
                    if !gaps.contains(gap) {
                        gaps.push(gap.clone());
                    }
                }
            }
            let detail = if gaps.is_empty() {
                "图谱没有命中记录".to_string()
            } else {
                gaps.join("；")
            };
            ForestDecision::Unknown {
                reason: format!("证据不足：{}", detail),
            }
        };
        ForestEvaluation {
            forest,
            grounded,
            decision,
        }
    }

    /// 从联合判定生成中文回答；只有唯一且正证据支持的候选才作答。
    pub fn answer_evaluation(
        &self,
        evaluation: &ForestEvaluation,
        include_evidence: bool,
    ) -> Option<String> {
        let index = match evaluation.decision {
            ForestDecision::Answer { index } => index,
            _ => return None,
        };
        let grounded = evaluation.grounded.get(index)?;
        let mut answer = self.render_outcome(&grounded.outcome);
        if include_evidence {
            let evidence = match &grounded.outcome {
                QueryOutcome::Values { evidence, .. }
                | QueryOutcome::Entities { evidence, .. }
                | QueryOutcome::EntitiesByObject { evidence, .. }
                | QueryOutcome::Truth { evidence, .. } => evidence.clone(),
                QueryOutcome::Unsupported(_) => Vec::new(),
            };
            if !evidence.is_empty() && !answer.contains("证据：") {
                answer.push_str(&format!("\n证据：{}", evidence.join("；")));
            }
        }
        Some(answer)
    }

    /// 一步完成解析森林 + 模型接地 + 唯一答案生成。
    pub fn answer_forest(
        &self,
        text: &str,
        model: &TinyModel,
        include_evidence: bool,
    ) -> Option<String> {
        let evaluation = self.evaluate(text, model);
        self.answer_evaluation(&evaluation, include_evidence)
    }

    /// 候选森林与接地状态的可读报告（用于「语义分析」命令与失败诊断）。
    pub fn render_forest_report(&self, evaluation: &ForestEvaluation) -> String {
        let mut out = String::new();
        out.push_str(&format!("语义森林：「{}」\n", evaluation.forest.source));
        if evaluation.forest.candidates.is_empty() {
            for step in &evaluation.forest.trace {
                out.push_str(&format!("  • {}\n", step));
            }
            out.push_str("判定：没有可求值的逻辑形式。");
            return out.trim_end().to_string();
        }
        for (index, grounded) in evaluation.grounded.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{}，成本 {}] {}\n",
                index + 1,
                grounded.candidate.strategy.label(),
                grounded.candidate.cost,
                grounded.candidate.form.render()
            ));
            out.push_str(&format!("   状态：{}", grounded.status.label()));
            if !grounded.gaps.is_empty() {
                out.push_str(&format!("（缺口：{}）", grounded.gaps.join("；")));
            }
            out.push('\n');
            let evidence = match &grounded.outcome {
                QueryOutcome::Values { evidence, .. }
                | QueryOutcome::Entities { evidence, .. }
                | QueryOutcome::EntitiesByObject { evidence, .. }
                | QueryOutcome::Truth { evidence, .. } => evidence.clone(),
                QueryOutcome::Unsupported(_) => Vec::new(),
            };
            if !evidence.is_empty() {
                out.push_str(&format!("   证据：{}\n", evidence.join("；")));
            }
        }
        match &evaluation.decision {
            ForestDecision::Answer { index } => {
                out.push_str(&format!("判定：唯一可答读法（候选 {}）。", index + 1));
            }
            ForestDecision::Ambiguous { indexes } => {
                out.push_str("判定：存在歧义；不同读法给出不同答案：");
                let answers: Vec<String> = indexes
                    .iter()
                    .filter_map(|index| evaluation.grounded.get(*index))
                    .map(|grounded| self.render_outcome(&grounded.outcome))
                    .collect();
                out.push_str(&answers.join(" / "));
            }
            ForestDecision::Unknown { reason } => {
                out.push_str(&format!("判定：证据不足（{}）。", reason));
            }
        }
        out.trim_end().to_string()
    }

    /// 渲染求解结果（中文生成）
    pub fn render_outcome(&self, outcome: &QueryOutcome) -> String {
        match outcome {
            QueryOutcome::Values {
                entity,
                predicate,
                values,
                ..
            } => {
                let surface = self.lexicon.surface_of(predicate);
                if values.is_empty() {
                    format!("知识图谱中没有「{}」的「{}」记录。", entity, surface)
                } else {
                    format!(
                        "根据知识图谱：{} 的 {} 是 {}。",
                        entity,
                        surface,
                        values.join("、")
                    )
                }
            }
            QueryOutcome::Entities {
                predicate,
                entities,
                ..
            } => {
                let surface = self.lexicon.surface_of(predicate);
                if entities.is_empty() {
                    format!("知识图谱中没有满足「{}」的实体。", surface)
                } else {
                    format!(
                        "满足「{}」的实体有 {} 个：{}。",
                        surface,
                        entities.len(),
                        entities.join("、")
                    )
                }
            }
            QueryOutcome::EntitiesByObject {
                predicate,
                object,
                entities,
                ..
            } => {
                let surface = self.lexicon.surface_of(predicate);
                if entities.is_empty() {
                    format!(
                        "知识图谱中没有以「{}」为「{}」对象的实体。",
                        object, surface
                    )
                } else {
                    format!(
                        "以「{}」为「{}」对象的实体有 {} 个：{}。",
                        object,
                        surface,
                        entities.len(),
                        entities.join("、")
                    )
                }
            }
            QueryOutcome::Truth {
                holds, statement, ..
            } => {
                format!(
                    "判定：{} —— 在知识图谱中{}。",
                    statement,
                    if *holds { "成立" } else { "不成立" }
                )
            }
            QueryOutcome::Unsupported(reason) => format!("无法求解：{}。", reason),
        }
    }

    /// 逻辑形式分析报告（解析 + 轨迹 + 模型求值）
    pub fn describe(&self, lf: &LogicalForm) -> String {
        let mut out = String::new();
        out.push_str(&format!("逻辑形式：{}\n", lf.render()));
        out.push_str("分析轨迹：\n");
        for step in &lf.trace {
            out.push_str(&format!("  • {}\n", step));
        }
        out.trim_end().to_string()
    }

    /// 求值报告（供「逻辑形式」命令展示）
    pub fn denotation_report(&self, lf: &LogicalForm, model: &TinyModel) -> String {
        if !lf.is_known() {
            return "未能得出可求值的逻辑形式（可尝试更短的说法，或先导入相关事实）。".to_string();
        }
        let outcome = self.solve(lf, model);
        let mut out = self.render_outcome(&outcome);
        let evidence = match &outcome {
            QueryOutcome::Values { evidence, .. }
            | QueryOutcome::Entities { evidence, .. }
            | QueryOutcome::EntitiesByObject { evidence, .. }
            | QueryOutcome::Truth { evidence, .. } => evidence.clone(),
            QueryOutcome::Unsupported(_) => Vec::new(),
        };
        if !evidence.is_empty() {
            out.push_str(&format!("\n证据：{}", evidence.join("；")));
        }
        out
    }
}
/// 森林追加：按逻辑形式签名去重，保留成本更低的读法。
fn push_forest_candidate(
    forest: &mut ParseForest,
    seen: &mut HashSet<String>,
    form: LogicalForm,
    strategy: ParseStrategy,
    cost: usize,
) {
    let signature = form.signature();
    if !seen.insert(signature.clone()) {
        if let Some(existing) = forest
            .candidates
            .iter_mut()
            .find(|candidate| candidate.form.signature() == signature)
        {
            if cost < existing.cost {
                existing.cost = cost;
                existing.strategy = strategy;
            }
        }
        return;
    }
    forest.candidates.push(ParseCandidate {
        form,
        strategy,
        cost,
    });
}
/// 把完整解析饱和为真值：一元谓词做存在闭包（无主语句），其余仅接受 t
fn saturate(item: Item) -> Option<Term> {
    match &item.ty {
        Type::Truth => Some(item.term),
        Type::Fun(dom, cod) if **dom == Type::Entity && **cod == Type::Truth => {
            Some(Term::Exists {
                var: "x".into(),
                body: Box::new(Term::apply(item.term, Term::Var("x".into()))),
            })
        }
        _ => None,
    }
}
/// 无断言标记时把 P(E) 形态识别为属性值查询
fn possessive_np(term: &Term) -> Option<(String, String)> {
    if let Term::App(f, a) = term {
        if let (Term::Pred(predicate), Term::Const(entity)) = (&**f, &**a) {
            return Some((entity.clone(), predicate.clone()));
        }
    }
    None
}
/// 两项的类型驱动组合：前向应用、后向应用、客体广义量词
fn combine_items(a: &Item, b: &Item) -> Vec<Item> {
    let mut out = Vec::new();
    let entity_ty = Type::Entity;
    let truth_ty = Type::Truth;
    let unary_ty = Type::fun(entity_ty.clone(), truth_ty.clone());
    let binary_ty = Type::fun(entity_ty.clone(), unary_ty.clone());
    let quant_ty = Type::fun(unary_ty.clone(), truth_ty.clone());
    // 前向应用：a 是函数，b 是参数
    if let Type::Fun(dom, cod) = &a.ty {
        if **dom == b.ty {
            out.push(Item {
                term: Term::apply(a.term.clone(), b.term.clone()),
                label: format!("{}({})", a.label, b.label),
                ty: (**cod).clone(),
            });
        }
    }
    // 后向应用：b 是函数，a 是参数
    if let Type::Fun(dom, cod) = &b.ty {
        if **dom == a.ty {
            out.push(Item {
                term: Term::apply(b.term.clone(), a.term.clone()),
                label: format!("{}({})", b.label, a.label),
                ty: (**cod).clone(),
            });
        }
    }
    // 客体广义量词：TV + 量化 NP → VP
    if a.ty == binary_ty && b.ty == quant_ty {
        let term = Term::Lam(
            "x".into(),
            Box::new(Term::App(
                Box::new(b.term.clone()),
                Box::new(Term::Lam(
                    "y".into(),
                    Box::new(Term::App(
                        Box::new(Term::App(
                            Box::new(a.term.clone()),
                            Box::new(Term::Var("y".into())),
                        )),
                        Box::new(Term::Var("x".into())),
                    )),
                )),
            )),
        );
        out.push(Item {
            label: format!("{}⟨{}⟩", a.label, b.label),
            term,
            ty: unary_ty.clone(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// 语义层对话记忆：逻辑形式栈 + 话题实体/谓词，支撑指代消解与省略补全
// ---------------------------------------------------------------------------
pub struct SemanticMemory {
    capacity: usize,
    forms: Vec<LogicalForm>,
    focus_entities: Vec<String>,
    last_predicate: Option<String>,
}

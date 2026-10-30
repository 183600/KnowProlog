//! Scryer Prolog 适配层：把动态事实装载、三态查询与统一绑定串成一条链路。
//!
//! 引擎对外只暴露两类结果：
//! - 事实装载：`load_facts*` / `load_temporal_facts*`，失败一律显式返回；
//! - 查询：`query*` 家族，其中 [`QueryOutcome`] 区分「有解 / 无解 / 引擎故障」。
//!
//! 事实按规范序比较，装载时只写差量；Prolog 源码中的文本统一经
//! [`escape_atom`] 转义，避免名称里的引号或反斜杠破坏语法。

use scryer_prolog::machine::parsed_results::{prolog_value_to_json_string, QueryResolution, Value};
use scryer_prolog::machine::Machine;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

use crate::bindings::{BindingSet, BoundTerm};
use crate::temporal::{TemporalFact, TimePoint, Validity};

/// 查询的三态结局：有解 / 无解 / 引擎故障。
///
/// 无解（`QueryResolution::False`）与故障（`Err`）绝不合并成同一个空
/// `Vec`：前者是「知识里没有」，后者是「我出错了」——认识论上不同，
/// 恢复策略也不同（见 [`crate::recovery`]）。
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOutcome {
    /// 查询成功：`Matches` 的绑定解（跳过空绑定行）；纯 `True` 为空的解集。
    Solutions(Vec<BindingSet>),
    /// 查询成功但无解：知识图谱里确实没有。
    NoSolution,
    /// 引擎执行失败（语法错误 / 未定义谓词 / consult 损坏等）：信息不销毁。
    EngineFault(String),
}

impl QueryOutcome {
    pub fn is_fault(&self) -> bool {
        matches!(self, QueryOutcome::EngineFault(_))
    }

    pub fn has_solutions(&self) -> bool {
        matches!(self, QueryOutcome::Solutions(solutions) if !solutions.is_empty())
    }

    pub fn fault_message(&self) -> Option<&str> {
        match self {
            QueryOutcome::EngineFault(message) => Some(message),
            _ => None,
        }
    }

    /// 简化视图：非解（无解 / 故障）给空 Vec（故障信息保留在枚举里）。
    pub fn into_solutions(self) -> Vec<BindingSet> {
        match self {
            QueryOutcome::Solutions(solutions) => solutions,
            _ => Vec::new(),
        }
    }
}

pub struct PrologEngine {
    machine: Machine,
    /// 已加载的动态事实（按规范序排序）
    loaded_facts: Vec<PrologFact>,
    /// 已加载的时序事实（`temporal_edge/8`）
    loaded_temporal_facts: Vec<TemporalFact>,
    /// 事实增量装载统计（全量重灌 vs 差量 assert/retract）。
    fact_load_stats: FactLoadStats,
    /// 最近一次引擎层故障（consult / 查询错误 / 崩溃），由调用方取走上报。
    last_fault: Option<String>,
    /// 引擎是否已在崩溃后停止服务（scryer 0.9.x 的 consult 解析错误会 panic，
    /// 崩溃后的内部状态不可信：此后一律返回 EngineFault，绝不静默当作无解）。
    poisoned: bool,
}

/// 动态事实装载统计：区分「整段重灌」与「只动差量」。
///
/// `last_added`/`last_removed` 记录最近一次实际写入引擎的正负事实条数；
/// 事实集未变化时两者都为 0。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FactLoadStats {
    pub full_loads: u64,
    pub incremental_loads: u64,
    pub last_added: usize,
    pub last_removed: usize,
}

/// 从 panic payload 提取可读消息。
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "未知 panic".to_string()
    }
}

/// 把类型化解集渲染成兼容的字符串行（未绑定项记为 `"null"`）。
///
/// [`PrologEngine::query`] 与 [`PrologEngine::query_checked`] 共用这一条
/// 渲染路径，保证两个入口对同一查询给出逐字相同的绑定。
fn render_binding_rows(solutions: Vec<BindingSet>) -> Vec<HashMap<String, String>> {
    solutions
        .into_iter()
        .map(|bindings| {
            bindings
                .iter()
                .map(|(name, term)| {
                    (
                        name.clone(),
                        term.render_raw().unwrap_or_else(|| "null".to_string()),
                    )
                })
                .collect()
        })
        .collect()
}

impl PrologEngine {
    pub fn new() -> Self {
        let machine = Machine::new_lib();

        let base_rules = r#"
            % 列表谓词（member/2、length/2 等）
            :- use_module(library(lists)).
            
            % 动态事实谓词声明（支持 retractall 更新事实）
            :- dynamic(oxigraph_entity/3).
            :- dynamic(oxigraph_relation/3).

            % 时序事实：temporal_edge(主语, 谓词, 宾语, 起, 止, 粒度, 状态/事件, 来源)
            % 起/止为 YYYYMMDD 整数；unbounded 表示该侧无界；unknown 表示时间未知
            % （未知不参与 holds_at，绝不伪装成永远为真）。
            :- dynamic(temporal_edge/8).

            temporal_lower(unbounded, _).
            temporal_lower(Start, T) :- integer(Start), Start =< T.

            temporal_upper(unbounded, _).
            temporal_upper(End, T) :- integer(End), T =< End.

            % 在时刻 T（YYYYMMDD 整数）为真的时序事实。
            holds_at(S, P, O, T) :-
                temporal_edge(S, P, O, Start, End, _, _, _),
                temporal_lower(Start, T),
                temporal_upper(End, T).

            state_at(S, P, O, T) :-
                temporal_edge(S, P, O, Start, End, _, state, _),
                temporal_lower(Start, T),
                temporal_upper(End, T).

            event_at(S, P, O, T) :-
                temporal_edge(S, P, O, Start, End, _, event, _),
                temporal_lower(Start, T),
                temporal_upper(End, T).

            % T 之前已经成立（止点早于 T；无界/未知止点不算）。
            temporal_before(S, P, O, T) :-
                temporal_edge(S, P, O, _, End, _, _, _),
                integer(End),
                End < T.

            % T 之后才开始成立（起点晚于 T）。
            temporal_after(S, P, O, T) :-
                temporal_edge(S, P, O, Start, _, _, _, _),
                integer(Start),
                Start > T.
            
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

        let mut engine = Self {
            machine,
            loaded_facts: Vec::new(),
            loaded_temporal_facts: Vec::new(),
            fact_load_stats: FactLoadStats::default(),
            last_fault: None,
            poisoned: false,
        };
        if let Err(error) = engine.consult_source(base_rules) {
            engine.note_fault(format!("基础规则加载失败: {}", error));
        }
        engine
    }

    /// 记录一条引擎层故障（同消息只打印一次 + 留存供调用方上报）。
    fn note_fault(&mut self, message: String) {
        if self.last_fault.as_deref() != Some(message.as_str()) {
            eprintln!("Prolog引擎故障: {}", message);
        }
        self.last_fault = Some(message);
    }

    /// 查看（不清空）最近一次引擎故障。
    pub fn last_fault(&self) -> Option<&str> {
        self.last_fault.as_deref()
    }

    /// 取走最近一次引擎故障（上报后清空，避免重复刷屏）。
    pub fn take_fault(&mut self) -> Option<String> {
        self.last_fault.take()
    }

    /// 引擎是否已在崩溃后停止服务。
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// 向 user 模块 consult 一段 Prolog 源码（供事实加载与故障注入）。
    ///
    /// Scryer 0.9.x 的 `consult_module_string` 不返回 Result，解析错误会在
    /// 引擎内部 panic。这里用 `catch_unwind` 兜底：崩溃被转成 `Err`，引擎
    /// 标记为停止服务（后续查询一律 [`QueryOutcome::EngineFault`]），
    /// 「我出错了」永远不会伪装成「知识里没有」。
    pub fn consult_source(&mut self, source: impl Into<String>) -> Result<(), String> {
        if self.poisoned {
            return Err("引擎已因先前的崩溃停止服务（需重建实例）".to_string());
        }
        let source = source.into();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.machine.consult_module_string("user", source);
        })) {
            Ok(()) => Ok(()),
            Err(payload) => {
                self.poisoned = true;
                Err(format!("consult 触发引擎崩溃：{}", panic_message(payload)))
            }
        }
    }

    pub fn load_facts(&mut self, facts: Vec<PrologFact>) {
        if let Err(error) = self.load_facts_checked(facts) {
            self.note_fault(format!("事实加载失败: {}", error));
        }
    }

    /// 最近一次事实装载统计（用于观测差量装载是否真的生效）。
    pub fn fact_load_stats(&self) -> FactLoadStats {
        self.fact_load_stats
    }

    /// 与 [`Self::load_facts`] 相同，但把 consult 失败显式交给调用方。
    ///
    /// 两次事实集之间只写差量：新事实 `consult`，被撤回的事实按谓词
    /// `retractall`，不再「清空 + 全量重灌」。事实集未变化时零写入。
    /// 实体事实以 `(名称, 类型)` 为撤回粒度（属性列表不参与 retract 合一），
    /// 因此该键下未变化的事实会被重新断言；关系事实则可精确撤回。
    pub fn load_facts_checked(&mut self, facts: Vec<PrologFact>) -> Result<(), String> {
        // 按规范序排序，使相同事实集在不同调用间可直接比较（与加载顺序无关）
        let mut facts = facts;
        facts.sort_by(Self::compare_facts);

        if facts == self.loaded_facts {
            self.fact_load_stats.last_added = 0;
            self.fact_load_stats.last_removed = 0;
            return Ok(());
        }

        let previous = std::mem::take(&mut self.loaded_facts);
        let previous_counts = Self::count_facts(&previous);
        let target_counts = Self::count_facts(&facts);

        // 需要撤回的事实键（BTreeSet 保证写入顺序确定，便于测试/复现）
        let mut reset_keys: BTreeSet<FactIdentity> = BTreeSet::new();
        let mut removed_count = 0usize;
        for (fact, before) in previous_counts.values() {
            let after = target_counts
                .get(&Self::fact_key(fact))
                .map(|(_, count)| *count)
                .unwrap_or(0);
            if *before <= after {
                continue;
            }
            removed_count += before - after;
            reset_keys.insert(fact.identity());
        }

        let mut retract_goals = Vec::with_capacity(reset_keys.len());
        for key in &reset_keys {
            match key {
                FactIdentity::Entity { name, entity_type } => retract_goals.push(format!(
                    "retractall(oxigraph_entity('{}', '{}', _))",
                    escape_atom(name),
                    escape_atom(entity_type)
                )),
                FactIdentity::Relation { from, rel_type, to } => retract_goals.push(format!(
                    "retractall(oxigraph_relation('{}', '{}', '{}'))",
                    escape_atom(from),
                    escape_atom(rel_type),
                    escape_atom(to)
                )),
            }
        }

        // 被撤回的实体/关系键在 retractall 后需要整体补回目标中该键的现存事实；
        // 其余键只补「目标比原先多出来」的重复条数。
        let mut added_count = 0usize;
        let mut additions: Vec<&PrologFact> = Vec::new();
        for (fact, after) in target_counts.values() {
            let before = previous_counts
                .get(&Self::fact_key(fact))
                .map(|(_, count)| *count)
                .unwrap_or(0);
            let reset = reset_keys.contains(&fact.identity());
            added_count += after.saturating_sub(before);
            let to_write = if reset {
                *after
            } else {
                after.saturating_sub(before)
            };
            additions.extend(std::iter::repeat(fact).take(to_write));
        }

        let result = (|| {
            if !retract_goals.is_empty() {
                self.run_goal_checked(&retract_goals.join(", "))?;
            }
            if !additions.is_empty() {
                let mut fact_string = String::with_capacity(additions.len() * 64);
                for fact in &additions {
                    Self::write_fact_source(&mut fact_string, fact);
                }
                self.consult_source(fact_string)?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                if previous.is_empty() {
                    self.fact_load_stats.full_loads += 1;
                } else {
                    self.fact_load_stats.incremental_loads += 1;
                }
                self.fact_load_stats.last_added = added_count;
                self.fact_load_stats.last_removed = removed_count;
                self.loaded_facts = facts;
                Ok(())
            }
            Err(error) => {
                // 引擎可能已部分撤回：清空已加载事实，迫使下一次同目标事实
                // 走完整装载，绝不因「记录里看着已加载」而提前返回。
                let _ = self.clear_dynamic_facts();
                self.loaded_facts.clear();
                self.fact_load_stats.last_added = 0;
                self.fact_load_stats.last_removed = 0;
                Err(error)
            }
        }
    }

    /// 事实的无碰撞规范键：属性按键排序，与 HashMap 迭代序无关。
    fn fact_key(fact: &PrologFact) -> String {
        fn push_part(buffer: &mut String, part: &str) {
            let _ = write!(buffer, "{}:{}", part.len(), part);
        }
        let mut key = String::new();
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                key.push('E');
                push_part(&mut key, name);
                push_part(&mut key, entity_type);
                let mut entries: Vec<_> = properties.iter().collect();
                entries.sort_unstable();
                for (property, value) in entries {
                    push_part(&mut key, property);
                    push_part(&mut key, value);
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                key.push('R');
                push_part(&mut key, from);
                push_part(&mut key, rel_type);
                push_part(&mut key, to);
            }
        }
        key
    }

    fn count_facts(facts: &[PrologFact]) -> HashMap<String, (PrologFact, usize)> {
        let mut counts: HashMap<String, (PrologFact, usize)> =
            HashMap::with_capacity(facts.len());
        for fact in facts {
            let entry = counts
                .entry(Self::fact_key(fact))
                .or_insert_with(|| (fact.clone(), 0));
            entry.1 += 1;
        }
        counts
    }

    fn write_fact_source(fact_string: &mut String, fact: &PrologFact) {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
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

    /// 执行一个确定性目标；scryer 内部 panic 与查询路径同样被收进故障态。
    fn run_goal_checked(&mut self, goal: &str) -> Result<(), String> {
        if self.poisoned {
            return Err("引擎已因先前的崩溃停止服务（需重建实例）".to_string());
        }
        let goal = format!("{}.", goal.trim_end_matches('.'));
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.machine.run_query(goal)
        })) {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(error.to_string()),
            Err(payload) => {
                self.poisoned = true;
                let message = format!("目标执行触发引擎崩溃：{}", panic_message(payload));
                self.note_fault(message.clone());
                Err(message)
            }
        }
    }

    /// 事实的规范比较顺序：先实体后关系，各自按字段字典序
    fn compare_facts(a: &PrologFact, b: &PrologFact) -> Ordering {
        match (a, b) {
            (
                PrologFact::Entity {
                    name: n1,
                    entity_type: t1,
                    ..
                },
                PrologFact::Entity {
                    name: n2,
                    entity_type: t2,
                    ..
                },
            ) => n1.cmp(n2).then_with(|| t1.cmp(t2)),
            (PrologFact::Entity { .. }, PrologFact::Relation { .. }) => Ordering::Less,
            (PrologFact::Relation { .. }, PrologFact::Entity { .. }) => Ordering::Greater,
            (
                PrologFact::Relation {
                    from: f1,
                    rel_type: r1,
                    to: o1,
                },
                PrologFact::Relation {
                    from: f2,
                    rel_type: r2,
                    to: o2,
                },
            ) => f1.cmp(f2).then_with(|| r1.cmp(r2)).then_with(|| o1.cmp(o2)),
        }
    }

    /// 加载时序事实到 `temporal_edge/8`；与 [`Self::load_facts`] 相互独立。
    pub fn load_temporal_facts(&mut self, facts: Vec<TemporalFact>) {
        if let Err(error) = self.load_temporal_facts_checked(facts) {
            self.note_fault(format!("时序事实加载失败: {}", error));
        }
    }

    /// 与 [`Self::load_temporal_facts`] 相同，但把 consult 失败显式交给调用方。
    pub fn load_temporal_facts_checked(&mut self, facts: Vec<TemporalFact>) -> Result<(), String> {
        let mut facts = facts;
        facts.sort_by(|left, right| {
            left.sort_key()
                .cmp(&right.sort_key())
                .then(left.validity.render().cmp(&right.validity.render()))
                .then(left.source.cmp(&right.source))
        });
        if facts == self.loaded_temporal_facts {
            return Ok(());
        }
        self.clear_dynamic_temporal_facts()?;
        let mut source = String::with_capacity(facts.len() * 96);
        for fact in &facts {
            let aspect = fact.aspect.tag();
            let granularity = match fact.validity {
                Validity::Span(span) => span.granularity.tag(),
                _ => "year",
            };
            let (start, end) = match fact.validity {
                Validity::Unknown => ("unknown".to_string(), "unknown".to_string()),
                Validity::Never => ("never".to_string(), "never".to_string()),
                Validity::Unbounded => ("unbounded".to_string(), "unbounded".to_string()),
                Validity::Span(span) => (
                    span.start
                        .map(|point| point.to_prolog_int().to_string())
                        .unwrap_or_else(|| "unbounded".to_string()),
                    span.end
                        .map(|point| point.to_prolog_int().to_string())
                        .unwrap_or_else(|| "unbounded".to_string()),
                ),
            };
            let source_label = if fact.source.trim().is_empty() {
                "知识图谱"
            } else {
                fact.source.as_str()
            };
            let _ = write!(
                source,
                "temporal_edge('{}', '{}', '{}', {}, {}, '{}', '{}', '{}').\n",
                escape_atom(&fact.subject),
                escape_atom(&fact.predicate),
                escape_atom(&fact.object),
                start,
                end,
                granularity,
                aspect,
                escape_atom(source_label)
            );
        }
        self.consult_source(source)?;
        self.loaded_temporal_facts = facts;
        Ok(())
    }

    fn clear_dynamic_temporal_facts(&mut self) -> Result<(), String> {
        self.machine
            .run_query("retractall(temporal_edge(_, _, _, _, _, _, _, _)).".to_string())
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 已加载的时序事实条数。
    pub fn temporal_fact_count(&self) -> usize {
        self.loaded_temporal_facts.len()
    }

    /// 生成“某时刻成立”的 Prolog 目标。
    pub fn holds_at_query(
        subject: &str,
        predicate: &str,
        object: &str,
        point: TimePoint,
    ) -> String {
        format!(
            "holds_at('{}', '{}', '{}', {})",
            escape_atom(subject),
            escape_atom(predicate),
            escape_atom(object),
            point.to_prolog_int()
        )
    }

    /// 生成“T 之前成立”的 Prolog 目标。
    pub fn temporal_before_query(
        subject: &str,
        predicate: &str,
        object: &str,
        point: TimePoint,
    ) -> String {
        format!(
            "temporal_before('{}', '{}', '{}', {})",
            escape_atom(subject),
            escape_atom(predicate),
            escape_atom(object),
            point.to_prolog_int()
        )
    }

    pub fn load_relevant_facts(&mut self, facts: Vec<PrologFact>) {
        self.load_facts(facts);
    }

    fn clear_dynamic_facts(&mut self) -> Result<(), String> {
        for goal in [
            "retractall(oxigraph_entity(_, _, _))",
            "retractall(oxigraph_relation(_, _, _))",
        ] {
            self.run_goal_checked(goal)?;
        }
        Ok(())
    }

    pub fn query(&mut self, query: &str) -> Vec<HashMap<String, String>> {
        render_binding_rows(self.query_outcome(query).into_solutions())
    }

    /// 与 [`Self::query`] 同解同序的类型化版本：绑定为 [`BoundTerm`]。
    ///
    /// 字符串行为（`Value::Var` → `"null"` 等）保留在 [`Self::query`] 中；
    /// 本 API 暴露语义：未绑定 = [`BoundTerm::Unbound`]，整数/浮点 = `Number`
    /// （raw 原样保留，value 可参与比较），列表/结构保持形状。
    pub fn query_typed(&mut self, query: &str) -> Vec<BindingSet> {
        self.query_outcome(query).into_solutions()
    }

    /// 带故障回执的字符串查询：无解仍是 `Ok(空 Vec)`，引擎故障是 `Err`。
    ///
    /// 这是给「同一轮必须区分查无与引擎故障」的调用方准备的；[`Self::query`]
    /// 保留为兼容视图（故障信息仍可在 [`Self::last_fault`] 取走）。
    pub fn query_checked(
        &mut self,
        query: &str,
    ) -> Result<Vec<HashMap<String, String>>, String> {
        match self.query_outcome(query) {
            QueryOutcome::Solutions(solutions) => Ok(render_binding_rows(solutions)),
            QueryOutcome::NoSolution => Ok(Vec::new()),
            QueryOutcome::EngineFault(message) => Err(message),
        }
    }

    /// 三态查询：有解 / 无解 / 引擎故障——`query` 与 `query_typed` 的公共真值层。
    ///
    /// 引擎崩溃（scryer 内部 panic）也被收进 `EngineFault`：崩溃后的实例
    /// 标记为停止服务，后续查询不触碰内部状态。
    pub fn query_outcome(&mut self, query: &str) -> QueryOutcome {
        if self.poisoned {
            let message = "引擎已因崩溃停止服务（需重建实例）".to_string();
            self.note_fault(message.clone());
            return QueryOutcome::EngineFault(message);
        }
        let query_str = format!("{}.", query.trim_end_matches('.'));
        let resolution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.machine.run_query(query_str)
        }));
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(payload) => {
                self.poisoned = true;
                let message = format!("查询触发引擎崩溃：{}", panic_message(payload));
                self.note_fault(message.clone());
                return QueryOutcome::EngineFault(message);
            }
        };
        match resolution {
            Ok(QueryResolution::Matches(matches)) => {
                // 直接提取变量绑定（如 Age/Relation/Count/Spouse）
                let solutions: Vec<BindingSet> = matches
                    .iter()
                    .filter(|query_match| !query_match.bindings.is_empty())
                    .map(|query_match| {
                        let mut typed = BindingSet::new();
                        for (name, value) in &query_match.bindings {
                            typed.insert(name.clone(), bound_from_value(value));
                        }
                        typed
                    })
                    .collect();
                QueryOutcome::Solutions(solutions)
            }
            // 查询成功但无可提取变量：目标成立（如 `true` 或全绑定查询）。
            Ok(QueryResolution::True) => QueryOutcome::Solutions(Vec::new()),
            Ok(QueryResolution::False) => QueryOutcome::NoSolution,
            Err(err) => {
                let message = err.to_string();
                self.note_fault(message.clone());
                QueryOutcome::EngineFault(message)
            }
        }
    }

    pub fn generate_prolog_query(pattern: &str, entities: &[String]) -> Option<String> {
        fn first_entity(entities: &[String]) -> Option<&str> {
            entities.first().map(String::as_str)
        }

        fn entity_property_query(
            entity: &str,
            entity_type: &str,
            property: &str,
            variable: &str,
        ) -> String {
            format!(
                "oxigraph_entity('{}', '{}', Properties), member({}-{}, Properties)",
                escape_atom(entity),
                escape_atom(entity_type),
                property,
                variable
            )
        }

        match pattern {
            "age_query" => first_entity(entities)
                .map(|person| format!("age_of('{}', Age)", escape_atom(person))),
            "relation_query" => {
                let from = entities.first()?;
                let to = entities.get(1)?;
                Some(format!(
                    "related('{}', Relation, '{}')",
                    escape_atom(from),
                    escape_atom(to)
                ))
            }
            "count_query" => first_entity(entities).map(|entity_type| {
                format!("count_entities('{}', Count)", escape_atom(entity_type))
            }),
            "spouse_query" => first_entity(entities)
                .map(|person| format!("spouse_of('{}', Spouse)", escape_atom(person))),
            "capital_query" => first_entity(entities)
                .map(|country| entity_property_query(country, "country", "capital", "Capital")),
            "altitude_query" => first_entity(entities).map(|place| {
                entity_property_query(place, "geographicfeature", "altitude", "Altitude")
            }),
            "chemical_formula_query" => first_entity(entities)
                .map(|chemical| entity_property_query(chemical, "chemical", "formula", "Formula")),
            "author_query" => first_entity(entities)
                .map(|work| entity_property_query(work, "literature", "author", "Author")),
            "champion_query" => first_entity(entities)
                .map(|event| entity_property_query(event, "sportevent", "champion", "Champion")),
            "birth_year_query" => first_entity(entities).map(|language| {
                entity_property_query(language, "programminglanguage", "birth_year", "Year")
            }),
            "largest_query" => {
                let entity_type = entities.first()?;
                let target = entities.get(1)?;
                Some(format!(
                    "oxigraph_entity(Entity, '{}', Properties), member(type-'{}', Properties), member(size_rank-'1', Properties)",
                    escape_atom(entity_type),
                    escape_atom(target)
                ))
            }
            "quantity_query" => {
                let entity = entities.first()?;
                let unit = entities.get(1)?;
                Some(format!(
                    "oxigraph_entity('{}', 'biologicalconcept', Properties), member(quantity-Quantity, Properties), member(unit-'{}', Properties)",
                    escape_atom(entity),
                    escape_atom(unit)
                ))
            }
            "concept_query" => first_entity(entities).map(|concept| {
                format!(
                    "oxigraph_entity('{}', Type, Properties)",
                    escape_atom(concept)
                )
            }),
            "property_query" => {
                if entities.len() >= 2 {
                    // 为基础知识属性查询生成通用查询，调整实体顺序
                    Some(format!(
                        "oxigraph_relation('{}', Property, '{}')",
                        escape_atom(&entities[1]),
                        escape_atom(&entities[0])
                    ))
                } else {
                    // 单实体查询，查找所有属性
                    first_entity(entities).map(|entity| {
                        format!(
                            "oxigraph_entity('{}', Type, Properties)",
                            escape_atom(entity)
                        )
                    })
                }
            }
            _ => None,
        }
    }
}

/// 转义 Prolog 单引号原子中的特殊字符，避免名称里的引号/反斜杠破坏事实或查询语法
pub(crate) fn escape_atom(text: &str) -> std::borrow::Cow<'_, str> {
    // 常见情况直接借用原字符串
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

/// Scryer 值 → 统一类型化绑定（字符串渲染约定只在 [`BoundTerm::render_raw`] 中体现）。
pub fn bound_from_value(value: &Value) -> BoundTerm {
    match value {
        Value::Integer(_) | Value::Float(_) => {
            BoundTerm::classify(&prolog_value_to_json_string(value.clone()))
        }
        Value::Rational(_) => {
            let raw = prolog_value_to_json_string(value.clone());
            match parse_fraction(&raw) {
                Some(value) => BoundTerm::number(raw, value),
                None => BoundTerm::text(raw),
            }
        }
        Value::Atom(atom) => BoundTerm::text(format!("{}", atom.as_str())),
        Value::String(text) => BoundTerm::text(text),
        Value::List(items) => BoundTerm::List {
            items: items.iter().map(bound_from_value).collect(),
        },
        Value::Structure(name, fields) => BoundTerm::Struct {
            name: format!("{}", name.as_str()),
            fields: fields.iter().map(bound_from_value).collect(),
        },
        Value::Var => BoundTerm::Unbound,
    }
}

/// 解析 "n/d" 形式的有理数（Scryer Rational 的文本形式）。
fn parse_fraction(raw: &str) -> Option<f64> {
    let (numerator, denominator) = raw.split_once('/')?;
    let numerator: f64 = numerator.trim().parse().ok()?;
    let denominator: f64 = denominator.trim().parse().ok()?;
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

/// 差量装载的撤回粒度：实体按 `(名称, 类型)`，关系按三元组。
///
/// 枚举声明顺序同时决定 `retractall` 目标的写入顺序（先实体、后关系），
/// 与字段字典序一起保证装载过程可复现。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum FactIdentity {
    Entity {
        name: String,
        entity_type: String,
    },
    Relation {
        from: String,
        rel_type: String,
        to: String,
    },
}

impl PrologFact {
    fn identity(&self) -> FactIdentity {
        match self {
            PrologFact::Entity {
                name, entity_type, ..
            } => FactIdentity::Entity {
                name: name.clone(),
                entity_type: entity_type.clone(),
            },
            PrologFact::Relation { from, rel_type, to } => FactIdentity::Relation {
                from: from.clone(),
                rel_type: rel_type.clone(),
                to: to.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_queries_keep_their_public_templates() {
        let one = vec!["甲".to_string()];
        let two = vec!["甲".to_string(), "乙".to_string()];
        let generate = PrologEngine::generate_prolog_query;

        assert_eq!(
            generate("age_query", &one).as_deref(),
            Some("age_of('甲', Age)")
        );
        assert_eq!(
            generate("relation_query", &two).as_deref(),
            Some("related('甲', Relation, '乙')")
        );
        assert_eq!(
            generate("count_query", &one).as_deref(),
            Some("count_entities('甲', Count)")
        );
        assert_eq!(
            generate("capital_query", &one).as_deref(),
            Some(
                "oxigraph_entity('甲', 'country', Properties), member(capital-Capital, Properties)"
            )
        );
        assert_eq!(
            generate("altitude_query", &one).as_deref(),
            Some("oxigraph_entity('甲', 'geographicfeature', Properties), member(altitude-Altitude, Properties)")
        );
        assert_eq!(
            generate("largest_query", &two).as_deref(),
            Some("oxigraph_entity(Entity, '甲', Properties), member(type-'乙', Properties), member(size_rank-'1', Properties)")
        );
        assert_eq!(
            generate("property_query", &two).as_deref(),
            Some("oxigraph_relation('乙', Property, '甲')")
        );
        assert_eq!(
            generate("property_query", &one).as_deref(),
            Some("oxigraph_entity('甲', Type, Properties)")
        );
        assert!(generate("unknown_query", &one).is_none());
        assert!(generate("relation_query", &one).is_none());
    }

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

    #[test]
    fn test_query_typed_exposes_numbers_and_unbound() {
        let mut engine = PrologEngine::new();

        // 数值：value 可比较，raw 原样保留
        let typed = engine.query_typed("X is 1 + 2");
        assert_eq!(
            typed[0].get("X").and_then(BoundTerm::numeric_value),
            Some(3.0)
        );
        assert_eq!(
            typed[0].get("X").and_then(BoundTerm::render_raw).as_deref(),
            Some("3")
        );

        // 两个 API 一致：同一查询的绑定渲染相同
        let string_api = engine.query("X is 1 + 2");
        assert_eq!(
            typed[0].get("X").and_then(BoundTerm::render_raw),
            string_api[0].get("X").cloned()
        );

        // 文本与结构：字符串取内容本身
        let typed = engine.query_typed("atom_string(hello, S)");
        if let Some(term) = typed.first().and_then(|bindings| bindings.get("S")) {
            assert!(matches!(term, BoundTerm::Text { .. }), "{:?}", term);
            assert_eq!(term.render_raw().as_deref(), Some("hello"));
        }
    }

    #[test]
    fn test_bound_from_value_keeps_raw_rendering() {
        assert_eq!(bound_from_value(&Value::Var), BoundTerm::Unbound);
        assert_eq!(BoundTerm::Unbound.render_raw(), None);
    }

    #[test]
    fn query_checked_keeps_faults_out_of_empty_results() {
        let mut engine = PrologEngine::new();
        assert_eq!(
            engine.query_checked("entity(不存在的人, person, _)"),
            Ok(Vec::new()),
            "无解必须是 Ok(空结果)"
        );
        let fault = engine
            .query_checked("definitely_not_a_predicate_xyz(1, 2)")
            .expect_err("引擎故障必须显式返回 Err");
        assert!(fault.contains("definitely_not_a_predicate_xyz") || !fault.is_empty());
    }

    #[test]
    fn query_outcome_separates_no_solution_from_engine_fault() {
        let mut engine = PrologEngine::new();

        // 有解
        let outcome = engine.query_outcome("X is 1 + 2");
        assert!(!outcome.is_fault());
        assert!(outcome.has_solutions());
        assert!(engine.last_fault().is_none());

        // 无解：查询本身成功，只是知识里确实没有
        let outcome = engine.query_outcome("entity(不存在的人, person, _)");
        assert_eq!(outcome, QueryOutcome::NoSolution);
        assert!(!outcome.is_fault());
        assert!(engine.last_fault().is_none(), "无解不是引擎故障");

        // 引擎故障：未定义谓词必须可判、可上报、信息不销毁
        let outcome = engine.query_outcome("definitely_not_a_predicate_xyz(1, 2)");
        match &outcome {
            QueryOutcome::EngineFault(message) => assert!(!message.is_empty()),
            other => panic!("未定义谓词应报引擎故障：{:?}", other),
        }
        assert!(engine.last_fault().is_some());
        assert!(engine.take_fault().is_some());
        assert!(engine.last_fault().is_none(), "故障取走后不重复刷屏");
    }

    #[test]
    fn temporal_facts_are_queryable_with_shared_time_variable() {
        let mut engine = PrologEngine::new();
        let facts = vec![
            TemporalFact::new("张三", "师傅", "李四").with_years(1990, 1999),
            TemporalFact::new("李四", "师傅", "王五").with_years(1995, 2000),
            TemporalFact::new("王五", "师傅", "赵六").with_years(2001, 2010),
        ];
        engine.load_temporal_facts(facts);
        assert_eq!(engine.temporal_fact_count(), 3);

        // 1995 年：前两跳共享时刻，第三跳尚未开始。
        let outcome = engine.query_outcome("holds_at('张三', '师傅', '李四', 19950601)");
        assert!(
            matches!(outcome, QueryOutcome::Solutions(_)),
            "{:?}",
            outcome
        );
        let outcome = engine.query_outcome(
            "holds_at('张三', '师傅', '李四', 19950601), \
             holds_at('李四', '师傅', '王五', 19950601)",
        );
        assert!(matches!(outcome, QueryOutcome::Solutions(_)));
        let outcome = engine.query_outcome("holds_at('王五', '师傅', '赵六', 19950601)");
        assert_eq!(outcome, QueryOutcome::NoSolution);

        // 相同的 T 作为共享变量参与合一：只有 1995 年同时成立的两条边。
        let rows = engine.query("holds_at(S, P, O, 19950601)");
        assert_eq!(rows.len(), 2, "{:?}", rows);
    }

    #[test]
    fn unknown_temporal_fact_is_not_always_true() {
        let mut engine = PrologEngine::new();
        engine.load_temporal_facts(vec![TemporalFact::new("张三", "师傅", "李四")]);
        let outcome = engine.query_outcome("holds_at('张三', '师傅', '李四', 19950601)");
        assert_eq!(outcome, QueryOutcome::NoSolution);
    }

    #[test]
    fn temporal_before_and_after_queries_work() {
        let mut engine = PrologEngine::new();
        engine.load_temporal_facts(vec![
            TemporalFact::new("东京", "成为首都", "日本").with_years(1868, 1868)
        ]);
        let before = engine.query_outcome(&PrologEngine::temporal_before_query(
            "东京",
            "成为首都",
            "日本",
            TimePoint::year(1900),
        ));
        assert!(matches!(before, QueryOutcome::Solutions(_)));
        let after = engine.query_outcome("temporal_after('东京', '成为首都', '日本', 18000101)");
        assert!(matches!(after, QueryOutcome::Solutions(_)));
        let not_before =
            engine.query_outcome("temporal_before('东京', '成为首都', '日本', 18000101)");
        assert_eq!(not_before, QueryOutcome::NoSolution);
    }

    #[test]
    fn incremental_reload_asserts_and_retracts_only_deltas() {
        let mut engine = PrologEngine::new();
        let entity_v1 = PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "person".to_string(),
            properties: {
                let mut props = HashMap::new();
                props.insert("age".to_string(), "18".to_string());
                props
            },
        };
        let edge = PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "teacher_of".to_string(),
            to: "李四".to_string(),
        };
        engine
            .load_facts_checked(vec![entity_v1, edge.clone()])
            .unwrap();
        let stats = engine.fact_load_stats();
        assert_eq!(stats.full_loads, 1);
        assert_eq!(stats.incremental_loads, 0);

        // 只改属性：未变化的关系必须仍在，属性必须被替换而不是并存。
        let entity_v2 = PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "person".to_string(),
            properties: {
                let mut props = HashMap::new();
                props.insert("age".to_string(), "19".to_string());
                props
            },
        };
        engine
            .load_facts_checked(vec![entity_v2, edge.clone()])
            .unwrap();
        let stats = engine.fact_load_stats();
        assert_eq!(stats.full_loads, 1, "差量装载不应退化成全量重灌");
        assert_eq!(stats.incremental_loads, 1);
        assert_eq!((stats.last_added, stats.last_removed), (1, 1));
        let ages = engine.query("age_of('张三', Age)");
        assert_eq!(ages.len(), 1, "{:?}", ages);
        assert_eq!(ages[0].get("Age").map(String::as_str), Some("19"));
        let edges = engine.query("oxigraph_relation(From, Type, To)");
        assert_eq!(edges.len(), 1, "未变化的关系不应被误删：{:?}", edges);

        // 相同事实集重复装载：零写入、零重复。
        engine
            .load_facts_checked(vec![
                PrologFact::Entity {
                    name: "张三".to_string(),
                    entity_type: "person".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("age".to_string(), "19".to_string());
                        props
                    },
                },
                edge,
            ])
            .unwrap();
        let stats = engine.fact_load_stats();
        assert_eq!(stats.incremental_loads, 1);
        assert_eq!((stats.last_added, stats.last_removed), (0, 0));

        // 撤回关系：精确 retract，不残留。
        engine.load_facts_checked(Vec::new()).unwrap();
        let stats = engine.fact_load_stats();
        assert_eq!(stats.incremental_loads, 2);
        assert_eq!(stats.last_removed, 2);
        let outcome = engine.query_outcome("oxigraph_relation(_, _, _)");
        assert_eq!(outcome, QueryOutcome::NoSolution);
    }

    #[test]
    fn incremental_reload_preserves_duplicate_relation_multiplicity() {
        let mut engine = PrologEngine::new();
        let edge = PrologFact::Relation {
            from: "A".to_string(),
            rel_type: "knows".to_string(),
            to: "B".to_string(),
        };
        engine
            .load_facts_checked(vec![edge.clone(), edge.clone()])
            .unwrap();
        let count = engine.query("findall(To, oxigraph_relation('A', knows, To), Toes), length(Toes, N)");
        assert_eq!(count[0].get("N").map(String::as_str), Some("2"));

        engine.load_facts_checked(vec![edge]).unwrap();
        let count = engine.query("findall(To, oxigraph_relation('A', knows, To), Toes), length(Toes, N)");
        assert_eq!(
            count[0].get("N").map(String::as_str),
            Some("1"),
            "撤回重复事实后应只剩目标条数"
        );
    }

    #[test]
    fn engine_crash_is_contained_and_reported() {
        let mut engine = PrologEngine::new();
        // scryer 0.9.x 对坏语法会在 consult 中 panic：必须被兜住并转成故障
        let error = engine
            .consult_source("this is (((not prolog")
            .expect_err("坏语法不得静默通过");
        assert!(error.contains("崩溃"), "{}", error);
        assert!(engine.is_poisoned());

        // 崩溃后的实例：一律 EngineFault，绝不伪装成无解
        let outcome = engine.query_outcome("X is 1 + 2");
        assert!(outcome.is_fault(), "{:?}", outcome);
        assert!(!matches!(outcome, QueryOutcome::NoSolution));
        assert!(engine.last_fault().is_some());
        assert!(
            engine.consult_source("p(a).").is_err(),
            "停止服务的引擎不接受事实"
        );
    }
}

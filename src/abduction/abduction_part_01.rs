// 溯因推理与假设生成引擎（纯符号，无神经网络、无权重训练）
//
// 灵感：后向链推理 + 假设生成。当目标（如某条待判定的陈述、某个开放问句）
// 无法由现有知识直接证明时，本模块做目标制导的后向链展开，沿 Horn 规则
// 逐层推进，收集"失败前沿"——缺失的基事实或未绑定的取值；再把前沿原子
// 变成候选假设（溯因 abduction），用纯计数的可信度模型排序：
//   • 谓词-取值先验：图谱中同谓词取值的出现频次；
//   • 类型匹配：主语/宾语类型与该谓词观察到的类型分布是否相容；
//   • 特征重叠：候选实体与主体共享的关系特征（Jaccard）；
//   • 功能谓词冲突：若同主语已有不同取值，则按封闭世界假设重罚。
// 每个假设都附带可回放的理由与冲突说明；还可以做"如果假设成立"的反事实
// 前向模拟（临时加入事实、正向饱和、自动回滚），由此得到可解释的
// "如果……那么……"式生成。全部过程零神经网络，规则、检索与组合生成。
use std::collections::{HashMap, HashSet};
use crate::prolog_engine::PrologFact;
use crate::resolution_engine::{
    apply_atom, freshen_clause, Atom, DerivedFact, ProofNode, ProofSource, ResolutionEngine,
    Solution, Substitution, Term,
};
/// 假设的来源类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisKind {
    /// 缺失的基事实：直接补全该前提即可证明目标
    MissingFact,
    /// 候选取值：为开放变量提议的常量（来自同谓词取值/同类型/特征相似实体）
    CandidateValue,
}
impl HypothesisKind {
    pub fn label(self) -> &'static str {
        match self {
            HypothesisKind::MissingFact => "缺失事实",
            HypothesisKind::CandidateValue => "候选取值",
        }
    }
}
/// 一条候选假设：若它成立，则目标的后向链可以继续/完成
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// 假设补全的基事实（含变量时不可采纳，仅作方向提示）
    pub atom: Atom,
    /// 触发该假设的失败前沿原子（展开前的形式，可能含变量）
    pub goal: Atom,
    pub kind: HypothesisKind,
    /// 可信度（纯计数评分，0~1）
    pub confidence: f64,
    /// 支持理由（可回放）
    pub reasons: Vec<String>,
    /// 与现有知识冲突的说明（空即无冲突）
    pub conflicts: Vec<String>,
    /// 是否与已知事实硬冲突（如功能谓词取值矛盾）：此类假设不进入联合解释场景
    pub contradicts_known: bool,
}
impl Hypothesis {
    pub fn is_adoptable(&self) -> bool {
        self.atom.is_ground()
    }
}
/// 溯因分析报告：目标是否可证、失败前沿与候选假设
#[derive(Debug, Clone)]
pub struct AbductionReport {
    pub goal: Atom,
    /// 若为真：目标已可证，solutions 非空
    pub provable: bool,
    pub solutions: Vec<Solution>,
    /// 后向链失败前沿：推不出时缺少的前提
    pub frontier: Vec<Atom>,
    /// 由前沿生成的候选假设（按可信度降序，已截断）
    pub hypotheses: Vec<Hypothesis>,
    pub notes: Vec<String>,
}
impl AbductionReport {
    pub fn best_hypothesis(&self) -> Option<&Hypothesis> {
        self.hypotheses
            .iter()
            .filter(|hypothesis| hypothesis.is_adoptable())
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}
/// 反事实模拟结果
#[derive(Debug, Clone)]
pub struct WhatIfResult {
    /// 临时假设（模拟用，不写入知识库）
    pub assumptions: Vec<Atom>,
    /// 由假设+现有规则正向推导出的新事实（含前提与规则来源）
    pub derived: Vec<DerivedFact>,
    /// 去重后的新事实（排除原知识库已有）
    pub new_facts: Vec<Atom>,
}
/// 联合解释场景：一组假设"协同成立"时目标可被后向链证明
///
/// 与单条假设不同，场景回答的是"哪些事实一起补齐才能解释目标"，
/// 相当于符号版的"多步推测 + 验证"：先枚举假设组合，再注入克隆引擎
/// 做 SLD 反驳验证，只有真正证明目标的组合才会被保留。
#[derive(Debug, Clone)]
pub struct Scenario {
    /// 场景包含的假设（均为基事实，按可信度降序）
    pub hypotheses: Vec<Hypothesis>,
    /// 注入这些假设后目标是否可证（SLD 反驳成功）
    pub verified: bool,
    /// 联合可信度：成员可信度均值 × 规模折扣（假设越多越保守）
    pub joint_confidence: f64,
    /// 验证证明（verified 时非空）
    pub solutions: Vec<Solution>,
}
impl Scenario {
    pub fn size(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn atoms(&self) -> Vec<Atom> {
        self.hypotheses
            .iter()
            .map(|hypothesis| hypothesis.atom.clone())
            .collect()
    }
}
/// 解释空间：当前目标在有界枚举内所有通过 SLD 验证的解释（场景）集合。
///
/// 这是**派生状态**：确认/否定只改变兼容性过滤，不写知识图谱；会话快照只
/// 保存用户承诺（确认/否定/暂缓），恢复后按当前知识重新计算，避免陈旧解释。
#[derive(Debug, Clone, Default)]
pub struct ExplanationSpace {
    explanations: Vec<Scenario>,
    /// 生成这些解释时看到的失败前沿（用于解释空间崩塌时说明缺少什么）
    frontier: Vec<Atom>,
    /// 枚举是否受假设池/组合预算/展示上限截断，可能漏解
    bounded: bool,
}
/// 一个答案分组：同一答案可能由多组解释分别支撑
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplanationAnswer {
    pub answer: String,
    pub scenario_indices: Vec<usize>,
}
impl ExplanationSpace {
    pub fn new(explanations: Vec<Scenario>, frontier: Vec<Atom>, bounded: bool) -> Self {
        Self {
            explanations: explanations
                .into_iter()
                .filter(|scenario| scenario.verified)
                .collect(),
            frontier,
            bounded,
        }
    }

    pub fn explanations(&self) -> &[Scenario] {
        &self.explanations
    }

    pub fn frontier(&self) -> &[Atom] {
        &self.frontier
    }

    pub fn len(&self) -> usize {
        self.explanations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.explanations.is_empty()
    }

    pub fn is_bounded(&self) -> bool {
        self.bounded
    }

    /// 某假设对解释空间的划分：(含它的解释数, 不含它的解释数)。
    pub fn split(&self, hypothesis: &Hypothesis) -> (usize, usize) {
        let with = self
            .explanations
            .iter()
            .filter(|scenario| {
                scenario
                    .hypotheses
                    .iter()
                    .any(|member| member.atom == hypothesis.atom)
            })
            .count();
        (with, self.explanations.len().saturating_sub(with))
    }

    /// 删除包含被否定假设的解释：用户否定的不只是某条前提，而是所有依赖它的解释。
    pub fn retain_compatible(&mut self, rejected: &[Hypothesis]) {
        if rejected.is_empty() {
            return;
        }
        self.explanations.retain(|scenario| {
            !scenario.hypotheses.iter().any(|member| {
                rejected
                    .iter()
                    .any(|rejected| rejected.atom == member.atom)
            })
        });
    }

    /// 按答案分组（保序）：用于判断答案在所有剩余解释下是否一致。
    pub fn answer_groups(&self, query_vars: &[String]) -> Vec<ExplanationAnswer> {
        let mut groups: Vec<ExplanationAnswer> = Vec::new();
        for (index, scenario) in self.explanations.iter().enumerate() {
            let answer = Self::scenario_answer(scenario, query_vars);
            match groups.iter_mut().find(|group| group.answer == answer) {
                Some(group) => group.scenario_indices.push(index),
                None => groups.push(ExplanationAnswer {
                    answer,
                    scenario_indices: vec![index],
                }),
            }
        }
        groups
    }

    /// 单个解释给出的答案；无查询变量时表示"目标成立"。
    ///
    /// 一个解释可能包含多条证明（同一组前提下有多个答案），此时用「或」连接，
    /// 避免只展示第一条解而谎称整个解释空间已经收敛到单一答案。
    pub fn scenario_answer(scenario: &Scenario, query_vars: &[String]) -> String {
        if query_vars.is_empty() {
            return "目标成立".to_string();
        }
        let mut answers: Vec<String> = Vec::new();
        for solution in &scenario.solutions {
            let mut parts: Vec<String> = Vec::new();
            for var in query_vars {
                if let Some(value) = solution.bindings.get(var) {
                    if value.is_empty() {
                        continue;
                    }
                    if var == "答案" {
                        parts.push(value.clone());
                    } else {
                        parts.push(format!("{}={}", var, value));
                    }
                }
            }
            let answer = if parts.is_empty() {
                "目标成立".to_string()
            } else {
                parts.join("、")
            };
            if !answers.contains(&answer) {
                answers.push(answer);
            }
        }
        if answers.is_empty() {
            "目标成立".to_string()
        } else {
            answers.join(" 或 ")
        }
    }

    /// 解释的前提描述：多条假设用"且"连接；同一答案的多个解释由调用方用"或"连接。
    pub fn scenario_premise(
        scenario: &Scenario,
        render: &dyn Fn(&Atom) -> String,
    ) -> String {
        scenario
            .hypotheses
            .iter()
            .map(|hypothesis| render(&hypothesis.atom))
            .collect::<Vec<_>>()
            .join(" 且 ")
    }
}
/// 渲染解释空间结论：收敛时给稳健答案，多答案时给条件化回答，空空间时给崩塌诊断。
pub fn render_explanation_conclusion(
    space: &ExplanationSpace,
    query_vars: &[String],
    render: &dyn Fn(&Atom) -> String,
) -> String {
    if space.is_empty() {
        let mut output = String::from(
            "【解释空间崩塌】有界枚举内曾存在的全部解释都已被当前回答排除。\n  这不是“知识库查无”，而是你的回答与所有可验证解释冲突。\n",
        );
        if !space.frontier().is_empty() {
            let frontier = space
                .frontier()
                .iter()
                .map(render)
                .collect::<Vec<_>>()
                .join("；");
            output.push_str(&format!("  仍然缺少的前提：{}\n", frontier));
        }
        output.push_str("  可补充新事实后重新提问，或输入「结束澄清」查看已排除清单。");
        return output;
    }

    let groups = space.answer_groups(query_vars);
    let mut output = String::new();
    if groups.len() == 1 {
        if query_vars.is_empty() {
            output.push_str(&format!(
                "【解释空间收敛】目标很可能成立，并且在剩余 {} 组解释下都成立。\n",
                space.len()
            ));
        } else {
            output.push_str(&format!(
                "【解释空间收敛】答案很可能是「{}」，并且在剩余 {} 组解释下都成立。\n",
                groups[0].answer,
                space.len()
            ));
        }
        for (order, index) in groups[0].scenario_indices.iter().enumerate() {
            let scenario = &space.explanations()[*index];
            output.push_str(&format!(
                "  解释 {}（可信度 {:.2}）：{}\n",
                order + 1,
                scenario.joint_confidence,
                ExplanationSpace::scenario_premise(scenario, render)
            ));
        }
    } else {
        output.push_str(
            "【条件化回答】不同解释会给出不同答案；不必确认完所有前提也能先看条件结论：\n",
        );
        for group in groups {
            let premises = group
                .scenario_indices
                .iter()
                .map(|index| {
                    ExplanationSpace::scenario_premise(&space.explanations()[*index], render)
                })
                .collect::<Vec<_>>()
                .join(" 或 ");
            if query_vars.is_empty() {
                output.push_str(&format!("  若 {}，则目标成立；\n", premises));
            } else {
                output.push_str(&format!(
                    "  若 {}，则答案是「{}」；\n",
                    premises, group.answer
                ));
            }
        }
    }
    if space.is_bounded() {
        output.push_str(
            "  说明：解释枚举有界（假设池 10 条、组合预算 60 次、展示上限 5 组），以上不是完备枚举。\n",
        );
    }
    output.trim_end().to_string()
}
/// 澄清回答的语义类别（纯模式匹配，无神经网络）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerKind {
    /// 肯定：是的 / 对 / 嗯 / 确认 …
    Yes,
    /// 否定：不是 / 不对 / 没有 …
    No,
    /// 不确定：不知道 / 也许 / 跳过 …
    Unknown,
    /// 无法识别
    Unrecognized,
}
/// 「全部确认 / 全部否定」这类批量回答的作用域。
///
/// 只有同时出现整体量词（都 / 全 / 全部）才算批量，普通「不是」仍然只否定当前一条。
pub fn clarification_batch_scope(text: &str) -> Option<AnswerKind> {
    let trimmed = text
        .trim()
        .trim_start_matches(['：', ':', ' '])
        .trim_end_matches(['。', '！', '!', '~', '～', ' ', '？', '?']);
    if trimmed.is_empty() {
        return None;
    }
    const REJECT: &[&str] = &[
        "全都不", "全不是", "都不", "都不对", "都不是", "全都不是", "全部不是", "全部不", "都不成立",
        "全部成立不", "都否", "全否", "全都否", "全部否", "都否定", "全部否定", "全都否定", "都错",
        "全错", "都排除", "全部排除", "全都排除",
    ];
    const CONFIRM: &[&str] = &[
        "全都确认", "全部确认", "全确认", "都确认", "全都对", "全部对", "全对", "都对", "都是",
        "全部是", "全都是", "全是的", "都成立", "全部成立", "全都成立", "都采纳", "全采纳",
        "全都采纳", "全部采纳",
    ];
    if REJECT.iter().any(|phrase| trimmed.contains(phrase)) {
        return Some(AnswerKind::No);
    }
    if CONFIRM.iter().any(|phrase| trimmed.contains(phrase)) {
        return Some(AnswerKind::Yes);
    }
    None
}
/// 把用户对澄清提问的回答分类为 是 / 不是 / 不确定
pub fn classify_answer(text: &str) -> AnswerKind {
    let trimmed = text
        .trim()
        .trim_start_matches(['：', ':', ' '])
        .trim_end_matches(['。', '！', '!', '~', '～', ' ', '？', '?']);
    if trimmed.is_empty() {
        return AnswerKind::Unrecognized;
    }
    let keywords: [(&str, AnswerKind); 37] = [
        ("不知道", AnswerKind::Unknown),
        ("不确定", AnswerKind::Unknown),
        ("不清楚", AnswerKind::Unknown),
        ("不晓得", AnswerKind::Unknown),
        ("下一个", AnswerKind::Unknown),
        ("跳过", AnswerKind::Unknown),
        ("也许", AnswerKind::Unknown),
        ("或许", AnswerKind::Unknown),
        ("可能", AnswerKind::Unknown),
        ("随便", AnswerKind::Unknown),
        ("不是", AnswerKind::No),
        ("不对", AnswerKind::No),
        ("没有", AnswerKind::No),
        ("没错", AnswerKind::Yes),
        ("确认", AnswerKind::Yes),
        ("正确", AnswerKind::Yes),
        ("是的", AnswerKind::Yes),
        ("对", AnswerKind::Yes),
        ("嗯", AnswerKind::Yes),
        ("好", AnswerKind::Yes),
        ("是", AnswerKind::Yes),
        ("有", AnswerKind::Yes),
        ("否", AnswerKind::No),
        ("错", AnswerKind::No),
        ("没", AnswerKind::No),
        ("无", AnswerKind::No),
        ("不", AnswerKind::No),
        ("yes", AnswerKind::Yes),
        ("y", AnswerKind::Yes),
        ("true", AnswerKind::Yes),
        ("no", AnswerKind::No),
        ("n", AnswerKind::No),
        ("false", AnswerKind::No),
        ("重问", AnswerKind::Unrecognized),
        ("重说", AnswerKind::Unrecognized),
        ("什么", AnswerKind::Unrecognized),
        ("再说", AnswerKind::Unrecognized),
    ];
    let lower = trimmed.to_lowercase();
    // 最长匹配优先："不确定" 优先于 "不"，"不是" 优先于 "不"
    let mut best: Option<(usize, AnswerKind)> = None;
    for (keyword, kind) in keywords {
        let matched =
            trimmed.starts_with(keyword) || lower.starts_with(keyword) || trimmed == keyword;
        if matched {
            let length = keyword.chars().count();
            if best.map(|(current, _)| length > current).unwrap_or(true) {
                best = Some((length, kind));
            }
        }
    }
    best.map(|(_, kind)| kind)
        .unwrap_or(AnswerKind::Unrecognized)
}
/// 一次澄清对话：逐条向用户确认待定假设，并记忆确认/否定/不确定
#[derive(Debug, Clone)]
pub struct ClarificationSession {
    queue: Vec<Hypothesis>,
    current: Option<Hypothesis>,
    /// 已确认的假设
    pub confirmed: Vec<Hypothesis>,
    /// 已否定的假设
    pub rejected: Vec<Hypothesis>,
    /// 记为不确定、暂缓的假设
    pub deferred: Vec<Hypothesis>,
    /// 当前问题的竞争解释集合（派生状态，不写入快照）
    space: Option<ExplanationSpace>,
    /// 本次会话是否曾经拥有过非空解释空间（区分"从未有解释"与"全被排除"）
    had_explanations: bool,
    /// 已处理的回答轮数（含批量回答）
    asked: usize,
    /// 本轮澄清最多追问多少轮，超过后直接给条件化回答
    max_questions: usize,
}
/// 默认最大追问轮数：超过后不再骚扰用户，改为给条件化/崩塌结论。
const DEFAULT_MAX_CLARIFICATION_QUESTIONS: usize = 8;
impl ClarificationSession {
    fn pool(hypotheses: &[Hypothesis]) -> Vec<Hypothesis> {
        let mut queue: Vec<Hypothesis> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        for hypothesis in hypotheses {
            if !hypothesis.is_adoptable() {
                continue;
            }
            let key = hypothesis.atom.display();
            match seen.get(&key).copied() {
                // 同一原子可能来自不同前沿：保留可信度更高、理由更充分的那条，
                // 否则先出现的弱重复会把后面的强证据挡在澄清队列之外。
                Some(index) => {
                    let existing = &mut queue[index];
                    if hypothesis.confidence > existing.confidence
                        || (hypothesis.confidence == existing.confidence
                            && hypothesis.reasons.len() > existing.reasons.len())
                    {
                        *existing = hypothesis.clone();
                    }
                }
                None => {
                    seen.insert(key, queue.len());
                    queue.push(hypothesis.clone());
                }
            }
        }
        queue
    }

    fn sort_by_confidence(queue: &mut [Hypothesis]) {
        queue.sort_by_cached_key(|hypothesis| ConfidenceDisplayKey {
            confidence: hypothesis.confidence,
            display: hypothesis.atom.display(),
        });
    }

    /// 假说对当前目标的判别力：采纳它之后目标是否更接近可证。
    ///
    /// 2 = 单独采纳即可证明整个目标；1 = 至少能让一个子目标可证；
    /// 0 = 对当前目标没有直接推进。用确定性克隆验证，不产生任何副作用。
    fn discrimination(
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
        hypothesis: &Hypothesis,
    ) -> u8 {
        if goals.is_empty() || !hypothesis.is_adoptable() {
            return 0;
        }
        let mut trial = engine.clone();
        if !trial.add_fact_atom(hypothesis.atom.clone()) {
            return 0;
        }
        if !trial.solve(goals, query_vars).is_empty() {
            return 2;
        }
        if goals
            .iter()
            .any(|goal| !trial.solve(std::slice::from_ref(goal), &[]).is_empty())
        {
            return 1;
        }
        0
    }

    fn sort_by_goal(
        queue: &mut [Hypothesis],
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
        space: Option<&ExplanationSpace>,
    ) {
        // 判别力只算一次：排序比较器里反复克隆引擎会把澄清变成 O(n log n) 次证明。
        let gains: HashMap<String, (u8, usize, bool)> = queue
            .iter()
            .map(|hypothesis| {
                let gain = Self::discrimination(engine, goals, query_vars, hypothesis);
                let (with, without) = space
                    .map(|space| space.split(hypothesis))
                    .unwrap_or((0, 0));
                // 平衡度 = min(含它, 不含它)：越均衡，一次回答越能砍掉一半解释空间。
                // in_space 区分"在解释中出现但全含它"与"根本不在解释空间里"。
                (hypothesis.atom.display(), (gain, with.min(without), with > 0))
            })
            .collect();
        queue.sort_by(|left, right| {
            let left_key = left.atom.display();
            let right_key = right.atom.display();
            let (left_gain, left_balance, left_in_space) =
                gains.get(&left_key).copied().unwrap_or((0, 0, false));
            let (right_gain, right_balance, right_in_space) =
                gains.get(&right_key).copied().unwrap_or((0, 0, false));
            right_gain
                .cmp(&left_gain)
                .then_with(|| right_balance.cmp(&left_balance))
                .then_with(|| right_in_space.cmp(&left_in_space))
                .then_with(|| {
                    right
                        .confidence
                        .partial_cmp(&left.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left_key.cmp(&right_key))
        });
    }

    /// 用待定假设开启澄清：按可信度降序逐条提问
    pub fn start(hypotheses: &[Hypothesis]) -> Self {
        let mut queue = Self::pool(hypotheses);
        Self::sort_by_confidence(&mut queue);
        let mut session = Self {
            queue,
            current: None,
            confirmed: Vec::new(),
            rejected: Vec::new(),
            deferred: Vec::new(),
            space: None,
            had_explanations: false,
            asked: 0,
            max_questions: DEFAULT_MAX_CLARIFICATION_QUESTIONS,
        };
        session.advance();
        session
    }

    /// 带目标的澄清：优先问“采纳后最能区分/证明当前目标”的假设，
    /// 可信度只作为同信息量时的排序依据。目标为空时退化为 [`Self::start`]。
    pub fn start_with_goal(
        hypotheses: &[Hypothesis],
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) -> Self {
        let mut queue = Self::pool(hypotheses);
        if goals.is_empty() {
            Self::sort_by_confidence(&mut queue);
        } else {
            Self::sort_by_goal(&mut queue, engine, goals, query_vars, None);
        }
        let mut session = Self {
            queue,
            current: None,
            confirmed: Vec::new(),
            rejected: Vec::new(),
            deferred: Vec::new(),
            space: None,
            had_explanations: false,
            asked: 0,
            max_questions: DEFAULT_MAX_CLARIFICATION_QUESTIONS,
        };
        session.advance();
        session
    }

    /// 挂载当前目标的解释空间；有解释时立即按"能砍掉多少解释"重排问题。
    pub fn attach_explanation_space(&mut self, space: ExplanationSpace) {
        self.had_explanations = !space.is_empty();
        self.space = Some(space);
    }

    /// 用最新会话世界刷新解释空间：保留用户已确认/否定的承诺，删除不兼容解释。
    pub fn refresh_explanation_space(&mut self, mut space: ExplanationSpace) {
        space.retain_compatible(&self.rejected);
        self.had_explanations |= !space.is_empty();
        self.space = Some(space);
    }

    pub fn explanation_space(&self) -> Option<&ExplanationSpace> {
        self.space.as_ref()
    }

    /// 把解释空间中新出现、尚未问过的前提并入候选队列。
    ///
    /// 解释空间每轮都会按当前知识重算，新场景可能引入原先不在待定池里的前提；
    /// 不把它们纳入队列，系统就只能在一组过时的候选上做判别，无法真正收敛。
    pub fn include_explanation_hypotheses(&mut self) -> usize {
        let Some(space) = &self.space else {
            return 0;
        };
        let mut seen: HashSet<String> = self
            .queue
            .iter()
            .chain(self.current.iter())
            .chain(self.confirmed.iter())
            .chain(self.rejected.iter())
            .chain(self.deferred.iter())
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        let mut additions: Vec<Hypothesis> = Vec::new();
        for scenario in space.explanations() {
            for hypothesis in &scenario.hypotheses {
                if !hypothesis.is_adoptable() {
                    continue;
                }
                if seen.insert(hypothesis.atom.display()) {
                    additions.push(hypothesis.clone());
                }
            }
        }
        let count = additions.len();
        self.queue.extend(additions);
        count
    }

    /// 解释空间是否曾经存在、现在却被全部排除。
    pub fn explanation_collapsed(&self) -> bool {
        self.had_explanations && self.space.as_ref().map(ExplanationSpace::is_empty).unwrap_or(false)
    }

    pub fn questions_asked(&self) -> usize {
        self.asked
    }

    pub fn max_questions(&self) -> usize {
        self.max_questions
    }

    pub fn max_questions_reached(&self) -> bool {
        self.asked >= self.max_questions
    }

    /// 结束后给出解释空间结论（收敛答案 / 条件化答案 / 崩塌诊断）。
    pub fn explanation_conclusion(
        &self,
        query_vars: &[String],
        render: &dyn Fn(&Atom) -> String,
    ) -> Option<String> {
        let space = self.space.as_ref()?;
        if space.is_empty() && !self.had_explanations {
            return None;
        }
        Some(render_explanation_conclusion(space, query_vars, render))
    }

    /// 一次回答之后按最新世界重排剩余问题：已采纳的事实时常改变最优提问。
    pub fn reprioritize(
        &mut self,
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) {
        if goals.is_empty() {
            return;
        }
        // 当前已取出的下一问必须回到候选池，否则重排会把这条问题丢掉。
        if let Some(current) = self.current.take() {
            self.queue.push(current);
        }
        let space = self.space.clone();
        Self::sort_by_goal(&mut self.queue, engine, goals, query_vars, space.as_ref());
        self.advance();
    }

    pub fn current(&self) -> Option<&Hypothesis> {
        self.current.as_ref()
    }

    pub fn remaining(&self) -> usize {
        self.queue.len() + usize::from(self.current.is_some())
    }

    pub fn is_finished(&self) -> bool {
        self.current.is_none() && self.queue.is_empty()
    }

    fn advance(&mut self) {
        // 队列按可信度降序排列，取最前面的问题
        self.current = if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        };
    }

    /// 把一条尚未写入的假设重新设为当前问题：预检被拒绝时保留问题并再次展示，
    /// 避免会话已推进到下一问、用户却在回答一个从未看到的提问。
    pub fn retry(&mut self, hypothesis: Hypothesis) {
        if let Some(next) = self.current.take() {
            self.queue.insert(0, next);
        }
        self.current = Some(hypothesis);
    }

    /// 批量写入被预检整体拒绝时，把整批假设按原顺序恢复为待问状态。
    pub fn restore_batch(&mut self, batch: Vec<Hypothesis>) {
        let mut iter = batch.into_iter();
        let Some(first) = iter.next() else {
            return;
        };
        if let Some(next) = self.current.take() {
            self.queue.insert(0, next);
        }
        self.current = Some(first);
        for hypothesis in iter.collect::<Vec<_>>().into_iter().rev() {
            self.queue.insert(0, hypothesis);
        }
    }

    /// 回答里是否**肯定地**提到了某个取值（排除「不是X」「并非X」这类否定出现）。
    fn mentions_affirmed(text: &str, value: &str) -> bool {
        if value.is_empty() {
            return false;
        }
        let mut cursor = 0usize;
        while let Some(offset) = text[cursor..].find(value) {
            let start = cursor + offset;
            let before = &text[..start];
            let negated = ["不是", "并非", "没有", "不", "没", "非"]
                .iter()
                .any(|prefix| before.ends_with(prefix));
            if !negated {
                return true;
            }
            cursor = start + value.len();
        }
        false
    }

    /// 从队列里找出用户更正到的替代假设：同谓词、同元数，且只有一个常量不同，
    /// 且该常量在回答中被肯定地提到。多个匹配时取最长取值，避免「李」抢先于「李四」。
    fn correction_target(&self, text: &str, current: &Hypothesis) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (index, candidate) in self.queue.iter().enumerate() {
            if candidate.atom.predicate != current.atom.predicate
                || candidate.atom.terms.len() != current.atom.terms.len()
            {
                continue;
            }
            let mut replacements: Vec<(&str, &str)> = Vec::new();
            let mut compatible = true;
            for (old, new) in current.atom.terms.iter().zip(&candidate.atom.terms) {
                if old == new {
                    continue;
                }
                match new {
                    Term::Const(value) => replacements.push((old.name(), value)),
                    Term::Var(_) => {
                        compatible = false;
                        break;
                    }
                }
            }
            if !compatible || replacements.is_empty() {
                continue;
            }
            for (old_value, new_value) in replacements {
                if old_value == new_value || !Self::mentions_affirmed(text, new_value) {
                    continue;
                }
                // 回答同时肯定旧值时无法确定用户到底选哪个，宁可回退到普通是否判定。
                if Self::mentions_affirmed(text, old_value) {
                    continue;
                }
                let length = new_value.chars().count();
                if best.map(|(_, best_len)| length > best_len).unwrap_or(true) {
                    best = Some((index, length));
                }
            }
        }
        best.map(|(index, _)| index)
    }

    /// 处理用户回答：返回该回答的结论，并推进到下一问。
    ///
    /// 除了是 / 不是 / 不知道，还支持从当前候选里选择更正取值：
    /// 「不是，是王五」会否定当前假设并直接确认队列中的「王五」假设。
    pub fn respond(&mut self, text: &str) -> ClarificationReply {
        let kind = classify_answer(text);
        let Some(current) = self.current.take() else {
            return ClarificationReply::Finished;
        };
        if kind != AnswerKind::Unrecognized {
            self.asked += 1;
        }
        // 更正优先于批量：「都不是，是王五」仍按更正处理，不能误否定全部。
        if kind != AnswerKind::Unknown {
            if let Some(index) = self.correction_target(text, &current) {
                let confirmed = self.queue.remove(index);
                self.rejected.push(current.clone());
                self.confirmed.push(confirmed.clone());
                self.advance();
                return ClarificationReply::Corrected {
                    rejected: current,
                    confirmed,
                };
            }
        }
        // 批量回答只在还有多条待问时生效；只剩一条时按普通是/不是处理。
        if !self.queue.is_empty() {
            if let Some(scope) = clarification_batch_scope(text) {
                let mut batch = vec![current];
                batch.append(&mut self.queue);
                return match scope {
                    AnswerKind::Yes => {
                        self.confirmed.extend(batch.iter().cloned());
                        ClarificationReply::ConfirmedAll(batch)
                    }
                    AnswerKind::No => {
                        self.rejected.extend(batch.iter().cloned());
                        ClarificationReply::RejectedAll(batch)
                    }
                    _ => ClarificationReply::Unrecognized,
                };
            }
        }
        match kind {
            AnswerKind::Yes => {
                self.confirmed.push(current.clone());
                self.advance();
                ClarificationReply::Confirmed(current)
            }
            AnswerKind::No => {
                self.rejected.push(current.clone());
                self.advance();
                ClarificationReply::Rejected(current)
            }
            AnswerKind::Unknown => {
                self.deferred.push(current.clone());
                self.advance();
                ClarificationReply::Deferred(current)
            }
            AnswerKind::Unrecognized => {
                // 无法识别：保留当前问题，等待用户再说一次
                self.current = Some(current);
                ClarificationReply::Unrecognized
            }
        }
    }

    /// 计数摘要（用于对话记忆）
    pub fn progress_text(&self) -> String {
        let space = match &self.space {
            Some(space) if !space.is_empty() => format!(
                "、解释空间 {} 组{}",
                space.len(),
                if space.is_bounded() { "（有界）" } else { "" }
            ),
            Some(_) if self.had_explanations => "、解释空间已崩塌".to_string(),
            _ => String::new(),
        };
        format!(
            "已确认 {} 条、否定 {} 条、不确定 {} 条、待问 {} 条{}",
            self.confirmed.len(),
            self.rejected.len(),
            self.deferred.len(),
            self.remaining(),
            space
        )
    }

    /// 尚未提问的队列（快照导出用；保持原有顺序）。
    pub fn queue(&self) -> &[Hypothesis] {
        &self.queue
    }

    /// 从快照部件重建会话（快照恢复用；不做重新排序，保持中断时的问题顺序）。
    pub fn from_parts(
        queue: Vec<Hypothesis>,
        current: Option<Hypothesis>,
        confirmed: Vec<Hypothesis>,
        rejected: Vec<Hypothesis>,
        deferred: Vec<Hypothesis>,
    ) -> Self {
        let asked = confirmed.len() + rejected.len() + deferred.len();
        Self {
            queue,
            current,
            confirmed,
            rejected,
            deferred,
            space: None,
            had_explanations: false,
            asked,
            max_questions: DEFAULT_MAX_CLARIFICATION_QUESTIONS,
        }
    }
}
/// 澄清回答的处理结论
#[derive(Debug, Clone)]
pub enum ClarificationReply {
    Confirmed(Hypothesis),
    Rejected(Hypothesis),
    Deferred(Hypothesis),
    /// 用户否定了当前假设，并从候选中更正到另一条取值
    Corrected {
        rejected: Hypothesis,
        confirmed: Hypothesis,
    },
    /// 批量确认：当前问题与剩余队列一次性全部确认
    ConfirmedAll(Vec<Hypothesis>),
    /// 批量否定：当前问题与剩余队列一次性全部否定
    RejectedAll(Vec<Hypothesis>),
    Unrecognized,
    Finished,
}
/// 溯因引擎：统计知识分布 + 失败前沿搜索 + 假设生成与排序 + 反事实模拟
#[derive(Clone)]
pub struct AbductionEngine {
    max_depth: usize,
    max_hypotheses: usize,
    max_frontiers: usize,
    max_scenario_size: usize,
    /// 谓词 → 宾语取值频次（含属性）
    pred_objects: HashMap<String, HashMap<String, u64>>,
    /// 谓词 → 主语类型频次
    pred_subject_types: HashMap<String, HashMap<String, u64>>,
    /// 谓词 → 宾语类型频次
    pred_object_types: HashMap<String, HashMap<String, u64>>,
    /// 谓词 → 主语 → 不同宾语集合（功能谓词/冲突检测）
    pred_subject_objects: HashMap<String, HashMap<String, HashSet<String>>>,
    /// 实体 → 类型
    entity_types: HashMap<String, String>,
    /// 实体 → 关系特征（"入/出:谓词=对方"）
    entity_features: HashMap<String, HashSet<String>>,
    /// 实体 → 参与过的谓词频次
    entity_predicates: HashMap<String, HashMap<String, u64>>,
    /// 已观察过的三元组：重复观察（快照重放 / 增量重载同一批事实）不得重复计数
    observed_triples: HashSet<String>,
    /// 本轮会话暂存的候选假设（供「采纳假设 N」）
    pending: Vec<Hypothesis>,
    /// 本轮会话已采纳的假设（加入归结事实集，但不写知识图谱）
    adopted: Vec<Atom>,
}

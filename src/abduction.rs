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
}

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
    ) {
        // 判别力只算一次：排序比较器里反复克隆引擎会把澄清变成 O(n log n) 次证明。
        let gains: HashMap<String, u8> = queue
            .iter()
            .map(|hypothesis| {
                (
                    hypothesis.atom.display(),
                    Self::discrimination(engine, goals, query_vars, hypothesis),
                )
            })
            .collect();
        queue.sort_by(|left, right| {
            let left_key = left.atom.display();
            let right_key = right.atom.display();
            let left_gain = gains.get(&left_key).copied().unwrap_or(0);
            let right_gain = gains.get(&right_key).copied().unwrap_or(0);
            right_gain
                .cmp(&left_gain)
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
            Self::sort_by_goal(&mut queue, engine, goals, query_vars);
        }
        let mut session = Self {
            queue,
            current: None,
            confirmed: Vec::new(),
            rejected: Vec::new(),
            deferred: Vec::new(),
        };
        session.advance();
        session
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
        Self::sort_by_goal(&mut self.queue, engine, goals, query_vars);
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
        format!(
            "已确认 {} 条、否定 {} 条、不确定 {} 条、待问 {} 条",
            self.confirmed.len(),
            self.rejected.len(),
            self.deferred.len(),
            self.remaining()
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
        Self {
            queue,
            current,
            confirmed,
            rejected,
            deferred,
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

impl AbductionEngine {
    pub fn new(max_depth: usize, max_hypotheses: usize) -> Self {
        Self {
            max_depth: max_depth.clamp(1, 12),
            max_hypotheses: max_hypotheses.clamp(1, 20),
            max_frontiers: 6,
            max_scenario_size: 2,
            pred_objects: HashMap::new(),
            pred_subject_types: HashMap::new(),
            pred_object_types: HashMap::new(),
            pred_subject_objects: HashMap::new(),
            entity_types: HashMap::new(),
            entity_features: HashMap::new(),
            entity_predicates: HashMap::new(),
            observed_triples: HashSet::new(),
            pending: Vec::new(),
            adopted: Vec::new(),
        }
    }

    /// 设置联合解释场景的最大假设条数（1~4，默认 2）
    pub fn set_max_scenario_size(&mut self, size: usize) {
        self.max_scenario_size = size.clamp(1, 4);
    }

    pub fn max_scenario_size(&self) -> usize {
        self.max_scenario_size
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn pending(&self) -> &[Hypothesis] {
        &self.pending
    }

    pub fn adopted(&self) -> &[Atom] {
        &self.adopted
    }

    /// 观察知识：只累积计数统计，不做任何学习/权重
    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        // 第一遍：先登记实体类型——关系谓词的类型统计依赖它，
        // 避免"关系事实先于实体类型出现"时类型分布为空
        for fact in facts {
            if let PrologFact::Entity {
                name, entity_type, ..
            } = fact
            {
                if !entity_type.trim().is_empty()
                    && self.entity_types.get(name).map(String::as_str) != Some(entity_type.as_str())
                {
                    self.entity_types.insert(name.clone(), entity_type.clone());
                }
            }
        }
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    self.observe_triple(from, rel_type, to);
                }
                PrologFact::Entity {
                    name, properties, ..
                } => {
                    // 实体类型已在第一遍登记，这里无需重复写入
                    for (key, value) in properties {
                        // 关系型属性在存储层可能以完整 IRI 形式留存，跳过以免污染
                        // 谓词取值统计与候选生成（如「师傅」的候选值变成 IRI）
                        if value.starts_with("http://") || value.starts_with("https://") {
                            continue;
                        }
                        self.observe_triple(name, key, value);
                    }
                }
            }
        }
    }

    fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        // 同一三元组只观察一次：`observe_facts` 会在导入/快照重放后反复调用，
        // 重复累计会把「出现次数」变成「被观察次数」，让理由里的计数失真。
        let mut key = String::with_capacity(subject.len() + predicate.len() + object.len() + 2);
        key.push_str(subject);
        key.push('\u{1f}');
        key.push_str(predicate);
        key.push('\u{1f}');
        key.push_str(object);
        if !self.observed_triples.insert(key) {
            return;
        }
        // 已存在的计数直接自增
        match self
            .pred_objects
            .get_mut(predicate)
            .and_then(|values| values.get_mut(object))
        {
            Some(count) => *count += 1,
            None => {
                *self
                    .pred_objects
                    .entry(predicate.to_string())
                    .or_default()
                    .entry(object.to_string())
                    .or_insert(0) += 1;
            }
        }

        if let Some(t) = self.entity_types.get(subject) {
            match self.pred_subject_types.get_mut(predicate) {
                Some(types) => match types.get_mut(t) {
                    Some(count) => *count += 1,
                    None => {
                        types.insert(t.clone(), 1);
                    }
                },
                None => {
                    *self
                        .pred_subject_types
                        .entry(predicate.to_string())
                        .or_default()
                        .entry(t.clone())
                        .or_insert(0) += 1;
                }
            }
        }
        if let Some(t) = self.entity_types.get(object) {
            match self.pred_object_types.get_mut(predicate) {
                Some(types) => match types.get_mut(t) {
                    Some(count) => *count += 1,
                    None => {
                        types.insert(t.clone(), 1);
                    }
                },
                None => {
                    *self
                        .pred_object_types
                        .entry(predicate.to_string())
                        .or_default()
                        .entry(t.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        match self
            .pred_subject_objects
            .get_mut(predicate)
            .and_then(|subjects| subjects.get_mut(subject))
        {
            Some(objects) => {
                // HashSet 的 insert 需要所有权：先做借用检查
                if !objects.contains(object) {
                    objects.insert(object.to_string());
                }
            }
            None => {
                self.pred_subject_objects
                    .entry(predicate.to_string())
                    .or_default()
                    .entry(subject.to_string())
                    .or_default()
                    .insert(object.to_string());
            }
        }

        // 特征串用可复用缓冲构造
        let mut feature = String::with_capacity(predicate.len() + subject.len() + object.len() + 4);
        feature.push_str("出:");
        feature.push_str(predicate);
        feature.push('=');
        feature.push_str(object);
        match self.entity_features.get_mut(subject) {
            Some(features) => {
                if !features.contains(feature.as_str()) {
                    features.insert(feature.clone());
                }
            }
            None => {
                self.entity_features
                    .entry(subject.to_string())
                    .or_default()
                    .insert(feature.clone());
            }
        }
        feature.clear();
        feature.push_str("入:");
        feature.push_str(predicate);
        feature.push('=');
        feature.push_str(subject);
        match self.entity_features.get_mut(object) {
            Some(features) => {
                if !features.contains(feature.as_str()) {
                    features.insert(feature.clone());
                }
            }
            None => {
                self.entity_features
                    .entry(object.to_string())
                    .or_default()
                    .insert(feature.clone());
            }
        }

        match self
            .entity_predicates
            .get_mut(subject)
            .and_then(|counts| counts.get_mut(predicate))
        {
            Some(count) => *count += 1,
            None => {
                *self
                    .entity_predicates
                    .entry(subject.to_string())
                    .or_default()
                    .entry(predicate.to_string())
                    .or_insert(0) += 1;
            }
        }
        match self
            .entity_predicates
            .get_mut(object)
            .and_then(|counts| counts.get_mut(predicate))
        {
            Some(count) => *count += 1,
            None => {
                *self
                    .entity_predicates
                    .entry(object.to_string())
                    .or_default()
                    .entry(predicate.to_string())
                    .or_insert(0) += 1;
            }
        }
    }

    /// 目标制导的溯因分析：
    ///   1. 先做完整的 SLD 后向链（含规则）；可证则返回证明；
    ///   2. 不可证则枚举失败前沿（有界、去重），并从前沿生成候选假设。
    pub fn analyze(
        &self,
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) -> AbductionReport {
        let goal = goals
            .first()
            .cloned()
            .unwrap_or_else(|| Atom::unary("空目标", Term::constant("")));
        let outcome = engine.solve_outcome(goals, query_vars);
        let search_status = outcome.render_status();
        let search_truncated = outcome.is_truncated();
        if !outcome.solutions.is_empty() {
            return AbductionReport {
                goal,
                provable: true,
                solutions: outcome.solutions,
                frontier: Vec::new(),
                hypotheses: Vec::new(),
                notes: vec!["目标可由现有事实与规则证明（后向链成功）".to_string()],
            };
        }

        let mut search = FrontierSearch {
            engine,
            max_depth: self.max_depth,
            max_frontiers: self.max_frontiers,
            budget: 20_000,
            results: Vec::new(),
            seen: HashSet::new(),
            freshen_counter: 0,
        };
        let mut missing = Vec::new();
        search.run(goals, &Substitution::new(), 0, &mut missing);
        let frontier = self.select_best_frontier(search.results);

        let mut hypotheses = Vec::new();
        for atom in &frontier {
            hypotheses.extend(self.hypotheses_for(engine, atom));
        }
        hypotheses.sort_by_cached_key(|hypothesis| ConfidenceDisplayKey {
            confidence: hypothesis.confidence,
            display: hypothesis.atom.display(),
        });
        dedupe_hypotheses(&mut hypotheses);
        if hypotheses.len() > self.max_hypotheses {
            hypotheses.truncate(self.max_hypotheses);
        }

        let mut notes = vec![format!(
            "后向链在深度上限 {} 内未能证明目标，改走溯因：从失败前沿生成候选假设",
            self.max_depth
        )];
        notes.push(format!("主后向链搜索结论：{}", search_status));
        if !outcome.frontier.is_empty() {
            notes.push(format!(
                "本次搜索实际失败的叶子：{}",
                outcome
                    .frontier
                    .iter()
                    .take(4)
                    .map(|entry| entry.describe())
                    .collect::<Vec<_>>()
                    .join("；")
            ));
        }
        if search_truncated {
            notes.push(
                "主后向链搜索被截断：不能据此断言目标不存在；下方失败前沿由另一条有界搜索重建，可能与该次实际路径不同"
                    .to_string(),
            );
        }
        if frontier.is_empty() {
            notes.push("失败前沿为空（目标含深层循环或超出搜索预算）".to_string());
        }

        AbductionReport {
            goal,
            provable: false,
            solutions: Vec::new(),
            frontier,
            hypotheses,
            notes,
        }
    }

    /// 对单个失败前沿原子生成候选假设
    fn hypotheses_for(&self, engine: &ResolutionEngine, atom: &Atom) -> Vec<Hypothesis> {
        if atom.is_ground() {
            if engine.has_fact(atom) {
                return Vec::new();
            }
            let (confidence, reasons, conflicts, contradicts_known) = self.score(atom);
            return vec![Hypothesis {
                atom: atom.clone(),
                goal: atom.clone(),
                kind: HypothesisKind::MissingFact,
                confidence,
                reasons,
                conflicts,
                contradicts_known,
            }];
        }

        // 开放原子：找出第一个未绑定变量，用候选取值逐一补全
        let Some(var_index) = atom.terms.iter().position(|term| term.is_var()) else {
            return Vec::new();
        };
        let candidates = self.candidate_values(engine, atom, var_index);
        let mut hypotheses = Vec::new();
        for candidate in candidates {
            let mut terms = atom.terms.clone();
            terms[var_index] = Term::constant(&candidate);
            let hypothesis = Atom::new(&atom.predicate, terms);
            if hypothesis.is_ground() && engine.has_fact(&hypothesis) {
                continue;
            }
            let (confidence, reasons, conflicts, contradicts_known) = self.score(&hypothesis);
            hypotheses.push(Hypothesis {
                atom: hypothesis,
                goal: atom.clone(),
                kind: HypothesisKind::CandidateValue,
                confidence,
                reasons,
                conflicts,
                contradicts_known,
            });
        }

        if hypotheses.is_empty() {
            // 无法生成任何具体取值：保留开放形式的"缺失事实"作为方向提示
            hypotheses.push(Hypothesis {
                atom: atom.clone(),
                goal: atom.clone(),
                kind: HypothesisKind::MissingFact,
                confidence: 0.05,
                reasons: vec!["知识库中缺少该谓词的可参考取值，无法给出具体候选".to_string()],
                conflicts: vec!["需要外部补充事实".to_string()],
                contradicts_known: false,
            });
        }
        // 每个前沿原子最多保留 3 条候选，保证假设集多样性
        hypotheses.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hypotheses.truncate(3);
        hypotheses
    }

    /// 候选取值：同谓词已知取值 → 与另一侧同类型实体 → 特征最相似实体
    fn candidate_values(
        &self,
        engine: &ResolutionEngine,
        atom: &Atom,
        var_index: usize,
    ) -> Vec<String> {
        let mut candidates: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let predicate: &str = &atom.predicate;

        // 1) 同谓词已知取值（按频次降序）
        if let Some(counts) = self.pred_objects.get(predicate) {
            let mut values: Vec<(&String, &u64)> = counts.iter().collect();
            let cmp =
                |a: &(&String, &u64), b: &(&String, &u64)| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0));
            // 取前 6 名
            if values.len() > 6 {
                values.select_nth_unstable_by(6, cmp);
                values.truncate(6);
            }
            values.sort_by(cmp);
            for (value, _) in values.into_iter() {
                if seen.insert(value.clone()) {
                    candidates.push(value.clone());
                }
            }
        }

        // 另一侧的基（用于类型/特征对比）
        let counterpart = atom.terms.iter().enumerate().find_map(|(index, term)| {
            if index != var_index {
                constant_name(term)
            } else {
                None
            }
        });

        if let Some(base) = counterpart {
            // 2) 同类型实体（共享主体类型时最自然）
            if let Some(base_type) = self.entity_types.get(base) {
                let mut same_type: Vec<&String> = self
                    .entity_types
                    .iter()
                    .filter(|(entity, etype)| *etype == base_type && entity.as_str() != base)
                    .map(|(entity, _)| entity)
                    .collect();
                // 取字典序前 5 个
                if same_type.len() > 5 {
                    same_type.select_nth_unstable(5);
                    same_type.truncate(5);
                }
                same_type.sort();
                for entity in same_type.into_iter() {
                    if seen.insert(entity.clone()) {
                        candidates.push(entity.clone());
                    }
                }
            }

            // 3) 特征最相似实体（Jaccard 排序）
            if let Some(base_features) = self.entity_features.get(base) {
                let mut similar: Vec<(f64, &String)> = self
                    .entity_features
                    .iter()
                    .filter(|(entity, _)| entity.as_str() != base)
                    .map(|(entity, features)| {
                        let inter = features.intersection(base_features).count();
                        let union = (features.len() + base_features.len())
                            .saturating_sub(inter)
                            .max(1) as f64;
                        (inter as f64 / union, entity)
                    })
                    .filter(|(score, _)| *score > 0.0)
                    .collect();
                let cmp = |a: &(f64, &String), b: &(f64, &String)| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(b.1))
                };
                // 取前 5 名
                if similar.len() > 5 {
                    similar.select_nth_unstable_by(5, cmp);
                    similar.truncate(5);
                }
                similar.sort_by(cmp);
                for (_, entity) in similar.into_iter() {
                    if seen.insert(entity.clone()) {
                        candidates.push(entity.clone());
                    }
                }
            }
        }

        // 过滤掉已能在知识库中被证明的取值
        candidates.retain(|value| {
            let mut terms = atom.terms.clone();
            terms[var_index] = Term::constant(value);
            !engine.has_fact(&Atom::new(&atom.predicate, terms))
        });
        candidates.truncate(6);
        candidates
    }

    /// 纯计数可信度评分：先验 + 类型匹配 + 特征重叠 − 功能谓词/类型冲突
    ///
    /// 返回（可信度, 支持理由, 冲突说明, 是否与已知事实硬冲突）。
    fn score(&self, atom: &Atom) -> (f64, Vec<String>, Vec<String>, bool) {
        let mut reasons: Vec<String> = Vec::new();
        let mut conflicts: Vec<String> = Vec::new();
        let mut contradicts_known = false;
        let mut score = 0.25f64;

        let predicate: &str = &atom.predicate;
        let subject = atom.terms.first().and_then(constant_name);
        let object = atom.terms.get(1).and_then(constant_name);

        // 1) 同谓词取值先验
        if let (Some(counts), Some(object)) = (self.pred_objects.get(predicate), object) {
            let total: u64 = counts.values().sum();
            if let Some(&count) = counts.get(object) {
                let share = count as f64 / total.max(1) as f64;
                score += 0.20 * share;
                reasons.push(format!(
                    "图谱中「{}」共 {} 条记录，取值「{}」出现 {} 次（占 {:.0}%）",
                    predicate,
                    total,
                    object,
                    count,
                    share * 100.0
                ));
            } else if total > 0 {
                reasons.push(format!(
                    "「{}」在图中尚无「{}」记录（候选来自类型/特征相似性）",
                    object, predicate
                ));
            }
        }

        // 2) 主语类型 × 谓词相容度
        if let Some(subject) = subject {
            if let Some(subject_type) = self.entity_types.get(subject) {
                if let Some(type_counts) = self.pred_subject_types.get(predicate) {
                    let total: u64 = type_counts.values().sum();
                    let count = type_counts.get(subject_type).copied().unwrap_or(0);
                    if count > 0 {
                        let share = count as f64 / total.max(1) as f64;
                        score += 0.12 * share;
                        reasons.push(format!(
                            "主体「{}」属于「{}」，该类型在「{}」主语中占 {:.0}%",
                            subject,
                            subject_type,
                            predicate,
                            share * 100.0
                        ));
                    } else {
                        conflicts.push(format!(
                            "主体类型「{}」未在「{}」的历史主语中出现",
                            subject_type, predicate
                        ));
                        score -= 0.05;
                    }
                }
            }
        }

        // 3) 宾语类型 × 谓词相容度
        if let Some(object) = object {
            if let Some(object_type) = self.entity_types.get(object) {
                if let Some(type_counts) = self.pred_object_types.get(predicate) {
                    let total: u64 = type_counts.values().sum();
                    let count = type_counts.get(object_type).copied().unwrap_or(0);
                    if count > 0 {
                        let share = count as f64 / total.max(1) as f64;
                        score += 0.12 * share;
                        reasons.push(format!(
                            "宾体「{}」属于「{}」，该类型在「{}」宾语中占 {:.0}%",
                            object,
                            object_type,
                            predicate,
                            share * 100.0
                        ));
                    } else if total > 0 {
                        // 类型不相容：宾体类型从未在该谓词宾语中出现过（有足够历史证据时）
                        conflicts.push(format!(
                            "宾体类型「{}」未在「{}」的历史宾语中出现（{} 次观测）",
                            object_type, predicate, total
                        ));
                        score -= 0.05;
                    }
                }
            }
        }

        // 4) 特征重叠（Jaccard）
        if let (Some(subject), Some(object)) = (subject, object) {
            if let (Some(subject_features), Some(object_features)) = (
                self.entity_features.get(subject),
                self.entity_features.get(object),
            ) {
                let inter = subject_features.intersection(object_features).count();
                let union = (subject_features.len() + object_features.len())
                    .saturating_sub(inter)
                    .max(1);
                if inter > 0 {
                    let jaccard = inter as f64 / union as f64;
                    score += 0.25 * jaccard;
                    reasons.push(format!(
                        "候选与主体共享 {} 个关系特征（Jaccard {:.2}）",
                        inter, jaccard
                    ));
                }
            }

            // 5) 参与谓词分布的相似度（互补证据，权重较小）
            if let (Some(subject_predicates), Some(object_predicates)) = (
                self.entity_predicates.get(subject),
                self.entity_predicates.get(object),
            ) {
                let inter = subject_predicates
                    .keys()
                    .filter(|key| object_predicates.contains_key(*key))
                    .count();
                let union = (subject_predicates.len() + object_predicates.len())
                    .saturating_sub(inter)
                    .max(1);
                if inter > 0 {
                    score += 0.10 * inter as f64 / union as f64;
                    reasons.push(format!("与主体共用 {} 个谓词上下文", inter));
                }
            }
        }

        // 6) 功能谓词冲突：同主语已有不同取值 ⇒ 封闭世界假设下重罚
        if let (Some(subject), Some(object)) = (subject, object) {
            if let Some(by_subject) = self.pred_subject_objects.get(predicate) {
                if let Some(existing) = by_subject.get(subject) {
                    if !existing.contains(object) && self.is_functional(predicate) {
                        conflicts.push(format!(
                            "「{}」是功能谓词，「{}」已有取值：{}（封闭世界假设）",
                            predicate,
                            subject,
                            existing.iter().cloned().collect::<Vec<_>>().join("、")
                        ));
                        score = score.min(0.08);
                        contradicts_known = true;
                    }
                }
            }
        }

        if reasons.is_empty() {
            reasons.push("无同谓词历史与类型证据（低置信度假设）".to_string());
        }
        (
            score.clamp(0.02, 0.95),
            reasons,
            conflicts,
            contradicts_known,
        )
    }

    /// 功能谓词：≥3 个主语且 90% 以上主语只有唯一取值
    fn is_functional(&self, predicate: &str) -> bool {
        let Some(by_subject) = self.pred_subject_objects.get(predicate) else {
            return false;
        };
        if by_subject.len() < 3 {
            return false;
        }
        let single = by_subject
            .values()
            .filter(|objects| objects.len() == 1)
            .count();
        single as f64 / by_subject.len() as f64 >= 0.9
    }

    /// 反事实模拟：临时加入假设事实，正向饱和推导新结论（不修改原知识库）
    pub fn what_if(
        &self,
        engine: &ResolutionEngine,
        assumptions: &[Atom],
        max_rounds: usize,
        max_facts: usize,
    ) -> WhatIfResult {
        // 假设作为额外基事实注入推导
        let ground_assumptions: Vec<Atom> = assumptions
            .iter()
            .filter(|assumption| assumption.is_ground())
            .cloned()
            .collect();
        let derived =
            engine.forward_chain_with_extra_facts(&ground_assumptions, max_rounds, max_facts);
        // 单遍同时构造“排除已知事实后的推导结果”和“去重后的新事实”
        let mut seen: HashSet<String> = HashSet::new();
        let mut new_facts: Vec<Atom> = Vec::new();
        let mut kept: Vec<DerivedFact> = Vec::with_capacity(derived.len());
        for item in derived {
            if engine.has_fact(&item.atom) {
                continue;
            }
            if seen.insert(item.atom.display()) {
                new_facts.push(item.atom.clone());
            }
            kept.push(item);
        }
        WhatIfResult {
            assumptions: assumptions.to_vec(),
            derived: kept,
            new_facts,
        }
    }

    /// 选择"最佳失败前沿"：缺失原子少者优先；同规模时谓词在图谱中
    /// 支撑越多（同谓词历史取值越多）越可信，最后按显示长度稳定排序。
    fn select_best_frontier(&self, frontiers: Vec<Vec<Atom>>) -> Vec<Atom> {
        let quality = |frontier: &Vec<Atom>| -> u64 {
            frontier
                .iter()
                .map(|atom| {
                    self.pred_objects
                        .get(atom.predicate.as_ref())
                        .map(|values| values.values().sum::<u64>())
                        .unwrap_or(0)
                })
                .sum()
        };
        // 排序键：规模升序 → 谓词支撑降序 → 显示长度升序
        let candidates: Vec<(Vec<Atom>, usize, u64, usize)> = frontiers
            .into_iter()
            .map(|frontier| {
                let size = frontier.len();
                let support = quality(&frontier);
                let display_len: usize = frontier.iter().map(|atom| atom.display().len()).sum();
                (frontier, size, support, display_len)
            })
            .collect();
        // 取极值（稳定升序排序后的首个）
        candidates
            .into_iter()
            .min_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.3.cmp(&b.3))
            })
            .map(|(frontier, _, _, _)| frontier)
            .unwrap_or_default()
    }

    /// 暂存候选假设，供「采纳假设 N」使用
    pub fn set_pending(&mut self, hypotheses: Vec<Hypothesis>) {
        self.pending = hypotheses;
    }

    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// 清空全部会话态（待定假设 + 已采纳假设），保留观察统计。
    /// 供 WarmBase 会话分叉：新会话不得继承其他会话的假设。
    pub fn reset_session(&mut self) {
        self.pending.clear();
        self.adopted.clear();
    }

    /// 取第 index（1 起）条假设并从待定列表移除
    pub fn take_pending(&mut self, index: usize) -> Option<Hypothesis> {
        if index == 0 || index > self.pending.len() {
            return None;
        }
        Some(self.pending.remove(index - 1))
    }

    /// 记录被采纳的假设（会话级事实，不写知识图谱）
    pub fn adopt(&mut self, hypothesis: Hypothesis) {
        self.adopted.push(hypothesis.atom);
    }

    /// 恢复历史会话采纳的假设（跨会话临时信念）
    pub fn restore_adopted(&mut self, atoms: Vec<Atom>) {
        if atoms.is_empty() {
            return;
        }
        // 用哈希集合做成员检查
        let mut new_atoms: Vec<Atom> = Vec::new();
        {
            let mut seen: HashSet<&Atom> = self.adopted.iter().collect();
            for atom in &atoms {
                if seen.insert(atom) {
                    new_atoms.push(atom.clone());
                }
            }
        }
        self.adopted.extend(new_atoms);
    }

    /// 两条假设是否互斥：同一主体的功能谓词给出不同取值（不会同时为真）
    pub fn hypotheses_conflict(&self, a: &Hypothesis, b: &Hypothesis) -> bool {
        if a.atom.predicate != b.atom.predicate {
            return false;
        }
        if a.atom.terms.len() != 2 || b.atom.terms.len() != 2 {
            return a.atom == b.atom;
        }
        a.atom.terms[0] == b.atom.terms[0]
            && a.atom.terms[1] != b.atom.terms[1]
            && self.is_functional(&a.atom.predicate)
    }

    /// 联合解释场景生成（生成式推理：多假设协同 → 验证 → 排序）
    ///
    /// 1. 复刻 `analyze` 的失败前沿搜索，但汇集**所有前沿**（而非最佳前沿）的假设池；
    /// 2. 枚举规模 1..=max_scenario_size 的假设组合，过滤互斥组合；
    /// 3. 把每个组合注入克隆引擎，做完整的 SLD 反驳验证：只有真正证明
    ///    目标的组合才会被保留（这一点与单假设 Top-K 的"启发式猜测"本质不同）；
    /// 4. 联合可信度 = 成员可信度均值 × 规模折扣（假设越多越保守），
    ///    按"规模最小 → 联合可信度最高"排序。
    pub fn scenarios(
        &self,
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) -> Vec<Scenario> {
        if goals.is_empty() {
            return Vec::new();
        }
        // 目标本身可证：无需假设，不存在"解释场景"
        if !engine.solve(goals, query_vars).is_empty() {
            return Vec::new();
        }
        self.scenarios_impl(engine, goals, query_vars)
    }

    /// 与 [`AbductionEngine::scenarios`] 相同，但调用方已确认目标在 `engine`
    /// 上不可证，因而跳过可证性检查。仅在 `engine` 状态与不可证结论一致时调用，
    /// 语义与 `scenarios` 等价。
    pub fn scenarios_assuming_unprovable(
        &self,
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) -> Vec<Scenario> {
        if goals.is_empty() {
            return Vec::new();
        }
        self.scenarios_impl(engine, goals, query_vars)
    }

    fn scenarios_impl(
        &self,
        engine: &ResolutionEngine,
        goals: &[Atom],
        query_vars: &[String],
    ) -> Vec<Scenario> {
        let mut search = FrontierSearch {
            engine,
            max_depth: self.max_depth,
            max_frontiers: self.max_frontiers,
            budget: 20_000,
            results: Vec::new(),
            seen: HashSet::new(),
            freshen_counter: 0,
        };
        let mut missing = Vec::new();
        search.run(goals, &Substitution::new(), 0, &mut missing);

        // 假设池：所有前沿的基假设，排除与已知事实硬冲突者
        // 循环解释过滤：假设集合里不能包含目标本身（"假设目标成立"不算解释）
        let mut pool: Vec<Hypothesis> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for frontier in &search.results {
            for atom in frontier {
                for hypothesis in self.hypotheses_for(engine, atom) {
                    if !hypothesis.is_adoptable() || hypothesis.contradicts_known {
                        continue;
                    }
                    if goals.iter().any(|goal| goal == &hypothesis.atom) {
                        continue;
                    }
                    if seen.insert(hypothesis.atom.display()) {
                        pool.push(hypothesis);
                    }
                }
            }
        }
        pool.sort_by_cached_key(|hypothesis| ConfidenceDisplayKey {
            confidence: hypothesis.confidence,
            display: hypothesis.atom.display(),
        });
        // 组合枚举上限：池 10 × 规模 2 ⇒ 45 组，足够覆盖典型解释空间
        pool.truncate(10);

        let max_size = self.max_scenario_size.min(pool.len()).clamp(1, 4);
        let mut scenarios: Vec<Scenario> = Vec::new();
        let mut budget = 60usize;
        let mut indices: Vec<usize> = Vec::new();
        // 组合验证复用一个引擎克隆：逐组合“加入假设 → 求解 → 按原顺序移除”。
        let mut simulation = engine.clone();
        for size in 1..=max_size {
            let _ = collect_combinations(&pool, size, 0, &mut indices, &mut |combo| {
                if budget == 0 {
                    return false;
                }
                budget -= 1;
                // 互斥组合直接跳过（功能谓词同主体不同取值）
                for i in 0..combo.len() {
                    for j in (i + 1)..combo.len() {
                        if self.hypotheses_conflict(&pool[combo[i]], &pool[combo[j]]) {
                            return true;
                        }
                    }
                }
                let members: Vec<Hypothesis> =
                    combo.iter().map(|&index| pool[index].clone()).collect();
                let mut added: Vec<Atom> = Vec::with_capacity(members.len());
                for hypothesis in &members {
                    let atom = hypothesis.atom.clone();
                    if simulation.add_fact_atom(atom.clone()) {
                        added.push(atom);
                    }
                }
                let solutions = simulation.solve(goals, query_vars);
                for atom in &added {
                    simulation.remove_fact_atom(atom);
                }
                if solutions.is_empty() {
                    return true;
                }
                let mean = members
                    .iter()
                    .map(|hypothesis| hypothesis.confidence)
                    .sum::<f64>()
                    / members.len() as f64;
                let discount = 1.0 / (1.0 + 0.2 * (members.len().saturating_sub(1)) as f64);
                scenarios.push(Scenario {
                    hypotheses: members,
                    verified: true,
                    joint_confidence: (mean * discount).clamp(0.02, 0.95),
                    solutions,
                });
                true
            });
        }
        // 没有可验证的组合：退回展示可信度最高的单条假设（明确标注未验证）
        if scenarios.is_empty() {
            if let Some(best) = pool.first() {
                scenarios.push(Scenario {
                    hypotheses: vec![best.clone()],
                    verified: false,
                    joint_confidence: best.confidence,
                    solutions: Vec::new(),
                });
            }
        }
        // 规模最小优先（最小基数解释），同规模按联合可信度降序
        scenarios.sort_by_cached_key(|scenario| ScenarioOrderKey {
            size: scenario.size(),
            joint_confidence: scenario.joint_confidence,
            first_display: scenario.hypotheses[0].atom.display(),
        });
        scenarios.truncate(5);
        scenarios
    }
}

/// 组合枚举：对 pool 取 size 个下标的组合，回调返回 false 时提前终止
fn collect_combinations(
    pool: &[Hypothesis],
    size: usize,
    start: usize,
    indices: &mut Vec<usize>,
    visit: &mut dyn FnMut(&[usize]) -> bool,
) -> bool {
    if indices.len() == size {
        return visit(indices);
    }
    for index in start..pool.len() {
        indices.push(index);
        let keep_going = collect_combinations(pool, size, index + 1, indices, visit);
        indices.pop();
        if !keep_going {
            return false;
        }
    }
    true
}

fn constant_name(term: &Term) -> Option<&str> {
    match term {
        Term::Const(name) => Some(name),
        Term::Var(_) => None,
    }
}

/// 置信度降序、同置信度按显示串升序的排序键
#[derive(PartialEq)]
struct ConfidenceDisplayKey {
    confidence: f64,
    display: String,
}

impl Eq for ConfidenceDisplayKey {}

impl PartialOrd for ConfidenceDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConfidenceDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .confidence
            .partial_cmp(&self.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 场景排序键：规模升序 → 联合可信度降序 → 首假设显示串升序
#[derive(PartialEq)]
struct ScenarioOrderKey {
    size: usize,
    joint_confidence: f64,
    first_display: String,
}

impl Eq for ScenarioOrderKey {}

impl PartialOrd for ScenarioOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScenarioOrderKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size
            .cmp(&other.size)
            .then_with(|| {
                other
                    .joint_confidence
                    .partial_cmp(&self.joint_confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| self.first_display.cmp(&other.first_display))
    }
}

/// 去重：同一基事实只保留置信度最高的一条。
/// 输入已按可信度降序排序，故保留每个原子的首次出现即可，
/// 同时保持顺序（调用方随后依赖该顺序做截断与「假设 N」编号）。
fn dedupe_hypotheses(hypotheses: &mut Vec<Hypothesis>) {
    let mut seen: HashSet<String> = HashSet::new();
    hypotheses.retain(|hypothesis| seen.insert(hypothesis.atom.display()));
}

/// 有界后向链失败前沿搜索：
/// 目标逐个展开——能匹配事实/规则就继续深入；无任何匹配时把当前原子
/// 记入"缺失前提"，继续处理剩余目标。多处分支会得到多个前沿，去重后供排序。
struct FrontierSearch<'a> {
    engine: &'a ResolutionEngine,
    max_depth: usize,
    max_frontiers: usize,
    budget: usize,
    results: Vec<Vec<Atom>>,
    seen: HashSet<String>,
    /// 全局单调递增的换名计数：每次规则换名取一个全新 tag，
    /// 避免递归不同层复用同一 tag 造成变量捕获
    freshen_counter: usize,
}

impl<'a> FrontierSearch<'a> {
    fn run(&mut self, goals: &[Atom], subst: &Substitution, depth: usize, missing: &mut Vec<Atom>) {
        if self.budget == 0 || self.results.len() >= self.max_frontiers {
            return;
        }
        if goals.is_empty() {
            self.record(missing);
            return;
        }
        self.budget = self.budget.saturating_sub(1);

        let goal = &goals[0];
        let applied = apply_atom(subst, goal);

        // 深度截断：把当前原子视为缺失前提（更深的证明超出预算）
        if depth > self.max_depth {
            missing.push(applied);
            self.run(&goals[1..], subst, depth, missing);
            missing.pop();
            return;
        }

        // 1) 与基事实归结：命中即可继续推进剩余目标
        //（把引擎引用拷出，避免与递归调用 `self.run` 的可变借用冲突）
        let engine = self.engine;
        let mut fact_matched = false;
        for fact in engine
            .facts_for(&applied.predicate)
            .iter()
            .filter(|fact| fact.arity() == applied.arity())
            .take(6)
        {
            let mut unified = subst.clone();
            if crate::resolution_engine::unify_atoms(&applied, fact, &mut unified) {
                if fact_matched {
                    break;
                }
                fact_matched = true;
                self.run(&goals[1..], &unified, depth, missing);
            }
        }
        if fact_matched {
            return;
        }

        // 2) 直接切点：把当前原子记为缺失前提（最小解释，最先记录）
        //（借出 `applied` 随递归入栈，返回后再取回；
        //  递归按栈序归还，`pop` 必然取回同一原子）
        missing.push(applied);
        self.run(&goals[1..], subst, depth, missing);
        let applied = missing.pop().unwrap_or_else(|| apply_atom(subst, goal));

        // 3) 与规则头归结：换名后合一，再把规则体前置到目标队列
        //    （规则解释作为备选前沿，与直接切点一起参与排序）
        for clause in engine.clauses_iter_for(&applied.predicate).take(6) {
            if self.budget == 0 {
                break;
            }
            let tag = self.freshen_counter;
            self.freshen_counter += 1;
            let fresh = freshen_clause(clause, tag);
            let mut unified = subst.clone();
            if !crate::resolution_engine::unify_atoms(&applied, &fresh.head, &mut unified) {
                continue;
            }
            // 规则体直接移动进组合目标队列（head 已合一，只需继续推进规则体）
            let mut combined: Vec<Atom> = fresh.body;
            combined.extend_from_slice(&goals[1..]);
            self.run(&combined, &unified, depth + 1, missing);
        }
    }

    fn record(&mut self, missing: &[Atom]) {
        if self.results.len() >= self.max_frontiers {
            return;
        }
        let mut key = String::new();
        for (index, atom) in missing.iter().enumerate() {
            if index > 0 {
                key.push('|');
            }
            key.push_str(&atom.display());
        }
        if !self.seen.insert(key) {
            return;
        }
        self.results.push(missing.to_vec());
    }
}

// ---------------------------------------------------------------------------
// 自然语言渲染：把结构化的证明/假设/反事实结果组织成可读回答
// （render 闭包由调用方提供，以复用知识图谱的谓词/句式渲染）
// ---------------------------------------------------------------------------

fn render_proof_tree(
    node: &ProofNode,
    render: &dyn Fn(&Atom) -> String,
    lines: &mut Vec<String>,
    indent: usize,
) {
    let padding = "    ".repeat(indent);
    let fact = render(&node.atom);
    match &node.source {
        ProofSource::Fact => lines.push(format!("{}{}（事实）", padding, fact)),
        ProofSource::Rule { source } => lines.push(format!(
            "{}{}（由 {}；本步置信度 {:.2}）",
            padding, fact, source, node.confidence
        )),
    }
    for child in &node.children {
        render_proof_tree(child, render, lines, indent + 1);
    }
}

/// 渲染溯因报告：可证则给证明，不可证则给"失败前沿 + 候选假设"
pub fn render_report(report: &AbductionReport, render: &dyn Fn(&Atom) -> String) -> String {
    let goal = render(&report.goal);
    if report.provable {
        let mut output = format!(
            "【溯因推理·后向链得证】目标「{}」可由现有知识证明。\n",
            goal
        );
        if let Some(solution) = report.solutions.first() {
            output.push_str(&format!("  证明置信度 {:.2}\n", solution.confidence));
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                render_proof_tree(proof, render, &mut lines, 1);
            }
            output.push_str("  证明树：\n");
            output.push_str(&lines.join("\n"));
        }
        return output;
    }

    let mut output = format!(
        "【溯因推理·假设生成】目标「{}」在后向链深度上限内无法由现有知识证明。\n",
        goal
    );
    if !report.frontier.is_empty() {
        let frontier = report
            .frontier
            .iter()
            .map(|atom| render(atom))
            .collect::<Vec<_>>()
            .join("；");
        output.push_str(&format!("  失败前沿（缺少的前提）：{}\n", frontier));
    }
    for note in &report.notes {
        output.push_str(&format!("  说明：{}\n", note));
    }
    if report.hypotheses.is_empty() {
        output.push_str("  未能生成候选假设（知识库中缺少可参考的同类事实）。\n");
        return output;
    }
    output.push_str("  候选假设（纯计数评分，按可信度降序）：\n");
    for (index, hypothesis) in report.hypotheses.iter().enumerate() {
        let assumption = render(&hypothesis.atom);
        output.push_str(&format!(
            "    假设 {}（可信度 {:.2}，{}）：{}",
            index + 1,
            hypothesis.confidence,
            hypothesis.kind.label(),
            assumption
        ));
        if !hypothesis.is_adoptable() {
            output.push_str("〔含未绑定变量，仅作方向提示〕");
        }
        output.push('\n');
        if !hypothesis.reasons.is_empty() {
            output.push_str(&format!(
                "        依据：{}\n",
                hypothesis.reasons.join("；")
            ));
        }
        if !hypothesis.conflicts.is_empty() {
            output.push_str(&format!(
                "        冲突：{}\n",
                hypothesis.conflicts.join("；")
            ));
        }
    }
    output.push_str(
        "  这些假设不会自动写入知识库；输入「采纳假设 N」可把它们加入本轮会话事实集，输入「澄清」可让系统就这些假设逐条向你确认（是 / 不是 / 不知道）。",
    );
    output
}

/// 渲染联合解释场景报告（生成式推理：多假设协同 + 注入验证）
pub fn render_scenarios(
    goal: &Atom,
    frontier: &[Atom],
    scenarios: &[Scenario],
    render: &dyn Fn(&Atom) -> String,
) -> String {
    let mut output = format!(
        "【生成式联合解释】目标「{}」无法由现有知识单独证明；枚举假设组合并注入克隆引擎验证（纯符号）。\n",
        render(goal)
    );
    if !frontier.is_empty() {
        let frontier_text = frontier
            .iter()
            .map(|atom| render(atom))
            .collect::<Vec<_>>()
            .join("；");
        output.push_str(&format!("  失败前沿（缺少的前提）：{}\n", frontier_text));
    }
    if scenarios.is_empty() {
        output.push_str(
            "  未能生成任何解释场景（知识库中缺少可参考的同类事实）。\n  可先补充相关事实，或用「溯因：<陈述>」查看单条假设。",
        );
        return output;
    }
    for (index, scenario) in scenarios.iter().enumerate() {
        let status = if scenario.verified {
            "注入后目标可证"
        } else {
            "未通过验证（仅为最弱候选，请谨慎采纳）"
        };
        output.push_str(&format!(
            "  场景 {}（{} 条假设，联合可信度 {:.2}，{}）：\n",
            index + 1,
            scenario.size(),
            scenario.joint_confidence,
            status
        ));
        for (member_index, hypothesis) in scenario.hypotheses.iter().enumerate() {
            output.push_str(&format!(
                "    假设 {}-{}（{:.2}，{}）：{}",
                index + 1,
                member_index + 1,
                hypothesis.confidence,
                hypothesis.kind.label(),
                render(&hypothesis.atom)
            ));
            if !hypothesis.is_adoptable() {
                output.push_str("〔含未绑定变量，仅作方向提示〕");
            }
            output.push('\n');
            if !hypothesis.reasons.is_empty() {
                output.push_str(&format!(
                    "        依据：{}\n",
                    hypothesis.reasons.join("；")
                ));
            }
        }
        if scenario.verified {
            if let Some(solution) = scenario.solutions.first() {
                output.push_str(&format!(
                    "    验证证明（注入假设后后向链得证，置信度 {:.2}）：\n",
                    solution.confidence
                ));
                let mut lines = Vec::new();
                for proof in &solution.proofs {
                    render_proof_tree(proof, render, &mut lines, 3);
                }
                output.push_str(&lines.join("\n"));
                output.push('\n');
            }
        }
    }
    output.push_str(
        "  场景中的假设都不会自动写入知识库；输入「澄清」可让系统就这些假设逐条向你确认，输入「采纳假设 N」可直接采纳单条。",
    );
    output
}

/// 渲染一条澄清提问（把待确认假设逆转为"是/不是"式问题）
pub fn render_clarification_question(
    session: &ClarificationSession,
    render: &dyn Fn(&Atom) -> String,
) -> String {
    match session.current() {
        Some(hypothesis) => format!(
            "【对话式澄清】系统想确认一条推测（可信度 {:.2}）：\n  「{}」是否成立？\n  请回答：是 / 不是 / 不知道；若是别的取值可答「不是，是<取值>」；也可答「全部确认」/「全部否定」处理剩余全部（剩余 {} 条）。",
            hypothesis.confidence,
            render(&hypothesis.atom),
            session.remaining()
        ),
        None => "【对话式澄清】没有需要确认的假设。".to_string(),
    }
}

/// 渲染澄清会话结束摘要
pub fn render_clarification_summary(
    session: &ClarificationSession,
    render: &dyn Fn(&Atom) -> String,
) -> String {
    let mut output = format!("【澄清结束】{}。\n", session.progress_text());
    for hypothesis in &session.confirmed {
        output.push_str(&format!("  ✓ 已确认：{}\n", render(&hypothesis.atom)));
    }
    for hypothesis in &session.rejected {
        output.push_str(&format!("  ✗ 已否定：{}\n", render(&hypothesis.atom)));
    }
    for hypothesis in &session.deferred {
        output.push_str(&format!("  ？ 不确定：{}\n", render(&hypothesis.atom)));
    }
    output.trim_end().to_string()
}

/// 渲染反事实模拟结果
pub fn render_what_if(result: &WhatIfResult, render: &dyn Fn(&Atom) -> String) -> String {
    let assumptions = result
        .assumptions
        .iter()
        .map(|atom| render(atom))
        .collect::<Vec<_>>()
        .join("；");
    let mut output = format!(
        "【反事实模拟·纯符号】临时假设：{}（模拟结束后自动回滚，不写入知识库）。\n",
        assumptions
    );
    if result.new_facts.is_empty() {
        output.push_str("  正向链未推出新结论（假设与现有规则没有可组合的推理路径）。\n");
        return output;
    }
    output.push_str(&format!(
        "  正向链推出 {} 条新结论：\n",
        result.new_facts.len()
    ));
    for derived in &result.derived {
        output.push_str(&format!(
            "    • {}（依据：{}；置信度 {:.2}，{} 跳）\n",
            render(&derived.atom),
            derived.rule_source,
            derived.confidence,
            derived.depth
        ));
    }
    output
}

/// 渲染待定假设清单（供「采纳假设 N」）
pub fn render_pending(pending: &[Hypothesis], render: &dyn Fn(&Atom) -> String) -> String {
    if pending.is_empty() {
        return "当前没有待定假设。可先用「溯因：<陈述>」或「假设生成：<问句>」生成候选。"
            .to_string();
    }
    let mut output =
        String::from("当前待定假设（未写入知识库，可用「采纳假设 N」加入会话事实集）：\n");
    for (index, hypothesis) in pending.iter().enumerate() {
        output.push_str(&format!(
            "  {}. （可信度 {:.2}，{}）{}\n",
            index + 1,
            hypothesis.confidence,
            hypothesis.kind.label(),
            render(&hypothesis.atom)
        ));
    }
    output.trim_end().to_string()
}

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

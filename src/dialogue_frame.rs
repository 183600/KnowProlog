// 话语帧（Dialogue Frame）——把一句话解析成带洞（hole）的查询帧，
// 洞的排序（sort）由本体给定，洞的填充由合一（unification）完成。
//
// 省略/指代/追问/话题回指处理为「同槽位、排序相容、最近优先」的帧间合一：
//
//   上一完整帧：问属性(姚明, 身高)
//   本轮片段：  问属性(□,    体重)
//   合一结果：  问属性(姚明,  体重)   ← 实体洞由上一帧的同槽位补入
//
// 复合问句（姚明的身高和体重是多少）解析为多个帧：第一帧照答，其余进入
// 「挂账」（pending）等待后续输入补全；挂账有容量与轮数上限，且只在
// 追问命中时消费（无半消费），不合法的合一逐字回退常规管线。
//
// 无源可填的洞还有第 4 种合法结局：澄清（clarification）——不猜，把
// 变量显式还给用户。用户短答（序号/实体名/都不是）被确定性解析为一次
// 绑定（σ 扩展），随后原问句被实例化并重放。绑定带出处（候选表 + 轮次），
// 因此「系统为什么把『体重是多少』当成姚明」逐字可断言。

use serde::{Deserialize, Serialize};

use crate::resolution_engine::{unify_atoms, Atom, Substitution, Term};

/// 帧原子谓词：问属性(实体槽, 属性槽)
const ASK_ATTRIBUTE: &str = "问属性";
/// 追问核心的最大字数（超限不接管，回退常规管线）
const FOLLOWUP_MAX_CORE_CHARS: usize = 16;
const DEFAULT_MAX_PENDING: usize = 4;
const DEFAULT_PENDING_TTL: u64 = 6;
const DEFAULT_CLARIFICATION_TTL: u64 = 3;

fn default_max_pending() -> usize {
    DEFAULT_MAX_PENDING
}

fn default_pending_ttl() -> u64 {
    DEFAULT_PENDING_TTL
}

fn default_clarification_ttl() -> u64 {
    DEFAULT_CLARIFICATION_TTL
}

/// 本体视图：洞的合法性 = 成员检查（排序判定），不做模糊猜测。
pub trait OntologyView {
    /// 归一化实体名；不是已登记实体返回 None
    fn canonical_entity(&self, raw: &str) -> Option<String>;
    /// 归一化属性/谓词词面（含同义词；保留逆向词面的表述方向）；未知返回 None
    fn canonical_attribute(&self, raw: &str) -> Option<String>;
}

/// 槽位填充来源：对话状态可打印、可审计（哪个洞由哪一轮的哪个槽填上）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotOrigin {
    Explicit,
    Inherited { turn: u64 },
    FromPending { turn: u64 },
    /// 无源可填的洞由用户回答澄清问题后绑定（第几轮发起的澄清）
    Clarified { turn: u64 },
}

impl Default for SlotOrigin {
    fn default() -> Self {
        SlotOrigin::Explicit
    }
}

impl SlotOrigin {
    pub fn label(&self) -> String {
        match self {
            SlotOrigin::Explicit => "本轮".to_string(),
            SlotOrigin::Inherited { turn } => format!("第 {} 轮的完整帧", turn),
            SlotOrigin::FromPending { turn } => format!("第 {} 轮的挂账", turn),
            SlotOrigin::Clarified { turn } => format!("第 {} 轮的澄清", turn),
        }
    }
}

/// AskAttr 查询帧：about（实体槽）× asks（属性槽）；任一槽可为洞（None）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskAttributeFrame {
    pub entity: Option<String>,
    pub attribute: Option<String>,
    #[serde(default)]
    pub entity_origin: SlotOrigin,
    #[serde(default)]
    pub attribute_origin: SlotOrigin,
    #[serde(default)]
    pub raw: String,
    #[serde(default)]
    pub turn: u64,
}

impl AskAttributeFrame {
    pub fn new(
        raw: &str,
        turn: u64,
        entity: Option<String>,
        attribute: Option<String>,
    ) -> Self {
        Self {
            entity,
            attribute,
            entity_origin: SlotOrigin::Explicit,
            attribute_origin: SlotOrigin::Explicit,
            raw: raw.to_string(),
            turn,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.entity.is_some() && self.attribute.is_some()
    }

    /// 完整帧的规范问句（交给常规问答管线执行）
    pub fn question(&self) -> Option<String> {
        Some(format!(
            "{}的{}是什么",
            self.entity.as_deref()?,
            self.attribute.as_deref()?
        ))
    }

    /// 可打印的帧形式：洞用 □ 显示
    pub fn display(&self) -> String {
        format!(
            "{} 的 {}",
            self.entity.as_deref().unwrap_or("□实体"),
            self.attribute.as_deref().unwrap_or("□属性")
        )
    }
}

/// 帧 → 一阶原子 `问属性(实体槽, 属性槽)`，洞为逻辑变量。
/// 两个帧的变量前缀必须不同，否则会错误共享变量。
fn frame_atom(frame: &AskAttributeFrame, prefix: &str) -> Atom {
    let entity = match &frame.entity {
        Some(value) => Term::constant(value),
        None => Term::var(&format!("{}实体", prefix)),
    };
    let attribute = match &frame.attribute {
        Some(value) => Term::constant(value),
        None => Term::var(&format!("{}属性", prefix)),
    };
    Atom::new(ASK_ATTRIBUTE, vec![entity, attribute])
}

/// 前帧原子的「投影」形式：会被后帧显式值覆盖的槽位在合一前投影为新鲜变量
/// （最新优先——对话中显式给出新值就是覆盖旧值），其余槽位保留原值参与
/// 合一。谓词同名同元数即排序（sort）相容；冲突被投影消解，洞才被填充。
fn project_older_atom(older: &AskAttributeFrame, newer: &AskAttributeFrame) -> Atom {
    let entity = if newer.entity.is_some() {
        Term::var("前掩实体")
    } else {
        match &older.entity {
            Some(value) => Term::constant(value),
            None => Term::var("前实体"),
        }
    };
    let attribute = if newer.attribute.is_some() {
        Term::var("前掩属性")
    } else {
        match &older.attribute {
            Some(value) => Term::constant(value),
            None => Term::var("前属性"),
        }
    };
    Atom::new(ASK_ATTRIBUTE, vec![entity, attribute])
}

/// 沿置换链读回槽位值；仍为变量（两边都是洞）返回 None
fn read_slot(subst: &Substitution, var: &str) -> Option<String> {
    let mut name = var.to_string();
    for _ in 0..64 {
        match subst.get(&name)? {
            Term::Const(value) => return Some(value.to_string()),
            Term::Var(next) => name = next.to_string(),
        }
    }
    None
}

/// 帧间合一：前帧的已填槽位补入后帧的洞（同槽位、排序相容、最近优先）。
/// 前帧被显式覆盖的槽位先投影掉，因此合一不会把「换话题」误判为冲突。
/// 返回值可能仍含洞（两帧同槽位都是洞时），完整性由调用方判定。
pub fn unify_frames(
    older: &AskAttributeFrame,
    newer: &AskAttributeFrame,
) -> Option<AskAttributeFrame> {
    let mut subst = Substitution::new();
    if !unify_atoms(
        &project_older_atom(older, newer),
        &frame_atom(newer, "后"),
        &mut subst,
    ) {
        return None;
    }
    let entity = newer.entity.clone().or_else(|| read_slot(&subst, "后实体"));
    let attribute = newer
        .attribute
        .clone()
        .or_else(|| read_slot(&subst, "后属性"));
    let entity_origin = if newer.entity.is_some() {
        newer.entity_origin.clone()
    } else {
        SlotOrigin::Inherited { turn: older.turn }
    };
    let attribute_origin = if newer.attribute.is_some() {
        newer.attribute_origin.clone()
    } else {
        SlotOrigin::Inherited { turn: older.turn }
    };
    Some(AskAttributeFrame {
        entity,
        attribute,
        entity_origin,
        attribute_origin,
        raw: newer.raw.clone(),
        turn: newer.turn,
    })
}

/// 澄清请求：一个无源可填的实体洞 + 确定性候选 + 原问句。
///
/// 这是变量求解的第四种合法结局：不猜（回退失败）、不静默择一，而是把
/// 自由变量显式还给用户，请用户做一次绑定；绑定后原问句被实例化并重放。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarificationRequest {
    /// 原问句（带洞帧的原始表述）
    pub raw: String,
    /// 洞所在问句的属性槽（已归一）
    pub attribute: String,
    /// 候选绑定（构造方保证确定性排序：焦点优先，其余按名称）
    pub candidates: Vec<String>,
    /// 发起轮次
    pub turn: u64,
}

impl ClarificationRequest {
    /// 澄清提示：候选编号列表 + 三种合法回答（序号 / 实体名 / 都不是）
    pub fn prompt(&self) -> String {
        let mut output = format!(
            "「{}」还缺一个实体。你想问谁的「{}」？\n",
            self.raw.trim(),
            self.attribute
        );
        for (index, candidate) in self.candidates.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", index + 1, candidate));
        }
        output.push_str("回复序号或实体名完成绑定；「都不是」放弃这次追问。");
        output
    }

    /// 确定性短答解析：序号（阿拉伯/中文数字）→ 候选名 → 都不是。
    /// 解析不了的回答一律 Unmatched（调用方放行给常规管线，零状态变化）。
    pub fn parse_reply(&self, text: &str) -> ClarificationReply {
        let reply = text.trim().trim_end_matches(|c: char| {
            matches!(c, '。' | '！' | '!' | '？' | '?' | ' ' | '～' | '~')
        });
        if reply.is_empty() {
            return ClarificationReply::Unmatched;
        }
        const CANCEL: [&str; 9] = [
            "都不是", "算了", "取消", "不用了", "不查了", "不问了", "结束澄清", "停止澄清",
            "退出澄清",
        ];
        if CANCEL.contains(&reply) {
            return ClarificationReply::Cancel;
        }
        if let Some(choice) = parse_choice_index(reply) {
            if choice >= 1 {
                if let Some(candidate) = self.candidates.get(choice - 1) {
                    return ClarificationReply::Candidate(candidate.clone());
                }
            }
            return ClarificationReply::OutOfRange;
        }
        if let Some(candidate) = self
            .candidates
            .iter()
            .find(|candidate| candidate.as_str() == reply)
        {
            return ClarificationReply::Candidate(candidate.clone());
        }
        ClarificationReply::Unmatched
    }
}

/// 澄清短答的确定性分类（解析规则之外的一切回答都放行）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClarificationReply {
    /// 绑定到某候选
    Candidate(String),
    /// 纯数字/中文数字但越界：重问（零状态变化）
    OutOfRange,
    /// 放弃本次追问
    Cancel,
    /// 不是对澄清的回答
    Unmatched,
}

/// 澄清应答结果（供对话层直接消费；重放问句由调用方交给常规问答管线）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClarificationOutcome {
    /// 绑定成功：question 为实例化后的原问句，note 为出处收据
    Bound { question: String, note: String },
    /// 用户放弃本次追问（洞随之撤销，其余状态不变）
    Cancelled { note: String },
    /// 序号越界：重问（零状态变化）
    Reprompt { prompt: String },
}

/// 一基序号解析：支持「1 / 1. / 第2 / 三个 / 二」等确定性写法
fn parse_choice_index(text: &str) -> Option<usize> {
    let spec = text
        .trim()
        .trim_start_matches('第')
        .trim_end_matches(|c: char| matches!(c, '.' | '。' | '、' | '个' | ')' | '）' | ' '));
    if spec.is_empty() {
        return None;
    }
    if spec.chars().all(|c| c.is_ascii_digit()) {
        return spec.parse().ok();
    }
    match spec {
        "一" => Some(1),
        "二" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        "七" => Some(7),
        "八" => Some(8),
        "九" => Some(9),
        "十" => Some(10),
        _ => None,
    }
}

/// 追问消解结果（已提交：挂账已结清、上一帧已更新）
#[derive(Debug, Clone)]
pub struct FollowUpResolution {
    pub frame: AskAttributeFrame,
    pub question: String,
    pub note: String,
    pub closed_pending: Option<AskAttributeFrame>,
}

/// 追问补全计划（纯计算，不修改状态）：提交后才消费挂账/更新上一帧，
/// 保证「不匹配的输入绝不改变挂账状态」。
#[derive(Debug, Clone)]
pub struct FollowUpPlan {
    pub frame: AskAttributeFrame,
    pub question: String,
    pending_index: Option<usize>,
    filled_from_pending: bool,
}

/// 话语帧记忆：上一完整帧 + 挂账（未竟之问）+ 澄清（待绑定变量），
/// 带容量与轮数上限。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMemory {
    #[serde(default)]
    turn: u64,
    #[serde(default)]
    last: Option<AskAttributeFrame>,
    #[serde(default)]
    pending: Vec<AskAttributeFrame>,
    #[serde(default = "default_max_pending")]
    max_pending: usize,
    #[serde(default = "default_pending_ttl")]
    pending_ttl: u64,
    #[serde(default)]
    unifications: u64,
    #[serde(default)]
    pending_closed: u64,
    /// 至多一条待绑定澄清（新请求覆盖旧请求）
    #[serde(default)]
    clarification: Option<ClarificationRequest>,
    #[serde(default = "default_clarification_ttl")]
    clarification_ttl: u64,
    #[serde(default)]
    clarifications_asked: u64,
    #[serde(default)]
    clarifications_bound: u64,
    #[serde(default)]
    clarifications_cancelled: u64,
    #[serde(default)]
    clarifications_expired: u64,
}

impl Default for FrameMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameMemory {
    pub fn new() -> Self {
        Self {
            turn: 0,
            last: None,
            pending: Vec::new(),
            max_pending: DEFAULT_MAX_PENDING,
            pending_ttl: DEFAULT_PENDING_TTL,
            unifications: 0,
            pending_closed: 0,
            clarification: None,
            clarification_ttl: DEFAULT_CLARIFICATION_TTL,
            clarifications_asked: 0,
            clarifications_bound: 0,
            clarifications_cancelled: 0,
            clarifications_expired: 0,
        }
    }

    /// 新一轮：轮次 +1，过期挂账与过期澄清自动清理（状态有界）
    pub fn begin_turn(&mut self) {
        self.turn += 1;
        let ttl = self.pending_ttl;
        let turn = self.turn;
        self.pending
            .retain(|frame| turn.saturating_sub(frame.turn) <= ttl);
        let expired = self
            .clarification
            .as_ref()
            .map(|request| turn.saturating_sub(request.turn) >= self.clarification_ttl)
            .unwrap_or(false);
        if expired {
            self.clarification = None;
            self.clarifications_expired += 1;
        }
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn last(&self) -> Option<&AskAttributeFrame> {
        self.last.as_ref()
    }

    pub fn pending(&self) -> &[AskAttributeFrame] {
        &self.pending
    }

    /// 当前待绑定澄清（至多一条）
    pub fn clarification(&self) -> Option<&ClarificationRequest> {
        self.clarification.as_ref()
    }

    /// 发起澄清：登记待绑定变量 + 候选表（新请求覆盖旧请求，至多一条）
    pub fn begin_clarification(
        &mut self,
        raw: &str,
        attribute: &str,
        candidates: Vec<String>,
    ) -> ClarificationRequest {
        let request = ClarificationRequest {
            raw: raw.trim().to_string(),
            attribute: attribute.to_string(),
            candidates,
            turn: self.turn,
        };
        self.clarifications_asked += 1;
        self.clarification = Some(request.clone());
        request
    }

    /// 提交绑定：σ 扩展为常量，实例化原问句成为上一帧；返回重放问句与出处收据
    pub fn commit_clarification(&mut self, entity: &str) -> Option<ClarificationOutcome> {
        let request = self.clarification.take()?;
        let frame = AskAttributeFrame {
            entity: Some(entity.to_string()),
            attribute: Some(request.attribute.clone()),
            entity_origin: SlotOrigin::Clarified { turn: request.turn },
            attribute_origin: SlotOrigin::Explicit,
            raw: request.raw.clone(),
            turn: self.turn,
        };
        let question = frame.question()?;
        let note = format!(
            "变量澄清：实体「{}」←第 {} 轮澄清（原问句「{}」）",
            entity, request.turn, request.raw
        );
        self.last = Some(frame);
        self.clarifications_bound += 1;
        Some(ClarificationOutcome::Bound { question, note })
    }

    /// 放弃澄清：撤销待绑定变量，其余状态不变
    pub fn cancel_clarification(&mut self) -> Option<ClarificationOutcome> {
        let request = self.clarification.take()?;
        self.clarifications_cancelled += 1;
        Some(ClarificationOutcome::Cancelled {
            note: format!("变量澄清：已放弃「{}」（可随时重新提问）。", request.raw),
        })
    }

    /// 登记完整帧：问句已被常规管线执行后成为「上一帧」
    pub fn record_last(&mut self, frame: AskAttributeFrame) {
        if frame.is_complete() {
            self.last = Some(frame);
        }
    }

    /// 复合问句：第一帧照答（登记为上一帧），其余挂账
    pub fn record_compound(&mut self, mut frames: Vec<AskAttributeFrame>) {
        if frames.len() < 2 {
            if let Some(frame) = frames.pop() {
                self.record_last(frame);
            }
            return;
        }
        let first = frames.remove(0);
        self.record_last(first);
        for frame in frames {
            self.push_pending(frame);
        }
    }

    pub fn push_pending(&mut self, frame: AskAttributeFrame) {
        if !frame.is_complete() {
            return;
        }
        // 同实体同属性的挂账去重（保留最新）
        self.pending
            .retain(|pending| !(pending.entity == frame.entity && pending.attribute == frame.attribute));
        self.pending.push(frame);
        while self.pending.len() > self.max_pending {
            self.pending.remove(0);
        }
    }

    /// 追问补全计划：属性明确时优先补全同名属性的挂账（用户正在补它），
    /// 其余洞由上一完整帧同槽位补入；仍不完整返回 None（调用方回退常规管线）。
    pub fn plan_followup(&self, current: &AskAttributeFrame) -> Option<FollowUpPlan> {
        if current.entity.is_none() && current.attribute.is_none() {
            return None;
        }
        let mut filled = current.clone();
        let mut filled_from_pending = false;
        let mut pending_index = None;

        if filled.entity.is_none() && filled.attribute.is_some() {
            if let Some(index) = self
                .pending
                .iter()
                .rposition(|pending| pending.entity.is_some() && pending.attribute == filled.attribute)
            {
                let pending = &self.pending[index];
                filled.entity = pending.entity.clone();
                filled.entity_origin = SlotOrigin::FromPending { turn: pending.turn };
                filled_from_pending = true;
                pending_index = Some(index);
            }
        }

        if !filled.is_complete() {
            let last = self.last.as_ref()?;
            filled = unify_frames(last, &filled)?;
        }
        if !filled.is_complete() {
            return None;
        }

        if pending_index.is_none() {
            if let Some(index) = self.pending.iter().rposition(|pending| {
                pending.entity == filled.entity && pending.attribute == filled.attribute
            }) {
                let pending = &self.pending[index];
                filled.entity_origin = SlotOrigin::FromPending { turn: pending.turn };
                filled.attribute_origin = SlotOrigin::FromPending { turn: pending.turn };
                pending_index = Some(index);
            }
        }

        let question = filled.question()?;
        Some(FollowUpPlan {
            frame: filled,
            question,
            pending_index,
            filled_from_pending,
        })
    }

    /// 提交计划：消费挂账、更新上一帧、生成收据
    pub fn commit_followup(&mut self, plan: FollowUpPlan) -> FollowUpResolution {
        let closed = plan.pending_index.and_then(|index| {
            if index < self.pending.len() {
                Some(self.pending.remove(index))
            } else {
                None
            }
        });
        self.unifications += 1;
        if closed.is_some() {
            self.pending_closed += 1;
        }
        let note = render_resolution_note(&plan.frame, closed.as_ref(), plan.filled_from_pending);
        self.last = Some(plan.frame.clone());
        FollowUpResolution {
            frame: plan.frame,
            question: plan.question,
            note,
            closed_pending: closed,
        }
    }

    /// 帧记忆报告（逐条展示来源轮次，可审计、确定性）
    pub fn report(&self) -> String {
        let mut output = format!(
            "话语帧记忆（带洞查询帧 + 合一 + 澄清；无神经网络）：轮次 {}，合一 {} 次，结清挂账 {} 次，澄清 {} 次（绑定 {}、放弃 {}、过期 {}）。\n",
            self.turn,
            self.unifications,
            self.pending_closed,
            self.clarifications_asked,
            self.clarifications_bound,
            self.clarifications_cancelled,
            self.clarifications_expired
        );
        match &self.last {
            Some(frame) => output.push_str(&format!(
                "上一完整帧：{}（第 {} 轮；实体←{}，属性←{}）。\n",
                frame.display(),
                frame.turn,
                frame.entity_origin.label(),
                frame.attribute_origin.label()
            )),
            None => output.push_str("上一完整帧：暂无。\n"),
        }
        if self.pending.is_empty() {
            output.push_str("挂账（未竟之问）：暂无。\n");
        } else {
            output.push_str(&format!(
                "挂账（未竟之问，容量 {}，{} 轮过期）：\n",
                self.max_pending, self.pending_ttl
            ));
            for frame in &self.pending {
                let remaining = self
                    .pending_ttl
                    .saturating_sub(self.turn.saturating_sub(frame.turn));
                output.push_str(&format!(
                    "  • {}（第 {} 轮挂起，剩余 {} 轮；追问「{}呢」可结清）\n",
                    frame.display(),
                    frame.turn,
                    remaining,
                    frame.attribute.as_deref().unwrap_or("?")
                ));
            }
        }
        match &self.clarification {
            Some(request) => {
                let remaining = self
                    .clarification_ttl
                    .saturating_sub(self.turn.saturating_sub(request.turn));
                output.push_str(&format!(
                    "澄清（等待绑定，第 {} 轮发起，剩余 {} 轮）：原问句「{}」，候选：",
                    request.turn,
                    remaining,
                    request.raw
                ));
                let list: Vec<String> = request
                    .candidates
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| format!("{}.{}", index + 1, candidate))
                    .collect();
                output.push_str(&list.join("；"));
                output.push_str("。回复序号或实体名绑定；「都不是」放弃。\n");
            }
            None => output.push_str("澄清（等待绑定）：暂无。\n"),
        }
        output
    }
}

fn render_resolution_note(
    frame: &AskAttributeFrame,
    closed: Option<&AskAttributeFrame>,
    filled_from_pending: bool,
) -> String {
    let mut note = format!(
        "话语帧合一：实体「{}」←{}，属性「{}」←{}",
        frame.entity.as_deref().unwrap_or("?"),
        frame.entity_origin.label(),
        frame.attribute.as_deref().unwrap_or("?"),
        frame.attribute_origin.label()
    );
    if let Some(pending) = closed {
        let source = if filled_from_pending {
            "补全"
        } else {
            "结清"
        };
        note.push_str(&format!(
            "；{}第 {} 轮挂账「{}」",
            source,
            pending.turn,
            pending.display()
        ));
    }
    note.push('。');
    note
}

fn trim_tail(input: &str) -> &str {
    input
        .trim()
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | ' ' | '～' | '~' | '…'
            )
        })
        .trim()
}

fn strip_polite(mut text: &str) -> &str {
    const PREFIXES: [&str; 7] = [
        "请问",
        "请告诉我",
        "我想知道",
        "我想问一下",
        "帮我查一下",
        "帮我查查",
        "麻烦问一下",
    ];
    loop {
        let mut changed = false;
        for prefix in PREFIXES {
            if let Some(rest) = text.strip_prefix(prefix) {
                let rest = rest.trim();
                if !rest.is_empty() {
                    text = rest;
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            return text;
        }
    }
}

/// 去掉问句尾巴（是多少/分别是什么/呢…），可叠加（「是多少呢」）
fn strip_question_suffixes(mut core: &str) -> &str {
    const SUFFIXES: [&str; 12] = [
        "分别是多少",
        "分别是什么",
        "分别是哪些",
        "分别是",
        "是多少",
        "是什么",
        "是哪些",
        "有哪些",
        "多少",
        "哪些",
        "什么",
        "呢",
    ];
    loop {
        let mut changed = false;
        for suffix in SUFFIXES {
            if let Some(rest) = core.strip_suffix(suffix) {
                let rest = rest.trim();
                if !rest.is_empty() {
                    core = rest;
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            return core;
        }
    }
}

/// 把一个（已去壳的）核心词解析成槽位填充：实体 → 属性 → 两者
fn parse_slots(
    core: &str,
    ontology: &dyn OntologyView,
    turn: u64,
) -> Option<AskAttributeFrame> {
    if let Some((left, right)) = core.split_once('的') {
        let left = left.trim();
        let right = right.trim();
        if left.is_empty() || right.is_empty() {
            return None;
        }
        let entity = ontology.canonical_entity(left)?;
        let attribute = ontology.canonical_attribute(right)?;
        return Some(AskAttributeFrame::new(
            core,
            turn,
            Some(entity),
            Some(attribute),
        ));
    }
    if let Some(entity) = ontology.canonical_entity(core) {
        return Some(AskAttributeFrame::new(core, turn, Some(entity), None));
    }
    if let Some(attribute) = ontology.canonical_attribute(core) {
        return Some(AskAttributeFrame::new(core, turn, None, Some(attribute)));
    }
    None
}

/// 省略追问：那X呢 / X呢 / X的Y呢。核心词按本体排序分类为实体或属性洞。
pub fn parse_followup(
    input: &str,
    ontology: &dyn OntologyView,
    turn: u64,
) -> Option<AskAttributeFrame> {
    let trimmed = trim_tail(input);
    let core = if let Some(rest) = trimmed.strip_prefix("那") {
        let rest = rest.strip_prefix('么').unwrap_or(rest);
        rest.strip_suffix('呢')?
    } else if trimmed.ends_with('呢') {
        trimmed.strip_suffix('呢')?
    } else {
        return None;
    };
    let core = core.trim();
    if core.is_empty() || core.contains(char::is_whitespace) {
        return None;
    }
    if core.chars().count() > FOLLOWUP_MAX_CORE_CHARS {
        return None;
    }
    parse_slots(core, ontology, turn)
}

/// 无主语的属性追问：「体重是多少/是什么」——属性显式、实体留洞
pub fn parse_bare_attribute_question(
    input: &str,
    ontology: &dyn OntologyView,
    turn: u64,
) -> Option<AskAttributeFrame> {
    let trimmed = trim_tail(input);
    let stripped = strip_question_suffixes(strip_polite(trimmed));
    if stripped == trimmed || stripped.is_empty() {
        return None;
    }
    if stripped.contains('的') || stripped.contains(char::is_whitespace) {
        return None;
    }
    if stripped.chars().count() > FOLLOWUP_MAX_CORE_CHARS {
        return None;
    }
    let attribute = ontology.canonical_attribute(stripped)?;
    Some(AskAttributeFrame::new(
        trimmed,
        turn,
        None,
        Some(attribute),
    ))
}

/// 复合属性问句：A的P1和P2[和P3]（是多少/分别是什么…）
#[derive(Debug, Clone)]
pub struct CompoundQuestion {
    pub entity: String,
    pub frames: Vec<AskAttributeFrame>,
}

pub fn parse_compound(
    input: &str,
    ontology: &dyn OntologyView,
    turn: u64,
) -> Option<CompoundQuestion> {
    let trimmed = trim_tail(input);
    let core = strip_question_suffixes(strip_polite(trimmed));
    if core.is_empty() || core.chars().count() > 80 {
        return None;
    }
    let (entity_raw, rest) = core.split_once('的')?;
    let entity = ontology.canonical_entity(entity_raw.trim())?;
    let parts: Vec<&str> = rest
        .split(|c: char| matches!(c, '和' | '与' | '跟' | '及' | '、' | '，' | ',' | '；' | ';'))
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 2 {
        return None;
    }
    let mut frames = Vec::with_capacity(parts.len());
    for part in parts {
        let attribute = ontology.canonical_attribute(part)?;
        frames.push(AskAttributeFrame::new(
            trimmed,
            turn,
            Some(entity.clone()),
            Some(attribute),
        ));
    }
    Some(CompoundQuestion { entity, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct FakeOntology {
        entities: HashSet<&'static str>,
        attributes: HashSet<&'static str>,
    }

    impl FakeOntology {
        fn new() -> Self {
            Self {
                entities: ["张三", "李四", "姚明"].into_iter().collect(),
                attributes: ["身高", "体重", "师傅", "师祖"].into_iter().collect(),
            }
        }
    }

    impl OntologyView for FakeOntology {
        fn canonical_entity(&self, raw: &str) -> Option<String> {
            self.entities
                .contains(raw.trim())
                .then(|| raw.trim().to_string())
        }

        fn canonical_attribute(&self, raw: &str) -> Option<String> {
            self.attributes
                .contains(raw.trim())
                .then(|| raw.trim().to_string())
        }
    }

    fn frame(
        entity: Option<&str>,
        attribute: Option<&str>,
        turn: u64,
    ) -> AskAttributeFrame {
        AskAttributeFrame::new(
            "测试",
            turn,
            entity.map(str::to_string),
            attribute.map(str::to_string),
        )
    }

    #[test]
    fn unify_fills_entity_hole_from_previous_frame() {
        let older = frame(Some("姚明"), Some("身高"), 1);
        let newer = frame(None, Some("体重"), 2);
        let unified = unify_frames(&older, &newer).expect("同槽位排序相容应合一成功");
        assert_eq!(unified.entity.as_deref(), Some("姚明"));
        assert_eq!(unified.attribute.as_deref(), Some("体重"));
        assert_eq!(unified.turn, 2);
        assert_eq!(unified.entity_origin, SlotOrigin::Inherited { turn: 1 });
        assert_eq!(unified.attribute_origin, SlotOrigin::Explicit);
    }

    #[test]
    fn unify_fills_attribute_hole_from_previous_frame() {
        let older = frame(Some("张三"), Some("师傅"), 3);
        let newer = frame(Some("李四"), None, 4);
        let unified = unify_frames(&older, &newer).unwrap();
        assert_eq!(unified.entity.as_deref(), Some("李四"));
        assert_eq!(unified.attribute.as_deref(), Some("师傅"));
        assert_eq!(unified.attribute_origin, SlotOrigin::Inherited { turn: 3 });
    }

    #[test]
    fn unify_of_two_holes_stays_incomplete() {
        let older = frame(Some("张三"), None, 1);
        let newer = frame(None, None, 2);
        let unified = unify_frames(&older, &newer).expect("合一本身成功");
        assert!(!unified.is_complete());
    }

    #[test]
    fn followup_entity_inherits_last_attribute() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.record_last(frame(Some("张三"), Some("师祖"), memory.turn()));
        memory.begin_turn();
        let fragment = parse_followup("那李四呢", &ontology, memory.turn()).unwrap();
        let plan = memory.plan_followup(&fragment).expect("应能补全实体洞");
        let resolution = memory.commit_followup(plan);
        assert_eq!(resolution.question, "李四的师祖是什么");
        assert!(resolution.note.contains("实体「李四」←本轮"));
        assert!(resolution.note.contains("属性「师祖」←第 1 轮的完整帧"));
    }

    #[test]
    fn followup_attribute_inherits_last_entity() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.record_last(frame(Some("姚明"), Some("身高"), memory.turn()));
        memory.begin_turn();
        let fragment = parse_followup("体重呢", &ontology, memory.turn()).unwrap();
        let resolution = memory.commit_followup(memory.plan_followup(&fragment).unwrap());
        assert_eq!(resolution.question, "姚明的体重是什么");
        assert_eq!(
            resolution.frame.entity_origin,
            SlotOrigin::Inherited { turn: 1 }
        );
    }

    #[test]
    fn followup_attribute_prefers_and_closes_pending() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.record_last(frame(Some("姚明"), Some("身高"), memory.turn()));
        memory.push_pending(frame(Some("姚明"), Some("体重"), memory.turn()));
        memory.begin_turn();
        let fragment = parse_followup("体重呢", &ontology, memory.turn()).unwrap();
        let resolution = memory.commit_followup(memory.plan_followup(&fragment).unwrap());
        assert_eq!(resolution.question, "姚明的体重是什么");
        assert_eq!(
            resolution.frame.entity_origin,
            SlotOrigin::FromPending { turn: 1 }
        );
        assert!(resolution.note.contains("补全第 1 轮挂账"));
        assert!(memory.pending().is_empty());
    }

    #[test]
    fn unmatched_followup_keeps_pending() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.record_last(frame(Some("张三"), Some("师傅"), memory.turn()));
        memory.push_pending(frame(Some("姚明"), Some("体重"), memory.turn()));
        memory.begin_turn();
        let fragment = parse_followup("那李四呢", &ontology, memory.turn()).unwrap();
        let resolution = memory.commit_followup(memory.plan_followup(&fragment).unwrap());
        assert_eq!(resolution.question, "李四的师傅是什么");
        assert_eq!(memory.pending().len(), 1, "未命中的挂账不得被消费");
        assert!(resolution.closed_pending.is_none());
    }

    #[test]
    fn followup_failed_fill_leaves_state_untouched() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.push_pending(frame(Some("姚明"), Some("体重"), memory.turn()));
        memory.begin_turn();
        let fragment = parse_followup("那李四呢", &ontology, memory.turn()).unwrap();
        assert!(
            memory.plan_followup(&fragment).is_none(),
            "没有上一完整帧时无法补全属性洞"
        );
        assert_eq!(memory.pending().len(), 1);
        assert!(memory.last().is_none());
    }

    #[test]
    fn unknown_core_returns_none_for_fallback() {
        let ontology = FakeOntology::new();
        assert!(parse_followup("那某甲呢", &ontology, 1).is_none());
        assert!(parse_followup("某甲呢", &ontology, 1).is_none());
        assert!(parse_bare_attribute_question("某甲是多少", &ontology, 1).is_none());
    }

    #[test]
    fn bare_attribute_question_is_parsed() {
        let ontology = FakeOntology::new();
        let fragment = parse_bare_attribute_question("体重是多少", &ontology, 2).unwrap();
        assert!(fragment.entity.is_none());
        assert_eq!(fragment.attribute.as_deref(), Some("体重"));
    }

    #[test]
    fn compound_question_splits_attributes() {
        let ontology = FakeOntology::new();
        let compound = parse_compound("姚明的身高和体重是多少", &ontology, 1).unwrap();
        assert_eq!(compound.entity, "姚明");
        let attributes: Vec<&str> = compound
            .frames
            .iter()
            .filter_map(|frame| frame.attribute.as_deref())
            .collect();
        assert_eq!(attributes, vec!["身高", "体重"]);
        let compound2 = parse_compound("姚明的身高、体重分别是多少？", &ontology, 1).unwrap();
        assert_eq!(compound2.frames.len(), 2);
    }

    #[test]
    fn compound_rejects_entity_enumeration() {
        let ontology = FakeOntology::new();
        assert!(parse_compound("张三和李四的师傅是多少", &ontology, 1).is_none());
        assert!(parse_compound("姚明的身高和某物是多少", &ontology, 1).is_none());
    }

    #[test]
    fn record_compound_answers_first_and_pends_rest() {
        let ontology = FakeOntology::new();
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        let compound = parse_compound("姚明的身高和体重是多少", &ontology, memory.turn()).unwrap();
        memory.record_compound(compound.frames);
        assert_eq!(memory.last().map(|f| f.display()).as_deref(), Some("姚明 的 身高"));
        assert_eq!(memory.pending().len(), 1);
        memory.begin_turn();
        let fragment = parse_followup("体重呢", &ontology, memory.turn()).unwrap();
        let resolution = memory.commit_followup(memory.plan_followup(&fragment).unwrap());
        assert_eq!(resolution.question, "姚明的体重是什么");
        assert!(memory.pending().is_empty());
    }

    #[test]
    fn pending_expires_after_ttl() {
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.push_pending(frame(Some("姚明"), Some("体重"), memory.turn()));
        for _ in 0..DEFAULT_PENDING_TTL {
            memory.begin_turn();
        }
        assert_eq!(memory.pending().len(), 1, "TTL 内应保留");
        memory.begin_turn();
        assert!(memory.pending().is_empty(), "超过 TTL 应过期");
    }

    #[test]
    fn pending_capacity_is_bounded() {
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        for (index, attribute) in ["身高", "体重", "师傅", "师祖", "别名"].iter().enumerate() {
            memory.push_pending(frame(Some("姚明"), Some(attribute), memory.turn() + index as u64));
        }
        assert_eq!(memory.pending().len(), DEFAULT_MAX_PENDING);
        assert_eq!(
            memory.pending().first().and_then(|f| f.attribute.as_deref()),
            Some("体重"),
            "容量满时丢弃最旧挂账"
        );
    }

    #[test]
    fn report_lists_pending_with_turn_sources() {
        let mut memory = FrameMemory::new();
        memory.begin_turn();
        memory.push_pending(frame(Some("姚明"), Some("体重"), memory.turn()));
        let report = memory.report();
        assert!(report.contains("挂账"));
        assert!(report.contains("姚明 的 体重"));
        assert!(report.contains("第 1 轮挂起"));
    }

    // ------------------------- 澄清：变量求解的第 4 种合法结局 -------------------------

    fn pending_clarification(memory: &mut FrameMemory) -> ClarificationRequest {
        memory.begin_turn();
        memory.begin_clarification(
            "体重是多少",
            "体重",
            vec!["叶莉".to_string(), "姚明".to_string()],
        )
    }

    #[test]
    fn clarification_prompt_lists_candidates_deterministically() {
        let mut memory = FrameMemory::new();
        let request = pending_clarification(&mut memory);
        let prompt = request.prompt();
        assert!(prompt.contains("体重是多少"), "{}", prompt);
        assert!(prompt.contains("1. 叶莉") && prompt.contains("2. 姚明"), "{}", prompt);
        assert!(memory.report().contains("等待绑定"));
        assert!(memory.report().contains("1.叶莉；2.姚明"));
    }

    #[test]
    fn clarification_binds_by_index_and_completes_frame() {
        let mut memory = FrameMemory::new();
        pending_clarification(&mut memory);
        let outcome = memory.commit_clarification("姚明").expect("有澄清才能绑定");
        match outcome {
            ClarificationOutcome::Bound { question, note } => {
                assert_eq!(question, "姚明的体重是什么");
                assert!(note.contains("第 1 轮澄清"), "{}", note);
            }
            other => panic!("应为绑定：{:?}", other),
        }
        let last = memory.last().expect("绑定应成为上一帧");
        assert_eq!(last.entity.as_deref(), Some("姚明"));
        assert_eq!(last.attribute.as_deref(), Some("体重"));
        assert_eq!(last.entity_origin, SlotOrigin::Clarified { turn: 1 });
        assert!(memory.clarification().is_none(), "绑定后至多一条澄清应被消费");
    }

    #[test]
    fn clarification_reply_parses_index_names_and_cancel() {
        let mut memory = FrameMemory::new();
        let request = pending_clarification(&mut memory);
        assert_eq!(
            request.parse_reply("2"),
            ClarificationReply::Candidate("姚明".to_string())
        );
        assert_eq!(
            request.parse_reply("第2个"),
            ClarificationReply::Candidate("姚明".to_string())
        );
        assert_eq!(
            request.parse_reply("一"),
            ClarificationReply::Candidate("叶莉".to_string())
        );
        assert_eq!(
            request.parse_reply("姚明"),
            ClarificationReply::Candidate("姚明".to_string())
        );
        assert_eq!(request.parse_reply("8"), ClarificationReply::OutOfRange);
        assert_eq!(request.parse_reply("都不是"), ClarificationReply::Cancel);
        assert_eq!(request.parse_reply("算了"), ClarificationReply::Cancel);
        assert_eq!(request.parse_reply("诸葛亮是谁"), ClarificationReply::Unmatched);
    }

    #[test]
    fn clarification_out_of_range_reply_keeps_state() {
        let mut memory = FrameMemory::new();
        let request = pending_clarification(&mut memory);
        assert_eq!(request.parse_reply("9"), ClarificationReply::OutOfRange);
        let before = memory.report();
        // 重问路径：不提交、不消费、计数不变
        assert_eq!(memory.report(), before);
        assert!(memory.clarification().is_some());
    }

    #[test]
    fn clarification_cancel_releases_without_binding() {
        let mut memory = FrameMemory::new();
        pending_clarification(&mut memory);
        match memory.cancel_clarification() {
            Some(ClarificationOutcome::Cancelled { note }) => {
                assert!(note.contains("已放弃"), "{}", note);
            }
            other => panic!("应为取消：{:?}", other),
        }
        assert!(memory.clarification().is_none());
        assert!(memory.last().is_none(), "取消不得产生上一帧");
        assert!(memory.cancel_clarification().is_none(), "只能取消一次");
    }

    #[test]
    fn clarification_is_at_most_one_and_new_replaces_old() {
        let mut memory = FrameMemory::new();
        pending_clarification(&mut memory);
        memory.begin_clarification("身高是多少", "身高", vec!["姚明".to_string()]);
        let request = memory.clarification().expect("仍有澄清");
        assert_eq!(request.attribute, "身高");
        assert_eq!(request.candidates, vec!["姚明".to_string()]);
    }

    #[test]
    fn clarification_expires_after_ttl() {
        let mut memory = FrameMemory::new();
        pending_clarification(&mut memory);
        for _ in 0..DEFAULT_CLARIFICATION_TTL - 1 {
            memory.begin_turn();
        }
        assert!(memory.clarification().is_some(), "TTL 内应保留");
        memory.begin_turn();
        assert!(memory.clarification().is_none(), "超过 TTL 应过期");
        assert!(memory.report().contains("过期 1"));
    }

    #[test]
    fn clarification_survives_serde_roundtrip() {
        let mut memory = FrameMemory::new();
        pending_clarification(&mut memory);
        let json = serde_json::to_string(&memory).expect("帧记忆应可序列化");
        let restored: FrameMemory = serde_json::from_str(&json).expect("帧记忆应可反序列化");
        let request = restored.clarification().expect("澄清应随记忆持久化");
        assert_eq!(request.raw, "体重是多少");
        assert_eq!(request.candidates.len(), 2);
        // 缺少澄清字段的快照仍可读：缺省即无澄清
        let stored_json = r#"{"turn":2,"last":null,"pending":[],"max_pending":4,"pending_ttl":6,"unifications":0,"pending_closed":0}"#;
        let stored: FrameMemory = serde_json::from_str(stored_json).expect("缺少澄清字段的快照应可反序列化");
        assert!(stored.clarification().is_none());
    }
}

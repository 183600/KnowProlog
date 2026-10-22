//! 会话槽账本（Slot Ledger）：把「一条断言」建模成「一个槽的一次状态转移」。
//!
//! 设计立场（与 [`crate::knowledge_fusion`] 的导入期融合互补）：
//!   - **最小单位是槽 `(主语, 谓词)`，不是三元组**：三元组是取值，槽是取值的
//!     竞争场；功能谓词同槽只允许一个赢家，非功能谓词允许多值并存；
//!   - **更正不是追加**：`226 → 229` 记录为 before/after 状态转移，旧值进入
//!     「被压制」区而不是被删除或否定；压制是视图排序，不是逻辑否定；
//!   - **强度决定门控**：显式更正可自动替换；直陈遇到已有（或图谱）取值的
//!     功能槽必须问一句，零写入；传闻只进证据区，不进入回答；
//!   - **可解释即记忆**：每次转移都带序号与来源，能回答「第几次记录、原来
//!     是什么、为什么现在是这个」；
//!   - **可撤销**：撤销一次更正恢复 before 取值，而不是清空整个槽。
//!
//! 本模块只维护会话内的槽状态，不写知识图谱，也不替代
//! [`crate::knowledge_fusion::fusion_plan`] 的批量物化；它提供的是对话期
//! 增量裁决的同一条语义：同一个槽上，当前谁说了算、为什么、从什么时候开始。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// 断言强度：决定写入门控，而不是决定真假。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AssertionStrength {
    /// 显式更正（「更正：」「改口：」）：允许替换功能槽的当前赢家。
    Correction,
    /// 直陈（「记住：」「A 的 P 是 B」）：冲突时先问，不静默覆盖。
    Direct,
    /// 传闻（「据说」「好像」「我记得」）：只进证据区，不进入回答。
    Hearsay,
}

impl AssertionStrength {
    pub fn label_zh(&self) -> &'static str {
        match self {
            AssertionStrength::Correction => "更正",
            AssertionStrength::Direct => "直陈",
            AssertionStrength::Hearsay => "传闻",
        }
    }

    /// 从自然语言前缀识别强度，并返回去掉前缀后的正文。
    ///
    /// 只认封闭的提示词表，避免把普通陈述误判为更正：
    /// 更正类优先级最高，其次是传闻类，其余为直陈。
    pub fn classify(text: &str) -> (Self, &str) {
        let text = text.trim();
        for prefix in [
            "更正：",
            "更正:",
            "纠正：",
            "纠正:",
            "修正：",
            "修正:",
            "改口：",
            "改口:",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                return (AssertionStrength::Correction, rest.trim());
            }
        }
        for prefix in ["据说", "听说", "好像", "我记得", "据传"] {
            if let Some(rest) = text.strip_prefix(prefix) {
                return (AssertionStrength::Hearsay, rest.trim());
            }
        }
        (AssertionStrength::Direct, text)
    }
}

/// 功能谓词的保守默认表：同一主语上只允许一个当前赢家。
///
/// 这些谓词在中文对话里最常见的语义就是「单值」；非功能谓词（职业、作者、
/// 别名等）不在此表，允许多值并存。调用方可用
/// [`SlotPolicy::with_functional`] 叠加谓词画像/律典里的功能声明。
pub const DEFAULT_FUNCTIONAL_PREDICATES: &[&str] = &[
    "年龄",
    "身高",
    "体重",
    "出生地",
    "出生于",
    "籍贯",
    "国籍",
    "民族",
    "性别",
    "生日",
    "出生日期",
    "逝世日期",
    "去世日期",
    "居住于",
    "位于",
    "成立时间",
    "成立日期",
    "首都",
    "面积",
    "人口",
    "age",
    "height",
    "weight",
    "birthplace",
    "birthdate",
    "nationality",
    "gender",
    "location",
    "located_in",
    "capital",
    "population",
    "area",
];

/// 槽策略：哪些谓词按功能谓词处理。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotPolicy {
    functional: BTreeSet<String>,
}

impl SlotPolicy {
    /// 使用默认功能谓词表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 使用空功能谓词表（用于严格测试或完全由调用方声明的场景）。
    pub fn empty() -> Self {
        Self {
            functional: BTreeSet::new(),
        }
    }

    /// 在默认表基础上追加功能谓词。
    pub fn with_defaults() -> Self {
        let mut policy = Self::default();
        for predicate in DEFAULT_FUNCTIONAL_PREDICATES {
            policy.functional.insert((*predicate).to_string());
        }
        policy
    }

    /// 追加功能谓词。
    pub fn with_functional<I, S>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for predicate in predicates {
            let predicate = predicate.into();
            if !predicate.trim().is_empty() {
                self.functional.insert(predicate.trim().to_string());
            }
        }
        self
    }

    /// 显式声明一个功能谓词或取消声明。
    pub fn declare_functional(&mut self, predicate: &str, functional: bool) {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return;
        }
        if functional {
            self.functional.insert(predicate.to_string());
        } else {
            self.functional.remove(predicate);
        }
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.functional.contains(predicate.trim())
    }

    pub fn functional_predicates(&self) -> impl Iterator<Item = &str> {
        self.functional.iter().map(String::as_str)
    }

    /// 从谓词画像/律典的约束集合继承功能声明（不覆盖默认表）。
    pub fn adopt_functional<I, S>(&mut self, predicates: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for predicate in predicates {
            let predicate = predicate.into();
            if !predicate.trim().is_empty() {
                self.functional.insert(predicate.trim().to_string());
            }
        }
    }
}

/// 槽键：`(主语, 谓词)`。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SlotKey {
    pub subject: String,
    pub predicate: String,
}

impl SlotKey {
    pub fn new(subject: &str, predicate: &str) -> Self {
        Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
        }
    }

    pub fn label_zh(&self) -> String {
        format!("{} 的 {}", self.subject, self.predicate)
    }
}

/// 一条待写入的槽断言。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotAssertion {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub strength: AssertionStrength,
    /// 来源说明（「记住」「改口」「图谱」等），用于历史与解释。
    pub source: String,
}

impl SlotAssertion {
    pub fn new(
        subject: &str,
        predicate: &str,
        object: &str,
        strength: AssertionStrength,
        source: &str,
    ) -> Self {
        Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            object: object.trim().to_string(),
            strength,
            source: source.trim().to_string(),
        }
    }

    pub fn key(&self) -> SlotKey {
        SlotKey::new(&self.subject, &self.predicate)
    }

    fn is_valid(&self) -> bool {
        !self.subject.is_empty() && !self.predicate.is_empty() && !self.object.is_empty()
    }
}

/// 一个被采用或被压制的取值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotValue {
    pub object: String,
    /// 支撑来源（去重、按首次出现顺序）。
    pub sources: Vec<String>,
    pub strength: AssertionStrength,
    pub first_step: u64,
    pub last_step: u64,
}

impl SlotValue {
    fn new(object: &str, source: &str, strength: AssertionStrength, step: u64) -> Self {
        Self {
            object: object.to_string(),
            sources: vec![source.to_string()],
            strength,
            first_step: step,
            last_step: step,
        }
    }

    fn support(&mut self, source: &str, strength: AssertionStrength, step: u64) {
        if !self.sources.iter().any(|existing| existing == source) {
            self.sources.push(source.to_string());
        }
        if strength > self.strength {
            self.strength = strength;
        }
        self.last_step = step;
    }

    pub fn source_label(&self) -> String {
        self.sources.join(" / ")
    }
}

/// 槽事件种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SlotEventKind {
    Assert,
    Correct,
    Corroborate,
    Retract,
    Suppress,
    Restore,
    Evidence,
    Conflict,
}

impl SlotEventKind {
    pub fn label_zh(&self) -> &'static str {
        match self {
            SlotEventKind::Assert => "直陈",
            SlotEventKind::Correct => "更正",
            SlotEventKind::Corroborate => "佐证",
            SlotEventKind::Retract => "撤回",
            SlotEventKind::Suppress => "压制",
            SlotEventKind::Restore => "恢复",
            SlotEventKind::Evidence => "证据",
            SlotEventKind::Conflict => "待裁决",
        }
    }
}

/// 一条不可变的槽事件：用于回答「第几次、改了什么、为什么」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotEvent {
    pub step: u64,
    pub kind: SlotEventKind,
    pub object: String,
    pub strength: AssertionStrength,
    pub source: String,
    /// 事件发生前的采用值。
    pub before: Vec<String>,
    /// 事件发生后的采用值。
    pub after: Vec<String>,
}

impl SlotEvent {
    /// 人类可读的一行解释。
    pub fn render_zh(&self) -> String {
        let before = if self.before.is_empty() {
            "（空）".to_string()
        } else {
            self.before.join(" / ")
        };
        let after = if self.after.is_empty() {
            "（空）".to_string()
        } else {
            self.after.join(" / ")
        };
        match self.kind {
            SlotEventKind::Correct => format!(
                "第 {} 次记录：{}「{}」（{} → {}；旧值被压制但未否定）",
                self.step,
                self.kind.label_zh(),
                self.object,
                before,
                after
            ),
            SlotEventKind::Restore => format!(
                "第 {} 次记录：{}「{}」（{} → {}）",
                self.step,
                self.kind.label_zh(),
                self.object,
                before,
                after
            ),
            SlotEventKind::Conflict => format!(
                "第 {} 次记录：{}，未写入「{}」（当前采用：{}；来源：{}）",
                self.step,
                self.kind.label_zh(),
                self.object,
                before,
                self.source
            ),
            _ => format!(
                "第 {} 次记录：{}「{}」（{}；当前采用：{}）",
                self.step,
                self.kind.label_zh(),
                self.object,
                self.source,
                after
            ),
        }
    }
}

/// 一次状态转移的增量结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotDelta {
    /// 槽原本为空，接受新值。
    Added,
    /// 同值同来源，幂等。
    Duplicate,
    /// 同值不同来源，多源佐证。
    Corroborated,
    /// 显式更正替换了功能槽的采用值。
    Corrected { from: Vec<String> },
    /// 直陈与当前采用值（或图谱基值）冲突，未写入。
    NeedsConfirmation {
        current: Vec<String>,
        requested: String,
    },
    /// 传闻只进证据区。
    HearsayRecorded,
    /// 撤回了一个采用值。
    Retracted { object: String },
    /// 撤销一次更正，恢复旧值。
    Restored { from: Vec<String> },
    /// 输入不完整，零写入。
    Invalid,
}

impl SlotDelta {
    pub fn label_zh(&self) -> &'static str {
        match self {
            SlotDelta::Added => "新增",
            SlotDelta::Duplicate => "重复",
            SlotDelta::Corroborated => "佐证",
            SlotDelta::Corrected { .. } => "更正",
            SlotDelta::NeedsConfirmation { .. } => "待裁决",
            SlotDelta::HearsayRecorded => "证据",
            SlotDelta::Retracted { .. } => "撤回",
            SlotDelta::Restored { .. } => "恢复",
            SlotDelta::Invalid => "无效",
        }
    }

    pub fn changed_state(&self) -> bool {
        matches!(
            self,
            SlotDelta::Added
                | SlotDelta::Corroborated
                | SlotDelta::Corrected { .. }
                | SlotDelta::Retracted { .. }
                | SlotDelta::Restored { .. }
                | SlotDelta::HearsayRecorded
        )
    }
}

/// 一次槽状态转移的完整回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotTransition {
    pub key: SlotKey,
    pub functional: bool,
    pub delta: SlotDelta,
    pub before: Vec<String>,
    pub after: Vec<String>,
    /// 当前被压制（视图层落选、逻辑上未否定）的取值。
    pub suppressed: Vec<String>,
    /// 当前只作为证据、不进入回答的取值。
    pub evidence: Vec<String>,
    pub step: u64,
}

impl SlotTransition {
    pub fn changed_state(&self) -> bool {
        self.delta.changed_state()
    }

    pub fn slot_label(&self) -> String {
        self.key.label_zh()
    }

    pub fn should_confirm(&self) -> bool {
        matches!(self.delta, SlotDelta::NeedsConfirmation { .. })
    }

    /// 面向用户的中文回执；显式区分「压制」与「否定」。
    pub fn render_zh(&self) -> String {
        match &self.delta {
            SlotDelta::Added => format!(
                "已记录：{} = {}（第 {} 次记录）。",
                self.slot_label(),
                self.after.join(" / "),
                self.step
            ),
            SlotDelta::Duplicate => format!(
                "「{} = {}」已经记住，本次没有重复写入。",
                self.slot_label(),
                self.after.join(" / ")
            ),
            SlotDelta::Corroborated => format!(
                "「{} = {}」得到多来源佐证（{}）。",
                self.slot_label(),
                self.after.join(" / "),
                self.suppressed.join(" / ")
            ),
            SlotDelta::Corrected { from } => format!(
                "已按你的更正更新：{} 由「{}」改为「{}」（第 {} 次记录；旧值被压制但未否定，可用「撤销修订」恢复）。",
                self.slot_label(),
                from.join(" / "),
                self.after.join(" / "),
                self.step
            ),
            SlotDelta::NeedsConfirmation { current, requested } => format!(
                "「{}」当前采用「{}」，与新的取值「{}」冲突；本次未写入。若确认改口，请用「改口：{}的{}是{}」。",
                self.slot_label(),
                current.join(" / "),
                requested,
                self.key.subject,
                self.key.predicate,
                requested
            ),
            SlotDelta::HearsayRecorded => format!(
                "已把传闻记为证据：{} = {}（不进入回答，也不覆盖当前取值）。",
                self.slot_label(),
                self.evidence.join(" / ")
            ),
            SlotDelta::Retracted { object } => format!(
                "已撤回会话事实：{} = {}（当前采用：{}）。",
                self.slot_label(),
                object,
                if self.after.is_empty() {
                    "（空）".to_string()
                } else {
                    self.after.join(" / ")
                }
            ),
            SlotDelta::Restored { from } => format!(
                "已撤销本次更正：{} 由「{}」恢复为「{}」。",
                self.slot_label(),
                from.join(" / "),
                self.after.join(" / ")
            ),
            SlotDelta::Invalid => "断言缺少主语、谓词或取值，未写入。".to_string(),
        }
    }
}

/// 一个槽的完整状态。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct SlotState {
    accepted: Vec<SlotValue>,
    suppressed: Vec<SlotValue>,
    evidence: Vec<SlotValue>,
    history: Vec<SlotEvent>,
}

impl SlotState {
    fn accepted_objects(&self) -> Vec<String> {
        self.accepted
            .iter()
            .map(|value| value.object.clone())
            .collect()
    }

    fn suppressed_objects(&self) -> Vec<String> {
        self.suppressed
            .iter()
            .map(|value| value.object.clone())
            .collect()
    }

    fn evidence_objects(&self) -> Vec<String> {
        self.evidence
            .iter()
            .map(|value| value.object.clone())
            .collect()
    }

    fn find_accepted(&mut self, object: &str) -> Option<&mut SlotValue> {
        self.accepted
            .iter_mut()
            .find(|value| value.object == object)
    }

    fn push_suppressed(&mut self, mut value: SlotValue) {
        if let Some(existing) = self
            .suppressed
            .iter_mut()
            .find(|candidate| candidate.object == value.object)
        {
            existing.last_step = value.last_step;
            for source in value.sources.drain(..) {
                if !existing.sources.contains(&source) {
                    existing.sources.push(source);
                }
            }
            return;
        }
        self.suppressed.push(value);
    }

    fn push_evidence(&mut self, mut value: SlotValue) {
        if let Some(existing) = self
            .evidence
            .iter_mut()
            .find(|candidate| candidate.object == value.object)
        {
            existing.last_step = value.last_step;
            for source in value.sources.drain(..) {
                if !existing.sources.contains(&source) {
                    existing.sources.push(source);
                }
            }
            return;
        }
        self.evidence.push(value);
    }

    fn remove_suppressed(&mut self, object: &str) -> Option<SlotValue> {
        let position = self
            .suppressed
            .iter()
            .position(|value| value.object == object)?;
        Some(self.suppressed.remove(position))
    }
}

/// 会话槽账本：`(主语, 谓词)` → 采用值 / 压制值 / 证据值 / 事件历史。
///
/// 内部用 `BTreeMap` 保证确定性遍历；序列化时展开为有序数组，避免 JSON
/// 对象键必须是字符串的限制（槽键是结构体）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotLedger {
    policy: SlotPolicy,
    slots: BTreeMap<SlotKey, SlotState>,
    step: u64,
}

impl Default for SlotLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize)]
struct SlotLedgerSnapshot {
    policy: SlotPolicy,
    step: u64,
    slots: Vec<(SlotKey, SlotState)>,
}

impl Serialize for SlotLedger {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let snapshot = SlotLedgerSnapshot {
            policy: self.policy.clone(),
            step: self.step,
            slots: self
                .slots
                .iter()
                .map(|(key, state)| (key.clone(), state.clone()))
                .collect(),
        };
        snapshot.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SlotLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let snapshot = SlotLedgerSnapshot::deserialize(deserializer)?;
        Ok(Self {
            policy: snapshot.policy,
            slots: snapshot.slots.into_iter().collect(),
            step: snapshot.step,
        })
    }
}

impl SlotLedger {
    /// 使用默认功能谓词表。
    pub fn new() -> Self {
        Self::with_policy(SlotPolicy::with_defaults())
    }

    pub fn with_policy(policy: SlotPolicy) -> Self {
        Self {
            policy,
            slots: BTreeMap::new(),
            step: 0,
        }
    }

    pub fn policy(&self) -> &SlotPolicy {
        &self.policy
    }

    pub fn policy_mut(&mut self) -> &mut SlotPolicy {
        &mut self.policy
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.policy.is_functional(predicate)
    }

    pub fn keys(&self) -> impl Iterator<Item = &SlotKey> {
        self.slots.keys()
    }

    pub fn accepted(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.slots
            .get(&SlotKey::new(subject, predicate))
            .map(SlotState::accepted_objects)
            .unwrap_or_default()
    }

    pub fn suppressed(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.slots
            .get(&SlotKey::new(subject, predicate))
            .map(SlotState::suppressed_objects)
            .unwrap_or_default()
    }

    pub fn evidence(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.slots
            .get(&SlotKey::new(subject, predicate))
            .map(SlotState::evidence_objects)
            .unwrap_or_default()
    }

    /// 槽账本到会话事实的投影：只包含当前采用的取值。
    pub fn projection(&self) -> Vec<(String, String, String)> {
        let mut projected = Vec::new();
        for (key, state) in &self.slots {
            for value in &state.accepted {
                projected.push((
                    key.subject.clone(),
                    key.predicate.clone(),
                    value.object.clone(),
                ));
            }
        }
        projected
    }

    /// 历史（只读）。
    pub fn history(&self, subject: &str, predicate: &str) -> &[SlotEvent] {
        self.slots
            .get(&SlotKey::new(subject, predicate))
            .map(|state| state.history.as_slice())
            .unwrap_or(&[])
    }

    /// 是否有任何历史（用于快照省略空账本）。
    pub fn has_history(&self) -> bool {
        self.slots.values().any(|state| !state.history.is_empty())
    }

    /// 应用一条断言。`base_values` 是同一槽上来自知识图谱的只读基值：
    /// 直陈遇到不同基值时只提问、零写入；显式更正才允许压制基值。
    pub fn apply(
        &mut self,
        assertion: SlotAssertion,
        base_values: &[String],
    ) -> Result<SlotTransition, String> {
        if !assertion.is_valid() {
            return Ok(SlotTransition {
                key: assertion.key(),
                functional: false,
                delta: SlotDelta::Invalid,
                before: Vec::new(),
                after: Vec::new(),
                suppressed: Vec::new(),
                evidence: Vec::new(),
                step: self.step,
            });
        }
        let key = assertion.key();
        let functional = self.policy.is_functional(&key.predicate);
        let state = self.slots.entry(key.clone()).or_default();
        let before = state.accepted_objects();
        let base_values: Vec<String> = base_values
            .iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
        let current: Vec<String> = if before.is_empty() {
            base_values.clone()
        } else {
            before.clone()
        };

        // 传闻只进证据区：不覆盖、不进入回答。
        if assertion.strength == AssertionStrength::Hearsay {
            let step = {
                self.step += 1;
                self.step
            };
            state.push_evidence(SlotValue::new(
                &assertion.object,
                &assertion.source,
                AssertionStrength::Hearsay,
                step,
            ));
            let event = SlotEvent {
                step,
                kind: SlotEventKind::Evidence,
                object: assertion.object.clone(),
                strength: assertion.strength,
                source: assertion.source.clone(),
                before: before.clone(),
                after: before.clone(),
            };
            state.history.push(event);
            return Ok(Self::transition(
                key,
                functional,
                SlotDelta::HearsayRecorded,
                before,
                state,
                step,
            ));
        }

        // 新值正是图谱基值（且基值唯一）：这不是冲突，而是把图谱已支持的
        // 取值显式纳入会话投影；直陈与更正都允许。
        if before.is_empty() && base_values.len() == 1 && base_values[0] == assertion.object {
            let step = {
                self.step += 1;
                self.step
            };
            state.accepted.push(SlotValue::new(
                &assertion.object,
                &assertion.source,
                assertion.strength,
                step,
            ));
            let event = SlotEvent {
                step,
                kind: SlotEventKind::Assert,
                object: assertion.object.clone(),
                strength: assertion.strength,
                source: assertion.source.clone(),
                before: Vec::new(),
                after: state.accepted_objects(),
            };
            state.history.push(event);
            return Ok(Self::transition(
                key,
                functional,
                SlotDelta::Added,
                Vec::new(),
                state,
                step,
            ));
        }

        // 同值：多来源佐证或幂等重复。
        if before.iter().any(|value| value == &assertion.object) {
            let step = {
                self.step += 1;
                self.step
            };
            let mut corroborated = false;
            if let Some(existing) = state.find_accepted(&assertion.object) {
                let known = existing.sources.iter().any(|s| s == &assertion.source);
                existing.support(&assertion.source, assertion.strength, step);
                corroborated = !known;
            }
            let delta = if corroborated {
                SlotDelta::Corroborated
            } else {
                SlotDelta::Duplicate
            };
            let event = SlotEvent {
                step,
                kind: if corroborated {
                    SlotEventKind::Corroborate
                } else {
                    SlotEventKind::Assert
                },
                object: assertion.object.clone(),
                strength: assertion.strength,
                source: assertion.source.clone(),
                before: before.clone(),
                after: state.accepted_objects(),
            };
            state.history.push(event);
            return Ok(Self::transition(
                key, functional, delta, before, state, step,
            ));
        }

        // 功能槽已有不同取值：显式更正才替换，直陈先问。
        if functional && !current.is_empty() {
            if assertion.strength != AssertionStrength::Correction {
                let step = self.step;
                return Ok(SlotTransition {
                    key,
                    functional,
                    delta: SlotDelta::NeedsConfirmation {
                        current: current.clone(),
                        requested: assertion.object.clone(),
                    },
                    before: before.clone(),
                    after: current.clone(),
                    suppressed: state.suppressed_objects(),
                    evidence: state.evidence_objects(),
                    step,
                });
            }
            let step = {
                self.step += 1;
                self.step
            };
            let old: Vec<SlotValue> = state.accepted.drain(..).collect();
            for value in old {
                state.push_suppressed(value);
            }
            // 图谱基值进入压制区（仅记录，不复制成会话事实）。
            for value in &base_values {
                if value != &assertion.object {
                    state.push_suppressed(SlotValue::new(
                        value,
                        "知识图谱",
                        AssertionStrength::Direct,
                        step,
                    ));
                }
            }
            state.accepted.push(SlotValue::new(
                &assertion.object,
                &assertion.source,
                assertion.strength,
                step,
            ));
            let event = SlotEvent {
                step,
                kind: SlotEventKind::Correct,
                object: assertion.object.clone(),
                strength: assertion.strength,
                source: assertion.source.clone(),
                before: current.clone(),
                after: state.accepted_objects(),
            };
            state.history.push(event);
            let delta = SlotDelta::Corrected { from: current };
            return Ok(Self::transition(
                key, functional, delta, before, state, step,
            ));
        }

        // 功能槽原本为空，但图谱已有不同基值：直陈先问，更正才压制。
        if functional
            && before.is_empty()
            && base_values
                .iter()
                .any(|value| value.as_str() != assertion.object.as_str())
        {
            if assertion.strength != AssertionStrength::Correction {
                let step = self.step;
                return Ok(SlotTransition {
                    key,
                    functional,
                    delta: SlotDelta::NeedsConfirmation {
                        current: base_values.clone(),
                        requested: assertion.object.clone(),
                    },
                    before: Vec::new(),
                    after: base_values,
                    suppressed: state.suppressed_objects(),
                    evidence: state.evidence_objects(),
                    step,
                });
            }
            let step = {
                self.step += 1;
                self.step
            };
            for value in &base_values {
                state.push_suppressed(SlotValue::new(
                    value,
                    "知识图谱",
                    AssertionStrength::Direct,
                    step,
                ));
            }
            state.accepted.push(SlotValue::new(
                &assertion.object,
                &assertion.source,
                assertion.strength,
                step,
            ));
            let event = SlotEvent {
                step,
                kind: SlotEventKind::Correct,
                object: assertion.object.clone(),
                strength: assertion.strength,
                source: assertion.source.clone(),
                before: base_values.clone(),
                after: state.accepted_objects(),
            };
            state.history.push(event);
            let delta = SlotDelta::Corrected { from: base_values };
            return Ok(Self::transition(
                key,
                functional,
                delta,
                Vec::new(),
                state,
                step,
            ));
        }

        // 空槽（或非功能槽）接受新值。
        let step = {
            self.step += 1;
            self.step
        };
        state.accepted.push(SlotValue::new(
            &assertion.object,
            &assertion.source,
            assertion.strength,
            step,
        ));
        let event = SlotEvent {
            step,
            kind: if assertion.strength == AssertionStrength::Correction {
                SlotEventKind::Correct
            } else {
                SlotEventKind::Assert
            },
            object: assertion.object.clone(),
            strength: assertion.strength,
            source: assertion.source.clone(),
            before: before.clone(),
            after: state.accepted_objects(),
        };
        state.history.push(event);
        Ok(Self::transition(
            key,
            functional,
            SlotDelta::Added,
            before,
            state,
            step,
        ))
    }

    /// 撤回某个槽上的一个采用值（`object` 为 `None` 时撤回全部采用值）。
    pub fn retract(
        &mut self,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
    ) -> Vec<SlotTransition> {
        let key = SlotKey::new(subject, predicate);
        let Some(state) = self.slots.get_mut(&key) else {
            return Vec::new();
        };
        let targets: Vec<String> = match object {
            Some(object) => state
                .accepted
                .iter()
                .filter(|value| value.object == object)
                .map(|value| value.object.clone())
                .collect(),
            None => state.accepted_objects(),
        };
        let mut transitions = Vec::new();
        for target in targets {
            let before = state.accepted_objects();
            state.accepted.retain(|value| value.object != target);
            let step = {
                self.step += 1;
                self.step
            };
            let event = SlotEvent {
                step,
                kind: SlotEventKind::Retract,
                object: target.clone(),
                strength: AssertionStrength::Direct,
                source: "撤回".to_string(),
                before: before.clone(),
                after: state.accepted_objects(),
            };
            state.history.push(event);
            transitions.push(Self::transition(
                key.clone(),
                self.policy.is_functional(&key.predicate),
                SlotDelta::Retracted { object: target },
                before,
                state,
                step,
            ));
        }
        transitions
    }

    /// 撤销某个槽最近一次更正：恢复更正前的采用值。
    pub fn undo_last_correction(
        &mut self,
        subject: &str,
        predicate: &str,
    ) -> Option<SlotTransition> {
        let key = SlotKey::new(subject, predicate);
        let functional = self.policy.is_functional(&key.predicate);
        let state = self.slots.get_mut(&key)?;
        let position = state
            .history
            .iter()
            .rposition(|event| event.kind == SlotEventKind::Correct)?;
        let target = state.history[position].clone();
        let before = state.accepted_objects();
        let step = {
            self.step += 1;
            self.step
        };
        // 当前取值进入压制区；更正前的取值恢复到采用区。
        let current: Vec<SlotValue> = state.accepted.drain(..).collect();
        for value in current {
            state.push_suppressed(value);
        }
        let mut restored = Vec::new();
        for object in &target.before {
            if let Some(mut value) = state.remove_suppressed(object) {
                value.last_step = step;
                value.sources.push("撤销更正".to_string());
                state.accepted.push(value);
            } else {
                state.accepted.push(SlotValue::new(
                    object,
                    "撤销更正",
                    AssertionStrength::Direct,
                    step,
                ));
            }
            restored.push(object.clone());
        }
        let event = SlotEvent {
            step,
            kind: SlotEventKind::Restore,
            object: restored.join(" / "),
            strength: AssertionStrength::Direct,
            source: "撤销更正".to_string(),
            before: before.clone(),
            after: state.accepted_objects(),
        };
        state.history.push(event);
        Some(Self::transition(
            key,
            functional,
            SlotDelta::Restored { from: before },
            state.accepted_objects(),
            state,
            step,
        ))
    }

    /// 从一批已存在的会话事实重建账本（快照缺省时使用）。
    ///
    /// 不做功能槽收敛：重建必须逐字保留事实集，否则账本会与读模型不一致。
    /// 冲突留给下一次显式更正或账本报告暴露。
    pub fn rebuild_from_facts<I>(facts: I, policy: SlotPolicy) -> Self
    where
        I: IntoIterator<Item = (String, String, String)>,
    {
        let mut ledger = Self::with_policy(policy);
        let mut step = 1u64;
        for (subject, predicate, object) in facts {
            let key = SlotKey::new(&subject, &predicate);
            let state = ledger.slots.entry(key).or_default();
            let before = state.accepted_objects();
            state.accepted.push(SlotValue::new(
                &object,
                "会话快照",
                AssertionStrength::Direct,
                step,
            ));
            let after = state.accepted_objects();
            state.history.push(SlotEvent {
                step,
                kind: SlotEventKind::Assert,
                object,
                strength: AssertionStrength::Direct,
                source: "会话快照".to_string(),
                before,
                after,
            });
            step += 1;
        }
        ledger.step = step.saturating_sub(1);
        ledger
    }

    /// 槽报告：所有槽的当前赢家与竞争值，按槽名排序。
    pub fn render_report(&self) -> String {
        if self.slots.is_empty() {
            return "槽账本：（空；还没有会话级槽断言）".to_string();
        }
        let functional: Vec<&str> = self.policy.functional_predicates().collect();
        let mut lines = vec![format!(
            "槽账本：{} 个槽，{} 次记录；功能谓词 {} 个（{}）。",
            self.slots.len(),
            self.step,
            functional.len(),
            if functional.is_empty() {
                "（无）".to_string()
            } else {
                functional.join("、")
            }
        )];
        for (key, state) in &self.slots {
            lines.push(format!(
                "  · {}（{}）当前采用：{}",
                key.label_zh(),
                if self.policy.is_functional(&key.predicate) {
                    "功能槽"
                } else {
                    "多值槽"
                },
                if state.accepted.is_empty() {
                    "（空）".to_string()
                } else {
                    state
                        .accepted
                        .iter()
                        .map(|value| format!("{}〔{}〕", value.object, value.source_label()))
                        .collect::<Vec<_>>()
                        .join(" / ")
                }
            ));
            if !state.suppressed.is_empty() {
                lines.push(format!(
                    "      压制（视图层，不否定）：{}",
                    state
                        .suppressed
                        .iter()
                        .map(|value| format!("{}〔{}〕", value.object, value.source_label()))
                        .collect::<Vec<_>>()
                        .join(" / ")
                ));
            }
            if !state.evidence.is_empty() {
                lines.push(format!(
                    "      证据（仅传闻，不进入回答）：{}",
                    state
                        .evidence
                        .iter()
                        .map(|value| format!("{}〔{}〕", value.object, value.source_label()))
                        .collect::<Vec<_>>()
                        .join(" / ")
                ));
            }
        }
        lines.join("\n")
    }

    /// 单个槽的完整改口史。
    pub fn render_history(&self, subject: &str, predicate: &str) -> String {
        let key = SlotKey::new(subject, predicate);
        let Some(state) = self.slots.get(&key) else {
            return format!("槽账本中没有「{}」的记录。", key.label_zh());
        };
        let mut lines = vec![format!(
            "「{}」的改口史（{} 条事件）：",
            key.label_zh(),
            state.history.len()
        )];
        if state.history.is_empty() {
            lines.push("  （无事件）".to_string());
        }
        for event in &state.history {
            lines.push(format!("  {}", event.render_zh()));
        }
        lines.push(format!(
            "  当前采用：{}",
            if state.accepted.is_empty() {
                "（空）".to_string()
            } else {
                state.accepted_objects().join(" / ")
            }
        ));
        if !state.suppressed.is_empty() {
            lines.push(format!(
                "  已压制（仍保留证据，不等于否定）：{}",
                state.suppressed_objects().join(" / ")
            ));
        }
        if !state.evidence.is_empty() {
            lines.push(format!(
                "  仅证据（传闻，不进入回答）：{}",
                state.evidence_objects().join(" / ")
            ));
        }
        lines.join("\n")
    }

    fn transition(
        key: SlotKey,
        functional: bool,
        delta: SlotDelta,
        before: Vec<String>,
        state: &SlotState,
        step: u64,
    ) -> SlotTransition {
        SlotTransition {
            key,
            functional,
            delta,
            before,
            after: state.accepted_objects(),
            suppressed: state.suppressed_objects(),
            evidence: state.evidence_objects(),
            step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(subject: &str, predicate: &str, object: &str, source: &str) -> SlotAssertion {
        SlotAssertion::new(
            subject,
            predicate,
            object,
            AssertionStrength::Direct,
            source,
        )
    }

    fn correction(subject: &str, predicate: &str, object: &str) -> SlotAssertion {
        SlotAssertion::new(
            subject,
            predicate,
            object,
            AssertionStrength::Correction,
            "改口",
        )
    }

    #[test]
    fn correction_records_before_and_after_instead_of_appending() {
        let mut ledger = SlotLedger::new();
        let first = ledger
            .apply(direct("姚明", "身高", "226", "记住"), &[])
            .expect("合法断言");
        assert_eq!(first.delta, SlotDelta::Added);
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["226"]);

        let second = ledger
            .apply(correction("姚明", "身高", "229"), &[])
            .expect("合法断言");
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["229"]);
        assert_eq!(ledger.suppressed("姚明", "身高"), vec!["226"]);
        assert!(second.changed_state());
        assert!(second.render_zh().contains("226"));
        assert!(second.render_zh().contains("229"));
        assert_eq!(ledger.history("姚明", "身高").len(), 2);
    }

    #[test]
    fn direct_conflict_on_functional_slot_asks_with_zero_write() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("姚明", "身高", "226", "记住"), &[])
            .unwrap();
        let step_before = ledger.step();
        let transition = ledger
            .apply(direct("姚明", "身高", "229", "记住"), &[])
            .unwrap();
        assert!(transition.should_confirm());
        assert_eq!(transition.after, vec!["226"]);
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["226"]);
        assert_eq!(ledger.step(), step_before, "提问不得推进状态");
        assert!(transition.render_zh().contains("改口"));
        assert!(
            transition.render_zh().contains("改口：姚明的身高是229"),
            "建议命令必须可直接执行：{}",
            transition.render_zh()
        );
    }

    #[test]
    fn direct_conflict_against_graph_base_asks_until_corrected() {
        let mut ledger = SlotLedger::new();
        let base = vec!["226".to_string()];
        let asked = ledger
            .apply(direct("姚明", "身高", "229", "记住"), &base)
            .unwrap();
        assert!(asked.should_confirm());
        assert!(ledger.accepted("姚明", "身高").is_empty());

        let corrected = ledger
            .apply(correction("姚明", "身高", "229"), &base)
            .unwrap();
        assert!(matches!(corrected.delta, SlotDelta::Corrected { .. }));
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["229"]);
        assert_eq!(ledger.suppressed("姚明", "身高"), vec!["226"]);
    }

    #[test]
    fn non_functional_slot_accepts_multiple_values() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("张三", "职业", "教师", "记住"), &[])
            .unwrap();
        ledger
            .apply(direct("张三", "职业", "作家", "记住"), &[])
            .unwrap();
        let mut accepted = ledger.accepted("张三", "职业");
        accepted.sort();
        assert_eq!(accepted, vec!["作家", "教师"]);
        assert!(!ledger.is_functional("职业"));
    }

    #[test]
    fn hearsay_only_lands_in_evidence() {
        let mut ledger = SlotLedger::new();
        let hearsay = SlotAssertion::new("姚明", "身高", "230", AssertionStrength::Hearsay, "据说");
        let transition = ledger.apply(hearsay, &[]).unwrap();
        assert_eq!(transition.delta, SlotDelta::HearsayRecorded);
        assert!(ledger.accepted("姚明", "身高").is_empty());
        assert_eq!(ledger.evidence("姚明", "身高"), vec!["230"]);
    }

    #[test]
    fn direct_assertion_of_the_only_graph_value_is_not_a_conflict() {
        let mut ledger = SlotLedger::new();
        let base = vec!["226".to_string()];
        let transition = ledger
            .apply(direct("姚明", "身高", "226", "记住"), &base)
            .unwrap();
        assert_eq!(transition.delta, SlotDelta::Added);
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["226"]);
        assert!(ledger.suppressed("姚明", "身高").is_empty());
    }

    #[test]
    fn same_value_from_new_source_is_corroboration() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("姚明", "身高", "226", "记住"), &[])
            .unwrap();
        let second = ledger
            .apply(direct("姚明", "身高", "226", "图谱"), &[])
            .unwrap();
        assert_eq!(second.delta, SlotDelta::Corroborated);
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["226"]);
    }

    #[test]
    fn undo_correction_restores_previous_value() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("姚明", "身高", "226", "记住"), &[])
            .unwrap();
        ledger
            .apply(correction("姚明", "身高", "229"), &[])
            .unwrap();
        let undone = ledger
            .undo_last_correction("姚明", "身高")
            .expect("有更正可撤销");
        assert!(matches!(undone.delta, SlotDelta::Restored { .. }));
        assert_eq!(ledger.accepted("姚明", "身高"), vec!["226"]);
        assert_eq!(ledger.suppressed("姚明", "身高"), vec!["229"]);
    }

    #[test]
    fn retract_removes_only_the_requested_value() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("张三", "职业", "教师", "记住"), &[])
            .unwrap();
        ledger
            .apply(direct("张三", "职业", "作家", "记住"), &[])
            .unwrap();
        let transitions = ledger.retract("张三", "职业", Some("教师"));
        assert_eq!(transitions.len(), 1);
        assert_eq!(ledger.accepted("张三", "职业"), vec!["作家"]);
    }

    #[test]
    fn classify_prefers_explicit_correction_over_hearsay() {
        let (strength, content) = AssertionStrength::classify("更正：姚明的身高是229");
        assert_eq!(strength, AssertionStrength::Correction);
        assert_eq!(content, "姚明的身高是229");
        let (strength, content) = AssertionStrength::classify("据说姚明的身高是230");
        assert_eq!(strength, AssertionStrength::Hearsay);
        assert_eq!(content, "姚明的身高是230");
        let (strength, _) = AssertionStrength::classify("姚明的身高是226");
        assert_eq!(strength, AssertionStrength::Direct);
    }

    #[test]
    fn history_states_current_winner_and_suppressed_not_negated() {
        let mut ledger = SlotLedger::new();
        ledger
            .apply(direct("姚明", "身高", "226", "记住"), &[])
            .unwrap();
        ledger
            .apply(correction("姚明", "身高", "229"), &[])
            .unwrap();
        let history = ledger.render_history("姚明", "身高");
        assert!(history.contains("当前采用：229"));
        assert!(history.contains("已压制"));
        assert!(history.contains("不等于否定"));
        assert!(history.contains("226"));
    }

    #[test]
    fn rebuild_from_facts_preserves_every_fact_verbatim() {
        let facts = vec![
            ("姚明".to_string(), "身高".to_string(), "226".to_string()),
            ("姚明".to_string(), "身高".to_string(), "229".to_string()),
        ];
        let ledger = SlotLedger::rebuild_from_facts(facts, SlotPolicy::with_defaults());
        let mut accepted = ledger.accepted("姚明", "身高");
        accepted.sort();
        assert_eq!(accepted, vec!["226", "229"]);
        assert_eq!(ledger.projection().len(), 2);
    }
}

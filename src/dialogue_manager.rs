//! 本体驱动的对话管理器：把「对话」本身当作一个本体来建模，
//! 用逻辑程序（Horn 规则 + 分层否定）推导对话策略，再由模板组合生成回答。
//!
//! 设计要点（纯符号主义，无神经网络、无权重训练）：
//! - 对话本体：每一轮是一个 `DialogueTurn` 实例，携带发言、意图（move）、
//!   焦点实体、属性、回答出处（provenance）与命中的策略；
//! - 策略程序：`POLICY_SOURCE` 用 Prolog 风格规则显式声明「什么状态下说什么话」，
//!   每轮把当前信息状态固化为 ABox 事实，交给 `LogicProgram` 做自底向上求值，
//!   推导出 `respond(...)` 策略集合，再由优先级选择落地；
//! - 组合生成：回答不是一次查表，而是由「结论 + 出处说明 + 追问建议」等
//!   discourse 片段按策略组装；支持多轮指代/省略、元话语（回顾、解释依据）、
//!   追问建议与主动续聊。
//!
//! 与 `SymbolicReasoner` 的关系：管理器不重复存储领域知识，通过
//! `DialogueKnowledge` 接口向推理器要「答案 / 实体 / 实体属性表」。

include!("dialogue_manager/dialogue_manager_part_01.rs");
include!("dialogue_manager/dialogue_manager_part_02.rs");
include!("dialogue_manager/dialogue_manager_part_03.rs");
include!("dialogue_manager/dialogue_manager_part_04.rs");

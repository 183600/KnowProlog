//! SFD：Shape-Filling Dialogue（形状约束驱动的对话与回答合成）。
//!
//! `shape_gate` 回答「数据是否合规、回答是否有据」，`relation_acquisition`
//! 回答「问什么、答什么、信什么」。本模块把两者接成一条对话策略：
//! 形状缺口 -> 议程 -> 一次追问 -> 用户回答 -> 会话覆盖层 -> 重新收敛，
//! 并在补全后回到用户最初的问题，给出带来源、缺失披露与冲突披露的画像。
//!
//! 约束：用户补充只进入 QRA 会话覆盖层，不写图谱；未确认的候选不参与回答。

include!("shape_dialogue/shape_dialogue_part_01.rs");
include!("shape_dialogue/shape_dialogue_part_02.rs");

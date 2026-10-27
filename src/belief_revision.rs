//! 推理提交协议（Inference Commit Protocol）：把一次知识变更变成可预览、可提交、
//! 可撤回的信念修订。
//!
//! 设计立场：
//!   - 不新造规则语言：规则仍然是 [`Clause`]，事实仍然是 [`Atom`]；
//!   - 不新造推理算法：前向闭包复用 [`ResolutionEngine::forward_chain`]，
//!     反向查询复用 [`ResolutionEngine::solve`]；
//!   - 提交前先做一致性闸门：功能谓词多值、定义域/值域互斥、规则不安全与
//!     「规则复活已撤回事实」都在 `plan` 阶段显式暴露；
//!   - 影响分析有回退：理由图能覆盖的变化直接报告；覆盖不了、推导被预算截断，
//!     就标记 `fallback_full_recompute`，绝不把不完整的影响说成完整；
//!   - 提交是原子的：`RevisionPlan` 持有候选引擎，`commit` 只在计划仍基于当前
//!     版本时替换状态；失败的计划不会留下半套事实或规则；
//!   - 回执可逆：回执保留原始操作，`undo_last` 用逆操作生成一次新的提交，
//!     不篡改历史。
//!
//! 边界：非单调片段（默认逻辑、ProbLog、显式否定）不在此承诺完备性；这里的
//! 影响集合只对正 Horn 片段与当前搜索预算负责。

include!("belief_revision/belief_revision_part_01.rs");
include!("belief_revision/belief_revision_part_02.rs");

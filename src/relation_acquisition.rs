//! QRA：Question-driven Relation Acquisition（问答式关系获取 / 以问代抽）。
//!
//! 目标：把「系统答不出」或「形状校验发现缺口」变成一个显式的槽位问题；
//! 用户回答先经过对话行为分类与形状闸门，再进入可审计的会话账本；
//! 只有被确认的候选才进入会话覆盖层，并可显式采纳到会话事实集。
//!
//! 与既有模块的分工：
//! - `predicate_registry` / `type_system` 提供谓词期望（定义域/值域/单值/互逆）；
//! - `shape_gate` 负责硬判定（实体未知/类型冲突/功能谓词冲突）与修复动作；
//! - 本模块负责问题生成、回答分类、账本状态机、覆盖层与模式诱导。
//!
//! 全部纯符号、零权重：账本、覆盖层与模式都按 BTree 排序，摘要由规范 JSON 得出。

include!("relation_acquisition/relation_acquisition_part_01.rs");
include!("relation_acquisition/relation_acquisition_part_02.rs");
include!("relation_acquisition/relation_acquisition_part_03.rs");

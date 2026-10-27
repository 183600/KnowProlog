//! 对话情景理论（Conversation Situation Theory, CST）。
//!
//! 编年史是事件流，本模块把事件流编译成一部**可查询的有限理论**：
//!
//!   s0 --e1--> s1 --e2--> s2 --e3--> ... --en--> sn
//!
//! 每个 `ei` 是一条编年史记录，每个 `si` 是事件之后的情景。情景、事件、
//! 问题、回答、取值都被具体化为常量，再交给 [`LogicProgram`] 查询。
//! 这样「第 N 轮相信什么」「哪条导入改变了回答」「如果撤销这次导入会怎样」
//! 都是同一套 Horn 目标下的不同查询。
//!
//! 设计边界（诚实优先）：
//!   - 只有效果可枚举的事件（文本导入回执、撤销、用户断言/撤回）参与
//!     反事实推断；规则教学、外链、JSON-LD/RDF 批量导入等显式弃权；
//!   - 信念折叠是「会话可枚举效果」的折叠，不是整库快照；它是情景理论
//!     的 ABox，不冒充 Oxigraph 的真值；
//!   - 归因只提名区间内触碰了问题实体/属性的事件，候选为空就明说
//!     「无法归因」，绝不用知识版本变化冒充原因；
//!   - 所有回答都从编年史投影，本模块不维护第二份可变对话状态。

include!("conversation_theory/conversation_theory_part_01.rs");
include!("conversation_theory/conversation_theory_part_02.rs");
include!("conversation_theory/conversation_theory_part_03.rs");

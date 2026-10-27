//! CRD：语境指称消歧（Contextual Reference Disambiguation）。
//!
//! 把消歧做成**语境约束求解**：
//! 歧义不是解析失败，而是一个显式的、可求解的约束问题；求解结果不是一条
//! 答案，而是一条**有作用域、有支持度、可撤销、可审计的绑定**。约束不足以
//! 裁决时，系统用图谱事实生成一个「能分裂候选集」的区分性问题（符号化的
//! 20 questions），而不是把候选名字罗列一遍。
//!
//! 数据流（M → C → E → D → Q → A → B → G）：
//! 1. **M 提及**：复用 `entity_linker::link_lattice` 与全部候选，不重写打分；
//! 2. **C 语境素描**：为每个指称抽中文特征（量词、同位角色、关系名词、
//!    疑问词期待），为候选实体建 `SenseCard`（类型、出/入谓词、邻居锚点、度数）；
//! 3. **E 证据表决**：硬约束（类型不相容、人名护栏）先否决，其余候选进入
//!    整数证据表决（精确 +4、已确认别名 +3、类型相容 +2、谓词相容 +2……
//!    f64 分数只经 `score_bucket` 量化为整数）；
//! 4. **D 裁决**：五选一——`Bound`（命中活跃绑定）/ `Unique`（存活候选唯一且
//!    领先次优 ≥ 边际）/ `Ambiguous`（需澄清）/ `Description`（描述性指称）/
//!    `Nil`（零候选）。歧义是一等状态，而不是被静默压掉；
//! 5. **Q 追问**：从义项卡找「把候选两两分开」的特征生成问题（类型标签 /
//!    功能谓词的不同取值 / 邻居锚点），选项 ≤3，可答「不知道/别问了」；
//! 6. **A 回答行为**：`Choose / Restate / Correct / Unknown / Decline`；
//! 7. **B 绑定账本**：`ReferenceBinding{key, entity, scope, status, support,
//!    turn, evidence}`，作用域 `Turn/Topic/Session`，随快照持久化；
//! 8. **G 生成与学习**：回答先查绑定再查图谱；同一绑定第二次被独立确认 →
//!    提名为待确认别名（不自动确认、不写图谱）。
//!
//! 设计边界：本模块**只读**实体链接器的候选与身份台账的折叠函数，不新建
//! 词表、不重写打分、不写知识图谱。零权重、零训练。

include!("contextual_reference/contextual_reference_part_01.rs");
include!("contextual_reference/contextual_reference_part_02.rs");

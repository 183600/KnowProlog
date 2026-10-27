//! FRC：帧规则编译（Frame Rule Compilation）。
//!
//! 在关系帧与三元组之间补一层谓词中心的语义角色表示：
//! - [`Role`] 定义 16 个核心角色，格标记负责把中文旁格映射到角色；
//! - [`FrameSchema`] 把谓词义项、角色清单、选择限制与双向模板放在一起；
//! - [`FrameAnnotator`] 从显式格标记与词序标注角色，未知谓词也能用构式规则；
//! - [`FrameInstance::project`] 把核心角色投影回 `RelationFrame`，旁格进限定语；
//! - [`FrameLedger`] 支持角色聚焦、省略继承与显式更正；
//! - [`FrameRuleCompiler`] 从标注正例反合一出 schema，并用反例与回译做硬校验。
//!
//! 全部纯符号、零训练；未登记 schema 时标注器仍按通用构式工作。

include!("frame_semantics/frame_semantics_part_01.rs");
include!("frame_semantics/frame_semantics_part_02.rs");

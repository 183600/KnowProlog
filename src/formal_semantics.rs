//! 形式语义学（Montague 风格）：把中文句子组合式地翻译为带类型的逻辑形式，
//! 再在由知识图谱事实构成的「微型模型」上求值，并支持从逻辑形式反向生成中文答句。
//!
//! 纯符号主义实现，不引入任何神经网络或权重训练：
//! - 组合原则：函数应用（前向/后向）、广义量词（所有/有些）、话题-述题「是」规则、
//!   合取/析取/否定/蕴含，全部按类型驱动；
//! - 模型论语义：谓词外延 = 图谱中满足该谓词的实体集合；类型断言沿子类闭包传播；
//! - 生成式回答：从求解结果组合中文答句（谓词表面形来自 SymbolicGenerator），
//!   检索到的证据以引用形式附加，保证可追溯。

include!("formal_semantics/formal_semantics_part_01.rs");
include!("formal_semantics/formal_semantics_part_02.rs");
include!("formal_semantics/formal_semantics_part_03.rs");

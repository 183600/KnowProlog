//! 语境分词：符号子词嵌入 + 字符格解码 + 读法记忆。
//!
//! 与 `causal_tokenization` 的因果锚定不同，这里把「对话语境」送回分词层：
//! 每个候选片段都带一个可命名的稀疏向量（类型/谓词/邻接特征），由当前
//! 焦点、期望类型与谓词组成语境向量，再在字符格上做 k-best Viterbi。
//! 未登录片段只在命中已知子词证据时才允许组合，且所有分数都能拆回
//! 「词表先验 / 语境余弦 / 模式相容 / 读法记忆 / 子词组合」五项。
//! 歧义边际只落在内容词时生成澄清问句；确认与纠正写入 Beta 读法记忆。

include!("contextual_segmentation/contextual_segmentation_part_01.rs");
include!("contextual_segmentation/contextual_segmentation_part_02.rs");

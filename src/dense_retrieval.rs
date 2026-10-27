//! 符号主义「稠密检索」：把每条知识当作一篇文档，用纯统计与图游走信号编码成
//! 稀疏高维向量（TF-IDF 词面 × 结构 PPR × 谓词匹配），查询与文档做加权内积。
//!
//! 与神经网络检索器不同，这里没有任何可学习参数：
//! - 词面向量：TF-IDF 倒排索引（df 来自语料统计，无训练）；
//! - 结构向量：在实体-实体图上做随机游走重启（Personalized PageRank）得到
//!   与查询实体结构相关的分布，作为「稠密」语义信号；
//! - 谓词匹配：查询词与谓词的归一化匹配作为精确信号。
//! 三种信号线性融合，输出可解释的证据排序，供生成式回答接地（RAG 的符号版）。
//!
//! [`DenseRetriever::search`] 额外提供精确约束（主体/谓词/客体/类型）与
//! 带证书的 top-k：倒排索引中的候选先按保守上界排序，上界低于当前第 k 名
//! 的文档整段跳过；扫描预算耗尽或候选生成被截断时，[`SearchReceipt`] 会
//! 明确标记 `certified_top_k = false`，保证「省略」始终可审计。

include!("dense_retrieval/dense_retrieval_part_01.rs");
include!("dense_retrieval/dense_retrieval_part_02.rs");
include!("dense_retrieval/dense_retrieval_part_03.rs");

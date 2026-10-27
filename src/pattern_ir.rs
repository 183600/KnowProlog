//! 统一模式中间表示（Pattern IR）与模式库（Pattern Library）。
//!
//! 设计立场：**一次命中 = 一组带绑定、带区间、可排序、可解释的解**，
//! 而不是又一个只返回布尔的正则。本模块把此前散落在三处的模式语言
//! （`config.toml [[prolog_rules]]` 的词性/词面序列、`nlu::IntentConfig`
//! 的 `{槽位}` 模板、教学/诱导的运行时模式）收敛到一套 IR 上：
//!
//!   - [`Elem`]：最小要素（词性 / 词面 / 带排序的槽位 / 任意 / 可选）；
//!   - [`PatternEntry`]：一条有稳定指纹（FNV-1a）的模式——来源、路由、优先级
//!     与 IR 一起构成身份，可落成 RDF（见 [`pattern_quads`]）；
//!   - [`PatternLibrary`]：编译、排序、命中、体检（重复/遮蔽/冲突）与教学；
//!   - [`MatchReport`]：一次匹配的全部候选——胜者、命中未胜、失败原因，
//!     即「为什么是它 / 为什么不是别的」的机器可读底稿。
//!
//! 匹配语义与既有 [`crate::nlp`] 的 `match_pattern_for_prolog` 保持一致
//! （锚定开头、标点跳过、人名合并、词面直配/逐字跨词/包含），保证查询路由
//! 换到本引擎后行为不变；新增的槽位排序、区间与落选原因是增量能力。
//!
//! 排序规则与既有 `analyze_query_intent` 一致：**优先级降序 + 配置序**
//! （稳定排序），特化度只用于展示与体检，不改变路由结果。

include!("pattern_ir/pattern_ir_part_01.rs");
include!("pattern_ir/pattern_ir_part_02.rs");
include!("pattern_ir/pattern_ir_part_03.rs");

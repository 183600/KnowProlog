//! REL：语义发布（Semantic Release）——把语言资产编译成可差分、可回归、
//! 可回滚的发布单元。
//!
//! 符号系统没有梯度，但有一组显式资产：实体类型、属性、抽取规则、Prolog
//! 规则、谓词画像、类型格、同义词、提示模板与插件贡献。本模块把这些资产
//! 投影成 [`SchemaSnapshot`]（单一正本由配置派生，不手写），与上一版做
//! [`SchemaDiff`]，按 breaking / compatible / behavioral 分类，再用 semver
//! 规则约束版本号。
//!
//! [`check`] 是发布总闸门：config lint → prompt lint → plugin eval → schema
//! diff → NLU 评测 → 恢复评测 → 身份评测 → 质量差分模糊测试。所有报告只含确定性数据
//! （BTreeMap 排序、无时间戳），同一资产必然得到同一摘要。

include!("release/release_part_01.rs");
include!("release/release_part_02.rs");

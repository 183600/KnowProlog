//! 声明式插件（符号专家）机制。
//!
//! 每个插件是一份 `manifest.toml`：身份（id/version/api）、依赖与冲突、
//! 以及若干**带类型的贡献**。宿主只认识固定的贡献类型与固定阶段，
//! 不加载动态库、不执行插件代码；插件只能声明「要什么」，不能自行行动。
//!
//! 本模块实现七个阶段中的前五个可执行阶段：
//! `Lexicon → Reference → Frame → Retrieve → Compose`（Reference 目前仅校验与登记），
//! 以及贯穿全程的 `Audit`（清单 lint、启用裁决、逐轮 trace、包内评测）。
//!
//! 运行模式与项目既有认识论一致：
//! - `off`：完全不加载贡献；
//! - `observe`：只计算「如果启用会怎样」并写入 trace，不影响回答；
//! - `on`：真正参与改写 / 配方 / 模板 / 风格。
//!
//! 安全边界：改写与模板都是受限的槽位语法（`{entity}` 等），不是正则、
//! 不是脚本；模板必须所有槽位都接地才渲染，否则回退原文，绝不凭空补值。

include!("plugin/plugin_part_01.rs");
include!("plugin/plugin_part_02.rs");
include!("plugin/plugin_part_03.rs");

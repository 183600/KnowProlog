//! 配置编译的绑定帧（Config-Compiled Binding Frames）。
//!
//! 把「一句话能绑定哪些变量、变量如何跨轮继承、缺槽时问什么、绑定完成后
//! 查什么、怎么措辞」从 Rust 硬编码搬到 config.toml，并在启动时编译成
//! 不可变的 [`FrameProgram`]。Rust 只负责通用的匹配、合一、执行与生成，
//! 不再认识「身高」「首都」「师傅」这些具体词。
//!
//! 语义约定：
//! - 模式里的 `{slot}` 是带排序的变量，字面量是锚点；
//! - 一次回答是槽位全部接地的绑定，一次追问是部分绑定；
//! - 澄清把自由变量显式还给用户，用户短答后完成一次绑定；
//! - 答案模板只能渲染已声明且已接地的槽位，生成不会凭空造值；
//! - 任何一步失败都零状态变化，调用方回退常规管线。

include!("binding_frames/binding_frames_part_01.rs");
include!("binding_frames/binding_frames_part_02.rs");

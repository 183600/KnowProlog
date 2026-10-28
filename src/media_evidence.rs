//! MDX：有界多模态证据层（Bounded Multimodal Evidence）。
//!
//! 本模块把「用户给了一张图 / 一张表 / 一段录音」变成可审计的符号证据，
//! 而不是把二进制塞进 RDF 或会话快照：
//!
//! - [`AssetRecord`]：内容寻址、发布后不可变的媒体资产；字节只存放在
//!   [`MediaStore`] 的受控内存与可选媒体根目录中；
//! - [`Anchor`]：资产上的符号锚点（图区域 / 表格行列 / 音频区间 / 文本区间），
//!   带抽取器、置信度、实体链接与定位符，可精确引用到「哪一块」；
//! - [`MediaSelector`]：把「这张图」「表格第三行第二列」「刚才那段录音最后
//!   一句」解析成资产选择 + 定位提示；歧义时返回编号候选，绝不瞎猜；
//! - [`SupportState`]：图谱文本与媒体观察的五态证据格——仅图谱 / 仅媒体 /
//!   相互印证 / 相互冲突 / 没有证据；
//! - [`SessionMedia`]：会话媒体台账，只保存逻辑引用（ID + 哈希 + 世代），
//!   容量硬上限、快照不落字节；资产删除或重发后旧引用稳定变为不可用。
//!
//! 纪律：所有资源无界增长都被类型化错误挡住；内容指纹用 FNV-1a 仅作完整性
//! 自检（与 `context_image` 一致，非密码学安全）；原始媒体**永不**进入推理器，
//! 媒体观察**永不**自动升格为图谱事实。

mod anchor;
mod asset;
mod selection;
mod selector;
mod session;
mod store;
mod support;
mod table;
mod turn;

#[cfg(test)]
mod tests;

pub use anchor::*;
pub use asset::*;
pub use selection::*;
pub use selector::*;
pub use session::*;
pub use store::*;
pub use support::*;
pub use table::*;
pub use turn::*;

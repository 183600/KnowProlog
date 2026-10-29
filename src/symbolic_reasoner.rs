//! 符号推理器：解析、合一、推理、渲染等分片实现（由 include! 组合为同一模块）。

include!("symbolic_reasoner/top_01.rs");
include!("symbolic_reasoner/top_02.rs");
include!("symbolic_reasoner/top2_01.rs");
include!("symbolic_reasoner/reasoner_01.rs");
include!("symbolic_reasoner/batch_01.rs");
include!("symbolic_reasoner/reasoner_02.rs");
include!("symbolic_reasoner/reasoner_03.rs");
include!("symbolic_reasoner/reasoner_04.rs");
include!("symbolic_reasoner/reasoner_05.rs");
include!("symbolic_reasoner/reasoner_06.rs");
include!("symbolic_reasoner/reasoner_07.rs");
include!("symbolic_reasoner/reasoner_08.rs");
include!("symbolic_reasoner/reasoner_09.rs");
include!("symbolic_reasoner/reasoner_10.rs");
include!("symbolic_reasoner/reasoner_11.rs");

#[cfg(test)]
mod tests {
    include!("symbolic_reasoner/tests_01.rs");
    include!("symbolic_reasoner/tests_02.rs");
    include!("symbolic_reasoner/tests_03.rs");
    include!("symbolic_reasoner/tests_04.rs");
    include!("symbolic_reasoner/tests_05.rs");
    include!("symbolic_reasoner/tests_06.rs");
    include!("symbolic_reasoner/tests_07.rs");
    include!("symbolic_reasoner/tests_core_paths.rs");
    include!("symbolic_reasoner/tests_batch.rs");
}

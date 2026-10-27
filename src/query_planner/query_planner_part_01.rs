// 逻辑查询计划编译器（Query Plan Compiler）
//
// 符号主义的"语义理解"：不把问题映射到稠密向量，而是把中文问句编译成
// 显式的逻辑查询计划（可打印、可检查、可执行）：
//
//   张三的师傅的师傅是谁   → FindObject { segments: [张三, 师傅, 师傅] }
//   李四的师傅是王五吗     → CheckTriple { subject_segments: [李四, 师傅], object: 王五 }
//   李四有几个师傅         → Count { subject: 李四, predicate: 师傅 }
//   李四有师傅吗           → Exist { subject: 李四, predicate: 师傅 }
//   谁的师傅是李四         → FindSubject { object: 李四, predicate: 师傅 }
//   谁和张三最像           → Similar { entity: 张三 }
//   张三对应李四，王五对应谁 → Analogy { a: 张三, b: 李四, c: 王五 }
//
// 计划只做句法/结构的模式枚举，不接触知识库；实体/谓词的归一化与执行
// 由 SymbolicReasoner 结合本体对齐与链接预测完成。
use crate::pattern_closure::ClosureOp;
use crate::symbolic_generator::predicate_labels;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryPlan {
    // 沿"的"链查找对象：张三 的 师傅 的 师傅 是谁
    FindObject {
        segments: Vec<String>,
    },
    // 反向查找主体：谁的 师傅 是 李四
    FindSubject {
        object: String,
        predicate: String,
    },
    // 正向关系查找：张三 认识 谁 / 谁被张三认识
    FindObjectByPredicate {
        subject: String,
        predicate: String,
    },
    // 关系查找 + 集合差：除了李四，张三还认识谁
    FindObjectExcept {
        subject: String,
        predicate: String,
        excluded: Vec<String>,
    },
    // 反查主体 + 集合差：除了张三，还有谁认识李四
    FindSubjectExcept {
        object: String,
        predicate: String,
        excluded: Vec<String>,
    },
    // 两点之间的路径：张三和王五之间是什么关系
    PathBetween {
        start: String,
        goal: String,
        max_hops: usize,
    },
    // 共同邻居：张三和李四共同认识谁
    CommonNeighbor {
        left: String,
        right: String,
    },
    // 路径上的中间人：张三通过谁认识王五
    Intermediary {
        start: String,
        goal: String,
        max_hops: usize,
    },
    // 三值校验：主语链最后一个词是谓词，如 [张三, 师傅] + object=李四
    CheckTriple {
        subject_segments: Vec<String>,
        object: String,
    },
    // 直接关系校验：张三 认识 李四 吗 / 张三不认识李四吗
    CheckRelation {
        subject: String,
        predicate: String,
        object: String,
        negated: bool,
    },
    // 计数：李四 有几个 师傅
    Count {
        subject: String,
        predicate: String,
    },
    // 存在性：李四 有没有 师傅
    Exist {
        subject: String,
        predicate: String,
    },
    // 闭包：李四 的 所有 上司（Plus 需要已启用的传递律）
    Closure {
        subject: String,
        predicate: String,
        operator: ClosureOp,
        count_only: bool,
        then: Vec<String>,
    },
    // 最近邻语义检索：谁和 张三 最像
    Similar {
        entity: String,
    },
    // 向量算术类比：a:b :: c:?
    Analogy {
        a: String,
        b: String,
        c: String,
    },
}
#[derive(Clone)]
pub struct QueryPlanner {
    max_question_chars: usize,
    max_chain: usize,
    max_path_hops: usize,
}
impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

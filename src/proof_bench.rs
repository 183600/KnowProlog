//! 可证基准 ProofBench：把评测从「像不像参考句子」推进到「有没有证明义务」。
//!
//! 与已有的 `dialogue_eval` / `shape_eval` / `link_eval` 的关系：
//! - 那些套件各自度量一个领域（意图、槽位、形状、链接），本模块度量**跨家族**
//!   的「问句 → 结构化解读 → 归结证明」闭环；
//! - 打分不看回答文本，而看归结解里的变量绑定与证明树里的基原子。措辞变化
//!   不影响得分；实体/属性/值错配或模板编造出的无根据 claim 无法通过；
//! - 每条期望都是一个 ground obligation（如 `身高(叶莉, 190)`），必须在系统
//!   本次搜索实际返回的证明树中出现；桥接事实（如 `配偶(姚明, 叶莉)`）同样
//!   要出现，防止「答案碰巧对、依据不对」。
//!
//! 三个新增指标：
//! - **证距（proof distance）**：不可答探针用有界溯因估计「最少补几条事实才能
//!   证出来」。报告显式标注 `bounded`，没找到场景不等于不存在；
//! - **反事实翻转率（twin flip）**：同一脚本替换一条支撑事实，正确答案必须翻转。
//!   只会抓显著实体或背模板的系统翻不动；
//! - **干扰项不变率（distractor stability）**：加入相似实体后答案不得改变。
//!
//! 认识论纪律：搜索被截断（`Incomplete`）时记 `Unchecked`，绝不把「没搜完」
//! 判成「答错」，也不把「没找到」说成「不存在」。
//!
//! 基准本身可审计：[`render_prolog`] 把语料、义务与孪生导出为纯 Prolog 事实，
//! 第三方可以 `consult` 后独立复现。

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::abduction::AbductionEngine;
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager};
use crate::prolog_engine::PrologFact;
use crate::resolution_engine::{Atom, ResolutionEngine, SearchStatus, Term};
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::view_render::ProofViewNode;
use crate::ReasoningConfig;

/// 基准家族：每个家族对应一类「像 LLM」的符号机制。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// 指代 / 省略：焦点栈把「他」「她的」「国籍呢」补全。
    Coref,
    /// 多跳：关系链 + 属性，必须给出桥接事实与最终事实。
    Multihop,
    /// 不可答：知识库确实没有，系统不得凭空证明。
    Unknown,
    /// 长程记忆：插入干扰轮后答案保持稳定，或焦点仍指向最近实体。
    Memory,
}

impl Family {
    pub fn label(self) -> &'static str {
        match self {
            Family::Coref => "coref",
            Family::Multihop => "multihop",
            Family::Unknown => "unknown",
            Family::Memory => "memory",
        }
    }

    pub const ALL: [Family; 4] = [
        Family::Coref,
        Family::Multihop,
        Family::Unknown,
        Family::Memory,
    ];
}

/// 一条知识事实。基准不使用 RDF 存储，直接编译成 `PrologFact` 与归结事实。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fact {
    Attribute {
        entity: &'static str,
        key: &'static str,
        value: &'static str,
    },
    Relation {
        from: &'static str,
        relation: &'static str,
        to: &'static str,
    },
}

/// 证明义务：本次证明树中必须出现的基原子。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Obligation {
    pub predicate: &'static str,
    pub subject: &'static str,
    pub object: &'static str,
}

impl Obligation {
    /// 与 [`crate::resolution_engine::Atom::display`] 完全一致，
    /// 这样证明树里的 `canonical` 可以直接字符串比较（不是自然语言匹配）。
    pub fn canonical(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.object)
    }
}

/// 溯因目标里的一个项：常量或变量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermSpec {
    Entity(&'static str),
    Var(&'static str),
}

impl TermSpec {
    fn term(self) -> Term {
        match self {
            TermSpec::Entity(name) => Term::constant(name),
            TermSpec::Var(name) => Term::var(name),
        }
    }

    fn name(self) -> &'static str {
        match self {
            TermSpec::Entity(name) | TermSpec::Var(name) => name,
        }
    }
}

/// 失败目标的原子模板（供证距估计）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goal {
    pub predicate: &'static str,
    pub subject: TermSpec,
    pub object: TermSpec,
}

impl Goal {
    fn atom(&self) -> Atom {
        Atom::pattern(self.predicate, self.subject.term(), self.object.term())
    }
}

/// 一个探针的期望。
#[derive(Debug, Clone)]
pub enum Probe {
    /// 可答：系统必须读对实体/属性，并在证明树中给出全部义务。
    Answer {
        entity: &'static str,
        /// 系统 `UtteranceFrame.attribute` 的期望词面（多跳时是「配偶的身高」）。
        attribute: &'static str,
        value: &'static str,
        obligations: &'static [Obligation],
    },
    /// 不可答：期望搜索结果为空且搜索已穷尽；证距由 [`Goal`] 估计。
    Unknown {
        entity: &'static str,
        attribute: &'static str,
        goals: &'static [Goal],
    },
    /// 未知实体：系统不得把它当成已知实体，也不得给出确定回答。
    UnknownEntity { entity: &'static str },
}

/// 一轮脚本。只有探针轮带 `probe`，其余轮负责建立焦点/干扰。
#[derive(Debug, Clone)]
pub struct BenchTurn {
    pub input: &'static str,
    pub probe: Option<Probe>,
}

impl BenchTurn {
    const fn context(input: &'static str) -> Self {
        Self { input, probe: None }
    }

    const fn probe(input: &'static str, probe: Probe) -> Self {
        Self {
            input,
            probe: Some(probe),
        }
    }
}

/// 孪生/干扰类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerturbationKind {
    /// 反事实：替换支撑事实后答案必须翻转。
    Flip,
    /// 干扰：加入相似事实后答案必须不变。
    Distractor,
}

impl PerturbationKind {
    pub fn label(self) -> &'static str {
        match self {
            PerturbationKind::Flip => "flip",
            PerturbationKind::Distractor => "distractor",
        }
    }
}

/// 一个可复现的扰动。
#[derive(Debug, Clone)]
pub struct Perturbation {
    pub label: &'static str,
    pub kind: PerturbationKind,
    /// 探针所在的轮（必须与场景中的探针轮一致）。
    pub turn: usize,
    pub remove: &'static [Fact],
    pub add: &'static [Fact],
    pub expected_before: &'static str,
    pub expected_after: &'static str,
}

/// 一个基准场景：小知识库 + 多轮脚本 + 一个探针 + 一个扰动。
#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: &'static str,
    pub family: Family,
    pub facts: &'static [Fact],
    pub turns: &'static [BenchTurn],
    pub perturbation: Perturbation,
}

impl Scenario {
    /// 探针所在轮（语料不变量：恰好一个）。
    pub fn probe_index(&self) -> usize {
        self.turns
            .iter()
            .position(|turn| turn.probe.is_some())
            .expect("每个场景必须恰好有一个探针轮")
    }
}

// ---------------------------------------------------------------------------
// 语料
// ---------------------------------------------------------------------------

/// 12 个场景 / 4 个家族 / 12 个孪生或干扰。
pub fn seed_corpus() -> &'static [Scenario] {
    static SEED: &[Scenario] = &[
        // ---------------- coref ----------------
        Scenario {
            name: "篮球家庭",
            family: Family::Coref,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "出生地",
                    value: "上海",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
                Fact::Relation {
                    from: "叶莉",
                    relation: "配偶",
                    to: "姚明",
                },
            ],
            turns: &[
                BenchTurn::context("姚明的身高是多少"),
                BenchTurn::probe(
                    "他的国籍是什么",
                    Probe::Answer {
                        entity: "姚明",
                        attribute: "国籍",
                        value: "中国",
                        obligations: &[Obligation {
                            predicate: "国籍",
                            subject: "姚明",
                            object: "中国",
                        }],
                    },
                ),
            ],
            perturbation: Perturbation {
                label: "姚明的国籍改为美国",
                kind: PerturbationKind::Flip,
                turn: 1,
                remove: &[Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                }],
                add: &[Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "美国",
                }],
                expected_before: "中国",
                expected_after: "美国",
            },
        },
        Scenario {
            name: "省略追问",
            family: Family::Coref,
            facts: &[
                Fact::Attribute {
                    entity: "张三",
                    key: "身高",
                    value: "180",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "出生地",
                    value: "北京",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "身高",
                    value: "175",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "国籍",
                    value: "中国",
                },
            ],
            turns: &[
                BenchTurn::context("张三的身高是多少"),
                BenchTurn::probe(
                    "国籍呢",
                    Probe::Answer {
                        entity: "张三",
                        attribute: "国籍",
                        value: "中国",
                        obligations: &[Obligation {
                            predicate: "国籍",
                            subject: "张三",
                            object: "中国",
                        }],
                    },
                ),
                BenchTurn::context("出生地呢"),
            ],
            perturbation: Perturbation {
                label: "张三的国籍改为新加坡",
                kind: PerturbationKind::Flip,
                turn: 1,
                remove: &[Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "中国",
                }],
                add: &[Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "新加坡",
                }],
                expected_before: "中国",
                expected_after: "新加坡",
            },
        },
        Scenario {
            name: "介绍后指代",
            family: Family::Coref,
            facts: &[
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
            ],
            turns: &[
                BenchTurn::context("叶莉是谁"),
                BenchTurn::probe(
                    "她的身高是多少",
                    Probe::Answer {
                        entity: "叶莉",
                        attribute: "身高",
                        value: "190",
                        obligations: &[Obligation {
                            predicate: "身高",
                            subject: "叶莉",
                            object: "190",
                        }],
                    },
                ),
            ],
            perturbation: Perturbation {
                label: "叶莉的身高改为188",
                kind: PerturbationKind::Flip,
                turn: 1,
                remove: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                }],
                add: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "188",
                }],
                expected_before: "190",
                expected_after: "188",
            },
        },
        // ---------------- multihop ----------------
        Scenario {
            name: "配偶身高",
            family: Family::Multihop,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
                Fact::Relation {
                    from: "叶莉",
                    relation: "配偶",
                    to: "姚明",
                },
            ],
            turns: &[BenchTurn::probe(
                "姚明的配偶的身高是多少",
                Probe::Answer {
                    entity: "姚明",
                    attribute: "配偶的身高",
                    value: "190",
                    obligations: &[
                        Obligation {
                            predicate: "配偶",
                            subject: "姚明",
                            object: "叶莉",
                        },
                        Obligation {
                            predicate: "身高",
                            subject: "叶莉",
                            object: "190",
                        },
                    ],
                },
            )],
            perturbation: Perturbation {
                label: "叶莉的身高改为188",
                kind: PerturbationKind::Flip,
                turn: 0,
                remove: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                }],
                add: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "188",
                }],
                expected_before: "190",
                expected_after: "188",
            },
        },
        Scenario {
            name: "配偶国籍",
            family: Family::Multihop,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
                Fact::Relation {
                    from: "叶莉",
                    relation: "配偶",
                    to: "姚明",
                },
            ],
            turns: &[BenchTurn::probe(
                "姚明的配偶的国籍是什么",
                Probe::Answer {
                    entity: "姚明",
                    attribute: "配偶的国籍",
                    value: "中国",
                    obligations: &[
                        Obligation {
                            predicate: "配偶",
                            subject: "姚明",
                            object: "叶莉",
                        },
                        Obligation {
                            predicate: "国籍",
                            subject: "叶莉",
                            object: "中国",
                        },
                    ],
                },
            )],
            perturbation: Perturbation {
                label: "叶莉的国籍改为加拿大",
                kind: PerturbationKind::Flip,
                turn: 0,
                remove: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                }],
                add: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "加拿大",
                }],
                expected_before: "中国",
                expected_after: "加拿大",
            },
        },
        Scenario {
            name: "三跳师门",
            family: Family::Multihop,
            facts: &[
                Fact::Attribute {
                    entity: "张三",
                    key: "身高",
                    value: "180",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "身高",
                    value: "175",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "身高",
                    value: "170",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Relation {
                    from: "张三",
                    relation: "配偶",
                    to: "李四",
                },
                Fact::Relation {
                    from: "李四",
                    relation: "师傅",
                    to: "王五",
                },
            ],
            turns: &[BenchTurn::probe(
                "张三的配偶的师傅的身高是多少",
                Probe::Answer {
                    entity: "张三",
                    attribute: "配偶的师傅的身高",
                    value: "170",
                    obligations: &[
                        Obligation {
                            predicate: "配偶",
                            subject: "张三",
                            object: "李四",
                        },
                        Obligation {
                            predicate: "师傅",
                            subject: "李四",
                            object: "王五",
                        },
                        Obligation {
                            predicate: "身高",
                            subject: "王五",
                            object: "170",
                        },
                    ],
                },
            )],
            perturbation: Perturbation {
                label: "王五的身高改为168",
                kind: PerturbationKind::Flip,
                turn: 0,
                remove: &[Fact::Attribute {
                    entity: "王五",
                    key: "身高",
                    value: "170",
                }],
                add: &[Fact::Attribute {
                    entity: "王五",
                    key: "身高",
                    value: "168",
                }],
                expected_before: "170",
                expected_after: "168",
            },
        },
        // ---------------- unknown ----------------
        Scenario {
            name: "未知属性",
            family: Family::Unknown,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "出生地",
                    value: "上海",
                },
                Fact::Attribute {
                    entity: "李娜",
                    key: "体重",
                    value: "55",
                },
            ],
            turns: &[BenchTurn::probe(
                "姚明的体重是多少",
                Probe::Unknown {
                    entity: "姚明",
                    attribute: "体重",
                    goals: &[Goal {
                        predicate: "体重",
                        subject: TermSpec::Entity("姚明"),
                        object: TermSpec::Var("答案"),
                    }],
                },
            )],
            perturbation: Perturbation {
                label: "加入李娜的体重作干扰",
                kind: PerturbationKind::Distractor,
                turn: 0,
                remove: &[],
                add: &[Fact::Attribute {
                    entity: "李娜",
                    key: "体重",
                    value: "58",
                }],
                expected_before: "unknown",
                expected_after: "unknown",
            },
        },
        Scenario {
            name: "未知桥接",
            family: Family::Unknown,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "李娜",
                    key: "体重",
                    value: "55",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
                Fact::Relation {
                    from: "叶莉",
                    relation: "配偶",
                    to: "姚明",
                },
            ],
            turns: &[BenchTurn::probe(
                "姚明的配偶的体重是多少",
                Probe::Unknown {
                    entity: "姚明",
                    attribute: "配偶的体重",
                    goals: &[
                        Goal {
                            predicate: "配偶",
                            subject: TermSpec::Entity("姚明"),
                            object: TermSpec::Var("V1"),
                        },
                        Goal {
                            predicate: "体重",
                            subject: TermSpec::Var("V1"),
                            object: TermSpec::Var("答案"),
                        },
                    ],
                },
            )],
            perturbation: Perturbation {
                label: "加入赵六的体重作干扰",
                kind: PerturbationKind::Distractor,
                turn: 0,
                remove: &[],
                add: &[Fact::Attribute {
                    entity: "赵六",
                    key: "体重",
                    value: "70",
                }],
                expected_before: "unknown",
                expected_after: "unknown",
            },
        },
        Scenario {
            name: "未知实体",
            family: Family::Unknown,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
            ],
            turns: &[BenchTurn::probe(
                "孙悟空的身高是多少",
                Probe::UnknownEntity {
                    entity: "孙悟空"
                },
            )],
            perturbation: Perturbation {
                label: "加入李娜的身高作干扰",
                kind: PerturbationKind::Distractor,
                turn: 0,
                remove: &[],
                add: &[Fact::Attribute {
                    entity: "李娜",
                    key: "身高",
                    value: "165",
                }],
                expected_before: "unknown",
                expected_after: "unknown",
            },
        },
        // ---------------- memory ----------------
        Scenario {
            name: "长上下文稳定",
            family: Family::Memory,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "出生地",
                    value: "上海",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "身高",
                    value: "180",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "出生地",
                    value: "北京",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "身高",
                    value: "175",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "出生地",
                    value: "广州",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "身高",
                    value: "170",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "出生地",
                    value: "成都",
                },
            ],
            turns: &[
                BenchTurn::context("姚明的出生地是哪里"),
                BenchTurn::context("叶莉的身高是多少"),
                BenchTurn::context("张三的国籍是什么"),
                BenchTurn::context("李四的出生地是哪里"),
                BenchTurn::context("王五的身高是多少"),
                BenchTurn::context("叶莉的国籍是什么"),
                BenchTurn::probe(
                    "姚明的出生地是哪里",
                    Probe::Answer {
                        entity: "姚明",
                        attribute: "出生地",
                        value: "上海",
                        obligations: &[Obligation {
                            predicate: "出生地",
                            subject: "姚明",
                            object: "上海",
                        }],
                    },
                ),
            ],
            perturbation: Perturbation {
                label: "加入李娜的出生地作干扰",
                kind: PerturbationKind::Distractor,
                turn: 6,
                remove: &[],
                add: &[Fact::Attribute {
                    entity: "李娜",
                    key: "出生地",
                    value: "武汉",
                }],
                expected_before: "上海",
                expected_after: "上海",
            },
        },
        Scenario {
            name: "焦点切换",
            family: Family::Memory,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Relation {
                    from: "姚明",
                    relation: "配偶",
                    to: "叶莉",
                },
                Fact::Relation {
                    from: "叶莉",
                    relation: "配偶",
                    to: "姚明",
                },
            ],
            turns: &[
                BenchTurn::context("姚明的身高是多少"),
                BenchTurn::context("叶莉的身高是多少"),
                BenchTurn::probe(
                    "她的国籍是什么",
                    Probe::Answer {
                        entity: "叶莉",
                        attribute: "国籍",
                        value: "中国",
                        obligations: &[Obligation {
                            predicate: "国籍",
                            subject: "叶莉",
                            object: "中国",
                        }],
                    },
                ),
            ],
            perturbation: Perturbation {
                label: "叶莉的国籍改为法国",
                kind: PerturbationKind::Flip,
                turn: 2,
                remove: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                }],
                add: &[Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "法国",
                }],
                expected_before: "中国",
                expected_after: "法国",
            },
        },
        Scenario {
            name: "长期记忆最近焦点",
            family: Family::Memory,
            facts: &[
                Fact::Attribute {
                    entity: "姚明",
                    key: "身高",
                    value: "226",
                },
                Fact::Attribute {
                    entity: "姚明",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "身高",
                    value: "190",
                },
                Fact::Attribute {
                    entity: "叶莉",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "身高",
                    value: "180",
                },
                Fact::Attribute {
                    entity: "张三",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "身高",
                    value: "175",
                },
                Fact::Attribute {
                    entity: "李四",
                    key: "国籍",
                    value: "中国",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "身高",
                    value: "170",
                },
                Fact::Attribute {
                    entity: "王五",
                    key: "国籍",
                    value: "中国",
                },
            ],
            turns: &[
                BenchTurn::context("姚明的身高是多少"),
                BenchTurn::context("叶莉的身高是多少"),
                BenchTurn::context("张三的身高是多少"),
                BenchTurn::context("李四的身高是多少"),
                BenchTurn::context("王五的身高是多少"),
                BenchTurn::probe(
                    "他的国籍是什么",
                    Probe::Answer {
                        entity: "王五",
                        attribute: "国籍",
                        value: "中国",
                        obligations: &[Obligation {
                            predicate: "国籍",
                            subject: "王五",
                            object: "中国",
                        }],
                    },
                ),
            ],
            perturbation: Perturbation {
                label: "王五的国籍改为日本",
                kind: PerturbationKind::Flip,
                turn: 5,
                remove: &[Fact::Attribute {
                    entity: "王五",
                    key: "国籍",
                    value: "中国",
                }],
                add: &[Fact::Attribute {
                    entity: "王五",
                    key: "国籍",
                    value: "日本",
                }],
                expected_before: "中国",
                expected_after: "日本",
            },
        },
    ];
    SEED
}

// ---------------------------------------------------------------------------
// 事实编译
// ---------------------------------------------------------------------------

/// 把基准事实编译成 `PrologFact`：实体属性聚合成一个 Entity，关系保持有向。
pub fn prolog_facts(facts: &[Fact]) -> Vec<PrologFact> {
    let mut attributes: BTreeMap<&'static str, BTreeMap<&'static str, &'static str>> =
        BTreeMap::new();
    let mut entities: BTreeSet<&'static str> = BTreeSet::new();
    let mut relations: Vec<(&'static str, &'static str, &'static str)> = Vec::new();
    for fact in facts {
        match fact {
            Fact::Attribute { entity, key, value } => {
                entities.insert(entity);
                attributes.entry(entity).or_default().insert(key, value);
            }
            Fact::Relation { from, relation, to } => {
                entities.insert(from);
                entities.insert(to);
                relations.push((from, relation, to));
            }
        }
    }
    let mut result = Vec::new();
    for entity in entities {
        let properties = attributes
            .remove(entity)
            .unwrap_or_default()
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        result.push(PrologFact::Entity {
            name: entity.to_string(),
            entity_type: "人物".to_string(),
            properties,
        });
    }
    for (from, rel_type, to) in relations {
        result.push(PrologFact::Relation {
            from: from.to_string(),
            rel_type: rel_type.to_string(),
            to: to.to_string(),
        });
    }
    result
}

/// 由基准事实构建一个独立的归结引擎（用于语料自检与证距估计）。
pub fn resolver_for(facts: &[Fact]) -> ResolutionEngine {
    let mut engine = ResolutionEngine::new(8, 8);
    for fact in facts {
        match fact {
            Fact::Attribute { entity, key, value } => {
                engine.add_fact(key, entity, value);
            }
            Fact::Relation { from, relation, to } => {
                engine.add_fact(relation, from, to);
            }
        }
    }
    engine
}

// ---------------------------------------------------------------------------
// 结果模型
// ---------------------------------------------------------------------------

/// 单个探针的判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Passed,
    Failed,
    /// 无法形式化或搜索被截断：既不奖励也不惩罚。
    Unchecked,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Passed => "passed",
            Verdict::Failed => "failed",
            Verdict::Unchecked => "unchecked",
        }
    }

    pub fn is_checked(self) -> bool {
        self != Verdict::Unchecked
    }
}

/// 有界证距估计。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofDistance {
    /// 失败目标的规范形式（`&` 连接）。
    pub goal: String,
    /// 预算内找到的最少**已验证**假设条数；None 表示预算内没有找到可验证场景。
    pub minimum_hypotheses: Option<usize>,
    /// 最少场景里的假设（规范原子），供人看「还差哪条事实」。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hypotheses: Vec<String>,
    /// 该场景是否经过注入后的 SLD 反驳验证；false 时只是失败前沿上的候选方向。
    #[serde(default)]
    pub verified: bool,
    /// 恒为 true：这是有界溯因估计，不是存在性证明。
    pub bounded: bool,
}

/// 单个探针的可序列化报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub scenario: String,
    pub family: Family,
    pub turn: usize,
    pub input: String,
    pub verdict: Verdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance: Option<ProofDistance>,
}

/// 一个扰动的可序列化结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerturbationReport {
    pub scenario: String,
    pub label: String,
    pub kind: PerturbationKind,
    pub passed: bool,
    pub expected_before: String,
    pub expected_after: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_observed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutated_observed: Option<String>,
    pub detail: String,
}

/// 家族汇总。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FamilyReport {
    pub probes: u32,
    pub passed: u32,
    pub failed: u32,
    pub unchecked: u32,
    pub pass_rate: f64,
    pub coverage: f64,
}

/// 完整报告（供基线棘轮、JSON 与 CI 消费）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofBenchReport {
    pub corpus_digest: String,
    pub scenarios: u32,
    pub turns: u32,
    pub probes: u32,
    pub passed: u32,
    pub failed: u32,
    pub unchecked: u32,
    pub families: BTreeMap<String, FamilyReport>,
    pub results: Vec<ProbeReport>,
    pub perturbations: Vec<PerturbationReport>,
    pub metrics: BTreeMap<String, f64>,
}

impl ProofBenchReport {
    /// 标量指标摊平（键稳定、可扩展）。
    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        self.metrics.clone()
    }

    /// 确定性文本报告。
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "可证基准 ProofBench：{} 个场景 / {} 轮 / {} 个探针，语料摘要 {}。\n",
            self.scenarios, self.turns, self.probes, self.corpus_digest
        ));
        let checked = self.passed + self.failed;
        out.push_str(&format!(
            "探针：通过 {}/{}（已检查 {}/{}，未检查 {}）\n",
            self.passed, self.probes, checked, self.probes, self.unchecked
        ));
        for family in Family::ALL {
            if let Some(report) = self.families.get(family.label()) {
                out.push_str(&format!(
                    "  {}：通过 {}/{}，覆盖率 {:.2}\n",
                    family.label(),
                    report.passed,
                    report.probes,
                    report.coverage
                ));
            }
        }

        let flips: Vec<&PerturbationReport> = self
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Flip)
            .collect();
        let flip_passed = flips.iter().filter(|p| p.passed).count();
        let distractors: Vec<&PerturbationReport> = self
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Distractor)
            .collect();
        let distractor_passed = distractors.iter().filter(|p| p.passed).count();
        out.push_str(&format!(
            "反事实翻转：{}/{}；干扰项不变：{}/{}\n",
            flip_passed,
            flips.len(),
            distractor_passed,
            distractors.len()
        ));

        let distances: Vec<&ProofDistance> = self
            .results
            .iter()
            .filter_map(|result| result.distance.as_ref())
            .collect();
        if !distances.is_empty() {
            let verified: Vec<usize> = distances
                .iter()
                .filter_map(|distance| distance.minimum_hypotheses)
                .collect();
            let unverified = distances.len().saturating_sub(verified.len());
            if verified.is_empty() {
                out.push_str(&format!(
                    "证距（有界溯因）：{} 个不可答探针，预算内没有找到可验证场景（{} 个仅有未验证候选）\n",
                    distances.len(), unverified
                ));
            } else {
                let mean = verified.iter().sum::<usize>() as f64 / verified.len() as f64;
                out.push_str(&format!(
                    "证距（有界溯因上界）：{} 个不可答探针，{} 个预算内验证，均值 {:.2} 条假设（min {}，max {}），{} 个仅有未验证候选\n",
                    distances.len(),
                    verified.len(),
                    mean,
                    verified.iter().min().copied().unwrap_or(0),
                    verified.iter().max().copied().unwrap_or(0),
                    unverified
                ));
            }
        }

        let failures: Vec<&ProbeReport> = self
            .results
            .iter()
            .filter(|result| result.verdict == Verdict::Failed)
            .collect();
        if failures.is_empty() {
            out.push_str("没有失败探针。\n");
        } else {
            out.push_str(&format!("失败探针（{}）：\n", failures.len()));
            for failure in failures {
                out.push_str(&format!(
                    "  - [{}] {} / 轮 {}「{}」：{}\n",
                    failure.family.label(),
                    failure.scenario,
                    failure.turn + 1,
                    failure.input,
                    failure.detail
                ));
                if let Some(distance) = &failure.distance {
                    out.push_str(&format!(
                        "      证距：{}（有界上界，min={:?}）\n",
                        distance.goal, distance.minimum_hypotheses
                    ));
                }
            }
        }
        let perturb_failures: Vec<&PerturbationReport> = self
            .perturbations
            .iter()
            .filter(|perturbation| !perturbation.passed)
            .collect();
        if !perturb_failures.is_empty() {
            out.push_str(&format!("未通过的扰动（{}）：\n", perturb_failures.len()));
            for failure in perturb_failures {
                out.push_str(&format!(
                    "  - [{}] {}「{}」：{}\n",
                    failure.kind.label(),
                    failure.scenario,
                    failure.label,
                    failure.detail
                ));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 执行
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TurnReading {
    input: String,
    normalized_question: Option<String>,
    entity: Option<String>,
    attribute: Option<String>,
}

struct ProbeOutcome {
    verdict: Verdict,
    observed: Option<String>,
    detail: String,
    distance: Option<ProofDistance>,
}

/// 孪生重跑时替换探针的期望值与义务（义务对象随事实一起翻转）。
struct AnswerExpectation<'a> {
    value: &'a str,
    obligations: &'a [String],
}

/// 跑种子语料。
pub fn evaluate_seed() -> ProofBenchReport {
    evaluate(seed_corpus())
}

/// 跑任意语料（供测试与第三方语料复用）。
pub fn evaluate(scenarios: &[Scenario]) -> ProofBenchReport {
    let mut results = Vec::new();
    let mut perturbations = Vec::new();
    let mut families: BTreeMap<String, FamilyReport> = BTreeMap::new();
    let mut turns = 0u32;

    for scenario in scenarios {
        let probe_index = scenario.probe_index();
        let probe_turn = scenario
            .turns
            .get(probe_index)
            .expect("probe_index 必须指向有效轮次");
        let probe = probe_turn.probe.as_ref().expect("探针轮必须有 probe");

        let base = run_scenario(scenario, scenario.facts, None);
        turns += scenario.turns.len() as u32;

        let expected = expected_text(probe, scenario.perturbation.expected_before);
        let result = ProbeReport {
            scenario: scenario.name.to_string(),
            family: scenario.family,
            turn: probe_index,
            input: probe_turn.input.to_string(),
            verdict: base.outcome.verdict,
            expected,
            observed: base.outcome.observed.clone(),
            detail: base.outcome.detail.clone(),
            distance: base.outcome.distance.clone(),
        };
        let family = families
            .entry(scenario.family.label().to_string())
            .or_default();
        family.probes += 1;
        match result.verdict {
            Verdict::Passed => family.passed += 1,
            Verdict::Failed => family.failed += 1,
            Verdict::Unchecked => family.unchecked += 1,
        }
        results.push(result);

        // 孪生/干扰：同一脚本，换一份知识库重跑。反事实翻转时，
        // 期望值与证明义务必须一起翻转，否则「答案对了、依据还是旧的」。
        let mutated_facts = apply_perturbation(scenario.facts, &scenario.perturbation);
        let mut mutated_obligations: Vec<String> = Vec::new();
        let mutated_answer = match probe {
            Probe::Answer { obligations, .. } => {
                for obligation in *obligations {
                    let canonical = if obligation.object == scenario.perturbation.expected_before {
                        format!(
                            "{}({}, {})",
                            obligation.predicate,
                            obligation.subject,
                            scenario.perturbation.expected_after
                        )
                    } else {
                        obligation.canonical()
                    };
                    mutated_obligations.push(canonical);
                }
                Some(AnswerExpectation {
                    value: scenario.perturbation.expected_after,
                    obligations: &mutated_obligations,
                })
            }
            _ => None,
        };
        let mutated = run_scenario(scenario, &mutated_facts, mutated_answer);
        perturbations.push(evaluate_perturbation(
            scenario,
            &base.outcome,
            &mutated.outcome,
        ));
    }

    for family in families.values_mut() {
        family.pass_rate = ratio(family.passed, family.passed + family.failed);
        family.coverage = ratio(family.passed + family.failed, family.probes);
    }

    let passed = results
        .iter()
        .filter(|result| result.verdict == Verdict::Passed)
        .count() as u32;
    let failed = results
        .iter()
        .filter(|result| result.verdict == Verdict::Failed)
        .count() as u32;
    let unchecked = results
        .iter()
        .filter(|result| result.verdict == Verdict::Unchecked)
        .count() as u32;
    let probes = results.len() as u32;
    let metrics = flatten(
        probes,
        passed,
        failed,
        unchecked,
        &families,
        &perturbations,
        &results,
    );

    ProofBenchReport {
        corpus_digest: corpus_digest(scenarios),
        scenarios: scenarios.len() as u32,
        turns,
        probes,
        passed,
        failed,
        unchecked,
        families,
        results,
        perturbations,
        metrics,
    }
}

struct ScenarioRun {
    outcome: ProbeOutcome,
}

fn run_scenario(
    scenario: &Scenario,
    facts: &[Fact],
    answer_override: Option<AnswerExpectation<'_>>,
) -> ScenarioRun {
    let prolog_facts = prolog_facts(facts);
    let config = ReasoningConfig {
        enable_link_prediction: false,
        ..ReasoningConfig::default()
    };
    let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
    reasoner.observe_facts(&prolog_facts);
    let mut manager = DialogueManager::new();
    let mut readings: Vec<TurnReading> = Vec::new();
    let probe_index = scenario.probe_index();

    for turn in scenario.turns {
        // 与产品对话路径一致：先解读，再走受控生成以推进焦点栈。
        let frame = manager.analyze(turn.input, &reasoner);
        let _reply = manager
            .plan_pure_dialogue(turn.input, &reasoner)
            .or_else(|| manager.plan_reply(turn.input, &reasoner));
        readings.push(TurnReading {
            input: turn.input.to_string(),
            normalized_question: frame.question,
            entity: frame.entity,
            attribute: frame.attribute,
        });
    }

    let reading = &readings[probe_index];
    let probe = scenario
        .turns
        .get(probe_index)
        .and_then(|turn| turn.probe.as_ref())
        .expect("探针轮必须有 probe");
    let outcome = evaluate_probe(probe, reading, &reasoner, scenario, facts, answer_override);
    ScenarioRun { outcome }
}

fn evaluate_probe(
    probe: &Probe,
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    scenario: &Scenario,
    facts: &[Fact],
    answer_override: Option<AnswerExpectation<'_>>,
) -> ProbeOutcome {
    match probe {
        Probe::Answer {
            entity,
            attribute,
            value,
            obligations,
        } => evaluate_answer(
            *entity,
            *attribute,
            *value,
            obligations,
            reading,
            reasoner,
            answer_override,
        ),
        Probe::Unknown {
            entity,
            attribute,
            goals,
        } => evaluate_unknown(*entity, *attribute, goals, reading, reasoner, facts),
        Probe::UnknownEntity { entity } => evaluate_unknown_entity(entity, reasoner, scenario),
    }
}

fn evaluate_answer(
    entity: &str,
    attribute: &str,
    value: &str,
    obligations: &[Obligation],
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    answer_override: Option<AnswerExpectation<'_>>,
) -> ProbeOutcome {
    let (value, obligations): (&str, Vec<String>) = match answer_override {
        Some(expectation) => (expectation.value, expectation.obligations.to_vec()),
        None => (
            value,
            obligations.iter().map(Obligation::canonical).collect(),
        ),
    };
    if reading.entity.as_deref() != Some(entity) || reading.attribute.as_deref() != Some(attribute)
    {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: reading.entity.clone(),
            detail: format!(
                "问句被读成实体 {:?} / 属性 {:?}，期望实体 {:?} / 属性 {:?}",
                reading.entity, reading.attribute, entity, attribute
            ),
            distance: None,
        };
    }
    let Some(normalized) = reading.normalized_question.as_deref() else {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: "没有生成规范化问句，无法进入证明路径".to_string(),
            distance: None,
        };
    };
    let Some(view) = reasoner.resolution_proof_view(normalized) else {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("无法把「{}」形式化为归结目标", normalized),
            distance: None,
        };
    };
    if view.solutions.is_empty() {
        if view.search_status == SearchStatus::Incomplete || view.search_truncated {
            return ProbeOutcome {
                verdict: Verdict::Unchecked,
                observed: None,
                detail: format!("搜索被截断（{}），不能判错", view.goal_display),
                distance: None,
            };
        }
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: format!("知识库未能证明目标：{}", view.goal_display),
            distance: None,
        };
    }

    let mut first_observed: Option<String> = None;
    let mut closest_missing: Vec<String> = Vec::new();
    for solution in &view.solutions {
        let observed = answer_binding(&solution.bindings).map(str::to_string);
        if first_observed.is_none() {
            first_observed = observed.clone();
        }
        let proof_atoms = collect_proof_atoms(&solution.proofs);
        let missing: Vec<String> = obligations
            .iter()
            .filter(|obligation| !proof_atoms.contains(*obligation))
            .cloned()
            .collect();
        if observed.as_deref() == Some(value) && missing.is_empty() {
            return ProbeOutcome {
                verdict: Verdict::Passed,
                observed,
                detail: format!(
                    "证明树覆盖 {} 条义务：{}",
                    obligations.len(),
                    obligations.join("，")
                ),
                distance: None,
            };
        }
        if closest_missing.is_empty() {
            closest_missing = missing;
        }
    }

    let detail = if !closest_missing.is_empty() {
        format!(
            "答案绑定为 {:?}，但证明树缺少义务：{}",
            first_observed,
            closest_missing.join("，")
        )
    } else {
        format!("答案绑定为 {:?}，与期望「{}」不一致", first_observed, value)
    };
    ProbeOutcome {
        verdict: Verdict::Failed,
        observed: first_observed,
        detail,
        distance: None,
    }
}

fn evaluate_unknown(
    entity: &str,
    attribute: &str,
    goals: &[Goal],
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    facts: &[Fact],
) -> ProbeOutcome {
    if reading.entity.as_deref() != Some(entity) || reading.attribute.as_deref() != Some(attribute)
    {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: reading.entity.clone(),
            detail: format!(
                "不可答问句被读成实体 {:?} / 属性 {:?}，期望实体 {:?} / 属性 {:?}",
                reading.entity, reading.attribute, entity, attribute
            ),
            distance: None,
        };
    }
    let Some(normalized) = reading.normalized_question.as_deref() else {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: "没有生成规范化问句，无法核对不可答性".to_string(),
            distance: None,
        };
    };
    let Some(view) = reasoner.resolution_proof_view(normalized) else {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("无法把「{}」形式化为归结目标", normalized),
            distance: Some(compute_distance(facts, goals)),
        };
    };
    if !view.solutions.is_empty() {
        let observed = view
            .solutions
            .first()
            .and_then(|solution| answer_binding(&solution.bindings))
            .map(str::to_string);
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed,
            detail: format!(
                "知识库本应不可答，却证明了 {} 个解：{}",
                view.solutions.len(),
                view.goal_display
            ),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    if view.search_status == SearchStatus::Incomplete || view.search_truncated {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("搜索被截断（{}），不能断言不可答", view.goal_display),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    // 形式证明为空还不够：回答管线也不能把别的话题证据包装成确定答案。
    if let Some(answer) = reasoner.answer_certain(&reading.input) {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(clip(&answer, 120)),
            detail: format!(
                "归结搜索已穷尽且无解，但回答管线仍给出确定答案（目标：{}）",
                view.goal_display
            ),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    ProbeOutcome {
        verdict: Verdict::Passed,
        observed: None,
        detail: format!("搜索已穷尽且无解，且没有确定回答：{}", view.goal_display),
        distance: Some(compute_distance(facts, goals)),
    }
}

fn evaluate_unknown_entity(
    entity: &str,
    reasoner: &SymbolicReasoner,
    scenario: &Scenario,
) -> ProbeOutcome {
    let known = reasoner.known_entity(entity);
    if known {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(entity.to_string()),
            detail: format!("「{}」本应未知，却被登记为已知实体", entity),
            distance: None,
        };
    }
    let input = scenario
        .turns
        .get(scenario.probe_index())
        .map(|turn| turn.input)
        .unwrap_or("");
    if let Some(answer) = reasoner.answer_certain(input) {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(clip(&answer, 80)),
            detail: format!("未知实体「{}」却给出了确定回答", entity),
            distance: None,
        };
    }
    ProbeOutcome {
        verdict: Verdict::Passed,
        observed: None,
        detail: format!("「{}」未被登记，且没有确定回答", entity),
        distance: None,
    }
}

fn evaluate_perturbation(
    scenario: &Scenario,
    base: &ProbeOutcome,
    mutated: &ProbeOutcome,
) -> PerturbationReport {
    let perturbation = &scenario.perturbation;
    let (passed, detail) = match perturbation.kind {
        PerturbationKind::Flip => {
            let flipped = base.verdict == Verdict::Passed
                && mutated.verdict == Verdict::Passed
                && base.observed.as_deref() == Some(perturbation.expected_before)
                && mutated.observed.as_deref() == Some(perturbation.expected_after)
                && base.observed != mutated.observed;
            let detail = if flipped {
                format!(
                    "答案按预期从「{}」翻转为「{}」",
                    perturbation.expected_before, perturbation.expected_after
                )
            } else {
                format!(
                    "期望「{}」→「{}」，实际 {:?}（{:?}）→ {:?}（{:?}）",
                    perturbation.expected_before,
                    perturbation.expected_after,
                    base.observed,
                    base.verdict,
                    mutated.observed,
                    mutated.verdict
                )
            };
            (flipped, detail)
        }
        PerturbationKind::Distractor => {
            let stable = base.verdict == Verdict::Passed
                && mutated.verdict == Verdict::Passed
                && base.observed == mutated.observed
                && perturbation.expected_before == perturbation.expected_after;
            let detail = if stable {
                format!("加入干扰后结论保持「{}」", perturbation.expected_after)
            } else {
                format!(
                    "期望结论不变（「{}」），实际 {:?}（{:?}）→ {:?}（{:?}）",
                    perturbation.expected_after,
                    base.observed,
                    base.verdict,
                    mutated.observed,
                    mutated.verdict
                )
            };
            (stable, detail)
        }
    };
    PerturbationReport {
        scenario: scenario.name.to_string(),
        label: perturbation.label.to_string(),
        kind: perturbation.kind,
        passed,
        expected_before: perturbation.expected_before.to_string(),
        expected_after: perturbation.expected_after.to_string(),
        base_observed: base.observed.clone(),
        mutated_observed: mutated.observed.clone(),
        detail,
    }
}

fn expected_text(probe: &Probe, perturbation_before: &str) -> Option<String> {
    match probe {
        Probe::Answer { value, .. } => Some((*value).to_string()),
        Probe::Unknown { .. } | Probe::UnknownEntity { .. } => {
            if perturbation_before == "unknown" {
                None
            } else {
                Some(perturbation_before.to_string())
            }
        }
    }
}

fn answer_binding(bindings: &[(String, String)]) -> Option<&str> {
    bindings
        .iter()
        .find(|(name, value)| name == "答案" && !value.is_empty())
        .map(|(_, value)| value.as_str())
        .or_else(|| {
            bindings
                .iter()
                .rev()
                .find(|(name, value)| !name.starts_with('V') && !value.is_empty())
                .map(|(_, value)| value.as_str())
        })
}

fn collect_proof_atoms(nodes: &[ProofViewNode]) -> BTreeSet<String> {
    let mut atoms = BTreeSet::new();
    fn walk(nodes: &[ProofViewNode], atoms: &mut BTreeSet<String>) {
        for node in nodes {
            atoms.insert(node.canonical.clone());
            walk(&node.children, atoms);
        }
    }
    walk(nodes, &mut atoms);
    atoms
}

// ---------------------------------------------------------------------------
// 证距
// ---------------------------------------------------------------------------

/// 有界溯因证距：目标可证则为 0；否则在预算内找最少假设联合场景。
pub fn compute_distance(facts: &[Fact], goals: &[Goal]) -> ProofDistance {
    let engine = resolver_for(facts);
    let atoms: Vec<Atom> = goals.iter().map(Goal::atom).collect();
    let mut query_vars: Vec<String> = Vec::new();
    for goal in goals {
        for term in [goal.subject, goal.object] {
            if let TermSpec::Var(name) = term {
                if !query_vars.iter().any(|var| var == name) {
                    query_vars.push(name.to_string());
                }
            }
        }
    }
    let goal_text = if atoms.is_empty() {
        "（空目标）".to_string()
    } else {
        atoms
            .iter()
            .map(Atom::display)
            .collect::<Vec<_>>()
            .join(" & ")
    };
    if !engine.solve(&atoms, &query_vars).is_empty() {
        return ProofDistance {
            goal: goal_text,
            minimum_hypotheses: Some(0),
            hypotheses: Vec::new(),
            verified: true,
            bounded: true,
        };
    }
    let mut abduction = AbductionEngine::new(4, 8);
    abduction.observe_facts(&prolog_facts(facts));
    let scenarios = abduction.scenarios(&engine, &atoms, &query_vars);
    let best_verified = scenarios
        .iter()
        .filter(|scenario| scenario.verified)
        .min_by_key(|scenario| scenario.size());
    if let Some(scenario) = best_verified {
        return ProofDistance {
            goal: goal_text,
            minimum_hypotheses: Some(scenario.size()),
            hypotheses: hypothesis_atoms(scenario),
            verified: true,
            bounded: true,
        };
    }
    // 没有通过注入验证的组合：仍把失败前沿上的最近候选写出来，
    // 但 minimum_hypotheses 保持 None，绝不把候选方向冒充成可证上界。
    match scenarios.iter().min_by_key(|scenario| scenario.size()) {
        Some(scenario) => ProofDistance {
            goal: goal_text,
            minimum_hypotheses: None,
            hypotheses: hypothesis_atoms(scenario),
            verified: false,
            bounded: true,
        },
        None => ProofDistance {
            goal: goal_text,
            minimum_hypotheses: None,
            hypotheses: Vec::new(),
            verified: false,
            bounded: true,
        },
    }
}

fn hypothesis_atoms(scenario: &crate::abduction::Scenario) -> Vec<String> {
    let mut hypotheses: Vec<String> = scenario.atoms().iter().map(Atom::display).collect();
    hypotheses.sort();
    hypotheses
}

fn apply_perturbation(facts: &[Fact], perturbation: &Perturbation) -> Vec<Fact> {
    let mut result: Vec<Fact> = facts.to_vec();
    for remove in perturbation.remove {
        result.retain(|fact| fact != remove);
    }
    result.extend_from_slice(perturbation.add);
    result
}

// ---------------------------------------------------------------------------
// 汇总 / 摘要 / 导出
// ---------------------------------------------------------------------------

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn flatten(
    probes: u32,
    passed: u32,
    failed: u32,
    unchecked: u32,
    families: &BTreeMap<String, FamilyReport>,
    perturbations: &[PerturbationReport],
    results: &[ProbeReport],
) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    let checked = passed + failed;
    metrics.insert("overall.pass_rate".to_string(), ratio(passed, checked));
    metrics.insert("overall.coverage".to_string(), ratio(checked, probes));
    metrics.insert("overall.passed".to_string(), passed as f64);
    metrics.insert("overall.failed".to_string(), failed as f64);
    metrics.insert("overall.unchecked".to_string(), unchecked as f64);
    for family in Family::ALL {
        if let Some(report) = families.get(family.label()) {
            metrics.insert(
                format!("family.{}.pass_rate", family.label()),
                report.pass_rate,
            );
            metrics.insert(
                format!("family.{}.coverage", family.label()),
                report.coverage,
            );
        }
    }
    let flips: Vec<&PerturbationReport> = perturbations
        .iter()
        .filter(|p| p.kind == PerturbationKind::Flip)
        .collect();
    let flip_passed = flips.iter().filter(|p| p.passed).count() as u32;
    let distractors: Vec<&PerturbationReport> = perturbations
        .iter()
        .filter(|p| p.kind == PerturbationKind::Distractor)
        .collect();
    let distractor_passed = distractors.iter().filter(|p| p.passed).count() as u32;
    metrics.insert(
        "twin.flip_rate".to_string(),
        ratio(flip_passed, flips.len() as u32),
    );
    metrics.insert(
        "twin.distractor_stability".to_string(),
        ratio(distractor_passed, distractors.len() as u32),
    );
    let perturb_passed = perturbations.iter().filter(|p| p.passed).count() as u32;
    metrics.insert(
        "perturbation.pass_rate".to_string(),
        ratio(perturb_passed, perturbations.len() as u32),
    );
    let distances: Vec<&ProofDistance> = results
        .iter()
        .filter_map(|result| result.distance.as_ref())
        .collect();
    let verified_distances: Vec<usize> = distances
        .iter()
        .filter_map(|distance| distance.minimum_hypotheses)
        .collect();
    if !verified_distances.is_empty() {
        metrics.insert(
            "proof_distance.mean".to_string(),
            verified_distances.iter().sum::<usize>() as f64 / verified_distances.len() as f64,
        );
        metrics.insert(
            "proof_distance.observed".to_string(),
            verified_distances.len() as f64,
        );
    }
    metrics.insert("proof_distance.probes".to_string(), distances.len() as f64);
    metrics.insert(
        "proof_distance.unverified".to_string(),
        distances.len().saturating_sub(verified_distances.len()) as f64,
    );
    metrics
}

/// 越大越差的指标（棘轮反向判定）。
pub const LOWER_IS_BETTER: [&str; 3] = [
    "proof_distance.mean",
    "proof_distance.unverified",
    "overall.unchecked",
];

/// 可写入磁盘的基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofBenchBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

pub const PROOF_BENCH_BASELINE_VERSION: u32 = 1;

impl ProofBenchBaseline {
    pub fn from_report(report: &ProofBenchReport) -> Self {
        Self {
            version: PROOF_BENCH_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            metrics: report.flatten_metrics(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct RatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<MetricDelta>,
    pub gains: Vec<MetricDelta>,
}

impl RatchetVerdict {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        if self.passed {
            out.push_str("ProofBench 棘轮检查通过：无指标低于基线。\n");
        } else {
            out.push_str("ProofBench 棘轮检查未通过：\n");
            if !self.digest_matches {
                out.push_str("  - 语料摘要不匹配：基准已漂移，基线不可直接比较\n");
            }
            for regression in &self.regressions {
                let direction = if LOWER_IS_BETTER.contains(&regression.metric.as_str()) {
                    "（越低越好）"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "  - {}: {:.6} → {:.6}{}\n",
                    regression.metric, regression.baseline, regression.current, direction
                ));
            }
        }
        if !self.gains.is_empty() {
            out.push_str(&format!("有 {} 项指标高于基线。\n", self.gains.len()));
        }
        out
    }
}

/// 与基线比较：语料摘要不同或任何指标退化超 ε 即不通过。
pub fn compare_baseline(
    baseline: &ProofBenchBaseline,
    current: &ProofBenchReport,
    epsilon: f64,
) -> RatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        let lower_is_better = LOWER_IS_BETTER.contains(&metric.as_str());
        let (is_regression, is_gain) = if lower_is_better {
            (
                current_value > baseline_value + epsilon,
                current_value < baseline_value - epsilon,
            )
        } else {
            (
                current_value < baseline_value - epsilon,
                current_value > baseline_value + epsilon,
            )
        };
        if is_regression {
            regressions.push(MetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(MetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    RatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

/// 确定性语料摘要（FNV-1a 64）：知识、问句或义务变化必须让基线失效。
pub fn corpus_digest(scenarios: &[Scenario]) -> String {
    let mut canonical = String::new();
    for scenario in scenarios {
        canonical.push_str("scenario=");
        canonical.push_str(scenario.name);
        canonical.push('|');
        canonical.push_str(scenario.family.label());
        canonical.push('\n');
        for fact in scenario.facts {
            canonical.push_str("fact=");
            match fact {
                Fact::Attribute { entity, key, value } => {
                    canonical.push_str(&format!("attr:{entity}|{key}|{value}"));
                }
                Fact::Relation { from, relation, to } => {
                    canonical.push_str(&format!("rel:{from}|{relation}|{to}"));
                }
            }
            canonical.push('\n');
        }
        for (index, turn) in scenario.turns.iter().enumerate() {
            canonical.push_str(&format!("turn={index}|{}|", turn.input));
            match &turn.probe {
                None => canonical.push_str("context"),
                Some(Probe::Answer {
                    entity,
                    attribute,
                    value,
                    obligations,
                }) => {
                    canonical.push_str(&format!("answer:{entity}|{attribute}|{value}|"));
                    for obligation in *obligations {
                        canonical.push_str(&obligation.canonical());
                        canonical.push(';');
                    }
                }
                Some(Probe::Unknown {
                    entity,
                    attribute,
                    goals,
                }) => {
                    canonical.push_str(&format!("unknown:{entity}|{attribute}|"));
                    for goal in *goals {
                        canonical.push_str(&format!(
                            "{}({},{})",
                            goal.predicate,
                            goal.subject.name(),
                            goal.object.name()
                        ));
                        canonical.push(';');
                    }
                }
                Some(Probe::UnknownEntity { entity }) => {
                    canonical.push_str(&format!("unknown_entity:{entity}"));
                }
            }
            canonical.push('\n');
        }
        let perturbation = &scenario.perturbation;
        canonical.push_str(&format!(
            "perturb={}|{}|{}|{}|{}|{}\n",
            perturbation.label,
            perturbation.kind.label(),
            perturbation.turn,
            perturbation.expected_before,
            perturbation.expected_after,
            scenario.facts.len()
        ));
        for fact in perturbation.remove {
            canonical.push_str(&format!("remove={fact:?}\n"));
        }
        for fact in perturbation.add {
            canonical.push_str(&format!("add={fact:?}\n"));
        }
    }
    fnv1a_hex(canonical.as_bytes())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

/// 导出为纯 Prolog 事实，供第三方 `consult` 后独立审计。
///
/// 只导出数据与义务，不导出任何评分逻辑：评分口径写在 [`ProofBenchReport`]。
pub fn render_prolog(scenarios: &[Scenario]) -> String {
    let mut out = String::new();
    out.push_str("% ProofBench 语料导出：可 consult 的纯事实（只读审计）。\n");
    out.push_str(&format!("% corpus_digest({}).\n", corpus_digest(scenarios)));
    for scenario in scenarios {
        out.push_str(&format!(
            "\n% ---- {} ({}) ----\n",
            scenario.name,
            scenario.family.label()
        ));
        out.push_str(&format!(
            "bench_scenario('{}', '{}').\n",
            escape_atom(scenario.name),
            scenario.family.label()
        ));
        for fact in scenario.facts {
            match fact {
                Fact::Attribute { entity, key, value } => out.push_str(&format!(
                    "bench_fact('{}', '{}', '{}').\n",
                    escape_atom(entity),
                    escape_atom(key),
                    escape_atom(value)
                )),
                Fact::Relation { from, relation, to } => out.push_str(&format!(
                    "bench_fact('{}', '{}', '{}').\n",
                    escape_atom(from),
                    escape_atom(relation),
                    escape_atom(to)
                )),
            }
        }
        for (index, turn) in scenario.turns.iter().enumerate() {
            let kind = match &turn.probe {
                None => "context",
                Some(Probe::Answer { .. }) => "answer",
                Some(Probe::Unknown { .. }) => "unknown",
                Some(Probe::UnknownEntity { .. }) => "unknown_entity",
            };
            out.push_str(&format!(
                "bench_turn('{}', {}, '{}', '{}').\n",
                escape_atom(scenario.name),
                index,
                escape_atom(turn.input),
                kind
            ));
            match &turn.probe {
                Some(Probe::Answer {
                    value, obligations, ..
                }) => {
                    out.push_str(&format!(
                        "bench_answer('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(value)
                    ));
                    for obligation in *obligations {
                        out.push_str(&format!(
                            "bench_obligation('{}', {}, '{}').\n",
                            escape_atom(scenario.name),
                            index,
                            escape_atom(&obligation.canonical())
                        ));
                    }
                }
                Some(Probe::Unknown { goals, .. }) => {
                    let goal = goals
                        .iter()
                        .map(|goal| {
                            format!(
                                "{}({},{})",
                                goal.predicate,
                                goal.subject.name(),
                                goal.object.name()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" & ");
                    out.push_str(&format!(
                        "bench_query('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(&goal)
                    ));
                }
                Some(Probe::UnknownEntity { entity }) => {
                    out.push_str(&format!(
                        "bench_unknown_entity('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(entity)
                    ));
                }
                None => {}
            }
        }
        out.push_str(&format!(
            "bench_perturbation('{}', '{}', '{}', '{}', '{}').\n",
            escape_atom(scenario.name),
            escape_atom(scenario.perturbation.label),
            scenario.perturbation.kind.label(),
            escape_atom(scenario.perturbation.expected_before),
            escape_atom(scenario.perturbation.expected_after)
        ));
    }
    out
}

fn escape_atom(text: &str) -> String {
    text.replace('\'', "''")
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_shape_is_stable() {
        let scenarios = seed_corpus();
        assert_eq!(scenarios.len(), 12);
        for family in Family::ALL {
            assert_eq!(
                scenarios
                    .iter()
                    .filter(|scenario| scenario.family == family)
                    .count(),
                3,
                "每个家族应恰好 3 个场景"
            );
        }
        for scenario in scenarios {
            let probes = scenario
                .turns
                .iter()
                .filter(|turn| turn.probe.is_some())
                .count();
            assert_eq!(probes, 1, "{} 应恰好一个探针", scenario.name);
            assert_eq!(
                scenario.perturbation.turn,
                scenario.probe_index(),
                "{} 的扰动必须指向探针轮",
                scenario.name
            );
        }
        let flips = scenarios
            .iter()
            .filter(|scenario| scenario.perturbation.kind == PerturbationKind::Flip)
            .count();
        let distractors = scenarios.len() - flips;
        assert_eq!(flips, 8);
        assert_eq!(distractors, 4);
    }

    #[test]
    fn corpus_answer_obligations_are_provable() {
        for scenario in seed_corpus() {
            let probe = scenario.turns[scenario.probe_index()]
                .probe
                .as_ref()
                .unwrap();
            if let Probe::Answer { obligations, .. } = probe {
                let engine = resolver_for(scenario.facts);
                for obligation in *obligations {
                    let atom =
                        Atom::rel(obligation.predicate, obligation.subject, obligation.object);
                    assert_eq!(
                        engine.solve(&[atom], &[]).len(),
                        1,
                        "{} 的义务 {} 必须由基础知识库证明",
                        scenario.name,
                        obligation.canonical()
                    );
                }
            }
        }
    }

    #[test]
    fn corpus_unknown_goals_are_genuinely_unprovable() {
        for scenario in seed_corpus() {
            let probe = scenario.turns[scenario.probe_index()]
                .probe
                .as_ref()
                .unwrap();
            if let Probe::Unknown { goals, .. } = probe {
                let engine = resolver_for(scenario.facts);
                let atoms: Vec<Atom> = goals.iter().map(Goal::atom).collect();
                assert!(
                    engine.solve(&atoms, &[]).is_empty(),
                    "{} 的不可答目标不应被基础知识库证明",
                    scenario.name
                );
            }
        }
    }

    #[test]
    fn report_is_deterministic_and_metrics_bounded() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second, "同一语料必须给出逐字节可复现的报告");
        assert_eq!(first.scenarios, 12);
        assert_eq!(first.probes, 12);
        assert_eq!(first.perturbations.len(), 12);
        assert!(first.passed + first.failed + first.unchecked == first.probes);
        for (metric, value) in &first.metrics {
            if metric.ends_with(".observed")
                || metric.ends_with(".passed")
                || metric.ends_with(".failed")
                || metric.ends_with(".unchecked")
                || metric.ends_with(".probes")
                || metric.ends_with(".unverified")
            {
                continue;
            }
            assert!(
                (0.0..=1.0).contains(value),
                "指标 {} 越界：{}",
                metric,
                value
            );
        }
        assert!(!first.corpus_digest.is_empty());
    }

    #[test]
    fn seed_corpus_exercises_all_mechanisms() {
        let report = evaluate_seed();
        assert!(
            report.families.values().all(|family| family.coverage > 0.0),
            "每个家族都必须至少有一个可检查探针：{}",
            report.render_text()
        );
        let flips = report
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Flip)
            .count();
        let distractors = report
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Distractor)
            .count();
        assert_eq!(flips, 8);
        assert_eq!(distractors, 4);
        let distances = report
            .results
            .iter()
            .filter_map(|result| result.distance.as_ref())
            .collect::<Vec<_>>();
        assert!(!distances.is_empty(), "不可答探针必须报告证距");
        assert!(distances.iter().all(|distance| distance.bounded));
        assert_eq!(distances.len(), 2, "两个 Probe::Unknown 都应给出证距对象");
        assert!(
            distances
                .iter()
                .any(|distance| distance.minimum_hypotheses.is_some()),
            "至少应有一个预算内可验证的证距估计"
        );
    }

    #[test]
    fn unknown_bridge_rejects_off_topic_evidence() {
        let report = evaluate_seed();
        let result = report
            .results
            .iter()
            .find(|result| result.scenario == "未知桥接")
            .expect("语料必须包含未知桥接场景");
        assert_eq!(
            result.verdict,
            Verdict::Passed,
            "「配偶」的话题证据不能冒充「配偶的体重」的确定答案：{}",
            result.detail
        );
        let distance = result.distance.as_ref().expect("应报告证距");
        assert!(distance.verified, "{:?}", distance);
    }

    #[test]
    fn baseline_ratchet_detects_regressions_and_drift() {
        let report = evaluate_seed();
        let baseline = ProofBenchBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 0.0);
        assert!(verdict.passed, "{}", verdict.render_text());
        assert!(verdict.digest_matches);

        let mut degraded = report.clone();
        degraded
            .metrics
            .insert("overall.pass_rate".to_string(), 0.0);
        let verdict = compare_baseline(&baseline, &degraded, 0.0);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "overall.pass_rate"));

        let mut drifted = report.clone();
        drifted.corpus_digest = "0000000000000000".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 0.0);
        assert!(!verdict.passed && !verdict.digest_matches);
    }

    #[test]
    fn baseline_roundtrips_through_disk() {
        let report = evaluate_seed();
        let baseline = ProofBenchBaseline::from_report(&report);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("proof_bench_baseline.json");
        baseline.save(&path).unwrap();
        let loaded = ProofBenchBaseline::load(&path).unwrap();
        assert_eq!(loaded.version, baseline.version);
        assert_eq!(loaded.corpus_digest, baseline.corpus_digest);
        assert_eq!(loaded.metrics, baseline.metrics);
    }

    #[test]
    fn prolog_export_is_auditable() {
        let text = render_prolog(seed_corpus());
        assert!(text.contains("bench_scenario('篮球家庭', 'coref')."));
        assert!(text.contains("bench_fact('叶莉', '身高', '190')."));
        assert!(text.contains("bench_obligation('三跳师门', 0, '师傅(李四, 王五)')."));
        assert!(text.contains("bench_query('未知桥接'"));
        assert!(text.contains("bench_perturbation('"));
        assert!(text.contains("corpus_digest("));
        assert!(text
            .lines()
            .all(|line| line.is_empty() || line.starts_with('%') || line.ends_with('.')));
    }

    #[test]
    fn proof_distance_reports_a_bounded_gap() {
        let facts = [
            Fact::Attribute {
                entity: "姚明",
                key: "身高",
                value: "226",
            },
            Fact::Attribute {
                entity: "李娜",
                key: "体重",
                value: "55",
            },
        ];
        let goals = [Goal {
            predicate: "体重",
            subject: TermSpec::Entity("姚明"),
            object: TermSpec::Var("答案"),
        }];
        let distance = compute_distance(&facts, &goals);
        assert!(distance.bounded);
        assert!(distance.verified);
        assert_eq!(distance.minimum_hypotheses, Some(1), "{:?}", distance);
        assert!(
            distance
                .hypotheses
                .iter()
                .any(|atom| atom.starts_with("体重(姚明")),
            "{:?}",
            distance
        );
    }
}

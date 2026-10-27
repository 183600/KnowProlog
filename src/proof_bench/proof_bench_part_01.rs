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

//! KPCS（KnowProlog Conformance Suite）：对话能力的符合性契约。
//!
//! 无权重、无训练的系统里，测试不是「事后验证」，而是能力规范与回归资产：
//! 本套件按八个维度约束「像 LLM 的对话行为」，只断言不变量与必要子串，
//! 不断言整句，给模板组合生成保留措辞自由度。
//!
//!   C1 上下文记忆：焦点实体跨轮继承与焦点栈维护
//!   C2 指代与省略：「他/它/…呢」解析为 inherited 帧
//!   C3 组合生成：回答由结论 + 追问等多个片段组装，而非整句查表
//!   C4 溯源闭包：回答中每个「引号词条」都 ∈ 知识库 ∪ 对话历史
//!   C5 拒答不幻觉：未知实体让位、未知属性坦白，用哨兵实体检测跨场景泄漏
//!   C6 元对话：回顾 / 解释依据 / 续聊 / 展开 / 寒暄
//!   C7 状态充分性：快照-恢复-续聊 ≡ 不中断运行（逐字一致）
//!   C8 确定性与幂等：同一 transcript 两次回放输出一致
//!
//! 错误处理 × 神经符号推理扩展（恢复维度）：
//!   C14 失败可命名：引擎故障显式「出错」，绝不伪装成「没有记录」；
//!       Off / Observe / On 三档各自的行为契约（原样 / 只记账 / 显式恢复）
//!   C15 推测不幻觉：联想提名必带「推测」定性 + 证据 + 追问，可被用户裁定作废
//!   C16 恢复确定性与有界：同 transcript 回放逐字一致；挂账有 TTL 不滞留
//!
//! 另含两条跨模块一致性测试：生成器→解析器往返（互为 oracle）、
//! 真实 SymbolicReasoner 全栈冒烟。

use std::collections::{BTreeMap, HashMap};

use crate::chinese_reasoning_nlu::{parse_lessons, LessonKind};
use crate::chronicle::{Chronicle, FsyncPolicy};
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager, DialogueReply, MoveKind};
use crate::epistemic::EpistemicStatus;
use crate::prolog_engine::PrologFact;
use crate::recovery::{RecoveryMode, RecoveryNomination};
use crate::symbolic_generator::SymbolicGenerator;
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::ReasoningConfig;
use tempfile::TempDir;

/// 脚本化知识替身：把「像 LLM」的能力与领域知识解耦，测试只检验对话层不变量。
#[derive(Default)]
struct ScriptedKnowledge {
    entities: Vec<String>,
    known: Vec<String>,
    attributes: BTreeMap<String, Vec<String>>,
    answers: BTreeMap<String, String>,
    values: BTreeMap<(String, String), Vec<String>>,
    profiles: BTreeMap<String, String>,
    nominations: BTreeMap<(String, String), Vec<RecoveryNomination>>,
    clues: BTreeMap<String, Vec<String>>,
    fault: Option<String>,
}

impl ScriptedKnowledge {
    fn with_entity(mut self, entity: &str, attributes: &[&str]) -> Self {
        self.entities.push(entity.to_string());
        self.known.push(entity.to_string());
        self.attributes.insert(
            entity.to_string(),
            attributes.iter().map(|a| a.to_string()).collect(),
        );
        self
    }

    fn with_answer(mut self, question: &str, answer: &str) -> Self {
        self.answers
            .insert(question.to_string(), answer.to_string());
        self
    }

    fn with_value(mut self, entity: &str, attribute: &str, value: &str) -> Self {
        self.values.insert(
            (entity.to_string(), attribute.to_string()),
            vec![value.to_string()],
        );
        self
    }

    fn with_profile(mut self, entity: &str, profile: &str) -> Self {
        self.profiles
            .insert(entity.to_string(), profile.to_string());
        self
    }

    fn with_nomination(
        mut self,
        entity: &str,
        attribute: &str,
        value: &str,
        score: f64,
        evidence: &str,
    ) -> Self {
        self.nominations
            .entry((entity.to_string(), attribute.to_string()))
            .or_default()
            .push(RecoveryNomination {
                value: value.to_string(),
                score,
                evidence: vec![evidence.to_string()],
            });
        self
    }

    fn with_clue(mut self, entity: &str, clue: &str) -> Self {
        self.clues
            .entry(entity.to_string())
            .or_default()
            .push(clue.to_string());
        self
    }

    fn with_fault(mut self, fault: &str) -> Self {
        self.fault = Some(fault.to_string());
        self
    }

    /// C4 溯源闭包用：词条是否可回溯到知识库（实体 / 属性 / 取值 / 画像）。
    fn traceable(&self, token: &str) -> bool {
        self.known.iter().any(|known| known == token)
            || self
                .attributes
                .values()
                .flatten()
                .any(|attribute| attribute == token)
            || self
                .values
                .keys()
                .any(|(entity, attribute)| entity == token || attribute == token)
            || self.profiles.contains_key(token)
    }
}

impl DialogueKnowledge for ScriptedKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        let mut found: Vec<String> = self
            .entities
            .iter()
            .filter(|entity| text.contains(entity.as_str()))
            .cloned()
            .collect();
        found.sort_by_key(|entity| std::cmp::Reverse(entity.chars().count()));
        found
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.known.iter().any(|known| known == entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        self.answers.get(question).cloned()
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        self.attributes.get(entity).cloned().unwrap_or_default()
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.values
            .get(&(entity.to_string(), attribute.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn describe(&self, entity: &str) -> Option<String> {
        self.profiles.get(entity).cloned()
    }

    fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        self.nominations
            .get(&(entity.to_string(), attribute.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn recovery_clues(&self, entity: &str) -> Vec<String> {
        self.clues.get(entity).cloned().unwrap_or_default()
    }

    fn engine_fault(&self) -> Option<String> {
        self.fault.clone()
    }
}

fn scripted() -> ScriptedKnowledge {
    ScriptedKnowledge::default()
        .with_entity("姚明", &["身高", "国籍", "出生地"])
        .with_entity("叶莉", &["身高", "国籍"])
        .with_answer("姚明是谁", "姚明是中国篮球运动员。")
        .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
        .with_value("姚明", "国籍", "中国")
        .with_value("姚明", "出生地", "上海")
        .with_value("叶莉", "身高", "190")
        .with_profile("姚明", "姚明是中国篮球运动员。")
}

fn ask(
    manager: &mut DialogueManager,
    knowledge: &dyn DialogueKnowledge,
    input: &str,
) -> DialogueReply {
    manager
        .plan_pure_dialogue(input, knowledge)
        .or_else(|| manager.plan_reply(input, knowledge))
        .unwrap_or_else(|| panic!("输入「{}」未被对话管理器接管", input))
}

/// transcript 回放：产出「策略 + 文本」序列；未接管则记 `[defer]`。
fn replay(
    manager: &mut DialogueManager,
    knowledge: &dyn DialogueKnowledge,
    inputs: &[&str],
) -> Vec<String> {
    inputs
        .iter()
        .map(|input| {
            let reply = manager
                .plan_pure_dialogue(input, knowledge)
                .or_else(|| manager.plan_reply(input, knowledge));
            match reply {
                Some(reply) => format!("[{}] {}", reply.plan.join("+"), reply.text),
                None => format!("[defer] {}", input.trim()),
            }
        })
        .collect()
}

/// 提取回答中所有「…」引号词条，供 C4 溯源审计。
fn quoted_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('「') {
        let after = &rest[start + '「'.len_utf8()..];
        let Some(end) = after.find('」') else { break };
        tokens.push(after[..end].to_string());
        rest = &after[end + '」'.len_utf8()..];
    }
    tokens
}

const TRANSCRIPT: [&str; 6] = [
    "姚明是谁",
    "他的身高呢",
    "还有呢",
    "我们聊了什么？",
    "你怎么知道的？",
    "详细一点",
];

#[test]
fn c1_focus_inherited_across_turns() {
    let kb = scripted();
    let mut manager = DialogueManager::new();
    ask(&mut manager, &kb, "姚明是谁");
    assert_eq!(manager.focus_entity(), Some("姚明"));
    ask(&mut manager, &kb, "叶莉是谁");
    assert_eq!(
        manager.focus_entity(),
        Some("叶莉"),
        "最近提及的实体应成为焦点"
    );
    let stack = manager.focus_stack();
    assert_eq!(stack.last().map(String::as_str), Some("叶莉"));
    assert!(
        stack.contains(&"姚明".to_string()),
        "旧话题应保留在焦点栈：{:?}",
        stack
    );

    let reply = ask(&mut manager, &kb, "身高呢");
    assert!(
        reply.plan.iter().any(|plan| plan == "clarify_entity"),
        "多焦点省略问句应触发消歧：{:?}",
        reply.plan
    );
    assert!(reply.text.contains("姚明") && reply.text.contains("叶莉"));
}

#[test]
fn c2_pronoun_and_ellipsis_resolution() {
    let kb = scripted();
    let mut manager = DialogueManager::new();
    ask(&mut manager, &kb, "姚明是谁");

    let frame = manager.analyze("他的身高呢", &kb);
    assert_eq!(frame.kind, MoveKind::AskAttribute);
    assert_eq!(frame.entity.as_deref(), Some("姚明"));
    assert!(frame.inherited, "指代消解应标记 inherited");
    let reply = ask(&mut manager, &kb, "他的身高呢");
    assert!(reply.text.contains("226"), "{}", reply.text);

    let frame = manager.analyze("它呢", &kb);
    assert_eq!(frame.kind, MoveKind::AskIdentity);
    assert!(frame.inherited);
    assert_eq!(frame.entity.as_deref(), Some("姚明"));

    let frame = manager.analyze("国籍呢", &kb);
    assert_eq!(frame.kind, MoveKind::AskAttribute);
    assert_eq!(frame.entity.as_deref(), Some("姚明"));
    assert_eq!(frame.attribute.as_deref(), Some("国籍"));
    let reply = ask(&mut manager, &kb, "国籍呢");
    assert!(reply.text.contains("中国"), "{}", reply.text);

    // 话题切换：最近焦点优先；显式实体的省略问句应规范化为可命中查询
    ask(&mut manager, &kb, "叶莉是谁");
    let frame = manager.analyze("她呢", &kb);
    assert_eq!(frame.kind, MoveKind::AskIdentity);
    assert_eq!(frame.entity.as_deref(), Some("叶莉"), "指代应指向最近焦点");
    assert!(frame.inherited);
    let frame = manager.analyze("那姚明呢", &kb);
    assert_eq!(frame.kind, MoveKind::AskIdentity);
    assert_eq!(frame.entity.as_deref(), Some("姚明"));
    assert_eq!(frame.question.as_deref(), Some("姚明是什么"));
    let reply = ask(&mut manager, &kb, "那姚明呢");
    assert!(reply.text.contains("姚明"), "{}", reply.text);
}

#[test]
fn c3_compositional_generation() {
    let kb = scripted();
    let mut manager = DialogueManager::new();
    let reply = ask(&mut manager, &kb, "姚明的身高是多少");
    assert!(
        reply.plan.iter().any(|plan| plan == "direct_answer"),
        "{:?}",
        reply.plan
    );
    assert!(
        reply.plan.iter().any(|plan| plan == "offer_related"),
        "追问建议应由策略推导：{:?}",
        reply.plan
    );
    assert!(
        reply.plan.len() >= 2,
        "回答应由多个片段策略组装：{:?}",
        reply.plan
    );
    assert!(reply.text.contains("226"));
    assert!(
        reply.text.contains("国籍") || reply.text.contains("出生地"),
        "应推荐未问过的属性：{}",
        reply.text
    );
    assert!(
        reply.text.lines().count() >= 2,
        "结论与追问片段应拼接成多行回答：{}",
        reply.text
    );

    let ignorance = ask(&mut manager, &kb, "姚明的体重是多少");
    assert!(ignorance.plan.iter().any(|plan| plan == "admit_ignorance"));
    assert!(ignorance.plan.iter().any(|plan| plan == "ask_back"));
    assert!(ignorance.text.contains("没有记录"));
    assert!(
        ignorance.text.contains("国籍") && ignorance.text.contains("出生地"),
        "坦白时应推荐尚未问过的属性：{}",
        ignorance.text
    );
}

#[test]
fn c4_traceability_closure() {
    let kb = scripted();
    let mut manager = DialogueManager::new();
    let mut history: Vec<String> = Vec::new();
    let inputs = [
        "姚明的体重是多少",
        "姚明的国籍呢",
        "还有呢",
        "详细一点",
        "我们聊了什么？",
        "你怎么知道的？",
    ];
    for input in inputs {
        history.push(input.to_string());
        let reply = ask(&mut manager, &kb, input);
        for token in quoted_tokens(&reply.text) {
            let from_kb = kb.traceable(&token);
            let from_history = history.iter().any(|line| line.contains(&token));
            assert!(
                from_kb || from_history,
                "回答「{}」引用了无出处的词条「{}」",
                reply.text,
                token
            );
        }
        history.push(reply.text.clone());
    }
}

#[test]
fn c5_no_cross_scenario_hallucination() {
    let kb = ScriptedKnowledge::default()
        .with_entity("诸葛亮", &["出生地"])
        .with_value("诸葛亮", "出生地", "琅琊");
    let mut manager = DialogueManager::new();

    // 未知实体：让位给其它管线，且不得污染焦点
    assert!(manager.plan_reply("姚明的身高是多少", &kb).is_none());
    assert!(manager.plan_pure_dialogue("姚明是谁", &kb).is_none());
    assert_eq!(manager.turn_count(), 0);
    assert_eq!(manager.focus_entity(), None, "未知实体不得进入焦点栈");

    // 已知实体 + 未知属性：坦白不知道，不编造值、不泄漏哨兵
    let reply = manager
        .plan_reply("诸葛亮的身高是多少", &kb)
        .expect("已知实体应生成坦白回复");
    assert!(
        reply.plan.iter().any(|plan| plan == "admit_ignorance"),
        "{:?}",
        reply.plan
    );
    assert!(
        !reply.text.contains("姚明"),
        "不得泄漏其它场景的实体：{}",
        reply.text
    );
    assert!(
        !reply.text.contains("226"),
        "不得泄漏其它场景的取值：{}",
        reply.text
    );
    assert!(!reply.text.contains("190"));
}

#[test]
fn c6_meta_dialogue() {
    let kb = scripted();
    let mut manager = DialogueManager::new();
    ask(&mut manager, &kb, "姚明的身高是多少");

    let recap = ask(&mut manager, &kb, "我们聊了什么？");
    assert!(recap.text.contains("聊了 1 轮"), "{}", recap.text);
    assert!(recap.text.contains("姚明"));

    let why = ask(&mut manager, &kb, "你怎么知道的？");
    assert!(why.text.contains("依据"));
    assert!(why.text.contains("知识图谱"));

    let cont = ask(&mut manager, &kb, "还有呢");
    assert!(
        cont.plan.iter().any(|plan| plan == "continue_topic"),
        "{:?}",
        cont.plan
    );
    assert!(cont.text.contains("姚明"));

    let elaborate = ask(&mut manager, &kb, "详细一点");
    assert!(elaborate.text.contains("姚明"));
    assert!(
        elaborate.text.contains('：'),
        "展开回答应列出条目：{}",
        elaborate.text
    );

    let greeting = ask(&mut manager, &kb, "你好");
    assert!(
        greeting.text.contains("姚明"),
        "寒暄应带上当前话题：{}",
        greeting.text
    );
}

#[test]
fn c7_state_sufficiency_snapshot_restore() {
    let kb = scripted();
    let split = 3;

    let mut full = DialogueManager::new();
    let full_outputs = replay(&mut full, &kb, &TRANSCRIPT);

    let mut partial = DialogueManager::new();
    let prefix = replay(&mut partial, &kb, &TRANSCRIPT[..split]);
    assert_eq!(prefix, full_outputs[..split]);
    let snapshot = partial.snapshot();

    let mut restored = DialogueManager::new();
    restored.restore(&snapshot);
    let suffix = replay(&mut restored, &kb, &TRANSCRIPT[split..]);
    assert_eq!(
        suffix,
        full_outputs[split..],
        "恢复后继续必须与不中断运行逐字一致"
    );
    assert_eq!(restored.snapshot(), full.snapshot());
    assert_eq!(
        restored.summary(),
        full.summary(),
        "摘要应覆盖全部可观察状态"
    );

    // 经磁盘持久化同样满足状态充分性
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("dialogue.manager.json");
    let path_str = path.to_string_lossy().to_string();
    partial.save_to_path(&path_str).expect("快照应可保存");
    let mut from_disk = DialogueManager::new();
    assert!(from_disk.restore_from_path(&path_str));
    assert_eq!(from_disk.snapshot(), snapshot);
    let disk_suffix = replay(&mut from_disk, &kb, &TRANSCRIPT[split..]);
    assert_eq!(disk_suffix, suffix);
}

#[test]
fn c8_deterministic_and_idempotent_replay() {
    let kb = scripted();
    let mut first = DialogueManager::new();
    let mut second = DialogueManager::new();
    let out_a = replay(&mut first, &kb, &TRANSCRIPT);
    let out_b = replay(&mut second, &kb, &TRANSCRIPT);
    assert_eq!(out_a, out_b, "同一 transcript 两次回放应完全一致");
    assert_eq!(first.snapshot(), second.snapshot());
    assert_eq!(first.summary(), second.summary());
}

/// C10 编年史持久性：transcript 逐轮落盘为哈希链档案，
/// 重启后可校验、可续写、可检索；同 transcript 两份档案逐字节一致。
/// 这是「真相源」语义：快照只是检查点，档案才是对话本身。
#[test]
fn c10_chronicle_transcript_is_durable_and_deterministic() {
    fn run_into(path: &str, inputs: &[&str], kb: &ScriptedKnowledge) -> Vec<String> {
        let mut manager = DialogueManager::new();
        let (mut chronicle, audit) = Chronicle::open(path, FsyncPolicy::EachAppend).unwrap();
        assert!(audit.is_clean());
        let mut outputs = Vec::new();
        for input in inputs {
            // 写前日志：输入先落盘（含当轮开始时的焦点），再处理
            chronicle
                .append_turn(input, "turn", manager.focus_entity(), manager.turn_count())
                .unwrap();
            let reply = ask(&mut manager, kb, input);
            chronicle
                .append_reply(&reply.text, &reply.plan, &[])
                .unwrap();
            outputs.push(format!("[{}] {}", reply.plan.join("+"), reply.text));
        }
        outputs
    }

    let kb = scripted();
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let path_a = dir_a.path().join("a.jsonl").to_string_lossy().to_string();
    let path_b = dir_b.path().join("b.jsonl").to_string_lossy().to_string();
    let out_a = run_into(&path_a, &TRANSCRIPT, &kb);
    let out_b = run_into(&path_b, &TRANSCRIPT, &kb);
    assert_eq!(out_a, out_b, "回放输出应确定");
    assert_eq!(
        std::fs::read(&path_a).unwrap(),
        std::fs::read(&path_b).unwrap(),
        "同 transcript 的两份档案必须逐字节一致（seq 是语义时戳）"
    );

    // 重启：重新打开后链完整、seq 连续，可继续追加
    let (mut reopened, audit) = Chronicle::open(&path_a, FsyncPolicy::EachAppend).unwrap();
    assert!(audit.is_clean(), "{}", audit.verdict());
    assert_eq!(reopened.len(), TRANSCRIPT.len() * 2);
    let focus_at_split = reopened
        .entries()
        .iter()
        .find(|entry| entry.kind == "turn" && entry.seq == 5)
        .and_then(|entry| entry.payload.get("focus"))
        .and_then(|focus| focus.as_str())
        .map(str::to_string);
    assert_eq!(
        focus_at_split.as_deref(),
        Some("姚明"),
        "写前日志记录的是当轮开始时的焦点（第 5 轮时话题仍在姚明）"
    );
    reopened
        .append_turn("继续", "turn", Some("姚明"), 6)
        .unwrap();
    let (reloaded, audit) = Chronicle::open(&path_a, FsyncPolicy::None).unwrap();
    assert!(audit.is_clean());
    assert_eq!(reloaded.last_seq(), (TRANSCRIPT.len() * 2 + 1) as u64);
    assert!(!reloaded.search("姚明").is_empty(), "历史轮次应可检索");
}

/// C9 谓词归一：换说法/换方向不换事实。
///
/// 图谱事实以「师傅」为规范槽位存储（导入按谓词画像自动统一方向）。
/// 本维度约束：别名问句（老师/学生）与互逆问句（徒弟、谁是…的师傅）
/// 与规范问句（师傅）必须命中同一批事实、给出方向一致的答案。
#[test]
fn c9_predicate_normalization_alias_and_inverse() {
    use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};

    // 知识：李四是张三的师傅 ⇒ 规范槽位事实 师傅(张三, 李四)；
    // 实体快照（真实导入会伴随 name 字面量，供按名检索与消歧）
    let mut properties = HashMap::new();
    properties.insert("国籍".to_string(), "中国".to_string());
    let facts = vec![
        PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        },
        PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人物".to_string(),
            properties: properties.clone(),
        },
        PrologFact::Entity {
            name: "李四".to_string(),
            entity_type: "人物".to_string(),
            properties,
        },
    ];
    let config = ReasoningConfig {
        enable_link_prediction: false,
        ..ReasoningConfig::default()
    };
    let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
    reasoner.observe_facts(&facts);

    // 谓词画像：师傅(规范) ↔ 徒弟(互逆)，别名 老师；徒弟别名 学生
    let mut registry = PredicateRegistry::new();
    let mut master = PredicateProfileConfig::default();
    master.name = "师傅".to_string();
    master.aliases = vec!["老师".to_string()];
    master.inverse = Some("徒弟".to_string());
    assert!(registry.declare(&master));
    let mut disciple = PredicateProfileConfig::default();
    disciple.name = "徒弟".to_string();
    disciple.aliases = vec!["学生".to_string()];
    disciple.inverse = Some("师傅".to_string());
    // 徒弟已被师傅的互逆声明折入同一槽位：声明并入槽位画像（无新建画像）
    registry.declare(&disciple);
    // 与主循环 sync_predicate_pairs 相同的同步方式
    for (surface, slot, inverted) in registry.alias_pairs() {
        if surface == slot {
            continue;
        } else if inverted {
            reasoner.register_predicate_query_surface(&surface, &slot);
        } else {
            reasoner.register_predicate_alias(&surface, &slot);
        }
    }

    let canonical_answer = reasoner
        .answer("张三的师傅是谁")
        .expect("规范谓词问句应作答");
    assert!(canonical_answer.contains("李四"), "{}", canonical_answer);

    // 换说法：别名「老师」命中同一槽位事实
    let alias_answer = reasoner.answer("张三的老师是谁").expect("别名问句应作答");
    assert!(alias_answer.contains("李四"), "{}", alias_answer);

    // 换方向：互逆词面「徒弟」与它的别名「学生」按槽位反方向命中同一事实
    let inverse_answer = reasoner.answer("李四的徒弟是谁").expect("互逆问句应作答");
    assert!(inverse_answer.contains("张三"), "{}", inverse_answer);
    let inverse_alias = reasoner.answer("李四的学生是谁").expect("互逆别名应作答");
    assert!(inverse_alias.contains("张三"), "{}", inverse_alias);

    // 反查主体：「谁是李四的徒弟」= 张三的师傅是李四 ⇒ 答张三
    let find_subject = reasoner.answer("谁是李四的徒弟").expect("反查应作答");
    assert!(find_subject.contains("张三"), "{}", find_subject);

    // 计数与存在：李四有徒弟吗 / 李四有几个徒弟
    let exist_answer = reasoner.answer("李四有徒弟吗").expect("存在句应作答");
    assert!(
        exist_answer.contains("是的") && exist_answer.contains("张三"),
        "{}",
        exist_answer
    );
    let count_answer = reasoner.answer("李四有几个徒弟").expect("计数应作答");
    assert!(
        count_answer.contains('1') && count_answer.contains("张三"),
        "{}",
        count_answer
    );

    // 判定校验：「张三的徒弟是李四吗」应得否定并给出正确值（李四的徒弟≠李四…）
    // 规范校验：李四的徒弟是张三 ⇔ 事实 师傅(张三, 李四)
    let check = reasoner.answer("李四的徒弟是张三吗").expect("校验句应作答");
    assert!(check.contains("是的"), "{}", check);

    // 别名/互逆问句不得回答出「没有记录」以外的幻觉事实
    let unknown = reasoner
        .answer("李四的老师是谁")
        .expect("已登记谓词的未知命中应坦白");
    assert!(
        unknown.contains("没有记录") || unknown.contains("没有"),
        "{}",
        unknown
    );

    // 对话管理器路径同样成立：目光从规范到别名到互逆，答案事实一致
    let mut manager = DialogueManager::new();
    let reply = manager
        .plan_reply("张三的师傅是谁", &reasoner)
        .expect("对话管理器应接管规范问句");
    assert!(reply.text.contains("李四"), "{}", reply.text);
    let swapped = manager
        .plan_reply("李四的徒弟是谁", &reasoner)
        .expect("对话管理器应接管互逆问句");
    assert!(swapped.text.contains("张三"), "{}", swapped.text);
}

/// 往返一致性：`render(generate(fact))` 应被中文解析器还原为同一实体/谓词/值，
/// 生成器与解析器互为 oracle，零人工标注；再次渲染应为不动点。
/// 覆盖动词化模板的全部谓词面 + 无模板的中文谓词回退。
#[test]
fn roundtrip_generator_parser_recovers_fact() {
    let mut generator = SymbolicGenerator::new(6, 0.3);
    let cases: [(&str, &str, &str); 16] = [
        ("knows", "甲", "乙"),
        ("age", "甲", "30"),
        ("capital", "甲", "乙"),
        ("author", "甲", "乙"),
        ("birthPlace", "甲", "乙"),
        ("founder", "甲", "乙"),
        ("memberOf", "甲", "乙"),
        ("locatedIn", "甲", "乙"),
        ("worksAt", "甲", "乙"),
        ("spouse", "甲", "乙"),
        ("child", "甲", "乙"),
        ("likes", "甲", "乙"),
        ("nationality", "甲", "乙"),
        ("occupation", "甲", "乙"),
        ("座右铭", "甲", "乙"),
        ("师傅", "甲", "乙"),
    ];
    for (predicate, subject, object) in cases {
        generator.observe_triple(subject, predicate, object);
        let sentence = generator.render_fact(subject, predicate, &[object.to_string()]);

        let outcome = parse_lessons(&sentence);
        assert!(
            outcome.errors.is_empty(),
            "谓词 {} 生成「{}」后解析失败：{:?}",
            predicate,
            sentence,
            outcome.errors
        );
        let lesson = outcome.lessons.first().expect("应解析出确定事实");
        assert_eq!(lesson.kind, LessonKind::DeterministicFact);
        let surface = lesson.head.predicate.to_string();
        let canonical = generator
            .predicate_surfaces()
            .get(&surface)
            .cloned()
            .unwrap_or_else(|| surface.clone());
        assert_eq!(
            canonical, predicate,
            "生成「{}」经解析后谓词应从「{}」还原为「{}」",
            sentence, surface, predicate
        );
        assert_eq!(lesson.head.terms[0].name(), subject, "主语应往返一致");
        assert_eq!(lesson.head.terms[1].name(), object, "宾语应往返一致");

        let rerendered = generator.render_fact(subject, &canonical, &[object.to_string()]);
        assert_eq!(rerendered, sentence, "解析后再渲染应为不动点");
    }
}

/// 全栈冒烟：真实 SymbolicReasoner 驱动的对话管理器应能作答、继承焦点、
/// 并对未知实体让位（不幻觉）。
#[test]
fn real_reasoner_stack_answers_and_defers() {
    let mut yao_properties = HashMap::new();
    yao_properties.insert("身高".to_string(), "226".to_string());
    yao_properties.insert("国籍".to_string(), "中国".to_string());
    let mut ye_properties = HashMap::new();
    ye_properties.insert("身高".to_string(), "190".to_string());
    let facts = vec![
        PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人物".to_string(),
            properties: yao_properties,
        },
        PrologFact::Entity {
            name: "叶莉".to_string(),
            entity_type: "人物".to_string(),
            properties: ye_properties,
        },
    ];
    let config = ReasoningConfig {
        enable_link_prediction: false,
        ..ReasoningConfig::default()
    };
    let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
    reasoner.observe_facts(&facts);

    let mut manager = DialogueManager::new();
    let reply = manager
        .plan_reply("姚明的身高是多少", &reasoner)
        .expect("真实推理器上已知实体应接管");
    assert!(reply.text.contains("226"), "{}", reply.text);
    assert_eq!(manager.focus_entity(), Some("姚明"));

    let followup = manager
        .plan_reply("他的国籍呢", &reasoner)
        .expect("指代消解后应可作答");
    assert!(followup.text.contains("中国"), "{}", followup.text);

    let profile = manager
        .plan_reply("姚明是谁", &reasoner)
        .expect("身份问句应回落到画像生成");
    assert!(profile.text.contains("姚明"));

    assert!(
        manager.plan_reply("诸葛亮的身高是多少", &reasoner).is_none(),
        "图谱外的实体应让位给其它管线"
    );

    // 教回系统生成过的句子：应记住同一条二元事实（生成→解析→记住 闭环）
    let report = reasoner.assert_nl_fact("姚明喜欢叶莉");
    assert!(report.contains("已记住"), "{}", report);
    assert!(report.contains("姚明喜欢叶莉"), "教回应还原动词化模板：{}", report);
}

/// C14 失败可命名：引擎故障必须显式说「我出错了」，绝不伪装成「没有记录」；
/// Observe 只记账不改文本；Off 原样输出、零副作用。
#[test]
fn c14_failures_are_named_and_never_masked() {
    // (a) On：故障优先、显式命名、可回溯、绝不出现「没有记录」式伪装
    let faulted = scripted().with_fault("Prolog 引擎故障：index out of bounds");
    let mut manager = DialogueManager::new();
    manager.set_recovery_mode(RecoveryMode::On);
    let reply = ask(&mut manager, &faulted, "姚明的身高是什么");
    assert!(
        reply.plan.contains(&"engine_fault".to_string()),
        "故障轮必须走 engine_fault 策略：{:?}",
        reply.plan
    );
    assert!(
        reply.text.contains("出错") || reply.text.contains("故障"),
        "故障必须被命名：{}",
        reply.text
    );
    assert!(
        !reply.text.contains("没有记录"),
        "引擎故障绝不能伪装成不知道：{}",
        reply.text
    );
    let notice = manager.last_recovery().expect("故障轮必须产生认识论视图");
    assert_eq!(notice.status, EpistemicStatus::EngineFault);
    assert!(notice.receipt_id.is_some(), "故障必须有回执 id");
    assert_eq!(manager.failure_ledger().open_count(), 1);

    // (b) Off：原样输出，且完全不记账（零副作用）
    let mut off = DialogueManager::new();
    let off_reply = ask(&mut off, &faulted, "姚明的身高是什么");
    assert!(
        off_reply.plan.contains(&"direct_answer".to_string()),
        "Off 模式应原样输出：{:?}",
        off_reply.plan
    );
    assert!(off.failure_ledger().is_empty(), "Off 模式不得记账");
    assert!(off.last_recovery().is_none());

    // (c) Observe：只记账不改文本，文本仍由常规管线回答
    let mut observe = DialogueManager::new();
    observe.set_recovery_mode(RecoveryMode::Observe);
    let observed = ask(&mut observe, &faulted, "姚明的身高是什么");
    assert!(observed.text.contains("226"), "{}", observed.text);
    assert_eq!(
        observe.last_recovery().map(|notice| notice.status),
        Some(EpistemicStatus::EngineFault),
        "Observe 模式故障必须入账"
    );

    // (d) 确无记录（Absent）与引擎故障（EngineFault）措辞必须可区分
    let kb = scripted();
    let mut absent = DialogueManager::new();
    absent.set_recovery_mode(RecoveryMode::On);
    let absent_reply = ask(&mut absent, &kb, "姚明的体重是多少");
    let absent_notice = absent.last_recovery().expect("缺记录轮必须产生视图");
    assert_eq!(absent_notice.status, EpistemicStatus::Absent);
    assert!(
        absent_reply.text.contains("确定") || absent_reply.text.contains("没有记录"),
        "缺记录应显式命名：{}",
        absent_reply.text
    );
    assert_ne!(
        absent_reply.text,
        reply.text,
        "「确定没有」与「引擎故障」的措辞绝不能同形"
    );
    assert!(
        absent.recovery_report().contains("absent"),
        "报告应给出状态码分布：{}",
        absent.recovery_report()
    );
}

/// C15 推测不幻觉：联想提名以「推测」呈现、必带证据与追问，
/// 用户裁定（不是）后回执关闭、推测作废、挂账清空。
#[test]
fn c15_speculation_is_evidence_backed_and_resolvable() {
    let kb = scripted()
        .with_nomination("姚明", "体重", "140公斤", 0.72, "链接预测：同位置中锋体重分布")
        .with_clue("姚明", "国籍");
    let mut manager = DialogueManager::new();
    manager.set_recovery_mode(RecoveryMode::On);

    let reply = ask(&mut manager, &kb, "姚明的体重是多少");
    assert!(reply.text.contains("140公斤"), "{}", reply.text);
    assert!(
        reply.text.contains("推测"),
        "提名必须以「推测」定性：{}",
        reply.text
    );
    assert!(
        reply.text.contains("依据") && reply.text.contains("链接预测"),
        "推测必须带证据链：{}",
        reply.text
    );
    let notice = manager.last_recovery().expect("推测轮必须产生视图");
    assert_eq!(notice.status, EpistemicStatus::Provisional);
    assert_eq!(notice.level, "L2");
    assert!(!notice.evidence.is_empty());
    assert!(notice.question.as_deref().is_some_and(|q| !q.is_empty()));
    assert_eq!(manager.recovery_state().pending.len(), 1, "推测应挂账待确认");

    // 用户否认：回执关闭、摘要改变、挂账清空、承诺不再引用
    let open_before = manager.failure_ledger().digest();
    let refute = ask(&mut manager, &kb, "不是");
    assert!(
        refute.text.contains("作废") || refute.text.contains("不会再"),
        "{}",
        refute.text
    );
    assert_eq!(manager.failure_ledger().open_count(), 0);
    assert_ne!(open_before, manager.failure_ledger().digest());
    assert!(manager.recovery_state().pending.is_empty());
    assert_eq!(
        manager.last_recovery().map(|notice| notice.status),
        Some(EpistemicStatus::Provisional)
    );
}

/// C16 恢复确定性与有界：同一 transcript 两次回放逐字一致、台账摘要一致；
/// 澄清挂账有 TTL（3 轮）不会无限滞留。
#[test]
fn c16_recovery_is_deterministic_and_bounded() {
    let kb = scripted()
        .with_nomination("姚明", "体重", "140公斤", 0.72, "链接预测：分布证据")
        .with_clue("姚明", "国籍");
    let run = || {
        let mut manager = DialogueManager::new();
        manager.set_recovery_mode(RecoveryMode::On);
        let outputs = replay(
            &mut manager,
            &kb,
            &[
                "姚明的体重是多少", // L2 推测 + 挂账
                "不是",             // 裁定否认、关闭回执
                "姚明的体重是多少", // 身份已结：允许新回执（新 id）
                "我们聊了什么？",
            ],
        );
        (
            outputs.join("\n"),
            manager.failure_ledger().digest(),
            manager.failure_ledger().len(),
        )
    };
    let first = run();
    let second = run();
    assert_eq!(first.0, second.0, "恢复回放必须逐字一致");
    assert_eq!(first.1, second.1, "台账摘要必须一致");
    assert_eq!(first.2, second.2);
    assert_eq!(first.2, 2, "否认后同身份应可再次入账（新回执）");

    // TTL：澄清挂账 3 轮未获回答即过期（无挂钟，按逻辑轮次）
    let mut manager = DialogueManager::new();
    manager.set_recovery_mode(RecoveryMode::On);
    ask(&mut manager, &kb, "叶莉的体重是多少");
    assert_eq!(manager.recovery_state().pending.len(), 1, "应挂一个澄清问题");
    for _ in 0..3 {
        ask(&mut manager, &kb, "叶莉的身高呢");
    }
    assert!(
        manager.recovery_state().pending.is_empty(),
        "澄清挂账应在 TTL 后过期，不留无限追问"
    );
}


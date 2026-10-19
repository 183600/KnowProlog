//! QDF：质量差分模糊测试（Quality-Delta Fuzzing）。
//!
//! 目标不是随机敲字符，而是做**受控世界变异**：
//! 1. `mutate → measure`：注入已知质量缺陷，护照的目标维度必须按预期方向移动；
//! 2. `measure → speak`：同一缺陷必须被翻译成正确的话语行为
//!    （披露冲突 / 追问缺口 / 说明无来源 / 类型澄清 / 不再复读已删事实）；
//! 3. `baseline → invariance`：同义、错字、代词、省略、标点变异不得改变答案。
//!
//! 离线、纯符号、零存储 I/O：种子世界与谓词画像都在内存中构造，
//! 同一 `seed` 两次运行得到逐字节一致的报告。失败用例本身就是最小复现
//! （每个用例只含一次世界变异 + 一个问题），可直接作为回归资产。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::dialogue_manager::{
    DialogueKnowledge, DialogueManager, DialogueReply, MoveKind,
};
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::quality::{build_report, ClaimWarrant, QualityIndex, QualityReport};
use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystem, TypeSystemConfig};
use crate::{EntityTypeConfig, PropertyConfig};
use std::collections::HashMap;

/// 模糊测试配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzConfig {
    pub cases: usize,
    pub seed: u64,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            cases: 96,
            seed: 20261019,
        }
    }
}

/// 世界变异（每个用例只施加一个，预言由构造给出）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualityMutation {
    Baseline,
    RemoveRequired,
    AddConflict,
    RemoveSource,
    RemoveType,
    AddDisjointType,
    RemoveEdge,
}

impl QualityMutation {
    pub fn label(self) -> &'static str {
        match self {
            QualityMutation::Baseline => "baseline",
            QualityMutation::RemoveRequired => "remove_required",
            QualityMutation::AddConflict => "add_conflict",
            QualityMutation::RemoveSource => "remove_source",
            QualityMutation::RemoveType => "remove_type",
            QualityMutation::AddDisjointType => "add_disjoint_type",
            QualityMutation::RemoveEdge => "remove_edge",
        }
    }
}

/// 问句族（覆盖直接问、标点、同义、错字、代词、省略、身份问）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuestionClass {
    Direct,
    Punctuation,
    Synonym,
    Typo,
    Pronoun,
    Ellipsis,
    Identity,
}

impl QuestionClass {
    pub fn label(self) -> &'static str {
        match self {
            QuestionClass::Direct => "direct",
            QuestionClass::Punctuation => "punctuation",
            QuestionClass::Synonym => "synonym",
            QuestionClass::Typo => "typo",
            QuestionClass::Pronoun => "pronoun",
            QuestionClass::Ellipsis => "ellipsis",
            QuestionClass::Identity => "identity",
        }
    }

    fn is_attribute(self) -> bool {
        !matches!(self, QuestionClass::Identity)
    }

    fn setup(self) -> Option<&'static str> {
        match self {
            QuestionClass::Pronoun | QuestionClass::Ellipsis => Some("张三是谁"),
            _ => None,
        }
    }

    fn question(self) -> &'static str {
        match self {
            QuestionClass::Direct => "张三的师傅是谁",
            QuestionClass::Punctuation => "张三的师傅是谁？",
            QuestionClass::Synonym => "张三的师父是谁",
            QuestionClass::Typo => "张三的师付是谁",
            QuestionClass::Pronoun => "他的师傅是谁",
            QuestionClass::Ellipsis => "师傅呢",
            QuestionClass::Identity => "王五是谁",
        }
    }
}

/// 覆盖矩阵的一格。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageCell {
    pub mutation: String,
    pub question: String,
    pub cases: usize,
    pub failed: usize,
}

/// 一个可复现的最小失败。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzFailure {
    pub case: usize,
    pub mutation: String,
    pub question: String,
    pub input: String,
    pub contract: String,
    pub detail: String,
}

/// 模糊测试报告（确定性，可按 seed 复现）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzReport {
    pub seed: u64,
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub coverage: Vec<CoverageCell>,
    pub failures: Vec<FuzzFailure>,
    pub digest: String,
}

impl FuzzReport {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "QDF 质量差分模糊测试：种子 {}，用例 {}（通过 {} / 失败 {} 例），契约失败 {} 条，覆盖 {} 格。",
            self.seed,
            self.cases,
            self.passed,
            self.failed,
            self.failures.len(),
            self.coverage.len()
        )];
        if self.failures.is_empty() {
            lines.push(
                "  所有变异方向与话语契约通过：删事实不再复读、加冲突必须披露、删来源只改引用、缺属性只追问。"
                    .to_string(),
            );
        } else {
            lines.push("  失败用例（最小复现 = 变异 + 问题）：".to_string());
            for failure in self.failures.iter().take(12) {
                lines.push(format!(
                    "  [{}] {} / {}：{}（{}）",
                    failure.case,
                    failure.mutation,
                    failure.question,
                    failure.contract,
                    failure.detail
                ));
            }
            if self.failures.len() > 12 {
                lines.push(format!("  ……其余 {} 条省略", self.failures.len() - 12));
            }
        }
        lines.push(format!("摘要：{}", self.digest));
        lines.join("\n")
    }
}

/// 确定性 PRNG（xorshift64*；只用于轮转用例顺序，不追求密码学强度）。
#[derive(Debug, Clone)]
struct FuzzRng {
    state: u64,
}

impl FuzzRng {
    fn new(seed: u64) -> Self {
        FuzzRng {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

#[derive(Debug, Clone)]
struct World {
    facts: Vec<PrologFact>,
    registry: PredicateRegistry,
    types: TypeSystem,
}

impl World {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .facts
            .iter()
            .filter_map(|fact| match fact {
                PrologFact::Entity { name, .. } => Some(name.clone()),
                PrologFact::Relation { .. } => None,
            })
            .collect();
        for fact in &self.facts {
            if let PrologFact::Relation { from, to, .. } = fact {
                names.push(from.clone());
                names.push(to.clone());
            }
        }
        names.sort();
        names.dedup();
        names
    }

    fn slots(&self, entity: &str) -> BTreeMap<String, Vec<String>> {
        let mut slots: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for fact in &self.facts {
            match fact {
                PrologFact::Entity { name, properties, .. } if name == entity => {
                    for (key, value) in properties {
                        slots.entry(key.clone()).or_default().push(value.clone());
                    }
                }
                PrologFact::Relation { from, rel_type, to } if from == entity => {
                    slots.entry(rel_type.clone()).or_default().push(to.clone());
                }
                _ => {}
            }
        }
        for values in slots.values_mut() {
            values.sort();
            values.dedup();
        }
        slots
    }
}

fn property(name: &str, required: bool) -> PropertyConfig {
    PropertyConfig {
        name: name.to_string(),
        data_type: "string".to_string(),
        required,
        extraction_patterns: Vec::new(),
    }
}

fn entity_types() -> Vec<EntityTypeConfig> {
    vec![EntityTypeConfig {
        label: "Person".to_string(),
        properties: vec![property("name", true), property("age", false)],
        examples: vec!["人物".to_string()],
    }]
}

fn type_system() -> TypeSystem {
    TypeSystem::from_config(&TypeSystemConfig {
        closure_gate: true,
        aliases: vec![TypeAliasConfig {
            alias: "Person".to_string(),
            canonical: "人物".to_string(),
        }],
        subclasses: Vec::new(),
        disjoint: vec![TypeDisjointConfig {
            left: "人物".to_string(),
            right: "地点".to_string(),
        }],
    })
}

fn entity(name: &str, entity_type: &str, properties: &[(&str, &str)]) -> PrologFact {
    PrologFact::Entity {
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        properties: properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<HashMap<_, _>>(),
    }
}

fn relation(from: &str, rel_type: &str, to: &str) -> PrologFact {
    PrologFact::Relation {
        from: from.to_string(),
        rel_type: rel_type.to_string(),
        to: to.to_string(),
    }
}

fn base_facts() -> Vec<PrologFact> {
    vec![
        entity("张三", "Person", &[("name", "张三"), ("age", "30")]),
        entity("李四", "Person", &[("name", "李四")]),
        entity("王五", "Person", &[("name", "王五")]),
        relation("张三", "师傅", "李四"),
        // 无画像谓词：只提供连接度，不参与类型推断（用于制造未落型实体）
        relation("王五", "认识", "张三"),
    ]
}

/// 基础画像：无来源标签（`with_source=false`）或单来源。
fn base_registry(with_source: bool) -> PredicateRegistry {
    let mut registry = PredicateRegistry::new();
    registry.admit("师傅", "");
    for alias in ["师父", "老师", "师付"] {
        registry.register_alias(alias, "师傅", "");
    }
    if with_source {
        registry.observe("张三", "师傅", "李四", Some("知识图谱"));
    }
    registry
}

fn mutate(mutation: QualityMutation) -> World {
    let mut facts = base_facts();
    let mut registry = base_registry(true);
    match mutation {
        QualityMutation::Baseline => {}
        QualityMutation::RemoveRequired => {
            for fact in facts.iter_mut() {
                if let PrologFact::Entity {
                    name, properties, ..
                } = fact
                {
                    if name == "张三" {
                        properties.remove("name");
                    }
                }
            }
        }
        QualityMutation::AddConflict => {
            registry.note_conflict("张三", "师傅", "王五");
        }
        QualityMutation::RemoveSource => {
            registry = base_registry(false);
        }
        QualityMutation::RemoveType => {
            facts.retain(|fact| {
                !matches!(
                    fact,
                    PrologFact::Entity { name, .. } if name == "王五"
                )
            });
        }
        QualityMutation::AddDisjointType => {
            facts.push(entity("张三", "地点", &[]));
        }
        QualityMutation::RemoveEdge => {
            facts.retain(|fact| {
                !matches!(
                    fact,
                    PrologFact::Relation { from, rel_type, to }
                        if from == "张三" && rel_type == "师傅" && to == "李四"
                )
            });
        }
    }
    World {
        facts,
        registry,
        types: type_system(),
    }
}

fn report_for(world: &World) -> QualityReport {
    build_report(
        &world.facts,
        &world.registry,
        &world.types,
        &entity_types(),
    )
}

fn index_for(world: &World, revision: u64) -> QualityIndex {
    let report = report_for(world);
    QualityIndex::from_report(&report, &world.facts, &world.registry).with_revision(revision)
}

/// 测试替身：回答完全由内存世界决定，确保「删事实后绝不返回旧值」可判定。
struct FuzzKnowledge {
    world: World,
    aliases: BTreeMap<String, String>,
}

impl FuzzKnowledge {
    fn new(world: World) -> Self {
        let mut aliases = BTreeMap::new();
        aliases.insert("师父".to_string(), "师傅".to_string());
        aliases.insert("老师".to_string(), "师傅".to_string());
        aliases.insert("师付".to_string(), "师傅".to_string());
        FuzzKnowledge { world, aliases }
    }

    fn canonical_attribute(&self, raw: &str) -> String {
        self.aliases
            .get(raw)
            .cloned()
            .unwrap_or_else(|| raw.to_string())
    }

    fn answer_attribute(&self, entity: &str, raw_attribute: &str) -> Option<String> {
        let attribute = self.canonical_attribute(raw_attribute);
        let slots = self.world.slots(entity);
        let values = slots.get(&attribute)?;
        if values.is_empty() {
            return None;
        }
        Some(format!(
            "{}的{}是{}。",
            entity,
            attribute,
            values.join("、")
        ))
    }
}

impl DialogueKnowledge for FuzzKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .world
            .names()
            .into_iter()
            .filter(|name| text.contains(name.as_str()))
            .collect();
        names.sort_by_key(|name| std::cmp::Reverse(name.chars().count()));
        names
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.world.names().iter().any(|name| name == entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        let rest = question.strip_suffix("是什么")?;
        let entity = self
            .world
            .names()
            .into_iter()
            .find(|name| rest == name || rest.starts_with(&format!("{}的", name)))?;
        if rest == entity {
            return Some(format!("{}是一位记录中的实体。", entity));
        }
        let attribute = rest.strip_prefix(&format!("{}的", entity))?;
        self.answer_attribute(&entity, attribute)
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        self.world.slots(entity).keys().cloned().collect()
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.world
            .slots(entity)
            .get(&self.canonical_attribute(attribute))
            .cloned()
            .unwrap_or_default()
    }

    fn describe(&self, entity: &str) -> Option<String> {
        Some(format!("{}是一位记录中的实体。", entity))
    }

    fn traceable_token(&self, token: &str) -> bool {
        self.world.names().iter().any(|name| name == token)
    }
}

/// `mutate → measure`：目标维度必须移动，且无关维度不被误伤。
fn check_quality_delta(
    mutation: QualityMutation,
    before: &QualityReport,
    after: &QualityReport,
) -> Result<(), String> {
    let before_zhangsan = before
        .passport_for("张三")
        .ok_or_else(|| "基线缺少张三护照".to_string())?;
    let after_zhangsan = after
        .passport_for("张三")
        .ok_or_else(|| "变异后缺少张三护照（实体被隐藏）".to_string())?;
    match mutation {
        QualityMutation::Baseline => Ok(()),
        QualityMutation::RemoveRequired => {
            if after_zhangsan.completeness >= before_zhangsan.completeness {
                return Err(format!(
                    "删必备属性后完整性未下降：{:.3} → {:.3}",
                    before_zhangsan.completeness, after_zhangsan.completeness
                ));
            }
            if !after_zhangsan
                .gaps
                .iter()
                .any(|gap| gap.contains("缺少必备属性「name」"))
            {
                return Err("删必备属性后没有产生可行动缺口".to_string());
            }
            Ok(())
        }
        QualityMutation::AddConflict => {
            if after_zhangsan.consistency >= before_zhangsan.consistency {
                return Err(format!(
                    "加冲突后一致性未下降：{:.3} → {:.3}",
                    before_zhangsan.consistency, after_zhangsan.consistency
                ));
            }
            if after.summary.conflicts == 0 {
                return Err("加冲突后冲突计数仍为 0".to_string());
            }
            Ok(())
        }
        QualityMutation::RemoveSource => {
            if after_zhangsan.provenance >= before_zhangsan.provenance {
                return Err(format!(
                    "删来源后可溯源性未下降：{:.3} → {:.3}",
                    before_zhangsan.provenance, after_zhangsan.provenance
                ));
            }
            Ok(())
        }
        QualityMutation::RemoveType => {
            let after_wangwu = after
                .passport_for("王五")
                .ok_or_else(|| "删类型后王五护照被隐藏".to_string())?;
            if !after_wangwu.types.is_empty() {
                return Err(format!("删类型后仍有类型：{:?}", after_wangwu.types));
            }
            if !after.summary.clarify_candidates.contains(&"王五".to_string()) {
                return Err("未落型实体没有进入澄清候选".to_string());
            }
            Ok(())
        }
        QualityMutation::AddDisjointType => {
            if after_zhangsan.consistency >= before_zhangsan.consistency {
                return Err(format!(
                    "加互斥类型后一致性未下降：{:.3} → {:.3}",
                    before_zhangsan.consistency, after_zhangsan.consistency
                ));
            }
            Ok(())
        }
        QualityMutation::RemoveEdge => {
            if after_zhangsan.connectivity >= before_zhangsan.connectivity {
                return Err(format!(
                    "删关系后连通性未下降：{:.3} → {:.3}",
                    before_zhangsan.connectivity, after_zhangsan.connectivity
                ));
            }
            Ok(())
        }
    }
}

fn ask(
    manager: &mut DialogueManager,
    knowledge: &dyn DialogueKnowledge,
    input: &str,
) -> Option<DialogueReply> {
    manager
        .plan_pure_dialogue(input, knowledge)
        .or_else(|| manager.plan_reply(input, knowledge))
}

/// `measure → speak` 与端到端契约。
fn check_speech(
    mutation: QualityMutation,
    class: QuestionClass,
    reply: Option<&DialogueReply>,
) -> Result<(), String> {
    let Some(reply) = reply else {
        return Err("已知实体的问题未被对话管理器接管（返回 None）".to_string());
    };
    let text = &reply.text;
    let plans = &reply.plan;
    let attribute_question = class.is_attribute();
    match mutation {
        QualityMutation::Baseline => {
            if attribute_question {
                if !text.contains("李四") {
                    return Err(format!("保义问法丢失答案：{}", text));
                }
                if !plans.iter().any(|plan| plan == "quality_cite_source") {
                    return Err("有来源却没有引用行为".to_string());
                }
            } else if !text.contains("王五") {
                return Err(format!("身份问法丢失焦点实体：{}", text));
            }
            Ok(())
        }
        QualityMutation::RemoveSource => {
            if attribute_question {
                if !text.contains("李四") {
                    return Err(format!("删来源不应改变真值：{}", text));
                }
                if !plans.iter().any(|plan| plan == "quality_no_source") {
                    return Err("无来源却没有改为保守引用".to_string());
                }
                if !text.contains("尚未登记来源") {
                    return Err(format!("无来源未在话语中说明：{}", text));
                }
            }
            Ok(())
        }
        QualityMutation::RemoveRequired => {
            if attribute_question {
                if !text.contains("李四") {
                    return Err(format!("缺必备属性时答案被误伤：{}", text));
                }
                if !plans.iter().any(|plan| plan == "quality_offer_gap") {
                    return Err("缺必备属性后没有追问缺口".to_string());
                }
            }
            Ok(())
        }
        QualityMutation::AddConflict => {
            if attribute_question {
                if !plans
                    .iter()
                    .any(|plan| plan == "quality_disclose_conflict")
                {
                    return Err("有冲突却没有披露行为".to_string());
                }
                if !(text.contains("李四") && text.contains("王五")) {
                    return Err(format!(
                        "冲突披露未同时给出已有值与另一种说法：{}",
                        text
                    ));
                }
            }
            Ok(())
        }
        QualityMutation::RemoveType => {
            if !attribute_question
                && !plans.iter().any(|plan| plan == "quality_clarify_type")
            {
                return Err(format!("未落型身份问句没有转类型澄清：{}", text));
            }
            Ok(())
        }
        QualityMutation::AddDisjointType => {
            if attribute_question {
                if !plans.iter().any(|plan| plan == "quality_offer_gap") {
                    return Err("互斥类型共存没有进入缺口话语".to_string());
                }
                if !text.contains("互斥类型") {
                    return Err(format!("互斥类型未披露：{}", text));
                }
            }
            Ok(())
        }
        QualityMutation::RemoveEdge => {
            if attribute_question {
                if text.contains("李四") {
                    return Err(format!("已删除事实仍被复读：{}", text));
                }
                if !plans.iter().any(|plan| plan == "admit_ignorance") {
                    return Err("已删除事实没有走坦白路径".to_string());
                }
            }
            Ok(())
        }
    }
}

fn run_case(
    case: usize,
    mutation: QualityMutation,
    class: QuestionClass,
    baseline: &QualityReport,
) -> Result<(), Vec<FuzzFailure>> {
    let world = mutate(mutation);
    let report = report_for(&world);
    let mut failures = Vec::new();
    if let Err(detail) = check_quality_delta(mutation, baseline, &report) {
        failures.push(FuzzFailure {
            case,
            mutation: mutation.label().to_string(),
            question: class.label().to_string(),
            input: class.question().to_string(),
            contract: "quality_delta".to_string(),
            detail,
        });
    }

    let knowledge = FuzzKnowledge::new(world);
    let mut manager = DialogueManager::new();
    manager.set_quality_index(Some(index_for(&knowledge.world, case as u64 + 1)));
    if let Some(setup) = class.setup() {
        let _ = ask(&mut manager, &knowledge, setup);
    }
    let reply = ask(&mut manager, &knowledge, class.question());
    if let Err(detail) = check_speech(mutation, class, reply.as_ref()) {
        failures.push(FuzzFailure {
            case,
            mutation: mutation.label().to_string(),
            question: class.label().to_string(),
            input: class.question().to_string(),
            contract: "speech_contract".to_string(),
            detail,
        });
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// 运行离线 QDF。无 I/O、无网络；同 seed 逐字节可复现。
pub fn run(config: &FuzzConfig) -> FuzzReport {
    const MUTATIONS: [QualityMutation; 7] = [
        QualityMutation::Baseline,
        QualityMutation::RemoveRequired,
        QualityMutation::AddConflict,
        QualityMutation::RemoveSource,
        QualityMutation::RemoveType,
        QualityMutation::AddDisjointType,
        QualityMutation::RemoveEdge,
    ];
    const QUESTIONS: [QuestionClass; 7] = [
        QuestionClass::Direct,
        QuestionClass::Punctuation,
        QuestionClass::Synonym,
        QuestionClass::Typo,
        QuestionClass::Pronoun,
        QuestionClass::Ellipsis,
        QuestionClass::Identity,
    ];
    let cells = MUTATIONS.len() * QUESTIONS.len();
    let cases = config.cases.max(1);
    let baseline = report_for(&mutate(QualityMutation::Baseline));
    let mut rng = FuzzRng::new(config.seed);
    let offset = rng.below(cells);
    let mut failures = Vec::new();
    let mut coverage: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();
    for case in 0..cases {
        let cell = (case + offset) % cells;
        let mutation = MUTATIONS[cell % MUTATIONS.len()];
        let question = QUESTIONS[cell / MUTATIONS.len()];
        let entry = coverage
            .entry((mutation.label().to_string(), question.label().to_string()))
            .or_insert((0, 0));
        entry.0 += 1;
        if let Err(case_failures) = run_case(case, mutation, question, &baseline) {
            entry.1 += case_failures.len();
            failures.extend(case_failures);
        }
    }
    let coverage: Vec<CoverageCell> = coverage
        .into_iter()
        .map(|((mutation, question), (cases, failed))| CoverageCell {
            mutation,
            question,
            cases,
            failed,
        })
        .collect();
    let failed = failures
        .iter()
        .map(|failure| failure.case)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let passed = cases.saturating_sub(failed);
    let digest = {
        let value = serde_json::json!({
            "seed": config.seed,
            "coverage": &coverage,
            "failures": &failures,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    };
    FuzzReport {
        seed: config.seed,
        cases,
        passed,
        failed,
        coverage,
        failures,
        digest,
    }
}

/// 只读诊断：给定世界与问题，返回本轮担保（供 CLI/测试检查，不参与模糊测试判定）。
pub fn warrant_probe(
    facts: &[PrologFact],
    registry: &PredicateRegistry,
    types: &TypeSystem,
    entity_types: &[EntityTypeConfig],
    revision: u64,
    entity: &str,
    attribute: Option<&str>,
) -> Option<ClaimWarrant> {
    let report = build_report(facts, registry, types, entity_types);
    let index = QualityIndex::from_report(&report, facts, registry).with_revision(revision);
    index.warrant_for(entity, attribute)
}

/// 让外部检查器知道一次问句是否走了知识问句路径。
pub fn is_attribute_reply(kind: &MoveKind) -> bool {
    matches!(kind, MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_byte_for_byte_deterministic() {
        let config = FuzzConfig {
            cases: 64,
            seed: 42,
        };
        let first = run(&config);
        let second = run(&config);
        assert_eq!(first, second);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.failed, 0, "{:?}", first.failures);
    }

    #[test]
    fn full_grid_covers_every_mutation_and_question() {
        let report = run(&FuzzConfig {
            cases: 49,
            seed: 7,
        });
        assert_eq!(report.coverage.len(), 49);
        assert!(report.coverage.iter().all(|cell| cell.cases >= 1));
        assert_eq!(report.failed, 0, "{:?}", report.failures);
    }

    #[test]
    fn mutation_sensitivity_is_visible() {
        let baseline = report_for(&mutate(QualityMutation::Baseline));
        for mutation in [
            QualityMutation::RemoveRequired,
            QualityMutation::AddConflict,
            QualityMutation::RemoveSource,
            QualityMutation::RemoveType,
            QualityMutation::AddDisjointType,
            QualityMutation::RemoveEdge,
        ] {
            let world = mutate(mutation);
            let report = report_for(&world);
            check_quality_delta(mutation, &baseline, &report)
                .unwrap_or_else(|error| panic!("{}：{}", mutation.label(), error));
        }
    }

    #[test]
    fn review_never_repeats_deleted_fact_as_current() {
        let base = mutate(QualityMutation::Baseline);
        let knowledge = FuzzKnowledge::new(base.clone());
        let mut manager = DialogueManager::new();
        manager.set_quality_index(Some(index_for(&base, 1)));
        ask(&mut manager, &knowledge, "张三的师傅是谁").expect("首轮应回答");
        let stale = mutate(QualityMutation::RemoveEdge);
        let current = FuzzKnowledge::new(stale);
        manager.set_quality_index(Some(index_for(&current.world, 2)));
        let review = ask(&mut manager, &current, "你确定吗").expect("复核应被接管");
        assert!(
            review.plan.iter().any(|plan| plan == "review_stale"),
            "{:?}",
            review.plan
        );
        assert!(
            review.text.contains("不再作为当前事实"),
            "{}",
            review.text
        );
    }

    #[test]
    fn review_discloses_new_conflict_and_confirms_version_advance() {
        let base = mutate(QualityMutation::Baseline);
        let knowledge = FuzzKnowledge::new(base.clone());
        let mut manager = DialogueManager::new();
        manager.set_quality_index(Some(index_for(&base, 1)));
        ask(&mut manager, &knowledge, "张三的师傅是谁").expect("首轮应回答");

        let conflicted = mutate(QualityMutation::AddConflict);
        let current = FuzzKnowledge::new(conflicted);
        manager.set_quality_index(Some(index_for(&current.world, 2)));
        let review = ask(&mut manager, &current, "复核一下").expect("复核应被接管");
        assert!(
            review.plan.iter().any(|plan| plan == "review_conflict"),
            "{:?}",
            review.plan
        );
        assert!(review.text.contains("王五"), "{}", review.text);

        let fresh = FuzzKnowledge::new(base.clone());
        manager.set_quality_index(Some(index_for(&base, 3)));
        let review = ask(&mut manager, &fresh, "现在还成立吗").expect("复核应被接管");
        assert!(
            review.plan.iter().any(|plan| plan == "review_confirmed"),
            "{:?}",
            review.plan
        );
        assert!(review.text.contains("从 1 推进到 3"), "{}", review.text);
    }

    #[test]
    fn review_discloses_replaced_conflict_even_when_count_is_unchanged() {
        let base = mutate(QualityMutation::Baseline);
        let knowledge = FuzzKnowledge::new(base.clone());
        let mut manager = DialogueManager::new();
        manager.set_quality_index(Some(index_for(&base, 1)));
        ask(&mut manager, &knowledge, "张三的师傅是谁").expect("首轮应回答");

        let old = index_for(&mutate(QualityMutation::AddConflict), 1)
            .warrant_for("张三", Some("师傅"))
            .expect("旧担保应在");
        let mut snapshot = manager.snapshot();
        snapshot.turns.last_mut().expect("应有上一轮").warrant = Some(old);
        manager.restore(&snapshot);

        let mut replaced = mutate(QualityMutation::Baseline);
        replaced.registry.note_conflict("张三", "师傅", "赵六");
        let current = FuzzKnowledge::new(replaced);
        manager.set_quality_index(Some(index_for(&current.world, 2)));
        let review = ask(&mut manager, &current, "你确定吗").expect("复核应被接管");
        assert!(
            review.plan.iter().any(|plan| plan == "review_conflict"),
            "{:?}",
            review.plan
        );
        assert!(review.text.contains("赵六"), "{}", review.text);
        assert!(!review.text.contains("王五"), "{}", review.text);
    }

    #[test]
    fn claim_warrant_probe_keeps_truth_and_source_separate() {
        let world = mutate(QualityMutation::RemoveSource);
        let warrant = warrant_probe(
            &world.facts,
            &world.registry,
            &world.types,
            &entity_types(),
            3,
            "张三",
            Some("师傅"),
        )
        .unwrap();
        assert!(!warrant.has_source());
        assert_eq!(warrant.known_values, vec!["李四".to_string()]);
        assert_eq!(warrant.revision, 3);
        assert!(warrant.citation_note().contains("未登记来源"));
    }
}

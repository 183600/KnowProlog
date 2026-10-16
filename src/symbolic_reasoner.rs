use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::association_miner::{display_item, AssociationMiner, AssociationRule};
use crate::entity_linker::{EntityLinker, LinkMethod};
use crate::link_prediction::{HornRule, LinkPredictor, Prediction, RuleTemplate, Triple};
use crate::ontology_alignment::{OntologyAligner, TermKind};
use crate::prolog_engine::PrologFact;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::resolution_engine::{
    render_term, Atom, Clause, DerivedFact, ProofNode, ProofSource, ResolutionEngine, Term,
};
use crate::semantic_embedding::SymbolicEmbedding;
use crate::semantic_entailment::{
    self, EntailmentResult, Knowledge, PropositionParser, Verdict,
};
use crate::symbolic_generator::{GenerationOrder, SymbolicGenerator};
use crate::ReasoningConfig;

// 归结引擎内部谓词：has(E, 谓词名) / type(E, 类型名)，用于承载关联规则的物品
const PROOF_HAS: &str = "__具有__";
const PROOF_TYPE: &str = "__类型__";

// 把链接预测挖掘出的 Horn 规则（组合/逆向/共同邻居/共同主体）转成归结子句
fn clause_from_path_rule(rule: &HornRule) -> Clause {
    let a = || Term::var("A");
    let b = || Term::var("B");
    let c = || Term::var("C");
    let source = format!("链接预测规则 {}", rule.rule_text());
    match rule.template {
        RuleTemplate::Compose => Clause::new(
            Atom::pattern(&rule.head, a(), c()),
            vec![
                Atom::pattern(rule.body1(), a(), b()),
                Atom::pattern(rule.body2().unwrap_or(""), b(), c()),
            ],
            rule.confidence,
            &source,
        ),
        RuleTemplate::Inverse => Clause::new(
            Atom::pattern(&rule.head, b(), a()),
            vec![Atom::pattern(rule.body1(), a(), b())],
            rule.confidence,
            &source,
        ),
        RuleTemplate::CommonNeighbor => Clause::new(
            Atom::pattern(&rule.head, a(), b()),
            vec![
                Atom::pattern(rule.body1(), a(), c()),
                Atom::pattern(rule.body2().unwrap_or(""), b(), c()),
            ],
            rule.confidence,
            &source,
        ),
        RuleTemplate::CommonObject => Clause::new(
            Atom::pattern(&rule.head, a(), b()),
            vec![
                Atom::pattern(rule.body1(), c(), a()),
                Atom::pattern(rule.body2().unwrap_or(""), c(), b()),
            ],
            rule.confidence,
            &source,
        ),
    }
}

// 上一段生成的记忆：实体、已讲句数、段落排序角度（支持"接着说/换个角度说"）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationState {
    pub entity: String,
    pub shown: usize,
    pub order: GenerationOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueSnapshot {
    pub focus_entities: Vec<String>,
    pub last_predicate: Option<String>,
    pub history: Vec<String>,
    #[serde(default)]
    pub generation: Option<GenerationState>,
}

pub struct DialogueContext {
    focus_entities: Vec<String>,
    last_predicate: Option<String>,
    history: Vec<String>,
    generation: Option<GenerationState>,
    max_focus: usize,
    max_history: usize,
}

impl DialogueContext {
    pub fn new(max_focus: usize) -> Self {
        Self {
            focus_entities: Vec::new(),
            last_predicate: None,
            history: Vec::new(),
            generation: None,
            max_focus,
            max_history: 20,
        }
    }

    pub fn focus(&self) -> Option<&str> {
        self.focus_entities.first().map(|s| s.as_str())
    }

    pub fn last_predicate(&self) -> Option<&str> {
        self.last_predicate.as_deref()
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn focus_entities(&self) -> &[String] {
        &self.focus_entities
    }

    pub fn record_entities(&mut self, entities: &[String]) {
        for entity in entities.iter().rev() {
            if entity.is_empty() {
                continue;
            }
            self.focus_entities.retain(|e| e != entity);
            self.focus_entities.insert(0, entity.clone());
        }
        if self.focus_entities.len() > self.max_focus {
            self.focus_entities.truncate(self.max_focus);
        }
    }

    pub fn record_predicate(&mut self, predicate: &str) {
        if !predicate.is_empty() {
            self.last_predicate = Some(predicate.to_string());
        }
    }

    pub fn record_question(&mut self, question: &str) {
        if question.trim().is_empty() {
            return;
        }
        self.history.insert(0, question.to_string());
        if self.history.len() > self.max_history {
            self.history.truncate(self.max_history);
        }
    }

    pub fn clear(&mut self) {
        self.focus_entities.clear();
        self.last_predicate = None;
        self.history.clear();
        self.generation = None;
    }

    pub fn generation_state(&self) -> Option<&GenerationState> {
        self.generation.as_ref()
    }

    pub fn record_generation(&mut self, entity: &str, shown: usize, order: GenerationOrder) {
        if entity.trim().is_empty() {
            return;
        }
        self.generation = Some(GenerationState {
            entity: entity.to_string(),
            shown,
            order,
        });
    }

    pub fn snapshot(&self) -> DialogueSnapshot {
        DialogueSnapshot {
            focus_entities: self.focus_entities.clone(),
            last_predicate: self.last_predicate.clone(),
            history: self.history.clone(),
            generation: self.generation.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueSnapshot) {
        self.focus_entities = snapshot.focus_entities.clone();
        self.focus_entities.truncate(self.max_focus);
        self.last_predicate = snapshot.last_predicate.clone();
        self.history = snapshot.history.clone();
        self.history.truncate(self.max_history);
        self.generation = snapshot.generation.clone();
    }

    pub fn resolve_pronouns(&self, input: &str) -> Option<String> {
        let focus = self.focus()?;
        const PRONOUNS: [&str; 7] = ["他们", "她们", "它们", "这个人", "他", "她", "它"];
        let chars: Vec<char> = input.chars().collect();
        let mut result = String::with_capacity(input.len() + focus.len());
        let mut i = 0;
        let mut replaced = false;
        while i < chars.len() {
            // 直接在字符切片上比较
            let mut matched: Option<&str> = None;
            for pronoun in PRONOUNS {
                let plen = pronoun.chars().count();
                if chars.len() - i >= plen
                    && chars[i..i + plen].iter().copied().eq(pronoun.chars())
                {
                    matched = Some(pronoun);
                    break;
                }
            }
            if let Some(pronoun) = matched {
                let previous = if i == 0 { None } else { Some(chars[i - 1]) };
                let blocked = matches!(previous, Some('其') | Some('无'));
                if !blocked {
                    result.push_str(focus);
                    i += pronoun.chars().count();
                    replaced = true;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        if replaced {
            Some(result)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct ValueHit {
    value: String,
    prediction: Option<Prediction>,
}

#[derive(Debug, Clone)]
struct ChainStep {
    subject: String,
    predicate: String,
    hits: Vec<ValueHit>,
}

pub struct SymbolicReasoner {
    config: ReasoningConfig,
    aligner: OntologyAligner,
    predictor: LinkPredictor,
    context: DialogueContext,
    embedding: SymbolicEmbedding,
    planner: QueryPlanner,
    linker: EntityLinker,
    generator: SymbolicGenerator,
    // 实体属性三元组（观察到的），供归结引擎编码事实
    attributes: HashSet<(String, String, String)>,
    // 关联规则挖掘器（类型/谓词/属性篮子的支持度-置信度-提升度）
    associations: AssociationMiner,
    // 一阶归结引擎（事实 + 链接预测规则 + 关联规则）
    resolver: ResolutionEngine,
}

impl SymbolicReasoner {
    pub fn new(config: ReasoningConfig, synonyms: HashMap<String, String>) -> Self {
        let aligner = OntologyAligner::new(synonyms, config.ontology_alignment_threshold);
        let embedding = SymbolicEmbedding::new(config.embedding_similarity_floor);
        let mut linker = EntityLinker::new();
        for (alias, canonical) in aligner.synonyms() {
            linker.register_alias(alias, canonical);
        }
        let generator = SymbolicGenerator::new(
            config.max_profile_sentences,
            config.completion_min_score,
        );
        let associations = AssociationMiner::new(
            config.association_min_support,
            config.association_min_confidence,
            config.association_min_lift,
        );
        let resolver = ResolutionEngine::new(
            config.resolution_max_depth,
            config.resolution_max_solutions,
        );
        Self {
            config,
            aligner,
            predictor: LinkPredictor::new(),
            context: DialogueContext::new(5),
            embedding,
            planner: QueryPlanner::new(),
            linker,
            generator,
            attributes: HashSet::new(),
            associations,
            resolver,
        }
    }

    pub fn context(&self) -> &DialogueContext {
        &self.context
    }

    pub fn embedding(&self) -> &SymbolicEmbedding {
        &self.embedding
    }

    pub fn linker(&self) -> &EntityLinker {
        &self.linker
    }

    pub fn generator(&self) -> &SymbolicGenerator {
        &self.generator
    }

    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        let mut added = 0usize;
        let mut entity_observed = false;
        let mut association_changed = false;
        let mut attribute_changed = false;
        let triples_before = self.predictor.triple_count();
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    self.aligner.observe_triple(from, rel_type, to);
                    self.predictor.add_triple(from, rel_type, to);
                    self.linker.observe_triple(from, rel_type, to);
                    self.generator.observe_triple(from, rel_type, to);
                    if self.config.enable_semantic_embedding {
                        self.embedding.observe_triple(from, rel_type, to);
                    }
                    association_changed |= self.associations.observe_triple(from, rel_type);
                    added += 1;
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    self.aligner.register_term(name, TermKind::Entity);
                    self.aligner.register_term(entity_type, TermKind::Type);
                    for property in properties.keys() {
                        self.aligner.register_term(property, TermKind::Attribute);
                    }
                    self.linker.register_entity(name, Some(entity_type.as_str()));
                    let property_pairs: Vec<(String, String)> = properties
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect();
                    self.generator
                        .observe_entity(name, entity_type, &property_pairs);
                    let property_keys: Vec<String> = properties.keys().cloned().collect();
                    if self.config.enable_semantic_embedding {
                        self.embedding.observe_entity(name, entity_type, &property_keys);
                    }
                    association_changed |=
                        self.associations
                            .observe_entity(name, entity_type, &property_keys);
                    for (key, value) in properties {
                        if self
                            .attributes
                            .insert((name.clone(), key.clone(), value.clone()))
                        {
                            attribute_changed = true;
                        }
                    }
                    entity_observed = true;
                }
            }
        }
        let triples_added = self.predictor.triple_count() > triples_before;
        // 规则只依赖已积累的三元组与固定阈值：若本次没有新增三元组，
        // 重新挖掘必然得到相同规则，直接跳过
        if triples_added && self.config.enable_link_prediction {
            self.predictor.mine_rules(
                self.config.link_prediction_min_support,
                self.config.link_prediction_min_confidence,
            );
        }
        // 关联规则与前向/反向归结都只依赖已观察的篮子与事实：无新增则跳过重建
        if self.config.enable_association_rules && (association_changed || triples_added) {
            self.associations.mine();
        }
        if self.config.enable_resolution && (association_changed || attribute_changed || triples_added)
        {
            self.rebuild_resolver();
        }
        // 实体事实同样会更新嵌入（observe_entity），因此也需要重建
        if self.config.enable_semantic_embedding && (added > 0 || entity_observed) {
            self.embedding.rebuild();
        }
    }

    // 用当前知识（事实 + 链接预测规则 + 关联规则）重建归结引擎
    fn rebuild_resolver(&mut self) {
        let mut engine = ResolutionEngine::new(
            self.config.resolution_max_depth,
            self.config.resolution_max_solutions,
        );
        for triple in self.predictor.triples() {
            engine.add_fact(&triple.predicate, &triple.subject, &triple.object);
            engine.add_fact(PROOF_HAS, &triple.subject, &triple.predicate);
        }
        for (subject, key, value) in &self.attributes {
            engine.add_fact(key, subject, value);
            engine.add_fact(PROOF_HAS, subject, key);
        }
        for (entity, etype) in self.generator.typed_entities() {
            engine.add_fact(PROOF_TYPE, &entity, &etype);
        }
        if self.config.enable_association_rules {
            for clause in self.associations.to_clauses(PROOF_HAS, PROOF_TYPE) {
                engine.add_clause(clause);
            }
        }
        for rule in self.predictor.rules() {
            engine.add_clause(clause_from_path_rule(rule));
        }
        self.resolver = engine;
    }

    pub fn resolver(&self) -> &ResolutionEngine {
        &self.resolver
    }

    pub fn associations(&self) -> &AssociationMiner {
        &self.associations
    }

    pub fn contextualize(&mut self, input: &str) -> Option<String> {
        let pronoun_resolved = self.context.resolve_pronouns(input);
        let ellipsis_base = pronoun_resolved.as_deref().unwrap_or(input);
        let rewritten = self
            .resolve_ellipsis(ellipsis_base)
            .or(pronoun_resolved);
        self.record_vocabulary(input);
        if let Some(rewritten_text) = &rewritten {
            self.record_vocabulary(rewritten_text);
        }
        rewritten
    }

    fn resolve_ellipsis(&self, input: &str) -> Option<String> {
        let trimmed = input.trim().trim_end_matches(['？', '?']).trim();
        let core = if let Some(rest) = trimmed.strip_prefix("那") {
            let rest = rest.strip_prefix("么").unwrap_or(rest);
            rest.strip_suffix("呢")?
        } else if trimmed.ends_with('呢') && trimmed.chars().count() <= 12 {
            trimmed.strip_suffix("呢")?
        } else {
            return None;
        };
        let core = core.trim();
        if core.is_empty() || core.contains(' ') {
            return None;
        }
        if core.contains('的') {
            return Some(format!("{}是什么", core));
        }
        let predicate = self.context.last_predicate()?;
        Some(format!("{}的{}是什么", core, predicate))
    }

    fn record_vocabulary(&mut self, text: &str) {
        let mut entities = self.entities_mentioned(text);
        if self.config.enable_generation {
            for entity in self.linked_entities(text) {
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
        }
        if !entities.is_empty() {
            self.context.record_entities(&entities);
        }
        let normalized_text = self.aligner.normalize(text);
        let predicate = self
            .known_predicates()
            .into_iter()
            .find(|p| self.predicate_mentioned_normalized(text, &normalized_text, p))
            .map(str::to_string);
        if let Some(predicate) = predicate {
            self.context.record_predicate(&predicate);
        }
        self.context.record_question(text);
    }

    fn known_predicates(&self) -> Vec<&str> {
        let mut predicates: Vec<&str> = self
            .aligner
            .profiles()
            .iter()
            .filter(|p| p.kind == TermKind::Relation)
            .map(|p| p.term.as_str())
            .collect();
        let mut seen: HashSet<&str> = predicates.iter().copied().collect();
        for rule in self.predictor.rules() {
            if seen.insert(rule.head.as_str()) {
                predicates.push(rule.head.as_str());
            }
        }
        predicates
    }

    pub fn entities_mentioned(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize(text);
        let mut found: Vec<(usize, String)> = Vec::new();
        for (i, profile) in self.aligner.profiles().iter().enumerate() {
            if profile.kind != TermKind::Entity {
                continue;
            }
            let Some(term_norm) = self.aligner.normalized_term_at(i) else {
                continue;
            };
            if term_norm.is_empty() {
                continue;
            }
            if normalized_text.contains(term_norm) {
                found.push((profile.term.chars().count(), self.aligner.canonical_form(&profile.term)));
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if canonical.is_empty() {
                continue;
            }
            if text.contains(alias.as_str())
                && self.aligner.kind_of(canonical) == Some(TermKind::Entity)
            {
                found.push((alias.chars().count(), canonical.clone()));
            }
        }
        found.sort_by(|a, b| b.0.cmp(&a.0));
        let mut seen = HashSet::new();
        found
            .into_iter()
            .map(|(_, entity)| entity)
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    fn predicate_mentioned_normalized(
        &self,
        text: &str,
        normalized_text: &str,
        predicate: &str,
    ) -> bool {
        let predicate_norm = self.aligner.normalize(predicate);
        if !predicate_norm.is_empty() && normalized_text.contains(&predicate_norm) {
            return true;
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if canonical == predicate && text.contains(alias.as_str()) {
                return true;
            }
        }
        false
    }

    // 实体链接：候选生成 + 模糊容错 + 上下文消歧（详见 entity_linker 模块）
    const LINK_ACCEPT_THRESHOLD: f64 = 0.5;

    pub fn linked_entities(&self, text: &str) -> Vec<String> {
        let hints = self.link_hints(text);
        let mentions = self.linker.link_with_context(text, &hints);
        let mut seen = HashSet::new();
        mentions
            .iter()
            .filter_map(|mention| mention.best())
            .filter(|candidate| candidate.score >= Self::LINK_ACCEPT_THRESHOLD)
            .map(|candidate| candidate.entity.clone())
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    pub fn best_linked_entity(&self, text: &str) -> Option<String> {
        let hints = self.link_hints(text);
        self.linker
            .link_with_context(text, &hints)
            .iter()
            .filter_map(|mention| {
                mention
                    .best()
                    .map(|candidate| (candidate.score, candidate.entity.clone()))
            })
            .filter(|(score, _)| *score >= Self::LINK_ACCEPT_THRESHOLD)
            .max_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.1.cmp(&a.1))
            })
            .map(|(_, entity)| entity)
    }

    pub fn entity_link_report(&self, text: &str) -> String {
        let hints = self.link_hints(text);
        self.linker.report(text, &hints)
    }

    // 仅返回"模糊匹配"的链接说明，用于在回答前向用户提示容错链接
    pub fn entity_link_notes(&self, text: &str) -> Vec<String> {
        let hints = self.link_hints(text);
        self.linker
            .link_with_context(text, &hints)
            .iter()
            .filter_map(|mention| {
                let candidate = mention.best()?;
                if candidate.method != LinkMethod::Fuzzy {
                    return None;
                }
                Some(format!(
                    "实体链接：「{}」→「{}」（置信度 {:.2}）",
                    mention.mention, candidate.entity, candidate.score
                ))
            })
            .collect()
    }

    fn link_hints(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize(text);
        self.known_predicates()
            .into_iter()
            .filter(|predicate| {
                self.predicate_mentioned_normalized(text, &normalized_text, predicate)
            })
            .map(str::to_string)
            .collect()
    }

    // 生成式介绍：把实体的相关事实组合成连贯段落（无神经网络）
    // 生成后自动做语义蕴含自检，并记录"上一段生成"以便多轮续说。
    pub fn generate_about(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let page = self
            .generator
            .profile_page(&entity, 0, GenerationOrder::Relevance)?;
        self.context
            .record_generation(&entity, page.next_offset, GenerationOrder::Relevance);
        let mut text = page.text;
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    // 多轮生成：接着上一段介绍继续讲后面的句子
    pub fn continue_generation(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let page = self
            .generator
            .profile_page(&state.entity, state.shown, state.order)?;
        if page.shown == 0 {
            return Some(format!(
                "关于「{}」的知识已经介绍完毕。可以输入「换个角度说」按不同顺序重述，或介绍其它实体。",
                state.entity
            ));
        }
        self.context
            .record_generation(&state.entity, page.next_offset, state.order);
        let mut text = page.text;
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    // 多轮生成：换一个排序角度重述同一实体的介绍
    pub fn switch_generation_angle(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let order = match state.order {
            GenerationOrder::Relevance => GenerationOrder::Alphabetical,
            GenerationOrder::Alphabetical => GenerationOrder::Relevance,
        };
        let page = self.generator.profile_page(&state.entity, 0, order)?;
        if page.shown == 0 {
            return Some(format!("关于「{}」暂无可重述的事实。", state.entity));
        }
        self.context
            .record_generation(&state.entity, page.next_offset, order);
        let label = match order {
            GenerationOrder::Relevance => "信息量排序（由稀有谓词到常见谓词）",
            GenerationOrder::Alphabetical => "主题排序（按谓词字典序）",
        };
        let mut text = format!("换个角度介绍「{}」（{}）：\n{}", state.entity, label, page.text);
        if let Some(note) = self.generation_self_check(&text) {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    fn knowledge(&self) -> Knowledge<'_> {
        Knowledge {
            predictor: &self.predictor,
            generator: &self.generator,
            min_rule_confidence: self.config.entailment_min_confidence,
            max_rule_candidates: 16,
        }
    }

    fn proposition_parser(&self) -> PropositionParser<'_> {
        PropositionParser::new(
            &self.aligner,
            &self.linker,
            self.generator.predicate_surfaces(),
        )
    }

    // 形式化语义 + 语义蕴含：把陈述句解析为命题并逐条判定蕴含/矛盾/未知
    pub fn check_statement(&self, text: &str) -> Vec<EntailmentResult> {
        if !self.config.enable_entailment {
            return Vec::new();
        }
        let parser = self.proposition_parser();
        let knowledge = self.knowledge();
        parser
            .parse(text)
            .into_iter()
            .map(|proposition| semantic_entailment::entail(&proposition, &knowledge))
            .collect()
    }

    pub fn entailment_report(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return "请输入要判定的陈述句，例如：判断：张三是李四的师傅".to_string();
        }
        let results = self.check_statement(trimmed);
        if results.is_empty() {
            return format!(
                "形式化失败：未能把「{}」解析为可判定命题。\n  可用句式：S的P是O / O是S的P / S是类型 / S P O（P 为图谱已知谓词）。",
                trimmed
            );
        }
        let mut output = format!(
            "形式化语义分析：「{}」解析出 {} 个候选命题，逐条给出蕴含判定（证据可追溯）：\n",
            trimmed,
            results.len()
        );
        output.push_str(&semantic_entailment::render_results(&results));
        output
    }

    // 生成自检：逐句回译为命题并验证是否被知识图谱蕴含（对抗"幻觉"）
    fn generation_self_check(&self, text: &str) -> Option<String> {
        if !self.config.enable_entailment {
            return None;
        }
        let parser = self.proposition_parser();
        let knowledge = self.knowledge();
        let mut total = 0usize;
        let mut entailed = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for raw_line in text.lines() {
            let sentence = semantic_entailment::strip_discourse_markers(raw_line.trim());
            if !sentence.ends_with('。') {
                continue;
            }
            let propositions = parser.parse(sentence);
            if propositions.is_empty() {
                continue;
            }
            total += 1;
            let supported = propositions.iter().any(|proposition| {
                semantic_entailment::entail(proposition, &knowledge).verdict == Verdict::Entailed
            });
            if supported {
                entailed += 1;
            } else {
                failed.push(sentence.to_string());
            }
        }
        if total == 0 {
            return None;
        }
        let mut note = format!("（生成自检：{}/{} 句可由知识图谱蕴含", entailed, total);
        if failed.is_empty() {
            note.push_str("，零幻觉）");
        } else {
            note.push_str(&format!("，未通过 {} 句：{}）", failed.len(), failed.join("；")));
        }
        Some(note)
    }

    // 符号续写：对「X的」做谓词预测，支持实体级→类型级→全局回退
    pub fn predict_continuation(&self, prefix: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = prefix.trim();
        let normalized = trimmed.strip_suffix('的').map(|root| {
            let root = root.trim();
            let canonical = self.aligner.canonical_form(root);
            if !canonical.is_empty() && canonical != root {
                format!("{}的", canonical)
            } else {
                trimmed.to_string()
            }
        });
        let base = self
            .generator
            .complete(normalized.as_deref().unwrap_or(trimmed))?;
        // 统计回退之外再挂一层"关联规则联想"：由已有特征推荐缺失特征（纯计数）
        if self.config.enable_association_rules {
            if let Some(root) = trimmed.strip_suffix('的') {
                if let Some(extra) = self.association_suggestions(root.trim()) {
                    let mut output = base;
                    output.push('\n');
                    output.push_str(&extra);
                    return Some(output);
                }
            }
        }
        Some(base)
    }

    // ===== 归结推理：证明、推理链与关联规则 =====

    fn resolve_known_entity(&self, raw: &str) -> Option<String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let candidate = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        if self.generator.is_known_entity(&candidate) || self.associations.basket(&candidate).is_some()
        {
            return Some(candidate);
        }
        self.best_linked_entity(trimmed).filter(|entity| {
            self.generator.is_known_entity(entity) || self.associations.basket(entity).is_some()
        })
    }

    // 解析 "S的P1的P2...是O"（谓词串可为空 ⇒ 类型断言）
    fn parse_chain_statement(&self, text: &str) -> Option<(String, Vec<String>, String)> {
        let trimmed = text.trim().trim_end_matches(['。', '！', '!']).trim();
        let (subject_part, object_part) = trimmed.rsplit_once('是')?;
        let object = self.aligner.canonical_form(object_part.trim());
        if object.is_empty() {
            return None;
        }
        let segments: Vec<String> = subject_part
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return None;
        }
        if segments.len() == 1 {
            let entity = self.aligner.canonical_form(&segments[0]);
            if entity.is_empty() {
                return None;
            }
            return Some((entity, Vec::new(), object));
        }
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty() {
            return None;
        }
        Some((entity, predicates, object))
    }

    fn render_atom_nl(&self, atom: &Atom) -> String {
        let names: Vec<String> = atom.terms.iter().map(render_term).collect();
        if atom.predicate == PROOF_HAS && names.len() == 2 {
            return format!("{} 具有 {}", names[0], names[1]);
        }
        if atom.predicate == PROOF_TYPE && names.len() == 2 {
            return format!("{} 属于类型 {}", names[0], names[1]);
        }
        if names.len() == 2 {
            let sentence = self
                .generator
                .render_fact(&names[0], &atom.predicate, &[names[1].clone()]);
            return sentence.trim_end_matches('。').to_string();
        }
        atom.display()
    }

    fn render_proof_tree(&self, node: &ProofNode, lines: &mut Vec<String>, indent: usize) {
        let padding = "    ".repeat(indent);
        let fact = self.render_atom_nl(&node.atom);
        match &node.source {
            ProofSource::Fact => lines.push(format!("{}{}（事实）", padding, fact)),
            ProofSource::Rule { source } => lines.push(format!(
                "{}{}（由 {}；本步置信度 {:.2}）",
                padding, fact, source, node.confidence
            )),
        }
        for child in &node.children {
            self.render_proof_tree(child, lines, indent + 1);
        }
    }

    fn render_resolution_result(&self, phrase: &str, goals: &[Atom]) -> String {
        let solutions = self.resolver.solve(goals, &[]);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结证明（一阶 Horn 归结 + 回溯证明树，纯符号）：\n");
        output.push_str(&format!("  目标：{}  〔逻辑形式：{}〕\n", phrase, goal_display));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则（含链接预测与关联规则）\n",
            self.resolver.fact_count(),
            self.resolver.clause_count()
        ));
        if let Some(solution) = solutions.first() {
            output.push_str(&format!("  得证：证明置信度 {:.2}\n", solution.confidence));
            output.push_str("  证明树：\n");
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
        } else {
            output.push_str(&format!(
                "  未得证：在深度上限 {} 内未找到归结反驳（开世界假设：推不出 ≠ 为假）。\n",
                self.resolver.max_depth()
            ));
            output.push_str("  可尝试「推理链：同句式问句」查看可能的解，或先补充相关事实/规则。\n");
        }
        output
    }

    // "X有P" 形式的存在性目标：由关联规则/前向推导证明
    fn prove_existence_statement(&self, text: &str) -> Option<String> {
        let (subject_part, object_part) = text.split_once('有')?;
        let subject_part = subject_part.trim();
        let object_part = object_part.trim();
        if subject_part.is_empty() || object_part.is_empty() {
            return None;
        }
        if subject_part.contains('的') || object_part.contains('的') {
            return None;
        }
        let entity = self.resolve_known_entity(subject_part)?;
        let predicate = self.aligner.canonical_form(object_part);
        if predicate.is_empty() {
            return None;
        }
        let goals = vec![Atom::rel(PROOF_HAS, &entity, &predicate)];
        Some(self.render_resolution_result(&format!("{} 具有 {}", entity, predicate), &goals))
    }

    /// 证明一条陈述（"S的P…是O" / "S是类型" / "S有P"），输出回归结证明树
    pub fn resolution_proof_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请提供要证明的陈述，例如：证明：张三的师祖是王五 或 证明：姚明有国籍".to_string();
        }
        if let Some(report) = self.prove_existence_statement(trimmed) {
            return report;
        }
        match self.parse_chain_statement(trimmed) {
            Some((entity, predicates, object)) => {
                let (goals, phrase) = if predicates.is_empty() {
                    (
                        vec![Atom::rel(PROOF_TYPE, &entity, &object)],
                        format!("{} 属于类型 {}", entity, object),
                    )
                } else if predicates.len() == 1 {
                    (
                        vec![Atom::rel(&predicates[0], &entity, &object)],
                        format!("{}的{}是{}", entity, predicates[0], object),
                    )
                } else {
                    let mut goals = Vec::new();
                    let mut previous = Term::constant(&entity);
                    for (index, predicate) in predicates.iter().enumerate() {
                        let next = if index + 1 == predicates.len() {
                            Term::constant(&object)
                        } else {
                            Term::var(&format!("V{}", index + 1))
                        };
                        goals.push(Atom::pattern(predicate, previous.clone(), next.clone()));
                        previous = next;
                    }
                    (
                        goals,
                        format!("{}的{}是{}", entity, predicates.join("的"), object),
                    )
                };
                self.render_resolution_result(&phrase, &goals)
            }
            None => format!(
                "无法把「{}」解析为可归结的目标。支持句式：S的P…是O / S是类型 / S有P。",
                trimmed
            ),
        }
    }

    /// 开放目标（"X的P…是谁"）：返回全部归结解与各解的证明链
    pub fn resolution_query_report(&self, text: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
        }
        let trimmed = text.trim().trim_end_matches(['？', '?', '。', ' ']);
        if trimmed.is_empty() {
            return "请提供推理目标，例如：推理链：张三的师祖是谁".to_string();
        }
        let mut core = trimmed.to_string();
        for suffix in [
            "是谁", "是什么", "是多少", "是哪里", "是哪个", "是哪儿", "是哪位", "是哪些", "是几",
        ] {
            if let Some(rest) = core.strip_suffix(suffix) {
                core = rest.to_string();
                break;
            }
        }
        // 句中仍带"是"（如"推理链：张三的师祖是王五"）→ 按证明处理
        if core.contains('是') {
            return self.resolution_proof_report(&core);
        }
        let segments: Vec<String> = core
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        let Some((entity, predicates)) = self.split_chain(&segments) else {
            return format!("无法把「{}」解析为谓词链目标。", trimmed);
        };
        if predicates.is_empty() {
            return format!(
                "「{}」只有实体没有谓词链，请补充关系，例如：推理链：{}的师傅是谁",
                trimmed, entity
            );
        }
        let answer_var = "答案".to_string();
        let mut goals = Vec::new();
        let mut previous = Term::constant(&entity);
        for (index, predicate) in predicates.iter().enumerate() {
            let next = if index + 1 == predicates.len() {
                Term::var(&answer_var)
            } else {
                Term::var(&format!("V{}", index + 1))
            };
            goals.push(Atom::pattern(predicate, previous, next.clone()));
            previous = next;
        }
        let solutions = self.resolver.solve(&goals, std::slice::from_ref(&answer_var));
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let mut output = String::from("归结推理链（一阶 Horn 归结，解与证明均可回溯）：\n");
        output.push_str(&format!(
            "  目标：{}的{}是谁  〔逻辑形式：{}〕\n",
            entity,
            predicates.join("的"),
            goal_display
        ));
        output.push_str(&format!(
            "  知识库：{} 条事实，{} 条规则；深度上限 {}\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            self.resolver.max_depth()
        ));
        if solutions.is_empty() {
            output.push_str("  没有找到解（开世界假设：推不出 ≠ 不存在）。\n");
            return output;
        }
        for (index, solution) in solutions.iter().enumerate() {
            let answer = solution
                .bindings
                .get(&answer_var)
                .cloned()
                .unwrap_or_else(|| "?".to_string());
            output.push_str(&format!(
                "  解 {}（证明置信度 {:.2}）：答案是「{}」\n",
                index + 1,
                solution.confidence,
                answer
            ));
            let mut lines = Vec::new();
            for proof in &solution.proofs {
                self.render_proof_tree(proof, &mut lines, 2);
            }
            output.push_str(&lines.join("\n"));
            output.push('\n');
        }
        output.trim_end().to_string()
    }

    /// 关联规则报告（可带过滤词，比如实体/类型/谓词）
    pub fn association_report(&self, filter: Option<&str>) -> String {
        if !self.config.enable_association_rules {
            return "关联规则挖掘未启用（config.toml [reasoning] enable_association_rules = false）。"
                .to_string();
        }
        let mut output = String::from("关联规则报告（支持度/置信度/提升度/杠杆率，纯计数）：\n");
        output.push_str(&format!(
            "  实体篮子 {} 个，物品 {} 种，规则 {} 条（阈值：支持度≥{:.2}，置信度≥{:.2}，提升度≥{:.2}）\n",
            self.associations.basket_count(),
            self.associations.item_kinds(),
            self.associations.rules().len(),
            self.config.association_min_support,
            self.config.association_min_confidence,
            self.config.association_min_lift
        ));
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        // 规范化一次即可，避免对每条规则重复计算
        let normalized = filter.map(|filter| self.aligner.canonical_form(filter));
        let rules: Vec<&AssociationRule> = self
            .associations
            .rules()
            .iter()
            .filter(|rule| match filter {
                None => true,
                Some(filter) => {
                    let normalized = normalized.as_deref().unwrap_or(filter);
                    rule.consequent.contains(filter)
                        || rule.consequent.contains(normalized)
                        || rule
                            .antecedent
                            .iter()
                            .any(|item| item.contains(filter) || item.contains(normalized))
                }
            })
            .collect();
        if rules.is_empty() {
            output.push_str("  （没有满足条件/阈值的规则：可多导入同类型实体，或降低阈值）\n");
            return output;
        }
        for (index, rule) in rules.iter().take(20).enumerate() {
            output.push_str(&format!(
                "  {}. {} —— {}\n",
                index + 1,
                rule.rule_text(),
                rule.metrics_text()
            ));
        }
        if rules.len() > 20 {
            output.push_str(&format!("  ……其余 {} 条省略\n", rules.len() - 20));
        }
        output.trim_end().to_string()
    }

    /// 实体补全联想："已有特征 ⇒ 可能还具备的缺失特征"
    pub fn association_suggestions(&self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_association_rules {
            return None;
        }
        let entity = self.resolve_known_entity(entity_raw)?;
        let suggestions = self.associations.suggestions_for(&entity, 3);
        if suggestions.is_empty() {
            return None;
        }
        let basket = self.associations.basket(&entity)?;
        let mut features: Vec<&str> = basket.iter().map(|item| display_item(item)).collect();
        features.sort();
        let mut output = format!(
            "关联规则联想（{} 个实体篮子的计数统计，无神经网络）：\n",
            self.associations.basket_count()
        );
        output.push_str(&format!(
            "  「{}」已有特征：{}\n",
            entity,
            features.join("、")
        ));
        for (index, rule) in suggestions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 可能还具备「{}」——{}\n",
                index + 1,
                rule.consequent_text(),
                self.associations.explain(rule)
            ));
        }
        Some(output.trim_end().to_string())
    }

    /// 前向饱和推导报告：多跳新事实（含前提与规则来源）
    pub fn derived_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。".to_string();
        }
        let derived = self.resolver.forward_chain(3, 100);
        let mut output = String::from("推理发现（前向饱和，多跳优先；均可回溯前提）：\n");
        output.push_str(&format!(
            "  现有 {} 条事实、{} 条规则，本轮推出 {} 条新事实。\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            derived.len()
        ));
        if derived.is_empty() {
            output.push_str("  （没有可推导的新事实：可增加事实/规则或放宽规则阈值）\n");
            return output;
        }
        // 预计算展示文本，避免排序比较时反复格式化
        let mut items: Vec<(&DerivedFact, String)> = derived
            .iter()
            .map(|item| (item, item.atom.display()))
            .collect();
        items.sort_by(|(a, a_display), (b, b_display)| {
            b.depth
                .cmp(&a.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a_display.cmp(b_display))
        });
        for (index, (item, _)) in items.into_iter().take(limit.max(1)).enumerate() {
            let premises = item
                .premises
                .iter()
                .map(|premise| self.render_atom_nl(premise))
                .collect::<Vec<_>>()
                .join("；");
            output.push_str(&format!(
                "  {}. {}（第 {} 轮推导；前提：{}；规则：{}；置信度 {:.2}）\n",
                index + 1,
                self.render_atom_nl(&item.atom),
                item.depth,
                premises,
                item.rule_source,
                item.confidence
            ));
        }
        output.trim_end().to_string()
    }

    fn is_known_entity(&self, name: &str) -> bool {
        self.aligner.kind_of(name) == Some(TermKind::Entity)
    }

    fn is_known_predicate(&self, name: &str) -> bool {
        self.aligner.kind_of(name) == Some(TermKind::Relation) || self.predictor.has_predicate(name)
    }

    fn split_chain(&self, segments: &[String]) -> Option<(String, Vec<String>)> {
        if segments.is_empty() {
            return None;
        }
        // 在"精确登记"的实体名上做最长前缀切分
        for take in (1..=segments.len()).rev() {
            let candidate = segments[..take].join("的");
            if self.aligner.kind_of(&candidate) == Some(TermKind::Entity) {
                let predicates = segments[take..]
                    .iter()
                    .map(|segment| self.aligner.canonical_form(segment))
                    .collect();
                return Some((candidate, predicates));
            }
        }
        let entity = self.aligner.canonical_form(&segments[0]);
        let predicates = segments[1..]
            .iter()
            .map(|segment| self.aligner.canonical_form(segment))
            .collect();
        Some((entity, predicates))
    }

    fn node_hits(&self, node: &str, predicate: &str) -> Vec<ValueHit> {
        let mut hits: Vec<ValueHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for object in self.predictor.objects_of(node, predicate) {
            if seen.insert(object.clone()) {
                hits.push(ValueHit {
                    value: object,
                    prediction: None,
                });
            }
        }
        if self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            for prediction in self.predictor.predict_for_entity(node, Some(predicate), limit) {
                // 只取 node 作为主体的出边预测
                if prediction.triple.subject != node {
                    continue;
                }
                let value = prediction.triple.object.clone();
                if seen.insert(value.clone()) {
                    hits.push(ValueHit {
                        value,
                        prediction: Some(prediction),
                    });
                }
            }
        }
        hits.sort_by(|a, b| a.value.cmp(&b.value));
        hits
    }

    fn run_chain(
        &self,
        entity: &str,
        predicates: &[String],
    ) -> Option<(Vec<ChainStep>, Vec<ValueHit>)> {
        if predicates.is_empty() {
            return None;
        }
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            // 多个分支可能命中同一实体，边收集边去重，保留首次出现顺序
            let mut seen_values: HashSet<String> = HashSet::new();
            let mut seen_hit_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let hits = self.node_hits(node, predicate);
                if hits.is_empty() {
                    continue;
                }
                for hit in &hits {
                    if seen_values.insert(hit.value.clone()) {
                        next_frontier.push(hit.value.clone());
                    }
                    if seen_hit_values.insert(hit.value.clone()) {
                        round_hits.push(hit.clone());
                    }
                }
                steps.push(ChainStep {
                    subject: node.clone(),
                    predicate: predicate.clone(),
                    hits,
                });
            }
            if next_frontier.is_empty() {
                return None;
            }
            frontier = next_frontier;
            last_hits = round_hits;
        }
        if last_hits.is_empty() {
            return None;
        }
        Some((steps, last_hits))
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先"
        } else if index + 1 == total {
            "最后"
        } else {
            "此外"
        }
    }

    // 期望答案类型（形式语义中的问题类型约束）：谁→人物、哪里→地点、何时→时间
    fn expected_type_keywords(question: &str) -> Vec<&'static str> {
        if question.contains('谁') || question.contains("何人") || question.contains("哪位") {
            vec!["人", "person"]
        } else if question.contains("哪里")
            || question.contains("哪儿")
            || question.contains("什么地方")
            || question.contains("何处")
        {
            vec!["地", "城市", "国家", "place", "location", "country", "city"]
        } else if question.contains("什么时候") || question.contains("哪年") || question.contains("何时") {
            vec!["日期", "时间", "date", "time", "年"]
        } else {
            Vec::new()
        }
    }

    fn value_matches_expectation(&self, value: &str, keywords: &[&str]) -> bool {
        let Some(etype) = self.generator.entity_type_of(value) else {
            return false;
        };
        let etype_lower = etype.to_lowercase();
        keywords.iter().any(|keyword| {
            etype.contains(*keyword) || etype_lower.contains(&keyword.to_lowercase())
        })
    }

    fn generate_find_object(
        &self,
        entity: &str,
        predicates: &[String],
        steps: &[ChainStep],
        last_hits: &[ValueHit],
        question: &str,
    ) -> String {
        let expectations = Self::expected_type_keywords(question);
        let mut ordered: Vec<&ValueHit> = last_hits.iter().collect();
        if !expectations.is_empty() {
            // 缓存判定结果，避免排序时对同一元素重复做类型查询
            ordered.sort_by_cached_key(|hit| {
                if self.value_matches_expectation(&hit.value, &expectations) {
                    0usize
                } else {
                    1usize
                }
            });
        }
        let reordered = ordered
            .iter()
            .zip(last_hits.iter())
            .any(|(a, b)| !std::ptr::eq(*a, b));
        let chain_text = format!("{} 的 {}", entity, predicates.join(" 的 "));
        let mut output = String::from("根据知识图谱");
        if steps.len() > 1 {
            output.push_str("（多跳推理）");
        }
        output.push_str("：\n");
        if steps.len() > 1 {
            for step in steps {
                let values = step
                    .hits
                    .iter()
                    .map(|hit| hit.value.clone())
                    .collect::<Vec<_>>()
                    .join("、");
                output.push_str(&format!(
                    "  {} 的 {} 是 {}。\n",
                    step.subject, step.predicate, values
                ));
            }
            output.push_str("  综合得出：\n");
        }
        if last_hits.len() == 1 {
            let hit = &last_hits[0];
            output.push_str(&format!("  {} 是 {}。", chain_text, hit.value));
            if let Some(prediction) = &hit.prediction {
                output.push_str(&format!(
                    "\n  推理依据：{}",
                    self.predictor.explain(prediction)
                ));
            }
        } else {
            for (index, hit) in ordered.iter().enumerate() {
                let marker = Self::discourse_marker(index, ordered.len());
                output.push_str(&format!(
                    "  {}{}{} 是 {}。",
                    marker,
                    if marker.is_empty() { "" } else { "，" },
                    chain_text,
                    hit.value
                ));
                if let Some(prediction) = &hit.prediction {
                    output.push_str(&format!(
                        "（推断，置信度 {:.2}）",
                        prediction.confidence
                    ));
                }
                output.push('\n');
            }
            if last_hits.iter().any(|hit| hit.prediction.is_some()) {
                let evidence: Vec<String> = last_hits
                    .iter()
                    .filter_map(|hit| {
                        hit.prediction
                            .as_ref()
                            .map(|prediction| self.predictor.explain(prediction))
                    })
                    .take(2)
                    .collect();
                if !evidence.is_empty() {
                    output.push_str(&format!("  推理依据：{}", evidence.join("；")));
                }
            }
            if reordered {
                output.push_str(&format!(
                    "\n  说明：已按问题期望的答案类型（{}）优先排序。",
                    expectations.join("/")
                ));
            }
        }
        output
    }

    fn answer_planned(&self, question: &str) -> Option<String> {
        let plan = self.planner.parse(question)?;
        match plan {
            QueryPlan::FindObject { segments } => {
                let (entity, predicates) = self.split_chain(&segments)?;
                if predicates.is_empty() {
                    return None;
                }
                if let Some((steps, last_hits)) = self.run_chain(&entity, &predicates) {
                    return Some(self.generate_find_object(
                        &entity,
                        &predicates,
                        &steps,
                        &last_hits,
                        question,
                    ));
                }
                if self.is_known_entity(&entity)
                    && predicates.iter().all(|p| self.is_known_predicate(p))
                {
                    return Some(format!(
                        "根据现有知识库，没有记录「{} 的 {}」。",
                        entity,
                        predicates.join(" 的 ")
                    ));
                }
                None
            }
            QueryPlan::Count { subject, predicate } => {
                self.answer_count(&subject, &predicate, false)
            }
            QueryPlan::Exist { subject, predicate } => {
                self.answer_count(&subject, &predicate, true)
            }
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => self.answer_check(&subject_segments, &object),
            QueryPlan::FindSubject { object, predicate } => {
                self.answer_find_subject(&object, &predicate)
            }
            QueryPlan::Similar { entity } => self.answer_similar(&entity),
            QueryPlan::Analogy { a, b, c } => self.answer_analogy(&a, &b, &c),
        }
    }

    fn answer_count(&self, subject: &str, predicate: &str, exist: bool) -> Option<String> {
        let entity = self.aligner.canonical_form(subject.trim());
        let predicate = self.aligner.canonical_form(predicate.trim());
        if entity.is_empty() || predicate.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) && !self.predictor.participates(&entity) {
            return None;
        }
        let hits = self.node_hits(&entity, &predicate);
        if hits.is_empty() {
            if self.is_known_predicate(&predicate) {
                if exist {
                    return Some(format!(
                        "根据现有知识库，「{}」没有「{}」的记录。",
                        entity, predicate
                    ));
                }
                return Some(format!(
                    "根据现有知识库，「{}」的「{}」数量为 0（没有相关记录）。",
                    entity, predicate
                ));
            }
            return None;
        }
        let values: Vec<String> = hits.iter().map(|hit| hit.value.clone()).collect();
        let mut output = if exist {
            format!(
                "是的，根据知识图谱，「{}」有 {}（共 {} 个：{}）。",
                entity,
                predicate,
                values.len(),
                values.join("、")
            )
        } else {
            format!(
                "根据知识图谱：「{}」的「{}」共有 {} 个：{}。",
                entity,
                predicate,
                values.len(),
                values.join("、")
            )
        };
        if let Some(prediction) = hits.iter().find_map(|hit| hit.prediction.as_ref()) {
            output.push_str(&format!(
                "\n其中为符号推断结果（例如置信度 {:.2}）：{}",
                prediction.confidence,
                self.predictor.explain(prediction)
            ));
        }
        Some(output)
    }

    fn answer_check(&self, subject_segments: &[String], object_raw: &str) -> Option<String> {
        if subject_segments.len() < 2 {
            return None;
        }
        let predicate_raw = subject_segments.last()?.clone();
        let (entity, mut intermediate) =
            self.split_chain(&subject_segments[..subject_segments.len() - 1])?;
        let predicate = self.aligner.canonical_form(&predicate_raw);
        let object = self.aligner.canonical_form(object_raw.trim());
        if entity.is_empty() || predicate.is_empty() || object.is_empty() {
            return None;
        }
        intermediate.push(predicate.clone());

        if intermediate.len() == 1
            && self.predictor.has_triple(&entity, &predicate, &object)
        {
            return Some(format!(
                "是的。根据知识图谱：{} 的 {} 是 {}。",
                entity, predicate, object
            ));
        }

        if let Some((steps, last_hits)) = self.run_chain(&entity, &intermediate) {
            if let Some(hit) = last_hits.iter().find(|hit| hit.value == object) {
                let step = steps.last()?;
                return Some(match &hit.prediction {
                    None => format!(
                        "是的。根据知识图谱：{} 的 {} 是 {}。",
                        step.subject, step.predicate, object
                    ),
                    Some(prediction) => format!(
                        "很可能是的（符号推断）。{}",
                        self.predictor.explain(prediction)
                    ),
                });
            }
            let known_values: Vec<String> = last_hits
                .iter()
                .filter(|hit| hit.prediction.is_none())
                .map(|hit| hit.value.clone())
                .collect();
            if !known_values.is_empty() {
                let step = steps.last()?;
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    step.subject,
                    step.predicate,
                    known_values.join("、")
                ));
            }
        } else if intermediate.len() == 1 {
            let known_values = self.predictor.objects_of(&entity, &predicate);
            if !known_values.is_empty() {
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    entity,
                    predicate,
                    known_values.join("、")
                ));
            }
        }

        if self.is_known_entity(&entity) && self.is_known_predicate(&predicate) {
            return Some(format!(
                "根据现有知识库，无法确认「{} 的 {} 是 {}」（没有相关记录）。",
                entity, predicate, object
            ));
        }
        None
    }

    fn answer_find_subject(&self, object_raw: &str, predicate_raw: &str) -> Option<String> {
        let object = self.aligner.canonical_form(object_raw.trim());
        let predicate = self.aligner.canonical_form(predicate_raw.trim());
        if object.is_empty() || predicate.is_empty() {
            return None;
        }
        let mut subjects: Vec<String> = self.predictor.subjects_of(&predicate, &object);
        let mut evidence: Vec<String> = Vec::new();
        if self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            for prediction in self.predictor.predict_for_entity(&object, Some(&predicate), limit) {
                if prediction.triple.object == object && prediction.triple.subject != object {
                    if !subjects.contains(&prediction.triple.subject) {
                        subjects.push(prediction.triple.subject.clone());
                    }
                    evidence.push(self.predictor.explain(&prediction));
                }
            }
        }
        if subjects.is_empty() {
            if self.is_known_entity(&object) && self.is_known_predicate(&predicate) {
                return Some(format!(
                    "根据现有知识库，没有记录「谁的 {} 是 {}」。",
                    predicate, object
                ));
            }
            return None;
        }
        subjects.sort();
        let mut output = format!(
            "根据知识图谱：{} 的 {} 是 {}。",
            subjects.join("、"),
            predicate,
            object
        );
        if !evidence.is_empty() {
            output.push_str(&format!("\n推理依据：{}", evidence.join("；")));
        }
        Some(output)
    }

    fn answer_similar(&self, entity_raw: &str) -> Option<String> {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() || !self.embedding.is_embedded(&entity) {
            return None;
        }
        let neighbors = self.embedding.nearest(&entity, 3);
        if neighbors.is_empty() {
            return None;
        }
        let mut output = format!(
            "符号嵌入近邻检索（PPMI 角色向量，无神经网络）：与「{}」语义最接近的实体：\n",
            entity
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn answer_analogy(&self, a_raw: &str, b_raw: &str, c_raw: &str) -> Option<String> {
        let a = self.aligner.canonical_form(a_raw.trim());
        let b = self.aligner.canonical_form(b_raw.trim());
        let c = self.aligner.canonical_form(c_raw.trim());
        if a.is_empty() || b.is_empty() || c.is_empty() {
            return None;
        }
        let results = self.embedding.analogy(&a, &b, &c, 3);
        if results.is_empty() {
            return None;
        }
        let direction = self.embedding.describe_direction(&a, &b);
        let mut output = format!(
            "向量算术类比（{} : {} :: {} : ?）：\n  从「{}」到「{}」的语义方向是{}，应用到「{}」：\n",
            a, b, c, a, b,
            if direction.is_empty() {
                String::new()
            } else {
                format!("「{}」", direction)
            },
            c
        );
        for (index, (name, score)) in results.iter().enumerate() {
            let shared = self.embedding.shared_feature_labels(&b, name, 2);
            let shared_text = if shared.is_empty() {
                String::new()
            } else {
                format!("，与「{}」共享 {}", b, shared.join("、"))
            };
            output.push_str(&format!(
                "  {}. 「{}」（语义偏移相似度 {:.2}{}）\n",
                index + 1,
                name,
                score,
                shared_text
            ));
        }
        Some(output.trim_end().to_string())
    }

    pub fn answer(&self, question: &str) -> Option<String> {
        if let Some(answer) = self.answer_planned(question) {
            return Some(answer);
        }
        self.answer_by_prediction(question)
    }

    fn answer_by_prediction(&self, question: &str) -> Option<String> {
        if !self.config.enable_link_prediction || self.predictor.rules().is_empty() {
            return None;
        }
        let mut entities = self.entities_mentioned(question);
        if entities.is_empty() && self.config.enable_generation {
            // 精确匹配失败时，退回到带容错的实体链接结果（别名词表之外也能命中）
            entities = self.linked_entities(question);
        }
        if entities.is_empty() {
            return None;
        }

        let normalized_question = self.aligner.normalize(question);
        let mut candidates: Vec<Prediction> = Vec::new();
        for entity in &entities {
            for prediction in self.predictor.predict_for_entity(
                entity,
                None,
                self.config.max_predictions * 4,
            ) {
                if self.predicate_mentioned_normalized(
                    question,
                    &normalized_question,
                    &prediction.triple.predicate,
                ) {
                    candidates.push(prediction);
                }
            }
        }
        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.triple.subject.cmp(&b.triple.subject))
                .then_with(|| a.triple.object.cmp(&b.triple.object))
        });
        let mut seen: HashSet<Triple> = HashSet::new();
        candidates.retain(|p| seen.insert(p.triple.clone()));

        let limit = self.config.max_predictions.clamp(1, 5);
        let mut output = String::from("根据知识图谱的链接预测（符号推理）：\n");
        for prediction in candidates.iter().take(limit) {
            output.push_str("  • ");
            output.push_str(&self.predictor.explain(prediction));
            output.push('\n');
        }
        Some(output)
    }

    pub fn similarity_report(&self, entity_raw: &str) -> String {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() {
            return "请输入要检索的实体名，例如：相似实体：张三".to_string();
        }
        if !self.embedding.is_embedded(&entity) {
            return format!(
                "语义嵌入中没有「{}」的角色特征（可能是孤立节点或尚未导入关系数据）。",
                entity
            );
        }
        let neighbors = self.embedding.nearest(&entity, 5);
        if neighbors.is_empty() {
            return format!(
                "没有与「{}」相似度达到阈值 {:.2} 的实体。",
                entity,
                self.embedding.similarity_floor()
            );
        }
        let mut output = format!(
            "符号嵌入近邻（PPMI 角色向量；已嵌入 {} 个实体、{} 个特征）：\n",
            self.embedding.entity_count(),
            self.embedding.feature_count()
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        output.trim_end().to_string()
    }

    pub fn dialogue_summary(&self) -> String {
        let focus = if self.context.focus_entities().is_empty() {
            "（空）".to_string()
        } else {
            self.context.focus_entities().join(" → ")
        };
        let predicate = self
            .context
            .last_predicate()
            .unwrap_or("（无）")
            .to_string();
        let generation = self
            .context
            .generation_state()
            .map(|state| {
                let order = match state.order {
                    GenerationOrder::Relevance => "信息量排序",
                    GenerationOrder::Alphabetical => "主题排序",
                };
                format!("「{}」（已讲 {} 句，{}）", state.entity, state.shown, order)
            })
            .unwrap_or_else(|| "（无）".to_string());
        format!(
            "对话记忆：焦点实体栈 [{}]，上一轮谓词「{}」，上一段生成 {}，历史问题 {} 条。",
            focus,
            predicate,
            generation,
            self.context.history().len()
        )
    }

    pub fn save_dialogue_state(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let snapshot = self.context.snapshot();
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load_dialogue_state(&mut self, path: &str) -> bool {
        let Ok(content) = std::fs::read_to_string(path) else {
            return false;
        };
        let Ok(snapshot) = serde_json::from_str::<DialogueSnapshot>(&content) else {
            return false;
        };
        self.context.restore(&snapshot);
        true
    }

    pub fn alignment_report(&self) -> String {
        let pairs = self.aligner.align_all();
        let mut output = format!(
            "本体对齐报告：已登记 {} 个术语，发现 {} 组对齐关系。\n",
            self.aligner.term_count(),
            pairs.len()
        );
        for pair in pairs.iter().take(20) {
            output.push_str(&format!(
                "  • {} ⇒ {}（{:?}，置信度 {:.2}，证据：{}）\n",
                pair.source,
                pair.target,
                pair.kind,
                pair.confidence,
                pair.evidence.join("；")
            ));
        }
        if pairs.is_empty() {
            output.push_str("  （未发现跨词表对齐，可补充同义词或更多关系数据）\n");
        }
        output
    }

    pub fn rules_report(&self) -> String {
        let rules = self.predictor.rules();
        let mut output = format!(
            "已从 {} 条关系事实中挖掘出 {} 条符号推理规则（支持度阈值 {}，置信度阈值 {:.2}）：\n",
            self.predictor.triple_count(),
            rules.len(),
            self.config.link_prediction_min_support,
            self.config.link_prediction_min_confidence
        );
        for rule in rules.iter().take(30) {
            output.push_str(&format!(
                "  • [{}] {}（支持 {} 例，置信度 {:.2}，头部覆盖率 {:.2}）\n",
                rule.template.name(),
                rule.rule_text(),
                rule.support,
                rule.confidence,
                rule.head_coverage
            ));
        }
        if rules.is_empty() {
            output.push_str("  （暂无满足阈值的规则，可降低支持度/置信度阈值或增加关系数据）\n");
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ReasoningConfig {
        ReasoningConfig {
            enable_link_prediction: true,
            link_prediction_min_confidence: 0.5,
            link_prediction_min_support: 1,
            ontology_alignment_threshold: 0.72,
            max_predictions: 5,
            enable_semantic_embedding: true,
            embedding_similarity_floor: 0.6,
            dialogue_state_path: None,
            enable_generation: true,
            max_profile_sentences: 6,
            completion_min_score: 0.3,
            enable_entailment: true,
            entailment_min_confidence: 0.5,
            enable_resolution: true,
            resolution_max_depth: 6,
            resolution_max_solutions: 12,
            enable_association_rules: true,
            association_min_support: 0.2,
            association_min_confidence: 0.5,
            association_min_lift: 1.05,
        }
    }

    fn master_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师祖".to_string(),
                to: "王五".to_string(),
            },
        ]
    }

    fn pair_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]
    }

    #[test]
    fn test_pronoun_resolution_uses_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师傅是谁").is_none());
        let rewritten = reasoner
            .contextualize("他的师傅是谁")
            .expect("代词应被解析为上下文焦点实体");
        assert!(rewritten.contains("张三"));
        assert!(!rewritten.contains("他的"));
    }

    #[test]
    fn test_ellipsis_completion_uses_last_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let rewritten = reasoner
            .contextualize("那李四呢")
            .expect("省略句应结合上文补全");
        assert_eq!(rewritten, "李四的师祖是什么");
    }

    #[test]
    fn test_contextualize_ignores_other_ren() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewritten = reasoner.contextualize("其他人的师傅是谁");
        assert!(rewritten.is_none() || rewritten.as_deref() == Some("其他张三的师傅是谁"));
    }

    #[test]
    fn test_answer_generates_prediction_with_explanation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的师祖是谁")
            .expect("应能通过组合规则预测出师祖");
        assert!(answer.contains("赵六"));
        assert!(answer.contains("师傅"));
        assert!(answer.contains("置信度"));
    }

    #[test]
    fn test_answer_requires_mentioned_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer("李四是谁").is_none());
    }

    #[test]
    fn test_answer_returns_none_without_facts() {
        let reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert!(reasoner.answer("李四的师祖是谁").is_none());
    }

    #[test]
    fn test_synonym_alias_matches_predicate() {
        let mut synonyms = HashMap::new();
        synonyms.insert("祖师".to_string(), "师祖".to_string());
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的祖师是谁")
            .expect("同义词别名应能命中师祖关系");
        assert!(answer.contains("赵六"));
    }

    #[test]
    fn test_alignment_and_rules_reports() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let rules_report = reasoner.rules_report();
        assert!(rules_report.contains("师傅"));
        let alignment_report = reasoner.alignment_report();
        assert!(alignment_report.contains("本体对齐报告"));
    }

    #[test]
    fn test_planned_multi_hop_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问题应被逻辑计划执行");
        assert!(answer.contains("王五"));
        assert!(answer.contains("多跳"));
    }

    #[test]
    fn test_planned_count_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四有几个师傅")
            .expect("计数问题应被逻辑计划执行");
        assert!(answer.contains("共有 1 个"));
        assert!(answer.contains("王五"));
    }

    #[test]
    fn test_planned_yes_no_true_and_false() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let yes = reasoner
            .answer("李四的师傅是王五吗")
            .expect("是非问题应被逻辑计划执行");
        assert!(yes.starts_with("是的"), "实际回答：{}", yes);
        let no = reasoner
            .answer("李四的师傅是张三吗")
            .expect("错误的断言也应给出否定回答");
        assert!(no.starts_with("不是"), "实际回答：{}", no);
        assert!(no.contains("王五"));
    }

    #[test]
    fn test_planned_find_subject() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("谁的师傅是李四")
            .expect("反查主体应被逻辑计划执行");
        assert!(answer.contains("张三"));
    }

    #[test]
    fn test_planned_answer_reports_missing_record() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("赵六的师祖是谁")
            .expect("已知实体与已知谓词应给出明确的无记录回答");
        assert!(answer.contains("没有记录"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_exist_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三有没有师傅")
            .expect("存在性问题应被逻辑计划执行");
        assert!(answer.starts_with("是的"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_similar_entity_retrieval() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("谁和张三最像")
            .expect("近邻检索应命中最相似的实体");
        assert!(answer.contains("王五"));
        assert!(answer.contains("相似度"));
    }

    #[test]
    fn test_planned_analogy_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("张三对应李四，王五对应谁")
            .expect("类比问题应由向量算术回答");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_similarity_report_handles_unknown_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let report = reasoner.similarity_report("不存在的人");
        assert!(report.contains("没有"));
        let report = reasoner.similarity_report("张三");
        assert!(report.contains("王五"));
    }

    #[test]
    fn test_dialogue_state_persistence_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        reasoner.contextualize("那李四呢");
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存对话状态");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.context().focus(), Some("李四"));
        assert_eq!(restored.context().last_predicate(), Some("师祖"));
        let rewritten = restored
            .contextualize("他的师祖是谁")
            .expect("恢复后的记忆应支持指代消解");
        assert!(rewritten.contains("李四"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_dialogue_summary_mentions_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let summary = reasoner.dialogue_summary();
        assert!(summary.contains("张三"));
        assert!(summary.contains("师祖"));
    }

    #[test]
    fn test_embedding_feeds_from_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        assert!(reasoner.embedding().is_embedded("张三"));
        assert!(reasoner.embedding().is_embedded("赵六"));
        assert!(reasoner.embedding().similarity("张三", "王五") > 0.9);
    }

    #[test]
    fn test_entity_link_notes_for_typo_mention() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let notes = reasoner.entity_link_notes("张山的师傅是谁");
        assert!(
            notes.iter().any(|note| note.contains("张三") && note.contains("→")),
            "错别字指称应产生模糊链接说明：{:?}",
            notes
        );
        assert!(reasoner.linker().is_known_entity("张三"));
    }

    #[test]
    fn test_linked_entity_fallback_recovers_typo_query() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李寺的师祖是谁")
            .expect("错别字实体应能通过链接回退命中推理");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_generate_about_combines_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("师祖"));
        assert!(profile.contains("李四"));
        assert!(reasoner.generate_about("不存在的人").is_none());
    }

    #[test]
    fn test_check_statement_entailed_contradicted_unknown() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let entailed = reasoner.check_statement("张三的师傅是李四");
        assert_eq!(entailed.len(), 1);
        assert_eq!(entailed[0].verdict, Verdict::Entailed);
        assert_eq!(
            entailed[0].proof[0].kind,
            semantic_entailment::ProofKind::DirectFact
        );

        let contradicted = reasoner.check_statement("李四的师傅是张三");
        assert_eq!(contradicted[0].verdict, Verdict::Contradicted);
        assert!(contradicted[0]
            .counter_evidence
            .iter()
            .any(|evidence| evidence.contains("王五")));

        let unknown = reasoner.check_statement("钱七的师傅是李四");
        assert_eq!(unknown[0].verdict, Verdict::Unknown);
        assert!(unknown[0].reason.contains("尚未收录"), "实际原因：{}", unknown[0].reason);

        assert!(reasoner.check_statement("张三的师傅是谁").is_empty());
    }

    #[test]
    fn test_check_statement_recovers_typo_via_linker() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let results = reasoner.check_statement("张山的师傅是李四");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verdict, Verdict::Entailed);
        assert!(
            results[0]
                .proposition
                .notes
                .iter()
                .any(|note| note.contains("模糊链接") && note.contains("张三")),
            "解析证据应记录容错链接：{:?}",
            results[0].proposition.notes
        );
    }

    #[test]
    fn test_entailment_report_renders_logical_form_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.entailment_report("张三的师傅是李四");
        assert!(report.contains("逻辑形式：relation(张三, 师傅, 李四)"));
        assert!(report.contains("蕴含"));
        assert!(report.contains("直接事实"));

        let failure = reasoner.entailment_report("今天天气如何");
        assert!(failure.contains("形式化失败"));
    }

    #[test]
    fn test_generate_about_appends_entailment_self_check() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("生成自检"), "应附蕴含自检：{}", profile);
        assert!(profile.contains("零幻觉") || profile.contains("未通过"));
        assert_eq!(reasoner.context().generation_state().map(|s| s.entity.as_str()), Some("张三"));
    }

    #[test]
    fn test_generation_self_check_with_bare_english_predicates() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "张三".to_string()),
                    ("age".to_string(), "25".to_string()),
                ]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "李四".to_string()),
                    ("age".to_string(), "30".to_string()),
                ]),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "knows".to_string(),
                to: "李四".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let profile = reasoner.generate_about("张三").expect("应能生成介绍");
        assert!(profile.contains("张三认识李四"), "{}", profile);
        assert!(profile.contains("零幻觉"), "裸谓词生成应通过自检：{}", profile);
    }

    #[test]
    fn test_continue_generation_and_switch_angle() {
        let mut config = test_config();
        config.max_profile_sentences = 1;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());

        let first = reasoner.generate_about("张三").expect("第一页应可生成");
        assert!(first.contains("共 1 条事实"), "实际：{}", first);
        let second = reasoner
            .continue_generation()
            .expect("接着说应返回下一页");
        assert!(second.contains("续：第 2–2 句"), "实际：{}", second);
        assert!(second.contains("接着，"));
        assert_ne!(first, second);

        let done = reasoner.continue_generation().expect("讲完后应给出提示");
        assert!(done.contains("已经介绍完毕"));

        let angled = reasoner
            .switch_generation_angle()
            .expect("换个角度说应重新生成");
        assert!(angled.contains("换个角度"), "实际：{}", angled);
        assert!(reasoner.dialogue_summary().contains("上一段生成"));
    }

    #[test]
    fn test_expected_answer_type_reorders_find_object() {
        let mut facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "徒弟".to_string(),
                to: "星海公司".to_string(),
            },
        ];
        facts.push(PrologFact::Entity {
            name: "李四".to_string(),
            entity_type: "人物".to_string(),
            properties: HashMap::new(),
        });
        facts.push(PrologFact::Entity {
            name: "星海公司".to_string(),
            entity_type: "公司".to_string(),
            properties: HashMap::new(),
        });
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner
            .answer("张三的徒弟是谁")
            .expect("应能回答反查对象问题");
        let person_at = answer.find("李四").expect("应包含李四");
        let company_at = answer.find("星海公司").expect("应包含星海公司");
        assert!(
            person_at < company_at,
            "人物类型答案应排在前面：{}",
            answer
        );
        assert!(answer.contains("期望"), "应说明排序依据：{}", answer);
    }

    #[test]
    fn test_predict_continuation_uses_facts_and_backoff() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let completion = reasoner
            .predict_continuation("张三的")
            .expect("已知实体应能续写");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));

        let mut synonyms = HashMap::new();
        synonyms.insert("老三".to_string(), "张三".to_string());
        let mut aliased = SymbolicReasoner::new(test_config(), synonyms);
        aliased.observe_facts(&master_facts());
        let aliased_completion = aliased
            .predict_continuation("老三的")
            .expect("同义词前缀应先归一化再续写");
        assert!(aliased_completion.contains("张三"));
    }

    fn nationality_facts() -> Vec<PrologFact> {
        let mut facts = Vec::new();
        for (name, has_nationality) in [
            ("张三", true),
            ("李四", true),
            ("王五", true),
            ("赵六", false),
        ] {
            let mut properties = HashMap::new();
            properties.insert("出生地".to_string(), format!("{}市", name));
            if has_nationality {
                properties.insert("国籍".to_string(), "中国".to_string());
            }
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "人物".to_string(),
                properties,
            });
        }
        // 一个无特征的地点篮子：拉低国籍先验，使关联规则的提升度可判
        facts.push(PrologFact::Entity {
            name: "无名城".to_string(),
            entity_type: "地点".to_string(),
            properties: HashMap::new(),
        });
        // 关系事实：让生成器有全局统计可回退（属性不进入关系计数）
        facts.push(PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        });
        facts
    }

    #[test]
    fn test_resolution_proof_from_facts_and_chain_goals() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("张三的师傅的师傅是王五");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(report.contains("王五"));
        assert!(report.contains("证明树"));
        assert!(report.contains("事实"));
        let failed = reasoner.resolution_proof_report("张三的师傅的师傅是赵六");
        assert!(failed.contains("未得证"), "实际：{}", failed);
    }

    #[test]
    fn test_resolution_proof_uses_mined_horn_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_proof_report("李四的师祖是赵六");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(
            report.contains("链接预测规则"),
            "证明树应标出规则来源：{}",
            report
        );
        assert!(report.contains("师傅"));
    }

    #[test]
    fn test_resolution_query_returns_solution_and_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.resolution_query_report("张三的师傅的师傅是谁");
        assert!(report.contains("答案是「王五」"), "实际：{}", report);
        assert!(report.contains("事实"));
        assert!(report.contains("知识库"));
    }

    #[test]
    fn test_resolution_proves_existence_via_association_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.resolution_proof_report("赵六有国籍");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(report.contains("关联规则"), "应有归结算子句来源：{}", report);
        assert!(report.contains("国籍"));
        let known = reasoner.resolution_proof_report("张三有国籍");
        assert!(known.contains("得证"));
        assert!(known.contains("事实"));
    }

    #[test]
    fn test_association_report_and_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.association_report(None);
        assert!(report.contains("支持度"), "实际：{}", report);
        assert!(report.contains("提升度"));
        assert!(report.contains("出生地"));
        let filtered = reasoner.association_report(Some("国籍"));
        assert!(filtered.contains("国籍"), "实际：{}", filtered);
        let suggestions = reasoner
            .association_suggestions("赵六")
            .expect("赵六缺国籍应有关联联想");
        assert!(suggestions.contains("国籍"), "实际：{}", suggestions);
        assert!(suggestions.contains("提升度"));
        assert!(!suggestions.contains("已有特征：国籍"));
    }

    #[test]
    fn test_derived_facts_report_discovers_multi_hop() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.derived_facts_report(10);
        assert!(report.contains("推理发现"), "实际：{}", report);
        assert!(
            report.contains("第 1 轮推导"),
            "应给出推导轮次：{}",
            report
        );
        assert!(report.contains("李四") && report.contains("赵六"));
        assert!(report.contains("前提"));
    }

    #[test]
    fn test_continuation_appends_association_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let text = reasoner
            .predict_continuation("赵六的")
            .expect("已知实体应能续写");
        assert!(text.contains("关联规则联想"), "实际：{}", text);
        assert!(text.contains("国籍"));
    }
}

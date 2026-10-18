// 形式化语义与语义蕴含（Formal Semantics & Semantic Entailment）
//
// 本模块把自然语言陈述转换为可判定的逻辑形式（命题），再在知识图谱上做
// 语义蕴含判定（entailment），全程符号化、零神经网络：
//   • 形式化：用句式模式「S的P是O」「O是S的P」「S是类型」「S P O」把句子
//     归一化为 relation(S,P,O) / attribute(S,P,O) / type(S,T)，并记录
//     同义归一、模糊链接等解析证据；
//   • 蕴含：把命题与知识图谱事实、属性记录、类型断言、对称关系、规则推断
//     逐项匹配（可直接回放证据链）；匹配失败且图谱有相反记录时按封闭世界
//     假设给出矛盾与反证；证据不足则明确报"未知"。
// 这套机制让系统能像 LLM 一样"听懂陈述句"，但判定结果完全可追溯、可复现。

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::entity_linker::EntityLinker;
use crate::link_prediction::LinkPredictor;
use crate::ontology_alignment::{OntologyAligner, TermKind};
use crate::symbolic_generator::SymbolicGenerator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropositionKind {
    Relation,
    Attribute,
    TypeMembership,
}

impl PropositionKind {
    pub fn label(self) -> &'static str {
        match self {
            PropositionKind::Relation => "关系",
            PropositionKind::Attribute => "属性",
            PropositionKind::TypeMembership => "类型",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Proposition {
    pub kind: PropositionKind,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub negative: bool,
    // 命中的形式化句式（用于解释解析来源）
    pub pattern: &'static str,
    pub subject_resolved: bool,
    pub object_is_literal: bool,
    pub notes: Vec<String>,
}

impl Proposition {
    pub fn render(&self) -> String {
        let body = match self.kind {
            PropositionKind::Relation => {
                format!("relation({}, {}, {})", self.subject, self.predicate, self.object)
            }
            PropositionKind::Attribute => {
                format!("attribute({}, {}, {})", self.subject, self.predicate, self.object)
            }
            PropositionKind::TypeMembership => {
                format!("type({}, {})", self.subject, self.object)
            }
        };
        if self.negative {
            format!("not({})", body)
        } else {
            body
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Entailed,
    Contradicted,
    Unknown,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Entailed => "蕴含（与知识图谱一致）",
            Verdict::Contradicted => "矛盾（与知识图谱不符）",
            Verdict::Unknown => "未知（证据不足）",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofKind {
    DirectFact,
    AttributeRecord,
    TypeAssertion,
    InverseRelation,
    RuleInference,
    ClosedWorld,
}

impl ProofKind {
    pub fn label(self) -> &'static str {
        match self {
            ProofKind::DirectFact => "直接事实",
            ProofKind::AttributeRecord => "属性记录",
            ProofKind::TypeAssertion => "类型断言",
            ProofKind::InverseRelation => "对称关系",
            ProofKind::RuleInference => "规则推断",
            ProofKind::ClosedWorld => "封闭世界假设",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProofStep {
    pub kind: ProofKind,
    pub detail: String,
}

impl ProofStep {
    fn new(kind: ProofKind, detail: String) -> Self {
        Self { kind, detail }
    }
}

#[derive(Debug, Clone)]
pub struct EntailmentResult {
    pub proposition: Proposition,
    pub verdict: Verdict,
    pub confidence: f64,
    pub proof: Vec<ProofStep>,
    pub counter_evidence: Vec<String>,
    pub reason: String,
}

pub struct Knowledge<'a> {
    pub predictor: &'a LinkPredictor,
    pub generator: &'a SymbolicGenerator,
    pub min_rule_confidence: f64,
    pub max_rule_candidates: usize,
}

const FUZZY_ACCEPT: f64 = 0.5;
const DISCOURSE_MARKERS: [&str; 5] = ["首先，", "此外，", "最后，", "接着，", "然后，"];

pub fn strip_discourse_markers(text: &str) -> &str {
    let mut current = text.trim();
    loop {
        let mut stripped = false;
        for marker in DISCOURSE_MARKERS {
            if let Some(rest) = current.strip_prefix(marker) {
                current = rest.trim();
                stripped = true;
            }
        }
        if !stripped {
            return current;
        }
    }
}

fn clean_sentence(text: &str) -> String {
    let mut body = strip_discourse_markers(text);
    for prefix in ["我认为", "我觉得", "据说", "听说"] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim();
        }
    }
    let trimmed = body.trim().trim_end_matches(|c: char| {
        matches!(
            c,
            '。' | '！' | '!' | '？' | '?' | '；' | ';' | '，' | ',' | '、' | ' ' | '\t'
        )
    });
    let trimmed = trimmed
        .trim_end_matches(|c: char| matches!(c, '吗' | '呢' | '吧' | '啊' | '呀'))
        .trim();
    trimmed.to_string()
}

struct Resolved {
    canonical: String,
    resolved: bool,
    notes: Vec<String>,
}

pub struct PropositionParser<'a> {
    aligner: &'a OntologyAligner,
    linker: &'a EntityLinker,
    // 表面积（含中文动词化形式）→ 规范谓词，按长度降序排列
    surfaces: Cow<'a, [(String, String)]>,
}

/// 构建“表面积 → 规范谓词”索引（按表面长度降序、再按字典序）。
///
/// 该索引只依赖对齐器与词法表面表。
pub fn build_predicate_surfaces(
    aligner: &OntologyAligner,
    lexical_surfaces: &HashMap<String, String>,
) -> Vec<(String, String)> {
    let mut surfaces: HashMap<String, String> = lexical_surfaces.clone();
    for profile in aligner.profiles() {
        if matches!(profile.kind, TermKind::Relation | TermKind::Attribute) {
            let canonical = aligner.canonical_form(&profile.term);
            if !canonical.is_empty() {
                surfaces.entry(canonical.clone()).or_insert(canonical);
            }
        }
    }
    for (alias, canonical) in aligner.synonyms() {
        let target = aligner.canonical_form(canonical);
        if matches!(
            aligner.kind_of(&target),
            Some(TermKind::Relation) | Some(TermKind::Attribute)
        ) {
            surfaces.entry(alias.clone()).or_insert(target);
        }
    }
    // 装饰-排序-还原（排序键为字符数）
    let mut decorated: Vec<(std::cmp::Reverse<usize>, String, String)> = surfaces
        .into_iter()
        .map(|(name, canonical)| {
            let len = name.chars().count();
            (std::cmp::Reverse(len), name, canonical)
        })
        .collect();
    decorated.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    decorated
        .into_iter()
        .map(|(_, name, canonical)| (name, canonical))
        .collect()
}

impl<'a> PropositionParser<'a> {
    pub fn new(
        aligner: &'a OntologyAligner,
        linker: &'a EntityLinker,
        lexical_surfaces: &HashMap<String, String>,
    ) -> Self {
        Self {
            aligner,
            linker,
            surfaces: Cow::Owned(build_predicate_surfaces(aligner, lexical_surfaces)),
        }
    }

    /// 使用调用方缓存的表面积索引构造解析器
    pub fn with_surfaces(
        aligner: &'a OntologyAligner,
        linker: &'a EntityLinker,
        surfaces: &'a [(String, String)],
    ) -> Self {
        Self {
            aligner,
            linker,
            surfaces: Cow::Borrowed(surfaces),
        }
    }

    /// 识别否定并把句子还原成肯定形式。
    ///
    /// 裸「不」只有在紧跟已知谓词表面时才视为否定词，避免把「不锈钢」这类
    /// 词内部的「不」误当成否定（否则「不锈钢是合金」会被解析成否定命题）。
    fn detect_negation(&self, body: &str) -> (bool, String) {
        for marker in ["不属于", "不是", "没有"] {
            if body.contains(marker) {
                let replacement = match marker {
                    "不属于" => "属于",
                    "不是" => "是",
                    _ => "有",
                };
                return (true, body.replace(marker, replacement));
            }
        }
        for (pos, _) in body.match_indices('不') {
            let rest = &body[pos + '不'.len_utf8()..];
            if self
                .surfaces
                .iter()
                .any(|(surface, _)| rest.starts_with(surface.as_str()))
            {
                let mut replaced = String::with_capacity(body.len() - '不'.len_utf8());
                replaced.push_str(&body[..pos]);
                replaced.push_str(rest);
                return (true, replaced);
            }
        }
        (false, body.to_string())
    }

    pub fn parse(&self, text: &str) -> Vec<Proposition> {
        let cleaned = clean_sentence(text);
        if cleaned.is_empty() || crate::function_words::contains_question_marker(&cleaned) {
            return Vec::new();
        }
        let (negative, body) = self.detect_negation(&cleaned);
        let mut props: Vec<Proposition> = Vec::new();
        if body.contains('的') {
            self.parse_de_shi(&body, negative, &mut props);
        }
        if props.is_empty() {
            self.parse_type_membership(&body, negative, &mut props);
        }
        if props.is_empty() {
            self.parse_predicate_infix(&body, negative, &mut props);
        }
        let mut seen: HashSet<(u8, String, String, String, bool)> = HashSet::new();
        props.retain(|p| {
            seen.insert((
                match p.kind {
                    PropositionKind::Relation => 0,
                    PropositionKind::Attribute => 1,
                    PropositionKind::TypeMembership => 2,
                },
                p.subject.clone(),
                p.predicate.clone(),
                p.object.clone(),
                p.negative,
            ))
        });
        props
    }

    fn resolve_entity(&self, text: &str) -> Resolved {
        let trimmed = text.trim();
        let mut notes = Vec::new();
        if trimmed.is_empty() {
            return Resolved {
                canonical: String::new(),
                resolved: false,
                notes,
            };
        }
        let canonical = self.aligner.canonical_form(trimmed);
        if !canonical.is_empty()
            && (self.aligner.kind_of(&canonical) == Some(TermKind::Entity)
                || self.linker.is_known_entity(&canonical))
        {
            if canonical != trimmed {
                notes.push(format!("同义归一：{} → {}", trimmed, canonical));
            }
            return Resolved {
                canonical,
                resolved: true,
                notes,
            };
        }
        let char_count = trimmed.chars().count();
        if (2..=12).contains(&char_count) {
            if let Some(candidate) = self
                .linker
                .link(trimmed)
                .iter()
                .filter_map(|mention| mention.best())
                .find(|candidate| candidate.score >= FUZZY_ACCEPT)
            {
                notes.push(format!(
                    "模糊链接：{} ≈ {}（置信度 {:.2}）",
                    trimmed, candidate.entity, candidate.score
                ));
                return Resolved {
                    canonical: candidate.entity.clone(),
                    resolved: true,
                    notes,
                };
            }
        }
        Resolved {
            canonical: trimmed.to_string(),
            resolved: false,
            notes,
        }
    }

    fn resolve_predicate(&self, text: &str) -> Option<(String, PropositionKind, Vec<String>)> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let mut notes = Vec::new();
        // surfaces 按字符数降序排列：精确匹配只可能落在等长的连续块内
        let surface_len = trimmed.chars().count();
        let block_start = self
            .surfaces
            .partition_point(|(surface, _)| surface.chars().count() > surface_len);
        let mapped = self.surfaces[block_start..]
            .iter()
            .take_while(|(surface, _)| surface.chars().count() == surface_len)
            .find(|(surface, _)| surface == trimmed)
            .map(|(_, canonical)| canonical.clone());
        let canonical = mapped
            .clone()
            .unwrap_or_else(|| self.aligner.canonical_form(trimmed));
        let kind = match self.aligner.kind_of(&canonical) {
            Some(TermKind::Relation) => PropositionKind::Relation,
            Some(TermKind::Attribute) => PropositionKind::Attribute,
            _ if mapped.is_some() => PropositionKind::Relation,
            _ => return None,
        };
        if canonical != trimmed {
            notes.push(format!("谓词归一：{} → {}", trimmed, canonical));
        }
        if let Some(surface) = mapped {
            if surface != canonical {
                notes.push(format!("中文表述：{} = {}", trimmed, canonical));
            }
        }
        Some((canonical, kind, notes))
    }

    fn parse_de_shi(&self, body: &str, negative: bool, props: &mut Vec<Proposition>) {
        let Some(pos_de) = body.find('的') else {
            return;
        };
        let rest = &body[pos_de + '的'.len_utf8()..];
        // 模式一：S的P是O
        if let Some(pos_shi) = rest.find('是') {
            let subject_text = &body[..pos_de];
            let predicate_text = &rest[..pos_shi];
            let object_text = &rest[pos_shi + '是'.len_utf8()..];
            self.push_fact_prop(
                subject_text,
                predicate_text,
                object_text,
                negative,
                "S的P是O",
                props,
            );
        }
        // 模式二：O是S的P
        if let Some(pos_shi) = body[..pos_de].rfind('是') {
            let object_text = &body[..pos_shi];
            let subject_text = &body[pos_shi + '是'.len_utf8()..pos_de];
            let predicate_text = &body[pos_de + '的'.len_utf8()..];
            self.push_fact_prop(
                subject_text,
                predicate_text,
                object_text,
                negative,
                "O是S的P",
                props,
            );
        }
    }

    fn push_fact_prop(
        &self,
        subject_text: &str,
        predicate_text: &str,
        object_text: &str,
        negative: bool,
        pattern: &'static str,
        props: &mut Vec<Proposition>,
    ) {
        let subject_text = subject_text.trim();
        let object_text = object_text.trim();
        if subject_text.is_empty() || object_text.is_empty() {
            return;
        }
        let Some((predicate, kind, predicate_notes)) = self.resolve_predicate(predicate_text) else {
            return;
        };
        let subject = self.resolve_entity(subject_text);
        if subject.canonical.is_empty() {
            return;
        }
        let object = self.resolve_entity(object_text);
        if object.canonical.is_empty() {
            return;
        }
        let mut notes = subject.notes;
        notes.extend(predicate_notes);
        notes.extend(object.notes);
        if !subject.resolved {
            notes.push(format!("实体「{}」尚未收录（按未登录实体处理）", subject.canonical));
        }
        let object_is_literal = !object.resolved && looks_literal(object_text);
        props.push(Proposition {
            kind,
            subject: subject.canonical,
            predicate,
            object: object.canonical,
            negative,
            pattern,
            subject_resolved: subject.resolved,
            object_is_literal,
            notes,
        });
    }

    fn parse_type_membership(&self, body: &str, negative: bool, props: &mut Vec<Proposition>) {
        let Some(pos_shi) = body.find('是') else {
            return;
        };
        let subject_text = body[..pos_shi].trim();
        let object_text = body[pos_shi + '是'.len_utf8()..].trim();
        if subject_text.is_empty()
            || object_text.is_empty()
            || subject_text.contains('的')
            || object_text.contains('的')
            || object_text.contains('是')
        {
            return;
        }
        let object_canonical = self.aligner.canonical_form(object_text);
        if self.aligner.kind_of(&object_canonical) != Some(TermKind::Type) {
            return;
        }
        let subject = self.resolve_entity(subject_text);
        if subject.canonical.is_empty() {
            return;
        }
        let mut notes = subject.notes.clone();
        if !subject.resolved {
            notes.push(format!("实体「{}」尚未收录（按未登录实体处理）", subject.canonical));
        }
        props.push(Proposition {
            kind: PropositionKind::TypeMembership,
            subject: subject.canonical,
            predicate: "是".to_string(),
            object: object_canonical,
            negative,
            pattern: "S是T（类型断言）",
            subject_resolved: subject.resolved,
            object_is_literal: false,
            notes,
        });
    }

    fn parse_predicate_infix(&self, body: &str, negative: bool, props: &mut Vec<Proposition>) {
        for (surface, predicate) in self.surfaces.iter() {
            let Some(pos) = body.find(surface.as_str()) else {
                continue;
            };
            if pos == 0 {
                continue;
            }
            let mut subject_text = body[..pos].trim();
            if let Some(rest) = subject_text.strip_suffix('的') {
                subject_text = rest.trim();
            }
            let object_text = body[pos + surface.len()..].trim();
            if subject_text.is_empty()
                || object_text.is_empty()
                || subject_text.contains('的')
                || subject_text.contains('是')
                || object_text.contains('的')
                || object_text.contains('是')
            {
                continue;
            }
            let subject = self.resolve_entity(subject_text);
            if subject.canonical.is_empty() {
                continue;
            }
            let object = self.resolve_entity(object_text);
            if object.canonical.is_empty() {
                continue;
            }
            let kind = match self.aligner.kind_of(&self.aligner.canonical_form(predicate)) {
                Some(TermKind::Attribute) => PropositionKind::Attribute,
                _ => PropositionKind::Relation,
            };
            let mut notes = subject.notes;
            notes.push(format!("中文表述：{} = {}", surface, predicate));
            notes.extend(object.notes);
            if !subject.resolved {
                notes.push(format!("实体「{}」尚未收录（按未登录实体处理）", subject.canonical));
            }
            let object_is_literal = !object.resolved && looks_literal(object_text);
            props.push(Proposition {
                kind,
                subject: subject.canonical,
                predicate: predicate.clone(),
                object: object.canonical,
                negative,
                pattern: "S P O（谓词夹在实体之间）",
                subject_resolved: subject.resolved,
                object_is_literal,
                notes,
            });
            return;
        }
    }
}

fn looks_literal(text: &str) -> bool {
    text.chars()
        .any(|c| c.is_ascii_digit() || matches!(c, '岁' | '元' | '米' | '年' | '%'))
}

fn positive_proof(prop: &Proposition, knowledge: &Knowledge<'_>) -> Option<(ProofStep, f64)> {
    match prop.kind {
        PropositionKind::Relation => {
            if knowledge
                .predictor
                .has_triple(&prop.subject, &prop.predicate, &prop.object)
            {
                return Some((
                    ProofStep::new(
                        ProofKind::DirectFact,
                        format!(
                            "知识图谱存储三元组（{}，{}，{}）",
                            prop.subject, prop.predicate, prop.object
                        ),
                    ),
                    1.0,
                ));
            }
            if knowledge
                .predictor
                .has_triple(&prop.object, &prop.predicate, &prop.subject)
                && knowledge.predictor.is_relation_symmetric(&prop.predicate)
            {
                return Some((
                    ProofStep::new(
                        ProofKind::InverseRelation,
                        format!(
                            "关系「{}」在图谱中呈双向（对称），由（{}，{}，{}）反向蕴含",
                            prop.predicate, prop.object, prop.predicate, prop.subject
                        ),
                    ),
                    0.9,
                ));
            }
            let candidates = knowledge.predictor.predict_for_entity(
                &prop.subject,
                Some(&prop.predicate),
                knowledge.max_rule_candidates,
            );
            for prediction in candidates {
                // predict_for_entity 同时匹配主体在左或右的候选：主客体相同的
                // 断言（如「张三的师傅是张三」）必须主体也一致，避免把
                // （李四，师傅，张三）当成 （张三，师傅，张三）的证据
                if prediction.triple.subject == prop.subject
                    && prediction.triple.object == prop.object
                    && prediction.confidence >= knowledge.min_rule_confidence
                {
                    return Some((
                        ProofStep::new(
                            ProofKind::RuleInference,
                            knowledge.predictor.explain(&prediction),
                        ),
                        prediction.confidence,
                    ));
                }
            }
            None
        }
        PropositionKind::Attribute => {
            if knowledge
                .generator
                .has_attribute(&prop.subject, &prop.predicate, &prop.object)
            {
                return Some((
                    ProofStep::new(
                        ProofKind::AttributeRecord,
                        format!(
                            "实体属性记录：{} 的 {} = {}",
                            prop.subject, prop.predicate, prop.object
                        ),
                    ),
                    1.0,
                ));
            }
            None
        }
        PropositionKind::TypeMembership => {
            if knowledge.generator.entity_type_of(&prop.subject) == Some(prop.object.as_str()) {
                return Some((
                    ProofStep::new(
                        ProofKind::TypeAssertion,
                        format!("类型断言：{} 的类型是 {}", prop.subject, prop.object),
                    ),
                    1.0,
                ));
            }
            None
        }
    }
}

fn known_values(prop: &Proposition, knowledge: &Knowledge<'_>) -> Vec<String> {
    match prop.kind {
        PropositionKind::Relation => knowledge
            .predictor
            .objects_of(&prop.subject, &prop.predicate),
        PropositionKind::Attribute => knowledge
            .generator
            .attribute_values(&prop.subject, &prop.predicate),
        PropositionKind::TypeMembership => knowledge
            .generator
            .entity_type_of(&prop.subject)
            .map(|t| vec![t.to_string()])
            .unwrap_or_default(),
    }
}

fn entity_unknown(prop: &Proposition, knowledge: &Knowledge<'_>) -> bool {
    !prop.subject_resolved
        && !knowledge.predictor.participates(&prop.subject)
        && knowledge.generator.entity_type_of(&prop.subject).is_none()
}

pub fn entail(proposition: &Proposition, knowledge: &Knowledge<'_>) -> EntailmentResult {
    let positive = positive_proof(proposition, knowledge);

    if proposition.negative {
        if let Some((step, confidence)) = positive {
            return EntailmentResult {
                proposition: proposition.clone(),
                verdict: Verdict::Contradicted,
                confidence,
                proof: vec![step],
                counter_evidence: vec![format!(
                    "知识图谱肯定地记录了该事实（{} {} {}）",
                    proposition.subject, proposition.predicate, proposition.object
                )],
                reason: "否定形式与已确认事实冲突".to_string(),
            };
        }
        // 仅在需要封闭世界假设证据时才收集图谱取值
        let values = known_values(proposition, knowledge);
        if !values.is_empty() && !values.iter().any(|value| value == &proposition.object) {
            return EntailmentResult {
                proposition: proposition.clone(),
                verdict: Verdict::Entailed,
                confidence: 0.85,
                proof: vec![ProofStep::new(
                    ProofKind::ClosedWorld,
                    format!(
                        "「{}」的「{}」在知识图谱中仅有记录：{}；均非「{}」，按封闭世界假设否定成立",
                        proposition.subject,
                        proposition.predicate,
                        values.join("、"),
                        proposition.object
                    ),
                )],
                counter_evidence: Vec::new(),
                reason: String::new(),
            };
        }
        return EntailmentResult {
            proposition: proposition.clone(),
            verdict: Verdict::Unknown,
            confidence: 0.0,
            proof: Vec::new(),
            counter_evidence: Vec::new(),
            reason: if entity_unknown(proposition, knowledge) {
                format!("实体「{}」尚未收录，无法判断否定陈述", proposition.subject)
            } else {
                format!(
                    "知识图谱没有「{} 的 {}」的可用记录，无法判断否定陈述",
                    proposition.subject, proposition.predicate
                )
            },
        };
    }

    if let Some((step, confidence)) = positive {
        return EntailmentResult {
            proposition: proposition.clone(),
            verdict: Verdict::Entailed,
            confidence,
            proof: vec![step],
            counter_evidence: Vec::new(),
            reason: String::new(),
        };
    }

    let values = known_values(proposition, knowledge);
    if !values.is_empty() {
        let mut counter: Vec<String> = values
            .iter()
            .map(|value| {
                if proposition.kind == PropositionKind::TypeMembership {
                    format!("知识图谱记录「{}」的类型是「{}」", proposition.subject, value)
                } else {
                    format!(
                        "知识图谱记录「{}」的「{}」是「{}」",
                        proposition.subject, proposition.predicate, value
                    )
                }
            })
            .collect();
        if proposition.object_is_literal {
            counter.push(format!(
                "陈述值「{}」被识别为字面量，与图谱记录不一致",
                proposition.object
            ));
        }
        return EntailmentResult {
            proposition: proposition.clone(),
            verdict: Verdict::Contradicted,
            confidence: 0.8,
            proof: vec![ProofStep::new(
                ProofKind::ClosedWorld,
                "肯定形式在图中找不到证据，但图谱对同一主体-谓词有相反记录（封闭世界假设）"
                    .to_string(),
            )],
            counter_evidence: counter,
            reason: "与知识图谱已记录的取值不符".to_string(),
        };
    }

    let reason = if entity_unknown(proposition, knowledge) {
        format!("实体「{}」尚未收录，无法判定", proposition.subject)
    } else if proposition.kind == PropositionKind::Relation
        && !knowledge.predictor.has_predicate(&proposition.predicate)
    {
        format!("谓词「{}」尚未收录，无法判定", proposition.predicate)
    } else {
        format!(
            "知识图谱中没有「{} 的 {}」的相关记录，证据不足",
            proposition.subject, proposition.predicate
        )
    };
    EntailmentResult {
        proposition: proposition.clone(),
        verdict: Verdict::Unknown,
        confidence: 0.0,
        proof: Vec::new(),
        counter_evidence: Vec::new(),
        reason,
    }
}

pub fn render_results(results: &[EntailmentResult]) -> String {
    let mut output = String::new();
    for (index, result) in results.iter().enumerate() {
        output.push_str(&format!(
            "  {}. 逻辑形式：{}\n",
            index + 1,
            result.proposition.render()
        ));
        if !result.proposition.notes.is_empty() {
            output.push_str(&format!(
                "     解析证据：{}\n",
                result.proposition.notes.join("；")
            ));
        }
        output.push_str(&format!("     判定：{}", result.verdict.label()));
        if result.verdict != Verdict::Unknown {
            output.push_str(&format!("，置信度 {:.2}", result.confidence));
        }
        output.push('\n');
        for step in &result.proof {
            output.push_str(&format!("     证据[{}]：{}\n", step.kind.label(), step.detail));
        }
        for evidence in &result.counter_evidence {
            output.push_str(&format!("     反证：{}\n", evidence));
        }
        if result.verdict == Verdict::Unknown && !result.reason.is_empty() {
            output.push_str(&format!("     原因：{}\n", result.reason));
        }
    }
    output.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        aligner: OntologyAligner,
        linker: EntityLinker,
        predictor: LinkPredictor,
        generator: SymbolicGenerator,
    }

    impl Fixture {
        fn new() -> Self {
            let mut aligner = OntologyAligner::new(HashMap::new(), 0.72);
            let mut linker = EntityLinker::new();
            let mut predictor = LinkPredictor::new();
            let mut generator = SymbolicGenerator::new(6, 0.3);

            aligner.register_term("人物", TermKind::Type);
            aligner.register_term("地点", TermKind::Type);
            for (name, etype) in [("张三", "人物"), ("李四", "人物"), ("王五", "人物"), ("北京", "地点")] {
                aligner.register_term(name, TermKind::Entity);
                linker.register_entity(name, Some(etype));
                generator.observe_entity(name, etype, &[]);
            }
            generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
            aligner.register_term("年龄", TermKind::Attribute);

            for (s, p, o) in [("张三", "师傅", "李四"), ("李四", "师傅", "王五"), ("张三", "出生地", "北京")] {
                aligner.observe_triple(s, p, o);
                predictor.add_triple(s, p, o);
                linker.observe_triple(s, p, o);
                generator.observe_triple(s, p, o);
            }
            Self {
                aligner,
                linker,
                predictor,
                generator,
            }
        }

        fn parse(&self, text: &str) -> Vec<Proposition> {
            PropositionParser::new(&self.aligner, &self.linker, self.generator.predicate_surfaces())
                .parse(text)
        }

        fn entail_first(&self, text: &str) -> EntailmentResult {
            let props = self.parse(text);
            assert!(!props.is_empty(), "「{}」应能形式化为命题", text);
            entail(
                &props[0],
                &Knowledge {
                    predictor: &self.predictor,
                    generator: &self.generator,
                    min_rule_confidence: 0.5,
                    max_rule_candidates: 16,
                },
            )
        }
    }

    #[test]
    fn test_parse_subject_predicate_object_pattern() {
        let fixture = Fixture::new();
        let props = fixture.parse("张三的师傅是李四。");
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].kind, PropositionKind::Relation);
        assert_eq!(props[0].subject, "张三");
        assert_eq!(props[0].predicate, "师傅");
        assert_eq!(props[0].object, "李四");
        assert!(!props[0].negative);
    }

    #[test]
    fn test_parse_inverse_pattern() {
        let fixture = Fixture::new();
        let props = fixture.parse("李四是张三的师傅");
        assert!(props.iter().any(|p| p.subject == "张三"
            && p.predicate == "师傅"
            && p.object == "李四"
            && p.pattern == "O是S的P"));
    }

    #[test]
    fn test_parse_type_membership_and_negation() {
        let fixture = Fixture::new();
        let props = fixture.parse("张三是人物");
        assert_eq!(props[0].kind, PropositionKind::TypeMembership);
        assert_eq!(props[0].object, "人物");

        let negative = fixture.parse("张三不是李四的师傅");
        assert!(negative.iter().any(|p| p.negative));
    }

    #[test]
    fn test_parse_questions_are_not_propositions() {
        let fixture = Fixture::new();
        assert!(fixture.parse("张三的师傅是谁").is_empty());
        assert!(fixture.parse("张三的年龄是多少").is_empty());
    }

    #[test]
    fn test_a_not_a_questions_are_not_propositions() {
        let fixture = Fixture::new();
        assert!(fixture.parse("张三是不是李四的师傅").is_empty());
        assert!(fixture.parse("张三有没有师傅").is_empty());
        assert!(fixture.parse("张三是否是李四的师傅").is_empty());
    }

    #[test]
    fn test_bare_bu_inside_word_is_not_negation() {
        let mut fixture = Fixture::new();
        fixture.aligner.register_term("合金", TermKind::Type);
        fixture.aligner.register_term("不锈钢", TermKind::Entity);
        fixture.linker.register_entity("不锈钢", Some("合金"));
        let props = fixture.parse("不锈钢是合金");
        assert_eq!(props.len(), 1, "「不锈钢」中的「不」不应被当成否定：{:?}", props);
        assert!(!props[0].negative);
        assert_eq!(props[0].subject, "不锈钢");
        assert_eq!(props[0].object, "合金");
    }

    #[test]
    fn test_bare_bu_before_predicate_is_negation() {
        let mut fixture = Fixture::new();
        fixture.aligner.observe_triple("张三", "knows", "李四");
        fixture.predictor.add_triple("张三", "knows", "李四");
        fixture.generator.observe_triple("张三", "knows", "李四");
        let props = fixture.parse("张三不认识李四");
        assert!(
            props.iter().any(|p| p.predicate == "knows" && p.negative),
            "「不认识」应解析为否定命题：{:?}",
            props.iter().map(Proposition::render).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_question_word_substrings_in_statements() {
        let mut fixture = Fixture::new();
        fixture.aligner.register_term("非洲国家", TermKind::Type);
        fixture.aligner.register_term("几内亚", TermKind::Entity);
        let props = fixture.parse("几内亚是非洲国家");
        assert_eq!(props.len(), 1, "「几内亚」是陈述句主语，不应被当成疑问句");
        assert_eq!(props[0].subject, "几内亚");

        fixture.aligner.register_term("数学分支", TermKind::Type);
        fixture.aligner.register_term("几何", TermKind::Entity);
        let props = fixture.parse("几何是数学分支");
        assert_eq!(props.len(), 1, "「几何」作主语时是陈述句");
    }

    #[test]
    fn test_parse_predicate_infix_with_lexical_surface() {
        let mut fixture = Fixture::new();
        fixture.aligner.observe_triple("张三", "knows", "李四");
        fixture.predictor.add_triple("张三", "knows", "李四");
        fixture.generator.observe_triple("张三", "knows", "李四");
        let props = fixture.parse("张三认识李四");
        assert!(
            props.iter().any(|p| p.predicate == "knows" && p.object == "李四"),
            "中文动词「认识」应映射回规范谓词 knows：{:?}",
            props.iter().map(Proposition::render).collect::<Vec<_>>()
        );
        let result = fixture.entail_first("张三认识李四");
        assert_eq!(result.verdict, Verdict::Entailed, "实际：{:?}", result);
    }

    #[test]
    fn test_entail_direct_and_attribute_and_type() {
        let fixture = Fixture::new();
        assert_eq!(fixture.entail_first("张三的师傅是李四").verdict, Verdict::Entailed);
        assert_eq!(fixture.entail_first("张三的年龄是30").verdict, Verdict::Entailed);
        assert_eq!(fixture.entail_first("张三是人物").verdict, Verdict::Entailed);
    }

    #[test]
    fn test_entail_contradiction_with_counter_evidence() {
        let fixture = Fixture::new();
        let result = fixture.entail_first("张三的师傅是王五");
        assert_eq!(result.verdict, Verdict::Contradicted);
        assert!(result.counter_evidence.iter().any(|e| e.contains("李四")));
    }

    #[test]
    fn test_entail_negative_statement_by_closed_world() {
        let fixture = Fixture::new();
        let result = fixture.entail_first("张三的师傅不是王五");
        assert_eq!(result.verdict, Verdict::Entailed);
        assert_eq!(result.proof[0].kind, ProofKind::ClosedWorld);

        let conflict = fixture.entail_first("张三的师傅不是李四");
        assert_eq!(conflict.verdict, Verdict::Contradicted);
    }

    #[test]
    fn test_entail_unknown_entity_has_reason() {
        let fixture = Fixture::new();
        let result = fixture.entail_first("赵六的师傅是李四");
        assert_eq!(result.verdict, Verdict::Unknown);
        assert!(result.reason.contains("尚未收录"), "实际原因：{}", result.reason);
    }

    #[test]
    fn test_entail_inverse_symmetric_relation() {
        let mut fixture = Fixture::new();
        // 双向配偶 + 单向另一对：使「配偶」被判定为对称关系
        for (s, o) in [("张三", "王五"), ("王五", "张三"), ("李四", "北京")] {
            fixture.aligner.observe_triple(s, "配偶", o);
            fixture.predictor.add_triple(s, "配偶", o);
        }
        let result = fixture.entail_first("北京的配偶是李四");
        assert_eq!(result.verdict, Verdict::Entailed, "实际：{:?}", result);
        assert_eq!(result.proof[0].kind, ProofKind::InverseRelation);
    }

    #[test]
    fn test_entail_via_mined_rule() {
        // 训练样例提供 师祖 的头部对（甲→丙），规则 师傅∘师傅 ⇒ 师祖 被挖掘；
        // 图谱缺失 张三的师祖(=王五) 的直接记录，蕴含应由规则推断补齐。
        let mut fixture = Fixture::new();
        for (s, p, o) in [
            ("甲", "师傅", "乙"),
            ("乙", "师傅", "丙"),
            ("甲", "师祖", "丙"),
        ] {
            fixture.aligner.observe_triple(s, p, o);
            fixture.predictor.add_triple(s, p, o);
        }
        fixture.predictor.mine_rules(1, 0.5);
        assert!(
            fixture
                .predictor
                .predict_for_entity("张三", Some("师祖"), 16)
                .iter()
                .any(|p| p.triple.object == "王五"),
            "规则应能推出 张三的师祖是王五"
        );
        let result = fixture.entail_first("张三的师祖是王五");
        assert_eq!(result.verdict, Verdict::Entailed, "实际：{:?}", result);
        assert_eq!(result.proof[0].kind, ProofKind::RuleInference);
        assert!(result.proof[0].detail.contains("师傅"));
    }

    #[test]
    fn test_render_results_is_explainable() {
        let fixture = Fixture::new();
        let results = vec![fixture.entail_first("张三的师傅是李四")];
        let text = render_results(&results);
        assert!(text.contains("relation(张三, 师傅, 李四)"));
        assert!(text.contains("直接事实"));
        assert!(text.contains("蕴含"));
    }
}

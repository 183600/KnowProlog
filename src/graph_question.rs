use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::conversational_graph::{EdgePattern, GraphPattern};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphQuestionKind {
    Conjunction,
    Existential,
    Equality,
    SharedObject,
}

impl GraphQuestionKind {
    pub fn label(&self) -> &'static str {
        match self {
            GraphQuestionKind::Conjunction => "合取约束",
            GraphQuestionKind::Existential => "存在量化",
            GraphQuestionKind::Equality => "等值连接",
            GraphQuestionKind::SharedObject => "共享对象",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledGraphQuestion {
    pub kind: GraphQuestionKind,
    pub pattern: GraphPattern,
    pub subject_variables: Vec<String>,
    pub focus_predicate: Option<String>,
    /// 「有多少 / 几个」类问句只要计数，不展开结果行。
    #[serde(default)]
    pub count_only: bool,
}

impl CompiledGraphQuestion {
    pub fn asks_for_entity(&self) -> bool {
        !self.pattern.select.is_empty()
    }

    pub fn asks_for_count(&self) -> bool {
        self.count_only
    }
}

/// 编译失败但结构可辨认时的缺口：哪些分句已经读懂，哪些没能解析。
///
/// 这不是「放宽」——它不猜测用户原意，只把「部分理解」摊开，避免把
/// 读不懂的合取句静默交给其他管线后给出另一份答案。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileGap {
    pub complete_clauses: Vec<String>,
    pub unresolved_clauses: Vec<String>,
}

impl CompileGap {
    pub fn render(&self) -> String {
        format!(
            "我只读懂了「{}」这部分条件，无法解析「{}」。请确认其中的实体、类型或谓词是否已录入知识库，或换一种说法。",
            self.complete_clauses.join("；"),
            self.unresolved_clauses.join("；")
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SurfaceMatch {
    start: usize,
    surface_len: usize,
    is_type: bool,
}

impl SurfaceMatch {
    fn end(&self) -> usize {
        self.start + self.surface_len
    }

    fn canonical<'a>(&self, lexicon: &'a PatternLexicon, surface: &'a str) -> &'a str {
        if self.is_type {
            lexicon
                .types
                .get(surface)
                .map(String::as_str)
                .unwrap_or(surface)
        } else {
            lexicon
                .entities
                .get(surface)
                .map(String::as_str)
                .unwrap_or(surface)
        }
    }
}

struct ConjunctionClause {
    predicate: String,
    object: String,
    object_is_type: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PatternLexicon {
    entities: BTreeMap<String, String>,
    predicates: BTreeMap<String, String>,
    types: BTreeMap<String, String>,
}

impl PatternLexicon {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_entity(&mut self, surface: impl Into<String>, canonical: impl Into<String>) {
        self.entities.insert(surface.into(), canonical.into());
    }

    pub fn add_predicate(&mut self, surface: impl Into<String>, canonical: impl Into<String>) {
        self.predicates.insert(surface.into(), canonical.into());
    }

    pub fn add_type(&mut self, surface: impl Into<String>, canonical: impl Into<String>) {
        self.types.insert(surface.into(), canonical.into());
    }

    pub fn is_known_entity(&self, surface: &str) -> bool {
        self.entities.contains_key(surface)
    }

    pub fn is_known_predicate(&self, surface: &str) -> bool {
        self.predicates.contains_key(surface)
    }

    pub fn is_known_type(&self, surface: &str) -> bool {
        self.types.contains_key(surface)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    pub fn resolve_object(&self, text: &str) -> Option<(String, bool)> {
        let found = self.longest_object(text)?;
        let surface = text.get(found.start..found.end())?;
        Some((found.canonical(self, surface).to_string(), found.is_type))
    }

    /// 解析追问里的限定词：先按实体/类型词面精确命中，再允许唯一类型前缀
    ///（「中国」→「中国人」、「科幻」→「科幻片」）。歧义时返回 None，
    /// 不替用户猜测限定范围。
    pub fn resolve_refinement(&self, text: &str) -> Option<(String, bool)> {
        let mut text = text.trim();
        text = text.trim_start_matches(['是', '有', '要', ' ']);
        text = text.trim_end_matches(['的', '呢', '吧', ' ']);
        if text.is_empty() {
            return None;
        }
        if let Some(found) = self.resolve_object(text) {
            return Some(found);
        }
        let mut matched: BTreeSet<&str> = BTreeSet::new();
        for (surface, canonical) in &self.types {
            if surface.contains(text) || canonical.contains(text) || text.contains(surface) {
                matched.insert(canonical.as_str());
            }
        }
        if matched.len() == 1 {
            let canonical = matched.into_iter().next()?;
            return Some((canonical.to_string(), true));
        }
        None
    }

    fn longest_type(&self, text: &str) -> Option<SurfaceMatch> {
        longest_surface(&self.types, text, true)
    }

    fn longest_object(&self, text: &str) -> Option<SurfaceMatch> {
        let entity = longest_surface(&self.entities, text, false);
        let entity_type = longest_surface(&self.types, text, true);
        match (entity, entity_type) {
            (Some(left), Some(right)) => {
                if right.surface_len > left.surface_len {
                    Some(right)
                } else if left.surface_len > right.surface_len {
                    Some(left)
                } else if left.start <= right.start {
                    Some(left)
                } else {
                    Some(right)
                }
            }
            (Some(found), None) | (None, Some(found)) => Some(found),
            (None, None) => None,
        }
    }

    fn longest_predicate(&self, text: &str) -> Option<(usize, SurfaceMatch, String)> {
        self.longest_predicate_exact(text)
            .or_else(|| self.longest_predicate_fallback(text))
    }

    fn longest_predicate_exact(&self, text: &str) -> Option<(usize, SurfaceMatch, String)> {
        let mut best: Option<(usize, SurfaceMatch, String)> = None;
        for (surface, canonical) in &self.predicates {
            if surface.is_empty() {
                continue;
            }
            if let Some(start) = text.find(surface.as_str()) {
                let candidate = SurfaceMatch {
                    start,
                    surface_len: surface.len(),
                    is_type: false,
                };
                let better = match &best {
                    None => true,
                    Some((best_start, best_match, _)) => {
                        start < *best_start
                            || (start == *best_start
                                && candidate.surface_len > best_match.surface_len)
                    }
                };
                if better {
                    best = Some((start, candidate, canonical.clone()));
                }
            }
        }
        best
    }

    fn longest_predicate_fallback(&self, text: &str) -> Option<(usize, SurfaceMatch, String)> {
        let mut best: Option<(usize, SurfaceMatch, String)> = None;
        for (surface, canonical) in &self.predicates {
            let Some(prefix) = surface
                .strip_suffix('地')
                .or_else(|| surface.strip_suffix('于'))
            else {
                continue;
            };
            if prefix.chars().count() < 2 {
                continue;
            }
            if let Some(start) = text.find(prefix) {
                let candidate = SurfaceMatch {
                    start,
                    surface_len: prefix.len(),
                    is_type: false,
                };
                let better = match &best {
                    None => true,
                    Some((best_start, best_match, _)) => {
                        start < *best_start
                            || (start == *best_start
                                && candidate.surface_len > best_match.surface_len)
                    }
                };
                if better {
                    best = Some((start, candidate, canonical.clone()));
                }
            }
        }
        best
    }

    fn entity_mentions(&self, text: &str) -> Vec<SurfaceMatch> {
        let mut found = Vec::new();
        for surface in self.entities.keys() {
            if surface.is_empty() {
                continue;
            }
            let mut search_from = 0usize;
            while let Some(offset) = text[search_from..].find(surface.as_str()) {
                let start = search_from + offset;
                found.push(SurfaceMatch {
                    start,
                    surface_len: surface.len(),
                    is_type: false,
                });
                search_from = start + surface.len().max(1);
            }
        }
        found.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then(right.surface_len.cmp(&left.surface_len))
        });
        let mut chosen: Vec<SurfaceMatch> = Vec::new();
        for candidate in found {
            let overlaps = chosen
                .last()
                .map(|last| candidate.start < last.end())
                .unwrap_or(false);
            if !overlaps {
                chosen.push(candidate);
            }
        }
        chosen
    }

    pub fn compile(&self, question: &str) -> Option<CompiledGraphQuestion> {
        let normalized = normalize_question(question)?;
        if let Some(compiled) = self.compile_equality(&normalized) {
            return Some(compiled);
        }
        if let Some(compiled) = self.compile_existential(&normalized) {
            return Some(compiled);
        }
        if let Some(compiled) = self.compile_conjunction(&normalized) {
            return Some(compiled);
        }
        self.compile_shared_object(&normalized)
    }

    /// 解析合取问句中的一个条件分句；只接受谓词 + 已知实体/类型 + 纯外壳。
    fn parse_conjunction_clause(&self, clause: &str) -> Option<ConjunctionClause> {
        let clause = trim_clause(clause);
        if clause.is_empty() {
            return None;
        }
        let (_, predicate_match, predicate) = self.longest_predicate(&clause)?;
        let after = &clause[predicate_match.end()..];
        let object = self.longest_object(after)?;
        let surface = &after[object.start..object.end()];
        let canonical = object.canonical(self, surface).to_string();
        let tail = trim_clause(&after[object.end()..]);
        if !tail.is_empty() && !self.is_scaffolding(&tail) {
            return None;
        }
        Some(ConjunctionClause {
            predicate,
            object: canonical,
            object_is_type: object.is_type,
        })
    }

    /// 结构化问句编译失败时，返回可渲染的「部分理解」缺口。
    /// 只处理合取句：这是最容易出现「丢掉一个分句也能凑出答案」的形状。
    pub fn compile_gap(&self, question: &str) -> Option<CompileGap> {
        if self.compile(question).is_some() {
            return None;
        }
        let normalized = normalize_question(question)?;
        let clauses = split_conjunctive(&normalized)?;
        let mut complete_clauses = Vec::new();
        let mut unresolved_clauses = Vec::new();
        for clause in clauses {
            let clause = trim_clause(&clause);
            if clause.is_empty() || self.is_scaffolding(&clause) {
                continue;
            }
            if self.parse_conjunction_clause(&clause).is_some() {
                complete_clauses.push(clause);
            } else {
                unresolved_clauses.push(clause);
            }
        }
        if complete_clauses.is_empty() || unresolved_clauses.is_empty() {
            return None;
        }
        Some(CompileGap {
            complete_clauses,
            unresolved_clauses,
        })
    }

    fn compile_conjunction(&self, question: &str) -> Option<CompiledGraphQuestion> {
        let clauses = split_conjunctive(question)?;
        let subject = "x".to_string();
        let mut edges = Vec::new();
        let mut parsed_clauses = 0usize;
        let mut unparsed = Vec::new();
        for clause in &clauses {
            let clause = trim_clause(clause);
            if clause.is_empty() {
                continue;
            }
            let Some(parts) = self.parse_conjunction_clause(&clause) else {
                unparsed.push(clause);
                continue;
            };
            if parts.object_is_type {
                let object_var = format!("v{}", edges.len());
                edges.push(EdgePattern::var_var(&subject, parts.predicate, &object_var));
                edges.push(EdgePattern::var_const(&object_var, "type", parts.object));
            } else {
                edges.push(EdgePattern::var_const(
                    &subject,
                    parts.predicate,
                    parts.object,
                ));
            }
            parsed_clauses += 1;
        }
        // 只允许丢掉纯问句外壳（如「有多少人」）；含有已知实体/类型或
        // 谓词的残句意味着「有一部分没看懂」，此时宁可放弃编译，也不能
        // 把用户没问的子集当成完整答案。
        if unparsed.iter().any(|clause| !self.is_scaffolding(clause)) {
            return None;
        }
        if parsed_clauses < 2 {
            return None;
        }
        if !contains_question_word(question) {
            return None;
        }
        let pattern = self.finish_pattern(edges, vec![subject.clone()])?;
        Some(CompiledGraphQuestion {
            kind: GraphQuestionKind::Conjunction,
            pattern,
            subject_variables: vec![subject],
            focus_predicate: None,
            count_only: contains_count_word(question),
        })
    }

    fn compile_existential(&self, question: &str) -> Option<CompiledGraphQuestion> {
        let (rest, marker) = find_marker(
            question,
            &["有没有", "是否有", "存在", "有多少", "有几个", "有几部"],
        )?;
        // 「有多少人既…又…」是计数合取句：前件里的类型词容易被误读成
        // 被量化的主体类型，因此把整句交给合取编译器。
        if marker == "有多少" && split_conjunctive(question).is_some() {
            return None;
        }
        let rest = trim_clause(rest);
        if rest.is_empty() {
            return None;
        }
        let item = "item".to_string();
        let mut edges = Vec::new();
        let intro_end = rest
            .find(['，', ','])
            .or_else(|| {
                self.longest_predicate(&rest)
                    .map(|(_, predicate, _)| predicate.start)
            })
            .unwrap_or(rest.len());
        let intro = &rest[..intro_end];
        if let Some(item_type) = self.longest_type(intro) {
            let surface = &intro[item_type.start..item_type.end()];
            let canonical = item_type.canonical(self, surface).to_string();
            edges.push(EdgePattern::var_const(&item, "type", canonical));
        }
        let clauses: Vec<String> = rest
            .split(['，', ',', '。', ';', '；'])
            .map(str::to_string)
            .collect();
        for clause in clauses {
            let clause = trim_clause(&clause);
            if clause.is_empty() {
                continue;
            }
            let Some((_, predicate_match, predicate)) = self.longest_predicate(&clause) else {
                if self.longest_type(&clause).is_some() {
                    continue;
                }
                return None;
            };
            let after = &clause[predicate_match.end()..];
            let after = after.strip_prefix("是").unwrap_or(after);
            let after = after.strip_prefix("为").unwrap_or(after);
            let object = self.longest_object(after)?;
            let surface = &after[object.start..object.end()];
            let canonical = object.canonical(self, surface).to_string();
            let tail = trim_clause(&after[object.end()..]);
            if !tail.is_empty() && !self.is_scaffolding(&tail) {
                return None;
            }
            if object.is_type {
                let object_var = format!("o{}", edges.len());
                edges.push(EdgePattern::var_var(&item, predicate, &object_var));
                edges.push(EdgePattern::var_const(&object_var, "type", canonical));
            } else {
                edges.push(EdgePattern::var_const(&item, predicate, canonical));
            }
        }
        if edges.iter().all(|edge| edge.predicate == "type") {
            return None;
        }
        let pattern = self.finish_pattern(edges, vec![item.clone()])?;
        Some(CompiledGraphQuestion {
            kind: GraphQuestionKind::Existential,
            pattern,
            subject_variables: vec![item],
            focus_predicate: None,
            count_only: contains_count_word(question),
        })
    }

    fn compile_equality(&self, question: &str) -> Option<CompiledGraphQuestion> {
        let (position, marker) =
            find_position(question, &["同一个", "同一", "共同", "相同", "一样", "一致"])?;
        let left = &question[..position];
        let right = &question[position + marker.len()..];
        let mentions = self.entity_mentions(left);
        if mentions.is_empty() {
            return None;
        }
        // 所有列出的实体都必须进入同一组等值边。「张三、李四和王五的
        // 出生地相同吗」不能只比较前两个人而静默丢掉王五。
        let canonicals: Vec<String> = mentions
            .iter()
            .map(|mention| {
                let surface = &left[mention.start..mention.end()];
                mention.canonical(self, surface).to_string()
            })
            .collect();
        let asks_for_other = mentions.len() == 1
            && contains_question_word(&left[mentions[0].end()..]);
        if mentions.len() == 1 && !asks_for_other {
            return None;
        }
        let tail = &left[mentions[mentions.len() - 1].end()..];
        let (_, _, predicate) = self
            .longest_predicate(tail)
            .or_else(|| self.longest_predicate(right))?;
        let value = "value".to_string();
        let mut edges = Vec::new();
        let mut subjects = Vec::new();
        let mut select = vec![value.clone()];
        if asks_for_other {
            let variable = "person2".to_string();
            edges.push(EdgePattern::const_var(&canonicals[0], &predicate, &value));
            edges.push(EdgePattern::var_var(&variable, &predicate, &value));
            subjects.push(canonicals[0].clone());
            subjects.push(variable.clone());
            select = vec![variable];
        } else {
            for canonical in &canonicals {
                edges.push(EdgePattern::const_var(canonical, &predicate, &value));
            }
            subjects.extend(canonicals.iter().cloned());
        }
        let pattern = self.finish_pattern(edges, select)?;
        Some(CompiledGraphQuestion {
            kind: GraphQuestionKind::Equality,
            pattern,
            subject_variables: subjects,
            focus_predicate: Some(predicate),
            count_only: contains_count_word(question),
        })
    }

    fn compile_shared_object(&self, question: &str) -> Option<CompiledGraphQuestion> {
        let position = question.find('都')?;
        let left = &question[..position];
        let right = &question[position + '都'.len_utf8()..];
        if !contains_question_word(right) {
            return None;
        }
        let mentions = self.entity_mentions(left);
        if mentions.len() < 2 {
            return None;
        }
        let (_, _, predicate) = self.longest_predicate(right)?;
        let object = "object".to_string();
        let edges: Vec<EdgePattern> = mentions
            .iter()
            .map(|mention| {
                let surface = &left[mention.start..mention.end()];
                let canonical = mention.canonical(self, surface).to_string();
                EdgePattern::const_var(canonical, &predicate, &object)
            })
            .collect();
        let subjects = mentions
            .iter()
            .map(|mention| {
                let surface = &left[mention.start..mention.end()];
                mention.canonical(self, surface).to_string()
            })
            .collect();
        let pattern = self.finish_pattern(edges, vec![object])?;
        Some(CompiledGraphQuestion {
            kind: GraphQuestionKind::SharedObject,
            pattern,
            subject_variables: subjects,
            focus_predicate: Some(predicate),
            count_only: contains_count_word(question),
        })
    }

    fn is_scaffolding(&self, clause: &str) -> bool {
        if self.longest_predicate(clause).is_some()
            || self.longest_type(clause).is_some()
            || !self.entity_mentions(clause).is_empty()
        {
            return false;
        }
        contains_question_word(clause)
            || ["那", "那么", "呢", "吧", "的"].contains(&clause.trim())
    }

    fn finish_pattern(&self, edges: Vec<EdgePattern>, select: Vec<String>) -> Option<GraphPattern> {
        let pattern = GraphPattern::new(edges, select);
        pattern.validate().ok()?;
        Some(pattern)
    }
}

fn longest_surface(
    map: &BTreeMap<String, String>,
    text: &str,
    is_type: bool,
) -> Option<SurfaceMatch> {
    let mut best: Option<SurfaceMatch> = None;
    for surface in map.keys() {
        if surface.is_empty() {
            continue;
        }
        if let Some(start) = text.find(surface.as_str()) {
            let candidate = SurfaceMatch {
                start,
                surface_len: surface.len(),
                is_type,
            };
            let better = match &best {
                None => true,
                Some(current) => {
                    candidate.surface_len > current.surface_len
                        || (candidate.surface_len == current.surface_len
                            && candidate.start < current.start)
                }
            };
            if better {
                best = Some(candidate);
            }
        }
    }
    best
}

fn contains_count_word(text: &str) -> bool {
    ["多少", "几个", "几部", "几位", "几项", "几种"]
        .iter()
        .any(|word| text.contains(word))
}

fn normalize_question(question: &str) -> Option<String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        return None;
    }
    let trimmed = trimmed.trim_end_matches(['？', '?', '！', '!', '。', '.', ' ']);
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn contains_question_word(text: &str) -> bool {
    ["谁", "哪", "什么", "多少", "几"]
        .iter()
        .any(|word| text.contains(word))
}

fn find_position<'a>(text: &'a str, markers: &[&'a str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|position| (position, *marker)))
        .min_by_key(|(position, _)| *position)
}

fn find_marker<'a>(text: &'a str, markers: &[&'a str]) -> Option<(&'a str, &'a str)> {
    let (position, marker) = find_position(text, markers)?;
    Some((&text[position + marker.len()..], marker))
}

fn split_conjunctive(question: &str) -> Option<Vec<String>> {
    const MARKERS: [&str; 5] = ["并且", "而且", "同时", "又", "既"];
    if !MARKERS.iter().any(|marker| question.contains(marker)) {
        return None;
    }
    let mut normalized = question.to_string();
    // 「既然」不是合取标记，先保护起来，避免被拆成「然…」。
    normalized = normalized.replace("既然", "\u{2}");
    for marker in MARKERS {
        normalized = normalized.replace(marker, "\u{1}");
    }
    normalized = normalized.replace('\u{2}', "既然");
    let parts: Vec<String> = normalized
        .split('\u{1}')
        .map(str::to_string)
        .filter(|clause| !trim_clause(clause).is_empty())
        .collect();
    if parts.len() < 2 {
        return None;
    }
    Some(parts)
}

fn trim_clause(clause: &str) -> String {
    let mut text = clause.trim();
    loop {
        let before = text;
        text = text.trim_start_matches([
            '，', ',', '。', '；', ';', '、', ' ', '\t', '\n', '？', '?', '！', '!',
        ]);
        for prefix in [
            "那么", "所以", "然后", "请问", "到底", "究竟", "都", "也", "还", "又", "既", "并且",
            "而且", "同时", "呢",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest;
                break;
            }
        }
        if text == before {
            break;
        }
    }
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversational_graph::{GraphFact, MatchOptions, PatternTerm};

    fn lexicon() -> PatternLexicon {
        let mut lexicon = PatternLexicon::new();
        for entity in [
            "张三",
            "李四",
            "王五",
            "赵六",
            "姚明",
            "刘翔",
            "电影A",
            "电影B",
            "金棕榈",
            "上海",
            "北京",
        ] {
            lexicon.add_entity(entity, entity);
        }
        for (surface, canonical) in [
            ("导演过", "导演"),
            ("导演", "导演"),
            ("主演", "主演"),
            ("认识", "认识"),
            ("获奖", "获奖"),
            ("拿过", "获奖"),
            ("获得", "获奖"),
            ("出生在", "出生地"),
            ("出生地", "出生地"),
        ] {
            lexicon.add_predicate(surface, canonical);
        }
        for entity_type in ["电影", "科幻片", "剧情片", "中国人", "美国人", "Person"]
        {
            lexicon.add_type(entity_type, entity_type);
        }
        lexicon
    }

    #[test]
    fn conjunction_compiles_both_clauses() {
        let compiled = lexicon()
            .compile("谁既导演过科幻片，又拿过金棕榈？")
            .expect("合取问句应可编译");
        assert_eq!(compiled.kind, GraphQuestionKind::Conjunction);
        assert_eq!(compiled.pattern.select, vec!["x".to_string()]);
        assert_eq!(compiled.pattern.edges.len(), 3);
        assert_eq!(compiled.pattern.edges[0].predicate, "导演");
        assert_eq!(compiled.pattern.edges[2].predicate, "获奖");
        assert!(compiled.pattern.edges[1].predicate == "type");
    }

    #[test]
    fn conjunction_runs_over_facts() {
        let compiled = lexicon()
            .compile("谁既导演过科幻片，又拿过金棕榈？")
            .unwrap();
        let facts = vec![
            GraphFact::relation("张三", "导演", "电影A"),
            GraphFact::type_assertion("电影A", "科幻片"),
            GraphFact::relation("张三", "获奖", "金棕榈"),
            GraphFact::relation("李四", "导演", "电影B"),
            GraphFact::type_assertion("电影B", "剧情片"),
            GraphFact::relation("李四", "获奖", "金棕榈"),
        ];
        let outcome = compiled
            .pattern
            .evaluate(&facts, MatchOptions::default())
            .unwrap();
        assert_eq!(outcome.len(), 1);
        assert_eq!(
            outcome.solutions[0].projected(&compiled.pattern.select),
            vec!["张三"]
        );
    }

    #[test]
    fn existential_compiles_item_constraints() {
        let compiled = lexicon()
            .compile("有没有一部电影，导演是中国人，主演是美国人？")
            .expect("存在问句应可编译");
        assert_eq!(compiled.kind, GraphQuestionKind::Existential);
        assert_eq!(compiled.pattern.select, vec!["item".to_string()]);
        assert_eq!(compiled.pattern.edges.len(), 5);
        assert_eq!(compiled.pattern.edges[0].predicate, "type");
        assert_eq!(
            compiled.pattern.edges[0].object,
            PatternTerm::constant("电影")
        );
    }

    #[test]
    fn equality_compiles_shared_value() {
        let compiled = lexicon()
            .compile("姚明和刘翔出生在同一个城市吗？")
            .expect("等值问句应可编译");
        assert_eq!(compiled.kind, GraphQuestionKind::Equality);
        assert_eq!(compiled.pattern.select, vec!["value".to_string()]);
        assert_eq!(compiled.pattern.edges.len(), 2);
        assert_eq!(compiled.pattern.edges[0].predicate, "出生地");
        assert_eq!(compiled.pattern.edges[1].predicate, "出生地");
    }

    #[test]
    fn equality_keeps_every_listed_entity() {
        let compiled = lexicon()
            .compile("张三、李四和王五的出生地相同吗？")
            .expect("多实体等值问句应可编译");
        assert_eq!(compiled.kind, GraphQuestionKind::Equality);
        assert_eq!(compiled.pattern.edges.len(), 3);
        assert_eq!(compiled.subject_variables.len(), 3);
        assert!(compiled
            .pattern
            .edges
            .iter()
            .all(|edge| edge.predicate == "出生地"));
        assert_eq!(
            compiled.pattern.select,
            vec!["value".to_string()],
            "应投影共享的取值"
        );

        let facts = vec![
            GraphFact::relation("张三", "出生地", "上海"),
            GraphFact::relation("李四", "出生地", "上海"),
            GraphFact::relation("王五", "出生地", "北京"),
        ];
        let outcome = compiled
            .pattern
            .evaluate(&facts, MatchOptions::default())
            .unwrap();
        assert!(
            outcome.solutions.is_empty(),
            "第三个人不同城时必须判为不相同，而不是退回只比较前两人"
        );
    }

    #[test]
    fn equality_accepts_shared_attribute_phrasings() {
        let lexicon = lexicon();
        for question in [
            "姚明和刘翔的出生地相同吗？",
            "姚明和刘翔出生在一样的城市吗？",
            "姚明和刘翔的出生地是否一致？",
        ] {
            let compiled = lexicon.compile(question).expect("等值问句应可编译");
            assert_eq!(compiled.kind, GraphQuestionKind::Equality, "{}", question);
            assert_eq!(compiled.pattern.edges.len(), 2, "{}", question);
            assert!(
                compiled
                    .pattern
                    .edges
                    .iter()
                    .all(|edge| edge.predicate == "出生地"),
                "{}",
                question
            );
        }
    }

    #[test]
    fn shared_object_compiles_common_neighbors() {
        let compiled = lexicon()
            .compile("张三和李四都认识谁？")
            .expect("共享对象问句应可编译");
        assert_eq!(compiled.kind, GraphQuestionKind::SharedObject);
        assert_eq!(compiled.pattern.edges.len(), 2);
        assert_eq!(compiled.pattern.edges[0].predicate, "认识");
        assert_eq!(compiled.pattern.select, vec!["object".to_string()]);
    }

    #[test]
    fn conjunction_keeps_the_clause_before_the_marker() {
        let lexicon = lexicon();
        for question in [
            "谁导演过科幻片，又拿过金棕榈？",
            "谁导演过科幻片并且拿过金棕榈？",
            "谁导演过科幻片而且拿过金棕榈？",
            "谁导演过科幻片同时拿过金棕榈？",
        ] {
            let compiled = lexicon.compile(question).expect("合取问句应可编译");
            assert_eq!(
                compiled.kind,
                GraphQuestionKind::Conjunction,
                "{}",
                question
            );
            assert!(
                compiled
                    .pattern
                    .edges
                    .iter()
                    .any(|edge| edge.predicate == "导演"),
                "{}",
                question
            );
            assert!(
                compiled
                    .pattern
                    .edges
                    .iter()
                    .any(|edge| edge.predicate == "获奖"),
                "{}",
                question
            );
        }
    }

    #[test]
    fn count_questions_are_marked_without_changing_constraints() {
        let lexicon = lexicon();
        let conjunction = lexicon
            .compile("有多少人既导演过科幻片，又拿过金棕榈？")
            .expect("计数合取问句应可编译");
        assert!(conjunction.asks_for_count());
        assert_eq!(conjunction.pattern.edges.len(), 3);

        let existential = lexicon
            .compile("有多少部电影，导演是中国人，主演是美国人？")
            .expect("计数存在问句应可编译");
        assert_eq!(existential.kind, GraphQuestionKind::Existential);
        assert!(existential.asks_for_count());
        assert_eq!(existential.pattern.edges.len(), 5);
    }

    #[test]
    fn conjunction_rejects_partially_understood_clauses() {
        let lexicon = lexicon();
        // 第三个分句含已知实体「姚明」，不能因为谓词「喜欢」不在词表就静默丢弃。
        assert!(lexicon
            .compile("谁既导演过科幻片，又拿过金棕榈，还喜欢姚明？")
            .is_none());
        // 纯计数外壳可以跳过，剩余两个条件仍是一份完整解释。
        assert!(lexicon
            .compile("有多少人既导演过科幻片，又拿过金棕榈？")
            .is_some());
    }

    #[test]
    fn partially_understood_conjunction_reports_the_gap() {
        let lexicon = lexicon();
        let gap = lexicon
            .compile_gap("谁既导演过科幻片，又喜欢姚明？")
            .expect("读懂一半的合取句应报告缺口");
        assert!(
            gap.complete_clauses
                .iter()
                .any(|clause| clause.contains("导演过科幻片")),
            "{:?}",
            gap
        );
        assert!(
            gap.unresolved_clauses
                .iter()
                .any(|clause| clause.contains("喜欢姚明")),
            "{:?}",
            gap
        );
        let text = gap.render();
        assert!(text.contains("无法解析"), "{}", text);
        assert!(text.contains("喜欢姚明"), "{}", text);
        // 完全读懂或完全读不懂时都不应伪造「部分理解」。
        assert!(lexicon
            .compile_gap("谁既导演过科幻片，又拿过金棕榈？")
            .is_none());
        assert!(lexicon.compile_gap("这台机器怎么启动？").is_none());
    }

    #[test]
    fn declarative_conjunction_is_not_mistaken_for_a_question() {
        assert!(lexicon()
            .compile("张三既导演过科幻片，又拿过金棕榈")
            .is_none());
        assert!(lexicon()
            .compile("既然张三导演过科幻片，那就算了")
            .is_none());
    }

    #[test]
    fn unknown_vocabulary_stays_uncompiled() {
        let lexicon = lexicon();
        assert!(lexicon
            .compile("谁既导演过科幻片，又拿过金棕榈？")
            .is_some());
        assert!(lexicon.compile("这台机器怎么启动？").is_none());
        assert!(lexicon.compile("").is_none());
        assert!(lexicon
            .compile("谁既导演过纪录片，又拿过金棕榈？")
            .is_none());
    }
}

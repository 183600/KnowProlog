// 符号化生成器（Symbolic Generator）——"语言模型即知识库"
//
// 本模块不训练任何权重，而是把知识图谱本身当作语言模型：
//   • 生成：把与实体相关的三元组按"信息量（谓词稀有度）"排序，由模板
//     组合成连贯的介绍段落（类似 LLM 的资料生成，但每句话都可回溯到事实）；
//   • 续写：对前缀「X的」用计数统计预测下一个谓词，并支持
//     实体级 → 类型级 → 全局 的统计回退（对应 n-gram backoff，对象是事实），
//     预测句子会给出支持度与回退层级，完全可解释。
//
// 全程零神经网络、零梯度：只有计数、排序与模板组合。

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// 段落生成时的排序策略：信息量（谓词稀有度）或主题（谓词字典序）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationOrder {
    Relevance,
    Alphabetical,
}

#[derive(Debug, Clone)]
pub struct ProfilePage {
    pub entity: String,
    pub text: String,
    pub total_lines: usize,
    pub shown: usize,
    pub next_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionLevel {
    Entity,
    Type,
    Global,
}

impl CompletionLevel {
    pub fn label(self) -> &'static str {
        match self {
            CompletionLevel::Entity => "实体级统计",
            CompletionLevel::Type => "类型级回退",
            CompletionLevel::Global => "全局回退",
        }
    }
}

#[derive(Debug, Clone)]
struct CompletionPrediction {
    predicate: String,
    level: CompletionLevel,
    score: f64,
    support: String,
    sentence: String,
}

#[derive(Debug, Clone)]
struct FactLine {
    rank: f64,
    sentence: String,
}

pub struct SymbolicGenerator {
    known_entities: HashSet<String>,
    entity_types: HashMap<String, String>,
    // 类型 → 成员实体索引，避免类型级统计/原型查找时全量扫描实体
    type_members: HashMap<String, Vec<String>>,
    attributes: HashMap<String, Vec<(String, String)>>,
    by_subject: HashMap<String, HashMap<String, Vec<String>>>,
    by_object: HashMap<String, HashMap<String, Vec<String>>>,
    predicate_totals: HashMap<String, usize>,
    entity_predicate_counts: HashMap<String, HashMap<String, usize>>,
    entity_relation_total: HashMap<String, usize>,
    triples: HashSet<(String, String, String)>,
    total_relations: usize,
    max_profile_sentences: usize,
    completion_min_score: f64,
    // 谓词的中文表述（含词汇化动词）→ 规范谓词，供形式化语义解析复用
    predicate_surfaces: HashMap<String, String>,
    lexicalized_predicates: HashSet<String>,
}

impl SymbolicGenerator {
    pub fn new(max_profile_sentences: usize, completion_min_score: f64) -> Self {
        Self {
            known_entities: HashSet::new(),
            entity_types: HashMap::new(),
            type_members: HashMap::new(),
            attributes: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            predicate_totals: HashMap::new(),
            entity_predicate_counts: HashMap::new(),
            entity_relation_total: HashMap::new(),
            triples: HashSet::new(),
            total_relations: 0,
            max_profile_sentences: max_profile_sentences.max(1),
            completion_min_score: completion_min_score.clamp(0.0, 1.0),
            predicate_surfaces: HashMap::new(),
            lexicalized_predicates: HashSet::new(),
        }
    }

    pub fn is_known_entity(&self, name: &str) -> bool {
        self.known_entities.contains(name)
    }

    pub fn relation_count(&self) -> usize {
        self.total_relations
    }

    pub fn entity_type_of(&self, name: &str) -> Option<&str> {
        self.entity_types.get(name).map(String::as_str)
    }

    pub fn has_attribute(&self, entity: &str, key: &str, value: &str) -> bool {
        self.attributes.get(entity).map_or(false, |pairs| {
            pairs.iter().any(|(k, v)| k == key && v == value)
        })
    }

    pub fn attribute_values(&self, entity: &str, key: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .attributes
            .get(entity)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .collect()
            })
            .unwrap_or_default();
        values.sort();
        values.dedup();
        values
    }

    pub fn predicate_surfaces(&self) -> &HashMap<String, String> {
        &self.predicate_surfaces
    }

    pub fn observe_entity(&mut self, name: &str, etype: &str, properties: &[(String, String)]) {
        if name.trim().is_empty() {
            return;
        }
        self.known_entities.insert(name.to_string());
        if !etype.trim().is_empty() {
            match self.entity_types.insert(name.to_string(), etype.to_string()) {
                Some(previous) => {
                    if previous != etype {
                        if let Some(members) = self.type_members.get_mut(&previous) {
                            if let Some(pos) = members.iter().position(|m| m == name) {
                                members.swap_remove(pos);
                            }
                        }
                        self.type_members
                            .entry(etype.to_string())
                            .or_default()
                            .push(name.to_string());
                    }
                }
                None => {
                    self.type_members
                        .entry(etype.to_string())
                        .or_default()
                        .push(name.to_string());
                }
            }
        }
        if !properties.is_empty() {
            let bucket = self.attributes.entry(name.to_string()).or_default();
            for (key, value) in properties {
                if key.trim().is_empty() {
                    continue;
                }
                if !bucket.iter().any(|(k, v)| k == key && v == value) {
                    bucket.push((key.clone(), value.clone()));
                }
            }
        }
        for (key, _) in properties {
            if !key.trim().is_empty() {
                self.register_predicate_surface(key);
            }
        }
    }

    fn register_predicate_surface(&mut self, predicate: &str) {
        if !self.lexicalized_predicates.insert(predicate.to_string()) {
            return;
        }
        // 规范谓词自身总是可解析；规范形式优先于其它谓词的词汇化表述
        self.predicate_surfaces
            .insert(predicate.to_string(), predicate.to_string());
        for surface in lexical_surfaces(predicate) {
            if *surface == predicate {
                continue;
            }
            self.predicate_surfaces
                .entry(surface.to_string())
                .or_insert_with(|| predicate.to_string());
        }
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.trim().is_empty() || predicate.trim().is_empty() || object.trim().is_empty() {
            return;
        }
        let key = (
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        );
        if !self.triples.insert(key) {
            return;
        }
        self.known_entities.insert(subject.to_string());
        self.known_entities.insert(object.to_string());

        // triples 已按 (s,p,o) 去重，此处到达即保证是新事实，无需再线性扫描去重
        let subjects = self.by_subject.entry(subject.to_string()).or_default();
        let objects = subjects.entry(predicate.to_string()).or_default();
        objects.push(object.to_string());
        let object_bucket = self.by_object.entry(object.to_string()).or_default();
        let subjects_rev = object_bucket.entry(predicate.to_string()).or_default();
        subjects_rev.push(subject.to_string());

        *self.predicate_totals.entry(predicate.to_string()).or_insert(0) += 1;
        *self
            .entity_predicate_counts
            .entry(subject.to_string())
            .or_default()
            .entry(predicate.to_string())
            .or_insert(0) += 1;
        *self
            .entity_relation_total
            .entry(subject.to_string())
            .or_insert(0) += 1;
        self.total_relations += 1;

        self.register_predicate_surface(predicate);
    }

    // 谓词动词化：把裸谓词（含 JSON-LD 英文谓词）渲染成自然中文句子
    fn lexicalize(&self, subject: &str, predicate: &str, objects: &[String]) -> String {
        let joined = objects.join("、");
        if let Some(template) = lexical_template(predicate) {
            return template.replace("{s}", subject).replace("{o}", &joined);
        }
        format!("{}的{}是{}。", subject, predicate, joined)
    }

    // 生成式介绍：把实体相关事实排序、模板化、加话语标记后组合成段落。
    pub fn profile(&self, entity: &str) -> Option<String> {
        let lines = self.profile_lines(entity, GenerationOrder::Relevance)?;
        Some(self.render_profile(entity, &lines, 0))
    }

    // 分页生成：支持"接着说"（offset 续航）与"换个角度说"（换排序）
    pub fn profile_page(
        &self,
        entity: &str,
        offset: usize,
        order: GenerationOrder,
    ) -> Option<ProfilePage> {
        let lines = self.profile_lines(entity, order)?;
        // offset 超出末尾时视为请求最后一页之后的内容，避免切片越界 panic
        let offset = offset.min(lines.len());
        let end = offset.saturating_add(self.max_profile_sentences).min(lines.len());
        let shown = end.saturating_sub(offset);
        Some(ProfilePage {
            entity: entity.to_string(),
            text: self.render_profile(entity, &lines, offset),
            total_lines: lines.len(),
            shown,
            next_offset: end,
        })
    }

    fn profile_lines(&self, entity: &str, order: GenerationOrder) -> Option<Vec<FactLine>> {
        if !self.is_known_entity(entity) {
            return None;
        }
        let mut lines: Vec<FactLine> = Vec::new();

        if let Some(predicates) = self.by_subject.get(entity) {
            for (predicate, objects) in predicates {
                let idf = self.predicate_idf(predicate);
                lines.push(FactLine {
                    rank: idf,
                    sentence: self.lexicalize(entity, predicate, objects),
                });
            }
        }
        if let Some(attributes) = self.attributes.get(entity) {
            for (key, value) in attributes {
                // 属性与关系是同一谓词时只保留关系句，避免重复生成
                let duplicated = self
                    .by_subject
                    .get(entity)
                    .map_or(false, |predicates| predicates.contains_key(key));
                if duplicated {
                    continue;
                }
                lines.push(FactLine {
                    rank: 0.5,
                    sentence: self.lexicalize(entity, key, &[value.clone()]),
                });
            }
        }
        let reverse_sentences = self.reverse_sentences(entity, 2);
        for sentence in reverse_sentences {
            lines.push(FactLine {
                rank: -1.0,
                sentence,
            });
        }

        match order {
            GenerationOrder::Relevance => lines.sort_by(|a, b| {
                b.rank
                    .partial_cmp(&a.rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.sentence.cmp(&b.sentence))
            }),
            GenerationOrder::Alphabetical => {
                lines.sort_by(|a, b| a.sentence.cmp(&b.sentence))
            }
        }
        Some(lines)
    }

    fn render_profile(&self, entity: &str, lines: &[FactLine], offset: usize) -> String {
        let type_note = self
            .entity_types
            .get(entity)
            .filter(|t| t.as_str() != "Entity")
            .map(|t| format!("类型：{}；", t))
            .unwrap_or_default();
        if lines.is_empty() {
            return format!(
                "关于「{}」的知识（{}符号检索生成）：知识图谱中暂无可用事实。",
                entity, type_note
            );
        }
        let end = offset.saturating_add(self.max_profile_sentences).min(lines.len());
        let page = &lines[offset..end];
        let continuation = offset > 0;

        let mut body = String::new();
        for (index, line) in page.iter().enumerate() {
            body.push_str("  ");
            if continuation && index == 0 {
                body.push_str("接着，");
            } else {
                body.push_str(&Self::discourse_marker(index, page.len()));
            }
            body.push_str(&line.sentence);
            body.push('\n');
        }
        let header = if continuation {
            format!(
                "关于「{}」的知识（续：第 {}–{} 句，共 {} 条事实）：\n",
                entity,
                offset + 1,
                end,
                lines.len()
            )
        } else {
            format!(
                "关于「{}」的知识（{}符号检索生成，共 {} 条事实）：\n",
                entity,
                type_note,
                page.len()
            )
        };
        format!("{}{}", header, body.trim_end())
    }

    fn reverse_sentences(&self, entity: &str, limit: usize) -> Vec<String> {
        let Some(predicates) = self.by_object.get(entity) else {
            return Vec::new();
        };
        let mut sentences: Vec<String> = Vec::new();
        let mut predicate_names: Vec<&String> = predicates.keys().collect();
        predicate_names.sort();
        for predicate in predicate_names {
            let duplicated = self
                .by_subject
                .get(entity)
                .map_or(false, |forward| forward.contains_key(predicate));
            if duplicated {
                continue;
            }
            let subjects = &predicates[predicate];
            sentences.push(self.lexicalize(&subjects.join("、"), predicate, &[entity.to_string()]));
            if sentences.len() >= limit {
                break;
            }
        }
        sentences
    }

    fn predicate_idf(&self, predicate: &str) -> f64 {
        let count = self
            .predicate_totals
            .get(predicate)
            .copied()
            .unwrap_or(0);
        ((1.0 + self.total_relations as f64) / (1.0 + count as f64)).ln()
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先，"
        } else if index + 1 == total {
            "最后，"
        } else {
            "此外，"
        }
    }

    // 统计式续写：给定前缀「X的」，预测下一个谓词并生成句子。
    pub fn complete(&self, prefix: &str) -> Option<String> {
        let trimmed = prefix
            .trim()
            .trim_end_matches(['。', '？', '?', '！', '!', ' ']);
        let entity = trimmed.strip_suffix('的')?.trim();
        if entity.is_empty() || !self.known_entities.contains(entity) {
            return None;
        }

        let predictions = self
            .entity_level_predictions(entity)
            .or_else(|| self.type_level_predictions(entity))
            .or_else(|| self.global_predictions());
        let predictions = predictions?;
        if predictions.is_empty() {
            return None;
        }

        let mut output = String::from("符号续写（知识图谱统计回退，无神经网络）：\n");
        output.push_str(&format!("  前缀：「{}的」\n", entity));
        for (index, prediction) in predictions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 谓词「{}」（{}，支持度 {}，置信度 {:.2}）：{}\n",
                index + 1,
                prediction.predicate,
                prediction.level.label(),
                prediction.support,
                prediction.score,
                prediction.sentence
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn entity_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let predicate_counts = self.entity_predicate_counts.get(entity)?;
        let total = self
            .entity_relation_total
            .get(entity)
            .copied()
            .unwrap_or(1)
            .max(1);
        let mut items: Vec<(&String, &usize)> = predicate_counts.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter().take(3) {
            let objects = self
                .by_subject
                .get(entity)
                .and_then(|m| m.get(predicate))
                .cloned()
                .unwrap_or_default();
            if objects.is_empty() {
                continue;
            }
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Entity,
                score: *count as f64 / total as f64,
                support: format!("{}/{}", count, total),
                sentence: self.lexicalize(entity, predicate, &objects),
            });
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn type_level_predictions(&self, entity: &str) -> Option<Vec<CompletionPrediction>> {
        let etype = self.entity_types.get(entity)?;
        let (type_counts, type_total) = self.type_predicate_counts(etype);
        if type_counts.is_empty() || type_total == 0 {
            return None;
        }
        let mut items: Vec<(&String, &usize)> = type_counts.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter() {
            let score = *count as f64 / type_total as f64;
            if score < self.completion_min_score {
                continue;
            }
            let Some((prototype, objects)) = self.prototype_for(etype, entity, predicate) else {
                continue;
            };
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Type,
                score,
                support: format!("{}/{}", count, type_total),
                sentence: format!(
                    "参照同类实体「{}」：{}的{}很可能是{}。",
                    prototype,
                    entity,
                    predicate,
                    objects.join("、")
                ),
            });
            if predictions.len() >= 3 {
                break;
            }
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn type_predicate_counts(&self, etype: &str) -> (HashMap<String, usize>, usize) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        let Some(members) = self.type_members.get(etype) else {
            return (counts, total);
        };
        for entity in members {
            if let Some(predicate_counts) = self.entity_predicate_counts.get(entity) {
                for (predicate, count) in predicate_counts {
                    *counts.entry(predicate.clone()).or_insert(0) += count;
                    total += count;
                }
            }
        }
        (counts, total)
    }

    fn prototype_for(
        &self,
        etype: &str,
        entity: &str,
        predicate: &str,
    ) -> Option<(String, Vec<String>)> {
        let members = self.type_members.get(etype)?;
        let mut best: Option<(usize, &str)> = None;
        for other in members {
            if other == entity {
                continue;
            }
            let count = self
                .entity_predicate_counts
                .get(other)
                .and_then(|m| m.get(predicate))
                .copied()
                .unwrap_or(0);
            if count == 0 {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((best_count, best_name)) => {
                    count > *best_count || (count == *best_count && other.as_str() < *best_name)
                }
            };
            if replace {
                best = Some((count, other.as_str()));
            }
        }
        let (_, prototype) = best?;
        let objects = self
            .by_subject
            .get(prototype)
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some((prototype.to_string(), objects))
    }

    fn global_predictions(&self) -> Option<Vec<CompletionPrediction>> {
        if self.total_relations == 0 {
            return None;
        }
        let mut items: Vec<(&String, &usize)> = self.predicate_totals.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut predictions: Vec<CompletionPrediction> = Vec::new();
        for (predicate, count) in items.into_iter() {
            let score = *count as f64 / self.total_relations as f64;
            if score < self.completion_min_score {
                continue;
            }
            let Some((prototype, objects)) = self.global_prototype(predicate) else {
                continue;
            };
            predictions.push(CompletionPrediction {
                predicate: predicate.clone(),
                level: CompletionLevel::Global,
                score,
                support: format!("{}/{}", count, self.total_relations),
                sentence: format!(
                    "（全局回退）知识图谱中最常见的关系是「{}」，例如「{}」的「{}」是「{}」。",
                    predicate,
                    prototype,
                    predicate,
                    objects.join("、")
                ),
            });
            if predictions.len() >= 3 {
                break;
            }
        }
        if predictions.is_empty() {
            None
        } else {
            Some(predictions)
        }
    }

    fn global_prototype(&self, predicate: &str) -> Option<(String, Vec<String>)> {
        let mut subjects: Vec<&String> = self
            .entity_predicate_counts
            .iter()
            .filter(|(_, m)| m.contains_key(predicate))
            .map(|(entity, _)| entity)
            .collect();
        subjects.sort();
        let prototype = subjects.first()?;
        let objects = self
            .by_subject
            .get(*prototype)
            .and_then(|m| m.get(predicate))
            .cloned()
            .unwrap_or_default();
        Some(((*prototype).clone(), objects))
    }
}

// 谓词动词化词典：规范谓词 → 中文句模板（{s}=主语，{o}=宾语）
fn lexical_template(predicate: &str) -> Option<&'static str> {
    Some(match predicate {
        "knows" | "认识" => "{s}认识{o}。",
        "age" | "年龄" => "{s}的年龄是{o}。",
        "capital" | "首都" => "{s}的首都是{o}。",
        "author" | "作者" => "{s}的作者是{o}。",
        "birthPlace" | "出生地" | "出生于" => "{s}出生于{o}。",
        "founder" | "创始人" => "{s}的创始人是{o}。",
        "memberOf" | "隶属于" => "{s}隶属于{o}。",
        "locatedIn" | "位于" => "{s}位于{o}。",
        "worksAt" | "工作于" => "{s}工作于{o}。",
        "spouse" | "配偶" | "妻子" | "丈夫" => "{s}的配偶是{o}。",
        "child" | "孩子" | "子女" => "{s}的子女是{o}。",
        "likes" | "喜欢" => "{s}喜欢{o}。",
        "nationality" | "国籍" => "{s}的国籍是{o}。",
        "occupation" | "职业" => "{s}的职业是{o}。",
        _ => return None,
    })
}

// 中文表述集合（首个为动词化时选择的面）→ 供形式化语义解析识别
fn lexical_surfaces(predicate: &str) -> &'static [&'static str] {
    match predicate {
        "knows" | "认识" => &["认识"],
        "age" | "年龄" => &["年龄"],
        "capital" | "首都" => &["首都"],
        "author" | "作者" => &["作者"],
        "birthPlace" | "出生地" | "出生于" => &["出生地", "出生于"],
        "founder" | "创始人" => &["创始人"],
        "memberOf" | "隶属于" => &["隶属于"],
        "locatedIn" | "位于" => &["位于", "所在地"],
        "worksAt" | "工作于" => &["工作于", "任职于"],
        "spouse" | "配偶" | "妻子" | "丈夫" => &["配偶"],
        "child" | "孩子" | "子女" => &["孩子", "子女"],
        "likes" | "喜欢" => &["喜欢"],
        "nationality" | "国籍" => &["国籍"],
        "occupation" | "职业" => &["职业"],
        _ => &[],
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn sample_generator() -> SymbolicGenerator {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_entity("李四", "人物", &[]);
        generator.observe_entity("王五", "人物", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("李四", "师傅", "王五");
        generator.observe_triple("张三", "师祖", "王五");
        generator
    }

    #[test]
    fn test_profile_generation_uses_facts_and_markers() {
        let generator = sample_generator();
        let profile = generator.profile("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("类型：人物"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("李四"));
        assert!(profile.contains("年龄"));
        assert!(profile.contains("30"));
        assert!(profile.contains("首先"));
    }

    #[test]
    fn test_profile_unknown_entity_is_none() {
        let generator = sample_generator();
        assert!(generator.profile("不存在的人").is_none());
    }

    #[test]
    fn test_complete_entity_level_prediction() {
        let generator = sample_generator();
        let completion = generator.complete("张三的").expect("实体级统计应有预测");
        assert!(completion.contains("实体级统计"));
        assert!(completion.contains("师傅"));
        assert!(completion.contains("张三的师傅是李四"));
    }

    #[test]
    fn test_complete_type_level_backoff() {
        let mut generator = sample_generator();
        generator.observe_entity("赵六", "人物", &[]);
        let completion = generator.complete("赵六的").expect("类型级回退应有预测");
        assert!(completion.contains("类型级回退"), "实际输出：{}", completion);
        assert!(completion.contains("师傅"));
        assert!(completion.contains("参照同类实体"));
    }

    #[test]
    fn test_complete_global_backoff() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("甲", "类型A", &[]);
        generator.observe_triple("乙", "朋友", "丙");
        let completion = generator.complete("甲的").expect("全局回退应有预测");
        assert!(completion.contains("全局回退"), "实际输出：{}", completion);
        assert!(completion.contains("朋友"));
    }

    #[test]
    fn test_complete_requires_known_entity_and_de_suffix() {
        let generator = sample_generator();
        assert!(generator.complete("不存在的人的").is_none());
        assert!(generator.complete("张三").is_none());
    }

    #[test]
    fn test_profile_respects_sentence_limit() {
        let mut generator = SymbolicGenerator::new(2, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let profile = generator.profile("张三").unwrap();
        let sentences = profile.matches('。').count();
        assert!(sentences <= 2, "截断后应不超过 2 句：{}", profile);
    }

    #[test]
    fn test_lexicalization_maps_bare_predicates_to_chinese() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_triple("张三", "knows", "李四");
        generator.observe_triple("李四", "age", "30");
        let profile = generator.profile("张三").unwrap();
        assert!(profile.contains("张三认识李四"), "裸谓词应动词化：{}", profile);
        let profile2 = generator.profile("李四").unwrap();
        assert!(profile2.contains("李四的年龄是30"), "英文属性应中文化：{}", profile2);
        let surfaces = generator.predicate_surfaces();
        assert_eq!(surfaces.get("认识").map(String::as_str), Some("knows"));
        assert_eq!(surfaces.get("age").map(String::as_str), Some("age"));
    }

    #[test]
    fn test_profile_page_supports_continuation_and_reorder() {
        let mut generator = SymbolicGenerator::new(1, 0.3);
        generator.observe_triple("张三", "师傅", "李四");
        generator.observe_triple("张三", "师祖", "王五");
        let first = generator
            .profile_page("张三", 0, GenerationOrder::Relevance)
            .expect("第一页应可生成");
        assert_eq!(first.shown, 1);
        assert_eq!(first.total_lines, 2);
        assert_eq!(first.next_offset, 1);
        let second = generator
            .profile_page("张三", first.next_offset, GenerationOrder::Relevance)
            .expect("第二页应可生成");
        assert_eq!(second.shown, 1);
        assert!(second.text.contains("续：第 2–2 句"), "续页应有页码：{}", second.text);
        assert!(second.text.contains("接着，"));
        assert_ne!(first.text, second.text);
        let alphabetical_first = generator
            .profile_page("张三", 0, GenerationOrder::Alphabetical)
            .expect("换排序也应可生成");
        assert_eq!(alphabetical_first.total_lines, 2);
        let done = generator
            .profile_page("张三", 2, GenerationOrder::Relevance)
            .expect("翻到末页应返回空页");
        assert_eq!(done.shown, 0);
    }
}

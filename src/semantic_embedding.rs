// 符号化知识嵌入（Symbolic Knowledge Embedding）
//
// 与神经嵌入的对照：LLM/知识图谱嵌入把实体压成不可解释的稠密向量，
// 靠梯度训练得到；本模块提供"可解释的稀疏嵌入"：
//   - 每个维度是一个可命名的符号特征，如 `师傅→`（出边）、`←师傅`（入边）、
//     `类型·Person`、`属性·birth_year`；
//   - 每个分量的权重是 PPMI（正点互信息），由全图共现计数直接算出，
//     无梯度、无迭代、无模型文件；
//   - 因此"相似度/最近邻/类比"的每个结论都能回溯到具体特征与计数。
//
// 支撑的上层能力：实体近邻检索（谁和 X 最像）、向量算术类比（A:B :: C:?）、
// 以及为本体对齐提供"角色分布"证据。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
struct EntityVector {
    weights: HashMap<Arc<str>, f64>,
    norm: f64,
}

#[derive(Clone)]
pub struct SymbolicEmbedding {
    counts: HashMap<String, HashMap<Arc<str>, f64>>,
    feature_totals: HashMap<Arc<str>, f64>,
    entity_masses: HashMap<String, f64>,
    /// 特征名驻留表：同一特征在全图中只存一份 Arc<str>
    feature_interner: HashMap<String, Arc<str>>,
    total_mass: f64,
    vectors: HashMap<String, EntityVector>,
    dirty: bool,
    similarity_floor: f64,
    max_features: usize,
    /// 已观测三元组（去重）：整图快照会被反复观察，重复观察不得抬高 PPMI 计数
    seen_triples: HashSet<(String, String, String)>,
    /// 已观测实体特征（实体, 特征名）：同上，实体事实重复观察亦不得抬高计数
    seen_entity_features: HashSet<(String, String)>,
}

impl SymbolicEmbedding {
    pub fn new(similarity_floor: f64) -> Self {
        Self {
            counts: HashMap::new(),
            feature_totals: HashMap::new(),
            entity_masses: HashMap::new(),
            feature_interner: HashMap::new(),
            total_mass: 0.0,
            vectors: HashMap::new(),
            dirty: false,
            similarity_floor,
            max_features: 128,
            seen_triples: HashSet::new(),
            seen_entity_features: HashSet::new(),
        }
    }

    pub fn similarity_floor(&self) -> f64 {
        self.similarity_floor
    }

    /// 计数是否比上次重建有更新（供调用方判断是否需要 `rebuild`）。
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// 已观测的唯一三元组数（去重后）。
    pub fn triple_count(&self) -> usize {
        self.seen_triples.len()
    }

    pub fn entity_count(&self) -> usize {
        self.counts.len()
    }

    pub fn feature_count(&self) -> usize {
        self.feature_totals.len()
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        // 重复观察（快照重放）是无操作：计数与向量都不变
        if !self
            .seen_triples
            .insert((subject.to_string(), predicate.to_string(), object.to_string()))
        {
            return;
        }
        // 复用同一个缓冲区构造特征名
        let mut feature = String::with_capacity(predicate.len() + 3);
        feature.push_str(predicate);
        feature.push('→');
        self.add_feature(subject, &feature);
        feature.clear();
        feature.push('←');
        feature.push_str(predicate);
        self.add_feature(object, &feature);
    }

    pub fn observe_entity(&mut self, name: &str, entity_type: &str, properties: &[String]) {
        self.observe_entity_keys(name, entity_type, properties.iter().map(|p| p.as_str()));
    }

    /// 与 `observe_entity` 等价，但直接以属性表的键为输入。
    pub fn observe_entity_map(
        &mut self,
        name: &str,
        entity_type: &str,
        properties: &HashMap<String, String>,
    ) {
        self.observe_entity_keys(name, entity_type, properties.keys().map(|p| p.as_str()));
    }

    fn observe_entity_keys<'a, I>(&mut self, name: &str, entity_type: &str, properties: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        if name.is_empty() {
            return;
        }
        let mut feature = String::new();
        if !entity_type.is_empty() {
            feature.clear();
            feature.push_str("类型·");
            feature.push_str(entity_type);
            self.observe_entity_feature(name, &feature);
        }
        for property in properties {
            if !property.is_empty() {
                feature.clear();
                feature.push_str("属性·");
                feature.push_str(property);
                self.observe_entity_feature(name, &feature);
            }
        }
    }

    /// 实体特征去重观测：同一（实体, 特征）重复出现时不重复计票。
    fn observe_entity_feature(&mut self, entity: &str, feature: &str) {
        if !self
            .seen_entity_features
            .insert((entity.to_string(), feature.to_string()))
        {
            return;
        }
        self.add_feature(entity, feature);
    }

    fn add_feature(&mut self, entity: &str, feature: &str) {
        // 已存在的键直接自增
        match self.counts.get_mut(entity).and_then(|map| map.get_mut(feature)) {
            Some(count) => *count += 1.0,
            None => {
                let interned = self.intern_feature(feature);
                self.counts
                    .entry(entity.to_string())
                    .or_default()
                    .insert(interned, 1.0);
            }
        }
        match self.feature_totals.get_mut(feature) {
            Some(total) => *total += 1.0,
            None => {
                let interned = self.intern_feature(feature);
                self.feature_totals.insert(interned, 1.0);
            }
        }
        match self.entity_masses.get_mut(entity) {
            Some(mass) => *mass += 1.0,
            None => {
                self.entity_masses.insert(entity.to_string(), 1.0);
            }
        }
        self.total_mass += 1.0;
        self.dirty = true;
    }

    fn intern_feature(&mut self, feature: &str) -> Arc<str> {
        if let Some(shared) = self.feature_interner.get(feature) {
            return shared.clone();
        }
        let shared: Arc<str> = Arc::from(feature);
        self.feature_interner
            .insert(feature.to_string(), shared.clone());
        shared
    }

    // PPMI 逐分量计算：pmi = ln( count(e,f) * total / (count(f) * mass(e)) )
    // 只保留正值，随后 L2 归一化，便于余弦相似度。
    pub fn rebuild(&mut self) {
        // 计数未变化：沿用上次重建的向量
        if !self.dirty && !self.vectors.is_empty() {
            return;
        }
        let mut vectors: HashMap<String, EntityVector> = HashMap::new();
        let total = self.total_mass;
        for (entity, features) in &self.counts {
            let mass = self.entity_masses.get(entity).copied().unwrap_or(0.0);
            if mass <= 0.0 || total <= 0.0 {
                continue;
            }
            // 排序截断后共享特征名
            let mut weights: Vec<(Arc<str>, f64)> = Vec::with_capacity(features.len());
            for (feature, count) in features {
                let feature_total = self
                    .feature_totals
                    .get(&**feature)
                    .copied()
                    .unwrap_or(0.0);
                if feature_total <= 0.0 || *count <= 0.0 {
                    continue;
                }
                let ratio = count * total / (feature_total * mass);
                let pmi = ratio.ln();
                if pmi > 0.0 {
                    weights.push((feature.clone(), pmi));
                }
            }
            let cmp = |a: &(Arc<str>, f64), b: &(Arc<str>, f64)| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            };
            // 先做 top-k 划分，再对保留项排序
            if weights.len() > self.max_features {
                weights.select_nth_unstable_by(self.max_features, cmp);
                weights.truncate(self.max_features);
            }
            weights.sort_by(cmp);
            let norm = weights.iter().map(|(_, w)| w * w).sum::<f64>().sqrt();
            if norm <= 0.0 {
                continue;
            }
            vectors.insert(
                entity.clone(),
                EntityVector {
                    weights: weights.into_iter().collect(),
                    norm,
                },
            );
        }
        self.vectors = vectors;
        self.dirty = false;
    }

    pub fn is_embedded(&self, entity: &str) -> bool {
        self.vectors.contains_key(entity)
    }

    pub fn embedded_entities(&self) -> Vec<String> {
        let mut names: Vec<String> = self.vectors.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        if a == b {
            return if self.vectors.contains_key(a) { 1.0 } else { 0.0 };
        }
        let (Some(va), Some(vb)) = (self.vectors.get(a), self.vectors.get(b)) else {
            return 0.0;
        };
        Self::cosine(va, vb)
    }

    fn cosine(va: &EntityVector, vb: &EntityVector) -> f64 {
        if va.norm <= 0.0 || vb.norm <= 0.0 {
            return 0.0;
        }
        let (small, large) = if va.weights.len() <= vb.weights.len() {
            (va, vb)
        } else {
            (vb, va)
        };
        let dot: f64 = small
            .weights
            .iter()
            .filter_map(|(feature, weight)| {
                large.weights.get(feature).map(|other| weight * other)
            })
            .sum();
        dot / (va.norm * vb.norm)
    }

    pub fn nearest(&self, entity: &str, k: usize) -> Vec<(String, f64)> {
        // 使用源实体向量
        let Some(va) = self.vectors.get(entity) else {
            return Vec::new();
        };
        let mut neighbors: Vec<(&str, f64)> = self
            .vectors
            .iter()
            .filter(|(name, _)| name.as_str() != entity)
            .map(|(name, vector)| (name.as_str(), Self::cosine(va, vector)))
            .filter(|(_, score)| *score >= self.similarity_floor)
            .collect();
        let cmp = |a: &(&str, f64), b: &(&str, f64)| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        };
        // 全序（分数+名称唯一）下先划分 top-k 再排序
        if neighbors.len() > k {
            neighbors.select_nth_unstable_by(k, cmp);
            neighbors.truncate(k);
        }
        neighbors.sort_by(cmp);
        // 克隆最终保留的邻居名称
        neighbors
            .into_iter()
            .map(|(name, score)| (name.to_string(), score))
            .collect()
    }

    // 向量算术类比：v(B) - v(A) + v(C) 的最近邻。
    // 方向 = 从 A 到 B 的语义变化（哪些角色特征出现/消失）。
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f64)> {
        let (Some(va), Some(vb), Some(vc)) = (
            self.vectors.get(a),
            self.vectors.get(b),
            self.vectors.get(c),
        ) else {
            return Vec::new();
        };
        let mut residual: HashMap<&str, f64> = HashMap::new();
        for (feature, weight) in &vb.weights {
            *residual.entry(&**feature).or_insert(0.0) += *weight / vb.norm;
        }
        for (feature, weight) in &va.weights {
            *residual.entry(&**feature).or_insert(0.0) -= *weight / va.norm;
        }
        for (feature, weight) in &vc.weights {
            *residual.entry(&**feature).or_insert(0.0) += *weight / vc.norm;
        }
        residual.retain(|_, value| value.abs() > 1e-12);
        let residual_norm: f64 = residual.values().map(|w| w * w).sum::<f64>().sqrt();
        if residual_norm <= 0.0 {
            return Vec::new();
        }
        let mut scored: Vec<(&str, f64)> = self
            .vectors
            .iter()
            .filter(|(name, _)| {
                name.as_str() != a && name.as_str() != b && name.as_str() != c
            })
            .map(|(name, vector)| {
                let dot: f64 = residual
                    .iter()
                    .filter_map(|(feature, weight)| {
                        vector.weights.get(*feature).map(|v| weight * v)
                    })
                    .sum();
                (name.as_str(), dot / (residual_norm * vector.norm))
            })
            .filter(|(_, score)| score.is_finite())
            .collect();
        let cmp = |x: &(&str, f64), y: &(&str, f64)| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.0.cmp(y.0))
        };
        // 全序（分数+名称唯一）下先划分 top-k 再排序
        if scored.len() > k {
            scored.select_nth_unstable_by(k, cmp);
            scored.truncate(k);
        }
        scored.sort_by(cmp);
        // 克隆最终保留的类比项名称
        scored
            .into_iter()
            .map(|(name, score)| (name.to_string(), score))
            .collect()
    }

    // 方向的可解释描述：从 a 到 b 哪些特征增强、哪些减弱。
    pub fn describe_direction(&self, a: &str, b: &str) -> String {
        let (Some(va), Some(vb)) = (self.vectors.get(a), self.vectors.get(b)) else {
            return String::new();
        };
        let mut deltas: HashMap<&str, f64> = HashMap::new();
        for (feature, weight) in &vb.weights {
            *deltas.entry(&**feature).or_insert(0.0) += *weight / vb.norm;
        }
        for (feature, weight) in &va.weights {
            *deltas.entry(&**feature).or_insert(0.0) -= *weight / va.norm;
        }
        let mut positive: Vec<(&str, f64)> = deltas
            .iter()
            .filter(|(_, value)| **value > 1e-12)
            .map(|(feature, value)| (*feature, *value))
            .collect();
        let mut negative: Vec<(&str, f64)> = deltas
            .iter()
            .filter(|(_, value)| **value < -1e-12)
            .map(|(feature, value)| (*feature, *value))
            .collect();
        positive.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        negative.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        let pos_text: Vec<String> = positive
            .iter()
            .take(3)
            .map(|(feature, _)| format!("「{}」", feature))
            .collect();
        let neg_text: Vec<String> = negative
            .iter()
            .take(3)
            .map(|(feature, _)| format!("「{}」", feature))
            .collect();
        match (pos_text.is_empty(), neg_text.is_empty()) {
            (false, false) => format!(
                "新增 {}、减弱 {}",
                pos_text.join(""),
                neg_text.join("")
            ),
            (false, true) => format!("新增 {}", pos_text.join("")),
            (true, false) => format!("减弱 {}", neg_text.join("")),
            (true, true) => "角色特征几乎无变化".to_string(),
        }
    }

    pub fn shared_features(&self, a: &str, b: &str, k: usize) -> Vec<(String, f64, f64)> {
        let (Some(ca), Some(cb)) = (self.counts.get(a), self.counts.get(b)) else {
            return Vec::new();
        };
        // 先排序截断，再克隆保留项的特征名
        let mut shared: Vec<(&str, f64, f64)> = ca
            .iter()
            .filter_map(|(feature, count_a)| {
                cb.get(feature)
                    .map(|count_b| (&**feature, *count_a, *count_b))
            })
            .collect();
        shared.sort_by(|x, y| {
            let mx = x.1.min(x.2);
            let my = y.1.min(y.2);
            my.partial_cmp(&mx)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.0.cmp(y.0))
        });
        shared.truncate(k);
        shared
            .into_iter()
            .map(|(feature, count_a, count_b)| (feature.to_string(), count_a, count_b))
            .collect()
    }

    pub fn explain_similarity(&self, a: &str, b: &str) -> String {
        let score = self.similarity(a, b);
        let shared = self.shared_features(a, b, 4);
        if shared.is_empty() {
            return format!("「{}」与「{}」没有共同的角色特征（相似度 {:.2}）", a, b, score);
        }
        let features: Vec<String> = shared
            .iter()
            .map(|(feature, count_a, count_b)| {
                if (count_a - count_b).abs() < 1e-9 {
                    format!("{}（各 {} 条）", feature, count_a)
                } else {
                    format!("{}（{} 条 / {} 条）", feature, count_a, count_b)
                }
            })
            .collect();
        format!(
            "「{}」与「{}」的符号嵌入相似度 {:.2}，共同角色特征：{}",
            a,
            b,
            score,
            features.join("、")
        )
    }

    pub fn shared_feature_labels(&self, a: &str, b: &str, k: usize) -> Vec<String> {
        self.shared_features(a, b, k)
            .into_iter()
            .map(|(feature, _, _)| feature)
            .collect()
    }
}

impl Default for SymbolicEmbedding {
    fn default() -> Self {
        Self::new(0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair_graph() -> SymbolicEmbedding {
        let mut embedding = SymbolicEmbedding::new(0.6);
        embedding.observe_triple("张三", "师傅", "李四");
        embedding.observe_triple("王五", "师傅", "赵六");
        embedding.observe_entity("张三", "Person", &[]);
        embedding.observe_entity("王五", "Person", &[]);
        embedding.rebuild();
        embedding
    }

    fn chain_graph() -> SymbolicEmbedding {
        let mut embedding = SymbolicEmbedding::new(0.6);
        embedding.observe_triple("张三", "师傅", "李四");
        embedding.observe_triple("李四", "师傅", "王五");
        embedding.observe_triple("王五", "师傅", "赵六");
        embedding.rebuild();
        embedding
    }

    #[test]
    fn test_same_role_entities_are_similar() {
        let embedding = pair_graph();
        let score = embedding.similarity("张三", "王五");
        assert!(score > 0.9, "同为师傅角色应高度相似，实际 {}", score);
        let other = embedding.similarity("李四", "赵六");
        assert!(other > 0.9, "同为徒弟角色应高度相似，实际 {}", other);
        let cross = embedding.similarity("张三", "李四");
        assert!(cross < score, "角色不同应低于角色相同");
    }

    #[test]
    fn test_nearest_returns_ranked_neighbors() {
        let embedding = pair_graph();
        let neighbors = embedding.nearest("张三", 3);
        assert!(!neighbors.is_empty());
        assert_eq!(neighbors[0].0, "王五");
        assert!(neighbors.iter().all(|(name, _)| name != "张三"));
        assert!(neighbors.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn test_similarity_floor_filters_weak_matches() {
        let mut strict = SymbolicEmbedding::new(0.99);
        strict.observe_triple("张三", "师傅", "李四");
        strict.observe_triple("王五", "师傅", "赵六");
        strict.observe_entity("张三", "Person", &[]);
        strict.observe_entity("王五", "Person", &[]);
        strict.observe_triple("张三", "配偶", "王五");
        strict.rebuild();
        let neighbors = strict.nearest("张三", 5);
        assert!(neighbors.iter().all(|(_, score)| *score >= 0.99));
    }

    #[test]
    fn test_analogy_recovers_symmetric_successor() {
        let embedding = pair_graph();
        let results = embedding.analogy("张三", "李四", "王五", 3);
        assert!(!results.is_empty(), "类比应有候选");
        assert_eq!(results[0].0, "赵六");
        let direction = embedding.describe_direction("张三", "李四");
        assert!(direction.contains("师傅"), "方向描述应包含特征名：{}", direction);
    }

    #[test]
    fn test_shared_features_are_explainable() {
        let embedding = chain_graph();
        let shared = embedding.shared_features("李四", "王五", 4);
        assert!(shared.iter().any(|(feature, _, _)| feature == "师傅→"));
        let explanation = embedding.explain_similarity("李四", "王五");
        assert!(explanation.contains("师傅"));
        assert!(explanation.contains("相似度"));
    }

    #[test]
    fn test_entity_facts_become_features() {
        let mut embedding = SymbolicEmbedding::new(0.5);
        embedding.observe_entity("张三", "Person", &["birth_year".to_string()]);
        embedding.observe_entity("李四", "Person", &["birth_year".to_string()]);
        embedding.rebuild();
        let shared = embedding.shared_feature_labels("张三", "李四", 4);
        assert!(shared.contains(&"类型·Person".to_string()));
        assert!(shared.contains(&"属性·birth_year".to_string()));
    }

    #[test]
    fn test_unknown_entity_returns_empty() {
        let embedding = pair_graph();
        assert!(embedding.nearest("不存在的人", 3).is_empty());
        assert_eq!(embedding.similarity("不存在的人", "张三"), 0.0);
        assert!(embedding.analogy("不存在", "张三", "王五", 3).is_empty());
    }

    #[test]
    fn test_rebuild_is_idempotent() {
        let mut embedding = pair_graph();
        let before = embedding.similarity("张三", "王五");
        embedding.rebuild();
        let after = embedding.similarity("张三", "王五");
        assert!((before - after).abs() < 1e-9);
        assert!(!embedding.dirty);
    }

    #[test]
    fn test_repeat_observation_is_noop() {
        let mut embedding = SymbolicEmbedding::new(0.6);
        embedding.observe_triple("张三", "师傅", "李四");
        embedding.observe_triple("王五", "师傅", "赵六");
        embedding.observe_entity("张三", "Person", &["age".to_string()]);
        embedding.rebuild();
        assert_eq!(embedding.triple_count(), 2);
        let mass = embedding.total_mass;
        let features = embedding.feature_count();

        // 整图快照重放：同一批事实再次观察必须是无操作
        embedding.observe_triple("张三", "师傅", "李四");
        embedding.observe_triple("王五", "师傅", "赵六");
        embedding.observe_entity("张三", "Person", &["age".to_string()]);
        assert!(!embedding.is_dirty(), "纯重复观察不应标脏");
        assert_eq!(embedding.triple_count(), 2);
        assert_eq!(embedding.total_mass, mass, "PPMI 计票不得因重放翻倍");
        assert_eq!(embedding.feature_count(), features);

        // 新增一条事实仍应正常生效
        embedding.observe_triple("李四", "师傅", "王五");
        assert!(embedding.is_dirty());
        assert_eq!(embedding.triple_count(), 3);
    }
}

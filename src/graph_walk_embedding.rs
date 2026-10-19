// 符号化知识图谱嵌入：随机游走重启（Random Walk with Restart / Personalized PageRank）
//
// 灵感：知识图谱嵌入（KGE）把每个实体映射为向量，用向量相似度做检索与推理。
// 本模块完全不做训练，而是把"嵌入"定义为从实体出发的随机游走平稳分布：
//
//     v_e(u) = 以 e 为起点、每步以阻尼 d 沿关系边随机走、以 1-d 概率跳回 e 时，
//              长期停留在节点 u 的概率（Personalized PageRank）。
//
//   • 向量分量即"该实体与各节点的结构亲近度"，天然稀疏、可读、可解释；
//   • 结构相似 = 两个实体 RWR 向量的余弦相似度（邻域结构/角色相近的实体）；
//   • 多源重启（把对话焦点实体一起当种子）→ 上下文敏感的话题延伸；
//   • 路径解释 = 在带权图上搜索最高游走概率的简单路径，输出可回放的关系链。
//
// 全部由幂迭代确定性求得：没有参数学习、没有梯度、没有神经网络。

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

/// 超过该规模不为每个实体预算向量（检索仍可按需单源计算）
pub const DEFAULT_MAX_EMBEDDED_ENTITIES: usize = 4000;

/// 单个嵌入向量保留的最大分量数
const MAX_VECTOR_LEN: usize = 64;

/// 排序用的量化键：幂迭代的浮点尾差（~1e-16）不应改变名次，
/// 对称结构应得到相同的排名键并按名称稳定排序（确定性、可复现）
fn rank_key(score: f64) -> i64 {
    if score.is_nan() {
        i64::MIN
    } else {
        (score * 1e9).round() as i64
    }
}

/// 幂迭代的可复用缓冲区
#[derive(Default)]
struct PprScratch {
    mass_a: Vec<f64>,
    has_a: Vec<bool>,
    nodes_a: Vec<u32>,
    mass_b: Vec<f64>,
    has_b: Vec<bool>,
    nodes_b: Vec<u32>,
}

/// 单源重启随机游走的索引化实现（语义与字符串哈希表版本一致）。
///
/// `out` 按（量化概率降序，实体名升序）截断到 `limit` 个分量后输出。
#[allow(clippy::too_many_arguments)]
fn single_source_ppr_indexed(
    adjacency: &[Vec<u32>],
    seed: u32,
    limit: usize,
    damping: f64,
    iterations: usize,
    scratch: &mut PprScratch,
    out: &mut Vec<(u32, f64)>,
) {
    let n = adjacency.len();
    scratch.mass_a.clear();
    scratch.mass_a.resize(n, 0.0);
    scratch.has_a.clear();
    scratch.has_a.resize(n, false);
    scratch.mass_b.clear();
    scratch.mass_b.resize(n, 0.0);
    scratch.has_b.clear();
    scratch.has_b.resize(n, false);
    scratch.nodes_a.clear();
    scratch.nodes_a.push(seed);
    scratch.mass_a[seed as usize] = 1.0;
    scratch.has_a[seed as usize] = true;

    for _ in 0..iterations {
        // 清理上一轮留下的下一轮缓冲区
        for &node in &scratch.nodes_b {
            scratch.mass_b[node as usize] = 0.0;
            scratch.has_b[node as usize] = false;
        }
        scratch.nodes_b.clear();
        for &node in &scratch.nodes_a {
            let mass = scratch.mass_a[node as usize];
            let list = &adjacency[node as usize];
            if list.is_empty() {
                // 悬挂节点：跳回种子的概率质量
                let idx = seed as usize;
                if !scratch.has_b[idx] {
                    scratch.has_b[idx] = true;
                    scratch.nodes_b.push(seed);
                }
                scratch.mass_b[idx] += mass * damping;
            } else {
                let share = mass * damping / list.len() as f64;
                for &neighbor in list {
                    let idx = neighbor as usize;
                    if !scratch.has_b[idx] {
                        scratch.has_b[idx] = true;
                        scratch.nodes_b.push(neighbor);
                    }
                    scratch.mass_b[idx] += share;
                }
            }
        }
        let seed_idx = seed as usize;
        if !scratch.has_b[seed_idx] {
            scratch.has_b[seed_idx] = true;
            scratch.nodes_b.push(seed);
        }
        scratch.mass_b[seed_idx] += 1.0 - damping;
        // 剪枝：丢弃低于最大分量 1e-4 的涓流并重新归一，控制支持集规模
        let max_mass = scratch
            .nodes_b
            .iter()
            .map(|&node| scratch.mass_b[node as usize])
            .fold(0.0f64, f64::max);
        if max_mass > 0.0 {
            let threshold = max_mass * 1e-4;
            let mut total = 0.0;
            scratch.nodes_b.retain(|&node| {
                let mass = scratch.mass_b[node as usize];
                if mass >= threshold {
                    total += mass;
                    true
                } else {
                    scratch.has_b[node as usize] = false;
                    scratch.mass_b[node as usize] = 0.0;
                    false
                }
            });
            if total > 0.0 {
                for &node in &scratch.nodes_b {
                    scratch.mass_b[node as usize] /= total;
                }
            }
        }
        std::mem::swap(&mut scratch.mass_a, &mut scratch.mass_b);
        std::mem::swap(&mut scratch.has_a, &mut scratch.has_b);
        std::mem::swap(&mut scratch.nodes_a, &mut scratch.nodes_b);
    }

    out.clear();
    out.extend(
        scratch
            .nodes_a
            .iter()
            .map(|&node| (node, scratch.mass_a[node as usize])),
    );
    let cmp = |a: &(u32, f64), b: &(u32, f64)| {
        rank_key(b.1)
            .cmp(&rank_key(a.1))
            .then_with(|| a.0.cmp(&b.0))
    };
    if limit != usize::MAX && out.len() > limit {
        out.select_nth_unstable_by(limit, cmp);
        out.truncate(limit);
    }
    out.sort_by(cmp);
}

pub struct GraphWalkEmbedding {
    // 无向结构边（关系方向不参与游走，保留谓词用于路径解释）
    neighbors: HashMap<String, Vec<(String, String)>>,
    entities: RwLock<Vec<String>>,
    damping: f64,
    iterations: usize,
    similarity_floor: f64,
    max_embedded_entities: usize,
    // L2 归一化后的截断 RWR 向量：实体 → [(命中实体, 概率质量)]
    vectors: RwLock<HashMap<String, Vec<(String, f64)>>>,
    // 有新边加入后向量即失效；真正的重算推迟到首次访问向量时（惰性重建）
    vectors_dirty: AtomicBool,
    edge_count: usize,
}

/// 深拷贝：向量表与失效标记随值复制。供 WarmBase 会话分叉使用；
/// 懒重建语义保持（脏标记照抄，首次访问仍会按同一批边重算，结果逐字节一致）。
impl Clone for GraphWalkEmbedding {
    fn clone(&self) -> Self {
        Self {
            neighbors: self.neighbors.clone(),
            entities: RwLock::new(self.entities.read().clone()),
            damping: self.damping,
            iterations: self.iterations,
            similarity_floor: self.similarity_floor,
            max_embedded_entities: self.max_embedded_entities,
            vectors: RwLock::new(self.vectors.read().clone()),
            vectors_dirty: AtomicBool::new(self.vectors_dirty.load(Ordering::Relaxed)),
            edge_count: self.edge_count,
        }
    }
}

impl GraphWalkEmbedding {
    pub fn new(damping: f64, iterations: usize, similarity_floor: f64) -> Self {
        Self {
            neighbors: HashMap::new(),
            entities: RwLock::new(Vec::new()),
            damping: damping.clamp(0.05, 0.99),
            iterations: iterations.clamp(1, 100),
            similarity_floor: similarity_floor.clamp(0.0, 1.0),
            max_embedded_entities: DEFAULT_MAX_EMBEDDED_ENTITIES,
            vectors: RwLock::new(HashMap::new()),
            vectors_dirty: AtomicBool::new(false),
            edge_count: 0,
        }
    }

    pub fn damping(&self) -> f64 {
        self.damping
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn similarity_floor(&self) -> f64 {
        self.similarity_floor
    }

    pub fn entity_count(&self) -> usize {
        self.ensure_vectors();
        self.entities.read().len()
    }

    /// 有向视角的边数（每条三元组贡献正反两条游走边）
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn is_embedded(&self, entity: &str) -> bool {
        self.neighbors.contains_key(entity)
    }

    /// 观察一条关系三元组：加入无向结构边（正反两条）
    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if subject == object {
            return;
        }
        // 先借用查找：重复边（快照重放）直接跳过
        let forward_exists = self
            .neighbors
            .get(subject)
            .map_or(false, |list| {
                list.iter()
                    .any(|(pred, target)| pred == predicate && target == object)
            });
        if !forward_exists {
            let forward = (predicate.to_string(), object.to_string());
            match self.neighbors.get_mut(subject) {
                Some(list) => list.push(forward),
                None => {
                    self.neighbors.insert(subject.to_string(), vec![forward]);
                }
            }
            self.edge_count += 1;
            self.vectors_dirty.store(true, Ordering::Release);
        }
        let backward_exists = self
            .neighbors
            .get(object)
            .map_or(false, |list| {
                list.iter()
                    .any(|(pred, target)| pred == predicate && target == subject)
            });
        if !backward_exists {
            let backward = (predicate.to_string(), subject.to_string());
            match self.neighbors.get_mut(object) {
                Some(list) => list.push(backward),
                None => {
                    self.neighbors.insert(object.to_string(), vec![backward]);
                }
            }
            self.edge_count += 1;
            self.vectors_dirty.store(true, Ordering::Release);
        }
    }

    /// 若新边加入后向量已失效则重算（惰性重建，语义与即时重建一致）
    fn ensure_vectors(&self) {
        if !self.vectors_dirty.load(Ordering::Acquire) {
            return;
        }
        self.rebuild_vectors();
    }

    /// 重建全部实体的符号嵌入（幂迭代；规模过大时跳过，保留单源按需计算）
    pub fn rebuild(&mut self) {
        self.rebuild_vectors();
    }

    fn rebuild_vectors(&self) {
        let mut entities: Vec<String> = self.neighbors.keys().cloned().collect();
        entities.sort();
        if entities.is_empty() || entities.len() > self.max_embedded_entities {
            self.vectors.write().clear();
            *self.entities.write() = entities;
            self.vectors_dirty.store(false, Ordering::Release);
            return;
        }
        // 索引化邻接表：名称哈希在重建开始时建立，幂迭代操作整数索引
        let index: HashMap<&str, u32> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (entity.as_str(), i as u32))
            .collect();
        let mut adjacency: Vec<Vec<u32>> = Vec::with_capacity(entities.len());
        for entity in &entities {
            let mut list = Vec::new();
            if let Some(neighbors) = self.neighbors.get(entity) {
                list.reserve(neighbors.len());
                for (_, target) in neighbors {
                    if let Some(&idx) = index.get(target.as_str()) {
                        list.push(idx);
                    }
                }
            }
            adjacency.push(list);
        }
        let mut vectors = HashMap::with_capacity(entities.len());
        let mut scratch = PprScratch::default();
        let mut result = Vec::new();
        for (idx, entity) in entities.iter().enumerate() {
            result.clear();
            single_source_ppr_indexed(
                &adjacency,
                idx as u32,
                MAX_VECTOR_LEN,
                self.damping,
                self.iterations,
                &mut scratch,
                &mut result,
            );
            let mut vector: Vec<(String, f64)> = result
                .iter()
                .map(|&(node, weight)| (entities[node as usize].clone(), weight))
                .collect();
            let norm = vector
                .iter()
                .map(|(_, weight)| weight * weight)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for (_, weight) in &mut vector {
                    *weight /= norm;
                }
            }
            vectors.insert(entity.clone(), vector);
        }
        *self.vectors.write() = vectors;
        *self.entities.write() = entities;
        self.vectors_dirty.store(false, Ordering::Release);
    }

    /// 单源重启随机游走：幂迭代求平稳分布，返回按概率降序的分量
    pub fn ppr(&self, seed: &str) -> Vec<(String, f64)> {
        if !self.neighbors.contains_key(seed) {
            return Vec::new();
        }
        self.single_source_ppr(seed)
    }

    fn single_source_ppr(&self, seed: &str) -> Vec<(String, f64)> {
        self.single_source_ppr_limit(seed, usize::MAX)
    }

    /// 幂迭代求平稳分布；limit 限制保留的分量数
    fn single_source_ppr_limit(&self, seed: &str, limit: usize) -> Vec<(String, f64)> {
        let mut current: HashMap<&str, f64> = HashMap::new();
        // 复用两张表交替迭代
        let mut next: HashMap<&str, f64> = HashMap::new();
        current.insert(seed, 1.0);
        for _ in 0..self.iterations {
            next.clear();
            for (node, mass) in &current {
                match self.neighbors.get(*node) {
                    Some(list) if !list.is_empty() => {
                        let share = mass * self.damping / list.len() as f64;
                        for (_, neighbor) in list {
                            *next.entry(neighbor.as_str()).or_insert(0.0) += share;
                        }
                    }
                    // 悬挂节点：跳回种子的概率质量
                    _ => {
                        *next.entry(seed).or_insert(0.0) += mass * self.damping;
                    }
                }
            }
            *next.entry(seed).or_insert(0.0) += 1.0 - self.damping;
            // 剪枝：丢弃低于最大分量 1e-4 的涓流并重新归一，控制支持集规模（近似但确定）
            let max_mass = next.values().copied().fold(0.0f64, f64::max);
            if max_mass > 0.0 {
                let threshold = max_mass * 1e-4;
                let mut total = 0.0;
                next.retain(|_, mass| {
                    if *mass >= threshold {
                        total += *mass;
                        true
                    } else {
                        false
                    }
                });
                if total > 0.0 {
                    for mass in next.values_mut() {
                        *mass /= total;
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        let mut result: Vec<(&str, f64)> = current.into_iter().collect();
        let cmp = |a: &(&str, f64), b: &(&str, f64)| {
            rank_key(b.1)
                .cmp(&rank_key(a.1))
                .then_with(|| a.0.cmp(b.0))
        };
        if limit != usize::MAX && result.len() > limit {
            result.select_nth_unstable_by(limit, cmp);
            result.truncate(limit);
        }
        result.sort_by(cmp);
        // 收集最终保留的分量节点名
        result
            .into_iter()
            .map(|(node, weight)| (node.to_string(), weight))
            .collect()
    }

    /// 结构最相关的实体（概率质量最大的分量，排除自身）
    pub fn related(&self, seed: &str, k: usize) -> Vec<(String, f64)> {
        self.ensure_vectors();
        let vectors = self.vectors.read();
        match vectors.get(seed) {
            // 克隆最终保留的 k 个分量
            Some(vector) => vector
                .iter()
                .filter(|(entity, _)| entity != seed)
                .take(k.max(1))
                .cloned()
                .collect(),
            None => {
                let mut computed = self.ppr(seed);
                computed.truncate(MAX_VECTOR_LEN);
                computed.retain(|(entity, _)| entity != seed);
                computed.truncate(k.max(1));
                computed
            }
        }
    }

    /// 结构相似度：两个实体 RWR 向量的余弦相似度（需要已重建嵌入）
    pub fn similarity(&self, a: &str, b: &str) -> Option<f64> {
        self.ensure_vectors();
        let vectors = self.vectors.read();
        let left = vectors.get(a)?;
        let right = vectors.get(b)?;
        let left_map: HashMap<&str, f64> =
            left.iter().map(|(name, w)| (name.as_str(), *w)).collect();
        let mut dot = 0.0;
        for (name, weight) in right {
            if let Some(other) = left_map.get(name.as_str()) {
                dot += weight * other;
            }
        }
        Some(dot.clamp(-1.0, 1.0))
    }

    /// 结构角色最接近的实体（按余弦相似度降序，过滤低于阈值者）
    pub fn nearest(&self, entity: &str, k: usize) -> Vec<(String, f64)> {
        self.ensure_vectors();
        let vectors = self.vectors.read();
        // 构建源实体的查找表
        let Some(left) = vectors.get(entity) else {
            return Vec::new();
        };
        let left_map: HashMap<&str, f64> =
            left.iter().map(|(name, w)| (name.as_str(), *w)).collect();
        let entities = self.entities.read();
        let mut scored: Vec<(&str, f64)> = entities
            .iter()
            .filter(|other| other.as_str() != entity)
            .filter_map(|other| {
                let right = vectors.get(other)?;
                let mut dot = 0.0;
                for (name, weight) in right {
                    if let Some(other_weight) = left_map.get(name.as_str()) {
                        dot += weight * other_weight;
                    }
                }
                let score = dot.clamp(-1.0, 1.0);
                if score >= self.similarity_floor {
                    Some((other.as_str(), score))
                } else {
                    None
                }
            })
            .collect();
        let cmp = |a: &(&str, f64), b: &(&str, f64)| {
            rank_key(b.1)
                .cmp(&rank_key(a.1))
                .then_with(|| a.0.cmp(b.0))
        };
        let keep = k.max(1);
        if scored.len() > keep {
            scored.select_nth_unstable_by(keep, cmp);
            scored.truncate(keep);
        }
        scored.sort_by(cmp);
        // 收集最终保留的邻居实体名
        scored
            .into_iter()
            .map(|(name, score)| (name.to_string(), score))
            .collect()
    }

    /// 多源重启：把多个种子实体（如对话焦点栈）作为共同起点做上下文向量
    pub fn context_related(&self, seeds: &[String], k: usize) -> Vec<(String, f64)> {
        self.ensure_vectors();
        let vectors = self.vectors.read();
        let mut merged: HashMap<String, f64> = HashMap::new();
        let mut used = 0usize;
        for seed in seeds {
            // 命中缓存时直接借用，未命中才计算临时向量
            let computed;
            let vector: &[(String, f64)] = match vectors.get(seed) {
                Some(vector) => vector.as_slice(),
                None => {
                    if !self.neighbors.contains_key(seed) {
                        continue;
                    }
                    computed = self.ppr(seed);
                    computed.as_slice()
                }
            };
            used += 1;
            for (entity, weight) in vector {
                *merged.entry(entity.clone()).or_insert(0.0) += *weight;
            }
        }
        if used == 0 {
            return Vec::new();
        }
        let seed_set: HashSet<&str> = seeds.iter().map(|s| s.as_str()).collect();
        let mut result: Vec<(String, f64)> = merged
            .into_iter()
            .filter(|(entity, _)| !seed_set.contains(entity.as_str()))
            .collect();
        let cmp = |a: &(String, f64), b: &(String, f64)| {
            rank_key(b.1)
                .cmp(&rank_key(a.1))
                .then_with(|| a.0.cmp(&b.0))
        };
        let keep = k.max(1);
        if result.len() > keep {
            result.select_nth_unstable_by(keep, cmp);
            result.truncate(keep);
        }
        result.sort_by(cmp);
        result
    }

    /// 最高游走概率的简单路径（用于解释两个实体为什么"结构相关"）
    pub fn best_path(
        &self,
        from: &str,
        to: &str,
        max_hops: usize,
    ) -> Option<Vec<(String, String)>> {
        if from == to || !self.neighbors.contains_key(from) || !self.neighbors.contains_key(to) {
            return None;
        }
        self.ensure_vectors();
        if self.entities.read().len() > self.max_embedded_entities {
            return None;
        }
        let max_hops = max_hops.clamp(1, 5);
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(from);
        let mut best: Option<(f64, Vec<(String, String)>)> = None;
        self.search_path(
            from,
            to,
            max_hops,
            1.0,
            &mut Vec::new(),
            &mut visited,
            &mut best,
        );
        best.map(|(_, path)| path)
    }

    #[allow(clippy::too_many_arguments)]
    fn search_path<'a>(
        &'a self,
        current: &'a str,
        target: &str,
        remaining: usize,
        score: f64,
        path: &mut Vec<(String, String)>,
        visited: &mut HashSet<&'a str>,
        best: &mut Option<(f64, Vec<(String, String)>)>,
    ) {
        if remaining == 0 {
            return;
        }
        let Some(list) = self.neighbors.get(current) else {
            return;
        };
        for (predicate, neighbor) in list {
            if neighbor == target {
                let step_score = score / list.len() as f64;
                path.push((predicate.clone(), neighbor.clone()));
                // 需要更新最优路径时克隆整条路径
                let replace = match best {
                    None => true,
                    Some((best_score, best_path)) => {
                        let diff = step_score - *best_score;
                        if diff.abs() < 1e-12 {
                            path.as_slice() < best_path.as_slice()
                        } else {
                            diff > 0.0
                        }
                    }
                };
                if replace {
                    *best = Some((step_score, path.clone()));
                }
                path.pop();
                continue;
            }
            if visited.contains(neighbor.as_str()) {
                continue;
            }
            let step_score = score / list.len() as f64;
            // 剪枝：剩余步数内不可能超过当前最优
            if let Some((best_score, _)) = best {
                if step_score <= *best_score {
                    continue;
                }
            }
            visited.insert(neighbor.as_str());
            path.push((predicate.clone(), neighbor.clone()));
            self.search_path(
                neighbor,
                target,
                remaining - 1,
                step_score,
                path,
                visited,
                best,
            );
            path.pop();
            visited.remove(neighbor.as_str());
        }
    }

    /// 路径渲染：`张三 —师傅→ 李四 —师祖→ 王五`
    pub fn render_path(from: &str, path: &[(String, String)]) -> String {
        let mut out = String::from(from);
        for (predicate, node) in path {
            out.push_str(&format!(" —{}→ {}", predicate, node));
        }
        out
    }

    /// 结构联想报告：相关的实体、结构角色近邻、解释路径
    pub fn report(&self, entity: &str, k: usize) -> String {
        if !self.is_embedded(entity) {
            return format!(
                "结构联想中没有「{}」：它没有出现在任何关系三元组中。",
                entity
            );
        }
        let mut output = String::from(
            "结构联想（随机游走重启 RWR 符号嵌入：向量=游走驻留分布，无神经网络）：\n",
        );
        output.push_str(&format!(
            "  图规模：{} 个实体 / {} 条游走边；阻尼 {:.2}，迭代 {} 轮。\n",
            self.entity_count(),
            self.edge_count(),
            self.damping,
            self.iterations
        ));
        let related = self.related(entity, k.max(1));
        if related.is_empty() {
            output.push_str("  （没有可游走到的邻近实体：该实体是孤立节点）\n");
            return output.trim_end().to_string();
        }
        output.push_str(&format!("  与「{}」结构最相关的实体：\n", entity));
        for (index, (name, weight)) in related.iter().enumerate() {
            output.push_str(&format!(
                "    {}. {}（游走驻留概率 {:.3}）\n",
                index + 1,
                name,
                weight
            ));
        }
        let neighbors = self.nearest(entity, 3);
        if !neighbors.is_empty() {
            output.push_str("  结构角色最接近（RWR 向量余弦）：\n");
            for (index, (name, score)) in neighbors.iter().enumerate() {
                output.push_str(&format!(
                    "    {}. {}（相似度 {:.3}）\n",
                    index + 1,
                    name,
                    score
                ));
            }
        }
        if let Some((top, _)) = related.first() {
            if let Some(path) = self.best_path(entity, top, 4) {
                output.push_str(&format!(
                    "  近邻解释路径：{}\n",
                    Self::render_path(entity, &path)
                ));
            }
        }
        output.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> GraphWalkEmbedding {
        let mut graph = GraphWalkEmbedding::new(0.85, 20, 0.2);
        graph.observe_triple("张三", "师傅", "李四");
        graph.observe_triple("李四", "师傅", "王五");
        graph.observe_triple("王五", "师傅", "赵六");
        graph.rebuild();
        graph
    }

    #[test]
    fn test_related_prefers_direct_neighbor() {
        let graph = chain();
        let related = graph.related("张三", 3);
        assert!(!related.is_empty());
        assert_eq!(related[0].0, "李四", "直接邻居应是游走概率最高者");
        assert!(related[0].1 > 0.0);
    }

    #[test]
    fn test_probability_mass_decays_with_distance() {
        let graph = chain();
        let related: HashMap<String, f64> = graph.related("张三", 3).into_iter().collect();
        let li = related.get("李四").copied().unwrap_or(0.0);
        let wang = related.get("王五").copied().unwrap_or(0.0);
        assert!(li > wang, "越远的实体驻留概率应越低");
    }

    #[test]
    fn test_structural_similarity_symmetric_positions() {
        let mut graph = GraphWalkEmbedding::new(0.85, 20, 0.2);
        // 甲、乙、丙 都有同一个老师 "导师"，结构角色相同
        graph.observe_triple("甲", "导师", "导师");
        graph.observe_triple("乙", "导师", "导师");
        graph.observe_triple("丙", "导师", "导师");
        graph.observe_triple("导师", "就职于", "大学");
        // 一个无关的小组件：与甲的结构向量无重叠，应被相似度区分开
        graph.observe_triple("老张", "朋友", "老王");
        graph.rebuild();
        let ab = graph.similarity("甲", "乙").expect("应有向量");
        let ac = graph.similarity("甲", "丙").expect("应有向量");
        assert!(ab > 0.5, "结构同位实体的余弦相似度应较高，实得 {}", ab);
        assert!((ab - ac).abs() < 1e-9, "对称结构相似度应相等");
        let nearest = graph.nearest("甲", 2);
        assert!(
            nearest.iter().any(|(name, _)| name == "乙" || name == "丙"),
            "nearest={:?}, ab={}, ac={}",
            nearest,
            ab,
            ac
        );
    }

    #[test]
    fn test_context_related_uses_multiple_seeds() {
        let graph = chain();
        let seeds = vec!["张三".to_string(), "赵六".to_string()];
        let related = graph.context_related(&seeds, 2);
        assert!(!related.is_empty());
        assert!(
            related
                .iter()
                .all(|(name, _)| name != "张三" && name != "赵六"),
            "种子实体本身不应出现在延伸结果里"
        );
    }

    #[test]
    fn test_best_path_explains_connection() {
        let graph = chain();
        let path = graph.best_path("张三", "赵六", 4).expect("存在三跳路径");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].1, "李四");
        assert_eq!(path[1].1, "王五");
        assert_eq!(path[2].1, "赵六");
        assert!(path.iter().all(|(predicate, _)| predicate == "师傅"));
        let rendered = GraphWalkEmbedding::render_path("张三", &path);
        assert!(rendered.contains("张三 —师傅→ 李四"));
    }

    #[test]
    fn test_unknown_entity_report_is_graceful() {
        let graph = chain();
        let report = graph.report("不存在的人", 3);
        assert!(report.contains("没有"));
        assert!(graph.ppr("不存在的人").is_empty());
        assert!(graph.best_path("张三", "不存在的人", 4).is_none());
    }

    #[test]
    fn test_report_contains_walk_evidence() {
        let graph = chain();
        let report = graph.report("张三", 3);
        assert!(report.contains("随机游走重启"));
        assert!(report.contains("李四"));
        assert!(report.contains("游走驻留概率"));
    }
}

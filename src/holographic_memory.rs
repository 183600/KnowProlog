//! 全息符号记忆（Vector Symbolic Architecture / 超维计算，纯符号、零训练）
//!
//! 与神经符号整合（neural-symbolic integration）的对照：本模块给系统提供一条
//! "像向量一样"的联想通道，但里面没有任何神经网络、没有梯度、没有权重训练：
//!
//! - 符号 → 超向量：每个符号（实体/谓词/值）由符号名的哈希确定性映射为一个
//!   D 维 ±1 双极向量（splitmix64 伪随机流）。相同符号永远得到相同向量，
//!   不同符号近似正交（期望余弦 0，涨落 ~1/√D）——这是"随机码本"，
//!   不是训练出来的嵌入。
//! - 绑定 bind：逐元素乘积 ⊙。对 ±1 向量它是精确自逆的：
//!   (a⊙b)⊙b = a；因此绑定同时就是解绑。三元组编码为
//!   T(s,p,o) = V(s) ⊙ 置换¹(V(p)) ⊙ 置换²(V(o))——角色位置用循环置换区分，
//!   避免乘积交换律导致"谁绑了谁"的角色混叠；三种角色的解缠互不污染。
//! - 叠加 bundle：整库全息图 M = Σ T，一张图存下全部事实；
//!   用两个已知角色的绑定作钥匙去乘 M 即"解缠"出第三个角色的近似向量。
//! - 净噪 cleanup：解缠结果是"目标向量 + 串扰噪声"，在对应角色的码本里取
//!   余弦最近邻（顶层 k 个）即可读回符号——这就是 VSA 的 clean-up memory。
//! - 置换 permute：循环移位编码时序；对话焦点序列叠加成
//!   C = Σ 移位^k(V(焦点))，按轮次反向移位即可回读"k 轮前谈的是谁"。
//! - 神经符号整合（符号主义复刻）：全息通道只做**软召回**（分级余弦候选），
//!   硬通道是三元组登记集合的**精确核查**；未获硬验证的候选一律标注"未登记"，
//!   不进入结论。软通道负责"想到"，硬通道负责"核实"，用符号约束抑制串扰幻觉。
//! - 类比迁移：A:B :: C:? 先用绑定的自逆性从事实中解缠出关系谓词 P
//!   （经硬验证的关系），再把 P 绑定到 C 上召回 D——若 D 未登记则是
//!   可追责的"全息推测"，若已登记则是精确检索。
//!
//! 所有操作可复算、可解释：任何召回都能给出余弦分数、所依据的角色键、
//! 以及是否通过符号核查。

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// 默认超维维度：4096 位双极向量（噪声随 1/√D 衰减）
pub const DEFAULT_DIMENSION: usize = 4096;
/// 维度下限（低于该值串扰过大）
const MIN_DIMENSION: usize = 256;
/// 维度上限（控制单键内存占用）
const MAX_DIMENSION: usize = 16384;

/// 全息召回的一条候选（符号 + 余弦分 + 是否通过三元组硬核查）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoloHit {
    pub symbol: String,
    /// 解缠向量与该符号码本向量的余弦（分级置信，越大越可信）
    pub score: f64,
    /// 硬通道核查：该候选构成的三元组是否登记在事实集合中
    pub verified: bool,
}

/// 全息类比候选：把 A:B 的关系迁移到 C 上得到的 D
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoloAnalogyHit {
    pub symbol: String,
    pub score: f64,
    pub verified: bool,
    /// 承载迁移的关系谓词及各自的关系召回分
    pub via: Vec<(String, f64)>,
}

/// 按轮次回读的对话焦点
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextRecall {
    /// 距今多少轮（0 = 最近一轮）
    pub age: usize,
    pub symbol: String,
    pub score: f64,
}

/// 自检统计：对抽样三元组做客体召回，top-1 是否精确命中
#[derive(Debug, Clone, Default)]
pub struct HoloSelfCheck {
    pub total: usize,
    pub top1_exact: usize,
    pub top1_verified: usize,
    pub examples: Vec<(String, String, String, f64)>,
}

impl HoloSelfCheck {
    /// top-1 与抽样三元组客体完全一致的比例
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.top1_exact as f64 / self.total as f64
        }
    }

    /// top-1 命中该键真实取值之一的比例（多值键场景下更有意义）
    pub fn verified_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.top1_verified as f64 / self.total as f64
        }
    }
}

/// 确定性生成符号的超向量：FNV-1a 播种 + splitmix64 取最高位 → ±1
fn generate_vector(symbol: &str, dim: usize) -> Vec<i8> {
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in symbol.as_bytes() {
        state ^= *byte as u64;
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        vector.push(if z >> 63 == 0 { 1 } else { -1 });
    }
    vector
}

/// 循环右移：out[(i + shift) % n] = v[i]；负 shift 为左移（逆置换）
fn rotate(values: &[f64], shift: isize) -> Vec<f64> {
    let n = values.len();
    let shift = shift.rem_euclid(n as isize) as usize;
    if shift == 0 {
        return values.to_vec();
    }
    let mut out = vec![0.0; n];
    for (i, value) in values.iter().enumerate() {
        out[(i + shift) % n] = *value;
    }
    out
}

/// i8 版本的循环右移（用于角色位置置换）
fn rotate_i8(values: &[i8], shift: isize) -> Vec<i8> {
    let n = values.len();
    let shift = shift.rem_euclid(n as isize) as usize;
    if shift == 0 {
        return values.to_vec();
    }
    let mut out = vec![0i8; n];
    for (i, value) in values.iter().enumerate() {
        out[(i + shift) % n] = *value;
    }
    out
}

/// 两个双极向量的绑定（逐元素乘积），输出 f64 以便叠加
fn bind_to_f64(a: &[i8], b: &[i8]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x * *y) as f64)
        .collect()
}

/// 双极绑定（逐元素乘积）：自逆、可交换
pub fn bind_bipolar(a: &[i8], b: &[i8]) -> Vec<i8> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// 逐元素余弦：外部分析/测试用
pub fn cosine(a: &[f64], b: &[i8]) -> f64 {
    let mut dot = 0.0;
    let mut norm_sq = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * (*y as f64);
        norm_sq += x * x;
    }
    if norm_sq <= f64::EPSILON {
        return 0.0;
    }
    dot / (norm_sq.sqrt() * (b.len() as f64).sqrt())
}

#[derive(Clone)]
pub struct HolographicMemory {
    dim: usize,
    similarity_floor: f64,
    max_hits: usize,
    /// 符号码本：已参与编码的符号 → 双极向量（懒生成、只存一份）
    vectors: HashMap<String, Vec<i8>>,
    symbol_set: HashSet<String>,
    subjects: Vec<String>,
    predicates: Vec<String>,
    objects: Vec<String>,
    subject_seen: HashSet<String>,
    predicate_seen: HashSet<String>,
    object_seen: HashSet<String>,
    /// 整库全息图：Σ V(s) ⊙ V(p) ⊙ V(o)
    memory: Vec<f64>,
    /// 硬通道：登记的三元组集合（精确核查，去重）
    triples: HashSet<(String, String, String)>,
    triple_order: Vec<(String, String, String)>,
    /// 时序通道：焦点符号的超维叠加（置换编码轮次）
    context: Vec<f64>,
    context_symbols: Vec<String>,
    context_seen: HashSet<String>,
    last_focus: Option<String>,
    context_len: usize,
}

impl HolographicMemory {
    pub fn new(dimension: usize, similarity_floor: f64, max_hits: usize) -> Self {
        let dim = dimension.clamp(MIN_DIMENSION, MAX_DIMENSION);
        let dim = if dim % 2 == 0 { dim } else { dim + 1 };
        Self {
            dim,
            similarity_floor: similarity_floor.max(0.0),
            max_hits: max_hits.max(1),
            vectors: HashMap::new(),
            symbol_set: HashSet::new(),
            subjects: Vec::new(),
            predicates: Vec::new(),
            objects: Vec::new(),
            subject_seen: HashSet::new(),
            predicate_seen: HashSet::new(),
            object_seen: HashSet::new(),
            memory: vec![0.0; dim],
            triples: HashSet::new(),
            triple_order: Vec::new(),
            context: vec![0.0; dim],
            context_symbols: Vec::new(),
            context_seen: HashSet::new(),
            last_focus: None,
            context_len: 0,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn similarity_floor(&self) -> f64 {
        self.similarity_floor
    }

    pub fn max_hits(&self) -> usize {
        self.max_hits
    }

    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    pub fn symbol_count(&self) -> usize {
        self.symbol_set.len()
    }

    pub fn role_counts(&self) -> (usize, usize, usize) {
        (
            self.subjects.len(),
            self.predicates.len(),
            self.objects.len(),
        )
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// 是否已把符号登记进码本（含作为任意角色出现过的符号）
    pub fn is_known(&self, symbol: &str) -> bool {
        self.symbol_set.contains(symbol)
    }

    /// 硬通道：三元组是否登记
    pub fn contains_triple(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.triples.contains(&(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ))
    }

    fn ensure_vector(&mut self, symbol: &str) {
        if !self.vectors.contains_key(symbol) {
            let vector = generate_vector(symbol, self.dim);
            self.vectors.insert(symbol.to_string(), vector);
        }
    }

    fn lookup(&self, symbol: &str) -> Cow<'_, [i8]> {
        match self.vectors.get(symbol) {
            Some(vector) => Cow::Borrowed(vector.as_ref()),
            None => Cow::Owned(generate_vector(symbol, self.dim)),
        }
    }

    fn register_role(list: &mut Vec<String>, seen: &mut HashSet<String>, symbol: &str) {
        if seen.insert(symbol.to_string()) {
            list.push(symbol.to_string());
        }
    }

    /// 观察一条三元组：登记硬通道并叠加进全息图（重复观察幂等）
    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        if !self.triples.insert((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        )) {
            return;
        }
        self.triple_order.push((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
        Self::register_role(&mut self.subjects, &mut self.subject_seen, subject);
        Self::register_role(&mut self.predicates, &mut self.predicate_seen, predicate);
        Self::register_role(&mut self.objects, &mut self.object_seen, object);
        self.symbol_set.insert(subject.to_string());
        self.symbol_set.insert(predicate.to_string());
        self.symbol_set.insert(object.to_string());
        self.ensure_vector(subject);
        self.ensure_vector(predicate);
        self.ensure_vector(object);
        let encoding = self.triple_encoding(subject, predicate, object);
        for (slot, value) in self.memory.iter_mut().zip(encoding.iter()) {
            *slot += *value;
        }
    }

    /// 解缠：M ⊙ key
    fn unbind(&self, key: &[f64]) -> Vec<f64> {
        self.memory
            .iter()
            .zip(key.iter())
            .map(|(m, k)| m * k)
            .collect()
    }

    /// 三元组编码：T = V(主体) ⊙ 置换¹(V(谓词)) ⊙ 置换²(V(客体))。
    /// 角色位置用置换区分，避免逐元素乘积交换律带来的"角色混叠"
    /// （否则 (王五, 师傅, 赵六) 会污染查询 "?--师傅-->王五"）。
    fn triple_encoding(&self, subject: &str, predicate: &str, object: &str) -> Vec<f64> {
        let vs = self.lookup(subject);
        let vp = rotate_i8(&self.lookup(predicate), 1);
        let vo = rotate_i8(&self.lookup(object), 2);
        vs.iter()
            .zip(vp.iter())
            .zip(vo.iter())
            .map(|((s, p), o)| (*s * *p * *o) as f64)
            .collect()
    }

    /// 净噪：解缠向量 vs 角色码本，余弦 top-k（有下限阈值）
    fn cleanup(&self, noisy: &[f64], candidates: &[String], topk: usize) -> Vec<(String, f64)> {
        let mut norm_sq = 0.0;
        for x in noisy {
            norm_sq += x * x;
        }
        if norm_sq <= f64::EPSILON {
            return Vec::new();
        }
        let scale = 1.0 / (norm_sq.sqrt() * (self.dim as f64).sqrt());
        let mut scored: Vec<(String, f64)> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let vector = self.lookup(candidate);
            let mut dot = 0.0;
            for (x, y) in noisy.iter().zip(vector.iter()) {
                dot += x * (*y as f64);
            }
            scored.push((candidate.clone(), dot * scale));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(topk);
        scored.retain(|(_, score)| *score >= self.similarity_floor);
        scored
    }

    /// 客体召回：(主体, 谓词) → ?客体；钥匙 = V(s) ⊙ 置换¹(V(p))，
    /// 解缠结果 = 置换²(V(客体)) + 串扰，逆置换后净噪
    pub fn recall_object(&self, subject: &str, predicate: &str) -> Vec<HoloHit> {
        if !self.is_known(subject) || !self.is_known(predicate) {
            return Vec::new();
        }
        let key = bind_to_f64(
            &self.lookup(subject),
            &rotate_i8(&self.lookup(predicate), 1),
        );
        let decoded = rotate(&self.unbind(&key), -2);
        self.cleanup(&decoded, &self.objects, self.max_hits)
            .into_iter()
            .map(|(symbol, score)| {
                let verified = self.contains_triple(subject, predicate, &symbol);
                HoloHit {
                    symbol,
                    score,
                    verified,
                }
            })
            .collect()
    }

    /// 主体反查：(谓词, 客体) → ?主体；钥匙 = 置换¹(V(p)) ⊙ 置换²(V(o))，
    /// 解缠结果 = V(主体) + 串扰，直接净噪
    pub fn recall_subject(&self, predicate: &str, object: &str) -> Vec<HoloHit> {
        if !self.is_known(predicate) || !self.is_known(object) {
            return Vec::new();
        }
        let key = bind_to_f64(
            &rotate_i8(&self.lookup(predicate), 1),
            &rotate_i8(&self.lookup(object), 2),
        );
        let decoded = self.unbind(&key);
        self.cleanup(&decoded, &self.subjects, self.max_hits)
            .into_iter()
            .map(|(symbol, score)| {
                let verified = self.contains_triple(&symbol, predicate, object);
                HoloHit {
                    symbol,
                    score,
                    verified,
                }
            })
            .collect()
    }

    /// 谓词召回：(主体, 客体) → ?谓词；钥匙 = V(s) ⊙ 置换²(V(o))，
    /// 解缠结果 = 置换¹(V(谓词)) + 串扰，逆置换后净噪
    pub fn recall_predicate(&self, subject: &str, object: &str) -> Vec<HoloHit> {
        if !self.is_known(subject) || !self.is_known(object) {
            return Vec::new();
        }
        let key = bind_to_f64(&self.lookup(subject), &rotate_i8(&self.lookup(object), 2));
        let decoded = rotate(&self.unbind(&key), -1);
        self.cleanup(&decoded, &self.predicates, self.max_hits)
            .into_iter()
            .map(|(symbol, score)| {
                let verified = self.contains_triple(subject, &symbol, object);
                HoloHit {
                    symbol,
                    score,
                    verified,
                }
            })
            .collect()
    }

    /// 全息类比：A:B :: C:? —— 先解缠出经硬验证的关系谓词，再迁移到 C 上召回 D
    pub fn analogy(&self, a: &str, b: &str, c: &str) -> Vec<HoloAnalogyHit> {
        if !self.is_known(a) || !self.is_known(b) || !self.is_known(c) {
            return Vec::new();
        }
        let relations = self.recall_predicate(a, b);
        let mut merged: HashMap<String, HoloAnalogyHit> = HashMap::new();
        for relation in relations {
            // 硬约束：关系必须已在事实中登记，串扰式关系候选不参与迁移
            if !relation.verified {
                continue;
            }
            for hit in self.recall_object(c, &relation.symbol) {
                let score = relation.score * hit.score;
                if score < self.similarity_floor {
                    continue;
                }
                merged
                    .entry(hit.symbol.clone())
                    .and_modify(|entry| {
                        entry.score = entry.score.max(score);
                        entry.verified |= hit.verified;
                        if !entry.via.iter().any(|(p, _)| p == &relation.symbol) {
                            entry.via.push((relation.symbol.clone(), relation.score));
                        }
                    })
                    .or_insert(HoloAnalogyHit {
                        symbol: hit.symbol,
                        score,
                        verified: hit.verified,
                        via: vec![(relation.symbol.clone(), relation.score)],
                    });
            }
        }
        let mut results: Vec<HoloAnalogyHit> = merged.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        results.truncate(self.max_hits);
        results
    }

    /// 把焦点符号压入时序叠加记忆（置换编码轮次；连续重复自动去重）
    pub fn push_focus(&mut self, symbol: &str) {
        let symbol = symbol.trim();
        if symbol.is_empty() {
            return;
        }
        if self.last_focus.as_deref() == Some(symbol) {
            return;
        }
        self.ensure_vector(symbol);
        self.context = rotate(&self.context, 1);
        let vector = self.vectors.get(symbol).unwrap();
        for i in 0..self.dim {
            self.context[i] += vector[i] as f64;
        }
        if self.context_seen.insert(symbol.to_string()) {
            self.context_symbols.push(symbol.to_string());
        }
        self.last_focus = Some(symbol.to_string());
        self.context_len += 1;
    }

    /// 按轮次回读焦点：age=0 最近一轮
    pub fn recent_focus(&self, age: usize) -> Option<ContextRecall> {
        if self.context_len == 0 || age >= self.context_len {
            return None;
        }
        let noisy = rotate(&self.context, -(age as isize));
        let top = self.cleanup(&noisy, &self.context_symbols, 1);
        top.into_iter()
            .next()
            .map(|(symbol, score)| ContextRecall { age, symbol, score })
    }

    /// 最近若干轮的焦点轨迹（0..max_age）
    pub fn context_trace(&self, max_age: usize) -> Vec<ContextRecall> {
        (0..max_age.min(self.context_len))
            .filter_map(|age| self.recent_focus(age))
            .collect()
    }

    /// 自检：均匀抽样三元组，检验 (s,p)→o 的召回质量。
    /// 注意 (s,p) 可能有多值客体：`top1_verified` 表示 top-1 是该键的真实取值之一，
    /// `top1_exact` 表示 top-1 恰好等于被抽样的那条三元组的客体。
    pub fn self_check(&self, samples: usize) -> HoloSelfCheck {
        let mut check = HoloSelfCheck::default();
        if self.triple_order.is_empty() || samples == 0 {
            return check;
        }
        let step = (self.triple_order.len() / samples).max(1);
        for (subject, predicate, object) in self.triple_order.iter().step_by(step).take(samples) {
            let hits = self.recall_object(subject, predicate);
            check.total += 1;
            if let Some(top) = hits.first() {
                if top.verified {
                    check.top1_verified += 1;
                }
                if &top.symbol == object {
                    check.top1_exact += 1;
                    check.examples.push((
                        subject.clone(),
                        predicate.clone(),
                        object.clone(),
                        top.score,
                    ));
                }
            }
        }
        check
    }

    /// 面向用户的全局报告：规模、操作语义、自检结果、样例与焦点轨迹
    pub fn report(&self) -> String {
        let (subjects, predicates, objects) = self.role_counts();
        let mut out = String::new();
        out.push_str(
            "全息符号记忆（Vector Symbolic Architecture：绑定/叠加/置换 + 净噪，纯符号、零训练）\n",
        );
        out.push_str(&format!(
            "  维度 D={} ｜ 符号 {} 个（主体 {} / 谓词 {} / 客体 {}）｜ 三元组 {} 条 ｜ 时序记忆 {} 轮\n",
            self.dim,
            self.symbol_count(),
            subjects,
            predicates,
            objects,
            self.triple_count(),
            self.context_len
        ));
        out.push_str(
            "  编码 M = Σ V(主体) ⊙ 置换¹(V(谓词)) ⊙ 置换²(V(客体))   ⊙ = 逐元素乘积（自逆绑定）\n",
        );
        out.push_str(
            "  角色用置换位置区分：解缠 ?客体 = 逆置换²( M ⊙ V(主) ⊙ 置换¹(V(谓)) )，再余弦净噪\n",
        );
        out.push_str(
            "  时序 C = Σ 置换^k(V(焦点))，反向置换即可按轮次回读；置换让不同轮次互不干扰\n",
        );
        if self.triple_count() > 0 {
            let check = self.self_check(12);
            out.push_str(&format!(
                "  自检：抽样 {} 条做客体召回，top-1 命中真实取值 {}/{}（{:.0}%）；多值键的 top-k 由硬核查过滤\n",
                check.total,
                check.top1_verified,
                check.total,
                if check.total == 0 {
                    0.0
                } else {
                    check.top1_verified as f64 / check.total as f64 * 100.0
                }
            ));
            for (subject, predicate, object, score) in check.examples.iter().take(5) {
                out.push_str(&format!(
                    "    样例 {} --{}--> {}   全息召回 {}（余弦 {:.3}，已核查✓）\n",
                    subject, predicate, object, object, score
                ));
            }
        } else {
            out.push_str("  尚无三元组：导入或教一些事实后再看全息自检。\n");
        }
        let trace = self.context_trace(6);
        if !trace.is_empty() {
            let rendered: Vec<String> = trace
                .iter()
                .map(|recall| {
                    format!(
                        "{} 轮前={}（{:.2}）",
                        recall.age, recall.symbol, recall.score
                    )
                })
                .collect();
            out.push_str(&format!("  焦点轨迹：{}\n", rendered.join(" ｜ ")));
        }
        out.push_str(
            "  神经符号整合：软通道（全息召回，分级余弦）只提出候选；硬通道（三元组登记）精确核查；\n",
        );
        out.push_str("  未获核查的候选标注「未登记」，不进入结论——用符号约束抑制串扰式幻觉。\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> HolographicMemory {
        HolographicMemory::new(4096, 0.01, 6)
    }

    fn feed(memory: &mut HolographicMemory, triples: &[(&str, &str, &str)]) {
        for (s, p, o) in triples {
            memory.observe_triple(s, p, o);
        }
    }

    #[test]
    fn test_symbol_vectors_deterministic_and_near_orthogonal() {
        let a1 = generate_vector("张三", 4096);
        let a2 = generate_vector("张三", 4096);
        let b = generate_vector("李四", 4096);
        let c = generate_vector("王五", 4096);
        assert_eq!(a1.as_slice(), a2.as_slice());
        let ab: f64 = a1.iter().zip(b.iter()).map(|(x, y)| (*x * *y) as f64).sum();
        let ac: f64 = a1.iter().zip(c.iter()).map(|(x, y)| (*x * *y) as f64).sum();
        assert!(
            ab.abs() / 4096.0 < 0.1,
            "独立符号向量余弦应接近 0，实际 {}",
            ab / 4096.0
        );
        assert!(ac.abs() / 4096.0 < 0.1);
    }

    #[test]
    fn test_bind_is_self_inverse() {
        let a = generate_vector("甲", 2048);
        let b = generate_vector("乙", 2048);
        let bound = bind_bipolar(&a, &b);
        let recovered = bind_bipolar(&bound, &b);
        assert_eq!(recovered.as_slice(), a.as_slice());
    }

    #[test]
    fn test_triple_recall_object_subject_predicate() {
        let mut memory = memory();
        feed(
            &mut memory,
            &[
                ("张三", "师傅", "李四"),
                ("李四", "师傅", "王五"),
                ("王五", "师傅", "赵六"),
                ("张三", "国籍", "中国"),
                ("李四", "国籍", "德国"),
            ],
        );

        let objects = memory.recall_object("张三", "师傅");
        assert_eq!(objects.first().map(|h| h.symbol.as_str()), Some("李四"));
        assert!(objects[0].verified);
        assert!(
            objects[0].score > 0.2,
            "余弦应显著为正：{}",
            objects[0].score
        );

        let subjects = memory.recall_subject("师傅", "王五");
        assert_eq!(subjects.first().map(|h| h.symbol.as_str()), Some("李四"));
        assert!(subjects[0].verified);

        let predicates = memory.recall_predicate("张三", "李四");
        assert_eq!(predicates.first().map(|h| h.symbol.as_str()), Some("师傅"));
        assert!(predicates[0].verified);
    }

    #[test]
    fn test_superposed_memory_recall_at_scale() {
        let mut memory = memory();
        // 20 个主体 × 3 谓词 × 3 客体 = 180 条三元组叠加进同一张全息图
        let subjects: Vec<String> = (0..20).map(|i| format!("实体{}", i)).collect();
        let predicates = ["属性A", "属性B", "属性C"];
        for (i, subject) in subjects.iter().enumerate() {
            for (j, predicate) in predicates.iter().enumerate() {
                for k in 0..3 {
                    let object = format!("值{}-{}-{}", i, j, k);
                    memory.observe_triple(subject, predicate, object.as_str());
                }
            }
        }
        assert_eq!(memory.triple_count(), 180);

        // 多值键 (s,p) 有 3 个真实客体：top-1 未必是抽样那条，但必须命中真实取值
        let check = memory.self_check(24);
        assert_eq!(check.total, 24);
        assert!(
            check.verified_rate() >= 0.95,
            "180 条叠加后 top-1 命中真实取值率应 ≥95%，实际 {}/{}",
            check.top1_verified,
            check.total
        );

        // 真实客体集合应整体占据 top-3（信号 ≫ 串扰噪声）
        for (i, subject) in subjects.iter().enumerate().step_by(4) {
            for (j, predicate) in predicates.iter().enumerate() {
                let hits = memory.recall_object(subject, predicate);
                let top3: Vec<&str> = hits.iter().take(3).map(|h| h.symbol.as_str()).collect();
                for k in 0..3 {
                    let expected = format!("值{}-{}-{}", i, j, k);
                    assert!(
                        top3.contains(&expected.as_str()),
                        "({}, {}) 的真实客体 {} 应进入 top-3，实际 top-3 = {:?}",
                        subject,
                        predicate,
                        expected,
                        top3
                    );
                }
            }
        }
    }

    #[test]
    fn test_role_permutation_prevents_conflation() {
        let mut memory = memory();
        // (王五, 师傅, 赵六) 与查询 "?--师傅-->王五" 共享符号但角色不同：
        // 置换编码必须让赵六拿不到满信号，否则会把"王五的师傅"当成"某人的师傅是王五"
        feed(
            &mut memory,
            &[
                ("李四", "师傅", "王五"),
                ("王五", "师傅", "赵六"),
                ("张三", "师傅", "李四"),
            ],
        );
        let subjects = memory.recall_subject("师傅", "王五");
        assert_eq!(
            subjects.first().map(|h| h.symbol.as_str()),
            Some("李四"),
            "角色置换后 top-1 应为真正的对象角色持有者：{:?}",
            subjects
        );
        let zhao = subjects
            .iter()
            .find(|h| h.symbol == "赵六")
            .map(|h| h.score)
            .unwrap_or(0.0);
        assert!(
            zhao < 0.1,
            "赵六只与查询共享角色错位的符号，不应获得高信号，实际 {:.3}",
            zhao
        );

        let objects = memory.recall_object("王五", "师傅");
        assert_eq!(objects.first().map(|h| h.symbol.as_str()), Some("赵六"));
    }

    #[test]
    fn test_analogy_transfers_relation() {
        let mut memory = memory();
        feed(
            &mut memory,
            &[
                ("张三", "师傅", "李四"),
                ("王五", "师傅", "赵六"),
                ("张三", "国籍", "中国"),
                ("李四", "国籍", "德国"),
            ],
        );

        let results = memory.analogy("张三", "李四", "王五");
        assert!(!results.is_empty(), "应至少召回一个类比候选");
        let top = &results[0];
        assert_eq!(top.symbol, "赵六");
        assert!(top.verified);
        assert!(top.via.iter().any(|(p, _)| p == "师傅"));

        let country = memory.analogy("张三", "中国", "李四");
        assert_eq!(country.first().map(|h| h.symbol.as_str()), Some("德国"));
    }

    #[test]
    fn test_context_temporal_recall_by_age() {
        let mut memory = memory();
        for entity in ["张三", "李四", "王五", "赵六", "钱七", "孙八"] {
            memory.push_focus(entity);
        }
        assert_eq!(memory.context_len(), 6);
        for (age, expected) in ["孙八", "钱七", "赵六", "王五", "李四", "张三"]
            .iter()
            .enumerate()
        {
            let recall = memory.recent_focus(age).expect("应能回读");
            assert_eq!(&recall.symbol, expected, "第 {} 轮前回读错误", age);
            assert!(recall.score > 0.1);
        }
        assert!(memory.recent_focus(6).is_none(), "超出记忆轮次应返回 None");
        // 连续重复焦点自动去重
        let before = memory.context_len();
        memory.push_focus("张三");
        memory.push_focus("张三");
        assert_eq!(memory.context_len(), before + 1);
    }

    #[test]
    fn test_observation_idempotent_and_unknown_symbols() {
        let mut memory = memory();
        feed(&mut memory, &[("张三", "师傅", "李四")]);
        let snapshot = memory.memory.clone();
        feed(&mut memory, &[("张三", "师傅", "李四")]);
        assert_eq!(memory.triple_count(), 1);
        assert_eq!(snapshot, memory.memory, "重复观察不应改变全息图");

        assert!(memory.recall_object("未登记实体", "师傅").is_empty());
        assert!(memory.recall_object("张三", "未登记谓词").is_empty());
        assert!(memory.analogy("张三", "李四", "未登记实体").is_empty());
    }

    #[test]
    fn test_report_contains_self_check() {
        let mut memory = memory();
        feed(
            &mut memory,
            &[
                ("张三", "师傅", "李四"),
                ("李四", "师傅", "王五"),
                ("张三", "国籍", "中国"),
            ],
        );
        let report = memory.report();
        assert!(report.contains("全息符号记忆"));
        assert!(report.contains("自检"));
        assert!(report.contains("神经符号整合"));
    }
}

//! 纪元失效的有界 LRU 缓存（Epoch-gated LRU）。
//!
//! 符号系统的缓存比神经缓存多一个筹码：失效必须由**显式纪元**驱动，
//! 而不是 TTL / 挂钟 / 启发式。本模块提供统一底座：
//!
//!   - `E` 是调用方定义的纪元（如 `u64` 知识版本，或 `CacheEpoch` 组合纪元）；
//!   - 条目只在 `epoch == 当前纪元` 时可命中；`put` 到新纪元时整体清空；
//!   - 容量溢出按「访问推进的 LRU 序」淘汰，顺序确定（`Vec` 维护，不用
//!     `HashSet` 遍历序），淘汰返回 1 以便调用方记 `evictions` 计数；
//!   - 命中/未命中全部由调用方入账（`PerfLedger`），本模块保持纯数据结构。
//!
//! 正确性契约：缓存只对纯函数计算启用；命中输出必须与重算输出逐字节一致。
//! 任何不确定（纪元不符、键不符、容量淘汰）一律按未命中处理。

use std::collections::HashMap;

/// 按纪元失效的有界 LRU 缓存：`V` 是值类型，`E` 是纪元类型（默认 `u64`）。
pub struct RevisionLru<V, E = u64> {
    epoch: Option<E>,
    capacity: usize,
    entries: HashMap<String, V>,
    order: Vec<String>,
}

impl<V, E: Copy + Eq> RevisionLru<V, E> {
    pub fn new(capacity: usize) -> Self {
        Self {
            epoch: None,
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// 当前纪元（尚无条目时为 `None`）。
    pub fn epoch(&self) -> Option<E> {
        self.epoch
    }

    /// 命中则触碰 LRU 并返回引用；纪元不符或未命中返回 None。
    pub fn get(&mut self, epoch: E, key: &str) -> Option<&V> {
        if self.epoch != Some(epoch) || !self.entries.contains_key(key) {
            return None;
        }
        self.order.retain(|cached| cached != key);
        self.order.push(key.to_string());
        self.entries.get(key)
    }

    /// 写入（返回被淘汰的条目数：0 或 1）。纪元变化时先清空（整体失效）。
    pub fn put(&mut self, epoch: E, key: String, value: V) -> usize {
        if self.epoch != Some(epoch) {
            self.entries.clear();
            self.order.clear();
            self.epoch = Some(epoch);
        }
        if self.entries.insert(key.clone(), value).is_none() {
            self.order.push(key);
        } else {
            self.order.retain(|cached| cached != &key);
            self.order.push(key);
        }
        if self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.entries.remove(&oldest);
                return 1;
            }
        }
        0
    }

    /// 人工失效：清空全部条目并回到「无纪元」状态（显式语义变化时使用）。
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.epoch = None;
    }

    /// 当前条目数（观测/测试用）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_invalidate_wholesale_on_epoch_change() {
        let mut cache: RevisionLru<u64, u64> = RevisionLru::new(4);
        assert!(cache.get(1, "a").is_none(), "未写入前任何查询都未命中");
        cache.put(1, "a".to_string(), 10);
        assert_eq!(cache.get(1, "a"), Some(&10));
        // 不同纪元：整体不可命中，即使键存在
        assert!(cache.get(2, "a").is_none());
        // 在新纪元写入会使所有已有条目整体失效
        cache.put(2, "b".to_string(), 20);
        assert!(cache.get(2, "a").is_none());
        assert_eq!(cache.get(2, "b"), Some(&20));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lru_eviction_order_is_deterministic_and_counted() {
        let mut cache: RevisionLru<u64, u64> = RevisionLru::new(2);
        assert_eq!(cache.put(7, "a".to_string(), 1), 0);
        assert_eq!(cache.put(7, "b".to_string(), 2), 0);
        // 触活 a：b 成为最久未用
        assert_eq!(cache.get(7, "a"), Some(&1));
        assert_eq!(cache.put(7, "c".to_string(), 3), 1, "容量 2 溢出淘汰 1 条");
        assert!(cache.get(7, "b").is_none(), "最久未用者被淘汰");
        assert_eq!(cache.get(7, "a"), Some(&1));
        assert_eq!(cache.get(7, "c"), Some(&3));
    }

    #[test]
    fn overwrite_same_key_does_not_grow_or_double_count() {
        let mut cache: RevisionLru<u64, u64> = RevisionLru::new(2);
        cache.put(1, "k".to_string(), 1);
        assert_eq!(cache.put(1, "k".to_string(), 2), 0);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(1, "k"), Some(&2));
    }

    #[test]
    fn composite_epochs_are_compared_as_a_whole() {
        #[derive(Clone, Copy, PartialEq, Eq)]
        struct Epoch {
            kb: u64,
            identity: u64,
        }
        let mut cache: RevisionLru<&str, Epoch> = RevisionLru::new(2);
        let prod = Epoch { kb: 3, identity: 0 };
        cache.put(prod, "q".to_string(), "answer");
        assert_eq!(cache.get(prod, "q"), Some(&"answer"));
        let admitted = Epoch { kb: 3, identity: 1 };
        assert!(
            cache.get(admitted, "q").is_none(),
            "身份纪元推进必须整体失效"
        );
    }
}

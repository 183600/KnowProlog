//! 预测答案草稿（Predictive Answer Cache）。
//!
//! 链接预测本身是纯符号计算：同一份事实、同一批规则与同一组参数必然给出
//! 同一批候选。轮末把焦点实体的候选提前算好，下一轮直接在纪元内复用，
//! 就不必在回答热路径上重复展开规则。
//!
//! 契约与 [`crate::cache::RevisionLru`] 一致：
//!   - 草稿只缓存纯函数结果，命中与重算必须逐字节一致；
//!   - 纪元推进（知识版本/规则重建）时整体不可命中；
//!   - 草稿不是知识：不写图谱、不确认候选、不改变回答文本；
//!   - 键包含实体、谓词过滤与上限，任何一项不同都按未命中处理。

use crate::cache::RevisionLru;
use crate::link_prediction::Prediction;

/// 预测草稿的键：实体 + 可选谓词 + 返回上限。
///
/// 采用长度前缀拼接，避免 `a#b` 与 `a` + `b` 之类的拼接歧义。
fn draft_key(entity: &str, predicate: Option<&str>, limit: usize) -> String {
    let mut key = String::with_capacity(entity.len() + predicate.map_or(1, str::len) + 24);
    key.push_str(&limit.to_string());
    key.push('#');
    key.push_str(&entity.len().to_string());
    key.push('#');
    key.push_str(entity);
    match predicate {
        Some(predicate) => {
            key.push('1');
            key.push_str(&predicate.len().to_string());
            key.push('#');
            key.push_str(predicate);
        }
        None => key.push('0'),
    }
    key
}

/// 按知识纪元失效的预测草稿表。
pub struct PredictionDrafts {
    entries: RevisionLru<Vec<Prediction>>,
}

impl PredictionDrafts {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RevisionLru::new(capacity),
        }
    }

    /// 命中的草稿是只读候选列表；调用方仍需按当前规则做证据复核。
    pub fn get(
        &mut self,
        epoch: u64,
        entity: &str,
        predicate: Option<&str>,
        limit: usize,
    ) -> Option<Vec<Prediction>> {
        self.entries
            .get(epoch, &draft_key(entity, predicate, limit))
            .cloned()
    }

    /// 写入草稿，返回被淘汰的条目数（0 或 1）。
    pub fn put(
        &mut self,
        epoch: u64,
        entity: &str,
        predicate: Option<&str>,
        limit: usize,
        predictions: Vec<Prediction>,
    ) -> usize {
        self.entries.put(
            epoch,
            draft_key(entity, predicate, limit),
            predictions,
        )
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

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
    use crate::link_prediction::Triple;

    fn prediction(object: &str) -> Prediction {
        Prediction {
            triple: Triple::new("张三", "师傅", object),
            rule_index: 0,
            confidence: 0.8,
            evidence: vec![Triple::new("张三", "师傅", object)],
        }
    }

    #[test]
    fn keys_distinguish_entity_predicate_and_limit() {
        let mut drafts = PredictionDrafts::new(8);
        drafts.put(1, "张三", Some("师傅"), 5, vec![prediction("李四")]);

        assert_eq!(
            drafts
                .get(1, "张三", Some("师傅"), 5)
                .expect("完全相同的键应命中")
                .len(),
            1
        );
        assert!(drafts.get(1, "张三", None, 5).is_none());
        assert!(drafts.get(1, "张三", Some("师祖"), 5).is_none());
        assert!(drafts.get(1, "张三", Some("师傅"), 6).is_none());
        assert!(drafts.get(1, "李四", Some("师傅"), 5).is_none());
    }

    #[test]
    fn epoch_change_invalidates_even_for_identical_key() {
        let mut drafts = PredictionDrafts::new(8);
        drafts.put(7, "张三", None, 5, vec![prediction("李四")]);
        assert!(drafts.get(8, "张三", None, 5).is_none());
        drafts.put(8, "张三", None, 5, vec![prediction("王五")]);
        let predictions = drafts.get(8, "张三", None, 5).expect("新纪元应命中");
        assert_eq!(predictions[0].triple.object, "王五");
        assert_eq!(drafts.len(), 1, "纪元推进时旧草稿应整体清空");
    }

    #[test]
    fn empty_result_is_a_valid_draft() {
        let mut drafts = PredictionDrafts::new(8);
        drafts.put(1, "孤立实体", Some("师傅"), 5, Vec::new());
        let cached = drafts
            .get(1, "孤立实体", Some("师傅"), 5)
            .expect("空结果也应缓存，避免反复枚举");
        assert!(cached.is_empty());
    }

    #[test]
    fn key_prefixes_do_not_collide() {
        let mut drafts = PredictionDrafts::new(8);
        drafts.put(1, "a#1", Some("b"), 5, vec![prediction("甲")]);
        drafts.put(1, "a", Some("1#b"), 5, vec![prediction("乙")]);
        assert_eq!(
            drafts.get(1, "a#1", Some("b"), 5).unwrap()[0].triple.object,
            "甲"
        );
        assert_eq!(
            drafts.get(1, "a", Some("1#b"), 5).unwrap()[0].triple.object,
            "乙"
        );
    }

    #[test]
    fn capacity_evicts_least_recently_used() {
        let mut drafts = PredictionDrafts::new(2);
        drafts.put(1, "a", None, 5, vec![prediction("甲")]);
        drafts.put(1, "b", None, 5, vec![prediction("乙")]);
        assert!(drafts.get(1, "a", None, 5).is_some());
        assert_eq!(
            drafts.put(1, "c", None, 5, vec![prediction("丙")]),
            1,
            "容量 2 溢出应淘汰最久未用的一条"
        );
        assert!(drafts.get(1, "b", None, 5).is_none());
        assert!(drafts.get(1, "a", None, 5).is_some());
    }
}

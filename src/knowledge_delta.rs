use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::context_image::{fnv1a64_update, FNV_OFFSET};
use crate::prolog_engine::PrologFact;

pub const DEFAULT_JOURNAL_CAPACITY: usize = 128;

fn fnv1a_field(hash: u64, value: &str) -> u64 {
    fnv1a64_update(fnv1a64_update(hash, value.as_bytes()), b"\x1f")
}

pub fn fingerprint_facts(facts: &[PrologFact]) -> String {
    let mut hash = FNV_OFFSET;
    for fact in facts {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                hash = fnv1a64_update(hash, b"E\x1f");
                hash = fnv1a_field(hash, name);
                hash = fnv1a_field(hash, entity_type);
                let mut keys: Vec<&String> = properties.keys().collect();
                keys.sort();
                for key in keys {
                    hash = fnv1a_field(hash, key);
                    hash = fnv1a_field(hash, &properties[key]);
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                hash = fnv1a64_update(hash, b"R\x1f");
                hash = fnv1a_field(hash, from);
                hash = fnv1a_field(hash, rel_type);
                hash = fnv1a_field(hash, to);
            }
        }
        hash = fnv1a64_update(hash, b"\n");
    }
    format!("{:016x}", hash)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeChange {
    Facts {
        generation: u64,
        facts: Vec<PrologFact>,
        source: String,
    },
    Aliases {
        generation: u64,
        aliases: Vec<(String, String)>,
        source: String,
    },
    Reset {
        generation: u64,
        source: String,
    },
}

impl KnowledgeChange {
    pub fn generation(&self) -> u64 {
        match self {
            KnowledgeChange::Facts { generation, .. }
            | KnowledgeChange::Aliases { generation, .. }
            | KnowledgeChange::Reset { generation, .. } => *generation,
        }
    }

    pub fn source(&self) -> &str {
        match self {
            KnowledgeChange::Facts { source, .. }
            | KnowledgeChange::Aliases { source, .. }
            | KnowledgeChange::Reset { source, .. } => source,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            KnowledgeChange::Facts { .. } => "facts",
            KnowledgeChange::Aliases { .. } => "aliases",
            KnowledgeChange::Reset { .. } => "reset",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgeSnapshot {
    pub generation: u64,
    pub facts: Arc<Vec<PrologFact>>,
    pub aliases: Vec<(String, String)>,
    pub fingerprint: String,
    pub journal_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeCatchup {
    UpToDate,
    Incremental(Vec<KnowledgeChange>),
    Rebuild,
}

struct KnowledgeState {
    generation: u64,
    facts: Arc<Vec<PrologFact>>,
    aliases: Vec<(String, String)>,
    journal: VecDeque<KnowledgeChange>,
    fingerprint: String,
}

pub struct KnowledgeBase {
    state: RwLock<KnowledgeState>,
    journal_capacity: usize,
}

impl KnowledgeBase {
    pub fn new(facts: Vec<PrologFact>) -> Self {
        Self::with_journal_capacity(facts, DEFAULT_JOURNAL_CAPACITY)
    }

    pub fn with_journal_capacity(facts: Vec<PrologFact>, journal_capacity: usize) -> Self {
        let fingerprint = fingerprint_facts(&facts);
        Self {
            state: RwLock::new(KnowledgeState {
                generation: 0,
                facts: Arc::new(facts),
                aliases: Vec::new(),
                journal: VecDeque::new(),
                fingerprint,
            }),
            journal_capacity: journal_capacity.max(1),
        }
    }

    /// 从基线快照 + 增量重建知识库（检查点 + 日志重放）。
    ///
    /// 只接受连续代数且不含 `Reset` 的增量；整体替换无法仅凭日志重建，
    /// 返回 `None`，调用方应改用新的全量快照。重建后的日志只包含本次重放
    /// 的增量，因此 `journal_len` 如实反映新库的日志窗口。
    pub fn from_snapshot(
        snapshot: &KnowledgeSnapshot,
        changes: &[KnowledgeChange],
    ) -> Option<Self> {
        let mut facts = (*snapshot.facts).clone();
        let mut aliases = snapshot.aliases.clone();
        let mut generation = snapshot.generation;
        for change in changes {
            if change.generation() != generation.saturating_add(1) {
                return None;
            }
            match change {
                KnowledgeChange::Facts { facts: batch, .. } => {
                    for fact in batch {
                        if !facts.contains(fact) {
                            facts.push(fact.clone());
                        }
                    }
                }
                KnowledgeChange::Aliases { aliases: batch, .. } => {
                    for pair in batch {
                        if !aliases.contains(pair) {
                            aliases.push(pair.clone());
                        }
                    }
                }
                KnowledgeChange::Reset { .. } => return None,
            }
            generation = change.generation();
        }
        let journal_capacity = DEFAULT_JOURNAL_CAPACITY;
        let mut journal: VecDeque<KnowledgeChange> = changes.iter().cloned().collect();
        while journal.len() > journal_capacity {
            journal.pop_front();
        }
        let fingerprint = fingerprint_facts(&facts);
        Some(Self {
            state: RwLock::new(KnowledgeState {
                generation,
                facts: Arc::new(facts),
                aliases,
                journal,
                fingerprint,
            }),
            journal_capacity,
        })
    }

    pub fn generation(&self) -> u64 {
        self.state.read().generation
    }

    pub fn snapshot(&self) -> KnowledgeSnapshot {
        let state = self.state.read();
        KnowledgeSnapshot {
            generation: state.generation,
            facts: Arc::clone(&state.facts),
            aliases: state.aliases.clone(),
            fingerprint: state.fingerprint.clone(),
            journal_len: state.journal.len(),
        }
    }

    pub fn journal_since(&self, generation: u64) -> Option<Vec<KnowledgeChange>> {
        let state = self.state.read();
        if generation == state.generation {
            return Some(Vec::new());
        }
        if generation > state.generation {
            return None;
        }
        let changes: Vec<KnowledgeChange> = state
            .journal
            .iter()
            .filter(|change| change.generation() > generation)
            .cloned()
            .collect();
        match changes.first() {
            Some(first) if first.generation() == generation.saturating_add(1) => Some(changes),
            _ => None,
        }
    }

    pub fn changes_since(&self, generation: u64) -> KnowledgeCatchup {
        let changes = match self.journal_since(generation) {
            Some(changes) => changes,
            None => return KnowledgeCatchup::Rebuild,
        };
        if changes.is_empty() {
            return KnowledgeCatchup::UpToDate;
        }
        if changes
            .iter()
            .any(|change| matches!(change, KnowledgeChange::Reset { .. }))
        {
            return KnowledgeCatchup::Rebuild;
        }
        KnowledgeCatchup::Incremental(changes)
    }

    pub fn missing_facts(&self, facts: &[PrologFact]) -> Vec<PrologFact> {
        let state = self.state.read();
        let mut missing: Vec<PrologFact> = Vec::new();
        for fact in facts {
            if state.facts.iter().any(|existing| existing == fact)
                || missing.iter().any(|existing| existing == fact)
            {
                continue;
            }
            missing.push(fact.clone());
        }
        missing
    }

    pub fn assert_facts(&self, facts: &[PrologFact], source: &str) -> Vec<PrologFact> {
        let mut state = self.state.write();
        let mut admitted: Vec<PrologFact> = Vec::new();
        for fact in facts {
            if state.facts.iter().any(|existing| existing == fact)
                || admitted.iter().any(|existing| existing == fact)
            {
                continue;
            }
            admitted.push(fact.clone());
        }
        if admitted.is_empty() {
            return admitted;
        }
        let mut merged = (*state.facts).clone();
        merged.extend(admitted.iter().cloned());
        state.facts = Arc::new(merged);
        state.generation = state.generation.saturating_add(1);
        state.fingerprint = fingerprint_facts(&state.facts);
        let change = KnowledgeChange::Facts {
            generation: state.generation,
            facts: admitted.clone(),
            source: source.to_string(),
        };
        state.journal.push_back(change);
        while state.journal.len() > self.journal_capacity {
            state.journal.pop_front();
        }
        admitted
    }

    pub fn replace_facts(&self, facts: Vec<PrologFact>, source: &str) -> bool {
        let mut state = self.state.write();
        let mut unique: Vec<PrologFact> = Vec::with_capacity(facts.len());
        for fact in facts {
            if !unique.contains(&fact) {
                unique.push(fact);
            }
        }
        if state.facts.as_slice() == unique.as_slice() {
            return false;
        }
        state.facts = Arc::new(unique);
        state.fingerprint = fingerprint_facts(&state.facts);
        state.generation = state.generation.saturating_add(1);
        let change = KnowledgeChange::Reset {
            generation: state.generation,
            source: source.to_string(),
        };
        state.journal.push_back(change);
        while state.journal.len() > self.journal_capacity {
            state.journal.pop_front();
        }
        true
    }

    pub fn record_aliases(
        &self,
        aliases: &[(String, String)],
        source: &str,
    ) -> Vec<(String, String)> {
        let mut state = self.state.write();
        let mut admitted: Vec<(String, String)> = Vec::new();
        for (surface, canonical) in aliases {
            let pair = (surface.trim().to_string(), canonical.trim().to_string());
            if pair.0.is_empty()
                || pair.1.is_empty()
                || pair.0 == pair.1
                || state.aliases.contains(&pair)
                || admitted.contains(&pair)
            {
                continue;
            }
            admitted.push(pair);
        }
        if admitted.is_empty() {
            return admitted;
        }
        state.aliases.extend(admitted.iter().cloned());
        state.generation = state.generation.saturating_add(1);
        let change = KnowledgeChange::Aliases {
            generation: state.generation,
            aliases: admitted.clone(),
            source: source.to_string(),
        };
        state.journal.push_back(change);
        while state.journal.len() > self.journal_capacity {
            state.journal.pop_front();
        }
        admitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn entity(name: &str, etype: &str) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: HashMap::new(),
        }
    }

    fn relation(from: &str, rel: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: rel.to_string(),
            to: to.to_string(),
        }
    }

    #[test]
    fn append_is_deduplicated_and_versioned() {
        let base = vec![entity("姚明", "人物")];
        let knowledge = KnowledgeBase::new(base);
        assert_eq!(knowledge.generation(), 0);
        let admitted = knowledge.assert_facts(
            &[
                relation("姚明", "队友", "叶莉"),
                relation("姚明", "队友", "叶莉"),
                entity("姚明", "人物"),
            ],
            "test",
        );
        assert_eq!(admitted.len(), 1);
        assert_eq!(knowledge.generation(), 1);
        assert_eq!(knowledge.snapshot().facts.len(), 2);
        assert!(knowledge
            .assert_facts(&[relation("姚明", "队友", "叶莉")], "test")
            .is_empty());
        assert_eq!(knowledge.generation(), 1);
    }

    #[test]
    fn incremental_catchup_reports_exact_changes() {
        let knowledge = KnowledgeBase::new(Vec::new());
        knowledge.assert_facts(&[relation("甲", "认识", "乙")], "one");
        knowledge.record_aliases(&[("乙乙".to_string(), "乙".to_string())], "two");
        let changes = match knowledge.changes_since(0) {
            KnowledgeCatchup::Incremental(changes) => changes,
            other => panic!("expected incremental, got {:?}", other),
        };
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].generation(), 1);
        assert_eq!(changes[0].kind(), "facts");
        assert_eq!(changes[1].generation(), 2);
        assert_eq!(changes[1].kind(), "aliases");
        assert_eq!(knowledge.changes_since(2), KnowledgeCatchup::UpToDate);
        assert_eq!(knowledge.changes_since(3), KnowledgeCatchup::Rebuild);
    }

    #[test]
    fn snapshot_plus_incremental_changes_rebuilds_same_state() {
        let knowledge = KnowledgeBase::new(vec![entity("姚明", "人物")]);
        let base = knowledge.snapshot();
        knowledge.assert_facts(&[relation("姚明", "队友", "叶莉")], "one");
        knowledge.record_aliases(&[("明哥".to_string(), "姚明".to_string())], "two");
        knowledge.assert_facts(&[entity("叶莉", "人物")], "three");
        let current = knowledge.snapshot();
        let changes = match knowledge.changes_since(base.generation) {
            KnowledgeCatchup::Incremental(changes) => changes,
            other => panic!("expected incremental, got {:?}", other),
        };
        let replayed = KnowledgeBase::from_snapshot(&base, &changes).expect("应可重放");
        let rebuilt = replayed.snapshot();
        assert_eq!(rebuilt.generation, current.generation);
        assert_eq!(rebuilt.facts.as_slice(), current.facts.as_slice());
        assert_eq!(rebuilt.aliases, current.aliases);
        assert_eq!(rebuilt.fingerprint, current.fingerprint);
        assert_eq!(
            replayed.changes_since(base.generation),
            KnowledgeCatchup::Incremental(changes)
        );
    }

    #[test]
    fn replay_rejects_reset_and_generation_gaps() {
        let knowledge = KnowledgeBase::new(Vec::new());
        let base = knowledge.snapshot();
        knowledge.assert_facts(&[relation("甲", "认识", "乙")], "one");
        let changes = knowledge.journal_since(base.generation).unwrap();
        let replayed = KnowledgeBase::from_snapshot(&base, &changes).expect("应可重放");
        assert_eq!(replayed.generation(), 1);

        knowledge.replace_facts(vec![relation("甲", "认识", "丙")], "reset");
        let with_reset = knowledge.journal_since(base.generation).unwrap();
        assert!(KnowledgeBase::from_snapshot(&base, &with_reset).is_none());

        let skipped = vec![KnowledgeChange::Facts {
            generation: 3,
            facts: vec![relation("甲", "认识", "丁")],
            source: "gap".to_string(),
        }];
        assert!(KnowledgeBase::from_snapshot(&base, &skipped).is_none());
    }

    #[test]
    fn pruned_journal_requires_rebuild() {
        let knowledge = KnowledgeBase::with_journal_capacity(Vec::new(), 1);
        knowledge.assert_facts(&[relation("甲", "认识", "乙")], "one");
        knowledge.assert_facts(&[relation("甲", "认识", "丙")], "two");
        assert_eq!(knowledge.changes_since(0), KnowledgeCatchup::Rebuild);
        assert!(matches!(
            knowledge.changes_since(1),
            KnowledgeCatchup::Incremental(_)
        ));
    }

    #[test]
    fn replace_forces_rebuild_and_changes_fingerprint() {
        let knowledge = KnowledgeBase::new(vec![relation("姚明", "队友", "叶莉")]);
        let before = knowledge.snapshot();
        assert!(knowledge.replace_facts(vec![relation("姚明", "队友", "王五")], "fix"));
        let after = knowledge.snapshot();
        assert_ne!(before.fingerprint, after.fingerprint);
        assert_eq!(after.generation, 1);
        assert_eq!(knowledge.changes_since(0), KnowledgeCatchup::Rebuild);
        let journal = knowledge.journal_since(0).unwrap();
        assert_eq!(journal.len(), 1);
        assert_eq!(journal[0].kind(), "reset");
        assert!(!knowledge.replace_facts(vec![relation("姚明", "队友", "王五")], "fix"));
        assert_eq!(knowledge.generation(), 1);
    }

    #[test]
    fn alias_records_are_deduplicated() {
        let knowledge = KnowledgeBase::new(Vec::new());
        let aliases = vec![
            ("明哥".to_string(), "姚明".to_string()),
            ("明哥".to_string(), "姚明".to_string()),
            (" ".to_string(), "姚明".to_string()),
        ];
        assert_eq!(knowledge.record_aliases(&aliases, "test").len(), 1);
        assert!(knowledge.record_aliases(&aliases, "test").is_empty());
        assert_eq!(knowledge.snapshot().aliases.len(), 1);
        assert_eq!(knowledge.generation(), 1);
    }

    #[test]
    fn fingerprint_is_order_independent_for_properties() {
        let mut first = HashMap::new();
        first.insert("身高".to_string(), "226".to_string());
        first.insert("国籍".to_string(), "中国".to_string());
        let mut second = HashMap::new();
        second.insert("国籍".to_string(), "中国".to_string());
        second.insert("身高".to_string(), "226".to_string());
        let a = vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人物".to_string(),
            properties: first,
        }];
        let b = vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人物".to_string(),
            properties: second,
        }];
        assert_eq!(fingerprint_facts(&a), fingerprint_facts(&b));
    }
}

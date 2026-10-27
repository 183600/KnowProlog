use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::prolog_engine::PrologFact;

pub const DEFAULT_JOURNAL_CAPACITY: usize = 128;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_step(hash: u64, bytes: &[u8]) -> u64 {
    let mut hash = hash;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a_field(hash: u64, value: &str) -> u64 {
    fnv1a_step(fnv1a_step(hash, value.as_bytes()), b"\x1f")
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
                hash = fnv1a_step(hash, b"E\x1f");
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
                hash = fnv1a_step(hash, b"R\x1f");
                hash = fnv1a_field(hash, from);
                hash = fnv1a_field(hash, rel_type);
                hash = fnv1a_field(hash, to);
            }
        }
        hash = fnv1a_step(hash, b"\n");
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
}

impl KnowledgeChange {
    pub fn generation(&self) -> u64 {
        match self {
            KnowledgeChange::Facts { generation, .. }
            | KnowledgeChange::Aliases { generation, .. } => *generation,
        }
    }

    pub fn source(&self) -> &str {
        match self {
            KnowledgeChange::Facts { source, .. }
            | KnowledgeChange::Aliases { source, .. } => source,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            KnowledgeChange::Facts { .. } => "facts",
            KnowledgeChange::Aliases { .. } => "aliases",
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

    pub fn changes_since(&self, generation: u64) -> KnowledgeCatchup {
        let state = self.state.read();
        if generation == state.generation {
            return KnowledgeCatchup::UpToDate;
        }
        if generation > state.generation {
            return KnowledgeCatchup::Rebuild;
        }
        let changes: Vec<KnowledgeChange> = state
            .journal
            .iter()
            .filter(|change| change.generation() > generation)
            .cloned()
            .collect();
        match changes.first() {
            Some(first) if first.generation() == generation.saturating_add(1) => {
                KnowledgeCatchup::Incremental(changes)
            }
            _ => KnowledgeCatchup::Rebuild,
        }
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

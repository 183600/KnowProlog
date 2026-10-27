/// 计算命中所有证书支撑集的最小事实集合。
///
/// 候选 ≤ [`MAX_EXACT_CANDIDATES`] 时精确枚举（同一大小取字典序最小，保证
/// 确定性）；超过时按覆盖数贪心并标记 `exact = false`。支撑集为空的证书
/// （来源与事实都缺失）计入 `unresolved`，不被静默忽略。
pub fn plan_retraction(certificates: &[ConflictCertificate]) -> RetractionPlan {
    let mut sets: Vec<BTreeSet<FactRef>> = Vec::new();
    let mut unresolved = 0usize;
    let mut candidates: BTreeSet<FactRef> = BTreeSet::new();
    for certificate in certificates {
        let support = certificate.support_facts();
        if support.is_empty() {
            unresolved += 1;
            continue;
        }
        candidates.extend(support.iter().cloned());
        sets.push(support);
    }
    if sets.is_empty() {
        return RetractionPlan {
            facts: Vec::new(),
            exact: true,
            candidates: 0,
            resolved: 0,
            unresolved,
        };
    }

    let candidates: Vec<FactRef> = candidates.into_iter().collect();
    let (selected, exact) = if candidates.len() <= MAX_EXACT_CANDIDATES {
        (
            exact_hitting_set(&sets, &candidates).unwrap_or_default(),
            true,
        )
    } else {
        (greedy_hitting_set(&sets, &candidates), false)
    };
    let selected_sets: BTreeSet<&FactRef> = selected.iter().collect();
    let resolved = sets
        .iter()
        .filter(|set| set.iter().any(|fact| selected_sets.contains(fact)))
        .count();
    RetractionPlan {
        facts: selected,
        exact,
        candidates: candidates.len(),
        resolved,
        unresolved,
    }
}
fn exact_hitting_set(
    sets: &[BTreeSet<FactRef>],
    candidates: &[FactRef],
) -> Option<Vec<FactRef>> {
    let count = candidates.len();
    if count == 0 {
        return None;
    }
    let mut best: Option<(u32, Vec<usize>)> = None;
    for mask in 1u32..(1u32 << count) {
        let size = mask.count_ones();
        if let Some((best_size, _)) = &best {
            if size > *best_size {
                continue;
            }
        }
        let hits = sets.iter().all(|set| {
            set.iter().any(|fact| {
                candidates
                    .iter()
                    .position(|candidate| candidate == fact)
                    .map(|index| mask & (1 << index) != 0)
                    .unwrap_or(false)
            })
        });
        if !hits {
            continue;
        }
        let indices: Vec<usize> = (0..count).filter(|index| mask & (1 << index) != 0).collect();
        let replace = match &best {
            None => true,
            Some((best_size, best_indices)) => {
                size < *best_size || (size == *best_size && indices < *best_indices)
            }
        };
        if replace {
            best = Some((size, indices));
        }
    }
    best.map(|(_, indices)| indices.into_iter().map(|index| candidates[index].clone()).collect())
}
fn greedy_hitting_set(sets: &[BTreeSet<FactRef>], candidates: &[FactRef]) -> Vec<FactRef> {
    let mut uncovered: Vec<&BTreeSet<FactRef>> = sets.iter().collect();
    let mut selected: Vec<FactRef> = Vec::new();
    while !uncovered.is_empty() {
        let mut best: Option<(usize, &FactRef)> = None;
        for candidate in candidates {
            let covered = uncovered.iter().filter(|set| set.contains(candidate)).count();
            let replace = match best {
                None => covered > 0,
                Some((best_covered, best_candidate)) => {
                    covered > best_covered
                        || (covered == best_covered && candidate < best_candidate)
                }
            };
            if replace {
                best = Some((covered, candidate));
            }
        }
        match best {
            Some((_, candidate)) if !selected.contains(candidate) => {
                selected.push(candidate.clone());
                uncovered.retain(|set| !set.contains(candidate));
            }
            _ => break,
        }
    }
    selected
}

// ---------------------------------------------------------------------------
// 与规则/派生账本对接
// ---------------------------------------------------------------------------
/// 把图谱事实、事实来源台账、规则账本与派生账本拼成支撑树。
///
/// 派生事实的来源由规则账本给出（规则出处 + 前提链），而不是谓词画像标签；
/// 没有派生记录的普通事实回落到 [`ProvenanceLedger`] 的事实级来源。两者都没有
/// 时保持空来源，让后续证书如实报告「来源未登记」。
pub fn build_supports(
    facts: &[FactRef],
    ledger: &ProvenanceLedger,
    rules: &RuleLedger,
    derived: &DerivedLedger,
) -> BTreeMap<String, Support> {
    let mut derived_by_fact: BTreeMap<String, &crate::graph_ledger::DerivedLedgerEntry> =
        BTreeMap::new();
    for entry in &derived.entries {
        if let Some(fact) = ledger_atom_fact(&entry.atom) {
            derived_by_fact.entry(fact.key()).or_insert(entry);
        }
    }
    let mut rules_by_source: BTreeMap<&str, &crate::graph_ledger::RuleLedgerEntry> = BTreeMap::new();
    for entry in &rules.entries {
        rules_by_source.entry(entry.source.as_str()).or_insert(entry);
    }

    let mut out: BTreeMap<String, Support> = BTreeMap::new();
    for fact in facts {
        let support = build_support_with(
            fact,
            ledger,
            &derived_by_fact,
            &rules_by_source,
            &mut BTreeSet::new(),
            0,
        );
        out.insert(fact.key(), support);
    }
    // 派生账本里的结论即使尚未出现在快照里也纳入（潜在/已见证判定都用得到）。
    for entry in &derived.entries {
        if let Some(fact) = ledger_atom_fact(&entry.atom) {
            if !out.contains_key(&fact.key()) {
                let support = build_support_with(
                    &fact,
                    ledger,
                    &derived_by_fact,
                    &rules_by_source,
                    &mut BTreeSet::new(),
                    0,
                );
                out.insert(fact.key(), support);
            }
        }
    }
    out
}
fn build_support_with(
    fact: &FactRef,
    ledger: &ProvenanceLedger,
    derived_by_fact: &BTreeMap<String, &crate::graph_ledger::DerivedLedgerEntry>,
    rules_by_source: &BTreeMap<&str, &crate::graph_ledger::RuleLedgerEntry>,
    visiting: &mut BTreeSet<String>,
    depth: usize,
) -> Support {
    if depth >= MAX_SUPPORT_DEPTH || !visiting.insert(fact.key()) {
        return ledger.support(fact.clone());
    }
    let support = match derived_by_fact.get(&fact.key()) {
        Some(entry) => {
            let (source, rule_id, confidence) = match rules_by_source.get(entry.rule_source.as_str())
            {
                Some(rule) => (
                    SourceId::new(SourceKind::Rule, &rule.source, &rule_canonical(rule)),
                    rule.id.clone(),
                    rule.confidence,
                ),
                None => (
                    SourceId::new(SourceKind::Rule, &entry.rule_source, &entry.id),
                    entry.rule_source.clone(),
                    entry.confidence,
                ),
            };
            let premises: Vec<Support> = entry
                .premises
                .iter()
                .filter_map(ledger_atom_fact)
                .map(|premise| {
                    build_support_with(
                        &premise,
                        ledger,
                        derived_by_fact,
                        rules_by_source,
                        visiting,
                        depth + 1,
                    )
                })
                .collect();
            if premises.is_empty() {
                ledger.support(fact.clone())
            } else {
                Support::derived(fact.clone(), source, &rule_id, confidence, premises)
            }
        }
        None => ledger.support(fact.clone()),
    };
    visiting.remove(&fact.key());
    support
}
fn rule_canonical(entry: &crate::graph_ledger::RuleLedgerEntry) -> String {
    let body = entry
        .body
        .iter()
        .map(LedgerAtom::display)
        .collect::<Vec<_>>()
        .join("&");
    format!("{}:-{}", entry.head.display(), body)
}
fn ledger_atom_fact(atom: &LedgerAtom) -> Option<FactRef> {
    match atom.terms.as_slice() {
        [subject, value] => Some(FactRef::new(
            &ledger_term_name(subject),
            &atom.predicate,
            &ledger_term_name(value),
        )),
        [subject] => Some(FactRef::new(
            &ledger_term_name(subject),
            &atom.predicate,
            "",
        )),
        _ => None,
    }
}
fn ledger_term_name(term: &crate::graph_ledger::LedgerTerm) -> String {
    if term.is_var {
        format!("?{}", term.name)
    } else {
        term.name.clone()
    }
}
fn local_from_iri(iri: &str) -> String {
    let base = crate::identity::DEFAULT_BASE_NS;
    crate::identity::entity_local_name(base, iri)
        .or_else(|| crate::identity::local_name(base, iri))
        .unwrap_or_else(|| iri.to_string())
}
fn predicate_from_iri(iri: &str) -> String {
    if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
        "type".to_string()
    } else {
        local_from_iri(iri)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn source(locator: &str, content: &str) -> SourceId {
        SourceId::new(SourceKind::Import, locator, content)
    }

    fn fact(subject: &str, predicate: &str, value: &str) -> FactRef {
        FactRef::new(subject, predicate, value)
    }

    fn supported(fact: FactRef, sources: &[SourceId]) -> Support {
        Support::fact(fact, sources.iter().cloned().collect())
    }

    fn empty_rules() -> RuleLedger {
        RuleLedger {
            synced: false,
            fingerprint: String::new(),
            entries: Vec::new(),
        }
    }

    fn empty_derived() -> DerivedLedger {
        DerivedLedger {
            materialized: false,
            revision: 0,
            fingerprint: String::new(),
            entries: Vec::new(),
        }
    }

    fn schema() -> ConflictSchema {
        let mut schema = ConflictSchema::new();
        schema.functional.insert("年龄".to_string());
        schema.disjoint.insert(("人物".to_string(), "地点".to_string()));
        schema
    }

    #[test]
    fn source_identity_separates_category_locator_and_content() {
        let first = source("体测表", "张三 38");
        let same = source("体测表", "张三 38");
        let different_content = source("体测表", "张三 39");
        let different_locator = source("体检表", "张三 38");
        assert_eq!(first, same, "同一定位符 + 同一内容 = 同一身份");
        assert_ne!(first, different_content, "内容摘要必须参与身份");
        assert_ne!(first, different_locator, "定位符必须参与身份");
        assert!(first.is_registered());
        assert!(!SourceId::unknown().is_registered());
        assert_eq!(SourceId::unknown().render(), "来源未登记");
        assert!(first.render().contains("导入:体测表#"));
    }

    #[test]
    fn source_kind_parses_human_labels_and_rejects_typos() {
        assert_eq!(SourceKind::parse("import"), Some(SourceKind::Import));
        assert_eq!(SourceKind::parse("导入"), Some(SourceKind::Import));
        assert_eq!(SourceKind::parse("人工确认"), Some(SourceKind::Manual));
        assert_eq!(SourceKind::parse("unknown"), Some(SourceKind::Unknown));
        assert_eq!(SourceKind::parse("导入n"), None);
    }

    #[test]
    fn ledger_joins_sources_and_reports_honest_coverage() {
        let first = fact("张三", "年龄", "38");
        let second = fact("张三", "师傅", "李四");
        let mut ledger = ProvenanceLedger::new();
        let a = source("甲批", "a");
        let b = source("乙批", "b");
        assert!(ledger.record(first.clone(), a.clone()));
        assert!(!ledger.record(first.clone(), a.clone()), "重复登记幂等");
        assert!(ledger.record(first.clone(), b.clone()), "同一事实可多来源");
        assert_eq!(ledger.sources(&first).len(), 2, "多来源必须保留全部");
        let coverage = ledger.coverage(&[first.clone(), second.clone()]);
        assert_eq!(coverage.total, 2);
        assert_eq!(coverage.registered, 1);
        assert_eq!(coverage.unregistered, vec![second]);
        assert!((coverage.ratio - 0.5).abs() < 1e-9);
        assert!(coverage.render_zh().contains("1/2"));
    }

    #[test]
    fn tsv_parser_rejects_malformed_rows_and_accepts_batches() {
        let text = "# comment\n姚明\t年龄\t38\timport\t体测表\t姚明38\n张三\t师傅\t李四\tmanual\t人工确认\n";
        let rows = parse_provenance_tsv(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, fact("姚明", "年龄", "38"));
        assert_eq!(rows[0].1.kind, SourceKind::Import);
        assert!(rows[0].1.render().contains("体测表"));
        assert!(parse_provenance_tsv("a\tb\tc\ttypo\td").is_err());
        assert!(parse_provenance_tsv("a\tb\tc\tunknown\td").is_err());
        assert!(parse_provenance_tsv("a\tb\tc").is_err());
    }

    #[test]
    fn support_coverage_counts_derived_rule_sources() {
        let premise = supported(fact("姚明", "出生年", "1986"), &[source("档案", "1986")]);
        let derived = Support::derived(
            fact("姚明", "年龄", "38"),
            source("规则甲", "甲"),
            "R1",
            0.9,
            vec![premise],
        );
        let mut supports = BTreeMap::new();
        supports.insert(derived.fact.key(), derived);
        let coverage = Coverage::from_supports(&[fact("姚明", "年龄", "38")], &supports);
        assert_eq!(coverage.registered, 1);
        assert!(coverage.unregistered.is_empty());

        drop(supports);
        let coverage = Coverage::from_supports(&[fact("姚明", "年龄", "38")], &BTreeMap::new());
        assert_eq!(coverage.registered, 0);
    }

    #[test]
    fn ledger_roundtrip_and_tamper_are_visible() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("provenance.json");
        let path_text = path.to_str().unwrap();
        assert!(matches!(ProvenanceLedger::load(path_text), ProvenanceLoad::Missing));

        let mut ledger = ProvenanceLedger::new();
        ledger.record(fact("张三", "年龄", "38"), source("体测表", "38"));
        ledger.save(path_text).unwrap();
        let ProvenanceLoad::Loaded(restored) = ProvenanceLedger::load(path_text) else {
            panic!("应能恢复来源台账");
        };
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.digest(), ledger.digest());

        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path_text).unwrap()).unwrap();
        value["sections"][PROVENANCE_SECTION][0]["fact"]["value"] = serde_json::json!("39");
        std::fs::write(path_text, value.to_string()).unwrap();
        assert!(matches!(
            ProvenanceLedger::load(path_text),
            ProvenanceLoad::Damaged(_)
        ));
    }

    #[test]
    fn witnessed_functional_conflict_has_candidates_and_exact_retraction() {
        let age = fact("姚明", "年龄", "38");
        let age_other = fact("姚明", "年龄", "39");
        let first = source("2002-体测表", "38");
        let second = source("2010-访谈", "39");
        let mut ledger = ProvenanceLedger::new();
        ledger.record(age.clone(), first.clone());
        ledger.record(age_other.clone(), second.clone());
        let supports = build_supports(
            &[age.clone(), age_other.clone()],
            &ledger,
            &empty_rules(),
            &empty_derived(),
        );
        let facts = vec![age, age_other];
        let certificates = detect_witnessed(&facts, &supports, &schema());
        assert_eq!(certificates.len(), 1);
        let certificate = &certificates[0];
        assert!(!certificate.latent);
        assert!(matches!(
            certificate.attribution,
            Attribution::Candidates { .. }
        ));
        assert_eq!(certificate.left_only.len(), 1);
        assert_eq!(certificate.right_only.len(), 1);
        assert!(certificate.common.is_empty());
        let report = ConflictReport::new(certificates, ledger.coverage(&facts));
        assert!(report.retraction.exact);
        assert_eq!(report.retraction.facts.len(), 1, "删一个值即可解决冲突");
        assert_eq!(report.retraction.resolved, 1);
        assert_eq!(report.retraction.unresolved, 0);
        let rendered = report.render_text();
        assert!(rendered.contains("已见证"), "{}", rendered);
        assert!(rendered.contains("候选肇事前提"), "{}", rendered);
        assert!(rendered.contains("2002-体测表"), "{}", rendered);
    }

    #[test]
    fn missing_sources_enter_insufficient_attribution_not_silent_choice() {
        let age = fact("姚明", "年龄", "38");
        let age_other = fact("姚明", "年龄", "39");
        let supports = build_supports(
            &[age.clone(), age_other.clone()],
            &ProvenanceLedger::new(),
            &empty_rules(),
            &empty_derived(),
        );
        let facts = vec![age, age_other];
        let report = ConflictReport::new(
            detect_witnessed(&facts, &supports, &schema()),
            Coverage {
                total: 2,
                registered: 0,
                ratio: 0.0,
                unregistered: facts,
            },
        );
        assert_eq!(report.insufficient_attribution(), 1);
        let rendered = report.render_text();
        assert!(rendered.contains("无法归因"), "{}", rendered);
        assert!(rendered.contains("来源未登记"), "{}", rendered);
        assert_eq!(
            report.retraction.facts.len(),
            1,
            "即使来源缺失，事实级撤回集仍可计算"
        );
        assert!(report.retraction.exact);
    }

    #[test]
    fn shared_premises_are_same_origin_not_independent_sources() {
        let premise = supported(
            fact("姚明", "出生年", "1986"),
            &[source("档案", "1986")],
        );
        let left = Support::derived(
            fact("姚明", "年龄", "38"),
            source("规则甲", "甲"),
            "R1",
            0.9,
            vec![premise.clone()],
        );
        let right = Support::derived(
            fact("姚明", "年龄", "39"),
            source("规则乙", "乙"),
            "R2",
            0.9,
            vec![premise],
        );
        let certificate = ConflictCertificate::new(
            ConflictPoint::Functional {
                subject: "姚明".to_string(),
                predicate: "年龄".to_string(),
                left_value: "38".to_string(),
                right_value: "39".to_string(),
            },
            left,
            right,
            false,
        );
        assert!(matches!(
            certificate.attribution,
            Attribution::SameOrigin { .. }
        ));
        let report = ConflictReport::new(
            vec![certificate],
            Coverage {
                total: 1,
                registered: 1,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        assert_eq!(
            report.retraction.facts,
            vec![fact("姚明", "出生年", "1986")],
            "同源矛盾只能撤回共同前提"
        );
    }

    #[test]
    fn minimal_hitting_set_covers_overlapping_certificates() {
        let a = supported(fact("A", "p", "1"), &[source("s", "a")]);
        let b = supported(fact("B", "p", "1"), &[source("s", "b")]);
        let c = supported(fact("C", "p", "1"), &[source("s", "c")]);
        let mut certificates = Vec::new();
        for (left, right) in [(&a, &b), (&b, &c)] {
            let point = ConflictPoint::Functional {
                subject: left.fact.subject.clone(),
                predicate: "p".to_string(),
                left_value: left.fact.value.clone(),
                right_value: right.fact.value.clone(),
            };
            certificates.push(ConflictCertificate::new(
                point,
                left.clone(),
                right.clone(),
                false,
            ));
        }
        let plan = plan_retraction(&certificates);
        assert!(plan.exact);
        assert_eq!(plan.facts.len(), 1, "两个共用 B 的证书只需撤回 B");
        assert_eq!(plan.facts, vec![fact("B", "p", "1")]);
        assert_eq!(plan.resolved, 2);
    }

    #[test]
    fn predicate_aliases_join_the_same_functional_slot() {
        let mut schema = schema();
        schema
            .predicate_aliases
            .insert("岁数".to_string(), "年龄".to_string());
        let first = fact("姚明", "年龄", "38");
        let second = fact("姚明", "岁数", "39");
        let supports = BTreeMap::new();
        let certificates = detect_witnessed(&[first.clone(), second.clone()], &supports, &schema);
        assert_eq!(certificates.len(), 1, "别名必须并入同一功能槽");
        assert_eq!(
            certificates[0].point,
            ConflictPoint::Functional {
                subject: "姚明".to_string(),
                predicate: "年龄".to_string(),
                left_value: "38".to_string(),
                right_value: "39".to_string(),
            }
        );
        assert_eq!(certificates[0].left.fact, first);
        assert_eq!(certificates[0].right.fact, second);
    }

    #[test]
    fn non_functional_predicates_and_same_values_are_not_conflicts() {
        let facts = vec![
            fact("张三", "喜欢", "李四"),
            fact("张三", "喜欢", "王五"),
            fact("张三", "年龄", "38"),
            fact("张三", "年龄", "38"),
        ];
        let supports = BTreeMap::new();
        let certificates = detect_witnessed(&facts, &supports, &schema());
        assert!(
            certificates.is_empty(),
            "非功能谓词多值、相同取值重复都不是冲突：{:?}",
            certificates
        );
    }

    #[test]
    fn witnessed_disjoint_types_use_type_aliases() {
        let mut schema = schema();
        schema
            .type_aliases
            .insert("Person".to_string(), "人物".to_string());
        let facts = vec![
            FactRef::type_assertion("姚明", "Person"),
            FactRef::type_assertion("姚明", "地点"),
        ];
        let certificates = detect_witnessed(&facts, &BTreeMap::new(), &schema);
        assert_eq!(certificates.len(), 1);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::DisjointTypes { .. }
        ));
        assert!(matches!(
            certificates[0].attribution,
            Attribution::Insufficient { .. }
        ));
    }

    fn clause(head: Atom, body: Vec<Atom>, source: &str) -> Clause {
        Clause::new(head, body, 0.9, source)
    }

    #[test]
    fn latent_functional_conflict_only_for_different_head_constants() {
        let left = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("38")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1986")],
            )],
            "规则甲",
        );
        let right = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("39")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1990")],
            )],
            "规则乙",
        );
        let schema = schema();
        let certificates = detect_latent(&[left.clone(), right.clone()], &schema);
        assert_eq!(
            certificates.len(),
            0,
            "出生年互相矛盾，两条规则不可能同时触发：{:?}",
            certificates
        );

        let compatible = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("39")]),
            vec![Atom::new(
                "出生年",
                vec![Term::var("X"), Term::constant("1986")],
            )],
            "规则乙",
        );
        let certificates = detect_latent(&[left, compatible], &schema);
        assert_eq!(certificates.len(), 1);
        assert!(certificates[0].latent);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::Functional { .. }
        ));

        let same = clause(
            Atom::new("年龄", vec![Term::var("X"), Term::constant("38")]),
            Vec::new(),
            "重复规则",
        );
        assert!(
            detect_latent(&[same.clone(), same], &schema).is_empty(),
            "头部常量相同不是冲突"
        );
    }

    #[test]
    fn latent_type_conflict_requires_declared_disjoint_types() {
        let left = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("人物")]),
            Vec::new(),
            "类型规则甲",
        );
        let right = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("地点")]),
            Vec::new(),
            "类型规则乙",
        );
        let certificates = detect_latent(&[left.clone(), right.clone()], &schema());
        assert_eq!(certificates.len(), 1);
        assert!(matches!(
            certificates[0].point,
            ConflictPoint::DisjointTypes { .. }
        ));

        let compatible = clause(
            Atom::new("type", vec![Term::var("X"), Term::constant("人物")]),
            Vec::new(),
            "类型规则丙",
        );
        assert!(detect_latent(&[left, compatible], &schema()).is_empty());
    }

    #[test]
    fn report_is_deterministic_and_serializable() {
        let a = supported(fact("A", "年龄", "1"), &[source("s", "a")]);
        let b = supported(fact("A", "年龄", "2"), &[source("s", "b")]);
        let cert = ConflictCertificate::new(
            ConflictPoint::Functional {
                subject: "A".to_string(),
                predicate: "年龄".to_string(),
                left_value: "1".to_string(),
                right_value: "2".to_string(),
            },
            a,
            b,
            false,
        );
        let first = ConflictReport::new(
            vec![cert.clone()],
            Coverage {
                total: 2,
                registered: 2,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        let second = ConflictReport::new(
            vec![cert],
            Coverage {
                total: 2,
                registered: 2,
                ratio: 1.0,
                unregistered: Vec::new(),
            },
        );
        assert_eq!(first, second);
        assert_eq!(first.to_json(), second.to_json());
        assert!(!first.certificates[0].id.is_empty());
    }

    #[test]
    fn import_journal_seeding_uses_real_batch_identity() {
        use crate::import_journal::ImportJournal;
        use crate::predicate_registry::{ImportAction, ImportReceipt};

        let receipt = ImportReceipt {
            added: 2,
            actions: vec![
                ImportAction::Literal {
                    subject: "http://example.org/kg/entity/姚明".to_string(),
                    predicate: "http://example.org/kg/年龄".to_string(),
                    value: "38".to_string(),
                },
                ImportAction::Named {
                    subject: "http://example.org/kg/entity/姚明".to_string(),
                    predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    object: "http://example.org/kg/Person".to_string(),
                },
            ],
            ..Default::default()
        };
        let journal = ImportJournal::from_entries(vec![receipt]);
        let mut ledger = ProvenanceLedger::new();
        assert_eq!(ledger.seed_from_import_journal(&journal), 2);

        let age = fact("姚明", "年龄", "38");
        let sources = ledger.sources(&age);
        assert_eq!(sources.len(), 1);
        let source = sources.iter().next().unwrap();
        assert_eq!(source.kind, SourceKind::Import);
        assert!(source.locator.starts_with("导入日志#1"));
        assert!(source.is_registered());
        assert!(ledger.sources(&FactRef::type_assertion("姚明", "Person")).len() == 1);
    }

    #[test]
    fn fusion_seeding_keeps_namespace_and_evidence_pointer() {
        use crate::knowledge_fusion::{
            FusionLedger, NamespaceKind, NamespaceRecord, ObjectKind, ScopedAssertion, TrustLevel,
        };

        let namespace = NamespaceRecord::new("batch-a", "甲批", NamespaceKind::Import, TrustLevel::Imported);
        let assertion = ScopedAssertion::new(
            "batch-a",
            "姚明",
            "年龄",
            "40",
            ObjectKind::Literal,
            "/data/ns.ttl",
        );
        let mut fusion = FusionLedger::new();
        fusion.upsert_namespace(namespace);
        fusion.record_assertion(assertion);
        let mut ledger = ProvenanceLedger::new();
        assert_eq!(ledger.seed_from_fusion_ledger(&fusion), 1);
        let sources = ledger.sources(&fact("姚明", "年龄", "40"));
        assert_eq!(sources.len(), 1);
        let source = sources.iter().next().unwrap();
        assert!(source.locator.contains("甲批"));
        assert!(source.locator.contains("/data/ns.ttl"));
        assert!(source.render().contains("导入:甲批"));
    }

    #[test]
    fn three_values_make_three_certificates_but_one_conflict_point() {
        let mut facts = Vec::new();
        let mut ledger = ProvenanceLedger::new();
        for value in ["38", "39", "40"] {
            let fact = fact("姚明", "年龄", value);
            ledger.record(fact.clone(), source("批", value));
            facts.push(fact);
        }
        let supports = build_supports(
            &facts,
            &ledger,
            &empty_rules(),
            &empty_derived(),
        );
        let report = ConflictReport::new(
            detect_witnessed(&facts, &supports, &schema()),
            ledger.coverage(&facts),
        );
        assert_eq!(report.certificates.len(), 3, "两两证书");
        assert_eq!(report.conflict_points(), 1, "但只有一个功能槽冲突点");
        assert_eq!(report.retraction.facts.len(), 2, "修复代价是删 2 条，不是 3 条");
        assert!(report.render_text().contains("覆盖 1 个冲突点"));
    }

    #[test]
    fn retraction_falls_back_to_greedy_beyond_exact_limit() {
        let mut certificates = Vec::new();
        for index in 0..(MAX_EXACT_CANDIDATES + 1) {
            let left = supported(
                fact(&format!("E{}", index), "年龄", "1"),
                &[source("s", &format!("l{}", index))],
            );
            let right = supported(
                fact(&format!("E{}", index), "年龄", "2"),
                &[source("s", &format!("r{}", index))],
            );
            certificates.push(ConflictCertificate::new(
                ConflictPoint::Functional {
                    subject: format!("E{}", index),
                    predicate: "年龄".to_string(),
                    left_value: "1".to_string(),
                    right_value: "2".to_string(),
                },
                left,
                right,
                false,
            ));
        }
        let plan = plan_retraction(&certificates);
        assert!(!plan.exact, "超过精确上限必须标记非精确");
        assert_eq!(plan.resolved, certificates.len());
    }
}

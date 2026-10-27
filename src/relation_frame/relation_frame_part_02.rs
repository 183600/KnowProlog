impl FrameGate {
    pub fn new(
        facts: Vec<FrameEdge>,
        negative_facts: Vec<FrameEdge>,
        entity_types: BTreeMap<String, BTreeSet<String>>,
        registry: PredicateRegistry,
        types: TypeSystem,
        laws: LawBook,
    ) -> Self {
        let normalize = |edge: FrameEdge| normalize_edge(&registry, &laws, &edge);
        let facts: BTreeSet<FrameEdge> = facts.into_iter().map(normalize).collect();
        let negative_facts: BTreeSet<FrameEdge> =
            negative_facts.into_iter().map(normalize).collect();
        let rules = ClosureRules::build(&facts, &registry, &laws);
        let index = ClosureIndex::build(&facts);
        Self {
            facts,
            negative_facts,
            entity_types,
            registry,
            types,
            rules,
            index,
            limits: ClosureLimits::default(),
            fact_validity: BTreeMap::new(),
        }
    }

    /// 注入事实有效期（sidecar）。同一个三元组以最后一次注入为准。
    pub fn with_fact_validity(
        mut self,
        entries: impl IntoIterator<Item = (FrameEdge, Validity)>,
    ) -> Self {
        self.fact_validity = entries.into_iter().collect();
        self
    }

    pub fn from_facts(
        facts: &[PrologFact],
        registry: PredicateRegistry,
        types: TypeSystem,
        laws: LawBook,
    ) -> Self {
        let mut edges = Vec::new();
        let mut entity_types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut validity_by_entity: BTreeMap<String, Validity> = BTreeMap::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    if !entity_type.is_empty() {
                        entity_types
                            .entry(name.clone())
                            .or_default()
                            .insert(entity_type.clone());
                    }
                    if let Some(validity) = temporal::validity_from_properties(properties) {
                        validity_by_entity.insert(name.clone(), validity);
                        if let Some(alias) = properties.get("name") {
                            validity_by_entity.insert(alias.clone(), validity);
                        }
                    }
                    // 实体属性（年龄、身高、首都等）在图中不是三元组关系，
                    // 但对帧而言同样是二元谓词；只收字面量值，跳过关系 IRI。
                    for (predicate, value) in properties {
                        if predicate.trim().is_empty()
                            || value.trim().is_empty()
                            || value.starts_with("http://")
                            || value.starts_with("https://")
                            || value.starts_with('<')
                        {
                            continue;
                        }
                        edges.push(FrameEdge::new(name, predicate, value));
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    edges.push(FrameEdge::new(from, rel_type, to));
                }
            }
        }
        let validity: Vec<(FrameEdge, Validity)> = edges
            .iter()
            .filter_map(|edge| {
                validity_by_entity
                    .get(&edge.subject)
                    .copied()
                    .filter(Validity::is_known)
                    .map(|validity| (edge.clone(), validity))
            })
            .collect();
        Self::new(edges, Vec::new(), entity_types, registry, types, laws)
            .with_fact_validity(validity)
    }

    pub fn with_limits(mut self, limits: ClosureLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn evaluate(&self, frame: &RelationFrame) -> FrameGateReport {
        let (normalized, inverted) =
            normalize_edge_detailed(&self.registry, &self.entity_types, frame);
        let negative = frame.polarity == Polarity::Negative;
        let frame_validity = frame
            .temporal
            .as_ref()
            .map(TemporalScope::to_validity)
            .unwrap_or(Validity::Unknown);
        let direct_positive = self.facts.contains(&normalized)
            && self.validities_compatible(frame_validity, &normalized);
        let direct_negative = self.negative_facts.contains(&normalized);

        let without = self.local_closure(&normalized, None);
        let with = if negative {
            None
        } else {
            Some(self.local_closure(&normalized, Some(&normalized)))
        };
        let inferred_positive = !direct_positive && without.contains_key(&normalized);
        let candidate_novel =
            !direct_positive && !direct_negative && !without.contains_key(&normalized);
        let closure_gain = with
            .as_ref()
            .map(|with| {
                with.len()
                    .saturating_sub(without.len())
                    .saturating_sub(usize::from(candidate_novel))
            })
            .unwrap_or(0);

        let mut findings = Vec::new();
        if inverted {
            findings.push(GateFinding::new(
                GateFindingKind::InverseNormalized,
                GateSeverity::Info,
                format!(
                    "谓词「{}」按互逆方向归一为 {}",
                    frame.predicate, normalized.predicate
                ),
            ));
        }

        let known_predicate = self.registry.lookup(&frame.predicate).is_some();
        if !known_predicate {
            findings.push(GateFinding::new(
                GateFindingKind::UnknownPredicate,
                GateSeverity::Warning,
                format!(
                    "谓词「{}」不在谓词画像中，闸门只做结构检查",
                    frame.predicate
                ),
            ));
        }

        let mut conflict = false;

        if direct_positive {
            if negative {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::NegativeConflict,
                    GateSeverity::Violation,
                    format!("否定帧与图谱事实 {} 冲突", normalized.render_text()),
                ));
            } else {
                findings.push(GateFinding::new(
                    GateFindingKind::DirectFact,
                    GateSeverity::Info,
                    format!("图谱直接记录 {}", normalized.render_text()),
                ));
            }
        }
        if direct_negative {
            if negative {
                findings.push(GateFinding::new(
                    GateFindingKind::ExplicitNegative,
                    GateSeverity::Info,
                    format!("存在显式否定记录 {}", normalized.render_text()),
                ));
            } else {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::NegativeConflict,
                    GateSeverity::Violation,
                    format!("肯定帧与显式否定记录 {} 冲突", normalized.render_text()),
                ));
            }
        }

        if inferred_positive {
            if negative {
                conflict = true;
                findings.push(
                    GateFinding::new(
                        GateFindingKind::NegativeConflict,
                        GateSeverity::Violation,
                        format!("否定帧与演绎闭包中的 {} 冲突", normalized.render_text()),
                    )
                    .with_proof(without.get(&normalized).cloned().unwrap_or_default()),
                );
            } else {
                findings.push(
                    GateFinding::new(
                        GateFindingKind::InferredFact,
                        GateSeverity::Info,
                        format!(
                            "局部闭包推出 {}（{} 步内）",
                            normalized.render_text(),
                            self.limits.max_depth
                        ),
                    )
                    .with_proof(without.get(&normalized).cloned().unwrap_or_default()),
                );
            }
        }

        if !negative {
            let values = self.functional_conflict_values(&normalized, frame_validity);
            if !values.is_empty() {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::FunctionalConflict,
                    GateSeverity::Violation,
                    format!(
                        "单值谓词「{}」对主语「{}」已有其他取值：{}",
                        normalized.predicate,
                        normalized.subject,
                        values.join("、")
                    ),
                ));
            }

            if let Some(violation) = self.schema_violation(&normalized) {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::SchemaViolation,
                    GateSeverity::Violation,
                    format!(
                        "{}「{}」不在声明的{}内：期望 {}，实际 {}",
                        violation.slot,
                        if violation.slot == "定义域" {
                            &violation.subject
                        } else {
                            &violation.object
                        },
                        if violation.slot == "定义域" {
                            "定义域"
                        } else {
                            "值域"
                        },
                        violation.expected.join(" / "),
                        violation.actual.join(" / ")
                    ),
                ));
            }
        }

        if closure_gain > 0 {
            findings.push(GateFinding::new(
                GateFindingKind::ClosureGain,
                GateSeverity::Info,
                format!("加入该帧后局部闭包新增 {} 条可复用的边", closure_gain),
            ));
        }

        let verdict = if conflict {
            GateVerdict::Conflict
        } else if direct_positive || direct_negative {
            GateVerdict::DirectSupport
        } else if inferred_positive {
            GateVerdict::InferredSupport
        } else {
            GateVerdict::Unsupported
        };
        let support_path = without.get(&normalized).cloned().unwrap_or_default();
        FrameGateReport {
            frame_id: frame.id.clone(),
            normalized,
            inverted,
            verdict,
            findings,
            closure_size: with
                .as_ref()
                .map(|closure| closure.len())
                .unwrap_or_else(|| without.len()),
            closure_gain,
            support_path,
        }
    }

    fn validities_compatible(&self, frame_validity: Validity, edge: &FrameEdge) -> bool {
        let fact_validity = self
            .fact_validity
            .get(edge)
            .copied()
            .unwrap_or(Validity::Unknown);
        if frame_validity.is_unknown() || fact_validity.is_unknown() {
            return true;
        }
        frame_validity.intersection(fact_validity).is_satisfiable()
    }

    fn functional_conflict_values(
        &self,
        edge: &FrameEdge,
        frame_validity: Validity,
    ) -> Vec<String> {
        let functional = self.rules.functional.contains(&edge.predicate)
            || self
                .registry
                .profile(&edge.predicate)
                .map(|profile| profile.functional)
                .unwrap_or(false);
        if !functional {
            return Vec::new();
        }
        self.facts
            .iter()
            .filter(|fact| {
                fact.subject == edge.subject
                    && fact.predicate == edge.predicate
                    && fact.object != edge.object
                    && self.validities_compatible(frame_validity, fact)
            })
            .map(|fact| fact.object.clone())
            .collect()
    }

    fn schema_violation(
        &self,
        edge: &FrameEdge,
    ) -> Option<crate::predicate_registry::SchemaViolation> {
        let subject_types = self
            .entity_types
            .get(&edge.subject)
            .cloned()
            .unwrap_or_default();
        let object_types = self
            .entity_types
            .get(&edge.object)
            .cloned()
            .unwrap_or_default();
        if self.types.closure_gate() {
            self.registry.check_schema_with_types(
                &edge.subject,
                &edge.predicate,
                &edge.object,
                &subject_types,
                &object_types,
                &self.types,
            )
        } else {
            self.registry.check_schema(
                &edge.subject,
                &edge.predicate,
                &edge.object,
                &subject_types,
                &object_types,
            )
        }
    }

    fn local_closure(
        &self,
        edge: &FrameEdge,
        extra: Option<&FrameEdge>,
    ) -> BTreeMap<FrameEdge, Vec<FrameEdge>> {
        let mut proofs: BTreeMap<FrameEdge, Vec<FrameEdge>> = BTreeMap::new();
        let mut queue: VecDeque<(FrameEdge, usize)> = VecDeque::new();
        let mut seeds = self.index.incident(&edge.subject);
        seeds.extend(self.index.incident(&edge.object));
        if let Some(extra) = extra {
            seeds.insert(extra.clone());
        }
        for seed in seeds {
            if proofs.len() >= self.limits.max_edges {
                break;
            }
            if proofs.insert(seed.clone(), Vec::new()).is_none() {
                queue.push_back((seed, 0));
            }
        }

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= self.limits.max_depth {
                continue;
            }
            for (derived, premises) in self.derive_from(&current, extra) {
                if proofs.len() >= self.limits.max_edges {
                    break;
                }
                // 保留 BFS 中首次（即最短）的证明；后续等价推导不覆盖，
                // 否则互逆边会把传递证明替换成循环引用。
                if proofs.contains_key(&derived) {
                    continue;
                }
                proofs.insert(derived.clone(), premises);
                queue.push_back((derived, depth + 1));
            }
        }
        proofs
    }

    fn derive_from(
        &self,
        edge: &FrameEdge,
        extra: Option<&FrameEdge>,
    ) -> Vec<(FrameEdge, Vec<FrameEdge>)> {
        let mut derived = Vec::new();
        if self.rules.transitive.contains(&edge.predicate) {
            for object in self.index.objects(&edge.predicate, &edge.object, extra) {
                if object == edge.subject {
                    continue;
                }
                let mut premises = vec![edge.clone()];
                let other = FrameEdge::new(&edge.object, &edge.predicate, &object);
                if other != *edge {
                    premises.push(other);
                }
                derived.push((
                    FrameEdge::new(&edge.subject, &edge.predicate, &object),
                    premises,
                ));
            }
        }
        if self.rules.symmetric.contains(&edge.predicate) {
            derived.push((
                FrameEdge::new(&edge.object, &edge.predicate, &edge.subject),
                vec![edge.clone()],
            ));
        }
        if let Some(inverse) = self.rules.inverse.get(&edge.predicate) {
            derived.push((
                FrameEdge::new(&edge.object, inverse, &edge.subject),
                vec![edge.clone()],
            ));
        }
        derived
    }
}
fn normalize_edge(registry: &PredicateRegistry, laws: &LawBook, edge: &FrameEdge) -> FrameEdge {
    let resolution = registry.resolve(&edge.predicate);
    let (canonical, inverted) = if resolution.known {
        (resolution.canonical, resolution.inverted)
    } else if let Some((canonical, inverted)) = laws.alias_map().get(edge.predicate.trim()) {
        (canonical.clone(), *inverted)
    } else {
        (edge.predicate.trim().to_string(), false)
    };
    if inverted {
        FrameEdge::new(&edge.object, &canonical, &edge.subject)
    } else {
        FrameEdge::new(&edge.subject, &canonical, &edge.object)
    }
}
fn normalize_edge_detailed(
    registry: &PredicateRegistry,
    _entity_types: &BTreeMap<String, BTreeSet<String>>,
    frame: &RelationFrame,
) -> (FrameEdge, bool) {
    let edge = FrameEdge::new(&frame.subject, &frame.predicate, &frame.object);
    let resolution = registry.resolve(&frame.predicate);
    if resolution.known {
        let normalized = if resolution.inverted {
            FrameEdge::new(&frame.object, &resolution.canonical, &frame.subject)
        } else {
            FrameEdge::new(&frame.subject, &resolution.canonical, &frame.object)
        };
        (normalized, resolution.inverted)
    } else {
        (edge, false)
    }
}
/// 宾语像属性值（数字、数量、日期等）时，用「X 的 P 是 V」句式；
/// 否则用更中性的「X 与 Y 之间存在 P 关系」，避免把动词关系说成属性。
fn value_like(object: &str) -> bool {
    let object = object.trim();
    if object.is_empty() {
        return false;
    }
    if object.chars().any(|ch| ch.is_ascii_digit()) {
        return true;
    }
    const VALUE_SUFFIXES: &[&str] = &[
        "岁",
        "年",
        "月",
        "日",
        "号",
        "元",
        "万元",
        "亿元",
        "米",
        "千米",
        "公里",
        "平方米",
        "平方公里",
        "千克",
        "公斤",
        "克",
        "吨",
        "人",
        "名",
        "个",
        "次",
        "天",
        "小时",
        "分钟",
        "%",
        "％",
    ];
    VALUE_SUFFIXES.iter().any(|suffix| object.ends_with(suffix))
}
/// 把闸门判定与证据整理成中文叙述。
pub fn narrate_frame(frame: &RelationFrame, report: &FrameGateReport) -> String {
    let mut output = String::new();
    let statement = if value_like(&frame.object) {
        if frame.polarity == Polarity::Negative {
            format!(
                "否定陈述：{}的{}不是{}。",
                frame.subject, frame.predicate, frame.object
            )
        } else {
            format!(
                "陈述：{}的{}是{}。",
                frame.subject, frame.predicate, frame.object
            )
        }
    } else if frame.polarity == Polarity::Negative {
        format!(
            "否定陈述：{}与{}之间不存在「{}」关系。",
            frame.subject, frame.object, frame.predicate
        )
    } else {
        format!(
            "陈述：{}与{}之间存在「{}」关系。",
            frame.subject, frame.object, frame.predicate
        )
    };
    output.push_str(&statement);
    if let Some(temporal) = &frame.temporal {
        output.push_str(&format!("时间限定：{}。", temporal.render_text()));
    }
    if frame.modality != Modality::Asserted {
        output.push_str(&format!("语气：{}。", frame.modality.label()));
    }
    output.push_str(&format!("闸门判定：{}。", report.verdict.label()));
    if !report.support_path.is_empty() {
        output.push_str(&format!(
            "证明路径：{}。",
            report
                .support_path
                .iter()
                .map(FrameEdge::render_text)
                .collect::<Vec<_>>()
                .join("；")
        ));
    }
    for finding in &report.findings {
        if finding.severity == GateSeverity::Info {
            continue;
        }
        output.push_str(&format!("{}：{}。", finding.kind.label(), finding.detail));
    }
    output
}
/// 一帧诊断样本：帧 + 闸门报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameDiagnosticSample {
    pub frame: RelationFrame,
    pub report: FrameGateReport,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameDiagnostics {
    pub dialect: String,
    pub frames: usize,
    pub direct_support: usize,
    pub inferred_support: usize,
    pub conflicts: usize,
    pub unsupported: usize,
    pub temporal_frames: usize,
    pub negative_frames: usize,
    pub modal_frames: usize,
    pub conjunction_frames: usize,
    pub closure_gain_total: usize,
    pub finding_counts: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<FrameDiagnosticSample>,
}
impl FrameDiagnostics {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "帧诊断：{} 条帧；直接支持 {}，推理支持 {}，冲突 {}，无证据 {}。\n",
            self.frames,
            self.direct_support,
            self.inferred_support,
            self.conflicts,
            self.unsupported
        );
        output.push_str(&format!(
            "限定覆盖：时间 {}，否定 {}，模态 {}，合取 {}；闭包总增益 {}。\n",
            self.temporal_frames,
            self.negative_frames,
            self.modal_frames,
            self.conjunction_frames,
            self.closure_gain_total
        ));
        if !self.finding_counts.is_empty() {
            output.push_str("发现计数：");
            output.push_str(
                &self
                    .finding_counts
                    .iter()
                    .map(|(kind, count)| format!("{}={}", kind, count))
                    .collect::<Vec<_>>()
                    .join("，"),
            );
            output.push('\n');
        }
        output
    }
}
/// 对一批帧跑闸门并汇总诊断。
pub fn diagnose(
    frames: &[RelationFrame],
    gate: &FrameGate,
    sample_limit: usize,
) -> FrameDiagnostics {
    let mut diagnostics = FrameDiagnostics {
        dialect: FRAME_DIALECT.to_string(),
        frames: frames.len(),
        ..Default::default()
    };
    for frame in frames {
        let report = gate.evaluate(frame);
        match report.verdict {
            GateVerdict::DirectSupport => diagnostics.direct_support += 1,
            GateVerdict::InferredSupport => diagnostics.inferred_support += 1,
            GateVerdict::Conflict => diagnostics.conflicts += 1,
            GateVerdict::Unsupported => diagnostics.unsupported += 1,
        }
        diagnostics.closure_gain_total += report.closure_gain;
        if frame.temporal.is_some() {
            diagnostics.temporal_frames += 1;
        }
        if frame.polarity == Polarity::Negative {
            diagnostics.negative_frames += 1;
        }
        if frame.modality != Modality::Asserted {
            diagnostics.modal_frames += 1;
        }
        if frame.qualifiers.iter().any(|qualifier| qualifier == "合取") {
            diagnostics.conjunction_frames += 1;
        }
        for finding in &report.findings {
            *diagnostics
                .finding_counts
                .entry(finding.kind.tag().to_string())
                .or_insert(0) += 1;
        }
        if diagnostics.samples.len() < sample_limit {
            diagnostics.samples.push(FrameDiagnosticSample {
                frame: frame.clone(),
                report,
            });
        }
    }
    diagnostics
}
/// 规范 JSON（用于诊断输出的稳定摘要）。
pub fn diagnostics_digest(diagnostics: &FrameDiagnostics) -> String {
    let value = serde_json::to_value(diagnostics).unwrap_or(serde_json::Value::Null);
    fnv1a_hex(canonical_json(&value).as_bytes())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use std::collections::HashMap;

    fn profile(name: &str) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        let mut master = profile("师傅");
        master.transitive = true;
        master.inverse = Some("徒弟".to_string());
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        registry.declare(&master);
        let mut apprentice = profile("徒弟");
        apprentice.inverse = Some("师傅".to_string());
        registry.declare(&apprentice);
        let mut age = profile("年龄");
        age.functional = true;
        registry.declare(&age);
        registry
    }

    fn gate(edges: &[(&str, &str, &str)]) -> FrameGate {
        let registry = registry();
        let laws = LawBook::from_registry(&registry);
        let facts = edges
            .iter()
            .map(|(s, p, o)| FrameEdge::new(s, p, o))
            .collect();
        FrameGate::new(
            facts,
            Vec::new(),
            BTreeMap::new(),
            registry,
            TypeSystem::new(),
            laws,
        )
    }

    #[test]
    fn parses_explicit_and_relative_temporal_scopes() {
        let year = parse_temporal("公司成立于1999年").expect("year");
        assert_eq!(year.normalized.as_deref(), Some("1999"));
        assert_eq!(year.relation, TemporalRelation::Point);

        let range = parse_temporal("从1991年到1999年").expect("range");
        assert_eq!(range.relation, TemporalRelation::Range);
        assert_eq!(range.start_year, Some(1991));
        assert_eq!(range.end_year, Some(1999));

        let relative = parse_temporal("他去年出生").expect("relative");
        assert_eq!(relative.granularity, TemporalGranularity::Relative);
        assert!(relative.to_time_span().is_none());
        assert_eq!(relative.to_validity(), Validity::Unknown);
        assert!(negation_cue("不丹的首都是廷布").is_none());
        assert!(negation_cue("张三不使用标注数据").is_some());
    }

    #[test]
    fn from_facts_maps_validity_across_name_alias() {
        let mut registry = PredicateRegistry::new();
        let mut capital = profile("首都");
        capital.functional = true;
        registry.declare(&capital);
        let laws = LawBook::from_registry(&registry);
        let mut kyoto: HashMap<String, String> = HashMap::new();
        kyoto.insert("name".to_string(), "京都".to_string());
        kyoto.insert("valid_to".to_string(), "1867".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "kyoto".to_string(),
                entity_type: "城市".to_string(),
                properties: kyoto,
            },
            PrologFact::Entity {
                name: "japan".to_string(),
                entity_type: "国家".to_string(),
                properties: HashMap::from([("name".to_string(), "日本".to_string())]),
            },
            PrologFact::Relation {
                from: "京都".to_string(),
                rel_type: "首都".to_string(),
                to: "日本".to_string(),
            },
        ];
        let gate = FrameGate::from_facts(&facts, registry, TypeSystem::new(), laws);
        let frame = RelationFrame::new("日本", "首都", "京都")
            .with_temporal(parse_temporal("1868年").expect("year"));
        assert_eq!(gate.evaluate(&frame).verdict, GateVerdict::Unsupported);
    }

    #[test]
    fn functional_conflicts_are_time_aware() {
        let mut registry = PredicateRegistry::new();
        let mut capital = profile("首都");
        capital.functional = true;
        registry.declare(&capital);
        let laws = LawBook::from_registry(&registry);
        let kyoto = FrameEdge::new("日本", "首都", "京都");
        let tokyo = FrameEdge::new("日本", "首都", "东京");
        let gate = FrameGate::new(
            vec![kyoto.clone(), tokyo.clone()],
            Vec::new(),
            BTreeMap::new(),
            registry,
            TypeSystem::new(),
            laws,
        )
        .with_fact_validity([
            (kyoto, Validity::from_years(-794, 1867)),
            (tokyo, Validity::from_years(1868, 9999)),
        ]);

        let frame = RelationFrame::new("日本", "首都", "东京")
            .with_temporal(parse_temporal("1868年").expect("year"));
        let report = gate.evaluate(&frame);
        assert_eq!(report.verdict, GateVerdict::DirectSupport);
        assert!(
            !report
                .findings
                .iter()
                .any(|finding| finding.kind == GateFindingKind::FunctionalConflict),
            "历史更替不应误报单值冲突：{:?}",
            report.findings
        );

        let osaka = RelationFrame::new("日本", "首都", "大阪")
            .with_temporal(parse_temporal("1800年").expect("year"));
        let report = gate.evaluate(&osaka);
        assert_eq!(report.verdict, GateVerdict::Conflict);
        let conflict = report
            .findings
            .iter()
            .find(|finding| finding.kind == GateFindingKind::FunctionalConflict)
            .expect("1800 年与京都有效期重叠，应报冲突");
        assert!(conflict.detail.contains("京都"));
        assert!(!conflict.detail.contains("东京"), "{}", conflict.detail);
    }

    #[test]
    fn temporal_scope_converts_to_interval_algebra() {
        let year = parse_temporal("公司成立于1999年").expect("year");
        let span = year.to_time_span().expect("year span");
        assert!(span.contains(TimePoint::day(1999, 6, 1)));
        assert_eq!(year.to_validity(), Validity::from_years(1999, 1999));

        let range = parse_temporal("从1991年到1999年").expect("range");
        let span = range.to_time_span().expect("range span");
        assert!(span.contains(TimePoint::year(1995)));
        assert!(!span.contains(TimePoint::year(2000)));
        assert_eq!(range.to_validity(), Validity::from_years(1991, 1999));
    }

    #[test]
    fn extracts_frames_with_evidence_and_modality() {
        let (frames, stats) = FrameExtractor::new().extract_with_stats("1991年，张三认识李四。");
        assert!(!frames.is_empty(), "应至少抽取一条帧：{:?}", stats);
        let frame = frames
            .iter()
            .find(|frame| frame.predicate == "认识")
            .expect("认识帧");
        assert_eq!(frame.subject, "张三");
        assert_eq!(frame.object, "李四");
        assert_eq!(
            frame.temporal.as_ref().and_then(|t| t.start_year),
            Some(1991)
        );
        assert_eq!(frame.evidence.sentence, "1991年，张三认识李四");
        assert!(frame.evidence.subject_span.is_some());
        assert_eq!(stats.temporal_frames, 1);

        let (negative, _) = FrameExtractor::new().extract_with_stats("张三不认识赵六。");
        let frame = negative
            .iter()
            .find(|frame| frame.predicate == "认识")
            .expect("否定帧");
        assert_eq!(frame.subject, "张三");
        assert_eq!(frame.polarity, Polarity::Negative);
    }

    #[test]
    fn gate_supports_direct_inferred_and_conflicting_frames() {
        let gate = gate(&[
            ("张三", "师傅", "李四"),
            ("李四", "师傅", "王五"),
            ("张三", "年龄", "18"),
        ]);

        let direct = gate.evaluate(&RelationFrame::new("张三", "师傅", "李四"));
        assert_eq!(direct.verdict, GateVerdict::DirectSupport);

        let inferred = gate.evaluate(&RelationFrame::new("张三", "师傅", "王五"));
        assert_eq!(inferred.verdict, GateVerdict::InferredSupport);
        assert_eq!(
            inferred.support_path,
            vec![
                FrameEdge::new("张三", "师傅", "李四"),
                FrameEdge::new("李四", "师傅", "王五"),
            ],
            "证明路径应保留最短传递证明，而不是互逆边造成的循环引用"
        );

        let conflict = gate.evaluate(&RelationFrame::new("张三", "年龄", "20"));
        assert_eq!(conflict.verdict, GateVerdict::Conflict);
        assert!(conflict
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::FunctionalConflict));

        let negative = gate.evaluate(
            &RelationFrame::new("张三", "师傅", "李四").with_polarity(Polarity::Negative),
        );
        assert_eq!(negative.verdict, GateVerdict::Conflict);
        assert!(negative
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::NegativeConflict));
    }

    #[test]
    fn inverse_aliases_are_normalized_before_gating() {
        let direct = registry().resolve("师傅");
        assert!(!direct.inverted, "师傅应为正向槽位：{:?}", direct);
        let inverse = registry().resolve("徒弟");
        assert!(inverse.inverted, "徒弟应为逆向别名：{:?}", inverse);
        let gate = gate(&[("李四", "师傅", "张三")]);
        let report = gate.evaluate(&RelationFrame::new("张三", "徒弟", "李四"));
        assert!(report.inverted);
        assert_eq!(report.normalized.predicate, "师傅");
        assert_eq!(report.verdict, GateVerdict::DirectSupport);
    }

    #[test]
    fn entity_properties_participate_in_direct_support_and_conflicts() {
        use std::collections::HashMap;

        let registry = registry();
        let laws = LawBook::from_registry(&registry);
        let mut properties = HashMap::new();
        properties.insert("年龄".to_string(), "25".to_string());
        properties.insert(
            "师傅".to_string(),
            "http://example.org/kg/entity/李四".to_string(),
        );
        let facts = vec![PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人物".to_string(),
            properties,
        }];
        let gate = FrameGate::from_facts(&facts, registry, TypeSystem::new(), laws);
        assert_eq!(gate.fact_count(), 1, "关系 IRI 不应作为属性边重复计入");

        let direct = gate.evaluate(&RelationFrame::new("张三", "年龄", "25"));
        assert_eq!(direct.verdict, GateVerdict::DirectSupport);

        let conflict = gate.evaluate(&RelationFrame::new("张三", "年龄", "30"));
        assert_eq!(conflict.verdict, GateVerdict::Conflict);
        assert!(conflict
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::FunctionalConflict));
    }

    #[test]
    fn diagnostics_are_deterministic_and_narrated() {
        let gate = gate(&[("张三", "师傅", "李四")]);
        let frames = vec![
            RelationFrame::new("张三", "师傅", "李四"),
            RelationFrame::new("张三", "师傅", "王五"),
        ];
        let first = diagnose(&frames, &gate, 2);
        let second = diagnose(&frames, &gate, 2);
        assert_eq!(first, second);
        assert_eq!(first.direct_support, 1);
        assert_eq!(first.unsupported, 1);
        assert_eq!(first.samples.len(), 2);
        let narration = narrate_frame(&frames[0], &first.samples[0].report);
        assert!(narration.contains("直接支持"));
        assert!(narration.contains("之间存在「师傅」关系"));
        let age = RelationFrame::new("张三", "年龄", "18");
        let age_report = gate.evaluate(&age);
        assert!(narrate_frame(&age, &age_report).contains("张三的年龄是18"));
        assert!(!diagnostics_digest(&first).is_empty());
    }
}

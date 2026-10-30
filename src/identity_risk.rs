//! 身份风险审计（Identity Risk Audit）：只读、确定性、按风险排序。
//!
//! 实体身份注册表把「提名 → 确认 → 合并」写成可回放的账本，但账本本身
//! 可能积累结构性风险：同一别名被确认给多个身份、待确认别名遮蔽规范名、
//! 重定向成环或悬空、合并跨越不相交类型。本模块不修改注册表，只扫描
//! 已落盘的记录并输出可排序的风险清单，供「身份台账」命令与发布检查使用。
//!
//! 风险等级是离散档位（Critical > High > Medium > Low），排序规则固定：
//! 先按等级降序，再按风险类型、涉及词面与说明字典序。摘要只覆盖风险清单，
//! 不依赖挂钟、随机数或哈希遍历序。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::entity_registry::{fold_surface, EntityRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskSeverity {
    pub fn code(self) -> &'static str {
        match self {
            RiskSeverity::Low => "low",
            RiskSeverity::Medium => "medium",
            RiskSeverity::High => "high",
            RiskSeverity::Critical => "critical",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            RiskSeverity::Low => "低",
            RiskSeverity::Medium => "中",
            RiskSeverity::High => "高",
            RiskSeverity::Critical => "严重",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityRiskKind {
    /// 同一别名词面被确认给多个身份。
    AliasCollision,
    /// 一个身份的规范名同时被另一个身份当作已确认别名。
    CanonicalShadow,
    /// 待确认别名遮蔽了另一个身份的规范名或已确认别名。
    PendingShadow,
    /// 重定向链成环。
    RedirectCycle,
    /// 重定向目标不存在。
    DanglingRedirect,
    /// 身份合并到自身。
    SelfRedirect,
    /// 合并双方的类型集合不相交。
    DisjointTypeMerge,
    /// 同一对身份重复合并。
    DuplicateMerge,
    /// 合并编年史引用了当前不存在的身份。
    OrphanMerge,
}

impl IdentityRiskKind {
    pub fn code(self) -> &'static str {
        match self {
            IdentityRiskKind::AliasCollision => "alias_collision",
            IdentityRiskKind::CanonicalShadow => "canonical_shadow",
            IdentityRiskKind::PendingShadow => "pending_shadow",
            IdentityRiskKind::RedirectCycle => "redirect_cycle",
            IdentityRiskKind::DanglingRedirect => "dangling_redirect",
            IdentityRiskKind::SelfRedirect => "self_redirect",
            IdentityRiskKind::DisjointTypeMerge => "disjoint_type_merge",
            IdentityRiskKind::DuplicateMerge => "duplicate_merge",
            IdentityRiskKind::OrphanMerge => "orphan_merge",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            IdentityRiskKind::AliasCollision => "别名碰撞",
            IdentityRiskKind::CanonicalShadow => "规范名被别名遮蔽",
            IdentityRiskKind::PendingShadow => "待确认别名遮蔽",
            IdentityRiskKind::RedirectCycle => "重定向环",
            IdentityRiskKind::DanglingRedirect => "悬空重定向",
            IdentityRiskKind::SelfRedirect => "自指向重定向",
            IdentityRiskKind::DisjointTypeMerge => "跨类型合并",
            IdentityRiskKind::DuplicateMerge => "重复合并",
            IdentityRiskKind::OrphanMerge => "合并记录悬空",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityRiskFinding {
    pub kind: IdentityRiskKind,
    pub severity: RiskSeverity,
    pub surfaces: Vec<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityRiskReport {
    pub identities: usize,
    pub merges: usize,
    pub findings: Vec<IdentityRiskFinding>,
    pub digest: String,
}

impl IdentityRiskReport {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for severity in [
            RiskSeverity::Critical,
            RiskSeverity::High,
            RiskSeverity::Medium,
            RiskSeverity::Low,
        ] {
            counts.insert(
                severity.code(),
                self.findings
                    .iter()
                    .filter(|finding| finding.severity == severity)
                    .count(),
            );
        }
        counts
    }

    pub fn render_text(&self) -> String {
        let counts = self.counts();
        let mut out = format!(
            "身份风险审计：{} 个身份 / {} 条合并 / {} 项风险（严重 {}，高 {}，中 {}，低 {}，摘要 {}）",
            self.identities,
            self.merges,
            self.findings.len(),
            counts["critical"],
            counts["high"],
            counts["medium"],
            counts["low"],
            self.digest
        );
        for finding in &self.findings {
            let _ = write!(
                out,
                "\n  [{}] {}：{}（{}）",
                finding.severity.label_zh(),
                finding.kind.label_zh(),
                finding.surfaces.join(" / "),
                finding.detail
            );
        }
        out
    }
}

impl EntityRegistry {
    /// 只读身份风险审计；不修改注册表，不写盘。
    pub fn risk_audit(&self) -> IdentityRiskReport {
        audit(self)
    }
}

/// 扫描实体身份注册表，输出确定性风险报告。
pub fn audit(registry: &EntityRegistry) -> IdentityRiskReport {
    let mut findings = Vec::new();
    let mut canonical_by_folded: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut confirmed_aliases: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut pending_aliases: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut records_by_name: BTreeMap<String, &crate::entity_registry::EntityRecord> =
        BTreeMap::new();

    for record in registry.records() {
        records_by_name.insert(record.canonical.clone(), record);
        canonical_by_folded
            .entry(fold_surface(&record.canonical))
            .or_default()
            .insert(record.canonical.clone());
        for (folded, alias) in &record.aliases {
            match alias.status {
                crate::entity_registry::EntityStatus::Confirmed => {
                    confirmed_aliases
                        .entry(folded.clone())
                        .or_default()
                        .insert(record.canonical.clone());
                }
                crate::entity_registry::EntityStatus::Pending => {
                    pending_aliases
                        .entry(folded.clone())
                        .or_default()
                        .insert(record.canonical.clone());
                }
                crate::entity_registry::EntityStatus::Rejected => {}
            }
        }
    }

    for (surface, owners) in &confirmed_aliases {
        if owners.len() > 1 {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::AliasCollision,
                severity: RiskSeverity::High,
                surfaces: std::iter::once(surface.clone())
                    .chain(owners.iter().cloned())
                    .collect(),
                detail: format!("别名词面「{}」同时确认给 {} 个身份", surface, owners.len()),
            });
        }
    }

    for (surface, canonicals) in &canonical_by_folded {
        if canonicals.len() > 1 {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::CanonicalShadow,
                severity: RiskSeverity::Critical,
                surfaces: std::iter::once(surface.clone())
                    .chain(canonicals.iter().cloned())
                    .collect(),
                detail: format!("折叠后「{}」对应多个规范身份", surface),
            });
        }
        if let Some(owners) = confirmed_aliases.get(surface) {
            let shadowed: BTreeSet<&String> = owners.difference(canonicals).collect();
            if !shadowed.is_empty() {
                findings.push(IdentityRiskFinding {
                    kind: IdentityRiskKind::CanonicalShadow,
                    severity: RiskSeverity::High,
                    surfaces: std::iter::once(surface.clone())
                        .chain(canonicals.iter().cloned())
                        .chain(shadowed.into_iter().cloned())
                        .collect(),
                    detail: format!("规范名「{}」同时被其它身份确认为别名", surface),
                });
            }
        }
    }

    for (surface, owners) in &pending_aliases {
        let mut shadowed: BTreeSet<String> = BTreeSet::new();
        if let Some(canonicals) = canonical_by_folded.get(surface) {
            for canonical in canonicals {
                if !owners.contains(canonical) {
                    shadowed.insert(canonical.clone());
                }
            }
        }
        if let Some(confirmed) = confirmed_aliases.get(surface) {
            for canonical in confirmed {
                if !owners.contains(canonical) {
                    shadowed.insert(canonical.clone());
                }
            }
        }
        if !shadowed.is_empty() {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::PendingShadow,
                severity: RiskSeverity::Medium,
                surfaces: std::iter::once(surface.clone())
                    .chain(owners.iter().cloned())
                    .chain(shadowed.into_iter())
                    .collect(),
                detail: "待确认别名与已有规范名/已确认别名重叠，签字前可能误绑".to_string(),
            });
        }
    }

    let mut cycle_keys: BTreeSet<String> = BTreeSet::new();
    for record in registry.records() {
        let Some(target) = record.merged_into.clone() else {
            continue;
        };
        if target == record.canonical {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::SelfRedirect,
                severity: RiskSeverity::Critical,
                surfaces: vec![record.canonical.clone()],
                detail: "身份重定向到自身".to_string(),
            });
            continue;
        }
        if !records_by_name.contains_key(&target) {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::DanglingRedirect,
                severity: RiskSeverity::High,
                surfaces: vec![record.canonical.clone(), target.clone()],
                detail: format!("重定向目标「{}」不存在", target),
            });
            continue;
        }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        seen.insert(record.canonical.clone());
        let mut cursor = target;
        loop {
            if !seen.insert(cursor.clone()) {
                let members: Vec<String> = seen.iter().cloned().collect();
                let key = members.join("\u{1}");
                if cycle_keys.insert(key) {
                    findings.push(IdentityRiskFinding {
                        kind: IdentityRiskKind::RedirectCycle,
                        severity: RiskSeverity::Critical,
                        surfaces: members.clone(),
                        detail: "重定向链成环，解析不会终止".to_string(),
                    });
                }
                break;
            }
            let Some(next) = records_by_name
                .get(&cursor)
                .and_then(|record| record.merged_into.clone())
            else {
                break;
            };
            if !records_by_name.contains_key(&next) {
                findings.push(IdentityRiskFinding {
                    kind: IdentityRiskKind::DanglingRedirect,
                    severity: RiskSeverity::High,
                    surfaces: vec![cursor.clone(), next.clone()],
                    detail: format!("重定向目标「{}」不存在", next),
                });
                break;
            }
            cursor = next;
        }
    }

    let mut merge_pairs: BTreeMap<(String, String), usize> = BTreeMap::new();
    for record in registry.merges() {
        *merge_pairs
            .entry((record.from.clone(), record.into.clone()))
            .or_default() += 1;
        if !records_by_name.contains_key(&record.from)
            || !records_by_name.contains_key(&record.into)
        {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::OrphanMerge,
                severity: RiskSeverity::Medium,
                surfaces: vec![record.from.clone(), record.into.clone()],
                detail: "合并编年史引用了不存在的身份".to_string(),
            });
        }
        if !record.from_types.is_empty()
            && !record.into_types.is_empty()
            && record.from_types.is_disjoint(&record.into_types)
        {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::DisjointTypeMerge,
                severity: RiskSeverity::High,
                surfaces: vec![record.from.clone(), record.into.clone()],
                detail: format!(
                    "来源类型 {} 与目标类型 {} 无交集",
                    record
                        .from_types
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、"),
                    record
                        .into_types
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                ),
            });
        }
    }
    for ((from, into), count) in merge_pairs {
        if count > 1 {
            findings.push(IdentityRiskFinding {
                kind: IdentityRiskKind::DuplicateMerge,
                severity: RiskSeverity::Low,
                surfaces: vec![from, into],
                detail: format!("同一对身份合并了 {} 次", count),
            });
        }
    }

    findings.sort_by(|a, b| {
        b.severity
            .cmp(&a.severity)
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.surfaces.cmp(&b.surfaces))
            .then_with(|| a.detail.cmp(&b.detail))
    });
    findings.dedup();

    let mut digest_input = String::new();
    for finding in &findings {
        let _ = write!(
            digest_input,
            "{}|{}|{}|{}\n",
            finding.kind.code(),
            finding.severity.code(),
            finding.surfaces.join("\u{1}"),
            finding.detail
        );
    }
    let digest = fnv1a_hex(digest_input.as_bytes());

    IdentityRiskReport {
        identities: records_by_name.len(),
        merges: registry.merges().len(),
        findings,
        digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_registry::{EntityProfileConfig, EntityRegistry, NominationOrigin};

    fn types(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn clean_registry_has_no_findings() {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("苹果", &types(&["水果"]), &[], "test");
        registry.observe_kb_entity("苹果公司", &types(&["组织"]), &[], "test");
        registry.nominate_alias("苹果果子", "苹果", NominationOrigin::Import, "模式");
        let report = audit(&registry);
        assert!(report.is_clean(), "{}", report.render_text());
        assert_eq!(report.identities, 2);
        assert!(report.render_text().contains("摘要"));
    }

    #[test]
    fn alias_collision_is_ranked_high() {
        let mut registry = EntityRegistry::new();
        registry.declare(&EntityProfileConfig {
            name: "苹果果实".to_string(),
            aliases: vec!["苹果".to_string()],
            etype: vec!["水果".to_string()],
            description: None,
        });
        registry.declare(&EntityProfileConfig {
            name: "苹果公司".to_string(),
            aliases: vec!["苹果".to_string()],
            etype: vec!["组织".to_string()],
            description: None,
        });
        let report = audit(&registry);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == IdentityRiskKind::AliasCollision
                && finding.severity == RiskSeverity::High));
        assert!(!report.render_text().contains("[低]"));
    }

    #[test]
    fn pending_alias_shadowing_canonical_is_flagged() {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("苹果", &types(&["水果"]), &[], "test");
        registry.observe_kb_entity("苹果公司", &types(&["组织"]), &[], "test");
        registry.nominate_alias("苹果公司", "苹果", NominationOrigin::Import, "模式");
        let report = audit(&registry);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == IdentityRiskKind::PendingShadow));
    }

    #[test]
    fn redirect_cycle_and_dangling_are_detected() {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("甲", &types(&["人物"]), &[], "test");
        registry.observe_kb_entity("乙", &types(&["人物"]), &[], "test");
        registry.merge("甲", "乙").expect("merge");
        let mut damaged: serde_json::Value =
            serde_json::from_str(&registry.to_json()).expect("json");
        let records = damaged["records"].as_object_mut().expect("records");
        records["甲"]["merged_into"] = serde_json::json!("乙");
        records["乙"]["merged_into"] = serde_json::json!("甲");
        let damaged = EntityRegistry::from_json(&damaged.to_string()).expect("restore");
        let report = audit(&damaged);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == IdentityRiskKind::RedirectCycle));

        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("甲", &types(&["人物"]), &[], "test");
        let mut damaged: serde_json::Value =
            serde_json::from_str(&registry.to_json()).expect("json");
        damaged["records"]["甲"]["merged_into"] = serde_json::json!("不存在");
        let damaged = EntityRegistry::from_json(&damaged.to_string()).expect("restore");
        let report = audit(&damaged);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == IdentityRiskKind::DanglingRedirect));
    }

    #[test]
    fn disjoint_type_merge_is_flagged_from_snapshot() {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("苹果", &types(&["水果"]), &[], "test");
        registry.observe_kb_entity("苹果公司", &types(&["组织"]), &[], "test");
        registry.merge("苹果", "苹果公司").expect("merge");
        let report = audit(&registry);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == IdentityRiskKind::DisjointTypeMerge));
    }

    #[test]
    fn report_is_deterministic_and_digest_changes_with_findings() {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("甲", &types(&["人物"]), &[], "test");
        registry.observe_kb_entity("乙", &types(&["组织"]), &[], "test");
        registry.merge("甲", "乙").expect("merge");
        let first = audit(&registry);
        let second = audit(&registry);
        assert_eq!(first, second);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.render_text(), second.render_text());
        assert!(!first.digest.is_empty());
    }
}

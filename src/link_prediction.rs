use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleTemplate {
    Compose,
    Inverse,
    CommonNeighbor,
    CommonObject,
}

impl RuleTemplate {
    pub fn name(&self) -> &'static str {
        match self {
            RuleTemplate::Compose => "组合",
            RuleTemplate::Inverse => "逆向",
            RuleTemplate::CommonNeighbor => "共同邻居",
            RuleTemplate::CommonObject => "共同主体",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HornRule {
    pub template: RuleTemplate,
    pub body: Vec<String>,
    pub head: String,
    pub support: usize,
    pub body_count: usize,
    pub confidence: f64,
    pub head_coverage: f64,
}

impl HornRule {
    pub fn body1(&self) -> &str {
        &self.body[0]
    }

    pub fn body2(&self) -> Option<&str> {
        self.body.get(1).map(|s| s.as_str())
    }

    pub fn rule_text(&self) -> String {
        match self.template {
            RuleTemplate::Compose => format!(
                "{}(A,B) ∧ {}(B,C) ⇒ {}(A,C)",
                self.body1(),
                self.body2().unwrap_or(""),
                self.head
            ),
            RuleTemplate::Inverse => format!("{}(A,B) ⇒ {}(B,A)", self.body1(), self.head),
            RuleTemplate::CommonNeighbor => format!(
                "{}(A,C) ∧ {}(B,C) ⇒ {}(A,B)",
                self.body1(),
                self.body2().unwrap_or(""),
                self.head
            ),
            RuleTemplate::CommonObject => format!(
                "{}(C,A) ∧ {}(C,B) ⇒ {}(A,B)",
                self.body1(),
                self.body2().unwrap_or(""),
                self.head
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub triple: Triple,
    pub rule_index: usize,
    pub confidence: f64,
    pub evidence: Vec<Triple>,
}

/// 候选项：延迟证据构造，排序截断后才还原完整证据链
struct Candidate {
    triple: Triple,
    rule_index: usize,
    confidence: f64,
    anchor: String,
    /// 枚举顺序（唯一、递增），作为并列时的稳定排序键
    order: usize,
}

pub struct LinkPredictor {
    triples: Vec<Triple>,
    pairs_by_pred: HashMap<String, HashSet<(String, String)>>,
    out_edges: HashMap<String, HashMap<String, HashSet<String>>>,
    in_edges: HashMap<String, HashMap<String, HashSet<String>>>,
    participants: HashSet<String>,
    rules: Vec<HornRule>,
    max_body_pairs: usize,
    max_enumeration: usize,
}

impl Default for LinkPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkPredictor {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            pairs_by_pred: HashMap::new(),
            out_edges: HashMap::new(),
            in_edges: HashMap::new(),
            participants: HashSet::new(),
            rules: Vec::new(),
            max_body_pairs: 20_000,
            max_enumeration: 5_000,
        }
    }

    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }

    pub fn rules(&self) -> &[HornRule] {
        &self.rules
    }

    pub fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        // 已存在的三元组直接跳过
        if self.has_triple(subject, predicate, object) {
            return;
        }
        let triple = Triple::new(subject, predicate, object);
        self.triples.push(triple);
        self.pairs_by_pred
            .entry(predicate.to_string())
            .or_default()
            .insert((subject.to_string(), object.to_string()));
        self.out_edges
            .entry(predicate.to_string())
            .or_default()
            .entry(subject.to_string())
            .or_default()
            .insert(object.to_string());
        self.in_edges
            .entry(predicate.to_string())
            .or_default()
            .entry(object.to_string())
            .or_default()
            .insert(subject.to_string());
        self.participants.insert(subject.to_string());
        self.participants.insert(object.to_string());
    }

    fn pairs_of(&self, predicate: &str) -> Option<&HashSet<(String, String)>> {
        self.pairs_by_pred.get(predicate)
    }

    pub fn has_triple(&self, subject: &str, predicate: &str, object: &str) -> bool {
        // out_edges 索引即事实集（add_triple 写入时已去重）
        self.out_edges
            .get(predicate)
            .and_then(|by_subject| by_subject.get(subject))
            .map_or(false, |objects| objects.contains(object))
    }

    pub fn has_predicate(&self, predicate: &str) -> bool {
        self.pairs_by_pred.contains_key(predicate)
    }

    // 数据驱动的对称关系判定：图中该关系的双向对占多数时视为对称，
    // 供语义蕴含做「逆向成立」的证据（如配偶、朋友等）。
    pub fn is_relation_symmetric(&self, predicate: &str) -> bool {
        let Some(pairs) = self.pairs_by_pred.get(predicate) else {
            return false;
        };
        if pairs.is_empty() {
            return false;
        }
        let Some(by_subject) = self.out_edges.get(predicate) else {
            return false;
        };
        // 反向对 (b,a) 存在 ⟺ by_subject[b] 含 a
        let mutual = pairs
            .iter()
            .filter(|(a, b)| by_subject.get(b).map_or(false, |objects| objects.contains(a)))
            .count();
        mutual >= 1 && mutual * 2 >= pairs.len()
    }

    pub fn objects_of(&self, subject: &str, predicate: &str) -> Vec<String> {
        let mut objects: Vec<String> = self
            .out_edges
            .get(predicate)
            .and_then(|by_subject| by_subject.get(subject))
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        objects.sort();
        objects
    }

    pub fn subjects_of(&self, predicate: &str, object: &str) -> Vec<String> {
        let mut subjects: Vec<String> = self
            .in_edges
            .get(predicate)
            .and_then(|by_object| by_object.get(object))
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        subjects.sort();
        subjects
    }

    pub fn participates(&self, entity: &str) -> bool {
        self.participants.contains(entity)
    }

    fn compose_body_pairs(&self, p1: &str, p2: &str) -> Option<HashSet<(String, String)>> {
        let in1 = self.in_edges.get(p1)?;
        let out2 = self.out_edges.get(p2)?;
        let mut pairs = HashSet::new();
        for (y, xs) in in1 {
            if let Some(zs) = out2.get(y) {
                for x in xs {
                    for z in zs {
                        if x == z {
                            continue;
                        }
                        pairs.insert((x.clone(), z.clone()));
                        if pairs.len() > self.max_body_pairs {
                            return None;
                        }
                    }
                }
            }
        }
        if pairs.is_empty() {
            None
        } else {
            Some(pairs)
        }
    }

    fn common_body_pairs(&self, p1: &str, p2: &str, outgoing: bool) -> Option<HashSet<(String, String)>> {
        let first = if outgoing {
            self.out_edges.get(p1)?
        } else {
            self.in_edges.get(p1)?
        };
        let second = if outgoing {
            self.out_edges.get(p2)?
        } else {
            self.in_edges.get(p2)?
        };
        let mut pairs = HashSet::new();
        for (anchor, one) in first {
            if let Some(two) = second.get(anchor) {
                for a in one {
                    for b in two {
                        if a == b {
                            continue;
                        }
                        pairs.insert((a.clone(), b.clone()));
                        if pairs.len() > self.max_body_pairs {
                            return None;
                        }
                    }
                }
            }
        }
        if pairs.is_empty() {
            None
        } else {
            Some(pairs)
        }
    }

    fn build_rule(
        &self,
        template: RuleTemplate,
        body: &[&str],
        head: &str,
        body_pairs: &HashSet<(String, String)>,
        min_support: usize,
        min_confidence: f64,
    ) -> Option<HornRule> {
        let head_pairs = self.pairs_of(head)?;
        // 从较小集合侧做包含计数，交集大小不变
        let support = if body_pairs.len() <= head_pairs.len() {
            body_pairs
                .iter()
                .filter(|pair| head_pairs.contains(*pair))
                .count()
        } else {
            head_pairs
                .iter()
                .filter(|pair| body_pairs.contains(*pair))
                .count()
        };
        if support < min_support || body_pairs.is_empty() {
            return None;
        }
        let confidence = support as f64 / body_pairs.len() as f64;
        if confidence < min_confidence {
            return None;
        }
        let head_coverage = if head_pairs.is_empty() {
            0.0
        } else {
            support as f64 / head_pairs.len() as f64
        };
        // 通过全部过滤后克隆体部/头部名称
        Some(HornRule {
            template,
            body: body.iter().map(|name| name.to_string()).collect(),
            head: head.to_string(),
            support,
            body_count: body_pairs.len(),
            confidence,
            head_coverage,
        })
    }

    pub fn mine_rules(&mut self, min_support: usize, min_confidence: f64) {
        let predicates: Vec<String> = self.pairs_by_pred.keys().cloned().collect();
        let mut rules = Vec::new();

        for p1 in &predicates {
            // 逆向规则只依赖 p1
            let reversed: Option<HashSet<(String, String)>> = self
                .pairs_of(p1)
                .map(|pairs1| pairs1.iter().map(|(o, s)| (s.clone(), o.clone())).collect());
            for p2 in &predicates {
                if let Some(reversed) = &reversed {
                    if let Some(rule) = self.build_rule(
                        RuleTemplate::Inverse,
                        &[p1.as_str()],
                        p2,
                        reversed,
                        min_support,
                        min_confidence,
                    ) {
                        rules.push(rule);
                    }
                }

                if let Some(body) = self.compose_body_pairs(p1, p2) {
                    // 支撑量不可能超过体部对数：先剪掉不足阈值的体部
                    if body.len() >= min_support {
                        let body_names = [p1.as_str(), p2.as_str()];
                        for p3 in &predicates {
                            if let Some(rule) = self.build_rule(
                                RuleTemplate::Compose,
                                &body_names,
                                p3,
                                &body,
                                min_support,
                                min_confidence,
                            ) {
                                rules.push(rule);
                            }
                        }
                    }
                }

                if let Some(body) = self.common_body_pairs(p1, p2, false) {
                    if body.len() >= min_support {
                        let body_names = [p1.as_str(), p2.as_str()];
                        for p3 in &predicates {
                            if let Some(rule) = self.build_rule(
                                RuleTemplate::CommonNeighbor,
                                &body_names,
                                p3,
                                &body,
                                min_support,
                                min_confidence,
                            ) {
                                rules.push(rule);
                            }
                        }
                    }
                }

                if let Some(body) = self.common_body_pairs(p1, p2, true) {
                    if body.len() >= min_support {
                        let body_names = [p1.as_str(), p2.as_str()];
                        for p3 in &predicates {
                            if let Some(rule) = self.build_rule(
                                RuleTemplate::CommonObject,
                                &body_names,
                                p3,
                                &body,
                                min_support,
                                min_confidence,
                            ) {
                                rules.push(rule);
                            }
                        }
                    }
                }
            }
        }

        rules.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| a.head.cmp(&b.head))
        });
        rules.dedup_by(|a, b| {
            a.template == b.template && a.body == b.body && a.head == b.head
        });
        self.rules = rules;
    }

    fn try_prediction<F>(
        &self,
        seen: &mut HashSet<Triple>,
        generated: &mut usize,
        rule_index: usize,
        confidence: f64,
        triple: Triple,
        anchor: &str,
        matches: F,
    ) -> Option<Candidate>
    where
        F: FnOnce(&Triple) -> bool,
    {
        // out_edges 索引即事实集
        if self.has_triple(&triple.subject, &triple.predicate, &triple.object)
            || !seen.insert(triple.clone())
        {
            return None;
        }
        *generated += 1;
        if !matches(&triple) {
            return None;
        }
        Some(Candidate {
            triple,
            rule_index,
            confidence,
            anchor: anchor.to_string(),
            order: *generated,
        })
    }

    /// 用规则与候选自带的锚点变量还原证据链（排序截断后才调用）
    fn materialize_prediction(&self, candidate: Candidate) -> Prediction {
        let triple = candidate.triple;
        let evidence = match self.rules.get(candidate.rule_index) {
            Some(rule) => match rule.template {
                RuleTemplate::Compose => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    vec![
                        Triple::new(&triple.subject, p1, &candidate.anchor),
                        Triple::new(&candidate.anchor, p2, &triple.object),
                    ]
                }
                RuleTemplate::Inverse => {
                    vec![Triple::new(&triple.object, rule.body1(), &triple.subject)]
                }
                RuleTemplate::CommonNeighbor => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    vec![
                        Triple::new(&triple.subject, p1, &candidate.anchor),
                        Triple::new(&triple.object, p2, &candidate.anchor),
                    ]
                }
                RuleTemplate::CommonObject => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    vec![
                        Triple::new(&candidate.anchor, p1, &triple.subject),
                        Triple::new(&candidate.anchor, p2, &triple.object),
                    ]
                }
            },
            None => Vec::new(),
        };
        Prediction {
            triple,
            rule_index: candidate.rule_index,
            confidence: candidate.confidence,
            evidence,
        }
    }

    fn enumerate_predictions_matching<F>(&self, cap: usize, matches: F) -> Vec<Candidate>
    where
        F: Fn(&Triple) -> bool,
    {
        let mut out = Vec::new();
        let mut seen: HashSet<Triple> = HashSet::new();
        let mut generated = 0usize;

        for (idx, rule) in self.rules.iter().enumerate() {
            match rule.template {
                RuleTemplate::Compose => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    let p3 = rule.head.as_str();
                    if let (Some(in1), Some(out2)) = (self.in_edges.get(p1), self.out_edges.get(p2)) {
                        for (y, xs) in in1 {
                            let Some(zs) = out2.get(y) else { continue };
                            for x in xs {
                                for z in zs {
                                    if x == z {
                                        continue;
                                    }
                                    let triple = Triple::new(x, p3, z);
                                    if let Some(pred) = self.try_prediction(
                                        &mut seen,
                                        &mut generated,
                                        idx,
                                        rule.confidence,
                                        triple,
                                        y,
                                        &matches,
                                    ) {
                                        out.push(pred);
                                    }
                                    if generated >= cap {
                                        return out;
                                    }
                                }
                            }
                        }
                    }
                }
                RuleTemplate::Inverse => {
                    let p1 = rule.body1();
                    let p2 = rule.head.as_str();
                    if let Some(pairs1) = self.pairs_of(p1) {
                        for (s, o) in pairs1 {
                            let triple = Triple::new(o, p2, s);
                            if let Some(pred) = self.try_prediction(
                                &mut seen,
                                &mut generated,
                                idx,
                                rule.confidence,
                                triple,
                                "",
                                &matches,
                            ) {
                                out.push(pred);
                            }
                            if generated >= cap {
                                return out;
                            }
                        }
                    }
                }
                RuleTemplate::CommonNeighbor => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    let p3 = rule.head.as_str();
                    if let (Some(in1), Some(in2)) = (self.in_edges.get(p1), self.in_edges.get(p2)) {
                        for (z, xs) in in1 {
                            let Some(ys) = in2.get(z) else { continue };
                            for x in xs {
                                for y in ys {
                                    if x == y {
                                        continue;
                                    }
                                    let triple = Triple::new(x, p3, y);
                                    if let Some(pred) = self.try_prediction(
                                        &mut seen,
                                        &mut generated,
                                        idx,
                                        rule.confidence,
                                        triple,
                                        z,
                                        &matches,
                                    ) {
                                        out.push(pred);
                                    }
                                    if generated >= cap {
                                        return out;
                                    }
                                }
                            }
                        }
                    }
                }
                RuleTemplate::CommonObject => {
                    let p1 = rule.body1();
                    let p2 = rule.body2().unwrap_or("");
                    let p3 = rule.head.as_str();
                    if let (Some(out1), Some(out2)) = (self.out_edges.get(p1), self.out_edges.get(p2)) {
                        for (z, xs) in out1 {
                            let Some(ys) = out2.get(z) else { continue };
                            for x in xs {
                                for y in ys {
                                    if x == y {
                                        continue;
                                    }
                                    let triple = Triple::new(x, p3, y);
                                    if let Some(pred) = self.try_prediction(
                                        &mut seen,
                                        &mut generated,
                                        idx,
                                        rule.confidence,
                                        triple,
                                        z,
                                        &matches,
                                    ) {
                                        out.push(pred);
                                    }
                                    if generated >= cap {
                                        return out;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }

    fn enumerate_predictions(&self, cap: usize) -> Vec<Candidate> {
        self.enumerate_predictions_matching(cap, |_| true)
    }

    fn compare_candidates(a: &Candidate, b: &Candidate) -> Ordering {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.triple.subject.cmp(&b.triple.subject))
            .then_with(|| a.triple.object.cmp(&b.triple.object))
            .then_with(|| a.order.cmp(&b.order))
    }

    fn sort_candidates(candidates: &mut [Candidate]) {
        candidates.sort_by(Self::compare_candidates);
    }

    /// 只保留排序最靠前的 `limit` 个候选，再对保留部分排序。
    /// 比较键是包含枚举序的全序，结果与"全量稳定排序后截断"一致。
    fn top_candidates(candidates: &mut Vec<Candidate>, limit: usize) {
        if limit < candidates.len() {
            candidates.select_nth_unstable_by(limit, Self::compare_candidates);
            candidates.truncate(limit);
        }
        Self::sort_candidates(candidates);
    }

    pub fn predict_missing(&self, limit: usize) -> Vec<Prediction> {
        let cap = self.max_enumeration.max(limit.saturating_mul(20));
        let mut candidates = self.enumerate_predictions(cap);
        Self::top_candidates(&mut candidates, limit);
        candidates
            .into_iter()
            .map(|candidate| self.materialize_prediction(candidate))
            .collect()
    }

    pub fn predict_for_entity(
        &self,
        entity: &str,
        predicate: Option<&str>,
        limit: usize,
    ) -> Vec<Prediction> {
        let mut candidates =
            self.enumerate_predictions_matching(self.max_enumeration, |triple| {
                (triple.subject == entity || triple.object == entity)
                    && predicate.map_or(true, |q| triple.predicate == q)
            });
        Self::top_candidates(&mut candidates, limit);
        candidates
            .into_iter()
            .map(|candidate| self.materialize_prediction(candidate))
            .collect()
    }

    pub fn explain(&self, prediction: &Prediction) -> String {
        let Some(rule) = self.rules.get(prediction.rule_index) else {
            return format!(
                "推断「{} 的 {} 是 {}」（置信度 {:.2}）",
                prediction.triple.subject,
                prediction.triple.predicate,
                prediction.triple.object,
                prediction.confidence
            );
        };
        let head = &prediction.triple;
        match rule.template {
            RuleTemplate::Compose => {
                let e0 = &prediction.evidence[0];
                let e1 = &prediction.evidence[1];
                format!(
                    "因为「{} 的 {} 是 {}」且「{} 的 {} 是 {}」，根据规则 {} 推断：「{} 的 {} 是 {}」（支持 {} 例，置信度 {:.2}）",
                    e0.subject, e0.predicate, e0.object,
                    e1.subject, e1.predicate, e1.object,
                    rule.rule_text(),
                    head.subject, head.predicate, head.object,
                    rule.support, rule.confidence
                )
            }
            RuleTemplate::Inverse => {
                let e0 = &prediction.evidence[0];
                format!(
                    "因为「{} 的 {} 是 {}」，根据规则 {} 推断：「{} 的 {} 是 {}」（支持 {} 例，置信度 {:.2}）",
                    e0.subject, e0.predicate, e0.object,
                    rule.rule_text(),
                    head.subject, head.predicate, head.object,
                    rule.support, rule.confidence
                )
            }
            RuleTemplate::CommonNeighbor => {
                let e0 = &prediction.evidence[0];
                let e1 = &prediction.evidence[1];
                format!(
                    "因为「{} 的 {} 是 {}」且「{} 的 {} 是 {}」，两者共享关联「{}」，根据规则 {} 推断：「{} 的 {} 是 {}」（支持 {} 例，置信度 {:.2}）",
                    e0.subject, e0.predicate, e0.object,
                    e1.subject, e1.predicate, e1.object,
                    e0.object,
                    rule.rule_text(),
                    head.subject, head.predicate, head.object,
                    rule.support, rule.confidence
                )
            }
            RuleTemplate::CommonObject => {
                let e0 = &prediction.evidence[0];
                let e1 = &prediction.evidence[1];
                format!(
                    "因为「{} 的 {} 是 {}」且「{} 的 {} 是 {}」，两者的 {} 都是「{}」，根据规则 {} 推断：「{} 的 {} 是 {}」（支持 {} 例，置信度 {:.2}）",
                    e0.subject, e0.predicate, e0.object,
                    e1.subject, e1.predicate, e1.object,
                    e0.predicate,
                    e0.subject,
                    rule.rule_text(),
                    head.subject, head.predicate, head.object,
                    rule.support, rule.confidence
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mine_compose_rule_and_predict() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.mine_rules(1, 0.5);

        let rules = predictor.rules();
        assert!(rules.iter().any(|r| {
            r.template == RuleTemplate::Compose
                && r.body == vec!["师傅".to_string(), "师傅".to_string()]
                && r.head == "师祖"
                && r.support == 1
        }));

        let predictions = predictor.predict_missing(10);
        assert!(predictions
            .iter()
            .any(|p| p.triple == Triple::new("李四", "师祖", "赵六")));
        let known_predictions: Vec<&Prediction> = predictions
            .iter()
            .filter(|p| p.triple == Triple::new("张三", "师祖", "王五"))
            .collect();
        assert!(known_predictions.is_empty(), "已知事实不应被重复预测");
    }

    #[test]
    fn test_inverse_rule() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "配偶", "李四");
        predictor.add_triple("李四", "配偶", "张三");
        predictor.add_triple("王五", "配偶", "赵六");
        predictor.mine_rules(1, 0.6);
        let inverse = predictor
            .rules()
            .iter()
            .find(|r| r.template == RuleTemplate::Inverse && r.body1() == "配偶" && r.head == "配偶");
        assert!(inverse.is_some(), "对称关系应挖掘出逆向规则");
        let predictions = predictor.predict_for_entity("赵六", Some("配偶"), 5);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].triple.object, "王五");
    }

    #[test]
    fn test_common_neighbor_rule() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "父亲", "张父");
        predictor.add_triple("张三妹", "父亲", "张父");
        predictor.add_triple("张三", "兄妹", "张三妹");
        predictor.add_triple("李四", "父亲", "李父");
        predictor.add_triple("李四妹", "父亲", "李父");
        predictor.mine_rules(1, 0.2);
        let rules = predictor.rules();
        assert!(rules.iter().any(|r| {
            r.template == RuleTemplate::CommonNeighbor
                && r.body == vec!["父亲".to_string(), "父亲".to_string()]
                && r.head == "兄妹"
        }));
        let predictions = predictor.predict_for_entity("李四", Some("兄妹"), 5);
        assert!(predictions
            .iter()
            .any(|p| p.triple == Triple::new("李四", "兄妹", "李四妹")));
    }

    #[test]
    fn test_confidence_filters_low_quality_rules() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("a", "r1", "b");
        predictor.add_triple("b", "r2", "c");
        predictor.add_triple("a", "r3", "c");
        predictor.add_triple("x", "r1", "y");
        predictor.add_triple("y", "r2", "z");
        predictor.mine_rules(1, 0.5);
        let rule = predictor
            .rules()
            .iter()
            .find(|r| r.head == "r3" && r.template == RuleTemplate::Compose)
            .expect("支持度1、置信度0.5的规则应被挖掘");
        assert_eq!(rule.support, 1);
        assert_eq!(rule.body_count, 2);
        assert!((rule.confidence - 0.5).abs() < 1e-9);
        predictor.mine_rules(1, 1.0);
        assert!(predictor.rules().iter().all(|r| r.confidence >= 1.0));
    }

    #[test]
    fn test_explain_contains_evidence_chain() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.mine_rules(1, 0.5);
        let predictions = predictor.predict_missing(10);
        let prediction = predictions
            .iter()
            .find(|p| p.triple == Triple::new("李四", "师祖", "赵六"))
            .or_else(|| predictions.first())
            .expect("应有预测结果");
        let text = predictor.explain(prediction);
        assert!(text.contains("规则"));
        assert!(text.contains("置信度"));
        assert!(text.contains("因为"));
    }

    #[test]
    fn test_no_rules_without_triples() {
        let mut predictor = LinkPredictor::new();
        predictor.mine_rules(1, 0.5);
        assert!(predictor.rules().is_empty());
        assert!(predictor.predict_missing(10).is_empty());
    }

    #[test]
    fn test_triple_accessors() {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("张三", "师傅", "王五");
        assert!(predictor.has_triple("张三", "师傅", "李四"));
        assert!(!predictor.has_triple("李四", "师傅", "张三"));
        assert!(predictor.has_predicate("师傅"));
        assert!(!predictor.has_predicate("师祖"));
        assert_eq!(
            predictor.objects_of("张三", "师傅"),
            vec!["李四".to_string(), "王五".to_string()]
        );
        assert_eq!(
            predictor.subjects_of("师傅", "李四"),
            vec!["张三".to_string()]
        );
        assert!(predictor.participates("张三"));
        assert!(predictor.participates("李四"));
        assert!(!predictor.participates("赵六"));
    }
}

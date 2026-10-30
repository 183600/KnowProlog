use crate::context_image::fnv1a_hex;
use std::collections::BTreeSet;

pub const MAX_TRACE_STEPS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanOutcome {
    Answered,
    Absent,
    NoPlan,
    UnknownEntity,
    UnknownPredicate,
    NoPath,
    Unsupported,
}

impl PlanOutcome {
    pub fn label(self) -> &'static str {
        match self {
            PlanOutcome::Answered => "已回答",
            PlanOutcome::Absent => "确无记录",
            PlanOutcome::NoPlan => "无法编译计划",
            PlanOutcome::UnknownEntity => "实体未知",
            PlanOutcome::UnknownPredicate => "谓词未知",
            PlanOutcome::NoPath => "无可用路径",
            PlanOutcome::Unsupported => "执行器不支持",
        }
    }

    pub fn is_answered(self) -> bool {
        self == PlanOutcome::Answered
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanOperator {
    Lookup,
    Expansion,
    Closure,
    Prediction,
    Verify,
    Count,
    Exist,
    FindSubject,
    Similar,
    Analogy,
}

impl PlanOperator {
    pub fn label(self) -> &'static str {
        match self {
            PlanOperator::Lookup => "直查",
            PlanOperator::Expansion => "律扩展",
            PlanOperator::Closure => "闭包",
            PlanOperator::Prediction => "预测",
            PlanOperator::Verify => "校验",
            PlanOperator::Count => "计数",
            PlanOperator::Exist => "存在",
            PlanOperator::FindSubject => "反查",
            PlanOperator::Similar => "近邻",
            PlanOperator::Analogy => "类比",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    pub subject: String,
    pub predicate: String,
    pub slot: String,
    pub inverted: bool,
    pub operator: PlanOperator,
    pub law_id: Option<String>,
    pub direct: usize,
    pub derived: usize,
    pub predicted: usize,
}

impl PlanStep {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: String,
        predicate: String,
        slot: String,
        inverted: bool,
        operator: PlanOperator,
        law_id: Option<String>,
        direct: usize,
        derived: usize,
        predicted: usize,
    ) -> Self {
        Self {
            subject,
            predicate,
            slot,
            inverted,
            operator,
            law_id,
            direct,
            derived,
            predicted,
        }
    }

    pub fn hit_count(&self) -> usize {
        self.direct + self.derived + self.predicted
    }

    fn render_line(&self, index: usize) -> String {
        let direction = if self.inverted { " ← " } else { " → " };
        let law = self
            .law_id
            .as_deref()
            .map(|id| format!("，律 {}", id))
            .unwrap_or_default();
        format!(
            "  第 {} 步：{} [{}]{}「{}」命中 {}（直接 {}，推导 {}，预测 {}{}）\n",
            index,
            self.subject,
            self.operator.label(),
            direction,
            self.predicate,
            self.hit_count(),
            self.direct,
            self.derived,
            self.predicted,
            law
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTrace {
    pub question: String,
    pub shape: &'static str,
    pub entity: Option<String>,
    pub outcome: PlanOutcome,
    pub revision: u64,
    pub steps: Vec<PlanStep>,
    pub omitted_steps: usize,
    pub laws: BTreeSet<String>,
    pub visited: usize,
    pub frontier_peak: usize,
}

impl PlanTrace {
    pub fn new(question: &str, shape: &'static str, revision: u64) -> Self {
        Self {
            question: question.to_string(),
            shape,
            entity: None,
            outcome: PlanOutcome::NoPlan,
            revision,
            steps: Vec::new(),
            omitted_steps: 0,
            laws: BTreeSet::new(),
            visited: 0,
            frontier_peak: 0,
        }
    }

    pub fn set_entity(&mut self, entity: &str) {
        self.entity = Some(entity.to_string());
    }

    pub fn observe_hop(&mut self, visited: usize, frontier: usize) {
        self.visited = self.visited.saturating_add(visited);
        self.frontier_peak = self.frontier_peak.max(frontier);
    }

    pub fn record_step(&mut self, step: PlanStep) {
        if let Some(law) = &step.law_id {
            self.laws.insert(law.clone());
        }
        if self.steps.len() < MAX_TRACE_STEPS {
            self.steps.push(step);
        } else {
            self.omitted_steps += 1;
        }
    }

    pub fn finish(&mut self, outcome: PlanOutcome) {
        self.outcome = outcome;
    }

    pub fn digest(&self) -> String {
        let mut material = String::new();
        material.push_str(self.shape);
        material.push('|');
        material.push_str(self.entity.as_deref().unwrap_or(""));
        material.push('|');
        material.push_str(self.outcome.label());
        material.push('|');
        material.push_str(&self.revision.to_string());
        material.push('|');
        for step in &self.steps {
            material.push_str(&step.subject);
            material.push('|');
            material.push_str(&step.slot);
            material.push('|');
            material.push_str(step.operator.label());
            material.push('|');
            material.push_str(step.law_id.as_deref().unwrap_or(""));
            material.push('|');
            material.push_str(&step.direct.to_string());
            material.push('|');
            material.push_str(&step.derived.to_string());
            material.push('|');
            material.push_str(&step.predicted.to_string());
            material.push('\n');
        }
        material.push_str(&self.omitted_steps.to_string());
        material.push('|');
        material.push_str(&self.visited.to_string());
        material.push('|');
        material.push_str(&self.frontier_peak.to_string());
        fnv1a_hex(material.as_bytes())
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "查询轨迹（{}，digest {}，知识版本 {}）：\n",
            self.shape,
            self.digest(),
            self.revision
        );
        if let Some(entity) = &self.entity {
            output.push_str(&format!("  起点：{}\n", entity));
        }
        for (index, step) in self.steps.iter().enumerate() {
            output.push_str(&step.render_line(index + 1));
        }
        if self.omitted_steps > 0 {
            output.push_str(&format!(
                "  ……另有 {} 步未展开（轨迹只保留前 {} 步）。\n",
                self.omitted_steps, MAX_TRACE_STEPS
            ));
        }
        output.push_str(&format!(
            "  结局：{}；访问节点 {}，前沿峰值 {}。\n",
            self.outcome.label(),
            self.visited,
            self.frontier_peak
        ));
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRun {
    pub text: Option<String>,
    pub trace: PlanTrace,
}

impl PlanRun {
    pub fn new(text: Option<String>, trace: PlanTrace) -> Self {
        Self { text, trace }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace() -> PlanTrace {
        let mut trace = PlanTrace::new("甲的上司是谁", "FindObject", 3);
        trace.set_entity("甲");
        trace.observe_hop(2, 2);
        trace.record_step(PlanStep::new(
            "甲".to_string(),
            "上司".to_string(),
            "上司".to_string(),
            false,
            PlanOperator::Closure,
            Some("law:transitive:上司".to_string()),
            1,
            1,
            0,
        ));
        trace.finish(PlanOutcome::Answered);
        trace
    }

    #[test]
    fn digest_and_render_are_deterministic() {
        let first = sample_trace();
        let second = sample_trace();
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.render_text(), second.render_text());
        assert!(first.render_text().contains("已回答"));
        assert!(first.render_text().contains("law:transitive:上司"));
    }

    #[test]
    fn trace_caps_steps_but_keeps_counts() {
        let mut trace = PlanTrace::new("q", "FindObject", 1);
        for _ in 0..(MAX_TRACE_STEPS + 5) {
            trace.record_step(PlanStep::new(
                "甲".to_string(),
                "P".to_string(),
                "P".to_string(),
                false,
                PlanOperator::Lookup,
                None,
                1,
                0,
                0,
            ));
        }
        assert_eq!(trace.steps.len(), MAX_TRACE_STEPS);
        assert_eq!(trace.omitted_steps, 5);
    }

    #[test]
    fn outcome_labels_are_stable() {
        assert!(PlanOutcome::Answered.is_answered());
        assert!(!PlanOutcome::Absent.is_answered());
        assert_eq!(PlanOutcome::UnknownEntity.label(), "实体未知");
    }
}

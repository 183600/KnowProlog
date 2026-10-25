// 指称锚定的查询重写（Query Anchoring）
//
// 实体链接器已经能给出「张山 ≈ 张三」这样的候选证据，但句法重写器只认
// 精确登记的词面；两边各自都很保守，于是错别字/别名问句常常在重写阶段
// 直接弃权。这个模块把 NER 的提及格当作重写先验：
//
//   1) 只从链接器已选中的指称里取候选（不重新发明边界）；
//   2) 只枚举少量「把指称替换成规范名」的变体（不做模糊改写）；
//   3) 每个变体仍交给现有 QueryPlanner / query_rewrite 编译；
//   4) 用链接方法、候选边际、实体类型 × 谓词定义域/值域、答案类型期待
//      做硬闸门与软加分；
//   5) 歧义或证据不足时弃权，返回空列表。
//
// 它只作为精确路径失败后的 fallback 使用：精确问句不会产生任何变体，
// 因此没有回归面。所有分数都是可复现的规则分，回执可打印。

use std::collections::BTreeSet;

use crate::entity_linker::{EntityLinker, LinkMethod, LinkedMention};
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::query_rewrite::{self, RelationLexicon, RelationRewrite};

/// 与 `SymbolicReasoner::LINK_ACCEPT_THRESHOLD` 保持同口径。
pub const ANCHOR_MIN_SCORE: f64 = 0.5;
/// 最佳候选与第二候选的边际；不足即视为歧义，整条指称不参与重写。
pub const ANCHOR_MIN_MARGIN: f64 = 0.08;
/// 单个变体最多替换的指称数（更多替换会迅速放大误读风险）。
pub const ANCHOR_MAX_REPLACEMENTS: usize = 2;
/// 一次最多返回的变体数（按分数、词典序稳定排序）。
pub const ANCHOR_MAX_VARIANTS: usize = 8;

/// 类型核对结论（与谓词定义域/值域、答案类型期待对齐）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeFit {
    Compatible,
    Disjoint,
    Unknown,
}

impl TypeFit {
    pub fn label(self) -> &'static str {
        match self {
            TypeFit::Compatible => "类型相容",
            TypeFit::Disjoint => "类型互斥",
            TypeFit::Unknown => "类型未知",
        }
    }
}

/// 指称锚定需要的只读证据视图。实现方可以是链接器 + 谓词画像 + 本体闭包。
pub trait AnchorEvidence {
    /// 实体的已知类型（可多个；空 = 未知）。
    fn entity_types(&self, entity: &str) -> Vec<String>;
    /// 谓词声明的定义域（空 = 不限制）。
    fn domain_of(&self, predicate: &str) -> Vec<String>;
    /// 谓词声明的值域（空 = 不限制）。
    fn range_of(&self, predicate: &str) -> Vec<String>;

    /// 类型关系：只有能证明互斥时才返回 `Disjoint`。
    fn type_fit(&self, actual: &str, expected: &str) -> TypeFit {
        default_type_fit(actual, expected)
    }
}

/// 无层级信息时的保守判定：只认同一类型/上下位词面包含，其余一律未知。
pub fn default_type_fit(actual: &str, expected: &str) -> TypeFit {
    let actual = actual.trim().to_lowercase();
    let expected = expected.trim().to_lowercase();
    if actual.is_empty() || expected.is_empty() {
        return TypeFit::Unknown;
    }
    if actual == expected {
        return TypeFit::Compatible;
    }
    const TOP_TYPES: [&str; 4] = ["实体", "entity", "thing", "资源"];
    if TOP_TYPES.contains(&expected.as_str()) || TOP_TYPES.contains(&actual.as_str()) {
        return TypeFit::Compatible;
    }
    // 「学生」⊑「人物」这类未显式声明层级的中文类型，词面包含只作为软证据；
    // 反向不成立（「人物」不能自动算作「学生」）。
    if actual.contains(&expected) || expected.contains(&actual) {
        return TypeFit::Compatible;
    }
    TypeFit::Unknown
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnchorReplacement {
    pub surface: String,
    pub canonical: String,
    pub method: LinkMethod,
    pub score: f64,
    pub start: usize,
    pub reasons: Vec<String>,
}

impl AnchorReplacement {
    fn receipt(&self) -> String {
        format!(
            "「{}」→「{}」（{}，置信度 {:.2}）",
            self.surface,
            self.canonical,
            self.method.label(),
            self.score
        )
    }
}

/// 一个通过编译与证据闸门的锚定变体。
#[derive(Debug, Clone)]
pub struct AnchoredQuestion {
    pub question: String,
    pub replacements: Vec<AnchorReplacement>,
    /// QueryPlanner 解析出的计划（若有）。
    pub planner_plan: Option<QueryPlan>,
    /// query_rewrite 编译出的关系重写（若有）。
    pub rewrite: Option<RelationRewrite>,
    pub score: f64,
    pub evidence: Vec<String>,
}

impl AnchoredQuestion {
    /// 可打印回执：替换、编译结果、类型/边际证据。
    pub fn receipt(&self) -> String {
        let replacements = self
            .replacements
            .iter()
            .map(AnchorReplacement::receipt)
            .collect::<Vec<_>>()
            .join("；");
        let plan = self
            .planner_plan
            .as_ref()
            .or_else(|| self.rewrite.as_ref().map(|rewrite| &rewrite.plan));
        let compiled = plan
            .map(|plan| {
                format!(
                    "{}：{}",
                    QueryPlanner::kind(plan),
                    QueryPlanner::format_plan(plan)
                )
            })
            .unwrap_or_else(|| "未编译".to_string());
        let mut output = format!(
            "指称锚定：{}；{}；置信度 {:.2}",
            replacements, compiled, self.score
        );
        if !self.evidence.is_empty() {
            output.push_str(&format!("；证据：{}", self.evidence.join("；")));
        }
        output
    }

    /// 实际用于执行的计划：优先句法规划器，其次关系重写器。
    pub fn preferred_plan(&self) -> Option<&QueryPlan> {
        self.planner_plan
            .as_ref()
            .or_else(|| self.rewrite.as_ref().map(|rewrite| &rewrite.plan))
    }
}

/// 从链接器提及格生成锚定变体。
///
/// `lexicon` 用于最终校验替换后的规范名确实可执行；`evidence` 提供类型
/// 与谓词画像。任何一条证据不足的指称都会被跳过，而不是降低闸门硬猜。
pub fn anchor_questions(
    question: &str,
    linker: &EntityLinker,
    hints: &[String],
    lexicon: &dyn RelationLexicon,
    evidence: &dyn AnchorEvidence,
) -> Vec<AnchoredQuestion> {
    let mut mentions = linker.link_with_context(question, hints);
    augment_fuzzy_windows(question, linker, &mut mentions);
    if mentions.is_empty() {
        return Vec::new();
    }
    let answer_noun = answer_type_noun(question);

    let mut candidates: Vec<(usize, usize, AnchorReplacement)> = Vec::new();
    for mention in mentions {
        let Some(best) = mention.best().cloned() else {
            continue;
        };
        if best.method == LinkMethod::ExactName && mention.mention == best.entity {
            continue;
        }
        if mention.mention == best.entity || best.score < ANCHOR_MIN_SCORE {
            continue;
        }
        let margin = match mention.candidates.get(1) {
            Some(second) => best.score - second.score,
            None => f64::INFINITY,
        };
        if margin < ANCHOR_MIN_MARGIN {
            continue;
        }
        let Some(canonical) = lexicon.canonical_entity(&best.entity) else {
            continue;
        };
        let len = mention.mention.chars().count();
        candidates.push((
            mention.start,
            len,
            AnchorReplacement {
                surface: mention.mention,
                canonical,
                method: best.method,
                score: best.score,
                start: mention.start,
                reasons: best.reasons.clone(),
            },
        ));
    }
    if candidates.is_empty() {
        return Vec::new();
    }
    // 长指称优先；同起点时保留更长跨度，避免「张山」覆盖「张山公司」。
    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let mut variants: Vec<AnchoredQuestion> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for mask in 1usize..(1usize << candidates.len()) {
        let chosen: Vec<&(usize, usize, AnchorReplacement)> = candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, candidate)| candidate)
            .collect();
        if chosen.is_empty() || chosen.len() > ANCHOR_MAX_REPLACEMENTS {
            continue;
        }
        let Some(variant) = replace_spans(question, &chosen) else {
            continue;
        };
        if !seen.insert(variant.clone()) {
            continue;
        }
        let Some(anchored) =
            compile_variant(&variant, &chosen, lexicon, evidence, answer_noun.as_deref())
        else {
            continue;
        };
        variants.push(anchored);
    }

    variants.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.question.cmp(&right.question))
    });
    variants.truncate(ANCHOR_MAX_VARIANTS);
    variants
}

/// 链接器按分隔符切片段；动词/语气词粘连时（「张山认识谁」）整段会过长而
/// 漏掉错别字。这里只在未覆盖的词面窗口上补模糊候选，作为锚定 fallback
/// 的边界假设；窗口两端不能是功能词，且仍需过分数/边际闸门。
fn augment_fuzzy_windows(question: &str, linker: &EntityLinker, mentions: &mut Vec<LinkedMention>) {
    const MAX_WINDOW_CHARS: usize = 8;
    let chars: Vec<char> = question.chars().collect();
    if chars.len() < 2 {
        return;
    }
    let mut occupied: Vec<(usize, usize)> = mentions
        .iter()
        .map(|mention| {
            let len = mention.mention.chars().count();
            (mention.start, mention.start + len)
        })
        .collect();

    let mut windows: Vec<(f64, LinkedMention)> = Vec::new();
    for start in 0..chars.len() {
        let max_len = MAX_WINDOW_CHARS.min(chars.len() - start);
        for len in 2..=max_len {
            let end = start + len;
            if occupied
                .iter()
                .any(|(left, right)| start < *right && end > *left)
            {
                continue;
            }
            let surface: String = chars[start..end].iter().collect();
            if !window_boundary_ok(&surface) {
                continue;
            }
            let mut candidates = linker.fuzzy_candidates(&surface);
            if candidates.is_empty() {
                continue;
            }
            let best = &candidates[0];
            // 独立窗口边界本身是一点额外证据，但不足以把弱候选抬过闸门。
            let score = (best.score + 0.05).min(0.9);
            if score < ANCHOR_MIN_SCORE {
                continue;
            }
            if candidates.len() > 1 && best.score - candidates[1].score < ANCHOR_MIN_MARGIN {
                continue;
            }
            candidates[0].score = score;
            candidates[0]
                .reasons
                .push("词面窗口边界独立命中".to_string());
            windows.push((
                score,
                LinkedMention {
                    mention: surface,
                    start,
                    candidates,
                },
            ));
        }
    }

    windows.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .1
                    .mention
                    .chars()
                    .count()
                    .cmp(&left.1.mention.chars().count())
            })
            .then_with(|| left.1.start.cmp(&right.1.start))
    });
    for (_, mention) in windows {
        let len = mention.mention.chars().count();
        let span = (mention.start, mention.start + len);
        if occupied
            .iter()
            .any(|(left, right)| span.0 < *right && span.1 > *left)
        {
            continue;
        }
        occupied.push(span);
        mentions.push(mention);
    }
}

fn window_boundary_ok(surface: &str) -> bool {
    let mut chars = surface.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let last = surface.chars().next_back().unwrap_or(first);
    if is_window_boundary_char(first) || is_window_boundary_char(last) {
        return false;
    }
    !surface
        .chars()
        .any(|c| c.is_whitespace() || "，。？！、；：,.?!;:()（）【】[]".contains(c))
}

fn is_window_boundary_char(c: char) -> bool {
    matches!(
        c,
        '的' | '了'
            | '着'
            | '过'
            | '吗'
            | '呢'
            | '吧'
            | '啊'
            | '呀'
            | '是'
            | '在'
            | '有'
            | '和'
            | '与'
            | '及'
            | '或'
            | '谁'
            | '哪'
            | '什'
            | '么'
            | '多'
            | '少'
            | '怎'
            | '如'
            | '何'
            | '为'
            | '介'
            | '绍'
            | '请'
            | '告'
            | '诉'
            | '我'
            | '你'
            | '您'
            | '他'
            | '她'
            | '它'
            | '们'
            | '这'
            | '那'
            | '认'
            | '识'
            | '叫'
            | '做'
            | '去'
            | '来'
            | '到'
            | '从'
            | '给'
            | '对'
            | '把'
            | '被'
            | '将'
            | '会'
            | '能'
            | '要'
            | '想'
            | '说'
            | '看'
            | '问'
            | '答'
            | '教'
            | '学'
            | '住'
            | '生'
            | '于'
            | '得'
    )
}

fn replace_spans(question: &str, chosen: &[&(usize, usize, AnchorReplacement)]) -> Option<String> {
    let mut spans: Vec<(usize, usize, &str)> = chosen
        .iter()
        .map(|(start, len, replacement)| (*start, *len, replacement.canonical.as_str()))
        .collect();
    spans.sort_by(|left, right| right.0.cmp(&left.0));
    let mut chars: Vec<char> = question.chars().collect();
    for (start, len, replacement) in spans {
        if start + len > chars.len() {
            return None;
        }
        let replacement_chars: Vec<char> = replacement.chars().collect();
        chars.splice(start..start + len, replacement_chars);
    }
    Some(chars.into_iter().collect())
}

fn compile_variant(
    question: &str,
    chosen: &[&(usize, usize, AnchorReplacement)],
    lexicon: &dyn RelationLexicon,
    evidence: &dyn AnchorEvidence,
    answer_noun: Option<&str>,
) -> Option<AnchoredQuestion> {
    let planner = QueryPlanner::new();
    let planner_plan = planner.parse(question);
    let rewrite = query_rewrite::rewrite(question, lexicon);
    if planner_plan.is_none() && rewrite.is_none() {
        return None;
    }

    let mut evidence_notes: Vec<String> = Vec::new();
    let mut score = chosen
        .iter()
        .map(|(_, _, replacement)| replacement.score)
        .sum::<f64>()
        / chosen.len() as f64;
    score -= 0.03 * (chosen.len().saturating_sub(1)) as f64;
    for (_, _, replacement) in chosen {
        if replacement.method == LinkMethod::Fuzzy {
            score -= 0.02;
        }
    }

    let mut kept_planner = planner_plan;
    let mut kept_rewrite = rewrite;
    let mut planner_ok = true;
    if let Some(plan) = kept_planner.as_ref() {
        match check_plan_types(plan, evidence, answer_noun) {
            TypeCheck::Rejected(notes) => {
                evidence_notes.extend(notes);
                planner_ok = false;
            }
            TypeCheck::Accepted { bonus, notes } => {
                score += bonus;
                evidence_notes.extend(notes);
            }
        }
    }
    let mut rewrite_ok = true;
    if let Some(rewrite) = kept_rewrite.as_ref() {
        match check_plan_types(&rewrite.plan, evidence, answer_noun) {
            TypeCheck::Rejected(notes) => {
                evidence_notes.extend(notes);
                rewrite_ok = false;
            }
            TypeCheck::Accepted { bonus, notes } => {
                score += bonus * 0.5;
                evidence_notes.extend(notes);
            }
        }
    }
    if !planner_ok {
        kept_planner = None;
    }
    if !rewrite_ok {
        kept_rewrite = None;
    }
    if kept_planner.is_none() && kept_rewrite.is_none() {
        return None;
    }

    for (_, _, replacement) in chosen {
        evidence_notes.push(format!(
            "{}（{}）",
            replacement.method.label(),
            replacement
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| format!("置信度 {:.2}", replacement.score))
        ));
    }

    Some(AnchoredQuestion {
        question: question.to_string(),
        replacements: chosen
            .iter()
            .map(|(_, _, replacement)| replacement.clone())
            .collect(),
        planner_plan: kept_planner,
        rewrite: kept_rewrite,
        score: score.max(0.0),
        evidence: evidence_notes,
    })
}

enum TypeCheck {
    Accepted { bonus: f64, notes: Vec<String> },
    Rejected(Vec<String>),
}

fn check_plan_types(
    plan: &QueryPlan,
    evidence: &dyn AnchorEvidence,
    answer_noun: Option<&str>,
) -> TypeCheck {
    let mut notes: Vec<String> = Vec::new();
    let mut bonus = 0.0;
    // 第四项标记 actual 是「实体名」还是「答案类型名词」；答案名词直接
    // 作为类型参与值域/定义域核对，不能再去实体画像里查同名实体。
    let mut checks: Vec<(&str, &str, &str, bool)> = Vec::new();

    match plan {
        QueryPlan::FindObject { segments } if segments.len() >= 2 => {
            checks.push((&segments[0], &segments[1], "定义域", false));
            if let Some(noun) = answer_noun {
                checks.push((
                    noun,
                    segments.last().map(String::as_str).unwrap_or(""),
                    "值域",
                    true,
                ));
            }
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => {
            checks.push((subject, predicate, "定义域", false));
            if let Some(noun) = answer_noun {
                checks.push((noun, predicate, "值域", true));
            }
        }
        QueryPlan::FindSubject { object, predicate } => {
            checks.push((object, predicate, "值域", false));
            if let Some(noun) = answer_noun {
                checks.push((noun, predicate, "定义域", true));
            }
        }
        QueryPlan::FindObjectExcept {
            subject, predicate, ..
        } => {
            checks.push((subject, predicate, "定义域", false));
            if let Some(noun) = answer_noun {
                checks.push((noun, predicate, "值域", true));
            }
        }
        QueryPlan::FindSubjectExcept {
            object, predicate, ..
        } => {
            checks.push((object, predicate, "值域", false));
            if let Some(noun) = answer_noun {
                checks.push((noun, predicate, "定义域", true));
            }
        }
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            ..
        } => {
            checks.push((subject, predicate, "定义域", false));
            checks.push((object, predicate, "值域", false));
        }
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } => {
            if let (Some(subject), Some(predicate)) =
                (subject_segments.first(), subject_segments.last())
            {
                if subject_segments.len() >= 2 {
                    checks.push((subject, predicate, "定义域", false));
                    checks.push((object, predicate, "值域", false));
                }
            }
        }
        QueryPlan::Count { subject, predicate }
        | QueryPlan::Exist { subject, predicate }
        | QueryPlan::Closure {
            subject, predicate, ..
        } => {
            checks.push((subject, predicate, "定义域", false));
        }
        _ => {}
    }

    for (actual, predicate, slot, actual_is_type) in checks {
        let actual_types = if actual_is_type {
            vec![actual.to_string()]
        } else {
            evidence.entity_types(actual)
        };
        let expected = if slot == "定义域" {
            evidence.domain_of(predicate)
        } else {
            evidence.range_of(predicate)
        };
        if actual_types.is_empty() || expected.is_empty() {
            continue;
        }
        let mut compatible = false;
        let mut disjoint = false;
        for actual in &actual_types {
            for expected in &expected {
                match evidence.type_fit(actual, expected) {
                    TypeFit::Compatible => compatible = true,
                    TypeFit::Disjoint => disjoint = true,
                    TypeFit::Unknown => {}
                }
            }
        }
        if compatible {
            bonus += 0.03;
            notes.push(format!(
                "{}「{}」∈ {} {}",
                slot,
                actual_types.join("/"),
                predicate,
                expected.join("/")
            ));
        } else if disjoint {
            return TypeCheck::Rejected(vec![format!(
                "{}闸门拒绝：{}「{}」类型 {} 与谓词「{}」{} {} 互斥",
                slot,
                if actual_is_type {
                    "答案类型"
                } else {
                    "实体"
                },
                actual,
                actual_types.join("/"),
                predicate,
                slot,
                expected.join("/")
            )]);
        }
    }

    TypeCheck::Accepted { bonus, notes }
}

/// 从问句里提取答案类型名词：「哪些学生」→「学生」；
/// 「张三教过的学生是谁」→「学生」。只做保守边界切分，不要求本体登记。
pub fn answer_type_noun(question: &str) -> Option<String> {
    const HEADS: [&str; 8] = ["哪些", "哪个", "哪位", "哪家", "哪门", "哪几", "什么", "啥"];
    const TAILS: [&str; 10] = [
        "分别是哪些人",
        "分别是哪些",
        "分别是谁",
        "都有哪些",
        "都有谁",
        "有哪些人",
        "有哪些",
        "是哪些人",
        "是哪些",
        "是谁",
    ];
    let text = question
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' ']);

    let mut best: Option<String> = None;
    for head in HEADS {
        let mut offset = 0;
        while let Some(position) = text[offset..].find(head) {
            let position = offset + position;
            let rest = &text[position + head.len()..];
            let noun: String = rest
                .chars()
                .take_while(|ch| valid_type_char(*ch))
                .take(10)
                .collect();
            if !noun.is_empty() && !is_wh_noun(&noun) {
                if best.as_ref().map_or(true, |current| {
                    noun.chars().count() > current.chars().count()
                }) {
                    best = Some(noun);
                }
            }
            offset = position + head.len();
        }
    }
    if best.is_some() {
        return best;
    }
    if let Some(core) = TAILS.iter().find_map(|tail| text.strip_suffix(tail)) {
        let core = core.trim();
        if let Some((head, noun)) = core.rsplit_once('的') {
            let noun = noun.trim();
            // 只有关系从句才把「的」后的名词当答案类型；「张三的师傅是谁」
            // 里的「师傅」是谓词，不是类型名词。要求从句核心带体标记/被动
            // 标记，宁可少判也不把谓词误当答案类型。
            let relative_clause = head
                .chars()
                .any(|ch| matches!(ch, '过' | '了' | '着' | '被' | '把' | '将'));
            if relative_clause
                && !noun.is_empty()
                && noun.chars().all(valid_type_char)
                && !is_wh_noun(noun)
            {
                return Some(noun.to_string());
            }
        }
    }
    None
}

fn valid_type_char(ch: char) -> bool {
    !matches!(
        ch,
        '的' | '是'
            | '被'
            | '把'
            | '将'
            | '吗'
            | '呢'
            | '吧'
            | '啊'
            | '过'
            | '了'
            | '着'
            | '不'
            | '没'
            | '和'
            | '跟'
            | '与'
            | '还'
            | '也'
            | '都'
            | '，'
            | ','
            | '？'
            | '?'
            | '。'
            | '！'
            | '!'
            | ' '
    )
}

fn is_wh_noun(noun: &str) -> bool {
    ["谁", "什么", "哪些", "哪个", "多少", "几"]
        .iter()
        .any(|word| noun.contains(word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_linker::EntityLinker;
    use std::collections::HashMap;

    struct FakeLexicon {
        entities: HashMap<String, String>,
        predicates: HashMap<String, String>,
    }

    impl FakeLexicon {
        fn new() -> Self {
            let entities = ["张三", "李四", "王五", "张珊", "苹果", "苹果公司"]
                .iter()
                .map(|name| (name.to_string(), name.to_string()))
                .collect();
            let predicates = ["师傅", "认识", "出生地", "教"]
                .iter()
                .map(|name| (name.to_string(), name.to_string()))
                .collect();
            Self {
                entities,
                predicates,
            }
        }
    }

    impl RelationLexicon for FakeLexicon {
        fn canonical_entity(&self, raw: &str) -> Option<String> {
            self.entities.get(raw.trim()).cloned()
        }

        fn canonical_predicate(&self, raw: &str) -> Option<String> {
            self.predicates.get(raw.trim()).cloned()
        }
    }

    struct FakeEvidence {
        types: HashMap<String, Vec<String>>,
        domains: HashMap<String, Vec<String>>,
        ranges: HashMap<String, Vec<String>>,
    }

    impl FakeEvidence {
        fn new() -> Self {
            let types = [
                ("张三", vec!["人物"]),
                ("李四", vec!["人物"]),
                ("王五", vec!["人物"]),
                ("张珊", vec!["人物"]),
                ("苹果", vec!["水果"]),
                ("苹果公司", vec!["组织"]),
            ]
            .into_iter()
            .map(|(name, types)| {
                (
                    name.to_string(),
                    types.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();
            let domains = [
                ("师傅", vec!["人物"]),
                ("认识", vec!["人物"]),
                ("出生地", vec!["人物"]),
            ]
            .into_iter()
            .map(|(name, types)| {
                (
                    name.to_string(),
                    types.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();
            let ranges = [
                ("师傅", vec!["人物"]),
                ("认识", vec!["人物"]),
                ("出生地", vec!["地点"]),
            ]
            .into_iter()
            .map(|(name, types)| {
                (
                    name.to_string(),
                    types.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();
            Self {
                types,
                domains,
                ranges,
            }
        }
    }

    impl AnchorEvidence for FakeEvidence {
        fn entity_types(&self, entity: &str) -> Vec<String> {
            self.types.get(entity).cloned().unwrap_or_default()
        }

        fn domain_of(&self, predicate: &str) -> Vec<String> {
            self.domains.get(predicate).cloned().unwrap_or_default()
        }

        fn range_of(&self, predicate: &str) -> Vec<String> {
            self.ranges.get(predicate).cloned().unwrap_or_default()
        }

        fn type_fit(&self, actual: &str, expected: &str) -> TypeFit {
            let disjoint = matches!(
                (actual, expected),
                ("人物", "地点") | ("地点", "人物") | ("水果", "人物") | ("人物", "水果")
            );
            if disjoint {
                TypeFit::Disjoint
            } else {
                default_type_fit(actual, expected)
            }
        }
    }

    fn linker() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三", Some("人物"));
        linker.register_entity("李四", Some("人物"));
        linker.register_entity("王五", Some("人物"));
        linker.register_entity("苹果", Some("水果"));
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("小明", "张三");
        linker.observe_triple("张三", "师傅", "李四");
        linker.observe_triple("李四", "师傅", "王五");
        linker
    }

    fn anchor(question: &str, evidence: &dyn AnchorEvidence) -> Vec<AnchoredQuestion> {
        let linker = linker();
        let hints = vec!["师傅".to_string(), "认识".to_string()];
        anchor_questions(question, &linker, &hints, &FakeLexicon::new(), evidence)
    }

    #[test]
    fn fuzzy_typo_is_rewritten_to_canonical_entity() {
        let variants = anchor("张山的师傅是谁", &FakeEvidence::new());
        let best = variants.first().expect("错别字应产生锚定变体");
        assert_eq!(best.question, "张三的师傅是谁");
        assert_eq!(best.replacements.len(), 1);
        assert_eq!(best.replacements[0].surface, "张山");
        assert_eq!(best.replacements[0].canonical, "张三");
        assert_eq!(
            best.planner_plan,
            Some(QueryPlan::FindObject {
                segments: vec!["张三".to_string(), "师傅".to_string()]
            })
        );
        assert!(best.score >= ANCHOR_MIN_SCORE);
        assert!(best.receipt().contains("张山"));
        assert!(best.receipt().contains("张三"));
    }

    #[test]
    fn exact_question_produces_no_variant() {
        assert!(anchor("张三的师傅是谁", &FakeEvidence::new()).is_empty());
    }

    #[test]
    fn alias_is_anchored() {
        let variants = anchor("小明的师傅是谁", &FakeEvidence::new());
        let best = variants.first().expect("别名应产生锚定变体");
        assert_eq!(best.question, "张三的师傅是谁");
        assert_eq!(best.replacements[0].method, LinkMethod::Alias);
    }

    #[test]
    fn ambiguous_mention_abstains() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三", Some("人物"));
        linker.register_entity("张珊", Some("人物"));
        linker.observe_triple("张三", "师傅", "王五");
        linker.observe_triple("张珊", "师傅", "王五");
        let variants = anchor_questions(
            "张山的师傅是谁",
            &linker,
            &[],
            &FakeLexicon::new(),
            &FakeEvidence::new(),
        );
        assert!(variants.is_empty(), "边际不足时不得硬猜：{:?}", variants);
    }

    #[test]
    fn disjoint_subject_type_rejects_variant() {
        struct PlaceDomain;
        impl AnchorEvidence for PlaceDomain {
            fn entity_types(&self, entity: &str) -> Vec<String> {
                FakeEvidence::new().entity_types(entity)
            }
            fn domain_of(&self, _predicate: &str) -> Vec<String> {
                vec!["地点".to_string()]
            }
            fn range_of(&self, _predicate: &str) -> Vec<String> {
                Vec::new()
            }
            fn type_fit(&self, actual: &str, expected: &str) -> TypeFit {
                if actual == expected {
                    TypeFit::Compatible
                } else {
                    TypeFit::Disjoint
                }
            }
        }
        assert!(anchor("张山的师傅是谁", &PlaceDomain).is_empty());
    }

    #[test]
    fn answer_type_conflict_rejects_variant() {
        let variants = anchor("张山认识哪些地点", &FakeEvidence::new());
        assert!(
            variants.is_empty(),
            "答案类型与谓词值域互斥时应弃权：{:?}",
            variants
        );
    }

    #[test]
    fn answer_type_noun_extraction_is_conservative() {
        assert_eq!(answer_type_noun("张山教过哪些学生"), Some("学生".into()));
        assert_eq!(answer_type_noun("张三教过的学生是谁"), Some("学生".into()));
        assert_eq!(answer_type_noun("张山的师傅是谁"), None);
        assert_eq!(answer_type_noun("什么是知识图谱"), None);
    }

    #[test]
    fn two_mentions_are_replaced_together() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三", Some("人物"));
        linker.register_entity("李四", Some("人物"));
        linker.observe_triple("张三", "认识", "李四");
        let variants = anchor_questions(
            "张山认识李寺吗",
            &linker,
            &[],
            &FakeLexicon::new(),
            &FakeEvidence::new(),
        );
        let combined = variants
            .iter()
            .find(|variant| variant.replacements.len() == 2)
            .expect("两个错别字应有一个联合替换变体");
        assert_eq!(combined.question, "张三认识李四吗");
    }
}

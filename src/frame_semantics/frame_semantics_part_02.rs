/// 一次角色更正：旧值、新值、步骤与所在帧。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleCorrection {
    pub role: Role,
    pub old: Option<String>,
    pub new: String,
    pub step: usize,
    pub frame_id: String,
}

/// 追问解析结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FollowupKind {
    Answer { role: Role, value: String, frame_id: String },
    Correction { role: Role, old: Option<String>, new: String },
    Clarify { role: Role },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FollowupResolution {
    pub kind: FollowupKind,
    pub focus: Option<Role>,
}

impl FollowupResolution {
    pub fn render_zh(&self) -> String {
        match &self.kind {
            FollowupKind::Answer { role, value, .. } => {
                format!("{}：{}", role.label_zh(), value)
            }
            FollowupKind::Correction { role, old, new } => match old {
                Some(old) => format!("已把{}从「{}」更正为「{}」", role.label_zh(), old, new),
                None => format!("已把{}补为「{}」", role.label_zh(), new),
            },
            FollowupKind::Clarify { role } => {
                format!("当前帧没有{}，请补充。", role.label_zh())
            }
            FollowupKind::Unknown => "没有识别出角色追问。".to_string(),
        }
    }
}

/// 会话级帧记忆：焦点角色、省略继承与显式更正。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameLedger {
    #[serde(default)]
    frames: Vec<FrameInstance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    focus: Option<Role>,
    #[serde(default)]
    corrections: Vec<RoleCorrection>,
}

impl FrameLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn current(&self) -> Option<&FrameInstance> {
        self.frames.last()
    }

    pub fn current_mut(&mut self) -> Option<&mut FrameInstance> {
        self.frames.last_mut()
    }

    pub fn frames(&self) -> &[FrameInstance] {
        &self.frames
    }

    pub fn focus(&self) -> Option<Role> {
        self.focus
    }

    pub fn focus_role(&mut self, role: Role) {
        self.focus = Some(role);
    }

    pub fn corrections(&self) -> &[RoleCorrection] {
        &self.corrections
    }

    pub fn observe(&mut self, frame: FrameInstance) -> String {
        let id = frame.id.clone();
        if self.focus.is_none() {
            self.focus = frame
                .explicit_roles()
                .into_iter()
                .next()
                .or_else(|| frame.roles.keys().next().copied());
        }
        self.frames.push(frame);
        if self.frames.len() > 64 {
            self.frames.remove(0);
        }
        id
    }

    pub fn answer(&self, role: Role) -> Option<&str> {
        self.current()?.usable(role)
    }

    fn role_by_value(&self, value: &str) -> Option<Role> {
        let current = self.current()?;
        current
            .roles
            .iter()
            .find(|(_, filler)| filler.is_usable() && filler.value == value)
            .map(|(role, _)| *role)
    }

    fn question_role(text: &str) -> Option<Role> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        let table: &[(&[&str], Role)] = &[
            (&["给谁", "交给谁", "接受者"], Role::Recipient),
            (&["为谁", "受益"], Role::Beneficiary),
            (&["为什么", "原因", "怎么会", "为何"], Role::Cause),
            (&["为了什么", "目的", "干嘛", "做什么"], Role::Purpose),
            (&["在哪", "哪里", "哪儿", "地点", "什么地方"], Role::Location),
            (&["什么时候", "何时", "时间", "哪天"], Role::Time),
            (&["用什么", "拿什么", "工具", "怎么修", "怎么去"], Role::Instrument),
            (&["从哪", "来源", "从哪里"], Role::Source),
            (&["到哪", "去哪", "终点", "到哪里"], Role::Goal),
            (&["结果"], Role::Result),
            (&["谁", "谁修", "谁去", "谁做"], Role::Agent),
        ];
        for (phrases, role) in table {
            if phrases.iter().any(|phrase| text.contains(phrase)) {
                return Some(*role);
            }
        }
        None
    }

    fn parse_correction(text: &str) -> Option<(String, String)> {
        let body = text
            .trim()
            .trim_end_matches(['。', '！', '!', '？', '?'])
            .trim();
        let rest = body
            .strip_prefix("不是")
            .or_else(|| body.strip_prefix("不对，"))
            .or_else(|| body.strip_prefix("不对,"))?;
        let split = rest.find('是')?;
        let old = trim_value(&rest[..split]).to_string();
        let new = trim_value(&rest[split + '是'.len_utf8()..]).to_string();
        if old.is_empty() || new.is_empty() {
            return None;
        }
        Some((old, new))
    }

    pub fn resolve_followup(&mut self, text: &str) -> FollowupResolution {
        if let Some((old, new)) = Self::parse_correction(text) {
            let role = self
                .role_by_value(&old)
                .or(self.focus)
                .or_else(|| Self::question_role(text));
            if let Some(role) = role {
                self.correct(role, &new);
                return FollowupResolution {
                    kind: FollowupKind::Correction {
                        role,
                        old: Some(old),
                        new,
                    },
                    focus: self.focus,
                };
            }
        }

        if let Some(role) = Self::question_role(text) {
            self.focus = Some(role);
            if let Some(value) = self.answer(role).map(str::to_string) {
                let frame_id = self
                    .current()
                    .map(|frame| frame.id.clone())
                    .unwrap_or_default();
                return FollowupResolution {
                    kind: FollowupKind::Answer {
                        role,
                        value,
                        frame_id,
                    },
                    focus: self.focus,
                };
            }
            return FollowupResolution {
                kind: FollowupKind::Clarify { role },
                focus: self.focus,
            };
        }

        FollowupResolution {
            kind: FollowupKind::Unknown,
            focus: self.focus,
        }
    }

    pub fn correct(&mut self, role: Role, new_value: &str) -> Option<RoleCorrection> {
        let new_value = trim_value(new_value).to_string();
        if new_value.is_empty() {
            return None;
        }
        let step = self.corrections.len() + 1;
        let frame = self.frames.last_mut()?;
        let old = frame.usable(role).map(str::to_string);
        let filler = RoleFiller {
            value: new_value.clone(),
            span: None,
            marker: role.default_marker().map(str::to_string),
            status: FillerStatus::Explicit,
            confidence: 0.95,
            source: "用户更正".to_string(),
        };
        frame.set_role(role, filler);
        let frame_id = frame.id.clone();
        let correction = RoleCorrection {
            role,
            old,
            new: new_value,
            step,
            frame_id,
        };
        self.corrections.push(correction.clone());
        self.focus = Some(role);
        Some(correction)
    }

    /// 把上一帧的指定角色继承到当前帧；只填空缺，不覆盖显式值。
    pub fn inherit_missing(&mut self, roles: &[Role]) -> usize {
        if self.frames.len() < 2 {
            return 0;
        }
        let last = self.frames.len() - 1;
        let (previous, current) = self.frames.split_at_mut(last);
        let previous = &previous[previous.len() - 1];
        let current = &mut current[0];
        let mut inherited = 0;
        for role in roles {
            if current.filler(*role).is_some() {
                continue;
            }
            if let Some(filler) = previous.filler(*role) {
                current.set_role(*role, RoleFiller::inherited(filler.value.clone(), &previous.id));
                inherited += 1;
            }
        }
        inherited
    }

    pub fn render_state(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "帧台账：{} 帧，焦点 {}，更正 {} 次",
            self.frames.len(),
            self.focus.map(Role::label_zh).unwrap_or("无"),
            self.corrections.len()
        );
        if let Some(frame) = self.current() {
            let _ = writeln!(output, "  当前：{}", frame.render_roles());
        }
        for correction in self.corrections.iter().rev().take(5) {
            let _ = writeln!(
                output,
                "  更正#{} {}：{} → {}",
                correction.step,
                correction.role.label_zh(),
                correction.old.as_deref().unwrap_or("无"),
                correction.new
            );
        }
        output
    }
}

/// 规则编译的一个标注正例。`types` 可选，用于归纳选择限制。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameExample {
    pub text: String,
    #[serde(default)]
    pub roles: BTreeMap<Role, String>,
    #[serde(default)]
    pub types: BTreeMap<Role, BTreeSet<String>>,
}

impl FrameExample {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            roles: BTreeMap::new(),
            types: BTreeMap::new(),
        }
    }

    pub fn with_role(mut self, role: Role, value: impl Into<String>) -> Self {
        self.roles.insert(role, value.into());
        self
    }

    pub fn with_type(mut self, role: Role, type_name: impl Into<String>) -> Self {
        self.types.entry(role).or_default().insert(type_name.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompileError {
    EmptyExamples,
    MissingRole { role: Role, text: String },
    AmbiguousOccurrence { role: Role, value: String, text: String },
    InconsistentSkeleton { first: String, second: String },
    PositiveMismatch { text: String, expected: String, actual: String },
    NegativeHit { text: String },
}

impl CompileError {
    pub fn render(&self) -> String {
        match self {
            CompileError::EmptyExamples => "没有正例，无法编译 schema".to_string(),
            CompileError::MissingRole { role, text } => {
                format!("正例「{}」里找不到角色「{}」的值", text, role.label_zh())
            }
            CompileError::AmbiguousOccurrence { role, value, text } => format!(
                "正例「{}」里「{}」出现多次，无法唯一定位角色「{}」",
                text,
                value,
                role.label_zh()
            ),
            CompileError::InconsistentSkeleton { first, second } => {
                format!("正例骨架不一致：{} ≠ {}", first, second)
            }
            CompileError::PositiveMismatch {
                text,
                expected,
                actual,
            } => format!(
                "回译校验失败：正例「{}」期望 {}，实际 {}",
                text, expected, actual
            ),
            CompileError::NegativeHit { text } => {
                format!("反例「{}」被 schema 命中，拒绝激活", text)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileOptions {
    pub min_support: usize,
    pub require_negative_clear: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            min_support: 1,
            require_negative_clear: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeResult {
    pub text: String,
    pub expected: bool,
    pub matched: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledRule {
    pub schema: FrameSchema,
    pub positive_hits: usize,
    pub negative_hits: usize,
    pub probes: Vec<ProbeResult>,
    pub diagnostics: Vec<String>,
}

/// 从标注例子反合一出 schema：字面锚点 + 角色槽 + 格标记。
pub struct FrameRuleCompiler;

impl FrameRuleCompiler {
    pub fn compile(
        predicate: &str,
        sense: &str,
        examples: &[FrameExample],
        negatives: &[String],
        options: &CompileOptions,
    ) -> Result<CompiledRule, CompileError> {
        if examples.is_empty() {
            return Err(CompileError::EmptyExamples);
        }

        let mut skeletons = Vec::with_capacity(examples.len());
        for example in examples {
            skeletons.push(skeleton_for(example)?);
        }
        let template = skeletons[0].clone();
        for skeleton in skeletons.iter().skip(1) {
            if skeleton != &template {
                return Err(CompileError::InconsistentSkeleton {
                    first: template,
                    second: skeleton.clone(),
                });
            }
        }

        let mut schema = FrameSchema::new(predicate, sense);
        schema.roles.clear();
        for role in placeholder_roles(&template) {
            let mut spec = RoleSpec::new(role).required();
            let markers = marker_before(&template, role);
            if !markers.is_empty() {
                spec = spec.with_markers(markers);
            }
            if let Some(allowed) = common_types(examples, role) {
                spec = spec.with_types(allowed);
            }
            schema.roles.push(spec);
        }
        schema.roles.sort_by_key(|spec| spec.role);
        schema.realize = Some(template.clone());
        schema.support = examples.len();
        schema.positives = examples.iter().map(|example| example.text.clone()).collect();
        schema.negatives = negatives.to_vec();
        schema.status = if examples.len() >= options.min_support {
            SchemaStatus::Active
        } else {
            SchemaStatus::Candidate
        };
        schema.refresh_id();

        let mut probes = Vec::new();
        let mut positive_hits = 0;
        for example in examples {
            let recovered = match_template(&template, &example.text);
            let matched = recovered
                .as_ref()
                .map(|roles| roles_match(roles, &example.roles))
                .unwrap_or(false);
            if matched {
                positive_hits += 1;
            } else {
                let actual = recovered
                    .map(|roles| render_roles_map(&roles))
                    .unwrap_or_else(|| "未命中".to_string());
                return Err(CompileError::PositiveMismatch {
                    text: example.text.clone(),
                    expected: render_roles_map(&example.roles),
                    actual,
                });
            }
            probes.push(ProbeResult {
                text: example.text.clone(),
                expected: true,
                matched,
            });
        }

        let mut negative_hits = 0;
        for negative in negatives {
            let matched = match_template(&template, negative).is_some();
            if matched {
                negative_hits += 1;
            }
            probes.push(ProbeResult {
                text: negative.clone(),
                expected: false,
                matched,
            });
        }
        if options.require_negative_clear && negative_hits > 0 {
            return Err(CompileError::NegativeHit {
                text: probes
                    .iter()
                    .find(|probe| !probe.expected && probe.matched)
                    .map(|probe| probe.text.clone())
                    .unwrap_or_default(),
            });
        }

        let diagnostics = vec![
            format!("模板：{}", template),
            format!("角色：{}", schema.roles.iter().map(|spec| spec.role.label_zh()).collect::<Vec<_>>().join("、")),
            format!("正例命中 {}/{}", positive_hits, examples.len()),
            format!("反例命中 {}", negative_hits),
        ];

        Ok(CompiledRule {
            schema,
            positive_hits,
            negative_hits,
            probes,
            diagnostics,
        })
    }

    pub fn compile_into(
        book: &mut FrameBook,
        predicate: &str,
        sense: &str,
        examples: &[FrameExample],
        negatives: &[String],
        options: &CompileOptions,
    ) -> Result<CompiledRule, CompileError> {
        let rule = Self::compile(predicate, sense, examples, negatives, options)?;
        book.register(rule.schema.clone());
        Ok(rule)
    }
}

fn skeleton_for(example: &FrameExample) -> Result<String, CompileError> {
    let mut occurrences: Vec<(usize, usize, Role)> = Vec::new();
    for (role, value) in &example.roles {
        let value = value.trim();
        if value.is_empty() {
            return Err(CompileError::MissingRole {
                role: *role,
                text: example.text.clone(),
            });
        }
        let matches: Vec<usize> = example
            .text
            .match_indices(value)
            .map(|(start, _)| start)
            .collect();
        if matches.is_empty() {
            return Err(CompileError::MissingRole {
                role: *role,
                text: example.text.clone(),
            });
        }
        if matches.len() > 1 {
            return Err(CompileError::AmbiguousOccurrence {
                role: *role,
                value: value.to_string(),
                text: example.text.clone(),
            });
        }
        let start = matches[0];
        occurrences.push((start, start + value.len(), *role));
    }
    occurrences.sort_by_key(|(start, end, _)| (*start, *end));
    for pair in occurrences.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(CompileError::AmbiguousOccurrence {
                role: pair[0].2,
                value: example.roles.get(&pair[0].2).cloned().unwrap_or_default(),
                text: example.text.clone(),
            });
        }
    }
    let mut skeleton = example.text.clone();
    for (start, end, role) in occurrences.iter().rev() {
        skeleton.replace_range(*start..*end, &format!("{{{}}}", role.as_str()));
    }
    Ok(skeleton)
}

fn placeholder_roles(template: &str) -> Vec<Role> {
    let mut roles = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open + 1..].find('}') else {
            break;
        };
        let token = &rest[open + 1..open + 1 + close];
        let name = token.split(':').next().unwrap_or(token);
        if let Some(role) = Role::parse(name) {
            if !roles.contains(&role) {
                roles.push(role);
            }
        }
        rest = &rest[open + 1 + close + 1..];
    }
    roles
}

fn marker_before(template: &str, role: Role) -> Vec<String> {
    let token = format!("{{{}}}", role.as_str());
    let Some(position) = template.find(&token) else {
        return Vec::new();
    };
    let before = &template[..position];
    let mut markers = Vec::new();
    for (marker, marker_role) in MARKERS {
        if before.ends_with(marker) && (marker_role == &role || role == Role::Agent) {
            markers.push((*marker).to_string());
        }
    }
    for marker in PASSIVE_MARKERS {
        if before.ends_with(marker) && role == Role::Agent {
            markers.push((*marker).to_string());
        }
    }
    markers.sort_by_key(|marker| std::cmp::Reverse(marker.len()));
    markers.dedup();
    markers
}

fn common_types(examples: &[FrameExample], role: Role) -> Option<BTreeSet<String>> {
    let mut common: Option<BTreeSet<String>> = None;
    for example in examples {
        let types = example.types.get(&role)?;
        if types.is_empty() {
            return None;
        }
        common = Some(match common {
            None => types.clone(),
            Some(existing) => existing.intersection(types).cloned().collect(),
        });
    }
    common.filter(|types| !types.is_empty())
}

fn roles_match(actual: &BTreeMap<Role, String>, expected: &BTreeMap<Role, String>) -> bool {
    expected.iter().all(|(role, value)| {
        actual
            .get(role)
            .map(|actual| actual.trim() == value.trim())
            .unwrap_or(false)
    })
}

fn render_roles_map(roles: &BTreeMap<Role, String>) -> String {
    roles
        .iter()
        .map(|(role, value)| format!("{}={}", role.label_zh(), value))
        .collect::<Vec<_>>()
        .join("，")
}

/// 把模板当作「字面锚点 + 角色槽」匹配文本；空槽不算命中。
pub fn match_template(template: &str, text: &str) -> Option<BTreeMap<Role, String>> {
    let parts = template_parts(template);
    let mut cursor = 0usize;
    let mut roles = BTreeMap::new();
    for (index, part) in parts.iter().enumerate() {
        match part {
            TemplatePart::Literal(literal) => {
                if literal.is_empty() {
                    continue;
                }
                if !text[cursor..].starts_with(literal.as_str()) {
                    return None;
                }
                cursor += literal.len();
            }
            TemplatePart::Slot(role) => {
                let next_literal = parts[index + 1..]
                    .iter()
                    .find_map(|part| match part {
                        TemplatePart::Literal(literal) if !literal.is_empty() => {
                            Some(literal.as_str())
                        }
                        _ => None,
                    });
                let (value, next_cursor) = match next_literal {
                    Some(literal) => {
                        let offset = text[cursor..].find(literal)?;
                        (&text[cursor..cursor + offset], cursor + offset)
                    }
                    None => (&text[cursor..], text.len()),
                };
                let value = trim_value(value).to_string();
                if value.is_empty() {
                    return None;
                }
                roles.insert(*role, value);
                cursor = next_cursor;
            }
        }
    }
    if cursor == text.len() {
        Some(roles)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplatePart {
    Literal(String),
    Slot(Role),
}

fn template_parts(template: &str) -> Vec<TemplatePart> {
    let mut parts = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        if open > 0 {
            parts.push(TemplatePart::Literal(rest[..open].to_string()));
        }
        let Some(close) = rest[open + 1..].find('}') else {
            parts.push(TemplatePart::Literal(rest[open..].to_string()));
            return parts;
        };
        let token = &rest[open + 1..open + 1 + close];
        let name = token.split(':').next().unwrap_or(token);
        if let Some(role) = Role::parse(name) {
            parts.push(TemplatePart::Slot(role));
        } else {
            parts.push(TemplatePart::Literal(rest[open..open + close + 2].to_string()));
        }
        rest = &rest[open + close + 2..];
    }
    if !rest.is_empty() {
        parts.push(TemplatePart::Literal(rest.to_string()));
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> FrameBook {
        FrameBook::with_core_schemas()
    }

    fn repair_frame(text: &str) -> FrameInstance {
        let book = book();
        let annotator = FrameAnnotator::new(&book);
        let frames = annotator.annotate(text);
        assert_eq!(frames.len(), 1, "{}", text);
        frames.into_iter().next().unwrap()
    }

    #[test]
    fn roles_have_stable_labels() {
        assert_eq!(Role::parse("agent"), Some(Role::Agent));
        assert_eq!(Role::parse("施事"), Some(Role::Agent));
        assert_eq!(Role::Agent.label_zh(), "施事");
        assert_eq!(Role::Instrument.default_marker(), Some("用"));
        assert!(Role::Patient.is_projected_object());
    }

    #[test]
    fn annotates_oblique_roles_and_projects() {
        let frame = repair_frame("张三昨天在北京用扳手修好了李四的电脑");
        assert_eq!(frame.usable(Role::Agent), Some("张三"));
        assert_eq!(frame.usable(Role::Patient), Some("李四的电脑"));
        assert_eq!(frame.usable(Role::Instrument), Some("扳手"));
        assert_eq!(frame.usable(Role::Location), Some("北京"));
        assert_eq!(frame.usable(Role::Time), Some("昨天"));

        let projected = frame.project().expect("projected");
        assert_eq!(projected.subject, "张三");
        assert_eq!(projected.predicate, "修");
        assert_eq!(projected.object, "李四的电脑");
        assert!(projected.qualifiers.contains(&"工具=扳手".to_string()));
        assert!(projected.qualifiers.contains(&"地点=北京".to_string()));
        assert!(projected.qualifiers.contains(&"时间=昨天".to_string()));
    }

    #[test]
    fn realizes_and_round_trips() {
        let book = book();
        let frame = repair_frame("张三昨天在北京用扳手修好了李四的电脑");
        let text = book.realize(&frame).expect("realized");
        assert_eq!(text, "张三昨天在北京用扳手修好了李四的电脑");
        let recovered = repair_frame(&text);
        for role in [Role::Agent, Role::Patient, Role::Instrument, Role::Location, Role::Time] {
            assert_eq!(recovered.usable(role), frame.usable(role), "{role:?}");
        }
    }

    #[test]
    fn handles_passive_and_ba() {
        let passive = repair_frame("李四的电脑被张三修好了");
        assert_eq!(passive.usable(Role::Agent), Some("张三"));
        assert_eq!(passive.usable(Role::Patient), Some("李四的电脑"));

        let ba = repair_frame("张三把电脑修好了");
        assert_eq!(ba.usable(Role::Agent), Some("张三"));
        assert_eq!(ba.usable(Role::Patient), Some("电脑"));
    }

    #[test]
    fn annotates_unseen_predicate_by_markers() {
        let book = book();
        let annotator = FrameAnnotator::new(&book);
        let frame = annotator
            .annotate_predicate("王五用锤子敲钉子", "敲")
            .expect("frame");
        assert_eq!(frame.usable(Role::Agent), Some("王五"));
        assert_eq!(frame.usable(Role::Instrument), Some("锤子"));
        assert_eq!(frame.usable(Role::Patient), Some("钉子"));
    }

    #[test]
    fn ledger_answers_focus_and_corrects_roles() {
        let mut ledger = FrameLedger::new();
        ledger.observe(repair_frame("张三在北京用扳手修好了电脑"));

        let answer = ledger.resolve_followup("那地点呢？");
        assert_eq!(
            answer.kind,
            FollowupKind::Answer {
                role: Role::Location,
                value: "北京".to_string(),
                frame_id: ledger.current().unwrap().id.clone(),
            }
        );

        let clarify = ledger.resolve_followup("为什么？");
        assert_eq!(clarify.kind, FollowupKind::Clarify { role: Role::Cause });

        let correction = ledger.resolve_followup("不是北京，是上海");
        assert_eq!(
            correction.kind,
            FollowupKind::Correction {
                role: Role::Location,
                old: Some("北京".to_string()),
                new: "上海".to_string(),
            }
        );
        assert_eq!(ledger.answer(Role::Location), Some("上海"));
        assert_eq!(ledger.corrections().len(), 1);
    }

    #[test]
    fn ledger_inherits_missing_roles() {
        let mut ledger = FrameLedger::new();
        ledger.observe(repair_frame("张三在北京用扳手修好了电脑"));
        let mut next = FrameInstance::new("看", "张三看了书");
        next.set_role(
            Role::Agent,
            RoleFiller::explicit("张三", None, None),
        );
        next.set_role(Role::Patient, RoleFiller::explicit("书", None, None));
        ledger.observe(next);
        let inherited = ledger.inherit_missing(&[Role::Location, Role::Instrument]);
        assert_eq!(inherited, 2);
        assert_eq!(ledger.answer(Role::Location), Some("北京"));
        assert_eq!(ledger.answer(Role::Instrument), Some("扳手"));
        assert_eq!(
            ledger.current().unwrap().roles[&Role::Location].status,
            FillerStatus::Inherited
        );
    }

    #[test]
    fn compiles_schema_from_two_examples() {
        let examples = vec![
            FrameExample::new("张三用扳手修好了电脑")
                .with_role(Role::Agent, "张三")
                .with_role(Role::Instrument, "扳手")
                .with_role(Role::Patient, "电脑"),
            FrameExample::new("李四用螺丝刀修好了收音机")
                .with_role(Role::Agent, "李四")
                .with_role(Role::Instrument, "螺丝刀")
                .with_role(Role::Patient, "收音机"),
        ];
        let negatives = vec!["张三修好了电脑".to_string()];
        let rule = FrameRuleCompiler::compile(
            "修",
            "repair/tool",
            &examples,
            &negatives,
            &CompileOptions::default(),
        )
        .expect("compiled");
        assert_eq!(rule.positive_hits, 2);
        assert_eq!(rule.negative_hits, 0);
        assert_eq!(
            rule.schema.realize.as_deref(),
            Some("{agent}用{instrument}修好了{patient}")
        );
        assert_eq!(rule.schema.status, SchemaStatus::Active);
        assert!(rule.schema.role_spec(Role::Instrument).unwrap().markers.contains(&"用".to_string()));
    }

    #[test]
    fn compiler_rejects_inconsistent_examples_and_negative_hits() {
        let inconsistent = vec![
            FrameExample::new("张三用扳手修好了电脑")
                .with_role(Role::Agent, "张三")
                .with_role(Role::Instrument, "扳手")
                .with_role(Role::Patient, "电脑"),
            FrameExample::new("李四拿螺丝刀修好了收音机")
                .with_role(Role::Agent, "李四")
                .with_role(Role::Instrument, "螺丝刀")
                .with_role(Role::Patient, "收音机"),
        ];
        let error = FrameRuleCompiler::compile(
            "修",
            "repair/tool",
            &inconsistent,
            &[],
            &CompileOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(error, CompileError::InconsistentSkeleton { .. }));

        let examples = vec![
            FrameExample::new("张三用扳手修好了电脑")
                .with_role(Role::Agent, "张三")
                .with_role(Role::Instrument, "扳手")
                .with_role(Role::Patient, "电脑"),
        ];
        let error = FrameRuleCompiler::compile(
            "修",
            "repair/tool",
            &examples,
            &["李四用螺丝刀修好了收音机".to_string()],
            &CompileOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(error, CompileError::NegativeHit { .. }));
    }

    #[test]
    fn compiled_schema_realizes_and_validates_types() {
        let examples = vec![
            FrameExample::new("张三用扳手修好了电脑")
                .with_role(Role::Agent, "张三")
                .with_role(Role::Instrument, "扳手")
                .with_role(Role::Patient, "电脑")
                .with_type(Role::Agent, "Person")
                .with_type(Role::Patient, "Device"),
            FrameExample::new("李四用螺丝刀修好了收音机")
                .with_role(Role::Agent, "李四")
                .with_role(Role::Instrument, "螺丝刀")
                .with_role(Role::Patient, "收音机")
                .with_type(Role::Agent, "Person")
                .with_type(Role::Patient, "Device"),
        ];
        let rule = FrameRuleCompiler::compile(
            "修",
            "repair/tool",
            &examples,
            &[],
            &CompileOptions::default(),
        )
        .expect("compiled");

        let mut frame = FrameInstance::new("修", "张三用扳手修好了电脑");
        frame.set_role(Role::Agent, RoleFiller::explicit("张三", None, None));
        frame.set_role(Role::Instrument, RoleFiller::explicit("扳手", None, None));
        frame.set_role(Role::Patient, RoleFiller::explicit("电脑", None, None));
        let text = rule.schema.realize(&frame).expect("realize");
        assert_eq!(text, "张三用扳手修好了电脑");

        let types = crate::type_system::TypeSystem::new();
        let mut actual = BTreeMap::new();
        actual.insert(Role::Agent, BTreeSet::from(["Place".to_string()]));
        let findings = rule.schema.validate(&frame, &actual, &types);
        assert!(findings.is_empty(), "unknown types stay open-world");

        let book = book();
        let mut compiled_book = FrameBook::new();
        assert!(compiled_book.register(rule.schema.clone()));
        let annotator = FrameAnnotator::new(&compiled_book);
        let annotated = annotator.annotate("王五用锤子修好了桌子");
        assert_eq!(annotated.len(), 1);
        assert_eq!(annotated[0].usable(Role::Agent), Some("王五"));
        assert_eq!(annotated[0].usable(Role::Instrument), Some("锤子"));
        assert_eq!(annotated[0].usable(Role::Patient), Some("桌子"));
        let _ = book;
    }
}

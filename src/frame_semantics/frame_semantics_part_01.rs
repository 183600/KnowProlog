use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::relation_frame::{EvidenceSpan, Polarity, RelationFrame};
use crate::type_system::{TypeSystem, TypeVerdict};

/// 谓词中心的语义角色。角色是语法标记与图谱三元组之间的共享接口。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Agent,
    Patient,
    Theme,
    Experiencer,
    Stimulus,
    Recipient,
    Beneficiary,
    Instrument,
    Location,
    Source,
    Goal,
    Time,
    Cause,
    Purpose,
    Manner,
    Result,
}

impl Role {
    pub const ALL: [Role; 16] = [
        Role::Agent,
        Role::Patient,
        Role::Theme,
        Role::Experiencer,
        Role::Stimulus,
        Role::Recipient,
        Role::Beneficiary,
        Role::Instrument,
        Role::Location,
        Role::Source,
        Role::Goal,
        Role::Time,
        Role::Cause,
        Role::Purpose,
        Role::Manner,
        Role::Result,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Agent => "agent",
            Role::Patient => "patient",
            Role::Theme => "theme",
            Role::Experiencer => "experiencer",
            Role::Stimulus => "stimulus",
            Role::Recipient => "recipient",
            Role::Beneficiary => "beneficiary",
            Role::Instrument => "instrument",
            Role::Location => "location",
            Role::Source => "source",
            Role::Goal => "goal",
            Role::Time => "time",
            Role::Cause => "cause",
            Role::Purpose => "purpose",
            Role::Manner => "manner",
            Role::Result => "result",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Role::Agent => "施事",
            Role::Patient => "受事",
            Role::Theme => "主题",
            Role::Experiencer => "经历者",
            Role::Stimulus => "刺激",
            Role::Recipient => "接受者",
            Role::Beneficiary => "受益者",
            Role::Instrument => "工具",
            Role::Location => "地点",
            Role::Source => "来源",
            Role::Goal => "终点",
            Role::Time => "时间",
            Role::Cause => "原因",
            Role::Purpose => "目的",
            Role::Manner => "方式",
            Role::Result => "结果",
        }
    }

    /// 默认格标记；`None` 表示无标记或由词序承担。
    pub fn default_marker(self) -> Option<&'static str> {
        match self {
            Role::Instrument => Some("用"),
            Role::Location => Some("在"),
            Role::Source => Some("从"),
            Role::Goal => Some("到"),
            Role::Recipient => Some("给"),
            Role::Cause => Some("因为"),
            Role::Purpose => Some("为了"),
            Role::Beneficiary => Some("为"),
            Role::Manner => Some("以"),
            Role::Patient => Some("把"),
            _ => None,
        }
    }

    pub fn is_projected_object(self) -> bool {
        matches!(
            self,
            Role::Patient
                | Role::Theme
                | Role::Stimulus
                | Role::Recipient
                | Role::Goal
                | Role::Result
                | Role::Source
        )
    }

    pub fn parse(text: &str) -> Option<Role> {
        let normalized = text.trim().to_ascii_lowercase();
        Role::ALL
            .iter()
            .copied()
            .find(|role| role.as_str() == normalized || role.label_zh() == text.trim())
    }
}

/// 角色填充的证据状态。生成端只消费 `Explicit / Inherited / Inferred`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillerStatus {
    Explicit,
    Inherited,
    Inferred,
    Unknown,
    Rejected,
}

impl FillerStatus {
    pub fn label_zh(self) -> &'static str {
        match self {
            FillerStatus::Explicit => "显式",
            FillerStatus::Inherited => "继承",
            FillerStatus::Inferred => "推断",
            FillerStatus::Unknown => "未知",
            FillerStatus::Rejected => "已拒绝",
        }
    }

    pub fn usable(self) -> bool {
        matches!(
            self,
            FillerStatus::Explicit | FillerStatus::Inherited | FillerStatus::Inferred
        )
    }
}

/// 一次角色填充：值、跨度、格标记、状态、置信度与来源。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleFiller {
    pub value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marker: Option<String>,
    pub status: FillerStatus,
    pub confidence: f64,
    pub source: String,
}

impl RoleFiller {
    pub fn explicit(
        value: impl Into<String>,
        span: Option<(usize, usize)>,
        marker: Option<String>,
    ) -> Self {
        Self {
            value: value.into().trim().to_string(),
            span,
            marker,
            status: FillerStatus::Explicit,
            confidence: 0.9,
            source: "文本".to_string(),
        }
    }

    pub fn inherited(value: impl Into<String>, from: &str) -> Self {
        Self {
            value: value.into(),
            span: None,
            marker: None,
            status: FillerStatus::Inherited,
            confidence: 0.7,
            source: format!("继承自 {from}"),
        }
    }

    pub fn is_usable(&self) -> bool {
        self.status.usable() && !self.value.trim().is_empty()
    }
}

/// 角色槽的声明：是否必需、允许的格标记、选择限制与排序权重。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleSpec {
    pub role: Role,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub markers: Vec<String>,
    #[serde(default)]
    pub allowed_types: BTreeSet<String>,
    #[serde(default = "default_role_weight")]
    pub weight: i32,
}

fn default_role_weight() -> i32 {
    1
}

impl RoleSpec {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            required: false,
            markers: role
                .default_marker()
                .map(|m| vec![m.to_string()])
                .unwrap_or_default(),
            allowed_types: BTreeSet::new(),
            weight: default_role_weight(),
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn with_markers<I, S>(mut self, markers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.markers = markers.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_types<I, S>(mut self, types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_types = types.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_weight(mut self, weight: i32) -> Self {
        self.weight = weight;
        self
    }
}

/// 帧 schema 的生命周期：候选、激活、退役。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaStatus {
    Candidate,
    Active,
    Retired,
}

impl Default for SchemaStatus {
    fn default() -> Self {
        SchemaStatus::Candidate
    }
}

impl SchemaStatus {
    pub fn label_zh(self) -> &'static str {
        match self {
            SchemaStatus::Candidate => "候选",
            SchemaStatus::Active => "激活",
            SchemaStatus::Retired => "退役",
        }
    }
}

/// 谓词义项 + 角色清单 + 格标记 + 选择限制 + 双向模板。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSchema {
    pub id: String,
    pub predicate: String,
    pub sense: String,
    #[serde(default)]
    pub surfaces: Vec<String>,
    pub roles: Vec<RoleSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realize: Option<String>,
    pub status: SchemaStatus,
    #[serde(default)]
    pub support: usize,
    #[serde(default)]
    pub positives: Vec<String>,
    #[serde(default)]
    pub negatives: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

impl FrameSchema {
    pub fn new(predicate: impl Into<String>, sense: impl Into<String>) -> Self {
        let predicate = predicate.into().trim().to_string();
        let mut schema = Self {
            id: String::new(),
            surfaces: vec![predicate.clone()],
            predicate,
            sense: sense.into().trim().to_string(),
            roles: vec![RoleSpec::new(Role::Agent), RoleSpec::new(Role::Patient)],
            realize: None,
            status: SchemaStatus::Candidate,
            support: 0,
            positives: Vec::new(),
            negatives: Vec::new(),
            sources: vec!["手工声明".to_string()],
        };
        schema.refresh_id();
        schema
    }

    pub fn with_surface(mut self, surface: impl Into<String>) -> Self {
        let surface = surface.into().trim().to_string();
        if !surface.is_empty() && !self.surfaces.contains(&surface) {
            self.surfaces.push(surface);
            self.refresh_id();
        }
        self
    }

    pub fn with_surfaces<I, S>(mut self, surfaces: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for surface in surfaces {
            self = self.with_surface(surface);
        }
        self
    }

    pub fn with_role(mut self, spec: RoleSpec) -> Self {
        self.roles.retain(|existing| existing.role != spec.role);
        self.roles.push(spec);
        self.roles.sort_by_key(|spec| spec.role);
        self.refresh_id();
        self
    }

    pub fn without_role(mut self, role: Role) -> Self {
        self.roles.retain(|existing| existing.role != role);
        self.refresh_id();
        self
    }

    pub fn with_realize(mut self, template: impl Into<String>) -> Self {
        self.realize = Some(template.into());
        self.refresh_id();
        self
    }

    pub fn activate(mut self) -> Self {
        self.status = SchemaStatus::Active;
        self
    }

    pub fn retire(mut self) -> Self {
        self.status = SchemaStatus::Retired;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        let source = source.into();
        if !source.trim().is_empty() && !self.sources.contains(&source) {
            self.sources.push(source);
        }
        self
    }

    pub fn refresh_id(&mut self) {
        let material = format!(
            "{}|{}|{}|{}",
            self.predicate,
            self.sense,
            self.roles
                .iter()
                .map(|spec| format!(
                    "{}:{}:{:?}",
                    spec.role.as_str(),
                    spec.required,
                    spec.markers
                ))
                .collect::<Vec<_>>()
                .join(","),
            self.realize.as_deref().unwrap_or("")
        );
        self.id = format!("schema:{}", fnv1a_hex(material.as_bytes()));
    }

    pub fn role_spec(&self, role: Role) -> Option<&RoleSpec> {
        self.roles.iter().find(|spec| spec.role == role)
    }

    pub fn required_roles(&self) -> Vec<Role> {
        self.roles
            .iter()
            .filter(|spec| spec.required)
            .map(|spec| spec.role)
            .collect()
    }

    pub fn check_selection(
        &self,
        role: Role,
        actual_types: &BTreeSet<String>,
        types: &TypeSystem,
    ) -> TypeVerdict {
        let Some(spec) = self.role_spec(role) else {
            return TypeVerdict::Unknown;
        };
        if spec.allowed_types.is_empty() {
            return TypeVerdict::Unknown;
        }
        types.sets_admit(actual_types, &spec.allowed_types)
    }

    pub fn validate(
        &self,
        frame: &FrameInstance,
        actual_types: &BTreeMap<Role, BTreeSet<String>>,
        types: &TypeSystem,
    ) -> Vec<SchemaFinding> {
        let mut findings = Vec::new();
        for spec in &self.roles {
            let filler = frame.filler(spec.role);
            match filler {
                None if spec.required => findings.push(SchemaFinding {
                    role: spec.role,
                    kind: SchemaFindingKind::MissingRequired,
                    detail: format!("缺少必需角色「{}」", spec.role.label_zh()),
                }),
                Some(filler) if filler.is_usable() => {
                    if let Some(actual) = actual_types.get(&spec.role) {
                        if self.check_selection(spec.role, actual, types) == TypeVerdict::Disjoint {
                            findings.push(SchemaFinding {
                                role: spec.role,
                                kind: SchemaFindingKind::SelectionViolation,
                                detail: format!("「{}」的类型与角色选择限制不相容", filler.value),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        findings
    }

    /// 用 schema 的双向模板实现一个帧；缺必需角色时返回 `None`。
    pub fn realize(&self, frame: &FrameInstance) -> Option<String> {
        let template = self.realize.as_deref()?;
        for role in self.required_roles() {
            if frame.usable(role).is_none() {
                return None;
            }
        }
        render_template(template, frame)
    }
}

/// schema 校验发现。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaFinding {
    pub role: Role,
    pub kind: SchemaFindingKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaFindingKind {
    MissingRequired,
    SelectionViolation,
}

/// 一次具体事件：角色填充带状态与证据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameInstance {
    pub id: String,
    pub predicate: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sense: Option<String>,
    pub roles: BTreeMap<Role, RoleFiller>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub evidence: EvidenceSpan,
    #[serde(default)]
    pub negated: bool,
    pub confidence: f64,
}

impl FrameInstance {
    pub fn new(predicate: impl Into<String>, text: &str) -> Self {
        let predicate = predicate.into().trim().to_string();
        let mut frame = Self {
            id: String::new(),
            predicate,
            sense: None,
            roles: BTreeMap::new(),
            schema: None,
            evidence: EvidenceSpan {
                sentence_index: 0,
                sentence: text.to_string(),
                start: 0,
                end: text.len(),
                ..EvidenceSpan::default()
            },
            negated: false,
            confidence: 0.0,
        };
        frame.refresh_id();
        frame
    }

    pub fn with_sense(mut self, sense: impl Into<String>) -> Self {
        self.sense = Some(sense.into());
        self.refresh_id();
        self
    }

    pub fn with_schema(mut self, schema_id: impl Into<String>) -> Self {
        self.schema = Some(schema_id.into());
        self.refresh_id();
        self
    }

    pub fn with_role(mut self, role: Role, filler: RoleFiller) -> Self {
        self.set_role(role, filler);
        self
    }

    pub fn set_role(&mut self, role: Role, filler: RoleFiller) {
        self.roles.insert(role, filler);
        self.refresh_id();
    }

    pub fn filler(&self, role: Role) -> Option<&RoleFiller> {
        self.roles.get(&role).filter(|filler| filler.is_usable())
    }

    pub fn usable(&self, role: Role) -> Option<&str> {
        self.filler(role).map(|filler| filler.value.as_str())
    }

    pub fn explicit_roles(&self) -> Vec<Role> {
        self.roles
            .iter()
            .filter(|(_, filler)| filler.status == FillerStatus::Explicit && filler.is_usable())
            .map(|(role, _)| *role)
            .collect()
    }

    pub fn refresh_id(&mut self) {
        let material = format!(
            "{}|{}|{}|{}",
            self.predicate,
            self.sense.as_deref().unwrap_or(""),
            self.roles
                .iter()
                .map(|(role, filler)| format!("{}={}", role.as_str(), filler.value))
                .collect::<Vec<_>>()
                .join(","),
            self.evidence.start
        );
        self.id = format!("frame:{}", fnv1a_hex(material.as_bytes()));
    }

    fn pick_role(&self, preferred: &[Role], used: &BTreeSet<Role>) -> Option<(Role, &RoleFiller)> {
        preferred.iter().copied().find_map(|role| {
            if used.contains(&role) {
                return None;
            }
            self.filler(role).map(|filler| (role, filler))
        })
    }

    /// 投影为带限定语的关系帧：核心角色落主语/宾语，旁格落 qualifiers。
    pub fn project(&self) -> Option<RelationFrame> {
        self.project_with_schema(None)
    }

    /// 带 schema 的投影：`RoleSpec::weight` 决定哪个角色优先成为宾语。
    pub fn project_with_schema(&self, schema: Option<&FrameSchema>) -> Option<RelationFrame> {
        let empty = BTreeSet::new();
        let (subject_role, subject) = self.pick_role(
            &[Role::Agent, Role::Experiencer, Role::Patient, Role::Theme],
            &empty,
        )?;
        let mut used = BTreeSet::new();
        used.insert(subject_role);
        let mut candidates: Vec<(Role, &RoleFiller)> = self
            .roles
            .iter()
            .filter(|(role, filler)| {
                !used.contains(role) && role.is_projected_object() && filler.is_usable()
            })
            .map(|(role, filler)| (*role, filler))
            .collect();
        candidates.sort_by_key(|(role, _)| {
            let weight = schema
                .and_then(|schema| schema.role_spec(*role))
                .map(|spec| spec.weight)
                .unwrap_or(0);
            (std::cmp::Reverse(weight), *role)
        });
        let (object_role, object) = candidates.into_iter().next()?;
        used.insert(object_role);

        let mut frame = RelationFrame::new(&subject.value, &self.predicate, &object.value);
        if self.negated {
            frame.polarity = Polarity::Negative;
        }
        frame.confidence = self.confidence.clamp(0.0, 1.0);
        frame.evidence = self.evidence.clone();
        frame.evidence.subject_span = subject.span;
        frame.evidence.object_span = object.span;
        frame.qualifiers = self
            .roles
            .iter()
            .filter(|(role, filler)| !used.contains(role) && filler.is_usable())
            .map(|(role, filler)| format!("{}={}", role.label_zh(), filler.value))
            .collect();
        frame.extraction.push("语义角色帧".to_string());
        Some(frame)
    }

    pub fn render_roles(&self) -> String {
        let mut output = format!("{}", self.predicate);
        if !self.roles.is_empty() {
            output.push('（');
            let mut first = true;
            for (role, filler) in &self.roles {
                if !filler.is_usable() {
                    continue;
                }
                if !first {
                    output.push_str("，");
                }
                first = false;
                let _ = write!(
                    output,
                    "{}={}〔{}〕",
                    role.label_zh(),
                    filler.value,
                    filler.status.label_zh()
                );
            }
            output.push('）');
        }
        output
    }
}

/// config.toml 的 `[frame_semantics]` 段；默认关闭。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FrameSemanticsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub schemas: Vec<FrameSchemaConfig>,
}

/// 一条可配置的 schema 声明。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FrameSchemaConfig {
    #[serde(default)]
    pub predicate: String,
    #[serde(default)]
    pub sense: String,
    #[serde(default)]
    pub surfaces: Vec<String>,
    #[serde(default)]
    pub roles: Vec<RoleSpec>,
    #[serde(default)]
    pub realize: Option<String>,
    #[serde(default)]
    pub status: SchemaStatus,
    #[serde(default)]
    pub support: usize,
    #[serde(default)]
    pub source: Option<String>,
}

impl FrameSchemaConfig {
    pub fn compile(&self) -> FrameSchema {
        let mut schema = FrameSchema::new(self.predicate.clone(), self.sense.clone());
        if !self.surfaces.is_empty() {
            schema.surfaces = self.surfaces.clone();
        }
        if !self.roles.is_empty() {
            schema.roles = self.roles.clone();
            schema.roles.sort_by_key(|spec| spec.role);
        }
        schema.realize = self.realize.clone();
        schema.status = self.status;
        schema.support = self.support;
        if let Some(source) = &self.source {
            schema.sources = vec![source.clone()];
        }
        schema.refresh_id();
        schema
    }
}

impl FrameSemanticsConfig {
    pub fn book(&self) -> FrameBook {
        FrameBook::from_config(self)
    }
}

/// 角色 schema 的注册表。默认不激活任何 schema，内置种子需显式装入。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameBook {
    #[serde(default)]
    schemas: BTreeMap<String, FrameSchema>,
}

impl FrameBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// 12 个核心角色上的一组可审计种子 schema。
    pub fn with_core_schemas() -> Self {
        let mut book = Self::new();
        book.register(
            FrameSchema::new("修", "repair")
                .with_surfaces(["修", "修好", "修理", "修过"])
                .with_role(RoleSpec::new(Role::Agent).required().with_weight(10))
                .with_role(RoleSpec::new(Role::Patient).required().with_weight(10))
                .with_role(RoleSpec::new(Role::Instrument).with_markers(["用", "拿", "使用"]))
                .with_role(RoleSpec::new(Role::Location))
                .with_role(RoleSpec::new(Role::Time))
                .with_role(RoleSpec::new(Role::Cause))
                .with_role(RoleSpec::new(Role::Purpose))
                .with_realize("{agent}[{time}][在{location}][用{instrument}]修好了{patient}[，因为{cause}][，为了{purpose}]")
                .with_source("内置种子")
                .activate(),
        );
        book.register(
            FrameSchema::new("去", "move")
                .with_surfaces(["去", "前往"])
                .without_role(Role::Patient)
                .with_role(RoleSpec::new(Role::Agent).required().with_weight(10))
                .with_role(
                    RoleSpec::new(Role::Goal)
                        .required()
                        .with_weight(10)
                        .with_markers(["到", "去", "往", "向"]),
                )
                .with_role(RoleSpec::new(Role::Source).with_markers(["从", "自"]))
                .with_role(RoleSpec::new(Role::Time))
                .with_role(RoleSpec::new(Role::Purpose).with_markers(["为了"]))
                .with_realize("{agent}[{time}][从{source}]去了{goal}[，为了{purpose}]")
                .with_source("内置种子")
                .activate(),
        );
        book.register(
            FrameSchema::new("给", "transfer")
                .with_surfaces(["给", "送给", "交给", "递给"])
                .with_role(RoleSpec::new(Role::Agent).required().with_weight(10))
                .with_role(RoleSpec::new(Role::Patient).required().with_weight(10))
                .with_role(
                    RoleSpec::new(Role::Recipient)
                        .required()
                        .with_weight(20)
                        .with_markers(["给", "对"]),
                )
                .with_role(RoleSpec::new(Role::Time))
                .with_realize("{agent}[{time}]把{patient}给了{recipient}")
                .with_source("内置种子")
                .activate(),
        );
        book.register(
            FrameSchema::new("看", "observe")
                .with_surfaces(["看", "看见", "看了", "观看", "看到", "看到了"])
                .with_role(RoleSpec::new(Role::Agent).required().with_weight(10))
                .with_role(RoleSpec::new(Role::Patient).required().with_weight(10))
                .with_role(RoleSpec::new(Role::Location))
                .with_role(RoleSpec::new(Role::Time))
                .with_realize("{agent}[{time}][在{location}]看了{patient}")
                .with_source("内置种子")
                .activate(),
        );
        book
    }

    /// 从配置编译：显式声明优先；没有声明时使用内置核心种子。
    pub fn from_config(config: &FrameSemanticsConfig) -> Self {
        let mut book = if config.schemas.is_empty() {
            Self::with_core_schemas()
        } else {
            Self::new()
        };
        for schema in &config.schemas {
            book.register(schema.compile());
        }
        book
    }

    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    pub fn register(&mut self, mut schema: FrameSchema) -> bool {
        if schema.id.is_empty() {
            schema.refresh_id();
        }
        self.schemas.insert(schema.id.clone(), schema).is_none()
    }

    pub fn remove(&mut self, schema_id: &str) -> Option<FrameSchema> {
        self.schemas.remove(schema_id)
    }

    pub fn schema(&self, schema_id: &str) -> Option<&FrameSchema> {
        self.schemas.get(schema_id)
    }

    pub fn schema_mut(&mut self, schema_id: &str) -> Option<&mut FrameSchema> {
        self.schemas.get_mut(schema_id)
    }

    pub fn activate(&mut self, schema_id: &str) -> bool {
        if let Some(schema) = self.schemas.get_mut(schema_id) {
            schema.status = SchemaStatus::Active;
            return true;
        }
        false
    }

    pub fn retire(&mut self, schema_id: &str) -> bool {
        if let Some(schema) = self.schemas.get_mut(schema_id) {
            schema.status = SchemaStatus::Retired;
            return true;
        }
        false
    }

    pub fn active_schemas(&self) -> Vec<&FrameSchema> {
        let mut schemas: Vec<&FrameSchema> = self
            .schemas
            .values()
            .filter(|schema| schema.status == SchemaStatus::Active)
            .collect();
        schemas.sort_by(|a, b| {
            a.predicate
                .cmp(&b.predicate)
                .then_with(|| a.sense.cmp(&b.sense))
                .then_with(|| a.id.cmp(&b.id))
        });
        schemas
    }

    pub fn schemas_for(&self, predicate: &str) -> Vec<&FrameSchema> {
        let predicate = predicate.trim();
        let mut schemas: Vec<&FrameSchema> = self
            .schemas
            .values()
            .filter(|schema| {
                schema.predicate == predicate || schema.surfaces.iter().any(|s| s == predicate)
            })
            .collect();
        schemas.sort_by(|a, b| a.sense.cmp(&b.sense).then_with(|| a.id.cmp(&b.id)));
        schemas
    }

    pub fn find_predicate(&self, text: &str) -> Option<PredicateHit> {
        let mut best: Option<(i64, PredicateHit)> = None;
        for schema in self.active_schemas() {
            for surface in &schema.surfaces {
                if surface.is_empty() {
                    continue;
                }
                for (start, _) in text.match_indices(surface.as_str()) {
                    let end = start + surface.len();
                    let mut score = (surface.len() as i64) * 100 + schema.support as i64;
                    if inside_time_word(text, start, end) {
                        score -= 10_000;
                    }
                    let before = &text[..start];
                    score += marker_hits(before).len() as i64 * 5;
                    if !before.trim().is_empty() {
                        score += 1;
                    }
                    if !text[end..].trim().is_empty() {
                        score += 1;
                    }
                    let candidate = PredicateHit {
                        schema_id: schema.id.clone(),
                        predicate: schema.predicate.clone(),
                        surface: surface.clone(),
                        start,
                        end,
                        support: schema.support,
                    };
                    let better = match &best {
                        None => true,
                        Some((current_score, current)) => {
                            score > *current_score
                                || (score == *current_score
                                    && (&candidate.schema_id, candidate.start)
                                        > (&current.schema_id, current.start))
                        }
                    };
                    if better {
                        best = Some((score, candidate));
                    }
                }
            }
        }
        best.map(|(_, hit)| hit)
    }

    pub fn schema_for_frame(&self, frame: &FrameInstance) -> Option<&FrameSchema> {
        frame
            .schema
            .as_deref()
            .and_then(|id| self.schemas.get(id))
            .or_else(|| {
                self.schemas_for(&frame.predicate)
                    .into_iter()
                    .find(|schema| schema.status == SchemaStatus::Active)
            })
    }

    pub fn project(&self, frame: &FrameInstance) -> Option<RelationFrame> {
        frame.project_with_schema(self.schema_for_frame(frame))
    }

    pub fn realize(&self, frame: &FrameInstance) -> Option<String> {
        self.schema_for_frame(frame)?.realize(frame)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        format!("book:{}", fnv1a_hex(canonical_json(&value).as_bytes()))
    }

    pub fn report(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "帧 schema 台账（{} 条）：", self.schemas.len());
        for schema in self.schemas.values() {
            let roles = schema
                .roles
                .iter()
                .map(|spec| {
                    if spec.required {
                        format!("{}*", spec.role.label_zh())
                    } else {
                        spec.role.label_zh().to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("、");
            let _ = writeln!(
                output,
                "- {}（{}，{}，支持 {}）：{}",
                schema.predicate,
                schema.sense,
                schema.status.label_zh(),
                schema.support,
                roles
            );
        }
        output
    }
}

/// 文本中命中的谓词表面形式。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateHit {
    pub schema_id: String,
    pub predicate: String,
    pub surface: String,
    pub start: usize,
    pub end: usize,
    pub support: usize,
}

const MARKERS: &[(&str, Role)] = &[
    ("因为", Role::Cause),
    ("由于", Role::Cause),
    ("为了", Role::Purpose),
    ("使用", Role::Instrument),
    ("用", Role::Instrument),
    ("拿", Role::Instrument),
    ("从", Role::Source),
    ("自", Role::Source),
    ("到", Role::Goal),
    ("往", Role::Goal),
    ("向", Role::Goal),
    ("给", Role::Recipient),
    ("对", Role::Recipient),
    ("在", Role::Location),
    ("于", Role::Location),
    ("为", Role::Beneficiary),
    ("把", Role::Patient),
    ("将", Role::Patient),
];

const PASSIVE_MARKERS: &[&str] = &["被", "由", "让"];

const TIME_WORDS: &[&str] = &[
    "昨天",
    "今天",
    "明天",
    "前天",
    "后天",
    "去年",
    "今年",
    "明年",
    "上周",
    "本周",
    "下周",
    "上个月",
    "这个月",
    "下个月",
    "早上",
    "上午",
    "中午",
    "下午",
    "晚上",
    "凌晨",
    "现在",
    "当时",
];

#[derive(Debug, Clone)]
struct MarkerHit {
    start: usize,
    end: usize,
    role: Role,
    marker: String,
    self_value: bool,
}

fn marker_hits(segment: &str) -> Vec<MarkerHit> {
    let mut hits: Vec<MarkerHit> = Vec::new();
    for (marker, role) in MARKERS {
        for (start, _) in segment.match_indices(marker) {
            hits.push(MarkerHit {
                start,
                end: start + marker.len(),
                role: *role,
                marker: (*marker).to_string(),
                self_value: false,
            });
        }
    }
    for word in TIME_WORDS {
        for (start, _) in segment.match_indices(word) {
            hits.push(MarkerHit {
                start,
                end: start + word.len(),
                role: Role::Time,
                marker: (*word).to_string(),
                self_value: true,
            });
        }
    }
    hits.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
    });
    let mut filtered: Vec<MarkerHit> = Vec::new();
    for hit in hits {
        if filtered
            .iter()
            .any(|existing| hit.start < existing.end && existing.start < hit.end)
        {
            continue;
        }
        filtered.push(hit);
    }
    filtered
}

fn inside_time_word(text: &str, start: usize, end: usize) -> bool {
    TIME_WORDS.iter().any(|word| {
        text.match_indices(word).any(|(word_start, _)| {
            let word_end = word_start + word.len();
            word_start <= start && end <= word_end && (word_end - word_start) > (end - start)
        })
    })
}

fn trim_value(text: &str) -> &str {
    text.trim().trim_matches(|c: char| {
        matches!(
            c,
            '，' | ',' | '。' | '.' | '；' | ';' | '：' | ':' | '！' | '!' | '？' | '?'
        )
    })
}

fn trim_marker_value(text: &str) -> &str {
    let trimmed = trim_value(text);
    trimmed.strip_prefix('了').map(str::trim).unwrap_or(trimmed)
}

fn trim_after_predicate(text: &str) -> &str {
    let mut trimmed = trim_value(text);
    for compound in ["好了", "完了", "过了", "着过"] {
        if let Some(rest) = trimmed.strip_prefix(compound) {
            return rest.trim();
        }
    }
    for particle in ["了", "着", "过", "完"] {
        if let Some(rest) = trimmed.strip_prefix(particle) {
            trimmed = rest.trim();
            break;
        }
    }
    trimmed
}

fn scan_markers(segment: &str, base: usize) -> (Vec<(Role, RoleFiller)>, Option<usize>) {
    let hits = marker_hits(segment);
    let first = hits.first().map(|hit| hit.start);
    let mut roles = Vec::new();
    for (index, hit) in hits.iter().enumerate() {
        let value = if hit.self_value {
            segment[hit.start..hit.end].to_string()
        } else {
            let end = hits
                .get(index + 1)
                .map(|next| next.start)
                .unwrap_or(segment.len());
            trim_marker_value(&segment[hit.end..end]).to_string()
        };
        if value.is_empty() {
            continue;
        }
        let span = if hit.self_value {
            (base + hit.start, base + hit.end)
        } else {
            let raw = &segment[hit.end..];
            let offset = raw.len() - raw.trim_start().len();
            let start = base + hit.end + offset;
            (start, start + value.len())
        };
        roles.push((
            hit.role,
            RoleFiller::explicit(value, Some(span), Some(hit.marker.clone())),
        ));
    }
    (roles, first)
}

fn passive_hit(segment: &str) -> Option<(usize, &'static str)> {
    PASSIVE_MARKERS
        .iter()
        .filter_map(|marker| segment.find(marker).map(|start| (start, *marker)))
        .min_by_key(|(start, _)| *start)
}

fn default_object_role(schema: Option<&FrameSchema>, frame: &FrameInstance) -> Role {
    let Some(schema) = schema else {
        return Role::Patient;
    };
    let candidates = [
        Role::Patient,
        Role::Theme,
        Role::Stimulus,
        Role::Goal,
        Role::Recipient,
        Role::Result,
    ];
    for role in candidates {
        if schema.role_spec(role).is_some() && frame.filler(role).is_none() {
            return role;
        }
    }
    candidates
        .into_iter()
        .find(|role| schema.role_spec(*role).is_some())
        .unwrap_or(Role::Patient)
}

fn detect_negation(before: &str, after: &str) -> bool {
    let tail = before.trim_end();
    tail.ends_with("没")
        || tail.ends_with("没有")
        || tail.ends_with("不")
        || tail.ends_with("未")
        || after.trim_start().starts_with("没")
        || after.trim_start().starts_with("未")
}

/// 基于格标记与词序的显式角色标注器。谓词未登记时可用 `annotate_predicate`。
pub struct FrameAnnotator<'a> {
    book: &'a FrameBook,
}

impl<'a> FrameAnnotator<'a> {
    pub fn new(book: &'a FrameBook) -> Self {
        Self { book }
    }

    pub fn annotate(&self, text: &str) -> Vec<FrameInstance> {
        let Some(hit) = self.book.find_predicate(text) else {
            return Vec::new();
        };
        let schema = self.book.schema(&hit.schema_id);
        let mut frame = self.annotate_with(text, &hit.surface, hit.start, hit.end, schema);
        frame.schema = Some(hit.schema_id);
        vec![frame]
    }

    /// 按句号/问号/感叹号/分号/换行切句后逐句标注，跨度仍指向整段文本。
    pub fn annotate_sentences(&self, text: &str) -> Vec<FrameInstance> {
        let mut frames = Vec::new();
        for (sentence_index, (base, sentence)) in split_sentences(text).into_iter().enumerate() {
            for mut frame in self.annotate(sentence) {
                frame.evidence.sentence_index = sentence_index;
                frame.evidence.start += base;
                frame.evidence.end += base;
                for filler in frame.roles.values_mut() {
                    if let Some((start, end)) = filler.span {
                        filler.span = Some((start + base, end + base));
                    }
                }
                frame.refresh_id();
                frames.push(frame);
            }
        }
        frames
    }

    /// 对未登记谓词做同样的格标记分析；谓词按字面最长匹配。
    pub fn annotate_predicate(&self, text: &str, predicate: &str) -> Option<FrameInstance> {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return None;
        }
        let start = text.find(predicate)?;
        let end = start + predicate.len();
        Some(self.annotate_with(text, predicate, start, end, None))
    }

    fn annotate_with(
        &self,
        text: &str,
        surface: &str,
        start: usize,
        end: usize,
        schema: Option<&FrameSchema>,
    ) -> FrameInstance {
        let before = &text[..start];
        let after = &text[end..];
        let mut frame = FrameInstance::new(
            schema
                .map(|s| s.predicate.clone())
                .unwrap_or_else(|| surface.to_string()),
            text,
        );
        if let Some(schema) = schema {
            frame.sense = Some(schema.sense.clone());
        }
        frame.negated = detect_negation(before, after);

        if let Some((passive_at, marker)) = passive_hit(before) {
            let patient_text = trim_value(&before[..passive_at]);
            if !patient_text.is_empty() {
                frame.set_role(
                    Role::Patient,
                    RoleFiller::explicit(patient_text, Some((0, passive_at)), None),
                );
            }
            let suffix = &before[passive_at + marker.len()..];
            let (roles, first) = scan_markers(suffix, passive_at + marker.len());
            let agent = trim_value(&suffix[..first.unwrap_or(suffix.len())]);
            if !agent.is_empty() {
                let offset = passive_at + marker.len();
                let raw = &suffix[..first.unwrap_or(suffix.len())];
                let leading = raw.len() - raw.trim_start().len();
                let agent_start = offset + leading;
                frame.set_role(
                    Role::Agent,
                    RoleFiller::explicit(
                        agent,
                        Some((agent_start, agent_start + agent.len())),
                        Some(marker.to_string()),
                    ),
                );
            }
            for (role, filler) in roles {
                frame.set_role(role, filler);
            }
        } else {
            let (before_roles, first) = scan_markers(before, 0);
            let agent_end = first.unwrap_or(before.len());
            let agent = trim_value(&before[..agent_end]);
            if !agent.is_empty() {
                let raw = &before[..agent_end];
                let leading = raw.len() - raw.trim_start().len();
                frame.set_role(
                    Role::Agent,
                    RoleFiller::explicit(agent, Some((leading, leading + agent.len())), None),
                );
            }
            for (role, filler) in before_roles {
                frame.set_role(role, filler);
            }
        }

        let after = after
            .strip_prefix(|c: char| c.is_whitespace())
            .unwrap_or(after);
        let (after_roles, first) = scan_markers(after, end);
        let mut has_object = false;
        for (role, filler) in after_roles {
            if role == Role::Patient || role == Role::Theme || role == Role::Goal {
                has_object = true;
            }
            frame.set_role(role, filler);
        }
        if let Some(first) = first {
            let remainder = trim_after_predicate(&after[..first]);
            if !remainder.is_empty() {
                let role = default_object_role(schema, &frame);
                let leading = after[..first].len() - after[..first].trim_start().len();
                frame.set_role(
                    role,
                    RoleFiller::explicit(
                        remainder,
                        Some((end + leading, end + leading + remainder.len())),
                        None,
                    ),
                );
                has_object = true;
            }
        } else {
            let remainder = trim_after_predicate(after);
            if !remainder.is_empty() {
                let role = default_object_role(schema, &frame);
                let leading = after.len() - after.trim_start().len();
                frame.set_role(
                    role,
                    RoleFiller::explicit(
                        remainder,
                        Some((end + leading, end + leading + remainder.len())),
                        None,
                    ),
                );
                has_object = true;
            }
        }
        if !has_object && frame.filler(Role::Patient).is_none() {
            if let Some(schema) = schema {
                if schema.role_spec(Role::Goal).is_some() {
                    frame.set_role(
                        Role::Goal,
                        RoleFiller {
                            value: String::new(),
                            span: None,
                            marker: None,
                            status: FillerStatus::Unknown,
                            confidence: 0.0,
                            source: "文本".to_string(),
                        },
                    );
                }
            }
        }

        let explicit = frame.explicit_roles().len();
        frame.confidence = if explicit == 0 {
            0.0
        } else {
            let total: f64 = frame
                .roles
                .values()
                .filter(|filler| filler.is_usable())
                .map(|filler| filler.confidence)
                .sum();
            (total / explicit as f64).clamp(0.0, 1.0)
        };
        frame.refresh_id();
        frame
    }
}

fn split_sentences(text: &str) -> Vec<(usize, &str)> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, character) in text.char_indices() {
        if matches!(character, '。' | '！' | '？' | '；' | '\n') {
            let end = index + character.len_utf8();
            let sentence = &text[start..end];
            if !sentence.trim().is_empty() {
                sentences.push((start, sentence));
            }
            start = end;
        }
    }
    if start < text.len() {
        let sentence = &text[start..];
        if !sentence.trim().is_empty() {
            sentences.push((start, sentence));
        }
    }
    sentences
}

/// 渲染带可选组的模板：`[用{instrument}]` 只在角色接地时出现。
pub fn render_template(template: &str, frame: &FrameInstance) -> Option<String> {
    let mut output = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('[') {
        output.push_str(&render_template_plain(&rest[..open], frame)?);
        let after_open = &rest[open + 1..];
        let close = after_open.find(']')?;
        if let Some(rendered) = render_template_plain(&after_open[..close], frame) {
            output.push_str(&rendered);
        }
        rest = &after_open[close + 1..];
    }
    output.push_str(&render_template_plain(rest, frame)?);
    Some(output)
}

fn render_template_plain(segment: &str, frame: &FrameInstance) -> Option<String> {
    let mut output = String::new();
    let mut rest = segment;
    while let Some(open) = rest.find('{') {
        output.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        let close = after_open.find('}')?;
        let token = &after_open[..close];
        let role_name = token.split(':').next().unwrap_or(token);
        let role = Role::parse(role_name)?;
        let value = frame.usable(role)?;
        output.push_str(value);
        rest = &after_open[close + 1..];
    }
    output.push_str(rest);
    Some(output)
}

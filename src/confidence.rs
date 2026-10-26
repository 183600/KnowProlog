//! PCC：可携带证明的置信度（Proof-Carrying Confidence）。
//!
//! 系统里原本有很多「置信度」：PQI 的意图后验、实体链接分数、词法证据分、软逻辑
//! 概率。它们尺度不同、语义不同，却经常被当成同一个数直接比较。本模块把置信度
//! 拆成四段可审计的契约：
//!
//! 1. **证据**：每个来源给出 [0,1] 的原始分；
//! 2. **校准**：用单调断点（PAV/保序回归的产物）把原始分映到
//!    `P(answer correct)`；
//! 3. **决策**：在 log-odds 上按来源折扣池化，再用风险阈值决定
//!    回答 / 含糊 / 澄清 / 拒答；
//! 4. **语言**：理解置信度与命题概率各用一套词表，绝不互相冒充。
//!
//! 关键约束：
//!
//! - **不训练**。校准表是断点，融合是加法，阈值是计数；所有数字都能在
//!   `config.toml` 里逐行追到。
//! - **概率不能加权平均**。加权和没有概率语义；这里先把每个来源减去先验，
//!   变成 log-odds 似然比，再乘折扣 `gamma` 后相加。
//! - **阈值自带证书**。`RiskCertificate` 用 Clopper-Pearson 单侧上界把
//!   「阈值处的错误率」写成可复算的计数证明；`config-lint` 会重算并核对。
//! - **默认关闭**。没有显式 `[reasoning.confidence] enabled = true` 时，
//!   本模块不接入任何回答路径，旧行为逐字节保持。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// 一个校准断点：原始分 `score` 对应历史正确率 `probability`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub score: f64,
    pub probability: f64,
}

/// 一个证据来源：名字、相关性折扣与校准断点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceSourceConfig {
    pub name: String,
    /// 相关性折扣：1.0 表示独立证据，越小越不重复计票。
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// 空表表示恒等映射（仍然过 clamp），便于先接来源后补校准。
    #[serde(default)]
    pub calibration: Vec<CalibrationPoint>,
}

/// 风险证书：用一个二项计数样本证明阈值处的错误率上界。
///
/// `n` 个被回答样本中有 `errors` 个错误；在置信水平 `1 - alpha` 下，
/// Clopper-Pearson 单侧上界必须不超过 `max_error_rate`，且配置里写下的
/// `claimed_upper_bound` 必须能被重新算出（误差不超过 `delta`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskCertificate {
    pub n: usize,
    pub errors: usize,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// 配置值与复算值允许的舍入差。
    #[serde(default = "default_claim_delta")]
    pub delta: f64,
    #[serde(default = "default_max_error_rate")]
    pub max_error_rate: f64,
    pub claimed_upper_bound: f64,
}

/// `[reasoning.confidence]`：置信度策略的源码。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_prior")]
    pub prior: f64,
    #[serde(default = "default_answer_threshold")]
    pub answer_threshold: f64,
    #[serde(default = "default_hedge_threshold")]
    pub hedge_threshold: f64,
    #[serde(default = "default_clarify_threshold")]
    pub clarify_threshold: f64,
    /// 判定阈值附近多宽的区间算「边界样本」。
    #[serde(default = "default_near_margin")]
    pub near_margin: f64,
    /// 非精确推理（BDD 超限近似）时给证据源的原始分。
    #[serde(default = "default_approximate_evidence")]
    pub approximate_evidence: f64,
    #[serde(default)]
    pub sources: Vec<ConfidenceSourceConfig>,
    #[serde(default)]
    pub certificate: Option<RiskCertificate>,
}

fn default_gamma() -> f64 {
    1.0
}

fn default_alpha() -> f64 {
    0.05
}

fn default_claim_delta() -> f64 {
    0.001
}

fn default_max_error_rate() -> f64 {
    0.05
}

fn default_prior() -> f64 {
    0.5
}

fn default_answer_threshold() -> f64 {
    0.86
}

fn default_hedge_threshold() -> f64 {
    0.62
}

fn default_clarify_threshold() -> f64 {
    0.55
}

fn default_near_margin() -> f64 {
    0.06
}

fn default_approximate_evidence() -> f64 {
    0.55
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            prior: default_prior(),
            answer_threshold: default_answer_threshold(),
            hedge_threshold: default_hedge_threshold(),
            clarify_threshold: default_clarify_threshold(),
            near_margin: default_near_margin(),
            approximate_evidence: default_approximate_evidence(),
            sources: Vec::new(),
            certificate: None,
        }
    }
}

/// 一条来源在本次融合中的贡献，保留原始分与校准后分数，便于逐行对账。
#[derive(Debug, Clone, PartialEq)]
pub struct SourceContribution {
    pub name: String,
    pub raw: Option<f64>,
    pub calibrated: Option<f64>,
    pub gamma: f64,
    pub log_odds: f64,
}

/// 融合结果：概率 + 每个来源的贡献（按配置顺序，确定性输出）。
#[derive(Debug, Clone, PartialEq)]
pub struct FusedConfidence {
    pub prior: f64,
    pub probability: f64,
    pub contributions: Vec<SourceContribution>,
}

/// 四档决策。四档不是修辞：每一档对应不同的后续动作。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceDecision {
    /// 证据足够：正常回答。
    Answer,
    /// 证据一般：回答但加限定，提示核对。
    Hedge,
    /// 没听懂或接近阈值：先问清再答。
    Clarify,
    /// 证据不足：拒答，不把猜测伪装成结论。
    Abstain,
}

impl ConfidenceDecision {
    pub fn code(self) -> &'static str {
        match self {
            ConfidenceDecision::Answer => "answer",
            ConfidenceDecision::Hedge => "hedge",
            ConfidenceDecision::Clarify => "clarify",
            ConfidenceDecision::Abstain => "abstain",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ConfidenceDecision::Answer => "回答",
            ConfidenceDecision::Hedge => "含糊回答",
            ConfidenceDecision::Clarify => "先澄清",
            ConfidenceDecision::Abstain => "拒答",
        }
    }

    /// 这一档是否允许把结论当作确定事实陈述。
    pub fn asserts(self) -> bool {
        matches!(self, ConfidenceDecision::Answer | ConfidenceDecision::Hedge)
    }
}

/// 逐命题置信信封：`[lower, upper]` 用万分比表示，避免浮点在计划指纹中漂移。
///
/// 区间不是频率概率，而是「已核实支持」与「仍可能被支持」之间的可审计范围：
/// 下界来自已登记的证据强度，上界扣除已知冲突；两端都保留来源等级，因此
/// 生成层可以据此断言、限定、先澄清或拒答，异步复核也可以据此计算 VOI。
pub const ENVELOPE_SCALE: u16 = 10_000;

/// 一条证据来源的保守强度等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeGrade {
    /// 图谱中已登记的断言（仍不是绝对真理）。
    Graph,
    /// 用户显式改口/更正后采纳的会话事实。
    SessionCorrection,
    /// 用户直陈的会话事实。
    SessionDirect,
    /// 转述、听说或低可信来源；正常回答路径不应把它当确定事实。
    SessionHearsay,
    /// 由规则/模型推出的结论。
    Inference,
}

impl EnvelopeGrade {
    pub fn base_lower_bps(self) -> u16 {
        match self {
            EnvelopeGrade::Graph => 9_000,
            EnvelopeGrade::SessionCorrection => 7_200,
            EnvelopeGrade::SessionDirect => 6_000,
            EnvelopeGrade::SessionHearsay => 3_500,
            EnvelopeGrade::Inference => 5_000,
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            EnvelopeGrade::Graph => "图谱断言",
            EnvelopeGrade::SessionCorrection => "会话更正",
            EnvelopeGrade::SessionDirect => "会话直陈",
            EnvelopeGrade::SessionHearsay => "传闻",
            EnvelopeGrade::Inference => "推断",
        }
    }
}

/// 区间决策：与四档 PCC 决策同构，但只看上下界是否跨过阈值。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeVerdict {
    /// 下界已经达到回答阈值。
    Assert,
    /// 下界只到含糊阈值：可以回答，但必须限定。
    Qualify,
    /// 下界不足、上界仍可能达阈值：先复核/澄清。
    Verify,
    /// 上界都不到含糊阈值：证据不足。
    Abstain,
}

impl EnvelopeVerdict {
    pub fn code(self) -> &'static str {
        match self {
            EnvelopeVerdict::Assert => "assert",
            EnvelopeVerdict::Qualify => "qualify",
            EnvelopeVerdict::Verify => "verify",
            EnvelopeVerdict::Abstain => "abstain",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            EnvelopeVerdict::Assert => "可断言",
            EnvelopeVerdict::Qualify => "需限定",
            EnvelopeVerdict::Verify => "需复核",
            EnvelopeVerdict::Abstain => "证据不足",
        }
    }

    pub fn asserts(self) -> bool {
        matches!(self, EnvelopeVerdict::Assert | EnvelopeVerdict::Qualify)
    }
}

/// 逐命题的置信区间（万分比，闭区间）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "ClaimEnvelopeWire")]
pub struct ClaimEnvelope {
    lower_bps: u16,
    upper_bps: u16,
}

#[derive(Deserialize)]
struct ClaimEnvelopeWire {
    lower_bps: u16,
    upper_bps: u16,
}

impl TryFrom<ClaimEnvelopeWire> for ClaimEnvelope {
    type Error = String;

    fn try_from(wire: ClaimEnvelopeWire) -> Result<Self, Self::Error> {
        if wire.lower_bps > ENVELOPE_SCALE || wire.upper_bps > ENVELOPE_SCALE {
            return Err(format!(
                "置信区间端点必须 <= {}，收到 {}..{}",
                ENVELOPE_SCALE, wire.lower_bps, wire.upper_bps
            ));
        }
        if wire.lower_bps > wire.upper_bps {
            return Err(format!(
                "置信区间下界 {} 不能高于上界 {}",
                wire.lower_bps, wire.upper_bps
            ));
        }
        Ok(Self {
            lower_bps: wire.lower_bps,
            upper_bps: wire.upper_bps,
        })
    }
}

impl ClaimEnvelope {
    /// 构造并规范化区间：端点钳制到 `[0, ENVELOPE_SCALE]`，上下界互换以保持
    /// 闭区间语义。需要严格校验外部输入时请用反序列化（越界/倒挂会报错）。
    pub fn new(lower_bps: u16, upper_bps: u16) -> Self {
        let lower = lower_bps.min(ENVELOPE_SCALE);
        let upper = upper_bps.min(ENVELOPE_SCALE);
        Self {
            lower_bps: lower.min(upper),
            upper_bps: lower.max(upper),
        }
    }

    pub fn point(probability: f64) -> Self {
        let value = to_bps(probability);
        Self {
            lower_bps: value,
            upper_bps: value,
        }
    }

    /// 由证据等级、佐证次数与冲突次数构造保守区间。
    ///
    /// 佐证按递减收益抬升下界；冲突同时压低上下界。所有运算都是整数，
    /// 同一输入永远得到同一区间。
    pub fn from_grade(grade: EnvelopeGrade, corroborations: usize, conflicts: usize) -> Self {
        let mut support = u32::from(grade.base_lower_bps());
        for _ in 0..corroborations.min(8) {
            support = ENVELOPE_SCALE as u32
                - (ENVELOPE_SCALE as u32 - support) * 9_000 / ENVELOPE_SCALE as u32;
        }
        let mut attack = 0u32;
        for _ in 0..conflicts.min(8) {
            attack = ENVELOPE_SCALE as u32
                - (ENVELOPE_SCALE as u32 - attack) * 7_000 / ENVELOPE_SCALE as u32;
        }
        let upper = ENVELOPE_SCALE as u32 - attack;
        let lower = support * upper / ENVELOPE_SCALE as u32;
        Self {
            lower_bps: lower.min(upper) as u16,
            upper_bps: upper as u16,
        }
    }

    pub fn lower_bps(self) -> u16 {
        self.lower_bps
    }

    pub fn upper_bps(self) -> u16 {
        self.upper_bps
    }

    pub fn lower(self) -> f64 {
        f64::from(self.lower_bps) / f64::from(ENVELOPE_SCALE)
    }

    pub fn upper(self) -> f64 {
        f64::from(self.upper_bps) / f64::from(ENVELOPE_SCALE)
    }

    pub fn width_bps(self) -> u16 {
        self.upper_bps - self.lower_bps
    }

    pub fn is_point(self) -> bool {
        self.lower_bps == self.upper_bps
    }

    /// 区间是否跨过给定阈值（下界不足而上界可达）。
    pub fn crosses(self, threshold_bps: u16) -> bool {
        self.lower_bps < threshold_bps && self.upper_bps >= threshold_bps
    }

    /// 只看区间上下界的四档决策；`answer_bps` 与 `hedge_bps` 会先排好序。
    pub fn verdict(self, answer_bps: u16, hedge_bps: u16) -> EnvelopeVerdict {
        let answer = answer_bps.min(ENVELOPE_SCALE);
        let hedge = hedge_bps.min(answer);
        if self.lower_bps >= answer {
            EnvelopeVerdict::Assert
        } else if self.lower_bps >= hedge {
            EnvelopeVerdict::Qualify
        } else if self.upper_bps >= hedge {
            EnvelopeVerdict::Verify
        } else {
            EnvelopeVerdict::Abstain
        }
    }

    /// VOI 优先级：跨回答阈值的区间最值得复核；与含糊/回答判定带的
    /// 重叠宽度提供次级排序。结果在 `[0, ENVELOPE_SCALE]`。
    pub fn deliberation_priority(self, answer_bps: u16, hedge_bps: u16) -> u16 {
        let answer = answer_bps.min(ENVELOPE_SCALE);
        let hedge = hedge_bps.min(answer);
        let overlap = self
            .upper_bps
            .min(answer)
            .saturating_sub(self.lower_bps.max(hedge));
        let answer_cross = u16::from(self.crosses(answer)) * 5_000;
        let hedge_cross = u16::from(self.crosses(hedge)) * 2_000;
        overlap
            .saturating_add(answer_cross)
            .saturating_add(hedge_cross)
            .min(ENVELOPE_SCALE)
    }

    /// 只渲染区间本身（不绑定决策阈值）。
    pub fn render_interval(self) -> String {
        format!("{}–{}%", percent(self.lower_bps), percent(self.upper_bps))
    }

    /// 对外审计措辞：百分比 + 四档结论。
    pub fn render_zh(self, answer_bps: u16, hedge_bps: u16) -> String {
        let verdict = self.verdict(answer_bps, hedge_bps);
        format!("{}（{}）", self.render_interval(), verdict.label_zh())
    }
}

fn to_bps(probability: f64) -> u16 {
    let clamped = if probability.is_finite() {
        probability.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (clamped * f64::from(ENVELOPE_SCALE)).round() as u16
}

fn percent(bps: u16) -> String {
    let whole = bps / 100;
    let fraction = bps % 100;
    if fraction == 0 {
        format!("{}", whole)
    } else {
        format!("{}.{:02}", whole, fraction)
    }
}

/// 一次决策的完整回执。
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub decision: ConfidenceDecision,
    pub confidence: f64,
    pub understanding: f64,
    pub near_boundary: bool,
    pub fused: FusedConfidence,
    pub certificate: Option<CertificateStatus>,
}

impl DecisionRecord {
    /// 两套词表绝不混用：理解置信度说「把握」，命题概率说「可能」。
    pub fn understanding_wording(&self) -> &'static str {
        understanding_wording(self.understanding)
    }

    pub fn proposition_wording(&self, probability: f64) -> &'static str {
        proposition_wording(probability)
    }

    /// 单行审计回执：每个来源的原始分、校准分、折扣与证书都可核对。
    pub fn render_receipt(&self) -> String {
        let contributions = self
            .fused
            .contributions
            .iter()
            .map(|contribution| match (contribution.raw, contribution.calibrated) {
                (Some(raw), Some(calibrated)) => format!(
                    "{} {:.2}→{:.2}×{:.2}",
                    contribution.name, raw, calibrated, contribution.gamma
                ),
                _ => format!("{} 未提供×{:.2}", contribution.name, contribution.gamma),
            })
            .collect::<Vec<_>>()
            .join("；");
        let mut line = format!(
            "置信度决策（PCC）：回答置信度={:.3}（{}）；决策={}；理解措辞={}",
            self.confidence,
            contributions,
            self.decision.label(),
            self.understanding_wording()
        );
        if self.near_boundary {
            line.push_str("；边界=接近判定阈值");
        }
        if let Some(certificate) = &self.certificate {
            line.push_str(&format!("；风险证书 {}", certificate.render_summary()));
        }
        line
    }
}

/// 证书复算结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CertificateStatus {
    pub n: usize,
    pub errors: usize,
    pub alpha: f64,
    pub upper_bound: f64,
    pub max_error_rate: f64,
    pub verified: bool,
}

impl CertificateStatus {
    pub fn render_summary(&self) -> String {
        format!(
            "n={}/错误={}/上界={:.4}≤{:.4}/α={:.2}{}",
            self.n,
            self.errors,
            self.upper_bound,
            self.max_error_rate,
            self.alpha,
            if self.verified { "（已验证）" } else { "（未通过）" }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSeverity {
    Error,
    Warning,
}

/// 契约问题；`config_contract` 负责把它翻译成统一的 `ConfigIssue`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidenceIssue {
    pub severity: ConfidenceSeverity,
    pub code: String,
    pub path: String,
    pub message: String,
}

impl ConfidenceIssue {
    fn error(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: ConfidenceSeverity::Error,
            code: code.to_string(),
            path: path.into(),
            message: message.into(),
        }
    }

    fn warning(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: ConfidenceSeverity::Warning,
            code: code.to_string(),
            path: path.into(),
            message: message.into(),
        }
    }
}

impl ConfidenceConfig {
    /// 依据名字查来源配置；同名来源在校验阶段已判错，运行期取第一个。
    pub fn source(&self, name: &str) -> Option<&ConfidenceSourceConfig> {
        self.sources.iter().find(|source| source.name == name)
    }

    /// 在 log-odds 上池化：`logit p = logit π + Σ γ_s (logit q_s − logit π)`。
    ///
    /// 每个来源先减先验，贡献的是似然比而不是原始概率；缺失来源直接跳过，
    /// 不把「没提供证据」误算成「证据为 0」。
    pub fn fuse(&self, observations: &[(&str, f64)]) -> FusedConfidence {
        let prior = clamp_probability(self.prior);
        let prior_log_odds = logit(prior);
        let mut total = prior_log_odds;
        let mut contributions = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            let raw = observations
                .iter()
                .find(|(name, _)| *name == source.name)
                .map(|(_, value)| clamp_probability(*value));
            let calibrated = raw.map(|value| calibrate(&source.calibration, value));
            let log_odds = match calibrated {
                Some(value) => {
                    let contribution = source.gamma.max(0.0) * (logit(value) - prior_log_odds);
                    total += contribution;
                    contribution
                }
                None => 0.0,
            };
            contributions.push(SourceContribution {
                name: source.name.clone(),
                raw,
                calibrated,
                gamma: source.gamma,
                log_odds,
            });
        }
        FusedConfidence {
            prior,
            probability: sigmoid(total),
            contributions,
        }
    }

    /// 决策：先看「是否听懂」，再看「证据是否够」，最后看是否贴着阈值边界。
    pub fn decide(
        &self,
        understanding: f64,
        observations: &[(&str, f64)],
        near_boundary: bool,
    ) -> DecisionRecord {
        let understanding = clamp_probability(understanding);
        let fused = self.fuse(observations);
        let decision = if understanding < self.clarify_threshold {
            ConfidenceDecision::Clarify
        } else if near_boundary && fused.probability < self.answer_threshold {
            ConfidenceDecision::Hedge
        } else if fused.probability >= self.answer_threshold {
            ConfidenceDecision::Answer
        } else if fused.probability >= self.hedge_threshold {
            ConfidenceDecision::Hedge
        } else {
            ConfidenceDecision::Abstain
        };
        DecisionRecord {
            decision,
            confidence: fused.probability,
            understanding,
            near_boundary,
            fused,
            certificate: self.certificate.as_ref().and_then(|certificate| {
                certificate
                    .verify()
                    .ok()
            }),
        }
    }

    /// 把回答/含糊阈值转成信封使用的万分比。
    pub fn envelope_thresholds_bps(&self) -> (u16, u16) {
        (to_bps(self.answer_threshold), to_bps(self.hedge_threshold))
    }

    /// 按当前阈值判断一个置信信封的决策档位。
    pub fn envelope_verdict(&self, envelope: ClaimEnvelope) -> EnvelopeVerdict {
        let (answer, hedge) = self.envelope_thresholds_bps();
        envelope.verdict(answer, hedge)
    }

    /// 按当前阈值渲染置信区间。
    pub fn render_envelope(&self, envelope: ClaimEnvelope) -> String {
        let (answer, hedge) = self.envelope_thresholds_bps();
        envelope.render_zh(answer, hedge)
    }

    /// 静态校验：阈值次序、校准单调性、证书可复算性。
    pub fn validate(&self) -> Vec<ConfidenceIssue> {
        let mut issues = Vec::new();
        let mut probability = |value: f64, path: &str, label: &str| {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                issues.push(ConfidenceIssue::error(
                    "confidence_probability_out_of_range",
                    path,
                    format!("{}={} 不在 [0,1]", label, value),
                ));
            }
        };
        probability(self.prior, "reasoning.confidence.prior", "先验");
        probability(
            self.answer_threshold,
            "reasoning.confidence.answer_threshold",
            "回答阈值",
        );
        probability(
            self.hedge_threshold,
            "reasoning.confidence.hedge_threshold",
            "含糊阈值",
        );
        probability(
            self.clarify_threshold,
            "reasoning.confidence.clarify_threshold",
            "澄清阈值",
        );
        probability(
            self.near_margin,
            "reasoning.confidence.near_margin",
            "边界宽度",
        );
        probability(
            self.approximate_evidence,
            "reasoning.confidence.approximate_evidence",
            "近似证据分",
        );

        if self.clarify_threshold > self.hedge_threshold {
            issues.push(ConfidenceIssue::error(
                "confidence_threshold_order",
                "reasoning.confidence.clarify_threshold",
                format!(
                    "澄清阈值 {:.3} 高于含糊阈值 {:.3}：会造成先含糊再澄清的倒挂",
                    self.clarify_threshold, self.hedge_threshold
                ),
            ));
        }
        if self.hedge_threshold > self.answer_threshold {
            issues.push(ConfidenceIssue::error(
                "confidence_threshold_order",
                "reasoning.confidence.hedge_threshold",
                format!(
                    "含糊阈值 {:.3} 高于回答阈值 {:.3}：回答档将永远不可达",
                    self.hedge_threshold, self.answer_threshold
                ),
            ));
        }

        let mut names = BTreeMap::new();
        for (index, source) in self.sources.iter().enumerate() {
            let path = format!("reasoning.confidence.sources[{}]", index);
            if source.name.trim().is_empty() {
                issues.push(ConfidenceIssue::error(
                    "confidence_source_unnamed",
                    format!("{}.name", path),
                    "证据来源名字为空，运行时无法按名提供观测",
                ));
            } else if let Some(previous) = names.insert(source.name.clone(), index) {
                issues.push(ConfidenceIssue::error(
                    "confidence_source_duplicate",
                    format!("{}.name", path),
                    format!(
                        "证据来源「{}」重复（首次出现在 sources[{}]）",
                        source.name, previous
                    ),
                ));
            }
            if !source.gamma.is_finite() || source.gamma <= 0.0 {
                issues.push(ConfidenceIssue::error(
                    "confidence_gamma_out_of_range",
                    format!("{}.gamma", path),
                    format!(
                        "gamma={} 必须为正：0 会让来源永远失效，负数会反转证据",
                        source.gamma
                    ),
                ));
            } else if source.gamma > 1.5 {
                issues.push(ConfidenceIssue::warning(
                    "confidence_gamma_overweighted",
                    format!("{}.gamma", path),
                    format!(
                        "gamma={} > 1.5：单一来源被过度放大，可能重复计票",
                        source.gamma
                    ),
                ));
            }
            let mut last_score: Option<f64> = None;
            let mut last_probability: Option<f64> = None;
            for (point_index, point) in source.calibration.iter().enumerate() {
                let point_path = format!("{}.calibration[{}]", path, point_index);
                if !point.score.is_finite() || !(0.0..=1.0).contains(&point.score) {
                    issues.push(ConfidenceIssue::error(
                        "confidence_calibration_score_out_of_range",
                        format!("{}.score", point_path),
                        format!("校准原始分 {} 不在 [0,1]", point.score),
                    ));
                }
                if !point.probability.is_finite()
                    || !(0.0..=1.0).contains(&point.probability)
                {
                    issues.push(ConfidenceIssue::error(
                        "confidence_calibration_probability_out_of_range",
                        format!("{}.probability", point_path),
                        format!("校准概率 {} 不在 [0,1]", point.probability),
                    ));
                }
                if let Some(previous) = last_score {
                    if point.score <= previous {
                        issues.push(ConfidenceIssue::error(
                            "confidence_calibration_score_order",
                            format!("{}.score", point_path),
                            format!(
                                "校准断点必须按原始分严格递增：{} 之后出现 {}",
                                previous, point.score
                            ),
                        ));
                    }
                }
                if let Some(previous) = last_probability {
                    if point.probability < previous {
                        issues.push(ConfidenceIssue::error(
                            "confidence_calibration_non_monotone",
                            format!("{}.probability", point_path),
                            format!(
                                "校准函数必须单调不减：概率从 {} 降到 {}；请先用 PAV 合并违规块",
                                previous, point.probability
                            ),
                        ));
                    }
                }
                last_score = Some(point.score);
                last_probability = Some(point.probability);
            }
        }

        if self.enabled && self.sources.is_empty() {
            issues.push(ConfidenceIssue::warning(
                "confidence_profile_without_sources",
                "reasoning.confidence.sources",
                "已启用置信度层但没有任何证据来源：所有回答都会退回先验",
            ));
        }

        match self.certificate.as_ref() {
            None => {
                if self.enabled {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_missing",
                        "reasoning.confidence.certificate",
                        "启用置信度层必须提供风险证书：阈值不能是魔法数",
                    ));
                }
            }
            Some(certificate) => {
                if certificate.n == 0 {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_empty_sample",
                        "reasoning.confidence.certificate.n",
                        "风险证书样本数 n=0，无法计算错误率上界",
                    ));
                }
                if certificate.errors > certificate.n {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_errors_exceed_n",
                        "reasoning.confidence.certificate.errors",
                        format!(
                            "错误数 {} 大于样本数 {}",
                            certificate.errors, certificate.n
                        ),
                    ));
                }
                if !certificate.alpha.is_finite()
                    || !(0.0..1.0).contains(&certificate.alpha)
                {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_alpha_out_of_range",
                        "reasoning.confidence.certificate.alpha",
                        format!("alpha={} 不在 (0,1)", certificate.alpha),
                    ));
                }
                if !certificate.delta.is_finite() || certificate.delta < 0.0 {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_delta_out_of_range",
                        "reasoning.confidence.certificate.delta",
                        format!("delta={} 必须为非负有限数", certificate.delta),
                    ));
                }
                if !certificate.max_error_rate.is_finite()
                    || !(0.0..1.0).contains(&certificate.max_error_rate)
                {
                    issues.push(ConfidenceIssue::error(
                        "confidence_certificate_error_rate_out_of_range",
                        "reasoning.confidence.certificate.max_error_rate",
                        format!(
                            "max_error_rate={} 不在 (0,1)",
                            certificate.max_error_rate
                        ),
                    ));
                }
                match certificate.verify() {
                    Ok(status) => {
                        if status.upper_bound
                            > certificate.claimed_upper_bound + certificate.delta
                        {
                            issues.push(ConfidenceIssue::error(
                                "confidence_certificate_stale",
                                "reasoning.confidence.certificate.claimed_upper_bound",
                                format!(
                                    "证书写的上界 {:.6} 低于复算值 {:.6}（容差 {}）",
                                    certificate.claimed_upper_bound,
                                    status.upper_bound,
                                    certificate.delta
                                ),
                            ));
                        }
                    }
                    Err(message) => issues.push(ConfidenceIssue::error(
                        "confidence_certificate_invalid",
                        "reasoning.confidence.certificate",
                        message,
                    )),
                }
            }
        }
        issues
    }
}

impl RiskCertificate {
    /// 复算并核对证书；失败时给出可定位的原因。
    pub fn verify(&self) -> Result<CertificateStatus, String> {
        if self.n == 0 {
            return Err("样本数 n=0".to_string());
        }
        if self.errors > self.n {
            return Err(format!("错误数 {} > 样本数 {}", self.errors, self.n));
        }
        if !self.alpha.is_finite() || !(0.0..1.0).contains(&self.alpha) {
            return Err(format!("alpha={} 不在 (0,1)", self.alpha));
        }
        if !self.delta.is_finite() || self.delta < 0.0 {
            return Err(format!("delta={} 必须为非负有限数", self.delta));
        }
        if !self.max_error_rate.is_finite()
            || !(0.0..1.0).contains(&self.max_error_rate)
        {
            return Err(format!(
                "max_error_rate={} 不在 (0,1)",
                self.max_error_rate
            ));
        }
        if !self.claimed_upper_bound.is_finite()
            || !(0.0..=1.0).contains(&self.claimed_upper_bound)
        {
            return Err(format!(
                "claimed_upper_bound={} 不在 [0,1]",
                self.claimed_upper_bound
            ));
        }
        let upper_bound = clopper_pearson_upper(self.n, self.errors, self.alpha);
        if (upper_bound - self.claimed_upper_bound).abs() > self.delta {
            return Err(format!(
                "证书上界 {:.6} 与复算值 {:.6} 相差超过容差 {}",
                self.claimed_upper_bound, upper_bound, self.delta
            ));
        }
        if upper_bound > self.max_error_rate {
            return Err(format!(
                "复算上界 {:.6} 超过允许错误率 {:.6}",
                upper_bound, self.max_error_rate
            ));
        }
        Ok(CertificateStatus {
            n: self.n,
            errors: self.errors,
            alpha: self.alpha,
            upper_bound,
            max_error_rate: self.max_error_rate,
            verified: true,
        })
    }
}

/// 理解置信度的词表：只说「我有多懂你在问什么」。
pub fn understanding_wording(confidence: f64) -> &'static str {
    if confidence >= 0.90 {
        "很有把握"
    } else if confidence >= 0.75 {
        "较有把握"
    } else if confidence >= 0.60 {
        "把握一般"
    } else if confidence >= 0.45 {
        "建议核对"
    } else {
        "无法判断"
    }
}

/// 命题概率的词表：只说「这件事有多可能」。
///
/// 与 [`crate::markov_logic::MarkovLogicEngine::hedge`] 共享同一张表，
/// 避免同一个 P 在两条通道里被说成两个程度。
pub fn proposition_wording(probability: f64) -> &'static str {
    if probability >= 0.9 {
        "极有可能"
    } else if probability >= 0.75 {
        "很可能"
    } else if probability >= 0.6 {
        "比较可能"
    } else if probability >= 0.5 {
        "有可能"
    } else if probability >= 0.35 {
        "不太确定，或许"
    } else {
        "可能性较低"
    }
}

/// 保序回归（PAV）：只数数，不训练。
///
/// 输入 `(原始分, 是否正确)` 样本，输出单调不减的断点。同分样本先聚合为
/// 一个块；出现下降时与左邻块合并，直到序列单调。相同分数输出同一个
/// 池化正确率，保证断点严格递增、插值唯一。
pub fn fit_isotonic(samples: &[(f64, bool)]) -> Vec<CalibrationPoint> {
    let mut scores: Vec<(f64, f64)> = samples
        .iter()
        .map(|(score, correct)| (score.clamp(0.0, 1.0), if *correct { 1.0 } else { 0.0 }))
        .collect();
    scores.sort_by(|a, b| a.0.total_cmp(&b.0));
    if scores.is_empty() {
        return Vec::new();
    }

    // 先按唯一分数聚合，记录每个分组在输出中的起点。
    let mut groups: Vec<(f64, f64, f64)> = Vec::new(); // score, sum, weight
    for (score, label) in scores {
        match groups.last_mut() {
            Some((last_score, sum, weight)) if *last_score == score => {
                *sum += label;
                *weight += 1.0;
            }
            _ => groups.push((score, label, 1.0)),
        }
    }

    // PAV：每个块记录起止分组与 (sum, weight)。
    let mut blocks: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (index, (_, sum, weight)) in groups.iter().enumerate() {
        blocks.push((index, index + 1, *sum, *weight));
        while blocks.len() >= 2 {
            let last = blocks[blocks.len() - 1];
            let previous = blocks[blocks.len() - 2];
            let last_mean = last.2 / last.3;
            let previous_mean = previous.2 / previous.3;
            if previous_mean <= last_mean {
                break;
            }
            blocks.pop();
            let merged = (
                previous.0,
                last.1,
                previous.2 + last.2,
                previous.3 + last.3,
            );
            *blocks.last_mut().expect("刚弹出的是最后一块") = merged;
        }
    }

    let mut points = Vec::with_capacity(groups.len());
    for (start, end, sum, weight) in blocks {
        let probability = sum / weight;
        for group in &groups[start..end] {
            points.push(CalibrationPoint {
                score: group.0,
                probability,
            });
        }
    }
    points
}

/// 断点插值。空表 = 恒等映射；单点 = 常数；区间外按端点钳制。
pub fn calibrate(points: &[CalibrationPoint], score: f64) -> f64 {
    let score = clamp_probability(score);
    let Some(first) = points.first() else {
        return score;
    };
    if score <= first.score {
        return clamp_probability(first.probability);
    }
    let Some(last) = points.last() else {
        return score;
    };
    if score >= last.score {
        return clamp_probability(last.probability);
    }
    for pair in points.windows(2) {
        let left = &pair[0];
        let right = &pair[1];
        if score >= left.score && score <= right.score {
            let width = right.score - left.score;
            if width <= f64::EPSILON {
                return clamp_probability(right.probability);
            }
            let ratio = (score - left.score) / width;
            let value = left.probability + ratio * (right.probability - left.probability);
            return clamp_probability(value);
        }
    }
    clamp_probability(last.probability)
}

/// `P[Bin(n,p) ≤ k]`：Clopper-Pearson 求根用的二项分布 CDF。
///
/// 用递推而不是阶乘，避免大 n 溢出；`p` 接近 1 时按对称性换算，
/// 保证尾部仍然可算。
pub fn binomial_cdf_at_most(n: usize, k: usize, p: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let p = clamp_probability(p);
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    // CDF(k;n,p) = 1 - CDF(n-k-1;n,1-p)，用较小的一侧递推。
    if p > 0.5 {
        return 1.0 - binomial_cdf_at_most(n, n - k - 1, 1.0 - p);
    }
    let q = 1.0 - p;
    let mut term = q.powf(n as f64);
    let mut sum = term;
    for index in 0..k {
        let ratio = (n - index) as f64 / (index + 1) as f64 * p / q;
        term *= ratio;
        sum += term;
    }
    sum.clamp(0.0, 1.0)
}

/// Clopper-Pearson 单侧上界：解 `P[Bin(n,p) ≤ errors] = alpha`。
///
/// 这就是「计数型风险证书」的全部数学：给定 n、errors 与 alpha，
/// 返回能被 `1-alpha` 置信度支持的最大错误率。
pub fn clopper_pearson_upper(n: usize, errors: usize, alpha: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let errors = errors.min(n);
    if errors == n {
        return 1.0;
    }
    let alpha = alpha.clamp(1e-12, 1.0 - 1e-12);
    if errors == 0 {
        return 1.0 - alpha.powf(1.0 / n as f64);
    }
    let mut low = errors as f64 / n as f64;
    let mut high = 1.0;
    for _ in 0..120 {
        let middle = (low + high) / 2.0;
        if binomial_cdf_at_most(n, errors, middle) > alpha {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

fn clamp_probability(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn logit(probability: f64) -> f64 {
    let probability = probability.clamp(1e-6, 1.0 - 1e-6);
    (probability / (1.0 - probability)).ln()
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> ConfidenceConfig {
        ConfidenceConfig {
            enabled: true,
            prior: 0.5,
            answer_threshold: 0.86,
            hedge_threshold: 0.62,
            clarify_threshold: 0.55,
            near_margin: 0.06,
            approximate_evidence: 0.55,
            sources: vec![
                ConfidenceSourceConfig {
                    name: "intent".to_string(),
                    gamma: 0.7,
                    calibration: vec![
                        CalibrationPoint {
                            score: 0.5,
                            probability: 0.55,
                        },
                        CalibrationPoint {
                            score: 0.9,
                            probability: 0.88,
                        },
                    ],
                },
                ConfidenceSourceConfig {
                    name: "evidence".to_string(),
                    gamma: 0.6,
                    calibration: vec![
                        CalibrationPoint {
                            score: 0.55,
                            probability: 0.65,
                        },
                        CalibrationPoint {
                            score: 1.0,
                            probability: 0.98,
                        },
                    ],
                },
                ConfidenceSourceConfig {
                    name: "margin".to_string(),
                    gamma: 0.3,
                    calibration: vec![
                        CalibrationPoint {
                            score: 0.0,
                            probability: 0.35,
                        },
                        CalibrationPoint {
                            score: 1.0,
                            probability: 0.95,
                        },
                    ],
                },
            ],
            certificate: Some(RiskCertificate {
                n: 240,
                errors: 4,
                alpha: 0.05,
                delta: 0.001,
                max_error_rate: 0.05,
                claimed_upper_bound: 0.0377,
            }),
        }
    }

    #[test]
    fn isotonic_regression_pools_a_violating_pair() {
        let points = fit_isotonic(&[(0.2, true), (0.4, false), (0.8, true)]);
        assert_eq!(points.len(), 3);
        assert!((points[0].probability - 0.5).abs() < 1e-12);
        assert!((points[1].probability - 0.5).abs() < 1e-12);
        assert!((points[2].probability - 1.0).abs() < 1e-12);
        for pair in points.windows(2) {
            assert!(pair[0].score < pair[1].score);
            assert!(pair[0].probability <= pair[1].probability);
        }
    }

    #[test]
    fn isotonic_regression_groups_equal_scores_and_is_deterministic() {
        let samples = [(0.3, false), (0.3, true), (0.7, true), (0.9, false)];
        let first = fit_isotonic(&samples);
        assert_eq!(first, fit_isotonic(&samples));
        assert_eq!(first.len(), 3);
        assert!((first[0].probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn calibration_is_monotone_and_clamps_outside_breakpoints() {
        let points = fit_isotonic(&[(0.1, false), (0.5, true), (0.9, true)]);
        let low = calibrate(&points, -1.0);
        let high = calibrate(&points, 2.0);
        assert_eq!(low, 0.0);
        assert_eq!(high, 1.0);
        let middle = calibrate(&points, 0.5);
        assert!((middle - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_odds_fusion_uses_likelihood_ratios_not_weighted_average() {
        let profile = profile();
        let fused = profile.fuse(&[("intent", 0.9), ("evidence", 1.0), ("margin", 1.0)]);
        // 三个来源都在先验之上，池化后必须高于任何单个来源的校准值。
        assert!(fused.probability > 0.95);
        assert_eq!(fused.contributions.len(), 3);
        assert!(fused.contributions.iter().all(|c| c.raw.is_some()));
    }

    #[test]
    fn missing_source_is_skipped_instead_of_counted_as_zero() {
        let profile = profile();
        let full = profile.fuse(&[("intent", 0.9), ("evidence", 1.0), ("margin", 1.0)]);
        let partial = profile.fuse(&[("intent", 0.9), ("evidence", 1.0)]);
        assert!(partial.probability < full.probability);
        let margin = partial
            .contributions
            .iter()
            .find(|contribution| contribution.name == "margin")
            .unwrap();
        assert_eq!(margin.raw, None);
        assert_eq!(margin.log_odds, 0.0);
    }

    #[test]
    fn near_boundary_evidence_degrades_a_confident_answer_to_hedge() {
        let profile = profile();
        let confident = profile.decide(
            0.9,
            &[("intent", 0.9), ("evidence", 1.0), ("margin", 1.0)],
            false,
        );
        assert_eq!(confident.decision, ConfidenceDecision::Answer);
        let boundary = profile.decide(
            0.9,
            &[("intent", 0.9), ("evidence", 0.55), ("margin", 0.0)],
            true,
        );
        assert_eq!(boundary.decision, ConfidenceDecision::Hedge);
    }

    #[test]
    fn low_understanding_always_clarifies() {
        let profile = profile();
        let record = profile.decide(
            0.2,
            &[("intent", 0.2), ("evidence", 1.0), ("margin", 1.0)],
            false,
        );
        assert_eq!(record.decision, ConfidenceDecision::Clarify);
    }

    #[test]
    fn two_vocabularies_do_not_mix() {
        assert_eq!(understanding_wording(0.95), "很有把握");
        assert_eq!(proposition_wording(0.95), "极有可能");
        assert_ne!(
            understanding_wording(0.8),
            proposition_wording(0.8),
            "理解置信度与命题概率必须各有词表"
        );
        for probability in [0.0, 0.1, 0.4, 0.55, 0.7, 0.8, 0.95, 1.0] {
            assert_eq!(
                proposition_wording(probability),
                crate::markov_logic::MarkovLogicEngine::hedge(probability)
            );
        }
    }

    #[test]
    fn clopper_pearson_upper_matches_the_reference_counts() {
        let upper = clopper_pearson_upper(240, 4, 0.05);
        assert!((upper - 0.037731613).abs() < 1e-6, "{}", upper);
        let zero = clopper_pearson_upper(10, 0, 0.05);
        assert!((zero - 0.258865551).abs() < 1e-6, "{}", zero);
        assert!((clopper_pearson_upper(10, 10, 0.05) - 1.0).abs() < 1e-12);
        assert!(clopper_pearson_upper(100, 5, 0.05) > clopper_pearson_upper(100, 4, 0.05));
    }

    #[test]
    fn certificate_verifies_and_rejects_a_stale_claim() {
        let certificate = profile().certificate.clone().unwrap();
        let status = certificate.verify().unwrap();
        assert!(status.verified);
        assert!(status.upper_bound <= status.max_error_rate);
        let mut stale = certificate.clone();
        stale.claimed_upper_bound = 0.01;
        assert!(stale.verify().is_err());
        let mut too_risky = certificate;
        too_risky.max_error_rate = 0.02;
        assert!(too_risky.verify().is_err());
    }

    #[test]
    fn validation_rejects_threshold_inversion_and_broken_calibration() {
        let mut broken = profile();
        broken.clarify_threshold = 0.8;
        broken.sources[0].calibration[1].probability = 0.3;
        let issues = broken.validate();
        assert!(issues
            .iter()
            .any(|issue| issue.code == "confidence_threshold_order"));
        assert!(issues
            .iter()
            .any(|issue| issue.code == "confidence_calibration_non_monotone"));
    }

    #[test]
    fn enabled_profile_without_certificate_is_an_error() {
        let mut broken = profile();
        broken.certificate = None;
        let issues = broken.validate();
        assert!(issues
            .iter()
            .any(|issue| issue.code == "confidence_certificate_missing"));
    }

    #[test]
    fn disabled_default_profile_keeps_legacy_behavior() {
        let profile = ConfidenceConfig::default();
        assert!(!profile.enabled);
        assert!(profile.validate().is_empty());
    }

    #[test]
    fn graph_envelope_asserts_but_conflicts_lower_the_ceiling() {
        let graph = ClaimEnvelope::from_grade(EnvelopeGrade::Graph, 0, 0);
        assert_eq!((graph.lower_bps(), graph.upper_bps()), (9_000, 10_000));
        assert_eq!(graph.verdict(8_600, 6_200), EnvelopeVerdict::Assert);
        assert_eq!(graph.deliberation_priority(8_600, 6_200), 0);

        let conflicted = ClaimEnvelope::from_grade(EnvelopeGrade::Graph, 0, 1);
        assert_eq!((conflicted.lower_bps(), conflicted.upper_bps()), (6_300, 7_000));
        assert_eq!(conflicted.verdict(8_600, 6_200), EnvelopeVerdict::Qualify);
        assert!(conflicted.deliberation_priority(8_600, 6_200) > 0);

        let contested = ClaimEnvelope::from_grade(EnvelopeGrade::Graph, 0, 2);
        assert!(contested.upper_bps() < 6_200);
        assert_eq!(contested.verdict(8_600, 6_200), EnvelopeVerdict::Abstain);
    }

    #[test]
    fn corroboration_only_lowers_uncertainty_and_grade_order_is_visible() {
        let base = ClaimEnvelope::from_grade(EnvelopeGrade::Graph, 0, 0);
        let supported = ClaimEnvelope::from_grade(EnvelopeGrade::Graph, 1, 0);
        assert!(supported.lower_bps() > base.lower_bps());
        assert_eq!(supported.upper_bps(), base.upper_bps());

        let direct = ClaimEnvelope::from_grade(EnvelopeGrade::SessionDirect, 0, 0);
        let correction = ClaimEnvelope::from_grade(EnvelopeGrade::SessionCorrection, 0, 0);
        let hearsay = ClaimEnvelope::from_grade(EnvelopeGrade::SessionHearsay, 0, 0);
        assert!(direct.lower_bps() > hearsay.lower_bps());
        assert!(correction.lower_bps() > direct.lower_bps());
        assert_eq!(direct.verdict(8_600, 6_200), EnvelopeVerdict::Verify);
        assert_eq!(hearsay.verdict(8_600, 6_200), EnvelopeVerdict::Verify);
    }

    #[test]
    fn point_envelopes_are_exact_and_priority_prefers_answer_crossing() {
        let point = ClaimEnvelope::point(0.9);
        assert!(point.is_point());
        assert_eq!(point.verdict(8_600, 6_200), EnvelopeVerdict::Assert);
        assert_eq!(point.deliberation_priority(8_600, 6_200), 0);

        let crossing = ClaimEnvelope::new(6_000, 9_000);
        let below = ClaimEnvelope::new(5_000, 8_000);
        assert!(crossing.crosses(8_600));
        assert!(!crossing.crosses(9_500));
        assert!(
            crossing.deliberation_priority(8_600, 6_200)
                > below.deliberation_priority(8_600, 6_200)
        );
    }

    #[test]
    fn envelope_wire_rejects_inverted_or_out_of_range_bounds() {
        let ok: ClaimEnvelope =
            serde_json::from_str(r#"{"lower_bps":6000,"upper_bps":9000}"#).unwrap();
        assert_eq!((ok.lower_bps(), ok.upper_bps()), (6_000, 9_000));
        assert!(serde_json::from_str::<ClaimEnvelope>(
            r#"{"lower_bps":9000,"upper_bps":6000}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ClaimEnvelope>(
            r#"{"lower_bps":1000,"upper_bps":10001}"#
        )
        .is_err());
    }

    #[test]
    fn configured_thresholds_drive_envelope_wording() {
        let profile = profile();
        let (answer, hedge) = profile.envelope_thresholds_bps();
        assert_eq!((answer, hedge), (8_600, 6_200));
        let envelope = ClaimEnvelope::new(6_300, 9_000);
        assert_eq!(profile.envelope_verdict(envelope), EnvelopeVerdict::Qualify);
        let text = profile.render_envelope(envelope);
        assert!(text.contains("63–90%"), "{}", text);
        assert!(text.contains("需限定"), "{}", text);
        let below = ClaimEnvelope::new(6_000, 9_000);
        assert_eq!(profile.envelope_verdict(below), EnvelopeVerdict::Verify);
    }
}

// 实体身份层：稳定身份（URI）× 指称解析（命名实体识别）的符号基础。
//
// 本模块不依赖任何神经网络，为纯符号系统提供两件 LLM 的核心装备：
//   1) 稳定身份：`entity_uri` 把规范名确定性编码成合法 IRI（本地名可无损反解），
//      身份注册表把「别名 / 曾用名 / 合并」记成显式台账，绝不静默改写；
//   2) 组合切分：指称栈（对话焦点 + 类型）上的代词消解与后缀类型推断，
//      为实体链接的全局解码（见 entity_linker::link_lattice）提供类型闸门。
//
// 设计边界：
// - EntityId 就是与图谱键一致的规范名（不做不透明 ID、不做全库 IRI 重写）；
// - 编码只转义 IRI 路径段中非法的字符，汉字与常见标点原样保留，可无损解析现有数据；
// - 代词消解「宁缺毋滥」：类型不匹配或证据不足时原样让位，绝不乱猜。

use std::collections::{BTreeMap, BTreeSet};

/// 默认 RDF 命名空间（与 lib.rs 的 BASE_NS 一致，单一正本）。
pub const DEFAULT_BASE_NS: &str = "http://example.org/kg/";

/// 实体 IRI 的路径前缀：`{base}entity/{编码后的规范名}`。
pub const ENTITY_PATH: &str = "entity/";

/// IRI 路径段中必须百分号转义的字符（其余字符，含汉字，按 IRI 规范原样保留）。
fn needs_escape(ch: char) -> bool {
    ch.is_control()
        || matches!(
            ch,
            ' ' | '/'
                | '?'
                | '#'
                | '%'
                | '"'
                | '<'
                | '>'
                | '\\'
                | '^'
                | '`'
                | '{'
                | '}'
                | '|'
                | '['
                | ']'
        )
}

/// 把本地名编码为 IRI 路径段：仅转义 `needs_escape` 的字符，每个字符按其 UTF-8
/// 字节逐字节百分号编码（大写十六进制）。编码确定性且可无损反解（`decode_segment`）。
pub fn encode_segment(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if needs_escape(ch) {
            let mut buf = [0u8; 4];
            for byte in ch.encode_utf8(&mut buf).as_bytes() {
                out.push('%');
                out.push_str(&format!("{:02X}", byte));
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// `encode_segment` 的逆：非法转义或非 UTF-8 字节序列返回 `None`（不静默容错）。
pub fn decode_segment(segment: &str) -> Option<String> {
    let bytes = segment.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let text = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(text, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// 规范名 → 实体 URI（确定性，可无损反解）。
pub fn entity_uri(base_ns: &str, name: &str) -> String {
    format!("{}{}{}", base_ns, ENTITY_PATH, encode_segment(name))
}

/// 实体 URI → 本地名：要求 `{base}entity/` 前缀，解码失败返回 `None`。
pub fn entity_local_name(base_ns: &str, iri: &str) -> Option<String> {
    let rest = iri.strip_prefix(base_ns)?;
    let local = rest.strip_prefix(ENTITY_PATH)?;
    decode_segment(local)
}

/// 任意 kg 本地名（含谓词/类型 IRI）：去掉 `{base}` 前缀后解码。
pub fn local_name(base_ns: &str, iri: &str) -> Option<String> {
    let rest = iri.strip_prefix(base_ns)?;
    let local = rest.strip_prefix(ENTITY_PATH).unwrap_or(rest);
    decode_segment(local)
}

// ---------------------------------------------------------------------------
// 身份注册表：规范名 ↔ 别名 ↔ 重定向（合并的显式台账）
// ---------------------------------------------------------------------------

/// 一次合并的编年记录（可审计：谁并入谁、搬了多少词面）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRecord {
    pub from: String,
    pub into: String,
    pub moved_aliases: usize,
    pub moved_redirects: usize,
}

/// 词面解析结论（`resolve` 的返回值，绝不猜测）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// 唯一规范身份
    Canonical(String),
    /// 经重定向（曾用名/已合并名）解析到规范身份
    Redirected { from: String, to: String },
    /// 多个规范身份（歧义：需上下文或澄清）
    Ambiguous(Vec<String>),
    /// 未登记
    Unknown,
}

impl Resolution {
    /// 解析到的规范身份（歧义/未知时为 None）。
    pub fn canonical(&self) -> Option<&str> {
        match self {
            Resolution::Canonical(id) => Some(id),
            Resolution::Redirected { to, .. } => Some(to),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Resolution::Canonical(id) => format!("规范身份「{}」", id),
            Resolution::Redirected { from, to } => {
                format!("重定向「{}」→「{}」", from, to)
            }
            Resolution::Ambiguous(ids) => format!("歧义（{}）", ids.join(" / ")),
            Resolution::Unknown => "未登记".to_string(),
        }
    }
}

/// 身份注册表：规范身份集合 + 别名表 + 重定向表 + 合并编年史。
///
/// 不变式：
/// - 别名词面永远指向规范名，且别名不静默移除（合并时整体搬运并记账）；
/// - 重定向链无环（`merge` 检出环即报错，不写入）；
/// - 所有遍历按字典序，审计输出确定可 golden。
#[derive(Debug, Clone, Default)]
pub struct IdentityRegistry {
    canonicals: BTreeSet<String>,
    aliases: BTreeMap<String, BTreeSet<String>>,
    redirects: BTreeMap<String, String>,
    merges: Vec<MergeRecord>,
}

impl IdentityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.canonicals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.canonicals.is_empty()
    }

    /// 登记一个规范身份；空名忽略。返回是否为新增。
    pub fn register(&mut self, name: &str) -> bool {
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        self.canonicals.insert(name.to_string())
    }

    pub fn is_known(&self, name: &str) -> bool {
        !matches!(self.resolve(name), Resolution::Unknown)
    }

    /// 登记别名 → 规范身份（规范身份不存在时一并登记）。
    ///
    /// 返回是否为新增词面；空名、自别名、指向歧义身份的别名均拒绝。
    pub fn register_alias(&mut self, surface: &str, canonical: &str) -> Result<bool, String> {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            return Err("别名与规范名都不能为空".to_string());
        }
        if surface == canonical {
            return Ok(false);
        }
        // 目标是重定向/别名时解析到最终规范名；歧义目标拒绝（不许把歧义写死）
        let target = match self.resolve(canonical) {
            Resolution::Canonical(id) => id,
            Resolution::Redirected { to, .. } => to,
            Resolution::Ambiguous(ids) => {
                return Err(format!(
                    "别名目标「{}」本身歧义（{}）",
                    canonical,
                    ids.join(" / ")
                ));
            }
            Resolution::Unknown => {
                self.register(canonical);
                canonical.to_string()
            }
        };
        if surface == target {
            return Ok(false);
        }
        // 词面若已是规范身份：别名会让它有两读，允许登记，resolve 如实返回歧义
        let bucket = self.aliases.entry(surface.to_string()).or_default();
        Ok(bucket.insert(target))
    }

    /// 合并身份：`from` 并入 `into`。
    ///
    /// - 检出环（into 经重定向链回到 from）或自合并即报错，不写入；
    /// - `from` 的全部别名搬到 `into`，指向 `from` 的重定向改指 `into`；
    /// - 写入重定向 `from → into` 并记一条合并编年。
    pub fn merge(&mut self, from: &str, into: &str) -> Result<MergeRecord, String> {
        let from = from.trim();
        let into = into.trim();
        if from.is_empty() || into.is_empty() {
            return Err("合并双方都不能为空".to_string());
        }
        if from == into {
            return Err("不能把身份并入自身".to_string());
        }
        // 目标是重定向时解析到最终身份；目标未知则先登记
        let into_final = match self.resolve(into) {
            Resolution::Canonical(id) => id,
            Resolution::Redirected { to, .. } => to,
            Resolution::Ambiguous(ids) => {
                return Err(format!("合并目标「{}」歧义（{}）", into, ids.join(" / ")));
            }
            Resolution::Unknown => {
                self.register(into);
                into.to_string()
            }
        };
        if from == into_final {
            return Err("不能把身份并入自身".to_string());
        }
        // 环检测：沿 into 的重定向链前进，若回到 from 则拒绝
        let mut cursor = into_final.clone();
        let mut seen = BTreeSet::new();
        while seen.insert(cursor.clone()) {
            match self.redirects.get(&cursor) {
                Some(next) => {
                    if next == from {
                        return Err(format!("合并会形成重定向环：{} → … → {}", into_final, from));
                    }
                    cursor = next.clone();
                }
                None => break,
            }
        }

        // 别名搬迁：所有指向 from 的词面改指 into_final
        let mut moved_aliases = 0usize;
        for targets in self.aliases.values_mut() {
            if targets.remove(from) {
                targets.insert(into_final.clone());
                moved_aliases += 1;
            }
        }
        // from 自己的别名（surface → from，含 from 作为词面）整体并入 into
        if let Some(own) = self.aliases.remove(from) {
            for surface in own {
                moved_aliases += 1;
                self.aliases
                    .entry(surface)
                    .or_default()
                    .insert(into_final.clone());
            }
        }
        // 指向 from 的重定向改指 into_final
        let mut moved_redirects = 0usize;
        for target in self.redirects.values_mut() {
            if target == from {
                *target = into_final.clone();
                moved_redirects += 1;
            }
        }
        self.redirects.insert(from.to_string(), into_final.clone());
        self.canonicals.remove(from);
        self.aliases.remove(from);
        let record = MergeRecord {
            from: from.to_string(),
            into: into_final,
            moved_aliases,
            moved_redirects,
        };
        self.merges.push(record.clone());
        Ok(record)
    }

    /// 解析词面：重定向 → 别名（唯一/歧义）→ 规范名 → 未登记。
    pub fn resolve(&self, surface: &str) -> Resolution {
        let surface = surface.trim();
        if surface.is_empty() {
            return Resolution::Unknown;
        }
        if let Some(to) = self.redirects.get(surface) {
            // 重定向链由 merge 保证无环；仍加 seen 防护，异常时按未登记处理
            let mut cursor = to.clone();
            let mut seen = BTreeSet::new();
            seen.insert(surface.to_string());
            while seen.insert(cursor.clone()) {
                match self.redirects.get(&cursor) {
                    Some(next) => cursor = next.clone(),
                    None => break,
                }
            }
            return Resolution::Redirected {
                from: surface.to_string(),
                to: cursor,
            };
        }
        let mut candidates: BTreeSet<String> = BTreeSet::new();
        if let Some(targets) = self.aliases.get(surface) {
            candidates.extend(targets.iter().cloned());
        }
        if self.canonicals.contains(surface) {
            candidates.insert(surface.to_string());
        }
        match candidates.len() {
            0 => Resolution::Unknown,
            1 => Resolution::Canonical(candidates.into_iter().next().expect("非空")),
            _ => Resolution::Ambiguous(candidates.into_iter().collect()),
        }
    }

    /// 某规范身份的全部可解析词面（别名 + 指向它的曾用名），字典序。
    pub fn surfaces_of(&self, id: &str) -> Vec<String> {
        let mut surfaces: BTreeSet<String> = BTreeSet::new();
        for (surface, targets) in &self.aliases {
            if targets.contains(id) {
                surfaces.insert(surface.clone());
            }
        }
        for (surface, target) in &self.redirects {
            if target == id {
                surfaces.insert(surface.clone());
            }
        }
        surfaces.into_iter().collect()
    }

    pub fn redirects(&self) -> Vec<(String, String)> {
        self.redirects
            .iter()
            .map(|(from, to)| (from.clone(), to.clone()))
            .collect()
    }

    pub fn merges(&self) -> &[MergeRecord] {
        &self.merges
    }

    /// 确定性审计视图：身份、别名、重定向、合并编年史。
    pub fn audit_lines(&self) -> Vec<String> {
        let alias_count: usize = self.aliases.values().map(BTreeSet::len).sum();
        let mut lines = vec![format!(
            "身份注册表：{} 个规范身份 / {} 条别名词面 / {} 条重定向 / {} 次合并",
            self.canonicals.len(),
            alias_count,
            self.redirects.len(),
            self.merges.len()
        )];
        for id in &self.canonicals {
            let surfaces = self.surfaces_of(id);
            if surfaces.is_empty() {
                lines.push(format!("  • {}（无别名）", id));
            } else {
                lines.push(format!("  • {} ← {}", id, surfaces.join("、")));
            }
        }
        for (from, to) in &self.redirects {
            lines.push(format!("  ↪ {} → {}", from, to));
        }
        for record in &self.merges {
            lines.push(format!(
                "  ▸ 合并：{} → {}（搬迁别名 {} 条、重定向 {} 条）",
                record.from, record.into, record.moved_aliases, record.moved_redirects
            ));
        }
        lines
    }
}

// ---------------------------------------------------------------------------
// 代词类型闸门：指称栈上的「类型过滤 + 显著度排序」依据
// ---------------------------------------------------------------------------

/// 代词的指称类型过滤条件。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PronounFilter {
    /// 只接受人物类型（「他 / 她 / 他们 / 这个人」）
    Person,
    /// 只接受已知的非人物类型（「它 / 它们」）：类型未知时不猜
    NonPerson,
    /// 只接受列出的类型关键词（「该公司」→ 公司/组织）
    Types(&'static [&'static str]),
}

fn type_hits(etype: &str, keyword: &str) -> bool {
    etype.contains(keyword) || etype.to_lowercase().contains(&keyword.to_lowercase())
}

/// 类型证据是否满足代词过滤（`etype` 未知时只有 Person 开放世界放行）。
pub fn type_allows(etype: Option<&str>, filter: PronounFilter) -> bool {
    match filter {
        // 开放世界：人物代词在类型未知时仍可解析（但绝不解析出非人物候选）
        PronounFilter::Person => match etype {
            None => true,
            Some(etype) => {
                type_hits(etype, "人") || type_hits(etype, "person") || type_hits(etype, "人物")
            }
        },
        // 类型未知不猜：只有确证「非人物」才允许「它/它们」指代
        PronounFilter::NonPerson => match etype {
            None => false,
            Some(etype) => !type_hits(etype, "人") && !type_hits(etype, "person"),
        },
        PronounFilter::Types(keywords) => match etype {
            None => false,
            Some(etype) => keywords.iter().any(|keyword| type_hits(etype, keyword)),
        },
    }
}

/// 代词的过滤条件（未列出的词面返回 None = 不是需消解的代词）。
pub fn pronoun_filter(pronoun: &str) -> Option<PronounFilter> {
    match pronoun {
        "他" | "她" | "他们" | "她们" | "这个人" => Some(PronounFilter::Person),
        "它" | "它们" => Some(PronounFilter::NonPerson),
        "该公司" | "这家公司" | "那家公司" | "该企业" => {
            Some(PronounFilter::Types(&[
                "公司",
                "组织",
                "organization",
                "企业",
            ]))
        }
        _ => None,
    }
}

/// 序数代词的焦点下标：「前者」= 最近一次提到的第一个实体，「后者」= 第二个。
pub fn ordinal_pronoun_index(pronoun: &str) -> Option<usize> {
    match pronoun {
        "前者" => Some(0),
        "后者" => Some(1),
        _ => None,
    }
}

/// 需消解的代词表（长词面优先，避免「他们」被「他」截断）。
pub const PRONOUNS: [&str; 13] = [
    "这家公司",
    "那家公司",
    "这个人",
    "该公司",
    "该企业",
    "他们",
    "她们",
    "它们",
    "前者",
    "后者",
    "他",
    "她",
    "它",
];

// ---------------------------------------------------------------------------
// OOV 后缀 → 暂定类型：提名节点只加类型标注，未经准入不进知识库
// ---------------------------------------------------------------------------

/// 实体后缀 → 暂定类型（只做提名标注；确认后由 `admit` 显式准入）。
pub const SUFFIX_TYPES: [(&str, &str); 24] = [
    ("公司", "组织"),
    ("集团", "组织"),
    ("大学", "组织"),
    ("学院", "组织"),
    ("医院", "组织"),
    ("银行", "组织"),
    ("博物馆", "组织"),
    ("图书馆", "组织"),
    ("研究院", "组织"),
    ("委员会", "组织"),
    ("政府", "组织"),
    ("法院", "组织"),
    ("城市", "地点"),
    ("自治区", "地点"),
    ("研究生", "人物"),
    ("学生", "人物"),
    ("老师", "人物"),
    ("教师", "人物"),
    ("教授", "人物"),
    ("博士", "人物"),
    ("医生", "人物"),
    ("律师", "人物"),
    ("工程师", "人物"),
    ("员工", "人物"),
];

/// 按后缀表推断 OOV 词的暂定类型，返回（命中的后缀，暂定类型）。
pub fn infer_suffix_type(text: &str) -> Option<(&'static str, &'static str)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    SUFFIX_TYPES
        .iter()
        .find(|(suffix, _)| text.ends_with(suffix))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_roundtrip_is_lossless_for_awkward_names() {
        for name in [
            "张三",
            "北京/大学",
            "ACME 公司",
            "name#tag",
            "100%纯",
            "a?b[c]d",
            "emoji😀名",
            "普通-name_1.0~",
        ] {
            let uri = entity_uri(DEFAULT_BASE_NS, name);
            assert!(
                uri.starts_with("http://example.org/kg/entity/"),
                "URI 前缀错误：{}",
                uri
            );
            assert!(!uri.contains(' '), "URI 不允许空格：{}", uri);
            assert_eq!(
                entity_local_name(DEFAULT_BASE_NS, &uri).as_deref(),
                Some(name),
                "往返必须无损：{}",
                uri
            );
        }
    }

    #[test]
    fn uri_keeps_han_characters_verbatim_for_backward_compatibility() {
        assert_eq!(
            entity_uri(DEFAULT_BASE_NS, "张三"),
            "http://example.org/kg/entity/张三"
        );
        assert_eq!(
            entity_uri(DEFAULT_BASE_NS, "北京/大学"),
            "http://example.org/kg/entity/北京%2F大学"
        );
        assert_eq!(
            entity_uri(DEFAULT_BASE_NS, "ACME 公司"),
            "http://example.org/kg/entity/ACME%20公司"
        );
    }

    #[test]
    fn local_name_handles_entity_and_plain_paths() {
        assert_eq!(
            local_name(DEFAULT_BASE_NS, "http://example.org/kg/teacher_of").as_deref(),
            Some("teacher_of")
        );
        assert_eq!(
            local_name(DEFAULT_BASE_NS, "http://example.org/kg/entity/李四").as_deref(),
            Some("李四")
        );
        assert_eq!(local_name(DEFAULT_BASE_NS, "http://other.org/x"), None);
        assert_eq!(
            entity_local_name(DEFAULT_BASE_NS, "http://example.org/kg/name"),
            None,
            "非实体路径不得被当成实体本地名"
        );
    }

    #[test]
    fn decode_rejects_bad_escapes() {
        assert_eq!(decode_segment("%2F"), Some("/".to_string()));
        assert_eq!(decode_segment("%ZZ"), None);
        assert_eq!(decode_segment("%E4"), None, "截断的 UTF-8 序列应报错");
        assert_eq!(decode_segment("abc%"), None);
    }

    #[test]
    fn registry_register_alias_and_resolve() {
        let mut registry = IdentityRegistry::new();
        assert!(registry.register("苹果公司"));
        assert!(!registry.register("苹果公司"), "重复登记返回 false");
        assert!(registry.register_alias("苹果", "苹果公司").unwrap());
        assert_eq!(
            registry.resolve("苹果"),
            Resolution::Canonical("苹果公司".to_string())
        );
        assert_eq!(
            registry.resolve("苹果公司"),
            Resolution::Canonical("苹果公司".to_string())
        );
        assert_eq!(registry.resolve("香蕉"), Resolution::Unknown);
        assert_eq!(registry.surfaces_of("苹果公司"), vec!["苹果".to_string()]);
    }

    #[test]
    fn registry_alias_to_ambiguous_target_is_rejected() {
        let mut registry = IdentityRegistry::new();
        registry.register_alias("苹果", "苹果公司").unwrap();
        registry.register_alias("苹果", "苹果果实").unwrap();
        assert_eq!(
            registry.resolve("苹果"),
            Resolution::Ambiguous(vec!["苹果公司".to_string(), "苹果果实".to_string()])
        );
        assert!(registry.register_alias("红苹果", "苹果").is_err());
    }

    #[test]
    fn registry_merge_moves_aliases_and_records_redirect() {
        let mut registry = IdentityRegistry::new();
        registry.register("苹果股份有限公司");
        registry.register_alias("苹果", "苹果股份有限公司").unwrap();
        let record = registry.merge("苹果股份有限公司", "苹果公司").unwrap();
        assert_eq!(record.moved_aliases, 1);
        assert_eq!(
            registry.resolve("苹果股份有限公司"),
            Resolution::Redirected {
                from: "苹果股份有限公司".to_string(),
                to: "苹果公司".to_string(),
            }
        );
        assert_eq!(
            registry.resolve("苹果"),
            Resolution::Canonical("苹果公司".to_string()),
            "被合并词面应随合并解析为新身份"
        );
        assert_eq!(
            registry.surfaces_of("苹果公司"),
            vec!["苹果".to_string(), "苹果股份有限公司".to_string()]
        );
        assert_eq!(registry.merges().len(), 1);
    }

    #[test]
    fn registry_merge_cycle_and_self_are_rejected() {
        let mut registry = IdentityRegistry::new();
        registry.register("甲");
        registry.register("乙");
        registry.merge("甲", "乙").unwrap();
        assert!(registry.merge("乙", "甲").is_err(), "成环必须拒绝");
        assert!(registry.merge("甲", "甲").is_err(), "自合并必须拒绝");
        assert_eq!(registry.redirects().len(), 1, "失败合并不写入");
    }

    #[test]
    fn registry_audit_lines_are_deterministic() {
        let mut registry = IdentityRegistry::new();
        registry.register("乙公司");
        registry.register("甲公司");
        registry.register_alias("甲", "甲公司").unwrap();
        registry.merge("乙公司", "甲公司").unwrap();
        let lines = registry.audit_lines();
        assert_eq!(lines, registry.audit_lines());
        assert!(lines[0].contains("身份注册表"));
        assert!(lines
            .iter()
            .any(|line| line.contains("甲公司 ← 乙公司、甲")));
        assert!(lines
            .iter()
            .any(|line| line.contains("合并：乙公司 → 甲公司")));
    }

    #[test]
    fn pronoun_filters_are_conservative() {
        assert!(type_allows(Some("人"), PronounFilter::Person));
        assert!(type_allows(Some("Person"), PronounFilter::Person));
        assert!(type_allows(None, PronounFilter::Person), "开放世界人物放行");
        assert!(!type_allows(Some("公司"), PronounFilter::Person));
        assert!(
            !type_allows(None, PronounFilter::NonPerson),
            "未知类型不猜「它」"
        );
        assert!(type_allows(Some("公司"), PronounFilter::NonPerson));
        assert!(!type_allows(Some("人"), PronounFilter::NonPerson));
        let org = PronounFilter::Types(&["公司", "组织"]);
        assert!(type_allows(Some("科技公司"), org));
        assert!(type_allows(Some("组织"), org));
        assert!(!type_allows(Some("人"), org));
        assert!(!type_allows(None, org));
        assert_eq!(pronoun_filter("他"), Some(PronounFilter::Person));
        assert_eq!(pronoun_filter("它"), Some(PronounFilter::NonPerson));
        assert_eq!(pronoun_filter("前者"), None, "序数代词走焦点下标");
        assert_eq!(ordinal_pronoun_index("前者"), Some(0));
        assert_eq!(ordinal_pronoun_index("后者"), Some(1));
    }

    #[test]
    fn suffix_type_inference() {
        assert_eq!(infer_suffix_type("未名科技公司"), Some(("公司", "组织")));
        assert_eq!(infer_suffix_type("未名研究院"), Some(("研究院", "组织")));
        assert_eq!(infer_suffix_type("王老师"), Some(("老师", "人物")));
        assert_eq!(infer_suffix_type("新生学生"), Some(("学生", "人物")));
        assert_eq!(infer_suffix_type("某某城市"), Some(("城市", "地点")));
        assert_eq!(infer_suffix_type("张三"), None);
        assert_eq!(infer_suffix_type("  "), None);
    }

    #[test]
    fn pronoun_table_is_longer_first() {
        for window in PRONOUNS.windows(2) {
            assert!(
                window[0].chars().count() >= window[1].chars().count(),
                "代词表必须长词面优先：{} 在 {} 之前",
                window[0],
                window[1]
            );
        }
    }
}

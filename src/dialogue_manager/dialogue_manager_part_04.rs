impl DialogueManager {
    /// 模板接管：Parity 双跑计数、On 用模板文本、接地失败回退组合生成器文本。
    fn apply_prompt_spec(
        &mut self,
        policy: &str,
        frame: &UtteranceFrame,
        entity: Option<&str>,
        answer: Option<&str>,
        more: &[String],
        composed: String,
    ) -> String {
        if self.prompt_mode == PromptMode::Off {
            return composed;
        }
        let Some((spec_id, bindings)) = self.prompt_bindings(policy, frame, entity, answer, more)
        else {
            self.prompt_stats.composed += 1;
            return composed;
        };
        self.prompt_stats.checked += 1;
        match prompt_spec::render(spec_id, &bindings) {
            RenderOutcome::Rendered { text, .. } => {
                self.prompt_stats.grounded += 1;
                if text != composed {
                    self.prompt_stats.parity_mismatches += 1;
                    self.prompt_stats.record_mismatch(format!(
                        "{}：模板与组合生成器文本不一致\n    模板: {}\n    组合: {}",
                        spec_id,
                        text.replace('\n', "⏎"),
                        composed.replace('\n', "⏎")
                    ));
                }
                match self.prompt_mode {
                    PromptMode::On => text,
                    _ => composed,
                }
            }
            RenderOutcome::Ungrounded { missing, .. } => {
                self.prompt_stats.ungrounded += 1;
                self.prompt_stats.record_missing(&missing);
                composed
            }
            RenderOutcome::Invalid { message, .. } => {
                self.prompt_stats.ungrounded += 1;
                self.prompt_stats
                    .record_mismatch(format!("{} 资产无效：{}", spec_id, message));
                composed
            }
        }
    }

    fn greet_text(&self) -> String {
        if self.turns.is_empty() {
            "你好！我是 KnowProlog 的符号主义对话层：知识来自图谱，推理依靠规则与逻辑程序，回答由模板组合生成。你可以直接问「姚明的身高是多少」这样的问题。"
                .to_string()
        } else if let Some(focus) = self.focus.last() {
            format!("又见面了！我们接着聊「{}」也行，换个话题也行。", focus)
        } else {
            "又见面了！继续问我吧。".to_string()
        }
    }

    fn thanks_text(&self) -> String {
        match self.focus.last() {
            Some(focus) => format!("不客气！关于「{}」还有想问的，随时说。", focus),
            None => "不客气！随时继续问我。".to_string(),
        }
    }

    fn farewell_text(&self) -> String {
        format!(
            "好的，再见！这次我们聊了 {} 轮，对话本体已经记入记忆。",
            self.turns.len()
        )
    }
}
fn negative_correction_body(input: &str) -> Option<&str> {
    for prefix in ["不是", "不对", "错了"] {
        if let Some(rest) = input.strip_prefix(prefix) {
            let rest = rest.trim_start_matches(['，', ',', '：', ':', ' ']).trim();
            if !rest.is_empty() {
                return Some(rest);
            }
        }
    }
    None
}
fn raw_input(frame: &UtteranceFrame) -> String {
    frame.raw.trim().to_string()
}
/// 属性键的展示词面：英文内置键（name/age 等）本地化为中文，
/// 未知键保持原样（查询仍用原始键，只有展示走词面）。
fn surface_attribute(attribute: &str) -> &str {
    crate::symbolic_generator::localized_predicate_key(attribute).unwrap_or(attribute)
}
/// 把回答文本归类为可宣讲的推理出处（纯符号标记识别）。
fn classify_provenance(answer: &str) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if answer.contains("链接预测") || answer.contains("预测") {
        tags.push("知识图谱链接预测（符号统计）".to_string());
    }
    if answer.contains("溯因") || answer.contains("假设") {
        tags.push("溯因假设生成（会话级推测）".to_string());
    }
    if answer.contains("归结") || answer.contains("证明") {
        tags.push("一阶归结证明".to_string());
    }
    if answer.contains("本体") || answer.contains("继承") {
        tags.push("OWL 本体推理（层级闭包）".to_string());
    }
    if answer.contains("软推理") || answer.contains("加权") || answer.contains("概率") {
        tags.push("加权软逻辑 / 概率推理".to_string());
    }
    if answer.contains("规则") {
        tags.push("会话规则后向链".to_string());
    }
    if tags.is_empty() {
        tags.push("知识图谱中的直接事实".to_string());
    }
    tags
}
/// 解析「第 N 步」追问（阿拉伯数字与 1–99 的中文数字）。
fn parse_step_request(text: &str) -> Option<usize> {
    let start = text.find('第')?;
    let tail = &text[start + '第'.len_utf8()..];
    let end = tail.find('步')?;
    let raw = tail[..end].trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(step) = raw.parse::<usize>() {
        return (step > 0).then_some(step);
    }
    let digit = |ch: char| match ch {
        '零' => Some(0),
        '一' => Some(1),
        '二' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    };
    if let Some(pos) = raw.find('十') {
        let tens = if pos == 0 {
            1
        } else {
            raw[..pos].chars().next().and_then(digit)?
        };
        let ones = if pos + '十'.len_utf8() == raw.len() {
            0
        } else {
            raw[pos + '十'.len_utf8()..]
                .chars()
                .next()
                .and_then(digit)?
        };
        return Some(tens * 10 + ones).filter(|step| *step > 0);
    }
    let mut value = 0usize;
    for ch in raw.chars() {
        value = value * 10 + digit(ch)?;
    }
    (value > 0).then_some(value)
}
fn turn_question(turn: &DialogueTurn) -> Option<String> {
    match (&turn.entity, &turn.attribute, &turn.kind) {
        (Some(entity), Some(attribute), _) => Some(format!("{}的{}是什么", entity, attribute)),
        (Some(entity), None, MoveKind::AskIdentity) => Some(format!("{}是谁", entity)),
        _ => None,
    }
}
fn excerpt(text: &str, limit: usize) -> String {
    // 推理器的多行回答常以「根据知识图谱：」这类标题开头；摘要取第一条
    // 实质内容行，避免「依据」只剩标题、无法与答案取值对齐。
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let first_line = lines
        .iter()
        .copied()
        .find(|line| !line.is_empty() && !line.ends_with('：') && !line.ends_with(':'))
        .or_else(|| lines.iter().copied().find(|line| !line.is_empty()))
        .unwrap_or("");
    let mut out: String = first_line.chars().take(limit).collect();
    if first_line.chars().count() > limit {
        out.push('…');
    }
    out
}
fn contains_pronoun(text: &str) -> bool {
    let cleaned = text
        .replace("其他", "")
        .replace("其中", "")
        .replace("其实", "");
    [
        "他们",
        "她们",
        "它们",
        "这个人",
        "那位",
        "这位",
        "他",
        "她",
        "它",
    ]
    .iter()
    .any(|pronoun| cleaned.contains(pronoun))
}
/// 「A 是 B 的 C」：判断句把 C 作为 B 的属性来校验，而不是问 A 的属性。
fn is_copula_relation_statement(text: &str) -> bool {
    let text = text
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim_end_matches('吗')
        .trim();
    let Some((left, right)) = text.split_once('是') else {
        return false;
    };
    !left.trim().is_empty() && !left.contains('的') && right.contains('的')
}
/// 同样的形状，但左侧必须是已登记实体；用于拦截查询板的「B 的 C」重扫，
/// 避免把「列出身高是226的人」这类板查询也一并挡掉。
fn is_entity_copula_relation_statement(text: &str, knowledge: &dyn DialogueKnowledge) -> bool {
    if !is_copula_relation_statement(text) {
        return false;
    }
    let text = text
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim_end_matches('吗')
        .trim();
    let Some((left, _)) = text.split_once('是') else {
        return false;
    };
    !knowledge.entities_in(left.trim()).is_empty()
}
/// 地点类类型标签（大小写无关）：这些实体的「多大」默认问面积。
fn is_place_type(etype: &str) -> bool {
    matches!(
        etype.trim().to_ascii_lowercase().as_str(),
        "city"
            | "country"
            | "location"
            | "place"
            | "region"
            | "province"
            | "state"
            | "island"
            | "城市"
            | "国家"
            | "地点"
            | "地区"
            | "省份"
            | "省"
            | "岛屿"
            | "行政区"
    )
}
fn split_entity_attribute(text: &str) -> Option<(&str, &str)> {
    let index = text.find('的')?;
    let head = text[..index].trim();
    let mut tail = text[index + '的'.len_utf8()..].trim();
    // 口语姓名问句的「叫什么名字」不是属性词面：整句交给查询规划器
    // 编译成「…的姓名」链，避免被改写成「师傅叫什么名字是什么」。
    if ["叫什么名字", "叫啥名字", "叫什麼名字", "叫啥"]
        .iter()
        .any(|suffix| tail.contains(suffix))
    {
        return None;
    }
    // 连续剥离（「…到底是多少呢」需要先剥「呢」再剥「是多少」）。
    for _ in 0..3 {
        let mut stripped_once = false;
        for suffix in [
            "是什么",
            "是多少",
            "是谁",
            "是哪个",
            "是哪里",
            "怎么样",
            "有哪些",
            "有哪几个",
            "有哪几位",
            "有哪几种",
            "多少",
            "呢",
            "吗",
        ] {
            if let Some(stripped) = tail.strip_suffix(suffix) {
                tail = stripped.trim();
                stripped_once = true;
                break;
            }
        }
        if !stripped_once {
            break;
        }
    }
    // 语气/强调填充词不承载属性：先把「到底是谁」剥成「到底」，再归一成属性。
    for filler in ["到底", "究竟", "来着", "呀", "啊", "嘛"] {
        if let Some(stripped) = tail.strip_suffix(filler) {
            let stripped = stripped.trim();
            if !stripped.is_empty() {
                tail = stripped;
                break;
            }
        }
    }
    // 明写属性 + 程度词（面积多大 / 身高有多高）：属性词面优先于程度词，
    // 否则「多大」会被当成属性名，问题被改写成「面积多大是什么」。
    for marker in ["多大", "多高", "多重", "几岁"] {
        let Some(candidate) = tail.strip_suffix(marker) else {
            continue;
        };
        let candidate = candidate.trim().trim_end_matches('有').trim();
        if !candidate.is_empty()
            && (crate::symbolic_generator::canonical_predicate_key(candidate).is_some()
                || candidate == "年龄")
        {
            tail = candidate;
            break;
        }
    }
    if tail.is_empty() || tail.contains('是') || tail.chars().count() > 16 {
        return None;
    }
    Some((head, tail))
}
/// 构造 `LogicProgram` 时使用的辅助类型别名占位（保持模块内 API 稳定）。
pub type DialogueFact = LAtom;
/// 便捷构造：把实体断言为对话事实（供扩展使用）。
pub fn entity_fact(entity: &str) -> LAtom {
    LAtom::new("entity", vec![LTerm::atom(entity)])
}
#[cfg(test)]
mod tests {
    include!("dialogue_manager__tests_01.rs");
    include!("dialogue_manager__tests_02.rs");
}

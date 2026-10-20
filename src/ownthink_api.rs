use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const OWNTHINK_BASE_URL: &str = "https://api.ownthink.com/kg";

/// 复用同一个连接池与 TLS 上下文。
fn shared_http_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new).clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousRequest {
    pub mention: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousResponse {
    pub message: String,
    pub data: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeRequest {
    pub entity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResponse {
    pub message: String,
    pub data: KnowledgeData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeData {
    pub entity: String,
    pub desc: String,
    pub tag: Vec<String>,
    pub avp: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EavRequest {
    pub entity: String,
    pub attribute: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EavResponse {
    pub message: String,
    pub data: EavData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EavData {
    pub entity: String,
    pub attribute: String,
    pub value: Vec<String>,
}

pub struct OwnthinkApiClient {
    client: reqwest::Client,
    base_url: String,
    timeout_seconds: u64,
    max_retries: usize,
    retry_delay_ms: u64,
}

impl OwnthinkApiClient {
    pub fn new() -> Self {
        Self {
            client: shared_http_client(),
            base_url: OWNTHINK_BASE_URL.to_string(),
            timeout_seconds: 10,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }

    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    pub fn with_retries(mut self, max_retries: usize, retry_delay_ms: u64) -> Self {
        self.max_retries = max_retries;
        self.retry_delay_ms = retry_delay_ms;
        self
    }

    async fn make_request_with_retry<T, R>(
        &self,
        request_fn: impl Fn() -> R,
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
    where
        T: serde::de::DeserializeOwned,
        R: std::future::Future<Output = Result<reqwest::Response, reqwest::Error>>,
    {
        let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

        for attempt in 0..=self.max_retries {
            match request_fn().await {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        match response.json::<T>().await {
                            Ok(result) => return Ok(result),
                            Err(e) => {
                                last_error = Some(Box::new(e));
                                if attempt < self.max_retries {
                                    eprintln!(
                                        "JSON解析失败 (尝试 {}/{}), 将重试...",
                                        attempt + 1,
                                        self.max_retries + 1
                                    );
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        self.retry_delay_ms,
                                    ))
                                    .await;
                                    continue;
                                }
                            }
                        }
                    } else if status.is_server_error() || status == 429 {
                        // 服务器错误或限流，可以重试
                        last_error = Some(format!("HTTP错误: {}", status).into());
                        if attempt < self.max_retries {
                            eprintln!(
                                "服务器错误 {} (尝试 {}/{}), 将重试...",
                                status,
                                attempt + 1,
                                self.max_retries + 1
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(
                                self.retry_delay_ms * (attempt + 1) as u64,
                            ))
                            .await;
                            continue;
                        }
                    } else {
                        // 客户端错误，不重试
                        return Err(format!("HTTP错误: {}", status).into());
                    }
                }
                Err(e) => {
                    last_error = Some(Box::new(e));
                    if attempt < self.max_retries {
                        eprintln!(
                            "网络请求失败 (尝试 {}/{}), 将重试...",
                            attempt + 1,
                            self.max_retries + 1
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(
                            self.retry_delay_ms * (attempt + 1) as u64,
                        ))
                        .await;
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| "未知错误".into()))
    }

    /// 获取歧义关系 - 根据名称(mention)返回可能的实体列表
    pub async fn get_ambiguous_entities(
        &self,
        mention: &str,
    ) -> Result<AmbiguousResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/ambiguous", self.base_url);
        let request = AmbiguousRequest {
            mention: mention.to_string(),
        };

        let response: AmbiguousResponse = self
            .make_request_with_retry(|| {
                self.client
                    .post(&url)
                    .json(&request)
                    .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                    .send()
            })
            .await?;

        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 获取实体的完整知识
    pub async fn get_entity_knowledge(
        &self,
        entity: &str,
    ) -> Result<KnowledgeResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/knowledge", self.base_url);
        let request = KnowledgeRequest {
            entity: entity.to_string(),
        };

        let response: KnowledgeResponse = self
            .make_request_with_retry(|| {
                self.client
                    .post(&url)
                    .json(&request)
                    .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                    .send()
            })
            .await?;

        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 获取实体的特定属性值
    pub async fn get_entity_attribute_value(
        &self,
        entity: &str,
        attribute: &str,
    ) -> Result<EavResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/eav", self.base_url);
        let request = EavRequest {
            entity: entity.to_string(),
            attribute: attribute.to_string(),
        };

        let response: EavResponse = self
            .make_request_with_retry(|| {
                self.client
                    .post(&url)
                    .json(&request)
                    .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                    .send()
            })
            .await?;

        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 解析歧义实体 - 返回最佳匹配的实体
    ///
    /// 未提供上下文（或上下文无法区分）时选权重最高者；
    /// 提供上下文时用 [`select_candidate_by_context`] 做确定性消歧。
    pub async fn resolve_entity_ambiguity(
        &self,
        mention: &str,
        context: Option<&str>,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let ambiguous_response = self.get_ambiguous_entities(mention).await?;

        if ambiguous_response.data.is_empty() {
            return Ok(None);
        }

        Ok(
            select_candidate_by_context(&ambiguous_response.data, context)
                .map(|index| ambiguous_response.data[index].0.clone()),
        )
    }

    /// 获取实体的简化知识 - 转换为内部知识表示
    pub async fn get_simplified_knowledge(
        &self,
        entity: &str,
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error + Send + Sync>> {
        let knowledge_response = self.get_entity_knowledge(entity).await?;
        let mut properties = HashMap::new();

        // 添加基础信息
        properties.insert("name".to_string(), knowledge_response.data.entity.clone());
        properties.insert("description".to_string(), knowledge_response.data.desc);
        properties.insert("tags".to_string(), knowledge_response.data.tag.join(", "));

        // 添加属性值对
        for avp in &knowledge_response.data.avp {
            if avp.len() >= 2 {
                properties.insert(avp[0].clone(), avp[1].clone());
            }
        }

        Ok(properties)
    }

    /// 批量解析实体歧义
    pub async fn batch_resolve_entities(
        &self,
        mentions: &[String],
    ) -> Result<HashMap<String, Option<String>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = HashMap::new();

        for mention in mentions {
            match self.resolve_entity_ambiguity(mention, None).await {
                Ok(resolved) => {
                    results.insert(mention.clone(), resolved);
                }
                Err(e) => {
                    eprintln!("解析实体 '{}' 时出错: {}", mention, e);
                    results.insert(mention.clone(), None);
                }
            }
        }

        Ok(results)
    }
}

impl Default for OwnthinkApiClient {
    fn default() -> Self {
        Self::new()
    }
}

/// 抽取候选名中的括号限定词（`[...]`、`（...）`、`(...)`、`【...】`）。
///
/// 例如 `苹果[蔷薇科苹果属果实]` → `["蔷薇科苹果属果实"]`；
/// 无括号时返回空表。限定词通常比全名更能区分同名义项。
pub fn candidate_qualifiers(name: &str) -> Vec<String> {
    let mut qualifiers = Vec::new();
    let mut closing: Option<char> = None;
    let mut current = String::new();
    for ch in name.chars() {
        match closing {
            None => match ch {
                '[' => {
                    closing = Some(']');
                    current.clear();
                }
                '（' => {
                    closing = Some('）');
                    current.clear();
                }
                '(' => {
                    closing = Some(')');
                    current.clear();
                }
                '【' => {
                    closing = Some('】');
                    current.clear();
                }
                _ => {}
            },
            Some(close) if ch == close => {
                let text = current.trim();
                if !text.is_empty() {
                    qualifiers.push(text.to_string());
                }
                closing = None;
                current.clear();
            }
            Some(_) => current.push(ch),
        }
    }
    qualifiers
}

/// 最长公共子串长度（按字符、忽略空白）。
///
/// 用于消歧：词面越长的连续匹配越有区分度，且不会像单字重叠那样
/// 把「苹果公司」与「蔷薇科苹果属」里的共同提及误判成同类证据。
pub fn longest_common_substring_len(a: &str, b: &str) -> u64 {
    let a: Vec<char> = a.chars().filter(|ch| !ch.is_whitespace()).collect();
    let b: Vec<char> = b.chars().filter(|ch| !ch.is_whitespace()).collect();
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut previous = vec![0u64; b.len() + 1];
    let mut best = 0u64;
    for i in 1..=a.len() {
        let mut current = vec![0u64; b.len() + 1];
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                current[j] = previous[j - 1] + 1;
                if current[j] > best {
                    best = current[j];
                }
            }
        }
        previous = current;
    }
    best
}

/// 去掉候选名中的括号限定词，只留主体名（与 [`candidate_qualifiers`] 互补）。
pub fn candidate_primary_name(name: &str) -> String {
    let mut result = String::new();
    let mut depth = 0usize;
    for ch in name.chars() {
        match ch {
            '[' | '（' | '(' | '【' => depth += 1,
            ']' | '）' | ')' | '】' => depth = depth.saturating_sub(1),
            _ if depth == 0 => result.push(ch),
            _ => {}
        }
    }
    result
}

/// 基于上下文对候选项做确定性消歧，返回最佳下标。
///
/// 规则（纯函数，便于离线测试）：
/// - 无可用上下文：选权重最高者，同权取先出现者；
/// - 有上下文：得分 = 2 × 主体名最长公共子串长度 + 括号限定词最长公共子串长度
///   （主体名权重更高，限定词提供补充证据）；得分高者胜，同分比权重，
///   仍相同取先出现者；
/// - 所有候选得分均为 0（上下文无关）时回退到权重最高者。
pub fn select_candidate_by_context(
    candidates: &[(String, u64)],
    context: Option<&str>,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let context = context.map(str::trim).filter(|text| !text.is_empty());
    let mut best_index = 0usize;
    let mut best_key = candidate_context_key(&candidates[0], context);
    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let key = candidate_context_key(candidate, context);
        if key > best_key {
            best_key = key;
            best_index = index;
        }
    }
    Some(best_index)
}

/// 候选的排序键：`(上下文得分, 权重)`；无上下文时得分恒为 0，退化为权重比较。
fn candidate_context_key(candidate: &(String, u64), context: Option<&str>) -> (u64, u64) {
    let (name, weight) = candidate;
    let score = match context {
        Some(context) => {
            let primary = candidate_primary_name(name);
            let name_score = 2 * longest_common_substring_len(&primary, context);
            let qualifier_score = candidate_qualifiers(name)
                .iter()
                .map(|qualifier| longest_common_substring_len(qualifier, context))
                .max()
                .unwrap_or(0);
            name_score + qualifier_score
        }
        None => 0,
    };
    (score, *weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ambiguous_entities() {
        let client = OwnthinkApiClient::new();

        match client.get_ambiguous_entities("苹果").await {
            Ok(response) => {
                assert_eq!(response.message, "success");
                assert!(!response.data.is_empty());
                println!("找到 {} 个相关实体", response.data.len());

                for (entity, weight) in &response.data[..3.min(response.data.len())] {
                    println!("  - {} (权重: {})", entity, weight);
                }
            }
            Err(e) => {
                println!("测试跳过，API调用失败: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_entity_knowledge() {
        let client = OwnthinkApiClient::new();

        match client.get_entity_knowledge("刘德华").await {
            Ok(response) => {
                assert_eq!(response.message, "success");
                println!("实体: {}", response.data.entity);
                println!("描述: {}", response.data.desc);
                println!("标签: {:?}", response.data.tag);
                println!("属性数量: {}", response.data.avp.len());
            }
            Err(e) => {
                println!("测试跳过，API调用失败: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_resolve_ambiguity() {
        let client = OwnthinkApiClient::new();

        match client.resolve_entity_ambiguity("苹果", None).await {
            Ok(Some(entity)) => {
                println!("消歧结果: {}", entity);
            }
            Ok(None) => {
                println!("未找到匹配的实体");
            }
            Err(e) => {
                println!("测试跳过，API调用失败: {}", e);
            }
        }
    }

    #[test]
    fn candidate_qualifiers_extracts_all_bracket_styles() {
        assert_eq!(
            candidate_qualifiers("苹果[蔷薇科苹果属果实]"),
            vec!["蔷薇科苹果属果实".to_string()]
        );
        assert_eq!(
            candidate_qualifiers("苹果（水果）"),
            vec!["水果".to_string()]
        );
        assert_eq!(
            candidate_qualifiers("Apple(company)【科技】"),
            vec!["company".to_string(), "科技".to_string()]
        );
        assert!(candidate_qualifiers("苹果公司").is_empty());
        assert!(
            candidate_qualifiers("空白[   ]").is_empty(),
            "纯空白限定词忽略"
        );
    }

    #[test]
    fn candidate_primary_name_strips_qualifiers() {
        assert_eq!(candidate_primary_name("苹果[蔷薇科苹果属果实]"), "苹果");
        assert_eq!(candidate_primary_name("苹果（水果）"), "苹果");
        assert_eq!(candidate_primary_name("Apple(company)【科技】"), "Apple");
        assert_eq!(candidate_primary_name("苹果公司"), "苹果公司");
    }

    #[test]
    fn longest_common_substring_len_measures_continuous_match() {
        assert_eq!(
            longest_common_substring_len("苹果公司", "苹果公司发布了新产品"),
            4
        );
        assert_eq!(longest_common_substring_len("苹果", "蔷薇科植物的果实"), 1);
        assert_eq!(longest_common_substring_len("果实", "蔷薇科植物的果实"), 2);
        assert_eq!(longest_common_substring_len("苹", "蔷薇科植物的果实"), 0);
        assert_eq!(longest_common_substring_len("", "任意上下文"), 0);
        assert_eq!(longest_common_substring_len("苹果", ""), 0);
        assert_eq!(
            longest_common_substring_len("苹 果", "这里有苹果"),
            2,
            "空白不参与匹配"
        );
    }

    #[test]
    fn select_candidate_uses_context_over_weight() {
        // 权重上水果更高，但上下文明显指向公司
        let candidates = vec![
            ("苹果公司".to_string(), 10_084_009),
            ("苹果[蔷薇科苹果属果实]".to_string(), 12_223_855),
        ];
        let index = select_candidate_by_context(&candidates, Some("苹果公司发布了新产品")).unwrap();
        assert_eq!(candidates[index].0, "苹果公司");

        // 上下文只命中括号限定词时选中水果（即使权重更低）
        let index =
            select_candidate_by_context(&candidates, Some("蔷薇科植物的果实有哪些")).unwrap();
        assert_eq!(candidates[index].0, "苹果[蔷薇科苹果属果实]");
    }

    #[test]
    fn select_candidate_falls_back_to_weight() {
        let candidates = vec![
            ("苹果[蔷薇科苹果属果实]".to_string(), 12_223_855),
            ("苹果公司".to_string(), 10_084_009),
        ];
        // 无上下文：选权重最高者
        assert_eq!(select_candidate_by_context(&candidates, None), Some(0));
        assert_eq!(
            select_candidate_by_context(&candidates, Some("   ")),
            Some(0)
        );
        // 上下文与所有候选都无关：仍回退到权重最高者
        assert_eq!(
            select_candidate_by_context(&candidates, Some("今天天气不错")),
            Some(0)
        );
        // 同分同权：取先出现者（稳定）
        let tied = vec![("甲".to_string(), 1), ("乙".to_string(), 1)];
        assert_eq!(select_candidate_by_context(&tied, None), Some(0));
        assert_eq!(select_candidate_by_context(&[], None), None);
    }
}

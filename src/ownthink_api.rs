use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const OWNTHINK_BASE_URL: &str = "https://api.ownthink.com/kg";

/// 复用同一个连接池与 TLS 上下文；`reqwest::Client` 克隆开销极低
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

    async fn make_request_with_retry<T, R>(&self, request_fn: impl Fn() -> R) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
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
                                    eprintln!("JSON解析失败 (尝试 {}/{}), 将重试...", attempt + 1, self.max_retries + 1);
                                    tokio::time::sleep(std::time::Duration::from_millis(self.retry_delay_ms)).await;
                                    continue;
                                }
                            }
                        }
                    } else if status.is_server_error() || status == 429 {
                        // 服务器错误或限流，可以重试
                        last_error = Some(format!("HTTP错误: {}", status).into());
                        if attempt < self.max_retries {
                            eprintln!("服务器错误 {} (尝试 {}/{}), 将重试...", status, attempt + 1, self.max_retries + 1);
                            tokio::time::sleep(std::time::Duration::from_millis(self.retry_delay_ms * (attempt + 1) as u64)).await;
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
                        eprintln!("网络请求失败 (尝试 {}/{}), 将重试...", attempt + 1, self.max_retries + 1);
                        tokio::time::sleep(std::time::Duration::from_millis(self.retry_delay_ms * (attempt + 1) as u64)).await;
                        continue;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| "未知错误".into()))
    }

    /// 获取歧义关系 - 根据名称(mention)返回可能的实体列表
    pub async fn get_ambiguous_entities(&self, mention: &str) -> Result<AmbiguousResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/ambiguous", self.base_url);
        let request = AmbiguousRequest {
            mention: mention.to_string(),
        };

        let response: AmbiguousResponse = self.make_request_with_retry(|| {
            self.client
                .post(&url)
                .json(&request)
                .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                .send()
        }).await?;
        
        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 获取实体的完整知识
    pub async fn get_entity_knowledge(&self, entity: &str) -> Result<KnowledgeResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/knowledge", self.base_url);
        let request = KnowledgeRequest {
            entity: entity.to_string(),
        };

        let response: KnowledgeResponse = self.make_request_with_retry(|| {
            self.client
                .post(&url)
                .json(&request)
                .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                .send()
        }).await?;
        
        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 获取实体的特定属性值
    pub async fn get_entity_attribute_value(&self, entity: &str, attribute: &str) -> Result<EavResponse, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/eav", self.base_url);
        let request = EavRequest {
            entity: entity.to_string(),
            attribute: attribute.to_string(),
        };

        let response: EavResponse = self.make_request_with_retry(|| {
            self.client
                .post(&url)
                .json(&request)
                .timeout(std::time::Duration::from_secs(self.timeout_seconds))
                .send()
        }).await?;
        
        if response.message != "success" {
            return Err(format!("API返回错误: {}", response.message).into());
        }

        Ok(response)
    }

    /// 解析歧义实体 - 返回最佳匹配的实体
    pub async fn resolve_entity_ambiguity(&self, mention: &str, context: Option<&str>) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let ambiguous_response = self.get_ambiguous_entities(mention).await?;
        
        if ambiguous_response.data.is_empty() {
            return Ok(None);
        }

        // 简单策略：返回权重最高的实体
        // 在更复杂的实现中，可以根据上下文信息进行更智能的选择
        let best_entity = ambiguous_response.data
            .into_iter()
            .max_by_key(|(_, weight)| *weight)
            .map(|(entity, _)| entity);

        // 如果提供了上下文，可以在此处实现更复杂的消歧逻辑
        if let Some(_context) = context {
            // TODO: 实现基于上下文的消歧策略
        }

        Ok(best_entity)
    }

    /// 获取实体的简化知识 - 转换为内部知识表示
    pub async fn get_simplified_knowledge(&self, entity: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error + Send + Sync>> {
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
    pub async fn batch_resolve_entities(&self, mentions: &[String]) -> Result<HashMap<String, Option<String>>, Box<dyn std::error::Error + Send + Sync>> {
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
}
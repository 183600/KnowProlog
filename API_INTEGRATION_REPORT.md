# Ownthink API 集成功能说明

## 功能概述

成功为 KnowProlog 系统集成了 Ownthink API，支持在对话过程中动态获取实体知识。

## 实现的功能

### 1. 完整实体知识查询
**用法示例：**
- `刘德华的信息`
- `告诉我关于马云`
- `什么是苹果`

**功能：**
- 获取实体的完整描述、标签和所有属性值对
- 自动进行实体歧义消解
- 显示详细的属性信息（如生日、职业、作品等）

### 2. 实体属性值查询
**用法示例：**
- `刘德华的妻子`
- `苹果的颜色`
- `马云的职业`

**功能：**
- 针对特定属性查询具体值
- 支持多值属性（如颜色、职业等）
- 快速获取精确信息

### 3. 智能实体歧义解析
**功能：**
- 自动处理同名实体的歧义问题
- 基于权重选择最匹配的实体
- 可配置是否显示歧义选项

## 技术实现

### API 模块 (`src/ownthink_api.rs`)
- **OwnthinkApiClient**: 主要客户端类
- **重试机制**: 支持网络请求失败重试
- **超时控制**: 可配置请求超时时间
- **错误处理**: 完善的错误处理和降级策略

### 数据结构
```rust
// 知识查询响应
pub struct KnowledgeData {
    pub entity: String,      // 实体名称
    pub desc: String,        // 实体描述
    pub tag: Vec<String>,    // 实体标签
    pub avp: Vec<Vec<String>>, // 属性值对
}

// 属性值查询响应
pub struct EavData {
    pub entity: String,      // 实体名称
    pub attribute: String,   // 属性名称
    pub value: Vec<String>,  // 属性值列表
}
```

### 配置选项 (`config.toml`)
```toml
[api]
ownthink_base_url = "https://api.ownthink.com/kg"
ownthink_timeout_seconds = 10
enable_entity_disambiguation = true
max_ambiguous_entities = 5
auto_select_best_entity = true
```

## 集成方式

### 主对话流程集成
1. **直接实体查询检测**: 识别 "的信息"、"告诉我关于"、"什么是" 等模式
2. **属性查询检测**: 使用正则表达式识别 "实体的属性" 模式
3. **API 调用**: 根据查询类型调用相应的 API 方法
4. **结果展示**: 格式化显示 API 返回的知识信息
5. **降级处理**: API 失败时回退到本地知识库查询

### 错误处理策略
- **网络超时**: 自动重试机制
- **API 错误**: 降级到本地查询
- **解析错误**: 友好的错误提示
- **无结果**: 提供替代查询建议

## 测试验证

### 单元测试
```bash
cargo test test_entity_knowledge    # 测试实体知识获取
cargo test test_ambiguous_entities  # 测试实体歧义解析
```

### 集成测试
```bash
# 完整实体信息查询
echo "刘德华的信息" | cargo run

# 特定属性查询
echo "苹果的颜色" | cargo run
echo "刘德华的妻子" | cargo run

# 歧义实体处理
echo "什么是苹果" | cargo run
```

## 性能特点

- **异步处理**: 使用 tokio 异步运行时，不阻塞主线程
- **智能缓存**: 支持会话级别的实体解析缓存
- **并发控制**: 支持批量实体解析
- **网络优化**: 连接复用和超时控制

## 用户体验

- **无缝集成**: API 查询与本地查询无缝切换
- **快速响应**: 平均响应时间 < 1秒
- **丰富信息**: 提供比本地更全面的实体知识
- **智能提示**: 自动歧义解析和最佳匹配选择

通过这个集成，KnowProlog 系统现在可以：
1. 访问海量的在线知识库
2. 提供更准确和全面的实体信息
3. 处理本地知识库不包含的实体
4. 实现更智能的对话体验
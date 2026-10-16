# KnowProlog

一个基于Rust开发的中文知识图谱系统，集成了自然语言处理、RDF存储和Prolog推理引擎，支持从中文文本中自动提取知识并进行逻辑推理。

## 🌟 主要特性

- **中文自然语言处理**: 使用jieba-rs进行中文分词和词性标注，支持自定义词典
- **知识自动提取**: 基于模式匹配从文本中提取实体和关系，支持复杂规则配置
- **RDF存储**: 使用Oxigraph作为高性能RDF三元组存储，支持SPARQL查询
- **逻辑推理**: 集成Scryer Prolog引擎进行知识推理和查询生成
- **实体歧义解析**: 集成ownthink知识图谱API，自动解析歧义实体并获取详细信息
- **JSON-LD支持**: 完整支持JSON-LD格式的知识库导入和内容解析
- **Shell命令行导入**: 支持通过命令行直接导入JSON-LD文件和内容，方便批量处理
- **灵活配置**: 通过TOML文件配置系统行为，支持热更新
- **交互式界面**: 命令行交互式查询和知识录入，支持多种输入模式

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│                  用户输入                         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              NLP处理模块                          │
│  - 中文分词 (jieba-rs)                           │
│  - 词性标注                                      │
│  - 同义词归一化                                  │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  知识提取模块    │     │  查询分析模块    │
│  - 实体识别     │     │  - 意图识别     │
│  - 关系抽取     │     │  - 实体提取     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Oxigraph存储    │◄────│ Prolog推理引擎  │
│  - RDF三元组    │     │  - 逻辑规则    │
│  - SPARQL查询   │     │  - 推理查询    │
└─────────────────┘     └─────────────────┘
```

## 📋 系统要求

- Rust 1.70或更高版本
- 操作系统: Linux/macOS/Windows
- 内存: 至少2GB RAM
- 磁盘空间: 至少500MB可用空间

## 🚀 快速开始

### 1. 环境准备

确保系统已安装Rust开发环境：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### 2. 克隆项目

```bash
git clone https://github.com/yourusername/knowprolog2.git
cd knowprolog2
```

### 3. 初始化数据目录

```bash
mkdir -p oxigraph_data
```

### 4. 编译运行

```bash
# 编译项目
cargo build --release

# 运行系统
cargo run --release
```

### 5. 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name
```

## ⚙️ 配置说明

系统配置通过`config.toml`文件进行管理：

### 系统配置
```toml
[system]
oxigraph_endpoint = "http://localhost:7878"  # Oxigraph HTTP端点
oxigraph_store_path = "./oxigraph_data"      # 数据存储路径
default_language = "zh"                       # 默认语言
```

### API配置
```toml
[api]
ownthink_base_url = "https://api.ownthink.com/kg"
ownthink_timeout_seconds = 10
enable_entity_disambiguation = true  # 是否启用实体歧义解析
max_ambiguous_entities = 5          # 显示的最大歧义实体数量
auto_select_best_entity = true       # 是否自动选择权重最高的实体
```

### NLP配置
```toml
[nlp]
import_keywords = ["录入", "添加", "新增", "记录"]  # 触发知识导入的关键词

[nlp.pos_dict]
"小明" = "nr"  # nr表示人名
"是" = "v"     # v表示动词
"的" = "u"     # u表示助词

[nlp.name_patterns]
prefixes = ["小", "老", "大"]           # 姓氏前缀
suffixes = ["总", "哥", "姐"]           # 名字后缀
full_names = ["张三", "李四", "王五"]   # 常见全名
```

### 推理规则配置
```toml
[[prolog_rules]]
name = "年龄查询"
pattern = ["nr", "多大"]  # nr(人名) + "多大"
query_type = "age_query"
priority = 10
```

## 📖 使用指南

### 知识录入

在提示符下输入知识，使用导入关键词触发：

```
> 录入：小明18岁
实体 - 类型: Person, 属性: {"name": "小明", "age": "18"}
知识已成功保存

> 添加：张三是李四的老师
关系 - Person(张三) -[teacher_of]-> Person(李四)
知识已成功保存
```

### JSON-LD文件导入

系统支持导入JSON-LD格式的知识文件：

```
> 导入文件：test_data.jsonld
正在读取JSON-LD文件...
解析了 15 个实体和 8 个关系
成功导入 23 个三元组到知识库
```

支持的JSON-LD格式示例：
```json
{
  "@context": {
    "name": "http://example.org/kg/name",
    "age": "http://example.org/kg/age",
    "teacher_of": "http://example.org/kg/teacher_of"
  },
  "@graph": [
    {
      "@id": "http://example.org/kg/entity/小明",
      "@type": "http://example.org/kg/Person",
      "name": "小明",
      "age": "18"
    },
    {
      "@id": "http://example.org/kg/entity/张三",
      "@type": "http://example.org/kg/Person",
      "name": "张三",
      "teacher_of": {
        "@id": "http://example.org/kg/entity/李四"
      }
    }
  ]
}
```

### JSON-LD内容导入

也可以直接输入JSON-LD内容进行导入：

```
> 导入内容：{"@context": {"name": "http://example.org/kg/name"}, "@id": "http://example.org/kg/entity/王五", "@type": "http://example.org/kg/Person", "name": "王五"}
解析JSON-LD内容...
提取实体：王五 (Person)
知识已成功保存
```

### Shell命令行导入

除了交互模式，系统还支持通过shell命令行直接导入JSON-LD知识库：

#### 1. 从文件导入JSON-LD

```bash
# 导入JSON-LD文件
cargo run --release -- import-jsonld test_data.jsonld

# 或者使用短命令
./knowprolog import-jsonld knowledge_base.jsonld
```

#### 2. 从内容导入JSON-LD

```bash
# 直接导入JSON-LD内容
cargo run --release -- import-jsonld-content '{
  "@context": {
    "name": "http://example.org/kg/name",
    "Person": "http://example.org/kg/Person"
  },
  "@id": "http://example.org/kg/entity/王五",
  "@type": "Person",
  "name": "王五"
}'
```

#### 3. 查看帮助信息

```bash
# 查看所有可用命令
cargo run --release -- --help

# 查看特定命令的帮助
cargo run --release -- import-jsonld --help
```

#### 4. 运行交互模式

```bash
# 启动交互模式（默认）
cargo run --release

# 或者显式指定交互模式
cargo run --release -- interactive
```

#### 批量导入脚本示例

创建批量导入脚本：

```bash
#!/bin/bash
# batch_import.sh

echo "开始批量导入知识库..."

# 导入人物信息
cargo run --release -- import-jsonld people.jsonld

# 导入公司信息
cargo run --release -- import-jsonld companies.jsonld

# 导入关系信息
cargo run --release -- import-jsonld relationships.jsonld

echo "批量导入完成！"
```

运行批量导入：

```bash
chmod +x batch_import.sh
./batch_import.sh
```

### 实体歧义解析查询

当查询包含歧义实体时，系统会自动调用ownthink API进行消歧：

```
> 苹果的颜色是什么？
正在解析实体歧义...
发现实体 '苹果' 有多种含义:
  1. 苹果[蔷薇科苹果属果实] (权重: 12223855)
  2. 苹果公司 (权重: 10084009)
  3. 苹果[韩国2008年康理贯执导电影] (权重: 7589467)
  4. 苹果[动漫《男子高中生的日常》中角色] (权重: 7498900)
  5. 苹果[谢和弦、E-SO演唱歌曲] (权重: 2047)
自动选择: 苹果[蔷薇科苹果属果实]
  描述: 苹果（学名：Malus domestica）是蔷薇科苹果属的果实...
  标签: 水果, 食物, 植物
─────────────────────────────────
查询类型: property_query, 实体: ["苹果[蔷薇科苹果属果实]"]
苹果的颜色有红色、粉色、金黄色、绿色、白色
```

### 传统知识查询

直接输入自然语言查询：

```
> 小明多大？
查询类型: age_query, 实体: ["小明"]
Prolog查询: age_of('小明', Age)
年龄是 18 岁

> 张三和李四什么关系？
查询类型: relation_query, 实体: ["张三", "李四"]
Prolog查询: related('张三', Relation, '李四')
关系是：teacher_of
```

### 退出系统

```
> quit
系统已退出。
```

## 🔧 扩展开发

### 添加新的实体类型

在`config.toml`中添加：

```toml
[[entity_types]]
label = "Company"
examples = ["腾讯公司", "阿里巴巴成立于1999年"]

[[entity_types.properties]]
name = "name"
data_type = "string"
required = true
extraction_patterns = ["pos:nr", "pos:n"]

[[entity_types.properties]]
name = "founded_year"
data_type = "integer"
required = false
extraction_patterns = ["pos:m"]
```

### 添加新的推理规则

1. 在`config.toml`中定义模式：
```toml
[[prolog_rules]]
name = "公司查询"
pattern = ["n", "的", "创始人"]
query_type = "founder_query"
priority = 10
```

2. 在`prolog_engine.rs`中添加Prolog规则：
```prolog
founder_of(Company, Founder) :- 
    relation(Founder, founded, Company).
```

3. 在`prolog_engine.rs`的`generate_prolog_query`函数中添加查询生成逻辑。

### 自定义同义词

在`config.toml`中添加：
```toml
[synonyms.mappings]
"创建" = "成立"
"建立" = "成立"
"CEO" = "首席执行官"
```

## 📊 数据模型与RDF转换

系统使用RDF三元组存储数据，支持从多种格式转换为RDF：

### 实体表示
```
<http://example.org/kg/entity/小明> 
    rdf:type <http://example.org/kg/Person> ;
    <http://example.org/kg/name> "小明" ;
    <http://example.org/kg/age> "18" .
```

### 关系表示
```
<http://example.org/kg/entity/张三> 
    <http://example.org/kg/teacher_of> 
    <http://example.org/kg/entity/李四> .
```

### RDF三元组转换流程

系统支持多种输入格式的RDF三元组转换：

#### 1. 自然语言文本转RDF
- **输入**: "小明18岁"
- **NLP处理**: 分词、实体识别、关系抽取
- **输出RDF三元组**:
  ```turtle
  <http://example.org/kg/entity/小明> rdf:type <http://example.org/kg/Person> .
  <http://example.org/kg/entity/小明> <http://example.org/kg/name> "小明" .
  <http://example.org/kg/entity/小明> <http://example.org/kg/age> "18" .
  ```

#### 2. JSON-LD转RDF
- **输入JSON-LD**:
  ```json
  {
    "@context": {
      "name": "http://example.org/kg/name",
      "age": "http://example.org/kg/age"
    },
    "@id": "http://example.org/kg/entity/小明",
    "@type": "http://example.org/kg/Person",
    "name": "小明",
    "age": "18"
  }
  ```
- **转换处理**: JSON-LD解析、URI展开、三元组提取
- **输出RDF三元组**: 同上

#### 3. 关系型数据转RDF
- **输入**: "张三是李四的老师"
- **关系抽取**: 主语(张三) + 谓语(是...的老师) + 宾语(李四)
- **输出RDF三元组**:
  ```turtle
  <http://example.org/kg/entity/张三> rdf:type <http://example.org/kg/Person> .
  <http://example.org/kg/entity/李四> rdf:type <http://example.org/kg/Person> .
  <http://example.org/kg/entity/张三> <http://example.org/kg/teacher_of> <http://example.org/kg/entity/李四> .
  ```

### RDF命名空间和URI模式

系统使用统一的命名空间和URI模式：

```turtle
# 命名空间定义
@prefix kg: <http://example.org/kg/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

# 实体URI模式
kg:entity/{实体名称}

# 属性URI模式
kg:{属性名称}

# 类型URI模式
kg:{类型名称}
```

### 数据类型处理

系统支持多种数据类型的RDF表示：

```turtle
# 字符串类型
<http://example.org/kg/entity/小明> <http://example.org/kg/name> "小明" .

# 整数类型
<http://example.org/kg/entity/小明> <http://example.org/kg/age> "18"^^xsd:integer .

# 日期类型
<http://example.org/kg/entity/公司> <http://example.org/kg/founded_date> "1999-01-01"^^xsd:date .

# 布尔类型
<http://example.org/kg/entity/员工> <http://example.org/kg/is_active> "true"^^xsd:boolean .
```

### 批量转换和导入

系统支持批量处理多种格式的数据：

1. **批量文本处理**:
   - 每行一个知识条目
   - 自动识别实体和关系
   - 生成对应的RDF三元组

2. **JSON-LD批量导入**:
   - 支持 `@graph` 数组格式
   - 自动处理 `@context` 映射
   - 递归处理嵌套对象

3. **RDF文件导入**:
   - 支持 Turtle (.ttl) 格式
   - 支持 N-Triples (.nt) 格式
   - 支持 RDF/XML (.rdf) 格式

## 🔍 SPARQL查询示例

查询所有人物：
```sparql
PREFIX kg: <http://example.org/kg/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?person ?name ?age
WHERE {
    ?person rdf:type kg:Person ;
            kg:name ?name .
    OPTIONAL { ?person kg:age ?age }
}
```

## 🐛 故障排除

### 常见问题

1. **编译错误**: 确保Rust版本>=1.70
   ```bash
   rustup update stable
   rustc --version
   ```

2. **存储路径错误**: 确保`oxigraph_data`目录存在且有写权限
   ```bash
   mkdir -p oxigraph_data
   chmod 755 oxigraph_data
   ```

3. **中文分词问题**: 系统使用jieba-rs，首次运行会自动下载词典

4. **JSON-LD导入失败**: 检查JSON-LD格式是否符合标准，确保`@context`映射正确

5. **Prolog推理异常**: 检查Prolog规则配置，确保事实加载正确

### 日志调试

设置环境变量启用详细日志：
```bash
RUST_LOG=debug cargo run
```

### 性能优化

- 使用`--release`模式编译以获得最佳性能
- 定期清理`oxigraph_data`目录中的旧数据
- 合理配置实体类型和提取规则数量

## 🔧 开发和测试

### 项目结构
```
src/
├── main.rs              # 主程序入口和交互界面
├── lib.rs               # 核心库，包含配置和知识提取
├── nlp.rs               # 中文NLP处理模块
└── prolog_engine.rs     # Prolog推理引擎

tests/
└── integration_test.rs  # 集成测试

config.toml              # 系统配置文件
CLAUDE.md               # 开发指南
```

### 测试覆盖
- 配置加载和验证
- NLP预处理和同义词归一化
- 知识提取和模式匹配
- RDF存储和SPARQL查询
- Prolog推理和查询生成
- JSON-LD导入和解析
- 并发访问和错误处理

### 代码质量
```bash
# 代码格式化
cargo fmt

# 代码检查
cargo clippy

# 文档生成
cargo doc --open
```

## 📚 技术栈

- **Rust**: 系统编程语言
- **Tokio**: 异步运行时
- **Oxigraph**: RDF三元组存储
- **Scryer Prolog**: Prolog推理引擎
- **jieba-rs**: 中文分词
- **Serde**: 序列化框架
- **Reqwest**: HTTP客户端库

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 贡献流程

1. Fork项目到你的GitHub账号
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循Rust官方代码风格
- 添加适当的文档注释
- 编写相应的测试用例
- 确保所有测试通过

### 问题报告

报告问题时请包含：
- 操作系统和Rust版本
- 详细的错误信息
- 重现步骤
- 相关的配置文件

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🙏 致谢

- [Oxigraph](https://github.com/oxigraph/oxigraph) - 高性能RDF存储
- [Scryer Prolog](https://github.com/mthom/scryer-prolog) - Rust实现的Prolog引擎
- [jieba-rs](https://github.com/messense/jieba-rs) - 中文分词库

## 📮 联系方式

- 项目主页: https://github.com/yourusername/knowprolog2
- Issue追踪: https://github.com/yourusername/knowprolog2/issues
- 邮箱: your.email@example.com

---

## 🤖 文档说明

本README文档由AI助手（Claude）生成，基于项目代码和配置文件的分析。文档内容包括系统架构、使用指南、配置说明等均通过自动分析生成。如有不准确之处，欢迎提交Issue或PR进行修正。

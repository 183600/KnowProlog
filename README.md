# KnowProlog

一个基于 Rust 开发的中文知识图谱系统，集成自然语言处理、RDF 存储与 Prolog 推理引擎，支持从中文文本自动提取知识并进行逻辑推理。

## 🌟 主要特性

- **中文自然语言处理**：使用 jieba-rs 分词与词性标注，支持自定义词典与同义词归一化
- **知识自动提取**：基于可配置模式从文本抽取实体、关系与属性
- **RDF 存储**：基于 Oxigraph 的高性能三元组存储，支持 SPARQL 1.1 查询
- **逻辑推理**：集成 Scryer Prolog 引擎，支持增量事实装载、三态查询（有解 / 无解 / 引擎故障）
- **实体消歧**：可选接入 ownthink 知识图谱 API，自动消歧并获取详细信息
- **JSON-LD 支持**：完整支持 JSON-LD 导入与内容解析（含 `@context`、`@graph`、嵌套对象）
- **命令行批量导入**：提供 `import-jsonld`、`import-rdf`、`export-rdf` 等子命令，便于脚本化批量处理
- **灵活配置**：全参数 TOML 配置（系统 / NLP / 实体类型 / 抽取规则 / 推理规则 / 谓词画像 / 类型体系 / 插件等）
- **交互式界面**：REPL 模式支持导入、查询、澄清、多轮上下文与指代消解

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│                    用户输入                       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│                  NLP 处理层                       │
│  - 中文分词 (jieba-rs)                            │
│  - 词性标注 + 词性证据溯源                        │
│  - 同义词归一化（扁平映射 / 词义感知）            │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  知识抽取层      │     │  查询分析层      │
│  - 模式抽取      │     │  - 意图匹配      │
│  - 属性桥校验    │     │  - 实体捕获      │
│  - 双向语义规则  │     │  - 软路由兜底    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ 存储层          │◄────│ 推理层           │
│  - Oxigraph RDF │     │  - Prolog 引擎  │
│  - SPARQL 查询  │     │  - 增量事实装载  │
│  - 事实快照     │     │  - 三态查询      │
└─────────────────┘     └─────────────────┘
```

### 核心模块职责

| 层级 | 模块 | 职责 |
|------|------|------|
| 交互 | `src/main_parts/`、`src/serve_api.rs` | CLI 子命令、交互循环、HTTP 服务、结果渲染 |
| 语言 | `src/nlp.rs`、`src/question_analyzer.rs`、`src/advanced_nlu.rs` | 分词、词性、同义词归一化、问句分类与关键词抽取 |
| 抽取 | `src/entity_relation_extractor.rs`、`src/lib/` | 实体、关系、属性的模式抽取与 JSON-LD 导入 |
| 存储 | `OxigraphManager`（`src/lib/`） | RDF 三元组读写、SPARQL 查询、事实导出 |
| 推理 | `src/prolog_engine.rs`、`src/resolution_engine.rs`、`src/planning.rs` | Prolog 事实装载、三态查询、查询计划与轨迹 |
| 对话 | `src/dialogue_manager/`、`src/contextual_reference.rs` | 多轮上下文、指代消解、澄清与挂账 |
| 恢复 | `src/recovery.rs`、`src/fulfillment.rs` | 失败分级、提名验证、状态化措辞 |
| 观测 | `src/perf_ledger.rs`、`src/startup_ledger.rs`、`src/provenance.rs` | 操作计数、启动分期、来源与冲突记录 |
| 质量 | `src/quality.rs`、`src/release.rs`、`src/shape_gate.rs` | 质量索引、发布检查、形状校验 |

## 📋 系统要求

- Rust 1.70 或更高版本
- 操作系统：Linux / macOS / Windows
- 内存：至少 2 GB RAM
- 磁盘空间：至少 500 MB 可用空间

## 🚀 快速开始

### 1. 环境准备

确保已安装 Rust 开发环境：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### 2. 克隆项目

```bash
git clone https://github.com/183600/KnowProlog.git
cd KnowProlog
```

### 3. 初始化数据目录

```bash
mkdir -p oxigraph_data
```

### 4. 编译与运行

```bash
# 编译（发布模式，性能最优）
cargo build --release

# 进入交互式 REPL
cargo run --release

# 或显式指定子命令
cargo run --release -- interactive
```

### 5. 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name
```

### 6. 常用子命令速览

| 命令 | 说明 |
|------|------|
| `import-jsonld <file>` | 导入 JSON-LD 文件，`-` 表示标准输入 |
| `import-rdf <file>` | 导入 Turtle / N-Triples / N-Quads / TriG / RDF/XML |
| `export-rdf <file>` | 导出默认图，格式由扩展名决定 |
| `ask <question>` | 只读问答，可用 `--json`、`--strict`、`--explain`、`--clarify` |
| `sparql <query>` | 执行 SPARQL 1.1 查询 |
| `batch-import <file>` | 按已采纳的批量教学规则导入自然语言句子 |

全局参数 `--config` 指定配置文件，`--store` 覆盖存储目录。两者对交互模式和所有子命令生效，脚本化运行时建议显式传入，避免依赖工作目录。

## ⚙️ 配置说明

系统配置通过 `config.toml` 管理，主要包含以下章节：

### 系统设置

```toml
[system]
oxigraph_endpoint = "http://localhost:7878"  # Oxigraph HTTP 端点
oxigraph_store_path = "./oxigraph_data"      # 数据存储路径
default_language = "zh"                       # 默认语言
```

### API 配置

```toml
[api]
ownthink_base_url = "https://api.ownthink.com/kg"
ownthink_timeout_seconds = 10
enable_entity_disambiguation = true   # 是否启用实体消歧
max_ambiguous_entities = 5            # 最大显示歧义实体数
auto_select_best_entity = true        # 自动选择权重最高实体
external_ledger_path = ""             # 外链名册落盘路径（留空用 <存储>/external_ledger.json）
external_ledger_capacity = 64         # 条目上限（LRU）
external_ledger_ttl_ticks = 0         # 逻辑轮次时限（0 = 不过期）
```

### NLP 配置

```toml
[nlp]
import_keywords = ["录入", "添加", "新增", "记录", "导入", "导入jsonld文件", "导入jsonld内容", "import jsonld", "import jsonld content"]
enable_nlu = true
nlu_confidence_threshold = 0.6
enable_local_semantic = true
disable_ownthink_api = true
semantic_similarity_threshold = 0.7
max_entity_candidates = 10
enable_advanced_entity_extraction = true
enable_relation_extraction = true
context_window_size = 5

[nlp.pos_dict]
"是" = "v"
"的" = "u"
"年龄" = "n"
"岁" = "q"
"多大" = "r"
# ... 更多词性映射

[nlp.name_patterns]
prefixes = ["小", "老", "大"]
suffixes = ["总", "哥", "姐"]
full_names = ["张三", "李四", "王五", "赵六"]
profession_suffix = ["医生", "老师", "工程师"]
```

### 推理配置（节选）

```toml
[reasoning]
enable_link_prediction = true
link_prediction_min_confidence = 0.6
enable_semantic_embedding = true
embedding_similarity_floor = 0.6
enable_entailment = true
enable_resolution = true
enable_association_rules = true
enable_abduction = true
enable_clarification = true
enable_clarification_lookahead = true
enable_graph_walk = true
enable_soft_logic = true
enable_causal_reasoning = true
enable_problog = true
enable_dialogue_manager = true
enable_discourse = true
enable_holographic_memory = true
require_predicate_admission = false
```

完整配置项见 `config.toml` 与 `PLAN.md#6-配置与数据`。

### 谓词画像

```toml
[[predicate_profiles]]
name = "师傅"
aliases = ["老师", "恩师", "师父"]
inverse = "徒弟"
transitive = true
domain = ["人物"]
range = ["人物"]
```

画像用于导入时的别名归一、互逆翻转、传递闭包与定义域/值域校验；导入与查询共用同一套画像。

### 类型体系

```toml
[type_system]
closure_gate = true

[[type_system.aliases]]
alias = "Person"
canonical = "人物"

[[type_system.subclasses]]
sub = "人物"
sup = "实体"

[[type_system.disjoint]]
left = "人物"
right = "地点"
```

用于实体类型的别名归一、子类闭包与互斥判定；模式闸门按「别名 + 子类 + 互斥」三重证据决定是否放行。

## 📖 使用指南

### 知识录入（交互模式）

在 `>` 提示符下输入，以导入关键词开头：

```
> 录入：小明18岁
实体 - 类型: Person, 属性: {"name": "小明", "age": "18"}
知识已成功保存

> 添加：张三是李四的老师
关系 - Person(张三) -[teacher_of]-> Person(李四)
知识已成功保存
```

### JSON-LD 文件导入

```
> 导入文件：test_data.jsonld
正在读取JSON-LD文件...
解析了 15 个实体和 8 个关系
成功导入 23 个三元组到知识库
```

支持的 JSON-LD 格式示例：

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

### JSON-LD 内容直接导入

```
> 导入内容：{"@context": {"name": "http://example.org/kg/name"}, "@id": "http://example.org/kg/entity/王五", "@type": "http://example.org/kg/Person", "name": "王五"}
解析JSON-LD内容...
提取实体：王五 (Person)
知识已成功保存
```

### 命令行批量导入（非交互模式）

```bash
# 导入 JSON-LD 文件
cargo run --release -- import-jsonld test_data.jsonld

# 从标准输入导入 JSON-LD
cat knowledge.jsonld | cargo run --release -- import-jsonld -

# 导入 Turtle / N-Triples / RDF/XML 等
cargo run --release -- import-rdf data.ttl

# 导出默认图（格式由扩展名决定）
cargo run --release -- export-rdf dump.ttl

# 单次问答（脚本化友好）
cargo run --release -- ask "姚明的身高是多少"
cargo run --release -- ask "姚明的身高是多少" --json --strict

# 执行 SPARQL 查询
cargo run --release -- sparql 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10'
```

查看所有子命令与参数：

```bash
cargo run --release -- --help
cargo run --release -- import-jsonld --help
```

### 实体消歧查询

当查询包含歧义实体时，系统自动调用 ownthink API 消歧：

```
> 苹果的颜色是什么？
正在解析实体歧义...
发现实体 '苹果' 有多种含义:
  1. 苹果[蔷薇科苹果属果实] (权重: 12223855)
  2. 苹果公司 (权重: 10084009)
  3. 苹果[韩国2008年康理贯执导电影] (权重: 7589467)
自动选择: 苹果[蔷薇科苹果属果实]
  描述: 苹果（学名：Malus domestica）是蔷薇科苹果属的果实...
  标签: 水果, 食物, 植物
─────────────────────────────────
查询类型: property_query, 实体: ["苹果[蔷薇科苹果属果实]"]
苹果的颜色有红色、粉色、金黄色、绿色、白色
```

### 传统知识查询

直接输入自然语言问题：

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

在 `config.toml` 的 `entity_types` 数组中登记：

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

同步建议：若该类型参与定义域/值域校验，请在 `[type_system]` 与 `[[predicate_profiles]]` 中同步声明别名与层级。

### 添加新的推理规则

1. 在 `[[prolog_rules]]` 中定义模式与 `query_type`：

```toml
[[prolog_rules]]
name = "公司查询"
pattern = ["n", "的", "创始人"]
query_type = "founder_query"
priority = 10
```

2. 确认 `PrologEngine::generate_prolog_query` 能生成对应查询，或在该函数中补充分支。

3. 为规则补充问句样例，验证命中顺序与实体捕获；同时把无解与故障样例加入测试，避免被同一个空结果吞掉。

### 自定义同义词

在 `config.toml` 的 `[synonyms.mappings]` 中添加扁平映射：

```toml
[synonyms.mappings]
"创建" = "成立"
"建立" = "成立"
"CEO" = "首席执行官"
```

进阶：如需词义感知归一化（区分「苹果」是水果还是公司），请配置 `[[synonym_graph.senses]]` 与 `SynonymContext`，详见 `src/synonym_graph.rs` 与 `PLAN.md#9-扩展指引`。

## 📊 数据模型与 RDF 转换

系统使用 RDF 三元组存储，命名空间统一为 `http://example.org/kg/`（即 `kg:`）。

### 实体表示

```turtle
kg:entity/小明
    rdf:type kg:Person ;
    kg:name "小明" ;
    kg:age "18" .
```

### 关系表示

```turtle
kg:entity/张三
    kg:teacher_of
    kg:entity/李四 .
```

### URI 模式

| 种类 | 模式 | 示例 |
|------|------|------|
| 实体 | `kg:entity/{名称}` | `kg:entity/小明` |
| 属性 | `kg:{属性名}` | `kg:name`、`kg:age` |
| 类型 | `kg:{类型名}` | `kg:Person`、`kg:Company` |

### 数据类型映射

| 逻辑类型 | RDF 表示 | 示例 |
|----------|----------|------|
| 字符串 | 简单字面量 | `"小明"` |
| 整数 | `^^xsd:integer` | `"18"^^xsd:integer` |
| 小数 | `^^xsd:decimal` | `"1.75"^^xsd:decimal` |
| 日期 | `^^xsd:date` | `"1999-01-01"^^xsd:date` |
| 布尔 | `^^xsd:boolean` | `"true"^^xsd:boolean` |

### 转换流程

1. **自然语言文本 → RDF**：分词 → 实体识别 → 关系抽取 → 三元组生成
2. **JSON-LD → RDF**：解析 `@context` → 展开 URI → 递归提取三元组（支持 `@graph`、嵌套对象、列表）
3. **RDF 文件导入**：支持 Turtle (`.ttl`)、N-Triples (`.nt`)、N-Quads (`.nq`)、TriG (`.trig`)、RDF/XML (`.rdf`)

### 批量导入

- 交互模式：`导入文件：<path>`、`导入内容：<json>`
- 命令行：`import-jsonld`、`import-rdf`、`export-rdf`，支持标准输入/输出管道

## 🔍 SPARQL 查询示例

查询所有人物及其年龄：

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

查询特定关系：

```sparql
PREFIX kg: <http://example.org/kg/>

SELECT ?from ?to
WHERE {
    ?from kg:teacher_of ?to .
}
```

## 🐛 故障排除

### 常见问题

1. **编译错误**：确保 Rust 版本 ≥ 1.70
   ```bash
   rustup update stable
   rustc --version
   ```

2. **存储路径错误**：确保 `oxigraph_data` 目录存在且可写
   ```bash
   mkdir -p oxigraph_data
   chmod 755 oxigraph_data
   ```

3. **中文分词问题**：首次运行会自动下载 jieba-rs 词典

4. **JSON-LD 导入失败**：检查格式是否符合标准，确保 `@context` 映射正确

5. **Prolog 推理异常**：检查规则配置，确认事实已正确加载；查看 `cargo run --release -- ask "..." --explain` 的调试输出

### 日志调试

启用详细日志：

```bash
RUST_LOG=debug cargo run --release
```

### 性能建议

- 发布模式编译：`cargo build --release`
- 定期清理旧数据：`rm -rf oxigraph_data/*`（谨慎操作）
- 合理配置实体类型与抽取规则数量，避免过度匹配

## 🔧 开发与测试

### 项目结构（核心）

```
src/
├── main.rs                      # CLI 入口（include! 分片）
├── lib.rs                       # 核心库，模块声明与 include! 组合
├── nlp.rs                       # 中文分词、词性标注、同义词归一化
├── prolog_engine.rs             # Prolog 推理引擎适配层
├── lib/                         # lib.rs 的分片实现
│   ├── lib_part_01.rs           # 配置、工具函数、RDF 结果渲染
│   ├── lib_part_02.rs           # OxigraphManager 存储实现
│   ├── lib_part_03.rs           # 直答缓存与事实快照
│   └── lib_part_04.rs           # 查询实体提取、响应格式化、配置加载
├── dialogue_manager/            # 多轮对话管理（分片实现）
├── main_parts/                  # main.rs 分片（CLI/交互/子命令）
└── ...                          # 其余 100+ 模块按职责组织

tests/
├── integration_test.rs          # 端到端集成测试
├── question_analysis_test.rs    # 问题分析测试
└── semantic_understanding_test.rs # 语义理解测试

config.toml                      # 系统配置
CLAUDE.md                        # 开发指南
PLAN.md                          # 维护者设计文档
```

### 代码质量检查

```bash
# 代码格式化
cargo fmt

# 代码检查
cargo clippy

# 类型检查（所有目标）
cargo check --all-targets

# 运行测试
cargo test

# 文档生成
cargo doc --open
```

### 提交前检查清单

```bash
cargo check --all-targets
cargo test
cargo fmt --check
```

约定：
- 库代码禁止 `unsafe`（`src/lib.rs` 已声明 `#![forbid(unsafe_code)]`）
- 公共接口与行为保持稳定，改动优先消除重复、澄清命名、补充注释
- 报告类输出保持确定性：排序固定、计数可复现、不依赖挂钟阈值
- 新增解析或推理分支时，优先补充单元测试与端到端用例
- 涉及配置的改动同时运行配置契约检查，确认没有悬空的实体类型或查询类型
- 不以修改测试、放宽断言或删除用例的方式让检查通过

## 📚 技术栈

| 组件 | 库 | 说明 |
|------|------|------|
| 语言 | Rust | 系统编程语言，内存安全且高性能 |
| 异步运行时 | Tokio | 全功能异步运行时 |
| RDF 存储 | Oxigraph | 高性能三元组存储，支持 SPARQL 1.1 |
| Prolog 引擎 | Scryer Prolog | 纯 Rust 实现的 ISO Prolog |
| 中文分词 | jieba-rs | 结巴中文分词 Rust 移植 |
| 序列化 | Serde / serde_json | 统一序列化框架 |
| HTTP 客户端 | Reqwest | 异步 HTTP 客户端 |
| 正则表达式 | regex | PCRE 兼容正则 |
| 命令行解析 | clap | 声明式 CLI 解析 |
| 配置解析 | toml | TOML 解析与序列化 |
| 唯一标识 | uuid | UUID v4 生成 |
| 并发原语 | parking_lot | 更快的锁与 RWLock |
| 加密 | aes-gcm / rand | 对称加密与随机数 |
| JSON-LD | json-ld | JSON-LD 处理 |

## 🤝 贡献指南

欢迎提交 Issue 与 Pull Request！

### 贡献流程

1. Fork 项目到你的 GitHub 账号
2. 创建功能分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 开启 Pull Request

### 代码规范

- 遵循 Rust 官方代码风格（`cargo fmt` / `cargo clippy`）
- 公共接口添加文档注释（`///`）
- 为新功能编写对应测试用例
- 确保所有测试通过（`cargo test`）
- 提交前运行检查清单（`cargo check --all-targets && cargo test && cargo fmt --check`）

### 问题报告

报告问题时请包含：
- 操作系统与 Rust 版本（`rustc --version`）
- 详细错误信息与堆栈
- 最小复现步骤
- 相关配置文件片段（脱敏后）

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Oxigraph](https://github.com/oxigraph/oxigraph) - 高性能 RDF 存储
- [Scryer Prolog](https://github.com/mthom/scryer-prolog) - Rust 实现的 Prolog 引擎
- [jieba-rs](https://github.com/messense/jieba-rs) - 中文分词库

## 📮 联系方式

- 项目主页: https://github.com/183600/KnowProlog
- Issue 追踪: https://github.com/183600/KnowProlog/issues

---

> **文档说明**：本 README 基于项目代码与配置文件分析生成。如有不准确之处，欢迎提交 Issue 或 PR 修正。

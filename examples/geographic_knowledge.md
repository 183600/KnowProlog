# 地理知识演示

## 录入地理知识

```
> 录入：法国的首都是巴黎
实体 - 类型: Country, 属性: {"name": "法国", "capital": "巴黎"}
知识已成功保存

> 录入：中国有14亿人口
实体 - 类型: Country, 属性: {"name": "中国", "population": "14"}
知识已成功保存
```

## 查询地理知识

```
> 法国的首都
查询类型: capital_query, 实体: ["法国"]
Prolog查询: country_capital('法国', Capital)
首都是 巴黎

> 中国有多少人
查询类型: population_query, 实体: ["中国"]
Prolog查询: country_population('中国', Population)
人口是 14 人
```

## 扩展更多领域

要支持更多问题类型，可以继续添加：

### 科学知识
```toml
[[entity_types]]
label = "Element"
examples = ["氢的原子序数是1", "氧的化学符号是O"]

[[entity_types.properties]]
name = "atomic_number"
data_type = "integer"
extraction_patterns = ["原子序数", "atomic number"]

[[entity_types.properties]]
name = "chemical_symbol"
data_type = "string"
extraction_patterns = ["化学符号", "symbol"]
```

### 历史知识
```toml
[[entity_types]]
label = "HistoricalEvent"
examples = ["第二次世界大战结束于1945年"]

[[entity_types.properties]]
name = "year"
data_type = "integer"
extraction_patterns = ["结束于", "发生于"]
```

### 技术知识
```toml
[[entity_types]]
label = "Technology"
examples = ["Python语言诞生于1991年"]

[[entity_types.properties]]
name = "creation_year"
data_type = "integer"
extraction_patterns = ["诞生于", "发明于"]
```
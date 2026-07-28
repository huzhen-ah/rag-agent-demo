# Milvus 数据与混合检索设计

本文说明 `langchain-rag` 中 Milvus 的数据结构、BM25 Function、索引、混合检索和数据生命周期。

## 1. 设计目标

当前项目需要在同一个 Collection 中支持两种检索：

```text
Dense Retrieval
→ 处理语义相似

BM25 Retrieval
→ 处理关键词、编号和专有名词匹配
```

两路结果通过 RRF 融合：

```text
Dense排名 ─┐
           ├→ RRF → Top 10
BM25排名 ──┘
```

Milvus 负责：

- 保存 chunk 文本、metadata 与向量字段。
- 为 Dense 向量建立 HNSW 索引。
- 对原始文本执行 Analyzer 与 BM25 Function。
- 为 Sparse 字段建立倒排索引。
- 执行 Dense 与 BM25 两路搜索。
- 使用 RRF 融合两路排名。

## 2. Database 与 Collection

当前连接：

```text
URI: http://localhost:19530
Database: rag_demo
Collection: rag_chunks
```

层级关系：

```text
Milvus Standalone
└── rag_demo
    └── rag_chunks
```

`rag_demo` 和 `rag_chunks` 都是项目自定义名称，不是 Milvus 内置对象。

当没有显式指定 Database 时，MilvusClient 默认使用 `default`。当前项目显式创建并连接 `rag_demo`，用于将本项目数据与其他 Collection 隔离。

## 3. Database 创建

`create_database()` 分两步连接：

```python
client = MilvusClient(
    uri=uri,
    token="{}:{}".format(user, pwd),
)
```

先连接 Milvus 服务，检查 Database：

```python
if db_name not in client.list_databases():
    client.create_database(db_name)
```

然后重新创建指向指定 Database 的 Client：

```python
client = MilvusClient(
    uri=uri,
    db_name=db_name,
    user=user,
    password=pwd,
)
```

返回的 `client` 后续操作：

```text
rag_demo
```

而不是默认 Database。

## 4. Collection Schema

当前 Schema：

```text
rag_chunks
├── pk       INT64                 Primary Key
├── text     VARCHAR               Chunk原文
├── dense    FLOAT_VECTOR          Dense向量
└── sparse   SPARSE_FLOAT_VECTOR   BM25稀疏表示
```

创建参数：

```python
schema = client.create_schema(
    auto_id=True,
    enable_dynamic_field=True,
)
```

### 4.1 `auto_id=True`

Milvus 自动为每条数据生成整数主键。

应用程序插入 Document 时不需要提供：

```python
{"pk": ...}
```

`add_documents()` 返回的 ID 列表就是 Milvus 生成的主键。

### 4.2 `enable_dynamic_field=True`

Document metadata 不需要全部提前声明为固定 Schema 字段。

例如：

```python
{
    "source": "documents/THREE_WEEK_PLAN.md",
    "start_index": 500,
}
```

PDF 还可能包含：

```python
{
    "page": 0,
}
```

这些 metadata 可以通过动态字段保存。

## 5. `text` 字段与 Analyzer

```python
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=65535,
    enable_analyzer=True,
    analyzer_params={"type": "chinese"},
)
```

`text` 有两个用途：

```text
1. 保存chunk原文
2. 作为BM25 Function输入
```

`enable_analyzer=True` 允许 Milvus 对该字段执行文本分析。

当前 Analyzer：

```python
{"type": "chinese"}
```

Analyzer 会影响：

- 文本如何分词。
- 英文大小写、标点和特殊字符如何处理。
- 哪些词项进入 BM25 统计。
- Query 与 Document 能否在相同词项上匹配。

因此，即使 BM25 公式和参数相同，不同 Analyzer 也可能产生不同排名。

## 6. Dense 字段

```python
schema.add_field(
    field_name="dense",
    datatype=DataType.FLOAT_VECTOR,
    dim=dim,
)
```

Dense 向量由本地 Qwen3-Embedding-0.6B 在 Python 客户端生成：

```text
text
→ HuggingFaceEmbeddings.embed_documents()
→ Dense Vector
```

Query 使用：

```text
question
→ HuggingFaceEmbeddings.embed_query()
→ Query Dense Vector
```

当前模型输出经过归一化：

```python
normalize_embeddings=True
```

项目使用 COSINE Metric 搜索 Dense 字段。

### 6.1 Dimension

Dense 向量必须具有固定维度。

当前代码通过：

```python
embedding_dim = len(
    embeddings.embed_query("测试")
)
```

获得实际维度，并将它传入 Schema：

```python
create_schema(
    client,
    dim=embedding_dim,
)
```

这样 Schema 与实际 Embedding 模型保持一致。

## 7. Sparse 字段

```python
schema.add_field(
    field_name="sparse",
    datatype=DataType.SPARSE_FLOAT_VECTOR,
)
```

Sparse 字段不需要声明：

```python
dim=...
```

### 7.1 逻辑理解

可以把 BM25 Sparse 表示理解为：

```text
每个词项对应一个坐标
每个chunk只保存非零坐标
```

例如，假设逻辑词表是：

```text
[公积金, 公司, 缴纳, 工资, 基数]
```

某个 chunk 的词频概念上是：

```text
[1, 2, 1, 0, 0]
```

稀疏形式只保留非零项：

```python
{
    0: 1.0,
    1: 2.0,
    2: 1.0,
}
```

这能帮助理解 Sparse Vector 与词频表示的关系，但不应据此断言 Milvus 某个版本的全部内部编码和存储细节。具体实现以对应版本源码与官方接口保证为准。

### 7.2 为什么不声明固定 Dimension

Dense Vector：

```text
每一维都有值
长度固定
必须声明dim
```

Sparse Vector：

```text
只保存非零坐标
词项空间可扩展
Schema不要求声明dim
```

所有 Sparse 数据仍然需要共享一致的坐标含义。对于内置 BM25，这套词项处理和统计由 Milvus 管理。

## 8. Milvus BM25 Function

Schema 中定义：

```python
bm25_function = Function(
    name="bm25_function",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
)

schema.add_function(bm25_function)
```

它声明：

```text
输入：text
处理：Milvus BM25 Function
输出：sparse
```

写入原始文本时：

```text
text
→ Analyzer
→ 词项统计
→ Sparse表示
```

搜索时：

```text
Query文本
→ 相同Analyzer
→ BM25检索
→ 排名结果
```

完整 BM25 还需要语料级统计，例如文档频率和平均文档长度。Sparse 字段不应被简单描述为“完整 BM25 分数”；最终相关性分数在搜索过程中结合 BM25 统计计算。

## 9. 为什么代码中出现两个 BM25 Function 对象

Schema 层：

```python
from pymilvus import Function
```

```python
Function(
    name="bm25_function",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
)
```

它用于创建 Milvus Collection 的真实服务端 Function。

LangChain 层：

```python
from langchain_milvus import BM25BuiltInFunction
```

```python
BM25BuiltInFunction(
    function_name="bm25_function",
    input_field_names="text",
    output_field_names="sparse",
    analyzer_params={"type": "chinese"},
)
```

它是 LangChain Milvus 适配器对同一条服务端能力的描述。

两者关系：

```text
BM25BuiltInFunction
→ 让LangChain知道如何使用text与sparse字段

pymilvus.Function
→ 在Collection Schema中定义真实服务端Function
```

当前项目选择手动创建 Schema，所以两个 API 层都需要描述这条关系。服务端并不会因此执行两次 BM25。

如果完全让 LangChain 自动创建 Collection，则可以由 `BM25BuiltInFunction` 帮助生成相应 Schema；当前项目没有采用这种方式。

## 10. Dense 索引

```python
index_params.add_index(
    field_name="dense",
    index_name="dense_hnsw_index",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 64,
    },
)
```

参数职责：

```text
M
→ 构建图时每个节点维护的邻接规模相关参数

efConstruction
→ 构建索引时的搜索候选规模相关参数
```

通常：

```text
参数增大
→ 可能提高召回
→ 同时增加构建时间、内存或索引体积
```

具体效果需要在实际数据集上评测，不能仅凭默认值判断。

### 10.1 查询参数 `ef`

```python
{
    "metric_type": "COSINE",
    "params": {
        "ef": 64,
    },
}
```

`ef` 用于 HNSW 查询阶段，与 `efConstruction` 不是同一个参数：

```text
efConstruction
→ 建索引时使用

ef
→ 查询时使用
```

## 11. Sparse 索引

```python
index_params.add_index(
    field_name="sparse",
    index_name="sparse_bm25_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)
```

Sparse 倒排索引用于：

```text
词项
→ 找到包含该词项的Documents
→ 计算并筛选BM25候选
```

它和 Dense HNSW 解决不同问题：

```text
HNSW
→ Dense语义邻近搜索

SPARSE_INVERTED_INDEX
→ Sparse词项检索
```

## 12. VectorStore 配置

```python
vector_store = Milvus(
    embedding_function=embeddings,
    builtin_function=bm25_function,
    vector_field=["dense", "sparse"],
    search_params=[
        {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
        {
            "metric_type": "BM25",
            "params": {},
        },
    ],
    ...
)
```

三个配置需要对齐：

```text
vector_field[0] = dense
search_params[0] = COSINE + ef
embedding_function = Qwen3 Embedding
```

```text
vector_field[1] = sparse
search_params[1] = BM25
builtin_function = BM25BuiltInFunction
```

如果字段顺序、Metric 或索引定义不一致，搜索可能报 Metric 不匹配等错误。

## 13. 写入流程

```python
ids = vector_store.add_documents(documents)
```

输入：

```text
List[Document]
```

每个 Document 包含：

```text
page_content
metadata
```

处理过程：

```text
page_content
├→ Embedding客户端 → dense
└→ Milvus服务端Function → sparse

metadata
└→ Milvus动态字段
```

输出：

```text
Milvus主键列表
```

当前代码只打印：

```python
len(ids)
```

## 14. 单路搜索

### 14.1 Dense 搜索

Dense 搜索需要先生成 Query Vector：

```python
query_vector = embeddings.embed_query(
    "第一周学习什么"
)
```

然后指定：

```python
anns_field="dense"
```

`anns_field` 表示本次 ANN Search 使用哪个向量字段。

### 14.2 BM25 搜索

内置 BM25 可以直接传入 Query 文本：

```python
results = client.search(
    collection_name="rag_chunks",
    data=["第一周学习什么"],
    anns_field="sparse",
    search_params={
        "metric_type": "BM25",
        "params": {},
    },
    limit=5,
    output_fields=["text"],
)
```

这里：

```text
anns_field = sparse
→ 使用Sparse字段和BM25索引

limit = 5
→ 对当前Query最多返回5条结果
```

如果 `data` 中包含多个 Query，每个 Query 分别返回自己的 Top 5。

## 15. 混合搜索

LangChain VectorStore 配置两个向量字段后：

```python
vector_field=["dense", "sparse"]
```

调用：

```python
vector_store.similarity_search(
    query,
    k=10,
    ranker_type="rrf",
    ranker_params={"k": 60},
)
```

会执行：

```text
Dense Search
+
BM25 Search
+
RRF
```

转换为 Retriever 后：

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
        "ranker_type": "rrf",
        "ranker_params": {"k": 60},
    },
)
```

调用：

```python
documents = retriever.invoke(query)
```

返回：

```text
List[Document]
```

## 16. RRF

RRF 不直接比较 Dense 相似度与 BM25 分数，而是使用各路排名：

```text
RRF(d) = Σ 1 / (c + rank_i(d))
```

当前：

```text
c = 60
```

代码参数：

```python
ranker_params={"k": 60}
```

这里的 `k=60` 是 RRF 平滑常数，不是返回数量。

返回数量由外层参数决定：

```python
k=10
```

候选 Document 来自各路检索结果的合并与融合。某个 Document 同时出现在 Dense 和 BM25 前排时，会从两路排名中累计贡献。

## 17. Collection 加载

```python
client.load_collection("rag_chunks")
```

表示让 Collection 进入可搜索状态，将所需数据和索引加载到查询节点。

对应操作：

```python
client.release_collection("rag_chunks")
```

表示从查询节点释放，不会删除持久化数据。

三种操作不能混淆：

```text
load_collection
→ 加载以供搜索

release_collection
→ 释放查询资源

drop_collection
→ 删除整个Collection
```

LangChain Milvus VectorStore 通常会在需要时处理 Collection 加载。独立使用 `MilvusClient.search()` 验证时，可以显式调用 `load_collection()`。

## 18. 数据生命周期

### 18.1 `demo.py`

```text
Collection不存在
→ 创建并入库

Collection已存在
→ 直接复用
```

### 18.2 `ingest.py`

```python
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
```

因此每次运行：

```bash
python ingest.py
```

都会：

```text
删除旧Collection
→ 重建Schema
→ 重建索引
→ 全量写入
```

这是开发阶段的全量重建策略，不是增量更新。

### 18.3 `delete`

```python
client.delete(
    collection_name="rag_chunks",
    filter="pk in [1, 2, 3]",
)
```

只删除符合条件的数据，保留：

- Collection。
- Schema。
- Function。
- 索引定义。
- 其他数据。

而：

```python
client.drop_collection("rag_chunks")
```

删除整个 Collection。

## 19. 如何验证入库成功

### 19.1 查看数据量

```python
print(
    client.get_collection_stats("rag_chunks")
)
```

重点检查：

```text
row_count > 0
```

### 19.2 查看 Schema

```python
print(
    client.describe_collection("rag_chunks")
)
```

确认存在：

```text
text
dense
sparse
BM25 Function
```

### 19.3 查看索引

```python
print(
    client.list_indexes("rag_chunks")
)
```

确认存在：

```text
dense_hnsw_index
sparse_bm25_index
```

### 19.4 执行 BM25 搜索

```python
results = client.search(
    collection_name="rag_chunks",
    data=["第一周学习什么"],
    anns_field="sparse",
    search_params={
        "metric_type": "BM25",
        "params": {},
    },
    limit=5,
    output_fields=["text"],
)
```

能够返回相关文本，说明：

```text
text已写入
→ BM25 Function可用
→ sparse检索链路可用
→ Sparse索引可搜索
```

## 20. 为什么看不到 Sparse 具体值

在 Attu 中可以看到：

- `sparse` 字段定义。
- BM25 Function。
- Sparse 索引。

但 BM25 Function 生成的 Sparse 字段不能像普通标量字段一样通过 `output_fields=["sparse"]` 取回具体值。

因此不要通过“是否能打印 Sparse Vector”判断入库是否成功。应通过：

```text
Schema
+ Function
+ Index
+ row_count
+ 实际BM25搜索
```

共同验证。

## 21. 一致性与认证

当前 Collection 和 VectorStore 使用：

```text
consistency_level = Strong
```

这有利于本地开发阶段在写入后立即读取最新数据，但在更大规模系统中，一致性级别需要结合延迟与吞吐需求选择。

当前代码连接信息：

```text
user = root
password = Milvus
```

这是本地演示配置。生产环境不应在代码中硬编码管理员账号，应使用：

- 环境变量或密钥管理系统。
- 应用专用账号。
- 最小权限。
- Milvus 认证与网络访问控制。

## 22. 当前边界

当前设计没有实现：

- 增量写入和文档版本管理。
- 稳定的业务主键。
- Collection Schema 自动迁移。
- Metadata 权限过滤。
- 多租户隔离。
- Analyzer 领域词典与同义词配置。
- Dense/BM25 单路与融合结果的框架版评测。
- 针对大规模数据的索引参数实验。

当前实现的目标是：

```text
明确Schema
→ 跑通Dense + BM25
→ 使用RRF融合
→ 为后续Reranker和生成提供候选
```

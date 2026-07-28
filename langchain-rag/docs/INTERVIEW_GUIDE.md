# 项目讲解与面试问答

本文用于帮助项目作者准确介绍 `langchain-rag`。重点不是背诵 LangChain 类名，而是说明系统解决了什么问题、数据如何流动、每个组件承担什么职责，以及当前版本有哪些边界。

## 1. 30 秒项目介绍

可以这样介绍：

> 这是一个基于 LangChain 和 Milvus Standalone 实现的本地 RAG Demo。系统先自行解析 TXT、Markdown 和 PDF，再用 LangChain 切分文档；入库时同时生成 Dense 向量和 Milvus 原生 BM25 Sparse 表示。查询时执行 Dense 与 BM25 混合检索，用 RRF 融合两路排名，再用本地 CrossEncoder 重排，最后把 Top 文档作为上下文交给本地 Qwen3-1.7B 生成答案。

这段话已经包含完整主链路：

```text
加载
→ 切分
→ Dense与BM25入库
→ 混合检索
→ RRF
→ CrossEncoder重排
→ LLM生成
```

## 2. 2 分钟项目介绍

可以按以下顺序展开。

### 2.1 为什么做两个版本

项目先实现了手写版 RAG，用于理解：

- Dense 检索。
- BM25。
- RRF。
- CrossEncoder Reranker。
- MRR 与 Recall@K。

随后实现 LangChain + Milvus 版本，用于学习主流组件如何表达同一条 RAG 流程。

两个版本的关系不是重复造轮子，而是：

```text
手写版
→ 理解算法和数据流

LangChain + Milvus版
→ 学习工程组件、持久化和框架编排
```

### 2.2 离线入库

当前代码自行加载文件：

- TXT、Markdown：一个文件先形成一个 `Document`。
- PDF：每页先形成一个 `Document`。

之后使用 `RecursiveCharacterTextSplitter` 切成更小的 `Document`。

每个 Chunk 的正文存放在：

```python
document.page_content
```

来源、页码和切分起点等信息存放在：

```python
document.metadata
```

Milvus Collection 同时包含：

- 原文 `text`。
- Dense 向量字段 `dense`。
- Sparse 向量字段 `sparse`。

Dense 向量由本地 Qwen3 Embedding 模型在客户端生成；Sparse 表示由 Milvus BM25 Function 根据 `text` 在服务端生成。

### 2.3 在线查询

用户问题进入 Retriever 后，会同时用于：

- Dense 语义检索。
- BM25 关键词检索。

两路候选使用 RRF 按排名融合。RRF 不直接比较两种检索分数，因此可以避免 Dense 相似度和 BM25 分数尺度不同的问题。

融合后的前 10 条候选交给 CrossEncoder，CrossEncoder 同时读取：

```text
Query + Document
```

重新计算相关性，保留前 5 条。

最后将这 5 条文档拼接成 Context，与用户问题一起交给本地生成模型。

## 3. 这个项目解决什么问题

纯 LLM 存在两个直接问题：

- 不知道本地私有资料。
- 参数知识无法随资料变化实时更新。

RAG 的作用是先从外部知识库找证据，再基于证据回答。

当前项目主要验证：

```text
非结构化文件
→ 可检索知识
→ 相关证据
→ 基于证据生成
```

它不是完整生产系统。目前没有多租户、权限控制、增量更新、监控、并发服务和前端。

## 4. 为什么同时使用 Dense 和 BM25

### 4.1 Dense 的优势

Dense 检索基于 Embedding，可以找到语义相近但字面不完全相同的文本。

例如：

```text
Query：怎样搭建智能体？
Document：Agent系统的实现步骤……
```

即使关键词不完全一致，Dense 仍可能召回。

### 4.2 BM25 的优势

BM25 对明确关键词、产品名、错误码、专有名词和数字通常更敏感。

例如：

```text
Qwen3-Embedding-0.6B
SPARSE_INVERTED_INDEX
19530
```

这些精确词项可能更适合关键词检索。

### 4.3 为什么不是二选一

两种方法的错误具有互补性：

```text
Dense
→ 擅长语义

BM25
→ 擅长词项匹配
```

混合检索的目标不是保证每次都优于单路，而是提高不同类型查询下的召回稳定性。

## 5. BM25 到底做了什么

BM25 的核心仍然是基于词项的相关性：

```text
一个Query词在当前文档中出现得越充分
→ 贡献通常越大

这个词在整个语料中越少见
→ 区分能力通常越强
```

与最朴素 TF-IDF 相比，BM25 还会：

- 对词频增长做饱和处理。
- 根据文档长度进行归一化。

当前 LangChain 版没有手写 BM25 公式。它使用 Milvus 原生 BM25 Function：

```text
text
→ Analyzer分词
→ 语料统计
→ Sparse表示
→ BM25搜索
```

因此可以说掌握 BM25 原理，但不能声称当前版本的 BM25 是自己实现的。

## 6. Sparse Vector 是什么

可以把词表中的每个词理解成一个维度。

一段文本只包含词表中的少数词，所以只有少数维度非零：

```text
{
    18: 0.73,
    205: 1.42,
    9017: 0.56
}
```

这只是概念示例，维度编号和数值不是当前项目的真实数据。

Sparse Vector 不等于简单 TF。它是供稀疏检索使用的向量表示，具体权重由 Milvus BM25 流程生成并用于 BM25 评分。

Milvus 的 `SPARSE_FLOAT_VECTOR` 不要求像 Dense 向量那样在 Schema 中声明固定 `dim`。

## 7. 为什么需要 RRF

Dense 和 BM25 的原始分数不能直接相加：

```text
Dense分数
→ 由向量相似度产生

BM25分数
→ 由词项相关性产生
```

两者不在同一个数值尺度上。

RRF 只看每个候选在各路结果中的排名：

```text
RRF_score(d)
= 各检索路线中 1 / (k + rank(d)) 的总和
```

一个 Chunk 如果在两路中都排名靠前，融合分数会更高。

RRF 操作的候选集合是 Dense 结果与 BM25 结果的并集。只出现在一路中的 Chunk 也可以参与排序，只是另一条检索路线不为它贡献分数。

当前代码中的两个 `k` 含义不同：

```python
top_k=10
```

表示混合 Retriever 最终取前 10 条候选。

```python
rrf_k=60
```

是 RRF 公式中的平滑常数，不是返回 60 条。

## 8. 为什么 RRF 后还要 Reranker

第一阶段检索的目标是：

```text
从全部Chunk中快速召回一小批候选
```

CrossEncoder 的目标是：

```text
对候选进行更精细的相关性判断
```

CrossEncoder 会把 Query 和每个候选文档放在一起编码，因此能建模两者之间更细的交互，但计算开销比向量检索大。

所以合理顺序是：

```text
全库
→ Dense + BM25快速召回
→ RRF Top 10
→ CrossEncoder精排
→ Top 5
```

不直接对全库运行 CrossEncoder，是因为成本会随文档数量线性增长。

## 9. 为什么叫 CrossEncoder

因为 Query 与 Document 会进入同一个模型共同编码：

```text
[Query, Document]
→ Transformer
→ 相关性分数
```

它与 Dense 检索常用的双塔方式不同：

```text
Query
→ 一个向量

Document
→ 预先计算的另一个向量

两个向量
→ 相似度
```

双塔适合大规模召回；CrossEncoder 适合小规模精排。

## 10. 为什么使用 Milvus

手写版可以把向量留在内存中，但它不适合持续保存和管理更大规模数据。

Milvus 在当前项目中负责：

- 持久化原文和向量。
- 管理 Collection Schema。
- 建立 Dense HNSW 索引。
- 建立 Sparse 倒排索引。
- 执行 Dense 搜索。
- 执行 BM25 搜索。
- 执行 RRF 混合排序。

本项目使用的是 Milvus Standalone，而不是 Lite 或 Kubernetes 集群。

选择 Standalone 的原因是：

```text
比Lite更接近独立数据库服务
同时不引入Kubernetes运维复杂度
```

## 11. HNSW 可以怎么解释

HNSW 是一种基于多层近邻图的近似最近邻索引。

全局理解即可：

```text
上层图节点少
→ 用于快速跳到可能相关的区域

下层图节点多
→ 在局部继续寻找更近的向量
```

它通过少检查一部分向量来提高速度，因此是近似搜索，不保证每次都找到理论上的绝对最近邻。

当前参数：

- `M=16`：影响每个节点维护的邻接关系规模。
- `efConstruction=64`：影响建图时搜索候选的范围。
- `ef=64`：影响查询时考察的候选范围。

通常候选范围越大，召回可能越好，但建索引或查询成本也会增加。

## 12. LangChain 在项目中负责什么

LangChain 没有替代所有业务逻辑。

当前使用它完成：

- 标准 `Document` 数据结构。
- 文档切分器。
- Embedding 接口包装。
- Milvus VectorStore 接入。
- Retriever 统一接口。
- Reranker 组件包装。
- 本地 Hugging Face 模型包装。
- Prompt、Runnable 和输出解析。
- LCEL Chain 编排。

当前仍由项目代码决定：

- 如何解析不同文件。
- Chunk 参数。
- Milvus Schema 和索引参数。
- 使用 Dense + BM25。
- RRF 参数。
- 召回数量和重排数量。
- Prompt 内容。
- 模型路径和设备。
- 首次入库与后续复用逻辑。

## 13. `VectorStore` 为什么可以产生 Retriever

`VectorStore` 表示向量存储及搜索能力。

```python
vector_store.as_retriever(...)
```

不是把数据库变成另一个数据库，而是返回一个遵守 LangChain Retriever 接口的适配器。

可以理解为：

```text
Milvus搜索方法
→ Retriever统一调用接口
```

上层 Reranker 和 Chain 不再需要关心底层是 Milvus、其他向量库还是自定义检索器。

## 14. 为什么 Retriever 使用 `invoke`

LangChain 将很多可执行组件统一为 Runnable。

因此：

```python
retriever.invoke(query)
```

表示执行 Retriever，并返回 `Document` 列表。

`invoke` 这个名字不是检索领域术语，而是 LangChain 为 Prompt、Retriever、Model、Parser 和 Chain 提供的统一执行接口。

## 15. Reranker 为什么包装了多层

当前对象关系：

```text
HuggingFaceCrossEncoder
→ 提供Query-Document打分模型

CrossEncoderReranker
→ 用分数重新排序并保留Top N

ContextualCompressionRetriever
→ 先调用基础Retriever，再调用Reranker
```

`ContextualCompressionRetriever` 是整个重排检索入口。

这里的 “Compression” 不是压缩字符串，也不是摘要，而是从候选文档中筛掉较不相关的内容。

## 16. LCEL Chain 怎么解释

当前核心结构：

```python
{
    "context": rerank_retriever | format_documents,
    "question": RunnablePassthrough(),
}
| prompt
| chat_model
| StrOutputParser()
```

输入问题分成两路：

```text
问题
├── Retriever → Reranker → format_documents → context
└── RunnablePassthrough → question
```

之后：

```text
context + question
→ Prompt
→ ChatModel
→ 字符串答案
```

这里的 `|` 被 LangChain Runnable 重载，用于组合执行流程，不是原生 Python 数据管道语法。

## 17. 为什么不直接把 Transformers Pipeline 交给 Chain

原始 Transformers `pipeline` 不天然遵守 LangChain 的模型接口。

当前包装关系：

```text
Transformers pipeline
→ HuggingFacePipeline
→ ChatHuggingFace
```

- `pipeline`：真正执行本地文本生成。
- `HuggingFacePipeline`：适配 LangChain 的 LLM 接口。
- `ChatHuggingFace`：再适配消息式 ChatModel 接口。

包装的目的主要是统一接口和接入 LCEL，不是提高模型能力。

## 18. 为什么自己加载文档

当前项目没有使用 `langchain-community` 中的通用文件 Loader，而是：

- 使用 Python 原生文件读取加载 TXT 和 Markdown。
- 使用 `pypdf` 解析 PDF。
- 自己构造 LangChain `Document`。

这样做可以明确控制：

- 一个原始文件先生成几个 `Document`。
- `metadata` 中保存哪些字段。
- PDF 页码从哪里开始。
- 不支持的文件如何处理。

需要如实说明：

```text
当前TXT和Markdown会一次性读入内存
```

因此它适合当前 Demo，不适合直接加载数 GB 的单个文本文件。大文件应改成流式读取或分段读取。

## 19. Chunk 参数怎么解释

当前参数：

```text
chunk_size = 500
chunk_overlap = 50
```

`chunk_size` 控制每个 Chunk 的目标长度；`chunk_overlap` 让相邻 Chunk 保留一部分重叠内容，降低知识点恰好被边界切断的风险。

这不是通用最优值。真实项目应根据：

- 文档类型。
- 文本语言。
- Embedding 模型。
- 问题粒度。
- 召回评测结果。
- LLM 上下文预算。

通过实验调整。

## 20. 如何评价检索效果

手写版已经使用：

- Recall@K。
- MRR。

### Recall@K

衡量前 K 条结果覆盖了多少相关文档：

```text
Recall@K
= 前K条中命中的相关文档数 / 全部相关文档数
```

它关注有没有召回。

### MRR

关注第一个相关结果出现得有多靠前：

```text
RR = 1 / 第一个相关结果的排名
MRR = 所有Query的RR平均值
```

它关注首个正确答案的位置。

评测并不是只能在 RRF 后做。可以分别评测：

```text
Dense
BM25
RRF
Reranker
```

这样才能判断效果提升或问题来自哪一层。

当前 LangChain 版尚未迁移一套正式评测数据和评测脚本，不能把手写版指标直接宣称为 LangChain 版指标。

## 21. 当前项目做得好的地方

可以客观描述为：

- 完整跑通本地端到端 RAG。
- 同时使用 Dense 与原生 BM25。
- 使用 RRF 解决异构检索分数融合问题。
- 使用 CrossEncoder 做二阶段排序。
- 显式设计 Milvus Schema、Function 和索引。
- 保留手写版，能够解释框架背后的算法。
- 入库、检索、重排和生成模块可以分阶段运行与排错。

不要泛化成：

- 生产级高并发系统。
- 大规模线上验证。
- 已完成自动化评测闭环。
- 已解决所有文档格式。

## 22. 当前项目的不足

面试中不必回避边界，可以直接说明：

- 文件加载不是流式的。
- 只有 TXT、Markdown 和文本型 PDF。
- 扫描 PDF 没有 OCR。
- 文档更新后需要全量重建 Collection。
- 没有稳定业务主键和去重策略。
- 没有正式 API 服务和前端。
- 凭据仍写在示例代码中。
- 没有并发、缓存和监控。
- 生成模型只有 1.7B，复杂指令能力有限。
- LangChain 版尚未接入正式评测集。

这些边界说明作者知道学习 Demo 与生产系统之间的差距。

## 23. 如果继续完善，优先做什么

合理优先级：

```text
1. 为LangChain版补检索评测
2. 增加稳定文档ID与增量更新
3. 将配置和凭据移出代码
4. 增加FastAPI服务
5. 增加日志、耗时和错误监控
6. 增加更多文档解析与OCR
7. 根据评测调Chunk和检索参数
```

如果下一阶段重点转向 Agent，则不必立刻把这个学习版本扩展成生产系统。

## 24. 常见追问

### 24.1 LangChain 和 Milvus 的区别是什么

LangChain 是应用组件与编排框架；Milvus 是向量数据库。

```text
LangChain
→ 连接和组织组件

Milvus
→ 保存、索引和检索数据
```

### 24.2 Embedding 模型是否必须自己实现

不必从零实现模型，但必须提供一个真正能计算向量的实现。

LangChain 的 `Embeddings` 是统一接口；当前项目使用 `HuggingFaceEmbeddings` 包装本地 Qwen3 Embedding 模型。

### 24.3 为什么 Query 和 Document 的 Embedding 配置不同

当前模型对查询使用 `query` Prompt，对文档不添加查询 Prompt，以匹配模型面向检索任务的编码方式。

两者最终必须处于同一个向量空间，才能计算相似度。

### 24.4 为什么 Dense 使用 COSINE

当前 Embedding 输出做了归一化，并使用 COSINE 衡量方向相似性。索引与查询都必须使用一致的 Metric。

### 24.5 为什么 Collection 要显式建 Schema

为了明确控制字段、向量维度、Analyzer、BM25 Function、索引类型和 Metric，而不是完全依赖自动推断。

### 24.6 `auto_id=True` 是什么

Milvus 自动为每行生成 `INT64` 主键。当前项目没有自行传入稳定 ID。

这适合快速跑通，但不利于按业务 ID 做幂等更新和删除。

### 24.7 `limit` 是什么

Milvus `search` 中的 `limit` 表示每个查询向量或查询文本最多返回多少条结果。

如果一次传入多个 Query，每个 Query 都有各自的 Top N 结果。

### 24.8 为什么看不到 Sparse 字段的具体内容

BM25 Function 生成的 Sparse 表示是检索中间数据，不能像普通标量字段那样直接通过 `output_fields` 取回完整内容。

应通过以下事实验证：

- Schema 中存在 Sparse 字段。
- BM25 Function 存在。
- Sparse 索引存在。
- Collection 有数据。
- 对 `sparse` 字段执行 BM25 搜索能够返回结果。

### 24.9 Reranker 会产生新文档吗

不会。它只对基础 Retriever 返回的候选重新打分、排序和截断。

### 24.10 如果 Dense 一路没有结果，还需要 RRF 吗

数学上仍可对剩余一路计算 RRF，但结果排序基本等价于该路原排名。

工程上可以继续走统一融合流程，也可以对单路做特殊处理；当前框架封装内部统一执行混合检索。

### 24.11 为什么不用更大的生成模型

当前目标是先在本地设备上跑通完整链路，1.7B 模型加载和调试成本较低。

如果重点变成答案质量，可以替换更强的本地模型或模型 API；Retriever 与 Chain 接口不需要全部重写。

### 24.12 框架做了这么多，还能体现个人能力吗

能力不体现在背诵构造函数，而体现在：

- 知道每层输入和输出。
- 能解释 Dense、BM25、RRF 和 Reranker。
- 能设计 Schema 和索引。
- 能定位召回、排序与生成问题。
- 知道框架封装的边界。
- 能基于评测做取舍。

手写版进一步证明了对核心检索算法的理解。

## 25. 不应该怎么说

以下说法不准确：

> LangChain 自动完成了整个 RAG。

实际情况是项目代码选择、配置并连接了各个组件。

> Sparse Vector 就是 TF。

Sparse Vector 是稀疏表示；当前权重由 Milvus BM25 流程生成，不等于简单词频。

> RRF 融合 Dense 和 BM25 的原始分数。

RRF 融合的是排名，不是直接相加原始分数。

> Reranker 在全库中重新搜索。

Reranker 只处理 Retriever 已召回的候选。

> 当前 LangChain 版已经有 MRR 1.0。

当前指标来自手写版实验，LangChain 版尚未建立对应评测结果。

> 当前系统支持任意大文件。

当前 TXT 和 Markdown 加载会一次性读取全文。

> 当前代码是生产级系统。

当前代码是以跑通流程和理解组件为目标的本地 Demo。

## 26. 面试时真正需要记住什么

不需要死记所有包路径，重点记住六层：

```text
1. Document与Chunk
2. Dense + BM25召回
3. RRF融合
4. CrossEncoder重排
5. Context + Question生成
6. 评测与排错
```

对于框架类名，只需要能够看懂当前对象关系：

```text
Milvus VectorStore
→ Retriever
→ ContextualCompressionRetriever
→ LCEL RAG Chain
```

真正重要的是能够回答：

- 数据从哪里来。
- 每一步输入和输出是什么。
- 为什么需要这一步。
- 参数改变会影响什么。
- 效果不好时先检查哪一层。
- 哪些能力是自己实现，哪些由框架或数据库提供。

## 27. 一句话总结

> 我先通过手写版理解 Dense、BM25、RRF、重排和评测，再用 LangChain 与 Milvus 将同一流程工程化，最终实现了本地混合检索、二阶段排序和生成式问答的端到端 Demo。

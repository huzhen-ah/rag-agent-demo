# 从手写 RAG 到 LangChain + Milvus

本文对照 `handwritten-rag` 与 `langchain-rag`，说明每个阶段如何迁移、哪些原理保持不变，以及哪些结果不能假设完全一致。

## 1. 迁移目标

手写版已经跑通：

```text
文档加载
→ 固定窗口切块
→ Dense Retrieval
→ BM25 Retrieval
→ RRF
→ CrossEncoder Reranker
→ Generator
```

框架版迁移目标：

```text
保留同一条RAG主流程
→ 使用LangChain标准接口组织组件
→ 使用Milvus保存和检索数据
→ 使用Milvus原生BM25与RRF
→ 保留本地Qwen3模型
```

本次迁移不是为了证明框架版的算法一定优于手写版，而是为了学习：

- Vector Database 数据建模。
- Dense 与 Sparse 多向量检索。
- LangChain VectorStore 和 Retriever。
- 标准 Reranker 与 ChatModel 包装。
- LCEL 组件编排。

## 2. 总体映射

| 阶段 | 手写版 | LangChain + Milvus 版 |
|---|---|---|
| 文档结构 | 自定义 `Document`、`Chunk` | `langchain_core.documents.Document` |
| 文档加载 | 自定义 TXT/MD/PDF 加载 | 自定义解析后生成 LangChain Document |
| 切分 | 自定义固定窗口与 overlap | `RecursiveCharacterTextSplitter` |
| Dense Embedding | 自定义 `Embedder` | `HuggingFaceEmbeddings` |
| Dense 存储 | Pickle + NumPy | Milvus `FLOAT_VECTOR` |
| Dense 搜索 | 全量矩阵乘法 | Milvus HNSW + COSINE |
| BM25 分词 | Jieba | Milvus Chinese Analyzer |
| BM25 统计 | 手写 TF、DF、IDF、avgdl | Milvus BM25 Function |
| BM25 索引 | 手写配置与候选检索 | `SPARSE_INVERTED_INDEX` |
| RRF | 手写排名融合 | Milvus RRF Ranker |
| Reranker 模型 | `sentence_transformers.CrossEncoder` | `HuggingFaceCrossEncoder` |
| Reranker 编排 | 自定义 `Reranker` | `CrossEncoderReranker` |
| 检索后重排 | 自定义 `RAG` 调用 | `ContextualCompressionRetriever` |
| 生成 | 手写 tokenize/generate/decode | Transformers Pipeline + LangChain ChatModel |
| Prompt 编排 | 字符串与 messages 手动构造 | `ChatPromptTemplate` |
| 完整流程 | 自定义 `RAG.answer_question()` | LCEL `rag_chain.invoke()` |

## 3. 文档模型迁移

### 3.1 手写版

手写版使用自己的数据结构区分：

```text
Document
Chunk
```

Chunk 保存：

- 文本。
- 来源。
- 整数 `chunk_id`。

### 3.2 LangChain 版

框架版统一使用：

```python
Document(
    page_content="文本",
    metadata={...},
)
```

原始 Document 与切分后的 chunk 都使用同一个类，通过所处阶段区分含义：

```text
切分前Document
→ 表示文件或PDF页面

切分后Document
→ 表示chunk
```

文本始终放在：

```python
document.page_content
```

附加信息放在：

```python
document.metadata
```

### 3.3 变化

手写版：

```text
类型名称直接表达Document与Chunk
```

框架版：

```text
统一Document接口减少类型数量
但需要根据流程位置判断它当前代表原文还是chunk
```

## 4. 文档加载迁移

两版都没有依赖通用 Loader 自动决定所有行为。

当前框架版仍然显式处理：

```text
.txt
.md
.pdf
```

区别是输出类型改为 LangChain `Document`。

TXT 和 Markdown：

```text
一个文件
→ 一个原始Document
```

PDF：

```text
一页
→ 一个原始Document
```

手写版 PDF 页码从 1 开始；当前 LangChain 版使用 `enumerate(reader.pages)`，metadata 中的页码从 0 开始。比较两版 metadata 时需要注意这一差异。

文档加载阶段不负责 overlap。Overlap 发生在后续 Text Splitter 阶段。

## 5. 切分迁移

### 5.1 手写版

手写版自己维护：

```text
window
overlap
start
end
```

### 5.2 LangChain 版

框架版使用：

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)
```

输出仍是 `List[Document]`。

### 5.3 不能假设结果逐块相同

当前实际配置已经不同：

```text
手写版：
chunk_size = 500
chunk_overlap = 100

LangChain版：
chunk_size = 500
chunk_overlap = 50
```

即使以后把两版参数调成相同，也不能默认 chunk 边界完全一致。

原因：

- 手写版按自己的窗口逻辑切分。
- `RecursiveCharacterTextSplitter` 会按照分隔符优先级寻找切分位置。
- 换行、段落和中文标点可能影响实际边界。

因此，迁移后的 chunk 数量、内容和相关性标签可能发生变化。

## 6. Embedding 迁移

两版使用同一个模型：

```text
Qwen3-Embedding-0.6B
```

### 6.1 手写版

自定义 `Embedder` 直接调用模型：

```text
文本
→ 模型编码
→ NumPy向量
```

### 6.2 LangChain 版

使用：

```python
HuggingFaceEmbeddings
```

统一接口：

```python
embed_documents(texts)
embed_query(query)
```

Embedding 配置保留：

```text
Document prompt = ""
Query prompt_name = "query"
normalize_embeddings = True
```

### 6.3 本质没有改变

`HuggingFaceEmbeddings` 是适配器，不是新的 Embedding 算法。

底层仍然是：

```text
Qwen3-Embedding
```

变化的是调用接口和 VectorStore 集成方式。

## 7. Dense 存储迁移

### 7.1 手写版

```text
chunks
+ chunk_embeddings
→ Pickle文件
```

查询时将全部 chunk embedding 加载到内存。

### 7.2 LangChain + Milvus 版

```text
text
dense
metadata
→ Milvus Collection
```

应用程序通过 VectorStore 操作 Collection，不再维护本地 Pickle Dense 索引。

### 7.3 工程差异

手写版适合：

- 小规模数据。
- 验证向量计算。
- 观察完整中间结果。

Milvus 适合：

- 持久化。
- 索引检索。
- 更大的数据规模。
- 多字段与 metadata。
- 后续增量写入和服务化扩展。

当前项目尚未实现完整增量更新，但 Milvus 提供了相应工程基础。

## 8. Dense 搜索迁移

### 8.1 手写版

归一化后使用：

```python
scores = chunk_embeddings @ query_embedding
```

这是对全部 chunk 的精确矩阵计算。

### 8.2 Milvus 版

使用：

```text
HNSW
+ COSINE
```

HNSW 属于近似最近邻索引。

### 8.3 结果不保证完全相同

即使：

- Embedding 模型相同。
- Query 相同。
- Chunk 相同。
- Metric 相同。

结果仍可能受到以下因素影响：

- HNSW 的近似搜索。
- `ef`。
- 索引构建参数。
- 数据写入与 Collection 状态。

小规模数据中结果可能非常接近，但不能把逐项一致作为接口保证。

## 9. BM25 迁移

### 9.1 手写版

手写版显式实现：

```text
Jieba分词
→ Chunk TF
→ DF
→ IDF
→ avgdl
→ 长度归一化
→ BM25 Score
```

可以直接观察：

- 分词结果。
- 词频表。
- 语料统计。
- 每个 Query 词的贡献。
- 最终分数。

### 9.2 Milvus 版

框架版使用：

```text
Milvus Chinese Analyzer
→ BM25 Function
→ SPARSE_FLOAT_VECTOR
→ SPARSE_INVERTED_INDEX
→ BM25 Search
```

应用程序提供原始文本，Milvus 服务端维护文本分析、语料统计和 Sparse 检索。

### 9.3 原理相同，细节不保证相同

相同点：

```text
都基于词项匹配
都考虑词频和文档频率
都考虑文档长度
都使用BM25相关性
```

不同点可能包括：

- Jieba 与 Milvus Chinese Analyzer 的分词结果。
- 文本归一化。
- 停用词处理。
- BM25 参数。
- IDF 与统计更新细节。
- 倒排索引实现。

因此：

```text
手写BM25与Milvus BM25
→ 属于相同算法体系
→ 不能默认原始分数和排名逐项一致
```

## 10. Sparse Vector 的迁移意义

手写版中的：

```python
Counter(chunk_words)
```

可以理解为：

```text
词项 → 词频
```

Milvus 将词项表示组织为 Sparse Vector 字段，用于倒排索引和检索。

从理解层看：

```text
手写词频字典
≈ 以词项为坐标的稀疏表示
```

但不能直接断言：

```text
Milvus内部sparse字段
= 手写Counter对象的原样序列化
```

Milvus 的内部编码、分区和索引存储属于数据库实现细节。

## 11. RRF 迁移

### 11.1 手写版

手写版显式：

```text
取Dense与BM25候选并集
→ 查找每个chunk在两路的rank
→ 累加1 / (c + rank)
→ 排序
```

### 11.2 Milvus 版

配置：

```python
ranker_type="rrf"
ranker_params={"k": 60}
```

Milvus 对多路搜索结果执行 RRF。

### 11.3 保持不变的核心

```text
不直接相加Dense原始分数与BM25原始分数
→ 只根据各路排名融合
```

### 11.4 参数名称差异

手写版通常把平滑常数写作：

```text
c = 60
```

Milvus 参数使用：

```python
{"k": 60}
```

它们在当前语境中承担相同角色：

```text
RRF公式平滑常数
```

外层：

```python
k=10
```

才是最终结果数量。

## 12. Reranker 迁移

两版使用同一个模型：

```text
Qwen3-Reranker-0.6B
```

### 12.1 手写版

```python
CrossEncoder(...)
→ model.predict(pairs)
→ np.argsort(scores)
→ Top K
```

所有步骤直接写在自定义 `Reranker` 中。

### 12.2 LangChain 版

```text
HuggingFaceCrossEncoder
→ 负责模型打分

CrossEncoderReranker
→ 负责构造文本对、排序和截取

ContextualCompressionRetriever
→ 负责把Retriever与Reranker串起来
```

算法流程没有增加：

```text
Query + Candidates
→ CrossEncoder Scores
→ Sort
→ Top N
```

增加的是标准接口和包装层。

## 13. Generator 迁移

### 13.1 手写版

```text
messages
→ tokenizer.apply_chat_template()
→ tokenizer()
→ model.generate()
→ 截取新token
→ tokenizer.decode()
→ answer
```

### 13.2 LangChain 版

```text
ChatHuggingFace
→ HuggingFacePipeline
→ transformers.pipeline
→ Model + Tokenizer
```

对应关系：

```text
transformers.pipeline
→ 封装tokenize、generate与decode

HuggingFacePipeline
→ 将Transformers Pipeline适配为LangChain LLM

ChatHuggingFace
→ 将LLM适配为Messages → AIMessage
```

没有增加新的生成算法，只增加接口适配。

## 14. Prompt 迁移

### 14.1 手写版

手动拼接：

```text
系统规则
+ 参考资料
+ 用户问题
→ messages
```

### 14.2 LangChain 版

```python
ChatPromptTemplate.from_messages(
    [
        ("system", "...{context}..."),
        ("human", "{question}\n/no_think"),
    ]
)
```

变量：

```text
context
question
```

由 LCEL 上游提供。

Prompt 的业务规则保持一致：

- 只依据参考资料。
- 不编造。
- 资料不足时明确返回无法确定。

## 15. RAG 编排迁移

### 15.1 手写版

自定义方法显式控制：

```python
def answer_question(question):
    candidates = retrieve(question)
    documents = rerank(question, candidates)
    prompt = build_prompt(question, documents)
    answer = generate(prompt)
    return answer
```

### 15.2 LangChain 版

```python
rag_chain = (
    {
        "context": rerank_retriever
            | format_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | chat_model
    | StrOutputParser()
)
```

执行：

```python
answer = rag_chain.invoke(question)
```

两版的数据流相同：

```text
Question
→ Documents
→ Context
→ Messages
→ Answer
```

区别是控制方式：

```text
手写版
→ Python函数和类显式调用

LangChain版
→ Runnable和LCEL组合
```

## 16. ID 迁移

### 16.1 手写版

当前手写版使用索引中的整数 `chunk_id`。

该 ID 与当前切分和排列有关，重建索引后可能变化。

### 16.2 Milvus 版

当前使用：

```text
auto_id = True
```

Milvus 为每条数据生成整数 `pk`。

Collection 删除并重建后，不能依赖旧 `pk` 继续指向同一个 chunk。

### 16.3 当前共同边界

两版目前都没有实现稳定业务 ID：

```text
source
+ chunk位置
+ 内容版本
→ 稳定标识
```

这是当前学习版有意保留的简化，不影响流程验证，但不适合需要长期引用与增量更新的生产场景。

## 17. 评测迁移

手写版已经评测：

```text
Dense
BM25
RRF
RRF + Reranker
```

指标：

```text
MRR
Recall@K
```

框架版当前尚未建立新的评测集和基线。

不能直接复用手写版 `relevant_chunk_ids`，因为：

- 切分器改变后 chunk 边界可能不同。
- Milvus `pk` 与手写 `chunk_id` 不是同一标识。
- Collection 重建后自动主键可能变化。

框架版后续评测需要：

```text
重新固定知识库版本
→ 固定切分配置
→ 建立可复现的相关文档标注
→ 分别评测Dense、BM25、RRF和Reranker
```

## 18. 可观察性变化

手写版可以直接观察：

- Dense 原始分数。
- BM25 逐词贡献。
- RRF 每路排名。
- Reranker 分数。
- 每个中间数据结构。

框架版默认返回更高层对象：

```text
Retriever
→ List[Document]
```

部分底层细节被框架和 Milvus 隐藏。

出现效果问题时，需要主动拆层验证：

```text
文档加载
→ 切分
→ Dense单路
→ BM25单路
→ RRF
→ Reranker
→ Prompt
→ Generation
```

这也是保留手写版的价值：它提供了算法理解和调试参照。

## 19. 迁移后的收益

### 19.1 数据持久化

```text
Pickle
→ Milvus Collection
```

不再依赖应用进程持有全部数据。

### 19.2 索引能力

```text
NumPy全量计算
→ HNSW

手写BM25候选检索
→ Sparse Inverted Index
```

### 19.3 标准接口

```text
VectorStore
Retriever
Document Compressor
ChatModel
Runnable
```

便于后续接入：

- Agent 工具。
- LangGraph Node。
- API 服务。
- 其他模型或数据源。

### 19.4 组件替换

理论上可以在保持上层接口的情况下替换：

```text
Embedding模型
VectorStore
Retriever
Reranker
ChatModel
```

实际替换时仍需检查字段、参数、模型输入格式和评测结果，不能仅凭接口相同假设效果相同。

## 20. 迁移成本

框架版增加了：

- 包拆分与版本依赖。
- 多层适配器。
- 类名和初始化参数。
- 隐藏的默认行为。
- 调试时的调用链长度。

例如生成模型：

```text
ChatHuggingFace
→ HuggingFacePipeline
→ transformers.pipeline
→ Model + Tokenizer
```

手写版的直接数据流更容易观察；框架版的统一接口更适合组件组合。两者解决的问题不同。

## 21. 两个版本如何保留

不应该用框架版覆盖手写版。

推荐同时保留：

```text
handwritten-rag
→ 展示算法理解、评测与可观察中间过程

langchain-rag
→ 展示Milvus、LangChain和标准组件编排
```

对同一问题，可以分别说明：

```text
算法原理如何实现
→ 看手写版

真实工具如何工程化组合
→ 看框架版
```

## 22. 结论

迁移前后，核心 RAG 流程没有变化：

```text
加载
→ 切分
→ 召回
→ 融合
→ 重排
→ 生成
```

真正发生变化的是：

```text
本地数据结构
→ LangChain Document

Pickle和NumPy
→ Milvus Collection与索引

手写BM25/RRF
→ Milvus原生BM25/RRF

显式Python调用
→ LangChain Runnable与LCEL
```

框架版不替代对算法原理的理解；手写版也不替代数据库索引、标准接口和工程生命周期设计。两个版本共同构成完整的学习与项目展示路径。

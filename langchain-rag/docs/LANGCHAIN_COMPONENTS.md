# LangChain 组件与包装关系

本文不介绍新的 RAG 算法，只解释当前项目中 LangChain 各组件的职责、输入输出与包装关系。

理解这些组件时，只问四个问题：

```text
它是什么接口？
它接收什么？
它返回什么？
它包装了谁？
```

## 1. 先看没有框架的流程

如果完全手写，在线 RAG 可以概括为：

```python
documents = retrieve(question)
documents = rerank(question, documents)
context = format_documents(documents)
messages = build_prompt(question, context)
answer = generate(messages)
```

LangChain 没有改变这条业务流程。它主要做了两件事：

```text
1. 为不同组件定义统一接口
2. 允许组件通过Runnable组合
```

当前框架代码对应：

```text
retrieve
→ VectorStoreRetriever

rerank
→ ContextualCompressionRetriever

build_prompt
→ ChatPromptTemplate

generate
→ ChatHuggingFace

组合
→ LCEL Runnable Chain
```

## 2. 当前项目的组件总表

| 组件 | 所在包 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `Document` | `langchain-core` | 文本、metadata | Document 对象 | 统一文档结构 |
| `RecursiveCharacterTextSplitter` | `langchain-text-splitters` | Documents | Documents | 切分文档 |
| `HuggingFaceEmbeddings` | `langchain-huggingface` | 文本 | Dense vectors | 适配 Embeddings 接口 |
| `Milvus` | `langchain-milvus` | Documents / Query | IDs / Documents | VectorStore 适配器 |
| `BM25BuiltInFunction` | `langchain-milvus` | 字段配置 | Function 描述 | 描述 Milvus BM25 路线 |
| `VectorStoreRetriever` | `langchain-core` 内部接口实现 | Query | Documents | 把 VectorStore 包装成 Retriever |
| `HuggingFaceCrossEncoder` | `langchain-community` | 文本对 | Scores | 调用本地 CrossEncoder |
| `CrossEncoderReranker` | `langchain-classic` | Query、Documents | Documents | 打分后排序并截取 |
| `ContextualCompressionRetriever` | `langchain-classic` | Query | Documents | 串联检索与重排 |
| `HuggingFacePipeline` | `langchain-huggingface` | 文本 | 文本 | 将 Transformers Pipeline 适配为 LLM |
| `ChatHuggingFace` | `langchain-huggingface` | Messages | `AIMessage` | 将 LLM 适配为 ChatModel |
| `ChatPromptTemplate` | `langchain-core` | 变量字典 | Messages | 构造聊天 Prompt |
| `RunnablePassthrough` | `langchain-core` | 任意输入 | 原输入 | 原样传递数据 |
| `StrOutputParser` | `langchain-core` | `AIMessage` | 字符串 | 提取最终文本 |

包路径不需要死记。更重要的是知道每个对象处于哪一层。

## 3. `Document`

```python
from langchain_core.documents import Document
```

标准结构：

```python
Document(
    page_content="正文",
    metadata={
        "source": "文件路径",
    },
)
```

固定字段：

```text
page_content
→ 正文

metadata
→ 附加信息
```

`page_content` 不是项目自定义字段；它是 LangChain `Document` 的标准属性。

当前数据流：

```text
原始文件
→ Document
→ 切分后的Document
→ Milvus
→ Retriever返回Document
→ Prompt Context
```

## 4. `Embeddings`

LangChain 的 Embeddings 是一个接口，不是具体模型。

核心方法：

```python
embed_documents(texts)
embed_query(text)
```

项目使用：

```python
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    ...
)
```

包装关系：

```text
LangChain Embeddings接口
└── HuggingFaceEmbeddings
    └── Sentence Transformers
        └── Qwen3-Embedding-0.6B
```

输入输出：

```text
embed_documents
List[str] → List[List[float]]

embed_query
str → List[float]
```

VectorStore 不需要知道 Qwen3 模型的底层推理细节，只依赖 Embeddings 接口。

## 5. `VectorStore`

VectorStore 表示：

```text
能够保存和检索向量数据的统一接口
```

常见方法：

```python
vector_store.add_documents(documents)
vector_store.similarity_search(query)
vector_store.as_retriever()
```

当前实例：

```python
vector_store = Milvus(...)
```

这里：

```text
Milvus
→ LangChain类

vector_store
→ 项目变量名

Milvus Standalone
→ 真正运行的数据库服务
```

对象关系：

```text
Python VectorStore对象
→ 通过网络连接Milvus服务
→ 操作rag_demo.rag_chunks
```

VectorStore 不是数据库本身，也不会把全部向量加载到 Python 进程。

## 6. `as_retriever()`

```python
retriever = vector_store.as_retriever(...)
```

它不会创建新的 Collection，也不会复制数据。

它只是把 VectorStore 的搜索能力包装成 Retriever 接口：

```text
VectorStore接口
Query → similarity_search() → Documents

Retriever接口
Query → invoke() → Documents
```

近似关系：

```python
retriever.invoke(query)
```

内部调用：

```python
vector_store.similarity_search(
    query,
    **search_kwargs,
)
```

为什么需要 Retriever？

因为 LangChain 后续组件只需要依赖统一接口：

```text
输入Query
输出List[Document]
```

底层可以替换为：

```text
Milvus
Elasticsearch
网页搜索
数据库查询
其他知识源
```

## 7. Retriever

Retriever 的职责只有：

```text
接收一个非结构化Query
→ 返回相关Documents
```

标准调用：

```python
documents = retriever.invoke(query)
```

当前基础 Retriever 内部执行：

```text
Dense
+ BM25
+ RRF
→ Top 10 Documents
```

Retriever 本身不负责：

- 生成最终答案。
- 构造 Prompt。
- 调用生成模型。
- 保存聊天历史。

## 8. `HuggingFaceCrossEncoder`

```python
from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)
```

它是本地 CrossEncoder 模型的 LangChain 适配器。

包装关系：

```text
LangChain CrossEncoder接口
└── HuggingFaceCrossEncoder
    └── sentence_transformers.CrossEncoder
        └── Qwen3-Reranker-0.6B
```

输入：

```python
[
    (query, document_text_1),
    (query, document_text_2),
]
```

输出：

```python
[
    score_1,
    score_2,
]
```

它只负责打分，不负责：

- 从 Milvus 召回 Document。
- 将 Document 转为文本对。
- 按分数排序。
- 截取 Top N。

当前 `langchain-huggingface` 尚未提供这个本地 CrossEncoder 适配类，因此官方方案仍从 `langchain-community` 导入。

## 9. `CrossEncoderReranker`

```python
reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=5,
)
```

它负责：

```text
接收Query和Documents
→ 读取document.page_content
→ 构造(query, document)文本对
→ 调用CrossEncoder获得分数
→ 按分数排序
→ 保留Top N
```

输入输出：

```text
Query + List[Document]
→ List[Document]
```

它被归类为 Document Compressor，因为：

```text
10个Documents
→ 5个Documents
```

这里压缩的是结果集合数量，不是压缩单个 Document 的字符串长度。

## 10. `ContextualCompressionRetriever`

```python
rerank_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=reranker,
)
```

名称拆解：

```text
Contextual
→ 根据当前Query处理

Compression
→ 减少或筛选检索结果

Retriever
→ 对外仍然输入Query、输出Documents
```

这里的 `Contextual` 不是最终提供给 LLM 的 RAG Context。它表示 Compressor 根据当前 Query 判断 Document 相关性。

对象关系：

```text
ContextualCompressionRetriever
├── base_retriever
│   └── Milvus混合Retriever
└── base_compressor
    └── CrossEncoderReranker
        └── HuggingFaceCrossEncoder
```

近似实现：

```python
def invoke(query):
    documents = base_retriever.invoke(query)

    documents = base_compressor.compress_documents(
        documents=documents,
        query=query,
    )

    return documents
```

执行顺序：

```text
Query
→ Retriever
→ 10个Documents
→ CrossEncoderReranker
→ HuggingFaceCrossEncoder打分
→ Reranker排序并保留5个
→ 5个Documents
```

它的意义是把：

```text
检索
→ 重排
```

重新包装成一个标准 Retriever。

## 11. LangChain 的 `LLM`

在日常语言中，LLM 表示大语言模型。

在 LangChain 接口中，`LLM` 更具体地表示：

```text
文本输入
→ 文本输出
```

概念接口：

```python
output = llm.invoke(prompt_text)
```

输入：

```text
str
```

输出：

```text
str
```

具体底层可以是：

- 本地 Transformers 模型。
- 远程模型接口。
- 推理服务。
- 其他文本生成后端。

## 12. Transformers `pipeline`

```python
text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    ...
)
```

这是 Hugging Face Transformers 的推理封装，不是 LangChain Pipeline。

它替代了手写流程中的：

```text
Tokenizer编码
→ 输入移动到Device
→ model.generate()
→ 截取和解码输出
```

它的原始返回格式通常是 Transformers 定义的结果结构，而不是 LangChain LLM 所要求的统一文本接口。

## 13. `HuggingFacePipeline`

```python
llm = HuggingFacePipeline(
    pipeline=text_generation_pipeline,
)
```

这个名字的准确理解是：

```text
LangChain对Hugging Face Pipeline的LLM适配器
```

它不是：

- 第二个生成模型。
- 另一个推理引擎。
- 再执行一次模型推理。
- 所谓“LangChain Pipeline”。

包装关系：

```text
LangChain LLM接口
└── HuggingFacePipeline
    └── transformers.pipeline
```

接口变化：

```text
Transformers调用与返回格式
→ LangChain文本输入/文本输出接口
```

包装完成后，该对象可以进入 LangChain Runnable Chain。

## 14. `ChatHuggingFace`

```python
chat_model = ChatHuggingFace(
    llm=llm,
    tokenizer=tokenizer,
)
```

它将普通 LLM 接口进一步包装成 ChatModel 接口。

接口变化：

```text
LLM
str → str

ChatModel
Messages → AIMessage
```

数据流：

```text
SystemMessage + HumanMessage
→ Tokenizer Chat Template
→ Prompt字符串
→ HuggingFacePipeline
→ transformers.pipeline
→ 生成文本
→ AIMessage
```

对象关系：

```text
ChatHuggingFace
└── HuggingFacePipeline
    └── transformers.pipeline
        ├── AutoModelForCausalLM
        └── AutoTokenizer
```

之所以不能直接把原始 Transformers Pipeline 传给当前 `ChatHuggingFace`，是因为该包装器要求底层对象先符合 LangChain LLM 接口。

## 15. `ChatPromptTemplate`

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "...{context}..."),
        ("human", "{question}\n/no_think"),
    ]
)
```

输入：

```python
{
    "context": "参考资料",
    "question": "用户问题",
}
```

输出：

```text
ChatPromptValue / Messages
```

概念结果：

```python
[
    SystemMessage(
        content="系统指令和参考资料"
    ),
    HumanMessage(
        content="用户问题"
    ),
]
```

它只负责格式化 Prompt，不执行模型推理。

## 16. `Runnable`

Runnable 是 LangChain 用来统一“可执行组件”的接口。

常见调用：

```python
result = runnable.invoke(input_value)
```

不同 Runnable 的输入输出可以不同：

```text
Retriever
str → List[Document]

Prompt
dict → Messages

ChatModel
Messages → AIMessage

OutputParser
AIMessage → str
```

只要前一个组件的输出能够作为后一个组件的输入，就可以把它们组成 Chain。

## 17. `RunnablePassthrough`

```python
"question": RunnablePassthrough()
```

它的行为近似：

```python
def passthrough(value):
    return value
```

输入：

```text
第一周主要学习什么
```

输出仍然是：

```text
第一周主要学习什么
```

当前项目用它保留原始 Question。

## 18. `StrOutputParser`

```python
| StrOutputParser()
```

ChatModel 返回：

```python
AIMessage(
    content="最终回答"
)
```

经过 `StrOutputParser`：

```python
"最终回答"
```

类型变化：

```text
AIMessage → str
```

因此：

```python
answer = rag_chain.invoke(question)
```

得到的是字符串，不需要再访问：

```python
answer.content
```

## 19. LCEL

LCEL 是：

```text
LangChain Expression Language
```

它不是独立于 Python 的新语言，而是基于：

- Python 对象。
- Runnable 接口。
- 运算符重载。
- 自动类型转换。

实现的嵌入式表达方式。

### 19.1 `|` 运算符

```python
A | B
```

表示：

```text
A的输出
→ B的输入
```

对于 LangChain Runnable，`|` 的核心思想近似：

```python
class Runnable:
    def __or__(self, other):
        other = coerce_to_runnable(other)

        return RunnableSequence(
            self,
            other,
        )
```

所以：

```python
rerank_retriever | format_documents
```

近似于：

```python
RunnableSequence(
    rerank_retriever,
    RunnableLambda(format_documents),
)
```

普通函数 `format_documents` 会被自动包装为 Runnable。

### 19.2 字典映射

```python
{
    "context": rerank_retriever | format_documents,
    "question": RunnablePassthrough(),
}
```

在 LCEL Chain 中近似转换为：

```python
RunnableParallel(
    context=rerank_retriever
        | RunnableLambda(format_documents),
    question=RunnablePassthrough(),
)
```

同一个输入会交给两条路线：

```text
Question
├→ Retriever → Reranker → format_documents → context
└→ RunnablePassthrough                    → question
```

两条路线的结果组成字典：

```python
{
    "context": "格式化后的参考资料",
    "question": "原始问题",
}
```

## 20. 完整 Chain 的类型变化

当前代码：

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

逐步执行：

```text
输入
str

经过字典映射
{"context": str, "question": str}

经过ChatPromptTemplate
ChatPromptValue / Messages

经过ChatHuggingFace
AIMessage

经过StrOutputParser
str
```

最终：

```python
answer = rag_chain.invoke(question)
```

接口：

```text
str → str
```

## 21. 对象创建与真正执行

以下代码只创建或组合对象，不会立即查询：

```python
vector_store = get_vector_store(...)
retriever = create_retriever(...)
rerank_retriever = create_rerank_retriever(...)
rag_chain = create_rag_chain(...)
```

其中模型初始化可能加载参数、占用内存，但不会因为 Chain 被创建就回答问题。

真正执行发生在：

```python
answer = rag_chain.invoke(question)
```

内部依次触发：

```text
Retriever
→ Reranker
→ Prompt
→ ChatModel
→ OutputParser
```

## 22. 为什么包分得这么散

当前项目涉及：

```text
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-milvus
langchain-community
langchain-classic
```

它们的定位：

```text
langchain-core
→ Document、Prompt、Runnable等基础接口

langchain-text-splitters
→ 文本切分器

langchain-huggingface
→ Hugging Face Embedding、LLM与ChatModel适配

langchain-milvus
→ Milvus VectorStore与BM25 Function适配

langchain-community
→ 当前本地HuggingFaceCrossEncoder适配

langchain-classic
→ 1.0后移出的经典Retriever与Compressor组件
```

包拆分有利于减少核心包与第三方集成的耦合，但代价是 import 路径和类名较多。

## 23. 需要掌握什么

不需要脱离文档默写全部 import 路径。

需要掌握：

```text
Document
→ 文本与metadata

Embeddings
→ 文本转向量

VectorStore
→ 存储与搜索

Retriever
→ Query转Documents

Reranker
→ Query + Documents重新排序

Prompt
→ 变量转Messages

ChatModel
→ Messages转AIMessage

OutputParser
→ AIMessage转最终输出

Runnable
→ 统一调用和组合接口
```

实际开发时可以查：

- 具体包路径。
- 初始化参数名称。
- 不常用 Search Params。
- 不同版本的迁移方式。

判断是否理解框架的标准不是能否默写类名，而是能否说明：

```text
每个对象的输入是什么
输出是什么
包装了谁
出现问题时属于哪一层
```

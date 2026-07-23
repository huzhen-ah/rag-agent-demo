# 手撕混合检索设计

## 署名与贡献说明

- 本文对应的项目代码由项目作者独立手写完成。
- 本文档由 OpenAI Codex 根据现有代码与实验结果起草。
- 文档内容由项目作者逐项审核、修改并最终确认。

## 1. 目标

本文说明手撕 RAG 中检索链路的设计与实现：

```text
Query
  ├── Dense Retrieval ─┐
  │                    ├→ RRF → Reranker → Final Chunks
  └── BM25 Retrieval ──┘
```

四个组件解决的问题不同：

| 组件 | 主要作用 |
|---|---|
| Dense Retrieval | 根据向量相似度召回语义相关 chunks |
| BM25 Retrieval | 根据关键词、术语和编号召回 chunks |
| RRF | 融合 Dense 与 BM25 的两份排名 |
| Reranker | 对融合候选进行更精细的相关性判断和重排 |

所有组件统一返回：

```python
{
    "chunk": chunk,
    "score": score,
    "rank": rank,
}
```

其中：

- `chunk` 是检索到的文本块。
- `score` 是当前阶段的内部得分。
- `rank` 是当前阶段从 1 开始的排名。

不同阶段的 score 含义不同，不能直接横向比较：

```text
Dense score      = 向量相似度
BM25 score       = BM25 关键词相关性分数
RRF score        = 两路倒数排名贡献之和
Reranker score   = CrossEncoder 相关性分数
```

## 2. 文档与 Chunk

数据结构定义在 `load.py`。

### Document

```python
@dataclass
class Document:
    content: str
    source: str
    page: int | None = None
```

字段含义：

- `content`：当前文档文本。
- `source`：来源文件路径。
- `page`：PDF 页码；TXT 和 Markdown 为 `None`。

当前 Loader 的实际行为是：

```text
一个 TXT/Markdown 文件 → 一个 Document
一个 PDF 非空页面       → 一个 Document
```

因此当前 PDF 的 Document 表示一页提取文本，而不是整份 PDF。

### Chunk

```python
@dataclass
class Chunk:
    chunk_id: int
    content: str
    source: str
    start: int
    end: int
    page: int | None = None
```

字段含义：

- `chunk_id`：当前索引中的整数编号。
- `content`：chunk 文本。
- `source`：来源文件。
- `start`、`end`：chunk 在当前 Document 文本中的字符范围；对于 PDF，它们是当前页提取文本内的范围。
- `page`：来源 PDF 页码。

### 切块策略

当前采用固定字符长度与 overlap：

```python
chunk_size = 500
chunk_overlap = 100
```

步长为：

```text
step = chunk_size - chunk_overlap
     = 400
```

相邻 chunks 共享 100 个字符，用于减少切块边界造成的上下文丢失。

## 3. Dense Retrieval

实现文件：

```text
embedding.py
dense_retrieval.py
```

### 3.1 Chunk 向量

构建索引时，Qwen3-Embedding-0.6B 将每个 chunk 转换成 dense embedding：

```python
chunk_embeddings = model.encode(
    chunk_contents,
    normalize_embeddings=True,
)
```

最终得到二维矩阵：

```text
chunk_embeddings.shape
=
(chunk数量, embedding维度)
```

### 3.2 Query 向量

查询时使用同一个模型生成 query embedding：

```python
query_embedding = embedder.embed_query(query)
```

Qwen3 Embedding 的 query 侧使用：

```python
prompt_name="query"
```

### 3.3 相似度

Dense Retrieval 使用：

```python
scores = chunk_embeddings @ query_embedding
```

由于 chunk 与 query 向量均已归一化，因此：

```text
向量点积 = 余弦相似度
```

### 3.4 Top-K

代码先通过 NumPy 找到分数最高的 K 个索引：

```python
top_k_index = np.argpartition(
    -scores,
    k - 1,
)[:k]
```

`argpartition` 只负责选出 Top-K 集合，不保证集合内部有序，因此还要按照 score 降序排序。

Dense Retrieval 擅长处理不同措辞表达的相同语义，但可能漏掉编号、缩写和必须精确匹配的专有词。

## 4. BM25 Retrieval

实现文件：

```text
bm25_retrieval.py
```

### 4.1 分词

中文文本使用 Jieba：

```python
words = [
    word.strip()
    for word in jieba.lcut(chunk.content)
    if word.strip()
]
```

chunk 分词结果不去重，因为 TF 和 chunk 长度都需要真实词频。

### 4.2 TF

设：

```text
t = 一个查询词
d = 当前 chunk
```

TF 是词 `t` 在 chunk `d` 中出现的次数：

```text
TF(t, d) = count(t, d)
```

代码为每个 chunk 保存一个词频字典：

```python
{
    "LoRA": 2,
    "参数": 3,
}
```

### 4.3 DF

DF 是包含词 `t` 的 chunk 数：

```text
DF(t) = 包含 t 的 chunk 数量
```

统计 DF 时，每个 chunk 对同一个词最多贡献一次。代码遍历的是当前 chunk 的 TF 字典键，因此不会因为一个词在同一 chunk 中出现多次而重复增加 DF。

### 4.4 BM25 IDF

设：

```text
N = chunk 总数
```

使用的 IDF 为：

```text
IDF(t)
=
ln[1 + (N - DF(t) + 0.5) / (DF(t) + 0.5)]
```

词在越少的 chunks 中出现，IDF 越大，对检索结果的区分能力越强。

### 4.5 BM25 公式

设：

```text
|d|    = 当前 chunk 的 token 数
avgdl  = 全部 chunks 的平均 token 数
k1     = TF 饱和参数，默认 1.5
b      = 长度归一化参数，默认 0.75
```

单个查询词的得分：

```text
IDF(t)
×
TF(t,d) × (k1 + 1)
/
[TF(t,d) + k1 × (1 - b + b × |d| / avgdl)]
```

完整 query 与当前 chunk 的 BM25 分数：

```text
BM25(query, chunk)
=
query 中所有去重词的单词得分之和
```

### 4.6 倒排候选

代码构建：

```python
word2chunks[word] = {chunk_index_1, chunk_index_2, ...}
```

查询时，先对 query 分词并去重，再从 `word2chunks` 找到包含任意查询词的 chunks：

```text
BM25候选集合
=
每个查询词对应chunk集合的并集
```

只对候选集合计算 BM25 分数，不遍历与 query 完全没有词语交集的 chunks。

BM25 擅长处理：

- 专有名词。
- 英文缩写。
- 数字与编号。
- 与原文措辞接近的问题。

它不擅长处理没有关键词重合的同义表达。

## 5. RRF

实现位置：

```text
rag.py -> RAG.rrf()
```

### 5.1 为什么不能直接相加

Dense 和 BM25 的 score 不在同一个量纲：

```text
Dense：0.82
BM25：7.31
```

数值大小不同不代表 BM25 比 Dense 更有信心，因此不能直接：

```text
0.82 + 7.31
```

RRF 忽略原始 score，只使用排名。

### 5.2 候选集合

RRF 使用两路 chunks 的并集：

```text
RRF候选
=
Dense Top-K chunks
∪
BM25 Top-K chunks
```

例如：

```text
Dense：{1, 3, 5}
BM25： {3, 5, 8}

并集： {1, 3, 5, 8}
```

只出现在一路的 chunk 仍然保留，不要求同时被两路召回。

### 5.3 RRF 分数

```text
RRF(chunk)
=
Σ 1 / (c + rank)
```

当前：

```text
c = 60
```

对于同时出现在 Dense 和 BM25 的 chunk：

```text
RRF(chunk)
=
1 / (60 + Dense排名)
+
1 / (60 + BM25排名)
```

只出现在一路时，只计算该路贡献。

计算全部候选分数后，按照 RRF score 降序排序并截取 Top-K。

### 5.4 单路为空

如果 Dense 或 BM25 一方没有结果，使用单路 RRF 不会改变原排序。因此代码直接返回非空一方的 Top-K。

## 6. Reranker

实现文件：

```text
reranker.py
```

RRF 只融合排名，没有直接判断 query 与 chunk 的完整文本关系。Reranker 使用 CrossEncoder 对每个文本对进行联合建模：

```text
(query, chunk_1) → score_1
(query, chunk_2) → score_2
(query, chunk_3) → score_3
```

初始化：

```python
self.model = CrossEncoder(
    model_path,
    device=str(device),
)
```

构建 pairs：

```python
pairs = [
    (query, result["chunk"].content)
    for result in rrf_results
]
```

批量打分：

```python
scores = self.model.predict(pairs)
```

按照分数降序排列：

```python
indexes = np.argsort(scores)[::-1]
```

Reranker 只能重新排列 RRF 提供的候选，不能找回候选池之外的 chunk。因此进入 Reranker 的候选数量通常大于最终交给生成模型的数量。

## 7. 最终参数

端到端 Demo 当前使用：

```text
Dense Top-15
BM25 Top-15
→ RRF Top-10，c=60
→ Reranker Top-5
→ Generator
```

参数关系：

```text
Dense候选数 = 15
BM25候选数  = 15
RRF候选数   = 10
Reranker输出 = 5
```

当前配置满足：

```text
Dense/BM25 候选数 > RRF 候选数 > Reranker 输出数
```

扩大前级候选池可以提高召回上限，但也会增加后续融合与 Reranker 的计算量。

## 8. 完整调用链

`demo.py` 初始化所有组件：

```python
dense_retrieval = DenseRetrieval(
    embedder,
    chunks,
    chunk_embeddings,
)

bm25_retrieval = BM25Retrieval(chunks)
reranker = Reranker(reranker_model_path, device)

rag = RAG(
    dense_retrieval,
    bm25_retrieval,
    reranker,
    generator,
)
```

`RAG.answer_question()` 执行：

```text
query
→ Dense Retrieval
→ BM25 Retrieval
→ RRF
→ Reranker
→ 拼接参考资料
→ Generator
→ answer
```

`RAG` 本身不重复保存 chunks。Dense 与 BM25 检索器持有相同的 chunks，并在返回结果中携带 Chunk 对象；RRF、Reranker 和 Generator 依次消费上一阶段的输出。

# RAG & Agent Demo

这是一个用于学习和作品集展示的 RAG 与 Agent 项目。目前包含两套可以对照阅读的 RAG 实现：

- `handwritten-rag`：不使用 RAG 编排框架，手写 BM25、RRF 和评测流程。
- `langchain-rag`：使用 LangChain 与 Milvus Standalone 重构同一条 RAG 链路。

后续将在本仓库继续实现手写 Agent 和 LangGraph Agent。

## 署名

- 本项目代码由项目作者在 OpenAI Codex 协助下完成。
- 项目文档由 OpenAI Codex 根据现有代码和实验结果起草。
- 文档内容由项目作者审核、修改并最终确认。

## 项目结构

```text
rag-agent-demo/
├── handwritten-rag/
│   ├── documents/
│   ├── dataset/
│   ├── docs/
│   ├── bm25_retrieval.py
│   ├── dense_retrieval.py
│   ├── reranker.py
│   ├── rag.py
│   └── demo.py
├── langchain-rag/
│   ├── documents/
│   ├── docs/
│   ├── ingest.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── rag.py
│   └── demo.py
└── README.md
```

## 整体流程

两套实现遵循同一个核心流程：

```text
文档加载与切分
→ Dense检索 + BM25检索
→ RRF融合
→ CrossEncoder重排
→ 本地LLM生成答案
```

## Handwritten RAG

手写版用于理解算法和数据流：

- NumPy 保存并检索 Dense Embedding。
- Jieba 分词。
- 手写 BM25。
- 手写 RRF。
- Qwen3-Reranker-0.6B 重排。
- Qwen3-1.7B 生成答案。
- 使用 MRR 和 Recall@K 分别评测各检索阶段。

当前小型评测集的最终结果：

| MRR | Recall@1 | Recall@3 | Recall@5 |
|---:|---:|---:|---:|
| 1.00 | 0.90 | 1.00 | 1.00 |

评测集只有 10 个问题，且问题与原文措辞接近。该结果只用于验证流程和建立回归基线，不代表真实业务效果。

运行：

```bash
cd handwritten-rag
python demo.py
```

详细说明见 [handwritten-rag/README.md](handwritten-rag/README.md)。

## LangChain + Milvus RAG

LangChain 版用于学习框架组件和向量数据库：

- 自行解析 TXT、Markdown 和 PDF。
- 使用 LangChain `Document` 和文本切分器。
- 使用 Milvus 保存原文、Dense 与 Sparse 数据。
- 使用 Qwen3-Embedding-0.6B 生成 Dense 向量。
- 使用 Milvus 原生 BM25 Function 和 Sparse 索引。
- 使用 RRF 执行混合检索。
- 使用 LangChain CrossEncoder 组件重排。
- 使用 LCEL 编排检索、Prompt、模型和输出解析。

运行前需要启动 Milvus Standalone，然后执行：

```bash
cd langchain-rag
python demo.py
```

详细说明见 [langchain-rag/README.md](langchain-rag/README.md)。

## 两个版本的对应关系

| 环节 | 手写版 | LangChain + Milvus 版 |
|---|---|---|
| 文档对象 | 自定义数据结构 | LangChain `Document` |
| Dense 存储 | NumPy / Pickle | Milvus |
| Dense 检索 | 矩阵乘法 | Milvus HNSW |
| BM25 | 手写实现 | Milvus 原生 BM25 |
| RRF | 手写实现 | Milvus 混合检索 |
| Reranker | 直接调用模型 | LangChain 组件包装 |
| 编排 | 普通 Python | LCEL |

## 本地模型

两个版本均使用：

```text
Qwen3-Embedding-0.6B
Qwen3-Reranker-0.6B
Qwen3-1.7B
```

模型权重不提交到 Git 仓库，需要分别放入各版本的 `models/` 目录。

## 当前边界

这是以理解和跑通流程为目标的本地 Demo，目前没有：

- API 服务和前端。
- 并发与流式输出。
- 增量索引。
- 权限控制。
- 生产级配置和监控。
- LangChain 版正式检索评测。

## 后续计划

```text
手写最小Agent
→ LangGraph Agent
→ FastAPI与演示页面
→ 测试、评测和项目说明完善
```

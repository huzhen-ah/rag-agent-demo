# RAG & Agent Demo

这是一个面向学习、实验和作品集展示的 RAG 与 Agent 项目。

项目从不依赖编排框架的手写实现开始，逐步迁移到 Milvus、LangChain 和 LangGraph，用于对照理解检索、重排、生成、工具调用与状态编排的底层流程和框架实现。

## 署名与贡献说明

- 本项目代码由项目作者独立手写完成。
- 项目文档由 OpenAI Codex 根据现有代码与实验结果起草。
- 文档内容由项目作者逐项审核、修改并最终确认。

## 项目路线

```text
手写混合检索 RAG
→ Milvus + LangChain RAG
→ 手写最小 Agent
→ LangGraph Agent
→ FastAPI / Demo / Docker
```

## 当前结构

```text
rag-agent-demo/
├── handwritten-rag/
│   ├── documents/
│   ├── docs/
│   ├── dataset/
│   ├── dense_retrieval.py
│   ├── bm25_retrieval.py
│   ├── reranker.py
│   ├── rag.py
│   ├── demo.py
│   └── README.md
├── .gitignore
└── README.md
```

后续将逐步加入：

```text
langchain-rag/
handwritten-agent/
langgraph-agent/
api/
demo/
```

## Handwritten RAG

当前已完成的手写 RAG 链路：

```text
Document Loading
→ Fixed-size Chunking
→ Dense Retrieval
→ BM25 Retrieval
→ RRF
→ Qwen3 Reranker
→ Qwen3 Generator
```

最终检索参数：

```text
Dense Top-15
BM25 Top-15
→ RRF Top-10，c=60
→ Reranker Top-5
```

最终检索评测结果：

| 配置 | MRR@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| 15 → 15 → 10 → 5 | 1.0000 | 0.90 | 1.00 | 1.00 |

评测集只有 10 个问题，且问题与原文措辞接近。该结果用于验证流程与建立回归基线，不代表真实业务准确率。

详细说明：

- [Handwritten RAG README](handwritten-rag/README.md)
- [检索设计](handwritten-rag/docs/RETRIEVAL_DESIGN.md)
- [检索评测](handwritten-rag/docs/EVALUATION.md)

## 本地模型

模型文件不提交到 Git 仓库。运行 Handwritten RAG 前，需要准备：

```text
handwritten-rag/models/
├── Qwen3-Embedding-0.6B
├── Qwen3-Reranker-0.6B
└── Qwen3-1.7B
```

下载示例：

```bash
cd handwritten-rag

hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir models/Qwen3-Embedding-0.6B

hf download Qwen/Qwen3-Reranker-0.6B \
  --local-dir models/Qwen3-Reranker-0.6B

hf download Qwen/Qwen3-1.7B \
  --local-dir models/Qwen3-1.7B
```

## 运行 Handwritten RAG

进入目录：

```bash
cd handwritten-rag
```

运行端到端 Demo：

```bash
python demo.py
```

运行检索评测：

```bash
python evaluate_retrieval_dense.py
python evaluate_retrieval_bm25.py
python evaluate_retrieval_rrf.py
python evaluate_retrieval_reranker.py
```

更详细的环境、结构和限制见 [handwritten-rag/README.md](handwritten-rag/README.md)。


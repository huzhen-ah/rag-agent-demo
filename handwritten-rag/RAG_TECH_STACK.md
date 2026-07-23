# RAG 求职技术栈（锁定版）

> 决策日期：2026-07-21  
> 目标：面向国内大模型应用、RAG、Agent 开发岗位，完成一个能讲原理、能评测、能部署的项目。  
> 规则：以下“主线”不再替换。以后出现新工具，只能作为补充调研，不得中途把项目切换到另一套框架或向量数据库。

## 一、最终主线

| 环节 | 锁定技术 | 用途 |
|---|---|---|
| 语言与模型基础 | Python 3.11、PyTorch、Transformers、Sentence Transformers | 模型加载、embedding、生成与工程实现 |
| 文档解析 | PyMuPDF；复杂 PDF 补充 MinerU；扫描件补充 PaddleOCR | PDF、版面文档和 OCR 文档解析 |
| 文本切块 | 先手写；随后使用 LangChain Text Splitters | 掌握 chunk、overlap、metadata、跨页处理 |
| Embedding | Qwen3-Embedding-0.6B | 生成 dense embedding；当前本机已经具备 |
| 关键词检索 | BM25；本地学习阶段使用 `bm25s` | 精确词、专有名词、编号和关键词召回 |
| 向量数据库 | **Milvus Standalone + PyMilvus** | 项目唯一的专业向量数据库 |
| Dense 索引 | Milvus HNSW，归一化向量使用 IP | 近似向量搜索与 metadata filter |
| 混合检索 | Milvus 原生 BM25 + dense search + RRF | 同时利用关键词匹配和语义匹配 |
| Reranker | Qwen3-Reranker-0.6B | 对混合召回候选做 cross-encoder 精排 |
| 生成模型 | Qwen3 系列；统一走 OpenAI-compatible 接口 | 根据检索上下文生成带依据的答案 |
| 本地模型服务 | vLLM（有 CUDA GPU 时）；当前 Mac 学习阶段继续 Transformers | 批处理、并发推理和 OpenAI-compatible 服务 |
| RAG 编排 | **先手撕，再用 LangChain 重写** | 既能解释底层，也满足招聘关键词 |
| Agent 编排 | LangGraph | 下一阶段实现有状态 Agent 工作流 |
| 检索评测 | Recall@K、Precision@K、MRR、nDCG | 用标注问题验证召回和排序，不凭感觉调参 |
| 生成评测 | Ragas + 人工测试集 | 评估 faithfulness、answer relevance、context relevance |
| API 服务 | FastAPI + Pydantic | 对外提供 ingestion、query、health 等接口 |
| 演示界面 | Streamlit | 构建可展示、可操作的作品集界面 |
| 业务数据 | PostgreSQL | 用户、任务、配置、评测记录等结构化数据 |
| 缓存与状态 | Redis | 查询缓存、会话状态、限流或异步任务状态 |
| 可观测性 | Python logging、LangSmith；服务指标用 Prometheus + Grafana | 检索链路追踪、延迟、错误和质量分析 |
| 测试 | pytest | 单元测试、检索回归测试和 API 测试 |
| 部署 | Linux、Git、Docker Compose；Kubernetes 作为加分项 | 形成可复现、可部署的完整项目 |

## 二、固定检索链路

本项目最终检索流程固定为：

1. 文档解析、清洗并保留 `doc_id`、页码、标题、来源等 metadata。
2. 文本切块，生成稳定的 `chunk_id`。
3. Qwen3-Embedding 生成归一化 dense embedding。
4. Milvus 同时保存原文、metadata、dense vector 和 BM25 sparse vector。
5. 查询分别进行 dense search 与 BM25 search。
6. 使用 RRF 合并两路召回结果。
7. 使用 Qwen3-Reranker 对候选精排。
8. 选择最终上下文，组装 prompt 并调用生成模型。
9. 返回答案、引用片段、来源和检索分数。
10. 使用固定测试集评测检索与答案质量。

## 三、明确不再摇摆的选型

- **向量数据库锁定 Milvus**。不把 FAISS、Chroma、Qdrant、Pinecone、Elasticsearch/OpenSearch 换成项目主库。
- FAISS/NumPy 只用于理解本地精确搜索或 ANN 原理，不作为最终项目存储层。
- Elasticsearch/OpenSearch 作为需要知道的企业搜索技术，不在当前项目中与 Milvus 重复建设。
- RAG 框架锁定 LangChain，不再并行学习 LlamaIndex、Haystack、Dify 等同类工具。
- Agent 框架锁定 LangGraph，不同时铺开 AutoGen、CrewAI 等框架。
- Embedding 和 reranker 锁定 Qwen3 同系列；模型尺寸可根据硬件升级，这不算更换技术路线。
- 服务接口统一采用 OpenAI-compatible 协议，使本地 Qwen、vLLM 或外部模型可以替换而不改业务架构。

## 四、为什么这样选

本次不是按单一厂商宣传选型，而是抽查近期国内 RAG/Agent 招聘信息后取共同交集。岗位反复要求的能力包括：

- Python、FastAPI 与基本后端工程能力。
- LangChain/LangGraph 或同类编排框架。
- Milvus、FAISS、Elasticsearch 等检索组件，其中 Milvus 在本次样本中被频繁直接点名。
- 文档解析、切块、embedding、向量数据库、混合召回、rerank 和效果评测的全链路能力。
- vLLM/TGI 等推理服务，Docker/Linux，以及 Redis、SQL、日志监控等生产工程能力。

Milvus 不是只做 dense vector。当前官方版本原生支持 BM25 full-text search、dense search、hybrid search 和 RRF，因此它可以独立承载本项目的关键词与向量混合检索，避免同时维护 Elasticsearch 与 Milvus。

## 五、固定学习顺序

1. 完成当前 NumPy dense mini-RAG。
2. 建立检索测试集，先实现 Recall@K、MRR 等基础评测。
3. 手写 BM25 + dense 两路召回和 RRF，验证混合检索收益。
4. 加入 Qwen3-Reranker，比较 rerank 前后的指标。
5. 将 pickle/NumPy 索引迁移到 Milvus Standalone。
6. 用 LangChain 重写同一条 RAG 链路，并与手写版对照。
7. 加入 Ragas、引用返回、失败处理和链路日志。
8. 使用 FastAPI 封装服务，Streamlit 制作演示页面。
9. 使用 Docker Compose 组织 Milvus、API、PostgreSQL、Redis 和前端。
10. 最后再进入 LangGraph Agent，并复用这套 RAG 作为 Agent 的知识检索工具。

## 六、求职交付标准

项目完成时必须能够展示和解释：

- 为什么纯 dense retrieval 会漏掉关键词、编号和专有名词。
- BM25、向量检索、HNSW、RRF、cross-encoder reranker 各自解决什么问题。
- 如何设计 chunk 与 metadata，如何进行增量更新、删除和权限过滤。
- 如何通过 Recall@K、MRR、nDCG 和 Ragas 判断一次改动是否真的有效。
- 如何控制上下文长度、引用来源、无答案拒答、延迟和并发。
- 如何从手写实现迁移到 LangChain、Milvus 和可部署 API。

## 七、调研依据

以下信息用于确定主线，不代表对整个招聘市场的严格统计普查：

- [智联招聘：AI 智能体开发工程师](https://www.zhaopin.com/jobdetail/CCL1378323050J40908085503.htm)：Milvus/FAISS/Elastic、LangChain/LangGraph、Python Web API、权限、缓存、可观测和 CI/CD。
- [智联招聘：AI 算法应用工程师](https://m.zhaopin.com/jobs/CC343084180J40907399912.htm)：文档解析、切片、Embedding、向量数据库、召回、重排、评测、FastAPI。
- [智联招聘：AI Agent 开发工程师](https://www.zhaopin.com/jobdetail/CC667241620J40864462209.htm)：Milvus/FAISS/PGVector、Redis、LangGraph、vLLM/TGI、Docker/Kubernetes。
- [智联招聘：大模型应用工程师](https://www.zhaopin.com/jobdetail/CCL1281633590J40828735901.htm)：Milvus/Qdrant/ES、检索精度优化、FastAPI、PostgreSQL、消息队列、Linux/Docker/Git。
- [智联招聘：高级 AI 应用开发工程师](https://www.zhaopin.com/jobdetail/CCL1487882050J40838974710.htm)：多路召回、Rerank、检索评测、vLLM、LangChain/LangGraph。
- [Milvus 官方：BM25 Function](https://milvus.io/docs/bm25-function.md)
- [Milvus 官方：Full Text Search 与 dense hybrid retrieval](https://milvus.io/docs/full_text_search_with_milvus.md)
- [Milvus 官方：Hybrid Search](https://milvus.io/docs/hybrid_search_with_milvus.md)
- [Qwen 官方：Qwen3 Embedding 与 Reranker](https://qwenlm.github.io/blog/qwen3-embedding/)
- [LangChain 官方：Vector store integrations](https://docs.langchain.com/oss/python/integrations/vectorstores)
- [Ragas 官方：RAG evaluation](https://docs.ragas.io/en/stable/references/evaluate/)


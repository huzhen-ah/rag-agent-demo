# 个人简历

## 基本信息

- 姓名：待填写
- 手机：待填写
- 邮箱：待填写
- 所在地：广州
- 求职方向：大模型应用 / RAG / Agent

## 专业概述

具备计算机专业背景和智能客服对话系统开发经历，理解传统对话系统与大模型应用的差异。近期完成 Mini LLM、手写 RAG 以及 LangChain + Milvus RAG 项目，能够解释文档切分、Dense 检索、BM25、RRF、CrossEncoder 重排、检索评测和生成式问答的完整流程。

## 技术能力

- Python：能够使用原生 Python 完成数据处理、模型推理和应用流程开发。
- 大模型基础：理解 Transformer、Embedding、生成模型及常见微调流程。
- RAG：掌握文档加载、Chunking、Dense Retrieval、BM25、RRF、Reranker 和生成。
- LangChain：能够使用 Document、Retriever、VectorStore、Reranker 和 LCEL 搭建 RAG。
- Milvus：能够设计 Collection Schema、Dense/Sparse 字段、HNSW 索引和 BM25 Function。
- 检索评测：理解并使用 MRR、Recall@K 对检索阶段进行评估。
- 对话系统：具有智能客服对话系统经验，接触过知识图谱和 Neo4j 数据更新。
- 深度学习：学习过 VAE、Flow、DDPM、DDIM 和 Stable Diffusion 等生成模型原理。

## 项目经历

### RAG & Agent Demo

- 使用原生 Python 实现文档加载、固定窗口切分和本地问答流程。
- 使用 Qwen3-Embedding-0.6B 生成归一化 Dense Embedding，并通过矩阵乘法完成语义检索。
- 使用 Jieba 分词并手写 BM25，包括词频、文档频率、IDF、词频饱和和文档长度归一化。
- 手写 RRF，将 Dense 与 BM25 两路候选按排名融合。
- 使用 Qwen3-Reranker-0.6B 对候选进行 CrossEncoder 重排。
- 使用 MRR 和 Recall@K 分别评测 Dense、BM25、RRF 和 Reranker。
- 使用 LangChain 与 Milvus Standalone 重构 RAG，完成 Dense + Milvus 原生 BM25 混合检索。
- 使用 LCEL 串联 Retriever、Reranker、Prompt、本地 Qwen3-1.7B 和输出解析。

### Mini LLM Demo

- 项目地址：`huzhen-ah/mini-llm-demo`
- 从底层流程理解并实现 Mini LLM 相关模块。
- 项目具体功能、训练数据和实验结果待根据实际仓库补充。

### 智能客服对话系统

- 参与智能客服对话系统相关开发。
- 具有对话系统业务经验，理解用户意图、知识查询和对话流程的基本问题。
- 接触 Neo4j 知识数据的定时或实时更新，理解数据源与知识库之间的同步和最终一致性。
- 公司名称、任职时间、岗位和具体职责待补充。

## 教育背景

- 学校：待填写
- 专业：计算机相关专业
- 学历：待填写
- 时间：待填写
- 毕业论文：隐马尔可夫模型相关应用

## 项目链接

- Mini LLM：<https://github.com/huzhen-ah/mini-llm-demo>
- RAG & Agent：<https://github.com/huzhen-ah/rag-agent-demo>

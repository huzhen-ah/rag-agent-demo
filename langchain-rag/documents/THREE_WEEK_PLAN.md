# 三周大模型训练、推理、RAG、Agent 学习计划

## 目标

用 3 周全职时间，从当前已经跑通的 `Qwen3-0.6B-Base` 出发，建立一条完整但不贪大的主流 LLM 工程学习路径，并形成一个可以展示、复盘、写进简历/作品集的项目。

1. 能用 Hugging Face Transformers 做本地推理和多轮对话。
2. 能理解 tokenizer、模型加载、生成参数、显存/内存、device 等核心概念。
3. 能用 `datasets`、`Trainer`、`peft`、`trl` 完成一次 LoRA/SFT 微调。
4. 能做一个最小可用 RAG：文档切分、embedding、向量检索、拼接 prompt、生成答案。
5. 能做一个最小 Agent：模型根据任务选择工具，执行工具，再整合结果回答。

这 3 周的核心不是“学完所有框架”，而是跑通一条完整主线：**推理 → 数据 → 微调 → 检索增强 → 工具调用/Agent**。

因为现在是全职状态，计划按每天 6-8 小时设计。每天不仅要学习，还要留下可验证资产：

- 可运行脚本。
- 实验记录。
- 错误排查记录。
- README 说明。
- 最终 demo。

三周结束时，目标不是“我看过这些东西”，而是能清楚展示：

> 我用 Hugging Face 生态，从本地 Qwen 推理开始，完成了 LoRA 微调、RAG 问答和一个最小 Agent 系统。

## 求职/作品集导向

这份计划按“学习 + 项目 + 面试表达”三条线并行：

1. **学习线**：理解 Transformers、datasets、PEFT、TRL、RAG、Agent 的核心机制。
2. **工程线**：每个模块都有独立脚本，可以复现，可以讲清楚输入输出。
3. **作品线**：最后整理成一个 `hf-finetune-lab` 项目，README 里写清架构、运行命令、实验结果和下一步优化。

最终可以包装成一个项目：

```text
本地大模型微调与 RAG/Agent 实验室
```

简历表达可以是：

```text
基于 Qwen3-0.6B 和 Hugging Face 生态，构建本地 LLM 实验项目，覆盖 Transformers 推理、Tokenizer 分析、LoRA/SFT 微调、文档 RAG 检索问答与 ReAct Agent 工具调用，实现从模型加载到应用增强的完整闭环。
```

## 当前起点

当前目录：

```text
/Users/huzhen/Desktop/hf-finetune-lab
```

已有模型：

```text
models/Qwen3-0.6B-Base
```

可用 Conda 环境：

```bash
conda run -n ENV_hf python valid_python.py
```

已跑通推理脚本：

```bash
conda run -n ENV_hf python run_qwen.py --prompt "你好，千问。" --max-new-tokens 80
```

当前环境里 `mps available: False`，所以先按 CPU 学习。0.6B 足够用于理解流程、调试代码、跑小样本微调和 RAG 原型。

## 推荐目录结构

三周内逐步把当前目录整理成：

```text
hf-finetune-lab/
  models/
    Qwen3-0.6B-Base/
  01_inference/
    run_qwen.py
    chat_qwen.py
    inspect_generation.py
  02_tokenizer/
    inspect_tokenizer.py
    token_count.py
  sft/
    data/
      sample_sft.jsonl
      train_sft.jsonl
      eval_sft.jsonl
    build_sft_data.py
  04_sft_lora/
    train_lora.py
    infer_lora.py
    outputs/
  05_rag/
    docs/
    build_index.py
    query_rag.py
  06_agent/
    tools.py
    simple_react_agent.py
  notes/
    day_logs.md
```

## 每周成果

### 第 1 周：推理和 Hugging Face 基础

目标：把模型当成一个可控的本地推理组件，而不是黑盒。

最终成果：

- 一个单轮推理脚本。
- 一个多轮聊天脚本。
- 一个 tokenizer 观察脚本。
- 一份生成参数实验记录。

#### Day 1：整理环境和基础推理

任务：

- 确认 `ENV_hf` 环境可用。
- 把 `run_qwen.py` 移入 `01_inference/`。
- 学会用 `AutoTokenizer.from_pretrained` 和 `AutoModelForCausalLM.from_pretrained`。
- 跑 5 个不同 prompt，观察输出质量和速度。

产出：

- `01_inference/run_qwen.py`
- `notes/day_logs.md` 记录：模型路径、环境、运行命令、遇到的问题。

验收：

```bash
conda run -n ENV_hf python 01_inference/run_qwen.py --prompt "解释一下什么是 LoRA。" --max-new-tokens 120
```

#### Day 2：多轮聊天

任务：

- 写 `chat_qwen.py`。
- 支持循环输入。
- 支持输入 `exit` 退出。
- 保留最近若干轮上下文。
- 尝试中文问答、英文问答、代码解释。

产出：

- `01_inference/chat_qwen.py`

验收：

```bash
conda run -n ENV_hf python 01_inference/chat_qwen.py
```

能连续问 3 轮，模型能看到前文。

#### Day 3：生成参数实验

任务：

- 对比 `temperature=0`、`0.3`、`0.7`、`1.0`。
- 对比 `top_p=0.8`、`0.9`、`1.0`。
- 对比 `max_new_tokens=32`、`128`、`512`。
- 理解贪心解码、采样、重复、幻觉。

产出：

- `01_inference/inspect_generation.py`
- `notes/day_logs.md` 增加实验结论。

验收：

能用自己的话解释：

- `max_new_tokens` 控制什么。
- `temperature` 变大会怎样。
- `top_p` 是怎么限制候选 token 的。

#### Day 4：tokenizer 基础

任务：

- 写脚本把文本编码成 token id。
- 打印 token、token id、decode 结果。
- 比较中文、英文、数字、代码的 token 数。
- 理解 context length 为什么重要。

产出：

- `02_tokenizer/inspect_tokenizer.py`
- `02_tokenizer/token_count.py`

验收：

```bash
conda run -n ENV_hf python 02_tokenizer/token_count.py --text "你好，世界。"
```

能输出 token 数量和 token 明细。

#### Day 5：模型加载和设备理解

任务：

- 理解 CPU、CUDA、MPS 的区别。
- 理解 `float32`、`float16`、`bfloat16`。
- 理解 `model.eval()` 和 `torch.no_grad()`。
- 观察模型文件大小、加载时间、推理速度。

产出：

- `notes/day_logs.md` 增加模型加载笔记。

验收：

能解释：

- 为什么当前机器跑的是 CPU。
- 为什么推理时不用反向传播。
- 为什么 0.6B 模型也有几百 MB 到 GB 级文件。

#### Day 6：小复盘和重构

任务：

- 整理 `01_inference/` 和 `02_tokenizer/`。
- 把重复加载模型的逻辑抽成简单函数。
- 给脚本加参数说明。
- 删除明显临时的测试代码。

产出：

- 结构清晰的推理和 tokenizer 脚本。

验收：

从零打开项目，能按 README/计划里的命令跑通推理。

#### Day 7：缓冲日

任务：

- 补前 6 天没跑通的内容。
- 记录最困惑的 3 个问题。
- 不新增大任务。

产出：

- `notes/day_logs.md` 周总结。

验收：

能口头复述 Hugging Face 本地推理的完整流程。

### 第 2 周：数据、SFT、LoRA 微调

目标：用小数据跑通一次训练闭环。不要追求效果，先追求流程正确。

最终成果：

- 一个 JSONL 指令数据集。
- 一个 LoRA/SFT 训练脚本。
- 一个加载 adapter 推理脚本。
- 一份训练日志和效果对比。

#### Day 8：SFT 数据格式

任务：

- 学习 instruction/input/output 或 messages 格式。
- 手写 20 条小样本数据。
- 数据主题建议：个人助手、古文解释、Python 教学、项目问答，任选一个。
- 用脚本检查 JSONL 每行合法性。

产出：

- `sft/data/sample_sft.jsonl`
- `sft/build_sft_data.py`

验收：

```bash
conda run -n ENV_hf python sft/build_sft_data.py
```

能检查并打印样本数量。

#### Day 9：datasets 加载数据

任务：

- 用 `datasets.load_dataset` 加载 JSONL。
- 打印样本。
- map 成训练需要的文本。
- 理解 train/eval split。

产出：

- `sft/load_sft_data.py`

验收：

能把 JSONL 转成模型训练文本，例如：

```text
用户：...
助手：...
```

#### Day 10：LoRA 概念和 PEFT

任务：

- 理解 LoRA 不是全量训练，而是训练低秩适配器。
- 理解 `r`、`lora_alpha`、`target_modules`、`lora_dropout`。
- 用 `peft` 包装模型。
- 打印可训练参数比例。

产出：

- `04_sft_lora/inspect_lora.py`

验收：

能看到 trainable params 远小于 total params。

#### Day 11：跑通最小 SFT/LoRA 训练

任务：

- 写 `train_lora.py`。
- 使用小 batch、小 max length、小数据集。
- 先只训练几十步。
- 保存 adapter。

产出：

- `04_sft_lora/train_lora.py`
- `04_sft_lora/outputs/`

验收：

```bash
conda run -n ENV_hf python 04_sft_lora/train_lora.py
```

能完成训练并保存 adapter。慢没关系，能结束最重要。

#### Day 12：加载 LoRA adapter 推理

任务：

- 写 `infer_lora.py`。
- 加载 base model。
- 加载 LoRA adapter。
- 对比微调前后同一个 prompt 的输出。

产出：

- `04_sft_lora/infer_lora.py`
- `notes/day_logs.md` 记录前后对比。

验收：

能证明 adapter 被加载了，而不是仍在裸跑 base model。

#### Day 13：理解 Trainer / TRL

任务：

- 对比 `transformers.Trainer` 和 `trl.SFTTrainer`。
- 看懂 training arguments。
- 理解 epoch、step、batch size、gradient accumulation、learning rate。
- 记录训练中最重要的 5 个参数。

产出：

- `notes/day_logs.md` 增加训练参数笔记。

验收：

能解释为什么小机器要用小 batch 和 LoRA。

#### Day 14：缓冲和复盘

任务：

- 修复训练脚本。
- 简化难维护的代码。
- 整理数据集格式。
- 记录第 2 周总结。

产出：

- 可重复运行的微调闭环。

验收：

从 base model → 数据 → 训练 → adapter → 推理，完整跑通一次。

### 第 3 周：RAG 和 Agent

目标：把模型从“只会生成”扩展成“能查资料、能用工具”的系统。

最终成果：

- 一个本地文档 RAG。
- 一个简单 ReAct Agent。
- 一个最终 demo：用户提问，系统检索/调用工具，模型整合回答。

#### Day 15：RAG 基础概念

任务：

- 理解 RAG = Retrieval-Augmented Generation。
- 准备 3-5 个小文档，放入 `05_rag/docs/`。
- 写文档读取和 chunking。
- 每个 chunk 保留来源文件名。

产出：

- `05_rag/docs/`
- `05_rag/chunk_docs.py`

验收：

能把文档切成多个 chunk，并打印 chunk 数量。

#### Day 16：Embedding 和向量索引

任务：

- 先用轻量方案：`sentence-transformers` 或 Hugging Face embedding 模型。
- 如果本地缺依赖，先用简单 TF-IDF/BM25 替代，流程优先。
- 建立 FAISS 或 Chroma 索引。
- 保存索引到本地。

产出：

- `05_rag/build_index.py`
- `05_rag/index/`

验收：

```bash
conda run -n ENV_hf python 05_rag/build_index.py
```

能完成索引构建。

#### Day 17：检索和拼接 Prompt

任务：

- 写 `query_rag.py`。
- 输入用户问题。
- 检索 Top-K chunk。
- 把 chunk 拼到 prompt 中。
- 调用 Qwen 生成答案。

产出：

- `05_rag/query_rag.py`

验收：

```bash
conda run -n ENV_hf python 05_rag/query_rag.py --question "文档里说了什么？"
```

回答中能引用检索到的文档内容。

#### Day 18：RAG 质量改进

任务：

- 调整 chunk size。
- 调整 Top-K。
- 加来源引用。
- 对比“裸模型回答”和“RAG 回答”。

产出：

- `notes/day_logs.md` 增加 RAG 对比。

验收：

能说明 RAG 为什么能降低幻觉，但不能完全消灭幻觉。

#### Day 19：Agent 基础和工具

任务：

- 理解 Agent = LLM + planning loop + tools + memory/context。
- 写 3 个工具：
  - `read_file(path)`
  - `calculator(expression)`
  - `search_docs(question)`
- 先不用复杂框架，手写工具调用流程。

产出：

- `06_agent/tools.py`

验收：

工具函数可单独运行和测试。

#### Day 20：Simple ReAct Agent

任务：

- 写 `simple_react_agent.py`。
- 让模型按固定格式输出：
  - Thought
  - Action
  - Action Input
  - Observation
  - Final Answer
- Python 解析 Action 并调用工具。
- 最多循环 3 轮，防止跑飞。

产出：

- `06_agent/simple_react_agent.py`

验收：

输入一个需要查文档或计算的问题，Agent 能调用工具再回答。

#### Day 21：最终整合 Demo

任务：

- 整理项目目录。
- 写最终运行说明。
- 做一次完整演示：
  - 本地模型推理。
  - LoRA adapter 推理。
  - RAG 问答。
  - Agent 工具调用。
- 记录下一阶段计划。

产出：

- `README.md`
- `notes/day_logs.md` 三周总结。

验收：

你能打开项目，向别人演示：

1. 我能本地加载 Qwen。
2. 我能准备数据并做 LoRA 微调。
3. 我能让模型查自己的资料回答。
4. 我能让模型调用工具完成简单任务。

## 每天固定节奏

全职学习建议每天 6-8 小时，按“概念 → 代码 → 验证 → 记录 → 表达”循环推进：

```text
09:30-10:30  概念输入：只看当天任务相关内容，不刷长教程
10:30-12:30  第一轮实现：写最小脚本，先跑通
12:30-14:00  休息
14:00-16:00  第二轮实现：补参数、错误处理、实验对比
16:00-17:00  Debug/重构：清理重复代码，保证可复现
17:00-18:00  记录输出：写 day_logs、README 片段、明日 TODO
20:00-21:00  可选加练：复述概念、看源码、补面试表达
```

如果某天状态不好，保底完成：

1. 一个能运行的脚本。
2. 一条可复现命令。
3. 一段错误/结论记录。

如果某天状态很好，追加完成：

1. 给脚本加命令行参数。
2. 写 README 说明。
3. 做一次前后对比实验。
4. 把当天内容整理成面试可讲的 3 分钟版本。

不要把大量时间花在看教程。每天必须有一个能运行的产物。

## 每日交付模板

每天结束时，在 `notes/day_logs.md` 里按这个模板记录：

````markdown
## Day N

### 今天完成

-

### 运行命令

```bash

```

### 遇到的问题

-

### 我现在能解释

-

### 明天第一件事

-
````

这个记录很重要。它不是日记，是你三周后的复盘材料、README 素材、面试表达素材。

## 每周作品集检查点

### 第 1 周结束

你应该能展示：

- 本地 Qwen 推理。
- 多轮聊天。
- tokenizer 如何把文本变成 token。
- 不同生成参数对输出的影响。

可写入 README 的项目亮点：

```text
实现本地 Qwen3-0.6B 推理与多轮对话脚本，支持生成参数调节，并通过 tokenizer 分析脚本观察中文、英文、代码等不同文本的 tokenization 行为。
```

### 第 2 周结束

你应该能展示：

- 一个小型 SFT 数据集。
- LoRA adapter 训练过程。
- 微调前后输出对比。
- 可训练参数比例。

可写入 README 的项目亮点：

```text
基于 PEFT/LoRA 完成小样本 SFT 微调，保存并加载 adapter，对比 base model 与 adapter model 的输出差异，理解低秩适配器训练流程。
```

### 第 3 周结束

你应该能展示：

- 文档 RAG 问答。
- 检索 chunk 和来源引用。
- 一个简单 ReAct Agent。
- 工具调用过程。

可写入 README 的项目亮点：

```text
实现最小 RAG 问答系统和手写 ReAct Agent，支持文档检索、上下文增强、工具选择、工具执行和最终答案整合。
```

## 关键原则

### 1. 小模型优先

这 3 周不要急着换大模型。0.6B 的优势是：

- 加载快。
- 出错成本低。
- 适合 CPU 调试。
- 能完整体验工程流程。

等流程熟了，再换 1.8B、4B 或 API 模型。

### 2. 先跑通，再优雅

第一版脚本允许朴素、重复、丑一点。先把链路跑通，再重构。

优先级：

```text
能运行 > 能复现 > 能解释 > 代码漂亮
```

### 3. 不追求训练效果

小数据、小模型、CPU 环境下，微调效果可能不惊艳。第 2 周真正目标是理解：

- 数据如何进模型。
- loss 如何下降。
- adapter 如何保存和加载。
- 推理时如何合并 base model 和 adapter。

### 4. RAG 和 Agent 先手写

LangChain、LlamaIndex 可以后面学。前三周建议先手写最小版本，因为这样能看清本质：

- RAG 本质是检索文本并塞进 prompt。
- Agent 本质是模型决定下一步调用什么工具。

框架是加速器，不是第一性原理。

## 必须掌握的命令

检查环境：

```bash
conda run -n ENV_hf python valid_python.py
```

运行基础推理：

```bash
conda run -n ENV_hf python run_qwen.py --prompt "解释一下什么是 RAG。" --max-new-tokens 120
```

进入项目目录：

```bash
cd /Users/huzhen/Desktop/hf-finetune-lab
```

Spyder 解释器建议：

```text
/opt/anaconda3/envs/ENV_hf/bin/python
```

## 三周结束后的下一步

如果三周计划完成，下一阶段建议：

1. 换更强模型：Qwen 1.8B/4B 或 API 模型。
2. 学量化：4-bit、8-bit、GGUF、llama.cpp。
3. 学部署：FastAPI、Gradio、OpenAI-compatible server。
4. 学生产 RAG：重排、混合检索、评估、引用、权限。
5. 学多 Agent：planner、executor、critic、memory。

## 最小成功标准

如果时间紧，至少完成这些：

- 第 1 周：`chat_qwen.py`
- 第 2 周：`train_lora.py` + `infer_lora.py`
- 第 3 周：`query_rag.py` + `simple_react_agent.py`

只要这 5 个脚本跑通，你就已经完成了从本地大模型到 RAG/Agent 的核心入门闭环。

## 全职版成功标准

因为你现在是全职投入，真正建议冲到这个标准：

1. `README.md` 能让别人按命令复现主要 demo。
2. `notes/day_logs.md` 记录 21 天关键命令、错误和结论。
3. 每个模块至少有一个可运行入口：
   - `01_inference/chat_qwen.py`
   - `02_tokenizer/token_count.py`
   - `04_sft_lora/train_lora.py`
   - `04_sft_lora/infer_lora.py`
   - `05_rag/query_rag.py`
   - `06_agent/simple_react_agent.py`
4. 能讲清楚 6 个问题：
   - Transformers 推理流程是什么。
   - tokenizer 为什么重要。
   - LoRA 为什么省资源。
   - SFT 数据如何组织。
   - RAG 为什么能减少幻觉。
   - Agent 和普通聊天机器人有什么区别。
5. 能做 10 分钟演示：
   - 本地推理 2 分钟。
   - tokenizer 分析 1 分钟。
   - LoRA 微调前后对比 2 分钟。
   - RAG 文档问答 2 分钟。
   - Agent 工具调用 2 分钟。
   - 总结下一步优化 1 分钟。

这就是三周后的“可展示版本”。它不需要完美，但必须完整、可运行、可解释。

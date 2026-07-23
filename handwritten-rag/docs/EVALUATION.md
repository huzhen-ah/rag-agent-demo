# 检索评测说明与实验结果

## 署名与贡献说明

- 本文对应的项目代码由项目作者独立手写完成。
- 本文档由 OpenAI Codex 根据现有代码与实验结果起草。
- 文档内容由项目作者逐项审核、修改并最终确认。

## 1. 评测目标

本项目分别评估四个检索阶段：

```text
Dense
BM25
Dense + BM25 + RRF
Dense + BM25 + RRF + Reranker
```

阶段评估用于回答：

- Dense 是否能召回语义相关 chunks？
- BM25 是否能补充关键词、编号和专有名词召回？
- RRF 是否能利用两路排名互补改善结果？
- Reranker 是否能将正确候选排到更靠前的位置？

本评测只衡量检索质量，不评估生成答案的正确性、完整性或忠实性。

## 2. 评测数据

评测文件：

```text
dataset/retrieval_eval.jsonl
```

当前包含 10 个查询，每行格式：

```json
{
  "question": "三周学习计划要跑通的完整主线是什么？",
  "relevant_chunk_ids": [0, 1]
}
```

字段含义：

- `question`：输入检索器的问题。
- `relevant_chunk_ids`：人工标注的相关 chunks，也称 ground truth。

一个问题可以对应一个或多个相关 chunks。

对于每个问题，检索器返回一个有序 chunk 列表：

```text
[3, 1, 8, 0, 5]
```

如果正确集合是：

```text
{0, 1}
```

那么：

```text
第2名 chunk 1：相关
第4名 chunk 0：相关
```

## 3. Recall@K

Recall@K 衡量：

> 所有标注相关 chunks 中，有多少出现在检索结果的前 K 名。

公式：

```text
Recall@K
=
|Top-K检索结果 ∩ 相关chunk集合|
/
|相关chunk集合|
```

使用上面的例子：

```text
检索结果：[3, 1, 8, 0, 5]
相关集合：{0, 1}
```

得到：

```text
Recall@1 = 0 / 2 = 0.0
Recall@3 = 1 / 2 = 0.5
Recall@5 = 2 / 2 = 1.0
```

代码实现：

```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    if len(retrieved_ids) == 0:
        return 0

    top_k_ids = retrieved_ids[:k]
    correct_ids = set(top_k_ids) & set(relevant_ids)

    return len(correct_ids) / len(relevant_ids)
```

每个问题分别计算 Recall@K，最终对全部问题取平均。

本项目输出：

```text
Recall@1
Recall@3
Recall@5
```

## 4. RR 与 MRR

RR 是 Reciprocal Rank，衡量单个问题中第一个相关 chunk 排得有多靠前：

```text
RR = 1 / 第一个相关chunk的排名
```

示例：

```text
第一个相关chunk排第1：RR = 1.0
第一个相关chunk排第2：RR = 0.5
第一个相关chunk排第3：RR = 0.3333
没有召回相关chunk：   RR = 0
```

MRR 是 Mean Reciprocal Rank，即全部问题 RR 的平均值：

```text
MRR
=
所有问题RR之和
/
问题数量
```

代码实现：

```python
def reciprocal_rank(retrieved_ids, relevant_ids):
    if len(retrieved_ids) == 0:
        return 0

    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            return 1 / (index + 1)

    return 0
```

MRR 只关注第一个相关结果。一个问题是否找全了多个相关 chunks，需要结合 Recall@K 判断。

当前代码在检索器返回的结果范围内计算 RR。如果返回列表中没有相关 chunk，则 RR 为 0。因此 MRR 也受候选返回深度影响：

```text
返回Top-5  → 实际计算MRR@5
返回Top-10 → 实际计算MRR@10
返回Top-15 → 实际计算MRR@15
```

因此下面两种结果可以同时成立：

```text
MRR = 1.0
Recall@1 < 1.0
```

原因是：每个问题的第一个结果都可以是相关的，但如果某个问题有两个相关 chunks，Top-1 最多只覆盖其中一个。

## 5. 评测脚本

| 脚本 | 评测阶段 |
|---|---|
| `evaluate_retrieval_dense.py` | Dense |
| `evaluate_retrieval_bm25.py` | BM25 |
| `evaluate_retrieval_rrf.py` | Dense + BM25 + RRF |
| `evaluate_retrieval_reranker.py` | 最终 RRF + Reranker 链路 |

当前脚本参数：

| 脚本 | 当前候选配置 |
|---|---|
| `evaluate_retrieval_dense.py` | Dense Top-15 |
| `evaluate_retrieval_bm25.py` | BM25 Top-15 |
| `evaluate_retrieval_rrf.py` | Dense Top-15 + BM25 Top-15 → RRF Top-10 |
| `evaluate_retrieval_reranker.py` | Dense Top-15 + BM25 Top-15 → RRF Top-10 → Reranker Top-5 |

运行：

```bash
python evaluate_retrieval_dense.py
python evaluate_retrieval_bm25.py
python evaluate_retrieval_rrf.py
python evaluate_retrieval_reranker.py
```

每个脚本执行相同的基本过程：

```text
读取 retrieval_eval.jsonl
→ 对每个 question 执行检索
→ 提取有序 chunk_id
→ 计算 Recall@1/3/5 和 RR
→ 对全部问题取平均
```

## 6. 受控 Top-5 阶段实验

项目开发过程中曾使用以下统一 Top-5 配置，逐阶段记录结果：

```text
Dense Top-5
BM25 Top-5
→ RRF Top-5
→ Reranker重排同一批Top-5
```

这组结果是参数扩大前的阶段实验记录。若要用当前脚本复现，需要把各阶段的候选和输出数量临时设回 5。

结果：

| 检索阶段 | MRR@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| Dense Top-5 | 0.4817 | 0.25 | 0.50 | 0.85 |
| BM25 Top-5 | 0.6833 | 0.35 | 0.95 | 0.95 |
| RRF Top-5 | 0.9000 | 0.70 | 0.95 | 0.95 |
| RRF Top-5 + Reranker | **1.0000** | **0.90** | 0.95 | 0.95 |

这个实验中，Reranker 的输入和输出是同一批 5 个候选，因此：

- Recall@5 不会发生变化。
- MRR 与 Recall@1 的提升来自候选顺序变化。
- Reranker 将每个问题的至少一个相关 chunk 排到了第一名。

## 7. 最终端到端配置

最终 Demo 使用更大的前级候选池：

```text
Dense Top-15
BM25 Top-15
→ RRF Top-10，c=60
→ Reranker Top-5
```

最终结果：

| 配置 | MRR@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| 15 → 15 → 10 → 5 | **1.0000** | **0.90** | **1.00** | **1.00** |

与受控 Top-5 实验相比：

```text
Recall@3：0.95 → 1.00
Recall@5：0.95 → 1.00
```

该提升来自更大的 Dense/BM25 候选池和 RRF 候选池，使此前漏掉的相关 chunk 有机会进入 Reranker。

因此不能把 `Recall@3/5` 的这部分提升单独归因于 Reranker。Reranker 的独立排序贡献应参考上一节的受控 Top-5 对比。

## 8. 实验结论

### Dense

受控 Top-5 阶段结果：

```text
MRR      = 0.4817
Recall@1 = 0.25
Recall@3 = 0.50
Recall@5 = 0.85
```

Dense 能召回大部分语义相关内容，但在当前评测集上的前排排序较弱。

### BM25

受控 Top-5 阶段结果：

```text
MRR      = 0.6833
Recall@1 = 0.35
Recall@3 = 0.95
Recall@5 = 0.95
```

当前问题包含 `Day 10`、`LoRA`、`Agent` 等与原文一致的关键词、编号和术语，因此 BM25 明显受益。

### RRF

受控 Top-5 阶段结果：

```text
MRR      = 0.9000
Recall@1 = 0.70
Recall@3 = 0.95
Recall@5 = 0.95
```

RRF 结合 Dense 与 BM25 的排名，将两路共同认可的 chunks 推到更靠前的位置。主要收益体现在排序，而不是扩大 Top-5 的召回上限。

### Reranker

受控 Top-5 候选池下：

```text
MRR      = 1.0000
Recall@1 = 0.90
Recall@3 = 0.95
Recall@5 = 0.95
```

Reranker 进一步改善前排顺序。最终扩大候选池后，Recall@3 和 Recall@5 均达到 1.0。

## 9. 结果解释边界

当前结果不能被解释为真实业务准确率，原因包括：

- 评测集只有 10 个问题。
- 问题来源于同一份学习计划。
- 多数问题与原文措辞接近。
- `relevant_chunk_ids` 由人工根据当前切块结果标注。
- 当前没有困难负样本、改写问题、歧义问题或跨文档问题。
- 当前没有单独的训练集、开发集和测试集。
- 修改文档或切块参数后，整数 `chunk_id` 可能变化，评测标注需要同步更新。

这些指标的合理用途是：

- 验证检索与评测代码已经跑通。
- 比较同一评测集上不同检索阶段的相对变化。
- 定位召回不足还是排序不足。
- 为后续 Milvus 与 LangChain 版本提供回归基线。

## 10. 后续评测方向

进入 Milvus 与 LangChain 阶段后，可以在保持当前测试集作为回归集的同时补充：

- 同义改写问题。
- 只依靠精确编号才能回答的问题。
- 只依靠语义才能回答的问题。
- 需要同时召回多个 chunks 的问题。
- 知识库中没有答案的问题。
- 多文档与相似干扰文档。

后续还可以增加：

```text
Precision@K
nDCG@K
检索延迟
Reranker延迟
生成答案忠实性
引用正确性
无答案拒答率
```

这些不属于当前手撕版的必做范围。

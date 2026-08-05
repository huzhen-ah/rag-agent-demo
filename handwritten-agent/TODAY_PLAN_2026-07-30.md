# 2026-07-30 手写 Agent 学习与开发计划

## 一、结论：当前路线是否属于主流方案

当前项目采用的核心路线属于主流 Tool-Calling Agent 方案：

```text
用户输入
→ 模型生成结构化 Tool Call
→ 运行时校验并执行 Tool
→ Tool Result 作为 Observation 回填
→ 模型基于新消息继续决策
→ 输出最终回答或达到执行上限
```

这条路线与当前主流 Agent 框架的核心运行方式一致：

- Tool 使用名称、描述和参数 Schema 对外声明。
- LLM 只负责提出 Tool Call，不直接执行 Python 函数。
- Agent Runtime 负责工具路由、参数传递、异常捕获和结果回填。
- 消息历史保存当前运行所需的短期状态。
- Agent 在“模型节点”和“工具节点”之间循环，直到模型给出最终回答或达到停止条件。
- 确定性计算交给普通代码，LLM 负责语义提取、判断和自然语言表达。

当前实现使用 Qwen Chat Template 原生支持的 Tool Calling 协议：

```html
<tool_call>
{"name": "tool_name", "arguments": {}}
</tool_call>
```

这比重新设计一套 `action/action_input` 文本协议更贴近模型原生能力，也更接近 Hugging Face 和主流模型服务的使用方式。

参考资料：

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/en/chat_templating_writing)

## 二、需要修正的两个设计认识

### 1. 暂时不额外设计复杂的 Agent State

当前 `messages` 已经保存了：

- System Prompt。
- 用户请求。
- 模型产生的 Tool Call。
- Tool 返回的 Observation。
- 模型后续决策。

对于当前单次运行、最多数步的手写 Agent，这已经是足够的最小状态。

今天只增加独立的执行轨迹 `trace`，用于调试和评测，不提前加入暂时用不到的 `requirements`、`evidence`、`score` 等重复字段。

以后进入 LangGraph、持久化、中断恢复或 Human-in-the-loop 阶段，再正式设计结构化 State。

### 2. 评分函数的教学实现与生产实现要区分

今天将 `calculate_job_fit_score` 封装成 Tool，用于完整学习：

```text
模型构造结构化 assessments
→ Tool 校验输入
→ Python 按固定公式计算
→ 模型解释计算结果
```

这样做适合当前“理解 Tool Calling”的教学目标。

但是在更可靠的生产工作流中，如果“计算匹配分”是每次岗位分析都必须执行的步骤，就不应该完全依赖 LLM 自觉调用。更合适的方案是：

```text
LLM提取并判断 requirements
→ 编排代码强制执行评分节点
→ LLM生成最终说明
```

也就是将必要的确定性步骤设计为 Workflow 的必经节点，而不是可选的 Agent Action。

## 三、今天的学习目标

今天不扩展到 LangGraph、FastAPI 或前端，只完成手写 Agent 的最小闭环。

完成后应当能够解释：

```text
Agent
= Model Decision
+ Tool Schema
+ Runtime Dispatch
+ Observation Feedback
+ Message State
+ Stop Condition
+ Error Handling
+ Execution Trace
```

重点理解以下边界：

1. LLM 决定“想调用什么”，Runtime 决定“是否允许以及怎样执行”。
2. Tool Schema 是模型与程序之间的接口契约。
3. Tool Result 和 Tool Error 都是 Observation。
4. `messages` 是当前最小 Agent 的短期状态。
5. `max_steps` 是安全停止条件，不是固定业务步骤数。
6. 语义判断可以由 LLM 完成，最终分数必须由确定性代码计算。

## 四、今天的开发任务

### Task 1：将评分函数封装为第三个 Tool

在 `tools.py` 中接入现有的 `calculate_job_fit_score`：

```text
calculate_job_fit
```

输入结构：

```json
{
  "assessments": [
    {
      "requirement": "熟悉RAG开发",
      "category": "hard_required",
      "status": "matched",
      "evidence": "已完成手写混合检索RAG项目"
    }
  ]
}
```

其中评分函数真正使用：

- `category`：`hard_required` 或 `bonus`
- `status`：`matched`、`partial` 或 `missing`

`requirement` 和 `evidence` 用于提高执行轨迹和最终解释的可读性，不参与数值公式。

验收要求：

- 合法输入返回 `score`、`hard_coverage` 和 `bonus_coverage`。
- 空列表返回明确错误。
- 非法 `category` 返回明确错误。
- 非法 `status` 返回明确错误。

### Task 2：完善 Agent 的执行轨迹

为每次运行记录：

```text
step
tool_name
arguments
observation
error
```

推荐的最小结构：

```python
trace = [
    {
        "step": 1,
        "tool_name": "read_resume",
        "arguments": {"resume_id": "main"},
        "observation": "...",
        "error": None,
    }
]
```

要求：

- Trace 由 Runtime 记录，不让 LLM 自己总结执行历史。
- 正常 Tool Result 和 Tool Error 都进入 Trace。
- 默认不要完整打印大段简历正文，只打印必要摘要或字符数。
- `run()` 返回最终回答时，应当允许调用方同时取得 Trace。

### Task 3：补齐错误反馈闭环

需要覆盖三层错误：

#### 解析错误

例如：

- `<tool_call>` 中不是合法 JSON。
- 缺少 `name`。
- `arguments` 不是对象。

处理目标：

- 不让整个 Python 程序直接崩溃。
- 将格式错误转成模型能看到的反馈。
- 允许模型在剩余步数内修正一次。

#### 路由错误

例如：

```text
模型调用不存在的工具
```

处理目标：

- Register 拒绝执行。
- Observation 明确说明工具不存在。
- 最好同时告诉模型当前允许使用的 Tool 名称。

#### 执行错误

例如：

- `resume_id` 不存在。
- 缺少必填参数。
- 参数类型不匹配。
- 工具内部读取文件失败。

处理目标：

- 捕获异常并生成结构化错误 Observation。
- Trace 中标记失败。
- 模型可以根据错误修改参数后重试。

### Task 4：收紧停止条件

保留现有的 `max_steps`，并明确其语义：

```text
模型最多进行多少轮决策
```

补充以下行为：

- 模型没有内容也没有 Tool Call：明确失败。
- 达到最大步数：返回包含执行轨迹的明确错误。
- 检测完全相同的 Tool 名称与参数连续重复调用。
- 连续重复时终止，避免无意义循环。

今天暂不实现：

- 总耗时限制。
- Tool 单次超时。
- Checkpoint。
- 中断恢复。
- 跨会话记忆。

### Task 5：完善 System Prompt

System Prompt 至少要说明：

1. 只能使用注册的 Tool。
2. 需要事实证据时必须先调用 Tool。
3. 不得编造简历和项目经历。
4. 完整岗位分析需要读取简历并搜索项目证据。
5. 匹配分必须由 `calculate_job_fit` 计算，不得自行编造。
6. Tool 返回错误时，应根据错误修正调用。
7. 已获得足够信息后应停止调用工具并给出最终答案。

注意：Prompt 只能引导模型，不能替代 Runtime 的参数校验、工具白名单和停止条件。

### Task 6：跑通三个验收场景

#### 场景 A：普通对话

输入示例：

```text
你能帮我做什么？
```

预期：

- 不调用 Tool。
- 直接给出简短回答。

#### 场景 B：完整岗位分析

输入一份包含 RAG、Python、Milvus 等要求的 JD。

预期轨迹：

```text
read_resume
→ search_project_evidence
→ calculate_job_fit
→ final answer
```

最终回答至少包含：

- 岗位核心要求。
- 匹配项和证据。
- 缺失项或不确定项。
- 工具计算得到的匹配分。
- 是否建议投递。
- 简历修改建议。

#### 场景 C：错误恢复

至少人为构造一种错误：

- 错误的 `resume_id`。
- 不存在的 Tool 名称。
- 缺少 Tool 参数。
- 非法 Tool Call JSON。

预期：

- 程序不直接崩溃。
- 错误被写入 Observation 和 Trace。
- 模型修正调用，或者在执行上限内安全结束。

## 五、建议执行顺序

```text
1. 为评分函数补充清晰类型和文档字符串
2. 注册 calculate_job_fit Tool
3. 单独测试三个 Tool
4. 给 Agent 增加 Trace
5. 处理 Parser 错误反馈
6. 增加重复调用检测和停止行为
7. 完善 System Prompt
8. 跑普通对话
9. 跑完整岗位分析
10. 跑错误恢复
11. 更新 GOAL.md 和项目 README
```

## 六、今日完成标准

满足以下条件才算完成：

- 三个 Tool 都能独立运行。
- 模型能完成至少一次多步 Tool Calling。
- 匹配分来自评分 Tool，而不是模型自由生成。
- Tool 错误不会直接导致 Agent 无信息崩溃。
- 能查看完整但不过度泄漏原文的执行轨迹。
- 重复调用或超出最大步数时能安全停止。
- 三个验收场景均留下测试结果。
- `GOAL.md` 中的 Tool Call 协议与实际代码保持一致。

## 七、今天明确不做的内容

- 不引入 LangChain Agent。
- 不引入 LangGraph。
- 不做长期记忆或持久化 State。
- 不做多 Agent。
- 不做自动投递。
- 不做自动写入定制简历。
- 不做 API 和前端。

这些能力会在手写 Agent 闭环稳定后分阶段加入。

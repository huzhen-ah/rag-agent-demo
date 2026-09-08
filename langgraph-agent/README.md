# LangGraph Agent

这是 `handwritten-agent` 的 LangGraph 重构版本，用于对照理解手写 Runtime 与官方框架之间的对应关系。

## 当前能力

- 使用 `MessagesState` 管理消息，并通过 Reducer 累计 `model_call_count`。
- 使用 `StateGraph` 编排模型、参数补充、人工审核和工具执行。
- 使用官方 `ToolNode` 执行工具。
- 使用 SQLite Checkpointer 保存线程状态。
- 使用 `interrupt()` 与 `Command(resume=...)` 实现 HITL。
- 通过 Policy 控制工具是否需要补参或审核。
- 支持 ToolCall 的 approve、edit 和 reject。
- 使用 `Send` 将待执行 ToolCall 调度为独立任务。

## 执行流程

```text
ModelNode
→ ToolArgsCompletionNode
→ ToolReviewNode
→ ToolNode
→ ModelNode 或 END
```

没有 ToolCall 时直接结束。缺少必填参数时暂停补参；需要审核时暂停等待 approve、edit 或 reject。

## 主要文件

```text
state.py    State定义
tools.py    工具定义
model.py    本地Qwen模型适配
nodes.py    业务Node
routers.py  条件路由
agent.py    图组装与运行入口
```

## 运行

模型权重放在：

```text
models/Qwen3-4B
```

然后执行：

```bash
cd langgraph-agent
python agent.py
```

发生中断后，根据终端打印的 `tool_call_id` 输入对应 JSON，再通过相同 `thread_id` 恢复执行。

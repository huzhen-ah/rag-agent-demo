# LangGraph Agent

这是 `handwritten-agent` 的 LangGraph 重构版本，用于对照理解手写 Runtime 与官方框架之间的对应关系。

## 当前能力

- 使用 `MessagesState` 管理消息，并通过 Reducer 累计 `model_call_count`。
- 使用 `StateGraph` 编排模型、工具与业务流程。
- 使用官方 `ToolNode` 执行工具。
- 使用 SQLite Checkpointer 保存线程状态。
- 使用 `interrupt()` 与 `Command(resume=...)` 实现 HITL。
- 通过 Policy 控制工具是否需要补参或审核。
- 支持 ToolCall 的 approve、edit 和 reject。
- 使用 `Send` 将待执行 ToolCall 调度为独立任务。
- 将参数补充、人工审核和工具执行封装为 Tool Workflow SubGraph。
- 使用 SQLite Store 保存跨线程的用户长期记忆。
- 使用官方 `stream(..., stream_mode="updates")` 输出节点更新事件。
- 使用 Supervisor + `task` Tool 实现 `agent_as_tool` MultiAgent。
- Resume Agent 负责简历与项目证据，RAG Agent 通过 HTTP 调用手写 RAG 服务。
- 支持 Skill Metadata 与 `read_skill` 按需加载 `SKILL.md`。

## 执行流程

```text
Supervisor ModelNode
├── read_skill → 读取SKILL.md → Supervisor ModelNode
└── task
    ├── Resume Agent → Tool Workflow SubGraph
    │   ├── ToolArgsCompletionNode
    │   ├── ToolReviewNode
    │   └── ToolNode
    └── RAG Agent → query_rag

子Agent结果 → Supervisor ModelNode → 最终回答
```

没有 ToolCall 时直接结束。缺少必填参数时暂停补参；需要审核时暂停等待 approve、edit 或 reject。Skill 只在命中用户请求时读取完整正文。

## 主要文件

```text
state.py        State定义
tools.py        工具定义
model.py        本地Qwen模型适配
nodes.py        业务Node
routers.py      条件路由
agent.py        Resume Agent与RAG Agent工厂
multi_agent.py  task Tool与Supervisor Agent工厂
skill.py        Skill Metadata加载与read_skill Tool
skills/         SKILL.md目录
demo.py         完整组装与运行入口
```

## 运行

模型权重放在：

```text
models/Qwen3-4B
```

如需使用 RAG Agent，先启动手写 RAG 服务：

```bash
conda activate ENV_rag
cd handwritten-rag
python rag_service.py
```

然后运行 LangGraph Agent：

```bash
cd langgraph-agent
python demo.py
```

发生中断后，根据终端打印的 `tool_call_id` 输入对应 JSON。`demo.py` 会通过同一 `thread_id` 从 Supervisor 根图恢复到产生中断的子 Agent。

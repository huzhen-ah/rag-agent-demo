# Handwritten Tool-Calling Agent

这是一个不依赖 LangChain、LangGraph 等 Agent 编排框架，使用原生 Python 手写的 Graph-based Tool-Calling Agent。

项目目标不是堆叠 API，而是实现并理解主流 Agent Runtime 的核心运行语义：

```text
Model 决策
→ Tool Calling
→ State Update
→ Conditional Routing
→ 下一轮执行或结束
```

当前版本是 **2026-08-07 Checkpoint 核心闭环快照**。它已经能够运行完整的 Model → Tool → Model 循环，并在每个成功提交的 Super-step 后保存可恢复的运行快照；HITL、Memory、Streaming、Subgraph 和 Multi-Agent 将在后续阶段基于当前 Runtime 继续扩展。

## 1. 当前实现范围

当前版本已经实现：

- Provider-agnostic 的内部 Message 与 ToolCall 协议。
- Qwen3 Tool Calling 格式适配。
- 模型原始输出解析与标准化。
- Runtime 生成并维护 `tool_call_id`。
- Tool Schema 生成、Tool Registry 和 Tool 调用。
- 基于 JSON Schema 的 Tool 参数校验。
- Tool 调用错误与可恢复执行错误的分类处理。
- Typed State、Reducer 和 State Update。
- 同一 Super-step 内多个 Node Update 的统一提交。
- `CompiledStateGraph` 统一持有 Reducer 映射并负责合并外部输入与 Node Updates。
- 使用加法 Reducer 累计模型调用次数，允许同一 Super-step 中多个模型 Node 分别提交调用增量。
- `StateGraph` Builder 与 `CompiledStateGraph`。
- Fixed Edge、Conditional Edge、Router 和 `path_map`。
- `START`、`END`、循环执行和最大 Super-step 限制。
- Graph 编译期结构校验与 Node 可达性校验。
- `StateSnapshot`、`Checkpointer`、`InMemoryCheckpointer` 和本地 `JsonlCheckpointer`。
- 使用 `thread_id` 隔离会话，使用 UUID `checkpoint_id` 和 `parent_checkpoint_id` 维护快照血缘。
- Super-step 成功提交后保存 `state + next_node_names`，支持从最新或指定 Checkpoint 恢复。
- 新 UserMessage 从历史 State 重新经过 `START`；没有新 Input Update 时从 `next_node_names` 继续。
- JSONL 落盘与进程重启恢复，一个 Agent 实例可以服务多个独立 thread。

当前 Resume Agent 注册了两个 Tool：

- `read_resume`：按照 `resume_id` 读取本地简历。
- `search_project_evidence`：按照岗位要求检索项目证据。

## 2. 核心运行流程

```text
User Input
    ↓
Agent.run(user_input, agent_state, thread_id, checkpoint_id)
    ↓
构造包含 UserMessage 的 input_update
    ↓
CompiledStateGraph.invoke(state, input_update, thread_id, checkpoint_id)
    ↓
按 thread_id 读取最新或指定 StateSnapshot
    ↓
通过 Reducer 将 input_update 合并到 State
    ↓
START → ModelNode
             ↓
       LocalChatModel
             ↓
       Parser / Adapter
             ↓
      AssistantMessage
        ├── 无 ToolCall → END
        └── 有 ToolCall → ToolNode
                              ↓
                    参数校验与 Tool 执行
                              ↓
                         ToolMessage
                              ↓
                           ModelNode
```

每个 Graph Super-step 遵循：

```text
Plan：确定本轮可执行 Nodes
→ Execute：各 Node 读取同一份旧 State 并返回 Partial Update
→ Update：一次性通过 Reducer 提交全部 Updates
→ Route：根据提交后的新 State 计算下一批 Nodes
```

Node 不直接修改共享 State。它的标准接口是：

```text
State → Partial State Update
```

## 3. 核心模块

| 文件 | 职责 |
|---|---|
| `state.py` | Message、ToolCall、AgentState、AgentStateUpdate 和 `add_messages` Reducer |
| `runtime.py` | 提取 State Reducer，并按照 Super-step 规则合并多个 Node Update |
| `graph.py` | StateGraph Builder、Graph 编译、Reducer 所有权、Transition、Router 调度和执行循环 |
| `checkpoint.py` | StateSnapshot、Checkpointer 接口、内存快照与 JSONL 本地持久化 |
| `model.py` | Qwen3 Model Adapter、消息格式转换和 ToolCall ID 标准化 |
| `parser.py` | 解析模型原始输出，提取 `content`、`name` 和 `arguments` |
| `nodes.py` | ModelNode 和 ToolNode |
| `routers.py` | 根据最新 State 决定进入 ToolNode 或结束 |
| `tools.py` | Tool 抽象、Tool Schema、参数校验、业务 Tool 和异常协议 |
| `tool_register.py` | Tool 注册、按名称查找以及 Tool Definition 导出 |
| `agent.py` | 组装 Resume Agent、注入 Checkpointer，并把 User Input、thread 和 checkpoint 交给 Runtime |

## 4. Message 与 ToolCall 协议

内部 ToolCall 使用统一结构：

```python
{
    "id": "call_8f31",
    "name": "read_resume",
    "args": {
        "resume_id": "main",
    },
}
```

模型适配层负责把内部的 `args` 转换成 Qwen Chat Template 使用的 `arguments`。

Tool 执行后生成 ToolMessage：

```python
{
    "id": "msg_9a12",
    "role": "tool",
    "content": "...",
    "tool_call_id": "call_8f31",
    "name": "read_resume",
    "status": "success",
}
```

其中：

- `ToolCall.id` 标识一次具体 Tool 调用。
- `ToolMessage.tool_call_id` 将执行结果关联回原 ToolCall。
- `ToolMessage.status` 是 Runtime 内部的 `success/error` 状态。
- 当前 Qwen Adapter 不向裸模型发送 `status`；模型通过 ToolMessage 的 `content` 理解执行结果。

## 5. State 与 Reducer

`AgentState` 是 Graph 在某一时刻的已提交状态：

```python
from operator import add


class AgentState(TypedDict, total=True):
    messages: Annotated[list[Message], add_messages]
    model_call_count: Annotated[int, add]
```

字段更新分为两类：

- 带 Reducer 的字段：通过 Reducer 将旧值与一个或多个 Update 合并。
- 不带 Reducer 的字段：本轮无写入时保持不变；只有一个写入时直接覆盖；多个 Node 同时写入时拒绝提交。

`messages` 使用 `add_messages`：

- 新 ID 追加消息。
- 已存在的 ID 替换对应消息。
- 合并前复制旧消息，避免破坏上一版 State，为后续 Checkpoint 保留正确语义。

`model_call_count` 使用标准库的 `operator.add`：

- ModelNode 每完成一次模型调用，返回增量 `1`，而不是返回旧值加一后的总数。
- Runtime 从 `Annotated` 中取得 `add`，并在提交 State Update 时主动调用它。
- 同一 Super-step 中多个模型 Node 可以分别返回增量，最终统一累加到旧值上。

Reducer 映射只由 `CompiledStateGraph` 持有。`Agent` 只构造 `input_update`，不解析 State Schema，也不直接调用底层 `apply_updates()`。

## 6. Graph 与 Transition

`StateGraph` 是声明 Graph 的 Builder，负责注册：

```text
Nodes
+ Fixed Edges
+ Conditional Edges
```

Builder 注册阶段与 `compile()` 阶段共同完成核心结构校验；`compile()` 最终将可变 Builder 编译为独立的 `CompiledStateGraph`：

- Graph 必须存在 `START` 入口。
- Node 不能同时拥有 Fixed Edge 与 Conditional Edge。
- Edge 端点必须合法。
- `path_map` 的目标必须是已注册 Node 或 `END`。
- 所有注册 Node 必须能够从 `START` 到达。

编译后，Fixed Edge 和 Conditional Edge 都通过统一接口工作：

```python
transition.resolve_targets(state)
```

条件转移支持两种核心语义：

```text
有 path_map：Router 返回 router_key，再映射到 target Node
无 path_map：Router 直接返回已注册 Node 名称或 END
```

## 7. Tool 参数与错误处理

Tool Definition 中的 `function.parameters` 是 JSON Schema。它既用于告诉模型如何生成参数，也用于 Runtime 在 Tool 执行前校验参数。

```text
arguments
→ jsonschema.validate()
→ 合法：执行 Tool
→ 非法：ToolInvocationException
```

当前参数校验覆盖：

- 缺少必填参数。
- 参数类型错误。
- 出现 Tool Schema 未声明的额外参数。

可恢复错误分为两类：

```text
ToolInvocationException
    模型生成的 Tool 参数不符合 Schema

ToolExecutionException
    Tool 执行期间可预期、可以反馈给模型的业务错误
```

具体业务异常继承 `ToolExecutionException`。例如：

```text
ToolExecutionException
└── ResumeNotFoundException
```

ToolNode 将这些可恢复异常转换成 `status="error"` 的 ToolMessage，保留原 `tool_call_id`，再让 Graph 回到 ModelNode。未声明为可恢复错误的程序 Bug 或系统故障继续向上抛出，不会被伪装成正常 Tool 结果。

## 8. Checkpoint 与会话 State

`Agent` 本身保存的是可共享的组件：

```text
Model
+ Tool Registry
+ Compiled Graph
+ System Prompt
```

Checkpoint 开启后，每次调用必须提供 `thread_id`。Runtime 先查询对应历史：

```text
没有历史Snapshot
→ 使用initial_state

存在历史Snapshot
→ 使用Snapshot.state，忽略传入的initial_state
```

若本次存在新的 `input_update`，Runtime 将其合并到 State 并从 `START` 开始新一轮执行；若 `input_update=None`，Runtime 直接从 Snapshot 的 `next_node_names` 恢复，不重复执行已经完成的 Node。

每个成功提交的 Super-step 生成一个 Snapshot：

```text
thread_id
checkpoint_id
parent_checkpoint_id
super_step
state
next_node_names
created_at
```

当前提供两种存储实现：

- `InMemoryCheckpointer`：用于快速调试和进程内恢复。
- `JsonlCheckpointer`：一行保存一个完整 Snapshot，支持进程重启后恢复。

JSONL 采用追加写入；`get/list` 逐行扫描，因此查询复杂度是 `O(n)`。当前版本面向单机小规模学习场景，不实现并发写锁、索引和文件压缩。

已通过最小恢复验证：ModelNode 完成后中断，恢复时只执行待执行的 ToolNode；再次恢复后继续 ModelNode，已提交的 Node 不会重复运行。

## 9. 运行方式

当前运行环境需要：

```text
Python 3.11
PyTorch
Transformers
jsonschema
```

进入项目目录：

```bash
cd handwritten-agent
```

激活环境：

```bash
conda activate ENV_agent
```

确认本地模型位于：

```text
models/Qwen3-1.7B
```

运行：

```bash
python agent.py
```

输入：

```text
exit
```

结束交互。

代码默认使用 `device="mps"`。没有可用 MPS 的环境需要在 `agent.py` 中改为 `device="cpu"` 或其他可用设备。

## 10. 当前边界

当前版本有意不实现以下生产能力：

- 数据库、远程 Checkpointer、并发写入和分布式一致性。
- Pending writes 与异步持久化。
- 分布式执行与并发 Tool 调度。
- 服务化、鉴权、配额和 Tool 沙箱。
- 完整的自动化测试、评测和可观测性体系。

以下 Agent 核心能力将在下一阶段最小手写：

- Human-in-the-loop（HITL）。
- Long-term Memory。
- Streaming。
- Subgraph。
- Multi-Agent。

当前错误 ToolMessage 可以让模型纠正参数或向用户追问，但“用户补充参数后确定性恢复原 ToolCall”尚未实现。该能力需要保存 `pending_tool_call`、中断 Graph，并在获得人工输入后恢复执行，将在 Checkpoint + HITL 阶段统一实现。

## 11. 设计文档

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：完整架构设计与协议说明。
- [`ADVANCED_CORE_PLAN_2026-08-06.md`](ADVANCED_CORE_PLAN_2026-08-06.md)：Checkpoint、HITL、Memory、Streaming、Subgraph 和 Multi-Agent 计划。
- [`TODAY_PLAN_2026-08-04.md`](TODAY_PLAN_2026-08-04.md)：Graph Runtime 核心实现计划。

## 12. 项目定位

这不是生产级 Agent Framework，也不是 LangGraph 的源码复刻。

它是一个规模可控但运行语义完整的学习型 Runtime，用于直接观察并解释：

```text
模型如何提出 Tool 调用
Runtime 如何赋予调用身份
Tool Result 如何回到消息历史
State 如何按 Reducer 更新
Graph 如何按 Super-step 执行
Router 如何决定下一跳
错误如何转化为可恢复 Observation
Checkpoint 如何保存并恢复 State 与下一批 Nodes
```

完成高级核心能力后，再使用 LangGraph 重构同一业务流程，对照理解框架为这些底层机制提供的抽象。

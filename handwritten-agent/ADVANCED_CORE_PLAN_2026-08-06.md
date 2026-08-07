# 手写 Agent 第二阶段核心能力计划

## 1. 阶段目标

第一阶段已经建立主流程：

```text
StateGraph
→ compile()
→ CompiledStateGraph.invoke()
→ Plan
→ Execute
→ Update
→ Route
→ 下一Super-step
→ END
```

第二阶段不重写这套 Runtime，而是在现有 Super-step 边界、State Update、Compiled Graph 和会话隔离之上，最小手写以下六项主流能力：

```text
Checkpoint
+ Human-in-the-loop（HITL）
+ Memory
+ Streaming
+ Subgraph
+ Multi-Agent
```

目标仍然是“麻雀虽小，五脏俱全”：实现每项能力不可缺少的运行语义和接口，不扩展到数据库、分布式调度或生产服务化。

## 2. 依赖顺序

```text
现有Graph Runtime
├──→ Checkpoint ──→ HITL
├──→ Streaming
├──→ Memory
└──→ Subgraph ──→ Multi-Agent
```

依赖原因：

- Checkpoint 保存可恢复的 Super-step 边界，HITL 才能真正暂停并恢复。
- Streaming 复用 Runtime 已有的 Plan、Execute、Update、Route 生命周期。
- Memory 与单次 Run State 分离，通过用户或会话命名空间共享长期信息。
- Subgraph 让一个 Compiled Graph 成为父 Graph 的可执行 Node，是 Multi-Agent 组合的基础。

## 3. Checkpoint

**状态：已于 2026-08-07 完成最小手写闭环。**

### 3.1 要理解的核心概念

Checkpoint 不是简单保存 `messages`，而是保存某个 Super-step 边界上能够继续执行 Graph 的运行快照。

最小 `StateSnapshot` 至少包含：

```text
thread_id
checkpoint_id
parent_checkpoint_id
super_step
state
next_node_names
created_at
```

其中：

- `state`：已提交 Reducer 后的 State。
- `next_node_names`：恢复时下一批需要执行的 Nodes。
- `super_step`：当前执行位置。
- `thread_id`：隔离不同会话。
- `checkpoint_id`：标识同一会话中的具体版本。
- `parent_checkpoint_id`：记录快照血缘，支持后续历史分支。
- `created_at`：记录创建时间，并用于历史排序。

### 3.2 最小实现

- 定义 `StateSnapshot` 协议。
- 定义 `Checkpointer` 接口。
- 实现单进程 `InMemoryCheckpointer`。
- 实现追加写入的本地 `JsonlCheckpointer`，支持进程重启恢复。
- 按 `thread_id` 隔离快照。
- 在 Super-step 提交完成后保存快照。
- 支持读取最新快照并继续执行。
- 支持读取指定 checkpoint，为后续 replay/time-travel 保留入口。

### 3.3 验收场景

```text
执行ModelNode
→ 保存Checkpoint
→ 模拟Run停止或进程重启
→ 根据thread_id读取Snapshot
→ 从next_node_names继续
→ 正常到达END
```

实际验收覆盖 `ModelNode → ToolNode → ModelNode` 分段恢复，确认已完成 Node 不会重复执行，最终 Snapshot 的 `next_node_names=()`。

### 3.4 暂不实现

- 数据库和远程 Checkpointer。
- Pending writes、异步持久化与并发文件写入。
- 分布式锁与跨进程一致性。
- Checkpoint 压缩、清理和迁移。

## 4. Human-in-the-loop（HITL）

### 4.1 要理解的核心概念

HITL 不是在 Node 内调用 `input()`，而是：

```text
Runtime到达中断点
→ 保存Checkpoint
→ 返回Interrupted结果
→ 外部系统获得人工输入
→ 使用thread_id和resume值恢复Graph
```

### 4.2 最小实现

- 支持 `interrupt_before` 或明确的 Interrupt Node。
- 中断时保存 `StateSnapshot`。
- 返回结构化中断信息，而不是阻塞 Runtime 等待终端输入。
- 支持携带 `resume_value` 恢复。
- 恢复后不重复执行已经提交完成的 Super-step。

### 4.3 验收场景

```text
Model提出调用高风险Tool
→ Runtime在ToolNode前暂停
→ 人工批准
→ 从Checkpoint恢复
→ ToolNode只执行一次
→ Graph继续到END
```

### 4.4 暂不实现

- Web 审批页面。
- 多级审批流。
- 审批超时和消息通知。

## 5. Memory

### 5.1 要理解的核心概念

必须区分：

```text
Checkpoint / Short-term State：同一thread的运行历史与恢复位置
Long-term Memory：跨thread、按用户命名空间保存的长期信息
```

Memory 不直接塞进全局变量，也不与 `AgentState` 生命周期混为一谈。

### 5.2 最小实现

- 定义 `Store` 接口。
- 实现 `InMemoryStore`。
- 使用 namespace 隔离数据，例如 `(user_id, "profile")`。
- 支持 `put`、`get`、`search/list`。
- 通过 Runtime Context 或明确的 Memory Tool 读写 Store。
- 演示同一用户跨 thread 读取长期信息，以及不同用户之间的隔离。

### 5.3 验收场景

```text
thread_A保存用户偏好
→ thread_A结束
→ 同一user_id创建thread_B
→ thread_B读取该偏好
→ 其他user_id无法读取
```

### 5.4 暂不实现

- 向量数据库和复杂语义检索。
- 自动记忆提取与遗忘策略。
- 隐私合规和数据生命周期平台。

## 6. Streaming

### 6.1 要理解的核心概念

Streaming 不是在代码里随意 `print()`，而是 Runtime 对外持续产生结构化事件。

最小事件生命周期：

```text
super_step_start
node_start
node_end
state_update
route
super_step_end
graph_end / graph_error
```

### 6.2 最小实现

- 增加 `stream()` 接口。
- `stream()` 复用同一套 Runtime 逻辑，不能复制一份独立执行循环。
- 至少支持 Node Update 事件和最终 State。
- `invoke()` 可以消费统一执行器的事件并返回最终 State。
- 为以后 token streaming 保留事件类型，但本阶段不手写模型逐 token 解码。

### 6.3 验收场景

执行一次 Model → Tool → Model 流程，调用者能够按顺序收到各 Node 的开始、结果、State Update、Route 和 Graph 结束事件。

### 6.4 暂不实现

- WebSocket/SSE 服务。
- 多机事件总线。
- 模型逐 token 回压控制。

## 7. Subgraph

### 7.1 要理解的核心概念

Subgraph 的核心是：一个 `CompiledStateGraph` 可以被父 Graph 当作 Node 调用，但父子 Graph 的 State 边界必须明确。

```text
Parent State
→ 输入映射
→ Child Graph.invoke()
→ Child State
→ 输出映射
→ Parent State Update
```

### 7.2 最小实现

- 允许 Compiled Graph 作为可调用 Node。
- 明确父子 State 相同和不同两种情况中的最小一种；优先实现显式输入/输出映射。
- 子图内部保持自己的 Node、Edge、Router 和 recursion limit。
- 子图返回父 Graph 可提交的 State Update。
- 为 Checkpoint namespace 保留父子层级信息。

### 7.3 验收场景

```text
父Graph
→ Resume Analysis Subgraph
→ 返回结构化分析Update
→ 父Graph继续生成最终回答
```

### 7.4 暂不实现

- 任意层级的复杂嵌套可视化。
- 分布式子图执行。
- 动态加载远程 Graph。

## 8. Multi-Agent

### 8.1 要理解的核心概念

Multi-Agent 不是创建多个 LLM 对象就结束，而是多个具有独立职责、Prompt、Tools 或 State 边界的 Agent，通过明确协议进行路由或交接。

最小采用 Supervisor 模式：

```text
Supervisor
├──→ Resume Agent
├──→ Evidence Agent
└──→ Final Answer
```

### 8.2 最小实现

- 每个子 Agent 使用独立的 Compiled Subgraph。
- Supervisor 通过结构化 route key 选择子 Agent。
- 明确共享 State 字段和 Agent 私有字段。
- 子 Agent 返回结构化结果，不直接修改其他 Agent 的私有状态。
- 至少完成一次 Supervisor → 子 Agent → Supervisor 的交接。

### 8.3 验收场景

```text
用户提出简历证据问题
→ Supervisor选择Resume/Evidence Agent
→ 子Agent完成任务并返回结构化结果
→ Supervisor决定是否继续委派
→ 输出最终答案
```

### 8.4 暂不实现

- Agent 自由生成和注册新 Agent。
- 分布式 Agent 网络。
- 复杂角色社会和长期自治任务。

## 9. 建议进度

### 2026-08-06

```text
Checkpoint
→ StateSnapshot
→ InMemoryCheckpointer
→ 中断恢复基础
→ HITL最小闭环
```

### 2026-08-07

```text
Memory Store
→ user/thread命名空间
→ Streaming事件协议
→ stream()最小闭环
```

### 2026-08-08

```text
Subgraph
→ 父子State映射
→ CompiledGraph作为Node
```

### 2026-08-09

```text
Multi-Agent Supervisor
→ 子Agent委派
→ 六项能力联调
```

如果概念讨论较多，2026-08-10 作为联调缓冲，不压缩关键理解过程。

## 10. 第二阶段完成标准

第二阶段完成后，应能清楚解释并演示：

```text
为什么Checkpoint必须保存next nodes
为什么HITL依赖Checkpoint
Checkpoint与Long-term Memory的区别
Streaming如何复用Runtime而不是复制循环
Subgraph如何处理父子State边界
Multi-Agent如何通过Supervisor和结构化协议协作
```

代码层面至少完成：

- 六项能力各自的最小可运行实现。
- 不同 `thread_id` 和 `user_id` 的数据隔离。
- 中断后不重复执行已完成 Node。
- Streaming 与 `invoke()` 使用同一执行语义。
- Subgraph 和 Multi-Agent 不绕开现有 State/Reducer/Runtime。
- 所有扩展都能够映射到 LangGraph 对应概念。

## 11. 明确非目标

本阶段不实现：

- 数据库、Redis、消息队列。
- 真正异步并发、分布式执行和任务调度平台。
- 完整生产级权限、安全、可观测性和成本控制。
- Web 服务、前端和部署。
- 对 LangGraph 内部源码逐行复刻。

这些属于后续工程化能力，不影响本阶段建立完整 Agent 核心认知。

## 12. 主流概念映射

| 手写能力 | LangGraph 对应概念 |
|---|---|
| `StateSnapshot` | State Snapshot |
| `InMemoryCheckpointer` | Checkpointer / In-memory Saver |
| `JsonlCheckpointer` | 本地持久化 Checkpointer |
| Interrupted Result + Resume | Interrupt / `Command(resume=...)` |
| `InMemoryStore` | Store / Long-term Memory |
| Runtime Events | `stream()` / Stream Modes |
| Compiled Graph as Node | Subgraph |
| Supervisor + Child Graphs | Multi-Agent / Handoffs |

参考：

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.langchain.com/oss/python/langgraph/memory
- https://docs.langchain.com/oss/python/langgraph/streaming
- https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- https://docs.langchain.com/oss/python/langchain/multi-agent

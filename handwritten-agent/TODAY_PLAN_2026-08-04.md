# 2026-08-04：Graph、Edge、Router 与 Runtime Loop

## 1. 今日核心目标

今天把昨天写死在 `agent.py` 中的：

```text
ModelNode
→ ToolNode
→ ModelNode
```

升级为由 Graph 驱动的通用 Agent Runtime：

```text
Initial State
→ START
→ Node
→ AgentStateUpdate
→ Reducer提交State
→ Edge / Router选择下一批Node
→ 下一Super-step
→ END
```

今天完成后，Model 与 Tool 之间可以循环任意轮数，不再由业务代码提前写死调用次数。

## 2. 对齐的主流 LangGraph 结构

今天严格保持下面三层边界：

```text
StateGraph Builder
→ compile()
→ Compiled Graph.invoke(initial_state)
```

三层职责：

### 2.1 StateGraph Builder

只负责声明 Graph：

- State Schema。
- Node Registry。
- Fixed Edges。
- Conditional Edges。
- `START` 与 `END`。

Builder 不执行 Node，也不保存某次 Run 的动态 State。

### 2.2 compile()

负责检查并固化 Graph Definition：

- Node 名称是否唯一。
- Edge 引用的 Node 是否存在。
- 是否存在从 `START` 出发的入口。
- Conditional Edge 的目标是否合法。
- 是否存在无法从 `START` 到达的孤立 Node。
- 将 Fixed Edge 和 Conditional Edge 编译为 Runtime 可以统一解析的 Transition 定义。

编译得到的 Graph Definition 不在 `invoke()` 期间随意修改。

### 2.3 Compiled Graph / Runtime

负责一次具体 Run：

- 接收 Initial State。
- 维护当前激活的 Node。
- 执行 Super-step。
- 收集 `AgentStateUpdate`。
- 调用 `apply_updates()` 提交 State。
- 根据编译后的 Transition 选择下一批 Node。
- 到达 `END` 或执行步数上限时停止。

## 3. Task 1：定义 Graph 的基础协议

先明确今天要使用的控制对象：

```text
START：虚拟入口，不是可执行Node
END：虚拟终点，不是可执行Node
Node：AgentState -> AgentStateUpdate
Router：AgentState -> RouteKey | list[RouteKey]
```

Graph 至少保存：

```text
nodes
fixed_edges
conditional_edges
state_schema
```

其中：

- Node 名称是 Graph 内部的稳定标识。
- Node 对象是真正的可调用计算单元。
- Edge 只描述控制流，不更新 State。
- Router 只读取 State 并返回路由结果，不执行 Node。

## 4. Task 2：实现 Node Registry

实现主流 Builder 风格的 Node 注册能力：

```text
add_node(node_name, node)
```

要求：

- 拒绝重复 Node 名称。
- 拒绝使用 `START`、`END` 作为普通 Node 名称。
- Node 必须是 callable。
- Graph 只通过 Node 名称引用 Node，不在 Edge 中直接保存业务调用结果。

当前 Resume Agent 注册：

```text
model -> ModelNode
tools -> ToolNode
```

## 5. Task 3：实现 Fixed Edge

Fixed Edge 表示确定性转移：

```text
add_edge(source, target)
```

当前 Graph 使用：

```text
START -> model
tools -> model
```

要求：

- 同一个 Source 可以声明一个或多个静态目标。
- 多个目标属于下一 Super-step 的多个激活 Node，而不是在当前 Node 内依次调用。
- `END` 可以作为目标，但不能作为可执行 Node。
- Edge 只保存拓扑，不在 Builder 阶段执行。

## 6. Task 4：实现 Conditional Edge 与 Router

Conditional Edge 表示根据最新 State 动态选择目标：

```text
add_conditional_edges(source, router, path_map=None)
```

Router 的职责只有一个：

```text
读取State
→ 返回RouteKey
```

当前 `route_after_model` 的语义：

```text
最新AssistantMessage存在非空tool_calls
→ "tools"

否则
→ "end"
```

再由 `path_map` 将 RouteKey 映射为真实目标：

```text
"tools" -> tools
"end"   -> END
```

如果省略 `path_map`，Router 返回值本身必须就是合法 Node 名称或 `END`。

要求：

- Router 不调用 ModelNode 或 ToolNode。
- Router 不修改 State。
- Router 不返回 `AgentStateUpdate`。
- Router 可以选择一个或多个下一目标，为后续并行 Super-step 保留主流语义。
- 同一个 Source 选择 Fixed Edge 或 Conditional Edge 中的一种路由机制；二者混用会同时调度两组目标，`compile()` 将其视为冲突。
- Fixed Edge 和 Conditional Edge 在 compile 后实现统一的 Transition Resolver 接口；Runtime 不编写 Resume Agent 专属的 `if model ... elif tools ...`。
- `START` 也允许作为 Fixed Edge 或 Conditional Edge 的 Source；当前 Resume Agent 使用固定入口。

## 7. Task 5：实现 compile()

Builder 完成声明以后，通过 `compile()` 产生可执行 Graph。

`compile()` 至少完成：

1. 校验所有 Node 名称和 Edge 端点。
2. 校验 `START` 入口。
3. 校验 Conditional Edge 的 `path_map` 目标。
4. 拒绝同一个 Source 同时混用 Fixed Edge 和 Conditional Edge。
5. 拒绝从 `END` 出发的 Edge。
6. 检查从 `START` 出发的可达性。
7. 从 State Schema 中编译 Reducer 映射。
8. 复制并固化 Graph Definition，避免后续修改 Builder 影响已编译 Graph。
9. 返回具有 `invoke()` 能力的 Compiled Graph。

Builder 和 Compiled Graph 必须是两个不同生命周期的对象：

```text
Builder可以继续组织定义
Compiled Graph负责稳定执行
```

## 8. Task 6：实现 Super-step Runtime

`invoke()` 按主流 Pregel / Bulk Synchronous Parallel 思想组织每一轮：

```text
Plan
→ 确定本Super-step要执行的Node

Execute
→ 所有Node读取同一份本轮State Snapshot
→ 各自返回AgentStateUpdate
→ 执行期间不提交其他Node的Update

Update
→ 收集本轮所有Update
→ 一次性调用apply_updates()
→ 产生下一份State
```

然后根据完成 Node 的 Transition，确定下一 Super-step 的激活 Node。

必须保持：

- Node 不直接调用下一个 Node。
- Node 不直接应用 Reducer。
- 同一 Super-step 的多个 Node 看不到彼此尚未提交的 Update。
- 如果任一 Node 执行失败，本 Super-step 不提交部分 State Update。
- Reducer 字段可以接收同一 Super-step 的多个写入。
- 无 Reducer 字段的多写冲突继续由 `apply_updates()` 拒绝。

今天先实现同步执行器，但保留 Super-step 的批量语义。实际 Python 并发、异步执行和分布式调度不改变这一执行模型，后续单独实现。

## 9. Task 7：循环停止与执行上限

当前 Resume Agent 的循环：

```text
START -> model

model
  -> tools：存在tool_calls
  -> END：不存在tool_calls

tools -> model
```

Runtime 停止条件：

- 没有待执行 Node。
- 当前分支到达 `END`。
- 超过 `recursion_limit` / 最大 Super-step 数时明确报错。

今天只实现基础执行上限。完整错误分类、重复 Tool Call 检测、恢复策略与 Trace 放到明天。

## 10. Task 8：替换 agent.py 中的手写顺序

删除业务代码对执行顺序的控制：

```text
手动调用ModelNode
→ 手动判断tool_calls
→ 手动调用ToolNode
→ 手动再次调用ModelNode
```

改为：

```text
构建Graph
→ compile()
→ invoke(initial_state)
```

`agent.py` 只负责：

- 创建 Model、Registry 和 Nodes。
- 声明 Graph。
- 构造 Initial State。
- 调用 Compiled Graph。
- 读取 Final State。

它不再知道 Model 与 Tool 需要循环几次。

## 11. 今日验证路径

至少验证三条控制流：

### 11.1 无 Tool Call

```text
START -> model -> END
```

### 11.2 一轮 Tool Call

```text
START -> model -> tools -> model -> END
```

### 11.3 多轮 Tool Call

```text
START
→ model
→ tools
→ model
→ tools
→ model
→ END
```

如果一次 AssistantMessage 中包含多个 Tool Call，仍由一个 ToolNode 在本轮为每个 Call 产生对应 ToolMessage。

## 12. 今日完成标准

- Graph Definition 与 Runtime Execution 已分离。
- Builder 具备 `add_node`、`add_edge`、`add_conditional_edges`、`compile`。
- `START` 和 `END` 只作为虚拟控制节点。
- Router 是纯路由函数，不修改 State、不执行 Node。
- Fixed Edge 与 Conditional Edge 由统一的编译后 Transition 结构解析。
- Runtime 按 `Plan -> Execute -> Update` 完成 Super-step。
- Node Update 只在 Super-step 边界统一提交。
- Runtime 支持循环执行，不再限制一轮 Tool Call。
- Runtime 具有基础 `recursion_limit`。
- `agent.py` 不再硬编码 ModelNode 与 ToolNode 的调用顺序。
- 三条验证路径全部通过。

完成这些以后，主流 Tool-Calling Agent 的核心执行流程成立：

```text
State
+ Node
+ Reducer
+ Edge
+ Router
+ Super-step Runtime
+ Loop
```

## 13. 今天明确不混入的能力

以下能力属于主流 LangGraph，但有自己的独立运行语义，今天不与核心 Graph Loop 混写：

- Retry Policy、完整错误模型与 Trace：明天实现。
- Checkpoint 与 StateSnapshot：周四实现。
- HITL 与 `Command(resume=...)`：Checkpoint 完成后实现。
- Streaming：周六实现。
- Subgraph 与 Multi-Agent：周末实现。
- `Send` 动态 Map-Reduce、真实异步并发和分布式执行：不影响当前 ReAct Graph 核心闭环，后续按需要扩展。

今天的代码结构必须为这些能力保留边界，但不提前把它们塞进 Node 或 Router。

## 14. 主流方案依据

- LangGraph Graph API：State、Nodes、Edges、`START`、`END`、Conditional Edges、compile 与循环。
- LangGraph Pregel Runtime：`Plan -> Execution -> Update` 三阶段 Super-step。
- LangGraph StateGraph：Node 的核心协议是 `State -> Partial<State>`，State 字段通过 Reducer 合并同一 Super-step 的 Update。

参考：

- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/oss/python/langgraph/pregel
- https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_edge
- https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges

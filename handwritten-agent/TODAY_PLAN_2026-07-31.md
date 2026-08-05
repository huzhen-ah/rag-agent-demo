# 2026-07-31 手写 Agent Runtime 计划

## 1. 今晚目标

把当前集中在 `Agent.run()` 中的 Tool Calling 循环，重构为一个遵循主流 LangGraph Graph API 语义的最小状态图 Runtime：

```text
State
→ Node
→ Partial State Update
→ per-key Reducer
→ New State
→ Edge
→ Next Super-step
```

今晚完成的是状态图的核心执行机制，不通过增加无关类或业务字段制造“框架感”。

完成后应当能够清楚解释并运行：

```text
START
→ model
→ conditional edge
   ├── tools
   │    └── fixed edge → model
   └── END
```

## 2. 与当前主流 LangGraph 方案核对后的结论

### 2.1 State

State 是 Node 之间共享的当前应用数据。

State Schema 定义：

- State 中允许存在哪些字段。
- 每个字段的数据类型。
- 每个字段收到新值时使用什么 Reducer。

Node 的标准接口是：

```text
State → Partial<State>
```

Node 读取当前完整 State，只返回本次需要写入的局部字段，不直接原地修改共享 State。

### 2.2 State Update

State Update 是 Node 本轮产生的局部写入，不是完整的新 State：

```python
{
    "messages": [new_message],
    "model_steps": 1,
}
```

未出现在 Update 中的字段保持原值。

### 2.3 Reducer

每个 State 字段拥有独立的更新规则：

```text
new_value = reducer(old_value, update_value)
```

规则：

- 显式指定 Reducer 的字段按照 Reducer 合并。
- 未指定 Reducer 的字段使用默认覆盖语义。
- Reducer 由 Runtime 调用，Node 不自行合并共享 State。

`messages` 不能简单无条件拼接；应实现与 `add_messages` 相同的核心语义：

- 新消息 ID 不存在时追加。
- 新消息 ID 已存在时替换对应消息。
- 保持其他消息原有顺序。

### 2.4 Edge 与 Router

Graph 中声明两类控制流：

```text
Fixed Edge
Conditional Edge
```

Fixed Edge直接声明目标：

```text
tools → model
```

Conditional Edge持有一个路由函数：

```text
model → route_after_model(state) → tools / END
```

Router不是与Edge并列的Runtime组件，而是Conditional Edge用于解析目标Node的函数。

Runtime不写业务判断：

```python
if current_node == "model":
    ...
```

Runtime只查询Graph中当前Node的出边，并让出边解析下一批Node。

### 2.5 Super-step

Super-step是Graph Runtime的一轮调度周期：

```text
读取本轮State
→ 执行本轮被调度的Node
→ 收集Node Updates
→ Reducer统一合并
→ 产生下一版State
→ 解析出边
→ 产生下一轮任务
```

同一个Super-step中的并行Node读取同一版State，它们的Updates在本轮边界统一合并。

今晚实现串行Agent图，但Runtime的数据结构保留“本轮任务列表”和“本轮Updates列表”，避免把一个Super-step错误地写死成只能有一个Node。

### 2.6 Graph结束条件

核心 `AgentState` 不增加通用 `status` 字段。

Graph是否继续由Runtime任务状态表达：

```text
存在next nodes → 继续执行
next nodes为空 → Graph完成
Node异常       → 本轮失败
超过步数限制   → Runtime失败
```

如果未来业务本身需要 `approval_status`、`application_status` 等字段，再把它们作为业务状态加入State。

### 2.7 State中今晚保存什么

核心State只保存Node间共享且会影响后续计算的数据：

```python
class AgentState(TypedDict):
    messages: Annotated[list[Message], add_messages]
    model_steps: int
```

Tool Call已经包含在Assistant Message中，Tool Result也进入Messages，因此不额外维护一份重复的 `pending_tool_calls`。

执行Trace由Runtime记录，不塞入Graph State：

```text
Graph State → Node之间共享的应用数据
Runtime Trace → Node执行、Edge解析、耗时和错误等运行事实
```

## 3. 今晚实现范围

### Task 1：定义内部消息协议

保留当前主流Tool Calling消息结构：

```text
system
user
assistant(content, tool_calls)
tool(tool_call_id, name, content)
```

内部Tool Call采用LangChain Core式扁平结构：

```python
class ToolCall(TypedDict, total=True):
    name: str
    args: dict[str, Any]
    id: str
    type: NotRequired[Literal["tool_call"]]
```

不定义内部 `FunctionCall`。Qwen原始输出中的 `arguments` 由Model Adapter标准化为内部 `args`；写回Qwen Chat Template时再执行反向转换。

要求：

- 每条Message有稳定ID。
- 每个Tool Call有独立 `id`。
- Tool Result通过 `tool_call_id` 与Tool Call关联。
- 不把Qwen私有文本协议泄漏到Runtime核心结构。

### Task 2：定义AgentState和State Update

建立：

```python
class AgentState(TypedDict):
    messages: Annotated[list[Message], add_messages]
    model_steps: int


class AgentStateUpdate(TypedDict, total=False):
    messages: list[Message]
    model_steps: int
```

明确：

- `AgentState`表示当前完整值。
- `AgentStateUpdate`表示Node本轮局部写入。
- State中不放模型实例、Registry、System Prompt或Runtime Trace。

### Task 3：实现Reducer机制

实现：

```python
apply_updates(
    state,
    updates,
    state_schema,
) -> AgentState
```

要求：

- 从State Schema读取字段类型与Reducer元数据。
- 每个字段独立归并。
- 无显式Reducer的字段默认覆盖。
- 输入State不原地修改。
- 返回下一版State。
- Update包含未知字段时明确报错。
- 同一Super-step中多个Update写入无Reducer字段时明确报冲突，不能依赖覆盖顺序。

### Task 4：拆出Model Node

Model Node负责：

```text
读取State.messages
→ 调用Model Adapter
→ 得到标准Model Response
→ 构造Assistant Message
→ 返回AgentStateUpdate
```

Model Node不负责：

- 执行Tool。
- 选择下一个Node。
- 修改Graph运行状态。
- 操作完整State对象。

### Task 5：拆出Tool Node

Tool Node负责：

```text
读取最后一条Assistant Message中的Tool Calls
→ Registry查找Tool
→ 执行Tool
→ 构造对应Tool Messages
→ 返回AgentStateUpdate
```

Tool Node不负责：

- 再次调用模型。
- 选择下一个Node。
- 控制Runtime循环。

今晚多个Tool Call先串行执行，但必须依靠 `tool_call_id` 关联结果，不能依赖返回顺序表达身份。

### Task 6：实现Graph Definition

定义Graph数据结构和构建接口：

```python
graph.add_node("model", model_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")
graph.add_conditional_edge(
    "model",
    route_after_model,
    {
        "tools": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "model")
```

Graph负责：

- 保存Node。
- 保存Fixed Edges。
- 保存Conditional Edges及其Router和path map。
- 校验Node名称和Edge目标。
- 根据已完成Node及新State解析下一批Node。

### Task 7：实现Super-step Runtime

Runtime主循环：

```text
接收Input Update
→ Reducer生成初始State
→ 从START解析首批任务
→ 执行当前Super-step的Node
→ 收集所有State Updates
→ Reducer统一合并
→ 生成下一版State
→ 解析已完成Node的所有出边
→ 生成下一Super-step任务
→ next为空时结束
```

Runtime必须保持通用：

- 不写 `model/tools` 业务分支。
- 不从Node名称推断下一步。
- 不允许Node直接修改共享State。
- 设置Super-step上限，避免Graph无限循环。
- 返回最终State和Runtime Trace。

## 4. 今晚暂不实现

以下能力今晚不提前混入核心状态图：

- Checkpoint与StateSnapshot。
- HITL与Interrupt。
- 长期Memory Store。
- `Command(update=..., goto=...)`。
- Subgraph。
- 多Agent。
- Tool并行执行。
- 持久化Trace。

这些能力依赖今晚的State、Reducer、Edge和Super-step Runtime，但不影响今晚按主流方式完成核心Graph执行。

## 5. 验收场景

### 场景A：普通回答

```text
START
→ model
→ ModelUpdate
→ Reducer
→ New State
→ Conditional Edge
→ END
```

要求：

- 不调用Tool。
- 最终答案存在于最后一条Assistant Message。
- Runtime的next tasks为空。

### 场景B：单Tool闭环

```text
START
→ model
→ Reducer
→ conditional edge
→ tools
→ Reducer
→ fixed edge
→ model
→ Reducer
→ conditional edge
→ END
```

要求：

- Tool Result与Tool Call ID正确关联。
- Model第二次调用能够看到Tool Result。
- Runtime没有针对Tool名称或业务流程写硬编码分支。

### 场景C：同轮多个Tool Call

要求：

- Tool Node处理同一Assistant Message中的全部Tool Calls。
- 每个Tool Result使用正确的 `tool_call_id`。
- 所有Tool Messages通过messages Reducer合并进入新State。

## 6. 今晚完成标准

- 能解释 `State → Update → Reducer → New State`。
- 能解释Fixed Edge、Conditional Edge和Router的关系。
- 能解释一个Super-step的输入、执行、归并和输出。
- Node只返回局部Update。
- Reducer统一生成新State，不原地修改输入State。
- Runtime不包含 `model/tools` 业务路由判断。
- Graph通过声明的Edges产生下一批任务。
- 普通回答、单Tool、多Tool三个场景能够运行。

## 7. 官方依据

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Use the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [StateGraph.add_edge](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_edge)
- [StateGraph.add_conditional_edges](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

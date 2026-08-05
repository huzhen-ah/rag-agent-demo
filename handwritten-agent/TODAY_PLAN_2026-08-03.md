# 2026-08-03 下午：State Update 与 Reducer Runtime

## 1. 上午进度

`state.py` 已经完成第一轮理解和实现：

- 定义 `ToolCall` 与四种 Message。
- 定义完整状态 `AgentState`。
- 定义局部写入 `AgentStateUpdate`。
- 使用 `Annotated[list[Message], add_messages]` 为 `messages` 声明 Reducer。
- `add_messages` 已具备按消息 ID 替换和追加的核心语义。

当前只完成了 State Schema 和一个具体 Reducer。下午开始实现真正消费这些声明的 State Runtime。

## 2. 下午目标

打通下面这条主流状态更新链路：

```text
AgentState
→ Node读取当前State
→ Node返回AgentStateUpdate
→ Runtime读取State Schema
→ 按字段选择Reducer或默认更新规则
→ 提交新的AgentState
```

下午不继续堆 Message 类型，也不进入 Edge、Router 和完整 Graph 循环。先把 `State → Update → Reducer → New State` 做扎实。

## 3. Task 1：读取 State Schema

理解并实现 Runtime 如何从 `AgentState` 中获得字段规则：

```python
get_type_hints(AgentState, include_extras=True)
```

需要掌握：

- 如何取得 `messages` 和 `model_steps` 的字段声明。
- 如何通过 `get_origin()`、`get_args()` 识别 `Annotated`。
- 如何从 `Annotated[list[Message], add_messages]` 中取得 `add_messages`。
- 如何判断某个字段没有显式 Reducer。

完成结果：Runtime 能把 Schema 编译成类似下面的字段规则：

```python
{
    "messages": add_messages,
    "model_steps": None,
}
```

## 4. Task 2：实现 `apply_updates`

实现统一的状态提交函数：

```python
apply_updates(
    state: AgentState,
    updates: list[AgentStateUpdate],
) -> AgentState
```

规则：

1. Node 只提交局部 Update，不直接修改共享 State。
2. Update 没有写入的字段保持原值。
3. 有显式 Reducer 的字段由 Runtime 调用 Reducer 合并。
4. 没有 Reducer 的字段采用覆盖语义。
5. 同一 Super-step 中，多个 Update 同时写入无 Reducer 字段时明确报冲突。
6. Update 出现 Schema 中不存在的字段时明确报错。
7. 只有所有字段都合并成功后，才返回新的完整 State。

示例：

```python
state = {
    "messages": [user_message],
    "model_steps": 0,
}

updates = [
    {
        "messages": [assistant_message],
        "model_steps": 1,
    }
]
```

提交结果：

```python
new_state = {
    "messages": [user_message, assistant_message],
    "model_steps": 1,
}
```

这里 `messages` 经过 `add_messages`，`model_steps` 直接采用新值。

## 5. Task 3：建立 Node 标准接口

明确手写 Runtime 中 Node 的统一协议：

```text
Node: AgentState → AgentStateUpdate
```

Node 的职责：

- 读取本轮完整 State。
- 完成本 Node 的计算。
- 只返回本轮产生的局部 Update。

Node 不负责：

- 调用 Reducer。
- 构造下一份完整 State。
- 选择下一个 Node。
- 控制 Runtime 循环。

先写一个不依赖模型的最小 Node，验证 Node 返回的 Update 可以被 `apply_updates` 正确提交。随后再把当前模型调用逻辑迁入 Model Node。

## 6. Task 4：开始拆 Model Node

如果前三项顺利完成，下午继续拆出 Model Node：

```text
读取 state["messages"]
→ 调用 Model Adapter
→ 得到标准化 Model Response
→ 构造 AssistantMessage
→ 返回 messages 与 model_steps 的局部 Update
```

Model Node 不执行 Tool，也不决定下一跳。

这一步下午至少完成接口和数据流；与现有 `model.py`、`parser.py` 的具体对接可以延续到晚上。

## 7. 下午完成标准

- 能从 `AgentState` 的 `Annotated` 元数据中取出 Reducer。
- `apply_updates` 能统一处理 Reducer 字段和默认覆盖字段。
- 多个 Update 写入无 Reducer字段时不会靠执行顺序偷偷覆盖。
- Node 只返回 `AgentStateUpdate`，不修改输入 State。
- 能完整解释一次 `State → Node → Update → Reducer → New State`。
- 如果进度正常，Model Node 的边界和输入输出已经落到代码中。

完成这些之后，下一阶段才进入：

```text
Graph Definition
→ Fixed Edge / Conditional Edge
→ Router
→ Super-step Runtime
```

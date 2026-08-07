# 手写 Tool-Calling Agent 架构设计

## 1. 文档目标

本项目不使用 LangChain、LangGraph 或其他 Agent 编排框架，使用原生 Python 手写一个小而完整的 Tool-Calling Agent。

设计目标不是复刻整个 LangGraph，而是实现主流 Agent 架构中最核心、可以独立解释和验证的部分：

```text
Model Adapter
+ Message Protocol
+ Response Parsing and Normalization
+ Tool System
+ Typed State
+ State Update Rules
+ Nodes
+ Conditional Routing
+ Runtime
+ Error and Stop Policies
+ Trace
+ Business Workflow
```

最终手写版应当满足：

- 结构与主流 Agent Runtime 的核心概念对应。
- 每个组件职责明确，可以单独替换和测试。
- 支持普通回答、单工具、多工具和多步工具调用。
- 确定性程序控制执行、校验、错误处理和停止条件。
- 不用复杂框架掩盖模型、工具、状态和循环之间的数据流。

## 2. 设计原则

### 2.1 LLM 只提出决策，不直接执行动作

裸模型的本质是 Next Token Generator。它可以生成：

```json
{
  "name": "read_resume",
  "arguments": {
    "resume_id": "main"
  }
}
```

这只是调用意图，不代表 Python 函数已经执行。

Agent Runtime 负责：

```text
解析意图
→ 标准化Tool Call
→ 生成调用ID
→ 校验工具和参数
→ 调度执行
→ 关联Tool Result
→ 更新State
→ 决定下一节点
```

### 2.2 确定性能力必须由程序控制

以下能力不得依赖模型自由生成：

- `tool_call_id` 的唯一性。
- Tool 白名单。
- 参数校验。
- Tool 实际执行。
- 岗位匹配分数计算。
- 错误计数。
- 重复调用检测。
- 最大执行步数。
- State 更新规则。

LLM 负责：

- 判断是否需要工具。
- 选择工具。
- 构造调用参数。
- 从 JD 中提取语义要求。
- 根据真实证据判断匹配状态。
- 组织最终自然语言回答。

### 2.3 内部标准协议与模型私有协议分离

Agent 内部使用统一的 Model Response、Tool Call、Tool Result 和 Message 结构。

内部 Tool Call 对齐 LangChain Core 的 provider-agnostic 结构：

```python
{
    "id": "call_8f31",
    "name": "read_resume",
    "args": {
        "resume_id": "main",
    },
}
```

内部协议不定义额外的 `FunctionCall` 嵌套。`function` 和 `arguments` 属于具体模型或服务商协议，由 Model Adapter 负责转换。

Qwen3 使用自己的 Chat Template：

- Tool Call 渲染为 `<tool_call>...</tool_call>`。
- Tool Result 渲染为 `<tool_response>...</tool_response>`。
- 模板读取 Tool Call 的 `name` 和 `arguments`；Model Adapter 将内部 `args` 转换为模板需要的 `arguments`。
- 模板读取 Tool Message 的 `content`。
- 当前模板不会把内部 `tool_call_id` 渲染给裸模型。

因此：

> `ToolCall.id` 是一次调用的内部身份；`ToolMessage.tool_call_id` 是指向它的关联字段。二者是否序列化给模型，由具体 Model Adapter 和 Chat Template 决定。

以后更换模型时，只更换协议适配逻辑，不修改 Agent Runtime 的核心循环。

### 2.4 必要步骤不完全依赖模型自觉

教学版允许模型调用 `calculate_job_fit`，用于完整学习 Tool Calling。

但对“完整岗位分析”而言，匹配分计算是必要的确定性步骤。后续正式 Workflow 可以将其设计为必经 Scoring Node：

```text
Model产生assessments
→ Runtime路由到Scoring Node
→ Python强制计算分数
→ Model生成最终说明
```

这体现主流 Agent 系统中的混合设计：

```text
Agentic Decision
+ Deterministic Workflow
```

## 3. 总体架构

```text
User Input
    ↓
Agent Runtime
    ↓
Model Node
    ↓
LocalChatModel
    ↓
Qwen3 Raw Text
    ↓
Response Parser
    ↓
Response Adapter
    ├── 无Tool Call → Final Route
    └── 有Tool Call → 为每个Call生成ID
                         ↓
                      Tool Node
                         ↓
               Registry查找与参数校验
                         ↓
                     Tool执行
                         ↓
                  Tool Result / Error
                         ↓
                    更新Agent State
                         ↓
                      Model Node
```

从 Graph 角度看：

```text
START
  ↓
MODEL
  ├── final answer ─────────→ END
  ├── tool calls ───────────→ TOOLS
  └── invalid response ─────→ ERROR_POLICY

TOOLS
  ├── completed/recoverable → MODEL
  └── fatal/limit reached ──→ END
```

## 4. 核心数据协议

### 4.1 Standard Model Response

```python
{
    "content": "我先读取你的简历。",
    "tool_calls": [
        {
            "id": "call_8f31",
            "name": "read_resume",
            "args": {
                "resume_id": "main",
            },
        }
    ],
    "raw_response": "<tool_call>...</tool_call>",
}
```

规则：

- `content` 始终为字符串，可以为空。
- `tool_calls` 始终为列表。
- 没有 Tool Call 时不生成 `ToolCall.id`。
- 每个 Tool Call 拥有独立 ID。
- `raw_response` 只用于调试和 Trace，不参与 Tool 执行。

### 4.2 Tool Call

```python
{
    "id": "call_8f31",
    "name": "read_resume",
    "args": {
        "resume_id": "main",
    },
}
```

字段语义：

- `id`：一次具体调用的唯一身份。
- `name`：要调用的 Tool 类型。
- `args`：该次调用的参数字典。

`tool_call_id` 由 Response Adapter 在解析出具体 Tool Call 后生成。

### 4.3 Tool Result

```python
{
    "tool_call_id": "call_8f31",
    "tool_name": "read_resume",
    "ok": True,
    "content": "简历正文……",
    "error": None,
    "metadata": {
        "duration_ms": 8,
    },
}
```

失败结果：

```python
{
    "tool_call_id": "call_8f31",
    "tool_name": "read_resume",
    "ok": False,
    "content": "",
    "error": {
        "type": "invalid_argument",
        "message": "resume_id不存在",
        "recoverable": True,
    },
    "metadata": {
        "duration_ms": 1,
    },
}
```

内部 Tool Result 使用统一结构。

写入 Qwen 消息历史时，Model Adapter 将其转换为当前 Chat Template 支持的 Tool Message：

```python
{
    "role": "tool",
    "content": "序列化后的Tool Result",
}
```

`tool_call_id` 仍保存在 State 和 Trace 中，不要求当前 Qwen 模板消费它。

### 4.4 Message

Agent 内部消息保持以下语义：

```text
system
→ user
→ assistant(content/tool_calls)
→ tool(tool result)
→ assistant(content/tool_calls)
→ ...
```

Assistant Tool Call Message：

```python
{
    "role": "assistant",
    "content": "我先读取简历。",
    "tool_calls": [
        {
            "id": "call_8f31",
            "name": "read_resume",
            "args": {
                "resume_id": "main",
            },
        }
    ],
}
```

Tool Message：

```python
{
    "role": "tool",
    "tool_call_id": "call_8f31",
    "name": "read_resume",
    "content": "序列化后的Tool Result",
}
```

内部结构保留完整关联字段；传给 Qwen 前允许由 Adapter 按模板要求裁剪。

## 5. Agent State

手写版使用轻量 Typed State：

```python
{
    "run_id": "run_1234",
    "messages": [],
    "step": 0,
    "trace": [],
    "consecutive_errors": 0,
    "call_signatures": {},
    "status": "running",
    "stop_reason": None,
    "final_answer": None,
}
```

字段职责：

- `run_id`：标识一次完整 Agent Run。
- `messages`：模型可见的短期对话状态。
- `step`：已执行的模型决策轮数。
- `trace`：Runtime 可观测事件。
- `consecutive_errors`：连续可恢复错误计数。
- `call_signatures`：检测重复 Tool Call。
- `status`：`running`、`completed` 或 `failed`。
- `stop_reason`：记录终止原因。
- `final_answer`：最终回答。

业务数据原则上通过 Tool Result 和 Messages 传递。

只有当后续确定性节点需要直接访问某类结构化业务数据时，才将其增加到 State，避免重复保存相同信息。

## 6. State Update 规则

State 不允许由任意组件随意整体替换。

使用明确更新操作：

```text
append_message
append_trace_event
increment_step
increment_error
reset_error_count
record_call_signature
complete_run
fail_run
```

关键规则：

- `messages` 只追加，不在运行中静默覆盖。
- `trace` 只追加。
- `step` 由 Runtime 在每次 Model Node 执行时递增。
- Tool 成功后重置连续错误数。
- 可恢复错误增加连续错误数。
- 只有 Runtime 可以设置 `status` 和 `stop_reason`。

这相当于手写版的 Reducer / State Update Semantics。

## 7. Nodes

### 7.1 Model Node

输入：

- 当前 Messages。
- Tool Definitions。

执行：

```text
序列化Qwen输入
→ 裸模型推理
→ Parser解析
→ Adapter标准化
→ 生成Tool Call ID
```

输出：

- Standard Model Response。
- Assistant Message State Update。

Model Node 不执行 Tool。

### 7.2 Tool Node

输入：

- Standard Tool Calls。
- Tool Registry。

对每个 Tool Call 执行：

```text
检查Tool是否注册
→ 校验args
→ 执行Tool
→ 捕获异常
→ 构造统一Tool Result
→ 写入Tool Message和Trace
```

第一版允许串行执行同轮多个 Tool Call，保留以后改为异步执行的结构。

Tool Node 不负责决定最终答案。

### 7.3 Error Policy

输入：

- 错误类型。
- 当前错误次数。
- 当前执行步数。
- 重复调用情况。

输出：

```text
retry_model
或 fail_run
```

可恢复错误转成 Observation 反馈给模型。

不可恢复错误安全终止，并在 State 中记录明确的 `stop_reason`。

### 7.4 Scoring Node

教学阶段：

- `calculate_job_fit` 保持为 Tool。

业务稳定阶段：

- 可升级为完整岗位分析的确定性必经节点。

评分公式始终由 Python 实现，LLM 不直接产生最终分数。

## 8. Routing

### 8.1 Model Node 后路由

```text
存在Tool Calls
→ Tool Node

不存在Tool Calls且content非空
→ Complete

不存在Tool Calls且content为空
→ Error Policy
```

### 8.2 Tool Node 后路由

```text
Tool执行成功
→ Model Node

Tool发生可恢复错误且未超过限制
→ 将错误Observation写入Messages
→ Model Node

连续错误超过限制
→ Fail

重复调用超过限制
→ Fail
```

### 8.3 全局停止条件

- 模型给出无 Tool Call 的最终回答。
- 达到 `max_steps`。
- 连续错误达到 `max_consecutive_errors`。
- 相同 Tool 名称和规范化参数重复超过限制。
- 模型返回空响应。
- 出现不可恢复的 Runtime 错误。

## 9. Tool System

### 9.1 Tool Definition

每个 Tool 必须具备：

- 唯一名称。
- 清晰描述。
- JSON Schema。
- Python Callable。
- 参数校验。
- 统一执行入口。

### 9.2 Registry

Registry 负责：

- 注册 Tool。
- 拒绝重复名称。
- 按名称查找 Tool。
- 导出 Tool Definitions。
- 提供允许的 Tool 名称。

Registry 是 Agent 的能力白名单。

### 9.3 参数校验

分为三层：

```text
Parser
校验Tool Call外层结构

Tool Wrapper
校验函数参数名称、必填字段和类型

Business Function
校验业务允许值和业务规则
```

Schema 只负责引导模型，不能替代 Runtime 校验。

### 9.4 当前业务 Tools

#### `read_resume`

受控读取指定简历。

#### `search_project_evidence`

根据岗位要求查询真实项目证据。

#### `calculate_job_fit`

根据结构化 assessments 使用固定公式计算匹配分。

## 10. Tool Call ID

### 10.1 产生位置

当前项目直接调用本地裸模型：

```text
裸模型生成name和arguments
→ Parser解析
→ Response Adapter将arguments标准化为内部args
→ Response Adapter确认存在Tool Call
→ Response Adapter为每个调用生成ID
```

因此 ID 不由裸模型负责。

如果未来部署独立 Model API：

```text
Model API调用裸模型
→ 服务端解析Tool Call
→ 服务端为每个Call生成ID
→ Agent沿用服务端ID
```

统一原则：

> 上游标准协议已经提供 ID 时原样沿用；上游没有提供时，由将调用意图实例化为具体 Tool Call 的确定性程序层生成。

### 10.2 使用位置

ID 用于：

- 关联 Tool Call 与 Tool Result。
- 区分同名 Tool 的多次调用。
- 区分失败调用和重试调用。
- Trace。
- 为以后异步 Tool 执行保留关联关系。

ID 不负责：

- 选择 Tool。
- 获取函数返回值。
- 判断 Tool 是否成功。
- 替代模型 API 的 Request ID。

## 11. Runtime

`Agent` 是当前手写项目的 Runtime / Orchestrator。

它负责：

```text
初始化State
→ 调用Model Node
→ 执行路由
→ 调用Tool Node
→ 应用State Update
→ 执行错误策略
→ 检查停止条件
→ 返回Run Result
```

Parser、Response Adapter、Registry、Tool 和 Model 都是 Runtime 使用的组件，不与 Agent Runtime 并列。

建议 Runtime 对外返回：

```python
{
    "run_id": "run_1234",
    "status": "completed",
    "stop_reason": "final_answer",
    "answer": "最终岗位分析……",
    "trace": [],
}
```

## 12. Trace

Trace 是 Runtime 事件，不是模型消息。

每次节点或 Tool 执行记录：

```python
{
    "run_id": "run_1234",
    "step": 1,
    "event": "tool_completed",
    "tool_call_id": "call_8f31",
    "tool_name": "read_resume",
    "status": "success",
    "duration_ms": 8,
    "error_type": None,
}
```

原则：

- Trace 保存执行事实。
- Messages 保存模型需要看到的上下文。
- 大段简历正文进入模型消息，但不完整复制进 Trace。
- Trace 中记录长度、摘要或统计信息。

## 13. 错误模型

最小错误分类：

```text
ModelGenerationError
ModelResponseParseError
UnknownToolError
ToolArgumentError
ToolExecutionError
AgentLimitError
RepeatedToolCallError
```

错误有三个不同出口：

```text
给模型
可修正、无敏感堆栈的Observation

给Runtime
结构化错误类型和recoverable标记

给开发者
Trace中的完整上下文
```

不得把未经处理的内部异常和堆栈直接作为最终用户回答。

## 14. 岗位分析业务闭环

目标流程：

```text
输入JD
→ 读取目标简历
→ 提取岗位核心要求
→ 为要求搜索项目证据
→ 形成结构化assessments
→ Python计算匹配分
→ 输出最终分析
```

最终答案必须包含：

- 岗位核心要求。
- 匹配项及真实证据。
- 部分匹配项。
- 缺失项和不确定项。
- 程序计算得到的匹配分。
- 是否建议投递。
- 简历修改建议。

模型不得：

- 编造简历经历。
- 编造项目证据。
- 自行替代评分公式。
- 在没有证据时把要求标记为完全匹配。

## 15. 与 LangGraph 的对应关系

| 手写版本 | LangGraph 概念 |
|---|---|
| `AgentState` | State Schema |
| State Update 函数 | Reducer / State Update |
| Model Node | Model Node |
| Tool Node | ToolNode |
| Routing 函数 | Conditional Edge |
| `Agent.run()` | Graph Runtime |
| `max_steps` | Step / Recursion Limit |
| Trace Event | Graph / Node / Tool Event |
| `StateSnapshot` + `Checkpointer` | Checkpointer / State Snapshot |
| 后续可选长期资料存储 | Store |

手写版的目的，是让以下数据流完全可见：

```text
State
→ Node
→ State Update
→ Router
→ Next Node
```

迁移到 LangGraph 时，不改变核心思想，只使用框架提供的标准 State、Node、Edge、Runtime 和持久化能力。

## 16. 当前实现范围

手写版必须实现：

- Model Adapter。
- Message Protocol。
- Response Parser。
- Response Adapter。
- 每个 Tool Call 的唯一 ID。
- Tool 抽象、Schema、校验和 Registry。
- 三个业务 Tool。
- Typed Run State。
- 明确 State Update 规则。
- Model Node。
- Tool Node。
- Conditional Routing。
- Runtime Loop。
- StateSnapshot、thread/checkpoint 标识和父子血缘。
- InMemory 与 JSONL Checkpointer。
- Super-step 后保存以及从 next nodes 恢复。
- 错误反馈和停止条件。
- Trace。
- 完整岗位分析闭环。

手写版暂不实现：

- 数据库、远程 Checkpointer 和分布式一致性。
- Pending writes 与异步持久化。
- 跨会话长期 Memory Store。
- Human-in-the-loop 中断恢复。
- 分布式节点执行。
- 复杂并行调度。
- Streaming API。
- Subgraph。
- 多 Agent。
- Web 服务和前端。

这些能力属于后续高级核心和工程化阶段，不影响当前手写版理解和呈现主流 Agent 核心架构。

## 17. 参考资料

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [LangChain Context](https://docs.langchain.com/oss/python/concepts/context)
- [Hugging Face Tool Use](https://huggingface.co/docs/transformers/en/chat_extras)
- [Hugging Face Chat Template Writing](https://huggingface.co/docs/transformers/chat_templating_writing)

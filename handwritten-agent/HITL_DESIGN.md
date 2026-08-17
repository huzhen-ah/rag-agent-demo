# HITL 设计方案

## 1. 目标与边界

本项目的 HITL 分为两层：

```text
通用 Runtime 能力
+
工具执行前的 HITL 业务编排
```

Runtime 不理解审核、补参或 Tool 策略，只负责：

```text
Task 执行
→ interrupt
→ pending writes
→ checkpoint 恢复
→ Command.resume
→ Task 从头重跑
```

工具 HITL 属于 Agent 编排层，负责根据 ToolCall 和策略决定是否向人工补参或请求审核。

## 2. 当前已完成的 Runtime 能力

- `interrupt(value)`、`Interrupt`、`GraphInterrupt`。
- `ContextVar`、`TaskContext`、`PregelScratchpad`。
- 稳定 `Task` 身份与 `TaskResult`。
- 基线 checkpoint。
- task-level pending writes：`update`、`interrupt`、`error`、`resume`。
- Super-step 中断时不提交 State。
- 恢复时复用已经成功的兄弟 Task。
- 同一个 Task 内按 interrupt 调用顺序匹配 resume 列表。
- 并行中断通过 `interrupt_id` 精确关联回复。
- `Command.resume` 只接受统一形式：

```python
Command(resume={interrupt_id: resume_value})
```

- `invoke()` 已区分普通 State 输入与 `Command.resume` 恢复输入。
- `Command.update`、`Command.goto`、`Command.graph` 暂未实现，传入时明确拒绝。

## 3. 工具 HITL 的 Graph 结构

采用外置 HITL，不把审批与补参写进 Tool function：

```text
START
  ↓
ModelNode
  ↓
router_after_model
  ├─ 无 tool_calls → END
  └─ 有 tool_calls → ToolArgsCompletionNode
                         ├─ 仍缺少必需参数 → ToolArgsCompletionNode
                         └─ 参数完整 → ToolReviewNode
                                            ↓
                                         ToolNode
                                            ↓
                                         ModelNode
```

保留 `router_after_model`。它只判断有没有 `tool_calls`，不处理 HITL 策略，也不调用 `interrupt()`。

`ToolArgsCompletionNode` 和 `ToolReviewNode` 都是实际 Node。前者负责补齐必需参数，后者负责审核最终 ToolCall；它们会读取 ToolCall、查询策略、按需调用 `interrupt()` 并返回 State Update。

`ToolNode` 只执行已经完成必要 HITL 处理的 ToolCall。

## 4. 为什么使用 HITL Node

Router 的职责是：

```text
读取 State → 返回下一执行目标
```

HITL 执行步骤需要：

```text
调用 interrupt()
→ 等待恢复
→ 解释结构化回复
→ 修改 ToolCall 或生成 ToolMessage
```

因此 HITL 应由 Node 承担，而不是塞进 Router。

该方案对应主流高层 Agent 的 `after_model` HITL middleware：模型产生 ToolCall 后、Tool 执行前，按 Tool 名称和策略决定是否中断。手撕版将 middleware 内部流程显式实现为 `ToolArgsCompletionNode` 和 `ToolReviewNode`。

## 5. HITL policy

Tool 不知道 HITL，底层 Runtime 也不知道业务策略。Agent 编排层维护 Tool 与 HITL 行为的映射：

```python
tool_hitl_policy = {
    "search": (),
    "read_resume": ("args_completion",),
    "delete_file": ("review",),
    "send_email": ("args_completion", "review"),
}
```

职责划分：

```text
Tool：声明参数 schema，执行实际功能
HITL policy：声明 Tool 需要哪些人工干预行为及顺序
ToolArgsCompletionNode：检查并补齐必需参数
ToolReviewNode：审核最终 ToolCall
ToolNode：执行最终 ToolCall
Runtime：执行 Node、暂停和恢复
```

一个 ToolCall 同时需要补参与审核时，顺序固定为：

```text
先补全参数 → 再审核最终参数 → 执行 Tool
```

## 6. 两类工具 HITL Node

### 6.1 args_completion

根据 Tool schema 检查 ToolCall 当前参数。只有配置了 `args_completion` 且确实缺少必需参数时，才请求人工补充。

interrupt payload 至少包含：

```text
行为类型
ToolCall ID
Tool 名称
当前参数
缺失参数
```

恢复后，将补充参数合并到对应 ToolCall 的 `args` 中。

一次反馈可以只补充部分参数。Node 返回更新后的 AssistantMessage，Runtime 通过 Reducer 替换原消息；随后 Router 根据最新 State 再次检查。只要仍有必需参数缺失，就重新进入 `ToolArgsCompletionNode` 并产生下一次中断。

### 6.2 review

向人工展示最终即将执行的 ToolCall。最小支持：

```text
approve：保留 ToolCall
edit：修改 ToolCall 参数后执行
reject：不执行该调用，并生成匹配 tool_call_id 的失败 ToolMessage
```

这三种决定只属于工具审核行为，不代表通用 HITL 只有这三种形式。

## 7. 多个 ToolCall

同一条 AssistantMessage 中的多个 ToolCall 视为同一批，默认彼此独立。当前手撕版采用：

```text
批量检查全部 ToolCall
→ 完成所有必要补参
→ 完成所有必要审核
→ ToolNode 批量执行允许执行的 ToolCall
```

如果两个 ToolCall 存在执行结果依赖，它们不应由模型放在同一批中，而应拆成多个 Model/Tool 轮次。

两个 HITL Node 都可以用一次结构化 interrupt 收集一批同类反馈；`resume_value` 内部按 `tool_call_id` 区分不同调用。

## 8. 自然语言反馈与 Runtime 的边界

用户不会保证直接提供标准结构化回复。例如：

```text
“你妈的，是李世民啊，赶紧去查查资料。”
```

Runtime 不负责理解这句话。调用 Runtime 前需要一个输入解释层：

```text
当前 Interrupt.value + 用户原话
→ 规则或 LLM 解析
→ 结构化 resume_value
→ Command(resume={interrupt_id: resume_value})
→ invoke()
```

简单情况可使用规则提取；复杂情况由 LLM 结合当前中断语义进行结构化解析。

## 9. Tool 内 interrupt 与外置 HITL

LangGraph 两种方式都支持：

- Tool 内调用 `interrupt()`：适合人工交互就是 Tool 固有语义的情况，例如 `ask_user`。
- Tool 外统一策略层：适合按 Tool 名称配置补参、审批和不同环境策略。

本项目采用外置方案：

```text
ToolArgsCompletionNode → ToolReviewNode → ToolNode → Tool function
```

这样 Tool function 不依赖 Graph、checkpoint、interrupt 或人工策略，便于复用和测试。

## 10. 当前完成状态

- Runtime 的 interrupt、pending writes、checkpoint 恢复和 `Command.resume` 已形成闭环。
- `ToolArgsCompletionNode` 支持批量检查、部分补参和循环补参。
- `ToolReviewNode` 支持 approve、edit 和 reject。
- `ToolNode` 会跳过已经由 reject 生成 ToolMessage 的 ToolCall，并执行其余调用。
- approve、edit 和 reject 已通过实际交互验证。
- 参数补全逻辑已完成静态审查；由于当前模型可能自行填写占位参数或直接追问，未增加依赖模型行为的端到端测试。

## 11. 暂不实现

- `Command.update`、`Command.goto`、`Command.graph`。
- Web 审批页面与自然语言解释层的生产实现。
- 审批超时、通知、多级权限与审计数据库。
- 将每个 ToolCall 动态拆成独立子图或 `Send` Task。
- 把 HITL 强耦合到 Tool function。

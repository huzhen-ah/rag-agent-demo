# 手写 Agent 目标

## 项目目标

使用原生 Python 手写一个最小但完整的 Tool-Calling Agent，用于理解：

```text
LLM决策
→ Tool调用
→ Observation
→ 状态更新
→ 继续循环或结束
```

业务场景是根据已有招聘 JD，结合个人简历和项目资料，分析岗位匹配程度并给出简历修改建议。

## 用户功能

用户提供一份已有 JD 后，Agent 输出：

- 岗位核心要求。
- 匹配项及真实证据。
- 缺失项和不确定项。
- 匹配分数。
- 是否建议投递。
- 简历修改建议。

手写版只提供修改建议，不生成或保存定制简历。

## Tools

### `read_resume`

读取本地主简历。

### `search_project_evidence`

从个人项目资料中检索与 JD 要求相关的真实证据。

### `calculate_job_fit`

按照固定规则计算匹配分数，避免由 LLM 随意生成分数。

## 需要手写的核心能力

- Tool 数据结构。
- Tool 注册表。
- Tool 描述与参数定义。
- Agent Prompt。
- JSON Action 解析。
- 消息和执行状态记录。
- 多步 Agent Loop。
- `final_answer` 结束动作。
- 最大执行步数。
- JSON 解析失败处理。
- Tool 不存在、参数错误和执行异常处理。
- Action 与 Observation 日志。

模型动作格式：

```json
{
  "action": "tool_name",
  "action_input": {}
}
```

最终回答格式：

```json
{
  "action": "final_answer",
  "action_input": {
    "answer": "最终分析结果"
  }
}
```

## 验收场景

### 普通对话

不调用 Tool，直接返回最终回答。

### 完整岗位分析

```text
读取简历
→ 检索项目证据
→ 计算匹配分
→ 返回分析结果
```

### 错误恢复

模型产生错误的 Tool 名称、参数或 JSON 时，Agent 能够返回错误信息并修正调用或安全结束。

## 不在手写版实现

- 招聘网站爬取。
- 自动投递。
- 自动生成和保存定制简历。
- 投递记录管理。
- 面试准备与复盘。
- Human-in-the-loop 中断恢复。
- Checkpoint 持久化。
- FastAPI 和前端。
- 定时文档同步。
- 多 Agent。
- LangGraph。

这些功能留给正式 LangGraph 版本。

## 完成标准

手写版本完成后，应能够清楚解释并演示：

```text
Agent
= LLM决策
+ Tool Calling
+ State
+ Loop
+ Error Handling
```

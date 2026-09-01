#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:32:58 2026

@author: huzhen
"""
from exceptions import ToolInvocationException, ToolExecutionException, GraphInterrupt
from hitl import _task_execution_context_var, Command
import uuid


class CompiledSubAgent:
    def __init__(self, name, description, runnable):
        self.name = name #subagent_type,就是哪个agent
        self.description = description #告诉supervisor agent什么时候用这个agent
        self.runnable = runnable #已经构建完成，可以runable，可以invoke的agent


class TaskTool:
    def __init__(self, subagents):
        self.name = "task"
        self.subagents_by_name = {subagent.name : subagent for subagent in subagents}
        self.available_agents = "\n".join(["- {} : {}".format(name,subagent.description) for name,subagent in self.subagents_by_name.items()])
        self.description = (
            "把一个完整、可独立执行的复杂任务委派给专业子Agent。\n"
            "可用子Agent：\n{}\n"
            "当多个任务互相独立时，可以在一次回复中生成多个task ToolCall。"
        ).format(self.available_agents)

        self.definition = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "交给子Agent执行的完整任务描述，必须包含必要上下文和预期输出。",
                        },
                        "subagent_type": {
                            "type": "string",
                            "description": "要使用的子Agent名称，必须来自可用子Agent列表。",
                        },
                    },
                    "required": ["description", "subagent_type"],
                    "additionalProperties": False,
                },
            },
        }

    def get_subagent(self, subagent_type):
        if subagent_type not in self.subagents_by_name:
            raise ToolInvocationException("subagent_type: {} 不存在，可用的subagents如下: {}".format(subagent_type, list(self.subagents_by_name)))

        return self.subagents_by_name[subagent_type]

    def run(self, arguments, node_runtime):
        description = arguments["description"]
        subagent_type = arguments["subagent_type"]
        subagent = self.get_subagent(subagent_type)

        task_execution_context = _task_execution_context_var.get()

        if task_execution_context is None:
            raise RuntimeError("task_execution_context必须是在CompiledStateGraph执行Task期间调用")

        graph_checkpoint_context = task_execution_context.graph_checkpoint_context

        task = task_execution_context.task

        subagent_checkpoint_ns_segment = "{}:{}".format(task.node_name, task.task_id)

        if graph_checkpoint_context.checkpoint_ns:
            subagent_checkpoint_ns = "{}|{}".format(graph_checkpoint_context.checkpoint_ns, subagent_checkpoint_ns_segment)
        else:
            subagent_checkpoint_ns = subagent_checkpoint_ns_segment

        subagent_checkpoint_map = {
                                    **graph_checkpoint_context.checkpoint_map,
                                    graph_checkpoint_context.checkpoint_ns : graph_checkpoint_context.checkpoint_id
                                  }

        if node_runtime.resume_map:
            graph_input = Command(resume=node_runtime.resume_map)
            input_update = None
        else:
            graph_input = subagent.runnable.create_initial_state()
            user_message = {
                                "id" : "msg_{}".format(uuid.uuid4().hex),
                                "role" : "user",
                                "content" : description

                           }

            input_update = {"messages":[user_message]}

        output = subagent.runnable.compiled_graph._run(
                            graph_input = graph_input,
                            input_update = input_update,
                            thread_id = graph_checkpoint_context.thread_id,
                            checkpoint_ns = subagent_checkpoint_ns,
                            checkpoint_map = subagent_checkpoint_map,
                            context = node_runtime.context,#当前其实就是存储个user_id,用于memory。
                            stream_writer = node_runtime.stream_writer
                        )

        if "__interrupt__" in output:
            raise GraphInterrupt(output["__interrupt__"])

        messages = output["messages"]
        for message in reversed(messages):
            if message["role"] == "assistant":
                content = message["content"]
                if content.strip():
                    return content

        raise ToolExecutionException("子agent没有返回有效的assistant_message")

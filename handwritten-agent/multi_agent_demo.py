#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import uuid

from agent import Agent
from checkpoint import JsonlCheckpointer
from hitl import Command
from model import LocalChatModel
from multi_agent import CompiledSubAgent, TaskTool
from tool_register import Register
from tools import read_resume_tool, search_project_evidence_tool


def build_multi_agent():
    chat_model = LocalChatModel(
        model_path="models/Qwen3-4B",
        device="mps",
    )
    checkpointer = JsonlCheckpointer(
        local_checkpoint_file="checkpoints/multi_agent_checkpoints.jsonl",
        local_pending_writes_file="checkpoints/multi_agent_pending_writes.jsonl",
    )

    resume_register = Register()
    resume_register.register(read_resume_tool)
    resume_register.register(search_project_evidence_tool)

    resume_agent = Agent(
        chat_model=chat_model,
        register=resume_register,
        system_prompt=(
            "你是简历分析Agent。"
            "读取简历时调用read_resume，查询项目证据时调用search_project_evidence。"
            "只能依据工具结果回答，不得编造经历。"
        ),
        checkpointer=checkpointer,
        tool_hitl_policy={
            "read_resume": ("args_completion", "review"),
            "search_project_evidence": ("review",),
        },
        max_steps=10,
    )

    compiled_resume_agent = CompiledSubAgent(
        name="resume_agent",
        description="负责读取简历、查询项目证据并完成求职相关分析。",
        runnable=resume_agent,
    )
    task_tool = TaskTool([compiled_resume_agent])

    supervisor_register = Register()
    supervisor_register.register(task_tool)

    supervisor_agent = Agent(
        chat_model=chat_model,
        register=supervisor_register,
        system_prompt=(
            "你是Supervisor Agent。"
            "凡是读取简历、查询项目证据或分析求职能力的任务，"
            "都必须调用task工具并选择resume_agent。"
            "description必须包含完整任务、必要上下文和预期输出。"
            "收到子Agent结果后，再向用户给出最终回答。"
        ),
        checkpointer=checkpointer,
        tool_hitl_policy={
            "task": ("args_completion",),
        },
        max_steps=10,
    )
    return supervisor_agent


def run_demo():
    supervisor_agent = build_multi_agent()
    supervisor_state = supervisor_agent.create_initial_state()
    thread_id = "thread_{}".format(uuid.uuid4().hex)
    checkpoint_ns = ""
    checkpoint_id = None

    while True:
        user_input = input("用户: ").strip()
        if user_input == "exit":
            break
        if not user_input:
            user_input = None

        supervisor_state = supervisor_agent.invoke(
            user_input,
            supervisor_state,
            thread_id,
            checkpoint_ns,
            checkpoint_id,
        )

        while "__interrupt__" in supervisor_state:
            command_resume = {}
            for interrupt_data in supervisor_state["__interrupt__"]:
                print(
                    "interrupt_request: ",
                    json.dumps(interrupt_data.value, ensure_ascii=False, indent=2),
                )
                resume_text = input("请输入resume_value(json): ")
                command_resume[interrupt_data.id] = json.loads(resume_text)

            supervisor_state = supervisor_agent.invoke(
                Command(resume=command_resume),
                supervisor_state,
                thread_id,
                checkpoint_ns,
                checkpoint_id,
            )

        print("assistant: ", supervisor_state["messages"][-1]["content"])


if __name__ == "__main__":
    run_demo()

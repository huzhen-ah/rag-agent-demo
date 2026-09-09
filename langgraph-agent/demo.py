#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:18:13 2026

@author: huzhen
"""

import sqlite3
import json

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from model import LocalQwenChatModel
from agent import create_resume_agent
from multi_agent import create_supervisor_agent


model = LocalQwenChatModel(
    model_path="models/Qwen3-4B",
    device="mps",
)


checkpoint_connection = sqlite3.connect(
    "checkpoints.sqlite",
    check_same_thread=False,
)
checkpointer = SqliteSaver(checkpoint_connection)

memory_connection = sqlite3.connect(
    "memory.sqlite",
    check_same_thread=False,
    isolation_level=None,
)
memory_store = SqliteStore(memory_connection)
memory_store.setup()


resume_agent = create_resume_agent(model)

subagents = [
    {
        "name": "resume_agent",
        "description": "负责读取、审核和分析用户简历",
        "runnable": resume_agent,
    }
]

supervisor_agent = create_supervisor_agent(
    model=model,
    subagents=subagents,
    checkpointer=checkpointer,
    store=memory_store,
)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "multi_agent_test_231123",
        }
    }
    context = {
        "user_id": "user_A",
    }

    initial_state = {
        "messages": [
            HumanMessage(
                content="请让简历专家读取main简历，并告诉我求职方向。"
            )
        ],
        "model_call_count": 0,
    }

    response = supervisor_agent.invoke(
        input=initial_state,
        config=config,
        context=context,
    )

    while "__interrupt__" in response:
        resume_map = {}

        for interrupt_data in response["__interrupt__"]:
            print(
                "interrupt_request:",
                json.dumps(
                    interrupt_data.value,
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            resume_text = input("请输入resume_value(json): ")
            resume_value = json.loads(resume_text)
            resume_map[interrupt_data.id] = resume_value

        response = supervisor_agent.invoke(
            input=Command(resume=resume_map),
            config=config,
            context=context,
        )

    print("final_response:", response["messages"][-1].content)
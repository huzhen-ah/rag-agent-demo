#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:46:52 2026

@author: huzhen
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from state import AgentState
from model import LocalQwenChatModel
from nodes import ModelNode, ToolReviewNode
from tools import read_resume, search_project_evidence
from langchain_core.messages import SystemMessage, HumanMessage



tools = [read_resume, search_project_evidence]

model = LocalQwenChatModel(
    model_path="models/Qwen3-4B",
    device="mps",
)

model_with_tools = model.bind_tools(tools)

model_node = ModelNode(model_with_tools)
tool_review_node = ToolReviewNode()
tool_node = ToolNode(tools)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", model_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tool_review", tool_review_node)
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", tools_condition,{"tools":"tool_review","__end__":END})
graph_builder.add_edge("tool_review", "tools")
graph_builder.add_edge("tools", "model")

connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(connection)
compiled_state_graph = graph_builder.compile(checkpointer=checkpointer)


initial_state = {
                    "messages" : [
                                    SystemMessage(content="你是一个求职助手。需要读取简历时必须调用工具，只能依据工具返回的内容回答。"),
                                    HumanMessage(content="请读取main的简历，并告诉我求职方向。")
                                 ],
                    "model_call_count" : 0
                }

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "hitl_test_1"
        }
    }
    response = compiled_state_graph.invoke(input=initial_state, config=config)
    print(response)
    interrupt_data = response["__interrupt__"][0]
    print("interrupt_request:", interrupt_data.value)

    review_response = {}

    for tool_call in interrupt_data.value["tool_calls"]:
        review_response[tool_call["id"]] = {
            "type": "approve"
        }

    response = compiled_state_graph.invoke(
        input=Command(resume=review_response),
        config=config
    )

    print("final_response:", response)

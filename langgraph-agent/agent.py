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
from langgraph.store.sqlite import SqliteStore
import sqlite3
from state import AgentState
from model import LocalQwenChatModel
from nodes import ModelNode, ToolArgsCompletionNode, ToolReviewNode, MemoryWriteNode
from tools import read_resume, search_project_evidence, update_user_profile
from langchain_core.messages import SystemMessage, HumanMessage
from routers import router_after_tool_review
import json

tools = [read_resume, search_project_evidence]
tool_hitl_policy = {
                        "read_resume":("args_completion", "review"),
                        "search_project_evidence": ("review",)
                   }
model = LocalQwenChatModel(
    model_path="models/Qwen3-4B",
    device="mps",
)

model_with_tools = model.bind_tools(tools)

model_node = ModelNode(model_with_tools)
tool_args_completion_node = ToolArgsCompletionNode(tools, tool_hitl_policy)
tool_review_node = ToolReviewNode(tool_hitl_policy)
tool_node = ToolNode(tools)
memory_write_node = MemoryWriteNode(model, update_user_profile)


tool_workflow_subgraph_builder = StateGraph(AgentState)
tool_workflow_subgraph_builder.add_node("tool_args_completion", tool_args_completion_node)
tool_workflow_subgraph_builder.add_node("tool_review", tool_review_node)
tool_workflow_subgraph_builder.add_node("tools", tool_node)
tool_workflow_subgraph_builder.add_edge(START, "tool_args_completion")
tool_workflow_subgraph_builder.add_conditional_edges(
                        source="tool_args_completion", 
                        path=tool_args_completion_node.router_after_args_completion
)
tool_workflow_subgraph_builder.add_conditional_edges(
                        source="tool_review", 
                        path=router_after_tool_review,
                        path_map={"finished":END}
)

tool_workflow_subgraph_builder.add_edge("tools", END)

compiled_tool_workflow_subgraph = tool_workflow_subgraph_builder.compile(checkpointer=True)






graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", model_node)
graph_builder.add_node("tool_workflow", compiled_tool_workflow_subgraph)
graph_builder.add_node("memory_write", memory_write_node)
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges(
                        source="model", 
                        path=tools_condition, 
                        path_map={"tools":"tool_workflow", "__end__":"memory_write"}
)
graph_builder.add_edge("tool_workflow", "model")
graph_builder.add_edge("memory_write", END)
checkpoint_connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_connection)
memory_connection = sqlite3.connect("memory.sqlite", check_same_thread=False, isolation_level=None)
memory_store = SqliteStore(memory_connection)
memory_store.setup()#也可不写，当然checkpointer.setup()也可以写。
compiled_state_graph = graph_builder.compile(checkpointer=checkpointer, store=memory_store)


initial_state = {
                    "messages" : [
                                    SystemMessage(content="你是一个求职助手。需要读取简历时必须调用工具，只能依据工具返回的内容回答。"),
                                    HumanMessage(content="请读取main简历，并告诉我求职方向。")
                                 ],
                    "model_call_count" : 0
                }

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "subgraph_hitl_test_1122"
        }
    }
    context = {"user_id" : "user_A"}
    response = compiled_state_graph.invoke(input=initial_state, config=config, context=context)
    while "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0]
        print("interrupt_request:", interrupt_data.value)
        print("................")
        
        resume_text = input("请输入resume_value(json): ")
        resume_response = json.loads(resume_text)
        
        response = compiled_state_graph.invoke(
            input=Command(resume=resume_response),
            config=config,
            context=context
        )

    print("final_response:", response)
# system_message = SystemMessage(
#     content="你是一个求职助手，只能依据已知信息回答。"
# )

# if __name__ == "__main__":
#     context = {"user_id": "user_A"}

#     thread_A_state = {
#         "messages": [
#             system_message,
#             HumanMessage(
#                 content=(
#                     "请记住：我长期希望在深圳找Agent工程师工作，"
#                     "我有9年深度学习经验。"
#                 )
#             ),
#         ],
#         "model_call_count": 0,
#     }
#     config_A={
#         "configurable": {
#             "thread_id": "memory_test_thread_A_gast"
#         }
#     }
#     for update_event in compiled_state_graph.stream(
#         input=thread_A_state,
#         config=config_A,
#         context=context,
#         stream_mode="updates"
#     ):
#         print("stream_event: ", update_event)

#     profile_item = memory_store.get(
#         ("user_A", "memories"),
#         "profile",
#     )
#     print("memory:", profile_item.value)

#     thread_B_state = {
#         "messages": [
#             system_message,
#             HumanMessage(
#                 content="我希望在哪个城市找什么工作？我有多少年经验？"
#             ),
#         ],
#         "model_call_count": 0,
#     }

#     response = compiled_state_graph.invoke(
#         input=thread_B_state,
#         config={
#             "configurable": {
#                 "thread_id": "memory_test_thread_B_gast"
#             }
#         },
#         context=context,
#     )

#     print("assistant:", response["messages"][-1].content)
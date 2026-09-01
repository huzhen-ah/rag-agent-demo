#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:21:24 2026

@author: huzhen
"""

from state import AgentState
from hitl import Send



def router_after_model(state: AgentState)->str:
    """
    这里有一点需要注意：返回的东西可能是list，也可能是str。
    还有一个：外层创建ConditionalTansition的时候，传入的path_map，如果为None，那么这个函数
    返回的东西一定是node_name，如果path_map不是None,那么返回的就是当前的router_key,通过这个key拿到node_name
    """
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "need_tools"
    else:
        return "finished"
        

def router_after_tool_review(state):
    messages = state["messages"]

    assistant_message = None
    assistant_message_index = None



    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] == "assistant":
            assistant_message = message
            assistant_message_index = index
            break

    if assistant_message is None:
        raise ValueError("router_after_tool_review找不到assistant_message")

    completed_tool_call_ids = set()
    for message in messages[assistant_message_index+1:]:
        if message["role"] == "tool":
            completed_tool_call_ids.add(message["tool_call_id"])

    sends = []
    for tool_call in assistant_message["tool_calls"]:
        if tool_call["id"] not in completed_tool_call_ids:
            sends.append(Send(node="tool_node",arg=tool_call))
    if sends:
        return sends
    return "model_node"

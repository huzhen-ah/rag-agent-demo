#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:00:26 2026

@author: huzhen
"""

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.types import Send



def router_after_tool_review(state):
    tool_message_ids = set()
    ai_message = None
    
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            ai_message = message
            break
        if isinstance(message, ToolMessage):
            tool_message_ids.add(message.tool_call_id)
    if ai_message is None:
        raise ValueError("在state中找不到AIMessage")
    
    pending_tool_calls = []
    for tool_call in ai_message.tool_calls:
        if tool_call["id"] not in tool_message_ids:
            pending_tool_calls.append(tool_call)
    if len(pending_tool_calls) == 0:
        return "model"
    
    sends = []
    for tool_call in pending_tool_calls:
        sends.append(Send("tools", [tool_call]))
    return sends
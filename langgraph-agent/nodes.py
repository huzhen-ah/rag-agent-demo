#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:00:38 2026

@author: huzhen
"""

from langgraph.types import interrupt
from copy import deepcopy
from langchain_core.messages import ToolMessage

class ModelNode:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, state):
        ai_message = self.model.invoke(state["messages"])
        update_state = {
                "messages" : [ai_message],
                "model_call_count" : 1
              }
        return update_state
    
class ToolReviewNode:
    def __init__(self, tool_hitl_policy):
        self.tool_hitl_policy = tool_hitl_policy
        
    def __call__(self, state):
        ai_message = state["messages"][-1]
        review_tool_calls = []
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] in self.tool_hitl_policy:
                if "review" in self.tool_hitl_policy[tool_call["name"]]:
                    review_tool_calls.append(tool_call)
        if not review_tool_calls:
            return {}
        review_request = {
                "type" : "tool_review",
                "tool_calls" : review_tool_calls
            }
        review_response = interrupt(review_request)
        updated_tool_calls = []
        rejected_tool_messages = []
        for tool_call in ai_message.tool_calls:
            tool_call_id = tool_call["id"]
            if tool_call_id not in review_response:#这里默认interrupt返回的结果必须是全部需要审核的，不能漏。
                updated_tool_calls.append(tool_call)
                continue
            decision_type = review_response[tool_call_id]["type"]
            if  decision_type== "approve":
                updated_tool_calls.append(tool_call)
            elif decision_type == "edit":
                updated_tool_call = deepcopy(tool_call)
                updated_tool_call["args"] = review_response[tool_call_id]["args"]
                updated_tool_calls.append(updated_tool_call)
            elif decision_type == "reject":
                updated_tool_calls.append(tool_call)
                rejected_tool_message = ToolMessage(
                                                    content=review_response[tool_call_id]["reason"],
                                                    name = tool_call["name"],
                                                    tool_call_id=tool_call["id"],
                                                    status = "error"
                                                   )
                rejected_tool_messages.append(rejected_tool_message)
            else:
                raise ValueError("当前只支持approve & edit & reject, 未支持的类型: {}".format(decision_type))
        updated_ai_message = ai_message.model_copy(update={"tool_calls":updated_tool_calls})
        updated_messages = [updated_ai_message]
        updated_messages.extend(rejected_tool_messages)
        return {"messages":updated_messages}
        
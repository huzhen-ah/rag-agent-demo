#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:00:38 2026

@author: huzhen
"""

from langgraph.types import interrupt



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
    def __call__(self, state):
        ai_message = state["messages"][-1]
        review_request = {
                "type" : "tool_review",
                "tool_calls" : ai_message.tool_calls
            }
        review_response = interrupt(review_request)
        for tool_call in ai_message.tool_calls:
            tool_call_id = tool_call["id"]
            if review_response[tool_call_id]["type"] != "approve":
                raise ValueError("当前只支持approve")
        return {}
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:00:38 2026

@author: huzhen
"""

from langgraph.types import interrupt
from copy import deepcopy
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
import json

class ModelNode:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt
        
    def __call__(self, state, runtime):
        messages = deepcopy(state["messages"])
        user_id = runtime.context["user_id"]
        namespace = (user_id, "memories")
        profile_item = runtime.store.get(namespace, "profile")
        system_prompt = self.system_prompt
        if profile_item is not None:
            profile = profile_item.value
            profile_content = json.dumps(profile, ensure_ascii=False)
            system_prompt += (
                "\n\n用户长期记忆：\n{}".format(profile_content)
            )
        messages = [SystemMessage(content=system_prompt)] + messages
        ai_message = self.model.invoke(messages)
        update_state = {
                "messages" : [ai_message],
                "model_call_count" : 1
              }
        return update_state
    
class ToolArgsCompletionNode:
    def __init__(self, tools, tool_hitl_policy):
        self.name2tool = {}
        for tool in tools:
            self.name2tool[tool.name] = tool
        
        self.tool_hitl_policy = tool_hitl_policy
        
    def get_missing_required_args(self, tool_call):
        tool_name = tool_call["name"]
        tool = self.name2tool[tool_name]
        tool_schema = tool.args_schema.model_json_schema()
        required_args = tool_schema.get("required",[])
        provided_args = tool_call["args"]
        missing_required_args = []
        for arg in required_args:
            if arg not in provided_args:
                missing_required_args.append(arg)
        return missing_required_args
    
    def router_after_args_completion(self, state):
        for tool_call in state["messages"][-1].tool_calls:
            behaviors = self.tool_hitl_policy.get(tool_call["name"], ())
            if "args_completion" not in behaviors:
                continue
            missing_required_args = self.get_missing_required_args(tool_call)
            if missing_required_args:
                return "tool_args_completion"
        return "tool_review"
        
            
    def __call__(self, state):
        ai_message = state["messages"][-1]
        missing_args_requests = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            behaviors = self.tool_hitl_policy.get(tool_name, ())
            if "args_completion" not in behaviors:
                continue
            missing_required_args = self.get_missing_required_args(tool_call)
            if not missing_required_args:
                continue
            missing_args_request = {
                                        "tool_call_id" : tool_call["id"],
                                        "tool_name" : tool_call["name"],
                                        "current_args" : tool_call["args"],
                                        "missing_args" : missing_required_args
                                   }
            missing_args_requests.append(missing_args_request)
        
        if not missing_args_requests:
            return {}
        args_completion_response = interrupt(
                    {
                        "type": "args_completion",
                        "requests": missing_args_requests
                    }
        )
        
        updated_ai_message = deepcopy(ai_message)
        
        for tool_call in updated_ai_message.tool_calls:
            tool_call_id = tool_call["id"]
            if tool_call_id not in args_completion_response:
                continue
            tool_call["args"].update(args_completion_response[tool_call_id])
        return {"messages" : [updated_ai_message]}
    
    
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
   
class MemoryWriteNode:
    def __init__(self, model, memory_tool):
        self.memory_tool = memory_tool
        self.model = model.bind_tools([memory_tool])
        self.system_message = SystemMessage(
                       content=(
                            "你是长期记忆提取器。"
                            "只分析latest_user_message中用户明确表达的长期求职信息。"
                            "existing_profile只用于理解用户对已有信息的修改。"
                            "临时查询、一次性要求、疑问和模型推测不能保存。"
                            "存在需要新增、修改或明确删除的长期信息时，"
                            "调用update_user_profile工具。"
                            "没有需要更新的信息时，不调用任何工具。"
                       )
        )
        
    def __call__(self, state, runtime):
        latest_user_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                latest_user_message = message
                break
            
        if latest_user_message is None:
            return {}
        
        user_id = runtime.context["user_id"]
        namespace = (user_id, "memories")
        key = "profile"
        profile_item = runtime.store.get(namespace, key)
        if profile_item is None:
            existing_profile = {}
        else:
            existing_profile = profile_item.value
        
        user_message = HumanMessage(
                        content = json.dumps(
                            {
                                "existing_profile" : existing_profile,
                                "latest_user_message" : latest_user_message.content
                            },
                            ensure_ascii=False
                        )
        )
        memory_extraction_messages = [self.system_message, user_message]
        
        response = self.model.invoke(memory_extraction_messages)
        
        tool_calls = response.tool_calls
        if not tool_calls:
            return {"model_call_count":1}
        
        if len(tool_calls) != 1:
            raise ValueError("长期记忆提取模型最多只能返回一个ToolCall")
        
        tool_call = tool_calls[0]
        if tool_call["name"] != self.memory_tool.name:
            raise ValueError("长期记忆提取模型只能调用:{}".format(self.memory_tool.name))
        profile_changes = self.memory_tool.invoke(tool_call["args"])
        profile_updates = profile_changes["updates"]
        fields_to_delete = profile_changes["fields_to_delete"]
        if not profile_updates and not fields_to_delete:
            return {"model_call_count":1}
        merged_profile = {**existing_profile, **profile_updates}
        if fields_to_delete:
            merged_profile = {k:v for k,v in merged_profile.items() if k not in fields_to_delete}
        
        if merged_profile:
            runtime.store.put(namespace, key, merged_profile)
        else:
            runtime.store.delete(namespace, key)
        
        return {"model_call_count" : 1}
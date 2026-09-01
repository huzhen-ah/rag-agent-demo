#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:11:00 2026

@author: huzhen
"""
from state import AgentState,AgentStateUpdate
import uuid
import json
from exceptions import ToolInvocationException,ToolExecutionException
from hitl import interrupt
from copy import deepcopy
import inspect



class ModelNode:
    def __init__(self,chat_model,tool_definitions):
        self.chat_model = chat_model
        self.tool_definitions = tool_definitions
    
    def __call__(self,state:AgentState, node_runtime)->AgentStateUpdate:
        messages = deepcopy(state["messages"])
        
        if node_runtime.memory_store is not None:
            if node_runtime.context is None or "user_id" not in node_runtime.context:
                raise ValueError("使用MemoryStore时，必须在context中提供user_id")
            user_id = node_runtime.context["user_id"]
            namespace = (user_id, "memories")
            memory_items = node_runtime.memory_store.search(namespace)
            if memory_items:
                memories = [{"key":item.key,"value":item.value} for item in memory_items]
                memory_content = json.dumps(memories, ensure_ascii=False)
                messages[0]["content"] += "\n\n用户长期记忆：\n{}".format(memory_content)
                
            
            
        response = self.chat_model.invoke(messages,self.tool_definitions)
        
        assistant_message = {
            "id":r"msg_{}".format(uuid.uuid4().hex),
            "role":"assistant",
            "content":response["content"],
            "tool_calls":response["tool_calls"]
        }
        
        agentStateUpdate = {"messages":[assistant_message],"model_call_count":1}
        return agentStateUpdate
    

class ToolNode:
    def __init__(self,register):
        self.register = register
        
    def __call__(self, tool_call, node_runtime)->AgentStateUpdate:
        """
        id
        content
        role
        tool_call_id
        name
        status
        """
        tool_call_id = tool_call["id"]
        name = tool_call["name"]
        args = tool_call["args"]
        try:
            tool = self.register.get(name)
            if "node_runtime" in inspect.signature(tool.run).parameters:
                content = tool.run(args, node_runtime=node_runtime)
            else:
                content = tool.run(args)
            content = json.dumps(content,ensure_ascii=False)
            status = "success"
        except ToolInvocationException as error:
            content = json.dumps("error:{}".format(error), ensure_ascii=False)
            status = "error"
        except ToolExecutionException as error:
            content = json.dumps("error:{}".format(error), ensure_ascii=False)
            status = "error"
        tool_message = {
                            "id" : "msg_{}".format(uuid.uuid4().hex),
                            "content" : content,
                            "role" : "tool",
                            "tool_call_id" : tool_call_id,
                            "name" : name,
                            "status" : status
                       }
        return {"messages" : [tool_message]}
    
class ToolArgsCompletionNode:
    def __init__(self, register, tool_hitl_policy):
        valid_behaviors = set([(), ("args_completion",), ("review",), ("args_completion", "review")])
        for tool_name, behavior in tool_hitl_policy.items():
            register.get(tool_name)
            if not isinstance(behavior, tuple):
                raise TypeError("{} 的HITL policy 必须是tuple".format(tool_name))
            if behavior not in valid_behaviors:
                raise ValueError("{}的HITL policy不合法:{}".format(tool_name,behavior))
        self.register = register
        self.tool_hitl_policy = tool_hitl_policy  
        
    def get_missing_required_args(self, tool_call):
        tool = self.register.get(tool_call["name"])
        required_args = tool.definition["function"]["parameters"].get("required",[])
        provided_args = tool_call["args"]
        missing_required_args = []
        for arg in required_args:
            if arg not in provided_args:
                missing_required_args.append(arg)
        return missing_required_args

    def router_after_toolArgsCompletionNode(self,state):
        missing_args = []
        for tool_call in state["messages"][-1]["tool_calls"]:
            if "args_completion" not in self.tool_hitl_policy.get(tool_call["name"],()):
                continue
            _missing_args = self.get_missing_required_args(tool_call)
            missing_args.extend(_missing_args)
        if len(missing_args) == 0:
            return "tool_review_node"
        else:
            return "needs_more_args"
    
    def __call__(self, state):
        assistant_message = deepcopy(state["messages"][-1])
        tool_calls = assistant_message["tool_calls"]
        missing_args_requests = []
        tool_id_2_args_existed = {}
        for tool_call in tool_calls:
            behaviors = self.tool_hitl_policy.get(tool_call["name"],())
            
            if "args_completion" not in behaviors:
                continue
            
            missing_args = self.get_missing_required_args(tool_call)
            
            tool_id_2_args_existed[tool_call["id"]] = tool_call["args"]
            if not missing_args:
                continue
            
            missing_args_requests.append(
                {
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_call["name"],
                    "current_args": tool_call["args"],
                    "missing_args": missing_args
                }
            )
        if not missing_args_requests:
            return {}
        
        
        args_completion_response = interrupt(
            {
                "type": "args_completion",
                "requests": missing_args_requests
            }
        )
        
        if not isinstance(args_completion_response, dict):
            raise TypeError("args_completion_response类型必须是dict")
        
        expected_tool_call_ids = set([request["tool_call_id"] for request in missing_args_requests])
        provided_tool_call_ids = set(args_completion_response)
        
        unknown_tool_call_ids = provided_tool_call_ids - expected_tool_call_ids
        if unknown_tool_call_ids:
            raise ValueError("以下tool_call_ids不在期望中:{}".format(unknown_tool_call_ids))
       
        tool_call_id_2_total_args = {}
        for _ in missing_args_requests:
            tool_call_id = _["tool_call_id"]
            tool_name = _["tool_name"]
            tool_call_id_2_total_args[tool_call_id] = set(self.register.get(tool_name).definition["function"]["parameters"]["properties"])
            
        
        
        unknown_tool_call_id_2_args = {}
        again_tool_call_id_2_args = {}
        for tool_call_id, arg_value_dict in args_completion_response.items():
            for arg,value in arg_value_dict.items():
                if arg not in tool_call_id_2_total_args[tool_call_id]:
                    if tool_call_id not in unknown_tool_call_id_2_args:
                        unknown_tool_call_id_2_args[tool_call_id] = set([arg])
                    else:
                        unknown_tool_call_id_2_args[tool_call_id].add(arg)
                if arg in tool_id_2_args_existed[tool_call_id]:
                    if tool_call_id not in again_tool_call_id_2_args:
                        again_tool_call_id_2_args[tool_call_id] = set([arg])
                    else:
                        again_tool_call_id_2_args[tool_call_id].add(arg)
        
        if len(unknown_tool_call_id_2_args):
            raise ValueError("interrupt反馈的args有些是违法的，即不是fucntion定义的参数: {}".format(unknown_tool_call_id_2_args))
            
        if len(again_tool_call_id_2_args):
            raise ValueError("interrupt反馈的args有些已经存在的: {}".format(again_tool_call_id_2_args))
        for tool_call in assistant_message["tool_calls"]:
            tool_call_id = tool_call["id"]
            if tool_call_id not in provided_tool_call_ids:
                continue
            tool_call["args"].update(args_completion_response[tool_call_id])
            
        return {"messages":[assistant_message]}  
            
            
            
class ToolReviewNode:
    def __init__(self, tool_hitl_policy):
        self.tool_hitl_policy = tool_hitl_policy
        
    def __call__(self, state):
        message = deepcopy(state["messages"][-1])
        tool_calls = message["tool_calls"]
        review_requests = []
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            tool_name = tool_call["name"]
            args = tool_call["args"]
            behaviors = self.tool_hitl_policy.get(tool_name,())
            if "review" not in behaviors:
                continue
            review_requests.append(
                        {
                            "tool_call_id" : tool_call_id,
                            "tool_name" : tool_name,
                            "args" : args
                        }
            )
        if len(review_requests) == 0:
            return {}
        
        review_response = interrupt(
                            {
                                "type" : "tool_review",
                                "requests" : review_requests
                            }
        )
        
        if not isinstance(review_response, dict):
            raise TypeError("review_response类型必须是dict")
            
        request_tool_call_ids = set([_["tool_call_id"] for _ in review_requests])
        response_tool_call_ids = set(review_response)
        if request_tool_call_ids != response_tool_call_ids:
            raise ValueError("interrupt返回的tool_call_ids与请求的tool_call_ids不一致，多了: {}, 少了: {}".format(response_tool_call_ids-request_tool_call_ids,request_tool_call_ids-response_tool_call_ids))
        
        review_ret = []
        update_flag = False
        
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            if tool_call_id not in review_response:
                continue
            response_type = review_response[tool_call_id]["type"]
            if response_type == "approve":
                continue
            if response_type == "edit":
                tool_call["args"] = review_response[tool_call_id]["args"]
                update_flag = True
            elif response_type == "reject":
                tool_message = {
                                  "id" : "msg_{}".format(uuid.uuid4().hex),
                                  "role" : "tool",
                                  "tool_call_id" : tool_call_id,
                                  "name" : tool_call["name"],
                                  "status" : "error",
                                  "content" : review_response[tool_call_id]["reason"]
                               }
                review_ret.append(tool_message)
            else:
                raise TypeError("不存在的reponse type: {}".format(response_type))
        if update_flag:
            review_ret = [message] + review_ret
        if len(review_ret) == 0:
            return {}
        else:
            return {"messages" : review_ret}
        
          
class MemoryWriteNode:
    def __init__(self, chat_model, memory_tool):
        self.chat_model = chat_model
        self.memory_tool = memory_tool
        self.tool_definitions = [memory_tool.definition]
        self.system_message = {
                                    "role": "system",
                                    "content": (
                                        "你是长期记忆提取器。"
                                        "只分析latest_user_message中用户明确表达的长期求职信息。"
                                        "existing_profile只用于理解用户对已有信息的修改。"
                                        "临时查询、一次性要求、疑问和模型推测不能保存。"
                                        "存在需要新增、修改或明确删除的长期信息时，调用update_user_profile工具。"
                                        "用户明确要求忘记或删除某项信息时，将对应字段名放入fields_to_delete。"
                                        "用户没有提到某个字段不代表删除，只有明确要求遗忘或删除时才能使用fields_to_delete。"
                                        "没有需要更新的信息时，不调用任何工具，也不要输出解释。"
                                    )
                                }
        
    def __call__(self, state, node_runtime=None):
        if node_runtime is None or node_runtime.memory_store is None:
            return {}
        
        if node_runtime.context is None or "user_id" not in node_runtime.context:
            raise ValueError("使用MemoryStore时，必须在context中提供user_id")
        
        latest_user_message = None
        for message in reversed(state["messages"]):
            if message["role"] == "user":
                latest_user_message = message
                break
        if latest_user_message is None:
            return {}
        
        user_id = node_runtime.context["user_id"]
        namespace = (user_id, "memories")
        key = "profile"
        
        profile_item = node_runtime.memory_store.get(namespace, key)
        
        if profile_item is None:
            existing_profile = {}
        else:
            existing_profile = profile_item.value
            
    
        user_message = {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "existing_profile": existing_profile,
                                    "latest_user_message": latest_user_message["content"],
                                },
                                ensure_ascii=False,
                            )
                        }
        
        memory_extraction_messages = [self.system_message, user_message]
        
        response = self.chat_model.invoke(memory_extraction_messages, self.tool_definitions)
        
        tool_calls = response["tool_calls"]
        if not tool_calls:
            return {"model_call_count":1}
        
        if len(tool_calls) != 1:
            raise ValueError("长期记忆提取模型最多只能返回一个ToolCall")
        
        tool_call = tool_calls[0]
        if tool_call["name"] != self.memory_tool.name:
            raise ValueError("长期记忆提取模型只能调用:{}".format(self.memory_tool.name))
            
        profile_changes = self.memory_tool.run(tool_call["args"])
        profile_updates = profile_changes["updates"]
        fields_to_delete = profile_changes["fields_to_delete"]
        if not profile_updates and not fields_to_delete:
            return {"model_call_count":1}
        
        merged_profiles = {**existing_profile, **profile_updates}
        if fields_to_delete:
            merged_profiles = {k:v for k,v in merged_profiles.items() if k not in fields_to_delete}
        if not merged_profiles:
            node_runtime.memory_store.delete(namespace, key)
        else:
            node_runtime.memory_store.put(namespace, key, merged_profiles)
        return {"model_call_count":1}
            
            
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

class ModelNode:
    def __init__(self,chat_model,tool_definitions):
        self.chat_model = chat_model
        self.tool_definitions = tool_definitions
    
    def __call__(self,state:AgentState)->AgentStateUpdate:
        response = self.chat_model.invoke(state["messages"],self.tool_definitions)
        
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
        
    def __call__(self,state:AgentState)->AgentStateUpdate:
        ret = {"messages":[]}
        """
        id
        content
        role
        tool_call_id
        name
        status
        """
        #先找到最后一条assistant_message
        assistant_message = None
        index = -1
        message_lens = len(state["messages"])
        while abs(index) <= message_lens:
            message = state["messages"][index]
            if message["role"] == "assistant":
                assistant_message = message
                break
            index = index - 1
        if assistant_message is None:
            raise ValueError("ToolNode找不到assistant message")
        
        passed_tool_call_ids = set()
        for i in range(1,abs(index)):
            message = state["messages"][-i]
            if message["role"] == "tool":
                passed_tool_call_ids.add(message["tool_call_id"])
        
        
        
        tool_calls = assistant_message["tool_calls"]
        
        tool_call_ids = set()
        for tool_call in tool_calls:
            tool_call_ids.add(tool_call["id"])
        
        unknown_tool_call_ids = passed_tool_call_ids - tool_call_ids
        if unknown_tool_call_ids:
            raise ValueError("assistant message后面不能出现不在assistant messages tool_calls里面的tool_call_ids:{}".format(unknown_tool_call_ids))
        for tool_call in tool_calls:
            try:
                tool_call_id = tool_call["id"]
                if tool_call_id in passed_tool_call_ids:
                    continue
                name = tool_call["name"]
                args = tool_call["args"]
                tool = self.register.get(name)
                content = tool.run(args)
                content = json.dumps(content,ensure_ascii=False)
                status = "success"
            except ToolInvocationException as error:
                content = json.dumps("error:{}".format(error),ensure_ascii=False)
                status = "error"
            except ToolExecutionException as error:
                content = json.dumps("error:{}".format(error),ensure_ascii=False)
                status = "error"
            
            
            message = {"id":r"msg_{}".format(uuid.uuid4().hex),
                        "content":content,
                        "role":"tool",
                        "tool_call_id":tool_call_id,
                        "name":name,
                        "status":status
                        }
            ret["messages"].append(message)
        return ret
    
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
        
                
            
            
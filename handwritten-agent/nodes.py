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
        
        messages = {"messages":[assistant_message],"model_steps":state["model_steps"]+1}
        return messages
    

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
        """
        tool_calls = state["messages"][-1]["tool_calls"]
        for tool_call in tool_calls:
            try:
                name = tool_call["name"]
                args = tool_call["args"]
                tool_call_id = tool_call["id"]
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
            
            
            
            
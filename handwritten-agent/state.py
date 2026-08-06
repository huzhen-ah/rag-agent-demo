#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:21:09 2026

@author: huzhen
"""

import copy
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from operator import add

class ToolCall(TypedDict, total=True):
    id: str
    name: str
    args: dict[str, Any]
    type: NotRequired[Literal["tool_call"]]


class BaseMessage(TypedDict, total=True):  # 基础消息形式，只包含id和content
    id: str
    content: str


class SystemMessage(BaseMessage, total=True):  # 系统消息
    role: Literal["system"]


class UserMessage(BaseMessage, total=True):  # 用户消息
    role: Literal["user"]


class AssistantMessage(BaseMessage, total=True):  # 助手消息
    role: Literal["assistant"]
    tool_calls: NotRequired[list[ToolCall]]


class ToolMessage(BaseMessage, total=True):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    status: Literal["success","error"]

Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


def add_messages(old_messages, update_messages):
    merged_messages = [copy.deepcopy(message) for message in old_messages]
    message_id2position = {_["id"]:position for position,_ in enumerate(merged_messages)}
    for message in update_messages:
        iid = message["id"]
        if not iid:
            raise ValueError("ID不能为空")
        if iid in message_id2position:
            position = message_id2position[iid]
            merged_messages[position] = message
        else:
            position = len(merged_messages)
            message_id2position[iid] = position
            merged_messages.append(message)
    return merged_messages

            
    

class AgentState(TypedDict, total=True):
    messages: Annotated[list[Message], add_messages]
    model_call_count: Annotated[int, add]


class AgentStateUpdate(TypedDict, total=False):
    messages: list[Message]
    model_call_count: int
    
    
if __name__ == "__main__":
    from typing import get_type_hints,get_origin,get_args
    
    keys_definitions = get_type_hints(AgentState,include_extras=True)
    
    key2reducer = {}
    for key,definition in keys_definitions.items():
        type_definition = get_origin(definition)
        if type_definition is Annotated:
            _,reducer = get_args(definition)
            
        else:
            reducer = None
        key2reducer[key] = reducer

    print(key2reducer)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:09:04 2026

@author: huzhen
"""

from state import AgentState,AgentStateUpdate
from typing import get_type_hints, get_origin, get_args, Annotated, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from memory import BaseStore
from streaming import StreamEvent

def get_state_reducers(StateClass):
    key2reducer = {}
    key_definitions = get_type_hints(StateClass,include_extras=True)
    for key,definitions in key_definitions.items():
        origin_type = get_origin(definitions)
        if origin_type is Annotated:
            key_type,reducer = get_args(definitions)
            key2reducer[key] = reducer
        else:
            key2reducer[key] = None
            
    return key2reducer


def apply_updates(old_state:AgentState,update_states:list[AgentStateUpdate],key2reducer):
    updates_by_key = {}
    for update_state in update_states:
        for key,value in update_state.items():
            if key not in updates_by_key:
                updates_by_key[key] = []
            updates_by_key[key].append(value)
    for key,value in updates_by_key.items():
        if key not in key2reducer:
            raise ValueError("{}不在key2reducer中".format(key))
        reducer = key2reducer[key]
        if reducer is None:
            if len(value) > 1:
                raise ValueError("没有reducer的key不能有多个值，一次Super-step对于无reducer的字段，只允许一个node写入")
            
    merged_state = deepcopy(old_state)
    
    for key,updates in updates_by_key.items():
        reducer = key2reducer[key]
        if reducer is None:
            merged_state[key] = updates[0]
        else:
            merged_values = merged_state[key]
            for _updates in updates:
                merged_values = reducer(merged_values,_updates)
            merged_state[key] = merged_values
    return merged_state
    
    
@dataclass
class NodeRuntime:
    """
    Callable是类型描述，表示接受的是一个函数，Callable[[参数类型列表]，返回值类型],
    Callable[[StreamEvent], None]整体表示类型
    """
    context: dict | None = None
    memory_store: BaseStore | None = None
    stream_writer: Callable[[StreamEvent], None] | None = None
    resume_map: dict = field(default_factory=dict)
    


        
    
    
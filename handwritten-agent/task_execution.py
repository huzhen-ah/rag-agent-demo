#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:49:15 2026

@author: huzhen
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PregelScratchpad:
    resume: list[Any] = field(default_factory=list)#这里的resume是一个list，里面的item才是真正的Command.resume中的resume_value
    _interrupt_index: int = 0
    
    def interrupt_counter(self):
        current_index = self._interrupt_index
        self._interrupt_index += 1
        return current_index
 

@dataclass(frozen=True)
class Task:#任务类，其实就是一个身份，任务id：task_id， 任务用的node名字:node_name
    task_id: str
    node_name: str
    input: Any
    

@dataclass
class TaskResult:
    task: Task
    channel: str#目前只能是update,interrupt,error
    value: Any
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:05:29 2026

@author: huzhen
"""
from contextvars import ContextVar
from dataclasses import dataclass
from task_execution import Task, PregelScratchpad


@dataclass(frozen=True)
class GraphCheckpointContext:#运行时上下文
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    checkpoint_map: dict[str,str]
    

    
@dataclass
class TaskExecutionContext:#任务执行时上下文
    graph_checkpoint_context: GraphCheckpointContext
    task: Task
    scratchpad: PregelScratchpad
    

class TaskExecutionContextVar:
    def __init__(self):
        self._task_execution_context_var = ContextVar(
            "current_task_execution_context",
            default=None,
        )

    def get(self):
        return self._task_execution_context_var.get()
    
    def set(self, task_execution_context):
        token = self._task_execution_context_var.set(task_execution_context)
        return token
    
    def reset(self, token):
        self._task_execution_context_var.reset(token)
        
task_execution_context_var = TaskExecutionContextVar()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:58:54 2026

@author: huzhen
"""
from contextvars import ContextVar
from typing import Any
from dataclasses import dataclass, field
import uuid
from exceptions import GraphInterrupt


@dataclass(slots=True)
class Interrupt:
    value: Any
    id: str
    
@dataclass
class Command:
    graph: str | None = None
    update: Any | None = None
    resume: Any | None = None
    goto: Any = ()
    
@dataclass
class PregelScratchpad:
    resume: list[Any] = field(default_factory=list)
    _interrupt_index: int = 0
    
    def interrupt_counter(self):
        current_index = self._interrupt_index
        self._interrupt_index += 1
        return current_index
    
@dataclass
class TaskContext:
    checkpoint_id: str
    task_id: str
    scratchpad: PregelScratchpad

@dataclass(frozen=True)
class Task:
    task_id: str
    node_name: str
    
@dataclass
class TaskResult:
    task: Task
    update: Any | None = None
    interrupts: tuple[Interrupt,...] = ()
    error: Exception | None = None
    
    
_task_context_var = ContextVar(
    "current_task_context",
    default=None,
)
    

def interrupt(value):
    task_context = _task_context_var.get()
    if task_context is None:
        raise RuntimeError("interrupt 必须在Runtime执行Task期间调用")
        
    scratchpad = task_context.scratchpad
    interrupt_index = scratchpad.interrupt_counter()
    if interrupt_index < len(scratchpad.resume):
        return scratchpad.resume[interrupt_index]
    
    interrupt_id = uuid.uuid5(
            namespace= uuid.NAMESPACE_OID, 
            name= "{}_{}".format(task_context.checkpoint_id,task_context.task_id)
    ).hex
    
    interrupt_data = Interrupt(value=value, id=interrupt_id)
    raise GraphInterrupt((interrupt_data,))
    
    
    

    
if __name__ == "__main__":
    
    task_context = TaskContext(
        checkpoint_id="checkpoint_1",
        task_id="task_1",
        scratchpad=PregelScratchpad(),
    )
    
    token = _task_context_var.set(task_context)
    print("token: ",token)
    print(_task_context_var.get())  # 得到 task_context
    
    _task_context_var.reset(token)
    
    print(_task_context_var.get())  # 得到 None
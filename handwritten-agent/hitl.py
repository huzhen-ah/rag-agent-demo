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


@dataclass(slots=True)#slot=True的意思是：这个类的实例只能拥有预先声明的字段，不能运行时随便添加新字段。
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
 
@dataclass(frozen=True)
class Task:
    task_id: str
    node_name: str
    
@dataclass(frozen=True)
class GraphCheckpointContext:
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    checkpoint_map: dict[str,str]
    

    
@dataclass
class TaskExecutionContext:
    graph_checkpoint: GraphCheckpointContext
    task: Task
    scratchpad: PregelScratchpad


@dataclass
class TaskResult:
    task: Task
    update: Any | None = None
    interrupts: tuple[Interrupt,...] = ()
    error: Exception | None = None
    
    
_task_execution_context_var = ContextVar(
    "current_task_execution_context",
    default=None,
)
    
def create_interrupt_id(checkpoint_id, task_id):
    interrupt_id = uuid.uuid5(
            namespace = uuid.NAMESPACE_OID, 
            name = "{}_{}".format(checkpoint_id, task_id)
    ).hex
    return interrupt_id

def interrupt(value):
    task_execution_context = _task_execution_context_var.get()
    if task_execution_context is None:
        raise RuntimeError("interrupt 必须在Runtime执行Task期间调用")
        
    scratchpad = task_execution_context.scratchpad
    interrupt_index = scratchpad.interrupt_counter()
    if interrupt_index < len(scratchpad.resume):
        return scratchpad.resume[interrupt_index]
    
    interrupt_id = create_interrupt_id(task_execution_context.graph_checkpoint.checkpoint_id,task_execution_context.task.task_id)
    interrupt_data = Interrupt(value=value, id=interrupt_id)
    raise GraphInterrupt((interrupt_data,))
    
    
    

    
if __name__ == "__main__":
    thread_id = "thread_id"
    checkpoint_ns = "checkpoint_ns"
    checkpoint_id = "checkpoint_1"
    checkpoint_map = {}
    graph_checkpoint = GraphCheckpointContext(thread_id, checkpoint_ns, checkpoint_id, checkpoint_map)
    task = Task(task_id="task_id", node_name="node_name")
    task_execution_context = TaskExecutionContext(
        graph_checkpoint = graph_checkpoint,
        task = task,
        scratchpad=PregelScratchpad(),
    )
    
    token = _task_execution_context_var.set(task_execution_context)
    print("token: ",token)
    print(_task_execution_context_var.get())  # 得到 task_execution_context
    
    _task_execution_context_var.reset(token)
    
    print(_task_execution_context_var.get())  # 得到 None
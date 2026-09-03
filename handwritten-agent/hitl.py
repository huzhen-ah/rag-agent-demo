#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:58:54 2026

@author: huzhen
"""

from typing import Any
from dataclasses import dataclass
import uuid
from exceptions import GraphInterrupt
from execution_context import task_execution_context_var

@dataclass(slots=True)#slot=True的意思是：这个类的实例只能拥有预先声明的字段，不能运行时随便添加新字段。
class Interrupt:
    value: Any
    id: str
 
# @dataclass
# class Command:
#     graph: str | None = None
#     update: Any | None = None
#     resume: Any | None = None#这里的resume并不是纯value，而是{interrupt_id:resume_value}
#     goto: Any = ()
    
@dataclass
class Command:
    resume: Any | None = None#这里的resume并不是纯value，而是{interrupt_id:resume_value}

    
def create_interrupt_id(checkpoint_id, task_id):
    """
    interrupt_id生成函数，保证只要输入相同，生成就唯一。
    """
    interrupt_id = uuid.uuid5(
            namespace = uuid.NAMESPACE_OID, 
            name = "{}_{}".format(checkpoint_id, task_id)
    ).hex
    return interrupt_id

def interrupt(value):
    """
    interrupt这个函数有意思，它即可以索取又可以提供。
    第一次执行到这个里的时候，它就是报GraphInterrupt异常，让用户提供信息。
    第二次执行到这里的时候，用户已经提供信息了，它就拿到这个信息返回，继续执行。
    为了保证第一次与第二次能对应上，所以要用同一个interrupt_id,create_interrupt_id就是保证生成的interrupt_id唯一。
    """
    task_execution_context = task_execution_context_var.get()
    if task_execution_context is None:
        raise RuntimeError("interrupt 必须在Runtime执行Task期间调用")
        
    scratchpad = task_execution_context.scratchpad
    interrupt_index = scratchpad.interrupt_counter()
    if interrupt_index < len(scratchpad.resume):
        return scratchpad.resume[interrupt_index]
    
    interrupt_id = create_interrupt_id(task_execution_context.graph_checkpoint_context.checkpoint_id,task_execution_context.task.task_id)
    interrupt_data = Interrupt(value=value, id=interrupt_id)
    raise GraphInterrupt((interrupt_data,))
    
    
    

    
if __name__ == "__main__":
    from task_execution import Task, PregelScratchpad
    from execution_context import GraphCheckpointContext, TaskExecutionContext
    
    
    thread_id = "thread_id"
    checkpoint_ns = "checkpoint_ns"
    checkpoint_id = "checkpoint_1"
    checkpoint_map = {}
    graph_checkpoint_context = GraphCheckpointContext(thread_id, checkpoint_ns, checkpoint_id, checkpoint_map)
    task = Task(task_id="task_id", node_name="node_name", input=None)
    task_execution_context = TaskExecutionContext(
        graph_checkpoint_context = graph_checkpoint_context,
        task = task,
        scratchpad=PregelScratchpad(),
    )
    
    token = task_execution_context_var.set(task_execution_context)
    print("token: ",token)
    print(": ",task_execution_context_var.get())  # 得到 task_execution_context
    
    task_execution_context_var.reset(token)
    
    print(": ",task_execution_context_var.get())  # 得到 None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:10:03 2026

@author: huzhen
"""
from hitl import _task_execution_context_var, Command
from exceptions import GraphInterrupt

class SubGraphNode:
    def __init__(self, subgraph, input_mapper, output_mapper):
        self.subgraph = subgraph
        self.input_mapper = input_mapper
        self.output_mapper = output_mapper
        
    def __call__(self, parent_state, runtime):
        
        current_task_execution_context = _task_execution_context_var.get()
        
        if current_task_execution_context is None:
            raise RuntimeError("SubGraphNode必须在Runtime执行Task期间调用")
            
        graph_checkpoint_context = current_task_execution_context.graph_checkpoint_context
        task = current_task_execution_context.task
        
        checkpoint_ns = graph_checkpoint_context.checkpoint_ns
        
        task_checkpoint_ns_segment = "{}:{}".format(task.node_name,task.task_id)
        
        if checkpoint_ns:
            task_checkpoint_ns = "{}|{}".format(checkpoint_ns,task_checkpoint_ns_segment)
        else:
            task_checkpoint_ns = task_checkpoint_ns_segment
            
        checkpoint_map = {
                                **graph_checkpoint_context.checkpoint_map,
                                graph_checkpoint_context.checkpoint_ns : graph_checkpoint_context.checkpoint_id
                         }
        
        if runtime.resume_map:
            graph_input = Command(resume = runtime.resume_map)
        else:
            graph_input = self.input_mapper(parent_state)
            
        output = self.subgraph._run(
            graph_input = graph_input,
            thread_id = graph_checkpoint_context.thread_id,
            checkpoint_ns = task_checkpoint_ns,
            checkpoint_map = checkpoint_map,
            context = runtime.context,#当前其实就是存储个user_id,用于memory。
            stream_writer = runtime.stream_writer
        )
        
        if "__interrupt__" in output:
            raise GraphInterrupt(output["__interrupt__"])
        parent_update = self.output_mapper(parent_state, output)
        return parent_update
            
        
        
        

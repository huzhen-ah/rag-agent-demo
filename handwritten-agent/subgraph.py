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
            
        graph_checkpoint = current_task_execution_context.graph_checkpoint
        task = current_task_execution_context.task
        
        current_checkpoint_ns = graph_checkpoint.checkpoint_ns
        
        child_checkpoint_ns_segment = "{}:{}".format(task.node_name,task.task_id)
        
        if current_checkpoint_ns:
            child_checkpoint_ns = "{}|{}".format(current_checkpoint_ns,child_checkpoint_ns_segment)
        else:
            child_checkpoint_ns = child_checkpoint_ns_segment
            
        child_checkpoint_map = {
                                    **graph_checkpoint.checkpoint_map,
                                    graph_checkpoint.checkpoint_ns : graph_checkpoint.checkpoint_id
                               }
        
        if runtime.resume_map:
            child_graph_input = Command(resume = runtime.resume_map)
        else:
            child_graph_input = self.input_mapper(parent_state)
            
        child_output = self.subgraph._run(
            graph_input = child_graph_input,
            thread_id = graph_checkpoint.thread_id,
            checkpoint_ns = child_checkpoint_ns,
            checkpoint_map = child_checkpoint_map,
            context = runtime.context,
            stream_writer = runtime.stream_writer
        )
        
        if "__interrupt__" in child_output:
            raise GraphInterrupt(child_output["__interrupt__"])
        parent_update = self.output_mapper(parent_state, child_output)
        return parent_update
            
        
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:59:18 2026

@author: huzhen
"""


from runtime import get_state_reducers,apply_updates
from collections.abc import Hashable
from checkpoint import StateSnapshot,Checkpointer
import time
import uuid

START = "__start__"
END = "__end__"


class StateGraph:
    def __init__(self,state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.fixed_edges = {}
        self.conditional_edges = {}
        
    def add_node(self,node_name,node):
        if not isinstance(node_name, str):
            raise TypeError("Node的名称必须是字符串")
        if not node_name.strip():
            raise ValueError("node_name不能为空")
        if node_name in {START,END}:
            raise ValueError("{}是Graph保留的名称".format(node_name))
        if not callable(node):
            raise TypeError("{}对应的Node必须是callable".format(node_name))
        if node_name in self.nodes:
            raise ValueError("Node-{}已经注册".format(node_name))
            
        self.nodes[node_name] = node
        return self
    
    def add_edge(self,source,target):
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("Edge的source or target必须是字符串")
            
        if not source.strip() or not target.strip():
            raise ValueError("Edge的source or target不能为空")
        
        if source == END:
            raise ValueError("END不能作为Edge的source")
            
        if target == START:
            raise ValueError("START不能作为Edge的target")
            
        if source != START and source not in self.nodes:
            raise ValueError("source Node-{}暂未注册".format(source))
        
        if target != END and target not in self.nodes:
            raise ValueError("target Node-{}暂未注册".format(target))
            
        if source not in self.fixed_edges:
            self.fixed_edges[source] = set()
        
        if target in self.fixed_edges[source]:
            raise ValueError("Edge {}->{} 已存在".format(source,target))
        self.fixed_edges[source].add(target)
        
        return self
    
    def add_conditional_edges(self,source,router,path_map=None):
        if not isinstance(source, str):
            raise TypeError("Edge的source or target必须是字符串")
        
        if source != START and source not in self.nodes:
            raise ValueError("Node-{}尚未添加".format(source))
            
        if not source.strip():
            raise ValueError("Edge的source or target不能为空")
            
        if source == END:
            raise ValueError("END不能作为Edge的source")
            
        if source in self.conditional_edges:
            raise ValueError("{}已经存在于conditional_edges".format(source))
        
        if not callable(router):
            raise TypeError("router必须是callable")

        self.conditional_edges[source] = {"router":router,"path_map":path_map}

        return self
    
    def get_reached_nodes_from_edges(self,source,reached_nodes):
        if source in self.fixed_edges:
            source_reached_nodes = self.fixed_edges[source]
        elif source in self.conditional_edges:
            path_map = self.conditional_edges[source]["path_map"]
            if path_map is None:
                source_reached_nodes = set(self.nodes) | {END}
            else:
                source_reached_nodes = set(path_map.values())
        else:
            return reached_nodes
        
        for child_node in source_reached_nodes:
            if child_node in reached_nodes:
                continue
            reached_nodes.add(child_node)
            if child_node != END:
                reached_nodes = self.get_reached_nodes_from_edges(child_node,reached_nodes)
            
        return reached_nodes
        
        
    def compile(self, checkpointer: Checkpointer):
        if START not in self.fixed_edges and START not in self.conditional_edges:
            raise ValueError("START必须至少存在于fixed_edges和conditional_edges中的一个，其实只能存在于一个")
        overlapping_sources = set(self.fixed_edges).intersection(set(self.conditional_edges))
        if overlapping_sources:
            raise ValueError("node不能同时拥有固定边与条件边，以下node同时拥有固定边与条件边：{}".format(overlapping_sources))
        
        nodes = self.nodes.copy()
        
        transitions = {}
        
        for source,targets in self.fixed_edges.items():
            transitions[source] = FixedTransition(targets.copy())
        
        for source,conditional_edge in self.conditional_edges.items():
            router = conditional_edge["router"]
            
            path_map = conditional_edge["path_map"]
            if path_map is not None and not isinstance(path_map, dict):
                raise TypeError("path_map要么是None,要么是dict")
            if isinstance(path_map, dict):
                for router_key,target in path_map.items():
                    if not isinstance(target, str):
                        raise TypeError("target 类型必须是str")
                    
                    if target != END and target not in self.nodes:
                        raise ValueError("source-{} 的router_key-{} 指向了未注册的Node-{}".format(source,router_key,target))
                path_map = path_map.copy()
            
            transitions[source] = ConditionalTransition(router,set(self.nodes),path_map)
            
        
        reached_nodes = set()
        reached_nodes = self.get_reached_nodes_from_edges(START, reached_nodes)
        unreached_nodes = set(self.nodes) - reached_nodes
        if unreached_nodes:
            raise ValueError("从START开始，存在未到达的nodes:{}".format(unreached_nodes))

        return CompiledStateGraph(self.state_schema,nodes,transitions,checkpointer)
    
class CompiledStateGraph:
    def __init__(self, state_schema, nodes, transitions, checkpointer=None):
        self.state_schema = state_schema
        self.nodes = nodes
        self.transitions = transitions
        self.checkpointer = checkpointer
        self.key2reducer = get_state_reducers(state_schema)
        
    def get_created_at(self):
        return str(time.time())

    def merge_updates(self,old_state,update_states):
        return apply_updates(old_state, update_states, self.key2reducer)

    def invoke(self, initial_state, input_update=None, thread_id=None, checkpoint_id=None, recursion_limit=25):
        if thread_id is None:
            raise ValueError("必须传入thread_id")
        
        stateSnapshot = self.checkpointer.get(thread_id,checkpoint_id)
        if stateSnapshot:
            parent_checkpoint_id = stateSnapshot.checkpoint_id
            state = stateSnapshot.state
            if input_update is not None:
                state = self.merge_updates(state, [input_update])
                start_transition = self.transitions[START]
                executable_node_names = start_transition.resolve_targets(state) - {END}
            else:
                executable_node_names = set(stateSnapshot.next_node_names)
            super_step = stateSnapshot.super_step
            recursion_limit += super_step
        else:
            parent_checkpoint_id = None
            state = initial_state
            if input_update is not None:
                state = self.merge_updates(state, [input_update])
            start_transition = self.transitions[START]
            executable_node_names = start_transition.resolve_targets(state) - {END}
            super_step = 0

        
        while executable_node_names:
            
            if super_step >= recursion_limit:
                raise RuntimeError("Graph 超过最大super-step步数:{}".format(recursion_limit))
            update_states = []
            
            for node_name in executable_node_names:
                node = self.nodes[node_name]
                update_state = node(state)
                update_states.append(update_state)
            state = self.merge_updates(state, update_states)
            
            active_node_names = set()
            for node_name in executable_node_names:
                if node_name in self.transitions:
                    active_node_names.update(self.transitions[node_name].resolve_targets(state))
            executable_node_names = active_node_names - {END}
            super_step += 1
            if self.checkpointer and thread_id:
                current_checkpoint_id = r"checkpoint_{}".format(uuid.uuid4().hex)
                stateSnapshot_dict = {
                                        "thread_id": thread_id,
                                        "checkpoint_id": current_checkpoint_id,
                                        "parent_checkpoint_id": parent_checkpoint_id,
                                        "super_step": super_step,
                                        "state": state,
                                        "next_node_names": tuple(executable_node_names),
                                        "created_at": self.get_created_at()
                                     }
                stateSnapshot = StateSnapshot(**stateSnapshot_dict)
                self.checkpointer.put(stateSnapshot)
                parent_checkpoint_id = current_checkpoint_id
        
        return state
        
        
        
class FixedTransition:
    def __init__(self,targets):
        self.targets = targets #某个source对应的targets
        
    def resolve_targets(self,state):
        return self.targets
    
class ConditionalTransition:
    def __init__(self,router,node_names,path_map=None):
        self.router = router
        self.node_names = node_names
        self.path_map = path_map
        
    def resolve_targets(self,state):
        router_result = self.router(state)
        if isinstance(router_result, (list,tuple)):
            router_results = router_result
        else:
            router_results = [router_result]
            
        for router_result in router_results:
            if not isinstance(router_result, Hashable):
                raise TypeError("Router返回的每个router_result必须是Hashable")
        
        
        if self.path_map is None:
            for router_result in router_results:
                if not isinstance(router_result, str):
                    raise TypeError("当path_map是None的时候，返回值类型必须是str")
                if router_result not in self.node_names and router_result != END:
                    raise ValueError("当path_map是None的时候,返回值必须是node_name or END")
            return set(router_results)
        
        for router_result in router_results:
            if router_result not in self.path_map:
                raise ValueError("Router返回了不存在于path_map中的key:{}".format(router_result))
        return {self.path_map[router_result] for router_result in router_results}
        
        

if __name__ == "__main__":
    
    from state import AgentState
    from nodes import ModelNode,ToolNode
    from routers import router_after_model
    from checkpoint import JsonlCheckpointer
    
    model_node = ModelNode(None, None)
    tool_node = ToolNode(None)
    graph = StateGraph(AgentState)
    
    graph.add_node("model_node", model_node)
    graph.add_node("tool_node", tool_node)
    
    graph.add_edge(START, "model_node")
    graph.add_edge("tool_node", "model_node")
    
    path_map={
                "need_tools":"tool_node",
                "finished":END
             }
    
    graph.add_conditional_edges("model_node", router_after_model,path_map)
    
    print(graph.nodes)
    print(graph.fixed_edges)
    print(graph.conditional_edges)
    
    jsonlCheckpointer = JsonlCheckpointer()
    compiledStateGraph = graph.compile(jsonlCheckpointer)
    
    print(compiledStateGraph.nodes is graph.nodes)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:46:52 2026

@author: huzhen
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import AgentState
from nodes import ModelNode, ToolArgsCompletionNode, ToolReviewNode, MemoryWriteNode
from tools import read_resume, search_project_evidence, update_user_profile, query_rag
from routers import router_after_tool_review



def create_resume_agent(model, checkpointer=None, store=None):
    tools = [read_resume, search_project_evidence]
    tool_hitl_policy = {
                            "read_resume":("args_completion", "review"),
                            "search_project_evidence": ("review",)
                       }
    model_node = ModelNode(model.bind_tools(tools), system_prompt="你是一个求职助手。需要读取简历时必须调用工具，只能依据工具返回的内容回答。")
    tool_args_completion_node = ToolArgsCompletionNode(tools, tool_hitl_policy)
    tool_review_node = ToolReviewNode(tool_hitl_policy)
    tool_node = ToolNode(tools)
    memory_write_node = MemoryWriteNode(model, update_user_profile)


    tool_workflow_subgraph_builder = StateGraph(AgentState)
    tool_workflow_subgraph_builder.add_node("tool_args_completion", tool_args_completion_node)
    tool_workflow_subgraph_builder.add_node("tool_review", tool_review_node)
    tool_workflow_subgraph_builder.add_node("tools", tool_node)
    tool_workflow_subgraph_builder.add_edge(START, "tool_args_completion")
    tool_workflow_subgraph_builder.add_conditional_edges(
                            source="tool_args_completion", 
                            path=tool_args_completion_node.router_after_args_completion
    )
    tool_workflow_subgraph_builder.add_conditional_edges(
                            source="tool_review", 
                            path=router_after_tool_review,
                            path_map={"finished":END}
    )
    
    tool_workflow_subgraph_builder.add_edge("tools", END)

    compiled_tool_workflow_subgraph = tool_workflow_subgraph_builder.compile(checkpointer=True)


    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("tool_workflow", compiled_tool_workflow_subgraph)
    graph_builder.add_node("memory_write", memory_write_node)
    graph_builder.add_edge(START, "model")
    graph_builder.add_conditional_edges(
                            source="model", 
                            path=tools_condition, 
                            path_map={"tools":"tool_workflow", "__end__":"memory_write"}
    )
    graph_builder.add_edge("tool_workflow", "model")
    graph_builder.add_edge("memory_write", END)
    
    compiled_resume_agent = graph_builder.compile(checkpointer=checkpointer, store=store)
    return compiled_resume_agent

def create_rag_agent(model, checkpointer=None, store=None):
    tools = [query_rag]
    model_node = ModelNode(model.bind_tools(tools), system_prompt="你是一个RAG检索助手。需要从RAG资料库检索相关数据时必须调用query_rag工具，只能依据工具返回的内容回答。")
    tool_node = ToolNode(tools)
    
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_edge(START, "model")
    graph_builder.add_conditional_edges("model", tools_condition, {"tools":"tools", "__end__":END})
    graph_builder.add_edge("tools", "model")
    
    rag_agent = graph_builder.compile(checkpointer=checkpointer, store=store)
    return rag_agent
    
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:33:47 2026

@author: huzhen
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from nodes import ModelNode
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from state import AgentState


def create_task_tool(subagents):
    agents_by_name = {}
    for subagent in subagents:
        name = subagent["name"]
        agents_by_name[name] = subagent
    
    agent_description_lines = []
    for name,subagent in agents_by_name.items():
        line = "- {}: {}".format(name,subagent["description"])
        agent_description_lines.append(line)
    
    task_tool_description = "把完整、可独立执行的复杂任务委派给专业子Agent。\n可用子Agent：\n{}".format("\n".join(agent_description_lines))
    
    @tool("task",description=task_tool_description)
    def task(description: str, subagent_type: str):
        subagent = agents_by_name[subagent_type]
        subagent_state = {
                    "messages": [HumanMessage(content=description)],
                    "model_call_count": 0
        }
        result = subagent["runnable"].invoke(input=subagent_state)
        
        final_content = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                final_content = message.content
                break
        if not final_content:
            raise ValueError("子agent没有返回有效的AIMessage")
        return final_content
    
    return task
        
    
def create_supervisor_agent(model, subagents, checkpointer=None, store=None):
    task_tool = create_task_tool(subagents)

    supervisor_model = model.bind_tools([task_tool])

    model_node = ModelNode(
        supervisor_model,
        system_prompt=(
            "你是求职助手Supervisor。"
            "遇到需要专业处理的任务时，必须通过task工具委派给合适的子Agent。"
            "子Agent返回结果后，由你整理成最终回答。"
        ),
    )

    tool_node = ToolNode([task_tool])

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "model")
    graph_builder.add_conditional_edges(
        "model",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    graph_builder.add_edge("tools", "model")

    return graph_builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
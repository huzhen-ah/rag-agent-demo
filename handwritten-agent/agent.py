#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:48:47 2026

@author: huzhen
"""

import uuid
from graph import StateGraph,START,END
from state import AgentState
from nodes import ModelNode, ToolNode, ToolArgsCompletionNode, ToolReviewNode
from routers import router_after_model
from hitl import Command
import json



class Agent:
    def __init__(self, chat_model, register, system_prompt, checkpointer, tool_hitl_policy, max_steps=5):
        self.chat_model = chat_model
        self.register = register
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.tool_hitl_policy = tool_hitl_policy
        self.max_steps = max_steps
        self.model_node = ModelNode(self.chat_model,self.register.get_tool_definitions())
        self.tool_node = ToolNode(self.register)
        self.tool_args_completion_node = ToolArgsCompletionNode(self.register, self.tool_hitl_policy)
        self.tool_review_node = ToolReviewNode(tool_hitl_policy)
        self.compiled_graph = self.init_graph()
        
    def init_graph(self):
        state_graph = StateGraph(AgentState)
        state_graph.add_node("model_node", self.model_node)
        state_graph.add_node("tool_node", self.tool_node)
        state_graph.add_node("tool_args_completion_node", self.tool_args_completion_node)
        state_graph.add_node("tool_review_node", self.tool_review_node)
        state_graph.add_edge(START, "model_node")
        state_graph.add_edge("tool_review_node", "tool_node")
        state_graph.add_edge("tool_node", "model_node")
        
        model_path_map = {
                        "need_tools":"tool_args_completion_node",
                        "finished":END
                   }
        
        state_graph.add_conditional_edges("model_node", router_after_model,model_path_map)
        
        tool_args_completion_path_map = {
                        "needs_more_args" : "tool_args_completion_node",
                        "tool_review_node" : "tool_review_node"
                   }
        
        state_graph.add_conditional_edges("tool_args_completion_node", self.tool_args_completion_node.router_after_toolArgsCompletionNode, tool_args_completion_path_map)
        
        return state_graph.compile(self.checkpointer)
    
    def create_initial_state(self):
        system_message = {
                            "id":r"msg_{}".format(uuid.uuid4().hex),
                            "role":"system",
                            "content":self.system_prompt
                         }
        agent_state = {
                            "messages":[system_message],
                            "model_call_count":0
                      }
        return agent_state
    
    def invoke(self, user_input, agent_state, thread_id, checkpoint_id):
        
        if isinstance(user_input, Command):
            return self.compiled_graph.invoke(user_input, None, thread_id, checkpoint_id, recursion_limit=self.max_steps)
        
        
        if user_input is not None:
            user_message = {
                                "id":r"msg_{}".format(uuid.uuid4().hex),
                                "role":"user",
                                "content":user_input
                           }
            input_update = {"messages":[user_message]}
        else:
            input_update = None

        agent_state = self.compiled_graph.invoke(agent_state, input_update, thread_id, checkpoint_id, recursion_limit=self.max_steps)


        return agent_state

        


if __name__ == "__main__":
    from model import LocalChatModel
    from tool_register import Register
    from tools import read_resume_tool, search_project_evidence_tool
    from checkpoint import JsonlCheckpointer

    register = Register()
    register.register(read_resume_tool)
    register.register(search_project_evidence_tool)

    chat_model = LocalChatModel(
        model_path="models/Qwen3-4B",
        device="mps",
    )

    system_prompt = (
        "你是一个求职助手。"
        "需要读取简历或查询项目证据时，"
        "必须调用相应工具。"
        "只能依据工具返回的内容回答，"
        "不得编造经历。"
    )
    
    tool_hitl_policy = {
        "read_resume": ("args_completion","review"),
        "search_project_evidence": ("review",)
    }
    checkpointer = JsonlCheckpointer()

    
    agent = Agent(
        chat_model= chat_model,
        register= register,
        system_prompt= system_prompt,
        checkpointer= checkpointer,
        tool_hitl_policy= tool_hitl_policy,
        max_steps= 5,
    )
    
    user_A_agent_state = agent.create_initial_state()
    thread_id = "thread_{}".format(uuid.uuid4().hex)
    checkpoint_id = None

    while True:
        user_input = input("用户: ")
        user_input = user_input.strip()
        if user_input == "exit":
            break
        if user_input == "":
            user_input = None
        user_A_agent_state = agent.invoke(user_input,user_A_agent_state, thread_id, checkpoint_id)
        # print(user_A_agent_state)
        while "__interrupt__" in user_A_agent_state:
            interrupts = user_A_agent_state["__interrupt__"]
            command_resume = {}
            for interrupt in interrupts:
                print(
                        "interrupt_request: ",
                        json.dumps(
                            interrupt.value,
                            ensure_ascii=False,
                            indent=2,
                            )
                      )
                resume_text = input("请输入resume_value(json): ")
                resume_value = json.loads(resume_text)
                command_resume[interrupt.id] = resume_value
            resume_command = Command(resume=command_resume)
            user_A_agent_state = agent.invoke(resume_command, user_A_agent_state, thread_id, checkpoint_id)
                
        print("assistant: ",user_A_agent_state["messages"][-1]["content"])
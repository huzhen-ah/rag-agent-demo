#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:48:47 2026

@author: huzhen
"""

from runtime import get_state_reducers,apply_updates
import uuid
from graph import StateGraph,START,END
from state import AgentState
from nodes import ModelNode,ToolNode
from routers import router_after_model
class Agent:
    def __init__(self, chat_model, register, system_prompt, max_steps=5):
        self.chat_model = chat_model
        self.register = register
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.key2reducer = get_state_reducers(AgentState)
        self.model_node = ModelNode(self.chat_model,self.register.get_tool_definitions())
        self.tool_node = ToolNode(self.register)
        self.compiled_graph = self.init_graph()
        
    def init_graph(self):
        state_graph = StateGraph(AgentState)
        state_graph.add_node("model_node", self.model_node)
        state_graph.add_node("tool_node", self.tool_node)
        state_graph.add_edge(START, "model_node")
        state_graph.add_edge("tool_node", "model_node")
        
        path_map = {
                        "need_tools":"tool_node",
                        "finished":END
                   }
        
        state_graph.add_conditional_edges("model_node", router_after_model,path_map)
        
        return state_graph.compile()
    
    def create_initial_state(self):
        system_message = {
                            "id":r"msg_{}".format(uuid.uuid4().hex),
                            "role":"system",
                            "content":self.system_prompt
                         }
        agent_state = {
                            "messages":[system_message],
                            "model_steps":0
                      }
        return agent_state
    
    def run(self,user_input,agent_state):
        

        
        user_message = {
                            "id":r"msg_{}".format(uuid.uuid4().hex),
                            "role":"user",
                            "content":user_input
                       }
        agent_state = apply_updates(agent_state, [{"messages":[user_message]}], self.key2reducer)


        agent_state = self.compiled_graph.invoke(agent_state,recursion_limit=self.max_steps)


        return agent_state

        


if __name__ == "__main__":
    from model import LocalChatModel
    from tool_register import Register
    from tools import read_resume_tool, search_project_evidence_tool

    register = Register()
    register.register(read_resume_tool)
    register.register(search_project_evidence_tool)

    chat_model = LocalChatModel(
        model_path="models/Qwen3-1.7B",
        device="mps",
    )

    system_prompt = (
        "你是一个求职助手。"
        "需要读取简历或查询项目证据时，"
        "必须调用相应工具。"
        "只能依据工具返回的内容回答，"
        "不得编造经历。"
    )

    agent = Agent(
        chat_model=chat_model,
        register=register,
        system_prompt=system_prompt,
        max_steps=5,
    )
    
    user_A_agent_state = agent.create_initial_state()
    while True:
        user_input = input("用户: ")
        user_input = user_input.strip()
        if user_input == "exit":
            break
        
        user_A_agent_state = agent.run(user_input,user_A_agent_state)
        print("assistant: ",user_A_agent_state["messages"][-1]["content"])

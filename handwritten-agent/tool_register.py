#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:08:40 2026

@author: huzhen
"""

class Register:
    def __init__(self):
        self.tools = {}
        
    def register(self,tool):
        if tool.name in self.tools:
            raise ValueError("{} 已经存在于注册表中".format(tool.name))
        self.tools[tool.name] = tool
    
    def get(self,name):
        if name not in self.tools:
            raise ValueError("{} 不在注册表中".format(name))
        return self.tools[name]
    
    def get_tool_definitions(self):
        definitions = []
        for tool in self.tools.values():
            definitions.append(tool.definition)
        return definitions
    
    
    

if __name__ == "__main__":
    from tools import read_resume_tool,search_project_evidence_tool

    register = Register()
    register.register(read_resume_tool)
    
    kwargs = {"resume_id":"main"}
    print(register.get("read_resume").run(kwargs))
    
    
    register.register(search_project_evidence_tool)
    
    requirement = "熟练掌握RAG,Milvus"
    kwargs = {"requirement":requirement}
    print(register.get("search_project_evidence").run(kwargs))
    print(register.get_tool_definitions())
    
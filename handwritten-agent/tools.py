#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:54:10 2026

@author: huzhen
"""
from transformers.utils import get_json_schema
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

class ToolInvocationException(Exception):
    pass

class ToolExecutionException(Exception):
    pass

class ResumeNotFoundException(ToolExecutionException):
    pass

class Tool:
    def __init__(self,function):
        self.function = function
        self.definition = get_json_schema(function)
        self.definition["function"]["parameters"]["additionalProperties"] = False
        self.name = self.definition["function"]["name"]
        
    def validate_parameters_pre_fun(self,instance):
        try:
            validate(instance=instance, schema=self.definition["function"]["parameters"])
        except ValidationError as error:
            raise ToolInvocationException("Tool-{}校验参数失败: {}".format(self.name,error.message)) from error
        
    def run(self,arguments):
        self.validate_parameters_pre_fun(arguments)
        return self.function(**arguments)
        
        
def read_resume(resume_id: str) -> str:
    """
    根据简历ID读取简历
    
    Args:
        resume_id: 需要读取的简历ID
    """
    
    resume_files = {
        "main":r"data/resumes/resume_main.md",
        "dialogue_nlp":r"data/resumes/resume_dialogue_nlp.md",
        "rag_agent":r"data/resumes/resume_rag_agent.md"
        }
    
    if resume_id not in resume_files:
        raise ResumeNotFoundException("resume_id: {} 不存在".format(resume_id))
        
    resume_file = resume_files[resume_id]
    with open(resume_file,"r",encoding="utf8") as f:
        data = f.read()
    
    return data

def search_project_evidence(requirement: str) -> list[dict]:
    """
    搜索项目相关证据
    
    Args:
        requirement: 一项项目要求
    """
    with open(r"data/evidence/project_evidence.json","r",encoding="utf8") as f:
        project_evidences = json.load(f)
    ret = []
    requirement = requirement.lower()
    for proj_evi in project_evidences:
        matched_keywords = []
        for word in proj_evi["keywords"]:
            if word.lower() in requirement:
                matched_keywords.append(word)
        if len(matched_keywords) == 0:
            continue
        ret.append({"project":proj_evi["project"],"matched_keywords":matched_keywords,"evidence":proj_evi["evidence"]})
    return ret


    
    
    
        
    
        
        

read_resume_tool = Tool(function=read_resume)
search_project_evidence_tool = Tool(function=search_project_evidence)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:07:56 2026

@author: huzhen
"""

from langchain_core.tools import tool
import json

@tool(parse_docstring=True)
def read_resume(resume_id: str) -> str:
    """
    根据简历ID读取简历。
    
    Args:
        resume_id: 需要读取的简历ID。
    """
    
    resume_files = {
        "main": "data/resumes/resume_main.md",
        "dialogue_nlp": "data/resumes/resume_dialogue_nlp.md",
        "rag_agent": "data/resumes/resume_rag_agent.md",
    }

    if resume_id not in resume_files:
        raise ValueError("resume_id: {} 不存在".format(resume_id))

    resume_file = resume_files[resume_id]

    with open(resume_file, "r", encoding="utf8") as f:
        return f.read()

@tool(parse_docstring=True)    
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
 
@tool(parse_docstring=True)
def update_user_profile(
            preferred_city: str | None = None,
            target_role: str | None = None,
            skills: list[str] | None = None,
            years_of_experience: int | None = None,
            fields_to_delete: list[str] | None = None
):
    """
    提取用户明确表达的长期求职信息。

    Args:
        preferred_city: 用户长期偏好的求职城市。
        target_role: 用户长期目标岗位。
        skills: 用户明确拥有的技能。
        years_of_experience: 用户明确说明的工作年限。
        fields_to_delete: 用户明确要求遗忘的字段名称。
    """
    profile_updates = {
                            "preferred_city" : preferred_city,
                            "target_role" : target_role,
                            "skills" : skills,
                            "years_of_experience" : years_of_experience
                      }
    
    if fields_to_delete is None:
        fields_to_delete = []
    
    unknown_fields = set(fields_to_delete) - set(profile_updates)
    if unknown_fields:
        raise ValueError("不能删除未知的长期记忆字段:{}".format(unknown_fields))
    profile_updates = {k:v for k,v in profile_updates.items() if v is not None}
    
    conflicting_fields = set(fields_to_delete).intersection(set(profile_updates))
    if conflicting_fields:
        raise ValueError("字段不能同时删除与更新: {}".format(conflicting_fields))
        
    ret = {"updates":profile_updates,"fields_to_delete":fields_to_delete}
    return ret
  
if __name__ == "__main__":
    resume_content = read_resume.invoke({"resume_id":"main"})
    # print(resume_content)
    
    from langchain_core.utils.function_calling import convert_to_openai_tool
    
    definition = convert_to_openai_tool(read_resume)
    print(json.dumps(definition,ensure_ascii=False,indent=4))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:31:25 2026

@author: huzhen
"""

def calculate_job_fit_score(assessments):
    
    # assessments = [
    #     {
    #         "category": "hard_required",
    #         "status": "matched",
    #     },
    #     {
    #         "category": "hard_required",
    #         "status": "partial",
    #     },
    #     {
    #         "category": "hard_required",
    #         "status": "missing",
    #     },
    #     {
    #         "category": "bonus",
    #         "status": "matched",
    #     },
    # ]
    
    if len(assessments) == 0:
        raise ValueError("匹配数据不能为空")
    hard_skills_required_num = 0
    bonus_skills_num = 0
    
    hard_skills_matched_num = 0
    bonus_skills_matched_num = 0
    for _ in assessments:
        category = _["category"]
        status = _["status"]
        if category == "hard_required":
            hard_skills_required_num += 1
            if status == "matched":
                hard_skills_matched_num += 1
            elif status == "partial":
                hard_skills_matched_num += 0.5
            elif status == "missing":
                continue
            else:
                raise ValueError("未知status:{}".format(status))
        elif category == "bonus":
            bonus_skills_num += 1
            if status == "matched":
                bonus_skills_matched_num += 1
            elif status == "partial":
                bonus_skills_matched_num += 0.5
            elif status == "missing":
                continue
            else:
                raise ValueError("未知status:{}".format(status))
        else:
            raise ValueError("未知类别:{}".format(category))
    if hard_skills_required_num == 0 and bonus_skills_num == 0:
        raise ValueError("硬性要求与加分项不能全空")
    if hard_skills_required_num and bonus_skills_num:
        hard_alpha = 0.8
        bonus_alpha = 0.2
    elif hard_skills_required_num:
        hard_alpha = 1
        bonus_alpha = 0
    else:
        hard_alpha = 0
        bonus_alpha = 1
    if hard_skills_required_num:
        hard_score = int(hard_skills_matched_num / hard_skills_required_num * hard_alpha * 100)
        hard_coverage = hard_skills_matched_num / hard_skills_required_num
    else:
        hard_score = 0
        hard_coverage = 0
    if bonus_skills_num:
        bonus_score = int(bonus_skills_matched_num / bonus_skills_num * bonus_alpha * 100)
        bonus_coverage = bonus_skills_matched_num / bonus_skills_num
    else:
        bonus_score = 0
        bonus_coverage = 0
    score = hard_score + bonus_score
    
    return {"score":score,"hard_coverage":hard_coverage,"bonus_coverage":bonus_coverage}
        
                
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:54:19 2026

@author: huzhen
"""

from dataclasses import dataclass


#frozen=True的意思是实例化之后，不能给实例重新赋值，但只是浅层限制，如果payload是dict,那仍
#可以给payload的内容赋值，即只限于class的直接字段，slots=True的意思是不能增加新字段
@dataclass(frozen=True, slots=True)
class StreamEvent:
    event_type: str
    payload: dict
    

    
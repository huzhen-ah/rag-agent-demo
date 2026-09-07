#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 08:55:29 2026

@author: huzhen
"""

from operator import add
from typing import Annotated
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    model_call_count: Annotated[int, add]
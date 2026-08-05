#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:00:31 2026

@author: huzhen
"""

class ToolInvocationException(Exception):
    pass

class ToolExecutionException(Exception):
    pass

class ResumeNotFoundException(ToolExecutionException):
    pass

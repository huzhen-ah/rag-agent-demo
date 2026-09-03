#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:47:36 2026

@author: huzhen
"""

from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class Send:
    node: str
    arg: Any


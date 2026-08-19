#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:28:00 2026

@author: huzhen
"""

from dataclasses import dataclass, asdict
from typing import Any
from abc import ABC, abstractmethod
import time
from copy import deepcopy
import os
import json


@dataclass
class MemoryItem:
    namespace:tuple[str,...]
    key: str
    value: dict[str, Any]
    created_at: str
    updated_at: str
    

class BaseStore(ABC):
    @abstractmethod
    def put(self, namespace, key, value):
        pass
    
    @abstractmethod
    def get(self, namespace, key):
        pass
    
    @abstractmethod
    def search(self, namespace):
        pass
    
    @abstractmethod
    def delete(self, namespace, key):
        pass
    
class InMemoryStore(BaseStore):
    def __init__(self):
        self.store = {}
        
    def put(self, namespace, key, value):
        value = deepcopy(value)
        if namespace not in self.store:
            self.store[namespace] = {}
        if key not in self.store[namespace]:
            created_at = updated_at = str(time.time())
            item = MemoryItem(namespace, key, value, created_at, updated_at)
            self.store[namespace][key] = item
        else:
            updated_at = str(time.time())
            self.store[namespace][key].updated_at = updated_at
            self.store[namespace][key].value = value
    
    def get(self, namespace, key):
        if namespace not in self.store:
            return None
        if key not in self.store[namespace]:
            return None
        return deepcopy(self.store[namespace][key])
    
    def search(self, namespace):
        if namespace not in self.store:
            return tuple()
        ret = []
        for key in self.store[namespace]:
            ret.append(deepcopy(self.store[namespace][key]))
        ret = sorted(ret,key=lambda x:x.updated_at,reverse=True)
        return tuple(ret)
    
    def delete(self, namespace, key):
        if namespace not in self.store:
            return 
        if key not in self.store[namespace]:
            return 
        del self.store[namespace][key]
        
class JsonlStore(BaseStore):
    def __init__(self, local_memory_store_file=r"memories/memory.jsonl"):
        self.local_memory_store_file = local_memory_store_file
        self.valid_path(self.local_memory_store_file)
        
    def valid_path(self, file):
        if not os.path.isfile(file):
            fold = os.path.split(file)[0]
            if fold and not os.path.isdir(fold):
                os.makedirs(fold)
            with open(file,"a",encoding="utf8") as _:
                pass

        
    def put(self, namespace, key, value):
        item = self.get(namespace, key)
        time_str = str(time.time())
        if item is None:
            item = MemoryItem(namespace, key, value, time_str, time_str)
        else:
            item.updated_at = time_str
            item.value = value
        item_dict = asdict(item)
        item_dict["operation"] = "put"
        item_str = json.dumps(item_dict, ensure_ascii=False)
        with open(self.local_memory_store_file, "a", encoding="utf8") as f:
            f.write(item_str+"\n")
            
    def get(self, namespace, key):
        item = None
        
        with open(self.local_memory_store_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["namespace"] = tuple(data["namespace"])
                if data["namespace"] == namespace and data["key"] == key:
                    if data["operation"] == "delete":
                        item = None
                    else:
                        item = data
        if item:
            del item["operation"]
            item = MemoryItem(**item)
        return item
                
    
    def search(self, namespace):
        key2item = {}
        with open(self.local_memory_store_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["namespace"] = tuple(item["namespace"])
                if item["namespace"] == namespace:
                    key = item["key"]
                    key2item[key] = item
                    if item["operation"] == "delete":
                        del key2item[key]
                    else:
                        del key2item[key]["operation"]
        if len(key2item) == 0:
            return tuple()
        items = sorted(key2item.values(), key=lambda x : x["updated_at"], reverse=True)
        items = tuple(MemoryItem(**item) for item in items)
        return items
        
    
    def delete(self, namespace, key):
        item = self.get(namespace, key)
        if item:
            with open(self.local_memory_store_file, "a", encoding="utf8") as f:
                data = {"operation":"delete", "namespace":list(item.namespace), "key":item.key}
                data = json.dumps(data, ensure_ascii=False)
                f.write(data+"\n")
    
    
    
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:53:53 2026

@author: huzhen
"""

from typing import Any, NamedTuple, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os
from hitl import Interrupt


class StateSnapshot(NamedTuple):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    super_step: int
    state: dict[str, Any]
    next_node_names: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class PendingWrite:
    task_id: str
    channel: Literal["update", "interrupt", "error", "resume"]
    value: Any
    
    
class Checkpointer(ABC):
    @abstractmethod
    def put(self, snapshot: StateSnapshot) -> None:
        pass

    @abstractmethod
    def get(self, thread_id: str, checkpoint_ns:str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        pass

    @abstractmethod
    def list(self, thread_id: str, checkpoint_ns:str) -> tuple[StateSnapshot, ...]:
        #同一 thread 的快照，按 created_at 从新到旧返回
        pass

    @abstractmethod
    def put_writes(self,thread_id, checkpoint_ns, checkpoint_id, writes) -> None:
        pass
    
    @abstractmethod
    def get_writes(self, thread_id, checkpoint_ns, checkpoint_id) -> tuple[PendingWrite, ...]:
        pass
    
    
class InMemoryCheckpointer(Checkpointer):
    def __init__(self):
        self.checkpoints: dict[str, dict[str, dict[str, StateSnapshot]]] = {} #{thread_id:checkpoint_ns:checkpoint_id:snapshot}
        self.checkpoint_key = tuple[str, str, str]
        self.write_key = tuple[str, str]
        self.pending_writes: dict[self.checkpoint_key,dict[self.write_key,PendingWrite]] = {}
        
    def put(self,snapshot: StateSnapshot) -> None:
        thread_id = snapshot.thread_id
        checkpoint_ns = snapshot.checkpoint_ns
        checkpoint_id = snapshot.checkpoint_id
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = {}
        if checkpoint_ns not in self.checkpoints[thread_id]:
            self.checkpoints[thread_id][checkpoint_ns] = {}
        self.checkpoints[thread_id][checkpoint_ns][checkpoint_id] = deepcopy(snapshot)

    def get(self,thread_id: str, checkpoint_ns:str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        if thread_id not in self.checkpoints:
            return None
        if checkpoint_ns not in self.checkpoints[thread_id]:
            return None
        if checkpoint_id:
            if checkpoint_id not in self.checkpoints[thread_id][checkpoint_ns]:
                return None
            return deepcopy(self.checkpoints[thread_id][checkpoint_ns][checkpoint_id])
        else:
            created_ats = [(cp.created_at,cp.checkpoint_id) for _,cp in self.checkpoints[thread_id][checkpoint_ns].items()]
            newest_checkpoint_id = sorted(created_ats,key=lambda x : x[0])[-1][1]
            return deepcopy(self.checkpoints[thread_id][checkpoint_ns][newest_checkpoint_id])

    def list(self,thread_id: str, checkpoint_ns:str) -> tuple[StateSnapshot,...]:
        #同一 thread 的快照，按 created_at 从新到旧返回
        if thread_id not in self.checkpoints:
            return tuple()
        if checkpoint_ns not in self.checkpoints[thread_id]:
            return tuple()
        created_ats = [(cp.created_at,cp.checkpoint_id) for _,cp in self.checkpoints[thread_id][checkpoint_ns].items()]
        sorted_checkpoint_ids = [_[1] for _ in sorted(created_ats,key=lambda x : x[0],reverse=True)]
        return tuple(deepcopy(self.checkpoints[thread_id][checkpoint_ns][checkpoint_id]) for checkpoint_id in sorted_checkpoint_ids)

    def put_writes(self, thread_id, checkpoint_ns, checkpoint_id, writes) -> None:
        checkpoint_key = (thread_id, checkpoint_ns, checkpoint_id)
        if checkpoint_key not in self.pending_writes:
            self.pending_writes[checkpoint_key] = {}
        for write in writes:
            write_key = (write.task_id,write.channel)
            self.pending_writes[checkpoint_key][write_key] = deepcopy(write)
            
    def get_writes(self, thread_id, checkpoint_ns, checkpoint_id) -> tuple[PendingWrite, ...]:
        checkpoint_key = (thread_id, checkpoint_ns, checkpoint_id)
        if checkpoint_key not in self.pending_writes:
            return ()
        return tuple(deepcopy(write) for write in self.pending_writes[checkpoint_key].values())
    
    
class JsonlCheckpointer(Checkpointer):
    def __init__(self,local_checkpoint_file=r"checkpoints/checkpoints.jsonl", local_pending_writes_file=r"checkpoints/pending_writes.jsonl"):
        self.local_checkpoint_file = local_checkpoint_file
        self.local_pending_writes_file = local_pending_writes_file
        self.valid_path(self.local_checkpoint_file)
        self.valid_path(self.local_pending_writes_file)
        
    def valid_path(self,file):
        if not os.path.isfile(file):
            fold = os.path.split(file)[0]
            if fold and not os.path.isdir(fold):
                os.makedirs(fold)
            with open(file,"a",encoding="utf8") as _:
                pass

    def _serialize(self,snapshot: StateSnapshot) -> str:
        snapshot_dict = snapshot._asdict()
        return json.dumps(snapshot_dict,ensure_ascii=False)

    def _deserialize(self,snapshot_json: str) -> StateSnapshot:
        snapshot_dict = json.loads(snapshot_json)
        snapshot_dict["next_node_names"] = tuple(snapshot_dict["next_node_names"])
        snapshot = StateSnapshot(**snapshot_dict)
        return snapshot

    def put(self,snapshot: StateSnapshot) -> None:
        snapshot_jsonl = self._serialize(snapshot)
        with open(self.local_checkpoint_file,"a",encoding="utf8") as f:
            f.write(snapshot_jsonl+"\n")

    def get(self,thread_id: str, checkpoint_ns:str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        ret = None

        with open(self.local_checkpoint_file,"r",encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                snapshot = self._deserialize(line)
                if thread_id == snapshot.thread_id:
                    if checkpoint_ns == snapshot.checkpoint_ns:
                        if checkpoint_id:
                            if checkpoint_id == snapshot.checkpoint_id:
                                return snapshot
                        else:
                            ret = snapshot
        return ret

    def list(self,thread_id: str, checkpoint_ns:str) -> tuple[StateSnapshot,...]:
        ret = []
        with open(self.local_checkpoint_file,"r",encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snapshot = self._deserialize(line)
                if thread_id == snapshot.thread_id and checkpoint_ns == snapshot.checkpoint_ns:
                    ret.append(snapshot)
        if ret:
            ret = ret[::-1]
        return tuple(ret)
    
    def _serialize_writes(self, thread_id, checkpoint_ns, checkpoint_id, writes):
        ret = []
        for write in writes:
            task_id = write.task_id
            channel = write.channel
            value = write.value
            
            if channel == "update":#AgentStateUpdate
                pass
            elif channel == "resume":
                pass
            elif channel == "interrupt":
                value = [{"value" : _.value, "id" : _.id} for _ in value]
            elif channel == "error":
                value = {"type":type(value).__name__,"message":str(value)}
            else:
                raise TypeError("channel必须是update|resume|interrupt|error")
            
            data = {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "channel": channel,
                        "value":value
                   }
                
            data_json = json.dumps(data,ensure_ascii=False)
            ret.append(data_json+"\n")
        return ret
    
    def put_writes(self, thread_id, checkpoint_ns, checkpoint_id, writes) -> None:
        json_writes = self._serialize_writes(thread_id, checkpoint_ns, checkpoint_id, writes)
        with open(self.local_pending_writes_file, "a", encoding="utf8") as f:
            f.writelines(json_writes)
            
    def _deserialize_writes(self,writes):
        ret = []
        for write in writes.values():
            task_id = write["task_id"]
            channel = write["channel"]
            value = write["value"]
            
            if channel == "update":
                pass
            elif channel == "resume":
                pass
            elif channel == "interrupt":
                value = tuple(Interrupt(v["value"], v["id"]) for v in value)
            elif channel == "error":
                value = RuntimeError("{}:{}".format(value["type"],value["message"]))
            else:
                raise TypeError("channel必须是update|resume|interrupt|error")
            ret.append(PendingWrite(task_id, channel, value))
        return tuple(ret)
    
    def get_writes(self, thread_id, checkpoint_ns, checkpoint_id) -> tuple[PendingWrite, ...]:
        writes = {}
        
        with open(self.local_pending_writes_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                if data["thread_id"] == thread_id and data["checkpoint_ns"] == checkpoint_ns and data["checkpoint_id"] == checkpoint_id:
                    write_key = (data["task_id"],data["channel"])
                    writes[write_key] = data
        if not writes:
            return ()
        
        return self._deserialize_writes(writes)
        
                
                    
                
                

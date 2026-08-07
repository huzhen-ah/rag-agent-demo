#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:53:53 2026

@author: huzhen
"""

from typing import Any, NamedTuple
from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os

class StateSnapshot(NamedTuple):
    thread_id: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    super_step: int
    state: dict[str, Any]
    next_node_names: tuple[str, ...]
    created_at: str


class Checkpointer(ABC):
    @abstractmethod
    def put(self,snapshot: StateSnapshot) -> None:
        pass

    @abstractmethod
    def get(self,thread_id: str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        pass


    @abstractmethod
    def list(self,thread_id: str) -> tuple[StateSnapshot,...]:
        #同一 thread 的快照，按 created_at 从新到旧返回
        pass



class InMemoryCheckpointer(Checkpointer):
    def __init__(self):
        self.checkpoints: dict[str, dict[str, StateSnapshot]] = {}

    def put(self,snapshot: StateSnapshot) -> None:
        thread_id = snapshot.thread_id
        checkpoint_id = snapshot.checkpoint_id
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = {}
        self.checkpoints[thread_id][checkpoint_id] = deepcopy(snapshot)

    def get(self,thread_id: str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        if thread_id not in self.checkpoints:
            return None
        if checkpoint_id:
            if checkpoint_id not in self.checkpoints[thread_id]:
                return None
            return deepcopy(self.checkpoints[thread_id][checkpoint_id])
        else:
            created_ats = [(cp.created_at,cp.checkpoint_id) for _,cp in self.checkpoints[thread_id].items()]
            newest_checkpoint_id = sorted(created_ats,key=lambda x : x[0])[-1][1]
            return deepcopy(self.checkpoints[thread_id][newest_checkpoint_id])

    def list(self,thread_id: str) -> tuple[StateSnapshot,...]:
        #同一 thread 的快照，按 created_at 从新到旧返回
        if thread_id not in self.checkpoints:
            return tuple()

        created_ats = [(cp.created_at,cp.checkpoint_id) for _,cp in self.checkpoints[thread_id].items()]
        sorted_checkpoint_ids = [_[1] for _ in sorted(created_ats,key=lambda x : x[0],reverse=True)]
        return tuple(deepcopy(self.checkpoints[thread_id][checkpoint_id]) for checkpoint_id in sorted_checkpoint_ids)

class JsonlCheckpointer(Checkpointer):
    def __init__(self,local_checkpoint_file=r"checkpoints/checkpoints.jsonl"):
        self.local_checkpoint_file = local_checkpoint_file
        self.valid_path(self.local_checkpoint_file)

    def valid_path(self,file):
        if not os.path.isfile(file):
            fold = os.path.split(self.local_checkpoint_file)[0]
            if fold and not os.path.isdir(fold):
                os.makedirs(fold)
            with open(file,"a",encoding="utf8") as f:
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

    def get(self,thread_id: str, checkpoint_id: str | None = None) -> StateSnapshot | None:
        ret = None

        with open(self.local_checkpoint_file,"r",encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                snapshot = self._deserialize(line)
                if thread_id == snapshot.thread_id:
                    if checkpoint_id:
                        if checkpoint_id == snapshot.checkpoint_id:
                            return snapshot
                    else:
                        ret = snapshot
        return ret



    def list(self,thread_id: str) -> tuple[StateSnapshot,...]:
        ret = []
        with open(self.local_checkpoint_file,"r",encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snapshot = self._deserialize(line)
                if thread_id == snapshot.thread_id:
                    ret.append(snapshot)
        if ret:
            ret = ret[::-1]
        return tuple(ret)

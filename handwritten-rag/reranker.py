#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:02:16 2026

@author: huzhen
"""

from sentence_transformers import CrossEncoder
import torch
import numpy as np

class Reranker:
    def __init__(self, model_path, device):
        self.model = CrossEncoder(
            model_path,
            device=str(device),
        )
   
    def rerank(self,query,rrf_results,k):
        if len(rrf_results) == 0:
            return []
        pairs = [
            (query, result["chunk"].content)
            for result in rrf_results
        ]
    
        scores = self.model.predict(pairs)
        indexes = np.argsort(scores,axis=-1)[::-1][:k]
        ret = []
        for i,index in enumerate(indexes):
            _ = {"chunk":rrf_results[index]["chunk"],"score":scores[index],"rank":i+1}
            ret.append(_)
        return ret
  
    
if __name__ == "__main__":
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    reranker_model_path = r"models/Qwen3-Reranker-0.6B"
    
    reranker = Reranker(reranker_model_path, device)
    

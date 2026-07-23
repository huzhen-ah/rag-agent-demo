#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:01:16 2026

@author: huzhen
"""


import numpy as np

class DenseRetrieval:
    def __init__(self,embedder,chunks,chunk_embeddings):
        self.embedder = embedder
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        
    def search_documents(self,query,k=5):
        assert k > 0
        assert k <= len(self.chunks)
        
        query_embedding = self.embedder.embed_query(query)
        scores = self.chunk_embeddings @ query_embedding
        top_k_index = np.argpartition(-scores,k-1)[:k]
        top_k_scores = scores[top_k_index]
        index_scores = sorted(zip(top_k_index,top_k_scores),key=lambda x : x[1],reverse=True)
        ret = []
        for i,index_score in enumerate(index_scores):
            index,score = index_score
            _ = {"chunk":self.chunks[index],"score":score,"rank":i+1}
            ret.append(_)
        return ret
    
    
        
        
    

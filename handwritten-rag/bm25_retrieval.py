#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:59:08 2026

@author: huzhen
"""
import jieba
import numpy as np
import os
import pickle

class BM25Retrieval:
    def __init__(self,chunks,bm25_config_path=r"bm25/bm25_config.pkl"):
        if chunks:
            self.chunks = chunks
            self.N = len(chunks)
            self.chunk_words = self.tokenizer_chunks()
            self.chunk_lens = [len(words) for words in self.chunk_words]
            self.avg_len = sum(len(words) for words in self.chunk_words) / self.N
            self.chunk_tfs = self.calculate_chunk_tfs()
            self.chunk_idfs = self.calculate_chunk_idf()
            self.word2chunks = self.calculate_word2chunks()
            self.save_bm25_config(bm25_config_path)
        else:
            self.load_bm25_config(bm25_config_path)
        
    def load_bm25_config(self,bm25_config_path):
        
        with open(bm25_config_path,"rb") as f:
            config = pickle.load(f)
        self.chunks = config["chunks"]
        self.N = config["N"]
        self.chunk_words = config["chunk_words"]
        self.chunk_lens = config["chunk_lens"]
        self.avg_len = config["avg_len"]
        self.chunk_tfs = config["chunk_tfs"]
        self.chunk_idfs = config["chunk_idfs"]
        self.word2chunks = config["word2chunks"]
        
    def save_bm25_config(self,bm25_config_path):
        folder = os.path.dirname(bm25_config_path)

        if folder and not os.path.isdir(folder):
            os.makedirs(folder)

        config = {
                     "chunks":self.chunks,
                     "N":self.N,
                     "chunk_words":self.chunk_words,
                     "chunk_lens":self.chunk_lens,
                     "avg_len":self.avg_len,
                     "chunk_tfs":self.chunk_tfs,
                     "chunk_idfs":self.chunk_idfs,
                     "word2chunks":self.word2chunks
                 }
        with open(bm25_config_path,"wb") as f:
            pickle.dump(config,f)
            
    def tokenizer_chunks(self):
        chunk_words = []
        for chunk in self.chunks:
            words = [w.strip() for w in jieba.lcut(chunk.content) if w.strip()]
            chunk_words.append(words)
        return chunk_words
           
    def calculate_chunk_tfs(self):
        chunk_tfs = []
        for words in self.chunk_words:
            tfs = {}
            for w in words:
                if w not in tfs:
                    tfs[w] = 1
                else:
                    tfs[w] += 1
            chunk_tfs.append(tfs)
        return chunk_tfs
    
    def calculate_chunk_idf(self):
        #ln[1 + (N - DF(t) + 0.5) / (DF(t) + 0.5)]
        IDF = {}
        for tfs in self.chunk_tfs:
            for w in tfs:
                if w not in IDF:
                    IDF[w] = 1
                else:
                    IDF[w] += 1
        for w,freq in IDF.items():
            IDF[w] = np.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
        return IDF
    
    def calculate_word2chunks(self):
        word2chunks = {}
        for i,tfs in enumerate(self.chunk_tfs):
            for w in tfs:
                if w not in word2chunks:
                    word2chunks[w] = set([i])
                else:
                    word2chunks[w].add(i)
        return word2chunks
    
    def search_documents(self,query,k=5,k1=1.5,b=0.75):
        # Σ IDF_BM25(t)
        #   ×
        #   TF(t,d) × (k1 + 1)
        #   /
        #   [TF(t,d) + k1 × (1 - b + b × |d| / avgdl)]
        query_words = [w.strip() for w in jieba.lcut(query) if w.strip()]
        if len(query_words) == 0:
            return []
        optional_chunk_ids = set()
        optional_query_words = []
        for w in set(query_words):
            if w not in self.word2chunks:
                continue
            else:
                optional_query_words.append(w)
                optional_chunk_ids = optional_chunk_ids.union(self.word2chunks[w])
        optional_chunk_ids = list(optional_chunk_ids)
        if len(optional_chunk_ids) == 0:
            return []
        bm25_scores = []
        for chunk_id in optional_chunk_ids:
            scores = 0
            for w in optional_query_words:
                tf_w = self.chunk_tfs[chunk_id].get(w,0)
                if tf_w == 0:
                    continue
                abs_d = self.chunk_lens[chunk_id]
                score = (self.chunk_idfs[w] * tf_w * (k1 + 1)) / (tf_w + k1 * (1 - b + b * abs_d / self.avg_len))
                scores += score
            bm25_scores.append(scores)
            
        
        index_scores = sorted(zip(optional_chunk_ids,bm25_scores),key=lambda x : x[1],reverse=True)[:k]
        ret = []
        for i,index_score in enumerate(index_scores):
            index,score = index_score
            _ = {"chunk":self.chunks[index],"score":score,"rank":i+1}
            ret.append(_)
        return ret
                
            
        
        
    
            
                
        
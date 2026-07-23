#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:43:04 2026

@author: huzhen
"""
import torch
import pickle
import json
from rag import RAG
from embedding import Embedder
from dense_retrieval import DenseRetrieval
from bm25_retrieval import BM25Retrieval
from reranker import Reranker


def recall_at_k(retrieved_ids, relevant_ids, k):
    if len(retrieved_ids) == 0:
        return 0
    k_ids = retrieved_ids[:k]
    correct_ids = set(k_ids) & set(relevant_ids)
    return len(correct_ids) / len(relevant_ids)

def reciprocal_rank(retrieved_ids, relevant_ids):
    if len(retrieved_ids) == 0:
        return 0
    first_corrent_id_index = None
    for index,iid in enumerate(retrieved_ids):
        if iid in relevant_ids:
            first_corrent_id_index = index + 1
            break
    if first_corrent_id_index is None:
        return 0
    return 1 / first_corrent_id_index 

if __name__ == "__main__":
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    reranker_model_path = r"models/Qwen3-Reranker-0.6B"
    dense_dataset_path = r"dataset/data.pkl"
    bm25_config_path = r"bm25/bm25_config.pkl"
    with open(dense_dataset_path,"rb") as f:
        dataset = pickle.load(f)
    embedder = Embedder(embedding_model_path, device)
    generator = None
    
    
    denseRetrieval = DenseRetrieval(embedder, dataset["chunks"], dataset["chunk_embeddings"])
    bm25Retrieval = BM25Retrieval(dataset["chunks"],bm25_config_path)
    reranker = Reranker(reranker_model_path, device)
    rag = RAG(denseRetrieval, bm25Retrieval, reranker, generator)
    total_rr = []
    recall_1_scores = []
    recall_3_scores = []
    recall_5_scores = []
    c = 60
    k = 5
    with open(r"dataset/retrieval_eval.jsonl","r",encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            relevant_ids = data["relevant_chunk_ids"]
            dense_chunks_ret = denseRetrieval.search_documents(question,k*3)
            bm25_chunks_ret = bm25Retrieval.search_documents(question,k*3)
            top_k_chunks = rag.rrf(dense_chunks_ret, bm25_chunks_ret,c,k*2)
            top_k_chunks = rag.reranker.rerank(question, top_k_chunks, k)
            retrieved_ids = [chunk["chunk"].chunk_id for chunk in top_k_chunks]
            recall_1_scores.append(recall_at_k(retrieved_ids, relevant_ids, 1))
            recall_3_scores.append(recall_at_k(retrieved_ids, relevant_ids, 3))
            recall_5_scores.append(recall_at_k(retrieved_ids, relevant_ids, 5))
            total_rr.append(reciprocal_rank(retrieved_ids, relevant_ids))
    print("MRR: ",sum(total_rr) / len(total_rr))
    print("Recall@K_1: ",sum(recall_1_scores)/len(recall_1_scores))
    print("Recall@K_3: ",sum(recall_3_scores)/len(recall_3_scores))
    print("Recall@K_5: ",sum(recall_5_scores)/len(recall_5_scores))
            
    

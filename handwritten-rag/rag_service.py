#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:04:05 2026

@author: huzhen
"""

from rag import RAG
import torch
from dense_retrieval import DenseRetrieval
from bm25_retrieval import BM25Retrieval
from reranker import Reranker
from embedding import Embedder
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


def build_rag():
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
    
    
    denseRetrieval = DenseRetrieval(embedder, dataset["chunks"], dataset["chunk_embeddings"])
    bm25Retrieval = BM25Retrieval(dataset["chunks"],bm25_config_path)
    reranker = Reranker(reranker_model_path, device)
    rag = RAG(denseRetrieval, bm25Retrieval, reranker, None)
    return rag

class RetrieveRequest(BaseModel):
    question: str
    
app = FastAPI()
rag = build_rag()

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    results = rag.retrieve(question=request.question, c=60, k=5)
    
    documents = []
    for result in results:
        chunk = result["chunk"]
        rank = result["rank"]
        score = float(result["score"])
        document = {
                        "chunk_id" : chunk.chunk_id,
                        "content" : chunk.content,
                        "source" : chunk.source,
                        "page" : chunk.page,
                        "score" : score,
                        "rank" : rank
                   }
        documents.append(document)
    return {"documents" : documents}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )

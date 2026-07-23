#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:48:54 2026

@author: huzhen
"""

import torch
from embedding import Embedder
from generator import Generator
from dense_retrieval import DenseRetrieval
from bm25_retrieval import BM25Retrieval
from reranker import Reranker
from load import load_documents,cut_documents
from rag import RAG
import pickle
import os

if __name__ == "__main__":
    
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    """
    STEP_1:
        构建数据集
    """
    
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    generate_model_path = r"models/Qwen3-1.7B"
    reranker_model_path = r"models/Qwen3-Reranker-0.6B"
    
    
    dense_dataset_path = r"dataset/data.pkl"
    bm25_config_path = r"bm25/bm25_config.pkl"
    file_paths = [r"documents/THREE_WEEK_PLAN.md"]
    embedder = Embedder(embedding_model_path,device)
    if not os.path.isfile(r"dataset/data.pkl"):

        documents = load_documents(file_paths)
        
        chunks = cut_documents(documents)
        
        chunk_embeddings = embedder.embed_chunks(chunks)
        
        configs = {
                    "chunks":chunks,
                    "chunk_embeddings":chunk_embeddings
                  }
        
        save_fold = "dataset"
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)
        with open(r"{}/data.pkl".format(save_fold),"wb") as f:
            pickle.dump(configs,f)
    """
    STEP_2:
        初始化RAG，并测试
    """
    with open(dense_dataset_path,"rb") as f:
        dataset = pickle.load(f)

    generator = Generator(generate_model_path, device)
    
    
    denseRetrieval = DenseRetrieval(embedder, dataset["chunks"], dataset["chunk_embeddings"])
    bm25Retrieval = BM25Retrieval(dataset["chunks"],bm25_config_path)
    reranker = Reranker(reranker_model_path, device)
    rag = RAG(denseRetrieval, bm25Retrieval, reranker, generator)
    k = 5
    c = 60
    while True:
        question = str(input("我:")).strip()
        if not question:
            continue
        if question == "exit":
            break
        answer = rag.answer_question(question,c,k)
        print("assistant: ",answer)

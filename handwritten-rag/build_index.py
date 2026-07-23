#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:31:20 2026

@author: huzhen
"""

import torch
from embedding import Embedder
from load import load_documents,cut_documents
import pickle
import os

if __name__ == "__main__":
    
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model_path = "models/Qwen3-Embedding-0.6B"
    file_paths = [r"documents/THREE_WEEK_PLAN.md"]
    
    
    embedder = Embedder(model_path,device)
    
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

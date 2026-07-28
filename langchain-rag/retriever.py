#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:14:40 2026

@author: huzhen
"""

from ingest import get_vector_store,create_embeddings


def create_retriever(vector_store,top_k=10,rrf_k=60):
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {
                "k":top_k,
                "ranker_type":"rrf",
                "ranker_params" : {
                    "k":rrf_k
            }
        }
    )
    return retriever


if __name__ == "__main__":
    uri = "http://localhost:19530"
    db_name = "rag_demo"
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    collection_name = "rag_chunks"
    user="root"
    pwd="Milvus"
    embeddings = create_embeddings(embedding_model_path)
    vector_store = get_vector_store(uri, user, pwd, db_name, collection_name, embeddings)
    retriever = create_retriever(vector_store)
    query = "第一周主要学习什么"
    documents = retriever.invoke(query)
    print(documents[0])
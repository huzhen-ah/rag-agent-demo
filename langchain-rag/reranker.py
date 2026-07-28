#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:52:40 2026

@author: huzhen
"""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)


def create_rerank_retriever(
    retriever,
    reranker_model_path,
    top_n=5,
    device="mps",
):
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=reranker_model_path,
        model_kwargs={
            "device": device,
            "local_files_only": True,
        },
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=top_n,
    )

    rerank_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=reranker,
    )

    return rerank_retriever

if __name__ == "__main__":
    from ingest import create_embeddings, get_vector_store
    from retriever import create_retriever

    uri = "http://localhost:19530"
    db_name = "rag_demo"
    collection_name = "rag_chunks"
    embedding_model_path = "models/Qwen3-Embedding-0.6B"
    reranker_model_path = "models/Qwen3-Reranker-0.6B"
    user = "root"
    pwd = "Milvus"

    embeddings = create_embeddings(embedding_model_path)

    vector_store = get_vector_store(
        uri,
        user,
        pwd,
        db_name,
        collection_name,
        embeddings,
    )

    retriever = create_retriever(
        vector_store=vector_store,
        top_k=10,
        rrf_k=60,
    )

    rerank_retriever = create_rerank_retriever(
        retriever=retriever,
        reranker_model_path=reranker_model_path,
        top_n=5,
        device="mps",
    )

    query = "第一周主要学习什么"

    documents = rerank_retriever.invoke(query)

    for rank, document in enumerate(documents, start=1):
        print("=" * 80)
        print("rank:", rank)
        print("content:", document.page_content)
        print("metadata:", document.metadata)

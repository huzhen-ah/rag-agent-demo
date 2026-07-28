#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:58:13 2026

@author: huzhen
"""


from ingest import create_database,create_schema,create_index_params,create_collection,load_documents,split_documents,create_embeddings, get_vector_store,ingest_documents
from retriever import create_retriever
from reranker import create_rerank_retriever
from generator import create_chat_model
from rag import create_rag_chain

from glob import glob



if __name__ == "__main__":



    uri = "http://localhost:19530"
    db_name = "rag_demo"
    collection_name = "rag_chunks"
    user="root"
    pwd="Milvus"
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    embeddings = create_embeddings(embedding_model_path)
    embedding_dim = len(embeddings.embed_query("测试"))

    client = create_database(uri, db_name=db_name, user=user, pwd=pwd)

    if not client.has_collection(collection_name):
        """
        STEP_1:
            处理文档
        """

        file_paths = glob(r"documents/*")

        chunk_size = 500
        chunk_overlap = 50
        add_start_index = True

        docs = load_documents(file_paths)
        documents = split_documents(docs, chunk_size, chunk_overlap, add_start_index)

        """
        STEP_2:
            建数据库，建表，把文档存入Milvus
        """
        schema = create_schema(client,dim=embedding_dim)
        index_params = create_index_params(client)
        create_collection(client, collection_name, schema, index_params)


        vector_store = get_vector_store(uri, user, pwd, db_name, collection_name, embeddings)
        ingest_documents(documents, vector_store)
    else:
        vector_store = get_vector_store(uri, user, pwd, db_name, collection_name, embeddings)



    """
    STEP_3:
        创建基础rrf_retriever
    """
    retriever = create_retriever(
        vector_store=vector_store,
        top_k=10,
        rrf_k=60,
    )

    """
    STEP_4:
        创建reranker_retriever
    """
    reranker_model_path = "models/Qwen3-Reranker-0.6B"
    generate_model_path = "models/Qwen3-1.7B"

    device = "mps"

    rerank_retriever = create_rerank_retriever(
        retriever=retriever,
        reranker_model_path=reranker_model_path,
        top_n=5,
        device=device,
    )

    """
    STEP_5:
        搭建rag_chain
    """
    chat_model = create_chat_model(
        model_path=generate_model_path,
        device=device,
    )

    rag_chain = create_rag_chain(
        rerank_retriever=rerank_retriever,
        chat_model=chat_model,
    )

    while True:
        question = input("我：").strip()

        if question == "exit":
            break

        if not question:
            continue

        answer = rag_chain.invoke(question)

        print("助手：", answer)
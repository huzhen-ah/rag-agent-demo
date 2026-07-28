#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:01:29 2026

@author: huzhen
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ingest import create_embeddings, get_vector_store
from retriever import create_retriever
from reranker import create_rerank_retriever
from generator import create_chat_model


def format_documents(documents):
    context = "\n\n".join(
        [
            "参考资料_{}：\n{}".format(
                index,
                document.page_content,
            )
            for index, document in enumerate(documents, start=1)
        ]
    )
    return context


def create_rag_chain(rerank_retriever, chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    你是一个严格依据参考资料回答问题的助手。

                    请遵守以下规则：
                    1. 只能依据提供的参考资料回答。
                    2. 不要编造参考资料中不存在的信息。
                    3. 如果参考资料不足，请回答“根据现有资料无法确定”。

                    参考资料：
                    {context}
                    """,
            ),
            (
                "human",
                "{question}\n/no_think",
            ),
        ]
    )

    rag_chain = (
        {
            "context": rerank_retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    uri = "http://localhost:19530"
    db_name = "rag_demo"
    collection_name = "rag_chunks"

    embedding_model_path = "models/Qwen3-Embedding-0.6B"
    reranker_model_path = "models/Qwen3-Reranker-0.6B"
    generate_model_path = "models/Qwen3-1.7B"

    user = "root"
    pwd = "Milvus"
    device = "mps"

    embeddings = create_embeddings(
        embedding_model_path
    )

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
        device=device,
    )

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
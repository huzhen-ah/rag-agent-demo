#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:01:40 2026

@author: huzhen
"""
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from glob import glob
from pymilvus import MilvusClient,DataType,Function,FunctionType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus,BM25BuiltInFunction
import pypdf


def load_text(file_path):
    documents = []
    with open(file_path,"r",encoding="utf8") as f:
        data = f.read()
    document = Document(page_content=data, metadata={"source":file_path})
    documents.append(document)
    return documents



def load_pdf(file_path):
    documents = []
    reader = pypdf.PdfReader(file_path)
    for i,page in enumerate(reader.pages):
        data = page.extract_text()
        if data is None:
            data = ""
        document = Document(page_content=data, metadata={"source":file_path,"page":i})
        documents.append(document)
    return documents


def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix in [".md",".txt"]:
            _ = load_text(file_path)
            documents.extend(_)
        elif suffix == ".pdf":
            _ = load_pdf(file_path)
            documents.extend(_)
    return documents

def split_documents(docs,chunk_size,chunk_overlap,add_start_index):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,add_start_index=add_start_index)
    documents = text_splitter.split_documents(docs)
    return documents


def create_embeddings(embedding_model_path):
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model_path,
        model_kwargs = {
                "device":"mps",
                "local_files_only":True
            },
        encode_kwargs={
            "prompt":"",
            "batch_size":8,
            "normalize_embeddings":True
            },
        query_encode_kwargs={
            "prompt_name":"query",
            "normalize_embeddings":True
            }
    )
    return embeddings

def create_database(uri,db_name,user,pwd):
    client = MilvusClient(uri=uri,token="{}:{}".format(user,pwd))
    if db_name not in client.list_databases():
        client.create_database(db_name)

    client = MilvusClient(uri=uri,db_name=db_name,user=user,password=pwd)
    return client

def create_schema(client,max_length=65535,dim=1024):
    schema = client.create_schema(auto_id=True,enable_dynamic_field=True)
    schema.add_field(
        field_name = "pk",
        datatype = DataType.INT64,
        is_primary = True
    )

    schema.add_field(
        field_name = "text",
        datatype = DataType.VARCHAR,
        max_length = max_length,
        enable_analyzer = True,
        analyzer_params= {"type":"chinese"}
    )

    schema.add_field(
        field_name = "dense",
        datatype = DataType.FLOAT_VECTOR,
        dim = dim
    )

    schema.add_field(
        field_name = "sparse",
        datatype = DataType.SPARSE_FLOAT_VECTOR
    )

    bm25_function = Function(
        name = "bm25_function",
        function_type = FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse"]
    )
    schema.add_function(bm25_function)

    return schema


def create_index_params(client,M=16,efConstruction=64):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_name = "dense_hnsw_index",
        index_type = "HNSW",
        metric_type = "COSINE",
        params = {
            "M":M,
            "efConstruction":efConstruction
        }
    )

    index_params.add_index(
        field_name = "sparse",
        index_name = "sparse_bm25_index",
        index_type = "SPARSE_INVERTED_INDEX",
        metric_type = "BM25"

    )
    return index_params

def create_collection(client,collection_name,schema,index_params):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name = collection_name,
                             schema = schema,
                             index_params = index_params,
                             consistency_level = "Strong"
    )

def get_vector_store(uri,user,pwd,db_name,collection_name,embeddings):
    bm25_function = BM25BuiltInFunction(
        function_name="bm25_function",
        input_field_names="text",
        output_field_names="sparse",
        analyzer_params={"type":"chinese"}
    )
    vector_store = Milvus(
        embedding_function=embeddings,
        builtin_function=bm25_function,
        connection_args={
            "uri": uri,
            "db_name": db_name,
            "user":user,
            "password":pwd
        },

        collection_name=collection_name,

        primary_field="pk",
        text_field="text",
        vector_field=["dense","sparse"],

        auto_id=True,
        enable_dynamic_field=True,

        search_params=[
            {"metric_type": "COSINE","params": {"ef": 64}},
            {"metric_type":"BM25","params":{}}
        ],

        consistency_level="Strong",
        drop_old=False,
    )
    return vector_store


def ingest_documents(documents,vector_store):
    ids = vector_store.add_documents(documents)
    print("写入数量： ",len(ids))

def main():
    file_paths = glob(r"documents/*")
    uri = "http://localhost:19530"
    db_name = "rag_demo"
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    chunk_size = 500
    chunk_overlap = 50
    add_start_index = True
    collection_name = "rag_chunks"
    user="root"
    pwd="Milvus"
    docs = load_documents(file_paths)
    documents = split_documents(docs, chunk_size, chunk_overlap, add_start_index)

    embeddings = create_embeddings(embedding_model_path)
    embedding_dim = len(embeddings.embed_query("测试"))

    client = create_database(uri, db_name=db_name, user=user, pwd=pwd)

    schema = create_schema(client,dim=embedding_dim)
    index_params = create_index_params(client)
    create_collection(client, collection_name, schema, index_params)


    vector_store = get_vector_store(uri, user, pwd, db_name, collection_name, embeddings)
    ingest_documents(documents, vector_store)




if __name__ == "__main__":
    main()














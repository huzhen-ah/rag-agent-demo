#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:58:15 2026

@author: huzhen
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace,
)


def create_chat_model(model_path, device="mps"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    )

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
    )

    chat_model = ChatHuggingFace(
        llm=llm,
        tokenizer=tokenizer,
    )

    return chat_model

if __name__ == "__main__":
    model_path = "models/Qwen3-1.7B"

    chat_model = create_chat_model(
        model_path=model_path,
        device="mps",
    )

    response = chat_model.invoke(
        "简单介绍一下RAG。"
    )

    print(response.content)
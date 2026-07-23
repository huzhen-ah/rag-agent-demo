#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:49:32 2026

@author: huzhen
"""
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

class Generator:
    def __init__(self,model_path,device):
        self.model_path = model_path
        self.device = device
        self.model = self.load_generate_model()
        self.tokenizer = self.load_tokenizer()
    
    def load_generate_model(self):
        generate_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32
            )
        generate_model.to(self.device)
        generate_model.eval()
        return generate_model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False
            )
        return tokenizer
    
    def generate(self,messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_token_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return answer
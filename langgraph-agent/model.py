import json
import re
import uuid
from typing import Any

import torch
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


class LocalQwenChatModel(BaseChatModel):
    model_path: str
    device: str = "mps"
    max_new_tokens: int = 300

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def model_post_init(self, context):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.float16,
        ).to(self.device)

    @property
    def _llm_type(self):
        return "local_qwen"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice not in (None, "auto"):
            raise ValueError("LocalQwenChatModel只支持自动选择Tool")

        tool_definitions = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=tool_definitions, **kwargs)

    def _to_qwen_messages(self, messages):
        qwen_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                qwen_message = {
                    "role": "system",
                    "content": message.content,
                }
            elif isinstance(message, HumanMessage):
                qwen_message = {
                    "role": "user",
                    "content": message.content,
                }
            elif isinstance(message, AIMessage):
                qwen_message = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    qwen_message["tool_calls"] = [
                        {
                            "name": tool_call["name"],
                            "arguments": tool_call["args"],
                        }
                        for tool_call in message.tool_calls
                    ]
            elif isinstance(message, ToolMessage):
                qwen_message = {
                    "role": "tool",
                    "content": message.content,
                }
            else:
                raise TypeError("不支持的Message类型: {}".format(type(message)))

            qwen_messages.append(qwen_message)

        return qwen_messages

    def _parse_response(self, response):
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        tool_call_blocks = re.findall(pattern, response, flags=re.DOTALL)
        content = re.sub(pattern, "", response, flags=re.DOTALL).strip()
        tool_calls = []

        for block in tool_call_blocks:
            try:
                tool_call = json.loads(block)
            except json.JSONDecodeError as error:
                raise ValueError("Tool Call不是合法JSON: {}".format(block)) from error

            name = tool_call.get("name")
            arguments = tool_call.get("arguments")

            if not isinstance(name, str):
                raise ValueError("Tool Call缺少合法name")
            if not isinstance(arguments, dict):
                raise ValueError("Tool Call缺少合法arguments")

            tool_calls.append(
                {
                    "name": name,
                    "args": arguments,
                    "id": "call_{}".format(uuid.uuid4().hex),
                }
            )

        return AIMessage(content=content, tool_calls=tool_calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        qwen_messages = self._to_qwen_messages(messages)
        tool_definitions = kwargs.get("tools", [])
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        model_inputs = self._tokenizer.apply_chat_template(
            qwen_messages,
            tools=tool_definitions,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self._model.device)

        outputs = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        response = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        ai_message = self._parse_response(response)
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)]
        )

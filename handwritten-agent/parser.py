#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:24:40 2026

@author: huzhen
"""

import json
import re


def parse_model_response(response):
    pattern = (
        r"<tool_call>\s*"
        r"(.*?)"
        r"\s*</tool_call>"
    )

    tool_call_blocks = re.findall(
        pattern,
        response,
        flags=re.DOTALL,
    )

    content = re.sub(
        pattern,
        "",
        response,
        flags=re.DOTALL,
    ).strip()

    tool_calls = []

    for block in tool_call_blocks:
        try:
            tool_call = json.loads(
                block
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Tool Call不是合法JSON：{}".format(
                    block
                )
            ) from error

        if not isinstance(tool_call, dict):
            raise ValueError(
                "Tool Call必须是JSON对象"
            )

        name = tool_call.get("name")
        arguments = tool_call.get(
            "arguments"
        )

        if not isinstance(name, str):
            raise ValueError(
                "Tool Call缺少合法name"
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool Call缺少合法arguments"
            )

        tool_calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

    return {
        "content": content,
        "tool_calls": tool_calls,
    }

if __name__ == "__main__":
    response = """
    我先读取你的简历。

    <tool_call>
    {
      "name": "read_resume",
      "arguments": {
        "resume_id": "main"
      }
    }
    </tool_call>
    """

    result = parse_model_response(
        response
    )

    print(result)
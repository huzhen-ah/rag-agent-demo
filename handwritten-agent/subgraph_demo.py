#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 21:02:44 2026

@author: huzhen
"""
from typing import TypedDict
from checkpoint import InMemoryCheckpointer
from graph import StateGraph, START, END
from hitl import Command, interrupt
from subgraph import SubGraphNode


class ParentState(TypedDict):
    name: str
    wife_name: str
    son_name: str


class ChildState(TypedDict):
    father_name: str
    mother_name: str
    son_name: str


def find_son_name_node(state):
    family_records = {
        ("李世民", "长孙皇后"): "李治",
    }
    return {
        "son_name": family_records[(state["father_name"], state["mother_name"])],
    }


def review_son_name_node(state):
    review_response = interrupt(
        {
            "type": "review_son_name",
            "son_name": state["son_name"],
        }
    )
    if review_response == "approve":
        return {}
    return {"son_name": review_response}


def parent_to_child(parent_state):
    return {
        "father_name": parent_state["name"],
        "mother_name": parent_state["wife_name"],
        "son_name": "",
    }


def child_to_parent(parent_state, child_state):
    return {
        "son_name": child_state["son_name"],
    }


if __name__ == "__main__":
    checkpointer = InMemoryCheckpointer()

    child_graph_builder = StateGraph(ChildState)
    child_graph_builder.add_node("find_son_name", find_son_name_node)
    child_graph_builder.add_node("review_son_name", review_son_name_node)
    child_graph_builder.add_edge(START, "find_son_name")
    child_graph_builder.add_edge("find_son_name", "review_son_name")
    child_graph_builder.add_edge("review_son_name", END)
    child_graph = child_graph_builder.compile(checkpointer)

    subgraph_node = SubGraphNode(
        child_graph,
        parent_to_child,
        child_to_parent,
    )

    parent_graph_builder = StateGraph(ParentState)
    parent_graph_builder.add_node("subgraph_node", subgraph_node)
    parent_graph_builder.add_edge(START, "subgraph_node")
    parent_graph_builder.add_edge("subgraph_node", END)
    parent_graph = parent_graph_builder.compile(checkpointer)

    thread_id = "subgraph_demo_thread"
    interrupted_output = parent_graph.invoke(
        {
            "name": "李世民",
            "wife_name": "长孙皇后",
            "son_name": "",
        },
        thread_id=thread_id,
    )

    assert "__interrupt__" in interrupted_output
    interrupts = interrupted_output["__interrupt__"]
    assert len(interrupts) == 1
    interrupt_data = interrupts[0]
    assert interrupt_data.value == {
        "type": "review_son_name",
        "son_name": "李治",
    }

    interrupted_parent_snapshot = checkpointer.get(thread_id, "")
    checkpoint_namespaces = set(checkpointer.checkpoints[thread_id])
    child_checkpoint_namespaces = checkpoint_namespaces - {""}
    assert len(child_checkpoint_namespaces) == 1
    child_checkpoint_ns = next(iter(child_checkpoint_namespaces))
    interrupted_child_snapshot = checkpointer.get(thread_id, child_checkpoint_ns)

    output = parent_graph.invoke(
        Command(resume={interrupt_data.id: "approve"}),
        thread_id=thread_id,
    )

    assert output == {
        "name": "李世民",
        "wife_name": "长孙皇后",
        "son_name": "李治",
    }

    parent_writes = checkpointer.get_writes(
        thread_id,
        "",
        interrupted_parent_snapshot.checkpoint_id,
    )
    child_writes = checkpointer.get_writes(
        thread_id,
        child_checkpoint_ns,
        interrupted_child_snapshot.checkpoint_id,
    )
    assert not any(write.channel == "resume" for write in parent_writes)
    assert any(
        write.channel == "resume" and write.value == ["approve"]
        for write in child_writes
    )

    print("output:", output)
    print("checkpoint_namespaces:", checkpoint_namespaces)

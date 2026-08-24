#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tempfile import TemporaryDirectory
from typing import TypedDict

from checkpoint import InMemoryCheckpointer, JsonlCheckpointer
from graph import END, START, StateGraph
from hitl import Command, interrupt
from subgraph import SubGraphNode


class ParentState(TypedDict):
    first_request: str
    second_request: str
    first_name: str
    second_name: str


class ChildState(TypedDict):
    person_label: str
    approved_name: str


def review_name_node(state):
    approved_name = interrupt(
        {
            "type": "name_review",
            "person_label": state["person_label"],
        }
    )
    return {"approved_name": approved_name}


def first_parent_to_child(parent_state):
    return {
        "person_label": parent_state["first_request"],
        "approved_name": "",
    }


def second_parent_to_child(parent_state):
    return {
        "person_label": parent_state["second_request"],
        "approved_name": "",
    }


def first_child_to_parent(parent_state, child_state):
    return {"first_name": child_state["approved_name"]}


def second_child_to_parent(parent_state, child_state):
    return {"second_name": child_state["approved_name"]}


def build_parent_graph(checkpointer, parallel):
    child_builder = StateGraph(ChildState)
    child_builder.add_node("review_name", review_name_node)
    child_builder.add_edge(START, "review_name")
    child_builder.add_edge("review_name", END)
    child_graph = child_builder.compile(checkpointer)

    first_subgraph_node = SubGraphNode(
        child_graph,
        first_parent_to_child,
        first_child_to_parent,
    )

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("first_subgraph", first_subgraph_node)
    parent_builder.add_edge(START, "first_subgraph")
    parent_builder.add_edge("first_subgraph", END)

    if parallel:
        second_subgraph_node = SubGraphNode(
            child_graph,
            second_parent_to_child,
            second_child_to_parent,
        )
        parent_builder.add_node("second_subgraph", second_subgraph_node)
        parent_builder.add_edge(START, "second_subgraph")
        parent_builder.add_edge("second_subgraph", END)

    return parent_builder.compile(checkpointer)


def initial_state():
    return {
        "first_request": "长子",
        "second_request": "次子",
        "first_name": "",
        "second_name": "",
    }


def verify_parallel_subgraphs():
    checkpointer = InMemoryCheckpointer()
    parent_graph = build_parent_graph(checkpointer, parallel=True)
    thread_id = "parallel_subgraph_thread"

    interrupted_output = parent_graph.invoke(initial_state(), thread_id=thread_id)
    interrupts = interrupted_output["__interrupt__"]
    assert len(interrupts) == 2

    interrupt_by_label = {
        interrupt_data.value["person_label"]: interrupt_data
        for interrupt_data in interrupts
    }
    command = Command(
        resume={
            interrupt_by_label["长子"].id: "李承乾",
            interrupt_by_label["次子"].id: "李泰",
        }
    )

    interrupted_snapshots = {
        checkpoint_ns: checkpointer.get(thread_id, checkpoint_ns)
        for checkpoint_ns in checkpointer.checkpoints[thread_id]
    }
    output = parent_graph.invoke(command, thread_id=thread_id)

    assert output == {
        "first_request": "长子",
        "second_request": "次子",
        "first_name": "李承乾",
        "second_name": "李泰",
    }

    root_snapshot = interrupted_snapshots[""]
    root_writes = checkpointer.get_writes(
        thread_id,
        "",
        root_snapshot.checkpoint_id,
    )
    assert not any(write.channel == "resume" for write in root_writes)

    child_checkpoint_namespaces = set(interrupted_snapshots) - {""}
    assert len(child_checkpoint_namespaces) == 2
    child_resume_values = set()
    for checkpoint_ns in child_checkpoint_namespaces:
        child_snapshot = interrupted_snapshots[checkpoint_ns]
        child_writes = checkpointer.get_writes(
            thread_id,
            checkpoint_ns,
            child_snapshot.checkpoint_id,
        )
        resume_write = next(
            write for write in child_writes if write.channel == "resume"
        )
        child_resume_values.add(resume_write.value[0])
    assert child_resume_values == {"李承乾", "李泰"}


def verify_jsonl_restart():
    with TemporaryDirectory() as temp_directory:
        checkpoint_file = "{}/checkpoints.jsonl".format(temp_directory)
        pending_writes_file = "{}/pending_writes.jsonl".format(temp_directory)
        thread_id = "jsonl_subgraph_thread"

        first_checkpointer = JsonlCheckpointer(
            checkpoint_file,
            pending_writes_file,
        )
        first_parent_graph = build_parent_graph(first_checkpointer, parallel=False)
        interrupted_output = first_parent_graph.invoke(
            initial_state(),
            thread_id=thread_id,
        )
        interrupt_data = interrupted_output["__interrupt__"][0]

        second_checkpointer = JsonlCheckpointer(
            checkpoint_file,
            pending_writes_file,
        )
        second_parent_graph = build_parent_graph(second_checkpointer, parallel=False)
        output = second_parent_graph.invoke(
            Command(resume={interrupt_data.id: "李承乾"}),
            thread_id=thread_id,
        )

        assert output == {
            "first_request": "长子",
            "second_request": "次子",
            "first_name": "李承乾",
            "second_name": "",
        }


if __name__ == "__main__":
    verify_parallel_subgraphs()
    verify_jsonl_restart()
    print("parallel subgraph HITL: passed")
    print("JSONL restart recovery: passed")

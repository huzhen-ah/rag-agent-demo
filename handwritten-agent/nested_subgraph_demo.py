#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TypedDict

from checkpoint import InMemoryCheckpointer
from graph import END, START, StateGraph
from hitl import Command, interrupt
from subgraph import SubGraphNode


class ParentState(TypedDict):
    name: str
    wife_name: str
    son_name: str


class FamilyState(TypedDict):
    father_name: str
    mother_name: str
    son_name: str


class EvidenceState(TypedDict):
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


def parent_to_family(parent_state):
    return {
        "father_name": parent_state["name"],
        "mother_name": parent_state["wife_name"],
        "son_name": "",
    }


def family_to_parent(parent_state, family_state):
    return {
        "son_name": family_state["son_name"],
    }


def family_to_evidence(family_state):
    return dict(family_state)


def evidence_to_family(family_state, evidence_state):
    return {
        "son_name": evidence_state["son_name"],
    }


if __name__ == "__main__":
    checkpointer = InMemoryCheckpointer()

    evidence_builder = StateGraph(EvidenceState)
    evidence_builder.add_node("find_son_name", find_son_name_node)
    evidence_builder.add_node("review_son_name", review_son_name_node)
    evidence_builder.add_edge(START, "find_son_name")
    evidence_builder.add_edge("find_son_name", "review_son_name")
    evidence_builder.add_edge("review_son_name", END)
    evidence_graph = evidence_builder.compile(checkpointer)

    evidence_subgraph_node = SubGraphNode(
        evidence_graph,
        family_to_evidence,
        evidence_to_family,
    )

    family_builder = StateGraph(FamilyState)
    family_builder.add_node("evidence_subgraph", evidence_subgraph_node)
    family_builder.add_edge(START, "evidence_subgraph")
    family_builder.add_edge("evidence_subgraph", END)
    family_graph = family_builder.compile(checkpointer)

    family_subgraph_node = SubGraphNode(
        family_graph,
        parent_to_family,
        family_to_parent,
    )

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("family_subgraph", family_subgraph_node)
    parent_builder.add_edge(START, "family_subgraph")
    parent_builder.add_edge("family_subgraph", END)
    parent_graph = parent_builder.compile(checkpointer)

    thread_id = "nested_subgraph_demo_thread"
    interrupt_events = list(
        parent_graph.stream(
            {
                "name": "李世民",
                "wife_name": "长孙皇后",
                "son_name": "",
            },
            thread_id=thread_id,
        )
    )
    assert [event.event_type for event in interrupt_events] == ["update", "interrupt"]
    assert interrupt_events[0].payload["node_name"] == "find_son_name"
    interrupted_output = interrupt_events[-1].payload["output"]

    assert "__interrupt__" in interrupted_output
    interrupt_data = interrupted_output["__interrupt__"][0]

    checkpoint_namespaces = set(checkpointer.checkpoints[thread_id])
    root_checkpoint_ns = ""
    family_checkpoint_ns = next(
        checkpoint_ns
        for checkpoint_ns in checkpoint_namespaces
        if checkpoint_ns.count("|") == 0 and checkpoint_ns
    )
    evidence_checkpoint_ns = next(
        checkpoint_ns
        for checkpoint_ns in checkpoint_namespaces
        if checkpoint_ns.count("|") == 1
    )
    assert interrupt_events[0].payload["checkpoint_ns"] == evidence_checkpoint_ns

    interrupted_snapshots = {
        checkpoint_ns: checkpointer.get(thread_id, checkpoint_ns)
        for checkpoint_ns in checkpoint_namespaces
    }

    resume_events = list(
        parent_graph.stream(
            Command(resume={interrupt_data.id: "approve"}),
            thread_id=thread_id,
        )
    )
    assert [event.event_type for event in resume_events] == [
        "update",
        "update",
        "update",
        "final",
    ]
    assert [event.payload["node_name"] for event in resume_events[:-1]] == [
        "review_son_name",
        "evidence_subgraph",
        "family_subgraph",
    ]
    assert [event.payload["checkpoint_ns"] for event in resume_events[:-1]] == [
        evidence_checkpoint_ns,
        family_checkpoint_ns,
        root_checkpoint_ns,
    ]
    output = resume_events[-1].payload["output"]

    assert output == {
        "name": "李世民",
        "wife_name": "长孙皇后",
        "son_name": "李治",
    }

    root_writes = checkpointer.get_writes(
        thread_id,
        root_checkpoint_ns,
        interrupted_snapshots[root_checkpoint_ns].checkpoint_id,
    )
    family_writes = checkpointer.get_writes(
        thread_id,
        family_checkpoint_ns,
        interrupted_snapshots[family_checkpoint_ns].checkpoint_id,
    )
    evidence_writes = checkpointer.get_writes(
        thread_id,
        evidence_checkpoint_ns,
        interrupted_snapshots[evidence_checkpoint_ns].checkpoint_id,
    )

    assert not any(write.channel == "resume" for write in root_writes)
    assert not any(write.channel == "resume" for write in family_writes)
    assert any(
        write.channel == "resume" and write.value == ["approve"]
        for write in evidence_writes
    )

    print("output:", output)
    print("checkpoint_namespaces:", checkpoint_namespaces)

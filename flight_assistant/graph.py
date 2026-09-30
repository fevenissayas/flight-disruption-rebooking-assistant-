"""The disruption graph: specialists, a policy cycle, and the bonus pauses."""

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore

from flight_assistant.nodes import (
    classifier,
    clarify,
    compensation_agent,
    escalation_agent,
    final_response_agent,
    needs_supervisor,
    policy_checker,
    rebooking_agent,
    refund_agent,
    supervisor,
)
from flight_assistant.policy import MAX_POLICY_ATTEMPTS
from flight_assistant.state import RebookingState
from flight_assistant.tools import ALL_TOOLS


def route_after_classifier(state: RebookingState) -> str:
    intent = state.get("intent")
    if intent == "rebook":
        return "clarify"
    if intent == "refund":
        return "refund"
    if intent == "compensation":
        return "compensation"
    return "escalation"


def route_after_policy(state: RebookingState) -> str:
    if state.get("policy_checked") and state.get("policy_passed"):
        if needs_supervisor(state):
            return "supervisor"
        return "final_response"
    if int(state.get("retry_count") or 0) >= MAX_POLICY_ATTEMPTS:
        return "escalation"
    source = state.get("solution_source")
    if source in {"rebooking", "refund", "compensation"}:
        return source
    return "escalation"


def route_after_supervisor(state: RebookingState) -> str:
    if state.get("supervisor_decision") == "approved":
        return "final_response"
    return "escalation"


def sqlite_checkpointer(path: str = "checkpoints.sqlite") -> SqliteSaver:
    """Checkpoint on disk so a thread is still there after the process restarts."""
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def build_graph(checkpointer=None, store=None):
    builder = StateGraph(RebookingState)
    builder.add_node("classifier", classifier)
    builder.add_node("clarify", clarify)
    builder.add_node("rebooking", rebooking_agent)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("refund", refund_agent)
    builder.add_node("compensation", compensation_agent)
    builder.add_node("policy_checker", policy_checker)
    builder.add_node("supervisor", supervisor)
    builder.add_node("escalation", escalation_agent)
    builder.add_node("final_response", final_response_agent)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "clarify": "clarify",
            "rebooking": "rebooking",
            "refund": "refund",
            "compensation": "compensation",
            "escalation": "escalation",
        },
    )
    builder.add_edge("clarify", "rebooking")
    # tools_condition returns "tools" or END. END here means "the model stopped
    # calling tools", so that branch is sent to the policy checker rather than
    # leaving the graph.
    builder.add_conditional_edges(
        "rebooking",
        tools_condition,
        {"tools": "tools", END: "policy_checker"},
    )
    builder.add_edge("tools", "rebooking")
    builder.add_edge("refund", "policy_checker")
    builder.add_edge("compensation", "policy_checker")
    builder.add_conditional_edges(
        "policy_checker",
        route_after_policy,
        {
            "final_response": "final_response",
            "supervisor": "supervisor",
            "escalation": "escalation",
            "rebooking": "rebooking",
            "refund": "refund",
            "compensation": "compensation",
        },
    )
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"final_response": "final_response", "escalation": "escalation"},
    )
    builder.add_edge("escalation", "final_response")
    builder.add_edge("final_response", END)

    if checkpointer is None:
        checkpointer = MemorySaver()
    if store is None:
        store = InMemoryStore()
    return builder.compile(checkpointer=checkpointer, store=store)

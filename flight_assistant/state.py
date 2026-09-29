"""Shared graph state. Every node reads it; each node writes its own fields."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

RESET_ATTEMPTS = [{"__reset__": True}]


def attempt_reducer(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Append policy attempts, or clear them when a new passenger message starts."""
    incoming = list(right or [])
    if incoming and isinstance(incoming[0], dict) and incoming[0].get("__reset__"):
        return incoming[1:]
    return list(left or []) + incoming


class RebookingState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: Annotated[list[dict], attempt_reducer]
    request_id: str
    booking_ref: str
    passenger_request: str
    intent: str
    constraints: dict[str, Any]
    booking: dict[str, Any]
    search_results: list[dict[str, Any]]
    fare_rules: dict[str, Any]
    proposed_solution: dict[str, Any]
    solution_source: str
    policy_checked: bool
    policy_passed: bool
    policy_reason: str
    retry_count: int
    escalated: bool
    escalation_note: str
    final_response: str

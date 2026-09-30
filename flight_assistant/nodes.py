"""Graph nodes. Each one updates only the fields it owns, plus the message log."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from flight_assistant.llm import get_llm
from flight_assistant.policy import MAX_POLICY_ATTEMPTS, evaluate_policy
from flight_assistant.prompts import (
    POLICY_TAG,
    classifier_system,
    compensation_system,
    escalation_system,
    final_response_system,
    propose_system,
    rebooking_system,
    refund_system,
)
from flight_assistant.schemas import (
    Classification,
    CompensationProposal,
    FlightProposal,
    RefundProposal,
)
from flight_assistant.state import RESET_ATTEMPTS, RebookingState
from flight_assistant.tools import REBOOKING_TOOLS, get_booking, get_fare_rules

MAX_TOOL_RESULTS = 4


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _passenger_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not _text(message).startswith(POLICY_TAG):
            return _text(message)
    return ""


def _conversation(messages: list) -> list:
    kept = []
    for message in messages:
        if isinstance(message, HumanMessage) and _text(message).startswith(POLICY_TAG):
            continue
        kept.append(message)
    return kept


def _payload(message: ToolMessage):
    content = message.content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return content


def _latest_tool(messages: list, name: str):
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == name:
            return _payload(message)
    return None


def _booking_from(payload) -> dict | None:
    if isinstance(payload, dict) and payload.get("booking_ref") and "error" not in payload:
        return payload
    return None


def _flights_from(payload) -> list | None:
    if isinstance(payload, list):
        return payload
    return None


def _tool_results_since_feedback(messages: list) -> int:
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) and _text(message).startswith(POLICY_TAG):
            start = index + 1
    return sum(isinstance(message, ToolMessage) for message in messages[start:])


def _json(value) -> str:
    return json.dumps(value, indent=2)


def _end_on_user(messages: list) -> list:
    """Keep the request ending on a user turn. Some providers reject a trailing assistant message."""
    if messages and isinstance(messages[-1], AIMessage):
        return [
            *messages,
            HumanMessage(content="Continue. Call a tool if booking or flight data is still missing."),
        ]
    return messages


def classifier(state: RebookingState) -> dict:
    """Read the latest passenger message and set intent plus constraints."""
    messages = state.get("messages", [])
    classification = get_llm().with_structured_output(Classification).invoke(
        [SystemMessage(content=classifier_system()), *_conversation(messages)]
    )
    return {
        "intent": classification.intent,
        "constraints": {
            "latest_arrival": classification.latest_arrival,
            "notes": classification.notes,
        },
        "passenger_request": _passenger_text(messages),
        "retry_count": 0,
        "policy_checked": False,
        "policy_passed": False,
        "policy_reason": "",
        "proposed_solution": {},
        "solution_source": "",
        "escalated": False,
        "escalation_note": "",
        "final_response": "",
        "attempts": RESET_ATTEMPTS,
        "messages": [AIMessage(content=f"Classified intent: {classification.intent}.")],
    }


def rebooking_agent(state: RebookingState) -> dict:
    """Call booking and flight tools, then propose one flight from the results."""
    messages = list(state.get("messages", []))
    booking = _booking_from(_latest_tool(messages, "get_booking")) or state.get("booking")
    flights = _flights_from(_latest_tool(messages, "search_flights"))
    if flights is None:
        flights = state.get("search_results")

    failed = bool(state.get("policy_checked")) and not state.get("policy_passed")
    over_cap = _tool_results_since_feedback(messages) >= MAX_TOOL_RESULTS
    if booking and flights and (failed or over_cap):
        return _propose(state, booking, flights)

    constraints = _json(state.get("constraints") or {})
    response = get_llm().bind_tools(REBOOKING_TOOLS).invoke(
        [
            SystemMessage(content=rebooking_system(state.get("booking_ref", ""), constraints)),
            *_end_on_user(messages),
        ]
    )
    if response.tool_calls and not over_cap:
        update = {"messages": [response]}
        if booking:
            update["booking"] = booking
        if flights:
            update["search_results"] = flights
        return update
    return _propose(state, booking, flights)


def _propose(state: RebookingState, booking: dict | None, flights: list | None) -> dict:
    constraints = _json(state.get("constraints") or {})
    cabin = (booking or {}).get("cabin_class", "unknown")
    proposal = get_llm().with_structured_output(FlightProposal).invoke(
        [
            SystemMessage(
                content=propose_system(constraints, cabin, state.get("policy_reason", ""))
            ),
            HumanMessage(
                content=(
                    "Booking:\n"
                    + _json(booking or {})
                    + "\n\nSearch results:\n"
                    + _json(flights or [])
                )
            ),
        ]
    )
    record = next(
        (
            flight
            for flight in (flights or [])
            if flight.get("flight_number") == proposal.flight_number
        ),
        None,
    )
    update = {
        "proposed_solution": {
            "type": "rebook",
            "flight_number": proposal.flight_number,
            "summary": proposal.summary,
            "flight": record,
        },
        "solution_source": "rebooking",
        "messages": [AIMessage(content=proposal.summary)],
    }
    if booking:
        update["booking"] = booking
    if flights is not None:
        update["search_results"] = flights
    return update


def refund_agent(state: RebookingState) -> dict:
    """Read the booking and fare rules, then propose a refund amount."""
    booking = _booking_from(json.loads(get_booking.invoke({"booking_ref": state["booking_ref"]})))
    fare_type = (booking or {}).get("fare_type", "")
    fare_rules = json.loads(get_fare_rules.invoke({"fare_type": fare_type}))
    if isinstance(fare_rules, dict) and "error" in fare_rules:
        fare_rules = {}
    proposal = get_llm().with_structured_output(RefundProposal).invoke(
        [
            SystemMessage(content=refund_system()),
            HumanMessage(
                content="Booking:\n"
                + _json(booking or {})
                + "\n\nFare rules:\n"
                + _json(fare_rules)
                + "\n\nPassenger: "
                + state.get("passenger_request", "")
            ),
        ]
    )
    return {
        "booking": booking or {},
        "fare_rules": fare_rules,
        "proposed_solution": {
            "type": "refund",
            "amount": proposal.amount,
            "summary": proposal.summary,
        },
        "solution_source": "refund",
        "messages": [AIMessage(content=proposal.summary)],
    }


def compensation_agent(state: RebookingState) -> dict:
    """Read the booking and delay, then propose a compensation amount."""
    booking = _booking_from(json.loads(get_booking.invoke({"booking_ref": state["booking_ref"]})))
    proposal = get_llm().with_structured_output(CompensationProposal).invoke(
        [
            SystemMessage(content=compensation_system()),
            HumanMessage(
                content="Booking:\n"
                + _json(booking or {})
                + "\n\nPassenger: "
                + state.get("passenger_request", "")
            ),
        ]
    )
    return {
        "booking": booking or {},
        "proposed_solution": {
            "type": "compensation",
            "amount": proposal.amount,
            "summary": proposal.summary,
        },
        "solution_source": "compensation",
        "messages": [AIMessage(content=proposal.summary)],
    }


def policy_checker(state: RebookingState) -> dict:
    """Accept or reject the proposal. A rejection goes back to the specialist."""
    passed, reason = evaluate_policy(
        state.get("intent", ""),
        state.get("proposed_solution"),
        state.get("booking"),
        state.get("search_results"),
        state.get("fare_rules"),
    )
    retry_count = int(state.get("retry_count") or 0)
    if not passed:
        retry_count += 1
    update = {
        "policy_checked": True,
        "policy_passed": passed,
        "policy_reason": reason,
        "retry_count": retry_count,
        "attempts": [
            {
                "attempt": retry_count if not passed else int(state.get("retry_count") or 0) + 1,
                "source": state.get("solution_source", ""),
                "solution": state.get("proposed_solution") or {},
                "passed": passed,
                "reason": reason,
            }
        ],
    }
    if not passed:
        update["messages"] = [
            HumanMessage(
                content=(
                    f"{POLICY_TAG} Attempt {retry_count} of {MAX_POLICY_ATTEMPTS} was rejected. "
                    f"{reason} Propose a different solution."
                )
            )
        ]
    return update


def escalation_agent(state: RebookingState) -> dict:
    """Write the note a human agent reads when the graph cannot finish the case."""
    snapshot = {
        "passenger_request": state.get("passenger_request", ""),
        "intent": state.get("intent", ""),
        "booking_ref": state.get("booking_ref", ""),
        "attempts": state.get("attempts") or [],
        "policy_reason": state.get("policy_reason", ""),
        "retry_count": state.get("retry_count", 0),
    }
    note = _text(
        get_llm().invoke(
            [
                SystemMessage(content=escalation_system()),
                HumanMessage(content=_json(snapshot)),
            ]
        )
    )
    return {
        "escalated": True,
        "escalation_note": note,
        "messages": [AIMessage(content=note)],
    }


def final_response_agent(state: RebookingState) -> dict:
    """Write the passenger-facing reply from the outcome already in state."""
    snapshot = {
        "passenger_name": (state.get("booking") or {}).get("passenger_name", ""),
        "passenger_request": state.get("passenger_request", ""),
        "intent": state.get("intent", ""),
        "constraints": state.get("constraints") or {},
        "proposed_solution": state.get("proposed_solution") or {},
        "policy_passed": state.get("policy_passed", False),
        "policy_reason": state.get("policy_reason", ""),
        "escalated": state.get("escalated", False),
        "escalation_note": state.get("escalation_note", ""),
    }
    reply = _text(
        get_llm().invoke(
            [
                SystemMessage(content=final_response_system()),
                HumanMessage(content=_json(snapshot)),
            ]
        )
    )
    return {
        "final_response": reply,
        "messages": [AIMessage(content=reply)],
    }

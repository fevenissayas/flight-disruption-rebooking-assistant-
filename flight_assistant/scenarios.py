"""Run the four graded scenarios and the follow-up on one thread.

Usage:
    python -m flight_assistant.scenarios
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from flight_assistant.graph import build_graph

NODE_NAMES = {
    "classifier",
    "rebooking",
    "tools",
    "refund",
    "compensation",
    "policy_checker",
    "escalation",
    "final_response",
}


@dataclass(frozen=True)
class Turn:
    name: str
    thread_id: str
    message: str
    request_id: str = ""
    booking_ref: str = ""
    send_booking: bool = True


TURNS = [
    Turn(
        name="Scenario 1 — cancelled flight, arrive by noon",
        thread_id="passenger-xk9l2p",
        request_id="R-7702",
        booking_ref="XK9L2P",
        message="My flight to Dubai was cancelled. I need to be there by tomorrow noon.",
    ),
    Turn(
        name="Scenario 1 follow-up — refund on the same thread",
        thread_id="passenger-xk9l2p",
        message="Actually, can I get a refund instead?",
        send_booking=False,
    ),
    Turn(
        name="Scenario 2 — six-hour delay, compensation",
        thread_id="passenger-dl6h01",
        request_id="R-7703",
        booking_ref="DL6H01",
        message="My flight is delayed 6 hours. Am I entitled to anything?",
    ),
    Turn(
        name="Scenario 3 — only business class is available",
        thread_id="passenger-bz3cls",
        request_id="R-7704",
        booking_ref="BZ3CLS",
        message="My flight was cancelled. Please put me on the next flight to New York.",
    ),
    Turn(
        name="Scenario 4 — complaint, ask for a manager",
        thread_id="passenger-cmp004",
        request_id="R-7705",
        booking_ref="CMP004",
        message="This is the third time you've cancelled on me. I want to speak to a manager.",
    ),
]


def _in_order(path: list[str], names: list[str]) -> bool:
    cursor = 0
    for name in names:
        try:
            cursor = path.index(name, cursor) + 1
        except ValueError:
            return False
    return True


def problems_for(turn: Turn, path: list[str], state: dict) -> list[str]:
    problems = []
    if not state.get("final_response"):
        problems.append("final response is empty")

    if turn.name.startswith("Scenario 1 —"):
        if not _in_order(path, ["classifier", "rebooking", "tools", "policy_checker", "final_response"]):
            problems.append("expected classifier → rebooking → tools → policy checker → final response")
        if path.count("policy_checker") != 1:
            problems.append("policy checker should pass on the first try")
        if "escalation" in path or "refund" in path:
            problems.append("scenario 1 should not escalate or refund")
        if not state.get("policy_passed"):
            problems.append("policy should pass")
        if (state.get("proposed_solution") or {}).get("flight_number") != "EK202":
            problems.append("expected flight EK202")

    elif turn.name.startswith("Scenario 1 follow-up"):
        if not _in_order(path, ["classifier", "refund", "policy_checker", "final_response"]):
            problems.append("expected classifier → refund → policy checker → final response")
        if "rebooking" in path or "escalation" in path:
            problems.append("follow-up should stay on the refund path")
        if state.get("intent") != "refund":
            problems.append("intent should be refund")
        if state.get("booking_ref") != "XK9L2P":
            problems.append("booking reference should still be XK9L2P")
        if not (state.get("booking") or {}).get("passenger_name"):
            problems.append("saved booking should still be available")
        if not state.get("policy_passed"):
            problems.append("refund policy should pass")
        amount = (state.get("proposed_solution") or {}).get("amount")
        if amount is None or abs(float(amount) - 850.0) > 0.01:
            problems.append("airline cancellation should refund the full fare 850")

    elif turn.name.startswith("Scenario 2"):
        if not _in_order(path, ["classifier", "compensation", "policy_checker", "final_response"]):
            problems.append("expected classifier → compensation → policy checker → final response")
        if path.count("policy_checker") != 1 or "escalation" in path:
            problems.append("compensation should pass policy once")
        amount = (state.get("proposed_solution") or {}).get("amount")
        if amount is None or abs(float(amount) - 200.0) > 0.01:
            problems.append("a 6 hour delay should pay 200")

    elif turn.name.startswith("Scenario 3"):
        policy_at = [index for index, name in enumerate(path) if name == "policy_checker"]
        if len(policy_at) != 3:
            problems.append("policy should fail three times (first try plus two retries)")
        elif "escalation" not in path or path.index("escalation") < policy_at[-1]:
            problems.append("escalation should follow the third failure")
        else:
            for earlier, later in zip(policy_at, policy_at[1:]):
                if "rebooking" not in path[earlier:later]:
                    problems.append("a failed check should return to rebooking")
                    break
        if path[-1] != "final_response":
            problems.append("the case should still end with a final response")
        if not state.get("escalated") or state.get("policy_passed"):
            problems.append("the case should be escalated after policy fails")
        if int(state.get("retry_count") or 0) != 3:
            problems.append("retry count should be 3")

    elif turn.name.startswith("Scenario 4"):
        if path != ["classifier", "escalation", "final_response"]:
            problems.append("expected classifier → escalation → final response")
        if not state.get("escalated"):
            problems.append("a manager request should escalate")

    return problems


def invoke_turn(graph, turn: Turn) -> tuple[list[str], dict]:
    payload: dict = {"messages": [HumanMessage(content=turn.message)]}
    if turn.send_booking:
        payload["request_id"] = turn.request_id
        payload["booking_ref"] = turn.booking_ref
    config = {"configurable": {"thread_id": turn.thread_id}, "recursion_limit": 60}
    path: list[str] = []
    for update in graph.stream(payload, config, stream_mode="updates"):
        path.extend(name for name in update if name in NODE_NAMES)
    state = dict(graph.get_state(config).values)
    return path, state


def run_all(graph) -> list[dict]:
    results = []
    for turn in TURNS:
        path, state = invoke_turn(graph, turn)
        results.append(
            {
                "turn": turn,
                "path": path,
                "state": state,
                "problems": problems_for(turn, path, state),
            }
        )
    return results


def _arrow(path: list[str]) -> str:
    return " → ".join(path)


def _tool_lines(messages: list) -> list[str]:
    lines = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"- {call['name']} {json.dumps(call['args'], sort_keys=True)}")
        elif isinstance(message, ToolMessage):
            preview = message.content if isinstance(message.content, str) else str(message.content)
            lines.append(f"- {message.name} returned {preview.splitlines()[0][:120]}")
    return lines


def render(diagram: str, results: list[dict]) -> str:
    parts = [
        "# Flight Disruption Rebooking Assistant — scenario output",
        "",
        "## Graph",
        "",
        "```mermaid",
        diagram.strip(),
        "```",
        "",
    ]
    for result in results:
        turn: Turn = result["turn"]
        state = result["state"]
        solution = state.get("proposed_solution") or {}
        parts.extend(
            [
                f"## {turn.name}",
                "",
                "Input:",
                "",
                "```json",
                json.dumps(
                    {
                        "thread_id": turn.thread_id,
                        "request_id": turn.request_id or None,
                        "booking_ref": turn.booking_ref if turn.send_booking else None,
                        "message": turn.message,
                    },
                    indent=2,
                ),
                "```",
                "",
                f"Path: `{_arrow(result['path'])}`",
                "",
                f"Intent: `{state.get('intent')}`",
                f"Policy passed: `{state.get('policy_passed')}`",
                f"Policy reason: {state.get('policy_reason') or '—'}",
                f"Retry count: `{state.get('retry_count')}`",
                f"Escalated: `{state.get('escalated')}`",
                "",
                "Proposed solution:",
                "",
                "```json",
                json.dumps(solution, indent=2),
                "```",
                "",
            ]
        )
        tools = _tool_lines(state.get("messages") or [])
        if tools:
            parts.append("Tool calls stored on this thread:")
            parts.append("")
            parts.extend(tools)
            parts.append("")
        if state.get("escalation_note"):
            parts.extend(["Handover note:", "", state["escalation_note"], ""])
        parts.extend(["Final response:", "", state.get("final_response") or "", ""])
        if result["problems"]:
            parts.append("Path check: failed")
            parts.append("")
            for problem in result["problems"]:
                parts.append(f"- {problem}")
            parts.append("")
        else:
            parts.extend(["Path check: matched the expected route.", ""])
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    graph = build_graph()
    diagram = graph.get_graph().draw_mermaid()
    results = run_all(graph)
    text = render(diagram, results)
    sys.stdout.write(text)
    with open("graph.mmd", "w", encoding="utf-8") as handle:
        handle.write(diagram if diagram.endswith("\n") else diagram + "\n")
    with open("scenario_output.md", "w", encoding="utf-8") as handle:
        handle.write(text)
    return 1 if any(result["problems"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

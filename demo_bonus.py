"""Live demo of every bonus path. Run:  .venv/bin/python demo_bonus.py

Uses the real model (LLM_PROVIDER from .env, ~18 small calls).
Delete checkpoints.sqlite to reset and run it again.
"""

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from flight_assistant.graph import build_graph, sqlite_checkpointer
from flight_assistant.parallel import parallel_search
from flight_assistant.stats import format_summary, summarize

STATES: list[dict] = []  # collected for the stats section at the end


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


def run(graph, payload: dict, thread_id: str) -> list[str]:
    """Stream until pause or finish; return node path."""
    path: list[str] = []
    try:
        for update in graph.stream(payload, cfg(thread_id), stream_mode="updates"):
            path.extend(update)
    except Exception as exc:  # GraphInterrupt surfaces here on some versions
        if "interrupt" not in type(exc).__name__.lower():
            raise
    print("   path so far:", " → ".join(path))
    return path


def resume(graph, thread_id: str, value) -> None:
    graph.invoke(Command(resume=value), cfg(thread_id))


def show_pause(graph, thread_id: str):
    snap = graph.get_state(cfg(thread_id))
    print("   paused at:", snap.next)
    print("   payload:", snap.interrupts[0].value if snap.interrupts else None)
    return snap


def main() -> None:
    graph = build_graph(checkpointer=sqlite_checkpointer())

    print("\n=== 1. CLARIFY: vague rebooking pauses before any search ===")
    run(graph, {"messages": [HumanMessage(content="Please rebook me.")],
                "request_id": "D-1", "booking_ref": "XK9L2P"}, "demo-clarify")
    show_pause(graph, "demo-clarify")
    resume(graph, "demo-clarify", "I must arrive by tomorrow noon.")
    st = graph.get_state(cfg("demo-clarify")).values
    print("   reply:", (st.get("final_response") or "")[:160])
    STATES.append({"state": st})

    print("\n=== 2. SUPERVISOR APPROVES the $850 refund ===")
    run(graph, {"messages": [HumanMessage(content="I want a refund instead.")],
                "request_id": "D-2", "booking_ref": "XK9L2P"}, "demo-approve")
    show_pause(graph, "demo-approve")
    resume(graph, "demo-approve", {"approved": True})
    st = graph.get_state(cfg("demo-approve")).values
    print("   decision:", st.get("supervisor_decision"), "| escalated:", st.get("escalated"))
    print("   reply:", (st.get("final_response") or "")[:160])
    STATES.append({"state": st})

    print("\n=== 3. SUPERVISOR DECLINES the $850 refund -> escalation ===")
    run(graph, {"messages": [HumanMessage(content="I want a refund instead.")],
                "request_id": "D-3", "booking_ref": "XK9L2P"}, "demo-decline")
    show_pause(graph, "demo-decline")
    resume(graph, "demo-decline", {"approved": False})
    st = graph.get_state(cfg("demo-decline")).values
    print("   decision:", st.get("supervisor_decision"), "| escalated:", st.get("escalated"))
    print("   reply:", (st.get("final_response") or "")[:160])
    STATES.append({"state": st})

    print("\n=== 4. SEAT MEMORY: aisle saved on one thread, recalled on another ===")
    graph.invoke({"messages": [HumanMessage(content="I prefer an aisle seat. I want a manager.")],
                  "booking_ref": "XK9L2P"}, cfg("demo-seat-a"))
    graph.invoke({"messages": [HumanMessage(content="I want a manager again.")],
                  "booking_ref": "XK9L2P"}, cfg("demo-seat-b"))
    st = graph.get_state(cfg("demo-seat-b")).values
    print("   seat_preference on new thread:", st.get("seat_preference"))
    STATES.append({"state": st})

    print("\n=== 5. PERSISTENCE: a brand-new graph object reads the old thread ===")
    fresh = build_graph(checkpointer=sqlite_checkpointer())
    old = fresh.get_state(cfg("demo-approve")).values
    print("   previous reply still there:", bool(old.get("final_response")))

    print("\n=== 6. PARALLEL SEARCH: one branch per airline, merged ===")
    print("   flights:", [f["flight_number"] for f in parallel_search("LHR", "DXB")])

    print("\n=== 7. STATISTICS over this demo's turns ===")
    print(format_summary(summarize(STATES)))


if __name__ == "__main__":
    main()

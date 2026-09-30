"""Bonus paths: approval pause, saved threads, memory, clarify, parallel search."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from flight_assistant.graph import build_graph, sqlite_checkpointer
from flight_assistant.llm import set_llm
from flight_assistant.parallel import parallel_search
from flight_assistant.schemas import Classification, RefundProposal
from flight_assistant.stats import summarize
from flight_assistant.tools import lookup_flights
from tests.test_graph import ScriptedChatModel, says


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}


def test_parallel_search_merges_every_airline_on_the_route():
    flights = parallel_search("LHR", "DXB")
    numbers = {flight["flight_number"] for flight in flights}
    assert numbers == {"EK202", "BA880", "LH441", "QR510"}
    assert lookup_flights("CDG", "JFK", "2026-09-30")
    assert {flight["flight_number"][:2] for flight in lookup_flights("CDG", "JFK", "2026-09-30")} == {
        "AF",
        "DL",
    }


def test_unclear_rebooking_pauses_before_search():
    model = ScriptedChatModel(
        steps=[Classification(intent="rebook", latest_arrival=None, notes="")]
    )
    set_llm(model)
    try:
        graph = build_graph(checkpointer=MemorySaver(), store=InMemoryStore())
        config = _config("clarify-1")
        graph.invoke(
            {
                "booking_ref": "XK9L2P",
                "messages": [HumanMessage(content="Please rebook me.")],
            },
            config,
        )
    finally:
        set_llm(None)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("clarify",)
    assert snapshot.interrupts[0].value["kind"] == "clarify"


def test_refund_over_300_pauses_until_a_supervisor_decides():
    model = ScriptedChatModel(
        steps=[
            Classification(intent="refund", latest_arrival=None, notes=""),
            RefundProposal(amount=850, summary="Full fare refund."),
            says("Declined, a person will follow up."),
            says("A supervisor declined the refund. A human agent will follow up."),
        ]
    )
    set_llm(model)
    try:
        graph = build_graph(checkpointer=MemorySaver(), store=InMemoryStore())
        config = _config("approve-1")
        graph.invoke(
            {
                "booking_ref": "XK9L2P",
                "request_id": "R-1",
                "messages": [HumanMessage(content="I want my money back.")],
            },
            config,
        )
        paused = graph.get_state(config)
        assert paused.interrupts[0].value["kind"] == "approval"
        assert paused.interrupts[0].value["amount"] == 850
        graph.invoke(Command(resume={"approved": False}), config)
    finally:
        set_llm(None)
    state = graph.get_state(config).values
    assert state["supervisor_decision"] == "declined"
    assert state["escalated"] is True
    assert state["final_response"]


def test_sqlite_thread_is_still_there_after_a_new_checkpointer(tmp_path):
    path = tmp_path / "desk.sqlite"
    model = ScriptedChatModel(
        steps=[
            Classification(intent="complaint", latest_arrival=None, notes=""),
            says("Handover: the passenger wants a manager."),
            says("A manager will follow up."),
        ]
    )
    set_llm(model)
    try:
        first = build_graph(checkpointer=sqlite_checkpointer(str(path)), store=InMemoryStore())
        config = _config("sqlite-1")
        first.invoke(
            {
                "booking_ref": "CMP004",
                "messages": [HumanMessage(content="I want to speak to a manager.")],
            },
            config,
        )
        second = build_graph(checkpointer=sqlite_checkpointer(str(path)), store=InMemoryStore())
        restored = second.get_state(config).values
    finally:
        set_llm(None)
    assert restored["final_response"] == "A manager will follow up."
    assert restored["booking_ref"] == "CMP004"


def test_store_remembers_an_aisle_seat_on_a_new_thread():
    model = ScriptedChatModel(
        steps=[
            Classification(intent="complaint", latest_arrival=None, notes=""),
            says("Noted."),
            says("A person will follow up."),
            Classification(intent="complaint", latest_arrival=None, notes=""),
            says("Noted again."),
            says("Still following up."),
        ]
    )
    store = InMemoryStore()
    set_llm(model)
    try:
        graph = build_graph(checkpointer=MemorySaver(), store=store)
        graph.invoke(
            {
                "booking_ref": "XK9L2P",
                "messages": [HumanMessage(content="I prefer an aisle seat. I want a manager.")],
            },
            _config("seat-a"),
        )
        graph.invoke(
            {
                "booking_ref": "XK9L2P",
                "messages": [HumanMessage(content="I want a manager again.")],
            },
            _config("seat-b"),
        )
    finally:
        set_llm(None)
    assert graph.get_state(_config("seat-b")).values["seat_preference"] == "aisle"


def test_statistics_from_finished_turns():
    summary = summarize(
        [
            {"state": {"intent": "rebook", "policy_passed": True, "escalated": False}},
            {"state": {"intent": "rebook", "policy_passed": False, "escalated": True}},
            {"state": {"intent": "refund", "policy_passed": True, "escalated": False}},
            {"state": {"intent": "complaint", "policy_passed": False, "escalated": True}},
        ]
    )
    assert summary["rebooking_success_rate"] == 0.5
    assert summary["escalation_rate"] == 0.5
    assert summary["by_intent"]["refund"] == 1

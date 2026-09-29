"""Drive all four scenarios with a scripted model to prove the graph routes."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from flight_assistant.llm import set_llm
from flight_assistant.scenarios import run_all
from flight_assistant.schemas import (
    Classification,
    CompensationProposal,
    FlightProposal,
    RefundProposal,
)
from flight_assistant.data import TOMORROW_NOON_ISO
from flight_assistant.graph import build_graph


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def says(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedChatModel(BaseChatModel):
    """Returns canned messages and structured objects in a fixed order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    steps: list = []
    index: int = 0

    def _next(self):
        if self.index >= len(self.steps):
            raise RuntimeError(f"Scripted model ran out of steps at {self.index}")
        item = self.steps[self.index]
        self.index += 1
        return item

    def _take_message(self) -> AIMessage:
        item = self._next()
        if not isinstance(item, AIMessage):
            raise RuntimeError(f"Expected an AIMessage, got {type(item).__name__}")
        return item

    def _take_struct(self):
        item = self._next()
        if isinstance(item, AIMessage) or not isinstance(item, BaseModel):
            raise RuntimeError(f"Expected a structured object, got {type(item).__name__}")
        return item

    def bind_tools(self, tools, **kwargs):
        return RunnableLambda(lambda _input: self._take_message())

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _input: self._take_struct())

    @override
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._take_message())])

    @property
    @override
    def _llm_type(self) -> str:
        return "scripted-chat-model"


def _script():
    rebook = Classification(
        intent="rebook",
        latest_arrival=TOMORROW_NOON_ISO,
        notes="Arrive in Dubai by noon tomorrow.",
    )
    return [
        rebook,
        tool_call("get_booking", {"booking_ref": "XK9L2P"}, "call-booking-1"),
        tool_call(
            "search_flights",
            {"origin": "LHR", "destination": "DXB", "date": "2026-09-30"},
            "call-search-1",
        ),
        says("I have the booking and the flight list."),
        FlightProposal(
            flight_number="EK202",
            summary="EK202 departs London at 01:40 and arrives in Dubai at 11:25.",
        ),
        says(
            "Amira Hassan, you are rebooked on EK202, arriving in Dubai at 11:25, "
            "before noon. No human agent needs to follow up."
        ),
        Classification(intent="refund", latest_arrival=None, notes="Passenger now wants a refund."),
        RefundProposal(amount=850, summary="The airline cancelled the flight, so the full fare is refunded."),
        says(
            "Amira Hassan, the full fare of 850 has been refunded because the airline "
            "cancelled the flight. No human agent needs to follow up."
        ),
        Classification(intent="compensation", latest_arrival=None, notes="Delay entitlement."),
        CompensationProposal(amount=200, summary="A 6 hour delay pays 200 in compensation."),
        says(
            "Noah Keller, a 6 hour delay entitles you to 200 in compensation. "
            "No human agent needs to follow up."
        ),
        Classification(intent="rebook", latest_arrival=None, notes="Next flight to New York."),
        tool_call("get_booking", {"booking_ref": "BZ3CLS"}, "call-booking-3"),
        tool_call(
            "search_flights",
            {"origin": "CDG", "destination": "JFK", "date": "2026-09-30"},
            "call-search-3",
        ),
        says("Only business class flights were returned."),
        FlightProposal(flight_number="AF011", summary="AF011 is the next flight, in business class."),
        FlightProposal(flight_number="DL404", summary="DL404 is the other listed flight, also business."),
        FlightProposal(flight_number="AF011", summary="AF011 remains the only other option."),
        says(
            "Sofia Martins asked to be rebooked to New York. Three proposals failed the cabin rule "
            "because only business class was available for an economy booking."
        ),
        says(
            "Sofia Martins, we could not rebook you in economy. A human agent will follow up."
        ),
        Classification(intent="complaint", latest_arrival=None, notes="Asked for a manager."),
        says(
            "Daniel Okonkwo wants a manager after a third cancellation. "
            "Nothing was rebooked or refunded."
        ),
        says(
            "Daniel Okonkwo, a manager will follow up with you about these cancellations."
        ),
    ]


def test_graph_diagram_has_the_eight_nodes():
    graph = build_graph()
    diagram = graph.get_graph().draw_mermaid()
    for name in (
        "classifier",
        "rebooking",
        "tools",
        "refund",
        "compensation",
        "policy_checker",
        "escalation",
        "final_response",
    ):
        assert name in diagram


def test_four_scenarios_and_the_follow_up_follow_the_expected_paths():
    model = ScriptedChatModel(steps=_script())
    set_llm(model)
    try:
        results = run_all(build_graph())
    finally:
        set_llm(None)

    for result in results:
        assert result["problems"] == [], (result["turn"].name, result["path"], result["problems"])
    assert model.index == len(model.steps)

    follow_up = results[1]["state"]
    assert follow_up["booking"]["passenger_name"] == "Amira Hassan"
    assert "booking reference" not in follow_up["final_response"].lower()
    assert results[3]["state"]["escalation_note"]
    assert results[2]["state"]["proposed_solution"]["amount"] == 200

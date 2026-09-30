"""Search several airlines at once with Send, then merge the lists."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from flight_assistant.data import FLIGHTS, normalize_place


class AirlineSearchState(TypedDict, total=False):
    origin: str
    destination: str
    airline: str
    flights: Annotated[list[dict], add]


def flights_for_route(origin: str, destination: str) -> list[dict]:
    key = (normalize_place(origin), normalize_place(destination))
    return list(FLIGHTS.get(key, []))


def _airlines_for(origin: str, destination: str) -> list[str]:
    codes = sorted({flight["flight_number"][:2] for flight in flights_for_route(origin, destination)})
    return codes


def _fan_out(state: AirlineSearchState):
    codes = _airlines_for(state.get("origin", ""), state.get("destination", ""))
    if not codes:
        return "finish"
    return [
        Send(
            "search_airline",
            {
                "origin": state.get("origin", ""),
                "destination": state.get("destination", ""),
                "airline": code,
            },
        )
        for code in codes
    ]


def _search_airline(state: AirlineSearchState) -> dict:
    code = state.get("airline", "")
    found = [
        flight
        for flight in flights_for_route(state.get("origin", ""), state.get("destination", ""))
        if str(flight.get("flight_number", "")).startswith(code)
    ]
    return {"flights": found}


def _finish(state: AirlineSearchState) -> dict:
    return {"flights": []}


def _build():
    builder = StateGraph(AirlineSearchState)
    builder.add_node("begin", lambda state: {})
    builder.add_node("search_airline", _search_airline)
    builder.add_node("finish", _finish)
    builder.add_edge(START, "begin")
    builder.add_conditional_edges("begin", _fan_out, ["search_airline", "finish"])
    builder.add_edge("search_airline", END)
    builder.add_edge("finish", END)
    return builder.compile()


_GRAPH = None


def parallel_search(origin: str, destination: str) -> list[dict]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build()
    result = _GRAPH.invoke({"origin": origin, "destination": destination, "flights": []})
    flights = list(result.get("flights") or [])
    flights.sort(key=lambda flight: (flight.get("departure", ""), flight.get("flight_number", "")))
    return flights

"""Mock tools. They return fixed data and do not call any external service."""

import json

from langchain_core.tools import tool

from flight_assistant.data import BOOKINGS, FARE_RULES


def lookup_booking(booking_ref: str) -> dict:
    booking = BOOKINGS.get(booking_ref.strip().upper())
    if booking is None:
        return {"error": f"No booking found for {booking_ref}."}
    return booking


def lookup_flights(origin: str, destination: str, date: str) -> list[dict]:
    # The date is accepted so the agent can call the tool naturally.
    # The schedule itself is fixed so each scenario is repeatable.
    _ = date
    from flight_assistant.parallel import parallel_search

    return parallel_search(origin, destination)


def lookup_fare_rules(fare_type: str) -> dict:
    rules = FARE_RULES.get(fare_type.strip().lower())
    if rules is None:
        return {"error": f"No fare rules found for {fare_type}."}
    return rules


@tool
def get_booking(booking_ref: str) -> str:
    """Load a booking by reference.

    Returns the passenger, route, cabin class, fare, and disruption
    (cancelled or delayed, including delay hours).
    """
    return json.dumps(lookup_booking(booking_ref), indent=2)


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search the fixed flight schedule for a city pair.

    Use IATA codes such as LHR, DXB, CDG, and JFK. City names such as Dubai
    are accepted too. `date` is YYYY-MM-DD. The mock schedule does not change
    with the date, so repeated calls return the same flights. Each flight
    includes times, cabin class, stops, and connection minutes.
    """
    return json.dumps(lookup_flights(origin, destination, date), indent=2)


@tool
def get_fare_rules(fare_type: str) -> str:
    """Load refund rules for a fare type.

    Returns whether the fare is refundable and the tax amount.
    """
    return json.dumps(lookup_fare_rules(fare_type), indent=2)


REBOOKING_TOOLS = [get_booking, search_flights]
ALL_TOOLS = [get_booking, search_flights, get_fare_rules]

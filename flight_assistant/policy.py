"""Airline policy checks. This module is plain Python and never calls a model."""

from datetime import timedelta

from flight_assistant.data import REFERENCE_NOW

MAX_POLICY_ATTEMPTS = 3
CABIN_RANK = {
    "economy": 0,
    "premium_economy": 1,
    "business": 2,
    "first": 3,
}


def parse_dt(value: str):
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def expected_refund(booking: dict, fare_rules: dict) -> float:
    airline_cancelled = (
        booking.get("disruption") == "cancelled" and booking.get("cancelled_by") == "airline"
    )
    if airline_cancelled or fare_rules.get("refundable"):
        return float(booking["fare_amount"])
    return float(fare_rules["tax_amount"])


def expected_compensation(booking: dict) -> float:
    if booking.get("disruption") == "cancelled":
        return 400.0
    delay = float(booking.get("delay_hours") or 0)
    if delay > 6:
        return 400.0
    if delay >= 3:
        return 200.0
    return 0.0


def _amounts_match(proposed, expected) -> bool:
    try:
        return abs(float(proposed) - float(expected)) < 0.01
    except (TypeError, ValueError):
        return False


def evaluate_policy(
    intent: str,
    solution: dict | None,
    booking: dict | None,
    search_results: list | None,
    fare_rules: dict | None,
) -> tuple[bool, str]:
    solution = solution or {}
    if solution.get("type") != intent:
        proposed = solution.get("type") or "nothing"
        return False, f"A {intent} request needs a {intent} solution, but the proposal was {proposed}."

    if intent == "rebook":
        return _check_rebook(solution, booking or {}, search_results or [])
    if intent == "refund":
        return _check_refund(solution, booking or {}, fare_rules or {})
    if intent == "compensation":
        return _check_compensation(solution, booking or {})
    return False, f"No automated policy applies to intent {intent}."


def _check_rebook(solution: dict, booking: dict, flights: list) -> tuple[bool, str]:
    if not booking or not booking.get("cabin_class"):
        return False, "The booking was not loaded, so the cabin class cannot be checked."
    number = solution.get("flight_number")
    match = next((flight for flight in flights if flight.get("flight_number") == number), None)
    if match is None:
        return False, f"Flight {number} is not in the search results. Propose a listed flight."

    booked = str(booking.get("cabin_class", "")).lower()
    offered = str(match.get("cabin_class", "")).lower()
    if booked not in CABIN_RANK or offered not in CABIN_RANK:
        return False, f"Cabin class {offered or 'missing'} cannot be compared with {booked or 'missing'}."
    if CABIN_RANK[offered] > CABIN_RANK[booked]:
        return (
            False,
            f"{number} is {offered}, and the booking is {booked}. "
            "A rebooking must be the same cabin or lower.",
        )

    departure = parse_dt(match["departure"])
    latest = REFERENCE_NOW + timedelta(hours=48)
    if departure < REFERENCE_NOW or departure > latest:
        return (
            False,
            f"{number} departs at {match['departure']}, outside the 48-hour window "
            f"starting {REFERENCE_NOW.isoformat()}.",
        )

    if int(match.get("stops") or 0) > 0:
        minutes = match.get("min_connection_minutes")
        if minutes is None or int(minutes) < 60:
            return (
                False,
                f"{number} has a connection of {minutes} minutes. At least 60 minutes is required.",
            )

    return True, f"{number} meets the cabin, 48-hour, and connection rules."


def _check_refund(solution: dict, booking: dict, fare_rules: dict) -> tuple[bool, str]:
    if not booking or "fare_amount" not in booking:
        return False, "The booking was not loaded, so the refund cannot be checked."
    if not fare_rules:
        return False, "The fare rules were not loaded, so the refund cannot be checked."
    expected = expected_refund(booking, fare_rules)
    if not _amounts_match(solution.get("amount"), expected):
        airline_cancelled = (
            booking.get("disruption") == "cancelled" and booking.get("cancelled_by") == "airline"
        )
        if airline_cancelled:
            why = "the airline cancelled the flight, so the refund is the full fare"
        elif fare_rules.get("refundable"):
            why = "the fare is refundable, so the refund is the full fare"
        else:
            why = "this is a passenger choice on a non-refundable fare, so only taxes are refunded"
        return (
            False,
            f"Refund amount {solution.get('amount')} was rejected because {why} "
            f"({expected:.2f}).",
        )
    return True, f"Refund of {expected:.2f} matches the fare rules."


def _check_compensation(solution: dict, booking: dict) -> tuple[bool, str]:
    if not booking:
        return False, "The booking was not loaded, so compensation cannot be checked."
    expected = expected_compensation(booking)
    if not _amounts_match(solution.get("amount"), expected):
        return (
            False,
            "Compensation was rejected. Under 3 hours is 0, 3 to 6 hours is 200, "
            f"and over 6 hours or a cancellation is 400. The amount must be {expected:.2f}, "
            f"not {solution.get('amount')}.",
        )
    return True, f"Compensation of {expected:.2f} matches the delay rules."

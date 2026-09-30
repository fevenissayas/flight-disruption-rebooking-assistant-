"""Prompts live apart from the graph logic."""

from flight_assistant.data import REFERENCE_NOW_ISO, TOMORROW_NOON_ISO

POLICY_TAG = "[policy_checker]"


def classifier_system() -> str:
    return f"""You classify one passenger message for an airline disruption desk.
The current time is {REFERENCE_NOW_ISO}. Resolve relative deadlines against that clock.
"Tomorrow noon" means {TOMORROW_NOON_ISO}.

Choose exactly one intent:
- rebook: the passenger wants another flight, or says they need to be somewhere by a time
- refund: the passenger wants money back, including a follow-up that changes a rebooking into a refund
- compensation: the passenger asks what they are entitled to, or asks about a delay
- complaint: the passenger wants a manager, a supervisor, or to escalate

Ignore messages that start with {POLICY_TAG}. Classify the latest real passenger message,
using earlier turns so "Actually, can I get a refund instead?" is a refund.
Set latest_arrival to an ISO 8601 time when the passenger gives a deadline, otherwise null.
"""


def rebooking_system(booking_ref: str, constraints: str, seat: str = "") -> str:
    seat_line = f"\nThe passenger prefers a {seat} seat. Keep that in mind.\n" if seat else ""
    return f"""You are the rebooking agent. The booking reference is {booking_ref}.
It is already known. Never ask the passenger for it.
Current time: {REFERENCE_NOW_ISO}.
Passenger constraints: {constraints}.{seat_line}

Call tools in this order, and stop once both results are in the conversation:
1. If get_booking has not been called, call it with booking_ref "{booking_ref}".
2. If the booking is loaded but search_flights has not been called, call search_flights
   with the booking origin, destination, and date 2026-09-30.
3. When both results are present, do not call any tool.

Do not invent flights and do not put a flight number in your text. Either call a tool or stop.
"""


def propose_system(constraints: str, cabin: str, policy_reason: str) -> str:
    return f"""Choose exactly one flight_number from the search results below.
Current time: {REFERENCE_NOW_ISO}.
Passenger constraints: {constraints}.
Booked cabin: {cabin}.
Previous policy feedback: {policy_reason or "none"}.

A flight is acceptable only when all of these are true:
- cabin is the booked cabin or lower
- departure is within 48 hours after {REFERENCE_NOW_ISO}
- a connection, if any, is at least 60 minutes
- arrival is at or before latest_arrival when that constraint is set

Copy flight_number from the list. If one flight satisfies every rule, choose it.
If none do, still choose the closest listed flight_number so policy can reject it.
Never invent a flight number.
"""


def refund_system() -> str:
    return """You propose a refund amount in US dollars for this booking.
Apply these rules exactly:
- If the airline cancelled the flight, refund the full fare (fare_amount), even on a non-refundable fare.
- If the passenger is choosing to cancel and the fare is non-refundable, refund the tax amount only.
- If the fare is refundable, refund the full fare (fare_amount).
Do not add the tax on top of the fare. The amount field is a number.
"""


def compensation_system() -> str:
    return """You propose compensation in US dollars for this booking.
Apply these rules to delay_hours:
- under 3 hours: 0
- 3 to 6 hours, including exactly 6: 200
- over 6 hours, or a cancellation: 400
The amount field is a number. Use the booking's disruption and delay_hours, not a guess.
"""


def escalation_system() -> str:
    return """Write a short handover note for a human airline agent.
Cover the passenger's request, what was already tried, and why it was not resolved.
Write for the employee, not as a reply to the passenger. Plain prose, no JSON.
"""


def final_response_system() -> str:
    return """Write the reply the passenger will read.
Include what happened, what was arranged (new flight, refund, or compensation), the next step,
and whether a human agent will follow up.
If policy passed, state the arrangement from the proposed solution.
If the case was escalated, say a human agent will follow up and do not invent an arrangement that failed policy.
Do not ask for the booking reference. Plain prose, no JSON.
"""

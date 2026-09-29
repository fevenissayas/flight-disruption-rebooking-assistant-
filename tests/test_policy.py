"""Policy rules, independent of the model."""

from datetime import timedelta

from flight_assistant.data import BOOKINGS, REFERENCE_NOW
from flight_assistant.policy import evaluate_policy, parse_dt
from flight_assistant.tools import lookup_fare_rules, lookup_flights


def test_rebook_accepts_the_economy_flight_that_arrives_before_noon():
    booking = BOOKINGS["XK9L2P"]
    flights = lookup_flights("LHR", "DXB", "2026-09-30")
    passed, reason = evaluate_policy(
        "rebook",
        {"type": "rebook", "flight_number": "EK202"},
        booking,
        flights,
        None,
    )
    assert passed, reason


def test_rebook_rejects_business_short_connection_and_late_departure():
    booking = BOOKINGS["XK9L2P"]
    flights = lookup_flights("London", "Dubai", "2026-09-30")
    for number in ("BA880", "LH441", "QR510"):
        passed, reason = evaluate_policy(
            "rebook",
            {"type": "rebook", "flight_number": number},
            booking,
            flights,
            None,
        )
        assert not passed, number
        assert number in reason


def test_business_only_route_cannot_rebook_an_economy_passenger():
    booking = BOOKINGS["BZ3CLS"]
    flights = lookup_flights("CDG", "JFK", "2026-09-30")
    assert flights
    assert all(flight["cabin_class"] == "business" for flight in flights)
    for flight in flights:
        passed, _reason = evaluate_policy(
            "rebook",
            {"type": "rebook", "flight_number": flight["flight_number"]},
            booking,
            flights,
            None,
        )
        assert not passed


def test_refund_is_full_fare_when_the_airline_cancelled():
    booking = BOOKINGS["XK9L2P"]
    rules = lookup_fare_rules(booking["fare_type"])
    passed, reason = evaluate_policy(
        "refund",
        {"type": "refund", "amount": 850},
        booking,
        None,
        rules,
    )
    assert passed, reason
    rejected, _reason = evaluate_policy(
        "refund",
        {"type": "refund", "amount": 145},
        booking,
        None,
        rules,
    )
    assert not rejected


def test_refund_is_taxes_only_when_the_passenger_cancels_a_saver_fare():
    booking = dict(BOOKINGS["XK9L2P"])
    booking["disruption"] = "none"
    booking["cancelled_by"] = None
    rules = lookup_fare_rules("saver")
    passed, reason = evaluate_policy(
        "refund",
        {"type": "refund", "amount": 145},
        booking,
        None,
        rules,
    )
    assert passed, reason


def test_refundable_fare_returns_the_full_fare():
    booking = dict(BOOKINGS["CMP004"])
    booking["disruption"] = "none"
    booking["cancelled_by"] = None
    rules = lookup_fare_rules("flex")
    passed, reason = evaluate_policy(
        "refund",
        {"type": "refund", "amount": 910},
        booking,
        None,
        rules,
    )
    assert passed, reason


def test_compensation_bands():
    delayed = dict(BOOKINGS["DL6H01"])
    rules = None
    cases = [
        (2.5, 0),
        (3, 200),
        (6, 200),
        (6.5, 400),
    ]
    for hours, amount in cases:
        delayed["disruption"] = "delayed"
        delayed["delay_hours"] = hours
        passed, reason = evaluate_policy(
            "compensation",
            {"type": "compensation", "amount": amount},
            delayed,
            None,
            rules,
        )
        assert passed, (hours, reason)
    cancelled = dict(BOOKINGS["XK9L2P"])
    passed, reason = evaluate_policy(
        "compensation",
        {"type": "compensation", "amount": 400},
        cancelled,
        None,
        rules,
    )
    assert passed, reason


def test_departure_window_includes_the_48_hour_boundary():
    boundary = (REFERENCE_NOW + timedelta(hours=48)).isoformat()
    assert parse_dt(boundary) == REFERENCE_NOW + timedelta(hours=48)

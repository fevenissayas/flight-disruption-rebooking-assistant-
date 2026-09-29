"""Fixed mock inventory. Every run sees the same bookings and flights."""

from datetime import datetime, timezone

# The disruption desk clock. Policy windows and "tomorrow noon" use this instant.
REFERENCE_NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
REFERENCE_NOW_ISO = REFERENCE_NOW.isoformat()
TOMORROW_NOON_ISO = "2026-09-30T12:00:00+00:00"

PLACE_ALIASES = {
    "LHR": "LHR",
    "LONDON": "LHR",
    "LONDON HEATHROW": "LHR",
    "DXB": "DXB",
    "DUBAI": "DXB",
    "CDG": "CDG",
    "PARIS": "CDG",
    "PARIS CDG": "CDG",
    "JFK": "JFK",
    "NEW YORK": "JFK",
    "NEW YORK JFK": "JFK",
}

FARE_RULES = {
    "saver": {
        "fare_type": "saver",
        "refundable": False,
        "tax_amount": 145.0,
        "description": "Non-refundable fare. Taxes can be refunded when the passenger chooses to cancel.",
    },
    "flex": {
        "fare_type": "flex",
        "refundable": True,
        "tax_amount": 160.0,
        "description": "Fully refundable fare.",
    },
}

BOOKINGS = {
    "XK9L2P": {
        "booking_ref": "XK9L2P",
        "passenger_name": "Amira Hassan",
        "origin": "LHR",
        "destination": "DXB",
        "origin_city": "London",
        "destination_city": "Dubai",
        "cabin_class": "economy",
        "fare_type": "saver",
        "fare_amount": 850.0,
        "tax_amount": 145.0,
        "flight_number": "BA105",
        "scheduled_departure": "2026-09-30T08:00:00+00:00",
        "disruption": "cancelled",
        "cancelled_by": "airline",
        "delay_hours": None,
    },
    "DL6H01": {
        "booking_ref": "DL6H01",
        "passenger_name": "Noah Keller",
        "origin": "JFK",
        "destination": "LHR",
        "origin_city": "New York",
        "destination_city": "London",
        "cabin_class": "economy",
        "fare_type": "saver",
        "fare_amount": 640.0,
        "tax_amount": 145.0,
        "flight_number": "DL412",
        "scheduled_departure": "2026-09-29T18:00:00+00:00",
        "disruption": "delayed",
        "cancelled_by": None,
        "delay_hours": 6,
    },
    "BZ3CLS": {
        "booking_ref": "BZ3CLS",
        "passenger_name": "Sofia Martins",
        "origin": "CDG",
        "destination": "JFK",
        "origin_city": "Paris",
        "destination_city": "New York",
        "cabin_class": "economy",
        "fare_type": "saver",
        "fare_amount": 720.0,
        "tax_amount": 145.0,
        "flight_number": "AF009",
        "scheduled_departure": "2026-09-29T16:00:00+00:00",
        "disruption": "cancelled",
        "cancelled_by": "airline",
        "delay_hours": None,
    },
    "CMP004": {
        "booking_ref": "CMP004",
        "passenger_name": "Daniel Okonkwo",
        "origin": "LHR",
        "destination": "DXB",
        "origin_city": "London",
        "destination_city": "Dubai",
        "cabin_class": "economy",
        "fare_type": "flex",
        "fare_amount": 910.0,
        "tax_amount": 160.0,
        "flight_number": "BA107",
        "scheduled_departure": "2026-09-29T20:00:00+00:00",
        "disruption": "cancelled",
        "cancelled_by": "airline",
        "delay_hours": None,
    },
}

# LHR-DXB includes valid and policy-breaking options so a retry can be exercised.
# CDG-JFK is business class only, so an economy booking can never pass.
FLIGHTS = {
    ("LHR", "DXB"): [
        {
            "flight_number": "EK202",
            "origin": "LHR",
            "destination": "DXB",
            "departure": "2026-09-30T01:40:00+00:00",
            "arrival": "2026-09-30T11:25:00+00:00",
            "cabin_class": "economy",
            "stops": 0,
            "min_connection_minutes": None,
            "seats_available": 6,
        },
        {
            "flight_number": "BA880",
            "origin": "LHR",
            "destination": "DXB",
            "departure": "2026-09-30T08:00:00+00:00",
            "arrival": "2026-09-30T18:40:00+00:00",
            "cabin_class": "business",
            "stops": 0,
            "min_connection_minutes": None,
            "seats_available": 2,
        },
        {
            "flight_number": "LH441",
            "origin": "LHR",
            "destination": "DXB",
            "departure": "2026-09-30T06:00:00+00:00",
            "arrival": "2026-09-30T18:10:00+00:00",
            "cabin_class": "economy",
            "stops": 1,
            "min_connection_minutes": 40,
            "seats_available": 3,
        },
        {
            "flight_number": "QR510",
            "origin": "LHR",
            "destination": "DXB",
            "departure": "2026-10-02T09:00:00+00:00",
            "arrival": "2026-10-02T19:00:00+00:00",
            "cabin_class": "economy",
            "stops": 0,
            "min_connection_minutes": None,
            "seats_available": 5,
        },
    ],
    ("CDG", "JFK"): [
        {
            "flight_number": "AF011",
            "origin": "CDG",
            "destination": "JFK",
            "departure": "2026-09-30T08:15:00+00:00",
            "arrival": "2026-09-30T16:40:00+00:00",
            "cabin_class": "business",
            "stops": 0,
            "min_connection_minutes": None,
            "seats_available": 3,
        },
        {
            "flight_number": "DL404",
            "origin": "CDG",
            "destination": "JFK",
            "departure": "2026-09-30T11:00:00+00:00",
            "arrival": "2026-09-30T20:30:00+00:00",
            "cabin_class": "business",
            "stops": 1,
            "min_connection_minutes": 90,
            "seats_available": 2,
        },
    ],
}


def normalize_place(value: str) -> str:
    cleaned = " ".join(value.strip().upper().replace("-", " ").split())
    return PLACE_ALIASES.get(cleaned, cleaned)

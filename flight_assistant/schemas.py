"""Structured outputs extracted from the specialist agents."""

from typing import Literal

from pydantic import BaseModel, Field


class Classification(BaseModel):
    intent: Literal["rebook", "refund", "compensation", "complaint"] = Field(
        description="What the passenger wants on this turn."
    )
    latest_arrival: str | None = Field(
        default=None,
        description="ISO 8601 deadline if the passenger gave one, otherwise null.",
    )
    notes: str = Field(default="", description="Short note about the constraint, or empty.")


class FlightProposal(BaseModel):
    flight_number: str = Field(description="Flight number copied from search_flights.")
    summary: str = Field(description="One sentence describing the proposed flight.")


class RefundProposal(BaseModel):
    amount: float = Field(description="Refund in US dollars.")
    summary: str = Field(description="One sentence explaining the refund.")


class CompensationProposal(BaseModel):
    amount: float = Field(description="Compensation in US dollars.")
    summary: str = Field(description="One sentence explaining the compensation.")

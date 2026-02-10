import os
from typing import List, Optional

from pydantic import BaseModel, Field
from tavily import TavilyClient

# NOTE: Do NOT add: from __future__ import annotations
# NAT introspects types and can break on postponed annotations.


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin city/airport (e.g., 'DEL' or 'New Delhi')")
    destination: str = Field(description="Destination city/airport (e.g., 'SIN' or 'Singapore')")
    depart_date: str = Field(description="Departure date in YYYY-MM-DD")
    return_date: Optional[str] = Field(default=None, description="Return date in YYYY-MM-DD (optional)")
    adults: int = Field(default=1, ge=1, le=9, description="Number of adult travelers")
    cabin: str = Field(default="economy", description="Cabin: economy/premium economy/business/first")
    max_results: int = Field(default=5, ge=1, le=10, description="How many web results to return")


class FlightOption(BaseModel):
    title: str
    url: str
    snippet: str


class FlightSearchOutput(BaseModel):
    query: str
    options: List[FlightOption]
    note: str


def _build_query(i: FlightSearchInput) -> str:
    """
    Web-search oriented query. Works across airlines/aggregators.
    """
    if i.return_date:
        trip = f"round trip {i.depart_date} to {i.return_date}"
    else:
        trip = f"one way {i.depart_date}"

    return (
        f"best flights {i.origin} to {i.destination} {trip} "
        f"{i.adults} adults {i.cabin} price"
    )


async def flight_search_tool(input_data: FlightSearchInput) -> FlightSearchOutput:
    """
    Uses Tavily to find flight search pages and relevant results.
    Does NOT book flights; returns links + snippets.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in environment (.env).")

    tavily = TavilyClient(api_key=api_key)

    query = _build_query(input_data)

    # setting max_results and include_* params. :contentReference[oaicite:3]{index=3}
    resp = tavily.search(
        query=query,
        max_results=input_data.max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = resp.get("results", []) or []

    options: List[FlightOption] = []
    for r in results[: input_data.max_results]:
        options.append(
            FlightOption(
                title=str(r.get("title", "")).strip() or "Flight result",
                url=str(r.get("url", "")).strip(),
                snippet=str(r.get("content", "")).strip()[:300],
            )
        )

    note = (
        "These are web search results for flight shopping (not bookings). "
        "Open links to see live prices/availability; they change frequently."
    )

    return FlightSearchOutput(query=query, options=options, note=note)

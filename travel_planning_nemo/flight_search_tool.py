import os
from typing import List, Optional

from pydantic import BaseModel, Field
from tavily import TavilyClient


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin city/airport code (e.g., DEL)")
    destination: str = Field(description="Destination city/airport/city (e.g., SIN or Singapore)")
    depart_date: str = Field(description="Departure date in YYYY-MM-DD")
    return_date: Optional[str] = Field(default=None, description="Return date in YYYY-MM-DD (optional)")
    adults: int = Field(default=1, ge=1, le=9, description="Number of adult travelers")
    cabin: str = Field(default="economy", description="Cabin: economy/premium economy/business/first")
    max_results: int = Field(default=5, ge=1, le=10, description="How many web results to return")

    # Optional itinerary context (for integration)
    day_start_time: str = Field(default="09:00", description="Typical itinerary day start time")
    pace: str = Field(default="moderate", description="Itinerary pace (slow/moderate/fast)")
    constraints: str = Field(default="", description="Trip constraints that may affect flight timing")
    special_requests: str = Field(default="", description="Special requests that may affect flight timing")


class FlightOption(BaseModel):
    title: str
    url: str
    snippet: str


class FlightTimingAdvice(BaseModel):
    recommended_arrival_window: str
    recommended_departure_window: str
    reasoning: str


class FlightSearchOutput(BaseModel):
    query: str
    options: List[FlightOption]
    timing_advice: FlightTimingAdvice
    flight_context_markdown: str
    note: str


def _build_query(i: FlightSearchInput) -> str:
    if i.return_date:
        trip = f"round trip {i.depart_date} to {i.return_date}"
    else:
        trip = f"one way {i.depart_date}"
    return f"flights {i.origin} to {i.destination} {trip} {i.adults} adults {i.cabin} prices"


def _timing_advice(i: FlightSearchInput) -> FlightTimingAdvice:
    """
    Heuristic timing suggestions (no hallucinated exact schedules).
    The ReAct agent can use this to plan Day 1/last day realistically.
    """
    # Default windows
    arrival = "Arrive afternoon (12:00–18:00) for an easy check-in + evening activity"
    depart = "Depart late morning/afternoon (10:00–16:00) to avoid very early rush"

    reasons = []

    pace = (i.pace or "").lower()
    if "slow" in pace or "relax" in pace:
        arrival = "Arrive afternoon/evening (14:00–20:00) for a relaxed first day"
        reasons.append("Relaxed pace → later arrival is fine.")
    elif "fast" in pace or "packed" in pace:
        arrival = "Arrive morning (07:00–11:00) to maximize Day 1"
        depart = "Depart evening (17:00–22:00) to maximize final day"
        reasons.append("Fast/packed pace → maximize usable daylight hours.")

    if "avoid" in (i.constraints or "").lower() and "early" in (i.constraints or "").lower():
        depart = "Depart afternoon/evening (13:00–21:00) to avoid early departures"
        reasons.append("Constraint mentions avoiding early times.")

    if "avoid long commutes" in (i.constraints or "").lower():
        reasons.append("Avoid long commutes → prefer arrival that avoids peak transit if possible.")

    reasons.append(f"Day start time is {i.day_start_time} → flights that don’t force a 04:00 wake-up are preferable.")
    reasoning = " ".join(reasons).strip() or "General travel comfort + check-in/check-out practicality."

    return FlightTimingAdvice(
        recommended_arrival_window=arrival,
        recommended_departure_window=depart,
        reasoning=reasoning,
    )


def flight_context_md(out: FlightSearchOutput) -> str:
    """
    A compact Markdown block that the itinerary tool can embed directly.
    """
    lines = []
    lines.append("**Flight shopping summary (web results):**")
    lines.append(f"- Query: `{out.query}`")
    lines.append(f"- Arrival window: {out.timing_advice.recommended_arrival_window}")
    lines.append(f"- Departure window: {out.timing_advice.recommended_departure_window}")
    lines.append(f"- Why: {out.timing_advice.reasoning}")
    lines.append("")
    lines.append("**Where to check live prices/availability:**")
    for opt in out.options[: min(5, len(out.options))]:
        if opt.url:
            lines.append(f"- [{opt.title}]({opt.url})")
        else:
            lines.append(f"- {opt.title}")
    lines.append("")
    lines.append("_Note: Prices/availability change frequently; open links for live details._")
    return "\n".join(lines)


async def flight_search_tool(input_data: FlightSearchInput) -> FlightSearchOutput:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY. Add it to .env (not only .env.template).")

    query = _build_query(input_data)

    tavily = TavilyClient(api_key=api_key)
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
                title=str(r.get("title") or "Flight result").strip(),
                url=str(r.get("url") or "").strip(),
                snippet=str(r.get("content") or "").strip()[:300],
            )
        )

    advice = _timing_advice(input_data)

    out = FlightSearchOutput(
        query=query,
        options=options,
        timing_advice=advice,
        flight_context_markdown="",  # filled next
        note="Web-based flight discovery only; not a booking system. Use links for live prices and schedules.",
    )
    out.flight_context_markdown = flight_context_md(out)
    return out

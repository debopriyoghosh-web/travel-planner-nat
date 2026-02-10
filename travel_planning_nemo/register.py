#from __future__ import annotations

from pydantic import BaseModel, Field


from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .travel_planning_nemo import (
    NvidiaChatConfig,
    build_user_prompt,
    call_nvidia_chat_completion,
)

from .flight_search_tool import FlightSearchInput, FlightSearchOutput, flight_search_tool


class TravelItineraryInput(BaseModel):
    destination: str = Field(description="Primary destination city/region/country")
    start_date: str = Field(description="Trip start date (e.g., 2026-03-10)")
    end_date: str = Field(description="Trip end date (e.g., 2026-03-14)")

    travelers: str = Field(default="1 adult", description="Who is traveling (e.g., '2 adults', 'family of 4')")
    budget: str = Field(default="mid-range", description="Budget level (e.g., 'budget', 'mid-range', 'luxury')")
    travel_style: str = Field(default="balanced", description="Style (e.g., 'relaxed', 'packed', 'balanced')")
    interests: str = Field(default="sightseeing, food", description="Comma-separated interests")

    day_start_time: str = Field(default="09:00", description="Typical day start time")
    pace: str = Field(default="moderate", description="Pace (slow/moderate/fast)")
    mobility: str = Field(default="no constraints", description="Mobility constraints if any")
    food_prefs: str = Field(default="no constraints", description="Food preferences/allergies")

    constraints: str = Field(default="", description="Hard constraints (must-dos, avoid, etc.)")
    special_requests: str = Field(default="", description="Any special requests")


class TravelItineraryOutput(BaseModel):
    itinerary_markdown: str = Field(description="Full itinerary in Markdown")


class TravelItineraryConfig(FunctionBaseConfig, name="travel_itinerary"):
    """NAT registration name: travel_itinerary"""
    pass


@register_function(config_type=TravelItineraryConfig)
async def travel_itinerary(config: TravelItineraryConfig, builder):
    """
    ReAct-friendly tool:
    - Accepts structured trip inputs
    - Returns final itinerary as Markdown only
    """
    async def _inner(input_data: TravelItineraryInput) -> TravelItineraryOutput:
        cfg = NvidiaChatConfig.from_env()

        system_prompt = (
            "You are a travel planner.\n"
            "Return ONLY the itinerary in Markdown.\n"
            "Follow the provided template headings exactly.\n"
            "Do not include analysis, meta commentary, or tool/action text.\n"
        )

        user_prompt = build_user_prompt(input_data.model_dump())

        itinerary_md = await call_nvidia_chat_completion(
            cfg=cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return TravelItineraryOutput(itinerary_markdown=itinerary_md)

    yield FunctionInfo.from_fn(
        _inner,
        description=(
            "Generate the FINAL trip itinerary in Markdown using the predefined template. "
            "Use this tool whenever the user requests a travel plan or itinerary."
        ),
    )

class FlightSearchConfig(FunctionBaseConfig, name="flight_search"):
    """NAT registration name: flight_search"""
    pass


@register_function(config_type=FlightSearchConfig)
async def flight_search(config: FlightSearchConfig, builder):
    async def _inner(input_data: FlightSearchInput) -> FlightSearchOutput:
        return await flight_search_tool(input_data)

    yield FunctionInfo.from_fn(
        _inner,
        description=(
            "Find flight shopping links and summaries using Tavily web search. "
            "Use this tool when the user asks about flights, prices, airlines, or routes."
        ),
    )
